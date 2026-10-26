# apps/ai_engine/stt_client.py

import logging
import re

from django.conf import settings
from groq import Groq

from apps.interviews.services.speech_cleanup import collapse_repeats

logger = logging.getLogger(__name__)

# Whisper "hallucinates" these when it is fed silence or room noise.
_SILENCE_HALLUCINATIONS = re.compile(
    r"^\W*(thank you(?: very much)?|thanks(?: for watching)?|thanks for watching!?|you|bye|okay|so|uh|um|hmm+)\W*$",
    re.IGNORECASE,
)
_NO_SPEECH_PROB_LIMIT = 0.6      # segment is probably not speech above this...
_AVG_LOGPROB_LIMIT = -1.0        # ...and the model is unsure about it
_COMPRESSION_RATIO_LIMIT = 2.4   # Whisper's own "this is repetitive garbage" threshold


class STTClient:
    """
    Wrapper around Groq's hosted Whisper large-v3 for speech-to-text.
    Used by the interview consumer to transcribe candidate audio chunks
    before they're passed to answer_analyzer.py.

    Reuses the same GROQ_API_KEY as LLMClient — Groq hosts both chat
    completions and Whisper transcription under one API key/client.
    The Groq client is created on first use (importing this module never
    fails when the key is missing).
    """

    MODEL = "whisper-large-v3"

    def __init__(self):
        self._client = None

    @property
    def client(self) -> Groq:
        if self._client is None:
            api_key = getattr(settings, "GROQ_API_KEY", None)
            if not api_key:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Add it to your .env and settings.py."
                )
            self._client = Groq(api_key=api_key, timeout=25.0, max_retries=1)
        return self._client

    def transcribe(
        self,
        audio_file_path: str,
        language: str | None = None,
        prompt: str = "",
    ) -> str:
        """
        Transcribes an audio file on disk and returns plain text ("" when
        the audio holds no real speech).

        language: ISO 639-1 code. Defaults to settings.STT_LANGUAGE ("en"
            unless changed). Set STT_LANGUAGE="" in settings/.env to let
            Whisper auto-detect (better for Hindi / Hinglish candidates).
        prompt: optional Whisper priming text (names, jargon).

        Segments that Whisper itself marks as "probably not speech"
        (high no_speech_prob + low confidence) are dropped, and a transcript
        that is only a known silence hallucination ("Thank you.") is
        returned as "" — so room noise is never scored as an answer.

        Raises RuntimeError on failure so the caller can decide how to
        handle a failed transcription.
        """
        if language is None:
            language = getattr(settings, "STT_LANGUAGE", "en")

        kwargs = dict(
            model=self.MODEL,
            prompt=prompt or None,
            response_format="verbose_json",
            temperature=0.0,
        )
        if language:
            kwargs["language"] = language

        try:
            with open(audio_file_path, "rb") as audio_file:
                audio_bytes = audio_file.read()

            text, looped = collapse_repeats(self._clean(
                self.client.audio.transcriptions.create(file=(audio_file_path, audio_bytes), **kwargs)
            ))
            if looped:
                # Whisper got stuck repeating itself (quiet / noisy audio). One more
                # try with a little randomness usually breaks the loop.
                logger.info("STT loop detected (%r) — transcribing once more.", text[:60])
                kwargs["temperature"] = 0.3
                retry, still_looped = collapse_repeats(self._clean(
                    self.client.audio.transcriptions.create(file=(audio_file_path, audio_bytes), **kwargs)
                ))
                text = "" if still_looped and len(retry.split()) <= 4 else (retry or text)
            return text

        except FileNotFoundError as exc:
            logger.error("Audio file not found: %s", audio_file_path)
            raise RuntimeError(f"Audio file not found: {audio_file_path}") from exc
        except Exception as exc:
            logger.error("Groq STT call failed: %s", exc, exc_info=True)
            raise RuntimeError(f"STT call failed: {exc}") from exc

    @staticmethod
    def _clean(response) -> str:
        if isinstance(response, str):
            text = response
            segments = None
        else:
            text = getattr(response, "text", "") or ""
            segments = getattr(response, "segments", None)
            if segments is None and isinstance(response, dict):
                segments = response.get("segments")

        if segments:
            kept = []
            for seg in segments:
                get = seg.get if isinstance(seg, dict) else (lambda k, d=None, _s=seg: getattr(_s, k, d))
                no_speech = get("no_speech_prob", 0.0) or 0.0
                logprob = get("avg_logprob", 0.0) or 0.0
                compression = get("compression_ratio", 0.0) or 0.0
                if no_speech > _NO_SPEECH_PROB_LIMIT and logprob < _AVG_LOGPROB_LIMIT:
                    continue
                if compression > _COMPRESSION_RATIO_LIMIT:
                    continue      # the same words looping over and over = classic Whisper hallucination
                kept.append((get("text", "") or "").strip())
            text = " ".join(t for t in kept if t)

        text = text.strip()
        if _SILENCE_HALLUCINATIONS.match(text) and len(text.split()) <= 3 and segments:
            # Only trust this rule when Whisper gave segment confidences:
            # a genuine one-word "okay" would otherwise be thrown away.
            confident = all(
                (seg.get("no_speech_prob", 0) if isinstance(seg, dict) else getattr(seg, "no_speech_prob", 0) or 0) < 0.3
                for seg in segments
            )
            if not confident:
                return ""
        return text


stt_client = STTClient()

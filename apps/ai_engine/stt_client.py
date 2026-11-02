# apps/ai_engine/stt_client.py

import logging
import re

from django.conf import settings
from groq import Groq

from apps.interviews.services.speech_cleanup import collapse_repeats
from apps.interviews.services.stt_vocab import prompt_echo

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

    # The full model hears accented speech and technical words noticeably better than turbo
    # ("Django" vs "Jango", "B.Tech CSE" vs "BTEC CIC") and low scores were traced to
    # transcription errors. Turbo is kept as the fallback if the full model fails or is
    # rate-limited. Set STT_MODEL=whisper-large-v3-turbo in .env to prefer speed.
    MODEL = "whisper-large-v3"
    MODEL_FALLBACK = "whisper-large-v3-turbo"
    # Hindi / auto-detected speech uses the full model too.
    MODEL_MULTILINGUAL = "whisper-large-v3"

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
        """Plain-text transcription (see transcribe_detailed)."""
        return self.transcribe_detailed(audio_file_path, language, prompt)[0]

    def transcribe_detailed(
        self,
        audio_file_path: str,
        language: str | None = None,
        prompt: str = "",
    ) -> tuple[str, str]:
        """
        Returns (text, spoken_language). spoken_language is Whisper's own
        detection as a lower-case name ("english", "hindi", ...) or "" when
        unknown — it is only meaningful when language="" (auto-detect).

        language: ISO 639-1 code ("en", "hi"). None = settings.STT_LANGUAGE;
            "" = let Whisper detect the language (English + Hindi interviews).
            STT_LANGUAGE="" in settings/.env makes auto-detect the default.
        The text is "" when the audio holds no real speech.
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
            model=getattr(settings, "STT_MODEL", None)
            or (self.MODEL if language == "en" else self.MODEL_MULTILINGUAL),
            prompt=prompt or None,
            response_format="verbose_json",
            temperature=0.0,
        )
        if language:
            kwargs["language"] = language

        try:
            with open(audio_file_path, "rb") as audio_file:
                audio_bytes = audio_file.read()

            raw = self._create(audio_file_path, audio_bytes, kwargs)
            spoken = self._spoken_language(raw)
            text, looped = collapse_repeats(self._clean(raw))
            if text and prompt and prompt_echo(text, prompt):
                logger.info("STT returned the priming hint itself (%r) - treating it as silence.", text[:60])
                text = ""
            if not text:
                self._log_dropped(raw, len(audio_bytes))
            if looped:
                # Whisper got stuck repeating itself (quiet / noisy audio). One more
                # try with a little randomness usually breaks the loop.
                logger.info("STT loop detected (%r) — transcribing once more.", text[:60])
                kwargs["temperature"] = 0.3
                retry, still_looped = collapse_repeats(self._clean(
                    self._create(audio_file_path, audio_bytes, kwargs)
                ))
                text = "" if still_looped and len(retry.split()) <= 4 else (retry or text)
            return text, spoken

        except FileNotFoundError as exc:
            logger.error("Audio file not found: %s", audio_file_path)
            raise RuntimeError(f"Audio file not found: {audio_file_path}") from exc
        except Exception as exc:
            logger.error("Groq STT call failed: %s", exc, exc_info=True)
            raise RuntimeError(f"STT call failed: {exc}") from exc

    def _create(self, path: str, audio_bytes: bytes, kwargs: dict):
        """One Whisper call; if the chosen model fails (quota, outage) the other one is tried once."""
        try:
            return self.client.audio.transcriptions.create(file=(path, audio_bytes), **kwargs)
        except Exception as exc:
            other = self.MODEL_FALLBACK if kwargs["model"] != self.MODEL_FALLBACK else self.MODEL
            logger.warning("STT model %s failed (%s) - trying %s.", kwargs["model"], str(exc)[:120], other)
            return self.client.audio.transcriptions.create(file=(path, audio_bytes), **dict(kwargs, model=other))

    @staticmethod
    def _spoken_language(response) -> str:
        lang = response.get("language") if isinstance(response, dict) else getattr(response, "language", "")
        return str(lang or "").strip().lower()

    @staticmethod
    def _log_dropped(response, n_bytes: int) -> None:
        """An empty result: log WHY (audio size, Whisper's raw text and per-segment confidence),
        so 'it did not hear me' can be told apart from 'the audio was silent'."""
        raw_text = response if isinstance(response, str) else (getattr(response, "text", "") or "")
        segs = [] if isinstance(response, str) else (getattr(response, "segments", None) or [])
        info = []
        for seg in segs[:6]:
            get = seg.get if isinstance(seg, dict) else (lambda k, d=None, _s=seg: getattr(_s, k, d))
            info.append(f"no_speech={get('no_speech_prob', 0):.2f} logprob={get('avg_logprob', 0):.2f} "
                        f"comp={get('compression_ratio', 0):.2f} text={str(get('text', ''))[:40]!r}")
        logger.info("STT returned nothing usable (audio=%d bytes, whisper_text=%r, segments=%s)",
                    n_bytes, raw_text[:80], info)

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

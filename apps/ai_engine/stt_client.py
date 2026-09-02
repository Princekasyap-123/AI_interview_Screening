# apps/ai_engine/stt_client.py

import logging

from django.conf import settings
from groq import Groq

logger = logging.getLogger(__name__)


class STTClient:
    """
    Wrapper around Groq's hosted Whisper large-v3 for speech-to-text.
    Used by the interview consumer to transcribe candidate audio chunks
    before they're passed to answer_analyzer.py.

    Reuses the same GROQ_API_KEY as LLMClient — Groq hosts both chat
    completions and Whisper transcription under one API key/client.
    """

    MODEL = "whisper-large-v3"

    def __init__(self):
        api_key = getattr(settings, "GROQ_API_KEY", None)
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env and settings.py."
            )
        self.client = Groq(api_key=api_key)

    def transcribe(
        self,
        audio_file_path: str,
        language: str = "en",
        prompt: str = "",
    ) -> str:
        """
        Transcribes an audio file on disk and returns plain text.

        audio_file_path: path to a saved audio chunk (wav/mp3/m4a/webm —
            Groq's Whisper endpoint accepts standard formats; the
            frontend's stt_recorder.js determines what format actually
            gets saved, make sure it matches something Whisper accepts).
        language: ISO 639-1 code. Defaults to "en" — change/parametrize
            if you need multi-language interviews later.
        prompt: optional Whisper "priming" text — e.g. candidate's name
            or domain jargon, can slightly improve accuracy on proper
            nouns/technical terms. Optional, safe to leave blank.

        Raises RuntimeError on failure so the caller (interview consumer
        or wherever audio arrives) can decide how to handle a failed
        transcription — e.g. store an empty transcript_text and let
        answer_analyzer.py's null-handling take over, rather than
        crashing the interview turn.
        """
        try:
            with open(audio_file_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    file=(audio_file_path, audio_file.read()),
                    model=self.MODEL,
                    language=language,
                    prompt=prompt or None,
                    response_format="text",
                )
            # response_format="text" returns a plain string directly
            # from the SDK, not an object with .text — confirm against
            # your installed groq SDK version if this errors; some
            # versions return an object even in text mode.
            transcript = response if isinstance(response, str) else getattr(response, "text", "")
            return transcript.strip()

        except FileNotFoundError as exc:
            logger.error("Audio file not found: %s", audio_file_path)
            raise RuntimeError(f"Audio file not found: {audio_file_path}") from exc
        except Exception as exc:
            logger.error("Groq STT call failed: %s", exc, exc_info=True)
            raise RuntimeError(f"STT call failed: {exc}") from exc


stt_client = STTClient()
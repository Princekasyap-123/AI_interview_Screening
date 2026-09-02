# apps/ai_engine/tts_client.py

"""
TTS is NOT a backend service in the current (no-GPU) phase — it runs
entirely client-side via the browser's SpeechSynthesis API
(frontend/static/js/tts_player.js). This file exists as the designated
swap point for when GPU-based TTS (XTTS-v2 + Wav2Lip/SadTalker, per
apps/avatar/services/) comes online, so the rest of the codebase
(interview_state.py, consumers.py) never has to change — they'd just
start receiving audio_url instead of relying on the frontend to speak
question text directly.

Until then, this module only provides a thin pass-through: the
consumer/FSM still call something for symmetry with stt_client, but the
"synthesis" is just handing back the text for the frontend to speak
itself.
"""

import logging

logger = logging.getLogger(__name__)


class TTSClient:
    """
    Placeholder implementation for the no-GPU phase. The interview
    consumer sends question text to the frontend as-is (via the
    "turn" WebSocket message's "text" field) and tts_player.js calls
    window.speechSynthesis.speak() directly — this class isn't even
    invoked in that path today.

    It exists so any future code that wants to call "synthesize speech"
    through a consistent interface can do so without knowing whether
    the backend is doing real TTS yet. Swap synthesize()'s internals
    for a real XTTS-v2 call once GPU is available; callers don't change.
    """

    def synthesize(self, text: str, voice: str = "default") -> dict:
        """
        Returns a dict describing how the frontend should produce audio
        for this text. In the current phase, this always tells the
        frontend to use its own browser TTS rather than fetching
        pre-rendered audio.

        Future GPU-phase return shape (not implemented yet):
            {"mode": "audio_url", "url": "/media/tts/xyz.mp3", "duration_ms": 3200}

        Current phase return shape:
            {"mode": "browser_tts", "text": text}
        """
        logger.debug(
            "TTSClient.synthesize called in no-GPU phase — deferring to "
            "browser-native speech synthesis for text: %.50s...", text
        )
        return {"mode": "browser_tts", "text": text}


tts_client = TTSClient()
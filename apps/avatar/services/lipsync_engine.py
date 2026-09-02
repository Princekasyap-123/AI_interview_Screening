# apps/avatar/services/lipsync_engine.py

"""
Lip-sync generation for the AI interviewer avatar — STUBBED for the
current no-GPU phase. Real implementation (Wav2Lip or SadTalker) is
deferred until GPU infrastructure is available, since both require a
GPU for anything close to real-time generation (CPU inference runs
several seconds-per-second of output video, unusable for a live call).

Current phase: apps/ai_engine/tts_client.py returns {"mode": "browser_tts"}
and the frontend renders a static logo panel (per the confirmed
interview_room.html design) — this module is never actually called yet.
Nothing in interview_state.py or consumers.py invokes it.

This file exists now so the swap-in point is defined ahead of time:
when GPU is available, generate_lipsync_clip() gets a real
Wav2Lip/SadTalker implementation, and only avatar/services/tts_engine.py
+ this file change — interview_state.py, consumers.py, and the frontend
message contract stay the same (frontend switches from "browser_tts"
mode to "audio_url" mode, per the note already left in tts_client.py).
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class LipsyncNotAvailableError(Exception):
    """
    Raised by generate_lipsync_clip() in the current no-GPU phase.
    Any code path that ends up calling this function today has a bug —
    it means something upstream is trying to use the avatar pipeline
    before it exists, rather than falling back to browser TTS.
    """


@dataclass
class LipsyncClipResult:
    """
    Shape the real implementation will eventually return. Defined now so
    downstream code (once written) can be built against a stable
    interface rather than guessing the shape later.
    """
    video_url: str
    duration_ms: int
    source_engine: str  # "wav2lip" | "sadtalker"


class LipsyncEngine:
    """
    Future home of Wav2Lip/SadTalker integration. Both take a static
    face image/video + an audio file and produce a lip-synced video clip:

      - Wav2Lip: simpler, faster per-frame, lower visual quality (lips
        only, minimal head movement). Lighter GPU requirement.
      - SadTalker: more natural head movement + lip-sync from a single
        photo, noticeably better visual quality, heavier GPU requirement.

    Planned real implementation, once GPU infra exists:
      1. Take avatar/static_avatar/ base image + an audio file (produced
         by tts_engine.py via XTTS-v2, NOT browser TTS at that point).
      2. Run inference (Wav2Lip or SadTalker, whichever is chosen based
         on the latency/quality tradeoff you land on).
      3. Save the resulting clip to MEDIA_ROOT, return a URL the
         frontend's <video> element can play, timed against when TTS
         audio starts.
      4. Likely needs a queue (Celery + GPU worker, NOT the same worker
         pool as CPU tasks like final rating generation) since inference
         is slow and blocking relative to a live interview's turn-taking
         pace — a synchronous call here would stall the whole interview
         loop.
    """

    def __init__(self):
        # Real implementation will load model weights here once, at
        # worker startup — Wav2Lip/SadTalker checkpoints are large and
        # shouldn't be reloaded per-request.
        pass

    def generate_lipsync_clip(self, audio_file_path: str, base_avatar_path: str) -> LipsyncClipResult:
        """
        Not implemented — current phase has no GPU to run this on.
        Calling this raises immediately rather than hanging or silently
        no-op'ing, so a caller finds out at development time, not in a
        live interview.
        """
        raise LipsyncNotAvailableError(
            "Lip-sync generation is not available in the current no-GPU "
            "phase. The interview flow should use browser-native TTS "
            "(apps.ai_engine.tts_client) and the static avatar panel "
            "instead of calling this method."
        )


lipsync_engine = LipsyncEngine()
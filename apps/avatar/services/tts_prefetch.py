# apps/avatar/services/tts_prefetch.py

"""
Pre-synthesizes the interviewer's lines (Kokoro) in a background thread
as soon as the questions exist, so each question starts playing at once
instead of waiting ~2-5 s for synthesis on the CPU.

The browser (tts_player.js -> TTSPlayer._chunks) requests audio per
sentence-sized chunk, and tts_engine caches by exact chunk text. So this
module splits text EXACTLY like TTSPlayer._chunks — keep the two in sync.
"""

import logging
import re
import threading

from apps.avatar.services.tts_engine import tts_engine

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"""(?<=[.!?…])\s+(?=[A-Z0-9"'(])""")


def split_chunks(text: str) -> list[str]:
    """Python port of TTSPlayer._chunks() in tts_player.js."""
    sentences = _SENTENCE_SPLIT.split(re.sub(r"\s+", " ", str(text)).strip())
    out: list[str] = []
    for s in sentences:
        while len(s) > 280:                       # very long sentence: cut at a comma/space
            cut = s.rfind(", ", 0, 280 + 2)
            if cut < 80:
                cut = s.rfind(" ", 0, 280 + 1)
            if cut < 80:
                cut = 280
            out.append(s[: cut + 1].strip())
            s = s[cut + 1:].strip()
        if not s:
            continue
        if out and len(out[-1]) < 40:
            out[-1] += " " + s
        else:
            out.append(s)
    return out or [str(text)]


def prefetch_in_background(texts: list[str]) -> None:
    """Synthesizes (and caches) every chunk of every text, in order, in one
    daemon thread. Cached chunks cost nothing; errors are only logged."""
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return

    def _run():
        done = 0
        for text in texts:
            for chunk in split_chunks(text):
                try:
                    tts_engine.synthesize(chunk)
                    done += 1
                except Exception as exc:                 # never break the interview
                    logger.warning("TTS prefetch failed for %r: %s", chunk[:60], exc)
        logger.info("TTS prefetch finished: %d chunk(s) ready", done)

    threading.Thread(target=_run, name="tts-prefetch", daemon=True).start()

# apps/interviews/services/echo_guard.py

"""
Detects "the microphone heard the interviewer's own voice".

On a phone the loudspeaker sits a few centimetres from the microphone and the
browser's echo cancellation is weak, so the AI's own question can come back as
the candidate's "answer". Speech detection (VAD) cannot catch that — the echo
IS speech — but the transcript gives it away: it is a near word-for-word copy
of something the interviewer just said.

is_echo() compares a transcript with the last few things the interviewer
spoke. Cheap (pure Python, microseconds), no LLM, no extra dependency.
"""

import re
from difflib import SequenceMatcher

_WORD = re.compile(r"[a-z0-9']+")

MIN_ECHO_WORDS = 4        # shorter transcripts are never called an echo ("okay", "yes sir")
MIN_COVERAGE = 0.7        # share of the transcript that is one unbroken run of the interviewer's words


def _words(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def is_echo(transcript: str, spoken_texts) -> bool:
    t_words = _words(transcript)
    if len(t_words) < MIN_ECHO_WORDS:
        return False
    for spoken in spoken_texts or ():
        s_words = _words(spoken)
        if len(s_words) < MIN_ECHO_WORDS:
            continue
        match = SequenceMatcher(None, t_words, s_words, autojunk=False).find_longest_match(
            0, len(t_words), 0, len(s_words)
        )
        if match.size >= MIN_ECHO_WORDS and match.size / len(t_words) >= MIN_COVERAGE:
            return True
    return False

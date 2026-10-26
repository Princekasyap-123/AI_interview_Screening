# apps/interviews/services/speech_cleanup.py

"""
Two small, dependency-free helpers for what comes out of speech-to-text.

collapse_repeats(): Whisper sometimes gets stuck on quiet / noisy audio and
    writes the same phrase over and over ("I am you actually I am you actually
    I am you actually"). The repeats are removed, and the caller is told when
    it was a real loop so it can transcribe the audio once more.

looks_unfinished(): did the candidate stop in the MIDDLE of a sentence
    ("I have worked with Django and")? Then the interviewer should keep
    listening instead of answering half a thought — like a person would.
"""

import re

_TOKEN = re.compile(r"\S+")
_STRIP = re.compile(r"[^\w']+", re.UNICODE)

MAX_BLOCK_WORDS = 10      # longest phrase that is checked for repeating
MIN_BLOCK_WORDS = 2       # single repeated words ("very very good", "no no") are left alone


def _norm(token: str) -> str:
    return _STRIP.sub("", token.lower())


def collapse_repeats(text: str) -> tuple[str, bool]:
    """Returns (text with repeated phrases reduced to one copy, looped).
    looped is True when some phrase appeared 3+ times in a row (a Whisper loop);
    a phrase said exactly twice is just collapsed."""
    tokens = _TOKEN.findall(text or "")
    if len(tokens) < MIN_BLOCK_WORDS * 2:
        return (text or "").strip(), False

    looped = False
    changed = True
    while changed:
        changed = False
        for n in range(min(MAX_BLOCK_WORDS, len(tokens) // 2), MIN_BLOCK_WORDS - 1, -1):
            i = 0
            while i + 2 * n <= len(tokens):
                block = [_norm(t) for t in tokens[i:i + n]]
                if any(block) and block == [_norm(t) for t in tokens[i + n:i + 2 * n]]:
                    copies = 2
                    while i + (copies + 1) * n <= len(tokens) and \
                            block == [_norm(t) for t in tokens[i + copies * n:i + (copies + 1) * n]]:
                        copies += 1
                    if copies >= 3:
                        looped = True
                    del tokens[i + n:i + copies * n]
                    changed = True
                else:
                    i += 1
    return " ".join(tokens).strip(), looped


# Words a sentence practically never ends on, with or without a full stop.
_NEVER_ENDS = frozenset({
    "and", "but", "because", "which", "or", "with", "to", "of", "the", "a", "an",
    "my", "our", "your", "their", "um", "uh", "umm", "uhh", "hmm", "as", "if",
    "when", "while", "from", "by", "about", "also", "then", "that's", "i'm", "i've",
})
# ...plus words that CAN end a sentence ("yes I can.") but, with no full stop, mean "more is coming".
_USUALLY_CONTINUES = frozenset({
    "so", "like", "actually", "basically", "is", "was", "were", "are", "am", "have", "has",
    "had", "will", "would", "can", "could", "should", "do", "did", "in", "on", "at", "for",
    "we", "i", "it", "they", "he", "she", "you",
})


def looks_unfinished(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False
    if text.endswith(("...", "…", ",", ";", ":", "-", "–", "—")):
        return True
    ended = text[-1] in ".?!"
    words = [w for w in (_norm(t) for t in _TOKEN.findall(text)) if w]
    if not words:
        return False
    last = words[-1]
    if last in _NEVER_ENDS:
        return True
    return (not ended) and last in _USUALLY_CONTINUES

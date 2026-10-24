# apps/interviews/services/abuse_filter.py

"""
Fast, deterministic check for abusive language in a candidate's spoken
answer (English, Hinglish in Roman script, and Hindi in Devanagari).

Used by InterviewFSM.submit_answer() BEFORE any LLM call, so obvious
profanity ends the interview at once. Subtler cases (an insult without a
swear word, abuse that Whisper translated into polite-looking English)
are caught by the LLM intent check in turn_intent.py.

Only strong profanity / slurs are listed. Mild or ambiguous words
("stupid", "saala", "shit", "bc", "mc") are deliberately NOT here — they
are often used casually or are ambiguous, and terminating an interview
must not be a false positive. The LLM check decides those in context.

The same list is mirrored in frontend/static/js/interview_assist.js for
the live (in-browser) check — keep both in sync.
"""

import re
import unicodedata

# Latin-script patterns (written WITHOUT double letters, because the text
# is collapsed first), matched against normalized text (lowercase,
# leetspeak mapped, repeated letters collapsed: "fuuuck" -> "fuck",
# "maaderchod" -> "maderchod", "gaandu" -> "gandu").
_LATIN_PATTERNS = [
    # English
    r"\bfuck\w*", r"\bf+u+c+k+\w*", r"\bmotherf\w*", r"\bmofo\b",
    r"\bbitch\w*", r"\bbastard\w*", r"\basshole\w*", r"\bcunt\w*",
    r"\bwhore\w*", r"\bslut\w*", r"\bdickhead\w*", r"\bretard\w*",
    r"\bnig+er\w*", r"\bfag+ot\w*", r"\bsisterf\w*",
    # Hinglish (Roman script)
    r"\bma?d[ae]?rchod\w*", r"\bmadarchd\w*",
    r"\bb[ae]?h?[ae]n ?chod\w*", r"\bbh[ae]nchod\w*", r"\bbhnchod\w*",
    r"\bchutiy\w*", r"\bchutia\w*", r"\bchutiye\w*", r"\bchut\b",
    r"\bbhosd\w*", r"\bbhosr\w*",
    r"\bgandu\b", r"\bgand mar\w*", r"\bgand\b",
    r"\blund\b", r"\blauda\b", r"\blavda\b", r"\blawda\b", r"\bloda\b", r"\blodu\b",
    r"\brand(i|ii|iy[ao]n?)\b", r"\bharami\w*", r"\bharamzad\w*", r"\bharamkhor\w*",
    r"\bkutiya\w*", r"\bkamin[ae]\b", r"\bjhant\w*",
    r"\bteri ma ki\b", r"\bteri maa ki\b", r"\bteri behen ki\b",
    r"\bbsdk\b", r"\bbkl\b", r"\bmkc\b",
]

# Devanagari roots, matched as substrings (no word boundaries in \b sense).
_DEVANAGARI_ROOTS = [
    "मादरचोद", "मादरजात", "बहनचोद", "बहनचोद", "भेनचोद", "भैनचोद", "बहनचो",
    "चूतिया", "चुतिया", "चूतिये", "चुतिये", "चूत",
    "भोसड़", "भोसड", "भोसडी",
    "गांडू", "गाँडू", "गांड", "गाँड",
    "लौड़ा", "लौडा", "लवड़ा", "लंड",
    "रंडी", "हरामी", "हरामज़ादा", "हरामजादा", "हरामखोर",
    "कुतिया", "कमीना", "कमीने", "झांट", "झाँट",
    "तेरी माँ की", "तेरी मां की", "तेरी बहन की",
]

_LEET = str.maketrans({"@": "a", "4": "a", "1": "i", "!": "i", "0": "o", "3": "e", "$": "s", "5": "s", "7": "t"})
_LATIN_RE = re.compile("|".join(_LATIN_PATTERNS))
_MASKED_RE = re.compile(r"\b[a-z]\*{2,}", re.IGNORECASE)


def _normalize_latin(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower().translate(_LEET)
    t = re.sub(r"[^a-z\s]", " ", t)          # punctuation / masking -> space
    t = re.sub(r"([a-z])\1+", r"\1", t)       # collapse repeated letters
    return re.sub(r"\s+", " ", t).strip()


def find_abusive_terms(text: str) -> list[str]:
    """Returns the abusive words found (empty list if none)."""
    if not text or not text.strip():
        return []
    found: list[str] = []
    masked = _MASKED_RE.search(text)          # "f***" style masked profanity
    if masked:
        found.append(masked.group(0))
    norm = _normalize_latin(text)
    for m in _LATIN_RE.finditer(norm):
        found.append(m.group(0))
    deva = unicodedata.normalize("NFC", text)
    for root in _DEVANAGARI_ROOTS:
        if unicodedata.normalize("NFC", root) in deva:
            found.append(root)
    # de-duplicate, keep order
    return list(dict.fromkeys(found))

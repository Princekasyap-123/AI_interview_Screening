# apps/interviews/services/language.py

"""
English / Hindi support for the interview.

A session has a language_mode:
    "en"    English only (the default — behaves exactly as before)
    "hi"    Hindi only
    "both"  starts in English; when the candidate answers in Hindi the
            interviewer switches to Hindi (voice AND text), and switches back
            when the candidate answers in English again.

The interview logic (questions, follow-ups, replies) is written in English.
When the active language is Hindi, every line the interviewer SPEAKS is
translated here, right before it is sent to the browser, so the FSM / prompts
stay untouched. If a translation ever fails the line is spoken in English
(with the English voice) instead of breaking the interview.
"""

import logging
import re
import threading

logger = logging.getLogger(__name__)

EN = "en"
HI = "hi"
MODE_EN, MODE_HI, MODE_BOTH = "en", "hi", "both"

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)

_HINDI_WHISPER_NAMES = {"hindi", "urdu"}     # Whisper often labels Hindi speech "urdu"
_MIN_WORDS_TO_SWITCH_BACK = 3                # one English word in a Hindi answer is not a switch


def initial_language(mode: str) -> str:
    return HI if mode == MODE_HI else EN


def stt_language_for(mode: str) -> str | None:
    """The `language` to pass to STTClient: None = project default (English), "" = auto-detect."""
    if mode == MODE_HI:
        return HI
    if mode == MODE_BOTH:
        return ""
    return None


def devanagari_ratio(text: str) -> float:
    letters = _LETTER.findall(text or "")
    if not letters:
        return 0.0
    return len(_DEVANAGARI.findall(text)) / len(letters)


def detect_language(text: str, whisper_language: str, mode: str, current: str) -> str:
    """Which language the interviewer should use for its NEXT line."""
    if mode == MODE_HI:
        return HI
    if mode != MODE_BOTH:
        return EN
    words = len((text or "").split())
    if words < 2:
        return current                    # "yes", "ok": too little to tell
    ratio = devanagari_ratio(text)
    heard_hindi = ratio >= 0.3 or (whisper_language in _HINDI_WHISPER_NAMES and words >= 3)
    if heard_hindi:
        return HI
    if words >= _MIN_WORDS_TO_SWITCH_BACK and ratio < 0.1:
        return EN
    return current


# ----------------------------------------------------------------------
# English -> Hindi translation of interviewer lines
# ----------------------------------------------------------------------

_cache: dict[str, str] = {}
_statics: list[tuple[str, str]] = []        # (english, hindi) lines written by hand
_lock = threading.Lock()
_MAX_CACHE = 600

_SYSTEM_PROMPT = (
    "You translate lines spoken by an AI job interviewer into natural, polite, spoken Hindi "
    "(Devanagari script). A Hindi text-to-speech voice reads your output aloud.\n"
    "Rules:\n"
    "- Keep the exact meaning. Use the respectful 'आप' form. Sound like a courteous human interviewer.\n"
    "- The interviewer is a woman: when she speaks about herself use feminine forms "
    "(पूछूँगी, बताऊँगी, सुन पाई). The candidate's gender is UNKNOWN: never use gendered verb forms "
    "for the candidate; prefer polite imperatives (बताइए, समझाइए, साझा कीजिए) over 'आप ... सकते/सकती हैं'.\n"
    "- Write English technical terms, tool names, company names and people's names in Devanagari "
    "transliteration (Python -> पायथन, Django -> जैंगो, SQL -> एस क्यू एल, White Force -> व्हाइट फोर्स, "
    "Rahul -> राहुल) so the Hindi voice pronounces them well.\n"
    "- Short sentences, each ending with '।' or '?'. Digits stay as digits.\n"
    "- Output ONLY the Hindi text: no quotes, no notes, no markdown, no English words in Latin letters."
)


def register_static(english: str, hindi: str) -> None:
    """A hand-written Hindi line for a fixed English line (no LLM call, no delay)."""
    english = (english or "").strip()
    if english and hindi:
        with _lock:
            _cache[english] = hindi.strip()
            if not any(e == english for e, _ in _statics):
                _statics.append((english, hindi.strip()))


# The candidate's gender is unknown, and the model sometimes still writes "आप बता सकती हैं".
# The interviewer's own lines use "हूँ" / "ऊँगी" (never "ती हैं"), so this only touches how the
# candidate is addressed. The respectful plural masculine is the neutral polite form for "आप".
_NEUTRALISE = (
    (re.compile(r"(?<=[ऀ-ॿ])ती हैं"), "ते हैं"),
    (re.compile(r"रही हैं"), "रहे हैं"),
    (re.compile(r"चुकी हैं"), "चुके हैं"),
)


def _neutralise(text: str) -> str:
    for pattern, repl in _NEUTRALISE:
        text = pattern.sub(repl, text)
    return text


def _looks_hindi(text: str) -> bool:
    return devanagari_ratio(text) >= 0.5


def _llm_translate(text: str) -> str | None:
    from apps.ai_engine.llm_client import llm_client       # imported late: no Groq key needed to import this module

    try:
        out = llm_client.generate_reasoning(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=text,
            temperature=0.2,
            max_tokens=1800,                 # Devanagari is token-hungry, and the model "thinks" first
            reasoning_effort="low",
        )
    except Exception as exc:                 # RuntimeError from the client, network errors...
        logger.warning("Hindi translation failed: %s", exc)
        return None
    out = (out or "").strip().strip('"“”«»').strip()
    if not out or not _looks_hindi(out):
        logger.warning("Hindi translation returned unusable text: %.80r", out)
        return None
    return _neutralise(out)


def to_hindi(text: str) -> str | None:
    """Hindi version of `text`, or None when it could not be translated."""
    text = (text or "").strip()
    if not text:
        return None
    if _looks_hindi(text):
        return text
    with _lock:
        hit = _cache.get(text)
        statics = list(_statics)
    if hit:
        return hit

    # "Of course. <question>": the fixed opening is already known, so only the rest is translated
    # (and the question itself was normally translated ahead of time — see warm_hindi_in_background).
    for english, hindi in statics:
        if text.startswith(english) and len(text) > len(english) + 1:
            rest = to_hindi(text[len(english):])
            return f"{hindi} {rest}" if rest else None

    result = _llm_translate(text)
    if result:
        with _lock:
            if len(_cache) >= _MAX_CACHE:
                for key in list(_cache)[: _MAX_CACHE // 4]:
                    if key not in {e for e, _ in _statics}:
                        _cache.pop(key, None)
            _cache[text] = result
    return result


def warm_hindi_in_background(texts: list[str]) -> None:
    """Translates (and pre-synthesises Hindi audio for) lines the interviewer is about to need,
    in one daemon thread, so switching to Hindi does not add a wait. Never raises."""
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return

    def _run():
        spoken = []
        for text in texts:
            hindi = to_hindi(text)
            if hindi:
                spoken.append(hindi)
        try:
            from apps.avatar.services.tts_prefetch import prefetch_in_background

            prefetch_in_background(spoken, lang=HI)
        except Exception as exc:
            logger.warning("Hindi TTS prefetch could not start: %s", exc)

    threading.Thread(target=_run, name="hindi-warmup", daemon=True).start()

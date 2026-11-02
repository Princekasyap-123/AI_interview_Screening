# apps/interviews/services/turn_intent.py

"""
Decides what a candidate's spoken reply IS before it is scored:
an answer, a request to repeat the question, a clarification / counter
question, or abusive language. Used by InterviewFSM.submit_answer().

Three layers now (latency-focused restructure):
  1. abuse_filter.find_abusive_terms(): instant keyword check — no LLM.
  2. ONE tiny, fast classify-only LLM call (FAST_MODEL, low reasoning
     effort, ~15 tokens of actual output) — just intent + evidence +
     has_aside. No reply text is generated here, so there's very little
     for the model to write and the call returns quickly.
  3. ONLY when a spoken reply is actually needed (clarification /
     candidate_question / an aside after a normal answer), the reply
     text is STREAMED token-by-token. As each sentence finishes, its
     Kokoro TTS audio is kicked off synthesizing in a background thread
     immediately — overlapping the rest of the LLM's writing with TTS
     synthesis, instead of doing them one after another. By the time the
     full turn reaches the frontend, its audio is already warm in
     Kokoro's cache.

  "repeat" never calls an LLM for its reply at all: the reply is the
  fixed REPEAT_REPLY constant below, and interview_state.py pre-warms
  the exact "REPEAT_REPLY + question" text for every question at
  session start — so a "can you repeat that?" is answered with zero
  LLM-reply latency and zero cold-cache TTS wait.

Never raises: on any LLM problem the reply is treated as a normal answer,
so the interview always continues.
"""

import logging
import re
import threading

from apps.ai_engine.llm_client import llm_client
from apps.ai_engine.prompts.turn_intent_prompt import (
    TURN_INTENT_CLASSIFY_PROMPT,
    TURN_REPLY_SYSTEM_PROMPT,
    build_classify_user_prompt,
    build_reply_user_prompt,
)
from apps.interviews.services import scheduling
from apps.interviews.services.abuse_filter import find_abusive_terms

logger = logging.getLogger(__name__)

INTENT_ANSWER = "answer"
INTENT_REPEAT = "repeat"
INTENT_CLARIFICATION = "clarification"
INTENT_CANDIDATE_QUESTION = "candidate_question"
INTENT_OFF_TOPIC = "off_topic"
INTENT_UNCLEAR = "unclear"      # background noise / someone else talking / not addressed to the interviewer
INTENT_ABUSIVE = "abusive"
INTENT_RESCHEDULE = "reschedule"    # "I am not ready" / "I don't want to give the interview now"
_VALID = {
    INTENT_ANSWER, INTENT_REPEAT, INTENT_CLARIFICATION, INTENT_CANDIDATE_QUESTION,
    INTENT_OFF_TOPIC, INTENT_UNCLEAR, INTENT_ABUSIVE, INTENT_RESCHEDULE,
}

# The candidate cannot / does not want to interview right now. Short replies only: a long
# answer that happens to contain "later" is an answer.
_RESCHEDULE_RE = re.compile(
    r"\bnot ready\b|\breschedule\b|\bre-schedule\b|\bpostpone\b|\bsome other time\b|\banother time\b|\banother day\b"
    r"|\b(?:don't|do not|dont|doesn't|can't|cannot|can not|won't) (?:want to |wish to |like to )?(?:give|take|do|attend|continue|have|appear for)(?: with)?(?: this| the| my)? interview\b"
    r"|\b(?:not|isn't|is not) (?:a )?(?:good|right|suitable|convenient) time\b|\bbad time\b"
    r"|\b(?:i(?:'m| am)|i have been) (?:very )?busy (?:right )?now\b"
    r"|\b(?:can|could) we (?:do|have|take|schedule|continue) (?:it|this|the interview) (?:later|tomorrow|another|some other|next)"
    r"|\b(?:abhi|aaj) nahi\b|\bbaad mein\b|\bkal karte\b|\btaiyaar nahi\b|\btayyar nahi\b"
    r"|अभी तैयार नहीं|बाद में|अभी नहीं|इंटरव्यू नहीं (?:देना|देनी|दे)|समय नहीं है",
    re.IGNORECASE,
)
_RESCHEDULE_MAX_WORDS = 22
# Built from the settings (working hours, one-month window) so the spoken rule is the real rule.
RESCHEDULE_ASK = (
    "No problem at all. When would be a comfortable time for you? Please tell me a date and a time "
    f"on a working day, Monday to Friday between {scheduling.hours_phrase()}, within the next {scheduling.max_days()} days."
)

# Obvious "please repeat" requests are answered without any LLM call at all.
_REPEAT_FAST_RE = re.compile(
    r"^\W*(?:sorry\W+|excuse me\W+|please\W+|ji\W+)*"
    r"(?:(?:can|could|would) you (?:please )?(?:repeat|say)(?: that| it| the question)?(?: again| once more)?(?: please)?"
    r"|repeat(?: that| it| the question)?(?: again| once more)?(?: please)?"
    r"|say (?:that|it) again(?: please)?"
    r"|pardon(?: me)?|come again|one more time(?: please)?"
    r"|(?:phir se|dobara|ek baar aur) (?:boliye|bataiye|bolo|batao)(?: please)?)\W*$",
    re.IGNORECASE,
)

# Replies longer than this, without "?" or Devanagari, are treated as
# answers without an LLM call.
LLM_CHECK_MAX_WORDS = 60
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# Same sentence-boundary regex as tts_prefetch.split_chunks / TTSPlayer._chunks
# — MUST stay identical, so a streamed sentence's cache key matches exactly
# what the frontend will request.
_SENTENCE_SPLIT = re.compile(r"""(?<=[.!?…])\s+(?=[A-Z0-9"'(])""")
_MERGE_UNDER_CHARS = 40  # mirrors split_chunks(): short trailing chunks get merged forward

REPEAT_REPLY = "Of course."
OFF_TOPIC_REPLY = (
    "I'm the AI interviewer for White Force, and I'm only able to help with this interview."
)
UNCLEAR_REPLY = (
    "Sorry, I couldn't hear that clearly — there seemed to be some background noise."
)
CANDIDATE_QUESTION_FALLBACK = (
    "That's a fair question. The recruiting team will share those details with you after this interview."
)
ABOUT_AI_REPLY = "I'm the AI interviewer from White Force, and I'll be the one asking the questions today."
COMPANY_REPLY = "This interview is being conducted by White Force."
HOW_KNOW_REPLY = (
    "Your application was shortlisted by the White Force recruiting team, "
    "and that is why you have been invited to this screening interview."
)
GREETING_REPLY = "I'm doing well, thank you for asking."
THANKS_REPLY = "You're welcome."
DURATION_REPLY = "This is a short screening interview, and it usually takes around fifteen to twenty minutes."

# Short questions ABOUT THE INTERVIEWER ("tell me about yourself", "who are you")
# and "which company is this" are answered with fixed text: no LLM call, so
# no latency and — more important — no chance of the model mixing up who is
# the interviewer and who is the candidate.
_ABOUT_AI_RE = re.compile(
    r"\b(?:tell me (?:something |more )?about yourself|who are you|what are you"
    r"|are you (?:a |an )?(?:real|human|person|robot|bot|ai|machine))\b",
    re.IGNORECASE,
)
_COMPANY_RE = re.compile(
    r"\b(?:which|what) (?:company|organi[sz]ation)\b|\bcompany(?:'s)? name\b"
    r"|\bname of (?:the|your|this) (?:company|organi[sz]ation)\b|\bwho (?:do you|are you) work(?:ing)? for\b",
    re.IGNORECASE,
)
# "How do you know me / why was I called / where did you get my details".
_HOW_KNOW_RE = re.compile(
    r"\bhow (?:do|did|come) you (?:know|have) (?:me|my)\b|\bwhere did you (?:get|find) (?:my|me)\b"
    r"|\bhow did you (?:get|find) (?:my|me)\b|\bwhy (?:am i|was i|have i been|did you) (?:being )?(?:interview|call|select|shortlist|contact|invit)"
    r"|\bhow (?:was|am) i (?:selected|shortlisted|chosen)\b|\bwho (?:told|gave) you my\b"
    r"|(?:aap|tum)\s+mujhe\s+kaise\s+(?:jaan|jaa?nt)|mere\s+(?:details|number|naam)\s+(?:kahan|kaise)"
    r"|आप\s+मुझे\s+कैसे\s+जान|मेरी\s+(?:जानकारी|डिटेल)\s+कहाँ",
    re.IGNORECASE,
)
_GREETING_RE = re.compile(
    r"^\W*(?:hi|hello|hey)?\W*(?:how are you|how're you|how do you do|kaise ho|kaisi ho|aap kaise ho|आप कैसे हैं|आप कैसी हैं)\W*(?:today)?\W*$",
    re.IGNORECASE,
)
_THANKS_RE = re.compile(r"^\W*(?:ok(?:ay)?\W+)?(?:thanks?|thank you|shukriya|dhanyavaad|धन्यवाद|शुक्रिया)(?: so much| a lot)?\W*$", re.IGNORECASE)
_DURATION_RE = re.compile(
    r"\bhow (?:long|much time)\b.*\b(?:interview|take|last|go)\b|\bhow many (?:more )?questions\b",
    re.IGNORECASE,
)
_SHORT_QUERY_MAX_WORDS = 14


def sanitize_reply(reply: str, question_text: str) -> str:
    """
    Guards the LLM-written reply against role confusion. A reply that asks
    the candidate something (contains "?") or just echoes the interview
    question is thrown away ("" -> the caller uses its fixed fallback).
    """
    reply = (reply or "").strip()
    if not reply or "?" in reply:
        return ""
    from difflib import SequenceMatcher
    r_words = re.findall(r"[a-z0-9']+", reply.lower())
    q_words = re.findall(r"[a-z0-9']+", (question_text or "").lower())
    if len(r_words) >= 6 and q_words:
        match = SequenceMatcher(None, r_words, q_words, autojunk=False).find_longest_match(
            0, len(r_words), 0, len(q_words)
        )
        if match.size >= 6:
            return ""
    return reply


class TurnIntent:
    def __init__(self, intent=INTENT_ANSWER, reply="", aside_reply="", evidence="", source="default"):
        self.intent = intent
        self.reply = reply
        self.aside_reply = aside_reply
        self.evidence = evidence
        self.source = source      # "keywords" | "llm" | "default"

    def __repr__(self):
        return f"TurnIntent({self.intent!r}, source={self.source!r})"


def _needs_llm_check(text: str) -> bool:
    words = len(text.split())
    return words <= LLM_CHECK_MAX_WORDS or "?" in text or bool(_DEVANAGARI.search(text))


def _prewarm_tts(sentence: str) -> None:
    """Synthesizes one sentence in a background thread so it's already
    cached by the time the frontend fetches it. Never blocks, never raises."""
    sentence = (sentence or "").strip()
    if not sentence:
        return

    def _run():
        try:
            from apps.avatar.services.tts_engine import tts_engine
            tts_engine.synthesize(sentence)
        except Exception as exc:
            logger.warning("TTS pre-warm failed for %r: %s", sentence[:60], exc)

    threading.Thread(target=_run, name="tts-prewarm", daemon=True).start()


def _stream_reply(system_prompt: str, user_prompt: str) -> str:
    """
    Streams the reply text and pre-warms Kokoro TTS per finished sentence,
    using the SAME merge rule as split_chunks()/_chunks() so the chunk
    text (and therefore the cache key) is identical to what the frontend
    will actually request. Returns the full reply text; "" on failure.
    """
    buf = ""
    finalized: list[str] = []
    pending: str | None = None

    def _emit(sentence: str) -> None:
        nonlocal pending
        sentence = sentence.strip()
        if not sentence:
            return
        if pending is not None and len(pending) < _MERGE_UNDER_CHARS:
            pending = pending + " " + sentence           # merges — not final yet, don't synth
        else:
            if pending is not None:
                finalized.append(pending)
                _prewarm_tts(pending)                      # previous chunk is now final
            pending = sentence

    try:
        for delta in llm_client.generate_text_stream(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=llm_client.FAST_MODEL,
            temperature=0.3,
            max_tokens=400,
            reasoning_effort="low",
        ):
            buf += delta
            parts = _SENTENCE_SPLIT.split(buf)
            while len(parts) > 1:
                _emit(parts.pop(0))
                buf = " ".join(parts)
                parts = _SENTENCE_SPLIT.split(buf)
    except Exception as exc:
        logger.warning("Reply streaming failed: %s", exc)

    _emit(buf)
    if pending is not None:
        finalized.append(pending)
        _prewarm_tts(pending)

    return " ".join(finalized).strip()


def classify_fast(transcript_text: str) -> "TurnIntent | None":
    """
    The part of the intent check that needs NO LLM: empty reply, abusive
    keywords, an obvious "please repeat", or a long reply that is clearly an
    answer. Returns None when an LLM check is needed (classify_reply_llm).
    """
    text = (transcript_text or "").strip()
    if not text:
        return TurnIntent()

    terms = find_abusive_terms(text)
    if terms:
        return TurnIntent(INTENT_ABUSIVE, evidence=", ".join(terms), source="keywords")

    if len(text.split()) <= 12 and _REPEAT_FAST_RE.match(text):
        return TurnIntent(INTENT_REPEAT, reply=REPEAT_REPLY, source="keywords")

    if len(text.split()) <= _RESCHEDULE_MAX_WORDS and _RESCHEDULE_RE.search(text):
        return TurnIntent(INTENT_RESCHEDULE, reply=RESCHEDULE_ASK, source="keywords")

    if len(text.split()) <= _SHORT_QUERY_MAX_WORDS:
        if _ABOUT_AI_RE.search(text):
            return TurnIntent(INTENT_CANDIDATE_QUESTION, reply=ABOUT_AI_REPLY, source="keywords")
        if _COMPANY_RE.search(text):
            return TurnIntent(INTENT_CANDIDATE_QUESTION, reply=COMPANY_REPLY, source="keywords")
        if _HOW_KNOW_RE.search(text):
            return TurnIntent(INTENT_CANDIDATE_QUESTION, reply=HOW_KNOW_REPLY, source="keywords")
        if _GREETING_RE.match(text):
            return TurnIntent(INTENT_CANDIDATE_QUESTION, reply=GREETING_REPLY, source="keywords")
        if _THANKS_RE.match(text):
            return TurnIntent(INTENT_CANDIDATE_QUESTION, reply=THANKS_REPLY, source="keywords")
        if _DURATION_RE.search(text):
            return TurnIntent(INTENT_CANDIDATE_QUESTION, reply=DURATION_REPLY, source="keywords")

    if not _needs_llm_check(text):
        return TurnIntent()
    return None


def classify_reply(question_text: str, transcript_text: str) -> TurnIntent:
    fast = classify_fast(transcript_text)
    return fast if fast is not None else classify_reply_llm(question_text, transcript_text)


def classify_reply_llm(question_text: str, transcript_text: str) -> TurnIntent:
    """The LLM intent check (plus streamed reply text when one is needed). Never raises."""
    text = (transcript_text or "").strip()
    if not text:
        return TurnIntent()

    try:
        result = llm_client.generate_json(
            system_prompt=TURN_INTENT_CLASSIFY_PROMPT,
            user_prompt=build_classify_user_prompt(question_text, text),
            model=llm_client.FAST_MODEL,
            temperature=0.1,
            max_tokens=512,
            reasoning_effort="low",
        )
    except Exception as exc:    # never let an intent-check problem stop the interview
        logger.warning("Turn intent check failed, treating reply as an answer: %s", exc)
        return TurnIntent()

    intent = str(result.get("intent", "")).strip().lower()
    if intent not in _VALID:
        logger.warning("Turn intent check returned unknown intent %r — treating as answer.", intent)
        return TurnIntent()

    if intent == INTENT_ABUSIVE:
        return TurnIntent(
            INTENT_ABUSIVE,
            evidence=str(result.get("evidence", "") or "").strip(),
            source="llm",
        )

    if intent == INTENT_REPEAT:
        return TurnIntent(INTENT_REPEAT, reply=REPEAT_REPLY, source="llm")

    if intent == INTENT_RESCHEDULE:
        return TurnIntent(INTENT_RESCHEDULE, reply=RESCHEDULE_ASK, source="llm")

    if intent == INTENT_UNCLEAR:
        return TurnIntent(INTENT_UNCLEAR, reply=UNCLEAR_REPLY, source="llm")

    if intent == INTENT_OFF_TOPIC:
        # Fixed, formal brush-off — no LLM text call, no cold TTS (pre-warmed
        # alongside REPEAT_REPLY in interview_state._prefetch_speech()).
        return TurnIntent(INTENT_OFF_TOPIC, reply=OFF_TOPIC_REPLY, source="llm")

    if intent == INTENT_CLARIFICATION:
        reply = sanitize_reply(
            _stream_reply(
                TURN_REPLY_SYSTEM_PROMPT,
                build_reply_user_prompt(question_text, text, "clarification"),
            ),
            question_text,
        )
        return TurnIntent(INTENT_CLARIFICATION, reply=reply, source="llm")

    if intent == INTENT_CANDIDATE_QUESTION:
        reply = sanitize_reply(
            _stream_reply(
                TURN_REPLY_SYSTEM_PROMPT,
                build_reply_user_prompt(question_text, text, "candidate_question"),
            ),
            question_text,
        ) or CANDIDATE_QUESTION_FALLBACK
        return TurnIntent(INTENT_CANDIDATE_QUESTION, reply=reply, source="llm")

    # INTENT_ANSWER
    has_aside = bool(result.get("has_aside"))
    if not has_aside:
        return TurnIntent(INTENT_ANSWER, source="llm")

    aside_reply = sanitize_reply(
        _stream_reply(
            TURN_REPLY_SYSTEM_PROMPT,
            build_reply_user_prompt(question_text, text, "candidate_question"),
        ),
        question_text,
    ) or CANDIDATE_QUESTION_FALLBACK
    return TurnIntent(INTENT_ANSWER, aside_reply=aside_reply, source="llm")
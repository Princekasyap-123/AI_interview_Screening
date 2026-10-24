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
from apps.interviews.services.abuse_filter import find_abusive_terms

logger = logging.getLogger(__name__)

INTENT_ANSWER = "answer"
INTENT_REPEAT = "repeat"
INTENT_CLARIFICATION = "clarification"
INTENT_CANDIDATE_QUESTION = "candidate_question"
INTENT_OFF_TOPIC = "off_topic"
INTENT_ABUSIVE = "abusive"
_VALID = {INTENT_ANSWER, INTENT_REPEAT, INTENT_CLARIFICATION, INTENT_CANDIDATE_QUESTION, INTENT_ABUSIVE}

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
CANDIDATE_QUESTION_FALLBACK = (
    "That's a fair question. The recruiting team will share those details with you after this interview."
)


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


def classify_reply(question_text: str, transcript_text: str) -> TurnIntent:
    text = (transcript_text or "").strip()
    if not text:
        return TurnIntent()

    terms = find_abusive_terms(text)
    if terms:
        return TurnIntent(INTENT_ABUSIVE, evidence=", ".join(terms), source="keywords")

    if not _needs_llm_check(text):
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
    except (RuntimeError, ValueError) as exc:
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

    if intent == INTENT_OFF_TOPIC:
        # Fixed, formal brush-off — no LLM text call, no cold TTS (pre-warmed
        # alongside REPEAT_REPLY in interview_state._prefetch_speech()).
        return TurnIntent(INTENT_OFF_TOPIC, reply=OFF_TOPIC_REPLY, source="llm")

    if intent == INTENT_CLARIFICATION:
        reply = _stream_reply(
            TURN_REPLY_SYSTEM_PROMPT,
            build_reply_user_prompt(question_text, text, "clarification"),
        )
        return TurnIntent(INTENT_CLARIFICATION, reply=reply, source="llm")

    if intent == INTENT_CANDIDATE_QUESTION:
        reply = _stream_reply(
            TURN_REPLY_SYSTEM_PROMPT,
            build_reply_user_prompt(question_text, text, "candidate_question"),
        ) or CANDIDATE_QUESTION_FALLBACK
        return TurnIntent(INTENT_CANDIDATE_QUESTION, reply=reply, source="llm")

    # INTENT_ANSWER
    has_aside = bool(result.get("has_aside"))
    if not has_aside:
        return TurnIntent(INTENT_ANSWER, source="llm")

    aside_reply = _stream_reply(
        TURN_REPLY_SYSTEM_PROMPT,
        build_reply_user_prompt(question_text, text, "candidate_question"),
    ) or CANDIDATE_QUESTION_FALLBACK
    return TurnIntent(INTENT_ANSWER, aside_reply=aside_reply, source="llm")
# apps/interviews/services/question_generator.py

import logging
import re

from django.db import transaction
from django.utils import timezone
from apps.candidates.services import resolve_candidate_profile

from apps.ai_engine.llm_client import llm_client
from apps.ai_engine.prompts.question_gen_prompt import (
    QUESTION_GEN_SYSTEM_PROMPT,
    build_question_gen_user_prompt,
)
from apps.interviews.models import InterviewSession, Question

logger = logging.getLogger(__name__)


class QuestionGenerationError(Exception):
    """Raised when question generation fails or returns malformed data."""


# Used only if the LLM fails to produce a usable self-intro question as
# question #1 — guarantees every interview opens the same, predictable
# way even if generation is degraded.
FALLBACK_INTRO_TEXT = (
    "To get started, could you please introduce yourself — walk me "
    "through your background, your current role, and what you've been "
    "working on recently?"
)

_INTRO_SIGNAL_PHRASES = (
    "introduce yourself",
    "tell me about yourself",
    "tell me a bit about yourself",
    "walk me through your background",
    "your background",
    "start by introducing",
)


def generate_questions_for_session(session: InterviewSession) -> list[Question]:
    """
    Generates the full question set for an InterviewSession using the
    candidate's profile + optional JD, and persists them as Question rows.

    Should be called once, right when the session moves from PENDING to
    IN_PROGRESS — not per-question. The consumer then just pulls rows in
    order as the interview progresses.

    Concurrency: guarded with select_for_update() on the session row.
    Two near-simultaneous calls (e.g. a duplicate WebSocket 'join'
    racing fsm.start() twice) could otherwise both pass the "does this
    session already have questions" check before either commits, and
    each insert a full question set — producing two Question(order=1)
    rows for the same session. That makes "next_order" lookups
    nondeterministic and is a very plausible cause of a question
    appearing to repeat. The lock closes that race at the source,
    independent of any fix on the consumer side.

    Raises QuestionGenerationError on any failure so the caller (consumer)
    can decide how to fail gracefully.
    """
    with transaction.atomic():
        locked_session = InterviewSession.objects.select_for_update().get(
            pk=session.pk
        )

        existing = list(locked_session.questions.order_by("order"))
        if existing:
            logger.warning(
                "generate_questions_for_session called on session %s "
                "which already has %d question(s) — skipping "
                "regeneration. If this session's questions look "
                "duplicated/repeated in the transcript, check for "
                "repeated 'WebSocket connected'/'join' log lines around "
                "the same timestamp — that indicates the race this lock "
                "is meant to prevent happened before this fix was applied.",
                locked_session.id,
                len(existing),
            )
            return existing

        candidate = locked_session.candidate
        candidate_profile = resolve_candidate_profile(candidate)

        logger.info(
            "Generating question set for session %s (candidate=%s, "
            "max_questions=%s, has_jd=%s)",
            locked_session.id,
            candidate.id,
            locked_session.max_questions,
            bool(locked_session.job_description_text.strip()),
        )

        try:
            result = llm_client.generate_json(
                system_prompt=QUESTION_GEN_SYSTEM_PROMPT,
                user_prompt=build_question_gen_user_prompt(
                    candidate_profile=candidate_profile,
                    job_description_text=locked_session.job_description_text,
                    num_questions=locked_session.max_questions,
                ),
            )
        except (RuntimeError, ValueError) as exc:
            logger.error(
                "Question generation failed for session %s: %s",
                locked_session.id,
                exc,
            )
            raise QuestionGenerationError(str(exc)) from exc

        questions_data = result.get("questions")
        if not questions_data or not isinstance(questions_data, list):
            logger.error(
                "Hallucination/malformed-output guard tripped for "
                "session %s — response had no valid 'questions' list. "
                "Raw top-level keys: %s",
                locked_session.id,
                list(result.keys()) if isinstance(result, dict) else type(result),
            )
            raise QuestionGenerationError(
                "LLM response did not contain a valid 'questions' list."
            )

        validated = _validate_and_sort(
            questions_data,
            expected_count=locked_session.max_questions,
            session_id=locked_session.id,
        )

        question_objs = [
            Question(
                session=locked_session,
                order=item["order"],
                text=item["text"],
                topic_tag=item.get("topic_tag", ""),
                generation_context={
                    "based_on": item.get("based_on", ""),
                    "candidate_profile_snapshot": candidate_profile,
                    "job_description_used": bool(
                        locked_session.job_description_text.strip()
                    ),
                    "generated_at": timezone.now().isoformat(),
                },
            )
            for item in validated
        ]
        Question.objects.bulk_create(question_objs)

    logger.info(
        "Generated %d questions for session %s (Q1 topic_tag=%s)",
        len(question_objs),
        session.id,
        question_objs[0].topic_tag if question_objs else "none",
    )
    return list(session.questions.order_by("order"))


def _validate_and_sort(
    questions_data: list[dict], expected_count: int, session_id=None
) -> list[dict]:
    """
    Validates the LLM's question list before it touches the DB:
    - every item has required keys and non-empty text
    - order values are sequential starting at 1 (re-numbered positionally
      if not, rather than hard-failing the whole interview)
    - no two questions are duplicate/near-duplicate text — this is the
      main hallucination signal that shows up live as "it keeps asking
      the same question"
    - question 1 is tagged "intro" AND actually reads like a self-intro
      prompt (falls back to FALLBACK_INTRO_TEXT if the LLM hallucinated
      a technical question into slot 1 but mistagged it "intro")
    - count matches what was requested (warns, doesn't hard-fail)
    """
    required_keys = {"order", "text"}
    for item in questions_data:
        missing = required_keys - item.keys()
        if missing:
            raise QuestionGenerationError(
                f"Question item missing required keys: {missing} — item: {item}"
            )
        if not isinstance(item["text"], str) or not item["text"].strip():
            raise QuestionGenerationError(f"Question item has empty text: {item}")

    sorted_data = sorted(questions_data, key=lambda q: q["order"])

    expected_orders = list(range(1, len(sorted_data) + 1))
    actual_orders = [q["order"] for q in sorted_data]
    if actual_orders != expected_orders:
        logger.error(
            "Hallucination guard: non-sequential/duplicate order values "
            "for session %s — got %s. Re-numbering positionally instead "
            "of hard-failing.",
            session_id,
            actual_orders,
        )
        for idx, item in enumerate(sorted_data, start=1):
            item["order"] = idx

    seen_normalized: dict[str, int] = {}
    for item in sorted_data:
        normalized = _normalize_question_text(item["text"])
        if normalized in seen_normalized:
            logger.error(
                "Hallucination guard: question order=%s duplicates "
                "question order=%s for session %s (normalized text "
                "match). Text: %r",
                item["order"],
                seen_normalized[normalized],
                session_id,
                item["text"],
            )
        else:
            seen_normalized[normalized] = item["order"]

    first = sorted_data[0]
    if first.get("topic_tag") != "intro":
        logger.warning(
            "First generated question was not tagged 'intro' for "
            "session %s — forcing tag. Original tag: %s",
            session_id,
            first.get("topic_tag"),
        )
        first["topic_tag"] = "intro"

    if not _looks_like_intro(first["text"]):
        logger.error(
            "Hallucination guard: question #1 for session %s is tagged "
            "'intro' but doesn't read like a self-intro prompt — "
            "replacing with fallback text. Original: %r",
            session_id,
            first["text"],
        )
        first["text"] = FALLBACK_INTRO_TEXT

    if len(sorted_data) != expected_count:
        logger.warning(
            "Requested %d questions but LLM returned %d for session %s "
            "— proceeding with what was returned.",
            expected_count,
            len(sorted_data),
            session_id,
        )

    return sorted_data


def _normalize_question_text(text: str) -> str:
    """Lowercase, strip punctuation/whitespace so trivial rewordings of
    the same question still match for the duplicate-text guard."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _looks_like_intro(text: str) -> bool:
    normalized = text.lower()
    return any(phrase in normalized for phrase in _INTRO_SIGNAL_PHRASES)
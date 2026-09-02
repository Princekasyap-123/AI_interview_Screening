# apps/interviews/services/question_generator.py

import logging

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


def generate_questions_for_session(session: InterviewSession) -> list[Question]:
    """
    Generates the full question set for an InterviewSession using the
    candidate's profile + optional JD, and persists them as Question rows.

    Should be called once, right when the session moves from PENDING to
    IN_PROGRESS — not per-question. The consumer then just pulls rows in
    order as the interview progresses.

    Raises QuestionGenerationError on any failure so the caller (consumer)
    can decide how to fail gracefully (e.g. tell candidate to retry,
    rather than starting an interview with zero questions).
    """
    if session.questions.exists():
        logger.warning(
            "generate_questions_for_session called on session %s which "
            "already has questions — skipping regeneration.",
            session.id,
        )
        return list(session.questions.order_by("order"))

    candidate = session.candidate
    candidate_profile = resolve_candidate_profile(candidate)

    try:
        result = llm_client.generate_json(
            system_prompt=QUESTION_GEN_SYSTEM_PROMPT,
            user_prompt=build_question_gen_user_prompt(
                candidate_profile=candidate_profile,
                job_description_text=session.job_description_text,
                num_questions=session.max_questions,
            ),
        )
    except (RuntimeError, ValueError) as exc:
        logger.error(
            "Question generation failed for session %s: %s", session.id, exc
        )
        raise QuestionGenerationError(str(exc)) from exc

    questions_data = result.get("questions")
    if not questions_data or not isinstance(questions_data, list):
        raise QuestionGenerationError(
            "LLM response did not contain a valid 'questions' list."
        )

    validated = _validate_and_sort(questions_data, expected_count=session.max_questions)

    with transaction.atomic():
        question_objs = [
            Question(
                session=session,
                order=item["order"],
                text=item["text"],
                topic_tag=item.get("topic_tag", ""),
                generation_context={
                    "based_on": item.get("based_on", ""),
                    "candidate_profile_snapshot": candidate_profile,
                    "job_description_used": bool(session.job_description_text.strip()),
                    "generated_at": timezone.now().isoformat(),
                },
            )
            for item in validated
        ]
        Question.objects.bulk_create(question_objs)

    logger.info(
        "Generated %d questions for session %s", len(question_objs), session.id
    )
    return list(session.questions.order_by("order"))



def _validate_and_sort(questions_data: list[dict], expected_count: int) -> list[dict]:
    """
    Validates the LLM's question list before it touches the DB:
    - every item has required keys
    - order values are sequential starting at 1, no gaps/duplicates
    - question 1 is tagged "intro"
    - count matches what was requested (logs a warning, doesn't hard-fail,
      since a slightly-off count shouldn't kill the whole interview)
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
        raise QuestionGenerationError(
            f"Question order values are not sequential from 1: got {actual_orders}"
        )

    if sorted_data[0].get("topic_tag") != "intro":
        logger.warning(
            "First generated question was not tagged 'intro' as instructed "
            "— forcing tag. Original tag: %s",
            sorted_data[0].get("topic_tag"),
        )
        sorted_data[0]["topic_tag"] = "intro"

    if len(sorted_data) != expected_count:
        logger.warning(
            "Requested %d questions but LLM returned %d — proceeding "
            "with what was returned.",
            expected_count,
            len(sorted_data),
        )

    return sorted_data
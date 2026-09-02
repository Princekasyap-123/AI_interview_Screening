# apps/interviews/services/final_rating.py

import logging

from django.db import transaction

from apps.ai_engine.llm_client import llm_client
from apps.ai_engine.prompts.final_rating_prompt import (
    FINAL_RATING_SYSTEM_PROMPT,
    build_final_rating_user_prompt,
)
from apps.candidates.services import resolve_candidate_profile
from apps.interviews.models import FinalRating, InterviewSession

logger = logging.getLogger(__name__)

REQUIRED_CATEGORY_KEYS = {"communication", "technical_depth", "relevance"}


class FinalRatingError(Exception):
    """Raised when final rating generation fails or returns malformed data."""


def generate_final_rating_for_session(session: InterviewSession) -> FinalRating:
    """
    Generates and persists the holistic FinalRating for a completed or
    terminated InterviewSession. Pure service logic — no Celery/retry
    concerns here, those live in interviews/tasks.py which wraps this
    function in a @shared_task.

    Raises FinalRatingError on failure so the calling task can decide
    retry behavior. Idempotent: returns the existing FinalRating if one
    already exists rather than regenerating.
    """
    if session.status not in (InterviewSession.Status.COMPLETED, InterviewSession.Status.TERMINATED):
        raise FinalRatingError(
            f"Cannot generate final rating for session {session.id} — "
            f"status is '{session.status}', expected COMPLETED or TERMINATED."
        )

    existing = getattr(session, "final_rating", None)
    if existing is not None:
        logger.info(
            "Session %s already has a FinalRating — returning existing.",
            session.id,
        )
        return existing

    candidate_profile = resolve_candidate_profile(session.candidate)
    qa_pairs = _build_qa_pairs(session)

    if not qa_pairs:
        logger.warning(
            "Session %s has no answered questions — creating a minimal "
            "FinalRating flagged for manual review.",
            session.id,
        )
        return _create_no_data_rating(session)

    was_terminated_early = session.status == InterviewSession.Status.TERMINATED
    termination_reason = (
        session.get_termination_reason_display() if was_terminated_early else ""
    )

    try:
        result = llm_client.generate_json(
            system_prompt=FINAL_RATING_SYSTEM_PROMPT,
            user_prompt=build_final_rating_user_prompt(
                candidate_profile=candidate_profile,
                qa_pairs=qa_pairs,
                was_terminated_early=was_terminated_early,
                termination_reason=termination_reason,
            ),
        )
        rating_data = _validate_rating(result)

    except (RuntimeError, ValueError) as exc:
        logger.error(
            "Final rating generation failed for session %s: %s", session.id, exc
        )
        raise FinalRatingError(str(exc)) from exc

    with transaction.atomic():
        final_rating = FinalRating.objects.create(
            session=session,
            overall_score=rating_data["overall_score"],
            category_scores=rating_data["category_scores"],
            strengths=rating_data["strengths"],
            concerns=rating_data["concerns"],
            content_flags=rating_data["content_flags"],
            recruiter_visible=True,
            candidate_visible=False,
        )

    logger.info(
        "Generated FinalRating for session %s: overall_score=%.1f",
        session.id,
        rating_data["overall_score"],
    )
    return final_rating


def _build_qa_pairs(session: InterviewSession) -> list[dict]:
    """
    Flattens the session's Question/Answer rows into the shape
    build_final_rating_user_prompt() expects. Only includes questions
    that were actually asked (asked_at is not null) — unasked questions
    from a session cut short shouldn't appear as "no answer given," they
    simply never happened.
    """
    pairs = []
    for question in session.questions.all().order_by("order"):
        if question.asked_at is None:
            continue

        answer = getattr(question, "answer", None)
        pairs.append({
            "order": question.order,
            "topic_tag": question.topic_tag,
            "question_text": question.text,
            "answer_transcript": answer.transcript_text if answer else "",
            "analysis": (
                answer.analysis_json
                if answer and all(
                    answer.analysis_json.get(k) is not None
                    for k in ("relevance", "depth", "communication")
                )
                else None
            ),
        })
    return pairs


def _validate_rating(result: dict) -> dict:
    if "overall_score" not in result:
        raise FinalRatingError("Final rating response missing 'overall_score'.")

    overall_score = result["overall_score"]
    if not isinstance(overall_score, (int, float)) or not (0 <= overall_score <= 100):
        raise FinalRatingError(f"'overall_score' is not a valid 0-100 number: {overall_score!r}")

    category_scores = result.get("category_scores", {})
    missing_categories = REQUIRED_CATEGORY_KEYS - category_scores.keys()
    if missing_categories:
        raise FinalRatingError(f"category_scores missing keys: {missing_categories}")

    for key, value in category_scores.items():
        if not isinstance(value, (int, float)) or not (0 <= value <= 100):
            raise FinalRatingError(f"category_scores['{key}'] is not a valid 0-100 number: {value!r}")

    return {
        "overall_score": overall_score,
        "category_scores": category_scores,
        "strengths": result.get("strengths", "") or "",
        "concerns": result.get("concerns", "") or "",
        "content_flags": result.get("content_flags", []) or [],
    }


def _create_no_data_rating(session: InterviewSession) -> FinalRating:
    """
    Fallback for a session that completed/terminated with zero answered
    questions (e.g. terminated on the very first proctoring violation
    before any question was asked). Creates a shell rating flagged for
    manual review rather than leaving no FinalRating row at all.
    """
    return FinalRating.objects.create(
        session=session,
        overall_score=0,
        category_scores={"communication": 0, "technical_depth": 0, "relevance": 0},
        strengths="",
        concerns="No questions were answered during this session.",
        content_flags=["no_data_needs_manual_review"],
        recruiter_visible=True,
        candidate_visible=False,
    )
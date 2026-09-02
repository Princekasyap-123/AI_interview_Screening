# apps/interviews/services/answer_analyzer.py

import logging

from django.utils import timezone

from apps.ai_engine.llm_client import llm_client
from apps.ai_engine.prompts.answer_analysis_prompt import (
    ANSWER_ANALYSIS_SYSTEM_PROMPT,
    build_answer_analysis_user_prompt,
)
from apps.interviews.models import Answer, Question

logger = logging.getLogger(__name__)

# Categories the LLM is asked to score — kept as a constant so the
# aggregate score calculation and validation stay in sync with the
# prompt schema in answer_analysis_prompt.py.
SCORE_CATEGORIES = ("relevance", "depth", "communication")


class AnswerAnalysisError(Exception):
    """Raised when answer analysis fails or returns malformed data."""


def analyze_answer(
    question: Question,
    transcript_text: str,
    candidate_profile: dict,
    audio_ref: str = "",
) -> Answer:
    """
    Analyzes a candidate's answer to a single question and persists the
    result as an Answer row.

    This is called once per question, right after STT produces the
    transcript. It does NOT decide what happens next (follow-up vs. move
    on) — that's interview_state.py's job, using the should_follow_up
    flag this function stores in analysis_json.

    Raises AnswerAnalysisError on failure. Callers should catch this and
    still let the interview continue — a failed analysis on one answer
    should never crash the whole session. See _build_fallback_analysis
    for what gets stored when that happens.
    """
    now = timezone.now()

    try:
        result = llm_client.generate_json(
            system_prompt=ANSWER_ANALYSIS_SYSTEM_PROMPT,
            user_prompt=build_answer_analysis_user_prompt(
                question_text=question.text,
                question_topic_tag=question.topic_tag,
                answer_transcript=transcript_text,
                candidate_profile=candidate_profile,
            ),
        )
        analysis = _validate_analysis(result)

    except (RuntimeError, ValueError) as exc:
        logger.error(
            "Answer analysis failed for question %s: %s — storing fallback.",
            question.id,
            exc,
        )
        analysis = _build_fallback_analysis(reason=str(exc))

    score = _compute_aggregate_score(analysis)

    answer, _ = Answer.objects.update_or_create(
        question=question,
        defaults={
            "transcript_text": transcript_text or "",
            "audio_ref": audio_ref,
            "analysis_json": analysis,
            "score": score,
            "answered_at": now,
            "analyzed_at": timezone.now(),
        },
    )

    logger.info(
        "Analyzed answer for question %s (session %s): score=%.1f follow_up=%s",
        question.id,
        question.session_id,
        score if score is not None else -1,
        analysis.get("should_follow_up"),
    )

    return answer


def _validate_analysis(result: dict) -> dict:
    """
    Validates the LLM's analysis JSON against the schema defined in
    answer_analysis_prompt.py. Fills in safe defaults for any missing
    optional keys rather than hard-failing, since a missing "notes" field
    shouldn't be treated the same as a missing score.
    """
    required_score_keys = set(SCORE_CATEGORIES)
    missing_scores = required_score_keys - result.keys()
    if missing_scores:
        raise AnswerAnalysisError(
            f"Analysis response missing required score keys: {missing_scores}"
        )

    for key in SCORE_CATEGORIES:
        value = result.get(key)
        if not isinstance(value, (int, float)) or not (0 <= value <= 10):
            raise AnswerAnalysisError(
                f"Score '{key}' is not a valid 0-10 number: {value!r}"
            )

    return {
        "relevance": result["relevance"],
        "depth": result["depth"],
        "communication": result["communication"],
        "should_follow_up": bool(result.get("should_follow_up", False)),
        "follow_up_reason": result.get("follow_up_reason", "") or "",
        "notes": result.get("notes", "") or "",
        "red_flags": result.get("red_flags", []) or [],
    }


def _build_fallback_analysis(reason: str) -> dict:
    """
    What gets stored when the LLM call fails outright (network error,
    malformed JSON, etc.). Scores are left null rather than zero, so the
    dashboard/final rating logic can distinguish "genuinely scored 0"
    from "analysis failed and needs manual review" — check for None,
    not falsy, when aggregating.
    """
    return {
        "relevance": None,
        "depth": None,
        "communication": None,
        "should_follow_up": False,
        "follow_up_reason": "",
        "notes": f"Automated analysis failed: {reason}. Needs manual review.",
        "red_flags": ["analysis_failed"],
    }


def _compute_aggregate_score(analysis: dict) -> float | None:
    """
    Averages the three sub-scores into Answer.score for fast querying.
    Returns None if any sub-score is missing (i.e. the fallback case),
    so a failed analysis doesn't silently show up as a score of 0 on
    the dashboard.
    """
    values = [analysis.get(key) for key in SCORE_CATEGORIES]
    if any(v is None for v in values):
        return None
    return round(sum(values) / len(values), 2)
# apps/interviews/services/answer_analyzer.py

import logging
import re

from django.conf import settings
from django.utils import timezone

from apps.ai_engine.llm_client import llm_client
from apps.ai_engine.prompts.answer_analysis_prompt import (
    build_answer_analysis_system_prompt,
    build_answer_analysis_user_prompt,
)
from apps.interviews.models import Answer, Question

logger = logging.getLogger(__name__)

# Categories the LLM is asked to score — kept as a constant so the
# aggregate score calculation and validation stay in sync with the
# prompt schema in answer_analysis_prompt.py.
SCORE_CATEGORIES = ("relevance", "depth", "communication")

# Guardrail independent of the LLM's own judgment (belt-and-suspenders
# alongside the prompt rule in answer_analysis_prompt.py): if the
# candidate plainly said they don't know / aren't sure, there is nothing
# to "expand on" — a follow-up question in that case just repeats
# "Could you expand on that?" at someone who already said they don't
# know, which reads as broken. Matches English + common Hinglish forms.
_NO_KNOWLEDGE_RE = re.compile(
    r"\b("
    r"i\s*(do\s*n['o]?t|dont)\s*know"
    r"|no\s*idea"
    r"|not\s*sure"
    r"|i\s*have\s*no\s*idea"
    r"|i\s*am\s*not\s*aware"
    r"|i['’]?m\s*not\s*aware"
    r"|i\s*can['’]?t\s*(answer|recall|remember)"
    r"|i\s*have\s*no\s*(clue|knowledge)"
    r"|mujhe\s*nahi\s*pata"
    r"|pata\s*nahi"
    r"|malum\s*nahi"
    r")\b",
    re.IGNORECASE,
)

# A relevance score at or below this is treated as "no real content"
# even if the transcript didn't literally match the phrases above
# (e.g. a rambling but entirely off-topic answer).
_NO_KNOWLEDGE_RELEVANCE_CEILING = 2


class AnswerAnalysisError(Exception):
    """Raised when answer analysis fails or returns malformed data."""


def build_full_transcript(question: Question, new_text: str) -> str:
    """
    The text that is analysed and stored for this question. On the first
    answer it is just the new transcript. On a follow-up it is
    "<first answer>\n[Follow-up] <new answer>", so the follow-up adds to the
    first answer instead of overwriting it (recruiter and final rating see both).
    """
    new_text = (new_text or "").strip()
    prior = (
        Answer.objects.filter(question_id=question.id)
        .values_list("transcript_text", flat=True)
        .first()
    )
    prior = (prior or "").strip()
    if not prior:
        return new_text
    if not new_text:
        return prior
    return f"{prior}\n[Follow-up] {new_text}"


def run_answer_analysis(
    *,
    question_text: str,
    topic_tag: str,
    transcript_text: str,
    candidate_profile: dict,
    is_white_collar: bool,
) -> dict:
    """
    The LLM half of analyze_answer(): NO database access, safe to run in a
    worker thread in parallel with the intent check (see InterviewFSM.submit_answer).
    Never raises: any LLM / validation problem returns a fallback analysis.
    An empty transcript is not sent to the LLM at all.
    """
    if not (transcript_text or "").strip():
        return {
            "relevance": 0,
            "depth": 0,
            "communication": 0,
            "should_follow_up": False,
            "follow_up_reason": "",
            "notes": "No speech was detected for this question.",
            "red_flags": [],
        }

    try:
        result = llm_client.generate_json(
            system_prompt=build_answer_analysis_system_prompt(is_white_collar),
            user_prompt=build_answer_analysis_user_prompt(
                question_text=question_text,
                question_topic_tag=topic_tag,
                answer_transcript=transcript_text,
                candidate_profile=candidate_profile,
            ),
            model=getattr(settings, "ANSWER_ANALYSIS_MODEL", None),
            reasoning_effort="low",
            max_tokens=1500,
        )
        analysis = _validate_analysis(result)
    except (RuntimeError, ValueError, AnswerAnalysisError) as exc:
        logger.error("Answer analysis failed: %s — storing fallback.", exc)
        analysis = _build_fallback_analysis(reason=str(exc))

    return _apply_no_knowledge_override(analysis, transcript_text)


def save_answer(
    question: Question,
    transcript_text: str,
    analysis: dict,
    audio_ref: str = "",
) -> Answer:
    """The database half of analyze_answer(): persists the (already combined) transcript."""
    score = _compute_aggregate_score(analysis)
    now = timezone.now()
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
        "Saved answer for question %s (session %s): score=%s follow_up=%s",
        question.id,
        question.session_id,
        score,
        analysis.get("should_follow_up"),
    )
    return answer


def analyze_answer(
    question: Question,
    transcript_text: str,
    candidate_profile: dict,
    audio_ref: str = "",
) -> Answer:
    """
    Analyses a candidate's answer to one question and persists it as an
    Answer row (convenience wrapper: build_full_transcript + run_answer_analysis
    + save_answer, run one after another). Never raises on LLM problems.
    """
    full = build_full_transcript(question, transcript_text)
    analysis = run_answer_analysis(
        question_text=question.text,
        topic_tag=question.topic_tag,
        transcript_text=full,
        candidate_profile=candidate_profile,
        is_white_collar=question.session.is_white_collar,
    )
    return save_answer(question, full, analysis, audio_ref=audio_ref)


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


def _apply_no_knowledge_override(analysis: dict, transcript_text: str) -> dict:
    """
    Forces should_follow_up to False when the candidate gave no real
    content to follow up on — regardless of what the LLM returned. This
    is a deterministic safety net on top of the prompt-level instruction
    in answer_analysis_prompt.py, because an LLM occasionally still asks
    to "expand on" an answer that was just "I don't know".

    Triggers on either:
      - the transcript matching a direct "I don't know" / "not sure" /
        "no idea" phrase (English + common Hinglish forms), or
      - an empty transcript, or
      - a relevance score at/below _NO_KNOWLEDGE_RELEVANCE_CEILING
        (i.e. the model itself judged the answer as essentially
        off-topic/empty, even if it didn't literally say "I don't know").
    """
    text = (transcript_text or "").strip()
    relevance = analysis.get("relevance")

    no_real_content = (
        not text
        or _NO_KNOWLEDGE_RE.search(text) is not None
        or (isinstance(relevance, (int, float)) and relevance <= _NO_KNOWLEDGE_RELEVANCE_CEILING)
    )

    if no_real_content and analysis.get("should_follow_up"):
        analysis = dict(analysis)
        analysis["should_follow_up"] = False
        analysis["follow_up_reason"] = ""
        note_suffix = "[auto] should_follow_up forced to false — no substantive answer to follow up on."
        existing_notes = (analysis.get("notes") or "").strip()
        analysis["notes"] = f"{existing_notes} {note_suffix}".strip() if existing_notes else note_suffix
        logger.info(
            "Overrode should_follow_up=False for a 'no knowledge'/empty answer "
            "(relevance=%s, transcript=%r).",
            relevance,
            text[:120],
        )

    return analysis


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
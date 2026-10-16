# apps/proctoring/services/risk_scorer.py

import logging

from django.utils import timezone

from apps.candidates.services import resolve_candidate_profile
from apps.interviews.models import InterviewSession
from apps.interviews.services.interview_state import InterviewFSM
from apps.proctoring.models import ProctoringFlag, ProctoringSession
from apps.proctoring.services.flag_processor import (
    WARN_ONLY_TYPES,
    FlagProcessingResult,
    process_flag,
    reason_for,
)

logger = logging.getLogger(__name__)

# Informational 0-100 score for the recruiter dashboard only — plays no
# role in warn/terminate. Uncounted (noise-level) signals add just 1.
SEVERITY_WEIGHTS = {
    ProctoringFlag.Severity.LOW: 3,
    ProctoringFlag.Severity.MEDIUM: 8,
    ProctoringFlag.Severity.HIGH: 20,
}
UNCOUNTED_FLAG_WEIGHT = 1
RISK_SCORE_CAP = 100.0


class ProctoringActionResult:
    """What the proctoring consumer should do next."""

    ACTION_NONE = "none"
    ACTION_WARN = "warn"
    ACTION_TERMINATE = "terminate"

    def __init__(
        self,
        action: str,
        message: str = "",
        interview_turn: dict | None = None,
        flag_type: str = "",
        count: int = 0,
        limit: int = 0,
    ):
        self.action = action
        self.message = message
        self.interview_turn = interview_turn
        self.flag_type = flag_type
        self.count = count  # how many counted violations of this type so far
        self.limit = limit  # warnings allowed for this type before termination


def get_or_create_proctoring_session(interview_session: InterviewSession) -> ProctoringSession:
    proctoring_session, created = ProctoringSession.objects.get_or_create(
        interview_session=interview_session
    )
    if created:
        logger.info("Created proctoring session for interview %s", interview_session.id)
    return proctoring_session


def handle_incoming_flag(
    interview_session: InterviewSession,
    flag_type: str,
    metadata: dict | None = None,
) -> ProctoringActionResult:
    """
    Main entry point for the proctoring consumer. Warn/terminate
    authority stays entirely server-side — the client only reports raw
    signals.
    """
    if not interview_session.is_active:
        logger.info(
            "Ignoring proctoring flag for session %s — not in_progress (status=%s)",
            interview_session.id,
            interview_session.status,
        )
        return ProctoringActionResult(action=ProctoringActionResult.ACTION_NONE)

    proctoring_session = get_or_create_proctoring_session(interview_session)

    if proctoring_session.terminated:
        return ProctoringActionResult(action=ProctoringActionResult.ACTION_NONE)

    result: FlagProcessingResult = process_flag(
        proctoring_session=proctoring_session,
        flag_type=flag_type,
        metadata=metadata,
    )

    _update_risk_score(proctoring_session)

    if result.is_terminating:
        return _terminate(interview_session, proctoring_session, flag_type, result)

    if result.is_countable:
        return _warn(proctoring_session, flag_type, result)

    return ProctoringActionResult(action=ProctoringActionResult.ACTION_NONE)


def _warn(
    proctoring_session: ProctoringSession,
    flag_type: str,
    result: FlagProcessingResult,
) -> ProctoringActionResult:
    # Kept for backwards compatibility with anything reading these fields.
    proctoring_session.warning_issued = True
    proctoring_session.warning_issued_at = timezone.now()
    proctoring_session.save(update_fields=["warning_issued", "warning_issued_at", "updated_at"])

    count = result.type_count
    reason = reason_for(flag_type)

    if flag_type in WARN_ONLY_TYPES:
        # Notice only — can never end the interview, so no "N more times".
        allowed = 0
        message = (
            f"Notice: {reason}. Please answer in your own words while looking "
            f"at the screen. This has been noted for the recruiting team."
        )
    else:
        allowed = result.warnings_allowed
        remaining = max(allowed - count, 0)
        if remaining > 0:
            unit = "time" if remaining == 1 else "times"
            message = (
                f"Warning {count} of {allowed}: {reason}. Please stay focused on the "
                f"interview — if this happens {remaining} more {unit}, the interview "
                f"will end automatically."
            )
        else:
            message = (
                f"Final warning ({count} of {allowed}): {reason}. If this happens "
                f"again, the interview will end automatically."
            )

    logger.info(
        "Issuing proctoring %s %d/%d (%s) for session %s",
        "notice" if allowed == 0 else "warning", count, allowed, flag_type,
        proctoring_session.id,
    )

    return ProctoringActionResult(
        action=ProctoringActionResult.ACTION_WARN,
        message=message,
        flag_type=flag_type,
        count=count,
        limit=allowed,  # 0 means "notice only" — the UI uses this
    )


def _terminate(
    interview_session: InterviewSession,
    proctoring_session: ProctoringSession,
    flag_type: str,
    result: FlagProcessingResult,
) -> ProctoringActionResult:
    proctoring_session.terminated = True
    proctoring_session.terminated_at = timezone.now()
    proctoring_session.save(update_fields=["terminated", "terminated_at", "updated_at"])

    logger.info(
        "Terminating interview %s — %s exceeded its warning limit (count=%d, total=%d)",
        interview_session.id, flag_type, result.type_count,
        proctoring_session.countable_violation_count,
    )

    message = (
        "This interview has ended. We repeatedly detected the following, even "
        f"after several warnings: {reason_for(flag_type)}. The session was "
        "closed automatically to keep the process fair for everyone. Your "
        "responses so far have been saved, and you do not need to take any "
        "further action — the recruiting team will follow up separately."
    )

    candidate_profile = resolve_candidate_profile(interview_session.candidate)
    fsm = InterviewFSM(session=interview_session, candidate_profile=candidate_profile)
    interview_turn = fsm.terminate_for_violation(explanation=message)

    return ProctoringActionResult(
        action=ProctoringActionResult.ACTION_TERMINATE,
        message=message,
        interview_turn=interview_turn,
        flag_type=flag_type,
        count=result.type_count,
        limit=result.warnings_allowed,
    )


def _update_risk_score(proctoring_session: ProctoringSession) -> None:
    raw_total = 0
    for flag in proctoring_session.flags.all():
        if flag.counted_toward_violation:
            raw_total += SEVERITY_WEIGHTS.get(flag.severity, 0)
        else:
            raw_total += UNCOUNTED_FLAG_WEIGHT
    proctoring_session.risk_score = min(raw_total, RISK_SCORE_CAP)
    proctoring_session.save(update_fields=["risk_score", "updated_at"])
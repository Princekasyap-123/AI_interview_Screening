# apps/proctoring/services/risk_scorer.py

import logging

from django.utils import timezone

from apps.interviews.models import InterviewSession
from apps.interviews.services.interview_state import InterviewFSM
from apps.candidates.services import resolve_candidate_profile
from apps.proctoring.models import ProctoringFlag, ProctoringSession
from apps.proctoring.services.flag_processor import FlagProcessingResult, process_flag

logger = logging.getLogger(__name__)

# Weights used to compute the informational 0-100 risk_score shown on
# the recruiter dashboard. This is separate from the binary warn/
# terminate logic (which uses countable_violation_count, not this score)
# — risk_score exists so a recruiter reviewing a COMPLETED interview can
# see "lots of low-level flags" even if it never hit 2 countable strikes.
SEVERITY_WEIGHTS = {
    ProctoringFlag.Severity.LOW: 3,
    ProctoringFlag.Severity.MEDIUM: 8,
    ProctoringFlag.Severity.HIGH: 20,
}
RISK_SCORE_CAP = 100.0

WARNING_MESSAGE_TEMPLATE = (
    "We noticed you left the interview screen. For fairness to all "
    "candidates, this interview must stay in focus and fullscreen for "
    "its full duration. This is your only warning — if it happens "
    "again, the interview will end automatically."
)

TERMINATION_MESSAGE_TEMPLATE = (
    "This interview has ended. We detected repeated activity outside "
    "the interview screen after a warning was given, so the session "
    "was closed automatically to keep the process fair for everyone. "
    "This has been noted along with your responses so far. You do not "
    "need to take any further action — the recruiting team will follow "
    "up separately."
)


class ProctoringActionResult:
    """
    What the proctoring consumer should do next, returned by
    handle_incoming_flag(). Mirrors the shape of InterviewFSM turn dicts
    so the consumer can relay it consistently.
    """

    ACTION_NONE = "none"
    ACTION_WARN = "warn"
    ACTION_TERMINATE = "terminate"

    def __init__(self, action: str, message: str = "", interview_turn: dict | None = None):
        self.action = action
        self.message = message
        # Populated only on ACTION_TERMINATE — the InterviewFSM's
        # CLOSE_INTERVIEW turn dict, so the proctoring consumer can relay
        # it into the interview group without importing FSM internals.
        self.interview_turn = interview_turn


def get_or_create_proctoring_session(interview_session: InterviewSession) -> ProctoringSession:
    proctoring_session, created = ProctoringSession.objects.get_or_create(
        interview_session=interview_session
    )
    if created:
        logger.info(
            "Created proctoring session for interview %s", interview_session.id
        )
    return proctoring_session


def handle_incoming_flag(
    interview_session: InterviewSession,
    flag_type: str,
    metadata: dict | None = None,
) -> ProctoringActionResult:
    """
    Main entry point for the proctoring consumer. Takes a raw client-
    reported signal, processes it, updates the aggregate risk score, and
    — if it crosses the countable threshold — decides whether this is
    strike 1 (warn) or strike 2 (terminate).

    This is the ONLY place that should call InterviewFSM.terminate_for_violation()
    on a proctoring basis — keeps the warn/terminate authority server-side
    and in one place, rather than trusting the client to self-report
    "this is my second violation" (the risk flagged back when consumers.py
    was first written).
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
        # Already terminated by an earlier flag in this same burst —
        # avoid double-terminating or re-sending messages.
        return ProctoringActionResult(action=ProctoringActionResult.ACTION_NONE)

    result: FlagProcessingResult = process_flag(
        proctoring_session=proctoring_session,
        flag_type=flag_type,
        metadata=metadata,
    )

    _update_risk_score(proctoring_session)

    if result.is_second_strike:
        return _terminate(interview_session, proctoring_session)

    if result.is_first_strike:
        return _warn(proctoring_session)

    return ProctoringActionResult(action=ProctoringActionResult.ACTION_NONE)


def _warn(proctoring_session: ProctoringSession) -> ProctoringActionResult:
    proctoring_session.warning_issued = True
    proctoring_session.warning_issued_at = timezone.now()
    proctoring_session.save(update_fields=["warning_issued", "warning_issued_at", "updated_at"])

    logger.info(
        "Issuing proctoring warning for session %s", proctoring_session.id
    )

    return ProctoringActionResult(
        action=ProctoringActionResult.ACTION_WARN,
        message=WARNING_MESSAGE_TEMPLATE,
    )


def _terminate(
    interview_session: InterviewSession, proctoring_session: ProctoringSession
) -> ProctoringActionResult:
    proctoring_session.terminated = True
    proctoring_session.terminated_at = timezone.now()
    proctoring_session.save(update_fields=["terminated", "terminated_at", "updated_at"])

    logger.info(
        "Terminating interview %s for repeated proctoring violation",
        interview_session.id,
    )

    candidate_profile = resolve_candidate_profile(interview_session.candidate)
    fsm = InterviewFSM(session=interview_session, candidate_profile=candidate_profile)
    interview_turn = fsm.terminate_for_violation(explanation=TERMINATION_MESSAGE_TEMPLATE)

    return ProctoringActionResult(
        action=ProctoringActionResult.ACTION_TERMINATE,
        message=TERMINATION_MESSAGE_TEMPLATE,
        interview_turn=interview_turn,
    )


def _update_risk_score(proctoring_session: ProctoringSession) -> None:
    """
    Recomputes the informational 0-100 risk_score from ALL flags logged
    so far (countable and non-countable both contribute, at different
    weights) — this is purely for the recruiter dashboard and plays no
    role in the warn/terminate decision above.
    """
    flags = proctoring_session.flags.all()
    raw_total = sum(SEVERITY_WEIGHTS.get(f.severity, 0) for f in flags)
    proctoring_session.risk_score = min(raw_total, RISK_SCORE_CAP)
    proctoring_session.save(update_fields=["risk_score", "updated_at"])
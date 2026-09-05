# apps/proctoring/services/flag_processor.py

import logging

from django.utils import timezone

from apps.proctoring.models import ProctoringFlag, ProctoringSession

logger = logging.getLogger(__name__)

MIN_TAB_SWITCH_DURATION_MS = 1500
MIN_WINDOW_BLUR_DURATION_MS = 2000
MIN_NO_FACE_DURATION_MS = 4000
MIN_GAZE_AWAY_DURATION_MS = 5000
# Identity mismatch needs a sustained signal, not a single misread
# frame during a head turn or lighting change — same philosophy as
# the other duration gates above.
MIN_FACE_MISMATCH_DURATION_MS = 4000

ALWAYS_COUNTABLE_TYPES = {
    ProctoringFlag.FlagType.FULLSCREEN_EXIT,
    ProctoringFlag.FlagType.MULTIPLE_FACES,
    ProctoringFlag.FlagType.DEVTOOLS_OPENED,
    ProctoringFlag.FlagType.COPY_PASTE,
}

DURATION_GATED_TYPES = {
    ProctoringFlag.FlagType.TAB_SWITCH: MIN_TAB_SWITCH_DURATION_MS,
    ProctoringFlag.FlagType.WINDOW_BLUR: MIN_WINDOW_BLUR_DURATION_MS,
    ProctoringFlag.FlagType.NO_FACE: MIN_NO_FACE_DURATION_MS,
    ProctoringFlag.FlagType.GAZE_AWAY: MIN_GAZE_AWAY_DURATION_MS,
    ProctoringFlag.FlagType.FACE_MISMATCH: MIN_FACE_MISMATCH_DURATION_MS,
}

SEVERITY_MAP = {
    ProctoringFlag.FlagType.TAB_SWITCH: ProctoringFlag.Severity.HIGH,
    ProctoringFlag.FlagType.WINDOW_BLUR: ProctoringFlag.Severity.MEDIUM,
    ProctoringFlag.FlagType.FULLSCREEN_EXIT: ProctoringFlag.Severity.HIGH,
    ProctoringFlag.FlagType.NO_FACE: ProctoringFlag.Severity.MEDIUM,
    ProctoringFlag.FlagType.MULTIPLE_FACES: ProctoringFlag.Severity.HIGH,
    ProctoringFlag.FlagType.GAZE_AWAY: ProctoringFlag.Severity.LOW,
    ProctoringFlag.FlagType.COPY_PASTE: ProctoringFlag.Severity.MEDIUM,
    ProctoringFlag.FlagType.DEVTOOLS_OPENED: ProctoringFlag.Severity.HIGH,
    # HIGH — an identity mismatch (someone else answering) is one of
    # the most serious signals this system can raise.
    ProctoringFlag.FlagType.FACE_MISMATCH: ProctoringFlag.Severity.HIGH,
}


class FlagProcessingResult:
    def __init__(self, flag: ProctoringFlag, is_countable: bool, is_first_strike: bool, is_second_strike: bool):
        self.flag = flag
        self.is_countable = is_countable
        self.is_first_strike = is_first_strike
        self.is_second_strike = is_second_strike


def process_flag(
    proctoring_session: ProctoringSession,
    flag_type: str,
    metadata: dict | None = None,
    occurred_at=None,
) -> FlagProcessingResult:
    metadata = metadata or {}
    occurred_at = occurred_at or timezone.now()

    if flag_type not in ProctoringFlag.FlagType.values:
        raise ValueError(f"Unknown flag_type: {flag_type}")

    is_countable = _is_countable(flag_type, metadata)
    severity = SEVERITY_MAP.get(flag_type, ProctoringFlag.Severity.LOW)

    flag = ProctoringFlag.objects.create(
        proctoring_session=proctoring_session,
        flag_type=flag_type,
        severity=severity,
        counted_toward_violation=is_countable,
        metadata=metadata,
        occurred_at=occurred_at,
    )

    is_first_strike = False
    is_second_strike = False

    if is_countable:
        proctoring_session.countable_violation_count += 1
        proctoring_session.save(update_fields=["countable_violation_count", "updated_at"])

        if proctoring_session.countable_violation_count == 1:
            is_first_strike = True
        elif proctoring_session.countable_violation_count >= 2:
            is_second_strike = True

    logger.info(
        "Processed flag %s (type=%s, countable=%s, count=%d) for proctoring session %s",
        flag.id,
        flag_type,
        is_countable,
        proctoring_session.countable_violation_count,
        proctoring_session.id,
    )

    return FlagProcessingResult(
        flag=flag,
        is_countable=is_countable,
        is_first_strike=is_first_strike,
        is_second_strike=is_second_strike,
    )


def _is_countable(flag_type: str, metadata: dict) -> bool:
    if flag_type in ALWAYS_COUNTABLE_TYPES:
        return True

    if flag_type in DURATION_GATED_TYPES:
        duration_ms = metadata.get("duration_ms")
        if duration_ms is None:
            logger.warning(
                "Flag type %s is duration-gated but no duration_ms was "
                "provided — counting it by default.",
                flag_type,
            )
            return True
        return duration_ms >= DURATION_GATED_TYPES[flag_type]

    return True
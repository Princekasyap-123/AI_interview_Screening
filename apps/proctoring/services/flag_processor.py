# apps/proctoring/services/flag_processor.py

import logging

from django.utils import timezone

from apps.proctoring.models import ProctoringFlag, ProctoringSession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Countability rules
# ---------------------------------------------------------------------
# Not every raw signal should burn one of the candidate's two strikes.
# A flag counts toward termination only if it crosses a minimum
# duration/confidence threshold — short blips are logged for the audit
# trail but don't trigger a warning on their own. Tune these thresholds
# based on real candidate behavior once you have data.

MIN_TAB_SWITCH_DURATION_MS = 1500       # ignore instant/accidental blur
MIN_WINDOW_BLUR_DURATION_MS = 2000
MIN_NO_FACE_DURATION_MS = 4000          # brief head-turn shouldn't count
MIN_GAZE_AWAY_DURATION_MS = 5000        # gaze is noisy, needs a longer bar

# Flag types that are always countable regardless of duration, because
# there's no legitimate accidental version of them.
ALWAYS_COUNTABLE_TYPES = {
    ProctoringFlag.FlagType.FULLSCREEN_EXIT,
    ProctoringFlag.FlagType.MULTIPLE_FACES,
    ProctoringFlag.FlagType.DEVTOOLS_OPENED,
    ProctoringFlag.FlagType.COPY_PASTE,
}

# Duration-gated flag types: (flag_type -> minimum duration_ms to count)
DURATION_GATED_TYPES = {
    ProctoringFlag.FlagType.TAB_SWITCH: MIN_TAB_SWITCH_DURATION_MS,
    ProctoringFlag.FlagType.WINDOW_BLUR: MIN_WINDOW_BLUR_DURATION_MS,
    ProctoringFlag.FlagType.NO_FACE: MIN_NO_FACE_DURATION_MS,
    ProctoringFlag.FlagType.GAZE_AWAY: MIN_GAZE_AWAY_DURATION_MS,
}

# Base severity per flag type — used by risk_scorer.py for the
# informational 0-100 risk score, independent of the countable/warn logic.
SEVERITY_MAP = {
    ProctoringFlag.FlagType.TAB_SWITCH: ProctoringFlag.Severity.HIGH,
    ProctoringFlag.FlagType.WINDOW_BLUR: ProctoringFlag.Severity.MEDIUM,
    ProctoringFlag.FlagType.FULLSCREEN_EXIT: ProctoringFlag.Severity.HIGH,
    ProctoringFlag.FlagType.NO_FACE: ProctoringFlag.Severity.MEDIUM,
    ProctoringFlag.FlagType.MULTIPLE_FACES: ProctoringFlag.Severity.HIGH,
    ProctoringFlag.FlagType.GAZE_AWAY: ProctoringFlag.Severity.LOW,
    ProctoringFlag.FlagType.COPY_PASTE: ProctoringFlag.Severity.MEDIUM,
    ProctoringFlag.FlagType.DEVTOOLS_OPENED: ProctoringFlag.Severity.HIGH,
}


class FlagProcessingResult:
    """
    Return value of process_flag(). Consumed by the proctoring consumer
    to decide what to tell the candidate/interview consumer next.
    """

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
    """
    Records a raw proctoring signal and decides whether it's countable
    toward the warn/terminate threshold.

    This function does NOT decide whether to actually warn or terminate
    — it only determines countability and updates the running count.
    risk_scorer.py (next file) is what the consumer calls to translate
    "is this now strike 1 or strike 2" into an actual action.

    metadata: flag-type-specific context, e.g. {"duration_ms": 3200}
        for duration-gated types, {"face_count": 2} for multiple_faces.
    """
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
    """
    Determines whether a raw flag crosses the threshold to count as a
    real violation, as opposed to noise (brief blur, momentary head turn).
    """
    if flag_type in ALWAYS_COUNTABLE_TYPES:
        return True

    if flag_type in DURATION_GATED_TYPES:
        duration_ms = metadata.get("duration_ms")
        if duration_ms is None:
            # No duration reported — be conservative and count it, since
            # missing data shouldn't let a real violation slip through
            # uncounted. Frontend should always send duration_ms for
            # these types; log a warning so it gets fixed.
            logger.warning(
                "Flag type %s is duration-gated but no duration_ms was "
                "provided — counting it by default.",
                flag_type,
            )
            return True
        return duration_ms >= DURATION_GATED_TYPES[flag_type]

    # Unrecognized-but-valid flag type with no explicit rule — default
    # to countable rather than silently ignoring a real signal.
    return True
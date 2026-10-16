# apps/proctoring/services/flag_processor.py

import logging

from django.utils import timezone

from apps.proctoring.models import ProctoringFlag, ProctoringSession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Tunable limits — edit these to make proctoring stricter or looser.
# ---------------------------------------------------------------------

# Per violation TYPE: counted violations 1..N are warnings, violation
# N+1 of the same type ends the interview (4 -> the 5th ends it).
WARNINGS_BEFORE_TERMINATION = 4

# Per-type overrides. Identity and phone are stricter than the default.
WARNING_LIMIT_OVERRIDES = {
    ProctoringFlag.FlagType.FACE_MISMATCH: 3,   # the 4th confirmed mismatch ends it
    ProctoringFlag.FlagType.PHONE_DETECTED: 3,  # the 4th confirmed phone sighting ends it
}

# These types show the candidate a notice and are logged for the
# recruiter, but can NEVER end the interview on their own. "Reading
# suspected" is a head-direction heuristic, not proof.
WARN_ONLY_TYPES = {
    ProctoringFlag.FlagType.READING_SUSPECTED,
}

# Safety valve across ALL strike-eligible types.
MAX_TOTAL_COUNTED_VIOLATIONS = 12

# A counted flag within this many seconds of ANY previous counted flag
# (warn-only notices excluded) is logged but not counted.
BURST_COOLDOWN_SECONDS = 5

# Object sightings are independent evidence (a phone in frame is not a
# side-effect of another event), so they are NOT dropped by the burst
# cooldown above — e.g. a phone raised right after a no_face flag still
# counts. The same-type cooldown below still applies to them.
BURST_COOLDOWN_EXEMPT_TYPES = {
    ProctoringFlag.FlagType.PHONE_DETECTED,
    ProctoringFlag.FlagType.NOTES_DETECTED,
}

# A counted flag within this many seconds of the previous counted flag
# of the SAME type is logged but not counted.
SAME_TYPE_COOLDOWN_SECONDS = 10

# Object sightings re-arrive every ~3 s while the object stays in view
# (OBJECT_REPORT_COOLDOWN_MS in proctoring_client.js). A shorter same-type
# cooldown for them means a phone kept in view escalates quickly:
# with a limit of 3 it ends the interview roughly 10 s after first sighting.
OBJECT_SAME_TYPE_COOLDOWN_SECONDS = 3
OBJECT_FLAG_TYPES = {
    ProctoringFlag.FlagType.PHONE_DETECTED,
    ProctoringFlag.FlagType.NOTES_DETECTED,
}

# Minimum sustained duration before a duration-gated flag counts.
MIN_TAB_SWITCH_DURATION_MS = 3000
MIN_WINDOW_BLUR_DURATION_MS = 4000
MIN_NO_FACE_DURATION_MS = 8000
MIN_MULTIPLE_FACES_DURATION_MS = 4000

# Never counts toward strikes (rough heuristic) — logged only.
NEVER_COUNTED_TYPES = {
    ProctoringFlag.FlagType.GAZE_AWAY,
}

# Counted as soon as they arrive (subject to the cooldowns above).
# The browser already confirms these over several checks before sending,
# so no extra duration gate is needed here.
ALWAYS_COUNTABLE_TYPES = {
    ProctoringFlag.FlagType.FULLSCREEN_EXIT,
    ProctoringFlag.FlagType.DEVTOOLS_OPENED,
    ProctoringFlag.FlagType.COPY_PASTE,
    ProctoringFlag.FlagType.FACE_MISMATCH,
    ProctoringFlag.FlagType.PHONE_DETECTED,
    ProctoringFlag.FlagType.NOTES_DETECTED,
    ProctoringFlag.FlagType.READING_SUSPECTED,
}

DURATION_GATED_TYPES = {
    ProctoringFlag.FlagType.TAB_SWITCH: MIN_TAB_SWITCH_DURATION_MS,
    ProctoringFlag.FlagType.WINDOW_BLUR: MIN_WINDOW_BLUR_DURATION_MS,
    ProctoringFlag.FlagType.NO_FACE: MIN_NO_FACE_DURATION_MS,
    ProctoringFlag.FlagType.MULTIPLE_FACES: MIN_MULTIPLE_FACES_DURATION_MS,
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
    ProctoringFlag.FlagType.FACE_MISMATCH: ProctoringFlag.Severity.HIGH,
    ProctoringFlag.FlagType.PHONE_DETECTED: ProctoringFlag.Severity.HIGH,
    ProctoringFlag.FlagType.NOTES_DETECTED: ProctoringFlag.Severity.MEDIUM,
    ProctoringFlag.FlagType.READING_SUSPECTED: ProctoringFlag.Severity.MEDIUM,
}

# Candidate-facing wording used in warnings and the termination message.
FLAG_REASONS = {
    ProctoringFlag.FlagType.TAB_SWITCH: "You switched away from the interview tab",
    ProctoringFlag.FlagType.WINDOW_BLUR: "The interview window lost focus",
    ProctoringFlag.FlagType.FULLSCREEN_EXIT: "You exited fullscreen",
    ProctoringFlag.FlagType.NO_FACE: "Your face was not visible to the camera",
    ProctoringFlag.FlagType.MULTIPLE_FACES: "More than one person was visible on camera",
    ProctoringFlag.FlagType.COPY_PASTE: "Copy/paste was detected",
    ProctoringFlag.FlagType.DEVTOOLS_OPENED: "Browser developer tools appear to be open",
    ProctoringFlag.FlagType.FACE_MISMATCH: (
        "The person on camera did not match the photo taken at the start of the interview"
    ),
    ProctoringFlag.FlagType.PHONE_DETECTED: (
        "A mobile phone or another electronic device was visible on camera"
    ),
    ProctoringFlag.FlagType.NOTES_DETECTED: "Notes, a book or papers were visible on camera",
    ProctoringFlag.FlagType.READING_SUSPECTED: (
        "You appear to be reading from something outside the screen while answering"
    ),
    ProctoringFlag.FlagType.GAZE_AWAY: "Your gaze moved away from the screen",
}


def warnings_allowed_for(flag_type: str) -> int:
    return WARNING_LIMIT_OVERRIDES.get(flag_type, WARNINGS_BEFORE_TERMINATION)


def reason_for(flag_type: str) -> str:
    return FLAG_REASONS.get(flag_type, "Unusual activity was detected")


class FlagProcessingResult:
    def __init__(
        self,
        flag: ProctoringFlag,
        is_countable: bool,
        type_count: int,
        warnings_allowed: int,
        is_terminating: bool,
        is_warn_only: bool = False,
        not_counted_reason: str = "",
    ):
        self.flag = flag
        self.is_countable = is_countable
        self.type_count = type_count
        self.warnings_allowed = warnings_allowed
        self.is_terminating = is_terminating
        self.is_warn_only = is_warn_only
        self.not_counted_reason = not_counted_reason


def process_flag(
    proctoring_session: ProctoringSession,
    flag_type: str,
    metadata: dict | None = None,
    occurred_at=None,
) -> FlagProcessingResult:
    """
    Records a raw proctoring signal and decides whether it counts as a
    strike. Every signal is stored; only some count. Per-type counts are
    derived from stored ProctoringFlag rows. Call this while holding the
    ProctoringSession row lock (the consumer does) so concurrent flags
    can't race on the count.
    """
    metadata = dict(metadata or {})
    occurred_at = occurred_at or timezone.now()

    if flag_type not in ProctoringFlag.FlagType.values:
        raise ValueError(f"Unknown flag_type: {flag_type}")

    meets_threshold = _meets_threshold(flag_type, metadata)

    not_counted_reason = ""
    if meets_threshold:
        not_counted_reason = _cooldown_reason(proctoring_session, flag_type, occurred_at)

    is_countable = meets_threshold and not not_counted_reason

    if not_counted_reason:
        metadata["not_counted_reason"] = not_counted_reason

    flag = ProctoringFlag.objects.create(
        proctoring_session=proctoring_session,
        flag_type=flag_type,
        severity=SEVERITY_MAP.get(flag_type, ProctoringFlag.Severity.LOW),
        counted_toward_violation=is_countable,
        metadata=metadata,
        occurred_at=occurred_at,
    )

    is_warn_only = flag_type in WARN_ONLY_TYPES
    allowed = warnings_allowed_for(flag_type)
    type_count = 0
    is_terminating = False

    if is_countable:
        proctoring_session.countable_violation_count += 1
        proctoring_session.save(update_fields=["countable_violation_count", "updated_at"])

        type_count = proctoring_session.flags.filter(
            flag_type=flag_type, counted_toward_violation=True
        ).count()

        if not is_warn_only:
            strike_total = (
                proctoring_session.flags.filter(counted_toward_violation=True)
                .exclude(flag_type__in=WARN_ONLY_TYPES)
                .count()
            )
            is_terminating = (
                type_count > allowed or strike_total > MAX_TOTAL_COUNTED_VIOLATIONS
            )

    logger.info(
        "Processed flag %s (type=%s, counted=%s, warn_only=%s, type_count=%d/%d, "
        "not_counted_reason=%s) for proctoring session %s",
        flag.id, flag_type, is_countable, is_warn_only, type_count, allowed,
        not_counted_reason or "-", proctoring_session.id,
    )

    return FlagProcessingResult(
        flag=flag,
        is_countable=is_countable,
        type_count=type_count,
        warnings_allowed=allowed,
        is_terminating=is_terminating,
        is_warn_only=is_warn_only,
        not_counted_reason=not_counted_reason,
    )


def _meets_threshold(flag_type: str, metadata: dict) -> bool:
    """Would this signal count, ignoring cooldowns?"""
    if flag_type in NEVER_COUNTED_TYPES:
        return False

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


def _cooldown_reason(proctoring_session: ProctoringSession, flag_type: str, now) -> str:
    """
    Returns "" if this flag may count, otherwise why it must not:
    "burst_cooldown" (right after another counted flag) or
    "repeat_cooldown" (right after a counted flag of the same type).
    Warn-only notices don't start a burst cooldown, so a "reading"
    notice can never hide a real strike that follows it.
    """
    counted = proctoring_session.flags.filter(counted_toward_violation=True)

    if flag_type not in BURST_COOLDOWN_EXEMPT_TYPES:
        last_any = (
            counted.exclude(flag_type__in=WARN_ONLY_TYPES)
            .order_by("-occurred_at")
            .values_list("occurred_at", flat=True)
            .first()
        )
        if last_any is not None and (now - last_any).total_seconds() < BURST_COOLDOWN_SECONDS:
            return "burst_cooldown"

    last_same = (
        counted.filter(flag_type=flag_type)
        .order_by("-occurred_at")
        .values_list("occurred_at", flat=True)
        .first()
    )
    same_type_cooldown = (
        OBJECT_SAME_TYPE_COOLDOWN_SECONDS
        if flag_type in OBJECT_FLAG_TYPES
        else SAME_TYPE_COOLDOWN_SECONDS
    )
    if last_same is not None and (now - last_same).total_seconds() < same_type_cooldown:
        return "repeat_cooldown"

    return ""
# apps/proctoring/models.py

import uuid

from django.db import models


class ProctoringSession(models.Model):
    """
    One proctoring record per InterviewSession — tracks cumulative risk
    and whether a warning has already been issued, since that's what
    decides whether the NEXT violation triggers a warning or a
    termination (per the agreed 1st-warn / 2nd-terminate flow).
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    interview_session = models.OneToOneField(
        "interviews.InterviewSession",
        on_delete=models.CASCADE,
        related_name="proctoring_session",
    )

    # Running count of violations that count toward the warn/terminate
    # threshold — NOT every flag (e.g. a single momentary "no_face" blip
    # might be too minor to count; see flag_processor.py for what
    # actually increments this vs. what's just logged).
    countable_violation_count = models.PositiveSmallIntegerField(default=0)

    warning_issued = models.BooleanField(default=False)
    warning_issued_at = models.DateTimeField(null=True, blank=True)

    terminated = models.BooleanField(default=False)
    terminated_at = models.DateTimeField(null=True, blank=True)

    # 0-100 aggregate risk score, computed by risk_scorer.py from all
    # flags (including non-countable ones) — informational for the
    # recruiter dashboard, distinct from the binary warn/terminate logic.
    risk_score = models.FloatField(default=0.0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Proctoring session"
        verbose_name_plural = "Proctoring sessions"

    def __str__(self):
        return f"Proctoring for {self.interview_session_id} (risk={self.risk_score})"


class ProctoringFlag(models.Model):
    """
    A single raw signal from the candidate's browser: tab switch, window
    blur, no face detected, multiple faces, gaze away, fullscreen exit,
    etc. Every signal is logged here regardless of severity — flag_processor.py
    decides which ones are "countable" toward termination vs. just noise.
    """

    class FlagType(models.TextChoices):
        TAB_SWITCH = "tab_switch", "Tab switched away"
        WINDOW_BLUR = "window_blur", "Window lost focus"
        FULLSCREEN_EXIT = "fullscreen_exit", "Exited fullscreen"
        NO_FACE = "no_face", "No face detected"
        MULTIPLE_FACES = "multiple_faces", "Multiple faces detected"
        GAZE_AWAY = "gaze_away", "Gaze directed away from screen"
        COPY_PASTE = "copy_paste", "Copy/paste attempted"
        DEVTOOLS_OPENED = "devtools_opened", "Developer tools opened"

    class Severity(models.TextChoices):
        LOW = "low", "Low"
        MEDIUM = "medium", "Medium"
        HIGH = "high", "High"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    proctoring_session = models.ForeignKey(
        ProctoringSession, on_delete=models.CASCADE, related_name="flags"
    )

    flag_type = models.CharField(max_length=30, choices=FlagType.choices)
    severity = models.CharField(max_length=10, choices=Severity.choices, default=Severity.LOW)

    # Whether this specific flag counted toward countable_violation_count
    # on the parent session — set by flag_processor.py at creation time,
    # kept here for audit/debugging so you can see why a candidate was
    # or wasn't warned for a given event.
    counted_toward_violation = models.BooleanField(default=False)

    # Raw client-reported context, e.g. {"duration_ms": 4200, "url_hint": "..."}
    # for tab switches, or {"face_count": 2} for multiple_faces. Free-form
    # on purpose since different flag types carry different metadata.
    metadata = models.JSONField(default=dict, blank=True)

    occurred_at = models.DateTimeField()
    logged_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["occurred_at"]
        indexes = [
            models.Index(fields=["proctoring_session", "flag_type"]),
        ]

    def __str__(self):
        return f"{self.flag_type} @ {self.occurred_at} (session {self.proctoring_session_id})"
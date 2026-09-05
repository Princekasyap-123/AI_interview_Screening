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

    countable_violation_count = models.PositiveSmallIntegerField(default=0)

    warning_issued = models.BooleanField(default=False)
    warning_issued_at = models.DateTimeField(null=True, blank=True)

    terminated = models.BooleanField(default=False)
    terminated_at = models.DateTimeField(null=True, blank=True)

    risk_score = models.FloatField(default=0.0)

    # Snapshot captured client-side ~3s after the candidate's camera
    # starts, before questions begin. Used for two purposes:
    #   1. Recruiter-facing audit trail — "this is who showed up".
    #   2. NOT used server-side for live matching — the live face
    #      descriptor comparison happens entirely client-side in
    #      proctoring_client.js against an in-memory reference, and
    #      only the resulting face_mismatch flag (with duration) is
    #      ever sent here. This field is purely the stored photo.
    reference_photo = models.ImageField(
        upload_to="proctoring_reference_photos/%Y/%m/", null=True, blank=True
    )
    reference_captured_at = models.DateTimeField(null=True, blank=True)

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
    face mismatch, etc. Every signal is logged here regardless of
    severity — flag_processor.py decides which ones are "countable"
    toward termination vs. just noise.
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
        FACE_MISMATCH = "face_mismatch", "Face does not match reference photo"

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

    counted_toward_violation = models.BooleanField(default=False)

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
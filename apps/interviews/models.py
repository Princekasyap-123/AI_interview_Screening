# apps/interviews/models.py

import uuid

from django.conf import settings
from django.db import models


class InterviewSession(models.Model):
    """
    One end-to-end interview attempt for a candidate.
    Tracks lifecycle state, question budget, and how/why it ended.
    """

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        IN_PROGRESS = "in_progress", "In progress"
        COMPLETED = "completed", "Completed"
        TERMINATED = "terminated", "Terminated early"

    class TerminationReason(models.TextChoices):
        NONE = "none", "Not terminated"
        PROCTORING_VIOLATION = "proctoring_violation", "Repeated proctoring violation"
        CANDIDATE_LEFT = "candidate_left", "Candidate disconnected"
        CANDIDATE_ENDED = "candidate_ended", "Candidate ended interview early"
        TECHNICAL_ERROR = "technical_error", "Technical error"
        ABUSIVE_LANGUAGE = "abusive_language", "Abusive or offensive language"
        RESCHEDULED = "rescheduled", "Candidate asked to reschedule"

    class JobRoleType(models.TextChoices):
        BLUE_COLLAR = "blue_collar", "Blue-collar"
        WHITE_COLLAR = "white_collar", "White-collar"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    candidate = models.ForeignKey(
        "candidates.Candidate",
        on_delete=models.CASCADE,
        related_name="interview_sessions",
    )

    job_description_text = models.TextField(blank=True)

    job_role_type = models.CharField(
        max_length=20,
        choices=JobRoleType.choices,
        default=JobRoleType.WHITE_COLLAR,
    )

    class LanguageMode(models.TextChoices):
        ENGLISH = "en", "English"
        HINDI = "hi", "Hindi"
        BOTH = "both", "English + Hindi"

    # Which language(s) the interview may be held in. "both" starts in English and
    # the interviewer switches to Hindi when the candidate speaks Hindi.
    language_mode = models.CharField(
        max_length=4, choices=LanguageMode.choices, default=LanguageMode.ENGLISH
    )
    # The language the interviewer is speaking right now ("en" / "hi").
    current_language = models.CharField(max_length=2, default="en")

    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.PENDING
    )

    max_questions = models.PositiveSmallIntegerField(default=8)
    questions_asked_count = models.PositiveSmallIntegerField(default=0)

    termination_reason = models.CharField(
        max_length=30,
        choices=TerminationReason.choices,
        default=TerminationReason.NONE,
    )

    termination_note = models.TextField(blank=True)
    # Set when the candidate was not ready and agreed a new slot (see services/scheduling.py).
    rescheduled_for = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    ended_at = models.DateTimeField(null=True, blank=True)

    access_token = models.CharField(
        max_length=64, unique=True, editable=False, blank=True
    )

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["status"]),
            models.Index(fields=["candidate", "status"]),
        ]

    def save(self, *args, **kwargs):
        if not self.access_token:
            self.access_token = uuid.uuid4().hex
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Interview {self.id} — {self.candidate} ({self.status})"

    @property
    def is_active(self):
        return self.status == self.Status.IN_PROGRESS

    @property
    def has_reached_question_limit(self):
        return self.questions_asked_count >= self.max_questions

    @property
    def is_white_collar(self):
        return self.job_role_type == self.JobRoleType.WHITE_COLLAR


class Question(models.Model):
    """
    A single AI-generated interview question, tied to the session and
    (optionally) the profile skill/topic it was generated from.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    session = models.ForeignKey(
        InterviewSession, on_delete=models.CASCADE, related_name="questions"
    )

    order = models.PositiveSmallIntegerField()
    text = models.TextField()

    topic_tag = models.CharField(max_length=100, blank=True)

    generation_context = models.JSONField(default=dict, blank=True)

    asked_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["session", "order"]
        unique_together = ("session", "order")

    def __str__(self):
        return f"Q{self.order} — {self.text[:60]}"

    @property
    def has_answer(self) -> bool:
        """
        Safe existence check for the reverse OneToOne `answer` accessor.
        Question.answer raises Question.answer.RelatedObjectDoesNotExist
        when unset — it does NOT behave like a queryset/manager. Use
        this instead of a bare `question.answer` check anywhere in
        interview_state.py or elsewhere.
        """
        return Answer.objects.filter(question_id=self.id).exists()


class Answer(models.Model):
    """
    Candidate's response to a Question: raw transcript plus backend-only
    analysis. Nothing on this model should ever reach a candidate-facing
    serializer — enforce that at the view/serializer layer.

    NOTE: this is a OneToOneField, so a Question can have at most ONE
    Answer row, ever. If a follow-up is asked and answered,
    analyze_answer() must update this same row rather than creating a
    second one — meaning the candidate's original first-answer
    transcript/analysis is overwritten by the follow-up answer once
    analyzed. Flagged as a separate open item — not addressed here.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    question = models.OneToOneField(
        Question, on_delete=models.CASCADE, related_name="answer"
    )

    transcript_text = models.TextField(blank=True)

    audio_ref = models.CharField(max_length=500, blank=True)

    analysis_json = models.JSONField(default=dict, blank=True)

    score = models.FloatField(null=True, blank=True)

    answered_at = models.DateTimeField(null=True, blank=True)
    analyzed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["question__order"]

    def __str__(self):
        return f"Answer to {self.question_id}"


class FinalRating(models.Model):
    """
    End-of-interview aggregate rating, generated by the Celery post-analysis
    task once a session completes or terminates. Recruiter-only, never
    exposed to the candidate.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    session = models.OneToOneField(
        InterviewSession, on_delete=models.CASCADE, related_name="final_rating"
    )

    overall_score = models.FloatField()

    category_scores = models.JSONField(default=dict, blank=True)

    strengths = models.TextField(blank=True)
    concerns = models.TextField(blank=True)

    content_flags = models.JSONField(default=list, blank=True)

    recruiter_visible = models.BooleanField(default=True)
    candidate_visible = models.BooleanField(default=False)

    generated_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Final rating"
        verbose_name_plural = "Final ratings"

    def __str__(self):
        return f"Rating for {self.session_id}: {self.overall_score}"


class ConversationTurn(models.Model):
    """
    One line of the interview, in order: something the interviewer said or something the
    candidate said. Kept so the recruiter sees the WHOLE conversation - in the language it
    was actually held in (English, Hindi or both) - not only the scored answers.
    """

    class Speaker(models.TextChoices):
        INTERVIEWER = "interviewer", "Interviewer"
        CANDIDATE = "candidate", "Candidate"

    session = models.ForeignKey(InterviewSession, on_delete=models.CASCADE, related_name="conversation")
    speaker = models.CharField(max_length=12, choices=Speaker.choices)
    # question / follow_up / reply / intro / closing  (interviewer)   answer / inaudible  (candidate)
    kind = models.CharField(max_length=20, blank=True)
    text = models.TextField(blank=True)             # exactly what was said / heard, in `language`
    text_en = models.TextField(blank=True)          # the English original of an interviewer line spoken in Hindi
    language = models.CharField(max_length=8, default="en")
    question = models.ForeignKey(Question, on_delete=models.SET_NULL, null=True, blank=True, related_name="+")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at", "id"]
        indexes = [models.Index(fields=["session", "created_at"])]

    def __str__(self):
        return f"{self.speaker}: {self.text[:40]}"

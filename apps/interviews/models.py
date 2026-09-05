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

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    candidate = models.ForeignKey(
        "candidates.Candidate",
        on_delete=models.CASCADE,
        related_name="interview_sessions",
    )

    job_description_text = models.TextField(blank=True)

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
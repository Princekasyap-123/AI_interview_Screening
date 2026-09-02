# apps/candidates/models.py

import uuid

from django.db import models


class Candidate(models.Model):
    """
    A person being screened. Created when a resume/profile is uploaded
    for interview, before any InterviewSession exists — one Candidate
    can have multiple InterviewSession rows over time (re-interviews,
    different roles, etc.) via the FK on InterviewSession.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=30, blank=True)

    resume_file = models.FileField(upload_to="resumes/%Y/%m/", blank=True, null=True)

    # Which role/JD this candidate is currently being screened for —
    # informational; a session can still override with its own
    # job_description_text (see InterviewSession) for one-off cases.
    applied_role = models.CharField(max_length=255, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["email"]),
        ]

    def __str__(self):
        return f"{self.name} <{self.email}>"


class ResumeProfile(models.Model):
    """
    Parsed/structured data extracted from a candidate's resume — this is
    what resolve_candidate_profile() (apps/candidates/services.py) reads
    from to build the dict passed into question generation, answer
    analysis, and final rating prompts.

    Field names here are the source of truth — if you change any of
    these, update resolve_candidate_profile() to match, since that
    function reads these exact attribute names via getattr().
    """

    class RoleLevel(models.TextChoices):
        INTERN = "intern", "Intern"
        JUNIOR = "junior", "Junior"
        MID = "mid", "Mid-level"
        SENIOR = "senior", "Senior"
        LEAD = "lead", "Lead / Staff"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    candidate = models.OneToOneField(
        Candidate, on_delete=models.CASCADE, related_name="resume_profile"
    )

    # List of skill strings, e.g. ["Django", "PostgreSQL", "Celery"].
    # Stored as JSON rather than a separate Skill model — simpler for
    # LLM prompt-building, and you're not querying "find all candidates
    # with skill X" at this stage. Revisit as a proper M2M if that
    # becomes a real product need later.
    skills = models.JSONField(default=list, blank=True)

    experience_years = models.PositiveSmallIntegerField(null=True, blank=True)

    role_level = models.CharField(
        max_length=20, choices=RoleLevel.choices, blank=True
    )

    # List of dicts: [{"title": "...", "description": "..."}, ...]
    # Matches the shape build_question_gen_user_prompt() expects.
    recent_projects = models.JSONField(default=list, blank=True)

    # Full raw text extracted from the resume file (via your existing
    # bulkresume parsing pipeline) — kept for reference/re-parsing,
    # not sent to the LLM directly since it's unstructured and noisy.
    raw_resume_text = models.TextField(blank=True)

    # Which parser/model produced this profile, and when — useful once
    # you're round-robining Groq keys or swapping extraction methods,
    # same pattern as your existing bulkresume app.
    parsed_at = models.DateTimeField(null=True, blank=True)
    parser_version = models.CharField(max_length=50, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Resume profile"
        verbose_name_plural = "Resume profiles"

    def __str__(self):
        return f"Profile for {self.candidate.name}"
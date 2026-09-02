# apps/candidates/serializers.py

from rest_framework import serializers

from apps.candidates.models import Candidate, ResumeProfile


class ResumeProfileSerializer(serializers.ModelSerializer):
    """
    Read-only-ish representation of a parsed profile. skills and
    recent_projects are JSONFields on the model — DRF handles them as
    plain list/dict pass-through automatically, no custom field needed.
    """

    class Meta:
        model = ResumeProfile
        fields = [
            "id",
            "skills",
            "experience_years",
            "role_level",
            "recent_projects",
            "raw_resume_text",
            "parsed_at",
            "parser_version",
        ]
        read_only_fields = fields  # profile is always written by the parser, not by clients directly


class CandidateSerializer(serializers.ModelSerializer):
    """
    Full candidate representation including nested profile, for the
    recruiter-facing "view candidate" endpoint.
    """
    resume_profile = ResumeProfileSerializer(read_only=True)

    class Meta:
        model = Candidate
        fields = [
            "id",
            "name",
            "email",
            "phone",
            "resume_file",
            "applied_role",
            "resume_profile",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]


class CandidateCreateSerializer(serializers.ModelSerializer):
    """
    Used for the intake endpoint: create a Candidate + upload a resume
    file in one request. Does NOT create ResumeProfile — parsing happens
    asynchronously (Celery task, not yet written) after upload, since
    your existing bulkresume app pattern is async/Celery-based and
    resume parsing shouldn't block the HTTP response.
    """

    class Meta:
        model = Candidate
        fields = ["name", "email", "phone", "resume_file", "applied_role"]

    def validate_resume_file(self, value):
        allowed_extensions = (".pdf", ".doc", ".docx", ".png", ".jpg", ".jpeg")
        if value and not value.name.lower().endswith(allowed_extensions):
            raise serializers.ValidationError(
                f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        return value
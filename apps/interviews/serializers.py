# apps/interviews/serializers.py

from rest_framework import serializers

from apps.interviews.models import Answer, FinalRating, InterviewSession, Question


class InterviewSessionCreateSerializer(serializers.Serializer):
    """
    Input shape for creating a session. Plain Serializer (not
    ModelSerializer) since candidate_id needs to resolve to a Candidate
    FK explicitly in the view, and we want tight control over exactly
    which fields a recruiter can set at creation time.
    """

    candidate_id = serializers.UUIDField()
    job_description_text = serializers.CharField(
        required=False, allow_blank=True, default=""
    )
    max_questions = serializers.IntegerField(
        required=False, min_value=1, max_value=30
    )


class AnswerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Answer
        fields = [
            "transcript_text",
            "analysis_json",
            "score",
            "answered_at",
            "analyzed_at",
        ]


class QuestionWithAnswerSerializer(serializers.ModelSerializer):
    """
    Recruiter-facing: question + its answer + analysis, all in one row.
    Never used on any candidate-facing endpoint.
    """
    answer = AnswerSerializer(read_only=True)

    class Meta:
        model = Question
        fields = ["order", "text", "topic_tag", "asked_at", "answer"]


class FinalRatingSerializer(serializers.ModelSerializer):
    class Meta:
        model = FinalRating
        fields = [
            "overall_score",
            "category_scores",
            "strengths",
            "concerns",
            "content_flags",
            "generated_at",
        ]


class InterviewSessionDetailSerializer(serializers.ModelSerializer):
    """
    Full recruiter-facing record: session metadata + every question with
    its answer/analysis + final rating if generated yet.
    """
    candidate_name = serializers.CharField(source="candidate.name", read_only=True)
    candidate_email = serializers.CharField(source="candidate.email", read_only=True)
    questions = QuestionWithAnswerSerializer(many=True, read_only=True)
    final_rating = FinalRatingSerializer(read_only=True)

    class Meta:
        model = InterviewSession
        fields = [
            "id",
            "candidate_name",
            "candidate_email",
            "status",
            "termination_reason",
            "termination_note",
            "max_questions",
            "questions_asked_count",
            "job_description_text",
            "created_at",
            "started_at",
            "ended_at",
            "questions",
            "final_rating",
        ]


class InterviewSessionSummarySerializer(serializers.ModelSerializer):
    """
    Lightweight list-view shape — no nested questions/transcript, just
    enough for a dashboard table row.
    """
    candidate_name = serializers.CharField(source="candidate.name", read_only=True)
    overall_score = serializers.FloatField(
        source="final_rating.overall_score", read_only=True, default=None
    )

    class Meta:
        model = InterviewSession
        fields = [
            "id",
            "candidate_name",
            "status",
            "termination_reason",
            "questions_asked_count",
            "max_questions",
            "overall_score",
            "created_at",
        ]
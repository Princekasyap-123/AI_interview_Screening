# apps/interviews/admin.py

from django.contrib import admin

from apps.interviews.models import Answer, FinalRating, InterviewSession, Question


class QuestionInline(admin.TabularInline):
    """
    Shows all questions for a session inline, ordered correctly. Read-
    only since questions are LLM-generated, not meant to be hand-edited
    from admin — editing here wouldn't reflect back into the generation
    context anyway.
    """
    model = Question
    extra = 0
    can_delete = False
    fields = ["order", "topic_tag", "text", "asked_at"]
    readonly_fields = ["order", "topic_tag", "text", "asked_at"]
    ordering = ["order"]

    def has_add_permission(self, request, obj=None):
        return False


class FinalRatingInline(admin.StackedInline):
    model = FinalRating
    extra = 0
    can_delete = False
    readonly_fields = [
        "id",
        "overall_score",
        "category_scores",
        "strengths",
        "concerns",
        "content_flags",
        "recruiter_visible",
        "candidate_visible",
        "generated_at",
    ]

    def has_add_permission(self, request, obj=None):
        return False


@admin.register(InterviewSession)
class InterviewSessionAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "candidate_name",
        "status",
        "termination_reason",
        "questions_asked_count",
        "max_questions",
        "has_rating",
        "created_at",
    ]
    list_filter = ["status", "termination_reason", "created_at"]
    search_fields = ["candidate__name", "candidate__email", "id"]
    readonly_fields = [
        "id",
        "created_at",
        "started_at",
        "ended_at",
        "questions_asked_count",
    ]
    autocomplete_fields = ["candidate"]
    inlines = [QuestionInline, FinalRatingInline]

    fieldsets = (
        (None, {
            "fields": ("id", "candidate", "job_description_text")
        }),
        ("Status", {
            "fields": (
                "status",
                "questions_asked_count",
                "max_questions",
                "termination_reason",
                "termination_note",
            )
        }),
        ("Timestamps", {
            "fields": ("created_at", "started_at", "ended_at"),
            "classes": ("collapse",),
        }),
    )

    def candidate_name(self, obj):
        return obj.candidate.name
    candidate_name.short_description = "Candidate"
    candidate_name.admin_order_field = "candidate__name"

    def has_rating(self, obj):
        return hasattr(obj, "final_rating")
    has_rating.boolean = True
    has_rating.short_description = "Rated"


@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    """
    Standalone registration for searching/filtering questions across all
    sessions — e.g. spot-checking what kinds of questions the LLM tends
    to generate for a given topic_tag.
    """
    list_display = ["session", "order", "topic_tag", "short_text", "asked_at"]
    list_filter = ["topic_tag"]
    search_fields = ["text", "session__candidate__name"]
    readonly_fields = ["id", "session", "order", "text", "topic_tag", "generation_context", "asked_at"]

    def short_text(self, obj):
        return obj.text[:80] + ("…" if len(obj.text) > 80 else "")
    short_text.short_description = "Question"

    def has_add_permission(self, request):
        return False


@admin.register(Answer)
class AnswerAdmin(admin.ModelAdmin):
    """
    Standalone registration — mainly useful for spot-checking
    transcripts/scores or investigating a specific flagged answer
    (e.g. one with red_flags in analysis_json) without drilling through
    a full session.
    """
    list_display = ["question", "score", "answered_at", "analyzed_at"]
    list_filter = ["analyzed_at"]
    search_fields = ["transcript_text", "question__session__candidate__name"]
    readonly_fields = [
        "id",
        "question",
        "transcript_text",
        "audio_ref",
        "analysis_json",
        "score",
        "answered_at",
        "analyzed_at",
    ]

    def has_add_permission(self, request):
        return False


@admin.register(FinalRating)
class FinalRatingAdmin(admin.ModelAdmin):
    list_display = [
        "session",
        "candidate_name",
        "overall_score",
        "recruiter_visible",
        "generated_at",
    ]
    list_filter = ["recruiter_visible", "candidate_visible", "generated_at"]
    search_fields = ["session__candidate__name", "session__candidate__email"]
    readonly_fields = [
        "id",
        "session",
        "overall_score",
        "category_scores",
        "strengths",
        "concerns",
        "content_flags",
        "generated_at",
    ]

    def candidate_name(self, obj):
        return obj.session.candidate.name
    candidate_name.short_description = "Candidate"

    def has_add_permission(self, request):
        return False
# apps/proctoring/admin.py

from django.contrib import admin

from apps.proctoring.models import ProctoringFlag, ProctoringSession


class ProctoringFlagInline(admin.TabularInline):
    model = ProctoringFlag
    extra = 0
    can_delete = False
    fields = ["flag_type", "severity", "counted_toward_violation", "occurred_at", "metadata"]
    readonly_fields = fields
    ordering = ["occurred_at"]

    def has_add_permission(self, request, obj=None):
        return False


@admin.register(ProctoringSession)
class ProctoringSessionAdmin(admin.ModelAdmin):
    """
    This is the primary "did this candidate cheat" view for a recruiter
    — shows the aggregate risk_score, whether a warning was issued, and
    whether the session was terminated, with the full flag timeline
    inline below.
    """
    list_display = [
        "interview_session",
        "candidate_name",
        "risk_score",
        "countable_violation_count",
        "warning_issued",
        "terminated",
        "updated_at",
    ]
    list_filter = ["warning_issued", "terminated", "updated_at"]
    search_fields = [
        "interview_session__candidate__name",
        "interview_session__candidate__email",
    ]
    readonly_fields = [
        "id",
        "interview_session",
        "countable_violation_count",
        "warning_issued",
        "warning_issued_at",
        "terminated",
        "terminated_at",
        "risk_score",
        "created_at",
        "updated_at",
    ]
    inlines = [ProctoringFlagInline]

    def candidate_name(self, obj):
        return obj.interview_session.candidate.name
    candidate_name.short_description = "Candidate"
    candidate_name.admin_order_field = "interview_session__candidate__name"

    def has_add_permission(self, request):
        # ProctoringSession is created automatically by
        # get_or_create_proctoring_session() when the first flag arrives
        # — no reason to hand-create one from admin.
        return False


@admin.register(ProctoringFlag)
class ProctoringFlagAdmin(admin.ModelAdmin):
    """
    Standalone registration — useful for cross-session analysis, e.g.
    "how often does no_face fire across all interviews" to help tune the
    duration thresholds in flag_processor.py.
    """
    list_display = [
        "proctoring_session",
        "candidate_name",
        "flag_type",
        "severity",
        "counted_toward_violation",
        "occurred_at",
    ]
    list_filter = ["flag_type", "severity", "counted_toward_violation"]
    search_fields = ["proctoring_session__interview_session__candidate__name"]
    readonly_fields = [
        "id",
        "proctoring_session",
        "flag_type",
        "severity",
        "counted_toward_violation",
        "metadata",
        "occurred_at",
        "logged_at",
    ]

    def candidate_name(self, obj):
        return obj.proctoring_session.interview_session.candidate.name
    candidate_name.short_description = "Candidate"

    def has_add_permission(self, request):
        return False
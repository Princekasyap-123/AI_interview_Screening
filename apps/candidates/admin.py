# apps/candidates/admin.py

from django.contrib import admin

from apps.candidates.models import Candidate, ResumeProfile


class ResumeProfileInline(admin.StackedInline):
    """
    Shows the parsed profile directly on the Candidate admin page, since
    it's a OneToOne relationship — avoids needing to click into a
    separate ResumeProfile record to see what was extracted from a
    candidate's resume.
    """
    model = ResumeProfile
    extra = 0
    can_delete = False
    readonly_fields = [
        "id",
        "parsed_at",
        "parser_version",
        "created_at",
        "updated_at",
    ]
    fields = [
        "skills",
        "experience_years",
        "role_level",
        "recent_projects",
        "raw_resume_text",
        "parsed_at",
        "parser_version",
    ]


@admin.register(Candidate)
class CandidateAdmin(admin.ModelAdmin):
    list_display = [
        "name",
        "email",
        "applied_role",
        "has_resume_profile",
        "created_at",
    ]
    list_filter = ["applied_role", "created_at"]
    search_fields = ["name", "email", "phone"]
    readonly_fields = ["id", "created_at", "updated_at"]
    inlines = [ResumeProfileInline]

    fieldsets = (
        (None, {
            "fields": ("id", "name", "email", "phone", "applied_role")
        }),
        ("Resume", {
            "fields": ("resume_file",)
        }),
        ("Timestamps", {
            "fields": ("created_at", "updated_at"),
            "classes": ("collapse",),
        }),
    )

    def has_resume_profile(self, obj):
        return hasattr(obj, "resume_profile")
    has_resume_profile.boolean = True
    has_resume_profile.short_description = "Profile parsed"


@admin.register(ResumeProfile)
class ResumeProfileAdmin(admin.ModelAdmin):
    """
    Standalone registration too — useful for searching/filtering across
    all parsed profiles directly (e.g. "find every candidate with Django
    in skills") without going through a specific Candidate first.
    """
    list_display = [
        "candidate",
        "role_level",
        "experience_years",
        "parser_version",
        "parsed_at",
    ]
    list_filter = ["role_level", "parser_version"]
    search_fields = ["candidate__name", "candidate__email"]
    readonly_fields = ["id", "created_at", "updated_at", "parsed_at"]
    autocomplete_fields = ["candidate"]


# Needed for autocomplete_fields=["candidate"] above to work — Django
# admin requires the referenced model's ModelAdmin to define
# search_fields, which CandidateAdmin already does.
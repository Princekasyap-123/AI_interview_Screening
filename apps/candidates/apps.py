# apps/candidates/apps.py
from django.apps import AppConfig


class CandidatesConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.candidates"

    def ready(self):
        import apps.candidates.signals  # noqa: F401
# config/celery.py

import os

from celery import Celery

# Set the default Django settings module before Celery does anything —
# must happen before `Celery(...)` is instantiated so the app can read
# CELERY_* settings from config/settings.py.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("ai_interview_screening")

# Using a string here means the worker doesn't have to serialize the
# settings object to child processes — namespace="CELERY" means every
# Celery-related setting in settings.py must be prefixed CELERY_
# (which they already are: CELERY_BROKER_URL, CELERY_RESULT_BACKEND, etc.)
app.config_from_object("django.conf:settings", namespace="CELERY")

# Auto-discovers tasks.py in every app listed in INSTALLED_APPS —
# this is what picks up apps/interviews/tasks.py and
# apps/candidates/tasks.py without needing to import them manually
# anywhere.
app.autodiscover_tasks()


@app.task(bind=True, ignore_result=True)
def debug_task(self):
    """
    Sanity-check task — run `python manage.py shell` then:
        from config.celery import debug_task
        debug_task.delay()
    and confirm it prints in the worker's console output. Useful for
    verifying the worker is actually connected to Redis and picking up
    tasks before you trust it with the real interview pipeline.
    """
    print(f"Request: {self.request!r}")
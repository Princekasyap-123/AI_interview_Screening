# config/__init__.py

# This ensures the Celery app is always loaded when Django starts, so
# that @shared_task (used throughout apps/interviews/tasks.py and
# apps/candidates/tasks.py) can find this app instance. Without this
# import, @shared_task calls would silently have no app bound to them
# and .delay() would fail or hang — this is the single most common
# reason a fresh Celery+Django setup "doesn't work" for no obvious
# reason, so don't skip it.

from config.celery import app as celery_app

__all__ = ("celery_app",)
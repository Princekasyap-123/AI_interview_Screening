# apps/interviews/tasks.py

import logging

from celery import shared_task

from apps.interviews.models import InterviewSession
from apps.interviews.services.final_rating import (
    FinalRatingError,
    generate_final_rating_for_session,
)

logger = logging.getLogger(__name__)


@shared_task(
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    autoretry_for=(FinalRatingError,),
    retry_backoff=True,
)
def generate_final_rating(self, session_id: str):
    """
    Celery entry point — fetches the session and delegates all actual
    rating logic to interviews.services.final_rating. Retries
    automatically on FinalRatingError (covers both LLM failures and
    validation failures) up to 3 times with backoff.
    """
    session = (
        InterviewSession.objects.select_related("candidate")
        .prefetch_related("questions__answer")
        .filter(id=session_id)
        .first()
    )

    if session is None:
        logger.error("generate_final_rating called with unknown session_id=%s", session_id)
        return

    try:
        generate_final_rating_for_session(session)
    except FinalRatingError as exc:
        logger.error(
            "generate_final_rating failed for session %s (attempt %d): %s",
            session_id, self.request.retries + 1, exc,
        )
        raise
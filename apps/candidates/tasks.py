# apps/candidates/tasks.py

import logging

from celery import shared_task
from django.utils import timezone

from apps.candidates.bulkresume_client import (
    BulkResumeAPIError,
    get_batch_status,
    submit_resume_for_parsing,
)
from apps.candidates.models import Candidate, ResumeProfile

logger = logging.getLogger(__name__)

MAX_POLL_ATTEMPTS = 10
POLL_RETRY_DELAY_SECONDS = 15


@shared_task(bind=True)
def submit_and_parse_resume(self, candidate_id: str):
    """
    Entry point: triggered automatically by the post_save signal in
    apps/candidates/signals.py whenever a Candidate is saved with a
    resume_file and no existing resume_profile — covers admin, the
    /api/candidates/ REST endpoint, and shell-created candidates alike.
    Do not call this directly from a view as well, or the resume gets
    submitted twice (signal already handles every creation path).

    Submits the file to the bulkresume service and kicks off polling.
    parsing_status is already set to PENDING by the signal before this
    task runs; here we only need to move it to FAILED on a submission
    error, since the polling task owns the PARSED/FAILED transitions
    for everything after that point.
    """
    candidate = Candidate.objects.filter(id=candidate_id).first()
    if candidate is None or not candidate.resume_file:
        logger.error(
            "submit_and_parse_resume: candidate %s missing or has no resume_file",
            candidate_id,
        )
        return

    try:
        batch_id = submit_resume_for_parsing(candidate.resume_file)
    except BulkResumeAPIError as exc:
        logger.error("Failed to submit resume for candidate %s: %s", candidate_id, exc)
        Candidate.objects.filter(id=candidate_id).update(
            parsing_status=Candidate.ParsingStatus.FAILED
        )
        return

    poll_resume_batch.apply_async(
        args=[candidate_id, batch_id], countdown=POLL_RETRY_DELAY_SECONDS
    )


@shared_task(bind=True)
def poll_resume_batch(self, candidate_id: str, batch_id: str, attempt: int = 1):
    """
    Polls the bulkresume batch status. Re-schedules itself if still
    processing, up to MAX_POLL_ATTEMPTS, then gives up, logs an error,
    and marks the candidate's parsing_status as FAILED (candidate keeps
    the fallback profile via resolve_candidate_profile until someone
    investigates and re-triggers parsing).
    """
    try:
        status_data = get_batch_status(batch_id)
    except BulkResumeAPIError as exc:
        logger.error(
            "Polling failed for batch %s (candidate %s): %s", batch_id, candidate_id, exc
        )
        Candidate.objects.filter(id=candidate_id).update(
            parsing_status=Candidate.ParsingStatus.FAILED
        )
        return

    resumes = status_data.get("resumes", [])
    if not resumes:
        logger.error("Batch %s returned no resumes entries", batch_id)
        Candidate.objects.filter(id=candidate_id).update(
            parsing_status=Candidate.ParsingStatus.FAILED
        )
        return

    resume_entry = resumes[0]
    status = resume_entry.get("status")

    if status == "done":
        _create_profile_from_parsed_data(candidate_id, resume_entry)
        return

    if status == "error":
        logger.error(
            "bulkresume parsing failed for candidate %s: %s",
            candidate_id, resume_entry.get("error_message"),
        )
        Candidate.objects.filter(id=candidate_id).update(
            parsing_status=Candidate.ParsingStatus.FAILED
        )
        return

    # status likely "processing" / "pending" — retry unless exhausted
    if attempt >= MAX_POLL_ATTEMPTS:
        logger.error(
            "Gave up polling batch %s for candidate %s after %d attempts "
            "(last status=%s)",
            batch_id, candidate_id, attempt, status,
        )
        Candidate.objects.filter(id=candidate_id).update(
            parsing_status=Candidate.ParsingStatus.FAILED
        )
        return

    poll_resume_batch.apply_async(
        args=[candidate_id, batch_id],
        kwargs={"attempt": attempt + 1},
        countdown=POLL_RETRY_DELAY_SECONDS,
    )


def _create_profile_from_parsed_data(candidate_id: str, resume_entry: dict) -> None:
    """
    Maps the bulkresume parsed "data" shape onto ResumeProfile fields.

    Mapping notes (bulkresume field -> ResumeProfile field):
    - data.skills            -> skills (direct copy, already a flat list)
    - data.experience        -> recent_projects (repurposed: bulkresume
                                 returns JOB history, not side-projects —
                                 mapped as {"title": f"{title} at {company}",
                                 "description": description} so the LLM
                                 prompt still gets meaningful work history
                                 context, since ResumeProfile has no
                                 separate "work_experience" field)
    - experience durations   -> experience_years (approximated by
                                 counting distinct experience entries'
                                 date ranges is unreliable from free text
                                 like "Mar 2024 - Present"; NOT parsed
                                 here, left null — see wiring note)
    - (nothing in response)  -> role_level (bulkresume doesn't infer
                                 this at all; left blank, needs separate
                                 logic if you want it auto-set)
    - data (full dict)       -> raw_resume_text (stored as-is for
                                 reference/debugging, not the actual
                                 raw text since bulkresume doesn't
                                 return unparsed text in this response)
    """
    candidate = Candidate.objects.filter(id=candidate_id).first()
    if candidate is None:
        logger.error("_create_profile_from_parsed_data: candidate %s not found", candidate_id)
        return

    data = resume_entry.get("data", {})

    recent_projects = [
        {
            "title": f"{exp.get('title', 'Role')} at {exp.get('company', 'Unknown')}",
            "description": exp.get("description", ""),
        }
        for exp in data.get("experience", [])
    ]

    ResumeProfile.objects.update_or_create(
        candidate=candidate,
        defaults={
            "skills": data.get("skills", []),
            "experience_years": None,  # see mapping note above
            "role_level": "",  # see mapping note above
            "recent_projects": recent_projects,
            "raw_resume_text": str(data),
            "parsed_at": timezone.now(),
            "parser_version": data.get("extraction_method", "bulkresume"),
        },
    )

    Candidate.objects.filter(id=candidate_id).update(
        parsing_status=Candidate.ParsingStatus.PARSED
    )

    logger.info("Created/updated ResumeProfile for candidate %s from bulkresume", candidate_id)
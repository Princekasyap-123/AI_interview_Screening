# apps/candidates/tasks.py

import logging

from celery import shared_task
from django.utils import timezone

from apps.candidates.bulkresume_client import BulkResumeAPIError, parse_resume
from apps.candidates.models import Candidate, ResumeProfile

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=2, default_retry_delay=20)
def submit_and_parse_resume(self, candidate_id: str):
    """
    Entry point: triggered automatically by the post_save signal in
    apps/candidates/signals.py whenever a Candidate is saved with a
    resume_file and no existing resume_profile — covers admin, the
    /api/candidates/ REST endpoint, and shell-created candidates alike.
    Do not call this directly from a view as well, or the resume gets
    submitted twice (signal already handles every creation path).

    Uses the new synchronous /resumes/parse/ endpoint — no batch_id or
    polling needed anymore, this single call submits the file and
    returns the parsed result directly. Retries up to 2 times on
    transient API failures before marking parsing_status as FAILED.
    """
    candidate = Candidate.objects.filter(id=candidate_id).first()
    if candidate is None or not candidate.resume_file:
        logger.error(
            "submit_and_parse_resume: candidate %s missing or has no resume_file",
            candidate_id,
        )
        return

    try:
        result = parse_resume(candidate.resume_file)
    except BulkResumeAPIError as exc:
        logger.error("Resume parsing failed for candidate %s: %s", candidate_id, exc)

        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)

        logger.error(
            "Gave up parsing resume for candidate %s after %d attempts",
            candidate_id, self.request.retries + 1,
        )
        Candidate.objects.filter(id=candidate_id).update(
            parsing_status=Candidate.ParsingStatus.FAILED
        )
        return

    _create_profile_from_parsed_data(candidate_id, result)


def _create_profile_from_parsed_data(candidate_id: str, result: dict) -> None:
    """
    Maps the new /resumes/parse/ response shape onto ResumeProfile
    fields.

    Mapping notes (API field -> ResumeProfile field):
    - data.skills          -> skills (direct copy)
    - data.experience      -> recent_projects (repurposed: this is JOB
                               history, not side-projects — mapped as
                               {"title": f"{title} at {company}",
                               "description": description})
    - experience_years     -> still NOT populated (no reliable signal
                               in the response)
    - role_level           -> still NOT populated, same reason

    NEW fields in this response with no dedicated model field yet:
    candidate_address, pincode_postal_code, hobbies, training, gender,
    date_of_birth, marital_status, father_name, mother_name,
    known_languages, other_urls, certifications, internships,
    parse_score, ocr_deep_dive_used — all currently only captured
    inside raw_resume_text (full data dict as a string), not queryable
    individually. Flag if you want dedicated fields for these.
    """
    candidate = Candidate.objects.filter(id=candidate_id).first()
    if candidate is None:
        logger.error("_create_profile_from_parsed_data: candidate %s not found", candidate_id)
        return

    data = result.get("data", {})

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
            "experience_years": None,
            "role_level": "",
            "recent_projects": recent_projects,
            "raw_resume_text": str(data),
            "parsed_at": timezone.now(),
            "parser_version": data.get("extraction_method", "resume_parse_v2"),
        },
    )

    Candidate.objects.filter(id=candidate_id).update(
        parsing_status=Candidate.ParsingStatus.PARSED
    )

    logger.info(
        "Created/updated ResumeProfile for candidate %s (parse_score=%s)",
        candidate_id, data.get("parse_score"),
    )
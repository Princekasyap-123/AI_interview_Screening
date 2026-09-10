# apps/candidates/bulkresume_client.py

import logging

import requests
from django.conf import settings

logger = logging.getLogger(__name__)

RESUME_PARSE_URL = "https://astro-buddy.in/django/api/resumes/parse/"

REQUEST_TIMEOUT_SECONDS = 60  # single-file synchronous parse can take
                                # longer than the old batch submit call


class BulkResumeAPIError(Exception):
    """Raised when the resume parsing service returns an error or unexpected shape."""


def _get_headers() -> dict:
    api_key = getattr(settings, "BULKRESUME_API_KEY", None)
    if not api_key:
        raise BulkResumeAPIError(
            "BULKRESUME_API_KEY is not set — cannot call the resume parsing service."
        )
    return {"x-api-key": api_key}


def parse_resume(file_field) -> dict:
    """
    Submits a single resume file and returns the parsed result
    synchronously — no batch_id/polling needed, this new endpoint
    parses and responds in one call.

    Returns the raw response dict, e.g.:
        {
            "id": 9649,
            "status": "done",
            "data": {"name": "...", "skills": [...], ...},
            ...
        }

    Raises BulkResumeAPIError on failure or if status != "done".
    """
    file_field.open("rb")
    try:
        files = {"file": (file_field.name, file_field.read())}
        response = requests.post(
            RESUME_PARSE_URL,
            files=files,
            headers=_get_headers(),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    finally:
        file_field.close()

    if response.status_code not in (200, 201):
        logger.error(
            "Resume parse failed: status=%d body=%s",
            response.status_code, response.text,
        )
        raise BulkResumeAPIError(
            f"resume parse returned {response.status_code}: {response.text}"
        )

    result = response.json()

    if result.get("status") != "done":
        logger.error(
            "Resume parse did not complete successfully: %s",
            result.get("error_message", "unknown error"),
        )
        raise BulkResumeAPIError(
            f"resume parse status={result.get('status')}: "
            f"{result.get('error_message', 'no error message')}"
        )

    return result
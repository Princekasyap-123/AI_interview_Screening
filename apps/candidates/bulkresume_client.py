# apps/candidates/bulkresume_client.py

import logging

import requests
from django.conf import settings

logger = logging.getLogger(__name__)

BULK_UPLOAD_URL = "https://astro-buddy.in/django/api/resumes/bulk-upload/"
BATCH_STATUS_URL_TEMPLATE = "https://astro-buddy.in/django/api/resumes/batch/{batch_id}/status/"

REQUEST_TIMEOUT_SECONDS = 30


class BulkResumeAPIError(Exception):
    """Raised when the bulkresume service returns an error or unexpected shape."""


def _get_headers() -> dict:
    api_key = getattr(settings, "BULKRESUME_API_KEY", None)
    if not api_key:
        raise BulkResumeAPIError(
            "BULKRESUME_API_KEY is not set — cannot call the bulkresume service."
        )
    return {"x-api-key": api_key}


def submit_resume_for_parsing(file_field) -> str:
    """
    Submits a single resume file (a Django FileField/File object, e.g.
    candidate.resume_file) to the bulkresume bulk-upload endpoint and
    returns the batch_id to poll.
    """
    file_field.open("rb")
    try:
        files = {"files": (file_field.name, file_field.read())}
        response = requests.post(
            BULK_UPLOAD_URL,
            files=files,
            headers=_get_headers(),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    finally:
        file_field.close()

    if response.status_code not in (200, 202):
        logger.error(
            "bulkresume bulk-upload failed: status=%d body=%s",
            response.status_code, response.text,
        )
        raise BulkResumeAPIError(
            f"bulk-upload returned {response.status_code}: {response.text}"
        )

    data = response.json()
    batch_id = data.get("batch_id")
    if not batch_id:
        raise BulkResumeAPIError(f"bulk-upload response missing batch_id: {data}")

    logger.info("Submitted resume for parsing, batch_id=%s", batch_id)
    return batch_id


def get_batch_status(batch_id: str) -> dict:
    """
    Polls the batch status endpoint. Returns the raw response dict —
    caller inspects resumes[0]["status"].
    """
    url = BATCH_STATUS_URL_TEMPLATE.format(batch_id=batch_id)
    response = requests.get(
        url, headers=_get_headers(), timeout=REQUEST_TIMEOUT_SECONDS
    )

    if response.status_code != 200:
        logger.error(
            "bulkresume batch status failed: batch=%s status=%d body=%s",
            batch_id, response.status_code, response.text,
        )
        raise BulkResumeAPIError(
            f"batch status returned {response.status_code}: {response.text}"
        )

    return response.json()
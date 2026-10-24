# apps/proctoring/services/attire_checker.py

"""
One-shot "is this reasonably professional attire?" check, white-collar
interviews only — reuses the same reference photo proctoring_client.js
already captures (_captureReferenceFace -> _sendReferencePhoto), so no
new browser capture or socket message is needed.

Deliberately NOT punitive: this never counts toward a warning or
termination (counted_toward_violation=False) and nothing is shown to the
candidate live. Judging someone's clothing from one webcam frame is
unreliable enough that it should never end or interrupt an interview —
it's purely an informational note for the recruiter's dashboard/report.

Runs in a background thread so it never delays the reference-photo ack.
"""

import logging
import threading

from apps.ai_engine.llm_client import llm_client
from apps.ai_engine.prompts.attire_check_prompt import ATTIRE_CHECK_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def check_attire_in_background(interview_session_id, image_base64: str) -> None:
    def _run():
        try:
            from django.utils import timezone
            from apps.proctoring.models import ProctoringFlag, ProctoringSession

            result = llm_client.generate_vision_json(
                system_prompt=ATTIRE_CHECK_SYSTEM_PROMPT,
                user_text="Judge this interview candidate's attire.",
                image_base64=image_base64,
            )
            professional = bool(result.get("professional", True))
            notes = str(result.get("notes", "") or "").strip()[:300]

            proctoring_session = ProctoringSession.objects.filter(
                interview_session_id=interview_session_id
            ).first()
            if proctoring_session is None:
                return

            ProctoringFlag.objects.create(
                proctoring_session=proctoring_session,
                flag_type=ProctoringFlag.FlagType.ATTIRE_CONCERN,
                severity=ProctoringFlag.Severity.LOW,
                counted_toward_violation=False,
                metadata={"professional": professional, "notes": notes, "source": "vision_llm"},
                occurred_at=timezone.now(),
            )
            if not professional:
                logger.info(
                    "Attire check flagged session %s as informal (notes=%r) — "
                    "informational only, recruiter dashboard.",
                    interview_session_id, notes,
                )
        except Exception as exc:
            logger.warning("Attire check failed for session %s: %s", interview_session_id, exc)

    threading.Thread(target=_run, name="attire-check", daemon=True).start()
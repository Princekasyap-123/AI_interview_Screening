# apps/proctoring/consumers.py

import base64
import logging

from asgiref.sync import async_to_sync
from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from django.core.files.base import ContentFile
from django.db import transaction
from django.utils import timezone

from apps.interviews.models import InterviewSession
from apps.proctoring.models import ProctoringFlag, ProctoringSession
from apps.proctoring.services.risk_scorer import (
    ProctoringActionResult,
    get_or_create_proctoring_session,
    handle_incoming_flag,
)

logger = logging.getLogger(__name__)

# Cap on the reference photo payload — a single JPEG snapshot should be
# well under this; generous ceiling just to reject anything malformed.
MAX_REFERENCE_PHOTO_BYTES = 3 * 1024 * 1024


class ProctoringConsumer(AsyncJsonWebsocketConsumer):
    """
    Receives raw proctoring signals from proctoring_client.js (tab
    switches, blur, face-mesh results, fullscreen exits, face-match
    checks, etc.) for a single interview session, and relays back only
    what the candidate is allowed to see: a warning on strike 1, or
    nothing at all (silent logging) otherwise.

    On strike 2 (terminate), this consumer also pushes the interview's
    CLOSE_INTERVIEW turn into the interview's own channel group.

    Expected incoming message shapes (from proctoring_client.js):
      {"type": "flag", "flag_type": "tab_switch", "metadata": {"duration_ms": 2100}}
      {"type": "reference_photo", "image_base64": "...", "image_format": "jpeg"}

    Outgoing message shapes:
      {"type": "warning", "message": "..."}
      {"type": "terminated", "message": "..."}
      {"type": "error", "message": "..."}
      (silent for non-countable or first-of-nothing flags, and for a
      successfully stored reference_photo — the client doesn't need an
      ack to proceed, it just starts its grace-window timer locally)
    """

    async def connect(self):
        self.session_id = self.scope["url_route"]["kwargs"]["session_id"]
        self.interview_session = await self._get_interview_session(self.session_id)

        if self.interview_session is None:
            logger.warning(
                "Proctoring WebSocket rejected — unknown interview session %s",
                self.session_id,
            )
            await self.close(code=4404)
            return

        self.interview_group_name = f"interview_{self.session_id}"

        self.proctoring_group_name = f"proctoring_{self.session_id}"
        await self.channel_layer.group_add(self.proctoring_group_name, self.channel_name)

        await self._ensure_proctoring_session()
        await self.accept()

        logger.info("Proctoring WebSocket connected for session %s", self.session_id)

    async def disconnect(self, close_code):
        if hasattr(self, "proctoring_group_name"):
            await self.channel_layer.group_discard(self.proctoring_group_name, self.channel_name)
        logger.info(
            "Proctoring WebSocket disconnected for session %s (code=%s)",
            getattr(self, "session_id", "unknown"),
            close_code,
        )

    async def receive_json(self, content, **kwargs):
        msg_type = content.get("type")

        if msg_type == "flag":
            await self._handle_flag(content)
        elif msg_type == "reference_photo":
            await self._handle_reference_photo(content)
        else:
            await self.send_json({
                "type": "error",
                "message": f"Unrecognized message type: {msg_type}",
            })

    # ------------------------------------------------------------------
    # Flag handling
    # ------------------------------------------------------------------

    async def _handle_flag(self, content):
        flag_type = content.get("flag_type")
        metadata = content.get("metadata", {})

        if not flag_type:
            await self.send_json({"type": "error", "message": "Missing flag_type."})
            return

        try:
            result = await self._process_flag_locked(flag_type, metadata)
        except ValueError as exc:
            logger.warning("Rejected invalid flag for session %s: %s", self.session_id, exc)
            await self.send_json({"type": "error", "message": str(exc)})
            return
        except Exception as exc:
            logger.error(
                "Unhandled error processing proctoring flag for session %s: %s",
                self.session_id, exc, exc_info=True,
            )
            await self.send_json({
                "type": "error",
                "message": "Something went wrong processing that event.",
            })
            return

        await self._relay_result(result)

    @database_sync_to_async
    def _process_flag_locked(self, flag_type: str, metadata: dict) -> ProctoringActionResult:
        with transaction.atomic():
            proctoring_session = (
                ProctoringSession.objects.select_for_update()
                .select_related("interview_session")
                .get(interview_session_id=self.session_id)
            )
            interview_session = proctoring_session.interview_session
            interview_session.refresh_from_db()

            return handle_incoming_flag(
                interview_session=interview_session,
                flag_type=flag_type,
                metadata=metadata,
            )

    async def _relay_result(self, result: ProctoringActionResult):
        if result.action == ProctoringActionResult.ACTION_WARN:
            await self.send_json({"type": "warning", "message": result.message})

        elif result.action == ProctoringActionResult.ACTION_TERMINATE:
            await self.send_json({"type": "terminated", "message": result.message})

            await self.channel_layer.group_send(
                self.interview_group_name,
                {
                    "type": "proctoring.terminate",
                    "turn": result.interview_turn,
                },
            )

    # ------------------------------------------------------------------
    # Reference photo handling
    # ------------------------------------------------------------------

    async def _handle_reference_photo(self, content):
        """
        Stores the candidate's start-of-interview snapshot for the
        recruiter dashboard's audit trail. This is purely a stored
        image — it plays NO role in live face-match detection, which
        happens entirely client-side (proctoring_client.js keeps its
        own in-memory descriptor and only ever sends the resulting
        face_mismatch flag, same as any other detector here).
        """
        image_base64 = content.get("image_base64")
        image_format = content.get("image_format", "jpeg")

        if not image_base64:
            await self.send_json({"type": "error", "message": "Missing image_base64."})
            return

        try:
            await self._save_reference_photo(image_base64, image_format)
        except ValueError as exc:
            logger.warning(
                "Rejected reference photo for session %s: %s", self.session_id, exc
            )
            await self.send_json({"type": "error", "message": str(exc)})
            return

        logger.info("Reference photo stored for session %s", self.session_id)

    @database_sync_to_async
    def _save_reference_photo(self, image_base64: str, image_format: str) -> None:
        try:
            image_bytes = base64.b64decode(image_base64, validate=True)
        except Exception as exc:
            raise ValueError(f"Invalid base64 image data: {exc}") from exc

        if len(image_bytes) > MAX_REFERENCE_PHOTO_BYTES:
            raise ValueError(
                f"Reference photo too large ({len(image_bytes)} bytes, "
                f"max {MAX_REFERENCE_PHOTO_BYTES})."
            )

        if not image_bytes:
            raise ValueError("Decoded reference photo is empty.")

        proctoring_session = ProctoringSession.objects.get(
            interview_session_id=self.session_id
        )

        safe_format = "".join(c for c in image_format if c.isalnum()) or "jpeg"
        proctoring_session.reference_photo.save(
            f"{self.session_id}.{safe_format}",
            ContentFile(image_bytes),
            save=False,
        )
        proctoring_session.reference_captured_at = timezone.now()
        proctoring_session.save(
            update_fields=["reference_photo", "reference_captured_at", "updated_at"]
        )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    @database_sync_to_async
    def _get_interview_session(self, session_id):
        return InterviewSession.objects.filter(id=session_id).first()

    @database_sync_to_async
    def _ensure_proctoring_session(self):
        get_or_create_proctoring_session(self.interview_session)
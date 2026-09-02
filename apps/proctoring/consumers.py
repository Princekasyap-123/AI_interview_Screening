# apps/proctoring/consumers.py

import logging

from asgiref.sync import async_to_sync
from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer
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


class ProctoringConsumer(AsyncJsonWebsocketConsumer):
    """
    Receives raw proctoring signals from proctoring_client.js (tab
    switches, blur, face-mesh results, fullscreen exits, etc.) for a
    single interview session, and relays back only what the candidate is
    allowed to see: a warning on strike 1, or nothing at all (silent
    logging) otherwise.

    On strike 2 (terminate), this consumer also pushes the interview's
    CLOSE_INTERVIEW turn into the interview's own channel group, so the
    candidate's interview WebSocket (a SEPARATE connection, driven by
    InterviewConsumer) closes cleanly with the explanation — this is the
    cross-consumer link that keeps the interview app from ever trusting
    a client-reported "this is strike 2" claim directly.

    Expected incoming message shape (from proctoring_client.js):
      {"type": "flag", "flag_type": "tab_switch", "metadata": {"duration_ms": 2100}}

    Outgoing message shapes:
      {"type": "warning", "message": "..."}
      {"type": "terminated", "message": "..."}
      (silent / no message sent for non-countable or first-of-nothing flags)
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

        # Interview group name — must exactly match the group name used
        # in InterviewConsumer.connect() (f"interview_{session_id}") so
        # a terminate event can be pushed into that group from here.
        self.interview_group_name = f"interview_{self.session_id}"

        # This consumer's own group — not strictly needed for a 1:1
        # candidate connection, but keeps the pattern consistent in case
        # you later add a supervisor/observer view into proctoring events.
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
        if content.get("type") != "flag":
            await self.send_json({
                "type": "error",
                "message": f"Unrecognized message type: {content.get('type')}",
            })
            return

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

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    @database_sync_to_async
    def _process_flag_locked(self, flag_type: str, metadata: dict) -> ProctoringActionResult:
        """
        Wraps handle_incoming_flag() in a locked transaction on the
        ProctoringSession row, so two near-simultaneous flags can't both
        read countable_violation_count as 0 and both resolve as "strike 1"
        (the race condition flagged when risk_scorer.py was written).
        """
        with transaction.atomic():
            # Refresh with a row lock before delegating to the risk
            # scorer, so the increment-and-check inside process_flag()
            # happens against a locked row for the duration of this call.
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

            # Push the interview's close turn into the INTERVIEW group so
            # InterviewConsumer (a separate WebSocket connection) relays
            # it to whatever's rendering the interview call itself.
            await self.channel_layer.group_send(
                self.interview_group_name,
                {
                    "type": "proctoring.terminate",
                    "turn": result.interview_turn,
                },
            )

        # ACTION_NONE: silent by design — non-countable or non-strike
        # flags are logged server-side only, nothing sent to candidate.

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    @database_sync_to_async
    def _get_interview_session(self, session_id):
        return InterviewSession.objects.filter(id=session_id).first()

    @database_sync_to_async
    def _ensure_proctoring_session(self):
        get_or_create_proctoring_session(self.interview_session)
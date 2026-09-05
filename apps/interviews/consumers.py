# apps/interviews/consumers.py

import base64
import logging
import os
import uuid

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from django.conf import settings

from apps.ai_engine.stt_client import stt_client
from apps.candidates.services import resolve_candidate_profile
from apps.interviews.models import InterviewSession
from apps.interviews.services.interview_state import InterviewFSM, TurnAction

logger = logging.getLogger(__name__)

AUDIO_SCRATCH_DIR = os.path.join(settings.MEDIA_ROOT, "interview_audio_scratch")

MAX_AUDIO_BYTES = 15 * 1024 * 1024


class InterviewConsumer(AsyncJsonWebsocketConsumer):
    """
    Drives one candidate's live interview session over a single WebSocket
    connection. This consumer is intentionally "dumb" about interview
    logic — every decision comes from InterviewFSM; this class only:
      - authenticates the connection to a session
      - relays candidate messages (join, ready_for_question,
        answer_submitted, end_interview, proctoring-triggered close via
        group_send) into FSM calls
      - runs STT on submitted audio before handing transcript text to
        the FSM
      - relays FSM turn dicts back out to the frontend as JSON

    Expected incoming message shapes (from interview_socket.js):
      {"type": "join"}
      {"type": "ready_for_question"}
      {"type": "answer_submitted", "question_id": "...", "audio_base64": "...", "audio_format": "webm"}
      {"type": "end_interview"}

    Outgoing message shapes (consumed by tts_player.js / proctoring_client.js):
      {"type": "turn", "action": "speak_question", "question_id": "...", "text": "..."}
      {"type": "turn", "action": "close_interview", "message": "..."}
      {"type": "info", "message": "..."}
      {"type": "error", "message": "..."}

    NOTE: proctoring-triggered termination no longer arrives as a client
    message — it's pushed server-side via channel_layer.group_send from
    ProctoringConsumer (see proctoring_terminate() below). The old
    "proctoring_violation" client message type has been removed entirely,
    since trusting a client-reported violation count was a security gap.
    """

    async def connect(self):
        self.session_id = self.scope["url_route"]["kwargs"]["session_id"]
        self.session = await self._get_session(self.session_id)

        if self.session is None:
            logger.warning("WebSocket connect rejected — unknown session %s", self.session_id)
            await self.close(code=4404)
            return

        self.group_name = f"interview_{self.session_id}"
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

        os.makedirs(AUDIO_SCRATCH_DIR, exist_ok=True)

        # Guards against duplicate 'join' messages on the same connection
        # calling fsm.start() twice before session.status flips to
        # IN_PROGRESS — see the earlier repeat-question race discussion.
        self._join_handled = False

        logger.info("WebSocket connected for session %s", self.session_id)

    async def disconnect(self, close_code):
        if hasattr(self, "group_name"):
            await self.channel_layer.group_discard(self.group_name, self.channel_name)
        logger.info(
            "WebSocket disconnected for session %s (code=%s)",
            getattr(self, "session_id", "unknown"),
            close_code,
        )

    async def receive_json(self, content, **kwargs):
        msg_type = content.get("type")

        try:
            if msg_type == "join":
                await self._handle_join()
            elif msg_type == "ready_for_question":
                await self._handle_ready_for_question()
            elif msg_type == "answer_submitted":
                await self._handle_answer_submitted(content)
            elif msg_type == "end_interview":
                await self._handle_end_interview()
            else:
                await self.send_json({
                    "type": "error",
                    "message": f"Unrecognized message type: {msg_type}",
                })
        except Exception as exc:
            logger.error(
                "Unhandled error processing message type=%s for session %s: %s",
                msg_type, self.session_id, exc, exc_info=True,
            )
            await self.send_json({
                "type": "error",
                "message": "Something went wrong. Please try again.",
            })

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    async def _handle_join(self):
        """
        Candidate has joined the call and is ready to begin. Triggers
        question generation (if not already done) and the intro turn.

        Idempotent against duplicate 'join' messages on the same
        connection — see the _join_handled flag set in connect().
        """
        if self._join_handled:
            logger.warning(
                "Duplicate 'join' message received for session %s — "
                "ignoring (already handled on this connection).",
                self.session_id,
            )
            return
        self._join_handled = True

        session = await self._refresh_session()

        if session.status == InterviewSession.Status.COMPLETED:
            await self.send_json({
                "type": "turn",
                "action": TurnAction.CLOSE_INTERVIEW,
                "message": "This interview has already been completed.",
            })
            return

        if session.status == InterviewSession.Status.TERMINATED:
            await self.send_json({
                "type": "turn",
                "action": TurnAction.CLOSE_INTERVIEW,
                "message": session.termination_note or "This interview session has ended.",
            })
            return

        if session.status == InterviewSession.Status.IN_PROGRESS:
            await self.send_json({
                "type": "info",
                "message": "Reconnected. Please wait for your next question.",
            })
            return

        fsm = await self._build_fsm(session)
        turn = await database_sync_to_async(fsm.start)()
        await self._send_turn(turn)

    async def _handle_ready_for_question(self):
        """
        Sent by the frontend once the company intro has finished being
        spoken (TTS speakEnd). Advances the FSM to the candidate's
        self-intro request — the actual first Question row.
        """
        session = await self._refresh_session()
        fsm = await self._build_fsm(session)

        turn = await database_sync_to_async(fsm.advance_to_first_question)()
        await self._send_turn(turn)

    async def _handle_answer_submitted(self, content):
        """
        Receives raw audio for a candidate's spoken answer, transcribes
        it via Groq Whisper (stt_client), then hands the resulting text
        to the FSM for analysis + next-turn decision.
        """
        question_id = content.get("question_id")
        audio_base64 = content.get("audio_base64")
        audio_format = content.get("audio_format", "webm")

        if not question_id:
            await self.send_json({"type": "error", "message": "Missing question_id."})
            return

        if not audio_base64:
            await self.send_json({"type": "error", "message": "Missing audio_base64."})
            return

        try:
            audio_path = await self._save_audio_to_scratch(audio_base64, audio_format)
        except ValueError as exc:
            await self.send_json({"type": "error", "message": str(exc)})
            return

        try:
            transcript_text = await self._transcribe_audio(audio_path)
        except RuntimeError as exc:
            logger.error(
                "STT failed for session %s question %s: %s",
                self.session_id, question_id, exc,
            )
            transcript_text = ""
        finally:
            await self._cleanup_audio_file(audio_path)

        session = await self._refresh_session()
        fsm = await self._build_fsm(session)

        turn = await database_sync_to_async(fsm.submit_answer)(
            question_id=question_id,
            transcript_text=transcript_text,
            audio_ref="",
        )
        await self._send_turn(turn)

    async def _handle_end_interview(self):
        """
        Candidate clicked "End interview" in the UI. Closes the session
        server-side (so DB status actually updates, not just the client
        tearing down its own media/UI) and sends the close_interview
        turn back so the existing handleInterviewClose() UI path runs.
        Closes the socket afterward — matches proctoring_terminate()'s
        pattern of closing with a distinct code (4000, vs 4001 for
        proctoring termination) so server logs/close codes distinguish
        the two paths if ever needed.
        """
        session = await self._refresh_session()
        fsm = await self._build_fsm(session)

        turn = await database_sync_to_async(fsm.end_interview_by_candidate)()
        await self._send_turn(turn)
        await self.close(code=4000)

    # ------------------------------------------------------------------
    # Group event handler — proctoring-triggered termination
    # ------------------------------------------------------------------

    async def proctoring_terminate(self, event):
        """
        Handles the group_send from ProctoringConsumer when a 2nd
        violation forces termination server-side. Channels dispatches
        group messages to a method named after the "type" key sent in
        group_send ("proctoring.terminate" -> proctoring_terminate).
        """
        turn = event["turn"]
        await self._send_turn(turn)
        await self.close(code=4001)

    # ------------------------------------------------------------------
    # Audio handling helpers
    # ------------------------------------------------------------------

    @database_sync_to_async
    def _save_audio_to_scratch(self, audio_base64: str, audio_format: str) -> str:
        try:
            audio_bytes = base64.b64decode(audio_base64, validate=True)
        except Exception as exc:
            raise ValueError(f"Invalid base64 audio data: {exc}") from exc

        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise ValueError(
                f"Audio payload too large ({len(audio_bytes)} bytes, "
                f"max {MAX_AUDIO_BYTES})."
            )

        if not audio_bytes:
            raise ValueError("Decoded audio payload is empty.")

        safe_format = "".join(c for c in audio_format if c.isalnum()) or "webm"
        filename = f"{uuid.uuid4()}.{safe_format}"
        filepath = os.path.join(AUDIO_SCRATCH_DIR, filename)

        with open(filepath, "wb") as f:
            f.write(audio_bytes)

        return filepath

    @database_sync_to_async
    def _transcribe_audio(self, audio_path: str) -> str:
        return stt_client.transcribe(audio_path)

    @database_sync_to_async
    def _cleanup_audio_file(self, audio_path: str) -> None:
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
        except OSError as exc:
            logger.warning("Failed to clean up scratch audio file %s: %s", audio_path, exc)

    # ------------------------------------------------------------------
    # Session / FSM helpers
    # ------------------------------------------------------------------

    async def _send_turn(self, turn: dict):
        await self.send_json({"type": "turn", **turn})

    async def _build_fsm(self, session: InterviewSession) -> InterviewFSM:
        candidate_profile = await database_sync_to_async(resolve_candidate_profile)(
            session.candidate
        )
        return InterviewFSM(session=session, candidate_profile=candidate_profile)

    @database_sync_to_async
    def _get_session(self, session_id):
        return (
            InterviewSession.objects.select_related("candidate")
            .filter(id=session_id)
            .first()
        )

    @database_sync_to_async
    def _refresh_session(self):
        self.session = (
            InterviewSession.objects.select_related("candidate")
            .get(id=self.session_id)
        )
        return self.session
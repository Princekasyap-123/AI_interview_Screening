# apps/interviews/consumers.py

import asyncio
import base64
import logging
import os
import shutil
import time
import uuid
from collections import deque
from urllib.parse import parse_qs

from asgiref.sync import sync_to_async
from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from django.conf import settings

from apps.ai_engine.stt_client import stt_client
from apps.candidates.services import resolve_candidate_profile
from apps.interviews.models import ConversationTurn, InterviewSession, Question
from apps.interviews.services.echo_guard import is_echo
from apps.interviews.services.stt_vocab import build_prompt, build_terms, correct_terms
from apps.interviews.services.interview_state import InterviewFSM, TurnAction
from apps.interviews.services.language import (
    HI,
    MODE_HI,
    detect_language,
    stt_language_for,
    to_hindi,
    warm_hindi_in_background,
)
from apps.interviews.services.speech_cleanup import looks_unfinished

MAX_CONTINUATIONS = 2           # per question: how often an unfinished sentence may be waited for
MAX_SILENT_ECHO_DROPS = 3        # per question; after that the normal "didn't catch that" path takes over

logger = logging.getLogger(__name__)

AUDIO_SCRATCH_DIR = os.path.join(settings.MEDIA_ROOT, "interview_audio_scratch")

MAX_AUDIO_BYTES = 15 * 1024 * 1024

# Work that mostly WAITS on the network (Groq STT / LLM calls inside the FSM)
# must not run on Channels' single shared "thread-sensitive" thread, or every
# candidate's turn queues behind everyone else's. With PostgreSQL each worker
# thread gets its own connection, so it is safe to run them in parallel.
# SQLite (local dev) allows one writer at a time, so keep it serialized there.
_DB_ENGINE = settings.DATABASES["default"]["ENGINE"]
_SERIALIZE_BLOCKING_WORK = _DB_ENGINE.endswith("sqlite3")


def blocking(func):
    """database_sync_to_async that runs in parallel worker threads on PostgreSQL."""
    return database_sync_to_async(func, thread_sensitive=_SERIALIZE_BLOCKING_WORK)


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

        # The interview link carries session id + access token; the socket
        # must present the same token (query string ?token=...), otherwise
        # anyone who learns a session UUID could join someone else's interview.
        token = (parse_qs(self.scope.get("query_string", b"").decode()).get("token") or [""])[0]
        if not token or token != self.session.access_token:
            logger.warning("WebSocket connect rejected — bad token for session %s", self.session_id)
            await self.close(code=4403)
            return

        # A finished interview cannot be re-entered (the link has expired).
        if self.session.status in ("completed", "terminated"):
            logger.info("WebSocket connect rejected — session %s already ended", self.session_id)
            await self.close(code=4410)
            return

        # One candidate message is processed at a time: overlapping submits
        # (e.g. a barge-in right after an auto-submit) queue up instead of racing.
        self._turn_lock = asyncio.Lock()

        # What the interviewer said most recently: a transcript that is a
        # word-for-word copy of it is the mic hearing the speaker, not the candidate.
        self._recent_spoken = deque(maxlen=4)
        self._echo_ignores = {}      # question_id -> how many echoes were silently dropped
        self._partial = {}           # question_id -> words heard so far in an answer that stopped mid-sentence
        self._continuations = {}     # question_id -> how many times we kept listening

        # Interview language: "en" / "hi" / "both" (see services/language.py). _lang is the
        # language the interviewer is speaking right now; in "both" it follows the candidate.
        self._lang_mode = getattr(self.session, "language_mode", "en") or "en"
        self._lang = getattr(self.session, "current_language", "en") or "en"
        if self._lang_mode == MODE_HI:
            self._lang = HI

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

        if msg_type == "ping":                 # keep-alive from interview_socket.js
            await self.send_json({"type": "pong"})
            return

        try:
            if msg_type == "join":
                async with self._turn_lock:
                    await self._handle_join()
            elif msg_type == "ready_for_question":
                async with self._turn_lock:
                    await self._handle_ready_for_question()
            elif msg_type == "answer_submitted":
                async with self._turn_lock:
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
            # Reconnect: pick up exactly where the interview was.
            fsm = await self._build_fsm(session)
            turn = await blocking(fsm.resume_turn)()
            await self._warm_hindi(fsm)
            await self._send_turn(turn)
            return

        fsm = await self._build_fsm(session)
        turn = await blocking(fsm.start)()
        await self._warm_hindi(fsm, include_intro=True)
        await self._send_turn(turn)

    async def _handle_ready_for_question(self):
        """
        Sent by the frontend once the company intro has finished being
        spoken (TTS speakEnd). Advances the FSM to the candidate's
        self-intro request — the actual first Question row.
        """
        session = await self._refresh_session()
        fsm = await self._build_fsm(session)

        turn = await blocking(fsm.advance_to_first_question)()
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
        interjected = bool(content.get("interjected"))   # candidate talked over the interviewer

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

        # Tell the browser right away that the audio arrived (it can show
        # "thinking" and play a filler word) while STT + the LLM work.
        await self.send_json({"type": "ack", "question_id": str(question_id)})

        spoken_language = ""
        transcript_text = ""
        try:
            transcript_text, spoken_language = await self._transcribe_audio(audio_path)
        except RuntimeError as exc:
            logger.error(
                "STT failed for session %s question %s: %s",
                self.session_id, question_id, exc,
            )
            transcript_text = ""
        finally:
            await self._keep_stt_sample(audio_path, transcript_text, spoken_language)
            await self._cleanup_audio_file(audio_path)

        logger.info(
            "Session %s: heard %r (reason=%s, speech=%sms, peak=%sdB, interjected=%s)",
            self.session_id, (transcript_text or "")[:200], content.get("reason"),
            content.get("speech_ms"), content.get("peak_db"), interjected,
        )

        session = await self._refresh_session()
        fsm = await self._build_fsm(session)

        # Echo: the transcript is just the interviewer's own words coming back
        # through the microphone (phone speaker next to the mic). Never score it.
        if transcript_text and is_echo(transcript_text, self._recent_spoken):
            drops = self._echo_ignores.get(str(question_id), 0)
            logger.info(
                "Session %s: transcript is an echo of the interviewer (drop #%d): %r",
                self.session_id, drops + 1, transcript_text[:80],
            )
            if drops < MAX_SILENT_ECHO_DROPS:
                self._echo_ignores[str(question_id)] = drops + 1
                if interjected:
                    # the interviewer was cut off by its own echo: say the question again
                    turn = await blocking(fsm.repeat_turn)(question_id)
                else:
                    turn = {"action": "listen_again", "question_id": str(question_id)}
                await self._send_turn(turn)
                return
            transcript_text = ""      # keeps echoing: fall through to the "didn't catch that" path

        # English <-> Hindi: follow the language the candidate actually spoke.
        if transcript_text:
            new_lang = detect_language(transcript_text, spoken_language, self._lang_mode, self._lang)
            if new_lang != self._lang:
                await self._switch_language(new_lang, fsm)

        if transcript_text:
            await self._log_candidate(transcript_text, question_id, self._lang)

        # Mid-sentence stop ("...worked with Django and"): keep listening and join the
        # next part to it, the way a person waits for the end of the thought.
        qkey = str(question_id)
        pending = self._partial.get(qkey, "")
        if transcript_text:
            combined = f"{pending} {transcript_text}".strip()
            used = self._continuations.get(qkey, 0)
            if used < MAX_CONTINUATIONS and content.get("reason") != "no_answer" and looks_unfinished(transcript_text):
                self._partial[qkey] = combined
                self._continuations[qkey] = used + 1
                logger.info("Session %s: answer stopped mid-sentence (%r) — listening on.",
                            self.session_id, transcript_text[-40:])
                await self._send_turn({"action": "listen_again", "question_id": qkey, "continue": True})
                return
            transcript_text = combined
            self._partial.pop(qkey, None)
        elif pending:
            transcript_text = pending           # nothing more came: what we have is the answer
            self._partial.pop(qkey, None)

        # The browser heard speech but Whisper returned nothing (noise, a cough,
        # a very quiet voice): ask again instead of scoring silence as the answer.
        # (A silent answer where the browser heard NOTHING still moves on as before.)
        if not (transcript_text or "").strip():
            heard_speech = interjected or content.get("reason") in ("silence", "said_done")
            if heard_speech:
                await self._log_candidate("", question_id, self._lang, kind="inaudible")
                turn = await blocking(fsm.no_speech_turn)(question_id)
                if turn is None and interjected:
                    turn = await blocking(fsm.repeat_turn)(question_id)
                if turn is not None:
                    logger.info("Session %s: empty transcript after speech — asking again.", self.session_id)
                    await self._send_turn(turn)
                    return

        turn = await blocking(fsm.submit_answer)(
            question_id=question_id,
            transcript_text=transcript_text,
            audio_ref="",
        )

        # Abusive language: the FSM already closed the session. Tell the
        # proctoring socket to show the "interview ended" screen, then
        # close this socket like a proctoring termination does.
        terminated_notice = turn.pop("_proctoring_terminated", None)
        await self._send_turn(turn)
        if terminated_notice:
            await self.channel_layer.group_send(
                f"proctoring_{self.session_id}",
                {"type": "proctoring.notify_terminated", **terminated_notice},
            )
            await self.close(code=4001)

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

        turn = await blocking(fsm.end_interview_by_candidate)()
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

    @sync_to_async(thread_sensitive=False)
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

    @sync_to_async(thread_sensitive=False)
    def _transcribe_audio(self, audio_path: str) -> tuple[str, str]:
        """(text, language Whisper heard). Only "both" needs Whisper's own language detection;
        English keeps the project default and "hi" forces Hindi, through the plain call."""
        language = stt_language_for(self._lang_mode)
        prompt, terms = self._stt_vocabulary(language)
        if language == "":                       # "both": let Whisper tell English from Hindi
            text, spoken = stt_client.transcribe_detailed(audio_path, language="", prompt=prompt)
            return correct_terms(text, terms), spoken
        text = stt_client.transcribe(audio_path, language=language, prompt=prompt)
        return correct_terms(text, terms), ""

    def _stt_vocabulary(self, language: str = "en") -> tuple[str, list[str]]:
        """Whisper priming text + the vocabulary used to fix near-miss spellings (built once per interview
        and language: Hindi speech gets a Hindi hint)."""
        cache = self.__dict__.setdefault("_stt_vocab_cache", {})
        cached = cache.get(language)
        if cached is None:
            try:
                profile = resolve_candidate_profile(self.session.candidate)
                jd = self.session.job_description_text
                name = (profile or {}).get("name") or getattr(self.session.candidate, "name", "")
                cached = (build_prompt(jd, profile, name, language), build_terms(jd, profile))
            except Exception as exc:             # the vocabulary only helps; never block a transcript
                logger.warning("Could not build the speech vocabulary: %s", exc)
                cached = ("", [])
            cache[language] = cached
        return cached

    @sync_to_async(thread_sensitive=False)
    def _keep_stt_sample(self, audio_path: str, transcript: str, spoken_language: str) -> None:
        """Opt-in (STT_KEEP_AUDIO=1): keeps each answer's audio with what Whisper heard, in media/stt_samples/,
        so real recordings can be used to compare speech engines. Off by default - it stores candidate voices."""
        if not getattr(settings, "STT_KEEP_AUDIO", False) or not audio_path or not os.path.exists(audio_path):
            return
        try:
            folder = os.path.join(str(settings.MEDIA_ROOT), "stt_samples")
            os.makedirs(folder, exist_ok=True)
            stem = f"{self.session_id}_{int(time.time())}_{self._lang_mode}"
            shutil.copyfile(audio_path, os.path.join(folder, stem + os.path.splitext(audio_path)[1]))
            with open(os.path.join(folder, stem + ".txt"), "w", encoding="utf-8") as fh:
                fh.write(f"language_mode={self._lang_mode} spoken={spoken_language}\n{transcript or ''}\n")
        except OSError as exc:
            logger.warning("Could not keep the STT sample: %s", exc)

    @sync_to_async(thread_sensitive=False)
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
        action = turn.get("action", "")
        action = getattr(action, "value", action)          # TurnAction is a str-Enum
        original = dict(turn)
        turn = await self._localize_turn(turn, str(action))
        if turn.get("text") and str(action).startswith("speak_"):
            self._recent_spoken.append(turn["text"])
        await self._log_interviewer(str(action), original, turn)
        await self.send_json({"type": "turn", **turn})

    # ------------------------------------------------------------------
    # Language (English / Hindi)
    # ------------------------------------------------------------------

    async def _localize_turn(self, turn: dict, action: str) -> dict:
        """Adds "language" to the turn and, when the interviewer is speaking Hindi,
        replaces the spoken text with its Hindi version. If a line cannot be
        translated it is sent in English and marked "en" — the browser then uses the
        English voice, so the candidate always hears something sensible."""
        turn = dict(turn)
        turn["language"] = "en"
        if self._lang != HI:
            return turn

        key = "text" if str(action).startswith("speak_") else ("message" if action == "close_interview" else "")
        if not key or not turn.get(key):
            turn["language"] = HI
            return turn

        hindi = await sync_to_async(to_hindi, thread_sensitive=False)(turn[key])
        if hindi:
            turn[key] = hindi
            turn["language"] = HI
        return turn

    # ------------------------------------------------------------------
    # Full conversation log (shown to the recruiter on the dashboard)
    # ------------------------------------------------------------------

    async def _log_interviewer(self, action: str, original: dict, sent: dict) -> None:
        key = "text" if action.startswith("speak_") else ("message" if action == "close_interview" else "")
        if not key or not sent.get(key):
            return
        kind = {"speak_company_intro": "intro", "speak_intro": "question", "speak_question": "question",
                "speak_follow_up": "reply", "close_interview": "closing"}.get(action, "reply")
        english = original.get(key) or ""
        spoken = sent.get(key) or ""
        await self._store_turn(
            "interviewer", kind, spoken, english if english != spoken else "",
            sent.get("language", "en"), sent.get("question_id") or original.get("question_id"),
        )

    async def _log_candidate(self, text: str, question_id, language: str, kind: str = "answer") -> None:
        await self._store_turn("candidate", kind, text or "", "", language, question_id)

    async def _store_turn(self, speaker, kind, text, text_en, language, question_id) -> None:
        try:
            await sync_to_async(self._write_turn, thread_sensitive=False)(
                speaker, kind, text, text_en, language, question_id
            )
        except Exception as exc:                  # a logging problem must never interrupt the interview
            logger.warning("Could not store conversation line for %s: %s", self.session_id, exc)

    def _write_turn(self, speaker, kind, text, text_en, language, question_id) -> None:
        qid = None
        if question_id:
            try:
                qid = Question.objects.filter(pk=question_id, session_id=self.session_id).values_list("pk", flat=True).first()
            except (ValueError, TypeError):
                qid = None
        ConversationTurn.objects.create(
            session_id=self.session_id, speaker=speaker, kind=kind, text=text, text_en=text_en,
            language=language or "en", question_id=qid,
        )

    async def _switch_language(self, new_lang: str, fsm: InterviewFSM) -> None:
        logger.info("Session %s: interview language %s -> %s", self.session_id, self._lang, new_lang)
        self._lang = new_lang
        fsm.session.current_language = new_lang
        await blocking(self._save_language)(new_lang)
        # The browser switches its voice + filler lines now, before the next question arrives.
        await self.send_json({"type": "language", "language": new_lang})
        if new_lang == HI:
            await self._warm_hindi(fsm)

    def _save_language(self, lang: str) -> None:
        InterviewSession.objects.filter(pk=self.session_id).update(current_language=lang)

    async def _warm_hindi(self, fsm: InterviewFSM, include_intro: bool = False) -> None:
        """Start translating + synthesising the upcoming questions in Hindi in the background."""
        if self._lang != HI:
            return
        try:
            texts = await blocking(fsm.upcoming_spoken_texts)(include_intro)
            warm_hindi_in_background(texts)
        except Exception as exc:                                  # best effort only
            logger.warning("Hindi warm-up failed for session %s: %s", self.session_id, exc)

    async def _build_fsm(self, session: InterviewSession) -> InterviewFSM:
        candidate_profile = await blocking(resolve_candidate_profile)(session.candidate)
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
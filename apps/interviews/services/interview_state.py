# apps/interviews/services/interview_state.py

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import transaction
from django.utils import timezone

from apps.interviews.models import Answer, InterviewSession, Question
from apps.interviews.services.answer_analyzer import (
    build_full_transcript,
    run_answer_analysis,
    save_answer,
)
from apps.interviews.services.question_generator import (
    QuestionGenerationError,
    generate_questions_for_session,
)
from apps.interviews.services.turn_intent import (
    REPEAT_REPLY,
    INTENT_ABUSIVE,
    INTENT_CANDIDATE_QUESTION,
    INTENT_CLARIFICATION,
    INTENT_ANSWER,
    INTENT_OFF_TOPIC,
    INTENT_REPEAT,
    INTENT_UNCLEAR,
    OFF_TOPIC_REPLY,
    UNCLEAR_REPLY,
    TurnIntent,
    classify_fast,
    classify_reply_llm,
)

logger = logging.getLogger(__name__)

MAX_FOLLOW_UPS_PER_QUESTION = getattr(settings, "MAX_FOLLOW_UPS_PER_QUESTION", 1)

MAX_ADVANCE_ATTEMPTS = 25

# How many times the candidate may ask to repeat / clarify / ask their own
# question on ONE question before their next reply is treated as the answer.
MAX_CANDIDATE_QUERIES_PER_QUESTION = 3
# Hard cap including "repeat" / "couldn't hear" turns, so a noisy room can never loop forever.
MAX_CANDIDATE_QUERIES_TOTAL = 6
QUERY_LIMIT_REPLY = (
    "I'm not able to take more questions right now — the recruiting team will "
    "be happy to answer them after the interview. Please give your answer to this one."
)

# Words that normally start a spoken question / instruction. Only these get
# lower-cased when the candidate's first name is put in front ("Rahul, tell me…");
# names, acronyms and "I" ("Python…", "SQL…", "I'd…") keep their capital letter.
_LOWERCASE_STARTERS = frozenset({
    "tell", "what", "how", "why", "can", "could", "would", "will", "walk", "describe",
    "explain", "please", "let's", "so", "now", "have", "has", "do", "does", "did", "which",
    "where", "when", "who", "give", "share", "talk", "you", "your", "in", "if", "suppose",
    "imagine", "since", "given", "as", "is", "are", "was", "were", "thank", "great", "okay",
    "alright", "first", "next", "lastly", "finally", "could", "briefly", "think", "consider",
})

ABUSE_TERMINATION_MESSAGE = (
    "This interview has ended because abusive or offensive language was "
    "used, which is against the interview rules explained at the start. "
    "Your responses so far have been saved, and the recruiting team has "
    "been informed."
)

COMPANY_INTRO_TEXT = (
    "Hi, and welcome. I'm the AI interviewer from White Force. Before we "
    "begin, a quick word about what happens today: I'll ask you a series "
    "of questions based on your background and the role you've applied "
    "for. Just answer naturally, in your own words — there's no need to "
    "overthink it. This isn't a live-graded test; everything is reviewed "
    "afterward by our recruiting team. Let's get started."
)


_QUERY_INTENTS = (
    INTENT_REPEAT, INTENT_CLARIFICATION, INTENT_CANDIDATE_QUESTION, INTENT_OFF_TOPIC, INTENT_UNCLEAR,
)


class TurnAction(str, Enum):
    SPEAK_COMPANY_INTRO = "speak_company_intro"
    SPEAK_INTRO = "speak_intro"
    SPEAK_QUESTION = "speak_question"
    SPEAK_FOLLOW_UP = "speak_follow_up"
    AWAIT_ANSWER = "await_answer"
    LISTEN_AGAIN = "listen_again"      # nothing to say: just reopen the answer window (echo dropped)
    CLOSE_INTERVIEW = "close_interview"
    ERROR = "error"


class InterviewFSM:
    """
    Drives one InterviewSession through:
      company intro (fixed, not scored)
      -> candidate self-intro request (question #1, "intro" tag)
      -> question loop (with optional single follow-up per question)
      -> silent close (natural completion, proctoring termination, or
         candidate-initiated early end).

    Every spoken question/follow-up has the candidate's first name
    prefixed at speak-time (not stored in the DB) so it doesn't read as
    a generic script — see _personalize().

    IMPORTANT — question progression is driven by whether an Answer
    exists for a question, NOT by Question.asked_at. asked_at is set
    the moment a question is *spoken*, which always happens before any
    answer comes in — so it can never be used to detect "has this
    question been answered yet". See submit_answer().
    """

    def __init__(self, session: InterviewSession, candidate_profile: dict):
        self.session = session
        self.candidate_profile = candidate_profile

    # ------------------------------------------------------------------
    # Entry point: start the interview
    # ------------------------------------------------------------------

    def start(self) -> dict:
        if self.session.status != InterviewSession.Status.PENDING:
            logger.warning(
                "start() called on session %s with status %s — ignoring.",
                self.session.id,
                self.session.status,
            )
            return self._current_turn()

        try:
            generate_questions_for_session(self.session)
        except QuestionGenerationError as exc:
            logger.error("Failed to start session %s: %s", self.session.id, exc)
            return {
                "action": TurnAction.ERROR,
                "message": (
                    "We're having trouble setting up your interview right "
                    "now. Please try rejoining in a moment."
                ),
            }

        self.session.status = InterviewSession.Status.IN_PROGRESS
        self.session.started_at = timezone.now()
        self.session.save(update_fields=["status", "started_at"])

        self._prefetch_speech()

        return {
            "action": TurnAction.SPEAK_COMPANY_INTRO,
            "text": COMPANY_INTRO_TEXT,
        }

    def _prefetch_speech(self) -> None:
        """
        Synthesizes the company intro and every question (exactly as they will
        be spoken, name included) in the background at session start, so each
        plays immediately instead of waiting for TTS on the CPU. The "repeat" /
        "off-topic" variants of a question are NOT made here (that tripled the
        start-up CPU load and delayed the first real sentence); they are made
        when that question is actually spoken (see _prefetch_variants).
        Best effort only.
        """
        try:
            from apps.avatar.services.tts_prefetch import prefetch_in_background

            question_texts = [
                self._personalize(q.text) for q in self.session.questions.order_by("order")
            ]
            prefetch_in_background([COMPANY_INTRO_TEXT] + question_texts)
        except Exception as exc:
            logger.warning("TTS prefetch could not start for session %s: %s", self.session.id, exc)

    def _prefetch_variants(self, spoken_text: str) -> None:
        """
        While a question is being spoken, pre-make the lines used if the
        candidate asks to repeat it or goes off-topic. "Of course." alone is
        under 40 chars and gets MERGED into the next sentence by the chunker,
        so the combined text is what must be warm.
        """
        try:
            from apps.avatar.services.tts_prefetch import prefetch_in_background

            prefetch_in_background([
                f"{REPEAT_REPLY} {spoken_text}",
                f"{OFF_TOPIC_REPLY} Let's continue — {spoken_text}",
            ])
        except Exception as exc:
            logger.warning("TTS variant prefetch failed for session %s: %s", self.session.id, exc)

    def advance_to_first_question(self) -> dict:
        if self.session.status != InterviewSession.Status.IN_PROGRESS:
            logger.warning(
                "advance_to_first_question called on session %s with "
                "status %s — ignoring.",
                self.session.id,
                self.session.status,
            )
            return self._current_turn()

        intro_question = self._get_question_by_order(1)
        if intro_question is None:
            return {
                "action": TurnAction.ERROR,
                "message": "No questions were generated for this session.",
            }

        if intro_question.asked_at is not None:
            logger.warning(
                "advance_to_first_question called again for session %s "
                "but question order=1 was already spoken at %s — "
                "ignoring duplicate call.",
                self.session.id,
                intro_question.asked_at,
            )
            return {"action": TurnAction.AWAIT_ANSWER}       # no-op for the frontend

        return self._speak_question_turn(intro_question, is_intro=True)

    # ------------------------------------------------------------------
    # Called after every candidate answer (post-STT, post-analysis)
    # ------------------------------------------------------------------

    def submit_answer(self, question_id, transcript_text: str, audio_ref: str = "") -> dict:
        if not self.session.is_active:
            logger.warning(
                "submit_answer called on inactive session %s (status=%s)",
                self.session.id,
                self.session.status,
            )
            return self._current_turn()

        question = self._get_question(question_id)
        if question is None:
            return {
                "action": TurnAction.ERROR,
                "message": "Unrecognized question — please rejoin the interview.",
            }

        transcript_text = (transcript_text or "").strip()
        last_spoken = self._last_spoken_text(question)
        is_first_answer_to_question = not question.has_answer

        # What gets analysed / stored: on a follow-up, the first answer + the
        # follow-up (never overwrite the first answer).
        full_transcript = build_full_transcript(question, transcript_text)
        analysis_kwargs = dict(
            question_text=question.text,
            topic_tag=question.topic_tag,
            transcript_text=full_transcript,
            candidate_profile=self.candidate_profile,
            is_white_collar=self.session.is_white_collar,
        )

        # --- Latency: the intent check and the answer analysis are two
        # independent LLM calls. When the reply needs an LLM intent check,
        # both run AT THE SAME TIME (the analysis is speculative and simply
        # unused if the reply turns out to be a question / repeat request).
        pool = None
        analysis_future = None
        intent = classify_fast(transcript_text)
        if intent is None:
            pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="turn")
            intent_future = pool.submit(classify_reply_llm, last_spoken, transcript_text)
            analysis_future = pool.submit(run_answer_analysis, **analysis_kwargs)
            intent = intent_future.result()
        elif intent.intent == INTENT_ANSWER:
            analysis_future = None            # computed below, nothing to overlap with
        logger.info(
            "Session %s question order=%d: reply intent=%s (source=%s)",
            self.session.id, question.order, intent.intent, intent.source,
        )

        try:
            if intent.intent == INTENT_ABUSIVE:
                return self.terminate_for_abuse(question, transcript_text, intent)

            if intent.intent in _QUERY_INTENTS:
                turn = self._reply_and_reask_turn(question, transcript_text, intent)
                if turn is not None:
                    return turn
                # Limit reached for this question: treat the reply as the answer.

            if analysis_future is not None:
                analysis = analysis_future.result()
            else:
                analysis = run_answer_analysis(**analysis_kwargs)
        finally:
            if pool is not None:
                pool.shutdown(wait=False)     # an unused speculative analysis just finishes in the background

        answer = save_answer(question, full_transcript, analysis, audio_ref=audio_ref)

        if is_first_answer_to_question:
            self.session.questions_asked_count += 1
            self.session.save(update_fields=["questions_asked_count"])
            logger.info(
                "Session %s: first answer recorded for question order=%d "
                "(id=%s) — questions_asked_count now %d.",
                self.session.id,
                question.order,
                question.id,
                self.session.questions_asked_count,
            )
        else:
            logger.info(
                "Session %s: follow-up answer recorded for question "
                "order=%d (id=%s) — questions_asked_count unchanged (%d).",
                self.session.id,
                question.order,
                question.id,
                self.session.questions_asked_count,
            )

        turn = self._decide_next_turn(question, answer)

        # The answer ended with a side question ("...and what are the next
        # rounds?") — answer it briefly before the next question.
        if intent.aside_reply and turn.get("text"):
            turn["text"] = f"{intent.aside_reply} {turn['text']}"
        return turn

    # ------------------------------------------------------------------
    # Reconnect: re-send whatever the candidate should be hearing right now
    # ------------------------------------------------------------------

    def resume_turn(self) -> dict:
        """
        Called when the candidate's socket (re)joins an IN_PROGRESS session.
        Re-speaks the question / follow-up that is currently open so the
        interview continues instead of dead-ending after a network drop.
        """
        last_asked = (
            self.session.questions.filter(asked_at__isnull=False).order_by("-order").first()
        )
        if last_asked is None:
            return {"action": TurnAction.SPEAK_COMPANY_INTRO, "text": COMPANY_INTRO_TEXT}

        if last_asked.has_answer:
            answer = last_asked.answer
            follow_up_count = last_asked.generation_context.get("follow_up_count", 0)
            if not (
                answer.analysis_json.get("should_follow_up")
                and follow_up_count < MAX_FOLLOW_UPS_PER_QUESTION
            ):
                return self._advance_to_next_question_or_close()

        return {
            "action": TurnAction.SPEAK_FOLLOW_UP,
            "question_id": str(last_asked.id),
            "text": f"Sorry, the connection dropped for a moment. {self._last_spoken_text(last_asked)}",
        }

    def no_speech_turn(self, question_id) -> dict | None:
        """
        The browser heard the candidate talking but Whisper returned nothing
        (noise, a very quiet voice, a cough). Ask once more instead of scoring
        silence as the answer. Returns None when the per-question limit of
        such re-asks is used up (the caller then continues normally).
        """
        question = self._get_question(question_id)
        if question is None:
            return None
        intent = TurnIntent(INTENT_UNCLEAR, reply="Sorry, I didn't catch that.", source="empty")
        return self._reply_and_reask_turn(question, "", intent)

    def repeat_turn(self, question_id) -> dict:
        """
        Re-speaks the open question without scoring anything. Used when an
        interruption turned out to contain no speech (cough, background noise).
        """
        question = self._get_question(question_id)
        if question is None:
            return {"action": TurnAction.ERROR, "message": "Unrecognized question — please rejoin the interview."}
        return {
            "action": TurnAction.SPEAK_FOLLOW_UP,
            "question_id": str(question.id),
            "text": self._last_spoken_text(question),
        }

    # ------------------------------------------------------------------
    # Candidate asked to repeat / clarify / asked their own question
    # ------------------------------------------------------------------

    def _reply_and_reask_turn(self, question: Question, transcript_text: str, intent: TurnIntent) -> dict | None:
        """
        Speaks a short formal reply and gives the SAME question back to the
        candidate (same question_id, so the frontend records the answer to
        it as usual). Not scored and not counted as an answer. Every such
        exchange is kept in question.generation_context["candidate_queries"]
        for the recruiter. Returns None once the per-question limit is hit.
        """
        queries = question.generation_context.setdefault("candidate_queries", [])

        # "Repeat" and "couldn't hear you" are not the candidate stalling, so they
        # do not use up the 3 real questions (a hard total cap still applies).
        real_queries = [q for q in queries if q.get("intent") not in (INTENT_REPEAT, INTENT_UNCLEAR)]
        is_soft = intent.intent in (INTENT_REPEAT, INTENT_UNCLEAR)
        over_limit = (
            len(queries) >= MAX_CANDIDATE_QUERIES_TOTAL
            or (not is_soft and len(real_queries) >= MAX_CANDIDATE_QUERIES_PER_QUESTION)
        )
        if over_limit:
            if not question.generation_context.get("queries_closed"):
                # First time over the limit: say so politely and ask for the
                # answer. The reply after this one IS taken as the answer —
                # a question is never scored as an answer.
                question.generation_context["queries_closed"] = True
                question.save(update_fields=["generation_context"])
                logger.info(
                    "Session %s question order=%d: candidate query limit reached — asking for the answer.",
                    self.session.id, question.order,
                )
                return {
                    "action": TurnAction.SPEAK_FOLLOW_UP,
                    "question_id": str(question.id),
                    "text": f"{QUERY_LIMIT_REPLY} {self._last_spoken_text(question)}",
                }
            logger.info(
                "Session %s question order=%d: candidate query limit already used — "
                "treating reply as the answer.",
                self.session.id, question.order,
            )
            return None

        queries.append({
            "intent": intent.intent,
            "transcript": (transcript_text or "")[:1000],
            "reply": intent.reply,
            "at": timezone.now().isoformat(),
        })
        question.save(update_fields=["generation_context"])

        last_text = self._last_spoken_text(question)
        reply = intent.reply.strip()

        if intent.intent == INTENT_REPEAT:
            text = f"{reply or 'Of course.'} {last_text}"
        elif intent.intent == INTENT_UNCLEAR:
            text = f"{reply or UNCLEAR_REPLY} Could you please answer again? {last_text}"
        elif intent.intent == INTENT_OFF_TOPIC:
            text = f"{reply or OFF_TOPIC_REPLY} Let's continue — {last_text}"
        elif intent.intent == INTENT_CLARIFICATION:
            text = reply or f"Let me put it another way. {last_text}"
        else:  # INTENT_CANDIDATE_QUESTION
            fallback = (
                "That's a fair question. The recruiting team will share those "
                "details with you after this interview."
            )
            text = f"{reply or fallback} Now, coming back to the question: {last_text}"

        return {
            "action": TurnAction.SPEAK_FOLLOW_UP,
            "question_id": str(question.id),
            "text": text.strip(),
        }

    def _last_spoken_text(self, question: Question) -> str:
        """The exact question / follow-up text the candidate last heard."""
        return question.generation_context.get("last_spoken_text") or self._personalize(question.text)

    # ------------------------------------------------------------------
    # Abusive language: end the interview immediately
    # ------------------------------------------------------------------

    def terminate_for_abuse(self, question: Question, transcript_text: str, intent: TurnIntent) -> dict:
        """
        Records an "abusive_language" proctoring flag (with the evidence and
        transcript, for the recruiter), stores the reply as this question's
        answer, and closes the session immediately. The returned turn has
        an internal "_proctoring_terminated" key: the interview consumer
        removes it and tells the proctoring socket to show the
        "interview ended" screen.
        """
        # Imported here: risk_scorer imports this module at load time.
        from apps.proctoring.models import ProctoringFlag, ProctoringSession
        from apps.proctoring.services.risk_scorer import (
            _update_risk_score,
            get_or_create_proctoring_session,
        )

        now = timezone.now()
        logger.warning(
            "Session %s: abusive language detected (source=%s, evidence=%r) — terminating.",
            self.session.id, intent.source, intent.evidence,
        )

        with transaction.atomic():
            get_or_create_proctoring_session(self.session)
            proctoring_session = ProctoringSession.objects.select_for_update().get(
                interview_session=self.session
            )
            ProctoringFlag.objects.create(
                proctoring_session=proctoring_session,
                flag_type=ProctoringFlag.FlagType.ABUSIVE_LANGUAGE,
                severity=ProctoringFlag.Severity.HIGH,
                counted_toward_violation=True,
                metadata={
                    "source": f"answer_{intent.source}",
                    "evidence": intent.evidence,
                    "question_id": str(question.id),
                    "transcript_excerpt": (transcript_text or "")[:500],
                },
                occurred_at=now,
            )
            proctoring_session.countable_violation_count += 1
            proctoring_session.terminated = True
            proctoring_session.terminated_at = now
            proctoring_session.save(update_fields=[
                "countable_violation_count", "terminated", "terminated_at", "updated_at",
            ])
            _update_risk_score(proctoring_session)

            Answer.objects.update_or_create(
                question=question,
                defaults={
                    "transcript_text": transcript_text or "",
                    "audio_ref": "",
                    "analysis_json": {
                        "relevance": None,
                        "depth": None,
                        "communication": None,
                        "should_follow_up": False,
                        "follow_up_reason": "",
                        "notes": f"Interview terminated: abusive language ({intent.evidence}).",
                        "red_flags": ["abusive_language"],
                    },
                    "score": None,
                    "answered_at": now,
                    "analyzed_at": now,
                },
            )

        turn = self._close_session(
            reason=InterviewSession.TerminationReason.ABUSIVE_LANGUAGE,
            status=InterviewSession.Status.TERMINATED,
            note=ABUSE_TERMINATION_MESSAGE,
        )
        turn["_proctoring_terminated"] = {
            "message": ABUSE_TERMINATION_MESSAGE,
            "flag_type": ProctoringFlag.FlagType.ABUSIVE_LANGUAGE.value,
        }
        return turn

    # ------------------------------------------------------------------
    # Called by the consumer when the candidate clicks "End interview"
    # ------------------------------------------------------------------

    def end_interview_by_candidate(self) -> dict:
        """
        Candidate explicitly ended the interview early. Closes the
        session with whatever was answered so far and still triggers
        final rating generation on the partial transcript.

        Distinct from terminate_for_violation: reason is
        CANDIDATE_ENDED, not PROCTORING_VIOLATION — the dashboard should
        treat these very differently (this is not a red flag on the
        candidate).
        """
        if self.session.status in (
            InterviewSession.Status.COMPLETED,
            InterviewSession.Status.TERMINATED,
        ):
            logger.info(
                "end_interview_by_candidate called on session %s already "
                "in terminal status=%s — returning existing close turn.",
                self.session.id,
                self.session.status,
            )
            return {
                "action": TurnAction.CLOSE_INTERVIEW,
                "message": self.session.termination_note or "This interview has already ended.",
            }

        logger.info("Session %s ended early by candidate.", self.session.id)

        return self._close_session(
            reason=InterviewSession.TerminationReason.CANDIDATE_ENDED,
            status=InterviewSession.Status.TERMINATED,
            note=(
                "You ended the interview early. Thank you for your time "
                "— our team will review your responses so far and follow "
                "up separately."
            ),
        )

    # ------------------------------------------------------------------
    # Core decision logic
    # ------------------------------------------------------------------

    def _decide_next_turn(self, question: Question, answer: Answer) -> dict:
        should_follow_up = answer.analysis_json.get("should_follow_up", False)
        follow_up_count = question.generation_context.get("follow_up_count", 0)

        can_still_follow_up = (
            should_follow_up and follow_up_count < MAX_FOLLOW_UPS_PER_QUESTION
        )

        if can_still_follow_up:
            return self._speak_follow_up_turn(question, answer)

        return self._advance_to_next_question_or_close()

    def _advance_to_next_question_or_close(self) -> dict:
        if self.session.has_reached_question_limit:
            return self._close_session(
                reason=InterviewSession.TerminationReason.NONE,
                status=InterviewSession.Status.COMPLETED,
            )

        next_order = self.session.questions_asked_count + 1
        attempts = 0

        while attempts < MAX_ADVANCE_ATTEMPTS:
            next_question = self._get_question_by_order(next_order)

            if next_question is None:
                logger.warning(
                    "Session %s ran out of questions at order %d "
                    "(max_questions=%d) — closing.",
                    self.session.id,
                    next_order,
                    self.session.max_questions,
                )
                return self._close_session(
                    reason=InterviewSession.TerminationReason.NONE,
                    status=InterviewSession.Status.COMPLETED,
                )

            if next_question.asked_at is None:
                return self._speak_question_turn(next_question, is_intro=False)

            logger.error(
                "Session %s: question order=%d (id=%s) was already "
                "asked at %s but was about to be re-served — this "
                "indicates a stale questions_asked_count or a "
                "duplicate-order Question row. Skipping to order=%d "
                "instead of repeating it.",
                self.session.id,
                next_question.order,
                next_question.id,
                next_question.asked_at,
                next_order + 1,
            )
            next_order += 1
            attempts += 1

        logger.error(
            "Session %s: hit MAX_ADVANCE_ATTEMPTS (%d) while looking for "
            "an unasked question starting from order=%d — closing session "
            "to avoid an infinite/looping interview instead of guessing "
            "further.",
            self.session.id,
            MAX_ADVANCE_ATTEMPTS,
            self.session.questions_asked_count + 1,
        )
        return self._close_session(
            reason=InterviewSession.TerminationReason.NONE,
            status=InterviewSession.Status.COMPLETED,
            note=(
                "That's the end of the interview. Thank you for your "
                "time — our team will follow up with you separately."
            ),
        )

    # ------------------------------------------------------------------
    # Turn builders
    # ------------------------------------------------------------------

    def _speak_question_turn(self, question: Question, is_intro: bool) -> dict:
        text = self._personalize(question.text)
        question.asked_at = timezone.now()
        question.generation_context["last_spoken_text"] = text
        question.save(update_fields=["asked_at", "generation_context"])
        self._prefetch_variants(text)

        return {
            "action": TurnAction.SPEAK_INTRO if is_intro else TurnAction.SPEAK_QUESTION,
            "question_id": str(question.id),
            "text": text,
        }

    def _speak_follow_up_turn(self, question: Question, answer: Answer) -> dict:
        follow_up_count = question.generation_context.get("follow_up_count", 0)
        question.generation_context["follow_up_count"] = follow_up_count + 1

        text = self._personalize(self._build_follow_up_text(question, answer))
        question.generation_context["last_spoken_text"] = text
        question.save(update_fields=["generation_context"])

        return {
            "action": TurnAction.SPEAK_FOLLOW_UP,
            "question_id": str(question.id),
            "text": text,
        }

    def _build_follow_up_text(self, question: Question, answer: Answer) -> str:
        reason = answer.analysis_json.get("follow_up_reason", "").strip()
        if reason:
            return f"Could you expand on that a bit? {reason}"
        return "Could you say a bit more about that?"

    def _personalize(self, text: str) -> str:
        first_name = self._get_first_name()
        if not first_name:
            return text
        if not text:
            return text
        first_word = text.split()[0].strip(",.:;!?").lower()
        if first_word in _LOWERCASE_STARTERS:
            text = text[0].lower() + text[1:]
        return f"{first_name}, {text}"

    def _get_first_name(self) -> str:
        full_name = self.candidate_profile.get("name", "").strip()
        if not full_name:
            return ""
        return full_name.split()[0]

    def _close_session(self, reason: str, status: str, note: str = "") -> dict:
        self.session.status = status
        self.session.termination_reason = reason
        self.session.termination_note = note
        self.session.ended_at = timezone.now()
        self.session.save(
            update_fields=[
                "status",
                "termination_reason",
                "termination_note",
                "ended_at",
            ]
        )

        logger.info(
            "Session %s closed: status=%s reason=%s",
            self.session.id,
            status,
            reason,
        )

        from apps.interviews.tasks import generate_final_rating
        session_id = str(self.session.id)
        transaction.on_commit(lambda: generate_final_rating.delay(session_id))

        default_message = (
            "That's the end of the interview. Thank you for your time "
            "— our team will follow up with you separately."
        )

        return {
            "action": TurnAction.CLOSE_INTERVIEW,
            "message": note or default_message,
        }

    # ------------------------------------------------------------------
    # Called by the proctoring app on a terminating violation
    # ------------------------------------------------------------------

    def terminate_for_violation(self, explanation: str, reason: str = "") -> dict:
        if self.session.status == InterviewSession.Status.TERMINATED:
            logger.info(
                "terminate_for_violation called on already-terminated "
                "session %s — returning existing close turn.",
                self.session.id,
            )
            return {
                "action": TurnAction.CLOSE_INTERVIEW,
                "message": self.session.termination_note or explanation,
            }

        if not self.session.is_active:
            logger.warning(
                "terminate_for_violation called on session %s with "
                "status=%s (not in_progress) — terminating anyway.",
                self.session.id,
                self.session.status,
            )

        logger.info("Terminating session %s for proctoring violation.", self.session.id)

        return self._close_session(
            reason=reason or InterviewSession.TerminationReason.PROCTORING_VIOLATION,
            status=InterviewSession.Status.TERMINATED,
            note=explanation,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_question(self, question_id) -> Question | None:
        try:
            question_id = uuid.UUID(str(question_id))
        except (ValueError, AttributeError, TypeError):
            return None
        try:
            return self.session.questions.filter(id=question_id).first()
        except ValidationError:
            return None

    def _get_question_by_order(self, order: int) -> Question | None:
        return self.session.questions.filter(order=order).first()

    def _current_turn(self) -> dict:
        return {
            "action": TurnAction.ERROR,
            "message": f"Session is in state '{self.session.status}', no turn to advance.",
        }
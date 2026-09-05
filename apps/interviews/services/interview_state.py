# apps/interviews/services/interview_state.py — FULL UPDATED FILE

import logging
from enum import Enum

from django.utils import timezone

from apps.interviews.models import Answer, InterviewSession, Question
from apps.interviews.services.answer_analyzer import analyze_answer
from apps.interviews.services.question_generator import (
    QuestionGenerationError,
    generate_questions_for_session,
)

logger = logging.getLogger(__name__)

MAX_FOLLOW_UPS_PER_QUESTION = 1

MAX_ADVANCE_ATTEMPTS = 25

COMPANY_INTRO_TEXT = (
    "Hi, and welcome. I'm the AI interviewer from White Force. Before we "
    "begin, a quick word about what happens today: I'll ask you a series "
    "of questions based on your background and the role you've applied "
    "for. Just answer naturally, in your own words — there's no need to "
    "overthink it. This isn't a live-graded test; everything is reviewed "
    "afterward by our recruiting team. Let's get started."
)


class TurnAction(str, Enum):
    SPEAK_COMPANY_INTRO = "speak_company_intro"
    SPEAK_INTRO = "speak_intro"
    SPEAK_QUESTION = "speak_question"
    SPEAK_FOLLOW_UP = "speak_follow_up"
    AWAIT_ANSWER = "await_answer"
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

        return {
            "action": TurnAction.SPEAK_COMPANY_INTRO,
            "text": COMPANY_INTRO_TEXT,
        }

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
            return self._current_turn()

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

        is_first_answer_to_question = not question.has_answer

        answer = analyze_answer(
            question=question,
            transcript_text=transcript_text,
            candidate_profile=self.candidate_profile,
            audio_ref=audio_ref,
        )

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

        return self._decide_next_turn(question, answer)

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
        question.asked_at = timezone.now()
        question.save(update_fields=["asked_at"])

        return {
            "action": TurnAction.SPEAK_INTRO if is_intro else TurnAction.SPEAK_QUESTION,
            "question_id": str(question.id),
            "text": self._personalize(question.text),
        }

    def _speak_follow_up_turn(self, question: Question, answer: Answer) -> dict:
        follow_up_count = question.generation_context.get("follow_up_count", 0)
        question.generation_context["follow_up_count"] = follow_up_count + 1
        question.save(update_fields=["generation_context"])

        follow_up_text = self._build_follow_up_text(question, answer)

        return {
            "action": TurnAction.SPEAK_FOLLOW_UP,
            "question_id": str(question.id),
            "text": self._personalize(follow_up_text),
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
        return f"{first_name}, {text[0].lower()}{text[1:]}" if text else text

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
        generate_final_rating.delay(str(self.session.id))

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

    def terminate_for_violation(self, explanation: str) -> dict:
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
            reason=InterviewSession.TerminationReason.PROCTORING_VIOLATION,
            status=InterviewSession.Status.TERMINATED,
            note=explanation,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_question(self, question_id) -> Question | None:
        return self.session.questions.filter(id=question_id).first()

    def _get_question_by_order(self, order: int) -> Question | None:
        return self.session.questions.filter(order=order).first()

    def _current_turn(self) -> dict:
        return {
            "action": TurnAction.ERROR,
            "message": f"Session is in state '{self.session.status}', no turn to advance.",
        }
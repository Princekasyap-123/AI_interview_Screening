# apps/interviews/services/interview_state.py

import logging
from enum import Enum

from django.utils import timezone
from apps.interviews.tasks import generate_final_rating

from apps.interviews.models import Answer, InterviewSession, Question
from apps.interviews.services.answer_analyzer import analyze_answer
from apps.interviews.services.question_generator import (
    QuestionGenerationError,
    generate_questions_for_session,
)

logger = logging.getLogger(__name__)

# Cap on follow-ups per question so a vague answer can't loop forever.
# One follow-up is enough to mimic a real interviewer probing once,
# without turning the session into an interrogation on a single topic.
MAX_FOLLOW_UPS_PER_QUESTION = 1


class TurnAction(str, Enum):
    """
    What the consumer should do next, returned by advance(). The consumer
    is dumb on purpose — it just renders/speaks whatever this FSM tells
    it to, it never decides interview logic itself.
    """
    SPEAK_INTRO = "speak_intro"
    SPEAK_QUESTION = "speak_question"
    SPEAK_FOLLOW_UP = "speak_follow_up"
    AWAIT_ANSWER = "await_answer"
    CLOSE_INTERVIEW = "close_interview"
    ERROR = "error"


class InterviewFSM:
    """
    Drives one InterviewSession through: intro -> question loop
    (with optional single follow-up per question) -> silent close.

    This class holds no long-lived state itself — session state lives in
    the DB (InterviewSession.questions_asked_count, Question rows,
    Answer rows) so it survives websocket reconnects. Each call to a
    method re-derives "where are we" from the DB.
    """

    def __init__(self, session: InterviewSession, candidate_profile: dict):
        self.session = session
        self.candidate_profile = candidate_profile

    # ------------------------------------------------------------------
    # Entry point: start the interview
    # ------------------------------------------------------------------

    def start(self) -> dict:
        """
        Call once when the candidate joins and the session should begin.
        Generates the question set and returns the intro turn.
        """
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
            logger.error(
                "Failed to start session %s: %s", self.session.id, exc
            )
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

        intro_question = self._get_question_by_order(1)
        return self._speak_question_turn(intro_question, is_intro=True)

    # ------------------------------------------------------------------
    # Called after every candidate answer (post-STT, post-analysis)
    # ------------------------------------------------------------------

    def submit_answer(self, question_id, transcript_text: str, audio_ref: str = "") -> dict:
        """
        Records + analyzes the candidate's answer to `question_id`, then
        decides the next turn: follow-up, next question, or close.

        This is the core of steps 3-7: analyse the answer, silently
        decide what happens next, never reveal scoring to the candidate.
        """
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

        answer = analyze_answer(
            question=question,
            transcript_text=transcript_text,
            candidate_profile=self.candidate_profile,
            audio_ref=audio_ref,
        )

        if question.asked_at is None:
            # First time this question was answered (not a follow-up
            # re-analysis) — count it toward the session's question budget.
            self.session.questions_asked_count += 1
            self.session.save(update_fields=["questions_asked_count"])

        return self._decide_next_turn(question, answer)

    # ------------------------------------------------------------------
    # Called by the proctoring app on a terminating violation
    # ------------------------------------------------------------------

    def terminate_for_violation(self, explanation: str) -> dict:
        """
        Public entry point for the proctoring app to force-close an
        active session after a repeated violation (step 11/12: second
        tab-switch/fullscreen-exit after a warning).

        Unlike a normal completion, this always returns a CLOSE_INTERVIEW
        turn with the candidate-facing explanation attached — the
        consumer must speak/display `message` verbatim so the candidate
        understands why the session ended, per the agreed UX (warning on
        1st violation, explained termination on 2nd).

        Safe to call even if the session isn't IN_PROGRESS (e.g. a race
        between two flag events) — it's idempotent and just returns the
        current close turn if already terminated.
        """
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

        logger.info(
            "Terminating session %s for proctoring violation.",
            self.session.id,
        )

        return self._close_session(
            reason=InterviewSession.TerminationReason.PROCTORING_VIOLATION,
            status=InterviewSession.Status.TERMINATED,
            note=explanation,
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
        next_question = self._get_question_by_order(next_order)

        if next_question is None:
            # We've run out of generated questions before hitting
            # max_questions (e.g. LLM returned fewer than requested).
            # Close gracefully rather than erroring out on the candidate.
            logger.warning(
                "Session %s ran out of questions at order %d (max_questions=%d)",
                self.session.id,
                next_order,
                self.session.max_questions,
            )
            return self._close_session(
                reason=InterviewSession.TerminationReason.NONE,
                status=InterviewSession.Status.COMPLETED,
            )

        return self._speak_question_turn(next_question, is_intro=False)

    # ------------------------------------------------------------------
    # Turn builders
    # ------------------------------------------------------------------

    def _speak_question_turn(self, question: Question, is_intro: bool) -> dict:
        question.asked_at = timezone.now()
        question.save(update_fields=["asked_at"])

        return {
            "action": TurnAction.SPEAK_INTRO if is_intro else TurnAction.SPEAK_QUESTION,
            "question_id": str(question.id),
            "text": question.text,
        }

    def _speak_follow_up_turn(self, question: Question, answer: Answer) -> dict:
        follow_up_count = question.generation_context.get("follow_up_count", 0)
        question.generation_context["follow_up_count"] = follow_up_count + 1
        question.save(update_fields=["generation_context"])

        follow_up_text = self._build_follow_up_text(question, answer)

        return {
            "action": TurnAction.SPEAK_FOLLOW_UP,
            "question_id": str(question.id),
            "text": follow_up_text,
        }

    def _build_follow_up_text(self, question: Question, answer: Answer) -> str:
        """
        Simple templated follow-up for now — asks the candidate to expand,
        using the reason the analyzer flagged. Keeping this templated
        rather than another LLM call keeps latency low for the most
        time-sensitive moment in the loop (candidate is waiting live).
        A future version could route this through llm_client.generate_fast
        for a more natural, context-specific follow-up if latency allows.
        """
        reason = answer.analysis_json.get("follow_up_reason", "").strip()
        if reason:
            return f"Could you expand on that a bit? {reason}"
        return "Could you say a bit more about that?"

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

        generate_final_rating.delay(str(self.session.id))

        default_message = (
            "That's the end of the interview. Thank you for your time "
            "— our team will follow up with you separately."
        )

        # Step 6/7: no result is disclosed here. The consumer must not
        # attach any score/analysis to this response — it's structurally
        # absent from this dict on purpose.
        return {
            "action": TurnAction.CLOSE_INTERVIEW,
            "message": note or default_message,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_question(self, question_id) -> Question | None:
        return self.session.questions.filter(id=question_id).first()

    def _get_question_by_order(self, order: int) -> Question | None:
        return self.session.questions.filter(order=order).first()

    def _current_turn(self) -> dict:
        """Fallback used when start()/submit_answer() are called out of
        sequence — reports current status rather than guessing a turn."""
        return {
            "action": TurnAction.ERROR,
            "message": f"Session is in state '{self.session.status}', no turn to advance.",
        }


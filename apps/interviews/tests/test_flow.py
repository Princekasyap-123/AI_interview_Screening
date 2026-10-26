# apps/interviews/tests/test_flow.py
#
# Regression tests for the interview turn flow: the bugs fixed in this
# change set plus the conversational behaviour (candidate asks something ->
# interviewer answers -> question continues). The LLM is always mocked.
#
# Run:  DJANGO_SECRET_KEY=x DB_ENGINE=sqlite python manage.py test apps.interviews

import time
from unittest import mock

from channels.routing import URLRouter
from channels.testing import WebsocketCommunicator
from django.test import TestCase, TransactionTestCase, override_settings

from apps.candidates.models import Candidate
from apps.interviews.consumers import InterviewConsumer
from apps.interviews.models import Answer, InterviewSession, Question
from apps.interviews.routing import websocket_urlpatterns
from apps.interviews.services import answer_analyzer as aa
from apps.interviews.services import turn_intent as ti
from apps.interviews.services.abuse_filter import find_abusive_terms
from apps.interviews.services.echo_guard import is_echo
from apps.interviews.services.speech_cleanup import collapse_repeats, looks_unfinished
from apps.avatar.services.tts_engine import normalize_for_tts
from apps.interviews.services.interview_state import InterviewFSM, TurnAction

PROFILE = {"name": "Mili Test", "skills": [], "experience_years": "3", "recent_projects": [], "role_level": "mid"}
GOOD = {"relevance": 7, "depth": 6, "communication": 8, "should_follow_up": False,
        "follow_up_reason": "", "notes": "", "red_flags": []}
QUESTIONS = ["Please introduce yourself.", "How do SQL joins work?", "Explain database indexes."]


def make_session(status="in_progress"):
    cand = Candidate.objects.create(name="Mili Test", email="t@example.com")
    s = InterviewSession.objects.create(candidate=cand, max_questions=3, status=status)
    for i, t in enumerate(QUESTIONS, 1):
        Question.objects.create(session=s, order=i, text=t, topic_tag="tech")
    return s


class AbuseFilterTests(TestCase):
    def test_no_false_positives(self):
        for t in ["I worked in Nigeria for two years", "Nigerian clients", "fire retardant coating",
                  "Randi is my colleague", "I studied at Lund University", "the Niger river"]:
            self.assertEqual(find_abusive_terms(t), [], t)

    def test_catches_real_abuse(self):
        for t in ["you are an asshole", "fuuuck you", "what a retard", "maaderchod", "teri maa ki",
                  "nigger", "arsehole", "f*** this"]:
            self.assertTrue(find_abusive_terms(t), t)


class AnalyzerTests(TestCase):
    def setUp(self):
        self.s = make_session()
        self.q1 = self.s.questions.get(order=1)

    def test_malformed_llm_reply_falls_back_instead_of_raising(self):
        with mock.patch.object(aa.llm_client, "generate_json", return_value={"relevance": 7}):
            a = aa.analyze_answer(self.q1, "I used Django", PROFILE)
        self.assertIn("analysis_failed", a.analysis_json["red_flags"])

    def test_followup_answer_is_appended_not_overwritten(self):
        with mock.patch.object(aa.llm_client, "generate_json", return_value=GOOD):
            aa.analyze_answer(self.q1, "First answer about Django.", PROFILE)
            aa.analyze_answer(self.q1, "More about Celery.", PROFILE)
        text = Answer.objects.get(question=self.q1).transcript_text
        self.assertIn("First answer about Django.", text)
        self.assertIn("[Follow-up] More about Celery.", text)

    def test_empty_transcript_makes_no_llm_call(self):
        with mock.patch.object(aa.llm_client, "generate_json") as llm:
            a = aa.analyze_answer(self.q1, "   ", PROFILE)
        llm.assert_not_called()
        self.assertEqual(a.score, 0)


class IntentTests(TestCase):
    def test_off_topic_is_a_valid_intent(self):
        with mock.patch.object(ti.llm_client, "generate_json", return_value={"intent": "off_topic"}):
            r = ti.classify_reply("Tell me about yourself", "what is the weather today")
        self.assertEqual(r.intent, ti.INTENT_OFF_TOPIC)

    def test_unclear_is_a_valid_intent(self):
        with mock.patch.object(ti.llm_client, "generate_json", return_value={"intent": "unclear"}):
            r = ti.classify_reply("Tell me about yourself", "wait one minute I am in an interview")
        self.assertEqual(r.intent, ti.INTENT_UNCLEAR)

    def test_obvious_repeat_needs_no_llm(self):
        with mock.patch.object(ti.llm_client, "generate_json") as llm:
            for t in ["Sorry, can you repeat that?", "Pardon?", "Could you say that again please"]:
                self.assertEqual(ti.classify_reply("q", t).intent, ti.INTENT_REPEAT, t)
        llm.assert_not_called()


class FSMTests(TestCase):
    def setUp(self):
        self.s = make_session()
        self.q1 = self.s.questions.get(order=1)
        self.fsm = InterviewFSM(self.s, PROFILE)
        self.patches = [
            mock.patch("apps.avatar.services.tts_prefetch.prefetch_in_background"),
            mock.patch("apps.interviews.tasks.generate_final_rating.delay"),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def test_personalize_keeps_acronyms_and_I(self):
        self.assertEqual(self.fsm._personalize("SQL joins kaise kaam karte hain?"), "Mili, SQL joins kaise kaam karte hain?")
        self.assertEqual(self.fsm._personalize("I would like to know."), "Mili, I would like to know.")
        self.assertEqual(self.fsm._personalize("Tell me about Django."), "Mili, tell me about Django.")

    def test_invalid_question_id_returns_error_turn(self):
        turn = self.fsm.submit_answer("not-a-uuid", "hello")
        self.assertEqual(turn["action"], TurnAction.ERROR)

    def test_empty_answer_moves_on_without_llm_scoring(self):
        with mock.patch.object(aa.llm_client, "generate_json") as llm:
            turn = self.fsm.submit_answer(self.q1.id, "")
        llm.assert_not_called()
        self.assertEqual(turn["action"], TurnAction.SPEAK_QUESTION)

    def test_candidate_question_gets_reply_then_same_question(self):
        """The conversational core: ask something mid-interview -> interviewer
        replies, then continues the SAME question; nothing is scored."""
        intent = ti.TurnIntent(ti.INTENT_CANDIDATE_QUESTION,
                               reply="The recruiting team shares salary details later.", source="llm")
        with mock.patch("apps.interviews.services.interview_state.classify_fast", return_value=None), \
             mock.patch("apps.interviews.services.interview_state.classify_reply_llm", return_value=intent), \
             mock.patch("apps.interviews.services.interview_state.run_answer_analysis", return_value=GOOD):
            turn = self.fsm.submit_answer(self.q1.id, "What is the salary for this role?")
        self.assertEqual(turn["question_id"], str(self.q1.id))
        self.assertTrue(turn["text"].startswith("The recruiting team shares salary details later."))
        self.assertIn("coming back to the question", turn["text"])
        self.assertIn("introduce yourself", turn["text"])
        self.assertFalse(Answer.objects.filter(question=self.q1).exists())
        self.s.refresh_from_db()
        self.assertEqual(self.s.questions_asked_count, 0)

    def test_unclear_reply_asks_again(self):
        intent = ti.TurnIntent(ti.INTENT_UNCLEAR, reply=ti.UNCLEAR_REPLY, source="llm")
        with mock.patch("apps.interviews.services.interview_state.classify_fast", return_value=None), \
             mock.patch("apps.interviews.services.interview_state.classify_reply_llm", return_value=intent), \
             mock.patch("apps.interviews.services.interview_state.run_answer_analysis", return_value=GOOD):
            turn = self.fsm.submit_answer(self.q1.id, "hold on, someone is at the door")
        self.assertIn("answer again", turn["text"])
        self.assertEqual(turn["question_id"], str(self.q1.id))
        self.assertFalse(Answer.objects.filter(question=self.q1).exists())

    def test_intent_check_and_analysis_run_in_parallel(self):
        def slow_intent(*a, **k):
            time.sleep(0.6)
            return ti.TurnIntent()

        def slow_analysis(**k):
            time.sleep(0.6)
            return dict(GOOD)

        with mock.patch("apps.interviews.services.interview_state.classify_fast", return_value=None), \
             mock.patch("apps.interviews.services.interview_state.classify_reply_llm", side_effect=slow_intent), \
             mock.patch("apps.interviews.services.interview_state.run_answer_analysis", side_effect=slow_analysis):
            t0 = time.perf_counter()
            turn = self.fsm.submit_answer(self.q1.id, "I have three years of Django experience.")
            elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 1.0, f"took {elapsed:.2f}s - calls ran one after another")
        self.assertEqual(turn["action"], TurnAction.SPEAK_QUESTION)

    def test_resume_turn_respeaks_open_question(self):
        self.s.status = InterviewSession.Status.IN_PROGRESS
        self.fsm._speak_question_turn(self.q1, is_intro=True)
        turn = self.fsm.resume_turn()
        self.assertEqual(turn["question_id"], str(self.q1.id))
        self.assertIn("connection dropped", turn["text"])
        self.assertIn("introduce yourself", turn["text"])

    def test_duplicate_ready_is_a_noop_not_an_error(self):
        self.fsm.advance_to_first_question()
        again = self.fsm.advance_to_first_question()
        self.assertEqual(again["action"], TurnAction.AWAIT_ANSWER)

    def test_final_rating_is_queued_only_after_commit(self):
        with mock.patch("apps.interviews.tasks.generate_final_rating.delay") as delay:
            with self.captureOnCommitCallbacks(execute=False) as callbacks:
                self.fsm.end_interview_by_candidate()
            delay.assert_not_called()
            for cb in callbacks:
                cb()
            delay.assert_called_once()


@override_settings(CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}})
class ConsumerTests(TransactionTestCase):
    def _app(self):
        return URLRouter(websocket_urlpatterns)

    async def _connect(self, session_id, token):
        comm = WebsocketCommunicator(self._app(), f"/ws/interview/{session_id}/?token={token}")
        ok, _ = await comm.connect()
        return comm, ok

    async def test_wrong_token_is_rejected(self):
        from asgiref.sync import sync_to_async
        s = await sync_to_async(make_session)()
        comm, ok = await self._connect(s.id, "wrong")
        self.assertFalse(ok)

    async def test_ping_pong_and_resume_on_join(self):
        from asgiref.sync import sync_to_async
        s = await sync_to_async(make_session)()
        comm, ok = await self._connect(s.id, s.access_token)
        self.assertTrue(ok)
        await comm.send_json_to({"type": "ping"})
        self.assertEqual((await comm.receive_json_from())["type"], "pong")

        # The interview is running and question 1 was asked: a (re)join re-speaks it.
        def ask_first():
            q = s.questions.get(order=1)
            InterviewFSM(s, PROFILE)._speak_question_turn(q, is_intro=True)
        await sync_to_async(ask_first)()
        await comm.send_json_to({"type": "join"})
        msg = await comm.receive_json_from()
        self.assertEqual(msg["type"], "turn")
        self.assertIn("introduce yourself", msg["text"])
        await comm.disconnect()


@override_settings(CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}})
class ConversationOverSocketTests(TransactionTestCase):
    """The whole path the browser uses: audio in -> STT -> intent -> spoken turn out."""

    async def _open(self):
        from asgiref.sync import sync_to_async
        s = await sync_to_async(make_session)()

        def ask_first():
            InterviewFSM(s, PROFILE)._speak_question_turn(s.questions.get(order=1), is_intro=True)
        await sync_to_async(ask_first)()
        comm = WebsocketCommunicator(URLRouter(websocket_urlpatterns), f"/ws/interview/{s.id}/?token={s.access_token}")
        ok, _ = await comm.connect()
        self.assertTrue(ok)
        q1 = await sync_to_async(lambda: str(s.questions.get(order=1).id))()
        return s, comm, q1

    def _audio(self, q1, **extra):
        import base64
        return {"type": "answer_submitted", "question_id": q1,
                "audio_base64": base64.b64encode(b"fake-audio").decode(), "audio_format": "webm", **extra}

    async def test_question_in_the_middle_gets_a_reply_then_the_question_continues(self):
        s, comm, q1 = await self._open()
        intent = ti.TurnIntent(ti.INTENT_CANDIDATE_QUESTION, reply="Salary details come from the recruiting team.", source="llm")
        with mock.patch("apps.interviews.consumers.stt_client.transcribe", return_value="What is the salary?"), \
             mock.patch("apps.interviews.services.interview_state.classify_fast", return_value=None), \
             mock.patch("apps.interviews.services.interview_state.classify_reply_llm", return_value=intent), \
             mock.patch("apps.interviews.services.interview_state.run_answer_analysis", return_value=GOOD):
            await comm.send_json_to(self._audio(q1, interjected=True, reason="silence"))
            ack = await comm.receive_json_from()
            self.assertEqual(ack["type"], "ack")
            turn = await comm.receive_json_from()
        self.assertEqual(turn["type"], "turn")
        self.assertEqual(turn["question_id"], q1)
        self.assertTrue(turn["text"].startswith("Salary details come from the recruiting team."))
        self.assertIn("introduce yourself", turn["text"])
        await comm.disconnect()

    async def test_noise_after_speech_asks_again_instead_of_scoring_nothing(self):
        from asgiref.sync import sync_to_async
        s, comm, q1 = await self._open()
        with mock.patch("apps.interviews.consumers.stt_client.transcribe", return_value=""):
            await comm.send_json_to(self._audio(q1, reason="silence"))
            await comm.receive_json_from()                 # ack
            turn = await comm.receive_json_from()
        self.assertEqual(turn["question_id"], q1)
        self.assertIn("didn't catch that", turn["text"])
        self.assertFalse(await sync_to_async(Answer.objects.filter(question_id=q1).exists)())
        await comm.disconnect()

    async def test_empty_interruption_just_repeats_the_question(self):
        s, comm, q1 = await self._open()
        with mock.patch("apps.interviews.consumers.stt_client.transcribe", return_value=""):
            for _ in range(4):        # after the re-ask limit it falls back to a plain repeat
                await comm.send_json_to(self._audio(q1, interjected=True, reason="silence"))
                await comm.receive_json_from()
                turn = await comm.receive_json_from()
                self.assertEqual(turn["question_id"], q1)
                self.assertIn("introduce yourself", turn["text"])
        await comm.disconnect()


class PhoneRegressionTests(TestCase):
    """The ngrok/phone session: 'tell me about yourself' got a nonsense reply and the
    4th question was scored as an answer."""

    def setUp(self):
        self.s = make_session()
        self.s.status = InterviewSession.Status.IN_PROGRESS
        self.fsm = InterviewFSM(self.s, PROFILE)
        self.q1 = self.s.questions.get(order=1)
        self.fsm._speak_question_turn(self.q1, is_intro=True)

    # --- fixed replies, no LLM, no role confusion ---
    def test_tell_me_about_yourself_gets_a_fixed_interviewer_reply(self):
        for said in ["Tell me about yourself.", "Can you tell me about yourself?", "Who are you?"]:
            with mock.patch("apps.interviews.services.turn_intent.llm_client") as llm:
                intent = ti.classify_fast(said)
                llm.generate_json.assert_not_called()
            self.assertEqual(intent.intent, ti.INTENT_CANDIDATE_QUESTION, said)
            self.assertEqual(intent.reply, ti.ABOUT_AI_REPLY)

    def test_company_name_question_gets_a_fixed_reply(self):
        intent = ti.classify_fast("From which company? Tell me about your company name.")
        self.assertEqual(intent.intent, ti.INTENT_CANDIDATE_QUESTION)
        self.assertIn("White Force", intent.reply)

    def test_real_answers_are_not_mistaken_for_questions_about_the_ai(self):
        answer = ("I am a backend developer with three years of experience, and if you want me to tell me about "
                  "yourself style summary I can say I built APIs with Django and Celery for logistics clients.")
        intent = ti.classify_fast(answer)
        self.assertTrue(intent is None or intent.intent == ti.INTENT_ANSWER)
        self.assertNotEqual(getattr(intent, "reply", ""), ti.ABOUT_AI_REPLY)

    def test_llm_reply_that_asks_the_candidate_a_question_is_rejected(self):
        bad = "Could you clarify what you would like to hear about my background?"
        self.assertEqual(ti.sanitize_reply(bad, self.q1.text), "")

    def test_llm_reply_that_copies_the_interview_question_is_rejected(self):
        q = "To get started, could you please introduce yourself and walk me through your background"
        self.assertEqual(ti.sanitize_reply("Could you please introduce yourself and walk me through your background.", q), "")

    def test_good_reply_passes(self):
        r = "The recruiting team will share those details after the interview."
        self.assertEqual(ti.sanitize_reply(r, self.q1.text), r)

    # --- the query limit no longer scores a question as an answer ---
    def test_query_limit_asks_for_the_answer_and_never_scores_a_question(self):
        q_intent = ti.TurnIntent(ti.INTENT_CANDIDATE_QUESTION, reply="Noted.", source="keywords")
        with mock.patch("apps.interviews.services.interview_state.classify_fast", return_value=q_intent):
            for _ in range(3):
                turn = self.fsm.submit_answer(self.q1.id, "What is the salary?")
                self.assertIn("introduce yourself", turn["text"])
            turn = self.fsm.submit_answer(self.q1.id, "And the company name?")     # 4th question
        self.assertIn("not able to take more questions", turn["text"])
        self.assertIn("introduce yourself", turn["text"])
        self.assertFalse(Answer.objects.filter(question=self.q1).exists())
        # the reply AFTER that is taken as the answer, whatever it is
        with mock.patch("apps.interviews.services.interview_state.classify_fast", return_value=q_intent), \
             mock.patch("apps.interviews.services.interview_state.run_answer_analysis", return_value=GOOD):
            self.fsm.submit_answer(self.q1.id, "And what about leaves?")
        self.assertTrue(Answer.objects.filter(question=self.q1).exists())

    def test_repeat_requests_do_not_use_up_the_question_allowance(self):
        rep = ti.TurnIntent(ti.INTENT_REPEAT, reply=ti.REPEAT_REPLY, source="keywords")
        with mock.patch("apps.interviews.services.interview_state.classify_fast", return_value=rep):
            for _ in range(5):
                turn = self.fsm.submit_answer(self.q1.id, "sorry, say that again")
                self.assertTrue(turn["text"].startswith("Of course."))
        q_intent = ti.TurnIntent(ti.INTENT_CANDIDATE_QUESTION, reply="Noted.", source="keywords")
        with mock.patch("apps.interviews.services.interview_state.classify_fast", return_value=q_intent):
            turn = self.fsm.submit_answer(self.q1.id, "What is the salary?")
        self.assertEqual(turn["text"].split()[0], "Noted.")        # 5 repeats earlier, but this question is still allowed

    def test_hard_total_cap_stops_endless_repeat_loops(self):
        rep = ti.TurnIntent(ti.INTENT_REPEAT, reply=ti.REPEAT_REPLY, source="keywords")
        with mock.patch("apps.interviews.services.interview_state.classify_fast", return_value=rep):
            for _ in range(6):
                self.fsm.submit_answer(self.q1.id, "again")
            turn = self.fsm.submit_answer(self.q1.id, "again")
        self.assertIn("not able to take more questions", turn["text"])


class EchoGuardTests(TestCase):
    SPOKEN = ["Of course. To get started, could you please introduce yourself - walk me through your "
              "background, your current role, and what you've been working on recently?"]

    def test_copy_of_the_interviewers_words_is_an_echo(self):
        self.assertTrue(is_echo("walk me through your background your current role", self.SPOKEN))
        self.assertTrue(is_echo("Could you please introduce yourself.", self.SPOKEN))

    def test_real_answers_and_short_replies_are_not(self):
        self.assertFalse(is_echo("Hi, I'm Mili, I have three years of Django experience.", self.SPOKEN))
        self.assertFalse(is_echo("yes sir", self.SPOKEN))
        self.assertFalse(is_echo("Tell me about yourself", self.SPOKEN))
        self.assertFalse(is_echo("My current role is backend developer and I work with Django", self.SPOKEN))

    def test_nothing_spoken_yet(self):
        self.assertFalse(is_echo("could you please introduce yourself", []))


@override_settings(CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}})
class EchoOverSocketTests(TransactionTestCase):
    LONG_Q = ("To get started, could you please introduce yourself and walk me through "
              "your background, your current role and what you have been working on recently?")

    async def _open(self):
        from asgiref.sync import sync_to_async
        s = await sync_to_async(make_session)()

        def ask_first():
            q = s.questions.get(order=1)
            q.text = self.LONG_Q
            q.save()
            InterviewFSM(s, PROFILE)._speak_question_turn(q, is_intro=True)
        await sync_to_async(ask_first)()
        comm = WebsocketCommunicator(URLRouter(websocket_urlpatterns), f"/ws/interview/{s.id}/?token={s.access_token}")
        ok, _ = await comm.connect()
        self.assertTrue(ok)
        q1 = await sync_to_async(lambda: str(s.questions.get(order=1).id))()
        return s, comm, q1

    def _audio(self, q1, **extra):
        import base64
        return {"type": "answer_submitted", "question_id": q1,
                "audio_base64": base64.b64encode(b"fake-audio").decode(), "audio_format": "webm", **extra}

    async def test_echoed_question_is_not_scored_and_the_mic_just_keeps_listening(self):
        from asgiref.sync import sync_to_async
        s, comm, q1 = await self._open()
        # the consumer remembers what it sent; send the first question again like a join/resume would
        await comm.send_json_to({"type": "join"})
        spoken = await comm.receive_json_from()
        with mock.patch("apps.interviews.consumers.stt_client.transcribe",
                        return_value="walk me through your background your current role and what you have been working on"):
            await comm.send_json_to(self._audio(q1, reason="silence"))
            self.assertEqual((await comm.receive_json_from())["type"], "ack")
            turn = await comm.receive_json_from()
        self.assertEqual(turn["action"], "listen_again")
        self.assertFalse(await sync_to_async(Answer.objects.filter(question_id=q1).exists)())
        await comm.disconnect()

    async def test_echo_after_a_barge_in_repeats_the_question(self):
        s, comm, q1 = await self._open()
        await comm.send_json_to({"type": "join"})
        await comm.receive_json_from()
        with mock.patch("apps.interviews.consumers.stt_client.transcribe",
                        return_value="walk me through your background your current role and what you have been working on"):
            await comm.send_json_to(self._audio(q1, interjected=True, reason="silence"))
            await comm.receive_json_from()
            turn = await comm.receive_json_from()
        self.assertEqual(turn["action"], "speak_follow_up")
        self.assertIn("introduce yourself", turn["text"])
        await comm.disconnect()


class SpeechCleanupTests(TestCase):
    def test_whisper_loop_is_collapsed_and_flagged(self):
        text, looped = collapse_repeats("I am you actually I am you actually I am you actually")
        self.assertEqual(text, "I am you actually")
        self.assertTrue(looped)

    def test_a_sentence_said_twice_is_collapsed_but_not_a_loop(self):
        text, looped = collapse_repeats("I don't know anything about that. I don't know anything about that.")
        self.assertEqual(text, "I don't know anything about that.")
        self.assertFalse(looped)

    def test_normal_speech_is_untouched(self):
        for t in ["I have worked with Django and Celery for two years.", "very very good", "no no no", "yes"]:
            self.assertEqual(collapse_repeats(t), (t, False))

    def test_unfinished_sentences(self):
        for t in ["I have worked with Django and", "I built the API, ", "Because", "I worked on the",
                  "So I started with", "my role was um", "I use Python and..."]:
            self.assertTrue(looks_unfinished(t), t)

    def test_finished_sentences(self):
        for t in ["I am a Java developer.", "I don't know that.", "Yes, I can.", "I think so.",
                  "I worked on a payments system.", "What is the salary?", "Yes", "No"]:
            self.assertFalse(looks_unfinished(t), t)

    def test_tts_text_is_plain_ascii_punctuation(self):
        out = normalize_for_tts("Walk\u2011through what you\u2019ve built \u2014 briefly\u2026")
        self.assertEqual(out, "Walk-through what you've built, briefly...")


@override_settings(CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}})
class MidSentenceTests(EchoOverSocketTests):
    """Stopping in the middle of a sentence keeps the interviewer listening; the parts are joined."""

    async def test_unfinished_answer_is_continued_and_joined(self):
        s, comm, q1 = await self._open()
        seen = {}

        def fake_submit(self_fsm, question_id, transcript_text, audio_ref=""):
            seen["text"] = transcript_text
            return {"action": TurnAction.SPEAK_QUESTION, "question_id": str(question_id), "text": "Next."}

        with mock.patch("apps.interviews.consumers.stt_client.transcribe",
                        side_effect=["I have worked with Django and", "Celery for background jobs."]), \
             mock.patch.object(InterviewFSM, "submit_answer", fake_submit):
            await comm.send_json_to(self._audio(q1, reason="silence"))
            await comm.receive_json_from()                       # ack
            turn = await comm.receive_json_from()
            self.assertEqual(turn["action"], "listen_again")
            self.assertTrue(turn["continue"])
            self.assertNotIn("text", seen)                       # nothing was scored yet

            await comm.send_json_to(self._audio(q1, reason="silence"))
            await comm.receive_json_from()
            turn = await comm.receive_json_from()
        self.assertEqual(seen["text"], "I have worked with Django and Celery for background jobs.")
        self.assertEqual(turn["text"], "Next.")
        await comm.disconnect()

    async def test_silence_after_an_unfinished_part_submits_what_was_said(self):
        s, comm, q1 = await self._open()
        seen = {}

        def fake_submit(self_fsm, question_id, transcript_text, audio_ref=""):
            seen["text"] = transcript_text
            return {"action": TurnAction.SPEAK_QUESTION, "question_id": str(question_id), "text": "Next."}

        with mock.patch("apps.interviews.consumers.stt_client.transcribe", side_effect=["I worked on the", ""]), \
             mock.patch.object(InterviewFSM, "submit_answer", fake_submit):
            await comm.send_json_to(self._audio(q1, reason="silence"))
            await comm.receive_json_from(); await comm.receive_json_from()
            await comm.send_json_to(self._audio(q1, reason="no_answer"))
            await comm.receive_json_from()
            await comm.receive_json_from()
        self.assertEqual(seen["text"], "I worked on the")
        await comm.disconnect()

# apps/interviews/tests/test_language.py
#
# English / Hindi interviews and the "another voice" proctoring flag.
# No network: the LLM, STT and TTS are always mocked.
#
# Run:  DJANGO_SECRET_KEY=x DB_ENGINE=sqlite python manage.py test apps.interviews.tests.test_language

import base64
import json
from unittest import mock

from asgiref.sync import sync_to_async
from channels.routing import URLRouter
from channels.testing import WebsocketCommunicator
from django.test import Client, TestCase, TransactionTestCase, override_settings
from django.utils import timezone

from apps.candidates.models import Candidate
from apps.interviews.models import InterviewSession, Question
from apps.interviews.routing import websocket_urlpatterns
from apps.interviews.services import language as lang
from apps.interviews.services.interview_state import InterviewFSM
from apps.proctoring.models import ProctoringFlag, ProctoringSession
from apps.proctoring.services.flag_processor import process_flag, reason_for

PROFILE = {"name": "Mili Test", "skills": [], "experience_years": "3", "recent_projects": [], "role_level": "mid"}
GOOD = {"relevance": 7, "depth": 6, "communication": 8, "should_follow_up": False,
        "follow_up_reason": "", "notes": "", "red_flags": []}
QUESTIONS = ["Please introduce yourself.", "How do SQL joins work?", "Explain database indexes."]
HINDI = "मेरे पास तीन साल का अनुभव है"
ENGLISH = "I have three years of experience with Django"


def fake_to_hindi(text):
    return "नमस्ते। " + text            # any Devanagari-heavy text will do


def make_session(mode="en", current=None):
    cand = Candidate.objects.create(name="Mili Test", email=f"{mode}@example.com")
    s = InterviewSession.objects.create(
        candidate=cand, max_questions=3, status="in_progress", language_mode=mode,
        current_language=current or lang.initial_language(mode),
    )
    for i, t in enumerate(QUESTIONS, 1):
        Question.objects.create(session=s, order=i, text=t, topic_tag="tech")
    return s


class DetectLanguageTests(TestCase):
    d = staticmethod(lang.detect_language)

    def test_english_only_never_switches(self):
        self.assertEqual(self.d(HINDI, "hindi", "en", "en"), "en")

    def test_hindi_only_never_switches(self):
        self.assertEqual(self.d(ENGLISH, "english", "hi", "hi"), "hi")

    def test_both_switches_to_hindi_on_devanagari_or_whisper_hindi(self):
        self.assertEqual(self.d(HINDI, "hindi", "both", "en"), "hi")
        self.assertEqual(self.d("mere paas teen saal ka anubhav hai", "hindi", "both", "en"), "hi")

    def test_both_switches_back_to_english(self):
        self.assertEqual(self.d(ENGLISH, "english", "both", "hi"), "en")

    def test_one_word_answers_do_not_flip_the_language(self):
        self.assertEqual(self.d("yes", "english", "both", "hi"), "hi")
        self.assertEqual(self.d("ठीक", "hindi", "both", "en"), "en")      # too short to judge

    def test_a_single_english_term_inside_hindi_keeps_hindi(self):
        self.assertEqual(self.d("मैंने Django पर काम किया है", "hindi", "both", "hi"), "hi")


class TranslationHelpersTests(TestCase):
    def test_candidate_is_addressed_gender_neutrally_but_the_interviewer_keeps_her_forms(self):
        out = lang._neutralise("क्या आप बता सकती हैं? मैं पूछूँगी, मैं सुन पाई।")
        self.assertIn("सकते हैं", out)
        self.assertIn("पूछूँगी", out)
        self.assertIn("सुन पाई", out)

    def test_fixed_lines_need_no_llm_and_prefixes_are_reused(self):
        import apps.interviews.services.interview_state  # noqa: F401  (registers the fixed Hindi lines)
        with mock.patch.object(lang, "_llm_translate", return_value="प्रश्न।") as llm:
            self.assertEqual(lang.to_hindi("Of course."), "ज़रूर।")
            llm.assert_not_called()
            self.assertEqual(lang.to_hindi("Of course. A brand new question?"), "ज़रूर। प्रश्न।")
            llm.assert_called_once_with(" A brand new question?".strip())

    def test_failed_translation_returns_none(self):
        with mock.patch.object(lang, "_llm_translate", return_value=None):
            self.assertIsNone(lang.to_hindi("Something never translated before 12345."))

    def test_hindi_input_is_returned_untouched(self):
        self.assertEqual(lang.to_hindi(HINDI), HINDI)


class BackgroundVoiceFlagTests(TestCase):
    def setUp(self):
        cand = Candidate.objects.create(name="P", email="p@example.com")
        s = InterviewSession.objects.create(candidate=cand, status="in_progress")
        self.ps = ProctoringSession.objects.create(interview_session=s)

    def _flag(self, seconds_ago):
        return process_flag(
            self.ps, "background_voice", {"voiced_ms": 3000},
            occurred_at=timezone.now() - timezone.timedelta(seconds=seconds_ago),
        )

    def test_it_is_a_known_flag_with_a_candidate_message(self):
        self.assertIn("background_voice", ProctoringFlag.FlagType.values)
        self.assertIn("voice", reason_for("background_voice").lower())

    def test_it_counts_as_a_violation_and_is_high_severity(self):
        r = self._flag(0)
        self.assertTrue(r.is_countable)
        self.assertFalse(r.is_terminating)
        self.assertEqual(r.flag.severity, "high")

    def test_the_fourth_confirmed_one_ends_the_interview(self):
        results = [self._flag(60 - 20 * i) for i in range(4)]       # 20 s apart: clear of the cooldowns
        self.assertEqual([r.is_countable for r in results], [True] * 4)
        self.assertEqual([r.is_terminating for r in results], [False, False, False, True])

    def test_a_repeat_within_the_cooldown_is_logged_but_not_counted(self):
        self._flag(30)
        self.assertFalse(self._flag(29).is_countable)


class TTSLanguageTests(TestCase):
    def _post(self, session, **body):
        return Client().post("/api/avatar/tts/", data=json.dumps({"session_id": str(session.id), **body}),
                             content_type="application/json")

    def test_hindi_requests_use_the_hindi_voice_and_default_stays_english(self):
        s = make_session("both")
        with mock.patch("apps.avatar.views.tts_engine.synthesize", return_value=b"RIFFwav") as synth:
            self.assertEqual(self._post(s, text="नमस्ते", lang="hi").status_code, 200)
            self.assertEqual(synth.call_args.kwargs["lang"], "hi")
            self._post(s, text="Hello")
            self.assertEqual(synth.call_args.kwargs["lang"], "en")
            self._post(s, text="Hello", lang="<script>")
            self.assertEqual(synth.call_args.kwargs["lang"], "en")

    def test_hindi_audio_is_cached_separately_from_english(self):
        from apps.avatar.services.tts_engine import tts_engine
        self.assertNotEqual(tts_engine._cache_path("x", tts_engine._voice_for("en")),
                            tts_engine._cache_path("x", tts_engine._voice_for("hi")))

    def test_hindi_sentences_split_at_the_danda(self):
        from apps.avatar.services.tts_prefetch import split_chunks
        text = ("नमस्ते, मैं व्हाइट फोर्स की इंटरव्यूअर हूँ और आज आपसे बात करूँगी। "
                "कृपया अपना परिचय दीजिए और अपने अनुभव के बारे में बताइए।")
        self.assertEqual(len(split_chunks(text)), 2)


@override_settings(CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}})
class LanguageOverSocketTests(TransactionTestCase):
    async def _open(self, mode, current=None):
        s = await sync_to_async(make_session)(mode, current)

        def ask_first():
            InterviewFSM(s, PROFILE)._speak_question_turn(s.questions.get(order=1), is_intro=True)
        await sync_to_async(ask_first)()
        comm = WebsocketCommunicator(URLRouter(websocket_urlpatterns), f"/ws/interview/{s.id}/?token={s.access_token}")
        ok, _ = await comm.connect()
        self.assertTrue(ok)
        q1 = await sync_to_async(lambda: str(s.questions.get(order=1).id))()
        return s, comm, q1

    @staticmethod
    def _audio(q1, **extra):
        return {"type": "answer_submitted", "question_id": q1,
                "audio_base64": base64.b64encode(b"fake-audio").decode(), "audio_format": "webm", **extra}

    async def _answer(self, comm, q1, transcript, whisper_language, mode_call="detailed"):
        target = "transcribe_detailed" if mode_call == "detailed" else "transcribe"
        value = (transcript, whisper_language) if mode_call == "detailed" else transcript
        msgs = []
        with mock.patch(f"apps.interviews.consumers.stt_client.{target}", return_value=value), \
             mock.patch("apps.interviews.consumers.to_hindi", side_effect=fake_to_hindi), \
             mock.patch("apps.interviews.consumers.warm_hindi_in_background"), \
             mock.patch("apps.interviews.services.interview_state.run_answer_analysis", return_value=GOOD), \
             mock.patch("apps.interviews.services.interview_state.classify_fast",
                        return_value=mock.Mock(intent="answer", aside_reply="", source="fast")):
            await comm.send_json_to(self._audio(q1, reason="silence"))
            msgs.append(await comm.receive_json_from())          # ack
            msgs.append(await comm.receive_json_from())          # language (if switched) or the turn
            if msgs[-1]["type"] != "turn":
                msgs.append(await comm.receive_json_from())
        return msgs

    async def test_english_only_interview_is_unchanged(self):
        s, comm, q1 = await self._open("en")
        msgs = await self._answer(comm, q1, HINDI, "hindi", mode_call="plain")   # even Hindi speech: stays English
        turn = msgs[-1]
        self.assertEqual([m["type"] for m in msgs], ["ack", "turn"])
        self.assertEqual(turn["language"], "en")
        self.assertIn("SQL joins", turn["text"])
        await comm.disconnect()

    async def test_both_mode_switches_to_hindi_when_the_candidate_speaks_hindi_and_back(self):
        s, comm, q1 = await self._open("both")
        msgs = await self._answer(comm, q1, HINDI, "hindi")
        self.assertEqual([m["type"] for m in msgs], ["ack", "language", "turn"])
        self.assertEqual(msgs[1]["language"], "hi")
        turn = msgs[2]
        self.assertEqual(turn["language"], "hi")
        self.assertTrue(turn["text"].startswith("नमस्ते।"))
        self.assertIn("SQL joins", turn["text"])                   # the (fake-)translated question
        self.assertEqual(await sync_to_async(lambda: InterviewSession.objects.get(pk=s.pk).current_language)(), "hi")

        q2 = turn["question_id"]
        msgs = await self._answer(comm, q2, ENGLISH, "english")
        self.assertEqual([m["type"] for m in msgs], ["ack", "language", "turn"])
        self.assertEqual(msgs[1]["language"], "en")
        self.assertEqual(msgs[2]["language"], "en")
        self.assertNotIn("नमस्ते", msgs[2]["text"])
        self.assertEqual(await sync_to_async(lambda: InterviewSession.objects.get(pk=s.pk).current_language)(), "en")
        await comm.disconnect()

    async def test_the_whole_conversation_is_stored_in_the_language_it_was_held_in(self):
        from apps.interviews.models import ConversationTurn
        s, comm, q1 = await self._open("both")
        await self._answer(comm, q1, HINDI, "hindi")                    # candidate speaks Hindi -> switch
        await comm.disconnect()

        def rows():
            return [(t.speaker, t.language, t.text, t.text_en) for t in ConversationTurn.objects.filter(session=s)]
        got = await sync_to_async(rows)()
        speakers = [r[0] for r in got]
        self.assertEqual(speakers[:2], ["candidate", "interviewer"])
        self.assertEqual(got[0][:3], ("candidate", "hi", HINDI))           # what the candidate said, as heard
        interviewer_hi = got[1]
        self.assertEqual(interviewer_hi[1], "hi")
        self.assertTrue(interviewer_hi[2].startswith("नमस्ते।"))          # the Hindi line that was spoken
        self.assertIn("SQL joins", interviewer_hi[3])                     # ...with its English original kept

    async def test_both_mode_asks_whisper_to_detect_the_language(self):
        s, comm, q1 = await self._open("both")
        with mock.patch("apps.interviews.consumers.stt_client.transcribe_detailed", return_value=("", "")) as stt:
            await comm.send_json_to(self._audio(q1, reason="silence"))
            await comm.receive_json_from()
            await comm.receive_json_from()
        self.assertEqual(stt.call_args.kwargs["language"], "")
        await comm.disconnect()

    async def test_hindi_only_forces_hindi_stt_and_speaks_hindi_from_the_start(self):
        s, comm, q1 = await self._open("hi")
        with mock.patch("apps.interviews.consumers.stt_client.transcribe", return_value=ENGLISH) as stt, \
             mock.patch("apps.interviews.consumers.to_hindi", side_effect=fake_to_hindi), \
             mock.patch("apps.interviews.services.interview_state.run_answer_analysis", return_value=GOOD), \
             mock.patch("apps.interviews.services.interview_state.classify_fast",
                        return_value=mock.Mock(intent="answer", aside_reply="", source="fast")):
            await comm.send_json_to(self._audio(q1, reason="silence"))
            await comm.receive_json_from()
            turn = await comm.receive_json_from()
        self.assertEqual(stt.call_args.kwargs["language"], "hi")
        self.assertEqual(turn["type"], "turn")                      # no "language" message: it never changes
        self.assertEqual(turn["language"], "hi")
        await comm.disconnect()

    async def test_a_failed_translation_falls_back_to_the_english_voice(self):
        s, comm, q1 = await self._open("both", current="hi")
        with mock.patch("apps.interviews.consumers.stt_client.transcribe_detailed", return_value=(HINDI, "hindi")), \
             mock.patch("apps.interviews.consumers.to_hindi", return_value=None), \
             mock.patch("apps.interviews.consumers.warm_hindi_in_background"), \
             mock.patch("apps.interviews.services.interview_state.run_answer_analysis", return_value=GOOD), \
             mock.patch("apps.interviews.services.interview_state.classify_fast",
                        return_value=mock.Mock(intent="answer", aside_reply="", source="fast")):
            await comm.send_json_to(self._audio(q1, reason="silence"))
            await comm.receive_json_from()
            turn = await comm.receive_json_from()
        self.assertEqual(turn["type"], "turn")
        self.assertEqual(turn["language"], "en")                    # English text + English voice, never silence
        self.assertIn("SQL joins", turn["text"])
        await comm.disconnect()


class CreateSessionLanguageTests(TestCase):
    """The recruiter picks the language when creating a session (dashboard form and API)."""

    def setUp(self):
        from django.contrib.auth import get_user_model
        self.user = get_user_model().objects.create_user("rec", password="x", is_staff=True)
        self.client.force_login(self.user)
        self.cand = Candidate.objects.create(name="Asha Verma", email="asha@example.com")

    def _form(self, **extra):
        r = self.client.post("/dashboard/", {"candidate_id": str(self.cand.id), "job_description_text": "jd",
                                             "max_questions": "3", **extra})
        self.assertEqual(r.status_code, 302)
        return InterviewSession.objects.latest("created_at")

    def test_form_default_is_english(self):
        s = self._form()
        self.assertEqual((s.language_mode, s.current_language), ("en", "en"))

    def test_form_both_starts_in_english(self):
        s = self._form(language_mode="both")
        self.assertEqual((s.language_mode, s.current_language), ("both", "en"))

    def test_form_hindi_starts_in_hindi(self):
        s = self._form(language_mode="hi")
        self.assertEqual((s.language_mode, s.current_language), ("hi", "hi"))

    def test_form_ignores_an_unknown_language(self):
        self.assertEqual(self._form(language_mode="klingon").language_mode, "en")

    def test_api_create_accepts_language(self):
        r = self.client.post("/api/interviews/create/", data=json.dumps(
            {"candidate_id": str(self.cand.id), "job_description_text": "jd", "language_mode": "hi"}),
            content_type="application/json")
        self.assertEqual(r.status_code, 201, r.content)
        s = InterviewSession.objects.get(pk=r.json()["id"])
        self.assertEqual((s.language_mode, s.current_language), ("hi", "hi"))

    def test_dashboard_shows_the_language_selector_and_the_label(self):
        s = self._form(language_mode="both")
        html = self.client.get("/dashboard/").content.decode()
        self.assertIn('id="languageMode"', html)
        self.assertIn("English + Hindi", html)
        self.assertIn("English + Hindi", self.client.get(f"/dashboard/?session_id={s.id}").content.decode())

    def test_room_page_receives_the_language(self):
        s = self._form(language_mode="hi")
        html = self.client.get(f"/interview/{s.id}/{s.access_token}/").content.decode()
        self.assertIn('const LANGUAGE_MODE = "hi"', html)
        self.assertIn('const START_LANGUAGE = "hi"', html)

    def test_finished_interview_link_shows_the_static_page(self):
        s = self._form(language_mode="en")
        url = f"/interview/{s.id}/{s.access_token}/"
        self.assertEqual(self.client.get(url).status_code, 200)
        for status in (InterviewSession.Status.COMPLETED, InterviewSession.Status.TERMINATED):
            InterviewSession.objects.filter(pk=s.pk).update(status=status)
            r = self.client.get(url)
            self.assertEqual(r.status_code, 410)
            html = r.content.decode()
            self.assertIn("This interview is over", html)
            self.assertNotIn("rulesOverlay", html)

    def test_room_page_loads_the_silero_vad(self):
        s = self._form(language_mode="en")
        html = self.client.get(f"/interview/{s.id}/{s.access_token}/").content.decode()
        self.assertIn("silero_vad.js", html)
        self.assertIn("silero_vad.onnx", html)


class FormalAnswersTests(TestCase):
    """Common candidate questions get a formal answer without an LLM call."""

    def test_how_do_you_know_me(self):
        from apps.interviews.services.turn_intent import HOW_KNOW_REPLY, INTENT_CANDIDATE_QUESTION, classify_fast
        for text in ("how do you know me", "How did you get my number?", "why was I called for this",
                     "aap mujhe kaise jaante ho", "आप मुझे कैसे जानती हैं"):
            r = classify_fast(text)
            self.assertEqual((r.intent, r.reply), (INTENT_CANDIDATE_QUESTION, HOW_KNOW_REPLY), text)

    def test_small_talk_and_duration(self):
        from apps.interviews.services.turn_intent import (
            DURATION_REPLY, GREETING_REPLY, THANKS_REPLY, classify_fast)
        self.assertEqual(classify_fast("hello how are you").reply, GREETING_REPLY)
        self.assertEqual(classify_fast("thank you").reply, THANKS_REPLY)
        self.assertEqual(classify_fast("how long will this interview take").reply, DURATION_REPLY)

    def test_normal_answers_are_not_hijacked(self):
        from apps.interviews.services.turn_intent import classify_fast
        r = classify_fast("I know my way around python and django")
        self.assertTrue(r is None or r.intent == "answer")

    def test_new_replies_have_hindi_lines(self):
        from apps.interviews.services import interview_state  # registers the static Hindi lines
        from apps.interviews.services.turn_intent import HOW_KNOW_REPLY
        self.assertIn("शॉर्टलिस्ट", lang._cache[HOW_KNOW_REPLY])

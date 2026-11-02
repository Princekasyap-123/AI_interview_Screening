# apps/interviews/tests/test_reschedule.py

"""Candidate who is not ready: slot rules (working days/hours, holidays, one month) and the spoken flow."""

from datetime import datetime, time, timedelta
from unittest import mock

from django.test import TestCase, override_settings
from django.utils import timezone

from apps.interviews.models import InterviewSession
from apps.interviews.services import scheduling
from apps.interviews.services.interview_state import InterviewFSM
from apps.interviews.services.turn_intent import INTENT_RESCHEDULE, classify_fast
from apps.interviews.tests.test_language import PROFILE, make_session


def day_after(days, weekday_ok=True, start=None):
    """First date >= today+days that is a weekday (or a Saturday when weekday_ok=False)."""
    d = (start or timezone.localdate()) + timedelta(days=days)
    while (d.weekday() < 5) != weekday_ok:
        d += timedelta(days=1)
    return d


def at(d, hh, mm=0):
    return timezone.make_aware(datetime.combine(d, time(hh, mm)))


@override_settings(INTERVIEW_HOLIDAYS=[], INTERVIEW_WORK_START_HOUR=10, INTERVIEW_WORK_END_HOUR=18)
class SlotRulesTests(TestCase):
    def test_a_normal_working_slot_is_accepted(self):
        self.assertEqual(scheduling.validate_slot(at(day_after(3), 11))[:2], (True, "ok"))

    def test_weekend_is_refused(self):
        ok, code, reason = scheduling.validate_slot(at(day_after(3, weekday_ok=False), 11))
        self.assertEqual((ok, code), (False, "weekend"))
        self.assertIn("Monday to Friday", reason)

    def test_outside_working_hours_is_refused(self):
        for hh in (7, 9, 18, 21):
            self.assertEqual(scheduling.validate_slot(at(day_after(3), hh))[1], "hours", hh)
        self.assertTrue(scheduling.validate_slot(at(day_after(3), 10))[0])
        self.assertTrue(scheduling.validate_slot(at(day_after(3), 17, 30))[0])

    def test_past_and_too_soon_are_refused(self):
        self.assertEqual(scheduling.validate_slot(at(timezone.localdate() - timedelta(days=1), 11))[1], "past")

    def test_more_than_a_month_is_refused(self):
        far = day_after(45)
        ok, code, reason = scheduling.validate_slot(at(far, 11))
        self.assertEqual((ok, code), (False, "too_far"))
        self.assertIn("30 days", reason)

    def test_the_last_day_of_the_window_is_allowed(self):
        last = day_after(0, start=timezone.localdate() + timedelta(days=30))
        if last == timezone.localdate() + timedelta(days=30):
            self.assertTrue(scheduling.validate_slot(at(last, 11))[0])

    def test_holidays_are_refused(self):
        d = day_after(3)
        with override_settings(INTERVIEW_HOLIDAYS=[f"{d.isoformat()}:Test Day"]):
            ok, code, reason = scheduling.validate_slot(at(d, 11))
        self.assertEqual((ok, code), (False, "holiday"))
        self.assertIn("Test Day", reason)


class IntentTests(TestCase):
    def test_not_ready_is_a_reschedule_request(self):
        for text in ("I am not ready.", "I don't want to give this interview", "can we do it tomorrow",
                     "abhi nahi, baad mein karte hain", "अभी तैयार नहीं हूँ"):
            r = classify_fast(text)
            self.assertIsNotNone(r, text)
            self.assertEqual(r.intent, INTENT_RESCHEDULE, text)

    def test_an_ordinary_answer_is_not(self):
        r = classify_fast("I built a Django service and shipped it later that year without any downtime " * 3)
        self.assertTrue(r is None or r.intent != INTENT_RESCHEDULE)


@override_settings(INTERVIEW_HOLIDAYS=[], INTERVIEW_WORK_START_HOUR=10, INTERVIEW_WORK_END_HOUR=18)
class FlowTests(TestCase):
    def setUp(self):
        self.session = make_session("en")
        self.session.questions.update(asked_at=timezone.now())
        self.q = self.session.questions.get(order=1)
        self.fsm = InterviewFSM(self.session, PROFILE)

    def say(self, text, parsed=None):
        parsed = parsed or {"date": None, "time": None, "refuses": False}
        with mock.patch.object(scheduling, "parse_slot", return_value=parsed):
            return self.fsm.submit_answer(str(self.q.id), text)

    def test_not_ready_asks_for_a_slot_and_keeps_the_interview_open(self):
        turn = self.say("I am not ready.")
        self.assertEqual(turn["action"], "speak_follow_up")
        self.assertIn("working day", turn["text"])
        self.session.refresh_from_db()
        self.assertEqual(self.session.status, "in_progress")

    def test_a_valid_slot_closes_the_interview_as_rescheduled(self):
        self.say("I am not ready.")
        d = day_after(3)
        turn = self.say("Tuesday at 3", {"date": d, "time": time(15, 0), "refuses": False})
        self.assertEqual(turn["action"], "close_interview")
        self.assertIn("rescheduled", turn["message"])
        self.session.refresh_from_db()
        self.assertEqual(self.session.status, "terminated")
        self.assertEqual(self.session.termination_reason, "rescheduled")
        self.assertEqual(timezone.localtime(self.session.rescheduled_for), at(d, 15))

    def test_a_weekend_is_refused_then_a_good_day_accepted(self):
        self.say("I am not ready.")
        turn = self.say("Saturday 11", {"date": day_after(3, weekday_ok=False), "time": time(11, 0), "refuses": False})
        self.assertEqual(turn["action"], "speak_follow_up")
        self.assertIn("Monday to Friday", turn["text"])
        turn = self.say("ok", {"date": day_after(3), "time": time(11, 0), "refuses": False})
        self.assertEqual(turn["action"], "close_interview")

    def test_late_evening_and_far_dates_are_refused(self):
        self.say("I am not ready.")
        t1 = self.say("9 pm", {"date": day_after(3), "time": time(21, 0), "refuses": False})
        self.assertIn("between 10 AM and 6 PM", t1["text"])
        t2 = self.say("in 2 months", {"date": day_after(45), "time": time(11, 0), "refuses": False})
        self.assertIn("30 days", t2["text"])

    def test_date_first_then_time(self):
        self.say("I am not ready.")
        d = day_after(3)
        t1 = self.say("on that day", {"date": d, "time": None, "refuses": False})
        self.assertIn("What time", t1["text"])
        t2 = self.say("twelve", {"date": None, "time": time(12, 0), "refuses": False})
        self.assertEqual(t2["action"], "close_interview")

    def test_garbled_replies_do_not_use_up_the_real_attempts(self):
        """Hindi speech-to-text noise used to burn all three attempts before a real answer arrived."""
        self.say("I am not ready.")
        for _ in range(4):
            self.assertEqual(self.say("गुड़क ले जाता है")["action"], "speak_follow_up")
        turn = self.say("sunday", {"date": day_after(3), "time": time(11, 0), "refuses": False})
        self.assertEqual(turn["action"], "close_interview")
        self.session.refresh_from_db()
        self.assertIsNotNone(self.session.rescheduled_for)

    def test_what_the_candidate_said_is_kept_for_the_dashboard_transcript(self):
        self.say("अब मुझे इंटरव्यू नहीं रहना है।")
        self.say("एक नवंबर दुपहर दो बजे।", {"date": day_after(3, weekday_ok=False), "time": time(14, 0), "refuses": False})
        self.q.refresh_from_db()
        saved = self.q.generation_context["candidate_queries"]
        self.assertEqual([e["intent"] for e in saved], ["reschedule", "reschedule"])
        self.assertEqual(saved[0]["transcript"], "अब मुझे इंटरव्यू नहीं रहना है।")
        self.assertIn("Monday to Friday", saved[1]["reply"])
        self.assertEqual(saved[1]["transcript"], "एक नवंबर दुपहर दो बजे।")

    def test_three_misses_hand_over_to_the_recruiting_team(self):
        self.say("I am not ready.")
        for _ in range(4):
            self.assertEqual(self.say("hmm")["action"], "speak_follow_up")
        turn = self.say("hmm")
        self.assertEqual(turn["action"], "close_interview")
        self.assertIn("recruiting team", turn["message"])
        self.session.refresh_from_db()
        self.assertIsNone(self.session.rescheduled_for)

    def test_changing_their_mind_resumes_the_question(self):
        self.say("I am not ready.")
        turn = self.say("actually I'm ready now, let's continue")
        self.assertEqual(turn["action"], "speak_follow_up")
        self.assertIn("let's continue", turn["text"].lower())
        self.session.refresh_from_db()
        self.assertEqual(self.session.status, "in_progress")

    def test_a_rescheduled_link_shows_the_static_page_with_the_new_time(self):
        self.say("I am not ready.")
        d = day_after(3)
        self.say("x", {"date": d, "time": time(15, 0), "refuses": False})
        self.session.refresh_from_db()
        r = self.client.get(f"/interview/{self.session.id}/{self.session.access_token}/")
        self.assertEqual(r.status_code, 410)
        self.assertIn("rescheduled", r.content.decode().lower())


class RateLimitFallbackTests(TestCase):
    """A Groq model out of daily quota must not silently break the conversation."""

    def test_a_rate_limited_model_hands_over_to_its_sibling(self):
        import httpx
        from groq import RateLimitError
        from apps.ai_engine.llm_client import LLMClient

        c = LLMClient()
        calls = []

        class Resp:
            choices = [mock.Mock(message=mock.Mock(content='{"intent": "answer"}'), finish_reason="stop")]
            usage = None

        def create(**kw):
            calls.append(kw["model"])
            if kw["model"] == c.FAST_MODEL:
                req = httpx.Request("POST", "https://api.groq.com/x")
                raise RateLimitError("limit", response=httpx.Response(429, request=req), body=None)
            return Resp()

        fake = mock.Mock()
        fake.chat.completions.create.side_effect = create
        c._client = fake

        out = c.generate_json("s", "u", model=c.FAST_MODEL)
        self.assertEqual(out, {"intent": "answer"})
        self.assertEqual(calls, [c.FAST_MODEL, c.REASONING_MODEL])

        calls.clear()                                   # the slow model is now skipped straight away
        c.generate_json("s", "u", model=c.FAST_MODEL)
        self.assertEqual(calls, [c.REASONING_MODEL])


class DashboardLinkTests(TestCase):
    """An ended interview's link must not look like a working one."""

    def setUp(self):
        from django.contrib.auth import get_user_model
        self.client.force_login(get_user_model().objects.create_user("rec2", password="x", is_staff=True))
        self.session = make_session("en")

    def page(self, host=None):
        extra = {"HTTP_HOST": host} if host else {}
        return self.client.get(f"/dashboard/?session_id={self.session.id}", **extra).content.decode()

    def test_live_session_shows_the_copyable_link(self):
        html = self.page()
        self.assertIn('id="interviewLinkInput"', html)
        self.assertNotIn("This link has expired", html)
        self.assertNotIn("points to <strong>localhost</strong>", html)    # a public host: no warning

    def test_a_localhost_link_warns_that_phones_cannot_open_it(self):
        self.assertIn("points to <strong>localhost</strong>", self.page(host="127.0.0.1:8000"))

    def test_ended_session_says_the_link_expired_and_offers_a_new_interview(self):
        InterviewSession.objects.filter(pk=self.session.pk).update(status="terminated")
        html = self.page()
        self.assertIn("This link has expired", html)
        self.assertIn("New interview for this candidate", html)
        self.assertNotIn('id="interviewLinkInput"', html)


class DashboardConversationTests(TestCase):
    """Every line of the interview shows on the dashboard, in the language it was held in."""

    def setUp(self):
        from django.contrib.auth import get_user_model
        self.client.force_login(get_user_model().objects.create_user("rec3", password="x", is_staff=True))
        self.session = make_session("both")

    def test_hindi_and_english_lines_show_with_the_english_original(self):
        from apps.interviews.models import ConversationTurn
        q = self.session.questions.get(order=1)
        ConversationTurn.objects.create(session=self.session, speaker="interviewer", kind="question", language="hi",
                                        text="कृपया अपना परिचय दीजिए।", text_en="Please introduce yourself.", question=q)
        ConversationTurn.objects.create(session=self.session, speaker="candidate", kind="answer", language="hi",
                                        text="मैं एक पाइथन डेवलपर हूँ।", question=q)
        ConversationTurn.objects.create(session=self.session, speaker="interviewer", kind="reply", language="en",
                                        text="Thank you. Next question.")
        ConversationTurn.objects.create(session=self.session, speaker="candidate", kind="inaudible", language="en", text="")
        html = self.client.get(f"/dashboard/?session_id={self.session.id}").content.decode()
        for needle in ("कृपया अपना परिचय दीजिए।", "Please introduce yourself.", "मैं एक पाइथन डेवलपर हूँ।",
                       "Thank you. Next question.", "could not be understood", "Full conversation"):
            self.assertIn(needle, html)
        self.assertLess(html.index("कृपया अपना परिचय"), html.index("मैं एक पाइथन"))     # in conversation order

    def test_older_interviews_without_a_log_explain_why(self):
        html = self.client.get(f"/dashboard/?session_id={self.session.id}").content.decode()
        self.assertIn("recorded for interviews held after this feature was added", html)


class RescheduledDashboardTests(TestCase):
    def setUp(self):
        from django.contrib.auth import get_user_model
        self.client.force_login(get_user_model().objects.create_user("rec4", password="x", is_staff=True))
        self.session = make_session("en")
        self.slot = at(day_after(3), 14)
        InterviewSession.objects.filter(pk=self.session.pk).update(
            status="terminated", termination_reason="rescheduled", rescheduled_for=self.slot, questions_asked_count=2)

    def detail(self):
        return self.client.get(f"/dashboard/?session_id={self.session.id}").content.decode()

    def test_the_reschedule_is_the_first_thing_the_recruiter_reads(self):
        html = self.detail()
        self.assertIn("Candidate asked to reschedule", html)
        self.assertIn('class="status-pill rescheduled"', html)
        self.assertNotIn("Terminated early</span>", html.split("resched-banner")[-1][:400])
        self.assertIn("New interview for this candidate", html)
        self.assertLess(html.index('<div class="resched-banner"'), html.index('<div class="meta-grid">'))
        self.assertNotIn(">Termination reason<", html)           # not repeated lower down

    def test_the_session_card_says_rescheduled_with_the_new_slot(self):
        html = self.client.get("/dashboard/").content.decode()
        self.assertIn('status-pill rescheduled">Rescheduled', html)
        self.assertIn("New slot:", html)

    def test_no_time_agreed_is_said_plainly(self):
        InterviewSession.objects.filter(pk=self.session.pk).update(rescheduled_for=None)
        self.assertIn("No time agreed", self.detail())

    def test_nothing_answered_means_no_rating_and_says_so(self):
        InterviewSession.objects.filter(pk=self.session.pk).update(questions_asked_count=0)
        self.assertIn("asked to reschedule before answering any question", self.detail())


class TrackingStatusTests(TestCase):
    def setUp(self):
        from django.contrib.auth import get_user_model
        from apps.proctoring.models import ProctoringSession
        self.client.force_login(get_user_model().objects.create_user("rec5", password="x", is_staff=True))
        self.session = make_session("en")
        self.proc = ProctoringSession.objects.create(interview_session=self.session)

    def page(self):
        return self.client.get(f"/dashboard/?session_id={self.session.id}").content.decode()

    def test_running_checks_are_reported(self):
        self.proc.tracking_status = {"face_tracking": True, "face_frames": 900, "object_detection": True}
        self.proc.save()
        self.assertIn("Camera checks were running: 900 frames analysed", self.page())

    def test_checks_that_never_ran_are_called_out(self):
        self.proc.tracking_status = {"face_tracking": False, "face_frames": 0}
        self.proc.save()
        self.assertIn("Face tracking was not running", self.page())


class RatingFallbackTests(TestCase):
    def test_a_finished_interview_without_a_rating_gets_one_without_celery(self):
        from apps.interviews.services import final_rating as fr
        session = make_session("en")
        InterviewSession.objects.filter(pk=session.pk).update(status="terminated")
        with mock.patch.object(fr, "generate_final_rating_for_session") as gen:
            fr.generate_rating_if_missing(session.pk)
        gen.assert_called_once()

    def test_an_existing_rating_is_left_alone(self):
        from apps.interviews.models import FinalRating
        from apps.interviews.services import final_rating as fr
        session = make_session("en")
        InterviewSession.objects.filter(pk=session.pk).update(status="completed")
        FinalRating.objects.create(session=session, overall_score=50)
        with mock.patch.object(fr, "generate_final_rating_for_session") as gen:
            self.assertTrue(fr.generate_rating_if_missing(session.pk))
        gen.assert_not_called()

    def test_a_failing_rating_never_raises(self):
        from apps.interviews.services import final_rating as fr
        session = make_session("en")
        InterviewSession.objects.filter(pk=session.pk).update(status="completed")
        with mock.patch.object(fr, "generate_final_rating_for_session", side_effect=RuntimeError("llm down")):
            self.assertFalse(fr.generate_rating_if_missing(session.pk))

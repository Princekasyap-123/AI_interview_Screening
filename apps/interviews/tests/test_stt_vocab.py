# apps/interviews/tests/test_stt_vocab.py

from django.test import SimpleTestCase

from apps.interviews.services.stt_vocab import build_prompt, build_terms, correct_terms


class PromptTests(SimpleTestCase):
    def test_company_skills_and_jd_terms_are_in_the_prompt(self):
        prompt = build_prompt("Need Django, Redis and Celery experience", {"skills": ["Flask"]}, "Mili")
        for word in ("White Force", "B.Tech", "Flask", "Django", "Redis", "Celery", "Mili"):
            self.assertIn(word, prompt)

    def test_prompt_is_bounded(self):
        self.assertLessEqual(len(build_prompt("Python " * 500, {"skills": ["x" * 30] * 100})), 480)

    def test_terms_not_in_the_jd_are_left_out_when_there_is_context(self):
        terms = build_terms("We use Django, Redis, Celery, Docker, Kafka, Linux, Nginx, Git, SQL, React and AWS daily.")
        self.assertNotIn("Laravel", terms)


class CorrectionTests(SimpleTestCase):
    def test_known_mishearings_from_real_interviews(self):
        self.assertEqual(correct_terms("I did BTEC CIC at Whiteboards"), "I did B.Tech CSE at White Force")
        self.assertEqual(correct_terms("deployed olama locally"), "deployed Ollama locally")

    def test_spelling_variants(self):
        self.assertEqual(correct_terms("I completed my B, TechCSC and work here"), "I completed my B.Tech CSE and work here")
        self.assertEqual(correct_terms("a B tech degree"), "a B.Tech degree")
        self.assertEqual(correct_terms("Celery with Riddies and Alama"), "Celery with Redis and Ollama")

    def test_near_miss_of_a_vocabulary_word_is_fixed(self):
        self.assertEqual(correct_terms("we used Postgress and Dockr", ["PostgreSQL", "Postgress"]), "we used Postgress and Dockr")
        self.assertIn("Kubernetes", correct_terms("I run Kubernetis clusters", ["Kubernetes"]))

    def test_ordinary_sentences_are_untouched(self):
        text = "I worked on a team that built a reliable payment system last year."
        self.assertEqual(correct_terms(text, build_terms("Python Django Redis")), text)
        self.assertEqual(correct_terms(""), "")


class HindiPromptTests(SimpleTestCase):
    def test_hindi_and_auto_detected_speech_get_no_hint(self):
        """A hint made Whisper repeat it back as the candidate's words on short, quiet audio."""
        for language in ("hi", ""):
            self.assertEqual(build_prompt("Python Django", {"skills": ["Django"]}, "Prince", language), "")

    def test_english_speech_keeps_the_english_hint(self):
        self.assertIn("Interview with a candidate", build_prompt("Python", {}, "Prince", "en"))


class PromptEchoTests(SimpleTestCase):
    PROMPT = build_prompt("Python Django Redis", {"skills": ["Django"]}, "Prince", "en")

    def test_the_hint_read_back_is_recognised(self):
        from apps.interviews.services.stt_vocab import prompt_echo
        self.assertTrue(prompt_echo("Interview with a candidate at White Force", self.PROMPT))
        self.assertTrue(prompt_echo("interview with a candidate at white force prince", self.PROMPT))

    def test_real_answers_and_plain_term_lists_are_not_echoes(self):
        from apps.interviews.services.stt_vocab import prompt_echo
        self.assertFalse(prompt_echo("I use Django, Redis and Celery daily", self.PROMPT))
        self.assertFalse(prompt_echo("Django Redis Celery", self.PROMPT))
        self.assertFalse(prompt_echo("मैं पाइथन डेवलपर हूँ", self.PROMPT))
        self.assertFalse(prompt_echo("anything", ""))


class EdgeHindiVoiceTests(SimpleTestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        from unittest import mock
        from apps.avatar.services import tts_engine as mod

        self.mod, self.mock = mod, mock
        self.engine = mod.KokoroTTSEngine()
        self.engine.hindi_engine = "edge"
        self.engine.cache_dir = Path(tempfile.mkdtemp())

    def test_hindi_uses_the_edge_voice_and_caches_it(self):
        with self.mock.patch.object(self.engine, "_edge_fetch_wav", return_value=b"RIFFwav") as fetch:
            self.assertEqual(self.engine.synthesize("नमस्ते, आप कैसे हैं?", lang="hi"), b"RIFFwav")
            self.assertEqual(self.engine.synthesize("नमस्ते, आप कैसे हैं?", lang="hi"), b"RIFFwav")
        self.assertEqual(fetch.call_count, 1)                      # second call came from the cache

    def test_english_never_touches_the_edge_voice(self):
        with self.mock.patch.object(self.engine, "_edge_fetch_wav") as fetch, \
                self.mock.patch.object(self.engine, "load", side_effect=self.mod.TTSUnavailableError("no model")):
            with self.assertRaises(self.mod.TTSUnavailableError):
                self.engine.synthesize("Hello there.", lang="en")
        fetch.assert_not_called()

    def test_a_failing_edge_voice_falls_back_to_kokoro_and_pauses_for_a_minute(self):
        with self.mock.patch.object(self.engine, "_edge_fetch_wav", side_effect=OSError("offline")) as fetch, \
                self.mock.patch.object(self.engine, "load", side_effect=self.mod.TTSUnavailableError("kokoro reached")):
            for _ in range(2):
                with self.assertRaises(self.mod.TTSUnavailableError):      # Kokoro (mocked) was reached both times
                    self.engine.synthesize("दूसरा वाक्य।", lang="hi")
        self.assertEqual(fetch.call_count, 1)                      # ...but Edge was only tried once (paused)

    def test_kokoro_only_setting_skips_edge(self):
        self.engine.hindi_engine = "kokoro"
        with self.mock.patch.object(self.engine, "_edge_fetch_wav") as fetch, \
                self.mock.patch.object(self.engine, "load", side_effect=self.mod.TTSUnavailableError("x")):
            with self.assertRaises(self.mod.TTSUnavailableError):
                self.engine.synthesize("नमस्ते।", lang="hi")
        fetch.assert_not_called()

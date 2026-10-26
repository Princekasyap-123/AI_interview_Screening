# apps/avatar/services/tts_engine.py

"""
Server-side text-to-speech for the AI interviewer, using Kokoro-82M
(Apache-2.0 model weights) through the kokoro-onnx runtime. Runs on the
CPU — no GPU and no paid API.

Why server-side instead of the browser's speechSynthesis:
  - every candidate hears the SAME professional voice (browser voices
    differ per OS/browser and are often robotic);
  - the browser receives real audio, so the avatar's mouth can follow
    the actual loudness of the voice (audio-driven lip-sync).

Model files (download once, see README notes in the delivery message):
    <BASE_DIR>/models/kokoro/kokoro-v1.0.onnx      (~310 MB, full precision)
    <BASE_DIR>/models/kokoro/voices-v1.0.bin       (~27 MB)
The full-precision model is used on purpose: on CPUs without VNNI the
int8 build measured ~3x SLOWER.

Optional settings (all have defaults):
    KOKORO_MODEL_DIR   Path   default BASE_DIR / "models" / "kokoro"
    KOKORO_MODEL_FILE  str    default "kokoro-v1.0.onnx"
    KOKORO_VOICE       str    default "af_heart"
    KOKORO_SPEED       float  default 0.95   (slightly calm interviewer pace)
    KOKORO_THREADS     int    default half the CPU cores (min 1)
"""

import hashlib
import io
import logging
import os
import threading
import time
import wave
from pathlib import Path

import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)

MAX_TEXT_CHARS = 600  # one sentence/chunk per request (the frontend splits long text)


class TTSUnavailableError(Exception):
    """Kokoro or its model files are not available — caller should fall back."""


def _setting(name, default):
    return getattr(settings, name, default)


class KokoroTTSEngine:
    def __init__(self):
        self._kokoro = None
        self._load_lock = threading.Lock()
        # Kokoro's phonemizer (espeak-ng) is not guaranteed thread-safe,
        # so synthesis calls are serialised. One chunk takes well under a
        # second on a normal CPU.
        self._synth_lock = threading.Lock()
        # Live requests (a sentence the candidate is waiting for) go before
        # background pre-synthesis: while any live request is waiting, the
        # prefetch thread pauses between chunks instead of hogging the CPU.
        self._urgent = 0
        self._urgent_lock = threading.Lock()

        self.model_dir = Path(_setting("KOKORO_MODEL_DIR", Path(settings.BASE_DIR) / "models" / "kokoro"))
        self.model_file = _setting("KOKORO_MODEL_FILE", "kokoro-v1.0.onnx")
        self.voice = _setting("KOKORO_VOICE", "af_heart")
        self.speed = float(_setting("KOKORO_SPEED", 0.95))
        self.threads = int(_setting("KOKORO_THREADS", max(1, (os.cpu_count() or 2) // 2)))
        self.cache_dir = Path(settings.MEDIA_ROOT) / "tts_cache"

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._kokoro is not None

    def load(self):
        """Loads the model once (thread-safe). Raises TTSUnavailableError."""
        if self._kokoro is not None:
            return self._kokoro
        with self._load_lock:
            if self._kokoro is not None:
                return self._kokoro
            model_path = self.model_dir / self.model_file
            voices_path = self.model_dir / "voices-v1.0.bin"
            if not model_path.exists() or not voices_path.exists():
                raise TTSUnavailableError(
                    f"Kokoro model files not found in {self.model_dir} "
                    f"(need {self.model_file} and voices-v1.0.bin)"
                )
            try:
                import onnxruntime as ort
                from kokoro_onnx import Kokoro
            except ImportError as exc:
                raise TTSUnavailableError(f"kokoro-onnx is not installed: {exc}") from exc

            so = ort.SessionOptions()
            so.intra_op_num_threads = self.threads
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
            kokoro = Kokoro.from_session(session, str(voices_path))
            # Warm-up so the first real sentence isn't slow.
            kokoro.create("Hello.", voice=self.voice, speed=self.speed, lang=self._lang(self.voice))
            self._kokoro = kokoro
            logger.info(
                "Kokoro TTS loaded (%s, voice=%s, threads=%d)", self.model_file, self.voice, self.threads
            )
        return self._kokoro

    def warm_up_in_background(self):
        """Starts loading the model without blocking the request."""
        if self.is_loaded:
            return
        def _run():
            try:
                self.load()
            except Exception as exc:  # logged, never raised into a request
                logger.warning("Kokoro warm-up failed: %s", exc)
        threading.Thread(target=_run, name="kokoro-warmup", daemon=True).start()

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    @staticmethod
    def _lang(voice: str) -> str:
        return "en-gb" if voice.startswith("b") else "en-us"

    def _cache_path(self, text: str) -> Path:
        key = hashlib.sha256(f"{self.model_file}|{self.voice}|{self.speed}|{text}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.wav"

    def synthesize(self, text: str, background: bool = False) -> bytes:
        """Returns 16-bit mono WAV bytes. Raises TTSUnavailableError / ValueError.
        background=True is for pre-synthesis: it yields to live requests."""
        text = (text or "").strip()
        if not text:
            raise ValueError("empty text")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"text longer than {MAX_TEXT_CHARS} characters")

        cache_file = self._cache_path(text)
        if cache_file.exists():
            return cache_file.read_bytes()

        kokoro = self.load()
        if background:
            while self._urgent:                 # a candidate is waiting for audio: go first
                time.sleep(0.03)
        else:
            with self._urgent_lock:
                self._urgent += 1
        try:
            with self._synth_lock:
                if cache_file.exists():         # another thread made it while we waited
                    return cache_file.read_bytes()
                samples, sample_rate = kokoro.create(
                    text, voice=self.voice, speed=self.speed, lang=self._lang(self.voice)
                )
        finally:
            if not background:
                with self._urgent_lock:
                    self._urgent -= 1

        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm.tobytes())
        data = buf.getvalue()

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(data)
            self._prune_cache()
        except OSError as exc:
            logger.warning("Could not write TTS cache file %s: %s", cache_file, exc)
        return data

    MAX_CACHE_FILES = 1500

    def _prune_cache(self) -> None:
        """Keeps the on-disk cache bounded: removes the oldest files beyond MAX_CACHE_FILES.
        Checked on ~1 in 25 writes so it costs nothing per request."""
        self._writes = getattr(self, "_writes", 0) + 1
        if self._writes % 25:
            return
        files = sorted(self.cache_dir.glob("*.wav"), key=lambda f: f.stat().st_mtime)
        for old in files[: max(0, len(files) - self.MAX_CACHE_FILES)]:
            try:
                old.unlink()
            except OSError:
                pass


tts_engine = KokoroTTSEngine()
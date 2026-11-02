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
    KOKORO_VOICE_HI    str    default "hf_beta"    (Hindi interviews; also hf_alpha, hm_omega, hm_psi)
    KOKORO_SPEED_HI    float  default 1.08   (Hindi pace)
    HINDI_TTS_ENGINE   str    default "edge"   ("edge" = Microsoft neural Indian voice, needs internet;
                                          "kokoro" = local model. Edge falls back to Kokoro on any failure.)
    EDGE_TTS_VOICE_HI  str    default "hi-IN-SwaraNeural"   (also hi-IN-MadhurNeural, male)
    EDGE_TTS_RATE_HI   str    default "+0%"    (e.g. "-5%" slower, "+8%" faster)
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
# phonemizer logs "words count mismatch" on nearly every sentence; the audio is fine, so only real errors are shown.
logging.getLogger("phonemizer").setLevel(logging.ERROR)

MAX_TEXT_CHARS = 600  # one sentence/chunk per request (the frontend splits long text)

_TTS_CHAR_MAP = str.maketrans({
    "‘": "'", "’": "'", "ʼ": "'", "´": "'", "`": "'",
    "“": '"', "”": '"',
    "‐": "-", "‑": "-", "‒": "-",
    "…": "...",
    "।": ".", "॥": ".",       # Devanagari danda: espeak reads a plain full stop as the pause
    " ": " ", " ": " ", "​": "",
})


def normalize_for_tts(text: str) -> str:
    """
    LLM text is full of typographic characters (curly apostrophes, non-breaking
    hyphens, em dashes). espeak/phonemizer chokes on them ("words count
    mismatch" warnings): the word is read wrongly or a pause lands mid-phrase.
    Plain ASCII punctuation reads cleanly. Applied before the cache key too,
    so cached audio is shared between the typographic and plain spelling.
    """
    text = (text or "").translate(_TTS_CHAR_MAP)
    text = text.replace("–", ", ").replace("—", ", ").replace(" - ", ", ")
    text = " ".join(text.split())
    return text.replace(" ,", ",").replace(",,", ",").strip()


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
        self.voice_hi = _setting("KOKORO_VOICE_HI", "hf_beta")
        self.speed = float(_setting("KOKORO_SPEED", 0.95))
        # Hindi at 0.95 sounds drawn-out; a natural conversational pace is a bit faster.
        self.speed_hi = float(_setting("KOKORO_SPEED_HI", 1.08))
        self.hindi_engine = str(_setting("HINDI_TTS_ENGINE", "edge")).lower()
        self.edge_voice_hi = _setting("EDGE_TTS_VOICE_HI", "hi-IN-SwaraNeural")
        self.edge_rate_hi = _setting("EDGE_TTS_RATE_HI", "+0%")
        self._edge_down_until = 0.0          # after a failure Hindi uses Kokoro for a minute instead of retrying every sentence
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
        if voice.startswith(("hf_", "hm_")):
            return "hi"
        return "en-gb" if voice.startswith("b") else "en-us"

    def _voice_for(self, lang: str) -> str:
        return self.voice_hi if lang == "hi" else self.voice

    def _speed_for(self, voice: str) -> float:
        return self.speed_hi if self._lang(voice) == "hi" else self.speed

    def _cache_path(self, text: str, voice: str) -> Path:
        key = hashlib.sha256(
            f"{self.model_file}|{voice}|{self._speed_for(voice)}|{text}".encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{key}.wav"

    def synthesize(self, text: str, background: bool = False, lang: str = "en") -> bytes:
        """Returns 16-bit mono WAV bytes. Raises TTSUnavailableError / ValueError.
        background=True is for pre-synthesis: it yields to live requests.
        lang="hi" speaks with the Hindi voice (the text must be Devanagari)."""
        voice = self._voice_for(lang)
        text = normalize_for_tts(text)
        if not text:
            raise ValueError("empty text")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"text longer than {MAX_TEXT_CHARS} characters")

        if lang == "hi" and self._edge_wanted():
            data = self._synthesize_edge(text)
            if data is not None:
                return data

        cache_file = self._cache_path(text, voice)
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
                    text, voice=voice, speed=self._speed_for(voice), lang=self._lang(voice)
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

    # ------------------------------------------------------------------
    # Hindi: Microsoft neural voice (natural Indian accent), Kokoro as the fallback
    # ------------------------------------------------------------------

    def _edge_wanted(self) -> bool:
        return self.hindi_engine == "edge" and time.time() >= self._edge_down_until

    def _synthesize_edge(self, text: str) -> "bytes | None":
        """WAV bytes from the Edge neural voice, or None (-> the caller uses Kokoro)."""
        cache_file = self._cache_path(text, f"edge:{self.edge_voice_hi}:{self.edge_rate_hi}")
        if cache_file.exists():
            return cache_file.read_bytes()
        try:
            data = self._edge_fetch_wav(text)
        except Exception as exc:
            self._edge_down_until = time.time() + 60
            logger.warning("Edge Hindi voice failed (%s) - using Kokoro for the next minute.", str(exc)[:160])
            return None
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(data)
            self._prune_cache()
        except OSError as exc:
            logger.warning("Could not write TTS cache file %s: %s", cache_file, exc)
        return data

    def _edge_fetch_wav(self, text: str) -> bytes:
        import asyncio

        import edge_tts
        import miniaudio

        async def fetch() -> bytes:
            mp3 = bytearray()
            comm = edge_tts.Communicate(text, self.edge_voice_hi, rate=self.edge_rate_hi)
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    mp3.extend(chunk["data"])
            return bytes(mp3)

        try:
            mp3 = asyncio.run(asyncio.wait_for(fetch(), timeout=20))
        except RuntimeError:                      # this thread already has a running loop: use a fresh one elsewhere
            box: dict = {}
            t = threading.Thread(target=lambda: box.update(v=asyncio.run(asyncio.wait_for(fetch(), timeout=20))))
            t.start(); t.join()
            mp3 = box["v"]
        if not mp3:
            raise RuntimeError("empty audio from the Edge voice")
        decoded = miniaudio.decode(mp3, output_format=miniaudio.SampleFormat.SIGNED16, nchannels=1, sample_rate=24000)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(decoded.samples.tobytes())
        return buf.getvalue()

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
# apps/avatar/services/lipsync_engine.py

"""
Real lip-sync for the AI interviewer via a MuseTalk GPU server.

Current setup (DEMO): the GPU server is the Kaggle notebook
`musetalk_kaggle_demo.ipynb`, exposed through a Cloudflare quick tunnel.
Later the same HTTP contract can be served by a company GPU machine —
only MUSETALK_URL changes.

Flow per sentence:
  text -> Kokoro WAV (tts_engine) -> POST {MUSETALK_URL}/lipsync -> MP4
The MP4 (lip-synced video WITH the voice) is cached on disk by
(voice settings + text), so repeated lines are instant.

Settings (.env -> settings.py):
  MUSETALK_URL       e.g. https://xxxx.trycloudflare.com   (empty = disabled)
  MUSETALK_TOKEN     shared secret, same as the Kaggle secret
  MUSETALK_TIMEOUT   seconds, default 60
"""

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from django.conf import settings

from apps.avatar.services.tts_engine import tts_engine

logger = logging.getLogger(__name__)


class LipsyncNotAvailableError(Exception):
    """The lip-sync server is not configured / not reachable / failed."""


@dataclass
class LipsyncClipResult:
    clip_id: str          # file name stem in the cache dir
    duration_ms: int
    infer_ms: int         # GPU render time reported by the server (0 if cached)
    source_engine: str = "musetalk"


def _setting(name, default=""):
    return getattr(settings, name, None) or os.environ.get(name, default)


class LipsyncEngine:
    def __init__(self):
        self.cache_dir = Path(settings.MEDIA_ROOT) / "lipsync_cache"
        self._health = {"ok": False, "checked": 0.0}

    @property
    def url(self) -> str:
        return (_setting("MUSETALK_URL") or "").rstrip("/")

    @property
    def headers(self) -> dict:
        return {"X-Token": _setting("MUSETALK_TOKEN")}

    @property
    def timeout(self) -> float:
        return float(_setting("MUSETALK_TIMEOUT", "60"))

    def is_available(self) -> bool:
        """Cheap check (cached 30 s) used by the page to decide on lip-sync mode."""
        if not self.url:
            return False
        now = time.time()
        if now - self._health["checked"] < 30:
            return self._health["ok"]
        ok = False
        try:
            r = requests.get(f"{self.url}/health", headers=self.headers, timeout=5)
            ok = r.ok and bool(r.json().get("ready"))
        except Exception as exc:
            logger.info("MuseTalk server not reachable: %s", exc)
        self._health = {"ok": ok, "checked": now}
        return ok

    def clip_path(self, clip_id: str) -> Path:
        return self.cache_dir / f"{clip_id}.mp4"

    def _clip_id(self, text: str) -> str:
        key = f"musetalk-v15|{tts_engine.model_file}|{tts_engine.voice}|{tts_engine.speed}|{text}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:40]

    def render_text(self, text: str) -> LipsyncClipResult:
        if not self.url:
            raise LipsyncNotAvailableError("MUSETALK_URL is not set")

        clip_id = self._clip_id(text)
        path = self.clip_path(clip_id)
        if path.exists():
            return LipsyncClipResult(clip_id=clip_id, duration_ms=0, infer_ms=0)

        wav = tts_engine.synthesize(text)
        t0 = time.time()
        try:
            r = requests.post(
                f"{self.url}/lipsync",
                headers=self.headers,
                files={"audio": ("speech.wav", wav, "audio/wav")},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            self._health = {"ok": False, "checked": time.time()}
            raise LipsyncNotAvailableError(f"MuseTalk request failed: {exc}") from exc
        if r.status_code != 200 or r.headers.get("Content-Type", "").split(";")[0] != "video/mp4":
            raise LipsyncNotAvailableError(f"MuseTalk returned HTTP {r.status_code}: {r.text[:200]}")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(r.content)
        tmp.replace(path)

        infer_ms = int(r.headers.get("X-Infer-Ms", 0))
        audio_ms = int(r.headers.get("X-Audio-Ms", 0))
        round_trip = int((time.time() - t0) * 1000)
        logger.info(
            "MuseTalk clip: audio %d ms, GPU render %d ms, round trip %d ms (RTF %.2f)",
            audio_ms, infer_ms, round_trip, round_trip / max(audio_ms, 1),
        )
        return LipsyncClipResult(clip_id=clip_id, duration_ms=audio_ms, infer_ms=infer_ms)


lipsync_engine = LipsyncEngine()
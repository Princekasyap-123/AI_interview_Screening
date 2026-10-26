# apps/avatar/views.py

"""
HTTP endpoints for the AI interviewer's voice.

  POST /api/avatar/tts/           {"session_id": "...", "text": "..."}  -> audio/wav
  GET  /api/avatar/tts/warmup/                                           -> 202 / 200

Candidates are not logged-in Django users, so access is tied to a valid
interview session id (a random UUID) instead of authentication. Text is
capped per request (the frontend sends one sentence at a time).
"""

import json
import logging

from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from apps.avatar.services.tts_engine import MAX_TEXT_CHARS, TTSUnavailableError, tts_engine
from apps.interviews.models import InterviewSession

logger = logging.getLogger(__name__)


TTS_MAX_REQUESTS_PER_MINUTE = 240      # one interview needs a few dozen; this only stops abuse


def _rate_limited(session_id) -> bool:
    """True when this session has used up its TTS requests for the current minute."""
    key = f"tts-rate:{session_id}"
    cache.add(key, 0, timeout=60)
    try:
        return cache.incr(key) > TTS_MAX_REQUESTS_PER_MINUTE
    except ValueError:                  # key expired between add() and incr()
        cache.set(key, 1, timeout=60)
        return False


def _session_exists(session_id) -> bool:
    if not session_id:
        return False
    try:
        return InterviewSession.objects.filter(pk=session_id).exists()
    except (ValueError, ValidationError):
        return False


@csrf_exempt
@require_POST
def tts_view(request):
    try:
        payload = json.loads(request.body or b"{}")
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"error": "invalid JSON"}, status=400)

    text = (payload.get("text") or "").strip()
    if not text:
        return JsonResponse({"error": "text is required"}, status=400)
    if len(text) > MAX_TEXT_CHARS:
        return JsonResponse({"error": f"text longer than {MAX_TEXT_CHARS} characters"}, status=400)
    if not _session_exists(payload.get("session_id")):
        return JsonResponse({"error": "unknown interview session"}, status=403)
    if _rate_limited(payload.get("session_id")):
        return JsonResponse({"error": "too many requests"}, status=429)

    try:
        audio = tts_engine.synthesize(text)
    except TTSUnavailableError as exc:
        logger.warning("TTS unavailable, browser voice will be used: %s", exc)
        return JsonResponse({"error": "tts unavailable"}, status=503)
    except Exception:
        logger.exception("TTS synthesis failed")
        return JsonResponse({"error": "tts failed"}, status=500)

    response = HttpResponse(audio, content_type="audio/wav")
    response["Cache-Control"] = "private, max-age=3600"
    return response


@require_GET
def tts_warmup_view(request):
    """Called when the interview page opens so the model is loaded before the first question."""
    if tts_engine.is_loaded:
        return JsonResponse({"status": "ready"})
    tts_engine.warm_up_in_background()
    return JsonResponse({"status": "loading"}, status=202)


# ----------------------------------------------------------------------
# Real lip-sync (MuseTalk GPU server) — see services/lipsync_engine.py
# ----------------------------------------------------------------------

import re

from django.http import FileResponse, Http404

from apps.avatar.services.lipsync_engine import LipsyncNotAvailableError, lipsync_engine

_CLIP_ID_RE = re.compile(r"^[0-9a-f]{40}$")


@require_GET
def lipsync_status_view(request):
    """The page asks this once at start-up to choose lip-sync or audio-only mode."""
    return JsonResponse({"available": lipsync_engine.is_available()})


@csrf_exempt
@require_POST
def lipsync_view(request):
    """POST {"session_id", "text"} -> {"clip_url", "infer_ms"}; 503 => use audio-only voice."""
    try:
        payload = json.loads(request.body or b"{}")
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"error": "invalid JSON"}, status=400)

    text = (payload.get("text") or "").strip()
    if not text:
        return JsonResponse({"error": "text is required"}, status=400)
    if len(text) > MAX_TEXT_CHARS:
        return JsonResponse({"error": f"text longer than {MAX_TEXT_CHARS} characters"}, status=400)
    if not _session_exists(payload.get("session_id")):
        return JsonResponse({"error": "unknown interview session"}, status=403)

    try:
        clip = lipsync_engine.render_text(text)
    except (LipsyncNotAvailableError, TTSUnavailableError) as exc:
        logger.warning("Lip-sync unavailable, page will use audio-only voice: %s", exc)
        return JsonResponse({"error": "lipsync unavailable"}, status=503)
    except Exception:
        logger.exception("Lip-sync failed")
        return JsonResponse({"error": "lipsync failed"}, status=500)

    return JsonResponse({
        "clip_url": f"/api/avatar/lipsync/clip/{clip.clip_id}.mp4",
        "infer_ms": clip.infer_ms,
    })


@require_GET
def lipsync_clip_view(request, clip_id):
    if not _CLIP_ID_RE.match(clip_id):
        raise Http404
    path = lipsync_engine.clip_path(clip_id)
    if not path.exists():
        raise Http404
    response = FileResponse(open(path, "rb"), content_type="video/mp4")
    response["Cache-Control"] = "private, max-age=86400"
    return response
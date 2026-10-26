# apps/ai_engine/llm_client.py

import json
import logging
import re
from typing import Any

from django.conf import settings
from groq import Groq, BadRequestError

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Thin wrapper around the Groq API. Models are exposed on purpose:
    - REASONING_MODEL: question generation + answer analysis (needs depth)
    - FAST_MODEL: quick decisions, e.g. "is this answer complete?" (needs speed)
    - VISION_MODEL: the attire-check agent (white-collar interviews only)

    NOTE: llama-3.3-70b-versatile and llama-3.1-8b-instant were
    deprecated by Groq on 2026-08-16. Migrated to Groq's recommended
    1:1 replacements per their deprecations page
    (console.groq.com/docs/deprecations). If these ever 404 again,
    check that page for the current recommended models before
    re-guessing — Groq's lineup changes over time. Same goes for
    VISION_MODEL — check console.groq.com/docs/vision if it 404s.

    gpt-oss models are REASONING models: they think before answering,
    and that thinking uses the same token budget as the answer. With a
    small max_tokens the budget can run out before any answer is written,
    which shows up as an empty response / "json_validate_failed" with
    failed_generation ''. So:
      - token limits below are sized with headroom for reasoning
      - reasoning_effort is set per call ("low" for structured/quick tasks)
      - JSON calls retry once without strict JSON mode and parse the text
    """

    REASONING_MODEL = "openai/gpt-oss-120b"
    FAST_MODEL = "openai/gpt-oss-20b"
    # Vision model for the attire-check agent. Groq's vision lineup changes
    # over time — if this 404s, check console.groq.com/docs/vision for the
    # current recommended model before re-guessing.
    VISION_MODEL = "qwen/qwen3.8-27b"

    # Models that accept Groq's reasoning_effort parameter.
    _REASONING_EFFORT_MODELS = ("openai/gpt-oss-120b", "openai/gpt-oss-20b")

    def __init__(self):
        # Created on first use, so importing this module (manage.py check,
        # migrations, tests, DEBUG runs without a key) never fails.
        self._client = None

    @property
    def client(self) -> Groq:
        if self._client is None:
            api_key = getattr(settings, "GROQ_API_KEY", None)
            if not api_key:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Add it to your .env and settings.py."
                )
            # timeout/retries: a stuck call must not freeze a live interview turn.
            self._client = Groq(api_key=api_key, timeout=25.0, max_retries=1)
        return self._client

    def _chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        json_mode: bool = False,
        temperature: float = 0.4,
        max_tokens: int = 2048,
        reasoning_effort: str | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if reasoning_effort and model in self._REASONING_EFFORT_MODELS:
            # Sent via extra_body so it works regardless of groq SDK version.
            kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}

        try:
            response = self.client.chat.completions.create(**kwargs)
        except BadRequestError:
            # Let callers (generate_json) inspect and retry this one.
            raise
        except Exception as exc:
            logger.error("Groq API call failed: %s", exc, exc_info=True)
            raise RuntimeError(f"LLM call failed: {exc}") from exc

        choice = response.choices[0]
        content = choice.message.content or ""
        if not content.strip():
            usage = getattr(response, "usage", None)
            logger.warning(
                "Groq returned empty content (model=%s, finish_reason=%s, "
                "max_tokens=%s, usage=%s) — the reasoning budget was most "
                "likely used up before an answer was written.",
                model, choice.finish_reason, max_tokens, usage,
            )
        return content

    def generate_reasoning(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        temperature: float = 0.4,
        max_tokens: int = 4096,
        reasoning_effort: str = "medium",
    ) -> str:
        return self._chat_or_raise(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.REASONING_MODEL,
            json_mode=json_mode,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

    def generate_fast(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        reasoning_effort: str = "low",
    ) -> str:
        return self._chat_or_raise(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.FAST_MODEL,
            json_mode=json_mode,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        reasoning_effort: str = "low",
    ) -> dict:
        """
        Returns parsed JSON (dict). Attempt 1 uses Groq's strict JSON mode.
        If Groq rejects the output (json_validate_failed) or the content
        is empty/unparseable, attempt 2 runs WITHOUT strict JSON mode and
        with a larger budget, then extracts the JSON object from the text.
        """
        model = model or self.REASONING_MODEL
        common = dict(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )

        # ---- attempt 1: strict JSON mode ----
        raw = ""
        try:
            raw = self._chat(json_mode=True, max_tokens=max_tokens, **common)
            parsed = self._extract_json(raw)
            if parsed is not None:
                return parsed
            logger.warning("JSON mode returned unparseable content, retrying. Raw: %.300s", raw)
        except BadRequestError as exc:
            logger.warning("Groq rejected JSON-mode output (%s) — retrying without strict JSON mode.", exc)

        # ---- attempt 2: plain mode, bigger budget, parse from text ----
        try:
            raw = self._chat(json_mode=False, max_tokens=max_tokens * 2, **common)
        except BadRequestError as exc:
            logger.error("Groq API call failed on JSON retry: %s", exc, exc_info=True)
            raise RuntimeError(f"LLM call failed: {exc}") from exc

        parsed = self._extract_json(raw)
        if parsed is None:
            logger.error("LLM returned invalid JSON after retry: %.1000s", raw)
            raise ValueError("LLM did not return valid JSON after retry.")
        return parsed

    def generate_text_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 400,
        reasoning_effort: str = "low",
    ):
        """
        Yields the reply text incrementally (Groq SSE streaming) instead of
        waiting for the full response. Plain text only — no JSON mode (the
        two don't mix reliably on Groq, and we don't need JSON here).

        Lets a caller act on each finished sentence (e.g. start Kokoro
        synthesis for it) WHILE the model is still writing the next one,
        instead of waiting for the whole reply before doing anything.

        Never raises mid-stream: on failure it logs and simply stops
        yielding, so the caller's accumulated-so-far text (or its own
        fallback) is what gets used.
        """
        model = model or self.FAST_MODEL
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if reasoning_effort and model in self._REASONING_EFFORT_MODELS:
            kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}

        try:
            stream = self.client.chat.completions.create(**kwargs)
            for event in stream:
                if not event.choices:
                    continue
                delta = event.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            logger.error("Groq streaming call failed: %s", exc, exc_info=True)
            return

    def generate_vision_json(
        self,
        system_prompt: str,
        user_text: str,
        image_base64: str,
        image_mime: str = "image/jpeg",
        temperature: float = 0.2,
        max_tokens: int = 300,
    ) -> dict:
        """
        One still frame + a text instruction -> parsed JSON. Used by the
        attire-check agent (white-collar interviews only). No reasoning_effort
        — the vision model isn't in _REASONING_EFFORT_MODELS.
        """
        kwargs: dict[str, Any] = {
            "model": self.VISION_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{image_mime};base64,{image_base64}"},
                        },
                    ],
                },
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            logger.error("Groq vision call failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Vision LLM call failed: {exc}") from exc

        content = response.choices[0].message.content or ""
        parsed = self._extract_json(content)
        if parsed is None:
            raise ValueError(f"Vision LLM did not return valid JSON: {content[:300]!r}")
        return parsed

    # ---------------- helpers ----------------

    def _chat_or_raise(self, **kwargs) -> str:
        """_chat() for the plain-text helpers: converts BadRequestError
        into the RuntimeError callers already expect."""
        try:
            return self._chat(**kwargs)
        except BadRequestError as exc:
            logger.error("Groq API call failed: %s", exc, exc_info=True)
            raise RuntimeError(f"LLM call failed: {exc}") from exc

    @staticmethod
    def _extract_json(raw: str) -> dict | None:
        """Parses a JSON object from model output, tolerating code fences
        or stray text around it. Returns None if nothing usable is found."""
        if not raw or not raw.strip():
            return None
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end <= start:
                return None
            try:
                data = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return None
        return data if isinstance(data, dict) else None


llm_client = LLMClient()
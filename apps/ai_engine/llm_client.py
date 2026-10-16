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
    Thin wrapper around the Groq API. Two models are exposed on purpose:
    - REASONING_MODEL: question generation + answer analysis (needs depth)
    - FAST_MODEL: quick decisions, e.g. "is this answer complete?" (needs speed)

    NOTE: llama-3.3-70b-versatile and llama-3.1-8b-instant were
    deprecated by Groq on 2026-08-16. Migrated to Groq's recommended
    1:1 replacements per their deprecations page
    (console.groq.com/docs/deprecations). If these ever 404 again,
    check that page for the current recommended models before
    re-guessing — Groq's lineup changes over time.

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

    # Models that accept Groq's reasoning_effort parameter.
    _REASONING_EFFORT_MODELS = ("openai/gpt-oss-120b", "openai/gpt-oss-20b")

    def __init__(self):
        api_key = getattr(settings, "GROQ_API_KEY", None)
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env and settings.py."
            )
        self.client = Groq(api_key=api_key)

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
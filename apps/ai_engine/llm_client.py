# apps/ai_engine/llm_client.py

import json
import logging
from typing import Any

from django.conf import settings
from groq import Groq

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
    """

    REASONING_MODEL = "openai/gpt-oss-120b"
    FAST_MODEL = "openai/gpt-oss-20b"
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
        max_tokens: int = 1024,
    ) -> str:
        try:
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

            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content

        except Exception as exc:
            logger.error("Groq API call failed: %s", exc, exc_info=True)
            raise RuntimeError(f"LLM call failed: {exc}") from exc

    def generate_reasoning(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        temperature: float = 0.4,
        max_tokens: int = 1024,
    ) -> str:
        return self._chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.REASONING_MODEL,
            json_mode=json_mode,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate_fast(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 256,
    ) -> str:
        return self._chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.FAST_MODEL,
            json_mode=json_mode,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> dict:
        model = model or self.REASONING_MODEL
        raw = self._chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            json_mode=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("LLM returned invalid JSON: %s", raw)
            raise ValueError(f"LLM did not return valid JSON: {exc}") from exc


llm_client = LLMClient()
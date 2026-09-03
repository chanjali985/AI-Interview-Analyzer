"""Pluggable LLM layer.

Providers all expose the same tiny interface so the rest of the app never cares
whether the model runs locally in Ollama or behind a hosted API.

    ollama  -> free, local, default
    openai  -> api.openai.com (or any OpenAI-compatible base URL)
    groq    -> Groq's OpenAI-compatible endpoint
    gemini  -> Google's OpenAI-compatible endpoint
"""
from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

import httpx

from ..config import settings

logger = logging.getLogger(__name__)

Message = dict[str, str]


class LLMError(RuntimeError):
    """Raised when the language model cannot be reached or returns nothing."""


# --------------------------------------------------------------------------- base
class LLMProvider(ABC):
    name: str = "base"

    def __init__(self, model: str, timeout: int) -> None:
        self.model = model
        self.timeout = timeout

    @abstractmethod
    def _complete(self, messages: list[Message], temperature: float, json_mode: bool) -> str:
        ...

    @abstractmethod
    def health(self) -> bool:
        ...

    def chat(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        messages: list[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_error: Exception | None = None
        for attempt in range(1, settings.LLM_MAX_RETRIES + 2):
            try:
                content = self._complete(messages, temperature, json_mode)
                if content and content.strip():
                    return content.strip()
                last_error = LLMError("Model returned an empty response")
            except Exception as exc:  # noqa: BLE001 - surfaced as LLMError below
                last_error = exc
                logger.warning("LLM call failed (attempt %s/%s): %s", attempt, settings.LLM_MAX_RETRIES + 1, exc)
            if attempt <= settings.LLM_MAX_RETRIES:
                time.sleep(min(2 ** attempt, 8))

        raise LLMError(f"{self.name} call failed: {last_error}")


# ------------------------------------------------------------------------- ollama
class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, model: str, timeout: int, base_url: str) -> None:
        super().__init__(model, timeout)
        self.base_url = base_url.rstrip("/")

    def _complete(self, messages: list[Message], temperature: float, json_mode: bool) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        return data.get("message", {}).get("content", "")

    def health(self) -> bool:
        try:
            with httpx.Client(timeout=5) as client:
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                models = [m.get("name", "") for m in response.json().get("models", [])]
            if not models:
                return False
            base = self.model.split(":")[0]
            return any(m == self.model or m.split(":")[0] == base for m in models)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Ollama health check failed: %s", exc)
            return False


# ------------------------------------------------------------- openai compatible
class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, model: str, timeout: int, base_url: str, api_key: str, name: str) -> None:
        super().__init__(model, timeout)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.name = name

    def _complete(self, messages: list[Message], temperature: float, json_mode: bool) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            if response.status_code >= 400:
                raise LLMError(f"HTTP {response.status_code}: {response.text[:300]}")
            data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "")

    def health(self) -> bool:
        return bool(self.api_key)


# --------------------------------------------------------------------------- none
class NullProvider(LLMProvider):
    """Used when LLM_PROVIDER=none — analysis falls back to heuristics."""

    name = "none"

    def __init__(self) -> None:
        super().__init__(model="none", timeout=1)

    def _complete(self, messages: list[Message], temperature: float, json_mode: bool) -> str:
        raise LLMError("No LLM provider is configured (LLM_PROVIDER=none)")

    def health(self) -> bool:
        return False


# ------------------------------------------------------------------------ factory
def build_provider() -> LLMProvider:
    provider = settings.LLM_PROVIDER
    model = settings.LLM_MODEL
    timeout = settings.LLM_TIMEOUT_SECONDS

    if provider == "ollama":
        return OllamaProvider(model, timeout, settings.OLLAMA_BASE_URL)
    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise LLMError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        return OpenAICompatibleProvider(model, timeout, settings.OPENAI_BASE_URL, settings.OPENAI_API_KEY, "openai")
    if provider == "groq":
        if not settings.GROQ_API_KEY:
            raise LLMError("LLM_PROVIDER=groq but GROQ_API_KEY is not set")
        return OpenAICompatibleProvider(model, timeout, settings.GROQ_BASE_URL, settings.GROQ_API_KEY, "groq")
    if provider == "gemini":
        if not settings.GEMINI_API_KEY:
            raise LLMError("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set")
        return OpenAICompatibleProvider(model, timeout, settings.GEMINI_BASE_URL, settings.GEMINI_API_KEY, "gemini")
    return NullProvider()


@lru_cache
def get_llm() -> LLMProvider:
    try:
        return build_provider()
    except LLMError as exc:
        logger.error("LLM provider unavailable: %s", exc)
        return NullProvider()


def reset_llm_cache() -> None:
    get_llm.cache_clear()


# -------------------------------------------------------------- JSON extraction
def extract_json(text: str) -> Any:
    """Pull the first valid JSON object/array out of a model response."""
    if not text:
        return None

    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        while start != -1:
            depth = 0
            in_string = False
            escape = False
            for index in range(start, len(cleaned)):
                char = cleaned[index]
                if in_string:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == opener:
                    depth += 1
                elif char == closer:
                    depth -= 1
                    if depth == 0:
                        candidate = cleaned[start : index + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break
            start = cleaned.find(opener, start + 1)
    return None


def chat_json(prompt: str, system: str | None = None, temperature: float = 0.1) -> Any:
    """Ask the model for JSON and parse it. Returns None when nothing usable came back."""
    llm = get_llm()
    raw = llm.chat(prompt, system=system, temperature=temperature, json_mode=True)
    parsed = extract_json(raw)
    if parsed is None:
        logger.warning("Could not parse JSON from model output: %.200s", raw)
    return parsed

"""The ONE function every module uses to talk to a language model.

    from app.llm import ask_llm, ask_llm_json

Provider is chosen in .env:
    LLM_PROVIDER=ollama   -> local model through Ollama (default, e.g. qwen2.5:3b)
    LLM_PROVIDER=mock     -> offline fake model for development and tests

Ollama's native /api/chat endpoint is used (not the OpenAI-compatible one)
because it lets us set num_ctx. Ollama's default context window is small and
silently cuts off long prompts such as topic labeling and quiz generation.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

import httpx

from .config import settings

_model_override: str | None = None
_no_think_models: set[str] = set()
_no_format_models: set[str] = set()

# Counters read by the evaluation script: how often the model needed a retry for valid JSON.
json_stats = {"calls": 0, "first_try": 0, "retries": 0, "failures": 0}


def use_model(name: str | None) -> None:
    """Switch model at runtime (used by the evaluation script to compare models)."""
    global _model_override
    _model_override = name


def current_model() -> str:
    return _model_override or settings.llm_model


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks that reasoning models such as Qwen3 add."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def _ollama_chat(messages: list[dict], json_mode: bool, temperature: float) -> str:
    payload: dict[str, Any] = {
        "model": current_model(),
        "messages": messages,
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": temperature, "num_ctx": settings.llm_num_ctx},
    }
    model = current_model()
    if json_mode and model not in _no_format_models:
        payload["format"] = "json"
    if settings.llm_think in ("on", "off") and model not in _no_think_models:
        # Thinking models (qwen3, qwen3.5, ...) think before answering. "off" skips that: much faster.
        payload["think"] = settings.llm_think == "on"
    try:
        resp = httpx.post(f"{settings.ollama_host}/api/chat", json=payload, timeout=settings.llm_timeout)
        if resp.status_code == 400 and "think" in resp.text.lower() and "think" in payload:
            # This model has no thinking switch (for example qwen2.5). Ask again without it.
            _no_think_models.add(model)
            payload.pop("think")
            resp = httpx.post(f"{settings.ollama_host}/api/chat", json=payload, timeout=settings.llm_timeout)
        if resp.status_code in (400, 501) and "format" in payload:
            # Some newer models (for example qwen3.5) do not support Ollama's strict JSON mode yet.
            # Ask again without it; the prompt still asks for JSON and extract_json() parses the reply.
            _no_format_models.add(model)
            payload.pop("format")
            resp = httpx.post(f"{settings.ollama_host}/api/chat", json=payload, timeout=settings.llm_timeout)
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {settings.ollama_host}. Start it with 'ollama serve' "
            "or set LLM_PROVIDER=mock in backend/.env."
        ) from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"{model} took longer than {int(settings.llm_timeout)}s to reply.") from exc
    if resp.status_code == 404:
        raise RuntimeError(f"Ollama has no model '{current_model()}'. Run: ollama pull {current_model()}")
    if resp.status_code >= 400:
        raise RuntimeError(f"Ollama error {resp.status_code} for {model}: {resp.text[:200]}")
    return resp.json()["message"]["content"]


def ask_llm(prompt: str, system: str | None = None, json_mode: bool = False, temperature: float = 0.2) -> str:
    """Send one prompt, get the reply text back."""
    if settings.llm_provider == "mock":
        from .llm_mock import mock_reply

        return mock_reply(prompt, system=system, json_mode=json_mode)
    if settings.llm_provider != "ollama":
        raise RuntimeError(f"Unknown LLM_PROVIDER: {settings.llm_provider} (use ollama or mock)")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return strip_thinking(_ollama_chat(messages, json_mode, temperature))


def extract_json(text: str) -> Any:
    """Parse JSON even when a small model wraps it in ``` fences or adds chatter."""
    text = strip_thinking(text)
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("Reply is not valid JSON")


def ask_llm_json(
    prompt: str,
    system: str | None = None,
    validate: Callable[[Any], Any] | None = None,
    retries: int = 3,
) -> Any:
    """Ask for JSON, validate it, and retry with the error message if it is broken.

    `validate` receives the parsed JSON and should return the cleaned value
    or raise ValueError explaining what is wrong.
    """
    last_error = ""
    current_prompt = prompt
    json_stats["calls"] += 1
    for attempt in range(retries):
        if attempt:
            json_stats["retries"] += 1
        reply = ask_llm(current_prompt, system=system, json_mode=True, temperature=0.2 + 0.2 * attempt)
        try:
            data = extract_json(reply)
            result = validate(data) if validate else data
            if attempt == 0:
                json_stats["first_try"] += 1
            return result
        except (ValueError, KeyError, TypeError) as exc:
            last_error = str(exc)
            current_prompt = (
                f"{prompt}\n\nYour previous reply was rejected: {last_error}\n"
                "Reply again with ONLY valid JSON in the exact format requested."
            )
            time.sleep(0.2 * attempt)
    json_stats["failures"] += 1
    raise RuntimeError(f"LLM did not return valid JSON after {retries} tries: {last_error}")

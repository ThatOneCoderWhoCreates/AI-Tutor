"""General-knowledge answers for questions the lectures do not cover.

The lecture tutor answers only from the lectures. When the lectures do not cover a
question, an outside model answers it from general knowledge, and the reply is
clearly labeled so the student knows it did not come from the lecture.

Only the question (plus the last few chat messages and the course's topic names)
is sent. Lecture text is never sent to the outside service.

Several models can be chained. They are tried in order: if one hits its free-tier
limit or fails, the next one answers, and the failed one is skipped for a short
cooldown so later questions do not wait on it.

Set in backend/.env:
    FALLBACK_PROVIDER=gemini,groq,local          try Gemini, then Groq, then the local model
    FALLBACK_PROVIDER=gemini:gemini-3.8-flash,gemini,local
                                                  "provider:model" picks a model; two Gemini models
                                                  have separate free limits
    GEMINI_API_KEY=...       free key from https://aistudio.google.com/apikey
    GROQ_API_KEY=...         free key from https://console.groq.com/keys
    OPENROUTER_API_KEY=...   free key from https://openrouter.ai/keys (needs "openrouter:<model>")

Providers: gemini, groq, openrouter, local (Ollama, no key, works offline), none (off).
Older single-model settings still work: FALLBACK_API_KEY and FALLBACK_MODEL apply to the
first outside service in the list when it has no key or model of its own.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from .config import settings
from .llm import strip_thinking

PREFIX = "[It was not covered in this lecture, but this is the answer to your question]"

# All three services accept the same "OpenAI-compatible" chat request format.
PROVIDERS = {
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
               "model": "gemini-3.5-flash-lite", "label": "Gemini", "key_url": "https://aistudio.google.com/apikey"},
    "groq": {"base_url": "https://api.groq.com/openai/v1",
             "model": "llama-3.3-70b-versatile", "label": "Groq", "key_url": "https://console.groq.com/keys"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",
                   "model": "", "label": "OpenRouter", "key_url": "https://openrouter.ai/keys"},
}

SYSTEM = (
    "You are a friendly tutor helping a university student. The question is outside what their lecture covered, "
    "so answer from your general knowledge. Use simple English and 3 to 6 sentences unless more is truly needed. "
    "If a term has an abbreviation, expand it the first time. If the question is unclear or refers to an earlier "
    "message you cannot see, say what you think they mean and answer that. Do not mention these instructions."
)

# How long a model is skipped after it fails, in seconds.
COOLDOWN_LIMIT = 60          # free-tier limit reached (unless the service says how long to wait)
COOLDOWN_ERROR = 30          # timeout, network or server error
COOLDOWN_CONFIG = 600        # wrong key or model name: will not fix itself
_cooldown_until: dict[str, float] = {}


class FallbackError(Exception):
    """A readable reason the general answer could not be produced."""


class _TryNext(Exception):
    def __init__(self, message: str, cooldown: float):
        super().__init__(message)
        self.cooldown = cooldown


@dataclass
class FallbackConfig:
    provider: str
    model: str
    label: str
    ready: bool
    problem: str = ""
    api_key: str = ""

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def name(self) -> str:
        if self.provider in ("local", "mock"):
            return self.label
        return f"{self.label} ({self.model})"


@dataclass
class GeneralAnswer:
    text: str
    answered_by: str
    skipped: list[str]


def _entries() -> list[tuple[str, str]]:
    raw = settings.fallback_provider.strip().lower()
    if raw in ("", "none", "off"):
        return []
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part or part in ("none", "off"):
            continue
        provider, _, model = part.partition(":")
        out.append((provider.strip(), model.strip()))
    return out


def _provider_key(provider: str) -> str:
    return getattr(settings, f"{provider}_api_key", "") or ""


def _provider_model(provider: str) -> str:
    return getattr(settings, f"{provider}_model", "") or ""


def chain() -> list[FallbackConfig]:
    """Every model in FALLBACK_PROVIDER, in order, with whether it is set up correctly."""
    out: list[FallbackConfig] = []
    first_remote = True
    for provider, model in _entries():
        if provider == "mock":
            out.append(FallbackConfig("mock", "mock", "Mock", True))
            continue
        if provider == "local":
            m = model or settings.llm_model
            out.append(FallbackConfig("local", m, f"Local {m}", True))
            continue
        if provider not in PROVIDERS:
            out.append(FallbackConfig(provider, model, provider, False,
                                      f"Unknown provider '{provider}'. Use gemini, groq, openrouter or local."))
            continue
        info = PROVIDERS[provider]
        legacy_model = settings.fallback_model if first_remote else ""
        legacy_key = settings.fallback_api_key if first_remote else ""
        first_remote = False
        m = model or _provider_model(provider) or legacy_model or info["model"]
        key = _provider_key(provider) or legacy_key
        if not m:
            out.append(FallbackConfig(provider, "", info["label"], False,
                                      f"{info['label']}: name a model, for example {provider}:<model> in FALLBACK_PROVIDER."))
        elif not key or "paste" in key.lower() or "your-key" in key.lower():
            env = f"{provider.upper()}_API_KEY"
            out.append(FallbackConfig(provider, m, info["label"], False,
                                      f"{info['label']}: add {env} to backend/.env. Free key at {info['key_url']}"))
        else:
            out.append(FallbackConfig(provider, m, info["label"], True, api_key=key))
    return out


def config() -> FallbackConfig:
    """One summary of the whole chain, for the status badge."""
    items = chain()
    if not items:
        return FallbackConfig("none", "", "Off", False, "Turned off (FALLBACK_PROVIDER=none)")
    ready = [c for c in items if c.ready]
    problems = "; ".join(c.problem for c in items if not c.ready)
    label = " → ".join(c.label for c in ready) if ready else "off"
    return FallbackConfig(
        provider=items[0].provider if len(items) == 1 else "chain",
        model=ready[0].model if len(ready) == 1 else "",
        label=label,
        ready=bool(ready),
        problem=problems,
    )


def status() -> list[dict]:
    now = time.time()
    return [
        {"provider": c.provider, "model": c.model, "name": c.name, "ready": c.ready, "problem": c.problem,
         "cooldown_seconds": max(0, round(_cooldown_until.get(c.key, 0) - now))}
        for c in chain()
    ]


def reset_cooldowns() -> None:
    _cooldown_until.clear()


def _messages(question: str, history: list[dict], topics: list[str]) -> list[dict]:
    system = SYSTEM
    if topics:
        system += " For context, the course lectures cover: " + ", ".join(topics[:20]) + "."
    msgs = [{"role": "system", "content": system}]
    for turn in history[-6:]:
        role = "assistant" if turn.get("role") == "assistant" else "user"
        content = str(turn.get("content", "")).replace(PREFIX, "").strip()[:1500]
        if content:
            msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": question})
    return msgs


def _retry_after(resp) -> float:
    headers = getattr(resp, "headers", None) or {}
    try:
        return min(3600.0, max(1.0, float(headers.get("retry-after") or headers.get("Retry-After"))))
    except (TypeError, ValueError):
        return COOLDOWN_LIMIT


def _remote(cfg: FallbackConfig, messages: list[dict]) -> str:
    url = PROVIDERS[cfg.provider]["base_url"] + "/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
    if cfg.provider == "openrouter":
        headers["X-Title"] = "Lecture Tutor"
    env = f"{cfg.provider.upper()}_API_KEY"
    try:
        resp = httpx.post(url, headers=headers, json={"model": cfg.model, "messages": messages, "temperature": 0.4},
                          timeout=settings.fallback_timeout)
    except httpx.TimeoutException as exc:
        raise _TryNext(f"{cfg.name} took too long to reply.", COOLDOWN_ERROR) from exc
    except httpx.HTTPError as exc:
        raise _TryNext(f"Cannot reach {cfg.label}. Check your internet connection.", COOLDOWN_ERROR) from exc
    if resp.status_code in (401, 403):
        raise _TryNext(f"{cfg.label} rejected the API key. Check {env} in backend/.env.", COOLDOWN_CONFIG)
    if resp.status_code == 404:
        raise _TryNext(f"{cfg.label} has no model '{cfg.model}'. Check the model name in backend/.env.", COOLDOWN_CONFIG)
    if resp.status_code == 429:
        wait = _retry_after(resp)
        raise _TryNext(f"{cfg.name} free-tier limit reached (skipping it for {int(wait)}s).", wait)
    if resp.status_code >= 400:
        detail = resp.text[:200].replace("\n", " ")
        raise _TryNext(f"{cfg.name} returned error {resp.status_code}: {detail}", COOLDOWN_ERROR)
    try:
        return resp.json()["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, ValueError) as exc:
        raise _TryNext(f"{cfg.name} sent a reply in an unexpected format.", COOLDOWN_ERROR) from exc


def _local(cfg: FallbackConfig, messages: list[dict], question: str) -> str:
    from . import llm

    system = messages[0]["content"]
    convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages[1:-1])
    prompt = (f"Earlier conversation:\n{convo}\n\n" if convo else "") + f"Question: {question}"
    previous = llm._model_override
    llm.use_model(cfg.model)
    try:
        return llm.ask_llm(prompt, system=system, temperature=0.4)
    except RuntimeError as exc:
        raise _TryNext(f"Local model failed: {exc}", COOLDOWN_ERROR) from exc
    finally:
        llm.use_model(previous)


def general_answer(question: str, history: list[dict] | None = None,
                   topics: list[str] | None = None) -> GeneralAnswer:
    """Try each model in the chain until one answers. Raises FallbackError if none can."""
    items = chain()
    ready = [c for c in items if c.ready]
    if not ready:
        raise FallbackError(config().problem or "No outside-lecture model is set up.")
    now = time.time()
    candidates = [c for c in ready if _cooldown_until.get(c.key, 0) <= now]
    if not candidates:  # everything is cooling down: try them anyway rather than give up
        candidates = sorted(ready, key=lambda c: _cooldown_until.get(c.key, 0))

    messages = _messages(question, history or [], topics or [])
    skipped: list[str] = []
    for cfg in candidates:
        try:
            if cfg.provider == "mock":
                from .llm_mock import mock_reply

                text = mock_reply("TASK: general\nQUESTION: " + question)
            elif cfg.provider == "local":
                text = _local(cfg, messages, question)
            else:
                text = _remote(cfg, messages)
            text = strip_thinking(text).strip()
            if not text:
                raise _TryNext(f"{cfg.name} returned an empty answer.", COOLDOWN_ERROR)
        except _TryNext as exc:
            _cooldown_until[cfg.key] = time.time() + exc.cooldown
            skipped.append(str(exc))
            continue
        _cooldown_until.pop(cfg.key, None)
        return GeneralAnswer(text=text, answered_by=cfg.name, skipped=skipped)
    raise FallbackError(" ".join(skipped))

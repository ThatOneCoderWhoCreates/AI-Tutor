"""Answers for questions outside the lectures, with a fake HTTP (HyperText Transfer Protocol) service."""
import httpx
import pytest
from fastapi.testclient import TestClient

from app import fallback
from app.config import settings
from app.main import app
from app.rag.answer import NOT_COVERED, is_refusal
from app.schemas import Citation

CIT = [Citation(n=1, chunk_id="c", lecture_id="l", start=0, end=1, concept="x")]


class FakeResponse:
    def __init__(self, status, body=None, text=""):
        self.status_code = status
        self._body = body or {}
        self.text = text

    def json(self):
        return self._body


def use(monkeypatch, **kw):
    monkeypatch.setattr(fallback, "settings", type(settings)(**kw))
    fallback.reset_cooldowns()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_refusal_detection():
    assert is_refusal("The lecture does not cover this [1].", CIT)
    assert is_refusal(NOT_COVERED, [])
    assert is_refusal("The sources provided do not mention transformers.", [])
    assert not is_refusal("Unlabeled data don't include labels [1].", CIT)
    long = "Overfitting means memorizing noise [1]. " * 4 + "The lecture does not cover dropout."
    assert not is_refusal(long, CIT)
    wordy = ("The provided lecture sources do not discuss the time complexity of merge sort. The text only covers "
             "heaps, arrays, and union-find structures without mentioning merge sort [1] [2] [3] [4]. Therefore, "
             "based strictly on the given materials, I cannot answer your question about merge sort.")
    assert is_refusal(wordy, CIT)
    mixed = ("A heap keeps the top item ready [1]. The sources do not cover heap sort. "
             "In a max heap the largest value is on top [2]. Removing it reorganizes the heap [2].")
    assert not is_refusal(mixed, CIT)


def test_config_explains_missing_key(monkeypatch):
    use(monkeypatch, fallback_provider="gemini", fallback_api_key="")
    cfg = fallback.config()
    assert not cfg.ready and "aistudio.google.com" in cfg.problem and "GEMINI_API_KEY" in cfg.problem
    use(monkeypatch, fallback_provider="none")
    assert fallback.config().provider == "none"


def test_gemini_request_and_answer(monkeypatch):
    sent = {}

    def fake_post(url, headers, json, timeout):
        sent.update(url=url, headers=headers, body=json)
        return FakeResponse(200, {"choices": [{"message": {"content": "<think>x</think>Transformers use attention."}}]})

    monkeypatch.setattr(httpx, "post", fake_post)
    use(monkeypatch, fallback_provider="gemini", fallback_api_key="abc123")
    history = [{"role": "user", "content": "What is ML?"}, {"role": "assistant", "content": fallback.PREFIX + "\n\nML is..."}]
    out = fallback.general_answer("What is a transformer?", history=history, topics=["gradient descent"])
    assert out.text == "Transformers use attention."
    assert out.answered_by == "Gemini (gemini-3.5-flash-lite)" and out.skipped == []
    assert sent["url"] == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer abc123"
    assert sent["body"]["model"] == "gemini-3.5-flash-lite"
    msgs = sent["body"]["messages"]
    assert msgs[0]["role"] == "system" and "gradient descent" in msgs[0]["content"]
    assert msgs[2]["content"] == "ML is..."
    assert msgs[-1] == {"role": "user", "content": "What is a transformer?"}


@pytest.mark.parametrize("status,phrase", [(401, "API key"), (404, "model name"), (429, "limit")])
def test_readable_errors(monkeypatch, status, phrase):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResponse(status, text="err"))
    use(monkeypatch, fallback_provider="groq", fallback_api_key="k")
    with pytest.raises(fallback.FallbackError, match=phrase):
        fallback.general_answer("hi")


def test_ask_uses_fallback_for_out_of_lecture_question(client, monkeypatch):
    use(monkeypatch, fallback_provider="mock")
    r = client.post("/ask", json={"question": "Who won the football world cup?", "verify": True,
                                  "history": [{"role": "user", "content": "hello"}]}).json()
    assert r["source"] == "general"
    assert r["answer"].startswith(fallback.PREFIX)
    assert r["citations"] == [] and r["claims"] is None


def test_ask_still_answers_from_lecture(client, monkeypatch):
    use(monkeypatch, fallback_provider="mock")
    r = client.post("/ask", json={"question": "What is overfitting?"}).json()
    assert r["source"] == "lecture" and r["citations"]


def test_ask_without_fallback_and_broken_fallback(client, monkeypatch):
    use(monkeypatch, fallback_provider="none")
    r = client.post("/ask", json={"question": "Who won the football world cup?"}).json()
    assert r["source"] == "none" and r["answer"] == NOT_COVERED and r["note"] is None

    use(monkeypatch, fallback_provider="gemini", fallback_api_key="")
    r = client.post("/ask", json={"question": "Who won the football world cup?"}).json()
    assert r["source"] == "none" and "GEMINI_API_KEY" in r["note"]


def test_health_reports_fallback(client):
    assert client.get("/health").json()["fallback"]["provider"] == "none"


def test_chain_switches_when_limit_reached(monkeypatch):
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append((url, json["model"], headers["Authorization"]))
        if "googleapis" in url:
            return FakeResponse(429, text="quota")
        return FakeResponse(200, {"choices": [{"message": {"content": "Answer from Groq."}}]})

    monkeypatch.setattr(httpx, "post", fake_post)
    use(monkeypatch, fallback_provider="gemini,groq,mock", gemini_api_key="g-key", groq_api_key="q-key")
    out = fallback.general_answer("hi")
    assert out.text == "Answer from Groq." and out.answered_by.startswith("Groq")
    assert "limit reached" in out.skipped[0]
    assert calls[0][2] == "Bearer g-key" and calls[1][2] == "Bearer q-key"

    calls.clear()
    out = fallback.general_answer("again")
    assert [c[0] for c in calls] == ["https://api.groq.com/openai/v1/chat/completions"]  # Gemini is cooling down
    assert out.skipped == []
    assert next(s for s in fallback.status() if s["provider"] == "gemini")["cooldown_seconds"] > 0


def test_chain_uses_last_option_and_reports_all_failures(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResponse(429))
    use(monkeypatch, fallback_provider="gemini:gemini-3.8-flash,gemini,mock", gemini_api_key="k")
    out = fallback.general_answer("hi")
    assert out.answered_by == "Mock" and len(out.skipped) == 2
    assert "gemini-3.8-flash" in out.skipped[0] and "gemini-3.5-flash-lite" in out.skipped[1]

    use(monkeypatch, fallback_provider="gemini,groq", gemini_api_key="k", groq_api_key="k")
    with pytest.raises(fallback.FallbackError, match="Gemini.*Groq"):
        fallback.general_answer("hi")


def test_summary_of_chain(monkeypatch):
    use(monkeypatch, fallback_provider="gemini,groq,local", gemini_api_key="k")
    cfg = fallback.config()
    assert cfg.ready and cfg.label.startswith("Gemini → Local")
    assert "GROQ_API_KEY" in cfg.problem

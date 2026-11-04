"""The Ollama call, tested with a fake HTTP (HyperText Transfer Protocol) response instead of a running server."""
import httpx
import pytest

from app import llm
from app.config import settings


class FakeResponse:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)


def test_ollama_payload_and_thinking_removed(monkeypatch):
    sent = {}

    def fake_post(url, json, timeout):
        sent.update(url=url, body=json)
        return FakeResponse(200, {"message": {"content": '<think>plan</think>{"questions": []}'}})

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(llm, "settings", type(settings)(llm_provider="ollama", llm_model="qwen2.5:7b"))
    reply = llm.ask_llm("TASK: quiz\nhello", json_mode=True)
    assert reply == '{"questions": []}'
    assert sent["url"].endswith("/api/chat")
    assert sent["body"]["model"] == "qwen2.5:7b"
    assert sent["body"]["format"] == "json"
    assert sent["body"]["options"]["num_ctx"] >= 8192


def test_ollama_offline_gives_clear_message(monkeypatch):
    def fake_post(url, json, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(llm, "settings", type(settings)(llm_provider="ollama", llm_model="qwen2.5:7b"))
    with pytest.raises(RuntimeError, match="ollama serve"):
        llm.ask_llm("hi")


def test_missing_model_tells_you_to_pull(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda url, json, timeout: FakeResponse(404, {}))
    monkeypatch.setattr(llm, "settings", type(settings)(llm_provider="ollama", llm_model="qwen2.5:14b"))
    with pytest.raises(RuntimeError, match="ollama pull qwen2.5:14b"):
        llm.ask_llm("hi")


def test_json_retry_recovers_from_bad_reply(monkeypatch):
    replies = iter(["not json at all", '{"ok": true}'])
    monkeypatch.setattr(llm, "ask_llm", lambda *a, **k: next(replies))
    assert llm.ask_llm_json("TASK: x") == {"ok": True}


def test_thinking_off_and_retry_for_models_without_it(monkeypatch):
    sent = []

    def fake_post(url, json, timeout):
        sent.append(dict(json))
        if "think" in json:
            return FakeResponse(400, {"error": "registry.ollama.ai/library/qwen2.5:7b does not support thinking"})
        return FakeResponse(200, {"message": {"content": "ok"}})

    monkeypatch.setattr(FakeResponse, "text", property(lambda self: str(self._body)), raising=False)
    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(llm, "settings", type(settings)(llm_provider="ollama", llm_model="qwen2.5:7b", llm_think="off"))
    llm._no_think_models.clear()
    assert llm.ask_llm("hi") == "ok"
    assert sent[0]["think"] is False and "think" not in sent[1]
    sent.clear()
    assert llm.ask_llm("again") == "ok"
    assert len(sent) == 1 and "think" not in sent[0]


def test_retry_without_json_mode_when_model_does_not_support_it(monkeypatch):
    sent = []

    def fake_post(url, json, timeout):
        sent.append(dict(json))
        if "format" in json:
            return FakeResponse(501, {"error": "not implemented"})
        return FakeResponse(200, {"message": {"content": '{"ok": 1}'}})

    monkeypatch.setattr(FakeResponse, "text", property(lambda self: str(self._body)), raising=False)
    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(llm, "settings", type(settings)(llm_provider="ollama", llm_model="qwen3.5:9b", llm_think="off"))
    llm._no_format_models.clear()
    assert llm.ask_llm_json("TASK: x") == {"ok": 1}
    assert "format" in sent[0] and "format" not in sent[1]
    sent.clear()
    llm.ask_llm_json("TASK: y")
    assert len(sent) == 1 and "format" not in sent[0]


def test_other_ollama_errors_are_readable(monkeypatch):
    monkeypatch.setattr(FakeResponse, "text", property(lambda self: "boom"), raising=False)
    monkeypatch.setattr(httpx, "post", lambda url, json, timeout: FakeResponse(500, {}))
    monkeypatch.setattr(llm, "settings", type(settings)(llm_provider="ollama", llm_model="m", llm_think="off"))
    with pytest.raises(RuntimeError, match="Ollama error 500"):
        llm.ask_llm("hi")

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["chunks"] == 10


def test_concepts_listed_in_course_order(client):
    names = [c["name"] for c in client.get("/concepts").json()]
    assert names[0] == "loss minimization"
    assert "gradient descent" in names and "overfitting" in names


def test_ask_returns_grounded_answer_with_timestamps(client):
    r = client.post("/ask", json={"question": "What is overfitting?", "verify": True})
    body = r.json()
    assert r.status_code == 200
    assert body["citations"], body
    assert any(c["concept"] == "overfitting" for c in body["citations"])
    assert body["claims"] and all("supported" in c for c in body["claims"])


def test_hybrid_search_finds_learning_rate(client):
    hits = app.state.retriever.search("what happens if the learning rate is too large", k=3)
    assert hits[0].concept == "learning rate"


def test_full_adaptive_loop(client):
    sid = "test-student"
    first = client.post("/next", json={"student_id": sid}).json()
    assert first["concept"] == "loss minimization"
    assert first["segments"][0]["start"] == 0

    quiz = client.post("/quiz", json={"concept": "gradient descent", "n": 2, "student_id": sid}).json()
    assert len(quiz["questions"]) == 2
    assert "answer_index" not in quiz["questions"][0]

    mastery = None
    for _ in range(4):
        q = client.post("/quiz", json={"concept": "gradient descent", "n": 1, "student_id": sid}).json()["questions"][0]
        from app.tutor.quiz import find_question

        right = find_question(q["id"])["answer_index"]
        res = client.post("/answer", json={"student_id": sid, "question_id": q["id"], "choice": right}).json()
        assert res["correct"] is True
        assert mastery is None or res["mastery"] > mastery
        mastery = res["mastery"]

    q = client.post("/quiz", json={"concept": "overfitting", "n": 1, "student_id": sid}).json()["questions"][0]
    wrong = (find_question(q["id"])["answer_index"] + 1) % 4
    res = client.post("/answer", json={"student_id": sid, "question_id": q["id"], "choice": wrong}).json()
    assert res["correct"] is False

    nxt = client.post("/next", json={"student_id": sid}).json()
    assert nxt["concept"] == "overfitting"


def test_rename_merges_topics(client):
    r = client.post("/concepts/rename", json={"old": "stochastic gradient descent", "new": "gradient descent"})
    assert r.json()["changed_chunks"] == 1
    names = [c["name"] for c in client.get("/concepts").json()]
    assert "stochastic gradient descent" not in names


def test_prompts_focus_on_concepts_not_examples():
    from app.tutor import concepts, quiz

    assert "Never name a topic after the example" in concepts.PROMPT
    assert "{lecture}" in concepts.PROMPT
    assert "Never ask about the details of an" in quiz.PROMPT
    assert "{lecture}" in quiz.PROMPT


def test_relabel_lecture(client):
    import time

    from app.config import settings
    from app.store import slugify

    (settings.quizzes_dir / f"{slugify('overfitting')}.json").write_text('{"questions": []}')
    r = client.post("/lectures/demo/relabel")
    assert r.status_code == 200 and r.json()["status"] == "labeling"
    for _ in range(50):
        if client.get("/lectures/demo/status").json()["status"] == "ready":
            break
        time.sleep(0.1)
    assert client.get("/lectures/demo/status").json()["status"] == "ready"
    assert not (settings.quizzes_dir / f"{slugify('overfitting')}.json").exists()
    assert client.get("/concepts").json()
    assert client.post("/lectures/nope/relabel").status_code == 400


def test_singular_question_finds_plural_topic(client, monkeypatch):
    from app.rag import answer as answer_mod

    calls = []

    def fake_llm(prompt, system=None, json_mode=False, temperature=0.2):
        calls.append(prompt)
        if "NOTE:" not in prompt:
            return "The lecture does not cover this."
        return "Regularization adds a penalty to the loss [1]."

    monkeypatch.setattr(answer_mod, "ask_llm", fake_llm)
    r = client.post("/ask", json={"question": "what is regularizations"}).json()
    assert len(calls) == 2 and "regularization" in calls[1]
    assert r["source"] == "lecture" and r["citations"][0]["concept"] == "regularization"

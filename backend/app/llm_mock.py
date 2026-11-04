"""Offline fake LLM so the whole app runs with no API key and no Ollama.

Every real prompt in this project starts with a line `TASK: <name>`.
The mock reads that line and returns a plausible reply in the right format.
Replies are dumb but correctly shaped, which is all the UI and tests need.
"""
from __future__ import annotations

import json
import re

STOP = set(
    "a an the is are was were be of to in on for and or with what how why when which does do this that it as by "
    "from at into about can we you i explain tell me".split()
)


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP and len(t) > 2}


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.split()) >= 5]


def _section(prompt: str, name: str) -> str:
    m = re.search(rf"^{name}:\s*(.*?)(?=^[A-Z_]+:|\Z)", prompt, re.S | re.M)
    return m.group(1).strip() if m else ""


def mock_reply(prompt: str, system: str | None = None, json_mode: bool = False) -> str:
    task_match = re.search(r"^TASK:\s*(\w+)", prompt, re.M)
    task = task_match.group(1) if task_match else ""

    if task == "answer":
        return _answer(prompt)
    if task == "quiz":
        return _quiz(prompt)
    if task == "concepts":
        return _concepts(prompt)
    if task == "verify":
        return _verify(prompt)
    if task == "queries":
        return _queries(prompt)
    if task == "general":
        q = _section(prompt, "QUESTION") or "this question"
        return f"This is a mock general-knowledge answer about: {q}"
    if json_mode:
        return json.dumps({"reply": "mock"})
    return "This is a mock reply. Set LLM_PROVIDER in backend/.env to use a real model."


def _answer(prompt: str) -> str:
    question = _section(prompt, "QUESTION")
    sources = re.findall(r"^\[(\d+)\][^\n]*\n(.*?)(?=^\[\d+\]|\Z)", _section(prompt, "SOURCES"), re.S | re.M)
    q_tokens = _tokens(question)
    scored = []
    for num, body in sources:
        overlap = len(q_tokens & _tokens(body))
        scored.append((overlap, int(num), body))
    scored.sort(key=lambda x: (-x[0], x[1]))
    useful = [s for s in scored if s[0] > 0][:2]
    if not useful:
        return "The lecture does not cover this."
    lines = []
    for _, num, body in useful:
        sents = _sentences(body) or [body.strip()]
        lines.append(f"{sents[0]} [{num}]")
    return " ".join(lines)


def _quiz(prompt: str) -> str:
    concept = _section(prompt, "CONCEPT") or "this topic"
    n_match = re.search(r"^COUNT:\s*(\d+)", prompt, re.M)
    n = int(n_match.group(1)) if n_match else 3
    material = _section(prompt, "MATERIAL")
    chunk_ids = re.findall(r"^\(chunk (\S+?)\)", material, re.M)
    sents = _sentences(re.sub(r"^\(chunk \S+?\)", "", material, flags=re.M))
    distractors = [
        f"{concept.capitalize()} is only used for storing data in databases.",
        f"{concept.capitalize()} was not discussed as a technique in machine learning.",
        f"{concept.capitalize()} always guarantees a perfect result with no tuning.",
    ]
    try:
        offset = len(json.loads(_section(prompt, "AVOID") or "[]"))
    except json.JSONDecodeError:
        offset = 0
    questions = []
    for i in range(min(n, max(1, len(sents)))):
        k = offset + i
        correct = sents[k % len(sents)] if sents else f"{concept} is covered in the lecture."
        options = distractors[:]
        pos = k % 4
        options.insert(pos, correct)
        questions.append(
            {
                "question": f"According to the lecture, which statement about {concept} is correct? (version {k + 1})",
                "options": options,
                "answer_index": pos,
                "explanation": f"The lecture says: {correct}",
                "source_chunk_id": chunk_ids[i % len(chunk_ids)] if chunk_ids else "",
            }
        )
    return json.dumps({"questions": questions})


def _concepts(prompt: str) -> str:
    raw = _section(prompt, "CHUNKS")
    try:
        chunks = json.loads(raw)
    except json.JSONDecodeError:
        return json.dumps({"labels": {}})
    labels = {}
    for c in chunks:
        title = (c.get("slide_title") or "").strip()
        if not title:
            words = [w for w in re.findall(r"[A-Za-z]+", c.get("text", "")) if w.lower() not in STOP]
            title = " ".join(words[:2]) or "general"
        labels[c["id"]] = title.lower()
    return json.dumps({"labels": labels})


def _queries(prompt: str) -> str:
    text = _section(prompt, "EXCERPT")
    n_match = re.search(r"Write (\d+) different questions", prompt)
    n = int(n_match.group(1)) if n_match else 3
    words = [w for w in re.findall(r"[A-Za-z]+", text) if w.lower() not in STOP and len(w) > 3]
    out = []
    for i in range(n):
        picked = words[i * 3 : i * 3 + 3] or words[:3] or ["this"]
        phrase = " ".join(picked).lower()
        out.append(phrase if i == 0 else f"what does the lecture say about {phrase}")
    return json.dumps({"questions": out})


def _verify(prompt: str) -> str:
    claim = _section(prompt, "CLAIM")
    evidence = _section(prompt, "EVIDENCE")
    c_tokens = _tokens(claim)
    ratio = len(c_tokens & _tokens(evidence)) / max(1, len(c_tokens))
    supported = ratio >= 0.5
    return json.dumps({"supported": supported, "reason": f"mock word overlap {ratio:.2f}"})

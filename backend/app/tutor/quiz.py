"""Practice questions for one topic, generated from that topic's lecture chunks.

Questions are cached in data/quizzes/<topic>.json so the same question can be
checked later, and so we do not pay for regeneration every time.
"""
from __future__ import annotations

import hashlib
import json
import random
import re

from ..config import settings
from ..llm import ask_llm_json
from ..schemas import Chunk
from ..store import read_json, slugify, write_json

OPTION_PREFIX = re.compile(r"^\s*(?:\(?[A-Da-d][).:]|[A-Da-d]\s*-)\s+")

MAX_CACHED = 15
MAX_MATERIAL_CHARS = 6000

SYSTEM = "You write fair multiple-choice practice questions for students, based only on the given lecture material."

PROMPT = """TASK: quiz
Write {n} multiple-choice questions that test understanding of the topic below.
Rules:
- Every question must be answerable from MATERIAL alone. Do not use outside facts.
- Exactly 4 options per question. Exactly one option is correct.
- Wrong options must be plausible, not silly.
- Test understanding of the CONCEPT, not exact wording. Use simple English.
- The lecture may use examples or stories (people, products, songs, houses). Never ask about the details of an
  example, such as what a person in the story likes or which item was chosen. A student who understands the
  concept but forgot the story must still be able to answer.
  Bad: "Paul prefers songs with which characteristics?"
  Good: "A music app predicts which new songs a user will like from songs they rated before. Which type of
  machine learning is this?"
- Good question types: what the concept is, why it is used, how it works, how it differs from a related concept,
  and which concept fits a new short scenario.
- Do not repeat any question listed in AVOID.
- "source_chunk_id" is the chunk id the answer comes from.
Reply with JSON only, in this format:
{{"questions": [{{"question": "...", "options": ["...", "...", "...", "..."], "answer_index": 0, "explanation": "...", "source_chunk_id": "..."}}]}}

LECTURE: {lecture}
CONCEPT: {concept}
COUNT: {n}
AVOID: {avoid}
MATERIAL:
{material}
"""


def _quiz_path(concept: str):
    return settings.quizzes_dir / f"{slugify(concept)}.json"


def question_id(concept: str, text: str) -> str:
    return f"{slugify(concept)}__{hashlib.md5(text.encode()).hexdigest()[:8]}"


def _validator(chunk_ids: set[str], default_chunk: str):
    def validate(data):
        items = data.get("questions") if isinstance(data, dict) else data
        if not isinstance(items, list) or not items:
            raise ValueError('Expected {"questions": [...]} with at least one question')
        clean = []
        for i, q in enumerate(items):
            if not isinstance(q, dict):
                raise ValueError(f"Question {i} is not an object")
            text = str(q.get("question", "")).strip()
            options = q.get("options")
            if not text:
                raise ValueError(f"Question {i} has no text")
            if not isinstance(options, list) or len(options) != 4:
                raise ValueError(f"Question {i} must have exactly 4 options")
            options = [OPTION_PREFIX.sub("", str(o)).strip() for o in options]
            if len(set(o.lower() for o in options)) != 4 or not all(options):
                raise ValueError(f"Question {i} has empty or duplicate options")
            idx = q.get("answer_index", q.get("answer", q.get("correct_answer")))
            if isinstance(idx, str):
                s = idx.strip()
                if len(s) == 1 and s.upper() in "ABCD":
                    idx = "ABCD".index(s.upper())
                elif s.isdigit():
                    idx = int(s)
                else:
                    s = OPTION_PREFIX.sub("", s).strip().lower()
                    idx = next((j for j, o in enumerate(options) if o.lower() == s), None)
            if not isinstance(idx, int) or not 0 <= idx <= 3:
                raise ValueError(f"Question {i} needs answer_index 0-3")
            src = str(q.get("source_chunk_id", "")).strip()
            clean.append({
                "question": text,
                "options": options,
                "answer_index": idx,
                "explanation": str(q.get("explanation", "")).strip(),
                "source_chunk_id": src if src in chunk_ids else default_chunk,
            })
        return clean
    return validate


def _material(chunks: list[Chunk]) -> str:
    parts, total = [], 0
    for c in chunks:
        block = f"(chunk {c.id}) {c.text}"
        if total + len(block) > MAX_MATERIAL_CHARS:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)


def _lecture_titles(chunks: list[Chunk]) -> str:
    titles = []
    for lid in dict.fromkeys(c.lecture_id for c in chunks):
        meta = read_json(settings.lectures_dir / lid / "meta.json", {}) or {}
        titles.append(meta.get("title") or lid)
    return "; ".join(titles) or "(title unknown)"


def generate_questions(concept: str, chunks: list[Chunk], n: int, avoid: list[str]) -> list[dict]:
    ids = {c.id for c in chunks}
    raw = ask_llm_json(
        PROMPT.format(n=n, lecture=_lecture_titles(chunks), concept=concept, avoid=json.dumps(avoid[-10:]),
                      material=_material(chunks)),
        system=SYSTEM,
        validate=_validator(ids, chunks[0].id),
    )
    rng = random.Random(concept)
    out = []
    for q in raw:
        order = list(range(4))
        rng.shuffle(order)
        options = [q["options"][j] for j in order]
        q["answer_index"] = order.index(q["answer_index"])
        q["options"] = options
        q["id"] = question_id(concept, q["question"])
        q["concept"] = concept
        out.append(q)
    return out


def load_bank(concept: str) -> list[dict]:
    return (read_json(_quiz_path(concept), {}) or {}).get("questions", [])


def save_bank(concept: str, questions: list[dict]) -> None:
    write_json(_quiz_path(concept), {"concept": concept, "questions": questions})


def get_quiz(concept: str, chunks: list[Chunk], n: int, seen_ids: set[str], fresh: bool = False) -> list[dict]:
    """Return n questions, preferring ones this student has not answered yet."""
    bank = load_bank(concept)
    unseen = [q for q in bank if q["id"] not in seen_ids]
    if (fresh or len(unseen) < n) and len(bank) < MAX_CACHED:
        new = generate_questions(concept, chunks, n, avoid=[q["question"] for q in bank])
        known = {q["id"] for q in bank}
        bank += [q for q in new if q["id"] not in known]
        save_bank(concept, bank)
        unseen = [q for q in bank if q["id"] not in seen_ids]
    picked = unseen[:n]
    if len(picked) < n:
        rest = [q for q in bank if q not in picked]
        random.shuffle(rest)
        picked += rest[: n - len(picked)]
    return picked


def find_question(qid: str) -> dict | None:
    slug = qid.split("__")[0]
    data = read_json(settings.quizzes_dir / f"{slug}.json", {}) or {}
    for q in data.get("questions", []):
        if q["id"] == qid:
            return q
    return None

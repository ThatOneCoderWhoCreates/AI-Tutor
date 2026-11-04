"""Step 3b: label each chunk with a topic (concept) using the LLM.

Slide titles are passed as strong hints. Labels are made in batches, and
earlier labels are passed forward so the same topic keeps the same name.
This step is never perfect, so the UI has a "review topics" screen to rename or merge.
"""
from __future__ import annotations

import json
import re

from ..llm import ask_llm_json
from ..schemas import Chunk
from ..rag.slides import slide_title

BATCH = 20

PROMPT = """TASK: concepts
You are organizing chunks of a lecture into study topics. The lecture is: {lecture}
Give every chunk a short topic name: 1 to 4 words, lowercase, e.g. "supervised learning".
Rules:
- A topic must be a concept of the lecture's subject, something a student would be tested on.
- Lecturers often use examples, stories or analogies (a person's music taste, buying a house, sorting fruit).
  Never name a topic after the example. Name the concept the example is used to teach.
  For example, a chunk about predicting which songs Paul likes, in a machine learning lecture,
  is "recommendation systems" or "supervised learning", never "music preferences".
- Neighboring chunks that discuss the same idea must get exactly the same name.
- The slide title is a strong hint. Use it when it names a real concept.
- Reuse a name from EXISTING whenever it fits instead of inventing a new one. Prefer fewer, broader topics.
- Do not use vague names like "lecture", "summary", "introduction" or "continued". Name the actual idea.
Reply with JSON only: {{"labels": {{"<chunk id>": "<topic name>"}}}}

EXISTING: {existing}

CHUNKS:
{chunks}
"""


def _clean_label(label: str) -> str:
    label = re.sub(r"\s+", " ", str(label)).strip().strip(".").lower()
    return label[:60]


def label_concepts(chunks: list[Chunk], lecture_title: str = "") -> list[Chunk]:
    existing: list[str] = []
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        payload = [
            {"id": c.id, "slide_title": slide_title(c.slide_text), "text": c.text[:400]}
            for c in batch
        ]
        ids = {c.id for c in batch}

        def validate(data):
            labels = data.get("labels", data) if isinstance(data, dict) else data
            if isinstance(labels, list):
                # small models sometimes reply [{"id": "...", "topic": "..."}] instead of a dict
                labels = {str(x.get("id")): x.get("topic") or x.get("label") or x.get("concept")
                          for x in labels if isinstance(x, dict)}
            if not isinstance(labels, dict):
                raise ValueError('Expected {"labels": {...}}')
            labels = {k: v for k, v in labels.items() if isinstance(v, str) and v.strip()}
            missing = ids - set(labels)
            if missing:
                raise ValueError(f"Missing labels for chunk ids: {sorted(missing)}")
            return {k: _clean_label(v) for k, v in labels.items() if k in ids}

        labels = ask_llm_json(
            PROMPT.format(lecture=lecture_title or "(title unknown)", existing=json.dumps(existing), chunks=json.dumps(payload, ensure_ascii=False)),
            validate=validate,
        )
        for c in batch:
            c.concept = labels.get(c.id) or "general"
            if c.concept not in existing:
                existing.append(c.concept)
    return chunks


def relabel_lecture(lecture_id: str, log=print) -> list[str]:
    """Redo topic labeling for a lecture that is already processed (no new transcription).

    Cached practice questions for the old and new topics are deleted so they are written again.
    """
    from ..config import settings
    from ..store import lecture_dir, load_chunks, read_json, save_chunks, slugify, write_json

    meta_path = lecture_dir(lecture_id) / "meta.json"
    meta = read_json(meta_path, {}) or {}
    chunks = load_chunks(lecture_id)
    if not chunks:
        raise RuntimeError(f"Lecture '{lecture_id}' has no chunks.")
    old = {c.concept for c in chunks}
    meta.update(status="labeling", error=None)
    write_json(meta_path, meta)
    try:
        log(f"Labeling {len(chunks)} chunks of '{meta.get('title', lecture_id)}'...")
        label_concepts(chunks, meta.get("title", lecture_id))
        save_chunks(lecture_id, chunks)
    except Exception as exc:
        meta.update(status="ready", error=f"Topic labeling failed: {str(exc)[:300]}")
        write_json(meta_path, meta)
        raise
    new = list(dict.fromkeys(c.concept for c in chunks))
    for topic in old | set(new):
        (settings.quizzes_dir / f"{slugify(topic)}.json").unlink(missing_ok=True)
    meta.update(status="ready")
    write_json(meta_path, meta)
    log(f"{len(new)} topics: {', '.join(new)}")
    return new


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Redo topic labeling for an already processed lecture")
    ap.add_argument("lecture_id")
    relabel_lecture(ap.parse_args().lecture_id)
    print("Done. Restart the backend or call POST /reindex.")

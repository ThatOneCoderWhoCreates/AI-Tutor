"""Reading and writing data on disk. Plain JSON files, no database needed.

data/
  lectures/<lecture_id>/meta.json        {"title": ..., "video": "video.mp4"}
  lectures/<lecture_id>/chunks.json      list of Chunk
  lectures/<lecture_id>/transcript.json  Whisper segments (cache)
  lectures/<lecture_id>/slides.json      slide changes + OCR text (cache)
  students/<student_id>.json             mastery state
  quizzes/<concept-slug>.json            generated questions (cache)
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

from .config import settings
from .schemas import Chunk, Segment

_lock = threading.Lock()


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "x"


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with _lock:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp.replace(path)


def lecture_dir(lecture_id: str) -> Path:
    return settings.lectures_dir / lecture_id


def list_lectures() -> list[dict]:
    out = []
    if not settings.lectures_dir.exists():
        return out
    for d in sorted(settings.lectures_dir.iterdir()):
        if not d.is_dir():
            continue
        meta = read_json(d / "meta.json", {}) or {}
        video = meta.get("video")
        out.append(
            {
                "id": d.name,
                "title": meta.get("title", d.name),
                "status": meta.get("status", "ready" if (d / "chunks.json").exists() else "missing"),
                "has_video": bool(video and (d / video).exists()),
                "duration": meta.get("duration"),
            }
        )
    return out


def load_chunks(lecture_id: str | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    for lec in list_lectures():
        if lecture_id and lec["id"] != lecture_id:
            continue
        raw = read_json(lecture_dir(lec["id"]) / "chunks.json", []) or []
        for item in raw:
            item.setdefault("lecture_id", lec["id"])
            chunks.append(Chunk(**item))
    return chunks


def save_chunks(lecture_id: str, chunks: list[Chunk]) -> None:
    write_json(lecture_dir(lecture_id) / "chunks.json", [c.model_dump() for c in chunks])


def merge_segments(chunks: list[Chunk], gap: float = 2.0) -> list[Segment]:
    """Turn the chunks of one concept into as few continuous video segments as possible."""
    segs: list[Segment] = []
    for c in sorted(chunks, key=lambda c: (c.lecture_id, c.start)):
        if segs and segs[-1].lecture_id == c.lecture_id and c.start - segs[-1].end <= gap:
            segs[-1].end = max(segs[-1].end, c.end)
        else:
            segs.append(Segment(lecture_id=c.lecture_id, start=c.start, end=c.end))
    return segs


def concept_map(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    """Concept name -> its chunks, in the order concepts first appear in the course."""
    lecture_order = {lec["id"]: i for i, lec in enumerate(list_lectures())}
    ordered = sorted(chunks, key=lambda c: (lecture_order.get(c.lecture_id, 0), c.start))
    out: dict[str, list[Chunk]] = {}
    for c in ordered:
        if c.concept:
            out.setdefault(c.concept, []).append(c)
    return out


def rename_concept(old: str, new: str) -> int:
    """Rename a concept everywhere. If `new` already exists, the two are merged."""
    new = new.strip().lower()
    changed = 0
    for lec in list_lectures():
        chunks = load_chunks(lec["id"])
        touched = False
        for c in chunks:
            if c.concept == old:
                c.concept = new
                changed += 1
                touched = True
        if touched:
            save_chunks(lec["id"], chunks)
    for sfile in settings.students_dir.glob("*.json") if settings.students_dir.exists() else []:
        state = read_json(sfile, {})
        if old in state.get("mastery", {}):
            old_p = state["mastery"].pop(old)
            state["mastery"][new] = max(old_p, state["mastery"].get(new, 0.0))
            a_old = state.get("attempts", {}).pop(old, [0, 0])
            a_new = state.setdefault("attempts", {}).get(new, [0, 0])
            state["attempts"][new] = [a_old[0] + a_new[0], a_old[1] + a_new[1]]
            write_json(sfile, state)
    old_quiz = settings.quizzes_dir / f"{slugify(old)}.json"
    if old_quiz.exists():
        old_quiz.unlink()
    return changed

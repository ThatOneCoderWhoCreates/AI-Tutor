"""Step 3a: merge transcript segments + slide text into chunks.

A new chunk starts when:
  - the slide changes (and the current chunk is at least MIN_SECONDS long), or
  - the current chunk would get longer than MAX_CHUNK_SECONDS.
"""
from __future__ import annotations

from bisect import bisect_right

from ..config import settings
from ..schemas import Chunk

MIN_SECONDS = 20.0


def _slide_index(slide_starts: list[float], t: float) -> int:
    return max(0, bisect_right(slide_starts, t) - 1)


def build_chunks(lecture_id: str, segments: list[dict], slides: list[dict],
                 max_seconds: float | None = None, min_seconds: float = MIN_SECONDS) -> list[Chunk]:
    max_seconds = max_seconds or settings.max_chunk_seconds
    slide_starts = [s["start"] for s in slides]

    groups: list[dict] = []
    cur: dict | None = None
    for seg in segments:
        sidx = _slide_index(slide_starts, seg["start"]) if slides else -1
        if cur is not None:
            duration = cur["end"] - cur["start"]
            slide_changed = sidx != cur["slides"][-1]
            too_long = seg["end"] - cur["start"] > max_seconds
            if (slide_changed and duration >= min_seconds) or too_long:
                groups.append(cur)
                cur = None
        if cur is None:
            cur = {"start": seg["start"], "end": seg["end"], "texts": [], "slides": [sidx]}
        cur["texts"].append(seg["text"])
        cur["end"] = seg["end"]
        if sidx != cur["slides"][-1]:
            cur["slides"].append(sidx)
    if cur is not None:
        groups.append(cur)

    chunks = []
    for i, g in enumerate(groups, start=1):
        slide_texts = []
        for sidx in g["slides"]:
            if sidx >= 0 and slides[sidx]["text"] and slides[sidx]["text"] not in slide_texts:
                slide_texts.append(slides[sidx]["text"])
        chunks.append(Chunk(
            id=f"{lecture_id}_{i:03d}",
            lecture_id=lecture_id,
            text=" ".join(t.strip() for t in g["texts"]).strip(),
            slide_text="\n".join(slide_texts),
            start=round(g["start"], 2),
            end=round(g["end"], 2),
        ))
    return chunks

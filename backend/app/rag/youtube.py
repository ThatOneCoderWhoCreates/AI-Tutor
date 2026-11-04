"""Ingest a lecture straight from a YouTube link.

The video is downloaded with yt-dlp (720p is enough for reading slides), then it
goes through the normal pipeline. Optionally, YouTube's own captions can be used
instead of Whisper. That is much faster, but auto-generated captions have no
punctuation and are usually less accurate than Whisper.

Command line (run from the backend folder):
    python -m app.rag.ingest --url "https://www.youtube.com/watch?v=..." --id lec1
    python -m app.rag.ingest --url "https://www.youtube.com/watch?v=..." --id lec1 --captions
"""
from __future__ import annotations

import html
import re
from pathlib import Path

from ..store import lecture_dir, write_json

FORMAT = "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/bv*[height<=720]+ba/b"
CAPTION_LANGS = ["en", "en-US", "en-GB", "en-IN", "en-orig"]

_TIME = r"(\d+:)?\d{2}:\d{2}[.,]\d{3}"


def _seconds(stamp: str) -> float:
    parts = stamp.replace(",", ".").split(":")
    parts = [float(p) for p in parts]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts
    return h * 3600 + m * 60 + s


def _clean_caption_line(line: str) -> str:
    line = re.sub(r"<[^>]+>", "", line)
    line = html.unescape(line).replace("​", "")
    line = re.sub(r"\[(music|applause|laughter)\]", "", line, flags=re.I)
    return re.sub(r"\s+", " ", line).strip()


def parse_vtt(text: str) -> list[dict]:
    """Turn a WebVTT caption file into transcript segments.

    YouTube auto-captions "roll": every cue repeats the previous line and adds
    a new one. We keep only the lines that were not in the previous cue, so
    each spoken word appears once.
    """
    segments: list[dict] = []
    prev_lines: list[str] = []
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    for block in blocks:
        lines = block.strip().split("\n")
        idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if idx is None:
            continue
        m = re.match(rf"\s*({_TIME})\s*-->\s*({_TIME})", lines[idx])
        if not m:
            continue
        start, end = _seconds(m.group(1)), _seconds(m.group(3))
        cue_lines = [c for c in (_clean_caption_line(ln) for ln in lines[idx + 1 :]) if c]
        new = [ln for ln in cue_lines if ln not in prev_lines]
        if cue_lines:
            prev_lines = cue_lines
        if new and end - start > 0.05:
            segments.append({"start": round(start, 2), "end": round(end, 2), "text": " ".join(new)})
    return _merge_short(segments)


def _merge_short(segments: list[dict], min_words: int = 12, max_seconds: float = 15.0) -> list[dict]:
    """Join tiny caption pieces into sentence-sized segments."""
    out: list[dict] = []
    for seg in segments:
        if out and len(out[-1]["text"].split()) < min_words and seg["end"] - out[-1]["start"] <= max_seconds:
            out[-1]["text"] += " " + seg["text"]
            out[-1]["end"] = max(out[-1]["end"], seg["end"])
        else:
            out.append(dict(seg))
    for cur, nxt in zip(out, out[1:]):
        if cur["start"] < nxt["start"] < cur["end"]:
            cur["end"] = nxt["start"]
    return out


def video_id(url: str) -> str | None:
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None


def download(url: str, lecture_id: str, want_captions: bool, log=print) -> dict:
    """Download the video (and captions if asked). Returns info about what was saved."""
    import yt_dlp

    d = lecture_dir(lecture_id)
    d.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": FORMAT,
        "merge_output_format": "mp4",
        "outtmpl": str(d / "video.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "overwrites": True,
    }
    if want_captions:
        opts.update({
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": CAPTION_LANGS,
            "subtitlesformat": "vtt",
        })
    log("0/4 Downloading from YouTube...")
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    video = d / "video.mp4"
    if not video.exists():
        found = sorted(p for p in d.glob("video.*") if p.suffix not in (".vtt", ".part", ".json"))
        if not found:
            raise RuntimeError("yt-dlp finished but no video file was saved")
        video = found[0]

    result = {"video": video, "title": info.get("title") or lecture_id, "duration": info.get("duration"),
              "captions": None}
    if want_captions:
        vtts = sorted(d.glob("video*.vtt"), key=lambda p: (".en." not in p.name, p.name))
        if vtts:
            segments = parse_vtt(vtts[0].read_text(encoding="utf-8", errors="ignore"))
            if segments:
                write_json(d / "transcript.json", segments)
                result["captions"] = len(segments)
                log(f"    Using YouTube captions ({len(segments)} segments) instead of Whisper")
        if result["captions"] is None:
            log("    No English captions found, Whisper will be used")
    return result


def ingest_youtube(url: str, lecture_id: str | None = None, title: str | None = None,
                   use_captions: bool = False, force: bool = False, log=print) -> int:
    from .ingest import _set_status, ingest_video

    lecture_id = lecture_id or f"yt-{video_id(url) or 'video'}"
    d = lecture_dir(lecture_id)
    d.mkdir(parents=True, exist_ok=True)
    _set_status(lecture_id, title=title or lecture_id, status="downloading", source_url=url, error=None)
    if force:
        for name in ("transcript.json", "slides.json"):
            (d / name).unlink(missing_ok=True)
    try:
        info = download(url, lecture_id, want_captions=use_captions, log=log)
    except Exception as exc:
        _set_status(lecture_id, status="error", error=f"Download failed: {str(exc)[:400]}")
        raise
    return ingest_video(Path(info["video"]), lecture_id, title or info["title"], force=False, log=log)

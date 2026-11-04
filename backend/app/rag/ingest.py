"""Full pipeline: video in -> labeled, indexed chunks out.

Command line (run from the backend folder):
    python -m app.rag.ingest path/to/lecture.mp4 --id lec1 --title "Lecture 1: Gradient Descent"
    python -m app.rag.ingest --url "https://www.youtube.com/watch?v=..." --id lec1

Results are cached in data/lectures/<id>/ so re-running is fast.
Use --force to redo transcription and slide reading.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from ..store import lecture_dir, read_json, save_chunks, write_json
from ..tutor.concepts import label_concepts
from .chunker import build_chunks
from .slides import detect_slides, video_duration
from .transcribe import transcribe


def _set_status(lecture_id: str, **fields) -> None:
    path = lecture_dir(lecture_id) / "meta.json"
    meta = read_json(path, {}) or {}
    meta.update(fields)
    write_json(path, meta)


def ingest_video(video: Path, lecture_id: str, title: str | None = None, force: bool = False,
                 log=print) -> int:
    d = lecture_dir(lecture_id)
    d.mkdir(parents=True, exist_ok=True)
    target = d / f"video{video.suffix.lower()}"
    if video.resolve() != target.resolve():
        shutil.copy2(video, target)
    _set_status(lecture_id, title=title or lecture_id, video=target.name, status="transcribing",
                duration=round(video_duration(target), 1), error=None)

    try:
        t0 = time.time()
        transcript_path = d / "transcript.json"
        if force or not transcript_path.exists():
            log("1/4 Transcribing audio (this is the slow step)...")
            segments = transcribe(target, d / "work")
            write_json(transcript_path, segments)
        else:
            log("1/4 Using cached transcript")
            segments = read_json(transcript_path, [])
        log(f"    {len(segments)} transcript segments")

        _set_status(lecture_id, status="reading_slides")
        slides_path = d / "slides.json"
        if force or not slides_path.exists():
            log("2/4 Detecting slide changes and reading slide text...")
            slides = detect_slides(target, d / "frames")
            write_json(slides_path, slides)
        else:
            log("2/4 Using cached slides")
            slides = read_json(slides_path, [])
        log(f"    {len(slides)} distinct slides")

        _set_status(lecture_id, status="chunking")
        log("3/4 Building chunks...")
        chunks = build_chunks(lecture_id, segments, slides)
        log(f"    {len(chunks)} chunks")

        _set_status(lecture_id, status="labeling")
        log("4/4 Labeling topics with the LLM...")
        label_concepts(chunks, title or lecture_id)
        save_chunks(lecture_id, chunks)
        topics = sorted({c.concept for c in chunks})
        log(f"    {len(topics)} topics: {', '.join(topics)}")

        _set_status(lecture_id, status="ready")
        log(f"Done in {time.time() - t0:.0f}s. Restart the backend (or call POST /reindex) to search it.")
        return len(chunks)
    except Exception as exc:
        _set_status(lecture_id, status="error", error=str(exc)[:500])
        raise


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest one lecture from a video file or a YouTube link")
    p.add_argument("video", type=Path, nargs="?", help="path to a video file")
    p.add_argument("--url", default=None, help="YouTube link instead of a file")
    p.add_argument("--captions", action="store_true", help="with --url: use YouTube captions instead of Whisper")
    p.add_argument("--id", default=None, help="short lecture id, e.g. lec1")
    p.add_argument("--title", default=None)
    p.add_argument("--force", action="store_true", help="redo transcription and slide OCR")
    args = p.parse_args()
    if args.url:
        from .youtube import ingest_youtube

        ingest_youtube(args.url, args.id, args.title, use_captions=args.captions, force=args.force)
    elif args.video:
        if not args.id:
            p.error("--id is required for a video file")
        ingest_video(args.video, args.id, args.title, force=args.force)
    else:
        p.error("give a video file or --url")


if __name__ == "__main__":
    main()

"""Step 2: find the moments the slide changes and read the slide text.

Pass 1: compare tiny grayscale frames sampled once per second (fast).
Pass 2: grab one full-resolution frame just after each change and OCR it with Tesseract.
Near-duplicate slides (animations, small camera moves) are merged.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from ..config import settings
from .transcribe import run_ffmpeg


def video_duration(video: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video)],
        capture_output=True, text=True,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


SAMPLE_W, SAMPLE_H = 160, 90


def detect_changes(video: Path, threshold: float, fps: float = 1.0) -> list[float]:
    """Sample tiny grayscale frames and flag the ones where enough pixels changed.

    `threshold` is the fraction of pixels that must change (0.01 = 1%).
    Slides on a plain background change only a few percent of pixels, so this
    is far more sensitive than ffmpeg's scene score. False alarms (webcam
    movement, cursor) are removed later when OCR text turns out identical.
    """
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
         "-vf", f"fps={fps},scale={SAMPLE_W}:{SAMPLE_H},format=gray",
         "-an", "-f", "rawvideo", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='ignore')[-800:]}")
    frame_size = SAMPLE_W * SAMPLE_H
    n = len(proc.stdout) // frame_size
    if n < 2:
        return []
    frames = np.frombuffer(proc.stdout[: n * frame_size], dtype=np.uint8).reshape(n, SAMPLE_H, SAMPLE_W)
    diffs = np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16)) > 30
    changed = diffs.reshape(n - 1, -1).mean(axis=1)
    return [round((i + 1) / fps, 2) for i, frac in enumerate(changed) if frac > threshold]


def ocr_image(path: Path) -> str:
    if not shutil.which("tesseract"):
        return ""
    proc = subprocess.run(["tesseract", str(path), "stdout", "--psm", "3"], capture_output=True, text=True)
    lines = [ln.strip() for ln in proc.stdout.splitlines()]
    return "\n".join(ln for ln in lines if len(re.findall(r"[A-Za-z]", ln)) >= 2)


def slide_title(text: str) -> str:
    for line in text.splitlines():
        if len(re.findall(r"[A-Za-z]", line)) >= 3:
            return line[:80]
    return ""


def _similar(a: str, b: str) -> bool:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta or not tb:
        return False
    return len(ta & tb) / max(1, len(ta | tb)) >= 0.8


def detect_slides(video: Path, frames_dir: Path, threshold: float | None = None) -> list[dict]:
    threshold = settings.scene_threshold if threshold is None else threshold
    frames_dir.mkdir(parents=True, exist_ok=True)
    duration = video_duration(video)
    starts = [0.0]
    for t in detect_changes(video, threshold):
        if t <= 1.0:
            continue
        if t - starts[-1] <= 1.0 and len(starts) > 1:
            starts[-1] = t
        else:
            starts.append(t)

    slides: list[dict] = []
    for i, t in enumerate(starts):
        grab = min(t + 1.0, max(0.0, duration - 0.1)) if duration else t + 1.0
        frame = frames_dir / f"slide_{i:04d}.png"
        try:
            run_ffmpeg(["-ss", f"{grab:.2f}", "-i", str(video), "-frames:v", "1", str(frame)])
        except RuntimeError:
            continue
        text = ocr_image(frame)
        if slides and _similar(slides[-1]["text"], text):
            continue
        slides.append({"start": round(t, 2), "frame": frame.name, "text": text, "title": slide_title(text)})
    return slides

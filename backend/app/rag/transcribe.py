"""Step 1: speech to text with timestamps.

WHISPER_BACKEND=mlx     -> mlx-whisper, fast on Apple-silicon Macs (pip install mlx-whisper)
WHISPER_BACKEND=faster  -> faster-whisper, works on any CPU (pip install faster-whisper)
Both run fully on your own machine.

WHISPER_MODEL=small is a good default. Use base for speed or large-v3-turbo for accuracy.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from ..config import settings


MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


def run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    cmd = ["ffmpeg", "-hide_banner", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-800:]}")
    return proc


def extract_audio(video: Path, out: Path) -> Path:
    run_ffmpeg(["-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(out)])
    return out


def _clean(segments: list[dict]) -> list[dict]:
    out = []
    for s in segments:
        text = (s.get("text") or "").strip()
        if text:
            out.append({"start": round(float(s["start"]), 2), "end": round(float(s["end"]), 2), "text": text})
    return out


def transcribe(video: Path, work_dir: Path) -> list[dict]:
    backend = settings.whisper_backend
    model = settings.whisper_model
    work_dir.mkdir(parents=True, exist_ok=True)

    if backend == "mlx":
        import mlx_whisper

        wav = extract_audio(video, work_dir / "audio.wav")
        repo = model if "/" in model else MLX_REPOS.get(model, f"mlx-community/whisper-{model}-mlx")
        result = mlx_whisper.transcribe(str(wav), path_or_hf_repo=repo)
        return _clean(result["segments"])

    if backend == "faster":
        from faster_whisper import WhisperModel

        wav = extract_audio(video, work_dir / "audio.wav")
        wm = WhisperModel(model, device="cpu", compute_type="int8")
        segments, _info = wm.transcribe(str(wav), vad_filter=True)
        return _clean([{"start": s.start, "end": s.end, "text": s.text} for s in segments])

    raise ValueError(f"Unknown WHISPER_BACKEND: {backend}")

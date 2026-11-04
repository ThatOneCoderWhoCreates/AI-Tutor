"""End-to-end ingestion on a generated 3-slide video. Whisper is replaced by a fake transcript."""
import shutil
import subprocess

import pytest

from app.rag import ingest as ingest_mod
from app.store import load_chunks, read_json, lecture_dir

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def _make_video(path):
    parts = []
    for i, color in enumerate(["white", "navy", "gray"]):
        part = path.parent / f"part{i}.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=640x360:d=30",
                        "-r", "10", str(part)], check=True)
        parts.append(part)
    listing = path.parent / "list.txt"
    listing.write_text("".join(f"file '{p.name}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-c", "copy", str(path)], check=True)


def test_ingest_video_end_to_end(tmp_path, monkeypatch):
    video = tmp_path / "lec.mp4"
    _make_video(video)
    fake = [{"start": i * 5.0, "end": i * 5.0 + 5, "text": f"The lecturer explains point number {i} in detail."}
            for i in range(18)]
    monkeypatch.setattr(ingest_mod, "transcribe", lambda v, d: fake)

    n = ingest_mod.ingest_video(video, "gen1", "Generated lecture", log=lambda *_: None)
    chunks = load_chunks("gen1")
    assert n == len(chunks) == 3
    assert [c.start for c in chunks] == [0.0, 30.0, 60.0]
    assert all(c.concept for c in chunks)
    assert read_json(lecture_dir("gen1") / "meta.json")["status"] == "ready"

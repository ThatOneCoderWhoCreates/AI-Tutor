import shutil
import subprocess

import pytest

from app.rag import youtube
from app.rag.answer import normalize_citations
from app.llm import extract_json, strip_thinking
from app.store import lecture_dir, load_chunks, read_json

ROLLING_VTT = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:02.500 align:start position:0%
today<00:00:00.500><c> we</c><00:00:00.900><c> talk</c><00:00:01.300><c> about</c>

00:00:02.500 --> 00:00:02.510 align:start position:0%
today we talk about

00:00:02.510 --> 00:00:05.000 align:start position:0%
today we talk about
gradient<00:00:03.000><c> descent</c><00:00:03.400><c> and</c><00:00:03.900><c> why</c><00:00:04.200><c> it</c><00:00:04.600><c> works</c>

00:00:05.000 --> 00:00:05.010 align:start position:0%
gradient descent and why it works

00:00:05.010 --> 00:00:08.000 align:start position:0%
gradient descent and why it works
[Music] the<00:00:05.500><c> gradient</c><00:00:06.000><c> points</c><00:00:06.500><c> uphill</c> &amp; we go down
"""


def test_parse_vtt_removes_rolling_duplicates():
    segs = youtube.parse_vtt(ROLLING_VTT)
    text = " ".join(s["text"] for s in segs)
    assert text == "today we talk about gradient descent and why it works the gradient points uphill & we go down"
    assert segs[0]["start"] == 0.0
    assert segs[-1]["end"] == 8.0
    assert all(s["end"] >= s["start"] for s in segs)


def test_video_id_from_common_link_formats():
    assert youtube.video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s") == "dQw4w9WgXcQ"
    assert youtube.video_id("https://youtu.be/dQw4w9WgXcQ?si=abc") == "dQw4w9WgXcQ"
    assert youtube.video_id("https://www.youtube.com/live/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert youtube.video_id("https://example.com/video") is None


def test_normalize_citations_from_small_models():
    assert normalize_citations("It goes downhill [Source 2].") == "It goes downhill [2]."
    assert normalize_citations("Both [source 1, 3] agree (Source 4).") == "Both [1, 3] agree [4]."
    assert normalize_citations("See [1][2].") == "See [1] [2]."


def test_strip_thinking_tags():
    assert strip_thinking("<think>let me reason</think>\nThe answer [1].") == "The answer [1]."
    assert extract_json('<think>hmm {"no": 1}</think> {"labels": {"a": "x"}}') == {"labels": {"a": "x"}}


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_ingest_youtube_with_captions(monkeypatch, tmp_path):
    """yt-dlp is faked: it 'downloads' a generated video plus a caption file."""

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=True):
            out = self.opts["outtmpl"].replace("%(ext)s", "mp4")
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=white:s=320x180:d=8",
                            "-r", "5", out], check=True)
            if self.opts.get("writeautomaticsub"):
                with open(out.replace(".mp4", ".en.vtt"), "w") as f:
                    f.write(ROLLING_VTT)
            return {"title": "Fake YouTube Lecture", "duration": 8}

    import yt_dlp

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    n = youtube.ingest_youtube("https://youtu.be/dQw4w9WgXcQ", "yt1", use_captions=True, log=lambda *_: None)
    meta = read_json(lecture_dir("yt1") / "meta.json")
    assert n >= 1
    assert meta["status"] == "ready"
    assert meta["title"] == "Fake YouTube Lecture"
    assert meta["source_url"].startswith("https://youtu.be/")
    assert "gradient descent" in " ".join(c.text for c in load_chunks("yt1"))

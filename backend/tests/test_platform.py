"""Windows and macOS handling, tested by pretending to be each system."""
import sys

import pytest

from app import platform_tools as pt


@pytest.fixture(autouse=True)
def fresh_cache():
    pt.find_tool.cache_clear()
    yield
    pt.find_tool.cache_clear()


def test_finds_tesseract_in_program_files_on_windows(tmp_path, monkeypatch):
    exe = tmp_path / "Tesseract-OCR" / "tesseract.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    monkeypatch.setattr(pt, "IS_WINDOWS", True)
    monkeypatch.setattr(pt, "IS_MAC", False)
    monkeypatch.setattr(pt.shutil, "which", lambda name: None)
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    assert pt.find_tool("tesseract") == str(exe)
    assert pt.install_hint("tesseract") == "winget install UB-Mannheim.TesseractOCR"


def test_finds_winget_ffmpeg_on_windows(tmp_path, monkeypatch):
    exe = tmp_path / "Microsoft" / "WinGet" / "Packages" / "Gyan.FFmpeg_abc" / "ffmpeg-7.1-full_build" / "bin" / "ffmpeg.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    monkeypatch.setattr(pt, "IS_WINDOWS", True)
    monkeypatch.setattr(pt.shutil, "which", lambda name: None)
    for var in ("ProgramFiles", "ProgramFiles(x86)"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert pt.find_tool("ffmpeg") == str(exe)


def test_env_override_wins(tmp_path, monkeypatch):
    exe = tmp_path / "my-ffmpeg"
    exe.write_bytes(b"")
    monkeypatch.setenv("FFMPEG_PATH", f'"{exe}"')
    assert pt.find_tool("ffmpeg") == str(exe)


def test_missing_tool_gives_install_command(monkeypatch):
    monkeypatch.setattr(pt, "IS_WINDOWS", False)
    monkeypatch.setattr(pt, "IS_MAC", True)
    monkeypatch.setattr(pt.shutil, "which", lambda name: None)
    monkeypatch.setattr(pt, "_mac_candidates", lambda name: [])
    with pytest.raises(pt.ToolMissing, match="brew install tesseract"):
        pt.tool_path("tesseract")


def test_run_tool_decodes_utf8(monkeypatch):
    monkeypatch.setattr(pt, "tool_path", lambda name: sys.executable)
    out = pt.run_tool("ffmpeg", ["-c", "import sys; sys.stdout.buffer.write('café → naïve'.encode('utf-8'))"])
    assert out.stdout == "café → naïve"


def test_whisper_backend_choice(monkeypatch):
    monkeypatch.setattr(pt, "IS_APPLE_SILICON", False)
    assert pt.default_whisper_backend() == "faster"

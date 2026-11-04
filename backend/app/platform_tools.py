"""Find and run the outside programs (ffmpeg, ffprobe, tesseract) on macOS, Windows and Linux.

- Looks on PATH first, then in the usual install folders (Homebrew on Mac, Program Files and
  winget on Windows), so a fresh Windows install works even before PATH is updated.
- FFMPEG_PATH, FFPROBE_PATH and TESSERACT_CMD in backend/.env override the search.
- Output is always decoded as UTF-8, so slide text with special characters does not crash on
  Windows, where the default console encoding is not UTF-8.
"""
from __future__ import annotations

import glob
import os
import platform
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

IS_WINDOWS = os.name == "nt"
IS_MAC = sys.platform == "darwin"
IS_APPLE_SILICON = IS_MAC and platform.machine() == "arm64"

ENV_OVERRIDE = {"ffmpeg": "FFMPEG_PATH", "ffprobe": "FFPROBE_PATH", "tesseract": "TESSERACT_CMD"}

INSTALL_HINT = {
    "ffmpeg": {"mac": "brew install ffmpeg", "win": "winget install Gyan.FFmpeg", "linux": "sudo apt install ffmpeg"},
    "ffprobe": {"mac": "brew install ffmpeg", "win": "winget install Gyan.FFmpeg", "linux": "sudo apt install ffmpeg"},
    "tesseract": {"mac": "brew install tesseract", "win": "winget install UB-Mannheim.TesseractOCR",
                  "linux": "sudo apt install tesseract-ocr"},
}


def _windows_candidates(name: str) -> list[str]:
    exe = f"{name}.exe"
    roots = [os.environ.get(k, "") for k in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")]
    out = []
    for root in filter(None, roots):
        if name == "tesseract":
            out.append(str(Path(root) / "Tesseract-OCR" / exe))
            out.append(str(Path(root) / "Programs" / "Tesseract-OCR" / exe))
        else:
            out.append(str(Path(root) / "ffmpeg" / "bin" / exe))
            # winget installs Gyan.FFmpeg under LOCALAPPDATA\Microsoft\WinGet\Packages\Gyan.FFmpeg*\ffmpeg-*\bin
            out += glob.glob(str(Path(root) / "Microsoft" / "WinGet" / "Packages" / "Gyan.FFmpeg*" / "*" / "bin" / exe))
    out.append(str(Path("C:/ffmpeg/bin") / exe))
    return out


def _mac_candidates(name: str) -> list[str]:
    return [f"/opt/homebrew/bin/{name}", f"/usr/local/bin/{name}"]


@lru_cache(maxsize=None)
def find_tool(name: str) -> str | None:
    override = os.environ.get(ENV_OVERRIDE.get(name, ""), "").strip().strip('"')
    if override and Path(override).exists():
        return override
    found = shutil.which(name)
    if found:
        return found
    candidates = _windows_candidates(name) if IS_WINDOWS else _mac_candidates(name) if IS_MAC else []
    return next((c for c in candidates if Path(c).exists()), None)


def install_hint(name: str) -> str:
    key = "win" if IS_WINDOWS else "mac" if IS_MAC else "linux"
    return INSTALL_HINT.get(name, {}).get(key, f"install {name}")


class ToolMissing(RuntimeError):
    pass


def tool_path(name: str) -> str:
    path = find_tool(name)
    if not path:
        raise ToolMissing(f"'{name}' was not found. Install it with: {install_hint(name)} "
                          f"(then open a new terminal), or set {ENV_OVERRIDE.get(name, name.upper())} in backend/.env.")
    return path


def run_tool(name: str, args: list[str], binary: bool = False) -> subprocess.CompletedProcess:
    """Run ffmpeg / ffprobe / tesseract. Text output is decoded as UTF-8; binary=True keeps raw bytes."""
    cmd = [tool_path(name), *args]
    kwargs: dict = {"capture_output": True}
    if not binary:
        kwargs.update(text=True, encoding="utf-8", errors="replace")
    if IS_WINDOWS:
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console window pops up
    return subprocess.run(cmd, **kwargs)


def default_whisper_backend() -> str:
    """mlx on Apple-silicon Macs (fast, uses the Mac GPU), faster-whisper everywhere else."""
    if IS_APPLE_SILICON:
        try:
            import importlib.util

            if importlib.util.find_spec("mlx_whisper"):
                return "mlx"
        except Exception:
            pass
    return "faster"

"""One-time setup for Lecture Tutor. Works the same on macOS, Windows and Linux.

    macOS:    /opt/homebrew/bin/python3.12 install.py
    Windows:  py -3.12 install.py

It creates the Python environment (backend/.venv), installs the Python and frontend
packages, creates backend/.env, checks ffmpeg, Tesseract and Ollama, and downloads the
local model named in backend/.env. Running it again is safe: finished steps are skipped
or quickly confirmed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
VENV = BACKEND / ".venv"
IS_WINDOWS = os.name == "nt"
IS_MAC = sys.platform == "darwin"

HINTS = {
    "ffmpeg": ("brew install ffmpeg", "winget install Gyan.FFmpeg"),
    "tesseract": ("brew install tesseract", "winget install UB-Mannheim.TesseractOCR"),
    "npm": ("brew install node", "winget install OpenJS.NodeJS.LTS"),
    "ollama": ("brew install ollama", "winget install Ollama.Ollama"),
}


def say(msg: str) -> None:
    print(f"\n==> {msg}", flush=True)


def hint(tool: str) -> str:
    mac, win = HINTS[tool]
    return win if IS_WINDOWS else mac if IS_MAC else f"install {tool} with your package manager"


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("   $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def find(tool: str) -> str | None:
    found = shutil.which(tool)
    if found:
        return found
    sys.path.insert(0, str(BACKEND))
    try:
        from app.platform_tools import find_tool  # searches the usual install folders too

        return find_tool(tool)
    except Exception:
        return None


def read_env_value(key: str, default: str) -> str:
    env = BACKEND / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(f"{key}="):
                return line.split("=", 1)[1].strip() or default
    return default


def main() -> int:
    v = sys.version_info
    if not ((3, 10) <= (v.major, v.minor) <= (3, 13)):
        cmd = "py -3.12 install.py" if IS_WINDOWS else "/opt/homebrew/bin/python3.12 install.py"
        print(f"This is Python {v.major}.{v.minor}. Please use Python 3.12 (3.10 to 3.13 work).\n"
              f"Install it ({'winget install Python.Python.3.12' if IS_WINDOWS else 'brew install python@3.12'}), then run: {cmd}")
        return 1

    say("Python environment (backend/.venv)")
    if venv_python().exists():
        print("   already exists")
    else:
        venv.create(VENV, with_pip=True)
        print("   created")

    say("Python packages (the first time takes several minutes)")
    run([venv_python(), "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
    run([venv_python(), "-m", "pip", "install", "-r", BACKEND / "requirements.txt"])

    say("Settings file (backend/.env)")
    env = BACKEND / ".env"
    if env.exists():
        print("   already exists, left unchanged")
    else:
        shutil.copyfile(BACKEND / ".env.example", env)
        print("   created from .env.example")

    say("Frontend packages")
    npm = find("npm")
    if npm:
        run([npm, "install"], cwd=FRONTEND)
    else:
        print(f"   Node.js is missing. Install it with: {hint('npm')}  (then open a new terminal and run this again)")

    say("Checking outside programs")
    missing = []
    for tool, why in (("ffmpeg", "needed to process lecture videos"),
                      ("tesseract", "reads slide text; optional but recommended")):
        path = find(tool)
        print(f"   {tool:<10} {'found: ' + path if path else 'MISSING (' + why + '). Install: ' + hint(tool)}")
        if not path:
            missing.append(tool)

    ollama = find("ollama")
    model = read_env_value("LLM_MODEL", "qwen3.5:9b")
    provider = read_env_value("LLM_PROVIDER", "ollama")
    if provider != "ollama":
        print(f"   ollama     skipped (LLM_PROVIDER={provider})")
    elif not ollama:
        print(f"   ollama     MISSING (runs the local model). Install: {hint('ollama')}")
        missing.append("ollama")
    else:
        print(f"   ollama     found: {ollama}")
        say(f"Local model {model} (several GB the first time)")
        try:
            run([ollama, "pull", model])
        except subprocess.CalledProcessError:
            print("   Could not download the model. Start Ollama (open the Ollama app, or run: ollama serve) "
                  f"and then run: ollama pull {model}")

    say("Done")
    if missing:
        print("   Install the missing programs listed above, open a new terminal, and run this script again.")
    py = "py" if IS_WINDOWS else "python3"
    print(f"   Start the app with:  {py} start.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Start Lecture Tutor with one command. Works the same on macOS, Windows and Linux.

    macOS:    python3 start.py
    Windows:  py start.py

Starts Ollama if it is not running, then the backend (port 8000) and the frontend
(port 5173), and opens the app in your browser. Press Ctrl+C once to stop everything.
Run install.py first.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
IS_WINDOWS = os.name == "nt"
APP_URL = "http://localhost:5173"
BACKEND_URL = "http://localhost:8000/health"
OLLAMA_URL = "http://localhost:11434/api/tags"


def venv_python() -> Path:
    return BACKEND / ".venv" / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def is_up(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2):
            return True
    except Exception:
        return False


def read_env_value(key: str, default: str) -> str:
    env = BACKEND / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(f"{key}="):
                return line.split("=", 1)[1].strip() or default
    return default


def spawn(cmd: list[str], cwd: Path, env: dict | None = None) -> subprocess.Popen:
    kwargs: dict = {"cwd": cwd, "env": env}
    if IS_WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True  # own process group, so Ctrl+C can stop its children too
    return subprocess.Popen([str(c) for c in cmd], **kwargs)


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if IS_WINDOWS:
        # stops npm and the node process it started, plus uvicorn's reloader child
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def main() -> int:
    # closing the terminal window (or another program asking it to stop) shuts everything down too
    signal.signal(signal.SIGTERM, _interrupt)
    if hasattr(signal, "SIGBREAK"):  # Windows: Ctrl+Break or closing the console
        signal.signal(signal.SIGBREAK, _interrupt)
    if not IS_WINDOWS:
        signal.signal(signal.SIGHUP, _interrupt)
    py = venv_python()
    if not py.exists():
        print("The Python environment is missing. Run install.py first "
              f"({'py -3.12 install.py' if IS_WINDOWS else '/opt/homebrew/bin/python3.12 install.py'}).")
        return 1
    npm = shutil.which("npm")
    if not npm:
        print("npm was not found. Install Node.js, open a new terminal, and run install.py again.")
        return 1
    if not (FRONTEND / "node_modules").exists():
        print("Frontend packages are missing. Run install.py first.")
        return 1

    procs: list[subprocess.Popen] = []
    if read_env_value("LLM_PROVIDER", "ollama") == "ollama" and not is_up(OLLAMA_URL):
        ollama = shutil.which("ollama")
        if ollama:
            print("Starting Ollama...")
            procs.append(spawn([ollama, "serve"], ROOT))
            for _ in range(20):
                if is_up(OLLAMA_URL):
                    break
                time.sleep(0.5)
        else:
            print("Ollama is not running and was not found. The app will start, but answers need Ollama.")

    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    print("Starting backend on http://localhost:8000 ...")
    procs.append(spawn([py, "-m", "uvicorn", "app.main:app", "--reload", "--port", "8000"], BACKEND, env))
    print(f"Starting frontend on {APP_URL} ...")
    procs.append(spawn([npm, "run", "dev"], FRONTEND))

    try:
        print("Waiting for the backend (the first start can take a minute)...")
        for _ in range(300):
            if is_up(BACKEND_URL):
                print(f"\nLecture Tutor is running: {APP_URL}\nPress Ctrl+C to stop.\n")
                webbrowser.open(APP_URL)
                break
            if any(p.poll() is not None for p in procs[-2:]):
                print("A part of the app stopped. See the messages above.")
                break
            time.sleep(1)
        while all(p.poll() is None for p in procs[-2:]):
            time.sleep(1)
        print("A part of the app stopped. See the messages above.")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        for p in reversed(procs):
            stop(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Tiny background-job runner for long tasks (search model training).

The status is kept in memory and in data/jobs/<name>.json, so the Lab page can
show progress and the last result even after the page is reloaded.
"""
from __future__ import annotations

import threading
import time
import traceback
from typing import Callable

from .config import settings
from .store import read_json, write_json

_lock = threading.Lock()
_running: dict[str, threading.Thread] = {}


def _path(name: str):
    return settings.data_dir / "jobs" / f"{name}.json"


def status(name: str) -> dict:
    data = read_json(_path(name), None) or {"name": name, "state": "idle"}
    if data.get("state") == "running" and name not in _running:
        data["state"] = "interrupted"
        data["message"] = "The backend restarted while this was running. Start it again."
    return data


def _update(name: str, **fields) -> None:
    data = read_json(_path(name), None) or {"name": name}
    data.update(fields)
    write_json(_path(name), data)


def start(name: str, fn: Callable[[Callable[[str, float | None], None]], dict]) -> dict:
    with _lock:
        if name in _running and _running[name].is_alive():
            raise RuntimeError("This job is already running.")

        def progress(message: str, fraction: float | None) -> None:
            _update(name, message=message, progress=None if fraction is None else round(fraction, 3))

        def run() -> None:
            try:
                result = fn(progress)
                _update(name, state="done", finished_at=int(time.time()), result=result, message="Finished")
            except Exception as exc:
                _update(name, state="error", finished_at=int(time.time()), message=str(exc)[:500],
                        trace=traceback.format_exc()[-2000:])
            finally:
                _running.pop(name, None)

        write_json(_path(name), {"name": name, "state": "running", "started_at": int(time.time()),
                                 "message": "Starting", "progress": 0})
        thread = threading.Thread(target=run, daemon=True)
        _running[name] = thread
        thread.start()
    return status(name)

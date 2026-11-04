"""Bayesian Knowledge Tracing (BKT): one mastery probability per student per topic.

Four numbers describe each topic:
  p_init   chance the student already knows it before practicing
  p_learn  chance they learn it after one practice question
  p_slip   chance they answer wrong even though they know it
  p_guess  chance they answer right without knowing it (1/4 for 4-option MCQ)

After each answer we use Bayes' rule to update "does the student know it?",
then add the chance they learned it from this attempt.

The four numbers start as fixed defaults. They can be learned from real answers
with Expectation-Maximization (see app/tutor/bkt_fit.py). Learned values are saved
in data/bkt_params.json and used automatically from then on.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from ..config import settings
from ..store import read_json, slugify, write_json


@dataclass(frozen=True)
class BKTParams:
    p_init: float = 0.2
    p_learn: float = 0.15
    p_slip: float = 0.1
    p_guess: float = 0.25

    def to_dict(self) -> dict:
        return {k: round(v, 4) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> "BKTParams":
        return cls(**{k: float(d[k]) for k in ("p_init", "p_learn", "p_slip", "p_guess")})


DEFAULT = BKTParams()
PARAMS = DEFAULT  # kept for older imports
MASTERED = 0.95

_UNSET = object()
_cache: dict = {"mtime": _UNSET, "data": None}


def _params_path():
    return settings.data_dir / "bkt_params.json"


def active_params() -> dict:
    """{"source": "default"|"fitted", "global": {...}, "per_topic": {...}, ...}"""
    path = _params_path()
    mtime = path.stat().st_mtime_ns if path.exists() else None
    if mtime != _cache["mtime"]:
        data = read_json(path, None) if mtime else None
        _cache.update(mtime=mtime, data=data)
    return _cache["data"] or {"source": "default", "global": DEFAULT.to_dict(), "per_topic": {}}


def params_for(concept: str) -> BKTParams:
    data = active_params()
    topic = data.get("per_topic", {}).get(concept)
    return BKTParams.from_dict(topic or data["global"])


def save_active_params(data: dict) -> None:
    write_json(_params_path(), data)
    _cache["mtime"] = _UNSET


def reset_params() -> None:
    _params_path().unlink(missing_ok=True)
    _cache["mtime"] = _UNSET


def bkt_update(p_known: float, correct: bool, params: BKTParams = DEFAULT) -> float:
    if correct:
        num = p_known * (1 - params.p_slip)
        den = num + (1 - p_known) * params.p_guess
    else:
        num = p_known * params.p_slip
        den = num + (1 - p_known) * (1 - params.p_guess)
    posterior = num / den if den else p_known
    return posterior + (1 - posterior) * params.p_learn


def _path(student_id: str):
    return settings.students_dir / f"{slugify(student_id)}.json"


def load_student(student_id: str) -> dict:
    state = read_json(_path(student_id), None)
    if not state:
        state = {"student_id": student_id, "mastery": {}, "attempts": {}, "history": []}
    return state


def save_student(state: dict) -> None:
    write_json(_path(state["student_id"]), state)


def mastery_of(state: dict, concept: str) -> float:
    if concept in state["mastery"]:
        return float(state["mastery"][concept])
    return params_for(concept).p_init


def record_answer(student_id: str, concept: str, question_id: str, correct: bool) -> float:
    state = load_student(student_id)
    new_p = bkt_update(mastery_of(state, concept), correct, params_for(concept))
    state["mastery"][concept] = round(new_p, 4)
    tries, right = state["attempts"].get(concept, [0, 0])
    state["attempts"][concept] = [tries + 1, right + int(correct)]
    state["history"].append(
        {"t": int(time.time()), "concept": concept, "question_id": question_id, "correct": correct, "mastery": round(new_p, 4)}
    )
    save_student(state)
    return new_p


def replay_all_students() -> int:
    """Recompute every student's mastery from their answer history with the current parameters."""
    if not settings.students_dir.exists():
        return 0
    count = 0
    for path in settings.students_dir.glob("*.json"):
        state = read_json(path, None)
        if not state:
            continue
        mastery: dict[str, float] = {}
        for h in state.get("history", []):
            c = h["concept"]
            p = mastery.get(c, params_for(c).p_init)
            mastery[c] = bkt_update(p, bool(h["correct"]), params_for(c))
            h["mastery"] = round(mastery[c], 4)
        state["mastery"] = {c: round(p, 4) for c, p in mastery.items()}
        write_json(path, state)
        count += 1
    return count

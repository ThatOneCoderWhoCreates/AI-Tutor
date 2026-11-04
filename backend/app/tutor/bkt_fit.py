"""Learn the four BKT numbers from answer data with Expectation-Maximization (EM).

BKT is a Hidden Markov Model (HMM) with two hidden states per student per topic:
"does not know" (0) and "knows" (1). We only see right/wrong answers.

EM repeats two steps until the numbers stop changing:
  E-step: with the current numbers, work out how likely the student was in each
          state at each answer (forward-backward algorithm).
  M-step: re-estimate the four numbers from those probabilities, for example
          p_guess = (expected right answers while not knowing) / (expected answers while not knowing).

Each run starts from several random starting points and keeps the best one,
because EM can get stuck in a poor local solution.

Command line (run from the backend folder):
    python -m app.tutor.bkt_fit --source simulated
    python -m app.tutor.bkt_fit --source app --apply
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np

from ..config import settings
from ..store import read_json, write_json
from .bkt import DEFAULT, BKTParams, bkt_update, replay_all_students, save_active_params

# Limits keep the model meaningful. Without them EM can "swap" the states
# (for example guess=0.9 and slip=0.9), which fits the data but means nothing.
BOUNDS = {"p_init": (0.01, 0.99), "p_learn": (0.01, 0.6), "p_slip": (0.01, 0.4), "p_guess": (0.01, 0.4)}
MIN_TOPIC_SEQUENCES = 20


@dataclass
class Sequence:
    student: str
    concept: str
    answers: list[int]


# ---------------- data ----------------
def sequences_from_app() -> list[Sequence]:
    """One sequence per (student, topic), in the order the answers were given."""
    out: list[Sequence] = []
    if not settings.students_dir.exists():
        return out
    for path in sorted(settings.students_dir.glob("*.json")):
        state = read_json(path, {}) or {}
        per_topic: dict[str, list[int]] = {}
        for h in state.get("history", []):
            per_topic.setdefault(h["concept"], []).append(int(bool(h["correct"])))
        for concept, answers in per_topic.items():
            out.append(Sequence(state.get("student_id", path.stem), concept, answers))
    return out


def random_true_params(concepts: list[str], rng: np.random.Generator) -> dict[str, BKTParams]:
    return {
        c: BKTParams(
            p_init=float(rng.uniform(0.1, 0.5)),
            p_learn=float(rng.uniform(0.05, 0.35)),
            p_slip=float(rng.uniform(0.05, 0.2)),
            p_guess=float(rng.uniform(0.1, 0.3)),
        )
        for c in concepts
    }


def simulate(true: dict[str, BKTParams], n_students: int, rng: np.random.Generator,
             min_q: int = 4, max_q: int = 12) -> list[Sequence]:
    """Make fake students who follow the BKT model exactly, with known 'true' numbers."""
    out = []
    for s in range(n_students):
        for concept, p in true.items():
            knows = rng.random() < p.p_init
            answers = []
            for _ in range(int(rng.integers(min_q, max_q + 1))):
                correct = (rng.random() >= p.p_slip) if knows else (rng.random() < p.p_guess)
                answers.append(int(correct))
                if not knows and rng.random() < p.p_learn:
                    knows = True
            out.append(Sequence(f"sim-{s}", concept, answers))
    return out


# ---------------- EM ----------------
def _pad(seqs: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    T = max(len(s) for s in seqs)
    obs = np.zeros((len(seqs), T), dtype=np.int8)
    mask = np.zeros((len(seqs), T), dtype=bool)
    for i, s in enumerate(seqs):
        obs[i, : len(s)] = s
        mask[i, : len(s)] = True
    return obs, mask


def _emissions(obs: np.ndarray, mask: np.ndarray, guess: float, slip: float) -> np.ndarray:
    """e[n, t, state] = chance of the observed answer in that state (1 where padded)."""
    e = np.empty(obs.shape + (2,))
    e[..., 0] = np.where(obs == 1, guess, 1 - guess)
    e[..., 1] = np.where(obs == 1, 1 - slip, slip)
    e[~mask] = 1.0
    return e


def _forward_backward(obs, mask, p: BKTParams):
    N, T = obs.shape
    A = np.array([[1 - p.p_learn, p.p_learn], [0.0, 1.0]])
    e = _emissions(obs, mask, p.p_guess, p.p_slip)

    alpha = np.zeros((N, T, 2))
    c = np.ones((N, T))
    a = np.stack([np.full(N, 1 - p.p_init), np.full(N, p.p_init)], axis=1) * e[:, 0]
    c[:, 0] = a.sum(1)
    alpha[:, 0] = a / c[:, 0:1]
    for t in range(1, T):
        valid = mask[:, t]
        a = (alpha[:, t - 1] @ A) * e[:, t]
        ct = a.sum(1)
        c[:, t] = np.where(valid, ct, 1.0)
        alpha[:, t] = np.where(valid[:, None], a / np.where(valid, ct, 1.0)[:, None], alpha[:, t - 1])

    beta = np.ones((N, T, 2))
    for t in range(T - 2, -1, -1):
        valid_next = mask[:, t + 1]
        b = ((e[:, t + 1] * beta[:, t + 1]) @ A.T) / c[:, t + 1:t + 2]
        beta[:, t] = np.where(valid_next[:, None], b, 1.0)

    gamma = alpha * beta
    gamma /= gamma.sum(2, keepdims=True)
    loglik = float(np.log(c[mask]).sum())
    return alpha, beta, gamma, c, e, loglik


def _m_step(obs, mask, p: BKTParams, alpha, beta, gamma, c, e) -> BKTParams:
    g0, g1 = gamma[..., 0], gamma[..., 1]
    o = obs.astype(float)
    m = mask.astype(float)

    p_init = float(g1[:, 0].mean())

    # expected "did not know -> knows" transitions between consecutive answers
    trans_mask = mask[:, 1:]
    xi01 = alpha[:, :-1, 0] * p.p_learn * e[:, 1:, 1] * beta[:, 1:, 1] / c[:, 1:]
    from_unknown = g0[:, :-1]
    p_learn = float((xi01 * trans_mask).sum() / max(1e-12, (from_unknown * trans_mask).sum()))

    p_guess = float((g0 * o * m).sum() / max(1e-12, (g0 * m).sum()))
    p_slip = float((g1 * (1 - o) * m).sum() / max(1e-12, (g1 * m).sum()))

    vals = {"p_init": p_init, "p_learn": p_learn, "p_slip": p_slip, "p_guess": p_guess}
    return BKTParams(**{k: float(np.clip(v, *BOUNDS[k])) for k, v in vals.items()})


def em_fit(seqs: list[list[int]], restarts: int = 5, max_iter: int = 200, tol: float = 1e-5,
           seed: int = 0) -> tuple[BKTParams, float, int]:
    """Returns (best parameters, log-likelihood, iterations used)."""
    seqs = [s for s in seqs if s]
    if not seqs:
        return DEFAULT, 0.0, 0
    obs, mask = _pad(seqs)
    rng = np.random.default_rng(seed)
    starts = [DEFAULT] + [
        BKTParams(p_init=float(rng.uniform(0.05, 0.6)), p_learn=float(rng.uniform(0.05, 0.4)),
                  p_slip=float(rng.uniform(0.02, 0.3)), p_guess=float(rng.uniform(0.05, 0.35)))
        for _ in range(max(0, restarts - 1))
    ]
    best = (DEFAULT, -np.inf, 0)
    for start in starts:
        p, prev, it = start, -np.inf, 0
        for it in range(1, max_iter + 1):
            alpha, beta, gamma, c, e, ll = _forward_backward(obs, mask, p)
            p = _m_step(obs, mask, p, alpha, beta, gamma, c, e)
            if ll - prev < tol:
                break
            prev = ll
        ll = _forward_backward(obs, mask, p)[-1]
        if ll > best[1]:
            best = (p, ll, it)
    return best


# ---------------- evaluation ----------------
def predictions(seq: list[int], p: BKTParams) -> list[float]:
    """Chance of a correct answer at each step, using only the answers before it."""
    known, out = p.p_init, []
    for a in seq:
        out.append(known * (1 - p.p_slip) + (1 - known) * p.p_guess)
        known = bkt_update(known, bool(a), p)
    return out


def auc(labels: list[int], scores: list[float]) -> float | None:
    """Area Under the ROC (Receiver Operating Characteristic) Curve: chance a random right answer
    was given a higher predicted score than a random wrong answer."""
    y = np.asarray(labels)
    s = np.asarray(scores)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate(seqs: list[Sequence], lookup) -> dict:
    labels, scores = [], []
    for s in seqs:
        labels += s.answers
        scores += predictions(s.answers, lookup(s.concept))
    if not labels:
        return {"answers": 0}
    y = np.asarray(labels)
    pr = np.clip(np.asarray(scores), 1e-6, 1 - 1e-6)
    return {
        "answers": len(labels),
        "log_likelihood_per_answer": float(np.mean(y * np.log(pr) + (1 - y) * np.log(1 - pr))),
        "accuracy": float(np.mean((pr >= 0.5) == (y == 1))),
        "auc": auc(labels, scores),
    }


# ---------------- full procedure ----------------
def fit_by_topic(seqs: list[Sequence], seed: int = 0) -> tuple[BKTParams, dict[str, BKTParams], dict]:
    global_p, global_ll, _ = em_fit([s.answers for s in seqs], seed=seed)
    per_topic: dict[str, BKTParams] = {}
    info: dict[str, dict] = {}
    for concept in sorted({s.concept for s in seqs}):
        topic_seqs = [s.answers for s in seqs if s.concept == concept]
        if len(topic_seqs) >= MIN_TOPIC_SEQUENCES:
            p, ll, iters = em_fit(topic_seqs, seed=seed)
            per_topic[concept] = p
            info[concept] = {"sequences": len(topic_seqs), "fitted": True, "iterations": iters}
        else:
            info[concept] = {"sequences": len(topic_seqs), "fitted": False}
    return global_p, per_topic, info


def fit_and_evaluate(seqs: list[Sequence], test_frac: float = 0.2, seed: int = 0) -> dict:
    """Split students into train and test, fit on train, compare default vs fitted on test.
    The parameters saved for use are then refit on all students."""
    t0 = time.time()
    students = sorted({s.student for s in seqs})
    rng = np.random.default_rng(seed)
    rng.shuffle(students)
    n_test = max(1, int(len(students) * test_frac)) if len(students) >= 5 else 0
    test_students = set(students[:n_test])
    train = [s for s in seqs if s.student not in test_students]
    test = [s for s in seqs if s.student in test_students]

    g_train, t_train, _ = fit_by_topic(train, seed=seed)
    metrics = None
    if test:
        metrics = {
            "default": evaluate(test, lambda c: DEFAULT),
            "fitted_global": evaluate(test, lambda c: g_train),
            "fitted_per_topic": evaluate(test, lambda c: t_train.get(c, g_train)),
        }

    g_all, t_all, info = fit_by_topic(seqs, seed=seed)
    return {
        "students": len(students),
        "sequences": len(seqs),
        "answers": sum(len(s.answers) for s in seqs),
        "train_students": len(students) - n_test,
        "test_students": n_test,
        "global": g_all.to_dict(),
        "per_topic": {c: p.to_dict() for c, p in t_all.items()},
        "topic_info": info,
        "test_metrics": metrics,
        "seconds": round(time.time() - t0, 2),
    }


def _last_fit_path():
    return settings.data_dir / "bkt_last_fit.json"


def run_fit(source: str = "app", n_students: int = 300, seed: int = 0, concepts: list[str] | None = None) -> dict:
    """Fit from app answers or from simulated students, and save the result (not yet applied)."""
    true = None
    if source == "simulated":
        rng = np.random.default_rng(seed)
        concepts = concepts or [f"topic {i + 1}" for i in range(5)]
        true = random_true_params(concepts, rng)
        seqs = simulate(true, n_students, rng)
    else:
        seqs = sequences_from_app()
    if not seqs:
        raise RuntimeError("No answers to learn from yet. Answer some practice questions, or use simulated students.")
    result = fit_and_evaluate(seqs, seed=seed)
    result["source"] = source
    result["fitted_at"] = int(time.time())
    if true:
        result["true_per_topic"] = {c: p.to_dict() for c, p in true.items()}
    write_json(_last_fit_path(), result)
    return result


def last_fit() -> dict | None:
    return read_json(_last_fit_path(), None)


def apply_last_fit() -> dict:
    fit = last_fit()
    if not fit:
        raise RuntimeError("Run a fit first.")
    if fit["source"] != "app":
        raise RuntimeError("Parameters fitted on simulated students are only for testing the method. "
                           "Fit on app answers to use them.")
    save_active_params({"source": "fitted", "global": fit["global"], "per_topic": fit["per_topic"],
                        "fitted_at": fit["fitted_at"], "answers": fit["answers"]})
    students = replay_all_students()
    return {"applied": True, "students_updated": students}


def _fmt_params(d: dict) -> str:
    return "  ".join(f"{k}={v:.3f}" for k, v in d.items())


def main() -> None:
    p = argparse.ArgumentParser(description="Fit BKT parameters with EM")
    p.add_argument("--source", choices=["app", "simulated"], default="app")
    p.add_argument("--students", type=int, default=300, help="number of simulated students")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--apply", action="store_true", help="use the fitted parameters in the app (source=app only)")
    args = p.parse_args()

    res = run_fit(args.source, args.students, args.seed)
    print(f"{res['students']} students, {res['sequences']} sequences, {res['answers']} answers, {res['seconds']}s\n")
    print("Global fit:   ", _fmt_params(res["global"]))
    for c, info in res["topic_info"].items():
        line = f"  {c:<28} {info['sequences']:>4} seq  "
        line += _fmt_params(res["per_topic"][c]) if info["fitted"] else "(too few sequences, uses global)"
        print(line)
        if res.get("true_per_topic"):
            print(f"  {'':<28} {'true':>8}  " + _fmt_params(res["true_per_topic"][c]))
    if res["test_metrics"]:
        print(f"\nHeld-out students ({res['test_students']}):")
        print(f"  {'parameters':<18}{'log-lik/ans':>12}{'accuracy':>10}{'AUC':>8}")
        for name, m in res["test_metrics"].items():
            a = "n/a" if m.get("auc") is None else f"{m['auc']:.3f}"
            print(f"  {name:<18}{m['log_likelihood_per_answer']:>12.4f}{m['accuracy']:>10.3f}{a:>8}")
    if args.apply:
        print("\n", json.dumps(apply_last_fit()))


if __name__ == "__main__":
    main()

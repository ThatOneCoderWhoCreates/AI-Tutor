import numpy as np
import pytest

from app.tutor import bkt, bkt_fit
from app.tutor.bkt import BKTParams


def test_em_recovers_known_parameters():
    true = BKTParams(p_init=0.3, p_learn=0.2, p_slip=0.1, p_guess=0.2)
    rng = np.random.default_rng(1)
    seqs = bkt_fit.simulate({"t": true}, 600, rng, min_q=6, max_q=12)
    fitted, ll, _ = bkt_fit.em_fit([s.answers for s in seqs], seed=1)
    for name in ("p_init", "p_learn", "p_slip", "p_guess"):
        assert abs(getattr(fitted, name) - getattr(true, name)) < 0.06, (name, fitted)


def test_em_log_likelihood_beats_defaults():
    true = BKTParams(p_init=0.5, p_learn=0.05, p_slip=0.2, p_guess=0.1)
    rng = np.random.default_rng(2)
    seqs = [s.answers for s in bkt_fit.simulate({"t": true}, 300, rng)]
    obs, mask = bkt_fit._pad(seqs)
    fitted, ll, _ = bkt_fit.em_fit(seqs)
    default_ll = bkt_fit._forward_backward(obs, mask, bkt.DEFAULT)[-1]
    assert ll > default_ll


def test_auc():
    assert bkt_fit.auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert bkt_fit.auc([1, 0], [0.5, 0.5]) == 0.5
    assert bkt_fit.auc([1, 1], [0.3, 0.4]) is None


def test_fit_on_app_answers_apply_and_replay():
    rng = np.random.default_rng(3)
    true = {"gradient descent": BKTParams(0.3, 0.25, 0.1, 0.15)}
    for s in bkt_fit.simulate(true, 40, rng):
        state = bkt.load_student(f"fit-{s.student}")
        for i, a in enumerate(s.answers):
            state["history"].append({"t": i, "concept": s.concept, "question_id": f"q{i}", "correct": bool(a),
                                     "mastery": 0.5})
        bkt.save_student(state)

    res = bkt_fit.run_fit("app")
    assert res["answers"] > 0 and res["test_metrics"] is not None
    assert "gradient descent" in res["per_topic"]

    out = bkt_fit.apply_last_fit()
    assert out["students_updated"] >= 40
    assert bkt.active_params()["source"] == "fitted"
    assert bkt.params_for("gradient descent") == BKTParams.from_dict(res["per_topic"]["gradient descent"])
    state = bkt.load_student("fit-sim-0")
    assert state["history"][-1]["mastery"] == state["mastery"]["gradient descent"]

    bkt.reset_params()
    assert bkt.active_params()["source"] == "default"
    for path in bkt.settings.students_dir.glob("fit-sim-*.json"):
        path.unlink()


def test_simulated_fit_cannot_be_applied():
    bkt_fit.run_fit("simulated", n_students=30)
    with pytest.raises(RuntimeError, match="simulated"):
        bkt_fit.apply_last_fit()

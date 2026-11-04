"""Scripted mock-student session for the final demo (PDF section 9).

A simulated student secretly knows some topics and not others. Every round they
press "What should I study next?", watch the suggested segment, and answer
3 practice questions. Each time they practice a topic they have a chance to
learn it. The printed table shows mastery rising for the weak topics.
Results are saved to eval/results/mock_student.csv and mock_student.png (chart for the report).

The simulated student's old answers are cleared at the start of each run.

Run from the backend folder (no server needed):
    python -m eval.mock_student
    python -m eval.mock_student --rounds 8 --student demo-student-2
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.store import slugify
from app.tutor.quiz import find_question


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--student", default="mock-student")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    rng = random.Random(args.seed)
    (settings.students_dir / f"{slugify(args.student)}.json").unlink(missing_ok=True)

    with TestClient(app) as client:
        concepts = [c["name"] for c in client.get("/concepts").json()]
        knows = {c: rng.random() < 0.4 for c in concepts}
        knows_at_start = dict(knows)
        print("Hidden truth (what the simulated student really knows):")
        for c in concepts:
            print(f"  {c:<32} {'knows' if knows[c] else 'does not know'}")
        print()

        header = f"{'round':<6}{'studied':<30}" + "".join(f"{c[:12]:>14}" for c in concepts)
        print(header)
        print("-" * len(header))
        start = {c["name"]: c["mastery"] for c in client.get("/concepts", params={"student_id": args.student}).json()}
        rows = [{"round": 0, "studied": "", **start}]
        for rnd in range(1, args.rounds + 1):
            nxt = client.post("/next", json={"student_id": args.student}).json()
            topic = nxt["concept"]
            if topic is None:
                print("All topics mastered.")
                break
            quiz = client.post("/quiz", json={"concept": topic, "n": 3, "student_id": args.student}).json()
            for q in quiz["questions"]:
                right = find_question(q["id"])["answer_index"]
                p_right = 0.9 if knows[topic] else 0.25
                choice = right if rng.random() < p_right else (right + 1 + rng.randrange(3)) % 4
                client.post("/answer", json={"student_id": args.student, "question_id": q["id"], "choice": choice})
                if not knows[topic] and rng.random() < 0.35:
                    knows[topic] = True
            mastery = {c["name"]: c["mastery"] for c in client.get("/concepts", params={"student_id": args.student}).json()}
            rows.append({"round": rnd, "studied": topic, **mastery})
            print(f"{rnd:<6}{topic[:28]:<30}" + "".join(f"{mastery[c] * 100:13.0f}%" for c in concepts))

    save_results(rows, concepts, knows_at_start)


def save_results(rows: list[dict], concepts: list[str], knew: dict[str, bool]) -> None:
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    with open(out / "mock_student.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "studied", *concepts])
        writer.writeheader()
        writer.writerows(rows)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\nSaved {out / 'mock_student.csv'} (install matplotlib for the chart)")
        return

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    rounds = [r["round"] for r in rows]
    for c in concepts:
        style = "-" if knew[c] else "--"
        ax.plot(rounds, [r[c] * 100 for r in rows], style, marker="o", markersize=3, linewidth=1.8,
                label=f"{c} ({'knew it' if knew[c] else 'did not know it'})")
    ax.axhline(95, color="gray", linewidth=0.8, linestyle=":")
    ax.text(rounds[0], 96.5, "mastered (95%)", ha="left", va="bottom", fontsize=8, color="gray")
    ax.set_xlabel("Study round (one 'What next?' suggestion + 3 practice questions)")
    ax.set_ylabel("Estimated mastery (%)")
    ax.set_ylim(0, 102)
    ax.set_xticks(rounds)
    ax.set_title("Simulated student: BKT mastery per topic over time")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "mock_student.png")
    print(f"\nSaved {out / 'mock_student.csv'} and {out / 'mock_student.png'}")


if __name__ == "__main__":
    main()

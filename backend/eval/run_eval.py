"""Evaluation for the final report. Writes eval/results/report.md.

1) Retrieval (no LLM needed): vector-only ("naive" baseline) vs keyword-only (BM25) vs hybrid.
   A hit means a retrieved chunk overlaps the time range where the answer is taught.
     Hit@1 = the top result is correct
     Hit@k = any of the top k results is correct
     MRR   = Mean Reciprocal Rank (1 if correct at rank 1, 0.5 at rank 2, ...)

2) Answers (--answers): the full /ask pipeline, once per model in --models.
     timestamp accuracy  = an answer cites a chunk inside the correct time range
     false refusal       = said "not covered" for a question the lecture does answer
     correct refusal     = said "not covered" for an out-of-scope question (expect: "not_covered")
     claim support       = share of answer sentences the trust layer marks as supported (--verify)

3) Quiz generation (--quiz): for every topic, generate questions and record how often the
   model produced valid JSON on the first try, and how long it took.

Run from the backend folder:
    python -m eval.run_eval eval/questions_demo.json
    python -m eval.run_eval eval/questions_lecture1.json --answers --quiz --lecture lecture1 --models qwen2.5:7b qwen3.5:9b
    python -m eval.run_eval eval/questions_demo.json --embedding-models BAAI/bge-small-en-v1.5 finetuned
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from app import llm
from app.config import settings
from app.rag.answer import NOT_COVERED, answer_question
from app.rag import finetune
from app.rag.embed import Embedder
from app.rag.index import Retriever
from app.schemas import AskResponse
from app.store import concept_map, load_chunks
from app.trust import verify_answer
from app.tutor.quiz import generate_questions

MODES = ["vector", "bm25", "hybrid"]
RESULTS = Path(__file__).parent / "results"


def in_scope(q: dict) -> bool:
    return q.get("expect") != "not_covered"


def overlaps(c, q) -> bool:
    return c.lecture_id == q["lecture_id"] and c.start < q["end"] and c.end > q["start"]


def eval_retrieval(retriever: Retriever, questions: list[dict], k: int) -> dict:
    questions = [q for q in questions if in_scope(q)]
    results = {}
    for mode in MODES:
        hit1 = hitk = rr = 0.0
        for q in questions:
            hits = retriever.search(q["question"], k=k, mode=mode)
            ranks = [i for i, c in enumerate(hits, start=1) if overlaps(c, q)]
            hit1 += 1 if ranks and ranks[0] == 1 else 0
            hitk += 1 if ranks else 0
            rr += 1 / ranks[0] if ranks else 0
        n = max(1, len(questions))
        results[mode] = {"hit@1": hit1 / n, f"hit@{k}": hitk / n, "mrr": rr / n}
    return results


def eval_answers(retriever: Retriever, questions: list[dict], model: str, verify: bool) -> dict:
    rows, times, support = [], [], []
    for i, q in enumerate(questions, start=1):
        print(f"  [{model}] answer {i}/{len(questions)}", end="\r", flush=True)
        t0 = time.time()
        try:
            resp, retrieved = answer_question(q["question"], retriever, use_fallback=False)
        except Exception as exc:  # one failed question must not stop the whole run
            print(f"\n  [{model}] question {i} failed: {str(exc)[:150]}")
            resp, retrieved = AskResponse(answer=f"ERROR: {exc}", citations=[], source="none"), []
        dt = time.time() - t0
        times.append(dt)
        refused = resp.answer.strip() == NOT_COVERED
        row = {
            "question": q["question"],
            "in_scope": in_scope(q),
            "seconds": round(dt, 2),
            "refused": refused,
            "timestamp_correct": in_scope(q) and any(
                c.lecture_id == q["lecture_id"] and c.start < q["end"] and c.end > q["start"] for c in resp.citations),
            "answer": resp.answer,
        }
        if verify and resp.citations:
            cited = {c.chunk_id for c in resp.citations}
            claims = verify_answer(resp.answer, [c for c in retrieved if c.id in cited])
            if claims:
                frac = sum(c.supported for c in claims) / len(claims)
                support.append(frac)
                row["claim_support"] = round(frac, 2)
        rows.append(row)
    print()

    tag = model.replace(":", "-").replace("/", "-")
    with open(RESULTS / f"answers_{tag}.csv", "w", newline="", encoding="utf-8") as f:
        fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    inside = [r for r in rows if r["in_scope"]]
    outside = [r for r in rows if not r["in_scope"]]
    return {
        "timestamp_accuracy": sum(r["timestamp_correct"] for r in inside) / max(1, len(inside)),
        "false_refusal": sum(r["refused"] for r in inside) / max(1, len(inside)),
        "correct_refusal": (sum(r["refused"] for r in outside) / len(outside)) if outside else None,
        "claim_support": statistics.mean(support) if support else None,
        "median_seconds": statistics.median(times),
    }


def eval_quiz(chunks, model: str) -> dict:
    cmap = concept_map(chunks)
    llm.json_stats.update(calls=0, first_try=0, retries=0, failures=0)
    ok, times, n_questions = 0, [], 0
    for i, (concept, cs) in enumerate(cmap.items(), start=1):
        print(f"  [{model}] quiz {i}/{len(cmap)}: {concept}", end="\r", flush=True)
        t0 = time.time()
        try:
            qs = generate_questions(concept, cs, 3, avoid=[])
            ok += 1
            n_questions += len(qs)
        except Exception as exc:  # count it as a failure and keep going
            print(f"\n  [{model}] quiz for '{concept}' failed: {str(exc)[:150]}")
        times.append(time.time() - t0)
    print()
    s = llm.json_stats
    return {
        "topics": len(cmap),
        "quiz_success": ok / max(1, len(cmap)),
        "first_try_valid": s["first_try"] / max(1, s["calls"]),
        "questions_made": n_questions,
        "median_quiz_seconds": statistics.median(times) if times else 0,
    }


def fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def write_report(n_q: int, n_chunks: int, embed: str, k: int, ret: dict, per_model: dict) -> Path:
    lines = [
        "# Evaluation Report",
        "",
        f"{n_q} test questions, {n_chunks} lecture chunks, embeddings: `{embed}`, top k = {k}.",
        "",
        "## Retrieval",
        "",
        "| Method | " + " | ".join(next(iter(ret.values())).keys()) + " |",
        "|---|" + "---|" * len(next(iter(ret.values()))),
    ]
    names = {"vector": "Vector only (baseline)", "bm25": "Keyword only (BM25)", "hybrid": "Hybrid (RRF)"}
    for mode, scores in ret.items():
        base, _, label = mode.partition(" (")
        name = names.get(base, base) + (f", {label.rstrip(')')}" if label else "")
        lines.append(f"| {name} | " + " | ".join(fmt(v) for v in scores.values()) + " |")

    if per_model:
        keys = []
        for res in per_model.values():
            for key in res:
                if key not in keys:
                    keys.append(key)
        lines += ["", "## Models", "", "| Metric | " + " | ".join(f"`{m}`" for m in per_model) + " |",
                  "|---|" + "---|" * len(per_model)]
        for key in keys:
            lines.append(f"| {key.replace('_', ' ')} | " + " | ".join(fmt(per_model[m].get(key)) for m in per_model) + " |")
        lines += ["", "Per-question answers are in `answers_<model>.csv` in this folder."]
    path = RESULTS / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("questions", type=Path)
    p.add_argument("--k", type=int, default=settings.top_k)
    p.add_argument("--answers", action="store_true", help="run the answer step for each model")
    p.add_argument("--verify", action="store_true", help="with --answers: also run the trust layer")
    p.add_argument("--quiz", action="store_true", help="test quiz generation for each model")
    p.add_argument("--models", nargs="+", default=None, help="Ollama models to compare (default: LLM_MODEL)")
    p.add_argument("--lecture", default=None, help="with --quiz: only test topics from this lecture id")
    p.add_argument("--embedding-models", nargs="+", default=None,
                   help="search models to compare, e.g. BAAI/bge-small-en-v1.5 finetuned")
    args = p.parse_args()

    RESULTS.mkdir(exist_ok=True)
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    chunks = load_chunks()
    retriever = Retriever()
    retriever.build(chunks)
    print(f"{len(questions)} questions, {len(chunks)} chunks, embeddings={retriever.embedder.name}\n")

    ret = eval_retrieval(retriever, questions, args.k)
    if args.embedding_models:
        ret = {}
        for name in args.embedding_models:
            model = str(finetune.output_dir(finetune.base_model_name())) if name == "finetuned" else name
            r = Retriever(Embedder(model_name=model), store="memory")
            r.build(chunks)
            label = "fine-tuned" if name == "finetuned" else Path(model).name
            for mode, scores in eval_retrieval(r, questions, args.k).items():
                if mode == "bm25":
                    ret.setdefault("bm25", scores)
                else:
                    ret[f"{mode} ({label})"] = scores
    width = max(len(m) for m in ret) + 2
    print(f"{'mode':<{width}}" + "  ".join(f"{m:>7}" for m in next(iter(ret.values()))))
    for mode, scores in ret.items():
        print(f"{mode:<{width}}" + "  ".join(f"{v:7.2f}" for v in scores.values()))

    per_model: dict[str, dict] = {}
    if args.answers or args.quiz:
        models = args.models or [settings.llm_model if settings.llm_provider != "mock" else "mock"]
        for model in models:
            llm.use_model(None if model == "mock" else model)
            print(f"\nModel: {model}")
            res: dict = {}
            if args.answers:
                res.update(eval_answers(retriever, questions, model, args.verify))
            if args.quiz:
                res.update(eval_quiz([c for c in chunks if not args.lecture or c.lecture_id == args.lecture], model))
            for key, val in res.items():
                print(f"  {key}: {fmt(val)}")
            per_model[model] = res
            # save after every model, so a later crash does not lose finished results
            write_report(len(questions), len(chunks), retriever.embedder.name, args.k, ret, per_model)
        llm.use_model(None)

    report = write_report(len(questions), len(chunks), retriever.embedder.name, args.k, ret, per_model)
    (RESULTS / "latest.json").write_text(json.dumps({"retrieval": ret, "models": per_model}, indent=2), encoding="utf-8")
    print(f"\nReport written to {report}")


if __name__ == "__main__":
    main()

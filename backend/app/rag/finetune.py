"""Fine-tune the search (embedding) model on your own lectures.

The base model (BAAI/bge-small-en-v1.5) was trained on general web text. Lecture
speech is different: informal, full of course-specific terms. Fine-tuning teaches
the model which questions belong with which parts of YOUR lectures.

Step 1, make training data: Qwen writes a few student-style questions for every
         chunk. Each (question, chunk) pair is one training example. No manual labels needed.
Step 2, train: contrastive learning with in-batch negatives (Multiple Negatives
         Ranking loss). In a batch of 16 pairs, each question should score its own
         chunk higher than the other 15 chunks. The loss pushes matching pairs
         together and non-matching pairs apart.
Step 3, check: some chunks are held out from training. Their questions are used to
         measure Hit@1 before and after training. The best epoch is kept.

Command line (run from the backend folder):
    python -m app.rag.finetune
    python -m app.rag.finetune --epochs 3 --use
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from ..config import settings
from ..llm import ask_llm_json
from ..schemas import Chunk
from ..store import load_chunks, read_json, write_json
from .embed import BGE_QUERY_PREFIX, active_embedding_model, set_active_embedding_model, uses_query_prefix
from .index import chunk_document

QUESTIONS_PER_CHUNK = 3

PROMPT = """TASK: queries
Write {n} different questions that a student might type into a search box, which this lecture excerpt answers.
Rules:
- Use your own words. Do not copy long phrases from the excerpt.
- Mix styles: one short keyword-style query, and full questions.
- Each question must be answerable from the excerpt alone.
Reply with JSON only: {{"questions": ["...", "..."]}}

EXCERPT:
{text}
"""


def _training_dir() -> Path:
    return settings.data_dir / "training"


def _models_dir() -> Path:
    return settings.data_dir / "models"


def _hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


# ---------------- step 1: training data ----------------
def _validate_questions(data):
    items = data.get("questions") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError('Expected {"questions": [...]}')
    out = []
    for q in items:
        q = re.sub(r"\s+", " ", str(q)).strip()
        if 2 <= len(q.split()) <= 40 and q.lower() not in {x.lower() for x in out}:
            out.append(q)
    if not out:
        raise ValueError("No usable questions")
    return out


def generate_queries(chunks: list[Chunk], n: int = QUESTIONS_PER_CHUNK, progress=None) -> dict[str, list[str]]:
    """chunk id -> questions. Cached in data/training/queries.json; only new or changed chunks call the LLM."""
    path = _training_dir() / "queries.json"
    cache = read_json(path, {}) or {}
    out: dict[str, list[str]] = {}
    for i, c in enumerate(chunks, start=1):
        doc = chunk_document(c)
        key = _hash(doc)
        if cache.get(c.id, {}).get("hash") == key:
            out[c.id] = cache[c.id]["queries"]
        else:
            try:
                qs = ask_llm_json(PROMPT.format(n=n, text=c.text[:1500]), validate=_validate_questions)[:n]
            except RuntimeError:
                qs = []
            cache[c.id] = {"hash": key, "queries": qs}
            out[c.id] = qs
            if i % 5 == 0:
                write_json(path, cache)
        if progress:
            progress(f"Writing practice search questions ({i}/{len(chunks)})", i / len(chunks))
    write_json(path, cache)
    return out


# ---------------- step 2: training ----------------
def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _batches(pairs: list[tuple[str, str, str]], batch_size: int, rng: random.Random) -> list[list]:
    """Shuffle, then fill batches so no chunk appears twice in a batch (it would be a false negative)."""
    pool = pairs[:]
    rng.shuffle(pool)
    batches: list[list] = []
    while pool:
        batch, used, rest = [], set(), []
        for p in pool:
            if len(batch) < batch_size and p[2] not in used:
                batch.append(p)
                used.add(p[2])
            else:
                rest.append(p)
        if len(batch) >= 2:
            batches.append(batch)
        pool = rest if len(batch) >= 2 else []
    return batches


def retrieval_scores(model, queries: list[tuple[str, str]], chunks: list[Chunk], prefix: str) -> dict:
    """Hit@1, Hit@3 and MRR (Mean Reciprocal Rank) of `queries` [(question, correct chunk id)] over all chunks."""
    import numpy as np

    if not queries:
        return {"hit@1": None, "hit@3": None, "mrr": None, "n": 0}
    ids = [c.id for c in chunks]
    d = model.encode([chunk_document(c) for c in chunks], normalize_embeddings=True, convert_to_numpy=True,
                     show_progress_bar=False)
    q = model.encode([prefix + x for x, _ in queries], normalize_embeddings=True, convert_to_numpy=True,
                     show_progress_bar=False)
    sims = q @ d.T
    hit1 = hit3 = rr = 0.0
    for row, (_, cid) in zip(sims, queries):
        order = [ids[j] for j in np.argsort(-row)]
        rank = order.index(cid) + 1
        hit1 += rank == 1
        hit3 += rank <= 3
        rr += 1 / rank
    n = len(queries)
    return {"hit@1": hit1 / n, "hit@3": hit3 / n, "mrr": rr / n, "n": n}


def finetune(chunks: list[Chunk], queries: dict[str, list[str]], base_model: str, out_dir: Path,
             epochs: int = 3, batch_size: int = 16, lr: float = 2e-5, val_frac: float = 0.15,
             seed: int = 0, progress=None) -> dict:
    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    rng = random.Random(seed)
    torch.manual_seed(seed)
    prefix = BGE_QUERY_PREFIX if uses_query_prefix(base_model) else ""
    usable = [c for c in chunks if queries.get(c.id)]
    if len(usable) < 4:
        raise RuntimeError("Need at least 4 chunks with questions to fine-tune. Add a real lecture first.")

    shuffled = usable[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    val_ids = {c.id for c in shuffled[:n_val]}
    train_pairs = [(prefix + q, chunk_document(c), c.id) for c in usable if c.id not in val_ids for q in queries[c.id]]
    val_queries = [(q, c.id) for c in usable if c.id in val_ids for q in queries[c.id]]

    device = _device()
    model = SentenceTransformer(base_model, device=device)
    model.max_seq_length = min(model.max_seq_length or 256, 256)

    if progress:
        progress("Measuring the base model", None)
    base_scores = retrieval_scores(model, val_queries, chunks, prefix)
    best = dict(base_scores)
    best_epoch = 0
    history = [{"epoch": 0, "loss": None, **base_scores}]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps_per_epoch = len(_batches(train_pairs, batch_size, random.Random(seed)))
    total = max(1, steps_per_epoch * epochs)
    warmup = max(1, int(0.1 * total))
    def lr_factor(s: int) -> float:
        if s < warmup:
            return (s + 1) / warmup
        return max(0.0, (total - s) / max(1, total - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scale = 20.0

    def to_device(features):
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in features.items()}

    step = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in _batches(train_pairs, batch_size, rng):
            qs = model(to_device(model.tokenize([b[0] for b in batch])))["sentence_embedding"]
            ds = model(to_device(model.tokenize([b[1] for b in batch])))["sentence_embedding"]
            qs, ds = F.normalize(qs, dim=-1), F.normalize(ds, dim=-1)
            scores = qs @ ds.T * scale
            labels = torch.arange(len(batch), device=scores.device)
            loss = (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)) / 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(float(loss.detach().cpu()))
            step += 1
            if progress:
                progress(f"Training epoch {epoch}/{epochs}", step / total)

        scores_now = retrieval_scores(model, val_queries, chunks, prefix)
        mean_loss = sum(losses) / max(1, len(losses))
        history.append({"epoch": epoch, "loss": mean_loss, **scores_now})
        better = (scores_now["mrr"] or 0) > (best["mrr"] or 0) or best_epoch == 0
        if better:
            best, best_epoch = dict(scores_now), epoch
            model.save(str(out_dir))

    result = {
        "base_model": base_model,
        "model_path": str(out_dir),
        "device": device,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "train_pairs": len(train_pairs),
        "train_chunks": len(usable) - n_val,
        "val_chunks": n_val,
        "val_questions": len(val_queries),
        "before": base_scores,
        "after": best,
        "history": history,
        "seconds": round(time.time() - t0, 1),
        "finished_at": int(time.time()),
    }
    write_json(out_dir / "training_log.json", result)
    write_json(_training_dir() / "last_run.json", result)
    return result


def base_model_name() -> str:
    """Always fine-tune from the original model in .env, not from an earlier fine-tuned copy."""
    return settings.embedding_model


def output_dir(base: str) -> Path:
    return _models_dir() / f"{Path(base).name}-lecture-ft"


def run(epochs: int = 3, progress=None) -> dict:
    chunks = load_chunks()
    if not chunks:
        raise RuntimeError("No lectures yet.")
    queries = generate_queries(chunks, progress=progress)
    base = base_model_name()
    return finetune(chunks, queries, base, output_dir(base), epochs=epochs, progress=progress)


def last_run() -> dict | None:
    return read_json(_training_dir() / "last_run.json", None)


def fmt(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune the search model on your lectures")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--use", action="store_true", help="switch the app to the fine-tuned model afterwards")
    args = p.parse_args()

    last = {"msg": ""}

    def progress(msg, frac):
        if msg != last["msg"]:
            print(("\n" if last["msg"] else "") + msg, end="", flush=True)
            last["msg"] = msg
        else:
            print(".", end="", flush=True)

    res = run(args.epochs, progress)
    print("\n")
    print(f"Trained on {res['train_pairs']} question-chunk pairs from {res['train_chunks']} chunks "
          f"({res['device']}, {res['seconds']}s). Checked on {res['val_questions']} questions "
          f"from {res['val_chunks']} held-out chunks.\n")
    print(f"{'':<10}{'hit@1':>8}{'hit@3':>8}{'mrr':>8}")
    for name in ("before", "after"):
        s = res[name]
        print(f"{name:<10}{fmt(s['hit@1']):>8}{fmt(s['hit@3']):>8}{fmt(s['mrr']):>8}")
    print(f"\nSaved to {res['model_path']} (best epoch: {res['best_epoch']})")
    if args.use:
        set_active_embedding_model(res["model_path"])
        print("The app now uses the fine-tuned model. Restart the backend or call POST /reindex.")
    else:
        print("To use it: python -m app.rag.finetune --use, or the Lab page in the app.")
    print(json.dumps({"active": active_embedding_model()}))


if __name__ == "__main__":
    main()

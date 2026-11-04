"""Lab endpoints, plus a real fine-tuning run on a tiny randomly initialized model (skipped without PyTorch)."""
import importlib.util
import time

import pytest
from fastapi.testclient import TestClient

from app.main import app

def _torch_works() -> bool:
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("sentence_transformers") is None:
        return False
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


HAS_TORCH = _torch_works()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_lectures_have_stats(client):
    demo = next(l for l in client.get("/lectures").json() if l["id"] == "demo")
    assert demo["chunks"] == 10 and demo["topics"] >= 5


def test_bkt_lab_flow(client):
    info = client.get("/lab/bkt").json()
    assert info["active"]["source"] == "default"
    assert info["default"]["p_guess"] == 0.25

    res = client.post("/lab/bkt/fit", json={"source": "simulated", "students": 120}).json()
    assert res["source"] == "simulated"
    assert set(res["true_per_topic"]) == set(res["per_topic"])
    assert res["test_metrics"]["fitted_per_topic"]["answers"] > 0

    r = client.post("/lab/bkt/apply")
    assert r.status_code == 503 and "simulated" in r.json()["detail"]

    assert client.post("/lab/bkt/reset").json()["reset"] is True


def test_embeddings_lab_status(client):
    info = client.get("/lab/embeddings").json()
    assert info["active_model"] == info["base_model"]
    assert info["using_finetuned"] is False
    assert info["can_train"] is False  # tests run with the offline hash backend
    assert client.post("/lab/embeddings/train", json={"epochs": 1}).status_code == 400


def test_mock_generates_training_questions():
    from app.rag.finetune import generate_queries
    from app.store import load_chunks

    qs = generate_queries(load_chunks("demo")[:3])
    assert len(qs) == 3 and all(len(v) == 3 for v in qs.values())


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch and sentence-transformers not installed")
def test_finetune_tiny_model_end_to_end(tmp_path):
    from sentence_transformers import SentenceTransformer, models
    from transformers import BertConfig, BertModel, BertTokenizerFast

    from app.rag import finetune
    from app.store import load_chunks

    chunks = load_chunks("demo")
    words = sorted({w for c in chunks for w in finetune.chunk_document(c).lower().split()})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words
    base = tmp_path / "bge-tiny"
    base.mkdir()
    (base / "vocab.txt").write_text("\n".join(vocab))
    BertTokenizerFast(vocab_file=str(base / "vocab.txt")).save_pretrained(base)
    BertModel(BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=64, max_position_embeddings=256)).save_pretrained(base)
    word = models.Transformer(str(base), max_seq_length=128)
    SentenceTransformer(modules=[word, models.Pooling(word.get_word_embedding_dimension())]).save(str(base / "st"))

    queries = finetune.generate_queries(chunks)
    out = tmp_path / "tuned"
    res = finetune.finetune(chunks, queries, str(base / "st"), out, epochs=2, batch_size=8, lr=1e-3,
                            val_frac=0.3)
    assert res["train_pairs"] > 0 and res["val_questions"] > 0
    assert res["best_epoch"] >= 1 and len(res["history"]) == 3
    assert (out / "training_log.json").exists()
    SentenceTransformer(str(out)).encode(["gradient descent"])

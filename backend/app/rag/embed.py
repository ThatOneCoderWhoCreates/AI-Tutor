"""Turn text into vectors (lists of numbers that capture meaning).

EMBEDDING_BACKEND=sentence-transformers  -> real model, e.g. BAAI/bge-small-en-v1.5 (default)
EMBEDDING_BACKEND=hash                   -> tiny offline fallback, word overlap only (tests / no internet)
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from ..config import settings
from ..store import read_json, write_json

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _active_path():
    return settings.data_dir / "active_models.json"


def active_embedding_model() -> str:
    """The model in use: a fine-tuned one if it was switched on in the Lab, else EMBEDDING_MODEL from .env."""
    chosen = (read_json(_active_path(), {}) or {}).get("embedding_model")
    if chosen and Path(chosen).exists():
        return chosen
    return settings.embedding_model


def set_active_embedding_model(path: str | None) -> None:
    """None switches back to the base model from .env."""
    data = read_json(_active_path(), {}) or {}
    data["embedding_model"] = path
    write_json(_active_path(), data)


def uses_query_prefix(model_name: str) -> bool:
    return "bge" in Path(model_name).name.lower() or "bge" in model_name.lower()


class Embedder:
    def __init__(self, backend: str | None = None, model_name: str | None = None):
        self.backend = backend or settings.embedding_backend
        self.model_name = model_name or active_embedding_model()
        self._model = None
        if self.backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        elif self.backend != "hash":
            raise ValueError(f"Unknown EMBEDDING_BACKEND: {self.backend}")

    @property
    def name(self) -> str:
        if self.backend == "hash":
            return "hash:hash1024"
        label = Path(self.model_name).name if Path(self.model_name).exists() else self.model_name
        return f"{self.backend}:{label}"

    @property
    def cache_key(self) -> str:
        """Changes when a fine-tuned model is retrained, so old stored vectors are not reused."""
        key = self.name
        path = Path(self.model_name)
        if self.backend != "hash" and path.exists():
            stamp = max((p.stat().st_mtime_ns for p in path.rglob("*") if p.is_file()), default=0)
            key += f"@{stamp}"
        return key

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        if self.backend == "hash":
            return np.vstack([_hash_vector(t) for t in texts])
        if is_query and uses_query_prefix(self.model_name):
            texts = [BGE_QUERY_PREFIX + t for t in texts]
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


def _hash_vector(text: str, dim: int = 1024) -> np.ndarray:
    words = re.findall(r"[a-z0-9]+", text.lower())
    feats = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
    vec = np.zeros(dim, dtype=np.float32)
    for f in feats:
        h = int(hashlib.md5(f.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0 if (h >> 20) % 2 == 0 else -1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec

"""Search over lecture chunks.

Three modes, so the evaluation can compare them:
    "vector"  -> meaning-based search only (the "naive" baseline)
    "bm25"    -> keyword search only
    "hybrid"  -> both, combined with Reciprocal Rank Fusion (RRF)

VECTOR_STORE=chroma stores vectors on disk with Chroma (default).
VECTOR_STORE=memory keeps them in a NumPy array (no extra install).
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
from rank_bm25 import BM25Okapi

from ..config import settings
from ..schemas import Chunk
from ..store import slugify
from .embed import Embedder

RRF_K = 60

STOPWORDS = set(
    "a an the is are was were be been of to in on for and or with what how why when which does do this that it its "
    "as by from at into about can we you i me my our your so if then than there their they them he she his her".split()
)


def stem(word: str) -> str:
    """Light stemming so singular and plural forms match: lists -> list, queues -> queue,
    queries -> query, hashes -> hash, indexes -> index. Short words and words like
    'class', 'status', 'analysis' are left alone."""
    w = word
    if len(w) <= 3 or w.endswith(("ss", "us", "is")):
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "sses", "xes", "zes")):
        return w[:-2]
    if w.endswith("s"):
        return w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def chunk_document(c: Chunk) -> str:
    """Text that gets indexed: slide text gives extra keywords."""
    return f"{c.slide_text}\n{c.text}".strip()


class Retriever:
    def __init__(self, embedder: Embedder | None = None, store: str | None = None):
        self.embedder = embedder or Embedder()
        self.store = store or settings.vector_store
        self.chunks: list[Chunk] = []
        self.by_id: dict[str, Chunk] = {}
        self._bm25: BM25Okapi | None = None
        self._matrix: np.ndarray | None = None
        self._collection = None

    # ---------- building ----------
    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        self.by_id = {c.id: c for c in self.chunks}
        docs = [chunk_document(c) for c in self.chunks]
        self._bm25 = BM25Okapi([tokenize(d) or ["_empty_"] for d in docs]) if docs else None
        if self.store == "chroma":
            self._build_chroma(docs)
        else:
            self._matrix = self.embedder.encode(docs) if docs else None

    def _build_chroma(self, docs: list[str]) -> None:
        import chromadb

        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        key = self.embedder.cache_key
        name = "chunks_" + slugify(self.embedder.name)[:40] + "_" + hashlib.md5(key.encode()).hexdigest()[:8]
        col = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
        existing = col.get(include=["documents"])
        have = dict(zip(existing["ids"], existing["documents"]))
        wanted = {c.id: d for c, d in zip(self.chunks, docs)}

        stale = [i for i in have if i not in wanted or have[i] != wanted[i]]
        if stale:
            col.delete(ids=stale)
        todo = [c for c in self.chunks if c.id not in have or c.id in stale]
        if todo:
            todo_docs = [wanted[c.id] for c in todo]
            vecs = self.embedder.encode(todo_docs)
            col.upsert(
                ids=[c.id for c in todo],
                embeddings=vecs.tolist(),
                documents=todo_docs,
                metadatas=[{"lecture_id": c.lecture_id, "start": c.start} for c in todo],
            )
        self._collection = col

    # ---------- searching ----------
    def _vector_ranking(self, query: str, depth: int, lecture_id: str | None) -> list[str]:
        if not self.chunks:
            return []
        q = self.embedder.encode([query], is_query=True)[0]
        if self.store == "chroma":
            where = {"lecture_id": lecture_id} if lecture_id else None
            n = min(depth, len(self.chunks))
            res = self._collection.query(query_embeddings=[q.tolist()], n_results=n, where=where)
            return [i for i in res["ids"][0] if i in self.by_id]
        scores = self._matrix @ q
        order = np.argsort(-scores)
        ids = [self.chunks[i].id for i in order]
        if lecture_id:
            ids = [i for i in ids if self.by_id[i].lecture_id == lecture_id]
        return ids[:depth]

    def _bm25_ranking(self, query: str, depth: int, lecture_id: str | None) -> list[str]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(tokenize(query) or ["_empty_"])
        order = np.argsort(-scores)
        ids = [self.chunks[i].id for i in order if scores[i] > 0]
        if lecture_id:
            ids = [i for i in ids if self.by_id[i].lecture_id == lecture_id]
        return ids[:depth]

    def search(self, query: str, k: int | None = None, mode: str = "hybrid", lecture_id: str | None = None) -> list[Chunk]:
        k = k or settings.top_k
        depth = max(20, k * 5)
        if mode == "vector":
            return [self.by_id[i] for i in self._vector_ranking(query, k, lecture_id)]
        if mode == "bm25":
            return [self.by_id[i] for i in self._bm25_ranking(query, k, lecture_id)]

        fused: dict[str, float] = {}
        for ranking in (self._vector_ranking(query, depth, lecture_id), self._bm25_ranking(query, depth, lecture_id)):
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        best = sorted(fused, key=lambda i: -fused[i])[:k]
        return [self.by_id[i] for i in best]

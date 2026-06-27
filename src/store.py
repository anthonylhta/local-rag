"""Stage 3 - an in-memory vector store with hand-rolled cosine retrieval.

No FAISS, no hosted vector DB: just a numpy matrix of embeddings alongside the
chunk metadata. Similarity is plain cosine, computed by hand with numpy so the
mechanics stay visible. The store persists to disk as a ``.npy`` matrix plus
two small JSON sidecars (the chunks and some metadata).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.chunk import Chunk


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a ``(d,)`` query and an ``(n, d)`` matrix -> ``(n,)``.

    cosine(a, b) = (a . b) / (||a|| * ||b||). We L2-normalise both sides and take
    a dot product, which for unit vectors *is* the cosine similarity.
    """
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    q_norm = np.linalg.norm(query)
    if q_norm == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    q = query / q_norm
    m_norms = np.linalg.norm(matrix, axis=1)
    m_norms[m_norms == 0] = 1e-12  # guard against zero-vectors
    m = matrix / m_norms[:, None]
    return (m @ q).astype(np.float32)


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    rank: int  # 0-based position in the ranked results


class VectorStore:
    def __init__(self, embedding_model: str = "", chunk_size: int = 0, overlap: int = 0):
        self.vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.chunks: list[Chunk] = []
        # metadata, persisted so `query`/`chat` can rebuild the matching embedder
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.overlap = overlap

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(chunks) != vectors.shape[0]:
            raise ValueError("number of chunks and vectors must match")
        if len(self.chunks) == 0:
            self.vectors = vectors
        else:
            if vectors.shape[1] != self.vectors.shape[1]:
                raise ValueError("embedding dimension mismatch")
            self.vectors = np.vstack([self.vectors, vectors])
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int = 4) -> list[SearchResult]:
        if len(self.chunks) == 0:
            return []
        scores = cosine_similarity(query_vector, self.vectors)
        k = min(k, len(scores))
        # argpartition gives the top-k cheaply, then we sort just those k by score
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [
            SearchResult(chunk=self.chunks[i], score=float(scores[i]), rank=r)
            for r, i in enumerate(top_idx)
        ]

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def save(self, index_dir: str | Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "vectors.npy", self.vectors)
        (index_dir / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        meta = {
            "count": len(self.chunks),
            "dim": int(self.vectors.shape[1]) if self.vectors.size else 0,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
        }
        (index_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: str | Path) -> VectorStore:
        index_dir = Path(index_dir)
        meta_path = index_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"no index found in {index_dir!s} (run `index` first)")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        store = cls(
            embedding_model=meta.get("embedding_model", ""),
            chunk_size=meta.get("chunk_size", 0),
            overlap=meta.get("overlap", 0),
        )
        store.vectors = np.load(index_dir / "vectors.npy")
        store.chunks = [
            Chunk.from_dict(d)
            for d in json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        ]
        return store

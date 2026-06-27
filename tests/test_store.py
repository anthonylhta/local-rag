"""Tests for stage 3: the vector store, hand-rolled cosine, and persistence."""

from __future__ import annotations

import numpy as np
import pytest

from src.chunk import Chunk
from src.store import VectorStore, cosine_similarity


def _store():
    chunks = [Chunk(f"chunk {i}", "a.md", i) for i in range(4)]
    vectors = np.array([[1, 0, 0], [0, 1, 0], [2, 0, 0], [-1, 0, 0]], dtype=np.float32)
    store = VectorStore(embedding_model="test", chunk_size=200, overlap=40)
    store.add(chunks, vectors)
    return store


def test_cosine_similarity_values():
    q = np.array([1, 0, 0], dtype=np.float32)
    m = np.array([[1, 0, 0], [0, 1, 0], [2, 0, 0], [-1, 0, 0]], dtype=np.float32)
    np.testing.assert_allclose(cosine_similarity(q, m), [1.0, 0.0, 1.0, -1.0], atol=1e-6)


def test_cosine_similarity_handles_empty_and_zero():
    assert cosine_similarity(np.array([1.0, 0.0]), np.zeros((0, 2), dtype=np.float32)).shape == (0,)
    out = cosine_similarity(np.zeros(2, dtype=np.float32), np.ones((3, 2), dtype=np.float32))
    np.testing.assert_array_equal(out, np.zeros(3))


def test_search_ranks_by_similarity():
    store = _store()
    results = store.search(np.array([1, 0, 0], dtype=np.float32), k=2)

    assert [r.rank for r in results] == [0, 1]
    assert results[0].score == pytest.approx(1.0)
    # the two parallel vectors (chunk 0 and chunk 2) are the closest
    assert {r.chunk.text for r in results} == {"chunk 0", "chunk 2"}


def test_search_empty_store():
    assert VectorStore().search(np.array([1.0, 0.0]), k=3) == []


def test_search_k_larger_than_store():
    store = _store()
    assert len(store.search(np.array([1, 0, 0], dtype=np.float32), k=99)) == 4


def test_add_dimension_mismatch():
    store = _store()
    with pytest.raises(ValueError):
        store.add([Chunk("x", "a.md", 0)], np.ones((1, 5), dtype=np.float32))


def test_add_count_mismatch():
    with pytest.raises(ValueError):
        VectorStore().add([Chunk("x", "a.md", 0)], np.ones((2, 3), dtype=np.float32))


def test_save_load_roundtrip(tmp_path):
    store = _store()
    store.save(tmp_path)

    assert {p.name for p in tmp_path.iterdir()} == {"vectors.npy", "chunks.json", "meta.json"}

    reloaded = VectorStore.load(tmp_path)
    assert len(reloaded) == len(store)
    assert reloaded.embedding_model == "test"
    q = np.array([1, 0, 0], dtype=np.float32)
    assert [r.chunk.text for r in reloaded.search(q, 2)] == [
        r.chunk.text for r in store.search(q, 2)
    ]


def test_load_missing_index(tmp_path):
    with pytest.raises(FileNotFoundError):
        VectorStore.load(tmp_path / "nope")

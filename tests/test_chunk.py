"""Tests for stage 1: loading and chunking."""

from __future__ import annotations

import pytest

from src.chunk import Chunk, chunk_documents, chunk_text, load_documents


def test_chunk_text_overlap_and_coverage():
    words = " ".join(f"w{i}" for i in range(450))
    chunks = chunk_text(words, chunk_size=200, overlap=40)

    assert [len(c.split()) for c in chunks] == [200, 200, 130]
    # the overlap is exact: tail of one chunk == head of the next
    c0, c1 = chunks[0].split(), chunks[1].split()
    assert c0[-40:] == c1[:40]
    # every word is covered, including the last
    assert "w449" in chunks[-1]


def test_chunk_text_shorter_than_window():
    assert chunk_text("only a few words", chunk_size=200, overlap=40) == ["only a few words"]


def test_chunk_text_empty():
    assert chunk_text("   ", chunk_size=200, overlap=40) == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-1, 0), (100, 100), (100, 150), (100, -1)],
)
def test_chunk_text_invalid_params(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_size=chunk_size, overlap=overlap)


def test_chunk_documents_preserves_source_and_index():
    docs = [("a.md", " ".join(f"w{i}" for i in range(300))), ("b.md", "short doc")]
    chunks = chunk_documents(docs, chunk_size=200, overlap=40)

    assert {c.source for c in chunks} == {"a.md", "b.md"}
    a_chunks = [c for c in chunks if c.source == "a.md"]
    assert [c.index for c in a_chunks] == list(range(len(a_chunks)))


def test_chunk_roundtrip_dict():
    c = Chunk(text="hi", source="a.md", index=3)
    assert Chunk.from_dict(c.to_dict()) == c


def test_load_documents(tmp_path):
    (tmp_path / "a.txt").write_text("alpha text", encoding="utf-8")
    (tmp_path / "b.md").write_text("# beta", encoding="utf-8")
    (tmp_path / "ignore.pdf").write_text("nope", encoding="utf-8")
    (tmp_path / "empty.md").write_text("   ", encoding="utf-8")

    docs = load_documents(tmp_path)
    sources = sorted(s for s, _ in docs)

    assert sources == [str(tmp_path / "a.txt"), str(tmp_path / "b.md")]


def test_load_documents_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "does-not-exist")

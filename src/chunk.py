"""Stage 1 - load and chunk documents.

Reads ``.txt`` / ``.md`` files from a folder and splits them into overlapping,
word-based chunks. The overlap keeps context from being lost at chunk
boundaries: a sentence that straddles a boundary still appears whole in one of
the two neighbouring chunks, so it stays retrievable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".txt", ".md"}


@dataclass
class Chunk:
    """A single retrievable piece of text plus where it came from."""

    text: str
    source: str  # path to the file this chunk came from
    index: int  # 0-based position of this chunk within its source file

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        return cls(text=d["text"], source=d["source"], index=d["index"])


def load_documents(folder: str | Path) -> list[tuple[str, str]]:
    """Return ``[(path, text), ...]`` for every supported file under ``folder``."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"corpus folder not found: {folder}")
    docs: list[tuple[str, str]] = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                docs.append((str(path), text))
    return docs


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """Split ``text`` into ~``chunk_size``-word chunks sharing ``overlap`` words.

    Word-based chunking is a simple, tokenizer-free proxy for token chunking and
    is plenty for learning the mechanics.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if window:
            chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break  # this window reached the end; nothing more to add
    return chunks


def chunk_documents(
    docs: list[tuple[str, str]], chunk_size: int = 200, overlap: int = 40
) -> list[Chunk]:
    """Flatten ``[(path, text)]`` into a list of :class:`Chunk` objects."""
    out: list[Chunk] = []
    for source, text in docs:
        for i, piece in enumerate(chunk_text(text, chunk_size, overlap)):
            out.append(Chunk(text=piece, source=source, index=i))
    return out

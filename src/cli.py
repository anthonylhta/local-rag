"""Stage 6 - the command-line interface that ties the pipeline together.

Usage::

    python -m src.cli index  [--docs docs] [--index index] [--chunk-size 200] [--overlap 40]
    python -m src.cli query  "your question" [--k 4] [--model llama3.2]
    python -m src.cli chat   [--k 4] [--model llama3.2]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running both as `python -m src.cli` (from repo root) and `python src/cli.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunk import chunk_documents, load_documents  # noqa: E402
from src.embed import DEFAULT_MODEL, Embedder  # noqa: E402
from src.rag import (  # noqa: E402
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    build_prompt,
    generate_stream,
)
from src.store import VectorStore  # noqa: E402

DEFAULT_DOCS = "docs"
DEFAULT_INDEX = "index"


def cmd_index(args) -> int:
    print(f"Loading documents from {args.docs!r} ...")
    docs = load_documents(args.docs)
    if not docs:
        print("No .txt/.md files found. Add some to the corpus folder and retry.")
        return 1
    print(f"  {len(docs)} document(s).")

    print(f"Chunking (size={args.chunk_size} words, overlap={args.overlap}) ...")
    chunks = chunk_documents(docs, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"  {len(chunks)} chunk(s).")

    print(f"Embedding with {args.embed_model!r} (first run downloads the model) ...")
    embedder = Embedder(args.embed_model)
    vectors = embedder.embed([c.text for c in chunks], show_progress=True)
    print(f"  vectors: {vectors.shape}")

    store = VectorStore(
        embedding_model=args.embed_model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    store.add(chunks, vectors)
    store.save(args.index)
    print(f"Saved index to {args.index!r} ({len(store)} chunks).")
    return 0


def _load_for_query(args):
    store = VectorStore.load(args.index)
    embedder = Embedder(store.embedding_model or DEFAULT_MODEL)
    return store, embedder


def _answer_streaming(question, store, embedder, k, model, host) -> None:
    q_vec = embedder.embed_one(question)
    results = store.search(q_vec, k=k)

    print("\nAnswer:")
    for token in generate_stream(build_prompt(question, results), model=model, host=host):
        print(token, end="", flush=True)

    print("\n\nSources:")
    for r in results:
        snippet = r.chunk.text[:160].replace("\n", " ")
        ellipsis = "..." if len(r.chunk.text) > 160 else ""
        print(f"  [{r.rank + 1}] {r.chunk.source}  (score {r.score:.3f})")
        print(f"      {snippet}{ellipsis}")


def cmd_query(args) -> int:
    store, embedder = _load_for_query(args)
    _answer_streaming(args.question, store, embedder, args.k, args.model, args.host)
    return 0


def cmd_chat(args) -> int:
    store, embedder = _load_for_query(args)
    print("Interactive RAG chat. Ask a question, or type 'exit' to quit.\n")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", ":q"}:
            break
        try:
            _answer_streaming(question, store, embedder, args.k, args.model, args.host)
        except RuntimeError as e:
            print(f"\n[error] {e}")
        print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="local-rag", description="A from-scratch, fully-offline RAG pipeline."
    )
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("index", help="ingest, chunk, embed and persist the corpus")
    pi.add_argument("--docs", default=DEFAULT_DOCS, help="corpus folder")
    pi.add_argument("--index", default=DEFAULT_INDEX, help="where to write the index")
    pi.add_argument("--chunk-size", type=int, default=200, help="words per chunk")
    pi.add_argument("--overlap", type=int, default=40, help="overlapping words between chunks")
    pi.add_argument("--embed-model", default=DEFAULT_MODEL, help="sentence-transformers model")
    pi.set_defaults(func=cmd_index)

    pq = sub.add_parser("query", help="ask a single question")
    pq.add_argument("question")
    pq.add_argument("--index", default=DEFAULT_INDEX)
    pq.add_argument("--k", type=int, default=4, help="number of chunks to retrieve")
    pq.add_argument("--model", default=DEFAULT_OLLAMA_MODEL, help="Ollama model")
    pq.add_argument("--host", default=DEFAULT_OLLAMA_HOST)
    pq.set_defaults(func=cmd_query)

    pc = sub.add_parser("chat", help="interactive question loop")
    pc.add_argument("--index", default=DEFAULT_INDEX)
    pc.add_argument("--k", type=int, default=4, help="number of chunks to retrieve")
    pc.add_argument("--model", default=DEFAULT_OLLAMA_MODEL, help="Ollama model")
    pc.add_argument("--host", default=DEFAULT_OLLAMA_HOST)
    pc.set_defaults(func=cmd_chat)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

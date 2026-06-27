# Milestones

## Milestone 1 — from-scratch RAG pipeline, fully offline ✅

- [x] ingest + chunk — load `.txt`/`.md`, split into overlapping word chunks
- [x] embed — encode chunks with sentence-transformers (`all-MiniLM-L6-v2`, 384-dim)
- [x] retrieve — hand-rolled cosine similarity, top-k over an in-memory store
- [x] generate — build a grounded prompt, answer via Ollama (`llama3.2`)
- [x] show sources — print the retrieved chunks (path + score) used for the answer
- [x] persist index — save/load the vector store to disk so indexing runs once

> **Verification:** ingest/chunk, embed, retrieve, show-sources and persist were
> run for real over `docs/` (3 docs → 6 chunks → `(6, 384)` vectors; a query for
> cosine similarity correctly ranks the two relevant chunks top-2). The Ollama
> generation path (request, streaming token parse, error handling) was verified
> against a stub of `/api/generate`; run it against a live `llama3.2` by
> installing Ollama and `ollama pull llama3.2`.

## Next up (after milestone 1)

- [ ] swap the in-memory store for FAISS (approximate nearest neighbour at scale)
- [ ] add a cross-encoder reranker over the top-k candidates
- [ ] support more file types (PDF, HTML) in the loader
- [ ] token-based chunking with a real tokenizer
- [ ] evaluation harness (retrieval hit-rate, answer faithfulness)

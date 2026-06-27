# Milestones

## Milestone 1 — from-scratch RAG pipeline, fully offline

- [ ] ingest + chunk — load `.txt`/`.md`, split into overlapping word chunks
- [ ] embed — encode chunks with sentence-transformers
- [ ] retrieve — hand-rolled cosine similarity, top-k over an in-memory store
- [ ] generate — build a grounded prompt, answer via Ollama (`llama3.2`)
- [ ] show sources — print the retrieved chunks (path + score) used for the answer
- [ ] persist index — save/load the vector store to disk so indexing runs once

## Next up (after milestone 1)

- [ ] swap the in-memory store for FAISS (approximate nearest neighbour at scale)
- [ ] add a cross-encoder reranker over the top-k candidates
- [ ] support more file types (PDF, HTML) in the loader
- [ ] token-based chunking with a real tokenizer
- [ ] evaluation harness (retrieval hit-rate, answer faithfulness)

# local-rag

A from-scratch, **fully-offline** Retrieval-Augmented Generation (RAG) pipeline.

Point it at a folder of your `.txt`/`.md` files, ask a question in the terminal,
and get an answer **grounded in retrieved chunks with sources shown** — no hosted
vector DB, no cloud LLM. Embeddings run locally via
[sentence-transformers](https://www.sbert.net/), similarity is hand-rolled with
numpy, and generation runs locally through [Ollama](https://ollama.com/). The
point is to learn the mechanics, so the moving parts are deliberately explicit.

```
question ─▶ embed ─▶ cosine top-k ─▶ build prompt ─▶ Ollama ─▶ answer + sources
                         ▲
            index: load ─▶ chunk ─▶ embed ─▶ store (save to disk)
```

## Requirements

- **Python 3.11+** (developed/tested on 3.10 as well; nothing 3.11-only is used)
- **[Ollama](https://ollama.com/)** installed and running, with a model pulled:
  ```bash
  ollama pull llama3.2
  ```
- Python deps from `requirements.txt` (`numpy`, `sentence-transformers`)

> **About "offline":** the very first run downloads two things over the network —
> the Python packages and the embedding model weights (`all-MiniLM-L6-v2`, ~90 MB)
> from Hugging Face. Both are then cached locally and every subsequent run is
> fully offline.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**1. Build the index** (load → chunk → embed → persist). Runs once; re-run when
the corpus changes:

```bash
python -m src.cli index --docs docs --index index
```

**2. Ask a one-off question:**

```bash
python -m src.cli query "How is cosine similarity computed here?" --k 4
```

**3. Or chat interactively:**

```bash
python -m src.cli chat
```

Example output:

```
Answer:
Cosine similarity is computed by hand with numpy: every vector is L2-normalised
and a single matrix-vector dot product scores the query against all chunks [2].

Sources:
  [1] docs/embeddings-and-similarity.md  (score 0.612)
      An embedding is a fixed-length vector of numbers that represents the meaning...
  [2] docs/embeddings-and-similarity.md  (score 0.581)
      To compare two vectors we use cosine similarity. Cosine similarity measures...
```

## How it works — stage by stage

Each stage is one small, readable module under `src/`.

### 1. Load & chunk — `src/chunk.py`
Reads every `.txt`/`.md` file under the corpus folder and splits each into
**word-based chunks with overlap** (default 200 words, 40 overlap). Overlap means
a sentence sitting on a chunk boundary still appears whole in a neighbour, so it
stays retrievable. Each chunk remembers its source file and position.

### 2. Embed — `src/embed.py`
Wraps a `sentence-transformers` model (`all-MiniLM-L6-v2`, 384-dim) that turns
each chunk into a vector capturing its meaning. The model loads lazily and is
cached after the first download.

### 3. Store & retrieve — `src/store.py`
An in-memory vector store: a numpy matrix of embeddings plus the chunk metadata.
Retrieval is **hand-rolled cosine similarity** — normalise, one dot product
against the whole matrix, take the top-k. The store **saves/loads to disk**
(`vectors.npy` + `chunks.json` + `meta.json`) so you embed once and query many
times.

### 4. Generate — `src/rag.py`
Builds a grounded prompt (instruction + numbered context passages + question) and
sends it to **Ollama** over its REST API using only the Python standard library.
A low temperature keeps the model anchored to the retrieved context. Supports
streaming so the terminal prints tokens as they arrive.

### 5. Show sources — `src/rag.py` / `src/cli.py`
Every answer is printed alongside the chunks that produced it: source path,
cosine score, and a snippet — so you can audit *why* the model said what it said.

### 6. CLI — `src/cli.py`
`index`, `query`, and `chat` subcommands wire the stages together.

## Project layout

```
local-rag/
├── docs/                 # the corpus (sample .md files about RAG)
├── index/                # persisted vector store (artifacts git-ignored)
├── src/
│   ├── chunk.py          # stage 1: load + chunk
│   ├── embed.py          # stage 2: embeddings
│   ├── store.py          # stage 3: in-memory store + cosine + persistence
│   ├── rag.py            # stages 4-5: prompt + Ollama generation
│   └── cli.py            # stage 6: command-line interface
├── requirements.txt
├── MILESTONES.md
├── LICENSE               # MIT
└── README.md
```

## What to swap in next

The pipeline is built to be replaced piece by piece:

- **FAISS** in place of the numpy store, once the corpus grows past the point
  where a brute-force scan is fast enough — gives approximate nearest-neighbour
  search that scales to millions of vectors.
- **A reranker** (a cross-encoder such as `cross-encoder/ms-marco-MiniLM-L-6-v2`)
  over the top-k candidates. Bi-encoder retrieval is fast but coarse; a
  cross-encoder rescoring the shortlist is usually the single biggest quality win.
- **More loaders** (PDF, HTML) and **token-based chunking** with a real tokenizer.
- **An eval harness** to measure retrieval hit-rate and answer faithfulness so
  changes can be compared objectively.

See `MILESTONES.md` for the running checklist.

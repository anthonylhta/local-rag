# CLAUDE.md

## What this is

`local-rag` is a from-scratch, fully-offline Retrieval-Augmented Generation
pipeline for the terminal. It points at a folder of `.txt`/`.md` files, chunks
them with overlap, embeds the chunks locally with sentence-transformers, stores
the vectors in an in-memory numpy index (with hand-rolled cosine similarity and
disk persistence), retrieves the top-k chunks for a question, and generates a
grounded answer with a local Ollama model — printing the answer alongside the
source chunks it used. No hosted vector DB and no cloud LLM; the goal is to learn
the mechanics, so each stage is a small, explicit module.

## Stack & commands

- **Python 3.11+**, **numpy** (vector math), **sentence-transformers**
  (`all-MiniLM-L6-v2` embeddings), **Ollama** (`llama3.2`, reached over its REST
  API with the standard library only).
- **Tooling:** ruff (lint + format), mypy (types), pytest (tests), `build`
  (packaging).

```bash
# one-time dev setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # installs the package + dev tools

# run (dev)
python -m src.cli index          # build/refresh the index from docs/
python -m src.cli query "..."    # ask one question
python -m src.cli chat           # interactive loop

# quality gates (mirror CI)
mypy src                         # typecheck
ruff check .                     # lint
ruff format --check .            # format check (use `ruff format .` to fix)
pytest                           # tests
python -m build                  # build the wheel/sdist
```

Ollama must be installed and running (`ollama pull llama3.2`) for the `query`/
`chat` generation step; everything else runs without it.

## Repository rules

**Commits**
- Voice: first person, plain capitalized sentences. **One logical change per
  commit.**
- Never narrate the process or who found what (no "after debugging…", no
  "as requested…").
- **No AI/Claude attribution anywhere** in commits or PR bodies — no
  `Co-Authored-By`, no "Generated with…" lines.

**Branches & PRs**
- `main` is **branch-protected**. Never push to `main`.
- Every change goes on a `<type>/<slug>` branch where `<type>` is one of
  `feat/`, `fix/`, `refactor/`, `chore/` → open a PR → wait for **green CI** →
  merge.
- **Never merge a PR — the maintainer merges on GitHub.** Stop at: *"PR open,
  green CI, tested, ready for review."* Merge permission is granted per-PR and is
  never assumed to carry across features.
- Do **not** reference local `notes/` files or ADR numbers in commit messages or
  PR bodies. `notes/` is gitignored, so such references resolve to nothing in the
  public history.

## Guardrails

- Only touch files inside this project's tree.
- Confirm before anything hard to reverse or outward-facing: deploys,
  force-pushes, deleting or overwriting files you did not create, or sending data
  to external services.
- **Read the real docs first.** Before writing code against a fast-moving
  dependency, read its *installed* docs/changelog — the current API may differ
  from training data. Heed deprecation notices.

"""Stages 4-5 - build a grounded prompt and generate an answer with Ollama.

Retrieval (stage 3) lives in the store. This module turns the retrieved chunks
into a prompt, calls a local Ollama model over its REST API (stdlib ``urllib``
only - no extra dependencies), and packages the answer together with the
sources that were used.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from src.embed import Embedder
from src.store import SearchResult, VectorStore

DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

SYSTEM_INSTRUCTION = (
    "You are a precise assistant. Answer the question using ONLY the context "
    "passages provided. Cite the passages you rely on with their [n] markers. "
    "If the answer is not contained in the context, say you don't know - do not "
    "guess or use outside knowledge."
)


@dataclass
class Answer:
    question: str
    text: str
    sources: list[SearchResult]


def build_prompt(question: str, results: list[SearchResult]) -> str:
    """Assemble the final prompt: instruction + numbered context + question."""
    blocks = [f"[{r.rank + 1}] (source: {r.chunk.source})\n{r.chunk.text}" for r in results]
    context = "\n\n".join(blocks) if blocks else "(no context retrieved)"
    return f"{SYSTEM_INSTRUCTION}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"


def _ollama_request(payload: dict, host: str, timeout: int):
    url = f"{host.rstrip('/')}/api/generate"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def _ollama_error_message(host: str, err: Exception) -> str:
    return (
        f"Could not reach Ollama at {host} ({err}).\n"
        "Make sure Ollama is installed and running (`ollama serve`) and that the "
        "model is pulled (e.g. `ollama pull llama3.2`)."
    )


def generate(
    prompt: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
    temperature: float = 0.2,
    timeout: int = 180,
) -> str:
    """Single-shot (non-streaming) generation."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        with _ollama_request(payload, host, timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body.get("response", "").strip()
    except urllib.error.URLError as e:
        raise RuntimeError(_ollama_error_message(host, e)) from e


def generate_stream(
    prompt: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
    temperature: float = 0.2,
    timeout: int = 180,
):
    """Yield response tokens as they arrive for a responsive terminal UX."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": temperature},
    }
    try:
        resp = _ollama_request(payload, host, timeout)
    except urllib.error.URLError as e:
        raise RuntimeError(_ollama_error_message(host, e)) from e
    with resp:
        for line in resp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line.decode("utf-8"))
            token = obj.get("response", "")
            if token:
                yield token
            if obj.get("done"):
                break


def answer(
    question: str,
    store: VectorStore,
    embedder: Embedder,
    k: int = 4,
    model: str = DEFAULT_OLLAMA_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
) -> Answer:
    """Full query->answer loop: embed -> retrieve -> prompt -> generate."""
    q_vec = embedder.embed_one(question)
    results = store.search(q_vec, k=k)
    prompt = build_prompt(question, results)
    text = generate(prompt, model=model, host=host)
    return Answer(question=question, text=text, sources=results)

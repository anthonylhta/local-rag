"""Tests for stages 4-5: prompt building and the Ollama client.

The Ollama client is exercised against a tiny stub HTTP server that mimics the
``/api/generate`` endpoint, so these tests need neither a running Ollama nor the
embedding model.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.chunk import Chunk
from src.rag import build_prompt, generate, generate_stream
from src.store import SearchResult


def _results():
    return [
        SearchResult(Chunk("Cosine is a dot product of unit vectors.", "docs/e.md", 2), 0.58, 0),
        SearchResult(
            Chunk("Overlap keeps boundary sentences retrievable.", "docs/c.md", 0), 0.41, 1
        ),
    ]


def test_build_prompt_numbers_and_attributes_sources():
    prompt = build_prompt("How does cosine work?", _results())
    assert "[1] (source: docs/e.md)" in prompt
    assert "[2] (source: docs/c.md)" in prompt
    assert "Question: How does cosine work?" in prompt
    assert prompt.rstrip().endswith("Answer:")


def test_build_prompt_no_results():
    assert "(no context retrieved)" in build_prompt("q", [])


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence the test server
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        if body.get("stream"):
            for tok in ["Cosine ", "similarity ", "[1]."]:
                self.wfile.write((json.dumps({"response": tok, "done": False}) + "\n").encode())
            self.wfile.write((json.dumps({"response": "", "done": True}) + "\n").encode())
        else:
            self.wfile.write(
                json.dumps({"response": "Cosine similarity [1].", "done": True}).encode()
            )


@pytest.fixture()
def stub_ollama() -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()


def test_generate_non_streaming(stub_ollama):
    assert generate("prompt", model="stub", host=stub_ollama) == "Cosine similarity [1]."


def test_generate_stream_yields_tokens(stub_ollama):
    tokens = list(generate_stream("prompt", model="stub", host=stub_ollama))
    assert tokens == ["Cosine ", "similarity ", "[1]."]
    assert "".join(tokens) == "Cosine similarity [1]."


def test_generate_unreachable_host_gives_helpful_error():
    with pytest.raises(RuntimeError, match="Could not reach Ollama"):
        generate("prompt", model="stub", host="http://127.0.0.1:1")

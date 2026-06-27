# Local Inference with Ollama

Ollama is a tool for running large language models locally. It downloads
quantised model weights, manages them, and exposes a simple REST API on
`http://localhost:11434`. Because everything runs on your own machine, no data
leaves your computer and there are no per-token costs.

This project generates answers with the `llama3.2` model through Ollama. You pull
the model once with `ollama pull llama3.2`, and Ollama serves it from then on.
The RAG pipeline talks to Ollama over its HTTP API using only the Python
standard library, so the project needs no extra networking dependencies.

The key endpoint is `POST /api/generate`. You send a JSON body containing the
model name and the prompt. If `stream` is false you get the whole response back
at once; if `stream` is true Ollama returns a sequence of JSON lines, each
carrying the next token, which lets the terminal print the answer as it is
produced. A `done` flag marks the final line.

Generation settings live under an `options` object. The most useful is
`temperature`: lower values (around 0.2) make the model more deterministic and
factual, which suits RAG, while higher values make it more creative. Keeping
temperature low helps the model stick to the retrieved context rather than
drifting into its own prior knowledge.

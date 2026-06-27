# What is Retrieval-Augmented Generation?

Retrieval-Augmented Generation (RAG) is a technique that grounds a language
model's answers in an external collection of documents instead of relying only
on what the model memorised during training. At query time the system retrieves
the passages most relevant to the question and passes them to the model as
context. The model then answers using that context.

RAG matters for three reasons. First, it reduces hallucination: when the model
is handed the relevant facts, it is far less likely to invent them. Second, it
keeps answers current and private without retraining — you simply change the
documents in the corpus. Third, it makes answers auditable, because the system
can show exactly which passages it used.

A typical RAG pipeline has two phases. The indexing phase runs once: load the
documents, split them into chunks, embed each chunk into a vector, and store the
vectors. The query phase runs per question: embed the question, retrieve the
top-k most similar chunks by vector similarity, build a prompt that includes
those chunks, and generate an answer with the language model.

The quality of a RAG system depends heavily on retrieval. If the right chunk is
never retrieved, no amount of clever prompting will recover the answer. That is
why chunking strategy, embedding quality, and the similarity metric all matter,
and why reranking the retrieved candidates is often the highest-leverage
improvement after a basic pipeline works.

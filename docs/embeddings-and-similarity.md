# Embeddings and Cosine Similarity

An embedding is a fixed-length vector of numbers that represents the meaning of
a piece of text. Texts with similar meaning land close together in the vector
space, even when they share no words. This is what lets a RAG system match a
question to a relevant passage that is phrased completely differently.

This project uses the `all-MiniLM-L6-v2` model from the sentence-transformers
library. It produces 384-dimensional vectors, is small enough to run on a CPU,
and is a strong general-purpose default. The model weights are downloaded once
from Hugging Face and then cached locally, so subsequent runs are fully offline.

To compare two vectors we use cosine similarity. Cosine similarity measures the
angle between vectors rather than their magnitude: it equals the dot product of
the two vectors divided by the product of their lengths. The result ranges from
-1 (opposite) through 0 (unrelated) to 1 (identical direction). Because it
ignores magnitude, cosine similarity is robust to differences in text length.

In this project cosine similarity is hand-rolled with numpy instead of a vector
database. We L2-normalise every vector — dividing it by its own length — and
then a single matrix-vector dot product gives the similarity of the query
against every stored chunk at once. Sorting those scores and taking the top few
gives the retrieved context. Doing it by hand makes the mechanics obvious before
swapping in an optimised library like FAISS.

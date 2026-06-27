"""Stage 2 - turn text into embedding vectors.

Thin wrapper around a ``sentence-transformers`` model. The model is loaded
lazily on first use. The *first ever* run downloads the model weights from
Hugging Face; after that the weights are cached locally and embedding runs
fully offline.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "all-MiniLM-L6-v2"  # 384-dim, small + fast, a solid default


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None  # lazy

    @property
    def model(self):
        if self._model is None:
            # imported lazily so the rest of the package works without torch
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts, batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """Embed a string or iterable of strings into a float32 ``(n, dim)`` array."""
        if isinstance(texts, str):
            texts = [texts]
        vecs = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=False,  # the store normalises during cosine
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

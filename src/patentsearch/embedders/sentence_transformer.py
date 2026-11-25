"""
Embedding model wrapper.

The demo uses BAAI/bge-small-en-v1.5 (set in config/demo.yaml): a
small, free, CPU-friendly embedding model
that performs well on retrieval benchmarks for its size -- chosen
specifically because this Space is expected to run on a free CPU-only
tier, where a large embedding model would make every search feel slow.

BGE models are trained asymmetrically: the *query* side benefits from a
short instruction prefix at inference time, while passages (the chunks
being searched over) are embedded as-is, with no prefix. Skipping the
query instruction doesn't break anything, but does measurably hurt
retrieval quality, so it's applied consistently here in one place
rather than left to every caller to remember.
"""

from __future__ import annotations

import numpy as np


class SentenceTransformerEmbedder:
    """Any sentence-transformers model, with an optional query prefix."""

    def __init__(self, model_name: str, query_instruction: str = ""):
        self.model_name = model_name
        self.query_instruction = query_instruction
        self._model = None  # lazily loaded sentence_transformers.SentenceTransformer

    @property
    def model(self):
        """Load the model once and reuse it for the process lifetime.

        On HuggingFace Spaces the weights are downloaded once and cached
        between restarts.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Embed a list of passages (chunks) for indexing or comparison."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single search query, with the query instruction prefix applied."""
        embedding = self.model.encode(
            self.query_instruction + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding.astype("float32")

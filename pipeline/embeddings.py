"""
Embedding model wrapper.

Uses BAAI/bge-small-en-v1.5: a small, free, CPU-friendly embedding model
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

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_model = None  # lazily loaded; type is sentence_transformers.SentenceTransformer


def get_model():
    """Lazily load the embedding model once and reuse it for the process lifetime.

    On HuggingFace Spaces this load happens a single time at startup
    (when app.py first calls into this module), not per-request -- the
    model weights (~130MB) are downloaded once and cached by Spaces'
    own storage between restarts.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_passages(texts: list[str]) -> np.ndarray:
    """Embed a list of passages (chunks) for indexing or comparison."""
    model = get_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embeddings.astype("float32")


def embed_query(query: str) -> np.ndarray:
    """Embed a single search query, with the asymmetric instruction prefix applied."""
    model = get_model()
    embedding = model.encode(
        _QUERY_INSTRUCTION + query,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embedding.astype("float32")

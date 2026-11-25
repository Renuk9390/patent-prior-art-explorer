"""The interface every embedding model wrapper provides."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Embed passages for indexing; returns float32, L2-normalised rows."""
        ...

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one search query (with any model-specific prefix)."""
        ...

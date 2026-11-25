"""The interface every vector store provides."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from patentsearch.core.types import Chunk, SearchResult


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None: ...

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[SearchResult]: ...

    def __len__(self) -> int: ...

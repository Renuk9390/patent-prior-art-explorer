"""
A minimal FAISS-backed vector store.

Kept deliberately small: the curated corpus here is a handful of
patents, and at most one uploaded document is added per visitor
session, so there's no need for FAISS's more advanced (approximate)
index types. IndexFlatIP -- exact inner-product search over normalized
vectors, which is equivalent to cosine similarity -- is exact, simple,
and fast enough at this scale. If the corpus ever grows into the
thousands of documents, swapping IndexFlatIP for an approximate index
(e.g. IndexIVFFlat or HNSW) is a change confined entirely to this file.
"""

from __future__ import annotations

import numpy as np

from patentsearch.core.types import Chunk, SearchResult


class FaissStore:
    """Wraps a FAISS IndexFlatIP index alongside the chunk metadata it indexes."""

    def __init__(self, dim: int):
        import faiss

        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings must have the same length")
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        if self.index.ntotal == 0:
            return []
        top_k = min(top_k, self.index.ntotal)
        query = query_embedding.reshape(1, -1)
        scores, indices = self.index.search(query, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(SearchResult(chunk=self.chunks[idx], score=float(score)))
        return results

    def __len__(self) -> int:
        return len(self.chunks)

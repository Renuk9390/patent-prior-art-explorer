"""Shared data types used across ingestion, search and display."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    """A single retrievable unit of text, plus the metadata needed to display it."""

    text: str
    doc_id: str
    doc_title: str
    domain: str
    url: str | None = None


@dataclass
class SearchResult:
    chunk: Chunk
    score: float

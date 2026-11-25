"""The interface every LLM provider provides."""

from __future__ import annotations

from typing import Protocol


class LLMProvider(Protocol):
    @property
    def is_configured(self) -> bool:
        """True if the provider has what it needs (for example an API key)."""
        ...

    def analyse(self, query: str, passage: str) -> str | None:
        """Return the structured prior-art analysis text, or None on any failure."""
        ...

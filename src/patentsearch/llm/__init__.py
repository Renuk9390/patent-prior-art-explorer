from __future__ import annotations

from patentsearch.config import LLMConfig
from patentsearch.llm.base import LLMProvider


def make_llm_provider(cfg: LLMConfig) -> LLMProvider | None:
    """Create the LLM provider named in the settings, or None for "none"."""
    if cfg.provider == "none":
        return None
    if cfg.provider == "groq":
        from patentsearch.llm.groq import GroqProvider

        return GroqProvider(cfg.models, cfg.timeout_seconds)
    raise ValueError(f"Unknown LLM provider: {cfg.provider!r}")

"""
Settings for a deployment, loaded from a YAML file.

The file is chosen in this order: the path passed to `load_config()`,
the PATENTSEARCH_CONFIG environment variable, then config/demo.yaml in
the repository. Any setting left out of the file keeps the default
below, so a missing file still gives a working demo configuration.
"""

from __future__ import annotations

import os
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "demo.yaml"


@dataclass
class EmbedderConfig:
    model_name: str = "BAAI/bge-small-en-v1.5"
    query_instruction: str = "Represent this sentence for searching relevant passages: "


@dataclass
class SearchConfig:
    top_k: int = 5
    highlight_sentences: int = 2


@dataclass
class CorpusConfig:
    source: str = "demo"


@dataclass
class LLMConfig:
    provider: str = "groq"
    models: list[str] = field(
        default_factory=lambda: ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
    )
    timeout_seconds: float = 20


@dataclass
class AppConfig:
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


def _fill(cls, values: dict):
    """Build dataclass `cls` from a dict, recursing into nested sections."""
    kwargs = {}
    for f in fields(cls):
        if f.name not in values:
            continue
        value = values[f.name]
        nested = f.default_factory() if f.default_factory is not MISSING else None
        if is_dataclass(nested) and isinstance(value, dict):
            value = _fill(type(nested), value)
        kwargs[f.name] = value
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown setting(s) in {cls.__name__}: {sorted(unknown)}")
    return cls(**kwargs)


def load_config(path: str | os.PathLike | None = None) -> AppConfig:
    """Load settings from YAML; fall back to defaults if no file exists."""
    path = Path(path or os.environ.get("PATENTSEARCH_CONFIG") or _DEFAULT_PATH)
    if not path.exists():
        print(f"[config] {path} not found -- using built-in defaults.", flush=True)
        return AppConfig()

    import yaml

    with open(path, encoding="utf-8") as fh:
        values = yaml.safe_load(fh) or {}
    return _fill(AppConfig, values)

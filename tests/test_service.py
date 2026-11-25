"""
Fast tests for the search service and settings. They use a tiny fake
embedder (word hashing), so they need no model download and run in
seconds:   python -m pytest tests
"""

import hashlib
import re

import numpy as np
import pytest

from patentsearch.config import AppConfig, load_config
from patentsearch.llm import make_llm_provider
from patentsearch.llm.groq import GroqProvider
from patentsearch.service import SearchService, _build_corpus_store


class FakeEmbedder:
    dim = 256

    def _vec(self, text):
        v = np.zeros(self.dim, dtype="float32")
        for w in re.findall(r"[a-z]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        return v / (np.linalg.norm(v) + 1e-9)

    def embed_passages(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, query):
        return self._vec(query)


class FakeLLM:
    is_configured = True

    def analyse(self, query, passage):
        return f"analysis of {query}"


@pytest.fixture
def service():
    config = AppConfig()
    embedder = FakeEmbedder()
    return SearchService(config, embedder, _build_corpus_store(config, embedder), FakeLLM())


def test_demo_config_file_loads():
    cfg = load_config()
    assert cfg.embedder.model_name == "BAAI/bge-small-en-v1.5"
    assert cfg.search.top_k == 5
    assert cfg.llm.provider == "groq"
    assert cfg.llm.models[0] == "openai/gpt-oss-20b"


def test_missing_config_file_uses_defaults(tmp_path):
    assert load_config(tmp_path / "nope.yaml") == AppConfig()


def test_unknown_setting_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("search:\n  topk: 3\n")
    with pytest.raises(ValueError):
        load_config(bad)


def test_search_returns_top_k_sorted(service):
    results = service.search("battery thermal runaway overheating")
    assert len(results) == service.config.search.top_k
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert "battery" in results[0].chunk.text.lower() or "thermal" in results[0].chunk.text.lower()


def test_uploaded_store_is_searched_too(service):
    upload = service.index_texts(
        ["A zebra-striped umbrella folds into a pocket-sized zebra case."],
        doc_id="uploaded", doc_title="mine.pdf", domain="Your uploaded document",
    )
    results = service.search("zebra umbrella", extra_stores=[upload, None])
    assert results[0].chunk.doc_id == "uploaded"


def test_explain_highlights_configured_number_of_sentences(service):
    passage = "Batteries can overheat. Cooling stops thermal runaway. The sky is blue."
    out = service.explain("battery thermal runaway", passage)
    assert len(out["highlighted"]) == service.config.search.highlight_sentences


def test_llm_analysis_and_none_provider(service):
    assert service.analyse("q", "p") == "analysis of q"
    service.llm = None
    assert service.analyse("q", "p") is None
    assert service.llm_configured is False


def test_groq_model_override(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "my/model")
    provider = make_llm_provider(AppConfig().llm)
    assert isinstance(provider, GroqProvider)
    assert provider.models[0] == "my/model"
    assert make_llm_provider(type(AppConfig().llm)(provider="none")) is None

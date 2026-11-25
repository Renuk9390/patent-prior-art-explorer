"""
SearchService: the single entry point the user interface calls.

The UI never talks to the embedding model, the vector store or the LLM
directly. Swapping any of them (a different model, Qdrant instead of
FAISS, a local LLM instead of Groq) means changing the settings file or
this service's wiring, never the UI.
"""

from __future__ import annotations

from collections.abc import Iterable

from patentsearch.config import AppConfig, load_config
from patentsearch.core.types import Chunk, SearchResult
from patentsearch.embedders.base import Embedder
from patentsearch.explain.highlight import explain_match
from patentsearch.llm import make_llm_provider
from patentsearch.llm.base import LLMProvider
from patentsearch.stores.base import VectorStore
from patentsearch.stores.faiss_store import FaissStore


class SearchService:
    def __init__(
        self,
        config: AppConfig,
        embedder: Embedder,
        corpus_store: VectorStore,
        llm: LLMProvider | None = None,
    ):
        self.config = config
        self.embedder = embedder
        self.corpus_store = corpus_store
        self.llm = llm

    # ── Construction ────────────────────────────────────────────────
    @classmethod
    def from_config(cls, config: AppConfig | None = None) -> "SearchService":
        """Build every component named in the settings and index the corpus."""
        config = config or load_config()

        from patentsearch.embedders.sentence_transformer import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(
            config.embedder.model_name, config.embedder.query_instruction
        )
        corpus_store = _build_corpus_store(config, embedder)
        return cls(config, embedder, corpus_store, make_llm_provider(config.llm))

    # ── Search ──────────────────────────────────────────────────────
    def search(
        self, query: str, extra_stores: Iterable[VectorStore | None] = ()
    ) -> list[SearchResult]:
        """Top results for `query` across the corpus and any extra stores
        (such as a visitor's uploaded PDF), best first."""
        top_k = self.config.search.top_k
        query_vec = self.embedder.embed_query(query)
        results = list(self.corpus_store.search(query_vec, top_k=top_k))
        for store in extra_stores:
            if store is not None and len(store) > 0:
                results += store.search(query_vec, top_k=top_k)
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def index_texts(
        self, texts: list[str], doc_id: str, doc_title: str, domain: str,
        url: str | None = None,
    ) -> VectorStore:
        """Index extra passages (for example an uploaded PDF) in their own store."""
        chunks = [
            Chunk(text=t, doc_id=doc_id, doc_title=doc_title, domain=domain, url=url)
            for t in texts
        ]
        embeddings = self.embedder.embed_passages(texts)
        store = FaissStore(dim=embeddings.shape[1])
        store.add(chunks, embeddings)
        return store

    # ── Explanation ─────────────────────────────────────────────────
    def explain(self, query: str, passage: str) -> dict:
        """Local sentence-level evidence: which sentences match the query best."""
        return explain_match(
            query, passage, self.embedder, top_n=self.config.search.highlight_sentences
        )

    @property
    def llm_configured(self) -> bool:
        return self.llm is not None and self.llm.is_configured

    def analyse(self, query: str, passage: str) -> str | None:
        """Optional LLM prior-art analysis; None if unavailable."""
        if self.llm is None:
            return None
        return self.llm.analyse(query, passage)


def _build_corpus_store(config: AppConfig, embedder: Embedder) -> VectorStore:
    """Embed and index the corpus named in the settings."""
    if config.corpus.source != "demo":
        raise ValueError(f"Unknown corpus source: {config.corpus.source!r}")

    from patentsearch.data import CORPUS

    chunks: list[Chunk] = []
    texts: list[str] = []
    for patent in CORPUS:
        for paragraph in patent["paragraphs"]:
            chunks.append(
                Chunk(
                    text=paragraph,
                    doc_id=patent["id"],
                    doc_title=patent["title"],
                    domain=patent["domain"],
                    url=patent["url"],
                )
            )
            texts.append(paragraph)

    embeddings = embedder.embed_passages(texts)
    store = FaissStore(dim=embeddings.shape[1])
    store.add(chunks, embeddings)
    print(f"Indexed {len(store)} passages from {len(CORPUS)} patents.", flush=True)
    return store

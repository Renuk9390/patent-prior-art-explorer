# Patent Prior-Art Explorer — Technical Documentation

## What this is

A small, free, explainable patent semantic search demo. It lets a
visitor search a curated set of real, publicly granted patents by
**meaning** rather than keyword overlap, and see exactly **why** each
result matched — including specific evidence sentences highlighted
within the passage, even when the query's exact words never appear in
the text.

This is deliberately scoped as **Phase 1** of a larger vision (see
"Future direction" below) — built to be simple, fully free to run, and
easy to extend, rather than to do everything at once.

## Pipeline

```
documents → chunking → embeddings → FAISS vector store → query
    → retrieval → sentence-level explainability / highlighting
    → (optional) LLM plain-language explanation → results
```

Each stage is its own module inside the `patentsearch` package under
`src/`, so any one piece can be swapped or upgraded later without
touching the others:

| Stage | File | What it does |
|---|---|---|
| Data types | `src/patentsearch/core/types.py` | The `Chunk` and `SearchResult` types shared by every stage. |
| Demo corpus | `src/patentsearch/data/demo_corpus.py` | The curated, hand-chunked set of real patents. |
| Chunking | `src/patentsearch/ingest/chunking.py` | Splits text into retrievable passages. Uploaded PDFs are chunked automatically with a lightweight, dependency-free sentence-grouping algorithm. |
| Embeddings | `src/patentsearch/embedders/sentence_transformer.py` | Wraps `BAAI/bge-small-en-v1.5`, a small, free, CPU-friendly embedding model. Applies the correct asymmetric query-instruction prefix that BGE models expect. |
| Vector store | `src/patentsearch/stores/faiss_store.py` | `FaissStore`, a thin wrapper around FAISS's `IndexFlatIP` (exact cosine similarity search). |
| Highlighting | `src/patentsearch/explain/highlight.py` | Fully local, deterministic sentence-level similarity scoring and highlighting (always runs). |
| Novelty Gap Graph | `src/patentsearch/explain/novelty_graph.py` | Parses the graph section of the LLM's analysis into nodes and edges. |
| LLM analysis | `src/patentsearch/llm/groq.py` | `GroqProvider`: optional structured prior-art analysis via a free Groq-hosted model; returns nothing instead of failing if unavailable. |
| Search service | `src/patentsearch/service.py` | `SearchService`, the single entry point the UI calls for searching, indexing uploads, highlighting and LLM analysis. The UI never calls the model, FAISS or the LLM directly. |
| Settings | `config/demo.yaml`, `src/patentsearch/config.py` | Model name, query prefix, number of results, LLM provider and models. Another deployment points the `PATENTSEARCH_CONFIG` environment variable at its own file. |
| Interfaces | `embedders/base.py`, `stores/base.py`, `llm/base.py` | The small contracts a replacement embedding model, vector store or LLM provider must meet. |
| App | `apps/gradio_app.py` | Gradio UI; renders results and calls only `SearchService`. |
| Entry point | `app.py` | Thin wrapper that HuggingFace Spaces runs; it makes `src/` importable and starts the UI. |

## Why these specific choices

**BGE-small over a larger/proprietary embedding model.** This Space is
expected to run on HuggingFace's free CPU tier. A small model keeps
search latency low without needing a GPU. BGE-small still performs
competitively on retrieval benchmarks for its size.

**FAISS `IndexFlatIP` over an approximate index or a hosted vector
database.** The corpus here is tiny (a handful of patents, plus at
most one uploaded document per session) — exact search is both fast
enough and simpler to reason about than an approximate index, and
running FAISS in-process means there's no separate database service to
provision, pay for, or keep alive.

**Deterministic, embedding-based explainability as the core layer,
with an LLM explanation as strictly optional polish.** A public demo
that depends on a third-party LLM API to function at all is fragile —
rate limits, pricing changes, or an expired key would silently break
the whole tool. By computing "why this matched" directly from sentence
embeddings (the same model already loaded for retrieval), the
explainability and highlighting features work offline, for free,
indefinitely. The optional Groq-based plain-language layer
(`src/patentsearch/llm/groq.py::llm_explain`) is wrapped in a broad
`try/except` specifically so that if it's ever unavailable, the
visitor sees a working demo with one fewer nice-to-have detail —
never a broken page.

**A small, real, curated corpus instead of arbitrary live patent-office
API calls.** Patent specifications are public-domain government
disclosures, not subject to the copyright restrictions that apply to
most web content — using real excerpts is both legitimate and
necessary for this to function as a credible patent search demo.
Curating ~6 diverse patents as static data (`src/patentsearch/data/demo_corpus.py`) avoids
any runtime dependency on an external patent-office API, which keeps
the demo simple and immune to that API ever changing or rate-limiting.

## Why an upload feature, and how it stays simple

Letting a visitor upload their own PDF and search it alongside the
demo corpus makes the tool genuinely useful, not just a fixed showcase.
This is implemented as a **per-session** FAISS index built fresh from
the uploaded file (`apps/gradio_app.py::on_upload`), held in Gradio's `gr.State` —
never merged into the shared corpus index, so one visitor's upload is
never visible to another, and the shared index never grows unbounded
across many uploads from many people.

## Future direction: the Novelty Gap Explorer

This pipeline is intentionally built so that a more ambitious system —
claim decomposition, claim-to-evidence alignment via natural language
inference, and automated gap detection — could be layered on top
later without rearchitecting anything here. That system needs a
capable LLM for claim decomposition and an NLI model for alignment,
each adding real cost, latency, and failure surface, which is why it's
treated as a deliberate **Phase 2** rather than built into this first
release.

## Deploying

See `README.md` in this folder for local development and Hugging Face
Space deployment instructions, including how to (optionally) configure
the `GROQ_API_KEY` secret for the LLM explanation layer.

## New changes made on November 2026
Your Groq models were retired. Groq shut down both llama-3.1-8b-instant and llama-3.3-70b-versatile on 16 August 2026, which matches the model_not_found errors in your log. Your API key and the app code were fine. That’s why the AI analysis stopped working a few weeks ago.

Groq’s official replacements are openai/gpt-oss-20b (the fast one) and openai/gpt-oss-120b. I changed the LLM code (then `pipeline/explain.py`, now `src/patentsearch/llm/groq.py`) in three ways:

New models: it now uses openai/gpt-oss-20b first, then openai/gpt-oss-120b.
Reasoning settings: these models do hidden “thinking” that uses up tokens, so I set reasoning effort to low and raised the token limit from 700 to 1500. Otherwise the answer could be cut off or come back empty.
Self-healing: if both models ever fail, the code asks Groq which models your key can use and tries those. This keeps the demo from breaking again the next time Groq retires a model.
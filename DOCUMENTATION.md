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

Each stage is its own module under `pipeline/`, so any one piece can
be swapped or upgraded later without touching the others:

| Stage | File | What it does |
|---|---|---|
| Chunking | `pipeline/chunking.py` | Splits text into retrievable passages. The curated corpus is pre-chunked by hand (`corpus/patents.py`); uploaded PDFs are chunked automatically with a lightweight, dependency-free sentence-grouping algorithm. |
| Embeddings | `pipeline/embeddings.py` | Wraps `BAAI/bge-small-en-v1.5`, a small, free, CPU-friendly embedding model. Applies the correct asymmetric query-instruction prefix that BGE models expect. |
| Vector store | `pipeline/vectorstore.py` | A thin wrapper around FAISS's `IndexFlatIP` (exact cosine similarity search), plus the chunk metadata needed to display a result. |
| Explainability | `pipeline/explain.py` | Two layers: a fully local, deterministic sentence-level similarity scorer (always runs), and an optional LLM-generated plain-language explanation (only if configured, fails silently if unavailable). |
| App | `app.py` | Gradio UI gluing the above together. |

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
(`pipeline/explain.py::llm_explain`) is wrapped in a broad
`try/except` specifically so that if it's ever unavailable, the
visitor sees a working demo with one fewer nice-to-have detail —
never a broken page.

**A small, real, curated corpus instead of arbitrary live patent-office
API calls.** Patent specifications are public-domain government
disclosures, not subject to the copyright restrictions that apply to
most web content — using real excerpts is both legitimate and
necessary for this to function as a credible patent search demo.
Curating ~6 diverse patents as static data (`corpus/patents.py`) avoids
any runtime dependency on an external patent-office API, which keeps
the demo simple and immune to that API ever changing or rate-limiting.

## Why an upload feature, and how it stays simple

Letting a visitor upload their own PDF and search it alongside the
demo corpus makes the tool genuinely useful, not just a fixed showcase.
This is implemented as a **per-session** FAISS index built fresh from
the uploaded file (`app.py::on_upload`), held in Gradio's `gr.State` —
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

Groq’s official replacements are openai/gpt-oss-20b (the fast one) and openai/gpt-oss-120b. I changed explain.py in three ways:

New models: it now uses openai/gpt-oss-20b first, then openai/gpt-oss-120b.
Reasoning settings: these models do hidden “thinking” that uses up tokens, so I set reasoning effort to low and raised the token limit from 700 to 1500. Otherwise the answer could be cut off or come back empty.
Self-healing: if both models ever fail, the code asks Groq which models your key can use and tries those. This keeps the demo from breaking again the next time Groq retires a model.
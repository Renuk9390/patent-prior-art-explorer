"""
Patent Prior-Art Explorer -- a small, free, explainable patent semantic
search demo.

Run locally with:   python app.py
Deploy on HuggingFace Spaces by pushing this folder to a Space with
sdk: gradio (see README.md's YAML frontmatter).

Architecture (see DOCUMENTATION.md for the deeper write-up of *why*
each piece was chosen):

    documents -> chunking -> embeddings -> FAISS vector store -> query
        -> retrieval -> sentence-level explainability/highlighting
        -> (optional) LLM plain-language explanation -> results

Everything except the optional LLM explanation step runs fully offline
on free, self-contained, CPU-friendly components, so the demo keeps
working even if a visitor never configures an LLM API key at all.

Design note on state: the curated corpus index is built once at
startup and kept as a plain module-level object, shared read-only
across every visitor -- it is deliberately NOT stored in gr.State,
since Gradio deep-copies a State component's *initial* value for every
new session, which would be wasteful (and risky -- FAISS index objects
aren't guaranteed to deep-copy cleanly) for a large shared index.
gr.State is used only for what's genuinely per-visitor: the optional
index built from an uploaded PDF, whose initial value is simply None.
"""

from __future__ import annotations

import html
import os

import gradio as gr

from corpus import CORPUS
from pipeline.chunking import chunk_text, extract_pdf_text
from pipeline.embeddings import embed_passages, embed_query
from pipeline.explain import explain_match, highlight_html, llm_explain
from pipeline.vectorstore import Chunk, VectorStore

TOP_K = 5

EXAMPLE_QUERIES = [
    "wireless charging of a mobile device",
    "preventing a battery from overheating",
    "cancelling background noise in earphones",
    "drone detecting obstacles without lidar",
    "a phone screen that can fold in half",
    "a skin patch that delivers medicine",
]

CUSTOM_CSS = """
/* ── base reset ────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

/* ── result card ───────────────────────────────────────────── */
.result-card {
    border: 1px solid #e3e8ef;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 16px;
    background: #ffffff;
    overflow-wrap: break-word;
    word-break: break-word;
}
@media (min-width: 640px) {
    .result-card { padding: 20px 22px; }
}

/* ── card header ───────────────────────────────────────────── */
.result-header {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}
.result-domain {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #c2880a;
    font-weight: 700;
}
.result-score {
    font-size: 11px;
    font-family: monospace;
    color: #6b7280;
    background: #f3f5f8;
    padding: 2px 8px;
    border-radius: 999px;
    white-space: nowrap;
}
.result-title {
    margin: 4px 0 10px 0;
    font-size: 15px;
    line-height: 1.4;
}
@media (min-width: 640px) {
    .result-title { font-size: 16px; }
}
.result-id a {
    font-weight: 400;
    font-size: 12px;
    color: #6b7280;
    word-break: break-all;
}

/* ── retrieved passage with highlighted evidence ───────────── */
.result-snippet {
    font-size: 13px;
    line-height: 1.65;
    color: #374151;
}
@media (min-width: 640px) {
    .result-snippet { font-size: 14px; }
}
.result-snippet mark {
    background: #faf0d8;
    color: #11203a;
    padding: 0 2px;
    border-radius: 3px;
    font-style: normal;
}

/* ── LLM analysis panel ────────────────────────────────────── */
.llm-panel {
    margin-top: 14px;
    border-top: 1px solid #e3e8ef;
    padding-top: 14px;
}
.llm-panel-header {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 12px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    color: #8b99ad;
}

/* ── individual analysis sections ──────────────────────────── */
.llm-section {
    margin-bottom: 10px;
}
.llm-section-label {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 5px;
}
.llm-section-label.concepts  { color: #0b6e4f; }
.llm-section-label.elements  { color: #1e56a0; }
.llm-section-label.relevance { color: #7c3aed; }

.llm-tag {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
}
.llm-tag.concepts  { background: #d1fae5; color: #065f46; }
.llm-tag.elements  { background: #dbeafe; color: #1e3a5f; }
.llm-tag.relevance { background: #ede9fe; color: #5b21b6; }

.llm-bullet-list {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
}
.llm-bullet-list li {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 12px;
    color: #374151;
    background: #f8fafc;
    border: 1px solid #e3e8ef;
    border-radius: 6px;
    padding: 3px 8px;
    line-height: 1.4;
}
.llm-bullet-list.elements li {
    background: #f0f6ff;
    border-color: #bfdbfe;
    color: #1e3a5f;
}
.llm-relevance-text {
    font-size: 13px;
    line-height: 1.6;
    color: #4b5563;
    font-style: italic;
    background: #faf8ff;
    border-left: 3px solid #7c3aed;
    padding: 8px 12px;
    border-radius: 0 6px 6px 0;
    margin: 0;
}
"""

# Shared, read-only, built once at process startup -- see the module
# docstring above for why this is a global rather than gr.State.
_CORPUS_STORE: VectorStore | None = None


def get_corpus_store() -> VectorStore:
    global _CORPUS_STORE
    if _CORPUS_STORE is None:
        print("Loading embedding model and indexing the demo corpus...")
        _CORPUS_STORE = _build_corpus_store()
        print(f"Indexed {len(_CORPUS_STORE)} passages from {len(CORPUS)} patents.")
    return _CORPUS_STORE


def _build_corpus_store() -> VectorStore:
    """Embed and index the curated demo corpus."""
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

    embeddings = embed_passages(texts)
    store = VectorStore(dim=embeddings.shape[1])
    store.add(chunks, embeddings)
    return store


def _parse_llm_sections(text: str) -> dict[str, list[str] | str]:
    """Parse the structured LLM output into its three sections.

    Returns a dict with keys 'concepts', 'elements', 'relevance'.
    Each of 'concepts' and 'elements' is a list of bullet strings;
    'relevance' is a plain string. Any section that can't be found
    gets an empty list / empty string so the template always has
    something to check without raising KeyError.
    """
    import re

    sections: dict[str, list[str] | str] = {
        "concepts": [],
        "elements": [],
        "relevance": "",
    }

    # Split on the three known section headers (case-insensitive).
    pattern = re.compile(
        r"(MATCHING CONCEPTS:|POTENTIAL SUPPORTED ELEMENTS:|RELEVANCE:)",
        re.IGNORECASE,
    )
    parts = pattern.split(text)

    current = None
    for part in parts:
        upper = part.strip().upper()
        if upper == "MATCHING CONCEPTS:":
            current = "concepts"
        elif upper == "POTENTIAL SUPPORTED ELEMENTS:":
            current = "elements"
        elif upper == "RELEVANCE:":
            current = "relevance"
        elif current == "concepts":
            bullets = [
                ln.lstrip("-•· ").strip()
                for ln in part.strip().splitlines()
                if ln.strip() and not ln.strip().upper().startswith("POTENTIAL")
            ]
            sections["concepts"] = [b for b in bullets if b]
        elif current == "elements":
            bullets = [
                ln.lstrip("-•· ").strip()
                for ln in part.strip().splitlines()
                if ln.strip() and not ln.strip().upper().startswith("RELEVANCE")
            ]
            sections["elements"] = [b for b in bullets if b]
        elif current == "relevance":
            sections["relevance"] = part.strip()

    return sections


def _render_llm_panel(llm_text: str) -> str:
    """Render the structured LLM analysis as a three-section HTML panel."""
    parsed = _parse_llm_sections(llm_text)
    concepts: list[str] = parsed["concepts"]  # type: ignore[assignment]
    elements: list[str] = parsed["elements"]  # type: ignore[assignment]
    relevance: str = parsed["relevance"]  # type: ignore[assignment]

    def bullet_items(items: list[str], css_class: str) -> str:
        if not items:
            return f'<li class="none">None identified</li>'
        return "".join(
            f"<li>{html.escape(item)}</li>" for item in items
        )

    concepts_html = (
        f'<div class="llm-section">'
        f'<div class="llm-section-label concepts">'
        f'<span class="llm-tag concepts">Matching Concepts</span></div>'
        f'<ul class="llm-bullet-list">{bullet_items(concepts, "concepts")}</ul>'
        f"</div>"
    )

    elements_html = (
        f'<div class="llm-section">'
        f'<div class="llm-section-label elements">'
        f'<span class="llm-tag elements">Potential Supported Elements</span></div>'
        f'<ul class="llm-bullet-list elements">{bullet_items(elements, "elements")}</ul>'
        f"</div>"
    )

    relevance_html = ""
    if relevance:
        relevance_html = (
            f'<div class="llm-section">'
            f'<div class="llm-section-label relevance">'
            f'<span class="llm-tag relevance">Relevance</span></div>'
            f'<p class="llm-relevance-text">{html.escape(relevance)}</p>'
            f"</div>"
        )

    return (
        f'<div class="llm-panel">'
        f'<div class="llm-panel-header">&#129302; AI Prior-Art Analysis</div>'
        f"{concepts_html}{elements_html}{relevance_html}"
        f"</div>"
    )


def render_card(result, query: str) -> str:
    chunk = result.chunk
    explanation = explain_match(query, chunk.text, top_n=2)
    snippet_html = highlight_html(chunk.text, explanation["highlighted"])
    llm_text = llm_explain(query, chunk.text, explanation["highlighted"])

    score_pct = f"{result.score * 100:.1f}%"
    title = html.escape(chunk.doc_title)
    domain = html.escape(chunk.domain)
    doc_id = html.escape(chunk.doc_id)

    if chunk.url:
        id_html = (
            f'<a href="{html.escape(chunk.url)}" target="_blank"'
            f' rel="noreferrer noopener">{doc_id} ↗</a>'
        )
    else:
        id_html = doc_id

    llm_block = _render_llm_panel(llm_text) if llm_text else ""

    return (
        f'<div class="result-card">'
        f'<div class="result-header">'
        f'<span class="result-domain">{domain}</span>'
        f'<span class="result-score">match {score_pct}</span>'
        f"</div>"
        f'<h3 class="result-title">{title}'
        f' <span class="result-id">({id_html})</span></h3>'
        f'<p class="result-snippet">{snippet_html}</p>'
        f"{llm_block}"
        f"</div>"
    )


def on_upload(file_path: str | None):
    """Parse, chunk, and index an uploaded PDF into a per-session VectorStore."""
    if not file_path:
        return None, "No file uploaded yet."

    try:
        text = extract_pdf_text(file_path)
    except Exception as exc:
        return None, f"Couldn't read that PDF: {exc}"

    if not text.strip():
        return None, (
            "Couldn't extract any text from that PDF \u2014 it may be a "
            "scanned image without a selectable text layer."
        )

    pieces = chunk_text(text)
    if not pieces:
        return None, "The PDF was read, but no usable text chunks were found."

    filename = os.path.basename(file_path)
    chunks = [
        Chunk(
            text=p,
            doc_id="uploaded",
            doc_title=filename,
            domain="Your uploaded document",
            url=None,
        )
        for p in pieces
    ]
    embeddings = embed_passages(pieces)
    store = VectorStore(dim=embeddings.shape[1])
    store.add(chunks, embeddings)

    status = (
        f"Indexed {len(chunks)} passages from \u201c{filename}\u201d \u2014 "
        "it will now be included alongside the demo corpus in your searches."
    )
    return store, status


def do_search(query: str, uploaded_store):
    query = (query or "").strip()
    if not query:
        return "<p style='color:#6b7280;'>Type a search query above, then press Search.</p>"

    query_vec = embed_query(query)
    corpus_store = get_corpus_store()
    results = list(corpus_store.search(query_vec, top_k=TOP_K))
    if uploaded_store is not None and len(uploaded_store) > 0:
        results += uploaded_store.search(query_vec, top_k=TOP_K)

    results.sort(key=lambda r: r.score, reverse=True)
    results = results[:TOP_K]

    if not results:
        return "<p style='color:#6b7280;'>No results found.</p>"

    return "\n".join(render_card(r, query) for r in results)


def build_app() -> gr.Blocks:
    # Warm the model and index at startup rather than on the first
    # visitor's request, so the first search isn't slowed down by a
    # cold model load.
    get_corpus_store()

    with gr.Blocks(css=CUSTOM_CSS, title="Patent Prior-Art Explorer") as demo:
        uploaded_store_state = gr.State(None)

        gr.Markdown(
            "# Patent Prior-Art Explorer\n"
            "A small, explainable semantic search demo built on free, "
            "open components \u2014 search a curated set of real granted "
            "patents by **meaning**, not just keywords, and see exactly "
            "**why** each result matched.\n\n"
            "Built by [Renukswamy Chikkamath](https://your-website-url-here)"
            ", based on his PhD research in AI for patent retrieval and "
            "explainable prior-art search."
        )

        with gr.Row():
            query_box = gr.Textbox(
                label="Search query",
                placeholder="e.g. wireless charging of a mobile device",
                scale=4,
            )
            search_button = gr.Button("Search", variant="primary", scale=1)

        gr.Examples(examples=EXAMPLE_QUERIES, inputs=query_box)

        with gr.Accordion("Optional: also search your own PDF", open=False):
            file_input = gr.File(
                label="Upload a patent, paper, or technical PDF", type="filepath"
            )
            upload_status = gr.Markdown("No file uploaded yet.")

        results_html = gr.HTML()

        with gr.Accordion("How this works", open=False):
            gr.Markdown(
                "**Semantic search:** your query and every document passage "
                "are embedded with the same free model (`BAAI/bge-small-en-v1.5`) "
                "and compared by meaning, so a search for \u201cpreventing a "
                "battery from overheating\u201d can surface a passage about "
                "\u201cthermal runaway suppression\u201d even though no words "
                "overlap.\n\n"
                "**Explainable retrieval & highlighting:** each matched "
                "passage is split into sentences, and every sentence is "
                "scored against your query with the same embedding model. "
                "The highest-scoring sentences are highlighted directly in "
                "the result \u2014 this runs entirely offline, with no "
                "external API call.\n\n"
                "**Optional LLM explanation:** if configured, a short "
                "plain-language explanation is also generated by a "
                "free-tier LLM. If that service is unavailable, the demo "
                "falls back to the embedding-based explanation above with "
                "no visible error."
            )

        search_button.click(
            do_search,
            inputs=[query_box, uploaded_store_state],
            outputs=[results_html],
        )
        query_box.submit(
            do_search,
            inputs=[query_box, uploaded_store_state],
            outputs=[results_html],
        )
        file_input.change(
            on_upload,
            inputs=[file_input],
            outputs=[uploaded_store_state, upload_status],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()

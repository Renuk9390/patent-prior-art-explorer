"""
Patent Prior-Art Explorer -- a small, free, explainable patent semantic
search demo.

Started by the thin app.py wrapper at the repo root (python app.py),
which is also what the HuggingFace Space runs.

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
from concurrent.futures import ThreadPoolExecutor, as_completed

import gradio as gr

from patentsearch.explain.highlight import highlight_html
from patentsearch.explain.novelty_graph import parse_novelty_graph
from patentsearch.ingest.chunking import chunk_text, extract_pdf_text
from patentsearch.service import SearchService

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
    border: 1px solid #e3e8ef !important;
    border-radius: 14px !important;
    padding: 16px !important;
    margin-bottom: 16px !important;
    background: #ffffff !important;
    overflow-wrap: break-word !important;
    word-break: break-word !important;
    width: 100% !important;
}
@media (min-width: 640px) {
    .result-card { padding: 20px 22px !important; }
}

/* ── card header ───────────────────────────────────────────── */
.result-header {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 6px !important;
    justify-content: space-between !important;
    align-items: center !important;
    margin-bottom: 6px !important;
}
.result-domain {
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #c2880a !important;
    font-weight: 700 !important;
}
.result-score {
    font-size: 11px !important;
    font-family: monospace !important;
    color: #374151 !important;
    background: #f3f5f8 !important;
    padding: 2px 8px !important;
    border-radius: 999px !important;
    white-space: nowrap !important;
}
.result-title {
    margin: 4px 0 10px 0 !important;
    font-size: 15px !important;
    line-height: 1.4 !important;
    color: #0b1220 !important;
}
@media (min-width: 640px) {
    .result-title { font-size: 16px !important; }
}
.result-id a {
    font-weight: 400 !important;
    font-size: 12px !important;
    color: #4b5b71 !important;
    word-break: break-all !important;
    text-decoration: underline !important;
}

/* ── retrieved passage with highlighted evidence ───────────── */
.result-snippet {
    font-size: 13px !important;
    line-height: 1.65 !important;
    color: #1f2937 !important;
}
@media (min-width: 640px) {
    .result-snippet { font-size: 14px !important; }
}
.result-snippet mark {
    background: #fef3c7 !important;
    color: #0b1220 !important;
    padding: 1px 3px !important;
    border-radius: 3px !important;
    font-style: normal !important;
}

/* ── LLM analysis panel ────────────────────────────────────── */
.llm-panel {
    margin-top: 14px !important;
    border-top: 2px solid #e3e8ef !important;
    padding-top: 14px !important;
}
.llm-panel-header {
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
    margin-bottom: 12px !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.10em !important;
    color: #6b7280 !important;
}

/* ── individual analysis sections ──────────────────────────── */
.llm-section {
    margin-bottom: 12px !important;
}
.llm-section-label {
    display: flex !important;
    align-items: center !important;
    gap: 5px !important;
    margin-bottom: 7px !important;
}
.llm-tag {
    display: inline-block !important;
    padding: 3px 10px !important;
    border-radius: 5px !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
.llm-tag.concepts  { background: #d1fae5 !important; color: #064e3b !important; }
.llm-tag.elements  { background: #dbeafe !important; color: #1e3a5f !important; }
.llm-tag.relevance { background: #ede9fe !important; color: #4c1d95 !important; }

/* Concept pills — green tinted */
.llm-bullet-list {
    list-style: none !important;
    margin: 0 !important;
    padding: 0 !important;
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 6px !important;
}
.llm-bullet-list li {
    display: inline-flex !important;
    font-size: 12px !important;
    color: #064e3b !important;
    background: #ecfdf5 !important;
    border: 1px solid #6ee7b7 !important;
    border-radius: 6px !important;
    padding: 4px 10px !important;
    line-height: 1.4 !important;
    font-weight: 500 !important;
}
/* Element pills — blue tinted, override the base li rule */
.llm-bullet-list.elements li {
    color: #1e3a5f !important;
    background: #eff6ff !important;
    border: 1px solid #93c5fd !important;
    font-weight: 400 !important;
}

/* Relevance block */
.llm-relevance-text {
    font-size: 13px !important;
    line-height: 1.7 !important;
    color: #1f2937 !important;
    font-style: italic !important;
    background: #f5f3ff !important;
    border-left: 3px solid #7c3aed !important;
    padding: 10px 14px !important;
    border-radius: 0 6px 6px 0 !important;
    margin: 0 !important;
    display: block !important;
}

/* ── collapsible <details> panels ──────────────────────────── */
.llm-details, .graph-details {
    margin-top: 10px !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid #e3e8ef !important;
}
.llm-details > summary, .graph-details > summary {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    padding: 10px 14px !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.10em !important;
    cursor: pointer !important;
    user-select: none !important;
    list-style: none !important;
}
.llm-details > summary::-webkit-details-marker { display: none !important; }
.llm-details > summary { background: #f8fafc !important; color: #374151 !important; }
.graph-details > summary { background: #f0f9ff !important; color: #075985 !important; }
.llm-details[open] > summary { border-bottom: 1px solid #e3e8ef !important; }
.graph-details[open] > summary { border-bottom: 1px solid #bae6fd !important; }
.summary-chevron {
    margin-left: auto !important;
    font-size: 10px !important;
    color: #9ca3af !important;
}
.llm-details-body, .graph-details-body {
    padding: 14px !important;
    background: #ffffff !important;
}

/* ── Novelty Gap Graph ──────────────────────────────────────── */
.ngraph-columns {
    display: grid !important;
    grid-template-columns: 1fr 1fr !important;
    gap: 12px !important;
    margin-bottom: 14px !important;
}
.ngraph-col-header {
    font-size: 10px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin-bottom: 8px !important;
}
.ngraph-col-header.claim-header   { color: #1d4ed8 !important; }
.ngraph-col-header.evidence-header { color: #15803d !important; }
.ngraph-node {
    border-radius: 8px !important;
    padding: 7px 10px !important;
    margin-bottom: 6px !important;
    font-size: 12px !important;
    line-height: 1.4 !important;
    color: #1e3a5f !important;
}
.ngraph-node .node-id {
    font-weight: 700 !important;
    font-size: 11px !important;
    margin-right: 4px !important;
}
.ngraph-node.claim-node {
    background: #eff6ff !important;
    border: 1px solid #93c5fd !important;
    color: #1e3a5f !important;
}
.ngraph-node.claim-node .node-id { color: #1d4ed8 !important; }
.ngraph-node.gap-node {
    background: #fff7ed !important;
    border: 1px solid #fdba74 !important;
    color: #7c2d12 !important;
}
.ngraph-node.gap-node .node-id { color: #c2410c !important; }
.ngraph-node.evidence-node {
    background: #f0fdf4 !important;
    border: 1px solid #86efac !important;
    color: #14532d !important;
}
.ngraph-node.evidence-node .node-id { color: #15803d !important; }
.ngraph-edges-header, .ngraph-gaps-header {
    font-size: 10px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin: 12px 0 6px 0 !important;
}
.ngraph-edges-header { color: #374151 !important; }
.ngraph-gaps-header  { color: #c2410c !important; }
.ngraph-edge {
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
    padding: 5px 8px !important;
    border-radius: 6px !important;
    margin-bottom: 5px !important;
    font-size: 12px !important;
    background: #f9fafb !important;
    border: 1px solid #e5e7eb !important;
}
.ngraph-edge .edge-src   { font-weight: 700 !important; color: #1d4ed8 !important; }
.ngraph-edge .edge-tgt   { font-weight: 700 !important; color: #15803d !important; }
.ngraph-edge .edge-arrow { color: #9ca3af !important; }
.edge-weight {
    margin-left: auto !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    padding: 1px 7px !important;
    border-radius: 99px !important;
    text-transform: uppercase !important;
}
.edge-weight.strong { background: #dcfce7 !important; color: #166534 !important; }
.edge-weight.weak   { background: #fef9c3 !important; color: #854d0e !important; }
.ngraph-gap-row {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    padding: 5px 8px !important;
    border-radius: 6px !important;
    margin-bottom: 5px !important;
    font-size: 12px !important;
    background: #fff7ed !important;
    border: 1px solid #fed7aa !important;
    color: #7c2d12 !important;
}
.ngraph-gap-row .gap-id { font-weight: 700 !important; color: #c2410c !important; }

/* ── pending AI analysis placeholder ───────────────────────── */
.llm-pending {
    margin-top: 12px !important;
    padding: 10px 14px !important;
    border-radius: 10px !important;
    background: #f8fafc !important;
    border: 1px dashed #cbd5e1 !important;
    font-size: 12px !important;
    color: #6b7280 !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}
.llm-pending-spinner {
    display: inline-block !important;
    animation: llm-pending-pulse 1.4s ease-in-out infinite !important;
}
@keyframes llm-pending-pulse {
    0%, 100% { opacity: 0.35; }
    50%      { opacity: 1; }
}
@media (prefers-reduced-motion: reduce) {
    .llm-pending-spinner { animation: none !important; }
}
"""

# Shared, read-only, built once at process startup -- see the module
# docstring above for why this is a global rather than gr.State.
_SERVICE: SearchService | None = None


def get_service() -> SearchService:
    global _SERVICE
    if _SERVICE is None:
        print("Loading embedding model and indexing the demo corpus...")
        _SERVICE = SearchService.from_config()
    return _SERVICE


def _parse_llm_sections(text: str) -> dict:
    """Parse MATCHING CONCEPTS, POTENTIAL SUPPORTED ELEMENTS, and RELEVANCE."""
    import re
    sections: dict = {"concepts": [], "elements": [], "relevance": ""}
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
                    and not ln.strip().upper().startswith("NOVELTY")
            ]
            sections["elements"] = [b for b in bullets if b]
        elif current == "relevance":
            # Stop at the NOVELTY GAP GRAPH heading
            chunk = re.split(r"NOVELTY GAP GRAPH", part, flags=re.IGNORECASE)[0]
            sections["relevance"] = chunk.strip()
    return sections


def _build_graph_svg(g: dict) -> str:
    """Render the Novelty Gap Graph as an inline SVG bipartite diagram.

    Left column  = CLAIM nodes  (from search query)   — blue / orange if gap
    Right column = EVIDENCE nodes (from retrieved passage) — green
    Lines between them show support edges, colour/dash-coded by strength.
    """
    claims   = g["claims"]    # {id: label}
    evidence = g["evidence"]  # {id: label}
    edges    = g["edges"]     # [(src, tgt, weight)]
    gap_set  = set(g["gaps"])

    if not claims and not evidence:
        return ""

    # ── Layout constants ──────────────────────────────────────────────
    SVG_W     = 680
    NODE_W    = 190
    NODE_H    = 56
    CLAIM_CX  = 115          # centre-x of all claim nodes
    EVID_CX   = SVG_W - 115  # centre-x of all evidence nodes
    ROW_H     = 74
    TOP_PAD   = 44           # space for column headers

    n_rows = max(len(claims), len(evidence), 1)
    SVG_H  = TOP_PAD + n_rows * ROW_H + 20

    def node_cy(idx: int, total: int) -> float:
        """Vertically centre nodes evenly within the canvas."""
        usable = SVG_H - TOP_PAD - 20
        spacing = usable / max(total, 1)
        return TOP_PAD + idx * spacing + spacing / 2

    claim_list    = list(claims.items())
    evidence_list = list(evidence.items())

    claim_y  = {cid: node_cy(i, len(claim_list))    for i, (cid, _) in enumerate(claim_list)}
    evid_y   = {eid: node_cy(i, len(evidence_list)) for i, (eid, _) in enumerate(evidence_list)}

    parts: list[str] = []

    # ── Column header labels ──────────────────────────────────────────
    parts.append(
        f'<text x="{CLAIM_CX}" y="22" text-anchor="middle" '
        f'font-size="11" font-weight="bold" fill="#1d4ed8" font-family="sans-serif">'
        f'SEARCH QUERY ELEMENTS</text>'
    )
    parts.append(
        f'<text x="{EVID_CX}" y="22" text-anchor="middle" '
        f'font-size="11" font-weight="bold" fill="#15803d" font-family="sans-serif">'
        f'PRIOR ART EVIDENCE</text>'
    )

    # ── Edges (drawn first so nodes sit on top) ───────────────────────
    for src, tgt, weight in edges:
        if src not in claim_y or tgt not in evid_y:
            continue
        x1 = CLAIM_CX + NODE_W / 2
        x2 = EVID_CX  - NODE_W / 2
        y1 = claim_y[src]
        y2 = evid_y[tgt]
        color  = "#16a34a" if weight == "strong" else "#d97706"
        sw     = "2.5"     if weight == "strong" else "1.5"
        dash   = ""        if weight == "strong" else 'stroke-dasharray="6,3"'
        # mid-point control for a slight curve
        mx = (x1 + x2) / 2
        parts.append(
            f'<path d="M{x1},{y1} C{mx},{y1} {mx},{y2} {x2},{y2}" '
            f'fill="none" stroke="{color}" stroke-width="{sw}" {dash} opacity="0.75"/>'
        )
        # weight label at midpoint
        lx, ly = mx, (y1 + y2) / 2 - 5
        parts.append(
            f'<rect x="{lx-22}" y="{ly-8}" width="44" height="16" rx="4" '
            f'fill="white" opacity="0.85"/>'
        )
        parts.append(
            f'<text x="{lx}" y="{ly+4}" text-anchor="middle" font-size="9" '
            f'font-weight="bold" fill="{color}" font-family="sans-serif">'
            f'{html.escape(weight.upper())}</text>'
        )

    def wrap_label(label: str, max_chars: int = 24) -> tuple[str, str]:
        """Split a label into up to two display lines."""
        if len(label) <= max_chars:
            return label, ""
        # try to split at a space near the midpoint
        mid = max_chars
        split_at = label.rfind(" ", 0, mid)
        if split_at == -1:
            split_at = mid
        line1 = label[:split_at].strip()
        line2 = label[split_at:].strip()
        if len(line2) > max_chars:
            line2 = line2[:max_chars - 1] + "…"
        return line1, line2

    def draw_node(cx: float, cy: float, node_id: str, label: str,
                  fill: str, stroke: str, id_color: str, label_color: str,
                  sub: str = "") -> None:
        x = cx - NODE_W / 2
        y = cy - NODE_H / 2
        # drop-shadow filter reference (defined once in <defs>)
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" '
            f'rx="9" fill="{fill}" stroke="{stroke}" stroke-width="1.5" '
            f'filter="url(#shadow)"/>'
        )
        line1, line2 = wrap_label(label)
        # node ID
        parts.append(
            f'<text x="{cx}" y="{cy - 12}" text-anchor="middle" '
            f'font-size="12" font-weight="bold" fill="{id_color}" font-family="sans-serif">'
            f'{html.escape(node_id)}</text>'
        )
        # label line 1
        parts.append(
            f'<text x="{cx}" y="{cy + 5}" text-anchor="middle" '
            f'font-size="9.5" fill="{label_color}" font-family="sans-serif">'
            f'{html.escape(line1)}</text>'
        )
        if line2:
            parts.append(
                f'<text x="{cx}" y="{cy + 18}" text-anchor="middle" '
                f'font-size="9.5" fill="{label_color}" font-family="sans-serif">'
                f'{html.escape(line2)}</text>'
            )
        if sub:
            parts.append(
                f'<text x="{cx}" y="{cy + (28 if line2 else 20)}" '
                f'text-anchor="middle" font-size="8.5" fill="#f97316" '
                f'font-family="sans-serif">{html.escape(sub)}</text>'
            )

    # ── Claim nodes ───────────────────────────────────────────────────
    for cid, label in claim_list:
        cy = claim_y[cid]
        if cid in gap_set:
            draw_node(CLAIM_CX, cy, cid, label,
                      fill="#fff7ed", stroke="#f97316",
                      id_color="#c2410c", label_color="#7c2d12",
                      sub="⚠ no evidence")
        else:
            draw_node(CLAIM_CX, cy, cid, label,
                      fill="#eff6ff", stroke="#3b82f6",
                      id_color="#1d4ed8", label_color="#1e3a5f")

    # ── Evidence nodes ────────────────────────────────────────────────
    for eid, label in evidence_list:
        cy = evid_y[eid]
        draw_node(EVID_CX, cy, eid, label,
                  fill="#f0fdf4", stroke="#22c55e",
                  id_color="#15803d", label_color="#14532d")

    defs = (
        '<defs>'
        '<filter id="shadow" x="-10%" y="-10%" width="120%" height="130%">'
        '<feDropShadow dx="0" dy="1" stdDeviation="2" flood-color="#00000018"/>'
        '</filter>'
        '</defs>'
    )

    return (
        f'<svg viewBox="0 0 {SVG_W} {SVG_H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;max-width:{SVG_W}px;display:block">'
        + defs
        + "\n".join(parts)
        + "</svg>"
    )


def _render_graph_html(llm_text: str) -> str:
    """Wrap the SVG bipartite graph inside a collapsible <details> block."""
    g = parse_novelty_graph(llm_text)
    if not g["claims"] and not g["evidence"]:
        return ""

    svg = _build_graph_svg(g)

    # Gaps summary below the SVG (text row for accessibility)
    gap_rows = ""
    if g["gaps"]:
        rows = "".join(
            f'<div class="ngraph-gap-row">'
            f'<span class="gap-id">{html.escape(cid)}</span>'
            f'⚠ No supporting evidence — '
            f'{html.escape(g["claims"].get(cid, ""))}'
            f"</div>"
            for cid in g["gaps"]
        )
        gap_rows = (
            f'<div class="ngraph-gaps-header">⚠ Novelty gaps</div>{rows}'
        )

    return (
        f'<details class="graph-details">'
        f"<summary>"
        f"📊 Novelty Gap Graph"
        f'<span class="summary-chevron">▼</span>'
        f"</summary>"
        f'<div class="graph-details-body">'
        f"{svg}{gap_rows}"
        f"</div>"
        f"</details>"
    )


def _render_llm_panel(llm_text: str) -> str:
    """Render AI Prior-Art Analysis as a collapsible <details> block,
    followed by a separate collapsible Novelty Gap Graph block."""
    parsed = _parse_llm_sections(llm_text)
    concepts: list = parsed["concepts"]
    elements: list = parsed["elements"]
    relevance: str = parsed["relevance"]

    def bullet_items(items: list, list_class: str = "") -> str:
        if not items:
            return '<li style="color:#9ca3af;font-style:italic">None identified</li>'
        return "".join(f"<li>{html.escape(item)}</li>" for item in items)

    concepts_html = (
        f'<div class="llm-section">'
        f'<div class="llm-section-label">'
        f'<span class="llm-tag concepts">Matching Concepts</span></div>'
        f'<ul class="llm-bullet-list">{bullet_items(concepts)}</ul>'
        f"</div>"
    )
    elements_html = (
        f'<div class="llm-section">'
        f'<div class="llm-section-label">'
        f'<span class="llm-tag elements">Potential Supported Elements</span></div>'
        f'<ul class="llm-bullet-list elements">{bullet_items(elements)}</ul>'
        f"</div>"
    )
    relevance_html = ""
    if relevance:
        relevance_html = (
            f'<div class="llm-section">'
            f'<div class="llm-section-label">'
            f'<span class="llm-tag relevance">Relevance</span></div>'
            f'<p class="llm-relevance-text">{html.escape(relevance)}</p>'
            f"</div>"
        )

    analysis_block = (
        f'<details class="llm-details">'
        f"<summary>"
        f"🤖 AI Prior-Art Analysis"
        f'<span class="summary-chevron">▼</span>'
        f"</summary>"
        f'<div class="llm-details-body">'
        f"{concepts_html}{elements_html}{relevance_html}"
        f"</div>"
        f"</details>"
    )

    graph_block = _render_graph_html(llm_text)

    return analysis_block + graph_block


_PENDING_PLACEHOLDER = (
    '<div class="llm-pending">'
    '<span class="llm-pending-spinner">⏳</span>'
    " Generating AI prior-art analysis &amp; novelty gap graph"
    "&hellip; results above are already final."
    "</div>"
)


def _card_shell(result, query: str):
    """Compute the fast, local part of a result card (no network calls).

    Returns (header_and_snippet_html, explanation_dict) -- the snippet
    HTML already has highlighting applied via the local embedding model,
    which is fast (no LLM call involved).
    """
    chunk = result.chunk
    explanation = get_service().explain(query, chunk.text)
    snippet_html = highlight_html(chunk.text, explanation["highlighted"])

    score_pct = f"{result.score * 100:.1f}%"
    title = html.escape(chunk.doc_title)
    domain = html.escape(chunk.domain)
    doc_id = html.escape(chunk.doc_id)

    id_html = (
        f'<a href="{html.escape(chunk.url)}" target="_blank"'
        f' rel="noreferrer noopener">{doc_id} ↗</a>'
        if chunk.url else doc_id
    )

    header_and_snippet = (
        f'<div class="result-header">'
        f'<span class="result-domain">{domain}</span>'
        f'<span class="result-score">match {score_pct}</span>'
        f"</div>"
        f'<h3 class="result-title">{title}'
        f' <span class="result-id">({id_html})</span></h3>'
        f'<p class="result-snippet">{snippet_html}</p>'
    )
    return header_and_snippet, explanation


def render_card_pending(result, query: str) -> str:
    """Render a card with its fast parts done and AI analysis still pending."""
    header_and_snippet, _ = _card_shell(result, query)
    return (
        f'<div class="result-card">'
        f"{header_and_snippet}"
        f"{_PENDING_PLACEHOLDER}"
        f"</div>"
    )


def render_card_with_llm(result, query: str, llm_text: str | None) -> str:
    """Render a card with its fast parts plus a completed (or absent) AI analysis."""
    header_and_snippet, _ = _card_shell(result, query)
    if llm_text:
        llm_block = _render_llm_panel(llm_text)
    else:
        reason = "no GROQ_API_KEY is configured" if not get_service().llm_configured \
            else "the AI service did not respond (see Space logs)"
        llm_block = (
            '<div class="llm-pending">&#9888;&#65039; AI analysis &amp; novelty '
            f"graph unavailable right now &mdash; {html.escape(reason)}. "
            "The retrieval results above are unaffected.</div>"
        )
    return (
        f'<div class="result-card">'
        f"{header_and_snippet}"
        f"{llm_block}"
        f"</div>"
    )


def render_card(result, query: str) -> str:
    """Render a fully-resolved card synchronously (used outside the
    streaming search path, e.g. for quick manual testing)."""
    llm_text = get_service().analyse(query, result.chunk.text)
    return render_card_with_llm(result, query, llm_text)


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
    store = get_service().index_texts(
        pieces, doc_id="uploaded", doc_title=filename, domain="Your uploaded document"
    )

    status = (
        f"Indexed {len(store)} passages from \u201c{filename}\u201d \u2014 "
        "it will now be included alongside the demo corpus in your searches."
    )
    return store, status


def do_search(query: str, uploaded_store):
    """Generator: yields fast results first, then streams in AI analysis.

    Phase 1 (fast, local-only): retrieval + embedding-based highlighting.
    Typically well under a second once the model is warm. Yielded
    immediately so the visitor sees results right away.

    Phase 2 (slow, network calls): one Groq LLM call per result card for
    the structured prior-art analysis + novelty gap graph. These five
    calls are independent of each other, so they're run concurrently in
    a thread pool rather than one after another -- this is what cuts
    the background wait from roughly 5x a single call's latency down to
    close to just one call's latency. As each card's analysis finishes,
    an updated HTML snapshot is yielded so the page fills in
    progressively rather than waiting for every card to finish at once.
    """
    query = (query or "").strip()
    if not query:
        yield "<p style='color:#6b7280;'>Type a search query above, then press Search.</p>"
        return

    service = get_service()
    results = service.search(query, extra_stores=[uploaded_store])

    if not results:
        yield "<p style='color:#6b7280;'>No results found.</p>"
        return

    # ── Phase 1: fast, local-only rendering ─────────────────────────
    pending_cards = [render_card_pending(r, query) for r in results]
    yield "\n".join(pending_cards)

    # ── Phase 2: parallel LLM calls, streamed in as they complete ───
    llm_texts: list[str | None] = [None] * len(results)
    done: list[bool] = [False] * len(results)

    def fetch_llm(index: int, result):
        return index, service.analyse(query, result.chunk.text)

    with ThreadPoolExecutor(max_workers=max(len(results), 1)) as pool:
        futures = [
            pool.submit(fetch_llm, i, r) for i, r in enumerate(results)
        ]
        for future in as_completed(futures):
            index, llm_text = future.result()
            llm_texts[index] = llm_text
            done[index] = True

            cards = [
                render_card_with_llm(results[i], query, llm_texts[i])
                if done[i]
                else pending_cards[i]
                for i in range(len(results))
            ]
            yield "\n".join(cards)


def build_app() -> gr.Blocks:
    # Warm the model and index at startup rather than on the first
    # visitor's request, so the first search isn't slowed down by a
    # cold model load.
    get_service()

    with gr.Blocks(title="Patent Prior-Art Explorer") as demo:
        uploaded_store_state = gr.State(None)

        gr.Markdown(
            "# Patent Prior-Art Explorer\n"
            "A small, explainable semantic search demo built on free, "
            "open components \u2014 search a curated set of real granted "
            "patents by **meaning**, not just keywords, and see exactly "
            "**why** each result matched.\n\n"
            "Built by [Renukswamy Chikkamath](https://renukswamy.com)"
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
                "**AI prior-art analysis (optional):** if an LLM key is "
                "configured, a free-tier open model reads each result and "
                "lists the matching concepts, which parts of your query the "
                "passage supports, and a short relevance explanation. It also "
                "draws a **Novelty Gap Graph** linking each element of your "
                "query to evidence in the passage, and flags elements with "
                "no support. If the LLM service is unavailable, a short "
                "notice appears and the search results and highlights still "
                "work.\n\n"
                "**Your own PDF:** an uploaded document is split into passages "
                "and indexed only for your session, then searched alongside "
                "the demo corpus."
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
    app.launch(css=CUSTOM_CSS)

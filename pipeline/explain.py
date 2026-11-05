"""
Explainability layer.

Two parts, deliberately layered so the demo never depends on the
fragile one to function:

  1. `explain_match()` -- always runs. Fully local, free, deterministic.
     Splits the matched chunk into sentences, scores each sentence's
     embedding similarity to the query, and returns the top-scoring
     sentences as evidence for highlighting.

  2. `llm_explain()` -- optional. If GROQ_API_KEY is configured, calls
     a free Groq-hosted LLM for structured prior-art analysis including
     a Novelty Gap Graph. Returns None (never raises) on any failure.

  3. `parse_novelty_graph()` -- parses the NOVELTY GAP GRAPH section
     of an LLM response into structured node/edge data ready to render.
"""

from __future__ import annotations

import html
import os
import re

from .chunking import split_sentences
from .embeddings import embed_passages, embed_query

# Primary model can be overridden without a code change by setting a
# GROQ_MODEL variable in the Space settings. Fallbacks are tried in order
# if the primary is retired or unavailable (Groq rotates models over time).
_GROQ_MODELS = [
    m for m in (
        os.environ.get("GROQ_MODEL"),
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
    ) if m
]
# Last failure reason, surfaced in the UI so a silent outage is visible.
LAST_LLM_ERROR: str | None = None
_GROQ_TIMEOUT_SECONDS = 20


def explain_match(query: str, chunk_text: str, top_n: int = 2) -> dict:
    """Score each sentence in `chunk_text` against the query."""
    sentences = split_sentences(chunk_text)
    if not sentences:
        return {"sentences": [], "highlighted": []}

    query_vec = embed_query(query)
    sentence_vecs = embed_passages(sentences)
    scores = sentence_vecs @ query_vec

    ranked = sorted(
        zip(sentences, scores.tolist()), key=lambda pair: pair[1], reverse=True
    )
    highlighted = [s for s, _ in ranked[:top_n]]

    return {
        "sentences": [
            {"text": s, "score": sc, "highlighted": s in highlighted}
            for s, sc in zip(sentences, scores.tolist())
        ],
        "highlighted": highlighted,
    }


def llm_explain(query: str, chunk_text: str, highlighted: list[str]) -> str | None:
    """Best-effort structured prior-art analysis via a free Groq LLM.

    Returns the raw LLM text string (covering all four sections:
    MATCHING CONCEPTS, POTENTIAL SUPPORTED ELEMENTS, RELEVANCE, and
    NOVELTY GAP GRAPH) or None on any failure.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("[llm_explain] GROQ_API_KEY is not set -- skipping AI analysis.", flush=True)
        return None

    try:
        from groq import Groq

        client = Groq(api_key=api_key, timeout=_GROQ_TIMEOUT_SECONDS)
        evidence = chunk_text.strip()

        prompt = (
            "You are assisting a patent attorney performing prior-art analysis.\n\n"
            "You are given:\n"
            "1. A search query (this represents the invention or claim being searched).\n"
            "2. A retrieved prior-art passage from a patent document.\n\n"
            "Your task is to:\n\n"
            "A. MATCHING TECHNICAL CONCEPTS\n"
            "- Identify key technical concepts in the RETRIEVED PASSAGE that are "
            "semantically related to the search query.\n\n"
            "B. POTENTIAL CLAIM ELEMENTS SUPPORTED\n"
            "- Break the SEARCH QUERY into its key technical components or functions.\n"
            "- For each, state whether the retrieved passage appears to support it.\n"
            "- Keep each element concise.\n\n"
            "C. RELEVANCE EXPLANATION\n"
            "- Explain in 1-2 sentences why the passage may be relevant for "
            "prior-art assessment, mentioning conceptual overlap even if exact "
            "wording differs.\n\n"
            "D. NOVELTY GAP GRAPH\n"
            "CRITICAL RULES — FOLLOW EXACTLY:\n"
            "- CLAIM nodes: Decompose ONLY the SEARCH QUERY TEXT into its "
            "component concepts. Use ONLY words or phrases that are literally "
            "present in, or directly and minimally implied by, the search query "
            "itself. Do NOT expand, infer, or add any technical detail not "
            "present in the query.\n"
            "- FORBIDDEN for CLAIM nodes: any word, phrase, component, mechanism, "
            "or technical concept that appears in the retrieved passage but NOT "
            "in the search query.\n"
            "- CORRECT example: query = 'preventing a battery from overheating' "
            "→ valid CLAIM nodes: C1: preventing overheating, C2: battery. "
            "WRONG: 'using a battery management system' or 'injecting coolant' "
            "— these come from the passage, not the query.\n"
            "- CORRECT example: query = 'wireless charging of a mobile device' "
            "→ valid CLAIM nodes: C1: wireless charging, C2: mobile device. "
            "WRONG: 'induction coil' or 'electromagnetic field' — passage words.\n"
            "- EVIDENCE nodes: Extract ONLY short phrases or partial sentences "
            "from the RETRIEVED PASSAGE that support a claim node. Do NOT use "
            "any words from the search query for evidence nodes.\n"
            "- GRAPH EDGES: connect a CLAIM node to an EVIDENCE node only when "
            "the passage phrase genuinely supports that query concept. "
            "Use (strong) or (weak).\n"
            "- GRAPH GAPS: list any CLAIM node that has no supporting phrase "
            "in the passage.\n\n"
            "Rules:\n"
            "- Do NOT make legal conclusions.\n"
            "- Do NOT state that a claim is novel, anticipated, patentable, or invalid.\n"
            "- Do NOT invent features not reasonably supported by the passage.\n"
            "- Focus only on the provided passage for evidence.\n\n"
            "Return ONLY in the following format:\n\n"
            "MATCHING CONCEPTS:\n"
            "- concept 1\n"
            "- concept 2\n\n"
            "POTENTIAL SUPPORTED ELEMENTS:\n"
            "- element 1\n"
            "- element 2\n\n"
            "RELEVANCE:\n"
            "Short explanation.\n\n"
            "NOVELTY GAP GRAPH\n\n"
            "GRAPH NODES\n\n"
            "CLAIM:\n"
            "C1: [element from search query]\n"
            "C2: [element from search query]\n\n"
            "EVIDENCE:\n"
            "E1: [phrase from retrieved passage]\n"
            "E2: [phrase from retrieved passage]\n\n"
            "GRAPH EDGES\n"
            "C1 -> E1 (strong)\n"
            "C2 -> E2 (weak)\n\n"
            "GRAPH GAPS\n"
            "C3 -> NONE\n\n"
            f"Search Query:\n{query}\n\n"
            f"Retrieved Prior-Art Passage:\n{evidence}\n"
        )

        global LAST_LLM_ERROR
        last_exc = None
        for model_name in _GROQ_MODELS:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=700,
                    temperature=0.2,
                )
                text = response.choices[0].message.content
                LAST_LLM_ERROR = None
                return text.strip() if text else None
            except Exception as exc:  # try the next model
                last_exc = exc
                print(f"[llm_explain] model={model_name} failed: "
                      f"{type(exc).__name__}: {exc}", flush=True)
        LAST_LLM_ERROR = f"{type(last_exc).__name__}: {last_exc}"
        return None
    except Exception as exc:
        LAST_LLM_ERROR = f"{type(exc).__name__}: {exc}"
        print(f"[llm_explain] unexpected failure: {LAST_LLM_ERROR}", flush=True)
        return None


def parse_novelty_graph(llm_text: str) -> dict:
    """Parse the NOVELTY GAP GRAPH section of an LLM response.

    Returns a dict with:
      claims   : {node_id: label}  e.g. {"C1": "wireless communication link"}
      evidence : {node_id: label}  e.g. {"E1": "A wireless comm link is established..."}
      edges    : list of (source, target, weight)  e.g. [("C1", "E1", "strong")]
      gaps     : list of claim node_ids with no evidence  e.g. ["C4", "C5"]
    """
    result: dict = {"claims": {}, "evidence": {}, "edges": [], "gaps": []}

    # Locate the NOVELTY GAP GRAPH block first
    graph_match = re.search(
        r"NOVELTY GAP GRAPH(.+?)(?=$)", llm_text, re.IGNORECASE | re.DOTALL
    )
    if not graph_match:
        return result
    graph_block = graph_match.group(1)

    # ── CLAIM nodes ──────────────────────────────────────────────
    claim_block = re.search(
        r"CLAIM:\s*\n(.*?)(?=EVIDENCE:|GRAPH EDGES|GRAPH GAPS|$)",
        graph_block,
        re.IGNORECASE | re.DOTALL,
    )
    if claim_block:
        for line in claim_block.group(1).splitlines():
            m = re.match(r"\s*(C\d+)\s*:\s*(.+)", line, re.IGNORECASE)
            if m:
                result["claims"][m.group(1).upper()] = m.group(2).strip()

    # ── EVIDENCE nodes ────────────────────────────────────────────
    evidence_block = re.search(
        r"EVIDENCE:\s*\n(.*?)(?=GRAPH EDGES|GRAPH GAPS|$)",
        graph_block,
        re.IGNORECASE | re.DOTALL,
    )
    if evidence_block:
        for line in evidence_block.group(1).splitlines():
            m = re.match(r"\s*(E\d+)\s*:\s*(.+)", line, re.IGNORECASE)
            if m:
                result["evidence"][m.group(1).upper()] = m.group(2).strip()

    # ── GRAPH EDGES ───────────────────────────────────────────────
    edges_block = re.search(
        r"GRAPH EDGES\s*\n(.*?)(?=GRAPH GAPS|$)",
        graph_block,
        re.IGNORECASE | re.DOTALL,
    )
    if edges_block:
        for line in edges_block.group(1).splitlines():
            m = re.match(
                r"\s*(C\d+)\s*->\s*(E\d+)\s*\((\w+)\)",
                line,
                re.IGNORECASE,
            )
            if m:
                result["edges"].append(
                    (m.group(1).upper(), m.group(2).upper(), m.group(3).lower())
                )

    # ── GRAPH GAPS ────────────────────────────────────────────────
    gaps_block = re.search(
        r"GRAPH GAPS\s*\n(.+?)$",
        graph_block,
        re.IGNORECASE | re.DOTALL,
    )
    if gaps_block:
        for line in gaps_block.group(1).splitlines():
            m = re.match(r"\s*(C\d+)\s*->\s*NONE", line, re.IGNORECASE)
            if m:
                result["gaps"].append(m.group(1).upper())

    return result


def highlight_html(chunk_text: str, highlighted: list[str]) -> str:
    """Wrap highlighted sentences in <mark> tags, safely HTML-escaped."""
    escaped = html.escape(chunk_text)
    for sentence in sorted(highlighted, key=len, reverse=True):
        escaped_sentence = html.escape(sentence)
        if escaped_sentence in escaped:
            escaped = escaped.replace(
                escaped_sentence, f"<mark>{escaped_sentence}</mark>"
            )
    return escaped

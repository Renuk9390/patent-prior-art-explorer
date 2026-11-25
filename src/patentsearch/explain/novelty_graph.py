"""
Parses the NOVELTY GAP GRAPH section of an LLM response into structured
node/edge data ready to render.
"""

from __future__ import annotations

import re


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

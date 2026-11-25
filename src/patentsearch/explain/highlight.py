"""
Local, deterministic explainability: always runs, no external API.

`explain_match()` splits a matched passage into sentences, scores each
sentence's embedding similarity to the query, and returns the top-scoring
sentences as evidence. `highlight_html()` wraps those sentences in <mark>
tags for display.
"""

from __future__ import annotations

import html

from patentsearch.embedders.base import Embedder
from patentsearch.ingest.chunking import split_sentences


def explain_match(
    query: str, chunk_text: str, embedder: Embedder, top_n: int = 2
) -> dict:
    """Score each sentence in `chunk_text` against the query."""
    sentences = split_sentences(chunk_text)
    if not sentences:
        return {"sentences": [], "highlighted": []}

    query_vec = embedder.embed_query(query)
    sentence_vecs = embedder.embed_passages(sentences)
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

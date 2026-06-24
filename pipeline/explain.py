"""
Explainability layer.

Two parts, deliberately layered so the demo never depends on the
fragile one to function:

  1. `explain_match()` -- always runs. Fully local, free, and
     deterministic. Splits the matched chunk into sentences, scores
     each sentence's embedding similarity to the query using the same
     model already loaded for retrieval, and returns the top-scoring
     sentences as the "evidence" driving the match. This is what makes
     highlighting work even when the query's exact words don't appear
     anywhere in the passage.

  2. `llm_explain()` -- optional. If a GROQ_API_KEY secret is configured
     on the Space, asks a fast, free-tier Llama model for a short,
     plain-language explanation of the match, on top of the
     embedding-based evidence above. Wrapped in a broad try/except with
     a short timeout: if the key is missing, the API is slow, or the
     call fails for any reason, this returns None and the UI silently
     falls back to the embedding-based explanation alone. The demo's
     core functionality never depends on this layer succeeding.
"""

from __future__ import annotations

import html
import os

from .chunking import split_sentences
from .embeddings import embed_passages, embed_query

_GROQ_MODEL = "llama-3.1-8b-instant"
_GROQ_TIMEOUT_SECONDS = 8


def explain_match(query: str, chunk_text: str, top_n: int = 2) -> dict:
    """Score each sentence in `chunk_text` against the query.

    Returns a dict with every sentence's similarity score, plus the
    top `top_n` sentences flagged as the highlighted evidence.
    """
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
    """Best-effort, plain-language explanation via a free Groq-hosted LLM.

    Returns None (never raises) if GROQ_API_KEY isn't configured or the
    call fails for any reason -- this layer is a nice-to-have polish on
    top of `explain_match()`, never a dependency for the demo to work.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        from groq import Groq

        client = Groq(api_key=api_key, timeout=_GROQ_TIMEOUT_SECONDS)
        evidence = " ".join(highlighted) if highlighted else chunk_text[:300]
        prompt = (
    "You are assisting a patent attorney performing prior-art analysis.\n\n"
    
    "You are given:\n"
    "1. A patent search query, invention concept, or claim-related search.\n"
    "2. A retrieved prior-art passage.\n"
    "3. Evidence sentences already identified as highly relevant.\n\n"

    "Your task is to:\n\n"

    "A. MATCHING TECHNICAL CONCEPTS\n"
    "- Identify the key technical concepts, components, functions, "
    "relationships, or mechanisms in the passage that are semantically "
    "related to the search query.\n\n"

    "B. POTENTIAL CLAIM ELEMENTS SUPPORTED\n"
    "- Extract technical features or claim-like limitations that appear "
    "to be disclosed in the passage.\n"
    "- Focus on structures, components, functions, processes, "
    "relationships, or technical effects.\n"
    "- Keep each element concise.\n\n"

    "C. RELEVANCE EXPLANATION\n"
    "- Explain in 1-2 concise sentences why the passage may be relevant "
    "for prior-art assessment.\n"
    "- Mention conceptual overlap even if exact wording differs.\n\n"

    "Rules:\n"
    "- Do NOT make legal conclusions.\n"
    "- Do NOT state that a claim is novel, anticipated, patentable, "
    "or invalid.\n"
    "- Do NOT invent features that are not reasonably supported by the passage.\n"
    "- If support appears weak or indirect, explicitly state that.\n"
    "- Focus only on the provided evidence passage.\n\n"

    "Return ONLY in the following format:\n\n"

    "MATCHING CONCEPTS:\n"
    "- concept 1\n"
    "- concept 2\n\n"

    "POTENTIAL SUPPORTED ELEMENTS:\n"
    "- element 1\n"
    "- element 2\n\n"

    "RELEVANCE:\n"
    "Short explanation.\n\n"

    f"Search Query:\n{query}\n\n"
    f"Evidence Passage:\n{evidence}\n"
)
        response = client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.2,
        )
        text = response.choices[0].message.content
        return text.strip() if text else None
    except Exception:
        # Network error, missing/invalid key, rate limit, SDK not
        # installed, malformed response -- any failure here should be
        # invisible to the visitor, not a broken demo.
        return None


def highlight_html(chunk_text: str, highlighted: list[str]) -> str:
    """Wrap highlighted sentences in <mark> tags, safely.

    Escapes the base text and the sentences being matched against it
    *before* comparing/inserting markup, so a chunk of text containing
    raw `<`, `>`, or `&` characters (which can happen with PDF-extracted
    text) can never break the surrounding HTML.
    """
    escaped = html.escape(chunk_text)
    for sentence in sorted(highlighted, key=len, reverse=True):
        escaped_sentence = html.escape(sentence)
        if escaped_sentence in escaped:
            escaped = escaped.replace(
                escaped_sentence, f"<mark>{escaped_sentence}</mark>"
            )
    return escaped

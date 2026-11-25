"""
Optional LLM layer: structured prior-art analysis via a free Groq-hosted model.

If GROQ_API_KEY is configured, `llm_explain()` asks the model for matching
concepts, supported query elements, a relevance explanation and a Novelty
Gap Graph. Returns None (never raises) on any failure, so the rest of the
app keeps working without it.
"""

from __future__ import annotations

import os

# Primary model can be overridden without a code change by setting a
# GROQ_MODEL variable in the Space settings. Fallbacks are tried in order
# if the primary is retired or unavailable (Groq rotates models over time).
_GROQ_MODELS = [
    m for m in (
        os.environ.get("GROQ_MODEL"),
        "openai/gpt-oss-20b",    # Groq's recommended replacement for llama-3.1-8b-instant
        "openai/gpt-oss-120b",   # larger fallback
    ) if m
]

# Model-name fragments that are NOT general chat/text models; skipped
# during automatic discovery.
_NON_CHAT_HINTS = ("whisper", "guard", "safeguard", "tts", "orpheus",
                   "playai", "embed", "vision-preview", "distil")
# Last failure reason, surfaced in the UI so a silent outage is visible.
LAST_LLM_ERROR: str | None = None
_GROQ_TIMEOUT_SECONDS = 20


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
        tried: set[str] = set()

        def attempt(model_name: str):
            """One call; reasoning models get a low effort + larger budget so
            hidden reasoning tokens don't consume the whole answer."""
            kwargs = dict(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.2,
            )
            if "gpt-oss" in model_name:
                kwargs["extra_body"] = {"reasoning_effort": "low"}
            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content
            return text.strip() if text else None

        def try_models(names):
            nonlocal last_exc
            for model_name in names:
                if model_name in tried:
                    continue
                tried.add(model_name)
                try:
                    result = attempt(model_name)
                    if result:
                        return result
                    print(f"[llm_explain] model={model_name} returned empty text", flush=True)
                except Exception as exc:
                    last_exc = exc
                    print(f"[llm_explain] model={model_name} failed: "
                          f"{type(exc).__name__}: {exc}", flush=True)
            return None

        result = try_models(_GROQ_MODELS)
        if result is None:
            # Self-healing: Groq retires models regularly. Ask the API which
            # models this key can actually use and try those instead.
            try:
                available = [m.id for m in client.models.list().data]
                print(f"[llm_explain] models available to this key: {available}", flush=True)
                candidates = [
                    m for m in available
                    if not any(h in m.lower() for h in _NON_CHAT_HINTS)
                ]
                candidates.sort(key=lambda m: (0 if "gpt-oss" in m else 1, m))
                result = try_models(candidates[:3])
            except Exception as exc:
                last_exc = last_exc or exc
                print(f"[llm_explain] model discovery failed: "
                      f"{type(exc).__name__}: {exc}", flush=True)

        if result:
            LAST_LLM_ERROR = None
            return result
        LAST_LLM_ERROR = f"{type(last_exc).__name__}: {last_exc}" if last_exc else "empty response"
        return None
    except Exception as exc:
        LAST_LLM_ERROR = f"{type(exc).__name__}: {exc}"
        print(f"[llm_explain] unexpected failure: {LAST_LLM_ERROR}", flush=True)
        return None

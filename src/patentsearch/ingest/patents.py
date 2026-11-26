"""
Turn a patent's abstract and claims into searchable, labelled passages.

Passages per patent:
  - the abstract                       -> label "Abstract"
  - each independent claim             -> label "Claim 1" (long claims are split:
                                          "Claim 1 (part 2)")
  - consecutive dependent claims, grouped to about `group_words` words
                                       -> label "Claims 2-5" (split into parts if
                                          longer than `max_words`)

Labels let a result say exactly where it matched, the way a patent
attorney cites prior art.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A claim starts with its number at the beginning of the text, of a line,
# or right after a sentence ends, followed by a capital letter or "(":
# "1. A method ..." / "...matrix. 2. The system ..." / "4. (canceled)"
_CLAIM_START = re.compile(r"(?:^|\n|(?<=[.;:])\s)\s*(\d{1,3})\s*\.\s+(?=[A-Z(])")

# Phrases that make a claim depend on an earlier one.
_DEPENDENCY = re.compile(
    r"\b(?:of|in|to|by|with|under|from|using|per)\s+(?:any\s+(?:one\s+)?of\s+)?claims?\s+\d+"
    r"|\b(?:according\s+to|as\s+(?:claimed|recited|defined|set\s+forth)\s+in)\s+(?:any\s+(?:one\s+)?of\s+)?claims?\s+\d+",
    re.IGNORECASE,
)
_CANCELLED = re.compile(r"^\(?\s*(?:canceled|cancelled|deleted)\s*\)?\.?$", re.IGNORECASE)
_HEADER = re.compile(r"^\s*(?:what\s+is\s+claimed\s+is|we\s+claim|i\s+claim|claims?)\s*:?\s*", re.IGNORECASE)


@dataclass
class Claim:
    number: int
    text: str
    independent: bool


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_claims(claims_text: str | None) -> list[Claim]:
    """Split a patent's full claims text into numbered claims.

    Cancelled claims are dropped. A claim is dependent if its opening
    words refer to another claim ("The method of claim 1, ...").
    """
    if not claims_text or not claims_text.strip():
        return []
    text = _HEADER.sub("", claims_text, count=1)

    starts = list(_CLAIM_START.finditer(text))
    if not starts:
        body = _clean(text)
        return [Claim(1, body, True)] if body else []

    claims: list[Claim] = []
    expected = 1
    for i, match in enumerate(starts):
        number = int(match.group(1))
        # Accept only the next claim number, so "... step 3. The ..." inside
        # a claim is not mistaken for a new claim.
        if number != expected:
            continue
        end = next(
            (m.start() for m in starts[i + 1:] if int(m.group(1)) == expected + 1),
            len(text),
        )
        body = _clean(text[match.end():end])
        expected += 1
        if not body or _CANCELLED.match(body):
            continue
        head = body[:200]
        claims.append(Claim(number, body, independent=not _DEPENDENCY.search(head)))
    return claims


def _windows(words: list[str], size: int) -> list[str]:
    """Split a word list into roughly equal parts of at most about `size` words."""
    if len(words) <= size:
        return [" ".join(words)]
    parts = -(-len(words) // size)  # ceiling division
    step = -(-len(words) // parts)
    return [" ".join(words[i:i + step]) for i in range(0, len(words), step)]


def build_passages(
    publication_number: str,
    abstract: str | None,
    claims_text: str | None,
    max_words: int = 250,
    group_words: int = 200,
) -> list[dict]:
    """All passages for one patent, in document order."""
    passages: list[dict] = []

    def add(section: str, label: str, text: str) -> None:
        passages.append({
            "passage_id": f"{publication_number}#{len(passages)}",
            "publication_number": publication_number,
            "section": section,
            "label": label,
            "text": text,
        })

    if abstract and abstract.strip():
        for k, part in enumerate(_windows(_clean(abstract).split(), max_words), start=1):
            add("abstract", "Abstract" if k == 1 else f"Abstract (part {k})", part)

    group: list[Claim] = []

    def flush_group() -> None:
        if not group:
            return
        first, last = group[0].number, group[-1].number
        label = f"Claim {first}" if first == last else f"Claims {first}-{last}"
        words = " ".join(f"{c.number}. {c.text}" for c in group).split()
        parts = _windows(words, max_words)   # a very long dependent claim is split too
        for k, part in enumerate(parts, start=1):
            add("dependent_claims", label + (f" (part {k})" if len(parts) > 1 else ""), part)
        group.clear()

    for claim in split_claims(claims_text):
        if claim.independent:
            flush_group()
            parts = _windows(claim.text.split(), max_words)
            for k, part in enumerate(parts, start=1):
                label = f"Claim {claim.number}" + (f" (part {k})" if len(parts) > 1 else "")
                add("independent_claim", label, part)
        else:
            group.append(claim)
            if sum(len(c.text.split()) for c in group) >= group_words:
                flush_group()
    flush_group()
    return passages


def first_independent_claim(claims_text: str | None) -> str | None:
    """The first independent claim: the usual query for a prior-art search."""
    for claim in split_claims(claims_text):
        if claim.independent:
            return claim.text
    return None

"""
Relevance labels from patent citations, and applicant (assignee) matching.

Google Patents records who added each citation in `citation.category`:
"SEA" (found in the examiner's search), "APP" (cited by the applicant),
combinations such as "APP,SEA", and rarer codes (PRS, TPO). A query may
cite several publications of the same cited family with different codes,
so labels are decided per (query, cited family) from all its codes.
"""

from __future__ import annotations

import re
from collections.abc import Iterable


def citation_source(categories: Iterable[str | None]) -> str:
    """'examiner', 'applicant', 'both' or 'other' for one (query, cited family)."""
    codes = set()
    for c in categories:
        codes.update(part.strip().upper() for part in (c or "").split(","))
    examiner, applicant = "SEA" in codes, "APP" in codes
    if examiner and applicant:
        return "both"
    if examiner:
        return "examiner"
    if applicant:
        return "applicant"
    return "other"


_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
    "llc", "lp", "llp", "plc", "gmbh", "ag", "sa", "sas", "nv", "bv", "kk", "kabushiki",
    "kaisha", "the", "oy", "ab", "spa", "srl", "pte", "pty",
}


def normalize_assignee(name: str) -> str:
    """Lower-case, drop punctuation and legal-form words: 'IBM Corp.' -> 'ibm'."""
    words = re.sub(r"[^a-z0-9 ]+", " ", (name or "").lower()).split()
    return " ".join(w for w in words if w not in _SUFFIXES)


def assignee_set(assignees: str | None) -> set[str]:
    """'A Inc; B GmbH' -> {'a', 'b'} (empty names dropped)."""
    names = (normalize_assignee(a) for a in (assignees or "").split(";"))
    return {n for n in names if n}


def same_applicant(a: str | None, b: str | None) -> bool:
    """True if the two documents share at least one normalised applicant name."""
    return bool(assignee_set(a) & assignee_set(b))


def build_qrels(citations, family_of: dict, query_of_family: dict,
                corpus_doc_of_family: dict, query_applicants: dict, doc_applicants: dict):
    """Relevance labels for a test set, from family-level citations.

    citations: DataFrame with columns publication_number (citing), cited, category.
    All publications of a test case's family count, because an application
    and its granted patent can list different citations.

    Returns a DataFrame: query_id, doc_id, source, same_applicant.
    """
    c = citations.assign(
        citing_family=citations["publication_number"].map(family_of),
        cited_family=citations["cited"].map(family_of),
    )
    c = c[c["citing_family"].isin(query_of_family) & c["cited_family"].isin(corpus_doc_of_family)]
    qrels = (
        c.groupby(["citing_family", "cited_family"])["category"]
        .apply(citation_source).rename("source").reset_index()
    )
    qrels["query_id"] = qrels["citing_family"].map(query_of_family)
    qrels["doc_id"] = qrels["cited_family"].map(corpus_doc_of_family)
    qrels["same_applicant"] = [
        same_applicant(query_applicants.get(q), doc_applicants.get(d))
        for q, d in zip(qrels["query_id"], qrels["doc_id"])
    ]
    return qrels[["query_id", "doc_id", "source", "same_applicant"]].reset_index(drop=True)

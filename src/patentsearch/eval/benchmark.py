"""
Benchmark helpers: turn passage scores into patent rankings, and score a
run against the relevance labels, overall and per subset:

  all        every cited corpus patent
  examiner   cited by the examiner only (prior art the applicant missed)
  applicant  cited by the applicant only
  same_app   cited patents with the same applicant as the query
  diff_app   cited patents from other applicants

Patent rankings use MaxP: a patent scores as its best-matching passage.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from patentsearch.eval.metrics import (
    average_precision, effort_to_recall, ndcg_at_k, pres, recall_at_k, reciprocal_rank,
)


def maxp(passage_order: Sequence[int], passage_doc: Sequence[str], top_docs: int = 100):
    """Patents in order of their best passage.

    passage_order: passage indices, best first. passage_doc: patent id per index.
    Returns (doc_ids, best_passage_index_per_doc), at most `top_docs` long.
    """
    docs, best, seen = [], [], set()
    for idx in passage_order:
        d = passage_doc[idx]
        if d not in seen:
            seen.add(d)
            docs.append(d)
            best.append(int(idx))
            if len(docs) == top_docs:
                break
    return docs, best


def _subsets(group) -> dict[str, set[str]]:
    return {
        "all": set(group["doc_id"]),
        "examiner": set(group.loc[group["source"] == "examiner", "doc_id"]),
        "applicant": set(group.loc[group["source"] == "applicant", "doc_id"]),
        "same_app": set(group.loc[group["same_applicant"], "doc_id"]),
        "diff_app": set(group.loc[~group["same_applicant"], "doc_id"]),
    }


def evaluate_run(run: dict[str, list[str]], qrels, n_max: int = 100):
    """Per-query metrics for one run.

    run: query_id -> ranked patent ids (best first, at least n_max long ideally).
    qrels: DataFrame with query_id, doc_id, source, same_applicant.
    Subset columns are NaN when the query has no relevant patent in that subset.
    """
    import pandas as pd

    rows = []
    for qid, group in qrels.groupby("query_id"):
        ranked = list(run.get(qid, []))[:n_max]
        subsets = _subsets(group)
        rel = subsets["all"]
        row = {
            "query_id": qid,
            "n_relevant": len(rel),
            "R@10": recall_at_k(ranked, rel, 10),
            "R@50": recall_at_k(ranked, rel, 50),
            "R@100": recall_at_k(ranked, rel, 100),
            "MAP@100": average_precision(ranked, rel, n_max),
            "nDCG@10": ndcg_at_k(ranked, rel, 10),
            "MRR": reciprocal_rank(ranked, rel),
            "PRES@100": pres(ranked, rel, n_max),
            "effort@80": effort_to_recall(ranked, rel, 0.8, n_max),
        }
        for name in ["examiner", "applicant", "same_app", "diff_app"]:
            s = subsets[name]
            row[f"{name}_R@10"] = recall_at_k(ranked, s, 10) if s else np.nan
            row[f"{name}_R@100"] = recall_at_k(ranked, s, 100) if s else np.nan
            row[f"{name}_PRES@100"] = pres(ranked, s, n_max) if s else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(per_query) -> dict[str, float]:
    """Mean of each metric over the queries where it is defined, plus
    reading-effort summaries (share reaching 80% recall in the top 100, and
    the median number of documents read among those)."""
    out = {}
    for col in per_query.columns:
        if col in ("query_id", "n_relevant", "effort@80"):
            continue
        out[col] = float(per_query[col].mean(skipna=True))
    effort = per_query["effort@80"]
    out["effort@80_reached"] = float(effort.notna().mean())
    out["effort@80_median"] = float(effort.dropna().median()) if effort.notna().any() else np.nan
    paired = per_query.dropna(subset=["examiner_R@100", "applicant_R@100"])
    out["paired_n"] = len(paired)
    out["paired_examiner_R@100"] = float(paired["examiner_R@100"].mean()) if len(paired) else np.nan
    out["paired_applicant_R@100"] = float(paired["applicant_R@100"].mean()) if len(paired) else np.nan
    return out


def same_applicant_share(run: dict[str, list[str]], query_applicants: dict[str, set],
                         doc_applicants: dict[str, set], k: int = 10) -> float:
    """Average share of the top-k results that share an applicant with the query."""
    shares = []
    for qid, ranked in run.items():
        qa = query_applicants.get(qid, set())
        top = ranked[:k]
        if not qa or not top:
            continue
        shares.append(sum(bool(qa & doc_applicants.get(d, set())) for d in top) / len(top))
    return float(np.mean(shares)) if shares else np.nan

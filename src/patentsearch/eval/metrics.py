"""
Ranking metrics for patent prior-art search.

Every function takes `ranked`: document ids best-first (one per patent),
and `relevant`: the set of relevant ids for the query. Standard metrics
follow their usual definitions; PRES follows Magdy & Jones (2010),
"PRES: a score metric for evaluating recall-oriented information
retrieval applications", SIGIR 2010.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / k


def reciprocal_rank(ranked: Sequence[str], relevant: set[str], k: int | None = None) -> float:
    for i, doc in enumerate(ranked[:k], start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def average_precision(ranked: Sequence[str], relevant: set[str], k: int | None = None) -> float:
    """AP over the top-k list; relevant documents not retrieved count as zero."""
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for i, doc in enumerate(ranked[:k], start=1):
        if doc in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant)


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG@k."""
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def pres(ranked: Sequence[str], relevant: set[str], n_max: int = 100) -> float:
    """Patent Retrieval Evaluation Score (Magdy & Jones, 2010).

    PRES = 1 - (mean(r_i) - (n + 1) / 2) / n_max, where r_i are the ranks of
    the n relevant documents. Relevant documents not found in the top n_max
    are placed just after the cut-off (ranks n_max + f + 1 ... n_max + n,
    f = number found), so PRES ranges from 0 to the recall at n_max.
    """
    n = len(relevant)
    if n == 0:
        return 0.0
    found = [i for i, d in enumerate(ranked[:n_max], start=1) if d in relevant]
    f = len(found)
    missing = [n_max + j for j in range(f + 1, n + 1)]
    mean_rank = (sum(found) + sum(missing)) / n
    return 1.0 - (mean_rank - (n + 1) / 2) / n_max


def effort_to_recall(
    ranked: Sequence[str], relevant: set[str], target: float = 0.8, n_max: int | None = None
) -> int | None:
    """Reading effort: how many documents a searcher must read, top-down,
    to find `target` (e.g. 80%) of the relevant documents.

    Returns None if the target is not reached within the list (or n_max).
    """
    if not relevant:
        return None
    needed = math.ceil(target * len(relevant) - 1e-9)
    hits = 0
    for i, doc in enumerate(ranked[:n_max], start=1):
        if doc in relevant:
            hits += 1
            if hits >= needed:
                return i
    return None

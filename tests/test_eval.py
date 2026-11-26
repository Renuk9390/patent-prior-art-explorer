import math

import pytest

from patentsearch.eval.metrics import (
    average_precision, effort_to_recall, ndcg_at_k, pres, precision_at_k,
    recall_at_k, reciprocal_rank,
)
from patentsearch.eval.qrels import assignee_set, citation_source, same_applicant

RANKED = ["a", "x", "b", "y", "z"]
REL = {"a", "b", "c"}


def test_recall_precision_rr():
    assert recall_at_k(RANKED, REL, 3) == pytest.approx(2 / 3)
    assert precision_at_k(RANKED, REL, 2) == 0.5
    assert reciprocal_rank(["x", "b"], REL) == 0.5
    assert reciprocal_rank(["x", "y"], REL) == 0.0


def test_average_precision_counts_missing_as_zero():
    assert average_precision(RANKED, REL) == pytest.approx((1 / 1 + 2 / 3) / 3)


def test_ndcg_binary():
    ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    got = 1 + 1 / math.log2(4)
    assert ndcg_at_k(RANKED, REL, 5) == pytest.approx(got / ideal)
    assert ndcg_at_k(["a", "b", "c"], REL, 3) == pytest.approx(1.0)


def test_pres_best_worst_and_bounds():
    assert pres(["a", "b", "c"], REL, n_max=100) == pytest.approx(1.0)      # best case
    assert pres(["x"] * 100, REL, n_max=100) == pytest.approx(0.0)          # none found
    # all found at the very top except one missing: PRES equals recall (2/3)
    assert pres(["a", "b"], REL, n_max=100) == pytest.approx(2 / 3)
    # finding documents later lowers PRES
    late = ["x"] * 50 + ["a", "b", "c"]
    assert 0 < pres(late, REL, n_max=100) < 1


def test_effort_to_recall():
    # 80% of 3 relevant -> need 3 (ceil 2.4): all three
    assert effort_to_recall(["a", "x", "b", "c"], REL, 0.8) == 4
    assert effort_to_recall(["a", "x", "b"], REL, 0.5) == 3      # need 2
    assert effort_to_recall(["a", "x"], REL, 0.8) is None
    assert effort_to_recall(["a", "b", "c"], REL, 0.8, n_max=2) is None


def test_citation_source_labels():
    assert citation_source(["SEA"]) == "examiner"
    assert citation_source(["APP"]) == "applicant"
    assert citation_source(["APP,SEA"]) == "both"
    assert citation_source(["APP", "PRS,SEA"]) == "both"        # two publications, one family
    assert citation_source(["PRS"]) == "other"
    assert citation_source([None]) == "other"


def test_applicant_matching():
    assert assignee_set("International Business Machines Corp; IBM UK Ltd") == {
        "international business machines", "ibm uk"}
    assert same_applicant("Google LLC", "GOOGLE INC.")
    assert not same_applicant("Google LLC", "Microsoft Technology Licensing, LLC")
    assert not same_applicant("", None)


def test_build_qrels_family_level():
    import pandas as pd
    from patentsearch.eval.qrels import build_qrels

    cit = pd.DataFrame({
        "publication_number": ["Q-A1", "Q-B2", "Q-B2", "R-A1", "R-A1"],
        "cited":              ["D1-B2", "D1-B2", "D2-B2", "D3-B2", "EP-X"],
        "category":           ["APP", "SEA", "APP", "SEA", "SEA"],
    })
    family_of = {"Q-A1": "q", "Q-B2": "q", "R-A1": "r", "D1-B2": "d1", "D2-B2": "d2", "D3-B2": "d3"}
    qrels = build_qrels(
        cit, family_of,
        query_of_family={"q": "Q-A1", "r": "R-A1"},
        corpus_doc_of_family={"d1": "D1-B2", "d2": "D2-B2", "d3": "D3-B2"},
        query_applicants={"Q-A1": "Google LLC", "R-A1": "Acme"},
        doc_applicants={"D1-B2": "GOOGLE INC", "D2-B2": "IBM", "D3-B2": "Other"},
    )
    rows = {(r.query_id, r.doc_id): (r.source, r.same_applicant) for r in qrels.itertuples()}
    assert rows == {
        ("Q-A1", "D1-B2"): ("both", True),        # APP on the application + SEA on the grant
        ("Q-A1", "D2-B2"): ("applicant", False),
        ("R-A1", "D3-B2"): ("examiner", False),   # EP-X is outside the corpus: ignored
    }

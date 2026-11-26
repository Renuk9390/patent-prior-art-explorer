import math

import pandas as pd
import pytest

from patentsearch.eval.benchmark import evaluate_run, maxp, same_applicant_share, summarize


def test_maxp_keeps_best_passage_per_patent():
    passage_doc = ["A", "A", "B", "C", "B"]
    docs, best = maxp([1, 0, 4, 3, 2], passage_doc, top_docs=10)
    assert docs == ["A", "B", "C"]
    assert best == [1, 4, 3]
    assert maxp([1, 0, 4, 3, 2], passage_doc, top_docs=2)[0] == ["A", "B"]


QRELS = pd.DataFrame({
    "query_id":       ["q1", "q1", "q1", "q2"],
    "doc_id":         ["A", "B", "C", "D"],
    "source":         ["examiner", "applicant", "both", "examiner"],
    "same_applicant": [False, True, False, False],
})


def test_evaluate_run_overall_and_subsets():
    run = {"q1": ["A", "x", "B"], "q2": ["y"]}
    pq = evaluate_run(run, QRELS).set_index("query_id")
    assert pq.loc["q1", "R@10"] == pytest.approx(2 / 3)
    assert pq.loc["q1", "examiner_R@10"] == 1.0
    assert pq.loc["q1", "applicant_R@10"] == 1.0
    assert pq.loc["q1", "same_app_R@10"] == 1.0        # B
    assert pq.loc["q1", "diff_app_R@10"] == 0.5        # A found, C not
    assert pq.loc["q2", "R@100"] == 0.0
    assert math.isnan(pq.loc["q2", "applicant_R@10"])  # q2 has no applicant citation
    assert pq.loc["q2", "PRES@100"] == pytest.approx(0.0)


def test_summarize_paired_and_effort():
    run = {"q1": ["A", "B", "C"], "q2": ["D"]}
    s = summarize(evaluate_run(run, QRELS))
    assert s["R@10"] == 1.0
    assert s["paired_n"] == 1                        # only q1 has both kinds
    assert s["effort@80_reached"] == 1.0
    assert s["effort@80_median"] == 2.0              # q1 needs 3 docs, q2 needs 1


def test_same_applicant_share():
    run = {"q1": ["A", "B"], "q2": ["D", "E"]}
    qa = {"q1": {"acme"}, "q2": {"ibm"}}
    da = {"A": {"acme"}, "B": {"other"}, "D": {"other"}, "E": {"ibm"}}
    assert same_applicant_share(run, qa, da, k=2) == pytest.approx(0.5)

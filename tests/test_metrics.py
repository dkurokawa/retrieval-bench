from __future__ import annotations

import pytest

from retrieval_bench.metrics import dcg_at_k, evaluate, mrr_at_k, ndcg_at_k, recall_at_k

# Fixture: ranking = ["a", "b", "c", "d"], qrels = {"a": 0, "b": 2, "c": 1}
# ("a" judged non-relevant, "d" unjudged). Reference values computed
# independently (see the derivation in scratchpad, reproduced here):
#   gains        = [0, 2, 1, 0]
#   dcg@10       = 0/log2(2) + 2/log2(3) + 1/log2(4) + 0/log2(5) = 1.761859507142915
#   idcg@10      = 2/log2(2) + 1/log2(3)                        = 2.6309297535714578
#   ndcg@10      = dcg/idcg                                     = 0.66967181649423
#   recall@10    = 2/2                                          = 1.0
#   recall@1     = 0/2                                          = 0.0
#   mrr@10       = 1/2 (first relevant doc "b" is at rank 2)    = 0.5

RANKING = ["a", "b", "c", "d"]
QRELS = {"a": 0, "b": 2, "c": 1}


def test_recall_at_k_counts_only_positively_relevant_docs() -> None:
    assert recall_at_k(RANKING, QRELS, k=10) == pytest.approx(1.0)
    assert recall_at_k(RANKING, QRELS, k=1) == pytest.approx(0.0)


def test_recall_at_k_returns_none_when_no_relevant_docs() -> None:
    assert recall_at_k(RANKING, {"a": 0}, k=10) is None
    assert recall_at_k(RANKING, {}, k=10) is None


def test_dcg_at_k_matches_hand_computation() -> None:
    assert dcg_at_k([0, 2, 1, 0], k=10) == pytest.approx(1.761859507142915)


def test_ndcg_at_k_matches_hand_computed_value() -> None:
    assert ndcg_at_k(RANKING, QRELS, k=10) == pytest.approx(0.66967181649423)


def test_ndcg_at_k_is_one_for_the_ideal_ranking() -> None:
    qrels = {"a": 2, "b": 1, "c": 0}
    assert ndcg_at_k(["a", "b", "c"], qrels, k=10) == pytest.approx(1.0)


def test_ndcg_at_k_is_zero_when_ranking_has_no_relevant_docs_in_window() -> None:
    qrels = {"z": 1}
    assert ndcg_at_k(["a", "b", "c"], qrels, k=10) == pytest.approx(0.0)


def test_ndcg_at_k_returns_none_when_no_relevant_docs() -> None:
    assert ndcg_at_k(RANKING, {"a": 0}, k=10) is None


def test_mrr_at_k_matches_hand_computed_value() -> None:
    assert mrr_at_k(RANKING, QRELS, k=10) == pytest.approx(0.5)


def test_mrr_at_k_is_zero_when_relevant_doc_is_outside_window() -> None:
    assert mrr_at_k(RANKING, QRELS, k=1) == pytest.approx(0.0)


def test_mrr_at_k_returns_none_when_no_relevant_docs() -> None:
    assert mrr_at_k(RANKING, {"a": 0}, k=10) is None


def test_evaluate_excludes_queries_with_no_relevant_docs_from_the_mean() -> None:
    run = {"q1": RANKING, "q2": ["a", "b"]}
    qrels = {"q1": QRELS, "q2": {"a": 0}}  # q2 has no positively-relevant doc.

    result = evaluate(run, qrels)

    assert "q2" not in result.per_query
    assert result.per_query["q1"]["ndcg@10"] == pytest.approx(0.66967181649423)
    # The mean over recall@10/ndcg@10/mrr@10 is over q1 alone.
    assert result.mean["recall@10"] == pytest.approx(1.0)
    assert result.mean["mrr@10"] == pytest.approx(0.5)


def test_evaluate_handles_a_query_missing_from_qrels_entirely() -> None:
    run = {"q1": RANKING}
    result = evaluate(run, qrels={})
    assert result.per_query == {}
    assert all(v == 0.0 for v in result.mean.values())


def test_evaluate_returns_all_four_metric_names() -> None:
    result = evaluate({"q1": RANKING}, {"q1": QRELS})
    assert set(result.mean) == {"recall@10", "recall@100", "ndcg@10", "mrr@10"}

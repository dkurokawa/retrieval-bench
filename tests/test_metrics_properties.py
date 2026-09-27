"""Property-based tests for the metric implementations, using hypothesis.

These check invariants that must hold for *any* input, complementing the
hand-computed examples in test_metrics.py:
- a perfectly-ordered ranking always scores nDCG == 1.0
- a ranking with no overlap with the relevant set always scores 0.0
- moving a more-relevant document earlier never decreases nDCG
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from retrieval_bench.metrics import mrr_at_k, ndcg_at_k, recall_at_k


@st.composite
def _ranked_qrels(draw: st.DrawFn) -> tuple[list[str], dict[str, int]]:
    n = draw(st.integers(min_value=2, max_value=8))
    doc_ids = [f"d{i}" for i in range(n)]
    relevances = draw(st.lists(st.integers(min_value=0, max_value=3), min_size=n, max_size=n))
    return doc_ids, dict(zip(doc_ids, relevances, strict=True))


@given(_ranked_qrels())
def test_perfect_ranking_has_ndcg_one(scenario: tuple[list[str], dict[str, int]]) -> None:
    doc_ids, qrels = scenario
    assume(any(r > 0 for r in qrels.values()))

    ideal_ranking = sorted(doc_ids, key=lambda d: (-qrels[d], d))
    ndcg = ndcg_at_k(ideal_ranking, qrels, k=len(ideal_ranking))
    assert ndcg == pytest.approx(1.0)


@given(st.integers(min_value=1, max_value=5), st.integers(min_value=1, max_value=5))
def test_disjoint_ranking_scores_zero_on_every_metric(n_relevant: int, n_ranking: int) -> None:
    relevant_ids = [f"r{i}" for i in range(n_relevant)]
    ranking_ids = [f"x{i}" for i in range(n_ranking)]
    qrels = dict.fromkeys(relevant_ids, 1)

    assert recall_at_k(ranking_ids, qrels, k=len(ranking_ids)) == pytest.approx(0.0)
    assert ndcg_at_k(ranking_ids, qrels, k=len(ranking_ids)) == pytest.approx(0.0)
    assert mrr_at_k(ranking_ids, qrels, k=len(ranking_ids)) == pytest.approx(0.0)


@given(
    _ranked_qrels(),
    st.data(),
)
def test_moving_a_more_relevant_doc_earlier_never_decreases_ndcg(
    scenario: tuple[list[str], dict[str, int]], data: st.DataObject
) -> None:
    doc_ids, qrels = scenario
    assume(any(r > 0 for r in qrels.values()))

    ranking = data.draw(st.permutations(doc_ids))
    idx = data.draw(st.integers(min_value=0, max_value=len(ranking) - 2))
    a, b = ranking[idx], ranking[idx + 1]
    assume(qrels[a] != qrels[b])

    ndcg_before = ndcg_at_k(ranking, qrels, k=len(ranking))
    swapped = list(ranking)
    swapped[idx], swapped[idx + 1] = swapped[idx + 1], swapped[idx]
    ndcg_after = ndcg_at_k(swapped, qrels, k=len(ranking))

    assert ndcg_before is not None
    assert ndcg_after is not None
    if qrels[b] > qrels[a]:
        # b (the more relevant doc) moved one position earlier.
        assert ndcg_after >= ndcg_before - 1e-9
    else:
        assert ndcg_after <= ndcg_before + 1e-9

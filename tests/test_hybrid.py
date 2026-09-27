from __future__ import annotations

from collections.abc import Sequence

import pytest

from retrieval_bench.chunking import Chunk
from retrieval_bench.retrievers.hybrid import HybridRetriever


class FixedRankingRetriever:
    """A stub retriever that always returns a pre-set ranking, for testing RRF."""

    def __init__(self, ranking: list[tuple[str, float]]) -> None:
        self._ranking = ranking
        self.indexed_doc_ids: set[str] | None = None

    def index(self, chunks: Sequence[Chunk]) -> None:
        self.indexed_doc_ids = {c.doc_id for c in chunks}

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        return self._ranking[:k]


def three_doc_chunks() -> list[Chunk]:
    return [
        Chunk(doc_id="d1", chunk_index=0, text="x"),
        Chunk(doc_id="d2", chunk_index=0, text="x"),
        Chunk(doc_id="d3", chunk_index=0, text="x"),
    ]


def test_rrf_fusion_matches_hand_computed_scores() -> None:
    bm25 = FixedRankingRetriever([("d1", 5.0), ("d2", 3.0), ("d3", 1.0)])
    dense = FixedRankingRetriever([("d2", 0.9), ("d3", 0.8), ("d1", 0.1)])
    hybrid = HybridRetriever(bm25, dense, rrf_k=60)  # type: ignore[arg-type]
    hybrid.index(three_doc_chunks())

    results = dict(hybrid.search("q", k=3))

    # rrf(d) = 1/(60+rank_bm25(d)) + 1/(60+rank_dense(d)), ranks are 1-based.
    expected = {
        "d1": 1 / (60 + 1) + 1 / (60 + 3),
        "d2": 1 / (60 + 2) + 1 / (60 + 1),
        "d3": 1 / (60 + 3) + 1 / (60 + 2),
    }
    assert results == pytest.approx(expected)


def test_rrf_orders_by_fused_score_descending() -> None:
    bm25 = FixedRankingRetriever([("d1", 5.0), ("d2", 3.0), ("d3", 1.0)])
    dense = FixedRankingRetriever([("d2", 0.9), ("d3", 0.8), ("d1", 0.1)])
    hybrid = HybridRetriever(bm25, dense, rrf_k=60)  # type: ignore[arg-type]
    hybrid.index(three_doc_chunks())

    ranked = hybrid.search("q", k=3)
    # d2 wins: rank 2 in bm25 + rank 1 in dense beats d1 (rank 1 + rank 3) and
    # d3 (rank 3 + rank 2).
    assert [doc_id for doc_id, _ in ranked] == ["d2", "d1", "d3"]


def test_rrf_counts_only_the_retrievers_a_doc_appears_in() -> None:
    # d4 only appears in the dense ranking.
    bm25 = FixedRankingRetriever([("d1", 1.0)])
    dense = FixedRankingRetriever([("d4", 1.0), ("d1", 0.5)])
    hybrid = HybridRetriever(bm25, dense, rrf_k=60)  # type: ignore[arg-type]
    hybrid.index(
        [
            Chunk(doc_id="d1", chunk_index=0, text="x"),
            Chunk(doc_id="d4", chunk_index=0, text="x"),
        ]
    )

    results = dict(hybrid.search("q", k=2))
    assert results["d4"] == pytest.approx(1 / (60 + 1))
    assert results["d1"] == pytest.approx(1 / (60 + 1) + 1 / (60 + 2))


def test_index_forwards_chunks_to_both_sub_retrievers() -> None:
    bm25 = FixedRankingRetriever([])
    dense = FixedRankingRetriever([])
    hybrid = HybridRetriever(bm25, dense)  # type: ignore[arg-type]
    hybrid.index(three_doc_chunks())

    assert bm25.indexed_doc_ids == {"d1", "d2", "d3"}
    assert dense.indexed_doc_ids == {"d1", "d2", "d3"}


def test_hybrid_only_fuses_each_retrievers_top_depth_candidates() -> None:
    # 5 candidates on each side, but depth=2: only ranks 1-2 from each side
    # are eligible to contribute. "tail" appears at rank 5 on both sides, so
    # with full-corpus fusion it would still show up (weakly); with
    # depth-limited fusion it must not appear at all.
    bm25 = FixedRankingRetriever([("a", 5.0), ("b", 4.0), ("c", 3.0), ("d", 2.0), ("tail", 1.0)])
    dense = FixedRankingRetriever([("b", 0.9), ("a", 0.8), ("e", 0.7), ("f", 0.6), ("tail", 0.5)])
    hybrid = HybridRetriever(bm25, dense, rrf_k=60, depth=2)  # type: ignore[arg-type]
    hybrid.index([Chunk(doc_id=f"d{i}", chunk_index=0, text="x") for i in range(5)])

    results = dict(hybrid.search("q", k=10))

    assert "tail" not in results
    assert set(results) == {"a", "b"}
    # a: bm25 rank 1, dense rank 2. b: bm25 rank 2, dense rank 1.
    assert results["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert results["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_hybrid_depth_does_not_limit_the_number_of_results_returned() -> None:
    # A doc within depth on just one side, with nothing beyond depth on the
    # other, must still come back if k allows it.
    bm25 = FixedRankingRetriever([("a", 1.0), ("b", 1.0)])
    dense = FixedRankingRetriever([("c", 1.0), ("d", 1.0)])
    hybrid = HybridRetriever(bm25, dense, rrf_k=60, depth=2)  # type: ignore[arg-type]
    hybrid.index([Chunk(doc_id=f"d{i}", chunk_index=0, text="x") for i in range(4)])

    results = hybrid.search("q", k=10)
    assert {doc_id for doc_id, _ in results} == {"a", "b", "c", "d"}


@pytest.mark.parametrize("rrf_k", [0, -1])
def test_hybrid_rejects_non_positive_rrf_k(rrf_k: int) -> None:
    bm25 = FixedRankingRetriever([])
    dense = FixedRankingRetriever([])
    with pytest.raises(ValueError, match="rrf_k"):
        HybridRetriever(bm25, dense, rrf_k=rrf_k)  # type: ignore[arg-type]


@pytest.mark.parametrize("depth", [0, -1])
def test_hybrid_rejects_non_positive_depth(depth: int) -> None:
    bm25 = FixedRankingRetriever([])
    dense = FixedRankingRetriever([])
    with pytest.raises(ValueError, match="depth"):
        HybridRetriever(bm25, dense, depth=depth)  # type: ignore[arg-type]


def test_hybrid_ranking_is_not_driven_by_doc_id_lexical_order() -> None:
    # The best-fused document has the lexically *last* id; a bug that let
    # doc_id ordering leak into ranking (instead of only breaking exact score
    # ties) would put it last instead of first.
    bm25 = FixedRankingRetriever([("zzz_best", 5.0), ("aaa_worst", 1.0)])
    dense = FixedRankingRetriever([("zzz_best", 0.9), ("aaa_worst", 0.1)])
    hybrid = HybridRetriever(bm25, dense, rrf_k=60)  # type: ignore[arg-type]
    hybrid.index(
        [
            Chunk(doc_id="zzz_best", chunk_index=0, text="x"),
            Chunk(doc_id="aaa_worst", chunk_index=0, text="x"),
        ]
    )

    ranked = hybrid.search("q", k=2)
    assert [doc_id for doc_id, _ in ranked] == ["zzz_best", "aaa_worst"]

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

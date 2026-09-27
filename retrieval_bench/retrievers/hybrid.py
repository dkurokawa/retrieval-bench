"""Reciprocal Rank Fusion of BM25 and dense retrieval rankings."""

from __future__ import annotations

from collections.abc import Sequence

from retrieval_bench.chunking import Chunk
from retrieval_bench.retrievers.bm25 import BM25Retriever
from retrieval_bench.retrievers.dense import DenseRetriever


class HybridRetriever:
    """Fuses BM25 and dense document rankings with Reciprocal Rank Fusion.

    score(d) = sum over retrievers r where d appears in r's top-``depth``
    ranking of 1 / (rrf_k + rank_r(d)), with rank_r(d) 1-based. Only each
    sub-retriever's top ``depth`` candidates are fused: a document that
    neither retriever surfaces within that depth gets no contribution from
    it (this does not fuse full rankings over the whole corpus).
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        rrf_k: int = 60,
        depth: int = 1000,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError(f"rrf_k must be positive, got {rrf_k}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k
        self.depth = depth

    @property
    def encoder_identifier(self) -> str | None:
        """Passthrough to the dense sub-retriever's encoder identifier, if any."""
        return getattr(self.dense, "encoder_identifier", None)

    def index(self, chunks: Sequence[Chunk]) -> None:
        chunk_list = list(chunks)
        self.bm25.index(chunk_list)
        self.dense.index(chunk_list)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        bm25_ranked = self.bm25.search(query, self.depth)
        dense_ranked = self.dense.search(query, self.depth)

        scores: dict[str, float] = {}
        for rank, (doc_id, _) in enumerate(bm25_ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, (doc_id, _) in enumerate(dense_ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:k]

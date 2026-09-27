"""Reciprocal Rank Fusion of BM25 and dense retrieval rankings."""

from __future__ import annotations

from collections.abc import Sequence

from retrieval_bench.chunking import Chunk
from retrieval_bench.retrievers.bm25 import BM25Retriever
from retrieval_bench.retrievers.dense import DenseRetriever


class HybridRetriever:
    """Fuses BM25 and dense document rankings with Reciprocal Rank Fusion.

    score(d) = sum over retrievers r where d appears in r's ranking of
    1 / (rrf_k + rank_r(d)), with rank_r(d) 1-based.
    """

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, rrf_k: int = 60) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k
        self._n_docs = 0

    def index(self, chunks: Sequence[Chunk]) -> None:
        chunk_list = list(chunks)
        self.bm25.index(chunk_list)
        self.dense.index(chunk_list)
        self._n_docs = len({c.doc_id for c in chunk_list})

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        pool = max(self._n_docs, k)
        bm25_ranked = self.bm25.search(query, pool)
        dense_ranked = self.dense.search(query, pool)

        scores: dict[str, float] = {}
        for rank, (doc_id, _) in enumerate(bm25_ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, (doc_id, _) in enumerate(dense_ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:k]

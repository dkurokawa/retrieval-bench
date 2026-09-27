"""Shared retriever protocol and chunk-to-document score aggregation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from retrieval_bench.chunking import Chunk


class Retriever(Protocol):
    """Common interface implemented by BM25, dense, and hybrid retrievers."""

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Build (or rebuild) the retriever's index over ``chunks``."""
        ...

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(doc_id, score)`` pairs, best first."""
        ...


def aggregate_max_by_doc(
    chunk_scores: Sequence[tuple[Chunk, float]],
) -> list[tuple[str, float]]:
    """Collapse chunk-level scores to per-document scores via max-pooling.

    A document's score is the maximum score across its chunks. Results are
    sorted by descending score, then ascending ``doc_id`` for determinism.
    """
    best: dict[str, float] = {}
    for c, score in chunk_scores:
        current = best.get(c.doc_id)
        if current is None or score > current:
            best[c.doc_id] = score
    return sorted(best.items(), key=lambda item: (-item[1], item[0]))

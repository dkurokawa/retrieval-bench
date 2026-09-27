"""Retrieval quality metrics: recall@k, nDCG@k, MRR@k.

All metrics treat qrels entries with relevance <= 0 as "judged not relevant"
(never as a hit), and exclude a query from a metric's average entirely if it
has no positively-relevant document in the qrels (there is nothing to
recall/rank correctly, so the metric is undefined rather than 0 for it).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

Qrels = dict[str, int]


def _positive_relevant(qrels: Qrels) -> dict[str, int]:
    return {doc_id: rel for doc_id, rel in qrels.items() if rel > 0}


def recall_at_k(ranked_doc_ids: Sequence[str], qrels: Qrels, k: int) -> float | None:
    """Fraction of relevant documents present in the top ``k`` results."""
    relevant = _positive_relevant(qrels)
    if not relevant:
        return None
    top_k = set(ranked_doc_ids[:k])
    hits = len(top_k & relevant.keys())
    return hits / len(relevant)


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """Discounted cumulative gain over the first ``k`` gains."""
    return sum(gain / math.log2(i + 2) for i, gain in enumerate(gains[:k]))


def ndcg_at_k(ranked_doc_ids: Sequence[str], qrels: Qrels, k: int) -> float | None:
    """Normalized discounted cumulative gain, using graded relevance."""
    relevant = _positive_relevant(qrels)
    if not relevant:
        return None
    gains = [qrels.get(doc_id, 0) for doc_id in ranked_doc_ids[:k]]
    dcg = dcg_at_k(gains, k)
    ideal_gains = sorted(relevant.values(), reverse=True)
    idcg = dcg_at_k(ideal_gains, k)
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(ranked_doc_ids: Sequence[str], qrels: Qrels, k: int) -> float | None:
    """Reciprocal rank of the first relevant document within the top ``k``."""
    relevant = _positive_relevant(qrels)
    if not relevant:
        return None
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


_METRIC_FNS: dict[str, tuple[str, int]] = {
    "recall@10": ("recall", 10),
    "recall@100": ("recall", 100),
    "ndcg@10": ("ndcg", 10),
    "mrr@10": ("mrr", 10),
}

_METRIC_IMPLS = {
    "recall": recall_at_k,
    "ndcg": ndcg_at_k,
    "mrr": mrr_at_k,
}


@dataclass(frozen=True)
class MetricResult:
    """Per-query metric values plus their averages over queries with a defined value."""

    per_query: dict[str, dict[str, float]]
    mean: dict[str, float]


def evaluate(run: dict[str, list[str]], qrels: dict[str, Qrels]) -> MetricResult:
    """Compute recall@10, recall@100, nDCG@10, and MRR@10 for a run.

    ``run`` maps query_id to a ranked list of doc_ids. ``qrels`` maps query_id
    to a mapping of doc_id to graded relevance.
    """
    per_query: dict[str, dict[str, float]] = {}
    sums: dict[str, float] = dict.fromkeys(_METRIC_FNS, 0.0)
    counts: dict[str, int] = dict.fromkeys(_METRIC_FNS, 0)

    for query_id, ranked_doc_ids in run.items():
        query_qrels = qrels.get(query_id, {})
        values: dict[str, float] = {}
        for metric_name, (kind, k) in _METRIC_FNS.items():
            value = _METRIC_IMPLS[kind](ranked_doc_ids, query_qrels, k)
            if value is not None:
                values[metric_name] = value
                sums[metric_name] += value
                counts[metric_name] += 1
        if values:
            per_query[query_id] = values

    mean = {
        metric_name: (sums[metric_name] / counts[metric_name] if counts[metric_name] else 0.0)
        for metric_name in _METRIC_FNS
    }
    return MetricResult(per_query=per_query, mean=mean)

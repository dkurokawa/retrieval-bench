"""Okapi BM25 retrieval over word-tokenized chunks, via scipy.sparse."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

import numpy as np
from scipy import sparse

from retrieval_bench.chunking import Chunk
from retrieval_bench.retrievers.base import aggregate_max_by_doc

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A short, generic English stopword list; opt-in via BM25Retriever(use_stopwords=True).
DEFAULT_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
    }
)


def tokenize(text: str, stopwords: frozenset[str] | None = None) -> list[str]:
    """Lowercase and split into alphanumeric tokens, optionally dropping stopwords."""
    tokens = _TOKEN_RE.findall(text.lower())
    if stopwords:
        return [t for t in tokens if t not in stopwords]
    return tokens


class BM25Retriever:
    """Okapi BM25 with the standard non-negative idf variant.

    score(q, d) = sum_t idf(t) * tf(t,d) * (k1+1) / (tf(t,d) + k1*(1-b+b*|d|/avgdl))
    idf(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
    """

    def __init__(self, k1: float = 0.9, b: float = 0.4, use_stopwords: bool = False) -> None:
        if k1 < 0:
            raise ValueError(f"k1 must be >= 0, got {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b must be between 0 and 1, got {b}")
        self.k1 = k1
        self.b = b
        self._stopwords = DEFAULT_STOPWORDS if use_stopwords else None
        self._chunks: list[Chunk] = []
        self._vocab: dict[str, int] = {}
        self._tf: sparse.csr_matrix | None = None
        self._idf: np.ndarray | None = None
        self._doc_len: np.ndarray | None = None
        self._avgdl: float = 0.0

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._vocab = {}
        rows: list[int] = []
        cols: list[int] = []
        data: list[int] = []
        doc_len = np.zeros(len(self._chunks), dtype=np.float64)

        for i, c in enumerate(self._chunks):
            tokens = tokenize(c.text, self._stopwords)
            doc_len[i] = len(tokens)
            for token, count in Counter(tokens).items():
                col = self._vocab.setdefault(token, len(self._vocab))
                rows.append(i)
                cols.append(col)
                data.append(count)

        n_chunks = len(self._chunks)
        n_vocab = len(self._vocab)
        self._tf = sparse.csr_matrix(
            (data, (rows, cols)), shape=(n_chunks, n_vocab), dtype=np.float64
        )
        self._doc_len = doc_len
        self._avgdl = float(doc_len.mean()) if n_chunks else 0.0

        df = np.asarray((self._tf > 0).sum(axis=0)).ravel()
        self._idf = np.log(1.0 + (n_chunks - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(doc_id, score)`` pairs with score > 0.

        Documents with no matching query term (score 0) are never returned:
        a BM25 score of 0 means "no evidence this document is relevant to
        this query," which is a different thing from "known relevant with a
        low score," and callers (recall/nDCG/hybrid fusion) should not treat
        it as a ranked candidate.
        """
        if self._tf is None or self._idf is None or self._doc_len is None:
            raise RuntimeError("BM25Retriever.search called before index()")
        if not self._chunks:
            return []

        tokens = tokenize(query, self._stopwords)
        scores = np.zeros(len(self._chunks), dtype=np.float64)
        if self._avgdl > 0:
            norm = 1.0 - self.b + self.b * self._doc_len / self._avgdl
        else:
            norm = np.ones_like(self._doc_len)

        for token in tokens:
            col = self._vocab.get(token)
            if col is None:
                continue
            tf = np.asarray(self._tf[:, col].todense()).ravel()
            denom = tf + self.k1 * norm
            contribution = np.divide(
                tf * (self.k1 + 1.0), denom, out=np.zeros_like(tf), where=denom > 0
            )
            scores += self._idf[col] * contribution

        chunk_scores = list(zip(self._chunks, scores.tolist(), strict=True))
        doc_scores = aggregate_max_by_doc(chunk_scores)
        ranked = [(doc_id, score) for doc_id, score in doc_scores if score > 0]
        return ranked[:k]

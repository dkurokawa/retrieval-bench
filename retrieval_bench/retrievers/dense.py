"""Embedding-based dense retrieval, with disk-cached, L2-normalized vectors."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from retrieval_bench.chunking import Chunk
from retrieval_bench.datasets import cache_root
from retrieval_bench.retrievers.base import aggregate_max_by_doc

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Encoder(Protocol):
    """Turns texts into dense vectors. Implemented by SentenceTransformerEncoder,
    and by fakes in tests."""

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """Encoder backed by the optional `sentence-transformers` package."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:  # pragma: no cover
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for the default dense encoder; "
                "install it with `uv sync --extra dense`"
            ) from exc
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        # Thin adapter over an optional, heavy ML dependency; exercised only in
        # the manual runs used to fill in the README's results table, not in CI.
        embeddings = self._model.encode(list(texts), show_progress_bar=False)
        return np.asarray(embeddings, dtype=np.float64)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized: np.ndarray = vectors / norms
    return normalized


def embedding_cache_path(
    cache_dir: Path, model_name: str, chunk_params: dict[str, Any], corpus_sha256: str
) -> Path:
    """Deterministic cache path keyed by model name, chunking params, and corpus hash."""
    key_material = json.dumps(
        {"model": model_name, "chunk_params": chunk_params, "corpus_sha256": corpus_sha256},
        sort_keys=True,
    )
    key = hashlib.sha256(key_material.encode("utf-8")).hexdigest()
    return cache_dir / f"{key}.npy"


def default_embedding_cache_dir() -> Path:
    return cache_root() / "embeddings"


class DenseRetriever:
    """Dense retrieval: encode chunks, L2-normalize, rank by inner product."""

    def __init__(
        self,
        model_name: str,
        chunk_params: dict[str, Any],
        corpus_sha256: str,
        cache_dir: Path | None = None,
        encoder: Encoder | None = None,
    ) -> None:
        self.model_name = model_name
        self.chunk_params = chunk_params
        self.corpus_sha256 = corpus_sha256
        self.cache_dir = cache_dir if cache_dir is not None else default_embedding_cache_dir()
        self._encoder = encoder
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None

    def _get_encoder(self) -> Encoder:
        if self._encoder is None:
            self._encoder = SentenceTransformerEncoder(self.model_name)
        return self._encoder

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        if not self._chunks:
            self._vectors = np.zeros((0, 0), dtype=np.float64)
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = embedding_cache_path(
            self.cache_dir, self.model_name, self.chunk_params, self.corpus_sha256
        )
        if cache_path.exists():
            self._vectors = np.load(cache_path)
            return

        encoder = self._get_encoder()
        raw_vectors = np.asarray(encoder.encode([c.text for c in self._chunks]), dtype=np.float64)
        vectors = _normalize(raw_vectors)
        np.save(cache_path, vectors)
        self._vectors = vectors

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._vectors is None:
            raise RuntimeError("DenseRetriever.search called before index()")
        if not self._chunks:
            return []
        encoder = self._get_encoder()
        query_vector = _normalize(np.asarray(encoder.encode([query]), dtype=np.float64))[0]
        scores = self._vectors @ query_vector
        chunk_scores = list(zip(self._chunks, scores.tolist(), strict=True))
        ranked = aggregate_max_by_doc(chunk_scores)
        return ranked[:k]

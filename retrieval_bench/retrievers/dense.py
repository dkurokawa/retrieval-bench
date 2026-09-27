"""Embedding-based dense retrieval, with disk-cached, L2-normalized vectors."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from retrieval_bench.chunking import Chunk
from retrieval_bench.datasets import cache_root
from retrieval_bench.retrievers.base import aggregate_max_by_doc

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Encoder(Protocol):
    """Turns texts into dense vectors. Implemented by SentenceTransformerEncoder,
    and by fakes in tests."""

    def identifier(self) -> str:
        """A stable string identifying this encoder: class, model, and version/
        revision where obtainable. Used as part of the embedding cache key, and
        recorded with each run for reproducibility."""
        ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def _model_revision(model: object) -> str | None:  # pragma: no cover - best-effort, no stable API
    """Best-effort lookup of the loaded model's source revision.

    sentence-transformers doesn't expose this uniformly across versions, so
    this is opportunistic: it returns None (rather than raising) whenever the
    attribute isn't there.
    """
    card_data = getattr(model, "model_card_data", None)
    revision = getattr(card_data, "base_model_revision", None)
    return str(revision) if revision else None


class SentenceTransformerEncoder:
    """Encoder backed by the optional `sentence-transformers` package."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:  # pragma: no cover
        try:
            import sentence_transformers as st_module
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for the default dense encoder; "
                "install it with `uv sync --extra dense`"
            ) from exc
        self._model = SentenceTransformer(model_name)
        found_revision = _model_revision(self._model)
        # Without a revision the identifier cannot tell two versions of the same
        # model name apart, so cached vectors could silently outlive a weights
        # update. Such an encoder is marked uncacheable instead.
        self.cacheable = found_revision is not None
        revision = found_revision or "unknown"
        self._identifier = (
            f"sentence-transformers:{model_name}:st={st_module.__version__}:revision={revision}"
        )

    def identifier(self) -> str:  # pragma: no cover
        return self._identifier

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


def chunks_signature(chunks: Sequence[Chunk]) -> str:
    """sha256 of the ordered chunk texts actually handed to encode(), so a
    changed corpus, chunking, or chunk order always misses the cache."""
    payload = json.dumps([c.text for c in chunks])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def embedding_cache_path(cache_dir: Path, encoder_id: str, chunks_sha256: str) -> Path:
    """Deterministic cache path keyed by the encoder's identity and exact chunk texts."""
    key_material = json.dumps(
        {"encoder": encoder_id, "chunks_sha256": chunks_sha256}, sort_keys=True
    )
    key = hashlib.sha256(key_material.encode("utf-8")).hexdigest()
    return cache_dir / f"{key}.npy"


def _load_cached_embeddings(cache_path: Path, expected_rows: int) -> np.ndarray | None:
    """Load a cached embedding matrix, discarding (and deleting) it if it's
    corrupt, truncated, or doesn't match the number of chunks it should cover."""
    if not cache_path.exists():
        return None
    try:
        loaded: np.ndarray = np.load(cache_path)
    except (OSError, ValueError, EOFError):
        cache_path.unlink(missing_ok=True)
        return None
    if loaded.ndim != 2 or loaded.shape[0] != expected_rows or loaded.shape[1] == 0:
        cache_path.unlink(missing_ok=True)
        return None
    return loaded


def _atomic_save(cache_path: Path, vectors: np.ndarray) -> None:
    """Write via a temp file in the same directory, then rename, so a crash or
    a concurrent reader can never observe a partially-written cache file."""
    tmp_path = cache_path.parent / f"{cache_path.stem}.{uuid.uuid4().hex}.tmp.npy"
    try:
        np.save(tmp_path, vectors)
        os.replace(tmp_path, cache_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def default_embedding_cache_dir() -> Path:
    return cache_root() / "embeddings"


class DenseRetriever:
    """Dense retrieval: encode chunks, L2-normalize, rank by inner product."""

    def __init__(
        self,
        model_name: str,
        cache_dir: Path | None = None,
        encoder: Encoder | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir if cache_dir is not None else default_embedding_cache_dir()
        self._encoder = encoder
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None
        self.encoder_identifier: str | None = None

    def _get_encoder(self) -> Encoder:
        if self._encoder is None:
            self._encoder = SentenceTransformerEncoder(self.model_name)
        return self._encoder

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        if not self._chunks:
            self._vectors = np.zeros((0, 0), dtype=np.float64)
            return

        encoder = self._get_encoder()
        self.encoder_identifier = encoder.identifier()

        if not getattr(encoder, "cacheable", True):
            raw = np.asarray(encoder.encode([c.text for c in self._chunks]), dtype=np.float64)
            self._vectors = _normalize(raw)
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        signature = chunks_signature(self._chunks)
        cache_path = embedding_cache_path(self.cache_dir, self.encoder_identifier, signature)

        cached = _load_cached_embeddings(cache_path, expected_rows=len(self._chunks))
        if cached is not None:
            self._vectors = cached
            return

        raw_vectors = np.asarray(encoder.encode([c.text for c in self._chunks]), dtype=np.float64)
        vectors = _normalize(raw_vectors)
        _atomic_save(cache_path, vectors)
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

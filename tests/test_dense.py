from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from retrieval_bench.chunking import Chunk
from retrieval_bench.retrievers.dense import DenseRetriever, embedding_cache_path


class FakeEncoder:
    """Deterministic bag-of-words encoder over a fixed 3-word vocabulary."""

    vocab = ("cat", "dog", "fish")

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        rows = []
        for text in texts:
            words = text.lower().split()
            rows.append([float(words.count(v)) for v in self.vocab])
        return np.array(rows, dtype=np.float64)


def make_retriever(tmp_path: Path, encoder: FakeEncoder | None = None) -> DenseRetriever:
    return DenseRetriever(
        model_name="fake-model",
        chunk_params={"chunk_size": None, "chunk_overlap": 0},
        corpus_sha256="deadbeef",
        cache_dir=tmp_path,
        encoder=encoder or FakeEncoder(),
    )


def test_dense_ranks_by_cosine_similarity(tmp_path: Path) -> None:
    retriever = make_retriever(tmp_path)
    chunks = [
        Chunk(doc_id="cat_doc", chunk_index=0, text="cat cat"),
        Chunk(doc_id="dog_doc", chunk_index=0, text="dog"),
        Chunk(doc_id="fish_doc", chunk_index=0, text="fish fish fish"),
    ]
    retriever.index(chunks)
    results = retriever.search("cat", k=3)
    assert results[0][0] == "cat_doc"
    assert results[0][1] == pytest.approx(1.0)
    # dog/fish are orthogonal to a pure "cat" query.
    assert dict(results)["dog_doc"] == pytest.approx(0.0, abs=1e-9)
    assert dict(results)["fish_doc"] == pytest.approx(0.0, abs=1e-9)


def test_dense_embeddings_are_cached_across_index_calls(tmp_path: Path) -> None:
    encoder = FakeEncoder()
    chunks = [
        Chunk(doc_id="d1", chunk_index=0, text="cat"),
        Chunk(doc_id="d2", chunk_index=0, text="dog"),
    ]
    retriever_a = make_retriever(tmp_path, encoder=encoder)
    retriever_a.index(chunks)
    assert encoder.calls == 1

    # A second retriever with the same (model, chunk_params, corpus_sha256) key
    # should hit the on-disk cache and never call the encoder to index.
    retriever_b = make_retriever(tmp_path, encoder=encoder)
    retriever_b.index(chunks)
    assert encoder.calls == 1

    results_a = retriever_a.search("cat", k=2)
    results_b = retriever_b.search("cat", k=2)
    assert results_a == results_b


def test_embedding_cache_path_changes_with_key_material(tmp_path: Path) -> None:
    base = embedding_cache_path(tmp_path, "model-a", {"chunk_size": None}, "sha-1")
    different_model = embedding_cache_path(tmp_path, "model-b", {"chunk_size": None}, "sha-1")
    different_chunking = embedding_cache_path(tmp_path, "model-a", {"chunk_size": 100}, "sha-1")
    different_corpus = embedding_cache_path(tmp_path, "model-a", {"chunk_size": None}, "sha-2")

    assert len({base, different_model, different_chunking, different_corpus}) == 4


def test_dense_search_before_index_raises(tmp_path: Path) -> None:
    retriever = make_retriever(tmp_path)
    with pytest.raises(RuntimeError):
        retriever.search("cat", k=10)


def test_dense_empty_index_returns_empty_results(tmp_path: Path) -> None:
    retriever = make_retriever(tmp_path)
    retriever.index([])
    assert retriever.search("cat", k=10) == []


def test_dense_without_encoder_fails_helpfully_when_sentence_transformers_missing(
    tmp_path: Path,
) -> None:
    # No `encoder=` passed, and this test environment doesn't have the
    # optional `[dense]` extra installed (mirrors the CI lint/test job).
    retriever = DenseRetriever(
        model_name="any-model",
        chunk_params={"chunk_size": None, "chunk_overlap": 0},
        corpus_sha256="deadbeef",
        cache_dir=tmp_path,
    )
    with pytest.raises(ImportError, match="uv sync --extra dense"):
        retriever.index([Chunk(doc_id="d1", chunk_index=0, text="cat")])

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from retrieval_bench.chunking import Chunk
from retrieval_bench.retrievers.dense import (
    DenseRetriever,
    chunks_signature,
    embedding_cache_path,
)


class FakeEncoder:
    """Deterministic bag-of-words encoder over a fixed 3-word vocabulary."""

    vocab = ("cat", "dog", "fish")

    def __init__(self, name: str = "fake-encoder") -> None:
        self.calls = 0
        self._name = name

    def identifier(self) -> str:
        return self._name

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


def test_dense_max_pooling_picks_the_true_max_among_mixed_scores(tmp_path: Path) -> None:
    retriever = make_retriever(tmp_path)
    chunks = [
        # doc1: worst chunk first, best chunk last.
        Chunk(doc_id="doc1", chunk_index=0, text="fish fish fish"),
        Chunk(doc_id="doc1", chunk_index=1, text="dog dog"),
        Chunk(doc_id="doc1", chunk_index=2, text="cat"),
        # doc2: best chunk first, worst chunk last.
        Chunk(doc_id="doc2", chunk_index=0, text="cat cat"),
        Chunk(doc_id="doc2", chunk_index=1, text="fish"),
    ]
    retriever.index(chunks)
    ranked = retriever.search("cat", k=2)
    scores = dict(ranked)
    # Both docs have a pure "cat" direction chunk somewhere (cosine 1.0);
    # max-pooling must find it regardless of its position in the chunk list.
    assert scores["doc1"] == pytest.approx(1.0)
    assert scores["doc2"] == pytest.approx(1.0)


def test_dense_max_pooling_with_a_negative_cosine_chunk(tmp_path: Path) -> None:
    # A chunk whose vector points *away* from the query (negative cosine)
    # must never win max-pooling over a same-document chunk with a
    # non-negative score, and must not corrupt that document's score.

    class SignedEncoder:
        """Encodes "+cat"/"-cat" as vectors pointing toward/away from "cat"."""

        def identifier(self) -> str:
            return "signed-encoder"

        def encode(self, texts: Sequence[str]) -> np.ndarray:
            vectors = []
            for text in texts:
                if text == "-cat":
                    vectors.append([-1.0, 0.0])
                elif text == "cat":
                    vectors.append([1.0, 0.0])
                else:
                    vectors.append([0.0, 1.0])
            return np.array(vectors, dtype=np.float64)

    retriever = DenseRetriever(model_name="signed", cache_dir=tmp_path, encoder=SignedEncoder())
    chunks = [
        Chunk(doc_id="doc1", chunk_index=0, text="-cat"),
        Chunk(doc_id="doc1", chunk_index=1, text="unrelated"),
        Chunk(doc_id="doc2", chunk_index=0, text="-cat"),
    ]
    retriever.index(chunks)
    results = dict(retriever.search("cat", k=2))
    # doc1's best chunk is "unrelated" (cosine 0.0), beating its own
    # anti-correlated "-cat" chunk (cosine -1.0).
    assert results["doc1"] == pytest.approx(0.0)
    # doc2 has only the anti-correlated chunk, so its max is negative and it
    # must be reported as such, not clamped to 0 or dropped.
    assert results["doc2"] == pytest.approx(-1.0)


def test_dense_embeddings_are_cached_across_index_calls(tmp_path: Path) -> None:
    encoder = FakeEncoder()
    chunks = [
        Chunk(doc_id="d1", chunk_index=0, text="cat"),
        Chunk(doc_id="d2", chunk_index=0, text="dog"),
    ]
    retriever_a = make_retriever(tmp_path, encoder=encoder)
    retriever_a.index(chunks)
    assert encoder.calls == 1

    # A second retriever with the same encoder identity and the same chunk
    # texts should hit the on-disk cache and never call encode() to index.
    retriever_b = make_retriever(tmp_path, encoder=encoder)
    retriever_b.index(chunks)
    assert encoder.calls == 1

    results_a = retriever_a.search("cat", k=2)
    results_b = retriever_b.search("cat", k=2)
    assert results_a == results_b


def test_dense_cache_misses_when_the_encoder_identity_differs(tmp_path: Path) -> None:
    chunks = [Chunk(doc_id="d1", chunk_index=0, text="cat")]

    encoder_a = FakeEncoder(name="encoder-a")
    make_retriever(tmp_path, encoder=encoder_a).index(chunks)
    assert encoder_a.calls == 1

    encoder_b = FakeEncoder(name="encoder-b")
    make_retriever(tmp_path, encoder=encoder_b).index(chunks)
    # Different encoder identity -> different cache key -> must re-encode,
    # not silently reuse encoder_a's cached vectors.
    assert encoder_b.calls == 1
    assert len(list(tmp_path.glob("*.npy"))) == 2


def test_dense_cache_misses_when_chunk_order_differs(tmp_path: Path) -> None:
    encoder = FakeEncoder()
    forward = [
        Chunk(doc_id="d1", chunk_index=0, text="cat"),
        Chunk(doc_id="d2", chunk_index=0, text="dog"),
    ]
    reversed_chunks = list(reversed(forward))

    make_retriever(tmp_path, encoder=encoder).index(forward)
    assert encoder.calls == 1

    make_retriever(tmp_path, encoder=encoder).index(reversed_chunks)
    # Same texts, different order -> different signature -> must re-encode.
    assert encoder.calls == 2
    assert len(list(tmp_path.glob("*.npy"))) == 2


def test_dense_discards_a_cached_file_with_the_wrong_row_count(tmp_path: Path) -> None:
    encoder = FakeEncoder()
    chunks = [
        Chunk(doc_id="d1", chunk_index=0, text="cat"),
        Chunk(doc_id="d2", chunk_index=0, text="dog"),
    ]
    signature = chunks_signature(chunks)
    cache_path = embedding_cache_path(tmp_path, encoder.identifier(), signature)
    tmp_path.mkdir(parents=True, exist_ok=True)
    # Plant a cache file for the right key, but with only 1 row instead of 2.
    np.save(cache_path, np.zeros((1, 3), dtype=np.float64))

    retriever = make_retriever(tmp_path, encoder=encoder)
    retriever.index(chunks)

    assert encoder.calls == 1  # the bad cache was discarded, not trusted
    results = dict(retriever.search("cat", k=2))
    assert results["d1"] == pytest.approx(1.0)
    assert results["d2"] == pytest.approx(0.0)


def test_dense_discards_a_corrupt_cache_file(tmp_path: Path) -> None:
    encoder = FakeEncoder()
    chunks = [Chunk(doc_id="d1", chunk_index=0, text="cat")]
    signature = chunks_signature(chunks)
    cache_path = embedding_cache_path(tmp_path, encoder.identifier(), signature)
    tmp_path.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"not a valid npy file at all")

    retriever = make_retriever(tmp_path, encoder=encoder)
    retriever.index(chunks)  # must not raise

    assert encoder.calls == 1
    # The cache file must have been overwritten with a valid array afterward.
    reloaded = np.load(cache_path)
    assert reloaded.shape[0] == 1


def test_chunks_signature_depends_on_order() -> None:
    a = [Chunk(doc_id="d1", chunk_index=0, text="x"), Chunk(doc_id="d2", chunk_index=0, text="y")]
    b = list(reversed(a))
    assert chunks_signature(a) != chunks_signature(b)


def test_chunks_signature_depends_on_text_not_doc_id() -> None:
    a = [Chunk(doc_id="d1", chunk_index=0, text="same text")]
    b = [Chunk(doc_id="different-id", chunk_index=99, text="same text")]
    assert chunks_signature(a) == chunks_signature(b)


def test_embedding_cache_path_changes_with_key_material(tmp_path: Path) -> None:
    base = embedding_cache_path(tmp_path, "encoder-a", "sig-1")
    different_encoder = embedding_cache_path(tmp_path, "encoder-b", "sig-1")
    different_signature = embedding_cache_path(tmp_path, "encoder-a", "sig-2")

    assert len({base, different_encoder, different_signature}) == 3


def test_dense_search_before_index_raises(tmp_path: Path) -> None:
    retriever = make_retriever(tmp_path)
    with pytest.raises(RuntimeError):
        retriever.search("cat", k=10)


def test_dense_empty_index_returns_empty_results(tmp_path: Path) -> None:
    retriever = make_retriever(tmp_path)
    retriever.index([])
    assert retriever.search("cat", k=10) == []


def test_dense_records_encoder_identifier_after_indexing(tmp_path: Path) -> None:
    retriever = make_retriever(tmp_path, encoder=FakeEncoder(name="my-encoder-id"))
    assert retriever.encoder_identifier is None
    retriever.index([Chunk(doc_id="d1", chunk_index=0, text="cat")])
    assert retriever.encoder_identifier == "my-encoder-id"


def test_dense_lazily_builds_the_default_encoder_when_none_is_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # No `encoder=` passed: DenseRetriever must fall back to
    # SentenceTransformerEncoder(model_name), built lazily on first use.
    # We substitute a fake class here so this test needs neither the
    # optional `[dense]` extra nor network access, regardless of whether
    # sentence-transformers happens to be installed in this environment.
    built_with: list[str] = []

    class DummySentenceTransformerEncoder:
        def __init__(self, model_name: str) -> None:
            built_with.append(model_name)

        def identifier(self) -> str:
            return "dummy-sentence-transformer"

        def encode(self, texts: Sequence[str]) -> np.ndarray:
            return np.ones((len(texts), 2), dtype=np.float64)

    monkeypatch.setattr(
        "retrieval_bench.retrievers.dense.SentenceTransformerEncoder",
        DummySentenceTransformerEncoder,
    )

    retriever = DenseRetriever(model_name="some-model", cache_dir=tmp_path)
    retriever.index([Chunk(doc_id="d1", chunk_index=0, text="cat")])

    assert built_with == ["some-model"]


def test_sentence_transformer_encoder_missing_dependency_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate `sentence-transformers` not being importable (the CI lint/test
    # job runs without the `[dense]` extra) without depending on whether it
    # actually happens to be installed in this environment.
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "sentence_transformers":
            raise ImportError("no module named sentence_transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from retrieval_bench.retrievers.dense import SentenceTransformerEncoder

    with pytest.raises(ImportError, match="uv sync --extra dense"):
        SentenceTransformerEncoder("any-model")


def test_uncacheable_encoder_never_reads_or_writes_the_cache(tmp_path: Path) -> None:
    # An encoder that can't pin its model version must not reuse vectors a
    # different version of the same model produced.
    encoder = FakeEncoder()
    encoder.cacheable = False  # type: ignore[attr-defined]
    chunks = [Chunk(doc_id="d1", chunk_index=0, text="cat")]

    make_retriever(tmp_path, encoder=encoder).index(chunks)
    make_retriever(tmp_path, encoder=encoder).index(chunks)

    assert encoder.calls == 2
    assert list(tmp_path.glob("*.npy")) == []

from __future__ import annotations

import pytest

from retrieval_bench.chunking import Chunk
from retrieval_bench.retrievers.bm25 import BM25Retriever, tokenize

# Reference values below were computed independently with the textbook BM25
# formula (see retrieval_bench/retrievers/bm25.py docstring), for a 2-chunk
# index with k1=0.9, b=0.4:
#   chunk1 = "cat dog dog" (tf: cat=1, dog=2, len=3)
#   chunk2 = "cat cat cat" (tf: cat=3, dog=0, len=3)
#   N=2, avgdl=3, df(cat)=2, df(dog)=1
#   idf(cat) = ln(1 + (2-2+0.5)/(2+0.5)) = 0.1823215567939546
#   idf(dog) = ln(1 + (2-1+0.5)/(1+0.5)) = 0.6931471805599453
#   score(chunk1, "dog")     = 0.908261822802687
#   score(chunk2, "dog")     = 0.0
#   score(chunk1, "cat dog") = 1.0905833795966415
#   score(chunk2, "cat dog") = 0.2664699676219336


def make_index() -> BM25Retriever:
    retriever = BM25Retriever(k1=0.9, b=0.4)
    chunks = [
        Chunk(doc_id="doc1", chunk_index=0, text="cat dog dog"),
        Chunk(doc_id="doc2", chunk_index=0, text="cat cat cat"),
    ]
    retriever.index(chunks)
    return retriever


def test_tokenize_lowercases_and_splits_alphanumerics() -> None:
    assert tokenize("Cat, Dog! 123") == ["cat", "dog", "123"]


def test_tokenize_can_drop_stopwords() -> None:
    from retrieval_bench.retrievers.bm25 import DEFAULT_STOPWORDS

    assert tokenize("the cat is on the mat", DEFAULT_STOPWORDS) == ["cat", "mat"]


def test_bm25_score_matches_hand_computed_value_single_term() -> None:
    retriever = make_index()
    results = dict(retriever.search("dog", k=10))
    assert results["doc1"] == pytest.approx(0.908261822802687)
    # doc2 has no "dog" at all (score 0): it must not be returned.
    assert "doc2" not in results


def test_bm25_score_matches_hand_computed_value_two_terms() -> None:
    retriever = make_index()
    results = dict(retriever.search("cat dog", k=10))
    assert results["doc1"] == pytest.approx(1.0905833795966415)
    assert results["doc2"] == pytest.approx(0.2664699676219336)


def test_bm25_ranks_by_descending_score() -> None:
    retriever = make_index()
    ranked = retriever.search("cat dog", k=10)
    assert [doc_id for doc_id, _ in ranked] == ["doc1", "doc2"]


def test_bm25_only_returns_documents_with_a_positive_score() -> None:
    # "dog" matches nothing in doc2, and "nonexistentword" matches nothing at
    # all: neither should ever appear in the results, at any k.
    retriever = make_index()
    assert dict(retriever.search("dog", k=10)).keys() == {"doc1"}
    assert retriever.search("nonexistentword", k=10) == []


def test_bm25_respects_k() -> None:
    retriever = make_index()
    assert len(retriever.search("cat", k=1)) == 1


def test_bm25_max_pools_multiple_chunks_per_document() -> None:
    retriever = BM25Retriever(k1=0.9, b=0.4)
    chunks = [
        Chunk(doc_id="doc1", chunk_index=0, text="dog dog dog"),
        Chunk(doc_id="doc1", chunk_index=1, text="cat cat cat"),
        Chunk(doc_id="doc2", chunk_index=0, text="cat cat cat"),
    ]
    retriever.index(chunks)
    results = dict(retriever.search("dog", k=10))
    # doc1's best chunk (chunk_index=0) should dominate; doc2 has no "dog" at
    # all, so it must be absent (not present with score 0.0).
    assert results["doc1"] > 0.0
    assert "doc2" not in results


def test_bm25_ranking_is_not_driven_by_doc_id_lexical_order() -> None:
    # The document with the lexically *last* id is the best match; a bug that
    # let doc_id ordering leak into ranking (instead of only breaking exact
    # score ties) would put it last instead of first.
    retriever = BM25Retriever(k1=0.9, b=0.4)
    chunks = [
        Chunk(doc_id="zzz_best_match", chunk_index=0, text="dog dog dog dog"),
        Chunk(doc_id="mmm_medium_match", chunk_index=0, text="dog cat cat cat"),
        Chunk(doc_id="aaa_no_match", chunk_index=0, text="cat cat cat cat"),
    ]
    retriever.index(chunks)
    ranked = retriever.search("dog", k=10)
    assert [doc_id for doc_id, _ in ranked] == ["zzz_best_match", "mmm_medium_match"]


def test_bm25_search_before_index_raises() -> None:
    retriever = BM25Retriever()
    with pytest.raises(RuntimeError):
        retriever.search("dog", k=10)


def test_bm25_empty_index_returns_empty_results() -> None:
    retriever = BM25Retriever()
    retriever.index([])
    assert retriever.search("dog", k=10) == []


def test_bm25_handles_all_empty_chunks_without_dividing_by_zero() -> None:
    retriever = BM25Retriever()
    retriever.index([Chunk(doc_id="doc1", chunk_index=0, text="")])
    # avgdl is 0 here; search must not raise, produce NaN/inf, or return a
    # zero-score "match".
    assert retriever.search("dog", k=10) == []


@pytest.mark.parametrize("k1", [-1.0, -0.01])
def test_bm25_rejects_negative_k1(k1: float) -> None:
    with pytest.raises(ValueError, match="k1"):
        BM25Retriever(k1=k1)


@pytest.mark.parametrize("b", [-0.01, 1.01, 2.0])
def test_bm25_rejects_b_outside_unit_interval(b: float) -> None:
    with pytest.raises(ValueError, match="b must be between 0 and 1"):
        BM25Retriever(b=b)


@pytest.mark.parametrize("b", [0.0, 1.0, 0.5])
def test_bm25_accepts_b_at_unit_interval_boundaries(b: float) -> None:
    BM25Retriever(b=b)  # must not raise


def test_bm25_accepts_k1_zero() -> None:
    BM25Retriever(k1=0.0)  # must not raise

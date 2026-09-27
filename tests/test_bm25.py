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
    assert results["doc2"] == pytest.approx(0.0)


def test_bm25_score_matches_hand_computed_value_two_terms() -> None:
    retriever = make_index()
    results = dict(retriever.search("cat dog", k=10))
    assert results["doc1"] == pytest.approx(1.0905833795966415)
    assert results["doc2"] == pytest.approx(0.2664699676219336)


def test_bm25_ranks_by_descending_score() -> None:
    retriever = make_index()
    ranked = retriever.search("dog", k=10)
    assert [doc_id for doc_id, _ in ranked] == ["doc1", "doc2"]


def test_bm25_unknown_query_term_contributes_nothing() -> None:
    retriever = make_index()
    results = dict(retriever.search("nonexistentword", k=10))
    assert results["doc1"] == pytest.approx(0.0)
    assert results["doc2"] == pytest.approx(0.0)


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
    # doc1's best chunk (chunk_index=0) should dominate; doc2 has no "dog" at all.
    assert results["doc1"] > 0.0
    assert results["doc2"] == pytest.approx(0.0)


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
    # avgdl is 0 here; search must not raise or produce NaN/inf scores.
    assert retriever.search("dog", k=10) == [("doc1", 0.0)]

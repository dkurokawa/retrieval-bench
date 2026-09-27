from __future__ import annotations

import pytest

from retrieval_bench.chunking import Chunk, chunk
from retrieval_bench.datasets import Doc


def make_doc(text: str, title: str = "") -> Doc:
    return Doc(doc_id="d1", title=title, text=text)


def test_no_split_when_size_is_none() -> None:
    doc = make_doc("one two three four five", title="Title")
    chunks = chunk(doc, size=None, overlap=0)
    assert chunks == [Chunk(doc_id="d1", chunk_index=0, text="Title one two three four five")]


def test_exact_multiple_of_size_no_overlap() -> None:
    doc = make_doc("w1 w2 w3 w4 w5 w6")
    chunks = chunk(doc, size=2, overlap=0)
    assert [c.text for c in chunks] == ["w1 w2", "w3 w4", "w5 w6"]
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert all(c.doc_id == "d1" for c in chunks)


def test_overlap_repeats_words_between_chunks() -> None:
    doc = make_doc("w1 w2 w3 w4 w5")
    chunks = chunk(doc, size=3, overlap=1)
    # step = size - overlap = 2
    assert [c.text for c in chunks] == ["w1 w2 w3", "w3 w4 w5"]


def test_last_chunk_may_be_shorter() -> None:
    doc = make_doc("w1 w2 w3 w4 w5")
    chunks = chunk(doc, size=2, overlap=0)
    assert [c.text for c in chunks] == ["w1 w2", "w3 w4", "w5"]


def test_empty_document_yields_single_empty_chunk() -> None:
    doc = make_doc("")
    chunks = chunk(doc, size=3, overlap=0)
    assert chunks == [Chunk(doc_id="d1", chunk_index=0, text="")]


@pytest.mark.parametrize("size,overlap", [(3, 3), (3, 4), (0, 0), (-1, 0)])
def test_invalid_size_overlap_combinations_raise(size: int, overlap: int) -> None:
    doc = make_doc("w1 w2 w3 w4")
    with pytest.raises(ValueError):
        chunk(doc, size=size, overlap=overlap)


def test_negative_overlap_raises() -> None:
    doc = make_doc("w1 w2 w3")
    with pytest.raises(ValueError):
        chunk(doc, size=2, overlap=-1)

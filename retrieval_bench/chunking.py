"""Splitting documents into overlapping, word-based chunks."""

from __future__ import annotations

from dataclasses import dataclass

from retrieval_bench.datasets import Doc


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of a document's words."""

    doc_id: str
    chunk_index: int
    text: str


def chunk(doc: Doc, size: int | None, overlap: int) -> list[Chunk]:
    """Split ``doc`` into chunks of ``size`` whitespace-separated words.

    ``size=None`` disables splitting: the whole document becomes one chunk.
    ``overlap`` is the number of words shared between consecutive chunks and
    must be smaller than ``size``.
    """
    contents = doc.contents
    if size is None:
        return [Chunk(doc_id=doc.doc_id, chunk_index=0, text=contents)]
    if size <= 0:
        raise ValueError(f"size must be a positive integer, got {size}")
    if overlap < 0:
        raise ValueError(f"overlap must not be negative, got {overlap}")
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")

    words = contents.split()
    if not words:
        return [Chunk(doc_id=doc.doc_id, chunk_index=0, text="")]

    step = size - overlap
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(words):
        piece = words[start : start + size]
        chunks.append(Chunk(doc_id=doc.doc_id, chunk_index=index, text=" ".join(piece)))
        if start + size >= len(words):
            break
        start += step
        index += 1
    return chunks

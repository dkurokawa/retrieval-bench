from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from retrieval_bench.datasets import Dataset, load, load_dir

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tiny"


def test_load_dir_parses_corpus_queries_and_qrels() -> None:
    dataset = load_dir(FIXTURE_DIR, name="tiny")

    assert isinstance(dataset, Dataset)
    assert dataset.name == "tiny"
    assert len(dataset.corpus) == 10
    assert dataset.corpus["d2"].title == "Bananas"
    assert dataset.corpus["d2"].text == "banana banana banana fruit yellow"
    assert dataset.corpus["d2"].contents == "Bananas banana banana banana fruit yellow"

    assert dataset.queries == {
        "q1": "banana fruit",
        "q2": "engine train",
        "q3": "water river",
        "q4": "unrelated query about nothing here",
    }

    assert dataset.qrels["q1"] == {"d2": 2, "d1": 1}
    assert dataset.qrels["q2"] == {"d4": 2, "d3": 1}
    # q4 has no judgments at all in the fixture.
    assert "q4" not in dataset.qrels


def test_corpus_sha256_matches_file_contents() -> None:
    dataset = load_dir(FIXTURE_DIR, name="tiny")
    expected = hashlib.sha256((FIXTURE_DIR / "corpus.jsonl").read_bytes()).hexdigest()
    assert dataset.corpus_sha256 == expected


def test_load_dir_is_deterministic() -> None:
    a = load_dir(FIXTURE_DIR, name="tiny")
    b = load_dir(FIXTURE_DIR, name="tiny")
    assert a.corpus_sha256 == b.corpus_sha256
    assert a.corpus.keys() == b.corpus.keys()


def test_load_missing_dataset_raises_with_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RBENCH_CACHE", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="rbench download scifact"):
        load("scifact")


def test_doc_contents_without_title() -> None:
    dataset = load_dir(FIXTURE_DIR, name="tiny")
    doc = dataset.corpus["d1"]
    stripped = doc.__class__(doc_id=doc.doc_id, title="", text=doc.text)
    assert stripped.contents == doc.text

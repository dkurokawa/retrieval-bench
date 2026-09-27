from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from typing import Any

import pytest

from retrieval_bench import datasets
from retrieval_bench.datasets import Dataset, DatasetSpec, download, load, load_dir

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tiny"


def _build_fake_zip_bytes(name: str) -> bytes:
    """A minimal, valid BEIR-format zip for `name`, matching the real layout."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{name}/corpus.jsonl", '{"_id": "d1", "title": "T", "text": "hello world"}\n')
        zf.writestr(f"{name}/queries.jsonl", '{"_id": "q1", "text": "hello"}\n')
        zf.writestr(f"{name}/qrels/test.tsv", "query-id\tcorpus-id\tscore\nq1\td1\t1\n")
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def _register_fake_dataset(monkeypatch: pytest.MonkeyPatch, name: str, payload: bytes) -> list[int]:
    """Register `name` in DATASET_REGISTRY and make urlopen return `payload`
    for it, regardless of URL. Returns a list whose length grows by one on
    each simulated network fetch, so tests can assert call counts."""
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setitem(
        datasets.DATASET_REGISTRY,
        name,
        DatasetSpec(name=name, url=f"https://example.invalid/{name}.zip", sha256=digest),
    )
    calls: list[int] = []

    def fake_urlopen(url: str) -> Any:
        calls.append(1)
        return _FakeResponse(payload)

    monkeypatch.setattr(datasets, "urlopen", fake_urlopen)
    return calls


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


def test_queries_and_qrels_sha256_match_file_contents_and_split_is_recorded() -> None:
    dataset = load_dir(FIXTURE_DIR, name="tiny")
    assert (
        dataset.queries_sha256
        == hashlib.sha256((FIXTURE_DIR / "queries.jsonl").read_bytes()).hexdigest()
    )
    assert (
        dataset.qrels_sha256
        == hashlib.sha256((FIXTURE_DIR / "qrels" / "test.tsv").read_bytes()).hexdigest()
    )
    assert dataset.split == "test"


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


def test_download_writes_a_complete_dataset_with_a_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _build_fake_zip_bytes("tinytest")
    calls = _register_fake_dataset(monkeypatch, "tinytest", payload)

    result_dir = download("tinytest", cache_dir=tmp_path)

    assert result_dir == tmp_path / "tinytest"
    assert (result_dir / "corpus.jsonl").exists()
    assert (result_dir / "queries.jsonl").exists()
    assert (result_dir / "qrels" / "test.tsv").exists()
    assert (result_dir / ".rbench-complete").exists()
    assert len(calls) == 1

    # A dataset downloaded this way must be loadable through the normal path.
    dataset = load("tinytest", cache_dir=tmp_path)
    assert dataset.queries == {"q1": "hello"}


def test_download_is_a_no_op_once_the_marker_is_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _build_fake_zip_bytes("tinytest")
    calls = _register_fake_dataset(monkeypatch, "tinytest", payload)

    download("tinytest", cache_dir=tmp_path)
    assert len(calls) == 1

    download("tinytest", cache_dir=tmp_path)
    assert len(calls) == 1  # no second network fetch


def test_download_recovers_from_an_interrupted_prior_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulate a process that died mid-extraction in a previous run: some
    # files exist, but the completion marker was never written.
    partial_dir = tmp_path / "tinytest"
    partial_dir.mkdir(parents=True)
    (partial_dir / "corpus.jsonl").write_text("truncated garbage, not valid jsonl")

    payload = _build_fake_zip_bytes("tinytest")
    calls = _register_fake_dataset(monkeypatch, "tinytest", payload)

    result_dir = download("tinytest", cache_dir=tmp_path)

    assert len(calls) == 1  # it re-fetched rather than trusting the partial dir
    assert (result_dir / ".rbench-complete").exists()
    # The stale partial content must be gone, replaced by the real download.
    dataset = load("tinytest", cache_dir=tmp_path)
    assert dataset.corpus["d1"].text == "hello world"


def test_load_raises_for_a_directory_missing_the_completion_marker(
    tmp_path: Path,
) -> None:
    # Files present (as an old, pre-marker partial download would leave) but
    # no marker: load() must refuse rather than risk reading a partial file.
    partial_dir = tmp_path / "scifact"
    (partial_dir / "qrels").mkdir(parents=True)
    (partial_dir / "corpus.jsonl").write_text("{}\n")
    (partial_dir / "queries.jsonl").write_text("{}\n")
    (partial_dir / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\n")

    with pytest.raises(FileNotFoundError, match="rbench download scifact"):
        load("scifact", cache_dir=tmp_path)


def test_download_rejects_a_zip_missing_a_required_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("tinytest/corpus.jsonl", "{}\n")
        # queries.jsonl and qrels/test.tsv are missing.
    payload = buf.getvalue()
    _register_fake_dataset(monkeypatch, "tinytest", payload)

    with pytest.raises(ValueError, match="missing required file"):
        download("tinytest", cache_dir=tmp_path)

    # No partial/complete directory must be left behind after the failure.
    assert not (tmp_path / "tinytest" / ".rbench-complete").exists()


def test_download_rejects_a_sha256_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _build_fake_zip_bytes("tinytest")
    monkeypatch.setitem(
        datasets.DATASET_REGISTRY,
        "tinytest",
        DatasetSpec(name="tinytest", url="https://example.invalid/x.zip", sha256="0" * 64),
    )
    monkeypatch.setattr(datasets, "urlopen", lambda url: _FakeResponse(payload))

    with pytest.raises(ValueError, match="sha256 mismatch"):
        download("tinytest", cache_dir=tmp_path)

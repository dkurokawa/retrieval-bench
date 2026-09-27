"""Loading and downloading BEIR-format retrieval datasets.

A BEIR-format dataset directory holds three files:

- ``corpus.jsonl``: one JSON object per line with ``_id``, ``title``, ``text``.
- ``queries.jsonl``: one JSON object per line with ``_id``, ``text``.
- ``qrels/<split>.tsv``: tab-separated ``query-id\tcorpus-id\tscore`` rows
  (with a header row), giving graded relevance judgments.

Datasets themselves are never committed to this repository; they are
downloaded on demand into a local cache directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen


@dataclass(frozen=True)
class Doc:
    """A single corpus document."""

    doc_id: str
    title: str
    text: str

    @property
    def contents(self) -> str:
        """Title and body concatenated, as fed to chunking and retrievers."""
        if self.title:
            return f"{self.title} {self.text}".strip()
        return self.text


@dataclass(frozen=True)
class Dataset:
    """An in-memory BEIR-format dataset."""

    name: str
    corpus: dict[str, Doc]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    corpus_sha256: str


@dataclass(frozen=True)
class DatasetSpec:
    """Download metadata for a registered dataset."""

    name: str
    url: str
    sha256: str


# sha256 of the zip files, computed from the URLs below on 2026-09-27.
DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "scifact": DatasetSpec(
        name="scifact",
        url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        sha256="536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
    ),
    "nfcorpus": DatasetSpec(
        name="nfcorpus",
        url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
        sha256="efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b",
    ),
}


def cache_root() -> Path:
    """Root cache directory, overridable via the ``RBENCH_CACHE`` env var."""
    override = os.environ.get("RBENCH_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "retrieval-bench"


def dataset_dir(name: str, cache_dir: Path | None = None) -> Path:
    """Path a dataset would be (or is) extracted to."""
    root = cache_dir if cache_dir is not None else cache_root()
    return root / name


def download(name: str, cache_dir: Path | None = None) -> Path:
    """Download and extract a registered BEIR dataset, verifying its sha256.

    No-ops (and returns the existing directory) if the dataset already looks
    downloaded.
    """
    if name not in DATASET_REGISTRY:
        known = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"unknown dataset {name!r}; known datasets: {known}")
    spec = DATASET_REGISTRY[name]
    root = cache_dir if cache_dir is not None else cache_root()
    root.mkdir(parents=True, exist_ok=True)
    target_dir = root / name
    if (target_dir / "corpus.jsonl").exists():
        return target_dir

    zip_path = root / f"{name}.zip"
    with urlopen(spec.url) as response:  # noqa: S310 - fixed, vetted https URL
        payload = response.read()
    zip_path.write_bytes(payload)

    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    if digest != spec.sha256:
        zip_path.unlink(missing_ok=True)
        raise ValueError(f"sha256 mismatch for {name}: expected {spec.sha256}, got {digest}")

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    zip_path.unlink()
    return target_dir


def load_dir(path: Path, name: str, split: str = "test") -> Dataset:
    """Load a BEIR-format dataset from an already-extracted directory."""
    corpus_path = path / "corpus.jsonl"
    queries_path = path / "queries.jsonl"
    qrels_path = path / "qrels" / f"{split}.tsv"

    corpus_bytes = corpus_path.read_bytes()
    corpus_sha256 = hashlib.sha256(corpus_bytes).hexdigest()

    corpus: dict[str, Doc] = {}
    for line in corpus_bytes.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        corpus[row["_id"]] = Doc(doc_id=row["_id"], title=row.get("title", ""), text=row["text"])

    queries: dict[str, str] = {}
    for line in queries_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        queries[row["_id"]] = row["text"]

    qrels: dict[str, dict[str, int]] = {}
    with qrels_path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()
    start = 1 if lines and lines[0].startswith("query-id") else 0
    for line in lines[start:]:
        if not line.strip():
            continue
        query_id, corpus_id, score = line.rstrip("\n").split("\t")
        qrels.setdefault(query_id, {})[corpus_id] = int(score)

    return Dataset(
        name=name, corpus=corpus, queries=queries, qrels=qrels, corpus_sha256=corpus_sha256
    )


def load(name: str, cache_dir: Path | None = None, split: str = "test") -> Dataset:
    """Load a dataset from the cache, raising if it hasn't been downloaded yet."""
    path = dataset_dir(name, cache_dir)
    if not (path / "corpus.jsonl").exists():
        raise FileNotFoundError(
            f"dataset {name!r} not found in {path}; run `rbench download {name}` first"
        )
    return load_dir(path, name, split=split)

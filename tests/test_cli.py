from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pytest

from retrieval_bench import datasets
from retrieval_bench.cli import _build_retriever, build_parser, main
from retrieval_bench.retrievers.dense import DenseRetriever
from retrieval_bench.retrievers.hybrid import HybridRetriever

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tiny"


@pytest.fixture
def cached_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Make the tiny fixture look like an already-downloaded 'scifact' dataset."""
    cache = tmp_path / "cache"
    monkeypatch.setenv("RBENCH_CACHE", str(cache))
    shutil.copytree(FIXTURE_DIR, cache / "scifact")
    (cache / "scifact" / ".rbench-complete").write_text("test-fixture", encoding="utf-8")
    return cache


def run_cli(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def test_run_command_prints_run_id_and_metrics(
    cached_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "results.duckdb"
    exit_code = run_cli(["run", "--dataset", "scifact", "--retriever", "bm25", "--db", str(db)])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "run_id=" in out
    assert "ndcg@10:" in out
    assert db.exists()


def test_compare_command_lists_recorded_runs(
    cached_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "results.duckdb"
    run_cli(["run", "--dataset", "scifact", "--retriever", "bm25", "--db", str(db)])
    capsys.readouterr()  # discard the `run` command's own output

    exit_code = run_cli(["compare", "--dataset", "scifact", "--db", str(db)])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "bm25" in out
    assert "ndcg@10" in out


def test_compare_command_markdown_output(
    cached_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "results.duckdb"
    run_cli(["run", "--dataset", "scifact", "--retriever", "bm25", "--db", str(db)])
    capsys.readouterr()

    run_cli(["compare", "--markdown", "--db", str(db)])
    out = capsys.readouterr().out
    assert out.startswith("| run_id")


def test_compare_with_no_runs_reports_that(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "results.duckdb"
    exit_code = run_cli(["compare", "--db", str(db)])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "no runs recorded" in out


def test_diff_command_orders_queries_by_absolute_delta(
    cached_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "results.duckdb"
    run_cli(["run", "--dataset", "scifact", "--retriever", "bm25", "--k1", "0.9", "--db", str(db)])
    run_id_a = capsys.readouterr().out.splitlines()[0].split("=")[1]

    run_cli(["run", "--dataset", "scifact", "--retriever", "bm25", "--k1", "2.5", "--db", str(db)])
    run_id_b = capsys.readouterr().out.splitlines()[0].split("=")[1]

    exit_code = run_cli(["diff", run_id_a, run_id_b, "--metric", "ndcg@10", "--db", str(db)])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "query_id" in out


def _base_args(**overrides: object) -> argparse.Namespace:
    defaults = dict(
        retriever="bm25",
        model=None,
        chunk_size=None,
        overlap=0,
        k1=0.9,
        b=0.4,
        rrf_k=60,
        rrf_depth=1000,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_retriever_bm25() -> None:
    retriever, params = _build_retriever(_base_args(retriever="bm25", k1=1.2, b=0.5))
    assert params == {"k1": 1.2, "b": 0.5}
    assert retriever.k1 == 1.2  # type: ignore[attr-defined]


def test_build_retriever_dense() -> None:
    retriever, params = _build_retriever(_base_args(retriever="dense"))
    assert isinstance(retriever, DenseRetriever)
    assert params["model"]


def test_build_retriever_hybrid() -> None:
    retriever, params = _build_retriever(_base_args(retriever="hybrid", rrf_k=42, rrf_depth=250))
    assert isinstance(retriever, HybridRetriever)
    assert params["rrf_k"] == 42
    assert params["rrf_depth"] == 250
    assert retriever.depth == 250


def test_build_retriever_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown retriever"):
        _build_retriever(_base_args(retriever="bogus"))


def test_download_command_delegates_to_datasets_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[str] = []

    def fake_download(name: str) -> Path:
        calls.append(name)
        return tmp_path / name

    monkeypatch.setattr(datasets, "download", fake_download)
    exit_code = main(["download", "scifact"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert calls == ["scifact"]
    assert "downloaded scifact" in out


def test_main_runs_end_to_end(
    cached_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "results.duckdb"
    exit_code = main(["run", "--dataset", "scifact", "--retriever", "bm25", "--db", str(db)])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "run_id=" in out


def test_run_command_parses_rrf_depth() -> None:
    # Full argparse round-trip for the flag; actually constructing a hybrid
    # retriever from it (real embeddings) is covered at the unit level by
    # test_build_retriever_hybrid, so this doesn't need network/[dense].
    parser = build_parser()
    args = parser.parse_args(
        ["run", "--dataset", "scifact", "--retriever", "hybrid", "--rrf-depth", "5"]
    )
    assert args.rrf_depth == 5


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--k1", "-1"],
        ["--b", "1.5"],
        ["--b", "-0.1"],
        ["--rrf-k", "0"],
        ["--rrf-depth", "0"],
        ["--rrf-depth", "-5"],
    ],
)
def test_run_command_rejects_invalid_hyperparameters(extra_args: list[str]) -> None:
    # argparse itself must reject these while parsing, before any dataset is
    # loaded or retriever constructed.
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--dataset", "scifact", "--retriever", "bm25", *extra_args])

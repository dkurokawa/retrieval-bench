from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from retrieval_bench.datasets import load_dir
from retrieval_bench.metrics import evaluate
from retrieval_bench.retrievers.bm25 import BM25Retriever
from retrieval_bench.retrievers.dense import DenseRetriever
from retrieval_bench.runs import connect, fetch_query_metrics, fetch_runs, git_sha, run, save_run

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tiny"


def test_run_produces_metrics_consistent_with_direct_evaluate() -> None:
    dataset = load_dir(FIXTURE_DIR, name="tiny")
    retriever = BM25Retriever(k1=0.9, b=0.4)

    result = run(
        dataset,
        retriever,
        retriever_name="bm25",
        retriever_params={"k1": 0.9, "b": 0.4},
        chunk_size=None,
        chunk_overlap=0,
    )

    assert result.dataset == "tiny"
    assert result.retriever == "bm25"
    # The tiny fixture has 4 queries, but only q1/q2/q3 have qrels judgments
    # (q4 has none at all) -- n_queries counts queries actually averaged over,
    # matching len(per_query), not every query in queries.jsonl.
    assert result.n_queries == 3
    assert result.n_queries == len(result.metrics.per_query)
    assert result.params == {"chunk_size": None, "chunk_overlap": 0, "k1": 0.9, "b": 0.4}
    assert result.git_sha is None or isinstance(result.git_sha, str)
    assert set(result.metrics.mean) == {"recall@10", "recall@100", "ndcg@10", "mrr@10"}

    assert result.meta["corpus_sha256"] == dataset.corpus_sha256
    assert result.meta["queries_sha256"] == dataset.queries_sha256
    assert result.meta["qrels_sha256"] == dataset.qrels_sha256
    assert result.meta["split"] == "test"
    assert result.meta["encoder_identifier"] is None  # bm25 has no encoder
    assert result.meta["dependency_versions"]["numpy"]
    assert result.meta["dependency_versions"]["scipy"]

    # Re-running the same retriever over the same chunking should give the
    # same run-level ranking (BM25 is deterministic) and hence identical
    # metrics computed by evaluate() directly.
    retriever2 = BM25Retriever(k1=0.9, b=0.4)
    from retrieval_bench.chunking import chunk

    chunks = [c for doc in dataset.corpus.values() for c in chunk(doc, None, 0)]
    retriever2.index(chunks)
    run_ranked = {
        qid: [doc_id for doc_id, _ in retriever2.search(qtext, 100)]
        for qid, qtext in dataset.queries.items()
    }
    expected = evaluate(run_ranked, dataset.qrels)
    assert result.metrics.mean == expected.mean


def test_save_and_fetch_run_round_trip(tmp_path: Path) -> None:
    dataset = load_dir(FIXTURE_DIR, name="tiny")
    retriever = BM25Retriever()
    result = run(
        dataset,
        retriever,
        retriever_name="bm25",
        retriever_params={"k1": 0.9, "b": 0.4},
        chunk_size=None,
        chunk_overlap=0,
    )

    db_path = tmp_path / "results.duckdb"
    con = connect(db_path)
    save_run(con, result)

    summaries = fetch_runs(con, dataset="tiny")
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.run_id == result.run_id
    assert summary.retriever == "bm25"
    assert summary.params == result.params
    assert summary.metrics == result.metrics.mean
    assert summary.meta == result.meta

    query_metrics = fetch_query_metrics(con, result.run_id)
    assert set(query_metrics) == set(result.metrics.per_query)
    for query_id, values in result.metrics.per_query.items():
        assert query_metrics[query_id] == values

    con.close()


def test_fetch_runs_filters_by_dataset(tmp_path: Path) -> None:
    dataset = load_dir(FIXTURE_DIR, name="tiny")
    db_path = tmp_path / "results.duckdb"
    con = connect(db_path)

    for name in ("tiny", "other"):
        result = run(
            dataset,
            BM25Retriever(),
            retriever_name="bm25",
            retriever_params={"k1": 0.9, "b": 0.4},
            chunk_size=None,
            chunk_overlap=0,
        )
        object.__setattr__(result, "dataset", name)
        save_run(con, result)

    assert len(fetch_runs(con, dataset="tiny")) == 1
    assert len(fetch_runs(con, dataset="other")) == 1
    assert len(fetch_runs(con)) == 2

    con.close()


def test_git_sha_returns_none_outside_a_git_repo(tmp_path: Path) -> None:
    assert git_sha(cwd=tmp_path) is None


def test_git_sha_returns_none_when_git_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert git_sha(cwd=tmp_path) is None


def test_run_only_searches_queries_present_in_the_loaded_qrels_split() -> None:
    dataset = load_dir(FIXTURE_DIR, name="tiny")
    searched_queries: list[str] = []

    class RecordingRetriever:
        def index(self, chunks: object) -> None:
            return None

        def search(self, query: str, k: int) -> list[tuple[str, float]]:
            searched_queries.append(query)
            return []

    run(
        dataset,
        RecordingRetriever(),  # type: ignore[arg-type]
        retriever_name="recording",
        retriever_params={},
        chunk_size=None,
        chunk_overlap=0,
    )

    # q4 ("unrelated query about nothing here") has no qrels row at all in
    # the fixture's test split and must never be searched.
    assert "unrelated query about nothing here" not in searched_queries
    assert set(searched_queries) == {"banana fruit", "engine train", "water river"}


def test_run_id_is_unique_across_runs() -> None:
    dataset = load_dir(FIXTURE_DIR, name="tiny")
    result_a = run(
        dataset,
        BM25Retriever(),
        retriever_name="bm25",
        retriever_params={},
        chunk_size=None,
        chunk_overlap=0,
    )
    result_b = run(
        dataset,
        BM25Retriever(),
        retriever_name="bm25",
        retriever_params={},
        chunk_size=None,
        chunk_overlap=0,
    )
    assert result_a.run_id != result_b.run_id


def test_run_records_the_encoder_identifier_for_a_dense_retriever(tmp_path: Path) -> None:
    dataset = load_dir(FIXTURE_DIR, name="tiny")

    class FakeEncoder:
        def identifier(self) -> str:
            return "fake-encoder-v1"

        def encode(self, texts: object) -> object:
            import numpy as np

            return np.ones((len(list(texts)), 2), dtype=np.float64)  # type: ignore[arg-type]

    retriever = DenseRetriever(model_name="fake-model", cache_dir=tmp_path, encoder=FakeEncoder())
    result = run(
        dataset,
        retriever,
        retriever_name="dense",
        retriever_params={"model": "fake-model"},
        chunk_size=None,
        chunk_overlap=0,
    )
    assert result.meta["encoder_identifier"] == "fake-encoder-v1"

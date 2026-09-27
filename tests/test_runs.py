from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from retrieval_bench.datasets import load_dir
from retrieval_bench.metrics import evaluate
from retrieval_bench.retrievers.bm25 import BM25Retriever
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
    assert result.n_queries == len(dataset.queries)
    assert result.params == {"chunk_size": None, "chunk_overlap": 0, "k1": 0.9, "b": 0.4}
    assert result.git_sha is None or isinstance(result.git_sha, str)
    assert set(result.metrics.mean) == {"recall@10", "recall@100", "ndcg@10", "mrr@10"}

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

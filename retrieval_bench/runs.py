"""Running a retriever over a dataset and recording results in DuckDB."""

from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from retrieval_bench.chunking import Chunk, chunk
from retrieval_bench.datasets import Dataset
from retrieval_bench.metrics import MetricResult, evaluate
from retrieval_bench.retrievers.base import Retriever

DEFAULT_DB_PATH = Path("results.duckdb")


@dataclass(frozen=True)
class RunResult:
    """Everything recorded for one retriever run over one dataset."""

    run_id: str
    dataset: str
    retriever: str
    params: dict[str, Any]
    created_at: datetime
    git_sha: str | None
    n_queries: int
    metrics: MetricResult


@dataclass(frozen=True)
class RunSummary:
    """A run's metadata plus its mean metrics, as read back from storage."""

    run_id: str
    created_at: datetime
    dataset: str
    retriever: str
    params: dict[str, Any]
    n_queries: int
    metrics: dict[str, float]


def git_sha(cwd: Path | None = None) -> str | None:
    """Current commit hash, or None if unavailable (not a repo, git missing, etc.)."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    sha = completed.stdout.strip()
    return sha or None


def run(
    dataset: Dataset,
    retriever: Retriever,
    *,
    retriever_name: str,
    retriever_params: dict[str, Any],
    chunk_size: int | None,
    chunk_overlap: int,
    k_search: int = 100,
) -> RunResult:
    """Chunk ``dataset``, index it with ``retriever``, and evaluate all queries."""
    chunks: list[Chunk] = []
    for doc in dataset.corpus.values():
        chunks.extend(chunk(doc, chunk_size, chunk_overlap))
    retriever.index(chunks)

    run_ranked: dict[str, list[str]] = {}
    for query_id, query_text in dataset.queries.items():
        results = retriever.search(query_text, k_search)
        run_ranked[query_id] = [doc_id for doc_id, _ in results]

    metric_result = evaluate(run_ranked, dataset.qrels)
    params: dict[str, Any] = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        **retriever_params,
    }
    return RunResult(
        run_id=uuid.uuid4().hex,
        dataset=dataset.name,
        retriever=retriever_name,
        params=params,
        created_at=datetime.now(UTC),
        git_sha=git_sha(),
        n_queries=len(dataset.queries),
        metrics=metric_result,
    )


def connect(db_path: Path = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Open (creating if needed) the results database, with schema ensured."""
    con = duckdb.connect(str(db_path))
    ensure_schema(con)
    return con


def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id VARCHAR PRIMARY KEY,
            created_at TIMESTAMP,
            dataset VARCHAR,
            retriever VARCHAR,
            params_json VARCHAR,
            git_sha VARCHAR,
            n_queries INTEGER
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS run_metrics (
            run_id VARCHAR,
            metric VARCHAR,
            value DOUBLE
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS query_metrics (
            run_id VARCHAR,
            query_id VARCHAR,
            metric VARCHAR,
            value DOUBLE
        )
        """
    )


def save_run(con: duckdb.DuckDBPyConnection, result: RunResult) -> None:
    """Persist a run's metadata, mean metrics, and per-query metrics."""
    con.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            result.run_id,
            result.created_at,
            result.dataset,
            result.retriever,
            json.dumps(result.params, sort_keys=True),
            result.git_sha,
            result.n_queries,
        ],
    )
    for metric, value in result.metrics.mean.items():
        con.execute("INSERT INTO run_metrics VALUES (?, ?, ?)", [result.run_id, metric, value])
    for query_id, values in result.metrics.per_query.items():
        for metric, value in values.items():
            con.execute(
                "INSERT INTO query_metrics VALUES (?, ?, ?, ?)",
                [result.run_id, query_id, metric, value],
            )


def fetch_runs(con: duckdb.DuckDBPyConnection, dataset: str | None = None) -> list[RunSummary]:
    """Read back recorded runs (optionally filtered by dataset), with their mean metrics."""
    if dataset is not None:
        rows = con.execute(
            "SELECT run_id, created_at, dataset, retriever, params_json, n_queries "
            "FROM runs WHERE dataset = ? ORDER BY created_at",
            [dataset],
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT run_id, created_at, dataset, retriever, params_json, n_queries "
            "FROM runs ORDER BY created_at"
        ).fetchall()

    summaries: list[RunSummary] = []
    for run_id, created_at, ds, retriever, params_json, n_queries in rows:
        metric_rows = con.execute(
            "SELECT metric, value FROM run_metrics WHERE run_id = ?", [run_id]
        ).fetchall()
        summaries.append(
            RunSummary(
                run_id=run_id,
                created_at=created_at,
                dataset=ds,
                retriever=retriever,
                params=json.loads(params_json),
                n_queries=n_queries,
                metrics=dict(metric_rows),
            )
        )
    return summaries


def fetch_query_metrics(con: duckdb.DuckDBPyConnection, run_id: str) -> dict[str, dict[str, float]]:
    """Per-query metric values for one run, as query_id -> {metric: value}."""
    rows = con.execute(
        "SELECT query_id, metric, value FROM query_metrics WHERE run_id = ?", [run_id]
    ).fetchall()
    result: dict[str, dict[str, float]] = {}
    for query_id, metric, value in rows:
        result.setdefault(query_id, {})[metric] = value
    return result

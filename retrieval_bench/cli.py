"""``rbench``: download datasets, run retrievers, and compare recorded results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from retrieval_bench import datasets
from retrieval_bench.retrievers.base import Retriever
from retrieval_bench.retrievers.bm25 import BM25Retriever
from retrieval_bench.retrievers.dense import DEFAULT_MODEL_NAME, DenseRetriever
from retrieval_bench.retrievers.hybrid import HybridRetriever
from retrieval_bench.runs import (
    DEFAULT_DB_PATH,
    connect,
    fetch_query_metrics,
    fetch_runs,
    run,
    save_run,
)


def _build_retriever(
    args: argparse.Namespace, dataset: datasets.Dataset
) -> tuple[Retriever, dict[str, Any]]:
    chunk_params = {"chunk_size": args.chunk_size, "chunk_overlap": args.overlap}

    if args.retriever == "bm25":
        return BM25Retriever(k1=args.k1, b=args.b), {"k1": args.k1, "b": args.b}

    if args.retriever == "dense":
        model_name = args.model or DEFAULT_MODEL_NAME
        dense = DenseRetriever(
            model_name=model_name,
            chunk_params=chunk_params,
            corpus_sha256=dataset.corpus_sha256,
        )
        return dense, {"model": model_name}

    if args.retriever == "hybrid":
        model_name = args.model or DEFAULT_MODEL_NAME
        bm25 = BM25Retriever(k1=args.k1, b=args.b)
        dense = DenseRetriever(
            model_name=model_name,
            chunk_params=chunk_params,
            corpus_sha256=dataset.corpus_sha256,
        )
        hybrid = HybridRetriever(bm25, dense, rrf_k=args.rrf_k)
        params = {"k1": args.k1, "b": args.b, "model": model_name, "rrf_k": args.rrf_k}
        return hybrid, params

    raise ValueError(f"unknown retriever {args.retriever!r}")


def cmd_download(args: argparse.Namespace) -> int:
    path = datasets.download(args.dataset)
    print(f"downloaded {args.dataset} to {path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    dataset = datasets.load(args.dataset)
    retriever, retriever_params = _build_retriever(args, dataset)
    result = run(
        dataset,
        retriever,
        retriever_name=args.retriever,
        retriever_params=retriever_params,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
    )
    con = connect(Path(args.db))
    try:
        save_run(con, result)
    finally:
        con.close()

    print(f"run_id={result.run_id}")
    for metric, value in sorted(result.metrics.mean.items()):
        print(f"  {metric}: {value:.4f}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    con = connect(Path(args.db))
    try:
        summaries = fetch_runs(con, dataset=args.dataset)
    finally:
        con.close()

    if not summaries:
        print("no runs recorded yet")
        return 0

    metric_names = sorted({m for s in summaries for m in s.metrics})
    headers = ["run_id", "dataset", "retriever", "params", *metric_names]
    rows: list[list[str]] = []
    for s in summaries:
        row = [
            s.run_id,
            s.dataset,
            s.retriever,
            json.dumps(s.params, sort_keys=True),
            *[f"{s.metrics.get(m, float('nan')):.4f}" for m in metric_names],
        ]
        rows.append(row)

    if args.markdown:
        print("| " + " | ".join(headers) + " |")
        print("| " + " | ".join("---" for _ in headers) + " |")
        for row in rows:
            print("| " + " | ".join(row) + " |")
    else:
        widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
        print("  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)))
        for row in rows:
            print("  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)))
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    con = connect(Path(args.db))
    try:
        metrics_a = fetch_query_metrics(con, args.run_a)
        metrics_b = fetch_query_metrics(con, args.run_b)
    finally:
        con.close()

    query_ids = sorted(set(metrics_a) | set(metrics_b))
    diffs: list[tuple[str, float, float, float]] = []
    for qid in query_ids:
        value_a = metrics_a.get(qid, {}).get(args.metric)
        value_b = metrics_b.get(qid, {}).get(args.metric)
        if value_a is None or value_b is None:
            continue
        diffs.append((qid, value_a, value_b, value_b - value_a))

    diffs.sort(key=lambda item: abs(item[3]), reverse=True)
    print(f"{'query_id':<20}{'a':>10}{'b':>10}{'delta':>10}")
    for qid, value_a, value_b, delta in diffs[: args.top]:
        print(f"{qid:<20}{value_a:>10.4f}{value_b:>10.4f}{delta:>+10.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rbench",
        description="Compare BM25, dense, and hybrid retrieval on public IR benchmarks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_download = subparsers.add_parser("download", help="download a BEIR dataset")
    p_download.add_argument("dataset", choices=sorted(datasets.DATASET_REGISTRY))
    p_download.set_defaults(func=cmd_download)

    p_run = subparsers.add_parser("run", help="run a retriever over a dataset and record metrics")
    p_run.add_argument("--dataset", required=True, choices=sorted(datasets.DATASET_REGISTRY))
    p_run.add_argument("--retriever", required=True, choices=["bm25", "dense", "hybrid"])
    p_run.add_argument("--model", default=None, help="sentence-transformers model name")
    p_run.add_argument("--chunk-size", type=int, default=None, dest="chunk_size")
    p_run.add_argument("--overlap", type=int, default=0)
    p_run.add_argument("--k1", type=float, default=0.9)
    p_run.add_argument("--b", type=float, default=0.4)
    p_run.add_argument("--rrf-k", type=int, default=60, dest="rrf_k")
    p_run.add_argument("--db", default=str(DEFAULT_DB_PATH))
    p_run.set_defaults(func=cmd_run)

    p_compare = subparsers.add_parser("compare", help="show recorded runs and their metrics")
    p_compare.add_argument("--dataset", default=None)
    p_compare.add_argument("--markdown", action="store_true")
    p_compare.add_argument("--db", default=str(DEFAULT_DB_PATH))
    p_compare.set_defaults(func=cmd_compare)

    p_diff = subparsers.add_parser("diff", help="compare two runs query by query")
    p_diff.add_argument("run_a")
    p_diff.add_argument("run_b")
    p_diff.add_argument("--metric", default="ndcg@10")
    p_diff.add_argument("--top", type=int, default=10)
    p_diff.add_argument("--db", default=str(DEFAULT_DB_PATH))
    p_diff.set_defaults(func=cmd_diff)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())

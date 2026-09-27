# retrieval-bench

[![CI](https://github.com/dkurokawa/retrieval-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/dkurokawa/retrieval-bench/actions/workflows/ci.yml)

Compare BM25, dense embedding, and hybrid retrieval on public IR benchmarks, under the same chunking and evaluation, and record every run so the numbers can be reproduced and diffed.

## Why

The quality of a RAG answer is capped by something that happens before generation: whether retrieval surfaced the right document at all. retrieval-bench runs keyword search (BM25), embedding search (dense), and their fusion (hybrid) over the same public, labeled datasets under matched conditions, and records recall / nDCG / MRR for each run. Changing the chunk size or the embedding model becomes a reproducible, diffable comparison instead of an anecdote.

This project only measures retrieval quality on public benchmark data. It does not evaluate generated answers, does not call any paid API, and is not tied to any specific product or business domain.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                    # BM25 + core tooling only
uv sync --extra dense       # also installs sentence-transformers, for dense/hybrid
```

## Usage

```bash
# Download a dataset into ~/.cache/retrieval-bench/ (override with RBENCH_CACHE)
uv run rbench download scifact

# Run a retriever and record recall@10, recall@100, nDCG@10, MRR@10
# (only queries with a qrels entry in the loaded split are searched/scored)
uv run rbench run --dataset scifact --retriever bm25
uv run rbench run --dataset scifact --retriever dense
uv run rbench run --dataset scifact --retriever hybrid --k1 0.9 --b 0.4 --rrf-k 60 --rrf-depth 1000

# Chunk the corpus before indexing (words per chunk, word overlap)
uv run rbench run --dataset scifact --retriever dense --chunk-size 64 --overlap 16

# List every recorded run for a dataset as a table
uv run rbench compare --dataset scifact
uv run rbench compare --dataset scifact --markdown

# Compare two runs query by query, largest metric delta first (error analysis)
uv run rbench diff <run_id_a> <run_id_b> --metric ndcg@10 --top 10
```

Every run is appended to a local DuckDB file (`results.duckdb` by default, override with `--db`) across three tables: `runs` (metadata, params, git commit, and a `meta_json` blob), `run_metrics` (mean per metric), and `query_metrics` (per-query values, for `diff`). `meta_json` records the corpus/queries/qrels sha256, the split, the dense encoder's identifier (model name, sentence-transformers version, and model revision where obtainable), and the numpy/scipy/sentence-transformers versions used — enough to tell exactly what data and code produced a given run's numbers.

BM25's `search()` only returns documents with a score > 0 (no shared term with the query is not evidence of relevance). Hybrid's Reciprocal Rank Fusion only fuses each sub-retriever's top `--rrf-depth` candidates (default 1000), not the whole corpus — a document neither retriever surfaces within that depth contributes nothing, matching how RRF is actually used at scale rather than a full-corpus fusion that happens to be affordable on small benchmarks. `rbench run` only searches and scores queries that have at least one qrels judgment in the loaded split (a shared `queries.jsonl` can otherwise include queries from other splits with no judgments here at all); `n_queries` is the number that actually contributed to the reported means, not every query in the file.

## Datasets

Datasets are **not** included in this repository; `rbench download` fetches them into a local cache. Both are [BEIR](https://github.com/beir-cellar/beir)-format IR benchmarks (corpus + queries + graded qrels), mirrored by the BEIR project (code: Apache-2.0) at `public.ukp.informatik.tu-darmstadt.de`.

| Dataset  | Task                                 | Source / citation                                                                                                          | License (per source)                                                                    |
| -------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| scifact  | Scientific claim verification retrieval | Wadden et al., *Fact or Fiction: Verifying Scientific Claims*, 2020 ([allenai/scifact](https://github.com/allenai/scifact)) | Claims and annotations: CC BY 4.0. Abstracts (corpus): ODC-By 1.0, from S2ORC. Per the [original LICENSE](https://github.com/allenai/scifact/blob/master/LICENSE.md) |
| nfcorpus | Medical/nutrition information retrieval | Boteva et al., *A Full-Text Learning to Rank Dataset for Medical Information Retrieval*, 2016 ([project page](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)) | Free for academic use; other uses of the NutritionFacts.org content require the author's terms. Per the project page |

The license column follows each dataset's original distributor, checked 2026-09-27. The BEIR re-hosts on HuggingFace list a different license (CC-BY-SA-4.0) on their cards; the original terms above take precedence. This repository only downloads the data for local evaluation and never redistributes it.

## Results (SciFact)

Measured with `rbench run` against the `test` split (300 of SciFact's 1109 queries have a qrels judgment there; only those 300 are searched and scored — see `n_queries` above), dense encoder `sentence-transformers/all-MiniLM-L6-v2` (mean-pooled, L2-normalized), BM25 `k1=0.9, b=0.4`, hybrid RRF `k=60, depth=1000`. Document score = max over its chunks (max-pooling); BM25 only counts documents with score > 0.

| Retriever | Chunking                | recall@10 | recall@100 | nDCG@10 | MRR@10 |
| --------- | ------------------------ | --------- | ---------- | ------- | ------ |
| BM25      | whole document (no split) | 0.7784    | 0.8852     | 0.6611  | 0.6304 |
| Dense     | whole document (no split) | 0.7833    | 0.9250     | 0.6451  | 0.6047 |
| Dense     | 64 words / 16 overlap    | 0.8180    | 0.9377     | 0.6749  | 0.6334 |
| Dense     | 256 words / 32 overlap   | 0.7943    | 0.9250     | 0.6500  | 0.6065 |
| Hybrid (BM25 + Dense) | whole document (no split) | 0.8272 | 0.9617 | 0.6937 | 0.6563 |

On this dataset, hybrid beats both single-method baselines on every metric, and for the dense retriever, chunking to 64 words with 16-word overlap outperforms indexing whole documents or 256-word chunks — i.e. the chunk-size/overlap choice measurably matters here, not just the retrieval method. The hybrid numbers above are marginally higher than in an earlier measurement of this same setup, because BM25 no longer contributes phantom zero-score "candidates" to the RRF fusion — real (if weak) BM25 matches now occupy the rank positions those used to crowd, and Hybrid's own recall@10/nDCG@10/MRR@10 (not recall@100) shift up by 0.0003–0.0033 as a result. BM25 and Dense's own numbers are unchanged from that earlier measurement: BM25's score-filtering only ever drops documents that couldn't have been legitimate matches anyway, and restricting search to the 300 qrels-judged queries (rather than searching all 1109 and only scoring 300) doesn't change what gets averaged, only how much gets computed.

**Measurement environment**: Apple M4 (CPU only, no GPU), macOS 26.2 (Darwin 25.2.0), Python 3.11.14, sentence-transformers 6.1.0. Measured 2026-09-27, commit `0a52054`.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict retrieval_bench
uv run pytest --cov=retrieval_bench --cov-fail-under=90
```

## What this is not

Not a PyPI-published package. Not an evaluation of LLM-generated answers — only retrieval is measured. Not a wrapper around any paid embedding API. Not tied to any specific product, business domain, or organization.

## License

MIT — see [LICENSE](LICENSE). Dataset licenses are separate; see the table above.

---

## 日本語要約

retrieval-bench は、RAG の検索段階（BM25 によるキーワード検索・埋め込みによる密検索・両者の融合であるハイブリッド検索）を、正解付きの公開ベンチマークデータ（BEIR 形式の SciFact・NFCorpus）で同じ条件で走らせ、recall・nDCG・MRR を記録して比較するツールです。データセット自体はリポジトリに含めず、`rbench download` で都度取得します。チャンクの長さや埋め込みモデルを変えたときの精度の変化を、DuckDB に記録した run 同士の比較（`rbench compare` / `rbench diff`）で再現可能な形にすることが目的で、生成モデルの回答評価や特定の業務・製品への言及は含みません。

上の結果表は SciFact で実際に `rbench run` を実行して得た数値です（Apple M4 / CPU のみ / macOS 26.2、2026-09-27 計測、commit `0a52054`）。検索・採点の対象は、SciFact の質問 1109 件のうち test split の qrels に正解が載っている 300 件のみです。このデータでは、BM25・Dense 単体よりも Hybrid が全指標で上回り、Dense はチャンク長を 64 語（重なり16語）にした場合が、無分割や 256 語チャンクより高い精度になりました。

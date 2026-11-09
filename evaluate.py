"""Evaluate one run file against one qrels file.

Example:
    python -m retrieval_bench.evaluate \
        --qrels data/toy_qrels.jsonl --run data/toy_run.bm25.jsonl \
        --metrics recall@10,ndcg@10 --per-query per_query.json

Queries present in the qrels but missing from the run are reported as
``missing`` and skipped instead of being scored as zero recall, which
keeps partial exports visible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from retrieval_bench.dataset import DatasetError, load_qrels, load_run
from retrieval_bench.metrics import METRICS, parse_metric


def evaluate(
    qrels: dict[str, dict[str, float]],
    run: dict[str, list[str]],
    specs: list[tuple[str, int]],
) -> tuple[dict[str, float], dict[str, dict[str, float]], int]:
    """Return mean metrics, per-query metrics, and the number of missing queries."""
    shared = sorted(set(qrels) & set(run))
    per_query: dict[str, dict[str, float]] = {}
    for query_id in shared:
        gains = qrels[query_id]
        ranked = run[query_id]
        per_query[query_id] = {
            f"{name}@{cutoff}": METRICS[name](ranked, gains, cutoff)
            for name, cutoff in specs
        }
    means: dict[str, float] = {}
    for name, cutoff in specs:
        key = f"{name}@{cutoff}"
        values = [scores[key] for scores in per_query.values()]
        means[key] = sum(values) / len(values) if values else 0.0
    return means, per_query, len(set(qrels) - set(run))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--metrics", default="recall@10,ndcg@10",
                        help="comma separated name@k specs")
    parser.add_argument("--per-query", type=Path,
                        help="write per-query scores to this path")
    parser.add_argument("--label", default="",
                        help="system label recorded in the output")
    args = parser.parse_args(argv)

    try:
        specs = [parse_metric(spec) for spec in args.metrics.split(",") if spec]
        qrels = load_qrels(args.qrels)
        run = load_run(args.run)
    except (DatasetError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    means, per_query, missing = evaluate(qrels, run, specs)
    if args.per_query:
        args.per_query.write_text(json.dumps(per_query, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")
    report = {
        "label": args.label,
        "queries": len(per_query),
        "missing": missing,
        "metrics": means,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

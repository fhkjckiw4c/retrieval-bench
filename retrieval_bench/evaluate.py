"""Evaluate one run file against one qrels file.

Example:
    python -m retrieval_bench.evaluate \
        --qrels data/toy_qrels.jsonl --run data/toy_run.bm25.jsonl \
        --metrics recall@10,ndcg@10 --per-query per_query.json

Queries present in the qrels but missing from the run are reported as
``missing`` and skipped instead of being scored as zero recall, which
keeps partial exports visible. ``--bootstrap`` adds a seeded confidence
interval per metric and ``--report`` writes a Markdown summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from retrieval_bench.dataset import DatasetError, load_qrels, load_run
from retrieval_bench.metrics import METRICS, parse_metric
from retrieval_bench.reporting import bootstrap, markdown


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
    parser.add_argument("--bootstrap", type=int, default=0, metavar="N",
                        help="bootstrap resamples per metric (0 disables)")
    parser.add_argument("--seed", type=int, default=0,
                        help="resampling seed, recorded for reproduction")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--report", type=Path,
                        help="write a Markdown summary to this path")
    args = parser.parse_args(argv)

    try:
        specs = [parse_metric(spec) for spec in args.metrics.split(",") if spec]
        qrels = load_qrels(args.qrels)
        run = load_run(args.run)
    except (DatasetError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    means, per_query, missing = evaluate(qrels, run, specs)

    intervals: dict[str, tuple[float, float]] = {}
    if args.bootstrap:
        for key in means:
            values = [scores[key] for scores in per_query.values()]
            _, low, high = bootstrap.bootstrap_ci(
                values,
                n_resamples=args.bootstrap,
                confidence=args.confidence,
                seed=args.seed,
            )
            intervals[key] = (low, high)

    if args.per_query:
        args.per_query.write_text(json.dumps(per_query, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")

    report = {
        "label": args.label,
        "queries": len(per_query),
        "missing": missing,
        "metrics": means,
    }
    if intervals:
        report["intervals"] = {key: list(ci) for key, ci in intervals.items()}
        report["bootstrap"] = {"resamples": args.bootstrap, "seed": args.seed,
                              "confidence": args.confidence}

    if args.report:
        args.report.write_text(
            markdown.render_report(args.label, means, intervals,
                                   queries=len(per_query), missing=missing,
                                   confidence=args.confidence),
            encoding="utf-8",
        )

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

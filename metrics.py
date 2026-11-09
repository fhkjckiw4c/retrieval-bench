"""Cutoff-aware retrieval metrics for a single ranked list.

Every metric takes the ranked doc ids for one query and the qrels gains
for that query. Duplicate doc ids are collapsed to their first occurrence
so that sloppy runs stay scoreable instead of silently reweighted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Gains = dict[str, float]


def _dedupe(ranked: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for doc_id in ranked:
        if doc_id not in seen:
            seen.add(doc_id)
            unique.append(doc_id)
    return unique


def _gain(doc_id: str, gains: Gains) -> float:
    return float(gains.get(doc_id, 0.0))


def recall_at_k(ranked: Sequence[str], gains: Gains, k: int) -> float:
    """Fraction of relevant docs retrieved in the top k."""
    if not gains:
        return 0.0
    hits = sum(1 for doc_id in _dedupe(ranked)[:k] if doc_id in gains)
    return hits / len(gains)


def mrr_at_k(ranked: Sequence[str], gains: Gains, k: int) -> float:
    """Reciprocal rank of the first relevant doc within the top k."""
    for rank, doc_id in enumerate(_dedupe(ranked)[:k], start=1):
        if doc_id in gains:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gains: Gains, k: int) -> float:
    """Graded nDCG@k; qrels relevance values are used as gains."""
    dcg = 0.0
    for rank, doc_id in enumerate(_dedupe(ranked)[:k], start=1):
        dcg += _gain(doc_id, gains) / math.log2(rank + 1)
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


METRICS = {
    "recall": recall_at_k,
    "mrr": mrr_at_k,
    "ndcg": ndcg_at_k,
}


def parse_metric(spec: str) -> tuple[str, int]:
    """Parse ``name@k`` into a metric function name and a cutoff."""
    name, _, cutoff = spec.partition("@")
    if name not in METRICS:
        raise ValueError(f"unknown metric {name!r}; expected one of {sorted(METRICS)}")
    if not cutoff.isdigit() or int(cutoff) <= 0:
        raise ValueError(f"metric {spec!r} needs a positive integer cutoff")
    return name, int(cutoff)

"""Seeded bootstrap confidence intervals over per-query scores.

The harness reports a macro average over queries, which says nothing
about how much that average would move under a different query sample.
Resampling queries with replacement gives a distribution of the mean
without assumptions about the metric's shape. The seed is explicit so a
reported interval can be recomputed.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _quantile(ordered: Sequence[float], q: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return ``(mean, low, high)`` for a percentile bootstrap interval.

    An empty input yields zeros so single-query runs stay printable.
    """
    if not values:
        return 0.0, 0.0, 0.0
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    rng = random.Random(seed)
    size = len(values)
    means = sorted(
        _mean([values[rng.randrange(size)] for _ in range(size)])
        for _ in range(n_resamples)
    )
    tail = (1.0 - confidence) / 2.0
    return _mean(values), _quantile(means, tail), _quantile(means, 1.0 - tail)

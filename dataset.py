"""JSONL loaders for qrels and run files.

Qrels rows: ``{"query_id": ..., "doc_id": ..., "relevance": <float>}``.
Run rows:   ``{"query_id": ..., "doc_id": ..., "score": <float>}``.

Relevance values at or below zero are treated as judged-not-relevant.
Rows are validated eagerly so a typo in a large export fails at line
granularity instead of producing quietly wrong numbers.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path


class DatasetError(ValueError):
    """Raised for malformed or inconsistent qrels/run files."""


def _rows(path: Path) -> Iterator[tuple[int, dict]]:
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{lineno}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise DatasetError(f"{path}:{lineno}: expected a JSON object")
            yield lineno, row


def _field(row: dict, key: str, path: Path, lineno: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise DatasetError(f"{path}:{lineno}: missing string field {key!r}")
    return value


def _number(row: dict, key: str, path: Path, lineno: int) -> float:
    value = row.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DatasetError(f"{path}:{lineno}: missing numeric field {key!r}")
    return float(value)


def load_qrels(path: Path) -> dict[str, dict[str, float]]:
    """Return ``query_id -> {doc_id: relevance}`` for positive judgements."""
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    for lineno, row in _rows(path):
        query_id = _field(row, "query_id", path, lineno)
        doc_id = _field(row, "doc_id", path, lineno)
        relevance = _number(row, "relevance", path, lineno)
        if relevance > 0:
            qrels[query_id][doc_id] = relevance
    if not qrels:
        raise DatasetError(f"{path}: no positive relevance judgements")
    return dict(qrels)


def load_run(path: Path) -> dict[str, list[str]]:
    """Return ``query_id -> doc ids`` ordered by descending score."""
    scored: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for lineno, row in _rows(path):
        query_id = _field(row, "query_id", path, lineno)
        doc_id = _field(row, "doc_id", path, lineno)
        scored[query_id].append((_number(row, "score", path, lineno), doc_id))
    run: dict[str, list[str]] = {}
    for query_id, pairs in scored.items():
        pairs.sort(key=lambda pair: (-pair[0], pair[1]))
        run[query_id] = [doc_id for _, doc_id in pairs]
    return run

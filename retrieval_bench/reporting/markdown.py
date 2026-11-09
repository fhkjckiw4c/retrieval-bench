"""Markdown rendering of an evaluation summary.

Kept separate from the CLI so the same table can be produced from a saved
JSON report. Output is plain Markdown with no formatting beyond a table.
"""

from __future__ import annotations


def render_report(
    label: str,
    metrics: dict[str, float],
    intervals: dict[str, tuple[float, float]],
    *,
    queries: int,
    missing: int,
    confidence: float = 0.95,
) -> str:
    """Render a summary table plus a short provenance header."""
    title = label or "unnamed run"
    lines = [f"## Run `{title}`", "", f"{queries} scored queries, {missing} missing.", ""]
    if intervals:
        level = f"{confidence * 100:.0f}%"
        lines += [f"| metric | mean | {level} CI |", "|--------|------|---------|"]
        for name in sorted(metrics):
            low, high = intervals[name]
            lines.append(f"| {name} | {metrics[name]:.4f} | {low:.4f} - {high:.4f} |")
    else:
        lines += ["| metric | mean |", "|--------|------|"]
        for name in sorted(metrics):
            lines.append(f"| {name} | {metrics[name]:.4f} |")
    lines += ["", "Intervals are percentile bootstraps over queries, not over docs.", ""]
    return "\n".join(lines)

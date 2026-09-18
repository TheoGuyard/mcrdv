#!/usr/bin/env python3
"""Figures for the baseline comparison."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from common import export, plotting, tidy  # noqa: E402


def _per_planner(data, column):
    """Median of `column` for each planner, within each instance group."""
    planners_seen = list(
        dict.fromkeys(row.get("sequence.type") for row in data.rows)
    )
    series = {}
    for planner in planners_seen:
        series[planner] = [
            tidy.spread(
                [r for r in group if r.get("sequence.type") == planner], column
            )[0]
            for group in data.groups
        ]
    return series


def gap(ax, data):
    """Cost of each planner per instance, relative to the best plan found."""
    series = _per_planner(data, "cost")
    best = [
        min(values[i] for values in series.values() if np.isfinite(values[i]))
        for i in range(len(data.labels))
    ]
    gaps = {
        planner: [100.0 * (v / b - 1.0) for v, b in zip(values, best)]
        for planner, values in series.items()
    }
    export.grouped_bars(ax, data.labels, gaps)


def usage(ax, data):
    """How many chasers each planner actually puts to work."""
    export.grouped_bars(ax, data.labels, _per_planner(data, "active_chasers"))


FIGURES = {"gap": gap, "usage": usage}


if __name__ == "__main__":
    raise SystemExit(plotting.main(__file__, FIGURES))

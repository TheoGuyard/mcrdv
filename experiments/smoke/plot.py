#!/usr/bin/env python3
"""Figures for the smoke check."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import export, plotting  # noqa: E402


def cost(ax, data):
    """Cost against catalogue size, over whatever replications exist."""
    export.band(ax, data.x, data.lo, data.hi, name="iqr")
    export.line(ax, data.x, data.median, name="cost", marker="o")


def by_planner(ax, data):
    """Cost per planner. The grouping axis is categorical, so: bars."""
    export.grouped_bars(ax, data.labels, {"cost": data.median})


FIGURES = {"cost": cost, "by_planner": by_planner}


if __name__ == "__main__":
    raise SystemExit(plotting.main(__file__, FIGURES))

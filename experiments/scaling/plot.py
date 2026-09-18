#!/usr/bin/env python3
"""Figures for the scaling experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import export, plotting, tidy  # noqa: E402


def cost(ax, data):
    """Mission cost against catalogue size."""
    export.band(ax, data.x, data.lo, data.hi, name="cost_iqr")
    export.line(ax, data.x, data.median, name="cost", marker="o")


def time(ax, data):
    """Where the wall clock goes as the catalogue grows.

    `plot.yaml` reduced `t_setup` for us; the search time is read off the
    same groups, so both curves come from one selection rather than two.
    """
    export.band(ax, data.x, data.lo, data.hi, name="setup_iqr")
    export.line(ax, data.x, data.median, name="setup", marker="o")

    solve = [tidy.spread(group, "t_solve") for group in data.groups]
    export.band(
        ax,
        data.x,
        [s[1] for s in solve],
        [s[2] for s in solve],
        name="solve_iqr",
    )
    export.line(ax, data.x, [s[0] for s in solve], name="solve", marker="s")


FIGURES = {"cost": cost, "time": time}


if __name__ == "__main__":
    raise SystemExit(plotting.main(__file__, FIGURES))

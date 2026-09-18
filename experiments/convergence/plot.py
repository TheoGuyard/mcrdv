#!/usr/bin/env python3
"""Figures for the convergence experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from common import export, plotting, tidy  # noqa: E402


def _segments(values, keep):
    """`values` with everything outside `keep` blanked out.

    Drawing one curve in two styles means drawing it twice, each copy holding
    NaN where the other takes over. The boundary point belongs to both, so the
    styles meet instead of leaving a gap.
    """
    out = np.where(keep, values, np.nan)
    edges = np.flatnonzero(keep[:-1] != keep[1:])
    for edge in edges:
        out[edge] = values[edge]
        out[edge + 1] = values[edge + 1]
    return out


def convergence(ax, data):
    """Median incumbent against time, with the spread across seeds.

    Each run logs on its own clock, so the traces are read onto one common grid
    before being reduced. They are staircases -- the incumbent holds until it
    improves -- so the resampling carries values forward rather than
    interpolating, which would invent improvements that never happened.

    The early part of a search has no feasible plan at all, and its "cost" is
    the cost of something that overruns a limit -- not comparable with what
    comes later. That stretch is drawn dashed. The line only goes solid once
    *every* replication has found a feasible plan: it summarises all of them, so
    calling it feasible while some are not is the misleading direction.
    """
    runs = [r for r in data.records if r.result.get("trace")]
    if not runs:
        return

    curves = [
        (
            [e["time"] for e in r.result["trace"]],
            [e["cost"] for e in r.result["trace"]],
            [float(e["feasible"]) for e in r.result["trace"]],
        )
        for r in runs
    ]

    # Stop at the first run's end, so every point of the band is backed by every
    # seed rather than thinning out as the longer runs carry on alone.
    horizon = min(max(times) for times, _, _ in curves)
    grid = np.linspace(0.0, horizon, 200)
    cost = np.vstack([tidy.resample(t, c, grid) for t, c, _ in curves])
    feasible = np.vstack([tidy.resample(t, f, grid) for t, _, f in curves])

    usable = ~np.isnan(cost).any(axis=0)
    grid, cost, feasible = grid[usable], cost[:, usable], feasible[:, usable]
    if grid.size == 0:
        return

    median = np.median(cost, axis=0)
    everywhere = np.nanmin(feasible, axis=0) > 0.0

    export.band(
        ax,
        grid,
        np.percentile(cost, 25, axis=0),
        np.percentile(cost, 75, axis=0),
        name="iqr",
    )
    if not everywhere.all():
        export.line(
            ax,
            grid,
            _segments(median, ~everywhere),
            name="infeasible",
            linestyle="--",
        )
    if everywhere.any():
        export.line(ax, grid, _segments(median, everywhere), name="feasible")

    ax.set_title(f"{len(curves)} seeds")


# Every figure in this experiment is the same drawing over a different
# selection, which the configuration already expresses.
FIGURES = {"iridium33": convergence, "cosmos2251": convergence}


if __name__ == "__main__":
    raise SystemExit(plotting.main(__file__, FIGURES))

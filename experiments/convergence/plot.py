#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from _common import plotter  # noqa: E402


def _segments(values, keep):
    out = np.where(keep, values, np.nan)
    edges = np.flatnonzero(keep[:-1] != keep[1:])
    for edge in edges:
        out[edge] = values[edge]
        out[edge + 1] = values[edge + 1]
    return out


def convergence(ax, data):
    records = [record for record in data.records if record.result.get("trace")]
    if not records:
        return

    curves = [
        (
            [log["time"] for log in record.result["trace"]],
            [log["cost"] for log in record.result["trace"]],
            [float(log["feasible"]) for log in record.result["trace"]],
        )
        for record in records
    ]

    horizon = min(max(times) for times, _, _ in curves)
    grid = np.linspace(0.0, horizon, 200)
    cost = np.vstack([plotter.resample(t, c, grid) for t, c, _ in curves])
    feas = np.vstack([plotter.resample(t, f, grid) for t, _, f in curves])

    usable = ~np.isnan(cost).any(axis=0)
    grid, cost, feas = grid[usable], cost[:, usable], feas[:, usable]
    if grid.size == 0:
        return

    median = np.median(cost, axis=0)
    isfeas = np.nanmin(feas, axis=0) > 0.0

    plotter.band(
        ax,
        grid,
        np.percentile(cost, 25, axis=0),
        np.percentile(cost, 75, axis=0),
        name="iqr",
    )

    if not isfeas.all():
        plotter.line(
            ax,
            grid,
            _segments(median, ~isfeas),
            name="infeasible",
            linestyle="--",
        )
    if isfeas.any():
        plotter.line(
            ax, 
            grid, 
            _segments(median, isfeas), 
            name="feasible"
        )

    ax.set_title(f"{len(curves)} seeds")


DRAW_MAP = {"convergence": convergence}


if __name__ == "__main__":
    raise SystemExit(plotter.main(__file__, DRAW_MAP))

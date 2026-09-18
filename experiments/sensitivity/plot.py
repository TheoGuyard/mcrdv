#!/usr/bin/env python3
"""Figures for the sensitivity sweeps: one per swept axis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from common import export, plotting, tidy  # noqa: E402


def sweep(ax, data):
    """One swept knob: cost against its values, relative to the best of them.

    Numeric axes become a curve with an interquartile band; categorical ones
    become bars. The absolute costs are the same across every axis, so what is
    worth seeing is which setting wins and by how much.
    """
    best = np.nanmin(data.median)
    gap = 100.0 * (data.median / best - 1.0)

    if data.numeric:
        export.band(
            ax,
            data.x,
            100.0 * (data.lo / best - 1.0),
            100.0 * (data.hi / best - 1.0),
            name="iqr",
        )
        export.line(ax, data.x, gap, name="gap", marker="o")
    else:
        export.grouped_bars(ax, data.labels, {"gap": gap})


def cost_vs_time(ax, data):
    """Every setting at once, as what it costs against what it takes.

    Knobs that only trade time for quality land on a common front; the ones that
    genuinely help sit below it. Easier to see in one scatter than in eight
    separate sweeps.
    """
    for axis in dict.fromkeys(row.get("sweep.axis") for row in data.rows):
        if axis is None:
            continue
        members = [r for r in data.rows if r.get("sweep.axis") == axis]
        values = list(dict.fromkeys(r.get("sweep.value") for r in members))
        points = [
            (
                tidy.spread(
                    [r for r in members if r.get("sweep.value") == v],
                    "t_solve",
                )[0],
                tidy.spread(
                    [r for r in members if r.get("sweep.value") == v], "cost"
                )[0],
            )
            for v in values
        ]
        export.points(
            ax,
            [p[0] for p in points],
            [p[1] for p in points],
            name=str(axis).rsplit(".", 1)[-1],
        )
    ax.legend(ncol=2)


FIGURES = {
    name: sweep
    for name in (
        "crossover",
        "ordering",
        "generation",
        "nb_neighbors",
        "pop_max",
        "grid_size",
        "use_hints",
        "allow_waiting",
    )
}
FIGURES["cost_vs_time"] = cost_vs_time


if __name__ == "__main__":
    raise SystemExit(plotting.main(__file__, FIGURES))

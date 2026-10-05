#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from _common import export, plotting, tidy  # noqa: E402

# How each method is drawn and what its columns are called. pgfplots column
# names must stay plain, so the `+` and `-` of the method names stay on screen.
METHODS = {
    "uniform": ("uniform", "uniform grid"),
    "uniform+hints": ("hints", "uniform grid + hints"),
}
NO_WAIT = "no-wait"

# Where the swept grid size sits in a flattened record.
GRID = "solver.args.schedule.args.grid_size"


def _grids(rows, method):
    members = [r for r in rows if r.get("method") == method]
    sizes = sorted({int(r[GRID]) for r in members})
    return sizes, [
        [r for r in members if int(r[GRID]) == size] for size in sizes
    ]


def _no_wait(rows, column):
    return tidy.spread([r for r in rows if r.get("method") == NO_WAIT], column)


def pareto(ax, data):
    for method, (name, label) in METHODS.items():
        sizes, groups = _grids(data.rows, method)
        if not sizes:
            continue
        calls = [1e6 * tidy.spread(g, "t_call")[0] for g in groups]
        gaps = [tidy.spread(g, "gap_mean")[0] for g in groups]
        marks = sizes if method == "uniform+hints" else None
        export.line(
            ax,
            calls,
            gaps,
            name=name,
            label=label,
            marker="o",
            annotate=marks,
        )
        if marks is None:
            continue
        for size, x, y in zip(sizes, calls, gaps):
            ax.annotate(
                str(size),
                (x, y),
                textcoords="offset points",
                xytext=(5, -9),
                fontsize=6,
            )

    call = _no_wait(data.rows, "t_call")[0]
    centre = _no_wait(data.rows, "gap_mean")[0]
    if np.isfinite(call) and np.isfinite(centre):
        export.points(
            ax,
            [1e6 * call],
            [centre],
            name="nowait",
            label=NO_WAIT,
            marker="x",
        )


def gap(ax, data):
    drawn = []
    for method, (name, label) in METHODS.items():
        sizes, groups = _grids(data.rows, method)
        if not sizes:
            continue
        spread = [tidy.spread(g, "gap_mean") for g in groups]
        export.band(
            ax,
            sizes,
            [s[1] for s in spread],
            [s[2] for s in spread],
            name=f"{name}_iqr",
        )
        export.line(
            ax,
            sizes,
            [s[0] for s in spread],
            name=name,
            label=label,
            marker="o",
        )
        drawn = sizes

    centre = _no_wait(data.rows, "gap_mean")[0]
    if np.isfinite(centre) and drawn:
        export.line(
            ax,
            drawn,
            [centre] * len(drawn),
            name="nowait",
            label=NO_WAIT,
            linestyle="--",
            color="0.4",
        )


def time(ax, data):
    for method, (name, label) in METHODS.items():
        sizes, groups = _grids(data.rows, method)
        if not sizes:
            continue
        export.line(
            ax,
            sizes,
            [1e6 * tidy.spread(g, "t_call")[0] for g in groups],
            name=f"{name}_call",
            label=f"call, {label}",
            marker="o",
        )
        export.line(
            ax,
            sizes,
            [1e6 * tidy.spread(g, "t_per_leg")[0] for g in groups],
            name=f"{name}_leg",
            label=f"new leg, {label}",
            marker="s",
            linestyle="--",
        )
    ax.legend(ncol=1)


def _per_sequence(records, key, method=None):
    legs, values = [], []
    for record in records:
        if method is not None and record.result.get("method") != method:
            continue
        sequences = record.result.get("sequences") or {}
        if key not in sequences:
            continue
        legs.append(np.asarray(sequences["legs"], dtype=float))
        values.append(np.asarray(sequences[key], dtype=float))
    if not legs:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(legs), np.concatenate(values)


def length(ax, data):
    for method, (name, label) in {
        **METHODS,
        NO_WAIT: ("nowait", NO_WAIT),
    }.items():
        legs, calls = _per_sequence(data.records, "t_call", method)
        if not legs.size:
            continue
        counts = np.unique(legs)
        export.line(
            ax,
            counts,
            [1e6 * np.median(calls[legs == n]) for n in counts],
            name=name,
            label=label,
            marker="o",
        )


DRAW_MAP = {
    "pareto": pareto,
    "gap": gap,
    "time": time,
    "length": length,
}


if __name__ == "__main__":
    raise SystemExit(plotting.main(__file__, DRAW_MAP))

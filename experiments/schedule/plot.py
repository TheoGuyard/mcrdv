#!/usr/bin/env python3
"""Figures for the schedule experiment."""

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
    """Grid sizes a method was run at, and its rows at each, smallest first."""
    members = [r for r in rows if r.get("method") == method]
    sizes = sorted({int(r[GRID]) for r in members})
    return sizes, [
        [r for r in members if int(r[GRID]) == size] for size in sizes
    ]


def _no_wait(rows, column):
    """Median of `column` over the no-wait runs: they ignore the grid, so
    every one of them measures the same thing."""
    return tidy.spread([r for r in rows if r.get("method") == NO_WAIT], column)


def pareto(ax, data):
    """Mean gap against time per call, one point per grid size.

    Both axes are medians over the seeds. A curve that sits lower at the same
    time is the better grid; one that bends flat is where refining it stops
    paying. The grid sizes are written along the default layer's curve only:
    the two curves nearly coincide, and labelling both makes neither legible.

    The sizes are handed to `export` as well as drawn, so the tikz figure
    writes them where matplotlib does. They are the one thing on this figure a
    reader cannot reconstruct from the curve itself.
    """
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
    """Mean gap against the grid size, with the spread over seeds.

    No-wait ignores the grid, so it is a flat dashed line across the sizes the
    other methods were run at -- the level the dynamic program starts from.
    """
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
    """Time per call over built tables, and time to tabulate a new leg.

    A call is dominated by the dynamic program once the legs are known; the
    first meeting of a leg costs a transfer evaluation per grid point, plus
    the scan that finds the hints when they are asked for.
    """
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
    """Per-sequence arrays of `key` and of the leg count, concatenated over the
    records given -- only those of `method` when one is named."""
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
    """Time per call against the number of legs, at one grid size.

    The dynamic program is linear in the legs for a fixed grid, so these should
    be straight lines; no-wait is the floor, a single pass with no table.
    """
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


def bound(ax, data):
    """Summed per-leg bound over the best schedule found, per length.

    One curve per instance, median and interquartile range over the bank. The
    ratio says how much of a plan's cost the screen can see without scheduling
    it; the dotted line at 1 is where the bound would stop being one.

    Every record selected is used. The bound and the reference do not depend
    on the layer measured, so `plot.yaml` should select a single method and
    grid size, or each sequence is counted once per configuration.
    """
    names = list(dict.fromkeys(r.get("instance.name") for r in data.rows))
    everything = []
    for name in names:
        records = [
            rec
            for rec, row in zip(data.records, data.rows)
            if row.get("instance.name") == name
        ]
        legs, bounds = _per_sequence(records, "bound")
        _, best = _per_sequence(records, "reference")
        usable = np.isfinite(best) & (best > 0)
        legs, ratio = legs[usable], bounds[usable] / best[usable]
        counts = np.unique(legs)
        if not counts.size:
            continue
        pick = [ratio[legs == n] for n in counts]
        export.band(
            ax,
            counts,
            [np.percentile(p, 25) for p in pick],
            [np.percentile(p, 75) for p in pick],
            name=f"{name}_iqr",
        )
        export.line(
            ax,
            counts,
            [np.median(p) for p in pick],
            name=name,
            marker="o",
        )
        everything.append(ratio)
    ax.axhline(1.0, color="0.4", linestyle=":", linewidth=1.0)
    if everything:
        above = float(np.mean(np.concatenate(everything) > 1.0 + 1e-9))
        ax.set_title(f"{100.0 * above:.1f}% of sequences above 1")


DRAW_MAP = {
    "pareto": pareto,
    "gap": gap,
    "time": time,
    "length": length,
    "bound": bound,
}


if __name__ == "__main__":
    raise SystemExit(plotting.main(__file__, DRAW_MAP))

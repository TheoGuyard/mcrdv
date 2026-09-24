#!/usr/bin/env python3
"""Figures for the sequence experiment.

Seven presets of the same search, and the question is what each one is worth:
how long it runs, and how good the plan it comes back with is. The presets are
not seven unrelated settings but a ladder -- deeper populations, more
iterations, more patience before giving up -- so the figures read as a curve
along it rather than as a comparison between rivals.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from _common import export, plotting, tidy  # noqa: E402


def _capped(rows):
    """Whether most of a group's runs were stopped by the clock."""
    flags = [bool(row.get("capped")) for row in rows]
    return bool(flags) and sum(flags) > len(flags) / 2


def _front(xs, ys):
    """The points nothing beats on both axes, cheapest effort first.

    A preset is dominated when another reaches a cost at least as low for no
    more effort, and dominated presets are exactly the ones not worth running.
    Nothing guarantees the ladder is free of them: preset 6 works a much larger
    population, so within a fixed budget it completes *fewer* iterations than
    preset 5 and still returns a better plan -- which leaves preset 5 below the
    front on the iteration axis and on it on the time axis.
    """
    order = sorted(range(len(xs)), key=lambda i: (xs[i], ys[i]))
    keep, best = [], math.inf
    for index in order:
        if ys[index] < best:
            keep.append(index)
            best = ys[index]
    return keep


def pareto(ax, data):
    """What the search spends, against the plan it comes back with.

    One point per preset -- the median of its runs on both axes -- with the
    individual runs behind them, so the spread of a stochastic search is
    visible rather than averaged away. The line is the Pareto front through
    them: a preset sitting above it is beaten outright by another that spends
    no more.

    What "spends" means is `x_value` in `plot.yaml`, so this one drawing is
    read twice. Against **search time** the question is which preset to run
    under a deadline, and the answer is where the front flattens -- an
    instance-dependent level rather than one the package can pick. Against
    **genetic iterations** the axis is the search's own clock, and the
    difference between the two figures is what a single iteration costs: the
    deep presets carry a far larger population, so they buy less per iteration
    and more per second.

    Presets 0 and 1 sit at zero iterations. That is not missing data: their
    iteration ceiling is zero, so they are construction heuristics that never
    enter the genetic loop, and the cost they reach is what the search starts
    from.

    A level marked `*` did not finish: its runs were stopped by the wall-clock
    budget, so its point says where the search had got to rather than what the
    preset is worth.
    """
    if not data.groups:
        return
    # What effort is measured in, which is the figure's own decision.
    column = str(data.spec.get("x_value", "t_solve"))

    # Every run behind the medians, one mark each.
    export.points(
        ax,
        [row.get(column) for group in data.groups for row in group],
        [row.get("cost") for group in data.groups for row in group],
        name="runs",
        color="0.75",
        s=9,
        zorder=2,
    )

    # Along the ladder, whatever order the folder happened to be read in.
    order = np.argsort(np.asarray(data.x, dtype=float))
    spent = [tidy.spread(data.groups[i], column)[0] for i in order]
    costs = [tidy.spread(data.groups[i], "cost")[0] for i in order]
    levels = [int(data.x[i]) for i in order]
    capped = [_capped(data.groups[i]) for i in order]

    front = _front(spent, costs)
    export.line(
        ax,
        [spent[i] for i in front],
        [costs[i] for i in front],
        name="front",
        label="front",
        color="0.2",
        drawstyle="steps-post",
        linewidth=1.0,
        zorder=3,
    )
    # The level beside each point, and the star saying the clock stopped it:
    # handed to `export` as well as drawn, so the tikz figure carries the
    # labels rather than a curve of anonymous markers.
    written = [
        f"{level}*" if hit else f"{level}"
        for level, hit in zip(levels, capped)
    ]
    export.points(
        ax,
        spent,
        costs,
        name="presets",
        label="presets",
        zorder=4,
        annotate=written,
    )
    for when, what, text in zip(spent, costs, written):
        ax.annotate(
            text,
            (when, what),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
        )
    if any(capped):
        ax.set_title(
            "* stopped by the time budget, not by the search",
            fontsize=8,
        )
    ax.legend(fontsize=7)


def cost(ax, data):
    """What each preset's plan costs, level by level."""
    export.band(ax, data.x, data.lo, data.hi, name="cost_iqr")
    export.line(ax, data.x, data.median, name="cost", marker="o")


def time(ax, data):
    """How long each preset searches for, level by level.

    The ladder is geometric -- each level multiplies the iteration ceiling and
    the population it works on -- so this is the axis on which a preset is
    chosen when there is a deadline rather than a quality target.
    """
    export.band(ax, data.x, data.lo, data.hi, name="time_iqr")
    export.line(ax, data.x, data.median, name="time", marker="o")


DRAW_MAP = {
    "pareto": pareto,
    "cost": cost,
    "time": time,
}


if __name__ == "__main__":
    raise SystemExit(plotting.main(__file__, DRAW_MAP))

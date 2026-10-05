#!/usr/bin/env python3
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from _common import export, plotting, tidy  # noqa: E402


def _capped(rows):
    flags = [bool(row.get("capped")) for row in rows]
    return bool(flags) and sum(flags) > len(flags) / 2


def _front(xs, ys):
    order = sorted(range(len(xs)), key=lambda i: (xs[i], ys[i]))
    keep, best = [], math.inf
    for index in order:
        if ys[index] < best:
            keep.append(index)
            best = ys[index]
    return keep


def pareto(ax, data):
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


DRAW_MAP = {"pareto": pareto}


if __name__ == "__main__":
    raise SystemExit(plotting.main(__file__, DRAW_MAP))

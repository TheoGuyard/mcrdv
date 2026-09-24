#!/usr/bin/env python3
"""Figures for the sensitivity sweeps: one instance parameter at a time.

Three drawings, read once per swept parameter. `plot.yaml` says which runs a
figure is about (`sweep.axis`) and what its abscissa is (`group_by`), so adding
a parameter to the campaign needs an entry there and no code here.

The question each parameter is asked twice: what does it cost to solve, and
what does the plan that comes back look like.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from _common import export, plotting, tidy  # noqa: E402

# The plan's shape, as the columns holding it and what each is called. All are
# drawn relative to the smallest setting of the swept parameter, which is what
# lets quantities sharing no units share an axis.
STRUCTURE = (
    ("active_chasers", "chasers used"),
    ("load_mean", "targets per chaser"),
    ("makespan", "mission duration"),
    ("wait_share", "share of time waiting"),
    ("raan_spread_mean", "RAAN spread per chaser"),
)


def _infeasible(data):
    """How many of the drawn runs came back infeasible.

    A horizon or a fleet can be cut until no plan satisfies the limits, and the
    planner says so rather than refusing to answer. A figure that did not
    mention it would show a cost curve bending for reasons a reader could not
    see.
    """
    rows = [row for group in data.groups for row in group] or data.rows
    return sum(row.get("feasible") is False for row in rows), len(rows)


def time(ax, data):
    """Where the wall clock goes as the parameter moves.

    Setup and search are drawn apart because they scale differently: setup
    tabulates the per-leg bounds and is quadratic in the number of targets,
    while the search runs against whatever budget it was given. Summing them
    would hide whichever term is actually growing.

    `plot.yaml` reduced `t_setup`; the search time is read off the same groups,
    so both curves come from one selection rather than two.
    """
    export.band(ax, data.x, data.lo, data.hi, name="setup_iqr")
    export.line(
        ax, data.x, data.median, name="setup", label="setup", marker="o"
    )

    solve = [tidy.spread(group, "t_solve") for group in data.groups]
    export.band(
        ax,
        data.x,
        [s[1] for s in solve],
        [s[2] for s in solve],
        name="solve_iqr",
    )
    export.line(
        ax,
        data.x,
        [s[0] for s in solve],
        name="solve",
        label="search",
        marker="s",
    )
    ax.legend(fontsize=7)


def cost(ax, data):
    """Whatever `plot.yaml` reduced, with the spread over seeds.

    Pointed at `cost` it is what the mission is judged on; pointed at
    `cost_per_target` it says whether a bigger instance is harder or merely
    longer -- a total that climbs while the per-target figure stays flat is an
    instance that grew rather than one that got worse. One drawing either way,
    since the only difference is which column was reduced.
    """
    export.band(ax, data.x, data.lo, data.hi, name="value_iqr")
    export.line(ax, data.x, data.median, name="value", marker="o")

    count, total = _infeasible(data)
    if count:
        ax.set_title(f"{count} of {total} runs exceed the limits", fontsize=8)


def throughput(ax, data):
    """Genetic iterations per second of search.

    The search runs against a wall-clock budget, so its *duration* is the
    budget and says nothing; what changes is how much of the search fits
    inside it. This is the figure that shows an instance getting harder rather
    than merely bigger.
    """
    export.band(ax, data.x, data.lo, data.hi, name="throughput_iqr")
    export.line(ax, data.x, data.median, name="throughput", marker="o")


def structure(ax, data):
    """The shape of the plan, relative to the smallest setting drawn.

    Quantities sharing no units -- a count of chasers, a duration, a share of
    time -- put on one axis by reading each against its own value at the left
    edge. What matters is which of them moves and how fast, not where any one
    of them starts.

    A flat line is a plan whose shape the parameter does not touch, which is as
    much of a result as a steep one: it says the planner absorbed the change
    somewhere else.
    """
    if not data.groups:
        return
    for column, label in STRUCTURE:
        values = [tidy.spread(group, column)[0] for group in data.groups]
        base = values[0]
        if not np.isfinite(base) or base == 0.0:
            continue
        export.line(
            ax,
            data.x,
            [value / base for value in values],
            name=column,
            label=label,
            marker="o",
            markersize=3,
        )
    # The line every curve is read against.
    ax.axhline(1.0, color="0.6", linewidth=0.8, zorder=1)
    export.series(
        ax,
        "unity",
        [data.x[0], data.x[-1]],
        kind="context",
        unity=[1.0, 1.0],
    )
    ax.legend(fontsize=7, ncol=2)


DRAW_MAP = {
    "time": time,
    "cost": cost,
    "throughput": throughput,
    "structure": structure,
}


if __name__ == "__main__":
    raise SystemExit(plotting.main(__file__, DRAW_MAP))

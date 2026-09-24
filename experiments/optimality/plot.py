#!/usr/bin/env python3
"""Figures for the optimality experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from _common import export, plotting, tidy  # noqa: E402


def _planners(rows):
    """Planner names in a stable order: the search by depth, then the rest."""
    names = sorted({r["planner"] for r in rows if r.get("planner")})
    return sorted(names, key=lambda n: (not n.startswith("hgs"), n))


def _sizes(rows):
    return sorted({int(r["num_targets"]) for r in rows if "num_targets" in r})


def _column(name):
    """A planner name as a pgfplots column: letters, digits, underscores."""
    return name.replace("-", "_")


def gap(ax, data):
    """Mean gap to the optimum against the instance size, per planner.

    The mean rather than the median: the search sits at the optimum on most
    runs, so its median is zero almost everywhere and hides exactly the misses
    this figure is about. How often a planner is optimal is the next figure.
    The axis is linear up to 1% and logarithmic above, so a search a fraction of
    a percent off and a construction far off both stay readable.
    """
    sizes = _sizes(data.rows)
    for name in _planners(data.rows):
        means = [
            tidy.column(
                [
                    r
                    for r in data.rows
                    if r.get("planner") == name and r["num_targets"] == n
                ],
                "gap",
            )
            for n in sizes
        ]
        export.line(
            ax,
            sizes,
            [m.mean() if m.size else np.nan for m in means],
            name=_column(name),
            label=name,
            marker="o",
        )
    ax.set_yscale("symlog", linthresh=1.0)
    # A gap is never negative, and when every run is optimal the automatic
    # limits would zoom into rounding noise around zero.
    ax.set_ylim(0.0, max(1.0, ax.get_ylim()[1]))
    ax.set_xticks(sizes)


def optimal(ax, data):
    """Share of runs that return the optimum, per planner and size.

    A run that returns an infeasible plan counts as a miss: it is not the
    optimum, whatever its cost.
    """
    sizes = _sizes(data.rows)
    for name in _planners(data.rows):
        share = []
        for n in sizes:
            members = [
                r
                for r in data.rows
                if r.get("planner") == name and r["num_targets"] == n
            ]
            share.append(
                100.0 * np.mean([bool(r.get("optimal")) for r in members])
                if members
                else np.nan
            )
        export.line(
            ax, sizes, share, name=_column(name), label=name, marker="o"
        )
    ax.set_ylim(-5.0, 105.0)
    ax.set_xticks(sizes)


DRAW_MAP = {"gap": gap, "optimal": optimal}


if __name__ == "__main__":
    raise SystemExit(plotting.main(__file__, DRAW_MAP))

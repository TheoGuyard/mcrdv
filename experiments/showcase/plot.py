#!/usr/bin/env python3
"""Figures describing one produced mission plan."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from mcrdv.orbit import DAY  # noqa: E402

from common import export, plotting, style  # noqa: E402


def _best(data):
    """The cheapest record of the selection: the plan the figures describe.

    A single solve of a stochastic search is an anecdote, so the campaign runs
    several seeds; the figures show the best of them, and the others stay in the
    results for the choice to be visible.
    """
    usable = [r for r in data.records if r.result.get("chasers")]
    if not usable:
        return None
    return min(usable, key=lambda r: r.result["cost"])


def _active(record):
    """The chasers that actually collect something, heaviest first.

    Idle chasers are dropped rather than drawn as zeros: the planner is free to
    leave them on the ground, and a row of empty bars says nothing about how the
    work that *was* assigned got distributed.
    """
    rows = [c for c in record.result["chasers"] if c["load"] > 0]
    return sorted(rows, key=lambda c: -c["load"])


def load(ax, data):
    """How the targets are shared out."""
    record = _best(data)
    if record is None:
        return
    active = _active(record)
    export.grouped_bars(
        ax,
        [str(c["chaser"]) for c in active],
        {"targets": [c["load"] for c in active]},
    )
    ax.set_title(f"{len(active)} chasers used")


def cost_time(ax, data):
    """Each chaser's cost against how long it flies.

    A chaser that is expensive because it flies for a year is a different
    problem from one that is expensive in a month, and the scatter separates
    them where a cost ranking alone would not.
    """
    record = _best(data)
    if record is None:
        return
    active = _active(record)
    export.points(
        ax,
        [c["time"] / 86_400.0 for c in active],
        [c["cost"] for c in active],
        name="chasers",
    )


def spread(ax, data):
    """Angular spread of each chaser's targets against what it cost.

    This is the geometric claim the whole approach rests on: a chaser assigned
    targets in nearby orbital planes should be cheap, because a plane change is
    what a transfer mostly pays for. If the relationship is not visible here,
    the clustering is not doing what it is supposed to.
    """
    record = _best(data)
    if record is None:
        return
    active = _active(record)
    export.points(
        ax,
        [np.degrees(c["raan_spread"]) for c in active],
        [c["cost"] / max(c["load"], 1) for c in active],
        name="chasers",
    )


def assignment(ax, data):
    """Which chaser collects which target, laid out in RAAN against time.

    The faint lines are the targets themselves: each drifts in right ascension
    under J2, and where two of them run close together they are cheap to visit
    in succession. The markers are where a chaser actually arrives, one colour
    per chaser, joined in visiting order.

    This is the picture the whole approach rests on. A chaser's markers should
    track a narrow band of the plot rather than jumping across it, because a
    plane change is what a transfer mostly pays for -- if the colours are
    scattered all over, the clustering is not doing its job.

    The angle is drawn **unwrapped**, so the axis runs past 360 and below zero
    rather than folding back. Wrapping to a single turn is what an angle usually
    wants, but here it cuts every track that crosses the seam into pieces at
    opposite edges of the plot, which is precisely the reading this figure
    exists to support. The cost is that two targets either side of zero look far
    apart when they are not; over one mission the drift is a few turns at most,
    so that trade is worth making.
    """
    best = _best(data)
    if best is None or "raan" not in best.result:
        return

    tracks = best.result["raan"]
    r0 = np.asarray(tracks["r0"], dtype=float)
    dr = np.asarray(tracks["dr"], dtype=float)
    horizon = float(tracks["horizon"])

    # The targets' drift, as context. Not exported: 100+ curves would swamp the
    # table, and each one is exactly two numbers (r0, dr) if it is ever wanted.
    when = np.linspace(0.0, horizon, 400)
    days = when / DAY
    for state in range(1, r0.size):
        ax.plot(
            days,
            np.degrees(r0[state] + dr[state] * when),
            color="0.7",
            alpha=0.25,
            linewidth=0.5,
            zorder=1,
        )

    # Where each chaser arrives. Drawn per chaser so the colours separate;
    # recorded once, as the tidy table a scatter wants.
    #
    # Not the shared palette here: it holds six colours, one of them grey, and a
    # swarm is fifteen chasers or more. Cycling six would repeat them and paint
    # one chaser the same grey as the background it has to stand out from, so
    # the colours are sampled across a map wide enough to keep them apart.
    visits = best.result.get("visits", [])
    colours = style.distinct(len(visits))

    every_time, every_raan, every_chaser = [], [], []
    for order, visit in enumerate(visits):
        at = np.asarray(visit["time"], dtype=float) / DAY
        angle = np.degrees(np.asarray(visit["raan"], dtype=float))
        colour = colours[order]
        ax.plot(at, angle, color=colour, alpha=0.65, linewidth=0.9, zorder=2)
        ax.scatter(
            at,
            angle,
            color=colour,
            s=14,
            zorder=3,
            linewidths=0.0,
            label=f"chaser {visit['chaser']}",
        )
        every_time.extend(at.tolist())
        every_raan.extend(angle.tolist())
        every_chaser.extend([visit["chaser"]] * at.size)

    export.series(
        ax, "visits", every_time, raan=every_raan, chaser=every_chaser
    )
    # Ticks on multiples of a quarter turn, wherever the unwrapped range lands.
    ax.yaxis.set_major_locator(MultipleLocator(90.0))


FIGURES = {
    "load": load,
    "cost_time": cost_time,
    "spread": spread,
    "assignment": assignment,
}


if __name__ == "__main__":
    raise SystemExit(plotting.main(__file__, FIGURES))

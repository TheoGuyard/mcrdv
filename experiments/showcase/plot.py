#!/usr/bin/env python3
"""Figures describing one produced mission plan.

What the planner decided, from four angles: which targets each chaser collects
and where they are when it does (`raan`), when it flies and when it loiters
(`timeline`), how the work and its cost are shared out (`load`, `cost_time`,
`spread`), and how the cost is spread over the legs (`legs`).

The assignment drawing is written once and registered for every Keplerian
element, so `plot.yaml` can ask for the same picture in any of them; right
ascension is the one that drives a transfer's cost, and the one drawn by
default.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from mcrdv.orbit import DAY  # noqa: E402
from mcrdv.problem import DEPOT  # noqa: E402

from _common import analysis, export, plotting, style  # noqa: E402

_DEGREES = 180.0 / math.pi

# How each element reads on an axis: what to multiply the stored value by, and
# whether it is an angle. Angles are drawn unwrapped and, where they wander far
# enough to need it, get their ticks on whole fractions of a turn.
AXIS = {
    "a": (1e-3, False),  # metres -> kilometres
    "e": (1.0, False),
    "i": (_DEGREES, True),
    "raan": (_DEGREES, True),
    "argp": (_DEGREES, True),
    "anomaly": (_DEGREES, True),
}


def _best(data):
    """The cheapest record of the selection: the plan the figures describe.

    A single solve of a stochastic search is an anecdote, so the campaign runs
    several seeds; the figures show the best of them, and the others stay in the
    results for the choice to be visible.
    """
    usable = [r for r in data.records if r.result.get("visits")]
    if not usable:
        return None
    return min(usable, key=lambda r: r.result["cost"])


def _active(record):
    """The chasers that actually collect something, heaviest first.

    Idle chasers are dropped rather than drawn as zeros: the planner is free to
    leave them on the ground, and a row of empty bars says nothing about how the
    work that *was* assigned got distributed.
    """
    rows = [c for c in record.result.get("chasers", []) if c["load"] > 0]
    return sorted(rows, key=lambda c: -c["load"])


def _ticks(ax):
    """Ticks on whole fractions of a turn, coarsened to the range drawn.

    For an angle that drifts several turns over a mission, ticks every quarter
    turn would be twenty unreadable labels; for one that barely moves, they
    would be a single label. So this applies only above a quarter turn of
    range, and leaves matplotlib to it below.
    """
    low, high = ax.get_ylim()
    span = high - low
    if span < 90.0:
        return
    step = next(
        (t for t in (90.0, 180.0, 360.0, 720.0) if span / t <= 10.0), 1440.0
    )
    ax.yaxis.set_major_locator(MultipleLocator(step))


def _drift(ax, data, element):
    """Which chaser collects which target, laid out in one element against time.

    The faint lines are the objects themselves, each carried through the
    mission by the secular J2 model; where two of them run close together they
    are cheap to visit in succession. The dashed line is the swarm's own orbit:
    every chaser starts on it and drifts with it until it departs, so every
    track begins there. The markers are where a chaser arrives at a target, one
    colour per chaser, joined in visiting order.

    An angle is drawn **unwrapped**, so the axis runs past 360 and below zero
    rather than folding back. Wrapping to a single turn is what an angle usually
    wants, but here it cuts every track that crosses the seam into pieces at
    opposite edges of the plot, which is precisely the reading these figures
    exist to support. The cost is that two objects either side of zero look far
    apart when they are not; over one mission the drift is a few turns at most,
    so that trade is worth making.

    A record written before this element was kept has no track for it, and
    draws nothing rather than failing: re-run the experiment to fill it.
    """
    best = _best(data)
    if best is None:
        return
    tracks = (best.result.get("tracks") or {}).get(element)
    if not tracks:
        return
    scale, is_angle = AXIS.get(element, (1.0, False))

    start = scale * np.asarray(tracks["start"], dtype=float)
    rate = scale * np.asarray(tracks["rate"], dtype=float)
    horizon = float(tracks["horizon"])

    # The objects' drift, as context.
    when = np.linspace(0.0, horizon, 400)
    days = when / DAY
    for state in range(1, start.size):
        ax.plot(
            days,
            start[state] + rate[state] * when,
            color="0.7",
            alpha=0.25,
            linewidth=0.5,
            zorder=1,
        )

    # Exported as two points per object rather than the four hundred drawn: a
    # drift is a straight line, and its ends are all a straight line needs. A
    # gap between them lets one `\addplot` carry every object without joining
    # the end of one to the start of the next, and it is written first so that
    # it lands behind the tracks that the figure is actually about.
    edges: list = []
    angles: list = []
    for state in range(1, start.size):
        edges += [0.0, horizon / DAY, np.nan]
        angles += [start[state], start[state] + rate[state] * horizon, np.nan]
    export.series(ax, "debris", edges, kind="context", **{element: angles})

    swarm = start[DEPOT] + rate[DEPOT] * when
    ax.plot(
        days,
        swarm,
        color="0.35",
        linestyle="--",
        linewidth=1.0,
        zorder=2,
    )
    # Exported, unlike the targets: it is one line, and it is the line every
    # track starts on.
    export.series(ax, "swarm", days, **{element: swarm})

    # Where each chaser goes. Drawn per chaser so the colours separate;
    # recorded once, as the tidy table a scatter wants.
    #
    # Not the shared palette here: it holds six colours, one of them grey, and a
    # swarm is fifteen chasers or more. Cycling six would repeat them and paint
    # one chaser the same grey as the background it has to stand out from, so
    # the colours are sampled across a map wide enough to keep them apart.
    visits = [v for v in best.result.get("visits", []) if element in v]
    colours = style.distinct(len(visits))

    for order, visit in enumerate(visits):
        at = np.asarray(visit["time"], dtype=float) / DAY
        values = scale * np.asarray(visit[element], dtype=float)
        # The track runs from the swarm state; the markers are the arrivals at
        # targets, so the shared start does not read as fifteen collections of
        # the same thing.
        states = np.asarray(visit.get("states", []), dtype=np.int64)
        if states.size != at.size:
            states = np.full(at.size, -1, dtype=np.int64)
        marks = states != DEPOT
        colour = colours[order]
        ax.plot(at, values, color=colour, alpha=0.65, linewidth=0.9, zorder=3)
        ax.scatter(
            at[marks],
            values[marks],
            color=colour,
            s=14,
            zorder=4,
            linewidths=0.0,
            label=f"chaser {visit['chaser']}",
        )
        # One table per chaser rather than one tidy table for all of them:
        # each is a track of its own, and a single table would have pgfplots
        # join the end of one chaser to the start of the next. `state` rides
        # along unplotted, 0 marking the swarm state each track starts on.
        export.series(
            ax,
            f"chaser{visit['chaser']}",
            at,
            aux=("state",),
            **{element: values, "state": states},
        )
    if is_angle:
        _ticks(ax)


def _drawing(element):
    """The drawing for one element, as `plot.yaml` asks for it by name."""

    def draw(ax, data):
        _drift(ax, data, element)

    draw.__name__ = element
    draw.__doc__ = f"The assignment read in {element}."
    return draw


def timeline(ax, data):
    """When each chaser flies and when it loiters, one row per chaser.

    The pale stretches are waiting on the orbit last reached, the coloured ones
    are transfers. This is the schedule layer's decision made visible: waiting
    is free and a plane drifts into alignment on its own, so a plan that is
    mostly pale is buying its delta-v with patience. Rows are ordered by the
    moment each chaser finishes, so the staircase shows how the swarm's work
    tails off -- and a row still flying at the horizon is one the mission
    duration limit is binding on.
    """
    record = _best(data)
    if record is None or not record.result.get("timeline"):
        return
    rows = sorted(record.result["timeline"], key=lambda r: -r["end"])
    colours = style.distinct(len(rows))

    # Each interval is exported as the segment it is drawn as: its two ends at
    # the row's height, then a gap. A bar of a given width is a length on the
    # time axis, which a table of widths alone could not place.
    spans = {"wait": [[], []], "transfer": [[], []]}
    for position, (row, colour) in enumerate(zip(rows, colours)):
        for kind, face in (("wait", "0.85"), ("transfer", colour)):
            bars = [
                (start / DAY, (end - start) / DAY)
                for start, end in row[kind]
                if end > start
            ]
            if bars:
                ax.broken_barh(
                    bars,
                    (position - 0.35, 0.7),
                    facecolors=face,
                    zorder=2 if kind == "wait" else 3,
                )
            for start, width in bars:
                spans[kind][0] += [start, start + width, np.nan]
                spans[kind][1] += [position, position, np.nan]

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([str(row["chaser"]) for row in rows])
    ax.invert_yaxis()
    flying = sum(sum(b - a for a, b in row["transfer"]) for row in rows)
    idle = sum(sum(b - a for a, b in row["wait"]) for row in rows)
    if flying + idle > 0:
        ax.set_title(
            f"{len(rows)} chasers, "
            f"{100.0 * idle / (flying + idle):.0f}% of their time waiting"
        )
    for kind, (start, height) in spans.items():
        export.series(ax, kind, start, kind="segments", **{kind: height})


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
        [c["time"] / DAY for c in active],
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


def legs(ax, data):
    """What the individual transfers cost, as a distribution.

    Every leg of every chaser, cheapest first, against the share of legs at or
    below it. A curve that climbs steeply and then flattens far to the right is
    a plan whose cost sits in a handful of expensive transfers -- worth knowing,
    because those are the legs a better sequence would have to remove.
    """
    record = _best(data)
    if record is None:
        return
    values = np.asarray(record.result.get("legs", ()), dtype=float)
    values = np.sort(values[np.isfinite(values)])
    if not values.size:
        return
    share = 100.0 * np.arange(1, values.size + 1) / values.size
    export.line(ax, values, share, name="legs")
    # How top-heavy the plan is, in one number: a tenth of the legs carrying
    # most of the cost is the shape worth noticing.
    dearest = values[-max(1, round(0.1 * values.size)) :]
    ax.set_title(
        f"{values.size} legs, median {np.median(values):.0f}, "
        f"dearest tenth carries {100.0 * dearest.sum() / values.sum():.0f}%"
    )


# The assignment drawing for every element the analysis knows about, and the
# figures about the plan's structure. `plot.yaml` decides which of these a run
# draws, and what their axes say.
DRAW_MAP = {element: _drawing(element) for element in analysis.ELEMENTS}
DRAW_MAP.update(
    {
        "timeline": timeline,
        "load": load,
        "cost_time": cost_time,
        "spread": spread,
        "legs": legs,
    }
)


if __name__ == "__main__":
    raise SystemExit(plotting.main(__file__, DRAW_MAP))

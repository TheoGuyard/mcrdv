#!/usr/bin/env python3
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
    usable = [r for r in data.records if r.result.get("visits")]
    if not usable:
        return None
    return min(usable, key=lambda r: r.result["cost"])


def _active(record):
    rows = [c for c in record.result.get("chasers", []) if c["load"] > 0]
    return sorted(rows, key=lambda c: -c["load"])


def _ticks(ax):
    low, high = ax.get_ylim()
    span = high - low
    if span < 90.0:
        return
    step = next(
        (t for t in (90.0, 180.0, 360.0, 720.0) if span / t <= 10.0), 1440.0
    )
    ax.yaxis.set_major_locator(MultipleLocator(step))


def _drift(ax, data, element):
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
    def draw(ax, data):
        _drift(ax, data, element)

    draw.__name__ = element
    return draw


def timeline(ax, data):
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

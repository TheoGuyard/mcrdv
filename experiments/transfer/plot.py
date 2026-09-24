#!/usr/bin/env python3
"""Figures for the transfer experiment.

One solver, one instance, one set of seeds -- and four different answers to the
question of what a transfer costs. Every figure here reads the *structure* of
the plans that come back rather than their score: when the chasers fly and when
they sit still, how long a single pause lasts, what a leg costs, how wide a
chaser's share of the sky is, how many chasers are used at all, and where in a
tour a given target ends up.

None of that was asked for. It is what the search does when the price of a
transfer changes underneath it, which is the claim these figures exist to
support: the transfer layer is not a parameter of the mission, it is the
definition of one.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from mcrdv.orbit import DAY  # noqa: E402

from _common import config as configuration  # noqa: E402
from _common import export, plotting, tidy  # noqa: E402

# The oracles, in the order they are drawn, named for what they price rather
# than for what they achieve: these figures are about the definition, not the
# score. An oracle missing from the records is simply skipped.
ORACLES = {
    "fuel": "delta-v",
    "arrival": "delta-v + arrival time",
    "urgency": "delta-v + urgency",
    "blackout": "delta-v, windows barred",
    "embargo": "delta-v, late start",
}

# The structural quantities followed as the weight on time is raised, each as
# the column holding it and what it is called on a figure. All are drawn as a
# ratio to the delta-v-only plan, which is what lets them share an axis.
STRUCTURE = (
    ("delta_v", "delta-v"),
    ("wait_time", "time spent waiting"),
    ("makespan", "mission duration"),
    ("active_chasers", "chasers used"),
)


def _oracles(data):
    """The records of each oracle present, in the order they are drawn."""
    grouped: dict = {}
    for record, row in zip(data.records, data.rows):
        grouped.setdefault(row.get("oracle"), []).append(record)
    return [(name, grouped[name]) for name in ORACLES if name in grouped]


def _legs(records):
    """Every leg of every run, as the rows `run.py` wrote them."""
    return [leg for record in records for leg in record.result.get("legs", ())]


def _distribution(ax, values, name, scale=1.0):
    """One sample, drawn as the share of it at or below each value.

    A distribution rather than a mean, because that is where the differences
    between these plans live: two oracles can spend the same total time waiting
    and reach it from pauses of entirely different lengths, and an average
    hides exactly the structure this experiment is about.
    """
    values = np.asarray(values, dtype=float) * scale
    values = np.sort(values[np.isfinite(values)])
    if not values.size:
        return
    share = 100.0 * np.arange(1, values.size + 1) / values.size
    export.line(
        ax,
        values,
        share,
        name=name,
        label=ORACLES[name],
        drawstyle="steps-post",
    )


def _embargo(data):
    """When the embargoed runs are allowed to start, in days, if any are here.

    Read from the configuration rather than from the plans: it is what the
    layer was told, and a figure that drew it from the first departure it
    found would be drawing the plan's answer as though it were the question.
    """
    for row in data.rows:
        if row.get("oracle") != "embargo":
            continue
        start = row.get("solver.args.transfer.args.start")
        if start is not None:
            return configuration.duration(start) / DAY
    return None


def progress(ax, data):
    """How the collections fill the horizon.

    Targets collected against time, per run. The shape of the curve is the
    shape of the mission: a plan that prices delta-v alone spreads its
    collections over the whole horizon, because waiting for a plane to drift
    into place costs nothing and the horizon is the only thing that stops it.
    Pricing the arrival time bends the same work into the first weeks and
    leaves the rest of the year empty. Nothing about the instance changed --
    only what a transfer was said to cost.

    The shaded stretch, where an embargoed oracle is among those drawn, is the
    span it may not depart in: its curve leaves the floor only once the date
    passes, and the mission it then flies is the same mission, late.
    """
    start = _embargo(data)
    if start:
        export.span(ax, 0.0, start)
    for name, records in _oracles(data):
        times = np.sort(
            np.array([leg["arrive"] for leg in _legs(records)]) / DAY
        )
        if not times.size:
            continue
        # Divided by the runs pooled, so an oracle drawn from more seeds than
        # another is not drawn taller than it, and the height stays a number of
        # targets rather than a count of whatever is in the folder.
        collected = np.arange(1, times.size + 1) / len(records)
        export.line(
            ax,
            times,
            collected,
            name=name,
            label=ORACLES[name],
            drawstyle="steps-post",
        )
    ax.legend(fontsize=7)


def rhythm(ax, data):
    """What a chaser's mission is made of: transfers, and waiting to start one.

    Per active chaser, averaged over the runs. The delta-v-only plan is mostly
    waiting, and that is how it is cheap: a chaser that loiters lets the J2
    drift perform the plane change it would otherwise have to buy. Pricing time
    takes that back -- the pauses shrink, and the chaser is done in a fraction
    of the mission the other plans take.
    """
    oracles = _oracles(data)
    if not oracles:
        return

    flying, idle = [], []
    for _, records in oracles:
        legs = _legs(records)
        chasers = {
            (index, leg["chaser"])
            for index, record in enumerate(records)
            for leg in record.result.get("legs", ())
        }
        count = max(len(chasers), 1)
        flying.append(sum(leg["flight"] for leg in legs) / count / DAY)
        idle.append(sum(leg["wait"] for leg in legs) / count / DAY)

    export.grouped_bars(
        ax,
        [ORACLES[name] for name, _ in oracles],
        {"in transfer": flying, "waiting": idle},
    )
    ax.tick_params(axis="x", labelrotation=15)
    ax.legend(fontsize=7)


def waits(ax, data):
    """How long a single pause lasts.

    Every wait between arriving somewhere and leaving it again, as a
    distribution. The curves are three different mission rhythms: pauses of
    weeks where waiting is free, almost none where the arrival time is priced,
    and -- under the window constraint -- pauses that end where a window
    closes, since a chaser held back leaves at the instant it may.
    """
    for name, records in _oracles(data):
        _distribution(
            ax, [leg["wait"] for leg in _legs(records)], name, scale=1.0 / DAY
        )
    ax.legend(fontsize=7, loc="lower right")


def legs(ax, data):
    """What a single transfer costs, on the scale every plan shares.

    Each leg re-priced in delta-v at the departure the plan actually chose, as
    a distribution. This is where the bill for hurrying appears: the curves
    part at the expensive end, where a plan that would not wait for the
    geometry has to buy the plane change instead of drifting into it.
    """
    for name, records in _oracles(data):
        _distribution(ax, [leg["delta_v"] for leg in _legs(records)], name)
    ax.legend(fontsize=7, loc="lower right")


def spread(ax, data):
    """How wide a slice of the sky one chaser is given.

    The angular spread of a chaser's targets in right ascension, over the
    active chasers. This is the assignment rather than the schedule: whether
    the oracle changed *which* targets are grouped together, not merely when
    they are collected. A plan under time pressure cannot wait for a distant
    plane to come round, so it either takes a wider slice and pays for it, or
    splits the work over more chasers -- which is what `load` counts.
    """
    for name, records in _oracles(data):
        _distribution(
            ax,
            [
                chaser["raan_spread"]
                for record in records
                for chaser in record.result.get("chasers", ())
                if chaser["load"] > 0
            ],
            name,
            scale=180.0 / np.pi,
        )
    ax.legend(fontsize=7, loc="lower right")


def load(ax, data):
    """How the work is shared out, chaser by chaser.

    Chasers ranked by how much they carry, averaged over the runs. A flat
    profile that stops abruptly is a plan that filled as few chasers as it
    could; one that tails off slowly has spread the same targets over more of
    the swarm. Which of the two comes back is a structural decision nobody
    asked the planner to make -- the fleet size is the same in every run here.
    """
    for name, records in _oracles(data):
        profiles = [
            sorted(
                (c["load"] for c in record.result.get("chasers", ())),
                reverse=True,
            )
            for record in records
        ]
        width = max((len(p) for p in profiles), default=0)
        if not width:
            continue
        padded = np.array([p + [0] * (width - len(p)) for p in profiles])
        export.line(
            ax,
            np.arange(1, width + 1),
            padded.mean(axis=0),
            name=name,
            label=ORACLES[name],
            marker="o",
            markersize=3,
        )
    ax.legend(fontsize=7)


def priority(ax, data):
    """Where in a tour the hazardous objects end up.

    The mean visiting position of a hazardous target beside that of every other
    one, where position 1 is the first target a chaser collects. An oracle that
    weighs the hazardous objects alone should pull them to the front of the
    tours and leave the rest in whatever order is cheapest -- a reordering of
    the sequences rather than a rescheduling of them, which is the search
    rearranging itself around a price only it can see.
    """
    oracles = _oracles(data)
    if not oracles:
        return

    def mean(legs, flag):
        chosen = [
            leg["position"] for leg in legs if leg.get("hazardous") is flag
        ]
        return float(np.mean(chosen)) if chosen else 0.0

    export.grouped_bars(
        ax,
        [ORACLES[name] for name, _ in oracles],
        {
            "hazardous": [
                mean(_legs(records), True) for _, records in oracles
            ],
            "the rest": [
                mean(_legs(records), False) for _, records in oracles
            ],
        },
    )
    ax.tick_params(axis="x", labelrotation=15)
    ax.legend(fontsize=7)


def blackout(ax, data):
    """Where in the cycle the chasers depart, against the window barred to them.

    Departures are folded onto one period rather than drawn along the mission:
    a window three days in ten is invisible on a year-long axis, and what
    matters is not when a chaser leaves but where in the cycle it leaves. The
    shaded stretch is the window. The delta-v-only oracle knows nothing of it
    and departs inside as often as anywhere else; the oracle that prices those
    departures at infinity leaves the stretch empty and piles its departures
    against the trailing edge -- a mission visibly built around a constraint
    that exists nowhere but in the transfer layer.

    The window is two numbers in the record -- the period and the duration --
    rather than a column, so it is drawn from those and carried into the tikz
    stub as the shaded stretch it is.
    """
    settings = next(
        (
            record.config.get("metrics")
            for record in data.records
            if record.config.get("metrics")
        ),
        None,
    )
    if not settings:
        return
    period = configuration.duration(settings.get("blackout_period", 0)) / DAY
    duration = (
        configuration.duration(settings.get("blackout_duration", 0)) / DAY
    )
    if period <= 0:
        return

    bins = np.linspace(0.0, period, 21)
    centres = 0.5 * (bins[:-1] + bins[1:])
    inside = {}
    for name, records in _oracles(data):
        phases = np.array(
            [(leg["depart"] / DAY) % period for leg in _legs(records)]
        )
        if not phases.size:
            continue
        inside[name] = int((phases < duration).sum()), phases.size
        counts, _ = np.histogram(phases, bins=bins)
        export.line(
            ax,
            centres,
            counts,
            name=name,
            label=ORACLES[name],
            drawstyle="steps-mid",
        )

    export.span(ax, 0.0, duration)
    ax.set_title(
        ", ".join(
            f"{ORACLES[name]}: {count} of {total} inside"
            for name, (count, total) in inside.items()
        ),
        fontsize=8,
    )
    ax.legend(fontsize=7)


def weighting(ax, data):
    """The same oracle restructuring the plan as its weight on time rises.

    Everything as a ratio to the plan built from delta-v alone, so quantities
    sharing no units can share an axis. None of this is a setting the planner
    was given: the weight is a price, and a shorter mission, fewer idle weeks
    and more of the swarm in use are what the search found worth buying at that
    price. The plan moves continuously between two structures, which is the
    point -- the oracle is a dial, not a switch.
    """
    rows = [
        row for row in data.rows if row.get("oracle") in ("fuel", "arrival")
    ]
    reference = [row for row in rows if row.get("oracle") == "fuel"]
    weights = sorted(
        {row["weight"] for row in rows if row.get("weight", 0.0) > 0.0}
    )
    if not reference or not weights:
        return

    for column, label in STRUCTURE:
        base = tidy.spread(reference, column)[0]
        if not base:
            continue
        ratios = [
            tidy.spread([r for r in rows if r.get("weight") == w], column)[0]
            / base
            for w in weights
        ]
        export.line(
            ax,
            weights,
            ratios,
            name=column,
            label=label,
            marker="o",
            markersize=3,
        )
    # The plan every ratio is read against, drawn as a line to read them
    # against and exported as the context it is.
    ax.axhline(1.0, color="0.6", linewidth=0.8, zorder=1)
    export.series(
        ax,
        "unity",
        [weights[0], weights[-1]],
        kind="context",
        unity=[1.0, 1.0],
    )
    ax.legend(fontsize=7)


DRAW_MAP = {
    "progress": progress,
    "rhythm": rhythm,
    "waits": waits,
    "legs": legs,
    "spread": spread,
    "load": load,
    "priority": priority,
    "blackout": blackout,
    "weighting": weighting,
}


if __name__ == "__main__":
    raise SystemExit(plotting.main(__file__, DRAW_MAP))

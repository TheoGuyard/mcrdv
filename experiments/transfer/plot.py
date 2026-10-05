#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from mcrdv.orbit import DAY  # noqa: E402

from _common import config as configuration  # noqa: E402
from _common import export, plotting, tidy  # noqa: E402

ORACLES = {
    "fuel": "delta-v",
    "arrival": "delta-v + arrival time",
    "urgency": "delta-v + urgency",
    "blackout": "delta-v, windows barred",
    "embargo": "delta-v, late start",
}

STRUCTURE = (
    ("delta_v", "delta-v"),
    ("wait_time", "time spent waiting"),
    ("makespan", "mission duration"),
    ("active_chasers", "chasers used"),
)


def _oracles(data):
    grouped: dict = {}
    for record, row in zip(data.records, data.rows):
        grouped.setdefault(row.get("oracle"), []).append(record)
    return [(name, grouped[name]) for name in ORACLES if name in grouped]


def _legs(records):
    return [leg for record in records for leg in record.result.get("legs", ())]


def _distribution(ax, values, name, scale=1.0):
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
    for row in data.rows:
        if row.get("oracle") != "embargo":
            continue
        start = row.get("solver.args.transfer.args.start")
        if start is not None:
            return configuration.duration(start) / DAY
    return None


def progress(ax, data):
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
    for name, records in _oracles(data):
        _distribution(
            ax, [leg["wait"] for leg in _legs(records)], name, scale=1.0 / DAY
        )
    ax.legend(fontsize=7, loc="lower right")


def legs(ax, data):
    for name, records in _oracles(data):
        _distribution(ax, [leg["delta_v"] for leg in _legs(records)], name)
    ax.legend(fontsize=7, loc="lower right")


def spread(ax, data):
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

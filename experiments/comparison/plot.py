#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from mcrdv.orbit import DAY  # noqa: E402

from _common import runner, plotter  # noqa: E402

# López Rivera & Naeije (2025), Neural combinatorial optimization for
# multi-rendezvous mission design, Adv. Space Res. 75(10), Table 16: over 1000
# of their missions per size, (mean, std) of the delta-v [km/s] of their
# attention model (NCO, beam search) and of the dynamic RAAN walk (DRW), and of
# their gap [%] to their genetic algorithm. Their missions are not released:
# ours are drawn the same way, and averages are compared.
LOPEZ_RIVERA = {
    "targets": (10, 30, 50),
    "nco_dv": ((106.4, 16.2), (573.5, 83.9), (1041.8, 68.2)),
    "drw_dv": ((122.1, 31.4), (208.1, 46.2), (261.2, 52.6)),
    "nco_gap": ((32.6, 25.5), (616.0, 133.0), (555.4, 97.4)),
    "drw_gap": ((50.5, 36.8), (50.4, 37.2), (63.9, 38.3)),
}

# Zona, Zavoli, Federici & Avanzini (2023), Evolutionary optimization for
# active debris removal mission planning, IEEE Access 11.
#
# Best total delta-v [m/s] by horizon [days], 20-day grid, keyed by
# (chasers, targets): bars of Figs. 6a, 6b and 9a, read to about 10 m/s;
# 3@10 at 720 days is exact (Table 7), 2@10 at 300 days read off Fig. 5a.
ZONA_BEST = {
    (2, 10): {
        300: 4330.0,
        360: 3217.0,
        420: 2645.0,
        480: 2353.0,
        540: 2020.0,
        600: 1716.0,
        660: 1640.0,
        720: 1565.0,
    },
    (3, 10): {
        300: 2493.0,
        360: 1827.0,
        420: 1459.0,
        480: 1325.0,
        540: 1208.0,
        600: 1173.0,
        660: 1121.0,
        720: 1083.56,
    },
    (4, 21): {
        360: 4896.0,
        420: 4184.0,
        480: 3886.0,
        540: 3167.0,
        600: 2817.0,
        660: 2811.0,
        720: 2688.0,
    },
}

# Their search at 720 days [m/s]: mean over 100 runs of their genetic
# algorithm (Tables 2, 4) and island model (Tables 9, 10), and the best known
# solution (Fig. 10 for 3@15, Fig. 9a for 4@21).
ZONA_SEARCH = {
    (3, 15): {"step": 30, "ga": 2104.6, "island": 1948.1, "best": 1916.0},
    (4, 21): {"step": 20, "ga": 3339.7, "island": 2987.8, "best": 2688.0},
}

# The tours of Narayanaswamy et al. (2023), one per method of their Table 2
# (2-opt and 3-opt return the Christofides tour, simulated annealing the
# Lin-Kernighan one), the exact one last, then the search.
TOURS = (
    "tsp-nn",
    "tsp-christofides",
    "tsp-2opt",
    "tsp-3opt",
    "tsp-lk",
    "tsp-sa",
    "tsp-exact",
    "mcrdv",
)


def _rows(data, **match):
    return [
        row
        for row in data.rows
        if all(row.get(key) == value for key, value in match.items())
    ]


def _costs(rows, scale=1e-3):
    values = np.array([row["cost"] for row in rows], dtype=float) * scale
    return values[np.isfinite(values)]


def _days(row):
    return runner.duration(row["instance.max_time"]) / DAY


def _rows_waiting(rows, waiting):
    return [row for row in rows if bool(row.get("waiting")) is waiting]


def _feasible(rows):
    """The runs whose plan respects every limit: an infeasible plan (one that
    overruns the horizon, say) is no mission, whatever its cost."""
    return [row for row in rows if row.get("feasible")]


# ============================================================================ #
# Narayanaswamy et al. (2023): the ODRC, one chaser and then several
# ============================================================================ #


def tours(ax, data):
    """Their tours and the search's, flown without and with waiting. A tour
    with no feasible run is left without a bar and starred."""
    labels, flown, waited = [], [], []
    starred = False
    for method in TOURS:
        rows = _rows(data, method=method, **{"instance.num_chasers": 1})
        if not rows:
            continue
        dropped = False
        for waiting, out in ((False, flown), (True, waited)):
            runs = _rows_waiting(rows, waiting)
            values = _costs(_feasible(runs))
            dropped |= bool(runs) and not values.size
            out.append(float(np.median(values)) if values.size else np.nan)
        labels.append(f"{method}*" if dropped else method)
        starred |= dropped
    plotter.grouped_bars(ax, labels, {"no wait": flown, "waiting": waited})
    if starred:
        ax.set_title("* exceeds the mission horizon", fontsize=8)
    ax.legend(fontsize=7)


def chasers(ax, data):
    """The search with one chaser to four, against their exact tour."""
    for waiting, name in ((False, "no_wait"), (True, "waiting")):
        rows = _feasible(_rows_waiting(_rows(data, method="mcrdv"), waiting))
        counts = sorted({int(row["instance.num_chasers"]) for row in rows})
        if not counts:
            continue
        spread = [
            _costs(
                [r for r in rows if int(r["instance.num_chasers"]) == count]
            )
            for count in counts
        ]
        plotter.line(
            ax,
            counts,
            [np.median(v) for v in spread],
            name=name,
            label=f"mcrdv, {name.replace('_', ' ')}",
            marker="o",
        )
        plotter.band(
            ax,
            counts,
            [np.percentile(v, 25) for v in spread],
            [np.percentile(v, 75) for v in spread],
            name=f"{name}_iqr",
        )
    exact = _costs(
        _feasible(_rows_waiting(_rows(data, method="tsp-exact"), False)),
    )
    if exact.size:
        span = ax.get_xlim()
        plotter.line(
            ax,
            list(span),
            [exact[0], exact[0]],
            name="exact_tsp",
            label="exact static TSP, no wait",
            linestyle="--",
            color="0.4",
        )
        ax.set_xlim(span)
    ax.set_xticks(sorted({int(r["instance.num_chasers"]) for r in data.rows}))
    ax.legend(fontsize=7)


# ============================================================================ #
# López Rivera & Naeije (2025): their missions and terms, one chaser
# ============================================================================ #


def _missions(rows):
    """Delta-v [km/s] on their terms, keyed by (size, mission)."""
    return {
        (int(row["num_targets"]), int(row.get("instance.draw", 0))): float(
            row["dv_lopez_rivera"]
        )
        / 1e3
        for row in _feasible(rows)
    }


def lopez_rivera(ax, data):
    """Mean delta-v of the dynamic RAAN walk and of mcrdv on missions drawn
    as theirs are, beside the means they report; the standard deviations
    are recorded beside the bars."""
    sizes = LOPEZ_RIVERA["targets"]
    ours = {
        method: _missions(_rows(data, method=method))
        for method in ("drw", "mcrdv")
    }
    bars, spread = {}, {}
    for key, label in (
        ("nco_dv", "nco_reported"),
        ("drw_dv", "drw_reported"),
    ):
        bars[label] = [mean for mean, _ in LOPEZ_RIVERA[key]]
        spread[key.replace("_dv", "_reported_std")] = [
            std for _, std in LOPEZ_RIVERA[key]
        ]
    for method, label in (("drw", "drw_ours"), ("mcrdv", "mcrdv_ours")):
        values = [
            np.array([v for (n, _), v in ours[method].items() if n == size])
            for size in sizes
        ]
        bars[label] = [v.mean() if v.size else np.nan for v in values]
        spread[f"{method}_ours_std"] = [
            v.std() if v.size else np.nan for v in values
        ]
        spread[f"{method}_ours_missions"] = [v.size for v in values]
    plotter.grouped_bars(ax, [str(size) for size in sizes], bars)
    plotter.series(
        ax, "spread", sizes, kind="marks", aux=tuple(spread), **spread
    )
    ax.legend(fontsize=7)


def lopez_rivera_gap(ax, data):
    """The mean gap of the dynamic RAAN walk to mcrdv on the same missions,
    beside the gaps they report against their genetic algorithm: a
    comparison that does not hang on the scale of the delta-v."""
    sizes = LOPEZ_RIVERA["targets"]
    walk = _missions(_rows(data, method="drw"))
    search = _missions(_rows(data, method="mcrdv"))
    gaps = [
        np.array(
            [
                100.0 * (walk[key] / dv - 1.0)
                for key, dv in search.items()
                if key[0] == size and key in walk
            ]
        )
        for size in sizes
    ]
    plotter.line(
        ax,
        sizes,
        [g.mean() if g.size else np.nan for g in gaps],
        name="drw_to_mcrdv",
        label="DRW to mcrdv, ours",
        marker="o",
    )
    for key, label, marker in (
        ("drw_gap", "DRW to their GA, reported", "s"),
        ("nco_gap", "NCO to their GA, reported", "^"),
    ):
        plotter.points(
            ax,
            sizes,
            [mean for mean, _ in LOPEZ_RIVERA[key]],
            name=key,
            label=label,
            marker=marker,
            facecolors="none",
            edgecolors="0.3",
        )
    ax.set_yscale("symlog", linthresh=10.0)
    ax.set_xticks(sizes)
    ax.legend(fontsize=7)


# ============================================================================ #
# Zona et al. (2023): Cerf's debris on their transfer model and grid
# ============================================================================ #


def horizon(ax, data):
    """The best plan over the seeds against the horizon, beside theirs.

    Which of their missions this is comes from `case` in the figure's entry,
    `[chasers, targets]`, so one function draws all three.
    """
    case = tuple(int(v) for v in data.spec["case"])
    rows = _feasible(_rows(data, method="mcrdv"))
    days = sorted({_days(row) for row in rows})
    best = [
        min(float(r["cost"]) for r in rows if _days(r) == day) for day in days
    ]
    plotter.line(ax, days, best, name="mcrdv", label="mcrdv", marker="o")

    theirs = ZONA_BEST.get(case, {})
    plotter.points(
        ax,
        list(theirs),
        list(theirs.values()),
        name="zona",
        label="Zona et al. (2023)",
        marker="s",
        facecolors="none",
        edgecolors="0.3",
    )
    fixed = _feasible(_rows(data, method="zona"))
    if fixed:
        plotter.points(
            ax,
            [_days(row) for row in fixed],
            [float(row["cost"]) for row in fixed],
            name="zona_tours",
            label="their tours, our oracle",
            marker="x",
            color="0.3",
        )
    ax.legend(fontsize=7)


def swarm(ax, data):
    """The whole catalogue at 720 days, from one chaser to six. A chaser
    count with no feasible run is left out, and said so in the title."""
    every = _rows(data, method="mcrdv")
    rows = _feasible(every)
    counts = sorted({int(row["instance.num_chasers"]) for row in rows})
    dropped = sorted(
        {int(row["instance.num_chasers"]) for row in every} - set(counts)
    )
    spread = [
        _costs(
            [r for r in rows if int(r["instance.num_chasers"]) == count],
            scale=1.0,
        )
        for count in counts
    ]
    plotter.line(
        ax,
        counts,
        [v.min() for v in spread],
        name="best",
        label="mcrdv, best",
        marker="o",
    )
    plotter.line(
        ax,
        counts,
        [np.median(v) for v in spread],
        name="median",
        label="mcrdv, median",
        linestyle="--",
    )

    search = ZONA_SEARCH[(4, 21)]
    for key, label, marker in (
        ("best", "Zona et al., best", "s"),
        ("island", "Zona et al., island GA mean", "D"),
        ("ga", "Zona et al., GA mean", "^"),
    ):
        plotter.points(
            ax,
            [4],
            [search[key]],
            name=f"zona_{key}",
            label=label,
            marker=marker,
            facecolors="none",
            edgecolors="0.3",
        )
    if dropped:
        plural = "s" if len(dropped) > 1 or dropped[0] > 1 else ""
        listed = ", ".join(str(count) for count in dropped)
        ax.set_title(
            f"no feasible plan with {listed} chaser{plural}", fontsize=8
        )
    ax.set_xticks(counts)
    ax.legend(fontsize=7)


DRAW_MAP = {
    "tours": tours,
    "chasers": chasers,
    "lopez_rivera": lopez_rivera,
    "lopez_rivera_gap": lopez_rivera_gap,
    "horizon": horizon,
    "swarm": swarm,
}


if __name__ == "__main__":
    raise SystemExit(plotter.main(__file__, DRAW_MAP))

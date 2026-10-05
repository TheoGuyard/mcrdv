#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import export, plotting, tidy  # noqa: E402


def _infeasible(data):
    rows = [row for group in data.groups for row in group] or data.rows
    return sum(row.get("feasible") is False for row in rows), len(rows)


def time(ax, data):
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
    export.band(ax, data.x, data.lo, data.hi, name="value_iqr")
    export.line(ax, data.x, data.median, name="value", marker="o")

    count, total = _infeasible(data)
    if count:
        ax.set_title(f"{count} of {total} runs exceed the limits", fontsize=8)


def throughput(ax, data):
    export.band(ax, data.x, data.lo, data.hi, name="throughput_iqr")
    export.line(ax, data.x, data.median, name="throughput", marker="o")


DRAW_MAP = {
    "time": time,
    "cost": cost,
    "throughput": throughput,
}


if __name__ == "__main__":
    raise SystemExit(plotting.main(__file__, DRAW_MAP))

#!/usr/bin/env python3
"""Showcase: solve, and keep the whole plan for the figures to read."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import analysis, driver, planners  # noqa: E402


def execute(config: dict) -> dict:
    """Solve, and describe the plan's structure alongside it.

    Both the aggregate metrics and the per-chaser breakdown are derived here,
    where the `Problem` is already in hand: the per-chaser numbers need
    drift-corrected right ascensions, and making a plotting script rebuild the
    catalogue to get them would be wasteful and easy to get subtly wrong.
    """
    result = planners.solve(config)
    problem = planners.problem(config)
    result.update(analysis.mission_metrics(result["mission"], problem))
    result["chasers"] = analysis.per_chaser(result["mission"], problem)
    # The assignment figures need the catalogue's drift, which only exists here:
    # a plotting script has the mission plan but not the orbits behind it, and
    # rebuilding them from the record would mean re-reading the catalogue. Every
    # element is kept, so which one a figure draws stays a plot.yaml decision
    # rather than something a run has to be repeated for.
    result["tracks"] = {
        element: analysis.tracks(problem, element)
        for element in analysis.ELEMENTS
    }
    result["visits"] = analysis.visits(result["mission"], problem)
    # The plan seen in time, and the spread of what its legs cost: both are
    # read off the mission plan alone, and both are what the structure figures
    # are about.
    result["timeline"] = analysis.timeline(result["mission"])
    result["legs"] = analysis.legs(result["mission"])
    return result


if __name__ == "__main__":
    raise SystemExit(driver.main(__file__, execute, warmup=planners.warm_up))

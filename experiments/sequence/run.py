#!/usr/bin/env python3
"""Sequence: what a preset of the search buys, and what it costs in time."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import config as configuration  # noqa: E402
from _common import driver, planners  # noqa: E402

# How close to its budget a run has to finish to count as having been stopped
# by the clock rather than by the search deciding it was done.
CLOSE = 0.98


def preset(config: dict) -> int:
    """The preset the run searched at.

    A configuration that names none gets `HgsConfig`'s own defaults, which are
    preset 4's; this experiment is about the presets, so leaving it out is a
    mistake rather than a default worth guessing at.
    """
    _, args = planners.layer(config, "sequence")
    if "preset" not in args:
        raise SystemExit(
            "a sequence run must name 'solver.args.sequence.args.preset'"
        )
    return int(args["preset"])


def budget(config: dict) -> float:
    """The wall-clock cap on the search, in seconds, or infinity."""
    _, args = planners.layer(config, "sequence")
    if "max_time" not in args:
        return float("inf")
    return float(configuration.duration(args["max_time"]))


def execute(config: dict) -> dict:
    """Solve, and say what the search was allowed and what it used.

    The two numbers this experiment is about -- what the plan cost and how long
    the search took to find it -- are already what a solve reports, so there is
    little to add. What does need adding is whether the run *finished*: a deep
    preset on a large catalogue will not exhaust its iterations inside any
    budget a campaign can give it, and a point that sits at the cap is the cap
    speaking rather than the preset. It is recorded rather than filtered out,
    because where the cap binds is worth seeing.
    """
    result = planners.solve(config)
    result["preset"] = preset(config)
    result["budget"] = budget(config)
    result["capped"] = result["t_solve"] >= CLOSE * result["budget"]
    result["t_total"] = result["t_setup"] + result["t_solve"]
    # A plan per run, seven presets over several catalogues and seeds: kept out
    # of the records, since nothing here reads a plan leg by leg.
    result.pop("mission", None)
    return result


if __name__ == "__main__":
    raise SystemExit(driver.main(__file__, execute, warmup=planners.warm_up))

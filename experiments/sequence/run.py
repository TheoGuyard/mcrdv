#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import runner, planners  # noqa: E402

# How close to its budget a run has to finish to count as having been stopped
# by the clock rather than by the search deciding it was done.
CLOSE = 0.98


def preset(config: dict) -> int:
    _, args = planners.layer(config, "sequence")
    if "preset" not in args:
        raise SystemExit(
            "a sequence run must name 'solver.args.sequence.args.preset'"
        )
    return int(args["preset"])


def budget(config: dict) -> float:
    _, args = planners.layer(config, "sequence")
    if "max_time" not in args:
        return float("inf")
    return float(runner.duration(args["max_time"]))


def execute(config: dict) -> dict:
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
    raise SystemExit(runner.main(__file__, execute, warmup=planners.warm_up))

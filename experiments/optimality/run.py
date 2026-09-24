#!/usr/bin/env python3
"""Optimality: the sequence layer against the exact optimum, on small cases."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import driver, exact, planners  # noqa: E402


def planner(config: dict) -> str:
    """One name per planner compared: the search at each preset, and whatever
    else by its type alone."""
    kind, args = planners.layer(config, "sequence")
    kind = kind or "hgs"
    preset = args.get("preset")
    if kind == "hgs" and preset is not None:
        return f"hgs-{preset}"
    return kind


def execute(config: dict) -> dict:
    """Enumerate the optimum, solve, and report how far apart they are.

    The optimum is recomputed by every run rather than shared between the runs
    of one instance: it takes seconds, and a run that depends on another run's
    output is one a job array can silently break.
    """
    problem = planners.problem(config)
    optimum = exact.solve(problem, planners.schedule(config))
    result = planners.solve(config)
    result.update(
        exact.report(
            optimum, result["cost"], result["feasible"], result["trace"]
        )
    )
    result["planner"] = planner(config)
    return result


def warm_up(configs) -> None:
    """Compile the planners, then the exact split, before anything is timed."""
    planners.warm_up(configs)
    exact.warm_up()


if __name__ == "__main__":
    raise SystemExit(driver.main(__file__, execute, warmup=warm_up))

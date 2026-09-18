#!/usr/bin/env python3
"""Baseline: the hybrid genetic search against the RAAN-sweep construction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import analysis, driver, planners  # noqa: E402


def execute(config: dict) -> dict:
    """Solve, and describe the shape of the plan as well as its cost.

    Two planners can reach a similar cost by very different means -- one using
    every chaser lightly, the other loading a few heavily -- and that difference
    is operationally the interesting part of beating a baseline.
    """
    result = planners.solve(config)
    result.update(
        analysis.mission_metrics(result["mission"], planners.problem(config))
    )
    return result


if __name__ == "__main__":
    raise SystemExit(driver.main(__file__, execute, warmup=planners.warm_up))

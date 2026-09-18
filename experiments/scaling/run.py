#!/usr/bin/env python3
"""Scaling: cost and running time against catalogue size."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import driver, planners  # noqa: E402


def execute(config: dict) -> dict:
    """Solve, then add what this experiment specifically wants to see.

    The setup share matters here. Building the table of leg bounds is quadratic
    in the catalogue size, so on the largest instance it eats into a fixed
    budget in a way it simply does not on the smallest, and a scaling curve that
    hides that is misleading.
    """
    result = planners.solve(config)
    total = result["t_setup"] + result["t_solve"]
    result["t_total"] = total
    result["setup_share"] = result["t_setup"] / total if total > 0 else 0.0
    result["cost_per_target"] = result["cost"] / max(result["num_targets"], 1)
    # The mission plan is not wanted here: 25 runs of it is a lot of pickle for
    # a figure that only ever reads scalars.
    result.pop("mission", None)
    return result


if __name__ == "__main__":
    raise SystemExit(driver.main(__file__, execute, warmup=planners.warm_up))

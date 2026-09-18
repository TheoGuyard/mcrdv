#!/usr/bin/env python3
"""Sensitivity: one-at-a-time sweeps over the search and layer parameters."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import driver, planners  # noqa: E402


def execute(config: dict) -> dict:
    """Solve, keeping the total time so the trade stays visible.

    Several of these knobs buy quality with time -- a finer departure grid, more
    neighbours, a larger population -- so a sweep reporting cost alone would
    rank the slowest setting best every time. The two are kept separate rather
    than folded into one number, which would only hide which of them moved.
    """
    result = planners.solve(config)
    result["t_total"] = result["t_setup"] + result["t_solve"]
    result.pop("mission", None)
    return result


if __name__ == "__main__":
    raise SystemExit(driver.main(__file__, execute, warmup=planners.warm_up))

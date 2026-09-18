#!/usr/bin/env python3
"""Convergence: the incumbent cost against elapsed search time."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import driver, planners  # noqa: E402


def execute(config: dict) -> dict:
    """Solve and keep whatever the run produced.

    Nothing here asks for the search log: `run.yaml` sets `log_save: true`, and
    that is what makes this experiment about convergence. A campaign that turned
    it off would still run, and would simply have nothing to plot -- which is
    the honest outcome, rather than a script quietly overriding its own
    configuration.
    """
    return planners.solve(config)


if __name__ == "__main__":
    raise SystemExit(driver.main(__file__, execute, warmup=planners.warm_up))

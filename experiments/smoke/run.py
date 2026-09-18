#!/usr/bin/env python3
"""Smoke check: one mcrdv run, so the pipeline can be exercised end to end."""

import sys
from pathlib import Path

# Keep `common` importable however the script is invoked -- from the repository
# root, from this folder, or by an absolute path on a compute node.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import driver, planners  # noqa: E402


def execute(config: dict) -> dict:
    """Solve, and keep the standard mcrdv numbers plus the plan itself."""
    return planners.solve(config)


if __name__ == "__main__":
    raise SystemExit(driver.main(__file__, execute, warmup=planners.warm_up))

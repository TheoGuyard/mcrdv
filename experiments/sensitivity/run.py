#!/usr/bin/env python3
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from mcrdv import Problem  # noqa: E402
from mcrdv.orbit import DAY, TAU  # noqa: E402

from _common import analysis, driver, planners  # noqa: E402


def spread(problem: Problem) -> float:
    angles = np.array([problem.states[t].r for t in problem.targets])
    if not angles.size:
        return 0.0
    centre = math.atan2(
        float(np.sin(angles).mean()), float(np.cos(angles).mean())
    )
    delta = np.abs(angles - centre) % TAU
    return float(np.degrees(np.minimum(delta, TAU - delta).mean()))


def shape(problem: Problem) -> dict:
    return {
        "num_chasers": problem.num_chasers,
        "max_load": problem.max_load,
        "horizon_days": problem.max_time / DAY,
        "capacity": problem.num_chasers * problem.max_load,
        "spread_deg": spread(problem),
    }


def execute(config: dict) -> dict:
    result = planners.solve(config)
    problem = planners.problem(config)

    result.update(analysis.mission_metrics(result["mission"], problem))
    result.update(shape(problem))

    chasers = analysis.per_chaser(result["mission"], problem)
    busy = sum(row["time"] for row in chasers if row["load"] > 0)
    # Waiting as a share of the time the swarm is underway: the absolute figure
    # grows with anything that lengthens the mission, and says less.
    result["wait_share"] = result["wait_total"] / busy if busy > 0 else 0.0

    total = result["t_setup"] + result["t_solve"]
    result["t_total"] = total
    result["setup_share"] = result["t_setup"] / total if total > 0 else 0.0
    result["cost_per_target"] = result["cost"] / max(result["num_targets"], 1)
    iterations = result.get("iterations")
    if iterations is not None and result["t_solve"] > 0:
        result["throughput"] = iterations / result["t_solve"]

    # A plan per run, over a hundred runs: nothing here reads one leg by leg.
    result.pop("mission", None)
    return result


if __name__ == "__main__":
    raise SystemExit(driver.main(__file__, execute, warmup=planners.warm_up))

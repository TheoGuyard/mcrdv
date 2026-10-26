#!/usr/bin/env python3
"""One instance parameter at a time, on synthetic clouds.

A synthetic cloud is one break-up: a parent on a circular orbit fragments at
one point, each fragment receives a random kick turned into orbital offsets by
the linearised Gauss equations, and J2 then spreads the nodes until the
mission starts. Configurations name it `instance.name: synthetic`, sized by
`num_targets`, seeded by `draw` and shaped by `instance.cloud` (see
`DEFAULTS`). For one draw, a smaller cloud is a prefix of a larger one.
"""

import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcrdv import Problem  # noqa: E402
from mcrdv.orbit import DAY, MU, RE, TAU, KepState  # noqa: E402

from _common import runner, planners  # noqa: E402

# The `instance.name` of a generated cloud.
NAME = "synthetic"

# The shape of a generated cloud, which `instance.cloud` may override.
DEFAULTS: Dict[str, Any] = {
    "altitude_km": 850.0,  # parent's circular orbit
    "inclination_deg": 98.8,  # parent's inclination
    "kick": 50.0,  # std of each kick component [m/s]
    "age": "3y",  # time from the break-up to the mission start
    "min_perigee_km": 250.0,  # lower fragments re-enter and are dropped
}

# Fragments drawn per batch. Fixed, so which fragments a cloud holds does not
# depend on how many are asked for.
_BATCH = 256


def cloud(count: int, draw: int = 0, **knobs: Any) -> List[KepState]:
    """`count` fragments of one break-up, observed `age` after it.

    A parent on a circular orbit fragments at one point, each fragment receives
    a random kick turned into orbital offsets by the linearised Gauss
    equations, and J2 then spreads the nodes until the mission starts. For one
    `draw`, a smaller cloud is a prefix of a larger one.
    """
    unknown = set(knobs) - set(DEFAULTS)
    if unknown:
        known = ", ".join(sorted(DEFAULTS))
        raise ValueError(
            f"unknown cloud parameter(s) {sorted(unknown)} (known: {known})"
        )
    if count < 1:
        raise ValueError("a cloud needs at least one fragment")

    shape = {**DEFAULTS, **knobs}
    a0 = RE + 1e3 * float(shape["altitude_km"])
    i0 = math.radians(float(shape["inclination_deg"]))
    sigma = float(shape["kick"])
    age = runner.duration(shape["age"])
    floor = RE + 1e3 * float(shape["min_perigee_km"])
    speed = math.sqrt(MU / a0)

    rng = np.random.default_rng(draw)
    # Argument of latitude of the break-up, shared by every fragment.
    u = float(rng.uniform(0.0, TAU))
    cos_u, sin_u = math.cos(u), math.sin(u)

    fragments: List[KepState] = []
    while len(fragments) < count:
        # Radial, along-track and out-of-plane kicks, and positions on orbit.
        kicks = rng.normal(0.0, sigma, size=(_BATCH, 3))
        anomalies = rng.uniform(0.0, TAU, size=_BATCH)

        for (radial, along, normal), anomaly in zip(kicks, anomalies):
            a = a0 + 2.0 * a0 * along / speed
            ex = (2.0 * cos_u * along + sin_u * radial) / speed
            ey = (2.0 * sin_u * along - cos_u * radial) / speed
            e = math.hypot(ex, ey)
            i = i0 + cos_u * normal / speed
            raan = sin_u * normal / (speed * math.sin(i0))
            if a * (1.0 - e) < floor:
                continue

            state = KepState(a, e, i, raan, math.atan2(ey, ex), anomaly)
            state = state.propagate(age)
            fragments.append(
                KepState(
                    state.a,
                    state.e,
                    state.i,
                    state.r % TAU,
                    state.o % TAU,
                    float(anomaly),
                )
            )
            if len(fragments) == count:
                break
    return fragments


planners.GENERATORS[NAME] = cloud


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

    result.update(planners.mission_metrics(result["mission"], problem))
    result.update(shape(problem))

    chasers = planners.per_chaser(result["mission"], problem)
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
    raise SystemExit(runner.main(__file__, execute, warmup=planners.warm_up))

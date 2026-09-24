#!/usr/bin/env python3
"""Transfer: how the definition of the transfer layer shapes a mission plan."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcrdv.orbit import DAY  # noqa: E402

from _common import analysis, driver, planners, transfers  # noqa: E402
from _common import config as configuration  # noqa: E402


def oracle(config: dict) -> str:
    """What the run's cost oracle is called, without its family name."""
    kind, _ = planners.layer(config, "transfer")
    kind = kind or "qlaw"
    return "fuel" if kind == "qlaw" else kind.replace("qlaw_", "")


def weight(config: dict) -> float:
    """What the oracle weighs time at, in m/s of delta-v per day.

    One number for the two oracles that trade fuel against time, so the plans
    they produce can be lined up along it; zero for the ones that do not.
    """
    kind, args = planners.layer(config, "transfer")
    fuel = float(args.get("factor_fuel", 1.0))
    if kind == "qlaw_arrival":
        per_second = float(args.get("factor_time", 0.0))
        return per_second * DAY / fuel if fuel > 0.0 else float("inf")
    if kind == "qlaw_urgency":
        return float(args.get("weight", 0.0)) / fuel if fuel > 0.0 else 0.0
    return 0.0


def yardstick(config: dict) -> dict:
    """The `metrics` block, with its durations read as seconds."""
    node = dict(config.get("metrics") or {})
    for key in ("blackout_period", "blackout_duration"):
        if key in node:
            node[key] = configuration.duration(node[key])
    return node


def execute(config: dict) -> dict:
    """Solve under one oracle, and describe the plan it produced.

    The plan's own `cost` is whatever its oracle weighed, so it says nothing
    across runs and no figure reads it. What the record keeps instead is the
    plan's *structure*, on a scale every oracle shares: every leg re-priced in
    delta-v at the departure the plan chose, every chaser's share of the work,
    and the aggregates of both. Which of those a figure reads is a plotting
    decision; this is simply the plan, written down.

    The hazard mask rides on the legs rather than being re-derived while
    drawing: it is a property of the catalogue, and a plan built by an oracle
    that never heard of it is still read against it -- which is what makes the
    runs comparable.
    """
    result = planners.solve(config)
    problem = planners.problem(config)

    # The Q-law with the run's physics and none of its objective: one scale
    # for plans built under oracles that disagree about what a leg is worth.
    delta_v = planners.delta_v(config)
    delta_v.initialize(problem)

    metrics = yardstick(config)
    legs = analysis.leg_table(result["mission"], delta_v)
    hazardous = transfers.hazardous(
        problem, float(metrics.get("hazard_share", 0.33))
    )
    for leg in legs:
        leg["hazardous"] = bool(hazardous[leg["target"]])

    result["legs"] = legs
    result["chasers"] = analysis.per_chaser(result["mission"], problem)
    result.update(analysis.mission_metrics(result["mission"], problem))
    result.update(analysis.fuel_and_time(result["mission"], delta_v))
    result.update(transfers.score(result["mission"], problem, metrics))

    result["oracle"] = oracle(config)
    result["weight"] = weight(config)
    return result


if __name__ == "__main__":
    raise SystemExit(driver.main(__file__, execute, warmup=planners.warm_up))

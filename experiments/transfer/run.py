#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcrdv.orbit import DAY  # noqa: E402

from _common import analysis, driver, planners, transfers  # noqa: E402
from _common import config as configuration  # noqa: E402


def oracle(config: dict) -> str:
    kind, _ = planners.layer(config, "transfer")
    kind = kind or "qlaw"
    return "fuel" if kind == "qlaw" else kind.replace("qlaw_", "")


def weight(config: dict) -> float:
    kind, args = planners.layer(config, "transfer")
    fuel = float(args.get("factor_fuel", 1.0))
    if kind == "qlaw_arrival":
        per_second = float(args.get("factor_time", 0.0))
        return per_second * DAY / fuel if fuel > 0.0 else float("inf")
    if kind == "qlaw_urgency":
        return float(args.get("weight", 0.0)) / fuel if fuel > 0.0 else 0.0
    return 0.0


def yardstick(config: dict) -> dict:
    node = dict(config.get("metrics") or {})
    for key in ("blackout_period", "blackout_duration"):
        if key in node:
            node[key] = configuration.duration(node[key])
    return node


def execute(config: dict) -> dict:
    result = planners.solve(config)
    problem = planners.problem(config)

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

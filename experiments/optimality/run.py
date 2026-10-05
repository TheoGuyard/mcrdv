#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import driver, exact, planners  # noqa: E402


def planner(config: dict) -> str:
    kind, args = planners.layer(config, "sequence")
    kind = kind or "hgs"
    preset = args.get("preset")
    if kind == "hgs" and preset is not None:
        return f"hgs-{preset}"
    return kind


def execute(config: dict) -> dict:
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
    planners.warm_up(configs)
    exact.warm_up()


if __name__ == "__main__":
    raise SystemExit(driver.main(__file__, execute, warmup=warm_up))

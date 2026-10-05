#!/usr/bin/env python3
import sys
from pathlib import Path

# Keep `common` importable however the script is invoked -- from the repository
# root, from this folder, or by an absolute path on a compute node.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from _common import benchmark, driver, planners  # noqa: E402
from _common import config as configuration  # noqa: E402


def execute(config: dict) -> dict:
    problem = planners.problem(config)
    layer = planners.schedule(config)
    reference = planners.schedule(
        configuration.derive(
            config, **{"solver.args.schedule": config.get("reference") or {}}
        )
    )

    spec = config.get("bank") or {}
    sequences = benchmark.bank(
        problem,
        benchmark.bounds(problem, planners.schedule(config)),
        lengths=spec.get("lengths", [2, 4, 8, 12, 16]),
        per_length=int(spec.get("per_length", 20)),
        width=int(spec.get("width", 10)),
        rng=np.random.default_rng(int(config.get("seed", 0))),
    )
    result = benchmark.measure(
        problem,
        layer,
        reference,
        sequences,
        repeats=int(config.get("repeats", 3)),
    )
    result["num_targets"] = problem.num_targets
    return result


if __name__ == "__main__":
    raise SystemExit(driver.main(__file__, execute, warmup=planners.warm_up))

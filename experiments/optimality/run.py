#!/usr/bin/env python3
"""The search against the exact optimum of small instances."""

import itertools
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcrdv import MissionPlan, Planner  # noqa: E402
from mcrdv.kernel import JIT, bind, schedule_evaluate  # noqa: E402
from mcrdv.planner import ScheduleLayer, SequenceLayer  # noqa: E402
from mcrdv.problem import DEPOT, ChaserPlan  # noqa: E402

from _common import planners, runner  # noqa: E402


@njit(**JIT)
def _split(best, num_targets, num_plans):
    """Cheapest cover of every target by at most `num_plans` disjoint plans."""
    size = 1 << num_targets
    value = np.full((num_plans + 1, size), np.inf)
    choice = np.zeros((num_plans + 1, size), dtype=np.int64)
    value[0, 0] = 0.0
    for k in range(1, num_plans + 1):
        value[k, 0] = 0.0
        for mask in range(1, size):
            best_value = value[k - 1, mask]
            best_choice = 0
            low = mask & -mask
            rest = mask ^ low
            sub = rest
            while True:
                part = sub | low
                cost = best[part]
                if cost < np.inf:
                    total = cost + value[k - 1, mask ^ part]
                    if total < best_value:
                        best_value = total
                        best_choice = part
                if sub == 0:
                    break
                sub = (sub - 1) & rest
            value[k, mask] = best_value
            choice[k, mask] = best_choice
    return value[num_plans, size - 1], choice


class OptimumSequence(SequenceLayer):
    """The cheapest feasible mission, by exhaustive enumeration of all possible
    sequences of at most `max_load` targets per chaser. After a solve,
    `priced` holds the number of sequences scheduled."""

    def __init__(
        self,
        schedule: ScheduleLayer | None = None,
        max_targets: int = 20
    ) -> None:
        super().__init__(schedule)
        self.priced = 0
        self.max_targets = max_targets

    def _orderings(self) -> Tuple[np.ndarray, Dict[int, Tuple[int, ...]], int]:
        n = self.problem.num_targets
        best = np.full(1 << n, np.inf)
        order: Dict[int, Tuple[int, ...]] = {}

        kernel = self.schedule.kernel()
        entry = bind(schedule_evaluate, kernel, np.zeros(2, dtype=np.int64))
        priced = 0
        for length in range(1, min(self.problem.max_load, n) + 1):
            for chosen in itertools.permutations(range(1, n + 1), length):
                sequence = (DEPOT,) + chosen
                cost, span, _, _, _ = entry(
                    kernel, np.array(sequence, dtype=np.int64)
                )
                priced += 1
                # The schedule layer's own notion of feasible: no time overrun,
                # and every leg flyable.
                if not math.isfinite(cost) or span > self.problem.max_time:
                    continue
                mask = 0
                for target in chosen:
                    mask |= 1 << (target - 1)
                if cost < best[mask]:
                    best[mask] = cost
                    order[mask] = sequence
        return best, order, priced

    def _evaluate(self) -> MissionPlan:
        n = self.problem.num_targets
        if n > self.max_targets:
            raise ValueError(
                f"{n} targets is too large (max {self.max_targets})"
            )

        best, order, self.priced = self._orderings()
        plans = min(self.problem.num_chasers, n)
        cost, choice = _split(best, n, plans)

        if not math.isfinite(cost):
            raise ValueError("no feasible mission exists on this instance")

        # Walk the split back, one chaser per subset of the optimum.
        sequences: List[Tuple[int, ...]] = []
        k, mask = plans, (1 << n) - 1
        while mask:
            part = int(choice[k, mask])
            if part:
                sequences.append(order[part])
                mask ^= part
            k -= 1

        tours = [self.schedule.evaluate(sequence) for sequence in sequences]
        idles = [ChaserPlan.empty()] * (self.problem.num_chasers - len(tours))
        return MissionPlan(
            [self.problem.refresh(plan) for plan in tours + idles]
        )


planners.SEQUENCES["optimum"] = OptimumSequence


# ============================================================================ #
# The experiment
# ============================================================================ #


def with_optimum(config: dict) -> dict:
    """The same configuration, with the optimum as its sequence layer."""
    return runner.derive(
        config, **{"solver.args.sequence": {"type": "optimum"}}
    )


def planner(config: dict) -> str:
    kind, args = planners.layer(config, "sequence")
    kind = kind or "hgs"
    preset = args.get("preset")
    if kind == "hgs" and preset is not None:
        return f"hgs-{preset}"
    return kind


def execute(config: dict) -> dict:
    problem = planners.problem(config)

    search = planners.sequence(config)
    exact = planners.sequence(with_optimum(config))

    start = time.perf_counter()
    mission = Planner(search).solve(problem, verbose=False)
    t_search = time.perf_counter() - start

    start = time.perf_counter()
    optimum = Planner(exact).solve(problem, verbose=False)
    t_exact = time.perf_counter() - start

    target = optimum.cost * (1.0 + 1e-9)
    usable = mission.feasible and optimum.cost > 0
    gap = 100.0 * (mission.cost / optimum.cost - 1.0) if usable else math.nan

    # When the search first held an optimal plan (needs `log_save: true`).
    trace = getattr(search, "trace", None) or ()
    reached = next(
        (e.time for e in trace if e.feasible and e.cost <= target), None
    )

    return {
        "planner": planner(config),
        "num_targets": problem.num_targets,
        "mission": mission,
        "cost": mission.cost,
        "feasible": mission.feasible,
        "optimum_mission": optimum,
        "optimum": optimum.cost,
        "gap": gap,
        "optimal": bool(usable and mission.cost <= target),
        "t_search": t_search,
        "t_exact": t_exact,
        "t_optimum": reached,
        "priced": exact.priced,
    }


def warmup(configs) -> None:
    planners.warm_up(list(configs) + [with_optimum(c) for c in configs])


if __name__ == "__main__":
    raise SystemExit(runner.main(__file__, execute, warmup=warmup))

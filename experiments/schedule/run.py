#!/usr/bin/env python3
"""The schedule layer on its own: how good a plan it finds, and how fast.

A bank of sequences, fixed by the instance, the transfer layer and the seed,
is scheduled by the layer under test and by a reference layer on a much finer
grid, which gives each sequence a gap and a time per call.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcrdv import Problem  # noqa: E402
from mcrdv.kernel import bind, schedule_evaluate  # noqa: E402
from mcrdv.planner import ScheduleLayer  # noqa: E402
from mcrdv.problem import DEPOT  # noqa: E402
from mcrdv.sequence.hgs import pair_lb  # noqa: E402

from _common import runner, planners  # noqa: E402

# Relative slack under which a gap counts as none.
TOLERANCE = 1e-9


def method(schedule: ScheduleLayer) -> str:
    """`no-wait`, `uniform+hints` or `uniform`: what a schedule layer does."""
    if not getattr(schedule, "allow_waiting", True):
        return "no-wait"
    if getattr(schedule, "use_hints", True):
        return "uniform+hints"
    return "uniform"


def bounds(problem: Problem, schedule: ScheduleLayer) -> np.ndarray:
    """The `(n, n)` per-leg bounds of the search, which the bank walks."""
    schedule.initialize(problem)
    return pair_lb(schedule, problem)


def bank(
    problem: Problem,
    bounds: np.ndarray,
    lengths: Sequence[int],
    per_length: int,
    width: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """`per_length` random walks from the depot of each length, each next
    target drawn among the `width` unvisited ones cheapest to reach (all of
    them when `width` is 0)."""
    num_targets = problem.num_targets
    out: List[np.ndarray] = []
    for length in lengths:
        if not 1 <= length <= num_targets:
            raise ValueError(
                f"cannot draw {length} targets out of {num_targets}"
            )
        for _ in range(per_length):
            free = np.ones(problem.num_states, dtype=bool)
            free[DEPOT] = False
            sequence = [DEPOT]
            for _ in range(length):
                candidates = np.flatnonzero(free)
                if width > 0:
                    near = np.argsort(
                        bounds[sequence[-1], candidates], kind="stable"
                    )
                    candidates = candidates[near[:width]]
                pick = int(candidates[rng.integers(candidates.size)])
                free[pick] = False
                sequence.append(pick)
            out.append(np.asarray(sequence, dtype=np.int64))
    return out


def _widths(kernel: Any) -> np.ndarray:
    """Width of every leg table the DP schedule has built (none for other
    schedule layers)."""
    legs = getattr(getattr(kernel, "context", None), "legs", None)
    if legs is None:
        return np.zeros(0)
    return np.array([table.shape[1] for table in legs.values()], dtype=float)


def measure(
    problem: Problem,
    schedule: ScheduleLayer,
    reference: ScheduleLayer,
    sequences: Sequence[np.ndarray],
    repeats: int = 3,
) -> Dict[str, Any]:
    """Schedule the bank with both layers, and report gaps and timings.

    Per-leg bounds are built before timing. A cold pass builds the leg tables;
    warm passes (`repeats`) time calls over tables already built, as in a
    search; their difference is the tabulation cost. The kernel is called
    directly, bypassing the layer's cache.
    """
    if not sequences:
        raise ValueError("an empty bank has nothing to measure")
    schedule.initialize(problem)
    reference.initialize(problem)

    bound = np.array([schedule.evaluate_lb(s) for s in sequences])
    for s in sequences:
        reference.evaluate_lb(s)

    kernel = schedule.kernel()
    entry = bind(schedule_evaluate, kernel, sequences[0])

    start = time.perf_counter()
    for s in sequences:
        entry(kernel, s)
    cold = time.perf_counter() - start

    calls = np.full(len(sequences), np.inf)
    cost = np.empty(len(sequences))
    span = np.empty(len(sequences))
    for _ in range(max(int(repeats), 1)):
        for index, s in enumerate(sequences):
            tick = time.perf_counter_ns()
            out = entry(kernel, s)
            calls[index] = min(calls[index], time.perf_counter_ns() - tick)
            cost[index], span[index] = out[0], out[1]
    calls *= 1e-9

    ref_kernel = reference.kernel()
    ref_entry = bind(schedule_evaluate, ref_kernel, sequences[0])
    best = np.array([ref_entry(ref_kernel, s)[0] for s in sequences])

    # Only sequences the reference schedules feasibly are scored.
    scored = np.isfinite(best) & (span <= problem.max_time)
    gap = np.full(len(sequences), np.nan)
    gap[scored] = 100.0 * (cost[scored] / best[scored] - 1.0)
    gaps = gap[scored]

    widths = _widths(kernel)
    tabulate = max(cold - float(calls.sum()), 0.0)
    return {
        "method": method(schedule),
        "num_sequences": len(sequences),
        "num_scored": int(scored.sum()),
        # Gaps to the reference [%], over the sequences scored.
        "gap_mean": float(gaps.mean()) if gaps.size else None,
        "gap_median": float(np.median(gaps)) if gaps.size else None,
        "gap_max": float(gaps.max()) if gaps.size else None,
        "optimal_share": (
            float(np.mean(gaps <= 100.0 * TOLERANCE)) if gaps.size else None
        ),
        # Seconds per call, over leg tables already built.
        "t_call": float(np.median(calls)),
        "t_call_mean": float(calls.mean()),
        "t_cold": cold,
        "t_tabulate": tabulate,
        "legs": int(widths.size),
        "grid_width": float(widths.mean()) if widths.size else None,
        "t_per_leg": tabulate / widths.size if widths.size else None,
        # Sequences whose bound exceeds the reference cost: must stay 0.
        "bound_violations": int(
            np.sum(bound[scored] > best[scored] * (1.0 + TOLERANCE))
        ),
        "sequences": {
            "legs": np.array([len(s) - 1 for s in sequences]),
            "cost": cost,
            "reference": best,
            "bound": bound,
            "gap": gap,
            "t_call": calls,
        },
    }


def execute(config: dict) -> dict:
    problem = planners.problem(config)
    layer = planners.schedule(config)
    reference = planners.schedule(
        runner.derive(
            config, **{"solver.args.schedule": config.get("reference") or {}}
        )
    )

    spec = config.get("bank") or {}
    sequences = bank(
        problem,
        bounds(problem, planners.schedule(config)),
        lengths=spec.get("lengths", [2, 4, 8, 12, 16]),
        per_length=int(spec.get("per_length", 20)),
        width=int(spec.get("width", 10)),
        rng=np.random.default_rng(int(config.get("seed", 0))),
    )
    result = measure(
        problem,
        layer,
        reference,
        sequences,
        repeats=int(config.get("repeats", 3)),
    )
    result["num_targets"] = problem.num_targets
    return result


if __name__ == "__main__":
    raise SystemExit(runner.main(__file__, execute, warmup=planners.warm_up))

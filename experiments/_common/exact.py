"""The exact optimum of a small instance, to measure a search against.

Exact with respect to the layers below the search. Every sequence of at most
`max_load` targets is scheduled once, the cheapest feasible ordering of each
subset of targets is kept, and a dynamic program over subsets then finds the
cheapest split of the catalogue into at most `num_chasers` plans. That is the
best mission any sequence layer could return on the same transfer and schedule
layers, so a gap measured against it belongs to the search alone -- not to the
time grid, and not to the transfer estimate.

Enumeration grows like `N^max_load` and the split like `3^N`, so this is for
instances of up to twenty or so targets, which is exactly what it is for: at 20
targets and 5 per chaser, it prices two million sequences and weighs a few
billion splits in around twenty seconds.
"""

from __future__ import annotations

import itertools
import math
import time
from typing import Any, Dict, List, NamedTuple, Sequence, Tuple

import numpy as np
from numba import njit

from mcrdv import Problem
from mcrdv.kernel import JIT, bind, schedule_evaluate
from mcrdv.planner import ScheduleLayer
from mcrdv.problem import DEPOT

# Past this many targets the split runs to tens of billions of subset pairs, and
# its tables to gigabytes.
MAX_TARGETS = 20

# Relative tolerance for calling a cost optimal. Both sides come out of the same
# dynamic program, so they differ only in the order the legs were summed.
TOLERANCE = 1e-9


class Optimum(NamedTuple):
    """The cheapest feasible mission, and what finding it took."""

    # Total cost, infinite when no feasible mission exists.
    cost: float
    # One sequence per chaser put to work, the depot first.
    sequences: List[Tuple[int, ...]]
    # Sequences the schedule layer was asked to price.
    priced: int
    # Seconds spent scheduling every sequence.
    t_enumerate: float
    # Seconds spent on the split into plans.
    t_split: float


def orderings(
    problem: Problem, schedule: ScheduleLayer
) -> Tuple[np.ndarray, Dict[int, Tuple[int, ...]], int]:
    """The cheapest feasible ordering of every subset of at most `max_load`.

    Subsets are bitmasks, target `d` being bit `d - 1`. Returns the cost of
    each subset (infinite where no ordering fits the mission horizon), the
    ordering reaching it, and how many sequences were priced.

    The schedule kernel is called directly rather than through
    `ScheduleLayer.evaluate`: every sequence is priced exactly once, so the
    layer's cache would only fill memory, and building a `ChaserPlan` for each
    of a few hundred thousand sequences would dominate the running time.
    """
    n = problem.num_targets
    best = np.full(1 << n, np.inf)
    order: Dict[int, Tuple[int, ...]] = {}

    kernel = schedule.kernel()
    entry = bind(schedule_evaluate, kernel, np.zeros(2, dtype=np.int64))
    priced = 0
    for length in range(1, min(problem.max_load, n) + 1):
        for chosen in itertools.permutations(range(1, n + 1), length):
            sequence = (DEPOT,) + chosen
            cost, span, _, _, _ = entry(
                kernel, np.array(sequence, dtype=np.int64)
            )
            priced += 1
            # The schedule layer's own notion of feasible: no time overrun,
            # and every leg flyable.
            if not math.isfinite(cost) or span > problem.max_time:
                continue
            mask = 0
            for target in chosen:
                mask |= 1 << (target - 1)
            if cost < best[mask]:
                best[mask] = cost
                order[mask] = sequence
    return best, order, priced


@njit(**JIT)
def _split(best, num_targets, num_plans):
    """Cheapest cover of every target by at most `num_plans` disjoint subsets.

    `value[k, mask]` is the cheapest cover of `mask` by at most `k` plans. The
    subset holding the lowest target of `mask` is enumerated over the submasks
    of the rest, which visits each split once rather than once per ordering of
    its parts. `choice[k, mask]` is that subset, or 0 where using fewer plans
    was as cheap.
    """
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


def solve(problem: Problem, schedule: ScheduleLayer) -> Optimum:
    """The cheapest feasible mission on `problem`, priced by `schedule`.

    The layer is initialized here, on this problem, so it can be handed over
    straight from `planners.schedule(config)`.
    """
    n = problem.num_targets
    if n > MAX_TARGETS:
        raise ValueError(
            f"{n} targets is too many to enumerate (at most {MAX_TARGETS})"
        )
    schedule.initialize(problem)

    start = time.perf_counter()
    best, order, priced = orderings(problem, schedule)
    enumerated = time.perf_counter()

    plans = min(problem.num_chasers, n)
    cost, choice = _split(best, n, plans)
    finished = time.perf_counter()

    sequences: List[Tuple[int, ...]] = []
    if math.isfinite(cost):
        k, mask = plans, (1 << n) - 1
        while mask:
            part = int(choice[k, mask])
            if part:
                sequences.append(order[part])
                mask ^= part
            k -= 1

    return Optimum(
        cost=float(cost),
        sequences=sequences,
        priced=priced,
        t_enumerate=enumerated - start,
        t_split=finished - enumerated,
    )


def report(
    optimum: Optimum,
    cost: float,
    feasible: bool,
    trace: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    """How a solve compares with the optimum, as columns of a result.

    `gap` is in percent and is left out for a plan that overruns a limit: an
    infeasible plan can be cheaper than the optimum, and a negative gap would
    read as the search beating the exhaustive enumeration. `t_optimum` needs a
    trace, and is the first logged moment the incumbent was both feasible and
    optimal -- `None` if that never happened, or if nothing was logged.
    """
    target = optimum.cost * (1.0 + TOLERANCE)
    usable = feasible and math.isfinite(optimum.cost) and optimum.cost > 0
    gap = 100.0 * (cost / optimum.cost - 1.0) if usable else math.nan
    reached = next(
        (
            entry["time"]
            for entry in trace
            if entry["feasible"] and entry["cost"] <= target
        ),
        None,
    )
    return {
        "optimum": optimum.cost,
        "optimum_chasers": len(optimum.sequences),
        "optimum_sequences": optimum.sequences,
        "gap": gap,
        "optimal": bool(usable and cost <= target),
        "t_optimum": reached,
        "t_exact": optimum.t_enumerate + optimum.t_split,
        "t_enumerate": optimum.t_enumerate,
        "t_split": optimum.t_split,
        "priced": optimum.priced,
    }


def warm_up() -> None:
    """Compile the split on a throwaway table, so that the first measured run
    does not pay the compiler inside `t_split`."""
    _split(np.zeros(4), 2, 1)

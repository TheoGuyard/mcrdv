"""The schedule layer on its own: how good a plan it finds, and how fast.

Inside a search the two are tangled up: a finer time grid prices each plan
better but leaves time for fewer of them, so an end-to-end cost at a fixed
budget mixes the scheduler's quality with the search's. Here they are pulled
apart. A fixed bank of sequences is drawn once per instance and seed; the layer
under test schedules every one, and a reference layer on a far finer grid
schedules them again, which gives each sequence a gap and a time per call.

The bank depends on the instance, the transfer layer and the seed, never on the
schedule layer being measured, so every configuration of a campaign that shares
those three is scored on exactly the same sequences.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence

import numpy as np

from mcrdv import Problem
from mcrdv.kernel import bind, schedule_evaluate
from mcrdv.planner import ScheduleLayer
from mcrdv.problem import DEPOT
from mcrdv.sequence.hgs import pair_lb

# Relative slack under which a gap counts as none: the two layers sum the same
# legs, but the reference may reach the same departures through another grid.
TOLERANCE = 1e-9


def method(schedule: ScheduleLayer) -> str:
    """A name for what a schedule layer does, to group results by.

    Waiting switched off makes the grid irrelevant, so `no-wait` stands alone;
    otherwise the name says whether the transfer layer's hints were merged in.
    """
    if not getattr(schedule, "allow_waiting", True):
        return "no-wait"
    if getattr(schedule, "use_hints", True):
        return "uniform+hints"
    return "uniform"


def bounds(problem: Problem, schedule: ScheduleLayer) -> np.ndarray:
    """The `(n, n)` per-leg bounds that the bank walks along.

    The search's own table, so that "cheap to reach" means here what it means
    to the neighbourhoods of the search.
    """
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
    """`per_length` sequences of each length, the depot first.

    Each one is a walk from the depot: the next target is drawn uniformly among
    the `width` still unvisited that are cheapest to reach by the per-leg bound,
    or among all of them when `width` is 0. A narrow width gives the kind of
    sequence a search ends up scheduling -- neighbours in orbit, one after the
    other -- and a wide one gives arbitrary sequences.
    """
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
    """Width of every leg table the layer has built so far.

    Reads the dynamic program's own table of legs where the layer keeps one;
    any other schedule layer reports none, and the tabulation columns come out
    empty rather than wrong.
    """
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

    The per-leg bounds are computed before anything is timed. A search builds
    all of them up front, so leaving them to the first call would bill a
    schedule for setup it never pays in practice.

    Two passes are timed. The cold one schedules each sequence once, building
    leg tables as it meets new legs; the warm ones run again over tables
    already built, which is the regime a search spends nearly all its time in.
    Their difference is what tabulation costs, reported per leg as well.

    The schedule kernel is called directly rather than through
    `ScheduleLayer.evaluate`, whose cache would answer the warm passes without
    running anything, and whose plan objects would be most of what got timed.
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

    # A sequence is scored only where the reference found a feasible schedule.
    # The no-wait schedule decides feasibility for every layer alike, so the
    # others fail there too, and a gap between two infinities is not a number.
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
        # Sequences whose summed per-leg bound exceeds the best schedule the
        # reference found. The screen in the search and the pruning in the
        # dynamic program both assume this never happens.
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

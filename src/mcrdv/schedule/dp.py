"""Schedule layer based on dynamic programming over discretized time grid."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from numba import njit, types
from numba.typed import Dict

from ..kernel import (
    JIT,
    KERNEL,
    ScheduleKernel,
    transfer_evaluate,
    transfer_evaluate_lb,
    transfer_hints,
)
from ..planner import ScheduleLayer


class DpContext(NamedTuple):
    # Candidate departure times shared by every leg, `grid_size + 1` of them.
    uniform: np.ndarray
    # Whether the transfer layer's hints are merged into each leg's grid.
    use_hints: bool
    # Whether the chaser may wait at an object before leaving it.
    allow_waiting: bool
    # Number of states, indexing the leg `i -> j` as `i * num_states + j`.
    num_states: int
    # Per-leg tables built on first use with rows `(times, dt, dc)`.
    legs: Any


# numba type of a leg table.
LEG = types.float64[:, ::1]


@njit(**JIT)
def _merge(grid1, grid2, tolerance):
    """Sorted union of two time grids, dropping any closer than `tolerance`
    to the one before. Kept apart from the kernel so that its sort, which is
    slow to compile, is cached."""
    grid = np.sort(np.concatenate((grid1, grid2)))
    keep = np.empty(grid.size, dtype=np.bool_)
    keep[0] = True
    for k in range(1, grid.size):
        keep[k] = grid[k] - grid[k - 1] > tolerance
    return grid[keep]


@njit(**KERNEL)
def _leg(kernel, i, j):
    """The table of the leg `i -> j`, built on first use."""
    context = kernel.context
    k = i * context.num_states + j
    if k in context.legs:
        return context.legs[k]

    # Merge uniform and leg-wise hints into a single time grid.
    transfer = kernel.transfer
    grid = context.uniform
    if context.use_hints:
        hints = transfer_hints(transfer, i, j)
        if hints.size > 0:
            grid = _merge(context.uniform, hints, 1e-9 * transfer.max_time)

    # Build the table of `(times, dt, dc)` for the leg `i -> j` over the grid.
    table = np.empty((3, grid.size))
    for t in range(grid.size):
        dt, dc = transfer_evaluate(transfer, i, j, grid[t])
        table[0, t] = grid[t]
        table[1, t] = dt
        table[2, t] = dc

    context.legs[k] = table
    return table


@njit(**KERNEL)
def _dp_schedule_evaluate(kernel, seq):
    """Schedule one chaser along a fixed sequence of at least two states."""
    context = kernel.context
    transfer = kernel.transfer
    max_time = transfer.max_time
    nseq = len(seq)
    legs = nseq - 1
    depart = np.zeros(nseq)
    arrive = np.zeros(nseq)
    costs = np.zeros(nseq)

    # -------------------------- No-wait schedule -------------------------- #
    clock = 0.0
    total = 0.0
    for i in range(legs):
        dt, dc = transfer_evaluate(transfer, seq[i], seq[i + 1], clock)
        depart[i] = clock
        clock += dt
        arrive[i + 1] = clock
        costs[i + 1] = dc
        total += dc
    depart[legs] = arrive[legs]

    # If no-wait schedule exceeds the horizon, no feasible schedule exists.
    if not context.allow_waiting or arrive[legs] > max_time:
        return total, arrive[legs], depart, arrive, costs

    # A feasible no-wait schedule provides an upper bound on the schedule cost.
    upper = total

    # Suffix sums of the per-leg lower bounds used in pruning.
    tail = np.zeros(nseq)
    for i in range(legs - 1, -1, -1):
        _, cost_lb = transfer_evaluate_lb(transfer, seq[i], seq[i + 1])
        tail[i] = tail[i + 1] + cost_lb

    # Fill the tables of `(times, dt, dc)` for each leg.
    table = _leg(kernel, seq[0], seq[1])
    widest = max(1, table.shape[1])
    tables = [table]
    for i in range(1, legs):
        table = _leg(kernel, seq[i], seq[i + 1])
        widest = max(widest, table.shape[1])
        tables.append(table)

    # ---------------------------- Forward pass ----------------------------- #

    # `cost[i, k]`: minimum cost to be ready to depart at the `k`-th time from
    # the `i`-th object early enough.
    # `back[i, k]`: index of the departure time from the previous object that
    # led to the minimum cost in `cost[i, k]`.
    cost = np.full((nseq, widest), np.inf)
    back = np.full((nseq, widest), -1, dtype=np.int64)

    cost[0, : tables[0].shape[1]] = 0.0
    for i in range(legs):
        table = tables[i]
        times = table[0]
        dts = table[1]
        dcs = table[2]
        terminal = i + 1 == legs
        if terminal:
            next_times = times  # the terminal grid is the horizon alone
            nwidth = 1
        else:
            next_times = tables[i + 1][0]
            nwidth = next_times.size

        row = cost[i]
        next_cand = cost[i + 1]
        next_back = back[i + 1]
        budget = upper - tail[i + 1]

        for l in range(times.size):
            # Skip if departure time is already worse than the best schedule.
            here_cost = row[l]
            if here_cost == np.inf:
                continue
            # Skip if the arrival time is beyond the mission horizon.
            arrival = times[l] + dts[l]
            if arrival > max_time:
                continue
            # Skip if the cost of the leg is already beyond the best schedule.
            candidate = here_cost + dcs[l]
            if candidate > budget:
                continue
            # Compute earliest possible departure from the next object.
            if terminal:
                k = 0
            else:
                lo = 0
                hi = nwidth
                while lo < hi:
                    mid = (lo + hi) >> 1
                    if next_times[mid] < arrival:
                        lo = mid + 1
                    else:
                        hi = mid
                if lo >= nwidth:
                    continue
                k = lo
            if candidate < next_cand[k]:
                next_cand[k] = candidate
                next_back[k] = l

        # Waiting is free so being ready earlier is never worse.
        for k in range(1, nwidth):
            if next_cand[k - 1] < next_cand[k]:
                next_cand[k] = next_cand[k - 1]
                next_back[k] = next_back[k - 1]

    # Return if the last leg is infeasible
    if not np.isfinite(cost[legs, 0]):
        return total, arrive[legs], depart, arrive, costs

    # ----------------------------- Backtracking ---------------------------- #

    # Retrieve best departure times from the backtracking table.
    chosen = np.empty(legs, dtype=np.int64)
    k = 0
    for i in range(legs, 0, -1):
        l = back[i, k]
        if l < 0:
            return total, arrive[legs], depart, arrive, costs
        chosen[i - 1] = l
        k = l

    # Rebuild arrival times and leg costs from the chosen departure times.
    total = 0.0
    for i in range(legs):
        table = tables[i]
        l = chosen[i]
        depart[i] = table[0, l]
        arrive[i + 1] = table[0, l] + table[1, l]
        costs[i + 1] = table[2, l]
        total += table[2, l]
    depart[legs] = arrive[legs]

    return total, arrive[legs], depart, arrive, costs


class DpSchedule(ScheduleLayer):
    """Schedule layer based on dynamic programming over discretized time.

    Each leg gets a grid of candidate departure times (`grid_size + 1` uniform
    points over the mission horizon, plus the transfer layer's hints) and the
    transfer is tabulated at those points the first time the leg is used. The
    dynamic program then only reads tables, pruning against the transfer
    layer's per-leg bounds. It assumes that leaving later never arrives
    earlier (a FIFO transfer), which is what lets a no-wait schedule that
    overruns the mission horizon be returned as it is.

    Parameters
    ----------
    allow_waiting : bool
        Whether the chaser is allowed to wait at each object before leaving.
    grid_size : int
        Number of uniform steps in each leg's grid of departure times.
    use_hints : bool
        Whether to merge the transfer layer's hints into the uniform grid.
    **kwargs : dict
        Keyword arguments passed to the :class:`ScheduleLayer` base class.
    """

    def __init__(
        self,
        *,
        allow_waiting: bool = True,
        grid_size: int = 32,
        use_hints: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.allow_waiting = bool(allow_waiting)
        self.grid_size = int(grid_size)
        self.use_hints = bool(use_hints)

        if self.grid_size < 1:
            raise ValueError("parameter 'grid_size' must be positive")

    def _kernel(self) -> ScheduleKernel:
        s = self.grid_size
        uniform = self.problem.max_time * np.arange(s + 1) / s
        context = DpContext(
            uniform,
            self.use_hints,
            self.allow_waiting,
            self.problem.num_states,
            Dict.empty(types.int64, LEG),
        )
        return ScheduleKernel(context, _dp_schedule_evaluate)

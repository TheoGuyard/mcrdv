"""Compiled kernels for planner layers."""

import math
import warnings
from typing import Any, NamedTuple
from weakref import WeakValueDictionary

import numpy as np
from numba import config, njit, objmode, typeof, types
from numba.core.errors import NumbaWarning

# ========================================================================= #
# Numba utilities
# ========================================================================= #

# Compilation kernel specifications.
JIT = {"cache": True, "nogil": True}

# Compilation kernel specifications for inline functions.
JIT_INLINE = {"cache": True, "nogil": True, "inline": "always"}

# Compilation kernel specifications for parallel evaluation.
JIT_PARALLEL = {"cache": True, "nogil": True, "parallel": True}

# Compilation specifications for functions taking a layer kernel.
KERNEL = {"nogil": True}


def bind(fn, *args):
    """Compile `fn` for the types of `args` and return its entry point to avoid
    re-deriving the types on every call.
    """
    if config.DISABLE_JIT:
        return fn
    sig = tuple(typeof(arg) for arg in args)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "Code running in object mode", NumbaWarning
        )
        fn.compile(sig)
    return fn.overloads[sig].entry_point


# ========================================================================= #
# Layer kernels
# ========================================================================= #


class TransferKernel(NamedTuple):
    """Compiled form of a transfer layer.

    The first fields are set up in `TransferLayer._kernel`, and the rest are
    filled in by `TransferLayer.initialize`.
    """

    # Layer data required to evaluate its functions.
    context: Any
    # `(kernel, i, j, t) -> (dt, dc)` of the transfer `i -> j` leaving at `t`.
    evaluate: Any
    # `(kernel, i, j) -> (dt, dc)` bounding the leg over `[0, max_time]`.
    evaluate_lb: Any = None
    # `(kernel, i, j) -> float64[::1]` of departure times where the leg is cheap.
    hints: Any = None

    # Maximum mission time in the problem instance bound to the layer [s].
    max_time: float = 0.0
    # Number of departures sampled by the default bound and hints.
    num_samples: int = 0
    # Number of hints returned by the default hints.
    num_hints: int = 0
    # Typed dict memo `(i, j, t) -> (dt, dc)`, or `None` when not memoizing.
    cache: Any = None
    # `(n, n, 2)` table of `(dt, dc)` bounds, or `None` when not memoizing.
    cache_lb: Any = None


class ScheduleKernel(NamedTuple):
    """Compiled form of a schedule layer.

    A layer writes the first fields in `ScheduleLayer._kernel`, and the rest
    are filled in by `ScheduleLayer.initialize`.
    """

    # Layer data required to evaluate its functions.
    context: Any
    # `(kernel, seq) -> (cost, time, depart, arrive, costs)`.
    evaluate: Any
    # `(kernel, seq) -> float` bounding the cost of any plan along `seq`.
    evaluate_lb: Any = None

    # Kernel of the transfer layer bound to the schedule layer.
    transfer: Any = None


class SequenceKernel(NamedTuple):
    """Compiled form of a sequence layer.

    A layer writes the first fields in `SequenceLayer._kernel`, and the rest
    are filled in by `SequenceLayer.initialize`.
    """

    # Layer data required to evaluate its functions.
    context: Any
    # `(kernel) -> (offsets, states, depart, arrive, costs, cost, time)` with
    # one plan per chaser: plan `c` spans `offsets[c]:offsets[c + 1]` of the
    # three per-state arrays, and has total cost `cost[c]` and time `time[c]`.
    evaluate: Any

    # Kernel of the schedule layer bound to the sequence layer.
    schedule: Any = None


# numba types of the transfer memoization
TRANSFER_CACHE_KEY = types.Tuple((types.int64, types.int64, types.float64))
TRANSFER_CACHE_VALUE = types.UniTuple(types.float64, 2)

# numba types of the schedule memoization
SCHEDULE_CACHE_KEY = types.UniTuple(types.int64, 2)
SCHEDULE_CACHE_VALUE = types.float64


# ========================================================================= #
# Transfer functions
# ========================================================================= #


@njit(**JIT_INLINE)
def _sample_time(k: int, num_samples: int, t: float) -> float:
    """The `k`-th of `num_samples` departure times spanning `[0, t]`."""
    if num_samples <= 1:
        return 0.0
    return t * k / (num_samples - 1)


@njit(**JIT)
def _cheapest_minima(costs, num_hints, max_time):
    """Sample times of the `num_hints` cheapest local minima of a profile."""
    num_samples = costs.size

    # Strict interior local minima of the sampled profile.
    keep = np.empty(num_samples, dtype=np.int64)
    count = 0
    for k in range(1, num_samples - 1):
        if costs[k] < costs[k - 1] and costs[k] <= costs[k + 1]:
            keep[count] = k
            count += 1

    # A profile that only decreases towards an endpoint is minimal there.
    if costs[0] <= costs[1]:
        keep[count] = 0
        count += 1
    if costs[num_samples - 1] < costs[num_samples - 2]:
        keep[count] = num_samples - 1
        count += 1
    keep = np.sort(keep[:count])

    # Too many windows: keep the cheapest, then restore time order.
    if count > num_hints:
        order = np.argsort(costs[keep], kind="mergesort")
        keep = np.sort(keep[order[:num_hints]])
        count = num_hints

    out = np.empty(count)
    for k in range(count):
        out[k] = _sample_time(keep[k], num_samples, max_time)
    return out


@njit(**KERNEL)
def _memoized_evaluate(kernel, cache, i, j, t):
    if cache is None:
        return kernel.evaluate(kernel, i, j, t)
    key = (i, j, t)
    if key in cache:
        return cache[key]
    value = kernel.evaluate(kernel, i, j, t)
    cache[key] = value
    return value


@njit(**KERNEL)
def _memoized_evaluate_lb(kernel, cache, i, j):
    if cache is None:
        return kernel.evaluate_lb(kernel, i, j)
    if not math.isnan(cache[i, j, 1]):
        return cache[i, j, 0], cache[i, j, 1]
    dt, dc = kernel.evaluate_lb(kernel, i, j)
    cache[i, j, 0] = dt
    cache[i, j, 1] = dc
    return dt, dc


@njit(**KERNEL)
def transfer_evaluate(kernel, i, j, t):
    """Time and cost of the transfer `i -> j` leaving at `t`."""
    return _memoized_evaluate(kernel, kernel.cache, i, j, t)


@njit(**KERNEL)
def transfer_evaluate_lb(kernel, i, j):
    """Bounds on the time and cost of the leg `i -> j` over the mission."""
    return _memoized_evaluate_lb(kernel, kernel.cache_lb, i, j)


@njit(**KERNEL)
def transfer_hints(kernel, i, j):
    """Departure times over the mission where the leg `i -> j` is cheap."""
    return kernel.hints(kernel, i, j)


@njit(**KERNEL)
def transfer_evaluate_lb_default(kernel, i, j):
    """Default implementation for `TransferLayer.evaluate_lb`. Sample times
    for transfer departures and return the cheapest."""
    num_samples = kernel.num_samples
    best_dt = np.inf
    best_dc = np.inf
    for k in range(num_samples):
        t = _sample_time(k, num_samples, kernel.max_time)
        dt, dc = kernel.evaluate(kernel, i, j, t)
        if dt < best_dt:
            best_dt = dt
        if dc < best_dc:
            best_dc = dc
    return best_dt, best_dc


@njit(**KERNEL)
def transfer_hints_default(kernel, i, j):
    """Default implementation for `TransferLayer.hints`. Sample times for
    transfer departures and return the best hints."""
    num_samples = kernel.num_samples
    num_hints = kernel.num_hints
    max_time = kernel.max_time
    if num_samples <= num_hints or num_samples < 2:
        count = min(num_samples, num_hints)
        out = np.empty(count)
        for k in range(count):
            out[k] = _sample_time(k, num_samples, max_time)
        return out

    costs = np.empty(num_samples)
    for k in range(num_samples):
        _, costs[k] = kernel.evaluate(
            kernel, i, j, _sample_time(k, num_samples, max_time)
        )
    return _cheapest_minima(costs, num_hints, max_time)


# ========================================================================= #
# Schedule functions
# ========================================================================= #


@njit(**KERNEL)
def schedule_evaluate(kernel, seq):
    n = len(seq)
    if n < 2:
        return 0.0, 0.0, np.zeros(n), np.zeros(n), np.zeros(n)
    return kernel.evaluate(kernel, seq)


@njit(**KERNEL)
def schedule_evaluate_lb(kernel, seq):
    return kernel.evaluate_lb(kernel, seq)


@njit(**KERNEL)
def schedule_evaluate_lb_default(kernel, seq):
    """Default implementation for `ScheduleLayer.evaluate_lb`. Sum per-leg
    lower bounds."""
    transfer = kernel.transfer
    total_lb = 0.0
    for k in range(len(seq) - 1):
        total_lb += transfer_evaluate_lb(transfer, seq[k], seq[k + 1])[1]
    return total_lb


# ========================================================================= #
# Sequence functions
# ========================================================================= #


@njit(**KERNEL)
def sequence_evaluate(kernel):
    return kernel.evaluate(kernel)


# ========================================================================= #
# Bridges to pure-python layers
# ========================================================================= #

# Python layers stand behind kernels whose functions call `objmode` hooks.
PY_LAYERS: WeakValueDictionary[int, Any] = WeakValueDictionary()


def register(layer: Any) -> int:
    """Make `layer` reachable from compiled code, returning its key."""
    key = id(layer)
    PY_LAYERS[key] = layer
    return key


def _call_pair(key, name, *args):
    return getattr(PY_LAYERS[key], name)(*args)


def _call_hints(key, i, j):
    hints = PY_LAYERS[key]._hints(i, j)
    return np.ascontiguousarray(hints, dtype=np.float64)


def _call_plan(key, seq):
    plan = PY_LAYERS[key]._evaluate(tuple(int(s) for s in seq))
    return (
        plan.cost,
        plan.time,
        np.ascontiguousarray(plan.depart, dtype=np.float64),
        np.ascontiguousarray(plan.arrive, dtype=np.float64),
        np.ascontiguousarray(plan.costs, dtype=np.float64),
    )


def _call_plan_lb(key, seq):
    return PY_LAYERS[key]._evaluate_lb(tuple(int(s) for s in seq))


@njit(**KERNEL)
def python_transfer_evaluate(kernel, i, j, t):
    key = kernel.context
    with objmode(dt="float64", dc="float64"):
        dt, dc = _call_pair(key, "_evaluate", i, j, t)
    return dt, dc


@njit(**KERNEL)
def python_transfer_evaluate_lb(kernel, i, j):
    key = kernel.context
    with objmode(dt="float64", dc="float64"):
        dt, dc = _call_pair(key, "_evaluate_lb", i, j)
    return dt, dc


@njit(**KERNEL)
def python_transfer_hints(kernel, i, j):
    key = kernel.context
    with objmode(out="float64[::1]"):
        out = _call_hints(key, i, j)
    return out


@njit(**KERNEL)
def python_schedule_evaluate(kernel, seq):
    key = kernel.context
    with objmode(
        cost="float64",
        span="float64",
        depart="float64[::1]",
        arrive="float64[::1]",
        costs="float64[::1]",
    ):
        cost, span, depart, arrive, costs = _call_plan(key, seq)
    return cost, span, depart, arrive, costs


@njit(**KERNEL)
def python_schedule_evaluate_lb(kernel, seq):
    key = kernel.context
    with objmode(value="float64"):
        value = _call_plan_lb(key, seq)
    return value

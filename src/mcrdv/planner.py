"""Mission planner with nested layer decomposition."""

import time
from abc import ABC
from typing import Any, Sequence, Tuple

import numpy as np
from numba.typed import Dict

from .kernel import (
    TRANSFER_CACHE_KEY,
    TRANSFER_CACHE_VALUE,
    ScheduleKernel,
    SequenceKernel,
    TransferKernel,
    bind,
    python_schedule_evaluate,
    python_schedule_evaluate_lb,
    python_transfer_evaluate,
    python_transfer_evaluate_lb,
    python_transfer_hints,
    register,
    transfer_hints_default,
    transfer_evaluate_lb_default,
    schedule_evaluate,
    schedule_evaluate_lb,
    schedule_evaluate_lb_default,
    sequence_evaluate,
    transfer_evaluate,
    transfer_evaluate_lb,
    transfer_hints,
)
from .problem import ChaserPlan, MissionPlan, Problem


def _overrides(layer: object, base: type, name: str) -> bool:
    """Whether the class of `layer` redefines the method `name` of `base`."""
    return getattr(type(layer), name) is not getattr(base, name)


# ========================================================================= #
# Layers
# ========================================================================= #


class TransferLayer(ABC):
    """Estimates the time and cost of transfers between states.

    A derived layer is written in one of two ways:

    - In Python by implementing `_evaluate`. Optionally, the functions
      `_evaluate_lb` and `_hints` can be implemented to replace the defaults.
    - In compiled mode, by returning a `TransferKernel` from `_kernel`, which
      can also expose `evaluate_lb` and `hints` to replace the defaults.

    Either way callers see the same `evaluate`, `evaluate_lb`, `hints` and
    `kernel`, all running through the compiled functions of `mcrdv.kernel`. A
    layer written in Python is reached from compiled code through `objmode`.

    Parameters
    ----------
    num_samples : int
        Number of departures sampled by the default bound and hints.
    num_hints : int
        Number of hints returned by the default hints.
    use_cache : bool
        Whether to memoize `evaluate` on its exact arguments. Only worth it
        when a transfer costs well over a hash lookup (~50 ns).
    use_cache_lb : bool
        Whether to memoize `evaluate_lb` in an `(n, n)` table.
    """

    # Problem instance tied to the layer (set during `initialize`).
    problem: Problem | None = None

    # Parameters controlling the default implementations and caching.
    num_samples: int = 64
    num_hints: int = 16
    use_cache: bool = False
    use_cache_lb: bool = True

    def __init__(
        self,
        *,
        num_samples: int = 64,
        num_hints: int = 16,
        use_cache: bool = False,
        use_cache_lb: bool = True,
    ) -> None:
        self.num_samples = int(num_samples)
        self.num_hints = int(num_hints)
        self.use_cache = bool(use_cache)
        self.use_cache_lb = bool(use_cache_lb)

        if self.num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        if self.num_hints < 0:
            raise ValueError("num_hints must be non-negative")

        # Everything below is set up by `initialize`.
        self._compiled: TransferKernel | None = None
        self._evaluate_entry = None
        self._evaluate_lb_entry = None
        self._hints_entry = None

    def initialize(self, problem: Problem) -> None:
        """Bind the layer to a problem instance, and clear outdated data."""
        self.problem = problem

        kernel = self._kernel()
        if kernel is None:
            has_evaluate_lb = _overrides(self, TransferLayer, "_evaluate_lb")
            has_hints = _overrides(self, TransferLayer, "_hints")
            kernel = TransferKernel(
                register(self),
                python_transfer_evaluate,
                python_transfer_evaluate_lb if has_evaluate_lb else None,
                python_transfer_hints if has_hints else None,
            )
        else:
            for name in ("_evaluate_lb", "_hints"):
                if _overrides(self, TransferLayer, name):
                    raise TypeError(
                        f"{type(self).__name__} publishes a kernel, so "
                        f"'{name[1:]}' is defined by the kernel, not in Python"
                    )

        n = problem.num_states

        evaluate_lb = kernel.evaluate_lb
        if evaluate_lb is None:
            evaluate_lb = transfer_evaluate_lb_default

        hints = kernel.hints
        if hints is None:
            hints = transfer_hints_default

        if self.use_cache:
            cache = Dict.empty(TRANSFER_CACHE_KEY, TRANSFER_CACHE_VALUE)
        else:
            cache = None

        if self.use_cache_lb:
            cache_lb = np.full((n, n, 2), np.nan)
        else:
            cache_lb = None

        self._compiled = kernel._replace(
            evaluate_lb=evaluate_lb,
            hints=hints,
            max_time=float(problem.max_time),
            num_samples=self.num_samples,
            num_hints=self.num_hints,
            cache=cache,
            cache_lb=cache_lb,
        )

        compiled = self._compiled
        self._evaluate_entry = bind(transfer_evaluate, compiled, 0, 0, 0.0)
        self._evaluate_lb_entry = bind(transfer_evaluate_lb, compiled, 0, 0)
        self._hints_entry = bind(transfer_hints, compiled, 0, 0)

    def evaluate(self, i: int, j: int, t: float) -> Tuple[float, float]:
        """Return the (time, cost) of the transfer `i -> j` at time `t`."""
        return self._evaluate_entry(self._compiled, i, j, t)

    def evaluate_lb(self, i: int, j: int) -> Tuple[float, float]:
        """Lower bounds on :meth:`evaluate` over the mission `[0, max_time]`."""
        return self._evaluate_lb_entry(self._compiled, i, j)

    def hints(self, i: int, j: int) -> np.ndarray:
        """Departure times in `[0, max_time]` where :meth:`evaluate` is cheap."""
        return self._hints_entry(self._compiled, i, j)

    def kernel(self) -> TransferKernel:
        """The compiled form of the layer, assembled by `initialize`."""
        return self._compiled

    # --------------- Hooks that derived layers can implement --------------- #

    def _kernel(self) -> TransferKernel | None:
        """Compiled layer kernel, or `None` for pure Python layers.

        When some `TransferKernel` is returned, the layer is compiled and the
        hooks `_evaluate`, `_evaluate_lb`, and `_hints` are ignored. When
        `None` is returned, the layer is pure Python and the hooks are used
        instead.
        """
        return None

    def _evaluate(self, i: int, j: int, t: float) -> Tuple[float, float]:
        """Evaluation of the (time, cost) of transfer `i -> j` at time `t`."""
        if self._compiled is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not implement '_evaluate'"
            )
        else:
            raise RuntimeError(
                f"{type(self).__name__} publishes a kernel, so the pure "
                "Python '_evaluate' should never be called in Python."
            )

    def _evaluate_lb(self, i: int, j: int) -> Tuple[float, float]:
        """Lower bounds on :meth:`evaluate` over all `t in [0, max_time]`.

        If not is overridden, defaults to `transfer_evaluate_lb_default`.
        """
        raise NotImplementedError

    def _hints(self, i: int, j: int) -> np.ndarray:
        """Times where :meth:`evaluate` is cheap over `t in [0, max_time]`.

        If not is overridden, defaults to `transfer_hints_default`.
        """
        raise NotImplementedError


# Imports default layer here to avoid circular dependencies.
from .transfer.qlaw import QlawTransfer  # noqa: E402


class ScheduleLayer(ABC):
    """Schedules a complete chaser plan along a sequence of targets to visit.

    A derived layer is written in one of two ways:

    - In Python by implementing `_evaluate`. Optionally, the function
      `_evaluate_lb` can be implemented to replace the default.
    - In compiled mode, by returning a `ScheduleKernel` from `_kernel`, which
      can also expose `evaluate_lb` to replace the default.

    Either way callers see the same `evaluate`, `evaluate_lb`, and `kernel`,
    all running through the compiled functions of `mcrdv.kernel`. A layer
    written in Python is reached from compiled code through `objmode`.

    Parameters
    ----------
    transfer : TransferLayer
        Transfer layer used to evaluate individual legs in a chaser plan.
    use_cache : bool
        Whether to memoize the plans of the sequences evaluated. Cached plans
        are shared, so their arrays are made read-only.
    use_cache_lb : bool
        Whether to memoize `evaluate_lb` per sequence. The default bound reads
        the transfer layer's own table, so this pays only for costlier bounds.
    """

    # Transfer layer used to evaluate individual legs in a chaser plan.
    transfer: TransferLayer
    # Problem instance tied to the layer (set during `initialize`).
    problem: Problem | None = None

    # Parameters controlling caching.
    use_cache: bool = True
    use_cache_lb: bool = False

    def __init__(
        self,
        transfer: TransferLayer | None = None,
        *,
        use_cache: bool = True,
        use_cache_lb: bool = False,
    ) -> None:
        self.transfer = transfer if transfer is not None else QlawTransfer()
        self.use_cache = bool(use_cache)
        self.use_cache_lb = bool(use_cache_lb)

        if not isinstance(self.transfer, TransferLayer):
            raise TypeError("'transfer' must be a TransferLayer instance")

        # Everything below is set up by `initialize`.
        self._cache: dict[tuple[int, ...], ChaserPlan] = {}
        self._cache_lb: dict[tuple[int, ...], float] = {}
        self._compiled: ScheduleKernel | None = None
        self._kernel_lb: ScheduleKernel | None = None
        self._evaluate_entry = None
        self._evaluate_lb_entry = None

    def initialize(self, problem: Problem) -> None:
        """Bind the layer to a problem instance, and clear outdated data."""
        self.transfer.initialize(problem)
        self.problem = problem

        self._cache = {}
        self._cache_lb = {}

        kernel = self._kernel()
        compiled = kernel is not None
        has_evaluate_lb = _overrides(self, ScheduleLayer, "_evaluate_lb")
        if not compiled:
            kernel = ScheduleKernel(
                register(self),
                python_schedule_evaluate,
                python_schedule_evaluate_lb if has_evaluate_lb else None,
            )
        elif has_evaluate_lb:
            raise TypeError(
                f"{type(self).__name__} publishes a kernel, so "
                "'evaluate_lb' is defined by the kernel, not in Python"
            )

        # The schedule keeps the transfer kernel it was assembled with, so a
        # transfer layer re-initialized elsewhere cannot change it underneath.
        evaluate_lb = kernel.evaluate_lb
        if evaluate_lb is None:
            evaluate_lb = schedule_evaluate_lb_default
        self._compiled = kernel._replace(
            evaluate_lb=evaluate_lb,
            transfer=self.transfer.kernel(),
        )

        # The default bound reads nothing but the transfer kernel (no context).
        self._kernel_lb = self._compiled
        if self._compiled.evaluate_lb is schedule_evaluate_lb_default:
            self._kernel_lb = self._compiled._replace(context=None)

        # The compiled functions are bound only where they do the work.
        seq = np.zeros(2, dtype=np.int64)
        if compiled:
            self._evaluate_entry = bind(schedule_evaluate, self._compiled, seq)
        else:
            self._evaluate_entry = None
        if has_evaluate_lb:
            self._evaluate_lb_entry = None
        else:
            self._evaluate_lb_entry = bind(
                schedule_evaluate_lb, self._kernel_lb, seq
            )

    def evaluate(self, sequence: Sequence[int]) -> ChaserPlan:
        """Schedule a complete chaser plan along a sequence of targets."""
        if len(sequence) == 0:
            raise ValueError("empty sequence received in the schedule layer")

        # Try to return a cached plan if it exists.
        key = tuple(sequence)
        if self.use_cache:
            hit = self._cache.get(key)
            if hit is not None:
                return hit

        # Dispatch between pure-Python and compiled kernel implementations.
        if self._evaluate_entry is None:
            plan = self._evaluate(key)
        else:
            seq = np.array(key, dtype=np.int64)
            cost, span, depart, arrive, costs = self._evaluate_entry(
                self._compiled, seq
            )
            plan = ChaserPlan(
                key,
                depart,
                arrive,
                costs,
                cost,
                span,
                max(len(key) - 1 - self.problem.max_load, 0),
                max(span - self.problem.max_time, 0.0),
            )

        # Cache new plan if requested.
        if self.use_cache:
            plan.depart.flags.writeable = False
            plan.arrive.flags.writeable = False
            plan.costs.flags.writeable = False
            self._cache[key] = plan

        return plan

    def evaluate_lb(self, sequence: Sequence[int]) -> float:
        """A lower bound on the cost of any plan along some sequence."""

        # Try to return a cached lower bound if it exists.
        key = tuple(sequence)
        if self.use_cache_lb:
            hit = self._cache_lb.get(key)
            if hit is not None:
                return hit

        # Dispatch between pure-Python and compiled kernel implementations.
        if self._evaluate_lb_entry is None:
            value = float(self._evaluate_lb(key))
        else:
            value = self._evaluate_lb_entry(
                self._kernel_lb, np.array(key, dtype=np.int64)
            )

        # Cache new lower bound if requested.
        if self.use_cache_lb:
            self._cache_lb[key] = value

        return value

    def kernel(self) -> ScheduleKernel:
        """The compiled form of the layer, assembled by `initialize`."""
        return self._compiled

    # --------------- Hooks that derived layers can implement --------------- #

    def _kernel(self) -> ScheduleKernel | None:
        """A compiled layer returns `ScheduleKernel(evaluate, context, ...)`.

        When some `ScheduleKernel` is returned, the layer is compiled and the
        hooks `evaluate` and `evaluate_lb` are ignored. When `None` is
        returned, the layer is pure Python and the hooks are used instead.
        """
        return None

    def _evaluate(self, sequence: Tuple[int, ...]) -> ChaserPlan:
        """Schedule a chaser plan along a sequence."""
        if self._compiled is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not implement '_evaluate'"
            )
        else:
            raise RuntimeError(
                f"{type(self).__name__} publishes a kernel, so the pure "
                "Python '_evaluate' should never be called in Python."
            )

    def _evaluate_lb(self, sequence: Tuple[int, ...]) -> float:
        """Lower bound on a chaser plan cost along a sequence.

        If not is overridden, defaults to `schedule_evaluate_lb_default`.
        """
        raise NotImplementedError


# Imports default layer here to avoid circular dependencies.
from .schedule.dp import DpSchedule  # noqa: E402


class SequenceLayer(ABC):
    """Sequences a mission plan into individual chaser sequences to visit.

    A derived layer is written in one of two ways:

    - In Python by implementing `_evaluate`.
    - In compiled mode, by returning a `SequenceKernel` from `_kernel`.

    Either way callers see the same `evaluate` and `kernel`. No layer sits
    above a sequence layer, so a layer written in Python gets no kernel.

    Parameters
    ----------
    schedule : ScheduleLayer
        Schedule layer used to evaluate chaser plans within the mission.
    """

    # Schedule layer used to evaluate chaser plans within the mission.
    schedule: ScheduleLayer
    # Problem instance tied to the layer (set during `initialize`).
    problem: Problem | None = None

    def __init__(self, schedule: ScheduleLayer | None = None) -> None:
        self.schedule = schedule if schedule is not None else DpSchedule()

        if not isinstance(self.schedule, ScheduleLayer):
            raise TypeError("'schedule' must be a ScheduleLayer instance")

        # Everything below is set up by `initialize`.
        self._compiled: SequenceKernel | None = None
        self._evaluate_entry: Any = None

    def initialize(self, problem: Problem) -> None:
        """Bind the layer to a problem instance, and clear outdated data."""
        self.schedule.initialize(problem)
        self.problem = problem

        kernel = self._kernel()
        if kernel is None:
            self._compiled = None
            self._evaluate_entry = None
        else:
            self._compiled = kernel._replace(schedule=self.schedule.kernel())
            self._evaluate_entry = bind(sequence_evaluate, self._compiled)

    def evaluate(self) -> MissionPlan:
        """Sequence a mission plan into individual chaser sequences."""

        # Dispatch between pure-Python and compiled kernel implementations.
        if self._evaluate_entry is None:
            return self._evaluate()

        offsets, states, depart, arrive, costs, cost, span = (
            self._evaluate_entry(self._compiled)
        )
        plans = []
        for c in range(len(offsets) - 1):
            lo, hi = offsets[c], offsets[c + 1]
            plan = ChaserPlan(
                tuple(states[lo:hi].tolist()),
                depart[lo:hi],
                arrive[lo:hi],
                costs[lo:hi],
                float(cost[c]),
                float(span[c]),
            )
            plans.append(self.problem.refresh(plan))
        return MissionPlan(plans)

    def kernel(self) -> SequenceKernel | None:
        """The compiled form of the layer, or `None` for a Python layer."""
        return self._compiled

    # --------------- Hooks that derived layers can implement --------------- #

    def _kernel(self) -> SequenceKernel | None:
        """Compiled layer kernel, or `None` for pure Python layers.

        When some `SequenceKernel` is returned, the layer is compiled and the
        hook `_evaluate` is ignored. When `None` is returned, the layer is pure
        Python and the hook is used instead.
        """
        return None

    def _evaluate(self) -> MissionPlan:
        """Sequence a mission plan into individual chaser sequences."""
        if self._compiled is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not implement '_evaluate'"
            )
        else:
            raise RuntimeError(
                f"{type(self).__name__} publishes a kernel, so the pure "
                "Python '_evaluate' should never be called in Python."
            )


# Imports default layer here to avoid circular dependencies.
from .sequence.hgs import HgsSequence  # noqa: E402

# ========================================================================= #
# Planner
# ========================================================================= #


class Planner:
    """Multi-chaser rendezvous mission planner with nested layers."""

    def __init__(self, sequence: SequenceLayer | None = None) -> None:
        self.sequence = sequence if sequence is not None else HgsSequence()

        if not isinstance(self.sequence, SequenceLayer):
            raise TypeError("expected 'sequence' of type a SequenceLayer")

    def solve(self, problem: Problem, verbose: bool = True) -> MissionPlan:
        """Plan a mission for a problem instance."""

        if verbose:
            print("Initializing layers...")

        start = time.perf_counter()

        self.sequence.initialize(problem)

        if verbose:
            elapsed = time.perf_counter() - start
            print(f"Layers initialized in {elapsed:.2f} seconds")
            print("Planning mission...")

        mission = self.sequence.evaluate()

        if verbose:
            elapsed = time.perf_counter() - start
            print(f"Mission planned in {elapsed:.2f} seconds")

        return mission

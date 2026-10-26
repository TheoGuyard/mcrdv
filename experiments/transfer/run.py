#!/usr/bin/env python3
"""How what a transfer is said to cost reshapes the plan that comes back.

The solver is left as it is and the transfer layer under it is replaced by one
of the Q-law variants below, which reuse mcrdv's estimate
(`_qlaw_transfer_evaluate`) and only change what a transfer costs:

    qlaw_arrival    delta-v + time at which each target is reached
    qlaw_urgency    delta-v + time at which the *hazardous* targets are reached
    qlaw_blackout   delta-v, departures forbidden inside recurring windows
    qlaw_embargo    delta-v, no departure before a given date

A forbidden departure is priced at an infinite cost, never an infinite
duration, so that waiting stays possible. Every plan is then scored alike:
legs re-priced in delta-v, hazard and blackout measured from `metrics`.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcrdv import MissionPlan, Problem  # noqa: E402
from mcrdv.kernel import KERNEL, TransferKernel, _sample_time  # noqa: E402
from mcrdv.orbit import DAY, RE, pack  # noqa: E402
from mcrdv.planner import TransferLayer  # noqa: E402
from mcrdv.transfer import QlawTransfer  # noqa: E402
from mcrdv.transfer.qlaw import (  # noqa: E402
    QlawContext,
    _qlaw_transfer_evaluate,
)

from _common import runner, planners  # noqa: E402

# ============================================================================ #
# Transfer layers
# ============================================================================ #


class _Shipped(NamedTuple):
    """Wraps a `QlawContext` the way mcrdv's Q-law kernel expects its layer."""

    context: QlawContext


def _shipped(layer: QlawTransfer) -> _Shipped:
    """mcrdv's Q-law, pricing the delta-v of a transfer alone."""
    return _Shipped(
        QlawContext(
            pack(layer.problem.states), layer.thrust, layer.factor, 0.0, 1.0
        )
    )


# ============================================================================ #
# qlaw_arrival: the arrival time enters the cost
# ============================================================================ #


class ArrivalContext(NamedTuple):
    qlaw: _Shipped
    factor_time: float  # cost per second of arrival time
    factor_fuel: float  # cost per m/s of delta-v


@njit(**KERNEL)
def _arrival_evaluate(kernel, i, j, t):
    context = kernel.context
    dt, dv = _qlaw_transfer_evaluate(context.qlaw, i, j, t)
    return dt, context.factor_fuel * dv + context.factor_time * (t + dt)


class ArrivalQlawTransfer(QlawTransfer):
    """Delta-v plus the arrival time `t + dt` of each transfer.

    The Q-law flies at full thrust, so pricing the transfer *duration* only
    rescales delta-v. Pricing the *arrival* time makes waiting cost something,
    and `factor_time / factor_fuel` sets the fuel/time trade-off.
    """

    def _kernel(self) -> TransferKernel:
        context = ArrivalContext(
            _shipped(self), self.factor_time, self.factor_fuel
        )
        return TransferKernel(context, _arrival_evaluate)


# ============================================================================ #
# qlaw_urgency: the arrival time of hazardous targets enters the cost
# ============================================================================ #


class UrgencyContext(NamedTuple):
    qlaw: _Shipped
    weight: np.ndarray  # cost per second of arrival time, per target
    factor_fuel: float


@njit(**KERNEL)
def _urgency_evaluate(kernel, i, j, t):
    context = kernel.context
    dt, dv = _qlaw_transfer_evaluate(context.qlaw, i, j, t)
    return (
        dt,
        context.factor_fuel * dv + context.weight[j] * (t + dt),
    )


def hazardous(problem: Problem, share: float) -> np.ndarray:
    """Mask of the `share` of targets at the highest altitude, which decay
    slowest: the ones worth removing first."""
    altitude = np.array([state.a for state in problem.states]) - RE
    mask = np.zeros(problem.num_states, dtype=bool)
    targets = np.asarray(problem.targets, dtype=np.int64)
    count = max(int(round(share * targets.size)), 1)
    ranked = targets[np.argsort(-altitude[targets], kind="stable")]
    mask[ranked[:count]] = True
    return mask


class UrgencyQlawTransfer(QlawTransfer):
    """Delta-v plus `weight` (m/s per day) times the arrival time at each
    hazardous target, a `share` of the catalogue (see `hazardous`)."""

    def __init__(
        self, *, weight: float = 5.0, share: float = 0.33, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.weight = float(weight)
        self.share = float(share)
        if not 0.0 < self.share <= 1.0:
            raise ValueError("share must lie in (0, 1]")

    def _kernel(self) -> TransferKernel:
        weights = np.where(
            hazardous(self.problem, self.share), self.weight / DAY, 0.0
        )
        context = UrgencyContext(_shipped(self), weights, self.factor_fuel)
        return TransferKernel(context, _urgency_evaluate)


# ============================================================================ #
# qlaw_blackout and qlaw_embargo: forbidden departures
# ============================================================================ #


class BlackoutContext(NamedTuple):
    qlaw: _Shipped
    period: float  # a window opens every `period` seconds...
    duration: float  # ...and stays open `duration` seconds
    factor_fuel: float


@njit(**KERNEL)
def _blackout_evaluate(kernel, i, j, t):
    context = kernel.context
    dt, dv = _qlaw_transfer_evaluate(context.qlaw, i, j, t)
    if (t % context.period) < context.duration:
        return dt, np.inf
    return dt, context.factor_fuel * dv


@njit(**KERNEL)
def _fuel_only_lb(kernel, i, j):
    """Lower bound of a leg with the prohibitions lifted: the sampled default
    would mostly sample infinite costs."""
    context = kernel.context
    best_dt = np.inf
    best_dc = np.inf
    for k in range(kernel.num_samples):
        t = _sample_time(k, kernel.num_samples, kernel.max_time)
        dt, dv = _qlaw_transfer_evaluate(context.qlaw, i, j, t)
        if dt < best_dt:
            best_dt = dt
        if context.factor_fuel * dv < best_dc:
            best_dc = context.factor_fuel * dv
    return best_dt, best_dc


@njit(**KERNEL)
def _blackout_hints(kernel, i, j):
    """The moment each window closes, as departure hints."""
    context = kernel.context
    count = int(kernel.max_time // context.period) + 1
    out = np.empty(count)
    kept = 0
    for k in range(count):
        opens = k * context.period + context.duration
        if opens <= kernel.max_time:
            out[kept] = opens
            kept += 1
    return out[:kept]


class BlackoutQlawTransfer(QlawTransfer):
    """Delta-v, with no departure inside windows opening every `period` and
    lasting `duration`."""

    def __init__(
        self, *, period: Any = "10d", duration: Any = "3d", **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.period = float(runner.duration(period))
        self.duration = float(runner.duration(duration))
        if self.period <= 0.0:
            raise ValueError("period must be positive")
        if not 0.0 <= self.duration < self.period:
            raise ValueError("duration must lie in [0, period)")

    def _kernel(self) -> TransferKernel:
        context = BlackoutContext(
            _shipped(self), self.period, self.duration, self.factor_fuel
        )
        return TransferKernel(
            context,
            _blackout_evaluate,
            evaluate_lb=_fuel_only_lb,
            hints=_blackout_hints,
        )


class EmbargoContext(NamedTuple):
    qlaw: _Shipped
    start: float  # no transfer may begin before this instant
    factor_fuel: float


@njit(**KERNEL)
def _embargo_evaluate(kernel, i, j, t):
    context = kernel.context
    dt, dv = _qlaw_transfer_evaluate(context.qlaw, i, j, t)
    if t < context.start:
        return dt, np.inf
    return dt, context.factor_fuel * dv


@njit(**KERNEL)
def _embargo_hints(kernel, i, j):
    """The instant the embargo lifts, as a departure hint."""
    out = np.empty(1)
    kept = 0
    if kernel.context.start <= kernel.max_time:
        out[0] = kernel.context.start
        kept = 1
    return out[:kept]


class EmbargoQlawTransfer(QlawTransfer):
    """Delta-v, with no departure at all before `start`."""

    def __init__(self, *, start: Any = "50d", **kwargs) -> None:
        super().__init__(**kwargs)
        self.start = float(runner.duration(start))
        if self.start < 0.0:
            raise ValueError("start must not be negative")

    def _kernel(self) -> TransferKernel:
        context = EmbargoContext(_shipped(self), self.start, self.factor_fuel)
        return TransferKernel(
            context,
            _embargo_evaluate,
            evaluate_lb=_fuel_only_lb,
            hints=_embargo_hints,
        )


planners.TRANSFERS.update(
    {
        "qlaw_arrival": ArrivalQlawTransfer,
        "qlaw_urgency": UrgencyQlawTransfer,
        "qlaw_blackout": BlackoutQlawTransfer,
        "qlaw_embargo": EmbargoQlawTransfer,
    }
)


# ============================================================================ #
# Scoring every plan on one scale, whatever its oracle
# ============================================================================ #


def delta_v_layer(config: Mapping[str, Any]) -> TransferLayer:
    """The configured Q-law pricing delta-v alone: only its physics (`thrust`,
    `factor`) is kept, so plans built under different cost oracles can be read
    on one scale."""
    _, args = planners.layer(config, "transfer")
    physics = {k: v for k, v in args.items() if k in ("thrust", "factor")}
    return QlawTransfer(**physics)


def leg_table(
    mission: MissionPlan, delta_v: TransferLayer
) -> List[Dict[str, Any]]:
    """Every leg of a mission, one row each, re-priced by `delta_v` (an
    initialized layer pricing delta-v, see `delta_v_layer`) at the
    departure the plan chose, so plans built under different oracles share
    one scale."""
    rows = []
    for index, plan in enumerate(mission.plans):
        if plan.load == 0:
            continue
        for k in range(len(plan) - 1):
            ready = float(plan.arrive[k])
            leaves = float(plan.depart[k])
            dt, dv = delta_v.evaluate(
                plan.sequence[k], plan.sequence[k + 1], leaves
            )
            rows.append(
                {
                    "chaser": index,
                    # Position in the tour of the target the leg reaches.
                    "position": k + 1,
                    "origin": int(plan.sequence[k]),
                    "target": int(plan.sequence[k + 1]),
                    "wait": max(leaves - ready, 0.0),
                    "depart": leaves,
                    "arrive": float(plan.arrive[k + 1]),
                    "flight": float(dt),
                    "delta_v": float(dv),
                }
            )
    return rows


def fuel_and_time(
    mission: MissionPlan, delta_v: TransferLayer
) -> Dict[str, Any]:
    """Totals of :func:`leg_table`: delta-v, flight and waiting time, and
    when the targets are collected."""
    rows = leg_table(mission, delta_v)
    times = np.array([row["arrive"] for row in rows], dtype=np.float64)
    return {
        "delta_v": float(sum(row["delta_v"] for row in rows)),
        "flight_time": float(sum(row["flight"] for row in rows)),
        "wait_time": float(sum(row["wait"] for row in rows)),
        "collection_mean": float(times.mean()) if times.size else 0.0,
        "collection_max": float(times.max()) if times.size else 0.0,
    }


def blacked_out(when: float, period: float, duration: float) -> bool:
    """Whether a departure at `when` falls inside a window."""
    return (when % period) < duration


def score(
    mission: MissionPlan, problem: Problem, metrics: Mapping[str, Any]
) -> Dict[str, Any]:
    """When the hazardous targets are collected and how many departures fall
    in a window, with both defined by `metrics` rather than by the oracle that
    built the plan, so that every plan is scored alike."""
    share = float(metrics.get("hazard_share", 0.33))
    period = float(metrics.get("blackout_period", 10.0 * DAY))
    duration = float(metrics.get("blackout_duration", 3.0 * DAY))
    mask = hazardous(problem, share)

    hazard_times: list = []
    plain_times: list = []
    departures = 0
    inside = 0
    for plan in mission.plans:
        if plan.load == 0:
            continue
        for position in range(1, len(plan)):
            when = float(plan.arrive[position])
            if mask[plan.sequence[position]]:
                hazard_times.append(when)
            else:
                plain_times.append(when)
        for position in range(len(plan) - 1):
            departures += 1
            if blacked_out(float(plan.depart[position]), period, duration):
                inside += 1

    times = np.asarray(hazard_times, dtype=np.float64)
    rest = np.asarray(plain_times, dtype=np.float64)
    return {
        "num_hazardous": int(times.size),
        "hazard_collection_mean": float(times.mean()) if times.size else 0.0,
        "hazard_collection_max": float(times.max()) if times.size else 0.0,
        "plain_collection_mean": float(rest.mean()) if rest.size else 0.0,
        "departures": departures,
        "blackout_departures": inside,
        "blackout_share": inside / departures if departures else 0.0,
    }


# ============================================================================ #
# The experiment
# ============================================================================ #


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
            node[key] = runner.duration(node[key])
    return node


def execute(config: dict) -> dict:
    result = planners.solve(config)
    problem = planners.problem(config)

    delta_v = delta_v_layer(config)
    delta_v.initialize(problem)

    metrics = yardstick(config)
    legs = leg_table(result["mission"], delta_v)
    mask = hazardous(problem, float(metrics.get("hazard_share", 0.33)))
    for leg in legs:
        leg["hazardous"] = bool(mask[leg["target"]])

    result["legs"] = legs
    result["chasers"] = planners.per_chaser(result["mission"], problem)
    result.update(planners.mission_metrics(result["mission"], problem))
    result.update(fuel_and_time(result["mission"], delta_v))
    result.update(score(result["mission"], problem, metrics))

    result["oracle"] = oracle(config)
    result["weight"] = weight(config)
    return result


if __name__ == "__main__":
    raise SystemExit(runner.main(__file__, execute, warmup=planners.warm_up))

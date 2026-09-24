"""Transfer layers written for the experiments rather than shipped with mcrdv.

The claim these exist to support is that the cost oracle is a plug-in: what a
mission is trying to achieve, and what it is forbidden to do, are written here
and nowhere else. The schedule layer, the search, and every line of the solver
stay as they are.

Each layer below is the shipped Q-law with a different notion of what a
transfer costs:

    qlaw            delta-v alone, and the reference every plan is read against
    qlaw_arrival    delta-v, and the time at which each target is collected
    qlaw_urgency    delta-v, and the time at which the *hazardous* ones are
    qlaw_blackout   delta-v, with departures forbidden inside given windows
    qlaw_embargo    delta-v, with no departure at all before a given date

They all reach the shipped estimate through `_qlaw_transfer_evaluate`, so the
physics is the package's own rather than a copy of it, and what changes is a
line of arithmetic on top.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, NamedTuple, Sequence

import numpy as np
from numba import njit

from mcrdv import MissionPlan, Problem
from mcrdv.kernel import KERNEL, TransferKernel

# `_sample_time` and `_qlaw_transfer_evaluate` are private to mcrdv, and used
# here on purpose: a layer that samples departures like the shipped bound, and
# prices them with the shipped estimate, should do both exactly as it does.
from mcrdv.kernel import _sample_time
from mcrdv.orbit import DAY, RE, pack
from mcrdv.transfer import QlawTransfer
from mcrdv.transfer.qlaw import QlawContext, _qlaw_transfer_evaluate


class _Shipped(NamedTuple):
    """What the shipped Q-law kernel reads of its layer: `.context`, nothing
    else. Wrapping its context this way lets it be called as it is."""

    context: QlawContext


def _shipped(layer: QlawTransfer) -> _Shipped:
    """The shipped Q-law set up to price the delta-v of a transfer alone."""
    return _Shipped(
        QlawContext(
            pack(layer.problem.states), layer.thrust, layer.factor, 0.0, 1.0
        )
    )


# ============================================================================ #
# Collecting sooner: the arrival time enters the cost
# ============================================================================ #


class ArrivalContext(NamedTuple):
    # The shipped Q-law, pricing delta-v alone.
    qlaw: _Shipped
    # Cost per second of the time at which a target is reached.
    factor_time: float
    # Cost per m/s of delta-v.
    factor_fuel: float


@njit(**KERNEL)
def _arrival_evaluate(kernel, i, j, t):
    context = kernel.context
    dt, dv = _qlaw_transfer_evaluate(context.qlaw, i, j, t)
    return dt, context.factor_fuel * dv + context.factor_time * (t + dt)


class ArrivalQlawTransfer(QlawTransfer):
    """Q-law transfers, costed on delta-v and on *when* each target is reached.

    The shipped layer's time term cannot trade anything against fuel: the
    Q-law estimate flies at full thrust, so a transfer's time of flight is its
    delta-v over the thrust, and `factor_time * dt + factor_fuel * dv` is a
    fixed multiple of the delta-v whatever the two factors are. Weighing them
    differently rescales the objective and changes no plan.

    What fuel is actually traded against is waiting. A chaser saves delta-v by
    loitering until the next plane aligns, and pays for it in mission time. So
    the time term here prices the arrival time `t + dt` of each transfer rather
    than its duration. Summed along a plan it is the total time at which the
    targets are collected, so the objective reads

        factor_fuel * total delta-v  +  factor_time * sum of collection times.

    Waiting now costs something, and the ratio of the two factors moves the
    plan along a genuine fuel/time trade-off.
    """

    def _kernel(self) -> TransferKernel:
        context = ArrivalContext(
            _shipped(self), self.factor_time, self.factor_fuel
        )
        return TransferKernel(context, _arrival_evaluate)


# ============================================================================ #
# Collecting the dangerous ones sooner: the arrival time, weighted per target
# ============================================================================ #


class UrgencyContext(NamedTuple):
    qlaw: _Shipped
    # Cost per second of the arrival time, per target.
    weight: np.ndarray
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
    """The `share` of targets whose orbits decay slowest, as a mask.

    A proxy for how much removing an object is worth: the higher it is, the
    longer it stays up, the longer it threatens everything crossing that
    altitude. The mask is what the oracle prices and what the figures count,
    so the two always agree on which objects these are.
    """
    altitude = np.array([state.a for state in problem.states]) - RE
    mask = np.zeros(problem.num_states, dtype=bool)
    targets = np.asarray(problem.targets, dtype=np.int64)
    count = max(int(round(share * targets.size)), 1)
    ranked = targets[np.argsort(-altitude[targets], kind="stable")]
    mask[ranked[:count]] = True
    return mask


class UrgencyQlawTransfer(QlawTransfer):
    """Q-law transfers, costed on delta-v and on when the hazardous ones go.

    The paper's removal-urgency oracle: `sigma_j * (t + tau)`, with a weight
    per target rather than one shared by all. Only the objects the mask picks
    out carry a weight, so the search is asked to collect those early and left
    free to collect the rest whenever it is cheapest.

    Parameters
    ----------
    weight : float
        Cost of a hazardous target's collection time, in m/s of delta-v per
        day. Zero reduces this to the shipped objective.
    share : float
        Fraction of the catalogue treated as hazardous.
    """

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
# Keeping out of the way: departures forbidden inside windows
# ============================================================================ #


class BlackoutContext(NamedTuple):
    qlaw: _Shipped
    # A window opens every `period` seconds and lasts `duration`.
    period: float
    duration: float
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
    """The cheapest this leg could be with the prohibitions lifted.

    For an oracle that only ever *raises* the fuel-only cost -- by forbidding a
    departure rather than by repricing one -- the transfer priced without the
    prohibition bounds the transfer priced with it. It is also a bound the
    sampled default cannot give, since most of what that samples is infinite
    and an infinite lower bound prunes the sequence that would have waited.

    Read by every layer whose context carries `qlaw` and `factor_fuel`, which
    is what makes it one function rather than one per prohibition.
    """
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
    """The moment each window closes.

    The schedule layer merges these into its grid, so a chaser held back by a
    window can leave the instant it lifts rather than at whatever uniform grid
    point happens to follow.
    """
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
    """Q-law transfers, with departures forbidden inside recurring windows.

    A ground-segment outage, a conjunction alert, a keep-out imposed by an
    operator: the mission may not *start* a transfer while a window is open,
    and the oracle says so by pricing those departures at infinity. The
    schedule layer then simply never chooses one.

    The cost goes to infinity, never the time of flight. The dynamic program
    decides that a sequence is infeasible when its no-wait schedule overruns
    the horizon, which assumes that leaving later never arrives earlier; an
    infinite duration breaks that, and whole sequences would be thrown away
    although waiting for the window to lift would have flown them.

    Parameters
    ----------
    period : float or str
        How often a window opens.
    duration : float or str
        How long one lasts. The share of time under blackout is their ratio.
    """

    def __init__(
        self, *, period: Any = "10d", duration: Any = "3d", **kwargs
    ) -> None:
        super().__init__(**kwargs)
        # Imported here: `config` knows nothing of mcrdv, and this is the only
        # layer that takes a duration.
        from . import config as configuration

        self.period = float(configuration.duration(period))
        self.duration = float(configuration.duration(duration))
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


# ============================================================================ #
# Standing off: nothing departs before a date
# ============================================================================ #


class EmbargoContext(NamedTuple):
    qlaw: _Shipped
    # No transfer may begin before this instant.
    start: float
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
    """The instant the embargo lifts.

    One number, and the one departure time the schedule layer would otherwise
    have to be lucky to hold: without it the first transfer leaves at whichever
    uniform grid point follows the date, and every chaser in the swarm loses
    the same few days for no reason.
    """
    out = np.empty(1)
    kept = 0
    if kernel.context.start <= kernel.max_time:
        out[0] = kernel.context.start
        kept = 1
    return out[:kept]


class EmbargoQlawTransfer(QlawTransfer):
    """Q-law transfers, with nothing at all departing before a given date.

    A swarm launched before it is cleared to work: the spacecraft are on orbit
    from the first day, drifting with everything else, and may not begin a
    transfer until commissioning is over, a licence is granted, or a launch
    window elsewhere has passed. The layer says so by pricing every earlier
    departure at infinity.

    It is a blackout with a single window, open from the start of the mission,
    and it is a separate layer because it says a different thing: a recurring
    outage is a rhythm the plan works around, while this is a date the whole
    mission begins at. The visible consequence is the same either way -- the
    plan is simply late -- which is what makes it worth drawing.

    The cost goes to infinity, never the time of flight, for the reason given
    in :class:`BlackoutQlawTransfer`.

    Parameters
    ----------
    start : float or str
        The instant the embargo lifts. Nothing departs before it.
    """

    def __init__(self, *, start: Any = "50d", **kwargs) -> None:
        super().__init__(**kwargs)
        from . import config as configuration

        self.start = float(configuration.duration(start))
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


def blacked_out(when: float, period: float, duration: float) -> bool:
    """Whether a departure at `when` falls inside a window."""
    return (when % period) < duration


# ============================================================================ #
# The yardstick
# ============================================================================ #


def score(
    mission: MissionPlan, problem: Problem, metrics: Mapping[str, Any]
) -> Dict[str, Any]:
    """Every plan measured the same way, whatever it was built to optimize.

    One yardstick for all of them: when the hazardous objects are collected,
    and how many departures fall inside a window. Which objects are hazardous
    and where the windows are come from `metrics`, not from the oracle that
    produced the plan, so a plan that never heard of either is still scored on
    both -- which is the point of putting them side by side.
    """
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
        # When the objects worth removing first actually go [s], and when the
        # others do: an oracle that hurries only the first leaves the second
        # where it found them, which is the whole point of weighing them apart.
        "hazard_collection_mean": float(times.mean()) if times.size else 0.0,
        "hazard_collection_max": float(times.max()) if times.size else 0.0,
        "plain_collection_mean": float(rest.mean()) if rest.size else 0.0,
        "departures": departures,
        "blackout_departures": inside,
        "blackout_share": inside / departures if departures else 0.0,
    }


def windows(horizon: float, period: float, duration: float) -> Sequence:
    """Every window over a mission, as `(start, end)` pairs, for drawing."""
    count = int(math.ceil(horizon / period))
    return [
        (k * period, min(k * period + duration, horizon))
        for k in range(count + 1)
        if k * period < horizon
    ]

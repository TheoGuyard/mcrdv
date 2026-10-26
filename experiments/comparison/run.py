#!/usr/bin/env python3
"""mcrdv beside planners of the literature, on their own instances and terms.

Layers written for this comparison, registered under the names the
configurations use:

    qlaw_frozen         transfer: mcrdv's Q-law on orbits frozen at the epoch
    qlaw_lopez_rivera   transfer: the time of flight and delta-v model of
                        López Rivera & Naeije, on their spacecraft
    shen_casalino       transfer: impulsive two-epoch estimate (Zona et al.)
    epoch               schedule: both dates of every leg on a grid of epochs
    fixed               sequence: published tours, flown as given
    raan_walk           sequence: best sweep of the right ascensions at the
                        epoch
    drw                 sequence: dynamic RAAN walk (López Rivera & Naeije)

and the instance `iridium33_rivera`: debris drawn as López Rivera & Naeije draw
theirs, from a model of the Iridium-33 cloud (built by `cloud.py`).

All literature planners but `fixed` route a single chaser.
"""

import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Sequence, Tuple

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcrdv import MissionPlan, Problem, load_states  # noqa: E402
from mcrdv.kernel import (  # noqa: E402
    JIT,
    KERNEL,
    ScheduleKernel,
    TransferKernel,
)
from mcrdv.orbit import (  # noqa: E402
    DO,
    DR,
    MU,
    RE,
    TAU,
    KepState,
    mee_at,
    mee_max_rates,
    pack,
)
from mcrdv.planner import (  # noqa: E402
    ScheduleLayer,
    SequenceLayer,
    TransferLayer,
)
from mcrdv.problem import DEPOT, ChaserPlan  # noqa: E402
from mcrdv.transfer import QlawTransfer  # noqa: E402
from mcrdv.transfer.qlaw import (  # noqa: E402
    QlawContext,
    _qlaw_transfer_evaluate,
)

from _common import runner, planners  # noqa: E402

# Standard gravity [m/s^2], for an exhaust velocity from a specific impulse.
G0 = 9.80665

# ============================================================================ #
# qlaw_frozen: orbits that do not drift (Narayanaswamy et al.)
# ============================================================================ #


class FrozenQlawTransfer(QlawTransfer):
    """mcrdv's Q-law on orbits frozen at `t = 0` (J2 drift rates set to zero):
    the dynamics of planners that ignore J2, such as Narayanaswamy et al."""

    def _kernel(self) -> TransferKernel:
        table = pack(self.problem.states)
        table[:, DR] = 0.0
        table[:, DO] = 0.0
        context = QlawContext(
            table,
            self.thrust,
            self.factor,
            self.factor_time,
            self.factor_fuel,
        )
        return TransferKernel(context, _qlaw_transfer_evaluate)


@njit(**JIT)
def relative_inclination(i1, r1, i2, r2):
    """Angle between two orbital planes [rad] (Narayanaswamy et al., Eq. 1)."""
    c = math.cos(i1) * math.cos(i2) + math.sin(i1) * math.sin(i2) * math.cos(
        r1 - r2
    )
    return math.acos(min(1.0, max(-1.0, c)))


def inclination(problem: Problem, tour: Sequence[int]) -> float:
    """Relative inclination [rad] summed along a tour from the depot, at the
    epoch: the static cost Narayanaswamy et al. sequence on (their Table 2)."""
    s = problem.states
    stops = [DEPOT] + [int(target) for target in tour]
    return float(
        sum(
            relative_inclination(s[a].i, s[a].r, s[b].i, s[b].r)
            for a, b in zip(stops, stops[1:])
        )
    )


# ============================================================================ #
# qlaw_lopez_rivera: their transfer model and spacecraft (López Rivera et al.)
# ============================================================================ #


class LopezRiveraContext(NamedTuple):
    table: np.ndarray  # packed states
    accel: float  # thrust acceleration at the wet mass [m/s^2]
    slope: float  # of the time of flight on the time-to-go
    intercept: float  # of the same regression [s]
    sm: float  # semi-major axis scaling of Q: m, n and r (Eq. 9)
    sn: float
    sr: float
    k: float  # periapsis penalty (Eq. 8)
    rp_min: float  # [m]
    wp: float
    m0: float  # wet mass [kg]
    tank: float  # fuel mass one tank holds [kg]
    veq: float  # exhaust velocity [m/s]
    flow: float  # fuel mass flow at full thrust [kg/s]


@njit(**JIT)
def lopez_rivera_dv(context, tof):
    """Delta-v [m/s] of a transfer lasting `tof` [s] at full thrust, a tank
    refilled whenever it runs dry (López Rivera & Naeije, Eq. 15)."""
    need = context.flow * tof
    m0, veq = context.m0, context.veq
    if need <= context.tank:
        return veq * math.log(m0 / (m0 - need))
    full = math.floor(need / context.tank)
    rest = need - full * context.tank
    tank = veq * math.log(m0 / (m0 - context.tank))
    return full * tank + veq * math.log(m0 / (m0 - rest))


@njit(**JIT)
def lopez_rivera_tof(context, src, dst, planar):
    """Time of flight [s] from the equinoctial state `src` to `dst`: their
    regression on the best time-to-go `sqrt(Q)` of the Q-law (Eqs. 7-14).
    Without `planar`, the plane (`h`, `k`) is left free."""
    sa, sf, sg, sh, sk, sl = src
    da, df, dg, dh, dk = dst
    ra, rf, rg, rh, rk, _ = mee_max_rates(sa, sf, sg, sh, sk, sl)
    f = context.accel
    scale = (1.0 + (abs(sa - da) / (context.sm * da)) ** context.sn) ** (
        1.0 / context.sr
    )
    rp = sa * (1.0 - math.sqrt(sf * sf + sg * sg))
    penalty = math.exp(context.k * (1.0 - rp / context.rp_min))
    q = (
        scale * ((sa - da) / (ra * f)) ** 2
        + ((sf - df) / (rf * f)) ** 2
        + ((sg - dg) / (rg * f)) ** 2
    )
    if planar:
        q += ((sh - dh) / (rh * f)) ** 2 + ((sk - dk) / (rk * f)) ** 2
    togo = math.sqrt((1.0 + context.wp * penalty) * q)
    return max(context.slope * togo + context.intercept, togo)


@njit(**KERNEL)
def _lopez_rivera_evaluate(kernel, i, j, t):
    context = kernel.context
    src = mee_at(context.table, i, t)
    a, f, g, h, k, _ = mee_at(context.table, j, t)
    tof = lopez_rivera_tof(context, src, (a, f, g, h, k), True)
    return tof, lopez_rivera_dv(context, tof)


class LopezRiveraTransfer(TransferLayer):
    """The transfer model of López Rivera & Naeije (2025): a leg lasts what
    their regression on RQ-law transfers predicts from the Q-law time-to-go
    (Eq. 14), at full thrust, and costs the delta-v its fuel buys (Eq. 15).

    The time-to-go is computed on the orbits at departure, the debris drifting
    under J2 between transfers (their Eq. 12). The cost of a leg depends on
    its duration alone, so a tour costs the sum of its legs (Eq. 16).

    Parameters
    ----------
    thrust, mass, fuel, isp : float
        Their spacecraft (Table 1): thrust [N], wet mass and fuel of one tank
        [kg], specific impulse [s].
    slope, intercept : float
        Their RQ-law regression (Table 4), the intercept in hours.
    rp_min : float or str
        Periapsis of the penalty of Q, which they do not give; the radius of
        the Earth, as in the RQ-law of Narayanaswamy & Damaren, makes it
        negligible.
    decommission : float or str
        Altitude of the circular orbit a chaser ends on.
    """

    def __init__(
        self,
        *,
        thrust: float = 0.36,
        mass: float = 1200.0,
        fuel: float = 450.0,
        isp: float = 3000.0,
        slope: float = 1.4309,
        intercept: float = -9.72,
        rp_min: Any = RE,
        decommission: Any = 250e3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.thrust = float(thrust)
        self.mass = float(mass)
        self.fuel = float(fuel)
        self.veq = float(isp) * G0
        self.slope = float(slope)
        self.intercept = float(intercept) * 3600.0
        self.rp_min = float(rp_min)
        self.decommission_radius = RE + float(decommission)
        if not 0.0 < self.fuel < self.mass:
            raise ValueError("the fuel must weigh less than the spacecraft")

    def _context(self) -> LopezRiveraContext:
        return LopezRiveraContext(
            pack(self.problem.states),
            self.thrust / self.mass,
            self.slope,
            self.intercept,
            3.0,
            4.0,
            2.0,
            100.0,
            self.rp_min,
            1.0,
            self.mass,
            self.fuel,
            self.veq,
            self.thrust / self.veq,
        )

    def _kernel(self) -> TransferKernel:
        return TransferKernel(self._context(), _lopez_rivera_evaluate)

    def decommission(self, i: int, t: float) -> Tuple[float, float]:
        """Time and delta-v from the state `i` at `t` to their circular
        decommissioning orbit, every other element free (their Sec. 2.4)."""
        context = self._context()
        tof = lopez_rivera_tof(
            context,
            mee_at(context.table, i, t),
            (self.decommission_radius, 0.0, 0.0, 0.0, 0.0),
            False,
        )
        return tof, lopez_rivera_dv(context, tof)

    def tour_dv(self, tof: float) -> float:
        """Delta-v of a whole tour of `tof` seconds of thrust, its fuel spent
        across the legs: the reading of Eq. 15 on complete tours."""
        return float(lopez_rivera_dv(self._context(), tof))


# The families López Rivera & Naeije fit to each element of the cloud (their
# Fig. 9), sampled within the range observed.
MARGINALS = {
    "a": "cauchy",
    "e": "inverse_gamma",
    "i": "cauchy",
    "r": "von_mises",
    "o": "von_mises",
    "t": "von_mises",
}


def _digamma(x: float) -> float:
    shift = 0.0
    while x < 6.0:
        shift -= 1.0 / x
        x += 1.0
    y = 1.0 / (x * x)
    series = y * (1 / 12 - y * (1 / 120 - y * (1 / 252 - y / 240)))
    return shift + math.log(x) - 0.5 / x - series


def _trigamma(x: float) -> float:
    shift = 0.0
    while x < 6.0:
        shift += 1.0 / (x * x)
        x += 1.0
    y = 1.0 / (x * x)
    series = 1 / x + y / 2 + y / x * (1 / 6 - y * (1 / 30 - y / 42))
    return shift + series


def _fit(family: str, x: np.ndarray) -> Tuple[float, float]:
    """Maximum likelihood parameters of a family on the observations `x`."""
    if family == "cauchy":
        loc = float(np.median(x))
        q1, q3 = np.percentile(x, [25, 75])
        scale = 0.5 * float(q3 - q1)
        for _ in range(1000):
            w = 1.0 / (scale**2 + (x - loc) ** 2)
            loc = float(w @ x / w.sum())
            scale = math.sqrt(x.size / (2.0 * w.sum()))
        return loc, scale
    if family == "inverse_gamma":
        y = 1.0 / x  # gamma distributed: shape, rate
        s = math.log(y.mean()) - float(np.log(y).mean())
        shape = (3 - s + math.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
        for _ in range(100):
            step = math.log(shape) - _digamma(shape) - s
            shape -= step / (1 / shape - _trigamma(shape))
        return shape, shape / float(y.mean())  # shape, scale
    if family == "von_mises":
        c, s = float(np.cos(x).mean()), float(np.sin(x).mean())
        r = math.hypot(c, s)
        if r < 0.53:
            kappa = 2 * r + r**3 + 5 * r**5 / 6
        elif r < 0.85:
            kappa = -0.4 + 1.39 * r + 0.43 / (1 - r)
        else:
            kappa = 1 / (r**3 - 4 * r**2 + 3 * r)
        return math.atan2(s, c) % TAU, kappa
    raise ValueError(f"unknown family '{family}'")


def _sample(family, params, lo, hi, count, rng) -> np.ndarray:
    """`count` draws of a fitted family, within `[lo, hi]`: by inverse
    transform when its inverse is at hand, by rejection otherwise."""
    if family == "cauchy":
        loc, scale = params
        cdf = [0.5 + math.atan((v - loc) / scale) / math.pi for v in (lo, hi)]
        u = rng.uniform(cdf[0], cdf[1], count)
        return loc + scale * np.tan(math.pi * (u - 0.5))
    out = np.empty(0)
    while out.size < count:
        if family == "inverse_gamma":
            shape, scale = params
            draws = 1.0 / rng.gamma(shape, 1.0 / scale, count)
        else:
            mean, kappa = params
            draws = rng.vonmises(mean, kappa, count) % TAU
        out = np.concatenate([out, draws[(draws >= lo) & (draws <= hi)]])
    return out[:count]


@lru_cache(maxsize=None)
def cloud_model(name: str) -> Dict[str, tuple]:
    """Each element's family, fitted parameters and observed range, on the
    objects of the catalogue `name` (its first row, the depot, excepted)."""
    states = load_states(planners.DATA / f"{name}.csv")[1:]
    model = {}
    for element, family in MARGINALS.items():
        x = np.array([getattr(s, element) for s in states])
        fitted = _fit(family, x)
        model[element] = (family, fitted, float(x.min()), float(x.max()))
    return model


def sample_cloud(name: str, count: int, draw: int) -> List[KepState]:
    """`count` debris drawn as López Rivera & Naeije draw theirs: every
    element independently, from the family fitted to the cloud."""
    rng = np.random.default_rng(draw)
    columns = {
        element: _sample(family, params, lo, hi, count, rng)
        for element, (family, params, lo, hi) in cloud_model(name).items()
    }
    return [
        KepState(*(float(columns[e][k]) for e in "aeirot"))
        for k in range(count)
    ]


# ============================================================================ #
# shen_casalino: impulsive transfers on drift orbits (Zona et al.)
# ============================================================================ #


class ShenCasalinoContext(NamedTuple):
    a: np.ndarray  # semi-major axis [m], per state
    i: np.ndarray  # inclination [rad]
    r: np.ndarray  # right ascension at the epoch [rad]
    dr: np.ndarray  # its J2 drift [rad/s]
    d_min: float  # shortest transfer flown [s]
    d_max: float  # longer transfers cost what this one does [s]
    tolerance: float  # node gap at arrival counted as aligned [rad]
    sample: float  # duration step of `evaluate` [s]
    free_injection: bool  # chasers start on their first target for free


@njit(**JIT)
def shen_casalino(context, i, j, t0, tf):
    """Delta-v [m/s] of an impulsive transfer leaving `i` at `t0` and meeting
    `j` at `tf`, as stated by Zona et al. (2023, Appendix).

    If the two planes align by themselves before `tf`, only the change of
    semi-major axis and inclination is paid. Otherwise two impulses change a,
    i and the node, the first one placing the chaser on a drift orbit whose
    precession closes part of the node gap; the split between the impulses
    minimizes the sum of their squares, in closed form.
    """
    a1, a2 = context.a[i], context.a[j]
    i1, i2 = context.i[i], context.i[j]
    a0 = 0.5 * (a1 + a2)
    i0 = 0.5 * (i1 + i2)
    v0 = math.sqrt(MU / a0)
    duration = tf - t0

    # Node gap at departure and at arrival, were the chaser not to move.
    gap0 = (context.r[j] - context.r[i]) + (context.dr[j] - context.dr[i]) * t0
    gapf = gap0 + (context.dr[j] - context.dr[i]) * duration
    closing = (gapf + math.pi) % TAU - math.pi

    # The planes align during the transfer, or nearly at its end.
    crossed = math.floor(gap0 / TAU) != math.floor(gapf / TAU)
    if crossed or abs(closing) <= context.tolerance:
        return 0.5 * v0 * math.sqrt(((a2 - a1) / a0) ** 2 + (i2 - i1) ** 2)

    # Node, semi-major axis and inclination changes, as velocities.
    x = closing * v0 * math.sin(i0)
    y = 0.5 * (a2 - a1) / a0 * v0
    z = (i2 - i1) * v0
    # Node change bought by the drift of a changed a or i over the transfer.
    rate = 0.5 * (context.dr[i] + context.dr[j])
    m = -7.0 * rate * math.sin(i0) * duration
    n = -rate * math.sin(i0) * math.tan(i0) * duration
    # Optimal split between the two impulses.
    node = (2.0 * x - m * y - n * z) / (4.0 + m * m + n * n)
    sa = 0.5 * (y + m * node)
    si = 0.5 * (z + n * node)
    first = math.sqrt(node * node + sa * sa + si * si)
    second = math.sqrt(node * node + (y - sa) ** 2 + (z - si) ** 2)
    return first + second


@njit(**JIT)
def shen_casalino_leg(context, i, j, t0, tf):
    """The cost the epoch schedule reads: free from the depot (injection),
    infinite under `d_min`, and capped at the `d_max` transfer beyond it."""
    if i == DEPOT and context.free_injection:
        return 0.0 if tf >= t0 else np.inf
    duration = tf - t0
    if duration < context.d_min or i == j:
        return np.inf
    if duration > context.d_max:
        return shen_casalino(context, i, j, t0, t0 + context.d_max)
    return shen_casalino(context, i, j, t0, tf)


@njit(**JIT)
def shen_casalino_table(context, epochs):
    """`table[i, j, h, k]`: the leg from `i` at `epochs[h]` to `j` at
    `epochs[k]`."""
    n = context.a.size
    count = epochs.size
    table = np.full((n, n, count, count), np.inf)
    for i in range(n):
        for j in range(n):
            for h in range(count):
                for k in range(h, count):
                    table[i, j, h, k] = shen_casalino_leg(
                        context, i, j, epochs[h], epochs[k]
                    )
    return table


@njit(**KERNEL)
def _shen_casalino_evaluate(kernel, i, j, t):
    """The cheapest transfer leaving at `t`, over the allowed durations."""
    context = kernel.context
    if i == DEPOT and context.free_injection:
        return 0.0, 0.0
    best_dt = context.d_min
    best_dv = np.inf
    duration = context.d_min
    while duration <= context.d_max:
        dv = shen_casalino(context, i, j, t, t + duration)
        if dv < best_dv:
            best_dt, best_dv = duration, dv
        duration += context.sample
    return best_dt, best_dv


class ShenCasalinoTransfer(TransferLayer):
    """The transfer model of Zona et al. (2023): a leg is priced between a
    departure and an arrival date (see `shen_casalino`).

    Use it with the `epoch` schedule, which reads :meth:`table`; `evaluate`
    answers the one-date question with the cheapest duration.

    Parameters
    ----------
    d_min, d_max : float or str
        Shortest and longest transfer durations.
    tolerance : float
        Node gap at arrival [deg] below which the planes count as aligned.
    sample : float or str
        Duration step over which `evaluate` looks for the cheapest transfer.
    free_injection : bool
        Whether a chaser starts on its first target for free.
    """

    def __init__(
        self,
        *,
        d_min: Any = "30d",
        d_max: Any = "200d",
        tolerance: float = 1.0,
        sample: Any = "1d",
        free_injection: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.d_min = float(runner.duration(d_min))
        self.d_max = float(runner.duration(d_max))
        self.tolerance = math.radians(float(tolerance))
        self.sample = float(runner.duration(sample))
        self.free_injection = bool(free_injection)
        if not 0.0 < self.d_min <= self.d_max:
            raise ValueError("durations must satisfy 0 < d_min <= d_max")
        if self.sample <= 0.0:
            raise ValueError("sample must be positive")

    def _context(self) -> ShenCasalinoContext:
        states = self.problem.states
        return ShenCasalinoContext(
            np.array([s.a for s in states]),
            np.array([s.i for s in states]),
            np.array([s.r for s in states]),
            np.array([s.dr for s in states]),
            self.d_min,
            self.d_max,
            self.tolerance,
            self.sample,
            self.free_injection,
        )

    def _kernel(self) -> TransferKernel:
        return TransferKernel(self._context(), _shen_casalino_evaluate)

    def table(self, epochs: np.ndarray) -> np.ndarray:
        """Every leg between two of `epochs` [s], as `table[i, j, h, k]`."""
        return shen_casalino_table(
            self._context(), np.ascontiguousarray(epochs, dtype=np.float64)
        )


# ============================================================================ #
# epoch: both dates of every leg, on a grid of encounter epochs (Zona et al.)
# ============================================================================ #


class EpochContext(NamedTuple):
    epochs: np.ndarray  # encounter epochs [s]
    table: np.ndarray  # `table[i, j, h, k]`: leg from `i` at `h` to `j` at `k`
    bound: np.ndarray  # `bound[i, j]`: cheapest leg over all epoch pairs
    shortest: float  # shortest leg on the grid [s], the first one excepted
    horizon: float  # mission horizon [s]


@njit(**JIT)
def _bounds(table):
    """The cheapest of every leg over all pairs of epochs."""
    n = table.shape[0]
    out = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            out[i, j] = table[i, j].min()
    return out


@njit(**KERNEL)
def _epoch_evaluate(kernel, seq):
    """Dynamic program choosing the epoch of every encounter along `seq`.

    `cost[l, k]` is the cheapest way to meet the `l`-th object at epoch `k`.
    A leg leaves its object at the epoch it was met: any wait is folded into
    the leg by the transfer model.
    """
    context = kernel.context
    epochs = context.epochs
    table = context.table
    count = epochs.size
    nseq = len(seq)
    legs = nseq - 1

    cost = np.full((nseq, count), np.inf)
    back = np.full((nseq, count), -1, dtype=np.int64)
    cost[0, 0] = 0.0
    for l in range(legs):
        a = seq[l]
        b = seq[l + 1]
        for h in range(count):
            here = cost[l, h]
            if here == np.inf:
                continue
            for k in range(h, count):
                candidate = here + table[a, b, h, k]
                if candidate < cost[l + 1, k]:
                    cost[l + 1, k] = candidate
                    back[l + 1, k] = h

    depart = np.zeros(nseq)
    arrive = np.zeros(nseq)
    costs = np.zeros(nseq)

    last = np.argmin(cost[legs])
    if cost[legs, last] == np.inf:
        # Too many legs for the horizon: price them at their bounds and date
        # them so that the plan overruns the horizon, for the search to see.
        total = 0.0
        for l in range(legs):
            costs[l + 1] = context.bound[seq[l], seq[l + 1]]
            total += costs[l + 1]
            arrive[l + 1] = l * context.shortest
            depart[l + 1] = arrive[l + 1]
        arrive[legs] = max(arrive[legs], context.horizon + context.shortest)
        depart[legs] = arrive[legs]
        return total, arrive[legs], depart, arrive, costs

    chosen = np.empty(nseq, dtype=np.int64)
    chosen[legs] = last
    for l in range(legs, 0, -1):
        chosen[l - 1] = back[l, chosen[l]]

    total = 0.0
    for l in range(1, nseq):
        arrive[l] = epochs[chosen[l]]
        depart[l] = arrive[l]
        costs[l] = table[seq[l - 1], seq[l], chosen[l - 1], chosen[l]]
        total += costs[l]
    # The chaser leaves the depot when it is put on its first target.
    depart[0] = arrive[1]
    return total, arrive[legs], depart, arrive, costs


@njit(**KERNEL)
def _epoch_evaluate_lb(kernel, seq):
    bound = kernel.context.bound
    total = 0.0
    for l in range(len(seq) - 1):
        total += bound[seq[l], seq[l + 1]]
    return total


class EpochSchedule(ScheduleLayer):
    """Both dates of every leg, on a grid of epochs `step` apart.

    Encounters fall on `0, step, ..., Tmax - step`, the grid of Zona et al.:
    their Table 8 meets its first debris at 0 and its last at `Tmax - step`.

    The transfer layer must provide `table(epochs)` (as `ShenCasalinoTransfer`
    does). The table is built once per problem; each schedule is then an exact
    dynamic program over the grid, `O(n m^2)` for `n` legs and `m` epochs.
    """

    def __init__(self, transfer=None, *, step: Any = "20d", **kwargs) -> None:
        super().__init__(transfer, **kwargs)
        self.step = float(runner.duration(step))
        if self.step <= 0.0:
            raise ValueError("step must be positive")
        if not hasattr(self.transfer, "table"):
            raise TypeError(
                "the epoch schedule prices legs between two dates, which "
                f"{type(self.transfer).__name__} does not provide"
            )

    def _kernel(self) -> ScheduleKernel:
        horizon = self.problem.max_time
        count = max(1, int(math.floor(horizon / self.step + 1e-9)))
        epochs = self.step * np.arange(count, dtype=np.float64)
        table = self.transfer.table(epochs)
        shortest = self.step * math.ceil(self.transfer.d_min / self.step)
        context = EpochContext(
            epochs, table, _bounds(table), shortest, float(horizon)
        )
        return ScheduleKernel(context, _epoch_evaluate, _epoch_evaluate_lb)


# ============================================================================ #
# The planners of the literature, as sequence layers
# ============================================================================ #


def _mission(
    layer: SequenceLayer, routes: Sequence[Sequence[int]]
) -> MissionPlan:
    """Schedule one tour per chaser from the depot, the rest staying idle."""
    problem = layer.problem
    if len(routes) > problem.num_chasers:
        raise ValueError(
            f"{len(routes)} tours for {problem.num_chasers} chasers"
        )
    plans = []
    for c in range(problem.num_chasers):
        route = tuple(int(t) for t in routes[c]) if c < len(routes) else ()
        if route:
            plan = layer.schedule.evaluate((DEPOT,) + route)
        else:
            plan = ChaserPlan.empty()
        plans.append(problem.refresh(plan))
    return MissionPlan(plans)


def _gap(a: np.ndarray, b: float) -> np.ndarray:
    """Angular distance between right ascensions [rad], in `[0, pi]`."""
    d = np.mod(np.abs(a - b), TAU)
    return np.minimum(d, TAU - d)


def _raans(layer: SequenceLayer) -> Tuple[np.ndarray, np.ndarray]:
    """Right ascension at the epoch, and its drift rate, of every state."""
    states = layer.problem.states
    return (
        np.array([s.r for s in states], dtype=np.float64),
        np.array([s.dr for s in states], dtype=np.float64),
    )


class FixedSequence(SequenceLayer):
    """Given tours, one per chaser, scheduled by the layer underneath.

    A tour may start with the depot (`0`) and need not visit every target, so
    a run should report how many targets were visited. Chasers beyond the
    last tour stay idle.
    """

    def __init__(
        self,
        schedule: ScheduleLayer | None = None,
        *,
        sequences: Sequence[Sequence[int]] = (),
    ) -> None:
        super().__init__(schedule)
        self.sequences = [tuple(int(t) for t in tour) for tour in sequences]

    def _evaluate(self) -> MissionPlan:
        targets = self.problem.targets
        seen: set = set()
        routes: List[Tuple[int, ...]] = []
        for tour in self.sequences:
            route = tour[1:] if tour and tour[0] == DEPOT else tour
            for target in route:
                if target not in targets:
                    raise ValueError(f"{target} is not a target")
                if target in seen:
                    raise ValueError(f"{target} is visited twice")
                seen.add(target)
            routes.append(route)
        return _mission(self, routes)


def raan_walk(raan: np.ndarray) -> List[int]:
    """The cheapest open tour from state 0, priced by the gap in right
    ascension at the epoch.

    On a circle, the candidates are the sweeps going one way up to some
    target and then back past the start to the others (either way round, or
    without turning); each is priced on the shortest arcs. This reproduces the
    exact tour of Narayanaswamy et al. (2023).
    """
    n = raan.size - 1
    if n == 0:
        return []
    up = np.mod(raan[1:] - raan[0], TAU)
    order = [int(k) + 1 for k in np.argsort(up, kind="stable")]

    candidates = [order, order[::-1]]
    for k in range(1, n):
        candidates.append(order[:k] + order[k:][::-1])
        candidates.append(order[k:][::-1] + order[:k])

    best, best_cost = order, math.inf
    for route in candidates:
        stops = np.array([0] + route)
        legs = np.mod(np.abs(np.diff(raan[stops])), TAU)
        cost = float(np.minimum(legs, TAU - legs).sum())
        if cost < best_cost:
            best, best_cost = route, cost
    return best


class RaanWalk(SequenceLayer):
    """One chaser along :func:`raan_walk`, the static planner of the
    literature."""

    def _evaluate(self) -> MissionPlan:
        raan, _ = _raans(self)
        return _mission(self, [raan_walk(raan)])


class _Greedy(SequenceLayer):
    """One chaser choosing its next target where and when it stands, then
    flying there at once (the clock follows the transfer layer)."""

    def _prepare(self) -> None:
        """Read what `_pick` needs from the problem, once per solve."""

    def _pick(self, at: int, when: float, left: np.ndarray) -> int:
        raise NotImplementedError

    def _evaluate(self) -> MissionPlan:
        self._prepare()
        transfer = self.schedule.transfer
        left = np.asarray(self.problem.targets, dtype=np.int64)
        at, when, route = DEPOT, 0.0, []
        while left.size:
            nxt = self._pick(at, when, left)
            dt, _ = transfer.evaluate(at, nxt, when)
            when += dt
            route.append(nxt)
            left = left[left != nxt]
            at = nxt
        return _mission(self, [route])


class DynamicRaanWalk(_Greedy):
    """The target nearest in right ascension at the current time: the dynamic
    RAAN walk of López Rivera and Naeije (2025)."""

    def _prepare(self) -> None:
        self._raan, self._rate = _raans(self)

    def _pick(self, at: int, when: float, left: np.ndarray) -> int:
        here = self._raan[at] + self._rate[at] * when
        there = self._raan[left] + self._rate[left] * when
        return int(left[np.argmin(_gap(there, here))])



planners.TRANSFERS.update(
    {
        "qlaw_frozen": FrozenQlawTransfer,
        "qlaw_lopez_rivera": LopezRiveraTransfer,
        "shen_casalino": ShenCasalinoTransfer,
    }
)
planners.SCHEDULES.update({"epoch": EpochSchedule})
planners.SEQUENCES.update(
    {
        "fixed": FixedSequence,
        "raan_walk": RaanWalk,
        "drw": DynamicRaanWalk,
    }
)
planners.GENERATORS["iridium33_rivera"] = lambda count, draw, **_: sample_cloud(
    "iridium33_rivera", count, draw
)


# ============================================================================ #
# The experiment
# ============================================================================ #


def method(config: dict) -> str:
    """What planned the mission: `method` when a run flies a published tour
    (the `fixed` layer cannot tell two papers apart), else the sequence layer,
    `mcrdv` for the search."""
    named = config.get("method")
    if named:
        return str(named)
    kind, _ = planners.layer(config, "sequence")
    return "mcrdv" if kind in ("", "hgs") else kind.replace("_", "-")


def waiting(config: dict) -> bool:
    """Whether the schedule layer let a chaser wait before a departure."""
    _, args = planners.layer(config, "schedule")
    return bool(args.get("allow_waiting", True))


def execute(config: dict) -> dict:
    result = planners.solve(config)
    problem = planners.problem(config)
    mission = result["mission"]
    result.update(planners.mission_metrics(mission, problem))
    tours = [list(plan.sequence[1:]) for plan in mission.plans if plan.load]
    # A published tour may stop short of the whole catalogue.
    result["visited"] = sum(len(tour) for tour in tours)
    result["tours"] = tours
    # The static cost Narayanaswamy et al. sequence on, whatever made the plan.
    result["inclination"] = sum(inclination(problem, tour) for tour in tours)
    result["method"] = method(config)
    result["waiting"] = waiting(config)
    if planners.layer(config, "transfer")[0] == "qlaw_lopez_rivera":
        result.update(lopez_rivera_cost(config, problem, mission))
    return result


def lopez_rivera_cost(config, problem, mission) -> dict:
    """A mission on López Rivera & Naeije's terms: each chaser ends with a
    transfer to their decommissioning orbit, which the plan does not include
    (the search does not see it). `dv_lopez_rivera` sums the delta-v of every
    leg, as their Eq. 16 does; `dv_tour` reads Eq. 15 on the fuel of the whole
    tour instead, the tank run down across the legs."""
    layer = planners.transfer(config)
    layer.initialize(problem)
    final_dv = final_tof = flown = 0.0
    for plan in mission.plans:
        if not plan.load:
            continue
        tof, dv = layer.decommission(plan.sequence[-1], float(plan.arrive[-1]))
        final_dv += dv
        final_tof += tof
        flown += float(plan.time) + tof
    return {
        "dv_decommission": final_dv,
        "tof_decommission": final_tof,
        "dv_lopez_rivera": float(mission.cost) + final_dv,
        "dv_tour": layer.tour_dv(flown),
    }


if __name__ == "__main__":
    raise SystemExit(runner.main(__file__, execute, warmup=planners.warm_up))

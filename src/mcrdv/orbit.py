"""Orbital state representation and geometric utilities."""

import math
from dataclasses import dataclass, field
from typing import Iterable, Self, Sequence, Tuple

import numpy as np
from numba import njit

from .kernel import JIT_INLINE

# ========================================================================= #
# Constants
# ========================================================================= #


# Full geometric turn [rad]
TAU = 2.0 * math.pi

# One day [s]
DAY = 86_400.0

# Earth gravitational parameter [m^3/s^2]
MU = 3.986_004e14

# Earth equatorial radius [m]
RE = 6.378_137e6

# Earth second zonal harmonic [-]
J2 = 1.082_626e-3


# ========================================================================= #
# Compiled kernels
# ========================================================================= #


NC = 8  # number of indices in a packed state table
SA = 0  # index for semi-major axis
SE = 1  # index for eccentricity
SI = 2  # index for inclination
SR = 3  # index for right ascension of the ascending node
SO = 4  # index for argument of periapsis
ST = 5  # index for true anomaly
DR = 6  # index for secular drift rate of the right ascension
DO = 7  # index for secular drift rate of the argument of periapsis


@njit(**JIT_INLINE)
def kep_drift_rates(a: float, e: float, i: float) -> Tuple[float, float]:
    """Secular J2 drift rates `(dr/dt,do/dt)` of an Keplerian orbit [rad/s]."""
    n = math.sqrt(MU / (a * a * a))
    p = a * (1.0 - e * e)
    f = 1.5 * J2 * (RE / p) ** 2 * n
    return -f * math.cos(i), f * (2.0 - 2.5 * math.sin(i) ** 2)


@njit(**JIT_INLINE)
def kep_to_mee(
    a: float,
    e: float,
    i: float,
    r: float,
    o: float,
    t: float,
) -> Tuple[float, float, float, float, float, float]:
    """Convert Keplerian elements `(a,e,i,r,o,t)` to modified equinoctial
    elements `(a,f,g,h,k,l)`."""
    w = o + r
    s = math.tan(0.5 * i)
    return (
        a,
        e * math.cos(w),
        e * math.sin(w),
        s * math.cos(r),
        s * math.sin(r),
        w + t,
    )


@njit(**JIT_INLINE)
def mee_at(
    s: np.ndarray,
    i: int,
    t: float,
) -> Tuple[float, float, float, float, float, float]:
    """Modified equinoctial elements `(a, f, g, h, k, l)` of the state `i` in a
    packed table `s`, carried to time `t` by its secular J2 drift."""
    r = s[i, SR] + s[i, DR] * t
    o = s[i, SO] + s[i, DO] * t
    w = o + r
    e = s[i, SE]
    n = math.tan(0.5 * s[i, SI])
    return (
        s[i, SA],
        e * math.cos(w),
        e * math.sin(w),
        n * math.cos(r),
        n * math.sin(r),
        w + s[i, ST],
    )


@njit(**JIT_INLINE)
def mee_max_rates(
    a: float, f: float, g: float, h: float, k: float, l: float
) -> Tuple[float, float, float, float, float, float]:
    """Fastest rate of change of each equinoctial element per thrust unit. The
    six values are *rates*, not elements: multiplying one by a thrust
    acceleration [m/s^2] gives the maximum achievable rate of its element.
    """
    v = f * f + g * g
    e = math.sqrt(v)
    r = 1.0 - v
    p = a * r
    q = math.sqrt(p / MU)
    s = 1.0 + h * h + k * k
    t = MU / (a * a * a)
    da = 2.0 * math.sqrt((1.0 + e) / (t * (1.0 - e)))
    df = 2.0 * q
    dg = 2.0 * q
    dh = 0.5 * q * s / (f + math.sqrt(1.0 - g * g))
    dk = 0.5 * q * s / (g + math.sqrt(1.0 - f * f))
    dl = q * (h * math.sin(l) - k * math.cos(l)) / math.sqrt(r) + math.sqrt(t)
    return da, df, dg, dh, dk, dl


# ========================================================================= #
# Orbital states
# ========================================================================= #


@dataclass(frozen=True, slots=True)
class KepState:
    """Orbital state in Keplerian elements at a reference time. The secular J2
    drift rates are derived on construction and carried through propagation."""

    a: float  # semi-major axis [m]
    e: float  # eccentricity [proportion]
    i: float  # inclination [rad]
    r: float  # right ascension of the ascending node [rad]
    o: float  # argument of periapsis [rad]
    t: float  # true anomaly [rad]
    dr: float = field(init=False)  # secular drift rate of `r` [rad/s]
    do: float = field(init=False)  # secular drift rate of `o` [rad/s]

    def __post_init__(self) -> None:
        dr, do = kep_drift_rates(self.a, self.e, self.i)
        object.__setattr__(self, "dr", dr)
        object.__setattr__(self, "do", do)

    def as_array(self) -> np.ndarray:
        """State elements as an array `[a, e, i, r, o, t]`."""
        return np.array([self.a, self.e, self.i, self.r, self.o, self.t])

    def propagate(self, dt: float) -> Self:
        """Propagate the state by `dt` seconds under J2 secular drift."""
        return KepState(
            self.a,
            self.e,
            self.i,
            self.r + self.dr * dt,
            self.o + self.do * dt,
            self.t,
        )


@dataclass(frozen=True, slots=True)
class MeeState:
    """Orbital state in modified equinoctial elements at a reference time."""

    a: float  # semi-major axis [m]
    f: float  # `e * cos(o + r)` [-1,+1]
    g: float  # `e * sin(o + r)` [-1,+1]
    h: float  # `tan(i / 2) * cos(r)` [-inf,+inf]
    k: float  # `tan(i / 2) * sin(r)` [-inf,+inf]
    l: float  # true longitude `r + o + t` [rad]

    @classmethod
    def from_kep(cls, kep: KepState) -> Self:
        """Convert from Keplerian elements."""
        return cls(*kep_to_mee(*kep.as_array()))

    def as_array(self) -> np.ndarray:
        """State elements as an array `[a, f, g, h, k, l]`."""
        return np.array([self.a, self.f, self.g, self.h, self.k, self.l])

    def max_rates(self) -> Self:
        """The fastest rate of change of each element per thrust unit."""
        return MeeState(*mee_max_rates(*self.as_array()))


def pack(states: Sequence[KepState]) -> np.ndarray:
    """Pack KepState into a table that numba kernels can read."""
    table = np.empty((len(states), NC), dtype=np.float64)
    for row, s in enumerate(states):
        table[row, SA] = s.a
        table[row, SE] = s.e
        table[row, SI] = s.i
        table[row, SR] = s.r
        table[row, SO] = s.o
        table[row, ST] = s.t
        table[row, DR] = s.dr
        table[row, DO] = s.do
    return table


# ========================================================================= #
# Geometric utilities
# ========================================================================= #


def circular_mean(angles: Iterable[float]) -> float:
    """Circular mean of angles, wrapped to `[0, 2*pi)`."""
    a = np.asarray(list(angles), dtype=np.float64)
    mean = math.atan2(float(np.sin(a).sum()), float(np.cos(a).sum()))
    return mean + TAU if mean < 0.0 else mean


def centroid(states: Sequence[KepState]) -> KepState:
    """Centroid of Keplerian states."""
    n = float(len(states))
    return KepState(
        sum(s.a for s in states) / n,
        sum(s.e for s in states) / n,
        sum(s.i for s in states) / n,
        circular_mean(s.r for s in states),
        circular_mean(s.o for s in states),
        circular_mean(s.t for s in states),
    )

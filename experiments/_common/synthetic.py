"""Synthetic debris clouds, for experiments that need to control an instance.

A shipped catalogue comes as one package: its size, its spread in altitude and
its spread in right ascension all change together from one catalogue to the
next, so comparing two of them changes everything at once. A generated cloud
changes one thing at a time.

The model is a single break-up. A parent on a circular orbit fragments at one
point of that orbit, and every fragment leaves with a small random velocity
kick, which the linearised Gauss equations turn into offsets in semi-major axis,
eccentricity, inclination and right ascension. The cloud is then observed `age`
after the event. The J2 drift rates differ from one fragment to the next, so
the right ascensions -- which is what a transfer mostly pays for -- fan out with
time, the way they do in real clouds. The defaults give a cloud shaped like
Fengyun-1C: a few hundred kilometres of altitude spread, and a median right
ascension some fifty degrees off the cloud's mean.

For one `draw`, every fragment comes from the same random stream, so a cloud of
`n` targets is the first `n` fragments of every larger cloud of that draw. A
scaling curve then adds debris to one cloud, rather than comparing clouds that
also differ in everything else.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

from mcrdv.orbit import MU, RE, TAU, KepState

from . import config as configuration

# The `instance.name` that selects a generated cloud rather than a catalogue.
NAME = "synthetic"

# The shape of a cloud, as `instance.cloud` may override it.
DEFAULTS: Dict[str, Any] = {
    # Altitude of the parent's circular orbit [km].
    "altitude_km": 850.0,
    # Inclination of the parent's orbit [deg].
    "inclination_deg": 98.8,
    # Standard deviation of each component of a fragment's kick [m/s].
    "kick": 50.0,
    # Time from the break-up to the mission start, as a duration.
    "age": "3y",
    # Fragments with a perigee below this altitude re-enter, and are dropped.
    "min_perigee_km": 250.0,
}

# Fragments drawn per batch. Fixed, so the stream -- and therefore which
# fragments a cloud holds -- does not depend on how many are asked for.
_BATCH = 256


def cloud(count: int, draw: int = 0, **knobs: Any) -> List[KepState]:
    """`count` fragments of one break-up, at the mission start.

    Parameters
    ----------
    count : int
        Number of fragments, which become the targets of the instance.
    draw : int
        Seed of the random stream. The same draw always yields the same cloud,
        and a smaller count yields a prefix of a larger one.
    **knobs
        Overrides of `DEFAULTS`.
    """
    unknown = set(knobs) - set(DEFAULTS)
    if unknown:
        known = ", ".join(sorted(DEFAULTS))
        raise ValueError(
            f"unknown cloud parameter(s) {sorted(unknown)} (known: {known})"
        )
    if count < 1:
        raise ValueError("a cloud needs at least one fragment")

    shape = {**DEFAULTS, **knobs}
    a0 = RE + 1e3 * float(shape["altitude_km"])
    i0 = math.radians(float(shape["inclination_deg"]))
    sigma = float(shape["kick"])
    age = configuration.duration(shape["age"])
    floor = RE + 1e3 * float(shape["min_perigee_km"])
    speed = math.sqrt(MU / a0)

    rng = np.random.default_rng(draw)
    # Where on the parent orbit the break-up happens, as an argument of
    # latitude. It decides how an out-of-plane kick splits between inclination
    # and right ascension, so it is a property of the event, shared by all.
    u = float(rng.uniform(0.0, TAU))
    cos_u, sin_u = math.cos(u), math.sin(u)

    fragments: List[KepState] = []
    while len(fragments) < count:
        # Radial, along-track and out-of-plane components of each kick.
        kicks = rng.normal(0.0, sigma, size=(_BATCH, 3))
        # Where each fragment is along its orbit at the mission start. Years
        # after the event this is uniform, and the Q-law estimate ignores it.
        anomalies = rng.uniform(0.0, TAU, size=_BATCH)

        for (radial, along, normal), anomaly in zip(kicks, anomalies):
            # Linearised Gauss equations about a circular orbit.
            a = a0 + 2.0 * a0 * along / speed
            ex = (2.0 * cos_u * along + sin_u * radial) / speed
            ey = (2.0 * sin_u * along - cos_u * radial) / speed
            e = math.hypot(ex, ey)
            i = i0 + cos_u * normal / speed
            raan = sin_u * normal / (speed * math.sin(i0))
            if a * (1.0 - e) < floor:
                continue

            # Carry the fragment through the years since the break-up: its
            # own drift rates move its node and its perigee.
            state = KepState(a, e, i, raan, math.atan2(ey, ex), anomaly)
            state = state.propagate(age)
            fragments.append(
                KepState(
                    state.a,
                    state.e,
                    state.i,
                    state.r % TAU,
                    state.o % TAU,
                    float(anomaly),
                )
            )
            if len(fragments) == count:
                break
    return fragments

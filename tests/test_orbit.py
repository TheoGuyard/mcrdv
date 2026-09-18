"""Orbit module: element conversions, drift and cluster geometry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mcrdv import DAY, centroid
from mcrdv.orbit import KepState, MU, MeeState, TAU, circular_mean
from mcrdv.orbit import J2, RE, kep_drift_rates, mee_at, mee_max_rates, pack

TOL = 1e-12


def rel(a, b):
    """Relative difference, absolute when the reference is zero."""
    return abs(a - b) / abs(b) if b != 0.0 else abs(a - b)


# ----------------------------- construction ----------------------------- #


def test_drift_rates_derived_on_construction():
    s = KepState(7.0e6, 1e-3, 1.2, 0.3, 0.4, 0.5)
    n = math.sqrt(MU / s.a**3)
    p = s.a * (1 - s.e**2)
    f = 1.5 * J2 * (RE / p) ** 2 * n
    assert rel(s.dr, -f * math.cos(s.i)) < TOL
    assert rel(s.do, f * (2.0 - 2.5 * math.sin(s.i) ** 2)) < TOL


def test_drift_rates_kernel_matches_its_interpreted_twin():
    args = (7.0e6, 1e-3, 1.2)
    assert kep_drift_rates(*args) == kep_drift_rates.py_func(*args)


def test_propagate_advances_only_the_two_drifting_angles():
    s = KepState(7.0e6, 1e-3, 1.2, 0.3, 0.4, 0.5)
    dt = 12.5 * DAY
    q = s.propagate(dt)
    assert (q.a, q.e, q.i, q.t) == (s.a, s.e, s.i, s.t)
    assert rel(q.r, s.r + s.dr * dt) < TOL
    assert rel(q.o, s.o + s.do * dt) < TOL
    # drift rates depend on a, e, i alone, so propagation leaves them alone
    assert (q.dr, q.do) == (s.dr, s.do)


def test_state_is_immutable():
    s = KepState(7.0e6, 1e-3, 1.2, 0.3, 0.4, 0.5)
    with pytest.raises(Exception):
        s.a = 1.0


def test_as_array_holds_the_six_elements():
    s = KepState(7.0e6, 1e-3, 1.2, 0.3, 0.4, 0.5)
    assert np.array_equal(s.as_array(), [s.a, s.e, s.i, s.r, s.o, s.t])


def test_pack_layout_matches_the_column_constants():
    states = [KepState(7.0e6 + k, 1e-3, 1.2, 0.3, 0.4, 0.5) for k in range(3)]
    table = pack(states)
    assert table.shape == (3, 8)
    assert table.dtype == np.float64 and table.flags["C_CONTIGUOUS"]
    for k, s in enumerate(states):
        assert list(table[k]) == [s.a, s.e, s.i, s.r, s.o, s.t, s.dr, s.do]


def test_propagating_by_nothing_changes_nothing():
    s = KepState(7.0e6, 1e-3, 1.2, 0.3, 0.4, 0.5)
    assert s.propagate(0.0) == s


def test_propagation_composes_over_successive_steps():
    s = KepState(7.0e6, 1e-3, 1.2, 0.3, 0.4, 0.5)
    once = s.propagate(10.0 * DAY)
    twice = s.propagate(4.0 * DAY).propagate(6.0 * DAY)
    assert rel(twice.r, once.r) < TOL and rel(twice.o, once.o) < TOL


def test_pack_accepts_an_empty_catalogue():
    table = pack([])
    assert table.shape == (0, 8) and table.dtype == np.float64


# ------------------------------ conversions ----------------------------- #


def test_mee_conversion_definition():
    s = KepState(7.0e6, 2e-3, 1.2, 0.3, 0.4, 0.5)
    m = MeeState.from_kep(s)
    w = s.o + s.r
    assert rel(m.a, s.a) < TOL
    assert rel(m.f, s.e * math.cos(w)) < TOL
    assert rel(m.g, s.e * math.sin(w)) < TOL
    assert rel(m.h, math.tan(s.i / 2) * math.cos(s.r)) < TOL
    assert rel(m.k, math.tan(s.i / 2) * math.sin(s.r)) < TOL
    assert rel(m.l, s.r + s.o + s.t) < TOL


def test_mee_at_equals_convert_after_propagate():
    states = [
        KepState(7.0e6, 2e-3, 1.2, 0.3, 0.4, 0.5),
        KepState(7.1e6, 1e-3, 1.5, 2.0, 1.0, 3.0),
    ]
    table = pack(states)
    t = 33.0 * DAY
    for i, s in enumerate(states):
        got = mee_at(table, i, t)
        want = MeeState.from_kep(s.propagate(t))
        for a, b in zip(got, (want.a, want.f, want.g, want.h, want.k, want.l)):
            assert rel(a, b) < TOL


def test_kernels_match_their_interpreted_twins():
    table = pack([KepState(7.0e6, 2e-3, 1.2, 0.3, 0.4, 0.5)])
    assert mee_at(table, 0, 1e6) == mee_at.py_func(table, 0, 1e6)
    m = mee_at(table, 0, 1e6)
    assert mee_max_rates(*m) == mee_max_rates.py_func(*m)


def test_max_rates_of_a_circular_equatorial_orbit():
    # f = g = h = k = 0, so the closed forms collapse to readable values.
    a = 7.0e6
    q = math.sqrt(a / MU)
    da, df, dg, dh, dk, _ = mee_max_rates(a, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert rel(da, 2.0 * math.sqrt(a**3 / MU)) < TOL
    assert rel(df, 2.0 * q) < TOL and rel(dg, 2.0 * q) < TOL
    assert rel(dh, 0.5 * q) < TOL and rel(dk, 0.5 * q) < TOL


def test_mee_round_trips_through_its_array():
    m = MeeState.from_kep(KepState(7.0e6, 2e-3, 1.2, 0.3, 0.4, 0.5))
    assert MeeState(*m.as_array()) == m
    assert np.array_equal(
        m.max_rates().as_array(), mee_max_rates(*m.as_array())
    )


# ---------------------------- cluster geometry --------------------------- #


def test_circular_mean_wraps_around_zero():
    assert (
        circular_mean([0.1, TAU - 0.1]) < 1e-12
        or circular_mean([0.1, TAU - 0.1]) > TAU - 1e-12
    )
    assert rel(circular_mean([0.5]), 0.5) < TOL
    assert rel(circular_mean([1.0, 2.0]), 1.5) < TOL


def test_circular_mean_is_always_in_range():
    rng = np.random.default_rng(0)
    for _ in range(50):
        m = circular_mean(rng.uniform(-10, 10, size=7))
        assert 0.0 <= m < TAU


def test_centroid_averages_the_scalars_and_wraps_the_angles():
    states = [
        KepState(7.0e6, 1e-3, 1.2, 0.1, 0.2, 0.3),
        KepState(7.2e6, 3e-3, 1.4, TAU - 0.1, 0.4, 0.5),
    ]
    c = centroid(states)
    assert rel(c.a, 7.1e6) < TOL
    assert rel(c.e, 2e-3) < TOL
    assert rel(c.i, 1.3) < TOL
    assert c.r < 1e-9 or c.r > TAU - 1e-9  # mean of +0.1 and -0.1
    assert rel(c.o, 0.3) < TOL


def test_centroid_of_a_single_state_is_that_state():
    s = KepState(7.0e6, 1e-3, 1.2, 0.3, 0.4, 0.5)
    c = centroid([s])
    for a, b in zip(c.as_array(), s.as_array()):
        assert rel(a, b) < TOL


def test_centroid_of_a_shipped_catalogue_sits_inside_its_spread(any_catalogue):
    # The depot every problem is built with. It is an average, so it must land
    # within the range of what it averages -- the wrapped angles excepted.
    targets = any_catalogue.states[1:]
    depot = any_catalogue.states[0]
    for attribute in ("a", "e", "i"):
        values = [getattr(d, attribute) for d in targets]
        assert min(values) <= getattr(depot, attribute) <= max(values)
    assert 0.0 <= depot.r < TAU and 0.0 <= depot.o < TAU

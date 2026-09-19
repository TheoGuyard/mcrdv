"""Transfer layer: leg cost, bounds, departure hints, and their caches."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numba import njit

from mcrdv import Problem
from mcrdv.kernel import (
    TransferKernel,
    _sample_time,
    python_transfer_evaluate,
    transfer_evaluate_lb_default,
    transfer_evaluate,
    transfer_evaluate_lb,
)
from mcrdv.orbit import KepState
from mcrdv.planner import TransferLayer
from mcrdv.transfer import QlawTransfer
from mcrdv.transfer.qlaw import _qlaw_transfer_evaluate

from .conftest import MAX_TIME, catalogue


@pytest.fixture(scope="module")
def layer(iridium):
    problem = iridium
    transfer = QlawTransfer()
    transfer.initialize(problem)
    return transfer, problem


# -------------------------------- physics -------------------------------- #


def test_a_leg_to_the_same_state_is_free(layer):
    transfer, problem = layer
    for i in (0, 1, 50, 110):
        dt, dc = transfer.evaluate(i, i, 0.0)
        assert dt == 0.0 and dc == 0.0


def test_cost_grows_with_the_plane_change():
    # Same orbit but for the RAAN: the further the plane, the dearer the leg.
    base = dict(a=7.0e6, e=1e-3, i=1.2, o=0.4, t=0.5)
    states = [KepState(r=0.0, **base)] + [
        KepState(r=x, **base) for x in (0.05, 0.2, 0.6)
    ]
    problem = Problem(states, 3, MAX_TIME, 3)
    transfer = QlawTransfer()
    transfer.initialize(problem)
    costs = [transfer.evaluate(0, j, 0.0)[1] for j in (1, 2, 3)]
    assert costs == sorted(costs)
    assert costs[0] > 0.0


def test_time_and_cost_are_tied_by_thrust_and_the_cost_factors():
    states = [
        KepState(7.0e6, 1e-3, 1.2, 0.0, 0.4, 0.5),
        KepState(7.1e6, 2e-3, 1.3, 0.2, 0.4, 0.5),
    ]
    problem = Problem(states, 1, MAX_TIME, 1)
    for ct, cf in ((0.0, 1.0), (1.0, 0.0), (0.5, 2.0)):
        transfer = QlawTransfer(factor_time=ct, factor_fuel=cf)
        transfer.initialize(problem)
        dt, dc = transfer.evaluate(0, 1, 0.0)
        dv = dt * transfer.thrust
        assert math.isclose(dc, ct * dt + cf * dv, rel_tol=1e-13)


def test_the_kernel_matches_its_interpreted_twin(layer):
    transfer, _ = layer
    kernel = transfer.kernel()
    assert _qlaw_transfer_evaluate(
        kernel, 3, 7, 1e6
    ) == _qlaw_transfer_evaluate.py_func(kernel, 3, 7, 1e6)


# --------------------------------- bounds -------------------------------- #


def test_the_bound_is_never_beaten_by_a_sampled_departure(layer):
    transfer, problem = layer
    for i, j in ((0, 5), (5, 17), (110, 1), (33, 90)):
        lt, lc = transfer.evaluate_lb(i, j)
        for k in range(0, transfer.num_samples, 17):
            dt, dc = transfer.evaluate(
                i, j, _sample_time(k, transfer.num_samples, MAX_TIME)
            )
            assert lt <= dt * (1 + 1e-12)
            assert lc <= dc * (1 + 1e-12)


def test_the_two_bound_components_are_minimised_independently():
    # With a time-only cost, the cheapest departure and the quickest one are
    # the same, so the pair is attainable; with mixed factors they need not be,
    # and each component must still bound its own quantity.
    problem = catalogue("iridium33")
    transfer = QlawTransfer(factor_time=1.0, factor_fuel=7.0)
    transfer.initialize(problem)
    lt, lc = transfer.evaluate_lb(5, 40)
    times, costs = zip(
        *(
            transfer.evaluate(
                5, 40, _sample_time(k, transfer.num_samples, MAX_TIME)
            )
            for k in range(transfer.num_samples)
        )
    )
    assert math.isclose(lt, min(times), rel_tol=1e-15)
    assert math.isclose(lc, min(costs), rel_tol=1e-15)


def test_the_bound_covers_the_mission_window(layer):
    transfer, problem = layer
    assert transfer.kernel().max_time == problem.max_time


def test_one_transfer_layer_can_back_several_schedule_layers(iridium):
    from mcrdv.schedule import DpSchedule

    transfer = QlawTransfer()
    a, b = DpSchedule(transfer=transfer), DpSchedule(
        transfer=transfer, allow_waiting=False
    )
    a.initialize(iridium)
    b.initialize(iridium)
    assert a.evaluate_lb((0, 5, 17)) == b.evaluate_lb((0, 5, 17))


# --------------------------------- caches -------------------------------- #


def test_the_bound_table_fills_as_legs_are_asked_for(iridium):
    transfer = QlawTransfer()
    transfer.initialize(iridium)
    table = transfer.kernel().cache_lb
    assert table.shape == (iridium.num_states, iridium.num_states, 2)
    assert np.all(np.isnan(table))
    bound = transfer.evaluate_lb(4, 9)
    assert tuple(table[4, 9]) == bound
    assert np.count_nonzero(~np.isnan(table[:, :, 1])) == 1


def test_the_memo_fills_as_transfers_are_asked_for(iridium):
    transfer = QlawTransfer(use_cache=True)
    transfer.initialize(iridium)
    value = transfer.evaluate(4, 9, 1e6)
    assert dict(transfer.kernel().cache) == {(4, 9, 1e6): value}
    assert transfer.evaluate(4, 9, 1e6) == value
    assert len(transfer.kernel().cache) == 1


def test_caching_changes_nothing_but_speed(iridium):
    plain = QlawTransfer(use_cache=False, use_cache_lb=False)
    cached = QlawTransfer(use_cache=True, use_cache_lb=True)
    plain.initialize(iridium)
    cached.initialize(iridium)
    assert plain.kernel().cache is None and plain.kernel().cache_lb is None
    for _ in range(2):  # the second round is served from the caches
        for i, j in ((0, 5), (5, 17), (110, 1), (33, 90), (12, 12)):
            assert plain.evaluate(i, j, 3e6) == cached.evaluate(i, j, 3e6)
            assert plain.evaluate_lb(i, j) == cached.evaluate_lb(i, j)
            assert np.array_equal(plain.hints(i, j), cached.hints(i, j))


def test_initializing_again_empties_the_caches(iridium, odrc):
    transfer = QlawTransfer(use_cache=True)
    transfer.initialize(iridium)
    transfer.evaluate(4, 9, 0.0)
    transfer.evaluate_lb(4, 9)
    transfer.initialize(odrc)
    assert len(transfer.kernel().cache) == 0
    assert transfer.kernel().cache_lb.shape == (
        odrc.num_states,
        odrc.num_states,
        2,
    )
    assert np.all(np.isnan(transfer.kernel().cache_lb))


# --------------------------------- hints --------------------------------- #


def test_hints_are_ordered_departure_times_inside_the_window(layer):
    transfer, _ = layer
    for i, j in ((0, 5), (5, 17), (110, 1), (33, 90), (12, 12)):
        h = transfer.hints(i, j)
        assert len(h) <= transfer.num_hints
        assert np.all(np.diff(h) > 0) if len(h) > 1 else True
        assert np.all((h >= 0.0) & (h <= MAX_TIME))


def test_a_hint_sits_at_a_local_minimum_of_the_profile(layer):
    transfer, _ = layer
    ns = transfer.num_samples
    costs = np.array(
        [
            transfer.evaluate(5, 40, _sample_time(k, ns, MAX_TIME))[1]
            for k in range(ns)
        ]
    )
    for t in transfer.hints(5, 40):
        k = int(round(t / MAX_TIME * (ns - 1)))
        left = costs[k - 1] if k > 0 else np.inf
        right = costs[k + 1] if k < ns - 1 else np.inf
        assert costs[k] <= max(left, right)


def test_fewer_samples_than_hints_returns_every_sample():
    problem = catalogue("odrc")
    transfer = QlawTransfer(num_samples=4, num_hints=16)
    transfer.initialize(problem)
    h = transfer.hints(1, 2)
    assert len(h) == 4
    assert math.isclose(h[0], 0.0, abs_tol=1e-9) and math.isclose(
        h[-1], MAX_TIME, rel_tol=1e-15
    )


def test_a_single_sample_hints_at_the_start():
    problem = catalogue("odrc")
    transfer = QlawTransfer(num_samples=1, num_hints=16)
    transfer.initialize(problem)
    assert list(transfer.hints(1, 2)) == [0.0]


# ------------------------------- the contract ----------------------------- #


def test_the_layer_publishes_a_usable_compiled_kernel(layer):
    transfer, _ = layer
    kernel = transfer.kernel()
    assert isinstance(kernel, TransferKernel)

    @njit
    def driver(kernel):
        dt, dc = transfer_evaluate(kernel, 3, 9, 1e6)
        return dt + dc + transfer_evaluate_lb(kernel, 3, 9)[1]

    expected = (
        sum(transfer.evaluate(3, 9, 1e6)) + transfer.evaluate_lb(3, 9)[1]
    )
    assert math.isclose(driver(kernel), expected, rel_tol=1e-13)


def test_construction_rejects_impossible_options():
    for kwargs, message in (
        (dict(thrust=0.0), "thrust"),
        (dict(factor=-1.0), "factor"),
        (dict(num_samples=0), "num_samples"),
        (dict(num_hints=-1), "num_hints"),
        (dict(factor_time=-1.0), "factor_time"),
        (dict(factor_fuel=-1.0), "factor_fuel"),
    ):
        with pytest.raises(ValueError, match=message):
            QlawTransfer(**kwargs)


def test_a_fresh_layer_holds_nothing_until_it_is_initialized(iridium):
    # No guard stands in front of the entry points -- every caller reaches a
    # layer through `Planner.solve`, which initializes the whole stack first --
    # so what the contract promises is simply that `initialize` is what fills
    # the layer in, and that it is what a caller must do.
    transfer = QlawTransfer()
    assert transfer.problem is None and transfer.kernel() is None
    transfer.initialize(iridium)
    assert transfer.problem is iridium
    assert transfer.kernel().context is not None


def test_initializing_again_repacks_the_new_catalogue(iridium, odrc):
    transfer = QlawTransfer()
    transfer.initialize(iridium)
    assert transfer.kernel().context.table.shape[0] == iridium.num_states

    transfer.initialize(odrc)
    assert transfer.problem is odrc
    assert transfer.kernel().context.table.shape[0] == odrc.num_states
    assert transfer.evaluate(0, 1, 0.0) == _qlaw_transfer_evaluate(
        transfer.kernel(), 0, 1, 0.0
    )


def test_a_bound_written_in_python_on_a_compiled_layer_is_rejected(iridium):
    # A compiled consumer could never see it, so the two would disagree.
    class Tighter(QlawTransfer):
        def _evaluate_lb(self, i, j):
            return 0.0, 0.0

    with pytest.raises(TypeError, match="is defined by the kernel"):
        Tighter().initialize(iridium)


@njit
def _halved_lb(kernel, i, j):
    dt, dc = transfer_evaluate_lb_default(kernel, i, j)
    return 0.5 * dt, 0.5 * dc


class HalvedBound(QlawTransfer):
    """A compiled layer customizing its bound where compiled callers see it."""

    def _kernel(self):
        return super()._kernel()._replace(evaluate_lb=_halved_lb)


def test_a_bound_customized_in_the_kernel_replaces_the_default(iridium):
    plain, halved = QlawTransfer(), HalvedBound()
    plain.initialize(iridium)
    halved.initialize(iridium)
    dt, dc = plain.evaluate_lb(5, 17)
    assert halved.evaluate_lb(5, 17) == (0.5 * dt, 0.5 * dc)


# --------------- a custom layer in plain Python still composes ------------ #


class ConstantTransfer(TransferLayer):
    """A deliberately trivial layer: `_evaluate`, and not one line more.

    Nothing here is numba-compiled and neither the bound nor the hints are
    written, so both come from the base class, which samples this `_evaluate`
    from compiled code.
    """

    def _evaluate(self, i, j, t):
        gap = abs(self.problem.states[i].r - self.problem.states[j].r)
        return 10.0 * gap, 3.0 * gap


def test_a_pure_python_transfer_layer_satisfies_the_contract(iridium):
    transfer = ConstantTransfer()
    transfer.initialize(iridium)
    # No compiled form of its own: the base class bridges to the Python one.
    assert transfer.kernel().evaluate is python_transfer_evaluate
    assert transfer.evaluate(4, 9, 0.0)[1] > 0.0
    # This leg costs the same whenever it departs, so the sampled bound is the
    # cost itself and the flat profile leaves a single hint at the window start.
    assert transfer.evaluate_lb(4, 9)[1] == transfer.evaluate(4, 9, 0.0)[1]
    assert list(transfer.hints(4, 9)) == [0.0]


def test_the_inherited_defaults_read_the_layers_own_sampling_knobs(iridium):
    transfer = ConstantTransfer()
    assert (transfer.num_samples, transfer.num_hints) == (512, 64)
    transfer = ConstantTransfer(num_samples=4, num_hints=2)
    transfer.initialize(iridium)
    assert (transfer.kernel().num_samples, transfer.kernel().num_hints) == (
        4,
        2,
    )
    assert len(transfer.hints(4, 9)) <= 2


def test_a_python_layer_may_write_its_own_bound_and_hints(iridium):
    class Custom(ConstantTransfer):
        def _evaluate_lb(self, i, j):
            return 1.0, 2.0

        def _hints(self, i, j):
            return [7.0, 8.0]

    transfer = Custom()
    transfer.initialize(iridium)
    assert transfer.evaluate_lb(4, 9) == (1.0, 2.0)
    assert list(transfer.hints(4, 9)) == [7.0, 8.0]


def test_an_uncompiled_layer_bounds_exactly_like_a_compiled_one(iridium):
    """The same compiled defaults run over both, so same answers."""

    class Interpreted(TransferLayer):
        def initialize(self, problem):
            self.inner = QlawTransfer()
            self.inner.initialize(problem)
            super().initialize(problem)

        def _evaluate(self, i, j, t):
            return self.inner.evaluate(i, j, t)

    compiled = QlawTransfer()
    compiled.initialize(iridium)
    interpreted = Interpreted()
    interpreted.initialize(iridium)

    for i, j in ((0, 5), (5, 17), (110, 1)):
        assert interpreted.evaluate_lb(i, j) == compiled.evaluate_lb(i, j)
        assert np.array_equal(interpreted.hints(i, j), compiled.hints(i, j))

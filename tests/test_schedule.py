"""Schedule layer: pricing a fixed sequence, and when to leave each object."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numba import njit

from mcrdv.kernel import (
    ScheduleKernel,
    python_schedule_evaluate,
    schedule_evaluate,
)
from mcrdv.problem import DEPOT, ChaserPlan
from mcrdv.schedule import DpSchedule
from mcrdv.planner import ScheduleLayer, TransferLayer
from mcrdv.transfer import QlawTransfer

from .conftest import catalogue
from .test_transfer import HalvedBound

SEQUENCES = [
    (DEPOT, 5),
    (DEPOT, 5, 17),
    (DEPOT, 12, 3, 44, 7),
    (DEPOT, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    (DEPOT, 100, 20, 60, 3, 88, 41),
]


@pytest.fixture(scope="module")
def layer(iridium):
    problem = iridium
    schedule = DpSchedule(transfer=QlawTransfer())
    schedule.initialize(problem)
    return schedule, problem


@njit
def run_kernel(kernel, seq):
    """A compiled caller: what a layer above would do with the kernel."""
    return schedule_evaluate(kernel, seq)


def run(kernel, sequence):
    return run_kernel(kernel, np.asarray(sequence, dtype=np.int64))


# ------------------------------ plan shape ------------------------------- #


def test_a_lone_object_is_priced_as_an_empty_plan(layer):
    schedule, _ = layer
    plan = schedule.evaluate([DEPOT])
    assert plan.sequence == (DEPOT,) and plan.cost == 0.0 and plan.time == 0.0
    assert schedule.evaluate([7]).sequence == (7,)


def test_an_empty_sequence_is_rejected(layer):
    schedule, _ = layer
    with pytest.raises(ValueError, match="empty sequence"):
        schedule.evaluate([])


@pytest.mark.parametrize("sequence", SEQUENCES)
def test_the_plan_is_internally_consistent(layer, sequence):
    schedule, _ = layer
    plan = schedule.evaluate(sequence)
    assert plan.sequence == sequence
    assert len(plan) == len(sequence)
    assert plan.costs[0] == 0.0  # nothing is paid to start
    assert plan.arrive[0] == 0.0
    assert plan.depart[-1] == plan.arrive[-1]  # nowhere left to go
    assert math.isclose(plan.cost, plan.costs.sum(), rel_tol=1e-12)
    assert plan.time == plan.arrive[-1]
    for i in range(len(plan) - 1):
        assert plan.depart[i] >= plan.arrive[i] - 1e-6  # never leave early
        assert (
            plan.arrive[i + 1] >= plan.depart[i] - 1e-6
        )  # never arrive early
    assert np.all(np.diff(plan.arrive) >= -1e-6)


def test_the_layer_measures_the_overruns_it_hands_back(iridium):
    problem = iridium
    schedule = DpSchedule(transfer=QlawTransfer())
    schedule.initialize(problem)
    for sequence in SEQUENCES + [tuple([DEPOT] + list(range(1, 40)))]:
        plan = schedule.evaluate(sequence)
        assert schedule.evaluate(sequence) is problem.refresh(plan)


def test_an_overlong_sequence_comes_back_marked_infeasible(iridium):
    problem = iridium
    schedule = DpSchedule(transfer=QlawTransfer())
    schedule.initialize(problem)
    plan = schedule.evaluate(
        tuple([DEPOT] + list(range(1, problem.max_load + 4)))
    )
    assert plan.excess_load == 3
    assert not plan.feasible


# ------------------------------ the waiting ------------------------------ #


@pytest.mark.parametrize("sequence", SEQUENCES)
def test_waiting_never_costs_more_than_leaving_at_once(iridium, sequence):
    problem = iridium
    waiting = DpSchedule(allow_waiting=True)
    nowait = DpSchedule(allow_waiting=False)
    waiting.initialize(problem)
    nowait.initialize(problem)
    assert waiting.evaluate(sequence).cost <= nowait.evaluate(
        sequence
    ).cost * (1 + 1e-12)


def test_waiting_strictly_helps_somewhere(iridium):
    problem = iridium
    waiting = DpSchedule(allow_waiting=True)
    nowait = DpSchedule(allow_waiting=False)
    waiting.initialize(problem)
    nowait.initialize(problem)
    gains = [
        nowait.evaluate(s).cost - waiting.evaluate(s).cost for s in SEQUENCES
    ]
    assert (
        max(gains) > 0.0
    ), "the dynamic program never beat the no-wait schedule"


def test_no_wait_departs_the_instant_it_arrives(iridium):
    problem = iridium
    schedule = DpSchedule(allow_waiting=False)
    schedule.initialize(problem)
    plan = schedule.evaluate(SEQUENCES[2])
    assert np.allclose(plan.depart[:-1], plan.arrive[:-1])


# ---------------------------- the search knobs ---------------------------- #


def test_a_finer_grid_is_never_worse(iridium):
    problem = iridium
    costs = []
    for grid_size in (4, 16, 64):
        schedule = DpSchedule(transfer=QlawTransfer(), grid_size=grid_size)
        schedule.initialize(problem)
        costs.append(schedule.evaluate(SEQUENCES[3]).cost)
    assert costs[1] <= costs[0] * (1 + 1e-9)
    assert costs[2] <= costs[1] * (1 + 1e-9)


def test_hints_only_add_candidates(iridium):
    problem = iridium
    with_hints = DpSchedule(transfer=QlawTransfer(), use_hints=True)
    without = DpSchedule(transfer=QlawTransfer(), use_hints=False)
    with_hints.initialize(problem)
    without.initialize(problem)
    for sequence in SEQUENCES:
        assert with_hints.evaluate(sequence).cost <= without.evaluate(
            sequence
        ).cost * (1 + 1e-12)


def test_the_cache_returns_the_very_same_plan(layer):
    schedule, _ = layer
    assert schedule.evaluate(SEQUENCES[1]) is schedule.evaluate(SEQUENCES[1])


def test_a_cached_plan_cannot_be_altered_by_whoever_holds_it(layer):
    schedule, _ = layer
    plan = schedule.evaluate(SEQUENCES[2])
    with pytest.raises(ValueError, match="read-only"):
        plan.depart[1] = 0.0


def test_caching_changes_nothing_but_speed(iridium):
    problem = iridium
    cached = DpSchedule(transfer=QlawTransfer(), use_cache=True)
    uncached = DpSchedule(transfer=QlawTransfer(), use_cache=False)
    cached.initialize(problem)
    uncached.initialize(problem)
    for sequence in SEQUENCES:
        a, b = cached.evaluate(sequence), uncached.evaluate(sequence)
        assert a.cost == b.cost and a.time == b.time
        assert np.array_equal(a.depart, b.depart)
    assert uncached.evaluate(SEQUENCES[0]) is not uncached.evaluate(
        SEQUENCES[0]
    )


def test_turning_every_bound_cache_off_changes_no_plan(iridium):
    # The dynamic program prunes against the per-leg bounds whether or not
    # anything remembers them.
    reference = DpSchedule(transfer=QlawTransfer())
    bare = DpSchedule(
        transfer=QlawTransfer(use_cache_lb=False), use_cache_lb=False
    )
    memoized = DpSchedule(
        transfer=QlawTransfer(use_cache=True), use_cache_lb=True
    )
    for schedule in (reference, bare, memoized):
        schedule.initialize(iridium)
    for sequence in SEQUENCES:
        expected = reference.evaluate(sequence)
        for schedule in (bare, memoized):
            plan = schedule.evaluate(sequence)
            assert plan.cost == expected.cost
            assert np.array_equal(plan.depart, expected.depart)
            assert schedule.evaluate_lb(sequence) == reference.evaluate_lb(
                sequence
            )


def test_a_list_and_a_tuple_are_the_same_sequence(layer):
    schedule, _ = layer
    assert schedule.evaluate(list(SEQUENCES[2])) is schedule.evaluate(
        SEQUENCES[2]
    )


# ----------------------------- lower bounds ------------------------------ #


@pytest.mark.parametrize("sequence", SEQUENCES)
def test_the_bound_never_exceeds_the_plan(layer, sequence):
    schedule, _ = layer
    assert schedule.evaluate_lb(sequence) <= schedule.evaluate(
        sequence
    ).cost * (1 + 1e-12)


def test_the_bound_is_the_sum_of_its_legs(layer):
    schedule, _ = layer
    for sequence in SEQUENCES:
        legs = sum(
            schedule.evaluate_lb((a, b))
            for a, b in zip(sequence, sequence[1:])
        )
        assert math.isclose(
            legs, schedule.evaluate_lb(sequence), rel_tol=1e-12
        )


def test_a_leg_to_the_same_state_bounds_at_zero(layer):
    schedule, _ = layer
    assert schedule.evaluate_lb((5, 5)) == 0.0
    assert schedule.evaluate_lb((7,)) == 0.0


def test_asking_for_a_bound_warms_the_table_the_program_prunes_against(
    iridium,
):
    schedule = DpSchedule(transfer=QlawTransfer())
    schedule.initialize(iridium)
    table = schedule.kernel().transfer.cache_lb
    assert np.all(np.isnan(table))
    schedule.evaluate_lb((4, 9))
    assert not np.isnan(table[4, 9, 1])


def test_a_bound_customized_in_the_transfer_kernel_is_the_one_everyone_uses(
    iridium,
):
    # Whichever of the program or `evaluate_lb` asks first, the leg's bound is
    # the transfer layer's own.
    transfer = HalvedBound()
    transfer.initialize(iridium)
    halved = transfer.evaluate_lb(5, 17)[1]

    program_first = DpSchedule(transfer=HalvedBound())
    program_first.initialize(iridium)
    program_first.evaluate((DEPOT, 5, 17))
    assert program_first.kernel().transfer.cache_lb[5, 17, 1] == halved
    assert program_first.evaluate_lb((5, 17)) == halved

    bound_first = DpSchedule(transfer=HalvedBound())
    bound_first.initialize(iridium)
    assert bound_first.evaluate_lb((5, 17)) == halved


# --------------------------- the compiled path ---------------------------- #


def test_the_kernel_answers_on_its_own(iridium):
    # Nothing is prepared from Python first: the kernel builds what it reads.
    schedule = DpSchedule(transfer=QlawTransfer())
    reference = DpSchedule(transfer=QlawTransfer())
    schedule.initialize(iridium)
    reference.initialize(iridium)
    kernel = schedule.kernel()
    assert isinstance(kernel, ScheduleKernel)
    for sequence in SEQUENCES:
        cost, time, depart, arrive, costs = run(kernel, sequence)
        plan = reference.evaluate(sequence)
        assert (cost, time) == (plan.cost, plan.time)
        assert np.array_equal(depart, plan.depart) and np.array_equal(
            costs, plan.costs
        )


def test_a_kernel_taken_early_stays_valid(iridium):
    schedule = DpSchedule(transfer=QlawTransfer())
    schedule.initialize(iridium)
    kernel = schedule.kernel()
    for d in range(
        1, 60
    ):  # many legs get tabulated after the kernel was taken
        schedule.evaluate((DEPOT, d, d + 1))
    for sequence in ((DEPOT, 42, 43), (DEPOT, 58, 59, 70)):
        cost, _, depart, _, _ = run(kernel, sequence)
        schedule._cache.clear()
        plan = schedule.evaluate(sequence)
        assert cost == plan.cost and np.array_equal(depart, plan.depart)


def test_the_kernel_prices_a_lone_state_as_an_empty_plan(layer):
    schedule, _ = layer
    cost, time, depart, arrive, costs = run(schedule.kernel(), (DEPOT,))
    assert (cost, time) == (0.0, 0.0)
    assert depart.shape == arrive.shape == costs.shape == (1,)
    assert run(schedule.kernel(), ())[2].shape == (0,)


def test_legs_are_only_tabulated_for_sequences_that_reach_the_dynamic_program(
    iridium,
):
    problem = iridium
    schedule = DpSchedule(transfer=QlawTransfer())
    schedule.initialize(problem)
    legs = schedule.kernel().context.legs
    # A long sequence whose no-wait schedule overruns the horizon is answered
    # without sampling a single departure window.
    schedule.evaluate(tuple([DEPOT] + list(range(1, 60))))
    assert len(legs) == 0
    schedule.evaluate((DEPOT, 5))
    assert len(legs) == 1
    times, dt, dc = legs[5]
    assert times[0] == 0.0 and times[-1] == problem.max_time
    assert (dt[3], dc[3]) == schedule.transfer.evaluate(DEPOT, 5, times[3])


class PythonTransfer(TransferLayer):
    """The Q-law physics behind a layer that publishes no compiled kernel.

    `_evaluate` and nothing else: the bound and the hints that drive the grids
    below come from the base class, sampling this through its bridge.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inner = QlawTransfer()

    def initialize(self, problem):
        self.inner.initialize(problem)
        super().initialize(problem)

    def _evaluate(self, i, j, t):
        return self.inner.evaluate(i, j, t)


def test_a_transfer_written_in_python_gives_the_same_plans_as_the_compiled_one():
    problem = catalogue("odrc")
    fast = DpSchedule(transfer=QlawTransfer())
    slow = DpSchedule(transfer=PythonTransfer())
    fast.initialize(problem)
    slow.initialize(problem)
    for sequence in (
        (DEPOT, 5),
        (DEPOT, 5, 3),
        (DEPOT, 12, 3, 2, 7),
        (DEPOT, 2, 6, 4, 3, 13),
    ):
        a, b = fast.evaluate(sequence), slow.evaluate(sequence)
        assert a.cost == b.cost and a.time == b.time
        assert np.array_equal(a.depart, b.depart)
        assert np.array_equal(a.costs, b.costs)


@pytest.mark.filterwarnings("ignore:Code running in object mode")
def test_a_schedule_written_in_python_still_publishes_a_kernel(odrc):
    class Delegating(ScheduleLayer):
        def initialize(self, problem):
            self.inner = DpSchedule(transfer=QlawTransfer())
            self.inner.initialize(problem)
            super().initialize(problem)

        def _evaluate(self, sequence):
            plan = self.inner.evaluate(sequence)
            return ChaserPlan.of(
                sequence, plan.depart, plan.arrive, plan.costs
            )

    layer = Delegating(QlawTransfer())
    layer.initialize(odrc)
    assert layer.kernel().evaluate is python_schedule_evaluate
    for sequence in ((DEPOT, 5), (DEPOT, 12, 3, 2, 7)):
        cost, time, depart, _, _ = run(layer.kernel(), sequence)
        plan = layer.evaluate(sequence)
        assert (
            math.isclose(cost, plan.cost, rel_tol=1e-12) and time == plan.time
        )
        assert np.array_equal(depart, plan.depart)


# --------------------- the bound the layer inherits ----------------------- #


def test_the_inherited_bound_is_the_sum_of_the_transfer_legs(iridium):
    """A schedule layer that writes no bound still has a usable one."""

    class Minimal(ScheduleLayer):

        def _evaluate(self, sequence):
            raise AssertionError("the bound must not need a real schedule")

    transfer = QlawTransfer()
    minimal = Minimal(transfer)
    minimal.initialize(iridium)

    for sequence in SEQUENCES:
        legs = sum(
            transfer.evaluate_lb(a, b)[1]
            for a, b in zip(sequence, sequence[1:])
        )
        assert math.isclose(minimal.evaluate_lb(sequence), legs, rel_tol=1e-15)


def test_a_memoized_bound_agrees_with_a_computed_one(iridium):
    class Minimal(ScheduleLayer):
        def _evaluate(self, sequence):
            raise AssertionError("unused")

    plain = Minimal(QlawTransfer())
    plain.initialize(iridium)
    memoized = DpSchedule(transfer=QlawTransfer(), use_cache_lb=True)
    memoized.initialize(iridium)
    for _ in range(2):
        for sequence in SEQUENCES:
            assert memoized.evaluate_lb(sequence) == plain.evaluate_lb(
                sequence
            )
    assert set(memoized._cache_lb) == set(SEQUENCES)


def test_a_bound_of_a_lone_state_is_zero_with_no_legs_to_pay_for(iridium):
    schedule = DpSchedule(transfer=QlawTransfer())
    schedule.initialize(iridium)
    assert schedule.evaluate_lb((7,)) == 0.0
    assert schedule.evaluate_lb(()) == 0.0


# ------------------------------- the stack -------------------------------- #


def test_construction_rejects_a_bad_stack():
    with pytest.raises(ValueError, match="grid_size"):
        DpSchedule(grid_size=0)


def test_a_fresh_layer_holds_nothing_until_it_is_initialized(iridium):
    # No guard stands in front of `evaluate` -- callers reach the layer through
    # `Planner.solve`, which initializes the whole stack first -- so the contract
    # is that `initialize` fills the layer in, and reaches the layer below too.
    schedule = DpSchedule()
    assert schedule.problem is None and schedule.transfer.problem is None
    assert schedule.kernel() is None
    schedule.initialize(iridium)
    assert schedule.problem is iridium
    assert schedule.transfer.problem is iridium  # the stack, not just the top


def test_initializing_again_drops_the_previous_instance(iridium, odrc):
    # Plans, bounds and leg tables are all memoized per state pair, so a
    # second `initialize` has to throw them away rather than answer for the
    # wrong catalogue.
    schedule = DpSchedule(transfer=QlawTransfer())
    schedule.initialize(iridium)
    schedule.evaluate((DEPOT, 5, 17))
    schedule.evaluate_lb((DEPOT, 5))
    assert schedule._cache and len(schedule.kernel().context.legs) > 0

    schedule.initialize(odrc)
    assert schedule.problem is odrc
    assert schedule._cache == {}
    assert len(schedule.kernel().context.legs) == 0
    assert schedule.kernel().transfer.cache_lb.shape == (
        odrc.num_states,
        odrc.num_states,
        2,
    )


def test_a_shared_transfer_moved_to_another_catalogue_leaves_the_schedule_alone(
    iridium, odrc
):
    # Each schedule keeps the transfer kernel it was assembled with.
    transfer = QlawTransfer()
    moved = DpSchedule(transfer=transfer)
    moved.initialize(odrc)
    moved.evaluate((DEPOT, 1, 2))
    DpSchedule(transfer=transfer).initialize(iridium)

    reference = DpSchedule(transfer=QlawTransfer())
    reference.initialize(odrc)
    for sequence in ((DEPOT, 3, 4), (DEPOT, 12, 3, 2, 7)):
        assert (
            moved.evaluate(sequence).cost == reference.evaluate(sequence).cost
        )
        assert moved.evaluate_lb(sequence) == reference.evaluate_lb(sequence)

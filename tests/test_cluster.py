"""Cluster sequence layer: one deterministic sweep in right ascension."""

from __future__ import annotations

import numpy as np
import pytest
from numba import njit

from mcrdv import Problem
from mcrdv.kernel import SequenceKernel, sequence_evaluate
from mcrdv.orbit import TAU, KepState
from mcrdv.problem import DEPOT, ChaserPlan
from mcrdv.schedule import DpSchedule
from mcrdv.sequence import ClusterSequence
from mcrdv.planner import ScheduleLayer
from mcrdv.transfer import QlawTransfer

from .conftest import MAX_TIME


def layer_for(problem: Problem) -> ClusterSequence:
    sequence = ClusterSequence()
    sequence.initialize(problem)
    return sequence


# ------------------------------- invariants ------------------------------ #


def test_every_target_is_collected_exactly_once(any_catalogue):
    mission = layer_for(any_catalogue).evaluate()
    visited = [d for plan in mission.plans for d in plan.sequence[1:]]
    assert sorted(visited) == list(any_catalogue.targets)


def test_one_plan_per_chaser_each_starting_at_the_depot(any_catalogue):
    mission = layer_for(any_catalogue).evaluate()
    assert len(mission.plans) == any_catalogue.num_chasers
    assert all(plan.sequence[0] == DEPOT for plan in mission.plans)


def test_each_bucket_is_swept_in_right_ascension(iridium):
    mission = layer_for(iridium).evaluate()
    for plan in mission.plans:
        raan = [iridium.states[d].r for d in plan.sequence[1:]]
        assert raan == sorted(raan)


def test_buckets_stay_within_the_load_limit_when_capacity_allows(iridium):
    mission = layer_for(iridium).evaluate()
    assert all(plan.load <= iridium.max_load for plan in mission.plans)


def test_the_sweep_is_deterministic(iridium):
    # One pass with no random source in it: the same instance must give the
    # same buckets every time, which is what makes it usable as a baseline.
    first = layer_for(iridium).evaluate()
    second = layer_for(iridium).evaluate()
    assert [p.sequence for p in first.plans] == [
        p.sequence for p in second.plans
    ]
    assert first.cost == second.cost


def test_a_chaser_each_when_the_swarm_outnumbers_the_targets():
    states = [KepState(7.0e6, 1e-3, 1.2, 0.3 * k, 0.4, 0.5) for k in range(5)]
    problem = Problem(states, 10, MAX_TIME, 4)
    mission = layer_for(problem).evaluate()
    assert len(mission.plans) == 10
    assert sorted(p.load for p in mission.plans) == [0] * 6 + [1] * 4


def test_one_chaser_collects_the_whole_catalogue():
    states = [KepState(7.0e6, 1e-3, 1.2, 0.3 * k, 0.4, 0.5) for k in range(5)]
    mission = layer_for(Problem(states, 1, MAX_TIME, 4)).evaluate()
    assert len(mission.plans) == 1
    assert sorted(mission.plans[0].sequence[1:]) == [1, 2, 3, 4]


def test_every_target_is_placed_even_when_the_buckets_all_fill_up():
    # Capacity is exactly tight, so the nearest-first pass fills every bucket
    # and the fallback has to place what is left rather than drop it.
    states = [KepState(7.0e6, 1e-3, 1.2, 0.5 * k, 0.4, 0.5) for k in range(7)]
    problem = Problem(states, 3, MAX_TIME, 2)
    mission = layer_for(problem).evaluate()
    visited = [d for p in mission.plans for d in p.sequence[1:]]
    assert sorted(visited) == [1, 2, 3, 4, 5, 6]
    assert all(p.load <= 2 for p in mission.plans)


def test_clusters_are_tighter_in_raan_than_a_round_robin(iridium):
    mission = layer_for(iridium).evaluate()

    def spread(indices):
        angles = sorted(iridium.states[d].r for d in indices)
        return max(angles) - min(angles) if len(angles) > 1 else 0.0

    clustered = [spread(p.sequence[1:]) for p in mission.plans if p.load > 1]
    targets = list(iridium.targets)
    k = iridium.num_chasers
    striped = [spread(targets[i::k]) for i in range(k)]
    assert sum(clustered) / len(clustered) < sum(striped) / len(striped)
    assert max(clustered) < TAU


# --------------------------- the compiled path ---------------------------- #


@njit
def run_kernel(kernel):
    """A compiled caller: what a layer above would do with the kernel."""
    return sequence_evaluate(kernel)


def test_the_layer_publishes_a_kernel_over_the_schedule_kernel(iridium):
    sequence = layer_for(iridium)
    kernel = sequence.kernel()
    assert isinstance(kernel, SequenceKernel)
    assert kernel.schedule is sequence.schedule.kernel()


def test_every_plan_is_the_one_the_schedule_layer_gives(any_catalogue):
    sequence = layer_for(any_catalogue)
    for plan in sequence.evaluate().plans:
        expected = sequence.schedule.evaluate(plan.sequence)
        assert (plan.cost, plan.time) == (expected.cost, expected.time)
        assert (plan.excess_load, plan.excess_time) == (
            expected.excess_load,
            expected.excess_time,
        )
        assert np.array_equal(plan.depart, expected.depart)
        assert np.array_equal(plan.costs, expected.costs)


def test_a_compiled_caller_gets_the_same_mission(iridium):
    sequence = layer_for(iridium)
    mission = sequence.evaluate()
    offsets, states, depart, _, _, cost, time = run_kernel(sequence.kernel())
    assert len(offsets) == len(mission.plans) + 1
    for c, plan in enumerate(mission.plans):
        lo, hi = offsets[c], offsets[c + 1]
        assert tuple(states[lo:hi]) == plan.sequence
        assert np.array_equal(depart[lo:hi], plan.depart)
        assert (cost[c], time[c]) == (plan.cost, plan.time)


@pytest.mark.filterwarnings("ignore:Code running in object mode")
def test_the_sweep_runs_over_a_schedule_layer_written_in_python(odrc):
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

    bridged = ClusterSequence(Delegating(QlawTransfer()))
    bridged.initialize(odrc)
    compiled = layer_for(odrc)
    a, b = bridged.evaluate(), compiled.evaluate()
    assert [p.sequence for p in a.plans] == [p.sequence for p in b.plans]
    assert [p.time for p in a.plans] == [p.time for p in b.plans]


# ------------------------------- the stack -------------------------------- #


def test_construction_rejects_a_bad_stack():
    with pytest.raises(TypeError, match="ScheduleLayer"):
        ClusterSequence(object())


def test_a_fresh_layer_holds_no_kernel_until_it_is_initialized(iridium):
    sequence = ClusterSequence()
    assert sequence.problem is None and sequence.kernel() is None
    sequence.initialize(iridium)
    assert sequence.problem is iridium and sequence.kernel() is not None


def test_initializing_again_moves_the_layer_to_the_new_instance(iridium, odrc):
    sequence = ClusterSequence()
    sequence.initialize(iridium)
    sequence.evaluate()
    sequence.initialize(odrc)
    mission = sequence.evaluate()
    assert len(mission.plans) == odrc.num_chasers
    visited = [d for p in mission.plans for d in p.sequence[1:]]
    assert sorted(visited) == list(odrc.targets)

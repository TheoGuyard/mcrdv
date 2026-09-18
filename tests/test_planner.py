"""The planner: how the three layers are wired, and what driving them does.

`Planner` is thin on purpose -- it holds the top layer, initializes the stack
and runs it -- so what is worth pinning here is the contract the layers rely
on. None of them guards its own entry points: they assume `solve` has
initialized them, and these tests are what keeps that assumption true.
"""

from __future__ import annotations

import pytest

from mcrdv import MissionPlan, Planner
from mcrdv.problem import DEPOT
from mcrdv.schedule import DpSchedule
from mcrdv.sequence import ClusterSequence, HgsSequence
from mcrdv.sequence.hgs import HgsConfig
from mcrdv.planner import ScheduleLayer, SequenceLayer, TransferLayer
from mcrdv.transfer import QlawTransfer


def cluster_stack() -> ClusterSequence:
    """The cheapest complete stack: one deterministic sweep, no search."""
    return ClusterSequence()


# ------------------------------- construction ----------------------------- #


def test_the_default_stack_is_the_full_one():
    planner = Planner()
    assert isinstance(planner.sequence, HgsSequence)
    assert isinstance(planner.sequence.schedule, DpSchedule)
    assert isinstance(planner.sequence.schedule.transfer, QlawTransfer)


def test_a_layer_given_explicitly_is_the_one_kept():
    sequence = cluster_stack()
    assert Planner(sequence).sequence is sequence


# ------------------------- initializing the stack ------------------------- #


def test_the_same_solver_can_be_moved_to_another_instance(odrc, iridium):
    planner = Planner(cluster_stack())
    first = planner.solve(odrc, verbose=False)
    second = planner.solve(iridium, verbose=False)

    assert len(first.plans) == odrc.num_chasers
    assert len(second.plans) == iridium.num_chasers
    assert sorted(d for p in second.plans for d in p.sequence[1:]) == list(
        iridium.targets
    )
    assert planner.sequence.schedule.problem is iridium


# --------------------------------- solving -------------------------------- #


def test_the_mission_covers_the_catalogue_once_with_one_plan_per_chaser(
    any_catalogue,
):
    mission = Planner(cluster_stack()).solve(any_catalogue, verbose=False)
    assert isinstance(mission, MissionPlan)
    assert len(mission.plans) == any_catalogue.num_chasers
    assert all(p.sequence[0] == DEPOT for p in mission.plans)
    visited = [d for p in mission.plans for d in p.sequence[1:]]
    assert sorted(visited) == list(any_catalogue.targets)


def test_the_search_stack_also_solves_end_to_end(odrc):
    config = HgsConfig.preset(2, seed=0, verbose=False)
    mission = Planner(HgsSequence(config=config)).solve(odrc, verbose=False)
    assert odrc.is_feasible(mission)
    assert sorted(d for p in mission.plans for d in p.sequence[1:]) == list(
        odrc.targets
    )


# ------------------------------- reporting -------------------------------- #


def test_a_verbose_run_reports_each_phase(odrc, capsys):
    Planner(cluster_stack()).solve(odrc)
    out = capsys.readouterr().out
    assert "Initializing layers" in out
    assert "Layers initialized" in out
    assert "Planning mission" in out
    assert "Mission planned" in out


def test_a_quiet_run_says_nothing(odrc, capsys):
    Planner(cluster_stack()).solve(odrc, verbose=False)
    assert capsys.readouterr().out == ""


# ------------------- a layer written by hand still composes ---------------- #


def test_a_sequence_layer_written_in_plain_python_drives_the_solver(odrc):
    """Nothing in `Planner` reaches past the interface it was handed."""

    class OneEach(SequenceLayer):
        """Give the first chasers one target apiece, in catalogue order."""

        def __init__(self):
            super().__init__()

        def initialize(self, problem):
            self.schedule.initialize(problem)
            self.problem = problem

        def evaluate(self):
            plans = [
                self.schedule.evaluate((DEPOT, d))
                for d in self.problem.targets
            ]
            while len(plans) < self.problem.num_chasers:
                plans.append(self.schedule.evaluate((DEPOT,)))
            return MissionPlan(plans)

    mission = Planner(OneEach()).solve(odrc, verbose=False)
    assert len(mission.plans) == odrc.num_chasers
    assert sorted(d for p in mission.plans for d in p.sequence[1:]) == list(
        odrc.targets
    )


def test_a_sequence_layer_written_with_the_hook_has_no_kernel(odrc):
    class OneEach(SequenceLayer):
        def _evaluate(self):
            plans = [
                self.schedule.evaluate((DEPOT, d))
                for d in self.problem.targets
            ]
            while len(plans) < self.problem.num_chasers:
                plans.append(self.schedule.evaluate((DEPOT,)))
            return MissionPlan(plans)

    sequence = OneEach()
    mission = Planner(sequence).solve(odrc, verbose=False)
    assert sequence.kernel() is None
    assert sorted(d for p in mission.plans for d in p.sequence[1:]) == list(
        odrc.targets
    )


def test_the_search_layer_publishes_no_kernel(odrc):
    sequence = HgsSequence(config=HgsConfig.preset(0, verbose=False))
    sequence.initialize(odrc)
    assert sequence.kernel() is None


def test_the_three_layer_interfaces_are_what_the_stack_is_checked_against():
    stack = cluster_stack()
    assert isinstance(stack, SequenceLayer)
    assert isinstance(stack.schedule, ScheduleLayer)
    assert isinstance(stack.schedule.transfer, TransferLayer)

"""Problem instance, chaser plans and feasibility."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mcrdv import ChaserPlan, DAY, MissionPlan, Problem
from mcrdv.orbit import KepState
from mcrdv.problem import DEPOT


def state(k=0):
    return KepState(7.0e6 + 1e3 * k, 1e-3, 1.2, 0.1 * k, 0.2, 0.3)


def problem(num_targets=5, num_chasers=3, max_load=4, max_time=100.0 * DAY):
    return Problem(
        [state(k) for k in range(num_targets + 1)],
        num_chasers,
        max_time,
        max_load,
    )


def plan(seq, arrive_last=0.0, cost=0.0):
    n = len(seq)
    costs, arrive = np.zeros(n), np.zeros(n)
    if n > 1:
        costs[-1], arrive[-1] = cost, arrive_last
    return ChaserPlan.of(seq, np.zeros(n), arrive, costs)


# ------------------------------- chaser plan ----------------------------- #


def test_plan_derives_cost_and_end_time():
    p = ChaserPlan.of(
        [0, 3, 7], [0.0, 10.0, 20.0], [0.0, 9.0, 19.0], [0.0, 2.0, 5.0]
    )
    assert p.cost == 7.0 and p.time == 19.0
    assert len(p) == 3 and p.load == 2 and not p.is_empty


def test_empty_plan_holds_only_its_starting_object():
    p = ChaserPlan.empty()
    assert (
        p.sequence == (DEPOT,) and len(p) == 1 and p.load == 0 and p.is_empty
    )
    assert p.cost == 0.0 and p.time == 0.0


def test_plan_takes_its_attributes_on_trust():
    # `of` deliberately does not check that the four attributes have equal
    # length. Keeping them in step is the caller's job, and every caller in the
    # package builds them together; a ragged plan is therefore accepted here
    # and only misbehaves where the short array is later indexed.
    p = ChaserPlan.of([0, 1], [0.0], [0.0, 1.0], [0.0, 1.0])
    assert len(p.sequence) == 2 and len(p.depart) == 1
    assert p.cost == 1.0 and p.time == 1.0


def test_sequence_is_a_tuple_so_it_can_key_a_cache():
    p = ChaserPlan.of([0, 3, 7], np.zeros(3), np.zeros(3), np.zeros(3))
    assert isinstance(p.sequence, tuple)
    assert {p.sequence: 1}[(0, 3, 7)] == 1


def test_mission_cost_is_the_sum_of_its_plans():
    m = MissionPlan(
        [plan([0, 1], cost=3.0), plan([0, 2], cost=4.0), ChaserPlan.empty()]
    )
    assert m.cost == 7.0 and len(m) == 3


# ------------------------------- overruns -------------------------------- #


def test_a_plan_reports_no_overrun_until_it_is_measured():
    p = ChaserPlan.of(
        [0, 1, 2], [0.0] * 3, [0.0, 10.0, 500.0], [0.0, 1.0, 1.0]
    )
    assert p.excess_load == 0 and p.excess_time == 0.0
    assert p.feasible  # nothing has told it about any limit yet


def test_refresh_measures_the_overruns_against_the_instance():
    prob = problem(num_targets=5, num_chasers=5, max_load=1, max_time=100.0)
    p = ChaserPlan.of(
        [0, 1, 2, 3], [0.0] * 4, [0.0, 10.0, 20.0, 250.0], [0.0, 1.0, 1.0, 1.0]
    )
    refreshed = prob.refresh(p)
    assert refreshed.excess_load == 2  # 3 targets for a limit of 1
    assert refreshed.excess_time == 150.0  # 250 s against a 100 s horizon
    assert not refreshed.feasible


def test_refresh_leaves_a_plan_alone_when_nothing_changes():
    prob = problem(max_load=4, max_time=100.0)
    p = prob.refresh(
        ChaserPlan.of([0, 1], [0.0, 0.0], [0.0, 10.0], [0.0, 1.0])
    )
    assert prob.refresh(p) is p


def test_refresh_clears_an_overrun_measured_elsewhere():
    strict = problem(num_chasers=5, max_load=1, max_time=10.0)
    roomy = problem(num_chasers=5, max_load=8, max_time=1000.0)
    p = strict.refresh(
        ChaserPlan.of([0, 1, 2], [0.0] * 3, [0.0, 5.0, 50.0], [0.0, 1.0, 1.0])
    )
    assert not p.feasible
    assert roomy.refresh(p).feasible


def test_a_plan_with_an_unflyable_leg_is_never_feasible():
    p = ChaserPlan.of([0, 1], [0.0, 0.0], [0.0, 1.0], [0.0, math.inf])
    assert p.cost == math.inf and not p.feasible


def test_cost_is_the_leg_sum_whether_feasible_or_not():
    prob = problem(num_chasers=5, max_load=1, max_time=10.0)
    p = prob.refresh(
        ChaserPlan.of([0, 1, 2], [0.0] * 3, [0.0, 5.0, 900.0], [0.0, 3.0, 4.0])
    )
    assert not p.feasible
    assert p.cost == 7.0  # the overrun does not touch the cost


def test_mission_feasibility_reads_its_plans():
    prob = problem(max_load=4, max_time=100.0)
    good = prob.refresh(
        ChaserPlan.of([0, 1], [0.0, 0.0], [0.0, 10.0], [0.0, 1.0])
    )
    bad = prob.refresh(
        ChaserPlan.of([0, 2], [0.0, 0.0], [0.0, 900.0], [0.0, 1.0])
    )
    assert MissionPlan([good, good]).feasible
    assert not MissionPlan([good, bad]).feasible


def test_the_plans_own_verdict_agrees_with_the_instance_away_from_the_boundary():
    prob = problem(max_load=4, max_time=100.0)
    for arrive, load in ((50.0, 2), (900.0, 2), (50.0, 6), (900.0, 6)):
        seq = list(range(load + 1))
        p = prob.refresh(
            ChaserPlan.of(
                seq,
                [0.0] * len(seq),
                [0.0] * (len(seq) - 1) + [arrive],
                [0.0] * len(seq),
            )
        )
        assert p.feasible == prob.is_feasible_plan(p)


# --------------------------------- problem -------------------------------- #


def test_depot_is_state_zero_and_target_indices_are_state_indices():
    p = problem(num_targets=5)
    assert DEPOT == 0
    assert p.num_targets == 5 and p.num_states == 6
    assert list(p.targets) == [1, 2, 3, 4, 5]


def test_states_pack_into_the_table_the_kernels_read():
    from mcrdv.orbit import pack

    p = problem(num_targets=3)
    table = pack(p.states)
    assert table.shape == (4, 8)
    assert table.dtype == np.float64 and table.flags["C_CONTIGUOUS"]
    assert table[2, 0] == p.states[2].a


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            dict(states=[state(0)], num_chasers=1, max_time=1.0, max_load=1),
            "length >= 2",
        ),
        (
            dict(
                states=[state(0), state(1)],
                num_chasers=0,
                max_time=1.0,
                max_load=1,
            ),
            "num_chasers",
        ),
        (
            dict(
                states=[state(0), state(1)],
                num_chasers=1,
                max_time=0.0,
                max_load=1,
            ),
            "positive",
        ),
        (
            dict(
                states=[state(0), state(1)],
                num_chasers=1,
                max_time=math.inf,
                max_load=1,
            ),
            "finite",
        ),
        (
            dict(
                states=[state(0), state(1)],
                num_chasers=1,
                max_time=1.0,
                max_load=0,
            ),
            "max_load",
        ),
        (
            dict(
                states=[state(k) for k in range(6)],
                num_chasers=2,
                max_time=1.0,
                max_load=2,
            ),
            "capacity",
        ),
    ],
)
def test_construction_rejects_impossible_instances(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Problem(**kwargs)


def test_capacity_check_is_exactly_tight():
    Problem(
        [state(k) for k in range(5)], 2, 1.0, 2
    )  # 2 x 2 == 4 targets, fine
    with pytest.raises(ValueError, match="capacity"):
        Problem([state(k) for k in range(6)], 2, 1.0, 2)


# ------------------------------- feasibility ------------------------------ #


def test_feasibility_checks_load_time_and_finiteness():
    p = problem(max_load=4, max_time=100.0)
    assert p.is_feasible_plan(plan([0, 1, 2], arrive_last=50.0))
    assert not p.is_feasible_plan(
        plan([0, 1, 2, 3, 4, 5], arrive_last=50.0)
    )  # load 5 > 4
    assert not p.is_feasible_plan(plan([0, 1], arrive_last=200.0))  # late
    assert not p.is_feasible_plan(plan([0, 1], cost=math.inf))  # unreachable


def test_time_feasibility_has_a_one_second_tolerance():
    p = problem(max_time=100.0)
    assert p.is_feasible_plan(plan([0, 1], arrive_last=100.9))
    assert not p.is_feasible_plan(plan([0, 1], arrive_last=101.5))


def test_mission_feasibility_requires_every_plan():
    p = problem(max_load=4, max_time=100.0)
    good, bad = plan([0, 1], arrive_last=10.0), plan([0, 2], arrive_last=500.0)
    assert p.is_feasible(MissionPlan([good, good]))
    assert not p.is_feasible(MissionPlan([good, bad]))


def test_str_reports_the_operational_limits():
    text = str(
        problem(num_targets=5, num_chasers=3, max_load=4, max_time=100.0 * DAY)
    )
    assert "num_chasers: 3" in text and "num_targets: 5" in text
    assert "100.00 days" in text and "4 targets" in text


# ------------------------------- rendering -------------------------------- #


def test_str_reports_the_headline_numbers():
    prob = problem(
        num_targets=5, num_chasers=3, max_load=4, max_time=100.0 * DAY
    )
    mission = MissionPlan(
        [
            prob.refresh(
                ChaserPlan.of(
                    [0, 2, 5],
                    [0.0, 1.0, 2.0],
                    [0.0, 3.0, 4.0],
                    [0.0, 7.0, 8.0],
                )
            )
        ]
        + [ChaserPlan.empty()] * 2
    )
    text = str(mission)
    assert "Mission plan" in text
    assert "Total cost   : 15.0" in text
    assert "Chasers used : 1/3" in text
    assert "Feasible     : True" in text
    assert text.splitlines()[-1] == "=" * 58


def test_str_survives_an_all_idle_mission():
    text = str(MissionPlan([ChaserPlan.empty()] * 3))
    assert "Chasers used : 0/3" in text


def test_str_reports_an_infeasible_plan():
    prob = problem(num_targets=5, num_chasers=5, max_load=1, max_time=10.0)
    bad = prob.refresh(
        ChaserPlan.of([0, 1, 2], [0.0] * 3, [0.0, 5.0, 900.0], [0.0, 1.0, 1.0])
    )
    assert "Feasible     : False" in str(MissionPlan([bad]))

"""What a produced mission plan looks like, as numbers.

The other experiments measure how well the planner does. This one asks what it
actually decided: how the work is shared between chasers, how long they idle,
how the cost is spread over legs, and how far each chaser's targets are spread
in right ascension -- which is the geometric quantity the whole problem turns
on, since a plane change is what a transfer mostly pays for.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np

from mcrdv import MissionPlan, Problem
from mcrdv.orbit import TAU
from mcrdv.planner import TransferLayer
from mcrdv.problem import DEPOT, ChaserPlan
from mcrdv.sequence.hgs import mean_raan

# The Keplerian elements of a state, each as the attribute holding it and the
# attribute holding its drift rate, or `None` where the model holds it fixed.
# A figure drawn against one of them names it, and everything below reads it
# from here: adding an element makes it available everywhere at once.
#
# Only the right ascension and the argument of periapsis drift under the
# secular J2 model this package propagates with. The other four are constants
# of that model, the true anomaly included -- the Q-law estimate leaves the
# fast angle out, so nothing in a plan depends on it. Their tracks are
# therefore flat, which is a property of the model and not of the orbits.
ELEMENTS = {
    "a": ("a", None),
    "e": ("e", None),
    "i": ("i", None),
    "raan": ("r", "dr"),
    "argp": ("o", "do"),
    "anomaly": ("t", None),
}


def waiting(plan: ChaserPlan) -> np.ndarray:
    """Seconds spent waiting at each visited state before leaving it.

    A chaser may loiter to catch a cheaper departure window, so the wait is the
    schedule layer's decision made visible: `depart - arrive` at every state it
    leaves. The final state is dropped, because nothing departs from it.
    """
    if len(plan) < 2:
        return np.zeros(0)
    return np.maximum(plan.depart[:-1] - plan.arrive[:-1], 0.0)


def raan_spread(plan: ChaserPlan, problem: Problem) -> float:
    """Angular spread of a plan's targets about their circular mean [rad].

    Reuses the search's own `mean_raan`, which reads each right ascension at the
    time that target is visited rather than at the catalogue epoch -- the orbits
    drift, and by hundreds of days apart the difference is the whole story.
    """
    if plan.load == 0:
        return 0.0
    centre = mean_raan(SimpleNamespace(problem=problem), plan)
    states = problem.states
    worst = 0.0
    for target, when in zip(plan.sequence[1:], plan.depart[1:]):
        angle = states[target].r + states[target].dr * when
        delta = abs(angle - centre) % TAU
        worst = max(worst, min(delta, TAU - delta))
    return worst


def per_chaser(mission: MissionPlan, problem: Problem) -> List[Dict[str, Any]]:
    """One row per chaser, idle ones included so the table stays rectangular."""
    rows = []
    for index, plan in enumerate(mission.plans):
        pauses = waiting(plan)
        rows.append(
            {
                "chaser": index,
                "load": plan.load,
                "cost": plan.cost,
                "time": plan.time,
                "feasible": plan.feasible,
                "excess_load": plan.excess_load,
                "excess_time": plan.excess_time,
                "wait_total": float(pauses.sum()),
                "wait_max": float(pauses.max()) if pauses.size else 0.0,
                "leg_cost_mean": (
                    float(plan.costs[1:].mean()) if plan.load else 0.0
                ),
                "leg_cost_max": (
                    float(plan.costs[1:].max()) if plan.load else 0.0
                ),
                "raan_spread": raan_spread(plan, problem),
            }
        )
    return rows


def mission_metrics(mission: MissionPlan, problem: Problem) -> Dict[str, Any]:
    """Aggregate shape of a mission plan, for one row of a results table."""
    rows = per_chaser(mission, problem)
    active = [row for row in rows if row["load"] > 0]
    loads = np.array([row["load"] for row in active], dtype=np.float64)
    costs = np.array([row["cost"] for row in active], dtype=np.float64)

    if not active:
        return {"active_chasers": 0, "idle_chasers": len(rows)}

    return {
        "active_chasers": len(active),
        "idle_chasers": len(rows) - len(active),
        "load_min": int(loads.min()),
        "load_max": int(loads.max()),
        "load_mean": float(loads.mean()),
        # How unevenly the work is shared: 0 when every active chaser carries
        # the same load, growing as one of them takes on more than its share.
        "load_imbalance": float(loads.max() / loads.mean() - 1.0),
        "cost_min": float(costs.min()),
        "cost_max": float(costs.max()),
        "cost_mean": float(costs.mean()),
        "makespan": float(max(row["time"] for row in rows)),
        "wait_total": float(sum(row["wait_total"] for row in rows)),
        "raan_spread_mean": float(
            np.mean([row["raan_spread"] for row in active])
        ),
    }


def leg_table(
    mission: MissionPlan, delta_v: TransferLayer
) -> List[Dict[str, Any]]:
    """Every leg of a mission, one row each, re-priced on a common scale.

    A plan's own `costs` are whatever its cost oracle weighed, so two plans
    built under different oracles cannot be compared leg by leg on them. Here
    each leg is re-read with `delta_v` -- an initialized transfer layer whose
    cost *is* the delta-v -- at the departure time the plan chose, which gives
    every plan the same physical scale whatever produced it.

    The row is the plan seen at its finest grain, and most of what a plan's
    structure amounts to is a reduction of it: what a transfer costs, how long
    a chaser sat still before starting one, where in a tour a target falls, and
    when in the mission that happened. Times are in seconds.
    """
    rows = []
    for index, plan in enumerate(mission.plans):
        if plan.load == 0:
            continue
        for k in range(len(plan) - 1):
            ready = float(plan.arrive[k])
            leaves = float(plan.depart[k])
            dt, dv = delta_v.evaluate(
                plan.sequence[k], plan.sequence[k + 1], leaves
            )
            rows.append(
                {
                    "chaser": index,
                    # Which visit of this chaser's tour the leg ends at: 1 is
                    # the first target it collects.
                    "position": k + 1,
                    "origin": int(plan.sequence[k]),
                    "target": int(plan.sequence[k + 1]),
                    "wait": max(leaves - ready, 0.0),
                    "depart": leaves,
                    "arrive": float(plan.arrive[k + 1]),
                    "flight": float(dt),
                    "delta_v": float(dv),
                }
            )
    return rows


def fuel_and_time(
    mission: MissionPlan, delta_v: TransferLayer
) -> Dict[str, Any]:
    """A plan's delta-v and its timeline, whatever objective produced it.

    The aggregates of :func:`leg_table`, which is where the common scale is
    explained. Times are in seconds.
    """
    rows = leg_table(mission, delta_v)
    times = np.array([row["arrive"] for row in rows], dtype=np.float64)
    return {
        "delta_v": float(sum(row["delta_v"] for row in rows)),
        "flight_time": float(sum(row["flight"] for row in rows)),
        "wait_time": float(sum(row["wait"] for row in rows)),
        # When the targets are collected: the quantity a time-aware oracle
        # prices, and what "removing debris sooner" means for the mission.
        "collection_mean": float(times.mean()) if times.size else 0.0,
        "collection_max": float(times.max()) if times.size else 0.0,
    }


def element_at(
    problem: Problem, state: int, when: float, element: str = "raan"
) -> float:
    """One state's `element` at time `when`, J2 drift included.

    Angles are in radians and the semi-major axis in metres, as the state
    holds them; what an axis makes of that is the figure's business.
    """
    attribute, rate = ELEMENTS[element]
    orbit = problem.states[state]
    value = getattr(orbit, attribute)
    return value + getattr(orbit, rate) * when if rate else value


def tracks(problem: Problem, element: str = "raan") -> Dict[str, Any]:
    """Enough to redraw every state's `element` over the mission.

    An element either drifts linearly under the secular J2 model or does not
    drift at all, so the whole history of a state is two numbers. Storing those
    rather than a sampled curve keeps a record small and lets a figure choose
    its own resolution; an element the model holds fixed simply has a rate of
    zero, which needs no special case anywhere.
    """
    attribute, rate = ELEMENTS[element]
    states = problem.states
    return {
        "start": np.array([getattr(s, attribute) for s in states]),
        "rate": (
            np.array([getattr(s, rate) for s in states])
            if rate
            else np.zeros(len(states))
        ),
        "horizon": float(problem.max_time),
    }


def visits(mission: MissionPlan, problem: Problem) -> List[Dict[str, Any]]:
    """Where each chaser is, and when, from the swarm state to its last target.

    A track begins on the swarm state at the mission start, which is the one
    place every chaser shares, and stays there until that chaser departs. That
    orbit drifts like any other, so a chaser leaving later leaves from a
    different angle -- which is why a figure that opened on the first target
    made the chasers look as though they had started apart.

    After that there is one point per target, at the time the chaser arrives:
    the arrival is what matters rather than the departure, being the moment the
    chaser and the target are actually in the same place. Every element is
    recorded, so one solve serves a figure drawn against any of them.
    """
    out = []
    for index, plan in enumerate(mission.plans):
        if plan.load == 0:
            continue
        # The swarm state at the mission start, and again at the moment this
        # chaser leaves it; between the two the chaser simply drifts with it.
        states = [DEPOT]
        times = [0.0]
        if plan.depart[0] > 0.0:
            states.append(DEPOT)
            times.append(float(plan.depart[0]))
        states += list(plan.sequence[1:])
        times += [float(t) for t in plan.arrive[1:]]

        row = {"chaser": index, "states": states, "time": times}
        for element in ELEMENTS:
            row[element] = [
                element_at(problem, s, t, element)
                for s, t in zip(states, times)
            ]
        out.append(row)
    return out


def timeline(mission: MissionPlan) -> List[Dict[str, Any]]:
    """Each chaser's mission as intervals: when it waits, when it transfers.

    A chaser alternates between sitting on the orbit it has just reached, where
    it waits for a departure window worth taking, and flying to its next
    target. The two together are what the schedule layer decided, and they are
    a plan seen in time rather than in cost -- which chasers are busy, which
    are loitering, and when each of them is finished.
    """
    out = []
    for index, plan in enumerate(mission.plans):
        if plan.load == 0:
            continue
        waits, transfers = [], []
        for leg in range(len(plan) - 1):
            ready = float(plan.arrive[leg])
            leaves = float(plan.depart[leg])
            if leaves > ready:
                waits.append([ready, leaves])
            transfers.append([leaves, float(plan.arrive[leg + 1])])
        out.append(
            {
                "chaser": index,
                "wait": waits,
                "transfer": transfers,
                "end": float(plan.time),
            }
        )
    return out


def legs(mission: MissionPlan) -> np.ndarray:
    """Every leg cost in the mission, for a distribution plot."""
    parts = [plan.costs[1:] for plan in mission.plans if plan.load > 0]
    return np.concatenate(parts) if parts else np.zeros(0)

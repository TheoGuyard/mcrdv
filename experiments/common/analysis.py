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
from mcrdv.problem import ChaserPlan
from mcrdv.sequence.hgs import mean_raan


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


def raan_at(problem: Problem, state: int, when: float) -> float:
    """Right ascension of one state at time `when` [rad], J2 drift included."""
    orbit = problem.states[state]
    return orbit.r + orbit.dr * when


def raan_tracks(problem: Problem) -> Dict[str, Any]:
    """Enough to redraw every state's right ascension over the mission.

    A right ascension drifts linearly under the secular J2 model, so the whole
    history of a state is two numbers. Storing those rather than a sampled curve
    keeps a record small and lets a figure choose its own resolution.
    """
    return {
        "r0": np.array([s.r for s in problem.states]),
        "dr": np.array([s.dr for s in problem.states]),
        "horizon": float(problem.max_time),
    }


def visits(mission: MissionPlan, problem: Problem) -> List[Dict[str, Any]]:
    """When each chaser reaches each of its targets, and where in RAAN.

    The arrival time is what matters here rather than the departure: it is the
    moment the chaser and the target are actually at the same place, which is
    the point the assignment figure marks.
    """
    out = []
    for index, plan in enumerate(mission.plans):
        if plan.load == 0:
            continue
        targets = list(plan.sequence[1:])
        times = [float(t) for t in plan.arrive[1:]]
        out.append(
            {
                "chaser": index,
                "targets": targets,
                "time": times,
                "raan": [
                    raan_at(problem, d, t) for d, t in zip(targets, times)
                ],
            }
        )
    return out


def legs(mission: MissionPlan) -> np.ndarray:
    """Every leg cost in the mission, for a distribution plot."""
    parts = [plan.costs[1:] for plan in mission.plans if plan.load > 0]
    return np.concatenate(parts) if parts else np.zeros(0)

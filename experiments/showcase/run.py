#!/usr/bin/env python3
"""The structure of a produced mission plan: who goes where and when, how the
legs are spread in time and cost, and every orbit's elements over the mission.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcrdv import MissionPlan, Problem  # noqa: E402
from mcrdv.problem import DEPOT  # noqa: E402

from _common import runner, planners  # noqa: E402

# Keplerian elements: the attribute holding each one and the attribute holding
# its secular J2 drift rate (`None` for the elements the model holds fixed).
ELEMENTS = {
    "a": ("a", None),
    "e": ("e", None),
    "i": ("i", None),
    "raan": ("r", "dr"),
    "argp": ("o", "do"),
    "anomaly": ("t", None),
}


def element_at(
    problem: Problem, state: int, when: float, element: str = "raan"
) -> float:
    """One state's `element` at time `when`, J2 drift included."""
    attribute, rate = ELEMENTS[element]
    orbit = problem.states[state]
    value = getattr(orbit, attribute)
    return value + getattr(orbit, rate) * when if rate else value


def tracks(problem: Problem, element: str = "raan") -> Dict[str, Any]:
    """Every state's `element` over the mission, as a start value and a
    linear drift rate per state."""
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
    """Where each active chaser is, and when: on the swarm's orbit until it
    departs, then at each target on arrival, with every element recorded."""
    out = []
    for index, plan in enumerate(mission.plans):
        if plan.load == 0:
            continue
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
    """Each active chaser's mission as `[start, end]` intervals of waiting
    and of transferring."""
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
    """Every leg cost in the mission."""
    parts = [plan.costs[1:] for plan in mission.plans if plan.load > 0]
    return np.concatenate(parts) if parts else np.zeros(0)


def execute(config: dict) -> dict:
    result = planners.solve(config)
    problem = planners.problem(config)
    result.update(planners.mission_metrics(result["mission"], problem))
    result["chasers"] = planners.per_chaser(result["mission"], problem)
    # Every element's track is kept, so which one a figure draws stays a
    # plot.yaml decision.
    result["tracks"] = {
        element: tracks(problem, element) for element in ELEMENTS
    }
    result["visits"] = visits(result["mission"], problem)
    result["timeline"] = timeline(result["mission"])
    result["legs"] = legs(result["mission"])
    return result


if __name__ == "__main__":
    raise SystemExit(runner.main(__file__, execute, warmup=planners.warm_up))

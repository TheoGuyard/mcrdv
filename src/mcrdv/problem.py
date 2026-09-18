"""Problem and solution representation."""

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Self, Sequence, Tuple

import numpy as np

from .orbit import DAY, KepState

# Conventional index of the swarm in a problem state list
DEPOT = 0


@dataclass(frozen=True, slots=True, eq=False)
class ChaserPlan:
    """The mission plan of a single chaser."""

    # State indices visited ordered as [depot, target_1, ..., target_N].
    sequence: Tuple[int, ...]
    # Departure time from each visited state [s].
    depart: np.ndarray
    # Arrival time at each visited state [s].
    arrive: np.ndarray
    # Cost of the leg arriving at each visited state.
    costs: np.ndarray
    # Total cost of the plan.
    cost: float
    # Total time of the plan [s].
    time: float
    # Excess in the load limit relative to a problem instance [targets].
    excess_load: int = 0
    # Excess in the time limit relative to a problem instance [s].
    excess_time: float = 0.0
    # Number of targets collected.
    load: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "load", len(self.sequence) - 1)

    def __len__(self) -> int:
        """Number of states visited, including the depot one."""
        return len(self.sequence)

    @classmethod
    def of(
        cls,
        sequence: Sequence[int],
        depart: np.ndarray,
        arrive: np.ndarray,
        costs: np.ndarray,
        excess_load: int = 0,
        excess_time: float = 0.0,
    ) -> Self:
        """Build a chaser plan from attributes."""
        costs = np.asarray(costs, dtype=np.float64)
        depart = np.asarray(depart, dtype=np.float64)
        arrive = np.asarray(arrive, dtype=np.float64)
        return cls(
            tuple(sequence),
            depart,
            arrive,
            costs,
            float(costs.sum()),
            float(arrive[-1]) if len(arrive) else 0.0,
            excess_load,
            excess_time,
        )

    @classmethod
    def empty(cls, start: int = DEPOT) -> Self:
        """Empty chaser plan with only the depot."""
        return cls((start,), np.zeros(1), np.zeros(1), np.zeros(1), 0.0, 0.0)

    @property
    def is_empty(self) -> bool:
        """Whether the chaser plan collects no target."""
        return self.load == 0

    @property
    def feasible(self) -> bool:
        """Whether the chaser plan is feasible."""
        return (
            self.excess_load == 0
            and self.excess_time == 0.0
            and math.isfinite(self.cost)
        )


@dataclass(frozen=True, slots=True, eq=False)
class MissionPlan:
    """Mission plan containing all individual chaser plans."""

    # Individual chaser plans.
    plans: Tuple[ChaserPlan, ...]

    def __init__(self, plans: Sequence[ChaserPlan]) -> None:
        object.__setattr__(self, "plans", tuple(plans))

    def __len__(self) -> int:
        return len(self.plans)

    def __str__(self) -> str:
        wide = 58
        side = "=" * (wide // 2 - 7)
        used = sum(1 for p in self.plans if p.load > 0)

        lines = [
            f"{side} Mission plan {side}",
            f"Feasible     : {self.feasible}",
            f"Total cost   : {self.cost:.1f}",
            f"Chasers used : {used}/{len(self.plans)}",
        ]

        for p, plan in enumerate(self.plans):
            lines.append("")
            lines.append(
                f"Chaser {p} (feasible: {plan.feasible}, cost: {plan.cost:.1f})"
            )
            lines.append(
                f"  {'src':<7}{'dst':<9}{'arrive':>13}{'depart':>13}{'leg cost':>14}"
            )

            last = len(plan) - 1
            for i in range(len(plan)):
                if i == last:
                    src, dst = str(plan.sequence[i]), "--"
                    arrive, depart = f"{plan.arrive[i]:.1f}", "--"
                elif i == 0:
                    src, dst = str(plan.sequence[i]), str(plan.sequence[1])
                    arrive, depart = "--", f"{plan.depart[0]:.1f}"
                else:
                    src, dst = str(plan.sequence[i]), str(plan.sequence[i + 1])
                    arrive, depart = (
                        f"{plan.arrive[i]:.1f}",
                        f"{plan.depart[i]:.1f}",
                    )
                lines.append(
                    f"  {src:<7}{dst:<9}{arrive:>13}{depart:>13}{plan.costs[i]:>14.2f}"
                )

        lines.append("=" * wide)
        return "\n".join(lines)

    @property
    def cost(self) -> float:
        """Total cost of the mission."""
        return sum(p.cost for p in self.plans)

    @property
    def feasible(self) -> bool:
        """Whether every plan overruns neither limit."""
        return all(p.feasible for p in self.plans)

    def save(self, path: Path, dec: int = 6) -> None:
        """Write a mission plan to file, with `dec` decimals precision."""
        head = "chaser,state,depart,arrive,cost"
        rows = [head]
        for p, plan in enumerate(self.plans):
            for s, d, a, c in zip(
                plan.sequence, plan.depart, plan.arrive, plan.costs
            ):
                rows.append(f"{p},{s},{d:.{dec}f},{a:.{dec}f},{c:.{dec}f}")
        with open(path, "w", encoding="utf-8") as file:
            file.write("\n".join(rows) + "\n")


@dataclass(frozen=True, slots=True, eq=False)
class Problem:
    """Problem instance."""

    # Problem states `[depot, target_1, ..., target_n]`.
    states: Tuple[KepState, ...]
    # Number of chasers available in the depot.
    num_chasers: int
    # Maximum mission time per chaser [s].
    max_time: float
    # Maximum mission load per chaser [targets].
    max_load: int

    def __post_init__(self) -> None:
        if len(self.states) <= 1:
            raise ValueError("'states' must be of length >= 2")
        if self.num_chasers <= 0:
            raise ValueError("'num_chasers' must be positive")
        if self.max_time <= 0.0:
            raise ValueError("'max_time' must be positive")
        if not math.isfinite(self.max_time):
            raise ValueError("'max_time' must be finite")
        if self.max_load <= 0:
            raise ValueError("'max_load' must be positive")
        if self.num_chasers * self.max_load < self.num_targets:
            raise ValueError("insufficient capacity over the swarm")

    def __str__(self) -> str:
        return (
            "Problem\n"
            f"  num_chasers: {self.num_chasers}\n"
            f"  num_targets: {self.num_targets}\n"
            f"  max_time   : {self.max_time / DAY:.2f} days\n"
            f"  max_load   : {self.max_load} targets"
        )

    @property
    def num_states(self) -> int:
        """Number of states, the depot included."""
        return len(self.states)

    @property
    def num_targets(self) -> int:
        """Number of targets to collect (depot excluded)."""
        return len(self.states) - 1

    @property
    def targets(self) -> range:
        """Target indices."""
        return range(1, len(self.states))

    def refresh(self, plan: ChaserPlan) -> ChaserPlan:
        """Refresh a `plan` excess values."""
        excess_load = max(plan.load - self.max_load, 0)
        excess_time = max(plan.time - self.max_time, 0.0)
        if excess_load == plan.excess_load and excess_time == plan.excess_time:
            return plan
        return replace(plan, excess_load=excess_load, excess_time=excess_time)

    def is_feasible(self, mission: MissionPlan, **kwargs) -> bool:
        """Whether all chaser plan respect operational limits."""
        return all(self.is_feasible_plan(p, **kwargs) for p in mission.plans)

    def is_feasible_plan(
        self, plan: ChaserPlan, time_tol: float = 1.0
    ) -> bool:
        """Whether a chaser plan respects operational limits."""
        return (
            plan.load <= self.max_load
            and plan.time <= self.max_time + time_tol
            and math.isfinite(plan.cost)
        )

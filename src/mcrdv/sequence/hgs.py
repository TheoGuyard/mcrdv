"""Hybrid genetic search: the sequence layer and its parts.

The population, the crossover operators and the generation are plain Python.
The two loops that dominate the running time are compiled: the local search,
and the split of a giant tour into chaser plans. They price candidate plans
through the schedule kernel directly, and remember the cost and time of every
plan priced in a compiled cache rather than the plan itself.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, NamedTuple, Optional, Self, Sequence

import numpy as np
from numba import njit, types
from numba.typed import Dict

from ..kernel import JIT, KERNEL, bind, schedule_evaluate
from ..orbit import TAU
from ..problem import DEPOT, ChaserPlan, MissionPlan, Problem
from ..planner import ScheduleLayer, SequenceLayer

__all__ = [
    "pair_lb",
    "Crossover",
    "Generation",
    "HgsConfig",
    "Ordering",
    "Context",
    "Individual",
    "Metric",
    "broken_pair_distance",
    "HgsSequence",
    "LogEntry",
    "Trace",
    "Population",
    "Subpopulation",
    "Search",
]


# ------------------------------ Configuration ------------------------------ #
# Hybrid genetic search configuration.


class Ordering(str, Enum):
    """Giant-tour ordering strategy."""

    # Keep the plans in the order they were built.
    INDEX = "index"
    # By circular mean RAAN, read at the times the targets are visited.
    RAAN = "raan"
    # Chain the plans greedily by the cost of the transfer joining them.
    NEAREST = "nearest"


class Crossover(str, Enum):
    """Crossover operator."""

    # Order crossover on the giant tour.
    OX = "ox"
    # Edge-recombination crossover on the giant tour.
    ERX = "erx"
    # Selective exchange of whole chaser plans.
    SREX = "srex"


class Generation(str, Enum):
    """Initial population generation strategy."""

    # Nearest-neighbor construction on the cheapest reachable targets.
    GREEDY = "greedy"
    # Random assignment of targets to chasers.
    RANDOM = "random"


# Preset levels, deepest search last:
# `(log_iter, max_iter, max_nimp, pop_init, pop_min, pop_max, nb_close, nb_elite)`.
_PRESETS = (
    (1, 0, 0, 1, 2, 2, 1, 1),
    (1, 0, 0, 5, 2, 3, 1, 1),
    (1, 50, 10, 6, 3, 6, 1, 1),
    (1, 500, 100, 10, 5, 10, 2, 2),
    (10, 5_000, 500, 20, 10, 20, 2, 3),
    (10, 50_000, 5_000, 24, 12, 32, 3, 4),
    (100, 200_000, 10_000, 50, 25, 65, 3, 8),
)


@dataclass(frozen=True, slots=True)
class HgsConfig:
    """Hybrid genetic search configuration.

    Immutable: derive a variant with `dataclasses.replace(config, seed=7)`.
    """

    # -- global ----------------------------------------------------------- #
    # Random seed.
    seed: int = 0
    # Whether to print a row of the search log as it runs.
    verbose: bool = True
    # Whether to save iteration logs.
    log_save: bool = False
    # Iteration step between logs.
    log_iter: int = 10
    # Maximum number of iterations.
    max_iter: int = 5_000
    # Maximum number of iterations without improvement.
    max_nimp: int = 500
    # Maximum time budget in seconds.
    max_time: float = math.inf

    # -- generation -------------------------------------------------------- #
    # Strategy used to build the initial population.
    generation: Generation = Generation.GREEDY
    # Number of individuals in the initial population.
    pop_init: int = 20

    # -- population -------------------------------------------------------- #
    # Minimum population size.
    pop_min: int = 10
    # Maximum population size.
    pop_max: int = 20
    # Number of nearest individuals averaged in the diversity measure.
    nb_close: int = 2
    # Number of elite individuals with no diversity measure.
    nb_elite: int = 3
    # Number of iterations between penalty adaptations.
    adapt_iter: int = 10
    # Distance below which two individuals count as clones.
    clone_eps: float = 1e-6

    # -- crossover --------------------------------------------------------- #
    # Number of candidates drawn in each parent tournament.
    nb_pick: int = 4
    # Infeasibility factor during the split of a giant tour.
    threshold_factor: float = 1.5
    # Giant-tour ordering.
    ordering: Ordering = Ordering.NEAREST
    # Crossover operator.
    crossover: Crossover = Crossover.OX

    # -- local search ------------------------------------------------------ #
    # Size of the neighborhood for local search moves.
    nb_neighbors: int = 10

    # -- penalties --------------------------------------------------------- #
    # Multiplier applied to increase a penalty.
    penalty_increase: float = 2.0
    # Multiplier applied to decrease a penalty.
    penalty_decrease: float = 0.5
    # Lower bound on the penalties range.
    penalty_min: float = 1e-9
    # Upper bound on the penalties range.
    penalty_max: float = 1e9
    # Target ratio of feasible individuals in the population.
    target_ratio: float = 0.9

    def __post_init__(self) -> None:
        if self.pop_min > self.pop_max:
            raise ValueError("'pop_min' must not exceed 'pop_max'")
        if self.pop_max < 1:
            raise ValueError("'pop_max' must be positive")
        if self.nb_neighbors < 0:
            raise ValueError("'nb_neighbors' must be non-negative")
        if not 0.0 <= self.target_ratio <= 1.0:
            raise ValueError("'target_ratio' must lie in [0, 1]")
        if self.penalty_min > self.penalty_max:
            raise ValueError("'penalty_min' must not exceed 'penalty_max'")
        # Accept the plain strings too, so a config can come from a file.
        object.__setattr__(self, "generation", Generation(self.generation))
        object.__setattr__(self, "ordering", Ordering(self.ordering))
        object.__setattr__(self, "crossover", Crossover(self.crossover))

    @classmethod
    def preset(cls, level: int, **overrides) -> Self:
        """Preset configuration, from 0 to 6; higher searches deeper.

        Any keyword given overrides the preset's own value for that field.
        """
        if not 0 <= level < len(_PRESETS):
            raise ValueError(
                f"HGS preset level must be between 0 and {len(_PRESETS) - 1}"
            )
        log, iters, nimp, init, lo, hi, close, elite = _PRESETS[level]
        fields = dict(
            log_iter=log,
            max_iter=iters,
            max_nimp=nimp,
            pop_init=init,
            pop_min=lo,
            pop_max=hi,
            nb_close=close,
            nb_elite=elite,
        )
        fields.update(overrides)  # an explicit override always wins
        return cls(**fields)


# ------------------------------- Pair bounds ------------------------------- #
# The per-leg lower bounds the local search screens candidate moves against.


def pair_lb(
    schedule: ScheduleLayer,
    problem: Problem,
    workers: int | None = None,
) -> np.ndarray:
    """The `(n, n)` table of per-leg lower bounds, `table[i, j]` for leg `i->j`.

    The local search rejects the overwhelming majority of candidate moves on
    this table alone -- a handful of array reads where pricing the move would
    cost a dynamic program. That is what makes a search over thousands of
    targets terminate, and it is why the table is built once, up front.

    It is derived from `ScheduleLayer.evaluate_lb` and nothing else, so any
    schedule layer supplies one: the table is a search device, not something a
    scheduler owes its callers. Being a sum over legs, it is additive by
    construction, which is what lets a candidate sequence be screened by
    summing its legs.

    Rows are independent and touch disjoint cells everywhere below, so the fill
    is spread over threads. The shipped layers sample departure windows in
    compiled code that releases the GIL, so that is most of the work running in
    parallel; a pure-Python layer simply gets no speed-up from it.
    """
    n = problem.num_states
    table = np.empty((n, n))
    evaluate_lb = schedule.evaluate_lb

    def fill(i: int) -> None:
        row = table[i]
        for j in range(n):
            row[j] = evaluate_lb((i, j))

    if workers == 1:
        for i in range(n):
            fill(i)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(fill, range(n)):
                pass
    return table


# ------------------------------- Individuals ------------------------------- #
# Individuals of the population, and the metric that ranks them.


@njit(**JIT)
def broken_pair_distance(a_pred, a_succ, b_pred, b_succ, num_targets):
    """Broken-pair distance between two individuals.

    Counts the targets whose neighbours differ between the two solutions, in
    either direction, plus those that start a plan in one and not in the other.
    Normalised by the catalogue size, so it reads as a fraction.
    """
    if num_targets == 0:
        return 0.0
    count = 0
    for d in range(1, num_targets + 1):
        asucc = a_succ[d]
        bsucc = b_succ[d]
        apred = a_pred[d]
        bpred = b_pred[d]
        if asucc != bsucc and asucc != bpred:
            count += 1
        if apred == DEPOT and bpred != DEPOT and bsucc != DEPOT:
            count += 1
    return count / num_targets


class Individual:
    """Individual in the population, as a candidate mission plan."""

    __slots__ = (
        "plans",
        "cost",
        "violation",
        "feasible",
        "active",
        "pred",
        "succ",
    )

    def __init__(self, problem: Problem, plans: Sequence[ChaserPlan]) -> None:
        n = problem.num_states
        # One plan per chaser, some possibly idle.
        self.plans: List[ChaserPlan] = list(plans)
        # Mission cost, the sum of the chaser plan costs.
        self.cost = 0.0
        # Total excess over the load limit, and over the duration limit.
        self.violation = (0.0, 0.0)
        # Whether every plan satisfies both operational limits.
        self.feasible = True
        # Number of chasers that actually collect something.
        self.active = 0
        # For each target, the object visited just before it, DEPOT for none.
        self.pred = np.zeros(n, dtype=np.int64)
        # For each target, the object visited just after it, DEPOT for none.
        self.succ = np.zeros(n, dtype=np.int64)
        self.refresh(problem)

    def refresh(self, problem: Problem) -> None:
        """Recompute the derived quantities after the plans were replaced."""
        n = problem.num_states

        pred = [DEPOT] * n
        succ = [DEPOT] * n
        cost = 0.0
        load_excess = 0.0
        time_excess = 0.0
        active = 0
        feasible = True

        for plan in self.plans:
            cost += plan.cost
            load_excess += plan.excess_load
            time_excess += plan.excess_time
            feasible &= problem.is_feasible_plan(plan)

            seq = plan.sequence
            if len(seq) > 1:
                active += 1
                last = len(seq) - 1
                for i in range(1, len(seq)):
                    d = seq[i]
                    pred[d] = DEPOT if i == 1 else seq[i - 1]
                    succ[d] = DEPOT if i == last else seq[i + 1]

        self.cost = cost
        self.violation = (load_excess, time_excess)
        self.feasible = feasible
        self.active = active
        self.pred = np.asarray(pred, dtype=np.int64)
        self.succ = np.asarray(succ, dtype=np.int64)

    def distance(self, other: Self, num_targets: int) -> float:
        """Broken-pair distance to another individual."""
        return broken_pair_distance(
            self.pred, self.succ, other.pred, other.succ, num_targets
        )

    def copy(self) -> Self:
        """A shallow copy: the plans are immutable, so they are shared."""
        clone = object.__new__(Individual)
        clone.plans = list(self.plans)
        clone.cost = self.cost
        clone.violation = self.violation
        clone.feasible = self.feasible
        clone.active = self.active
        clone.pred = self.pred
        clone.succ = self.succ
        return clone

    def __repr__(self) -> str:
        return (
            f"Individual(cost={self.cost:.1f}, feasible={self.feasible}, "
            f"active={self.active}, violation={self.violation})"
        )


class Metric:
    """Unified metric to evaluate both feasible and infeasible individuals.

    A plan that overruns a limit is not discarded: it is charged for the
    excess. The two penalties start scaled to the instance and are retuned
    during the search, so the population keeps straddling the feasible
    boundary, which is where the good solutions sit.
    """

    __slots__ = ("penalties", "config")

    def __init__(self, problem: Problem, schedule, config) -> None:
        scale = max(
            sum(schedule.evaluate_lb((DEPOT, d)) for d in problem.targets), 1.0
        )
        clamp = lambda x: min(max(x, config.penalty_min), config.penalty_max)
        # Constraint penalties, for the load limit and the time limit.
        self.penalties = (
            clamp(scale / problem.max_load),
            clamp(scale / problem.max_time),
        )
        self.config = config

    def plan_value(self, plan: ChaserPlan) -> float:
        """Penalized cost of a single chaser plan.

        The plan already knows how far it overruns each limit, so this is three
        multiplications and no comparison -- which matters, because it runs
        millions of times per search iteration.
        """
        return (
            plan.cost
            + self.penalties[0] * plan.excess_load
            + self.penalties[1] * plan.excess_time
        )

    def value(self, individual: Individual) -> float:
        """Penalized cost of a whole mission plan."""
        load, time = self.penalties
        return (
            individual.cost
            + load * individual.violation[0]
            + time * individual.violation[1]
        )

    def adapt(self, ratios: Sequence[float]) -> None:
        """Raise the penalties that are too weak, lower those that are too strong."""
        config = self.config
        self.penalties = tuple(
            min(
                max(
                    penalty
                    * (
                        config.penalty_increase
                        if ratio < config.target_ratio
                        else config.penalty_decrease
                    ),
                    config.penalty_min,
                ),
                config.penalty_max,
            )
            for penalty, ratio in zip(self.penalties, ratios)
        )

    def __repr__(self) -> str:
        return f"Metric(penalties={self.penalties})"


# ------------------------------- Population -------------------------------- #
# Feasible and infeasible subpopulations, ranked by cost and by diversity.


class Subpopulation:
    """Subpopulation of feasible or infeasible individuals.

    Individuals are ranked by *biased fitness*: their rank on penalized cost
    plus their rank on how isolated they are, so a cheap solution and an unusual
    one both earn their place. The pairwise distance matrix is maintained
    incrementally, since every insertion and every eviction needs it.
    """

    __slots__ = ("individuals", "score", "dist", "config", "num_targets")

    def __init__(self, config: HgsConfig, num_targets: int) -> None:
        # The individuals of the subpopulation.
        self.individuals: List[Individual] = []
        # Biased fitness of each individual (lower is better).
        self.score: List[float] = []
        # Pairwise broken-pair distances between the individuals.
        self.dist: List[List[float]] = []
        self.config = config
        self.num_targets = num_targets

    def __len__(self) -> int:
        return len(self.individuals)

    @property
    def is_empty(self) -> bool:
        """Whether no individual is held."""
        return not self.individuals

    def add(self, metric: Metric, individual: Individual) -> None:
        """Insert an individual, trimming the subpopulation if it overflows."""
        row = [
            individual.distance(other, self.num_targets)
            for other in self.individuals
        ]
        for j, previous in enumerate(self.dist):
            previous.append(row[j])
        row.append(0.0)
        self.dist.append(row)
        self.individuals.append(individual)

        self.update_score(metric)
        if len(self.individuals) > self.config.pop_max:
            self.survivors_selection(metric)

    def remove(self, index: int) -> None:
        """Drop one individual and the distance row and column that describe it."""
        del self.individuals[index]
        del self.dist[index]
        for row in self.dist:
            del row[index]

    def update_score(self, metric: Metric) -> None:
        """Recompute the biased fitness of every individual."""
        n = len(self.individuals)
        self.score = [0.0] * n
        if n <= 1:
            return
        denom = n - 1

        # Rank on penalized cost, cheapest first.
        values = [metric.value(i) for i in self.individuals]
        cost_pos = [0] * n
        for rank, i in enumerate(sorted(range(n), key=values.__getitem__)):
            cost_pos[i] = rank

        # Rank on diversity, most isolated first, from the maintained matrix.
        close = min(self.config.nb_close, n - 1)
        spread = [0.0] * n
        if close > 0:
            for i in range(n):
                row = self.dist[i]
                others = sorted(row[j] for j in range(n) if j != i)
                spread[i] = sum(others[:close]) / close
        div_pos = [0] * n
        for rank, i in enumerate(
            sorted(range(n), key=spread.__getitem__, reverse=True)
        ):
            div_pos[i] = rank

        scale = 1.0 - self.config.nb_elite / n
        for i in range(n):
            base = cost_pos[i] / denom
            self.score[i] = (
                base
                if cost_pos[i] < self.config.nb_elite
                else base + scale * div_pos[i] / denom
            )

    def nearest_min(self, i: int) -> float:
        """Distance from individual `i` to its closest neighbour."""
        row = self.dist[i]
        return min((row[j] for j in range(len(row)) if j != i), default=np.inf)

    def worst_index(self) -> int:
        """The individual to drop: a clone if there is one, else the worst ranked."""
        worst = 0
        worst_clone = self.nearest_min(0) <= self.config.clone_eps
        worst_score = self.score[0]
        for i in range(1, len(self.individuals)):
            clone = self.nearest_min(i) <= self.config.clone_eps
            score = self.score[i]
            if (clone and not worst_clone) or (
                clone == worst_clone and score > worst_score
            ):
                worst, worst_clone, worst_score = i, clone, score
        return worst

    def survivors_selection(self, metric: Metric) -> None:
        """Trim the subpopulation back to its minimum size."""
        while len(self.individuals) > self.config.pop_min:
            self.remove(self.worst_index())
            self.update_score(metric)

    def best(self, metric: Metric) -> Optional[Individual]:
        """The cheapest individual held, if any."""
        if not self.individuals:
            return None
        return min(self.individuals, key=metric.value)


class Population:
    """Feasible and infeasible individual populations.

    Keeping the two apart is what lets the search cross the feasible boundary
    freely: an infeasible parent may carry exactly the structure a feasible
    child needs. The recent feasibility record paces the penalty adaptation.
    """

    __slots__ = ("feasible", "infeasible", "windows", "count", "config")

    def __init__(self, config: HgsConfig, num_targets: int) -> None:
        # Individuals respecting both operational limits.
        self.feasible = Subpopulation(config, num_targets)
        # Individuals violating at least one limit.
        self.infeasible = Subpopulation(config, num_targets)
        # Recent feasibility record, one window per limit.
        self.windows: tuple[List[bool], List[bool]] = ([], [])
        # Insertions so far, which paces the penalty adaptation.
        self.count = 0
        self.config = config

    def __len__(self) -> int:
        return len(self.feasible) + len(self.infeasible)

    @property
    def is_empty(self) -> bool:
        """Whether the population holds nothing."""
        return len(self) == 0

    def add(self, metric: Metric, individual: Individual) -> None:
        """Insert an individual and periodically retune the penalties."""
        self.count += 1
        window = self.config.adapt_iter
        for c, history in enumerate(self.windows):
            history.append(individual.violation[c] == 0.0)
            if window > 0 and len(history) > window:
                del history[: len(history) - window]

        target = self.feasible if individual.feasible else self.infeasible
        target.add(metric, individual)

        if window > 0 and self.count % window == 0:
            ratios = [sum(history) / window for history in self.windows]
            metric.adapt(ratios)
            self.feasible.update_score(metric)
            self.infeasible.update_score(metric)

    def pick(self, rng) -> Optional[Individual]:
        """Binary tournament over the whole population; the best rank wins."""
        total = len(self)
        if total == 0:
            return None
        nf = len(self.feasible)
        draws = min(max(self.config.nb_pick, 1), total)

        chosen = 0
        best_score = np.inf
        for d in range(draws):
            g = int(rng.integers(total))
            score = (
                self.feasible.score[g]
                if g < nf
                else self.infeasible.score[g - nf]
            )
            if d == 0 or score < best_score:
                chosen, best_score = g, score

        source = self.feasible if chosen < nf else self.infeasible
        return source.individuals[
            chosen if chosen < nf else chosen - nf
        ].copy()

    def best(self, metric: Metric) -> Optional[Individual]:
        """The cheapest feasible individual, or the cheapest one at all."""
        return self.feasible.best(metric) or self.infeasible.best(metric)


# ------------------------------- Giant tours ------------------------------- #
# Giant tours: every chaser plan of an individual chained into one permutation.


def mean_raan(context: Context, plan) -> float:
    """Circular mean of the RAAN of a plan's targets, at their visiting times."""
    states = context.problem.states
    sin = cos = 0.0
    any_target = False
    for d, t in zip(plan.sequence[1:], plan.depart[1:]):
        angle = states[d].r + states[d].dr * t
        sin += math.sin(angle)
        cos += math.cos(angle)
        any_target = True
    if not any_target:
        return 0.0
    mean = math.atan2(sin, cos)
    return mean + TAU if mean < 0.0 else mean


def _order_by_nearest(context: Context, plans: List) -> List:
    """Chain the plans greedily, each following the one cheapest to reach."""
    if len(plans) <= 1:
        return plans
    proximity = context.proximity
    head = [p.sequence[1] for p in plans]
    pending = list(range(len(plans)))

    first = min(pending, key=lambda i: proximity(DEPOT, head[i]))
    pending.remove(first)
    order = [first]
    while pending:
        tail = plans[order[-1]].sequence[-1]
        nxt = min(pending, key=lambda i: proximity(tail, head[i]))
        pending.remove(nxt)
        order.append(nxt)
    return [plans[i] for i in order]


def giant_tour(context: Context, individual: Individual) -> List[int]:
    """Concatenate the chaser plans into a single permutation of the targets.

    The order the plans are chained in decides which targets end up adjacent in
    the tour, and therefore which groupings a crossover can preserve; chaining
    them by proximity keeps the tour meaningful rather than arbitrary.
    """
    active = [p for p in individual.plans if p.load > 0]
    ordering = context.config.ordering
    if ordering is Ordering.RAAN:
        active = sorted(active, key=lambda p: mean_raan(context, p))
    elif ordering is Ordering.NEAREST:
        active = _order_by_nearest(context, active)
    return [d for plan in active for d in plan.sequence[1:]]


# -------------------------------- Crossover -------------------------------- #
# Recombination: the crossover operators drawing a child from two parents.


def order_crossover(
    context: Context, first: Sequence[int], second: Sequence[int]
) -> List[int]:
    """Order crossover: keep a block of the first parent, fill from the second."""
    n = len(first)
    if n < 2:
        return list(first)
    rng = context.rng
    start = int(rng.integers(n))
    end = int(rng.integers(n))
    while end == start:
        end = int(rng.integers(n))

    child = [-1] * n
    taken = bytearray(context.problem.num_states)
    i = start
    while True:
        child[i] = first[i]
        taken[first[i]] = 1
        if i == end:
            break
        i = (i + 1) % n

    # The second parent is read from a fixed offset while the write cursor
    # advances independently, so every position the block left free is filled
    # exactly once. Coupling the two skips positions and returns a child
    # shorter than its parents.
    stop = (end + 1) % n
    pos = stop
    for offset in range(n):
        target = second[(stop + offset) % n]
        if not taken[target]:
            child[pos] = target
            taken[target] = 1
            pos = (pos + 1) % n
    return child


def edge_recombination(
    context: Context, first: Sequence[int], second: Sequence[int]
) -> List[int]:
    """Edge recombination: prefer successions both parents agree on."""
    n = len(first)
    if n < 2:
        return list(first)
    rng = context.rng
    adjacency = [set() for _ in range(context.problem.num_states)]
    for tour in (first, second):
        for a, b in zip(tour, tour[1:]):
            adjacency[a].add(b)
            adjacency[b].add(a)

    child: List[int] = []
    visited = bytearray(context.problem.num_states)
    current = first[0]

    while len(child) < n:
        child.append(current)
        visited[current] = 1
        # Sorted, so the tie-break depends on the seed alone and never on a
        # set's iteration order.
        peers = sorted(adjacency[current])
        for peer in peers:
            adjacency[peer].discard(current)
        live = [d for d in peers if not visited[d]]
        adjacency[current].clear()
        if len(child) == n:
            break
        if live:
            fewest = min(len(adjacency[d]) for d in live)
            tied = [d for d in live if len(adjacency[d]) == fewest]
            current = tied[int(rng.integers(len(tied)))]
        else:
            free = [d for d in first if not visited[d]]
            if not free:
                break
            current = free[int(rng.integers(len(free)))]
    return child


def selective_route_exchange(
    context: Context, first: Individual, second: Individual
) -> List[List[int]]:
    """Replace a block of the first parent's plans by plans of the second."""
    rng = context.rng
    left = [list(p.sequence[1:]) for p in first.plans if p.load > 0]
    right = [list(p.sequence[1:]) for p in second.plans if p.load > 0]
    if not left:
        return right
    if not right:
        return left

    count = 1 + int(rng.integers(min(len(left), len(right))))
    start_left = int(rng.integers(len(left)))
    start_right = int(rng.integers(len(right)))

    states = context.problem.num_states
    removed = [False] * len(left)
    injected = bytearray(states)
    buckets: List[List[int]] = []

    for t in range(count):
        removed[(start_left + t) % len(left)] = True
        donor = right[(start_right + t) % len(right)]
        for d in donor:
            injected[d] = 1
        buckets.append(list(donor))

    served = bytearray(injected)
    for i, bucket in enumerate(left):
        if removed[i]:
            continue
        kept = [d for d in bucket if not injected[d]]
        for d in kept:
            served[d] = 1
        buckets.append(kept)

    orphans: List[int] = []
    for i in range(len(left)):
        if not removed[i]:
            continue
        for d in left[i]:
            if not served[d]:
                served[d] = 1
                orphans.append(d)

    while len(buckets) < context.problem.num_chasers:
        buckets.append([])
    for target in orphans:
        cheapest_insert(context, buckets, target)
    return buckets


def recombine(context: Context) -> Individual:
    """Recombine two parents drawn from the population into a child."""
    first = context.population.pick(context.rng)
    second = context.population.pick(context.rng)
    if first is None or second is None:
        return context.generate()

    operator = context.config.crossover
    if operator is Crossover.SREX:
        return context.assemble(
            selective_route_exchange(context, first, second)
        )

    tour_a = giant_tour(context, first)
    if not tour_a:
        return context.assemble(
            selective_route_exchange(context, first, second)
        )
    tour_b = giant_tour(context, second)

    if operator is Crossover.ERX:
        child = edge_recombination(context, tour_a, tour_b)
    else:
        child = order_crossover(context, tour_a, tour_b)

    # Occasionally allow one more chaser than the parent used, so that the
    # number of active chasers can grow as well as shrink.
    extra = 1 if context.rng.random() < 0.1 else 0
    goal = min(max(first.active, 1) + extra, context.problem.num_chasers)
    # A tuple, so every `tour[i:j + 1]` slice below is one too and the
    # schedule layer can use it as a cache key without copying.
    return context.assemble(split(context, tuple(child), goal))


def cheapest_insert(
    context: Context, buckets: List[List[int]], target: int
) -> None:
    """Place a target where it costs the least, over all plans and positions."""
    plan, value = context.plan, context.value
    best = None
    best_delta = math.inf

    for b, bucket in enumerate(buckets):
        base = value(plan(bucket))
        for position in range(len(bucket) + 1):
            candidate = bucket[:position] + [target] + bucket[position:]
            delta = value(plan(candidate)) - base
            if delta < best_delta:
                best = (b, position)
                best_delta = delta
    if best is None:
        buckets[0].append(target)
    else:
        buckets[best[0]].insert(best[1], target)


# --------------------------------- Arrays ---------------------------------- #
# The data the compiled loops read and edit, and the cache of priced plans.


class Plans(NamedTuple):
    """The chaser plans of one individual, as arrays a compiled search edits."""

    # Sequence of plan `s` is `seq[s, :size[s]]`, the depot first.
    seq: np.ndarray
    # Number of states in each plan, the depot included.
    size: np.ndarray
    # Cost of each plan, as the schedule layer priced it.
    cost: np.ndarray
    # Time of each plan [s], as the schedule layer priced it.
    time: np.ndarray


class SearchData(NamedTuple):
    """Everything the compiled local search and split read, besides the plans."""

    # Kernel of the schedule layer pricing candidate plans.
    schedule: Any
    # Typed dict from the key of a sequence to the `(cost, time)` of its plan.
    cache: Any
    # `(n, n)` per-leg lower bounds the moves are screened against.
    bounds: np.ndarray
    # `(n, width)` closest peers of each target, nearest first.
    neighbors: np.ndarray
    # Indices of the targets to collect.
    targets: np.ndarray
    # Maximum number of targets a chaser collects.
    max_load: int
    # Maximum mission time per chaser [s].
    max_time: float


# numba types of the plan cache: two hashes of a sequence -> (cost, time).
PLAN_KEY = types.UniTuple(types.uint64, 2)
PLAN_VALUE = types.UniTuple(types.float64, 2)


@njit(**JIT)
def _mix(h):
    """splitmix64 finalizer: spreads every bit of `h` over the whole word."""
    h = (h ^ (h >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return h ^ (h >> np.uint64(31))


@njit(**JIT)
def _sequence_key(seq, size):
    """Two independent 64-bit hashes of `seq[:size]`.

    Two sequences share a key only if both hashes collide, which among millions
    of plans has odds around 1e-26: the key stands for the sequence itself.
    """
    h1 = np.uint64(0x9E3779B97F4A7C15)
    h2 = np.uint64(size)
    for k in range(size):
        x = np.uint64(seq[k])
        h1 = _mix(h1 ^ x)
        h2 = _mix(h2 + x + np.uint64(0x632BE59BD9B4E019))
    return h1, h2


@njit(**KERNEL)
def _price(data, seq, size):
    """Cost and time of the plan along `seq[:size]`, each sequence priced once."""
    key = _sequence_key(seq, size)
    if key in data.cache:
        return data.cache[key]
    cost, time, _, _, _ = schedule_evaluate(data.schedule, seq[:size])
    data.cache[key] = (cost, time)
    return cost, time


@njit(**JIT)
def _value(max_load, max_time, penalties, size, cost, time):
    """Penalized cost of a plan, its overruns measured as the schedule layer
    measures them."""
    excess_load = max(size - 1 - max_load, 0)
    excess_time = max(time - max_time, 0.0)
    return cost + penalties[0] * excess_load + penalties[1] * excess_time


@njit(**JIT)
def _bound(bounds, seq, size):
    """Lower bound on the cost of a candidate chaser sequence."""
    total = 0.0
    for k in range(1, size):
        total += bounds[seq[k - 1], seq[k]]
    return total


# ---------------------------- Candidate builders --------------------------- #
# Each writes a candidate sequence into `out` and returns its size, standing for
# the tuple arithmetic of `hgs.Search`.


@njit(**JIT)
def _removed(seq, size, pos, out):
    """`seq[:pos] + seq[pos + 1:]`."""
    for k in range(pos):
        out[k] = seq[k]
    for k in range(pos + 1, size):
        out[k - 1] = seq[k]
    return size - 1


@njit(**JIT)
def _inserted(seq, size, pos, node, out):
    """`seq[:pos] + (node,) + seq[pos:]`."""
    for k in range(pos):
        out[k] = seq[k]
    out[pos] = node
    for k in range(pos, size):
        out[k + 1] = seq[k]
    return size + 1


@njit(**JIT)
def _replaced(seq, size, pos, node, out):
    """`seq[:pos] + (node,) + seq[pos + 1:]`."""
    for k in range(size):
        out[k] = seq[k]
    out[pos] = node
    return size


@njit(**JIT)
def _swapped(seq, size, i, j, out):
    """`seq` with the states at `i` and `j` exchanged."""
    for k in range(size):
        out[k] = seq[k]
    out[i] = seq[j]
    out[j] = seq[i]
    return size


@njit(**JIT)
def _reversed(seq, size, start, end, out):
    """`seq[:start] + seq[start:end + 1][::-1] + seq[end + 1:]`."""
    for k in range(size):
        out[k] = seq[k]
    for k in range(start, end + 1):
        out[k] = seq[start + end - k]
    return size


@njit(**JIT)
def _joined(head, cut, tail, start, tail_size, out):
    """`head[:cut] + tail[start:tail_size]`."""
    for k in range(cut):
        out[k] = head[k]
    for k in range(start, tail_size):
        out[cut + k - start] = tail[k]
    return cut + tail_size - start


# ------------------------------ Local search ------------------------------- #
# The moves of `hgs.Search`, compiled one for one over `Plans`.


@njit(**JIT)
def _plan_value(plans, s, max_load, max_time, penalties):
    """Penalized cost of plan `s`."""
    return _value(
        max_load,
        max_time,
        penalties,
        plans.size[s],
        plans.cost[s],
        plans.time[s],
    )


@njit(**JIT)
def _store(plans, slot, candidate, size, cost, time):
    """Replace plan `slot` by a priced candidate."""
    for k in range(size):
        plans.seq[slot, k] = candidate[k]
    plans.size[slot] = size
    plans.cost[slot] = cost
    plans.time[slot] = time


@njit(**JIT)
def _register(plans, s, holder, rank):
    """Record where each target of one plan currently sits."""
    for position in range(1, plans.size[s]):
        node = plans.seq[s, position]
        holder[node] = s
        rank[node] = position


@njit(**KERNEL)
def _accept_one(data, plans, penalties, slot, candidate, size, base):
    """Screen a candidate replacement for one plan, then price it for real."""
    if _bound(data.bounds, candidate, size) >= base:
        return False
    cost, time = _price(data, candidate, size)
    if (
        _value(data.max_load, data.max_time, penalties, size, cost, time)
        >= base
    ):
        return False
    _store(plans, slot, candidate, size, cost, time)
    return True


@njit(**KERNEL)
def _accept_two(data, plans, penalties, s1, c1, size1, s2, c2, size2, base):
    """Screen a candidate replacement for a pair of plans."""
    if _bound(data.bounds, c1, size1) + _bound(data.bounds, c2, size2) >= base:
        return False
    cost1, time1 = _price(data, c1, size1)
    cost2, time2 = _price(data, c2, size2)
    value1 = _value(
        data.max_load, data.max_time, penalties, size1, cost1, time1
    )
    value2 = _value(
        data.max_load, data.max_time, penalties, size2, cost2, time2
    )
    if value1 + value2 >= base:
        return False
    _store(plans, s1, c1, size1, cost1, time1)
    _store(plans, s2, c2, size2, cost2, time2)
    return True


@njit(**KERNEL)
def _intra_relocate(data, plans, work, penalties, s, pos):
    """Move one target elsewhere in its own plan."""
    seq = plans.seq[s]
    n = plans.size[s]
    if n < 3 or pos == 0 or pos >= n:
        return False
    base = _plan_value(plans, s, data.max_load, data.max_time, penalties)
    node = seq[pos]
    rest = _removed(seq, n, pos, work[1])

    for insert_at in range(1, n):
        if insert_at == pos:
            continue
        size = _inserted(work[1], rest, insert_at, node, work[0])
        if _accept_one(data, plans, penalties, s, work[0], size, base):
            return True
    return False


@njit(**KERNEL)
def _intra_swap(data, plans, work, penalties, s, pos):
    """Exchange two targets within the same plan."""
    seq = plans.seq[s]
    n = plans.size[s]
    if n < 3 or pos == 0 or pos >= n:
        return False
    base = _plan_value(plans, s, data.max_load, data.max_time, penalties)
    for other in range(pos + 1, n):
        size = _swapped(seq, n, pos, other, work[0])
        if _accept_one(data, plans, penalties, s, work[0], size, base):
            return True
    return False


@njit(**KERNEL)
def _two_opt(data, plans, work, penalties, s, pos):
    """Reverse a segment of a plan."""
    seq = plans.seq[s]
    n = plans.size[s]
    if n < 3 or pos == 0 or pos >= n - 1:
        return False
    base = _plan_value(plans, s, data.max_load, data.max_time, penalties)
    for end in range(pos + 1, n):
        size = _reversed(seq, n, pos, end, work[0])
        if _accept_one(data, plans, penalties, s, work[0], size, base):
            return True
    return False


@njit(**KERNEL)
def _inter_relocate(data, plans, work, penalties, s1, pos1, s2, pos2):
    """Hand one target over to another chaser, or trade two of them."""
    if s1 == s2:
        return False
    seq1 = plans.seq[s1]
    seq2 = plans.seq[s2]
    n1 = plans.size[s1]
    n2 = plans.size[s2]
    if pos1 == 0 or pos1 >= n1 or pos2 == 0 or pos2 > n2:
        return False
    base = _plan_value(
        plans, s1, data.max_load, data.max_time, penalties
    ) + _plan_value(plans, s2, data.max_load, data.max_time, penalties)
    node = seq1[pos1]

    at = min(pos2, n2)
    size1 = _removed(seq1, n1, pos1, work[0])
    size2 = _inserted(seq2, n2, at, node, work[1])
    if _accept_two(
        data, plans, penalties, s1, work[0], size1, s2, work[1], size2, base
    ):
        return True

    if pos2 < n2:
        size1 = _replaced(seq1, n1, pos1, seq2[pos2], work[0])
        size2 = _replaced(seq2, n2, pos2, node, work[1])
        if _accept_two(
            data,
            plans,
            penalties,
            s1,
            work[0],
            size1,
            s2,
            work[1],
            size2,
            base,
        ):
            return True
    return False


@njit(**KERNEL)
def _two_opt_star(data, plans, work, penalties, s1, pos1, s2, pos2):
    """Exchange the tails of two plans."""
    if s1 == s2:
        return False
    seq1 = plans.seq[s1]
    seq2 = plans.seq[s2]
    n1 = plans.size[s1]
    n2 = plans.size[s2]
    if pos1 == 0 or pos1 > n1 or pos2 == 0 or pos2 > n2:
        return False
    base = _plan_value(
        plans, s1, data.max_load, data.max_time, penalties
    ) + _plan_value(plans, s2, data.max_load, data.max_time, penalties)
    size1 = _joined(seq1, pos1, seq2, pos2, n2, work[0])
    size2 = _joined(seq2, pos2, seq1, pos1, n1, work[1])
    return _accept_two(
        data, plans, penalties, s1, work[0], size1, s2, work[1], size2, base
    )


class Clocks(NamedTuple):
    """Where each target sits, and when plans last changed, across the sweeps."""

    # The plan currently holding each target.
    holder: np.ndarray
    # The position of each target in the plan holding it.
    rank: np.ndarray
    # Reading of the clock when each target was last examined.
    tested: np.ndarray
    # Reading of the clock when each plan last changed.
    touched: np.ndarray
    # Two scratch rows the moves build their candidates in.
    work: np.ndarray


@njit(**KERNEL)
def sweep_once(data, plans, clocks, order, entries, penalties, sweep, clock):
    """One sweep of the catalogue, in the order given.

    `order` and `entries` are drawn per sweep by the caller: the entry point
    into each target's neighbour ring comes from `entries`. Returns whether any
    move was accepted, and the clock, which ticks once per accepted move.
    """
    num_plans = plans.size.size
    neighbors = data.neighbors
    width = neighbors.shape[1]
    holder = clocks.holder
    rank = clocks.rank
    tested = clocks.tested
    touched = clocks.touched
    work = clocks.work
    # The first position of a plan, as a variable: a literal would have the
    # moves compiled a second time, for its own numba type.
    first = np.int64(1)

    improved = False
    for at in range(order.size):
        u = order[at]
        last_tested = tested[u]
        tested[u] = clock

        # Offer u to each of its neighbors, from a random entry point.
        done = False
        if width:
            start = entries[at] % width
            for offset in range(width):
                if start + offset >= width:
                    v = neighbors[u, start + offset - width]
                else:
                    v = neighbors[u, start + offset]
                su = holder[u]
                sv = holder[v]
                if su == sv:
                    continue
                if max(touched[su], touched[sv]) <= last_tested:
                    continue
                pu = rank[u]
                pv = rank[v]
                if (
                    _inter_relocate(
                        data, plans, work, penalties, su, pu, sv, pv + 1
                    )
                    or (
                        pu == 1
                        and _inter_relocate(
                            data, plans, work, penalties, sv, pv, su, pu
                        )
                    )
                    or _two_opt_star(
                        data, plans, work, penalties, su, pu, sv, pv + 1
                    )
                ):
                    clock += 1
                    touched[su] = clock
                    touched[sv] = clock
                    _register(plans, su, holder, rank)
                    _register(plans, sv, holder, rank)
                    improved = True
                    done = True
                    break
        if done:
            continue

        # Once the plans have settled, try waking an idle chaser up.
        if sweep > 1:
            idle = -1
            for k in range(num_plans):
                if plans.size[k] <= 1:
                    idle = k
                    break
            su = holder[u]
            pu = rank[u]
            if (
                idle >= 0
                and idle != su
                and (
                    _two_opt_star(
                        data, plans, work, penalties, su, pu, idle, first
                    )
                    or _inter_relocate(
                        data, plans, work, penalties, su, pu, idle, first
                    )
                )
            ):
                clock += 1
                touched[su] = clock
                touched[idle] = clock
                _register(plans, su, holder, rank)
                _register(plans, idle, holder, rank)
                improved = True
                continue

        # Rearrange u's own plan, but only if that plan has moved on.
        if touched[holder[u]] > last_tested:
            for move in range(3):
                su = holder[u]
                pu = rank[u]
                if move == 0:
                    moved = _intra_relocate(
                        data, plans, work, penalties, su, pu
                    )
                elif move == 1:
                    moved = _intra_swap(data, plans, work, penalties, su, pu)
                else:
                    moved = _two_opt(data, plans, work, penalties, su, pu)
                if moved:
                    clock += 1
                    touched[su] = clock
                    _register(plans, su, holder, rank)
                    improved = True

    return improved, clock


class Search:
    """Local search operator, sweeping the catalogue with `sweep_once`.

    Every move is screened twice. First against the additive lower bound, which
    is a handful of table reads; only if that cannot rule the move out is the
    schedule layer asked for a real plan. The bound is why a search over
    thousands of targets terminates at all -- it rejects the overwhelming
    majority of candidates without running a dynamic program.

    The `tested` / `touched` clocks carry the other half of the pruning: a pair
    of plans that has not changed since the last time a target was examined
    cannot yield a new improving move, so it is skipped outright.
    """

    __slots__ = ("context", "individual", "plans", "clocks")

    def __init__(self, context: Context, individual: Individual) -> None:
        self.context = context
        self.individual = individual

        # A plan never holds more states than the instance has.
        states = context.problem.num_states
        count = len(individual.plans)
        seq = np.zeros((count, states), dtype=np.int64)
        size = np.empty(count, dtype=np.int64)
        cost = np.empty(count)
        time = np.empty(count)
        for s, plan in enumerate(individual.plans):
            seq[s, : len(plan.sequence)] = plan.sequence
            size[s] = len(plan.sequence)
            cost[s] = plan.cost
            time[s] = plan.time
        self.plans = Plans(seq, size, cost, time)
        self.clocks = Clocks(
            np.zeros(states, dtype=np.int64),
            np.zeros(states, dtype=np.int64),
            np.zeros(states, dtype=np.int64),
            np.ones(count, dtype=np.int64),
            np.empty((2, states), dtype=np.int64),
        )

    def run(self) -> None:
        """Sweep the catalogue until no move helps.

        The sweeps are driven from here, so that the random draws stay the ones
        NumPy makes, one pair per sweep.
        """
        context = self.context
        penalties = context.metric.penalties
        for s in range(len(self.individual.plans)):
            _register(self.plans, s, self.clocks.holder, self.clocks.rank)

        sweep = 0
        clock = 1
        improved = True
        while improved:
            sweep += 1
            order = context.rng.permutation(context.targets)
            # One draw for the whole sweep: the entry point into each target's
            # neighbour ring.
            entries = context.rng.integers(0, 1 << 30, size=len(order))
            with np.errstate(over="ignore"):  # the cache keys wrap on purpose
                improved, clock = context.run_sweep(
                    context.search,
                    self.plans,
                    self.clocks,
                    order,
                    entries,
                    penalties,
                    sweep,
                    clock,
                )

        # Hand the plans that changed back as the schedule layer's own plans.
        plans = self.individual.plans
        for s in range(len(plans)):
            if self.clocks.touched[s] > 1:
                sequence = self.plans.seq[s, : self.plans.size[s]]
                plans[s] = context.schedule.evaluate(tuple(sequence.tolist()))
        self.individual.refresh(context.problem)


# ---------------------------------- Split ---------------------------------- #
# The split of a giant tour back into chaser plans, compiled.


@njit(**KERNEL)
def _split(data, tour, goal, penalties, load_cap, time_cap):
    """Split a giant tour into `goal` chaser plans at most.

    The shortest-path pass of `hgs.split`: `memo[s, j]` is the least penalized
    cost of covering the first `j` targets with `s` chasers, and the scan from
    a target stops at the first segment past the relaxed limits. Returns the
    cut positions: plan `s` flies `tour[cuts[s]:cuts[s + 1]]`.
    """
    n = tour.size
    k = goal

    # `priced[i, :count[i]]` values the segments starting at `i`, priced once
    # each, extending one target at a time from the departure orbit.
    priced = np.empty((n, n))
    count = np.full(n, -1, dtype=np.int64)
    segment = np.empty(n + 1, dtype=np.int64)
    segment[0] = DEPOT

    memo = np.full((k + 1, n + 1), np.inf)
    pred = np.zeros((k + 1, n + 1), dtype=np.int64)
    memo[0, 0] = 0.0

    for s in range(1, k + 1):
        for i in range(s - 1, n):
            base = memo[s - 1, i]
            if base == np.inf:
                continue
            if count[i] < 0:
                count[i] = 0
                size = 1
                for j in range(i, n):
                    segment[size] = tour[j]
                    size += 1
                    cost, time = _price(data, segment, size)
                    priced[i, count[i]] = _value(
                        data.max_load,
                        data.max_time,
                        penalties,
                        size,
                        cost,
                        time,
                    )
                    count[i] += 1
                    if size - 1 > load_cap or time > time_cap:
                        break
            for offset in range(count[i]):
                candidate = base + priced[i, offset]
                at = i + offset + 1
                if candidate < memo[s, at]:
                    memo[s, at] = candidate
                    pred[s, at] = i

    best_k = 1
    for s in range(2, k + 1):
        if memo[s, n] < memo[best_k, n]:
            best_k = s
    if memo[best_k, n] == np.inf:
        return np.array([0, n])

    cuts = np.empty(best_k + 1, dtype=np.int64)
    j = n
    for s in range(best_k, 0, -1):
        cuts[s] = j
        j = pred[s, j]
    cuts[0] = j
    return cuts


def split(context: Context, tour: Sequence[int], goal: int) -> List[List[int]]:
    """Split a giant tour into individual chaser plans.

    The shortest-path pass itself runs compiled, in `_split`; what is left here
    is cutting the tour where it says.
    """
    if len(tour) == 0:
        return []
    config, problem = context.config, context.problem
    with np.errstate(over="ignore"):  # the cache keys wrap around on purpose
        cuts = context.run_split(
            context.search,
            np.array(tour, dtype=np.int64),
            max(goal, 1),
            context.metric.penalties,
            config.threshold_factor * problem.max_load,
            config.threshold_factor * problem.max_time,
        )
    return [list(tour[a:b]) for a, b in zip(cuts[:-1], cuts[1:])]


# --------------------------------- Context --------------------------------- #
# Context: everything one hybrid genetic search run holds and does.


class Context:
    """Context manager holding all pieces of the algorithm."""

    __slots__ = (
        "problem",
        "schedule",
        "config",
        "rng",
        "targets",
        "metric",
        "population",
        "neighbors",
        "start_time",
        "iteration",
        "stalled",
        "trace",
        "bounds",
        "bounds_rows",
        "search",
        "run_sweep",
        "run_split",
        "_empty",
    )

    _DEPOT_HEAD = (DEPOT,)

    def __init__(
        self, problem: Problem, schedule: ScheduleLayer, config: HgsConfig
    ) -> None:
        # The instance being solved.
        self.problem = problem
        # The layer pricing a chaser plan along a fixed sequence.
        self.schedule = schedule
        # The parameters of the run.
        self.config = config
        # The random source, seeded from the configuration.
        self.rng = np.random.default_rng(config.seed)
        # Indices of the targets to collect.
        self.targets = np.asarray(problem.targets, dtype=np.int64)

        # Per-leg lower bounds, the screen every local-search move is tried
        # against. Built here rather than asked of the schedule layer: only
        # this search needs it.
        self.bounds = pair_lb(schedule, problem)
        # The screen reads one scalar per leg, and nested Python lists serve
        # that pattern about 2.7x faster than a numpy array does. The mirror
        # costs roughly 4x the memory of the table, which is the right trade
        # here: ~110 MB on the largest shipped catalogue.
        self.bounds_rows: List[List[float]] = self.bounds.tolist()

        # The penalized objective.
        self.metric = Metric(problem, schedule, config)
        # The individuals currently held.
        self.population = Population(config, problem.num_targets)
        # For each target, its `nb_neighbors` closest peers.
        self.neighbors: List[List[int]] = []
        # Start of the run.
        self.start_time = time.perf_counter()
        # Number of iterations.
        self.iteration = 0
        # Number of iterations without improvement.
        self.stalled = 0
        # Algorithmic trace, one entry per logged iteration.
        self.trace: List = []
        self._empty: tuple = ()

        n = problem.num_states
        # What the compiled loops read, and the cost and time of every plan
        # they priced. The neighbours are filled in by `build_neighbors`.
        self.search = SearchData(
            schedule.kernel(),
            Dict.empty(PLAN_KEY, PLAN_VALUE),
            self.bounds,
            np.zeros((n, 0), dtype=np.int64),
            self.targets,
            problem.max_load,
            float(problem.max_time),
        )

        # Bind the compiled loops once, on arguments of the types they get.
        plans = Plans(
            np.zeros((1, n), dtype=np.int64),
            np.ones(1, dtype=np.int64),
            np.zeros(1),
            np.zeros(1),
        )
        clocks = Clocks(
            np.zeros(n, dtype=np.int64),
            np.zeros(n, dtype=np.int64),
            np.zeros(n, dtype=np.int64),
            np.ones(1, dtype=np.int64),
            np.empty((2, n), dtype=np.int64),
        )
        penalties = self.metric.penalties
        one = np.zeros(1, dtype=np.int64)
        self.run_sweep = bind(
            sweep_once, self.search, plans, clocks, one, one, penalties, 1, 1
        )
        self.run_split = bind(_split, self.search, one, 1, penalties, 0.0, 0.0)

    # ------------------------------ Helpers ------------------------------- #

    def plan(self, bucket: Sequence[int]) -> ChaserPlan:
        """Schedule a chaser over `bucket`, starting from the departure orbit."""
        if type(bucket) is tuple:
            return self.schedule.evaluate(self._DEPOT_HEAD + bucket)
        return self.schedule.evaluate((DEPOT, *bucket))

    def value(self, plan: ChaserPlan) -> float:
        """Penalized cost of a chaser plan."""
        return self.metric.plan_value(plan)

    def proximity(self, i: int, j: int) -> float:
        """The cheapest the transfer from `i` to `j` can ever be."""
        return self.bounds_rows[i][j]

    def assemble(self, buckets: Sequence[Sequence[int]]) -> Individual:
        """Turn target buckets into a mission plan, one bucket per chaser."""
        chasers = self.problem.num_chasers
        if len(buckets) > chasers:
            raise ValueError(f"{len(buckets)} buckets for {chasers} chasers")
        plans = [
            self.plan(buckets[c] if c < len(buckets) else self._empty)
            for c in range(chasers)
        ]
        return Individual(self.problem, plans)

    def improve(self, individual: Individual) -> None:
        """Improve an individual by local search, until no move helps."""
        Search(self, individual).run()

    def insert(self, individual: Individual) -> None:
        """Insert an individual into the population, retuning the penalties."""
        old = self.best_value()
        self.population.add(self.metric, individual)
        new = self.best_value()
        self.stalled = 0 if new < old else self.stalled + 1
        self.iteration += 1

    def best(self) -> Optional[Individual]:
        """The cheapest individual held, feasible if any is."""
        return self.population.best(self.metric)

    def best_value(self) -> float:
        """Penalized cost of the incumbent, infinite while none exists."""
        best = self.best()
        return self.metric.value(best) if best is not None else np.inf

    def elapsed(self) -> float:
        """Seconds since the run started."""
        return time.perf_counter() - self.start_time

    def terminate(self) -> bool:
        """Whether the run should stop."""
        return (
            self.iteration >= self.config.max_iter
            or self.stalled >= self.config.max_nimp
            or self.elapsed() >= self.config.max_time
        )

    # ---------------------------- Generation ------------------------------- #

    def generate(self) -> Individual:
        """Build one initial individual, by whichever strategy is configured."""
        if self.config.generation is Generation.RANDOM:
            return self.generate_random()
        return self.generate_greedy()

    def generate_greedy(self) -> Individual:
        """Grow each chaser plan by repeatedly appending the cheapest target."""
        problem = self.problem
        available = np.zeros(problem.num_states, dtype=bool)
        available[self.targets] = True

        # Targets ordered by how cheaply the swarm reaches them, then shuffled
        # within a short window so repeated calls seed different plans.
        depot_row = self.bounds[DEPOT]
        ordered = list(
            self.targets[np.argsort(depot_row[self.targets], kind="stable")]
        )
        self._window_shuffle(ordered, 4)

        buckets: List[List[int]] = []
        for start in ordered:
            start = int(start)
            if not available[start] or len(buckets) >= problem.num_chasers:
                continue
            available[start] = False
            bucket = [start]
            while True:
                nxt = self._cheapest_extension(bucket, available)
                if nxt is None:
                    break
                available[nxt] = False
                bucket.append(nxt)
            buckets.append(bucket)

        leftovers = [int(d) for d in self.targets if available[d]]
        self._place_leftovers(buckets, leftovers)
        return self.assemble(buckets)

    def generate_random(self) -> Individual:
        """Fill each chaser plan with random targets while it stays feasible."""
        remaining = [int(d) for d in self.rng.permutation(self.targets)]

        buckets: List[List[int]] = []
        while remaining and len(buckets) < self.problem.num_chasers:
            bucket: List[int] = []
            rejected: List[int] = []
            while remaining:
                candidate = remaining.pop()
                bucket.append(candidate)
                if not self._admissible(bucket):
                    bucket.pop()
                    rejected.append(candidate)
            remaining = rejected
            if not bucket:
                break  # nothing fits anywhere; stop rather than spin
            buckets.append(bucket)

        self._place_leftovers(buckets, remaining)
        return self.assemble(buckets)

    def _window_shuffle(self, order: List[int], width: int) -> None:
        """Perturb an ordering locally, keeping its overall structure."""
        n = len(order)
        if n < 2:
            return
        draws = self.rng.random(n - 1)
        for i in range(n - 1):
            end = min(i + width, n - 1)
            j = i + int(draws[i] * (end - i + 1))
            order[i], order[j] = order[j], order[i]

    def _admissible(self, bucket: Sequence[int]) -> bool:
        """Whether a chaser can fly this bucket within the operational limits."""
        return len(
            bucket
        ) <= self.problem.max_load and self.problem.is_feasible_plan(
            self.plan(bucket)
        )

    def _cheapest_extension(
        self, bucket: List[int], available: np.ndarray
    ) -> Optional[int]:
        """The cheapest target that can still be appended to `bucket`."""
        if len(bucket) >= self.problem.max_load:
            return None
        candidates = np.flatnonzero(available)
        if candidates.size == 0:
            return None
        row = self.bounds[bucket[-1]]
        candidates = candidates[np.argsort(row[candidates], kind="stable")]

        trial = list(bucket) + [0]
        for candidate in candidates:
            trial[-1] = int(candidate)
            if self._admissible(trial):
                return int(candidate)
        return None

    @staticmethod
    def _place_leftovers(
        buckets: List[List[int]], leftovers: Sequence[int]
    ) -> None:
        """Append the targets no plan could take, so that none is ever dropped."""
        if not leftovers:
            return
        if not buckets:
            buckets.append([])
        buckets[-1].extend(sorted(leftovers))

    # ---------------------------- Crossover -------------------------------- #

    def crossover(self) -> Individual:
        """Recombine two parents drawn from the population into a child."""
        return recombine(self)

    def giant_tour(self, individual: Individual) -> List[int]:
        """Concatenate the chaser plans into a single permutation of the targets."""
        return giant_tour(self, individual)

    def split(self, tour: Sequence[int], goal: int) -> List[List[int]]:
        """Split a giant tour into individual chaser plans."""
        return split(self, tuple(tour), goal)

    # --------------------------- Neighborhood ------------------------------ #

    def build_neighbors(self) -> None:
        """Rank the potential successors of every target by proximity.

        Done in one vectorised pass: the search asks for these lists once and
        then reads them millions of times, so the `n^2` sort is paid here
        rather than lazily.
        """
        size = self.config.nb_neighbors
        targets = self.targets
        self.neighbors = [[] for _ in range(self.problem.num_states)]
        if size == 0 or targets.size < 2:
            return

        block = self.bounds[np.ix_(targets, targets)].copy()
        # Self sorts strictly first and is then dropped, which leaves ties
        # among the others resolved by target index, as a stable sort must.
        np.fill_diagonal(block, -np.inf)
        order = np.argsort(block, axis=1, kind="stable")[:, 1 : size + 1]
        picked = targets[order]
        for row, i in enumerate(targets):
            self.neighbors[i] = [int(v) for v in picked[row]]

        # The compiled sweep reads them as one table, the depot row empty.
        width = len(self.neighbors[targets[0]])
        table = np.zeros((self.problem.num_states, width), dtype=np.int64)
        for i in targets:
            table[i] = self.neighbors[i]
        self.search = self.search._replace(neighbors=table)


# ----------------------------- Sequence layer ------------------------------ #
# The sequence layer driving the hybrid genetic search.


@dataclass(frozen=True, slots=True)
class LogEntry:
    """One row of the algorithmic trace."""

    # Time in seconds since the start of the run.
    time: float
    # Iteration number.
    iter: int
    # Number of iterations without improvement.
    nimp: int
    # Cost of the incumbent.
    cost: float
    # Whether the incumbent satisfies both operational limits. Early in a run
    # there may be no feasible individual at all, and `cost` is then the cost of
    # a plan that overruns something -- not comparable with what comes later.
    feasible: bool
    # Number of feasible individuals in the population.
    num_feasible: int
    # Number of infeasible individuals in the population.
    num_infeasible: int
    # Current penalty for the load limit.
    penalty_load: float
    # Current penalty for the time limit.
    penalty_time: float


# Algorithmic trace, one entry per logged iteration.
Trace = List[LogEntry]

_HEADER = (
    f" {'time':>7} {'iter':>7} {'nimp':>7} {'cost':>13} {'pop-f':>7} "
    f"{'pop-i':>7} {'pop-r':>7} {'pen-l':>7} {'pen-t':>7}"
)


class HgsSequence(SequenceLayer):
    """Sequence layer driving the hybrid genetic search.

    A population of mission plans is recombined and locally improved, kept in
    two halves -- those respecting the operational limits and those not -- so
    that the search can cross the feasible boundary and come back. Individuals
    are ranked by cost *and* by how different they are from their neighbours,
    which is what stops the population collapsing onto one solution.
    """

    __slots__ = ("schedule", "config", "context")

    def __init__(
        self, *, config: Optional[HgsConfig] = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.config = config if config is not None else HgsConfig()
        self.context: Optional[Context] = None

    @property
    def trace(self) -> Trace:
        """The trace of the last run."""
        return self.context.trace

    # ------------------------------ Logging -------------------------------- #

    def _log_row(self, context: Context) -> None:
        """Report one row of the search, print it, keep it, or neither.

        Printing and keeping are independent, but they describe the same row and
        share a cadence, so they are decided together: the incumbent is the
        costly part to look up, and it is fetched only once something is going
        to use it.
        """
        config = context.config
        if not (config.verbose or config.log_save):
            return
        if context.iteration % max(config.log_iter, 1) != 0:
            return

        best = context.best()
        population = context.population

        if config.verbose:
            cost = f"{best.cost:.1f}" if best is not None else "--"
            total = max(len(population), 1)
            print(
                f" {context.elapsed():7.1f} {context.iteration:7} {context.stalled:>7} "
                f"{cost:>13} {len(population.feasible):>7} {len(population.infeasible):>7} "
                f"{len(population.feasible) / total:>7.2f} "
                f"{context.metric.penalties[0]:>7.2e} {context.metric.penalties[1]:>7.2e}"
            )

        if config.log_save:
            context.trace.append(
                LogEntry(
                    time=context.elapsed(),
                    iter=context.iteration,
                    nimp=context.stalled,
                    cost=best.cost if best is not None else float("inf"),
                    feasible=best.feasible if best is not None else False,
                    num_feasible=len(population.feasible),
                    num_infeasible=len(population.infeasible),
                    penalty_load=context.metric.penalties[0],
                    penalty_time=context.metric.penalties[1],
                )
            )

    @staticmethod
    def _log_footer(context: Context) -> None:
        if context.stalled >= context.config.max_nimp:
            print("Number of iterations without improvement reached")
        if context.iteration >= context.config.max_iter:
            print("Maximum number of iterations reached")
        if context.elapsed() >= context.config.max_time:
            print("Maximum time limit reached")

    # ------------------------------- Layer --------------------------------- #

    def initialize(self, problem: Problem) -> None:
        super().initialize(problem)
        self.context = Context(problem, self.schedule, self.config)
        self.context.build_neighbors()

    def evaluate(self) -> MissionPlan:
        context = self.context
        verbose = context.config.verbose

        context.start_time = time.perf_counter()
        context.trace.clear()

        if verbose:
            print("Constructing initial population...")
            print(_HEADER)

        # `_log_row` decides for itself whether anything is printed or kept, so
        # the loops below call it unconditionally rather than testing a flag
        # that no longer governs both.
        for _ in range(context.config.pop_init):
            child = context.generate()
            context.improve(child)
            context.insert(child)
            self._log_row(context)

        if verbose:
            print("Starting genetic search...")

        context.iteration = 0
        context.stalled = 0

        while not context.terminate():
            child = context.crossover()
            context.improve(child)
            context.insert(child)
            self._log_row(context)

        self._log_row(context)
        if verbose:
            self._log_footer(context)

        return self._finalize(context)

    @staticmethod
    def _finalize(context: Context) -> MissionPlan:
        """Turn the best individual into a mission plan, one entry per chaser."""
        best = context.best()
        if best is None:
            raise RuntimeError("the search ended with an empty population")
        plans = list(best.plans)
        while len(plans) < context.problem.num_chasers:
            plans.append(ChaserPlan.empty())
        return MissionPlan(plans)

"""Solving with mcrdv: from a configuration to a problem, a stack, a result.

    instance: {name: iridium33, num_chasers: 15, max_load: 10, max_time: 365d}
    solver:
      type: mcrdv
      args:
        transfer: {type: qlaw, args: {factor: 1.5}}
        schedule: {type: dp, args: {grid_size: 32}}
        sequence: {type: hgs, args: {preset: 3, max_time: 3m}}

`instance` options:
- `name`: a catalogue in `data/`, or a generated instance (see `GENERATORS`)
  shaped by `cloud`.
- `num_targets` + `draw`: a random subset of a catalogue (a smaller subset of
  one draw is contained in every larger one), or the size of a generated one.
- `pick`: targets named by their 1-based row in the catalogue, in that order.
- `depot`: `centroid` (default, the targets' mean orbit) or `first` (the
  catalogue's first row, which is then not a target; a generated instance
  departs from the first row of the catalogue of its name).

Every `solver` entry is a `type` plus the `args` it is built with. Layer types
are looked up in `TRANSFERS`, `SCHEDULES` and `SEQUENCES`, which hold the
layers mcrdv ships; an experiment adds its own layers to them in its `run.py`.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import fields
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from mcrdv import MissionPlan, Problem, centroid, load_states
from mcrdv.orbit import TAU, KepState
from mcrdv.planner import ScheduleLayer, SequenceLayer, TransferLayer
from mcrdv.problem import ChaserPlan
from mcrdv.schedule import DpSchedule
from mcrdv.sequence import ClusterSequence, HgsSequence
from mcrdv.sequence.hgs import HgsConfig, mean_raan
from mcrdv.transfer import QlawTransfer

from . import runner

# Catalogues live at the repository root, beside `experiments/`.
DATA = Path(__file__).resolve().parents[2] / "data"

# Where the chasers may start.
DEPOTS = ("centroid", "first")

# Generated instances: `instance.name` -> `(count, draw, **cloud) -> states`.
GENERATORS: Dict[str, Callable[..., Sequence[KepState]]] = {}

# Layer types a configuration can name; experiments add their own.
TRANSFERS: Dict[str, Any] = {"qlaw": QlawTransfer}
SCHEDULES: Dict[str, Any] = {"dp": DpSchedule}
SEQUENCES: Dict[str, Any] = {"hgs": HgsSequence, "cluster": ClusterSequence}

# Where a configuration names what solves it.
SOLVER = "solver"

# `(config) -> result`: one solver run on one configuration.
Runner = Callable[[Mapping[str, Any]], Dict[str, Any]]

# Solvers a configuration can name, filled in by `register`.
SOLVERS: Dict[str, Runner] = {}

# The layers of the mcrdv stack, each a key of `solver.args`.
STACK = ("transfer", "schedule", "sequence")

# Search knobs given as durations (`60s`, `5m`).
_DURATIONS = ("max_time",)

# What only `HgsSequence` understands: its preset and `HgsConfig` fields.
_SEARCH_KNOBS = {"preset"} | {field.name for field in fields(HgsConfig)}


# ============================================================================ #
# Instances
# ============================================================================ #


@lru_cache(maxsize=None)
def _states(name: str) -> Tuple[KepState, ...]:
    """A catalogue, read once per process."""
    return tuple(load_states(DATA / f"{name}.csv"))


@lru_cache(maxsize=None)
def _subset(
    name: str, count: int, draw: int, skip: int = 0
) -> Tuple[KepState, ...]:
    """`count` targets drawn from the catalogue, in catalogue order, never
    from its first `skip` rows."""
    states = _states(name)[skip:]
    if not 1 <= count <= len(states):
        raise ValueError(
            f"cannot draw {count} targets from {name}, "
            f"which holds {len(states)}"
        )
    picked = np.random.default_rng(draw).permutation(len(states))[:count]
    return tuple(states[index] for index in np.sort(picked))


@lru_cache(maxsize=None)
def _picked(
    name: str, rows: Tuple[int, ...], skip: int = 0
) -> Tuple[KepState, ...]:
    """The targets at the 1-based `rows`, counted after the first `skip`."""
    states = _states(name)[skip:]
    for row in rows:
        if not 1 <= row <= len(states):
            raise ValueError(f"{name} has no target {row}")
    if len(set(rows)) != len(rows):
        raise ValueError(f"a target is picked twice from {name}")
    return tuple(states[row - 1] for row in rows)


@lru_cache(maxsize=None)
def _generated(
    name: str, count: int, draw: int, shape: str
) -> Tuple[KepState, ...]:
    """A generated instance; `shape` is `instance.cloud` as JSON, hashable."""
    return tuple(GENERATORS[name](count, draw, **json.loads(shape)))


def depot(config: Mapping[str, Any]) -> str:
    """Where the chasers start: `centroid` or `first`."""
    kind = str(runner.get(config, "instance.depot", "centroid"))
    if kind not in DEPOTS:
        known = ", ".join(DEPOTS)
        raise ValueError(f"unknown depot '{kind}' (known: {known})")
    return kind


def targets(config: Mapping[str, Any]) -> Tuple[KepState, ...]:
    """The targets an instance describes (see the module docstring)."""
    name = str(runner.get(config, "instance.name"))
    count = runner.get(config, "instance.num_targets")
    draw = int(runner.get(config, "instance.draw", 0))
    pick = runner.get(config, "instance.pick")
    skip = 1 if depot(config) == "first" else 0

    if pick is not None:
        if count is not None:
            raise ValueError("'pick' and 'num_targets' are exclusive")
        return _picked(name, tuple(int(row) for row in pick), skip)

    if name in GENERATORS:
        if count is None:
            raise ValueError(f"a {name} instance needs 'num_targets'")
        if skip and not (DATA / f"{name}.csv").exists():
            raise ValueError(f"a {name} cloud has no first row to depart from")
        shape = runner.get(config, "instance.cloud") or {}
        return _generated(
            name, int(count), draw, json.dumps(shape, sort_keys=True)
        )
    if count is None:
        return _states(name)[skip:]
    return _subset(name, int(count), draw, skip)


def num_targets(config: Mapping[str, Any]) -> int:
    return len(targets(config))


def problem(config: Mapping[str, Any]) -> Problem:
    """The `Problem` an instance describes, the depot as state 0."""
    chosen = list(targets(config))
    if depot(config) == "first":
        start = _states(str(runner.get(config, "instance.name")))[0]
    else:
        start = centroid(chosen)
    states = [start] + chosen
    return Problem(
        states,
        int(runner.get(config, "instance.num_chasers")),
        float(runner.duration(runner.get(config, "instance.max_time"))),
        int(runner.get(config, "instance.max_load")),
    )


# ============================================================================ #
# Stacks
# ============================================================================ #


def spec(node: Any) -> Tuple[str, Dict[str, Any]]:
    """The `type` and `args` of a solver or a layer. Any other key is an
    error: a knob written beside `type` belongs under `args`."""
    if node is None:
        node = {}
    if not isinstance(node, Mapping):
        raise TypeError(f"expected a 'type' and its 'args', got {node!r}")
    unknown = sorted(set(node) - {"type", "args"})
    if unknown:
        raise ValueError(
            f"unexpected key(s) {unknown} beside 'type' and 'args'; "
            "arguments belong under 'args'"
        )
    args = node.get("args") or {}
    if not isinstance(args, Mapping):
        raise TypeError(f"'args' must be a mapping, got {args!r}")
    return str(node.get("type", "")), copy.deepcopy(dict(args))


def layer(config: Mapping[str, Any], name: str) -> Tuple[str, Dict[str, Any]]:
    """The `type` and `args` of one layer of the mcrdv stack."""
    return spec(runner.get(config, f"{SOLVER}.args.{name}"))


def solver_type(config: Mapping[str, Any]) -> str:
    """The solver a configuration names, `mcrdv` by default."""
    return spec(runner.get(config, SOLVER))[0] or "mcrdv"


def register(name: str) -> Callable[[Runner], Runner]:
    """Decorator registering a `(config) -> result` solver under `name`."""

    def keep(runner: Runner) -> Runner:
        SOLVERS[name] = runner
        return runner

    return keep


def _pick(table: Dict[str, Any], kind: str, default: str, what: str) -> Any:
    """What is registered under `kind`, or under `default` if none is named."""
    chosen = kind or default
    if chosen not in table:
        known = ", ".join(sorted(table))
        raise ValueError(f"unknown {what} type '{chosen}' (known: {known})")
    return table[chosen]


def transfer(config: Mapping[str, Any]) -> TransferLayer:
    kind, args = layer(config, "transfer")
    return _pick(TRANSFERS, kind, "qlaw", "transfer")(**args)


def schedule(config: Mapping[str, Any]) -> ScheduleLayer:
    kind, args = layer(config, "schedule")
    inner = transfer(config)
    return _pick(SCHEDULES, kind, "dp", "schedule")(transfer=inner, **args)


def hgs_config(node: Any, seed: int) -> HgsConfig:
    """The search configuration, with the run's `seed` forced in (a seed among
    the layer's args would make every replication identical) and the search
    quiet unless `verbose` is asked for."""
    knobs = copy.deepcopy(dict(node)) if isinstance(node, Mapping) else {}
    knobs.pop("seed", None)
    for key in _DURATIONS:
        if key in knobs:
            knobs[key] = runner.duration(knobs[key])
    knobs["seed"] = int(seed)
    knobs.setdefault("verbose", False)

    preset = knobs.pop("preset", None)
    if preset is None:
        return HgsConfig(**knobs)
    return HgsConfig.preset(int(preset), **knobs)


def sequence(config: Mapping[str, Any]) -> SequenceLayer:
    inner = schedule(config)
    kind, args = layer(config, "sequence")
    cls = _pick(SEQUENCES, kind, "hgs", "sequence")
    seed = int(runner.get(config, "seed", 0))

    if cls is HgsSequence:
        return HgsSequence(schedule=inner, config=hgs_config(args, seed))

    # A layer with no search ignores the search knobs a shared base hands it,
    # so `hgs` and another layer can be two values of one sweep axis.
    knobs = {k: v for k, v in args.items() if k not in _SEARCH_KNOBS}
    return cls(inner, **knobs)


# ============================================================================ #
# Warm-up and solving
# ============================================================================ #


# A throwaway instance, solved only to trigger compilation.
_WARMUP = {
    "instance": {
        "name": "cosmos1408",
        "num_chasers": 4,
        "max_load": 2,
        "max_time": "365d",
    },
    "seed": 0,
}


def _shape(config: Mapping[str, Any]) -> tuple:
    """The layer types of a configuration, which decide what gets compiled."""
    return tuple(layer(config, name)[0] for name in STACK)


def warm_up(configs: Sequence[Mapping[str, Any]]) -> None:
    """Pay numba's compilation (about ten seconds) for every distinct stack
    the configurations use, before any run is timed. Other solvers are
    skipped."""
    instance = problem(_WARMUP)
    stacks = dict.fromkeys(
        _shape(config) for config in configs if solver_type(config) == "mcrdv"
    )
    for shape in stacks:
        warm = runner.derive(
            _WARMUP,
            solver={
                "type": "mcrdv",
                "args": {
                    "transfer": {"type": shape[0]},
                    "schedule": {"type": shape[1]},
                    # The shallowest preset compiles the search without
                    # entering the genetic loop.
                    "sequence": {"type": shape[2], "args": {"preset": 0}},
                },
            },
        )
        stack = sequence(warm)
        stack.initialize(instance)
        stack.evaluate()


def solve(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Run the solver a configuration names, and return its result."""
    return _pick(SOLVERS, solver_type(config), "mcrdv", "solver")(config)


@register("mcrdv")
def mcrdv(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the stack, solve, and report the plan with the usual numbers.

    Setup (the per-leg bound table) and search are timed apart. The search
    trace is empty unless the sequence layer's args ask for `log_save`.
    """
    stack = sequence(config)
    instance = problem(config)

    start = time.perf_counter()
    stack.initialize(instance)
    setup = time.perf_counter() - start

    start = time.perf_counter()
    mission = stack.evaluate()
    solve_time = time.perf_counter() - start

    return {
        "mission": mission,
        "cost": mission.cost,
        "feasible": mission.feasible,
        "active_chasers": sum(1 for plan in mission.plans if plan.load > 0),
        "makespan": max((plan.time for plan in mission.plans), default=0.0),
        "num_targets": num_targets(config),
        "t_setup": setup,
        "t_solve": solve_time,
        # Genetic iterations completed; `None` for a layer with no search.
        "iterations": getattr(
            getattr(stack, "context", None), "iteration", None
        ),
        "trace": [
            {
                "time": e.time,
                "iter": e.iter,
                "nimp": e.nimp,
                "cost": e.cost,
                "feasible": e.feasible,
                "num_feasible": e.num_feasible,
                "num_infeasible": e.num_infeasible,
                "penalty_load": e.penalty_load,
                "penalty_time": e.penalty_time,
            }
            for e in (getattr(stack, "trace", ()) or ())
        ],
    }


# ============================================================================ #
# Reading a plan (times in seconds, angles in radians)
# ============================================================================ #


def waiting(plan: ChaserPlan) -> np.ndarray:
    """Time waited at each visited state before leaving it (the last one,
    which nothing leaves, excluded)."""
    if len(plan) < 2:
        return np.zeros(0)
    return np.maximum(plan.depart[:-1] - plan.arrive[:-1], 0.0)


def raan_spread(plan: ChaserPlan, problem: Problem) -> float:
    """Largest angular distance of a plan's targets to their circular mean,
    each node read at the time it is visited."""
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
    """One row per chaser, idle ones included."""
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
    """The shape of a mission plan, as one row of a results table."""
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
        # 0 when every active chaser carries the same load.
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

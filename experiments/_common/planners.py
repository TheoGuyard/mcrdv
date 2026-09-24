"""Turning configurations into mission planners, and the problem it solves.

A configuration names what solves an instance under `solver`, and everything
named is written the same way: a `type`, and the `args` it is built from.

    solver:
      type: mcrdv
      args:
        transfer: {type: qlaw, args: {factor: 1.5}}
        schedule: {type: dp, args: {grid_size: 32}}
        sequence: {type: hgs, args: {preset: 3, max_time: 3m}}

mcrdv's arguments are its three layers, each written as a `type` and its own
`args`, so the same shape nests all the way down. Another solver registers
itself in `SOLVERS` and gives whatever `args` it wants; nothing else in the
harness knows what any of them mean.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import numpy as np

from mcrdv import Problem, centroid, load_states
from mcrdv.orbit import KepState
from mcrdv.planner import ScheduleLayer, SequenceLayer, TransferLayer
from mcrdv.schedule import DpSchedule
from mcrdv.sequence import ClusterSequence, HgsSequence
from mcrdv.sequence.hgs import HgsConfig
from mcrdv.transfer import QlawTransfer

from . import config as configuration
from . import synthetic
from .transfers import (
    ArrivalQlawTransfer,
    BlackoutQlawTransfer,
    EmbargoQlawTransfer,
    UrgencyQlawTransfer,
)

# Catalogues live at the repository root, beside `experiments/`.
DATA = Path(__file__).resolve().parents[2] / "data"

# Where a configuration names what solves it.
SOLVER = "solver"

# `(config) -> result`: what running one solver on one configuration produces.
Runner = Callable[[Mapping[str, Any]], Dict[str, Any]]

# Solvers a configuration can name, filled in by `register` below.
SOLVERS: Dict[str, Runner] = {}

# The layers of the mcrdv stack, outermost last: `solver.args` holds one entry
# per name, and each is a `type` with its own `args`.
STACK = ("transfer", "schedule", "sequence")

# Names accepted in a configuration, per layer of the mcrdv stack. The four
# beside the shipped Q-law are written in `transfers.py`, and are what the
# `transfer` experiment is about.
TRANSFERS = {
    "qlaw": QlawTransfer,
    "qlaw_arrival": ArrivalQlawTransfer,
    "qlaw_urgency": UrgencyQlawTransfer,
    "qlaw_blackout": BlackoutQlawTransfer,
    "qlaw_embargo": EmbargoQlawTransfer,
}
SCHEDULES = {"dp": DpSchedule}
SEQUENCES = {"hgs": HgsSequence, "cluster": ClusterSequence}

# Search knobs that are times, and so accept `60s` / `5m` style strings.
_DURATIONS = ("max_time",)

# What `HgsSequence` alone understands: its preset, and the fields of the
# search configuration its arguments are flattened into.
_SEARCH_KNOBS = {"preset"} | {field.name for field in fields(HgsConfig)}


# ============================================================================ #
# Problems
# ============================================================================ #


@lru_cache(maxsize=None)
def _states(name: str) -> Tuple[KepState, ...]:
    """The catalogue `name`, read once per process."""
    return tuple(load_states(DATA / f"{name}.csv"))


@lru_cache(maxsize=None)
def _subset(name: str, count: int, draw: int) -> Tuple[KepState, ...]:
    """`count` targets of the catalogue `name`, in catalogue order.

    The prefix of one random permutation, so for a given draw a smaller subset
    is contained in every larger one.
    """
    states = _states(name)
    if not 1 <= count <= len(states):
        raise ValueError(
            f"cannot draw {count} targets from {name}, "
            f"which holds {len(states)}"
        )
    picked = np.random.default_rng(draw).permutation(len(states))[:count]
    return tuple(states[index] for index in np.sort(picked))


@lru_cache(maxsize=None)
def _cloud(count: int, draw: int, shape: str) -> Tuple[KepState, ...]:
    """A synthetic cloud, generated once per process. `shape` is the
    `instance.cloud` mapping as JSON, which is what makes it cacheable."""
    return tuple(synthetic.cloud(count, draw, **json.loads(shape)))


def targets(config: Mapping[str, Any]) -> Tuple[KepState, ...]:
    """The targets an instance describes.

    `instance.name` is a catalogue in `data/`, or `synthetic` for a generated
    break-up cloud shaped by `instance.cloud` (see `synthetic.py`).
    `instance.num_targets` sizes a synthetic cloud, and draws that many targets
    from a catalogue, which is otherwise taken whole. `instance.draw` seeds
    which targets those are. It is a separate knob from the run's `seed`
    because the two are separate axes: replications of a search on one
    instance, and different instances of one size.
    """
    name = str(configuration.get(config, "instance.name"))
    count = configuration.get(config, "instance.num_targets")
    draw = int(configuration.get(config, "instance.draw", 0))

    if name == synthetic.NAME:
        if count is None:
            raise ValueError("a synthetic instance needs 'num_targets'")
        shape = configuration.get(config, "instance.cloud") or {}
        return _cloud(int(count), draw, json.dumps(shape, sort_keys=True))
    if count is None:
        return _states(name)
    return _subset(name, int(count), draw)


def num_targets(config: Mapping[str, Any]) -> int:
    return len(targets(config))


def problem(config: Mapping[str, Any]) -> Problem:
    chosen = list(targets(config))
    states = [centroid(chosen)] + chosen
    return Problem(
        states,
        int(configuration.get(config, "instance.num_chasers")),
        float(
            configuration.duration(
                configuration.get(config, "instance.max_time")
            )
        ),
        int(configuration.get(config, "instance.max_load")),
    )


# ============================================================================ #
# Layers
# ============================================================================ #


def spec(node: Any) -> Tuple[str, Dict[str, Any]]:
    """The `type` and the `args` of whatever a configuration names.

    One shape for a solver and for a layer alike, which is what lets a solver's
    arguments hold layers written exactly like the solver itself. A key beside
    the two is an error rather than something quietly ignored: `grid_size`
    written at the top of a layer is a configuration that meant to say
    `args: {grid_size: ...}`, and saying so is cheaper than a silent default.
    """
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
    return spec(configuration.get(config, f"{SOLVER}.args.{name}"))


def solver_type(config: Mapping[str, Any]) -> str:
    """The solver a configuration names, `mcrdv` when it names none."""
    return spec(configuration.get(config, SOLVER))[0] or "mcrdv"


def register(name: str) -> Callable[[Runner], Runner]:
    """Register a solver under `name`, as a decorator.

    A solver is `(config) -> result`: it reads a configuration, solves whatever
    that describes, and returns the columns worth keeping. Registering one is
    all the harness needs -- sweeping, chunking, recording and plotting do not
    know what a solver is.
    """

    def keep(runner: Runner) -> Runner:
        SOLVERS[name] = runner
        return runner

    return keep


def _pick(table: Dict[str, Any], kind: str, default: str, what: str) -> Any:
    """What is registered under `kind`, or the default when none is named."""
    chosen = kind or default
    if chosen not in table:
        known = ", ".join(sorted(table))
        raise ValueError(f"unknown {what} type '{chosen}' (known: {known})")
    return table[chosen]


def transfer(config: Mapping[str, Any]) -> TransferLayer:
    kind, args = layer(config, "transfer")
    return _pick(TRANSFERS, kind, "qlaw", "transfer")(**args)


def delta_v(config: Mapping[str, Any]) -> TransferLayer:
    """The configured Q-law with its objective stripped: it prices delta-v.

    Plans built under different cost oracles are only comparable on the same
    physical scale, so this re-reads them with the thrust and the estimate the
    run flew with, and nothing it was optimizing. Only those two knobs are
    kept: what an oracle weighs, forbids or hurries is exactly what this is
    stripping away.
    """
    _, args = layer(config, "transfer")
    physics = {k: v for k, v in args.items() if k in ("thrust", "factor")}
    return QlawTransfer(**physics)


def schedule(config: Mapping[str, Any]) -> ScheduleLayer:
    kind, args = layer(config, "schedule")
    inner = transfer(config)
    return _pick(SCHEDULES, kind, "dp", "schedule")(transfer=inner, **args)


def hgs_config(node: Any, seed: int) -> HgsConfig:
    """Build the search configuration, with the run's seed forced in.

    The seed belongs to the run, not to the search knobs: replications are an
    axis of a campaign, so a `seed:` written among the sequence layer's
    arguments would silently make every replication identical.

    Every other knob is taken as written, including `log_save` -- whether a run
    keeps its search log is a property of the experiment being run, so it
    belongs in the configuration that describes it and nowhere else. The one
    imposition is that the search stays quiet unless asked: a run reports
    through the driver, and a per-iteration log interleaved with that helps
    nobody.
    """
    knobs = copy.deepcopy(dict(node)) if isinstance(node, Mapping) else {}
    knobs.pop("seed", None)
    for key in _DURATIONS:
        if key in knobs:
            knobs[key] = configuration.duration(knobs[key])
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
    seed = int(configuration.get(config, "seed", 0))

    if cls is HgsSequence:
        return HgsSequence(schedule=inner, config=hgs_config(args, seed))

    # A campaign that sweeps the sequence layer shares one base configuration,
    # so a layer with no search of its own is handed search knobs that mean
    # nothing to it. Dropping the ones that belong to the search is what lets
    # `hgs` and `cluster` be two values of a single axis; anything else is
    # still passed through, and still an error if it is wrong.
    knobs = {k: v for k, v in args.items() if k not in _SEARCH_KNOBS}
    return cls(inner, **knobs)


# ============================================================================ #
# Warm-up
# ============================================================================ #


# The warm-up instance: the smallest catalogue, solved only to trigger
# compilation.
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
    """Pay the numba compilation of every stack this task will use.

    Compiling costs eight to twelve seconds and is then free for the rest of the
    process. Doing it here, on a throwaway four-target instance, keeps it out of
    the measured runs -- otherwise the first run of a task carries ten seconds
    that have nothing to do with the instance it reports on.

    Every distinct combination of layer types is warmed, not just the first: an
    experiment comparing two sequence layers compiles two separate kernels, and
    warming only one leaves the other's compilation inside a measured run.

    Configurations naming another solver are skipped: there is nothing here to
    compile for them, and whatever they need is their own business.
    """
    instance = problem(_WARMUP)
    stacks = dict.fromkeys(
        _shape(config) for config in configs if solver_type(config) == "mcrdv"
    )
    for shape in stacks:
        warm = configuration.derive(
            _WARMUP,
            solver={
                "type": "mcrdv",
                "args": {
                    "transfer": {"type": shape[0]},
                    "schedule": {"type": shape[1]},
                    # The shallowest preset compiles the whole search; it
                    # simply does not enter the genetic loop.
                    "sequence": {"type": shape[2], "args": {"preset": 0}},
                },
            },
        )
        stack = sequence(warm)
        stack.initialize(instance)
        stack.evaluate()


# ============================================================================ #
# Running
# ============================================================================ #


def solve(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Run the solver a configuration names, and return what it produced.

    The one entry point an experiment's `execute` needs: which solver runs is
    `solver.type`, and what it is given is `solver.args`.
    """
    return _pick(SOLVERS, solver_type(config), "mcrdv", "solver")(config)


@register("mcrdv")
def mcrdv(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the stack from `solver.args`, solve, and report the usual numbers.

    Setup and search are timed apart because they scale differently: setup
    builds the table of per-leg bounds, which is quadratic in the catalogue size
    and reaches twenty seconds on the largest one, while the search is what the
    paper is about. Summing them would hide the term that actually grows.

    The configuration decides everything, this function nothing: a run that
    wants its search log kept says `log_save: true` among the sequence layer's
    arguments, and the returned trace is empty otherwise. That keeps what an
    experiment does readable from the file that describes it, rather than split
    between a configuration and the argument some script happened to pass.
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
        # Genetic iterations completed, `None` for a layer with no search. At a
        # fixed budget this is the search's throughput.
        "iterations": getattr(
            getattr(stack, "context", None), "iteration", None
        ),
        # Empty unless a trace was asked for, so this needs no condition: the
        # layer simply kept nothing.
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

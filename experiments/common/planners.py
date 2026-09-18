"""Turning configurations into mission planners, and the problem it solves."""

from __future__ import annotations

import copy
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from mcrdv import Problem, centroid, load_states
from mcrdv.orbit import KepState
from mcrdv.planner import ScheduleLayer, SequenceLayer, TransferLayer
from mcrdv.schedule import DpSchedule
from mcrdv.sequence import ClusterSequence, HgsSequence
from mcrdv.sequence.hgs import HgsConfig
from mcrdv.transfer import QlawTransfer

from . import config as configuration

# Catalogues live at the repository root, beside `experiments/`.
DATA = Path(__file__).resolve().parents[2] / "data"

# Names accepted in a configuration, per layer.
TRANSFERS = {"qlaw": QlawTransfer}
SCHEDULES = {"dp": DpSchedule}
SEQUENCES = {"hgs": HgsSequence, "cluster": ClusterSequence}

# Search knobs that are times, and so accept `60s` / `5m` style strings.
_DURATIONS = ("max_time",)


# ============================================================================ #
# Problems
# ============================================================================ #


@lru_cache(maxsize=None)
def _states(name: str) -> Tuple[KepState, ...]:
    """The catalogue `name`, read once per process."""
    return tuple(load_states(DATA / f"{name}.csv"))


def num_targets(config: Mapping[str, Any]) -> int:
    return len(_states(str(configuration.get(config, "instance.name"))))


def problem(config: Mapping[str, Any]) -> Problem:
    targets = list(_states(str(configuration.get(config, "instance.name"))))
    states = [centroid(targets)] + targets
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


def _split(node: Any) -> Tuple[str, Dict[str, Any]]:
    """A layer's `type` and its remaining keyword arguments."""
    if node is None:
        node = {}
    if not isinstance(node, Mapping):
        raise TypeError(f"expected a layer mapping, got {node!r}")
    knobs = copy.deepcopy(dict(node))
    return str(knobs.pop("type", "")), knobs


def _pick(table: Dict[str, type], kind: str, default: str, what: str) -> type:
    """The class registered under `kind`, or the default when none is named."""
    chosen = kind or default
    if chosen not in table:
        known = ", ".join(sorted(table))
        raise ValueError(f"unknown {what} type '{chosen}' (known: {known})")
    return table[chosen]


def transfer(config: Mapping[str, Any]) -> TransferLayer:
    kind, knobs = _split(config.get("transfer"))
    return _pick(TRANSFERS, kind, "qlaw", "transfer")(**knobs)


def schedule(config: Mapping[str, Any]) -> ScheduleLayer:
    kind, knobs = _split(config.get("schedule"))
    inner = transfer(config)
    return _pick(SCHEDULES, kind, "dp", "schedule")(transfer=inner, **knobs)


def hgs_config(node: Any, seed: int) -> HgsConfig:
    """Build the search configuration, with the run's seed forced in.

    The seed belongs to the run, not to the search knobs: replications are an
    axis of a campaign, so a `seed:` written under `sequence.config` would
    silently make every replication identical.

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
    kind, knobs = _split(config.get("sequence"))
    cls = _pick(SEQUENCES, kind, "hgs", "sequence")
    seed = int(configuration.get(config, "seed", 0))

    if cls is HgsSequence:
        node = knobs.pop("config", {}) or {}
        if "preset" in knobs:
            node = {**node, "preset": knobs.pop("preset")}
        return HgsSequence(
            schedule=inner, config=hgs_config(node, seed), **knobs
        )

    # A campaign that sweeps `sequence.type` shares one base configuration, so a
    # layer with no search of its own is handed search knobs that mean nothing
    # to it. Dropping them is what lets `hgs` and `cluster` be two values of a
    # single axis; anything else is still passed through, and still an error if
    # it is wrong.
    knobs.pop("preset", None)
    knobs.pop("config", None)
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
    return tuple(
        str(configuration.get(config, f"{layer}.type", ""))
        for layer in ("transfer", "schedule", "sequence")
    )


def warm_up(configs: Sequence[Mapping[str, Any]]) -> None:
    """Pay the numba compilation of every stack this task will use.

    Compiling costs eight to twelve seconds and is then free for the rest of the
    process. Doing it here, on a throwaway four-target instance, keeps it out of
    the measured runs -- otherwise the first run of a task carries ten seconds
    that have nothing to do with the instance it reports on.

    Every distinct combination of layer types is warmed, not just the first: an
    experiment comparing two sequence layers compiles two separate kernels, and
    warming only one leaves the other's compilation inside a measured run.
    """
    instance = problem(_WARMUP)
    for shape in dict.fromkeys(_shape(config) for config in configs):
        warm = {
            **_WARMUP,
            "transfer": {"type": shape[0]},
            "schedule": {"type": shape[1]},
            # The shallowest preset compiles the whole search; it simply does
            # not enter the genetic loop.
            "sequence": {"type": shape[2], "preset": 0},
        }
        layer = sequence(warm)
        layer.initialize(instance)
        layer.evaluate()


# ============================================================================ #
# Running
# ============================================================================ #


def solve(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the stack, solve, and report the usual mcrdv numbers.

    Setup and search are timed apart because they scale differently: setup
    builds the table of per-leg bounds, which is quadratic in the catalogue size
    and reaches twenty seconds on the largest one, while the search is what the
    paper is about. Summing them would hide the term that actually grows.

    The configuration decides everything, this function nothing: a run that
    wants its search log kept says `log_save: true` under `sequence.config`,
    and the returned trace is empty otherwise. That keeps what an experiment
    does readable from the file that describes it, rather than split between a
    configuration and the argument some script happened to pass.
    """
    layer = sequence(config)
    instance = problem(config)

    start = time.perf_counter()
    layer.initialize(instance)
    setup = time.perf_counter() - start

    start = time.perf_counter()
    mission = layer.evaluate()
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
            for e in (getattr(layer, "trace", ()) or ())
        ],
    }

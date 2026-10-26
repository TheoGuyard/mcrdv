"""Running experiments: configurations, campaigns, the `run.py` loop, records.

Nothing here knows what an experiment solves.

- A **configuration** is one YAML file describing exactly one run: a nested
  mapping addressed by dotted paths (`solver.args.schedule.args.grid_size`),
  with durations that may carry a unit (`90s`, `5m`, `2h`, `365d`, `3y`).
- A **campaign** expands a sweep of `sweeps.yaml` into many configurations.
- `main` is the loop behind every `run.py`; it writes one **record** (a pickle
  of the configuration and the result) per run.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import math
import pickle
import secrets
import string
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
)

import yaml

# ============================================================================ #
# Configurations
# ============================================================================ #

_UNITS = {
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86_400.0,
    "y": 365.0 * 86_400.0,
}


def duration(value: Any) -> float:
    """Seconds from a number or a suffixed string such as `365d`."""
    if isinstance(value, bool):
        raise TypeError("a duration cannot be a boolean")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if text in ("inf", "infinity", "none"):
        return math.inf
    unit = _UNITS.get(text[-1:])
    if unit is None:
        return float(text)
    return float(text[:-1]) * unit


def get(tree: Mapping[str, Any], path: str, default: Any = None) -> Any:
    """Read a dotted `path`, or `default` where it does not exist."""
    node: Any = tree
    for key in path.split("."):
        if not isinstance(node, Mapping) or key not in node:
            return default
        node = node[key]
    return node


def put(tree: Dict[str, Any], path: str, value: Any) -> Dict[str, Any]:
    """Assign `value` at a dotted `path`, creating intermediate mappings."""
    keys = path.split(".")
    node = tree
    for key in keys[:-1]:
        nxt = node.get(key)
        if not isinstance(nxt, dict):
            nxt = {} if nxt is None else dict(nxt)
            node[key] = nxt
        node = nxt
    node[keys[-1]] = value
    return tree


def derive(base: Mapping[str, Any], **overrides: Any) -> Dict[str, Any]:
    """A deep copy of `base` with dotted `overrides` applied, passed as
    `**{"a.b.c": value}`. Values are copied too, so configurations sharing a
    YAML anchor never share an object."""
    out = copy.deepcopy(dict(base))
    for path, value in overrides.items():
        put(out, path, copy.deepcopy(value))
    return out


def flatten(tree: Any, prefix: str = "") -> Dict[str, Any]:
    """A nested mapping as dotted keys, keeping only the leaves a table can
    hold (no plans, arrays or traces)."""
    out: Dict[str, Any] = {}
    if not isinstance(tree, Mapping):
        return out
    for key, value in tree.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(flatten(value, f"{path}."))
        elif value is None or isinstance(value, (str, int, float, bool)):
            out[path] = value
    return out


def load_config(path: Path) -> Dict[str, Any]:
    """Read one configuration file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: a configuration must be a mapping")
    return raw


def save_config(path: Path, config: Mapping[str, Any]) -> Path:
    """Write one configuration file, keys in the order they were built."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            dict(config), handle, sort_keys=False, default_flow_style=False
        )
    return path


# YAML files that sit beside configurations without being one.
NOT_A_RUN = ("manifest.yaml", "plot.yaml")


def collect(sources: Iterable[Path], default: Path) -> list[Path]:
    """The configuration files named on the command line: files, campaign
    directories (`run-NNNN.yaml`) or folders of YAML files, in sorted order;
    `default` when nothing is named."""
    found: list[Path] = []
    for source in sources:
        source = Path(source)
        if source.is_dir():
            batch = sorted(source.glob("run-*.yaml"))
            if not batch:
                batch = [
                    p
                    for p in sorted(source.glob("*.yaml"))
                    if p.name not in NOT_A_RUN
                ]
            if not batch:
                raise SystemExit(f"{source} holds no configuration files")
            found.extend(batch)
        elif source.exists():
            found.append(source)
        else:
            raise SystemExit(f"{source} does not exist")

    if not found:
        if not Path(default).exists():
            raise SystemExit(f"{default} missing, and no configuration given")
        found = [Path(default)]
    return found


def share(configs: list[Path], task_id: int, num_tasks: int) -> list[Path]:
    """The configurations of one array task, taken by stride so that cheap
    and expensive runs spread evenly over the tasks."""
    if num_tasks < 1:
        raise ValueError("'num_tasks' must be positive")
    if not 0 <= task_id < num_tasks:
        raise ValueError(f"task id {task_id} outside [0, {num_tasks})")
    return [
        path
        for index, path in enumerate(configs)
        if index % num_tasks == task_id
    ]


# ============================================================================ #
# Campaigns: sweeps.yaml
# ============================================================================ #
#
#     sweeps:
#       myexperiment:
#         tasks: 10                 # cluster resources; `defaults` fills in
#         base: {...}               # written as a run.yaml is
#         set: {dotted.path: value} # applied to `base` once
#         campaign:                 # arms, concatenated
#           - axes:
#               set: [{...}, {...}] # bundles of values moving together
#               some.dotted.path: [1, 2, 3]
#               seed: {range: 5}
#             mode: grid            # or one_at_a_time


def load_sweeps(path: Path) -> Dict[str, Dict[str, Any]]:
    """Every sweep of `sweeps.yaml`, with `defaults` filled in."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path} does not exist")
    document = load_config(path)
    defaults = document.get("defaults") or {}
    out = {}
    for name, sweep in (document.get("sweeps") or {}).items():
        merged = {**defaults, **(sweep or {})}
        merged.setdefault("experiment", name)
        out[name] = merged
    return out


def values(spec: Any) -> List[Any]:
    """One axis's values: a list as written, or `{range: ...}`."""
    if isinstance(spec, Mapping):
        if set(spec) != {"range"}:
            raise SystemExit(f"an axis is a list or a range, got {spec!r}")
        bounds = spec["range"]
        bounds = [bounds] if isinstance(bounds, int) else list(bounds)
        return list(range(*bounds))
    if not isinstance(spec, list):
        raise SystemExit(f"an axis is a list or a range, got {spec!r}")
    return list(spec)


def expand(base: Mapping[str, Any], sweep: Mapping[str, Any]) -> Iterator:
    """Every configuration a sweep's campaign describes."""
    fixed = sweep.get("set") or {}
    if not isinstance(fixed, Mapping):
        raise SystemExit(f"a sweep's 'set' is a mapping, got {fixed!r}")
    base = derive(base, **fixed)

    for arm in sweep.get("campaign") or []:
        axes = dict((arm or {}).get("axes") or {})
        bundles = [{}]
        if "set" in axes:
            bundles = [dict(bundle) for bundle in axes.pop("set")]
        axes = {path: values(spec) for path, spec in axes.items()}
        mode = str((arm or {}).get("mode", "grid"))

        for bundle in bundles:
            start = derive(base, **bundle)
            if mode == "grid":
                yield from grid(start, **axes)
            elif mode == "one_at_a_time":
                replicates = axes.pop("seed", [0])
                yield from one_at_a_time(start, seed=replicates, **axes)
            else:
                raise SystemExit(
                    f"unknown mode '{mode}' (known: grid, one_at_a_time)"
                )


def grid(
    base: Mapping[str, Any], **axes: Sequence[Any]
) -> Iterator[Dict[str, Any]]:
    """Every combination of the axes (dotted paths), the first varying
    slowest so that the replications of one setting stay together."""
    if not axes:
        yield derive(base)
        return
    paths = list(axes)
    choices = [list(axes[path]) for path in paths]
    for combination in itertools.product(*choices):
        yield derive(base, **dict(zip(paths, combination)))


def one_at_a_time(
    base: Mapping[str, Any], seed: Sequence[Any] = (0,), **axes: Sequence[Any]
) -> Iterator[Dict[str, Any]]:
    """Vary one axis at a time from `base`, replicated over `seed`. Each
    configuration records what moved under `sweep.axis` and `sweep.value`."""
    for path, choices in axes.items():
        for value in choices:
            for replicate in seed:
                yield derive(
                    base,
                    **{
                        path: value,
                        "seed": replicate,
                        "sweep.axis": path,
                        "sweep.value": value,
                    },
                )


def write_campaign(
    out: Path, experiment: str, configs: Sequence[Mapping[str, Any]]
) -> Path:
    """Write a campaign: one `run-NNNN.yaml` per configuration, and a
    manifest counting them."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)

    for index, config in enumerate(configs, start=1):
        save_config(out / f"run-{index:04d}.yaml", config)

    save_config(
        out / "manifest.yaml",
        {
            "experiment": experiment,
            "generated": datetime.now().isoformat(timespec="seconds"),
            "configs": len(configs),
        },
    )
    return out


# ============================================================================ #
# Records: one pickle per run
# ============================================================================ #

# Records are named `YYYYMMDD_HHMMSS_xxxxxxxx`: time-ordered, and unique
# without coordination. Saved figures are named the same way.
STAMP = "%Y%m%d_%H%M%S"
TOKEN = 8


@dataclass
class RunRecord:
    """One run: what it was asked to do, and what came of it."""

    # Stem of the file this record lives in.
    run_id: str
    # The resolved configuration, copied verbatim.
    config: Dict[str, Any] = field(default_factory=dict)
    # Whatever the experiment's `execute` returned (or `error`).
    result: Dict[str, Any] = field(default_factory=dict)

    def row(self) -> Dict[str, Any]:
        """Configuration and result as one flat row of dotted keys."""
        row = {"run_id": self.run_id}
        row.update(flatten(self.config))
        row.update(flatten(self.result))
        return row


def token(length: int = TOKEN) -> str:
    """Random lowercase letters, from `secrets` so that array tasks started
    at the same instant never share a sequence."""
    return "".join(
        secrets.choice(string.ascii_lowercase) for _ in range(length)
    )


def new_name(when: datetime | None = None) -> str:
    """A fresh name, `YYYYMMDD_HHMMSS_xxxxxxxx`."""
    when = when or datetime.now()
    return f"{when.strftime(STAMP)}_{token()}"


def write_record(
    out: Path, config: Dict[str, Any], result: Dict[str, Any]
) -> Path:
    """Write one record, returning where it landed."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    while True:
        run_id = new_name()
        path = out / f"{run_id}.pkl"
        if not path.exists():
            break

    record = RunRecord(run_id=run_id, config=config, result=result)
    with open(path, "wb") as handle:
        pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


class _Unpickler(pickle.Unpickler):
    """Reads records written when `RunRecord` lived in `_common.record`."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "_common.record":
            module = __name__
        return super().find_class(module, name)


def read_record(path: Path) -> RunRecord:
    """Read one record back."""
    with open(path, "rb") as handle:
        return _Unpickler(handle).load()


def record_paths(source: Path) -> List[Path]:
    """Every record under `source`, oldest first."""
    return sorted(Path(source).glob("*.pkl"))


def load_records(source: Path) -> Iterator[RunRecord]:
    """Every record under `source`; one that cannot be read is reported and
    skipped rather than stopping the load."""
    for path in record_paths(source):
        try:
            yield read_record(path)
        except Exception as error:
            print(f"skipping {path.name}: {type(error).__name__}: {error}")


# ============================================================================ #
# The loop behind every run.py
# ============================================================================ #

# `(config) -> result`: perform one run and return whatever is worth keeping.
Execute = Callable[[Dict[str, Any]], Dict[str, Any]]
# `(configs) -> None`: prepare the process before any run is measured.
Warmup = Callable[[Sequence[Dict[str, Any]]], None]


def parser(description: str = "") -> argparse.ArgumentParser:
    """The command line every experiment shares."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "configs",
        type=Path,
        nargs="*",
        help="config files or a campaign directory "
        "(default: run.yaml beside run.py)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where records go (default: results/ beside run.py)",
    )
    p.add_argument(
        "--task-id",
        type=int,
        default=0,
        help="index of this task within the job array",
    )
    p.add_argument(
        "--num-tasks",
        type=int,
        default=1,
        help="number of tasks sharing the work",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="list the configurations that would run, and stop",
    )
    return p


def summarise(result: Dict[str, Any]) -> str:
    """The first few scalar results, for the progress line."""
    row = flatten(result)
    if not row:
        return "done"
    parts = []
    for key, value in list(row.items())[:4]:
        parts.append(
            f"{key}={value:.4g}"
            if isinstance(value, float)
            else f"{key}={value}"
        )
    return " ".join(parts)


def main(
    script: str,
    execute: Execute,
    warmup: Optional[Warmup] = None,
    argv: Optional[Sequence[str]] = None,
) -> int:
    """Run this task's share of an experiment: collect the configurations,
    warm up once, then `execute` each one and write its record (a run that
    raises is recorded with its `error`). `script` is the calling `run.py`,
    which locates `run.yaml` and `results/`."""
    here = Path(script).resolve().parent
    args = parser(here.name).parse_args(argv)

    out = args.out or here / "results"
    paths = collect(args.configs, here / "run.yaml")
    mine = share(paths, args.task_id, args.num_tasks)

    print(
        f"{here.name}: {len(paths)} configurations, {len(mine)} in this task"
    )

    if args.dry_run:
        for path in mine:
            print(f"  {path}")
        return 0

    configs = [load_config(path) for path in mine]

    if warmup is not None:
        start = time.perf_counter()
        warmup(configs)
        print(f"warmed up in {time.perf_counter() - start:.1f}s", flush=True)

    written = 0
    failed = 0
    for position, (path, config) in enumerate(zip(mine, configs), start=1):
        print(f"  {position}/{len(mine)} {path.name}", end="", flush=True)
        start = time.perf_counter()
        try:
            result = execute(config)
        except Exception as error:
            failed += 1
            result = {
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            print(f"  -> {result['error']}", flush=True)
        else:
            elapsed = time.perf_counter() - start
            print(f"  -> {summarise(result)} [{elapsed:.2f}s]", flush=True)

        write_record(out, config, result)
        written += 1

    print(
        f"wrote {written} records to {out}"
        + (f", {failed} failed" if failed else "")
    )
    return 0

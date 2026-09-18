"""Configuration files: one file describes exactly one run.

The core treats a configuration as an opaque mapping. Nothing here knows what a
transfer layer is, or that `sequence.preset` means anything -- that belongs to
`planners.py`, so an experiment driving something other than mcrdv can use its
own schema and still get sweeping, chunking, recording and plotting for free.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml

# Duration suffixes accepted wherever a time is configured.
_UNITS = {
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86_400.0,
    "y": 365.0 * 86_400.0,
}


def duration(value: Any) -> float:
    """Seconds from a number or a suffixed string such as `365d` or `1.5h`.

    Mission horizons are naturally written in days and search budgets in
    minutes; spelling both as raw seconds makes a configuration unreadable and
    easy to get wrong by a factor of sixty.
    """
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


# ============================================================================ #
# Dotted paths
# ============================================================================ #


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
    """A deep copy of `base` with dotted `overrides` applied.

    Keyword names cannot contain dots, so callers pass nested paths through
    `**{"sequence.config.crossover": "ox"}`; plain names work as usual.
    """
    out = copy.deepcopy(dict(base))
    for path, value in overrides.items():
        put(out, path, value)
    return out


def flatten(tree: Any, prefix: str = "") -> Dict[str, Any]:
    """A nested mapping as dotted keys, for filtering and tabulating.

    Only leaves that a table could hold are kept. A `MissionPlan`, a numpy array
    or a list of log entries is dropped rather than stringified: it stays
    reachable on the record itself, and a column full of `<object at 0x...>`
    helps nobody.
    """
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


# ============================================================================ #
# Files
# ============================================================================ #


def load(path: Path) -> Dict[str, Any]:
    """Read one configuration file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: a configuration must be a mapping")
    return raw


def save(path: Path, config: Mapping[str, Any]) -> Path:
    """Write one configuration file, keys in the order they were built."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            dict(config), handle, sort_keys=False, default_flow_style=False
        )
    return path


# Files that live beside configurations but are not one.
NOT_A_RUN = ("manifest.yaml", "plot.yaml")


def collect(sources: Iterable[Path], default: Path) -> list[Path]:
    """The configuration files named by command-line arguments.

    A source may be a file, or a directory: a campaign directory of
    `run-NNNN.yaml`, or an experiment folder holding a single `run.yaml`. With
    no source at all the experiment's own `run.yaml` is used, which is the
    one-shot case. Directories are read in sorted order, so a task's share of
    them is the same on every machine.
    """
    found: list[Path] = []
    for source in sources:
        source = Path(source)
        if source.is_dir():
            batch = sorted(source.glob("run-*.yaml"))
            if not batch:
                # Not a campaign directory. Take every configuration in it,
                # minus the files that sit alongside one without being one --
                # `plot.yaml` in an experiment folder would otherwise be run as
                # if it described a solve.
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
    """The share of `configs` belonging to one array task.

    Strided rather than contiguous: runs differ in cost by orders of magnitude
    -- a thirteen-target instance solves in well under a second, a
    583-target one takes minutes -- so a stride spreads the expensive ones over
    the tasks instead of piling them into the last block.
    """
    if num_tasks < 1:
        raise ValueError("'num_tasks' must be positive")
    if not 0 <= task_id < num_tasks:
        raise ValueError(f"task id {task_id} outside [0, {num_tasks})")
    return [
        path
        for index, path in enumerate(configs)
        if index % num_tasks == task_id
    ]

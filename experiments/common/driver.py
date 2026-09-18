"""The run loop every experiment delegates to.

It knows nothing about what an experiment does. It collects configuration files,
takes this task's share of them, calls the experiment's `execute` once per
configuration, writes a record, and keeps one failure from killing the rest of
the chunk. Everything solver-specific -- building a problem, timing a solve,
paying a compiler up front -- happens inside `execute`, or in the optional
`warmup` hook.
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from . import config as configuration
from . import record as recording

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
    row = configuration.flatten(result)
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
    """Run this task's share of an experiment.

    Parameters
    ----------
    script : str
        Script calling `run.py`, normally `__file__`, to locate the experiment.
    execute : callable
        `(config) -> result`. Performs one run. Anything it returns is kept.
    warmup : callable, optional
        `(configs) -> None`, run once before the first measured run.
    """
    here = Path(script).resolve().parent
    args = parser(here.name).parse_args(argv)

    out = args.out or here / "results"
    paths = configuration.collect(args.configs, here / "run.yaml")
    mine = configuration.share(paths, args.task_id, args.num_tasks)

    print(
        f"{here.name}: {len(paths)} configurations, {len(mine)} in this task"
    )

    if args.dry_run:
        for path in mine:
            print(f"  {path}")
        return 0

    configs = [configuration.load(path) for path in mine]

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

        recording.write(out, config, result)
        written += 1

    print(
        f"wrote {written} records to {out}"
        + (f", {failed} failed" if failed else "")
    )
    return 0

"""Turning one configuration into the many a campaign needs.

Sweeping is deliberately not something `run.py` knows about. A sweep script in
`experiments/sweeps/` reads an experiment's `run.yaml`, yields the
variations it wants, and writes them to a disposable directory; the runner then
treats each one as an ordinary single-run configuration.

The generated files are transient. Nothing is lost by deleting them once a
campaign has finished, because every configuration is copied verbatim into the
record of the run it produced.
"""

from __future__ import annotations

import argparse
import itertools
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
)

from . import config as configuration

# Where campaigns are staged, beside the experiments rather than inside one, so
# that clearing a campaign can never touch a result.
TMP = Path(__file__).resolve().parents[1] / "tmp"

# `(base) -> configs`: the variations a campaign wants.
Campaign = Callable[[Dict[str, Any]], Iterable[Dict[str, Any]]]


# ============================================================================ #
# Sweep mechanics
# ============================================================================ #


def grid(
    base: Mapping[str, Any], **axes: Sequence[Any]
) -> Iterator[Dict[str, Any]]:
    """Every combination of the given axes, as full configurations.

    Axis names are dotted paths, so nested knobs are reachable:
    `grid(base, **{"schedule.grid_size": [8, 32], "seed": range(3)})`.
    The first axis varies slowest, which keeps the replications of one setting
    together and a partial campaign therefore still useful.
    """
    if not axes:
        yield configuration.derive(base)
        return
    paths = list(axes)
    values = [list(axes[path]) for path in paths]
    for combination in itertools.product(*values):
        yield configuration.derive(base, **dict(zip(paths, combination)))


def one_at_a_time(
    base: Mapping[str, Any], seed: Sequence[Any] = (0,), **axes: Sequence[Any]
) -> Iterator[Dict[str, Any]]:
    """Vary one axis at a time from `base`, replicated over `seed`.

    What a sensitivity study actually wants: a full factorial over a dozen
    search knobs is unaffordable, and the question being asked is "does this
    choice matter, and which value should I use", not "how do these interact".

    Every value of an axis is emitted, including the one `base` already holds,
    so each axis yields a complete curve that can be plotted on its own. Each
    configuration records which knob moved, under `sweep.axis` and `sweep.value`,
    so the results table can be grouped without re-deriving it.
    """
    for path, values in axes.items():
        for value in values:
            for replicate in seed:
                config = configuration.derive(
                    base,
                    **{
                        path: value,
                        "seed": replicate,
                        "sweep.axis": path,
                        "sweep.value": value,
                    },
                )
                yield config


# ============================================================================ #
# Writing a campaign
# ============================================================================ #


def directory(experiment: str, when: Optional[datetime] = None) -> Path:
    """A fresh campaign directory, `tmp/<experiment>_YYYYMMDD_HHMMSS`."""
    when = when or datetime.now()
    return TMP / f"{experiment}_{when.strftime('%Y%m%d_%H%M%S')}"


def write(
    out: Path,
    experiment: str,
    configs: Sequence[Mapping[str, Any]],
    force: bool = False,
) -> Path:
    """Write a campaign's configurations and its manifest."""
    out = Path(out)
    if out.exists() and any(out.iterdir()) and not force:
        raise SystemExit(
            f"{out} is not empty; pass --force to write into it anyway"
        )
    out.mkdir(parents=True, exist_ok=True)

    for index, config in enumerate(configs, start=1):
        configuration.save(out / f"run-{index:04d}.yaml", config)

    # The manifest is what makes a coverage check possible after the fact
    # without the configurations having to be permanent: while this directory
    # still exists, the records can be counted against what was asked for.
    configuration.save(
        out / "manifest.yaml",
        {
            "experiment": experiment,
            "generated": datetime.now().isoformat(timespec="seconds"),
            "configs": len(configs),
        },
    )
    return out


def main(
    script: str,
    experiment: str,
    campaign: Campaign,
    argv: Optional[Sequence[str]] = None,
) -> int:
    """The command line every sweep script shares.

    Prints the campaign directory on the last line, and nothing else under
    `--quiet`, so it can be captured straight into a shell variable.
    """
    parser = argparse.ArgumentParser(
        description=f"generate a {experiment} campaign"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where to write (default: a fresh tmp/ directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="count and list the configurations, write nothing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="write into a directory that is not empty",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print only the campaign directory",
    )
    args = parser.parse_args(argv)

    root = Path(script).resolve().parents[1]
    base = configuration.load(root / experiment / "run.yaml")
    configs = list(campaign(base))

    if args.dry_run:
        print(f"{experiment}: {len(configs)} configurations")
        for index, config in enumerate(configs, start=1):
            flat = configuration.flatten(config)
            shown = {
                k: v
                for k, v in flat.items()
                if k not in ("name", "description")
            }
            print(f"  run-{index:04d}.yaml  {shown}")
        return 0

    out = write(
        args.out or directory(experiment), experiment, configs, args.force
    )
    if not args.quiet:
        print(f"{experiment}: {len(configs)} configurations")
    print(out)
    return 0

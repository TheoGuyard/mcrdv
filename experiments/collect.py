#!/usr/bin/env python3
"""Status of an experiment's records, and a flat index of them.

Nothing is merged: a record already carries its own configuration, so there is
no scattered state to reassemble. What is worth doing after a campaign is
checking that every configuration actually produced something -- a job array
that loses a task otherwise leaves a perfectly healthy-looking results directory
with holes in it -- and writing a table that can be eyeballed without unpickling
anything.

    ./collect.py scaling
    ./collect.py scaling --campaign tmp/scaling_20260917_184012
    ./collect.py scaling --results $SCRATCH/mcrdv/scaling
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import config as configuration  # noqa: E402
from common import record as recording  # noqa: E402

# Columns that lead the index when present; everything else follows sorted.
LEADING = (
    "run_id",
    "name",
    "instance.name",
    "num_targets",
    "instance.chasers",
    "instance.load",
    "seed",
    "sweep.axis",
    "sweep.value",
    "sequence.type",
)
TRAILING = ("t_setup", "t_solve", "error")


def order(columns):
    """Order index columns: what a run was, what came of it, what it cost."""
    rest = set(columns)
    head = [c for c in LEADING if c in rest]
    tail = [c for c in TRAILING if c in rest]
    return head + sorted(rest - set(head) - set(tail)) + tail


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="name of the experiment folder")
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="where the records are (default: <experiment>/results)",
    )
    parser.add_argument(
        "--campaign",
        type=Path,
        default=None,
        help="a campaign directory, to check coverage against",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="index file (default: <results>/index.csv)",
    )
    args = parser.parse_args(argv)

    here = Path(__file__).resolve().parent
    results = args.results or here / args.experiment / "results"
    if not Path(results).exists():
        raise SystemExit(f"{results} does not exist")

    records = list(recording.load(Path(results)))
    if not records:
        raise SystemExit(f"no records in {results}")

    rows = [r.row() for r in records]
    out = Path(args.out or Path(results) / "index.csv")
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=order({k for row in rows for k in row}),
            restval="",
        )
        writer.writeheader()
        writer.writerows(rows)

    failed = [r for r in records if r.failed]
    print(f"{len(records)} records in {results} -> {out}")
    if failed:
        print(f"WARNING: {len(failed)} runs failed", file=sys.stderr)
        for r in failed[:5]:
            print(f"  {r.run_id}: {r.result['error']}", file=sys.stderr)

    # Coverage is only checkable while the campaign directory still exists,
    # which is exactly the moment before it gets cleared -- so check now.
    if args.campaign:
        manifest = Path(args.campaign) / "manifest.yaml"
        if not manifest.exists():
            print(f"no manifest in {args.campaign}", file=sys.stderr)
            return 1
        expected = int(configuration.load(manifest).get("configs", 0))
        if len(records) < expected:
            print(
                f"WARNING: {expected - len(records)} of {expected} configurations "
                "produced no record; a task probably died",
                file=sys.stderr,
            )
        elif len(records) > expected:
            print(
                f"note: {len(records)} records for {expected} configurations -- "
                "this directory holds runs from more than one campaign"
            )
        else:
            print(f"complete: all {expected} configurations produced a record")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

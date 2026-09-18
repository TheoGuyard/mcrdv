#!/usr/bin/env python3
"""Campaign for the smoke check: both small catalogues, both sequence layers.

Small enough to run in a minute, wide enough to exercise chunking, several
distinct layer stacks, and replication over seeds.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import generate  # noqa: E402

EXPERIMENT = "smoke"

INSTANCES = [
    {
        "name": "cosmos1408",
        "num_chasers": 4,
        "max_load": 2,
        "max_time": "365d",
    },
    {"name": "odrc", "num_chasers": 13, "max_load": 2, "max_time": "365d"},
]


def campaign(base):
    """Every catalogue, both planners, two seeds."""
    return generate.grid(
        base,
        instance=INSTANCES,
        **{"sequence.type": ["hgs", "cluster"]},
        seed=[0, 1],
    )


if __name__ == "__main__":
    raise SystemExit(generate.main(__file__, EXPERIMENT, campaign))

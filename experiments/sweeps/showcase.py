#!/usr/bin/env python3
"""Campaign for the showcase experiment: a few good solves per catalogue."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import generate  # noqa: E402

EXPERIMENT = "showcase"

INSTANCES = [
    {"name": "odrc", "num_chasers": 13, "max_load": 2, "max_time": "365d"},
    {
        "name": "iridium33",
        "num_chasers": 15,
        "max_load": 8,
        "max_time": "365d",
    },
    {
        "name": "cosmos2251",
        "num_chasers": 50,
        "max_load": 15,
        "max_time": "365d",
    },
]


def campaign(base):
    return generate.grid(base, instance=INSTANCES, seed=range(3))


if __name__ == "__main__":
    raise SystemExit(generate.main(__file__, EXPERIMENT, campaign))

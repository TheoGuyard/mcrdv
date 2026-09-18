#!/usr/bin/env python3
"""Campaign for the scaling experiment: every catalogue, replicated.

The swarm of each catalogue is part of the instance. odrc, cosmos1408 and
iridium33 reuse the settings the test suite already fixes; the two large ones
have no established swarm, and these are a starting point -- `Problem` only
requires that num_chasers * max_load >= targets.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import generate  # noqa: E402

EXPERIMENT = "scaling"

INSTANCES = [
    {
        "name": "cosmos1408",
        "num_chasers": 4,
        "max_load": 2,
        "max_time": "365d",
    },  # 4 targets
    {
        "name": "odrc",
        "num_chasers": 13,
        "max_load": 2,
        "max_time": "365d",
    },  # 13 targets
    {
        "name": "iridium33",
        "num_chasers": 15,
        "max_load": 8,
        "max_time": "365d",
    },  # 110 targets
    {
        "name": "cosmos2251",
        "num_chasers": 50,
        "max_load": 15,
        "max_time": "365d",
    },  # 583 targets
    {
        "name": "fengyun1C",
        "num_chasers": 80,
        "max_load": 30,
        "max_time": "365d",
    },  # 1847 targets
]


def campaign(base):
    """Every catalogue, five seeds each."""
    return generate.grid(base, instance=INSTANCES, seed=range(5))


if __name__ == "__main__":
    raise SystemExit(generate.main(__file__, EXPERIMENT, campaign))

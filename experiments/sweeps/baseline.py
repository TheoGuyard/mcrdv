#!/usr/bin/env python3
"""Campaign for the baseline comparison: both planners, three catalogues.

The baseline is deterministic, so its replications are identical by
construction. They are generated anyway rather than special-cased: it costs
milliseconds, and it keeps the two planners strictly comparable in the table.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import generate  # noqa: E402

EXPERIMENT = "baseline"

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
    return generate.grid(
        base,
        instance=INSTANCES,
        **{"sequence.type": ["hgs", "cluster"]},
        seed=range(5),
    )


if __name__ == "__main__":
    raise SystemExit(generate.main(__file__, EXPERIMENT, campaign))

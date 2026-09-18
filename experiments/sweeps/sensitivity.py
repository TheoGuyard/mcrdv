#!/usr/bin/env python3
"""Campaign for the sensitivity experiment.

One axis at a time, not a full factorial: the search alone has a dozen knobs and
their product is unaffordable. A one-at-a-time sweep answers the question a
reader actually has -- "does this choice matter, and which value should I use" --
without claiming to map interactions.

Held to a single mid-sized catalogue for the same reason: every axis here is
multiplied by the seeds, and this already expands to well over a hundred runs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import generate  # noqa: E402

EXPERIMENT = "sensitivity"


def campaign(base):
    return generate.one_at_a_time(
        base,
        seed=range(5),
        **{
            # The search.
            "sequence.config.crossover": ["ox", "erx", "srex"],
            "sequence.config.ordering": ["index", "raan", "nearest"],
            "sequence.config.generation": ["greedy", "random"],
            "sequence.config.nb_neighbors": [0, 5, 10, 20, 40],
            "sequence.config.pop_max": [10, 20, 40],
            # The layers below it.
            "schedule.grid_size": [8, 16, 32, 64, 128],
            "schedule.use_hints": [True, False],
            "schedule.allow_waiting": [True, False],
        },
    )


if __name__ == "__main__":
    raise SystemExit(generate.main(__file__, EXPERIMENT, campaign))

"""Shared core of the mcrdv paper experiments.

Every experiment folder holds a `run.yaml` describing exactly one run, a
`run.py` saying how to perform it, a `plot.yaml` selecting results, and a
`plot.py` drawing them. Sweeping across many runs is a separate concern and
lives in `experiments/sweeps/`.

The modules below split along one line. `config`, `record`, `driver`,
`generate` and `plotting` know nothing about what an experiment solves, so an
experiment driving some planner other than mcrdv still gets sweeping, chunking,
recording and plotting for free. `planners` and `analysis` are where knowledge
of mission planners lives.
"""

from . import config, driver, generate, planners, record

__all__ = ["config", "driver", "generate", "planners", "record"]

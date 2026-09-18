"""Shared fixtures.

The suite tests the Python package on its own terms, so every fixture here is
built from the catalogues shipped in `data/`, with the swarm chosen below.
These are the instances the whole suite works on; nothing is read from
anywhere else.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mcrdv import DAY, Problem, centroid, load_states

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# Mission horizon every fixture below is built with.
MAX_TIME = 365.0 * DAY

# The catalogues the suite works on, each with the swarm it is given as
# `(num_chasers, max_load)`. Capacity exceeds the target count in all three, so
# a feasible mission exists and the load limit still bites on some sequences.
CATALOGUES = {
    "cosmos1408": (4, 2),  # 4 targets: small enough to follow by hand
    "odrc": (13, 2),  # 13 targets: a quick instance with a tight load
    "iridium33": (15, 8),  # 110 targets: the suite's working instance
}


def catalogue(
    name: str,
    num_chasers: int | None = None,
    max_load: int | None = None,
    max_time: float = MAX_TIME,
) -> Problem:
    """Build a problem from a shipped catalogue, the depot centroid first."""
    chasers, load = CATALOGUES[name]
    targets = load_states(DATA / f"{name}.csv")
    return Problem(
        [centroid(targets)] + targets,
        chasers if num_chasers is None else num_chasers,
        max_time,
        load if max_load is None else max_load,
    )


@pytest.fixture(scope="session", params=list(CATALOGUES))
def any_catalogue(request) -> Problem:
    """Each shipped catalogue in turn, for invariants that must hold on all."""
    return catalogue(request.param)


@pytest.fixture(scope="session")
def iridium() -> Problem:
    """The mid-sized catalogue, 110 targets over 15 chasers."""
    return catalogue("iridium33")


@pytest.fixture(scope="session")
def odrc() -> Problem:
    """The small catalogue, 13 targets over 13 chasers with a load limit of 2."""
    return catalogue("odrc")

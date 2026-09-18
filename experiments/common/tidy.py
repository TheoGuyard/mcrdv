"""Reducing replications to something drawable.

Just enough group-by to keep the plotting scripts short. pandas would do this in
one line, but it is a heavy dependency to add to a project whose solver needs
only numpy and numba, and the reductions here fit in a page.

Rows come from `record.row()`, never from a file: results live in pickles, and
flattening them is `config.flatten`'s job.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple

import numpy as np


def column(rows: Sequence[Dict[str, Any]], name: str) -> np.ndarray:
    """One numeric column of a group, with missing and non-finite values gone."""
    values = [row.get(name) for row in rows]
    kept = [
        float(v)
        for v in values
        if isinstance(v, (int, float))
        and not isinstance(v, bool)
        and math.isfinite(v)
    ]
    return np.asarray(kept, dtype=np.float64)


def spread(
    rows: Sequence[Dict[str, Any]],
    name: str,
    lo: float = 25.0,
    hi: float = 75.0,
) -> Tuple[float, float, float]:
    """Median and interquartile range of a column over replications.

    Median rather than mean, and quartiles rather than a standard deviation,
    because a metaheuristic's worst seed is an outlier in the literal sense and
    should not drag the reported curve with it.
    """
    values = column(rows, name)
    if values.size == 0:
        return math.nan, math.nan, math.nan
    return (
        float(np.median(values)),
        float(np.percentile(values, lo)),
        float(np.percentile(values, hi)),
    )


def resample(
    times: Sequence[float], values: Sequence[float], grid: np.ndarray
) -> np.ndarray:
    """A step-wise trace read onto a common time grid.

    A convergence trace is a staircase: the incumbent holds its value until it
    improves. Linear interpolation would invent improvements between the
    samples, so this carries the last value forward instead, and leaves the grid
    undefined before the trace begins.
    """
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if times.size == 0:
        return np.full(grid.shape, np.nan)
    where = np.searchsorted(times, grid, side="right") - 1
    out = np.where(where >= 0, values[np.clip(where, 0, None)], np.nan)
    return out

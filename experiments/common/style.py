"""One look for every figure in the paper.

Figures are drawn to be read on screen while the experiments are running; the
paper itself redraws them from the exported tables. So this aims at legibility,
not at matching the journal's typography -- that is tikz's job.
"""

from __future__ import annotations

from typing import Any, Tuple

import matplotlib
import numpy as np

# The backend is left to matplotlib. Plotting happens where results are being
# looked at, not on a compute node, so the interactive default is the useful
# one; on a machine with no display matplotlib falls back to a file-writing
# backend by itself, and `can_display` below is how that is noticed.
import matplotlib.pyplot as plt  # noqa: E402

# Backends that can write a figure but never show one.
_HEADLESS = frozenset(
    {"agg", "pdf", "ps", "svg", "template", "cairo", "pgf", "webagg_core"}
)


# Colour-blind safe, and distinguishable in greyscale when printed.
PALETTE = (
    "#0173b2",  # blue
    "#de8f05",  # orange
    "#029e73",  # green
    "#cc78bc",  # violet
    "#ca9161",  # tan
    "#949494",  # grey
)

RC = {
    "figure.figsize": (5.0, 3.2),
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.6,
    "lines.markersize": 4,
}


def apply() -> None:
    """Install the shared style."""
    plt.rcParams.update(RC)
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=list(PALETTE))


def figure(**kwargs: Any) -> Tuple[plt.Figure, plt.Axes]:
    """A styled figure and its axes."""
    apply()
    return plt.subplots(**kwargs)


def close(fig: plt.Figure) -> None:
    """Release a figure. A plot script draws many, and matplotlib keeps every
    one it is not told to let go of."""
    plt.close(fig)


def distinct(count: int) -> list:
    """`count` colours that stay tellable apart.

    The shared palette is six colours chosen to survive colour-blindness and
    greyscale, which is the right trade for a figure with a handful of series.
    Past that no qualitative palette helps -- fifteen hues cannot all be
    distinct -- so these are sampled across a map built for exactly that, and
    the ends are trimmed because its darkest and lightest are hard to see
    against a white page.
    """
    if count <= len(PALETTE):
        return list(PALETTE[:count])
    spread = np.linspace(0.05, 0.95, count)
    return [matplotlib.colormaps["turbo"](value) for value in spread]


def can_display() -> bool:
    """Whether this machine can actually put a window on screen."""
    return matplotlib.get_backend().lower() not in _HEADLESS


def show() -> None:
    """Display every figure drawn so far, and block until they are closed."""
    plt.show()

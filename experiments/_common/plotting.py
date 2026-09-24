"""Selecting results to draw, without inventing a plotting language.

`plot.yaml` says *which* results a figure is about: where they are, how
to filter them, what to group by, what to reduce. It does not say how to draw
them, because the figures that matter here -- resampled convergence staircases,
sweeps that switch between a curve and bars depending on the axis, a plan read
leg by leg -- are not naturally declarative, and a configuration expressive
enough for them would be a worse language than the Python it replaced.

So selection is data and rendering is code. Each figure in the YAML names a
drawing function of the experiment's `plot.py` under `draw`, which receives the
rows already selected and reduced. A figure without `draw` uses the function
registered under its own `name`.

The two are kept apart so that `plot.py` never needs to know what data it is
shown. Drawing the same figure for another catalogue is a new YAML entry with
its own `name` and `source`, pointing at the same `draw` -- no Python changes.

    source: {folder: results, select: {instance.name: iridium33}}

    figures:
      - name: gap
        draw: gap
        source: {select: {solver.args.schedule.args.grid_size: 32}}

`source` says where a figure's records come from and which of them it is about:
one question, written in one place. The file's block is the default and a
figure's own is read against it, so the folder is written once and each figure
adds only what it selects.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import matplotlib
import numpy as np

from . import config as configuration
from . import export, record, style, tidy

# `(ax, data) -> None`: draw one figure from prepared data.
Figure = Callable[[Any, "Data"], None]


@dataclass
class Data:
    """What a figure function is handed.

    `rows` and `records` are always populated; the reduced arrays are filled in
    only when the figure asked for a reduction. A figure that needs the mission
    plans, or a search trace, reaches them through `records`.
    """

    # The rows that passed `select`, flattened config and result together.
    rows: List[Dict[str, Any]]
    # The records those rows came from, in the same order.
    records: List[record.RunRecord]
    # The figure's own entry from `plot.yaml`.
    spec: Dict[str, Any] = field(default_factory=dict)

    # Group keys, as numbers where they are numeric.
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # Group keys as written, for a categorical axis.
    labels: List[str] = field(default_factory=list)
    # The reduction of `value` over each group.
    median: np.ndarray = field(default_factory=lambda: np.zeros(0))
    lo: np.ndarray = field(default_factory=lambda: np.zeros(0))
    hi: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # The rows of each group, in the order `x` and `labels` are in.
    groups: List[List[Dict[str, Any]]] = field(default_factory=list)

    @property
    def numeric(self) -> bool:
        """Whether the grouping axis is numeric, and so should be a curve."""
        return self.x.size > 0 and not np.isnan(self.x).any()


# ============================================================================ #
# Selection
# ============================================================================ #


def matches(row: Mapping[str, Any], select: Mapping[str, Any]) -> bool:
    """Whether a row satisfies every clause of a `select` block.

    A scalar is an equality test and a list is membership, which covers what a
    figure actually needs without a query syntax. Missing keys never match:
    filtering on a column that does not exist is a mistake worth seeing as an
    empty figure rather than silently ignoring.
    """
    for key, wanted in select.items():
        value = row.get(key)
        if isinstance(wanted, (list, tuple, set)):
            if value not in wanted:
                return False
        elif value != wanted:
            return False
    return True


def _number(value: Any) -> float:
    """A group key as a number, or NaN when it is not one."""
    if isinstance(value, bool) or value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


# What a `source` block may hold: where the records are, and which of them a
# figure is about.
SOURCE_KEYS = {"folder", "select"}


def source(
    node: Any, default: Tuple[Any, Mapping[str, Any]] = (None, {})
) -> Tuple[Any, Dict[str, Any]]:
    """Where a figure's records come from, and which of them it is about.

    A bare string is the folder, which is the common case. A mapping gives the
    folder, the `select` that picks records out of it, or both. A figure's own
    `source` is read against the file's: the folder is written once at the top,
    each figure adds the selection it is about, and where both name a key the
    figure wins. Which records a figure describes and where they came from are
    one question, so they are written in one place.
    """
    folder, select = default
    if node is None:
        return folder, dict(select)
    if isinstance(node, str):
        return node, dict(select)
    if not isinstance(node, Mapping):
        raise SystemExit(f"a 'source' is a folder or a mapping, got {node!r}")
    unknown = sorted(set(node) - SOURCE_KEYS)
    if unknown:
        raise SystemExit(
            f"unexpected key(s) {unknown} in 'source' "
            f"(known: {', '.join(sorted(SOURCE_KEYS))})"
        )
    return (
        node.get("folder", folder),
        {**select, **(node.get("select") or {})},
    )


def prepare(
    rows: Sequence[Dict[str, Any]],
    records: Sequence[record.RunRecord],
    spec: Mapping[str, Any],
    select: Optional[Mapping[str, Any]] = None,
) -> Data:
    """Apply one figure's selection, grouping and reduction.

    The selection is normally the one `main` worked out from the file's
    `source` and the figure's; on its own, a figure's own block is used.
    """
    if select is None:
        select = source(spec.get("source"))[1]
    kept = [
        (row, rec) for row, rec in zip(rows, records) if matches(row, select)
    ]
    data = Data(
        rows=[row for row, _ in kept],
        records=[rec for _, rec in kept],
        spec=dict(spec),
    )

    group_by = spec.get("group_by")
    reduce = str(spec.get("reduce", "raw"))
    if not group_by or reduce == "raw":
        return data

    value = spec.get("value")
    if not value:
        raise SystemExit(
            f"figure '{spec.get('name')}' reduces but names no 'value'"
        )

    buckets: Dict[Any, List[Dict[str, Any]]] = {}
    for row in data.rows:
        if group_by in row:
            buckets.setdefault(row[group_by], []).append(row)

    # Sort numerically where the axis allows it, alphabetically otherwise, so a
    # curve is drawn left to right and a bar chart keeps a stable order.
    keys = list(buckets)
    if all(not math.isnan(_number(k)) for k in keys):
        keys.sort(key=_number)
    else:
        keys.sort(key=str)

    x, labels, median, lo, hi, groups = [], [], [], [], [], []
    for key in keys:
        members = buckets[key]
        centre, low, high = tidy.spread(members, value)
        if math.isnan(centre):
            continue
        if reduce == "mean":
            values = tidy.column(members, value)
            centre = float(values.mean()) if values.size else math.nan
        x.append(_number(key))
        labels.append(str(key))
        median.append(centre)
        lo.append(low)
        hi.append(high)
        groups.append(members)

    data.x = np.asarray(x, dtype=np.float64)
    data.labels = labels
    data.median = np.asarray(median, dtype=np.float64)
    data.lo = np.asarray(lo, dtype=np.float64)
    data.hi = np.asarray(hi, dtype=np.float64)
    data.groups = groups
    return data


# ============================================================================ #
# Drawing
# ============================================================================ #


def decorate(ax, spec: Mapping[str, Any]) -> None:
    """Apply the axis settings a figure declared."""
    if spec.get("xlabel"):
        ax.set_xlabel(str(spec["xlabel"]))
    if spec.get("ylabel"):
        ax.set_ylabel(str(spec["ylabel"]))
    if spec.get("title"):
        ax.set_title(str(spec["title"]))
    if spec.get("x_scale"):
        ax.set_xscale(str(spec["x_scale"]))
    if spec.get("y_scale"):
        ax.set_yscale(str(spec["y_scale"]))


def draw(
    name: str, figure: Figure, data: Data, out: Path, save: bool
) -> List[Path]:
    """Draw one figure, and write it out only when asked.

    A figure being displayed is kept open: closing it is what frees the memory
    when a script draws nine of them in a row, but it would also leave
    `style.show` with nothing to put on screen.
    """
    fig, ax = style.figure()
    figure(ax, data)
    decorate(ax, data.spec)
    if ax.get_legend_handles_labels()[0] and data.spec.get("legend", True):
        ax.legend()

    if not save:
        # A window title makes nine open figures tellable apart. Not every
        # backend provides a manager, and a missing one is no reason to fail.
        manager = getattr(fig.canvas, "manager", None)
        if manager is not None:
            manager.set_window_title(name)
        return []

    # Saved figures are named like records -- timestamp then a random suffix --
    # so a figure sorts beside the runs it was drawn from, and re-saving adds a
    # version rather than overwriting the one already cited somewhere. The
    # figure's own name survives in the `.tex` header.
    written = export.save(
        fig,
        record.name(),
        out,
        xname=str(data.spec.get("xname", "x")),
        label=name,
    )
    style.close(fig)
    return written


def main(
    script: str,
    figures: Mapping[str, Figure],
    argv: Optional[Sequence[str]] = None,
) -> int:
    """Draw every figure named in an experiment's `plot.yaml`.

    Parameters
    ----------
    script : str
        The calling `plot.py`, normally `__file__`. Locates the experiment.
    figures : mapping
        Drawing function name -> drawing function. A figure entry picks one
        with its `draw` key, or by its `name` when it has none.
    """
    here = Path(script).resolve().parent
    parser = argparse.ArgumentParser(description=f"figures for {here.name}")
    parser.add_argument("--config", type=Path, default=here / "plot.yaml")
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="override the results directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=here / "figures",
        help="where --save writes",
    )
    parser.add_argument(
        "--only", type=str, default=None, help="draw just this one figure"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="write figures to --out instead of displaying them",
    )
    args = parser.parse_args(argv)

    # Looking at a figure is the common case, so it is the default; writing the
    # pdf and the tikz table is what you do once it is worth keeping.
    if not args.save and not style.can_display():
        raise SystemExit(
            f"no display available (matplotlib backend is "
            f"{matplotlib.get_backend()}) -- pass --save to write the figures"
        )

    spec = configuration.load(args.config)
    root = source(spec.get("source"), ("results", {}))

    # Records are read once per folder, and only for the folders the figures
    # drawn actually name: a file whose figures all read `results` loads it
    # once, and one that compares two result sets loads each.
    loaded: Dict[Path, Tuple[List[record.RunRecord], List[Dict[str, Any]]]] = (
        {}
    )

    def read(folder: Any):
        path = Path(args.results or (here / str(folder)))
        if path not in loaded:
            if not path.exists():
                raise SystemExit(f"{path} missing -- run the experiment first")
            found = list(record.load(path))
            if not found:
                raise SystemExit(f"no records in {path}")
            loaded[path] = (found, [r.row() for r in found])
            print(f"{here.name}: {len(found)} records from {path}")
        return loaded[path]

    entries = spec.get("figures", [])
    # Figure names padded to a column, so the counts beside them line up.
    width = max((len(str(e.get("name", ""))) for e in entries), default=0)

    written: List[Path] = []
    drawn: List[str] = []
    for entry in entries:
        name = str(entry.get("name", ""))
        if args.only and name != args.only:
            continue
        kind = str(entry.get("draw", name))
        if kind not in figures:
            known = ", ".join(sorted(figures))
            print(
                f"  skipping '{name}': no drawing function '{kind}' "
                f"(set 'draw' to one of: {known})"
            )
            continue
        if "select" in entry:
            raise SystemExit(
                f"figure '{name}': 'select' belongs under 'source', "
                "beside the folder it selects from"
            )
        folder, select = source(entry.get("source"), root)
        records, rows = read(folder)
        data = prepare(rows, records, entry, select)
        if not data.rows:
            print(
                f"  skipping '{name}': none of the {len(rows)} records "
                "matched its selection"
            )
            continue
        drawn.append(name)
        paths = draw(name, figures[kind], data, args.out, args.save)
        written += paths
        # How much of the folder a figure is actually about: a results
        # directory holds every run ever made in it, and `select` decides which
        # of them each figure describes. The files are named by timestamp, so
        # this is also where a saved figure is tied back to its name.
        tail = f" -> {paths[0].stem}" if paths else ""
        print(f"  {name:<{width}}  {len(data.rows)}/{len(rows)} records{tail}")

    for path in written:
        try:
            shown = path.relative_to(here.parent)
        except ValueError:
            shown = path
        print(f"wrote {shown}")

    if not args.save and drawn:
        print(
            f"showing {len(drawn)}: {', '.join(drawn)}  (--save to write them)"
        )
        style.show()
    return 0

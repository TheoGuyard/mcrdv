"""Figures: `plot.yaml` selects the results, `plot.py` draws them, and every
figure is exported for pgfplots.

    source: {folder: results, select: {instance.name: iridium33}}
    figures:
      - name: gap                 # the figure's label
        draw: gap                 # drawing function of plot.py (default: name)
        source: {select: {solver.args.schedule.args.grid_size: 32}}
        group_by: grid_size       # optional reduction...
        value: cost               # ...of this column...
        reduce: median+iqr        # ...or `raw` to get the rows untouched
        xlabel: ...               # also ylabel, x_scale, y_scale, legend

A figure's `source` is read against the file's: the folder is written once,
each figure adds its own `select` (equality, or membership for a list).

Drawing functions receive a `Data` and draw with `line`, `points`, `band`,
`grouped_bars`, `span` or `series`, which draw with matplotlib *and* record
the columns drawn; with `--save`, each figure is written as `.pdf`, its data
as `.dat` and a compiling tikz skeleton as `.tex`, so the paper's figure
cannot disagree with the one on screen. Without `--save`, figures are shown.
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
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from . import runner

# `(ax, data) -> None`: draw one figure from prepared data.
Figure = Callable[[Any, "Data"], None]


# ============================================================================ #
# Style
# ============================================================================ #


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


def _new_figure(**kwargs: Any) -> Tuple[plt.Figure, plt.Axes]:
    """A figure and its axes, in the shared style."""
    plt.rcParams.update(RC)
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=list(PALETTE))
    return plt.subplots(**kwargs)


def distinct(count: int) -> list:
    """`count` colours that stay tellable apart: the palette, or samples of
    `turbo` (ends trimmed) beyond its six colours."""
    if count <= len(PALETTE):
        return list(PALETTE[:count])
    spread = np.linspace(0.05, 0.95, count)
    return [matplotlib.colormaps["turbo"](value) for value in spread]


def can_display() -> bool:
    """Whether this machine can put a window on screen."""
    return matplotlib.get_backend().lower() not in _HEADLESS


# ============================================================================ #
# Reductions
# ============================================================================ #


def column(rows: Sequence[Dict[str, Any]], name: str) -> np.ndarray:
    """One numeric column, missing and non-finite values dropped."""
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
    """Median and interquartile range of a column over replications."""
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
    """A staircase trace (an incumbent over time) read on a common grid:
    the last value is carried forward, NaN before the trace begins."""
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if times.size == 0:
        return np.full(grid.shape, np.nan)
    where = np.searchsorted(times, grid, side="right") - 1
    out = np.where(where >= 0, values[np.clip(where, 0, None)], np.nan)
    return out


# ============================================================================ #
# Selection
# ============================================================================ #


@dataclass
class Data:
    """What a drawing function receives.

    `rows` (flat config + result) and `records` (to reach plans and traces)
    are always filled; the group arrays only when the figure reduces.
    """

    rows: List[Dict[str, Any]]
    records: List[runner.RunRecord]
    # The figure's own entry from `plot.yaml`.
    spec: Dict[str, Any] = field(default_factory=dict)
    # Group keys, and median, quartiles and rows of `value` per group.
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    median: np.ndarray = field(default_factory=lambda: np.zeros(0))
    lo: np.ndarray = field(default_factory=lambda: np.zeros(0))
    hi: np.ndarray = field(default_factory=lambda: np.zeros(0))
    groups: List[List[Dict[str, Any]]] = field(default_factory=list)


def matches(row: Mapping[str, Any], select: Mapping[str, Any]) -> bool:
    """Whether a row satisfies every clause of a `select` (equality, or
    membership for a list). A missing key never matches."""
    for key, wanted in select.items():
        value = row.get(key)
        if isinstance(wanted, (list, tuple, set)):
            if value not in wanted:
                return False
        elif value != wanted:
            return False
    return True


def _as_number(value: Any) -> float:
    """A group key as a number, or NaN when it is not one."""
    if isinstance(value, bool) or value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


# What a `source` block may hold.
SOURCE_KEYS = {"folder", "select"}


def source(
    node: Any, default: Tuple[Any, Mapping[str, Any]] = (None, {})
) -> Tuple[Any, Dict[str, Any]]:
    """The folder and `select` of a `source` (a folder name or a mapping),
    read against `default`; the figure wins where both name a key."""
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
    records: Sequence[runner.RunRecord],
    spec: Mapping[str, Any],
    select: Mapping[str, Any],
) -> Data:
    """Apply one figure's selection, then its grouping and reduction (median
    and interquartile range of `value` per `group_by` key) unless `raw`."""
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

    # Numeric keys in numeric order, others alphabetically.
    keys = list(buckets)
    if all(not math.isnan(_as_number(k)) for k in keys):
        keys.sort(key=_as_number)
    else:
        keys.sort(key=str)

    x, median, lo, hi, groups = [], [], [], [], []
    for key in keys:
        members = buckets[key]
        centre, low, high = spread(members, value)
        if math.isnan(centre):
            continue
        x.append(_as_number(key))
        median.append(centre)
        lo.append(low)
        hi.append(high)
        groups.append(members)

    data.x = np.asarray(x, dtype=np.float64)
    data.median = np.asarray(median, dtype=np.float64)
    data.lo = np.asarray(lo, dtype=np.float64)
    data.hi = np.asarray(hi, dtype=np.float64)
    data.groups = groups
    return data


# ============================================================================ #
# Exporting what is drawn
# ============================================================================ #


class Series(NamedTuple):
    """One drawn thing, and the columns that reproduce it."""

    # Identity of the series, and the stem of its file when it needs its own.
    name: str
    # Shared abscissa of the columns.
    x: np.ndarray
    # Column name -> values, all as long as `x`.
    columns: Dict[str, np.ndarray]
    # Symbolic labels where the abscissa is categorical, else `None`.
    labels: Optional[Sequence[str]] = None
    # How it was drawn: a key of `DRAWN`, or "band".
    kind: str = "line"
    # Columns written to the table but never plotted.
    aux: Tuple[str, ...] = ()
    # Legend text, when it differs from `name`.
    legend: str = ""
    # Column whose value is written beside each point, if any.
    annotate: str = ""


def _record(ax: plt.Axes, series: Series) -> None:
    """Attach a series to the figure holding `ax`."""
    figure = ax.get_figure()
    if not hasattr(figure, "_mcrdv_series"):
        figure._mcrdv_series = []
    figure._mcrdv_series.append(series)


def _array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


# ============================================================================ #
# Drawing calls that record what they draw
# ============================================================================ #


# matplotlib's step styles, and the series kind each one exports as.
STEPPED = {
    "steps-mid": "steps_mid",
    "steps-post": "steps_post",
    "steps-pre": "steps_pre",
}


def _labelled(
    name: str, columns: Dict[str, np.ndarray], annotate: Optional[Sequence]
) -> Tuple[Dict[str, np.ndarray], Tuple[str, ...], str]:
    """Add a text column labelling each point, when `annotate` is given."""
    if annotate is None:
        return columns, (), ""
    column = f"{name}_label"
    # Kept as text: a label such as `5*` is not a number.
    written = np.asarray([str(value) for value in annotate])
    return {**columns, column: written}, (column,), column


def line(ax: plt.Axes, x, y, name: str, annotate=None, **kwargs: Any):
    """A line, recorded as one column, optionally labelled point by point."""
    x, y = _array(x), _array(y)
    kwargs.setdefault("label", name)
    artist = ax.plot(x, y, **kwargs)
    kind = STEPPED.get(str(kwargs.get("drawstyle", "")), "line")
    columns, aux, labels = _labelled(name, {name: y}, annotate)
    _record(
        ax,
        Series(
            name,
            x,
            columns,
            kind=kind,
            legend=str(kwargs["label"]),
            aux=aux,
            annotate=labels,
        ),
    )
    return artist


def points(ax: plt.Axes, x, y, name: str, annotate=None, **kwargs: Any):
    """A scatter, recorded as one column, optionally labelled point by point."""
    x, y = _array(x), _array(y)
    kwargs.setdefault("label", name)
    artist = ax.scatter(x, y, **kwargs)
    columns, aux, labels = _labelled(name, {name: y}, annotate)
    _record(
        ax,
        Series(
            name,
            x,
            columns,
            kind="marks",
            legend=str(kwargs["label"]),
            aux=aux,
            annotate=labels,
        ),
    )
    return artist


def band(ax: plt.Axes, x, lo, hi, name: str, **kwargs: Any):
    """A shaded interval, recorded as a `_lo` and a `_hi` column."""
    x, lo, hi = _array(x), _array(lo), _array(hi)
    kwargs.setdefault("alpha", 0.2)
    kwargs.setdefault("linewidth", 0.0)
    artist = ax.fill_between(x, lo, hi, **kwargs)
    _record(
        ax, Series(name, x, {f"{name}_lo": lo, f"{name}_hi": hi}, kind="band")
    )
    return artist


def series(
    ax: plt.Axes,
    name: str,
    x,
    kind: str = "line",
    aux: Sequence[str] = (),
    annotate: str = "",
    **columns: Sequence[float],
) -> None:
    """Record columns without drawing anything, for a figure drawn on screen
    in a way the table should not mirror. `kind` says how pgfplots should draw
    them, `aux` which columns are only to be read."""
    _record(
        ax,
        Series(
            name,
            _array(x),
            {k: _array(v) for k, v in columns.items()},
            kind=kind,
            aux=tuple(aux),
            annotate=annotate,
        ),
    )


def span(ax: plt.Axes, lo: float, hi: float, **kwargs: Any):
    """A shaded stretch of the abscissa, behind everything else (context,
    kept on the figure rather than in a table)."""
    kwargs.setdefault("color", "0.85")
    kwargs.setdefault("zorder", 0)
    kwargs.setdefault("linewidth", 0.0)
    artist = ax.axvspan(lo, hi, **kwargs)
    figure = ax.get_figure()
    if not hasattr(figure, "_mcrdv_spans"):
        figure._mcrdv_spans = []
    figure._mcrdv_spans.append((float(lo), float(hi)))
    return artist


def grouped_bars(
    ax: plt.Axes,
    labels: Sequence[str],
    series: Dict[str, Sequence[float]],
    **kwargs: Any,
):
    """One bar per category per series, each series recorded as a column;
    the categories keep their names for pgfplots' `symbolic x coords`."""
    positions = np.arange(len(labels), dtype=np.float64)
    width = 0.8 / max(len(series), 1)
    artists = []
    for index, (name, heights) in enumerate(series.items()):
        values = _array(heights)
        offset = (index - (len(series) - 1) / 2) * width
        artists.append(
            ax.bar(positions + offset, values, width, label=name, **kwargs)
        )
        _record(
            ax,
            Series(
                name,
                positions,
                {name: values},
                labels=list(labels),
                kind="bars",
            ),
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    return artists


# ============================================================================ #
# Writing the tables
# ============================================================================ #


def _number(value: float) -> str:
    """A number with enough digits to round-trip a plotted point."""
    if not np.isfinite(value):
        return "nan" if np.isnan(value) else ("inf" if value > 0 else "-inf")
    return f"{value:.10g}"


def _cell(value: Any) -> str:
    """One cell: a number, or a word with its spaces squeezed out (the table
    is whitespace-separated)."""
    if isinstance(value, (str, np.str_)):
        return "".join(str(value).split()) or "{}"
    return _number(value)


def _numeric(values: np.ndarray) -> bool:
    """Whether a column holds numbers."""
    return values.dtype.kind in "fiu"


def _groups(series: Sequence[Series]) -> List[List[Series]]:
    """Group the series sharing an abscissa into one table each."""
    groups: List[List[Series]] = []
    for item in series:
        for group in groups:
            head = group[0]
            if head.x.shape == item.x.shape and np.allclose(
                head.x, item.x, equal_nan=True
            ):
                group.append(item)
                break
        else:
            groups.append([item])
    return groups


def _pgf_name(text: str) -> str:
    """A column name pgfplots can read: letters, digits and underscores."""
    kept = "".join(c if c.isalnum() else "_" for c in str(text).strip())
    while "__" in kept:
        kept = kept.replace("__", "_")
    kept = kept.strip("_")
    return kept if kept[:1].isalpha() else f"c{kept}"


def _table(group: Sequence[Series], xname: str) -> str:
    """One whitespace-separated table, with a header `\\addplot table` can read."""
    head = group[0]
    columns: Dict[str, np.ndarray] = {}
    for item in group:
        columns.update(item.columns)

    names = [_pgf_name(name) for name in [xname] + list(columns)]
    lines = [" ".join(names)]
    if head.labels is not None:
        lines[0] = " ".join(["label"] + names)

    for row in range(head.x.size):
        cells = [_number(head.x[row])]
        cells += [_cell(values[row]) for values in columns.values()]
        if head.labels is not None:
            cells = [_pgf_name(head.labels[row])] + cells
        lines.append(" ".join(cells))
    return "\n".join(lines) + "\n"


# ============================================================================ #
# Writing the tikz skeleton
# ============================================================================ #


def _escape(text: str) -> str:
    """Make a label safe to paste into LaTeX (a bare `%` would comment out
    the rest of the line)."""
    for char in ("\\", "&", "%", "$", "#", "_", "{", "}"):
        text = text.replace(char, "\\" + char)
    return text


# The `\addplot` options of each kind of series.
DRAWN = {
    "line": "",
    "marks": "only marks",
    "steps_mid": "const plot mark mid",
    "steps_post": "const plot",
    "steps_pre": "const plot mark right",
    # `ybar` is set on the axis (see `_grouped`), the plot only fills.
    "bars": "fill",
    "segments": "no marks, line width=2pt",
    # Background, kept out of the legend and the colour cycle.
    "context": (
        "draw=gray, draw opacity=0.3, line width=0.3pt, no marks, forget plot"
    ),
}


def _worded(label: str, position: float) -> bool:
    """Whether a tick label is a word rather than its own coordinate."""
    text = label.replace("−", "-").strip()
    if not text or text.startswith("$"):
        return False
    try:
        return not math.isclose(float(text), position, rel_tol=1e-9)
    except ValueError:
        return True


def _ticks(ax: plt.Axes, skip: str = "") -> List[str]:
    """Axis options carrying worded tick labels (a chaser, a catalogue) over
    to pgfplots, with reversed and rotated axes kept as drawn."""
    out = []
    for axis, key in ((ax.yaxis, "y"), (ax.xaxis, "x")):
        if key == skip:
            continue
        positions = list(axis.get_ticklocs())
        labels = [t.get_text() for t in axis.get_ticklabels()]
        if not labels or len(labels) != len(positions):
            continue
        if not any(
            _worded(label, position)
            for label, position in zip(labels, positions)
        ):
            continue
        out.append(f"{key}tick={{{', '.join(f'{p:g}' for p in positions)}}}")
        out.append(
            f"{key}ticklabels={{{', '.join(_escape(l) for l in labels)}}}"
        )
    if ax.yaxis_inverted():
        out.append("y dir=reverse")
    labels = ax.xaxis.get_ticklabels()
    angle = labels[0].get_rotation() if labels else 0.0
    if angle:
        out.append(f"xticklabel style={{rotate={angle:g}, anchor=north east}}")
    return out


def _grouped(files: Dict[str, Sequence[Series]], categories: int) -> List[str]:
    """Axis options placing grouped bars side by side, with a bar width
    computed from how many bars the figure holds, and `ymin=0` when every
    bar is non-negative."""
    count = sum(
        item.kind == "bars" for group in files.values() for item in group
    )
    if not count:
        return []
    # pgfplots' default plotter box, in points, over the span the categories
    # cover once `enlarge x limits` has widened it.
    unit = 240.0 / (max(categories - 1, 1) * 1.4)
    width = min(0.6 * unit / count, 20.0)
    out = ["ybar", f"bar width={width:.3g}pt", "enlarge x limits=0.2"]
    if all(
        (values >= 0).all()
        for group in files.values()
        for item in group
        if item.kind == "bars"
        for values in item.columns.values()
        if _numeric(values)
    ):
        out.append("ymin=0")
    return out


def _scaled(ax: plt.Axes) -> List[str]:
    """Fixed-point tick labels on a linear axis whose range is narrow against
    its magnitude, where pgfplots would print `1.14, 1.14, 1.14`."""
    out = []
    for axis, key in ((ax.xaxis, "x"), (ax.yaxis, "y")):
        if axis.get_scale() == "log":
            continue
        lo, hi = sorted(axis.get_view_interval())
        ticks = [t for t in axis.get_ticklocs() if lo <= t <= hi]
        if len(ticks) < 2:
            continue
        step = min(
            (b - a for a, b in zip(ticks, ticks[1:]) if b > a), default=0.0
        )
        size = max(abs(ticks[0]), abs(ticks[-1]))
        if step <= 0.0 or size <= 0.0 or step / size >= 0.02:
            continue
        decimals = max(0, -int(math.floor(math.log10(step))))
        out += [
            f"scaled {key} ticks=false",
            f"{key} tick label style={{/pgf/number format/.cd, fixed, "
            f"precision={decimals}, 1000 sep={{}}}}",
        ]
    return out


def _stub(
    files: Dict[str, Sequence[Series]],
    xname: str,
    ax: plt.Axes,
    label: str = "",
) -> str:
    """A tikz skeleton that compiles against the exported tables and draws
    each series as it was drawn: marks stay marks, bars stay bars, bands
    become `fill between`, and `aux` columns are never plotted."""
    options = [f"xlabel={{{_escape(ax.get_xlabel() or xname)}}}"]
    if ax.get_ylabel():
        options.append(f"ylabel={{{_escape(ax.get_ylabel())}}}")
    if ax.get_xscale() == "log":
        options.append("xmode=log")
    if ax.get_yscale() == "log":
        options.append("ymode=log")

    body: List[str] = []
    needs_fillbetween = False
    # Room for point labels at the edge of the axis.
    if any(item.annotate for group in files.values() for item in group):
        options += ["clip=false", "enlarge x limits={upper, value=0.06}"]
    # No legend on screen, no legend entries.
    legend = ax.get_legend() is not None
    # A categorical abscissa is declared up front, by its labels.
    names = [
        item.labels
        for group in files.values()
        for item in group
        if item.labels is not None
    ]
    categorical = bool(names)
    if categorical:
        coordinates = [_pgf_name(label) for label in names[0]]
        options.append(f"symbolic x coords={{{', '.join(coordinates)}}}")
        options.append("xtick=data")
        if any(
            coordinate != str(label)
            for coordinate, label in zip(coordinates, names[0])
        ):
            spelled = ", ".join(_escape(str(label)) for label in names[0])
            options.append(f"xticklabels={{{spelled}}}")
        options += _grouped(files, len(names[0]))
    # NaN marks a break in a curve, which pgfplots must not join across.
    has_gaps = any(
        _numeric(values) and not np.isfinite(values).all()
        for group in files.values()
        for item in group
        for values in item.columns.values()
    )
    if has_gaps:
        options.append("unbounded coords=jump")
    options += _ticks(ax, skip="x" if categorical else "")
    options += _scaled(ax)

    # Shaded stretches first, so the series sit on top of them.
    for lo, hi in getattr(ax.get_figure(), "_mcrdv_spans", ()):
        body.append(
            f"    \\fill [black!10] ({{axis cs:{_number(lo)},0}} |- "
            f"{{rel axis cs:0,0}}) rectangle "
            f"({{axis cs:{_number(hi)},0}} |- {{rel axis cs:0,1}});"
        )

    for filename, group in files.items():
        for item in group:
            if item.kind == "band":
                needs_fillbetween = True
                low, high = (_pgf_name(c) for c in item.columns)
                body += [
                    f"    % band: {item.name}",
                    f"    \\addplot [draw=none, name path={low}] "
                    f"table[x={_pgf_name(xname)}, y={low}] {{{filename}}};",
                    f"    \\addplot [draw=none, name path={high}] "
                    f"table[x={_pgf_name(xname)}, y={high}] {{{filename}}};",
                    f"    \\addplot [fill=blue, fill opacity=0.15] "
                    f"fill between[of={low} and {high}];",
                ]
                continue
            drawn = [DRAWN.get(item.kind, "")]
            # Point labels read from a text column of the table.
            meta = ""
            if item.annotate:
                meta = f", meta={_pgf_name(item.annotate)}"
                drawn += [
                    "nodes near coords",
                    "point meta=explicit symbolic",
                    "every node near coord/.append style="
                    "{font=\\tiny, anchor=south west}",
                ]
            # `\addplot+` adds to the cycle list instead of replacing it.
            kept = [option for option in drawn if option]
            style = f"+ [{', '.join(kept)}]" if kept else ""
            abscissa = "label" if item.labels is not None else _pgf_name(xname)
            plotted = [c for c in item.columns if c not in item.aux]
            for column in plotted:
                body.append(
                    f"    \\addplot{style} table"
                    f"[x={abscissa}, y={_pgf_name(column)}{meta}]"
                    f" {{{filename}}};"
                )
                if not legend:
                    continue
                entry = (
                    (item.legend or item.name) if len(plotted) == 1 else column
                )
                body.append(
                    f"    \\addlegendentry{{{_escape(entry.replace('_', ' '))}}}"
                )

    head = [
        "% Generated by experiments/common/export.py -- adjust styling freely."
    ]
    if label:
        head.append(f"% Figure: {label}")
    if needs_fillbetween:
        head.append("% Requires: \\usepgfplotslibrary{fillbetween}")
    if categorical:
        head.append(
            "% The table also has a 'label' column: for named ticks, set"
            " symbolic x coords={...} and use x=label."
        )

    return "\n".join(
        [
            *head,
            "\\begin{tikzpicture}",
            "  \\begin{axis}[" + ", ".join(options) + "]",
            *body,
            "  \\end{axis}",
            "\\end{tikzpicture}",
            "",
        ]
    )


def save_figure(
    fig: plt.Figure,
    name: str,
    out: Path,
    xname: str = "x",
    label: str = "",
) -> List[Path]:
    """Write `<name>.pdf`, the tables of what it plots (`<name>.dat`, or one
    `<name>__<series>.dat` per abscissa) and `<name>.tex`; `label` names the
    figure inside the `.tex`. Returns every path written."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    path = out / f"{name}.pdf"
    fig.savefig(path)
    written.append(path)

    series: List[Series] = list(getattr(fig, "_mcrdv_series", ()))
    if not series:
        return written

    axes = fig.get_axes()
    # The abscissa column is named after the x label unless `xname` is given.
    axis_label = axes[0].get_xlabel() if axes else ""
    xname = xname if xname != "x" else (_slug(axis_label) or "x")

    groups = _groups(series)
    files: Dict[str, Sequence[Series]] = {}
    for group in groups:
        stem = name if len(groups) == 1 else f"{name}__{group[0].name}"
        path = out / f"{stem}.dat"
        path.write_text(_table(group, xname), encoding="utf-8")
        written.append(path)
        files[path.name] = group

    stub = out / f"{name}.tex"
    stub.write_text(
        _stub(files, xname, axes[0], label) if axes else "", encoding="utf-8"
    )
    written.append(stub)
    return written


def _slug(text: str) -> str:
    """A column name from an axis label: letters, digits and underscores."""
    kept = [c if c.isalnum() else "_" for c in text.strip().lower()]
    slug = "".join(kept).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    # pgfplots column names cannot start with a digit.
    return slug if slug[:1].isalpha() else ""


# ============================================================================ #
# Drawing figures from plot.yaml
# ============================================================================ #


def decorate(ax, spec: Mapping[str, Any]) -> None:
    """Apply the axis settings a figure declared."""
    if spec.get("xlabel"):
        ax.set_xlabel(str(spec["xlabel"]))
    if spec.get("ylabel"):
        ax.set_ylabel(str(spec["ylabel"]))
    if spec.get("x_scale"):
        ax.set_xscale(str(spec["x_scale"]))
    if spec.get("y_scale"):
        ax.set_yscale(str(spec["y_scale"]))


def draw(
    name: str, figure: Figure, data: Data, out: Path, save: bool
) -> List[Path]:
    """Draw one figure; with `save`, write it (named like a record) and
    close it, otherwise keep it open for display."""
    fig, ax = _new_figure()
    figure(ax, data)
    decorate(ax, data.spec)
    if ax.get_legend_handles_labels()[0] and data.spec.get("legend", True):
        ax.legend()

    if not save:
        manager = getattr(fig.canvas, "manager", None)
        if manager is not None:
            manager.set_window_title(name)
        return []

    written = save_figure(
        fig,
        runner.new_name(),
        out,
        xname=str(data.spec.get("xname", "x")),
        label=name,
    )
    plt.close(fig)
    return written


def main(
    script: str,
    figures: Mapping[str, Figure],
    argv: Optional[Sequence[str]] = None,
) -> int:
    """Draw every figure of an experiment's `plot.yaml`; `script` is the
    calling `plot.py`, `figures` maps `draw` names to drawing functions."""
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

    if not args.save and not can_display():
        raise SystemExit(
            f"no display available (matplotlib backend is "
            f"{matplotlib.get_backend()}) -- pass --save to write the figures"
        )

    spec = runner.load_config(args.config)
    root = source(spec.get("source"), ("results", {}))

    # Records are read once per folder actually named by a drawn figure.
    loaded: Dict[
        Path, Tuple[List[runner.RunRecord], List[Dict[str, Any]]]
    ] = {}

    def read(folder: Any):
        path = Path(args.results or (here / str(folder)))
        if path not in loaded:
            if not path.exists():
                raise SystemExit(f"{path} missing -- run the experiment first")
            found = list(runner.load_records(path))
            if not found:
                raise SystemExit(f"no records in {path}")
            loaded[path] = (found, [r.row() for r in found])
            print(f"{here.name}: {len(found)} records from {path}")
        return loaded[path]

    entries = spec.get("figures", [])
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
        # How many of the folder's records the figure is about, and which
        # timestamped files hold it.
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
        plt.show()
    return 0

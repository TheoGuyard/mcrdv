"""Draw a figure and keep the points it drew, for tikz.

matplotlib is how the results get looked at; the paper redraws them with
pgfplots. The two would drift apart if the exported table were rebuilt from the
result CSV, because a figure almost never shows raw rows -- it shows medians,
interquartile bands and curves resampled onto a common grid, all computed inside
`plot.py`. So the export is recorded by the same call that draws, and the table
is by construction the figure.

    fig, ax = style.figure()
    export.line(ax, t, median, name="hgs")
    export.band(ax, t, lo, hi, name="hgs_iqr")
    export.save(fig, "convergence", out)      # .pdf + .dat + .tex
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

import numpy as np

from .style import plt


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


def _record(ax: plt.Axes, series: Series) -> None:
    """Attach a series to the figure holding `ax`."""
    figure = ax.get_figure()
    if not hasattr(figure, "_mcrdv_series"):
        figure._mcrdv_series = []
    figure._mcrdv_series.append(series)


def _array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


# ============================================================================ #
# Drawing
# ============================================================================ #


def line(ax: plt.Axes, x, y, name: str, **kwargs: Any):
    """A line, recorded as one column."""
    x, y = _array(x), _array(y)
    kwargs.setdefault("label", name)
    artist = ax.plot(x, y, **kwargs)
    _record(ax, Series(name, x, {name: y}))
    return artist


def points(ax: plt.Axes, x, y, name: str, **kwargs: Any):
    """A scatter, recorded as one column."""
    x, y = _array(x), _array(y)
    kwargs.setdefault("label", name)
    artist = ax.scatter(x, y, **kwargs)
    _record(ax, Series(name, x, {name: y}))
    return artist


def band(ax: plt.Axes, x, lo, hi, name: str, **kwargs: Any):
    """A shaded interval, recorded as a `_lo` and a `_hi` column."""
    x, lo, hi = _array(x), _array(lo), _array(hi)
    kwargs.setdefault("alpha", 0.2)
    kwargs.setdefault("linewidth", 0.0)
    artist = ax.fill_between(x, lo, hi, **kwargs)
    _record(ax, Series(name, x, {f"{name}_lo": lo, f"{name}_hi": hi}))
    return artist


def series(ax: plt.Axes, name: str, x, **columns: Sequence[float]) -> None:
    """Record columns without drawing anything.

    For a figure matplotlib draws in a way the table should not mirror: a
    scatter coloured by category is fifteen `ax.scatter` calls on screen, but
    one tidy `x y category` table is what pgfplots wants for `scatter/classes`.
    Draw however reads best, then record the shape the paper needs.
    """
    _record(
        ax, Series(name, _array(x), {k: _array(v) for k, v in columns.items()})
    )


def bars(
    ax: plt.Axes, labels: Sequence[str], heights, name: str, **kwargs: Any
):
    """Categorical bars, recorded with their labels for `symbolic x coords`."""
    return grouped_bars(ax, labels, {name: heights}, **kwargs)


def grouped_bars(
    ax: plt.Axes,
    labels: Sequence[str],
    series: Dict[str, Sequence[float]],
    **kwargs: Any,
):
    """One bar per category per series, recorded as one column each.

    The categories keep their names in the exported table, so pgfplots can read
    it with `symbolic x coords` instead of the integer positions matplotlib
    needs.
    """
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
            ax, Series(name, positions, {name: values}, labels=list(labels))
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    return artists


# ============================================================================ #
# Writing
# ============================================================================ #


def _number(value: float) -> str:
    """A number with enough digits to round-trip a plotted point."""
    if not np.isfinite(value):
        return "nan" if np.isnan(value) else ("inf" if value > 0 else "-inf")
    return f"{value:.10g}"


def _groups(series: Sequence[Series]) -> List[List[Series]]:
    """Group series that share an abscissa, so they share a table.

    Series on different grids -- per-seed convergence curves, say -- cannot be
    columns of one table without padding it ragged, so each gets its own file.
    """
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


def _table(group: Sequence[Series], xname: str) -> str:
    """One whitespace-separated table, with a header `\\addplot table` can read."""
    head = group[0]
    columns: Dict[str, np.ndarray] = {}
    for item in group:
        columns.update(item.columns)

    names = [xname] + list(columns)
    lines = [" ".join(names)]
    if head.labels is not None:
        lines[0] = " ".join(["label"] + names)

    for row in range(head.x.size):
        cells = [_number(head.x[row])]
        cells += [_number(values[row]) for values in columns.values()]
        if head.labels is not None:
            cells = [str(head.labels[row]).replace(" ", "_")] + cells
        lines.append(" ".join(cells))
    return "\n".join(lines) + "\n"


def _escape(text: str) -> str:
    """Make an axis label safe to paste into LaTeX.

    Matters more than it looks: a `%` in a label such as "gap to best [%]" is a
    comment character, so an unescaped stub silently swallows the rest of the
    line and produces a plot with no axis label rather than an error.
    """
    for char in ("\\", "&", "%", "$", "#", "_", "{", "}"):
        text = text.replace(char, "\\" + char)
    return text


def _is_band(item: Series) -> bool:
    """Whether a series is the `_lo`/`_hi` pair produced by :func:`band`."""
    names = list(item.columns)
    return (
        len(names) == 2
        and names[0].endswith("_lo")
        and names[1].endswith("_hi")
    )


def _stub(
    files: Dict[str, Sequence[Series]],
    xname: str,
    ax: plt.Axes,
    label: str = "",
) -> str:
    """A tikz skeleton that already compiles against the exported tables.

    Bands come out as the `fill between` idiom rather than two unrelated curves,
    because that is the one part a reader would otherwise have to look up, and
    getting it wrong produces a plot that is subtly wrong rather than broken.
    """
    options = [f"xlabel={{{_escape(ax.get_xlabel() or xname)}}}"]
    if ax.get_ylabel():
        options.append(f"ylabel={{{_escape(ax.get_ylabel())}}}")
    if ax.get_xscale() == "log":
        options.append("xmode=log")
    if ax.get_yscale() == "log":
        options.append("ymode=log")

    body: List[str] = []
    needs_fillbetween = False
    categorical = any(
        item.labels is not None for group in files.values() for item in group
    )
    # A series drawn in pieces -- a curve that changes style partway, a line
    # broken where an angle wraps -- carries NaN in the gaps. pgfplots joins
    # straight across them unless told not to.
    has_gaps = any(
        not np.isfinite(values).all()
        for group in files.values()
        for item in group
        for values in item.columns.values()
    )
    if has_gaps:
        options.append("unbounded coords=jump")
    for filename, group in files.items():
        for item in group:
            if _is_band(item):
                needs_fillbetween = True
                low, high = list(item.columns)
                body += [
                    f"    % band: {item.name}",
                    f"    \\addplot [draw=none, name path={low}] "
                    f"table[x={xname}, y={low}] {{{filename}}};",
                    f"    \\addplot [draw=none, name path={high}] "
                    f"table[x={xname}, y={high}] {{{filename}}};",
                    f"    \\addplot [fill=blue, fill opacity=0.15] "
                    f"fill between[of={low} and {high}];",
                ]
                continue
            for column in item.columns:
                body += [
                    f"    \\addplot table[x={xname}, y={column}] {{{filename}}};",
                    f"    \\addlegendentry{{{_escape(column.replace('_', ' '))}}}",
                ]

    head = [
        "% Generated by experiments/common/export.py -- adjust styling freely."
    ]
    if label:
        # The file stem is a timestamp, so this is the only place the figure
        # says which one it is.
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


def save(
    fig: plt.Figure,
    name: str,
    out: Path,
    xname: str = "x",
    formats: Sequence[str] = ("pdf",),
    label: str = "",
) -> List[Path]:
    """Write the figure, the table of what it plots, and a tikz stub.

    `name` is the shared stem of the three files; `label` is what the figure is
    called, which the caller may want recorded when the stem is a timestamp
    rather than something readable.

    Returns every path written, so a caller can report or check them.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for suffix in formats:
        path = out / f"{name}.{suffix}"
        fig.savefig(path)
        written.append(path)

    series: List[Series] = list(getattr(fig, "_mcrdv_series", ()))
    if not series:
        return written

    axes = fig.get_axes()
    # Fall back to the x-axis label for the column name, when the caller left
    # `xname` at its default. Kept distinct from `label`, which names the figure.
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

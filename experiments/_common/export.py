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

import math
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

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
    # How it was drawn, which decides what the tikz stub does with it:
    # a curve, markers alone, a shaded interval, or bars.
    kind: str = "line"
    # Columns the table carries but nothing plots -- the class each row
    # belongs to, say. They are written, and never given an `\\addplot`.
    aux: Tuple[str, ...] = ()
    # What the legend calls it, where that is not its name: `name` is a stem
    # for a file and a column, `legend` is a phrase for a reader.
    legend: str = ""
    # The column whose value is written beside each point: the grid size a
    # point was measured at, the preset a run searched with. Empty for a
    # series whose points speak for themselves.
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
# Drawing
# ============================================================================ #


# How matplotlib says a curve steps, and what this module calls it. A
# histogram drawn as steps and exported as a line is a different figure: the
# steps say the count holds over a bin, the line says it slopes between two.
STEPPED = {
    "steps-mid": "steps_mid",
    "steps-post": "steps_post",
    "steps-pre": "steps_pre",
}


def _labelled(
    name: str, columns: Dict[str, np.ndarray], annotate: Optional[Sequence]
) -> Tuple[Dict[str, np.ndarray], Tuple[str, ...], str]:
    """Add the column that labels each point, where a caller gave one.

    Matplotlib writes such a label with `annotate`, one call per point, which
    leaves nothing in the table. Here it is a column like any other -- never
    plotted, but written, so the tikz stub can put the same words in the same
    places instead of the figures disagreeing about what is labelled.
    """
    if annotate is None:
        return columns, (), ""
    column = f"{name}_label"
    # Kept as text: a label is a word the figure shows -- `1024`, or `5*` for
    # a preset the clock stopped -- and reading it as a number would lose the
    # ones that are not numbers and reformat the ones that are.
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
    """Record columns without drawing anything.

    For a figure matplotlib draws in a way the table should not mirror: a plan
    read leg by leg is a dozen `ax.plot` calls on screen, and one table per
    chaser is what pgfplots wants. Draw however reads best, then record the
    shape the paper needs, saying with `kind` how it should be drawn again and
    with `aux` which columns are there to be read rather than plotted.
    """
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
    """A shaded stretch of the abscissa, behind everything else.

    A window a chaser may not depart in, an interval a reading is suspect
    over: context rather than data, so it is two numbers rather than a column,
    and it is kept on the figure instead of in a table.
    """
    kwargs.setdefault("color", "0.85")
    kwargs.setdefault("zorder", 0)
    kwargs.setdefault("linewidth", 0.0)
    artist = ax.axvspan(lo, hi, **kwargs)
    figure = ax.get_figure()
    if not hasattr(figure, "_mcrdv_spans"):
        figure._mcrdv_spans = []
    figure._mcrdv_spans.append((float(lo), float(hi)))
    return artist


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
# Writing
# ============================================================================ #


def _number(value: float) -> str:
    """A number with enough digits to round-trip a plotted point."""
    if not np.isfinite(value):
        return "nan" if np.isnan(value) else ("inf" if value > 0 else "-inf")
    return f"{value:.10g}"


def _cell(value: Any) -> str:
    """One cell of the table: a number, or a word written as it stands.

    A whitespace-separated table has no way to hold a space, so a label that
    contains one is squeezed; pgfplots would otherwise read the tail of it as
    the next column.
    """
    if isinstance(value, (str, np.str_)):
        return "".join(str(value).split()) or "{}"
    return _number(value)


def _numeric(values: np.ndarray) -> bool:
    """Whether a column holds numbers, and so can be tested for gaps."""
    return values.dtype.kind in "fiu"


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


def _column(text: str) -> str:
    """A name pgfplots can read: letters, digits and underscores.

    A series is named for a person -- "fuel only", "the rest" -- and a table
    column is named for a parser, which cannot find a column whose name holds
    a space. Both the header and the `\\addplot` that reads it go through here,
    so they cannot disagree; the human name survives in the legend.
    """
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

    names = [_column(name) for name in [xname] + list(columns)]
    lines = [" ".join(names)]
    if head.labels is not None:
        lines[0] = " ".join(["label"] + names)

    for row in range(head.x.size):
        cells = [_number(head.x[row])]
        cells += [_cell(values[row]) for values in columns.values()]
        if head.labels is not None:
            cells = [_column(head.labels[row])] + cells
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
    if item.kind == "band":
        return True
    names = list(item.columns)
    return (
        len(names) == 2
        and names[0].endswith("_lo")
        and names[1].endswith("_hi")
    )


# What each kind of series becomes, as the options of its `\addplot`.
DRAWN = {
    "line": "",
    "marks": "only marks",
    # A value that holds over an interval rather than sloping to the next.
    "steps_mid": "const plot mark mid",
    "steps_post": "const plot",
    "steps_pre": "const plot mark right",
    # `ybar` belongs to the axis, not to the plot: it is what shifts the bars
    # of a group off one another. Set here it would draw all of them at the
    # same abscissa, one hiding the next. So the plot only says to fill.
    "bars": "fill",
    # Intervals: a bar of a given length along the axis, drawn as the segment
    # it is, with the gaps between them left as gaps.
    "segments": "no marks, line width=2pt",
    # Context: what the figure is read against rather than what it is about.
    # `forget plot` keeps it out of the legend and leaves the colour cycle for
    # the series that matter.
    "context": (
        "draw=gray, draw opacity=0.3, line width=0.3pt, no marks, forget plot"
    ),
}


def _worded(label: str, position: float) -> bool:
    """Whether a tick label says something its coordinate does not.

    Matplotlib labels every tick, most of them with the number underneath --
    in its own spelling, with a Unicode minus, and as `$\\mathdefault{10^{1}}$`
    on a log axis. None of that should cross into pgfplots, which formats its
    own numbers. A label is worth carrying over only when it is a word: a
    chaser, a catalogue, a setting.
    """
    text = label.replace("−", "-").strip()
    if not text or text.startswith("$"):
        return False
    try:
        return not math.isclose(float(text), position, rel_tol=1e-9)
    except ValueError:
        return True


def _ticks(ax: plt.Axes, skip: str = "") -> List[str]:
    """Axis options for ticks matplotlib was told to label itself.

    A categorical axis -- one chaser per row, one catalogue per bar -- carries
    labels that are not its coordinates, and they are as much part of the
    figure as the points are. pgfplots is given the positions and the words,
    except on an axis `skip` names, which says it for itself.
    """
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
    # Rows counted downwards on screen are counted downwards on the page too.
    if ax.yaxis_inverted():
        out.append("y dir=reverse")
    # Labels long enough to have been turned on screen are turned on the page,
    # for the same reason: side by side they would run into one another.
    labels = ax.xaxis.get_ticklabels()
    angle = labels[0].get_rotation() if labels else 0.0
    if angle:
        out.append(f"xticklabel style={{rotate={angle:g}, anchor=north east}}")
    return out


def _grouped(files: Dict[str, Sequence[Series]], categories: int) -> List[str]:
    """Axis options that put bars of a group side by side, and keep them there.

    `ybar` belongs to the axis: it is what shifts a group's bars off one
    another, and matplotlib's `bar` already did that by hand. The width is the
    part pgfplots cannot work out, because it has no idea how many bars will
    arrive: left at its default, thirteen chasers overlap and four oracles run
    into the next category. It is set here from what the figure holds, as a
    fraction of the distance between two categories on an axis of the default
    width.
    """
    count = sum(
        item.kind == "bars" for group in files.values() for item in group
    )
    if not count:
        return []
    # pgfplots' default plotting box, in points, over the span the categories
    # cover once `enlarge x limits` has widened it.
    unit = 240.0 / (max(categories - 1, 1) * 1.4)
    width = min(0.6 * unit / count, 20.0)
    out = ["ybar", f"bar width={width:.3g}pt", "enlarge x limits=0.2"]
    # A bar is drawn from zero, so the axis has to reach it: left to fit the
    # data, pgfplots starts at the shortest bar and the reader compares the
    # tops of bars whose bottoms are off the figure.
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
    """Axis options for a range too narrow for pgfplots' default formatting.

    Left to itself, pgfplots factors a common power out of the tick labels and
    prints the rest to three digits. On an axis running from 113 800 to
    115 800 -- a mission cost, where the whole question is a two-percent
    difference -- that prints `1.14`, `1.14`, `1.14` and the figure says
    nothing. Matplotlib prints the numbers in full, and so should the paper.

    The test is the tick step against the magnitude of the values: only an
    axis whose labels would collide gets the fixed format, and the decimals
    come from the step, so a narrow range near zero keeps its digits too.
    """
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
    """A tikz skeleton that already compiles against the exported tables.

    It is meant to draw the same data as the figure beside it, which means
    following how each series was drawn rather than plotting every column it
    can find: markers stay markers, bars stay bars, and a column carried for
    reading -- which chaser a row belongs to -- is written to the table and
    never plotted. Bands come out as the `fill between` idiom rather than two
    unrelated curves, because that is the one part a reader would otherwise
    have to look up, and getting it wrong produces a plot that is subtly wrong
    rather than broken.
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
    # A label sits beside its point, and the point at the end of a curve sits
    # on the edge of the axis, so the last label of a series is drawn half
    # outside it. Clipping is what cuts it in two, and a little room on the
    # right is what keeps it from leaning on whatever the figure is set beside.
    if any(item.annotate for group in files.values() for item in group):
        options += ["clip=false", "enlarge x limits={upper, value=0.06}"]
    # A figure drawn without a legend is one whose series do not need naming --
    # fifteen chasers, say -- so the stub does not name them either.
    legend = ax.get_legend() is not None
    # A categorical abscissa is read by its labels rather than its positions,
    # which pgfplots wants declared up front.
    names = [
        item.labels
        for group in files.values()
        for item in group
        if item.labels is not None
    ]
    categorical = bool(names)
    if categorical:
        # The coordinates are what the table holds, and the labels are what a
        # reader sees: "collection time" cannot be a coordinate, but it can be
        # written under one.
        coordinates = [_column(label) for label in names[0]]
        options.append(f"symbolic x coords={{{', '.join(coordinates)}}}")
        options.append("xtick=data")
        if any(
            coordinate != str(label)
            for coordinate, label in zip(coordinates, names[0])
        ):
            spelled = ", ".join(_escape(str(label)) for label in names[0])
            options.append(f"xticklabels={{{spelled}}}")
        options += _grouped(files, len(names[0]))
    # A series drawn in pieces -- a curve that changes style partway, a line
    # broken where an angle wraps -- carries NaN in the gaps. pgfplots joins
    # straight across them unless told not to.
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

    # Shaded stretches go down before anything else, so the series sit on top
    # of them. The abscissa is read in axis coordinates and the height in
    # relative ones, which spans the axis whatever its limits turn out to be.
    for lo, hi in getattr(ax.get_figure(), "_mcrdv_spans", ()):
        body.append(
            f"    \\fill [black!10] ({{axis cs:{_number(lo)},0}} |- "
            f"{{rel axis cs:0,0}}) rectangle "
            f"({{axis cs:{_number(hi)},0}} |- {{rel axis cs:0,1}});"
        )

    for filename, group in files.items():
        for item in group:
            if _is_band(item):
                needs_fillbetween = True
                low, high = (_column(c) for c in item.columns)
                body += [
                    f"    % band: {item.name}",
                    f"    \\addplot [draw=none, name path={low}] "
                    f"table[x={_column(xname)}, y={low}] {{{filename}}};",
                    f"    \\addplot [draw=none, name path={high}] "
                    f"table[x={_column(xname)}, y={high}] {{{filename}}};",
                    f"    \\addplot [fill=blue, fill opacity=0.15] "
                    f"fill between[of={low} and {high}];",
                ]
                continue
            drawn = [DRAWN.get(item.kind, "")]
            # A label per point, read from a column of the table and printed
            # beside its marker. `explicit symbolic` is what stops pgfplots
            # treating the column as a number to be reformatted: the table
            # already holds the word the figure showed.
            meta = ""
            if item.annotate:
                meta = f", meta={_column(item.annotate)}"
                drawn += [
                    "nodes near coords",
                    "point meta=explicit symbolic",
                    "every node near coord/.append style="
                    "{font=\\tiny, anchor=south west}",
                ]
            # `\addplot+` keeps the cycle list and adds to it; `\addplot[...]`
            # replaces it, which would draw every styled series in black.
            kept = [option for option in drawn if option]
            style = f"+ [{', '.join(kept)}]" if kept else ""
            # A categorical abscissa is read by its labels, not its positions.
            abscissa = "label" if item.labels is not None else _column(xname)
            plotted = [c for c in item.columns if c not in item.aux]
            for column in plotted:
                body.append(
                    f"    \\addplot{style} table"
                    f"[x={abscissa}, y={_column(column)}{meta}]"
                    f" {{{filename}}};"
                )
                if not legend:
                    continue
                # What the series is called, unless its columns are the
                # several things it holds.
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

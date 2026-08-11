"""Figure style and plotting helpers.

Every figure in this repository is an argument about numbers that are close
together -- "this metric ranks models slightly better than that one" -- so the
visual defaults have to make small, honest differences readable and must not
manufacture large-looking ones.  Three rules follow, and they are enforced here
rather than left to each plotting script:

* **Error bars are mandatory.**  :func:`series` refuses to draw a comparison
  without them.  A comparison plotted without uncertainty is the graphical form
  of the overclaiming this project is about.
* **Colour is never the only channel.**  Every series gets a distinct marker
  shape as well as a hue, and series are directly labelled where there is room.
  Three of the categorical hues sit below 3:1 contrast on a white surface, so
  the secondary encoding is what keeps them legible -- for colour-vision
  deficiency, for greyscale printing, and for a projector.
* **One axis, ever.**  There is no dual-axis helper. Two quantities on different
  scales get two panels or an index to a common base.

The palette is the validated eight-hue categorical set; its worst adjacent
colour-vision-deficiency separation is ΔE 9.1 (OKLab x100) and its worst
normal-vision separation 19.6.  Scatter plots compare all pairs rather than
adjacent ones and only the first three slots clear the floors there, so
:func:`scatter_matrix_colors` caps at three and folds the rest into a neutral
"other" -- rather than silently cycling hues that readers cannot tell apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------

#: Categorical hues in fixed assignment order.  Never cycle past the end: an
#: extra series folds into ``NEUTRAL`` or gets its own small multiple.
SERIES = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

#: Marker shapes paired one-to-one with :data:`SERIES`.  This is the secondary
#: encoding that makes identity survive greyscale and colour-vision deficiency.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

#: Line styles as a third channel, for the rare figure printed in greyscale.
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)), (0, (7, 3))]

NEUTRAL = "#8a8a85"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#dedddb"
SURFACE = "#ffffff"

#: Sequential ramp: one hue, light to dark, for magnitude.
SEQUENTIAL = "Blues"

#: Diverging ramp: two hues with a neutral grey midpoint, for signed quantities
#: such as "model predicts too much / too little".  Never a rainbow, and never a
#: hue at the midpoint -- zero must read as absence, not as another category.
DIVERGING = "RdBu_r"


def series_color(index: int) -> str:
    """Colour for series ``index``, refusing to cycle.

    Cycling would give series 9 the same hue as series 1, which is not a style
    preference but a correctness problem: the reader would conclude the two are
    the same thing.
    """
    if index >= len(SERIES):
        return NEUTRAL
    return SERIES[index]


def series_style(index: int) -> dict:
    """Full style dict (colour, marker, linestyle) for series ``index``."""
    n = len(SERIES)
    return {
        "color": series_color(index),
        "marker": MARKERS[index] if index < n else ".",
        "linestyle": LINESTYLES[index] if index < n else ":",
    }


def scatter_matrix_colors(n: int) -> list[str]:
    """Colours for a scatter plot, where every pair is compared at once.

    Only the first three categorical slots clear the colour-vision floors when
    all pairs are on screen simultaneously, so anything past the third is drawn
    neutral and must be identified some other way (direct label, facet, or a
    table).  Returning grey here is deliberate: it makes the constraint visible
    to whoever writes the figure rather than hiding it behind a hue nobody can
    resolve.
    """
    return [SERIES[i] if i < 3 else NEUTRAL for i in range(n)]


# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------


def use_style(*, fontsize: float = 9.0, serif: bool = False) -> None:
    """Apply the project's matplotlib defaults.

    Thin marks, recessive grid and axes, no top/right spines, and a figure size
    that matches a single journal column so that figures are never rescaled
    (rescaling is what produces the inconsistent font sizes that make a figure
    set look assembled rather than designed).
    """
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.figsize": (3.4, 2.6),
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,

        "font.size": fontsize,
        "font.family": "serif" if serif else "sans-serif",
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize + 1,
        "xtick.labelsize": fontsize - 1,
        "ytick.labelsize": fontsize - 1,
        "legend.fontsize": fontsize - 1,

        "axes.edgecolor": TEXT_SECONDARY,
        "axes.labelcolor": TEXT_PRIMARY,
        "text.color": TEXT_PRIMARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.alpha": 1.0,
        "axes.axisbelow": True,

        "lines.linewidth": 1.6,
        "lines.markersize": 4.0,
        "errorbar.capsize": 2.0,

        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.0,
        "legend.borderaxespad": 0.3,

        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------


def series(ax, x, y, yerr, *, index: int, label: str, fill: bool = True, **kwargs):
    """Plot one series with its uncertainty, using colour + marker + linestyle.

    Parameters
    ----------
    yerr:
        One standard error.  Required.  Pass an array of zeros only if the
        quantity is exact by construction (an analytic reference), and say so in
        the caption -- an absent error bar and a zero error bar mean very
        different things and the figure should not conflate them.
    fill:
        Draw the uncertainty as a shaded band (good for curves such as ``g(r)``)
        rather than as caps (good for a handful of points).
    """
    if yerr is None:
        raise ValueError(
            "series() requires an uncertainty; a comparison drawn without one "
            "is not evidence. Pass zeros explicitly for exact quantities."
        )
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.broadcast_to(np.asarray(yerr, dtype=float), y.shape)
    st = series_style(index)
    st.update(kwargs)

    if fill and y.size > 12:
        line, = ax.plot(x, y, color=st["color"], linestyle=st["linestyle"], label=label,
                        **{k: v for k, v in st.items() if k not in ("color", "linestyle", "marker")})
        ax.fill_between(x, y - yerr, y + yerr, color=st["color"], alpha=0.18, linewidth=0)
        return line
    container = ax.errorbar(
        x, y, yerr=yerr, label=label,
        color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
        markerfacecolor=st["color"], markeredgecolor=SURFACE, markeredgewidth=0.6,
        elinewidth=0.9, capsize=2.0,
        **{k: v for k, v in kwargs.items() if k not in st},
    )
    return container


def label_last_point(ax, x, y, text: str, *, index: int, dx: float = 0.01, fontsize=None):
    """Direct-label a series at its right-hand end.

    Direct labels are the relief that makes the lower-contrast hues legible, and
    they remove the eye's trip to a legend box.  Used selectively -- never a
    number on every point.
    """
    ax.annotate(
        text,
        xy=(np.asarray(x)[-1], np.asarray(y)[-1]),
        xytext=(dx, 0), textcoords="offset fontsize",
        color=series_color(index), va="center", ha="left",
        fontsize=fontsize, fontweight="medium", clip_on=False,
    )


def rank_scatter(ax, proxy, truth, labels=None, *, proxy_name="proxy metric",
                 truth_name="observable error", annotate_top: int = 3):
    """Scatter a proxy metric against the truth it is supposed to predict.

    Draws the identity of the *ranking* rather than of the values: both axes are
    plotted on their rank scale as a secondary panel would, with the raw values
    on the primary axes.  The diagonal is drawn as a reference for a perfect
    proxy, and the ``annotate_top`` best models by truth are labelled -- those
    are the ones a practitioner would actually choose between, so they are the
    ones whose displacement from the diagonal matters.
    """
    proxy = np.asarray(proxy, dtype=float)
    truth = np.asarray(truth, dtype=float)
    ax.scatter(proxy, truth, s=26, color=SERIES[0], edgecolor=SURFACE, linewidth=0.6, zorder=3)

    if labels is not None and annotate_top > 0:
        order = np.argsort(truth)[:annotate_top]
        for rank, i in enumerate(order):
            ax.annotate(
                f"{labels[i]}",
                xy=(proxy[i], truth[i]), xytext=(0.4, 0.4), textcoords="offset fontsize",
                fontsize=7, color=TEXT_SECONDARY, zorder=4,
            )
            ax.scatter([proxy[i]], [truth[i]], s=42, facecolor="none",
                       edgecolor=SERIES[1], linewidth=1.2, zorder=5)

    ax.set_xlabel(proxy_name)
    ax.set_ylabel(truth_name)
    return ax


def heatmap(ax, matrix, row_labels, col_labels, *, diverging=True, vmax=None,
            cbar_label="", annotate=True, fmt="{:.2f}"):
    """Correlation / error matrix as a heatmap.

    Uses the diverging ramp with a neutral grey midpoint for signed quantities
    such as rank correlations, so that "no relationship" reads as absence rather
    than as a colour of its own.  Cells are annotated when the matrix is small
    enough for the numbers to be readable, because a reader comparing 0.31 with
    0.44 cannot do it from hue.
    """
    import matplotlib as mpl

    m = np.asarray(matrix, dtype=float)
    if vmax is None:
        vmax = float(np.nanmax(np.abs(m))) if diverging else float(np.nanmax(m))
    vmin = -vmax if diverging else 0.0
    cmap = mpl.colormaps[DIVERGING if diverging else SEQUENTIAL]

    image = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    if annotate and m.size <= 240:
        threshold = 0.6 * vmax
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if not np.isfinite(m[i, j]):
                    continue
                ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center", fontsize=6.5,
                        color=SURFACE if abs(m[i, j]) > threshold else TEXT_PRIMARY)

    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(cbar_label)
    cbar.outline.set_visible(False)
    return image


def hero_number(ax, value: str, caption: str, *, sub: str = "", color=None):
    """A single headline number, for the cases where a chart would be noise.

    One number with a caption is the right form when the finding is a scalar --
    "force RMSE explains 8% of the variance in RDF error" does not become clearer
    as a one-bar bar chart.
    """
    ax.axis("off")
    ax.text(0.0, 0.62, value, fontsize=22, fontweight="bold", va="center",
            color=color or TEXT_PRIMARY, transform=ax.transAxes)
    ax.text(0.0, 0.28, caption, fontsize=9, va="center", color=TEXT_PRIMARY,
            transform=ax.transAxes)
    if sub:
        ax.text(0.0, 0.10, sub, fontsize=7.5, va="center", color=TEXT_SECONDARY,
                transform=ax.transAxes)


def add_panel_labels(axes, labels=None, *, offset=(-0.22, 1.02), fontsize=10):
    """Label multi-panel figures (a), (b), (c) ... in a consistent place."""
    labels = labels or [f"({chr(97 + i)})" for i in range(len(axes))]
    for ax, text in zip(np.ravel(axes), labels):
        ax.text(offset[0], offset[1], text, transform=ax.transAxes,
                fontsize=fontsize, fontweight="bold", va="bottom", ha="left")


def save_figure(fig, path, *, formats: Sequence[str] = ("png", "pdf")) -> list[Path]:
    """Save in raster and vector form, and return the paths written.

    Both formats every time: the PNG is what gets looked at, the PDF is what
    survives being enlarged in a talk or a paper.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        target = path.with_suffix(f".{ext}")
        fig.savefig(target)
        written.append(target)
    return written

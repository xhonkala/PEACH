"""Shared visualization defaults — Tufte-inspired, colorblind-safe.

Central place for all PEACH plot styling. Every pl.* function should
import from here rather than hardcoding magic numbers.

Design principles (Tufte):
  - Maximize data-ink ratio: remove gridlines, borders, redundant labels
  - Let the data speak: muted backgrounds, no 3D effects, no chart junk
  - Above all else, show the data

Colorblind safety:
  - Categorical palette based on Wong (2011) Nature Methods
  - Sequential/diverging palettes from ColorBrewer
"""

# ---------------------------------------------------------------------------
# Categorical palette — Wong (2011), colorblind-safe, 8 colors
# ---------------------------------------------------------------------------
CATEGORICAL_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # pink
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#000000",  # black
]

# Named semantic colors
COLOR_PRIMARY = "#0072B2"
COLOR_NEGATIVE = "#D55E00"
COLOR_POSITIVE = "#009E73"
COLOR_MUTED = "#999999"

# ---------------------------------------------------------------------------
# Colorscales for continuous data
# ---------------------------------------------------------------------------
SEQUENTIAL_COLORSCALE = "Viridis"        # perceptually uniform, colorblind-safe
DIVERGING_COLORSCALE = "RdBu_r"          # centered at 0
HEAT_COLORSCALE = "YlOrRd"              # for magnitude/intensity

# ---------------------------------------------------------------------------
# Marker defaults
# ---------------------------------------------------------------------------
SCATTER_MARKER = dict(size=3, opacity=0.5)
SCATTER_MARKER_BG = dict(size=2, opacity=0.15, color=COLOR_MUTED)

# ---------------------------------------------------------------------------
# Layout template — clean, minimal chrome
# ---------------------------------------------------------------------------
LAYOUT_DEFAULTS = dict(
    font=dict(family="Arial, Helvetica, sans-serif", size=12),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=60, r=30, t=40, b=50),
    # Kill gridlines by default
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        linecolor="#333",
        linewidth=1,
        ticks="outside",
        ticklen=4,
        tickwidth=1,
        tickcolor="#333",
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        linecolor="#333",
        linewidth=1,
        ticks="outside",
        ticklen=4,
        tickwidth=1,
        tickcolor="#333",
    ),
    # Legend: outside, no box
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
    ),
)


def apply_style(fig, *, title=None, xaxis_title=None, yaxis_title=None,
                height=None, width=None):
    """Apply PEACH style defaults to a plotly figure.

    Call this once at the end of every plot function, before save/show.
    Specific overrides (e.g. ternary axes) can be applied after.
    """
    fig.update_layout(**LAYOUT_DEFAULTS)
    updates = {}
    if title is not None:
        updates["title"] = dict(text=title, x=0.02, xanchor="left",
                                font=dict(size=14))
    if xaxis_title is not None:
        updates["xaxis_title"] = xaxis_title
    if yaxis_title is not None:
        updates["yaxis_title"] = yaxis_title
    if height is not None:
        updates["height"] = height
    if width is not None:
        updates["width"] = width
    if updates:
        fig.update_layout(**updates)
    return fig


def save_and_show(fig, *, save_path=None, show=True):
    """Shared save/show logic."""
    if save_path:
        fig.write_html(save_path)
    if show:
        fig.show()
    return fig

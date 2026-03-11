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
    font=dict(family="Arial, Helvetica, sans-serif", size=11, color="#333"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=55, r=20, t=35, b=45),
    # Minimal axis chrome — Tufte range-frame aesthetic
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        linecolor="#aaa",
        linewidth=0.5,
        ticks="outside",
        ticklen=3,
        tickwidth=0.5,
        tickcolor="#aaa",
        tickfont=dict(size=10),
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        linecolor="#aaa",
        linewidth=0.5,
        ticks="outside",
        ticklen=3,
        tickwidth=0.5,
        tickcolor="#aaa",
        tickfont=dict(size=10),
    ),
    # Legend: no box, no background — invisible container
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
        font=dict(size=10),
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
                                font=dict(size=12, color="#555"))
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
    """Shared save/show logic. Infers format from file extension.

    - .html → interactive HTML (use for 3D plots)
    - .png, .pdf, .svg → static image via kaleido
    - No extension → defaults to .png
    """
    if save_path:
        import os
        _, ext = os.path.splitext(save_path)
        ext = ext.lower()
        if ext == ".html":
            fig.write_html(save_path)
        elif ext in (".png", ".pdf", ".svg", ".jpeg", ".jpg", ".webp"):
            fig.write_image(save_path)
        else:
            if not ext:
                save_path = save_path + ".png"
                fig.write_image(save_path)
            else:
                raise ValueError(
                    f"Unrecognized file extension '{ext}'. "
                    f"Supported: .html, .png, .pdf, .svg, .jpeg, .jpg, .webp"
                )
    if show:
        fig.show()
    return fig

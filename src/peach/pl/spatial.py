"""
Spatial archetype visualization functions.

Interactive plotly visualizations for spatial archetype analysis results.
Requires spatial analysis to have been run via ``pc.tl.spatial_neighbors()``.

Main Functions:
- nhood_enrichment(): Heatmap of neighborhood enrichment z-scores
- co_occurrence(): Line plot of distance-dependent co-occurrence ratios
- spatial_archetypes(): Spatial scatter plot of cells colored by archetype
- interaction_boundaries(): Spatial map of boundary scores between cell types
- spatial_autocorr(): Dot plot of Moran's I / Geary's C per archetype
- cross_correlations(): Diverging dot plot of per-archetype correlation between cell types
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go
from anndata import AnnData


def nhood_enrichment(
    adata: AnnData,
    *,
    uns_key: str = "archetype_nhood_enrichment",
    cluster_key: str = "archetypes",
    title: str = "Archetype Neighborhood Enrichment",
    colorscale: str = "RdBu_r",
    save_path: str | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Heatmap of archetype neighborhood enrichment z-scores.

    Positive values indicate spatial co-localization; negative values
    indicate spatial separation.

    Parameters
    ----------
    adata : AnnData
        Annotated data with neighborhood enrichment results.
    uns_key : str, default: "archetype_nhood_enrichment"
        Key in ``adata.uns`` containing enrichment results.
    cluster_key : str, default: "archetypes"
        Column in ``adata.obs`` with archetype labels (used for axis labels).
    title : str, default: "Archetype Neighborhood Enrichment"
        Plot title.
    colorscale : str, default: "RdBu_r"
        Plotly colorscale. RdBu_r: red=enriched, blue=depleted.
    save_path : str | None, default: None
        Path to save the figure as HTML.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if uns_key not in adata.uns:
        # Try squidpy's default key format
        squidpy_key = f"{cluster_key}_nhood_enrichment"
        if squidpy_key in adata.uns:
            uns_key = squidpy_key
        else:
            raise ValueError(
                f"No enrichment results found. Run pc.tl.archetype_nhood_enrichment(adata) first."
            )

    result = adata.uns[uns_key]
    zscore = result["zscore"]

    # Get archetype labels
    if cluster_key in adata.obs.columns:
        labels = sorted(adata.obs[cluster_key].unique())
    else:
        labels = [f"archetype_{i}" for i in range(zscore.shape[0])]

    # Symmetric color range centered on 0
    vmax = np.abs(zscore).max()

    fig = go.Figure(
        data=go.Heatmap(
            z=zscore,
            x=labels,
            y=labels,
            colorscale=colorscale,
            zmid=0,
            zmin=-vmax,
            zmax=vmax,
            text=np.round(zscore, 2),
            texttemplate="%{text}",
            colorbar=dict(title="z-score"),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Archetype",
        yaxis_title="Archetype",
        width=600,
        height=500,
        plot_bgcolor="white",
    )

    if save_path:
        fig.write_html(save_path)
        print(f"  Saved to {save_path}")

    fig.show()
    return fig


def co_occurrence(
    adata: AnnData,
    *,
    uns_key: str = "archetype_co_occurrence",
    cluster_key: str = "archetypes",
    title: str = "Archetype Spatial Co-occurrence",
    save_path: str | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Line plot of distance-dependent archetype co-occurrence ratios.

    Shows how the co-occurrence ratio between archetype pairs changes
    with spatial distance. Values > 1 indicate co-occurrence above
    chance; values < 1 indicate avoidance.

    Parameters
    ----------
    adata : AnnData
        Annotated data with co-occurrence results.
    uns_key : str, default: "archetype_co_occurrence"
        Key in ``adata.uns`` containing co-occurrence results.
    cluster_key : str, default: "archetypes"
        Column in ``adata.obs`` with archetype labels.
    title : str, default: "Archetype Spatial Co-occurrence"
        Plot title.
    save_path : str | None, default: None
        Path to save the figure as HTML.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if uns_key not in adata.uns:
        squidpy_key = f"{cluster_key}_co_occurrence"
        if squidpy_key in adata.uns:
            uns_key = squidpy_key
        else:
            raise ValueError(
                f"No co-occurrence results found. Run pc.tl.archetype_co_occurrence(adata) first."
            )

    result = adata.uns[uns_key]
    occ = result["occ"]
    interval = result["interval"]

    # Get archetype labels
    if cluster_key in adata.obs.columns:
        labels = sorted(adata.obs[cluster_key].unique())
    else:
        labels = [f"archetype_{i}" for i in range(occ.shape[0])]

    # Distance midpoints
    distances = (interval[:-1] + interval[1:]) / 2

    fig = go.Figure()

    # Plot each archetype pair
    n_archetypes = occ.shape[0]
    for i in range(n_archetypes):
        for j in range(i, n_archetypes):
            fig.add_trace(
                go.Scatter(
                    x=distances,
                    y=occ[i, j, :],
                    mode="lines",
                    name=f"{labels[i]} - {labels[j]}",
                    visible="legendonly" if i != j else True,  # show self-pairs by default
                )
            )

    # Reference line at ratio = 1 (expected by chance)
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="expected")

    fig.update_layout(
        title=title,
        xaxis_title="Spatial Distance",
        yaxis_title="Co-occurrence Ratio",
        width=800,
        height=500,
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#F0F0F0"),
        yaxis=dict(gridcolor="#F0F0F0"),
    )

    if save_path:
        fig.write_html(save_path)
        print(f"  Saved to {save_path}")

    fig.show()
    return fig


_ARCHETYPE_COLORS = [
    "#E64B35",  # red
    "#4DBBD5",  # teal
    "#00A087",  # green
    "#3C5488",  # navy
    "#F39B7F",  # salmon
    "#8491B4",  # slate
    "#91D1C2",  # mint
    "#DC9C6C",  # amber
    "#7E6148",  # brown
    "#B09C85",  # khaki
    "#E377C2",  # pink
    "#7F7F7F",  # gray (no_archetype)
]


def spatial_archetypes(
    adata: AnnData,
    *,
    spatial_key: str = "spatial",
    color_key: str = "archetypes",
    point_size: float = 2.0,
    opacity: float = 0.7,
    title: str = "Spatial Archetype Map",
    save_path: str | None = None,
    colors: list[str] | None = None,
    legend_marker_size: float = 12.0,
    **kwargs: Any,
) -> go.Figure:
    """Scatter plot of cells on spatial coordinates, colored by archetype.

    Parameters
    ----------
    adata : AnnData
        Annotated data with spatial coordinates and archetype assignments.
    spatial_key : str, default: "spatial"
        Key in ``adata.obsm`` with 2D spatial coordinates.
    color_key : str, default: "archetypes"
        Column in ``adata.obs`` to color cells by.
    point_size : float, default: 2.0
        Size of scatter points.
    opacity : float, default: 0.7
        Point opacity.
    title : str, default: "Spatial Archetype Map"
        Plot title.
    save_path : str | None, default: None
        Path to save the figure as HTML.
    colors : list[str] | None, default: None
        Custom color list. If None, uses a perceptually distinct palette
        designed for archetype visualization.
    legend_marker_size : float, default: 12.0
        Size of legend marker dots for readability.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if spatial_key not in adata.obsm:
        raise ValueError(f"No spatial coordinates at adata.obsm['{spatial_key}'].")

    coords = adata.obsm[spatial_key]

    if color_key not in adata.obs.columns:
        raise ValueError(f"'{color_key}' not found in adata.obs.")

    categories = sorted(adata.obs[color_key].unique())
    palette = colors if colors is not None else _ARCHETYPE_COLORS

    fig = go.Figure()

    for i, cat in enumerate(categories):
        mask = adata.obs[color_key] == cat
        color = palette[i % len(palette)]
        # Use light gray for no_archetype regardless of position
        if str(cat) == "no_archetype":
            color = "#D3D3D3"

        # Data trace (small markers, no legend entry)
        fig.add_trace(
            go.Scattergl(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                name=str(cat),
                marker=dict(size=point_size, opacity=opacity, color=color),
                legendgroup=str(cat),
                showlegend=False,
            )
        )

        # Legend-only trace (large marker, single invisible point)
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=str(cat),
                marker=dict(size=legend_marker_size, color=color),
                legendgroup=str(cat),
                showlegend=True,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Spatial X",
        yaxis_title="Spatial Y",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        width=800,
        height=700,
        legend=dict(
            itemsizing="constant",
            font=dict(size=12),
        ),
    )

    if save_path:
        fig.write_html(save_path)
        print(f"  Saved to {save_path}")

    fig.show()
    return fig


def interaction_boundaries(
    adata: AnnData,
    *,
    spatial_key: str = "spatial",
    score_key: str = "boundary_score",
    point_size: float = 2.0,
    colorscale: str = "Inferno",
    title: str | None = None,
    save_path: str | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Spatial map of interaction boundary scores between cell types.

    Colors cells by their boundary score from
    ``pc.tl.archetype_interaction_boundaries()``. High scores indicate
    cells at spatial fronts where archetype composition diverges
    between cell types.

    Parameters
    ----------
    adata : AnnData
        Annotated data with spatial coordinates and boundary scores.
    spatial_key : str, default: "spatial"
        Key in ``adata.obsm`` with 2D spatial coordinates.
    score_key : str, default: "boundary_score"
        Column in ``adata.obs`` with boundary scores.
    point_size : float, default: 2.0
        Size of scatter points.
    colorscale : str, default: "Inferno"
        Plotly colorscale. Inferno: dark=low, bright=high boundaries.
    title : str | None, default: None
        Plot title. Auto-generated from boundary result if None.
    save_path : str | None, default: None
        Path to save the figure as HTML.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if spatial_key not in adata.obsm:
        raise ValueError(f"No spatial coordinates at adata.obsm['{spatial_key}'].")
    if score_key not in adata.obs.columns:
        raise ValueError(
            f"'{score_key}' not found in adata.obs. "
            f"Run pc.tl.archetype_interaction_boundaries(adata) first."
        )

    coords = adata.obsm[spatial_key]
    scores = adata.obs[score_key].values

    if title is None:
        result = adata.uns.get("archetype_interaction_boundaries", {})
        ct_a = result.get("cell_type_a", "A")
        ct_b = result.get("cell_type_b", "B")
        title = f"Interaction Boundaries: {ct_a} vs {ct_b}"

    fig = go.Figure(
        data=go.Scattergl(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            marker=dict(
                size=point_size,
                color=scores,
                colorscale=colorscale,
                colorbar=dict(title="Boundary<br>Score"),
                opacity=0.8,
            ),
            hovertemplate="x: %{x:.0f}<br>y: %{y:.0f}<br>score: %{marker.color:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Spatial X",
        yaxis_title="Spatial Y",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        width=800,
        height=700,
    )

    if save_path:
        fig.write_html(save_path)
        print(f"  Saved to {save_path}")

    fig.show()
    return fig


def spatial_autocorr(
    adata: AnnData,
    *,
    uns_key: str = "archetype_spatial_autocorr",
    title: str | None = None,
    save_path: str | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Dot plot of spatial autocorrelation per archetype weight.

    Horizontal lollipop chart showing Moran's I (or Geary's C) for each
    archetype, ordered by value. Filled markers indicate significance
    (p < 0.05); open markers indicate non-significant.

    Parameters
    ----------
    adata : AnnData
        Annotated data with spatial autocorrelation results in
        ``adata.uns[uns_key]``.
    uns_key : str, default: "archetype_spatial_autocorr"
        Key in ``adata.uns`` with autocorrelation DataFrame.
    title : str | None, default: None
        Plot title. Auto-detected from data if None.
    save_path : str | None, default: None
        Path to save the figure as HTML.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if uns_key not in adata.uns:
        raise ValueError(
            f"No autocorrelation results found at adata.uns['{uns_key}']. "
            f"Run pc.tl.archetype_spatial_autocorr(adata) first."
        )

    df = adata.uns[uns_key]

    # Detect whether Moran's I or Geary's C
    if "I" in df.columns:
        stat_col, stat_name = "I", "Moran's I"
        ref_val = 0.0
        interpret_high = "spatially clustered"
        interpret_low = "spatially dispersed"
    elif "C" in df.columns:
        stat_col, stat_name = "C", "Geary's C"
        ref_val = 1.0
        interpret_high = "spatially dispersed"
        interpret_low = "spatially clustered"
    else:
        raise ValueError(f"Expected 'I' or 'C' column in autocorrelation DataFrame.")

    pval_col = "pval_norm" if "pval_norm" in df.columns else "pval_z_sim"

    # Sort by statistic value (highest = most spatially structured at top)
    df_sorted = df.sort_values(stat_col, ascending=True)
    archetypes = df_sorted.index.tolist()
    values = df_sorted[stat_col].values
    pvals = df_sorted[pval_col].values
    sig = pvals < 0.05

    if title is None:
        title = f"Spatial Autocorrelation of Archetype Weights"

    fig = go.Figure()

    # Lollipop stems
    for i, arch in enumerate(archetypes):
        fig.add_trace(
            go.Scatter(
                x=[ref_val, values[i]],
                y=[arch, arch],
                mode="lines",
                line=dict(color="#888888", width=1.5),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Significant points (filled)
    if sig.any():
        fig.add_trace(
            go.Scatter(
                x=values[sig],
                y=np.array(archetypes)[sig],
                mode="markers+text",
                marker=dict(size=10, color="#2C3E50", symbol="circle"),
                text=[f"  {v:.3f} (p={p:.1e})" for v, p in zip(values[sig], pvals[sig])],
                textposition="middle right",
                textfont=dict(size=10),
                name="significant (p<0.05)",
                showlegend=True,
                hovertemplate="%{y}<br>" + stat_name + ": %{x:.4f}<extra></extra>",
            )
        )

    # Non-significant points (open)
    if (~sig).any():
        fig.add_trace(
            go.Scatter(
                x=values[~sig],
                y=np.array(archetypes)[~sig],
                mode="markers+text",
                marker=dict(
                    size=10, color="white", symbol="circle",
                    line=dict(color="#2C3E50", width=1.5),
                ),
                text=[f"  {v:.3f} (n.s.)" for v in values[~sig]],
                textposition="middle right",
                textfont=dict(size=10, color="#999999"),
                name="not significant",
                showlegend=True,
                hovertemplate="%{y}<br>" + stat_name + ": %{x:.4f}<extra></extra>",
            )
        )

    # Reference line
    fig.add_vline(
        x=ref_val, line_dash="dash", line_color="#CCCCCC",
        annotation_text="no spatial pattern",
        annotation_font_color="#999999",
        annotation_font_size=10,
    )

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>{stat_name} per archetype weight — "
                 f"higher = {interpret_high}</sup>",
        ),
        xaxis_title=stat_name,
        yaxis_title="Archetype weight dimension",
        width=600,
        height=max(300, 55 * len(archetypes) + 120),
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#F0F0F0", zeroline=False),
        yaxis=dict(gridcolor="#F0F0F0"),
        margin=dict(l=120, t=80),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0.5, xanchor="center"),
    )

    if save_path:
        fig.write_html(save_path)
        print(f"  Saved to {save_path}")

    fig.show()
    return fig


def cross_correlations(
    adata: AnnData,
    *,
    uns_key: str = "archetype_interaction_boundaries",
    title: str | None = None,
    save_path: str | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Diverging dot plot of per-archetype cross-correlations between cell types.

    Shows Spearman r for each archetype dimension: how the local weight
    of that archetype in cell type A's neighbors correlates with its weight
    in cell type B's neighbors across space.

    Positive r (right): archetype co-localizes across cell types.
    Negative r (left): archetype anti-correlates — one type goes up
    while the other goes down in the same neighborhood.

    Parameters
    ----------
    adata : AnnData
        Annotated data with interaction boundary results in
        ``adata.uns[uns_key]``.
    uns_key : str, default: "archetype_interaction_boundaries"
        Key in ``adata.uns`` with boundary results dict.
    title : str | None, default: None
        Plot title. Auto-generated from cell type names if None.
    save_path : str | None, default: None
        Path to save the figure as HTML.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if uns_key not in adata.uns:
        raise ValueError(
            f"No boundary results at adata.uns['{uns_key}']. "
            f"Run pc.tl.archetype_interaction_boundaries(adata) first."
        )

    result = adata.uns[uns_key]
    df = result["cross_correlations"]
    ct_a = result.get("cell_type_a", "A")
    ct_b = result.get("cell_type_b", "B")

    if len(df) == 0:
        raise ValueError("No cross-correlations computed (too few cells with both types).")

    # Sort by absolute correlation (strongest at top)
    df_sorted = df.sort_values("spearman_r", key=abs, ascending=True)
    archetypes = df_sorted["archetype"].tolist()
    r_vals = df_sorted["spearman_r"].values
    pvals = df_sorted["pvalue"].values
    sig = pvals < 0.05

    # Include mean weights in hover and label for context
    has_mean_w = "mean_weight_a" in df_sorted.columns
    mean_wa = df_sorted["mean_weight_a"].values if has_mean_w else None
    mean_wb = df_sorted["mean_weight_b"].values if has_mean_w else None

    if title is None:
        title = f"Cross-Cell-Type Archetype Correlations"

    # Color by direction: blue for co-localization, red for anti-correlation
    colors = np.where(r_vals > 0, "#3182BD", "#E6550D")

    fig = go.Figure()

    # Stems from zero
    for i, arch in enumerate(archetypes):
        fig.add_trace(
            go.Scatter(
                x=[0, r_vals[i]],
                y=[arch, arch],
                mode="lines",
                line=dict(color=colors[i], width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Build labels: include significance and mean weights
    labels = []
    hover_texts = []
    for i in range(len(archetypes)):
        sig_mark = " *" if sig[i] else ""
        labels.append(f"  r={r_vals[i]:+.2f}{sig_mark}")
        hover_parts = [
            f"{archetypes[i]}",
            f"Spearman r: {r_vals[i]:+.3f}",
            f"p-value: {pvals[i]:.2e}",
        ]
        if has_mean_w:
            hover_parts.append(f"Mean weight in {ct_a}: {mean_wa[i]:.3f}")
            hover_parts.append(f"Mean weight in {ct_b}: {mean_wb[i]:.3f}")
        hover_texts.append("<br>".join(hover_parts))

    # Dots with direct labels
    fig.add_trace(
        go.Scatter(
            x=r_vals,
            y=archetypes,
            mode="markers+text",
            marker=dict(
                size=np.where(sig, 11, 8),
                color=colors,
                symbol=np.where(sig, "circle", "circle-open").tolist(),
                line=dict(color=colors, width=1.5),
            ),
            text=labels,
            textposition=np.where(r_vals >= 0, "middle right", "middle left").tolist(),
            textfont=dict(size=10),
            showlegend=False,
            hovertext=hover_texts,
            hoverinfo="text",
        )
    )

    # Zero reference
    fig.add_vline(x=0, line_color="#CCCCCC", line_width=1)

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>For each archetype, does its weight in "
                 f"nearby {ct_a} correlate with its weight in nearby {ct_b}?<br>"
                 f"r>0: both types enriched together | r&lt;0: one up, the other down"
                 f" | * p&lt;0.05</sup>",
        ),
        xaxis_title=f"Spearman r (local {ct_a} weight vs local {ct_b} weight)",
        xaxis=dict(range=[-1.05, 1.05], gridcolor="#F0F0F0", zeroline=False),
        yaxis=dict(gridcolor="#F0F0F0"),
        yaxis_title="Archetype weight dimension",
        width=650,
        height=max(300, 55 * len(archetypes) + 140),
        plot_bgcolor="white",
        margin=dict(l=120, t=100),
        annotations=[
            dict(
                x=-0.7, y=1.0, xref="x", yref="paper",
                text=f"{ct_a} and {ct_b}<br>in opposition",
                showarrow=False,
                font=dict(color="#E6550D", size=10),
                align="center",
            ),
            dict(
                x=0.7, y=1.0, xref="x", yref="paper",
                text=f"{ct_a} and {ct_b}<br>co-enriched",
                showarrow=False,
                font=dict(color="#3182BD", size=10),
                align="center",
            ),
        ],
    )

    if save_path:
        fig.write_html(save_path)
        print(f"  Saved to {save_path}")

    fig.show()
    return fig

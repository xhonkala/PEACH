"""Archetype comparison visualizations: MMD heatmap, contrast volcano, similarity heatmap."""

import numpy as np
import plotly.graph_objects as go
from anndata import AnnData

from ._style import (
    COLOR_NEGATIVE,
    COLOR_PRIMARY,
    DIVERGING_COLORSCALE,
    SEQUENTIAL_COLORSCALE,
    apply_style,
    save_and_show,
)


def mmd_heatmap(
    adata: AnnData,
    *,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Heatmap of K x K MMD matrix between archetypes.

    Parameters
    ----------
    adata : AnnData
        Must have MMD results in uns['peach_archetype_mmd'].
    save_path : str or None
    show : bool

    Returns
    -------
    go.Figure
    """
    mmd_data = adata.uns.get("peach_archetype_mmd")
    if mmd_data is None:
        raise ValueError("No MMD results. Run pc.tl.archetype_mmd() first.")

    mmd_matrix = np.asarray(mmd_data["mmd_matrix"])
    K = mmd_matrix.shape[0]
    labels = [f"A{i}" for i in range(K)]

    fig = go.Figure(data=go.Heatmap(
        z=mmd_matrix,
        x=labels,
        y=labels,
        colorscale=SEQUENTIAL_COLORSCALE,
        colorbar=dict(title="MMD", thickness=12, len=0.6),
    ))
    apply_style(fig, title="Archetype MMD similarity",
                xaxis_title="Archetype", yaxis_title="Archetype")
    return save_and_show(fig, save_path=save_path, show=show)


def contrast_volcano(
    adata: AnnData,
    pair: tuple[int, int],
    *,
    fdr_threshold: float = 0.05,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Volcano plot for one archetype pair contrast: delta-beta vs -log10(p).

    Parameters
    ----------
    adata : AnnData
        Must have contrast results in uns['peach_archetype_contrasts'].
    pair : tuple[int, int]
        Archetype pair (j, k).
    fdr_threshold : float
    save_path : str or None
    show : bool

    Returns
    -------
    go.Figure
    """
    contrast_data = adata.uns.get("peach_archetype_contrasts")
    if contrast_data is None:
        raise ValueError("No contrast results. Run pc.tl.archetype_contrasts() first.")

    pair_key = str(pair)
    delta = np.asarray(contrast_data["delta_beta"][pair_key])
    pvals = np.asarray(contrast_data["pvalues_fdr"][pair_key])
    names = list(contrast_data["feature_names"])

    neg_log_p = -np.log10(np.maximum(pvals, 1e-300))
    colors = [COLOR_NEGATIVE if p < fdr_threshold else COLOR_PRIMARY for p in pvals]

    # Error bars from delta_se (95% CI) if available
    se = np.asarray(contrast_data["delta_se"][pair_key]) if "delta_se" in contrast_data else None
    error_x_kwargs = {}
    if se is not None:
        error_x_kwargs = dict(
            error_x=dict(type="data", array=1.96 * se, visible=True,
                         width=0, thickness=0.5, color="rgba(0,0,0,0.15)"),
            customdata=se,
        )

    fig = go.Figure(data=go.Scatter(
        x=delta,
        y=neg_log_p,
        mode="markers",
        text=names,
        hovertemplate="%{text}<br>\u0394\u03b2=%{x:.3f}<br>-log10(q)=%{y:.1f}<extra></extra>",
        marker=dict(size=5, opacity=0.6, color=colors),
        **error_x_kwargs,
    ))

    fig.add_hline(y=-np.log10(fdr_threshold), line_dash="dot",
                  line_color="#999", line_width=1)

    j, k = pair
    apply_style(fig, title=f"Contrast: A{j} vs A{k}",
                xaxis_title=f"\u03b2_{j} \u2212 \u03b2_{k}",
                yaxis_title="-log\u2081\u2080(FDR q)")
    return save_and_show(fig, save_path=save_path, show=show)


def contrast_volcano_grid(
    adata: AnnData,
    *,
    fdr_threshold: float = 0.05,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Small-multiple grid of volcano plots for all pairwise Wald contrasts.

    Parameters
    ----------
    adata : AnnData
        Must have contrast results in uns['peach_archetype_contrasts'].
    fdr_threshold : float
    save_path : str or None
    show : bool

    Returns
    -------
    go.Figure
    """
    from plotly.subplots import make_subplots
    import math

    contrast_data = adata.uns.get("peach_archetype_contrasts")
    if contrast_data is None:
        raise ValueError("No contrast results. Run pc.tl.archetype_contrasts() first.")

    pairs = [tuple(p) if isinstance(p, list) else p for p in contrast_data["pairs"]]
    n_pairs = len(pairs)
    n_cols = min(3, n_pairs)
    n_rows = math.ceil(n_pairs / n_cols)

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"A{j} vs A{k}" for j, k in pairs],
        shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.04, vertical_spacing=0.08,
    )

    names = list(contrast_data["feature_names"])

    for idx, pair in enumerate(pairs):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        pair_key = str(pair) if str(pair) in contrast_data["delta_beta"] else str(tuple(pair))

        delta = np.asarray(contrast_data["delta_beta"][pair_key])
        pvals = np.asarray(contrast_data["pvalues_fdr"][pair_key])
        neg_log_p = -np.log10(np.maximum(pvals, 1e-300))
        colors = [COLOR_NEGATIVE if p < fdr_threshold else COLOR_PRIMARY for p in pvals]

        fig.add_trace(
            go.Scatter(
                x=delta, y=neg_log_p,
                mode="markers", text=names,
                marker=dict(size=4, opacity=0.5, color=colors),
                showlegend=False,
                hovertemplate="%{text}<br>Δβ=%{x:.3f}<br>-log10(q)=%{y:.1f}<extra></extra>",
            ),
            row=row, col=col,
        )
        fig.add_hline(y=-np.log10(fdr_threshold), line_dash="dot",
                      line_color="#999", line_width=0.5, row=row, col=col)

    apply_style(fig, title="Pairwise Wald Contrasts")
    fig.update_layout(height=250 * n_rows, width=300 * n_cols)
    return save_and_show(fig, save_path=save_path, show=show)


def feature_similarity_heatmap(
    adata: AnnData,
    *,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Heatmap of Spearman correlation between archetype beta vectors.

    Parameters
    ----------
    adata : AnnData
        Must have feature similarity results in
        uns['peach_archetype_feature_similarity'].
    save_path : str or None
    show : bool

    Returns
    -------
    go.Figure
    """
    sim_data = adata.uns.get("peach_archetype_feature_similarity")
    if sim_data is None:
        raise ValueError(
            "No feature similarity results. "
            "Run pc.tl.archetype_feature_similarity() first."
        )

    spearman = np.asarray(sim_data["spearman_matrix"])
    K = spearman.shape[0]
    labels = [f"A{i}" for i in range(K)]

    fig = go.Figure(data=go.Heatmap(
        z=spearman,
        x=labels,
        y=labels,
        colorscale=DIVERGING_COLORSCALE,
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(title="\u03c1", thickness=12, len=0.6),
    ))
    apply_style(fig, title="Archetype feature similarity (Spearman)",
                xaxis_title="Archetype", yaxis_title="Archetype")
    return save_and_show(fig, save_path=save_path, show=show)

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


def _add_top_labels(
    fig: go.Figure,
    delta: np.ndarray,
    pvals: np.ndarray,
    neg_log_p: np.ndarray,
    names: list[str],
    *,
    n_labels: int = 10,
    fdr_threshold: float = 0.05,
    xref: str = "x",
    yref: str = "y",
    font_size: int = 8,
    textangle: int = 0,
) -> None:
    """Annotate a volcano plot with text labels for top features.

    Selection strategy: among significant features (FDR < threshold), pick
    those with the largest absolute effect size. If fewer than n_labels are
    significant, fill remaining slots by lowest p-value regardless of
    significance.
    """
    abs_delta = np.abs(delta)

    # Significant mask
    sig_mask = pvals < fdr_threshold
    n_sig = int(sig_mask.sum())

    if n_sig >= n_labels:
        # Among significant, take top N by |delta|
        sig_indices = np.where(sig_mask)[0]
        top_within_sig = np.argsort(abs_delta[sig_indices])[::-1][:n_labels]
        label_indices = sig_indices[top_within_sig]
    else:
        # Take all significant, then fill by lowest p-value
        sig_indices = set(np.where(sig_mask)[0].tolist())
        remaining_n = n_labels - n_sig
        by_pval = np.argsort(pvals)
        extra = [i for i in by_pval if i not in sig_indices][:remaining_n]
        label_indices = np.array(list(sig_indices) + extra, dtype=int)

    for i in label_indices:
        fig.add_annotation(
            x=delta[i],
            y=neg_log_p[i],
            text=names[i],
            showarrow=False,
            xref=xref,
            yref=yref,
            font=dict(size=font_size, color="#333"),
            yshift=7,
            xanchor="center",
            textangle=textangle,
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
    labels = [f"A{i+1}" for i in range(K)]

    fig = go.Figure(data=go.Heatmap(
        z=mmd_matrix,
        x=labels,
        y=labels,
        colorscale=SEQUENTIAL_COLORSCALE,
        colorbar=dict(title="MMD", thickness=12, len=0.6),
    ))

    # Axis labels: between-fit vs within-fit
    is_between = mmd_data.get("is_between_fit", False)
    if is_between:
        x_title = "Fit B archetypes"
        y_title = "Fit A archetypes"
    else:
        x_title = "Archetypes"
        y_title = "Archetypes"

    apply_style(fig, title="Archetype MMD similarity",
                xaxis_title=x_title, yaxis_title=y_title)
    return save_and_show(fig, save_path=save_path, show=show)


def contrast_volcano(
    adata: AnnData,
    pair: tuple[int, int],
    *,
    fdr_threshold: float = 0.05,
    n_labels: int = 10,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Volcano plot for one archetype pair contrast: delta-beta vs -log10(p).

    Parameters
    ----------
    adata : AnnData
        Must have contrast results in uns['peach_archetype_contrasts'].
    pair : tuple[int, int]
        Archetype pair (j, k), 0-indexed.
    fdr_threshold : float
        Significance threshold for FDR-corrected p-values.
    n_labels : int
        Number of top features to label on the plot (by absolute effect size
        among significant features, falling back to lowest p-value).
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

    # Add text labels for top N features
    if n_labels > 0:
        _add_top_labels(fig, delta, pvals, neg_log_p, names,
                        n_labels=n_labels, fdr_threshold=fdr_threshold)

    j, k = pair
    apply_style(fig, title=f"Contrast: A{j+1} vs A{k+1}",
                xaxis_title=f"\u03b2\u2081 \u2212 \u03b2\u2082 (A{j+1} vs A{k+1})",
                yaxis_title="-log\u2081\u2080(FDR q)",
                width=750)
    return save_and_show(fig, save_path=save_path, show=show)


def contrast_volcano_grid(
    adata: AnnData,
    *,
    fdr_threshold: float = 0.05,
    n_labels: int = 3,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Small-multiple grid of volcano plots for all pairwise Wald contrasts.

    Parameters
    ----------
    adata : AnnData
        Must have contrast results in uns['peach_archetype_contrasts'].
    fdr_threshold : float
        Significance threshold for FDR-corrected p-values.
    n_labels : int
        Number of top features to label per subplot (by absolute effect size
        among significant features, falling back to lowest p-value).
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
        subplot_titles=[f"A{j+1} vs A{k+1}" for j, k in pairs],
        shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.04, vertical_spacing=0.12,
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
                hovertemplate="%{text}<br>\u0394\u03b2=%{x:.3f}<br>-log10(q)=%{y:.1f}<extra></extra>",
            ),
            row=row, col=col,
        )
        fig.add_hline(y=-np.log10(fdr_threshold), line_dash="dot",
                      line_color="#999", line_width=0.5, row=row, col=col)

        # Add text labels for top N features per subplot
        if n_labels > 0:
            _add_top_labels(fig, delta, pvals, neg_log_p, names,
                            n_labels=n_labels, fdr_threshold=fdr_threshold,
                            xref=f"x{idx+1}" if idx > 0 else "x",
                            yref=f"y{idx+1}" if idx > 0 else "y",
                            font_size=6, textangle=-45)

    apply_style(fig, title="Pairwise Wald Contrasts")
    grid_width = min(800, 270 * n_cols)
    fig.update_layout(height=250 * n_rows, width=grid_width)
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
    labels = [f"A{i+1}" for i in range(K)]

    # Format rho values as text annotations on each cell
    text_matrix = [[f"{spearman[i, j]:.2f}" for j in range(K)] for i in range(K)]

    fig = go.Figure(data=go.Heatmap(
        z=spearman,
        x=labels,
        y=labels,
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorscale=DIVERGING_COLORSCALE,
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(title="\u03c1", thickness=12, len=0.6),
    ))

    # Axis labels: between-fit vs within-fit
    is_between = sim_data.get("is_between_fit", False)
    if is_between:
        x_title = "Fit B archetypes"
        y_title = "Fit A archetypes"
    else:
        x_title = "Archetypes"
        y_title = "Archetypes"

    apply_style(fig, title="Archetype feature similarity (Spearman)",
                xaxis_title=x_title, yaxis_title=y_title)
    return save_and_show(fig, save_path=save_path, show=show)

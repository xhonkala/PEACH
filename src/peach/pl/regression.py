"""Regression visualization: coefficient heatmaps, R² plots, pattern summaries."""

import numpy as np
import plotly.graph_objects as go
from anndata import AnnData

from peach._core.utils.feature_utils import resolve_regression_result

from ._style import (
    CATEGORICAL_PALETTE,
    COLOR_PRIMARY,
    DIVERGING_COLORSCALE,
    apply_style,
    save_and_show,
)


def _get_regression_data(adata, feature_type="genes"):
    """Helper to extract regression results from adata.uns.

    Tries namespaced key first (e.g. peach_simplex_regression_genes),
    falls back to generic peach_simplex_regression.
    """
    result = resolve_regression_result(adata, prefer=feature_type)
    if result is None:
        raise ValueError(
            "No regression results found. Run pc.tl.feature_simplex_regression() first."
        )
    return result


def coefficient_heatmap(
    adata: AnnData,
    *,
    top_n: int = 50,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Heatmap of vertex coefficients (β_k) for top features by R².

    Parameters
    ----------
    adata : AnnData
        Must have regression results in ``uns['peach_simplex_regression']``.
    top_n : int
        Number of top features to display, ranked by R².
    save_path : str or None
        If provided, save figure as HTML to this path.
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    reg = _get_regression_data(adata)
    coefs = np.asarray(reg["vertex_coefficients"])  # [n_features, K]
    names = list(reg["feature_names"])
    r2 = np.asarray(reg["r_squared_degree1"])

    top_idx = np.argsort(r2)[-top_n:][::-1]
    top_coefs = coefs[top_idx]
    top_names = [names[i] for i in top_idx]

    K = coefs.shape[1]
    arch_names = [f"Archetype {k}" for k in range(K)]

    fig = go.Figure(data=go.Heatmap(
        z=top_coefs,
        x=arch_names,
        y=top_names,
        colorscale=DIVERGING_COLORSCALE,
        zmid=0,
        colorbar=dict(title="β", thickness=12, len=0.6),
    ))
    n_shown = min(top_n, len(top_names))
    apply_style(fig, title=f"Vertex coefficients — top {n_shown} by R²",
                height=max(400, n_shown * 18))

    return save_and_show(fig, save_path=save_path, show=show)


def interaction_heatmap(
    adata: AnnData,
    *,
    top_n: int = 50,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Heatmap of interaction coefficients (β_{jk}) for top features.

    Parameters
    ----------
    adata : AnnData
        Must have degree-2 regression results in ``uns['peach_simplex_regression']``.
    top_n : int
        Number of top features to display, ranked by R².
    save_path : str or None
        If provided, save figure as HTML to this path.
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    reg = _get_regression_data(adata)

    if reg.get("interaction_coefficients") is None:
        raise ValueError("No interaction coefficients found. Run with max_degree=2.")

    int_coefs = np.asarray(reg["interaction_coefficients"])
    names = list(reg["feature_names"])
    pairs = reg.get("interaction_pairs", [])
    r2 = np.asarray(reg.get("r_squared_degree2", reg["r_squared_degree1"]))

    top_idx = np.argsort(r2)[-top_n:][::-1]
    top_coefs = int_coefs[top_idx]
    top_names = [names[i] for i in top_idx]
    pair_names = [f"({p[0]},{p[1]})" for p in pairs]

    fig = go.Figure(data=go.Heatmap(
        z=top_coefs,
        x=pair_names,
        y=top_names,
        colorscale=DIVERGING_COLORSCALE,
        zmid=0,
        colorbar=dict(title="β_int", thickness=12, len=0.6),
    ))
    n_shown = min(top_n, len(top_names))
    apply_style(fig, title=f"Interaction coefficients — top {n_shown} by R²",
                height=max(400, n_shown * 18))

    return save_and_show(fig, save_path=save_path, show=show)


def r2_barplot(
    adata: AnnData,
    *,
    top_n: int = 50,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Horizontal bar plot of features ranked by R².

    Parameters
    ----------
    adata : AnnData
        Must have regression results in ``uns['peach_simplex_regression']``.
    top_n : int
        Number of top features to display.
    save_path : str or None
        If provided, save figure as HTML to this path.
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    reg = _get_regression_data(adata)
    r2 = np.asarray(reg["r_squared_degree1"])
    names = list(reg["feature_names"])

    top_idx = np.argsort(r2)[-top_n:][::-1]
    top_r2 = r2[top_idx]
    top_names = [names[i] for i in top_idx]

    # Reverse so highest is at top of horizontal bar chart
    fig = go.Figure(data=go.Bar(
        x=top_r2[::-1],
        y=top_names[::-1],
        orientation="h",
        marker_color=COLOR_PRIMARY,
    ))
    n_shown = min(top_n, len(top_names))
    apply_style(fig, title=f"Top {n_shown} features by R²",
                xaxis_title="R²",
                height=max(400, n_shown * 18))

    return save_and_show(fig, save_path=save_path, show=show)


def vertex_radar(
    adata: AnnData,
    feature: str,
    *,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Spider/radar plot of vertex coefficients for one feature.

    Parameters
    ----------
    adata : AnnData
        Must have regression results in ``uns['peach_simplex_regression']``.
    feature : str
        Feature name to visualize.
    save_path : str or None
        If provided, save figure as HTML to this path.
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    reg = _get_regression_data(adata)
    names = list(reg["feature_names"])
    coefs = np.asarray(reg["vertex_coefficients"])

    if feature not in names:
        raise ValueError(f"Feature '{feature}' not found in regression results.")

    idx = names.index(feature)
    betas = coefs[idx]
    K = len(betas)
    arch_names = [f"Archetype {k}" for k in range(K)]

    fig = go.Figure(data=go.Scatterpolar(
        r=list(betas) + [betas[0]],  # close the polygon
        theta=arch_names + [arch_names[0]],
        fill="toself",
        fillcolor=f"rgba(0, 114, 178, 0.15)",
        line=dict(color=COLOR_PRIMARY, width=2),
        name=feature,
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, gridcolor="#ddd", linewidth=0),
            angularaxis=dict(linewidth=0, gridcolor="#ddd"),
            bgcolor="white",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=60, t=40, b=40),
        font=dict(family="Arial, Helvetica, sans-serif", size=12),
    )
    apply_style(fig, title=feature)
    # Radar plots need the polar layout preserved, so re-apply polar specifics
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, gridcolor="#eee", linewidth=0),
            angularaxis=dict(linewidth=0, gridcolor="#eee"),
            bgcolor="white",
        ),
    )

    return save_and_show(fig, save_path=save_path, show=show)


def regression_volcano(
    adata: AnnData,
    *,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Scatter plot: R² vs max vertex contrast (max β − min β).

    Parameters
    ----------
    adata : AnnData
        Must have regression results in ``uns['peach_simplex_regression']``.
    save_path : str or None
        If provided, save figure as HTML to this path.
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    reg = _get_regression_data(adata)
    r2 = np.asarray(reg["r_squared_degree1"])
    coefs = np.asarray(reg["vertex_coefficients"])
    names = list(reg["feature_names"])

    contrast = np.ptp(coefs, axis=1)  # max - min per feature

    fig = go.Figure(data=go.Scatter(
        x=contrast,
        y=r2,
        mode="markers",
        text=names,
        hovertemplate="%{text}<br>R²=%{y:.3f}<br>Contrast=%{x:.2f}<extra></extra>",
        marker=dict(size=4, opacity=0.5, color=COLOR_PRIMARY),
    ))
    apply_style(fig, title="R² vs vertex contrast",
                xaxis_title="max β − min β",
                yaxis_title="R²")

    return save_and_show(fig, save_path=save_path, show=show)


def pattern_summary(
    adata: AnnData,
    *,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Bar chart of pattern counts from classification results.

    Parameters
    ----------
    adata : AnnData
        Must have pattern classification results in ``uns['peach_feature_patterns']``.
    save_path : str or None
        If provided, save figure as HTML to this path.
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    if "peach_feature_patterns" not in adata.uns:
        raise ValueError(
            "No pattern classification results found. "
            "Run pc.tl.classify_feature_patterns() first."
        )

    patterns = adata.uns["peach_feature_patterns"]
    counts = patterns["pattern_counts"]
    labels = list(counts.keys())
    values = list(counts.values())

    # One color per pattern type
    colors = [CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]
              for i in range(len(labels))]

    fig = go.Figure(data=go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
    ))
    apply_style(fig, title="Feature pattern distribution",
                yaxis_title="Count")

    return save_and_show(fig, save_path=save_path, show=show)

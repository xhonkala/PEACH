"""GMM decomposition visualization: component scatter, BIC curve, heatmap, stability."""

import numpy as np
import plotly.graph_objects as go
from anndata import AnnData


def _get_gmm_data(adata: AnnData) -> dict:
    """Helper to extract GMM results from adata.uns."""
    if "peach_gmm" not in adata.uns:
        raise ValueError(
            "No GMM results found. Run pc.tl.feature_simplex_decomposition() first."
        )
    return adata.uns["peach_gmm"]


def component_scatter(
    adata: AnnData,
    *,
    pca_key: str = "X_pca",
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """2D PCA scatter colored by GMM component.

    Parameters
    ----------
    adata : AnnData
        Must have GMM results in ``uns['peach_gmm']`` and labels in
        ``obsm['peach_gmm_labels']``.
    pca_key : str
        Key in ``obsm`` for PCA coordinates.
    save_path : str or None
        If provided, save figure as HTML to this path.
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    _get_gmm_data(adata)  # validate GMM results exist

    if "peach_gmm_labels" not in adata.obsm:
        raise ValueError("No GMM labels found in adata.obsm['peach_gmm_labels'].")

    labels = np.asarray(adata.obsm["peach_gmm_labels"]).ravel()

    if pca_key not in adata.obsm:
        raise ValueError(f"adata.obsm['{pca_key}'] not found.")

    pca = adata.obsm[pca_key]

    fig = go.Figure()
    unique_labels = np.unique(labels[labels >= 0])
    for label in unique_labels:
        mask = labels == label
        fig.add_trace(go.Scatter(
            x=pca[mask, 0],
            y=pca[mask, 1],
            mode="markers",
            marker={"size": 3, "opacity": 0.6},
            name=f"Component {label}",
        ))

    # Unassigned cells (-1)
    unassigned = labels == -1
    if np.any(unassigned):
        fig.add_trace(go.Scatter(
            x=pca[unassigned, 0],
            y=pca[unassigned, 1],
            mode="markers",
            marker={"size": 2, "opacity": 0.3, "color": "gray"},
            name="Unassigned",
        ))

    fig.update_layout(
        title="GMM Components in PCA Space",
        xaxis_title="PC1",
        yaxis_title="PC2",
    )

    if save_path:
        fig.write_html(save_path)
    if show:
        fig.show()
    return fig


def gmm_bic_curve(
    adata: AnnData,
    *,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Line plot of BIC vs number of components.

    Parameters
    ----------
    adata : AnnData
        Must have GMM results in ``uns['peach_gmm']``.
    save_path : str or None
        If provided, save figure as HTML to this path.
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    gmm = _get_gmm_data(adata)
    bic = np.asarray(gmm["bic_values"])
    n_range = np.asarray(gmm["n_components_tested"])
    optimal = int(gmm["n_components_optimal"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=n_range,
        y=bic,
        mode="lines+markers",
        name="BIC",
    ))
    fig.add_vline(
        x=optimal,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Optimal: {optimal}",
    )
    fig.update_layout(
        title="BIC Curve",
        xaxis_title="Number of Components",
        yaxis_title="BIC",
    )

    if save_path:
        fig.write_html(save_path)
    if show:
        fig.show()
    return fig


def component_heatmap(
    adata: AnnData,
    *,
    top_n: int = 20,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Heatmap of per-component feature profiles.

    Shows the top ``top_n`` features ranked by variance across components.

    Parameters
    ----------
    adata : AnnData
        Must have GMM results with feature profiles in ``uns['peach_gmm']``.
    top_n : int
        Number of top features to display, ranked by cross-component variance.
    save_path : str or None
        If provided, save figure as HTML to this path.
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    gmm = _get_gmm_data(adata)
    profiles = gmm.get("component_feature_profiles")
    if profiles is None:
        raise ValueError(
            "No feature profiles found. "
            "Run feature_simplex_decomposition() with characterize_features=True."
        )

    profiles = np.asarray(profiles)

    # Use top features by variance across components
    var = np.var(profiles, axis=0)
    n_features = profiles.shape[1]
    actual_top_n = min(top_n, n_features)
    top_idx = np.argsort(var)[-actual_top_n:][::-1]
    top_profiles = profiles[:, top_idx]

    # Feature names (try from regression results or var_names)
    feature_names = None
    if "peach_simplex_regression" in adata.uns:
        feature_names = adata.uns["peach_simplex_regression"].get("feature_names")
    if feature_names is not None and len(feature_names) == n_features:
        top_names = [feature_names[i] for i in top_idx]
    elif adata.n_vars == n_features:
        top_names = [adata.var_names[i] for i in top_idx]
    else:
        top_names = [f"Feature {i}" for i in top_idx]

    comp_names = [f"Comp {i}" for i in range(len(profiles))]

    fig = go.Figure(data=go.Heatmap(
        z=top_profiles,
        x=top_names,
        y=comp_names,
        colorscale="Viridis",
        colorbar={"title": "Mean value"},
    ))
    fig.update_layout(
        title=f"Component Feature Profiles (top {actual_top_n} by variance)",
        height=max(400, len(comp_names) * 40),
    )

    if save_path:
        fig.write_html(save_path)
    if show:
        fig.show()
    return fig


def component_stability(
    adata: AnnData,
    *,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Bar plot of component stability scores.

    Parameters
    ----------
    adata : AnnData
        Must have GMM results in ``uns['peach_gmm']``.
    save_path : str or None
        If provided, save figure as HTML to this path.
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    gmm = _get_gmm_data(adata)
    scores = np.asarray(gmm["component_stability_scores"])
    n = len(scores)

    fig = go.Figure(data=go.Bar(
        x=[f"Comp {i}" for i in range(n)],
        y=scores,
        marker_color="steelblue",
    ))
    fig.add_hline(
        y=0.7,
        line_dash="dash",
        line_color="red",
        annotation_text="Stability threshold",
    )
    fig.update_layout(
        title="Component Stability Scores",
        yaxis_title="Stability",
        yaxis_range=[0, 1.05],
    )

    if save_path:
        fig.write_html(save_path)
    if show:
        fig.show()
    return fig

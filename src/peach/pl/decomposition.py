"""GMM decomposition visualization: component scatter, BIC curve, heatmap, stability."""

import numpy as np
import plotly.graph_objects as go
from anndata import AnnData

from ._style import (
    CATEGORICAL_PALETTE,
    COLOR_MUTED,
    COLOR_NEGATIVE,
    COLOR_PRIMARY,
    SCATTER_MARKER,
    SEQUENTIAL_COLORSCALE,
    apply_style,
    save_and_show,
)


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
    for i, label in enumerate(unique_labels):
        mask = labels == label
        color = CATEGORICAL_PALETTE[int(i) % len(CATEGORICAL_PALETTE)]
        fig.add_trace(go.Scatter(
            x=pca[mask, 0],
            y=pca[mask, 1],
            mode="markers",
            marker=dict(**SCATTER_MARKER, color=color),
            name=f"Component {int(label)}",
        ))

    # Unassigned cells (-1)
    unassigned = labels == -1
    if np.any(unassigned):
        fig.add_trace(go.Scatter(
            x=pca[unassigned, 0],
            y=pca[unassigned, 1],
            mode="markers",
            marker=dict(size=2, opacity=0.15, color=COLOR_MUTED),
            name="Unassigned",
        ))

    apply_style(fig, title="GMM components",
                xaxis_title="PC1", yaxis_title="PC2")

    return save_and_show(fig, save_path=save_path, show=show)


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
        line=dict(color=COLOR_PRIMARY, width=2),
        marker=dict(size=6, color=COLOR_PRIMARY),
        name="BIC",
    ))
    # Mark optimal with a single dot, no annotation text clutter
    opt_idx = list(n_range).index(optimal) if optimal in n_range else None
    if opt_idx is not None:
        fig.add_trace(go.Scatter(
            x=[optimal],
            y=[bic[opt_idx]],
            mode="markers",
            marker=dict(size=12, color=COLOR_NEGATIVE,
                        symbol="circle-open", line=dict(width=2)),
            name=f"Optimal (k={optimal})",
            showlegend=True,
        ))

    apply_style(fig, title="BIC curve",
                xaxis_title="Components", yaxis_title="BIC")

    return save_and_show(fig, save_path=save_path, show=show)


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
    from peach._core.utils.feature_utils import resolve_regression_result
    _reg = resolve_regression_result(adata, prefer="genes")
    if _reg is not None:
        feature_names = _reg.get("feature_names")
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
        colorscale=SEQUENTIAL_COLORSCALE,
        colorbar=dict(title="Mean", thickness=12, len=0.6),
    ))
    apply_style(fig, title=f"Component feature profiles — top {actual_top_n}",
                height=max(300, len(comp_names) * 40))

    return save_and_show(fig, save_path=save_path, show=show)


def component_archetype_summary(
    adata: AnnData,
    *,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """2x2 panel: component sizes, weight profiles, archetype distances, entropy.

    Parameters
    ----------
    adata : AnnData
        Must have GMM results in ``uns['peach_gmm']`` and archetype weights
        in ``obsm['cell_archetype_weights']``.
    save_path : str or None
    show : bool

    Returns
    -------
    go.Figure
    """
    from plotly.subplots import make_subplots
    from scipy.stats import entropy as sp_entropy

    gmm = _get_gmm_data(adata)
    assignments = np.asarray(gmm["component_assignments"])
    n_stable = int(gmm["n_components_stable"])
    simplex_means = np.asarray(gmm["component_simplex_means"])  # [n_comp, K]

    weights = adata.obsm.get("cell_archetype_weights")
    if weights is None:
        raise ValueError("No archetype weights in adata.obsm.")
    weights = np.asarray(weights)
    K = weights.shape[1]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Component sizes", "Weight profiles",
                        "Archetype proximity", "Weight entropy"],
        horizontal_spacing=0.14, vertical_spacing=0.16,
    )

    comp_labels = [f"C{i}" for i in range(n_stable)]
    arch_labels = [f"A{k+1}" for k in range(K)]

    # Assign each component a color by its dominant archetype
    dominant_arch = np.argmax(simplex_means[:n_stable], axis=1)
    comp_colors = [
        CATEGORICAL_PALETTE[int(dominant_arch[i]) % len(CATEGORICAL_PALETTE)]
        for i in range(n_stable)
    ]

    # Panel 1: Component sizes (bar), colored by dominant archetype
    sizes = [int(np.sum(assignments == c)) for c in range(n_stable)]
    fig.add_trace(go.Bar(x=comp_labels, y=sizes, marker_color=comp_colors,
                         showlegend=False), row=1, col=1)

    # Panel 2: Weight profiles (heatmap of simplex means)
    fig.add_trace(go.Heatmap(
        z=simplex_means[:n_stable],
        x=arch_labels, y=comp_labels,
        colorscale=SEQUENTIAL_COLORSCALE,
        colorbar=dict(title="Weight", thickness=10, len=0.4, y=0.8),
        showscale=True,
    ), row=1, col=2)

    # Panel 3: Mean distance to nearest archetype per component
    max_w = weights.max(axis=1)  # proximity = max weight
    mean_prox = [float(max_w[assignments == c].mean()) if np.any(assignments == c) else 0
                 for c in range(n_stable)]
    fig.add_trace(go.Bar(x=comp_labels, y=mean_prox, marker_color=comp_colors,
                         showlegend=False), row=2, col=1)

    # Panel 4: Weight entropy distribution per component (box)
    eps = 1e-10
    cell_entropy = sp_entropy(weights + eps, axis=1)
    for c in range(n_stable):
        mask = assignments == c
        if not np.any(mask):
            continue
        fig.add_trace(go.Box(
            y=cell_entropy[mask], name=f"C{c}",
            marker_color=CATEGORICAL_PALETTE[int(dominant_arch[c]) % len(CATEGORICAL_PALETTE)],
            showlegend=False, boxmean=True,
        ), row=2, col=2)

    # Add invisible legend traces for archetype color mapping
    for k in range(K):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=8, color=CATEGORICAL_PALETTE[k % len(CATEGORICAL_PALETTE)]),
            name=f"A{k+1}",
            showlegend=True,
        ))

    apply_style(fig, title="GMM Component Summary — colored by dominant archetype")
    fig.update_layout(
        height=650, width=850,
        legend=dict(
            title=dict(text="Dominant<br>archetype", font=dict(size=10)),
            orientation="v", x=1.02, y=0.35, xanchor="left",
            bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=10),
        ),
    )
    fig.update_xaxes(title_text="Component", row=1, col=1)
    fig.update_yaxes(title_text="Cells", row=1, col=1)
    fig.update_xaxes(title_text="Archetype", row=1, col=2)
    fig.update_xaxes(title_text="Component", row=2, col=1)
    fig.update_yaxes(title_text="Max weight", row=2, col=1)
    fig.update_xaxes(title_text="Component", row=2, col=2)
    fig.update_yaxes(title_text="Entropy", row=2, col=2)

    return save_and_show(fig, save_path=save_path, show=show)


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

    # Color bars by whether they meet the stability threshold
    colors = [COLOR_PRIMARY if s >= 0.7 else COLOR_MUTED for s in scores]

    fig = go.Figure(data=go.Bar(
        x=[f"Comp {i}" for i in range(n)],
        y=scores,
        marker_color=colors,
    ))
    fig.add_hline(
        y=0.7,
        line_dash="dot",
        line_color=COLOR_MUTED,
        line_width=1,
    )
    apply_style(fig, title="Component stability",
                yaxis_title="Stability score")
    fig.update_yaxes(range=[0, 1.05])

    return save_and_show(fig, save_path=save_path, show=show)


def component_neighborhood_graph(
    adata: AnnData,
    *,
    edge_threshold: float | None = None,
    show: bool = True,
    save: str | None = None,
) -> go.Figure:
    """3D network graph of GMM components in archetypal weight space.

    Nodes are GMM components positioned at their weight-space centroids,
    sized by cell count and colored by dominant archetype. Edges connect
    components whose centroids are within ``edge_threshold`` Euclidean
    distance, revealing the topology of subpopulation relationships on
    the simplex.

    Parameters
    ----------
    adata : AnnData
        Must have GMM results in ``uns['peach_gmm']`` with at least
        ``component_assignments`` and one of ``component_weight_means``
        or ``component_simplex_means``.
    edge_threshold : float or None
        Maximum Euclidean distance (in weight space) for drawing an edge.
        Default: median pairwise distance between component centroids.
    show : bool
        Whether to call ``fig.show()``.
    save : str or None
        If provided, save figure to this path (HTML recommended for 3D).

    Returns
    -------
    go.Figure
        Interactive 3D plotly figure.
    """
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import entropy as sp_entropy

    gmm = _get_gmm_data(adata)
    assignments = np.asarray(gmm["component_assignments"])
    n_stable = int(gmm["n_components_stable"])

    # Get component centroids — prefer weight_means, fall back to simplex_means
    centroids_raw = gmm.get("component_weight_means")
    if centroids_raw is None:
        centroids_raw = gmm.get("component_simplex_means")
    if centroids_raw is None:
        raise ValueError(
            "No component centroids found. Need 'component_weight_means' or "
            "'component_simplex_means' in adata.uns['peach_gmm']."
        )
    centroids_raw = np.asarray(centroids_raw)[:n_stable]
    K = centroids_raw.shape[1]

    # Project to 3D using first 3 weight dimensions
    centroids_3d = centroids_raw[:, :3]

    # Archetype map for coloring
    archetype_map = gmm.get("component_archetype_map")
    if archetype_map is not None:
        archetype_map = np.asarray(archetype_map)[:n_stable].astype(int)
    else:
        # Fall back: dominant archetype = argmax of centroid weights
        archetype_map = np.argmax(centroids_raw, axis=1)

    # Pairwise distances in full weight space (not just the 3D projection)
    if n_stable < 2:
        dist_matrix = np.zeros((1, 1))
    else:
        dist_matrix = squareform(pdist(centroids_raw, metric="euclidean"))

    # Determine edge threshold
    if edge_threshold is None:
        if n_stable < 2:
            edge_threshold = 1.0
        else:
            pairwise_dists = pdist(centroids_raw, metric="euclidean")
            edge_threshold = float(np.median(pairwise_dists))

    # Build edge list
    edges = []
    for i in range(n_stable):
        for j in range(i + 1, n_stable):
            if dist_matrix[i, j] <= edge_threshold:
                edges.append((i, j))

    # Node sizes: sqrt(n_cells) scaled to [8, 25]
    cell_counts = np.array(
        [int(np.sum(assignments == c)) for c in range(n_stable)]
    )
    sqrt_counts = np.sqrt(cell_counts.astype(float))
    if sqrt_counts.max() > sqrt_counts.min():
        node_sizes = 8 + 17 * (sqrt_counts - sqrt_counts.min()) / (
            sqrt_counts.max() - sqrt_counts.min()
        )
    else:
        node_sizes = np.full(n_stable, 16.0)

    # Node colors by dominant archetype
    node_colors = [
        CATEGORICAL_PALETTE[int(archetype_map[c]) % len(CATEGORICAL_PALETTE)]
        for c in range(n_stable)
    ]

    # Per-component entropy for hover info
    weights = adata.obsm.get("cell_archetype_weights")
    comp_entropies = np.full(n_stable, np.nan)
    if weights is not None:
        weights = np.asarray(weights)
        eps = 1e-10
        cell_entropy = sp_entropy(weights + eps, axis=1)
        for c in range(n_stable):
            mask = assignments == c
            if np.any(mask):
                comp_entropies[c] = float(cell_entropy[mask].mean())

    # Build figure
    fig = go.Figure()

    # Cell scatter background — low-alpha cloud showing data distribution
    cell_weights = adata.obsm.get("cell_archetype_weights")
    if cell_weights is not None:
        cell_weights = np.asarray(cell_weights)
        cell_3d = cell_weights[:, :3]
        fig.add_trace(go.Scatter3d(
            x=cell_3d[:, 0],
            y=cell_3d[:, 1],
            z=cell_3d[:, 2],
            mode="markers",
            marker=dict(size=1.5, color=COLOR_MUTED, opacity=0.03),
            hoverinfo="skip",
            name="Cells",
            showlegend=True,
        ))

    # Edges — draw individual edges so width can vary by distance
    if edges:
        edge_dists = [dist_matrix[i, j] for i, j in edges]
        min_d = min(edge_dists)
        max_d = max(edge_dists)

        for idx, (i, j) in enumerate(edges):
            d = dist_matrix[i, j]
            if max_d > min_d:
                # Invert: closer = thicker, range [1, 5]
                w = 1 + 4 * (1 - (d - min_d) / (max_d - min_d))
            else:
                w = 3
            fig.add_trace(go.Scatter3d(
                x=[centroids_3d[i, 0], centroids_3d[j, 0]],
                y=[centroids_3d[i, 1], centroids_3d[j, 1]],
                z=[centroids_3d[i, 2], centroids_3d[j, 2]],
                mode="lines",
                line=dict(color=COLOR_MUTED, width=w),
                showlegend=False,
                hoverinfo="skip",
            ))

    # Hover text for nodes
    hover_texts = []
    for c in range(n_stable):
        parts = [
            f"<b>C{c}</b>",
            f"Cells: {cell_counts[c]:,}",
            f"Dominant archetype: A{int(archetype_map[c]) + 1}",
        ]
        if not np.isnan(comp_entropies[c]):
            parts.append(f"Mean entropy: {comp_entropies[c]:.3f}")
        hover_texts.append("<br>".join(parts))

    # Component nodes — one trace per dominant archetype for legend grouping
    unique_archetypes = sorted(set(int(archetype_map[c]) for c in range(n_stable)))
    for arch_idx in unique_archetypes:
        comp_mask = [c for c in range(n_stable) if int(archetype_map[c]) == arch_idx]
        arch_color = CATEGORICAL_PALETTE[arch_idx % len(CATEGORICAL_PALETTE)]
        fig.add_trace(go.Scatter3d(
            x=centroids_3d[comp_mask, 0],
            y=centroids_3d[comp_mask, 1],
            z=centroids_3d[comp_mask, 2],
            mode="markers+text",
            marker=dict(
                size=[node_sizes[c] for c in comp_mask],
                color=arch_color,
                opacity=0.85,
                line=dict(width=1, color="#333"),
            ),
            text=[f"C{c}" for c in comp_mask],
            textposition="top center",
            textfont=dict(size=9, color="#333"),
            hovertext=[hover_texts[c] for c in comp_mask],
            hoverinfo="text",
            name=f"A{arch_idx + 1} dominant",
            showlegend=True,
            legendgroup=f"arch_{arch_idx}",
        ))

    # Archetype reference vertices — identity basis projected to 3D
    # For K >= 3, the first 3 archetypes sit at (1,0,0), (0,1,0), (0,0,1).
    # For archetypes beyond the 3rd, project their K-dim identity vector
    # into the first 3 dimensions (all zeros for k >= 3).
    arch_positions = np.eye(K)[:, :3]

    for k in range(K):
        arch_color = CATEGORICAL_PALETTE[k % len(CATEGORICAL_PALETTE)]
        fig.add_trace(go.Scatter3d(
            x=[arch_positions[k, 0]],
            y=[arch_positions[k, 1]],
            z=[arch_positions[k, 2]],
            mode="markers+text",
            marker=dict(
                size=10,
                color=arch_color,
                symbol="diamond",
                opacity=0.6,
                line=dict(width=1.5, color="#555"),
            ),
            text=[f"A{k+1}"],
            textposition="bottom center",
            textfont=dict(size=10, color="#555"),
            hoverinfo="text",
            hovertext=[f"<b>Archetype A{k+1}</b>"],
            name=f"A{k+1} vertex",
            showlegend=True,
            legendgroup=f"arch_{k}",
        ))

    # Style
    apply_style(fig, title="GMM Component Neighborhood Graph")
    fig.update_layout(
        height=650,
        width=700,
        legend=dict(
            title=dict(text="Legend", font=dict(size=10)),
            x=1.0, y=0.95, xanchor="left",
            bgcolor="rgba(255,255,255,0.8)", borderwidth=0,
            font=dict(size=9),
        ),
        scene=dict(
            xaxis_title="A1 weight",
            yaxis_title="A2 weight",
            zaxis_title="A3 weight",
            xaxis=dict(
                showgrid=True, gridcolor="#eee", gridwidth=0.5,
                zeroline=False, linecolor="#aaa", linewidth=0.5,
            ),
            yaxis=dict(
                showgrid=True, gridcolor="#eee", gridwidth=0.5,
                zeroline=False, linecolor="#aaa", linewidth=0.5,
            ),
            zaxis=dict(
                showgrid=True, gridcolor="#eee", gridwidth=0.5,
                zeroline=False, linecolor="#aaa", linewidth=0.5,
            ),
            bgcolor="white",
        ),
    )

    return save_and_show(fig, save_path=save, show=show)

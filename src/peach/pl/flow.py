"""Flow matching visualization."""

import logging
from itertools import combinations

import numpy as np
import plotly.graph_objects as go
from anndata import AnnData

from peach._core.utils.feature_utils import resolve_regression_result

logger = logging.getLogger(__name__)

from ._style import (
    CATEGORICAL_PALETTE,
    COLOR_MUTED,
    COLOR_NEGATIVE,
    COLOR_POSITIVE,
    COLOR_PRIMARY,
    DIVERGING_COLORSCALE,
    HEAT_COLORSCALE,
    SCATTER_MARKER_BG,
    apply_style,
    save_and_show,
)


def velocity_quiver(
    adata: AnnData,
    flow_result,
    *,
    pca_key: str = "X_pca",
    n_arrows: int = 200,
    arrow_alpha: float = 0.3,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """2D PCA quiver plot showing transport direction.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with PCA coordinates.
    flow_result : FlowWithinResult
        Result from ``pc.tl.flow_within()``.
    pca_key : str
        Key in ``adata.obsm`` for PCA coordinates.
    n_arrows : int
        Number of arrows to draw (subsampled from source cells).
    arrow_alpha : float
        Opacity for arrow color (0.0 = transparent, 1.0 = opaque).
    save_path : str or None
        If provided, save figure to this path (format inferred from extension).
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    pca = adata.obsm[pca_key]
    source_pca = pca[flow_result["source_mask"]]
    transported = flow_result["transported"]

    # Subsample
    n = min(n_arrows, len(source_pca))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(source_pca), size=n, replace=False)

    x = source_pca[idx, 0]
    y = source_pca[idx, 1]
    dx = transported[idx, 0] - x
    dy = transported[idx, 1] - y

    # Convert hex arrow color to rgba with specified alpha
    hex_color = COLOR_NEGATIVE.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    arrow_rgba = f"rgba({r},{g},{b},{arrow_alpha})"

    fig = go.Figure()

    # Background: all cells — minimal ink
    fig.add_trace(go.Scatter(
        x=pca[:, 0], y=pca[:, 1],
        mode="markers",
        marker=SCATTER_MARKER_BG,
        name="All cells",
        showlegend=False,
    ))

    # Arrows via annotations — full displacement, thin, single color
    for i in range(n):
        fig.add_annotation(
            x=x[i] + dx[i],
            y=y[i] + dy[i],
            ax=x[i], ay=y[i],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2, arrowsize=1,
            arrowwidth=0.8,
            arrowcolor=arrow_rgba,
        )

    apply_style(fig, title="Flow velocity field",
                xaxis_title="PC1", yaxis_title="PC2")

    return save_and_show(fig, save_path=save_path, show=show)


def gene_alignment_barplot(
    adata: AnnData,
    alignment_result,
    *,
    n_top: int = 20,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Top aligned and opposed genes horizontal bar plot.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (used for consistency with PEACH API).
    alignment_result : GeneAlignmentResult
        Result from ``pc.tl.flow_gene_alignment()``.
    n_top : int
        Number of top genes to show in each direction.
    save_path : str or None
        If provided, save figure to this path (format inferred from extension).
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    scores = alignment_result["alignment_scores"]
    names = alignment_result["gene_names"]

    sorted_idx = np.argsort(scores)
    top_aligned = sorted_idx[-n_top:][::-1]
    top_opposed = sorted_idx[:n_top]

    combined_idx = np.concatenate([top_opposed, top_aligned])
    combined_names = [names[i] for i in combined_idx]
    combined_scores = scores[combined_idx]

    colors = [COLOR_NEGATIVE if s < 0 else COLOR_PRIMARY for s in combined_scores]

    fig = go.Figure(data=go.Bar(
        x=combined_scores,
        y=combined_names,
        orientation="h",
        marker_color=colors,
    ))
    apply_style(fig, title=f"Gene–flow alignment (top {n_top} each)",
                xaxis_title="Alignment score",
                height=max(400, len(combined_names) * 15))

    return save_and_show(fig, save_path=save_path, show=show)


def jacobian_heatmap(
    adata: AnnData,
    jacobian_result,
    *,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Mean Jacobian matrix as heatmap.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (used for consistency with PEACH API).
    jacobian_result : FlowJacobianResult
        Result from ``pc.tl.flow_jacobian()``.
    save_path : str or None
        If provided, save figure to this path (format inferred from extension).
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    jac = jacobian_result["mean_jacobian"]
    dim = jac.shape[0]
    labels = [f"PC{i+1}" for i in range(dim)]

    fig = go.Figure(data=go.Heatmap(
        z=jac,
        x=labels,
        y=labels,
        colorscale=DIVERGING_COLORSCALE,
        zmid=0,
        colorbar=dict(title="∂v/∂x", thickness=12, len=0.6),
    ))
    t_val = jacobian_result["t"]
    apply_style(fig, title=f"Mean flow Jacobian (t={t_val:.2f})",
                xaxis_title="Input PC", yaxis_title="Output PC")

    return save_and_show(fig, save_path=save_path, show=show)


def trajectory_ribbon(
    adata: AnnData,
    flow_result,
    flow_model=None,
    *,
    n_sample: int = 100,
    n_steps: int = 20,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Sample of transported cells colored by time step.

    If ``flow_model`` is provided, recomputes the full trajectory.
    Otherwise, falls back to linear interpolation between source
    and transported positions.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with PCA coordinates.
    flow_result : FlowWithinResult
        Result from ``pc.tl.flow_within()``.
    flow_model : FlowModel or None
        Trained flow model. If provided, full trajectory is computed.
    n_sample : int
        Number of cells to sample for the trajectory.
    n_steps : int
        Number of time steps for the trajectory.
    save_path : str or None
        If provided, save figure to this path (format inferred from extension).
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    pca = adata.obsm[flow_result["pca_key"]]
    source_pca = pca[flow_result["source_mask"]]

    # Subsample
    rng = np.random.default_rng(42)
    n = min(n_sample, len(source_pca))
    idx = rng.choice(len(source_pca), size=n, replace=False)

    if flow_model is not None:
        traj = flow_model.transport(
            source_pca[idx], n_steps=n_steps, return_trajectory=True
        )
    else:
        # Linear interpolation fallback
        transported = flow_result["transported"][idx]
        source = source_pca[idx]
        times = np.linspace(0, 1, n_steps + 1)
        traj = np.stack([(1 - t) * source + t * transported for t in times])

    fig = go.Figure()

    # Sequential blue→red using a perceptual ramp
    n_frames = len(traj)
    for step in range(n_frames):
        t_val = step / max(n_frames - 1, 1)
        # Interpolate from blue (COLOR_PRIMARY) to vermillion (COLOR_NEGATIVE)
        r = int(0 + t_val * 213)
        g = int(114 - t_val * 114)
        b = int(178 - t_val * 178)
        color = f"rgb({r},{g},{b})"

        show_in_legend = step % max(1, n_frames // 5) == 0
        fig.add_trace(go.Scatter(
            x=traj[step, :, 0],
            y=traj[step, :, 1],
            mode="markers",
            marker=dict(size=2, opacity=0.4, color=color),
            name=f"t={t_val:.2f}" if show_in_legend else "",
            showlegend=show_in_legend,
        ))

    apply_style(fig, title="Flow trajectory",
                xaxis_title="PC1", yaxis_title="PC2")

    return save_and_show(fig, save_path=save_path, show=show)


def flow_magnitude(
    adata: AnnData,
    flow_result,
    *,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """2D scatter colored by transport magnitude (‖transported − source‖).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with PCA coordinates.
    flow_result : FlowWithinResult
        Result from ``pc.tl.flow_within()``.
    save_path : str or None
        If provided, save figure to this path (format inferred from extension).
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    pca = adata.obsm[flow_result["pca_key"]]
    source_pca = pca[flow_result["source_mask"]]
    transported = flow_result["transported"]

    magnitude = np.linalg.norm(transported - source_pca, axis=1)

    fig = go.Figure(data=go.Scatter(
        x=source_pca[:, 0],
        y=source_pca[:, 1],
        mode="markers",
        hovertemplate="PC1=%{x:.2f}<br>PC2=%{y:.2f}<br>‖Δ‖=%{marker.color:.3f}<extra></extra>",
        marker=dict(
            size=3,
            opacity=0.6,
            color=magnitude,
            colorscale=HEAT_COLORSCALE,
            colorbar=dict(title="‖Δ‖", thickness=12, len=0.6),
        ),
    ))
    apply_style(fig, title="Transport magnitude",
                xaxis_title="PC1", yaxis_title="PC2")

    return save_and_show(fig, save_path=save_path, show=show)


def density_comparison(
    adata: AnnData,
    flow_result,
    *,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """KDE comparison: source vs transported vs target in PC1.

    Displays overlaid histograms (probability density) for the
    source, transported, and target distributions in the first
    principal component.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with PCA coordinates.
    flow_result : FlowWithinResult
        Result from ``pc.tl.flow_within()``.
    save_path : str or None
        If provided, save figure to this path (format inferred from extension).
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    pca = adata.obsm[flow_result["pca_key"]]
    source_pc1 = pca[flow_result["source_mask"], 0]
    target_pc1 = pca[flow_result["target_mask"], 0]
    transported_pc1 = flow_result["transported"][:, 0]

    fig = go.Figure()
    for data, name, color in [
        (source_pc1, "Source", COLOR_PRIMARY),
        (transported_pc1, "Transported", COLOR_POSITIVE),
        (target_pc1, "Target", COLOR_NEGATIVE),
    ]:
        fig.add_trace(go.Histogram(
            x=data,
            name=name,
            opacity=0.5,
            marker_color=color,
            nbinsx=50,
            histnorm="probability density",
        ))

    apply_style(fig, title="Density comparison (PC1)",
                xaxis_title="PC1", yaxis_title="Density")
    fig.update_layout(barmode="overlay")

    return save_and_show(fig, save_path=save_path, show=show)


def archetype_correspondence(
    flow_between_result,
    *,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """K_src × K_tgt heatmap for archetype correspondence.

    Parameters
    ----------
    flow_between_result : FlowBetweenResult
        Result from ``pc.tl.flow_between()`` with archetype
        correspondence computed.
    save_path : str or None
        If provided, save figure to this path (format inferred from extension).
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    if flow_between_result["archetype_correspondence"] is None:
        raise ValueError("No archetype correspondence computed.")

    fig = go.Figure()
    for pair, corr_matrix in flow_between_result["archetype_correspondence"].items():
        src, tgt = pair
        K_src, K_tgt = corr_matrix.shape
        fig.add_trace(go.Heatmap(
            z=corr_matrix,
            x=[f"Target {i}" for i in range(K_tgt)],
            y=[f"Source {i}" for i in range(K_src)],
            colorscale="Blues",
            colorbar=dict(title="Weight", thickness=12, len=0.6),
        ))
        break  # Show first pair only

    apply_style(fig, title="Archetype correspondence",
                xaxis_title="Target archetypes",
                yaxis_title="Source archetypes")

    return save_and_show(fig, save_path=save_path, show=show)


# ---------------------------------------------------------------------------
# Helpers for soft_assignment_flow
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert '#RRGGBB' to (r, g, b) ints."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _blend_rgb(
    rgb_a: tuple[int, int, int],
    rgb_b: tuple[int, int, int],
    weight_a: float = 0.5,
    opacity: float = 0.45,
) -> str:
    """Blend two RGB colors and return an rgba string."""
    r = int(rgb_a[0] * weight_a + rgb_b[0] * (1 - weight_a))
    g = int(rgb_a[1] * weight_a + rgb_b[1] * (1 - weight_a))
    b = int(rgb_a[2] * weight_a + rgb_b[2] * (1 - weight_a))
    return f"rgba({r},{g},{b},{opacity})"


def _clean_feature_name(name: str, feature_type: str) -> str:
    """Strip verbose prefixes for readability."""
    if feature_type == "pathways" and name.startswith("HALLMARK_"):
        return name[len("HALLMARK_"):]
    return name


# ---------------------------------------------------------------------------
# flow_topo_landscape — matplotlib topographic contour map
# ---------------------------------------------------------------------------


def _get_feature_expression(X, mask, feat_idx):
    """Extract expression values for a feature, handling sparse matrices.

    Parameters
    ----------
    X : array-like or sparse matrix
        Expression matrix (adata.X or similar).
    mask : np.ndarray
        Boolean mask selecting cells.
    feat_idx : int
        Column index of the feature.

    Returns
    -------
    np.ndarray
        1D array of expression values for the masked cells.
    """
    col = X[mask, feat_idx]
    if hasattr(col, "toarray"):
        return np.asarray(col.toarray()).ravel()
    elif hasattr(col, "todense"):
        return np.asarray(col.todense()).ravel()
    return np.asarray(col).ravel()


def flow_topo_landscape(
    adata: AnnData,
    flow_result: dict,
    flow_model,
    *,
    features: list[str] | None = None,
    n_features: int = 5,
    n_timepoints: int = 20,
    n_eval_points: int = 300,
    n_grid: int = 80,
    feature_type: str = "genes",
    show_velocity: bool = True,
    show: bool = True,
    save: str | None = None,
) -> "matplotlib.figure.Figure":
    """Topographic contour map of feature expression and Jacobian expansion.

    Draws contour lines in PC1 x PC2 space showing expression levels at
    source and target cell distributions, with Jacobian-derived
    expansion/contraction bands through the flow field between them.
    Each feature gets a distinct color and line style, mimicking the
    layered look of a topographic map.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with PCA coordinates in
        ``adata.obsm["X_pca"]`` and PCA loadings in
        ``adata.varm["PCs"]``.
    flow_result : dict
        Result from ``pc.tl.flow_within()``. Must contain
        ``source_mask``, ``target_mask``, and ``pca_key``.
    flow_model : FlowModel
        Trained flow model with ``.transport()`` and ``.jacobian()``
        methods.
    features : list of str or None
        Feature names to plot. If None, auto-selects top ``n_features``
        from ``flow_gene_alignment`` scores.
    n_features : int
        Number of features to auto-select when ``features`` is None.
    n_timepoints : int
        Number of intermediate timepoints for trajectory and expansion
        computation.
    n_eval_points : int
        Number of source cells to subsample for trajectory computation.
    n_grid : int
        Resolution of the interpolation grid (n_grid x n_grid).
    feature_type : str
        ``"genes"`` to read from ``adata.X``, or ``"pathways"`` to
        read from ``adata.obsm["pathway_scores"]``.
    show_velocity : bool
        Whether to draw quiver arrows showing flow velocity. Defaults
        to ``True``. Set to ``False`` to suppress the arrows and show
        only the topographic contours and cell scatter.
    show : bool
        Whether to call ``plt.show()``.
    save : str or None
        If provided, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        The topographic landscape figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter

    # ------------------------------------------------------------------
    # 1. Masks and 2D positions
    # ------------------------------------------------------------------
    source_mask = flow_result["source_mask"]
    target_mask = flow_result.get("target_mask", ~source_mask)
    pca_key = flow_result.get("pca_key", "X_pca")

    pca = adata.obsm[pca_key]
    src_2d = pca[source_mask, :2]
    tgt_2d = pca[target_mask, :2]
    n_pcs = pca.shape[1]

    # ------------------------------------------------------------------
    # 2. Select features
    # ------------------------------------------------------------------
    if features is None:
        from peach.tl.flow import flow_gene_alignment

        align = flow_gene_alignment(adata, flow_result, n_permutations=0, per_cell=False)
        scores = align["alignment_scores"]
        names = align["gene_names"]
        top_idx = np.argsort(np.abs(scores))[-n_features:][::-1]
        features = [names[i] for i in top_idx]

    n_feat = len(features)
    logger.info("Computing flow topology for %d features across %d timepoints",
                n_feat, n_timepoints)

    # ------------------------------------------------------------------
    # 3. Resolve feature indices and expression source
    # ------------------------------------------------------------------
    if feature_type == "pathways":
        if "pathway_scores" not in adata.obsm:
            raise ValueError(
                "adata.obsm['pathway_scores'] not found. "
                "Run pc.pp.compute_pathway_scores() first."
            )
        pw_names = list(adata.uns.get("pathway_scores_pathways", []))
        feat_indices = []
        for f in features:
            if f not in pw_names:
                raise ValueError(
                    f"Pathway '{f}' not found in pathway_scores_pathways."
                )
            feat_indices.append(pw_names.index(f))
        expr_matrix = adata.obsm["pathway_scores"]
        is_pathway = True
    else:
        var_names = list(adata.var_names)
        feat_indices = []
        for f in features:
            if f not in var_names:
                raise ValueError(f"Feature '{f}' not found in adata.var_names")
            feat_indices.append(var_names.index(f))
        expr_matrix = adata.X
        is_pathway = False

    # PCA loadings for expansion computation
    loadings = adata.varm.get("PCs")  # [n_genes, n_pcs]

    # ------------------------------------------------------------------
    # 4. Compute trajectory
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    n_src = int(source_mask.sum())
    n_traj = min(n_eval_points, n_src)
    traj_idx = rng.choice(n_src, size=n_traj, replace=False)

    src_pca_full = pca[source_mask]
    traj_points = src_pca_full[traj_idx]

    trajectory = flow_model.transport(
        traj_points, n_steps=n_timepoints, return_trajectory=True
    )
    # trajectory: [n_steps+1, n_traj, dim]

    timepoints = np.linspace(0, 1, n_timepoints + 1)

    # ------------------------------------------------------------------
    # 5. Set up interpolation grid
    # ------------------------------------------------------------------
    all_2d = np.vstack([src_2d, tgt_2d])
    x_min, x_max = all_2d[:, 0].min(), all_2d[:, 0].max()
    y_min, y_max = all_2d[:, 1].min(), all_2d[:, 1].max()
    pad = 0.05 * max(x_max - x_min, y_max - y_min)
    xi = np.linspace(x_min - pad, x_max + pad, n_grid)
    yi = np.linspace(y_min - pad, y_max + pad, n_grid)
    Xi, Yi = np.meshgrid(xi, yi)

    # ------------------------------------------------------------------
    # 6. Build the figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Background: faint scatter of source and target cells
    ax.scatter(
        src_2d[:, 0], src_2d[:, 1],
        s=4, alpha=0.15, c="steelblue", rasterized=True,
    )
    ax.scatter(
        tgt_2d[:, 0], tgt_2d[:, 1],
        s=4, alpha=0.15, c="coral", rasterized=True,
    )

    # ------------------------------------------------------------------
    # 6a. Quiver arrows showing flow velocity (behind contours)
    # ------------------------------------------------------------------
    if show_velocity:
        # Compute velocity from consecutive trajectory timepoints
        # trajectory shape: [n_steps+1, n_traj, dim]
        n_frames = len(trajectory)
        quiver_x_all = []
        quiver_y_all = []
        quiver_u_all = []
        quiver_v_all = []

        for t_idx in range(n_frames - 1):
            pos_2d = trajectory[t_idx, :, :2]
            next_2d = trajectory[t_idx + 1, :, :2]
            vel_2d = next_2d - pos_2d

            quiver_x_all.append(pos_2d[:, 0])
            quiver_y_all.append(pos_2d[:, 1])
            quiver_u_all.append(vel_2d[:, 0])
            quiver_v_all.append(vel_2d[:, 1])

        quiver_x = np.concatenate(quiver_x_all)
        quiver_y = np.concatenate(quiver_y_all)
        quiver_u = np.concatenate(quiver_u_all)
        quiver_v = np.concatenate(quiver_v_all)

        # Subsample to avoid overcrowding: keep every Nth point
        n_quiver_total = len(quiver_x)
        max_arrows = 2000
        if n_quiver_total > max_arrows:
            step = max(1, n_quiver_total // max_arrows)
            q_idx = np.arange(0, n_quiver_total, step)
            quiver_x = quiver_x[q_idx]
            quiver_y = quiver_y[q_idx]
            quiver_u = quiver_u[q_idx]
            quiver_v = quiver_v[q_idx]

        ax.quiver(
            quiver_x, quiver_y, quiver_u, quiver_v,
            color="0.25", alpha=0.07, scale=None,
            headwidth=4, headlength=5, headaxislength=4,
            linewidth=0.3, zorder=1,
        )

    # Colors and line styles for features
    feat_colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_feat))
    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    # contour() only accepts string linestyles, so map tuples for contour use
    contour_ls = [ls if isinstance(ls, str) else "--" for ls in line_styles]

    for fi, (feat_name, feat_idx) in enumerate(zip(features, feat_indices)):
        color = feat_colors[fi]
        ls = line_styles[fi % len(line_styles)]
        cls = contour_ls[fi % len(contour_ls)]

        # --- Source expression contours ---
        if is_pathway:
            src_expr = np.asarray(expr_matrix[source_mask, feat_idx]).ravel()
        else:
            src_expr = _get_feature_expression(expr_matrix, source_mask, feat_idx)

        src_grid = griddata(
            src_2d, src_expr, (Xi, Yi), method="cubic", fill_value=0,
        )
        src_grid = gaussian_filter(src_grid, sigma=1.5)

        # Skip if grid is essentially flat
        src_range = src_grid.max() - src_grid.min()
        if src_range > 1e-10:
            lw_src = np.linspace(0.3, 1.5, 5)
            ax.contour(
                Xi, Yi, src_grid, levels=5,
                colors=[color], linestyles=cls,
                linewidths=lw_src, alpha=0.7,
                zorder=2,
            )

        # --- Target expression contours ---
        if is_pathway:
            tgt_expr = np.asarray(expr_matrix[target_mask, feat_idx]).ravel()
        else:
            tgt_expr = _get_feature_expression(expr_matrix, target_mask, feat_idx)

        tgt_grid = griddata(
            tgt_2d, tgt_expr, (Xi, Yi), method="cubic", fill_value=0,
        )
        tgt_grid = gaussian_filter(tgt_grid, sigma=1.5)

        tgt_range = tgt_grid.max() - tgt_grid.min()
        if tgt_range > 1e-10:
            lw_tgt = np.linspace(0.3, 1.5, 5)
            ax.contour(
                Xi, Yi, tgt_grid, levels=5,
                colors=[color], linestyles=cls,
                linewidths=lw_tgt, alpha=0.7,
                zorder=2,
            )

        # --- Flow field: expansion contours at intermediate timepoints ---
        if loadings is not None:
            # For pathways, use mean loading across pathway genes
            # (single-gene loading vector for genes)
            if is_pathway:
                # Approximate: use first n_pcs components of the pathway
                # score column projected through PCA loadings. Since
                # pathway scores live in obsm (not gene space), we skip
                # the Jacobian expansion overlay for pathways.
                loading_vec = None
            else:
                loading_vec = loadings[feat_idx, :n_pcs]

            if loading_vec is not None:
                flow_positions = []
                flow_expansion = []

                for t_idx in range(1, n_timepoints):
                    t_val = float(timepoints[t_idx])
                    traj_2d = trajectory[t_idx, :, :2]  # [n_traj, 2]

                    # Compute Jacobian at this timepoint
                    n_jac = min(100, len(traj_points))
                    jac_sub = trajectory[t_idx, :n_jac, :]  # full dim
                    jacs = flow_model.jacobian(jac_sub, t_val)  # [n_jac, dim, dim]
                    mean_jac = jacs.mean(axis=0)  # [dim, dim]

                    # Per-feature expansion: l^T J l
                    exp_val = loading_vec @ mean_jac @ loading_vec  # scalar

                    flow_positions.append(traj_2d)
                    flow_expansion.append(np.full(len(traj_2d), exp_val))

                if flow_positions:
                    all_flow_pos = np.vstack(flow_positions)
                    all_flow_exp = np.concatenate(flow_expansion)

                    flow_grid = griddata(
                        all_flow_pos, all_flow_exp, (Xi, Yi),
                        method="cubic", fill_value=0,
                    )
                    flow_grid = gaussian_filter(flow_grid, sigma=2.0)

                    # Expansion contours (positive = expansion, negative = contraction)
                    flow_max = flow_grid.max()
                    flow_min = flow_grid.min()

                    if flow_max > 0.001:
                        pos_levels = np.linspace(0.001, flow_max, 4)
                        ax.contour(
                            Xi, Yi, flow_grid, levels=pos_levels,
                            colors=[color], linestyles="-",
                            linewidths=0.8, alpha=0.5,
                            zorder=3,
                        )
                    if flow_min < -0.001:
                        neg_levels = np.linspace(flow_min, -0.001, 4)
                        ax.contour(
                            Xi, Yi, flow_grid, levels=neg_levels,
                            colors=[color], linestyles=":",
                            linewidths=0.6, alpha=0.4,
                            zorder=3,
                        )

        logger.info("  Feature %d/%d: %s", fi + 1, n_feat, feat_name)

    # ------------------------------------------------------------------
    # 7. Labels and legend
    # ------------------------------------------------------------------
    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_title("Flow Feature Topology", fontsize=14, fontweight="bold")

    legend_elements = []
    for fi, feat_name in enumerate(features):
        color = feat_colors[fi]
        ls = line_styles[fi % len(line_styles)]
        legend_elements.append(
            Line2D(
                [0], [0], color=color, linestyle=ls, linewidth=1.5,
                label=feat_name,
            )
        )
    # Source / target markers
    legend_elements.append(
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor="steelblue", markersize=5,
            alpha=0.5, label="Source cells",
        )
    )
    legend_elements.append(
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor="coral", markersize=5,
            alpha=0.5, label="Target cells",
        )
    )
    # Flow arrow indicator
    legend_elements.append(
        Line2D(
            [0], [0], marker=r"$\rightarrow$", color="w",
            markerfacecolor="0.25", markersize=8,
            alpha=0.5, label="Flow velocity",
        )
    )

    ax.legend(
        handles=legend_elements, loc="upper right", fontsize=9,
        framealpha=0.9, edgecolor="#ccc",
    )

    ax.set_facecolor("white")
    fig.tight_layout()

    # ------------------------------------------------------------------
    # 8. Save and show
    # ------------------------------------------------------------------
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# soft_assignment_flow
# ---------------------------------------------------------------------------

def soft_assignment_heatmap(
    adata: AnnData,
    flow_result: dict,
    *,
    adata_b: AnnData | None = None,
    pca_key: str = "X_pca",
    n_neighbors: int = 10,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Heatmap of soft archetype assignment correspondence between source and target.

    Uses cKDTree to find nearest neighbors in PCA space between transported
    source cells and target cells, then builds a K_source x K_target
    correspondence matrix from archetype weights.

    Parameters
    ----------
    adata : AnnData
        Source annotated data matrix with archetype weights in
        ``adata.obsm['cell_archetype_weights']``.
    flow_result : dict
        Output of ``pc.tl.flow_within()`` or ``pc.tl.flow_between()``.
    adata_b : AnnData or None
        Separate target AnnData. If None, target cells come from ``adata``
        using ``flow_result['target_mask']``.
    pca_key : str
        Key in ``adata.obsm`` for PCA coordinates.
    n_neighbors : int
        Number of nearest neighbors for soft matching.
    save_path : str or None
        If provided, save figure to this path.
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    from scipy.spatial import cKDTree

    transported = flow_result["transported"]

    if adata_b is not None:
        target_pca = adata_b.obsm[pca_key]
        weights_target = np.asarray(adata_b.obsm["cell_archetype_weights"])
        K_b = weights_target.shape[1]
    else:
        target_pca = adata.obsm[pca_key][flow_result["target_mask"]]
        weights_all = np.asarray(adata.obsm["cell_archetype_weights"])
        weights_target = weights_all[flow_result["target_mask"]]
        K_b = weights_target.shape[1]

    weights_source_all = np.asarray(adata.obsm["cell_archetype_weights"])
    weights_source_sub = weights_source_all[flow_result["source_mask"]]
    K_a = weights_source_sub.shape[1]

    # Find nearest target neighbors for each transported cell
    tree = cKDTree(target_pca)
    _, nn_idx = tree.query(transported, k=n_neighbors)

    # Build correspondence matrix: K_a x K_b
    correspondence = np.zeros((K_a, K_b))
    for i in range(len(transported)):
        w_source = weights_source_sub[i]  # [K_a]
        w_target_nn = weights_target[nn_idx[i]].mean(axis=0)  # [K_b]
        correspondence += np.outer(w_source, w_target_nn)
    correspondence /= len(transported)

    # Normalize rows
    row_sums = correspondence.sum(axis=1, keepdims=True)
    correspondence = correspondence / np.where(row_sums > 0, row_sums, 1)

    arch_a = [f"A{i+1} (source)" for i in range(K_a)]
    arch_b = [f"A{j+1} (target)" for j in range(K_b)]

    fig = go.Figure(data=go.Heatmap(
        z=correspondence,
        x=arch_b,
        y=arch_a,
        colorscale="Blues",
        text=np.round(correspondence, 2).astype(str),
        texttemplate="%{text}",
        textfont_size=10,
    ))

    apply_style(fig, title="Soft archetype assignment correspondence",
                xaxis_title="Target archetypes", yaxis_title="Source archetypes")

    return save_and_show(fig, save_path=save_path, show=show)


def soft_assignment_flow(
    adata: AnnData,
    *,
    top_n: int = 15,
    feature_type: str = "genes",
    pairs: list[tuple[int, int]] | None = None,
    alpha: float = 0.05,
    degree: int = 1,
    show: bool = True,
    save: str | None = None,
) -> go.Figure:
    """Sankey diagram of feature flow between archetype pairs.

    For each archetype pair, shows which features are exclusive to one
    archetype (directed flow) versus shared between both (bidirectional).
    Link width encodes coefficient magnitude; color encodes which archetype
    dominates.

    Left nodes represent **source archetypes** (dominant coefficient),
    right nodes represent **target archetypes** for feature correspondence.

    Parameters
    ----------
    adata : AnnData
        Must contain simplex regression results in ``adata.uns``
        (run ``pc.tl.feature_simplex_regression()`` first).
    top_n : int
        Number of top features per pair, selected by |delta| for exclusive
        features and by min(coef_a, coef_b) for shared features.
    feature_type : str
        ``"genes"`` or ``"pathways"`` — controls which regression result
        to load and how feature names are cleaned.
    pairs : list of (int, int) or None
        Archetype pairs to include. If None, all K*(K-1)/2 pairs are used.
    alpha : float
        FDR significance threshold. Only features with FDR < alpha are
        included. Uses ``f_pvalue_fdr`` (degree 1) or
        ``interaction_pvalues_fdr`` (degree 2) from the regression result.
    degree : int
        Which regression degree to display. ``1`` = linear (vertex)
        exclusive features, ``2`` = interaction/second-degree features.
        Degree 2 requires that the regression was run with
        ``max_degree >= 2``.
    show : bool
        Whether to call ``fig.show()``.
    save : str or None
        If provided, save figure to this path (format inferred from extension).

    Returns
    -------
    go.Figure
        Plotly Sankey figure.
    """
    # ------------------------------------------------------------------
    # 1. Resolve regression data
    # ------------------------------------------------------------------
    reg = resolve_regression_result(adata, feature_type=feature_type)
    if reg is None:
        raise ValueError(
            "No regression results found. "
            "Run pc.tl.feature_simplex_regression() first."
        )

    coefs = np.asarray(reg["vertex_coefficients"])  # [n_features, K]
    feature_names = list(reg["feature_names"])
    K = coefs.shape[1]

    # ------------------------------------------------------------------
    # 1b. FDR filtering — build boolean mask of significant features
    # ------------------------------------------------------------------
    if degree == 1:
        # Use per-feature F-test FDR (degree 1 model significance)
        fdr_array = reg.get("f_pvalue_fdr")
        if fdr_array is None:
            raise ValueError(
                "Regression result missing 'f_pvalue_fdr'. "
                "Re-run pc.tl.feature_simplex_regression()."
            )
        fdr_array = np.asarray(fdr_array)
        sig_mask = fdr_array < alpha  # [n_features]
    elif degree == 2:
        int_fdr = reg.get("interaction_pvalues_fdr")
        if int_fdr is None:
            raise ValueError(
                "Regression result has no degree-2 interaction data. "
                "Re-run pc.tl.feature_simplex_regression(max_degree=2)."
            )
        int_fdr = np.asarray(int_fdr)
        # Feature is significant at degree 2 if ANY interaction term passes FDR
        sig_mask = np.any(int_fdr < alpha, axis=1)  # [n_features]
        # For degree 2, use interaction coefficients for magnitude
        int_coefs = np.asarray(reg["interaction_coefficients"])  # [n_features, n_pairs]
        int_pairs = reg.get("interaction_pairs", [])
    else:
        raise ValueError(f"degree must be 1 or 2, got {degree}")

    n_sig = int(sig_mask.sum())
    if n_sig == 0:
        raise ValueError(
            f"No features pass FDR < {alpha} at degree {degree}. "
            "Try increasing alpha or checking regression results."
        )

    # Optional Wald contrasts for FDR annotation on hover
    contrasts = adata.uns.get("peach_archetype_contrasts")

    # ------------------------------------------------------------------
    # 2. Determine pairs
    # ------------------------------------------------------------------
    if pairs is None:
        pairs = list(combinations(range(K), 2))

    # Archetype colors — cycle palette if K > len(palette)
    palette = CATEGORICAL_PALETTE
    arch_colors = [palette[i % len(palette)] for i in range(K)]
    arch_rgb = [_hex_to_rgb(c) for c in arch_colors]

    # ------------------------------------------------------------------
    # 3. Build Sankey node and link arrays
    # ------------------------------------------------------------------
    # Nodes: left column = source archetypes (0..K-1)
    #        right column = target archetypes (K..2K-1)
    node_labels = (
        [f"A{i+1} (source)" for i in range(K)]
        + [f"A{i+1} (target)" for i in range(K)]
    )
    node_colors = arch_colors + arch_colors
    # Position left vs right via x/y
    node_x = [0.01] * K + [0.99] * K
    # Spread archetypes vertically
    node_y = [
        (i + 0.5) / K for i in range(K)
    ] + [
        (i + 0.5) / K for i in range(K)
    ]

    sources = []
    targets = []
    values = []
    link_colors = []
    link_labels = []

    for a, b in pairs:
        if degree == 1:
            # ----- Degree 1: vertex coefficient differences -----
            coef_a = coefs[:, a]  # [n_features]
            coef_b = coefs[:, b]  # [n_features]
            delta = coef_a - coef_b  # positive => enriched at a

            # Shared threshold: both coefficients above median of all positive coefs
            pos_coefs = coefs[coefs > 0]
            shared_thresh = np.median(pos_coefs) if len(pos_coefs) > 0 else 0.0
            shared_mask = (coef_a > shared_thresh) & (coef_b > shared_thresh)
            shared_strength = np.minimum(coef_a, coef_b)

            # Apply FDR mask
            exclusive_delta = delta.copy()
            exclusive_delta[shared_mask] = 0.0
            exclusive_delta[~sig_mask] = 0.0  # zero out non-significant
            abs_delta = np.abs(exclusive_delta)

            n_exclusive = max(1, top_n - top_n // 3)  # ~2/3 exclusive
            n_shared = top_n - n_exclusive             # ~1/3 shared

            excl_idx = np.argsort(abs_delta)[-n_exclusive:][::-1]
            excl_idx = excl_idx[abs_delta[excl_idx] > 1e-8]

            # Shared features: must also pass FDR
            shared_scores = shared_strength.copy()
            shared_scores[~shared_mask] = -np.inf
            shared_scores[~sig_mask] = -np.inf  # FDR filter
            shr_idx = np.argsort(shared_scores)[-n_shared:][::-1]
            shr_idx = shr_idx[shared_mask[shr_idx] & sig_mask[shr_idx]]

            # Try to get Wald FDR values for hover
            fdr_vals = None
            if contrasts is not None:
                pair_key = str((a, b))
                if "pvalues_fdr" in contrasts and pair_key in contrasts["pvalues_fdr"]:
                    fdr_vals = np.asarray(contrasts["pvalues_fdr"][pair_key])

            # --- Exclusive features: directed link from dominant archetype ---
            for idx in excl_idx:
                name = _clean_feature_name(feature_names[idx], feature_type)
                dominant = a if delta[idx] > 0 else b
                subordinate = b if delta[idx] > 0 else a
                magnitude = abs(delta[idx])

                sources.append(dominant)
                targets.append(subordinate + K)
                values.append(float(magnitude))

                link_colors.append(
                    _blend_rgb(arch_rgb[dominant], arch_rgb[subordinate],
                               weight_a=0.85, opacity=0.40)
                )

                hover = f"{name}  |  A{dominant+1} \u2192 A{subordinate+1}"
                hover += f"\n\u03b2(A{a+1})={coef_a[idx]:.3f}  \u03b2(A{b+1})={coef_b[idx]:.3f}"
                hover += f"\nFDR={fdr_array[idx]:.2e}"
                if fdr_vals is not None:
                    hover += f"  Wald FDR={fdr_vals[idx]:.2e}"
                link_labels.append(hover)

            # --- Shared features: two reciprocal links (a->b and b->a) ---
            for idx in shr_idx:
                name = _clean_feature_name(feature_names[idx], feature_type)
                shared_mag = float(shared_strength[idx])
                blended = _blend_rgb(arch_rgb[a], arch_rgb[b],
                                     weight_a=0.5, opacity=0.25)

                hover_base = f"{name} (shared)"
                hover_base += f"\n\u03b2(A{a+1})={coef_a[idx]:.3f}  \u03b2(A{b+1})={coef_b[idx]:.3f}"
                hover_base += f"\nFDR={fdr_array[idx]:.2e}"
                if fdr_vals is not None:
                    hover_base += f"  Wald FDR={fdr_vals[idx]:.2e}"

                # a -> b direction
                sources.append(a)
                targets.append(b + K)
                values.append(shared_mag)
                link_colors.append(blended)
                link_labels.append(hover_base)

                # b -> a direction
                sources.append(b)
                targets.append(a + K)
                values.append(shared_mag)
                link_colors.append(blended)
                link_labels.append(hover_base)

        else:
            # ----- Degree 2: interaction coefficients -----
            # Find the interaction column index for pair (a, b)
            pair_col = None
            for col_i, (pa, pb) in enumerate(int_pairs):
                if (pa == a and pb == b) or (pa == b and pb == a):
                    pair_col = col_i
                    break
            if pair_col is None:
                continue  # no interaction term for this pair

            int_col_coefs = int_coefs[:, pair_col]  # [n_features]
            int_col_fdr = np.asarray(reg["interaction_pvalues_fdr"])[:, pair_col]

            # Filter: significant interaction AND overall significance
            pair_sig = (int_col_fdr < alpha) & sig_mask
            if not np.any(pair_sig):
                continue

            abs_int = np.abs(int_col_coefs)
            abs_int[~pair_sig] = 0.0

            top_idx = np.argsort(abs_int)[-top_n:][::-1]
            top_idx = top_idx[abs_int[top_idx] > 1e-8]

            for idx in top_idx:
                name = _clean_feature_name(feature_names[idx], feature_type)
                int_val = int_col_coefs[idx]
                magnitude = abs(int_val)

                # Positive interaction: synergy (a->b), negative: antagonism (b->a)
                if int_val > 0:
                    src, tgt = a, b
                else:
                    src, tgt = b, a

                sources.append(src)
                targets.append(tgt + K)
                values.append(float(magnitude))

                link_colors.append(
                    _blend_rgb(arch_rgb[src], arch_rgb[tgt],
                               weight_a=0.7, opacity=0.45)
                )

                hover = f"{name}  |  A{src+1} \u2194 A{tgt+1} (interaction)"
                hover += f"\n\u03b3={int_val:.3f}  FDR={int_col_fdr[idx]:.2e}"
                link_labels.append(hover)

    # ------------------------------------------------------------------
    # 4. Guard against empty diagrams
    # ------------------------------------------------------------------
    if not sources:
        raise ValueError(
            f"No features passed the selection threshold (FDR < {alpha}, "
            f"degree {degree}). Try increasing alpha or top_n, or check "
            "that regression coefficients are non-trivial."
        )

    # ------------------------------------------------------------------
    # 5. Build figure
    # ------------------------------------------------------------------
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=node_labels,
            color=node_colors,
            x=node_x,
            y=node_y,
            pad=15,
            thickness=20,
            line=dict(color="#aaa", width=0.5),
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            label=link_labels,
        ),
    ))

    degree_label = "Linear" if degree == 1 else "Interaction"
    title = f"Feature Correspondence (Degree {degree}: {degree_label}, FDR < {alpha})"
    if feature_type != "genes":
        title += f" [{feature_type}]"

    # Add annotations labeling left and right columns
    fig.update_layout(
        annotations=[
            dict(
                x=0.01, y=1.08, xref="paper", yref="paper",
                text="Source Archetypes", showarrow=False,
                font=dict(size=13, color="#444"),
                xanchor="left",
            ),
            dict(
                x=0.99, y=1.08, xref="paper", yref="paper",
                text="Target Archetypes", showarrow=False,
                font=dict(size=13, color="#444"),
                xanchor="right",
            ),
        ]
    )

    apply_style(fig, title=title, height=max(450, K * 80), width=950)
    fig.update_layout(
        legend=dict(x=1.02, y=1, xanchor="left"),
    )

    return save_and_show(fig, save_path=save, show=show)

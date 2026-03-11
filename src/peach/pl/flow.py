"""Flow matching visualization."""

import numpy as np
import plotly.graph_objects as go
from anndata import AnnData

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
            arrowcolor=COLOR_NEGATIVE,
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

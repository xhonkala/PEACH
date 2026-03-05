"""Flow matching visualization."""

import numpy as np
import plotly.graph_objects as go
from anndata import AnnData


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
        If provided, save figure as HTML to this path.
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    pca = adata.obsm[pca_key]
    source_pca = pca[flow_result.source_mask]
    transported = flow_result.transported

    # Subsample
    n = min(n_arrows, len(source_pca))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(source_pca), size=n, replace=False)

    x = source_pca[idx, 0]
    y = source_pca[idx, 1]
    dx = transported[idx, 0] - x
    dy = transported[idx, 1] - y

    # Scale arrows
    scale = 0.3

    fig = go.Figure()

    # Background: all cells
    fig.add_trace(go.Scatter(
        x=pca[:, 0], y=pca[:, 1],
        mode="markers",
        marker={"size": 2, "opacity": 0.2, "color": "gray"},
        name="All cells",
        showlegend=True,
    ))

    # Arrows via annotations
    for i in range(n):
        fig.add_annotation(
            x=x[i] + dx[i] * scale,
            y=y[i] + dy[i] * scale,
            ax=x[i],
            ay=y[i],
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="red",
        )

    fig.update_layout(
        title="Flow Velocity Field",
        xaxis_title="PC1",
        yaxis_title="PC2",
        showlegend=True,
    )

    if save_path:
        fig.write_html(save_path)
    if show:
        fig.show()
    return fig


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
        If provided, save figure as HTML to this path.
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    scores = alignment_result.alignment_scores
    names = alignment_result.gene_names

    sorted_idx = np.argsort(scores)
    top_aligned = sorted_idx[-n_top:][::-1]
    top_opposed = sorted_idx[:n_top]

    combined_idx = np.concatenate([top_opposed, top_aligned])
    combined_names = [names[i] for i in combined_idx]
    combined_scores = scores[combined_idx]

    colors = ["red" if s < 0 else "blue" for s in combined_scores]

    fig = go.Figure(data=go.Bar(
        x=combined_scores,
        y=combined_names,
        orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        title=f"Gene Alignment with Flow (top {n_top} each direction)",
        xaxis_title="Alignment Score",
        height=max(400, len(combined_names) * 15),
    )

    if save_path:
        fig.write_html(save_path)
    if show:
        fig.show()
    return fig


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
        If provided, save figure as HTML to this path.
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    jac = jacobian_result.mean_jacobian
    dim = jac.shape[0]
    labels = [f"PC{i+1}" for i in range(dim)]

    fig = go.Figure(data=go.Heatmap(
        z=jac,
        x=labels,
        y=labels,
        colorscale="RdBu_r",
        zmid=0,
        colorbar={"title": "dv/dx"},
    ))
    fig.update_layout(
        title=f"Mean Flow Jacobian (t={jacobian_result.t})",
        xaxis_title="Input PC",
        yaxis_title="Output PC",
    )

    if save_path:
        fig.write_html(save_path)
    if show:
        fig.show()
    return fig


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
        If provided, save figure as HTML to this path.
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    pca = adata.obsm[flow_result.pca_key]
    source_pca = pca[flow_result.source_mask]

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
        transported = flow_result.transported[idx]
        source = source_pca[idx]
        times = np.linspace(0, 1, n_steps + 1)
        traj = np.stack([(1 - t) * source + t * transported for t in times])

    fig = go.Figure()
    for step in range(len(traj)):
        t_val = step / max(len(traj) - 1, 1)
        fig.add_trace(go.Scatter(
            x=traj[step, :, 0],
            y=traj[step, :, 1],
            mode="markers",
            marker={
                "size": 3,
                "opacity": 0.4,
                "color": f"rgb({int(255 * t_val)}, 0, {int(255 * (1 - t_val))})",
            },
            name=f"t={t_val:.2f}",
            showlegend=step % max(1, len(traj) // 5) == 0,
        ))

    fig.update_layout(
        title="Flow Trajectory",
        xaxis_title="PC1",
        yaxis_title="PC2",
    )

    if save_path:
        fig.write_html(save_path)
    if show:
        fig.show()
    return fig


def flow_magnitude(
    adata: AnnData,
    flow_result,
    *,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """2D scatter colored by transport magnitude (||transported - source||).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with PCA coordinates.
    flow_result : FlowWithinResult
        Result from ``pc.tl.flow_within()``.
    save_path : str or None
        If provided, save figure as HTML to this path.
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    pca = adata.obsm[flow_result.pca_key]
    source_pca = pca[flow_result.source_mask]
    transported = flow_result.transported

    magnitude = np.linalg.norm(transported - source_pca, axis=1)

    fig = go.Figure(data=go.Scatter(
        x=source_pca[:, 0],
        y=source_pca[:, 1],
        mode="markers",
        marker={
            "size": 4,
            "color": magnitude,
            "colorscale": "Hot",
            "colorbar": {"title": "||transport||"},
        },
    ))
    fig.update_layout(
        title="Flow Magnitude",
        xaxis_title="PC1",
        yaxis_title="PC2",
    )

    if save_path:
        fig.write_html(save_path)
    if show:
        fig.show()
    return fig


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
        If provided, save figure as HTML to this path.
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    pca = adata.obsm[flow_result.pca_key]
    source_pc1 = pca[flow_result.source_mask, 0]
    target_pc1 = pca[flow_result.target_mask, 0]
    transported_pc1 = flow_result.transported[:, 0]

    fig = go.Figure()
    for data, name, color in [
        (source_pc1, "Source", "blue"),
        (transported_pc1, "Transported", "green"),
        (target_pc1, "Target", "red"),
    ]:
        fig.add_trace(go.Histogram(
            x=data,
            name=name,
            opacity=0.5,
            marker_color=color,
            nbinsx=50,
            histnorm="probability density",
        ))

    fig.update_layout(
        title="Density Comparison (PC1)",
        xaxis_title="PC1",
        yaxis_title="Density",
        barmode="overlay",
    )

    if save_path:
        fig.write_html(save_path)
    if show:
        fig.show()
    return fig


def archetype_correspondence(
    flow_between_result,
    *,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """K_src x K_tgt heatmap for archetype correspondence.

    Parameters
    ----------
    flow_between_result : FlowBetweenResult
        Result from ``pc.tl.flow_between()`` with archetype
        correspondence computed.
    save_path : str or None
        If provided, save figure as HTML to this path.
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    if flow_between_result.archetype_correspondence is None:
        raise ValueError("No archetype correspondence computed.")

    fig = go.Figure()
    for pair, corr_matrix in flow_between_result.archetype_correspondence.items():
        src, tgt = pair
        K_src, K_tgt = corr_matrix.shape
        fig.add_trace(go.Heatmap(
            z=corr_matrix,
            x=[f"Tgt {i}" for i in range(K_tgt)],
            y=[f"Src {i}" for i in range(K_src)],
            colorscale="Blues",
            name=f"{src} -> {tgt}",
        ))
        break  # Show first pair only

    fig.update_layout(
        title="Archetype Correspondence",
        xaxis_title="Target Archetypes",
        yaxis_title="Source Archetypes",
    )

    if save_path:
        fig.write_html(save_path)
    if show:
        fig.show()
    return fig

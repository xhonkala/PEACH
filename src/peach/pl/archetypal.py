"""
Archetypal visualization functions.

This module provides publication-ready interactive visualization tools for
archetypal analysis results. All plots are built with Plotly for interactivity
and can be exported to various formats for publication.

Main Functions:
- archetypal_space(): Interactive 3D visualization of archetypal coordinate space
- archetypal_space_multi(): Compare multiple archetypal fits side-by-side
- training_metrics(): Training diagnostics and convergence analysis
- elbow_curve(): Hyperparameter selection support with cross-validation

Features:
- Interactive Plotly-based plots with zoom, pan, and hover
- Gene expression coloring with smart layer selection
- Publication-ready aesthetics and customization options
- Automatic legend and colorbar positioning
"""

from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from anndata import AnnData

from .._core.viz.results_viz import visualize_archetypal_space_3d_multi as _viz_3d_multi

# Import existing battle-tested functions
from .._core.viz.results_viz import visualize_archetypal_space_3d_single as _viz_3d
from .._core.viz.training_viz import plot_training_metrics as _plot_training


def archetypal_space(
    adata: AnnData,
    *,
    archetype_coords_key: str = "archetype_coordinates",
    pca_key: str = "X_pca",
    color_by: str | None = None,
    use_layer: str = "logcounts",
    cell_size: float = 2.0,
    cell_opacity: float = 0.6,
    archetype_size: float = 8.0,
    archetype_color: str = "red",
    show_archetype_labels: bool = True,
    show_connections: bool = True,
    color_scale: str = "viridis",
    categorical_colors: dict | None = None,
    title: str = "Archetypal Space Visualization",
    auto_scale: bool = True,
    save_path: str | None = None,
    fixed_ranges: dict | None = None,
    legend_marker_scale: float = 1.0,
    legend_font_size: int = 12,
    # Conditional centroid parameters
    show_centroids: bool = False,
    centroid_condition: str | None = None,
    centroid_order: list | None = None,
    centroid_groupby: str | None = None,
    centroid_size: float = 20.0,
    centroid_start_symbol: str = "circle",
    centroid_end_symbol: str = "diamond",
    centroid_line_width: float = 6.0,
    centroid_colors: dict | None = None,
    **kwargs,
) -> go.Figure:
    """Visualize cells in 3D archetypal coordinate space.

    Creates an interactive 3D scatter plot showing cells positioned
    in PCA space with archetype positions and optional coloring by
    gene expression or metadata.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with archetypal coordinates.
    archetype_coords_key : str, default: "archetype_coordinates"
        Key in adata.uns containing archetype coordinates [n_archetypes, n_pcs].
    pca_key : str, default: "X_pca"
        Key in adata.obsm containing PCA coordinates [n_cells, n_pcs].
    color_by : str | None, default: None
        Column in adata.obs (categorical/continuous) or gene name in
        adata.var.index for expression coloring.
    use_layer : str, default: "logcounts"
        Layer for gene expression. Falls back to adata.X if not found.
    cell_size : float, default: 2.0
        Size of cell points.
    cell_opacity : float, default: 0.6
        Opacity of cell points (0-1).
    archetype_size : float, default: 8.0
        Size of archetype diamond markers.
    archetype_color : str, default: "red"
        Color for archetype markers.
    show_archetype_labels : bool, default: True
        Whether to show 'Arch1', 'Arch2', etc. labels.
    show_connections : bool, default: True
        Whether to draw lines connecting all archetype pairs.
    color_scale : str, default: "viridis"
        Plotly color scale for continuous variables.
    categorical_colors : dict | None, default: None
        Custom colors for categorical variables {category: color}.
    title : str, default: "Archetypal Space Visualization"
        Plot title.
    auto_scale : bool, default: True
        Whether to auto-scale axes using 1st-99th percentiles.
    save_path : str | None, default: None
        Path to save HTML file.
    fixed_ranges : dict | None, default: None
        Fixed axis ranges {'x': (min, max), 'y': (min, max), 'z': (min, max)}.
    legend_marker_scale : float, default: 1.0
        Scale factor for legend marker sizes.
    legend_font_size : int, default: 12
        Font size for legend text.
    show_centroids : bool, default: False
        Whether to display condition centroids on the plot.
        Requires centroids computed via pc.tl.compute_conditional_centroids().
    centroid_condition : str | None, default: None
        Column name in adata.obs for condition centroids.
        Must have centroids pre-computed via pc.tl.compute_conditional_centroids().
    centroid_order : list | None, default: None
        Order of condition levels for trajectory line.
        If provided, draws a line connecting centroids in this order.
        Example: ['chemo-naive', 'IDS'] for treatment timeline.
    centroid_groupby : str | None, default: None
        Column name for multi-group trajectories.
        If provided, draws separate trajectory per group with different colors.
    centroid_size : float, default: 20.0
        Size of centroid markers.
    centroid_start_symbol : str, default: "circle"
        Plotly symbol for first centroid in trajectory.
    centroid_end_symbol : str, default: "diamond"
        Plotly symbol for last centroid in trajectory.
    centroid_line_width : float, default: 6.0
        Width of trajectory line connecting centroids.
    centroid_colors : dict | None, default: None
        Custom colors for centroid markers/lines.
        If centroid_groupby used: {group: color} (e.g., {'long': 'magenta', 'short': 'cyan'}).
        Otherwise: {'default': color}.
    **kwargs
        Additional arguments passed to underlying visualization.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 3D scatter plot containing:
        - Cell points colored by color_by (with colorbar if continuous)
        - Archetype positions as diamond markers
        - Archetype labels (if show_archetype_labels=True)
        - Hull edges connecting archetypes (if show_connections=True)
        - Condition centroids with trajectory lines (if show_centroids=True)

    Raises
    ------
    ValueError
        If adata.obsm['archetype_distances'] not found (run
        pc.tl.archetypal_coordinates() first).

    Examples
    --------
    >>> # Color by cell type metadata
    >>> fig = pc.pl.archetypal_space(adata, color_by="cell_type")
    >>> fig.show()

    >>> # Color by gene expression
    >>> fig = pc.pl.archetypal_space(adata, color_by="CD3D")
    >>> fig.show()

    >>> # Custom styling
    >>> fig = pc.pl.archetypal_space(
    ...     adata, color_by="pseudotime", color_scale="plasma", cell_opacity=0.4, archetype_size=12.0
    ... )

    >>> # With condition trajectory centroids
    >>> pc.tl.compute_conditional_centroids(adata, "treatment_phase")
    >>> fig = pc.pl.archetypal_space(
    ...     adata,
    ...     show_centroids=True,
    ...     centroid_condition="treatment_phase",
    ...     centroid_order=["chemo-naive", "IDS"],
    ...     centroid_colors={"default": "magenta"},
    ... )

    >>> # Multi-group trajectories (treatment × response)
    >>> pc.tl.compute_conditional_centroids(adata, "treatment_phase", groupby="response")
    >>> fig = pc.pl.archetypal_space(
    ...     adata,
    ...     show_centroids=True,
    ...     centroid_condition="treatment_phase",
    ...     centroid_groupby="response",
    ...     centroid_order=["chemo-naive", "IDS"],
    ...     centroid_colors={"long": "magenta", "short": "cyan"},
    ... )
    """
    # Input validation
    if "archetype_distances" not in adata.obsm:
        raise ValueError("Archetypal distances not found. Run pc.tl.archetypal_coordinates() first.")

    # Delegate to existing visualization function
    fig = _viz_3d(
        adata=adata,
        archetype_coords_key=archetype_coords_key,
        pca_key=pca_key,
        color_by=color_by,
        use_layer=use_layer,
        cell_size=cell_size,
        cell_opacity=cell_opacity,
        archetype_size=archetype_size,
        archetype_color=archetype_color,
        show_archetype_labels=show_archetype_labels,
        show_connections=show_connections,
        color_scale=color_scale,
        categorical_colors=categorical_colors,
        title=title,
        auto_scale=auto_scale,
        save_path=save_path,
        fixed_ranges=fixed_ranges,
        legend_marker_scale=legend_marker_scale,
        legend_font_size=legend_font_size,
        # Conditional centroid parameters
        show_centroids=show_centroids,
        centroid_condition=centroid_condition,
        centroid_order=centroid_order,
        centroid_groupby=centroid_groupby,
        centroid_size=centroid_size,
        centroid_start_symbol=centroid_start_symbol,
        centroid_end_symbol=centroid_end_symbol,
        centroid_line_width=centroid_line_width,
        centroid_colors=centroid_colors,
        **kwargs,
    )

    return fig


def archetypal_space_multi(
    adata_list: list[AnnData],
    *,
    archetype_coords_key: str = "archetype_coordinates",
    pca_key: str = "X_pca",
    labels_list: list[str] | None = None,
    color_by: str | list[str] | None = None,
    color_values: Any | list[Any] | None = None,
    cell_size: float = 2.0,
    cell_opacity: float = 0.6,
    archetype_size: float = 8.0,
    archetype_colors: list[str] | None = None,
    show_labels: bool | list[int] = True,
    auto_scale: bool = True,
    range_reference: int | Any | None = None,
    fixed_ranges: dict[str, tuple[float, float]] | None = None,
    color_scale: str = "viridis",
    categorical_colors: dict[str, str] | None = None,
    title: str = "Multi-Archetypal Space Comparison",
    save_path: str | Path | None = None,
) -> go.Figure:
    """Compare multiple archetypal analysis fits in 3D PCA space.

    Creates an interactive 3D scatter plot comparing multiple archetypal fits,
    useful for comparing different conditions, treatments, or parameter settings.

    Parameters
    ----------
    adata_list : list of AnnData
        List of AnnData objects with PCA coordinates and archetype results
    archetype_coords_key : str, default: "archetype_coordinates"
        Key in adata.uns containing archetype coordinates
    pca_key : str, default: "X_pca"
        Key in adata.obsm containing PCA coordinates
    labels_list : list of str | None, default: None
        Labels for each dataset (defaults to 'Set 1', 'Set 2', etc.)
    color_by : str | list of str | None, default: None
        Column(s) to color cells by - single string or list per dataset
    color_values : array | list of arrays | None, default: None
        Direct color values - single array or list per dataset
    cell_size : float, default: 2.0
        Size of cell points
    cell_opacity : float, default: 0.6
        Opacity of cell points (0-1)
    archetype_size : float, default: 8.0
        Size of archetype markers
    archetype_colors : list of str | None, default: None
        Colors for archetype markers per dataset
    show_labels : bool | list of int, default: True
        Which datasets to show archetype labels for (bool, list of indices)
    auto_scale : bool, default: True
        Whether to auto-scale axes based on all data
    range_reference : int | AnnData | None, default: None
        Reference dataset index or AnnData for axis scaling
    fixed_ranges : dict | None, default: None
        Fixed axis ranges {'x': (min, max), 'y': (min, max), 'z': (min, max)}
    color_scale : str, default: 'viridis'
        Plotly color scale for continuous variables
    categorical_colors : dict | None, default: None
        Custom colors for categorical variables
    title : str, default: 'Multi-Archetypal Space Comparison'
        Plot title
    save_path : str | Path | None, default: None
        Optional path to save HTML file

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 3D comparison plot

    Examples
    --------
    >>> # Compare treatment conditions
    >>> fig = pc.pl.archetypal_space_multi(
    ...     adata_list=[adata_control, adata_treated],
    ...     labels_list=["Control", "Treated"],
    ...     color_by=["cell_type", "cell_type"],
    ...     title="Treatment Effect on Archetypal Space",
    ... )
    >>> fig.show()

    >>> # Compare different archetype numbers
    >>> fig = pc.pl.archetypal_space_multi(
    ...     adata_list=[adata_k3, adata_k5, adata_k7],
    ...     labels_list=["K=3", "K=5", "K=7"],
    ...     show_labels=[2],  # Only show labels for K=7
    ...     title="Archetype Number Comparison",
    ... )
    >>> fig.show()
    """
    # Delegate to existing visualization function
    fig = _viz_3d_multi(
        adata_list=adata_list,
        archetype_coords_key=archetype_coords_key,
        pca_key=pca_key,
        labels_list=labels_list,
        color_by=color_by,
        color_values=color_values,
        cell_size=cell_size,
        cell_opacity=cell_opacity,
        archetype_size=archetype_size,
        archetype_colors=archetype_colors,
        show_labels=show_labels,
        auto_scale=auto_scale,
        range_reference=range_reference,
        fixed_ranges=fixed_ranges,
        color_scale=color_scale,
        categorical_colors=categorical_colors,
        title=title,
        save_path=save_path,
    )

    return fig


def training_metrics(
    history: dict, *, height: int = 400, width: int = 800, display: bool = True, **kwargs
) -> go.Figure:
    """Visualize training metrics over epochs.

    Creates interactive Plotly visualization with loss components,
    stability metrics, and convergence analysis.

    Parameters
    ----------
    history : dict
        Training history dictionary from pc.tl.train_archetypal().
        Expected keys: 'loss', 'archetypal_loss', 'KLD', 'rmse',
        'vertex_stability_latent', 'vertex_stability_pca', 'loss_delta'.
    height : int, default: 400
        Base plot height in pixels (actual height is 2x for 3 rows).
    width : int, default: 800
        Plot width in pixels.
    display : bool, default: True
        Whether to display the plot immediately via fig.show().
    **kwargs
        Additional arguments passed to plot_training_metrics.

    Returns
    -------
    plotly.graph_objects.Figure or None
        Interactive training metrics plot with 3-row layout:
        - Row 1 (40%): Loss metrics (loss, archetypal_loss, KLD, rmse)
        - Row 2 (30%): Stability metrics (vertex_stability_latent/pca)
        - Row 3 (30%): Convergence (loss_delta with rolling mean)

        Returns None only if history is empty.

    Examples
    --------
    >>> results = pc.tl.train_archetypal(adata, n_archetypes=5)
    >>> fig = pc.pl.training_metrics(results["history"], display=False)
    >>> fig.write_html("training.html")
    """
    return _plot_training(history=history, height=height, width=width, display=display, **kwargs)


def elbow_curve(cv_summary, *, metrics: list[str] = ["archetype_r2", "rmse"], **kwargs) -> go.Figure:
    """Plot elbow curves for hyperparameter selection.

    Parameters
    ----------
    cv_summary : CVSummary
        Cross-validation results from pc.tl.hyperparameter_search()
    metrics : list[str], default: ["archetype_r2", "rmse"]
        Metrics to plot
    **kwargs
        Additional arguments passed to plot_elbow_curve

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive elbow curve plot
    """
    return cv_summary.plot_elbow_curve(metrics, **kwargs)


def archetype_positions(
    adata: AnnData,
    *,
    coords_key: str = "archetype_coordinates",
    title: str = "Archetype Positions in PCA Space",
    figsize: tuple = (15, 6),
    cmap: str = "tab10",
    show_distances: bool = True,
    save_path: str | None = None,
    **kwargs,
) -> Any:
    """Visualize archetype positions in PCA space with distance matrix.

    Creates a two-panel visualization showing archetype positions in the
    first two principal components and a pairwise distance matrix heatmap.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with archetype coordinates
    coords_key : str, default: "archetype_coordinates"
        Key in adata.uns containing archetype coordinates
    title : str, default: "Archetype Positions in PCA Space"
        Main figure title
    figsize : tuple, default: (15, 6)
        Figure size as (width, height)
    cmap : str, default: 'tab10'
        Colormap for archetype points
    show_distances : bool, default: True
        Whether to show distance matrix panel
    save_path : str | None, default: None
        Path to save the figure
    **kwargs
        Additional arguments passed to plot_archetype_positions

    Returns
    -------
    matplotlib.figure.Figure
        Figure with archetype position visualizations

    Examples
    --------
    >>> fig = pc.pl.archetype_positions(adata)
    >>> plt.show()

    >>> # Save high-resolution figure
    >>> fig = pc.pl.archetype_positions(adata, title="Helsinki EOC Archetype Positions", save_path="archetypes.png")

    Notes
    -----
    The visualization includes:
    - Left panel: Archetype positions in PC1-PC2 space with convex hull
    - Right panel: Pairwise distance matrix with values

    Requires at least 2 dimensions in archetype coordinates.
    For 3D visualization, use `archetype_positions_3d()`.
    """
    from .._core.viz.training_viz import plot_archetype_positions as _plot_positions

    # Get archetype coordinates from AnnData
    if coords_key not in adata.uns:
        raise ValueError(f"adata.uns['{coords_key}'] not found. Run pc.tl.train_archetypal() first.")

    coords = adata.uns[coords_key]

    return _plot_positions(
        archetype_coordinates=coords,
        title=title,
        figsize=figsize,
        cmap=cmap,
        show_distances=show_distances,
        save_path=save_path,
        **kwargs,
    )


def archetype_positions_3d(
    adata: AnnData,
    *,
    coords_key: str = "archetype_coordinates",
    title: str = "Archetype Positions in 3D PCA Space",
    figsize: tuple = (12, 10),
    cmap: str = "tab10",
    save_path: str | None = None,
    **kwargs,
) -> Any:
    """Visualize archetype positions in 3D PCA space.

    Creates an interactive 3D visualization of archetype positions with
    convex hull edges connecting the archetypes.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with archetype coordinates
    coords_key : str, default: "archetype_coordinates"
        Key in adata.uns containing archetype coordinates
    title : str, default: "Archetype Positions in 3D PCA Space"
        Figure title
    figsize : tuple, default: (12, 10)
        Figure size as (width, height)
    cmap : str, default: 'tab10'
        Colormap for archetype points
    save_path : str | None, default: None
        Path to save the figure
    **kwargs
        Additional arguments passed to plot_archetype_distances_3d

    Returns
    -------
    matplotlib.figure.Figure
        3D visualization of archetypes

    Examples
    --------
    >>> # Basic 3D visualization
    >>> fig = pc.pl.archetype_positions_3d(adata)
    >>> plt.show()

    >>> # Custom visualization
    >>> fig = pc.pl.archetype_positions_3d(adata, cmap="Set1", title="3D Archetype Hull")

    Notes
    -----
    Requires at least 3 dimensions in archetype coordinates.
    The visualization includes convex hull edges connecting archetypes.
    """
    from .._core.viz.training_viz import plot_archetype_distances_3d as _plot_3d

    # Get archetype coordinates from AnnData
    if coords_key not in adata.uns:
        raise ValueError(f"adata.uns['{coords_key}'] not found. Run pc.tl.train_archetypal() first.")

    coords = adata.uns[coords_key]

    return _plot_3d(
        archetype_coordinates=coords, title=title, figsize=figsize, cmap=cmap, save_path=save_path, **kwargs
    )


def archetype_statistics(adata: AnnData, *, coords_key: str = "archetype_coordinates", verbose: bool = True) -> dict:
    """Compute and display statistics about archetype positions.

    Calculates pairwise distances, identifies nearest/farthest archetype
    pairs, and computes convex hull metrics when possible.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with archetype coordinates.
    coords_key : str, default: "archetype_coordinates"
        Key in adata.uns containing archetype coordinates.
    verbose : bool, default: True
        Whether to print statistics to console.

    Returns
    -------
    dict
        Statistics dictionary with keys:
        - n_archetypes : int - Number of archetypes
        - n_dimensions : int - Embedding dimensions
        - mean_distance : float - Mean pairwise Euclidean distance
        - std_distance : float - Std of pairwise distances
        - min_distance : float - Minimum pairwise distance
        - max_distance : float - Maximum pairwise distance
        - distance_range : float - max - min distance
        - nearest_pair : tuple[int, int] - Indices of nearest pair (0-based)
        - farthest_pair : tuple[int, int] - Indices of farthest pair (0-based)
        - distance_matrix : np.ndarray - Full pairwise distance matrix
        - hull_volume : float | None - Convex hull volume (3D+ only)
        - hull_area : float | None - Convex hull surface area (3D+ only)

    Raises
    ------
    ValueError
        If adata.uns[coords_key] not found.

    Examples
    --------
    >>> stats = pc.pl.archetype_statistics(adata)
    [STATS] Archetype Statistics
    ==================================================
    Number of archetypes: 5
    ...

    >>> # Quiet mode
    >>> stats = pc.pl.archetype_statistics(adata, verbose=False)
    >>> print(f"Nearest pair: A{stats['nearest_pair'][0] + 1}-A{stats['nearest_pair'][1] + 1}")
    """
    from .._core.viz.training_viz import compute_archetype_statistics as _compute_stats

    # Get archetype coordinates from AnnData
    if coords_key not in adata.uns:
        raise ValueError(f"adata.uns['{coords_key}'] not found. Run pc.tl.train_archetypal() first.")

    coords = adata.uns[coords_key]

    return _compute_stats(archetype_coordinates=coords, verbose=verbose)

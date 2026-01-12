"""
Results visualization functions.

This module provides specialized plotting functions for statistical analysis
results from archetypal analysis. Creates publication-ready visualizations
with proper aesthetic formatting and interactive features.

Main Functions:
- dotplot(): Publication-ready dotplots for statistical test results

Features:
- Automatic detection of result types (genes, pathways, patterns)
- Smart column detection and fallback handling
- Publication-quality formatting and spacing
- Interactive hover information and zooming
- Support for effect sizes and significance levels
"""

import pandas as pd

# Import existing battle-tested functions
from .._core.viz.results_viz import create_dotplot_visualization as _create_dotplot


def dotplot(
    results_df: pd.DataFrame,
    *,
    x_col: str = "archetype",
    y_col: str = "gene",
    size_col: str = "mean_archetype",
    color_col: str = "pvalue",
    top_n_per_group: int = 10,
    filter_zero_p: bool = True,
    log_transform_p: bool = True,
    max_log_p: float = 300.0,
    title: str = "Gene-Archetype Associations",
    figsize: tuple[float, float] = (12, 8),
    color_palette: str = "plasma",
    save_path: str | None = None,
    **kwargs,
) -> "plt.Figure":
    """Create dotplot visualization for statistical results.

    Creates publication-ready dotplot showing statistical test results
    with effect sizes encoded as dot size and significance as color.

    Parameters
    ----------
    results_df : pd.DataFrame
        Statistical test results from pc.tl.gene_associations(),
        pc.tl.pathway_associations(), or pc.tl.pattern_analysis().
    x_col : str, default: "archetype"
        Column for x-axis (groups).
    y_col : str, default: "gene"
        Column for y-axis (features).
    size_col : str, default: "mean_archetype"
        Column for dot size (effect magnitude).
    color_col : str, default: "pvalue"
        Column for dot color (significance).
    top_n_per_group : int, default: 10
        Number of top results per group.
    filter_zero_p : bool, default: True
        Whether to filter out p-values of exactly 0.
    log_transform_p : bool, default: True
        Whether to apply -log10 transformation to p-values.
    max_log_p : float, default: 300.0
        Maximum -log10(p-value) cap.
    title : str, default: "Gene-Archetype Associations"
        Plot title.
    figsize : tuple[float, float], default: (12, 8)
        Figure size as (width, height).
    color_palette : str, default: "plasma"
        Matplotlib colormap name.
    save_path : str | None, default: None
        Path to save the figure.
    **kwargs
        Additional arguments passed to create_dotplot_visualization.

    Returns
    -------
    matplotlib.figure.Figure
        Dotplot figure with:
        - X-axis: Groups (archetypes/patterns)
        - Y-axis: Features (genes/pathways), sorted by effect size
        - Dot size: Effect magnitude (with legend)
        - Dot color: -log10(p-value) (with colorbar)
        - Background panels grouping features by archetype

    Notes
    -----
    Auto-detects result type and adjusts columns:
    - Gene results: Uses defaults
    - Pathway results: Switches y_col to 'pathway'
    - Pattern results: Uses pattern_name, effect_range
    - Exclusivity results: Uses tradeoff_score, effect_range

    Examples
    --------
    >>> gene_results = pc.tl.gene_associations(adata)
    >>> fig = pc.pl.dotplot(gene_results)
    >>> plt.show()

    >>> # Pathway results (auto-detected)
    >>> pathway_results = pc.tl.pathway_associations(adata)
    >>> fig = pc.pl.dotplot(pathway_results, title="Pathway Associations")
    """
    return _create_dotplot(
        results_df=results_df,
        x_col=x_col,
        y_col=y_col,
        size_col=size_col,
        color_col=color_col,
        top_n_per_group=top_n_per_group,
        filter_zero_p=filter_zero_p,
        log_transform_p=log_transform_p,
        max_log_p=max_log_p,
        title=title,
        figsize=figsize,
        color_palette=color_palette,
        save_path=save_path,
        **kwargs,
    )

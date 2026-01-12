"""Visualization functions for pattern analysis results."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .results import dotplot as create_dotplot_visualization


def pattern_dotplot(
    pattern_df: pd.DataFrame,
    pattern_type: str | None = None,
    top_n: int = 20,
    min_effect_size: float = 0.5,
    max_pvalue: float = 0.05,
    figsize: tuple[float, float] = (12, 8),
    title: str | None = None,
    save_path: str | None = None,
    **kwargs,
) -> plt.Figure:
    """Create dotplot for pattern analysis results.

    Visualizes the top features (pathways/genes) for each pattern,
    showing effect sizes and significance levels.

    Parameters
    ----------
    pattern_df : pd.DataFrame
        Results from archetype_exclusive_patterns, specialization_patterns,
        or tradeoff_patterns functions.
    pattern_type : str, optional
        Type of pattern for title generation ("exclusive", "specialization",
        "tradeoff"). If None, inferred from data.
    top_n : int, default: 20
        Number of top features to show.
    min_effect_size : float, default: 0.5
        Minimum absolute effect size to include.
    max_pvalue : float, default: 0.05
        Maximum p-value to include.
    figsize : tuple, default: (12, 8)
        Figure size.
    title : str, optional
        Custom title. If None, auto-generated based on pattern_type.
    save_path : str, optional
        Path to save the figure.
    **kwargs
        Additional arguments passed to create_dotplot_visualization.

    Returns
    -------
    matplotlib.figure.Figure
        Dotplot figure with:
        - X-axis: Pattern codes or archetype names
        - Y-axis: Feature names (genes/pathways)
        - Dot size: Effect size magnitude
        - Dot color: -log10(p-value) (significance)

        Returns placeholder figure with message if input is empty
        or no features pass filters.

    Examples
    --------
    >>> exclusive = pc.tl.archetype_exclusive_patterns(adata)
    >>> fig = pc.pl.pattern_dotplot(exclusive, pattern_type="exclusive")

    >>> tradeoffs = pc.tl.tradeoff_patterns(adata, tradeoffs="pairs")
    >>> fig = pc.pl.pattern_dotplot(tradeoffs, min_effect_size=1.0, top_n=30, title="Strong Tradeoff Patterns")
    """
    if pattern_df.empty:
        warnings.warn("Input DataFrame is empty. No patterns to visualize.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No patterns found", ha="center", va="center", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        return fig

    # Filter by effect size and p-value
    filtered_df = pattern_df.copy()

    # Handle different column names for effect size
    effect_col = None
    for col in ["mean_diff", "mean_effect_size", "effect_size", "log_fold_change"]:
        if col in filtered_df.columns:
            effect_col = col
            break

    if effect_col:
        filtered_df = filtered_df[np.abs(filtered_df[effect_col]) >= min_effect_size]

    # Filter by p-value
    pval_col = None
    for col in ["pvalue", "p_value", "min_pvalue"]:
        if col in filtered_df.columns:
            pval_col = col
            break

    if pval_col:
        filtered_df = filtered_df[filtered_df[pval_col] <= max_pvalue]

    # Select top N features
    if len(filtered_df) > top_n:
        # Sort by effect size or p-value
        if effect_col:
            filtered_df = filtered_df.nlargest(top_n, effect_col, keep="all")
        elif pval_col:
            filtered_df = filtered_df.nsmallest(top_n, pval_col, keep="all")
        else:
            filtered_df = filtered_df.head(top_n)

    # Determine feature column name
    feature_col = None
    for col in ["pathway", "gene", "feature"]:
        if col in filtered_df.columns:
            feature_col = col
            break

    # Determine pattern column name
    pattern_col = None
    for col in ["pattern_code", "pattern", "archetype"]:
        if col in filtered_df.columns:
            pattern_col = col
            break

    # Infer pattern type if not provided
    if pattern_type is None:
        if "pattern_code" in filtered_df.columns:
            if "exclusive" in str(filtered_df["pattern_code"].iloc[0]):
                pattern_type = "exclusive"
            elif "specialization" in str(filtered_df.get("pattern_type", "")).lower():
                pattern_type = "specialization"
            else:
                pattern_type = "tradeoff"

    # Generate title if not provided
    if title is None:
        if pattern_type == "exclusive":
            title = "Archetype-Exclusive Features"
        elif pattern_type == "specialization":
            title = "Specialization Patterns (vs Centroid)"
        elif pattern_type == "tradeoff":
            title = "Tradeoff/Mutual Exclusivity Patterns"
        else:
            title = "Pattern Analysis Results"

    # Prepare data for dotplot
    dotplot_df = filtered_df.copy()

    # Rename columns to match dotplot expectations
    if feature_col and feature_col != "gene":
        dotplot_df["gene"] = dotplot_df[feature_col]
    if pattern_col and pattern_col != "archetype":
        dotplot_df["archetype"] = dotplot_df[pattern_col]
    if effect_col and effect_col != "mean_archetype":
        dotplot_df["mean_archetype"] = np.abs(dotplot_df[effect_col])
    if pval_col and pval_col != "pvalue":
        dotplot_df["pvalue"] = dotplot_df[pval_col]

    # Create dotplot
    fig = create_dotplot_visualization(
        dotplot_df,
        x_col="archetype",
        y_col="gene",
        size_col="mean_archetype",
        color_col="pvalue",
        title=title,
        figsize=figsize,
        save_path=save_path,
        **kwargs,
    )

    return fig


def pattern_summary_barplot(
    pattern_results: dict[str, pd.DataFrame], figsize: tuple[float, float] = (14, 6), save_path: str | None = None
) -> plt.Figure:
    """Create summary barplot showing pattern counts across different analyses.

    Parameters
    ----------
    pattern_results : dict
        Dictionary with pattern DataFrames from different analyses
        e.g., {'exclusive': exclusive_df, 'specialization': spec_df, 'tradeoff': trade_df}
    figsize : tuple, default: (14, 6)
        Figure size
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    plt.Figure
        The summary barplot figure

    Examples
    --------
    >>> results = {
    ...     "exclusive": pc.tl.archetype_exclusive_patterns(adata),
    ...     "specialization": pc.tl.specialization_patterns(adata),
    ...     "tradeoff_pairs": pc.tl.tradeoff_patterns(adata, tradeoffs="pairs"),
    ... }
    >>> fig = pc.pl.pattern_summary_barplot(results)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Count patterns by type
    pattern_counts = {}
    significant_counts = {}

    for name, df in pattern_results.items():
        if not df.empty:
            pattern_counts[name] = len(df)
            if "significant" in df.columns:
                significant_counts[name] = df["significant"].sum()
            elif "fdr_pvalue" in df.columns:
                significant_counts[name] = (df["fdr_pvalue"] <= 0.05).sum()
            else:
                significant_counts[name] = pattern_counts[name]

    # Plot total patterns
    ax1 = axes[0]
    if pattern_counts:
        bars1 = ax1.bar(pattern_counts.keys(), pattern_counts.values(), color="skyblue", edgecolor="black")
        ax1.set_title("Total Patterns Identified", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Number of Patterns", fontsize=12)
        ax1.set_xlabel("Pattern Type", fontsize=12)
        ax1.tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f"{int(height)}", ha="center", va="bottom")

    # Plot significant patterns
    ax2 = axes[1]
    if significant_counts:
        bars2 = ax2.bar(significant_counts.keys(), significant_counts.values(), color="coral", edgecolor="black")
        ax2.set_title("Significant Patterns (FDR < 0.05)", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Number of Patterns", fontsize=12)
        ax2.set_xlabel("Pattern Type", fontsize=12)
        ax2.tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f"{int(height)}", ha="center", va="bottom")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def pattern_heatmap(
    pattern_df: pd.DataFrame,
    adata,
    top_n: int = 30,
    cluster_patterns: bool = True,
    cluster_features: bool = True,
    figsize: tuple[float, float] = (10, 12),
    cmap: str = "RdBu_r",
    save_path: str | None = None,
    **kwargs,
) -> plt.Figure:
    """Create heatmap showing pattern expression across archetypes.

    Parameters
    ----------
    pattern_df : pd.DataFrame
        Pattern analysis results
    adata : AnnData
        Annotated data object with archetype assignments and scores
    top_n : int, default: 30
        Number of top features to show
    cluster_patterns : bool, default: True
        Whether to cluster patterns (rows)
    cluster_features : bool, default: True
        Whether to cluster features (columns)
    figsize : tuple, default: (10, 12)
        Figure size
    cmap : str, default: 'RdBu_r'
        Colormap for the heatmap
    save_path : str, optional
        Path to save the figure
    **kwargs
        Additional arguments passed to sns.heatmap

    Returns
    -------
    plt.Figure
        The heatmap figure
    """
    # Implementation would create a heatmap showing how features vary across patterns
    # This is a placeholder for the full implementation
    warnings.warn("pattern_heatmap is not yet implemented")
    fig, ax = plt.subplots(figsize=figsize)
    ax.text(0.5, 0.5, "Heatmap visualization coming soon", ha="center", va="center", fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig

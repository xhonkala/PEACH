"""
Results Visualization Module for Archetypal Analysis
===================================================

PURPOSE: Create publication-ready visualizations for gene-archetype and pathway-archetype association results.

ARCHITECTURAL ROLE:
- Translates statistical test results into interpretable visualizations
- Provides dotplot and heatmap visualizations for association patterns
- Implements filtering and ranking for focused biological interpretation
- Supports both gene expression and pathway activity association results

DESIGN PRINCIPLES:
- Publication-ready plots with customizable aesthetics
- Automatic filtering for top associations per archetype
- Flexible color schemes and sizing based on statistical significance
- Integration with matplotlib/seaborn for consistent styling
- Handles both continuous (expression/activity) and categorical (enrichment) data

=== MODULE API INVENTORY ===

MAIN FUNCTIONS:
 create_dotplot_visualization(results_df, x_col='archetype', y_col='gene', size_col='mean_archetype', color_col='pvalue', top_n_per_group=10, filter_zero_p=True, log_transform_p=True, max_log_p=300, title='Gene-Archetype Associations', figsize=(12, 8), color_palette='plasma', save_path=None) -> matplotlib.figure.Figure
    Purpose: Create dotplot visualization similar to R ggplot2 implementation
    Inputs: results_df(DataFrame with association results), various plotting parameters
    Outputs: matplotlib Figure object
    Side Effects: Optional file saving, plot display

 create_heatmap_visualization(results_df, x_col='archetype', y_col='gene', value_col='log_fold_change', significance_col='pvalue', top_n_per_group=15, filter_zero_p=True, title='Association Heatmap', figsize=(10, 8), cmap='RdBu_r', save_path=None) -> matplotlib.figure.Figure
    Purpose: Create heatmap visualization for association patterns
    Inputs: results_df(DataFrame), plotting parameters
    Outputs: matplotlib Figure object
    Side Effects: Optional file saving, plot display

 visualize_archetypal_space_3d_single(adata, archetype_coords_key='archetype_coordinates', pca_key='X_pca', color_by=None, cell_size=2.0, cell_opacity=0.6, archetype_size=8.0, archetype_color='red', show_archetype_labels=True, show_connections=True, color_scale='viridis', categorical_colors=None, title='Archetypal Space Visualization', auto_scale=True, fixed_ranges=None, save_path=None) -> plotly.graph_objects.Figure
    Purpose: Interactive 3D visualization of single archetypal analysis result in PCA space
    Inputs: adata(AnnData with PCA and archetype coordinates), coloring and styling parameters
    Outputs: plotly Figure object with interactive 3D scatter plot
    Side Effects: Optional HTML file saving, requires plotly

 visualize_archetypal_space_3d_multi(adata_list, archetype_coords_key='archetype_coordinates', pca_key='X_pca', labels_list=None, color_by=None, color_values=None, cell_size=2.0, cell_opacity=0.6, archetype_size=8.0, archetype_colors=None, show_labels=True, auto_scale=True, range_reference=None, fixed_ranges=None, color_scale='viridis', categorical_colors=None, title='Multi-Archetypal Space Comparison', save_path=None) -> plotly.graph_objects.Figure
    Purpose: Interactive 3D comparison of multiple archetypal analysis results
    Inputs: adata_list(List of AnnData objects), comparison and styling parameters
    Outputs: plotly Figure object with multi-dataset 3D visualization
    Side Effects: Optional HTML file saving, requires plotly

 filter_top_associations(results_df, group_col='archetype', ranking_col='log_fold_change', top_n=10, filter_zero_p=True, p_col='pvalue') -> pd.DataFrame
    Purpose: Filter results to top N associations per group (archetype)
    Inputs: results_df(DataFrame), filtering parameters
    Outputs: Filtered DataFrame

 visualize_training_results(training_results, title='Model Training Progress', figsize=(12, 8), show_final_analysis=True) -> matplotlib.figure.Figure
    Purpose: Create comprehensive 4-subplot training visualization with loss components, performance metrics, stability, and constraint compliance
    Inputs: training_results(Dict from train_vae with 'history' key), plotting parameters
    Outputs: matplotlib Figure object with 2x2 subplot layout
    Side Effects: Optional final analysis summary printing

 print_training_metrics_summary(training_results) -> None
    Purpose: Print comprehensive training metrics summary to console with progress indicators
    Inputs: training_results(Dict from train_vae with 'history' key)
    Outputs: None (prints to console)
    Side Effects: Console output with metrics evolution, stability analysis, and convergence assessment

 visualize_training_progress_with_summary(training_results, title='Complete Training Analysis', figsize=(12, 8), show_plots=True, save_path=None) -> matplotlib.figure.Figure
    Purpose: Complete training analysis combining metrics summary printing and visualization
    Inputs: training_results(Dict from train_vae), title, figure parameters, display/save options
    Outputs: matplotlib Figure object
    Side Effects: Console output, optional plot display, optional file saving

 visualize_zinb_training_results(training_results, title='ZINB Model Training Progress', figsize=(12, 8), show_final_analysis=True) -> matplotlib.figure.Figure
    Purpose: Specialized ZINB training visualization with reconstruction loss components and convergence analysis
    Inputs: training_results(Dict from ZINB fit() with 'history' key), plotting parameters
    Outputs: matplotlib Figure object with ZINB-specific plots
    Side Effects: Optional final analysis summary printing

UTILITY FUNCTIONS:
 prepare_plotting_data(results_df, x_col, y_col, size_col, color_col, filter_zero_p, p_col) -> pd.DataFrame
 apply_log_transform_p(results_df, p_col, max_log_p) -> pd.DataFrame
 create_faceted_dotplot(results_df, x_col, y_col, size_col, color_col, facet_col, title, figsize, color_palette) -> matplotlib.figure.Figure

EXTERNAL DEPENDENCIES:
 matplotlib.pyplot: Core plotting functionality
 seaborn: Statistical visualization enhancements
 pandas: DataFrame operations and data manipulation
 numpy: Numerical operations and transformations
 typing: Type hints for function signatures

DATA FLOW PATTERNS:
 Input: Statistical test results DataFrame (from test_archetype_*_associations functions)
 Filter: Top N associations per archetype based on effect size or significance
 Transform: Log-transform p-values, handle zero p-values, prepare plotting coordinates
 Visualize: Create dotplot or heatmap with appropriate aesthetics
 Output: Publication-ready matplotlib Figure object

ERROR HANDLING:
 Missing column validation with informative error messages
 Empty DataFrame handling
 Invalid parameter validation
 Graceful handling of plotting edge cases (all zeros, single values)
 File saving error handling with fallback options

BIOLOGICAL INTERPRETATION:
 Dotplot size represents effect magnitude (mean difference, fold change)
 Color intensity represents statistical significance (-log10 p-value)
 Faceting by archetype enables cross-archetype comparison
 Top N filtering focuses on most biologically relevant associations
 Consistent color schemes enable cross-plot comparison
"""

import matplotlib
import numpy as np
import pandas as pd

# Set backend before importing pyplot to avoid GUI issues
matplotlib.use("Agg")  # Use non-interactive backend to avoid GUI issues
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns

# Try to import plotly for 3D visualizations
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. 3D visualizations will not work. Install with: pip install plotly")

# Set style for publication-ready plots
plt.style.use("default")
sns.set_palette("husl")


def create_dotplot_visualization(
    results_df: pd.DataFrame,
    x_col: str = "archetype",  # Updated for Python workflow
    y_col: str = "gene",  # Updated for Python workflow
    size_col: str = "mean_archetype",  # Updated for Python workflow
    color_col: str = "pvalue",  # Updated for Python workflow
    top_n_per_group: int = 10,
    filter_zero_p: bool = True,
    log_transform_p: bool = True,
    max_log_p: float = 300.0,
    title: str = "Gene-Archetype Associations",
    figsize: tuple[float, float] = (12, 8),
    color_palette: str = "plasma",
    save_path: str | Path | None = None,
    **kwargs,
) -> plt.Figure:
    """
    Create a dotplot visualization for gene-archetype or pathway-archetype associations.

    Replicates the R ggplot2 dotplot functionality with faceting, background panels,
    and publication-ready aesthetics.

    Args:
        results_df: DataFrame with association results containing columns specified by other parameters
        x_col: Column name for x-axis (typically archetype names)
        y_col: Column name for y-axis (typically gene/pathway names)
        size_col: Column name for dot size (typically effect size or mean difference)
        color_col: Column name for dot color (typically p-value)
        top_n_per_group: Number of top associations to show per archetype
        filter_zero_p: Whether to filter out p-values of exactly 0 (edge cases)
        log_transform_p: Whether to apply -log10 transformation to p-values
        max_log_p: Maximum -log10(p-value) to cap extremely small p-values
        title: Plot title
        figsize: Figure size as (width, height)
        color_palette: Matplotlib/seaborn color palette name
        save_path: Optional path to save the figure
        **kwargs: Additional arguments passed to matplotlib scatter

    Returns
    -------
        matplotlib Figure object

    Raises
    ------
        ValueError: If required columns are missing from results_df

    Examples
    --------
        # Basic usage with gene association results (uses defaults)
        fig = create_dotplot_visualization(
            gene_results,
            title='Gene-Archetype Associations',
            top_n_per_group=15
        )

        # Pathway associations with custom parameters
        fig = create_dotplot_visualization(
            pathway_results,
            y_col='pathway',  # Change from 'gene' to 'pathway'
            color_col='fdr_pvalue',  # Use FDR-corrected p-values
            title='Pathway-Archetype Associations'
        )

        # Pattern analysis results (auto-detects and adjusts columns)
        fig = create_dotplot_visualization(
            pattern_results,
            title='Archetypal Pattern Associations'
            # Function auto-detects: x_col='pattern_name', size_col='effect_range', color_col='max_positive_effect'
        )

        # Exclusivity results (completely auto-detected)
        fig = create_dotplot_visualization(
            exclusivity_results,
            title='Mutual Exclusivity Patterns'
            # Function auto-detects: y_col='pathway', size_col='tradeoff_score', color_col='effect_range'
        )

        # Manual column specification (if needed)
        fig = create_dotplot_visualization(
            any_results,
            x_col='custom_x',           # Custom x-axis
            y_col='custom_feature',     # Custom features
            size_col='custom_size',     # Custom sizing
            color_col='custom_color',   # Custom coloring
            title='Custom Visualization'
        )
    """
    # Import required for legend creation
    from matplotlib.lines import Line2D

    # Validate input DataFrame and columns
    if results_df.empty:
        raise ValueError("results_df is empty")

    # SMART COLUMN DETECTION: Auto-detect and adjust column mappings
    available_cols = list(results_df.columns)
    is_pattern_data = any(col in available_cols for col in ["positive_patterns", "negative_patterns", "tradeoff_score"])
    is_exclusivity_data = any(col in available_cols for col in ["effect_range", "max_positive_effect"])

    if is_exclusivity_data:
        # EXCLUSIVITY DATA: Auto-adjust for mutual exclusivity results
        if y_col == "gene" and "pathway" in available_cols:
            y_col = "pathway"
            print("[STATS] Exclusivity data detected: Using 'pathway' for y-axis")
        elif y_col == "gene" and "gene" not in available_cols:
            # Try to find any suitable feature column
            feature_cols = [col for col in available_cols if col in ["pathway", "feature", "gene_name"]]
            if feature_cols:
                y_col = feature_cols[0]
                print(f"[STATS] Exclusivity data detected: Using '{y_col}' for y-axis")

        if size_col == "mean_archetype" and "tradeoff_score" in available_cols:
            size_col = "tradeoff_score"
            print("[STATS] Exclusivity data detected: Using 'tradeoff_score' for dot sizes")

        if color_col == "pvalue" and "effect_range" in available_cols:
            color_col = "effect_range"
            print("[STATS] Exclusivity data detected: Using 'effect_range' for colors")

    elif is_pattern_data:
        # PATTERN DATA: Auto-adjust column mappings for pattern data
        if size_col == "mean_archetype" and "effect_range" in available_cols:
            size_col = "effect_range"
            print("[STATS] Pattern data detected: Using 'effect_range' for dot sizes")
        elif size_col == "mean_archetype" and "tradeoff_score" in available_cols:
            size_col = "tradeoff_score"
            print("[STATS] Pattern data detected: Using 'tradeoff_score' for dot sizes")

        if color_col == "pvalue" and "max_positive_effect" in available_cols:
            color_col = "max_positive_effect"
            print("[STATS] Pattern data detected: Using 'max_positive_effect' for colors")

        # Use pattern_name if available instead of archetype
        if x_col == "archetype" and "pattern_name" in available_cols:
            x_col = "pattern_name"
            print("[STATS] Pattern data detected: Using 'pattern_name' for x-axis")

    # GENERAL FALLBACKS: Try to find suitable columns if defaults don't exist
    if y_col not in available_cols:
        # Try common feature column names
        feature_candidates = ["pathway", "gene", "feature", "gene_name", "pathway_name"]
        found_feature_col = None
        for candidate in feature_candidates:
            if candidate in available_cols:
                found_feature_col = candidate
                break

        if found_feature_col:
            y_col = found_feature_col
            print(f"[STATS] Auto-detected feature column: Using '{y_col}' for y-axis")
        else:
            raise ValueError(f"No suitable y-axis column found. Available columns: {available_cols}")

    if size_col not in available_cols:
        # Try common size column names
        size_candidates = ["mean_archetype", "tradeoff_score", "effect_range", "log_fold_change", "mean_diff"]
        found_size_col = None
        for candidate in size_candidates:
            if candidate in available_cols:
                found_size_col = candidate
                break

        if found_size_col:
            size_col = found_size_col
            print(f"[STATS] Auto-detected size column: Using '{size_col}' for dot sizes")
        else:
            raise ValueError(f"No suitable size column found. Available columns: {available_cols}")

    if color_col not in available_cols:
        # Try common color column names
        color_candidates = ["pvalue", "fdr_pvalue", "effect_range", "max_positive_effect", "log_fold_change"]
        found_color_col = None
        for candidate in color_candidates:
            if candidate in available_cols:
                found_color_col = candidate
                break

        if found_color_col:
            color_col = found_color_col
            print(f"[STATS] Auto-detected color column: Using '{color_col}' for colors")
        else:
            raise ValueError(f"No suitable color column found. Available columns: {available_cols}")

    # Check if we're using fdr_pvalue as color and it contains zeros - only disable if truly problematic
    if color_col == "fdr_pvalue" and log_transform_p:
        # Check if the column actually has problematic zeros that would cause visualization issues
        fdr_values = results_df[color_col].dropna()
        if len(fdr_values) > 0:
            zero_count = (fdr_values == 0.0).sum()
            total_count = len(fdr_values)
            zero_fraction = zero_count / total_count

            # Only disable log transform if a large fraction are exactly zero (which would make visualization poor)
            if zero_fraction > 0.5:
                log_transform_p = False
                print(
                    f"[WARNING]  Disabled log transform for {color_col} due to {zero_count}/{total_count} values being exactly 0.0"
                )
            elif zero_count > 0:
                print(
                    f"[STATS] Found {zero_count}/{total_count} zero values in {color_col}, but proceeding with log transform (handled automatically)"
                )
        else:
            log_transform_p = False
            print(f"[WARNING]  Disabled log transform for {color_col} - no valid values found")

    # Early validation to prevent empty data issues
    if len(results_df) == 0:
        raise ValueError("results_df is empty after filtering")

    # Validate that we have data for all required columns
    for col in [x_col, y_col, size_col, color_col]:
        if results_df[col].isna().all():
            raise ValueError(f"Column '{col}' contains only NaN values")

    required_cols = [x_col, y_col, size_col, color_col]
    missing_cols = [col for col in required_cols if col not in available_cols]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available: {available_cols}")

    # Filter and prepare data - prioritize effect size columns for ranking
    ranking_col = size_col
    if "log_fold_change" in results_df.columns:
        ranking_col = "log_fold_change"

    plot_data = filter_top_associations(
        results_df=results_df,
        group_col=x_col,
        ranking_col=ranking_col,  # Rank by effect size preferentially
        top_n=top_n_per_group,
        filter_zero_p=filter_zero_p,
        p_col=color_col,
    )

    if plot_data.empty:
        warnings.warn("No data remaining after filtering")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data to display", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig

    # Validate we have data for all required columns
    if plot_data[size_col].isna().all():
        warnings.warn(f"All values in size column '{size_col}' are NaN")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"No valid {size_col} data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig

    # Apply log transformation to p-values if requested
    if log_transform_p:
        # Only apply log transform if the column contains p-values (has values between 0 and 1)
        color_values = plot_data[color_col].dropna()
        is_pvalue_column = len(color_values) > 0 and color_values.min() >= 0 and color_values.max() <= 1

        if is_pvalue_column:
            plot_data = apply_log_transform_p(plot_data, color_col, max_log_p)
            color_label = f"-log10({color_col})"
            color_column = f"log_{color_col}"

            # Validate the log-transformed column was created successfully
            if color_column not in plot_data.columns:
                print(f"[WARNING]  Log transformation failed, using original {color_col}")
                color_label = color_col
                color_column = color_col
        else:
            print(
                f"[WARNING]  {color_col} doesn't appear to be a p-value column (range: {color_values.min():.3f}-{color_values.max():.3f}), skipping log transform"
            )
            color_label = color_col
            color_column = color_col
    else:
        color_label = color_col
        color_column = color_col

    # Final validation that color column exists and has valid data
    if color_column not in plot_data.columns:
        raise ValueError(f"Color column '{color_column}' not found in plot data")

    color_data = plot_data[color_column].dropna()
    if len(color_data) == 0:
        raise ValueError(f"Color column '{color_column}' contains only NaN values")

    # Adjust figure size based on content and detect pathway data
    n_items = len(plot_data[y_col].unique())
    n_archetypes = len(plot_data[x_col].unique())

    # Detect if this is pathway data (longer names, typically fewer items per archetype)
    is_pathway_data = (
        y_col.lower() in ["pathway", "pathways"]
        or any(
            name.startswith("GOBP_") or name.startswith("KEGG_") or len(name) > 30
            for name in plot_data[y_col].unique()[:5]
        )  # Check first 5 names
    )

    # SAFETY LIMITS: Prevent matplotlib from trying to create massive images
    MAX_FIGURE_HEIGHT = 200  # Maximum height in inches (reasonable for high-res displays)
    MAX_FIGURE_WIDTH = 50  # Maximum width in inches

    if is_pathway_data:
        # Pathway data needs more height but EXTREMELY NARROW columns for space efficiency
        base_height = max(figsize[1], n_items * 0.35)  # Compact height
        base_width = max(figsize[0], n_archetypes * 1.0 + 15)  # EXTREMELY NARROW columns, MAXIMUM space for labels

        # Apply safety limits
        base_height = min(base_height, MAX_FIGURE_HEIGHT)
        base_width = min(base_width, MAX_FIGURE_WIDTH)

        adjusted_figsize = (base_width, base_height)
        font_size_y = 10  # LARGER font for pathway names
        font_size_x = 11  # Font for x-axis
        font_size_title = 14  # Title
    else:
        # Gene data - extremely tight spacing
        if n_items > 20:
            base_height = max(figsize[1], n_items * 0.25)  # EXTREMELY narrow columns
            base_width = max(figsize[0], n_archetypes * 0.8 + 10)  # MUCH NARROWER
        else:
            base_height = figsize[1]
            base_width = max(figsize[0], n_archetypes * 0.8 + 10)  # MUCH NARROWER

        # Apply safety limits
        base_height = min(base_height, MAX_FIGURE_HEIGHT)
        base_width = min(base_width, MAX_FIGURE_WIDTH)

        adjusted_figsize = (base_width, base_height)
        font_size_y = 10  # LARGER font for gene names
        font_size_x = 10  # Font for x-axis
        font_size_title = 13  # Title

    # Warn user if we're hitting safety limits
    if base_height >= MAX_FIGURE_HEIGHT or base_width >= MAX_FIGURE_WIDTH:
        warnings.warn(
            f"[WARNING]  Large plot detected (n_items={n_items}, n_archetypes={n_archetypes}). "
            f"Applied size limits: {adjusted_figsize}. Consider filtering to fewer associations."
        )

    # Additional safety check for unreasonable data sizes
    if n_items > 2000:
        raise ValueError(
            f"Too many items to plot ({n_items}). Consider using top_n_per_group to filter to <2000 total items."
        )

    if n_archetypes > 50:
        raise ValueError(f"Too many archetypes to plot ({n_archetypes}). Consider grouping or filtering archetypes.")

    # Create single plot with archetypes as columns (like R ggplot2 version)
    fig, ax = plt.subplots(figsize=adjusted_figsize)

    # Get unique archetypes
    archetypes = sorted(plot_data[x_col].unique())
    archetype_positions = {arch: i for i, arch in enumerate(archetypes)}

    # Create grouped gene positions - group genes by archetype
    # Sort genes within each archetype by effect size (size_col) instead of p-value
    gene_y_positions = {}
    current_y_pos = 0
    archetype_y_ranges = {}  # Track y-range for each archetype

    for arch in archetypes:
        arch_data = plot_data[plot_data[x_col] == arch].copy()

        # Sort by effect size (descending) instead of p-value
        arch_genes = arch_data.sort_values(size_col, ascending=False)[y_col].unique()

        # Record y-range for this archetype
        start_y = current_y_pos

        for gene in arch_genes:
            gene_y_positions[f"{arch}_{gene}"] = current_y_pos
            current_y_pos += 1

        # Record archetype range (no spacing between groups for compact layout)
        archetype_y_ranges[arch] = (start_y, current_y_pos - 1)
        # Removed spacing: current_y_pos += 1.0  # Creates empty columns, not desired

    # Add position columns
    plot_data["x_pos"] = plot_data[x_col].map(archetype_positions)
    plot_data["y_pos"] = plot_data.apply(lambda row: gene_y_positions[f"{row[x_col]}_{row[y_col]}"], axis=1)

    # Initial validation that we have data to work with
    if len(gene_y_positions) == 0:
        raise ValueError("No valid data points to plot after processing")

    # Get scaling parameters with NaN handling and negative value protection
    size_values = plot_data[size_col].dropna()
    if len(size_values) == 0:
        # No valid size values
        plot_data["scaled_size"] = 150  # Default size
        size_min = size_max = 150
        size_values_for_scaling = size_values  # Empty series
        size_col_for_scaling = size_col
    else:
        # Check if we have negative values (matplotlib scatter requires positive sizes)
        has_negative = (size_values < 0).any()
        if has_negative:
            print(f"[STATS] Detected negative values in {size_col}, using absolute values for dot sizes")
            # Use absolute values for size scaling, but apply better scaling for log fold changes
            size_values_for_scaling = size_values.abs()
            plot_data[f"{size_col}_abs"] = plot_data[size_col].abs()
            size_col_for_scaling = f"{size_col}_abs"

            # For log fold changes, also provide better legend labels showing original range
            if "log_fold_change" in size_col:
                original_min, original_max = size_values.min(), size_values.max()
                print(f"   Original {size_col} range: [{original_min:.3f}, {original_max:.3f}]")
                print(f"   Using absolute values for sizing: [0, {size_values_for_scaling.max():.3f}]")
        else:
            size_values_for_scaling = size_values
            size_col_for_scaling = size_col

        size_min, size_max = size_values_for_scaling.min(), size_values_for_scaling.max()
        size_range = (50, 400)  # Point size range for matplotlib

        # Scale sizes with protection against division by zero
        if size_max > size_min and not np.isclose(size_max, size_min):
            # Use the appropriate column for scaling (absolute values if original had negatives)
            # Additional protection against division by zero
            denominator = size_max - size_min
            if abs(denominator) > 1e-10:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    plot_data["scaled_size"] = size_range[0] + (
                        plot_data[size_col_for_scaling] - size_min
                    ) / denominator * (size_range[1] - size_range[0])
                # Fill NaN values with default size
                plot_data["scaled_size"] = plot_data["scaled_size"].fillna(np.mean(size_range))
            else:
                # Denominator too small, use default size
                plot_data["scaled_size"] = np.mean(size_range)
        else:
            # All values are the same or very close
            plot_data["scaled_size"] = np.mean(size_range)

    # Add HORIZONTAL background panels for each archetype - highlights genes/pathways per archetype
    # This visually groups genes/pathways that belong to each archetype horizontally
    for i, (arch, (y_min, y_max)) in enumerate(archetype_y_ranges.items()):
        # Create horizontal bands that span all archetypes but group by gene/pathway sets
        ax.axhspan(y_min - 0.3, y_max + 0.3, alpha=0.06, color=plt.cm.Set3(i % 12), zorder=0)

        # Optional: Add subtle archetype column highlights (much more subtle than before)
        ax.axvspan(i - 0.05, i + 0.05, alpha=0.02, color=plt.cm.Pastel1(i % 9), zorder=0)

    # Validate data before scatter plot to prevent matplotlib errors
    x_data = plot_data["x_pos"].values
    y_data = plot_data["y_pos"].values
    size_data = plot_data["scaled_size"].values
    color_data = plot_data[color_column].values

    # Check for any issues that could cause matplotlib concatenation errors
    if len(x_data) == 0 or len(y_data) == 0:
        raise ValueError("No data points to plot - empty position arrays")

    # Check for infinite or NaN values and ensure reasonable ranges
    x_valid = np.isfinite(x_data)
    y_valid = np.isfinite(y_data)
    size_valid = np.isfinite(size_data) & (size_data > 0) & (size_data < 10000)  # Size must be positive and reasonable
    color_valid = np.isfinite(color_data)

    # Additional debug info for size_col = log_fold_change case
    # if 'log_fold_change' in size_col:
    #     print(f" Debug info for log_fold_change:")
    #     print(f"   Original size_col: {size_col}")
    #     print(f"   Size data range: {size_data.min():.6f} to {size_data.max():.6f}")
    #     print(f"   Color data range: {color_data.min():.6f} to {color_data.max():.6f}")
    #     print(f"   Valid sizes: {size_valid.sum()}/{len(size_valid)}")
    #     print(f"   Size data dtype: {size_data.dtype}")
    #     print(f"   Color data dtype: {color_data.dtype}")
    #     print(f"   X data range: {x_data.min()} to {x_data.max()}")
    #     print(f"   Y data range: {y_data.min()} to {y_data.max()}")
    #     print(f"   Data arrays are contiguous: x={x_data.flags['C_CONTIGUOUS']}, y={y_data.flags['C_CONTIGUOUS']}, s={size_data.flags['C_CONTIGUOUS']}, c={color_data.flags['C_CONTIGUOUS']}")

    # Find rows where all values are valid
    all_valid = x_valid & y_valid & size_valid & color_valid

    if not all_valid.any():
        raise ValueError("No valid data points - all contain NaN/inf/invalid values")

    if not all_valid.all():
        n_invalid = (~all_valid).sum()

        # Detailed explanation of what makes data invalid
        invalid_reasons = []
        if (~x_valid).any():
            invalid_reasons.append(f"{(~x_valid).sum()} with invalid x positions")
        if (~y_valid).any():
            invalid_reasons.append(f"{(~y_valid).sum()} with invalid y positions")
        if (~size_valid).any():
            invalid_reasons.append(f"{(~size_valid).sum()} with invalid sizes (NaN/inf/zero/too large)")
        if (~color_valid).any():
            invalid_reasons.append(f"{(~color_valid).sum()} with invalid color values (NaN/inf)")

        reason_text = ", ".join(invalid_reasons)
        print(f"[WARNING]  Filtering out {n_invalid} invalid data points before plotting: {reason_text}")

        # Filter the plot_data DataFrame to remove invalid rows entirely
        # This prevents empty rows from appearing in the plot
        plot_data = plot_data[all_valid].reset_index(drop=True)

        # Update the data arrays
        x_data = plot_data["x_pos"].values
        y_data = plot_data["y_pos"].values
        size_data = plot_data["scaled_size"].values
        color_data = plot_data[color_column].values

    # Create y-tick labels and positions AFTER filtering to avoid empty rows
    y_tick_positions = []
    y_tick_labels = []

    for arch in archetypes:
        arch_data = plot_data[plot_data[x_col] == arch].copy()
        if len(arch_data) == 0:
            continue  # Skip empty archetype groups

        arch_genes = arch_data.sort_values(size_col, ascending=False)[y_col].unique()

        for gene in arch_genes:
            # Use the actual y_pos from the filtered data
            gene_rows = arch_data[arch_data[y_col] == gene]
            if len(gene_rows) > 0:
                pos = gene_rows.iloc[0]["y_pos"]  # Get actual y position
                y_tick_positions.append(pos)
                # Clean gene/pathway names - allow much longer names since we have more space
                clean_label = gene

                # Truncate from the end if too long, keeping the beginning (including prefixes)
                max_chars = 75 if is_pathway_data else 55  # MUCH more chars for pathways
                if len(clean_label) > max_chars:
                    clean_label = clean_label[: max_chars - 3] + "..."

                y_tick_labels.append(clean_label)

    # Final validation that we have data to plot
    if len(y_tick_positions) == 0:
        raise ValueError("No valid data points to plot after filtering")

    # Create the scatter plot with validated data
    scatter = ax.scatter(
        x=x_data,
        y=y_data,
        s=size_data,
        c=color_data,
        cmap=color_palette,
        alpha=0.9,
        edgecolors="white",
        linewidth=0.5,
        **kwargs,
    )

    # Set archetype labels on x-axis
    ax.set_xticks(range(len(archetypes)))
    ax.set_xticklabels(archetypes, rotation=45, ha="right", fontweight="bold")

    # Set gene/pathway labels on y-axis with dynamic font sizing
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels, fontsize=font_size_y)  # Dynamic font based on content type
    ax.tick_params(axis="x", labelsize=font_size_x)  # Apply x-axis font size

    # Remove redundant archetype labels on right side - x-axis labels are sufficient

    # Customize the plot with larger fonts
    ax.set_xlabel("Archetype", fontweight="bold", fontsize=font_size_x + 1)
    ax.set_ylabel(y_col.replace("_", " ").title(), fontweight="bold", fontsize=font_size_y + 1)
    ax.set_title(title, fontsize=font_size_title, fontweight="bold", pad=20)

    # Add colorbar for significance - adjust positioning based on data type
    if is_pathway_data:
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.4, pad=0.02)  # Smaller, more padding
        cbar.set_label(color_label, fontweight="bold", fontsize=10)
        cbar.ax.tick_params(labelsize=9)
    else:
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.02)  # Smaller, more padding
        cbar.set_label(color_label, fontweight="bold", fontsize=11)
        cbar.ax.tick_params(labelsize=10)

    # Add minor grid lines
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Create size legend with protection against edge cases
    # Use the same values that were used for scaling to avoid mismatch issues
    if len(size_values_for_scaling) > 0 and size_max > size_min and not np.isclose(size_max, size_min):
        # FIX: Use actual data range from the plot, not scaled values
        # This ensures legend accurately represents what's actually plotted
        actual_data_min = plot_data[size_col].min()
        actual_data_max = plot_data[size_col].max()
        actual_data_mid = (actual_data_min + actual_data_max) / 2

        # Protected calculation to avoid division issues
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if np.isfinite(actual_data_mid) and actual_data_max > actual_data_min:
                size_legend_values = [actual_data_min, actual_data_mid, actual_data_max]
            else:
                # Fallback to simple labeling if calculation produces invalid result
                size_legend_values = (
                    [actual_data_min, actual_data_max] if actual_data_max > actual_data_min else [actual_data_min]
                )
            size_legend_labels = [f"{val:.2f}" for val in size_legend_values]

        # Create legend elements with protected scaling
        legend_elements = []
        size_range = (50, 400)  # Ensure size_range is defined here
        for val, label in zip(size_legend_values, size_legend_labels, strict=False):
            # Protected division to prevent division by zero using ACTUAL data range
            if actual_data_max > actual_data_min and not np.isclose(actual_data_max, actual_data_min):
                # Additional protection against numerical issues
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    denominator = actual_data_max - actual_data_min
                    if abs(denominator) > 1e-10:  # Additional numerical check
                        scaled_size = size_range[0] + (val - actual_data_min) / denominator * (
                            size_range[1] - size_range[0]
                        )
                    else:
                        scaled_size = np.mean(size_range)
            else:
                scaled_size = np.mean(size_range)
            # Create legend elements using Line2D instead of scatter to avoid empty array concatenation issues
            # FIXED: Match the actual plot scaling - matplotlib scatter uses area (s) but Line2D uses radius (markersize)
            # Convert from scatter area to Line2D radius: markersize = sqrt(s/π)
            legend_markersize = np.sqrt(scaled_size / np.pi)  # No additional scaling - direct conversion
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="gray",
                    markersize=legend_markersize,
                    alpha=0.7,
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                    label=label,
                )
            )

        # Add size legend - dynamic positioning to avoid overlap with colorbar
        if is_pathway_data:
            # Position high for pathway data
            legend_bbox = (1.02, 0.85)  # High position, slightly more to the right
            legend_fontsize = 9
        else:
            # Even higher for gene data which has longer colorbar
            legend_bbox = (1.02, 0.90)  # Very high position
            legend_fontsize = 10

        size_legend = ax.legend(
            handles=legend_elements,
            title=size_col.replace("_", " ").title(),
            loc="center left",
            bbox_to_anchor=legend_bbox,
            frameon=True,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize + 2,
            markerscale=1.0,
        )
        size_legend.get_frame().set_facecolor("white")
        size_legend.get_frame().set_alpha(0.9)
    elif len(size_values_for_scaling) > 0:
        # All values are the same - create simple legend
        default_size = 225  # Middle of range (50, 400)
        # Create legend elements using Line2D instead of scatter to avoid empty array concatenation issues
        # FIXED: Match the actual plot scaling - convert from scatter area to Line2D radius
        legend_markersize = np.sqrt(default_size / np.pi)  # No additional scaling - direct conversion
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=legend_markersize,
                alpha=0.7,
                markeredgecolor="white",
                markeredgewidth=0.5,
                label=f"{size_values_for_scaling.iloc[0]:.2f}",
            )
        ]

        # Adjust position for data type to avoid overlap
        if is_pathway_data:
            legend_bbox = (1.02, 0.85)  # Higher position
            legend_fontsize = 8
        else:
            legend_bbox = (1.02, 0.90)  # Very high position for gene data
            legend_fontsize = 9

        size_legend = ax.legend(
            handles=legend_elements,
            title=size_col.replace("_", " ").title(),
            loc="center left",
            bbox_to_anchor=legend_bbox,
            frameon=True,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize + 1,
            markerscale=1.0,
        )
        size_legend.get_frame().set_facecolor("white")
        size_legend.get_frame().set_alpha(0.9)

    # Set axis limits - EXTREMELY tight bounds for minimal horizontal spacing
    ax.set_xlim(-0.2, len(archetypes) - 0.8)  # EXTREMELY tight horizontal bounds for minimal spacing
    ax.set_ylim(-0.5, max(y_tick_positions) + 0.5)  # Keep vertical as is

    # Adjust layout based on data type - give MAXIMUM space for labels
    plt.tight_layout()
    if is_pathway_data:
        plt.subplots_adjust(right=0.65, left=0.30)  # MAXIMUM space for pathway names, minimal legend space
    else:
        plt.subplots_adjust(right=0.70, left=0.20)  # Much more space for gene names

    # Save if requested
    if save_path:
        try:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"[OK] Figure saved to: {save_path}")
        except Exception as e:
            warnings.warn(f"Failed to save figure: {e}")

    return fig


def create_heatmap_visualization(
    results_df: pd.DataFrame,
    x_col: str = "archetype",  # Updated for Python workflow
    y_col: str = "gene",  # Updated for Python workflow
    value_col: str = "log_fold_change",  # Updated for Python workflow
    significance_col: str = "pvalue",  # Updated for Python workflow
    top_n_per_group: int = 15,
    filter_zero_p: bool = True,
    significance_threshold: float = 0.05,
    title: str = "Association Heatmap",
    figsize: tuple[float, float] = (10, 8),
    cmap: str = "RdBu_r",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Create a heatmap visualization for association patterns.

    Args:
        results_df: DataFrame with association results
        x_col: Column name for x-axis (archetypes)
        y_col: Column name for y-axis (genes/pathways)
        value_col: Column name for heatmap values (effect sizes)
        significance_col: Column name for significance values
        top_n_per_group: Number of top associations per archetype
        filter_zero_p: Whether to filter zero p-values
        significance_threshold: P-value threshold for significance marking
        title: Plot title
        figsize: Figure size
        cmap: Colormap name
        save_path: Optional save path

    Returns
    -------
        matplotlib Figure object
    """
    # Filter data
    plot_data = filter_top_associations(
        results_df=results_df,
        group_col=x_col,
        ranking_col=value_col,
        top_n=top_n_per_group,
        filter_zero_p=filter_zero_p,
        p_col=significance_col,
    )

    if plot_data.empty:
        warnings.warn("No data remaining after filtering")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data to display", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig

    # Create pivot table for heatmap
    heatmap_data = plot_data.pivot_table(index=y_col, columns=x_col, values=value_col, fill_value=0)

    # Create significance mask
    sig_data = plot_data.pivot_table(index=y_col, columns=x_col, values=significance_col, fill_value=1)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(heatmap_data, cmap=cmap, center=0, annot=False, fmt=".3f", cbar_kws={"label": value_col}, ax=ax)

    # Add significance markers
    for i, y_val in enumerate(heatmap_data.index):
        for j, x_val in enumerate(heatmap_data.columns):
            if sig_data.loc[y_val, x_val] < significance_threshold:
                ax.text(j + 0.5, i + 0.5, "*", ha="center", va="center", fontsize=12, fontweight="bold", color="white")

    # Customize plot
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=12, fontweight="bold")
    ax.set_ylabel(y_col.replace("_", " ").title(), fontsize=12, fontweight="bold")

    # Rotate labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Add significance legend
    ax.text(
        0.02,
        0.98,
        f"* p < {significance_threshold}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    # Save if requested
    if save_path:
        try:
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"[OK] Heatmap saved to: {save_path}")
        except Exception as e:
            warnings.warn(f"Failed to save heatmap: {e}")

    return fig


def filter_top_associations(
    results_df: pd.DataFrame,
    group_col: str = "archetype",
    ranking_col: str = "log_fold_change",
    top_n: int = 10,
    filter_zero_p: bool = True,
    p_col: str = "pvalue",
) -> pd.DataFrame:
    """
    Filter results to top N associations per group (archetype).

    Selects the highest-ranked associations within each group after
    filtering invalid data. Validates that selected rows have valid
    values for plotting.

    Parameters
    ----------
    results_df : pd.DataFrame
        Input DataFrame with association results.
    group_col : str, default: 'archetype'
        Column to group by.
    ranking_col : str, default: 'log_fold_change'
        Column to rank by. Uses absolute value for fold changes.
    top_n : int, default: 10
        Number of top associations per group.
    filter_zero_p : bool, default: True
        Whether to filter out p == 0 (edge cases).
    p_col : str, default: 'pvalue'
        P-value column name.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with same columns as input, containing
        at most top_n rows per group. Empty DataFrame if no valid
        data remains after filtering.

    Notes
    -----
    Filtering steps:
    1. Remove rows where p_col == 0 (if filter_zero_p=True)
    2. Remove rows with NaN/inf in ranking_col or p_col
    3. Sort by ranking_col (descending, absolute value for fold changes)
    4. Take top_n valid rows per group

    Warns if fewer than top_n valid rows found for any group.
    """
    if results_df.empty:
        return results_df.copy()

    # Handle special case for pattern data with list columns
    if group_col in results_df.columns and results_df[group_col].dtype == object:
        # Check if this column contains lists (unhashable)
        sample_val = results_df[group_col].iloc[0] if len(results_df) > 0 else None
        if isinstance(sample_val, list):
            # Convert lists to strings for grouping
            results_df = results_df.copy()
            results_df[group_col] = results_df[group_col].astype(str)

    filtered_data = []

    for group in results_df[group_col].unique():
        group_data = results_df[results_df[group_col] == group].copy()

        # Filter out p == 0 if requested (edge cases where statistical tests can't break ties)
        if filter_zero_p and p_col in group_data.columns:
            group_data = group_data[group_data[p_col] != 0]

        if group_data.empty:
            continue

        # ENHANCED: Validate data before selecting top N
        # Check for valid values in key columns needed for plotting
        required_cols = [ranking_col, p_col]
        for col in required_cols:
            if col in group_data.columns:
                # Remove rows with NaN, inf, or invalid values in critical columns
                group_data = group_data[
                    group_data[col].notna()
                    & np.isfinite(group_data[col])
                    & (group_data[col] != np.inf)
                    & (group_data[col] != -np.inf)
                ]

        if group_data.empty:
            print(f"[WARNING]  No valid data for group {group} after validation")
            continue

        # Order by ranking column from highest to lowest (for effect size)
        # Use absolute value for effect sizes to get largest magnitude effects
        if "fold_change" in ranking_col.lower():
            sorted_data = group_data.reindex(group_data[ranking_col].abs().sort_values(ascending=False).index)
        else:
            sorted_data = group_data.sort_values(ranking_col, ascending=False)

        # Take up to top_n VALID entries - expand search if we need more valid data
        valid_entries = 0
        selected_rows = []

        for idx, row in sorted_data.iterrows():
            # Double-check this row is valid for plotting
            row_valid = True
            for col in required_cols:
                if col in row and (pd.isna(row[col]) or not np.isfinite(row[col])):
                    row_valid = False
                    break

            if row_valid:
                selected_rows.append(row)
                valid_entries += 1
                if valid_entries >= top_n:
                    break

        if selected_rows:
            group_result = pd.DataFrame(selected_rows)
            filtered_data.append(group_result)
        else:
            print(f"[WARNING]  No valid plottable data found for group {group}")

    if filtered_data:
        result = pd.concat(filtered_data, ignore_index=True)
        if len(filtered_data) > 0:
            total_selected = len(result)
            total_groups = len(filtered_data)
            avg_per_group = total_selected / total_groups if total_groups > 0 else 0
            if avg_per_group < top_n * 0.5:  # Less than half the requested data
                print(
                    f"[STATS] Data quality note: Selected {total_selected} valid entries across {total_groups} groups (avg {avg_per_group:.1f} per group, requested {top_n})"
                )
        return result
    else:
        return pd.DataFrame()


def apply_log_transform_p(results_df: pd.DataFrame, p_col: str, max_log_p: float = 300.0) -> pd.DataFrame:
    """
    Apply -log10 transformation to p-values with capping for extreme values.

    Args:
        results_df: Input DataFrame
        p_col: P-value column name
        max_log_p: Maximum -log10(p-value) to prevent numerical issues

    Returns
    -------
        DataFrame with added log_p column
    """
    results_df = results_df.copy()

    # Validate input column exists
    if p_col not in results_df.columns:
        raise ValueError(f"Column '{p_col}' not found in DataFrame")

    # Calculate -log10(p-value), handling zeros and very small values
    p_values = results_df[p_col].values

    # Handle NaN values first
    nan_mask = np.isnan(p_values)

    # Replace zeros and very small values with very small number to avoid log(0)
    p_values = np.where((p_values <= 0) | (p_values < 1e-300), 1e-300, p_values)

    # Calculate -log10 and cap extreme values
    log_p_values = -np.log10(p_values)
    log_p_values = np.minimum(log_p_values, max_log_p)

    # Restore NaN values where original p-values were NaN
    log_p_values[nan_mask] = np.nan

    # Validate the result
    if np.any(np.isinf(log_p_values[~nan_mask])):
        print(f"[WARNING]  Warning: Infinite values detected in log transformation of {p_col}")
        log_p_values = np.where(np.isinf(log_p_values), max_log_p, log_p_values)

    results_df[f"log_{p_col}"] = log_p_values

    return results_df


def prepare_plotting_data(
    results_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    size_col: str,
    color_col: str,
    filter_zero_p: bool = True,
    p_col: str = "p",
) -> pd.DataFrame:
    """
    Prepare and validate data for plotting.

    Args:
        results_df: Input DataFrame
        x_col, y_col, size_col, color_col: Column names for plotting
        filter_zero_p: Whether to filter zero p-values
        p_col: P-value column name

    Returns
    -------
        Prepared DataFrame

    Raises
    ------
        ValueError: If required columns are missing
    """
    # Validate columns exist
    required_cols = [x_col, y_col, size_col, color_col]
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Copy and filter data
    plot_data = results_df.copy()

    # Filter zero p-values if requested
    if filter_zero_p and p_col in plot_data.columns:
        plot_data = plot_data[plot_data[p_col] != 0]

    # Remove rows with NaN in essential columns
    plot_data = plot_data.dropna(subset=required_cols)

    return plot_data


def visualize_archetypal_space_3d_single(
    adata,
    archetype_coords_key: str = "archetype_coordinates",
    pca_key: str = "X_pca",
    color_by: str | None = None,
    use_layer: str | None = "logcounts",  # NEW: Default to logcounts for gene expression
    cell_size: float = 2.0,
    cell_opacity: float = 0.6,
    archetype_size: float = 8.0,
    archetype_color: str = "red",
    show_archetype_labels: bool = True,
    show_connections: bool = True,
    color_scale: str = "viridis",
    categorical_colors: dict[str, str] | None = None,
    title: str = "Archetypal Space Visualization",
    auto_scale: bool = True,
    fixed_ranges: dict[str, tuple[float, float]] | None = None,
    save_path: str | Path | None = None,
    legend_marker_scale: float = 1.0,
    legend_font_size: int = 12,
    # Conditional centroid parameters
    show_centroids: bool = False,
    centroid_condition: str | None = None,
    centroid_order: list[str] | None = None,
    centroid_groupby: str | None = None,
    centroid_size: float = 20.0,
    centroid_start_symbol: str = "circle",
    centroid_end_symbol: str = "diamond",
    centroid_line_width: float = 6.0,
    centroid_colors: dict[str, str] | None = None,
) -> go.Figure:
    """
    Visualize a single archetypal analysis result in 3D PCA space.

    Creates an interactive 3D scatter plot showing cells in PCA space with archetype
    positions, colored by categorical variables or continuous gene expression.

    Args:
        adata: AnnData object with PCA coordinates and archetype results
        archetype_coords_key: Key in adata.uns containing archetype coordinates [n_archetypes, n_pca_components]
        pca_key: Key in adata.obsm containing PCA coordinates [n_cells, n_pca_components]
        color_by: Column in adata.obs to color cells by (categorical or continuous) OR gene name from adata.var.index
        use_layer: AnnData layer to use for gene expression (None=adata.X, 'logcounts'=default for genes, 'raw', etc.)
        cell_size: Size of cell points
        cell_opacity: Opacity of cell points (0-1)
        archetype_size: Size of archetype markers
        archetype_color: Color for archetype markers
        show_archetype_labels: Whether to show archetype labels
        show_connections: Whether to draw lines connecting archetypes
        color_scale: Plotly color scale for continuous variables
        categorical_colors: Custom colors for categorical variables {category: color}
        title: Plot title
        auto_scale: Whether to auto-scale axes based on data
        fixed_ranges: Fixed axis ranges {'x': (min, max), 'y': (min, max), 'z': (min, max)}
        save_path: Optional path to save HTML file
        legend_marker_scale: Scale factor for legend marker sizes (e.g., 2.0 for double size)
        legend_font_size: Font size for legend text

    Returns
    -------
        plotly Figure object

    Raises
    ------
        ImportError: If plotly is not available
        ValueError: If required data is missing from AnnData

    Examples
    --------
        # Basic usage with categorical coloring
        fig = visualize_archetypal_space_3d_single(
            adata,
            color_by='anatomical_location',
            title='Archetypes by Anatomical Location'
        )
        fig.show()

        # Gene expression gradient (uses logcounts by default)
        fig = visualize_archetypal_space_3d_single(
            adata,
            color_by='CD8A',  # Gene name - automatically uses logcounts layer
            color_scale='plasma',
            title='CD8A Expression in Archetypal Space'
        )

        # Use raw counts instead of logcounts
        fig = visualize_archetypal_space_3d_single(
            adata,
            color_by='CD8A',
            use_layer=None,  # Use adata.X instead of logcounts
            color_scale='plasma'
        )
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for 3D visualizations. Install with: pip install plotly")

    # Validate inputs
    if pca_key not in adata.obsm:
        raise ValueError(f"PCA coordinates not found in adata.obsm['{pca_key}']")

    if archetype_coords_key not in adata.uns:
        raise ValueError(f"Archetype coordinates not found in adata.uns['{archetype_coords_key}']")

    # Get data
    pca_coords = adata.obsm[pca_key][:, :3]  # First 3 PCA components
    archetype_coords = adata.uns[archetype_coords_key][:, :3]  # First 3 components

    # Determine axis ranges
    if fixed_ranges:
        x_range = fixed_ranges.get("x", [pca_coords[:, 0].min(), pca_coords[:, 0].max()])
        y_range = fixed_ranges.get("y", [pca_coords[:, 1].min(), pca_coords[:, 1].max()])
        z_range = fixed_ranges.get("z", [pca_coords[:, 2].min(), pca_coords[:, 2].max()])
    elif auto_scale:
        # Use 1st and 99th percentiles with margin
        def get_axis_range(coords, axis_idx, margin_factor=0.75):
            percentiles = np.percentile(coords[:, axis_idx], [1, 99])
            margin = (percentiles[1] - percentiles[0]) * margin_factor
            return [percentiles[0] - margin, percentiles[1] + margin]

        x_range = get_axis_range(pca_coords, 0)
        y_range = get_axis_range(pca_coords, 1)
        z_range = get_axis_range(pca_coords, 2)
    else:
        # Use full data range
        x_range = [pca_coords[:, 0].min(), pca_coords[:, 0].max()]
        y_range = [pca_coords[:, 1].min(), pca_coords[:, 1].max()]
        z_range = [pca_coords[:, 2].min(), pca_coords[:, 2].max()]

    # Initialize figure
    fig = go.Figure()

    # Handle cell coloring and track colorbar presence
    has_colorbar = False
    if color_by is not None:
        if color_by in adata.obs.columns:
            # Metadata column
            color_values = adata.obs[color_by]
            # Check if continuous (will have colorbar)
            if pd.api.types.is_numeric_dtype(color_values):
                has_colorbar = True
        elif color_by in adata.var.index:
            # Gene expression - always continuous, always has colorbar
            gene_idx = adata.var.index.get_loc(color_by)

            # SMART LAYER SELECTION for gene expression
            if use_layer is not None and use_layer in adata.layers:
                # Use specified layer (default: logcounts)
                layer_data = adata.layers[use_layer]
                if hasattr(layer_data, "toarray"):  # Sparse matrix
                    color_values = layer_data[:, gene_idx].toarray().flatten()
                else:
                    color_values = layer_data[:, gene_idx]
            elif use_layer == "logcounts" and "logcounts" not in adata.layers:
                # Fallback: logcounts requested but not available, warn and use X
                warnings.warn(
                    f"Layer 'logcounts' not found in adata.layers. Using adata.X instead. "
                    f"Available layers: {list(adata.layers.keys())}"
                )
                if hasattr(adata.X, "toarray"):
                    color_values = adata.X[:, gene_idx].toarray().flatten()
                else:
                    color_values = adata.X[:, gene_idx]
            else:
                # Use adata.X (when use_layer=None explicitly specified)
                if hasattr(adata.X, "toarray"):
                    color_values = adata.X[:, gene_idx].toarray().flatten()
                else:
                    color_values = adata.X[:, gene_idx]

            has_colorbar = True  # Gene expression always gets colorbar
        else:
            warnings.warn(f"'{color_by}' not found in adata.obs or adata.var. Using default coloring.")
            color_values = None
    else:
        color_values = None

    # Add cells to plot
    if color_values is not None:
        if pd.api.types.is_numeric_dtype(color_values):
            # Continuous coloring
            fig.add_trace(
                go.Scatter3d(
                    x=pca_coords[:, 0],
                    y=pca_coords[:, 1],
                    z=pca_coords[:, 2],
                    mode="markers",
                    marker=dict(
                        size=cell_size,
                        color=color_values,
                        colorscale=color_scale,
                        opacity=cell_opacity,
                        colorbar=dict(
                            title=color_by,
                            x=1.02,  # Position colorbar to avoid overlap with plot
                            thickness=15,  # Make colorbar slightly thinner
                            len=0.8,  # Make colorbar shorter to leave room for legend
                        ),
                    ),
                    name="Cells",
                    hovertemplate=f"<b>PC1:</b> %{{x:.2f}}<br><b>PC2:</b> %{{y:.2f}}<br><b>PC3:</b> %{{z:.2f}}<br><b>{color_by}:</b> %{{marker.color:.2f}}<extra></extra>",
                )
            )
        else:
            # Categorical coloring
            color_values = pd.Categorical(color_values)
            categories = color_values.categories

            # Set up colors
            if categorical_colors is None:
                if len(categories) <= 10:
                    colors = px.colors.qualitative.Set1[: len(categories)]
                else:
                    colors = px.colors.sample_colorscale("hsv", len(categories))
            else:
                colors = [
                    categorical_colors.get(
                        cat, f"rgb({np.random.randint(0, 255)},{np.random.randint(0, 255)},{np.random.randint(0, 255)})"
                    )
                    for cat in categories
                ]

            # Add each category as separate trace for proper legend
            for i, category in enumerate(categories):
                mask = color_values == category
                if mask.sum() > 0:
                    fig.add_trace(
                        go.Scatter3d(
                            x=pca_coords[mask, 0],
                            y=pca_coords[mask, 1],
                            z=pca_coords[mask, 2],
                            mode="markers",
                            marker=dict(
                                size=cell_size,
                                color=colors[i],
                                opacity=cell_opacity,
                                line=dict(width=0),  # Remove marker outlines for cleaner look
                            ),
                            name=f"{category}",
                            legendgroup="cells",
                            hovertemplate=f"<b>PC1:</b> %{{x:.2f}}<br><b>PC2:</b> %{{y:.2f}}<br><b>PC3:</b> %{{z:.2f}}<br><b>{color_by}:</b> {category}<extra></extra>",
                        )
                    )
    else:
        # Default coloring
        fig.add_trace(
            go.Scatter3d(
                x=pca_coords[:, 0],
                y=pca_coords[:, 1],
                z=pca_coords[:, 2],
                mode="markers",
                marker=dict(size=cell_size, color="lightblue", opacity=cell_opacity),
                name="Cells",
                hovertemplate="<b>PC1:</b> %{x:.2f}<br><b>PC2:</b> %{y:.2f}<br><b>PC3:</b> %{z:.2f}<extra></extra>",
            )
        )

    # Add archetypes
    n_archetypes = archetype_coords.shape[0]

    if show_archetype_labels:
        # Show markers with text labels
        fig.add_trace(
            go.Scatter3d(
                x=archetype_coords[:, 0],
                y=archetype_coords[:, 1],
                z=archetype_coords[:, 2],
                mode="markers+text",
                marker=dict(size=archetype_size, color=archetype_color, symbol="diamond"),
                text=[f"Arch{i + 1}" for i in range(n_archetypes)],
                textposition="top center",
                textfont=dict(size=12, color="black"),
                name="Archetypes",
                hovertemplate="<b>Archetype %{text}</b><br><b>PC1:</b> %{x:.2f}<br><b>PC2:</b> %{y:.2f}<br><b>PC3:</b> %{z:.2f}<extra></extra>",
            )
        )
    else:
        # Show only markers
        fig.add_trace(
            go.Scatter3d(
                x=archetype_coords[:, 0],
                y=archetype_coords[:, 1],
                z=archetype_coords[:, 2],
                mode="markers",
                marker=dict(size=archetype_size, color=archetype_color, symbol="diamond"),
                name="Archetypes",
                hovertemplate="<b>Archetype %{pointNumber}</b><br><b>PC1:</b> %{x:.2f}<br><b>PC2:</b> %{y:.2f}<br><b>PC3:</b> %{z:.2f}<extra></extra>",
            )
        )

    # Add connections between archetypes
    if show_connections and n_archetypes > 1:
        for i in range(n_archetypes):
            for j in range(i + 1, n_archetypes):
                fig.add_trace(
                    go.Scatter3d(
                        x=[archetype_coords[i, 0], archetype_coords[j, 0]],
                        y=[archetype_coords[i, 1], archetype_coords[j, 1]],
                        z=[archetype_coords[i, 2], archetype_coords[j, 2]],
                        mode="lines",
                        line=dict(color=archetype_color, width=2),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    # Add conditional centroids with trajectory lines
    if show_centroids and centroid_condition is not None:
        centroid_key = "conditional_centroids"
        if centroid_key not in adata.uns or centroid_condition not in adata.uns[centroid_key]:
            warnings.warn(
                f"Centroids for '{centroid_condition}' not found in adata.uns['{centroid_key}']. "
                f"Run pc.tl.compute_conditional_centroids(adata, '{centroid_condition}') first."
            )
        else:
            centroid_data = adata.uns[centroid_key][centroid_condition]

            # Determine trajectory order
            if centroid_order is not None:
                # Validate order contains valid levels
                available_levels = set(centroid_data["centroids_3d"].keys())
                invalid_levels = set(centroid_order) - available_levels
                if invalid_levels:
                    warnings.warn(f"Invalid levels in centroid_order: {invalid_levels}. Available: {available_levels}")
                display_order = [l for l in centroid_order if l in available_levels]
            else:
                display_order = centroid_data["levels"]

            # Set default colors
            if centroid_colors is None:
                centroid_colors = {}

            # Check if using groupby
            if centroid_groupby is not None and centroid_data.get("group_centroids_3d") is not None:
                # Multi-group trajectories
                group_colors = ["magenta", "cyan", "lime", "orange", "purple", "yellow"]
                groups = list(centroid_data["group_centroids_3d"].keys())

                for g_idx, group_name in enumerate(groups):
                    group_centroids_3d = centroid_data["group_centroids_3d"].get(group_name, {})
                    group_counts = centroid_data.get("group_cell_counts", {}).get(group_name, {})

                    # Get ordered coordinates for this group
                    ordered_coords = []
                    ordered_levels = []
                    for level in display_order:
                        if level in group_centroids_3d:
                            ordered_coords.append(group_centroids_3d[level])
                            ordered_levels.append(level)

                    if len(ordered_coords) < 1:
                        continue

                    # Get color for this group
                    color = centroid_colors.get(group_name, group_colors[g_idx % len(group_colors)])

                    # Build symbol list: first = start, last = end, middle = circle
                    symbols = []
                    for i, level in enumerate(ordered_levels):
                        if i == 0:
                            symbols.append(centroid_start_symbol)
                        elif i == len(ordered_levels) - 1:
                            symbols.append(centroid_end_symbol)
                        else:
                            symbols.append("circle")

                    # Build hover text
                    hover_texts = []
                    for level in ordered_levels:
                        count = group_counts.get(level, 0)
                        coords = group_centroids_3d[level]
                        hover_texts.append(
                            f"<b>{group_name} - {level}</b><br>"
                            f"<b>Cells:</b> {count}<br>"
                            f"<b>PC1:</b> {coords[0]:.3f}<br>"
                            f"<b>PC2:</b> {coords[1]:.3f}<br>"
                            f"<b>PC3:</b> {coords[2]:.3f}<extra></extra>"
                        )

                    # Add combined lines+markers trace (R template pattern)
                    fig.add_trace(
                        go.Scatter3d(
                            x=[c[0] for c in ordered_coords],
                            y=[c[1] for c in ordered_coords],
                            z=[c[2] for c in ordered_coords],
                            mode="lines+markers",
                            line=dict(color=color, width=centroid_line_width),
                            marker=dict(
                                size=centroid_size,
                                color=color,
                                symbol=symbols,
                                opacity=1,
                                line=dict(color="black", width=2),
                            ),
                            name=f"{group_name} ({ordered_levels[0]}→{ordered_levels[-1]})",
                            legendgroup="centroids",
                            hoverinfo="text",
                            hovertext=hover_texts,
                        )
                    )
            else:
                # Single trajectory (no groupby)
                centroids_3d = centroid_data["centroids_3d"]
                cell_counts = centroid_data["cell_counts"]

                # Get ordered coordinates
                ordered_coords = []
                ordered_levels = []
                for level in display_order:
                    if level in centroids_3d:
                        ordered_coords.append(centroids_3d[level])
                        ordered_levels.append(level)

                if len(ordered_coords) >= 1:
                    # Get color
                    color = centroid_colors.get("default", "magenta")

                    # Build symbol list
                    symbols = []
                    for i, level in enumerate(ordered_levels):
                        if i == 0:
                            symbols.append(centroid_start_symbol)
                        elif i == len(ordered_levels) - 1:
                            symbols.append(centroid_end_symbol)
                        else:
                            symbols.append("circle")

                    # Build hover text
                    hover_texts = []
                    for level in ordered_levels:
                        count = cell_counts.get(level, 0)
                        coords = centroids_3d[level]
                        hover_texts.append(
                            f"<b>{centroid_condition}: {level}</b><br>"
                            f"<b>Cells:</b> {count}<br>"
                            f"<b>PC1:</b> {coords[0]:.3f}<br>"
                            f"<b>PC2:</b> {coords[1]:.3f}<br>"
                            f"<b>PC3:</b> {coords[2]:.3f}<extra></extra>"
                        )

                    # Add combined lines+markers trace
                    fig.add_trace(
                        go.Scatter3d(
                            x=[c[0] for c in ordered_coords],
                            y=[c[1] for c in ordered_coords],
                            z=[c[2] for c in ordered_coords],
                            mode="lines+markers",
                            line=dict(color=color, width=centroid_line_width),
                            marker=dict(
                                size=centroid_size,
                                color=color,
                                symbol=symbols,
                                opacity=1,
                                line=dict(color="black", width=2),
                            ),
                            name=f"{centroid_condition} ({ordered_levels[0]}→{ordered_levels[-1]})",
                            legendgroup="centroids",
                            hoverinfo="text",
                            hovertext=hover_texts,
                        )
                    )

    # Update layout with legend configuration
    # Note: itemsizing='constant' helps make legend markers more visible
    # The legend_marker_scale is applied through trace marker size
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode="cube",
            xaxis=dict(title="PC1", range=x_range),
            yaxis=dict(title="PC2", range=y_range),
            zaxis=dict(title="PC3", range=z_range),
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.25 if has_colorbar else 1.02,  # SMART: Move legend right only when colorbar present
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.5)",
            borderwidth=1,
            itemsizing="constant",  # Use constant sizing for legend items
            itemwidth=30 * legend_marker_scale,  # Scale the legend item width
            font=dict(size=legend_font_size),
            tracegroupgap=5,
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    # Save if requested
    if save_path:
        try:
            fig.write_html(save_path)
            print(f"[OK] 3D visualization saved to: {save_path}")
        except Exception as e:
            warnings.warn(f"Failed to save figure: {e}")

    return fig


def visualize_archetypal_space_3d_multi(
    adata_list: list,
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
    """
    Visualize multiple archetypal analysis results in 3D PCA space for comparison.

    Creates an interactive 3D scatter plot comparing multiple archetypal fits,
    useful for comparing different conditions, treatments, or parameter settings.

    Args:
        adata_list: List of AnnData objects with PCA coordinates and archetype results
        archetype_coords_key: Key in adata.uns containing archetype coordinates
        pca_key: Key in adata.obsm containing PCA coordinates
        labels_list: Labels for each dataset (defaults to 'Set 1', 'Set 2', etc.)
        color_by: Column(s) to color cells by - single string or list per dataset
        color_values: Direct color values - single array or list per dataset
        cell_size: Size of cell points
        cell_opacity: Opacity of cell points (0-1)
        archetype_size: Size of archetype markers
        archetype_colors: Colors for archetype markers per dataset
        show_labels: Which datasets to show archetype labels for (bool, list of indices)
        auto_scale: Whether to auto-scale axes based on all data
        range_reference: Reference dataset index or AnnData for axis scaling
        fixed_ranges: Fixed axis ranges {'x': (min, max), 'y': (min, max), 'z': (min, max)}
        color_scale: Plotly color scale for continuous variables
        categorical_colors: Custom colors for categorical variables
        title: Plot title
        save_path: Optional path to save HTML file

    Returns
    -------
        plotly Figure object

    Examples
    --------
        # Compare treatment conditions
        fig = visualize_archetypal_space_3d_multi(
            adata_list=[adata_control, adata_treated],
            labels_list=['Control', 'Treated'],
            color_by=['cell_type', 'cell_type'],
            title='Treatment Effect on Archetypal Space'
        )

        # Compare different archetype numbers
        fig = visualize_archetypal_space_3d_multi(
            adata_list=[adata_k3, adata_k5, adata_k7],
            labels_list=['K=3', 'K=5', 'K=7'],
            show_labels=[2],  # Only show labels for K=7
            title='Archetype Number Comparison'
        )
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for 3D visualizations. Install with: pip install plotly")

    # Handle single AnnData input
    if not isinstance(adata_list, list):
        adata_list = [adata_list]

    n_datasets = len(adata_list)

    # Default parameters
    if labels_list is None:
        labels_list = [f"Set {i + 1}" for i in range(n_datasets)]

    if archetype_colors is None:
        archetype_colors = ["red", "blue", "green", "purple", "orange", "cyan"]

    # Process show_labels parameter
    if isinstance(show_labels, bool):
        if show_labels:
            show_labels = list(range(n_datasets))
        else:
            show_labels = [0]  # Show only first
    elif not isinstance(show_labels, list):
        show_labels = [0]

    # Extract PCA coordinates from all datasets
    pca_coords_list = []
    archetype_coords_list = []

    for i, adata in enumerate(adata_list):
        if pca_key not in adata.obsm:
            raise ValueError(f"PCA coordinates not found in adata_list[{i}].obsm['{pca_key}']")
        if archetype_coords_key not in adata.uns:
            raise ValueError(f"Archetype coordinates not found in adata_list[{i}].uns['{archetype_coords_key}']")

        pca_coords_list.append(adata.obsm[pca_key][:, :3])
        archetype_coords_list.append(adata.uns[archetype_coords_key][:, :3])

    # Determine axis ranges
    def get_axis_range(coords, axis_idx, margin_factor=0.75):
        percentiles = np.percentile(coords[:, axis_idx], [1, 99])
        margin = (percentiles[1] - percentiles[0]) * margin_factor
        return [percentiles[0] - margin, percentiles[1] + margin]

    if fixed_ranges:
        x_range = fixed_ranges.get("x")
        y_range = fixed_ranges.get("y")
        z_range = fixed_ranges.get("z")
    elif range_reference is not None:
        if isinstance(range_reference, int) and 0 <= range_reference < n_datasets:
            ref_coords = pca_coords_list[range_reference]
        else:
            # Assume it's an AnnData object
            ref_coords = range_reference.obsm[pca_key][:, :3]

        x_range = get_axis_range(ref_coords, 0)
        y_range = get_axis_range(ref_coords, 1)
        z_range = get_axis_range(ref_coords, 2)
    elif auto_scale:
        # Use all coordinates
        all_coords = np.vstack(pca_coords_list)
        x_range = get_axis_range(all_coords, 0)
        y_range = get_axis_range(all_coords, 1)
        z_range = get_axis_range(all_coords, 2)
    else:
        # Use first dataset
        x_range = get_axis_range(pca_coords_list[0], 0)
        y_range = get_axis_range(pca_coords_list[0], 1)
        z_range = get_axis_range(pca_coords_list[0], 2)

    # Initialize figure
    fig = go.Figure()

    # Process each dataset
    for i, adata in enumerate(adata_list):
        pca_coords = pca_coords_list[i]
        dataset_label = labels_list[i] if i < len(labels_list) else f"Set {i + 1}"

        # Handle coloring
        this_color_by = color_by
        if isinstance(color_by, list) and i < len(color_by):
            this_color_by = color_by[i]

        this_color_values = None
        if color_values is not None:
            if isinstance(color_values, list) and i < len(color_values):
                this_color_values = color_values[i]
            else:
                this_color_values = color_values

        # Get color values
        if this_color_values is not None:
            color_vals = this_color_values
        elif this_color_by is not None:
            if this_color_by in adata.obs.columns:
                color_vals = adata.obs[this_color_by]
            elif this_color_by in adata.var.index:
                gene_idx = adata.var.index.get_loc(this_color_by)
                if hasattr(adata.X, "toarray"):
                    color_vals = adata.X[:, gene_idx].toarray().flatten()
                else:
                    color_vals = adata.X[:, gene_idx]
            else:
                color_vals = None
        else:
            color_vals = None

        # Add cells
        if color_vals is not None and pd.api.types.is_numeric_dtype(color_vals):
            # Continuous coloring
            fig.add_trace(
                go.Scatter3d(
                    x=pca_coords[:, 0],
                    y=pca_coords[:, 1],
                    z=pca_coords[:, 2],
                    mode="markers",
                    marker=dict(
                        size=cell_size,
                        color=color_vals,
                        colorscale=color_scale,
                        opacity=cell_opacity,
                        colorbar=dict(title=this_color_by or "Value"),
                    ),
                    name=f"Cells - {dataset_label}",
                    legendgroup=f"cells_{i}",
                    hovertemplate=f"<b>{dataset_label}</b><br><b>PC1:</b> %{{x:.2f}}<br><b>PC2:</b> %{{y:.2f}}<br><b>PC3:</b> %{{z:.2f}}<extra></extra>",
                )
            )
        elif color_vals is not None:
            # Categorical coloring
            color_vals = pd.Categorical(color_vals)
            categories = color_vals.categories

            # Set up colors
            if categorical_colors is None:
                if len(categories) <= 10:
                    colors = px.colors.qualitative.Set1[: len(categories)]
                else:
                    colors = px.colors.sample_colorscale("hsv", len(categories))
            else:
                colors = [
                    categorical_colors.get(
                        cat, f"rgb({np.random.randint(0, 255)},{np.random.randint(0, 255)},{np.random.randint(0, 255)})"
                    )
                    for cat in categories
                ]

            # Add each category as separate trace
            for j, category in enumerate(categories):
                mask = color_vals == category
                if mask.sum() > 0:
                    fig.add_trace(
                        go.Scatter3d(
                            x=pca_coords[mask, 0],
                            y=pca_coords[mask, 1],
                            z=pca_coords[mask, 2],
                            mode="markers",
                            marker=dict(size=cell_size, color=colors[j], opacity=cell_opacity),
                            name=f"{category} - {dataset_label}",
                            legendgroup=f"cells_{i}",
                            hovertemplate=f"<b>{dataset_label}</b><br><b>{category}</b><br><b>PC1:</b> %{{x:.2f}}<br><b>PC2:</b> %{{y:.2f}}<br><b>PC3:</b> %{{z:.2f}}<extra></extra>",
                        )
                    )
        else:
            # Default coloring
            default_colors = ["orange", "skyblue", "lightgreen", "pink", "gold", "lightcyan"]
            default_color = default_colors[i % len(default_colors)]

            fig.add_trace(
                go.Scatter3d(
                    x=pca_coords[:, 0],
                    y=pca_coords[:, 1],
                    z=pca_coords[:, 2],
                    mode="markers",
                    marker=dict(size=cell_size, color=default_color, opacity=cell_opacity),
                    name=f"Cells - {dataset_label}",
                    legendgroup=f"cells_{i}",
                    hovertemplate=f"<b>{dataset_label}</b><br><b>PC1:</b> %{{x:.2f}}<br><b>PC2:</b> %{{y:.2f}}<br><b>PC3:</b> %{{z:.2f}}<extra></extra>",
                )
            )

    # Add archetypes for each dataset
    for i, archetype_coords in enumerate(archetype_coords_list):
        current_color = archetype_colors[i % len(archetype_colors)]
        dataset_label = labels_list[i] if i < len(labels_list) else f"Set {i + 1}"
        should_show_labels = i in show_labels

        n_archetypes = archetype_coords.shape[0]

        if should_show_labels:
            # Show markers with text
            fig.add_trace(
                go.Scatter3d(
                    x=archetype_coords[:, 0],
                    y=archetype_coords[:, 1],
                    z=archetype_coords[:, 2],
                    mode="markers+text",
                    marker=dict(size=archetype_size, color=current_color, symbol="diamond"),
                    text=[f"{dataset_label}-Arch{j + 1}" for j in range(n_archetypes)],
                    textposition="top center",
                    textfont=dict(size=12, color="black"),
                    name=f"Archetypes - {dataset_label}",
                    hovertemplate="<b>%{text}</b><br><b>PC1:</b> %{x:.2f}<br><b>PC2:</b> %{y:.2f}<br><b>PC3:</b> %{z:.2f}<extra></extra>",
                )
            )
        else:
            # Show only markers
            fig.add_trace(
                go.Scatter3d(
                    x=archetype_coords[:, 0],
                    y=archetype_coords[:, 1],
                    z=archetype_coords[:, 2],
                    mode="markers",
                    marker=dict(size=archetype_size, color=current_color, symbol="diamond"),
                    name=f"Archetypes - {dataset_label}",
                    hovertemplate=f"<b>{dataset_label} Archetype %{{pointNumber}}</b><br><b>PC1:</b> %{{x:.2f}}<br><b>PC2:</b> %{{y:.2f}}<br><b>PC3:</b> %{{z:.2f}}<extra></extra>",
                )
            )

        # Add connections between archetypes
        if n_archetypes > 1:
            for a1 in range(n_archetypes):
                for a2 in range(a1 + 1, n_archetypes):
                    fig.add_trace(
                        go.Scatter3d(
                            x=[archetype_coords[a1, 0], archetype_coords[a2, 0]],
                            y=[archetype_coords[a1, 1], archetype_coords[a2, 1]],
                            z=[archetype_coords[a1, 2], archetype_coords[a2, 2]],
                            mode="lines",
                            line=dict(color=current_color, width=2),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode="cube",
            xaxis=dict(title="PC1", range=x_range),
            yaxis=dict(title="PC2", range=y_range),
            zaxis=dict(title="PC3", range=z_range),
        ),
        legend=dict(
            itemsizing="constant",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            font=dict(size=10),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.5)",
            borderwidth=1,
            tracegroupgap=5,
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    # Save if requested
    if save_path:
        try:
            fig.write_html(save_path)
            print(f"[OK] Multi-dataset 3D visualization saved to: {save_path}")
        except Exception as e:
            warnings.warn(f"Failed to save figure: {e}")

    return fig


# Example usage and testing functions
def create_example_data() -> pd.DataFrame:
    """Create example data for testing visualization functions."""
    np.random.seed(42)

    archetypes = ["archetype_1", "archetype_2", "archetype_3", "archetype_4"]
    genes = [f"gene_{i}" for i in range(1, 51)]

    data = []
    for arch in archetypes:
        for gene in np.random.choice(genes, 12, replace=False):
            # Create data with both Python and R-style column names for compatibility
            mean_arch = np.random.uniform(0.1, 2.0)
            log_fc = np.random.normal(0, 1)
            pval = np.random.exponential(0.01)

            data.append(
                {
                    # Python workflow columns (harmonized structure)
                    "archetype": arch,
                    "gene": gene,
                    "pvalue": pval,
                    "log_fold_change": log_fc,
                    "mean_archetype": mean_arch,
                    "mean_other": mean_arch - log_fc,
                    "significant": pval < 0.05,
                    "fdr_pvalue": pval * 1.2,  # Mock FDR correction
                    "statistic": np.random.uniform(1000, 5000),
                    "n_archetype_cells": np.random.randint(50, 200),
                    "n_other_cells": np.random.randint(500, 1000),
                }
            )

    return pd.DataFrame(data)


def create_example_3d_usage():
    """
    Example usage of the 3D visualization functions.

    This demonstrates how to use the single archetype visualization
    with categorical coloring for anatomical location.
    """
    print("Example: 3D Archetypal Space Visualization")
    print("=" * 50)

    if not PLOTLY_AVAILABLE:
        print("[ERROR] Plotly not available. Install with: pip install plotly")
        return

    print("""
# Example 1: Single Archetype Fit with Categorical Coloring
# ========================================================

# Assuming you have an AnnData object 'adata' with:
# - adata.obsm['X_pca']: PCA coordinates [n_cells, n_components]
# - adata.uns['archetype_coordinates']: Archetype positions [n_archetypes, n_components]  
# - adata.obs['anatomical_location']: Categorical metadata

# Example import (function is defined in this same file)

# Basic usage with anatomical location coloring
fig = visualize_archetypal_space_3d_single(
    adata,
    color_by='anatomical_location',
    title='Archetypes by Anatomical Location',
    archetype_color='red',
    cell_size=2.0,
    archetype_size=10.0,
    show_archetype_labels=True,
    show_connections=True
)

# Display the interactive plot
fig.show()

# Save as HTML file
fig.write_html('archetypal_space_anatomical.html')

# Example 2: Gene Expression Gradient
# ===================================

# Color by gene expression (continuous)
fig_gene = visualize_archetypal_space_3d_single(
    adata,
    color_by='CD8A',  # Gene name in adata.var.index
    color_scale='plasma',
    title='CD8A Expression in Archetypal Space'
)
fig_gene.show()

# Example 3: Multi-Dataset Comparison
# ===================================

# Compare different conditions or treatments
fig_multi = visualize_archetypal_space_3d_multi(
    adata_list=[adata_control, adata_treated],
    labels_list=['Control', 'Treated'],
    color_by=['cell_type', 'cell_type'],  # Color both by cell type
    title='Treatment Effect on Archetypal Space',
    archetype_colors=['red', 'blue'],
    show_labels=[0, 1]  # Show labels for both datasets
)
fig_multi.show()

# Example 4: Archetype Number Comparison
# ======================================

# Compare different numbers of archetypes
fig_k_compare = visualize_archetypal_space_3d_multi(
    adata_list=[adata_k3, adata_k5, adata_k7],
    labels_list=['K=3', 'K=5', 'K=7'],
    show_labels=[2],  # Only show labels for K=7
    title='Archetype Number Comparison',
    auto_scale=True
)
fig_k_compare.show()

# Custom color schemes for categorical data
custom_colors = {
    'Peritoneum': '#FF6B6B',
    'Omentum': '#4ECDC4', 
    'Mesentery': '#45B7D1',
    'Tumor': '#96CEB4'
}

fig_custom = visualize_archetypal_space_3d_single(
    adata,
    color_by='anatomical_location',
    categorical_colors=custom_colors,
    title='Custom Colored Anatomical Locations'
)
fig_custom.show()
""")


if __name__ == "__main__":
    # Test the visualization functions
    print("Testing results visualization functions...")

    # Create example data for 2D plots
    test_data = create_example_data()
    print(f"Created test data: {test_data.shape}")

    # Test dotplot
    try:
        fig1 = create_dotplot_visualization(test_data, title="Test Gene-Archetype Associations", top_n_per_group=8)
        print("[OK] Dotplot visualization test passed")
        plt.show()
    except Exception as e:
        print(f"[ERROR] Dotplot test failed: {e}")

    # Test heatmap
    try:
        fig2 = create_heatmap_visualization(test_data, title="Test Association Heatmap", top_n_per_group=10)
        print("[OK] Heatmap visualization test passed")
        plt.show()
    except Exception as e:
        print(f"[ERROR] Heatmap test failed: {e}")

    # Show 3D visualization examples
    create_example_3d_usage()

    print("Visualization testing completed!")


# =============================================================================
# TRAINING RESULTS VISUALIZATION FUNCTIONS
# =============================================================================


def visualize_training_results(
    training_results: dict,
    title: str = "Model Training Progress",
    figsize: tuple[int, int] = (12, 8),
    show_final_analysis: bool = True,
) -> matplotlib.figure.Figure:
    """
    Create comprehensive training visualization with 4 subplots.

    Parameters
    ----------
    training_results : dict
        Dictionary from train_vae() containing 'history' key and
        optionally 'final_analysis' key.
    title : str, default: 'Model Training Progress'
        Main title for the figure.
    figsize : tuple[int, int], default: (12, 8)
        Figure size as (width, height).
    show_final_analysis : bool, default: True
        Whether to print final analysis summary to console.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with 2x2 subplot layout:
        - [0,0] Loss Components: loss, archetypal_loss, kld_loss
        - [0,1] Performance Metrics: archetype_r2, rmse
        - [1,0] Archetype Stability: drift_mean, stability_mean, variance_mean
        - [1,1] Constraint Compliance: violation_rate, A/B_sum_error

    Raises
    ------
    ValueError
        If training_results doesn't contain required 'history' key.

    Notes
    -----
    Panels display placeholder text if corresponding metrics are not
    available in history (e.g., stability metrics require multi-epoch
    training with track_stability=True).

    Examples
    --------
    >>> results = pc.tl.train_archetypal(adata, n_archetypes=5)
    >>> fig = visualize_training_results(results)
    >>> plt.show()
    """
    if "history" not in training_results:
        raise ValueError("training_results must contain 'history' key from train_vae()")

    history = training_results["history"]

    # Create subplot layout
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)

    # Plot 1: Loss components
    ax1 = axes[0, 0]
    if "loss" in history:
        ax1.plot(history["loss"], label="Total Loss", color="red", linewidth=2)
    if "archetypal_loss" in history:
        ax1.plot(history["archetypal_loss"], label="Archetypal Loss", color="blue", linewidth=2)
    if "kld_loss" in history:
        ax1.plot(history["kld_loss"], label="KLD Loss", color="green", linewidth=2)

    ax1.set_title("Loss Components")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Performance metrics
    ax2 = axes[0, 1]
    if "archetype_r2" in history:
        ax2.plot(history["archetype_r2"], label="Archetype R²", color="purple", linewidth=2)
    if "rmse" in history:
        ax2.plot(history["rmse"], label="RMSE", color="orange", linewidth=2)

    ax2.set_title("Performance Metrics")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("R² / RMSE")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Stability metrics (if available)
    ax3 = axes[1, 0]
    stability_metrics = ["archetype_drift_mean", "archetype_stability_mean", "archetype_variance_mean"]
    available_stability = [m for m in stability_metrics if m in history and len(history[m]) > 0]

    stability_plotted = False
    colors = ["darkorange", "darkgreen", "darkblue"]
    for i, metric in enumerate(available_stability):
        if len(history[metric]) > 1:
            color = colors[i % len(colors)]
            ax3.plot(
                history[metric],
                label=metric.replace("archetype_", "").replace("_", " ").title(),
                color=color,
                linewidth=2,
            )
            stability_plotted = True

    if stability_plotted:
        ax3.set_title("Archetype Stability")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Stability Metric")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(
            0.5,
            0.5,
            "No stability metrics\navailable\n(requires multi-epoch training)",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=10,
        )
        ax3.set_title("Archetype Stability (N/A)")

    # Plot 4: Constraint violations (if available)
    ax4 = axes[1, 1]
    constraint_metrics = ["constraint_violation_rate", "A_sum_error", "B_sum_error"]
    available_constraints = [m for m in constraint_metrics if m in history and len(history[m]) > 0]

    if available_constraints:
        colors = ["red", "blue", "green"]
        for i, metric in enumerate(available_constraints[:3]):  # Plot at most 3 to avoid clutter
            color = colors[i % len(colors)]
            ax4.plot(history[metric], label=metric.replace("_", " ").title(), color=color, linewidth=2)
        ax4.set_title("Constraint Compliance")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Error/Violation Rate")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(
            0.5, 0.5, "No constraint metrics\navailable", ha="center", va="center", transform=ax4.transAxes, fontsize=10
        )
        ax4.set_title("Constraint Compliance (N/A)")

    plt.tight_layout()

    # Display final analysis if requested and available
    if show_final_analysis and "final_analysis" in training_results:
        final_analysis = training_results["final_analysis"]
        if isinstance(final_analysis, dict):
            print("\n Final Model Analysis:")

            if "archetype_analysis" in final_analysis:
                arch_analysis = final_analysis["archetype_analysis"]
                if "Y" in arch_analysis:
                    print(f"   Archetype coordinates shape: {arch_analysis['Y'].shape}")
                if "A" in arch_analysis:
                    print(f"   A matrix shape: {arch_analysis['A'].shape}")
                if "B" in arch_analysis:
                    print(f"   B matrix shape: {arch_analysis['B'].shape}")

            if "constraint_analysis" in final_analysis:
                const_analysis = final_analysis["constraint_analysis"]
                if "A_matrix" in const_analysis:
                    print(f"   A matrix compliance: {const_analysis['A_matrix']}")
                if "B_matrix" in const_analysis:
                    print(f"   B matrix compliance: {const_analysis['B_matrix']}")

    return fig


def print_training_metrics_summary(training_results: dict) -> None:
    """
    Print comprehensive training metrics summary to console.

    Args:
        training_results: Dictionary from train_vae() containing 'history'

    Raises
    ------
        ValueError: If training_results doesn't contain required 'history' key
    """
    if "history" not in training_results:
        raise ValueError("training_results must contain 'history' key from train_vae()")

    history = training_results["history"]

    print("\n[STATS] Training Metrics Summary:")

    # Print key metrics evolution
    key_metrics = ["loss", "archetypal_loss", "kld_loss", "archetype_r2", "rmse"]
    available_metrics = [m for m in key_metrics if m in history and len(history[m]) > 0]

    for metric in available_metrics:
        values = history[metric]
        initial_val = values[0] if len(values) > 0 else 0
        final_val = values[-1] if len(values) > 0 else 0
        improvement = final_val - initial_val

        # For loss metrics, improvement should be negative (decreasing)
        # For R² metrics, improvement should be positive (increasing)
        if "loss" in metric.lower():
            improvement_symbol = "" if improvement < 0 else "[STATS]"
        elif "r2" in metric.lower():
            improvement_symbol = "[STATS]" if improvement > 0 else ""
        else:
            improvement_symbol = "[STATS]"

        print(f"   {improvement_symbol} {metric}: {initial_val:.4f} → {final_val:.4f} (Δ{improvement:+.4f})")

    # Check for stability metrics
    stability_metrics = ["archetype_drift_mean", "archetype_stability_mean", "archetype_variance_mean"]
    available_stability = [m for m in stability_metrics if m in history and len(history[m]) > 0]

    if available_stability:
        print("\n Archetype Stability Metrics:")
        for metric in available_stability:
            values = history[metric]
            if len(values) > 1:
                final_val = values[-1]
                metric_name = metric.replace("archetype_", "").replace("_", " ").title()
                print(f"    {metric_name}: {final_val:.4f}")
    else:
        print("\n[WARNING] No stability metrics found (requires multi-epoch training with track_stability=True)")

    # Check convergence
    if "loss" in history and len(history["loss"]) > 5:
        recent_losses = history["loss"][-5:]  # Last 5 epochs
        loss_variance = np.var(recent_losses)
        if loss_variance < 1e-6:
            print(f"\n[OK] Training appears converged (loss variance in last 5 epochs: {loss_variance:.2e})")
        else:
            print(f"\n Training may still be converging (loss variance in last 5 epochs: {loss_variance:.2e})")


def visualize_training_progress_with_summary(
    training_results: dict,
    title: str = "Complete Training Analysis",
    figsize: tuple[int, int] = (12, 8),
    show_plots: bool = True,
    save_path: str = None,
) -> matplotlib.figure.Figure:
    """
    Complete training analysis combining metrics summary and visualization.

    Args:
        training_results: Dictionary from train_vae() containing 'history'
        title: Main title for the figure
        figsize: Figure size as (width, height)
        show_plots: Whether to display plots with plt.show()
        save_path: Optional path to save the figure

    Returns
    -------
        matplotlib Figure object

    Raises
    ------
        ValueError: If training_results doesn't contain required 'history' key
    """
    # Print comprehensive metrics summary
    print_training_metrics_summary(training_results)

    # Create visualization
    print("\n[STATS] Generating training plots...")
    try:
        fig = visualize_training_results(
            training_results=training_results, title=title, figsize=figsize, show_final_analysis=True
        )

        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f" Training plots saved to: {save_path}")

        # Show if requested
        if show_plots:
            plt.show()

        print("[OK] Training plots generated successfully")
        return fig

    except Exception as e:
        print(f"[WARNING] Could not generate training plots: {e}")
        raise


def visualize_zinb_training_results(
    training_results: dict,
    title: str = "ZINB Model Training Progress",
    figsize: tuple[int, int] = (12, 8),
    show_final_analysis: bool = True,
) -> matplotlib.figure.Figure:
    """
    Specialized training visualization for ZINB models with reconstruction loss components.

    Args:
        training_results: Dictionary from ZINB fit() containing 'history' and ZINB-specific metrics
        title: Main title for the figure
        figsize: Figure size as (width, height)
        show_final_analysis: Whether to print final analysis summary

    Returns
    -------
        matplotlib Figure object

    Raises
    ------
        ValueError: If training_results doesn't contain required 'history' key
    """
    if "history" not in training_results:
        raise ValueError("training_results must contain 'history' key from ZINB fit()")

    history = training_results["history"]
    use_reconstruction = training_results.get("use_reconstruction_loss", False)

    # Create subplot layout - adjust based on reconstruction loss usage
    if use_reconstruction:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        # Flatten axes array for consistent indexing
        axes_flat = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(figsize[0], figsize[1] // 2))
        # Ensure axes is always iterable
        axes_flat = axes if hasattr(axes, "__len__") else [axes]

    fig.suptitle(title, fontsize=16)

    # Plot 1: Loss evolution
    ax1 = axes_flat[0]
    if "loss" in history:
        ax1.plot(history["loss"], label="Total Loss", color="red", linewidth=2)
    if "main_zinb_loss" in history:
        ax1.plot(history["main_zinb_loss"], label="ZINB Loss", color="blue", linewidth=2)

    ax1.set_title("ZINB Loss Components")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss convergence analysis
    ax2 = axes_flat[1]
    if "loss" in history and len(history["loss"]) > 1:
        # Calculate loss differences for convergence analysis
        loss_diffs = np.diff(history["loss"])
        ax2.plot(loss_diffs, label="Loss Change", color="green", linewidth=2)
        ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax2.set_title("Loss Convergence")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss Change")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(
            0.5,
            0.5,
            "Insufficient data\nfor convergence analysis",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=10,
        )
        ax2.set_title("Loss Convergence (N/A)")

    # Plots 3&4: Reconstruction loss components (if enabled)
    if use_reconstruction and len(axes_flat) > 2:
        # Plot 3: Consistency loss
        ax3 = axes_flat[2]
        if "consistency_loss" in history and history["consistency_loss"]:
            ax3.plot(history["consistency_loss"], label="Consistency Loss", color="orange", linewidth=2)
            ax3.set_title("Reconstruction Consistency")
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("MSE Loss")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(
                0.5,
                0.5,
                "No consistency loss\ndata available",
                ha="center",
                va="center",
                transform=ax3.transAxes,
                fontsize=10,
            )
            ax3.set_title("Consistency Loss (N/A)")

        # Plot 4: Prediction consistency
        ax4 = axes_flat[3]
        if "prediction_consistency" in history and history["prediction_consistency"]:
            ax4.plot(history["prediction_consistency"], label="Prediction Consistency", color="purple", linewidth=2)
            ax4.set_title("PCA Prediction Consistency")
            ax4.set_xlabel("Epoch")
            ax4.set_ylabel("MSE Loss")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(
                0.5,
                0.5,
                "No prediction consistency\ndata available",
                ha="center",
                va="center",
                transform=ax4.transAxes,
                fontsize=10,
            )
            ax4.set_title("Prediction Consistency (N/A)")

    plt.tight_layout()

    # Show final analysis if requested
    if show_final_analysis:
        print("\n ZINB Model Training Analysis:")
        print(f"   Model Type: {training_results.get('model_type', 'ZINB')}")
        print(f"   Total Epochs: {training_results.get('n_epochs', len(history.get('loss', [])))}")
        print(f"   Final Loss: {training_results.get('final_loss', 0.0):.4f}")
        print(f"   Reconstruction Loss: {'Enabled' if use_reconstruction else 'Disabled'}")

        if "loss" in history and len(history["loss"]) > 1:
            improvement = history["loss"][0] - history["loss"][-1]
            print(f"   Loss Improvement: {improvement:+.4f}")

            # Convergence check
            if len(history["loss"]) > 5:
                recent_variance = np.var(history["loss"][-5:])
                convergence_status = "Converged" if recent_variance < 1e-6 else "Still improving"
                print(f"   Convergence: {convergence_status} (variance: {recent_variance:.2e})")

    return fig

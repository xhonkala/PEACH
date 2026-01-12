#!/usr/bin/env python
"""
WORKFLOW 08: Comprehensive Visualization
=========================================

This workflow demonstrates PEACH's comprehensive visualization capabilities:
1. Archetypal space plots (2D and 3D)
2. Training metrics and diagnostics
3. Gene/pattern dotplots and heatmaps
4. Archetype statistics and summaries

All plotting functions save figures to the current directory and can optionally display them.

Example usage:
    python WORKFLOW_08.py

Requirements:
    - peach
    - scanpy
    - matplotlib
    - Trained model with results (from WORKFLOW_03-05)
"""

import scanpy as sc
import peach as pc
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path


def safe_plt_show():
    """Show matplotlib figure only if backend is interactive."""
    backend = matplotlib.get_backend()
    if backend not in ('agg', 'Agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg'):
        plt.show()

# =============================================================================
# Configuration
# =============================================================================

# Data path
data_path = Path("data/hsc_10k.h5ad")  # Use smaller dataset for testing

# Training parameters
n_archetypes = 5
hidden_dims = [256, 128, 64]
max_epochs = 100
random_state = 42

# Visualization settings
save_plots = True  # Save figures to files
show_plots = False  # Display interactively (set True for jupyter)

# =============================================================================
# Step 1: Prepare Complete Analysis (All Prerequisites)
# =============================================================================

print("Preparing complete analysis pipeline...")
adata = sc.read_h5ad(data_path)
print(f"  Shape: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

# PCA
if 'X_pca' not in adata.obsm:
    print("  Running PCA...")
    sc.tl.pca(adata, n_comps=13)

# Train model
print(f"  Training model ({n_archetypes} archetypes, {max_epochs} epochs)...")
results = pc.tl.train_archetypal(
    adata,
    n_archetypes=n_archetypes,
    n_epochs=max_epochs,
    model_config={
        'hidden_dims': hidden_dims,
    },
    early_stopping=True,
    early_stopping_patience=10,
    seed=random_state,
    device='cpu',
)

# Compute coordinates and assign
print("  Computing coordinates and assignments...")
pc.tl.archetypal_coordinates(adata)
pc.tl.assign_archetypes(adata, percentage_per_archetype=0.15)

# Extract weights
print("  Extracting archetype weights...")
weights = pc.tl.extract_archetype_weights(adata, model=results['model'], verbose=False)

# Compute gene associations
print("  Computing gene associations...")
gene_assoc = pc.tl.gene_associations(adata, obs_key='archetypes', verbose=False)

print("  Preparation complete!\n")

# =============================================================================
# Step 2: Archetypal Space Visualizations
# =============================================================================

print("Creating archetypal space plots...")

# 2D archetypal space (basic)
print("  [1/3] 2D archetypal space...")
fig = pc.pl.archetypal_space(
    adata,
    color_by='archetypes',  # Color by archetype assignment
    save_path='archetypal_space_2d.png' if save_plots else None,
)
if show_plots:
    fig.show()

# 2D archetypal space with multiple colorings
print("  [2/3] Multi-colored archetypal space...")
# Note: archetypal_space_multi expects a LIST of AnnData objects
# For comparing multiple datasets or conditions
fig = pc.pl.archetypal_space_multi(
    [adata],  # Single dataset comparison
    labels_list=['HSC Dataset'],
    color_by='archetypes',
    title='Archetypal Space Comparison',
    save_path='archetypal_space_multi.png' if save_plots else None,
)
if show_plots:
    fig.show()

# 3D archetypal space
print("  [3/3] 3D archetypal space...")
fig = pc.pl.archetype_positions_3d(
    adata,
    save_path='archetypal_space_3d.png' if save_plots else None,
)
if show_plots:
    fig.show()

# =============================================================================
# Step 3: Training Diagnostics
# =============================================================================

print("\nCreating training diagnostic plots...")

# Training metrics over time
# display=True shows the plot AND returns the figure
print("  [1/2] Training metrics...")
fig = pc.pl.training_metrics(
    results['history'],  # Pass history dict from training results
    display=show_plots,
)
if save_plots and fig is not None:
    fig.write_image('training_metrics.png')
    print("      Saved: training_metrics.png")

# Elbow plot (if multiple n_archetypes tested)
print("  [2/2] Elbow curve...")
# Note: This requires hyperparameter search results (WORKFLOW_02)
# pc.pl.elbow_curve(
#     cv_summary,
#     save='elbow_curve.png' if save_plots else None,
#     show=show_plots,
# )
print("      (Requires cv_summary from hyperparameter search)")

# =============================================================================
# Step 4: Gene/Pattern Visualizations
# =============================================================================

print("\nCreating gene association plots...")

# Dotplot of top genes per archetype
print("  [1/2] Gene dotplot...")
if len(gene_assoc) > 0:
    fig = pc.pl.dotplot(
        gene_assoc,  # Results DataFrame from gene_associations
        x_col='archetype',
        y_col='gene',
        top_n_per_group=10,
        title='Gene-Archetype Associations',
        save_path='gene_dotplot.png' if save_plots else None,
    )
    print("      Saved: gene_dotplot.png")

# Pattern heatmap
print("  [2/2] Pattern heatmap...")
# Note: Requires pattern analysis (conditional_associations, etc.)
# pattern_results = pc.tl.pattern_analysis(adata, ...)
# pc.pl.pattern_heatmap(
#     pattern_results,
#     save='pattern_heatmap.png' if save_plots else None,
#     show=show_plots,
# )
print("      (Requires pattern_results from pattern_analysis)")

# =============================================================================
# Step 5: Archetype Statistics
# =============================================================================

print("\nCreating archetype statistics plots...")

# Archetype statistics summary (returns dict, not a plot)
print("  [1/2] Archetype statistics...")
stats = pc.pl.archetype_statistics(adata)
print(f"      Computed statistics for {len(stats)} keys")

# Archetype positions in PCA space
print("  [2/2] Archetype positions...")
fig = pc.pl.archetype_positions(
    adata,
    save_path='archetype_positions.png' if save_plots else None,
)
if show_plots:
    safe_plt_show()

# =============================================================================
# Step 6: CellRank Visualizations (if available)
# =============================================================================

print("\nCellRank-related plots (if velocity available)...")

# Fate probabilities
if 'velocity' in adata.layers:
    print("  [1/2] Fate probabilities...")
    # pc.pl.fate_probabilities(
    #     adata,
    #     save='fate_probabilities.png' if save_plots else None,
    #     show=show_plots,
    # )

    # Gene trends along lineages
    print("  [2/2] Gene trends...")
    # pc.pl.gene_trends(
    #     adata,
    #     genes=gene_list[:5],
    #     save='gene_trends.png' if save_plots else None,
    #     show=show_plots,
    # )
    print("      (Requires CellRank setup from WORKFLOW_06)")
else:
    print("      Velocity not available - skipping CellRank plots")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*70)
print("WORKFLOW 08 COMPLETE")
print("="*70)
print("Visualizations created:")
if save_plots:
    print("  • archetypal_space_2d.png - Basic 2D projection")
    print("  • archetypal_space_multi.png - Multiple colorings")
    print("  • archetypal_space_3d.png - 3D interactive view")
    print("  • training_metrics.png - Training history")
    print("  • gene_dotplot.png - Top genes per archetype")
    print("  • archetype_statistics.png - Statistical summary")
    print("  • archetype_positions.png - Positions in PCA space")
    print("\n  All plots saved to current directory")
else:
    print("  Plots generated (save_plots=False)")

print("\nVisualization functions used:")
print("  • pc.pl.archetypal_space() - 2D projections")
print("  • pc.pl.archetypal_space_multi() - Multiple views")
print("  • pc.pl.archetype_positions_3d() - 3D interactive")
print("  • pc.pl.training_metrics() - Training diagnostics")
print("  • pc.pl.dotplot() - Gene expression patterns")
print("  • pc.pl.archetype_statistics() - Summary statistics")

print("\nComplete PEACH workflow finished!")
print("See docs/INDEX.md for full documentation")
print("="*70)

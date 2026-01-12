#!/usr/bin/env python
"""
WORKFLOW 02: Hyperparameter Search with Cross-Validation
=========================================================

This workflow demonstrates how to find optimal hyperparameters for archetypal analysis:
1. Load preprocessed data (with PCA from WORKFLOW_01)
2. Configure search space for hyperparameters
3. Run cross-validation grid search
4. Analyze results and select best configuration

The hyperparameter search tests combinations of:
- Number of archetypes (discrete values)
- Hidden layer dimensions (architecture options)
- Inflation factor (PCA inflation for Deep AA)
- CV folds and training settings

Output structure (CVSummary):
- cv_summary.ranked_configs: List of configs ranked by performance
- cv_summary.summary_df: DataFrame with all results
- cv_summary.config_results: Detailed per-config results

Example usage:
    python WORKFLOW_02.py

Requirements:
    - peach
    - scanpy
    - matplotlib (for optional elbow plot)
    - Data with PCA (from WORKFLOW_01 or equivalent)
"""

import scanpy as sc
import peach as pc
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

# Data path - should have PCA already computed
data_path = Path("data/HSC.h5ad")

# Hyperparameter search space
n_archetypes_range = [3, 4, 5]  # Number of archetypes to test
hidden_dims_options = [
    [128, 64],           # Simpler architecture
    [256, 128, 64],      # Deeper architecture
]
inflation_factor_range = [1.5]  # PCA inflation factor

# Cross-validation settings
cv_folds = 3                    # Number of CV folds
max_epochs_cv = 50              # Max epochs per fold
early_stopping_patience = 5     # Stop if no improvement
speed_preset = 'fast'           # 'fast', 'balanced', or 'thorough'
subsample_fraction = 0.5        # Use 50% of data for CV
max_cells_cv = 5000             # Maximum cells per CV run
random_state = 42               # For reproducibility

# =============================================================================
# Step 1: Load Data with PCA
# =============================================================================

print("Loading data...")
adata = sc.read_h5ad(data_path)
print(f"  Shape: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

# Ensure PCA exists (required for archetypal analysis)
if 'X_pca' not in adata.obsm:
    print("  Running PCA (required for archetypal analysis)...")
    sc.tl.pca(adata, n_comps=13)
    print(f"  PCA computed: {adata.obsm['X_pca'].shape}")
else:
    print(f"  PCA found: {adata.obsm['X_pca'].shape}")

# =============================================================================
# Step 2: Run Hyperparameter Search
# =============================================================================

print("\nRunning hyperparameter search...")
print(f"  Configurations to test:")
print(f"    n_archetypes: {n_archetypes_range}")
print(f"    hidden_dims: {len(hidden_dims_options)} options")
print(f"    inflation_factor: {inflation_factor_range}")
print(f"  Total combinations: {len(n_archetypes_range) * len(hidden_dims_options) * len(inflation_factor_range)}")
print(f"  CV folds: {cv_folds}")
print(f"  This may take several minutes...")

cv_summary = pc.tl.hyperparameter_search(
    adata,
    n_archetypes_range=n_archetypes_range,
    hidden_dims_options=hidden_dims_options,
    inflation_factor_range=inflation_factor_range,
    cv_folds=cv_folds,
    max_epochs_cv=max_epochs_cv,
    early_stopping_patience=early_stopping_patience,
    speed_preset=speed_preset,
    subsample_fraction=subsample_fraction,
    max_cells_cv=max_cells_cv,
    random_state=random_state,
    device='cpu',  # Use 'cuda' if GPU available
)

print("  Hyperparameter search complete!")

# =============================================================================
# Step 3: Analyze Results
# =============================================================================

print("\nAnalyzing results...")

# Access ranked configurations (best to worst)
ranked_configs = cv_summary.ranked_configs

print(f"\nTop 3 configurations:")
for i, config in enumerate(ranked_configs[:3], 1):
    print(f"\n  {i}. Configuration:")
    print(f"     Performance (R²): {config['metric_value']:.4f} ± {config['std_error']:.4f}")
    print(f"     Settings: {config['config_summary']}")
    # Access hyperparameters dict
    hparams = config['hyperparameters']
    print(f"     Details:")
    print(f"       - n_archetypes: {hparams['n_archetypes']}")
    print(f"       - hidden_dims: {hparams['hidden_dims']}")
    print(f"       - inflation_factor: {hparams['inflation_factor']}")

# Get best configuration
best_config = ranked_configs[0]
print(f"\nRecommended configuration:")
print(f"  n_archetypes = {best_config['hyperparameters']['n_archetypes']}")
print(f"  hidden_dims = {best_config['hyperparameters']['hidden_dims']}")
print(f"  inflation_factor = {best_config['hyperparameters']['inflation_factor']}")
print(f"  Expected R² = {best_config['metric_value']:.4f}")

# =============================================================================
# Step 4: Optional - Generate Elbow Plot
# =============================================================================

print("\nGenerating elbow plot...")

# Get summary DataFrame
summary_df = cv_summary.summary_df

# Create elbow plot showing performance vs n_archetypes
fig, ax = plt.subplots(figsize=(10, 6))

# Group by n_archetypes and calculate mean/std
grouped = summary_df.groupby('n_archetypes')['mean_archetype_r2'].agg(['mean', 'std'])

# Plot with error bars
ax.errorbar(
    grouped.index,
    grouped['mean'],
    yerr=grouped['std'],
    marker='o',
    linestyle='-',
    capsize=5,
    markersize=10,
    linewidth=2,
    color='#2E86AB'
)

ax.set_xlabel('Number of Archetypes', fontsize=13)
ax.set_ylabel('Mean Archetype R²', fontsize=13)
ax.set_title('Hyperparameter Search: Elbow Plot', fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xticks(n_archetypes_range)

# Save plot
plot_path = "elbow_plot.png"
fig.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"  Saved: {plot_path}")
plt.close(fig)

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*70)
print("WORKFLOW 02 COMPLETE")
print("="*70)
print(f"Configurations tested: {len(ranked_configs)}")
print(f"Best performance: R² = {best_config['metric_value']:.4f}")
print(f"\nBest hyperparameters:")
print(f"  • n_archetypes: {best_config['hyperparameters']['n_archetypes']}")
print(f"  • hidden_dims: {best_config['hyperparameters']['hidden_dims']}")
print(f"  • inflation_factor: {best_config['hyperparameters']['inflation_factor']}")
print(f"\nOutputs:")
print(f"  • cv_summary.ranked_configs - Ranked configurations")
print(f"  • cv_summary.summary_df - Full results DataFrame")
print(f"  • elbow_plot.png - Visual comparison")
print("\nNext workflow: WORKFLOW_03 (Model Training with best config)")
print("="*70)

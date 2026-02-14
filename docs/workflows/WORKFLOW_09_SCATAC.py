#!/usr/bin/env python
"""
WORKFLOW 09: scATAC-seq Archetypal Analysis
============================================

This workflow demonstrates end-to-end archetypal analysis on scATAC-seq data:
1. Load scATAC-seq peak count matrix
2. TF-IDF + LSI preprocessing (replaces normalization + PCA for RNA)
3. Hyperparameter search over n_archetypes
4. Train final model with selected configuration
5. Compute archetype distances, positions, and assignments
6. Gene/peak associations and characterization

Key difference from scRNA-seq:
    scATAC-seq uses TF-IDF normalization + LSI (Truncated SVD) instead of
    log-normalization + PCA. PEACH's VAE operates on dimensionally-reduced
    features regardless of modality — just point it at 'X_lsi' instead of 'X_pca'.

AnnData keys created:
    - adata.obsm['X_lsi']:              LSI embeddings [n_cells, n_components]
    - adata.uns['lsi']:                 Variance ratio and component loadings
    - adata.uns['archetype_coordinates']: Archetype positions in LSI space
    - adata.obsm['archetype_distances']: Cell-archetype distances
    - adata.obs['archetypes']:          Categorical archetype assignments

Example usage:
    python WORKFLOW_09_SCATAC.py

Requirements:
    - peach
    - scanpy
    - scikit-learn (for TruncatedSVD, installed with peach)
    - Data file: data/ovary_ATAC.h5ad (or modify data_path below)
"""

import scanpy as sc
import peach as pc
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

# Data path — scATAC-seq peak count matrix in .h5ad format
data_path = Path("data/ovary_ATAC.h5ad")

# LSI parameters
n_lsi_components = 50  # Standard for scATAC-seq (30-50)
drop_first = True      # First component captures sequencing depth, not biology

# Hyperparameter search
n_archetypes_range = [3, 4, 5, 6, 7]
cv_folds = 3
max_epochs_cv = 15

# Final model training
n_archetypes = 5       # Override after inspecting CV results
n_epochs = 50
hidden_dims = [256, 128, 64]
device = "cpu"         # "cuda" or "mps" for GPU

# Assignment parameters
percentage_per_archetype = 0.1  # Top 10% closest cells per archetype

# =============================================================================
# Step 1: Load scATAC-seq Data
# =============================================================================

print("Loading scATAC-seq data...")
adata = sc.read_h5ad(data_path)
print(f"  Shape: {adata.n_obs:,} cells x {adata.n_vars:,} peaks")
print(f"  Existing obsm keys: {list(adata.obsm.keys())}")

# Check sparsity
import scipy.sparse as sp
if sp.issparse(adata.X):
    density = adata.X.nnz / (adata.n_obs * adata.n_vars)
    print(f"  Sparsity: {1 - density:.1%} sparse ({density:.1%} non-zero)")

# =============================================================================
# Step 2: TF-IDF + LSI Preprocessing
# =============================================================================

# If your data already has X_lsi computed (e.g., by ArchR or Signac), skip this step:
if "X_lsi" in adata.obsm:
    print(f"\nExisting X_lsi found: {adata.obsm['X_lsi'].shape}")
    print("  Using pre-computed LSI embeddings (skipping TF-IDF + LSI)")
    # If you want to recompute: del adata.obsm['X_lsi'] and re-run
else:
    print(f"\nRunning TF-IDF + LSI preprocessing (n_components={n_lsi_components})...")
    pc.pp.prepare_atacseq(
        adata,
        n_components=n_lsi_components,
        drop_first=drop_first,
    )

print(f"  LSI embeddings: {adata.obsm['X_lsi'].shape}")

if "lsi" in adata.uns:
    var_explained = adata.uns["lsi"]["variance_ratio"].sum() * 100
    print(f"  Total variance explained: {var_explained:.1f}%")

# =============================================================================
# Step 3: Hyperparameter Search
# =============================================================================

print(f"\nRunning hyperparameter search over n_archetypes={n_archetypes_range}...")
cv_summary = pc.tl.hyperparameter_search(
    adata,
    n_archetypes_range=n_archetypes_range,
    cv_folds=cv_folds,
    max_epochs_cv=max_epochs_cv,
    pca_key="X_lsi",  # Point to LSI embeddings instead of PCA
    device=device,
)

# Display results
print("\nCross-validation results:")
ranked = cv_summary.ranked_configs
for i, config in enumerate(ranked[:5]):
    print(f"  #{i+1}: n_archetypes={config['hyperparameters']['n_archetypes']}, "
          f"R²={config['metric_value']:.4f}")

# Elbow plot
pc.pl.elbow_curve(cv_summary)

# Update n_archetypes based on CV results if desired
best_n = ranked[0]['hyperparameters']['n_archetypes']
print(f"\nBest n_archetypes by CV: {best_n}")
# n_archetypes = best_n  # Uncomment to auto-select

# =============================================================================
# Step 4: Train Final Model
# =============================================================================

print(f"\nTraining final model: {n_archetypes} archetypes, {n_epochs} epochs...")
results = pc.tl.train_archetypal(
    adata,
    n_archetypes=n_archetypes,
    n_epochs=n_epochs,
    hidden_dims=hidden_dims,
    pca_key="X_lsi",  # Use LSI embeddings
    device=device,
)

final_r2 = results.get("final_archetype_r2", "N/A")
print(f"  Final archetype R²: {final_r2}")

# Training metrics
pc.pl.training_metrics(results["history"])

# =============================================================================
# Step 5: Distances, Positions & Assignments
# =============================================================================

print("\nComputing archetype coordinates and distances...")
pc.tl.archetypal_coordinates(adata, pca_key="X_lsi")

print(f"  Archetype positions: {adata.uns['archetype_coordinates'].shape}")
print(f"  Distance matrix: {adata.obsm['archetype_distances'].shape}")

# Assign cells to archetypes
pc.tl.assign_archetypes(adata, percentage_per_archetype=percentage_per_archetype)

# Show assignment distribution
print(f"\nArchetype assignments:")
print(adata.obs["archetypes"].value_counts())

# Extract barycentric weights (A matrix)
weights = pc.tl.extract_archetype_weights(adata, pca_key="X_lsi")
print(f"\nWeight matrix shape: {weights.shape}")
print(f"  Row sums (should be ~1.0): min={weights.sum(axis=1).min():.4f}, max={weights.sum(axis=1).max():.4f}")

# Visualize archetype positions in LSI space
pc.pl.archetype_positions(adata, coords_key="archetype_coordinates")
pc.pl.archetype_positions_3d(adata, coords_key="archetype_coordinates")

# =============================================================================
# Step 6: Characterization
# =============================================================================

print("\nRunning gene/peak associations...")
gene_results = pc.tl.gene_associations(adata, use_layer=None)

# Top associated peaks per archetype
print("\nTop 10 peaks per archetype:")
sig_genes = gene_results[gene_results["significant"]]
for arch in sorted(sig_genes["archetype"].unique()):
    arch_genes = sig_genes[sig_genes["archetype"] == arch].nlargest(10, "log_fold_change")
    print(f"\n  {arch}:")
    for _, row in arch_genes.iterrows():
        print(f"    {row['gene']:30s} LFC={row['log_fold_change']:.3f}  FDR={row['fdr_pvalue']:.2e}")

# If cell type annotations are available, test conditional associations
if "Cell_Type" in adata.obs.columns or "cell_type" in adata.obs.columns:
    ct_col = "Cell_Type" if "Cell_Type" in adata.obs.columns else "cell_type"
    print(f"\nTesting cell type associations ({ct_col})...")
    cond_results = pc.tl.conditional_associations(adata, obs_column=ct_col)
    sig_cond = cond_results[cond_results["significant"]]
    print(f"  {len(sig_cond)} significant archetype-cell_type associations")

    for _, row in sig_cond.iterrows():
        direction = "enriched" if row["odds_ratio"] > 1 else "depleted"
        print(f"    {row['archetype']} x {row['condition']}: OR={row['odds_ratio']:.2f} ({direction})")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("WORKFLOW 09 COMPLETE: scATAC-seq Archetypal Analysis")
print("=" * 70)
print(f"Dataset: {adata.n_obs:,} cells x {adata.n_vars:,} peaks")
print(f"LSI: {adata.obsm['X_lsi'].shape[1]} components")
print(f"Archetypes: {n_archetypes} (R²={final_r2})")
print(f"\nKey outputs:")
print(f"  adata.obsm['X_lsi']               - LSI embeddings")
print(f"  adata.uns['archetype_coordinates'] - Archetype positions in LSI space")
print(f"  adata.obsm['archetype_distances']  - Cell-archetype distance matrix")
print(f"  adata.obs['archetypes']            - Categorical assignments")
print(f"  adata.obsm['cell_archetype_weights'] - Barycentric coordinates (A matrix)")
print(f"\nNote: For downstream visualization (UMAP, etc.), scATAC-seq workflows")
print(f"typically use X_lsi for neighbor graph construction:")
print(f"  sc.pp.neighbors(adata, use_rep='X_lsi')")
print(f"  sc.tl.umap(adata)")
print("=" * 70)

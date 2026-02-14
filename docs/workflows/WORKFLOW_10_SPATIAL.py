#!/usr/bin/env python
"""
WORKFLOW 10: Spatial Archetypal Analysis
=========================================

This workflow demonstrates archetypal analysis on spatial transcriptomics data:
1. Load spatial transcriptomics data (Slide-seq, MERFISH, Visium, etc.)
2. Standard scRNA-seq preprocessing + PCA
3. Hyperparameter search and final model training
4. Compute archetype assignments
5. Build spatial neighbor graph and test archetype co-localization
6. Visualize spatial archetype patterns

Key concept:
    Archetypes are learned from gene expression (same as scRNA-seq). The spatial
    component comes AFTER archetype assignment — we ask whether cells assigned to
    the same/different archetypes are spatially co-localized or separated.

Spatial-specific AnnData keys created:
    - adata.obsp['spatial_connectivities']:    Spatial neighbor graph
    - adata.obsp['spatial_distances']:         Spatial distance matrix
    - adata.uns['archetype_nhood_enrichment']: Z-score matrix [n_arch x n_arch]
    - adata.uns['archetype_co_occurrence']:    Distance-dependent ratios

Requirements:
    - peach
    - peach[spatial] (squidpy >= 1.3.0): pip install peach[spatial]
    - scanpy
    - Data file: data/melanoma/melanoma_LN.h5ad (Slide-seq v2) or modify below

Example usage:
    python WORKFLOW_10_SPATIAL.py
"""

import numpy as np
import scanpy as sc
import peach as pc
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

# Data path — spatial transcriptomics with obsm['spatial'] coordinates
data_path = Path("data/melanoma/melanoma_LN.h5ad")

# Preprocessing
n_top_genes = 2000  # For highly variable gene selection
n_pcs = 30

# Hyperparameter search
n_archetypes_range = [4, 5, 6, 7, 8]
cv_folds = 3
max_epochs_cv = 15

# Final model
n_archetypes = 5       # Override after inspecting CV results
n_epochs = 50
hidden_dims = [256, 128, 64]
device = "cpu"

# Assignment
percentage_per_archetype = 0.1

# Spatial analysis
n_spatial_neighbors = 10     # Neighbors for spatial graph
coord_type = "generic"       # "generic" for Slide-seq/MERFISH, "grid" for Visium
n_perms = 1000               # Permutations for enrichment test
co_occurrence_intervals = 50  # Distance bins for co-occurrence

# =============================================================================
# Step 1: Load Spatial Data
# =============================================================================

print("Loading spatial transcriptomics data...")
adata = sc.read_h5ad(data_path)
print(f"  Shape: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
print(f"  obsm keys: {list(adata.obsm.keys())}")

# Verify spatial coordinates exist
assert "spatial" in adata.obsm, "No spatial coordinates found at adata.obsm['spatial']"
coords = adata.obsm["spatial"]
print(f"  Spatial coordinates: {coords.shape}")
print(f"  X range: [{coords[:, 0].min():.1f}, {coords[:, 0].max():.1f}]")
print(f"  Y range: [{coords[:, 1].min():.1f}, {coords[:, 1].max():.1f}]")

# Show cell type distribution if available
if "Cell_Type" in adata.obs.columns:
    print(f"\n  Cell types ({adata.obs['Cell_Type'].nunique()}):")
    print(adata.obs["Cell_Type"].value_counts().to_string())

# =============================================================================
# Step 2: Preprocessing (Standard scRNA-seq Pipeline)
# =============================================================================

print("\nPreprocessing...")

# Normalize and log-transform
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# HVG selection and PCA
sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
sc.pp.pca(adata, n_comps=n_pcs)
print(f"  PCA: {adata.obsm['X_pca'].shape}")

# =============================================================================
# Step 3: Hyperparameter Search
# =============================================================================

print(f"\nRunning hyperparameter search over n_archetypes={n_archetypes_range}...")
cv_summary = pc.tl.hyperparameter_search(
    adata,
    n_archetypes_range=n_archetypes_range,
    cv_folds=cv_folds,
    max_epochs_cv=max_epochs_cv,
    device=device,
)

# Display ranked results
ranked = cv_summary.ranked_configs
print("\nCross-validation results:")
for i, config in enumerate(ranked[:5]):
    print(f"  #{i+1}: n_archetypes={config['hyperparameters']['n_archetypes']}, "
          f"R²={config['metric_value']:.4f}")

pc.pl.elbow_curve(cv_summary)

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
    device=device,
)

final_r2 = results.get("final_archetype_r2", "N/A")
print(f"  Final archetype R²: {final_r2}")
pc.pl.training_metrics(results["history"])

# =============================================================================
# Step 5: Archetype Distances, Positions & Assignments
# =============================================================================

print("\nComputing archetype coordinates and distances...")
pc.tl.archetypal_coordinates(adata)

print(f"  Archetype positions: {adata.uns['archetype_coordinates'].shape}")
print(f"  Distance matrix: {adata.obsm['archetype_distances'].shape}")

pc.tl.assign_archetypes(adata, percentage_per_archetype=percentage_per_archetype)
print(f"\nArchetype assignments:")
print(adata.obs["archetypes"].value_counts())

# Extract weights
weights = pc.tl.extract_archetype_weights(adata)
print(f"\nWeight matrix: {weights.shape}")

# Archetype position plots
pc.pl.archetype_positions(adata)

# =============================================================================
# Step 6: Spatial Neighbor Graph
# =============================================================================

print(f"\nBuilding spatial neighbor graph (n_neighs={n_spatial_neighbors})...")
pc.tl.spatial_neighbors(
    adata,
    n_neighs=n_spatial_neighbors,
    coord_type=coord_type,
)

print(f"  Connectivity matrix: {adata.obsp['spatial_connectivities'].shape}")

# =============================================================================
# Step 7: Neighborhood Enrichment Analysis
# =============================================================================

print(f"\nTesting neighborhood enrichment ({n_perms} permutations)...")
nhood_result = pc.tl.archetype_nhood_enrichment(
    adata,
    n_perms=n_perms,
)

zscore = nhood_result["zscore"]
print(f"  Z-score matrix: {zscore.shape}")
print(f"  Z-score range: [{zscore.min():.2f}, {zscore.max():.2f}]")
print(f"\n  Interpretation:")
print(f"    Positive z-score → archetypes co-localize (found near each other)")
print(f"    Negative z-score → archetypes segregate (found apart)")

# Identify strongest co-localizations
n_arch = zscore.shape[0]
labels = sorted(adata.obs["archetypes"].unique())
print(f"\n  Top co-localizations:")
pairs = []
for i in range(n_arch):
    for j in range(i + 1, n_arch):
        pairs.append((zscore[i, j], labels[i], labels[j]))
pairs.sort(reverse=True)
for z, a, b in pairs[:5]:
    status = "co-localize" if z > 0 else "segregate"
    print(f"    {a} - {b}: z={z:.2f} ({status})")

# Plot
pc.pl.nhood_enrichment(adata)

# =============================================================================
# Step 8: Co-occurrence Analysis
# =============================================================================

print(f"\nComputing distance-dependent co-occurrence ({co_occurrence_intervals} intervals)...")
cooc_result = pc.tl.archetype_co_occurrence(
    adata,
    interval=co_occurrence_intervals,
)

occ = cooc_result["occ"]
print(f"  Co-occurrence tensor: {occ.shape}")
print(f"    [n_archetypes={occ.shape[0]}, n_archetypes={occ.shape[1]}, n_distances={occ.shape[2]}]")
print(f"  Ratio range: [{occ.min():.3f}, {occ.max():.3f}]")
print(f"    >1 = co-occur more than chance; <1 = avoid each other")

pc.pl.co_occurrence(adata)

# =============================================================================
# Step 9: Spatial Archetype Visualization
# =============================================================================

print("\nPlotting spatial archetype map...")
pc.pl.spatial_archetypes(adata)

# If cell types available, also plot them spatially for comparison
if "Cell_Type" in adata.obs.columns:
    pc.pl.spatial_archetypes(adata, color_key="Cell_Type", title="Spatial Cell Type Map")

# =============================================================================
# Step 10: Gene Characterization
# =============================================================================

print("\nRunning gene associations...")
gene_results = pc.tl.gene_associations(adata)

sig_genes = gene_results[gene_results["significant"]]
print(f"  {len(sig_genes)} significant gene-archetype associations")

print("\nTop 10 genes per archetype:")
for arch in sorted(sig_genes["archetype"].unique()):
    arch_genes = sig_genes[sig_genes["archetype"] == arch].nlargest(10, "log_fold_change")
    print(f"\n  {arch}:")
    for _, row in arch_genes.iterrows():
        print(f"    {row['gene']:20s} LFC={row['log_fold_change']:.3f}  FDR={row['fdr_pvalue']:.2e}")

# Conditional associations with cell type
if "Cell_Type" in adata.obs.columns:
    print(f"\nTesting cell type enrichment in archetypes...")
    cond_results = pc.tl.conditional_associations(adata, obs_column="Cell_Type")
    sig_cond = cond_results[cond_results["significant"]]
    print(f"  {len(sig_cond)} significant archetype-cell_type associations")
    for _, row in sig_cond.nlargest(10, "odds_ratio").iterrows():
        print(f"    {row['archetype']} x {row['condition']}: OR={row['odds_ratio']:.2f}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("WORKFLOW 10 COMPLETE: Spatial Archetypal Analysis")
print("=" * 70)
print(f"Dataset: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
print(f"Archetypes: {n_archetypes} (R²={final_r2})")
print(f"\nKey outputs:")
print(f"  adata.uns['archetype_coordinates']        - Archetype positions in PCA space")
print(f"  adata.obsm['archetype_distances']         - Cell-archetype distances")
print(f"  adata.obs['archetypes']                   - Categorical assignments")
print(f"  adata.obsp['spatial_connectivities']      - Spatial neighbor graph")
print(f"  adata.uns['archetype_nhood_enrichment']   - Enrichment z-scores")
print(f"  adata.uns['archetype_co_occurrence']      - Distance-dependent co-occurrence")
print(f"\nBiological interpretation:")
print(f"  - Positive nhood enrichment z-score: archetypes co-localize spatially")
print(f"  - Co-occurrence ratio > 1: archetypes found together at that distance")
print(f"  - Cross-reference with gene/cell-type associations to interpret spatial patterns")
print("=" * 70)

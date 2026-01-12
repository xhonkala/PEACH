#!/usr/bin/env python
"""
WORKFLOW 01: Data Loading & PCA Preprocessing
==============================================

This workflow demonstrates the foundational data preprocessing steps for archetypal analysis:
1. Load single-cell RNA-seq data
2. Basic quality control (QC) and filtering
3. Normalization
4. Principal Component Analysis (PCA)

The output creates essential AnnData keys used by all downstream workflows:
- adata.obsm['X_pca']: PCA-transformed cell embeddings (n_cells × n_components)
- adata.varm['PCs']: Principal component loadings (n_genes × n_components)
- adata.uns['pca']: Variance explained and PCA parameters

Note: This workflow uses standard scanpy preprocessing and is compatible with any
scVerse-ecosystem single-cell dataset.

Example usage:
    python WORKFLOW_01.py

Requirements:
    - scanpy
    - numpy
    - Data file: data/HSC.h5ad (or modify data_path below)
"""

import numpy as np
import scanpy as sc
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

# Data path - modify this to use your own dataset
data_path = Path("data/HSC.h5ad")

# PCA parameters
n_pcs = 13  # Number of principal components to compute

# =============================================================================
# Step 1: Load Data
# =============================================================================

print("Loading single-cell data...")
adata = sc.read_h5ad(data_path)
print(f"  Shape: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
print(f"  Layers: {list(adata.layers.keys())}")

# =============================================================================
# Step 2: Quality Control
# =============================================================================

print("\nCalculating QC metrics...")
sc.pp.calculate_qc_metrics(adata, inplace=True)
print(f"  Median genes per cell: {adata.obs['n_genes_by_counts'].median():.0f}")
print(f"  Median counts per cell: {adata.obs['total_counts'].median():.0f}")

# Optional: Filter cells and genes based on QC metrics
# Uncomment and adjust thresholds as needed:
# sc.pp.filter_cells(adata, min_genes=200)
# sc.pp.filter_genes(adata, min_cells=3)

# =============================================================================
# Step 3: Normalization
# =============================================================================

print("\nNormalizing data...")

# Check if data is already normalized (max value < 20 suggests log-transformed)
max_val = adata.X.max() if hasattr(adata.X, 'max') else np.max(adata.X)

if max_val > 20:
    print("  Applying normalization (target_sum=1e4) and log1p transform...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
else:
    print("  Data appears already normalized (skipping)")

print(f"  Data range: 0 to {adata.X.max():.2f}" if hasattr(adata.X, 'max') else f"  Data range: 0 to {np.max(adata.X):.2f}")

# =============================================================================
# Step 4: Principal Component Analysis (PCA)
# =============================================================================

print(f"\nRunning PCA (n_comps={n_pcs})...")
sc.tl.pca(adata, n_comps=n_pcs)

print(f"  Created adata.obsm['X_pca']: {adata.obsm['X_pca'].shape}")
print(f"  Created adata.varm['PCs']: {adata.varm['PCs'].shape}")
print(f"  Created adata.uns['pca'] with keys: {list(adata.uns['pca'].keys())}")

# Show variance explained
if 'variance_ratio' in adata.uns['pca']:
    var_ratio = adata.uns['pca']['variance_ratio']
    cum_var = np.cumsum(var_ratio)
    print(f"\nVariance explained:")
    print(f"  PC1: {var_ratio[0]*100:.2f}%")
    print(f"  First 5 PCs: {cum_var[4]*100:.2f}%")
    print(f"  All {n_pcs} PCs: {cum_var[-1]*100:.2f}%")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*70)
print("WORKFLOW 01 COMPLETE")
print("="*70)
print(f"Dataset: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
print(f"PCA: {n_pcs} components computed")
print("\nKey outputs created:")
print(f"  • adata.obsm['X_pca']  - Cell embeddings in PCA space")
print(f"  • adata.varm['PCs']    - Gene loadings for each PC")
print(f"  • adata.uns['pca']     - Variance and parameters")
print("\nNext workflow: WORKFLOW_02 (Hyperparameter Search)")
print("="*70)

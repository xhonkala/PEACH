#!/usr/bin/env python
"""
WORKFLOW 06: CellRank Integration for Lineage Analysis
=======================================================

This workflow demonstrates how to integrate PEACH archetypes with CellRank for lineage tracing:
1. Setup CellRank with ConnectivityKernel (no velocity required)
2. Use archetype assignments as terminal states
3. Compute fate probabilities via GPCCA
4. Compute lineage pseudotimes
5. Compute lineage drivers (genes driving fate decisions)

NOTE: This workflow uses ConnectivityKernel which does NOT require RNA velocity.
CellRank can work with just a k-NN graph from PCA coordinates.

Example usage:
    python WORKFLOW_06_CELLRANK.py

Requirements:
    - peach
    - scanpy
    - cellrank >= 2.0
"""

import scanpy as sc
import numpy as np
import peach as pc
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

data_path = Path("data/helsinki_fit.h5ad")
output_dir = Path("tests")
output_dir.mkdir(exist_ok=True)

# Training parameters
n_archetypes = 5
hidden_dims = [256, 128, 64]
n_epochs = 50
seed = 42

# =============================================================================
# Step 1: Prepare Data with Model and Assignments
# =============================================================================

print("=" * 70)
print("WORKFLOW 06: CellRank Integration for Lineage Analysis")
print("=" * 70)

print("\nStep 1: Preparing data...")
adata = sc.read_h5ad(data_path)
print(f"  Shape: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

# Ensure PCA exists
pca_key = 'X_pca' if 'X_pca' in adata.obsm else 'X_PCA'
if pca_key not in adata.obsm:
    print("  Running PCA...")
    sc.tl.pca(adata, n_comps=13)
    pca_key = 'X_pca'
print(f"  PCA key: {pca_key}")

# Check if we need to train - require both distances AND weights to skip training
if 'archetype_distances' in adata.obsm and 'cell_archetype_weights' in adata.obsm:
    print("  Using existing archetypal model from data file")
    n_archetypes = adata.obsm['archetype_distances'].shape[1]
else:
    # Train model (needed to extract weights)
    print(f"  Training model ({n_archetypes} archetypes)...")
    results = pc.tl.train_archetypal(
        adata,
        n_archetypes=n_archetypes,
        n_epochs=n_epochs,
        model_config={'hidden_dims': hidden_dims},
        seed=seed,
        device='cpu',
    )
    # Compute archetype coordinates after training
    print("  Computing archetype coordinates...")
    pc.tl.archetypal_coordinates(adata)

    # Extract archetype weights using the trained model
    print("  Extracting archetype weights...")
    pc.tl.extract_archetype_weights(adata, model=results['model'])

# Assign cells to archetypes
if 'archetypes' not in adata.obs:
    print("  Assigning cells to archetypes...")
    pc.tl.assign_archetypes(adata)

print(f"  Archetype distribution:")
print(adata.obs['archetypes'].value_counts().to_string(header=False))

# =============================================================================
# Step 2: Setup CellRank with ConnectivityKernel
# =============================================================================

print("\n" + "-" * 70)
print("Step 2: CellRank Setup (ConnectivityKernel)")
print("-" * 70)

# Compute neighbors if not present
n_pcs = min(11, adata.obsm[pca_key].shape[1])
if 'neighbors' not in adata.uns:
    print(f"  Computing neighbors (n_pcs={n_pcs})...")
    sc.pp.neighbors(adata, n_neighbors=30, n_pcs=n_pcs, use_rep=pca_key)

# Setup CellRank using archetype assignments as terminal states
print("  Setting up CellRank with archetype terminal states...")
ck, g = pc.tl.setup_cellrank(
    adata,
    terminal_obs_key='archetypes',  # Use archetypes as terminal states
    n_neighbors=30,
    n_pcs=n_pcs,
    compute_paga=False,
    tol=1e-4,  # More permissive GMRES tolerance
    verbose=True
)

print("  CellRank kernel initialized")
print(f"  GPCCA estimator stored in adata.uns['cellrank_gpcca']")
print(f"  Fate probabilities shape: {adata.obsm['fate_probabilities'].shape}")
print(f"  Lineage names: {adata.uns['lineage_names']}")

# =============================================================================
# Step 3: Compute Lineage Pseudotimes
# =============================================================================

print("\n" + "-" * 70)
print("Step 3: Lineage Pseudotimes")
print("-" * 70)

print("  Computing pseudotimes for each lineage...")
pc.tl.compute_lineage_pseudotimes(adata)

# Report pseudotime stats
for lineage in adata.uns['lineage_names']:
    pt_key = f'pseudotime_to_{lineage}'
    if pt_key in adata.obs.columns:
        pt_vals = adata.obs[pt_key].dropna()
        print(f"    {lineage}: {len(pt_vals):,} cells, "
              f"range [{pt_vals.min():.3f}, {pt_vals.max():.3f}]")

# =============================================================================
# Step 4: Compute Lineage Drivers
# =============================================================================

print("\n" + "-" * 70)
print("Step 4: Lineage Drivers")
print("-" * 70)

# Pick a target lineage (first archetype)
target_lineage = adata.uns['lineage_names'][0]
print(f"  Computing drivers for lineage: {target_lineage}")

drivers = pc.tl.compute_lineage_drivers(
    adata,
    lineage=target_lineage,
    n_genes=50,
    method='correlation'
)

if drivers is not None and len(drivers) > 0:
    print(f"  Top 10 driver genes for {target_lineage}:")
    for i, row in drivers.head(10).iterrows():
        print(f"    {row['gene']:15} corr={row['correlation']:+.3f} p={row['pvalue']:.2e}")
else:
    print("  No drivers computed (check gene expression data)")

# =============================================================================
# Step 5: Compute Transition Frequencies
# =============================================================================

print("\n" + "-" * 70)
print("Step 5: Transition Frequencies")
print("-" * 70)

transitions = pc.tl.compute_transition_frequencies(adata)
print(f"  Transition matrix shape: {transitions.shape}")
print("  Transition frequencies (archetype -> archetype):")
print(transitions.to_string())

# =============================================================================
# Step 6: Visualization
# =============================================================================

print("\n" + "-" * 70)
print("Step 6: Visualization")
print("-" * 70)

# Fate probability plot
print("  Plotting fate probabilities...")
try:
    fig = pc.pl.fate_probabilities(
        adata,
        states=[target_lineage],
        basis='pca'
    )
    if fig is not None:
        output_path = output_dir / 'workflow_06_fate_probabilities.png'
        fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
        print(f"    Saved: {output_path}")
except Exception as e:
    print(f"    Skipped fate plot: {e}")

# 3D archetypal space colored by fate probability
print("  Creating 3D archetypal space visualization...")
fate_col = f'fate_to_{target_lineage}'
if 'fate_probabilities' in adata.obsm:
    lineage_idx = adata.uns['lineage_names'].index(target_lineage)
    adata.obs[fate_col] = adata.obsm['fate_probabilities'][:, lineage_idx]

    fig = pc.pl.archetypal_space(
        adata,
        color_by=fate_col,
        color_scale='viridis',
        title=f'Fate Probability toward {target_lineage}'
    )
    if fig is not None:
        output_path = output_dir / 'workflow_06_archetypal_space_fate.html'
        fig.write_html(str(output_path))
        print(f"    Saved: {output_path}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("WORKFLOW 06 COMPLETE")
print("=" * 70)

print("\nKey Outputs:")
print(f"  adata.obs['archetypes']          - Archetype assignments")
print(f"  adata.obsm['fate_probabilities'] - Fate probs ({adata.obsm['fate_probabilities'].shape})")
print(f"  adata.uns['lineage_names']       - {adata.uns['lineage_names']}")
print(f"  adata.uns['cellrank_gpcca']      - GPCCA estimator (for downstream)")

print("\nPseudotime keys in adata.obs:")
pt_keys = [k for k in adata.obs.columns if k.startswith('pseudotime_')]
for key in pt_keys:
    print(f"  {key}")

print("\nCellRank Functions Used:")
print("  pc.tl.setup_cellrank(adata, terminal_obs_key='archetypes')")
print("  pc.tl.compute_lineage_pseudotimes(adata)")
print("  pc.tl.compute_lineage_drivers(adata, lineage=..., method='correlation')")
print("  pc.tl.compute_transition_frequencies(adata)")

print("\nNext workflows:")
print("  WORKFLOW_08: Comprehensive Visualization")
print("=" * 70)

#!/usr/bin/env python
"""
WORKFLOW 04: Archetype Coordinates & Cell Assignment
=====================================================

This workflow demonstrates how to characterize cells in terms of archetypes:
1. Compute archetype coordinates (distances in PCA space)
2. Extract archetype weights (barycentric coordinates from model)
3. Assign cells to nearest archetypes (categorical labels)

CRITICAL DISTINCTION:
- archetype_distances: Euclidean distances from each cell to archetype positions in PCA space
- cell_archetype_weights: Barycentric coordinates from the trained model (sum to 1 per cell)

These can disagree! Distance-based and weight-based assignments may differ for ~60% of cells.
This is expected and reflects the difference between geometric and learned representations.

Example usage:
    python WORKFLOW_04.py

Requirements:
    - peach
    - scanpy
    - Trained model (from WORKFLOW_03)
"""

import scanpy as sc
import peach as pc
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

# Data path
data_path = Path("data/HSC.h5ad")

# Training parameters (from WORKFLOW_03)
n_archetypes = 5
hidden_dims = [256, 128, 64]
max_epochs = 50
random_state = 42

# =============================================================================
# Step 1: Load Data and Train Model (Prerequisites)
# =============================================================================

print("Loading data and training model (prerequisite)...")
adata = sc.read_h5ad(data_path)
print(f"  Shape: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

# Ensure PCA exists
if 'X_pca' not in adata.obsm:
    print("  Running PCA...")
    sc.tl.pca(adata, n_comps=13)

# Train archetypal model (required for downstream analyses)
print(f"  Training model ({n_archetypes} archetypes)...")
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
print(f"  Training complete!")

# Check that archetype coordinates were stored
if 'archetype_coordinates' in adata.uns:
    arch_coords = adata.uns['archetype_coordinates']
    print(f"  Archetype coordinates: {arch_coords.shape}")
else:
    print("  Warning: archetype_coordinates not found in adata.uns")

# =============================================================================
# Step 2: Compute Archetype Distances (Geometric)
# =============================================================================

print("\nComputing archetype distances...")
print("  (Euclidean distances in PCA space)")

pc.tl.archetypal_coordinates(adata)

# Access the created distances
distances = adata.obsm['archetype_distances']
print(f"  Created: adata.obsm['archetype_distances']")
print(f"  Shape: {distances.shape} (cells × archetypes)")
print(f"  Distance range: [{distances.min():.4f}, {distances.max():.4f}]")

# Find nearest archetype for each cell (distance-based)
nearest_by_distance = distances.argmin(axis=1)
print(f"\n  Distance-based assignment:")
print(f"    Archetype 0: {(nearest_by_distance == 0).sum():,} cells")
print(f"    Archetype 1: {(nearest_by_distance == 1).sum():,} cells")
print(f"    Archetype 2: {(nearest_by_distance == 2).sum():,} cells")
print(f"    Archetype 3: {(nearest_by_distance == 3).sum():,} cells")
print(f"    Archetype 4: {(nearest_by_distance == 4).sum():,} cells")

# =============================================================================
# Step 3: Extract Archetype Weights (Model-based)
# =============================================================================

print("\nExtracting archetype weights...")
print("  (Barycentric coordinates from trained model)")

weights = pc.tl.extract_archetype_weights(adata, model=results['model'])

print(f"  Returned: weights array")
print(f"  Shape: {weights.shape} (cells × archetypes)")
print(f"  Weights sum to 1 per cell: {weights.sum(axis=1).mean():.6f}")
print(f"  Weight range: [{weights.min():.4f}, {weights.max():.4f}]")

# Also stored in adata
if 'cell_archetype_weights' in adata.obsm:
    stored_weights = adata.obsm['cell_archetype_weights']
    print(f"  Also stored: adata.obsm['cell_archetype_weights']")
    print(f"  Shape: {stored_weights.shape}")

# Find dominant archetype for each cell (weight-based)
nearest_by_weight = weights.argmax(axis=1)
print(f"\n  Weight-based assignment:")
print(f"    Archetype 0: {(nearest_by_weight == 0).sum():,} cells")
print(f"    Archetype 1: {(nearest_by_weight == 1).sum():,} cells")
print(f"    Archetype 2: {(nearest_by_weight == 2).sum():,} cells")
print(f"    Archetype 3: {(nearest_by_weight == 3).sum():,} cells")
print(f"    Archetype 4: {(nearest_by_weight == 4).sum():,} cells")

# =============================================================================
# Step 4: Assign Cells to Archetypes (Categorical)
# =============================================================================

print("\nAssigning cells to archetypes...")
print("  (Creates categorical labels based on dominant weight)")

pc.tl.assign_archetypes(adata)

# Access the categorical assignments
if 'archetypes' in adata.obs.columns:
    assignments = adata.obs['archetypes']
    print(f"  Created: adata.obs['archetypes']")
    print(f"  Type: {assignments.dtype}")
    print(f"  Categories: {list(assignments.cat.categories)}")

    # Count cells per archetype
    print(f"\n  Cell counts per archetype:")
    for cat in assignments.cat.categories:
        count = (assignments == cat).sum()
        print(f"    {cat}: {count:,} cells")

# =============================================================================
# Step 5: Compare Distance vs Weight Assignments
# =============================================================================

print("\nComparing distance-based vs weight-based assignments...")

# How often do they agree?
agreement = (nearest_by_distance == nearest_by_weight).sum()
total = len(nearest_by_distance)
agreement_pct = agreement / total * 100

print(f"  Cells where distance and weight agree: {agreement:,} / {total:,} ({agreement_pct:.1f}%)")
print(f"  Cells where they disagree: {total - agreement:,} ({100-agreement_pct:.1f}%)")
print(f"\n  Note: Disagreement is EXPECTED (~40-60% of cells).")
print(f"        Distance is geometric, weights are learned.")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*70)
print("WORKFLOW 04 COMPLETE")
print("="*70)
print(f"Key outputs created:")
print(f"  • adata.obsm['archetype_distances'] - Euclidean distances")
print(f"  • adata.obsm['cell_archetype_weights'] - Barycentric coordinates")
print(f"  • adata.obs['archetypes'] - Categorical assignments")
print(f"\nNext workflows:")
print(f"  • WORKFLOW_05: Gene/Pathway Enrichment Analysis")
print(f"  • WORKFLOW_06: CellRank Integration (requires velocity)")
print(f"  • WORKFLOW_08: Visualization")
print("="*70)

#!/usr/bin/env python
"""
WORKFLOW 03: Archetypal Model Training
=======================================

This workflow demonstrates how to train a final archetypal analysis model:
1. Load data with PCA preprocessing
2. Configure training parameters based on hyperparameter search (WORKFLOW_02)
3. Train the archetypal model
4. Access and understand training results

The training function returns a TrainingResults dict with the following structure:

GUARANTEED KEYS (always present):
- history: Training metrics per epoch (dict)
- final_model: Trained model object
- model: Same as final_model
- final_optimizer: Optimizer state
- final_analysis: Analysis results dict
- epoch_archetype_positions: Archetype positions over training
- training_config: Configuration used for training

OPTIONAL KEYS (use .get() to access):
- final_archetype_r2: Final R² score (if computed)
- final_rmse, final_mae, final_loss: Other metrics
- convergence_epoch: Epoch where convergence occurred

Additionally, the function modifies adata.uns['archetype_coordinates'] in-place.

Example usage:
    python WORKFLOW_03.py

Requirements:
    - peach
    - scanpy
    - Data with PCA (from WORKFLOW_01)
"""

import scanpy as sc
import peach as pc
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

# Data path
data_path = Path("data/HSC.h5ad")

# Training parameters (from hyperparameter search)
n_archetypes = 5                # Number of archetypes
hidden_dims = [256, 128, 64]    # Encoder architecture
inflation_factor = 1.5          # PCA inflation factor
max_epochs = 100                # Maximum training epochs
early_stopping_patience = 10    # Stop if no improvement for N epochs
random_state = 42               # For reproducibility

# =============================================================================
# Step 1: Load Data with PCA
# =============================================================================

print("Loading data...")
adata = sc.read_h5ad(data_path)
print(f"  Shape: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

# Ensure PCA exists
if 'X_pca' not in adata.obsm:
    print("  Running PCA...")
    sc.tl.pca(adata, n_comps=13)
    print(f"  PCA computed: {adata.obsm['X_pca'].shape}")
else:
    print(f"  PCA found: {adata.obsm['X_pca'].shape}")

# =============================================================================
# Step 2: Train Archetypal Model
# =============================================================================

print(f"\nTraining archetypal model...")
print(f"  n_archetypes: {n_archetypes}")
print(f"  hidden_dims: {hidden_dims}")
print(f"  inflation_factor: {inflation_factor}")
print(f"  max_epochs: {max_epochs}")
print(f"  This may take several minutes...")

results = pc.tl.train_archetypal(
    adata,
    n_archetypes=n_archetypes,
    n_epochs=max_epochs,
    model_config={
        'hidden_dims': hidden_dims,
        'inflation_factor': inflation_factor,
    },
    early_stopping=True,
    early_stopping_patience=early_stopping_patience,
    seed=random_state,
    device='cpu',  # Use 'cuda' if GPU available
)

print("  Training complete!")

# =============================================================================
# Step 3: Examine Training Results
# =============================================================================

print("\nExamining training results...")

# Access guaranteed keys
history = results['history']
final_model = results['final_model']
training_config = results['training_config']

print(f"\nTraining history:")
print(f"  Epochs completed: {len(history.get('loss', []))}")
if 'loss' in history:
    print(f"  Final loss: {history['loss'][-1]:.6f}")
if 'archetype_r2' in history:
    print(f"  Final archetype R²: {history['archetype_r2'][-1]:.4f}")

# Access optional keys safely with .get()
final_r2 = results.get('final_archetype_r2')
convergence_epoch = results.get('convergence_epoch')

if final_r2 is not None:
    print(f"\nModel performance:")
    print(f"  Final archetype R²: {final_r2:.4f}")

if convergence_epoch is not None:
    print(f"  Converged at epoch: {convergence_epoch}")

# Check that adata was modified
print(f"\nModifications to adata:")
if 'archetype_coordinates' in adata.uns:
    print(f"  adata.uns['archetype_coordinates'] created")
    coords = adata.uns['archetype_coordinates']
    print(f"  Shape: {coords.shape if hasattr(coords, 'shape') else 'N/A'}")
else:
    print(f"  Warning: adata.uns['archetype_coordinates'] not found")

# =============================================================================
# Step 4: Accessing the Trained Model
# =============================================================================

print("\nTrained model access:")
print(f"  Model type: {type(final_model).__name__}")
print(f"  Model ready for inference and analysis")

# The model can be used for:
# - Computing archetype coordinates (WORKFLOW_04)
# - Computing archetype weights
# - Gene/pathway associations (WORKFLOW_05)

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*70)
print("WORKFLOW 03 COMPLETE")
print("="*70)
print(f"Trained model:")
print(f"  • n_archetypes: {n_archetypes}")
print(f"  • Epochs run: {len(history.get('loss', []))}")
if final_r2:
    print(f"  • Final R²: {final_r2:.4f}")
print(f"\nKey outputs:")
print(f"  • results['final_model'] - Trained model for inference")
print(f"  • results['history'] - Training metrics per epoch")
print(f"  • adata.uns['archetype_coordinates'] - Cell coordinates")
print("\nNext workflow: WORKFLOW_04 (Archetype Coordinates & Assignment)")
print("="*70)

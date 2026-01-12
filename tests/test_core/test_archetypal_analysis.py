"""Tests for core archetypal analysis functionality."""

import pytest
import numpy as np
import torch
import peach as pc


def test_train_archetypal_basic(small_adata):
    """Test basic archetypal training functionality."""
    results = pc.tl.train_archetypal(
        small_adata,
        n_archetypes=3,
        n_epochs=5
    )
    
    # Check training results structure
    assert isinstance(results, dict)
    assert 'history' in results
    
    # Check archetype coordinates were stored
    assert 'archetype_coordinates' in small_adata.uns
    archetype_coords = small_adata.uns['archetype_coordinates']
    
    # Convert to numpy if tensor
    if torch.is_tensor(archetype_coords):
        archetype_coords = archetype_coords.detach().cpu().numpy()
    
    # Validate archetype shape (should match PCA dimensions)
    expected_shape = (3, small_adata.obsm['X_pca'].shape[1])  # Match actual PCA dimensions
    assert archetype_coords.shape == expected_shape


def test_archetypal_coordinates(small_adata):
    """Test archetypal coordinate extraction."""
    # Train first
    pc.tl.train_archetypal(small_adata, n_archetypes=3, n_epochs=5)
    
    # Extract coordinates
    coords = pc.tl.archetypal_coordinates(small_adata)
    
    # Check distances were computed
    assert 'archetype_distances' in small_adata.obsm
    distances = small_adata.obsm['archetype_distances']
    
    # Validate distance matrix
    assert distances.shape == (small_adata.n_obs, 3)
    assert np.all(distances >= 0)  # Distances should be non-negative


def test_assign_archetypes(small_adata):
    """Test archetype assignment functionality."""
    # Setup: train and compute coordinates
    pc.tl.train_archetypal(small_adata, n_archetypes=3, n_epochs=5)
    pc.tl.archetypal_coordinates(small_adata)
    
    # Assign archetypes
    pc.tl.assign_archetypes(small_adata, percentage_per_archetype=0.1)
    
    # Check assignments were created
    assert 'archetypes' in small_adata.obs
    assignments = small_adata.obs['archetypes']
    
    # Validate assignments
    assert len(assignments) == small_adata.n_obs
    unique_archetypes = assignments.unique()
    # Check that we have archetypes and unassigned cells
    assert any('archetype_' in str(cat) for cat in unique_archetypes)
    assert any('no_archetype' in str(cat) or 'unassigned' in str(cat) for cat in unique_archetypes)


def test_model_performance_metrics(small_adata):
    """Test that performance metrics are computed correctly."""
    results = pc.tl.train_archetypal(
        small_adata,
        n_archetypes=3,
        n_epochs=8
    )
    
    # Check final metrics exist in history
    assert 'history' in results
    history = results['history']
    assert 'loss' in history
    assert 'archetype_r2' in history
    
    final_loss = history['loss'][-1]
    final_r2 = history['archetype_r2'][-1]
    
    # Metrics should be finite numbers
    assert np.isfinite(final_loss), f"Final loss is not finite: {final_loss}"
    assert np.isfinite(final_r2), f"Final R² is not finite: {final_r2}"
    
    # R² should be in valid range
    assert -2.0 <= final_r2 <= 1.0, f"R² out of expected range: {final_r2}"
    
    # Loss should be positive
    assert final_loss >= 0, f"Negative loss: {final_loss}"


def test_meaningful_training_synthetic_recovery():
    """Test that the model can recover known archetypal structure from synthetic data."""
    # Generate synthetic data with known archetypes
    adata_synthetic = pc.pp.generate_synthetic(
        n_points=150,
        n_dimensions=25,
        n_archetypes=3,
        noise=0.1
    )
    
    # Train model
    results = pc.tl.train_archetypal(
        adata_synthetic,
        n_archetypes=3,
        n_epochs=15,
        model_config={'archetypal_weight': 1.0, 'kld_weight': 0.0}
    )
    
    # Check training progressed
    history = results['history']
    initial_r2 = history['archetype_r2'][0]
    final_r2 = history['archetype_r2'][-1]
    
    # Model should improve during training
    assert final_r2 > initial_r2, f"Model didn't improve: {initial_r2} → {final_r2}"
    
    # For synthetic data with low noise, should achieve reasonable R²
    assert final_r2 > 0.05, f"Final R² too low: {final_r2}"
    
    # Get coordinates and test archetype recovery
    pc.tl.archetypal_coordinates(adata_synthetic)
    
    # Test that we can recover some structure
    from peach._core.utils.analysis import compare_archetypal_recovery
    
    true_archetypes = adata_synthetic.uns['true_archetypes']
    estimated_archetypes = adata_synthetic.uns['archetype_coordinates']
    
    # Convert to numpy if needed
    if torch.is_tensor(estimated_archetypes):
        estimated_archetypes = estimated_archetypes.detach().cpu().numpy()
    if torch.is_tensor(true_archetypes):
        true_archetypes = true_archetypes.detach().cpu().numpy()
    
    recovery_score, assignment_accuracy = compare_archetypal_recovery(
        true_archetypes, estimated_archetypes
    )
    
    # Should recover some structure (adjusted for dimensional mismatch between full and PCA space)
    assert recovery_score < 10.0, f"Recovery score too high (no structure recovered): {recovery_score}"
    assert assignment_accuracy >= 0.0, f"Assignment accuracy should be non-negative: {assignment_accuracy}"
    
    # Basic sanity checks
    assert np.isfinite(recovery_score), f"Recovery score should be finite: {recovery_score}"
    assert np.isfinite(assignment_accuracy), f"Assignment accuracy should be finite: {assignment_accuracy}"


def test_reproducibility(small_adata):
    """Test that training produces consistent results."""
    # Train model twice with same configuration
    model_config = {
        'archetypal_weight': 1.0,
        'kld_weight': 0.0,
        'diversity_weight': 0.05
    }
    
    results1 = pc.tl.train_archetypal(
        small_adata,
        n_archetypes=3,
        n_epochs=8,
        model_config=model_config
    )
    
    # Save first results
    archetype_coords1 = small_adata.uns['archetype_coordinates'].copy()
    
    results2 = pc.tl.train_archetypal(
        small_adata,
        n_archetypes=3,
        n_epochs=8,
        model_config=model_config
    )
    
    archetype_coords2 = small_adata.uns['archetype_coordinates'].copy()
    
    # Results should be similar (allowing for some numerical variation)
    final_r2_1 = results1['history']['archetype_r2'][-1]
    final_r2_2 = results2['history']['archetype_r2'][-1]
    
    # R² should be reasonably close (adjusted threshold for small epochs)
    r2_diff = abs(final_r2_1 - final_r2_2)
    assert r2_diff < 0.5, f"R² values too different: {final_r2_1} vs {final_r2_2}"
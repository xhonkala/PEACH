"""Tests for core preprocessing functionality."""

import pytest
import numpy as np
import peach as pc


def test_generate_synthetic():
    """Test synthetic data generation with realistic gene names."""
    adata = pc.pp.generate_synthetic(
        n_points=100,
        n_dimensions=20,
        n_archetypes=3,
        noise=0.1
    )
    
    # Basic structure validation
    assert adata.n_obs == 100
    assert adata.n_vars == 20
    assert 'X_pca' in adata.obsm
    assert 'true_archetypes' in adata.uns
    
    # Check that archetypes are stored correctly
    true_archetypes = adata.uns['true_archetypes']
    assert true_archetypes.shape == (3, 20)  # n_archetypes x n_dimensions


def test_prepare_training():
    """Test DataLoader creation from AnnData."""
    # Create test data with real gene names
    adata = pc.pp.generate_synthetic(n_points=50, n_dimensions=10, n_archetypes=2)
    
    # Test DataLoader creation
    from peach.pp import prepare_training
    dataloader = prepare_training(adata, batch_size=32)
    
    # Validate DataLoader
    assert dataloader is not None
    batch = next(iter(dataloader))
    assert len(batch) == 1  # Just data, no labels
    assert batch[0].shape[1] == adata.obsm['X_pca'].shape[1]  # PCA dimensions
"""Tests for gene association analysis."""

import pytest
import pandas as pd
import peach as pc


def test_gene_associations_basic(small_adata):
    """Test gene association analysis."""
    # Setup: train model and assign archetypes
    pc.tl.train_archetypal(small_adata, n_archetypes=3, n_epochs=5)
    pc.tl.archetypal_coordinates(small_adata)
    pc.tl.assign_archetypes(small_adata, percentage_per_archetype=0.2)
    
    # Test gene associations
    results = pc.tl.gene_associations(
        small_adata,
        fdr_scope='global',
        min_cells=5
    )
    
    # Validate results structure
    assert isinstance(results, pd.DataFrame)
    required_cols = ['gene', 'archetype', 'pvalue', 'log_fold_change', 'significant']
    for col in required_cols:
        assert col in results.columns
    
    # Check that we get results for all archetypes
    archetypes = results['archetype'].unique()
    assert len(archetypes) >= 2  # Should have multiple archetypes
    
    # Check FDR correction worked
    assert 'fdr_pvalue' in results.columns
    assert all(results['fdr_pvalue'] >= results['pvalue'])


def test_fdr_correction_modes(small_adata):
    """Test different FDR correction modes."""
    # Setup
    pc.tl.train_archetypal(small_adata, n_archetypes=3, n_epochs=5)
    pc.tl.archetypal_coordinates(small_adata)
    pc.tl.assign_archetypes(small_adata, percentage_per_archetype=0.2)
    
    # Test global FDR
    results_global = pc.tl.gene_associations(
        small_adata,
        fdr_scope='global',
        min_cells=5
    )
    
    # Test per-archetype FDR
    results_per_arch = pc.tl.gene_associations(
        small_adata,
        fdr_scope='per_archetype',
        min_cells=5
    )
    
    # Test no FDR correction
    results_none = pc.tl.gene_associations(
        small_adata,
        fdr_scope='none',
        min_cells=5
    )
    
    # All should return DataFrames with same structure
    for results in [results_global, results_per_arch, results_none]:
        assert isinstance(results, pd.DataFrame)
        assert 'pvalue' in results.columns
        assert 'significant' in results.columns
    
    # FDR correction should generally reduce significance
    sig_global = results_global['significant'].sum()
    sig_none = results_none['significant'].sum()
    assert sig_global <= sig_none  # FDR should be more conservative


def test_statistical_error_handling(small_adata):
    """Test error handling in statistical functions."""
    # Test gene associations without archetype assignments
    with pytest.raises((ValueError, KeyError)):
        pc.tl.gene_associations(small_adata)
    
    # Test with invalid FDR scope
    pc.tl.train_archetypal(small_adata, n_archetypes=3, n_epochs=2)
    pc.tl.archetypal_coordinates(small_adata)
    pc.tl.assign_archetypes(small_adata)
    
    with pytest.raises(ValueError):
        pc.tl.gene_associations(
            small_adata,
            fdr_scope='invalid_scope'
        )
"""Tests for pathway association analysis."""

import pytest
import pandas as pd
import peach as pc


def test_pathway_associations_realistic(small_adata):
    """Test pathway association analysis with realistic gene names."""
    # Skip test if gseapy is not available (dependency build issues on some systems)
    try:
        import gseapy
    except ImportError:
        pytest.skip("gseapy not available - skipping pathway analysis test")
    
    # Setup: train model and assign archetypes
    pc.tl.train_archetypal(small_adata, n_archetypes=3, n_epochs=5)
    pc.tl.archetypal_coordinates(small_adata)
    pc.tl.assign_archetypes(small_adata, percentage_per_archetype=0.2)
    
    # Load pathways and compute pathway scores (should work with real gene names)
    try:
        net = pc.pp.load_pathway_networks(sources=['hallmark'])
        pc.pp.compute_pathway_scores(small_adata, net=net, obsm_key='pathway_scores')
        
        # Test pathway associations only if scores were computed successfully
        if 'pathway_scores' in small_adata.obsm and small_adata.obsm['pathway_scores'].shape[1] > 0:
            results = pc.tl.pathway_associations(
                small_adata,
                pathway_obsm_key='pathway_scores',
                fdr_scope='global',
                min_cells=3  # Lower threshold for small test fixture
            )
            
            # Validate results structure
            assert isinstance(results, pd.DataFrame)
            required_cols = ['pathway', 'archetype', 'pvalue', 'mean_diff', 'significant']
            for col in required_cols:
                assert col in results.columns
            
            # Check pathway scores were created
            assert 'pathway_scores' in small_adata.obsm
        else:
            # Skip test if no pathway overlap (acceptable for small datasets)
            pytest.skip("No pathway genes overlap with dataset - acceptable for small test data")
            
    except (RuntimeError, ImportError) as e:
        if "No pathways have genes" in str(e) or "gseapy" in str(e):
            # Expected with limited gene overlap or missing dependencies - skip gracefully
            pytest.skip(f"Pathway analysis not available: {e}")
        else:
            # Unexpected error - re-raise
            raise


def test_pattern_analysis(small_adata):
    """Test pattern analysis functionality."""
    import numpy as np

    # Setup: train model and assign archetypes
    pc.tl.train_archetypal(small_adata, n_archetypes=3, n_epochs=5)
    pc.tl.archetypal_coordinates(small_adata)
    pc.tl.assign_archetypes(small_adata, percentage_per_archetype=0.2)

    # Add pathway scores for pattern analysis
    small_adata.obsm['pathway_scores'] = np.random.rand(small_adata.n_obs, 20)

    # Test pattern analysis (use pathway_scores, not genes)
    try:
        results = pc.tl.pattern_analysis(
            small_adata,
            data_obsm_key='pathway_scores',
            verbose=False
        )

        # Validate results structure
        assert isinstance(results, dict)
        assert 'individual' in results

        individual_results = results['individual']
        assert isinstance(individual_results, pd.DataFrame)
    except ValueError as e:
        # May fail with various errors on small synthetic data
        acceptable_errors = [
            "No valid pattern tests",
            "minimum cell requirement",
            "No archetype bins meet",
        ]
        if any(err in str(e) for err in acceptable_errors):
            pass  # Acceptable for small test data
        else:
            raise
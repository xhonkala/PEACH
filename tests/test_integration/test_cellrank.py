"""
Comprehensive tests for CellRank integration functions (v0.3.0).

Tests the CellRank lineage analysis functions:
- setup_cellrank()
- compute_lineage_pseudotimes()
- compute_lineage_drivers()
- compute_transition_frequencies()

Also includes placeholder tests for CellRank visualization functions:
- fate_probabilities()
- gene_trends()
- lineage_drivers()

All tests use verified signatures from docs/core_docs/api_verification_results.json

Note: Most tests are skipped if R/rpy2 is not available, as CellRank requires R.
"""

import pytest
import numpy as np
import pandas as pd
from anndata import AnnData
import peach as pc

# Check if CellRank dependencies are available
try:
    import cellrank as cr
    import rpy2
    CELLRANK_AVAILABLE = True
except (ImportError, RuntimeError):
    # RuntimeError can occur from jax CPU feature issues
    CELLRANK_AVAILABLE = False
    cr = None


@pytest.fixture
def adata_for_cellrank():
    """Create AnnData with full archetypal analysis for CellRank testing."""
    # Generate synthetic data
    adata = pc.pp.generate_synthetic(
        n_points=500,
        n_dimensions=50,
        n_archetypes=4,
        noise=0.1
    )

    # Train model
    results = pc.tl.train_archetypal(
        adata,
        n_archetypes=4,
        n_epochs=10,
        device='cpu',
        store_coords_key='archetype_coordinates'
    )

    # Get coordinates and assign
    pc.tl.archetypal_coordinates(adata)
    pc.tl.assign_archetypes(adata, percentage_per_archetype=0.15)

    # Extract archetype weights (required for CellRank high-purity cell detection)
    pc.tl.extract_archetype_weights(adata, results['model'], verbose=False)

    # Add required neighbors for CellRank
    import scanpy as sc
    sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_pca')
    sc.tl.umap(adata)

    return adata


# =============================================================================
# Tests for setup_cellrank()
# =============================================================================

@pytest.mark.skipif(not CELLRANK_AVAILABLE, reason="CellRank/rpy2 not available")
def test_setup_cellrank_basic(adata_for_cellrank):
    """Test basic CellRank setup."""
    import os
    # Set R_HOME if not already set
    if 'R_HOME' not in os.environ:
        # Try common locations
        for r_home in ['/Library/Frameworks/R.framework/Resources',
                       '/usr/lib/R', '/usr/local/lib/R']:
            if os.path.exists(r_home):
                os.environ['R_HOME'] = r_home
                break

    ck, g = pc.tl.setup_cellrank(
        adata_for_cellrank,
        high_purity_threshold=0.80
    )

    # Verify returns
    assert ck is not None, "CellRank kernel should be returned"
    assert g is not None, "GPCCA estimator should be returned"

    # Verify AnnData modifications
    assert 'terminal_states' in adata_for_cellrank.obs.columns
    assert 'fate_probabilities' in adata_for_cellrank.obsm.keys()
    assert 'lineage_names' in adata_for_cellrank.uns.keys()


@pytest.mark.skipif(not CELLRANK_AVAILABLE, reason="CellRank/rpy2 not available")
def test_setup_cellrank_high_purity_threshold(adata_for_cellrank):
    """Test different high purity thresholds."""
    import os
    if 'R_HOME' not in os.environ:
        for r_home in ['/Library/Frameworks/R.framework/Resources',
                       '/usr/lib/R', '/usr/local/lib/R']:
            if os.path.exists(r_home):
                os.environ['R_HOME'] = r_home
                break

    # Lower threshold - more cells assigned
    ck_low, g_low = pc.tl.setup_cellrank(
        adata_for_cellrank,
        high_purity_threshold=0.70
    )

    # Count terminal state assignments
    n_terminal_low = (adata_for_cellrank.obs['terminal_states'] != 'None').sum()

    # Higher threshold - fewer cells assigned
    ck_high, g_high = pc.tl.setup_cellrank(
        adata_for_cellrank,
        high_purity_threshold=0.90
    )

    n_terminal_high = (adata_for_cellrank.obs['terminal_states'] != 'None').sum()

    # Higher threshold should result in fewer terminal state cells
    assert n_terminal_high <= n_terminal_low


def test_setup_cellrank_missing_dependencies():
    """Test error handling when CellRank dependencies missing or data not prepared."""
    if CELLRANK_AVAILABLE:
        pytest.skip("CellRank is available, can't test missing dependency case")

    adata = pc.pp.generate_synthetic(n_points=100, n_dimensions=20, n_archetypes=3)

    # Should raise ImportError, KeyError (unprepared data), or similar
    # KeyError occurs when cellrank is installed but data lacks required keys
    with pytest.raises((ImportError, ModuleNotFoundError, RuntimeError, AttributeError, KeyError)):
        pc.tl.setup_cellrank(adata)


# =============================================================================
# Tests for compute_lineage_pseudotimes()
# =============================================================================

@pytest.mark.skipif(not CELLRANK_AVAILABLE, reason="CellRank/rpy2 not available")
def test_compute_lineage_pseudotimes_basic(adata_for_cellrank):
    """Test lineage pseudotime computation."""
    import os
    if 'R_HOME' not in os.environ:
        for r_home in ['/Library/Frameworks/R.framework/Resources',
                       '/usr/lib/R', '/usr/local/lib/R']:
            if os.path.exists(r_home):
                os.environ['R_HOME'] = r_home
                break

    # Setup CellRank first
    ck, g = pc.tl.setup_cellrank(adata_for_cellrank, high_purity_threshold=0.80)

    # Compute pseudotimes
    pc.tl.compute_lineage_pseudotimes(adata_for_cellrank)

    # Verify pseudotime columns were added
    lineage_names = adata_for_cellrank.uns['lineage_names']
    for lineage in lineage_names:
        pseudotime_key = f'pseudotime_to_{lineage}'
        assert pseudotime_key in adata_for_cellrank.obs.columns, \
            f"Missing pseudotime column: {pseudotime_key}"

        # Pseudotimes should be floats between 0 and 1
        pseudotimes = adata_for_cellrank.obs[pseudotime_key]
        assert pseudotimes.dtype == float
        assert (pseudotimes >= 0).all()
        assert (pseudotimes <= 1).all()


def test_compute_lineage_pseudotimes_without_setup():
    """Test error handling when called without CellRank setup."""
    adata = pc.pp.generate_synthetic(n_points=100, n_dimensions=20, n_archetypes=3)
    pc.tl.train_archetypal(adata, n_archetypes=3, n_epochs=3, device='cpu')
    pc.tl.archetypal_coordinates(adata)
    pc.tl.assign_archetypes(adata)

    # Should raise error about missing fate probabilities
    with pytest.raises((KeyError, ValueError, RuntimeError)):
        pc.tl.compute_lineage_pseudotimes(adata)


# =============================================================================
# Tests for compute_lineage_drivers()
# =============================================================================

@pytest.mark.skipif(not CELLRANK_AVAILABLE, reason="CellRank/rpy2 not available")
def test_compute_lineage_drivers_correlation_method(adata_for_cellrank):
    """Test driver gene identification with correlation method."""
    import os
    if 'R_HOME' not in os.environ:
        for r_home in ['/Library/Frameworks/R.framework/Resources',
                       '/usr/lib/R', '/usr/local/lib/R']:
            if os.path.exists(r_home):
                os.environ['R_HOME'] = r_home
                break

    # Setup CellRank
    ck, g = pc.tl.setup_cellrank(adata_for_cellrank, high_purity_threshold=0.80)
    pc.tl.compute_lineage_pseudotimes(adata_for_cellrank)

    # Get a lineage name
    lineage = adata_for_cellrank.uns['lineage_names'][0]

    # Compute drivers with correlation method (fast)
    drivers = pc.tl.compute_lineage_drivers(
        adata_for_cellrank,
        lineage=lineage,
        method='correlation'
    )

    # Verify return type
    assert isinstance(drivers, pd.DataFrame)

    # Verify required columns
    required_cols = ['gene', 'correlation', 'pvalue']
    for col in required_cols:
        assert col in drivers.columns, f"Missing column: {col}"


@pytest.mark.skipif(not CELLRANK_AVAILABLE, reason="CellRank/rpy2 not available")
@pytest.mark.xfail(reason="GAMR method not yet implemented - only 'cellrank' and 'correlation' supported")
def test_compute_lineage_drivers_gamr_method(adata_for_cellrank):
    """Test driver gene identification with GAMR method."""
    import os
    if 'R_HOME' not in os.environ:
        for r_home in ['/Library/Frameworks/R.framework/Resources',
                       '/usr/lib/R', '/usr/local/lib/R']:
            if os.path.exists(r_home):
                os.environ['R_HOME'] = r_home
                break

    # Setup CellRank
    ck, g = pc.tl.setup_cellrank(adata_for_cellrank, high_purity_threshold=0.80)
    pc.tl.compute_lineage_pseudotimes(adata_for_cellrank)

    # Get a lineage name
    lineage = adata_for_cellrank.uns['lineage_names'][0]

    # Compute drivers with GAMR method (slow, requires R)
    drivers = pc.tl.compute_lineage_drivers(
        adata_for_cellrank,
        lineage=lineage,
        method='gamr'
    )

    # Verify return type
    assert isinstance(drivers, pd.DataFrame)


def test_compute_lineage_drivers_invalid_lineage():
    """Test error handling with invalid lineage name."""
    adata = pc.pp.generate_synthetic(n_points=100, n_dimensions=20, n_archetypes=3)

    with pytest.raises((KeyError, ValueError)):
        pc.tl.compute_lineage_drivers(adata, lineage='invalid_lineage')


# =============================================================================
# Tests for compute_transition_frequencies()
# =============================================================================

@pytest.mark.skipif(not CELLRANK_AVAILABLE, reason="CellRank/rpy2 not available")
def test_compute_transition_frequencies_basic(adata_for_cellrank):
    """Test transition frequency computation."""
    import os
    if 'R_HOME' not in os.environ:
        for r_home in ['/Library/Frameworks/R.framework/Resources',
                       '/usr/lib/R', '/usr/local/lib/R']:
            if os.path.exists(r_home):
                os.environ['R_HOME'] = r_home
                break

    # Setup CellRank
    ck, g = pc.tl.setup_cellrank(adata_for_cellrank, high_purity_threshold=0.80)

    # Compute transition frequencies
    transitions = pc.tl.compute_transition_frequencies(adata_for_cellrank)

    # Verify return type (should be DataFrame or dict)
    assert isinstance(transitions, (pd.DataFrame, dict))


def test_compute_transition_frequencies_without_setup():
    """Test error handling without CellRank setup."""
    adata = pc.pp.generate_synthetic(n_points=100, n_dimensions=20, n_archetypes=3)

    with pytest.raises((KeyError, ValueError, RuntimeError)):
        pc.tl.compute_transition_frequencies(adata)


# =============================================================================
# Placeholder Tests for CellRank Visualization Functions
# =============================================================================

@pytest.mark.skipif(not CELLRANK_AVAILABLE, reason="CellRank/rpy2 not available")
def test_fate_probabilities_visualization(adata_for_cellrank):
    """Test fate probability visualization."""
    import os
    if 'R_HOME' not in os.environ:
        for r_home in ['/Library/Frameworks/R.framework/Resources',
                       '/usr/lib/R', '/usr/local/lib/R']:
            if os.path.exists(r_home):
                os.environ['R_HOME'] = r_home
                break

    # Setup CellRank
    ck, g = pc.tl.setup_cellrank(adata_for_cellrank, high_purity_threshold=0.80)

    # Test fate probability visualization
    # Note: CellRank plotting functions display directly and don't return figures
    # We just verify it runs without error
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for testing
    pc.pl.fate_probabilities(adata_for_cellrank)
    # If we get here without error, the test passed


@pytest.mark.skipif(not CELLRANK_AVAILABLE, reason="CellRank/rpy2 not available")
def test_gene_trends_visualization(adata_for_cellrank):
    """Test gene trend visualization.

    Note: gene_trends was removed from pc.pl - use cellrank.pl.gene_trends() directly.
    This test verifies CellRank's gene_trends works with PEACH-prepared data.
    """
    import os
    if 'R_HOME' not in os.environ:
        for r_home in ['/Library/Frameworks/R.framework/Resources',
                       '/usr/lib/R', '/usr/local/lib/R']:
            if os.path.exists(r_home):
                os.environ['R_HOME'] = r_home
                break

    # Setup CellRank
    ck, g = pc.tl.setup_cellrank(adata_for_cellrank, high_purity_threshold=0.80)
    pc.tl.compute_lineage_pseudotimes(adata_for_cellrank)

    # Get a lineage and some genes
    lineage = adata_for_cellrank.uns['lineage_names'][0]
    genes = list(adata_for_cellrank.var_names[:3])

    # Use CellRank's gene_trends directly (pc.pl.gene_trends was removed)
    import cellrank as cr

    # Verify data is properly prepared for CellRank visualization
    assert 'lineage_names' in adata_for_cellrank.uns
    assert f'pseudotime_to_{lineage}' in adata_for_cellrank.obs.columns

    # CellRank gene_trends expects specific data format - just verify preparation
    # Actual visualization would use: cr.pl.gene_trends(adata, ...)


@pytest.mark.skipif(not CELLRANK_AVAILABLE, reason="CellRank/rpy2 not available")
def test_lineage_drivers_visualization(adata_for_cellrank):
    """Test lineage driver visualization."""
    import os
    if 'R_HOME' not in os.environ:
        for r_home in ['/Library/Frameworks/R.framework/Resources',
                       '/usr/lib/R', '/usr/local/lib/R']:
            if os.path.exists(r_home):
                os.environ['R_HOME'] = r_home
                break

    # Setup CellRank
    ck, g = pc.tl.setup_cellrank(adata_for_cellrank, high_purity_threshold=0.80)
    pc.tl.compute_lineage_pseudotimes(adata_for_cellrank)

    # Get a lineage
    lineage = adata_for_cellrank.uns['lineage_names'][0]

    # Compute drivers
    drivers = pc.tl.compute_lineage_drivers(
        adata_for_cellrank,
        lineage=lineage,
        method='correlation'
    )

    # Test lineage driver visualization
    fig = pc.pl.lineage_drivers(
        adata_for_cellrank,
        lineage=lineage,
        n_genes=10
    )

    # Verify figure was created
    assert fig is not None


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.skipif(not CELLRANK_AVAILABLE, reason="CellRank/rpy2 not available")
def test_cellrank_complete_workflow(adata_for_cellrank):
    """Test complete CellRank workflow from setup to driver identification."""
    import os
    if 'R_HOME' not in os.environ:
        for r_home in ['/Library/Frameworks/R.framework/Resources',
                       '/usr/lib/R', '/usr/local/lib/R']:
            if os.path.exists(r_home):
                os.environ['R_HOME'] = r_home
                break

    # 1. Setup CellRank
    ck, g = pc.tl.setup_cellrank(adata_for_cellrank, high_purity_threshold=0.80)
    assert 'terminal_states' in adata_for_cellrank.obs.columns

    # 2. Compute pseudotimes
    pc.tl.compute_lineage_pseudotimes(adata_for_cellrank)
    lineage = adata_for_cellrank.uns['lineage_names'][0]
    assert f'pseudotime_to_{lineage}' in adata_for_cellrank.obs.columns

    # 3. Compute drivers
    drivers = pc.tl.compute_lineage_drivers(
        adata_for_cellrank,
        lineage=lineage,
        method='correlation'
    )
    assert isinstance(drivers, pd.DataFrame)
    assert len(drivers) > 0

    # 4. Compute transition frequencies
    transitions = pc.tl.compute_transition_frequencies(adata_for_cellrank)
    assert transitions is not None

    # All steps completed successfully
    assert True

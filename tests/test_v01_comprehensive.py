"""
Comprehensive v0.1 Release Test Suite
======================================

This test module verifies all core PEACH functionality for the v0.1 release.
Tests are organized by module (pp, tl, pl) and cover:

1. Preprocessing functions (pp)
2. Tools functions (tl) - Training, coordinates, assignments, statistics
3. Plotting functions (pl) - Basic visualization

All tests use synthetic data or small subsets for speed.
CellRank tests are skipped if dependencies are unavailable.
"""

import pytest
import numpy as np
import pandas as pd
from anndata import AnnData
import peach as pc

# Set matplotlib backend for headless testing
import matplotlib
matplotlib.use('Agg')


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def synthetic_adata():
    """Generate synthetic data with known archetypal structure."""
    return pc.pp.generate_synthetic(
        n_points=200,
        n_dimensions=30,
        n_archetypes=3,
        noise=0.1
    )


@pytest.fixture
def trained_adata(synthetic_adata):
    """Synthetic data with trained model and assignments."""
    # Train model
    results = pc.tl.train_archetypal(
        synthetic_adata,
        n_archetypes=3,
        n_epochs=5,
        device='cpu'
    )

    # Extract coordinates and assign
    pc.tl.archetypal_coordinates(synthetic_adata)
    pc.tl.assign_archetypes(synthetic_adata, percentage_per_archetype=0.15)

    # Store model reference for weight extraction
    synthetic_adata.uns['trained_model'] = results['model']

    return synthetic_adata, results


@pytest.fixture
def adata_with_weights(trained_adata):
    """Trained data with cell-archetype weights extracted."""
    adata, results = trained_adata

    # Extract weights
    pc.tl.extract_archetype_weights(adata, results['model'], verbose=False)

    return adata, results


# =============================================================================
# PP (PREPROCESSING) TESTS
# =============================================================================

class TestPreprocessing:
    """Tests for pp module functions."""

    def test_generate_synthetic_basic(self):
        """Test basic synthetic data generation."""
        adata = pc.pp.generate_synthetic(
            n_points=100,
            n_dimensions=20,
            n_archetypes=3,
            noise=0.1
        )

        # Verify structure
        assert adata.n_obs == 100
        assert adata.n_vars == 20
        assert 'X_pca' in adata.obsm
        assert 'true_archetypes' in adata.uns

        # Verify archetype shape
        true_arch = adata.uns['true_archetypes']
        assert true_arch.shape == (3, 20)

    def test_generate_synthetic_different_sizes(self):
        """Test synthetic generation with various sizes."""
        configs = [
            (50, 10, 2),
            (200, 50, 4),
            (500, 100, 5),
        ]

        for n_points, n_dims, n_arch in configs:
            adata = pc.pp.generate_synthetic(
                n_points=n_points,
                n_dimensions=n_dims,
                n_archetypes=n_arch,
                noise=0.1
            )

            assert adata.n_obs == n_points
            assert adata.n_vars == n_dims
            assert adata.uns['true_archetypes'].shape[0] == n_arch

    def test_prepare_training(self, synthetic_adata):
        """Test DataLoader creation."""
        dataloader = pc.pp.prepare_training(synthetic_adata, batch_size=32)

        assert dataloader is not None

        # Get first batch
        batch = next(iter(dataloader))
        assert len(batch) == 1  # Just data, no labels
        assert batch[0].shape[1] == synthetic_adata.obsm['X_pca'].shape[1]


# =============================================================================
# TL (TOOLS) - TRAINING TESTS
# =============================================================================

class TestTraining:
    """Tests for training functions."""

    def test_train_archetypal_basic(self, synthetic_adata):
        """Test basic training returns expected structure."""
        results = pc.tl.train_archetypal(
            synthetic_adata,
            n_archetypes=3,
            n_epochs=5,
            device='cpu'
        )

        # Check guaranteed keys
        assert 'history' in results
        assert 'final_model' in results
        assert 'model' in results
        assert 'training_config' in results

        # Check archetype coordinates stored
        assert 'archetype_coordinates' in synthetic_adata.uns

    def test_train_archetypal_history_metrics(self, synthetic_adata):
        """Test training history contains expected metrics."""
        results = pc.tl.train_archetypal(
            synthetic_adata,
            n_archetypes=3,
            n_epochs=8,
            device='cpu'
        )

        history = results['history']

        # Core metrics
        assert 'loss' in history
        assert 'archetype_r2' in history

        # Verify epochs match
        assert len(history['loss']) == 8
        assert len(history['archetype_r2']) == 8

        # Verify metrics are valid
        assert all(np.isfinite(history['loss']))
        assert all(np.isfinite(history['archetype_r2']))

    def test_train_archetypal_model_config(self, synthetic_adata):
        """Test custom model configuration."""
        results = pc.tl.train_archetypal(
            synthetic_adata,
            n_archetypes=4,
            n_epochs=3,
            model_config={
                'hidden_dims': [64, 32],
                'inflation_factor': 2.0,
            },
            device='cpu'
        )

        assert results is not None
        assert 'archetype_coordinates' in synthetic_adata.uns

        # Should have 4 archetypes
        coords = synthetic_adata.uns['archetype_coordinates']
        assert coords.shape[0] == 4


# =============================================================================
# TL (TOOLS) - COORDINATES AND ASSIGNMENT TESTS
# =============================================================================

class TestCoordinatesAndAssignment:
    """Tests for coordinate extraction and assignment functions."""

    def test_archetypal_coordinates(self, synthetic_adata):
        """Test coordinate extraction."""
        # Train first
        pc.tl.train_archetypal(synthetic_adata, n_archetypes=3, n_epochs=3, device='cpu')

        # Extract coordinates
        result = pc.tl.archetypal_coordinates(synthetic_adata)

        # Check distances stored
        assert 'archetype_distances' in synthetic_adata.obsm

        distances = synthetic_adata.obsm['archetype_distances']
        assert distances.shape == (synthetic_adata.n_obs, 3)
        assert np.all(distances >= 0)  # Distances are non-negative

    def test_assign_archetypes(self, synthetic_adata):
        """Test archetype assignment."""
        # Setup
        pc.tl.train_archetypal(synthetic_adata, n_archetypes=3, n_epochs=3, device='cpu')
        pc.tl.archetypal_coordinates(synthetic_adata)

        # Assign
        pc.tl.assign_archetypes(synthetic_adata, percentage_per_archetype=0.15)

        # Check assignments
        assert 'archetypes' in synthetic_adata.obs
        assignments = synthetic_adata.obs['archetypes']

        # Should have archetype labels and no_archetype
        unique_labels = set(assignments.unique())
        assert any('archetype_' in str(label) for label in unique_labels)

    def test_extract_archetype_weights(self, trained_adata):
        """Test weight extraction."""
        adata, results = trained_adata

        # Extract weights
        weights = pc.tl.extract_archetype_weights(
            adata,
            results['model'],
            verbose=False
        )

        # Check weights stored
        assert 'cell_archetype_weights' in adata.obsm

        # Check shape and constraints
        assert weights.shape == (adata.n_obs, 3)

        # Weights should roughly sum to 1 (barycentric)
        row_sums = weights.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-3)


# =============================================================================
# TL (TOOLS) - STATISTICAL TESTS
# =============================================================================

class TestStatistical:
    """Tests for statistical analysis functions."""

    def test_gene_associations_basic(self, trained_adata):
        """Test basic gene association analysis."""
        adata, _ = trained_adata

        results = pc.tl.gene_associations(
            adata,
            fdr_scope='global',
            min_cells=5,
            verbose=False
        )

        # Check structure
        assert isinstance(results, pd.DataFrame)

        required_cols = ['gene', 'archetype', 'pvalue', 'log_fold_change']
        for col in required_cols:
            assert col in results.columns

    def test_gene_associations_fdr_scopes(self, trained_adata):
        """Test different FDR correction scopes."""
        adata, _ = trained_adata

        for scope in ['global', 'per_archetype', 'none']:
            results = pc.tl.gene_associations(
                adata,
                fdr_scope=scope,
                min_cells=5,
                verbose=False
            )

            assert isinstance(results, pd.DataFrame)
            assert 'fdr_pvalue' in results.columns

    def test_pattern_analysis_basic(self, trained_adata):
        """Test pattern analysis returns expected structure."""
        adata, _ = trained_adata

        # Add synthetic pathway scores
        n_pathways = 10
        adata.obsm['pathway_scores'] = np.random.rand(adata.n_obs, n_pathways)

        results = pc.tl.pattern_analysis(
            adata,
            data_obsm_key='pathway_scores',
            verbose=False
        )

        # Should return dict with DataFrames
        assert isinstance(results, dict)
        assert 'individual' in results


# =============================================================================
# PL (PLOTTING) TESTS
# =============================================================================

class TestPlotting:
    """Tests for plotting functions."""

    def test_archetypal_space_basic(self, adata_with_weights):
        """Test basic archetypal space plot."""
        adata, _ = adata_with_weights

        fig = pc.pl.archetypal_space(adata)

        assert fig is not None
        assert hasattr(fig, 'show')

    def test_archetypal_space_with_color(self, adata_with_weights):
        """Test archetypal space with color by archetype."""
        adata, _ = adata_with_weights

        fig = pc.pl.archetypal_space(adata, color_by='archetypes')

        assert fig is not None

    def test_training_metrics_plot(self, trained_adata):
        """Test training metrics visualization."""
        _, results = trained_adata

        # Pass display=False to get the figure object instead of displaying
        fig = pc.pl.training_metrics(results['history'], display=False)

        assert fig is not None
        assert hasattr(fig, 'show')

    def test_archetype_positions_plot(self, trained_adata):
        """Test archetype positions plot."""
        adata, _ = trained_adata

        fig = pc.pl.archetype_positions(adata)

        assert fig is not None

    def test_archetype_positions_3d_plot(self, trained_adata):
        """Test 3D archetype positions plot."""
        adata, _ = trained_adata

        fig = pc.pl.archetype_positions_3d(adata)

        assert fig is not None


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """End-to-end workflow tests."""

    def test_complete_workflow_synthetic(self):
        """Test complete workflow on synthetic data."""
        # 1. Generate synthetic data
        adata = pc.pp.generate_synthetic(
            n_points=150,
            n_dimensions=25,
            n_archetypes=3,
            noise=0.1
        )

        # 2. Train model
        results = pc.tl.train_archetypal(
            adata,
            n_archetypes=3,
            n_epochs=8,
            device='cpu'
        )

        assert 'archetype_coordinates' in adata.uns

        # 3. Extract coordinates
        pc.tl.archetypal_coordinates(adata)
        assert 'archetype_distances' in adata.obsm

        # 4. Assign archetypes
        pc.tl.assign_archetypes(adata, percentage_per_archetype=0.15)
        assert 'archetypes' in adata.obs

        # 5. Extract weights
        pc.tl.extract_archetype_weights(adata, results['model'], verbose=False)
        assert 'cell_archetype_weights' in adata.obsm

        # 6. Gene associations
        gene_results = pc.tl.gene_associations(adata, min_cells=5, verbose=False)
        assert len(gene_results) > 0

        # 7. Visualization
        fig = pc.pl.archetypal_space(adata, color_by='archetypes')
        assert fig is not None

        # All steps completed
        assert True

    def test_archetype_recovery_synthetic(self):
        """Test that model can recover known structure."""
        # Generate data with clear archetypal structure
        adata = pc.pp.generate_synthetic(
            n_points=200,
            n_dimensions=30,
            n_archetypes=3,
            noise=0.05  # Low noise for recovery
        )

        # Train
        results = pc.tl.train_archetypal(
            adata,
            n_archetypes=3,
            n_epochs=15,
            device='cpu'
        )

        # Check training improved
        history = results['history']
        if len(history['archetype_r2']) >= 2:
            initial_r2 = history['archetype_r2'][0]
            final_r2 = history['archetype_r2'][-1]

            # Model should improve or maintain performance
            assert final_r2 >= initial_r2 - 0.1, f"R2 degraded: {initial_r2} -> {final_r2}"


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Tests for proper error handling."""

    def test_train_without_pca_raises(self):
        """Test training without PCA raises error."""
        adata = AnnData(np.random.rand(100, 50))

        with pytest.raises(ValueError, match="PCA"):
            pc.tl.train_archetypal(adata, n_archetypes=3, n_epochs=1)

    def test_coordinates_without_training_raises(self, synthetic_adata):
        """Test coordinate extraction without training raises error."""
        with pytest.raises(ValueError):
            pc.tl.archetypal_coordinates(synthetic_adata)

    def test_assign_without_coordinates_raises(self, synthetic_adata):
        """Test assignment without coordinates raises error."""
        pc.tl.train_archetypal(synthetic_adata, n_archetypes=3, n_epochs=2, device='cpu')

        with pytest.raises(ValueError):
            pc.tl.assign_archetypes(synthetic_adata)

    def test_gene_associations_without_assignments_raises(self, synthetic_adata):
        """Test gene associations without assignments raises error."""
        pc.tl.train_archetypal(synthetic_adata, n_archetypes=3, n_epochs=2, device='cpu')
        pc.tl.archetypal_coordinates(synthetic_adata)

        with pytest.raises((ValueError, KeyError)):
            pc.tl.gene_associations(synthetic_adata)


# =============================================================================
# PERFORMANCE AND CONSTRAINTS TESTS
# =============================================================================

class TestConstraints:
    """Tests for mathematical constraints."""

    def test_weights_sum_to_one(self, adata_with_weights):
        """Test barycentric weights sum to 1."""
        adata, _ = adata_with_weights

        weights = adata.obsm['cell_archetype_weights']
        row_sums = weights.sum(axis=1)

        assert np.allclose(row_sums, 1.0, atol=1e-2)

    def test_weights_non_negative(self, adata_with_weights):
        """Test weights are non-negative (soft constraint)."""
        adata, _ = adata_with_weights

        weights = adata.obsm['cell_archetype_weights']

        # Allow small numerical violations
        assert (weights >= -0.1).all()

    def test_distances_non_negative(self, trained_adata):
        """Test distances are non-negative."""
        adata, _ = trained_adata

        distances = adata.obsm['archetype_distances']

        assert (distances >= 0).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

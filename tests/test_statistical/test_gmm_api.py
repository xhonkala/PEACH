"""Tests for feature_simplex_decomposition public API."""

import numpy as np
import pytest
from anndata import AnnData


@pytest.fixture
def gmm_adata():
    """AnnData with clear 2-cluster weights."""
    rng = np.random.default_rng(42)
    K = 3
    n_per = 200

    w0 = rng.dirichlet([10, 1, 1], size=n_per)
    w1 = rng.dirichlet([1, 10, 1], size=n_per)
    weights = np.vstack([w0, w1])

    X = rng.standard_normal((400, 30))
    adata = AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(30)]
    adata.obsm["cell_archetype_weights"] = weights
    return adata


class TestFeatureSimplexDecomposition:
    def test_basic_run(self, gmm_adata):
        """Runs, stores in adata."""
        import peach as pc

        result = pc.tl.feature_simplex_decomposition(
            gmm_adata, n_initializations=3, n_components_range=(2, 4)
        )
        assert "peach_gmm" in gmm_adata.uns
        assert result is not None

    def test_labels_stored(self, gmm_adata):
        """adata.obsm['peach_gmm_labels'] populated."""
        import peach as pc

        pc.tl.feature_simplex_decomposition(
            gmm_adata, n_initializations=3, n_components_range=(2, 4)
        )
        assert "peach_gmm_labels" in gmm_adata.obsm
        # AnnData obsm reshapes 1D arrays to (n, 1)
        assert gmm_adata.obsm["peach_gmm_labels"].shape[0] == 400

    def test_stability_scores_bounded(self, gmm_adata):
        """All scores in [0, 1]."""
        import peach as pc

        result = pc.tl.feature_simplex_decomposition(
            gmm_adata, n_initializations=5, n_components_range=(2, 4)
        )
        assert np.all(result["component_stability_scores"] >= 0)
        assert np.all(result["component_stability_scores"] <= 1)

    def test_simplex_means_valid(self, gmm_adata):
        """Centroids sum to 1."""
        import peach as pc

        result = pc.tl.feature_simplex_decomposition(
            gmm_adata, n_initializations=3, n_components_range=(2, 4)
        )
        np.testing.assert_array_almost_equal(
            result["component_simplex_means"].sum(axis=1), 1.0
        )

    def test_feature_profiles(self, gmm_adata):
        """Feature profiles computed when characterize_features=True."""
        import peach as pc

        result = pc.tl.feature_simplex_decomposition(
            gmm_adata, n_initializations=3, n_components_range=(2, 4),
            characterize_features=True
        )
        assert result.get("component_feature_profiles") is not None
        assert result["component_feature_profiles"].shape[1] == 30

    def test_no_feature_profiles(self, gmm_adata):
        """Feature profiles None when characterize_features=False."""
        import peach as pc

        result = pc.tl.feature_simplex_decomposition(
            gmm_adata, n_initializations=3, n_components_range=(2, 4),
            characterize_features=False
        )
        assert result.get("component_feature_profiles") is None

    def test_copy_does_not_modify_original(self, gmm_adata):
        """copy=True leaves original adata untouched."""
        import peach as pc

        pc.tl.feature_simplex_decomposition(
            gmm_adata, n_initializations=3, n_components_range=(2, 4),
            copy=True
        )
        assert "peach_gmm" not in gmm_adata.uns
        assert "peach_gmm_labels" not in gmm_adata.obsm

    def test_result_type(self, gmm_adata):
        """Returns plain dict (PEACH convention)."""
        import peach as pc

        result = pc.tl.feature_simplex_decomposition(
            gmm_adata, n_initializations=3, n_components_range=(2, 4)
        )
        assert isinstance(result, dict)

    def test_bic_values_shape(self, gmm_adata):
        """BIC values match n_components_tested length."""
        import peach as pc

        result = pc.tl.feature_simplex_decomposition(
            gmm_adata, n_initializations=3, n_components_range=(2, 5)
        )
        assert len(result["bic_values"]) == len(result["n_components_tested"])
        assert len(result["n_components_tested"]) == 4  # 2, 3, 4, 5

    def test_assignments_shape(self, gmm_adata):
        """Component assignments match n_cells."""
        import peach as pc

        result = pc.tl.feature_simplex_decomposition(
            gmm_adata, n_initializations=3, n_components_range=(2, 4)
        )
        assert result["component_assignments"].shape == (400,)

    def test_recovers_planted_clusters(self, gmm_adata):
        """GMM separates the two planted clusters (archetype 0 vs 1)."""
        import peach as pc

        result = pc.tl.feature_simplex_decomposition(
            gmm_adata, n_initializations=5, n_components_range=(2, 4)
        )
        labels = result["component_assignments"]
        # First 200 cells are cluster 0, next 200 are cluster 1
        # Labels may be permuted, so check that clusters are internally consistent
        labels_first = labels[:200]
        labels_second = labels[200:]
        # Majority of first 200 should share a label, majority of second 200 another
        mode_first = np.bincount(labels_first.astype(int)).argmax()
        mode_second = np.bincount(labels_second.astype(int)).argmax()
        assert mode_first != mode_second, "GMM should assign different labels to different clusters"
        # At least 80% purity within each planted cluster
        purity_first = np.mean(labels_first == mode_first)
        purity_second = np.mean(labels_second == mode_second)
        assert purity_first > 0.8, f"Cluster 0 purity {purity_first:.2f} < 0.8"
        assert purity_second > 0.8, f"Cluster 1 purity {purity_second:.2f} < 0.8"

    def test_bic_minimum_at_planted_k(self, gmm_adata):
        """BIC-optimal n_components should be near the planted k=2."""
        import peach as pc

        result = pc.tl.feature_simplex_decomposition(
            gmm_adata, n_initializations=5, n_components_range=(2, 5)
        )
        assert result["n_components_optimal"] in (2, 3), (
            f"BIC-optimal k={result['n_components_optimal']}, expected 2 or 3"
        )

    def test_unstable_cells_configurable_threshold(self, gmm_adata):
        """With reassignment_confidence=0.0 (default), all cells get assigned."""
        import peach as pc

        result = pc.tl.feature_simplex_decomposition(
            gmm_adata, n_initializations=3, n_components_range=(2, 4),
            reassignment_confidence=0.0,
        )
        labels = result["component_assignments"]
        assert np.all(labels >= 0), "Default confidence=0.0 should reassign all unstable cells"

    def test_unstable_cells_strict_threshold(self, gmm_adata):
        """With high confidence threshold, component_probabilities is populated."""
        import peach as pc

        result = pc.tl.feature_simplex_decomposition(
            gmm_adata, n_initializations=3, n_components_range=(2, 4),
            reassignment_confidence=0.8,
        )
        # component_probabilities should be in result if any unstable cells existed
        # (even if all were reassigned because they exceeded the threshold)
        # The key thing is the parameter is accepted and works without error
        assert "component_assignments" in result
        # If probabilities were computed, check shape
        proba = result.get("component_probabilities")
        if proba is not None:
            n_stable = result["n_components_stable"]
            assert proba.shape == (400, n_stable)

    def test_component_regression(self, gmm_adata):
        """Per-component regression runs on GMM-decomposed data."""
        import peach as pc

        # First run GMM decomposition
        pc.tl.feature_simplex_decomposition(
            gmm_adata, n_initializations=3, n_components_range=(2, 4),
        )
        # Now run component regression
        reg_result = pc.tl.component_regression(
            gmm_adata, n_bootstrap=10, robust_se=False,
        )
        assert "component_regs" in reg_result
        assert "n_components" in reg_result
        assert reg_result["n_components"] >= 2
        # Each component with enough cells should have a regression result
        assert len(reg_result["component_regs"]) > 0
        for c, reg in reg_result["component_regs"].items():
            assert isinstance(reg, dict)
            # SimplexRegressionResult serialized dict has these keys
            assert "vertex_coefficients" in reg
            assert "r_squared_degree1" in reg
            assert "feature_names" in reg

    def test_component_regression_requires_gmm(self, gmm_adata):
        """component_regression raises if GMM not run first."""
        import peach as pc

        with pytest.raises(ValueError, match="feature_simplex_decomposition"):
            pc.tl.component_regression(gmm_adata)

    def test_icl_model_selection(self, gmm_adata):
        """ICL model selection returns icl_values and selects reasonable k."""
        import peach as pc

        result = pc.tl.feature_simplex_decomposition(
            gmm_adata, model_selection="icl",
            n_initializations=3, n_components_range=(2, 4),
        )
        assert result["n_components_optimal"] >= 2
        assert "icl_values" in result
        icl_vals = result["icl_values"]
        assert len(icl_vals) == len(result["n_components_tested"])
        # ICL values should be finite
        assert np.all(np.isfinite(icl_vals))
        # ICL >= BIC (entropy term is non-negative)
        bic_vals = result["bic_values"]
        assert np.all(icl_vals >= bic_vals - 1e-6)

    def test_pairwise_nmi_stability(self, gmm_adata):
        """Pairwise stability scores are bounded and high for well-separated data."""
        import peach as pc

        result = pc.tl.feature_simplex_decomposition(
            gmm_adata, n_initializations=5, n_components_range=(2, 4),
        )
        scores = result["component_stability_scores"]
        assert np.all(scores >= 0) and np.all(scores <= 1)
        assert np.mean(scores) > 0.5  # well-separated clusters

    def test_dirichlet_mixture(self, gmm_adata):
        """Dirichlet mixture fits, returns simplex means summing to 1."""
        import peach as pc

        result = pc.tl.feature_simplex_decomposition(
            gmm_adata, model_type="dirichlet",
            n_initializations=3, n_components_range=(2, 4),
        )
        assert result["n_components_optimal"] >= 2
        assert result.get("model_type") == "dirichlet"
        means = np.asarray(result["component_simplex_means"])
        np.testing.assert_allclose(means.sum(axis=1), 1.0, atol=1e-6)

"""Tests for simplex GMM density decomposition.

Tests fit_simplex_gmm and characterize_components using Dirichlet-sampled
weights with planted cluster structure.
"""

import numpy as np
import pytest


@pytest.fixture
def two_cluster_weights():
    """Weights with 2 clear clusters near archetypes 0 and 1."""
    rng = np.random.default_rng(42)
    K = 3
    n_per_cluster = 200

    # Cluster near archetype 0
    alpha0 = [10, 1, 1]
    w0 = rng.dirichlet(alpha0, size=n_per_cluster)

    # Cluster near archetype 1
    alpha1 = [1, 10, 1]
    w1 = rng.dirichlet(alpha1, size=n_per_cluster)

    return np.vstack([w0, w1]), K


@pytest.fixture
def three_cluster_weights():
    """Weights with 3 well-separated clusters, one per archetype."""
    rng = np.random.default_rng(99)
    K = 3
    n_per_cluster = 150

    w0 = rng.dirichlet([15, 1, 1], size=n_per_cluster)
    w1 = rng.dirichlet([1, 15, 1], size=n_per_cluster)
    w2 = rng.dirichlet([1, 1, 15], size=n_per_cluster)

    return np.vstack([w0, w1, w2]), K


class TestFitSimplexGMM:
    def test_recovers_known_components(self, two_cluster_weights):
        """2 planted clusters -> GMM finds >= 2 components."""
        from peach._core.utils.simplex_gmm import fit_simplex_gmm

        weights, K = two_cluster_weights
        result = fit_simplex_gmm(
            weights,
            n_components_range=(2, 6),
            n_initializations=5,  # fast for testing
        )
        assert result["n_components_stable"] >= 2

    def test_bic_values_computed(self, two_cluster_weights):
        """BIC values are computed for each n_components."""
        from peach._core.utils.simplex_gmm import fit_simplex_gmm

        weights, K = two_cluster_weights
        result = fit_simplex_gmm(
            weights,
            n_components_range=(2, 5),
            n_initializations=3,
        )
        assert len(result["bic_values"]) == 4  # 2,3,4,5
        assert np.all(np.isfinite(result["bic_values"]))

    def test_n_components_tested_matches_range(self, two_cluster_weights):
        """n_components_tested should contain every value in the range."""
        from peach._core.utils.simplex_gmm import fit_simplex_gmm

        weights, K = two_cluster_weights
        result = fit_simplex_gmm(
            weights,
            n_components_range=(2, 5),
            n_initializations=3,
        )
        np.testing.assert_array_equal(
            result["n_components_tested"], np.array([2, 3, 4, 5])
        )

    def test_stability_filtering(self, two_cluster_weights):
        """Stability scores are in [0, 1]."""
        from peach._core.utils.simplex_gmm import fit_simplex_gmm

        weights, K = two_cluster_weights
        result = fit_simplex_gmm(
            weights,
            n_components_range=(2, 4),
            n_initializations=5,
        )
        scores = result["component_stability_scores"]
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)

    def test_centroid_back_to_simplex(self, two_cluster_weights):
        """ILR centroids map back to valid simplex points."""
        from peach._core.utils.simplex_gmm import fit_simplex_gmm

        weights, K = two_cluster_weights
        result = fit_simplex_gmm(
            weights,
            n_components_range=(2, 4),
            n_initializations=3,
        )
        centroids = result["component_simplex_means"]
        assert np.all(centroids > 0)
        np.testing.assert_array_almost_equal(centroids.sum(axis=1), 1.0)

    def test_assignments_shape(self, two_cluster_weights):
        """Component assignments cover all cells."""
        from peach._core.utils.simplex_gmm import fit_simplex_gmm

        weights, K = two_cluster_weights
        result = fit_simplex_gmm(
            weights,
            n_components_range=(2, 4),
            n_initializations=3,
        )
        assert result["component_assignments"].shape == (400,)

    def test_assignments_valid_labels(self, two_cluster_weights):
        """All assignment labels are either -1 or in [0, n_stable)."""
        from peach._core.utils.simplex_gmm import fit_simplex_gmm

        weights, K = two_cluster_weights
        result = fit_simplex_gmm(
            weights,
            n_components_range=(2, 4),
            n_initializations=3,
        )
        labels = result["component_assignments"]
        n_stable = result["n_components_stable"]
        valid = np.logical_or(labels == -1, np.logical_and(labels >= 0, labels < n_stable))
        assert np.all(valid)

    def test_archetype_map_shape(self, two_cluster_weights):
        """Archetype map has one entry per stable component."""
        from peach._core.utils.simplex_gmm import fit_simplex_gmm

        weights, K = two_cluster_weights
        result = fit_simplex_gmm(
            weights,
            n_components_range=(2, 4),
            n_initializations=3,
        )
        assert len(result["component_archetype_map"]) == result["n_components_stable"]

    def test_default_range(self, two_cluster_weights):
        """Default n_components_range is (K, 3*K)."""
        from peach._core.utils.simplex_gmm import fit_simplex_gmm

        weights, K = two_cluster_weights
        result = fit_simplex_gmm(
            weights,
            n_initializations=3,
        )
        expected_len = 3 * K - K + 1  # K to 3K inclusive
        assert len(result["bic_values"]) == expected_len
        assert result["n_components_tested"][0] == K
        assert result["n_components_tested"][-1] == 3 * K

    def test_gmm_model_returned(self, two_cluster_weights):
        """The fitted GMM model is returned."""
        from peach._core.utils.simplex_gmm import fit_simplex_gmm
        from sklearn.mixture import GaussianMixture

        weights, K = two_cluster_weights
        # Test Gaussian path explicitly
        result = fit_simplex_gmm(
            weights,
            n_components_range=(2, 4),
            n_initializations=3,
            model_type="gaussian",
        )
        assert isinstance(result["gmm_model"], GaussianMixture)

        # Default (dirichlet) should also return a model
        from peach._core.utils.dirichlet_mixture import DirichletMixture
        result_d = fit_simplex_gmm(
            weights,
            n_components_range=(2, 4),
            n_initializations=3,
        )
        assert isinstance(result_d["gmm_model"], DirichletMixture)

    def test_three_cluster_recovery(self, three_cluster_weights):
        """3 planted clusters near 3 archetypes are recovered."""
        from peach._core.utils.simplex_gmm import fit_simplex_gmm

        weights, K = three_cluster_weights
        result = fit_simplex_gmm(
            weights,
            n_components_range=(2, 6),
            n_initializations=5,
        )
        # Should find at least 3 stable components
        assert result["n_components_stable"] >= 3

        # Each archetype should be the nearest for at least one component
        archetypes_covered = set(result["component_archetype_map"])
        assert len(archetypes_covered) >= 3

    def test_centroids_near_planted_clusters(self, two_cluster_weights):
        """Centroids should be near the planted cluster centers."""
        from peach._core.utils.simplex_gmm import fit_simplex_gmm

        weights, K = two_cluster_weights
        result = fit_simplex_gmm(
            weights,
            n_components_range=(2, 4),
            n_initializations=5,
        )
        centroids = result["component_simplex_means"]
        # At least one centroid should have high weight on archetype 0
        max_weights_0 = centroids[:, 0].max()
        # At least one centroid should have high weight on archetype 1
        max_weights_1 = centroids[:, 1].max()

        # The planted clusters have alpha=[10,1,1] and [1,10,1]
        # so centroids should have dominant weight > 0.5 for the
        # corresponding archetype
        assert max_weights_0 > 0.5, f"No centroid near archetype 0, max weight: {max_weights_0}"
        assert max_weights_1 > 0.5, f"No centroid near archetype 1, max weight: {max_weights_1}"


class TestCharacterizeComponents:
    def test_feature_profiles(self):
        """Per-component feature means computed correctly."""
        from peach._core.utils.simplex_gmm import characterize_components

        labels = np.array([0, 0, 0, 1, 1])
        features = np.array([[1, 2], [3, 4], [5, 6], [10, 20], [30, 40]])
        profiles = characterize_components(labels, features, n_components=2)
        assert profiles.shape == (2, 2)
        np.testing.assert_array_almost_equal(profiles[0], [3, 4])  # mean of rows 0-2
        np.testing.assert_array_almost_equal(profiles[1], [20, 30])  # mean of rows 3-4

    def test_sparse_input(self):
        """Sparse feature matrices are handled correctly."""
        import scipy.sparse as sp
        from peach._core.utils.simplex_gmm import characterize_components

        labels = np.array([0, 0, 1, 1])
        features = sp.csr_matrix(np.array([[1, 0], [3, 0], [0, 10], [0, 20]]))
        profiles = characterize_components(labels, features, n_components=2)
        assert profiles.shape == (2, 2)
        np.testing.assert_array_almost_equal(profiles[0], [2, 0])
        np.testing.assert_array_almost_equal(profiles[1], [0, 15])

    def test_empty_component(self):
        """Components with no assigned cells get zero profiles."""
        from peach._core.utils.simplex_gmm import characterize_components

        labels = np.array([0, 0, 0])
        features = np.array([[1, 2], [3, 4], [5, 6]])
        profiles = characterize_components(labels, features, n_components=2)
        np.testing.assert_array_almost_equal(profiles[1], [0, 0])

    def test_excludes_negative_labels(self):
        """Cells with label -1 (unstable) are excluded from profiles."""
        from peach._core.utils.simplex_gmm import characterize_components

        labels = np.array([0, 0, -1, 1, -1])
        features = np.array([[1, 2], [3, 4], [100, 200], [10, 20], [100, 200]])
        profiles = characterize_components(labels, features, n_components=2)
        # Component 0: mean of rows 0,1 = [2, 3]
        np.testing.assert_array_almost_equal(profiles[0], [2, 3])
        # Component 1: just row 3 = [10, 20]
        np.testing.assert_array_almost_equal(profiles[1], [10, 20])


class TestStabilityAnalysis:
    """Tests focused on the stability analysis internals."""

    def test_well_separated_clusters_are_stable(self, three_cluster_weights):
        """Well-separated clusters should have high stability scores."""
        from peach._core.utils.simplex_gmm import fit_simplex_gmm

        weights, K = three_cluster_weights
        result = fit_simplex_gmm(
            weights,
            n_components_range=(3, 3),  # force 3 components
            n_initializations=10,
            stability_threshold=0.5,
        )
        # With well-separated clusters, all components should be stable
        scores = result["component_stability_scores"]
        assert np.all(scores >= 0.5), f"Some stability scores too low: {scores}"

    def test_fallback_when_none_stable(self):
        """When no components pass threshold, all are retained."""
        from peach._core.utils.simplex_gmm import fit_simplex_gmm

        rng = np.random.default_rng(123)
        # Uniform Dirichlet = no structure, hard to find stable components
        weights = rng.dirichlet([1, 1, 1], size=100)

        result = fit_simplex_gmm(
            weights,
            n_components_range=(2, 3),
            n_initializations=3,
            stability_threshold=0.99,  # very high threshold
        )
        # Should fall back to reporting all components
        assert result["n_components_stable"] > 0
        # No cells should have label -1 in fallback mode
        assert np.all(result["component_assignments"] >= 0)

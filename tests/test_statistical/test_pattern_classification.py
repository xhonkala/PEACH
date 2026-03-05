import numpy as np
import pytest


class TestClassifyPatterns:
    """Test pattern classification from regression coefficients."""

    def test_exclusive_pattern(self):
        """One high beta, others near zero -> archetype-exclusive."""
        from peach._core.utils.pattern_classification import classify_single_feature

        betas = np.array([10.0, 0.5, 0.3])
        interactions = None
        r2 = 0.8
        p_betas = np.array([1e-10, 0.5, 0.7])
        p_interactions = None
        result = classify_single_feature(betas, interactions, r2, p_betas, p_interactions)
        assert result["pattern"] == "archetype-exclusive"

    def test_flat_pattern_low_r2(self):
        """Low R^2 -> flat/ubiquitous."""
        from peach._core.utils.pattern_classification import classify_single_feature

        betas = np.array([3.0, 3.1, 2.9])
        result = classify_single_feature(
            betas, None, r2=0.01, p_betas=np.array([0.5, 0.5, 0.5]),
            p_interactions=None, r2_threshold=0.05
        )
        assert result["pattern"] == "flat"

    def test_gradient_pattern(self):
        """Ordered coefficients, high R^2 -> monotonic gradient."""
        from peach._core.utils.pattern_classification import classify_single_feature

        betas = np.array([8.0, 4.0, 1.0])
        result = classify_single_feature(
            betas, None, r2=0.7,
            p_betas=np.array([1e-10, 1e-5, 0.01]),
            p_interactions=None
        )
        assert result["pattern"] == "monotonic-gradient"

    def test_shared_pattern(self):
        """2+ elevated betas -> multi-archetype shared."""
        from peach._core.utils.pattern_classification import classify_single_feature

        betas = np.array([8.0, 7.5, 1.0])
        result = classify_single_feature(
            betas, None, r2=0.6,
            p_betas=np.array([1e-10, 1e-10, 0.3]),
            p_interactions=None
        )
        assert result["pattern"] == "multi-archetype-shared"

    def test_ridge_pattern(self):
        """Positive significant interaction -> ridge/blend-enriched."""
        from peach._core.utils.pattern_classification import classify_single_feature

        betas = np.array([5.0, 5.0, 1.0])
        interactions = np.array([3.0, 0.1, 0.1])  # strong (0,1) interaction
        result = classify_single_feature(
            betas, interactions, r2=0.7,
            p_betas=np.array([1e-5, 1e-5, 0.3]),
            p_interactions=np.array([1e-5, 0.5, 0.5])
        )
        assert result["pattern"] == "ridge"

    def test_valley_pattern(self):
        """Negative significant interaction -> valley/blend-depleted."""
        from peach._core.utils.pattern_classification import classify_single_feature

        betas = np.array([5.0, 5.0, 1.0])
        interactions = np.array([-3.0, 0.1, 0.1])
        result = classify_single_feature(
            betas, interactions, r2=0.7,
            p_betas=np.array([1e-5, 1e-5, 0.3]),
            p_interactions=np.array([1e-5, 0.5, 0.5])
        )
        assert result["pattern"] == "valley"

    def test_antagonistic_pattern(self):
        """High spread, some high some low -> antagonistic."""
        from peach._core.utils.pattern_classification import classify_single_feature

        betas = np.array([10.0, -2.0, 8.0, -3.0])
        result = classify_single_feature(
            betas, None, r2=0.7,
            p_betas=np.array([1e-10, 1e-5, 1e-10, 1e-5]),
            p_interactions=None
        )
        assert result["pattern"] == "antagonistic"


class TestClassifyAllFeatures:
    """Test batch classification."""

    def test_classifies_all(self):
        """classify_all_features returns list of dicts, one per feature."""
        from peach._core.utils.pattern_classification import classify_all_features

        n_features = 5
        K = 3
        vertex_coefficients = np.random.default_rng(42).standard_normal((n_features, K)) * 5
        r_squared = np.array([0.8, 0.01, 0.6, 0.7, 0.5])
        vertex_pvalues = np.full((n_features, K), 0.001)

        results = classify_all_features(
            vertex_coefficients, None, r_squared, vertex_pvalues, None
        )
        assert len(results) == n_features
        assert all(isinstance(r, dict) for r in results)
        assert all("pattern" in r for r in results)

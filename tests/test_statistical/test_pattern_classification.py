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

    def test_flat_by_cv(self):
        """Nearly equal betas with decent R^2 -> flat due to low CV."""
        from peach._core.utils.pattern_classification import classify_single_feature

        betas = np.array([5.0, 5.1, 4.9, 5.05])
        result = classify_single_feature(
            betas, None, r2=0.15, p_betas=np.array([0.01, 0.01, 0.01, 0.01]),
            p_interactions=None
        )
        assert result["pattern"] == "flat"
        assert result["details"]["reason"] == "low_cv"

    def test_gradient_pattern(self):
        """Two high betas with clear gap from low betas -> gradient."""
        from peach._core.utils.pattern_classification import classify_single_feature

        betas = np.array([8.0, 7.5, 2.0, 1.5])
        result = classify_single_feature(
            betas, None, r2=0.7,
            p_betas=np.array([1e-10, 1e-5, 0.01, 0.05]),
            p_interactions=None
        )
        assert result["pattern"] == "gradient"

    def test_shared_pattern_becomes_gradient(self):
        """Two high betas with gap from rest -> gradient (was multi-archetype-shared)."""
        from peach._core.utils.pattern_classification import classify_single_feature

        betas = np.array([8.0, 7.5, 1.0])
        result = classify_single_feature(
            betas, None, r2=0.6,
            p_betas=np.array([1e-10, 1e-10, 0.3]),
            p_interactions=None
        )
        assert result["pattern"] == "gradient"

    def test_monotonic_fallback(self):
        """Evenly spaced betas with no clear gap -> monotonic."""
        from peach._core.utils.pattern_classification import classify_single_feature

        betas = np.array([8.0, 6.0, 4.0, 2.0])
        result = classify_single_feature(
            betas, None, r2=0.5,
            p_betas=np.array([1e-10, 1e-5, 0.01, 0.05]),
            p_interactions=None
        )
        assert result["pattern"] == "monotonic"

    def test_negative_betas_not_exclusive(self):
        """Mixed positive/negative betas should NOT be classified as exclusive."""
        from peach._core.utils.pattern_classification import classify_single_feature

        # One positive, two negative -- should be gradient or monotonic, not exclusive
        betas = np.array([2.0, -1.5, -1.0])
        result = classify_single_feature(
            betas, None, r2=0.5,
            p_betas=np.array([0.01, 0.01, 0.01]),
        )
        assert result["pattern"] != "archetype-exclusive"

    def test_negative_betas_exclusive_when_dominant(self):
        """One very large |beta| with small others -> exclusive even with negatives."""
        from peach._core.utils.pattern_classification import classify_single_feature

        betas = np.array([10.0, -0.5, 0.3])
        result = classify_single_feature(
            betas, None, r2=0.8,
            p_betas=np.array([0.001, 0.5, 0.7]),
        )
        assert result["pattern"] == "archetype-exclusive"


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

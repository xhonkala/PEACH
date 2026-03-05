import numpy as np
import pytest


class TestPermutationTest:
    """Test generic permutation testing framework."""

    def test_significant_signal_detected(self):
        """Known signal should yield p < 0.05."""
        from peach._core.utils.resampling import permutation_test

        rng = np.random.default_rng(42)
        x = rng.normal(5.0, 1.0, size=200)  # strong signal away from 0

        def fit_fn(data):
            return np.mean(data)

        def stat_fn(model):
            return abs(model)

        def shuffle_fn(data, rng):
            return rng.choice([-1, 1], size=len(data)) * data  # sign-flip null

        result = permutation_test(
            fit_fn, stat_fn, x, shuffle_fn=shuffle_fn, n_permutations=199, seed=42
        )
        assert result["p_value"] < 0.05
        assert result["observed_stat"] > 4.0
        assert len(result["null_distribution"]) == 199

    def test_null_signal_not_detected(self):
        """No signal should yield p > 0.05."""
        from peach._core.utils.resampling import permutation_test

        rng = np.random.default_rng(42)
        x = rng.normal(0.0, 1.0, size=200)

        def fit_fn(data):
            return np.mean(data)

        def stat_fn(model):
            return abs(model)

        def shuffle_fn(data, rng):
            return rng.choice([-1, 1], size=len(data)) * data

        result = permutation_test(
            fit_fn, stat_fn, x, shuffle_fn=shuffle_fn, n_permutations=199, seed=42
        )
        assert result["p_value"] > 0.05


class TestBootstrapCI:
    """Test generic bootstrap CI framework."""

    def test_ci_covers_true_mean(self):
        """95% CI should cover the true mean for normal data."""
        from peach._core.utils.resampling import bootstrap_ci

        rng = np.random.default_rng(42)
        x = rng.normal(5.0, 1.0, size=500)

        def fit_fn(data):
            return np.mean(data)

        def stat_fn(model):
            return model

        result = bootstrap_ci(fit_fn, stat_fn, x, n_bootstrap=1000, seed=42)
        assert result["ci_lower"] < 5.0 < result["ci_upper"]
        assert len(result["bootstrap_distribution"]) == 1000

    def test_narrow_ci_with_low_variance(self):
        """Low-variance data should give narrow CIs."""
        from peach._core.utils.resampling import bootstrap_ci

        x = np.full(500, 3.0) + np.random.default_rng(42).normal(0, 0.01, 500)

        def fit_fn(data):
            return np.mean(data)

        def stat_fn(model):
            return model

        result = bootstrap_ci(fit_fn, stat_fn, x, n_bootstrap=500, seed=42)
        ci_width = result["ci_upper"] - result["ci_lower"]
        assert ci_width < 0.1

    def test_custom_ci_level(self):
        """90% CI should be narrower than 95% CI."""
        from peach._core.utils.resampling import bootstrap_ci

        rng = np.random.default_rng(42)
        x = rng.normal(0, 1.0, size=500)

        def fit_fn(data):
            return np.mean(data)

        def stat_fn(model):
            return model

        r95 = bootstrap_ci(fit_fn, stat_fn, x, ci_level=0.95, n_bootstrap=500, seed=42)
        r90 = bootstrap_ci(fit_fn, stat_fn, x, ci_level=0.90, n_bootstrap=500, seed=42)
        w95 = r95["ci_upper"] - r95["ci_lower"]
        w90 = r90["ci_upper"] - r90["ci_lower"]
        assert w90 < w95

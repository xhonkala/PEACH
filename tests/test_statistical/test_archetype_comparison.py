import numpy as np
import pytest


class TestOlsFitCovariance:
    def test_returns_covariance_when_requested(self):
        from peach._core.utils.simplex_regression import ols_fit, scheffe_design_matrix

        rng = np.random.default_rng(42)
        K = 3
        n = 200
        weights = rng.dirichlet([1] * K, size=n)
        W, _ = scheffe_design_matrix(weights, degree=1)
        Y = weights @ rng.standard_normal((K, 10)) + rng.normal(0, 0.1, (n, 10))

        result = ols_fit(W, Y, robust_se=True, return_covariance=True)
        assert "covariance" in result
        assert len(result["covariance"]) == 10
        assert result["covariance"][0].shape == (K, K)

    def test_no_covariance_by_default(self):
        from peach._core.utils.simplex_regression import ols_fit, scheffe_design_matrix

        rng = np.random.default_rng(42)
        K = 3
        n = 200
        weights = rng.dirichlet([1] * K, size=n)
        W, _ = scheffe_design_matrix(weights, degree=1)
        Y = weights @ rng.standard_normal((K, 10)) + rng.normal(0, 0.1, (n, 10))

        result = ols_fit(W, Y, robust_se=True)
        assert "covariance" not in result

    def test_classical_covariance(self):
        from peach._core.utils.simplex_regression import ols_fit, scheffe_design_matrix

        rng = np.random.default_rng(42)
        K = 3
        n = 200
        weights = rng.dirichlet([1] * K, size=n)
        W, _ = scheffe_design_matrix(weights, degree=1)
        Y = weights @ rng.standard_normal((K, 10)) + rng.normal(0, 0.1, (n, 10))

        result = ols_fit(W, Y, robust_se=False, return_covariance=True)
        assert "covariance" in result
        # Classical covariance should be symmetric positive semi-definite
        cov = result["covariance"][0]
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)

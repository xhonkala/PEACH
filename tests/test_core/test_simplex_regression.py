import numpy as np
import pytest
import scipy.sparse as sp


class TestScheffeDesignMatrix:
    """Test Scheffe polynomial design matrix construction."""

    def test_degree1_is_weights(self):
        """Degree 1 design matrix = weights themselves (no intercept)."""
        from peach._core.utils.simplex_regression import scheffe_design_matrix

        rng = np.random.default_rng(42)
        W = rng.dirichlet([1, 1, 1], size=100)
        X1, pairs = scheffe_design_matrix(W, degree=1)
        np.testing.assert_array_almost_equal(X1, W)
        assert pairs == []

    def test_degree2_adds_interactions(self):
        """Degree 2 appends K-choose-2 interaction columns."""
        from peach._core.utils.simplex_regression import scheffe_design_matrix

        rng = np.random.default_rng(42)
        K = 4
        W = rng.dirichlet([1] * K, size=100)
        X2, pairs = scheffe_design_matrix(W, degree=2)
        n_interactions = K * (K - 1) // 2  # 6
        assert X2.shape == (100, K + n_interactions)
        assert len(pairs) == n_interactions
        # First K columns are weights
        np.testing.assert_array_almost_equal(X2[:, :K], W)
        # Interaction columns are products
        for idx, (j, k) in enumerate(pairs):
            np.testing.assert_array_almost_equal(
                X2[:, K + idx], W[:, j] * W[:, k]
            )

    def test_degree1_no_pairs(self):
        from peach._core.utils.simplex_regression import scheffe_design_matrix

        rng = np.random.default_rng(42)
        W = rng.dirichlet([1, 1], size=50)
        _, pairs = scheffe_design_matrix(W, degree=1)
        assert pairs == []

    def test_k1_degree2_raises(self):
        """K=1 with degree=2 should raise ValueError (degree > K)."""
        import pytest
        from peach._core.utils.simplex_regression import scheffe_design_matrix

        W = np.ones((50, 1))
        with pytest.raises(ValueError, match="degree=2 exceeds K=1"):
            scheffe_design_matrix(W, degree=2)

    def test_k2_degree2(self):
        """K=2 with degree=2 should produce 1 interaction column."""
        from peach._core.utils.simplex_regression import scheffe_design_matrix

        rng = np.random.default_rng(42)
        W = rng.dirichlet([1, 1], size=50)
        X, pairs = scheffe_design_matrix(W, degree=2)
        assert X.shape == (50, 3)  # 2 + 1 interaction
        assert len(pairs) == 1


class TestOLSFit:
    """Test OLS regression engine with known coefficients."""

    def test_recovers_known_coefficients_dense(self):
        """OLS recovers planted beta from noiseless simplex regression."""
        from peach._core.utils.simplex_regression import ols_fit

        rng = np.random.default_rng(42)
        K = 3
        W = rng.dirichlet([1] * K, size=500)
        true_beta = np.array([[10.0, 2.0, 5.0]])  # 1 feature, 3 archetypes
        Y = W @ true_beta.T  # [500, 1]

        result = ols_fit(W, Y)
        np.testing.assert_array_almost_equal(result["coefficients"], true_beta, decimal=5)
        assert result["r_squared"][0] > 0.99

    def test_recovers_with_noise(self):
        """OLS R^2 is reasonable with moderate noise."""
        from peach._core.utils.simplex_regression import ols_fit

        rng = np.random.default_rng(42)
        K = 4
        n = 1000
        W = rng.dirichlet([1] * K, size=n)
        true_beta = rng.standard_normal((5, K)) * 3
        noise = rng.normal(0, 0.5, size=(n, 5))
        Y = W @ true_beta.T + noise

        result = ols_fit(W, Y)
        assert result["coefficients"].shape == (5, K)
        assert all(r2 > 0.1 for r2 in result["r_squared"])

    def test_sparse_feature_matrix(self):
        """OLS handles sparse Y via chunked processing."""
        from peach._core.utils.simplex_regression import ols_fit

        rng = np.random.default_rng(42)
        K = 3
        W = rng.dirichlet([1] * K, size=200)
        Y_dense = rng.random((200, 10))
        Y_sparse = sp.csr_matrix(Y_dense)

        result_dense = ols_fit(W, Y_dense)
        result_sparse = ols_fit(W, Y_sparse)
        np.testing.assert_array_almost_equal(
            result_dense["coefficients"], result_sparse["coefficients"], decimal=5
        )


class TestHC3StandardErrors:
    """Test heteroscedasticity-consistent standard errors."""

    def test_hc3_se_shape(self):
        """HC3 SEs have correct shape [n_features, p]."""
        from peach._core.utils.simplex_regression import ols_fit

        rng = np.random.default_rng(42)
        K = 3
        W = rng.dirichlet([1] * K, size=300)
        Y = rng.random((300, 10))

        result = ols_fit(W, Y, robust_se=True)
        assert result["standard_errors"].shape == (10, K)
        assert np.all(result["standard_errors"] > 0)

    def test_hc3_se_heteroscedastic_data(self):
        """HC3 SEs should be larger than OLS SEs with heteroscedastic noise."""
        from peach._core.utils.simplex_regression import ols_fit

        rng = np.random.default_rng(42)
        K = 3
        n = 500
        W = rng.dirichlet([1] * K, size=n)
        true_beta = np.array([[10.0, 2.0, 5.0]])
        # Strong heteroscedastic noise: variance proportional to w_0
        noise = rng.normal(0, 1, size=(n, 1)) * W[:, 0:1] * 5
        Y = W @ true_beta.T + noise

        result_hc3 = ols_fit(W, Y, robust_se=True)
        result_ols = ols_fit(W, Y, robust_se=False)
        # HC3 SEs must be larger (not just 80% of OLS)
        assert np.mean(result_hc3["standard_errors"]) > np.mean(result_ols["standard_errors"])


class TestFTest:
    """Test overall model F-test."""

    def test_significant_model(self):
        """Strong signal -> small p-value."""
        from peach._core.utils.simplex_regression import ols_fit

        rng = np.random.default_rng(42)
        K = 3
        W = rng.dirichlet([1] * K, size=500)
        true_beta = np.array([[10.0, 0.0, 0.0]])  # strong archetype-specific
        Y = W @ true_beta.T

        result = ols_fit(W, Y)
        assert result["f_pvalues"][0] < 1e-10

    def test_null_model(self):
        """Noise independent of weights -> large p-value, low R^2."""
        from peach._core.utils.simplex_regression import ols_fit

        rng = np.random.default_rng(42)
        K = 3
        W = rng.dirichlet([1] * K, size=500)
        # True null: Y is random noise unrelated to W (not just constant)
        Y = rng.standard_normal((500, 1))

        result = ols_fit(W, Y)
        assert result["f_pvalues"][0] > 0.05
        assert result["r_squared"][0] < 0.05

    def test_underdetermined_raises(self):
        """n < p should raise ValueError."""
        from peach._core.utils.simplex_regression import ols_fit

        rng = np.random.default_rng(42)
        W = rng.dirichlet([1, 1, 1], size=2)  # 2 cells, 3 params
        Y = rng.standard_normal((2, 5))
        with pytest.raises(ValueError, match="Underdetermined"):
            ols_fit(W, Y)

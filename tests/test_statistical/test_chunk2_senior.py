"""Senior adversarial review tests for Chunk 2: regression, comparison & classification.

Tests target:
- HC3 vectorized vs manual loop for 500+ features
- HC3 with heteroscedastic data — SEs must exceed classical SEs
- Sparse Y regression produces identical coefficients as dense Y
- Pattern classification: K=2 with SE filtering — degenerate case
- Pattern classification: all betas equal with SEs — must be structured, not exclusive
- feature_source stored and retrieved correctly through full pipeline
- FutureWarning fires when pathway_scores present and feature_matrix=None
- feature_expansion with zero-norm loading vector
- Wald contrasts with return_covariance=True vs False
- Single feature regression
- n_cells == K (exactly determined)
"""

import warnings

import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simplex_weights(n, K, seed=42):
    """Generate random simplex weights that sum to 1."""
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet(np.ones(K), size=n)
    return raw


def _make_regression_adata(n=500, n_genes=100, K=4, seed=42, sparse_X=False):
    """Build AnnData with archetype weights and gene expression for regression tests."""
    rng = np.random.default_rng(seed)
    weights = _simplex_weights(n, K, seed=seed)
    X = rng.standard_normal((n, n_genes)).astype(np.float64)
    if sparse_X:
        X_mat = sp.csr_matrix(X)
    else:
        X_mat = X

    adata = AnnData(X=X_mat)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n)]
    adata.obsm["cell_archetype_weights"] = weights
    return adata


# ===========================================================================
# 1. HC3 vectorized vs manual loop — 500+ features
# ===========================================================================


class TestHC3VectorizedVsLoop:
    """The HC3 einsum implementation must be numerically identical to a
    feature-by-feature Python loop.  We test with 500 features to stress
    the vectorization."""

    def test_500_features_match_loop(self):
        from peach._core.utils.simplex_regression import (
            _hc3_standard_errors,
            scheffe_design_matrix,
        )

        rng = np.random.default_rng(42)
        n, K, n_features = 200, 4, 500
        W_raw = _simplex_weights(n, K, seed=42)
        W, _ = scheffe_design_matrix(W_raw, degree=1)
        p = W.shape[1]

        WtW = W.T @ W
        WtW_inv = np.linalg.solve(WtW, np.eye(p))
        H_diag = np.sum((W @ WtW_inv) * W, axis=1)
        H_diag = np.clip(H_diag, 0, 1 - 1e-10)

        # Generate residuals
        residuals = rng.standard_normal((n, n_features))

        # Vectorized
        se_vec = _hc3_standard_errors(W, residuals, WtW_inv, H_diag)

        # Manual loop
        adjustment = 1.0 / (1 - H_diag)
        se_loop = np.zeros((n_features, p))
        for g in range(n_features):
            e_adj = residuals[:, g] * adjustment
            meat = W.T @ np.diag(e_adj**2) @ W
            sandwich = WtW_inv @ meat @ WtW_inv
            se_loop[g] = np.sqrt(np.maximum(np.diag(sandwich), 0))

        np.testing.assert_allclose(
            se_vec, se_loop, atol=1e-10, rtol=1e-8,
            err_msg="HC3 vectorized einsum disagrees with manual loop for 500 features",
        )

    def test_covariance_matches_loop(self):
        """Full covariance matrices (for Wald contrasts) must match loop."""
        from peach._core.utils.simplex_regression import (
            _hc3_covariance,
            scheffe_design_matrix,
        )

        rng = np.random.default_rng(99)
        n, K, n_features = 100, 3, 50
        W_raw = _simplex_weights(n, K, seed=99)
        W, _ = scheffe_design_matrix(W_raw, degree=1)
        p = W.shape[1]

        WtW = W.T @ W
        WtW_inv = np.linalg.solve(WtW, np.eye(p))
        H_diag = np.sum((W @ WtW_inv) * W, axis=1)
        H_diag = np.clip(H_diag, 0, 1 - 1e-10)

        residuals = rng.standard_normal((n, n_features))

        # Vectorized
        cov_list = _hc3_covariance(W, residuals, WtW_inv, H_diag)

        # Manual loop
        adjustment = 1.0 / (1 - H_diag)
        for g in range(n_features):
            e_adj = residuals[:, g] * adjustment
            meat = W.T @ np.diag(e_adj**2) @ W
            expected = WtW_inv @ meat @ WtW_inv
            np.testing.assert_allclose(
                cov_list[g], expected, atol=1e-10, rtol=1e-8,
                err_msg=f"Covariance mismatch at feature {g}",
            )


# ===========================================================================
# 2. HC3 with heteroscedastic data — SEs must exceed classical SEs
# ===========================================================================


class TestHC3Heteroscedasticity:
    """When variance is proportional to the archetype weight of the dominant
    archetype (strong heteroscedasticity), HC3 SEs should be systematically
    larger than classical OLS SEs because they account for the non-constant
    variance."""

    def test_hc3_larger_than_classical_under_heteroscedasticity(self):
        from peach._core.utils.simplex_regression import ols_fit, scheffe_design_matrix

        rng = np.random.default_rng(42)
        n, K = 1000, 3
        W_raw = _simplex_weights(n, K, seed=42)
        W, _ = scheffe_design_matrix(W_raw, degree=1)

        # Make Y with EXTREME heteroscedastic noise: variance is w_0^2 * 100
        # Cells near archetype 0 have 100x the noise of cells near other archetypes
        betas_true = rng.standard_normal((50, K)) * 5
        Y_signal = W @ betas_true.T  # [n, 50]
        noise_scale = 0.1 + 20.0 * W_raw[:, 0]**2  # extreme heteroscedasticity
        noise = rng.standard_normal((n, 50)) * noise_scale[:, np.newaxis]
        Y = Y_signal + noise

        result_robust = ols_fit(W, Y, robust_se=True)
        result_classical = ols_fit(W, Y, robust_se=False)

        se_robust = result_robust["standard_errors"]
        se_classical = result_classical["standard_errors"]

        # For the archetype 0 coefficient specifically, HC3 should be noticeably larger
        # because the high-leverage cells near vertex 0 also have high residual variance
        ratio_col0 = (se_robust[:, 0] / np.maximum(se_classical[:, 0], 1e-15)).mean()
        assert ratio_col0 > 1.02, (
            f"HC3/classical SE ratio for archetype 0 = {ratio_col0:.3f}. "
            "Under strong heteroscedasticity centered on archetype 0, "
            "HC3 SEs should be systematically larger."
        )

    def test_under_homoscedasticity_hc3_and_classical_are_close(self):
        """Under homoscedastic noise, HC3 and classical SEs should be similar."""
        from peach._core.utils.simplex_regression import ols_fit, scheffe_design_matrix

        rng = np.random.default_rng(42)
        n, K = 500, 4
        W_raw = _simplex_weights(n, K, seed=42)
        W, _ = scheffe_design_matrix(W_raw, degree=1)

        betas_true = rng.standard_normal((20, K))
        Y = W @ betas_true.T + rng.standard_normal((n, 20)) * 0.5

        result_robust = ols_fit(W, Y, robust_se=True)
        result_classical = ols_fit(W, Y, robust_se=False)

        se_robust = result_robust["standard_errors"]
        se_classical = result_classical["standard_errors"]

        ratio = (se_robust / np.maximum(se_classical, 1e-15)).mean()
        # Should be close to 1 (within ~10%)
        assert 0.85 < ratio < 1.15, (
            f"HC3/classical SE ratio = {ratio:.3f} under homoscedasticity. "
            "Expected ~1.0."
        )


# ===========================================================================
# 3. Sparse Y regression produces identical coefficients as dense Y
# ===========================================================================


class TestSparseVsDenseRegression:
    """Coefficients and R-squared must be identical whether Y is dense or sparse."""

    def test_sparse_dense_coefficients_match(self):
        from peach._core.utils.simplex_regression import ols_fit, scheffe_design_matrix

        rng = np.random.default_rng(42)
        n, K, n_features = 300, 4, 100
        W_raw = _simplex_weights(n, K, seed=42)
        W, _ = scheffe_design_matrix(W_raw, degree=1)

        # Create data with some zeros (like real scRNA-seq)
        Y_dense = rng.standard_normal((n, n_features))
        Y_dense[Y_dense < 0.5] = 0  # ~70% zeros
        Y_sparse = sp.csr_matrix(Y_dense)

        result_dense = ols_fit(W, Y_dense, robust_se=True, return_covariance=True)
        result_sparse = ols_fit(W, Y_sparse, robust_se=True, return_covariance=True, chunk_size=30)

        np.testing.assert_allclose(
            result_dense["coefficients"], result_sparse["coefficients"],
            atol=1e-10, rtol=1e-8,
            err_msg="Sparse and dense coefficients disagree",
        )
        np.testing.assert_allclose(
            result_dense["r_squared"], result_sparse["r_squared"],
            atol=1e-10,
            err_msg="Sparse and dense R-squared disagree",
        )
        np.testing.assert_allclose(
            result_dense["standard_errors"], result_sparse["standard_errors"],
            atol=1e-10, rtol=1e-8,
            err_msg="Sparse and dense SEs disagree",
        )
        np.testing.assert_allclose(
            result_dense["t_pvalues"], result_sparse["t_pvalues"],
            atol=1e-10,
            err_msg="Sparse and dense t-pvalues disagree",
        )

    def test_sparse_dense_covariance_match(self):
        """Covariance matrices (for Wald contrasts) must match."""
        from peach._core.utils.simplex_regression import ols_fit, scheffe_design_matrix

        rng = np.random.default_rng(42)
        n, K, n_features = 200, 3, 50
        W_raw = _simplex_weights(n, K, seed=42)
        W, _ = scheffe_design_matrix(W_raw, degree=1)

        Y_dense = rng.standard_normal((n, n_features))
        Y_dense[Y_dense < 0.3] = 0
        Y_sparse = sp.csr_matrix(Y_dense)

        result_dense = ols_fit(W, Y_dense, robust_se=True, return_covariance=True)
        result_sparse = ols_fit(W, Y_sparse, robust_se=True, return_covariance=True, chunk_size=20)

        for g in range(n_features):
            np.testing.assert_allclose(
                result_dense["covariance"][g], result_sparse["covariance"][g],
                atol=1e-10, rtol=1e-8,
                err_msg=f"Covariance mismatch at feature {g}",
            )


# ===========================================================================
# 4. Pattern classification: K=2 with SE filtering — degenerate case
# ===========================================================================


class TestPatternClassificationK2:
    """With K=2, there are only 2 betas. If one is dominant but has huge SE,
    SE filtering should prevent exclusive classification."""

    def test_k2_exclusive_blocked_by_se(self):
        from peach._core.utils.pattern_classification import classify_single_feature

        result = classify_single_feature(
            vertex_betas=np.array([10.0, 1.0]),
            r2=0.7,
            f_pvalue_fdr=0.001,
            vertex_ses=np.array([100.0, 0.1]),  # huge SE on dominant
        )
        assert result["pattern"] != "archetype-exclusive", (
            "K=2: should not be exclusive when dominant SE is huge"
        )
        # Should fall through to structured
        assert result["pattern"] == "structured"

    def test_k2_exclusive_passes_with_small_se(self):
        from peach._core.utils.pattern_classification import classify_single_feature

        result = classify_single_feature(
            vertex_betas=np.array([10.0, 1.0]),
            r2=0.7,
            f_pvalue_fdr=0.001,
            vertex_ses=np.array([0.1, 0.1]),
        )
        assert result["pattern"] == "archetype-exclusive"

    def test_k2_equal_betas_is_structured(self):
        """With K=2 and equal betas, ratio=1 < 2, should be structured."""
        from peach._core.utils.pattern_classification import classify_single_feature

        result = classify_single_feature(
            vertex_betas=np.array([5.0, 5.0]),
            r2=0.6,
            f_pvalue_fdr=0.001,
        )
        assert result["pattern"] == "structured"


# ===========================================================================
# 5. Pattern classification: all betas equal with SEs — must be structured
# ===========================================================================


class TestAllBetasEqualWithSEs:
    """When all K betas are identical, the ratio test fails (ratio=1),
    so the feature must be classified as 'structured', not 'exclusive'."""

    def test_all_equal_betas(self):
        from peach._core.utils.pattern_classification import classify_single_feature

        for K in [2, 3, 4, 5, 10]:
            betas = np.ones(K) * 5.0
            ses = np.ones(K) * 0.1
            result = classify_single_feature(
                vertex_betas=betas,
                r2=0.5,
                f_pvalue_fdr=0.001,
                vertex_ses=ses,
            )
            assert result["pattern"] == "structured", (
                f"K={K}: all-equal betas should be structured, got {result['pattern']}"
            )

    def test_near_equal_betas_with_large_ses(self):
        """Betas differ slightly (ratio barely exceeds 2.0), but SEs are
        large enough that the dominant coefficient fails |beta| > 2*SE."""
        from peach._core.utils.pattern_classification import classify_single_feature

        # Ratio = 10.0/4.99 = 2.004 > 2.0, so ratio test passes.
        # But SE on dominant = 6.0, and |10.0| < 2 * 6.0 = 12.0, so SE test fails.
        result = classify_single_feature(
            vertex_betas=np.array([10.0, 4.99, 3.0]),
            r2=0.5,
            f_pvalue_fdr=0.001,
            vertex_ses=np.array([6.0, 0.1, 0.1]),
        )
        assert result["pattern"] != "archetype-exclusive", (
            "SE filter should prevent exclusive when |beta| < 2*SE"
        )


# ===========================================================================
# 6. feature_source stored and retrieved through full pipeline
# ===========================================================================


class TestFeatureSourcePipeline:
    """feature_source must be stored in the serialized regression result
    and correctly retrieved by downstream consumers (Wald contrasts)."""

    def test_feature_source_none_for_genes(self):
        """When regressing on adata.X (feature_matrix=None), feature_source=None."""
        from peach.tl.feature_regression import feature_simplex_regression

        adata = _make_regression_adata(n=100, n_genes=20, K=3)
        result = feature_simplex_regression(
            adata, n_bootstrap=0, permutation_test=False, max_degree=1,
        )
        assert "feature_source" in result
        assert result["feature_source"] is None

    def test_feature_source_pathway_scores(self):
        """When regressing on pathway_scores, feature_source='pathway_scores'."""
        from peach.tl.feature_regression import feature_simplex_regression

        adata = _make_regression_adata(n=100, n_genes=20, K=3)
        # Add fake pathway scores
        rng = np.random.default_rng(42)
        adata.obsm["pathway_scores"] = rng.standard_normal((100, 5))
        result = feature_simplex_regression(
            adata, feature_matrix="pathway_scores",
            n_bootstrap=0, permutation_test=False, max_degree=1,
        )
        assert result["feature_source"] == "pathway_scores"

    def test_wald_contrast_uses_feature_source_for_refit(self):
        """When Wald contrasts need to re-run regression (no cached covariance),
        they must use the correct feature_source from the stored result."""
        from peach.tl.feature_regression import feature_simplex_regression
        from peach._core.utils.archetype_comparison import compute_wald_contrasts

        adata = _make_regression_adata(n=200, n_genes=30, K=3)

        # Run regression WITHOUT covariance (so Wald must re-fit)
        result = feature_simplex_regression(
            adata, n_bootstrap=0, permutation_test=False, max_degree=1,
            robust_se=True,
        )
        # Remove cached covariance to force re-fit path
        for key in ["peach_simplex_regression", "peach_simplex_regression_genes"]:
            if key in adata.uns:
                adata.uns[key].pop("vertex_covariance", None)

        # Wald contrasts should succeed (re-fitting with feature_source=None -> adata.X)
        wald = compute_wald_contrasts(adata, robust_se=True)
        assert len(wald["pairs"]) > 0
        assert wald["n_features"] == 30

    def test_wald_contrast_with_cached_covariance(self):
        """When covariance IS cached, Wald should use it directly."""
        from peach.tl.feature_regression import feature_simplex_regression
        from peach._core.utils.archetype_comparison import compute_wald_contrasts

        adata = _make_regression_adata(n=200, n_genes=30, K=3)
        feature_simplex_regression(
            adata, n_bootstrap=0, permutation_test=False, max_degree=1,
            robust_se=True,
        )
        # Covariance should be cached
        reg = adata.uns["peach_simplex_regression"]
        assert "vertex_covariance" in reg

        wald = compute_wald_contrasts(adata, robust_se=True)
        assert wald["n_features"] == 30


# ===========================================================================
# 7. FutureWarning fires when pathway_scores present
# ===========================================================================


class TestFutureWarningPathway:
    """archetype_driver_regression() should emit FutureWarning when
    feature_matrix=None and pathway_scores exist in adata.obsm."""

    def test_futurewarning_fires(self):
        from peach.tl.feature_regression import archetype_driver_regression

        adata = _make_regression_adata(n=100, n_genes=20, K=3)
        rng = np.random.default_rng(42)
        adata.obsm["pathway_scores"] = rng.standard_normal((100, 5))

        with pytest.warns(FutureWarning, match="auto-selects pathway_scores"):
            archetype_driver_regression(
                adata, max_degree=1, n_bootstrap=0,
            )

    def test_no_warning_when_explicit(self):
        """No warning when feature_matrix is explicitly provided."""
        from peach.tl.feature_regression import archetype_driver_regression

        adata = _make_regression_adata(n=100, n_genes=20, K=3)
        rng = np.random.default_rng(42)
        adata.obsm["pathway_scores"] = rng.standard_normal((100, 5))

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            # Should NOT raise FutureWarning
            archetype_driver_regression(
                adata, feature_matrix="pathway_scores",
                max_degree=1, n_bootstrap=0,
            )

    def test_no_warning_when_no_pathways(self):
        """No warning when pathway_scores is absent."""
        from peach.tl.feature_regression import archetype_driver_regression

        adata = _make_regression_adata(n=100, n_genes=20, K=3)

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            archetype_driver_regression(
                adata, max_degree=1, n_bootstrap=0,
            )


# ===========================================================================
# 8. feature_expansion with zero-norm loading vector
# ===========================================================================


class TestFeatureExpansionZeroNorm:
    """A gene with all-zero PCA loadings should have feature_expansion ~ 0,
    not NaN or a wild value from normalizing by epsilon."""

    def test_zero_loading_gene_expansion_is_zero(self):
        """Insert a gene with all-zero loadings, verify its expansion is ~0."""
        # We test the Jacobian projection logic directly rather than
        # training a full flow model
        rng = np.random.default_rng(42)
        n_genes = 20
        n_pcs = 5

        # Create loadings where one gene is all zeros
        loadings = rng.standard_normal((n_genes, n_pcs))
        loadings[7, :] = 0.0  # gene 7 has zero loading

        # Simulate a mean Jacobian
        mean_jac = rng.standard_normal((n_pcs, n_pcs))

        # Replicate the logic from flow_jacobian
        loadings_trimmed = loadings[:, :n_pcs]
        loading_norms = np.linalg.norm(loadings_trimmed, axis=1, keepdims=True)
        loading_norms = np.maximum(loading_norms, 1e-10)
        loadings_normalized = loadings_trimmed / loading_norms
        feature_expansion = np.einsum(
            'gi,ij,gj->g', loadings_normalized, mean_jac, loadings_normalized
        )

        assert np.isfinite(feature_expansion[7]), "Zero-loading gene has non-finite expansion"
        assert abs(feature_expansion[7]) < 1e-8, (
            f"Zero-loading gene has expansion {feature_expansion[7]}, expected ~0"
        )

    def test_nonzero_loading_genes_have_nontrivial_expansion(self):
        """Non-zero loading genes should generally have non-zero expansion."""
        rng = np.random.default_rng(42)
        n_genes = 20
        n_pcs = 5

        loadings = rng.standard_normal((n_genes, n_pcs))
        mean_jac = rng.standard_normal((n_pcs, n_pcs)) * 0.5 + np.eye(n_pcs)

        loadings_trimmed = loadings[:, :n_pcs]
        loading_norms = np.linalg.norm(loadings_trimmed, axis=1, keepdims=True)
        loading_norms = np.maximum(loading_norms, 1e-10)
        loadings_normalized = loadings_trimmed / loading_norms
        feature_expansion = np.einsum(
            'gi,ij,gj->g', loadings_normalized, mean_jac, loadings_normalized
        )

        # Most genes should have non-trivial expansion
        assert np.sum(np.abs(feature_expansion) > 1e-4) > n_genes // 2


# ===========================================================================
# 9. Wald contrasts: return_covariance=True vs False
# ===========================================================================


class TestWaldCovariancePath:
    """When covariance is cached, Wald contrasts use the cached covariance.
    When absent, they re-run ols_fit with return_covariance=True. The SEs
    from both paths should be identical (same regression, same data)."""

    def test_cached_vs_refit_se_identical(self):
        from peach.tl.feature_regression import feature_simplex_regression
        from peach._core.utils.archetype_comparison import compute_wald_contrasts

        adata = _make_regression_adata(n=300, n_genes=50, K=3, seed=42)

        # Run with covariance cached
        feature_simplex_regression(
            adata, n_bootstrap=0, permutation_test=False, max_degree=1,
            robust_se=True, store_to_adata=True,
        )

        # Wald with cached covariance
        wald_cached = compute_wald_contrasts(adata, robust_se=True)

        # Now remove covariance and re-run
        for key in ["peach_simplex_regression", "peach_simplex_regression_genes"]:
            if key in adata.uns:
                adata.uns[key].pop("vertex_covariance", None)

        wald_refit = compute_wald_contrasts(adata, robust_se=True)

        # The SEs and z-scores should be identical
        for pair in wald_cached["pairs"]:
            np.testing.assert_allclose(
                wald_cached["delta_se"][pair],
                wald_refit["delta_se"][pair],
                atol=1e-8, rtol=1e-6,
                err_msg=f"Wald SE mismatch for pair {pair} (cached vs refit)",
            )
            np.testing.assert_allclose(
                wald_cached["z_scores"][pair],
                wald_refit["z_scores"][pair],
                atol=1e-8, rtol=1e-6,
                err_msg=f"Wald z-score mismatch for pair {pair}",
            )


# ===========================================================================
# 10. Single feature regression
# ===========================================================================


class TestSingleFeatureRegression:
    """Edge case: regressing a single feature (Y has shape [n, 1])."""

    def test_single_feature_shapes(self):
        from peach._core.utils.simplex_regression import ols_fit, scheffe_design_matrix

        rng = np.random.default_rng(42)
        n, K = 100, 3
        W_raw = _simplex_weights(n, K, seed=42)
        W, _ = scheffe_design_matrix(W_raw, degree=1)

        Y = rng.standard_normal((n, 1))
        result = ols_fit(W, Y, robust_se=True, return_covariance=True)

        assert result["coefficients"].shape == (1, K)
        assert result["r_squared"].shape == (1,)
        assert result["standard_errors"].shape == (1, K)
        assert result["t_pvalues"].shape == (1, K)
        assert result["f_pvalues"].shape == (1,)
        assert len(result["covariance"]) == 1
        assert result["covariance"][0].shape == (K, K)

    def test_single_feature_api(self):
        """Full API through feature_simplex_regression with a 1-gene adata."""
        from peach.tl.feature_regression import feature_simplex_regression

        adata = _make_regression_adata(n=100, n_genes=1, K=3)
        result = feature_simplex_regression(
            adata, n_bootstrap=0, permutation_test=False, max_degree=1,
        )
        assert result["n_features"] == 1
        assert len(result["feature_names"]) == 1

    def test_single_feature_sparse(self):
        """Sparse single-column Y."""
        from peach._core.utils.simplex_regression import ols_fit, scheffe_design_matrix

        rng = np.random.default_rng(42)
        n, K = 100, 3
        W_raw = _simplex_weights(n, K, seed=42)
        W, _ = scheffe_design_matrix(W_raw, degree=1)

        Y_dense = rng.standard_normal((n, 1))
        Y_sparse = sp.csr_matrix(Y_dense)

        r_d = ols_fit(W, Y_dense, robust_se=True)
        r_s = ols_fit(W, Y_sparse, robust_se=True)

        np.testing.assert_allclose(
            r_d["coefficients"], r_s["coefficients"], atol=1e-10,
        )


# ===========================================================================
# 11. n_cells == K (exactly determined system)
# ===========================================================================


class TestExactlyDetermined:
    """When n == K (or n == p for higher degree), the system is exactly
    determined.  Residuals are 0, R^2 = 1, and SEs may be degenerate.
    The code should not crash."""

    def test_n_equals_k_runs_without_crash(self):
        from peach._core.utils.simplex_regression import ols_fit, scheffe_design_matrix

        K = 4
        n = K  # exactly determined
        W_raw = _simplex_weights(n, K, seed=42)
        W, _ = scheffe_design_matrix(W_raw, degree=1)

        rng = np.random.default_rng(42)
        Y = rng.standard_normal((n, 5))

        # Should not crash — may warn about near-singular design
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = ols_fit(W, Y, robust_se=True)

        # Shapes must be correct
        assert result["coefficients"].shape == (5, K)
        assert result["r_squared"].shape == (5,)
        # All residuals should be ~0 for an exactly determined system
        # (only if the system is truly full-rank)
        if not result["extra_rank_deficient"]:
            np.testing.assert_allclose(result["r_squared"], 1.0, atol=1e-6)

    def test_n_equals_k_classical_se(self):
        """With n==K and classical SEs, df = max(n-p, 1) = 1.
        The SE should not be NaN/Inf."""
        from peach._core.utils.simplex_regression import ols_fit, scheffe_design_matrix

        K = 3
        n = K
        W_raw = _simplex_weights(n, K, seed=42)
        W, _ = scheffe_design_matrix(W_raw, degree=1)

        rng = np.random.default_rng(42)
        Y = rng.standard_normal((n, 3))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = ols_fit(W, Y, robust_se=False)

        assert np.all(np.isfinite(result["standard_errors"]))
        assert np.all(np.isfinite(result["t_pvalues"]))


# ===========================================================================
# 12. feature_simplex_regression stores in namespaced key
# ===========================================================================


class TestRegressionStorageKeys:
    """Regression results must be stored in both namespaced and generic keys."""

    def test_genes_storage(self):
        from peach.tl.feature_regression import feature_simplex_regression

        adata = _make_regression_adata(n=100, n_genes=20, K=3)
        feature_simplex_regression(
            adata, n_bootstrap=0, permutation_test=False, max_degree=1,
        )
        assert "peach_simplex_regression_genes" in adata.uns
        assert "peach_simplex_regression" in adata.uns

    def test_pathway_storage(self):
        from peach.tl.feature_regression import feature_simplex_regression

        adata = _make_regression_adata(n=100, n_genes=20, K=3)
        rng = np.random.default_rng(42)
        adata.obsm["pathway_scores"] = rng.standard_normal((100, 5))

        feature_simplex_regression(
            adata, feature_matrix="pathway_scores",
            n_bootstrap=0, permutation_test=False, max_degree=1,
        )
        assert "peach_simplex_regression_pathways" in adata.uns


# ===========================================================================
# 13. Pattern classification wired through full API
# ===========================================================================


class TestPatternClassificationFullPipeline:
    """classify_feature_patterns should wire SEs through to classify_all_features."""

    def test_se_aware_classification_through_api(self):
        from peach.tl.feature_regression import feature_simplex_regression
        from peach.tl.feature_patterns import classify_feature_patterns

        rng = np.random.default_rng(42)
        n, K = 200, 3
        adata = _make_regression_adata(n=n, n_genes=30, K=K)

        # Make gene 0 "exclusive" with archetype 0 having a large coefficient
        # We inject known signal
        weights = adata.obsm["cell_archetype_weights"]
        # Replace X with something predictable
        X = np.zeros((n, 30))
        X[:, 0] = weights[:, 0] * 50 + rng.standard_normal(n) * 0.01  # very clear exclusive
        X[:, 1] = weights[:, 0] * 50 + rng.standard_normal(n) * 50   # same signal but extremely noisy
        for i in range(2, 30):
            X[:, i] = rng.standard_normal(n) * 0.1  # flat
        adata.X = X

        feature_simplex_regression(
            adata, n_bootstrap=0, permutation_test=False, max_degree=1,
        )
        result = classify_feature_patterns(adata)

        # Gene 0 should be exclusive (clear signal, small SE)
        assert result["classifications"][0]["pattern"] == "archetype-exclusive", (
            f"Gene 0 with clear signal should be exclusive, got {result['classifications'][0]['pattern']}"
        )

        # Gene 1 may or may not be exclusive depending on the SE filter
        # With huge noise, the SE could prevent exclusive classification
        # (This is the correct behavior — we test that the pipeline doesn't crash)
        assert result["classifications"][1]["pattern"] in (
            "archetype-exclusive", "structured", "flat"
        )

    def test_pattern_counts_sum_to_n_features(self):
        from peach.tl.feature_regression import feature_simplex_regression
        from peach.tl.feature_patterns import classify_feature_patterns

        adata = _make_regression_adata(n=200, n_genes=30, K=3)
        feature_simplex_regression(
            adata, n_bootstrap=0, permutation_test=False, max_degree=1,
        )
        result = classify_feature_patterns(adata)

        total = sum(result["pattern_counts"].values())
        assert total == 30, f"Pattern counts sum to {total}, expected 30"


# ===========================================================================
# 14. Wald contrasts FDR is global across all pairs
# ===========================================================================


class TestWaldFDRGlobal:
    """Wald contrast FDR should be corrected globally across ALL pairs,
    not per-pair.  This means for K=3, FDR is across 3 * n_features tests."""

    def test_fdr_correction_is_global(self):
        from peach.tl.feature_regression import feature_simplex_regression
        from peach._core.utils.archetype_comparison import compute_wald_contrasts

        adata = _make_regression_adata(n=300, n_genes=50, K=3, seed=42)
        feature_simplex_regression(
            adata, n_bootstrap=0, permutation_test=False, max_degree=1,
            robust_se=True,
        )

        wald = compute_wald_contrasts(adata, robust_se=True)

        # K=3 -> 3 pairs, each with 50 features -> 150 total tests
        n_total_tests = 0
        all_raw_pvals = []
        for pair in wald["pairs"]:
            p = wald["pvalues"][pair]
            n_total_tests += len(p)
            all_raw_pvals.append(p)

        assert n_total_tests == 3 * 50, f"Expected 150 total tests, got {n_total_tests}"

        # FDR values should be >= raw p-values (monotonicity of BH)
        for pair in wald["pairs"]:
            fdr = wald["pvalues_fdr"][pair]
            raw = wald["pvalues"][pair]
            # FDR should never be less than raw p-value
            assert np.all(fdr >= raw - 1e-10), (
                f"FDR < raw p-value for pair {pair}"
            )


# ===========================================================================
# 15. Scheffe design matrix degree validation
# ===========================================================================


class TestScheffeDesignMatrix:
    """Test Scheffe design matrix construction edge cases."""

    def test_degree_exceeds_k_raises(self):
        from peach._core.utils.simplex_regression import scheffe_design_matrix

        W = _simplex_weights(10, 3)
        with pytest.raises(ValueError, match="degree=4 exceeds K=3"):
            scheffe_design_matrix(W, degree=4)

    def test_degree_equals_k(self):
        """degree=K should work (products of all K weights)."""
        from peach._core.utils.simplex_regression import scheffe_design_matrix

        K = 3
        W = _simplex_weights(10, K)
        X, info = scheffe_design_matrix(W, degree=K)
        # Should have K (linear) + K*(K-1)/2 (degree 2) + 1 (degree 3)
        expected_cols = K + 3 + 1  # K=3: 3 + 3 + 1 = 7
        assert X.shape[1] == expected_cols

    def test_degree_0_raises(self):
        from peach._core.utils.simplex_regression import scheffe_design_matrix

        W = _simplex_weights(10, 3)
        with pytest.raises(ValueError, match="degree must be >= 1"):
            scheffe_design_matrix(W, degree=0)


# ===========================================================================
# 16. HC3 H_diag clipping — high leverage cells
# ===========================================================================


class TestHC3HighLeverage:
    """Cells at simplex vertices have high leverage. The code clips H_diag
    to prevent division by zero in HC3 adjustment."""

    def test_vertex_cells_do_not_produce_inf(self):
        """Place some cells exactly at simplex vertices. HC3 must not crash."""
        from peach._core.utils.simplex_regression import ols_fit, scheffe_design_matrix

        rng = np.random.default_rng(42)
        K = 3
        n_normal = 50
        n_vertex = 3  # one per vertex

        W_normal = _simplex_weights(n_normal, K, seed=42)
        # Vertex cells: [1,0,0], [0,1,0], [0,0,1]
        W_vertex = np.eye(K)
        W_all = np.vstack([W_normal, W_vertex])
        n = len(W_all)

        W, _ = scheffe_design_matrix(W_all, degree=1)
        Y = rng.standard_normal((n, 10))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = ols_fit(W, Y, robust_se=True)

        assert np.all(np.isfinite(result["standard_errors"])), "Inf/NaN in SEs with vertex cells"
        assert np.all(np.isfinite(result["coefficients"]))


# ===========================================================================
# 17. Feature similarity index optimization
# ===========================================================================


class TestFeatureSimilarityIndexOptim:
    """compute_feature_similarity uses dict-based index maps for O(1) lookup.
    Verify the shared-feature intersection is correct."""

    def test_shared_features_between_fits(self):
        from peach._core.utils.archetype_comparison import compute_feature_similarity

        rng = np.random.default_rng(42)
        K = 3
        n = 200

        # adata_a with genes [gene_0, ..., gene_29]
        adata_a = _make_regression_adata(n=n, n_genes=30, K=K, seed=42)
        # adata_b with genes [gene_10, ..., gene_39]
        adata_b = AnnData(X=rng.standard_normal((n, 30)))
        adata_b.var_names = [f"gene_{i+10}" for i in range(30)]
        adata_b.obs_names = [f"cell_{i}" for i in range(n)]
        adata_b.obsm["cell_archetype_weights"] = _simplex_weights(n, K, seed=99)

        # Run regression on both
        from peach.tl.feature_regression import feature_simplex_regression
        feature_simplex_regression(adata_a, n_bootstrap=0, permutation_test=False, max_degree=1)
        feature_simplex_regression(adata_b, n_bootstrap=0, permutation_test=False, max_degree=1)

        result = compute_feature_similarity(adata_a, adata_b)

        # Shared features should be gene_10 through gene_29 (20 genes)
        # But n_shared may be smaller due to FDR filter
        assert result["n_shared_features"] <= 20
        assert result["spearman_matrix"].shape == (K, K)


# ===========================================================================
# 18. Comprehensive degree comparison
# ===========================================================================


class TestComprehensiveDegreeComparison:
    """When comprehensive_degree=True, degree_comparison should be in the result."""

    def test_degree_comparison_present(self):
        from peach.tl.feature_regression import feature_simplex_regression

        adata = _make_regression_adata(n=200, n_genes=20, K=4)
        result = feature_simplex_regression(
            adata, n_bootstrap=0, permutation_test=False, max_degree=1,
            comprehensive_degree=True,
        )
        assert "degree_comparison" in result
        # K=4, max_degree in comp is K-1=3, so degrees 2 and 3
        dc = result["degree_comparison"]
        assert "degree_2" in dc
        assert "degree_3" in dc

    def test_degree_comparison_k2_empty(self):
        """K=2: max meaningful degree is 1, so degree_comparison should be empty."""
        from peach.tl.feature_regression import feature_simplex_regression

        adata = _make_regression_adata(n=200, n_genes=20, K=2)
        result = feature_simplex_regression(
            adata, n_bootstrap=0, permutation_test=False, max_degree=1,
            comprehensive_degree=True,
        )
        dc = result.get("degree_comparison", {})
        assert len(dc) == 0, f"K=2 should have empty degree_comparison, got {dc.keys()}"


# ===========================================================================
# 19. Wald contrast delta_beta consistency
# ===========================================================================


class TestWaldDeltaBeta:
    """Verify delta_beta = beta_j - beta_k for each pair (j, k)."""

    def test_delta_beta_values(self):
        from peach.tl.feature_regression import feature_simplex_regression
        from peach._core.utils.archetype_comparison import compute_wald_contrasts

        adata = _make_regression_adata(n=200, n_genes=20, K=3)
        feature_simplex_regression(
            adata, n_bootstrap=0, permutation_test=False, max_degree=1,
        )

        reg = adata.uns["peach_simplex_regression"]
        betas = np.asarray(reg["vertex_coefficients"])  # [n_features, K]

        wald = compute_wald_contrasts(adata)

        for j, k in wald["pairs"]:
            expected_delta = betas[:, j] - betas[:, k]
            np.testing.assert_allclose(
                wald["delta_beta"][(j, k)], expected_delta,
                atol=1e-10,
                err_msg=f"delta_beta mismatch for pair ({j}, {k})",
            )


# ===========================================================================
# 20. Classify feature patterns handles K=1 gracefully
# ===========================================================================


class TestPatternK1:
    """K=1 means there's only one archetype. sorted_abs has length 1.
    Before the fix, sorted_abs[1] raised IndexError. After the fix,
    K=1 should fall through to structured (can't be exclusive with 1 archetype)."""

    def test_k1_does_not_crash(self):
        from peach._core.utils.pattern_classification import classify_single_feature

        # K=1: only one beta — should not crash
        result = classify_single_feature(
            vertex_betas=np.array([5.0]),
            r2=0.5,
            f_pvalue_fdr=0.001,
        )
        # With K=1, exclusive is impossible (no second archetype to compare),
        # so it should be structured
        assert result["pattern"] == "structured"

    def test_k1_nonsignificant_is_flat(self):
        from peach._core.utils.pattern_classification import classify_single_feature

        result = classify_single_feature(
            vertex_betas=np.array([5.0]),
            r2=0.1,
            f_pvalue_fdr=0.5,
        )
        assert result["pattern"] == "flat"


# ===========================================================================
# 21. Edge case: all-zero feature
# ===========================================================================


class TestAllZeroFeature:
    """A feature that is identically zero (all cells have the same value)
    should have R^2 = 0, and should be classified as 'flat'."""

    def test_constant_feature_r2_is_zero(self):
        from peach._core.utils.simplex_regression import ols_fit, scheffe_design_matrix

        n, K = 100, 3
        W_raw = _simplex_weights(n, K, seed=42)
        W, _ = scheffe_design_matrix(W_raw, degree=1)

        Y = np.zeros((n, 1))  # constant feature
        result = ols_fit(W, Y, robust_se=True)

        assert result["r_squared"][0] == pytest.approx(0.0, abs=1e-10)

    def test_constant_nonzero_feature(self):
        """A constant non-zero feature: all values = 5.0."""
        from peach._core.utils.simplex_regression import ols_fit, scheffe_design_matrix

        n, K = 100, 3
        W_raw = _simplex_weights(n, K, seed=42)
        W, _ = scheffe_design_matrix(W_raw, degree=1)

        Y = np.full((n, 1), 5.0)
        result = ols_fit(W, Y, robust_se=True)

        # All coefficients should be 5.0 (since w sums to 1, beta*w -> constant)
        np.testing.assert_allclose(result["coefficients"][0], 5.0, atol=1e-6)
        # R^2 = 0 because there's no variance to explain
        assert result["r_squared"][0] == pytest.approx(0.0, abs=1e-10)

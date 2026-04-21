import numpy as np
import pytest
from anndata import AnnData


@pytest.fixture
def regression_adata():
    """AnnData with known archetypal structure for regression testing."""
    rng = np.random.default_rng(42)
    K = 3
    n = 500
    n_genes = 50

    weights = rng.dirichlet([1] * K, size=n)
    true_beta = rng.standard_normal((n_genes, K)) * 5
    true_beta[0] = [10.0, 0.0, 0.0]  # exclusive
    true_beta[1] = [3.0, 3.0, 3.0]  # flat
    true_beta[2] = [8.0, 4.0, 1.0]  # gradient

    noise = rng.normal(0, 0.3, size=(n, n_genes))
    X = weights @ true_beta.T + noise

    adata = AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obsm["cell_archetype_weights"] = weights
    return adata


class TestFeatureSimplexRegression:

    def test_basic_run(self, regression_adata):
        """Runs without error, stores result in adata.uns."""
        import peach as pc
        result = pc.tl.feature_simplex_regression(regression_adata, n_bootstrap=0)
        assert "peach_simplex_regression" in regression_adata.uns
        assert result is not None
        assert np.asarray(result["vertex_coefficients"]).shape == (50, 3)
        assert len(result["r_squared_degree1"]) == 50

    def test_recovers_exclusive_gene(self, regression_adata):
        """Gene 0 (archetype-exclusive) should have high beta_0, low others."""
        import peach as pc
        result = pc.tl.feature_simplex_regression(regression_adata, n_bootstrap=0)
        gene0_betas = np.asarray(result["vertex_coefficients"])[0]
        assert gene0_betas[0] > 8.0
        assert gene0_betas[1] < 2.0
        assert gene0_betas[2] < 2.0

    def test_flat_gene_low_r2(self, regression_adata):
        """Gene 1 (flat) should have low R^2."""
        import peach as pc
        result = pc.tl.feature_simplex_regression(regression_adata, n_bootstrap=0)
        assert result["r_squared_degree1"][1] < 0.1

    def test_degree2_adds_interactions(self, regression_adata):
        """max_degree=2 produces interaction coefficients."""
        import peach as pc
        result = pc.tl.feature_simplex_regression(regression_adata, max_degree=2, n_bootstrap=0)
        assert result.get("interaction_coefficients") is not None
        K = 3
        n_interactions = K * (K - 1) // 2
        assert np.asarray(result["interaction_coefficients"]).shape == (50, n_interactions)

    def test_residuals_stored(self, regression_adata):
        """store_residuals=True puts residuals in obsm."""
        import peach as pc
        pc.tl.feature_simplex_regression(regression_adata, store_residuals=True, n_bootstrap=0)
        assert "peach_residuals" in regression_adata.obsm
        assert regression_adata.obsm["peach_residuals"].shape == (500, 50)

    def test_bootstrap_cis(self, regression_adata):
        """Bootstrap CIs are computed when n_bootstrap > 0."""
        import peach as pc
        result = pc.tl.feature_simplex_regression(regression_adata, n_bootstrap=50)
        assert result.get("vertex_ci_lower") is not None
        assert result.get("vertex_ci_upper") is not None
        assert np.asarray(result["vertex_ci_lower"]).shape == (50, 3)
        # CIs should bracket the point estimate
        assert np.all(np.asarray(result["vertex_ci_lower"]) <= np.asarray(result["vertex_coefficients"]) + 1e-6)
        assert np.all(np.asarray(result["vertex_ci_upper"]) >= np.asarray(result["vertex_coefficients"]) - 1e-6)

    def test_fdr_correction(self, regression_adata):
        """F-test p-values are BH-corrected: valid, monotone, and detects signal."""
        import peach as pc
        result = pc.tl.feature_simplex_regression(regression_adata, n_bootstrap=0)
        fdr = np.asarray(result["f_pvalue_fdr"])
        raw = np.asarray(result["f_pvalue"])
        assert "f_pvalue_fdr" in result
        assert len(fdr) == 50
        assert np.all(fdr >= 0) and np.all(fdr <= 1)
        # BH: FDR >= raw p-value (adjusted is never more optimistic than raw)
        assert np.all(fdr >= raw - 1e-10)
        # At least one feature is significant at FDR < 0.05 — fixture has strong signal
        assert np.any(fdr < 0.05), "No significant features found despite strong planted signal"
        # FDR is monotone non-decreasing when sorted by raw p-value
        sort_idx = np.argsort(raw)
        assert np.all(np.diff(fdr[sort_idx]) >= -1e-10)

    def test_vertex_contrasts_sum_to_zero(self, regression_adata):
        """vertex_contrasts must sum to zero across archetypes for every feature."""
        import peach as pc
        result = pc.tl.feature_simplex_regression(regression_adata, n_bootstrap=0)
        contrasts = np.asarray(result["vertex_contrasts"])
        assert contrasts.shape == (50, 3)
        np.testing.assert_allclose(contrasts.sum(axis=1), 0, atol=1e-10,
                                   err_msg="vertex_contrasts do not sum to zero per feature")

    def test_vertex_contrasts_equals_centered_betas(self, regression_adata):
        """vertex_contrasts == vertex_coefficients - row_mean(vertex_coefficients)."""
        import peach as pc
        result = pc.tl.feature_simplex_regression(regression_adata, n_bootstrap=0)
        betas = np.asarray(result["vertex_coefficients"])
        contrasts = np.asarray(result["vertex_contrasts"])
        np.testing.assert_allclose(contrasts, betas - betas.mean(axis=1, keepdims=True),
                                   atol=1e-10, err_msg="vertex_contrasts not equal to centered betas")

    def test_vertex_pvalues_test_contrast_null(self, regression_adata):
        """vertex_pvalues should be small for exclusive gene (large contrast) and
        near-uniform for flat gene (contrast ≈ 0)."""
        import peach as pc
        result = pc.tl.feature_simplex_regression(regression_adata, n_bootstrap=0)
        # Gene 0: exclusive to archetype 0 — its contrast for archetype 0 should be significant
        assert result["vertex_pvalues"][0, 0] < 0.05
        # Gene 1: flat betas — all contrasts ≈ 0, no archetype should be significant after FDR
        assert np.all(np.asarray(result["vertex_pvalues_fdr"])[1] > 0.05)

    def test_convenience_gene_wrapper(self, regression_adata):
        """pc.tl.gene_simplex_regression() is a convenience wrapper."""
        import peach as pc
        result = pc.tl.gene_simplex_regression(regression_adata, n_bootstrap=0)
        assert np.asarray(result["vertex_coefficients"]).shape[0] == 50

    def test_permutation_test_returns_pvalues(self, regression_adata):
        """permutation_test=True produces per-feature p-values."""
        import peach as pc
        result = pc.tl.feature_simplex_regression(
            regression_adata, permutation_test=True, n_permutations=50,
            n_bootstrap=0,
        )
        assert result.get("permutation_pvalue") is not None
        assert result.get("permutation_pvalue_fdr") is not None
        assert np.asarray(result["permutation_pvalue"]).shape == (50,)
        assert np.asarray(result["permutation_pvalue_fdr"]).shape == (50,)
        assert np.all(np.asarray(result["permutation_pvalue"]) >= 0)
        assert np.all(np.asarray(result["permutation_pvalue"]) <= 1)

    def test_permutation_test_detects_signal(self, regression_adata):
        """Genes with strong archetype signal get low permutation p-values."""
        import peach as pc
        result = pc.tl.feature_simplex_regression(
            regression_adata, permutation_test=True, n_permutations=99,
            n_bootstrap=0,
        )
        # Gene 0 (exclusive, strong signal) should be significant
        assert result["permutation_pvalue"][0] < 0.05
        # Gene 2 (gradient, strong signal) should be significant
        assert result["permutation_pvalue"][2] < 0.05

    def test_permutation_test_disabled_by_default(self, regression_adata):
        """permutation_test=False (default) leaves fields as None."""
        import peach as pc
        result = pc.tl.feature_simplex_regression(
            regression_adata, n_bootstrap=0,
        )
        assert result.get("permutation_pvalue") is None
        assert result.get("permutation_pvalue_fdr") is None


class TestPathwaySimplexRegression:
    def test_basic_run(self):
        """pathway_simplex_regression uses obsm['pathway_scores']."""
        import peach as pc

        rng = np.random.default_rng(42)
        K = 3
        n = 200
        n_pathways = 15

        weights = rng.dirichlet([1] * K, size=n)
        pathway_scores = rng.standard_normal((n, n_pathways))

        adata = AnnData(np.zeros((n, 10)))
        adata.obsm["cell_archetype_weights"] = weights
        adata.obsm["pathway_scores"] = pathway_scores

        result = pc.tl.pathway_simplex_regression(adata, n_bootstrap=0)
        assert np.asarray(result["vertex_coefficients"]).shape == (n_pathways, K)
        assert len(result["r_squared_degree1"]) == n_pathways


class TestRankInfo:
    def test_ols_fit_returns_rank_info(self):
        """ols_fit should report effective rank and expected rank."""
        from peach._core.utils.simplex_regression import ols_fit, scheffe_design_matrix
        rng = np.random.default_rng(42)
        # Normal simplex data: Scheffe (no intercept) has full rank K
        W = rng.dirichlet([1, 1, 1, 1], size=200)
        X, _ = scheffe_design_matrix(W, degree=1)
        Y = rng.standard_normal((200, 10))
        result = ols_fit(X, Y, robust_se=True)
        assert "effective_rank" in result
        assert result["effective_rank"] == 4  # full rank K
        assert "expected_rank" in result
        assert result["expected_rank"] == 4
        assert result["extra_rank_deficient"] is False

    def test_ols_fit_detects_extra_deficiency(self):
        """When columns are collinear beyond sum-to-1, flag extra deficiency."""
        from peach._core.utils.simplex_regression import ols_fit
        rng = np.random.default_rng(42)
        W = rng.dirichlet([1, 1, 1], size=200)
        # Add a column that's a linear combo of existing columns
        W_degen = np.column_stack([W, W[:, 0] + W[:, 1]])
        Y = rng.standard_normal((200, 10))
        result = ols_fit(W_degen, Y, robust_se=True)
        assert result["extra_rank_deficient"] is True
        assert result["effective_rank"] < W_degen.shape[1]


class TestCovarianceCache:
    def test_covariance_stored_by_default(self, regression_adata):
        """Regression result should include vertex_covariance for Wald reuse."""
        import peach as pc
        result = pc.tl.feature_simplex_regression(regression_adata, n_bootstrap=0)
        assert "vertex_covariance" in result
        assert result["vertex_covariance"] is not None
        cov_list = result["vertex_covariance"]
        assert len(cov_list) == 50  # n_features
        assert np.array(cov_list[0]).shape == (3, 3)  # K x K


class TestK2EdgeCase:
    def test_k2_regression(self):
        """Simplex regression works with K=2 archetypes."""
        import peach as pc

        rng = np.random.default_rng(42)
        K = 2
        n = 300
        n_genes = 20

        weights = rng.dirichlet([1] * K, size=n)
        true_beta = np.array([[5.0, 1.0]] * 10 + [[1.0, 5.0]] * 10)
        noise = rng.normal(0, 0.3, size=(n, n_genes))
        X = weights @ true_beta.T + noise

        adata = AnnData(X)
        adata.var_names = [f"gene_{i}" for i in range(n_genes)]
        adata.obsm["cell_archetype_weights"] = weights

        result = pc.tl.feature_simplex_regression(adata, n_bootstrap=0)
        assert np.asarray(result["vertex_coefficients"]).shape == (n_genes, K)
        # With K=2 the first gene should have high beta_0
        assert np.asarray(result["vertex_coefficients"])[0, 0] > 3.0

    def test_k2_degree2(self):
        """K=2 with degree=2 produces 1 interaction column."""
        import peach as pc

        rng = np.random.default_rng(42)
        K = 2
        n = 200
        n_genes = 10

        weights = rng.dirichlet([1] * K, size=n)
        X = rng.standard_normal((n, n_genes))

        adata = AnnData(X)
        adata.var_names = [f"gene_{i}" for i in range(n_genes)]
        adata.obsm["cell_archetype_weights"] = weights

        result = pc.tl.feature_simplex_regression(adata, max_degree=2, n_bootstrap=0)
        assert result.get("interaction_coefficients") is not None
        assert np.asarray(result["interaction_coefficients"]).shape == (n_genes, 1)  # K*(K-1)/2 = 1


@pytest.fixture
def driver_adata():
    """AnnData where two pathway scores predict archetype weights."""
    rng = np.random.default_rng(99)
    K = 3
    n = 400
    n_pathways = 10

    weights = rng.dirichlet([1] * K, size=n)
    pathway_scores = np.zeros((n, n_pathways))
    pathway_scores[:, 0] = weights[:, 0] + rng.normal(0, 0.1, n)  # predicts archetype 0
    pathway_scores[:, 1] = weights[:, 1] + rng.normal(0, 0.1, n)  # predicts archetype 1
    pathway_scores[:, 2:] = rng.standard_normal((n, n_pathways - 2))  # noise

    adata = AnnData(np.zeros((n, 5)))
    adata.var_names = [f"gene_{i}" for i in range(5)]
    adata.obsm["cell_archetype_weights"] = weights
    adata.obsm["pathway_scores"] = pathway_scores
    return adata


class TestArchetypeDriverRegression:

    def test_basic_run(self, driver_adata):
        """Runs without error, returns expected shapes."""
        import peach as pc
        result = pc.tl.archetype_driver_regression(
            driver_adata, feature_matrix="pathway_scores", n_bootstrap=0
        )
        assert "main_coefficients" in result
        assert np.array(result["main_coefficients"]).shape == (3, 10)  # [K, n_pathways]
        assert np.array(result["r_squared"]).shape == (2,)  # K-1 ILR components
        assert "peach_driver_regression" in driver_adata.uns

    def test_recovers_signal(self, driver_adata):
        """Pathways 0 and 1 (which predict weight) should have higher absolute
        back-transformed coefficients for their respective archetypes than noise pathways."""
        import peach as pc
        result = pc.tl.archetype_driver_regression(
            driver_adata, feature_matrix="pathway_scores", n_bootstrap=0
        )
        coefs = np.array(result["main_coefficients"])  # [K, n_pathways]
        # Pathway 0 should have a large effect on archetype 0 relative to noise pathways
        assert abs(coefs[0, 0]) > abs(coefs[0, 2:]).mean()
        # Pathway 1 should have a large effect on archetype 1 relative to noise pathways
        assert abs(coefs[1, 1]) > abs(coefs[1, 2:]).mean()

    def test_fdr_fields_present(self, driver_adata):
        """FDR-corrected p-values are present and valid."""
        import peach as pc
        result = pc.tl.archetype_driver_regression(
            driver_adata, feature_matrix="pathway_scores", n_bootstrap=0
        )
        fdr = np.array(result["main_pvalues_fdr"])
        assert fdr.shape == (2, 10)  # [K-1, n_pathways]
        assert np.all(fdr >= 0) and np.all(fdr <= 1)

    def test_near_singular_design_warns_not_crashes(self):
        """Near-singular design matrix triggers RuntimeWarning, not silent garbage."""
        import warnings as _warnings
        import peach as pc

        rng = np.random.default_rng(7)
        K = 3
        n = 200
        weights = rng.dirichlet([1] * K, size=n)

        # Perfectly collinear pathway scores: column 1 is a copy of column 0
        pathway_scores = rng.standard_normal((n, 5))
        pathway_scores[:, 1] = pathway_scores[:, 0]  # exact collinearity

        adata = AnnData(np.zeros((n, 3)))
        adata.var_names = ["g0", "g1", "g2"]
        adata.obsm["cell_archetype_weights"] = weights
        adata.obsm["pathway_scores"] = pathway_scores

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            result = pc.tl.archetype_driver_regression(
                adata, feature_matrix="pathway_scores", n_bootstrap=0
            )
        runtime_warns = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert len(runtime_warns) > 0, "Expected RuntimeWarning for near-singular design"
        assert result is not None

    def test_bootstrap_cis_shape(self, driver_adata):
        """Bootstrap CIs have correct shape [K, n_pathways]."""
        import peach as pc
        result = pc.tl.archetype_driver_regression(
            driver_adata, feature_matrix="pathway_scores", n_bootstrap=50
        )
        assert result.get("main_ci_lower") is not None
        assert np.array(result["main_ci_lower"]).shape == (3, 10)
        assert np.array(result["main_ci_upper"]).shape == (3, 10)


class TestComprehensiveDegree:

    def test_basic_run(self):
        """comprehensive_degree=True returns degree_comparison dict."""
        import peach as pc

        rng = np.random.default_rng(42)
        K = 4
        n = 300
        n_genes = 20

        weights = rng.dirichlet([1] * K, size=n)
        true_beta = rng.standard_normal((n_genes, K)) * 3
        X = weights @ true_beta.T + rng.normal(0, 0.5, (n, n_genes))

        adata = AnnData(X)
        adata.var_names = [f"gene_{i}" for i in range(n_genes)]
        adata.obsm["cell_archetype_weights"] = weights

        result = pc.tl.feature_simplex_regression(
            adata, max_degree=1, comprehensive_degree=True, n_bootstrap=0
        )
        assert "degree_comparison" in result
        dc = result["degree_comparison"]
        assert "degree_2" in dc

    def test_incremental_f_nonnegative(self):
        """Incremental F-statistics are non-negative (clamped from numerical noise)."""
        import peach as pc

        rng = np.random.default_rng(42)
        K = 4
        n = 300
        n_genes = 30

        weights = rng.dirichlet([1] * K, size=n)
        X = rng.standard_normal((n, n_genes))

        adata = AnnData(X)
        adata.var_names = [f"gene_{i}" for i in range(n_genes)]
        adata.obsm["cell_archetype_weights"] = weights

        result = pc.tl.feature_simplex_regression(
            adata, max_degree=1, comprehensive_degree=True, n_bootstrap=0
        )
        dc = result["degree_comparison"]
        for deg_key, deg_info in dc.items():
            f_stats = np.asarray(deg_info["incremental_f"])
            assert np.all(f_stats >= 0), f"{deg_key}: negative incremental F-statistics found"

    def test_delta_r2_finite(self):
        """delta_r2 entries are finite (not NaN/inf from degenerate SS computation)."""
        import peach as pc

        rng = np.random.default_rng(42)
        K = 3
        n = 200
        n_genes = 15

        weights = rng.dirichlet([1] * K, size=n)
        X = weights @ rng.standard_normal((K, n_genes)) + rng.normal(0, 0.3, (n, n_genes))

        adata = AnnData(X)
        adata.var_names = [f"gene_{i}" for i in range(n_genes)]
        adata.obsm["cell_archetype_weights"] = weights

        result = pc.tl.feature_simplex_regression(
            adata, max_degree=1, comprehensive_degree=True, n_bootstrap=0
        )
        for deg_key, deg_info in result["degree_comparison"].items():
            assert np.all(np.isfinite(deg_info["delta_r2"])), \
                f"{deg_key}: non-finite delta_r2 values"

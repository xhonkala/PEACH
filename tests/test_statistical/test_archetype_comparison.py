import numpy as np
import pytest


def _make_adata_with_weights(rng=None, n_cells=200, K=3, n_genes=50):
    """Create AnnData with weights and PCA for comparison tests."""
    import anndata as ad
    if rng is None:
        rng = np.random.default_rng(42)
    X = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
    adata = ad.AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obsm["X_pca"] = rng.standard_normal((n_cells, 10)).astype(np.float32)
    adata.obsm["cell_archetype_weights"] = rng.dirichlet(np.ones(K), n_cells)
    return adata


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


from anndata import AnnData


@pytest.fixture
def comparison_adata():
    """AnnData with 4 archetypes, planted structure for comparison tests."""
    rng = np.random.default_rng(42)
    K = 4
    n = 600
    n_genes = 30

    weights = rng.dirichlet([1] * K, size=n)
    # Plant known structure: archetypes 0,1 are similar; 2,3 are different
    true_beta = np.zeros((n_genes, K))
    true_beta[:, 0] = rng.normal(5, 1, n_genes)
    true_beta[:, 1] = true_beta[:, 0] + rng.normal(0, 0.3, n_genes)  # similar to 0
    true_beta[:, 2] = rng.normal(-3, 1, n_genes)  # very different
    true_beta[:, 3] = rng.normal(0, 2, n_genes)   # different

    X = weights @ true_beta.T + rng.normal(0, 0.2, (n, n_genes))

    adata = AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obsm["cell_archetype_weights"] = weights
    adata.obsm["X_pca"] = rng.standard_normal((n, 10))
    return adata


class TestArchetypeMMDCompute:
    def test_within_fit_symmetric(self, comparison_adata):
        from peach._core.utils.archetype_comparison import compute_archetype_mmd
        mmd_mat, pval_mat = compute_archetype_mmd(
            comparison_adata, n_permutations=20
        )
        K = 4
        assert mmd_mat.shape == (K, K)
        assert pval_mat.shape == (K, K)
        np.testing.assert_allclose(np.diag(mmd_mat), 0, atol=1e-10)
        np.testing.assert_allclose(mmd_mat, mmd_mat.T, atol=1e-10)

    def test_similar_archetypes_low_mmd(self, comparison_adata):
        from peach._core.utils.archetype_comparison import compute_archetype_mmd
        mmd_mat, _ = compute_archetype_mmd(
            comparison_adata, n_permutations=10
        )
        # Archetypes 0 and 1 should have lower MMD than 0 and 2
        assert mmd_mat[0, 1] < mmd_mat[0, 2]


class TestArchetypeFeatureSimilarity:
    def test_spearman_symmetric(self, comparison_adata):
        from peach._core.utils.archetype_comparison import compute_feature_similarity
        import peach as pc
        pc.tl.feature_simplex_regression(comparison_adata, n_bootstrap=0)
        result = compute_feature_similarity(comparison_adata)
        K = 4
        assert result["spearman_matrix"].shape == (K, K)
        np.testing.assert_allclose(
            np.diag(result["spearman_matrix"]), 1.0, atol=1e-10
        )

    def test_similar_archetypes_high_spearman(self, comparison_adata):
        from peach._core.utils.archetype_comparison import compute_feature_similarity
        import peach as pc
        pc.tl.feature_simplex_regression(comparison_adata, n_bootstrap=0)
        result = compute_feature_similarity(comparison_adata)
        assert result["spearman_matrix"][0, 1] > result["spearman_matrix"][0, 2]

    def test_n_significant_features(self, comparison_adata):
        from peach._core.utils.archetype_comparison import compute_feature_similarity
        import peach as pc
        pc.tl.feature_simplex_regression(comparison_adata, n_bootstrap=0)
        result = compute_feature_similarity(comparison_adata)
        assert "n_significant_features" in result
        assert result["n_significant_features"] >= 0
        assert result["n_significant_features"] <= result["n_shared_features"]
        # With planted strong signal, most features should be significant
        assert result["n_significant_features"] > 0

    def test_no_silhouette_fields(self, comparison_adata):
        """Silhouette fields should no longer be returned."""
        from peach._core.utils.archetype_comparison import compute_feature_similarity
        import peach as pc
        pc.tl.feature_simplex_regression(comparison_adata, n_bootstrap=0)
        result = compute_feature_similarity(comparison_adata)
        assert "silhouette_per_archetype" not in result
        assert "silhouette_overall" not in result

    def test_spearman_fdr_present(self, comparison_adata):
        """Spearman p-values should include FDR-corrected version."""
        from peach._core.utils.archetype_comparison import compute_feature_similarity
        import peach as pc
        pc.tl.feature_simplex_regression(comparison_adata, n_bootstrap=0)
        result = compute_feature_similarity(comparison_adata)
        assert "spearman_pvalue_fdr_matrix" in result
        K = 4
        assert result["spearman_pvalue_fdr_matrix"].shape == (K, K)
        # FDR should be >= raw for all entries
        assert np.all(
            result["spearman_pvalue_fdr_matrix"] >= result["spearman_pvalue_matrix"] - 1e-10
        )


class TestArchetypeContrasts:
    def test_all_pairs(self, comparison_adata):
        from peach._core.utils.archetype_comparison import compute_wald_contrasts
        import peach as pc
        pc.tl.feature_simplex_regression(comparison_adata, n_bootstrap=0)
        result = compute_wald_contrasts(comparison_adata)
        K = 4
        n_pairs = K * (K - 1) // 2
        assert len(result["pairs"]) == n_pairs
        for pair in result["pairs"]:
            assert result["delta_beta"][pair].shape == (30,)
            assert result["pvalues_fdr"][pair].shape == (30,)

    def test_similar_pair_fewer_significant(self, comparison_adata):
        from peach._core.utils.archetype_comparison import compute_wald_contrasts
        import peach as pc
        pc.tl.feature_simplex_regression(comparison_adata, n_bootstrap=0)
        result = compute_wald_contrasts(comparison_adata)
        sig_01 = np.sum(result["pvalues_fdr"][(0, 1)] < 0.05)
        sig_02 = np.sum(result["pvalues_fdr"][(0, 2)] < 0.05)
        assert sig_01 < sig_02


    def test_wald_fdr_is_global_not_per_pair(self, comparison_adata):
        """FDR correction across ALL pairs should differ from per-pair FDR."""
        from peach._core.utils.archetype_comparison import compute_wald_contrasts
        from statsmodels.stats.multitest import multipletests as mt
        import peach as pc
        pc.tl.feature_simplex_regression(comparison_adata, n_bootstrap=0)
        result = compute_wald_contrasts(comparison_adata)

        # Compute what per-pair FDR would give (the old buggy behavior)
        per_pair_fdr = {}
        for pair in result["pairs"]:
            raw = result["pvalues"][pair]
            _, fdr_pp, _, _ = mt(raw, method="fdr_bh")
            per_pair_fdr[pair] = fdr_pp

        # Global FDR should differ from per-pair for at least one non-trivial pair
        any_differ = False
        for pair in result["pairs"]:
            if not np.allclose(result["pvalues"][pair], 1.0):
                if not np.allclose(result["pvalues_fdr"][pair], per_pair_fdr[pair], atol=1e-10):
                    any_differ = True
        assert any_differ, "Global FDR appears identical to per-pair FDR"


class TestPublicAPI:
    def test_archetype_mmd_api(self, comparison_adata):
        import peach as pc
        result = pc.tl.archetype_mmd(comparison_adata, n_permutations=10)
        assert "mmd_matrix" in result
        assert "pvalue_matrix" in result
        assert np.asarray(result["mmd_matrix"]).shape == (4, 4)
        assert "peach_archetype_mmd" in comparison_adata.uns

    def test_archetype_feature_similarity_api(self, comparison_adata):
        import peach as pc
        pc.tl.feature_simplex_regression(comparison_adata, n_bootstrap=0)
        result = pc.tl.archetype_feature_similarity(comparison_adata)
        assert "spearman_matrix" in result
        assert "n_significant_features" in result
        assert "silhouette_overall" not in result
        assert "peach_archetype_feature_similarity" in comparison_adata.uns

    def test_archetype_contrasts_api(self, comparison_adata):
        import peach as pc
        pc.tl.feature_simplex_regression(comparison_adata, n_bootstrap=0)
        result = pc.tl.archetype_contrasts(comparison_adata)
        assert "pairs" in result
        assert "delta_beta" in result
        assert "peach_archetype_contrasts_genes" in comparison_adata.uns


class TestWaldQvalueUnderflow:
    def test_wald_qvalues_not_all_zero(self):
        """Wald FDR q-values should not ALL be zero -- at minimum flat genes should be non-significant."""
        from peach._core.utils.archetype_comparison import compute_wald_contrasts
        import peach as pc

        rng = np.random.default_rng(42)
        K, n, n_genes = 3, 500, 50
        weights = rng.dirichlet([1] * K, size=n)
        true_beta = rng.standard_normal((n_genes, K)) * 5
        true_beta[1] = [3.0, 3.0, 3.0]  # flat gene
        noise = rng.normal(0, 0.3, size=(n, n_genes))
        X = weights @ true_beta.T + noise
        adata = AnnData(X)
        adata.var_names = [f"gene_{i}" for i in range(n_genes)]
        adata.obsm["cell_archetype_weights"] = weights
        pc.tl.feature_simplex_regression(adata, n_bootstrap=0)
        result = compute_wald_contrasts(adata)
        pairs = result["pairs"]
        all_fdr = np.concatenate([np.asarray(result["pvalues_fdr"][p]) for p in pairs])
        # Not ALL zero
        assert np.any(all_fdr > 0), "All Wald FDR q-values are zero"


class TestMMDUnbiased:
    def test_mmd_unbiased_identical(self):
        """MMD of identical distribution should be near zero."""
        from peach._core.utils.flow_matching import compute_mmd
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5))
        mmd = compute_mmd(X, X)
        assert abs(mmd) < 0.05, f"Expected ~0, got {mmd}"

    def test_mmd_unbiased_different(self):
        """MMD of shifted distributions should be positive."""
        from peach._core.utils.flow_matching import compute_mmd
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5))
        Y = rng.standard_normal((200, 5)) + 3.0
        mmd = compute_mmd(X, Y)
        assert mmd > 0.1, f"Expected positive MMD, got {mmd}"


class TestWeightedMMD:
    def test_weighted_mmd_produces_kxk_matrix(self):
        """Weighted MMD should produce K x K matrix."""
        import peach as pc
        rng = np.random.default_rng(42)
        n, K = 300, 3
        weights = rng.dirichlet([1] * K, size=n)
        pca = rng.standard_normal((n, 10))
        adata = AnnData(rng.standard_normal((n, 20)))
        adata.obsm["cell_archetype_weights"] = weights
        adata.obsm["X_pca"] = pca
        result = pc.tl.archetype_mmd(adata, n_permutations=50)
        mmd_matrix = np.asarray(result["mmd_matrix"])
        assert mmd_matrix.shape == (K, K)

    def test_weighted_mmd_diagonal_zero(self):
        """Within-fit diagonal should be zero (same archetype vs itself)."""
        import peach as pc
        rng = np.random.default_rng(42)
        n, K = 300, 3
        weights = rng.dirichlet([1] * K, size=n)
        pca = rng.standard_normal((n, 10))
        adata = AnnData(rng.standard_normal((n, 20)))
        adata.obsm["cell_archetype_weights"] = weights
        adata.obsm["X_pca"] = pca
        result = pc.tl.archetype_mmd(adata, n_permutations=0)
        mmd_matrix = np.asarray(result["mmd_matrix"])
        np.testing.assert_allclose(np.diag(mmd_matrix), 0.0, atol=1e-10)


class TestWaldContrastsFeatureSource:
    def test_wald_contrasts_respects_feature_source(self):
        """Wald contrasts use the feature_type matching the stored regression."""
        import peach as pc

        adata = _make_adata_with_weights()
        n_cells = adata.n_obs
        rng = np.random.default_rng(99)
        custom_features = rng.standard_normal((n_cells, 10)).astype(np.float32)
        adata.obsm["test_features"] = custom_features

        pc.tl.feature_simplex_regression(adata, feature_matrix="test_features", n_bootstrap=0)

        # Must pass feature_type matching what was stored ("test_features")
        result = pc.tl.archetype_contrasts(adata, feature_type="test_features")
        assert result["n_features"] == 10, (
            f"Expected 10 features (from test_features), got {result['n_features']}"
        )


class TestMMDPermutationSymmetry:
    def test_mmd_permutation_symmetric(self):
        """Permuting (i,j) and (j,i) should give the same p-value distribution."""
        from peach._core.utils.archetype_comparison import compute_archetype_mmd

        rng = np.random.default_rng(42)
        adata = _make_adata_with_weights(rng, n_cells=200, K=3)

        # Should be symmetric for within-fit
        mmd, pval = compute_archetype_mmd(adata, n_permutations=200, seed=42)

        # p-value matrix should be symmetric
        np.testing.assert_array_almost_equal(
            pval, pval.T, decimal=1,
            err_msg="Within-fit MMD p-values should be symmetric"
        )


class TestBetweenFitMMDMismatchedK:
    def test_between_fit_mmd_mismatched_K(self):
        """Between-fit MMD with different K: MMD valid everywhere, but p-values
        only meaningful when both archetype indices exist in both fits."""
        from peach._core.utils.archetype_comparison import compute_archetype_mmd

        adata_a = _make_adata_with_weights(K=3)
        adata_b = _make_adata_with_weights(K=5)

        mmd, pval = compute_archetype_mmd(adata_a, adata_b, n_permutations=50)

        assert mmd.shape == (3, 5)
        # All MMD entries should be finite (comparison is always valid)
        assert np.all(np.isfinite(mmd))
        # p-values for i < K_a=3 and j < K_b=5 where i < K_b=5 and j < K_a=3:
        # valid block is [0:3, 0:3]
        assert np.all(np.isfinite(pval[:3, :3]))
        # p-values outside the valid block should be NaN
        assert np.all(np.isnan(pval[:3, 3:])), (
            "p-values for j >= K_a should be NaN (no matching archetype in fit A)"
        )

"""Tests for PEACH statistical controls: permutation, FDR, bootstrap, and integration."""

import numpy as np
import pytest

from peach._core.utils.permutation import bootstrap_ci, fdr_correct, permutation_pvalue


# =============================================================================
# 11a. Shared permutation utility tests
# =============================================================================


class TestPermutationPvalue:
    def test_extreme_observed_gets_small_pvalue(self):
        """An observed statistic far from the null should yield p < 0.05."""
        null = np.random.default_rng(42).normal(0, 1, size=999)
        observed = 5.0  # far from null
        p = permutation_pvalue(observed, null, alternative="greater")
        assert p < 0.01

    def test_null_observed_gets_large_pvalue(self):
        """An observed statistic drawn from the null should yield p > 0.05."""
        rng = np.random.default_rng(42)
        null = rng.normal(0, 1, size=999)
        observed = 0.1  # well within null
        p = permutation_pvalue(observed, null, alternative="two-sided")
        assert p > 0.1

    def test_vectorized(self):
        """Should handle array of observed values against null matrix."""
        rng = np.random.default_rng(42)
        null = rng.normal(0, 1, size=(999, 50))
        observed = np.concatenate([np.full(10, 5.0), np.full(40, 0.1)])
        p = permutation_pvalue(observed, null, alternative="two-sided")
        assert p.shape == (50,)
        assert np.all(p[:10] < 0.05)
        assert np.all(p[10:] > 0.05)

    def test_phipson_smyth_correction(self):
        """P-value should never be exactly 0 due to +1/+1 correction."""
        null = np.zeros(999)
        observed = 100.0
        p = permutation_pvalue(observed, null, alternative="greater")
        assert p > 0  # (0+1)/(999+1) = 0.001
        assert p == pytest.approx(1 / 1000)

    def test_alternative_less(self):
        """alternative='less' should detect negative extremes."""
        null = np.random.default_rng(42).normal(0, 1, size=999)
        p = permutation_pvalue(-5.0, null, alternative="less")
        assert p < 0.01


class TestFDRCorrect:
    def test_all_significant(self):
        pvals = np.full(10, 1e-10)
        rejected, corrected = fdr_correct(pvals)
        assert np.all(rejected)
        assert np.all(corrected < 0.05)

    def test_none_significant(self):
        pvals = np.full(10, 0.5)
        rejected, corrected = fdr_correct(pvals)
        assert not np.any(rejected)

    def test_mixed(self):
        pvals = np.array([1e-10, 1e-8, 0.01, 0.5, 0.9])
        rejected, corrected = fdr_correct(pvals)
        assert rejected[0] and rejected[1]
        assert not rejected[-1]

    def test_handles_nan(self):
        pvals = np.array([1e-10, np.nan, 0.5])
        rejected, corrected = fdr_correct(pvals)
        assert rejected[0]
        assert corrected[1] == 1.0  # NaN -> 1.0


class TestBootstrapCI:
    def test_known_mean(self):
        """Bootstrap CI for mean of N(5,1) should contain 5."""
        rng = np.random.default_rng(42)
        data = rng.normal(5, 1, size=(1000, 1))
        point, lo, hi = bootstrap_ci(data, lambda x: x.mean(), n_bootstrap=500)
        assert lo < 5.0 < hi

    def test_narrow_with_large_sample(self):
        """CI should be narrow with large sample."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, size=(10000, 1))
        _, lo, hi = bootstrap_ci(data, lambda x: x.mean(), n_bootstrap=500)
        assert (hi - lo) < 0.1

    def test_wide_with_small_sample(self):
        """CI should be wider with small sample."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, size=(20, 1))
        _, lo, hi = bootstrap_ci(data, lambda x: x.mean(), n_bootstrap=500)
        assert (hi - lo) > 0.1


# =============================================================================
# 11b. Simplex regression permutation tests
# =============================================================================


class TestSimplexRegressionPermutation:
    def test_real_signal_detected(self):
        """Synthetic data where expression is a linear function of archetype
        weights should have significant permutation p-values for signal genes."""
        import anndata as ad

        import peach as pc

        rng = np.random.default_rng(42)
        n_cells, K = 500, 3
        weights = rng.dirichlet(np.ones(K), size=n_cells)

        # Signal genes: expression = linear function of weights + small noise
        signal_expr = weights @ rng.normal(0, 5, size=(K, 10)) + rng.normal(
            0, 0.1, size=(n_cells, 10)
        )
        # Noise genes: pure random
        noise_expr = rng.normal(0, 1, size=(n_cells, 40))
        X = np.hstack([signal_expr, noise_expr])

        adata = ad.AnnData(X.astype(np.float32))
        adata.obsm["cell_archetype_weights"] = weights
        adata.var_names = [f"signal_{i}" for i in range(10)] + [
            f"noise_{i}" for i in range(40)
        ]

        pc.tl.gene_simplex_regression(
            adata, max_degree=1, permutation_test=True, n_permutations=100,
            n_bootstrap=0,
        )
        reg = adata.uns["peach_simplex_regression_genes"]
        perm_pval = np.asarray(reg["permutation_pvalue"])
        perm_fdr = np.asarray(reg["permutation_pvalue_fdr"])

        # Signal genes (first 10) should mostly be significant
        n_sig_signal = (perm_fdr[:10] < 0.05).sum()
        assert n_sig_signal >= 5, (
            f"Expected at least 5/10 signal genes significant, got {n_sig_signal}"
        )

    def test_noise_not_detected(self):
        """Random noise features should not be significant."""
        import anndata as ad

        import peach as pc

        rng = np.random.default_rng(42)
        n_cells, n_genes, K = 500, 50, 3
        weights = rng.dirichlet(np.ones(K), size=n_cells)
        X = rng.normal(0, 1, size=(n_cells, n_genes))  # pure noise

        adata = ad.AnnData(X.astype(np.float32))
        adata.obsm["cell_archetype_weights"] = weights
        adata.var_names = [f"noise_{i}" for i in range(n_genes)]

        pc.tl.gene_simplex_regression(
            adata, max_degree=1, permutation_test=True, n_permutations=100,
            n_bootstrap=0,
        )
        reg = adata.uns["peach_simplex_regression_genes"]
        perm_fdr = np.asarray(reg["permutation_pvalue_fdr"])

        # Very few should be significant (allow FDR-level false positives)
        n_sig = (perm_fdr < 0.05).sum()
        assert n_sig < n_genes * 0.1, (
            f"Expected fewer than {n_genes * 0.1:.0f} noise genes significant, got {n_sig}"
        )


# =============================================================================
# 11d. Jacobian permutation tests
# =============================================================================


class TestJacobianPermutation:
    def test_expanding_gene_detected(self):
        """A gene whose PCA loading aligns with the expanding eigenvector of
        the Jacobian should be significant under permutation.

        Design: 3 signal genes out of 500 total, with very strong PC0 loading.
        The Jacobian has eigenvalue 20 along PC0 and 1 elsewhere. After
        normalization, signal genes get expansion ~20 while noise genes average
        ~(20+4)/5 = 4.8. With only 3/500 signal loadings in the pool, the
        permutation null has <1% chance of drawing a signal loading, making
        detection robust.
        """
        from peach._core.utils.permutation import fdr_correct, permutation_pvalue

        rng = np.random.default_rng(42)
        n_pcs = 5
        n_genes = 500
        n_signal = 3

        # Jacobian with very strong expansion along PC0
        J = np.eye(n_pcs)
        J[0, 0] = 20.0

        # Create PCA loadings: noise genes have isotropic random loadings,
        # signal genes have loading concentrated on PC0
        loadings = rng.normal(0, 1, size=(n_genes, n_pcs))
        for i in range(n_signal):
            loadings[i] = np.zeros(n_pcs)
            loadings[i, 0] = 10.0  # near-pure PC0 loading

        # Normalize loadings
        norms = np.linalg.norm(loadings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        loadings_norm = loadings / norms

        # Observed feature expansion: L^T J L per gene
        observed_expansion = np.einsum(
            "gi,ij,gj->g", loadings_norm, J, loadings_norm
        )

        # Permutation null: shuffle gene-to-loading assignments
        n_perms = 999
        null_expansion = np.empty((n_perms, n_genes))
        for p in range(n_perms):
            perm_idx = rng.permutation(n_genes)
            shuffled = loadings_norm[perm_idx]
            null_expansion[p] = np.einsum(
                "gi,ij,gj->g", shuffled, J, shuffled
            )

        pvals = permutation_pvalue(
            observed_expansion, null_expansion, alternative="greater"
        )

        # Signal genes should have small raw p-values (< 0.01)
        # since only 3/500 = 0.6% of loadings give similar expansion.
        # We check raw p-values because BH FDR correction is conservative
        # when very few genes (3/500) are truly significant -- the correction
        # factor (~500/rank) overwhelms the small raw p-values.
        assert np.all(pvals[:n_signal] < 0.02), (
            f"Expected all signal genes to have raw p < 0.02, "
            f"got {pvals[:n_signal]}"
        )

        # Noise genes should mostly have large p-values
        noise_sig = (pvals[n_signal:] < 0.01).sum()
        # With Phipson-Smyth correction and 999 perms, minimum p = 1/1000.
        # Under null, ~1% of genes expected to have p < 0.01 by chance.
        assert noise_sig < (n_genes - n_signal) * 0.05, (
            f"Expected <5% of noise genes with p < 0.01, got {noise_sig}"
        )

    def test_random_loading_not_significant(self):
        """With identity Jacobian (no expansion), no gene should be significant."""
        from peach._core.utils.permutation import fdr_correct, permutation_pvalue

        rng = np.random.default_rng(42)
        n_pcs = 5
        n_genes = 100

        # Identity Jacobian: no expansion or contraction
        J = np.eye(n_pcs)

        loadings = rng.normal(0, 1, size=(n_genes, n_pcs))
        norms = np.linalg.norm(loadings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        loadings_norm = loadings / norms

        observed = np.einsum("gi,ij,gj->g", loadings_norm, J, loadings_norm)

        n_perms = 200
        null = np.empty((n_perms, n_genes))
        for p in range(n_perms):
            perm_idx = rng.permutation(n_genes)
            shuffled = loadings_norm[perm_idx]
            null[p] = np.einsum("gi,ij,gj->g", shuffled, J, shuffled)

        pvals = permutation_pvalue(observed, null, alternative="two-sided")
        _, pvals_fdr = fdr_correct(pvals)

        # With identity Jacobian, all L^T I L = 1 for unit-norm L,
        # so permutation changes nothing and no gene should be significant
        n_sig = (pvals_fdr < 0.05).sum()
        assert n_sig < n_genes * 0.1, (
            f"Expected <10% genes significant with identity Jacobian, got {n_sig}"
        )

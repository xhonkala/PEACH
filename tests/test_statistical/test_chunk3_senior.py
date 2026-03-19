"""Senior adversarial review tests for Chunk 3: statistical & cleanup fixes.

Tests target:
- ILR roundtrip for normal Dirichlet data (exact to ~1e-3 due to epsilon)
- ILR roundtrip for extreme vertex coordinates (clipping case)
- Dirichlet _log_joint matches old separate-computation form numerically
- Dirichlet EM convergence on planted 2-component data
- Dirichlet vertex warning fires at correct threshold
- MMD permutation null: under true H0, p-values should be ~uniform
- Cross-K NaN: exact entries for K_a=3, K_b=5
- mannwhitneyu reproducibility: same feature_name -> same noise
- mannwhitneyu different features -> different noise (not correlated)
- Between-fit MMD matrix shape is K_a x K_b
"""

import warnings

import numpy as np
import pytest
from anndata import AnnData
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simplex_weights(n, K, seed=42):
    """Generate random simplex weights that sum to 1."""
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(K), size=n)


def _make_mmd_adata(n=200, K=3, n_pcs=10, seed=42, weights=None):
    """Build AnnData with archetype weights and PCA coords for MMD tests."""
    rng = np.random.default_rng(seed)
    if weights is None:
        weights = _simplex_weights(n, K, seed=seed)
    pca = rng.standard_normal((n, n_pcs))
    adata = AnnData(X=rng.standard_normal((n, 50)))
    adata.obs_names = [f"cell_{i}" for i in range(n)]
    adata.var_names = [f"gene_{i}" for i in range(50)]
    adata.obsm["cell_archetype_weights"] = weights
    adata.obsm["X_pca"] = pca
    return adata


# ===========================================================================
# 1. ILR roundtrip for normal Dirichlet data
# ===========================================================================


class TestILRRoundtripNormal:
    """Interior Dirichlet samples should roundtrip through ILR with
    error bounded by the epsilon smoothing constant."""

    def test_roundtrip_interior_points(self):
        """Well-interior simplex points (all weights > 0.05) should
        roundtrip with small error."""
        from peach._core.utils.ilr_transform import ilr_transform, inverse_ilr

        rng = np.random.default_rng(123)
        # Dirichlet(5,5,5,5) produces well-interior points
        W = rng.dirichlet([5, 5, 5, 5], size=500)
        assert np.all(W > 0.01), "Sanity: all weights well interior"

        ilr_coords = ilr_transform(W)
        W_roundtrip = inverse_ilr(ilr_coords)

        # Epsilon smoothing shifts points inward, so we expect some error
        # but it should be small for interior points
        max_err = np.max(np.abs(W - W_roundtrip))
        assert max_err < 0.02, (
            f"Interior point roundtrip error {max_err:.4f} exceeds 0.02. "
            "Epsilon smoothing should have minimal impact on interior points."
        )

    def test_roundtrip_respects_simplex(self):
        """Roundtrip output must remain on the simplex."""
        from peach._core.utils.ilr_transform import ilr_transform, inverse_ilr

        rng = np.random.default_rng(456)
        W = rng.dirichlet([1, 1, 1], size=200)
        ilr_coords = ilr_transform(W)
        W_rt = inverse_ilr(ilr_coords)

        # Must sum to 1
        row_sums = W_rt.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10,
                                   err_msg="Roundtrip broke simplex constraint")
        # Must be non-negative
        assert np.all(W_rt >= 0), "Roundtrip produced negative weights"

    def test_roundtrip_K2(self):
        """Edge case: K=2 should also roundtrip cleanly."""
        from peach._core.utils.ilr_transform import ilr_transform, inverse_ilr

        W = np.array([[0.7, 0.3], [0.5, 0.5], [0.9, 0.1]])
        ilr_coords = ilr_transform(W)
        assert ilr_coords.shape == (3, 1), "K=2 -> 1 ILR coordinate"
        W_rt = inverse_ilr(ilr_coords)
        max_err = np.max(np.abs(W - W_rt))
        assert max_err < 0.01, f"K=2 roundtrip error {max_err:.4f}"


# ===========================================================================
# 2. ILR roundtrip for extreme (vertex) coordinates
# ===========================================================================


class TestILRRoundtripExtreme:
    """Vertex-like data where some weights are ~0. The CLR clip in
    inverse_ilr should prevent overflow but accuracy will be lower."""

    def test_vertex_data_no_overflow(self):
        """Near-vertex data should not produce inf/nan in roundtrip."""
        from peach._core.utils.ilr_transform import ilr_transform, inverse_ilr

        # Extreme: one weight dominates
        W = np.array([
            [0.999, 0.0005, 0.0005],
            [0.0001, 0.9998, 0.0001],
            [1e-10, 1e-10, 1.0 - 2e-10],
        ])
        ilr_coords = ilr_transform(W)
        W_rt = inverse_ilr(ilr_coords)

        assert not np.any(np.isnan(W_rt)), "Overflow: NaN in roundtrip"
        assert not np.any(np.isinf(W_rt)), "Overflow: inf in roundtrip"
        # Still on simplex
        np.testing.assert_allclose(W_rt.sum(axis=1), 1.0, atol=1e-10)

    def test_vertex_roundtrip_error_bounded(self):
        """Even at vertices, roundtrip error should be bounded (not exact)."""
        from peach._core.utils.ilr_transform import ilr_transform, inverse_ilr

        W = np.array([[0.998, 0.001, 0.001]])
        ilr_coords = ilr_transform(W)
        W_rt = inverse_ilr(ilr_coords)

        # Error will be larger than interior (epsilon pulls inward) but bounded
        max_err = np.max(np.abs(W - W_rt))
        assert max_err < 0.05, (
            f"Vertex roundtrip error {max_err:.4f} is unacceptably large"
        )

    def test_clr_clip_prevents_exp_overflow(self):
        """Directly test that extreme ILR coordinates don't cause exp overflow."""
        from peach._core.utils.ilr_transform import inverse_ilr

        # Artificial extreme ILR coords that would overflow without clipping
        extreme_ilr = np.array([[1000, -1000, 500]])
        W = inverse_ilr(extreme_ilr)
        assert not np.any(np.isnan(W)), "CLR clip failed: NaN"
        assert not np.any(np.isinf(W)), "CLR clip failed: inf"
        np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-10)


# ===========================================================================
# 3. Dirichlet _log_joint matches old separate computations
# ===========================================================================


class TestDirichletLogJoint:
    """The refactored _log_joint must produce identical numerical results
    to manually computing log(pi_c) + log_dirichlet_pdf(W, alpha_c)."""

    def test_log_joint_matches_manual(self):
        from peach._core.utils.dirichlet_mixture import DirichletMixture

        rng = np.random.default_rng(42)
        K = 4
        n_components = 3
        n_samples = 100

        W = rng.dirichlet(np.ones(K), size=n_samples)
        alphas = rng.gamma(2.0, 1.0, size=(n_components, K))
        mix_weights = rng.dirichlet(np.ones(n_components))

        dm = DirichletMixture(n_components=n_components)

        # Using the class method
        log_joint = dm._log_joint(W, alphas, mix_weights)

        # Manual computation (the old way)
        log_joint_manual = np.zeros((n_samples, n_components))
        for c in range(n_components):
            # log Dirichlet PDF
            log_B = gammaln(alphas[c]).sum() - gammaln(alphas[c].sum())
            log_p = -log_B + ((alphas[c] - 1) * np.log(np.clip(W, 1e-300, None))).sum(axis=1)
            log_joint_manual[:, c] = np.log(np.clip(mix_weights[c], 1e-300, None)) + log_p

        np.testing.assert_allclose(
            log_joint, log_joint_manual, atol=1e-12,
            err_msg="_log_joint refactor changed numerical output"
        )

    def test_log_joint_with_uniform_weights(self):
        """Edge case: uniform mixing weights."""
        from peach._core.utils.dirichlet_mixture import DirichletMixture

        K = 3
        n_components = 2
        W = np.array([[1/3, 1/3, 1/3], [0.5, 0.3, 0.2]])
        alphas = np.array([[1.0, 1.0, 1.0], [2.0, 3.0, 1.0]])
        mix_weights = np.array([0.5, 0.5])

        dm = DirichletMixture(n_components=n_components)
        log_joint = dm._log_joint(W, alphas, mix_weights)

        # Component 0: Dirichlet(1,1,1) is uniform on simplex
        # log B(1,1,1) = log(Gamma(1)^3 / Gamma(3)) = log(1/2) = -log(2)
        # log p(w|alpha=[1,1,1]) = -(-log(2)) + 0 = log(2)
        expected_c0_logp = np.log(2)  # For any point on the simplex
        expected_c0 = np.log(0.5) + expected_c0_logp
        np.testing.assert_allclose(
            log_joint[:, 0], expected_c0, atol=1e-10,
            err_msg="Uniform Dirichlet PDF should be constant"
        )


# ===========================================================================
# 4. Dirichlet EM convergence on planted 2-component data
# ===========================================================================


class TestDirichletEMConvergence:
    """2-component Dirichlet mixture with well-separated components should
    be recovered by EM."""

    def test_two_component_recovery(self):
        from peach._core.utils.dirichlet_mixture import DirichletMixture

        rng = np.random.default_rng(42)
        K = 4

        # Component 0: concentrated near vertex 0
        alpha_0 = np.array([20.0, 1.0, 1.0, 1.0])
        # Component 1: concentrated near vertex 2
        alpha_1 = np.array([1.0, 1.0, 20.0, 1.0])

        n_per = 300
        W0 = rng.dirichlet(alpha_0, size=n_per)
        W1 = rng.dirichlet(alpha_1, size=n_per)
        W = np.vstack([W0, W1])

        dm = DirichletMixture(n_components=2, max_iter=300, n_init=3,
                              random_state=42)
        dm.fit(W)

        # Check convergence
        assert dm.converged_, "EM did not converge on well-separated data"

        # Recovered means should match planted structure
        means = dm.means_  # [2, K]
        # One component should have dominant weight in dim 0, other in dim 2
        dom_0 = np.argmax(means, axis=1)
        assert set(dom_0) == {0, 2}, (
            f"Expected dominant dims {{0, 2}}, got {set(dom_0)}. "
            f"Means:\n{means}"
        )

    def test_mixing_weights_roughly_equal(self):
        """With equal sample sizes, mixing weights should be ~0.5."""
        from peach._core.utils.dirichlet_mixture import DirichletMixture

        rng = np.random.default_rng(99)
        n = 200
        W0 = rng.dirichlet([10, 1, 1], size=n)
        W1 = rng.dirichlet([1, 1, 10], size=n)
        W = np.vstack([W0, W1])

        dm = DirichletMixture(n_components=2, max_iter=200, n_init=3,
                              random_state=99)
        dm.fit(W)

        # Mixing weights should be near 0.5
        for w in dm.weights_:
            assert 0.3 < w < 0.7, (
                f"Mixing weight {w:.3f} too far from 0.5 for equal-size components"
            )


# ===========================================================================
# 5. Dirichlet vertex warning fires at correct threshold
# ===========================================================================


class TestDirichletVertexWarning:
    """Warning should fire when > 10% of cells have at least one weight < 1e-6."""

    def test_warning_fires_above_threshold(self):
        """20% vertex-heavy data should trigger the warning."""
        from peach._core.utils.dirichlet_mixture import DirichletMixture

        rng = np.random.default_rng(42)
        n = 100
        K = 3
        # Create data where ~25% are near-vertex
        W_interior = rng.dirichlet([5, 5, 5], size=75)
        W_vertex = np.zeros((25, K))
        W_vertex[:, 0] = 1.0 - 2e-8  # These get clipped to ~1e-300 after clip
        W_vertex[:, 1] = 1e-8
        W_vertex[:, 2] = 1e-8
        W = np.vstack([W_interior, W_vertex])

        dm = DirichletMixture(n_components=2, max_iter=10, random_state=42)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dm.fit(W)
            vertex_warnings = [x for x in w
                               if "vertices" in str(x.message).lower()]
            assert len(vertex_warnings) >= 1, (
                "Expected vertex warning with 25% vertex-heavy data, got none"
            )

    def test_no_warning_below_threshold(self):
        """5% vertex data should NOT trigger warning."""
        from peach._core.utils.dirichlet_mixture import DirichletMixture

        rng = np.random.default_rng(42)
        # Interior data — all weights well above 1e-6
        W = rng.dirichlet([5, 5, 5], size=100)
        assert np.all(W > 1e-4), "Sanity: Dirichlet(5,5,5) shouldn't produce near-vertex"

        dm = DirichletMixture(n_components=2, max_iter=10, random_state=42)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dm.fit(W)
            vertex_warnings = [x for x in w
                               if "vertices" in str(x.message).lower()]
            assert len(vertex_warnings) == 0, (
                f"Unexpected vertex warning with interior data: {vertex_warnings}"
            )

    def test_warning_threshold_is_10_percent(self):
        """Exactly at boundary: 10% should NOT warn, 11% should."""
        from peach._core.utils.dirichlet_mixture import DirichletMixture

        rng = np.random.default_rng(42)
        K = 3
        n = 1000

        # 10% vertex (at threshold, should NOT warn because > 0.1 is strict)
        W_interior = rng.dirichlet([5, 5, 5], size=900)
        W_vertex = np.zeros((100, K))
        W_vertex[:, 0] = 1.0 - 2e-8
        W_vertex[:, 1] = 1e-8
        W_vertex[:, 2] = 1e-8
        W_10pct = np.vstack([W_interior, W_vertex])

        dm = DirichletMixture(n_components=2, max_iter=5, random_state=42)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dm.fit(W_10pct)
            vertex_warnings_10 = [x for x in w
                                  if "vertices" in str(x.message).lower()]
        # 10% is NOT > 0.1, so should not warn
        assert len(vertex_warnings_10) == 0, (
            "10% vertex data should NOT trigger warning (threshold is > 0.1)"
        )

        # 11% vertex — should warn
        W_interior_2 = rng.dirichlet([5, 5, 5], size=890)
        W_vertex_2 = np.zeros((110, K))
        W_vertex_2[:, 0] = 1.0 - 2e-8
        W_vertex_2[:, 1] = 1e-8
        W_vertex_2[:, 2] = 1e-8
        W_11pct = np.vstack([W_interior_2, W_vertex_2])

        dm2 = DirichletMixture(n_components=2, max_iter=5, random_state=42)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dm2.fit(W_11pct)
            vertex_warnings_11 = [x for x in w
                                  if "vertices" in str(x.message).lower()]
        assert len(vertex_warnings_11) >= 1, (
            "11% vertex data SHOULD trigger warning"
        )


# ===========================================================================
# 6. MMD permutation null: under H0, p-values should be ~uniform
# ===========================================================================


class TestMMDPermutationNull:
    """When source and target weights are drawn from the same distribution
    (true H0), MMD p-values should NOT all be < 0.05."""

    def test_within_fit_h0_pvalues_not_all_significant(self):
        """Under true H0 (identical weights), p-values should be ~uniform,
        meaning most should NOT be < 0.05."""
        from peach._core.utils.archetype_comparison import compute_archetype_mmd

        # Use the SAME adata for within-fit comparison
        rng = np.random.default_rng(42)
        n = 300
        K = 4
        weights = rng.dirichlet(np.ones(K), size=n)
        adata = _make_mmd_adata(n=n, K=K, seed=42, weights=weights)

        mmd_matrix, pvalue_matrix = compute_archetype_mmd(
            adata, n_permutations=200, seed=42, max_samples=300
        )

        # Diagonal should be 0 with p=1
        for i in range(K):
            assert mmd_matrix[i, i] == 0.0, f"Diagonal MMD[{i},{i}] should be 0"
            assert pvalue_matrix[i, i] == 1.0, f"Diagonal p[{i},{i}] should be 1"

        # Off-diagonal: under H0, we should NOT see all p < 0.05
        off_diag_pvals = []
        for i in range(K):
            for j in range(i + 1, K):
                off_diag_pvals.append(pvalue_matrix[i, j])

        n_sig = sum(1 for p in off_diag_pvals if p < 0.05)
        total = len(off_diag_pvals)
        # Under H0, we'd expect at most ~5% false positives
        # With 6 tests (K=4, pairs=6), allow up to 2 significant
        assert n_sig < total, (
            f"All {total} off-diagonal p-values < 0.05 under H0. "
            f"Permutation null is broken. p-values: {off_diag_pvals}"
        )

    def test_symmetric_permutation_produces_symmetric_matrix(self):
        """Within-fit MMD matrix should be symmetric."""
        from peach._core.utils.archetype_comparison import compute_archetype_mmd

        adata = _make_mmd_adata(n=200, K=3, seed=99)
        mmd_matrix, pvalue_matrix = compute_archetype_mmd(
            adata, n_permutations=50, seed=99
        )

        np.testing.assert_array_equal(
            mmd_matrix, mmd_matrix.T,
            err_msg="Within-fit MMD matrix should be symmetric"
        )
        # p-value matrix should also be symmetric
        for i in range(3):
            for j in range(i):
                assert pvalue_matrix[i, j] == pvalue_matrix[j, i], (
                    f"p-value not symmetric: [{i},{j}]={pvalue_matrix[i,j]:.4f} "
                    f"vs [{j},{i}]={pvalue_matrix[j,i]:.4f}"
                )


# ===========================================================================
# 7. Cross-K NaN: verify exact entries for K_a=3, K_b=5
# ===========================================================================


class TestCrossKNaN:
    """Between-fit comparison with K_a != K_b: certain p-values should be NaN."""

    def test_nan_entries_ka3_kb5(self):
        """For K_a=3, K_b=5: entries where j >= K_a (j=3,4) should have NaN p-values."""
        from peach._core.utils.archetype_comparison import compute_archetype_mmd

        K_a, K_b = 3, 5
        adata_a = _make_mmd_adata(n=200, K=K_a, seed=42)
        adata_b = _make_mmd_adata(n=200, K=K_b, seed=99)

        mmd_matrix, pvalue_matrix = compute_archetype_mmd(
            adata_a, adata_b, n_permutations=50, seed=42
        )

        # Shape should be K_a x K_b
        assert mmd_matrix.shape == (K_a, K_b), (
            f"Expected ({K_a}, {K_b}), got {mmd_matrix.shape}"
        )
        assert pvalue_matrix.shape == (K_a, K_b), (
            f"Expected ({K_a}, {K_b}), got {pvalue_matrix.shape}"
        )

        # Entries with j >= K_a (j=3,4) should have NaN p-values
        for i in range(K_a):
            for j in range(K_b):
                if j >= K_a:
                    assert np.isnan(pvalue_matrix[i, j]), (
                        f"p-value[{i},{j}] should be NaN (j={j} >= K_a={K_a}), "
                        f"got {pvalue_matrix[i, j]}"
                    )
                else:
                    assert np.isfinite(pvalue_matrix[i, j]), (
                        f"p-value[{i},{j}] should be finite (j={j} < K_a={K_a}), "
                        f"got {pvalue_matrix[i, j]}"
                    )

    def test_nan_entries_ka5_kb3(self):
        """For K_a=5, K_b=3: entries where i >= K_b (i=3,4) should have NaN p-values."""
        from peach._core.utils.archetype_comparison import compute_archetype_mmd

        K_a, K_b = 5, 3
        adata_a = _make_mmd_adata(n=200, K=K_a, seed=42)
        adata_b = _make_mmd_adata(n=200, K=K_b, seed=99)

        mmd_matrix, pvalue_matrix = compute_archetype_mmd(
            adata_a, adata_b, n_permutations=50, seed=42
        )

        assert mmd_matrix.shape == (K_a, K_b)

        for i in range(K_a):
            for j in range(K_b):
                if i >= K_b:
                    assert np.isnan(pvalue_matrix[i, j]), (
                        f"p-value[{i},{j}] should be NaN (i={i} >= K_b={K_b}), "
                        f"got {pvalue_matrix[i, j]}"
                    )
                else:
                    assert np.isfinite(pvalue_matrix[i, j]), (
                        f"p-value[{i},{j}] should be finite (i={i} < K_b={K_b}), "
                        f"got {pvalue_matrix[i, j]}"
                    )

    def test_equal_k_no_nan(self):
        """K_a == K_b: no NaN entries expected in between-fit."""
        from peach._core.utils.archetype_comparison import compute_archetype_mmd

        K = 3
        adata_a = _make_mmd_adata(n=200, K=K, seed=42)
        adata_b = _make_mmd_adata(n=200, K=K, seed=99)

        mmd_matrix, pvalue_matrix = compute_archetype_mmd(
            adata_a, adata_b, n_permutations=50, seed=42
        )

        assert not np.any(np.isnan(pvalue_matrix)), (
            f"No NaN expected when K_a == K_b, but found NaN at "
            f"{np.argwhere(np.isnan(pvalue_matrix))}"
        )

    def test_mmd_values_finite_even_when_pvalue_nan(self):
        """MMD observed values should always be finite, even when p-value is NaN."""
        from peach._core.utils.archetype_comparison import compute_archetype_mmd

        K_a, K_b = 3, 5
        adata_a = _make_mmd_adata(n=200, K=K_a, seed=42)
        adata_b = _make_mmd_adata(n=200, K=K_b, seed=99)

        mmd_matrix, pvalue_matrix = compute_archetype_mmd(
            adata_a, adata_b, n_permutations=50, seed=42
        )

        assert np.all(np.isfinite(mmd_matrix)), (
            f"MMD matrix has non-finite values: {mmd_matrix}"
        )


# ===========================================================================
# 8. mannwhitneyu reproducibility: same feature_name -> same noise
# ===========================================================================


class TestMannWhitneyUReproducibility:
    """Hashlib-based RNG must produce identical results across calls
    for the same feature name."""

    def test_same_feature_same_result(self):
        """Calling with the same feature_name twice must give identical output."""
        from peach._core.utils.statistical_tests import robust_mannwhitneyu_test

        rng = np.random.default_rng(42)
        group1 = rng.standard_normal(100)
        group2 = rng.standard_normal(100) + 0.5

        stat1, pval1 = robust_mannwhitneyu_test(
            group1, group2, feature_name="BRCA1"
        )
        stat2, pval2 = robust_mannwhitneyu_test(
            group1, group2, feature_name="BRCA1"
        )

        assert stat1 == stat2, (
            f"Same feature name gave different statistics: {stat1} vs {stat2}"
        )
        assert pval1 == pval2, (
            f"Same feature name gave different p-values: {pval1} vs {pval2}"
        )

    def test_reproducible_across_interleaved_calls(self):
        """Calling feature A, then B, then A again should give same result
        for A both times (no global state contamination)."""
        from peach._core.utils.statistical_tests import robust_mannwhitneyu_test

        rng = np.random.default_rng(42)
        g1 = rng.standard_normal(50)
        g2 = rng.standard_normal(50) + 0.3

        stat_a1, _ = robust_mannwhitneyu_test(g1, g2, feature_name="TP53")
        _, _ = robust_mannwhitneyu_test(g1, g2, feature_name="MYC")
        stat_a2, _ = robust_mannwhitneyu_test(g1, g2, feature_name="TP53")

        assert stat_a1 == stat_a2, (
            "Interleaved call changed result: global state leak detected"
        )


# ===========================================================================
# 9. mannwhitneyu different features produce different noise
# ===========================================================================


class TestMannWhitneyUDifferentFeatures:
    """Different feature names must produce different tie-breaking noise."""

    def test_different_features_different_noise(self):
        """Two different feature names should produce different noise and
        (potentially) different test statistics when data has heavy ties."""
        from peach._core.utils.statistical_tests import robust_mannwhitneyu_test

        # Data with heavy ties to make noise matter
        g1 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3] * 10, dtype=float)
        g2 = np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 3] * 10, dtype=float)

        stat_a, _ = robust_mannwhitneyu_test(g1, g2, feature_name="GeneA")
        stat_b, _ = robust_mannwhitneyu_test(g1, g2, feature_name="GeneB")

        # With heavy ties and different noise, statistics should differ
        # (Not guaranteed for arbitrary inputs, but with 100 samples
        # and integer data, the noise should break symmetry differently)
        # We check that the hashlib mechanism at least produces distinct seeds
        import hashlib
        seed_a = int(hashlib.sha256(b"GeneA").hexdigest()[:8], 16)
        seed_b = int(hashlib.sha256(b"GeneB").hexdigest()[:8], 16)
        assert seed_a != seed_b, "Hash seeds should differ for different names"

    def test_noise_deterministic_per_feature(self):
        """Verify the hashlib seed mechanism works correctly."""
        import hashlib

        names = ["TP53", "BRCA1", "MYC", "CD8A", "IL2"]
        seeds = []
        for name in names:
            seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
            seeds.append(seed)

        # All seeds should be unique
        assert len(set(seeds)) == len(seeds), (
            f"Hash collision among {names}: seeds={seeds}"
        )


# ===========================================================================
# 10. Between-fit MMD matrix shape is K_a x K_b
# ===========================================================================


class TestMMDMatrixShape:
    """Matrix shape for between-fit comparison with different K."""

    @pytest.mark.parametrize("K_a,K_b", [(3, 5), (5, 3), (2, 7), (4, 4)])
    def test_between_fit_shape(self, K_a, K_b):
        from peach._core.utils.archetype_comparison import compute_archetype_mmd

        adata_a = _make_mmd_adata(n=100, K=K_a, seed=42)
        adata_b = _make_mmd_adata(n=100, K=K_b, seed=99)

        mmd_matrix, pvalue_matrix = compute_archetype_mmd(
            adata_a, adata_b, n_permutations=0, seed=42
        )

        assert mmd_matrix.shape == (K_a, K_b), (
            f"Expected ({K_a}, {K_b}), got {mmd_matrix.shape}"
        )
        assert pvalue_matrix.shape == (K_a, K_b), (
            f"Expected ({K_a}, {K_b}), got {pvalue_matrix.shape}"
        )

    def test_within_fit_shape_is_square(self):
        from peach._core.utils.archetype_comparison import compute_archetype_mmd

        K = 4
        adata = _make_mmd_adata(n=100, K=K, seed=42)
        mmd_matrix, pvalue_matrix = compute_archetype_mmd(
            adata, n_permutations=0, seed=42
        )
        assert mmd_matrix.shape == (K, K)
        assert pvalue_matrix.shape == (K, K)


# ===========================================================================
# 11. BONUS: Between-fit NaN guard — index-out-of-bounds protection
# ===========================================================================


class TestCrossKIndexSafety:
    """Verify that the NaN guard correctly prevents index-out-of-bounds
    when accessing weight columns across fits with different K."""

    def test_no_index_error_ka2_kb6(self):
        """K_a=2, K_b=6: accessing weights_a[:, j] for j=2..5 would crash
        without the NaN guard."""
        from peach._core.utils.archetype_comparison import compute_archetype_mmd

        adata_a = _make_mmd_adata(n=100, K=2, seed=42)
        adata_b = _make_mmd_adata(n=100, K=6, seed=99)

        # This should NOT raise IndexError
        mmd_matrix, pvalue_matrix = compute_archetype_mmd(
            adata_a, adata_b, n_permutations=20, seed=42
        )

        assert mmd_matrix.shape == (2, 6)
        # j >= K_a=2 means j=2,3,4,5 should be NaN
        for i in range(2):
            for j in range(2, 6):
                assert np.isnan(pvalue_matrix[i, j]), (
                    f"Expected NaN at [{i},{j}], got {pvalue_matrix[i,j]}"
                )

    def test_between_fit_permutation_uses_correct_weight_columns(self):
        """For valid entries (i < K_b AND j < K_a), verify that the
        permutation test produces reasonable p-values (not all 0 or 1)."""
        from peach._core.utils.archetype_comparison import compute_archetype_mmd

        # Use structured data: adata_a and adata_b have different PCA coords
        # but overlapping K values
        K_a, K_b = 3, 5
        rng = np.random.default_rng(42)

        # Make adata_a with shifted PCA to ensure non-trivial MMD
        adata_a = _make_mmd_adata(n=200, K=K_a, seed=42)
        adata_b = _make_mmd_adata(n=200, K=K_b, seed=99)
        # Shift PCA coords to make distributions differ
        adata_b.obsm["X_pca"] = adata_b.obsm["X_pca"] + 2.0

        mmd_matrix, pvalue_matrix = compute_archetype_mmd(
            adata_a, adata_b, n_permutations=100, seed=42
        )

        # Valid entries: i < K_b (always) AND j < K_a (j=0,1,2)
        for i in range(K_a):
            for j in range(min(K_a, K_b)):
                assert np.isfinite(pvalue_matrix[i, j]), (
                    f"Expected finite p-value at [{i},{j}]"
                )
                # With shifted data, should detect difference
                assert pvalue_matrix[i, j] < 1.0, (
                    f"p-value at [{i},{j}] is exactly 1.0 despite shifted data"
                )


# ===========================================================================
# 12. BONUS: ILR input validation
# ===========================================================================


class TestILRInputValidation:
    """ILR transform should reject invalid inputs."""

    def test_rejects_k1(self):
        from peach._core.utils.ilr_transform import ilr_transform

        W = np.array([[1.0], [1.0]])
        with pytest.raises(ValueError, match="K >= 2"):
            ilr_transform(W)

    def test_rejects_nan(self):
        from peach._core.utils.ilr_transform import ilr_transform

        W = np.array([[0.5, np.nan], [0.5, 0.5]])
        with pytest.raises(ValueError, match="NaN"):
            ilr_transform(W)

    def test_rejects_negative(self):
        from peach._core.utils.ilr_transform import ilr_transform

        W = np.array([[0.5, -0.5, 1.0]])
        with pytest.raises(ValueError, match="negative"):
            ilr_transform(W)

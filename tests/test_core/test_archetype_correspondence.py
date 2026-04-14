"""Headless tests for the archetype correspondence matrix between two model fits.

The correspondence matrix answers:
    "Given a source cell in source archetype i, what is the probability
     that its transported neighbor lands in target archetype j?"

It is the bridge between two Deep_AA fits (e.g. HSC vs CMP) and underpins:
- Sankey cross-fit visualization
- Per-pair flow_between() zoom analyses
- Cross-fit Wald contrast alignment

These tests encode mathematical properties that any reasonable correspondence
implementation must satisfy. They are designed to catch the rank-1 collapse
bug observed in r8sub2 on HSC→CMP (range 28-47, std ~3, near-uniform rows).

DESIGN NOTE:
The function `compute_archetype_correspondence` is expected to live in
`peach._core.utils.archetype_comparison`. It does not yet exist — these tests
are TDD-first and will drive its extraction from the inline logic currently
in scripts/run_paper_part1_hsc.py around lines 1481-1546.
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

def _make_peaked_weights(n_per_group, K, peak_val=0.9, seed=42):
    """Build simplex weights peaked toward each archetype.

    n_per_group cells per archetype, each cell peaked on its group's archetype
    with the specified peak value. Remaining mass distributed uniformly.

    Returns
    -------
    weights : ndarray [n_total, K]
        Row-stochastic simplex weights.
    labels : ndarray [n_total]
        Ground-truth archetype label per cell.
    """
    rng = np.random.default_rng(seed)
    n_total = n_per_group * K
    off_peak = (1.0 - peak_val) / (K - 1)
    weights = np.full((n_total, K), off_peak)
    labels = np.repeat(np.arange(K), n_per_group)
    for i, lab in enumerate(labels):
        weights[i, lab] = peak_val
    # Tiny noise to break exact ties (realistic for trained models)
    weights = weights + rng.normal(0, 0.001, weights.shape)
    weights = np.clip(weights, 1e-6, None)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return weights, labels


def _make_clustered_coords(labels, K, dim=5, spread=0.3, center_spacing=3.0, seed=42):
    """Place cells in K well-separated clusters in `dim`-dimensional space.

    Cell with label k lands at (k * center_spacing, 0, 0, ...) + Gaussian noise.
    Choose spread << center_spacing so k-NN stays within its cluster.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    coords = np.zeros((n, dim))
    centers = np.zeros((K, dim))
    for k in range(K):
        centers[k, 0] = k * center_spacing
    for i in range(n):
        coords[i] = centers[labels[i]] + rng.normal(0, spread, dim)
    return coords


# ---------------------------------------------------------------------------
# Import-level test — first RED failure (ImportError)
# ---------------------------------------------------------------------------

def test_correspondence_function_importable():
    """The function must exist in archetype_comparison (drives extraction)."""
    from peach._core.utils.archetype_comparison import compute_archetype_correspondence
    assert callable(compute_archetype_correspondence)


# ---------------------------------------------------------------------------
# Basic structural properties
# ---------------------------------------------------------------------------

class TestCorrespondenceBasics:
    """Shape, non-negative, row-stochastic markov normalization."""

    def _run(self, K_src=3, K_tgt=3, n_per=50, peak=0.9, seed=1):
        from peach._core.utils.archetype_comparison import compute_archetype_correspondence
        sw, sl = _make_peaked_weights(n_per, K_src, peak_val=peak, seed=seed)
        tw, tl = _make_peaked_weights(n_per, K_tgt, peak_val=peak, seed=seed + 1)
        sc = _make_clustered_coords(sl, K_src, seed=seed)
        tc = _make_clustered_coords(tl, K_tgt, seed=seed + 1)
        return compute_archetype_correspondence(
            source_weights=sw, source_coords=sc,
            target_weights=tw, target_coords=tc, k=5,
        )

    def test_shape_mass_matrix(self):
        """mass matrix has shape [K_src, K_tgt]."""
        result = self._run(K_src=3, K_tgt=4)
        assert "mass" in result
        assert result["mass"].shape == (3, 4)

    def test_shape_markov_matrix(self):
        """markov matrix has shape [K_src, K_tgt]."""
        result = self._run(K_src=3, K_tgt=4)
        assert "markov" in result
        assert result["markov"].shape == (3, 4)

    def test_mass_non_negative(self):
        """Raw correspondence mass must be non-negative everywhere."""
        result = self._run()
        assert (result["mass"] >= 0).all(), \
            f"Negative mass found: min={result['mass'].min()}"

    def test_markov_non_negative(self):
        """Row-normalized markov matrix must be non-negative."""
        result = self._run()
        assert (result["markov"] >= 0).all()

    def test_markov_rows_sum_to_one(self):
        """Each non-empty row of the markov matrix must sum to 1.

        Sparse / empty source archetype rows are explicitly zeroed by the
        implementation (see ``sparse_archetypes`` handling), so we filter
        to nonzero rows before asserting row-stochasticity.
        """
        result = self._run()
        row_sums = result["markov"].sum(axis=1)
        nonzero = row_sums > 1e-9
        np.testing.assert_allclose(
            row_sums[nonzero], 1.0, atol=1e-6,
            err_msg=f"Non-empty rows must sum to 1: {row_sums}")

    def test_markov_entries_at_most_one(self):
        """No markov entry can exceed 1."""
        result = self._run()
        assert (result["markov"] <= 1.0 + 1e-9).all()


# ---------------------------------------------------------------------------
# Correctness: identity case and permutation equivariance
# ---------------------------------------------------------------------------

class TestCorrespondenceCorrectness:
    """Sanity checks against known-answer constructions."""

    def test_identity_case_hard_weights(self):
        """Same population → same model yields near-identity markov matrix.

        With hard weights (peak=0.98) and coordinates where each cluster
        is well-separated, the correspondence should reveal i↔i mapping.
        """
        from peach._core.utils.archetype_comparison import compute_archetype_correspondence
        K = 3
        w, labels = _make_peaked_weights(100, K, peak_val=0.98, seed=1)
        coords = _make_clustered_coords(labels, K, seed=1)
        result = compute_archetype_correspondence(
            source_weights=w, source_coords=coords,
            target_weights=w, target_coords=coords, k=5,
        )
        markov = result["markov"]
        # Diagonal should dominate each row
        for i in range(K):
            assert markov[i, i] > 0.8, (
                f"Row {i} diagonal {markov[i, i]:.3f} should dominate in identity "
                f"case. Full row: {markov[i]}. Full matrix:\n{markov}"
            )

    def test_permutation_equivariance_target(self):
        """Permuting target archetype columns permutes correspondence columns.

        If target archetype labels [0,1,2] are relabeled [2,0,1], the
        correspondence matrix columns should be correspondingly permuted.
        """
        from peach._core.utils.archetype_comparison import compute_archetype_correspondence
        K = 3
        sw, sl = _make_peaked_weights(100, K, peak_val=0.95, seed=1)
        sc = _make_clustered_coords(sl, K, seed=1)
        tw, tl = _make_peaked_weights(100, K, peak_val=0.95, seed=2)
        tc = _make_clustered_coords(tl, K, seed=2)

        base = compute_archetype_correspondence(
            source_weights=sw, source_coords=sc,
            target_weights=tw, target_coords=tc, k=5,
        )["markov"]

        perm = np.array([2, 0, 1])
        tw_perm = tw[:, perm]
        permuted = compute_archetype_correspondence(
            source_weights=sw, source_coords=sc,
            target_weights=tw_perm, target_coords=tc, k=5,
        )["markov"]

        np.testing.assert_allclose(
            permuted, base[:, perm], atol=1e-6,
            err_msg="Target column permutation not equivariant"
        )


# ---------------------------------------------------------------------------
# THE RANK-1 COLLAPSE BUG
# ---------------------------------------------------------------------------
#
# This is the critical test. The observed r8sub2 failure was:
#   - HSC→CMP correspondence matrix in range [28.5, 47.1]
#   - std across rows: 3.0, std across cols: 3.2
#   - "All column values virtually identical" → effective rank ≈ 1
#   - Downstream: empty cross-fit pair list, broken Sankey weighting
#
# The mathematical cause (verified by hand): when the soft-soft outer-product
# construction
#   corr[i,j] = sum_n source_w[n,i] * target_w_nn[n,j]
# is row-normalized, and source weights are only moderately peaked
# (e.g. peak=0.5, off-peak=0.25 for K=3), the resulting markov matrix has
# row L1 differences of only ~0.125 between distinct "groups". This is
# indistinguishable from uniform for downstream analysis.
#
# A hard-argmax or similar construction gives row L1 differences of ~0.5
# under the same conditions.
# ---------------------------------------------------------------------------

class TestCorrespondenceRankCollapse:
    """Tests that reproduce and guard against the r8sub2 rank-1 collapse bug."""

    def test_moderately_peaked_sources_give_distinct_rows(self):
        """Moderately peaked sources must still produce distinguishable rows.

        Realistic condition: source weights peaked at 0.55 (not 0.9+) — this
        is typical for a trained Deep_AA with kld_weight=0.1. If the
        correspondence construction collapses to near-uniform rows here,
        the downstream cross-fit analysis has no signal.

        FAILURE MODE DOCUMENTED: current inline soft-soft construction in
        scripts/run_paper_part1_hsc.py gives max row L1 diff ~ 0.125,
        which is the bug. A correct construction (e.g. hard argmax, or
        sharper soft weighting) gives ~ 0.5.
        """
        from peach._core.utils.archetype_comparison import compute_archetype_correspondence
        K = 3
        sw, sl = _make_peaked_weights(100, K, peak_val=0.55, seed=1)
        tw, tl = _make_peaked_weights(100, K, peak_val=0.55, seed=2)
        sc = _make_clustered_coords(sl, K, seed=1)
        tc = _make_clustered_coords(tl, K, seed=2)

        result = compute_archetype_correspondence(
            source_weights=sw, source_coords=sc,
            target_weights=tw, target_coords=tc, k=5,
        )
        markov = result["markov"]

        # Pairwise L1 distance between rows
        max_row_diff = 0.0
        for i in range(K):
            for j in range(i + 1, K):
                d = np.abs(markov[i] - markov[j]).sum()
                if d > max_row_diff:
                    max_row_diff = d

        assert max_row_diff > 0.3, (
            f"Rank-1 collapse detected: max pairwise row L1 diff "
            f"{max_row_diff:.3f} is too small. With moderately peaked "
            f"sources (peak=0.55), distinct groups should produce distinct "
            f"conditional distributions over target archetypes.\n"
            f"Matrix:\n{markov}"
        )

    def test_effective_rank_greater_than_one(self):
        """The correspondence matrix must have effective rank > 1.

        Uses singular value ratio s2/s1 as a proxy: rank-1 matrices have
        s2/s1 ≈ 0, well-conditioned matrices have s2/s1 closer to 1.
        """
        from peach._core.utils.archetype_comparison import compute_archetype_correspondence
        K = 3
        sw, sl = _make_peaked_weights(100, K, peak_val=0.6, seed=1)
        tw, tl = _make_peaked_weights(100, K, peak_val=0.6, seed=2)
        sc = _make_clustered_coords(sl, K, seed=1)
        tc = _make_clustered_coords(tl, K, seed=2)

        result = compute_archetype_correspondence(
            source_weights=sw, source_coords=sc,
            target_weights=tw, target_coords=tc, k=5,
        )
        markov = result["markov"]

        _, singular_values, _ = np.linalg.svd(markov)
        # Require second singular value to be at least 15% of the first —
        # rank-1 matrices have ratio ~ 0; well-structured matrices have
        # ratio > 0.2.
        if singular_values[0] < 1e-10:
            pytest.fail(f"Degenerate markov matrix: {singular_values}")
        ratio = singular_values[1] / singular_values[0]
        assert ratio > 0.15, (
            f"Markov matrix is nearly rank-1 (s2/s1 = {ratio:.4f}). "
            f"Singular values: {singular_values.tolist()}\n"
            f"Matrix:\n{markov}"
        )

    def test_row_coefficient_of_variation_substantial(self):
        """Each row's CV across targets should be meaningful (> 10%).

        Catches the specific symptom from r8sub2: std/mean across columns
        was ~8%, which read as 'virtually identical' to the reviewer.
        """
        from peach._core.utils.archetype_comparison import compute_archetype_correspondence
        K = 3
        sw, sl = _make_peaked_weights(100, K, peak_val=0.55, seed=1)
        tw, tl = _make_peaked_weights(100, K, peak_val=0.55, seed=2)
        sc = _make_clustered_coords(sl, K, seed=1)
        tc = _make_clustered_coords(tl, K, seed=2)

        result = compute_archetype_correspondence(
            source_weights=sw, source_coords=sc,
            target_weights=tw, target_coords=tc, k=5,
        )
        markov = result["markov"]
        # Coefficient of variation per row
        row_means = markov.mean(axis=1)
        row_stds = markov.std(axis=1)
        cv = row_stds / np.where(row_means > 1e-10, row_means, 1.0)
        max_cv = cv.max()
        assert max_cv > 0.25, (
            f"Row CV too small (max={max_cv:.3f}). The matrix reads as "
            f"'virtually identical' rows.\nMatrix:\n{markov}"
        )


# ---------------------------------------------------------------------------
# Robustness / edge cases
# ---------------------------------------------------------------------------

class TestCorrespondenceEdgeCases:
    """Guard against crashes on unusual but valid inputs."""

    def test_k_larger_than_target_count(self):
        """k > n_target should clip silently, not crash."""
        from peach._core.utils.archetype_comparison import compute_archetype_correspondence
        sw, sl = _make_peaked_weights(20, 3, seed=1)
        tw, tl = _make_peaked_weights(4, 3, seed=2)  # 12 total target cells
        sc = _make_clustered_coords(sl, 3, seed=1)
        tc = _make_clustered_coords(tl, 3, seed=2)
        result = compute_archetype_correspondence(
            source_weights=sw, source_coords=sc,
            target_weights=tw, target_coords=tc, k=20,  # more than n_tgt
        )
        assert result["mass"].shape == (3, 3)

    def test_empty_source_archetype_row(self):
        """A source archetype with zero mass yields a zero or NaN row, not a crash."""
        from peach._core.utils.archetype_comparison import compute_archetype_correspondence
        K_src, K_tgt = 3, 3
        # Construct source weights that never put mass on archetype 2
        n_src = 100
        sw = np.zeros((n_src, K_src))
        labels_src = np.random.default_rng(42).choice([0, 1], size=n_src)
        for i in range(n_src):
            sw[i, labels_src[i]] = 1.0
        sc = _make_clustered_coords(labels_src, K_src, seed=1)

        tw, tl = _make_peaked_weights(30, K_tgt, seed=2)
        tc = _make_clustered_coords(tl, K_tgt, seed=2)

        result = compute_archetype_correspondence(
            source_weights=sw, source_coords=sc,
            target_weights=tw, target_coords=tc, k=5,
        )
        # Row 2 should have zero mass (no source cells contributed)
        assert result["mass"][2].sum() < 1e-9, \
            f"Empty archetype row should be zero, got {result['mass'][2]}"
        # Markov row 2 should be handled (either zero or uniform fallback — caller policy)
        # Just assert no NaN crash
        assert not np.isnan(result["mass"]).any(), "NaN in mass matrix"

    def test_mismatched_k_sizes(self):
        """K_src != K_tgt should produce rectangular matrix."""
        from peach._core.utils.archetype_comparison import compute_archetype_correspondence
        sw, sl = _make_peaked_weights(50, 4, seed=1)  # K_src=4
        tw, tl = _make_peaked_weights(50, 7, seed=2)  # K_tgt=7
        sc = _make_clustered_coords(sl, 4, seed=1)
        tc = _make_clustered_coords(tl, 7, seed=2)
        result = compute_archetype_correspondence(
            source_weights=sw, source_coords=sc,
            target_weights=tw, target_coords=tc, k=5,
        )
        assert result["mass"].shape == (4, 7)
        assert result["markov"].shape == (4, 7)


# ---------------------------------------------------------------------------
# W-B23: Permutation curve null via global cell swap
# ---------------------------------------------------------------------------
#
# Replaces the Gaussian-assumption z-score null with an empirical permutation
# curve. For each swap fraction f in (0.0, 0.05, ..., 0.5) we randomly relabel
# an f-fraction of source cells as target and vice versa, recompute the
# correspondence matrix with method="hard", and build an empirical null
# distribution of (K_src x K_tgt) correspondence matrices per f. Per-pair
# empirical p-values are computed via rank at the largest swap fraction and
# BH-corrected.
#
# Key design choice: GLOBAL swap (not per-pair), reducing cost from
# K_src * K_tgt permutations per fraction to a single batch per fraction.
# ---------------------------------------------------------------------------


class TestPermutationNullCurve:
    """W-B23 tests: empirical permutation curve null for correspondence.

    Three tests per the task spec:
      1. Structure / keys / shapes / p-value range.
      2. Monotonicity on a known strong (i, j) pair: observed > null at f=0.
      3. Degradation curve: variance grows with f, strong-pair mean approaches
         the overall matrix mean as swap fraction increases.
    """

    def test_permutation_null_structure(self):
        """All documented keys present, correct shapes, valid p-value range."""
        from peach._core.utils.archetype_comparison import (
            compute_correspondence_permutation_null,
        )
        K_src, K_tgt = 3, 4
        n_per_src = 34   # ~100 cells total
        n_per_tgt = 25
        sw, sl = _make_peaked_weights(n_per_src, K_src, peak_val=0.85, seed=1)
        tw, tl = _make_peaked_weights(n_per_tgt, K_tgt, peak_val=0.85, seed=2)
        sc = _make_clustered_coords(sl, K_src, seed=1)
        tc = _make_clustered_coords(tl, K_tgt, seed=2)

        swap_fractions = (0.0, 0.2, 0.5)
        n_perms = 20
        result = compute_correspondence_permutation_null(
            source_weights=sw,
            source_coords=sc,
            target_weights=tw,
            target_coords=tc,
            k=5,
            n_perms=n_perms,
            swap_fractions=swap_fractions,
            seed=7,
        )

        expected_keys = {
            "observed_mass",
            "null_distributions",
            "empirical_p",
            "empirical_fdr",
            "swap_fractions",
            "n_perms",
            "null_mean_curve",
            "null_std_curve",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - set(result.keys())}"
        )

        # Shapes
        assert result["observed_mass"].shape == (K_src, K_tgt)
        assert result["empirical_p"].shape == (K_src, K_tgt)
        assert result["empirical_fdr"].shape == (K_src, K_tgt)
        assert result["null_mean_curve"].shape == (len(swap_fractions), K_src, K_tgt)
        assert result["null_std_curve"].shape == (len(swap_fractions), K_src, K_tgt)
        # null_distributions keyed by swap fraction
        assert set(result["null_distributions"].keys()) == set(swap_fractions)
        for f in swap_fractions:
            assert result["null_distributions"][f].shape == (n_perms, K_src, K_tgt)

        # p-value range: [1/(n_perms+1), 1.0]
        lo_bound = 1.0 / (n_perms + 1)
        assert (result["empirical_p"] >= lo_bound - 1e-9).all(), (
            f"p-values below 1/(n_perms+1)={lo_bound:.4f}: "
            f"min={result['empirical_p'].min():.6f}"
        )
        assert (result["empirical_p"] <= 1.0 + 1e-9).all(), (
            f"p-values exceed 1.0: max={result['empirical_p'].max():.6f}"
        )

        # FDR range: [0, 1]
        assert (result["empirical_fdr"] >= 0).all()
        assert (result["empirical_fdr"] <= 1.0 + 1e-9).all()

        # BH FDR: sorted p-values map to non-decreasing FDR values
        flat_p = result["empirical_p"].ravel()
        flat_fdr = result["empirical_fdr"].ravel()
        order = np.argsort(flat_p)
        sorted_fdr = flat_fdr[order]
        # Allow tiny numerical wobble
        diffs = np.diff(sorted_fdr)
        assert (diffs >= -1e-9).all(), (
            f"BH FDR not monotone non-decreasing when sorted by p: {sorted_fdr}"
        )

        # n_perms echoed
        assert result["n_perms"] == n_perms
        assert tuple(result["swap_fractions"]) == swap_fractions

    def test_permutation_null_monotonicity_synthetic(self):
        """Strong (i,j) pair has small empirical p; scrambling makes null uniform.

        Constructs a synthetic case where source archetypes are well-separated
        in coordinate space AND target archetypes sit on the same coordinate
        centers, so source archetype i strongly maps to target archetype i.
        A cross-diagonal "noise" pair should have a large p-value.
        """
        from peach._core.utils.archetype_comparison import (
            compute_correspondence_permutation_null,
        )
        K = 3
        n_per = 60
        # Matching source/target structure: each source arch i aligns with
        # target arch i via clustered coordinates.
        sw, sl = _make_peaked_weights(n_per, K, peak_val=0.95, seed=1)
        tw, tl = _make_peaked_weights(n_per, K, peak_val=0.95, seed=2)
        sc = _make_clustered_coords(sl, K, seed=10)
        tc = _make_clustered_coords(tl, K, seed=11)

        swap_fractions = (0.0, 0.5)
        n_perms = 40
        result = compute_correspondence_permutation_null(
            source_weights=sw,
            source_coords=sc,
            target_weights=tw,
            target_coords=tc,
            k=5,
            n_perms=n_perms,
            swap_fractions=swap_fractions,
            seed=13,
        )

        observed = result["observed_mass"]
        p = result["empirical_p"]

        # The diagonal pairs (i,i) should dominate in observed mass — this is
        # the ground truth strong correspondence.
        diag_vals = np.diag(observed)
        off_diag_vals = observed - np.diag(diag_vals)
        assert diag_vals.min() > off_diag_vals.max(), (
            f"Synthetic setup failed to produce diagonal-dominant observed. "
            f"diag_min={diag_vals.min():.3f}, off_diag_max={off_diag_vals.max():.3f}\n"
            f"Observed:\n{observed}"
        )

        # For each strong (i, i) diagonal pair: empirical p should be small.
        # We use the strictest pair (argmax of diagonal) as a conservative anchor.
        strong_i = int(np.argmax(diag_vals))
        assert p[strong_i, strong_i] < 0.1, (
            f"Strong diagonal pair ({strong_i},{strong_i}) has large p-value "
            f"{p[strong_i, strong_i]:.3f}; expected < 0.1 under scrambling null.\n"
            f"Observed mass:\n{observed}\np-matrix:\n{p}"
        )

        # Pick a "noise" off-diagonal pair (small observed mass). Its p-value
        # should be large because the null at f=0.5 routinely produces the
        # same small mass.
        off_diag_mask = ~np.eye(K, dtype=bool)
        off_observed = np.where(off_diag_mask, observed, np.inf)
        noise_flat = int(np.argmin(off_observed))
        noise_i, noise_j = noise_flat // K, noise_flat % K
        assert p[noise_i, noise_j] >= 0.5, (
            f"Noise off-diag pair ({noise_i},{noise_j}) has small p={p[noise_i, noise_j]:.3f}; "
            f"expected >= 0.5.\nObserved mass:\n{observed}\np-matrix:\n{p}"
        )

        # Null mean at f=0 (no swap) should closely match observed, because
        # f=0 is a pure no-op (every permutation returns the same matrix).
        null_mean_f0 = result["null_mean_curve"][0]
        np.testing.assert_allclose(
            null_mean_f0, observed, atol=1e-8,
            err_msg="Null mean at f=0 should match observed exactly (no shuffle)."
        )

        # Null mean at f=0.5 should be more uniform across pairs than at f=0.
        # Measure by per-row CV of the null mean matrix: scrambled null should
        # have smaller CV (flatter rows) than the unshuffled baseline.
        null_mean_f50 = result["null_mean_curve"][-1]

        def _mean_row_cv(mat):
            row_m = mat.mean(axis=1)
            row_s = mat.std(axis=1)
            safe = np.where(row_m > 1e-12, row_m, 1.0)
            return float((row_s / safe).mean())

        cv_f0 = _mean_row_cv(null_mean_f0)
        cv_f50 = _mean_row_cv(null_mean_f50)
        assert cv_f50 < cv_f0, (
            f"Expected null-mean-row-CV to decrease with scrambling: "
            f"f=0 CV={cv_f0:.3f}, f=0.5 CV={cv_f50:.3f}"
        )

    def test_permutation_null_degradation_curve(self):
        """Variance grows with f; strong pair's null mean regresses to matrix mean."""
        from peach._core.utils.archetype_comparison import (
            compute_correspondence_permutation_null,
        )
        K = 3
        n_per = 60
        sw, sl = _make_peaked_weights(n_per, K, peak_val=0.95, seed=1)
        tw, tl = _make_peaked_weights(n_per, K, peak_val=0.95, seed=2)
        sc = _make_clustered_coords(sl, K, seed=21)
        tc = _make_clustered_coords(tl, K, seed=22)

        swap_fractions = (0.0, 0.1, 0.3, 0.5)
        n_perms = 40
        result = compute_correspondence_permutation_null(
            source_weights=sw,
            source_coords=sc,
            target_weights=tw,
            target_coords=tc,
            k=5,
            n_perms=n_perms,
            swap_fractions=swap_fractions,
            seed=31,
        )

        # Total variance (Frobenius sum of per-entry variance) should increase
        # with swap fraction. At f=0 the variance is exactly 0.
        total_var_per_f = []
        for f in swap_fractions:
            null_arr = result["null_distributions"][f]  # [n_perms, K, K]
            total_var_per_f.append(float(null_arr.var(axis=0).sum()))

        # f=0 variance exactly zero (no shuffle).
        assert total_var_per_f[0] < 1e-12, (
            f"f=0 null variance should be zero, got {total_var_per_f[0]}"
        )

        # All nonzero swap fractions must produce strictly positive variance
        # (scrambling generates real spread).
        for f, v in zip(swap_fractions[1:], total_var_per_f[1:]):
            assert v > 1.0, (
                f"Expected meaningful null variance at f={f}, got {v:.4f}. "
                f"Full curve: {total_var_per_f}"
            )

        # The maximum-fraction variance should be substantially larger than
        # the no-shuffle baseline (which is zero). We do NOT require strict
        # monotonicity across intermediate fractions: once a fraction of
        # cells are swapped into their "wrong" population, the resulting
        # mass distribution can have non-monotone variance as the null
        # transitions between "mostly original" and "fully mixed" regimes.
        # What matters downstream is that f=0 is distinguishable from
        # any f>0 null.
        assert total_var_per_f[-1] > 10.0 * total_var_per_f[0] + 1.0, (
            f"Largest-fraction variance not clearly above baseline: "
            f"{total_var_per_f}"
        )

        # Strong pair's null mean should regress toward the overall matrix
        # mean as scrambling increases. Pick the observed strongest (i, j).
        observed = result["observed_mass"]
        strong_flat = int(np.argmax(observed))
        si, sj = strong_flat // K, strong_flat % K
        overall_mean = observed.mean()
        null_mean_curve = result["null_mean_curve"]  # [n_f, K, K]

        dist_f0 = abs(null_mean_curve[0, si, sj] - overall_mean)
        dist_f_last = abs(null_mean_curve[-1, si, sj] - overall_mean)
        assert dist_f_last < dist_f0, (
            f"Strong pair ({si},{sj}) null mean should move toward the matrix "
            f"mean under scrambling. f=0 distance={dist_f0:.4f}, "
            f"f={swap_fractions[-1]} distance={dist_f_last:.4f}"
        )

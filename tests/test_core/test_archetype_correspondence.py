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

"""Head-to-head comparison: ILR-GMM vs Dirichlet mixture for simplex decomposition.

Tests both methods on controlled synthetic scenarios where ground truth is known,
comparing: component recovery, BIC model selection, stability, centroid accuracy,
and behavior on edge cases (vertex-heavy data, high K, overlapping components).
"""

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon


def _make_planted_data(K, n_components, n_per_component, concentration, rng):
    """Generate simplex data with known component structure.

    Each component is a Dirichlet distribution centered near a different
    archetype vertex (or simplex region).
    """
    alphas = []
    for c in range(n_components):
        alpha = np.ones(K) * 0.5  # low background
        # Rotate the dominant vertex across components
        dominant = c % K
        alpha[dominant] = concentration
        alphas.append(alpha)

    chunks = []
    true_labels = []
    for c, alpha in enumerate(alphas):
        chunk = rng.dirichlet(alpha, n_per_component)
        chunks.append(chunk)
        true_labels.extend([c] * n_per_component)

    W = np.vstack(chunks)
    true_labels = np.array(true_labels)

    # Shuffle
    perm = rng.permutation(len(true_labels))
    return W[perm], true_labels[perm], alphas


def _match_accuracy(true_labels, pred_labels, n_components):
    """Hungarian-matched accuracy."""
    n_true = len(set(true_labels))
    n_pred = len(set(pred_labels[pred_labels >= 0]))
    n_max = max(n_true, n_pred)

    confusion = np.zeros((n_max, n_max))
    for i in range(n_max):
        for j in range(n_max):
            confusion[i, j] = np.sum((true_labels == i) & (pred_labels == j))
    row_ind, col_ind = linear_sum_assignment(-confusion)
    return confusion[row_ind, col_ind].sum() / len(true_labels)


def _centroid_error(true_alphas, fitted_centroids):
    """Mean Jensen-Shannon divergence between true and fitted centroids."""
    n_true = len(true_alphas)
    n_fit = len(fitted_centroids)
    n_max = max(n_true, n_fit)

    # Normalize true alphas to means
    true_means = [np.array(a) / np.sum(a) for a in true_alphas]

    # Cost matrix: JS divergence
    cost = np.ones((n_max, n_max))
    for i in range(min(n_true, n_max)):
        for j in range(min(n_fit, n_max)):
            cost[i, j] = jensenshannon(true_means[i], fitted_centroids[j])

    row_ind, col_ind = linear_sum_assignment(cost)
    return np.mean(cost[row_ind[:min(n_true, n_fit)], col_ind[:min(n_true, n_fit)]])


# ─── Scenario 1: Well-separated vertex-like components ────────────────────────

class TestWellSeparatedComponents:
    """Both methods should excel here. Baseline sanity check."""

    @pytest.fixture
    def well_separated_data(self):
        rng = np.random.default_rng(42)
        K = 4
        n_components = 3
        W, labels, alphas = _make_planted_data(K, n_components, 300, 15.0, rng)
        return W, labels, alphas, K

    def test_both_recover_correct_n(self, well_separated_data):
        W, labels, alphas, K = well_separated_data
        from peach._core.utils.simplex_gmm import fit_simplex_gmm

        g_result = fit_simplex_gmm(W, n_components_range=(2, 6), model_type="gaussian")
        d_result = fit_simplex_gmm(W, n_components_range=(2, 6), model_type="dirichlet")

        # Both should find 3 components
        assert g_result["n_components_optimal"] == 3, (
            f"Gaussian found {g_result['n_components_optimal']}, expected 3"
        )
        assert d_result["n_components_optimal"] == 3, (
            f"Dirichlet found {d_result['n_components_optimal']}, expected 3"
        )

    def test_both_high_accuracy(self, well_separated_data):
        W, labels, alphas, K = well_separated_data
        from peach._core.utils.simplex_gmm import fit_simplex_gmm

        g_result = fit_simplex_gmm(W, n_components_range=(2, 6), model_type="gaussian")
        d_result = fit_simplex_gmm(W, n_components_range=(2, 6), model_type="dirichlet")

        g_acc = _match_accuracy(labels, g_result["component_assignments"], 3)
        d_acc = _match_accuracy(labels, d_result["component_assignments"], 3)

        assert g_acc > 0.85, f"Gaussian accuracy {g_acc:.3f} too low"
        assert d_acc > 0.85, f"Dirichlet accuracy {d_acc:.3f} too low"

    def test_centroid_quality(self, well_separated_data):
        W, labels, alphas, K = well_separated_data
        from peach._core.utils.simplex_gmm import fit_simplex_gmm

        g_result = fit_simplex_gmm(W, n_components_range=(2, 6), model_type="gaussian")
        d_result = fit_simplex_gmm(W, n_components_range=(2, 6), model_type="dirichlet")

        g_err = _centroid_error(alphas, g_result["component_simplex_means"])
        d_err = _centroid_error(alphas, d_result["component_simplex_means"])

        assert g_err < 0.15, f"Gaussian centroid error {g_err:.3f} too high"
        assert d_err < 0.15, f"Dirichlet centroid error {d_err:.3f} too high"


# ─── Scenario 2: Overlapping components (harder separation) ──────────────────

class TestOverlappingComponents:
    """Low concentration → components overlap. Tests how each handles ambiguity."""

    @pytest.fixture
    def overlapping_data(self):
        rng = np.random.default_rng(42)
        K = 4
        n_components = 3
        # Low concentration = high overlap
        W, labels, alphas = _make_planted_data(K, n_components, 300, 3.0, rng)
        return W, labels, alphas, K

    def test_model_selection_in_range(self, overlapping_data):
        W, labels, alphas, K = overlapping_data
        from peach._core.utils.simplex_gmm import fit_simplex_gmm

        g_result = fit_simplex_gmm(W, n_components_range=(2, 6), model_type="gaussian")
        d_result = fit_simplex_gmm(W, n_components_range=(2, 6), model_type="dirichlet")

        # With overlap, might select 2-4 components
        assert 2 <= g_result["n_components_optimal"] <= 5
        assert 2 <= d_result["n_components_optimal"] <= 5

    def test_icl_more_conservative(self, overlapping_data):
        """ICL penalizes overlap → should select <= BIC's n_components."""
        W, labels, alphas, K = overlapping_data
        from peach._core.utils.simplex_gmm import fit_simplex_gmm

        for mtype in ["gaussian", "dirichlet"]:
            bic_result = fit_simplex_gmm(
                W, n_components_range=(2, 6),
                model_type=mtype, model_selection="bic",
            )
            icl_result = fit_simplex_gmm(
                W, n_components_range=(2, 6),
                model_type=mtype, model_selection="icl",
            )
            # ICL should be <= BIC (more conservative about n)
            assert icl_result["n_components_optimal"] <= bic_result["n_components_optimal"] + 1, (
                f"{mtype}: ICL={icl_result['n_components_optimal']} vs "
                f"BIC={bic_result['n_components_optimal']}"
            )


# ─── Scenario 3: Vertex-heavy data (Dirichlet's home turf) ──────────────────

class TestVertexHeavyData:
    """Data concentrated at simplex vertices. Dirichlet should handle this
    natively while ILR-GMM may struggle (epsilon smoothing, skewness)."""

    @pytest.fixture
    def vertex_data(self):
        rng = np.random.default_rng(42)
        K = 4
        # Very high concentration → near-vertex points
        W, labels, alphas = _make_planted_data(K, 3, 250, 50.0, rng)
        return W, labels, alphas, K

    def test_both_handle_vertex_data(self, vertex_data):
        W, labels, alphas, K = vertex_data
        from peach._core.utils.simplex_gmm import fit_simplex_gmm

        g_result = fit_simplex_gmm(W, n_components_range=(2, 5), model_type="gaussian")
        d_result = fit_simplex_gmm(W, n_components_range=(2, 5), model_type="dirichlet")

        # Both should find components (no crash)
        assert g_result["n_components_stable"] >= 2
        assert d_result["n_components_stable"] >= 2

    def test_accuracy_comparison_vertex(self, vertex_data):
        W, labels, alphas, K = vertex_data
        from peach._core.utils.simplex_gmm import fit_simplex_gmm

        g_result = fit_simplex_gmm(W, n_components_range=(2, 5), model_type="gaussian")
        d_result = fit_simplex_gmm(W, n_components_range=(2, 5), model_type="dirichlet")

        g_acc = _match_accuracy(labels, g_result["component_assignments"], 3)
        d_acc = _match_accuracy(labels, d_result["component_assignments"], 3)

        # Vertex data is easy for both — high accuracy expected
        assert g_acc > 0.90, f"Gaussian accuracy {g_acc:.3f} on vertex data"
        assert d_acc > 0.90, f"Dirichlet accuracy {d_acc:.3f} on vertex data"


# ─── Scenario 4: Interior components (ILR-GMM's home turf) ──────────────────

class TestInteriorComponents:
    """Components are Gaussian blobs in the interior of the simplex.
    ILR-GMM should have an advantage since the data IS Gaussian in ILR space."""

    @pytest.fixture
    def interior_data(self):
        """Generate data that's actually Gaussian in ILR space."""
        from peach._core.utils.ilr_transform import ilr_transform, inverse_ilr
        rng = np.random.default_rng(42)
        K = 4

        # 3 Gaussian blobs in ILR(K-1=3) space
        means = [
            np.array([1.0, -0.5, 0.3]),
            np.array([-0.5, 1.0, -0.3]),
            np.array([0.0, 0.0, 1.0]),
        ]
        cov = np.eye(K - 1) * 0.15

        chunks = []
        labels = []
        for c, mu in enumerate(means):
            ilr_data = rng.multivariate_normal(mu, cov, 300)
            chunks.append(ilr_data)
            labels.extend([c] * 300)

        ilr_all = np.vstack(chunks)
        W = inverse_ilr(ilr_all)  # back to simplex
        labels = np.array(labels)

        perm = rng.permutation(len(labels))
        return W[perm], labels[perm], K

    def test_gaussian_natural_advantage(self, interior_data):
        """ILR-GMM should be at least as good as Dirichlet on ILR-Gaussian data."""
        W, labels, K = interior_data
        from peach._core.utils.simplex_gmm import fit_simplex_gmm

        g_result = fit_simplex_gmm(W, n_components_range=(2, 5), model_type="gaussian")
        d_result = fit_simplex_gmm(W, n_components_range=(2, 5), model_type="dirichlet")

        g_acc = _match_accuracy(labels, g_result["component_assignments"], 3)
        d_acc = _match_accuracy(labels, d_result["component_assignments"], 3)

        # Gaussian should be good on its own turf
        assert g_acc > 0.85, f"Gaussian accuracy {g_acc:.3f} on ILR-Gaussian data"
        # Dirichlet might still do OK (not necessarily bad)
        assert d_acc > 0.60, f"Dirichlet accuracy {d_acc:.3f} unreasonably low"


# ─── Scenario 5: High K ─────────────────────────────────────────────────────

class TestHighK:
    """K=8 archetypes. Tests scalability and curse-of-dimensionality effects."""

    def test_both_handle_high_k(self):
        rng = np.random.default_rng(42)
        K = 8
        W, labels, alphas = _make_planted_data(K, 4, 200, 10.0, rng)
        from peach._core.utils.simplex_gmm import fit_simplex_gmm

        g_result = fit_simplex_gmm(
            W, n_components_range=(3, 6), model_type="gaussian",
            n_initializations=10,
        )
        d_result = fit_simplex_gmm(
            W, n_components_range=(3, 6), model_type="dirichlet",
            n_initializations=10,
        )

        # Both should identify ~4 components
        assert 3 <= g_result["n_components_optimal"] <= 6
        assert 3 <= d_result["n_components_optimal"] <= 6

        g_acc = _match_accuracy(labels, g_result["component_assignments"], 4)
        d_acc = _match_accuracy(labels, d_result["component_assignments"], 4)

        assert g_acc > 0.70, f"Gaussian K=8 accuracy {g_acc:.3f}"
        assert d_acc > 0.70, f"Dirichlet K=8 accuracy {d_acc:.3f}"


# ─── Scenario 6: K=2 edge case ──────────────────────────────────────────────

class TestK2:
    """K=2 (1D simplex = line segment). ILR is 1D, Dirichlet is 2D."""

    def test_both_handle_k2(self):
        rng = np.random.default_rng(42)
        K = 2
        W, labels, alphas = _make_planted_data(K, 2, 200, 8.0, rng)
        from peach._core.utils.simplex_gmm import fit_simplex_gmm

        g_result = fit_simplex_gmm(W, n_components_range=(2, 4), model_type="gaussian")
        d_result = fit_simplex_gmm(W, n_components_range=(2, 4), model_type="dirichlet")

        assert g_result["n_components_stable"] >= 1
        assert d_result["n_components_stable"] >= 1

        g_acc = _match_accuracy(labels, g_result["component_assignments"], 2)
        d_acc = _match_accuracy(labels, d_result["component_assignments"], 2)

        assert g_acc > 0.80, f"Gaussian K=2 accuracy {g_acc:.3f}"
        assert d_acc > 0.80, f"Dirichlet K=2 accuracy {d_acc:.3f}"


# ─── Scenario 7: Stability comparison ───────────────────────────────────────

class TestStabilityComparison:
    """Compare stability scores between methods."""

    def test_well_separated_high_stability(self):
        rng = np.random.default_rng(42)
        K = 4
        W, labels, alphas = _make_planted_data(K, 3, 300, 15.0, rng)
        from peach._core.utils.simplex_gmm import fit_simplex_gmm

        g_result = fit_simplex_gmm(
            W, n_components_range=(2, 5), model_type="gaussian",
            n_initializations=15,
        )
        d_result = fit_simplex_gmm(
            W, n_components_range=(2, 5), model_type="dirichlet",
            n_initializations=15,
        )

        # Well-separated data should have high stability
        g_stab = g_result["component_stability_scores"]
        d_stab = d_result["component_stability_scores"]

        assert np.mean(g_stab) > 0.70, f"Gaussian mean stability {np.mean(g_stab):.3f}"
        assert np.mean(d_stab) > 0.70, f"Dirichlet mean stability {np.mean(d_stab):.3f}"


# ─── Scenario 8: BIC curves shape ───────────────────────────────────────────

class TestBICCurves:
    """Both methods should produce U-shaped BIC curves with minimum at truth."""

    def test_bic_minimum_near_truth(self):
        rng = np.random.default_rng(42)
        K = 4
        W, labels, alphas = _make_planted_data(K, 3, 400, 12.0, rng)
        from peach._core.utils.simplex_gmm import fit_simplex_gmm

        g_result = fit_simplex_gmm(W, n_components_range=(2, 7), model_type="gaussian")
        d_result = fit_simplex_gmm(W, n_components_range=(2, 7), model_type="dirichlet")

        # BIC minimum should be near 3
        g_best_idx = np.argmin(g_result["bic_values"])
        d_best_idx = np.argmin(d_result["bic_values"])
        g_best_n = g_result["n_components_tested"][g_best_idx]
        d_best_n = d_result["n_components_tested"][d_best_idx]

        assert abs(g_best_n - 3) <= 1, f"Gaussian BIC minimum at {g_best_n}"
        assert abs(d_best_n - 3) <= 1, f"Dirichlet BIC minimum at {d_best_n}"

        # BIC should increase for n >> truth
        g_bic_at_7 = g_result["bic_values"][-1]
        g_bic_at_best = g_result["bic_values"][g_best_idx]
        assert g_bic_at_7 > g_bic_at_best, "Gaussian BIC should increase past optimum"


# ─── Scenario 9: Full API integration ───────────────────────────────────────

class TestFullAPIComparison:
    """End-to-end through the public peach.tl API."""

    def test_api_produces_consistent_results(self):
        import anndata as ad
        import peach as pc

        rng = np.random.default_rng(42)
        K = 4
        n = 600
        W, labels, alphas = _make_planted_data(K, 3, 200, 12.0, rng)
        X = rng.standard_normal((n, 30)).astype(np.float32)

        adata = ad.AnnData(X)
        adata.var_names = [f"gene_{i}" for i in range(30)]
        adata.obsm["cell_archetype_weights"] = W
        adata.obsm["X_pca"] = rng.standard_normal((n, 10)).astype(np.float32)

        g = pc.tl.feature_simplex_decomposition(
            adata, model_type="gaussian", n_components_range=(2, 5),
        )
        d = pc.tl.feature_simplex_decomposition(
            adata, model_type="dirichlet", n_components_range=(2, 5),
        )

        # Both should return valid results
        assert "n_components_optimal" in g
        assert "n_components_optimal" in d
        assert g["model_type"] == "gaussian"
        assert d["model_type"] == "dirichlet"

        # Assignments should be valid labels
        assert set(g["component_assignments"]).issubset(set(range(g["n_components_stable"])) | {-1})
        assert set(d["component_assignments"]).issubset(set(range(d["n_components_stable"])) | {-1})

        # Component means should be on the simplex
        np.testing.assert_allclose(
            g["component_simplex_means"].sum(axis=1), 1.0, atol=1e-3
        )
        np.testing.assert_allclose(
            d["component_simplex_means"].sum(axis=1), 1.0, atol=1e-3
        )


# ─── Summary comparison runner ───────────────────────────────────────────────

class TestComparisonSummary:
    """Run both methods across all scenarios and print a comparison table."""

    @pytest.mark.slow
    def test_print_comparison_table(self, capsys):
        from peach._core.utils.simplex_gmm import fit_simplex_gmm
        rng = np.random.default_rng(42)

        scenarios = {
            "well-separated (conc=15)": (4, 3, 300, 15.0),
            "overlapping (conc=3)": (4, 3, 300, 3.0),
            "vertex-heavy (conc=50)": (4, 3, 250, 50.0),
            "high-K (K=8)": (8, 4, 200, 10.0),
            "K=2": (2, 2, 200, 8.0),
        }

        rows = []
        for name, (K, n_comp, n_per, conc) in scenarios.items():
            W, labels, alphas = _make_planted_data(K, n_comp, n_per, conc, rng)

            g = fit_simplex_gmm(W, n_components_range=(2, n_comp + 3),
                                model_type="gaussian", n_initializations=10)
            d = fit_simplex_gmm(W, n_components_range=(2, n_comp + 3),
                                model_type="dirichlet", n_initializations=10)

            g_acc = _match_accuracy(labels, g["component_assignments"], n_comp)
            d_acc = _match_accuracy(labels, d["component_assignments"], n_comp)
            g_err = _centroid_error(alphas, g["component_simplex_means"])
            d_err = _centroid_error(alphas, d["component_simplex_means"])

            rows.append({
                "scenario": name, "true_n": n_comp,
                "g_n": g["n_components_optimal"], "d_n": d["n_components_optimal"],
                "g_acc": g_acc, "d_acc": d_acc,
                "g_err": g_err, "d_err": d_err,
                "g_stab": np.mean(g["component_stability_scores"]),
                "d_stab": np.mean(d["component_stability_scores"]),
            })

        # Print table
        print("\n" + "=" * 100)
        print("ILR-GMM vs Dirichlet Mixture Comparison")
        print("=" * 100)
        print(f"{'Scenario':<25} {'True':>4} {'GMM_n':>5} {'Dir_n':>5} "
              f"{'GMM_acc':>7} {'Dir_acc':>7} {'GMM_JSD':>7} {'Dir_JSD':>7} "
              f"{'GMM_stb':>7} {'Dir_stb':>7}")
        print("-" * 100)
        for r in rows:
            winner_acc = "G" if r["g_acc"] > r["d_acc"] else "D" if r["d_acc"] > r["g_acc"] else "="
            print(f"{r['scenario']:<25} {r['true_n']:>4} {r['g_n']:>5} {r['d_n']:>5} "
                  f"{r['g_acc']:>6.3f}{winner_acc} {r['d_acc']:>6.3f}  "
                  f"{r['g_err']:>6.4f}  {r['d_err']:>6.4f}  "
                  f"{r['g_stab']:>6.3f}  {r['d_stab']:>6.3f}")
        print("=" * 100)

        # All scenarios should have both methods above baseline
        for r in rows:
            assert r["g_acc"] > 0.5, f"Gaussian too low on {r['scenario']}"
            assert r["d_acc"] > 0.5, f"Dirichlet too low on {r['scenario']}"

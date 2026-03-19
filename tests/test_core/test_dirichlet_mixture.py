"""Tests for DirichletMixture: convergence, recovery, edge cases."""

import numpy as np
import pytest
from peach._core.utils.dirichlet_mixture import DirichletMixture


class TestDirichletConvergence:
    def test_converges_on_well_separated_components(self):
        """Two clearly distinct Dirichlet components should converge."""
        rng = np.random.default_rng(42)
        K = 3
        W1 = rng.dirichlet([10.0, 0.5, 0.5], 200)
        W2 = rng.dirichlet([0.5, 0.5, 10.0], 200)
        W = np.vstack([W1, W2])
        model = DirichletMixture(n_components=2, max_iter=200, random_state=42)
        model.fit(W)
        assert model.converged_, "Should converge on well-separated data"

    def test_bic_selects_correct_n_components(self):
        """BIC should prefer 2 components for 2-component data."""
        rng = np.random.default_rng(42)
        W1 = rng.dirichlet([8, 1, 1], 300)
        W2 = rng.dirichlet([1, 1, 8], 300)
        W = np.vstack([W1, W2])
        bics = {}
        for nc in [1, 2, 3, 4]:
            model = DirichletMixture(n_components=nc, max_iter=200, random_state=42)
            model.fit(W)
            bics[nc] = model.bic(W)
        assert bics[2] < bics[1], "BIC should prefer 2 over 1"
        assert bics[2] < bics[4], "BIC should prefer 2 over 4"


class TestDirichletRecovery:
    def test_recovers_alpha_direction(self):
        """Fitted alpha direction should correlate with ground truth."""
        rng = np.random.default_rng(42)
        alpha_true = np.array([10.0, 2.0, 0.5])
        W = rng.dirichlet(alpha_true, 500)
        model = DirichletMixture(n_components=1, max_iter=200, random_state=42)
        model.fit(W)
        fitted_mean = model.means_[0]
        true_mean = alpha_true / alpha_true.sum()
        correlation = np.corrcoef(fitted_mean, true_mean)[0, 1]
        assert correlation > 0.9, f"Direction correlation = {correlation}"

    def test_predict_separates_components(self):
        """Predictions should separate well-separated components."""
        rng = np.random.default_rng(42)
        W1 = rng.dirichlet([20, 1, 1], 200)
        W2 = rng.dirichlet([1, 1, 20], 200)
        W = np.vstack([W1, W2])
        true_labels = np.array([0] * 200 + [1] * 200)
        model = DirichletMixture(n_components=2, max_iter=200, random_state=42)
        model.fit(W)
        pred = model.predict(W)
        # Match labels (may be permuted)
        from scipy.optimize import linear_sum_assignment

        confusion = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                confusion[i, j] = np.sum((true_labels == i) & (pred == j))
        row_ind, col_ind = linear_sum_assignment(-confusion)
        accuracy = confusion[row_ind, col_ind].sum() / len(true_labels)
        assert accuracy > 0.85, f"Separation accuracy = {accuracy}"


class TestDirichletEdgeCases:
    def test_single_component(self):
        rng = np.random.default_rng(42)
        W = rng.dirichlet([2, 2, 2], 100)
        model = DirichletMixture(n_components=1, max_iter=100, random_state=42)
        model.fit(W)
        assert model.alphas_.shape == (1, 3)

    def test_vertex_data_warns(self):
        W = np.zeros((100, 3))
        W[:80, 0] = 1.0
        W[80:, 1] = 1.0
        model = DirichletMixture(n_components=2, max_iter=50, random_state=42)
        with pytest.warns(UserWarning, match="vertices"):
            model.fit(W)

    def test_k_equals_2(self):
        rng = np.random.default_rng(42)
        W = rng.dirichlet([5, 2], 200)
        model = DirichletMixture(n_components=2, max_iter=100, random_state=42)
        model.fit(W)
        assert model.alphas_.shape == (2, 2)

    def test_predict_returns_valid_labels(self):
        rng = np.random.default_rng(42)
        W = rng.dirichlet([3, 3, 3], 100)
        model = DirichletMixture(n_components=3, max_iter=100, random_state=42)
        model.fit(W)
        labels = model.predict(W)
        assert set(labels).issubset({0, 1, 2})
        assert len(labels) == 100

    def test_predict_proba_sums_to_one(self):
        rng = np.random.default_rng(42)
        W = rng.dirichlet([3, 3, 3], 100)
        model = DirichletMixture(n_components=2, max_iter=100, random_state=42)
        model.fit(W)
        proba = model.predict_proba(W)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_n_init_picks_best(self):
        """Multiple inits should pick the best log-likelihood."""
        rng = np.random.default_rng(42)
        W = rng.dirichlet([5, 1, 1], 200)
        model_1 = DirichletMixture(
            n_components=2, max_iter=100, n_init=1, random_state=42
        )
        model_1.fit(W)
        model_5 = DirichletMixture(
            n_components=2, max_iter=100, n_init=5, random_state=42
        )
        model_5.fit(W)
        assert model_5.log_likelihood_ >= model_1.log_likelihood_ - 1e-6

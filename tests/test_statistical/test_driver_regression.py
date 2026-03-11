"""Tests for archetype driver regression (flipped: features predict weights)."""

import numpy as np
import pytest
from anndata import AnnData


@pytest.fixture
def driver_adata():
    """AnnData with geneset scores driving known archetype structure."""
    rng = np.random.default_rng(42)
    K = 3
    n = 500
    n_genesets = 10

    # Create geneset scores
    geneset_scores = rng.standard_normal((n, n_genesets))

    # Geneset 0 drives archetype 0 specialization
    # Make weights dependent on geneset scores
    logits = np.zeros((n, K))
    logits[:, 0] = geneset_scores[:, 0] * 3  # geneset 0 drives archetype 0
    logits[:, 1] = geneset_scores[:, 1] * 2  # geneset 1 drives archetype 1
    # Softmax to get weights
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    weights = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    # Create dummy expression
    X = rng.standard_normal((n, 50))

    adata = AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(50)]
    adata.obsm["cell_archetype_weights"] = weights
    adata.obsm["pathway_scores"] = geneset_scores
    return adata


class TestArchetypeDriverRegression:
    def test_basic_run(self, driver_adata):
        """Runs without error and stores result in adata.uns."""
        import peach as pc

        result = pc.tl.archetype_driver_regression(
            driver_adata, feature_matrix="pathway_scores", n_bootstrap=0
        )
        assert result is not None
        assert "peach_driver_regression" in driver_adata.uns

    def test_ilr_regressions_run(self, driver_adata):
        """K-1 regressions in ILR space with correct shapes."""
        import peach as pc

        result = pc.tl.archetype_driver_regression(
            driver_adata, feature_matrix="pathway_scores", n_bootstrap=0
        )
        K = 3
        assert np.asarray(result["main_coefficients_ilr"]).shape == (K - 1, 10)
        assert np.asarray(result["r_squared"]).shape == (K - 1,)

    def test_back_transform_to_simplex(self, driver_adata):
        """Coefficients map back to per-archetype with shape [K, n_features]."""
        import peach as pc

        result = pc.tl.archetype_driver_regression(
            driver_adata, feature_matrix="pathway_scores", n_bootstrap=0
        )
        K = 3
        assert np.asarray(result["main_coefficients"]).shape == (K, 10)

    def test_max_interaction_features_guard(self, driver_adata):
        """Error raised when n_features > max_interaction_features at degree=2."""
        import peach as pc

        with pytest.raises(ValueError, match="max_interaction_features"):
            pc.tl.archetype_driver_regression(
                driver_adata,
                feature_matrix=driver_adata.X,  # 50 features > threshold
                feature_names=[f"g_{i}" for i in range(50)],
                max_degree=2,
                max_interaction_features=10,
                n_bootstrap=0,
            )

    def test_intercept_included(self, driver_adata):
        """Intercept present (features are not compositional)."""
        import peach as pc

        result = pc.tl.archetype_driver_regression(
            driver_adata, feature_matrix="pathway_scores", n_bootstrap=0
        )
        assert result.get("intercepts") is not None
        assert np.asarray(result["intercepts"]).shape == (2,)  # K-1

    def test_degree1_no_interactions(self, driver_adata):
        """max_degree=1 produces no interaction terms."""
        import peach as pc

        result = pc.tl.archetype_driver_regression(
            driver_adata,
            feature_matrix="pathway_scores",
            max_degree=1,
            n_bootstrap=0,
        )
        assert result.get("interaction_coefficients_ilr") is None
        assert result.get("interaction_coefficients") is None
        assert result.get("interaction_pvalues") is None

    def test_degree2_has_interactions(self, driver_adata):
        """max_degree=2 produces interaction terms with correct shape."""
        import peach as pc

        result = pc.tl.archetype_driver_regression(
            driver_adata, feature_matrix="pathway_scores", max_degree=2, n_bootstrap=0
        )
        K = 3
        n_features = 10
        n_interactions = n_features * (n_features - 1) // 2  # 45
        assert np.asarray(result["interaction_coefficients_ilr"]).shape == (K - 1, n_interactions)
        assert np.asarray(result["interaction_coefficients"]).shape == (K, n_interactions)
        assert np.asarray(result["interaction_pvalues"]).shape == (K - 1, n_interactions)

    def test_r_squared_positive(self, driver_adata):
        """R-squared should be positive for data with real signal."""
        import peach as pc

        result = pc.tl.archetype_driver_regression(
            driver_adata, feature_matrix="pathway_scores", n_bootstrap=0
        )
        # The signal is strong (genesets 0 and 1 drive archetypes 0 and 1)
        assert np.all(np.asarray(result["r_squared"]) > 0.0)

    def test_driving_features_have_largest_coefficients(self, driver_adata):
        """Geneset 0 should have largest effect on archetype 0."""
        import peach as pc

        result = pc.tl.archetype_driver_regression(
            driver_adata,
            feature_matrix="pathway_scores",
            max_degree=1,
            n_bootstrap=0,
        )
        # main_coefficients is [K, n_features]
        # Archetype 0 should have largest absolute coefficient for feature 0
        arch0_coefs = np.abs(np.asarray(result["main_coefficients"])[0, :])
        assert np.argmax(arch0_coefs) == 0, (
            f"Expected feature 0 to have largest effect on archetype 0, "
            f"but argmax was {np.argmax(arch0_coefs)}"
        )

    def test_pvalues_shape_and_range(self, driver_adata):
        """P-values should be in [0, 1] with correct shape."""
        import peach as pc

        result = pc.tl.archetype_driver_regression(
            driver_adata, feature_matrix="pathway_scores", n_bootstrap=0
        )
        assert np.asarray(result["main_pvalues"]).shape == (2, 10)  # [K-1, n_features]
        assert np.all(np.asarray(result["main_pvalues"]) >= 0.0)
        assert np.all(np.asarray(result["main_pvalues"]) <= 1.0)

    def test_copy_does_not_modify_original(self, driver_adata):
        """copy=True should not modify the original adata."""
        import peach as pc

        result = pc.tl.archetype_driver_regression(
            driver_adata,
            feature_matrix="pathway_scores",
            n_bootstrap=0,
            copy=True,
        )
        assert "peach_driver_regression" not in driver_adata.uns

    def test_bootstrap_cis(self, driver_adata):
        """Bootstrap CIs should have correct shape when enabled."""
        import peach as pc

        result = pc.tl.archetype_driver_regression(
            driver_adata,
            feature_matrix="pathway_scores",
            max_degree=1,
            n_bootstrap=50,  # small for speed
        )
        K = 3
        assert result.get("main_ci_lower") is not None
        assert result.get("main_ci_upper") is not None
        assert np.asarray(result["main_ci_lower"]).shape == (K, 10)
        assert np.asarray(result["main_ci_upper"]).shape == (K, 10)
        # Lower should be <= upper
        assert np.all(np.asarray(result["main_ci_lower"]) <= np.asarray(result["main_ci_upper"]))

    def test_returns_serialized_dict(self, driver_adata):
        """Return value should be a serialized dict with no None values."""
        import peach as pc

        result = pc.tl.archetype_driver_regression(
            driver_adata,
            feature_matrix="pathway_scores",
            max_degree=1,
            n_bootstrap=0,
        )
        assert isinstance(result, dict)
        assert "main_coefficients" in result
        assert "r_squared" in result
        # None fields should be excluded
        assert "interaction_coefficients_ilr" not in result
        assert "main_ci_lower" not in result

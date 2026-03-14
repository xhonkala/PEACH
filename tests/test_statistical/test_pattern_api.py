import numpy as np
import pytest
from anndata import AnnData


@pytest.fixture
def classified_adata():
    """AnnData with regression results ready for classification."""
    rng = np.random.default_rng(42)
    K = 3
    n = 500
    n_genes = 20

    weights = rng.dirichlet([1] * K, size=n)
    true_beta = rng.standard_normal((n_genes, K)) * 5
    true_beta[0] = [10.0, 0.0, 0.0]  # exclusive
    true_beta[1] = [3.0, 3.0, 3.0]  # flat (equal across archetypes)
    true_beta[2] = [8.0, 4.0, 1.0]  # structured

    noise = rng.normal(0, 0.3, size=(n, n_genes))
    X = weights @ true_beta.T + noise

    adata = AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obsm["cell_archetype_weights"] = weights
    return adata


class TestClassifyFeaturePatterns:
    def test_runs_after_regression(self, classified_adata):
        import peach as pc

        pc.tl.feature_simplex_regression(classified_adata, n_bootstrap=0)
        result = pc.tl.classify_feature_patterns(classified_adata)
        assert "peach_feature_patterns" in classified_adata.uns
        assert len(result["classifications"]) == 20

    def test_gene0_classified_exclusive(self, classified_adata):
        import peach as pc

        pc.tl.feature_simplex_regression(classified_adata, n_bootstrap=0)
        result = pc.tl.classify_feature_patterns(classified_adata)
        assert result["classifications"][0]["pattern"] == "archetype-exclusive"

    def test_gene1_classified_flat(self, classified_adata):
        import peach as pc

        pc.tl.feature_simplex_regression(classified_adata, n_bootstrap=0)
        result = pc.tl.classify_feature_patterns(classified_adata)
        assert result["classifications"][1]["pattern"] == "flat"

    def test_pattern_counts(self, classified_adata):
        import peach as pc

        pc.tl.feature_simplex_regression(classified_adata, n_bootstrap=0)
        result = pc.tl.classify_feature_patterns(classified_adata)
        assert sum(result["pattern_counts"].values()) == 20

    def test_valid_pattern_categories(self, classified_adata):
        import peach as pc

        pc.tl.feature_simplex_regression(classified_adata, n_bootstrap=0)
        result = pc.tl.classify_feature_patterns(classified_adata)
        valid = {"flat", "archetype-exclusive", "interaction", "structured"}
        for c in result["classifications"]:
            assert c["pattern"] in valid, f"Unexpected pattern: {c['pattern']}"

    def test_r2_in_classifications(self, classified_adata):
        import peach as pc

        pc.tl.feature_simplex_regression(classified_adata, n_bootstrap=0)
        result = pc.tl.classify_feature_patterns(classified_adata)
        for c in result["classifications"]:
            assert "r2" in c

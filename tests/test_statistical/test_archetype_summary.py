import numpy as np
import pytest
from anndata import AnnData


@pytest.fixture
def summary_adata():
    """AnnData with regression + classification results."""
    rng = np.random.default_rng(42)
    K = 3
    n = 500
    n_genes = 20

    weights = rng.dirichlet([1] * K, size=n)
    true_beta = rng.standard_normal((n_genes, K)) * 5
    true_beta[0] = [10.0, 0.0, 0.0]
    true_beta[1] = [3.0, 3.0, 3.0]

    noise = rng.normal(0, 0.3, size=(n, n_genes))
    X = weights @ true_beta.T + noise

    adata = AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obsm["cell_archetype_weights"] = weights

    # Run regression + classification
    import peach as pc

    pc.tl.feature_simplex_regression(adata, n_bootstrap=0)
    pc.tl.classify_feature_patterns(adata)

    return adata


class TestArchetypeSummary:
    def test_single_archetype(self, summary_adata):
        """Returns structured dict for one archetype."""
        import peach as pc

        result = pc.tl.archetype_summary(summary_adata, archetype_idx=0)
        assert isinstance(result, dict)
        assert result["archetype_idx"] == 0
        assert "top_enriched" in result
        assert "top_depleted" in result
        assert len(result["top_enriched"]) <= 20

    def test_all_archetypes(self, summary_adata):
        """Returns list of dicts for all K archetypes."""
        import peach as pc

        result = pc.tl.archetype_summary(summary_adata)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_includes_regression(self, summary_adata):
        """Top enriched/depleted features present."""
        import peach as pc

        result = pc.tl.archetype_summary(summary_adata, archetype_idx=0)
        assert len(result["top_enriched"]) > 0
        assert "feature" in result["top_enriched"][0]
        assert "coefficient" in result["top_enriched"][0]

    def test_graceful_without_optional(self, summary_adata):
        """Works with only regression (no GMM, no drivers)."""
        import peach as pc

        result = pc.tl.archetype_summary(
            summary_adata, include_drivers=False, include_gmm=False
        )
        assert isinstance(result, list)
        # No error even without optional results

    def test_invalid_archetype_raises(self, summary_adata):
        """Invalid archetype index raises ValueError."""
        import peach as pc

        with pytest.raises(ValueError, match="out of range"):
            pc.tl.archetype_summary(summary_adata, archetype_idx=10)

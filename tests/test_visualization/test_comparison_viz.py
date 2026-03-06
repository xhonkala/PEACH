import numpy as np
import pytest
from anndata import AnnData
import plotly.graph_objects as go


@pytest.fixture
def viz_adata():
    """AnnData with comparison results pre-computed."""
    rng = np.random.default_rng(42)
    K = 3
    n = 300
    n_genes = 20

    weights = rng.dirichlet([1] * K, size=n)
    true_beta = rng.standard_normal((n_genes, K)) * 5
    X = weights @ true_beta.T + rng.normal(0, 0.2, (n, n_genes))

    adata = AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obsm["cell_archetype_weights"] = weights
    adata.obsm["X_pca"] = rng.standard_normal((n, 10))

    import peach as pc
    pc.tl.feature_simplex_regression(adata, n_bootstrap=0)
    pc.tl.archetype_mmd(adata, n_permutations=10)
    pc.tl.archetype_feature_similarity(adata)
    pc.tl.archetype_contrasts(adata)
    return adata


class TestMMDHeatmap:
    def test_returns_figure(self, viz_adata):
        import peach as pc
        fig = pc.pl.mmd_heatmap(viz_adata, show=False)
        assert isinstance(fig, go.Figure)

    def test_has_heatmap_trace(self, viz_adata):
        import peach as pc
        fig = pc.pl.mmd_heatmap(viz_adata, show=False)
        assert any(isinstance(t, go.Heatmap) for t in fig.data)


class TestContrastVolcano:
    def test_returns_figure(self, viz_adata):
        import peach as pc
        fig = pc.pl.contrast_volcano(viz_adata, pair=(0, 1), show=False)
        assert isinstance(fig, go.Figure)

    def test_scatter_trace(self, viz_adata):
        import peach as pc
        fig = pc.pl.contrast_volcano(viz_adata, pair=(0, 1), show=False)
        assert any(isinstance(t, go.Scatter) for t in fig.data)


class TestFeatureSimilarityHeatmap:
    def test_returns_figure(self, viz_adata):
        import peach as pc
        fig = pc.pl.feature_similarity_heatmap(viz_adata, show=False)
        assert isinstance(fig, go.Figure)

    def test_has_heatmap_trace(self, viz_adata):
        import peach as pc
        fig = pc.pl.feature_similarity_heatmap(viz_adata, show=False)
        assert any(isinstance(t, go.Heatmap) for t in fig.data)

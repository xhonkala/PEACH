"""Tests for regression visualization functions."""

import numpy as np
import pytest
from anndata import AnnData
import plotly.graph_objects as go

from peach.pl.regression import (
    coefficient_heatmap,
    interaction_heatmap,
    r2_barplot,
    vertex_radar,
    regression_volcano,
    pattern_summary,
)


@pytest.fixture
def viz_adata():
    """AnnData with regression + classification results for visualization."""
    rng = np.random.default_rng(42)
    K = 3
    n = 200
    n_genes = 20

    weights = rng.dirichlet([1] * K, size=n)
    true_beta = rng.standard_normal((n_genes, K)) * 5
    noise = rng.normal(0, 0.3, size=(n, n_genes))
    X = weights @ true_beta.T + noise

    adata = AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obsm["cell_archetype_weights"] = weights

    import peach as pc

    pc.tl.feature_simplex_regression(adata, max_degree=2, n_bootstrap=0)
    pc.tl.classify_feature_patterns(adata)
    return adata


@pytest.fixture
def viz_adata_degree1():
    """AnnData with degree-1 only regression (no interactions)."""
    rng = np.random.default_rng(99)
    K = 3
    n = 100
    n_genes = 10

    weights = rng.dirichlet([1] * K, size=n)
    true_beta = rng.standard_normal((n_genes, K)) * 3
    X = weights @ true_beta.T + rng.normal(0, 0.5, size=(n, n_genes))

    adata = AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obsm["cell_archetype_weights"] = weights

    import peach as pc

    pc.tl.feature_simplex_regression(adata, max_degree=1, n_bootstrap=0)
    return adata


class TestCoefficientHeatmap:
    def test_returns_figure(self, viz_adata):
        fig = coefficient_heatmap(viz_adata, show=False)
        assert isinstance(fig, go.Figure)

    def test_has_heatmap_trace(self, viz_adata):
        fig = coefficient_heatmap(viz_adata, show=False)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Heatmap)

    def test_top_n_limits_features(self, viz_adata):
        fig = coefficient_heatmap(viz_adata, top_n=5, show=False)
        # y-axis should have 5 features
        assert len(fig.data[0].y) == 5

    def test_top_n_exceeds_features(self, viz_adata):
        # 20 genes but top_n=100 -- should show all 20
        fig = coefficient_heatmap(viz_adata, top_n=100, show=False)
        assert len(fig.data[0].y) == 20

    def test_save_path(self, viz_adata, tmp_path):
        save_path = str(tmp_path / "coef_heatmap.html")
        fig = coefficient_heatmap(viz_adata, save_path=save_path, show=False)
        assert isinstance(fig, go.Figure)
        assert (tmp_path / "coef_heatmap.html").exists()

    def test_missing_regression_raises(self):
        adata = AnnData(np.zeros((10, 5)))
        with pytest.raises(ValueError, match="No regression results"):
            coefficient_heatmap(adata, show=False)


class TestInteractionHeatmap:
    def test_returns_figure(self, viz_adata):
        fig = interaction_heatmap(viz_adata, show=False)
        assert isinstance(fig, go.Figure)

    def test_has_heatmap_trace(self, viz_adata):
        fig = interaction_heatmap(viz_adata, show=False)
        assert isinstance(fig.data[0], go.Heatmap)

    def test_top_n_limits_features(self, viz_adata):
        fig = interaction_heatmap(viz_adata, top_n=5, show=False)
        assert len(fig.data[0].y) == 5

    def test_no_interactions_raises(self, viz_adata_degree1):
        with pytest.raises(ValueError, match="No interaction coefficients"):
            interaction_heatmap(viz_adata_degree1, show=False)

    def test_pair_labels_present(self, viz_adata):
        fig = interaction_heatmap(viz_adata, show=False)
        # K=3 gives C(3,2)=3 pairs
        assert len(fig.data[0].x) == 3


class TestR2Barplot:
    def test_returns_figure(self, viz_adata):
        fig = r2_barplot(viz_adata, show=False)
        assert isinstance(fig, go.Figure)

    def test_has_bar_trace(self, viz_adata):
        fig = r2_barplot(viz_adata, show=False)
        assert isinstance(fig.data[0], go.Bar)

    def test_top_n_limits_features(self, viz_adata):
        fig = r2_barplot(viz_adata, top_n=5, show=False)
        assert len(fig.data[0].y) == 5

    def test_r2_values_are_nonnegative(self, viz_adata):
        fig = r2_barplot(viz_adata, show=False)
        # All bar x-values (R^2) should be >= 0
        assert all(v >= 0 for v in fig.data[0].x)

    def test_bars_are_horizontal(self, viz_adata):
        fig = r2_barplot(viz_adata, show=False)
        assert fig.data[0].orientation == "h"


class TestVertexRadar:
    def test_returns_figure(self, viz_adata):
        fig = vertex_radar(viz_adata, "gene_0", show=False)
        assert isinstance(fig, go.Figure)

    def test_has_scatterpolar_trace(self, viz_adata):
        fig = vertex_radar(viz_adata, "gene_0", show=False)
        assert isinstance(fig.data[0], go.Scatterpolar)

    def test_polygon_is_closed(self, viz_adata):
        fig = vertex_radar(viz_adata, "gene_0", show=False)
        # K=3 archetypes => 4 points (3 + 1 to close)
        assert len(fig.data[0].r) == 4
        assert fig.data[0].r[0] == fig.data[0].r[-1]

    def test_missing_feature_raises(self, viz_adata):
        with pytest.raises(ValueError, match="not found"):
            vertex_radar(viz_adata, "nonexistent_gene", show=False)

    def test_different_features_give_different_plots(self, viz_adata):
        fig0 = vertex_radar(viz_adata, "gene_0", show=False)
        fig1 = vertex_radar(viz_adata, "gene_1", show=False)
        # Different features should have different coefficient values
        assert fig0.data[0].r != fig1.data[0].r


class TestRegressionVolcano:
    def test_returns_figure(self, viz_adata):
        fig = regression_volcano(viz_adata, show=False)
        assert isinstance(fig, go.Figure)

    def test_has_scatter_trace(self, viz_adata):
        fig = regression_volcano(viz_adata, show=False)
        assert isinstance(fig.data[0], go.Scatter)

    def test_correct_number_of_points(self, viz_adata):
        fig = regression_volcano(viz_adata, show=False)
        # Should have one point per feature (20 genes)
        assert len(fig.data[0].x) == 20

    def test_hover_text_has_feature_names(self, viz_adata):
        fig = regression_volcano(viz_adata, show=False)
        text = fig.data[0].text
        assert "gene_0" in text

    def test_contrast_values_nonnegative(self, viz_adata):
        fig = regression_volcano(viz_adata, show=False)
        # ptp (peak to peak) is always >= 0
        assert all(v >= 0 for v in fig.data[0].x)


class TestPatternSummary:
    def test_returns_figure(self, viz_adata):
        fig = pattern_summary(viz_adata, show=False)
        assert isinstance(fig, go.Figure)

    def test_has_bar_trace(self, viz_adata):
        fig = pattern_summary(viz_adata, show=False)
        assert isinstance(fig.data[0], go.Bar)

    def test_counts_sum_to_n_features(self, viz_adata):
        fig = pattern_summary(viz_adata, show=False)
        total = sum(fig.data[0].y)
        assert total == 20  # n_genes

    def test_missing_patterns_raises(self):
        adata = AnnData(np.zeros((10, 5)))
        with pytest.raises(ValueError, match="No pattern classification"):
            pattern_summary(adata, show=False)

    def test_pattern_labels_are_strings(self, viz_adata):
        fig = pattern_summary(viz_adata, show=False)
        assert all(isinstance(x, str) for x in fig.data[0].x)


class TestImportsFromPackage:
    """Verify functions are accessible via peach.pl namespace."""

    def test_import_from_pl(self):
        from peach.pl import (
            coefficient_heatmap,
            interaction_heatmap,
            r2_barplot,
            vertex_radar,
            regression_volcano,
            pattern_summary,
        )
        # All should be callables
        assert callable(coefficient_heatmap)
        assert callable(interaction_heatmap)
        assert callable(r2_barplot)
        assert callable(vertex_radar)
        assert callable(regression_volcano)
        assert callable(pattern_summary)

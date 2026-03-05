"""Tests for GMM decomposition visualization functions."""

import numpy as np
import pytest
from anndata import AnnData
import plotly.graph_objects as go

from peach.pl.decomposition import (
    component_scatter,
    gmm_bic_curve,
    component_heatmap,
    component_stability,
)


@pytest.fixture
def gmm_viz_adata():
    """AnnData with GMM decomposition results for visualization."""
    rng = np.random.default_rng(42)
    K = 3
    n = 400
    n_genes = 20

    # Two distinct clusters in weight space
    w0 = rng.dirichlet([10, 1, 1], size=200)
    w1 = rng.dirichlet([1, 10, 1], size=200)
    weights = np.vstack([w0, w1])

    X = rng.standard_normal((n, n_genes))
    adata = AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obsm["cell_archetype_weights"] = weights
    adata.obsm["X_pca"] = rng.standard_normal((n, 10))

    import peach as pc

    pc.tl.feature_simplex_decomposition(
        adata,
        n_initializations=3,
        n_components_range=(2, 4),
    )
    return adata


@pytest.fixture
def empty_adata():
    """AnnData with no GMM results."""
    return AnnData(np.zeros((10, 5)))


class TestComponentScatter:
    def test_returns_figure(self, gmm_viz_adata):
        fig = component_scatter(gmm_viz_adata, show=False)
        assert isinstance(fig, go.Figure)

    def test_has_scatter_traces(self, gmm_viz_adata):
        fig = component_scatter(gmm_viz_adata, show=False)
        assert len(fig.data) >= 1
        for trace in fig.data:
            assert isinstance(trace, go.Scatter)

    def test_trace_names_contain_component(self, gmm_viz_adata):
        fig = component_scatter(gmm_viz_adata, show=False)
        component_traces = [t for t in fig.data if "Component" in t.name]
        assert len(component_traces) >= 1

    def test_custom_pca_key(self, gmm_viz_adata):
        rng = np.random.default_rng(0)
        gmm_viz_adata.obsm["X_custom_pca"] = rng.standard_normal(
            (gmm_viz_adata.n_obs, 5)
        )
        fig = component_scatter(gmm_viz_adata, pca_key="X_custom_pca", show=False)
        assert isinstance(fig, go.Figure)

    def test_missing_pca_raises(self, gmm_viz_adata):
        with pytest.raises(ValueError, match="not found"):
            component_scatter(gmm_viz_adata, pca_key="X_nonexistent", show=False)

    def test_missing_gmm_raises(self, empty_adata):
        with pytest.raises(ValueError, match="No GMM results"):
            component_scatter(empty_adata, show=False)

    def test_missing_labels_raises(self, gmm_viz_adata):
        del gmm_viz_adata.obsm["peach_gmm_labels"]
        with pytest.raises(ValueError, match="No GMM labels"):
            component_scatter(gmm_viz_adata, show=False)

    def test_save_path(self, gmm_viz_adata, tmp_path):
        save_path = str(tmp_path / "scatter.html")
        fig = component_scatter(gmm_viz_adata, save_path=save_path, show=False)
        assert isinstance(fig, go.Figure)
        assert (tmp_path / "scatter.html").exists()

    def test_layout_titles(self, gmm_viz_adata):
        fig = component_scatter(gmm_viz_adata, show=False)
        assert "PCA" in fig.layout.title.text or "GMM" in fig.layout.title.text
        assert fig.layout.xaxis.title.text == "PC1"
        assert fig.layout.yaxis.title.text == "PC2"


class TestGmmBicCurve:
    def test_returns_figure(self, gmm_viz_adata):
        fig = gmm_bic_curve(gmm_viz_adata, show=False)
        assert isinstance(fig, go.Figure)

    def test_has_scatter_trace(self, gmm_viz_adata):
        fig = gmm_bic_curve(gmm_viz_adata, show=False)
        assert len(fig.data) >= 1
        assert isinstance(fig.data[0], go.Scatter)

    def test_mode_is_lines_markers(self, gmm_viz_adata):
        fig = gmm_bic_curve(gmm_viz_adata, show=False)
        assert fig.data[0].mode == "lines+markers"

    def test_correct_number_of_points(self, gmm_viz_adata):
        fig = gmm_bic_curve(gmm_viz_adata, show=False)
        n_tested = len(gmm_viz_adata.uns["peach_gmm"]["n_components_tested"])
        assert len(fig.data[0].x) == n_tested

    def test_missing_gmm_raises(self, empty_adata):
        with pytest.raises(ValueError, match="No GMM results"):
            gmm_bic_curve(empty_adata, show=False)

    def test_save_path(self, gmm_viz_adata, tmp_path):
        save_path = str(tmp_path / "bic_curve.html")
        fig = gmm_bic_curve(gmm_viz_adata, save_path=save_path, show=False)
        assert isinstance(fig, go.Figure)
        assert (tmp_path / "bic_curve.html").exists()

    def test_layout_titles(self, gmm_viz_adata):
        fig = gmm_bic_curve(gmm_viz_adata, show=False)
        assert "BIC" in fig.layout.title.text
        assert fig.layout.xaxis.title.text == "Number of Components"
        assert fig.layout.yaxis.title.text == "BIC"


class TestComponentHeatmap:
    def test_returns_figure(self, gmm_viz_adata):
        fig = component_heatmap(gmm_viz_adata, show=False)
        assert isinstance(fig, go.Figure)

    def test_has_heatmap_trace(self, gmm_viz_adata):
        fig = component_heatmap(gmm_viz_adata, show=False)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Heatmap)

    def test_top_n_limits_features(self, gmm_viz_adata):
        fig = component_heatmap(gmm_viz_adata, top_n=5, show=False)
        # x-axis should have 5 features
        assert len(fig.data[0].x) == 5

    def test_top_n_exceeds_features(self, gmm_viz_adata):
        # 20 genes but top_n=100 -- should show all 20
        fig = component_heatmap(gmm_viz_adata, top_n=100, show=False)
        assert len(fig.data[0].x) == 20

    def test_y_axis_has_component_labels(self, gmm_viz_adata):
        fig = component_heatmap(gmm_viz_adata, show=False)
        for label in fig.data[0].y:
            assert "Comp" in label

    def test_missing_gmm_raises(self, empty_adata):
        with pytest.raises(ValueError, match="No GMM results"):
            component_heatmap(empty_adata, show=False)

    def test_missing_profiles_raises(self, gmm_viz_adata):
        # Remove feature profiles to trigger the specific error
        del gmm_viz_adata.uns["peach_gmm"]["component_feature_profiles"]
        with pytest.raises(ValueError, match="No feature profiles"):
            component_heatmap(gmm_viz_adata, show=False)

    def test_save_path(self, gmm_viz_adata, tmp_path):
        save_path = str(tmp_path / "heatmap.html")
        fig = component_heatmap(gmm_viz_adata, save_path=save_path, show=False)
        assert isinstance(fig, go.Figure)
        assert (tmp_path / "heatmap.html").exists()


class TestComponentStability:
    def test_returns_figure(self, gmm_viz_adata):
        fig = component_stability(gmm_viz_adata, show=False)
        assert isinstance(fig, go.Figure)

    def test_has_bar_trace(self, gmm_viz_adata):
        fig = component_stability(gmm_viz_adata, show=False)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Bar)

    def test_correct_number_of_bars(self, gmm_viz_adata):
        fig = component_stability(gmm_viz_adata, show=False)
        n_stable = gmm_viz_adata.uns["peach_gmm"]["n_components_stable"]
        assert len(fig.data[0].x) == n_stable

    def test_scores_in_unit_interval(self, gmm_viz_adata):
        fig = component_stability(gmm_viz_adata, show=False)
        scores = fig.data[0].y
        assert all(0 <= s <= 1 for s in scores)

    def test_bar_labels_contain_comp(self, gmm_viz_adata):
        fig = component_stability(gmm_viz_adata, show=False)
        for label in fig.data[0].x:
            assert "Comp" in label

    def test_missing_gmm_raises(self, empty_adata):
        with pytest.raises(ValueError, match="No GMM results"):
            component_stability(empty_adata, show=False)

    def test_save_path(self, gmm_viz_adata, tmp_path):
        save_path = str(tmp_path / "stability.html")
        fig = component_stability(gmm_viz_adata, save_path=save_path, show=False)
        assert isinstance(fig, go.Figure)
        assert (tmp_path / "stability.html").exists()

    def test_yaxis_range(self, gmm_viz_adata):
        fig = component_stability(gmm_viz_adata, show=False)
        assert fig.layout.yaxis.range[0] == 0
        assert fig.layout.yaxis.range[1] == 1.05


class TestImportsFromPackage:
    """Verify functions are accessible via peach.pl namespace."""

    def test_import_from_pl(self):
        from peach.pl import (
            component_scatter,
            gmm_bic_curve,
            component_heatmap,
            component_stability,
        )
        assert callable(component_scatter)
        assert callable(gmm_bic_curve)
        assert callable(component_heatmap)
        assert callable(component_stability)

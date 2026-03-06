"""Tests for flow matching visualization functions."""

import numpy as np
import pytest
from anndata import AnnData
import plotly.graph_objects as go

from peach.pl.flow import (
    velocity_quiver,
    gene_alignment_barplot,
    jacobian_heatmap,
    trajectory_ribbon,
    flow_magnitude,
    density_comparison,
    archetype_correspondence,
)
from peach._core.types import (
    FlowWithinResult,
    FlowBetweenResult,
    GeneAlignmentResult,
    FlowJacobianResult,
)


@pytest.fixture(scope="module")
def flow_viz_data():
    """AnnData + FlowWithinResult for visualization tests.

    Trains a small flow model on synthetic data with two separated
    clusters in PCA space.
    """
    rng = np.random.default_rng(42)
    n = 200
    dim = 10

    pca_a = rng.normal(0, 1, (n, dim))
    pca_b = rng.normal(3, 1, (n, dim))

    X = rng.standard_normal((2 * n, 50))
    adata = AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(50)]
    adata.obsm["X_pca"] = np.vstack([pca_a, pca_b])
    adata.obs["treatment"] = ["Base"] * n + ["PD1"] * n
    adata.varm["PCs"] = rng.standard_normal((50, dim))

    import peach as pc

    flow_result = pc.tl.flow_within(
        adata,
        source={"treatment": "Base"},
        target={"treatment": "PD1"},
        n_epochs=50,
        hidden_dims=(32, 32),
    )
    return adata, flow_result


@pytest.fixture(scope="module")
def alignment_data(flow_viz_data):
    """GeneAlignmentResult from the flow model."""
    adata, flow_result = flow_viz_data
    import peach as pc

    return pc.tl.flow_gene_alignment(adata, flow_result, n_top=20)


@pytest.fixture
def jacobian_data():
    """Synthetic FlowJacobianResult for visualization."""
    rng = np.random.default_rng(42)
    dim = 5
    return FlowJacobianResult(
        jacobian_det=rng.standard_normal(100),
        feature_expansion=rng.standard_normal(50),
        mean_jacobian=rng.standard_normal((dim, dim)),
        t=0.5,
    )


@pytest.fixture
def between_result_with_correspondence():
    """FlowBetweenResult with archetype_correspondence populated."""
    rng = np.random.default_rng(42)
    K_src, K_tgt = 4, 3
    corr_matrix = rng.random((K_src, K_tgt))
    # Normalize rows to sum to 1
    corr_matrix = corr_matrix / corr_matrix.sum(axis=1, keepdims=True)

    return FlowBetweenResult(
        condition_key="condition",
        condition_labels=["A", "B"],
        flows={},
        archetype_correspondence={("A", "B"): corr_matrix},
    )


@pytest.fixture
def between_result_no_correspondence():
    """FlowBetweenResult without archetype_correspondence."""
    return FlowBetweenResult(
        condition_key="condition",
        condition_labels=["A", "B"],
        flows={},
        archetype_correspondence=None,
    )


# ---------- velocity_quiver ----------

class TestVelocityQuiver:
    def test_returns_figure(self, flow_viz_data):
        adata, flow_result = flow_viz_data
        fig = velocity_quiver(adata, flow_result, show=False)
        assert isinstance(fig, go.Figure)

    def test_has_scatter_trace(self, flow_viz_data):
        adata, flow_result = flow_viz_data
        fig = velocity_quiver(adata, flow_result, show=False)
        assert len(fig.data) >= 1
        assert isinstance(fig.data[0], go.Scatter)

    def test_has_annotations(self, flow_viz_data):
        adata, flow_result = flow_viz_data
        fig = velocity_quiver(adata, flow_result, n_arrows=50, show=False)
        assert len(fig.layout.annotations) == 50

    def test_n_arrows_capped_by_source(self, flow_viz_data):
        adata, flow_result = flow_viz_data
        n_source = flow_result.source_mask.sum()
        fig = velocity_quiver(adata, flow_result, n_arrows=10000, show=False)
        assert len(fig.layout.annotations) == n_source

    def test_custom_pca_key(self, flow_viz_data):
        adata, flow_result = flow_viz_data
        adata.obsm["X_custom"] = adata.obsm["X_pca"].copy()
        fig = velocity_quiver(adata, flow_result, pca_key="X_custom", show=False)
        assert isinstance(fig, go.Figure)

    def test_layout_titles(self, flow_viz_data):
        adata, flow_result = flow_viz_data
        fig = velocity_quiver(adata, flow_result, show=False)
        assert fig.layout.xaxis.title.text == "PC1"
        assert fig.layout.yaxis.title.text == "PC2"
        assert "Velocity" in fig.layout.title.text or "Flow" in fig.layout.title.text

    def test_save_path(self, flow_viz_data, tmp_path):
        adata, flow_result = flow_viz_data
        save_path = str(tmp_path / "quiver.html")
        fig = velocity_quiver(adata, flow_result, save_path=save_path, show=False)
        assert isinstance(fig, go.Figure)
        assert (tmp_path / "quiver.html").exists()


# ---------- gene_alignment_barplot ----------

class TestGeneAlignmentBarplot:
    def test_returns_figure(self, flow_viz_data, alignment_data):
        adata, _ = flow_viz_data
        fig = gene_alignment_barplot(adata, alignment_data, show=False)
        assert isinstance(fig, go.Figure)

    def test_has_bar_trace(self, flow_viz_data, alignment_data):
        adata, _ = flow_viz_data
        fig = gene_alignment_barplot(adata, alignment_data, show=False)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Bar)

    def test_horizontal_orientation(self, flow_viz_data, alignment_data):
        adata, _ = flow_viz_data
        fig = gene_alignment_barplot(adata, alignment_data, show=False)
        assert fig.data[0].orientation == "h"

    def test_n_top_controls_bars(self, flow_viz_data, alignment_data):
        adata, _ = flow_viz_data
        fig = gene_alignment_barplot(adata, alignment_data, n_top=5, show=False)
        # 5 top + 5 bottom = 10 bars
        assert len(fig.data[0].y) == 10

    def test_default_n_top_is_20(self, flow_viz_data, alignment_data):
        adata, _ = flow_viz_data
        fig = gene_alignment_barplot(adata, alignment_data, n_top=20, show=False)
        assert len(fig.data[0].y) == 40

    def test_title_contains_alignment(self, flow_viz_data, alignment_data):
        adata, _ = flow_viz_data
        fig = gene_alignment_barplot(adata, alignment_data, show=False)
        assert "alignment" in fig.layout.title.text.lower()

    def test_save_path(self, flow_viz_data, alignment_data, tmp_path):
        adata, _ = flow_viz_data
        save_path = str(tmp_path / "alignment.html")
        fig = gene_alignment_barplot(
            adata, alignment_data, save_path=save_path, show=False
        )
        assert isinstance(fig, go.Figure)
        assert (tmp_path / "alignment.html").exists()


# ---------- jacobian_heatmap ----------

class TestJacobianHeatmap:
    def test_returns_figure(self, flow_viz_data, jacobian_data):
        adata, _ = flow_viz_data
        fig = jacobian_heatmap(adata, jacobian_data, show=False)
        assert isinstance(fig, go.Figure)

    def test_has_heatmap_trace(self, flow_viz_data, jacobian_data):
        adata, _ = flow_viz_data
        fig = jacobian_heatmap(adata, jacobian_data, show=False)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Heatmap)

    def test_heatmap_dimensions(self, flow_viz_data, jacobian_data):
        adata, _ = flow_viz_data
        fig = jacobian_heatmap(adata, jacobian_data, show=False)
        dim = jacobian_data.mean_jacobian.shape[0]
        assert len(fig.data[0].x) == dim
        assert len(fig.data[0].y) == dim

    def test_pc_labels(self, flow_viz_data, jacobian_data):
        adata, _ = flow_viz_data
        fig = jacobian_heatmap(adata, jacobian_data, show=False)
        assert fig.data[0].x[0] == "PC1"
        assert fig.data[0].y[0] == "PC1"

    def test_colorscale_diverging(self, flow_viz_data, jacobian_data):
        adata, _ = flow_viz_data
        fig = jacobian_heatmap(adata, jacobian_data, show=False)
        assert fig.data[0].zmid == 0

    def test_title_contains_t(self, flow_viz_data, jacobian_data):
        adata, _ = flow_viz_data
        fig = jacobian_heatmap(adata, jacobian_data, show=False)
        assert "t=0.5" in fig.layout.title.text

    def test_layout_titles(self, flow_viz_data, jacobian_data):
        adata, _ = flow_viz_data
        fig = jacobian_heatmap(adata, jacobian_data, show=False)
        assert fig.layout.xaxis.title.text == "Input PC"
        assert fig.layout.yaxis.title.text == "Output PC"

    def test_save_path(self, flow_viz_data, jacobian_data, tmp_path):
        adata, _ = flow_viz_data
        save_path = str(tmp_path / "jacobian.html")
        fig = jacobian_heatmap(
            adata, jacobian_data, save_path=save_path, show=False
        )
        assert isinstance(fig, go.Figure)
        assert (tmp_path / "jacobian.html").exists()


# ---------- trajectory_ribbon ----------

class TestTrajectoryRibbon:
    def test_returns_figure_interpolation(self, flow_viz_data):
        adata, flow_result = flow_viz_data
        fig = trajectory_ribbon(adata, flow_result, show=False)
        assert isinstance(fig, go.Figure)

    def test_has_scatter_traces(self, flow_viz_data):
        adata, flow_result = flow_viz_data
        fig = trajectory_ribbon(
            adata, flow_result, n_steps=5, n_sample=10, show=False
        )
        # n_steps + 1 time points
        assert len(fig.data) == 6

    def test_traces_are_scatter(self, flow_viz_data):
        adata, flow_result = flow_viz_data
        fig = trajectory_ribbon(
            adata, flow_result, n_steps=5, n_sample=10, show=False
        )
        for trace in fig.data:
            assert isinstance(trace, go.Scatter)

    def test_n_sample_capped(self, flow_viz_data):
        adata, flow_result = flow_viz_data
        n_source = flow_result.source_mask.sum()
        fig = trajectory_ribbon(
            adata, flow_result, n_sample=100000, n_steps=3, show=False
        )
        # Each trace should have n_source points
        assert len(fig.data[0].x) == n_source

    def test_layout_titles(self, flow_viz_data):
        adata, flow_result = flow_viz_data
        fig = trajectory_ribbon(adata, flow_result, show=False)
        assert fig.layout.xaxis.title.text == "PC1"
        assert fig.layout.yaxis.title.text == "PC2"

    def test_save_path(self, flow_viz_data, tmp_path):
        adata, flow_result = flow_viz_data
        save_path = str(tmp_path / "trajectory.html")
        fig = trajectory_ribbon(
            adata, flow_result, save_path=save_path, show=False
        )
        assert isinstance(fig, go.Figure)
        assert (tmp_path / "trajectory.html").exists()


# ---------- flow_magnitude ----------

class TestFlowMagnitude:
    def test_returns_figure(self, flow_viz_data):
        adata, flow_result = flow_viz_data
        fig = flow_magnitude(adata, flow_result, show=False)
        assert isinstance(fig, go.Figure)

    def test_has_scatter_trace(self, flow_viz_data):
        adata, flow_result = flow_viz_data
        fig = flow_magnitude(adata, flow_result, show=False)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Scatter)

    def test_scatter_has_color(self, flow_viz_data):
        adata, flow_result = flow_viz_data
        fig = flow_magnitude(adata, flow_result, show=False)
        marker = fig.data[0].marker
        assert marker.color is not None
        assert len(marker.color) == flow_result.source_mask.sum()

    def test_color_is_magnitude(self, flow_viz_data):
        adata, flow_result = flow_viz_data
        fig = flow_magnitude(adata, flow_result, show=False)
        source_pca = adata.obsm["X_pca"][flow_result.source_mask]
        expected_mag = np.linalg.norm(
            flow_result.transported - source_pca, axis=1
        )
        np.testing.assert_allclose(fig.data[0].marker.color, expected_mag)

    def test_colorbar_present(self, flow_viz_data):
        adata, flow_result = flow_viz_data
        fig = flow_magnitude(adata, flow_result, show=False)
        assert fig.data[0].marker.colorbar is not None

    def test_layout_titles(self, flow_viz_data):
        adata, flow_result = flow_viz_data
        fig = flow_magnitude(adata, flow_result, show=False)
        assert fig.layout.xaxis.title.text == "PC1"
        assert fig.layout.yaxis.title.text == "PC2"

    def test_save_path(self, flow_viz_data, tmp_path):
        adata, flow_result = flow_viz_data
        save_path = str(tmp_path / "magnitude.html")
        fig = flow_magnitude(
            adata, flow_result, save_path=save_path, show=False
        )
        assert isinstance(fig, go.Figure)
        assert (tmp_path / "magnitude.html").exists()


# ---------- density_comparison ----------

class TestDensityComparison:
    def test_returns_figure(self, flow_viz_data):
        adata, flow_result = flow_viz_data
        fig = density_comparison(adata, flow_result, show=False)
        assert isinstance(fig, go.Figure)

    def test_has_three_traces(self, flow_viz_data):
        adata, flow_result = flow_viz_data
        fig = density_comparison(adata, flow_result, show=False)
        assert len(fig.data) == 3

    def test_traces_are_histogram(self, flow_viz_data):
        adata, flow_result = flow_viz_data
        fig = density_comparison(adata, flow_result, show=False)
        for trace in fig.data:
            assert isinstance(trace, go.Histogram)

    def test_trace_names(self, flow_viz_data):
        adata, flow_result = flow_viz_data
        fig = density_comparison(adata, flow_result, show=False)
        names = {t.name for t in fig.data}
        assert names == {"Source", "Transported", "Target"}

    def test_overlay_barmode(self, flow_viz_data):
        adata, flow_result = flow_viz_data
        fig = density_comparison(adata, flow_result, show=False)
        assert fig.layout.barmode == "overlay"

    def test_layout_titles(self, flow_viz_data):
        adata, flow_result = flow_viz_data
        fig = density_comparison(adata, flow_result, show=False)
        assert fig.layout.xaxis.title.text == "PC1"
        assert fig.layout.yaxis.title.text == "Density"

    def test_save_path(self, flow_viz_data, tmp_path):
        adata, flow_result = flow_viz_data
        save_path = str(tmp_path / "density.html")
        fig = density_comparison(
            adata, flow_result, save_path=save_path, show=False
        )
        assert isinstance(fig, go.Figure)
        assert (tmp_path / "density.html").exists()


# ---------- archetype_correspondence ----------

class TestArchetypeCorrespondence:
    def test_returns_figure(self, between_result_with_correspondence):
        fig = archetype_correspondence(
            between_result_with_correspondence, show=False
        )
        assert isinstance(fig, go.Figure)

    def test_has_heatmap_trace(self, between_result_with_correspondence):
        fig = archetype_correspondence(
            between_result_with_correspondence, show=False
        )
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Heatmap)

    def test_heatmap_dimensions(self, between_result_with_correspondence):
        fig = archetype_correspondence(
            between_result_with_correspondence, show=False
        )
        # 4 source archetypes, 3 target archetypes
        z = np.array(fig.data[0].z)
        assert z.shape == (4, 3)

    def test_axis_labels(self, between_result_with_correspondence):
        fig = archetype_correspondence(
            between_result_with_correspondence, show=False
        )
        assert len(fig.data[0].x) == 3
        assert len(fig.data[0].y) == 4
        assert "Target" in fig.data[0].x[0] or "Tgt" in fig.data[0].x[0]
        assert "Source" in fig.data[0].y[0] or "Src" in fig.data[0].y[0]

    def test_no_correspondence_raises(self, between_result_no_correspondence):
        with pytest.raises(ValueError, match="No archetype correspondence"):
            archetype_correspondence(
                between_result_no_correspondence, show=False
            )

    def test_layout_titles(self, between_result_with_correspondence):
        fig = archetype_correspondence(
            between_result_with_correspondence, show=False
        )
        assert "correspondence" in fig.layout.title.text.lower()
        assert "target" in fig.layout.xaxis.title.text.lower()
        assert "source" in fig.layout.yaxis.title.text.lower()

    def test_save_path(self, between_result_with_correspondence, tmp_path):
        save_path = str(tmp_path / "correspondence.html")
        fig = archetype_correspondence(
            between_result_with_correspondence,
            save_path=save_path,
            show=False,
        )
        assert isinstance(fig, go.Figure)
        assert (tmp_path / "correspondence.html").exists()


# ---------- Import tests ----------

class TestImportsFromPackage:
    """Verify functions are accessible via peach.pl namespace."""

    def test_import_from_pl(self):
        from peach.pl import (
            velocity_quiver,
            gene_alignment_barplot,
            jacobian_heatmap,
            trajectory_ribbon,
            flow_magnitude,
            density_comparison,
            archetype_correspondence,
        )
        assert callable(velocity_quiver)
        assert callable(gene_alignment_barplot)
        assert callable(jacobian_heatmap)
        assert callable(trajectory_ribbon)
        assert callable(flow_magnitude)
        assert callable(density_comparison)
        assert callable(archetype_correspondence)

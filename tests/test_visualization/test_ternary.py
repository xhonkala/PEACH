"""Tests for ternary facet plots."""

import numpy as np
import pytest
from anndata import AnnData
import plotly.graph_objects as go


@pytest.fixture
def ternary_adata():
    """AnnData with Dirichlet-sampled archetype weights for ternary plotting."""
    rng = np.random.default_rng(42)
    K = 4
    n = 200
    weights = rng.dirichlet([1] * K, size=n)
    X = rng.standard_normal((n, 30)).astype(np.float32)
    adata = AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(30)]
    adata.obsm["cell_archetype_weights"] = weights
    adata.obs["cluster"] = rng.choice(["A", "B"], size=n)
    adata.obs["score"] = rng.standard_normal(n)
    return adata


class TestTernaryFacet:
    """Tests for ternary_facet()."""

    def test_returns_figure(self, ternary_adata):
        """Valid plotly figure returned for default arguments."""
        from peach.pl.ternary import ternary_facet

        fig = ternary_facet(ternary_adata, archetypes=(0, 1, 2), show=False)
        assert isinstance(fig, go.Figure)
        # Should have exactly one trace
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Scatterternary)

    def test_weight_renormalization(self, ternary_adata):
        """Renormalized weights in trace should sum to ~1."""
        from peach.pl.ternary import ternary_facet

        fig = ternary_facet(ternary_adata, archetypes=(0, 1, 2), show=False)
        trace = fig.data[0]
        sums = np.array(trace.a) + np.array(trace.b) + np.array(trace.c)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_color_by_gene(self, ternary_adata):
        """Color by gene expression produces colored markers."""
        from peach.pl.ternary import ternary_facet

        fig = ternary_facet(
            ternary_adata, archetypes=(0, 1, 2), color_by="gene_0", show=False
        )
        assert isinstance(fig, go.Figure)
        # Marker should have color array set
        marker = fig.data[0].marker
        assert marker.color is not None
        assert len(marker.color) == ternary_adata.n_obs

    def test_color_by_obs_numeric(self, ternary_adata):
        """Color by numeric obs column."""
        from peach.pl.ternary import ternary_facet

        fig = ternary_facet(
            ternary_adata, archetypes=(0, 1, 2), color_by="score", show=False
        )
        assert isinstance(fig, go.Figure)
        marker = fig.data[0].marker
        assert marker.color is not None

    def test_color_by_density(self, ternary_adata):
        """Density coloring produces a figure."""
        from peach.pl.ternary import ternary_facet

        fig = ternary_facet(
            ternary_adata, archetypes=(0, 1, 2), color_by="density", show=False
        )
        assert isinstance(fig, go.Figure)

    def test_different_archetype_triples(self, ternary_adata):
        """Different archetype combinations work."""
        from peach.pl.ternary import ternary_facet

        for triple in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]:
            fig = ternary_facet(ternary_adata, archetypes=triple, show=False)
            assert isinstance(fig, go.Figure)
            # Check axis labels match archetype indices
            layout = fig.layout.ternary
            assert str(triple[0]) in layout.aaxis.title.text
            assert str(triple[1]) in layout.baxis.title.text
            assert str(triple[2]) in layout.caxis.title.text

    def test_invalid_archetypes_raises(self, ternary_adata):
        """Non-existent archetype indices raise ValueError."""
        from peach.pl.ternary import ternary_facet

        with pytest.raises(ValueError, match="out of range"):
            ternary_facet(ternary_adata, archetypes=(0, 1, 10), show=False)

    def test_missing_weights_raises(self):
        """Missing archetype weights raise KeyError."""
        from peach.pl.ternary import ternary_facet

        adata = AnnData(np.random.randn(50, 10))
        with pytest.raises(KeyError):
            ternary_facet(adata, archetypes=(0, 1, 2), show=False)

    def test_save_path(self, ternary_adata, tmp_path):
        """Save to HTML file."""
        from peach.pl.ternary import ternary_facet

        outfile = str(tmp_path / "ternary_test.html")
        fig = ternary_facet(
            ternary_adata, archetypes=(0, 1, 2), save_path=outfile, show=False
        )
        assert isinstance(fig, go.Figure)
        import os

        assert os.path.exists(outfile)
        assert os.path.getsize(outfile) > 0

    def test_custom_marker_kwargs(self, ternary_adata):
        """Custom marker_size and marker_opacity are respected."""
        from peach.pl.ternary import ternary_facet

        fig = ternary_facet(
            ternary_adata,
            archetypes=(0, 1, 2),
            show=False,
            marker_size=8,
            marker_opacity=0.9,
        )
        marker = fig.data[0].marker
        assert marker.size == 8
        assert marker.opacity == 0.9


class TestTernaryFacetGrid:
    """Tests for ternary_facet_grid()."""

    def test_all_facets_count(self, ternary_adata):
        """All C(K,3) facets generated for K=4."""
        from peach.pl.ternary import ternary_facet_grid

        figures = ternary_facet_grid(ternary_adata, facets="all", show=False)
        # C(4,3) = 4
        assert len(figures) == 4
        assert all(isinstance(f, go.Figure) for f in figures)

    def test_selected_facets(self, ternary_adata):
        """Specific facets generated."""
        from peach.pl.ternary import ternary_facet_grid

        triples = [(0, 1, 2), (1, 2, 3)]
        figures = ternary_facet_grid(ternary_adata, facets=triples, show=False)
        assert len(figures) == 2

    def test_grid_with_color(self, ternary_adata):
        """Color_by propagates to all facets."""
        from peach.pl.ternary import ternary_facet_grid

        figures = ternary_facet_grid(
            ternary_adata, facets=[(0, 1, 2)], color_by="gene_0", show=False
        )
        assert len(figures) == 1
        marker = figures[0].data[0].marker
        assert marker.color is not None

    def test_single_facet_grid(self, ternary_adata):
        """Grid with a single triple works."""
        from peach.pl.ternary import ternary_facet_grid

        figures = ternary_facet_grid(
            ternary_adata, facets=[(0, 1, 2)], show=False
        )
        assert len(figures) == 1

    def test_k3_all_facets(self):
        """With K=3, 'all' produces exactly 1 facet."""
        from peach.pl.ternary import ternary_facet_grid

        rng = np.random.default_rng(99)
        K = 3
        n = 100
        weights = rng.dirichlet([1] * K, size=n)
        adata = AnnData(rng.standard_normal((n, 10)).astype(np.float32))
        adata.obsm["cell_archetype_weights"] = weights

        figures = ternary_facet_grid(adata, facets="all", show=False)
        assert len(figures) == 1


class TestImportFromPackage:
    """Verify ternary functions are accessible from peach.pl."""

    def test_import_ternary_facet(self):
        from peach.pl import ternary_facet

        assert callable(ternary_facet)

    def test_import_ternary_facet_grid(self):
        from peach.pl import ternary_facet_grid

        assert callable(ternary_facet_grid)

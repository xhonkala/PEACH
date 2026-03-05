import numpy as np
import pytest
from anndata import AnnData


@pytest.fixture
def flow_adata():
    """AnnData with two conditions in PCA space."""
    rng = np.random.default_rng(42)
    n = 200
    dim = 10

    # Two conditions: shifted in PCA space
    pca_a = rng.normal(0, 1, (n, dim))
    pca_b = rng.normal(3, 1, (n, dim))

    X = rng.standard_normal((2 * n, 50))
    adata = AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(50)]
    adata.obsm["X_pca"] = np.vstack([pca_a, pca_b])
    adata.obs["treatment"] = ["Base"] * n + ["PD1"] * n
    # Add PCA loadings
    adata.varm["PCs"] = rng.standard_normal((50, dim))
    return adata


class TestFlowWithin:
    def test_basic_run(self, flow_adata):
        """Runs on single adata with obs-defined subsets."""
        import peach as pc

        result = pc.tl.flow_within(
            flow_adata,
            source={"treatment": "Base"},
            target={"treatment": "PD1"},
            n_epochs=50,
            hidden_dims=(32, 32),
        )
        assert result is not None
        assert result.transported.shape[1] == 10

    def test_mmd_improves(self, flow_adata):
        """mmd_after < mmd_before."""
        import peach as pc

        result = pc.tl.flow_within(
            flow_adata,
            source={"treatment": "Base"},
            target={"treatment": "PD1"},
            n_epochs=200,
            hidden_dims=(64, 64),
        )
        assert result.mmd_after < result.mmd_before


class TestFlowBetween:
    def test_two_adatas(self):
        """Concatenates and trains flows between two AnnDatas."""
        import peach as pc

        rng = np.random.default_rng(42)
        dim = 5
        adata1 = AnnData(rng.standard_normal((100, 20)))
        adata1.obsm["X_pca"] = rng.normal(0, 1, (100, dim))
        adata2 = AnnData(rng.standard_normal((100, 20)))
        adata2.obsm["X_pca"] = rng.normal(3, 1, (100, dim))

        result = pc.tl.flow_between(
            [adata1, adata2],
            n_epochs=50,
            hidden_dims=(32, 32),
        )
        assert len(result.flows) == 1  # one pair

    def test_pca_dim_mismatch_errors(self):
        """Different n_PCs raises ValueError."""
        import peach as pc

        rng = np.random.default_rng(42)
        adata1 = AnnData(rng.standard_normal((50, 10)))
        adata1.obsm["X_pca"] = rng.normal(0, 1, (50, 5))
        adata2 = AnnData(rng.standard_normal((50, 10)))
        adata2.obsm["X_pca"] = rng.normal(0, 1, (50, 10))  # different dim

        with pytest.raises(ValueError, match="dimensions"):
            pc.tl.flow_between([adata1, adata2], n_epochs=10, hidden_dims=(16,))


class TestGeneAlignment:
    def test_returns_rankings(self, flow_adata):
        """Top aligned/opposed genes returned."""
        import peach as pc

        flow_result = pc.tl.flow_within(
            flow_adata,
            source={"treatment": "Base"},
            target={"treatment": "PD1"},
            n_epochs=50,
            hidden_dims=(32, 32),
        )
        alignment = pc.tl.flow_gene_alignment(flow_adata, flow_result, n_top=10)
        assert len(alignment.top_aligned) == 10
        assert len(alignment.top_opposed) == 10
        assert alignment.alignment_scores.shape == (50,)

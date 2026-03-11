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
        assert result["transported"].shape[1] == 10

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
        assert result["mmd_after"] < result["mmd_before"]


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
        assert len(result["flows"]) == 1  # one pair

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
        assert len(alignment["top_aligned"]) == 10
        assert len(alignment["top_opposed"]) == 10
        assert alignment["alignment_scores"].shape == (50,)


class TestFlowJacobian:
    def test_basic_run(self, flow_adata):
        """Jacobian returns correct shapes."""
        import peach as pc
        from peach._core.utils.flow_matching import FlowModel, compute_mmd

        # Train model manually (flow_within doesn't expose it)
        source_pca = flow_adata.obsm["X_pca"][:200]
        target_pca = flow_adata.obsm["X_pca"][200:]
        dim = source_pca.shape[1]

        model = FlowModel(dim, hidden_dims=(32, 32), lr=1e-3)
        model.train(source_pca, target_pca, n_epochs=50, batch_size=64)

        source_mask = np.array([True] * 200 + [False] * 200)
        target_mask = ~source_mask
        transported = model.transport(source_pca, n_steps=10)

        flow_result = {
            "source_mask": source_mask,
            "target_mask": target_mask,
            "transported": transported,
            "losses": [0.0],
            "mmd_before": compute_mmd(source_pca, target_pca),
            "mmd_after": compute_mmd(transported, target_pca),
            "pca_key": "X_pca",
        }

        jac_result = pc.tl.flow_jacobian(flow_adata, flow_result, model, t=0.5)
        assert jac_result["jacobian_det"].shape == (200,)
        assert jac_result["mean_jacobian"].shape == (dim, dim)
        assert jac_result["feature_expansion"].shape == (50,)
        assert jac_result["t"] == 0.5

    def test_custom_evaluation_points(self, flow_adata):
        """Works with custom evaluation points subset."""
        import peach as pc
        from peach._core.utils.flow_matching import FlowModel, compute_mmd

        source_pca = flow_adata.obsm["X_pca"][:200]
        target_pca = flow_adata.obsm["X_pca"][200:]
        dim = source_pca.shape[1]

        model = FlowModel(dim, hidden_dims=(32, 32), lr=1e-3)
        model.train(source_pca, target_pca, n_epochs=50, batch_size=64)

        source_mask = np.array([True] * 200 + [False] * 200)
        target_mask = ~source_mask
        transported = model.transport(source_pca, n_steps=10)
        flow_result = {
            "source_mask": source_mask, "target_mask": target_mask,
            "transported": transported, "losses": [0.0],
            "mmd_before": 1.0, "mmd_after": 0.5, "pca_key": "X_pca",
        }

        # Evaluate on just 10 points
        eval_pts = source_pca[:10]
        jac_result = pc.tl.flow_jacobian(
            flow_adata, flow_result, model, evaluation_points=eval_pts
        )
        assert jac_result["jacobian_det"].shape == (10,)


class TestFlowSignificance:
    def test_returns_pvalue(self, flow_adata):
        """Permutation test returns p_value and null distribution."""
        import peach as pc

        result = pc.tl.flow_significance(
            flow_adata,
            source={"treatment": "Base"},
            target={"treatment": "PD1"},
            n_permutations=5,
            n_epochs_per_perm=20,
            hidden_dims=(16, 16),
            n_steps=5,
        )
        assert "p_value" in result
        assert "observed_stat" in result
        assert "null_distribution" in result
        assert 0.0 <= result["p_value"] <= 1.0
        assert len(result["null_distribution"]) == 5

    def test_shifted_distributions_significant(self, flow_adata):
        """Well-separated conditions should have small p-value (or at least observed > null mean)."""
        import peach as pc

        result = pc.tl.flow_significance(
            flow_adata,
            source={"treatment": "Base"},
            target={"treatment": "PD1"},
            n_permutations=9,
            n_epochs_per_perm=50,
            hidden_dims=(32, 32),
            n_steps=10,
        )
        # Observed MMD should be larger than typical null values
        assert result["observed_stat"] > np.mean(result["null_distribution"])

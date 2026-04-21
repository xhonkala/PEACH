"""Tests for vectorized Jacobian computation and flow gene alignment permutation stats."""

import numpy as np
import pytest


def _make_flow_fixture(n_cells=100, n_genes=50, K=3, return_model=True, seed=42):
    """Create minimal AnnData + trained flow for testing."""
    import anndata as ad
    from peach._core.utils.flow_matching import FlowModel
    import torch

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
    adata = ad.AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obsm["X_pca"] = rng.standard_normal((n_cells, 10)).astype(np.float32)
    adata.obsm["cell_archetype_weights"] = rng.dirichlet(np.ones(K), n_cells).astype(np.float32)
    adata.varm["PCs"] = rng.standard_normal((n_genes, 10)).astype(np.float32)
    adata.obs["condition"] = ["source"] * (n_cells // 2) + ["target"] * (n_cells - n_cells // 2)

    import peach as pc
    flow_result = pc.tl.flow_within(
        adata,
        source={"condition": "source"},
        target={"condition": "target"},
        n_epochs=50,
        hidden_dims=(32, 32),
        return_model=return_model,
        random_state=seed,
    )
    return adata, flow_result


def _make_feature_graph_result(seed=42):
    """Deprecated — flow_feature_graph removed. Retained as placeholder."""
    return None


def test_jacobian_vectorized_correct():
    """Vectorized Jacobian should produce finite, non-zero values."""
    from peach._core.utils.flow_matching import FlowModel

    model = FlowModel(dim=5, hidden_dims=(32, 32))
    source = np.random.randn(50, 5).astype(np.float32)
    target = np.random.randn(50, 5).astype(np.float32)
    model.train(source, target, n_epochs=20, batch_size=32)
    points = source[:10]
    jac = model.jacobian(points, t=0.5)
    assert jac.shape == (10, 5, 5)
    assert np.all(np.isfinite(jac))
    assert not np.allclose(jac, 0, atol=1e-8), "Jacobian is all zeros"


def test_jacobian_matches_finite_difference():
    """Jacobian should closely match finite-difference estimate after sufficient training."""
    from peach._core.utils.flow_matching import FlowModel

    rng = np.random.default_rng(99)
    model = FlowModel(dim=3, hidden_dims=(32, 32), random_state=99)
    source = rng.standard_normal((50, 3)).astype(np.float32)
    target = (rng.standard_normal((50, 3)) + 2.0).astype(np.float32)
    model.train(source, target, n_epochs=200, batch_size=32)
    point = source[:1]
    jac = model.jacobian(point, t=0.5)[0]
    eps = 1e-4
    jac_fd = np.zeros((3, 3))
    for k in range(3):
        p_plus = point.copy(); p_minus = point.copy()
        p_plus[0, k] += eps; p_minus[0, k] -= eps
        v_plus = model.velocity_at(p_plus, 0.5)[0]
        v_minus = model.velocity_at(p_minus, 0.5)[0]
        jac_fd[:, k] = (v_plus - v_minus) / (2 * eps)
    np.testing.assert_allclose(jac, jac_fd, atol=1e-3, rtol=1e-2)


def test_batch_size_imbalance_warning():
    """Training with >5x size imbalance should emit a warning."""
    from peach._core.utils.flow_matching import FlowModel

    model = FlowModel(dim=3, hidden_dims=(16, 16))
    source = np.random.randn(10, 3).astype(np.float32)
    target = np.random.randn(60, 3).astype(np.float32)  # 6x imbalance
    with pytest.warns(UserWarning, match="imbalance"):
        model.train(source, target, n_epochs=5, batch_size=8)


def test_no_imbalance_warning_for_balanced():
    """Training with balanced sizes should NOT emit a warning."""
    from peach._core.utils.flow_matching import FlowModel

    model = FlowModel(dim=3, hidden_dims=(16, 16))
    source = np.random.randn(30, 3).astype(np.float32)
    target = np.random.randn(30, 3).astype(np.float32)
    # Should not raise any warning
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.train(source, target, n_epochs=5, batch_size=16)


def test_flow_gene_alignment_permutation():
    """flow_gene_alignment with n_permutations > 0 should return p-values."""
    import anndata as ad
    from peach.tl.flow import flow_gene_alignment

    rng = np.random.default_rng(42)
    n_cells = 100
    n_genes = 50
    n_pcs = 10

    # Create synthetic AnnData with PCA loadings
    X = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
    adata = ad.AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obsm["X_pca"] = rng.standard_normal((n_cells, n_pcs)).astype(np.float32)
    adata.varm["PCs"] = rng.standard_normal((n_genes, n_pcs)).astype(np.float32)

    # Create a fake flow_result
    source_mask = np.zeros(n_cells, dtype=bool)
    source_mask[:50] = True
    transported = adata.obsm["X_pca"][:50] + rng.standard_normal((50, n_pcs)).astype(np.float32) * 0.1

    flow_result = {
        "source_mask": source_mask,
        "transported": transported,
        "pca_key": "X_pca",
    }

    # Without permutations
    result_no_perm = flow_gene_alignment(adata, flow_result)
    assert "alignment_pvalues" not in result_no_perm

    # With permutations
    result_perm = flow_gene_alignment(adata, flow_result, n_permutations=100, random_state=42)
    assert "alignment_pvalues" in result_perm
    assert "alignment_pvalues_fdr" in result_perm
    assert "null_mean" in result_perm
    assert "null_std" in result_perm
    assert len(result_perm["alignment_pvalues"]) == n_genes
    assert len(result_perm["alignment_pvalues_fdr"]) == n_genes
    # p-values should be in [0, 1]
    assert np.all(result_perm["alignment_pvalues"] >= 0)
    assert np.all(result_perm["alignment_pvalues"] <= 1)
    assert np.all(result_perm["alignment_pvalues_fdr"] >= 0)
    assert np.all(result_perm["alignment_pvalues_fdr"] <= 1)


def test_dopri5_default_solver():
    """FlowModel and flow_within should default to dopri5 solver."""
    from peach._core.utils.flow_matching import FlowModel

    model = FlowModel(dim=3, hidden_dims=(16, 16))
    assert model.solver_method == "dopri5"

    # Verify it can train and transport with dopri5
    source = np.random.randn(30, 3).astype(np.float32)
    target = np.random.randn(30, 3).astype(np.float32)
    model.train(source, target, n_epochs=10, batch_size=16)
    transported = model.transport(source[:5], n_steps=10)
    assert transported.shape == (5, 3)
    assert np.all(np.isfinite(transported))


def test_ot_cfm_training():
    """OT-CFM training should produce decreasing losses."""
    pot = pytest.importorskip("ot")
    from peach._core.utils.flow_matching import FlowModel

    rng = np.random.default_rng(42)
    # Use shifted distributions so there's actual transport to learn
    source = rng.standard_normal((100, 5)).astype(np.float32)
    target = (rng.standard_normal((100, 5)) + 3.0).astype(np.float32)
    model = FlowModel(dim=5, hidden_dims=(32, 32))
    losses = model.train(source, target, n_epochs=80, batch_size=64, use_ot=True)
    assert len(losses) == 80
    # Compare mean of first 10 vs last 10 epochs (smoothed) to avoid single-epoch noise
    assert np.mean(losses[-10:]) < np.mean(losses[:10])


def test_holdout_validation():
    """flow_within with holdout_fraction should return holdout MMD."""
    import anndata as ad
    from peach.tl.flow import flow_within

    rng = np.random.default_rng(42)
    n = 200
    pca = rng.standard_normal((n, 5)).astype(np.float32)
    adata = ad.AnnData(rng.standard_normal((n, 10)))
    adata.obsm["X_pca"] = pca
    adata.obs["group"] = ["A"] * 100 + ["B"] * 100
    result = flow_within(
        adata, {"group": "A"}, {"group": "B"},
        n_epochs=50, batch_size=64, holdout_fraction=0.2,
    )
    assert "holdout_mmd" in result
    assert result["holdout_fraction"] == 0.2
    assert result["holdout_mmd"] >= 0


def test_per_cell_gene_alignment():
    """flow_gene_alignment with per_cell=True should return per-cell scores."""
    import anndata as ad
    from peach.tl.flow import flow_gene_alignment

    rng = np.random.default_rng(42)
    n_cells, n_genes, n_pcs = 100, 50, 10
    X = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
    adata = ad.AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obsm["X_pca"] = rng.standard_normal((n_cells, n_pcs)).astype(np.float32)
    adata.varm["PCs"] = rng.standard_normal((n_genes, n_pcs)).astype(np.float32)

    source_mask = np.zeros(n_cells, dtype=bool)
    source_mask[:50] = True
    transported = adata.obsm["X_pca"][:50] + rng.standard_normal((50, n_pcs)).astype(np.float32) * 0.5
    flow_result = {"source_mask": source_mask, "transported": transported, "pca_key": "X_pca"}

    result = flow_gene_alignment(adata, flow_result, per_cell=True)
    assert "per_cell_alignment" in result
    assert "per_cell_gene_names" in result
    assert "per_cell_gene_indices" in result
    n_top_feat = min(2500, n_genes)
    assert result["per_cell_alignment"].shape == (50, n_top_feat)
    assert len(result["per_cell_gene_names"]) == n_top_feat
    assert len(result["per_cell_gene_indices"]) == n_top_feat

    # Also test with explicit cap smaller than n_genes
    result_capped = flow_gene_alignment(
        adata, flow_result, per_cell=True, n_top_features=10
    )
    assert result_capped["per_cell_alignment"].shape == (50, 10)
    assert len(result_capped["per_cell_gene_names"]) == 10
    assert len(result_capped["per_cell_gene_indices"]) == 10


def test_bifurcation_scoring():
    """flow_bifurcation should return divergence and eigenvalue arrays."""
    import anndata as ad
    from peach.tl.flow import flow_within, flow_bifurcation

    rng = np.random.default_rng(42)
    n, dim = 100, 5
    source = rng.standard_normal((n, dim)).astype(np.float32)
    target = rng.standard_normal((n, dim)).astype(np.float32)
    adata = ad.AnnData(rng.standard_normal((n * 2, 10)))
    adata.obsm["X_pca"] = np.vstack([source, target])
    adata.obs["group"] = ["A"] * n + ["B"] * n

    flow_result = flow_within(
        adata, {"group": "A"}, {"group": "B"},
        n_epochs=30, batch_size=64, return_model=True,
    )
    bif = flow_bifurcation(adata, flow_result, flow_result["model"], n_timepoints=5)
    assert "divergence" in bif
    assert "bifurcation_score" in bif
    assert "eigenvalue_real" in bif
    assert "eigenvalue_imag" in bif
    assert "n_saddle_points" not in bif  # removed: was mathematically meaningless
    assert len(bif["bifurcation_score"]) == n
    assert bif["divergence"].shape == (5, n)
    assert bif["eigenvalue_real"].shape == (5, n, dim)
    assert bif["eigenvalue_imag"].shape == (5, n, dim)
    assert len(bif["timepoints"]) == 5


def test_temporal_graph_deprecated():
    """flow_temporal_feature_graph should raise DeprecationWarning + NotImplementedError."""
    import pytest
    import warnings
    from peach.tl.flow import flow_within, flow_temporal_feature_graph
    import anndata as ad

    rng = np.random.default_rng(42)
    n, dim, n_genes = 40, 3, 10
    adata = ad.AnnData(rng.standard_normal((n * 2, n_genes)).astype(np.float32))
    adata.obsm["X_pca"] = rng.standard_normal((n * 2, dim)).astype(np.float32)
    adata.varm["PCs"] = rng.standard_normal((n_genes, dim)).astype(np.float32)
    adata.obs["group"] = ["A"] * n + ["B"] * n

    flow_result = flow_within(adata, {"group": "A"}, {"group": "B"},
                              n_epochs=5, hidden_dims=(8,), return_model=True)

    with pytest.warns(DeprecationWarning, match="flow_temporal_feature_graph is deprecated"):
        with pytest.raises(NotImplementedError):
            flow_temporal_feature_graph(adata, flow_result, flow_result["model"])


def test_feature_graph_deprecated():
    """flow_feature_graph should raise DeprecationWarning + NotImplementedError."""
    import pytest
    from peach.tl.flow import flow_feature_graph

    adata, flow_result = _make_flow_fixture(return_model=True)
    model = flow_result["model"]

    with pytest.warns(DeprecationWarning, match="flow_feature_graph is deprecated"):
        with pytest.raises(NotImplementedError):
            flow_feature_graph(adata, flow_result, model)


def test_gene_alignment_t_parameter_changes_output():
    """Verify that different t values produce different alignment scores
    when a model is available."""
    import peach as pc

    # Use the shared synthetic flow fixture
    adata, flow_result = _make_flow_fixture(return_model=True)

    result_t01 = pc.tl.flow_gene_alignment(adata, flow_result, t=0.1)
    result_t09 = pc.tl.flow_gene_alignment(adata, flow_result, t=0.9)

    # Different t values MUST produce different alignment scores
    assert not np.allclose(
        result_t01["alignment_scores"],
        result_t09["alignment_scores"],
        atol=1e-6,
    ), "t parameter had no effect on alignment scores"

    # Both should report instantaneous mode
    assert result_t01["velocity_mode"] == "instantaneous"
    assert result_t09["velocity_mode"] == "instantaneous"


def test_gene_alignment_no_model_uses_displacement():
    """Without a model in flow_result, t is ignored and displacement is used."""
    import peach as pc

    adata, flow_result = _make_flow_fixture(return_model=False)

    result_default = pc.tl.flow_gene_alignment(adata, flow_result)
    result_with_t = pc.tl.flow_gene_alignment(adata, flow_result, t=0.3)

    # Without model, t has no effect (both use displacement)
    np.testing.assert_array_equal(
        result_default["alignment_scores"],
        result_with_t["alignment_scores"],
    )
    assert result_default.get("velocity_mode") == "displacement"


def test_feature_expansion_invariant_to_loading_scale():
    """feature_expansion should depend on flow direction, not PCA loading magnitude."""
    import peach as pc

    adata, flow_result = _make_flow_fixture(return_model=True)
    model = flow_result["model"]

    # Compute feature expansion
    jac_result = pc.tl.flow_jacobian(adata, flow_result, model)

    # Scale PCA loadings by 10x — should NOT change feature_expansion
    adata2 = adata.copy()
    adata2.varm["PCs"] = adata.varm["PCs"] * 10.0
    jac_result2 = pc.tl.flow_jacobian(adata2, flow_result, model)

    np.testing.assert_allclose(
        jac_result["feature_expansion"],
        jac_result2["feature_expansion"],
        atol=1e-6,
        err_msg="feature_expansion should be invariant to PCA loading scale",
    )


def test_jacobian_det_nonzero_after_training():
    """After real training, Jacobian determinant should not be all zeros."""
    from peach._core.utils.flow_matching import FlowModel

    rng = np.random.default_rng(42)
    model = FlowModel(dim=5, hidden_dims=(64, 64))
    source = rng.standard_normal((100, 5)).astype(np.float32)
    target = source + rng.standard_normal((100, 5)).astype(np.float32) * 2
    model.train(source, target, n_epochs=100, batch_size=64)
    jac = model.jacobian(source[:5], t=0.5)
    dets = np.linalg.det(jac)
    assert not np.allclose(dets, 0, atol=1e-6), f"Jacobian dets all ~0: {dets}"
    assert np.all(np.isfinite(dets))


def test_per_cell_jacobian_expansion_shape():
    """flow_jacobian with per_cell_features=True returns correct shape."""
    import peach as pc

    adata, flow_result = _make_flow_fixture(n_genes=50, return_model=True)
    model = flow_result["model"]

    result = pc.tl.flow_jacobian(adata, flow_result, model, per_cell_features=True)

    assert "per_cell_expansion" in result
    assert "per_cell_expansion_gene_names" in result
    assert "per_cell_expansion_gene_indices" in result

    n_source = flow_result["source_mask"].sum()
    n_top_feat = min(2500, 50)  # 50 genes in fixture
    assert result["per_cell_expansion"].shape == (n_source, n_top_feat)
    assert len(result["per_cell_expansion_gene_names"]) == n_top_feat
    assert np.all(np.isfinite(result["per_cell_expansion"]))


def test_per_cell_expansion_mean_matches_aggregated():
    """Mean of per-cell expansion should approximate aggregated feature_expansion."""
    import peach as pc
    from scipy.stats import spearmanr

    adata, flow_result = _make_flow_fixture(n_genes=30, return_model=True)
    model = flow_result["model"]

    result = pc.tl.flow_jacobian(
        adata, flow_result, model,
        per_cell_features=True, n_top_features=30,  # all genes
    )

    agg = result["feature_expansion"]
    pc_mean = result["per_cell_expansion"].mean(axis=0)
    pc_idx = result["per_cell_expansion_gene_indices"]

    # Not exact (mean of quadratic forms != quadratic form of mean),
    # but should be correlated
    rho, _ = spearmanr(agg[pc_idx], pc_mean)
    assert rho > 0.8, (
        f"Per-cell expansion mean should correlate with aggregated: Spearman={rho:.4f}"
    )


def test_per_cell_expansion_disabled():
    """per_cell_features=False should not include per-cell keys."""
    import peach as pc

    adata, flow_result = _make_flow_fixture(return_model=True)
    model = flow_result["model"]

    result = pc.tl.flow_jacobian(
        adata, flow_result, model, per_cell_features=False
    )
    assert "per_cell_expansion" not in result
    assert "per_cell_expansion_gene_names" not in result
    # Aggregated feature_expansion should still be present
    assert "feature_expansion" in result


def test_per_cell_expansion_invariant_to_loading_scale():
    """Per-cell expansion should not change when loadings are scaled."""
    import peach as pc

    adata, flow_result = _make_flow_fixture(n_genes=30, return_model=True)
    model = flow_result["model"]

    result1 = pc.tl.flow_jacobian(
        adata, flow_result, model,
        per_cell_features=True, n_top_features=30,
    )

    adata2 = adata.copy()
    adata2.varm["PCs"] = adata.varm["PCs"] * 10.0
    result2 = pc.tl.flow_jacobian(
        adata2, flow_result, model,
        per_cell_features=True, n_top_features=30,
    )

    np.testing.assert_allclose(
        result1["per_cell_expansion"],
        result2["per_cell_expansion"],
        atol=1e-5,
        err_msg="Per-cell expansion should be invariant to loading scale",
    )


# ── flow_map_jacobian tests ──────────────────────────────────────────────────

def test_flow_map_jacobian_shapes():
    """flow_map_jacobian returns correct shapes for phi and J."""
    from peach._core.utils.flow_matching import FlowModel

    rng = np.random.default_rng(42)
    dim = 4
    src = rng.standard_normal((30, dim)).astype(np.float32)
    tgt = rng.standard_normal((30, dim)).astype(np.float32) + 2.0

    model = FlowModel(dim, hidden_dims=(16, 16), random_state=42)
    model.train(src, tgt, n_epochs=30, batch_size=16)

    result = model.flow_map_jacobian(src, t_eval=[0.5], n_steps=10)
    assert result['phi'].shape == (1, 30, dim)
    assert result['J'].shape == (1, 30, dim, dim)
    assert result['t_eval'] == [0.5]

    result3 = model.flow_map_jacobian(src, t_eval=[0.25, 0.5, 0.75], n_steps=10)
    assert result3['phi'].shape == (3, 30, dim)
    assert result3['J'].shape == (3, 30, dim, dim)


def test_flow_map_jacobian_identity_at_t0():
    """J at t≈0 (very small t) should be close to identity."""
    from peach._core.utils.flow_matching import FlowModel

    rng = np.random.default_rng(0)
    dim = 3
    src = rng.standard_normal((20, dim)).astype(np.float32)
    tgt = rng.standard_normal((20, dim)).astype(np.float32) + 3.0

    model = FlowModel(dim, hidden_dims=(16, 16), random_state=0)
    model.train(src, tgt, n_epochs=50, batch_size=16)

    # At t=0.02 with 200 midpoint steps, J should be very close to I.
    result = model.flow_map_jacobian(src, t_eval=[0.02], n_steps=200)
    J_small_t = result['J'][0]  # [n_cells, dim, dim]
    I = np.eye(dim)
    for j in J_small_t:
        np.testing.assert_allclose(j, I, atol=0.05,
            err_msg="J at t≈0 should be near identity")


def test_flow_map_jacobian_finite_and_nonidentity():
    """J at t=0.5 should be finite and not all identity after real transport."""
    from peach._core.utils.flow_matching import FlowModel

    rng = np.random.default_rng(1)
    dim = 4
    src = rng.standard_normal((50, dim)).astype(np.float32)
    tgt = (rng.standard_normal((50, dim)) + 5.0).astype(np.float32)

    model = FlowModel(dim, hidden_dims=(32, 32), random_state=1)
    model.train(src, tgt, n_epochs=100, batch_size=32)

    result = model.flow_map_jacobian(src[:10], t_eval=[0.5], n_steps=20)
    J_mid = result['J'][0]

    assert np.all(np.isfinite(J_mid)), "J contains non-finite values"
    I = np.eye(dim)
    # After real transport, at least some cells should have J != I
    max_dev = max(np.linalg.norm(j - I) for j in J_mid)
    assert max_dev > 0.01, f"All J near identity after training (max_dev={max_dev})"


def test_flow_map_jacobian_phi_matches_transport_midpoint():
    """phi from flow_map_jacobian at t=1 should match transport() when both use midpoint."""
    from peach._core.utils.flow_matching import FlowModel

    rng = np.random.default_rng(7)
    dim = 3
    src = rng.standard_normal((20, dim)).astype(np.float32)
    tgt = (rng.standard_normal((20, dim)) + 2.0).astype(np.float32)

    # Use midpoint for both to get exact agreement (both use fixed-step RK2 at n_steps=50)
    model = FlowModel(dim, hidden_dims=(16, 16), solver_method='midpoint', random_state=7)
    model.train(src, tgt, n_epochs=50, batch_size=16)

    fmj = model.flow_map_jacobian(src, t_eval=[1.0], n_steps=50)
    phi_fmj = fmj['phi'][0]
    phi_transport = model.transport(src, n_steps=50)
    np.testing.assert_allclose(phi_fmj, phi_transport, atol=1e-3,
        err_msg="phi from flow_map_jacobian should match transport() at t=1 (both midpoint)")


def test_flow_map_jacobian_phi_dopri5_approximate():
    """With default dopri5 transport, phi from flow_map_jacobian deviates slightly.

    flow_map_jacobian uses torchdiffeq midpoint internally (autograd nesting prevents
    dopri5). This test documents the expected discrepancy between the two solvers.
    """
    from peach._core.utils.flow_matching import FlowModel

    rng = np.random.default_rng(13)
    dim = 3
    src = rng.standard_normal((20, dim)).astype(np.float32)
    tgt = (rng.standard_normal((20, dim)) + 2.0).astype(np.float32)

    model = FlowModel(dim, hidden_dims=(16, 16), solver_method='dopri5', random_state=13)
    model.train(src, tgt, n_epochs=50, batch_size=16)

    fmj = model.flow_map_jacobian(src, t_eval=[1.0], n_steps=100)
    phi_fmj = fmj['phi'][0]
    phi_transport = model.transport(src)  # dopri5 adaptive

    # phi should be in the same ballpark but not exact — solver methods differ.
    # Tolerance is loose by design: this documents the known discrepancy.
    assert phi_fmj.shape == phi_transport.shape
    max_dev = np.abs(phi_fmj - phi_transport).max()
    assert max_dev < 1.0, (
        f"phi from midpoint flow_map_jacobian deviates too much from dopri5 transport: "
        f"max_dev={max_dev:.4f}. Increase n_steps in flow_map_jacobian to reduce error."
    )


def test_flow_jacobian_scalar_t_backward_compat():
    """Scalar t returns same key structure as the old API."""
    import peach as pc

    adata, flow_result = _make_flow_fixture(n_genes=30, return_model=True)
    model = flow_result["model"]

    result = pc.tl.flow_jacobian(adata, flow_result, model, t=0.5,
                                 per_cell_features=True, n_top_features=10, n_steps=10)

    assert isinstance(result['t'], float)
    assert result['jacobian_det'].ndim == 1      # [n_cells]
    assert result['feature_expansion'].ndim == 1  # [n_genes]
    assert result['per_cell_expansion'].ndim == 2  # [n_cells, n_top]
    assert result['phi'].ndim == 2               # [n_cells, dim]
    assert np.all(np.isfinite(result['jacobian_det']))
    assert np.all(np.isfinite(result['feature_expansion']))


def test_flow_jacobian_list_t_shapes():
    """List t returns arrays with leading n_t dimension."""
    import peach as pc

    adata, flow_result = _make_flow_fixture(n_genes=30, return_model=True)
    model = flow_result["model"]
    n_source = flow_result["source_mask"].sum()

    result = pc.tl.flow_jacobian(adata, flow_result, model,
                                 t=[0.25, 0.5, 0.75],
                                 per_cell_features=True, n_top_features=10, n_steps=10)

    assert result['timepoints'] == [0.25, 0.5, 0.75]
    assert result['jacobian_det'].shape == (3, n_source)
    assert result['feature_expansion'].shape == (3, 30)
    assert result['per_cell_expansion'].shape == (3, n_source, 10)
    assert result['phi'].shape == (3, n_source, adata.obsm['X_pca'].shape[1])


def test_flow_jacobian_uses_transported_positions():
    """phi in result should differ from source positions — cells were transported."""
    import peach as pc

    adata, flow_result = _make_flow_fixture(return_model=True, seed=5)
    model = flow_result["model"]
    src_positions = adata.obsm['X_pca'][flow_result['source_mask']]

    result = pc.tl.flow_jacobian(adata, flow_result, model, t=0.5,
                                 per_cell_features=False, n_steps=20)

    # phi at t=0.5 must differ from starting positions
    assert not np.allclose(result['phi'], src_positions, atol=1e-4), \
        "phi should be transported positions, not source positions"


def test_validate_source_obs_names_raises_on_mismatch():
    """_validate_source_obs_names must raise ValueError when adata obs_names don't match flow_result source_obs_names."""
    import anndata as ad
    import peach as pc
    from peach.tl.flow import _validate_source_obs_names

    rng = np.random.default_rng(99)
    n_cells, n_genes = 60, 20
    X = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
    adata = ad.AnnData(X)
    adata.obsm["X_pca"] = rng.standard_normal((n_cells, 10)).astype(np.float32)
    adata.obs["condition"] = ["source"] * 30 + ["target"] * 30

    flow_result = pc.tl.flow_within(
        adata,
        source={"condition": "source"},
        target={"condition": "target"},
        n_epochs=10,
        hidden_dims=(16, 16),
        return_model=True,
        random_state=99,
    )

    # Different adata with same shape but different obs_names (no X_pca set → auto-generates different index)
    adata_wrong = ad.AnnData(rng.standard_normal((n_cells, n_genes)).astype(np.float32))
    adata_wrong.obs_names = [f"cell_alt_{i}" for i in range(n_cells)]

    with pytest.raises(ValueError, match="obs_names"):
        _validate_source_obs_names(adata_wrong, flow_result)

    # Original adata should pass without error
    _validate_source_obs_names(adata, flow_result)


def test_liouville_consistency():
    """Liouville equation: integrate(trace(dv/dx) dt, 0→T) ≈ mean log|det J(T)|.

    This cross-validates the two Jacobian computation paths:
    - flow_model.jacobian() [velocity Jacobian, dv/dx] via jacrev
    - flow_model.flow_map_jacobian() [flow map J, ∂φ_t/∂x₀] via coupled ODE

    Liouville's theorem: d/dt log|det J(t)| = div v(φ(t), t) = trace(dv/dx)|_{φ(t)}
    So: log|det J(T)| ≈ ∫₀ᵀ trace(dv/dx|_{φ(t)}) dt
    """
    from peach._core.utils.flow_matching import FlowModel
    from scipy.integrate import trapezoid

    rng = np.random.default_rng(77)
    dim = 3
    n_cells = 20
    src = rng.standard_normal((n_cells, dim)).astype(np.float32)
    tgt = (rng.standard_normal((n_cells, dim)) + 3.0).astype(np.float32)

    model = FlowModel(dim, hidden_dims=(32, 32), random_state=77)
    model.train(src, tgt, n_epochs=150, batch_size=32)

    T = 0.5
    n_steps = 100
    t_grid = np.linspace(0.0, T, n_steps + 1)

    # Flow map J at T — gives log|det J(T)| per cell
    fmj = model.flow_map_jacobian(src, t_eval=[T], n_steps=n_steps)
    logdet_J = fmj['J'][0]  # [n_cells, dim, dim]
    _, logdets = np.linalg.slogdet(logdet_J)  # [n_cells]

    # Transport trajectory to get positions at each grid point
    trajectory = model.transport(src, n_steps=n_steps, return_trajectory=True)
    # trajectory: [n_steps+1, n_cells, dim]  at times np.linspace(0, 1, n_steps+1)
    # We only need up to T: index = round(T * n_steps)
    t_idx_end = round(T * n_steps)
    traj_T = trajectory[:t_idx_end + 1]  # [t_idx_end+1, n_cells, dim]

    # Compute divergence = trace(dv/dx) at each trajectory time and position
    t_eval_grid = np.linspace(0.0, T, traj_T.shape[0])
    div_over_time = np.zeros((len(t_eval_grid), n_cells))
    for ti, t_val in enumerate(t_eval_grid):
        jac_v = model.jacobian(traj_T[ti], float(t_val))  # [n_cells, dim, dim]
        div_over_time[ti] = np.trace(jac_v, axis1=1, axis2=2)

    # Integrate divergence over time per cell via trapezoidal rule
    integrated_div = trapezoid(div_over_time, t_eval_grid, axis=0)  # [n_cells]

    # Mean agreement across cells — individual cells vary due to stochastic flow
    mean_logdet = np.mean(logdets)
    mean_integrated = np.mean(integrated_div)

    np.testing.assert_allclose(
        mean_logdet, mean_integrated, atol=0.3,
        err_msg=(
            f"Liouville: mean log|det J(T)|={mean_logdet:.4f} should ≈ "
            f"∫trace(dv/dx)dt={mean_integrated:.4f}. "
            "Large deviation indicates inconsistency between jacobian() and "
            "flow_map_jacobian()."
        )
    )


# ── compute_mmd tests ────────────────────────────────────────────────────────

def test_compute_mmd_same_distribution_near_zero():
    """MMD(X, X) should be near zero."""
    from peach._core.utils.flow_matching import compute_mmd

    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 5)).astype(np.float32)
    mmd = compute_mmd(X, X, random_state=42)
    assert np.isfinite(mmd)
    assert abs(mmd) < 0.05, f"MMD(X, X) should be ~0, got {mmd}"


def test_compute_mmd_different_distributions_positive():
    """MMD between well-separated distributions should be clearly positive."""
    from peach._core.utils.flow_matching import compute_mmd

    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 5)).astype(np.float32)
    Y = (rng.standard_normal((200, 5)) + 5.0).astype(np.float32)
    mmd = compute_mmd(X, Y, random_state=0)
    assert mmd > 0.1, f"MMD between shifted distributions should be large, got {mmd}"


def test_compute_mmd_random_state_reproducible():
    """Same random_state must produce the same MMD when subsampling occurs."""
    from peach._core.utils.flow_matching import compute_mmd

    rng = np.random.default_rng(7)
    X = rng.standard_normal((10000, 4)).astype(np.float32)
    Y = rng.standard_normal((10000, 4)).astype(np.float32)

    mmd_a = compute_mmd(X, Y, max_samples=200, random_state=42)
    mmd_b = compute_mmd(X, Y, max_samples=200, random_state=42)
    assert mmd_a == mmd_b, "Same random_state must give identical MMD"

    mmd_c = compute_mmd(X, Y, max_samples=200, random_state=99)
    assert np.isfinite(mmd_c)


# ── velocity_at and _VelocityWrapper tests ───────────────────────────────────

def test_velocity_at_shape_and_finite():
    """velocity_at should return shape [n, dim] with finite values."""
    from peach._core.utils.flow_matching import FlowModel

    rng = np.random.default_rng(5)
    model = FlowModel(dim=4, hidden_dims=(16, 16), random_state=5)
    src = rng.standard_normal((40, 4)).astype(np.float32)
    tgt = rng.standard_normal((40, 4)).astype(np.float32)
    model.train(src, tgt, n_epochs=10, batch_size=16)

    pts = src[:8]
    v = model.velocity_at(pts, t=0.5)
    assert v.shape == (8, 4)
    assert np.all(np.isfinite(v))


def test_velocity_wrapper_scalar_t():
    """_VelocityWrapper must handle scalar t (t.dim() == 0) without error."""
    import torch
    from peach._core.utils.flow_matching import FlowModel, _VelocityWrapper

    model = FlowModel(dim=3, hidden_dims=(16, 16), random_state=0)
    src = np.random.randn(20, 3).astype(np.float32)
    tgt = np.random.randn(20, 3).astype(np.float32)
    model.train(src, tgt, n_epochs=5, batch_size=16)

    wrapper = _VelocityWrapper(model.velocity_net)
    x = torch.randn(5, 3)
    t_scalar = torch.tensor(0.5)  # dim() == 0
    v = wrapper(x, t_scalar)
    assert v.shape == (5, 3)
    assert torch.all(torch.isfinite(v))


# ── _trained guard tests ─────────────────────────────────────────────────────

def test_transport_requires_training():
    """transport() before train() must raise RuntimeError."""
    from peach._core.utils.flow_matching import FlowModel

    model = FlowModel(dim=3, hidden_dims=(16, 16))
    with pytest.raises(RuntimeError, match="train"):
        model.transport(np.random.randn(5, 3).astype(np.float32))


def test_jacobian_requires_training():
    """jacobian() before train() must raise RuntimeError."""
    from peach._core.utils.flow_matching import FlowModel

    model = FlowModel(dim=3, hidden_dims=(16, 16))
    with pytest.raises(RuntimeError, match="train"):
        model.jacobian(np.random.randn(5, 3).astype(np.float32), t=0.5)


# ── batch size and input validation tests ────────────────────────────────────

def test_batch_size_larger_than_population():
    """Training with batch_size > len(source) should work via replacement sampling."""
    from peach._core.utils.flow_matching import FlowModel

    rng = np.random.default_rng(3)
    src = rng.standard_normal((20, 4)).astype(np.float32)
    tgt = rng.standard_normal((20, 4)).astype(np.float32)
    model = FlowModel(dim=4, hidden_dims=(16, 16), random_state=3)
    losses = model.train(src, tgt, n_epochs=10, batch_size=64)
    assert len(losses) == 10
    assert all(np.isfinite(l) for l in losses)


def test_train_validates_input_dim():
    """train() must raise ValueError when source or target dim != model dim."""
    from peach._core.utils.flow_matching import FlowModel

    model = FlowModel(dim=5, hidden_dims=(16, 16))
    wrong_src = np.random.randn(20, 3).astype(np.float32)
    right_tgt = np.random.randn(20, 5).astype(np.float32)
    with pytest.raises(ValueError, match="source"):
        model.train(wrong_src, right_tgt, n_epochs=1)

    right_src = np.random.randn(20, 5).astype(np.float32)
    wrong_tgt = np.random.randn(20, 3).astype(np.float32)
    with pytest.raises(ValueError, match="target"):
        model.train(right_src, wrong_tgt, n_epochs=1)


# ── phi discrepancy under default solver pairing ─────────────────────────────

def test_phi_discrepancy_midpoint_vs_dopri5_is_bounded():
    """phi from flow_map_jacobian (midpoint) should be within a reasonable bound
    of dopri5 transport at t=1 with sufficient n_steps."""
    from peach._core.utils.flow_matching import FlowModel

    rng = np.random.default_rng(42)
    dim = 4
    src = rng.standard_normal((30, dim)).astype(np.float32)
    tgt = (rng.standard_normal((30, dim)) + 2.0).astype(np.float32)

    model = FlowModel(dim, hidden_dims=(32, 32), random_state=42)
    model.train(src, tgt, n_epochs=100, batch_size=32)

    phi_dopri5 = model.transport(src)  # dopri5 adaptive
    fmj = model.flow_map_jacobian(src, t_eval=[1.0], n_steps=100)
    phi_midpoint = fmj['phi'][0]

    mae = np.abs(phi_dopri5 - phi_midpoint).mean()
    assert np.isfinite(mae)
    assert mae < 0.5, (
        f"phi discrepancy (dopri5 vs midpoint n_steps=100) is {mae:.4f}. "
        "Large values indicate midpoint step size needs increasing."
    )


def test_t_eval_zero_raises():
    """flow_map_jacobian must reject t_eval containing 0.0 or values > 1.0."""
    from peach._core.utils.flow_matching import FlowModel

    rng = np.random.default_rng(0)
    src = rng.standard_normal((10, 3)).astype(np.float32)
    tgt = rng.standard_normal((10, 3)).astype(np.float32)
    model = FlowModel(dim=3, hidden_dims=(16, 16), random_state=0)
    model.train(src, tgt, n_epochs=5)

    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        model.flow_map_jacobian(src, t_eval=[0.0, 0.5])

    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        model.flow_map_jacobian(src, t_eval=[1.5])

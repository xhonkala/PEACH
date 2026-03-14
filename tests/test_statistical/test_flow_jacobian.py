"""Tests for vectorized Jacobian computation and flow gene alignment permutation stats."""

import numpy as np
import pytest


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
    """Jacobian should approximately match finite-difference estimate."""
    from peach._core.utils.flow_matching import FlowModel

    model = FlowModel(dim=3, hidden_dims=(16, 16))
    source = np.random.randn(30, 3).astype(np.float32)
    target = np.random.randn(30, 3).astype(np.float32)
    model.train(source, target, n_epochs=50, batch_size=16)
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
    np.testing.assert_allclose(jac, jac_fd, atol=1e-2, rtol=1e-1)


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
    assert result["per_cell_alignment"].shape == (50, n_genes)
    assert "alignment_scores" in result


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
    assert "n_saddle_points" in bif
    assert len(bif["bifurcation_score"]) == n
    assert bif["divergence"].shape == (5, n)
    assert bif["eigenvalue_real"].shape == (5, n, dim)
    assert bif["eigenvalue_imag"].shape == (5, n, dim)
    assert len(bif["timepoints"]) == 5


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

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

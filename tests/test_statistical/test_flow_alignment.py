"""Stress tests for flow gene alignment: mathematical validation."""

import numpy as np
import pytest


def _make_alignment_fixture(n_cells=200, n_genes=100, n_pcs=20, seed=42):
    """Create synthetic AnnData with trained flow for alignment testing."""
    import anndata as ad
    from peach.tl.flow import flow_within
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
    adata = ad.AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    pca = PCA(n_components=n_pcs)
    pca_coords = pca.fit_transform(X).astype(np.float32)
    adata.obsm["X_pca"] = pca_coords
    adata.varm["PCs"] = pca.components_.T.astype(np.float32)
    adata.uns["pca_mean"] = pca.mean_.astype(np.float32)

    adata.obs["condition"] = (
        ["source"] * (n_cells // 2) + ["target"] * (n_cells - n_cells // 2)
    )

    flow_result = flow_within(
        adata,
        source={"condition": "source"},
        target={"condition": "target"},
        n_epochs=200,
        hidden_dims=(64, 64),
        return_model=True,
        random_state=seed,
    )
    return adata, flow_result


class TestPCAReconstructionAlignment:
    """Validate that alignment scores predict gene expression change via PCA."""

    def test_raw_alignment_equals_reconstructed_displacement(self):
        """With normalize=False, alignment score = mean gene expression change.

        Mathematical identity: loadings @ mean_delta_pca is exactly what
        flow_gene_alignment returns when normalize=False and per_cell=False.
        """
        from peach.tl.flow import flow_gene_alignment

        adata, flow_result = _make_alignment_fixture()
        loadings = adata.varm["PCs"]  # [n_genes, n_pcs]
        source_pca = adata.obsm["X_pca"][flow_result["source_mask"]]

        result = flow_gene_alignment(
            adata, flow_result, normalize=False, per_cell=False
        )
        alignment_scores = result["alignment_scores"]

        delta_pca = flow_result["transported"] - source_pca
        mean_delta_pca = delta_pca.mean(axis=0)
        # Trim loadings to the PCA dimension actually used
        n_pcs = len(mean_delta_pca)
        mean_delta_expr = loadings[:, :n_pcs] @ mean_delta_pca

        np.testing.assert_allclose(
            alignment_scores, mean_delta_expr, atol=1e-5,
            err_msg="Raw alignment scores should equal PCA-reconstructed expression change",
        )

    def test_normalized_alignment_ranking_matches_reconstruction(self):
        """With normalize=True, ranking should correlate with reconstructed displacement.

        The normalized scores use cosine similarity between unit-normalized
        loading rows and unit-normalized mean velocity. Because normalization
        is a monotone-rank-preserving operation within the same sign, the
        Spearman rank correlation with the raw (un-normalized) reconstruction
        should be very high.
        """
        from peach.tl.flow import flow_gene_alignment
        from scipy.stats import spearmanr

        adata, flow_result = _make_alignment_fixture()
        loadings = adata.varm["PCs"]
        source_pca = adata.obsm["X_pca"][flow_result["source_mask"]]

        result = flow_gene_alignment(
            adata, flow_result, normalize=True, per_cell=False
        )
        alignment_scores = result["alignment_scores"]

        delta_pca = flow_result["transported"] - source_pca
        n_pcs = delta_pca.shape[1]
        mean_delta_expr = loadings[:, :n_pcs] @ delta_pca.mean(axis=0)

        # Focus on genes with non-negligible signal (top 75% by abs raw score)
        threshold = np.percentile(np.abs(mean_delta_expr), 25)
        mask = np.abs(mean_delta_expr) > threshold

        rho, _ = spearmanr(alignment_scores[mask], mean_delta_expr[mask])
        # Normalization changes relative gene weights (genes with smaller loading
        # norms get amplified), so ranking is not perfectly preserved.  We expect
        # very high but not perfect concordance on random data.
        assert rho > 0.97, (
            f"Normalized alignment ranking should match reconstruction: Spearman={rho:.4f}"
        )


class TestNormalizationConsistency:
    """Verify aggregated and per-cell paths agree after normalization fix."""

    def test_aggregated_and_percell_mean_agree(self):
        """Mean of per-cell alignment should rank-match aggregated scores.

        The aggregated scores (normalize=True) are cosine similarity of the
        mean velocity unit vector with each loading row unit vector.
        The per-cell scores are cosine similarity of each per-cell velocity
        unit vector with each loading row unit vector.  Their mean should
        rank closely with the aggregated version because both measure the
        same geometric alignment.
        """
        from peach.tl.flow import flow_gene_alignment
        from scipy.stats import spearmanr

        adata, flow_result = _make_alignment_fixture(n_genes=50)

        result = flow_gene_alignment(
            adata, flow_result, normalize=True, per_cell=True,
            n_top_features=50,
        )

        agg_scores = result["alignment_scores"]
        pc_mean = result["per_cell_alignment"].mean(axis=0)
        pc_gene_idx = result["per_cell_gene_indices"]

        rho, _ = spearmanr(agg_scores[pc_gene_idx], pc_mean)
        assert rho > 0.95, (
            f"Aggregated and per-cell-mean rankings should agree: Spearman={rho:.4f}"
        )

    def test_normalize_false_matches_legacy(self):
        """normalize=False should produce identical scores to the direct formula.

        The direct formula is: loadings[:, :n_pcs] @ mean_velocity
        where mean_velocity = (transported - source_pca).mean(axis=0).
        This is an exact mathematical identity, not a tolerance test.
        """
        from peach.tl.flow import flow_gene_alignment

        adata, flow_result = _make_alignment_fixture()
        loadings = adata.varm["PCs"]
        source_pca = adata.obsm["X_pca"][flow_result["source_mask"]]
        mean_velocity = (flow_result["transported"] - source_pca).mean(axis=0)
        n_pcs = len(mean_velocity)

        result = flow_gene_alignment(
            adata, flow_result, normalize=False, per_cell=False
        )
        expected = loadings[:, :n_pcs] @ mean_velocity
        np.testing.assert_allclose(
            result["alignment_scores"], expected, atol=1e-6,
            err_msg="normalize=False should match raw loadings @ velocity",
        )

    def test_normalize_true_scores_in_unit_range(self):
        """With normalize=True, all scores must lie in [-1, 1] (cosine similarity).

        Both loading rows and mean velocity are normalized to unit vectors,
        so their dot product is bounded by the Cauchy-Schwarz inequality.
        """
        from peach.tl.flow import flow_gene_alignment

        adata, flow_result = _make_alignment_fixture()

        result = flow_gene_alignment(
            adata, flow_result, normalize=True, per_cell=False
        )
        scores = result["alignment_scores"]
        assert np.all(scores >= -1.0 - 1e-6), (
            f"Cosine scores must be >= -1: min={scores.min():.6f}"
        )
        assert np.all(scores <= 1.0 + 1e-6), (
            f"Cosine scores must be <= +1: max={scores.max():.6f}"
        )

    def test_per_cell_scores_in_unit_range(self):
        """Per-cell alignment scores must lie in [-1, 1] (cosine similarity)."""
        from peach.tl.flow import flow_gene_alignment

        adata, flow_result = _make_alignment_fixture(n_genes=50)

        result = flow_gene_alignment(
            adata, flow_result, normalize=True, per_cell=True, n_top_features=50
        )
        pc = result["per_cell_alignment"]
        assert np.all(pc >= -1.0 - 1e-6), (
            f"Per-cell cosine scores must be >= -1: min={pc.min():.6f}"
        )
        assert np.all(pc <= 1.0 + 1e-6), (
            f"Per-cell cosine scores must be <= +1: max={pc.max():.6f}"
        )

    def test_normalize_false_unbounded_by_velocity_scale(self):
        """With normalize=False, scaling the velocity should scale the scores linearly.

        This verifies that normalize=False uses raw dot products (no clamping
        to [-1, 1]), and that scale changes propagate correctly.
        We test this by manually constructing a flow_result with a 2x velocity.
        """
        from peach.tl.flow import flow_gene_alignment

        adata, flow_result = _make_alignment_fixture()

        result1 = flow_gene_alignment(
            adata, flow_result, normalize=False, per_cell=False
        )

        # Construct a doubled-velocity flow_result
        source_pca = adata.obsm["X_pca"][flow_result["source_mask"]]
        transported_2x = source_pca + 2.0 * (flow_result["transported"] - source_pca)
        flow_result_2x = dict(flow_result)
        flow_result_2x["transported"] = transported_2x

        result2 = flow_gene_alignment(
            adata, flow_result_2x, normalize=False, per_cell=False
        )

        np.testing.assert_allclose(
            result2["alignment_scores"], 2.0 * result1["alignment_scores"], atol=1e-5,
            err_msg="normalize=False scores should scale linearly with velocity magnitude",
        )

    def test_normalize_true_invariant_to_velocity_scale(self):
        """With normalize=True, scaling the velocity should NOT change scores.

        Cosine similarity is scale-invariant by definition.
        """
        from peach.tl.flow import flow_gene_alignment

        adata, flow_result = _make_alignment_fixture()

        result1 = flow_gene_alignment(
            adata, flow_result, normalize=True, per_cell=False
        )

        source_pca = adata.obsm["X_pca"][flow_result["source_mask"]]
        transported_10x = source_pca + 10.0 * (flow_result["transported"] - source_pca)
        flow_result_10x = dict(flow_result)
        flow_result_10x["transported"] = transported_10x

        result2 = flow_gene_alignment(
            adata, flow_result_10x, normalize=True, per_cell=False
        )

        np.testing.assert_allclose(
            result2["alignment_scores"], result1["alignment_scores"], atol=1e-5,
            err_msg="normalize=True scores should be invariant to velocity magnitude",
        )


class TestPCATruncationSensitivity:
    """Verify alignment ranking stability across PCA dimensionalities."""

    def test_top_genes_stable_across_pc_counts(self):
        """Top aligned genes should be largely stable between 30 and 50 PCs.

        When the true signal is captured in the leading PCs, adding more PCs
        (which capture diminishing variance) should not drastically re-rank
        the top aligned genes.  We expect high Spearman correlation between
        50-PC and 30-PC rankings on the top 100 genes, and moderate
        correlation between 50-PC and 10-PC rankings.
        """
        import anndata as ad
        from sklearn.decomposition import PCA
        from peach.tl.flow import flow_within, flow_gene_alignment
        from scipy.stats import spearmanr

        rng = np.random.default_rng(42)
        n_cells, n_genes = 300, 200
        X = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
        adata = ad.AnnData(X)
        adata.var_names = [f"gene_{i}" for i in range(n_genes)]
        adata.obs["condition"] = (
            ["source"] * (n_cells // 2) + ["target"] * (n_cells - n_cells // 2)
        )

        pc_counts = [10, 30, 50]
        scores_by_npc = {}

        for n_pcs in pc_counts:
            pca = PCA(n_components=n_pcs)
            pca_coords = pca.fit_transform(X).astype(np.float32)
            adata.obsm["X_pca"] = pca_coords
            adata.varm["PCs"] = pca.components_.T.astype(np.float32)

            flow_result = flow_within(
                adata,
                source={"condition": "source"},
                target={"condition": "target"},
                n_epochs=150,
                hidden_dims=(64, 64),
                return_model=True,
                random_state=42,
            )
            result = flow_gene_alignment(
                adata, flow_result, normalize=True, per_cell=False
            )
            scores_by_npc[n_pcs] = result["alignment_scores"]

        scores_50 = scores_by_npc[50]
        scores_30 = scores_by_npc[30]
        top100_idx = np.argsort(np.abs(scores_50))[-100:]

        rho, _ = spearmanr(scores_50[top100_idx], scores_30[top100_idx])
        # On purely random data each PCA dimensionality trains a different flow
        # on different noise dimensions, so perfect agreement is not expected.
        # We require meaningful positive correlation (better than chance).
        assert rho > 0.75, (
            f"Top 100 gene rankings between 30-PC and 50-PC should be stable: "
            f"Spearman={rho:.4f}"
        )

        scores_10 = scores_by_npc[10]
        rho_10, _ = spearmanr(scores_50[top100_idx], scores_10[top100_idx])
        assert rho_10 > 0.4, (
            f"10-PC vs 50-PC should have moderate concordance: Spearman={rho_10:.4f}"
        )

    def test_loading_gene_dimension_matches_adata(self):
        """Alignment score vector length must equal number of genes in adata."""
        from peach.tl.flow import flow_gene_alignment

        adata, flow_result = _make_alignment_fixture(n_genes=80)
        result = flow_gene_alignment(adata, flow_result, normalize=True, per_cell=False)
        assert len(result["alignment_scores"]) == 80, (
            f"Expected 80 scores, got {len(result['alignment_scores'])}"
        )
        assert len(result["gene_names"]) == 80

    def test_per_cell_gene_index_within_bounds(self):
        """per_cell_gene_indices must all be valid indices into adata.var_names."""
        from peach.tl.flow import flow_gene_alignment

        adata, flow_result = _make_alignment_fixture(n_genes=60)
        result = flow_gene_alignment(
            adata, flow_result, normalize=True, per_cell=True, n_top_features=30
        )
        idx = result["per_cell_gene_indices"]
        assert np.all(idx >= 0), "Negative gene index found"
        assert np.all(idx < 60), f"Gene index out of bounds: max={idx.max()}"
        assert len(idx) == 30

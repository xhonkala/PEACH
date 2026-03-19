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


@pytest.mark.slow
class TestBiologicalValidationHSC:
    """Validate alignment on real HSC CMP->Mono transition.

    Requires hsc_10k.h5ad in ~/Desktop/peach/data/.
    Run with: pytest -m slow
    """

    @pytest.fixture(scope="class")
    def hsc_flow(self):
        """Load HSC data, ensure PCA, train CMP->Mono flow."""
        import os
        import peach as pc

        data_path = os.path.expanduser("~/Desktop/peach/data/hsc_10k.h5ad")
        if not os.path.exists(data_path):
            pytest.skip(f"HSC data not found at {data_path}")

        adata = pc.pp.load_data(data_path)

        # Ensure PCA with loadings exists (need varm['PCs'] for alignment)
        if "PCs" not in adata.varm:
            import scanpy as sc
            if "log1p" not in adata.uns.get("log1p", {}):
                sc.pp.normalize_total(adata)
                sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
            sc.pp.pca(adata, n_comps=50)  # stores X_pca + PCs in varm

        # Find cell type column
        celltype_col = None
        for col in ["cell_type", "celltype", "CellType", "label"]:
            if col in adata.obs.columns:
                celltype_col = col
                break
        if celltype_col is None:
            pytest.skip("No cell type column found in HSC data")

        cell_types = adata.obs[celltype_col].unique()
        # Match CMP/Mono by abbreviation or full ontology name
        cmp_patterns = ["CMP", "common myeloid progenitor"]
        mono_patterns = ["Mono", "monocyte"]
        cmp_type = next(
            (ct for ct in cell_types
             if any(p.lower() in str(ct).lower() for p in cmp_patterns)),
            None,
        )
        mono_type = next(
            (ct for ct in cell_types
             if any(p.lower() in str(ct).lower() for p in mono_patterns)),
            None,
        )
        if cmp_type is None or mono_type is None:
            pytest.skip(f"Need CMP and Mono cell types, found: {list(cell_types)}")

        flow_result = pc.tl.flow_within(
            adata,
            source={celltype_col: cmp_type},
            target={celltype_col: mono_type},
            n_epochs=500,
            hidden_dims=(128, 128, 128),
            return_model=True,
            random_state=42,
        )

        return adata, flow_result

    @staticmethod
    def _resolve_gene_names(adata, gene_ids):
        """Map var_names (possibly Ensembl IDs) to gene symbols if available."""
        # Try common gene symbol columns
        for col in ["feature_name", "gene_symbols", "gene_name", "symbol"]:
            if col in adata.var.columns:
                id_to_sym = dict(zip(adata.var_names, adata.var[col]))
                return [id_to_sym.get(g, g) for g in gene_ids]
        return list(gene_ids)  # already symbols

    def test_myeloid_tfs_top_aligned(self, hsc_flow):
        """Canonical myeloid TFs should appear in top 100 aligned genes."""
        from peach.tl.flow import flow_gene_alignment

        adata, flow_result = hsc_flow
        result = flow_gene_alignment(
            adata, flow_result, normalize=True, per_cell=False, n_top=100
        )

        top_aligned_symbols = set(self._resolve_gene_names(adata, result["top_aligned"]))
        all_symbols = set(self._resolve_gene_names(adata, adata.var_names))
        # Monocyte effector genes + TFs — TFs often filtered by HVG, so include
        # effector markers that survive HVG filtering
        monocyte_markers = {
            # TFs (may be filtered by HVG)
            "SPI1", "CEBPA", "CEBPB", "IRF8",
            # Effector genes (survive HVG, canonical monocyte markers)
            "FCN1", "CD14", "S100A8", "S100A9", "S100A12", "VCAN", "MNDA",
            "CST3", "LYZ", "MS4A6A",
        }
        present = monocyte_markers & all_symbols
        found = present & top_aligned_symbols

        top10_symbols = self._resolve_gene_names(adata, result["top_aligned"][:10])
        print(f"Monocyte markers in dataset: {present}")
        print(f"Monocyte markers in top 100 aligned: {found}")
        print(f"Top 10 aligned: {top10_symbols}")

        assert len(found) >= 3, (
            f"Expected >= 3 monocyte markers in top 100 aligned, found {len(found)}: {found}. "
            f"Present in data: {present}. Top 10: {top10_symbols}"
        )

    def test_progenitor_markers_top_opposed(self, hsc_flow):
        """HSC/progenitor markers should appear in top 100 opposed genes.

        CMP→Mono flow: opposed direction = progenitor/stem state, NOT erythroid
        (erythroid is a different lineage branch).
        """
        from peach.tl.flow import flow_gene_alignment

        adata, flow_result = hsc_flow
        result = flow_gene_alignment(
            adata, flow_result, normalize=True, per_cell=False, n_top=100
        )

        top_opposed_symbols = set(self._resolve_gene_names(adata, result["top_opposed"]))
        all_symbols = set(self._resolve_gene_names(adata, adata.var_names))
        # HSC/progenitor markers expected in the opposed (source) direction
        progenitor_markers = {
            "SPINK2", "HOPX", "ERG", "CDK6", "CRHBP", "AVP", "BEX1",
            "PRSS57", "FSTL1", "MDK",
        }
        present = progenitor_markers & all_symbols
        found = present & top_opposed_symbols

        top10_symbols = self._resolve_gene_names(adata, result["top_opposed"][:10])
        print(f"Progenitor markers in dataset: {present}")
        print(f"Progenitor markers in top 100 opposed: {found}")
        print(f"Top 10 opposed: {top10_symbols}")

        assert len(found) >= 2, (
            f"Expected >= 2 progenitor markers in top 100 opposed, found {len(found)}: {found}. "
            f"Present in data: {present}. Top 10: {top10_symbols}"
        )

    def test_jacobian_expansion_concordance(self, hsc_flow):
        """Jacobian expansion scores should positively correlate with alignment."""
        from peach.tl.flow import flow_gene_alignment, flow_jacobian
        from scipy.stats import spearmanr

        adata, flow_result = hsc_flow
        model = flow_result["model"]

        align_result = flow_gene_alignment(
            adata, flow_result, normalize=True, per_cell=False
        )
        jac_result = flow_jacobian(
            adata, flow_result, model, per_cell_features=False
        )

        alignment = align_result["alignment_scores"]
        expansion = jac_result["feature_expansion"]

        if len(expansion) == 0:
            pytest.skip("No feature expansion (PCA loadings missing)")

        rho, pval = spearmanr(alignment, expansion)
        print(f"Alignment vs expansion Spearman: rho={rho:.4f}, p={pval:.2e}")

        assert rho > 0, (
            f"Alignment and Jacobian expansion should be positively correlated: "
            f"Spearman={rho:.4f}, p={pval:.2e}"
        )

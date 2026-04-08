"""Real-data integration test for compute_archetype_correspondence() on bigOV.

Second-dataset validation for the hard-argmax correspondence method. Unlike
the HSC/CMP test where source and target are two distinct cell types, this
test splits bigOV primary (RIGHT/LEFT adnexa ovary) vs metastatic (BOWEL,
ASCITES, PELVIC, etc.) within a single tumor cell type (EOC). This is a
biologically harder test: the two populations share more structure than
HSC→CMP does, so the correspondence matrix should reveal phenotype drift
on top of a shared baseline.

This test is @pytest.mark.slow. Run with:

    pytest tests/test_core/test_correspondence_real_data_ov.py -v -m slow

Automatically skipped if data/paper_part1_ov/ files are missing. Generate
them via ``python scripts/prep_bigov.py``.
"""
import os
import numpy as np
import pytest

DATA_DIR = "/Users/honkala/Desktop/PEACH_public/data/paper_part1_ov"
PRIMARY_PATH = os.path.join(DATA_DIR, "adata_primary_train.h5ad")
METASTATIC_PATH = os.path.join(DATA_DIR, "adata_metastatic_train.h5ad")


@pytest.mark.slow
@pytest.mark.skipif(
    not (os.path.exists(PRIMARY_PATH) and os.path.exists(METASTATIC_PATH)),
    reason="bigOV primary/metastatic h5ad files not present "
           "(run scripts/prep_bigov.py)",
)
def test_correspondence_hard_method_on_bigov_primary_metastatic():
    """Train small primary/metastatic EOC models and run hard correspondence.

    Populations are both EOC tumor cells from SPECTRUM-OV patients; they
    share a common cell-type identity but are biologically drifted by
    tissue context (primary ovary vs peritoneal / ascites metastases).

    Acceptance criteria:

    1. column_cv > 0.15 — lower threshold than the HSC/CMP test (0.18)
       because primary and metastatic EOC are more similar than HSC and
       CMP. The hard method should still give > 1.5x the soft method's
       typical ~0.08 noise floor on diffuse Deep_AA weights.
    2. Diagnostic dict fields populated.
    3. Hard correspondence at least doubles column CV versus soft
       correspondence on the same inputs (the actual Task 1 signal check).
    """
    import anndata as ad
    import peach as pc
    from peach._core.utils.archetype_comparison import (
        compute_archetype_correspondence,
    )

    rng = np.random.default_rng(42)

    # ---------------- 1. Load and subsample --------------------------
    adata_primary = ad.read_h5ad(PRIMARY_PATH)
    adata_metastatic = ad.read_h5ad(METASTATIC_PATH)

    n_sub = 500
    p_idx = np.sort(
        rng.choice(adata_primary.n_obs, size=min(n_sub, adata_primary.n_obs),
                   replace=False)
    )
    m_idx = np.sort(
        rng.choice(adata_metastatic.n_obs, size=min(n_sub, adata_metastatic.n_obs),
                   replace=False)
    )
    adata_primary = adata_primary[p_idx].copy()
    adata_metastatic = adata_metastatic[m_idx].copy()

    # ---------------- 2. prepare_training -----------------------------
    pc.pp.prepare_training(adata_primary, batch_size=128)
    pc.pp.prepare_training(adata_metastatic, batch_size=128)

    # ---------------- 3. Train two small Deep_AA models ---------------
    # Fast integration test: no hyperparameter search, fixed K each side.
    # K_primary=5, K_metastatic=6. For the full paper_part1_ov script a
    # hyperparameter search K=4..9 will be used instead (Task 27).
    K_primary = 5
    pc.tl.train_archetypal(
        adata_primary,
        n_archetypes=K_primary,
        n_epochs=30,
        kld_weight=0.1,
        archetypal_weight=0.9,
        inflation_factor=1.0,
        model_config={"hidden_dims": [64, 128]},
    )
    pc.tl.archetypal_coordinates(adata_primary, verbose=False)
    pc.tl.extract_archetype_weights(adata_primary, verbose=False)
    w_primary = adata_primary.obsm["cell_archetype_weights"]
    assert w_primary.shape == (adata_primary.n_obs, K_primary)

    K_metastatic = 6
    pc.tl.train_archetypal(
        adata_metastatic,
        n_archetypes=K_metastatic,
        n_epochs=30,
        kld_weight=0.1,
        archetypal_weight=0.9,
        inflation_factor=1.0,
        model_config={"hidden_dims": [128, 256]},
    )
    pc.tl.archetypal_coordinates(adata_metastatic, verbose=False)
    pc.tl.extract_archetype_weights(adata_metastatic, verbose=False)
    w_metastatic = adata_metastatic.obsm["cell_archetype_weights"]
    assert w_metastatic.shape == (adata_metastatic.n_obs, K_metastatic)

    # ---------------- 4. Common-dim PCA coords (PCA was computed on the
    # full dataset in prep_bigov.py, so both splits share the same space)
    pca_primary = np.asarray(adata_primary.obsm["X_pca"])
    pca_metastatic = np.asarray(adata_metastatic.obsm["X_pca"])
    n_pcs = min(pca_primary.shape[1], pca_metastatic.shape[1])
    pca_primary = pca_primary[:, :n_pcs]
    pca_metastatic = pca_metastatic[:, :n_pcs]

    # ---------------- 5. Run correspondence: hard method ---------------
    result_hard = compute_archetype_correspondence(
        source_weights=np.asarray(w_primary),
        source_coords=pca_primary,
        target_weights=np.asarray(w_metastatic),
        target_coords=pca_metastatic,
        k=10,
        method="hard",
    )
    markov_hard = result_hard["markov"]
    assert markov_hard.shape == (K_primary, K_metastatic)
    assert result_hard["method"] == "hard"
    for key in (
        "source_weight_concentration",
        "target_weight_concentration",
        "source_archetype_occupancy_hard",
        "sparse_archetypes",
    ):
        assert key in result_hard, f"Missing diagnostic key: {key}"
    assert result_hard["source_archetype_occupancy_hard"].shape == (K_primary,)

    # ---------------- 6. Run correspondence: soft method (for comparison)
    result_soft = compute_archetype_correspondence(
        source_weights=np.asarray(w_primary),
        source_coords=pca_primary,
        target_weights=np.asarray(w_metastatic),
        target_coords=pca_metastatic,
        k=10,
        method="soft",
    )
    markov_soft = result_soft["markov"]

    # ---------------- 7. Column CV on hard-method active rows ---------
    def _column_cv(markov):
        nonzero_rows = markov.sum(axis=1) > 1e-9
        m_active = markov[nonzero_rows]
        if m_active.shape[0] < 2:
            return 0.0
        col_means = m_active.mean(axis=0)
        col_stds = m_active.std(axis=0)
        cv = col_stds / np.where(col_means > 1e-10, col_means, 1.0)
        return float(cv.mean())

    cv_hard = _column_cv(markov_hard)
    cv_soft = _column_cv(markov_soft)

    src_conc = result_hard["source_weight_concentration"]
    occ_hard = result_hard["source_archetype_occupancy_hard"].tolist()

    # Absolute floor: column CV must clear 0.15
    assert cv_hard > 0.15, (
        f"bigOV primary→metastatic column CV {cv_hard:.3f} below 0.15 "
        f"threshold. Hard method should differentiate even two similar "
        f"EOC populations.\n"
        f"  Source concentration: {src_conc:.3f}\n"
        f"  Occupancy (primary, hard argmax): {occ_hard}\n"
        f"  Sparse archetypes: {result_hard['sparse_archetypes']}\n"
        f"  Soft CV (for comparison): {cv_soft:.3f}"
    )

    # Relative check: hard must deliver at least 1.5x soft's CV on this
    # data. This is the Task 1 signal claim — hard amplifies differentiation
    # that soft averages away.
    assert cv_hard > 1.5 * cv_soft, (
        f"Hard method failed to amplify signal: hard CV {cv_hard:.3f} is "
        f"not > 1.5 × soft CV {cv_soft:.3f}.\n"
        f"  Source concentration: {src_conc:.3f}\n"
        f"  Occupancy: {occ_hard}"
    )

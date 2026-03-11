"""v0.5.0 Reviewer-Grade E2E: HSC CMP vs CD14+ Monocyte.

Full pipeline with statistical controls, interleaved visualizations,
HALLMARK pathway scoring, and flow-aligned gene/geneset analysis.

Usage:
    /Users/honkala/miniconda3/envs/archetype/bin/python tests/test_e2e_hsc_v12.py
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import warnings
import logging
import numpy as np
import scipy.sparse as sp
import pandas as pd
from collections import Counter

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.disable(logging.WARNING)

import torch
torch.set_num_threads(1)

# --- Config -----------------------------------------------------------------
HSC_PATH = "/Users/honkala/Desktop/cross_recons/data/HSC.h5ad"
K = 4
N_PCS = 20
MONO_SUBSAMPLE = 9000
TRAIN_EPOCHS = 150
FLOW_EPOCHS = 300
SEED = 42
OUTPUT_DIR = "/Users/honkala/Desktop/PEACH_public/tests/e2e_outputs_v12"
N_BOOTSTRAP = 200       # 1000 for final notebook; 200 for dev iteration (~20min/celltype)
N_PERMUTATIONS = 200    # 1000 for final notebook; 200 for dev iteration
ALPHA = 0.05


def section(title):
    print(f"\n{'='*60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*60}", flush=True)


def step(msg):
    print(f"  -> {msg}", end="", flush=True)


def done(extra=""):
    print(f" OK {extra}", flush=True)


def report(lines):
    """Print indented report lines for reviewer audit trail."""
    for line in lines:
        print(f"      {line}", flush=True)


def save_viz(fig, filename):
    """Save plotly figure to HTML."""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.write_html(path)
    return path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()

    # === 1. Data Loading & Gene Symbol Mapping ==============================
    section("1. Data Loading & Gene Symbol Mapping")

    step("Loading HSC dataset")
    import anndata as ad
    import gc
    adata_full = ad.read_h5ad(HSC_PATH)
    done(f"({adata_full.shape[0]:,} cells x {adata_full.shape[1]:,} genes)")

    # Capture original gene count before any manipulation
    n_genes_original = adata_full.n_vars

    # Swap ENSG -> gene symbols
    step("Mapping ENSG IDs to gene symbols")

    # Clean gene symbols: handle NaN, empty, whitespace
    symbols_raw = list(adata_full.var["gene_symbols"].values)
    symbols = []
    for i, s in enumerate(symbols_raw):
        if pd.isna(s) or str(s).strip() == "":
            symbols.append(f"UNNAMED_{i}")
        else:
            symbols.append(str(s).strip())
    n_unnamed = sum(1 for s in symbols if s.startswith("UNNAMED_"))

    counts = Counter(symbols)
    n_duplicates = sum(1 for s, c in counts.items() if c > 1)
    seen = Counter()
    unique_symbols = []
    for s in symbols:
        if counts[s] > 1:
            seen[s] += 1
            unique_symbols.append(f"{s}_{seen[s]}")
        else:
            unique_symbols.append(s)
    var_names = unique_symbols
    done()
    report([
        f"Total genes: {len(var_names)}",
        f"Unique symbols: {len(set(symbols))}",
        f"Duplicate symbols (suffixed): {n_duplicates}",
        f"Unnamed/NaN symbols: {n_unnamed}",
        f"Sample mapping: {list(adata_full.var_names[:3])} -> {var_names[:3]}",
    ])

    # Assertions: gene symbol integrity
    assert len(var_names) == n_genes_original, f"Gene count mismatch: {len(var_names)} vs {n_genes_original}"
    assert len(var_names) == len(set(var_names)), f"var_names not unique after dedup"

    # Extract PCA loadings if available (needed for flow gene alignment later)
    step("Extracting PCA loadings")
    if "PCs" in adata_full.varm:
        pca_loadings = adata_full.varm["PCs"][:, :N_PCS].copy()
        done(f"(shape: {pca_loadings.shape})")
    else:
        pca_loadings = None
        done("(not available in varm)")

    # Subset CMP and Mono populations
    rng = np.random.default_rng(SEED)
    cell_data = {}
    for name, ct, subsample in [
        ("CMP", "common myeloid progenitor", None),
        ("Mono", "CD14-positive monocyte", MONO_SUBSAMPLE),
    ]:
        step(f"Subsetting {name}")
        mask = adata_full.obs["cell_type"] == ct
        idx = np.where(mask)[0]
        n_total = len(idx)
        if subsample and len(idx) > subsample:
            idx = rng.choice(idx, size=subsample, replace=False)
            idx.sort()
        pca = adata_full.obsm["X_pca"][idx, :N_PCS].copy()
        obs = adata_full.obs.iloc[idx].copy().reset_index(drop=True)
        cell_data[name] = {"pca": pca, "obs": obs, "idx": idx}
        sub_note = f" (subsampled from {n_total:,})" if subsample else ""
        done(f"({len(idx):,} cells{sub_note}, PCA: {pca.shape})")

    # Compute PCA loadings if not in dataset (must happen before del adata_full)
    if pca_loadings is None:
        step("Computing PCA loadings (not in dataset)")
        from sklearn.decomposition import PCA
        # Use CMP subset as representative (faster than full dataset)
        X_for_pca = adata_full.X[cell_data["CMP"]["idx"]]
        if hasattr(X_for_pca, "toarray"):
            X_for_pca = X_for_pca.toarray()
        pca_model = PCA(n_components=N_PCS, random_state=SEED)
        pca_model.fit(X_for_pca)
        pca_loadings = pca_model.components_.T  # [n_genes, N_PCS]
        del X_for_pca
        done(f"(shape: {pca_loadings.shape})")

    del adata_full
    gc.collect()

    # === 2. Archetype Fitting ================================================
    section(f"2. Archetype Fitting (K={K})")
    import peach as pc

    models = {}
    for name in ["CMP", "Mono"]:
        step(f"Training K={K} on {name}")
        t1 = time.time()

        # Lightweight AnnData for training (avoids OMP crash with large sparse X)
        adata_train = ad.AnnData(
            X=sp.csr_matrix((len(cell_data[name]["pca"]), 1)),
            obs=cell_data[name]["obs"].copy(),
        )
        adata_train.obsm["X_pca"] = cell_data[name]["pca"]

        result = pc.tl.train_archetypal(
            adata_train, n_archetypes=K,
            n_epochs=TRAIN_EPOCHS, kld_weight=0.1, archetypal_weight=0.9,
            seed=SEED,
        )
        elapsed = time.time() - t1
        r2 = result.get("final_archetype_r2", None)
        r2_str = f"R2={r2:.3f}" if r2 is not None else "R2=N/A"
        done(f"({r2_str}, {elapsed:.1f}s)")

        # Workflow 04 steps: weights, coordinates, assignment
        pc.tl.extract_archetype_weights(adata_train)
        pc.tl.archetypal_coordinates(adata_train, verbose=False)

        weights = adata_train.obsm["cell_archetype_weights"]
        cell_data[name]["weights"] = weights
        cell_data[name]["uns"] = dict(adata_train.uns)
        cell_data[name]["arch_dist"] = adata_train.obsm["archetype_distances"]
        models[name] = result

        # Training diagnostics (positive controls)
        report([
            f"TRAINING DIAGNOSTICS:",
            f"  Final R2: {r2:.4f} (expect >0.5 for meaningful structure)" if r2 is not None else "  Final R2: N/A",
            f"  Constraint satisfaction: {result.get('history', {}).get('constraints_satisfied', ['N/A'])[-1]}",
            f"  Weight sum check: mean={weights.sum(axis=1).mean():.6f} (expect 1.0)",
            f"  Weight range: [{weights.min():.4f}, {weights.max():.4f}]",
            f"  Dominant cells per archetype:",
        ])
        for k in range(K):
            dominant = np.sum(weights.argmax(axis=1) == k)
            report([f"    Archetype {k}: {dominant} cells ({100*dominant/len(weights):.1f}%)"])

        # Assertions: training quality
        assert r2 is not None, "R2 is None — training may have failed"
        assert r2 > 0.0, f"R2 is non-positive ({r2:.4f}) — degenerate model"
        assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-4), \
            f"Weight sums deviate: max error {np.abs(weights.sum(axis=1) - 1.0).max()}"
        for k_idx in range(K):
            n_dominant = int(np.sum(weights.argmax(axis=1) == k_idx))
            assert n_dominant > 0, f"Archetype {k_idx} has zero dominant cells — degenerate"

    # Reload sparse X with gene symbol var_names for downstream regression
    step("Reloading sparse X for regression")
    adata_disk = ad.read_h5ad(HSC_PATH, backed="r")
    for name in ["CMP", "Mono"]:
        idx = cell_data[name]["idx"]
        X_sparse = sp.csr_matrix(adata_disk.X[idx])
        var_df = pd.DataFrame(index=var_names)

        adata_obj = ad.AnnData(X=X_sparse, obs=cell_data[name]["obs"].copy(), var=var_df)
        adata_obj.obsm["X_pca"] = cell_data[name]["pca"]
        adata_obj.obsm["cell_archetype_weights"] = cell_data[name]["weights"]
        adata_obj.obsm["archetype_distances"] = cell_data[name]["arch_dist"]
        for k, v in cell_data[name]["uns"].items():
            adata_obj.uns[k] = v
        # Store PCA loadings for flow gene alignment later
        if pca_loadings is not None:
            adata_obj.varm["PCs"] = pca_loadings
        pc.tl.assign_archetypes(adata_obj)

        # Assertions: sparse X rebuild
        assert adata_obj.shape == (len(cell_data[name]["idx"]), len(var_names)), \
            f"Shape mismatch after rebuild: {adata_obj.shape}"

        # Assertions: archetype assignment
        assert "archetypes" in adata_obj.obs.columns, "assign_archetypes did not create obs['archetypes']"
        n_assigned = int((adata_obj.obs["archetypes"] != "no_archetype").sum())
        assert n_assigned > 0, "All cells assigned to no_archetype — assignment failed"

        cell_data[name]["adata"] = adata_obj
    del adata_disk
    gc.collect()
    done()

    adata_cmp = cell_data["CMP"]["adata"]
    adata_mono = cell_data["Mono"]["adata"]

    report([
        f"CMP adata: {adata_cmp.shape}, var_names sample: {list(adata_cmp.var_names[:3])}",
        f"Mono adata: {adata_mono.shape}, var_names sample: {list(adata_mono.var_names[:3])}",
        f"PCA loadings stored: {pca_loadings is not None}",
    ])

    # === 3. Simplex Regression (Full Statistical Controls) ====================
    section("3. Simplex Regression")

    gene_regs = {}  # Save gene regression dicts (overwritten by pathway reg in Section 7)
    for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
        step(f"Feature simplex regression on {name} (bootstrap={N_BOOTSTRAP}, perm={N_PERMUTATIONS})")
        t1 = time.time()
        reg = pc.tl.feature_simplex_regression(
            ad_obj, max_degree=2,
            n_bootstrap=N_BOOTSTRAP,
            permutation_test=True,
            n_permutations=N_PERMUTATIONS,
            robust_se=True,
        )
        elapsed = time.time() - t1
        done(f"({elapsed:.1f}s)")
        gene_regs[name] = reg  # Save before pathway regression overwrites uns

        r2_d1 = np.asarray(reg["r_squared_degree1"])
        r2_d2 = np.asarray(reg["r_squared_degree2"])
        f_pval = np.asarray(reg["f_pvalue"])
        f_pval_fdr = np.asarray(reg["f_pvalue_fdr"])
        perm_pval = reg.get("permutation_pvalue")
        perm_pval_fdr = reg.get("permutation_pvalue_fdr")
        if perm_pval is not None:
            perm_pval = np.asarray(perm_pval)
        if perm_pval_fdr is not None:
            perm_pval_fdr = np.asarray(perm_pval_fdr)

        # Positive control: R2 distribution
        n_sig_f = int(np.sum(f_pval_fdr < ALPHA))
        n_sig_perm = int(np.sum(perm_pval_fdr < ALPHA)) if perm_pval_fdr is not None else "N/A"

        report([
            f"REGRESSION SUMMARY ({name}):",
            f"  Features tested: {len(r2_d1)}",
            f"  Degree-1 R2: median={np.median(r2_d1):.4f}, mean={np.mean(r2_d1):.4f}, max={np.max(r2_d1):.4f}",
            f"  Degree-2 R2: median={np.median(r2_d2):.4f}, mean={np.mean(r2_d2):.4f}, max={np.max(r2_d2):.4f}",
            f"  R2 improvement (d2-d1): median={np.median(r2_d2 - r2_d1):.4f}",
            f"",
            f"STATISTICAL CONTROLS:",
            f"  F-test significant (FDR<{ALPHA}): {n_sig_f}/{len(r2_d1)} ({100*n_sig_f/len(r2_d1):.1f}%)",
            f"  Permutation test significant (FDR<{ALPHA}): {n_sig_perm}",
            f"  Negative control: median F-test p-value = {np.median(f_pval):.4f}",
            f"    (expect ~0.5 for features with no archetype dependence)",
            f"",
            f"TOP 10 GENES (by R2):",
        ])
        top_idx = np.argsort(r2_d1)[-10:][::-1]
        for i in top_idx:
            gene = reg["feature_names"][i]
            ci_lo = reg.get("vertex_ci_lower")
            ci_hi = reg.get("vertex_ci_upper")
            ci_str = ""
            if ci_lo is not None and ci_hi is not None:
                ci_lo_arr = np.asarray(ci_lo)
                ci_hi_arr = np.asarray(ci_hi)
                # Mean CI width across archetypes for this feature
                ci_widths = ci_hi_arr[i] - ci_lo_arr[i]
                ci_str = f", mean_CI_width={np.mean(ci_widths):.3f}"
            perm_str = ""
            if perm_pval is not None:
                perm_str = f", perm_p={perm_pval[i]:.4f}"
            report([f"  {gene}: R2={r2_d1[i]:.4f}, F_p={f_pval[i]:.2e}{perm_str}{ci_str}"])

        # Assertions: regression quality
        assert len(r2_d1) == ad_obj.n_vars, f"R2 length mismatch: {len(r2_d1)} vs {ad_obj.n_vars}"
        # R2 can be slightly negative for OLS when model is worse than mean
        assert np.all(r2_d1 >= -0.1), f"R2 deeply negative (d1): min={r2_d1.min():.4f}"
        assert np.all(r2_d1 <= 1.0 + 1e-6), "R2 (d1) > 1 found"
        # F2: Degree-2 R2 bounds (catch NaN/Inf from interaction terms)
        assert np.all(np.isfinite(r2_d2)), f"Non-finite R2 (d2) found: {np.sum(~np.isfinite(r2_d2))} features"
        assert np.all(r2_d2 >= -0.1), f"R2 deeply negative (d2): min={r2_d2.min():.4f}"
        assert np.all(r2_d2 <= 1.0 + 1e-6), "R2 (d2) > 1 found"
        # F3: Nested model invariant — degree-2 must weakly improve on degree-1
        r2_diff = r2_d2 - r2_d1
        n_violations = int(np.sum(r2_diff < -1e-6))
        if n_violations > 0:
            worst = r2_diff.min()
            report([f"  WARNING: {n_violations} features where d2 R2 < d1 R2 (worst: {worst:.6f})"])
        assert n_violations == 0, \
            f"Nested model invariant violated: {n_violations} features have d2 R2 < d1 R2 (min diff: {r2_diff.min():.6f})"
        n_neg_r2 = int(np.sum(r2_d1 < 0))
        if n_neg_r2 > 0:
            report([f"  NOTE: {n_neg_r2} features with negative R2 (model worse than mean)"])
        assert n_sig_f > 0, "Zero F-test significant genes — regression may be degenerate"
        # F6: F-test / permutation concordance
        if perm_pval_fdr is not None:
            assert isinstance(n_sig_perm, int), "Permutation test produced no integer count"
            assert n_sig_perm > 0, "Permutation test found zero significant genes despite F-test finding some"
            concordance = min(n_sig_f, n_sig_perm) / max(n_sig_f, n_sig_perm) if max(n_sig_f, n_sig_perm) > 0 else 0
            report([
                f"  F-test vs permutation concordance: {concordance:.2f}",
                f"    F-test sig: {n_sig_f}, Perm sig: {n_sig_perm}",
                f"    (expect rough agreement; low concordance suggests model mis-specification)",
            ])

        # Interleaved visualizations
        step(f"  Coefficient heatmap ({name})")
        pc.pl.coefficient_heatmap(
            ad_obj, top_n=30, show=False,
            save_path=os.path.join(OUTPUT_DIR, f"coef_heatmap_{name}.html"),
        )
        done()

        step(f"  R2 barplot ({name})")
        pc.pl.r2_barplot(
            ad_obj, top_n=30, show=False,
            save_path=os.path.join(OUTPUT_DIR, f"r2_barplot_{name}.html"),
        )
        done()

        step(f"  Regression volcano ({name})")
        pc.pl.regression_volcano(
            ad_obj, show=False,
            save_path=os.path.join(OUTPUT_DIR, f"regression_volcano_{name}.html"),
        )
        done()

    # === 4. Pattern Classification =============================================
    section("4. Pattern Classification")

    for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
        step(f"Classifying feature patterns for {name}")
        patterns = pc.tl.classify_feature_patterns(ad_obj)
        counts = patterns["pattern_counts"]
        n_total = patterns["n_features"]
        done()

        report([
            f"PATTERN CLASSIFICATION ({name}):",
            f"  Total features classified: {n_total}",
        ])
        for ptype, pcount in sorted(counts.items(), key=lambda x: -x[1]):
            pct = 100 * pcount / n_total
            report([f"  {ptype}: {pcount} ({pct:.1f}%)"])

        # Negative control: expect mostly flat patterns
        n_flat = counts.get("flat", 0)
        flat_pct = 100 * n_flat / n_total
        report([
            f"",
            f"CONTROL: {flat_pct:.1f}% flat patterns",
            f"  (expect >80% — most genes should NOT be archetype-dependent)",
            f"  Non-flat patterns: {n_total - n_flat} ({100 - flat_pct:.1f}%)",
        ])

        # Assertions: pattern classification
        assert n_total == ad_obj.n_vars, f"Pattern count mismatch: {n_total} vs {ad_obj.n_vars}"
        assert sum(counts.values()) == n_total, "Pattern counts don't sum to n_features"
        assert n_flat > 0, "Zero flat patterns — unexpected for genome-wide data"

    # === 5. GMM Simplex Decomposition ========================================
    section("5. GMM Simplex Decomposition")

    for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
        step(f"Fitting GMM on {name}")
        t1 = time.time()
        gmm = pc.tl.feature_simplex_decomposition(ad_obj, characterize_features=True)
        elapsed = time.time() - t1
        done(f"({elapsed:.1f}s)")

        n_opt = gmm["n_components_optimal"]
        n_stab = gmm["n_components_stable"]
        report([
            f"GMM RESULTS ({name}):",
            f"  Optimal components (BIC): {n_opt}",
            f"  Stable components: {n_stab}",
            f"  Stability scores: {gmm['component_stability_scores']}",
            f"  Archetype map: {gmm['component_archetype_map']}",
        ])

        # Assertions: GMM sanity
        assert n_opt >= K, f"Optimal components ({n_opt}) < K ({K}) — shouldn't happen"
        assert n_stab >= 1, f"Zero stable components — GMM failed"
        assert "component_assignments" in gmm, "Missing component_assignments"
        labels = np.asarray(gmm["component_assignments"])
        assert len(labels) == ad_obj.n_obs, f"Label count mismatch: {len(labels)} vs {ad_obj.n_obs}"

    # GMM visualizations on CMP (representative)
    step("GMM visualizations (CMP)")
    for viz_name, viz_fn, filename in [
        ("component_scatter", lambda: pc.pl.component_scatter(adata_cmp, show=False), "gmm_scatter_CMP.html"),
        ("bic_curve", lambda: pc.pl.gmm_bic_curve(adata_cmp, show=False), "gmm_bic_CMP.html"),
        ("component_heatmap", lambda: pc.pl.component_heatmap(adata_cmp, top_n=30, show=False), "gmm_heatmap_CMP.html"),
        ("component_stability", lambda: pc.pl.component_stability(adata_cmp, show=False), "gmm_stability_CMP.html"),
    ]:
        fig = viz_fn()
        if fig is not None:
            save_viz(fig, filename)
    done()

    # === 6. Archetype Comparison ==============================================
    section("6. Archetype Comparison")

    # 6a. Within-fit MMD
    for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
        step(f"Within-fit MMD for {name}")
        mmd_result = pc.tl.archetype_mmd(ad_obj, n_permutations=100)
        mask = ~np.eye(K, dtype=bool)
        mmd_vals = mmd_result["mmd_matrix"][mask]
        pvals = mmd_result["pvalue_matrix"][mask]
        done()
        report([
            f"WITHIN-FIT MMD ({name}):",
            f"  MMD range: {mmd_vals.min():.4f} - {mmd_vals.max():.4f}",
            f"  p-values (off-diag): min={pvals.min():.4f}, max={pvals.max():.4f}",
            f"  Significant pairs (p<{ALPHA}): {int(np.sum(pvals < ALPHA))}/{len(pvals)}",
            f"  CONTROL: All off-diagonal pairs should be significant",
            f"    (archetypes occupy distinct regions of the simplex)",
        ])

        # Assertions: MMD sanity
        assert mmd_result["mmd_matrix"].shape == (K, K), f"MMD shape wrong: {mmd_result['mmd_matrix'].shape}"
        assert np.all(np.diag(mmd_result["mmd_matrix"]) == 0), "Diagonal MMD should be 0"
        assert np.all(mmd_vals >= 0), "Negative MMD values"

        fig = pc.pl.mmd_heatmap(ad_obj, show=False)
        save_viz(fig, f"mmd_heatmap_{name}.html")

    # 6b. Between-fit MMD
    step("Between-fit MMD (CMP vs Mono)")
    mmd_between = pc.tl.archetype_mmd(adata_cmp, adata_mono, n_permutations=100)
    done()
    report([
        f"BETWEEN-FIT MMD (CMP vs Mono):",
        f"  Matrix shape: {np.asarray(mmd_between['mmd_matrix']).shape}",
        f"  MMD matrix:",
        f"  {np.array2string(np.asarray(mmd_between['mmd_matrix']), precision=4)}",
        f"  CONTROL: Between-fit MMD should be larger than within-fit MMD",
        f"    (different cell types should occupy different simplex regions)",
    ])

    # 6c. Feature similarity
    for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
        step(f"Feature similarity for {name}")
        sim = pc.tl.archetype_feature_similarity(ad_obj)
        done()
        report([
            f"FEATURE SIMILARITY ({name}):",
            f"  Overall silhouette: {sim['silhouette_overall']:.3f}",
            f"  Per-archetype: {np.array2string(np.asarray(sim['silhouette_per_archetype']), precision=3)}",
            f"  CONTROL: Positive silhouette = archetypes have distinct feature profiles",
        ])
        fig = pc.pl.feature_similarity_heatmap(ad_obj, show=False)
        save_viz(fig, f"feature_similarity_{name}.html")

    # 6d. Between-fit feature similarity
    step("Between-fit feature similarity")
    sim_between = pc.tl.archetype_feature_similarity(adata_cmp, adata_mono)
    done()
    report([
        f"BETWEEN-FIT FEATURE SIMILARITY:",
        f"  Shared features: {sim_between['n_shared_features']}",
        f"  Spearman matrix:",
        f"  {np.array2string(np.asarray(sim_between['spearman_matrix']), precision=3)}",
        f"  CONTROL: Diagonal > off-diagonal = matched archetypes share features",
    ])

    # 6e. Wald contrasts
    for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
        step(f"Wald contrasts for {name}")
        contrasts = pc.tl.archetype_contrasts(ad_obj)
        done()
        report([f"WALD CONTRASTS ({name}):"])
        for pair in contrasts["pairs"]:
            key = str(tuple(pair)) if not isinstance(pair, str) else pair
            pvals_fdr = contrasts["pvalues_fdr"][key]
            n_sig = int(np.sum(pvals_fdr < ALPHA))
            n_total_genes = len(pvals_fdr)
            report([f"  Pair {pair}: {n_sig}/{n_total_genes} significant (FDR<{ALPHA})"])
        report([
            f"  CONTROL: Each archetype pair should have many significant genes",
            f"    (archetypes represent distinct transcriptional programs)",
        ])

        # Contrast volcano for first pair
        fig = pc.pl.contrast_volcano(ad_obj, pair=(0, 1), show=False)
        save_viz(fig, f"contrast_volcano_{name}_0v1.html")

    # === 7. HALLMARK Pathway Scoring & Regression ==============================
    section("7. HALLMARK Pathway Scoring & Regression")

    step("Loading HALLMARK gene sets")
    net = pc.pp.load_pathway_networks(["hallmark"], verbose=False)
    n_pathways = net["source"].nunique()
    n_genes_in_sets = net["target"].nunique()
    done(f"({n_pathways} pathways, {n_genes_in_sets} genes)")

    pathway_regs = {}  # Save pathway regression dicts for Section 9 concordance
    for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
        step(f"Computing pathway scores for {name}")
        pc.pp.compute_pathway_scores(ad_obj, net, verbose=False)
        pathway_names = ad_obj.uns["pathway_scores_pathways"]
        scores = ad_obj.obsm["pathway_scores"]
        done(f"({scores.shape[1]} pathways scored)")

        report([
            f"PATHWAY SCORING ({name}):",
            f"  Pathways scored: {scores.shape[1]}",
            f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]",
            f"  Mean score: {scores.mean():.4f}",
            f"  Most variable pathway: {pathway_names[scores.var(axis=0).argmax()]}",
        ])

        # Assertions: pathway scoring
        assert scores.shape[0] == ad_obj.n_obs, f"Score rows mismatch: {scores.shape[0]} vs {ad_obj.n_obs}"
        assert scores.shape[1] > 0, "Zero pathways scored"
        assert np.all(np.isfinite(scores)), "Non-finite pathway scores"

        # Regress pathways on archetype weights
        step(f"Pathway simplex regression on {name}")
        t1 = time.time()
        pw_reg = pc.tl.pathway_simplex_regression(
            ad_obj, n_bootstrap=N_BOOTSTRAP, robust_se=True,
            feature_names=list(pathway_names),
        )
        elapsed = time.time() - t1
        done(f"({elapsed:.1f}s)")

        pathway_regs[name] = pw_reg
        pw_r2 = np.asarray(pw_reg["r_squared_degree1"])
        pw_pval_fdr = np.asarray(pw_reg["f_pvalue_fdr"])
        n_sig_pw = int(np.sum(pw_pval_fdr < ALPHA))
        report([
            f"PATHWAY REGRESSION ({name}):",
            f"  Pathways tested: {len(pw_r2)}",
            f"  R2: median={np.median(pw_r2):.4f}, max={np.max(pw_r2):.4f}",
            f"  Significant (F-test FDR<{ALPHA}): {n_sig_pw}/{len(pw_r2)}",
            f"",
            f"TOP 10 PATHWAYS (by R2):",
        ])
        top_pw = np.argsort(pw_r2)[-10:][::-1]
        for i in top_pw:
            report([f"  {pw_reg['feature_names'][i]}: R2={pw_r2[i]:.4f}, p={pw_reg['f_pvalue'][i]:.2e}"])

        # Assertions: pathway regression
        assert len(pw_r2) == scores.shape[1], "Pathway R2 count mismatch"
        assert np.all(np.isfinite(pw_r2)), "Non-finite pathway R2"
        # F8: Verify feature names are actual pathway names (not generic feature_0, etc.)
        assert pw_reg["feature_names"][0] != "feature_0", \
            f"Pathway regression returned generic names: {pw_reg['feature_names'][:3]}"

    # === 8. Flow Matching & Feature Alignment =================================
    section("8. Flow Matching (CMP -> Mono)")

    step("Building combined adata for flow")
    adata_cmp_flow = ad.AnnData(
        X=sp.csr_matrix((adata_cmp.n_obs, 1)),
        obs=adata_cmp.obs.copy(),
    )
    adata_cmp_flow.obsm["X_pca"] = cell_data["CMP"]["pca"]
    adata_cmp_flow.obs["cell_type_label"] = "CMP"

    adata_mono_flow = ad.AnnData(
        X=sp.csr_matrix((adata_mono.n_obs, 1)),
        obs=adata_mono.obs.copy(),
    )
    adata_mono_flow.obsm["X_pca"] = cell_data["Mono"]["pca"]
    adata_mono_flow.obs["cell_type_label"] = "Mono"

    adata_combined = ad.concat([adata_cmp_flow, adata_mono_flow])
    done(f"({adata_combined.n_obs:,} cells)")

    step(f"Training flow CMP -> Mono ({FLOW_EPOCHS} epochs)")
    t1 = time.time()
    flow_result = pc.tl.flow_within(
        adata_combined,
        source={"cell_type_label": "CMP"},
        target={"cell_type_label": "Mono"},
        pca_key="X_pca",
        hidden_dims=(64, 64, 64),
        n_epochs=FLOW_EPOCHS,
        batch_size=256,
        n_steps=50,
        device="cpu",
    )
    elapsed = time.time() - t1
    mmd_reduction = (flow_result["mmd_before"] - flow_result["mmd_after"]) / flow_result["mmd_before"] * 100
    done(f"({elapsed:.1f}s)")
    report([
        f"FLOW TRAINING RESULTS:",
        f"  MMD before: {flow_result['mmd_before']:.4f}",
        f"  MMD after:  {flow_result['mmd_after']:.4f}",
        f"  MMD reduction: {mmd_reduction:.1f}%",
        f"  CONTROL: MMD should decrease substantially (>50% reduction expected)",
        f"    Good transport = source distribution mapped close to target",
    ])

    # Assertions: flow training
    assert flow_result["mmd_after"] < flow_result["mmd_before"], \
        f"Flow did not reduce MMD: {flow_result['mmd_before']:.4f} -> {flow_result['mmd_after']:.4f}"
    # F2: Minimum convergence quality — trivial MMD decrease would invalidate gene alignment
    assert mmd_reduction > 20.0, \
        f"Flow MMD reduction too small: {mmd_reduction:.1f}% (expect >20%)"
    assert flow_result["transported"].shape[1] == N_PCS, \
        f"Transported shape wrong: {flow_result['transported'].shape}"

    # Flow visualizations
    step("Flow visualizations")
    for viz_name, viz_fn, filename in [
        ("flow_magnitude", lambda: pc.pl.flow_magnitude(adata_combined, flow_result, show=False), "flow_magnitude.html"),
        ("density_comparison", lambda: pc.pl.density_comparison(adata_combined, flow_result, show=False), "density_comparison.html"),
        ("velocity_quiver", lambda: pc.pl.velocity_quiver(adata_combined, flow_result, show=False), "velocity_quiver.html"),
    ]:
        fig = viz_fn()
        if fig is not None:
            save_viz(fig, filename)
    done()

    # 8b. Gene-level flow alignment
    step("Gene-level flow alignment")
    n_top_genes = 50
    # Build thin adata with correct gene dimensions for alignment
    adata_for_alignment = ad.AnnData(
        X=sp.csr_matrix((adata_combined.n_obs, len(var_names))),
        var=pd.DataFrame(index=var_names),
    )
    adata_for_alignment.obsm["X_pca"] = adata_combined.obsm["X_pca"]
    if pca_loadings is not None:
        adata_for_alignment.varm["PCs"] = pca_loadings
    alignment = pc.tl.flow_gene_alignment(adata_for_alignment, flow_result, n_top=n_top_genes)
    scores_align = alignment["alignment_scores"]
    done()
    report([
        f"GENE-LEVEL FLOW ALIGNMENT:",
        f"  Genes scored: {len(scores_align)}",
        f"  Alignment score range: [{scores_align.min():.4f}, {scores_align.max():.4f}]",
        f"  Mean |alignment|: {np.abs(scores_align).mean():.4f}",
        f"",
        f"TOP 10 FLOW-ALIGNED GENES (upregulated CMP->Mono):",
    ])
    for gene in alignment["top_aligned"][:10]:
        idx = list(alignment["gene_names"]).index(gene)
        report([f"  {gene}: score={scores_align[idx]:.4f}"])
    report([f"", f"TOP 10 FLOW-OPPOSED GENES (downregulated CMP->Mono):"])
    for gene in alignment["top_opposed"][:10]:
        idx = list(alignment["gene_names"]).index(gene)
        report([f"  {gene}: score={scores_align[idx]:.4f}"])

    # Assertions: gene alignment
    assert len(alignment["top_aligned"]) == n_top_genes, \
        f"Expected {n_top_genes} aligned genes, got {len(alignment['top_aligned'])}"
    assert len(alignment["top_opposed"]) == n_top_genes, \
        f"Expected {n_top_genes} opposed genes, got {len(alignment['top_opposed'])}"
    assert len(scores_align) == len(var_names), \
        f"Alignment scores length mismatch: {len(scores_align)} vs {len(var_names)}"
    # F3: Non-trivial alignment — if all scores are zero, flow velocity is degenerate
    assert np.abs(scores_align).max() > 1e-6, \
        f"All gene alignment scores are ~zero (max |score|={np.abs(scores_align).max():.2e}) — flow may be degenerate"

    # 8c. Geneset-level flow alignment (HALLMARK)
    step("HALLMARK geneset flow alignment")
    gene_to_score = dict(zip(alignment["gene_names"], scores_align))
    pathway_gene_sets = net.groupby("source")["target"].apply(set).to_dict()

    # F9: Report gene symbol overlap between dataset and HALLMARK
    total_pw_genes = sum(len(gs) for gs in pathway_gene_sets.values())
    matched_pw_genes = sum(1 for gs in pathway_gene_sets.values() for g in gs if g in gene_to_score)
    overlap_rate = matched_pw_genes / total_pw_genes if total_pw_genes > 0 else 0

    pathway_alignment = {}
    for pw_name, pw_genes in pathway_gene_sets.items():
        matched = [gene_to_score[g] for g in pw_genes if g in gene_to_score]
        if len(matched) >= 5:  # require at least 5 genes for reliable estimate
            pathway_alignment[pw_name] = {
                "mean_score": float(np.mean(matched)),
                "n_genes": len(matched),
                "n_aligned": sum(1 for s in matched if s > 0),
                "n_opposed": sum(1 for s in matched if s < 0),
            }
    done(f"({len(pathway_alignment)} pathways with >=5 genes)")
    report([
        f"GENE OVERLAP: {matched_pw_genes}/{total_pw_genes} pathway genes in var_names ({100*overlap_rate:.1f}%)",
    ])

    # F5: Permutation null for geneset alignment significance
    step("Geneset alignment permutation null (500 permutations)")
    all_scores_list = list(gene_to_score.values())
    rng_gs = np.random.default_rng(SEED)
    n_perm_gs = 500
    for pw_name in pathway_alignment:
        n_g = pathway_alignment[pw_name]["n_genes"]
        null_means = np.array([
            float(np.mean(rng_gs.choice(all_scores_list, size=n_g, replace=False)))
            for _ in range(n_perm_gs)
        ])
        obs_mean = pathway_alignment[pw_name]["mean_score"]
        pval_gs = (np.sum(np.abs(null_means) >= np.abs(obs_mean)) + 1) / (n_perm_gs + 1)
        pathway_alignment[pw_name]["pvalue"] = float(pval_gs)
    n_sig_gs = sum(1 for v in pathway_alignment.values() if v["pvalue"] < ALPHA)
    done(f"({n_sig_gs}/{len(pathway_alignment)} significant at p<{ALPHA})")

    report([
        f"HALLMARK GENESET FLOW ALIGNMENT:",
        f"  Pathways scored: {len(pathway_alignment)}",
        f"  Significant (permutation p<{ALPHA}): {n_sig_gs}",
        f"",
        f"TOP 10 FLOW-ALIGNED PATHWAYS (activated CMP->Mono):",
    ])
    aligned_pw = sorted(pathway_alignment.items(), key=lambda x: x[1]["mean_score"], reverse=True)
    for pw_name, pw_info in aligned_pw[:10]:
        p_str = f", p={pw_info['pvalue']:.3f}" if "pvalue" in pw_info else ""
        report([f"  {pw_name}: mean={pw_info['mean_score']:.4f} ({pw_info['n_aligned']}/{pw_info['n_genes']} genes aligned{p_str})"])

    report([f"", f"TOP 10 FLOW-OPPOSED PATHWAYS (deactivated CMP->Mono):"])
    opposed_pw = sorted(pathway_alignment.items(), key=lambda x: x[1]["mean_score"])
    for pw_name, pw_info in opposed_pw[:10]:
        p_str = f", p={pw_info['pvalue']:.3f}" if "pvalue" in pw_info else ""
        report([f"  {pw_name}: mean={pw_info['mean_score']:.4f} ({pw_info['n_opposed']}/{pw_info['n_genes']} genes opposed{p_str})"])

    # Assertions: geneset alignment
    assert overlap_rate > 0.1, \
        f"Pathway gene overlap too low ({100*overlap_rate:.1f}%) — check gene symbol format"
    assert len(pathway_alignment) > 0, "No HALLMARK pathways had >=5 matched genes"

    # === 9. Summary & Cross-Validation ========================================
    section("9. Summary & Cross-Validation")

    # 9a. Cross-cell-type regression concordance (gene-level)
    step("Cross-cell-type gene regression concordance")
    reg_cmp = gene_regs["CMP"]
    reg_mono = gene_regs["Mono"]
    # Both share same var_names, so feature order is identical
    r2_cmp = np.asarray(reg_cmp["r_squared_degree1"])
    r2_mono = np.asarray(reg_mono["r_squared_degree1"])
    from scipy.stats import spearmanr
    rho, pval = spearmanr(r2_cmp, r2_mono)
    done()
    report([
        f"CROSS-CELL-TYPE R2 CONCORDANCE:",
        f"  Spearman rho: {rho:.4f} (p={pval:.2e})",
        f"  CONTROL: Moderate positive correlation expected",
        f"    (shared biology -> shared archetype-dependent genes)",
        f"    Very high rho (>0.9) would suggest batch effect, not biology",
        f"    Negative rho would suggest data problem",
    ])
    assert rho > 0, f"Negative R2 correlation between cell types: rho={rho:.4f}"
    assert pval < 0.05, f"R2 correlation not significant: p={pval:.2e}"

    # 9b. Pathway regression concordance
    step("Pathway regression concordance")
    if "CMP" in pathway_regs and "Mono" in pathway_regs:
        pw_r2_cmp = np.asarray(pathway_regs["CMP"]["r_squared_degree1"])
        pw_r2_mono = np.asarray(pathway_regs["Mono"]["r_squared_degree1"])
        if len(pw_r2_cmp) == len(pw_r2_mono):
            pw_rho, pw_pval = spearmanr(pw_r2_cmp, pw_r2_mono)
            done()
            report([
                f"PATHWAY R2 CONCORDANCE:",
                f"  Spearman rho: {pw_rho:.4f} (p={pw_pval:.2e})",
                f"  Pathways: {len(pw_r2_cmp)}",
            ])
        else:
            done("(pathway count mismatch — skipped)")
    else:
        done("(pathway regression not available — skipped)")

    # 9c. Summary table
    report([
        f"",
        f"{'='*58}",
        f"  PIPELINE SUMMARY",
        f"{'='*58}",
        f"  Cell types: CMP ({adata_cmp.n_obs:,} cells), Mono ({adata_mono.n_obs:,} cells)",
        f"  Archetypes: K={K}",
        f"  CMP training R2: {models['CMP'].get('final_archetype_r2', 0):.3f}",
        f"  Mono training R2: {models['Mono'].get('final_archetype_r2', 0):.3f}",
        f"  Genes tested: {adata_cmp.n_vars:,}",
        f"  CMP sig genes (F-test FDR<{ALPHA}): {int(np.sum(np.asarray(gene_regs['CMP']['f_pvalue_fdr']) < ALPHA))}",
        f"  Mono sig genes (F-test FDR<{ALPHA}): {int(np.sum(np.asarray(gene_regs['Mono']['f_pvalue_fdr']) < ALPHA))}",
        f"  HALLMARK pathways scored: {len(pathway_alignment)}",
        f"  Flow MMD: {flow_result['mmd_before']:.4f} -> {flow_result['mmd_after']:.4f}",
        f"  Flow-aligned genes (top {n_top_genes}): {alignment['top_aligned'][:5]}",
        f"{'='*58}",
    ])

    # === Final Summary ========================================================
    section("Final Summary")
    elapsed_total = time.time() - t0
    n_files = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.html')])
    print(f"  Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)", flush=True)
    print(f"  HTML files: {n_files} in {OUTPUT_DIR}", flush=True)
    print(f"\n  ALL SECTIONS PASSED", flush=True)


if __name__ == "__main__":
    main()

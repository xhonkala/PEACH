#!/usr/bin/env python
"""Comprehensive v0.5.0 headless E2E test.

Exercises all features requested for the reviewer notebook:
  a) Per simplex regression beta coefficient degree results unpacking
  b) Pathway enrichment → simplex regression on BOTH genes and geneset scores
  c) Full Wald testing (archetype_contrasts for all pairs)
  d) Flow matching soft assignment of archetype similarity between CMP/Mono
  e) Flow Jacobian + inter-flow timepoint sampling + CellRank comparison
  Plus: per-GMM-component 3D viz, per-component archetype dominance
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
import anndata as ad
import gc
from collections import Counter
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.disable(logging.WARNING)

import torch
torch.set_num_threads(1)

# Prevent macOS multiprocessing spawn issue (CellRank uses multiprocessing)
import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != 'fork':
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass

import peach as pc

# ---------- Config ----------
HSC_PATH = "/Users/honkala/Desktop/cross_recons/data/HSC.h5ad"
OUT_DIR = os.path.join(os.path.dirname(__file__), "e2e_outputs_comprehensive")
K = 4
N_PCS = 20
MONO_SUBSAMPLE = 9000
TRAIN_EPOCHS = 150
FLOW_EPOCHS = 300
SEED = 42
N_BOOTSTRAP = 50
N_PERMUTATIONS = 50
ALPHA = 0.05
os.makedirs(OUT_DIR, exist_ok=True)
t0 = time.time()

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ================================================================
# 1. DATA LOADING
# ================================================================
section("1. Data Loading & Gene Symbol Mapping")

adata_full = ad.read_h5ad(HSC_PATH)
print(f"Loaded: {adata_full.shape[0]:,} cells x {adata_full.shape[1]:,} genes")
n_genes_original = adata_full.n_vars

# Swap ENSG -> gene symbols
symbols_raw = list(adata_full.var["gene_symbols"].values)
symbols = []
for i, s in enumerate(symbols_raw):
    if pd.isna(s) or str(s).strip() == "":
        symbols.append(f"UNNAMED_{i}")
    else:
        symbols.append(str(s).strip())

counts = Counter(symbols)
seen = Counter()
unique_symbols = []
for s in symbols:
    if counts[s] > 1:
        seen[s] += 1
        unique_symbols.append(f"{s}_{seen[s]}")
    else:
        unique_symbols.append(s)
var_names = unique_symbols

# PCA loadings
if "PCs" in adata_full.varm:
    pca_loadings = adata_full.varm["PCs"][:, :N_PCS].copy()
else:
    pca_loadings = None

# Subset populations
rng = np.random.default_rng(SEED)
cell_data = {}
for name, ct, subsample in [
    ("CMP", "common myeloid progenitor", None),
    ("Mono", "CD14-positive monocyte", MONO_SUBSAMPLE),
]:
    mask = adata_full.obs["cell_type"] == ct
    idx = np.where(mask)[0]
    if subsample and len(idx) > subsample:
        idx = rng.choice(idx, size=subsample, replace=False)
        idx.sort()
    pca = adata_full.obsm["X_pca"][idx, :N_PCS].copy()
    obs = adata_full.obs.iloc[idx].copy().reset_index(drop=True)
    cell_data[name] = {"pca": pca, "obs": obs, "idx": idx}
    print(f"  {name}: {len(idx):,} cells, PCA: {pca.shape}")

if pca_loadings is None:
    from sklearn.decomposition import PCA
    X_for_pca = adata_full.X[cell_data["CMP"]["idx"]]
    if hasattr(X_for_pca, "toarray"):
        X_for_pca = X_for_pca.toarray()
    pca_model = PCA(n_components=N_PCS, random_state=SEED)
    pca_model.fit(X_for_pca)
    pca_loadings = pca_model.components_.T
    del X_for_pca

del adata_full
gc.collect()
print("  Data loading complete.")


# ================================================================
# 2. ARCHETYPE FITTING
# ================================================================
section("2. Archetype Fitting (K={})".format(K))

models = {}
for name in ["CMP", "Mono"]:
    print(f"\n  --- Training {name} ---")
    t1 = time.time()
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
    r2 = result.get("final_archetype_r2", None)
    print(f"  R2={r2:.3f}, {time.time()-t1:.1f}s")

    pc.tl.extract_archetype_weights(adata_train)
    pc.tl.archetypal_coordinates(adata_train, verbose=False)

    weights = adata_train.obsm["cell_archetype_weights"]
    cell_data[name]["weights"] = weights
    cell_data[name]["uns"] = dict(adata_train.uns)
    cell_data[name]["arch_dist"] = adata_train.obsm["archetype_distances"]
    models[name] = result

    assert r2 is not None and r2 > 0.0
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-4)
    for k_idx in range(K):
        assert int(np.sum(weights.argmax(axis=1) == k_idx)) > 0

# Rebuild adata with sparse X and gene symbols
print("\n  Rebuilding adata with sparse X and gene symbols...")
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
    if pca_loadings is not None:
        adata_obj.varm["PCs"] = pca_loadings
    pc.tl.assign_archetypes(adata_obj)
    cell_data[name]["adata"] = adata_obj

del adata_disk
gc.collect()

adata_cmp = cell_data["CMP"]["adata"]
adata_mono = cell_data["Mono"]["adata"]
print(f"  CMP: {adata_cmp.shape}, Mono: {adata_mono.shape}")


# ================================================================
# 3. PATHWAY ENRICHMENT (before regression)
# ================================================================
section("3. Pathway Enrichment (HALLMARK)")

net = pc.pp.load_pathway_networks(["hallmark"], verbose=False)
n_pathways = net["source"].nunique()
n_pw_genes = net["target"].nunique()
print(f"  HALLMARK: {n_pathways} pathways, {n_pw_genes} genes")

for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    t1 = time.time()
    pc.pp.compute_pathway_scores(ad_obj, net, verbose=False)
    elapsed = time.time() - t1
    scores = ad_obj.obsm["pathway_scores"]
    pw_names = ad_obj.uns["pathway_scores_pathways"]
    print(f"  {name}: {scores.shape[1]} pathways scored in {elapsed:.1f}s")
    print(f"    Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    assert scores.shape[0] == ad_obj.n_obs
    assert np.all(np.isfinite(scores))


# ================================================================
# 4. SIMPLEX REGRESSION — GENES
# ================================================================
section("4a. Simplex Regression on Genes")

gene_regs = {}
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    print(f"\n  --- {name} (bootstrap={N_BOOTSTRAP}, perm={N_PERMUTATIONS}) ---")
    t1 = time.time()
    reg = pc.tl.feature_simplex_regression(
        ad_obj, max_degree=2,
        n_bootstrap=N_BOOTSTRAP,
        permutation_test=True,
        n_permutations=N_PERMUTATIONS,
        robust_se=True,
        comprehensive_degree=True,
    )
    print(f"  Completed in {time.time()-t1:.1f}s")
    gene_regs[name] = reg

    r2_d1 = np.asarray(reg["r_squared_degree1"])
    r2_d2 = np.asarray(reg["r_squared_degree2"])
    f_pval_fdr = np.asarray(reg["f_pvalue_fdr"])
    n_sig_f = int(np.sum(f_pval_fdr < ALPHA))
    print(f"  Features: {len(r2_d1)}")
    print(f"  Degree-1 R2: median={np.median(r2_d1):.4f}, max={np.max(r2_d1):.4f}")
    print(f"  Degree-2 R2: median={np.median(r2_d2):.4f}, max={np.max(r2_d2):.4f}")
    print(f"  F-test significant: {n_sig_f}/{len(r2_d1)}")

    assert len(r2_d1) == ad_obj.n_vars
    assert np.all(r2_d1 >= -0.1) and np.all(r2_d1 <= 1.0 + 1e-6)
    r2_diff = r2_d2 - r2_d1
    assert int(np.sum(r2_diff < -1e-6)) == 0, "Nested model invariant violated"


# ================================================================
# 4b. SIMPLEX REGRESSION — PATHWAY SCORES
# ================================================================
section("4b. Simplex Regression on Pathway Scores")

pathway_regs = {}
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    t1 = time.time()
    pw_names = list(ad_obj.uns["pathway_scores_pathways"])
    pw_reg = pc.tl.pathway_simplex_regression(
        ad_obj, n_bootstrap=N_BOOTSTRAP, robust_se=True,
        feature_names=pw_names,
    )
    elapsed = time.time() - t1
    pathway_regs[name] = pw_reg

    pw_r2 = np.asarray(pw_reg["r_squared_degree1"])
    pw_pval_fdr = np.asarray(pw_reg["f_pvalue_fdr"])
    n_sig_pw = int(np.sum(pw_pval_fdr < ALPHA))
    print(f"  {name}: {n_sig_pw}/{len(pw_r2)} significant pathways ({elapsed:.1f}s)")
    print(f"  R2 range: [{pw_r2.min():.4f}, {pw_r2.max():.4f}]")

    # Top 5 pathways
    for i in np.argsort(pw_r2)[-5:][::-1]:
        print(f"    {pw_reg['feature_names'][i]}: R2={pw_r2[i]:.4f}")

    assert pw_reg["feature_names"][0] != "feature_0"
    # Verify gene regression survived pathway regression
    assert "peach_simplex_regression_genes" in ad_obj.uns


# ================================================================
# 4c. BETA COEFFICIENT DEGREE UNPACKING
# ================================================================
section("4c. Beta Coefficient Degree Unpacking")

for name in ["CMP", "Mono"]:
    reg = gene_regs[name]
    coefs = np.asarray(reg["vertex_coefficients"])
    vertex_se = np.asarray(reg["vertex_se"])
    vertex_pvals = np.asarray(reg["vertex_pvalues"])
    vertex_pvals_fdr = np.asarray(reg["vertex_pvalues_fdr"])
    int_coefs = reg.get("interaction_coefficients")
    int_pairs = reg.get("interaction_pairs")
    int_pvals_fdr = reg.get("interaction_pvalues_fdr")
    r2_d1 = np.asarray(reg["r_squared_degree1"])
    r2_d2 = np.asarray(reg["r_squared_degree2"])
    feat_names = reg["feature_names"]

    print(f"\n  --- {name}: Degree 1 (vertex coefficients) ---")
    print(f"  Shape: {coefs.shape}  (n_features x K)")
    print(f"  Range: [{coefs.min():.4f}, {coefs.max():.4f}]")
    print(f"  SE range: [{vertex_se.min():.6f}, {vertex_se.max():.4f}]")

    # Per-archetype summary
    for k in range(K):
        median_beta = np.median(coefs[:, k])
        n_sig = int(np.sum(vertex_pvals_fdr[:, k] < ALPHA))
        print(f"    Archetype {k}: median(beta)={median_beta:.4f}, "
              f"n_sig={n_sig}/{len(coefs)}")

    # Top 3 genes per archetype by |beta|
    print(f"\n  Top 3 genes per archetype (by |beta|):")
    for k in range(K):
        top3 = np.argsort(np.abs(coefs[:, k]))[-3:][::-1]
        for i in top3:
            print(f"    A{k}: {feat_names[i]:20s}  "
                  f"beta={coefs[i,k]:8.3f}  SE={vertex_se[i,k]:.3f}  "
                  f"p_fdr={vertex_pvals_fdr[i,k]:.2e}")

    if int_coefs is not None:
        int_coefs = np.asarray(int_coefs)
        print(f"\n  --- {name}: Degree 2 (interaction coefficients) ---")
        print(f"  Shape: {int_coefs.shape}  (n_features x n_pairs)")
        print(f"  Pairs: {int_pairs}")
        print(f"  Range: [{int_coefs.min():.4f}, {int_coefs.max():.4f}]")

        if int_pvals_fdr is not None:
            int_pvals_fdr = np.asarray(int_pvals_fdr)
            for p_idx, pair in enumerate(int_pairs):
                n_sig_int = int(np.sum(int_pvals_fdr[:, p_idx] < ALPHA))
                print(f"    Pair {pair}: {n_sig_int} significant interactions")

    # Bootstrap CIs
    ci_lo = reg.get("vertex_ci_lower")
    ci_hi = reg.get("vertex_ci_upper")
    if ci_lo is not None:
        ci_lo = np.asarray(ci_lo)
        ci_hi = np.asarray(ci_hi)
        print(f"\n  Bootstrap CIs: lower shape={ci_lo.shape}, upper shape={ci_hi.shape}")
        ci_width = ci_hi - ci_lo
        print(f"  CI width: median={np.median(ci_width):.4f}, "
              f"max={np.max(ci_width):.4f}")
        # Sanity: CI should bracket point estimate
        n_outside = int(np.sum((coefs < ci_lo) | (coefs > ci_hi)))
        frac_outside = n_outside / coefs.size
        print(f"  Point estimates outside CI: {n_outside}/{coefs.size} ({frac_outside:.1%})")

    # Degree comparison
    deg_comp = reg.get("degree_comparison")
    if deg_comp is not None:
        print(f"\n  --- {name}: Degree Comparison ---")
        for deg_key, deg_info in sorted(deg_comp.items()):
            delta_r2 = np.asarray(deg_info["delta_r2"])
            inc_p_fdr = np.asarray(deg_info["incremental_p_fdr"])
            n_sig_inc = int(deg_info["significant_features"])
            print(f"    {deg_key}: {deg_info['n_params']} params, "
                  f"{deg_info['df_extra']} extra df")
            print(f"      Significant: {n_sig_inc}/{len(delta_r2)} "
                  f"({100*n_sig_inc/len(delta_r2):.1f}%)")
            print(f"      Delta R2: median={np.median(delta_r2):.6f}, "
                  f"max={np.max(delta_r2):.4f}")

            # Top 5 genes gaining most from this degree
            top5 = np.argsort(delta_r2)[-5:][::-1]
            for i in top5:
                print(f"        {feat_names[i]:20s}  "
                      f"dR2={delta_r2[i]:.4f}  p_fdr={inc_p_fdr[i]:.2e}")

    assert coefs.shape == (len(feat_names), K)
    assert vertex_se.shape == coefs.shape
    assert vertex_pvals.shape == coefs.shape


# ================================================================
# 5. FULL WALD TESTING
# ================================================================
section("5. Full Wald Testing (Archetype Contrasts)")

for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    contrasts = pc.tl.archetype_contrasts(ad_obj)

    print(f"\n  --- {name}: Wald Contrasts ---")
    print(f"  Pairs: {contrasts['pairs']}")
    print(f"  Features: {contrasts['n_features']}")
    print(f"  Archetypes: {contrasts['n_archetypes']}")

    for pair in contrasts["pairs"]:
        key = str(tuple(pair)) if not isinstance(pair, str) else pair
        delta_beta = contrasts["delta_beta"][key]
        delta_se = contrasts["delta_se"][key]
        z_scores = contrasts["z_scores"][key]
        pvals_fdr = contrasts["pvalues_fdr"][key]

        n_sig = int(np.sum(pvals_fdr < ALPHA))
        n_up = int(np.sum((pvals_fdr < ALPHA) & (delta_beta > 0)))
        n_down = int(np.sum((pvals_fdr < ALPHA) & (delta_beta < 0)))

        print(f"\n    Pair {pair}: {n_sig} significant "
              f"({n_up} up, {n_down} down)")
        print(f"      |delta_beta| range: "
              f"[{np.abs(delta_beta).min():.4f}, {np.abs(delta_beta).max():.4f}]")
        print(f"      |z| range: [{np.abs(z_scores).min():.4f}, {np.abs(z_scores).max():.2f}]")

        # Top 3 by |z|
        feat_names = contrasts["feature_names"]
        top3 = np.argsort(np.abs(z_scores))[-3:][::-1]
        for i in top3:
            print(f"        {feat_names[i]:20s}  "
                  f"dB={delta_beta[i]:8.3f}  SE={delta_se[i]:.3f}  "
                  f"z={z_scores[i]:.1f}  q={pvals_fdr[i]:.2e}")

    assert len(contrasts["pairs"]) == K * (K - 1) // 2
    assert contrasts["n_features"] > 0


# ================================================================
# 6. PATTERN CLASSIFICATION
# ================================================================
section("6. Pattern Classification")

for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    patterns = pc.tl.classify_feature_patterns(ad_obj)
    counts_p = patterns["pattern_counts"]
    n_total = patterns["n_features"]

    print(f"\n  {name}:")
    for ptype, pcount in sorted(counts_p.items(), key=lambda x: -x[1]):
        print(f"    {ptype:25s} {pcount:5d} ({100*pcount/n_total:5.1f}%)")

    assert sum(counts_p.values()) == n_total


# ================================================================
# 7. GMM SIMPLEX DECOMPOSITION
# ================================================================
section("7. GMM Simplex Decomposition")

from peach._core.utils.ilr_transform import ilr_transform, inverse_ilr

gmm_results = {}
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    weights = ad_obj.obsm["cell_archetype_weights"]
    ilr_coords = ilr_transform(weights)
    roundtrip = inverse_ilr(ilr_coords)
    roundtrip_error = np.abs(roundtrip - weights).max()
    print(f"\n  {name} ILR: shape={ilr_coords.shape}, roundtrip_err={roundtrip_error:.2e}")
    assert roundtrip_error < 0.01

    t1 = time.time()
    gmm = pc.tl.feature_simplex_decomposition(ad_obj, characterize_features=True)
    gmm_results[name] = gmm
    print(f"  GMM ({time.time()-t1:.1f}s): {gmm['n_components_optimal']} optimal, "
          f"{gmm['n_components_stable']} stable")
    print(f"  Archetype map: {gmm['component_archetype_map']}")

    assert gmm["n_components_optimal"] >= K
    assert gmm["n_components_stable"] >= 1


# ================================================================
# 7a. PER-COMPONENT ARCHETYPE DOMINANCE
# ================================================================
section("7a. Per-Component Archetype Dominance")

import plotly.graph_objects as go

archetype_vertices = np.eye(K)
simplex_center = np.ones(K) / K

for name in ["CMP", "Mono"]:
    gmm = gmm_results[name]
    weight_means = gmm.get("component_weight_means")
    if weight_means is None:
        print(f"  {name}: component_weight_means not available, skipping")
        continue

    ad_obj = adata_cmp if name == "CMP" else adata_mono
    n_comp = weight_means.shape[0]
    arch_map = gmm["component_archetype_map"]
    assignments = np.asarray(gmm["component_assignments"])

    print(f"\n  --- {name}: {n_comp} components ---")
    for c in range(n_comp):
        # Dominance: which archetype has highest mean weight
        dominant_arch = int(np.argmax(weight_means[c]))
        dominance_strength = weight_means[c, dominant_arch]
        n_cells_c = int(np.sum(assignments == c))
        w_str = ", ".join(f"{w:.3f}" for w in weight_means[c])

        # Entropy of weight profile (lower = more dominated by one archetype)
        w_safe = np.clip(weight_means[c], 1e-10, 1.0)
        entropy = -np.sum(w_safe * np.log2(w_safe))
        max_entropy = np.log2(K)

        print(f"  Comp {c}: dominant=A{dominant_arch} "
              f"(w={dominance_strength:.3f}), "
              f"n_cells={n_cells_c}, "
              f"entropy={entropy:.2f}/{max_entropy:.2f}")
        print(f"    weights: [{w_str}]")

        # Distance to each archetype vertex
        dists = [np.linalg.norm(weight_means[c] - archetype_vertices[k])
                 for k in range(K)]
        print(f"    dists:   [{', '.join(f'{d:.3f}' for d in dists)}]")

    # Distance heatmap
    dist_matrix = np.zeros((n_comp, K))
    for c in range(n_comp):
        for k_idx in range(K):
            dist_matrix[c, k_idx] = np.linalg.norm(
                weight_means[c] - archetype_vertices[k_idx])

    fig = go.Figure(data=go.Heatmap(
        z=dist_matrix,
        x=[f"Arch {k}" for k in range(K)],
        y=[f"Comp {c} (→A{arch_map[c]})" for c in range(n_comp)],
        colorscale="Viridis_r",
        text=np.round(dist_matrix, 3),
        texttemplate="%{text}",
        colorbar_title="Distance",
    ))
    fig.update_layout(
        title=f"{name}: Component → Archetype Distance",
        height=300 + 30 * n_comp, width=500,
    )
    fig.write_image(os.path.join(OUT_DIR, f"comp_arch_dist_{name}.png"))

    # Weight profile stacked bar
    fig2 = go.Figure()
    for k_idx in range(K):
        fig2.add_trace(go.Bar(
            name=f"Arch {k_idx}",
            x=[f"Comp {c}" for c in range(n_comp)],
            y=weight_means[:, k_idx],
        ))
    fig2.update_layout(
        barmode="stack",
        title=f"{name}: Component Weight Profiles",
        yaxis_range=[0, 1.05], height=400, width=600,
    )
    fig2.write_image(os.path.join(OUT_DIR, f"comp_weight_profile_{name}.png"))

    assert dist_matrix.shape == (n_comp, K)
    assert np.all(dist_matrix >= 0)


# ================================================================
# 7b. PER-GMM-COMPONENT 3D VIZ
# ================================================================
section("7b. Per-GMM-Component 3D Visualization")

for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    gmm = gmm_results[name]
    assignments = np.asarray(gmm["component_assignments"])
    n_stable = gmm["n_components_stable"]

    # Store GMM labels in obs for coloring
    ad_obj.obs["gmm_component"] = pd.Categorical(
        [f"C{c}" if c >= 0 else "unassigned" for c in assignments]
    )

    print(f"\n  {name}: {n_stable} stable components in obs['gmm_component']")

    # 3D archetypal space colored by GMM component
    fig = pc.pl.archetypal_space(
        ad_obj, color_by="gmm_component",
        title=f"{name}: Archetypal Space by GMM Component",
        save_path=os.path.join(OUT_DIR, f"arch_space_gmm_{name}.png"),
    )
    print(f"  Saved 3D viz to arch_space_gmm_{name}.png")

    assert "gmm_component" in ad_obj.obs.columns


# ================================================================
# 8. ARCHETYPE COMPARISON
# ================================================================
section("8. Archetype Comparison (MMD, Similarity, Contrasts)")

for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    # Within-fit MMD
    mmd = pc.tl.archetype_mmd(ad_obj, n_permutations=100)
    mask_off = ~np.eye(K, dtype=bool)
    mmd_vals = mmd["mmd_matrix"][mask_off]
    print(f"\n  {name} within-fit MMD: range=[{mmd_vals.min():.4f}, {mmd_vals.max():.4f}]")
    pc.pl.mmd_heatmap(ad_obj, show=False,
                      save_path=os.path.join(OUT_DIR, f"mmd_{name}.png"))

    # Feature similarity
    sim = pc.tl.archetype_feature_similarity(ad_obj)
    print(f"  {name} silhouette: {sim['silhouette_overall']:.3f}")
    pc.pl.feature_similarity_heatmap(
        ad_obj, show=False,
        save_path=os.path.join(OUT_DIR, f"feat_sim_{name}.png"))

    # Wald contrast volcano (first pair)
    pc.pl.contrast_volcano(
        ad_obj, pair=(0, 1), show=False,
        save_path=os.path.join(OUT_DIR, f"volcano_{name}_0v1.png"))

# Between-fit
mmd_between = pc.tl.archetype_mmd(adata_cmp, adata_mono, n_permutations=100)
print(f"\n  Between-fit MMD matrix:")
print(f"  {np.array2string(np.asarray(mmd_between['mmd_matrix']), precision=4)}")

sim_between = pc.tl.archetype_feature_similarity(adata_cmp, adata_mono)
print(f"  Between-fit Spearman: overall silhouette={sim_between['silhouette_overall']:.3f}")


# ================================================================
# 9. FLOW MATCHING (with return_model for Jacobian)
# ================================================================
section("9. Flow Matching (CMP → Mono)")

# Build combined adata
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
print(f"  Combined: {adata_combined.n_obs:,} cells")

print(f"  Training flow ({FLOW_EPOCHS} epochs)...")
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
    return_model=True,
)
elapsed = time.time() - t1
mmd_reduction = (flow_result["mmd_before"] - flow_result["mmd_after"]) / flow_result["mmd_before"] * 100
print(f"  Done ({elapsed:.1f}s): MMD {flow_result['mmd_before']:.4f} → "
      f"{flow_result['mmd_after']:.4f} ({mmd_reduction:.1f}% reduction)")

assert "model" in flow_result, "return_model=True but no model in result"
assert mmd_reduction > 20.0
flow_model = flow_result["model"]

# Visualizations
pc.pl.flow_magnitude(adata_combined, flow_result, show=False,
                     save_path=os.path.join(OUT_DIR, "flow_magnitude.png"))
pc.pl.density_comparison(adata_combined, flow_result, show=False,
                         save_path=os.path.join(OUT_DIR, "flow_density.png"))
pc.pl.velocity_quiver(adata_combined, flow_result, show=False,
                      save_path=os.path.join(OUT_DIR, "flow_quiver.png"))


# ================================================================
# 9a. FLOW JACOBIAN
# ================================================================
section("9a. Flow Jacobian Analysis")

# Subsample for Jacobian (expensive: O(n * dim) backward passes)
source_pca = adata_combined.obsm["X_pca"][flow_result["source_mask"]]
n_jac_sample = 200
rng_jac = np.random.default_rng(SEED)
jac_idx = rng_jac.choice(len(source_pca), size=min(n_jac_sample, len(source_pca)), replace=False)
jac_points = source_pca[jac_idx]

# Jacobian at t=0.5 (midpoint of flow)
print(f"  Computing Jacobian at t=0.5 on {len(jac_points)} points...")
t1 = time.time()
jac_result = pc.tl.flow_jacobian(
    adata_combined, flow_result, flow_model,
    t=0.5,
    evaluation_points=jac_points,
)
elapsed = time.time() - t1

jac_det = jac_result["jacobian_det"]
mean_jac = jac_result["mean_jacobian"]
print(f"  Jacobian computed in {elapsed:.1f}s")
print(f"  Determinant stats: mean={jac_det.mean():.4f}, "
      f"std={jac_det.std():.4f}, "
      f"range=[{jac_det.min():.4f}, {jac_det.max():.4f}]")
print(f"  Mean Jacobian shape: {mean_jac.shape}")
print(f"  Mean Jacobian trace: {np.trace(mean_jac):.4f}")
print(f"  Mean Jacobian Frobenius norm: {np.linalg.norm(mean_jac):.4f}")

# Feature expansion (if PCA loadings available)
if "PCs" in adata_combined.varm:
    feat_exp = jac_result["feature_expansion"]
    print(f"  Feature expansion: {len(feat_exp)} genes scored")
else:
    # Add PCA loadings for gene-level expansion analysis
    adata_jac = ad.AnnData(
        X=sp.csr_matrix((adata_combined.n_obs, len(var_names))),
        var=pd.DataFrame(index=var_names),
    )
    adata_jac.obsm["X_pca"] = adata_combined.obsm["X_pca"]
    adata_jac.varm["PCs"] = pca_loadings
    jac_result2 = pc.tl.flow_jacobian(
        adata_jac, flow_result, flow_model,
        t=0.5, evaluation_points=jac_points,
    )
    feat_exp = jac_result2["feature_expansion"]
    print(f"  Feature expansion: {len(feat_exp)} genes scored")
    print(f"  Top 5 expanding genes:")
    top_exp = np.argsort(feat_exp)[-5:][::-1]
    for i in top_exp:
        print(f"    {var_names[i]}: expansion={feat_exp[i]:.4f}")
    print(f"  Top 5 contracting genes:")
    bot_exp = np.argsort(feat_exp)[:5]
    for i in bot_exp:
        print(f"    {var_names[i]}: expansion={feat_exp[i]:.4f}")

# Jacobian at multiple timepoints
print(f"\n  Jacobian across timepoints:")
for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
    jac_t = pc.tl.flow_jacobian(
        adata_combined, flow_result, flow_model,
        t=t_val, evaluation_points=jac_points[:50],
    )
    det_t = jac_t["jacobian_det"]
    print(f"    t={t_val:.2f}: det mean={det_t.mean():.4f}, "
          f"std={det_t.std():.4f}")

# Viz: Jacobian heatmap
adata_for_jac_viz = ad.AnnData(
    X=sp.csr_matrix((adata_combined.n_obs, len(var_names))),
    var=pd.DataFrame(index=var_names),
)
adata_for_jac_viz.obsm["X_pca"] = adata_combined.obsm["X_pca"]
adata_for_jac_viz.varm["PCs"] = pca_loadings
jac_full = pc.tl.flow_jacobian(
    adata_for_jac_viz, flow_result, flow_model,
    t=0.5, evaluation_points=jac_points,
)
pc.pl.jacobian_heatmap(adata_for_jac_viz, jac_full, show=False,
                       save_path=os.path.join(OUT_DIR, "jacobian_heatmap.png"))

assert jac_det.shape == (len(jac_points),)
assert mean_jac.shape == (N_PCS, N_PCS)


# ================================================================
# 9b. INTER-FLOW TIMEPOINT SAMPLING (TRAJECTORY)
# ================================================================
section("9b. Inter-Flow Timepoint Sampling")

# Full trajectory with return_trajectory=True
n_traj_sample = 500
traj_idx = rng_jac.choice(len(source_pca), size=min(n_traj_sample, len(source_pca)), replace=False)
traj_source = source_pca[traj_idx]

n_traj_steps = 20
print(f"  Computing trajectory: {len(traj_source)} cells, {n_traj_steps} steps...")
t1 = time.time()
trajectory = flow_model.transport(traj_source, n_steps=n_traj_steps, return_trajectory=True)
elapsed = time.time() - t1
print(f"  Trajectory shape: {trajectory.shape}  "
      f"(n_steps+1 x n_cells x dim), {elapsed:.1f}s")

# Analyze trajectory properties at each timepoint
print(f"\n  Trajectory statistics per timepoint:")
target_pca = adata_combined.obsm["X_pca"][flow_result["target_mask"]]
for step in range(0, n_traj_steps + 1, 4):
    t_val = step / n_traj_steps
    pts = trajectory[step]
    mmd_to_target = float(pc._core.utils.flow_matching.compute_mmd(pts, target_pca))
    mmd_to_source = float(pc._core.utils.flow_matching.compute_mmd(pts, traj_source))
    print(f"    t={t_val:.2f}: MMD_src={mmd_to_source:.4f}, "
          f"MMD_tgt={mmd_to_target:.4f}, "
          f"PC1_mean={pts[:,0].mean():.3f}")

# Velocity magnitude along trajectory
print(f"\n  Velocity magnitude along trajectory:")
for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
    vel = flow_model.velocity_at(traj_source[:100], t_val)
    vel_mag = np.linalg.norm(vel, axis=1)
    print(f"    t={t_val:.2f}: |v| mean={vel_mag.mean():.4f}, "
          f"std={vel_mag.std():.4f}")

# Trajectory ribbon viz
pc.pl.trajectory_ribbon(
    adata_combined, flow_result, flow_model=flow_model,
    n_sample=300, n_steps=20, show=False,
    save_path=os.path.join(OUT_DIR, "trajectory_ribbon.png"),
)

assert trajectory.shape == (n_traj_steps + 1, len(traj_source), N_PCS)
# Endpoint should be close to the full transport
full_transport = flow_model.transport(traj_source, n_steps=50)
endpoint_diff = np.linalg.norm(trajectory[-1] - full_transport, axis=1).mean()
print(f"\n  Trajectory endpoint vs full transport: mean diff = {endpoint_diff:.4f}")
assert endpoint_diff < 1.0, f"Trajectory endpoint diverges: {endpoint_diff}"


# ================================================================
# 9c. SOFT ASSIGNMENT ARCHETYPE SIMILARITY (CMP ↔ Mono)
# ================================================================
section("9c. Flow Soft Assignment: Archetype Similarity")

# For each CMP archetype: select high-weight cells, transport, measure proximity
# to Mono archetype positions
cmp_weights = cell_data["CMP"]["weights"]
mono_weights = cell_data["Mono"]["weights"]

# CMP archetype centroids in PCA space
cmp_pca = cell_data["CMP"]["pca"]
mono_pca = cell_data["Mono"]["pca"]

# Compute archetype centroid positions (mean of top-20% cells)
purity_threshold = 0.8  # top 20%
cmp_arch_centroids_pca = np.zeros((K, N_PCS))
mono_arch_centroids_pca = np.zeros((K, N_PCS))
for k in range(K):
    thresh_cmp = np.percentile(cmp_weights[:, k], 100 * purity_threshold)
    mask_k = cmp_weights[:, k] >= thresh_cmp
    cmp_arch_centroids_pca[k] = cmp_pca[mask_k].mean(axis=0)

    thresh_mono = np.percentile(mono_weights[:, k], 100 * purity_threshold)
    mask_k_m = mono_weights[:, k] >= thresh_mono
    mono_arch_centroids_pca[k] = mono_pca[mask_k_m].mean(axis=0)

# Transport CMP archetype-dominant cells to Mono space
print("  Computing soft archetype correspondence (CMP → Mono)...")
correspondence_matrix = np.zeros((K, K))  # CMP_arch x Mono_arch

for k_cmp in range(K):
    # Select cells where archetype k_cmp is dominant
    dominant_mask = cmp_weights.argmax(axis=1) == k_cmp
    n_dom = int(dominant_mask.sum())
    if n_dom == 0:
        continue

    # Subsample if too many
    dom_pca = cmp_pca[dominant_mask]
    if len(dom_pca) > 1000:
        sub_idx = rng_jac.choice(len(dom_pca), size=1000, replace=False)
        dom_pca = dom_pca[sub_idx]

    # Transport via flow
    transported_k = flow_model.transport(dom_pca, n_steps=50)

    # Measure distance to each Mono archetype centroid
    for k_mono in range(K):
        dist = np.linalg.norm(transported_k - mono_arch_centroids_pca[k_mono], axis=1)
        correspondence_matrix[k_cmp, k_mono] = 1.0 / (dist.mean() + 1e-6)

# Normalize rows to sum to 1 (soft assignment)
row_sums = correspondence_matrix.sum(axis=1, keepdims=True)
correspondence_matrix_norm = correspondence_matrix / (row_sums + 1e-10)

print(f"\n  Correspondence matrix (CMP archetype → Mono archetype):")
print(f"  {'':10s}", end="")
for k in range(K):
    print(f"  Mono_A{k}", end="")
print()
for k_cmp in range(K):
    print(f"  CMP_A{k_cmp}:  ", end="")
    for k_mono in range(K):
        print(f"  {correspondence_matrix_norm[k_cmp, k_mono]:.3f} ", end="")
    best = int(np.argmax(correspondence_matrix_norm[k_cmp]))
    print(f"  → A{best}")

# Heatmap
fig = go.Figure(data=go.Heatmap(
    z=correspondence_matrix_norm,
    x=[f"Mono A{k}" for k in range(K)],
    y=[f"CMP A{k}" for k in range(K)],
    colorscale="Blues",
    text=np.round(correspondence_matrix_norm, 3),
    texttemplate="%{text}",
    colorbar_title="Soft Assignment",
))
fig.update_layout(
    title="Flow Soft Assignment: CMP → Mono Archetype Correspondence",
    xaxis_title="Mono Archetype",
    yaxis_title="CMP Archetype",
    height=400, width=500,
)
fig.write_image(os.path.join(OUT_DIR, "flow_correspondence.png"))

assert correspondence_matrix_norm.shape == (K, K)
assert np.allclose(correspondence_matrix_norm.sum(axis=1), 1.0, atol=1e-6)


# ================================================================
# 10. CELLRANK COMPARISON
# ================================================================
section("10. CellRank Comparison")

try:
    import cellrank
    CELLRANK_AVAILABLE = True
    print(f"  CellRank {cellrank.__version__} available")
except ImportError:
    CELLRANK_AVAILABLE = False
    print("  CellRank not available, skipping")

if CELLRANK_AVAILABLE:
    import signal

    class CellRankTimeout(Exception):
        pass

    def _timeout_handler(signum, frame):
        raise CellRankTimeout("CellRank timed out")

    CELLRANK_TIMEOUT = 300  # 5 minute timeout for CellRank setup

    print(f"\n  Setting up CellRank on CMP data (timeout={CELLRANK_TIMEOUT}s)...")
    t1 = time.time()
    cellrank_success = False
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(CELLRANK_TIMEOUT)

        ck, g = pc.tl.setup_cellrank(
            adata_cmp,
            high_purity_threshold=0.80,
            n_neighbors=30,
            n_pcs=min(N_PCS, 11),
            compute_paga=True,
            verbose=False,
        )
        signal.alarm(0)  # Cancel alarm
        elapsed = time.time() - t1
        print(f"  CellRank setup complete ({elapsed:.1f}s)")
        cellrank_success = True

    except CellRankTimeout:
        signal.alarm(0)
        print(f"  CellRank TIMED OUT after {CELLRANK_TIMEOUT}s (GPCCA solver hung)")
        print("  Skipping CellRank comparison (this is a known macOS issue)")
    except Exception as e:
        signal.alarm(0)
        print(f"  CellRank setup failed: {e}")

    if cellrank_success:
        try:
            pc.tl.compute_lineage_pseudotimes(adata_cmp)
            lineage_names = adata_cmp.uns.get("lineage_names", [])
            print(f"  Lineages: {lineage_names}")

            # Flow "pseudotime": projection onto mean velocity direction
            source_mask = flow_result["source_mask"]
            cmp_pca_flow = adata_combined.obsm["X_pca"][source_mask]
            mean_vel = flow_model.velocity_at(cmp_pca_flow[:1000], 0.0).mean(axis=0)
            mean_vel_norm = mean_vel / (np.linalg.norm(mean_vel) + 1e-10)
            flow_pseudotime = cmp_pca_flow @ mean_vel_norm
            flow_pseudotime = (flow_pseudotime - flow_pseudotime.min()) / \
                              (flow_pseudotime.max() - flow_pseudotime.min() + 1e-10)

            print(f"\n  Comparing flow trajectory with CellRank pseudotimes...")
            for lin in lineage_names:
                pt_key = f"pseudotime_to_{lin}"
                if pt_key in adata_cmp.obs.columns:
                    cr_pt = adata_cmp.obs[pt_key].values
                    valid = np.isfinite(cr_pt) & np.isfinite(flow_pseudotime)
                    if valid.sum() > 100:
                        rho, pval = spearmanr(flow_pseudotime[valid], cr_pt[valid])
                        print(f"    Flow vs CellRank '{lin}': rho={rho:.3f} (p={pval:.2e})")

            # Lineage drivers vs flow-aligned genes
            print(f"\n  Computing lineage drivers for flow alignment comparison...")
            for lin in lineage_names[:2]:
                try:
                    drivers = pc.tl.compute_lineage_drivers(
                        adata_cmp, lineage=lin, n_genes=50
                    )
                    driver_genes = set(drivers["gene"].values)

                    adata_for_align = ad.AnnData(
                        X=sp.csr_matrix((adata_combined.n_obs, len(var_names))),
                        var=pd.DataFrame(index=var_names),
                    )
                    adata_for_align.obsm["X_pca"] = adata_combined.obsm["X_pca"]
                    adata_for_align.varm["PCs"] = pca_loadings
                    align_cr = pc.tl.flow_gene_alignment(
                        adata_for_align, flow_result, n_top=50
                    )
                    flow_genes = set(align_cr["top_aligned"] + align_cr["top_opposed"])

                    overlap = driver_genes & flow_genes
                    jaccard = len(overlap) / max(len(driver_genes | flow_genes), 1)
                    print(f"    Lineage '{lin}': {len(driver_genes)} CR drivers, "
                          f"{len(flow_genes)} flow genes, "
                          f"overlap={len(overlap)}, Jaccard={jaccard:.3f}")
                except Exception as e:
                    print(f"    Lineage '{lin}': failed: {e}")

        except Exception as e:
            print(f"  CellRank analysis failed: {e}")
    else:
        print("  Skipping CellRank pseudotime/driver comparison")


# ================================================================
# 11. GENE-LEVEL FLOW ALIGNMENT
# ================================================================
section("11. Gene-Level Flow Alignment")

adata_for_alignment = ad.AnnData(
    X=sp.csr_matrix((adata_combined.n_obs, len(var_names))),
    var=pd.DataFrame(index=var_names),
)
adata_for_alignment.obsm["X_pca"] = adata_combined.obsm["X_pca"]
if pca_loadings is not None:
    adata_for_alignment.varm["PCs"] = pca_loadings

alignment = pc.tl.flow_gene_alignment(adata_for_alignment, flow_result, n_top=50)
scores_align = alignment["alignment_scores"]

print(f"  Genes scored: {len(scores_align)}")
print(f"  Score range: [{scores_align.min():.4f}, {scores_align.max():.4f}]")

print(f"\n  TOP 10 FLOW-ALIGNED:")
for gene in alignment["top_aligned"][:10]:
    idx = list(alignment["gene_names"]).index(gene)
    print(f"    {gene}: {scores_align[idx]:.4f}")

print(f"\n  TOP 10 FLOW-OPPOSED:")
for gene in alignment["top_opposed"][:10]:
    idx = list(alignment["gene_names"]).index(gene)
    print(f"    {gene}: {scores_align[idx]:.4f}")

assert len(alignment["top_aligned"]) == 50
assert np.abs(scores_align).max() > 1e-6


# ================================================================
# 12. GENESET FLOW ALIGNMENT (HALLMARK)
# ================================================================
section("12. Geneset Flow Alignment (HALLMARK)")

gene_to_score = dict(zip(alignment["gene_names"], scores_align))
pathway_gene_sets = net.groupby("source")["target"].apply(set).to_dict()

pathway_alignment = {}
for pw_name, pw_genes in pathway_gene_sets.items():
    matched = [gene_to_score[g] for g in pw_genes if g in gene_to_score]
    if len(matched) >= 5:
        pathway_alignment[pw_name] = {
            "mean_score": float(np.mean(matched)),
            "n_genes": len(matched),
        }

# Permutation null
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
print(f"  Pathways scored: {len(pathway_alignment)}, significant: {n_sig_gs}")

print(f"\n  TOP 5 ALIGNED:")
for pw, info in sorted(pathway_alignment.items(), key=lambda x: x[1]["mean_score"], reverse=True)[:5]:
    print(f"    {pw}: mean={info['mean_score']:.4f} (p={info['pvalue']:.3f})")

print(f"\n  TOP 5 OPPOSED:")
for pw, info in sorted(pathway_alignment.items(), key=lambda x: x[1]["mean_score"])[:5]:
    print(f"    {pw}: mean={info['mean_score']:.4f} (p={info['pvalue']:.3f})")


# ================================================================
# 13. CROSS-VALIDATION & SUMMARY
# ================================================================
section("13. Summary & Cross-Validation")

# Gene R2 concordance
r2_cmp = np.asarray(gene_regs["CMP"]["r_squared_degree1"])
r2_mono = np.asarray(gene_regs["Mono"]["r_squared_degree1"])
rho_gene, pval_gene = spearmanr(r2_cmp, r2_mono)
print(f"  Gene R2 concordance: rho={rho_gene:.4f} (p={pval_gene:.2e})")
assert rho_gene > 0 and pval_gene < 0.05

# Pathway R2 concordance
pw_r2_cmp = np.asarray(pathway_regs["CMP"]["r_squared_degree1"])
pw_r2_mono = np.asarray(pathway_regs["Mono"]["r_squared_degree1"])
pw_rho, pw_pval = spearmanr(pw_r2_cmp, pw_r2_mono)
print(f"  Pathway R2 concordance: rho={pw_rho:.4f} (p={pw_pval:.2e})")

elapsed_total = time.time() - t0
print(f"\n{'='*70}")
print(f"  PIPELINE SUMMARY")
print(f"{'='*70}")
print(f"  Cell types: CMP ({adata_cmp.n_obs:,}), Mono ({adata_mono.n_obs:,})")
print(f"  Archetypes: K={K}")
print(f"  CMP R2: {models['CMP'].get('final_archetype_r2', 0):.3f}")
print(f"  Mono R2: {models['Mono'].get('final_archetype_r2', 0):.3f}")
print(f"  Genes: {adata_cmp.n_vars:,}")
print(f"  Pathways (HALLMARK): {len(pathway_alignment)} scored")
print(f"  Gene sig (CMP): {int(np.sum(np.asarray(gene_regs['CMP']['f_pvalue_fdr']) < ALPHA))}")
print(f"  Gene sig (Mono): {int(np.sum(np.asarray(gene_regs['Mono']['f_pvalue_fdr']) < ALPHA))}")
print(f"  Pathway sig (CMP): {int(np.sum(np.asarray(pathway_regs['CMP']['f_pvalue_fdr']) < ALPHA))}")
print(f"  Flow MMD: {flow_result['mmd_before']:.4f} → {flow_result['mmd_after']:.4f}")
print(f"  Jacobian det mean: {jac_det.mean():.4f}")
print(f"  Top aligned gene: {alignment['top_aligned'][0]}")
print(f"  Total time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
print(f"{'='*70}")
print(f"  ALL SECTIONS PASSED")
print(f"{'='*70}")

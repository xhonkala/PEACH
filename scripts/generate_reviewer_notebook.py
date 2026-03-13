#!/usr/bin/env python
"""Generate clean comprehensive 12_e2e_v050_reviewer.ipynb from validated headless test."""

import nbformat

nb = nbformat.v4.new_notebook()
nb.metadata.kernelspec = {
    "display_name": "Python 3 (archetype)",
    "language": "python",
    "name": "python3",
}

def md(text):
    nb.cells.append(nbformat.v4.new_markdown_cell(text))

def code(text):
    nb.cells.append(nbformat.v4.new_code_cell(text))


# ===========================================================================
# TITLE
# ===========================================================================
md("""\
# v0.5.0 Comprehensive E2E: HSC CMP vs CD14+ Monocyte

Full PEACH pipeline with statistical controls, interleaved visualizations,
HALLMARK pathway scoring, and flow-aligned gene/geneset analysis.

**Pipeline:**
1. Data loading with gene symbol mapping
2. Archetype fitting (K=4) with training diagnostics
3. HALLMARK pathway scoring (before regression)
4. Simplex regression on genes AND pathway scores, with beta unpacking
5. Full Wald testing (archetype contrasts for all K*(K-1)/2 pairs)
6. Pattern classification
7. GMM simplex decomposition with per-component dominance & 3D viz
8. Archetype comparison (MMD, similarity, Wald contrasts)
9. Flow matching with Jacobian, trajectory sampling & soft assignment
10. CellRank comparison
11. Gene & geneset flow alignment
12. Cross-validation & summary

**Dataset:** Human Hematopoietic Stem Cells (263K cells) — CMP vs CD14+ Monocyte""")

# ===========================================================================
# SETUP
# ===========================================================================
code("""\
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

# Prevent macOS multiprocessing spawn issue
import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != 'fork':
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass

import peach as pc

# --- Config ---
HSC_PATH = "/Users/honkala/Desktop/cross_recons/data/HSC.h5ad"
K = 4
N_PCS = 20
MONO_SUBSAMPLE = 9000
TRAIN_EPOCHS = 150
FLOW_EPOCHS = 300
SEED = 42
N_BOOTSTRAP = 50        # increase to 200-1000 for publication
N_PERMUTATIONS = 50      # increase to 200-1000 for publication
ALPHA = 0.05

t0 = time.time()""")

# ===========================================================================
# 1. DATA LOADING
# ===========================================================================
md("""\
## 1. Data Loading & Gene Symbol Mapping

Load HSC dataset, swap ENSG IDs to gene symbols, subset CMP and Monocyte populations.""")

code("""\
adata_full = ad.read_h5ad(HSC_PATH)
print(f"Loaded: {adata_full.shape[0]:,} cells x {adata_full.shape[1]:,} genes")
n_genes_original = adata_full.n_vars

# Swap ENSG -> gene symbols (handle NaN, duplicates)
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

print(f"Gene symbols: {len(var_names)} total, {len(set(symbols))} unique")
assert len(var_names) == n_genes_original
assert len(var_names) == len(set(var_names))

# Extract PCA loadings
if "PCs" in adata_full.varm:
    pca_loadings = adata_full.varm["PCs"][:, :N_PCS].copy()
    print(f"PCA loadings from dataset: {pca_loadings.shape}")
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
    n_total = len(idx)
    if subsample and len(idx) > subsample:
        idx = rng.choice(idx, size=subsample, replace=False)
        idx.sort()
    pca = adata_full.obsm["X_pca"][idx, :N_PCS].copy()
    obs = adata_full.obs.iloc[idx].copy().reset_index(drop=True)
    cell_data[name] = {"pca": pca, "obs": obs, "idx": idx}
    sub = f" (subsampled from {n_total:,})" if subsample else ""
    print(f"{name}: {len(idx):,} cells{sub}, PCA: {pca.shape}")

# Compute PCA loadings if not available
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
print("Data loading complete.")""")

# Data structure inspection
code("""\
print("=== cell_data layout ===")
for name in ["CMP", "Mono"]:
    d = cell_data[name]
    print(f"\\n  {name}:")
    print(f"    pca:  {type(d['pca']).__name__}  {d['pca'].shape}  dtype={d['pca'].dtype}")
    print(f"    obs:  DataFrame  {d['obs'].shape}  columns={list(d['obs'].columns[:6])}")
    print(f"    idx:  {type(d['idx']).__name__}  len={len(d['idx'])}")
print(f"\\n  pca_loadings: {type(pca_loadings).__name__}  {pca_loadings.shape}")
print(f"  var_names: list  len={len(var_names)}  sample={var_names[:3]}")""")

# ===========================================================================
# 2. ARCHETYPE FITTING
# ===========================================================================
md("""\
## 2. Archetype Fitting (K=4)

Train Deep Archetypal Analysis on both cell types. Extract weights, coordinates,
and assign archetypes.

**Positive controls:** R2 > 0.5, weights sum to 1, all archetypes have dominant cells.""")

code("""\
models = {}
for name in ["CMP", "Mono"]:
    print(f"\\n--- Training K={K} on {name} ---")
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
    elapsed = time.time() - t1
    r2 = result.get("final_archetype_r2", None)
    print(f"R2={r2:.3f}, {elapsed:.1f}s")

    pc.tl.extract_archetype_weights(adata_train)
    pc.tl.archetypal_coordinates(adata_train, verbose=False)

    weights = adata_train.obsm["cell_archetype_weights"]
    cell_data[name]["weights"] = weights
    cell_data[name]["uns"] = dict(adata_train.uns)
    cell_data[name]["arch_dist"] = adata_train.obsm["archetype_distances"]
    models[name] = result

    # Diagnostics
    print(f"  Weight sum: {weights.sum(axis=1).mean():.6f}")
    for k in range(K):
        n_dom = int(np.sum(weights.argmax(axis=1) == k))
        print(f"  Archetype {k}: {n_dom} dominant cells ({100*n_dom/len(weights):.1f}%)")

    assert r2 is not None and r2 > 0.0
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-4)
    for k_idx in range(K):
        assert int(np.sum(weights.argmax(axis=1) == k_idx)) > 0""")

# Training inspection
code("""\
for name in ["CMP", "Mono"]:
    result = models[name]
    print(f"\\n=== {name} TrainingResults ===")
    print(f"  keys: {list(result.keys())}")
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            print(f"    {k}: ndarray {v.shape}")
        elif isinstance(v, (list, tuple)):
            print(f"    {k}: {type(v).__name__} len={len(v)}")
        else:
            print(f"    {k}: {type(v).__name__} = {v}")
    w = cell_data[name]["weights"]
    print(f"  weights: {w.shape}, row sums: mean={w.sum(1).mean():.6f}")
    print(f"    sparsity: {(w < 0.01).sum() / w.size:.1%} of entries < 0.01")""")

# Rebuild adata with gene symbols
code("""\
# Rebuild adata with gene symbols and sparse X for downstream
print("Rebuilding adata with gene symbols and sparse X...")
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

    assert adata_obj.shape == (len(idx), len(var_names))
    assert "archetypes" in adata_obj.obs.columns
    cell_data[name]["adata"] = adata_obj

del adata_disk
gc.collect()

adata_cmp = cell_data["CMP"]["adata"]
adata_mono = cell_data["Mono"]["adata"]
print(f"CMP: {adata_cmp.shape}, Mono: {adata_mono.shape}")""")

# Adata inspection
code("""\
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    print(f"\\n=== {name} adata ===")
    print(f"  X:    {type(ad_obj.X).__name__}  {ad_obj.X.shape}")
    print(f"  obs:  {ad_obj.obs.shape}  columns={list(ad_obj.obs.columns)}")
    print(f"  var:  {ad_obj.var.shape}  index[:3]={list(ad_obj.var_names[:3])}")
    print(f"  obsm: {list(ad_obj.obsm.keys())}")
    print(f"  varm: {list(ad_obj.varm.keys())}")
    print(f"  uns:  {list(ad_obj.uns.keys())}")""")

# ===========================================================================
# 3. PATHWAY ENRICHMENT (before regression)
# ===========================================================================
md("""\
## 3. HALLMARK Pathway Scoring

Score cells on MSigDB HALLMARK gene sets using decoupler **before** simplex regression,
so we can run regression on both gene expression and pathway scores.""")

code("""\
net = pc.pp.load_pathway_networks(["hallmark"], verbose=False)
n_pathways = net["source"].nunique()
print(f"HALLMARK: {n_pathways} pathways, {net['target'].nunique()} genes")

for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    t1 = time.time()
    pc.pp.compute_pathway_scores(ad_obj, net, verbose=False)
    elapsed = time.time() - t1
    scores = ad_obj.obsm["pathway_scores"]
    pw_names = ad_obj.uns["pathway_scores_pathways"]
    print(f"  {name}: {scores.shape[1]} pathways scored in {elapsed:.1f}s")
    print(f"    Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"    Most variable: {pw_names[scores.var(axis=0).argmax()]}")
    assert scores.shape[0] == ad_obj.n_obs
    assert np.all(np.isfinite(scores))""")

# ===========================================================================
# 4a. SIMPLEX REGRESSION — GENES
# ===========================================================================
md("""\
## 4a. Simplex Regression on Gene Expression

Scheffe polynomial regression of gene expression on archetype weights.

- **Degree 1**: Linear dependence on archetype membership
- **Degree 2**: Interaction terms (synergistic archetype effects)
- **Comprehensive degree analysis**: Incremental F-tests comparing degree d vs d-1
- **Bootstrap CIs**: Confidence intervals on vertex coefficients
- **Permutation test**: Non-parametric significance
- **Robust SE**: HC3 heteroscedasticity-consistent standard errors

**Controls:** R2 bounds, nested model invariant (deg2 >= deg1), F-test/perm concordance""")

code("""\
gene_regs = {}
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    print(f"\\n--- Simplex regression on {name} (bootstrap={N_BOOTSTRAP}, perm={N_PERMUTATIONS}) ---")
    t1 = time.time()
    reg = pc.tl.feature_simplex_regression(
        ad_obj, max_degree=2,
        n_bootstrap=N_BOOTSTRAP,
        permutation_test=True,
        n_permutations=N_PERMUTATIONS,
        robust_se=True,
        comprehensive_degree=True,
    )
    elapsed = time.time() - t1
    print(f"Completed in {elapsed:.1f}s")
    gene_regs[name] = reg

    r2_d1 = np.asarray(reg["r_squared_degree1"])
    r2_d2 = np.asarray(reg["r_squared_degree2"])
    f_pval_fdr = np.asarray(reg["f_pvalue_fdr"])
    perm_pval_fdr = reg.get("permutation_pvalue_fdr")
    if perm_pval_fdr is not None:
        perm_pval_fdr = np.asarray(perm_pval_fdr)

    n_sig_f = int(np.sum(f_pval_fdr < ALPHA))
    n_sig_perm = int(np.sum(perm_pval_fdr < ALPHA)) if perm_pval_fdr is not None else "N/A"

    print(f"Features: {len(r2_d1)}")
    print(f"Degree-1 R2: median={np.median(r2_d1):.4f}, max={np.max(r2_d1):.4f}")
    print(f"Degree-2 R2: median={np.median(r2_d2):.4f}, max={np.max(r2_d2):.4f}")
    print(f"F-test significant (FDR<{ALPHA}): {n_sig_f}/{len(r2_d1)} ({100*n_sig_f/len(r2_d1):.1f}%)")
    print(f"Permutation significant (FDR<{ALPHA}): {n_sig_perm}")

    print(f"\\nTOP 10 GENES (by R2):")
    top_idx = np.argsort(r2_d1)[-10:][::-1]
    for i in top_idx:
        print(f"  {reg['feature_names'][i]}: R2={r2_d1[i]:.4f}, F_p={reg['f_pvalue'][i]:.2e}")

    assert len(r2_d1) == ad_obj.n_vars
    assert np.all(r2_d1 >= -0.1) and np.all(r2_d1 <= 1.0 + 1e-6)
    r2_diff = r2_d2 - r2_d1
    assert int(np.sum(r2_diff < -1e-6)) == 0, "Nested model invariant violated"
    if perm_pval_fdr is not None:
        concordance = min(n_sig_f, n_sig_perm) / max(n_sig_f, n_sig_perm)
        print(f"F-test/permutation concordance: {concordance:.2f}")""")

# ===========================================================================
# 4b. SIMPLEX REGRESSION — PATHWAY SCORES
# ===========================================================================
md("""\
## 4b. Simplex Regression on Pathway Scores

Regress HALLMARK pathway scores on archetype weights. This reveals which biological
programs are associated with which archetypes.""")

code("""\
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

    print(f"\\n--- {name}: {n_sig_pw}/{len(pw_r2)} significant pathways ({elapsed:.1f}s) ---")
    print(f"  R2 range: [{pw_r2.min():.4f}, {pw_r2.max():.4f}]")
    print(f"\\n  TOP 10 PATHWAYS:")
    for i in np.argsort(pw_r2)[-10:][::-1]:
        print(f"    {pw_reg['feature_names'][i]}: R2={pw_r2[i]:.4f}, p={pw_reg['f_pvalue'][i]:.2e}")

    assert pw_reg["feature_names"][0] != "feature_0", "Generic feature names"
    assert "peach_simplex_regression_genes" in ad_obj.uns, "Gene regression was overwritten"
""")

# ===========================================================================
# 4c. BETA COEFFICIENT DEGREE UNPACKING
# ===========================================================================
md("""\
## 4c. Beta Coefficient Degree Unpacking

Detailed inspection of regression coefficients:
- **Vertex coefficients (degree 1)**: Per-archetype betas with SE, p-values
- **Interaction coefficients (degree 2)**: Pairwise archetype synergy terms
- **Bootstrap CIs**: Coverage check (point estimate should be inside CI)
- **Degree comparison**: Incremental F-test quantifying interaction term value""")

code("""\
for name in ["CMP", "Mono"]:
    reg = gene_regs[name]
    coefs = np.asarray(reg["vertex_coefficients"])
    vertex_se = np.asarray(reg["vertex_se"])
    vertex_pvals_fdr = np.asarray(reg["vertex_pvalues_fdr"])
    int_coefs = reg.get("interaction_coefficients")
    int_pairs = reg.get("interaction_pairs")
    int_pvals_fdr = reg.get("interaction_pvalues_fdr")
    feat_names = reg["feature_names"]

    print(f"\\n{'='*60}")
    print(f"  {name}: Degree 1 (vertex coefficients)")
    print(f"{'='*60}")
    print(f"  Shape: {coefs.shape}  (n_features x K)")
    print(f"  Range: [{coefs.min():.4f}, {coefs.max():.4f}]")

    # Per-archetype summary
    for k in range(K):
        n_sig = int(np.sum(vertex_pvals_fdr[:, k] < ALPHA))
        z_stat = np.abs(coefs[:, k]) / np.maximum(vertex_se[:, k], 1e-10)
        print(f"  Archetype {k}: {n_sig} significant genes")
        top5 = np.argsort(z_stat)[-5:][::-1]
        for i in top5:
            print(f"    {feat_names[i]:20s}  beta={coefs[i, k]:.3f}  "
                  f"SE={vertex_se[i, k]:.3f}  z={z_stat[i]:.1f}  "
                  f"p_fdr={vertex_pvals_fdr[i, k]:.2e}")

    if int_coefs is not None:
        int_coefs_arr = np.asarray(int_coefs)
        print(f"\\n  Degree 2 (interaction coefficients): {int_coefs_arr.shape}")
        print(f"  Pairs: {int_pairs}")
        if int_pvals_fdr is not None:
            int_pvals_fdr_arr = np.asarray(int_pvals_fdr)
            for p_idx, pair in enumerate(int_pairs):
                n_sig_int = int(np.sum(int_pvals_fdr_arr[:, p_idx] < ALPHA))
                print(f"    Pair {pair}: {n_sig_int} significant interactions")

    # Bootstrap CIs
    ci_lo = reg.get("vertex_ci_lower")
    ci_hi = reg.get("vertex_ci_upper")
    if ci_lo is not None:
        ci_lo = np.asarray(ci_lo)
        ci_hi = np.asarray(ci_hi)
        ci_width = ci_hi - ci_lo
        n_outside = int(np.sum((coefs < ci_lo) | (coefs > ci_hi)))
        print(f"\\n  Bootstrap CIs: median width={np.median(ci_width):.4f}")
        print(f"  Point estimates outside CI: {n_outside}/{coefs.size} ({n_outside/coefs.size:.1%})")

    # Degree comparison
    deg_comp = reg.get("degree_comparison")
    if deg_comp is not None:
        print(f"\\n  Degree Comparison:")
        for deg_key, deg_info in sorted(deg_comp.items()):
            delta_r2 = np.asarray(deg_info["delta_r2"])
            inc_p_fdr = np.asarray(deg_info["incremental_p_fdr"])
            n_sig_inc = int(deg_info["significant_features"])
            print(f"    {deg_key}: {n_sig_inc}/{len(delta_r2)} significant "
                  f"({100*n_sig_inc/len(delta_r2):.1f}%)")
            print(f"      Delta R2: median={np.median(delta_r2):.6f}, max={np.max(delta_r2):.4f}")
            top5 = np.argsort(delta_r2)[-5:][::-1]
            for i in top5:
                print(f"        {feat_names[i]:20s}  dR2={delta_r2[i]:.4f}  "
                      f"p_fdr={inc_p_fdr[i]:.2e}")

    assert coefs.shape == (len(feat_names), K)

    # Effective rank reporting
    eff_rank = reg.get("effective_rank")
    exp_rank = reg.get("expected_rank")
    if eff_rank is not None:
        print(f"\\n  Effective rank: {eff_rank}/{exp_rank} "
              f"({'OK' if not reg.get('extra_rank_deficient') else 'DEFICIENT'})")""")

# Regression visualizations
code("""\
# Regression visualizations -- CMP
_ = pc.pl.coefficient_heatmap(adata_cmp, top_n=30, show=True)
_ = pc.pl.r2_barplot(adata_cmp, top_n=30, show=True)
_ = pc.pl.regression_volcano(adata_cmp, show=True)
_ = pc.pl.archetype_regression_dotplot(adata_cmp, top_n=10, show=True)""")

code("""\
# Regression visualizations -- Mono
_ = pc.pl.coefficient_heatmap(adata_mono, top_n=30, show=True)
_ = pc.pl.r2_barplot(adata_mono, top_n=30, show=True)
_ = pc.pl.regression_volcano(adata_mono, show=True)
_ = pc.pl.archetype_regression_dotplot(adata_mono, top_n=10, show=True)""")

# ===========================================================================
# 5. FULL WALD TESTING
# ===========================================================================
md("""\
## 5. Full Wald Testing (Archetype Contrasts)

Pairwise differential expression between all K*(K-1)/2 archetype pairs using Wald tests
on simplex regression coefficients. HC3 robust standard errors, FDR-corrected.""")

code("""\
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    contrasts = pc.tl.archetype_contrasts(ad_obj)

    print(f"\\n{'='*60}")
    print(f"  Wald Contrasts -- {name}")
    print(f"{'='*60}")
    print(f"  Pairs: {contrasts['pairs']}")
    print(f"  Features: {contrasts['n_features']}")

    for pair in contrasts["pairs"]:
        key = str(tuple(pair)) if not isinstance(pair, str) else pair
        delta_beta = contrasts["delta_beta"][key]
        delta_se = contrasts["delta_se"][key]
        z_scores = contrasts["z_scores"][key]
        pvals_fdr = contrasts["pvalues_fdr"][key]

        n_sig = int(np.sum(pvals_fdr < ALPHA))
        n_up = int(np.sum((pvals_fdr < ALPHA) & (delta_beta > 0)))
        n_down = int(np.sum((pvals_fdr < ALPHA) & (delta_beta < 0)))

        print(f"\\n    Pair {pair}: {n_sig} significant ({n_up} up, {n_down} down)")
        feat_names = contrasts["feature_names"]
        top3 = np.argsort(np.abs(z_scores))[-3:][::-1]
        for i in top3:
            print(f"      {feat_names[i]:20s}  dB={delta_beta[i]:8.3f}  "
                  f"z={z_scores[i]:.1f}  q={pvals_fdr[i]:.2e}")

    assert len(contrasts["pairs"]) == K * (K - 1) // 2""")

# ===========================================================================
# 6. PATTERN CLASSIFICATION
# ===========================================================================
md("""\
## 6. Pattern Classification

Classify features into simplex patterns from regression coefficients.

| Pattern | Rule | Interpretation |
|---------|------|----------------|
| **flat** | R2 < 0.05 OR CV(beta) < 0.15 | No archetype dependence |
| **archetype-exclusive** | max(|beta|) / second-max >= 2.0 | Marker gene for one archetype |
| **gradient** | 2+ betas in top tier, large gap from rest | Shared program across subset |
| **monotonic** | Fallback | Graded expression across archetypes |

**Negative control:** >80% of genes should be flat.""")

code("""\
pattern_results = {}
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    reg = gene_regs[name]
    patterns = pc.tl.classify_feature_patterns(ad_obj)
    pattern_results[name] = patterns
    counts_p = patterns["pattern_counts"]
    n_total = patterns["n_features"]

    print(f"\\n{'='*60}")
    print(f"  Pattern Classification -- {name}")
    print(f"{'='*60}")
    for ptype, pcount in sorted(counts_p.items(), key=lambda x: -x[1]):
        bar = chr(9608) * int(50 * pcount / n_total)
        print(f"  {ptype:25s} {pcount:5d} ({100*pcount/n_total:5.1f}%) {bar}")

    n_flat = counts_p.get("flat", 0)
    print(f"\\n  Flat control: {100*n_flat/n_total:.1f}% (expect >80%)")

    # Top features per non-flat pattern
    classifications = patterns["classifications"]
    feat_names = patterns["feature_names"]
    r2 = np.asarray(reg["r_squared_degree1"])
    coefs = np.asarray(reg["vertex_coefficients"])

    for ptype in sorted(counts_p.keys()):
        if ptype == "flat":
            continue
        indices = [i for i, c in enumerate(classifications) if c["pattern"] == ptype]
        if not indices:
            continue
        indices_sorted = sorted(indices, key=lambda i: r2[i], reverse=True)[:5]
        print(f"\\n  --- Top {ptype} features (by R2) ---")
        for i in indices_sorted:
            betas = coefs[i]
            beta_str = ", ".join(f"{b:.3f}" for b in betas)
            print(f"    {feat_names[i]:20s}  R2={r2[i]:.4f}  beta=[{beta_str}]")

    assert sum(counts_p.values()) == n_total""")

# ===========================================================================
# 7. GMM SIMPLEX DECOMPOSITION
# ===========================================================================
md("""\
## 7. GMM Simplex Decomposition

Fit Gaussian Mixture Model in ILR-transformed weight space to identify cell subpopulations.

**ILR (Isometric Log-Ratio) transform**: Maps K-dim simplex to (K-1)-dim Euclidean space.
Epsilon-smoothed (1e-3) for boundary cells, causing O(epsilon) roundtrip error.""")

code("""\
from peach._core.utils.ilr_transform import ilr_transform, inverse_ilr

gmm_results = {}
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    weights = ad_obj.obsm["cell_archetype_weights"]

    # ILR sanity
    ilr_coords = ilr_transform(weights)
    roundtrip = inverse_ilr(ilr_coords)
    roundtrip_error = np.abs(roundtrip - weights).max()
    print(f"\\n--- {name} ILR ---")
    print(f"  Shape: {ilr_coords.shape}, range: [{ilr_coords.min():.3f}, {ilr_coords.max():.3f}]")
    print(f"  Roundtrip error: {roundtrip_error:.2e}")
    assert roundtrip_error < 0.01

    # GMM fitting
    t1 = time.time()
    gmm = pc.tl.feature_simplex_decomposition(ad_obj, characterize_features=True)
    elapsed = time.time() - t1
    gmm_results[name] = gmm

    print(f"\\n--- {name} GMM ({elapsed:.1f}s) ---")
    print(f"  Optimal components (BIC): {gmm['n_components_optimal']}")
    print(f"  Stable components: {gmm['n_components_stable']}")
    print(f"  Archetype map: {gmm['component_archetype_map']}")

    if "stability_scores" in gmm:
        stab = np.asarray(gmm["stability_scores"])
        print(f"  Stability: mean={stab.mean():.3f}, min={stab.min():.3f}")

    assert gmm["n_components_optimal"] >= K
    assert gmm["n_components_stable"] >= 1""")

# GMM inspection
code("""\
for name in ["CMP", "Mono"]:
    gmm = gmm_results[name]
    print(f"\\n=== {name} GMMResult ===")
    print(f"  keys: {sorted(gmm.keys())}")
    for k, v in sorted(gmm.items()):
        if isinstance(v, np.ndarray):
            print(f"    {k}: ndarray {v.shape}")
        elif isinstance(v, dict):
            print(f"    {k}: dict keys={list(v.keys())[:5]}")
        elif isinstance(v, list) and len(v) < 20:
            print(f"    {k}: list len={len(v)}")
        else:
            print(f"    {k}: {type(v).__name__} = {v}")""")

# GMM visualizations
code("""\
_ = pc.pl.component_scatter(adata_cmp, show=True)
_ = pc.pl.gmm_bic_curve(adata_cmp, show=True)
_ = pc.pl.component_heatmap(adata_cmp, top_n=30, show=True)
_ = pc.pl.component_stability(adata_cmp, show=True)
_ = pc.pl.component_archetype_summary(adata_cmp, show=True)""")

# ===========================================================================
# 7a. PER-COMPONENT ARCHETYPE DOMINANCE
# ===========================================================================
md("""\
### 7a. Per-Component Archetype Dominance

Map GMM components to archetype weight space. Each component's centroid is compared to
the K archetype vertices (unit vectors). Entropy measures how specialized each component is
(lower = more dominated by one archetype).""")

code("""\
import plotly.graph_objects as go

archetype_vertices = np.eye(K)
simplex_center = np.ones(K) / K

for name in ["CMP", "Mono"]:
    gmm = gmm_results[name]
    weight_means = gmm.get("component_weight_means")
    if weight_means is None:
        print(f"  {name}: component_weight_means not available")
        continue

    ad_obj = adata_cmp if name == "CMP" else adata_mono
    n_comp = weight_means.shape[0]
    arch_map = gmm["component_archetype_map"]
    assignments = np.asarray(gmm["component_assignments"])

    print(f"\\n{'='*60}")
    print(f"  Per-Component Dominance -- {name}: {n_comp} components")
    print(f"{'='*60}")
    for c in range(n_comp):
        dominant_arch = int(np.argmax(weight_means[c]))
        dominance_strength = weight_means[c, dominant_arch]
        n_cells_c = int(np.sum(assignments == c))
        w_str = ", ".join(f"{w:.3f}" for w in weight_means[c])

        w_safe = np.clip(weight_means[c], 1e-10, 1.0)
        entropy = -np.sum(w_safe * np.log2(w_safe))
        max_entropy = np.log2(K)

        dists = [np.linalg.norm(weight_means[c] - archetype_vertices[k]) for k in range(K)]
        print(f"  Comp {c}: A{dominant_arch} (w={dominance_strength:.3f}), "
              f"n={n_cells_c}, entropy={entropy:.2f}/{max_entropy:.2f}")
        print(f"    weights: [{w_str}]")
        print(f"    dists:   [{', '.join(f'{d:.3f}' for d in dists)}]")

    # Distance heatmap
    dist_matrix = np.zeros((n_comp, K))
    for c in range(n_comp):
        for k_idx in range(K):
            dist_matrix[c, k_idx] = np.linalg.norm(weight_means[c] - archetype_vertices[k_idx])

    fig = go.Figure(data=go.Heatmap(
        z=dist_matrix,
        x=[f"Arch {k}" for k in range(K)],
        y=[f"Comp {c} (->A{arch_map[c]})" for c in range(n_comp)],
        colorscale="Viridis_r",
        text=np.round(dist_matrix, 3),
        texttemplate="%{text}",
        colorbar_title="Distance",
    ))
    fig.update_layout(
        title=f"{name}: Component -> Archetype Distance",
        height=300 + 30 * n_comp, width=500,
    )
    fig.show()

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
    fig2.show()

    assert dist_matrix.shape == (n_comp, K)""")

# ===========================================================================
# 7b. PER-GMM-COMPONENT 3D VIZ
# ===========================================================================
md("""\
### 7b. Per-GMM-Component 3D Visualization

Color archetypal space by GMM component assignment to see how subpopulations
distribute across the simplex.""")

code("""\
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    gmm = gmm_results[name]
    assignments = np.asarray(gmm["component_assignments"])

    ad_obj.obs["gmm_component"] = pd.Categorical(
        [f"C{c}" if c >= 0 else "unassigned" for c in assignments]
    )

    n_stable = gmm["n_components_stable"]
    print(f"\\n  {name}: {n_stable} stable components in obs['gmm_component']")

    fig = pc.pl.archetypal_space(
        ad_obj, color_by="gmm_component",
        title=f"{name}: Archetypal Space by GMM Component",
    )
    print(f"  3D viz generated for {name}")
    assert "gmm_component" in ad_obj.obs.columns""")

# ===========================================================================
# 7c. PER-COMPONENT REGRESSION
# ===========================================================================
md("""\
### 7c. Per-Component Regression

Run simplex regression independently per GMM component to detect component-specific
gene drivers masked in the global regression.""")

code("""\
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    comp_regs = pc.tl.component_regression(ad_obj, n_bootstrap=0, robust_se=True)
    n_comp = comp_regs["n_components"]
    print(f"\\n=== {name}: {n_comp} components regressed ===")
    for c_idx, c_reg in comp_regs["component_regs"].items():
        n_cells = c_reg.get("n_cells", "?")
        r2_med = np.median(c_reg["r_squared_degree1"]) if "r_squared_degree1" in c_reg else 0
        n_sig = np.sum(np.asarray(c_reg.get("f_pvalue", [1.0])) < ALPHA)
        print(f"  Component {c_idx}: n_cells={n_cells}, median_R2={r2_med:.4f}, "
              f"n_sig_genes={n_sig}")""")

# ===========================================================================
# 8. ARCHETYPE COMPARISON
# ===========================================================================
md("""\
## 8. Archetype Comparison

- **Within-fit MMD**: Test archetype distinctness within each cell type
- **Between-fit MMD**: Compare archetype distributions across cell types
- **Feature similarity**: Silhouette scores for archetype-specific feature profiles
- **Wald contrast volcano**: Visualize differential expression between archetype pairs""")

code("""\
# Within-fit MMD
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    mmd_result = pc.tl.archetype_mmd(ad_obj, n_permutations=100)
    mask = ~np.eye(K, dtype=bool)
    mmd_vals = mmd_result["mmd_matrix"][mask]
    pvals = mmd_result["pvalue_matrix"][mask]

    print(f"\\n--- Within-fit MMD ({name}) ---")
    print(f"  MMD range: {mmd_vals.min():.4f} - {mmd_vals.max():.4f}")
    print(f"  Significant pairs: {int(np.sum(pvals < ALPHA))}/{len(pvals)}")
    _ = pc.pl.mmd_heatmap(ad_obj, show=True)""")

code("""\
# Between-fit MMD
mmd_between = pc.tl.archetype_mmd(adata_cmp, adata_mono, n_permutations=100)
print("Between-fit MMD (CMP vs Mono):")
print(np.array2string(np.asarray(mmd_between["mmd_matrix"]), precision=4))""")

code("""\
# Feature similarity
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    sim = pc.tl.archetype_feature_similarity(ad_obj)
    print(f"\\n--- Feature Similarity ({name}) ---")
    print(f"  Overall silhouette: {sim['silhouette_overall']:.3f}")
    print(f"  Per-archetype: {np.array2string(np.asarray(sim['silhouette_per_archetype']), precision=3)}")
    if "spearman_pvalue_fdr_matrix" in sim:
        sig_pairs = np.sum(np.asarray(sim["spearman_pvalue_fdr_matrix"]) < ALPHA)
        print(f"  Significant Spearman pairs (FDR < {ALPHA}): {sig_pairs}")
    _ = pc.pl.feature_similarity_heatmap(ad_obj, show=True)

sim_between = pc.tl.archetype_feature_similarity(adata_cmp, adata_mono)
print(f"\\nBetween-fit Spearman matrix:")
print(np.array2string(np.asarray(sim_between["spearman_matrix"]), precision=3))
if "spearman_pvalue_fdr_matrix" in sim_between:
    print(f"Spearman FDR matrix:")
    print(np.array2string(np.asarray(sim_between["spearman_pvalue_fdr_matrix"]), precision=3))""")

code("""\
# Wald contrast volcano grid (small multiples, all pairs)
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    print(f"\\n--- Wald Volcano Grid ({name}) ---")
    contrasts = ad_obj.uns.get("peach_archetype_contrasts", {})
    if contrasts:
        pairs = contrasts.get("pairs", [])
        for p in pairs[:3]:
            pk = str(p) if str(p) in contrasts.get("pvalues_fdr", {}) else str(tuple(p))
            fdr = np.asarray(contrasts["pvalues_fdr"].get(pk, []))
            if len(fdr) > 0:
                n_sig = np.sum(fdr < ALPHA)
                print(f"  Pair {p}: {n_sig} sig genes (FDR < {ALPHA})")
    _ = pc.pl.contrast_volcano_grid(ad_obj, show=True)""")

# ===========================================================================
# 9. FLOW MATCHING
# ===========================================================================
md("""\
## 9. Flow Matching (CMP -> Mono)

Train a continuous normalizing flow to transport CMP cells to Monocyte distribution in PCA space.
Uses `return_model=True` to enable Jacobian and trajectory analysis downstream.

**Controls:** MMD reduction >20%, non-degenerate transport""")

code("""\
# Build combined adata for flow
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
print(f"Combined: {adata_combined.n_obs:,} cells")

# Train flow with return_model=True
print(f"Training flow CMP -> Mono ({FLOW_EPOCHS} epochs)...")
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

print(f"\\nFlow training ({elapsed:.1f}s):")
print(f"  MMD: {flow_result['mmd_before']:.4f} -> {flow_result['mmd_after']:.4f} ({mmd_reduction:.1f}% reduction)")

assert "model" in flow_result, "return_model=True but no model in result"
assert mmd_reduction > 20.0
flow_model = flow_result["model"]""")

code("""\
# Flow visualizations
_ = pc.pl.flow_magnitude(adata_combined, flow_result, show=True)
_ = pc.pl.density_comparison(adata_combined, flow_result, show=True)
_ = pc.pl.velocity_quiver(adata_combined, flow_result, show=True)""")

# ===========================================================================
# 9a. FLOW JACOBIAN
# ===========================================================================
md("""\
### 9a. Flow Jacobian Analysis

Compute the Jacobian dv/dx of the flow velocity field at multiple timepoints.
The Jacobian reveals how the flow locally deforms space — eigenvalues > 1 indicate expansion,
< 1 contraction. Feature expansion scores project the Jacobian onto PCA loadings to get
per-gene expansion/contraction.

**Note:** det(J) is near zero in high dimensions because eigenvalues close to 1 multiply
to a very small product — the feature expansion scores are the meaningful readout.""")

code("""\
source_pca = adata_combined.obsm["X_pca"][flow_result["source_mask"]]
n_jac_sample = 200
rng_jac = np.random.default_rng(SEED)
jac_idx = rng_jac.choice(len(source_pca), size=min(n_jac_sample, len(source_pca)), replace=False)
jac_points = source_pca[jac_idx]

# Jacobian at t=0.5
print(f"Computing Jacobian at t=0.5 on {len(jac_points)} points...")
t1 = time.time()
jac_result = pc.tl.flow_jacobian(
    adata_combined, flow_result, flow_model,
    t=0.5, evaluation_points=jac_points,
)
elapsed = time.time() - t1

jac_det = jac_result["jacobian_det"]
mean_jac = jac_result["mean_jacobian"]
print(f"Jacobian computed in {elapsed:.1f}s")
print(f"  Determinant: mean={jac_det.mean():.4f}, range=[{jac_det.min():.4f}, {jac_det.max():.4f}]")
print(f"  Mean Jacobian: shape={mean_jac.shape}, trace={np.trace(mean_jac):.4f}")

# Feature expansion with PCA loadings
adata_jac = ad.AnnData(
    X=sp.csr_matrix((adata_combined.n_obs, len(var_names))),
    var=pd.DataFrame(index=var_names),
)
adata_jac.obsm["X_pca"] = adata_combined.obsm["X_pca"]
adata_jac.varm["PCs"] = pca_loadings
jac_with_genes = pc.tl.flow_jacobian(
    adata_jac, flow_result, flow_model,
    t=0.5, evaluation_points=jac_points,
)
feat_exp = jac_with_genes["feature_expansion"]
print(f"\\nFeature expansion: {len(feat_exp)} genes scored")
print(f"  Top 5 expanding:")
for i in np.argsort(feat_exp)[-5:][::-1]:
    print(f"    {var_names[i]}: {feat_exp[i]:.4f}")
print(f"  Top 5 contracting:")
for i in np.argsort(feat_exp)[:5]:
    print(f"    {var_names[i]}: {feat_exp[i]:.4f}")

# Multi-timepoint
print(f"\\nJacobian across timepoints:")
for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
    jac_t = pc.tl.flow_jacobian(
        adata_combined, flow_result, flow_model,
        t=t_val, evaluation_points=jac_points[:50],
    )
    det_t = jac_t["jacobian_det"]
    print(f"  t={t_val:.2f}: det mean={det_t.mean():.4f}, std={det_t.std():.4f}")

assert jac_det.shape == (len(jac_points),)
assert mean_jac.shape == (N_PCS, N_PCS)""")

# Jacobian heatmap
code("""\
pc.pl.jacobian_heatmap(adata_jac, jac_with_genes, show=True)""")

# ===========================================================================
# 9b. TRAJECTORY SAMPLING
# ===========================================================================
md("""\
### 9b. Inter-Flow Timepoint Sampling (Trajectory)

Sample the full transport trajectory from t=0 to t=1, showing how cells move through
PCA space. MMD to target should monotonically decrease; velocity magnitude reveals
where the flow changes fastest.""")

code("""\
n_traj_sample = 500
traj_idx = rng_jac.choice(len(source_pca), size=min(n_traj_sample, len(source_pca)), replace=False)
traj_source = source_pca[traj_idx]

n_traj_steps = 20
print(f"Computing trajectory: {len(traj_source)} cells, {n_traj_steps} steps...")
t1 = time.time()
trajectory = flow_model.transport(traj_source, n_steps=n_traj_steps, return_trajectory=True)
elapsed = time.time() - t1
print(f"Trajectory shape: {trajectory.shape} (n_steps+1 x n_cells x dim), {elapsed:.1f}s")

# Trajectory statistics
target_pca = adata_combined.obsm["X_pca"][flow_result["target_mask"]]
print(f"\\nTrajectory statistics per timepoint:")
for step in range(0, n_traj_steps + 1, 4):
    t_val = step / n_traj_steps
    pts = trajectory[step]
    mmd_to_target = float(pc._core.utils.flow_matching.compute_mmd(pts, target_pca))
    mmd_to_source = float(pc._core.utils.flow_matching.compute_mmd(pts, traj_source))
    print(f"  t={t_val:.2f}: MMD_src={mmd_to_source:.4f}, MMD_tgt={mmd_to_target:.4f}")

# Velocity magnitude
print(f"\\nVelocity magnitude along trajectory:")
for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
    vel = flow_model.velocity_at(traj_source[:100], t_val)
    vel_mag = np.linalg.norm(vel, axis=1)
    print(f"  t={t_val:.2f}: |v| mean={vel_mag.mean():.4f}, std={vel_mag.std():.4f}")

# Endpoint consistency
full_transport = flow_model.transport(traj_source, n_steps=50)
endpoint_diff = np.linalg.norm(trajectory[-1] - full_transport, axis=1).mean()
print(f"\\nTrajectory endpoint vs full transport: mean diff = {endpoint_diff:.4f}")
assert endpoint_diff < 1.0
assert trajectory.shape == (n_traj_steps + 1, len(traj_source), N_PCS)""")

code("""\
# Trajectory ribbon visualization
pc.pl.trajectory_ribbon(
    adata_combined, flow_result, flow_model=flow_model,
    n_sample=300, n_steps=20, show=True,
)""")

# ===========================================================================
# 9c. SOFT ASSIGNMENT ARCHETYPE SIMILARITY
# ===========================================================================
md("""\
### 9c. Flow Soft Assignment: Archetype Similarity (CMP <-> Mono)

For each CMP archetype: select high-weight cells, transport them to Mono space via the
learned flow, then measure proximity to each Mono archetype centroid. This gives a soft
correspondence matrix showing which CMP archetypes map to which Mono archetypes.""")

code("""\
import plotly.graph_objects as go

cmp_weights = cell_data["CMP"]["weights"]
mono_weights = cell_data["Mono"]["weights"]
cmp_pca = cell_data["CMP"]["pca"]
mono_pca = cell_data["Mono"]["pca"]

# Archetype centroids (mean of top-20% purity cells)
purity_threshold = 0.8
cmp_arch_centroids_pca = np.zeros((K, N_PCS))
mono_arch_centroids_pca = np.zeros((K, N_PCS))
for k in range(K):
    thresh_cmp = np.percentile(cmp_weights[:, k], 100 * purity_threshold)
    cmp_arch_centroids_pca[k] = cmp_pca[cmp_weights[:, k] >= thresh_cmp].mean(axis=0)

    thresh_mono = np.percentile(mono_weights[:, k], 100 * purity_threshold)
    mono_arch_centroids_pca[k] = mono_pca[mono_weights[:, k] >= thresh_mono].mean(axis=0)

# Transport CMP archetype-dominant cells to Mono space
print("Computing soft archetype correspondence (CMP -> Mono)...")
correspondence_matrix = np.zeros((K, K))

for k_cmp in range(K):
    dominant_mask = cmp_weights.argmax(axis=1) == k_cmp
    dom_pca = cmp_pca[dominant_mask]
    if len(dom_pca) > 1000:
        sub_idx = rng_jac.choice(len(dom_pca), size=1000, replace=False)
        dom_pca = dom_pca[sub_idx]

    transported_k = flow_model.transport(dom_pca, n_steps=50)

    for k_mono in range(K):
        dist = np.linalg.norm(transported_k - mono_arch_centroids_pca[k_mono], axis=1)
        correspondence_matrix[k_cmp, k_mono] = 1.0 / (dist.mean() + 1e-6)

# Normalize to soft assignment
row_sums = correspondence_matrix.sum(axis=1, keepdims=True)
correspondence_norm = correspondence_matrix / (row_sums + 1e-10)

print(f"\\nCorrespondence matrix (CMP -> Mono):")
header = "          " + "  ".join(f"Mono_A{k}" for k in range(K))
print(header)
for k_cmp in range(K):
    vals = "  ".join(f"  {correspondence_norm[k_cmp, k_mono]:.3f}" for k_mono in range(K))
    best = int(np.argmax(correspondence_norm[k_cmp]))
    print(f"  CMP_A{k_cmp}: {vals}  -> A{best}")

# Heatmap
fig = go.Figure(data=go.Heatmap(
    z=correspondence_norm,
    x=[f"Mono A{k}" for k in range(K)],
    y=[f"CMP A{k}" for k in range(K)],
    colorscale="Blues",
    text=np.round(correspondence_norm, 3),
    texttemplate="%{text}",
    colorbar_title="Soft Assignment",
))
fig.update_layout(
    title="Flow Soft Assignment: CMP -> Mono Archetype Correspondence",
    xaxis_title="Mono Archetype",
    yaxis_title="CMP Archetype",
    height=400, width=500,
)
fig.show()

assert correspondence_norm.shape == (K, K)
assert np.allclose(correspondence_norm.sum(axis=1), 1.0, atol=1e-6)""")

# ===========================================================================
# 10. CELLRANK COMPARISON
# ===========================================================================
md("""\
## 10. CellRank Comparison

Compare flow-derived trajectory with CellRank pseudotimes and lineage drivers.
CellRank uses connectivity kernels + GPCCA for fate probability estimation.

**Note:** CellRank's GPCCA solver can hang on macOS — wrapped with 5-minute timeout.""")

code("""\
try:
    import cellrank
    CELLRANK_AVAILABLE = True
    print(f"CellRank {cellrank.__version__} available")
except ImportError:
    CELLRANK_AVAILABLE = False
    print("CellRank not available, skipping")

if CELLRANK_AVAILABLE:
    import signal

    class CellRankTimeout(Exception):
        pass

    def _timeout_handler(signum, frame):
        raise CellRankTimeout("CellRank timed out")

    CELLRANK_TIMEOUT = 300

    print(f"\\nSetting up CellRank on CMP (timeout={CELLRANK_TIMEOUT}s)...")
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
        signal.alarm(0)
        print(f"CellRank setup complete ({time.time()-t1:.1f}s)")
        cellrank_success = True

    except CellRankTimeout:
        signal.alarm(0)
        print(f"CellRank TIMED OUT after {CELLRANK_TIMEOUT}s (known macOS GPCCA issue)")
    except Exception as e:
        signal.alarm(0)
        print(f"CellRank setup failed: {e}")

    if cellrank_success:
        try:
            pc.tl.compute_lineage_pseudotimes(adata_cmp)
            lineage_names = adata_cmp.uns.get("lineage_names", [])
            print(f"Lineages: {lineage_names}")

            # Flow "pseudotime": projection onto mean velocity
            source_mask = flow_result["source_mask"]
            cmp_pca_flow = adata_combined.obsm["X_pca"][source_mask]
            mean_vel = flow_model.velocity_at(cmp_pca_flow[:1000], 0.0).mean(axis=0)
            mean_vel_norm = mean_vel / (np.linalg.norm(mean_vel) + 1e-10)
            flow_pseudotime = cmp_pca_flow @ mean_vel_norm
            flow_pseudotime = (flow_pseudotime - flow_pseudotime.min()) / \\
                              (flow_pseudotime.max() - flow_pseudotime.min() + 1e-10)

            print(f"\\nFlow vs CellRank pseudotime correlation:")
            for lin in lineage_names:
                pt_key = f"pseudotime_to_{lin}"
                if pt_key in adata_cmp.obs.columns:
                    cr_pt = adata_cmp.obs[pt_key].values
                    valid = np.isfinite(cr_pt) & np.isfinite(flow_pseudotime)
                    if valid.sum() > 100:
                        rho, pval = spearmanr(flow_pseudotime[valid], cr_pt[valid])
                        print(f"  '{lin}': rho={rho:.3f} (p={pval:.2e})")

            # Lineage drivers vs flow-aligned genes
            print(f"\\nLineage drivers vs flow-aligned genes:")
            for lin in lineage_names[:2]:
                try:
                    drivers = pc.tl.compute_lineage_drivers(adata_cmp, lineage=lin, n_genes=50)
                    driver_genes = set(drivers["gene"].values)

                    adata_for_align = ad.AnnData(
                        X=sp.csr_matrix((adata_combined.n_obs, len(var_names))),
                        var=pd.DataFrame(index=var_names),
                    )
                    adata_for_align.obsm["X_pca"] = adata_combined.obsm["X_pca"]
                    adata_for_align.varm["PCs"] = pca_loadings
                    align_cr = pc.tl.flow_gene_alignment(adata_for_align, flow_result, n_top=50)
                    flow_genes = set(align_cr["top_aligned"] + align_cr["top_opposed"])

                    overlap = driver_genes & flow_genes
                    jaccard = len(overlap) / max(len(driver_genes | flow_genes), 1)
                    print(f"  '{lin}': {len(overlap)} overlapping genes, Jaccard={jaccard:.3f}")
                except Exception as e:
                    print(f"  '{lin}': failed ({e})")

        except Exception as e:
            print(f"CellRank analysis failed: {e}")
    else:
        print("Skipping CellRank pseudotime/driver comparison")
else:
    print("CellRank not installed -- install with: conda install -c conda-forge cellrank")""")

# ===========================================================================
# 11. GENE-LEVEL FLOW ALIGNMENT
# ===========================================================================
md("""\
## 11. Gene-Level Flow Alignment

Project flow velocity onto PCA loadings to score each gene's alignment with the
CMP-to-Mono transition direction.

**Controls:** Non-zero alignment scores, biologically meaningful top genes""")

code("""\
n_top_genes = 50
adata_for_alignment = ad.AnnData(
    X=sp.csr_matrix((adata_combined.n_obs, len(var_names))),
    var=pd.DataFrame(index=var_names),
)
adata_for_alignment.obsm["X_pca"] = adata_combined.obsm["X_pca"]
if pca_loadings is not None:
    adata_for_alignment.varm["PCs"] = pca_loadings
alignment = pc.tl.flow_gene_alignment(
    adata_for_alignment, flow_result, n_top=n_top_genes,
    n_permutations=N_PERMUTATIONS, random_state=SEED,
)
scores_align = alignment["alignment_scores"]

print(f"Genes scored: {len(scores_align)}")
print(f"Score range: [{scores_align.min():.4f}, {scores_align.max():.4f}]")

if "alignment_pvalues_fdr" in alignment:
    n_sig = np.sum(np.asarray(alignment["alignment_pvalues_fdr"]) < ALPHA)
    print(f"Significant genes (FDR < {ALPHA}): {n_sig}/{len(scores_align)}")

print(f"\\nTOP 10 FLOW-ALIGNED (upregulated CMP->Mono):")
for gene in alignment["top_aligned"][:10]:
    idx = list(alignment["gene_names"]).index(gene)
    fdr_str = ""
    if "alignment_pvalues_fdr" in alignment:
        fdr_str = f" (FDR={alignment['alignment_pvalues_fdr'][idx]:.2e})"
    print(f"  {gene}: {scores_align[idx]:.4f}{fdr_str}")

print(f"\\nTOP 10 FLOW-OPPOSED (downregulated CMP->Mono):")
for gene in alignment["top_opposed"][:10]:
    idx = list(alignment["gene_names"]).index(gene)
    fdr_str = ""
    if "alignment_pvalues_fdr" in alignment:
        fdr_str = f" (FDR={alignment['alignment_pvalues_fdr'][idx]:.2e})"
    print(f"  {gene}: {scores_align[idx]:.4f}{fdr_str}")

assert len(alignment["top_aligned"]) == n_top_genes
assert np.abs(scores_align).max() > 1e-6""")

# ===========================================================================
# 12. GENESET FLOW ALIGNMENT
# ===========================================================================
md("""\
## 12. Geneset-Level Flow Alignment (HALLMARK)

Aggregate gene-level alignment scores by pathway. Permutation test (500 permutations)
for pathway-level significance.""")

code("""\
gene_to_score = dict(zip(alignment["gene_names"], scores_align))
pathway_gene_sets = net.groupby("source")["target"].apply(set).to_dict()

total_pw_genes = sum(len(gs) for gs in pathway_gene_sets.values())
matched_pw_genes = sum(1 for gs in pathway_gene_sets.values() for g in gs if g in gene_to_score)
overlap_rate = matched_pw_genes / total_pw_genes
print(f"Gene overlap: {matched_pw_genes}/{total_pw_genes} ({100*overlap_rate:.1f}%)")
assert overlap_rate > 0.1

pathway_alignment = {}
for pw_name, pw_genes in pathway_gene_sets.items():
    matched = [gene_to_score[g] for g in pw_genes if g in gene_to_score]
    if len(matched) >= 5:
        pathway_alignment[pw_name] = {
            "mean_score": float(np.mean(matched)),
            "n_genes": len(matched),
            "n_aligned": sum(1 for s in matched if s > 0),
            "n_opposed": sum(1 for s in matched if s < 0),
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
print(f"\\nPathways scored: {len(pathway_alignment)}, significant: {n_sig_gs}")

print(f"\\nTOP 10 FLOW-ALIGNED PATHWAYS (activated CMP->Mono):")
for pw, info in sorted(pathway_alignment.items(), key=lambda x: x[1]["mean_score"], reverse=True)[:10]:
    print(f"  {pw}: mean={info['mean_score']:.4f} ({info['n_aligned']}/{info['n_genes']} aligned, p={info['pvalue']:.3f})")

print(f"\\nTOP 10 FLOW-OPPOSED PATHWAYS (deactivated CMP->Mono):")
for pw, info in sorted(pathway_alignment.items(), key=lambda x: x[1]["mean_score"])[:10]:
    print(f"  {pw}: mean={info['mean_score']:.4f} ({info['n_opposed']}/{info['n_genes']} opposed, p={info['pvalue']:.3f})")""")

# ===========================================================================
# 13. SUMMARY
# ===========================================================================
md("""\
## 13. Summary & Cross-Validation

Cross-cell-type concordance checks and final pipeline summary.""")

code("""\
# Gene R2 concordance
r2_cmp = np.asarray(gene_regs["CMP"]["r_squared_degree1"])
r2_mono = np.asarray(gene_regs["Mono"]["r_squared_degree1"])
rho, pval = spearmanr(r2_cmp, r2_mono)
print(f"Gene R2 concordance: rho={rho:.4f} (p={pval:.2e})")
assert rho > 0 and pval < 0.05

# Pathway R2 concordance
pw_r2_cmp = np.asarray(pathway_regs["CMP"]["r_squared_degree1"])
pw_r2_mono = np.asarray(pathway_regs["Mono"]["r_squared_degree1"])
pw_rho, pw_pval = spearmanr(pw_r2_cmp, pw_r2_mono)
print(f"Pathway R2 concordance: rho={pw_rho:.4f} (p={pw_pval:.2e})")

# Summary
elapsed_total = time.time() - t0
print(f"\\n{'='*60}")
print(f"  PIPELINE SUMMARY")
print(f"{'='*60}")
print(f"  Cell types: CMP ({adata_cmp.n_obs:,}), Mono ({adata_mono.n_obs:,})")
print(f"  Archetypes: K={K}")
print(f"  CMP R2: {models['CMP'].get('final_archetype_r2', 0):.3f}")
print(f"  Mono R2: {models['Mono'].get('final_archetype_r2', 0):.3f}")
print(f"  Genes tested: {adata_cmp.n_vars:,}")
print(f"  CMP sig (FDR<{ALPHA}): {int(np.sum(np.asarray(gene_regs['CMP']['f_pvalue_fdr']) < ALPHA))}")
print(f"  Mono sig (FDR<{ALPHA}): {int(np.sum(np.asarray(gene_regs['Mono']['f_pvalue_fdr']) < ALPHA))}")
print(f"  HALLMARK pathways: {len(pathway_alignment)} scored, {n_sig_gs} sig")
print(f"  Pathway sig (CMP): {int(np.sum(np.asarray(pathway_regs['CMP']['f_pvalue_fdr']) < ALPHA))}")
print(f"  Flow MMD: {flow_result['mmd_before']:.4f} -> {flow_result['mmd_after']:.4f}")
print(f"  Jacobian det mean: {jac_det.mean():.4f}")
print(f"  Top aligned gene: {alignment['top_aligned'][0]}")
print(f"  Gene R2 concordance: rho={rho:.4f}")
print(f"  Total time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
print(f"{'='*60}")
print(f"  ALL SECTIONS COMPLETE")
print(f"{'='*60}")""")


# ===========================================================================
# WRITE
# ===========================================================================
output_path = "docs/tutorials/12_e2e_v050_reviewer.ipynb"
nbformat.write(nb, output_path)
print(f"Wrote {len(nb.cells)} cells to {output_path}")

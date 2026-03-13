#!/usr/bin/env python
"""Train CMP and Mono archetypes and save annotated AnnDatas for downstream notebooks.

Produces two h5ad files with all shared analysis pre-computed:
  - Training (K=4, 150 epochs)
  - Weight extraction, distances, assignments
  - HALLMARK pathway scoring
  - Simplex regression on genes AND pathways (degree 2, bootstrap, permutation)
  - Wald contrasts (all K*(K-1)/2 pairs, global FDR)
  - Pattern classification
  - GMM simplex decomposition

The downstream notebooks (12a_regression, 12b_decomposition, 12c_flow) load these
and jump straight into their analyses.

Usage:
    conda run -n archetype python scripts/train_hsc_baseline.py
"""

import os
import time
import warnings
import logging

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import scipy.sparse as sp
import pandas as pd
import anndata as ad
import gc
from collections import Counter

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.disable(logging.WARNING)

import torch
torch.set_num_threads(1)

import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != "fork":
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass

import peach as pc

# ---------- Config ----------
HSC_PATH = "/Users/honkala/Desktop/cross_recons/data/HSC.h5ad"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "trained")
K = 4
N_PCS = 20
MONO_SUBSAMPLE = 9000
TRAIN_EPOCHS = 150
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

    cell_data[name]["weights"] = adata_train.obsm["cell_archetype_weights"]
    cell_data[name]["uns"] = dict(adata_train.uns)
    cell_data[name]["arch_dist"] = adata_train.obsm["archetype_distances"]

    assert r2 is not None and r2 > 0.0
    assert np.allclose(cell_data[name]["weights"].sum(axis=1), 1.0, atol=1e-4)

# Rebuild adata with sparse X and gene symbols
print("\n  Rebuilding adata with sparse X and gene symbols...")
adata_disk = ad.read_h5ad(HSC_PATH, backed="r")
adatas = {}
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
    adatas[name] = adata_obj

del adata_disk
gc.collect()

adata_cmp = adatas["CMP"]
adata_mono = adatas["Mono"]
print(f"  CMP: {adata_cmp.shape}, Mono: {adata_mono.shape}")


# ================================================================
# 3. PATHWAY ENRICHMENT
# ================================================================
section("3. Pathway Enrichment (HALLMARK)")

net = pc.pp.load_pathway_networks(["hallmark"], verbose=False)
n_pathways = net["source"].nunique()
print(f"  HALLMARK: {n_pathways} pathways")

for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    t1 = time.time()
    pc.pp.compute_pathway_scores(ad_obj, net, verbose=False)
    scores = ad_obj.obsm["pathway_scores"]
    print(f"  {name}: {scores.shape[1]} pathways scored in {time.time()-t1:.1f}s")


# ================================================================
# 4. SIMPLEX REGRESSION — GENES
# ================================================================
section("4a. Simplex Regression on Genes")

for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    print(f"\n  --- {name} ---")
    t1 = time.time()
    reg = pc.tl.feature_simplex_regression(
        ad_obj, max_degree=2,
        n_bootstrap=N_BOOTSTRAP,
        permutation_test=True,
        n_permutations=N_PERMUTATIONS,
        robust_se=True,
        comprehensive_degree=True,
    )
    r2_d1 = np.asarray(reg["r_squared_degree1"])
    n_sig = int(np.sum(np.asarray(reg["f_pvalue_fdr"]) < ALPHA))
    print(f"  {time.time()-t1:.1f}s, {n_sig}/{len(r2_d1)} significant, "
          f"median R2={np.median(r2_d1):.4f}, max R2={np.max(r2_d1):.4f}")


# ================================================================
# 4b. SIMPLEX REGRESSION — PATHWAYS
# ================================================================
section("4b. Simplex Regression on Pathway Scores")

for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    pw_names = list(ad_obj.uns["pathway_scores_pathways"])
    pw_reg = pc.tl.pathway_simplex_regression(
        ad_obj, n_bootstrap=N_BOOTSTRAP, robust_se=True,
        feature_names=pw_names,
    )
    pw_r2 = np.asarray(pw_reg["r_squared_degree1"])
    n_sig = int(np.sum(np.asarray(pw_reg["f_pvalue_fdr"]) < ALPHA))
    print(f"  {name}: {n_sig}/{len(pw_r2)} significant, max R2={np.max(pw_r2):.4f}")


# ================================================================
# 5. WALD CONTRASTS
# ================================================================
section("5. Wald Contrasts")

for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    contrasts = pc.tl.archetype_contrasts(ad_obj)
    n_pairs = len(contrasts["pairs"])
    total_sig = sum(
        int(np.sum(np.asarray(contrasts["pvalues_fdr"][str(tuple(p))]) < ALPHA))
        for p in contrasts["pairs"]
    )
    print(f"  {name}: {n_pairs} pairs, {total_sig} total significant features")


# ================================================================
# 6. PATTERN CLASSIFICATION
# ================================================================
section("6. Pattern Classification")

for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    patterns = pc.tl.classify_feature_patterns(ad_obj)
    counts_p = patterns["pattern_counts"]
    n_total = patterns["n_features"]
    non_flat = sum(v for k, v in counts_p.items() if k != "flat")
    print(f"  {name}: {non_flat}/{n_total} non-flat features")


# ================================================================
# 7. GMM SIMPLEX DECOMPOSITION
# ================================================================
section("7. GMM Simplex Decomposition")

for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    t1 = time.time()
    gmm = pc.tl.feature_simplex_decomposition(ad_obj, characterize_features=True)
    print(f"  {name}: {gmm['n_components_stable']} stable components ({time.time()-t1:.1f}s)")


# ================================================================
# SAVE
# ================================================================
section("Saving annotated AnnDatas")


def _sanitize_value(v):
    """Recursively convert non-serializable values for h5ad."""
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    if isinstance(v, (torch.nn.Module,)):
        return None  # will be removed
    if isinstance(v, dict):
        return {str(dk): _sanitize_value(dv) for dk, dv in v.items()}
    if isinstance(v, (list, tuple)):
        # h5ad needs homogeneous arrays; convert mixed lists to string arrays
        if len(v) > 0 and not all(isinstance(x, (int, float, np.integer, np.floating)) for x in v):
            try:
                arr = np.array(v)
                if arr.dtype == object:
                    return np.array([str(x) for x in v])
                return arr
            except (ValueError, TypeError):
                return np.array([str(x) for x in v])
        return v
    return v


def sanitize_for_h5ad(adata):
    """Remove non-serializable objects from uns before saving."""
    keys_to_remove = []
    for k, v in list(adata.uns.items()):
        if isinstance(v, (torch.nn.Module,)):
            keys_to_remove.append(k)
        elif hasattr(v, "__module__") and "torch" in str(getattr(v, "__module__", "")):
            keys_to_remove.append(k)
        else:
            adata.uns[k] = _sanitize_value(v)
    for k in keys_to_remove:
        del adata.uns[k]
        print(f"    Removed non-serializable uns['{k}']")


# Store config for reproducibility
config = {
    "K": K, "N_PCS": N_PCS, "TRAIN_EPOCHS": TRAIN_EPOCHS,
    "SEED": SEED, "N_BOOTSTRAP": N_BOOTSTRAP, "N_PERMUTATIONS": N_PERMUTATIONS,
    "MONO_SUBSAMPLE": MONO_SUBSAMPLE, "HSC_PATH": HSC_PATH,
}

for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    ad_obj.uns["training_config"] = config
    sanitize_for_h5ad(ad_obj)
    path = os.path.join(OUT_DIR, f"hsc_{name.lower()}_v050.h5ad")
    ad_obj.write_h5ad(path)
    print(f"  Saved {name}: {path} ({os.path.getsize(path) / 1e6:.1f} MB)")

elapsed = time.time() - t0
print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
print("  DONE")

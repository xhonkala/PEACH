"""Reconstruct step1_holdout_cache.h5ad from the existing train cache.

Uses the train PCA loadings to reproject holdout cells, then computes
barycentric archetype weights via NNLS simplex projection (no VAE model needed).
"""
import sys
import numpy as np
import scipy.sparse as sp
from scipy.optimize import nnls
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import scanpy as sc
import peach as pc

DATA_DIR = Path(__file__).parents[1] / "data" / "paper_part2"

print("Loading train cache...")
adata_train = sc.read_h5ad(DATA_DIR / "step1_cache.h5ad")
print(f"  train: {adata_train.shape}, PCs: {adata_train.varm['PCs'].shape}")
print(f"  K: {adata_train.obsm['cell_archetype_weights'].shape[1]}")

print("Loading raw holdout...")
adata_hold = sc.read_h5ad(DATA_DIR / "adata_tnbc_holdout.h5ad")
print(f"  holdout raw: {adata_hold.shape}")

# Filter holdout to same gene set as train
shared_genes = [g for g in adata_train.var_names if g in set(adata_hold.var_names)]
missing = adata_train.n_vars - len(shared_genes)
print(f"  genes present in holdout: {len(shared_genes)} / {adata_train.n_vars} ({missing} missing)")
adata_hold = adata_hold[:, shared_genes].copy()
print(f"  holdout filtered: {adata_hold.shape}")

# Reproject holdout into train PCA space using train's PCA loadings + mean
n_comps = adata_train.obsm["X_pca"].shape[1]
PCs = adata_train.varm["PCs"][:, :n_comps]   # [n_genes, n_comps_used]
X_tr = adata_train.X
if sp.issparse(X_tr):
    X_tr = X_tr.toarray()
ref_mean = np.asarray(X_tr, dtype=np.float64).mean(axis=0)

X_ho = adata_hold.X
if sp.issparse(X_ho):
    X_ho = X_ho.toarray()

# Zero out missing gene positions (already filtered to shared, so no gaps)
X_ho_centered = np.asarray(X_ho, dtype=np.float64) - ref_mean
X_pca_hold = (X_ho_centered @ PCs).astype(np.float32)
adata_hold.obsm["X_pca"] = X_pca_hold
print(f"  reprojected X_pca: {X_pca_hold.shape}")

# Transfer archetype artifacts from train
for key in ("archetype_coordinates",):
    if key in adata_train.uns:
        adata_hold.uns[key] = adata_train.uns[key]

# Compute archetype distances + coordinates
pc.tl.archetypal_coordinates(adata_hold, verbose=False)
print("  archetypal_coordinates done")

# Compute simplex weights via NNLS projection onto archetype vertices
arch_coords = np.array(adata_hold.uns["archetype_coordinates"])  # [K, dim]
cell_coords = adata_hold.obsm["X_pca"]                           # [n, dim]
K = arch_coords.shape[0]

# Augmented system for sum-to-one constraint: [arch^T; 1^T] w = [x; 1]
A = np.vstack([arch_coords.T, np.ones((1, K))])  # [dim+1, K]
weights = np.zeros((len(adata_hold), K), dtype=np.float32)
print(f"  computing NNLS weights for {len(adata_hold)} cells (K={K})...")
for i in range(len(adata_hold)):
    b = np.append(cell_coords[i], 1.0)
    w, _ = nnls(A, b)
    s = w.sum()
    weights[i] = (w / s) if s > 1e-12 else np.full(K, 1.0 / K)
    if (i + 1) % 1000 == 0:
        print(f"    {i+1}/{len(adata_hold)}")

adata_hold.obsm["cell_archetype_weights"] = weights
print(f"  cell_archetype_weights: {weights.shape}")

pc.tl.assign_archetypes(adata_hold, percentage_per_archetype=0.15, verbose=False)
print("  assign_archetypes done")

out_path = DATA_DIR / "step1_holdout_cache.h5ad"
print(f"Saving to {out_path} ...")
adata_hold.write_h5ad(out_path)
print(f"Done — {out_path.stat().st_size / 1e9:.2f} GB written")

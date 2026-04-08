"""Dataset preparation for Paper Part 1: HSC + CMP.

Steps:
  1. Load HSCCMP.h5ad (logcounts, MT/RB already removed)
  2. Filter scrublet > 0.25
  3. Filter ribosomal protein genes (RPS, RPL, MRPS, MRPL)
  4. Trim outlier cells (>|5 SD| on any of first 30 PCs)
  5. ENSG → gene symbols
  6. Slice PCA to N_PCS
  7. 80/20 stratified holdout split by cell_type
  8. Save: adata_hsc_train.h5ad, adata_holdout.h5ad, adata_cmp.h5ad
"""

import os
import re
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(DATA_DIR, "paper_part1")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
SCRUBLET_THRESHOLD = 0.25
HOLDOUT_FRACTION = 0.20
PC_OUTLIER_SD = 5.0
RP_PATTERNS = [r"^RPS", r"^RPL", r"^MRPS", r"^MRPL"]

# -----------------------------------------------------------------------
# 1. Load
# -----------------------------------------------------------------------
print("Loading HSCCMP.h5ad...")
adata = ad.read_h5ad(os.path.join(DATA_DIR, "HSCCMP.h5ad"))
print(f"  Raw: {adata.shape[0]} cells × {adata.shape[1]} genes")
print(f"  Cell types: {dict(adata.obs['cell_type'].value_counts())}")

# -----------------------------------------------------------------------
# 2. Scrublet filtering
# -----------------------------------------------------------------------
n_before = adata.shape[0]
scrub_mask = adata.obs["scrublet_scores"] <= SCRUBLET_THRESHOLD
adata = adata[scrub_mask].copy()
n_after = adata.shape[0]
print(f"  Scrublet filter (≤{SCRUBLET_THRESHOLD}): {n_before} → {n_after} "
      f"(removed {n_before - n_after}, {(n_before - n_after)/n_before*100:.1f}%)")
print(f"  Cell types after: {dict(adata.obs['cell_type'].value_counts())}")

# -----------------------------------------------------------------------
# 3. Convert gene names from ENSG to gene symbols
# -----------------------------------------------------------------------
if "gene_symbols" in adata.var.columns:
    print(f"  Converting var_names: ENSG → gene_symbols")
    adata.var_names = adata.var["gene_symbols"].values
    adata.var_names_make_unique()
    print(f"  var_names now: {adata.var_names[:3].tolist()}")

# -----------------------------------------------------------------------
# 3b. Filter ribosomal protein genes (RP contamination in Wilcoxon/Wald)
# -----------------------------------------------------------------------
rp_regex = re.compile("|".join(RP_PATTERNS))
rp_mask = adata.var_names.str.match(rp_regex)
n_rp = int(rp_mask.sum())
n_before_rp = adata.shape[1]
if n_rp > 0:
    adata = adata[:, ~rp_mask].copy()
print(f"  RP gene filter ({RP_PATTERNS}): {n_before_rp} → {adata.shape[1]} genes "
      f"(removed {n_rp} RP genes)")

# -----------------------------------------------------------------------
# 3c. Trim outlier cells by PC z-score (|z| > PC_OUTLIER_SD on any of first 30 PCs)
# -----------------------------------------------------------------------
pca_pre = adata.obsm["X_pca"][:, :30]
pca_mean = pca_pre.mean(axis=0, keepdims=True)
pca_std = pca_pre.std(axis=0, keepdims=True)
pca_std[pca_std < 1e-10] = 1.0
pca_z = (pca_pre - pca_mean) / pca_std
outlier_mask = (np.abs(pca_z) > PC_OUTLIER_SD).any(axis=1)
n_outlier = int(outlier_mask.sum())
n_before_outlier = adata.shape[0]
if n_outlier > 0:
    adata = adata[~outlier_mask].copy()
print(f"  PC outlier trim (|z|>{PC_OUTLIER_SD} on first 30 PCs): "
      f"{n_before_outlier} → {adata.shape[0]} cells (removed {n_outlier})")
print(f"  Cell types after outlier trim: {dict(adata.obs['cell_type'].value_counts())}")

# -----------------------------------------------------------------------
# 4. Use original X_pca but compute PCA loadings on HVGs for gene alignment
# -----------------------------------------------------------------------
print("Using original X_pca from dataset...")
print(f"  X_pca shape: {adata.obsm['X_pca'].shape}")

# Compute PCA loadings (varm['PCs']) from HVGs — needed for gene alignment
# We regress X_hvg onto X_pca to get pseudo-loadings: PCs = (X_pca^T X_pca)^-1 X_pca^T X_hvg
print("Computing PCA loadings (varm['PCs']) for gene alignment...")
hvg_mask = adata.var.get("HVG_intersect3000", pd.Series(False, index=adata.var.index))
if hvg_mask.sum() > 0:
    import scipy.sparse as sp
    X_hvg = adata[:, hvg_mask].X
    if sp.issparse(X_hvg):
        X_hvg = X_hvg.toarray()
    X_hvg = np.asarray(X_hvg, dtype=np.float64)
    pca_mat_full = adata.obsm["X_pca"]
    # Least-squares: loadings = (PCA^T PCA)^-1 PCA^T X_hvg
    # This gives [n_pcs, n_hvg_genes] loadings
    PtP = pca_mat_full.T @ pca_mat_full
    PtX = pca_mat_full.T @ X_hvg
    loadings = np.linalg.solve(PtP, PtX)  # [n_pcs, n_hvg]
    # Store as varm['PCs'] — scanpy convention is [n_genes, n_pcs]
    # We need full-gene loadings, so pad with zeros for non-HVG genes
    full_loadings = np.zeros((adata.shape[1], pca_mat_full.shape[1]), dtype=np.float32)
    hvg_indices = np.where(hvg_mask.values)[0]
    full_loadings[hvg_indices] = loadings.T.astype(np.float32)
    adata.varm["PCs"] = full_loadings
    print(f"  varm['PCs'] shape: {full_loadings.shape} ({hvg_mask.sum()} HVG loadings + zero-padded)")
else:
    print("  WARNING: No HVG column found, cannot compute loadings")

# -----------------------------------------------------------------------
# 5. Select n_pcs — compute variance from PCA matrix, use 30 PCs
# -----------------------------------------------------------------------
N_PCS = 30
pca_mat = adata.obsm["X_pca"]
variances = np.var(pca_mat, axis=0)
total_var = variances.sum()
vr = variances / total_var
cumvar = np.cumsum(vr)
print(f"  PCA variance (from matrix): PC1={vr[0]:.4f}, cumvar@{N_PCS}={cumvar[N_PCS-1]:.4f}")

# Slice PCA to N_PCS components
adata.obsm["X_pca"] = adata.obsm["X_pca"][:, :N_PCS]
adata.uns["n_pcs_selected"] = N_PCS
adata.uns["cumvar_at_selection"] = float(cumvar[N_PCS - 1])
print(f"  Selected n_pcs={N_PCS}")

# -----------------------------------------------------------------------
# 5. 80/20 stratified holdout split
# -----------------------------------------------------------------------
rng = np.random.default_rng(SEED)
holdout_mask = np.zeros(adata.shape[0], dtype=bool)

for ct in adata.obs["cell_type"].unique():
    ct_idx = np.where(adata.obs["cell_type"] == ct)[0]
    n_holdout = int(len(ct_idx) * HOLDOUT_FRACTION)
    holdout_idx = rng.choice(ct_idx, size=n_holdout, replace=False)
    holdout_mask[holdout_idx] = True

adata.obs["holdout"] = holdout_mask
train_mask = ~holdout_mask

print(f"  Holdout split: train={train_mask.sum()}, holdout={holdout_mask.sum()}")
for ct in adata.obs["cell_type"].unique():
    ct_train = ((adata.obs["cell_type"] == ct) & train_mask).sum()
    ct_hold = ((adata.obs["cell_type"] == ct) & holdout_mask).sum()
    print(f"    {ct}: train={ct_train}, holdout={ct_hold}")

# -----------------------------------------------------------------------
# 6. Split and save
# -----------------------------------------------------------------------
# HSC train (80% of HSC cells)
hsc_train_mask = train_mask & (adata.obs["cell_type"] == "hematopoietic stem cell").values
adata_hsc_train = adata[hsc_train_mask].copy()
print(f"\n  HSC train: {adata_hsc_train.shape[0]} cells")

# Full holdout (20% of both HSC + CMP)
adata_holdout = adata[holdout_mask].copy()
print(f"  Holdout (HSC+CMP): {adata_holdout.shape[0]} cells")

# CMP train (80%) and holdout (20%)
cmp_train_mask = train_mask & (adata.obs["cell_type"] == "common myeloid progenitor").values
cmp_holdout_mask = holdout_mask & (adata.obs["cell_type"] == "common myeloid progenitor").values
adata_cmp_train = adata[cmp_train_mask].copy()
adata_cmp_holdout = adata[cmp_holdout_mask].copy()
print(f"  CMP train: {adata_cmp_train.shape[0]} cells")
print(f"  CMP holdout: {adata_cmp_holdout.shape[0]} cells")

# HSC holdout only
hsc_holdout_mask = holdout_mask & (adata.obs["cell_type"] == "hematopoietic stem cell").values
adata_hsc_holdout = adata[hsc_holdout_mask].copy()
print(f"  HSC holdout: {adata_hsc_holdout.shape[0]} cells")

# Save
paths = {
    "hsc_train": os.path.join(OUT_DIR, "adata_hsc_train.h5ad"),
    "hsc_holdout": os.path.join(OUT_DIR, "adata_hsc_holdout.h5ad"),
    "cmp_train": os.path.join(OUT_DIR, "adata_cmp_train.h5ad"),
    "cmp_holdout": os.path.join(OUT_DIR, "adata_cmp_holdout.h5ad"),
    "holdout_all": os.path.join(OUT_DIR, "adata_holdout.h5ad"),
    "full": os.path.join(OUT_DIR, "adata_full_prepped.h5ad"),
}

for name, path in paths.items():
    print(f"  Saving {name} → {path}")

adata_hsc_train.write_h5ad(paths["hsc_train"])
adata_hsc_holdout.write_h5ad(paths["hsc_holdout"])
adata_cmp_train.write_h5ad(paths["cmp_train"])
adata_cmp_holdout.write_h5ad(paths["cmp_holdout"])
adata_holdout.write_h5ad(paths["holdout_all"])
adata.write_h5ad(paths["full"])

print("\n=== Dataset preparation complete ===")
print(f"  HSC train: {adata_hsc_train.shape}")
print(f"  HSC holdout: {adata_hsc_holdout.shape}")
print(f"  CMP train: {adata_cmp_train.shape}")
print(f"  CMP holdout: {adata_cmp_holdout.shape}")
print(f"  Full holdout: {adata_holdout.shape}")
print(f"  Full holdout: {adata_holdout.shape}")
print(f"  n_pcs: {N_PCS} (cumvar={cumvar[N_PCS-1]:.4f})")

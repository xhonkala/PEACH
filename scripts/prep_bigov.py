"""Dataset preparation for Paper Part 1 OV variant: bigOV primary + metastatic.

All 10K cells are EOC (epithelial ovarian cancer) from the SPECTRUM MSK cohort,
CD45N (immune-depleted). Tissue site is parsed from the barcode (field 3 after
splitting on '_'):

  SPECTRUM-OV-XXX_S1_CD45N_TISSUE_BARCODE

Primary sites: RIGHT, LEFT (adnexa)
Metastatic sites: BOWEL, INFRACOLIC, PELVIC, ASCITES, LARGE, BLADDER, ANTERIOR,
                  HEPATIC, CECUM, INFRARENAL, LUQ, PELVIS

Steps:
  1. Load bigOV_10k.h5ad (logcounts already present, X_pca has only 11 comps)
  2. Parse tissue from barcode, assign primary/metastatic label
  3. Drop cells with unknown tissue (barcode doesn't split cleanly)
  4. Filter ribosomal protein genes (RPS, RPL, MRPS, MRPL)
  5. Recompute PCA with 30 components (existing X_pca is only 11)
  6. Trim outlier cells (|z|>5 on any of first 30 new PCs)
  7. 80/20 stratified holdout split per group (stratified by patient within each group)
  8. Save to data/paper_part1_ov/
"""

import os
import re
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

# ---------------------------------------------------------------------------
# Paths + config
# ---------------------------------------------------------------------------
INPUT_PATH = os.path.expanduser("~/Desktop/data/bigOV_10k.h5ad")
PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(PROJECT_DIR, "data", "paper_part1_ov")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
HOLDOUT_FRACTION = 0.20
PC_OUTLIER_SD = 5.0
N_PCS = 30
N_HVG = 3000
RP_PATTERNS = [r"^RPS", r"^RPL", r"^MRPS", r"^MRPL"]

PRIMARY_TISSUES = {"RIGHT", "LEFT"}
METASTATIC_TISSUES = {
    "BOWEL", "INFRACOLIC", "PELVIC", "ASCITES", "LARGE", "BLADDER",
    "ANTERIOR", "HEPATIC", "CECUM", "INFRARENAL", "LUQ", "PELVIS",
}

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
print(f"Loading {INPUT_PATH}...")
adata = ad.read_h5ad(INPUT_PATH)
print(f"  Raw: {adata.shape[0]} cells × {adata.shape[1]} genes")
print(f"  celltype2025: {dict(adata.obs['celltype2025'].value_counts())}")

# ---------------------------------------------------------------------------
# 2. Parse tissue + patient from barcode
# ---------------------------------------------------------------------------
print("Parsing patient + tissue from barcodes...")
idx_parts = pd.Series(adata.obs_names).str.split("_", expand=True)
adata.obs["patient"] = idx_parts[0].values
adata.obs["tissue_site"] = idx_parts[3].values

print(f"  unique patients: {adata.obs['patient'].nunique()}")
print(f"  unique tissue sites: {adata.obs['tissue_site'].nunique()}")

# ---------------------------------------------------------------------------
# 3. Assign primary/metastatic and drop unclassified
# ---------------------------------------------------------------------------
def classify_site(site):
    if site in PRIMARY_TISSUES:
        return "primary"
    if site in METASTATIC_TISSUES:
        return "metastatic"
    return "unknown"

adata.obs["group"] = adata.obs["tissue_site"].map(classify_site)
print(f"\n  Group assignment:")
print(f"    {dict(adata.obs['group'].value_counts())}")

n_before = adata.shape[0]
adata = adata[adata.obs["group"].isin(["primary", "metastatic"])].copy()
print(f"  Dropped {n_before - adata.shape[0]} unknown-tissue cells → {adata.shape[0]} remain")

# ---------------------------------------------------------------------------
# 4. Filter ribosomal protein genes (match HSC prep)
# ---------------------------------------------------------------------------
rp_regex = re.compile("|".join(RP_PATTERNS))
rp_mask = adata.var_names.str.match(rp_regex)
n_rp = int(rp_mask.sum())
if n_rp > 0:
    adata = adata[:, ~rp_mask].copy()
print(f"\n  RP gene filter: removed {n_rp} RP genes → {adata.shape[1]} genes")

# ---------------------------------------------------------------------------
# 5. Recompute PCA with 30 components
# ---------------------------------------------------------------------------
# bigOV's original X_pca has only 11 components, insufficient for paper_part1
# which needs 30. Recompute from scratch.
print(f"\n  Computing HVGs (top {N_HVG})...")
sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat", inplace=True)
print(f"    HVGs: {adata.var['highly_variable'].sum()}")

print(f"  Computing PCA ({N_PCS} components)...")
# Scale subset on HVGs only, compute PCA
adata_hvg = adata[:, adata.var["highly_variable"]].copy()
sc.pp.scale(adata_hvg, max_value=10)
sc.tl.pca(adata_hvg, n_comps=N_PCS, svd_solver="arpack")

# Copy PCA back to main adata + pad loadings to full gene space
adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
adata.uns["pca"] = adata_hvg.uns["pca"]

# Full-gene loadings: HVG rows filled, non-HVG rows zero
hvg_indices = np.where(adata.var["highly_variable"].values)[0]
full_loadings = np.zeros((adata.shape[1], N_PCS), dtype=np.float32)
full_loadings[hvg_indices] = adata_hvg.varm["PCs"].astype(np.float32)
adata.varm["PCs"] = full_loadings

variances = np.var(adata.obsm["X_pca"], axis=0)
total_var = variances.sum()
vr = variances / total_var
cumvar = np.cumsum(vr)
print(f"    PCA variance: PC1={vr[0]:.4f}, cumvar@{N_PCS}={cumvar[-1]:.4f}")
adata.uns["n_pcs_selected"] = N_PCS
adata.uns["cumvar_at_selection"] = float(cumvar[-1])

# ---------------------------------------------------------------------------
# 6. Trim PC outliers
# ---------------------------------------------------------------------------
pca_mat = adata.obsm["X_pca"]
pca_mean = pca_mat.mean(axis=0, keepdims=True)
pca_std = pca_mat.std(axis=0, keepdims=True)
pca_std[pca_std < 1e-10] = 1.0
pca_z = (pca_mat - pca_mean) / pca_std
outlier_mask = (np.abs(pca_z) > PC_OUTLIER_SD).any(axis=1)
n_outlier = int(outlier_mask.sum())
if n_outlier > 0:
    adata = adata[~outlier_mask].copy()
print(f"\n  PC outlier trim (|z|>{PC_OUTLIER_SD}): removed {n_outlier} → {adata.shape[0]} cells")
print(f"  Post-trim groups: {dict(adata.obs['group'].value_counts())}")

# Also retain cell_type for compat with paper_part1 script (uses cell_type for stratification)
adata.obs["cell_type"] = adata.obs["group"].astype("category")

# ---------------------------------------------------------------------------
# 7. 80/20 stratified holdout within each group (stratified by patient too)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
holdout_mask = np.zeros(adata.shape[0], dtype=bool)

# Within each group, stratify by patient so each patient contributes to both splits
for group in ["primary", "metastatic"]:
    group_idx = np.where(adata.obs["group"] == group)[0]
    patients_in_group = adata.obs["patient"].iloc[group_idx].unique()
    for pat in patients_in_group:
        pat_idx = group_idx[adata.obs["patient"].iloc[group_idx].values == pat]
        if len(pat_idx) < 5:
            # Tiny patient contribution — put all in train (nothing to holdout)
            continue
        n_holdout = max(1, int(len(pat_idx) * HOLDOUT_FRACTION))
        chosen = rng.choice(pat_idx, size=n_holdout, replace=False)
        holdout_mask[chosen] = True

adata.obs["holdout"] = holdout_mask
train_mask = ~holdout_mask

print(f"\n  Holdout split: train={train_mask.sum()}, holdout={holdout_mask.sum()}")
for group in ["primary", "metastatic"]:
    gt = ((adata.obs["group"] == group) & train_mask).sum()
    gh = ((adata.obs["group"] == group) & holdout_mask).sum()
    print(f"    {group}: train={gt}, holdout={gh}")

# ---------------------------------------------------------------------------
# 8. Split and save
# ---------------------------------------------------------------------------
primary_train_mask = train_mask & (adata.obs["group"] == "primary").values
primary_holdout_mask = holdout_mask & (adata.obs["group"] == "primary").values
metastatic_train_mask = train_mask & (adata.obs["group"] == "metastatic").values
metastatic_holdout_mask = holdout_mask & (adata.obs["group"] == "metastatic").values

adata_primary_train = adata[primary_train_mask].copy()
adata_primary_holdout = adata[primary_holdout_mask].copy()
adata_metastatic_train = adata[metastatic_train_mask].copy()
adata_metastatic_holdout = adata[metastatic_holdout_mask].copy()
adata_holdout_all = adata[holdout_mask].copy()

paths = {
    "primary_train":     os.path.join(OUT_DIR, "adata_primary_train.h5ad"),
    "primary_holdout":   os.path.join(OUT_DIR, "adata_primary_holdout.h5ad"),
    "metastatic_train":  os.path.join(OUT_DIR, "adata_metastatic_train.h5ad"),
    "metastatic_holdout":os.path.join(OUT_DIR, "adata_metastatic_holdout.h5ad"),
    "holdout_all":       os.path.join(OUT_DIR, "adata_holdout.h5ad"),
    "full":              os.path.join(OUT_DIR, "adata_full_prepped.h5ad"),
}

print()
for name, path in paths.items():
    print(f"  Saving {name} → {path}")

adata_primary_train.write_h5ad(paths["primary_train"])
adata_primary_holdout.write_h5ad(paths["primary_holdout"])
adata_metastatic_train.write_h5ad(paths["metastatic_train"])
adata_metastatic_holdout.write_h5ad(paths["metastatic_holdout"])
adata_holdout_all.write_h5ad(paths["holdout_all"])
adata.write_h5ad(paths["full"])

print("\n=== bigOV preparation complete ===")
print(f"  Primary train:    {adata_primary_train.shape}")
print(f"  Primary holdout:  {adata_primary_holdout.shape}")
print(f"  Metastatic train: {adata_metastatic_train.shape}")
print(f"  Metastatic holdout:{adata_metastatic_holdout.shape}")
print(f"  n_pcs: {N_PCS} (cumvar={cumvar[-1]:.4f})")
print(f"  n_patients: {adata.obs['patient'].nunique()}")

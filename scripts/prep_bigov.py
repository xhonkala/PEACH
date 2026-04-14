"""Dataset preparation for Paper Part 1 OV variant: bigOV primary + metastatic.

All 10K cells are EOC (epithelial ovarian cancer) from the SPECTRUM MSK cohort,
CD45N (immune-depleted). Tissue site is parsed from the barcode (field 3 after
splitting on '_'):

  SPECTRUM-OV-XXX_S1_CD45N_TISSUE_BARCODE

Primary sites: RIGHT, LEFT (adnexa)
Metastatic sites: BOWEL, INFRACOLIC, PELVIC, ASCITES, LARGE, BLADDER, ANTERIOR,
                  HEPATIC, CECUM, INFRARENAL, LUQ, PELVIS

Steps:
  1. Load bigOV_10k.h5ad (logcounts)
  2. Parse tissue from barcode, assign primary/metastatic label
  3. Drop cells with unknown tissue (barcode doesn't split cleanly)
  4. MT/RB 3-MAD cell filter + MT/RB/MRPL/MRPS/MALAT1 gene filter
  5. PCA on all genes, unscaled, n_comps = 11 (validated scree elbow)
  6. Compute HVGs for downstream gene analysis (NOT used for PCA)
  7. 80/20 stratified holdout split per group (stratified by patient within each group)
  8. Save to data/paper_part1_ov/
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paper_part1_prep import apply_mt_rb_mad_filter

# ---------------------------------------------------------------------------
# Paths + config
# ---------------------------------------------------------------------------
INPUT_PATH = os.path.expanduser("~/Desktop/data/bigOV_10k.h5ad")
PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(PROJECT_DIR, "data", "paper_part1_ov")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
HOLDOUT_FRACTION = 0.20
N_HVG = 3000
MT_RB_N_MADS = 3.0
# PCA dimensions: validated by visual scree elbow on OV data. The automated
# kneedle detector recovers 11 PCs, matching the user's prior selection.
N_PCS = 11

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
# 4. MT/RB 3-MAD cell filter + MT/RB/MALAT1 gene filter (shared helper)
# ---------------------------------------------------------------------------
adata = apply_mt_rb_mad_filter(adata, n_mads=MT_RB_N_MADS)

# ---------------------------------------------------------------------------
# 5. PCA: all genes, unscaled, n_comps = validated scree elbow
# ---------------------------------------------------------------------------
# Cell QC is done (MT/RB MAD + tissue classification). PCA just selects
# dimensions. No scaling (destroys convex-hull extremes), no HVG subsetting
# (misses full variance structure). All genes, unscaled logcounts.
print(f"\n  Computing PCA on ALL {adata.shape[1]} genes, unscaled, n_comps={N_PCS}...")
sc.tl.pca(adata, n_comps=N_PCS, svd_solver="arpack")

vr_scanpy = np.asarray(adata.uns["pca"]["variance_ratio"])
cumvar = np.cumsum(vr_scanpy)
print(
    f"    PCA variance (scanpy): PC1={vr_scanpy[0]:.4f}, "
    f"cumvar@{N_PCS}={cumvar[-1]:.4f}"
)
adata.uns["n_pcs_selected"] = N_PCS
adata.uns["cumvar_at_selection"] = float(cumvar[-1])
print(f"  {adata.shape[0]} cells x {N_PCS} PCs")

# HVGs still useful downstream (gene alignment, simplex regression) but NOT for PCA.
print(f"\n  Computing HVGs (top {N_HVG}) for downstream gene analysis...")
sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat", inplace=True)
print(f"    HVGs: {adata.var['highly_variable'].sum()}")

# Also retain cell_type for compat with paper_part1 script (uses cell_type for stratification)
adata.obs["cell_type"] = adata.obs["group"].astype("category")

# ---------------------------------------------------------------------------
# 7. 80/20 stratified holdout (stratified by patient within each group)
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

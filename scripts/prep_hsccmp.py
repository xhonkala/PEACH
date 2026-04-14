"""Dataset preparation for Paper Part 1: HSC + CMP.

Steps:
  1. Load HSCCMP.h5ad (logcounts)
  2. Filter scrublet > 0.25
  3. ENSG -> gene symbols
  4. MT/RB 3-MAD cell filter + MT/RB/MRPL/MRPS/MALAT1 gene filter
  5. PCA on all genes, unscaled, n_comps = 13 (validated scree elbow)
  6. 80/20 stratified holdout split by cell_type
  7. Save: adata_hsc_train.h5ad, adata_holdout.h5ad, adata_cmp.h5ad
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paper_part1_prep import apply_mt_rb_mad_filter

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(DATA_DIR, "paper_part1")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
SCRUBLET_THRESHOLD = 0.25
HOLDOUT_FRACTION = 0.20
MT_RB_N_MADS = 3.0
# PCA dimensions: validated by visual scree elbow on HSC data (tutorial
# WORKFLOW_01_DATA_LOAD.py). The automated elbow detector finds PC 5 for
# HSC (dominated by PC1) but the gradual decline from PC 5-13 still carries
# signal. 13 PCs is the empirically validated choice from prior runs.
N_PCS = 13

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
# 4. MT/RB 3-MAD cell filter + MT/RB/MALAT1 gene filter (shared helper)
# -----------------------------------------------------------------------
adata = apply_mt_rb_mad_filter(adata, n_mads=MT_RB_N_MADS)

# -----------------------------------------------------------------------
# 5. PCA: all genes, unscaled, n_comps = validated scree elbow
# -----------------------------------------------------------------------
# Cell QC is done (scrublet + MT/RB MAD). PCA selects dimensions only —
# no scaling (destroys convex-hull extremes), no HVG subsetting (misses
# full gene-level variance structure). All genes, unscaled logcounts.
print(f"\n  Computing PCA on ALL {adata.shape[1]} genes, unscaled, n_comps={N_PCS}...")
sc.tl.pca(adata, n_comps=N_PCS, svd_solver="arpack")

vr_scanpy = np.asarray(adata.uns["pca"]["variance_ratio"])
cumvar = np.cumsum(vr_scanpy)
print(
    f"  PCA variance (scanpy): PC1={vr_scanpy[0]:.4f}, "
    f"cumvar@{N_PCS}={cumvar[-1]:.4f}"
)
adata.uns["n_pcs_selected"] = N_PCS
adata.uns["cumvar_at_selection"] = float(cumvar[-1])
print(f"  {adata.shape[0]} cells × {N_PCS} PCs")

# -----------------------------------------------------------------------
# 6. 80/20 stratified holdout split
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
# 7. Split and save
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

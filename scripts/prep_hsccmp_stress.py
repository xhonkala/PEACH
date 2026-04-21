"""Dataset preparation for Paper Part 1 r15.1 STRESS-FILTERED variant.

Mirrors prep_hsccmp.py but restricts to the curated stress-response gene
set (HSR / OSR / UPR / HySR / DDR, 529 unique symbols) BEFORE running
PCA + archetype training. The goal is to ask: does an archetype
structure exist WITHIN the stress-response transcriptional program
itself? Option B from the r14 review discussion.

Steps:
  1. Load HSCCMP.h5ad (logcounts)
  2. Filter scrublet > 0.25
  3. ENSG -> gene symbols
  4. MT/RB 3-MAD cell filter + MT/RB/MRPL/MRPS/MALAT1 gene filter
  5. Restrict to stress_genes_flat.txt (intersection with var_names)
  6. PCA on the stress-gene subset, unscaled
  7. 80/20 stratified holdout split by cell_type
  8. Save: adata_hsc_train_stress.h5ad, adata_holdout_stress.h5ad,
          adata_cmp_train_stress.h5ad, etc.

n_pcs choice: with ~500 genes the scree elbow is typically earlier than
with ~28K genes. Start at n_pcs=8 (rough heuristic: ceil(sqrt(n_genes/8)));
adjust after inspecting the scree output. Downstream archetype CV in the
main script is tolerant to a modest n_pcs range.
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paper_part1_prep import apply_mt_rb_mad_filter

# Load the curated stress gene list built from mmc2.xlsx.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
from stress_genes.load_stress_genes import STRESS_GENES_FLAT, STRESS_SIGNATURES

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(DATA_DIR, "paper_part1")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
SCRUBLET_THRESHOLD = 0.25
HOLDOUT_FRACTION = 0.20
MT_RB_N_MADS = 3.0
# Stress-gene subset PCA: with ~500 genes, 8 PCs is a conservative start
# (roughly cumvar > 0.5 threshold). Adjust after scree inspection.
N_PCS = 8

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
print(f"  Scrublet filter (≤{SCRUBLET_THRESHOLD}): {n_before} → {n_after}")

# -----------------------------------------------------------------------
# 3. Convert gene names from ENSG to gene symbols
# -----------------------------------------------------------------------
if "gene_symbols" in adata.var.columns:
    print("  Converting var_names: ENSG → gene_symbols")
    adata.var_names = adata.var["gene_symbols"].values
    adata.var_names_make_unique()

# -----------------------------------------------------------------------
# 4. MT/RB 3-MAD cell filter + MT/RB/MALAT1 gene filter (shared helper)
# -----------------------------------------------------------------------
adata = apply_mt_rb_mad_filter(adata, n_mads=MT_RB_N_MADS)
print(f"  After QC: {adata.shape[0]} cells × {adata.shape[1]} genes")

# -----------------------------------------------------------------------
# 5. Restrict to curated stress gene set
# -----------------------------------------------------------------------
stress_set = set(g.upper() for g in STRESS_GENES_FLAT)
var_upper = pd.Series(adata.var_names).str.upper()
keep_mask = var_upper.isin(stress_set).values
n_stress_in_data = int(keep_mask.sum())
print(
    f"  Stress-gene restriction: {n_stress_in_data} / {len(STRESS_GENES_FLAT)} "
    f"stress genes found in var_names ({adata.shape[1]} total before filter)"
)
# Per-signature coverage so we know which subprograms survive the filter
for sig, genes in STRESS_SIGNATURES.items():
    sig_set = set(g.upper() for g in genes)
    hit = var_upper.isin(sig_set).sum()
    print(f"    {sig}: {hit}/{len(genes)}")
if n_stress_in_data < 50:
    raise RuntimeError(
        f"Too few stress genes present in the dataset ({n_stress_in_data}). "
        f"Check var_names (case, ENSG→symbol conversion) before continuing."
    )
adata = adata[:, keep_mask].copy()
adata.var_names_make_unique()
print(f"  Stress-filtered adata: {adata.shape[0]} cells × {adata.shape[1]} genes")

# -----------------------------------------------------------------------
# 6. PCA on stress-gene subset, unscaled
# -----------------------------------------------------------------------
# Smaller feature set → lower n_pcs start (8). Full scree printed below so
# the actual elbow can be re-validated from the log.
print(
    f"\n  Computing PCA on {adata.shape[1]} stress genes, unscaled, n_comps={N_PCS}..."
)
sc.tl.pca(adata, n_comps=N_PCS, svd_solver="arpack")

vr_scanpy = np.asarray(adata.uns["pca"]["variance_ratio"])
cumvar = np.cumsum(vr_scanpy)
print(f"  PCA per-PC variance ratio: {[f'{v:.4f}' for v in vr_scanpy]}")
print(f"  PCA cumulative variance:   {[f'{v:.4f}' for v in cumvar]}")
print(
    f"  PC1 fraction: {vr_scanpy[0]:.4f}; "
    f"cumvar@{N_PCS}: {cumvar[-1]:.4f}"
)
adata.uns["n_pcs_selected"] = N_PCS
adata.uns["cumvar_at_selection"] = float(cumvar[-1])

# -----------------------------------------------------------------------
# 7. 80/20 stratified holdout split
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
# 8. Split and save with _stress suffix
# -----------------------------------------------------------------------
hsc_train_mask = train_mask & (adata.obs["cell_type"] == "hematopoietic stem cell").values
adata_hsc_train = adata[hsc_train_mask].copy()

adata_holdout = adata[holdout_mask].copy()

cmp_train_mask = train_mask & (adata.obs["cell_type"] == "common myeloid progenitor").values
cmp_holdout_mask = holdout_mask & (adata.obs["cell_type"] == "common myeloid progenitor").values
adata_cmp_train = adata[cmp_train_mask].copy()
adata_cmp_holdout = adata[cmp_holdout_mask].copy()

hsc_holdout_mask = holdout_mask & (adata.obs["cell_type"] == "hematopoietic stem cell").values
adata_hsc_holdout = adata[hsc_holdout_mask].copy()

paths = {
    "hsc_train": os.path.join(OUT_DIR, "adata_hsc_train_stress.h5ad"),
    "hsc_holdout": os.path.join(OUT_DIR, "adata_hsc_holdout_stress.h5ad"),
    "cmp_train": os.path.join(OUT_DIR, "adata_cmp_train_stress.h5ad"),
    "cmp_holdout": os.path.join(OUT_DIR, "adata_cmp_holdout_stress.h5ad"),
    "holdout_all": os.path.join(OUT_DIR, "adata_holdout_stress.h5ad"),
    "full": os.path.join(OUT_DIR, "adata_full_prepped_stress.h5ad"),
}

for name, path in paths.items():
    print(f"  Saving {name} → {path}")

adata_hsc_train.write_h5ad(paths["hsc_train"])
adata_hsc_holdout.write_h5ad(paths["hsc_holdout"])
adata_cmp_train.write_h5ad(paths["cmp_train"])
adata_cmp_holdout.write_h5ad(paths["cmp_holdout"])
adata_holdout.write_h5ad(paths["holdout_all"])
adata.write_h5ad(paths["full"])

print("\n=== Stress-filtered dataset preparation complete ===")
print(f"  HSC train:    {adata_hsc_train.shape}")
print(f"  HSC holdout:  {adata_hsc_holdout.shape}")
print(f"  CMP train:    {adata_cmp_train.shape}")
print(f"  CMP holdout:  {adata_cmp_holdout.shape}")
print(f"  Full holdout: {adata_holdout.shape}")
print(f"  Full prepped: {adata.shape}")
print(f"  Stress genes retained: {adata.shape[1]} of {len(STRESS_GENES_FLAT)} curated")
print(f"  n_pcs: {N_PCS} (cumvar={cumvar[N_PCS-1]:.4f})")

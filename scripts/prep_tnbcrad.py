"""TNBC prep pipeline — Part 2 Step 1 input.

Mirrors prep_hsccmp.py recipe. Input: data/GSE246613_TNBC_ONLY_TRAIN.h5ad
(31,503 malignant cells by inferCNV). Output: three h5ad files in
data/paper_part2/ + a lightweight prep_report.html logging diagnostics.

Recipe (no deviations):
  1. adata.X = adata.layers["logcounts"].copy()  (no normalize, no scale)
  2. Coerce cohort/treatment/response_group/majority_voting -> category
  3. apply_mt_rb_mad_filter(n_mads=3.0)
  4. sc.pp.pca(n_comps=50, zero_center=False, use_highly_variable=False)
  5. Slice to N_PCS=12  (scree-locked 2026-04-14)
  6. safe_stratified_split on cohort × response × treatment (fallback: cohort)
  7. Write full_prepped / train / holdout + prep_report.html
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display on remote runs
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _paper_part1_prep import apply_mt_rb_mad_filter, safe_stratified_split  # noqa: E402

INPUT_PATH = REPO_ROOT / "data" / "GSE246613_TNBC_ONLY_TRAIN.h5ad"
OUT_DIR = REPO_ROOT / "data" / "paper_part2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PCS = 12                    # scree-locked 2026-04-14
MAD_N_MADS = 3.0
TEST_SIZE = 0.20
MIN_STRATUM_SIZE = 10
RANDOM_STATE = 42


def main() -> None:
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Loading {INPUT_PATH} ...")
    adata = sc.read_h5ad(INPUT_PATH)
    print(f"  Loaded {adata.shape}")

    # 1. logcounts -> X (no normalize, no scale)
    adata.X = adata.layers["logcounts"].copy()

    # 2. Categorical coercion
    for col in ("cohort", "treatment", "response_group", "majority_voting"):
        if col in adata.obs.columns:
            adata.obs[col] = adata.obs[col].astype("category")

    # 3. MAD filter
    n_in = adata.n_obs
    print(f"[{time.strftime('%H:%M:%S')}] MAD filter ...")
    adata = apply_mt_rb_mad_filter(adata, n_mads=MAD_N_MADS)
    n_dropped = n_in - adata.n_obs
    print(f"  After MAD filter: {adata.shape}  (dropped {n_dropped} cells)")

    # 4. PCA
    print(f"[{time.strftime('%H:%M:%S')}] Running sc.pp.pca(n_comps=50) ...")
    sc.pp.pca(adata, n_comps=50, zero_center=False, use_highly_variable=False)

    # 5. Slice to N_PCS=12
    full_var_ratio = np.asarray(adata.uns["pca"]["variance_ratio"])
    full_var = np.asarray(adata.uns["pca"]["variance"])
    adata.obsm["X_pca"] = adata.obsm["X_pca"][:, :N_PCS]
    adata.uns["pca"]["variance_ratio"] = full_var_ratio[:N_PCS]
    adata.uns["pca"]["variance"] = full_var[:N_PCS]

    # 6. Stratified 80/20 split
    primary = (adata.obs["cohort"].astype(str) + "|" +
               adata.obs["response_group"].astype(str) + "|" +
               adata.obs["treatment"].astype(str))
    fallback = adata.obs["cohort"].astype(str)
    train_idx, holdout_idx, diag = safe_stratified_split(
        primary, fallback,
        test_size=TEST_SIZE, min_stratum_size=MIN_STRATUM_SIZE,
        random_state=RANDOM_STATE,
    )
    print(f"[{time.strftime('%H:%M:%S')}] Split: "
          f"train={len(train_idx)} holdout={len(holdout_idx)} "
          f"fallback_cells={diag['n_fallback_cells']} "
          f"random_fallback={diag['n_random_fallback_cells']}")

    # 7. Write outputs
    full_path = OUT_DIR / "adata_tnbc_full_prepped.h5ad"
    train_path = OUT_DIR / "adata_tnbc_train.h5ad"
    holdout_path = OUT_DIR / "adata_tnbc_holdout.h5ad"
    print(f"[{time.strftime('%H:%M:%S')}] Writing {full_path}")
    adata.write_h5ad(full_path)
    print(f"[{time.strftime('%H:%M:%S')}] Writing {train_path}")
    adata[train_idx].copy().write_h5ad(train_path)
    print(f"[{time.strftime('%H:%M:%S')}] Writing {holdout_path}")
    adata[holdout_idx].copy().write_h5ad(holdout_path)

    # Scree sanity plot in output dir
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, 51), full_var_ratio, "o-")
    ax.axvline(N_PCS, ls="--", color="red", label=f"N_PCS={N_PCS}")
    ax.set_xlabel("PC")
    ax.set_ylabel("variance ratio")
    ax.set_title("TNBC prep PCA scree (50 PCs)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "prep_scree.png", dpi=140)
    plt.close(fig)

    # Minimal provenance HTML
    cumvar_at_k = float(full_var_ratio[:N_PCS].sum())
    prep_html = OUT_DIR / "prep_report.html"
    prep_html.write_text(f"""<!DOCTYPE html><html><body>
<h1>TNBC Prep Report</h1>
<p>Date: {time.strftime('%Y-%m-%d %H:%M')}</p>
<h2>Pipeline</h2>
<ul>
  <li>Input: {INPUT_PATH.name}</li>
  <li>MAD filter (n_mads={MAD_N_MADS}): {n_in} -> {adata.n_obs} cells (dropped {n_dropped})</li>
  <li>PCA: 50 components, no scale, no HVG, zero_center=False</li>
  <li>N_PCS sliced to {N_PCS}, cumulative variance = {cumvar_at_k:.3f}</li>
  <li>Stratified split (cohort × response × treatment, fallback cohort):
      train={len(train_idx)} holdout={len(holdout_idx)}
      fallback_cells={diag['n_fallback_cells']}
      random_fallback_cells={diag['n_random_fallback_cells']}
      collapsed_strata={diag['n_collapsed_strata']}
  </li>
</ul>
<p>Elapsed: {time.time() - t0:.1f}s</p>
</body></html>
""")
    print(f"[{time.strftime('%H:%M:%S')}] Done. Elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

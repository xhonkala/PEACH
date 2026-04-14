"""One-off diagnostic: run the planned Part 2 prep PCA and dump the scree curve.

Mirrors the prep_tnbcrad.py recipe (MAD filter → fresh PCA, no scale, no HVG)
up to the PCA step so the resulting elbow is what the real prep would produce.
Writes a PNG + a text table of variance ratios for visual inspection.

Run once to pick N_PCS; then hardcode the value in prep_tnbcrad.py.
"""
from __future__ import annotations

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

# Reuse the Part 1 helper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paper_part1_prep import apply_mt_rb_mad_filter  # type: ignore

INPUT_PATH = "/Users/honkala/Desktop/PEACH_public/data/GSE246613_TNBC_ONLY_TRAIN.h5ad"
OUT_DIR = "/Users/honkala/Desktop/PEACH_public/outputs/diagnostic"
os.makedirs(OUT_DIR, exist_ok=True)
PNG_PATH = os.path.join(OUT_DIR, "tnbc_pca_scree.png")
TXT_PATH = os.path.join(OUT_DIR, "tnbc_pca_scree.txt")

N_COMPS = 50


def main() -> None:
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Loading {INPUT_PATH} ...")
    adata = sc.read_h5ad(INPUT_PATH)
    print(f"  Loaded: {adata.shape}, layers: {list(adata.layers.keys())}")

    # Assign logcounts to .X (recipe: no scaling, no re-normalization)
    print(f"[{time.strftime('%H:%M:%S')}] Assigning logcounts layer → .X")
    adata.X = adata.layers["logcounts"].copy()

    # MAD filter per Part 1 recipe
    print(f"[{time.strftime('%H:%M:%S')}] Applying MT/RB MAD filter (n_mads=3.0) ...")
    n_before = adata.n_obs
    adata = apply_mt_rb_mad_filter(adata, n_mads=3.0)
    print(f"  After MAD filter: {adata.shape}  (dropped {n_before - adata.n_obs} cells)")

    # PCA — no scale, no HVG
    print(f"[{time.strftime('%H:%M:%S')}] Running sc.pp.pca(n_comps={N_COMPS}, zero_center=False) ...")
    sc.pp.pca(adata, n_comps=N_COMPS, zero_center=False, use_highly_variable=False)

    var_ratio = adata.uns["pca"]["variance_ratio"]
    var_abs = adata.uns["pca"]["variance"]
    cumvar = np.cumsum(var_ratio)

    # Save text table
    with open(TXT_PATH, "w") as f:
        f.write("PC\tvar\tvar_ratio\tcumvar\tdelta_var_ratio\n")
        for i in range(N_COMPS):
            delta = var_ratio[i] - var_ratio[i - 1] if i > 0 else 0.0
            f.write(f"{i+1}\t{var_abs[i]:.4f}\t{var_ratio[i]:.4f}\t{cumvar[i]:.4f}\t{delta:+.5f}\n")
    print(f"[{time.strftime('%H:%M:%S')}] Wrote variance table: {TXT_PATH}")

    # Scree + cumvar figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: scree (variance ratio)
    axes[0].plot(range(1, N_COMPS + 1), var_ratio, "o-", color="#2563eb")
    axes[0].set_xlabel("PC")
    axes[0].set_ylabel("variance ratio")
    axes[0].set_title("Scree — variance ratio per PC")
    axes[0].grid(alpha=0.3)

    # Panel 2: log scree
    axes[1].plot(range(1, N_COMPS + 1), var_ratio, "o-", color="#2563eb")
    axes[1].set_xlabel("PC")
    axes[1].set_ylabel("variance ratio (log)")
    axes[1].set_yscale("log")
    axes[1].set_title("Scree — log scale")
    axes[1].grid(alpha=0.3, which="both")

    # Panel 3: cumulative variance
    axes[2].plot(range(1, N_COMPS + 1), cumvar, "o-", color="#059669")
    for thresh, color in [(0.70, "#f59e0b"), (0.80, "#ea580c"), (0.85, "#dc2626"), (0.90, "#991b1b")]:
        axes[2].axhline(thresh, ls="--", alpha=0.4, color=color, label=f"{thresh:.0%}")
    axes[2].set_xlabel("PC")
    axes[2].set_ylabel("cumulative variance ratio")
    axes[2].set_title("Cumulative variance")
    axes[2].legend(loc="lower right", fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.suptitle(f"TNBC Part 2 prep PCA scree  (n={adata.n_obs}, genes={adata.n_vars})")
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
    print(f"[{time.strftime('%H:%M:%S')}] Wrote scree figure: {PNG_PATH}")

    # Top-10 table to stdout
    print("\nTop 15 PCs (variance_ratio, cumulative):")
    print(f"  {'PC':>3}  {'var_ratio':>9}  {'cumvar':>7}  {'delta':>8}")
    for i in range(min(15, N_COMPS)):
        delta = var_ratio[i] - var_ratio[i - 1] if i > 0 else 0.0
        print(f"  {i+1:>3}  {var_ratio[i]:>9.4f}  {cumvar[i]:>7.4f}  {delta:>+8.5f}")

    # Heuristic candidate PCs
    # Kaiser-ish: PCs where delta(var_ratio) inflects hardest
    deltas = np.abs(np.diff(var_ratio))
    second_deriv = np.diff(deltas)
    top_elbow_cands = np.argsort(-np.abs(second_deriv))[:5] + 2  # +2: diff shifts index
    print(f"\nTop-5 elbow candidates by |2nd derivative|: {sorted(top_elbow_cands.tolist())}")

    # Simple “10× drop” rule: first PC where var_ratio < 0.1 * var_ratio[0]
    threshold = 0.1 * var_ratio[0]
    first_below = int(np.argmax(var_ratio < threshold))
    if first_below > 0:
        print(f"10%-of-PC1 rule: PC{first_below + 1} is the first PC below 10% of PC1's variance ratio")

    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

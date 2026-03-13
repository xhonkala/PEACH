#!/usr/bin/env python
"""Generate 12a_regression.ipynb — simplex regression, driver regression, Wald contrasts, comparison."""

import nbformat

nb = nbformat.v4.new_notebook()
nb.metadata.kernelspec = {
    "display_name": "Python 3 (archetype)",
    "language": "python",
    "name": "python3",
}

DATA_DIR = "/Users/honkala/Desktop/PEACH_public/data/trained"


def md(text):
    nb.cells.append(nbformat.v4.new_markdown_cell(text))


def code(text):
    nb.cells.append(nbformat.v4.new_code_cell(text))


# ===========================================================================
# TITLE
# ===========================================================================
md("""\
# 12a. Simplex Regression & Archetype Comparison

Forward simplex regression (archetype weights → features), driver regression
(features → archetype position), Wald contrasts, pattern classification, and
archetype comparison (MMD, feature similarity).

**Loads pre-trained AnnDatas** with archetype weights, pathway scores, and
regression results already computed.""")

# ===========================================================================
# SETUP
# ===========================================================================
code("""\
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import anndata as ad
import peach as pc

ALPHA = 0.05
K = 4""")

code(f"""\
# Load pre-trained AnnDatas
adata_cmp = ad.read_h5ad("{DATA_DIR}/hsc_cmp_v050.h5ad")
adata_mono = ad.read_h5ad("{DATA_DIR}/hsc_mono_v050.h5ad")
print(f"CMP: {{adata_cmp.shape}}")
print(f"Mono: {{adata_mono.shape}}")
print(f"\\nStored keys in uns: {{sorted(adata_cmp.uns.keys())}}")
print(f"Stored keys in obsm: {{sorted(adata_cmp.obsm.keys())}}")""")

# ===========================================================================
# 1. SIMPLEX REGRESSION INSPECTION
# ===========================================================================
md("""\
## 1. Simplex Regression Results

Scheffe polynomial regression on the weight simplex. Degree 1 (vertex effects)
captures linear feature–archetype relationships; degree 2 (interactions) captures
synergistic effects between archetype pairs.""")

code("""\
from peach._core.utils.feature_utils import resolve_regression_result

for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    reg = resolve_regression_result(ad_obj, prefer="genes")
    coefs = np.asarray(reg["vertex_coefficients"])
    vertex_pvals_fdr = np.asarray(reg["vertex_pvalues_fdr"])
    r2_d1 = np.asarray(reg["r_squared_degree1"])
    r2_d2 = np.asarray(reg["r_squared_degree2"])
    feat_names = reg["feature_names"]

    print(f"\\n{'='*60}")
    print(f"  {name}: {len(feat_names)} features, K={K}")
    print(f"{'='*60}")
    print(f"  Degree-1 R²: median={np.median(r2_d1):.4f}, max={np.max(r2_d1):.4f}")
    print(f"  Degree-2 R²: median={np.median(r2_d2):.4f}, max={np.max(r2_d2):.4f}")

    # Effective rank
    eff_rank = reg.get("effective_rank")
    if eff_rank is not None:
        print(f"  Effective rank: {eff_rank}/{reg.get('expected_rank')}")

    # Top 3 genes per archetype
    print(f"\\n  Top 3 per archetype (|β|):")
    for k in range(K):
        top3 = np.argsort(np.abs(coefs[:, k]))[-3:][::-1]
        for i in top3:
            print(f"    A{k+1}: {feat_names[i]:20s} β={coefs[i,k]:8.3f}  "
                  f"q={vertex_pvals_fdr[i,k]:.2e}")""")

# ===========================================================================
# 1b. DEGREE COMPARISON
# ===========================================================================
md("""\
### Degree Comparison

F-tests for incremental variance explained by adding interaction (degree 2)
and cubic (degree 3) terms.""")

code("""\
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    reg = resolve_regression_result(ad_obj, prefer="genes")
    deg_comp = reg.get("degree_comparison")
    if deg_comp is None:
        print(f"{name}: no degree comparison available")
        continue

    print(f"\\n--- {name} ---")
    for deg_key, deg_info in sorted(deg_comp.items()):
        delta_r2 = np.asarray(deg_info["delta_r2"])
        inc_p_fdr = np.asarray(deg_info["incremental_p_fdr"])
        n_sig = int(deg_info["significant_features"])
        print(f"  {deg_key}: {n_sig}/{len(delta_r2)} significant "
              f"({100*n_sig/len(delta_r2):.1f}%)")

        top5 = np.argsort(delta_r2)[-5:][::-1]
        for i in top5:
            print(f"    {reg['feature_names'][i]:20s} ΔR²={delta_r2[i]:.4f}  "
                  f"q={inc_p_fdr[i]:.2e}")""")

# ===========================================================================
# 1c. PATHWAY REGRESSION
# ===========================================================================
md("""\
### Pathway Regression (HALLMARK)

Same Scheffe regression on pathway activity scores instead of raw gene expression.""")

code("""\
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    pw_reg = resolve_regression_result(ad_obj, prefer="pathways")
    if pw_reg is None:
        print(f"{name}: no pathway regression found")
        continue

    pw_r2 = np.asarray(pw_reg["r_squared_degree1"])
    pw_fdr = np.asarray(pw_reg["f_pvalue_fdr"])
    n_sig = int(np.sum(pw_fdr < ALPHA))
    print(f"\\n{name}: {n_sig}/{len(pw_r2)} significant pathways")

    for i in np.argsort(pw_r2)[-5:][::-1]:
        print(f"  {pw_reg['feature_names'][i]:45s} R²={pw_r2[i]:.4f}")""")

# ===========================================================================
# 2. REGRESSION VISUALIZATIONS
# ===========================================================================
md("""\
## 2. Regression Visualizations""")

code("""\
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    print(f"\\n--- {name} ---")
    _ = pc.pl.coefficient_heatmap(ad_obj, top_n=30, show=True)
    _ = pc.pl.r2_barplot(ad_obj, top_n=30, show=True)
    _ = pc.pl.regression_volcano(ad_obj, alpha=0.05, show=True)
    _ = pc.pl.archetype_regression_dotplot(ad_obj, top_n=10, show=True)
    _ = pc.pl.archetype_regression_dotplot(ad_obj, top_n=20, exclusive_only=True, show=True)""")

# ===========================================================================
# 3. WALD CONTRASTS
# ===========================================================================
md("""\
## 3. Wald Contrasts

Pairwise differential expression between all K*(K-1)/2 archetype pairs.
Global Benjamini-Hochberg FDR across ALL pairs (not per-pair).""")

code("""\
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    contrasts = ad_obj.uns.get("peach_archetype_contrasts")
    if contrasts is None:
        print(f"{name}: no contrasts found, recomputing...")
        contrasts = pc.tl.archetype_contrasts(ad_obj)

    pairs = [tuple(p) if isinstance(p, (list, np.ndarray)) else p
             for p in contrasts["pairs"]]
    feat_names = list(contrasts["feature_names"])

    print(f"\\n{'='*60}")
    print(f"  {name}: Wald Contrasts — {len(pairs)} pairs, {len(feat_names)} features")
    print(f"{'='*60}")

    for pair in pairs:
        key = str(tuple(pair))
        delta = np.asarray(contrasts["delta_beta"][key])
        pvals_fdr = np.asarray(contrasts["pvalues_fdr"][key])
        z = np.asarray(contrasts["z_scores"][key])
        n_sig = int(np.sum(pvals_fdr < ALPHA))
        n_up = int(np.sum((pvals_fdr < ALPHA) & (delta > 0)))

        print(f"\\n  A{pair[0]} vs A{pair[1]}: {n_sig} significant ({n_up} up)")
        top3 = np.argsort(np.abs(z))[-3:][::-1]
        for i in top3:
            print(f"    {feat_names[i]:20s} Δβ={delta[i]:8.3f} z={z[i]:7.1f} "
                  f"q={pvals_fdr[i]:.2e}")""")

md("""\
### Wald Volcano Grid""")

code("""\
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    print(f"--- {name} ---")
    _ = pc.pl.contrast_volcano_grid(ad_obj, n_labels=5, show=True)""")

# ===========================================================================
# 4. PATTERN CLASSIFICATION
# ===========================================================================
md("""\
## 4. Pattern Classification

Features classified by their simplex regression profile:
- **flat**: R² < 0.05 or low coefficient variation
- **exclusive**: one archetype dominates (max β / 2nd β ≥ 2)
- **monotonic**: ordered gradient across archetypes (|ρ| > 0.9)
- **gradient**: multi-archetype enrichment""")

code("""\
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    pat = ad_obj.uns.get("peach_feature_patterns")
    if pat is None:
        pat = pc.tl.classify_feature_patterns(ad_obj)

    counts = pat["pattern_counts"]
    n_total = pat["n_features"]
    print(f"\\n{name}:")
    for ptype, pcount in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {ptype:25s} {pcount:5d} ({100*pcount/n_total:5.1f}%)")""")

# ===========================================================================
# 5. ARCHETYPE COMPARISON
# ===========================================================================
md("""\
## 5. Archetype Comparison

- **MMD**: Maximum Mean Discrepancy between archetype cell populations
- **Feature similarity**: Spearman correlation of regression coefficient vectors
- **Between-fit comparison**: CMP vs Mono archetype correspondence""")

code("""\
# Within-fit MMD
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    mmd = ad_obj.uns.get("peach_archetype_mmd")
    if mmd is None:
        mmd = pc.tl.archetype_mmd(ad_obj, n_permutations=50)
    mmd_mat = np.asarray(mmd["mmd_matrix"])
    K_mmd = mmd_mat.shape[0]
    mask_off = ~np.eye(K_mmd, dtype=bool)
    print(f"{name} within-fit MMD: range=[{mmd_mat[mask_off].min():.4f}, "
          f"{mmd_mat[mask_off].max():.4f}]")""")

code("""\
# Feature similarity
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    sim = ad_obj.uns.get("peach_archetype_feature_similarity")
    if sim is None:
        sim = pc.tl.archetype_feature_similarity(ad_obj)
    print(f"{name} silhouette: {sim['silhouette_overall']:.3f}")

    fdr_mat = sim.get("spearman_pvalue_fdr_matrix")
    if fdr_mat is not None:
        n_sig_pairs = int(np.sum(np.asarray(fdr_mat) < ALPHA))
        n_total_pairs = fdr_mat.size
        print(f"  Spearman FDR < 0.05: {n_sig_pairs}/{n_total_pairs} pairs")""")

code("""\
# Between-fit comparison (CMP vs Mono)
mmd_between = pc.tl.archetype_mmd(adata_cmp, adata_b=adata_mono, n_permutations=50)
sim_between = pc.tl.archetype_feature_similarity(adata_cmp, adata_b=adata_mono)
print(f"Between-fit MMD matrix:\\n{np.array2string(np.asarray(mmd_between['mmd_matrix']), precision=4)}")
print(f"Between-fit silhouette: {sim_between['silhouette_overall']:.3f}")""")

code("""\
# Comparison visualizations
_ = pc.pl.mmd_heatmap(adata_cmp, show=True)
_ = pc.pl.feature_similarity_heatmap(adata_cmp, show=True)""")

# ===========================================================================
# 6. DRIVER REGRESSION
# ===========================================================================
md("""\
## 6. Driver Regression (Reverse Direction)

ILR-transformed archetype position as response, gene expression as predictors.
Identifies genes that *predict* archetype position (vs forward regression which
identifies genes *associated with* archetype weights).""")

code("""\
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    print(f"\\n--- {name}: Driver Regression ---")
    driver = pc.tl.archetype_driver_regression(ad_obj)

    main_coefs = np.asarray(driver["main_coefficients"])
    main_pvals = np.asarray(driver["main_pvalues"])
    r2 = np.asarray(driver["r_squared"])

    print(f"  Main coefficients: {main_coefs.shape}")
    print(f"  R² per ILR component: {np.array2string(r2, precision=4)}")

    # FDR-corrected p-values
    fdr = driver.get("main_pvalues_fdr")
    if fdr is not None:
        fdr = np.asarray(fdr)
        n_sig = int(np.sum(fdr < ALPHA))
        print(f"  Significant drivers (FDR < 0.05): {n_sig}/{fdr.shape[0] * fdr.shape[1]}")

    # Top genes by max |coefficient| across components
    max_abs = np.max(np.abs(main_coefs), axis=0)
    top10 = np.argsort(max_abs)[-10:][::-1]
    feat_names = list(driver["feature_names"])
    print(f"\\n  Top 10 driver genes:")
    for i in top10:
        coef_str = ", ".join(f"{main_coefs[c, i]:.3f}" for c in range(main_coefs.shape[0]))
        p_str = f"q={fdr.ravel()[i]:.2e}" if fdr is not None else ""
        print(f"    {feat_names[i]:20s} β=[{coef_str}]  {p_str}")""")

# ===========================================================================
# 7. CROSS-CONCORDANCE
# ===========================================================================
md("""\
## 7. Cross-Population Concordance

Spearman correlation between CMP and Mono regression R² values — tests whether
the same genes are archetype-associated in both populations.""")

code("""\
from scipy.stats import spearmanr
from peach._core.utils.feature_utils import resolve_regression_result

reg_cmp = resolve_regression_result(adata_cmp, prefer="genes")
reg_mono = resolve_regression_result(adata_mono, prefer="genes")

r2_cmp = np.asarray(reg_cmp["r_squared_degree1"])
r2_mono = np.asarray(reg_mono["r_squared_degree1"])
rho, pval = spearmanr(r2_cmp, r2_mono)
print(f"Gene R² concordance: ρ={rho:.4f} (p={pval:.2e})")

pw_cmp = resolve_regression_result(adata_cmp, prefer="pathways")
pw_mono = resolve_regression_result(adata_mono, prefer="pathways")
if pw_cmp is not None and pw_mono is not None:
    pw_r2_cmp = np.asarray(pw_cmp["r_squared_degree1"])
    pw_r2_mono = np.asarray(pw_mono["r_squared_degree1"])
    rho_pw, pval_pw = spearmanr(pw_r2_cmp, pw_r2_mono)
    print(f"Pathway R² concordance: ρ={rho_pw:.4f} (p={pval_pw:.2e})")""")

# ===========================================================================
# 8. ARCHETYPE PHENOTYPE RADAR + RIDGE
# ===========================================================================
md("""\
## 8. Archetype Phenotype Visualization

Radar-ridgeplot: radar shows feature coefficient profiles across archetypes,
violins show expression distributions in archetype-dominant cells.""")

code("""\
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    print(f"\\n--- {name}: Gene Radar-Ridgeplot ---")
    figs = pc.pl.archetype_radar_ridgeplot(ad_obj, top_n=8, show=True)
    if isinstance(figs, tuple):
        print(f"  Radar + Ridge: {len(figs)} figures")""")

code("""\
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    print(f"\\n--- {name}: Pathway Radar-Ridgeplot ---")
    figs = pc.pl.archetype_radar_ridgeplot(ad_obj, top_n=5, feature_type="pathways", show=True)
    if isinstance(figs, tuple):
        print(f"  Radar + Ridge: {len(figs)} figures")""")

# ===========================================================================
# 9. SOFT ASSIGNMENT FEATURE FLOW
# ===========================================================================
md("""\
## 9. Feature Flow Between Archetypes

Sankey diagram showing features that span archetype pairs (shared enrichment)
vs features exclusive to one archetype in each pair (tradeoffs).""")

code("""\
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    print(f"\\n--- {name}: Degree 1 (FDR < 0.05) ---")
    try:
        _ = pc.pl.soft_assignment_flow(ad_obj, top_n=10, alpha=0.05, degree=1, show=True)
    except Exception as e:
        print(f"  Degree 1: {e}")
    print(f"\\n--- {name}: Degree 2 ---")
    try:
        _ = pc.pl.soft_assignment_flow(ad_obj, top_n=10, alpha=0.05, degree=2, show=True)
    except Exception as e:
        print(f"  Degree 2: {e}")""")


# ===========================================================================
# WRITE
# ===========================================================================
output_path = "docs/tutorials/12a_regression.ipynb"
nbformat.write(nb, output_path)
print(f"Wrote {len(nb.cells)} cells to {output_path}")

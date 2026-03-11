# v0.5.0 Reviewer-Grade E2E Notebook Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `tests/test_e2e_hsc_v12.py` — a headless runner that exercises the full v0.5.0 pipeline with proper statistical controls, interleaved visualizations, HALLMARK pathway scoring, and flow-aligned gene/geneset analysis. Once passing, convert to `docs/tutorials/12_e2e_v050_reviewer.ipynb`.

**Architecture:** Single headless Python script with section/step/done logging pattern (matching existing `test_e2e_hsc.py`). Gene names swapped from ENSG to symbols at load time. Full bootstrap + permutation tests enabled. Each analysis step immediately followed by its visualization(s). New sections for HALLMARK pathway scoring and flow-gene/geneset alignment.

**Tech Stack:** PEACH v0.5.0 API, plotly (HTML output), decoupler (MSigDB HALLMARK), scipy.sparse

---

## Context for Implementer

### Key Files
- **Pattern to follow:** `tests/test_e2e_hsc.py` — existing headless runner
- **PEACH public API:** `src/peach/_core/tools_schema.py` for inputs, `src/peach/_core/types_index.py` for outputs
- **Pathway scoring:** `pc.pp.load_pathway_networks(["hallmark"])` then `pc.pp.compute_pathway_scores(adata, net)`
- **Flow gene alignment:** `pc.tl.flow_gene_alignment(adata, flow_result, n_top=50)` returns dict with `alignment_scores`, `top_aligned`, `top_opposed`
- **Viz functions:** All accept `show=False, save_path=...` and return `go.Figure`

### Gene Name Swap
HSC data has `adata.var["gene_symbols"]` with human-readable symbols. `var_names` are ENSG IDs. Swap at load time:
```python
symbols = adata_full.var["gene_symbols"].values
# Handle duplicates: append suffix
from collections import Counter
counts = Counter(symbols)
seen = Counter()
unique = []
for s in symbols:
    if counts[s] > 1:
        seen[s] += 1
        unique.append(f"{s}_{seen[s]}")
    else:
        unique.append(s)
var_names = unique
```

### Contrast Key Serialization
`to_serializable()` stringifies tuple keys. Use `str(tuple(pair))` to look up `pvalues_fdr`, `delta_beta`, etc.

### Viz Doubling Fix
Always use `show=False` when calling `pc.pl.*` functions. The headless runner saves HTML files with `save_path=`.

### Pathway Scoring Requirements
- `compute_pathway_scores` matches genes by `adata.var_names` against MSigDB gene symbols
- Must swap var_names to symbols BEFORE computing pathway scores
- HALLMARK has ~50 gene sets; gene overlap will be checked automatically

---

## Task 1: Scaffold — Data Loading with Gene Symbol Swap

**Files:**
- Create: `tests/test_e2e_hsc_v12.py`

**Step 1: Write the scaffold**

Create the file with config, helpers, and Section 1 (data loading with gene symbol swap):

```python
"""v0.5.0 Reviewer-Grade E2E: HSC CMP vs CD14+ Monocyte.

Full pipeline with statistical controls, interleaved visualizations,
HALLMARK pathway scoring, and flow-aligned gene/geneset analysis.

Usage:
    /Users/honkala/miniconda3/envs/archetype/bin/python tests/test_e2e_hsc_v12.py
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import warnings
import logging
import numpy as np
import scipy.sparse as sp
import pandas as pd
from collections import Counter

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.disable(logging.WARNING)

import torch
torch.set_num_threads(1)

# ─── Config ──────────────────────────────────────────────────────────────
HSC_PATH = "/Users/honkala/Desktop/cross_recons/data/HSC.h5ad"
K = 4
N_PCS = 20
MONO_SUBSAMPLE = 9000
TRAIN_EPOCHS = 150
FLOW_EPOCHS = 300
SEED = 42
OUTPUT_DIR = "/Users/honkala/Desktop/PEACH_public/tests/e2e_outputs_v12"
N_BOOTSTRAP = 1000
N_PERMUTATIONS = 1000
ALPHA = 0.05


def section(title):
    print(f"\n{'='*60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*60}", flush=True)


def step(msg):
    print(f"  -> {msg}", end="", flush=True)


def done(extra=""):
    print(f" OK {extra}", flush=True)


def report(lines):
    """Print indented report lines for reviewer audit trail."""
    for line in lines:
        print(f"      {line}", flush=True)


def save_viz(fig, filename):
    """Save plotly figure to HTML."""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.write_html(path)
    return path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()

    # ─── 1. Data Loading ─────────────────────────────────────────────
    section("1. Data Loading & Gene Symbol Mapping")

    step("Loading HSC dataset")
    import anndata as ad
    import gc
    adata_full = ad.read_h5ad(HSC_PATH)
    done(f"({adata_full.shape[0]:,} cells x {adata_full.shape[1]:,} genes)")

    # Swap ENSG -> gene symbols
    step("Mapping ENSG IDs to gene symbols")
    symbols = list(adata_full.var["gene_symbols"].values)
    counts = Counter(symbols)
    n_duplicates = sum(1 for s, c in counts.items() if c > 1)
    seen = Counter()
    unique_symbols = []
    for s in symbols:
        if counts[s] > 1:
            seen[s] += 1
            unique_symbols.append(f"{s}_{seen[s]}")
        else:
            unique_symbols.append(s)
    var_names = unique_symbols
    done()
    report([
        f"Total genes: {len(var_names)}",
        f"Unique symbols: {len(set(symbols))}",
        f"Duplicate symbols (suffixed): {n_duplicates}",
        f"Sample mapping: {list(adata_full.var_names[:3])} -> {var_names[:3]}",
    ])

    rng = np.random.default_rng(SEED)
    cell_data = {}
    for name, ct, subsample in [
        ("CMP", "common myeloid progenitor", None),
        ("Mono", "CD14-positive monocyte", MONO_SUBSAMPLE),
    ]:
        step(f"Subsetting {name}")
        mask = adata_full.obs["cell_type"] == ct
        idx = np.where(mask)[0]
        n_total = len(idx)
        if subsample and len(idx) > subsample:
            idx = rng.choice(idx, size=subsample, replace=False)
            idx.sort()
        pca = adata_full.obsm["X_pca"][idx, :N_PCS].copy()
        obs = adata_full.obs.iloc[idx].copy().reset_index(drop=True)
        cell_data[name] = {"pca": pca, "obs": obs, "idx": idx}
        sub_note = f" (subsampled from {n_total:,})" if subsample else ""
        done(f"({len(idx):,} cells{sub_note}, PCA: {pca.shape})")

    del adata_full
    gc.collect()

    # ─── SECTION 2-9 WILL BE ADDED IN SUBSEQUENT TASKS ───────────────

    section("Summary")
    elapsed_total = time.time() - t0
    n_files = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.html')])
    print(f"  Total time: {elapsed_total:.1f}s", flush=True)
    print(f"  HTML files: {n_files} in {OUTPUT_DIR}", flush=True)
    print(f"\n  END-TO-END TEST PASSED", flush=True)


if __name__ == "__main__":
    main()
```

**Step 2: Run to verify scaffold works**

Run: `/Users/honkala/miniconda3/envs/archetype/bin/python tests/test_e2e_hsc_v12.py 2>tests/e2e_outputs_v12/stderr.log`
Expected: Loads data, prints gene symbol mapping stats, prints PASSED.

---

## Task 2: Archetype Fitting + Training Diagnostics

**Files:**
- Modify: `tests/test_e2e_hsc_v12.py`

**Step 1: Add Section 2 after data loading**

Insert before the Summary section. Train both cell types, extract weights, coordinates, assign archetypes. Reload sparse X with symbol var_names. Report training diagnostics as positive/negative controls.

```python
    # ─── 2. Archetype Fitting ────────────────────────────────────────
    section("2. Archetype Fitting (K={K})")
    import peach as pc

    models = {}
    for name in ["CMP", "Mono"]:
        step(f"Training K={K} on {name}")
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
        elapsed = time.time() - t1
        r2 = result.get("final_archetype_r2", None)
        done(f"(R2={r2:.3f}, {elapsed:.1f}s)")

        pc.tl.extract_archetype_weights(adata_train)
        pc.tl.archetypal_coordinates(adata_train, verbose=False)

        weights = adata_train.obsm["cell_archetype_weights"]
        cell_data[name]["weights"] = weights
        cell_data[name]["uns"] = dict(adata_train.uns)
        cell_data[name]["arch_dist"] = adata_train.obsm["archetype_distances"]
        models[name] = result

        # Training diagnostics (positive controls)
        report([
            f"TRAINING DIAGNOSTICS:",
            f"  Final R2: {r2:.4f} (expect >0.5 for meaningful structure)",
            f"  Constraint satisfaction: {result.get('constraints_satisfied', 'N/A')}",
            f"  Weight sum check: mean={weights.sum(axis=1).mean():.6f} (expect 1.0)",
            f"  Weight range: [{weights.min():.4f}, {weights.max():.4f}]",
            f"  Dominant cells per archetype:",
        ])
        for k in range(K):
            dominant = np.sum(weights.argmax(axis=1) == k)
            report([f"    Archetype {k}: {dominant} cells ({100*dominant/len(weights):.1f}%)"])

    # Reload sparse X with gene symbol var_names
    step("Reloading sparse X for regression")
    adata_disk = ad.read_h5ad(HSC_PATH, backed="r")
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
        # Store PCA loadings for flow gene alignment later
        adata_obj.varm["PCs"] = np.zeros((len(var_names), N_PCS))  # placeholder, will be set from full data
        pc.tl.assign_archetypes(adata_obj)
        cell_data[name]["adata"] = adata_obj
    del adata_disk
    gc.collect()
    done()

    adata_cmp = cell_data["CMP"]["adata"]
    adata_mono = cell_data["Mono"]["adata"]
```

Note: PCA loadings for flow alignment need to come from the full dataset. Add this during the data loading step:
```python
    # In data loading, before deleting adata_full:
    pca_loadings = adata_full.varm["PCs"][:, :N_PCS].copy() if "PCs" in adata_full.varm else None
```
Then when rebuilding adata objects:
```python
    if pca_loadings is not None:
        adata_obj.varm["PCs"] = pca_loadings
```

**Step 2: Run to verify training completes**

Run: `/Users/honkala/miniconda3/envs/archetype/bin/python tests/test_e2e_hsc_v12.py 2>tests/e2e_outputs_v12/stderr.log`
Expected: Both cell types train with R2 > 0.5, diagnostics printed, PASSED.

---

## Task 3: Simplex Regression with Full Statistical Controls + Viz

**Files:**
- Modify: `tests/test_e2e_hsc_v12.py`

**Step 1: Add Section 3 — full regression with bootstrap + permutation**

```python
    # ─── 3. Simplex Regression (Full Statistical Controls) ───────────
    section("3. Simplex Regression")

    for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
        step(f"Feature simplex regression on {name} (bootstrap={N_BOOTSTRAP}, perm={N_PERMUTATIONS})")
        t1 = time.time()
        reg = pc.tl.feature_simplex_regression(
            ad_obj, max_degree=2,
            n_bootstrap=N_BOOTSTRAP,
            permutation_test=True,
            n_permutations=N_PERMUTATIONS,
            robust_se=True,
        )
        elapsed = time.time() - t1
        done(f"({elapsed:.1f}s)")

        r2_d1 = reg["r_squared_degree1"]
        r2_d2 = reg["r_squared_degree2"]
        f_pval = reg["f_pvalue"]
        f_pval_fdr = reg["f_pvalue_fdr"]
        perm_pval = reg.get("permutation_pvalue")
        perm_pval_fdr = reg.get("permutation_pvalue_fdr")

        # Positive control: R2 distribution
        n_sig_f = int(np.sum(f_pval_fdr < ALPHA))
        n_sig_perm = int(np.sum(perm_pval_fdr < ALPHA)) if perm_pval_fdr is not None else "N/A"

        report([
            f"REGRESSION SUMMARY ({name}):",
            f"  Features tested: {len(r2_d1)}",
            f"  Degree-1 R2: median={np.median(r2_d1):.4f}, mean={np.mean(r2_d1):.4f}, max={np.max(r2_d1):.4f}",
            f"  Degree-2 R2: median={np.median(r2_d2):.4f}, mean={np.mean(r2_d2):.4f}, max={np.max(r2_d2):.4f}",
            f"  R2 improvement (d2-d1): median={np.median(r2_d2 - r2_d1):.4f}",
            f"",
            f"STATISTICAL CONTROLS:",
            f"  F-test significant (FDR<{ALPHA}): {n_sig_f}/{len(r2_d1)} ({100*n_sig_f/len(r2_d1):.1f}%)",
            f"  Permutation test significant (FDR<{ALPHA}): {n_sig_perm}",
            f"  Negative control: median F-test p-value = {np.median(f_pval):.4f}",
            f"    (expect ~0.5 for features with no archetype dependence)",
            f"",
            f"TOP 10 GENES (by R2):",
        ])
        top_idx = np.argsort(r2_d1)[-10:][::-1]
        for i in top_idx:
            gene = reg["feature_names"][i]
            ci_lo = reg["vertex_ci_lower"][i] if reg.get("vertex_ci_lower") is not None else None
            ci_hi = reg["vertex_ci_upper"][i] if reg.get("vertex_ci_upper") is not None else None
            ci_str = ""
            if ci_lo is not None:
                ci_range = np.max(ci_hi) - np.min(ci_lo)
                ci_str = f", CI width={ci_range:.3f}"
            perm_str = ""
            if perm_pval is not None:
                perm_str = f", perm_p={perm_pval[i]:.4f}"
            report([f"  {gene}: R2={r2_d1[i]:.4f}, F_p={f_pval[i]:.2e}{perm_str}{ci_str}"])

        # Interleaved visualizations
        step(f"  Coefficient heatmap ({name})")
        fig = pc.pl.coefficient_heatmap(ad_obj, top_n=30, show=False)
        save_viz(fig, f"coef_heatmap_{name}.html")
        done()

        step(f"  R2 barplot ({name})")
        fig = pc.pl.r2_barplot(ad_obj, top_n=30, show=False)
        save_viz(fig, f"r2_barplot_{name}.html")
        done()

        step(f"  Regression volcano ({name})")
        fig = pc.pl.regression_volcano(ad_obj, show=False)
        save_viz(fig, f"regression_volcano_{name}.html")
        done()
```

**Step 2: Run to verify**

Run: `/Users/honkala/miniconda3/envs/archetype/bin/python tests/test_e2e_hsc_v12.py 2>tests/e2e_outputs_v12/stderr.log`
Expected: Full regression with bootstrap + permutation, stats printed, 6 HTML files generated.

---

## Task 4: Pattern Classification with Controls + Viz

**Files:**
- Modify: `tests/test_e2e_hsc_v12.py`

**Step 1: Add Section 4**

```python
    # ─── 4. Pattern Classification ───────────────────────────────────
    section("4. Pattern Classification")

    for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
        step(f"Classifying feature patterns for {name}")
        patterns = pc.tl.classify_feature_patterns(ad_obj)
        counts = patterns["pattern_counts"]
        n_total = patterns["n_features"]
        done()

        report([
            f"PATTERN CLASSIFICATION ({name}):",
            f"  Total features classified: {n_total}",
        ])
        for ptype, pcount in sorted(counts.items(), key=lambda x: -x[1]):
            pct = 100 * pcount / n_total
            report([f"  {ptype}: {pcount} ({pct:.1f}%)"])

        # Negative control: expect mostly flat patterns
        n_flat = counts.get("flat", 0)
        flat_pct = 100 * n_flat / n_total
        report([
            f"",
            f"CONTROL: {flat_pct:.1f}% flat patterns",
            f"  (expect >80% — most genes should NOT be archetype-dependent)",
            f"  Non-flat patterns: {n_total - n_flat} ({100 - flat_pct:.1f}%)",
        ])
```

**Step 2: Run to verify**

---

## Task 5: GMM Simplex Decomposition + Viz

**Files:**
- Modify: `tests/test_e2e_hsc_v12.py`

**Step 1: Add Section 5**

```python
    # ─── 5. GMM Simplex Decomposition ────────────────────────────────
    section("5. GMM Simplex Decomposition")

    for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
        step(f"Fitting GMM on {name}")
        t1 = time.time()
        gmm = pc.tl.feature_simplex_decomposition(ad_obj, characterize_features=True)
        elapsed = time.time() - t1
        done(f"({elapsed:.1f}s)")

        report([
            f"GMM RESULTS ({name}):",
            f"  Optimal components (BIC): {gmm['n_components_optimal']}",
            f"  Stable components: {gmm['n_components_stable']}",
            f"  Stability scores: {gmm['component_stability_scores']}",
            f"  Archetype map: {gmm['component_archetype_map']}",
        ])

    # Viz on CMP only (representative)
    step("GMM visualizations (CMP)")
    for viz_name, viz_fn, filename in [
        ("component_scatter", lambda: pc.pl.component_scatter(adata_cmp, show=False), "gmm_scatter_CMP.html"),
        ("bic_curve", lambda: pc.pl.gmm_bic_curve(adata_cmp, show=False), "gmm_bic_CMP.html"),
        ("component_heatmap", lambda: pc.pl.component_heatmap(adata_cmp, top_n=30, show=False), "gmm_heatmap_CMP.html"),
        ("component_stability", lambda: pc.pl.component_stability(adata_cmp, show=False), "gmm_stability_CMP.html"),
    ]:
        fig = viz_fn()
        if fig is not None:
            save_viz(fig, filename)
    done()
```

---

## Task 6: Archetype Comparison (MMD, Similarity, Contrasts) + Viz

**Files:**
- Modify: `tests/test_e2e_hsc_v12.py`

**Step 1: Add Section 6**

```python
    # ─── 6. Archetype Comparison ─────────────────────────────────────
    section("6. Archetype Comparison")

    # 6a. Within-fit MMD
    for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
        step(f"Within-fit MMD for {name}")
        mmd_result = pc.tl.archetype_mmd(ad_obj, n_permutations=100)
        mask = ~np.eye(K, dtype=bool)
        mmd_vals = mmd_result["mmd_matrix"][mask]
        pvals = mmd_result["pvalue_matrix"][mask]
        done()
        report([
            f"WITHIN-FIT MMD ({name}):",
            f"  MMD range: {mmd_vals.min():.4f} - {mmd_vals.max():.4f}",
            f"  p-values (off-diag): min={pvals.min():.4f}, max={pvals.max():.4f}",
            f"  Significant pairs (p<{ALPHA}): {int(np.sum(pvals < ALPHA))}/{len(pvals)}",
            f"  CONTROL: All off-diagonal pairs should be significant",
            f"    (archetypes occupy distinct regions of the simplex)",
        ])

        fig = pc.pl.mmd_heatmap(ad_obj, show=False)
        save_viz(fig, f"mmd_heatmap_{name}.html")

    # 6b. Between-fit MMD
    step("Between-fit MMD (CMP vs Mono)")
    mmd_between = pc.tl.archetype_mmd(adata_cmp, adata_mono, n_permutations=100)
    done()
    report([
        f"BETWEEN-FIT MMD (CMP vs Mono):",
        f"  Matrix shape: {np.asarray(mmd_between['mmd_matrix']).shape}",
        f"  MMD matrix:",
        f"  {np.array2string(np.asarray(mmd_between['mmd_matrix']), precision=4)}",
        f"  CONTROL: Between-fit MMD should be larger than within-fit MMD",
        f"    (different cell types should occupy different simplex regions)",
    ])

    # 6c. Feature similarity
    for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
        step(f"Feature similarity for {name}")
        sim = pc.tl.archetype_feature_similarity(ad_obj)
        done()
        report([
            f"FEATURE SIMILARITY ({name}):",
            f"  Overall silhouette: {sim['silhouette_overall']:.3f}",
            f"  Per-archetype: {np.array2string(np.asarray(sim['silhouette_per_archetype']), precision=3)}",
            f"  CONTROL: Positive silhouette = archetypes have distinct feature profiles",
        ])
        fig = pc.pl.feature_similarity_heatmap(ad_obj, show=False)
        save_viz(fig, f"feature_similarity_{name}.html")

    # 6d. Between-fit feature similarity
    step("Between-fit feature similarity")
    sim_between = pc.tl.archetype_feature_similarity(adata_cmp, adata_mono)
    done()
    report([
        f"BETWEEN-FIT FEATURE SIMILARITY:",
        f"  Shared features: {sim_between['n_shared_features']}",
        f"  Spearman matrix:",
        f"  {np.array2string(np.asarray(sim_between['spearman_matrix']), precision=3)}",
        f"  CONTROL: Diagonal > off-diagonal = matched archetypes share features",
    ])

    # 6e. Wald contrasts
    for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
        step(f"Wald contrasts for {name}")
        contrasts = pc.tl.archetype_contrasts(ad_obj)
        done()
        report([f"WALD CONTRASTS ({name}):"])
        for pair in contrasts["pairs"]:
            key = str(tuple(pair)) if not isinstance(pair, str) else pair
            pvals_fdr = contrasts["pvalues_fdr"][key]
            n_sig = int(np.sum(pvals_fdr < ALPHA))
            n_total_genes = len(pvals_fdr)
            report([f"  Pair {pair}: {n_sig}/{n_total_genes} significant (FDR<{ALPHA})"])
        report([
            f"  CONTROL: Each archetype pair should have many significant genes",
            f"    (archetypes represent distinct transcriptional programs)",
        ])

        # Contrast volcano for first pair
        fig = pc.pl.contrast_volcano(ad_obj, pair=(0, 1), show=False)
        save_viz(fig, f"contrast_volcano_{name}_0v1.html")
```

---

## Task 7: HALLMARK Pathway Scoring

**Files:**
- Modify: `tests/test_e2e_hsc_v12.py`

**Step 1: Add Section 7 — Load HALLMARK, score, regress**

```python
    # ─── 7. HALLMARK Pathway Scoring ─────────────────────────────────
    section("7. HALLMARK Pathway Scoring & Regression")

    step("Loading HALLMARK gene sets")
    net = pc.pp.load_pathway_networks(["hallmark"], verbose=False)
    n_pathways = net["source"].nunique()
    n_genes_in_sets = net["target"].nunique()
    done(f"({n_pathways} pathways, {n_genes_in_sets} genes)")

    for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
        step(f"Computing pathway scores for {name}")
        pc.pp.compute_pathway_scores(ad_obj, net, verbose=False)
        pathway_names = ad_obj.uns["pathway_scores_pathways"]
        scores = ad_obj.obsm["pathway_scores"]
        done(f"({scores.shape[1]} pathways scored)")

        report([
            f"PATHWAY SCORING ({name}):",
            f"  Pathways scored: {scores.shape[1]}",
            f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]",
            f"  Mean score: {scores.mean():.4f}",
            f"  Most variable pathway: {pathway_names[scores.var(axis=0).argmax()]}",
        ])

        # Regress pathways on archetype weights
        step(f"Pathway simplex regression on {name}")
        t1 = time.time()
        pw_reg = pc.tl.pathway_simplex_regression(
            ad_obj, n_bootstrap=N_BOOTSTRAP, robust_se=True,
        )
        elapsed = time.time() - t1
        done(f"({elapsed:.1f}s)")

        pw_r2 = pw_reg["r_squared_degree1"]
        report([
            f"PATHWAY REGRESSION ({name}):",
            f"  Pathways tested: {len(pw_r2)}",
            f"  R2: median={np.median(pw_r2):.4f}, max={np.max(pw_r2):.4f}",
            f"  Significant (F-test FDR<{ALPHA}): {int(np.sum(pw_reg['f_pvalue_fdr'] < ALPHA))}/{len(pw_r2)}",
            f"",
            f"TOP 10 PATHWAYS (by R2):",
        ])
        top_pw = np.argsort(pw_r2)[-10:][::-1]
        for i in top_pw:
            report([f"  {pw_reg['feature_names'][i]}: R2={pw_r2[i]:.4f}, p={pw_reg['f_pvalue'][i]:.2e}"])
```

---

## Task 8: Flow Matching + Gene/Geneset Alignment

**Files:**
- Modify: `tests/test_e2e_hsc_v12.py`

**Step 1: Add Section 8 — flow training, gene alignment, geneset alignment**

```python
    # ─── 8. Flow Matching & Feature Alignment ────────────────────────
    section("8. Flow Matching (CMP -> Mono)")

    step("Building combined adata for flow")
    adata_cmp_flow = ad.AnnData(
        X=sp.csr_matrix((adata_cmp.n_obs, 1)),
        obs=adata_cmp.obs.copy(),
    )
    adata_cmp_flow.obsm["X_pca"] = cell_data["CMP"]["pca"]
    adata_cmp_flow.obs["cell_type_label"] = "CMP"

    adata_mono_flow = ad.AnnData(
        X=sp.csr_matrix((adata_mono.n_obs, 1)),
        obs=adata_mono.obs.copy(),
    )
    adata_mono_flow.obsm["X_pca"] = cell_data["Mono"]["pca"]
    adata_mono_flow.obs["cell_type_label"] = "Mono"

    adata_combined = ad.concat([adata_cmp_flow, adata_mono_flow])
    # Add PCA loadings for gene alignment
    if pca_loadings is not None:
        adata_combined.varm["PCs"] = pca_loadings
    done(f"({adata_combined.n_obs:,} cells)")

    step(f"Training flow CMP -> Mono ({FLOW_EPOCHS} epochs)")
    t1 = time.time()
    flow_result = pc.tl.flow_within(
        adata_combined,
        source={"cell_type_label": "CMP"},
        target={"cell_type_label": "Mono"},
        pca_key="X_pca",
        hidden_dims=(64, 64, 64),
        n_epochs=FLOW_EPOCHS,
        batch_size=256,
        n_steps=50,
        device="cpu",
    )
    elapsed = time.time() - t1
    mmd_reduction = (flow_result["mmd_before"] - flow_result["mmd_after"]) / flow_result["mmd_before"] * 100
    done(f"({elapsed:.1f}s)")
    report([
        f"FLOW TRAINING RESULTS:",
        f"  MMD before: {flow_result['mmd_before']:.4f}",
        f"  MMD after:  {flow_result['mmd_after']:.4f}",
        f"  MMD reduction: {mmd_reduction:.1f}%",
        f"  CONTROL: MMD should decrease substantially (>90% reduction expected)",
        f"    Good transport = source distribution mapped close to target",
    ])

    # Flow visualizations
    step("Flow visualizations")
    for viz_name, viz_fn, filename in [
        ("flow_magnitude", lambda: pc.pl.flow_magnitude(adata_combined, flow_result, show=False), "flow_magnitude.html"),
        ("density_comparison", lambda: pc.pl.density_comparison(adata_combined, flow_result, show=False), "density_comparison.html"),
        ("velocity_quiver", lambda: pc.pl.velocity_quiver(adata_combined, flow_result, show=False), "velocity_quiver.html"),
    ]:
        fig = viz_fn()
        if fig is not None:
            save_viz(fig, filename)
    done()

    # 8b. Gene-level flow alignment
    step("Gene-level flow alignment")
    n_top_genes = 50
    # Need var_names on combined adata for gene alignment
    adata_combined.var_names = var_names
    alignment = pc.tl.flow_gene_alignment(adata_combined, flow_result, n_top=n_top_genes)
    scores = alignment["alignment_scores"]
    done()
    report([
        f"GENE-LEVEL FLOW ALIGNMENT:",
        f"  Genes scored: {len(scores)}",
        f"  Alignment score range: [{scores.min():.4f}, {scores.max():.4f}]",
        f"  Mean |alignment|: {np.abs(scores).mean():.4f}",
        f"",
        f"TOP {n_top_genes} FLOW-ALIGNED GENES (upregulated CMP->Mono):",
    ])
    for gene in alignment["top_aligned"][:10]:
        idx = alignment["gene_names"].index(gene)
        report([f"  {gene}: score={scores[idx]:.4f}"])
    report([f"", f"TOP {n_top_genes} FLOW-OPPOSED GENES (downregulated CMP->Mono):"])
    for gene in alignment["top_opposed"][:10]:
        idx = alignment["gene_names"].index(gene)
        report([f"  {gene}: score={scores[idx]:.4f}"])

    # 8c. Geneset-level flow alignment (HALLMARK)
    step("HALLMARK geneset flow alignment")
    # Strategy: for each HALLMARK pathway, compute mean alignment score
    # of its member genes. This tells us which pathways are collectively
    # flow-aligned (activated) or flow-opposed (deactivated).
    gene_to_score = dict(zip(alignment["gene_names"], scores))
    pathway_gene_sets = net.groupby("source")["target"].apply(set).to_dict()

    pathway_alignment = {}
    for pw_name, pw_genes in pathway_gene_sets.items():
        matched = [gene_to_score[g] for g in pw_genes if g in gene_to_score]
        if len(matched) >= 5:  # require at least 5 genes for reliable estimate
            pathway_alignment[pw_name] = {
                "mean_score": np.mean(matched),
                "n_genes": len(matched),
                "n_aligned": sum(1 for s in matched if s > 0),
                "n_opposed": sum(1 for s in matched if s < 0),
            }
    done(f"({len(pathway_alignment)} pathways with >=5 genes)")

    # Sort by absolute mean alignment
    sorted_pw = sorted(pathway_alignment.items(), key=lambda x: abs(x[1]["mean_score"]), reverse=True)

    report([
        f"HALLMARK GENESET FLOW ALIGNMENT:",
        f"  Pathways scored: {len(pathway_alignment)}",
        f"",
        f"TOP 10 FLOW-ALIGNED PATHWAYS (activated CMP->Mono):",
    ])
    aligned_pw = sorted(pathway_alignment.items(), key=lambda x: x[1]["mean_score"], reverse=True)
    for pw_name, pw_info in aligned_pw[:10]:
        report([f"  {pw_name}: mean={pw_info['mean_score']:.4f} ({pw_info['n_aligned']}/{pw_info['n_genes']} genes aligned)"])

    report([f"", f"TOP 10 FLOW-OPPOSED PATHWAYS (deactivated CMP->Mono):"])
    opposed_pw = sorted(pathway_alignment.items(), key=lambda x: x[1]["mean_score"])
    for pw_name, pw_info in opposed_pw[:10]:
        report([f"  {pw_name}: mean={pw_info['mean_score']:.4f} ({pw_info['n_opposed']}/{pw_info['n_genes']} genes opposed)"])
```

Note: The combined adata for flow needs var_names set to `var_names` (the gene symbols). It was built from sparse placeholders so var_names defaults to integers. Set `adata_combined.var_names = var_names` before calling `flow_gene_alignment`. But adata_combined has 1-column X, not full gene expression. The `flow_gene_alignment` function uses `adata.varm["PCs"]` (PCA loadings) not X, so var_names just need to match the PCA loadings dimension.

Actually: `adata_combined` was built with 1-column X. Its var_names will be `["0"]`. For `flow_gene_alignment` we need:
- `adata.varm["PCs"]` with shape `[n_genes, n_PCs]`
- `adata.var_names` with `n_genes` entries

So we need to rebuild `adata_combined` with `n_genes` columns (can be zeros), or build a separate adata just for the alignment call. Simpler: build a thin adata with the right shape:

```python
    adata_for_alignment = ad.AnnData(
        X=sp.csr_matrix((adata_combined.n_obs, len(var_names))),
        var=pd.DataFrame(index=var_names),
    )
    adata_for_alignment.obsm["X_pca"] = adata_combined.obsm["X_pca"]
    if pca_loadings is not None:
        adata_for_alignment.varm["PCs"] = pca_loadings
    alignment = pc.tl.flow_gene_alignment(adata_for_alignment, flow_result, n_top=n_top_genes)
```

**Step 2: Run to verify**

---

## Task 9: Summary Section

**Files:**
- Modify: `tests/test_e2e_hsc_v12.py`

**Step 1: Update summary**

```python
    # ─── Summary ─────────────────────────────────────────────────────
    section("Summary")
    elapsed_total = time.time() - t0
    n_files = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.html')])
    print(f"  Total time: {elapsed_total:.1f}s", flush=True)
    print(f"  HTML files: {n_files} in {OUTPUT_DIR}", flush=True)
    print(f"  CMP: {adata_cmp.n_obs:,} cells, K={K}", flush=True)
    print(f"  Mono: {adata_mono.n_obs:,} cells, K={K}", flush=True)
    print(f"  Flow: MMD {flow_result['mmd_before']:.4f} -> {flow_result['mmd_after']:.4f}", flush=True)
    print(f"  Pathways: {n_pathways} HALLMARK sets scored", flush=True)
    print(f"\n  END-TO-END TEST PASSED", flush=True)
```

---

## Task 10: Full End-to-End Run + Fix Any Issues

**Step 1: Run full script**

Run: `/Users/honkala/miniconda3/envs/archetype/bin/python tests/test_e2e_hsc_v12.py 2>tests/e2e_outputs_v12/stderr.log`

**Step 2: Check stderr for errors**

Run: `cat tests/e2e_outputs_v12/stderr.log`

**Step 3: Fix any issues and re-run until PASSED**

---

## Task 11: Convert to Notebook

**Files:**
- Create: `docs/tutorials/12_e2e_v050_reviewer.ipynb`

**Step 1: Convert passing script to notebook**

Split the script into cells:
- Cell 0: Markdown title + overview
- Cell 1: Imports + config
- Cell 2-N: One cell per section, with markdown cells before each explaining the step
- Interleave markdown cells with `show=True` (for notebook display) instead of `save_path`
- Add markdown cells documenting the statistical controls and what each test means

Use `show=True` in notebook (not `show=False` + `save_path`).

**Step 2: Verify notebook runs**

Open in Jupyter, restart kernel, run all.

---

## Execution Notes

- Tasks 1-9 are sequential (each adds a section to the script)
- Task 10 is the integration test
- Task 11 is the notebook conversion (only after Task 10 passes)
- Estimated total runtime: ~15-20 minutes for full script (training + bootstrap + permutation on 28K genes)
- The PCA loadings need to be extracted from `adata_full` before it's deleted in Task 1

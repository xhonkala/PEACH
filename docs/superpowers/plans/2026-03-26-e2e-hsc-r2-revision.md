# E2E HSC Pipeline Round 2 Revision

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 6 root-cause bugs and address ~25 review items in the HSC e2e pipeline

**Architecture:** Three categories — (A) PEACH library fixes that fix root-cause bugs affecting all dotplots/plots, (B) e2e script lookup-order fixes that unblock downstream data flow, (C) e2e script viz/analysis improvements. A must precede B+C. B+C are independent.

**Tech Stack:** Python, PEACH (scanpy-style package), plotly, matplotlib, numpy, scipy

**Conda env:** `archetype`

---

## Root-Cause Summary

Six bugs discovered during exploration:

| # | Bug | Impact | Location |
|---|-----|--------|----------|
| RC1 | Dotplot sorts globally by max\|β\| instead of grouping by dominant archetype | All dotplots show ungrouped features | `src/peach/pl/regression.py:520` |
| RC2 | Pathway regression overwrites generic `peach_simplex_regression` key | Step 14 "?" archetypes, step 13 wrong cross-refs | `src/peach/tl/feature_regression.py:194` + 6 lookup sites in e2e script |
| RC3 | Steps 10/16 look in `obsm` for `archetype_coordinates` which only lives in `uns`; use matplotlib instead of `pc.pl.archetypal_space` | Broken archetype space plots | `scripts/run_e2e_hsc.py` steps 10, 16 |
| RC4 | BIC minimum always picks max components (penalty too weak for Dirichlet) | Step 8 selects 35 components | `src/peach/_core/utils/simplex_gmm.py:287` |
| RC5 | `flow_topo_landscape` returns matplotlib, e2e treats it as plotly; no param to disable velocity traces | Step 15 topo plots clogged | `src/peach/pl/flow.py:668` + `scripts/run_e2e_hsc.py` step 15 |
| RC6 | Soft assignment heatmap degenerates to rank-1 (outer product of near-constant archetype weights) | Step 12 all columns identical | `src/peach/pl/flow.py:956` |

---

## Task 1: Fix dotplot archetype grouping (RC1)

**Files:**
- Modify: `src/peach/pl/regression.py:515-520`

The y-axis sort at line 520 is `sorted(selected, key=lambda i: -np.max(np.abs(coefs[i])))`. This globally ranks by max |β|. We need two-level sort: primary = dominant archetype index, secondary = descending |β| within that archetype.

- [ ] **Step 1: Fix the sort in `archetype_regression_dotplot`**

In `src/peach/pl/regression.py`, replace line 520:

```python
selected = sorted(selected, key=lambda i: -np.max(np.abs(coefs[i])))
```

with:

```python
# Group features by dominant archetype, then rank by |beta| within group
selected = sorted(selected, key=lambda i: (np.argmax(np.abs(coefs[i])), -np.max(np.abs(coefs[i]))))
```

This sorts by (dominant_archetype_ascending, max_abs_beta_descending). Features for A1 appear at top, then A2, etc. Within each group, strongest signal first.

- [ ] **Step 2: Add archetype group separators to the dotplot**

After the sort, add visual separators. After line 546 (`gene_labels = [names[i] for i in selected]`), add archetype prefix to labels:

```python
# Prefix gene labels with dominant archetype for visual grouping
dom_archs = [np.argmax(np.abs(coefs[i])) for i in selected]
gene_labels = [f"[A{dom_archs[gi]+1}] {names[i]}" for gi, i in enumerate(selected)]
```

- [ ] **Step 3: Verify with quick test**

Run:
```bash
conda run -n archetype python -c "
import scanpy as sc, peach as pc
adata = sc.read_h5ad('/Users/honkala/Desktop/peach/data/hsc_10k.h5ad')
keep = ['hematopoietic stem cell','common myeloid progenitor','CD14-positive monocyte']
adata = adata[adata.obs['cell_type'].isin(keep)].copy()
adata.var_names = adata.var['gene_symbols'].values.copy()
sc.pp.pca(adata, n_comps=13)
pc.pp.prepare_training(adata)
pc.tl.train_archetypal(adata, n_archetypes=4, n_epochs=50)
pc.tl.archetypal_coordinates(adata)
pc.tl.assign_archetypes(adata)
pc.tl.extract_archetype_weights(adata)
pc.tl.gene_simplex_regression(adata, max_degree=1)
fig = pc.pl.archetype_regression_dotplot(adata, top_n=5, show=False)
# Check that y-axis labels are grouped by archetype
labels = [t['y'][0] if isinstance(t['y'], (list,tuple)) else t['y'] for t in fig.data if hasattr(t,'y')]
print('Y labels grouped?', labels)
print('SUCCESS')
"
```

Expected: Y-axis labels prefixed with `[A1]`, `[A2]`, etc., grouped by archetype.

---

## Task 2: Fix generic regression key overwrite (RC2)

**Files:**
- Modify: `src/peach/tl/feature_regression.py:191-194`
- Modify: `scripts/run_e2e_hsc.py` — 6 lookup sites

### Part A: Fix the library overwrite

- [ ] **Step 1: Make generic key write conditional**

In `src/peach/tl/feature_regression.py`, lines 191-194 currently:

```python
if store_to_adata:
    suffix = regression_storage_suffix(feature_matrix)
    store_result(adata, f"simplex_regression_{suffix}", serialized)
    store_result(adata, "simplex_regression", serialized)
```

Change to only write generic key when it's gene regression (suffix == "genes"):

```python
if store_to_adata:
    suffix = regression_storage_suffix(feature_matrix)
    store_result(adata, f"simplex_regression_{suffix}", serialized)
    if suffix == "genes":
        store_result(adata, "simplex_regression", serialized)
```

- [ ] **Step 2: Verify the suffix function**

Read `regression_storage_suffix` to confirm: gene regression returns `"genes"`, pathway returns `"pathways"`. If different, adjust the conditional.

### Part B: Fix all lookup sites in the e2e script

- [ ] **Step 3: Fix all 6 lookup sites**

Every occurrence of `adata.uns.get("peach_simplex_regression") or adata.uns.get("peach_simplex_regression_genes")` in `scripts/run_e2e_hsc.py` must be reversed to prefer the gene-specific key:

```python
# BEFORE (broken: returns pathway result after step 3 pathway regression)
gene_reg = adata.uns.get("peach_simplex_regression") or adata.uns.get("peach_simplex_regression_genes")

# AFTER (correct: prefers gene-specific key)
gene_reg = adata.uns.get("peach_simplex_regression_genes") or adata.uns.get("peach_simplex_regression")
```

Search for all occurrences with `grep -n 'peach_simplex_regression.*or.*peach_simplex_regression'` and fix each one. Known locations from exploration: lines ~1133, ~1415, ~2104, ~2189, ~2289, ~2330, ~2527.

- [ ] **Step 4: Verify archetype lookup now works**

After fixing, the archetype "?" in step 14 should resolve. The `arch_assoc` dict will be keyed by gene symbols (from the 2500-gene regression) instead of pathway names (from the 50-pathway regression).

---

## Task 3: Fix archetypal_space in steps 10 and 16 (RC3)

**Files:**
- Modify: `scripts/run_e2e_hsc.py` — steps 10 and 16 archetype space sections

The current code uses `sub.obsm.get("archetype_coordinates")` which always returns None (the key lives in `sub.uns`). Even if corrected, it renders a 2D matplotlib scatter. The PEACH function `pc.pl.archetypal_space(sub)` works — the subset adatas have all required keys after `run_subset_model`.

- [ ] **Step 1: Fix step 10 archetypal space**

Find the matplotlib scatter block in step 10 (around lines 1817-1842 currently). Replace the entire `fig, axes = plt.subplots(...)` block for archetypal space with per-group calls to the PEACH function:

```python
for gname, sub in lineage_adatas.items():
    try:
        fig_arch = pc.pl.archetypal_space(sub, color_by="cell_type_short",
                                           title=f"Archetypal space: {gname}",
                                           show=False)
        html += safe_plotly_html(report, fig_arch, f"Archetypal space: {gname}")
    except Exception as e:
        html += error_html(f"Archetypal space ({gname}) failed: {e}")
```

Do NOT try to facet them side by side — separate plotly figures are fine.

- [ ] **Step 2: Fix step 16 archetypal space**

Same pattern for step 16 (around lines 2794-2815). Replace the matplotlib scatter block with per-branch PEACH function calls:

```python
for branch, sub in branch_adatas.items():
    try:
        fig_arch = pc.pl.archetypal_space(sub, color_by="cell_type_short",
                                           title=f"Archetypal space: {branch}",
                                           show=False)
        html += safe_plotly_html(report, fig_arch, f"Archetypal space: {branch}")
    except Exception as e:
        html += error_html(f"Archetypal space ({branch}) failed: {e}")
```

---

## Task 4: Fix BIC component selection (RC4)

**Files:**
- Modify: `src/peach/_core/utils/simplex_gmm.py` — `_fit_dirichlet` and `_fit_gaussian`
- Modify: `scripts/run_e2e_hsc.py` — step 8 decomposition call

BIC minimum is inappropriate for Dirichlet mixtures: each component adds only K+1 ≈ 8 parameters, so BIC penalty is ~68 per component — trivial compared to likelihood improvement. Result: always picks max.

### Approach: Add BIC elbow detection

- [ ] **Step 1: Add elbow detection function to simplex_gmm.py**

Add a utility function at module level (near top, after imports):

```python
def _find_bic_elbow(n_range, bic_values):
    """Find elbow in BIC curve using second derivative (maximum curvature).

    Returns the n_components at the elbow, or argmin(BIC) if no elbow found.
    """
    bic = np.asarray(bic_values, dtype=float)
    ns = np.asarray(n_range, dtype=float)
    if len(bic) < 3:
        return n_range[int(np.argmin(bic))]

    # Normalize to [0,1] for curvature calculation
    ns_norm = (ns - ns.min()) / max(ns.max() - ns.min(), 1)
    bic_norm = (bic - bic.min()) / max(bic.max() - bic.min(), 1)

    # Second derivative (discrete)
    d2 = np.diff(bic_norm, 2)
    # Elbow = point of maximum positive second derivative (where curve bends from steep to flat)
    if len(d2) > 0 and np.any(d2 > 0):
        elbow_idx = int(np.argmax(d2)) + 1  # +1 because diff reduces length
        return n_range[elbow_idx]

    return n_range[int(np.argmin(bic))]
```

- [ ] **Step 2: Add `model_selection="bic_elbow"` option to `_fit_dirichlet`**

In the `_fit_dirichlet` function, after the BIC/ICL fitting loop that collects all scores, add an elbow detection branch. Find the section where `best_n` is determined from `best_score` and add:

```python
# After the fitting loop, if model_selection == "bic_elbow":
if model_selection == "bic_elbow":
    best_n = _find_bic_elbow(n_range, bic_values)
    # Re-fit at the elbow
    best_model = DirichletMixture(n_components=best_n, n_init=3, random_state=random_state)
    best_model.fit(weights)
```

This requires accumulating `bic_values` in a list during the loop. Add `bic_values = []` before the loop and `bic_values.append(bic)` inside.

- [ ] **Step 3: Same for `_fit_gaussian`**

Apply the identical pattern to `_fit_gaussian` (if it exists and has the same BIC minimum issue).

- [ ] **Step 4: Update e2e script to use bic_elbow and tighter range**

In `scripts/run_e2e_hsc.py`, step 8 decomposition call:

```python
K = adata.obsm["cell_archetype_weights"].shape[1]
decomp_result = pc.tl.feature_simplex_decomposition(
    adata, model_type="dirichlet",
    n_components_range=(K, 3 * K),  # 3*K not 5*K
    model_selection="bic_elbow",     # elbow instead of minimum
    n_initializations=20,
    stability_threshold=0.7,
)
```

- [ ] **Step 5: Add boundary warning**

After the decomposition call, check if optimal is at boundary:

```python
n_opt = decomp_result.get("n_components_optimal", 0)
_, max_c = (K, 3 * K)
if isinstance(n_opt, int) and n_opt >= max_c:
    html += error_html(
        f"WARNING: Optimal components ({n_opt}) at search boundary ({max_c}). "
        "Consider widening n_components_range."
    )
```

- [ ] **Step 6: Verify the `model_selection` parameter propagates**

Check `src/peach/tl/feature_decomposition.py` → `src/peach/_core/utils/simplex_gmm.py` to ensure `model_selection` is passed through. If not, add it.

---

## Task 5: Fix flow_topo_landscape velocity traces (RC5)

**Files:**
- Modify: `src/peach/pl/flow.py` — `flow_topo_landscape` function
- Modify: `scripts/run_e2e_hsc.py` — step 15

- [ ] **Step 1: Add `show_velocity` parameter to flow_topo_landscape**

In `src/peach/pl/flow.py`, add `show_velocity: bool = True` to the function signature. Wrap the quiver drawing block (lines ~668-709) in `if show_velocity:`.

- [ ] **Step 2: Increase cell scatter visibility**

In the same function, change source/target scatter parameters:
- `s=1` → `s=4`
- `alpha=0.03` → `alpha=0.15`

- [ ] **Step 3: Update e2e script step 15 to disable velocity**

In step 15, change the `flow_topo_landscape` call to pass `show_velocity=False`. Also remove the broken plotly trace-removal code that silently fails:

```python
fig_topo = pc.pl.flow_topo_landscape(
    adata, fr, model, n_features=5, n_eval_points=200,
    show_velocity=False,
    show=False,
    save=os.path.join(OUTPUT_DIR, f"topo_{pair_key}.png"),
)
# Remove the broken plotly trace-removal try/except block
```

---

## Task 6: Fix soft assignment heatmap (RC6)

**Files:**
- Modify: `scripts/run_e2e_hsc.py` — step 12

The rank-1 degeneracy happens because cells within one cell type have near-identical archetype weights. The fix: instead of the outer-product correspondence, use a discrete assignment transition matrix.

- [ ] **Step 1: Add a discrete transition matrix after the PEACH heatmap**

After the `soft_assignment_heatmap` call in step 12, add a secondary analysis that computes the transition matrix from discrete archetype assignments (more robust than soft weights):

```python
# Discrete archetype transition matrix (more robust than soft assignment)
try:
    src_archs = adata.obs.loc[fr["source_mask"], "archetypes"].values
    # Assign transported cells to nearest target archetype
    transported = fr["transported"]
    arch_positions = np.asarray(adata.uns["archetype_coordinates"])
    dists = np.linalg.norm(transported[:, None, :] - arch_positions[None, :, :], axis=2)
    tgt_archs = np.argmin(dists, axis=1)
    # Build transition counts
    K = arch_positions.shape[0]
    trans_matrix = np.zeros((K, K))
    for sa, ta in zip(src_archs, tgt_archs):
        si = int(str(sa).replace("A", "")) - 1 if isinstance(sa, str) else int(sa)
        trans_matrix[si, ta] += 1
    # Normalize rows to proportions
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_prop = trans_matrix / row_sums

    fig_trans, ax_trans = plt.subplots(figsize=(6, 5))
    im_trans = ax_trans.imshow(trans_prop, cmap="YlOrRd", vmin=0, vmax=1)
    ax_trans.set_xticks(range(K))
    ax_trans.set_xticklabels([f"A{k+1}\n(target)" for k in range(K)])
    ax_trans.set_yticks(range(K))
    ax_trans.set_yticklabels([f"A{k+1}\n(source)" for k in range(K)])
    for i in range(K):
        for j in range(K):
            ax_trans.text(j, i, f"{trans_prop[i,j]:.0%}", ha="center", va="center",
                         fontsize=9, color="white" if trans_prop[i,j] > 0.5 else "black")
    plt.colorbar(im_trans, ax=ax_trans, label="Proportion", shrink=0.8)
    ax_trans.set_title(f"Discrete archetype transition: {pair_key}")
    fig_trans.tight_layout()
    html += report.fig_to_img(fig_trans,
                              caption=f"Discrete transition matrix: {pair_key} (rows=source archetype, cols=target after transport)")
    plt.close("all")
except Exception as e:
    html += error_html(f"Discrete transition matrix ({pair_key}) failed: {e}")
    plt.close("all")
```

- [ ] **Step 2: Remove the Sankey placeholder note**

Replace the note about Sankey being "available in notebook mode" with the actual discrete transition matrix we just added. Delete lines that say "A Sankey diagram ... would be informative here."

---

## Task 7: Run naming (non-overwriting outputs)

**Files:**
- Modify: `scripts/run_e2e_hsc.py` — constants section near top

- [ ] **Step 1: Add run timestamp to output directory**

Near line 62, change:

```python
OUTPUT_DIR = "outputs/e2e_hsc"
REPORT_PATH = os.path.join(OUTPUT_DIR, "e2e_hsc_report.html")
```

to:

```python
_RUN_ID = time.strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"outputs/e2e_hsc/{_RUN_ID}"
REPORT_PATH = os.path.join(OUTPUT_DIR, "e2e_hsc_report.html")
```

Also add a symlink to latest at the end of main():

```python
latest_link = "outputs/e2e_hsc/latest"
if os.path.islink(latest_link):
    os.unlink(latest_link)
os.symlink(_RUN_ID, latest_link)
```

---

## Task 8: E2E script improvements (non-root-cause)

**Files:**
- Modify: `scripts/run_e2e_hsc.py` — various steps

These are independent of root-cause fixes and can be done in parallel.

### 8a. Step 1: Use C5:BP pathways

- [ ] **Step 1: Change pathway collection**

Find the `compute_pathway_scores` call. Change to use C5:BP (biological processes) collection:

```python
pc.pp.compute_pathway_scores(adata, collection="C5:BP")
```

### 8b. Step 4: Fix heatmap vmin/vmax

- [ ] **Step 2: Fix enrichment heatmap LogNorm**

In step 4, the `vmin`/`vmax` calculation can produce invalid values when odds_ratio contains inf or 0. Fix:

```python
finite_vals = pivot.values[np.isfinite(pivot.values) & (pivot.values > 0)]
if len(finite_vals) == 0:
    html += error_html(f"Enrichment heatmap ({col}): no finite positive odds ratios")
    continue
vmin = max(0.1, finite_vals.min())
vmax = max(vmin * 10, finite_vals.max())
# Clip inf values for display
display_vals = np.clip(pivot.values, vmin, vmax)
```

### 8c. Step 5: Volcano font size + gene set contrasts

- [ ] **Step 3: Double volcano font size**

In step 5 volcano grid, change `textfont_size=11` to `textfont_size=18` and increase subplot size:

```python
fig_vg.update_traces(textfont_size=18)
fig_vg.update_layout(font=dict(size=15), width=1800, height=1200)
```

- [ ] **Step 4: Add gene set contrasts**

After gene contrasts, add pathway contrasts. If pathway regression results exist, run contrasts on those:

```python
pw_reg = adata.uns.get("peach_simplex_regression_pathways")
if pw_reg is not None:
    html += "<h3>Gene Set Contrasts</h3>"
    try:
        # Temporarily swap regression for contrasts
        _gene_reg_stash = adata.uns.pop("peach_simplex_regression", None)
        adata.uns["peach_simplex_regression"] = pw_reg
        pw_contrast = pc.tl.archetype_contrasts(adata)
        # Restore
        if _gene_reg_stash is not None:
            adata.uns["peach_simplex_regression"] = _gene_reg_stash
        # Display top pathway contrasts per pair
        pw_feat_names = list(pw_contrast.get("feature_names", []))
        for pair in pw_contrast.get("pairs", []):
            pair_key = str(tuple(pair) if isinstance(pair, list) else pair)
            pvals = np.asarray(pw_contrast["pvalues_fdr"][pair_key])
            delta = np.asarray(pw_contrast["delta_beta"][pair_key])
            j, k = pair
            sig_mask = pvals < 0.05
            if sig_mask.any():
                sig_idx = np.where(sig_mask)[0]
                sorted_sig = sig_idx[np.argsort(np.abs(delta[sig_idx]))[::-1]][:10]
                rows = []
                for i in sorted_sig:
                    rows.append({
                        "Pathway": pw_feat_names[i],
                        "Δβ": f"{delta[i]:.3f}",
                        "FDR q": f"{pvals[i]:.2e}",
                        "Direction": f"higher in A{j+1}" if delta[i] > 0 else f"higher in A{k+1}",
                    })
                html += report.df_to_html(pd.DataFrame(rows),
                                          caption=f"Top pathway contrasts: A{j+1} vs A{k+1}")
    except Exception as e:
        html += error_html(f"Pathway contrasts failed: {e}")
```

### 8d. Step 6: Statistical notes

- [ ] **Step 5: Add MMD statistical notes**

After MMD results in step 6, add:

```python
html += report.text(
    "MMD basis: maximum mean discrepancy computed on cell archetype weight vectors. "
    "Permutation test shuffles archetype labels to build null distribution. "
    "Spearman ρ computed across all FDR-significant features from simplex regression (not intersection-only)."
)
```

### 8e. Step 11: Gene set level conserved/exclusive

- [ ] **Step 6: Add gene set analysis in step 11**

After per-lineage gene regression, add pathway-level conserved/exclusive analysis. If pathway regression results exist in the per-lineage subsets, compare at pathway level:

```python
html += "<h3>Conserved vs Exclusive Pathways Across Lineages</h3>"
# Compare pathway regression results between lineage subsets
```

### 8f. Step 13: Alignment table with magnitudes

- [ ] **Step 7: Enrich alignment table**

In step 13, the alignment table (lines ~2142-2149) is just a list of gene names. Replace with a DataFrame including magnitude, FDR p-value, and archetype association:

```python
# Replace the simple aligned/opposed two-column table with a detailed one
align_scores = align.get("alignment_scores", [])
align_pvals = align.get("alignment_pvalues_fdr", np.ones_like(align_scores))
if len(align_scores) > 0:
    sorted_idx = np.argsort(align_scores)
    top_aligned_idx = sorted_idx[-20:][::-1]
    top_opposed_idx = sorted_idx[:20]
    for label, indices in [("Flow-aligned (top 20)", top_aligned_idx),
                            ("Flow-opposed (top 20)", top_opposed_idx)]:
        rows = []
        for gi in indices:
            gname = adata.var_names[gi]
            rows.append({
                "Gene": ensembl_to_symbol(adata, [gname])[0],
                "Alignment score": f"{align_scores[gi]:.4f}",
                "FDR q": f"{align_pvals[gi]:.2e}" if gi < len(align_pvals) else "N/A",
                "Significant": "Yes" if gi < len(align_pvals) and align_pvals[gi] < 0.05 else "No",
                "Archetype": f"A{arch_assoc.get(gname, '?')}" if gname in arch_assoc else "?",
            })
        html += report.df_to_html(pd.DataFrame(rows), caption=f"{label}: {pair_key}")
```

### 8g. Step 15: Expansion plot redesign

- [ ] **Step 8: Fix expansion violin reference line and redesign**

Change `axhline(0.0)` reference line (already done in prior batch — verify it's 0.0 not 1.0).

Replace the current per-gene violin subplots with a scatter of expression vs flow displacement:

```python
# Per-gene: x = source expression, y = expression change (delta from flow transport)
# Using PCA reconstruction for expression change
if "PCs" in adata.varm:
    source_pca = adata.obsm["X_pca"][fr["source_mask"]]
    transported = fr["transported"]
    delta_pca = transported - source_pca
    loadings = adata.varm["PCs"]
    n_pcs = delta_pca.shape[1]
    delta_expr = delta_pca @ loadings[:, :n_pcs].T
    source_expr = source_pca @ loadings[:, :n_pcs].T

    for gene_name in top_gene_names[:5]:
        gi = list(adata.var_names).index(gene_name)
        fig_exp, ax_exp = plt.subplots(figsize=(5, 4))
        ax_exp.scatter(source_expr[:, gi], delta_expr[:, gi], s=2, alpha=0.3, c="#0072B2")
        ax_exp.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax_exp.set_xlabel("Source expression (PCA reconstructed)")
        ax_exp.set_ylabel("Expression change along flow")
        ax_exp.set_title(gene_name)
        ax_exp.spines[["top", "right"]].set_visible(False)
        fig_exp.tight_layout()
        html += report.fig_to_img(fig_exp, caption=f"Expression vs flow change: {gene_name}")
        plt.close("all")
```

---

## Task 9: Radar plot similarity ordering

**Files:**
- Modify: `src/peach/pl/regression.py` — `archetype_radar` function

- [ ] **Step 1: Add `order_by_similarity` parameter**

Add `order_by_similarity: bool = False` to the `archetype_radar` signature. When True, compute Spearman correlation between archetype vertex coefficient vectors, then use spectral ordering (Fiedler vector) to determine angular position:

```python
if order_by_similarity:
    from scipy.spatial.distance import squareform, pdist
    from scipy.sparse.csgraph import laplacian
    # Compute Spearman correlation between archetype columns
    from scipy.stats import spearmanr
    corr_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            corr_matrix[i, j], _ = spearmanr(coefs[:, i], coefs[:, j])
    # Convert to distance, compute Laplacian, get Fiedler vector
    dist_matrix = 1 - corr_matrix
    np.fill_diagonal(dist_matrix, 0)
    L = laplacian(np.maximum(0, corr_matrix), normed=True)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    fiedler = eigenvectors[:, 1]  # second smallest eigenvalue
    order = np.argsort(fiedler)
    arch_labels = [f"A{order[k]+1}" for k in range(K)]
    coefs = coefs[:, order]  # Reorder columns
```

- [ ] **Step 2: Enable in e2e script**

Change the radar call in step 3:

```python
fig_radar = pc.pl.archetype_radar(adata, top_n=8, order_by_similarity=True, show=False)
```

Update the note to say angles ARE reflecting similarity.

---

## Task 10: Cross-step synthesis fixes

**Files:**
- Modify: `scripts/run_e2e_hsc.py` — `step_synthesis` function

- [ ] **Step 1: Fix component distribution shift (must sum to 100%)**

In the component distribution tracking section of `step_synthesis`, ensure distributions are normalized:

```python
src_dist = pd.Series(assignments[source_mask]).value_counts(normalize=True)
tgt_dist = pd.Series(assignments[target_mask]).value_counts(normalize=True)
# Align indices
all_comps = sorted(set(src_dist.index) | set(tgt_dist.index))
src_dist = src_dist.reindex(all_comps, fill_value=0.0)
tgt_dist = tgt_dist.reindex(all_comps, fill_value=0.0)
```

- [ ] **Step 2: Match components across fits by MMD before comparing**

For per-lineage or per-branch component comparisons, components are fit independently and won't have matching indices. Add a Hungarian matching step:

```python
from scipy.optimize import linear_sum_assignment
# Compute pairwise MMD or centroid distance between components across fits
# Then match using Hungarian algorithm
cost_matrix = ...  # [n_comp_1, n_comp_2] distance matrix
row_ind, col_ind = linear_sum_assignment(cost_matrix)
# Reindex component 2 to match component 1
```

This is relevant in the synthesis step when comparing source vs target component distributions that come from different archetype model fits.

- [ ] **Step 3: Add archetype-exclusive feature labels to synthesis table**

In the gene overlap section, cross-reference with archetype-exclusive features from step 3:

```python
# For each aligned+expanding gene, look up which archetype it's exclusive to
gene_reg = adata.uns.get("peach_simplex_regression_genes")
if gene_reg:
    feat_names = list(gene_reg["feature_names"])
    vertex_coefs = np.asarray(gene_reg["vertex_coefficients"])
    for gene in overlap_genes:
        if gene in feat_names:
            idx = feat_names.index(gene)
            dom_arch = int(np.argmax(np.abs(vertex_coefs[idx]))) + 1
            # Add to synthesis table row
```

---

## Task 11: Statistical controls (bootstrapping + permutation nulls)

**Files:**
- Modify: `src/peach/tl/feature_regression.py` — add bootstrap CI
- Modify: `src/peach/tl/comparison.py` — add label permutation null for MMD
- Create: `src/peach/_core/utils/permutation.py` — shared permutation utilities
- Create: `tests/test_core/test_statistical_controls.py` — unit tests for all controls
- Modify: `scripts/run_e2e_hsc.py` — wire controls into pipeline steps

### Context

Paper Part 1 requires permutation/bootstrap controls at 4 pipeline steps. Part 2's dispositive test (R/NR label-swap) depends on this infrastructure being solid. All controls share a common pattern: (1) compute observed statistic, (2) generate null distribution by shuffling labels/weights, (3) compute empirical p-value, (4) FDR correct across features.

### 11a. Shared permutation utilities

- [ ] **Step 1: Create `src/peach/_core/utils/permutation.py`**

```python
"""Shared permutation testing utilities for PEACH statistical controls."""

import numpy as np
from statsmodels.stats.multitest import multipletests


def permutation_pvalue(observed, null_distribution, alternative="two-sided"):
    """Compute empirical p-value from a null distribution.

    Parameters
    ----------
    observed : float or np.ndarray
        Observed test statistic(s). Shape [n_features] for vectorized.
    null_distribution : np.ndarray
        Null samples. Shape [n_permutations] for scalar, or [n_permutations, n_features].
    alternative : str
        "two-sided", "greater", or "less".

    Returns
    -------
    p_value : float or np.ndarray
        Empirical p-value(s).
    """
    null = np.asarray(null_distribution)
    obs = np.asarray(observed)
    n_perm = null.shape[0]

    if alternative == "greater":
        count = (null >= obs).sum(axis=0)
    elif alternative == "less":
        count = (null <= obs).sum(axis=0)
    else:  # two-sided
        count = (np.abs(null) >= np.abs(obs)).sum(axis=0)

    # +1/+1 correction (Phipson & Smyth 2010)
    return (count + 1) / (n_perm + 1)


def fdr_correct(pvalues, alpha=0.05, method="fdr_bh"):
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvalues : np.ndarray
        Raw p-values.
    alpha : float
        FDR threshold.
    method : str
        Correction method (default: Benjamini-Hochberg).

    Returns
    -------
    rejected : np.ndarray[bool]
        Which hypotheses are rejected.
    pvalues_corrected : np.ndarray
        Corrected p-values.
    """
    pvals = np.asarray(pvalues).ravel()
    # Handle NaN/inf
    valid = np.isfinite(pvals)
    corrected = np.ones_like(pvals)
    rejected = np.zeros_like(pvals, dtype=bool)
    if valid.sum() > 0:
        rej, corr, _, _ = multipletests(pvals[valid], alpha=alpha, method=method)
        rejected[valid] = rej
        corrected[valid] = corr
    return rejected, corrected


def bootstrap_ci(data, statistic_fn, n_bootstrap=1000, ci=0.95, seed=42):
    """Bootstrap confidence interval for a statistic.

    Parameters
    ----------
    data : np.ndarray
        Input data, shape [n_samples, ...].
    statistic_fn : callable
        Function that takes data array and returns scalar or 1D array.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (default 0.95).
    seed : int
        Random seed.

    Returns
    -------
    point_estimate : float or np.ndarray
        Statistic on original data.
    ci_low : float or np.ndarray
        Lower CI bound.
    ci_high : float or np.ndarray
        Upper CI bound.
    """
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    point_estimate = statistic_fn(data)

    boot_stats = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_stats.append(statistic_fn(data[idx]))
    boot_stats = np.array(boot_stats)

    alpha = 1 - ci
    ci_low = np.percentile(boot_stats, 100 * alpha / 2, axis=0)
    ci_high = np.percentile(boot_stats, 100 * (1 - alpha / 2), axis=0)
    return point_estimate, ci_low, ci_high
```

- [ ] **Step 2: Write unit tests for shared utilities**

Create `tests/test_core/test_statistical_controls.py`:

```python
"""Tests for PEACH statistical controls: permutation, FDR, bootstrap."""

import numpy as np
import pytest
from peach._core.utils.permutation import permutation_pvalue, fdr_correct, bootstrap_ci


class TestPermutationPvalue:
    def test_extreme_observed_gets_small_pvalue(self):
        """An observed statistic far from the null should yield p < 0.05."""
        null = np.random.default_rng(42).normal(0, 1, size=999)
        observed = 5.0  # far from null
        p = permutation_pvalue(observed, null, alternative="greater")
        assert p < 0.01

    def test_null_observed_gets_large_pvalue(self):
        """An observed statistic drawn from the null should yield p > 0.05."""
        rng = np.random.default_rng(42)
        null = rng.normal(0, 1, size=999)
        observed = 0.1  # well within null
        p = permutation_pvalue(observed, null, alternative="two-sided")
        assert p > 0.1

    def test_vectorized(self):
        """Should handle array of observed values against null matrix."""
        rng = np.random.default_rng(42)
        null = rng.normal(0, 1, size=(999, 50))
        observed = np.concatenate([np.full(10, 5.0), np.full(40, 0.1)])
        p = permutation_pvalue(observed, null, alternative="two-sided")
        assert p.shape == (50,)
        assert np.all(p[:10] < 0.05)
        assert np.all(p[10:] > 0.05)

    def test_phipson_smyth_correction(self):
        """P-value should never be exactly 0 due to +1/+1 correction."""
        null = np.zeros(999)
        observed = 100.0
        p = permutation_pvalue(observed, null, alternative="greater")
        assert p > 0  # (0+1)/(999+1) = 0.001

    def test_alternative_less(self):
        """alternative='less' should detect negative extremes."""
        null = np.random.default_rng(42).normal(0, 1, size=999)
        p = permutation_pvalue(-5.0, null, alternative="less")
        assert p < 0.01


class TestFDRCorrect:
    def test_all_significant(self):
        pvals = np.full(10, 1e-10)
        rejected, corrected = fdr_correct(pvals)
        assert np.all(rejected)
        assert np.all(corrected < 0.05)

    def test_none_significant(self):
        pvals = np.full(10, 0.5)
        rejected, corrected = fdr_correct(pvals)
        assert not np.any(rejected)

    def test_mixed(self):
        pvals = np.array([1e-10, 1e-8, 0.01, 0.5, 0.9])
        rejected, corrected = fdr_correct(pvals)
        assert rejected[0] and rejected[1]
        assert not rejected[-1]

    def test_handles_nan(self):
        pvals = np.array([1e-10, np.nan, 0.5])
        rejected, corrected = fdr_correct(pvals)
        assert rejected[0]
        assert corrected[1] == 1.0  # NaN → 1.0


class TestBootstrapCI:
    def test_known_mean(self):
        """Bootstrap CI for mean of N(5,1) should contain 5."""
        rng = np.random.default_rng(42)
        data = rng.normal(5, 1, size=(1000, 1))
        point, lo, hi = bootstrap_ci(data, lambda x: x.mean(), n_bootstrap=500)
        assert lo < 5.0 < hi

    def test_narrow_with_large_sample(self):
        """CI should be narrow with large sample."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, size=(10000, 1))
        _, lo, hi = bootstrap_ci(data, lambda x: x.mean(), n_bootstrap=500)
        assert (hi - lo) < 0.1

    def test_wide_with_small_sample(self):
        """CI should be wider with small sample."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, size=(20, 1))
        _, lo, hi = bootstrap_ci(data, lambda x: x.mean(), n_bootstrap=500)
        assert (hi - lo) > 0.1
```

- [ ] **Step 3: Run tests**

```bash
conda run -n archetype pytest tests/test_core/test_statistical_controls.py -v
```

Expected: all pass.

### 11b. Simplex regression permutation test

- [ ] **Step 4: Add `permutation_test` to `gene_simplex_regression`**

Check if `feature_regression.py` already has a `permutation_test` parameter. If not, add it. The permutation shuffles `cell_archetype_weights` row-wise (breaking cell-gene correspondence) and re-runs the Scheffe regression to build a null distribution of R² and F-statistics per feature.

In `src/peach/tl/feature_regression.py`, add to the regression function:

```python
if permutation_test and n_permutations > 0:
    from peach._core.utils.permutation import permutation_pvalue, fdr_correct
    null_r2 = np.zeros((n_permutations, n_features))
    rng = np.random.default_rng(seed)
    for p in range(n_permutations):
        shuffled_weights = weights[rng.permutation(n_cells)]
        # Re-run regression with shuffled weights
        null_result = _scheffe_regression(X_features, shuffled_weights, max_degree=max_degree)
        null_r2[p] = null_result["r_squared_degree1"]
    perm_pvals = permutation_pvalue(r2_d1, null_r2, alternative="greater")
    _, perm_fdr = fdr_correct(perm_pvals)
    result["permutation_pvalues"] = perm_pvals
    result["permutation_pvalues_fdr"] = perm_fdr
```

- [ ] **Step 5: Add test for simplex regression permutation**

In `tests/test_core/test_statistical_controls.py`, add:

```python
class TestSimplexRegressionPermutation:
    def test_real_signal_detected(self):
        """Features with known archetype structure should have low permutation p-values."""
        import peach as pc
        from peach._core.utils.synthetic import generate_synthetic_simplex
        adata = generate_synthetic_simplex(n_cells=500, n_archetypes=3, n_genes=100, seed=42)
        # The first few genes have strong archetype signal by construction
        pc.tl.gene_simplex_regression(adata, max_degree=1, permutation_test=True,
                                       n_permutations=100)
        reg = adata.uns["peach_simplex_regression_genes"]
        perm_fdr = np.asarray(reg["permutation_pvalues_fdr"])
        # At least some features should be significant
        assert (perm_fdr < 0.05).sum() > 0

    def test_noise_not_detected(self):
        """Random noise features should not be significant."""
        import anndata as ad
        rng = np.random.default_rng(42)
        n_cells, n_genes, K = 500, 50, 3
        weights = rng.dirichlet(np.ones(K), size=n_cells)
        X = rng.normal(0, 1, size=(n_cells, n_genes))  # pure noise
        adata = ad.AnnData(X)
        adata.obsm["cell_archetype_weights"] = weights
        adata.var_names = [f"noise_{i}" for i in range(n_genes)]
        import peach as pc
        pc.tl.gene_simplex_regression(adata, max_degree=1, permutation_test=True,
                                       n_permutations=100)
        reg = adata.uns["peach_simplex_regression_genes"]
        perm_fdr = np.asarray(reg["permutation_pvalues_fdr"])
        # Very few should be significant (allow FDR-level false positives)
        assert (perm_fdr < 0.05).sum() < n_genes * 0.1
```

- [ ] **Step 6: Wire into e2e script**

In `scripts/run_e2e_hsc.py` step 3, add `permutation_test=True, n_permutations=200` to the gene_simplex_regression call. Report permutation-significant count alongside F-test count.

### 11c. Bootstrap CI on Wald contrast Δβ

- [ ] **Step 7: Add bootstrap CI to `archetype_contrasts`**

In `src/peach/tl/comparison.py` (or wherever `archetype_contrasts` lives), add `bootstrap_ci=False, n_bootstrap=500` parameters. When enabled, resample cells with replacement, recompute Δβ for each pair, and report 95% CI:

```python
if bootstrap_ci and n_bootstrap > 0:
    from peach._core.utils.permutation import bootstrap_ci as _bootstrap_ci
    for pair_key, pair in zip(pair_keys, pairs):
        j, k = pair
        def delta_beta_fn(data_subset):
            # Refit regression on bootstrap sample, compute Δβ
            ...
        _, ci_lo, ci_hi = _bootstrap_ci(cell_data, delta_beta_fn, n_bootstrap=n_bootstrap)
        result[f"delta_beta_ci_low"][pair_key] = ci_lo
        result[f"delta_beta_ci_high"][pair_key] = ci_hi
```

- [ ] **Step 8: Add test for Wald bootstrap CI**

```python
class TestWaldBootstrapCI:
    def test_ci_contains_true_value(self):
        """Bootstrap CI should contain the point estimate."""
        # Use synthetic data with known archetype structure
        ...

    def test_ci_excludes_zero_for_real_differences(self):
        """For features with true archetype differences, CI should not contain 0."""
        ...
```

- [ ] **Step 9: Wire into e2e script step 5**

Add `bootstrap_ci=True, n_bootstrap=200` to the contrasts call. Display CI in the top contrasts table.

### 11d. Jacobian permutation null

- [ ] **Step 10: Add permutation test to `flow_jacobian`**

In `src/peach/tl/flow.py` `flow_jacobian` function, add `n_permutations=0` parameter. When > 0, shuffle PCA loadings across genes and recompute feature_expansion to build null:

```python
if n_permutations > 0:
    from peach._core.utils.permutation import permutation_pvalue, fdr_correct
    null_expansion = np.zeros((n_permutations, n_features))
    rng = np.random.default_rng(42)
    loadings = adata.varm["PCs"][:, :n_pcs]
    for p in range(n_permutations):
        shuffled_loadings = loadings[rng.permutation(loadings.shape[0])]
        # Recompute L^T J L for each gene with shuffled loadings
        for gi in range(n_features):
            L = shuffled_loadings[gi]
            null_expansion[p, gi] = L @ J @ L
    perm_pvals = permutation_pvalue(expansion, null_expansion, alternative="two-sided")
    _, perm_fdr = fdr_correct(perm_pvals)
    result["expansion_pvalues"] = perm_pvals
    result["expansion_pvalues_fdr"] = perm_fdr
```

- [ ] **Step 11: Add test for Jacobian permutation**

```python
class TestJacobianPermutation:
    def test_expanding_gene_detected(self):
        """A gene whose PCA loading aligns with flow Jacobian should be significant."""
        ...

    def test_random_loading_not_significant(self):
        """Random PCA loadings should produce non-significant expansion scores."""
        ...
```

- [ ] **Step 12: Wire into e2e script step 14**

Add `n_permutations=200` to the `flow_jacobian` call. Display FDR-significant count. Remove the "no permutation null" warning text added in prior batch (now it has one).

### 11e. Label permutation null for MMD

- [ ] **Step 13: Add label permutation to `archetype_mmd`**

Check if `archetype_mmd` already computes permutation p-values (the exploration said it does by default with `n_permutations=1000`). If so, this step is just about displaying them — already handled in Batch 2. If the permutation shuffles cell assignments within archetypes but not archetype labels themselves, add a separate label-permutation mode for the e2e step 6 where we want to test "are these archetypes actually different?"

- [ ] **Step 14: Wire MMD p-values into e2e step 6 display**

Verify MMD p-values are being extracted and displayed (should already be done from prior batch). Add the label-permutation interpretation note.

### 11f. Infrastructure for Part 2 R/NR label-swap (design only, defer implementation)

- [ ] **Step 15: Document R/NR label-swap design in the plan**

This is the dispositive test for the biological hypothesis and will be implemented when Part 2 data is ready. Document the design:

```
R/NR Label-Swap Permutation Protocol:
1. Pool all cells from R and NR lineages at each timepoint
2. For each permutation (n=1000):
   a. Randomly reassign R/NR labels (preserving per-timepoint cell counts)
   b. Fit archetype model on permuted data
   c. Compute stress diversity metrics (alpha: Shannon on stress pathway scores;
      beta: Bray-Curtis on archetype composition between timepoints)
   d. Record R_diversity - NR_diversity
3. Compare observed R-NR diversity difference to null distribution
4. FDR correct across timepoints
```

Add as a comment block in the e2e script near step 12 or as a separate planning document. This does NOT need implementation now — just the design documented so it can be picked up later.

---

## Execution Order

**Must be sequential:**
1. Tasks 1-2 (library fixes) → must come first, all downstream depends on correct dotplot grouping and regression key lookup
2. Tasks 3-6 (more library + script fixes) → independent of each other, can run in parallel
3. Tasks 7-10 (script improvements) → independent, can run in parallel
4. Task 11 (statistical controls) → depends on Tasks 1-2 for correct regression results, can otherwise run in parallel with 7-10

**Estimated total:** Tasks 1-2 are ~15 min each. Tasks 3-6 are ~10 min each. Tasks 7-10 are ~15 min each. Task 11 is ~45 min. Total ~3-4 hours of agent time.

**After all tasks:** Run full pipeline to verify:
```bash
conda run -n archetype python scripts/run_e2e_hsc.py
```

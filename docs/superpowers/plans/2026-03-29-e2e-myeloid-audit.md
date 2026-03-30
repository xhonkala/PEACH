# E2E Myeloid Pipeline Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit and fix the `run_e2e_myeloid.py` pipeline — resolve systemic bugs (gene/pathway naming, FDR floor, axis scaling), overhaul 2nd-degree pattern classification, replace step 6 with flow-based comparison, fix per-step visualization and reporting issues, and prototype diversity metrics.

**Architecture:** Changes span library code (`src/peach/`) and the e2e script (`scripts/run_e2e_myeloid.py`). Library changes land first (feature_type propagation, axis scaling, pattern classification, volcano labels). Script changes follow, modifying steps 3-17 for correct behavior. A headless run on real data validates all changes end-to-end.

**Tech Stack:** Python, PEACH (local editable install in conda `archetype` env), scikit-bio for diversity metrics, plotly/matplotlib for viz. Data: `/Users/honkala/Desktop/FRTNBC/data/tnbc_myeloid_preprocessed.h5ad`.

**Key context files to read before any task:**
- `src/peach/_core/tools_schema.py` — function parameter reference
- `src/peach/_core/types_index.py` — return type reference
- `scripts/run_e2e_myeloid.py` — the 2540-line pipeline being modified

**Conda env:** Always run Python via `conda run -n archetype python`

---

## File Map

### Library files (modify)
| File | Responsibility | Changes |
|------|---------------|---------|
| `src/peach/pl/regression.py` | Regression plotting (dotplot, radar, etc.) | Add `feature_type` param to `archetype_regression_dotplot()`, remove [An] prefix, pass `order_by_similarity=True` |
| `src/peach/pl/comparison.py` | Comparison plotting (volcano, MMD heatmap) | Increase volcano label font, add jitter offset |
| `src/peach/_core/viz/results_viz.py` | 3D archetype space visualization | Fix axis scaling to include archetype positions |
| `src/peach/_core/utils/pattern_classification.py` | Feature pattern classifier | Overhaul 2nd-degree classification: transition-rising/falling, tradeoff, cooperative |
| `src/peach/tl/feature_patterns.py` | Pattern classification API | Pass interaction data through to classifier, return enriched results |

### Script file (modify)
| File | Changes |
|------|---------|
| `scripts/run_e2e_myeloid.py` | Steps 3-17 fixes, new diversity step, flow-based step 6, pathway display fixes |

---

## Task 1: Add `feature_type` to `archetype_regression_dotplot()`

**Files:**
- Modify: `src/peach/pl/regression.py:471-514`

Currently `archetype_regression_dotplot()` hardcodes `feature_type="genes"` via `_get_regression_data(adata)`. All other regression plotting functions accept `feature_type` — this one was missed.

- [ ] **Step 1: Add `feature_type` parameter to the function signature**

In `src/peach/pl/regression.py`, change the function signature and data retrieval:

```python
def archetype_regression_dotplot(
    adata: AnnData,
    *,
    top_n: int = 10,
    exclusive_only: bool = False,
    degree: int = 1,
    feature_type: str = "genes",
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
```

And change line 508 from:
```python
reg = _get_regression_data(adata)
```
to:
```python
reg = _get_regression_data(adata, feature_type=feature_type)
```

- [ ] **Step 2: Remove the [An] archetype prefix from gene labels**

The uncommitted change in `pl/regression.py` adds `[A1]`, `[A2]` prefixes to gene names on the y-axis. Find and remove this prefix logic. The sorting-by-dominant-archetype grouping can stay, but the label should be the bare gene/pathway name.

Search for `[A` or `archetype prefix` or `f"[A{` in the file and remove the prefix string formatting while keeping the sort order.

- [ ] **Step 3: Verify with import check**

Run: `conda run -n archetype python -c "from peach.pl.regression import archetype_regression_dotplot; import inspect; sig = inspect.signature(archetype_regression_dotplot); print(sig)"`

Expected: signature includes `feature_type: str = 'genes'`

- [ ] **Step 4: Commit**

```bash
git add src/peach/pl/regression.py
git commit -m "Add feature_type param to archetype_regression_dotplot, remove [An] prefix"
```

---

## Task 2: Fix archetype space axis scaling

**Files:**
- Modify: `src/peach/_core/viz/results_viz.py:1297-1316`

The `auto_scale` branch computes axis ranges from cell PCA coordinates only, excluding archetype vertex positions which are extremal by construction.

- [ ] **Step 1: Read the current axis range code**

Read `src/peach/_core/viz/results_viz.py` lines 1290-1320 and find where `archetype_coords` is defined (earlier in the function, around line 1295). Confirm `archetype_coords` is a numpy array of shape `[K, 3]`.

- [ ] **Step 2: Include archetype positions in axis range calculation**

Replace the `auto_scale` block (lines 1302-1311) with:

```python
elif auto_scale:
    # Use 1st and 99th percentiles with margin, including archetype positions
    def get_axis_range(coords, axis_idx, margin_factor=0.75, extra_points=None):
        percentiles = np.percentile(coords[:, axis_idx], [1, 99])
        lo, hi = percentiles[0], percentiles[1]
        # Expand to include archetype vertices if they fall outside
        if extra_points is not None and len(extra_points) > 0:
            lo = min(lo, extra_points[:, axis_idx].min())
            hi = max(hi, extra_points[:, axis_idx].max())
        margin = (hi - lo) * margin_factor
        return [lo - margin, hi + margin]

    x_range = get_axis_range(pca_coords, 0, extra_points=archetype_coords)
    y_range = get_axis_range(pca_coords, 1, extra_points=archetype_coords)
    z_range = get_axis_range(pca_coords, 2, extra_points=archetype_coords)
```

- [ ] **Step 3: Also fix the `else` (full data range) branch**

Replace lines 1313-1316 similarly:

```python
else:
    combined = np.vstack([pca_coords, archetype_coords])
    x_range = [combined[:, 0].min(), combined[:, 0].max()]
    y_range = [combined[:, 1].min(), combined[:, 1].max()]
    z_range = [combined[:, 2].min(), combined[:, 2].max()]
```

- [ ] **Step 4: Verify import**

Run: `conda run -n archetype python -c "from peach._core.viz.results_viz import plot_archetypal_space; print('OK')"`

- [ ] **Step 5: Commit**

```bash
git add src/peach/_core/viz/results_viz.py
git commit -m "Fix archetype space axis scaling to include vertex positions"
```

---

## Task 3: Improve volcano plot labels

**Files:**
- Modify: `src/peach/pl/comparison.py:17-69`

Current volcano labels use `font_size=6` (called with 6 in the volcano grid), no collision avoidance. Need 2x font and basic jitter.

- [ ] **Step 1: Read `_add_top_labels()` and `contrast_volcano_grid()`**

Read `src/peach/pl/comparison.py` lines 17-69 (the label function) and find where it's called in `contrast_volcano_grid()` (around line 270). Note the current `font_size=6, textangle=-45` call.

- [ ] **Step 2: Increase font and add vertical jitter**

In `_add_top_labels()`, change the annotation loop (around line 57-69):

```python
for idx_pos, i in enumerate(label_indices):
    # Alternate y-shift to reduce label overlap
    y_jitter = 10 + (idx_pos % 3) * 8  # 10, 18, 26 pixel cycling
    fig.add_annotation(
        x=delta[i],
        y=neg_log_p[i],
        text=names[i],
        showarrow=True,
        arrowhead=0,
        arrowwidth=0.5,
        arrowcolor="#999",
        ax=0,
        ay=-y_jitter,
        xref=xref,
        yref=yref,
        font=dict(size=font_size, color="#333"),
        xanchor="center",
        textangle=textangle,
    )
```

- [ ] **Step 3: Update the call site in `contrast_volcano_grid()` to use larger font**

Find the `_add_top_labels(...)` call (around line 270-274) and change `font_size=6` to `font_size=10`:

```python
_add_top_labels(fig, delta, pvals, neg_log_p, names,
                n_labels=n_labels, fdr_threshold=fdr_threshold,
                xref=f"x{idx+1}" if idx > 0 else "x",
                yref=f"y{idx+1}" if idx > 0 else "y",
                font_size=10, textangle=-30)
```

- [ ] **Step 4: Commit**

```bash
git add src/peach/pl/comparison.py
git commit -m "Improve volcano plot labels: larger font, jitter to reduce overlap"
```

---

## Task 4: Overhaul 2nd-degree pattern classification

**Files:**
- Modify: `src/peach/_core/utils/pattern_classification.py:6-77`
- Modify: `src/peach/tl/feature_patterns.py:11-99`

The current classification only categorizes into flat/exclusive/interaction/structured. The 2nd-degree interaction terms need proper classification:
- **γ_{jk} > 0**: gene rises along the j→k edge (above linear interpolation of vertex values)
- **γ_{jk} < 0**: gene falls along the j→k edge
- **Cooperative**: β_j high AND β_k high, low at other archetypes (shared program)
- **Tradeoff**: β_j high, β_k low or vice versa (distinguishes two archetypes)
- **Transition-enriched**: significant γ_{jk} but modest vertex values (peaks in transition zone)

- [ ] **Step 1: Read current implementation**

Read `src/peach/_core/utils/pattern_classification.py` (full file, ~124 lines) and `src/peach/tl/feature_patterns.py` lines 11-99.

- [ ] **Step 2: Add interaction detail classification to `classify_single_feature()`**

Add parameters for interaction coefficients and pairs. After the existing classification, add interaction sub-classification. Replace the full function in `pattern_classification.py`:

```python
def classify_single_feature(
    vertex_betas,
    r2,
    f_pvalue_fdr,
    interaction_f_pvalue_fdr=None,
    fdr_threshold=0.05,
    exclusive_ratio=2.0,
    vertex_ses=None,
    interaction_betas=None,
    interaction_pairs=None,
    interaction_pvalues_fdr=None,
):
    """Classify a single feature into a biological pattern type.

    Parameters
    ----------
    vertex_betas : array-like, shape [K]
        Regression coefficients per archetype.
    r2 : float
        R-squared of the regression.
    f_pvalue_fdr : float
        FDR-corrected F-test p-value.
    interaction_f_pvalue_fdr : float or None
        FDR-corrected interaction F-test p-value (overall model comparison).
    fdr_threshold : float
        Significance threshold.
    exclusive_ratio : float
        Min ratio of max |beta| to second max for exclusive classification.
    vertex_ses : array-like or None
        Standard errors per archetype.
    interaction_betas : array-like or None, shape [n_pairs]
        Per-pair interaction coefficients from degree-2 regression.
    interaction_pairs : list of tuples or None
        The (j, k) archetype index pairs corresponding to interaction_betas.
    interaction_pvalues_fdr : array-like or None, shape [n_pairs]
        FDR-corrected p-values for each interaction term.
    """
    # NaN guard
    if np.isnan(r2):
        return {"pattern": "flat", "r2": float(r2), "details": {"reason": "nan_r2"}}
    if np.isnan(f_pvalue_fdr):
        return {"pattern": "flat", "r2": float(r2), "details": {"reason": "nan_pvalue"}}

    # Rule 1: nonsignificant F-test -> flat
    if f_pvalue_fdr > fdr_threshold:
        return {"pattern": "flat", "r2": float(r2), "details": {"reason": "nonsignificant"}}

    # Rule 2: archetype-exclusive
    abs_betas = np.abs(vertex_betas)
    sorted_abs = np.sort(abs_betas)[::-1]
    if len(sorted_abs) >= 2 and sorted_abs[0] > 0 and sorted_abs[0] / max(sorted_abs[1], 1e-10) >= exclusive_ratio:
        dominant = int(np.argmax(abs_betas))
        se_passes = True
        if vertex_ses is not None:
            dominant_se = vertex_ses[dominant]
            if dominant_se > 0 and abs_betas[dominant] < 2 * dominant_se:
                se_passes = False
        if se_passes:
            return {
                "pattern": "archetype-exclusive",
                "r2": float(r2),
                "details": {"dominant_archetype": dominant},
            }

    # Rule 3: interaction — now with sub-classification
    if interaction_f_pvalue_fdr is not None and interaction_f_pvalue_fdr < fdr_threshold:
        interaction_detail = _classify_interaction_detail(
            vertex_betas, interaction_betas, interaction_pairs,
            interaction_pvalues_fdr, fdr_threshold
        )
        return {
            "pattern": "interaction",
            "r2": float(r2),
            "details": {
                "dominant_archetype": int(np.argmax(abs_betas)),
                "interaction_detail": interaction_detail,
            },
        }

    # Rule 4: structured fallback
    return {
        "pattern": "structured",
        "r2": float(r2),
        "details": {"dominant_archetype": int(np.argmax(abs_betas))},
    }


def _classify_interaction_detail(vertex_betas, interaction_betas, interaction_pairs,
                                  interaction_pvalues_fdr, fdr_threshold):
    """Sub-classify significant interaction terms.

    For each significant interaction pair (j, k):
    - gamma > 0: gene rises along j-k edge (transition-rising)
    - gamma < 0: gene falls along j-k edge (transition-falling)
    - Cooperative: beta_j and beta_k both high relative to other vertices,
      gene stays high between them
    - Tradeoff: beta_j high, beta_k low (or vice versa)

    Returns list of per-pair classifications.
    """
    if interaction_betas is None or interaction_pairs is None:
        return []

    K = len(vertex_betas)
    detail = []
    betas = np.asarray(vertex_betas)
    int_betas = np.asarray(interaction_betas)
    median_abs_beta = np.median(np.abs(betas))

    for pair_idx, (j, k) in enumerate(interaction_pairs):
        # Skip non-significant pairs
        if interaction_pvalues_fdr is not None:
            if pair_idx < len(interaction_pvalues_fdr) and interaction_pvalues_fdr[pair_idx] >= fdr_threshold:
                continue

        gamma = float(int_betas[pair_idx]) if pair_idx < len(int_betas) else 0.0
        beta_j = float(betas[j])
        beta_k = float(betas[k])

        # Classify the vertex relationship
        # "High" = above median |beta| for this gene
        j_high = abs(beta_j) > median_abs_beta
        k_high = abs(beta_k) > median_abs_beta
        same_sign = np.sign(beta_j) == np.sign(beta_k) and beta_j != 0 and beta_k != 0

        if j_high and k_high and same_sign:
            pair_type = "cooperative"
        elif (j_high and not k_high) or (not j_high and k_high):
            pair_type = "tradeoff"
        elif not j_high and not k_high and abs(gamma) > median_abs_beta:
            pair_type = "transition-enriched"
        else:
            pair_type = "gradient"

        # Transition direction from gamma sign
        transition = "rising" if gamma > 0 else "falling"

        detail.append({
            "pair": (int(j), int(k)),
            "pair_type": pair_type,
            "transition": transition,
            "gamma": gamma,
            "beta_j": beta_j,
            "beta_k": beta_k,
        })

    return detail
```

- [ ] **Step 3: Update `classify_all_features()` to pass interaction data**

In the same file, update `classify_all_features()` to accept and forward interaction data:

```python
def classify_all_features(
    vertex_coefficients,
    r_squared,
    f_pvalue_fdr,
    interaction_f_pvalue_fdr=None,
    fdr_threshold=0.05,
    exclusive_ratio=2.0,
    vertex_ses=None,
    interaction_coefficients=None,
    interaction_pairs=None,
    interaction_pvalues_fdr=None,
):
    """Classify all features into biological pattern types."""
    n_features = len(r_squared)
    results = []
    for i in range(n_features):
        int_fdr = (
            interaction_f_pvalue_fdr[i]
            if interaction_f_pvalue_fdr is not None
            else None
        )
        feat_ses = vertex_ses[i] if vertex_ses is not None else None
        feat_int_betas = (
            interaction_coefficients[i]
            if interaction_coefficients is not None
            else None
        )
        feat_int_pvals = (
            interaction_pvalues_fdr[i]
            if interaction_pvalues_fdr is not None
            else None
        )
        results.append(
            classify_single_feature(
                vertex_coefficients[i],
                r_squared[i],
                f_pvalue_fdr[i],
                interaction_f_pvalue_fdr=int_fdr,
                fdr_threshold=fdr_threshold,
                exclusive_ratio=exclusive_ratio,
                vertex_ses=feat_ses,
                interaction_betas=feat_int_betas,
                interaction_pairs=interaction_pairs,
                interaction_pvalues_fdr=feat_int_pvals,
            )
        )
    return results
```

- [ ] **Step 4: Update `classify_feature_patterns()` in `feature_patterns.py`**

Read `src/peach/tl/feature_patterns.py` and update the call to `classify_all_features()` (around line 64-72) to pass interaction data from the regression result:

```python
interaction_coefs = regression_result.interaction_coefficients
interaction_pairs_list = regression_result.interaction_pairs
int_pvals_fdr = regression_result.interaction_pvalues_fdr

classifications = classify_all_features(
    vertex_coefficients=regression_result.vertex_coefficients,
    r_squared=regression_result.r_squared_degree1,
    f_pvalue_fdr=regression_result.f_pvalue_fdr,
    interaction_f_pvalue_fdr=interaction_f_pvalue_fdr,
    fdr_threshold=fdr_threshold,
    exclusive_ratio=exclusive_ratio,
    vertex_ses=vertex_ses,
    interaction_coefficients=np.asarray(interaction_coefs) if interaction_coefs is not None else None,
    interaction_pairs=interaction_pairs_list if interaction_pairs_list else None,
    interaction_pvalues_fdr=np.asarray(int_pvals_fdr) if int_pvals_fdr is not None else None,
)
```

Note: `regression_result` may be a dict (from `adata.uns`) or a `SimplexRegressionResult`. Use `.get()` for dict access. Check how the existing code handles this (the function already has a validation step at the top).

- [ ] **Step 5: Verify**

Run: `conda run -n archetype python -c "from peach._core.utils.pattern_classification import classify_single_feature; import numpy as np; r = classify_single_feature(np.array([2.0, 0.1, 0.1, 0.3]), 0.8, 0.001, interaction_f_pvalue_fdr=0.01, interaction_betas=np.array([0.5, -0.2, 0.1, 0.3, -0.1, 0.05]), interaction_pairs=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)], interaction_pvalues_fdr=np.array([0.01, 0.5, 0.5, 0.5, 0.5, 0.5])); print(r)"`

Expected: Returns classification with `interaction_detail` containing pair-level types.

- [ ] **Step 6: Commit**

```bash
git add src/peach/_core/utils/pattern_classification.py src/peach/tl/feature_patterns.py
git commit -m "Overhaul 2nd-degree pattern classification: transition/cooperative/tradeoff/gradient"
```

---

## Task 5: Fix step 3 — pathway display, interaction reclassification, mutual exclusivity

**Files:**
- Modify: `scripts/run_e2e_myeloid.py:516-753`

Three changes in step 3:
1. All pathway plotting calls must pass `feature_type="pathways"`
2. Replace cooperative/tradeoff classification with new scheme from Task 4
3. Add mutual exclusivity tradeoff accounting table

- [ ] **Step 1: Read step 3 fully**

Read `scripts/run_e2e_myeloid.py` lines 516-753.

- [ ] **Step 2: Fix pathway dotplot and radar calls**

After the pathway regression (around line 685), add dedicated pathway dotplot and radar calls using `feature_type="pathways"`:

```python
# Pathway dotplot (exclusive pathways)
try:
    fig_pw_dot = pc.pl.archetype_regression_dotplot(
        adata, top_n=10, exclusive_only=True,
        feature_type="pathways", show=False)
    html += safe_plotly_html(report, fig_pw_dot,
                             "Pathway regression dotplot (exclusive pathways)")
except Exception as e:
    html += error_html(f"Pathway dotplot failed: {e}")

# Pathway radar
try:
    fig_pw_radar = pc.pl.archetype_radar(
        adata, top_n=8, feature_type="pathways",
        order_by_similarity=True, show=False)
    html += safe_plotly_html(report, fig_pw_radar,
                             "Pathway radar (top features, similarity-ordered)")
except Exception as e:
    html += error_html(f"Pathway radar failed: {e}")
```

- [ ] **Step 3: Also fix the gene radar call to use `order_by_similarity=True`**

Change the existing gene radar call (around line 580):

```python
fig_radar = pc.pl.archetype_radar(adata, top_n=8, order_by_similarity=True, show=False)
```

- [ ] **Step 4: Replace cooperative/tradeoff classification block with new scheme**

Replace lines 618-679 (the cooperative/tradeoff block) with a reclassification using the new pattern detail from `classify_feature_patterns()`. The new code should:

```python
# Interaction term reclassification (using new pattern detail)
try:
    int_coefs = gene_reg.get("interaction_coefficients")
    int_pairs = gene_reg.get("interaction_pairs", [])
    int_fdr = gene_reg.get("interaction_pvalues_fdr")
    vertex_coefs = np.asarray(gene_reg["vertex_coefficients"])
    feat_names = list(gene_reg["feature_names"])

    if int_coefs is not None and len(int_pairs) > 0 and int_fdr is not None:
        int_coefs = np.asarray(int_coefs)
        int_fdr = np.asarray(int_fdr)

        interaction_rows = []
        for feat_idx in range(len(feat_names)):
            for pair_idx, (j, k) in enumerate(int_pairs):
                if int_fdr[feat_idx, pair_idx] < 0.05:
                    gamma = int_coefs[feat_idx, pair_idx]
                    beta_j = vertex_coefs[feat_idx, j]
                    beta_k = vertex_coefs[feat_idx, k]

                    # Classify vertex relationship
                    median_abs = np.median(np.abs(vertex_coefs[feat_idx]))
                    j_high = abs(beta_j) > median_abs
                    k_high = abs(beta_k) > median_abs
                    same_sign = (np.sign(beta_j) == np.sign(beta_k)
                                 and beta_j != 0 and beta_k != 0)

                    if j_high and k_high and same_sign:
                        pair_type = "cooperative"
                    elif (j_high and not k_high) or (not j_high and k_high):
                        pair_type = "tradeoff"
                    elif not j_high and not k_high and abs(gamma) > median_abs:
                        pair_type = "transition-enriched"
                    else:
                        pair_type = "gradient"

                    transition = "rising" if gamma > 0 else "falling"

                    interaction_rows.append({
                        "Feature": feat_names[feat_idx],
                        "Pair": f"A{j+1}-A{k+1}",
                        "Type": pair_type,
                        "Transition": transition,
                        "beta_j": f"{beta_j:.3f}",
                        "beta_k": f"{beta_k:.3f}",
                        "gamma": f"{gamma:.3f}",
                        "FDR q": f"{int_fdr[feat_idx, pair_idx]:.2e}",
                    })

        if interaction_rows:
            int_df = pd.DataFrame(interaction_rows)
            type_counts = int_df["Type"].value_counts()
            cards = [metric_card(len(interaction_rows), "Significant interactions")]
            for t in ["tradeoff", "cooperative", "transition-enriched", "gradient"]:
                cards.append(metric_card(int(type_counts.get(t, 0)), t.capitalize()))
            html += metric_grid(cards)

            html += report.text(
                "Interaction classification: <b>Cooperative</b> = high at both archetypes "
                "(shared program). <b>Tradeoff</b> = high at one, low at other (distinguishes "
                "archetypes). <b>Transition-enriched</b> = peaks in blending zone. "
                "<b>Gradient</b> = moderate signal. Transition direction: rising (γ>0) = gene "
                "increases along edge; falling (γ<0) = gene decreases.")

            for itype in ["tradeoff", "cooperative", "transition-enriched", "gradient"]:
                sub = int_df[int_df["Type"] == itype].head(20)
                if len(sub) > 0:
                    html += report.df_to_html(sub, caption=f"Top {itype} interactions")
        else:
            html += report.text("No significant interaction terms at FDR < 0.05.")
    else:
        html += report.text("Interaction terms not available in regression results.")
except Exception as e:
    html += error_html(f"Interaction classification failed: {e}")
```

- [ ] **Step 5: Add mutual exclusivity tradeoff table**

After the interaction classification block, add a pairwise mutual exclusivity accounting:

```python
# Mutual exclusivity: pairwise tradeoff accounting
try:
    if int_coefs is not None and len(int_pairs) > 0 and int_fdr is not None:
        me_rows = []
        for pair_idx, (j, k) in enumerate(int_pairs):
            # Find genes with tradeoff pattern for this pair
            for feat_idx in range(len(feat_names)):
                if int_fdr[feat_idx, pair_idx] >= 0.05:
                    continue
                beta_j = vertex_coefs[feat_idx, j]
                beta_k = vertex_coefs[feat_idx, k]
                median_abs = np.median(np.abs(vertex_coefs[feat_idx]))
                # Tradeoff: one high, one low
                if (abs(beta_j) > median_abs) != (abs(beta_k) > median_abs):
                    if abs(beta_j) > abs(beta_k):
                        direction = f"A{j+1}→A{k+1}"
                    else:
                        direction = f"A{k+1}→A{j+1}"
                    me_rows.append({
                        "archetype_high": f"A{j+1}" if abs(beta_j) > abs(beta_k) else f"A{k+1}",
                        "archetype_low": f"A{k+1}" if abs(beta_j) > abs(beta_k) else f"A{j+1}",
                        "gene": feat_names[feat_idx],
                        "direction": direction,
                        "beta_high": f"{max(abs(beta_j), abs(beta_k)):.3f}",
                        "beta_low": f"{min(abs(beta_j), abs(beta_k)):.3f}",
                        "gamma": f"{int_coefs[feat_idx, pair_idx]:.3f}",
                        "fdr": f"{int_fdr[feat_idx, pair_idx]:.2e}",
                    })

        if me_rows:
            me_df = pd.DataFrame(me_rows)
            # Summary: how many tradeoff genes per archetype pair
            pair_summary = me_df.groupby(["archetype_high", "archetype_low"]).size().reset_index(name="n_genes")
            html += report.df_to_html(pair_summary,
                                      caption="Mutual exclusivity: tradeoff gene counts per archetype pair")
            html += report.df_to_html(me_df.head(40),
                                      caption="Mutual exclusivity: tradeoff genes (top 40)")
except Exception as e:
    html += error_html(f"Mutual exclusivity table failed: {e}")
```

- [ ] **Step 6: Commit**

```bash
git add scripts/run_e2e_myeloid.py
git commit -m "Step 3: fix pathway display, reclassify interactions, add mutual exclusivity table"
```

---

## Task 6: Fix step 5 — volcano, confusion matrix, pathway contrasts

**Files:**
- Modify: `scripts/run_e2e_myeloid.py` (step 5 function, around lines 827-930)

- [ ] **Step 1: Read step 5**

Read `scripts/run_e2e_myeloid.py` step 5 (`step5_wald_contrasts`).

- [ ] **Step 2: Add `groupby Direction` to top 10 contrasts table**

Find where top 10 contrasts are displayed. Add `.sort_values(["Direction", ...])` or group the table:

```python
# Sort by direction then effect size for readability
top_df = top_df.sort_values(["Direction", "abs_delta_beta"], ascending=[True, False])
```

- [ ] **Step 3: Replace multi-pair significant genes text with confusion matrix**

Find the "Genes significant in multiple archetype pair contrasts" section. Replace the text-heavy direction-by-pair format with a gene × archetype-pair matrix:

```python
# Multi-pair genes as confusion matrix
if multi_pair_genes:  # however this is currently collected
    # Build a matrix: rows=genes, columns=archetype pairs, values=direction (+/-/ns)
    all_pairs_str = [f"A{j+1}-A{k+1}" for j, k in contrast_result["pairs"]]
    matrix_rows = []
    for gene_name, gene_pairs in multi_pair_genes.items():
        row = {"Gene": gene_name}
        for pair_str in all_pairs_str:
            if pair_str in gene_pairs:
                row[pair_str] = gene_pairs[pair_str]  # "+" or "-"
            else:
                row[pair_str] = ""
        matrix_rows.append(row)
    if matrix_rows:
        conf_df = pd.DataFrame(matrix_rows).set_index("Gene")
        html += report.df_to_html(conf_df, caption="Multi-pair contrast direction matrix (+ = up in first archetype, - = down)")
```

Adapt this to the actual data structure in the current script — the key idea is a gene × pair matrix with +/- signs.

- [ ] **Step 4: Fix pathway contrasts to show pathways not genes**

Find the "Top 10 pathway contrasts" section. Ensure it passes `feature_type="pathways"` when retrieving regression data, or explicitly uses the pathway regression result and pathway feature names.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_e2e_myeloid.py
git commit -m "Step 5: groupby Direction, confusion matrix for multi-pair genes, fix pathway contrasts"
```

---

## Task 7: Rewrite step 6 as flow-based within-fit comparison

**Files:**
- Modify: `scripts/run_e2e_myeloid.py` (step 6 function, around lines 933-987)

Replace Gaussian kernel `archetype_mmd` with pairwise `flow_within` between archetype obs labels. Keep Spearman feature similarity.

- [ ] **Step 1: Read current step 6 and the `flow_within` API**

Read `scripts/run_e2e_myeloid.py` lines 933-987 and `src/peach/tl/flow.py` lines 14-142.

Confirm archetype labels in `adata.obs['archetypes']` use format `'archetype_0'`, `'archetype_1'`, etc.

- [ ] **Step 2: Rewrite `step6_within_fit_comparisons()`**

Replace the function body. The new version runs `flow_within` for each archetype pair using obs labels:

```python
def step6_within_fit_comparisons(adata, report):
    """Step 6: Flow-based within-fit pairwise comparison + feature similarity."""
    import peach as pc

    html = ""

    # Get archetype labels
    if "archetypes" not in adata.obs.columns:
        html += error_html("No 'archetypes' column — skipping within-fit comparisons.")
        report.add_section("Within-Fit Comparisons", html, step_num=6)
        return None

    arch_labels = sorted([a for a in adata.obs["archetypes"].unique()
                          if a != "no_archetype" and not pd.isna(a)])
    K = len(arch_labels)
    html += report.text(f"Pairwise flow comparison for {K} archetypes: {', '.join(arch_labels)}")

    # Pairwise flow_within
    flow_pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            flow_pairs.append((arch_labels[i], arch_labels[j]))

    flow_rows = []
    for src_label, tgt_label in flow_pairs:
        n_src = int((adata.obs["archetypes"] == src_label).sum())
        n_tgt = int((adata.obs["archetypes"] == tgt_label).sum())

        if n_src < 50 or n_tgt < 50:
            flow_rows.append({
                "Source": src_label, "Target": tgt_label,
                "N_source": n_src, "N_target": n_tgt,
                "MMD_before": float("nan"), "MMD_after": float("nan"),
                "MMD_reduction": float("nan"),
                "Status": f"Skipped (<50 cells)",
            })
            continue

        try:
            fr = pc.tl.flow_within(
                adata,
                source={"archetypes": src_label},
                target={"archetypes": tgt_label},
                n_epochs=200,
                hidden_dims=(128, 128),
                batch_size=min(128, min(n_src, n_tgt) // 2),
                return_model=False,
                name=f"wf_{src_label}_to_{tgt_label}",
            )
            mmd_b = fr["mmd_before"]
            mmd_a = fr["mmd_after"]
            reduction = 1 - mmd_a / max(mmd_b, 1e-10)
            flow_rows.append({
                "Source": src_label, "Target": tgt_label,
                "N_source": n_src, "N_target": n_tgt,
                "MMD_before": mmd_b, "MMD_after": mmd_a,
                "MMD_reduction": reduction,
                "Status": "OK",
            })
        except Exception as e:
            log.warning(f"Flow {src_label}->{tgt_label} failed: {e}")
            flow_rows.append({
                "Source": src_label, "Target": tgt_label,
                "N_source": n_src, "N_target": n_tgt,
                "MMD_before": float("nan"), "MMD_after": float("nan"),
                "MMD_reduction": float("nan"),
                "Status": f"Failed: {e}",
            })

    flow_df = pd.DataFrame(flow_rows)
    html += report.df_to_html(flow_df, caption="Pairwise flow-based archetype comparison")

    # Build K×K similarity matrix (MMD reduction as similarity)
    sim_matrix = np.full((K, K), np.nan)
    for _, row in flow_df.iterrows():
        if row["Status"] == "OK":
            i = arch_labels.index(row["Source"])
            j = arch_labels.index(row["Target"])
            sim_matrix[i, j] = row["MMD_reduction"]
            sim_matrix[j, i] = row["MMD_reduction"]
    np.fill_diagonal(sim_matrix, 1.0)

    # Heatmap of flow similarity
    try:
        import plotly.graph_objects as go
        arch_short = [f"A{i+1}" for i in range(K)]
        fig = go.Figure(data=go.Heatmap(
            z=sim_matrix, x=arch_short, y=arch_short,
            colorscale="Blues", text=np.round(sim_matrix, 3).astype(str),
            texttemplate="%{text}", textfont_size=10,
        ))
        fig.update_layout(title="Flow-based archetype similarity (MMD reduction)",
                          xaxis_title="Target", yaxis_title="Source",
                          width=500, height=450)
        html += safe_plotly_html(report, fig, "Flow-based similarity: higher = more similar phenotype after transport")
    except Exception as e:
        html += error_html(f"Flow similarity heatmap failed: {e}")

    # Feature similarity (Spearman on regression coefficients) — keep existing
    log.info("Computing within-fit feature similarity...")
    try:
        sim_result = pc.tl.archetype_feature_similarity(adata)
        n_sig_feat = sim_result.get("n_significant_features", "?")
        html += report.text(f"Spearman \u03c1 computed on {n_sig_feat} FDR-significant (q<0.05) "
                            "vertex \u03b2 coefficients from simplex regression.")

        fig_sim = pc.pl.feature_similarity_heatmap(adata, show=False)
        html += safe_plotly_html(report, fig_sim, "Feature similarity (Spearman \u03c1) heatmap")
    except Exception as e:
        html += error_html(f"Feature similarity failed: {e}")

    report.add_section("Within-Fit Comparisons (Flow + Feature Similarity)", html, step_num=6)
    return flow_df
```

- [ ] **Step 3: Update main() to capture step 6 return value**

In `main()`, step 6 now returns a DataFrame. Update:

```python
flow_comparison_df = None
# -- Step 6 ---
t0 = time.time()
try:
    flow_comparison_df = step6_within_fit_comparisons(adata, report)
    log.info(f"Step 6 done in {time.time() - t0:.1f}s")
except Exception as e:
    ...
```

- [ ] **Step 4: Commit**

```bash
git add scripts/run_e2e_myeloid.py
git commit -m "Step 6: replace Gaussian kernel MMD with pairwise flow-based comparison"
```

---

## Task 8: Fix step 7 driver regression + step 9 component characterization

**Files:**
- Modify: `scripts/run_e2e_myeloid.py` (steps 7 and 9)

- [ ] **Step 1: Read steps 7 and 9**

Read step 7 (`step7_driver_regression`) and step 9 (`step9_component_characterization`).

- [ ] **Step 2: Fix pathway driver regression singular matrix**

The pathway score matrix likely has collinear features. Add a rank check and variance-based filtering before calling driver regression:

```python
# Filter to top pathways by variance to avoid singularity
pw_scores = adata.obsm.get("pathway_scores")
if pw_scores is not None:
    pw_var = np.var(pw_scores, axis=0)
    n_keep = min(pw_scores.shape[1], max(20, adata.obsm["cell_archetype_weights"].shape[1] * 3))
    top_pw_idx = np.argsort(pw_var)[-n_keep:]
    # Store filtered pathway names and pass to driver regression
    pw_names_all = adata.uns.get("pathway_scores_pathways", [f"pw_{i}" for i in range(pw_scores.shape[1])])
    pw_names_filtered = [pw_names_all[i] for i in top_pw_idx]
    pw_filtered = pw_scores[:, top_pw_idx]
    # Run with filtered matrix
    pw_driver = pc.tl.archetype_driver_regression(
        adata, feature_matrix=pw_filtered,
        feature_names=pw_names_filtered, max_degree=1)
```

Adapt to the actual API of `archetype_driver_regression` — read `tools_schema.py` entry for this function first.

- [ ] **Step 3: Add shared genes table with coefficient breakdown**

In step 7 where shared genes between simplex and driver regression are reported, add a DataFrame showing both sets of coefficients plus classification (flat/interaction):

```python
# Shared genes: coefficient comparison table
if shared_genes:
    shared_rows = []
    for gene in list(shared_genes)[:30]:
        row = {"Gene": gene}
        # Simplex regression coefficients
        if gene in gene_reg_names:
            idx = gene_reg_names.index(gene)
            row["Simplex_R2"] = f"{simplex_r2[idx]:.3f}"
            row["Simplex_max_beta"] = f"{np.max(np.abs(simplex_coefs[idx])):.3f}"
        # Driver regression coefficients
        if gene in driver_names:
            idx = driver_names.index(gene)
            row["Driver_max_beta"] = f"{np.max(np.abs(driver_coefs[idx])):.3f}"
        shared_rows.append(row)
    html += report.df_to_html(pd.DataFrame(shared_rows), caption="Shared genes: coefficient comparison")
```

- [ ] **Step 4: Step 9 — report raw ARI, majority vote, MMD**

Find the component-archetype summary section in step 9. Add explicit raw values table:

```python
# Raw similarity metrics table
html += report.df_to_html(pd.DataFrame([{
    "ARI": f"{ari_value:.4f}",
    "Majority vote accuracy": f"{majority_acc:.4f}",
    "Mean pairwise MMD": f"{mean_mmd:.4f}",
}]), caption="Component-archetype similarity: raw metrics")
```

- [ ] **Step 5: Step 9 — add pathway characterization**

After gene component_regression, add pathway version:

```python
# Pathway characterization per component
if "pathway_scores" in adata.obsm:
    try:
        pw_comp_reg = pc.tl.component_regression(adata, feature_type="pathways")
        fig_pw = pc.pl.component_heatmap(adata, feature_type="pathways", show=False)
        html += safe_plotly_html(report, fig_pw, "Component pathway characterization")
    except Exception as e:
        html += error_html(f"Pathway component characterization failed: {e}")
```

Check whether `component_regression` and `component_heatmap` accept `feature_type`. If not, pass `feature_matrix="pathway_scores"` as appropriate.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_e2e_myeloid.py
git commit -m "Fix step 7 pathway singularity, step 9 raw metrics + pathway characterization"
```

---

## Task 9: Fix steps 10-11 per-lineage improvements

**Files:**
- Modify: `scripts/run_e2e_myeloid.py` (steps 10-11)

- [ ] **Step 1: Read steps 10 and 11**

Read `step10_per_dose_models()` and `step11_per_dose_regression()`.

- [ ] **Step 2: Step 10 — add hyperparameter search QC**

After the hyperparameter search for each dose, add training metrics reporting:

```python
# Report CV search results per dose
if cv is not None:
    ranked = cv.rank_by_metric("r2")
    cv_rows = []
    for r in ranked[:5]:
        hp = r["hyperparameters"]
        cv_rows.append({
            "K": hp["n_archetypes"],
            "Hidden": str(hp.get("hidden_dims", "?")),
            "Mean R2": f"{r['metric_value']:.4f}",
        })
    html += report.df_to_html(pd.DataFrame(cv_rows),
                              caption=f"{dose}: top 5 CV configurations")
```

- [ ] **Step 3: Step 10 — fix archetype space scaling in manual scatter**

In the per-dose scatter plot section (around line 1450), add archetype positions to axis limits:

```python
# Get archetype positions for axis padding
arch_pos = sub.uns.get("archetype_coordinates")
coords = sub.obsm.get("archetype_coordinates")
if coords is not None and coords.shape[1] >= 2:
    ax.scatter(coords[:, 0], coords[:, 1], s=2, alpha=0.3, c="#0072B2")
    # Mark archetype vertices
    if arch_pos is not None:
        arch_pos = np.asarray(arch_pos)
        if arch_pos.shape[1] >= 2:
            ax.scatter(arch_pos[:, 0], arch_pos[:, 1], s=80, c="red",
                       marker="^", zorder=5, edgecolors="black", linewidth=0.5)
    # Expand limits to include vertices
    all_pts = coords
    if arch_pos is not None and arch_pos.shape[1] >= 2:
        all_pts = np.vstack([coords, arch_pos[:, :2]])
    margin = 0.1 * (all_pts.max(axis=0) - all_pts.min(axis=0))
    ax.set_xlim(all_pts[:, 0].min() - margin[0], all_pts[:, 0].max() + margin[0])
    ax.set_ylim(all_pts[:, 1].min() - margin[1], all_pts[:, 1].max() + margin[1])
```

Apply the same pattern to steps 16 and 17 manual scatter plots.

- [ ] **Step 4: Step 11 — prioritize exclusive features in dotplots**

Change dotplot calls to use `exclusive_only=True`:

```python
fig_dot = pc.pl.archetype_regression_dotplot(
    sub, top_n=10, exclusive_only=True, show=False)
```

- [ ] **Step 5: Commit**

```bash
git add scripts/run_e2e_myeloid.py
git commit -m "Steps 10-11: add CV QC, fix scatter scaling, exclusive dotplots"
```

---

## Task 10: Fix steps 12-14 flow analysis

**Files:**
- Modify: `scripts/run_e2e_myeloid.py` (steps 12-14)

- [ ] **Step 1: Read steps 12-14**

Read `step12_between_dose_flow()`, `step13_sinkhorn_flow()`, and `step14_jacobian()`.

- [ ] **Step 2: Step 12 — add correspondence metric explanation**

After each soft assignment heatmap, add interpretation text:

```python
html += report.text(
    "Soft assignment correspondence: source cells are transported via the learned flow field, "
    "then matched to their k-nearest target neighbors in PCA space. The matrix entry [i,j] shows "
    "the fraction of source archetype i's transported mass that lands near target archetype j. "
    "Uniform rows indicate diffuse transitions; concentrated rows indicate canalization.")
```

- [ ] **Step 3: Step 13 — filter alignment table to significant genes**

Find the gene alignment table section. Add significance filtering:

```python
# Filter to only significant alignment scores
if "alignment_pvalues" in align_result:
    raw_p = np.asarray(align_result["alignment_pvalues"])
    sig_mask = raw_p < 0.05
    if sig_mask.sum() > 0:
        sig_scores = scores[sig_mask]
        sig_names = [names[i] for i in range(len(names)) if sig_mask[i]]
        # Use sig_scores and sig_names for the table
    else:
        html += report.text("No genes with raw p < 0.05 for alignment significance.")
```

Also add explanatory text for the null:

```python
html += report.text(
    "Null: permutation of gene-to-PCA-loading assignments. Tests whether a specific gene's "
    "alignment with the flow field is stronger than expected for a random gene-loading pairing.")
```

- [ ] **Step 4: Step 13 — clarify gene expression delta**

For the "gene expression delta along flow" section, add clarification:

```python
html += report.text(
    "Gene expression delta: PCA-reconstructed expression change. Computed as "
    "(PCA loadings) \u00d7 (\u0394 PCA coordinates) between transported and source positions. "
    "Units are log-normalized expression change. Filtered to FDR-significant flow-aligned genes.")
```

- [ ] **Step 5: Step 14 — filter to raw-p significant, trace FDR**

Filter the top-20 expanding/contracting tables to only genes with raw p < 0.05:

```python
# Filter expanding genes to significant only
if "expansion_pvalues" in jac:
    raw_p = np.asarray(jac["expansion_pvalues"])
    sig_mask = raw_p < 0.05
    n_raw_sig = int(sig_mask.sum())
    html += report.text(f"Genes with raw p < 0.05: {n_raw_sig} / {len(raw_p)}")

    fdr_p = jac.get("expansion_pvalues_fdr")
    if fdr_p is not None:
        fdr_p = np.asarray(fdr_p)
        n_fdr_sig = int((fdr_p < 0.05).sum())
        html += report.text(f"Genes with FDR q < 0.05: {n_fdr_sig} / {len(fdr_p)}")
        if n_fdr_sig == 0 and n_raw_sig > 0:
            html += report.text("Note: no genes survive FDR correction. Showing raw-p significant genes.")

    # Use raw-p mask for display
    if n_raw_sig > 0:
        sig_expansion = expansion_scores[sig_mask]
        sig_names = [gene_names[i] for i in range(len(gene_names)) if sig_mask[i]]
        # Sort and display
```

- [ ] **Step 6: Commit**

```bash
git add scripts/run_e2e_myeloid.py
git commit -m "Steps 12-14: add correspondence explanation, filter to significant, trace FDR"
```

---

## Task 11: Fix steps 15-17

**Files:**
- Modify: `scripts/run_e2e_myeloid.py` (steps 15-17)

- [ ] **Step 1: Read steps 15-17**

Read `step15_gene_deep_dive()`, `step16_per_response()` (if present), `step17_per_response_per_dose()` (if present), and any lineage comparison steps.

- [ ] **Step 2: Step 15 — debug violin plots**

The violin plot code IS present (lines 1990-2017) but may fail silently if `jac_results` is empty or `per_cell_expansion` is None. Add diagnostic logging:

```python
log.info(f"Violin plots: {len(jac_results)} Jacobian results available")
for pair_key, jac in jac_results.items():
    per_cell = jac.get("per_cell_expansion")
    gene_names = jac.get("per_cell_expansion_gene_names", [])
    log.info(f"  {pair_key}: per_cell={'present' if per_cell is not None else 'MISSING'}, "
             f"n_genes={len(gene_names)}")
```

If per_cell_expansion is missing, it means `flow_jacobian()` was not called with `per_cell_features=True`. Check the step 14 call and ensure this parameter is set.

- [ ] **Step 3: Steps 16-17 — exclusive dotplots + pathway + scaling**

Apply the same fixes as steps 10-11:

```python
# Exclusive dotplots
fig_dot = pc.pl.archetype_regression_dotplot(
    sub, top_n=10, exclusive_only=True, show=False)

# Pathway dotplots (if pathway_scores exist)
if "pathway_scores" in sub.obsm:
    fig_pw_dot = pc.pl.archetype_regression_dotplot(
        sub, top_n=10, exclusive_only=True,
        feature_type="pathways", show=False)
```

Fix manual scatter scaling using the same pattern from Task 9 Step 3.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_e2e_myeloid.py
git commit -m "Steps 15-17: debug violin plots, exclusive dotplots, pathway display, scaling"
```

---

## Task 12: Fix FDR numerical floor display

**Files:**
- Modify: `scripts/run_e2e_myeloid.py` (anywhere FDR values are displayed in tables)

- [ ] **Step 1: Search for FDR display formatting**

Search the script for all places where FDR q-values are formatted for display (look for `:.2e`, `:.3e`, `FDR`, `fdr`, `q value`).

- [ ] **Step 2: Add a display helper for extreme p-values**

Add a helper function near the top of the script:

```python
def fmt_pval(p, threshold=1e-300):
    """Format p-value, replacing machine-epsilon floor with readable text."""
    if p is None or np.isnan(p):
        return "NA"
    if p < threshold:
        return "< 1e-300"
    if p > 0.99:
        return f"{p:.3f}"
    return f"{p:.2e}"
```

- [ ] **Step 3: Apply to all FDR display points**

Replace raw `f"{fdr:.2e}"` formatting with `fmt_pval(fdr)` throughout the script where FDR values appear in tables. Key locations:
- Step 3: interaction FDR table
- Step 3: pathway FDR values
- Step 5: contrast tables
- Step 13: alignment FDR
- Step 14: expansion FDR

- [ ] **Step 4: Commit**

```bash
git add scripts/run_e2e_myeloid.py
git commit -m "Add fmt_pval helper for readable extreme p-values across all steps"
```

---

## Task 13: Prototype diversity metrics

**Files:**
- Modify: `scripts/run_e2e_myeloid.py` (add new step function)

Add a diversity metrics step after step 6 (within-fit comparisons). Uses all genes or gene set scores for now.

- [ ] **Step 1: Verify scikit-bio is available**

Run: `conda run -n archetype python -c "import skbio; print(skbio.__version__)"`

If not installed: `conda install -n archetype -c conda-forge scikit-bio` (ask user first).

If skbio is unavailable, fall back to `scipy.stats.entropy` for Shannon and manual Bray-Curtis.

- [ ] **Step 2: Add diversity step function**

Add after step 6:

```python
def step6b_diversity_metrics(adata, report):
    """Step 6b: Alpha and beta diversity of gene expression per archetype."""
    from scipy.stats import entropy as shannon_entropy
    from scipy.spatial.distance import braycurtis, pdist, squareform

    html = ""

    if "archetypes" not in adata.obs.columns:
        html += error_html("No 'archetypes' column — skipping diversity metrics.")
        report.add_section("Diversity Metrics", html, step_num="6b")
        return

    arch_labels = sorted([a for a in adata.obs["archetypes"].unique()
                          if a != "no_archetype" and not pd.isna(a)])

    # Alpha diversity: Shannon entropy of gene expression per cell, averaged per archetype
    # Use non-negative expression (shift if needed since X is log-normalized)
    X = np.asarray(adata.X.todense()) if hasattr(adata.X, "todense") else np.asarray(adata.X)
    # Shift to non-negative for entropy computation
    X_shifted = X - X.min(axis=1, keepdims=True) + 1e-10

    alpha_rows = []
    for label in arch_labels:
        mask = (adata.obs["archetypes"] == label).values
        cells = X_shifted[mask]
        # Per-cell Shannon entropy (how diverse is each cell's expression profile)
        per_cell_entropy = np.array([shannon_entropy(c / c.sum()) for c in cells])
        alpha_rows.append({
            "Archetype": label,
            "N cells": int(mask.sum()),
            "Mean Shannon H": f"{per_cell_entropy.mean():.4f}",
            "Median Shannon H": f"{np.median(per_cell_entropy):.4f}",
            "Std Shannon H": f"{per_cell_entropy.std():.4f}",
        })
    html += report.df_to_html(pd.DataFrame(alpha_rows),
                              caption="Alpha diversity: Shannon entropy of expression per archetype")

    # Beta diversity: Bray-Curtis between archetype mean expression profiles
    mean_profiles = []
    for label in arch_labels:
        mask = (adata.obs["archetypes"] == label).values
        mean_profiles.append(X_shifted[mask].mean(axis=0))
    mean_profiles = np.array(mean_profiles)

    bc_matrix = squareform(pdist(mean_profiles, metric="braycurtis"))
    bc_df = pd.DataFrame(bc_matrix,
                         index=[f"A{i+1}" for i in range(len(arch_labels))],
                         columns=[f"A{i+1}" for i in range(len(arch_labels))])
    html += report.df_to_html(bc_df, caption="Beta diversity: Bray-Curtis between archetype mean profiles")

    # Bray-Curtis heatmap
    try:
        import plotly.graph_objects as go
        fig = go.Figure(data=go.Heatmap(
            z=bc_matrix, x=bc_df.columns.tolist(), y=bc_df.index.tolist(),
            colorscale="Viridis", text=np.round(bc_matrix, 3).astype(str),
            texttemplate="%{text}", textfont_size=10,
        ))
        fig.update_layout(title="Bray-Curtis dissimilarity between archetypes",
                          width=500, height=450)
        html += safe_plotly_html(report, fig, "Bray-Curtis dissimilarity (0=identical, 1=completely different)")
    except Exception as e:
        html += error_html(f"Bray-Curtis heatmap failed: {e}")

    # Gene set diversity (if pathway scores available)
    if "pathway_scores" in adata.obsm:
        pw_scores = np.asarray(adata.obsm["pathway_scores"])
        pw_names = adata.uns.get("pathway_scores_pathways", [])
        pw_alpha_rows = []
        for label in arch_labels:
            mask = (adata.obs["archetypes"] == label).values
            pw_cells = pw_scores[mask]
            # Variance of pathway scores per archetype (higher = more diverse)
            pw_var = pw_cells.var(axis=0).mean()
            pw_alpha_rows.append({
                "Archetype": label,
                "Mean pathway score variance": f"{pw_var:.4f}",
            })
        html += report.df_to_html(pd.DataFrame(pw_alpha_rows),
                                  caption="Pathway score diversity per archetype")

    # Weight entropy: how committed are cells to one archetype
    weights = np.asarray(adata.obsm.get("cell_archetype_weights", np.array([])))
    if weights.size > 0:
        # Clip for log stability
        w_clipped = np.clip(weights, 1e-10, 1.0)
        weight_entropy = -np.sum(w_clipped * np.log(w_clipped), axis=1)
        entropy_rows = []
        for label in arch_labels:
            mask = (adata.obs["archetypes"] == label).values
            ent = weight_entropy[mask]
            entropy_rows.append({
                "Archetype": label,
                "Mean weight entropy": f"{ent.mean():.4f}",
                "Median": f"{np.median(ent):.4f}",
            })
        html += report.df_to_html(pd.DataFrame(entropy_rows),
                                  caption="Archetype weight entropy (higher = less committed)")

    # Per-condition diversity (if treatment/pCR columns exist)
    for condition_col in ["treatment", "pCR"]:
        if condition_col not in adata.obs.columns:
            continue
        groups = sorted(adata.obs[condition_col].unique())
        cond_rows = []
        for grp in groups:
            mask = (adata.obs[condition_col] == grp).values
            cells = X_shifted[mask]
            per_cell_ent = np.array([shannon_entropy(c / c.sum()) for c in cells])
            cond_rows.append({
                "Group": grp, "N cells": int(mask.sum()),
                "Mean Shannon H": f"{per_cell_ent.mean():.4f}",
            })
        html += report.df_to_html(pd.DataFrame(cond_rows),
                                  caption=f"Expression diversity by {condition_col}")

    report.add_section("Diversity Metrics (Prototype)", html, step_num="6b")
```

- [ ] **Step 3: Wire into main()**

In `main()`, add after step 6:

```python
# -- Step 6b (diversity) --------------------------------------------------
t0 = time.time()
try:
    step6b_diversity_metrics(adata, report)
    log.info(f"Step 6b done in {time.time() - t0:.1f}s")
except Exception as e:
    log.error(f"Step 6b failed: {e}", exc_info=True)
    report.add_section("Diversity Metrics", error_html(f"Step 6b failed: {e}"), step_num="6b")
```

- [ ] **Step 4: Commit**

```bash
git add scripts/run_e2e_myeloid.py
git commit -m "Add step 6b: prototype diversity metrics (Shannon, Bray-Curtis, weight entropy)"
```

---

## Task 14: Headless run on real data

**Files:**
- Run: `scripts/run_e2e_myeloid.py`
- Output: `outputs/e2e_myeloid/e2e_myeloid_report.html`

- [ ] **Step 1: Verify conda env and data availability**

```bash
conda run -n archetype python -c "import peach; print(peach.__version__)"
ls -la /Users/honkala/Desktop/FRTNBC/data/tnbc_myeloid_preprocessed.h5ad
```

- [ ] **Step 2: Run the pipeline headless**

```bash
conda run -n archetype python scripts/run_e2e_myeloid.py 2>&1 | tee outputs/e2e_myeloid/run_log.txt
```

This will take significant time (15-45 min depending on flow training). Monitor for:
- Step failures (any `[ERROR]` in log)
- Pathway display correctness (feature_names in pathway sections)
- FDR value display (no more 5.104e-307 or uniform 9.728e-01)
- Flow-based step 6 completing (21 pairwise flows)
- Diversity metrics step completing
- Violin plots rendering in step 15

- [ ] **Step 3: Check output report exists and has content**

```bash
ls -la outputs/e2e_myeloid/e2e_myeloid_report.html
```

Open the HTML report and verify each section renders. Look for:
- Pathway plots showing pathway names (not gene names)
- Archetype space plots with vertices visible
- Volcano labels readable
- Interaction tables with new classification types
- Flow similarity heatmap in step 6
- Diversity metrics tables in step 6b
- Violin plots in step 15

- [ ] **Step 4: Log any issues for iteration**

Create a brief summary of any remaining issues found during the headless run. If critical issues exist, they should be fixed before generating the final dated report.

- [ ] **Step 5: Generate dated report**

After all issues are resolved, ensure the report path includes the run timestamp:

```python
_RUN_ID = time.strftime("%Y%m%d_%H%M%S")
REPORT_PATH = os.path.join(OUTPUT_DIR, f"e2e_myeloid_report_{_RUN_ID}.html")
```

The script already generates timestamped output if configured. Verify the final report is saved with a date stamp.

- [ ] **Step 6: Commit all final changes**

```bash
git add -A
git commit -m "E2E myeloid audit: all fixes verified via headless run"
```

---

## Execution Notes

- **Task dependency:** Tasks 1-4 (library changes) must complete before Tasks 5-13 (script changes). Tasks 5-13 are largely independent of each other.
- **Parallelizable:** Tasks 1, 2, 3 can run in parallel (different library files). Tasks 5-12 can mostly run in parallel after library tasks complete.
- **Task 14 (headless run)** must wait for all other tasks.
- **Iteration:** Task 14 may reveal additional issues requiring a second pass.
- **Memory:** The myeloid dataset is ~10K cells. Step 6 trains K*(K-1)/2 flow models. With reduced epochs (200) and smaller hidden dims, this should complete in ~5-10 min total for step 6.

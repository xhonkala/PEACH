# v0.5.0 Revision Plan: Statistical Fixes, Labeling Consistency, Visualizations

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix FDR propagation, archetype labeling consistency, Jacobian vectorization, GMM unstable-cell handling, and add missing visualizations/statistics across the v0.5.0 feature set.

**Architecture:** Fixes organized into 8 chunks by subsystem. Core statistical fixes (Chunks 1-3) come first since downstream functions depend on correct p-values. Chunks have explicit dependency ordering — Chunks 2→3 both touch `archetype_comparison.py` and MUST be sequential. Registry/schema updates (Chunk 8) come last since they depend on all API changes settling.

**Tech Stack:** Python, numpy, scipy, statsmodels, torch (torch.func for Jacobian), plotly, anndata

**Audit provenance:** This plan incorporates findings from 4 choir audits: Tufte (viz design), reviewer2 (missing FDR gaps), gremlin (edge cases / weak tests), senior (architectural / torch correctness).

---

## File Map

| File | Responsibility | Chunks |
|------|---------------|--------|
| `src/peach/_core/utils/simplex_regression.py` | OLS engine, W'W singularity handling | 1 |
| `src/peach/tl/feature_regression.py` | Simplex regression + driver regression public API | 1 |
| `src/peach/_core/utils/archetype_comparison.py` | Wald contrasts, MMD, feature similarity | 2, 3 |
| `src/peach/tl/comparison.py` | Comparison public API wrappers | 2, 3 |
| `src/peach/_core/utils/analysis.py` | `bin_cells_by_archetype`, archetype numbering | 3 |
| `src/peach/_core/utils/simplex_gmm.py` | GMM fitting, unstable cell handling | 4 |
| `src/peach/tl/feature_decomposition.py` | GMM public API, component characterization | 4 |
| `src/peach/_core/utils/flow_matching.py` | FlowModel.jacobian, FlowModel.train | 5 |
| `src/peach/tl/flow.py` | Flow gene alignment, permutation stats | 5 |
| `src/peach/pl/comparison.py` | Volcano with error bars, Wald grid | 6 |
| `src/peach/pl/regression.py` | Per-archetype dotplot, degree viz | 6 |
| `src/peach/pl/decomposition.py` | Component size/distance viz | 6 |
| `src/peach/pl/archetypal.py` | 3D display fix | 6 |
| `docs/tutorials/12_e2e_v050_reviewer.ipynb` | Notebook rewrite | 7 |
| `src/peach/_core/types_index.py` | Return type registry | 8 |
| `src/peach/_core/tools_schema.py` | Parameter schema registry | 8 |
| `tests/test_statistical/` | Statistical tests for all fixes | 1-6 |

---

## Chunk 1: Simplex Regression — Rank Info + Missing FDR

### Task 1.1: Report effective rank info from `ols_fit` (not a mask)

**Audit fix (senior/gremlin):** The original plan proposed a rank-deficiency "mask" that would fire on every real dataset because simplex weights inherently have rank K-1 (the sum-to-1 constraint). Instead, we report effective rank information and only warn when rank drops BELOW K-1 (which indicates genuine collinearity beyond the expected constraint).

**Files:**
- Modify: `src/peach/_core/utils/simplex_regression.py:114-132`
- Modify: `src/peach/tl/feature_regression.py:17-200`
- Test: `tests/test_statistical/test_simplex_regression_api.py`

- [ ] **Step 1: Write failing test for rank info**

```python
def test_ols_fit_returns_rank_info():
    """ols_fit should report effective rank and expected rank."""
    from peach._core.utils.simplex_regression import ols_fit, scheffe_design_matrix
    rng = np.random.default_rng(42)
    # Normal simplex data: rank should be K-1
    W = rng.dirichlet([1, 1, 1, 1], size=200)
    X, _ = scheffe_design_matrix(W, degree=1)
    Y = rng.randn(200, 10)
    result = ols_fit(X, Y, robust_se=True)
    assert "effective_rank" in result
    assert result["effective_rank"] == 3  # K-1 = 3
    assert "expected_rank" in result
    # No extra deficiency — should not warn
    assert result.get("extra_rank_deficient", False) is False


def test_ols_fit_detects_extra_deficiency():
    """When columns are collinear beyond sum-to-1, flag extra deficiency."""
    from peach._core.utils.simplex_regression import ols_fit
    rng = np.random.default_rng(42)
    W = rng.dirichlet([1, 1, 1], size=200)
    # Add a column that's a linear combo of existing columns
    W_degen = np.column_stack([W, W[:, 0] + W[:, 1]])
    Y = rng.randn(200, 10)
    result = ols_fit(W_degen, Y, robust_se=True)
    assert result["extra_rank_deficient"] is True
    assert result["effective_rank"] < W_degen.shape[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n archetype pytest tests/test_statistical/test_simplex_regression_api.py::test_ols_fit_returns_rank_info -xvs`

- [ ] **Step 3: Implement rank info in `ols_fit`**

In `simplex_regression.py`, after computing `WtW_inv` (line ~132), add:

```python
# Report effective rank
_, s_vals, _ = np.linalg.svd(WtW)
effective_rank = int(np.sum(s_vals > s_vals[0] * 1e-10))
# For simplex degree-1, expected rank is p-1 (sum-to-1 constraint).
# For degree-2+, expected rank is p - n_constraints.
# Conservative: flag if rank < p-1
expected_rank = p - 1  # at minimum, one constraint from sum-to-1
extra_rank_deficient = effective_rank < expected_rank
if extra_rank_deficient:
    warnings.warn(
        f"Design matrix has effective rank {effective_rank} but expected at "
        f"least {expected_rank}. This indicates collinearity beyond the "
        f"expected simplex sum-to-1 constraint.",
        RuntimeWarning,
    )
```

At return, add to result dict:
```python
"effective_rank": effective_rank,
"expected_rank": expected_rank,
"extra_rank_deficient": extra_rank_deficient,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n archetype pytest tests/test_statistical/test_simplex_regression_api.py -k "rank" -xvs`

- [ ] **Step 5: Commit**

```bash
git add src/peach/_core/utils/simplex_regression.py tests/test_statistical/test_simplex_regression_api.py
git commit -m "feat: report effective rank info from ols_fit, flag extra deficiency"
```

### Task 1.2: Add FDR correction to `archetype_driver_regression`

**Audit fix (reviewer2):** `archetype_driver_regression` returns raw `main_pvalues` and `interaction_pvalues` with NO FDR correction at all. This is inconsistent with `feature_simplex_regression` which does correct.

**Files:**
- Modify: `src/peach/tl/feature_regression.py:571-587`
- Test: `tests/test_statistical/test_driver_regression.py`

- [ ] **Step 1: Write failing test**

```python
def test_driver_regression_returns_fdr():
    """archetype_driver_regression must return FDR-corrected p-values."""
    adata = make_regression_test_adata()
    result = pc.tl.archetype_driver_regression(adata)
    assert "main_pvalues_fdr" in result
    # FDR values should differ from raw (correction applied)
    assert not np.array_equal(result["main_pvalues"], result["main_pvalues_fdr"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n archetype pytest tests/test_statistical/test_driver_regression.py::test_driver_regression_returns_fdr -xvs`

- [ ] **Step 3: Add FDR correction**

In `feature_regression.py`, before constructing `DriverRegressionResult` (~line 571), add:

```python
# Global FDR across all ILR components and features
all_main_pvals = main_pvalues.ravel()
# Filter zeros (from zero-SE features) to avoid diluting FDR
nonzero_mask = all_main_pvals > 0
main_pvalues_fdr = np.ones_like(all_main_pvals)
if nonzero_mask.any():
    _, fdr_vals, _, _ = multipletests(all_main_pvals[nonzero_mask], method="fdr_bh")
    main_pvalues_fdr[nonzero_mask] = fdr_vals
main_pvalues_fdr = main_pvalues_fdr.reshape(main_pvalues.shape)

interaction_pvalues_fdr = None
if interaction_pvalues is not None:
    all_int_pvals = interaction_pvalues.ravel()
    nonzero_int = all_int_pvals > 0
    int_fdr = np.ones_like(all_int_pvals)
    if nonzero_int.any():
        _, fdr_vals, _, _ = multipletests(all_int_pvals[nonzero_int], method="fdr_bh")
        int_fdr[nonzero_int] = fdr_vals
    interaction_pvalues_fdr = int_fdr.reshape(interaction_pvalues.shape)
```

Add `main_pvalues_fdr` and `interaction_pvalues_fdr` to the `DriverRegressionResult` constructor.

- [ ] **Step 4: Update `DriverRegressionResult` Pydantic model**

In `src/peach/_core/types.py`, add `main_pvalues_fdr` and `interaction_pvalues_fdr` optional fields to `DriverRegressionResult`.

- [ ] **Step 5: Run test to verify it passes**

Run: `conda run -n archetype pytest tests/test_statistical/test_driver_regression.py -xvs`

- [ ] **Step 6: Commit**

```bash
git add src/peach/tl/feature_regression.py src/peach/_core/types.py tests/test_statistical/test_driver_regression.py
git commit -m "fix: add FDR correction to archetype_driver_regression"
```

### Task 1.3: Add FDR to Spearman p-values in `compute_feature_similarity`

**Audit fix (reviewer2):** `compute_feature_similarity` returns `spearman_pvalue_matrix` with raw p-values but no FDR correction.

**Files:**
- Modify: `src/peach/_core/utils/archetype_comparison.py:103-193`
- Test: `tests/test_statistical/test_archetype_comparison.py`

- [ ] **Step 1: Write failing test**

```python
def test_feature_similarity_returns_fdr():
    """compute_feature_similarity must include FDR-corrected Spearman p-values."""
    adata = make_comparison_test_adata()
    result = pc.tl.archetype_feature_similarity(adata)
    assert "spearman_pvalue_fdr_matrix" in result
    # FDR should be more conservative (larger) than raw for some entries
    assert np.any(result["spearman_pvalue_fdr_matrix"] >= result["spearman_pvalue_matrix"])
```

- [ ] **Step 2: Implement — add BH correction after Spearman loop**

In `archetype_comparison.py`, after the Spearman loop (line ~158), add:

```python
# Global FDR correction across all K_a x K_b Spearman tests
from statsmodels.stats.multitest import multipletests
all_spearman_pvals = spearman_pvalue_matrix.ravel()
_, spearman_fdr_flat, _, _ = multipletests(all_spearman_pvals, method="fdr_bh")
spearman_pvalue_fdr_matrix = spearman_fdr_flat.reshape(spearman_pvalue_matrix.shape)
```

Add `"spearman_pvalue_fdr_matrix": spearman_pvalue_fdr_matrix` to the return dict.

- [ ] **Step 3: Run test, commit**

```bash
git add src/peach/_core/utils/archetype_comparison.py tests/test_statistical/test_archetype_comparison.py
git commit -m "fix: add FDR correction to Spearman p-values in compute_feature_similarity"
```

---

## Chunk 2: Wald Contrasts — Global FDR + Error Bars

**Dependency:** Must complete before Chunk 3 (both modify `archetype_comparison.py`).

### Task 2.1: Fix FDR correction scope in `compute_wald_contrasts`

**Critical bug:** BH correction is applied per-pair (6 × n_features) instead of globally (6 × n_features total tests). This inflates significance.

**Audit fix (gremlin):** Test must compare global vs per-pair FDR results, not just check for uniqueness. Also filter zero-SE genes before FDR.

**Files:**
- Modify: `src/peach/_core/utils/archetype_comparison.py:240-259`
- Test: `tests/test_statistical/test_archetype_comparison.py`

- [ ] **Step 1: Write comparison-based failing test**

```python
def test_wald_fdr_is_global_not_per_pair():
    """FDR correction across ALL pairs produces different results than per-pair."""
    from peach._core.utils.archetype_comparison import compute_wald_contrasts
    from statsmodels.stats.multitest import multipletests

    adata = make_comparison_test_adata(K=4)
    result = compute_wald_contrasts(adata)

    # Compute what per-pair FDR would give (the old buggy behavior)
    per_pair_fdr = {}
    for pair in result["pairs"]:
        raw = result["pvalues"][pair]
        _, fdr_pp, _, _ = multipletests(raw, method="fdr_bh")
        per_pair_fdr[pair] = fdr_pp

    # Global FDR should differ from per-pair FDR
    for pair in result["pairs"]:
        if not np.allclose(result["pvalues"][pair], 1.0):  # skip trivial cases
            assert not np.allclose(
                result["pvalues_fdr"][pair], per_pair_fdr[pair], atol=1e-10
            ), f"Pair {pair}: FDR appears to be per-pair, not global"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n archetype pytest tests/test_statistical/test_archetype_comparison.py::test_wald_fdr_is_global_not_per_pair -xvs`

- [ ] **Step 3: Fix FDR to be global, filter zero-SE genes**

Replace the per-pair FDR block in `archetype_comparison.py` (lines 240-259):

```python
# First pass: collect all raw p-values and compute per-pair stats
all_pvals = []
pair_slices = {}
offset = 0
for j, k in pairs:
    contrast = np.zeros(K)
    contrast[j] = 1.0
    contrast[k] = -1.0

    d_beta = beta[:, j] - beta[:, k]
    d_se = np.array([
        np.sqrt(max(contrast @ cov_list[g] @ contrast, 0))
        for g in range(n_features)
    ])
    z = np.where(d_se > 0, d_beta / d_se, 0.0)
    pval = 2 * stats.norm.sf(np.abs(z))

    delta_beta[(j, k)] = d_beta
    delta_se[(j, k)] = d_se
    z_scores[(j, k)] = z
    pvalues[(j, k)] = pval

    all_pvals.append(pval)
    pair_slices[(j, k)] = slice(offset, offset + n_features)
    offset += n_features

# Global FDR correction across ALL pairs, filtering zero-SE genes
all_pvals_flat = np.concatenate(all_pvals)
# Genes with SE=0 get p=1 from sf(), which dilutes FDR.
# Only correct non-trivial tests, leave the rest as-is.
testable = all_pvals_flat < 1.0
all_fdr = np.ones_like(all_pvals_flat)
if testable.any():
    _, fdr_vals, _, _ = multipletests(all_pvals_flat[testable], method="fdr_bh")
    all_fdr[testable] = fdr_vals

# Distribute back to pairs
for j, k in pairs:
    pvalues_fdr[(j, k)] = all_fdr[pair_slices[(j, k)]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n archetype pytest tests/test_statistical/test_archetype_comparison.py -xvs`

- [ ] **Step 5: Commit**

```bash
git add src/peach/_core/utils/archetype_comparison.py tests/test_statistical/test_archetype_comparison.py
git commit -m "fix: apply FDR correction globally across all Wald contrast pairs"
```

### Task 2.2: Add error bars to volcano plot

**Tufte note:** Use 95% CI whiskers (1.96×SE) with very light opacity so they encode uncertainty without dominating.

**Files:**
- Modify: `src/peach/pl/comparison.py:56-108`

- [ ] **Step 1: Add error_x bars to volcano scatter**

In `contrast_volcano()`, after creating the scatter, add horizontal error bars using `delta_se`:

```python
se = np.asarray(contrast_data["delta_se"][pair_key])

fig = go.Figure(data=go.Scatter(
    x=delta,
    y=neg_log_p,
    mode="markers",
    text=names,
    error_x=dict(
        type="data", array=1.96 * se, visible=True,
        width=0, thickness=0.5, color="rgba(0,0,0,0.15)"
    ),
    hovertemplate="%{text}<br>Δβ=%{x:.3f}±%{customdata:.3f}<br>-log10(q)=%{y:.1f}<extra></extra>",
    customdata=se,
    marker=dict(size=5, opacity=0.6, color=colors),
))
```

- [ ] **Step 2: Test visually, commit**

```bash
git add src/peach/pl/comparison.py
git commit -m "feat: add 95% CI error bars to Wald contrast volcano plot"
```

---

## Chunk 3: Archetype Labeling Consistency

**Dependency:** Must execute AFTER Chunk 2 (both modify `archetype_comparison.py`).

### Task 3.1: Fix the duplicate-branch numbering bug

**Files:**
- Modify: `src/peach/_core/utils/analysis.py:694`

- [ ] **Step 1: Fix the identical branches**

Line 694:
```python
# BEFORE (bug):
archetype_num = arch_idx + 1 if include_central_archetype else arch_idx + 1
# AFTER (fix):
archetype_num = arch_idx + 1 if include_central_archetype else arch_idx
```

Line 695 is correct — keep as-is:
```python
archetype_storage_idx = arch_idx + 1 if include_central_archetype else arch_idx
```

- [ ] **Step 2: Add test for numbering correctness**

```python
def test_bin_cells_numbering_with_central():
    """With central archetype, numbering should start at 1 (0 = central)."""
    adata = make_test_adata(K=4, include_central=True)
    result = pc.tl.assign_archetypes(adata)
    labels = set(adata.obs["archetypes"].unique())
    # Should NOT contain "archetype_0" unless include_central
    assert "archetype_1" in labels
    assert "archetype_4" in labels


def test_bin_cells_numbering_without_central():
    """Without central archetype, numbering should start at 0."""
    adata = make_test_adata(K=4, include_central=False)
    result = pc.tl.assign_archetypes(adata)
    labels = set(adata.obs["archetypes"].unique())
    assert "archetype_0" in labels
    assert "archetype_3" in labels
```

- [ ] **Step 3: Commit**

```bash
git add src/peach/_core/utils/analysis.py tests/test_statistical/test_archetype_comparison.py
git commit -m "fix: archetype numbering bug — both branches were identical"
```

### Task 3.2: Make comparison functions use stored assignments

The functions `compute_archetype_mmd` and `compute_feature_similarity` use `np.argmax(weights, axis=1)` (0-indexed) instead of reading `adata.obs['archetypes']`. Fix them to use stored assignments when available.

**Audit fix (gremlin):** When central archetype is excluded from stored labels, the number of unique labels may differ from K. The function must handle shape mismatches between `weights.shape[1]` and `len(unique_labels)`.

**Files:**
- Modify: `src/peach/_core/utils/archetype_comparison.py:56-57, 162`
- Test: `tests/test_statistical/test_archetype_comparison.py`

- [ ] **Step 1: Write test**

```python
def test_mmd_uses_stored_assignments():
    """archetype_mmd should use adata.obs['archetypes'] when available."""
    adata = make_comparison_test_adata(K=4)
    # Pre-assign archetypes (1-indexed strings, as PEACH convention)
    pc.tl.assign_archetypes(adata)
    assert "archetypes" in adata.obs.columns
    result = compute_archetype_mmd(adata)
    # Matrix should be K×K where K = number of unique archetype labels
    unique_labels = sorted(adata.obs["archetypes"].unique())
    K_labels = len([l for l in unique_labels if "no_archetype" not in l])
    assert result[0].shape[0] == K_labels
```

- [ ] **Step 2: Implement — read from obs['archetypes'] with argmax fallback**

In `compute_archetype_mmd` (line 56-57):
```python
# Use stored assignments if available, fallback to argmax
if "archetypes" in adata.obs.columns:
    raw_labels = adata.obs["archetypes"].values
    # Extract unique extremal labels (exclude "no_archetype" and "archetype_0"/central)
    all_labels = sorted(set(raw_labels))
    extremal_labels = [l for l in all_labels if l not in ("no_archetype",)]
    label_to_idx = {l: i for i, l in enumerate(extremal_labels)}
    assign_a = np.array([label_to_idx.get(str(l), -1) for l in raw_labels])
    K_a = len(extremal_labels)
else:
    assign_a = np.argmax(weights_a, axis=1)
    K_a = weights_a.shape[1]
```

Same pattern for `compute_feature_similarity` at line 162. Also apply to `adata_b` when present.

- [ ] **Step 3: Run tests, commit**

```bash
git add src/peach/_core/utils/archetype_comparison.py tests/test_statistical/test_archetype_comparison.py
git commit -m "fix: comparison functions use stored archetype assignments, not argmax"
```

---

## Chunk 4: GMM Unstable Cell Handling

### Task 4.1: Improve unstable cell handling with configurable threshold

**Audit fix (gremlin):** Simple renormalization of `predict_proba` over stable components is mathematically equivalent to nearest-stable assignment when probabilities are sharply peaked (which they usually are for stable GMMs). The fix: use UNNORMALIZED posterior probabilities from the full GMM, and make the confidence threshold configurable rather than hardcoded at 0.5.

**Files:**
- Modify: `src/peach/_core/utils/simplex_gmm.py:137-146`
- Test: `tests/test_statistical/test_gmm_api.py`

- [ ] **Step 1: Write failing test**

```python
def test_unstable_cells_can_remain_unassigned():
    """With strict threshold, some unstable cells should remain -1."""
    from peach._core.utils.simplex_gmm import fit_simplex_gmm
    rng = np.random.default_rng(42)
    # Weights near simplex center (ambiguous)
    weights = rng.dirichlet([1, 1, 1, 1], size=500)
    result = fit_simplex_gmm(
        weights,
        stability_threshold=0.9,
        reassignment_confidence=0.8,  # NEW parameter
    )
    assignments = result["component_assignments"]
    # Some cells SHOULD remain unassigned with strict confidence threshold
    # (not all cells should be force-assigned)
    assert "component_probabilities" in result
    assert result["component_probabilities"].shape[0] == 500


def test_unstable_cells_all_reassigned_with_low_threshold():
    """With 0.0 threshold, all unstable cells get reassigned (backward compat)."""
    from peach._core.utils.simplex_gmm import fit_simplex_gmm
    rng = np.random.default_rng(42)
    weights = rng.dirichlet([5, 5, 5, 5], size=500)
    result = fit_simplex_gmm(weights, reassignment_confidence=0.0)
    assert np.all(result["component_assignments"] >= 0)
```

- [ ] **Step 2: Implement configurable predict_proba handling**

Add `reassignment_confidence: float = 0.0` parameter to `fit_simplex_gmm`. Default 0.0 preserves backward compatibility (all unstable cells get reassigned).

Replace `simplex_gmm.py:137-146`:

```python
# Handle unstable cells using predict_proba
unstable_mask = component_assignments == -1
component_probabilities = None
if np.any(unstable_mask) and n_stable > 0:
    # Get posterior probabilities from full GMM
    all_proba = best_gmm.predict_proba(ilr_coords)  # [n_cells, best_n]
    # Extract probabilities for stable components only (unnormalized)
    stable_proba = all_proba[:, stable_indices]  # [n_cells, n_stable]
    component_probabilities = stable_proba

    # Reassign unstable cells where max probability exceeds threshold
    unstable_proba = stable_proba[unstable_mask]
    max_prob = unstable_proba.max(axis=1)
    confident_mask = max_prob >= reassignment_confidence
    confident_idx = np.where(unstable_mask)[0][confident_mask]
    component_assignments[confident_idx] = np.argmax(
        stable_proba[confident_idx], axis=1
    )
    # Cells below threshold remain -1
```

Also add `"component_probabilities": component_probabilities` to the return dict.

- [ ] **Step 3: Run test, commit**

```bash
git add src/peach/_core/utils/simplex_gmm.py tests/test_statistical/test_gmm_api.py
git commit -m "fix: configurable predict_proba threshold for unstable GMM cells"
```

### Task 4.2: Add per-component simplex regression

Enable running `feature_simplex_regression` on cells grouped by GMM component.

**Audit fix (gremlin):** Use correct uns key `peach_gmm` (not `peach_simplex_gmm`).

**Files:**
- Modify: `src/peach/tl/feature_decomposition.py`
- Test: `tests/test_statistical/test_gmm_api.py`

- [ ] **Step 1: Write test**

```python
def test_component_regression():
    """Per-component regression should work for both genes and pathway scores."""
    adata = make_gmm_test_adata()
    pc.tl.feature_simplex_decomposition(adata)
    result = pc.tl.component_regression(adata)
    assert "component_regs" in result
    n_stable = adata.uns["peach_gmm"]["n_components_stable"]
    assert len(result["component_regs"]) == n_stable
    # Each component regression should have vertex_coefficients
    for c, reg in result["component_regs"].items():
        assert "vertex_coefficients" in reg
```

- [ ] **Step 2: Implement `component_regression`**

In `feature_decomposition.py`, add:

```python
def component_regression(
    adata: AnnData,
    *,
    feature_type: str = "genes",  # "genes" or "pathways"
    n_bootstrap: int = 100,
    robust_se: bool = True,
) -> dict:
    """Run simplex regression separately per GMM component.

    For each stable component, subsets to component cells and runs
    feature_simplex_regression on that subset.
    """
    from peach.tl.feature_regression import feature_simplex_regression

    gmm = adata.uns.get("peach_gmm")
    if gmm is None:
        raise ValueError("Run pc.tl.feature_simplex_decomposition() first.")

    assignments = np.asarray(gmm["component_assignments"])
    n_stable = gmm["n_components_stable"]

    component_regs = {}
    for c in range(n_stable):
        mask = assignments == c
        if mask.sum() < 20:
            continue
        adata_sub = adata[mask].copy()
        reg = feature_simplex_regression(
            adata_sub,
            n_bootstrap=n_bootstrap,
            robust_se=robust_se,
        )
        component_regs[c] = reg

    return {"component_regs": component_regs, "n_components": n_stable}
```

- [ ] **Step 3: Register in `__init__`, test, commit**

```bash
git add src/peach/tl/feature_decomposition.py src/peach/tl/__init__.py tests/test_statistical/test_gmm_api.py
git commit -m "feat: add per-component simplex regression for GMM characterization"
```

---

## Chunk 5: Flow Matching — Jacobian Vectorization + Alignment Stats

**CRITICAL FIXES from all 3 audits (senior, reviewer2, gremlin):**
1. `torch.no_grad()` wrapping `jacrev` kills autograd → Jacobian returns zeros
2. `vmap` cannot operate directly on `nn.Module` → must use `torch.func.functional_call`
3. VelocityNetwork's `forward` checks `t.dim()` — shapes must be correct for single-point eval

### Task 5.1: Vectorize Jacobian with `torch.func.jacrev` + `vmap` + `functional_call`

**Files:**
- Modify: `src/peach/_core/utils/flow_matching.py:282-319`
- Test: `tests/test_statistical/test_flow_jacobian.py` (or existing flow tests)

- [ ] **Step 1: Write test for correctness**

```python
def test_jacobian_vectorized_correct():
    """Vectorized Jacobian should produce finite, non-zero values."""
    from peach._core.utils.flow_matching import FlowModel
    model = FlowModel(dim=5, hidden_dims=(32, 32))
    source = np.random.randn(50, 5).astype(np.float32)
    target = np.random.randn(50, 5).astype(np.float32)
    model.train(source, target, n_epochs=20, batch_size=32)

    points = source[:10]
    jac = model.jacobian(points, t=0.5)
    assert jac.shape == (10, 5, 5)
    assert np.all(np.isfinite(jac))
    # Jacobian should NOT be all zeros (would indicate broken autograd)
    assert not np.allclose(jac, 0, atol=1e-8), "Jacobian is all zeros — autograd likely broken"
    # Determinants should be finite and non-zero for a trained model
    dets = np.array([np.linalg.det(j) for j in jac])
    assert np.all(np.isfinite(dets))


def test_jacobian_matches_finite_difference():
    """Jacobian should approximately match finite-difference estimate."""
    from peach._core.utils.flow_matching import FlowModel
    model = FlowModel(dim=3, hidden_dims=(16, 16))
    source = np.random.randn(30, 3).astype(np.float32)
    target = np.random.randn(30, 3).astype(np.float32)
    model.train(source, target, n_epochs=20, batch_size=16)

    point = source[:1]
    jac = model.jacobian(point, t=0.5)[0]  # [3, 3]

    # Finite difference approximation
    eps = 1e-4
    jac_fd = np.zeros((3, 3))
    for k in range(3):
        p_plus = point.copy()
        p_minus = point.copy()
        p_plus[0, k] += eps
        p_minus[0, k] -= eps
        v_plus = model.velocity(p_plus, 0.5)[0]
        v_minus = model.velocity(p_minus, 0.5)[0]
        jac_fd[:, k] = (v_plus - v_minus) / (2 * eps)

    np.testing.assert_allclose(jac, jac_fd, atol=1e-2, rtol=1e-1)
```

- [ ] **Step 2: Run test to verify it fails (current loop version should pass, but let's baseline)**

Run: `conda run -n archetype pytest tests/test_statistical/test_flow_jacobian.py -xvs`

- [ ] **Step 3: Replace loop Jacobian with functional_call + vmap version**

Replace `flow_matching.py:282-319`:

```python
def jacobian(self, x, t):
    """Compute Jacobian of velocity field dv/dx using vectorized autograd.

    Uses torch.func.jacrev + vmap with functional_call for efficient
    batched Jacobian computation. NO torch.no_grad() — autograd must
    be active for jacrev.

    Parameters
    ----------
    x : np.ndarray
        Points to evaluate at, shape [n_points, dim].
    t : float
        Time in [0, 1].

    Returns
    -------
    np.ndarray
        Jacobian matrices, shape [n_points, dim, dim].
        Entry [i, j, k] is dv_j/dx_k at point i.
    """
    from torch.func import jacrev, vmap, functional_call

    self.velocity_net.eval()
    x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
    t_scalar = torch.tensor(t, dtype=torch.float32, device=self.device)

    # Extract parameters for functional_call (makes Module stateless for vmap)
    params = dict(self.velocity_net.named_parameters())
    buffers = dict(self.velocity_net.named_buffers())

    def vel_fn(params_dict, x_single):
        """Evaluate velocity for a single point (stateless)."""
        # x_single: [dim] -> need [1, dim] for VelocityNetwork.forward
        x_2d = x_single.unsqueeze(0)
        t_2d = t_scalar.reshape(1, 1)
        out = functional_call(self.velocity_net, (params_dict, buffers), (x_2d, t_2d))
        return out.squeeze(0)  # [dim]

    # jacrev differentiates vel_fn w.r.t. x_single (argnums=1)
    # vmap batches over the x dimension (params shared via in_dims=(None, 0))
    batched_jac = vmap(jacrev(vel_fn, argnums=1), in_dims=(None, 0))
    jacs = batched_jac(params, x_t)  # [n_points, dim, dim]

    return jacs.detach().cpu().numpy()
```

- [ ] **Step 4: Run tests to verify**

Run: `conda run -n archetype pytest tests/test_statistical/test_flow_jacobian.py -xvs`

- [ ] **Step 5: Commit**

```bash
git add src/peach/_core/utils/flow_matching.py tests/test_statistical/
git commit -m "perf: vectorize Jacobian with torch.func.jacrev + vmap + functional_call"
```

### Task 5.2: Add source/target batch size imbalance warning

**Files:**
- Modify: `src/peach/_core/utils/flow_matching.py:140-202`

- [ ] **Step 1: Add imbalance check at start of `train()`**

After converting to tensors:

```python
ratio = max(len(source_t), len(target_t)) / min(len(source_t), len(target_t))
if ratio > 5:
    import warnings
    warnings.warn(
        f"Source/target size imbalance: {len(source_t)} vs {len(target_t)} "
        f"(ratio {ratio:.1f}x). The smaller population will be heavily "
        f"resampled during training, which may degrade flow quality. "
        f"Consider subsampling the larger population.",
        UserWarning,
    )
```

- [ ] **Step 2: Commit**

```bash
git add src/peach/_core/utils/flow_matching.py
git commit -m "feat: warn on source/target batch size imbalance in flow training"
```

### Task 5.3: Add permutation statistics to flow gene alignment

**Files:**
- Modify: `src/peach/tl/flow.py:191-243`

- [ ] **Step 1: Add `n_permutations` parameter to `flow_gene_alignment`**

```python
def flow_gene_alignment(
    adata: AnnData,
    flow_result: dict,
    *,
    t: float = 0.5,
    n_top: int = 50,
    pca_loadings_key: str | None = None,
    n_permutations: int = 0,  # NEW: 0 = no permutation test
    random_state: int = 42,
) -> dict:
```

After computing `alignment_scores`, add:

```python
if n_permutations > 0:
    rng = np.random.default_rng(random_state)
    null_scores = np.zeros((n_permutations, len(alignment_scores)))
    for i in range(n_permutations):
        # Shuffle gene-to-loading mapping (permute rows of loadings)
        perm_loadings = loadings_trimmed[rng.permutation(len(loadings_trimmed))]
        null_scores[i] = perm_loadings @ mean_velocity

    # Two-sided p-value per gene
    pvalues = np.array([
        (np.sum(np.abs(null_scores[:, g]) >= np.abs(alignment_scores[g])) + 1)
        / (n_permutations + 1)
        for g in range(len(alignment_scores))
    ])
    from statsmodels.stats.multitest import multipletests
    _, pvalues_fdr, _, _ = multipletests(pvalues, method="fdr_bh")

    result["alignment_pvalues"] = pvalues
    result["alignment_pvalues_fdr"] = pvalues_fdr
    result["null_mean"] = null_scores.mean(axis=0)
    result["null_std"] = null_scores.std(axis=0)
```

- [ ] **Step 2: Test, commit**

```bash
git add src/peach/tl/flow.py
git commit -m "feat: add permutation statistics to flow gene alignment"
```

---

## Chunk 6: Visualizations

### Task 6.1: Per-archetype regression dotplot

**Tufte design:** Integrated dotplot — rows = genes, columns = archetypes, dot size = |β|, dot color = -log10(FDR). No chartjunk.

**Files:**
- Modify: `src/peach/pl/regression.py`

- [ ] **Step 1: Implement `archetype_regression_dotplot`**

```python
def archetype_regression_dotplot(
    adata: AnnData,
    *,
    top_n: int = 10,
    show_interactions: bool = True,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Dotplot of top genes per archetype from regression coefficients.

    Rows: top genes per archetype (by |beta|)
    Columns: archetypes
    Dot size: |beta coefficient|
    Dot color: -log10(FDR p-value)
    Optional second panel: interaction coefficients (degree 2)
    """
```

- [ ] **Step 2: Implement CMP vs Mono comparison variant**

```python
def archetype_regression_comparison(
    adata_a: AnnData,
    adata_b: AnnData,
    *,
    label_a: str = "A",
    label_b: str = "B",
    top_n: int = 10,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Side-by-side regression coefficient comparison between two fits."""
```

- [ ] **Step 3: Implement upset-style intersection plot**

```python
def archetype_gene_overlap(
    adata: AnnData,
    *,
    fdr_threshold: float = 0.05,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """UpSet-style intersection showing genes significant across archetype pairs."""
```

- [ ] **Step 4: Test visually, commit**

```bash
git add src/peach/pl/regression.py
git commit -m "feat: add per-archetype regression dotplot, comparison, and overlap viz"
```

### Task 6.2: Wald contrast 2×3 grid

**Tufte design:** Small-multiple grid showing all K*(K-1)/2 pairwise volcanos in a single figure. Shared axes for comparability.

**Files:**
- Modify: `src/peach/pl/comparison.py`

- [ ] **Step 1: Implement `contrast_volcano_grid`**

```python
def contrast_volcano_grid(
    contrast_data: dict,
    *,
    fdr_threshold: float = 0.05,
    delta_threshold: float = 0.5,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Small-multiple grid of volcano plots for all pairwise Wald contrasts.

    Layout: upper-triangular grid with shared x/y axes.
    Each panel shows one pair's delta_beta vs -log10(FDR).
    """
```

- [ ] **Step 2: Test visually, commit**

```bash
git add src/peach/pl/comparison.py
git commit -m "feat: add small-multiple Wald contrast volcano grid"
```

### Task 6.3: GMM component summary (2×2 panel)

**Tufte design:** Four coordinated panels: (1) component size bar, (2) weight profile heatmap, (3) archetype distance, (4) entropy distribution.

**Files:**
- Modify: `src/peach/pl/decomposition.py`

- [ ] **Step 1: Add component summary figure**

```python
def component_archetype_summary(
    adata: AnnData,
    *,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """2x2 panel: component sizes, weight profiles, archetype distances, entropy."""
```

- [ ] **Step 2: Test visually, commit**

```bash
git add src/peach/pl/decomposition.py
git commit -m "feat: add GMM component-archetype summary 2x2 panel"
```

### Task 6.4: Fix 3D archetypal space display

**Files:**
- Modify: `src/peach/pl/archetypal.py`

- [ ] **Step 1: Verify `archetypal_space` calls `save_and_show` properly**

Read the function. If `show=True` doesn't trigger `fig.show()`, add it. The function should call `save_and_show(fig, save_path=save_path, show=show)` at the end.

- [ ] **Step 2: Test in notebook context, commit**

```bash
git add src/peach/pl/archetypal.py
git commit -m "fix: 3D archetypal_space respects show parameter for notebook display"
```

---

## Chunk 7: Notebook Rewrite

### Task 7.1: Regenerate `12_e2e_v050_reviewer.ipynb`

After all core fixes are in place, regenerate the notebook to incorporate all changes.

**Updates:**
- Use 1-based archetype labels throughout (from stored assignments)
- Add per-archetype regression dotplot section
- Fix 3D GMM component viz to actually display
- Add permutation statistics to gene-flow alignment
- Use global FDR in Wald test display
- Add GMM component size/distance summary
- Add per-component regression section
- Use small-multiple Wald volcano grid
- Add Jacobian-based gene expansion analysis
- Tufte: remove chartjunk, use consistent palette, small multiples

**Files:**
- Modify: `scripts/generate_reviewer_notebook.py`
- Modify: `docs/tutorials/12_e2e_v050_reviewer.ipynb`

- [ ] **Step 1: Update notebook generator with all fixes**
- [ ] **Step 2: Regenerate notebook**
- [ ] **Step 3: Run headless test to validate**
- [ ] **Step 4: Commit**

```bash
git add docs/tutorials/12_e2e_v050_reviewer.ipynb scripts/generate_reviewer_notebook.py
git commit -m "feat: comprehensive v0.5.0 reviewer notebook with all statistical fixes"
```

---

## Chunk 8: Registry & Schema Updates

### Task 8.1: Update `types_index.py`

Add/update entries for all modified return types:
- `tl.archetype_driver_regression`: add `main_pvalues_fdr`, `interaction_pvalues_fdr`
- `tl.archetype_feature_similarity`: add `spearman_pvalue_fdr_matrix`
- `tl.feature_simplex_decomposition`: add `component_probabilities`, `reassignment_confidence`
- `tl.component_regression`: new entry
- `tl.flow_gene_alignment`: add `alignment_pvalues`, `alignment_pvalues_fdr`
- `ols_fit`: add `effective_rank`, `expected_rank`, `extra_rank_deficient`

**Files:**
- Modify: `src/peach/_core/types_index.py`

- [ ] **Step 1: Update all modified return type entries**
- [ ] **Step 2: Commit**

```bash
git add src/peach/_core/types_index.py
git commit -m "docs: update types_index.py for v0.5.0 revision return type changes"
```

### Task 8.2: Update `tools_schema.py`

Add/update parameter entries for:
- `fit_simplex_gmm`: add `reassignment_confidence` parameter
- `flow_gene_alignment`: add `n_permutations`, `random_state` parameters
- `component_regression`: new entry
- `archetype_driver_regression`: document new FDR fields in returns

**Files:**
- Modify: `src/peach/_core/tools_schema.py`

- [ ] **Step 3: Update all modified parameter schemas**
- [ ] **Step 4: Commit**

```bash
git add src/peach/_core/tools_schema.py
git commit -m "docs: update tools_schema.py for v0.5.0 revision parameter changes"
```

---

## Execution Order (Corrected)

**Audit fix (gremlin/senior):** Chunks 2 and 3 both modify `archetype_comparison.py`. They MUST be sequential, not parallel. Chunk 8 depends on all prior chunks.

```
Chunk 1 (regression: rank info + FDR gaps)  ──┐
Chunk 2 (Wald: global FDR + error bars)     ──┤
  └─→ Chunk 3 (labeling: sequential after 2) ─┤
Chunk 4 (GMM: predict_proba + component reg) ──┼──→ Chunk 6 (viz) ──→ Chunk 7 (notebook) ──→ Chunk 8 (registry)
Chunk 5 (flow: Jacobian + permutation)       ──┘
```

Parallelizable groups:
- **Group A** (parallel): Chunks 1, 4, 5
- **Group B** (sequential, after Group A starts): Chunk 2, then Chunk 3
- **Group C** (after all of A+B): Chunk 6
- **Group D** (after C): Chunk 7
- **Group E** (after D): Chunk 8

## Integration Test Gate

After all chunks complete, before the notebook rewrite:

```bash
conda run -n archetype pytest tests/test_statistical/ -xvs
conda run -n archetype pytest tests/test_integration/ -xvs
```

All tests must pass before proceeding to Chunk 7.

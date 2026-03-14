# v0.5.0 Methodological Revision Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden PEACH v0.5.0 statistical methods to publication-ready rigor based on redteam analysis of simplex regression, pattern classification, GMM decomposition, archetype comparison, and flow matching.

**Architecture:** Six chunks targeting specific modules. Chunks 1-2 are quick wins (pattern classification fix, covariance caching). Chunk 3 enhances GMM (ICL + Dirichlet option + pairwise NMI stability). Chunk 4 overhauls archetype comparison (weighted MMD, unbiased estimator, drop silhouette, t-distribution Wald). Chunk 5 is the largest — flow matching improvements (minibatch OT-CFM, dopri5 default, holdout validation, per-cell gene alignment, eigenvalue bifurcation). Chunk 6 updates registries.

**Tech Stack:** numpy, scipy, scikit-learn, torch, statsmodels, ot (Python Optimal Transport), flow_matching (Facebook Research)

**Dependencies between chunks:**
- Chunk 2 (cache covariance) must complete before Chunk 4 (Wald uses cached cov)
- Chunk 1 (patterns) and Chunk 3 (GMM) are independent
- Chunk 5 (flow) is fully independent
- Chunk 6 (registries) runs last after all code changes

**Conda environment:** `archetype`

**Test data:** Synthetic Dirichlet-generated weights + planted signal (see existing test fixtures). Real data tests use `~/Desktop/peach/data/hsc_10k.h5ad`.

---

## Chunk 1: Pattern Classification — Principled Simplification

**Rationale:** Current pattern classification uses arbitrary thresholds and ignores p-values entirely. Replace with three principled categories: flat (F-test not significant), exclusive (significant + dominant vertex), interaction (partial F-test degree-2 vs degree-1 significant). Everything else is "structured" with no sub-classification.

**Files:**
- Modify: `src/peach/_core/utils/pattern_classification.py` (full rewrite, ~177 → ~120 lines)
- Modify: `src/peach/tl/feature_patterns.py:11-79` (pass FDR p-values, add `fdr_threshold` param)
- Modify: `tests/test_statistical/test_pattern_classification.py` (rewrite for new categories)
- Modify: `tests/test_statistical/test_pattern_api.py` (update expected patterns)

### Task 1.1: Rewrite pattern classification engine

- [ ] **Step 1: Write failing tests for new classification logic**

```python
# tests/test_statistical/test_pattern_classification.py
import numpy as np
import pytest
from peach._core.utils.pattern_classification import classify_single_feature, classify_all_features


class TestClassifySingleFeature:
    def test_flat_nonsignificant_f_test(self):
        """Feature with f_pvalue_fdr > threshold -> flat regardless of R2."""
        result = classify_single_feature(
            vertex_betas=np.array([5.0, 0.1, 0.2]),
            r2=0.3,
            f_pvalue_fdr=0.2,  # not significant
            interaction_f_pvalue_fdr=None,
            fdr_threshold=0.05,
            exclusive_ratio=2.0,
        )
        assert result["pattern"] == "flat"
        assert result["details"]["reason"] == "nonsignificant"

    def test_exclusive_dominant_vertex(self):
        """Significant feature with dominant vertex -> exclusive."""
        result = classify_single_feature(
            vertex_betas=np.array([10.0, 1.0, 0.5]),
            r2=0.8,
            f_pvalue_fdr=0.001,
            interaction_f_pvalue_fdr=None,
            fdr_threshold=0.05,
            exclusive_ratio=2.0,
        )
        assert result["pattern"] == "archetype-exclusive"
        assert result["details"]["dominant_archetype"] == 0

    def test_interaction_significant_partial_f(self):
        """Significant degree-2 improvement -> interaction."""
        result = classify_single_feature(
            vertex_betas=np.array([3.0, 3.0, 3.0]),
            r2=0.5,
            f_pvalue_fdr=0.001,
            interaction_f_pvalue_fdr=0.01,  # degree-2 significantly better
            fdr_threshold=0.05,
            exclusive_ratio=2.0,
        )
        assert result["pattern"] == "interaction"

    def test_structured_fallback(self):
        """Significant, not exclusive, no interaction -> structured."""
        result = classify_single_feature(
            vertex_betas=np.array([5.0, 4.0, 3.0]),
            r2=0.4,
            f_pvalue_fdr=0.001,
            interaction_f_pvalue_fdr=0.3,  # degree-2 not better
            fdr_threshold=0.05,
            exclusive_ratio=2.0,
        )
        assert result["pattern"] == "structured"

    def test_exclusive_uses_absolute_magnitudes(self):
        """Negative dominant beta still counts as exclusive."""
        result = classify_single_feature(
            vertex_betas=np.array([-10.0, -1.0, 0.5]),
            r2=0.6,
            f_pvalue_fdr=0.001,
            interaction_f_pvalue_fdr=None,
            fdr_threshold=0.05,
            exclusive_ratio=2.0,
        )
        assert result["pattern"] == "archetype-exclusive"

    def test_interaction_without_degree2_data(self):
        """If interaction_f_pvalue_fdr is None, skip interaction check."""
        result = classify_single_feature(
            vertex_betas=np.array([3.0, 3.0, 3.0]),
            r2=0.3,
            f_pvalue_fdr=0.001,
            interaction_f_pvalue_fdr=None,
            fdr_threshold=0.05,
            exclusive_ratio=2.0,
        )
        assert result["pattern"] == "structured"  # not interaction


class TestClassifyAllFeatures:
    def test_batch_classification(self):
        """classify_all_features handles multiple features."""
        n = 5
        K = 3
        vertex_betas = np.array([
            [10.0, 0.1, 0.2],  # exclusive
            [3.0, 3.0, 3.0],   # flat (will be nonsig)
            [5.0, 4.0, 3.0],   # structured
            [8.0, 1.0, 0.5],   # exclusive
            [2.0, 2.0, 2.0],   # flat
        ])
        r2 = np.array([0.8, 0.01, 0.4, 0.7, 0.02])
        f_pvalue_fdr = np.array([0.001, 0.5, 0.001, 0.001, 0.8])
        interaction_f_pvalue_fdr = np.array([0.3, 0.9, 0.3, 0.3, 0.9])

        results = classify_all_features(
            vertex_betas, r2, f_pvalue_fdr, interaction_f_pvalue_fdr,
        )
        assert len(results) == 5
        assert results[0]["pattern"] == "archetype-exclusive"
        assert results[1]["pattern"] == "flat"
        assert results[4]["pattern"] == "flat"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/honkala/Desktop/PEACH_public && conda run -n archetype pytest tests/test_statistical/test_pattern_classification.py -v`
Expected: FAIL — old function signatures don't match

- [ ] **Step 3: Rewrite pattern_classification.py**

```python
# src/peach/_core/utils/pattern_classification.py
"""Feature pattern classification from simplex regression significance."""

import numpy as np


def classify_single_feature(
    vertex_betas,
    r2,
    f_pvalue_fdr,
    interaction_f_pvalue_fdr=None,
    fdr_threshold=0.05,
    exclusive_ratio=2.0,
):
    """Classify a single feature's regression pattern using statistical tests.

    Classification priority:
    1. flat: F-test FDR p-value > fdr_threshold (model not significant)
    2. archetype-exclusive: max(|beta|) / second_max(|beta|) >= exclusive_ratio
    3. interaction: partial F-test (degree-2 vs degree-1) FDR p-value < fdr_threshold
    4. structured: significant but doesn't fit above categories

    Parameters
    ----------
    vertex_betas : np.ndarray [K]
        Regression coefficients for each archetype vertex.
    r2 : float
        R-squared value from the regression.
    f_pvalue_fdr : float
        FDR-corrected F-test p-value for the overall model.
    interaction_f_pvalue_fdr : float or None
        FDR-corrected partial F-test p-value for degree-2 vs degree-1.
        None if degree-2 was not computed.
    fdr_threshold : float
        FDR threshold for significance (default 0.05).
    exclusive_ratio : float
        Minimum ratio of max(|beta|) to second_max(|beta|) for exclusive.

    Returns
    -------
    dict with keys: pattern, r2, details
    """
    # Guard NaN
    if np.isnan(r2) or np.isnan(f_pvalue_fdr):
        return {"pattern": "flat", "r2": float(r2),
                "details": {"reason": "nan_values"}}

    # Rule 1: Non-significant model -> flat
    if f_pvalue_fdr > fdr_threshold:
        return {"pattern": "flat", "r2": float(r2),
                "details": {"reason": "nonsignificant"}}

    # Rule 2: Exclusive — dominant vertex by absolute magnitude
    sorted_abs = np.sort(np.abs(vertex_betas))[::-1]
    if sorted_abs[0] > 0 and sorted_abs[0] / max(sorted_abs[1], 1e-10) >= exclusive_ratio:
        return {
            "pattern": "archetype-exclusive",
            "r2": float(r2),
            "details": {"dominant_archetype": int(np.argmax(np.abs(vertex_betas)))},
        }

    # Rule 3: Interaction — degree-2 significantly improves over degree-1
    if interaction_f_pvalue_fdr is not None and interaction_f_pvalue_fdr < fdr_threshold:
        return {
            "pattern": "interaction",
            "r2": float(r2),
            "details": {"interaction_fdr": float(interaction_f_pvalue_fdr)},
        }

    # Rule 4: Structured — significant but no dominant pattern
    return {
        "pattern": "structured",
        "r2": float(r2),
        "details": {"dominant_archetype": int(np.argmax(vertex_betas))},
    }


def classify_all_features(
    vertex_coefficients,
    r_squared,
    f_pvalue_fdr,
    interaction_f_pvalue_fdr=None,
    fdr_threshold=0.05,
    exclusive_ratio=2.0,
):
    """Classify all features at once.

    Parameters
    ----------
    vertex_coefficients : np.ndarray [n_features, K]
    r_squared : np.ndarray [n_features]
    f_pvalue_fdr : np.ndarray [n_features]
        FDR-corrected F-test p-values.
    interaction_f_pvalue_fdr : np.ndarray [n_features] or None
        FDR-corrected partial F-test p-values (degree-2 vs degree-1).
    fdr_threshold : float
    exclusive_ratio : float

    Returns
    -------
    list[dict]
    """
    n_features = len(r_squared)
    results = []
    for i in range(n_features):
        int_fdr = (
            float(interaction_f_pvalue_fdr[i])
            if interaction_f_pvalue_fdr is not None
            else None
        )
        results.append(
            classify_single_feature(
                vertex_coefficients[i],
                r_squared[i],
                float(f_pvalue_fdr[i]),
                int_fdr,
                fdr_threshold=fdr_threshold,
                exclusive_ratio=exclusive_ratio,
            )
        )
    return results
```

- [ ] **Step 4: Update feature_patterns.py to pass FDR p-values**

In `src/peach/tl/feature_patterns.py`, update `classify_feature_patterns` to:
- Add `fdr_threshold: float = 0.05` parameter
- Compute partial F-test p-values for interaction detection (degree-2 vs degree-1)
- Pass `f_pvalue_fdr` and `interaction_f_pvalue_fdr` to `classify_all_features`
- Remove `r2_threshold` and `cv_threshold` parameters (no longer used)

The partial F-test FDR values come from `regression_result`. If `regression_result` has a `degree_comparison` key (from `comprehensive_degree=True`), use `degree_comparison["degree_2"]["incremental_p_fdr"]`. Otherwise, compute it inline by comparing degree-1 and degree-2 R² values using the incremental F-test formula already in `_comprehensive_degree_comparison`.

Key changes to `classify_feature_patterns`:
```python
def classify_feature_patterns(
    adata: AnnData,
    *,
    regression_result=None,
    fdr_threshold: float = 0.05,
    exclusive_ratio: float = 2.0,
) -> dict:
```

- Pass `regression_result.f_pvalue_fdr` as the significance gate
- Compute interaction significance from degree-2 vs degree-1 comparison if `regression_result.r_squared_degree2` is not None
- Call `classify_all_features(vertex_coefficients, r_squared, f_pvalue_fdr, interaction_f_pvalue_fdr)`

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/honkala/Desktop/PEACH_public && conda run -n archetype pytest tests/test_statistical/test_pattern_classification.py tests/test_statistical/test_pattern_api.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/peach/_core/utils/pattern_classification.py src/peach/tl/feature_patterns.py tests/test_statistical/test_pattern_classification.py tests/test_statistical/test_pattern_api.py
git commit -m "refactor: pattern classification uses FDR significance instead of arbitrary thresholds"
```

---

## Chunk 2: Simplex Regression — Cache Covariance for Wald Reuse

**Rationale:** Wald contrasts currently re-run degree-1 regression with `return_covariance=True`. Instead, cache the covariance from the initial regression so Wald contrasts can reuse it without redundant computation.

**Files:**
- Modify: `src/peach/tl/feature_regression.py:82-192` (add `return_covariance=True` to initial ols_fit, store in result)
- Modify: `src/peach/_core/utils/archetype_comparison.py:227-320` (use cached covariance if available)
- Modify: `src/peach/_core/types.py` (add `vertex_covariance` field to `SimplexRegressionResult`)
- Test: `tests/test_statistical/test_simplex_regression_api.py` (verify covariance stored)

### Task 2.1: Cache covariance in simplex regression

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_statistical/test_simplex_regression_api.py
def test_covariance_stored_by_default(self, regression_adata):
    """Regression result should include vertex_covariance for Wald reuse."""
    import peach as pc
    result = pc.tl.feature_simplex_regression(regression_adata, n_bootstrap=0)
    assert "vertex_covariance" in result
    assert result["vertex_covariance"] is not None
    # Should be a list of [K, K] matrices, one per feature
    cov_list = result["vertex_covariance"]
    assert len(cov_list) == 50  # n_features
    assert np.array(cov_list[0]).shape == (3, 3)  # K x K
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/honkala/Desktop/PEACH_public && conda run -n archetype pytest tests/test_statistical/test_simplex_regression_api.py::TestFeatureSimplexRegression::test_covariance_stored_by_default -v`
Expected: FAIL — `vertex_covariance` not in result

- [ ] **Step 3: Add vertex_covariance to SimplexRegressionResult**

In `src/peach/_core/types.py`, add to `SimplexRegressionResult`:
```python
vertex_covariance: list | None = None  # list of [K, K] covariance matrices per feature
```

- [ ] **Step 4: Pass return_covariance=True in feature_simplex_regression**

In `src/peach/tl/feature_regression.py`, change the degree-1 `ols_fit` call:
```python
result1 = ols_fit(W1, Y, robust_se=robust_se, return_covariance=True)
```

Then include it in the `SimplexRegressionResult` constructor:
```python
vertex_covariance=result1.get("covariance"),
```

And ensure `to_serializable()` handles the list of arrays (convert each np.ndarray to list).

- [ ] **Step 5: Update Wald contrasts to use cached covariance**

In `src/peach/_core/utils/archetype_comparison.py`, modify `compute_wald_contrasts`:
- Check if `reg` dict has `vertex_covariance`
- If present, use it directly instead of re-running `ols_fit`
- If not present (backward compat), fall back to re-running regression
- Switch from z-scores (normal) to t-statistics with `df = n_cells - K`

Key code change in `compute_wald_contrasts`:
```python
# Use cached covariance if available, else re-run
if reg.get("vertex_covariance") is not None:
    beta = np.asarray(reg["vertex_coefficients"])
    cov_list = [np.asarray(c) for c in reg["vertex_covariance"]]
else:
    # Fallback: re-run regression (backward compat)
    Y, _ = resolve_features(adata, None, feat_names)
    W, _ = scheffe_design_matrix(weights, degree=1)
    fit = ols_fit(W, Y, robust_se=robust_se, return_covariance=True)
    beta = fit["coefficients"]
    cov_list = fit["covariance"]

# Use t-distribution instead of normal
n_cells = adata.n_obs
df_contrast = max(n_cells - K, 1)
# ...
pval = 2 * stats.t.sf(np.abs(z), df=df_contrast)  # was: stats.norm.sf
```

- [ ] **Step 6: Run full test suite for regression + comparison**

Run: `cd /Users/honkala/Desktop/PEACH_public && conda run -n archetype pytest tests/test_statistical/test_simplex_regression_api.py tests/test_statistical/test_archetype_comparison.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/peach/_core/types.py src/peach/tl/feature_regression.py src/peach/_core/utils/archetype_comparison.py tests/test_statistical/test_simplex_regression_api.py
git commit -m "feat: cache regression covariance for Wald reuse, switch Wald to t-distribution"
```

---

## Chunk 3: GMM Decomposition — ICL, Dirichlet, Pairwise NMI

**Rationale:** Three enhancements: (1) ICL alongside BIC for model selection (better for overlapping clusters), (2) Dirichlet mixture option as the theoretically correct simplex model, (3) pairwise NMI stability instead of first-init reference bias.

**Files:**
- Modify: `src/peach/_core/utils/simplex_gmm.py` (ICL, Dirichlet EM, pairwise NMI)
- Modify: `src/peach/tl/feature_decomposition.py` (expose `model_type` and `model_selection` params)
- Create: `src/peach/_core/utils/dirichlet_mixture.py` (~150 lines, EM for Dirichlet mixture)
- Test: `tests/test_statistical/test_gmm_api.py` (new tests for ICL, Dirichlet, NMI stability)

### Task 3.1: Add ICL model selection

- [ ] **Step 1: Write failing test for ICL**

```python
# Add to tests/test_statistical/test_gmm_api.py
def test_icl_model_selection(self, gmm_adata):
    """model_selection='icl' should work and may select fewer components."""
    import peach as pc
    result = pc.tl.feature_simplex_decomposition(
        gmm_adata, model_selection="icl",
        n_initializations=3, n_components_range=(2, 6),
    )
    assert result["n_components_optimal"] >= 2
    assert "icl_values" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/honkala/Desktop/PEACH_public && conda run -n archetype pytest tests/test_statistical/test_gmm_api.py::TestFeatureSimplexDecomposition::test_icl_model_selection -v`
Expected: FAIL

- [ ] **Step 3: Implement ICL in fit_simplex_gmm**

In `src/peach/_core/utils/simplex_gmm.py`, add ICL computation after BIC:

```python
def _compute_icl(gmm, X):
    """ICL = BIC + 2 * entropy(posterior probabilities)."""
    bic = gmm.bic(X)
    proba = gmm.predict_proba(X)
    # Entropy of assignment: -sum(p * log(p)), clamp for log(0)
    entropy = -np.sum(proba * np.log(np.clip(proba, 1e-300, 1.0)))
    return bic + 2 * entropy
```

Add `model_selection: str = "bic"` parameter to `fit_simplex_gmm`. In the BIC scan loop, compute both BIC and ICL, select by the chosen criterion. Return both `bic_values` and `icl_values` in the result dict.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/honkala/Desktop/PEACH_public && conda run -n archetype pytest tests/test_statistical/test_gmm_api.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/peach/_core/utils/simplex_gmm.py src/peach/tl/feature_decomposition.py tests/test_statistical/test_gmm_api.py
git commit -m "feat: add ICL model selection for GMM decomposition"
```

### Task 3.2: Replace first-init stability with pairwise NMI

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_statistical/test_gmm_api.py
def test_pairwise_nmi_stability(self, gmm_adata):
    """Stability analysis should use pairwise NMI, not first-init reference."""
    import peach as pc
    result = pc.tl.feature_simplex_decomposition(
        gmm_adata, n_initializations=5, n_components_range=(2, 4),
    )
    # Stability scores should still be in [0, 1]
    scores = result["component_stability_scores"]
    assert np.all(scores >= 0) and np.all(scores <= 1)
    # For well-separated clusters, stability should be high
    assert np.mean(scores) > 0.5
```

- [ ] **Step 2: Rewrite _compute_stability to use pairwise NMI**

Replace the first-init reference approach with:
1. Fit GMM `n_initializations` times with different seeds
2. Compute pairwise NMI between all `n_init choose 2` label vectors
3. For each run, use Hungarian matching to align components to a consensus
4. Per-component stability = mean recovery rate across all pairwise comparisons

```python
def _compute_stability(ilr_coords, n_components, covariance_type, n_initializations, random_state):
    from sklearn.metrics import normalized_mutual_info_score

    rng = np.random.default_rng(random_state)
    all_labels = []
    for i in range(n_initializations):
        gmm = GaussianMixture(
            n_components=n_components, covariance_type=covariance_type,
            n_init=1, random_state=int(rng.integers(0, 2**31)),
        )
        gmm.fit(ilr_coords)
        all_labels.append(gmm.predict(ilr_coords))

    # Pairwise NMI + per-component recovery via Hungarian matching
    n_runs = len(all_labels)
    component_recovery = np.zeros(n_components)
    n_pairs = 0

    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            confusion = np.zeros((n_components, n_components))
            for r in range(n_components):
                for t in range(n_components):
                    confusion[r, t] = np.sum(
                        (all_labels[i] == r) & (all_labels[j] == t)
                    )
            row_ind, col_ind = linear_sum_assignment(-confusion)
            for c in range(n_components):
                count_c = np.sum(all_labels[i] == c)
                if count_c == 0:
                    continue
                matched = col_ind[c]
                recovered = np.sum(
                    (all_labels[i] == c) & (all_labels[j] == matched)
                )
                component_recovery[c] += recovered / count_c
            n_pairs += 1

    return component_recovery / max(n_pairs, 1)
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/honkala/Desktop/PEACH_public && conda run -n archetype pytest tests/test_statistical/test_gmm_api.py -v`

- [ ] **Step 4: Commit**

```bash
git add src/peach/_core/utils/simplex_gmm.py tests/test_statistical/test_gmm_api.py
git commit -m "refactor: pairwise NMI stability replaces first-init reference in GMM"
```

### Task 3.3: Add Dirichlet mixture option

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_statistical/test_gmm_api.py
def test_dirichlet_mixture(self, gmm_adata):
    """model_type='dirichlet' should fit a Dirichlet mixture on raw weights."""
    import peach as pc
    result = pc.tl.feature_simplex_decomposition(
        gmm_adata, model_type="dirichlet",
        n_initializations=3, n_components_range=(2, 4),
    )
    assert result["n_components_optimal"] >= 2
    assert result["model_type"] == "dirichlet"
    # Centroids should be valid simplex points
    means = result["component_simplex_means"]
    np.testing.assert_allclose(means.sum(axis=1), 1.0, atol=1e-6)
```

- [ ] **Step 2: Implement Dirichlet mixture EM**

Create `src/peach/_core/utils/dirichlet_mixture.py`:

```python
"""Dirichlet Mixture Model via EM.

Fits a mixture of Dirichlet distributions directly on the simplex,
avoiding the ILR approximation. Each component k has concentration
parameters alpha_k [K], and mixing proportions pi [n_components].

Uses fixed-point iteration for Dirichlet MLE (Minka 2000).

Reference: Minka, T. (2000). "Estimating a Dirichlet distribution."
"""
import numpy as np
from scipy.special import digamma, polygamma, gammaln


class DirichletMixture:
    def __init__(self, n_components, max_iter=100, tol=1e-4, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.alphas_ = None      # [n_components, K]
        self.weights_ = None     # [n_components]
        self.converged_ = False

    def fit(self, W):
        """Fit Dirichlet mixture to simplex data W [n, K]."""
        rng = np.random.default_rng(self.random_state)
        n, K = W.shape
        nc = self.n_components

        # Initialize: k-means on log(W + eps)
        from sklearn.cluster import KMeans
        log_W = np.log(np.clip(W, 1e-10, None))
        km = KMeans(n_clusters=nc, n_init=3, random_state=self.random_state)
        init_labels = km.fit_predict(log_W)

        # Initialize alphas from cluster means via moment matching
        self.alphas_ = np.ones((nc, K)) * 2.0
        self.weights_ = np.ones(nc) / nc
        for k in range(nc):
            mask = init_labels == k
            if mask.sum() > 1:
                mean_k = W[mask].mean(axis=0)
                var_k = W[mask].var(axis=0).mean()
                s = max((mean_k[0] * (1 - mean_k[0]) / max(var_k, 1e-10) - 1), 1.0)
                self.alphas_[k] = mean_k * s
                self.weights_[k] = mask.sum() / n

        # EM iterations
        log_W_safe = np.log(np.clip(W, 1e-300, None))  # [n, K]
        prev_ll = -np.inf
        for iteration in range(self.max_iter):
            # E-step: compute responsibilities
            log_resp = np.zeros((n, nc))
            for k in range(nc):
                log_resp[:, k] = (
                    np.log(max(self.weights_[k], 1e-300))
                    + self._log_dirichlet_pdf(log_W_safe, self.alphas_[k])
                )
            # Log-sum-exp normalization
            max_log = log_resp.max(axis=1, keepdims=True)
            log_resp -= max_log
            resp = np.exp(log_resp)
            resp /= resp.sum(axis=1, keepdims=True)

            # Log-likelihood
            ll = np.sum(max_log.ravel() + np.log(resp.sum(axis=1)))

            # M-step: update weights and alphas
            Nk = resp.sum(axis=0)  # [nc]
            self.weights_ = Nk / n

            for k in range(nc):
                if Nk[k] < 1e-10:
                    continue
                # Weighted sufficient statistics
                weighted_log_mean = (resp[:, k:k+1] * log_W_safe).sum(axis=0) / Nk[k]
                # Fixed-point Dirichlet MLE (Minka 2000)
                self.alphas_[k] = self._dirichlet_mle_fixedpoint(
                    self.alphas_[k], weighted_log_mean
                )

            if abs(ll - prev_ll) < self.tol:
                self.converged_ = True
                break
            prev_ll = ll

        return self

    def predict(self, W):
        """Predict component assignments."""
        resp = self.predict_proba(W)
        return np.argmax(resp, axis=1)

    def predict_proba(self, W):
        """Compute posterior probabilities."""
        log_W_safe = np.log(np.clip(W, 1e-300, None))
        n = len(W)
        log_resp = np.zeros((n, self.n_components))
        for k in range(self.n_components):
            log_resp[:, k] = (
                np.log(max(self.weights_[k], 1e-300))
                + self._log_dirichlet_pdf(log_W_safe, self.alphas_[k])
            )
        max_log = log_resp.max(axis=1, keepdims=True)
        log_resp -= max_log
        resp = np.exp(log_resp)
        resp /= resp.sum(axis=1, keepdims=True)
        return resp

    def bic(self, W):
        """Bayesian Information Criterion."""
        n, K = W.shape
        n_params = self.n_components * K + self.n_components - 1
        ll = self._log_likelihood(W)
        return -2 * ll + n_params * np.log(n)

    def _log_likelihood(self, W):
        log_W_safe = np.log(np.clip(W, 1e-300, None))
        n = len(W)
        log_resp = np.zeros((n, self.n_components))
        for k in range(self.n_components):
            log_resp[:, k] = (
                np.log(max(self.weights_[k], 1e-300))
                + self._log_dirichlet_pdf(log_W_safe, self.alphas_[k])
            )
        max_log = log_resp.max(axis=1, keepdims=True)
        return np.sum(max_log.ravel() + np.log(np.exp(log_resp - max_log).sum(axis=1)))

    @staticmethod
    def _log_dirichlet_pdf(log_W, alpha):
        """Log Dirichlet PDF: log Dir(w|alpha) for each row of log_W."""
        # log B(alpha) = sum(gammaln(alpha)) - gammaln(sum(alpha))
        log_B = np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
        # sum((alpha_k - 1) * log(w_k))
        return np.sum((alpha - 1) * log_W, axis=1) - log_B

    @staticmethod
    def _dirichlet_mle_fixedpoint(alpha, mean_log_x, max_iter=50):
        """Minka's fixed-point iteration for Dirichlet MLE."""
        alpha = alpha.copy()
        for _ in range(max_iter):
            alpha_sum = alpha.sum()
            alpha_new = alpha.copy()
            for j in range(len(alpha)):
                alpha_new[j] = alpha[j] * (
                    digamma(alpha_sum) + mean_log_x[j] - digamma(alpha[j])
                ) / digamma(alpha_sum)
                alpha_new[j] = max(alpha_new[j], 1e-6)  # prevent collapse
            if np.max(np.abs(alpha_new - alpha)) < 1e-6:
                break
            alpha = alpha_new
        return alpha
```

- [ ] **Step 3: Integrate Dirichlet option into fit_simplex_gmm**

Add `model_type: str = "gaussian"` parameter to `fit_simplex_gmm`. When `model_type == "dirichlet"`:
- Skip ILR transform (work on raw weights)
- Use `DirichletMixture` instead of `GaussianMixture`
- Stability analysis uses `DirichletMixture` fits
- Centroids are the Dirichlet means: `alpha_k / sum(alpha_k)` (already on the simplex)
- Include `model_type` in the return dict

- [ ] **Step 4: Expose model_type in tl API**

In `src/peach/tl/feature_decomposition.py`, add `model_type: str = "gaussian"` parameter and pass through to `fit_simplex_gmm`.

- [ ] **Step 5: Run all GMM tests**

Run: `cd /Users/honkala/Desktop/PEACH_public && conda run -n archetype pytest tests/test_statistical/test_gmm_api.py -v`

- [ ] **Step 6: Commit**

```bash
git add src/peach/_core/utils/dirichlet_mixture.py src/peach/_core/utils/simplex_gmm.py src/peach/tl/feature_decomposition.py tests/test_statistical/test_gmm_api.py
git commit -m "feat: add Dirichlet mixture option and pairwise NMI stability for GMM decomposition"
```

---

## Chunk 4: Archetype Comparison — Weighted MMD, Unbiased Estimator, Drop Silhouette

**Rationale:** MMD currently uses hard argmax assignments (loses continuous weight info) and biased estimator. Comparison should use weighted kernel expectations on full weight vectors, unbiased MMD, Spearman with FDR pre-filter (no silhouette), and cached covariance for Wald.

**Depends on:** Chunk 2 (cached covariance)

**Files:**
- Modify: `src/peach/_core/utils/archetype_comparison.py` (weighted MMD, drop silhouette, use cached cov)
- Modify: `src/peach/_core/utils/flow_matching.py:339-393` (unbiased MMD estimator)
- Modify: `src/peach/tl/comparison.py` (update API, remove silhouette from result)
- Modify: `src/peach/_core/types.py` (remove silhouette fields from ArchetypeFeatureSimilarityResult)
- Test: `tests/test_statistical/test_archetype_comparison.py`

### Task 4.1: Fix MMD to use unbiased estimator

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_statistical/test_archetype_comparison.py
def test_mmd_unbiased_identical_distributions():
    """MMD of a distribution with itself should be near zero (unbiased)."""
    from peach._core.utils.flow_matching import compute_mmd
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 5))
    mmd = compute_mmd(X, X)
    # Unbiased MMD of identical distributions should be near 0
    # Biased estimator gives positive value due to diagonal K(x,x)=1
    assert abs(mmd) < 0.05, f"MMD of identical distribution should be ~0, got {mmd}"
```

- [ ] **Step 2: Fix compute_mmd to exclude diagonal**

In `src/peach/_core/utils/flow_matching.py`, update `compute_mmd`:
```python
# Unbiased estimator: exclude diagonal (K(x_i, x_i) = 1)
n = len(X)
m = len(Y)
mmd2 = (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1)) \
     + (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1)) \
     - 2 * K_XY.sum() / (n * m)
```

- [ ] **Step 3: Run test**

Run: `cd /Users/honkala/Desktop/PEACH_public && conda run -n archetype pytest tests/test_statistical/test_archetype_comparison.py::test_mmd_unbiased_identical_distributions -v`

- [ ] **Step 4: Commit**

```bash
git add src/peach/_core/utils/flow_matching.py tests/test_statistical/test_archetype_comparison.py
git commit -m "fix: use unbiased MMD estimator (exclude kernel diagonal)"
```

### Task 4.2: Implement weighted MMD on full weight vectors

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_statistical/test_archetype_comparison.py
def test_weighted_mmd_uses_full_weights():
    """Weighted MMD should use continuous weights, not hard assignments."""
    import peach as pc
    rng = np.random.default_rng(42)
    n = 300
    K = 3

    # Create adata with weights and PCA
    weights = rng.dirichlet([1]*K, size=n)
    pca = rng.standard_normal((n, 10))
    adata = AnnData(rng.standard_normal((n, 20)))
    adata.obsm["cell_archetype_weights"] = weights
    adata.obsm["X_pca"] = pca

    result = pc.tl.archetype_mmd(adata, n_permutations=50)
    mmd_matrix = np.asarray(result["mmd_matrix"])
    assert mmd_matrix.shape == (K, K)
    # Diagonal should be near zero (same weighted population)
    for i in range(K):
        assert mmd_matrix[i, i] < 0.1
```

- [ ] **Step 2: Rewrite compute_archetype_mmd to use weighted kernels**

In `src/peach/_core/utils/archetype_comparison.py`, replace hard assignment MMD with weighted MMD:

```python
def compute_archetype_mmd(adata, adata_b=None, *, pca_key="X_pca",
                          n_permutations=1000, seed=42):
    """Weighted MMD using full archetype weight vectors."""
    from .flow_matching import _rbf_kernel  # extract helper

    rng = np.random.default_rng(seed)
    weights_a = get_archetype_weights(adata)
    pca_a = adata.obsm[pca_key]
    K_a = weights_a.shape[1]

    if adata_b is not None:
        weights_b = get_archetype_weights(adata_b)
        pca_b = adata_b.obsm[pca_key]
        K_b = weights_b.shape[1]
    else:
        weights_b = weights_a
        pca_b = pca_a
        K_b = K_a

    # Subsample for kernel computation
    max_n = 5000
    if len(pca_a) > max_n:
        idx = rng.choice(len(pca_a), max_n, replace=False)
        pca_a = pca_a[idx]; weights_a = weights_a[idx]
    if adata_b is not None and len(pca_b) > max_n:
        idx = rng.choice(len(pca_b), max_n, replace=False)
        pca_b = pca_b[idx]; weights_b = weights_b[idx]

    # Compute kernel matrix once
    from scipy.spatial.distance import cdist, pdist
    bandwidth = np.median(pdist(pca_a[:min(500, len(pca_a))]))
    bandwidth = max(bandwidth, 1e-6)

    def _weighted_mmd(pca_x, w_x_i, pca_y, w_y_i, bw):
        """Weighted MMD for archetype i between populations x and y."""
        K_xx = np.exp(-cdist(pca_x, pca_x, 'sqeuclidean') / (2 * bw**2))
        K_yy = np.exp(-cdist(pca_y, pca_y, 'sqeuclidean') / (2 * bw**2))
        K_xy = np.exp(-cdist(pca_x, pca_y, 'sqeuclidean') / (2 * bw**2))

        # Weighted expectations (exclude diagonal for unbiased)
        wx = w_x_i / w_x_i.sum()
        wy = w_y_i / w_y_i.sum()

        W_xx = np.outer(wx, wx)
        np.fill_diagonal(W_xx, 0)
        W_yy = np.outer(wy, wy)
        np.fill_diagonal(W_yy, 0)
        W_xy = np.outer(wx, wy)

        mmd2 = (W_xx * K_xx).sum() / (1 - np.sum(wx**2)) \
             + (W_yy * K_yy).sum() / (1 - np.sum(wy**2)) \
             - 2 * (W_xy * K_xy).sum()
        return float(mmd2)

    mmd_matrix = np.zeros((K_a, K_b))
    pvalue_matrix = np.ones((K_a, K_b))

    for i in range(K_a):
        for j in range(K_b):
            if adata_b is None and j <= i:
                if j < i:
                    mmd_matrix[i, j] = mmd_matrix[j, i]
                    pvalue_matrix[i, j] = pvalue_matrix[j, i]
                continue

            w_a_i = weights_a[:, i]
            w_b_j = weights_b[:, j]

            observed = _weighted_mmd(pca_a, w_a_i, pca_b, w_b_j, bandwidth)
            mmd_matrix[i, j] = observed

            if n_permutations > 0:
                # Permute by shuffling weight assignments
                combined_pca = np.vstack([pca_a, pca_b]) if adata_b is not None else pca_a
                combined_w_i = np.concatenate([w_a_i, w_b_j]) if adata_b is not None else w_a_i
                n_a = len(w_a_i)
                null_mmds = np.empty(n_permutations)
                for p in range(n_permutations):
                    perm = rng.permutation(len(combined_pca))
                    perm_pca_a = combined_pca[perm[:n_a]]
                    perm_w_a = combined_w_i[perm[:n_a]]
                    perm_pca_b = combined_pca[perm[n_a:]]
                    perm_w_b = combined_w_i[perm[n_a:]]
                    null_mmds[p] = _weighted_mmd(
                        perm_pca_a, perm_w_a, perm_pca_b, perm_w_b, bandwidth
                    )
                pvalue_matrix[i, j] = (np.sum(null_mmds >= observed) + 1) / (n_permutations + 1)

    if adata_b is None:
        mmd_matrix = np.maximum(mmd_matrix, mmd_matrix.T)
        for i in range(K_a):
            for j in range(i):
                pvalue_matrix[i, j] = pvalue_matrix[j, i]

    return mmd_matrix, pvalue_matrix
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/honkala/Desktop/PEACH_public && conda run -n archetype pytest tests/test_statistical/test_archetype_comparison.py -v`

- [ ] **Step 4: Commit**

```bash
git add src/peach/_core/utils/archetype_comparison.py tests/test_statistical/test_archetype_comparison.py
git commit -m "feat: weighted MMD using full archetype weight vectors"
```

### Task 4.3: Drop silhouette, add FDR pre-filter to Spearman

- [ ] **Step 1: Update compute_feature_similarity**

In `src/peach/_core/utils/archetype_comparison.py`, modify `compute_feature_similarity`:
- Remove silhouette computation entirely
- Add FDR pre-filter: only include features where at least one vertex has `vertex_pvalue_fdr < 0.05`
- Return `n_significant_features` count

- [ ] **Step 2: Update ArchetypeFeatureSimilarityResult type**

In `src/peach/_core/types.py`, remove `silhouette_per_archetype` and `silhouette_overall` fields. Add `n_significant_features: int`.

- [ ] **Step 3: Update tl/comparison.py API**

Remove `pca_key` parameter from `archetype_feature_similarity` (no longer needed without silhouette).

- [ ] **Step 4: Update tests**

- [ ] **Step 5: Run tests**

Run: `cd /Users/honkala/Desktop/PEACH_public && conda run -n archetype pytest tests/test_statistical/test_archetype_comparison.py -v`

- [ ] **Step 6: Commit**

```bash
git add src/peach/_core/utils/archetype_comparison.py src/peach/_core/types.py src/peach/tl/comparison.py tests/test_statistical/test_archetype_comparison.py
git commit -m "refactor: drop silhouette from feature similarity, add FDR pre-filter to Spearman"
```

---

## Chunk 5: Flow Matching — OT-CFM, dopri5, Holdout, Gene Alignment, Bifurcation

**Rationale:** Five improvements: (1) minibatch OT-CFM for better transport quality, (2) dopri5 default solver, (3) holdout validation for transport quality scoring, (4) per-cell per-gene alignment via PCA loading correlation, (5) eigenvalue-based bifurcation detection.

**Files:**
- Modify: `src/peach/_core/utils/flow_matching.py` (OT-CFM training, dopri5 default, holdout, bifurcation)
- Modify: `src/peach/tl/flow.py` (holdout param, per-cell alignment, bifurcation API)
- Test: `tests/test_statistical/test_flow_jacobian.py` (holdout, bifurcation, alignment tests)

**New dependency:** `ot` (Python Optimal Transport) — add to environment.yml

### Task 5.1: Switch default solver to dopri5

- [ ] **Step 1: Change default in FlowModel and flow_within**

In `src/peach/_core/utils/flow_matching.py`:
```python
class FlowModel:
    def __init__(self, dim, hidden_dims=(128, 128, 128), lr=1e-3,
                 solver_method="dopri5", device="cpu"):  # was "euler"
```

In `src/peach/tl/flow.py`, update `flow_within` and `flow_between` defaults:
```python
solver_method: str = "dopri5",  # was "euler"
```

- [ ] **Step 2: Verify existing Jacobian test still passes**

Run: `cd /Users/honkala/Desktop/PEACH_public && conda run -n archetype pytest tests/test_statistical/test_flow_jacobian.py -v`

- [ ] **Step 3: Commit**

```bash
git add src/peach/_core/utils/flow_matching.py src/peach/tl/flow.py
git commit -m "chore: switch default ODE solver from euler to dopri5"
```

### Task 5.2: Add minibatch OT-CFM training

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_statistical/test_flow_jacobian.py
def test_ot_cfm_training():
    """Training with use_ot=True should use Sinkhorn-coupled minibatches."""
    from peach._core.utils.flow_matching import FlowModel
    source = np.random.randn(100, 5).astype(np.float32)
    target = np.random.randn(100, 5).astype(np.float32)
    model = FlowModel(dim=5, hidden_dims=(32, 32))
    losses = model.train(source, target, n_epochs=50, batch_size=64, use_ot=True)
    assert len(losses) == 50
    assert losses[-1] < losses[0]  # should improve
```

- [ ] **Step 2: Add Sinkhorn minibatch coupling to FlowModel.train**

In `src/peach/_core/utils/flow_matching.py`, add `use_ot: bool = False` parameter to `FlowModel.train`:

```python
def train(self, source, target, n_epochs=1000, batch_size=256, use_ot=False):
    from flow_matching.path import CondOTProbPath
    prob_path = CondOTProbPath()

    # ... existing setup ...

    for epoch in range(n_epochs):
        n = min(len(source_t), len(target_t), batch_size)
        idx_s = torch.randint(0, len(source_t), (n,), device=self.device)
        idx_t = torch.randint(0, len(target_t), (n,), device=self.device)
        x0 = source_t[idx_s]
        x1 = target_t[idx_t]

        if use_ot:
            # Minibatch OT coupling via Sinkhorn (Tong et al. 2023)
            import ot as pot
            cost = torch.cdist(x0, x1).detach().cpu().numpy()
            # Sinkhorn with entropic regularization
            coupling = pot.sinkhorn(
                np.ones(n) / n, np.ones(n) / n, cost, reg=0.1
            )
            # Sample pairs from coupling
            coupling_flat = coupling.ravel()
            coupling_flat /= coupling_flat.sum()
            pair_idx = np.random.choice(n * n, size=n, p=coupling_flat)
            idx_i = pair_idx // n
            idx_j = pair_idx % n
            x0 = x0[idx_i]
            x1 = x1[idx_j]

        t = torch.rand(len(x0), device=self.device)
        path_sample = prob_path.sample(x_0=x0, x_1=x1, t=t)
        v_pred = self.velocity_net(path_sample.x_t, t)
        loss = torch.mean((v_pred - path_sample.dx_t) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._losses.append(loss.item())

    self.velocity_net.eval()
    return self._losses
```

- [ ] **Step 3: Expose use_ot in tl/flow.py**

Add `use_ot: bool = False` to `flow_within` and `flow_between`, pass to `model.train()`.

- [ ] **Step 4: Run tests**

Run: `cd /Users/honkala/Desktop/PEACH_public && conda run -n archetype pytest tests/test_statistical/test_flow_jacobian.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/peach/_core/utils/flow_matching.py src/peach/tl/flow.py tests/test_statistical/test_flow_jacobian.py
git commit -m "feat: add minibatch OT-CFM training via Sinkhorn coupling"
```

### Task 5.3: Add holdout validation

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_statistical/test_flow_jacobian.py
def test_holdout_validation():
    """flow_within with holdout_fraction should report holdout_mmd."""
    import anndata as ad
    from peach.tl.flow import flow_within

    rng = np.random.default_rng(42)
    n = 200
    pca = rng.standard_normal((n, 5)).astype(np.float32)
    adata = ad.AnnData(rng.standard_normal((n, 10)))
    adata.obsm["X_pca"] = pca
    adata.obs["group"] = ["A"] * 100 + ["B"] * 100

    result = flow_within(
        adata, {"group": "A"}, {"group": "B"},
        n_epochs=50, batch_size=64, holdout_fraction=0.2,
    )
    assert "holdout_mmd" in result
    assert result["holdout_fraction"] == 0.2
    assert result["holdout_mmd"] >= 0
```

- [ ] **Step 2: Implement holdout in flow_within**

In `src/peach/tl/flow.py`, add `holdout_fraction: float = 0.0` parameter to `flow_within`:

```python
# After building masks, split source into train and holdout
if holdout_fraction > 0:
    rng_ho = np.random.default_rng(random_state)
    n_source = source_pca.shape[0]
    n_holdout = max(int(n_source * holdout_fraction), 1)
    perm = rng_ho.permutation(n_source)
    holdout_idx = perm[:n_holdout]
    train_idx = perm[n_holdout:]
    source_train = source_pca[train_idx]
    source_holdout = source_pca[holdout_idx]
else:
    source_train = source_pca
    source_holdout = None

# Train on source_train (not full source)
losses = model.train(source_train, target_pca, ...)

# Transport full source for result
transported = model.transport(source_pca, n_steps=n_steps)

# Holdout scoring
if source_holdout is not None:
    holdout_transported = model.transport(source_holdout, n_steps=n_steps)
    holdout_mmd = compute_mmd(holdout_transported, target_pca)
    result["holdout_mmd"] = holdout_mmd
    result["holdout_fraction"] = holdout_fraction
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/honkala/Desktop/PEACH_public && conda run -n archetype pytest tests/test_statistical/test_flow_jacobian.py::test_holdout_validation -v`

- [ ] **Step 4: Commit**

```bash
git add src/peach/tl/flow.py tests/test_statistical/test_flow_jacobian.py
git commit -m "feat: add holdout validation for flow transport quality"
```

### Task 5.4: Per-cell per-gene alignment via PCA loading correlation

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_statistical/test_flow_jacobian.py
def test_per_cell_gene_alignment():
    """flow_gene_alignment with per_cell=True returns [n_source, n_genes] scores."""
    import anndata as ad
    from peach.tl.flow import flow_gene_alignment

    rng = np.random.default_rng(42)
    n_cells = 100
    n_genes = 50
    n_pcs = 10

    X = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
    adata = ad.AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obsm["X_pca"] = rng.standard_normal((n_cells, n_pcs)).astype(np.float32)
    adata.varm["PCs"] = rng.standard_normal((n_genes, n_pcs)).astype(np.float32)

    source_mask = np.zeros(n_cells, dtype=bool)
    source_mask[:50] = True
    transported = adata.obsm["X_pca"][:50] + rng.standard_normal((50, n_pcs)).astype(np.float32) * 0.5
    flow_result = {"source_mask": source_mask, "transported": transported, "pca_key": "X_pca"}

    result = flow_gene_alignment(adata, flow_result, per_cell=True)
    assert "per_cell_alignment" in result
    assert result["per_cell_alignment"].shape == (50, n_genes)
    # Global alignment should still be present
    assert "alignment_scores" in result
    assert len(result["alignment_scores"]) == n_genes
```

- [ ] **Step 2: Add per_cell mode to flow_gene_alignment**

In `src/peach/tl/flow.py`, add `per_cell: bool = False` parameter to `flow_gene_alignment`:

```python
if per_cell:
    # Per-cell velocity vectors
    velocity_per_cell = flow_result["transported"] - source_pca  # [n_source, n_pcs]
    # Correlation between each cell's velocity and each gene's loading vector
    # Normalize both for cosine similarity
    vel_norm = velocity_per_cell / (np.linalg.norm(velocity_per_cell, axis=1, keepdims=True) + 1e-10)
    load_norm = loadings_trimmed / (np.linalg.norm(loadings_trimmed, axis=1, keepdims=True) + 1e-10)
    per_cell_alignment = vel_norm @ load_norm.T  # [n_source, n_genes]
    result["per_cell_alignment"] = per_cell_alignment
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/honkala/Desktop/PEACH_public && conda run -n archetype pytest tests/test_statistical/test_flow_jacobian.py::test_per_cell_gene_alignment -v`

- [ ] **Step 4: Commit**

```bash
git add src/peach/tl/flow.py tests/test_statistical/test_flow_jacobian.py
git commit -m "feat: per-cell per-gene alignment scores via PCA loading correlation"
```

### Task 5.5: Eigenvalue-based bifurcation scoring

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_statistical/test_flow_jacobian.py
def test_bifurcation_scoring():
    """flow_bifurcation should return divergence and eigenvalue structure."""
    from peach._core.utils.flow_matching import FlowModel
    from peach.tl.flow import flow_bifurcation

    import anndata as ad
    rng = np.random.default_rng(42)
    n = 100
    dim = 5
    source = rng.standard_normal((n, dim)).astype(np.float32)
    target = rng.standard_normal((n, dim)).astype(np.float32)

    adata = ad.AnnData(rng.standard_normal((n * 2, 10)))
    adata.obsm["X_pca"] = np.vstack([source, target])
    adata.obs["group"] = ["A"] * n + ["B"] * n

    from peach.tl.flow import flow_within
    flow_result = flow_within(
        adata, {"group": "A"}, {"group": "B"},
        n_epochs=30, batch_size=64, return_model=True,
    )

    bif = flow_bifurcation(
        adata, flow_result, flow_result["model"],
        n_timepoints=5,
    )
    assert "divergence" in bif  # trace(J) per cell
    assert "bifurcation_score" in bif  # max divergence along trajectory
    assert "eigenvalue_real" in bif  # real parts of eigenvalues
    assert len(bif["bifurcation_score"]) == n  # one per source cell
```

- [ ] **Step 2: Implement flow_bifurcation**

In `src/peach/tl/flow.py`, add:

```python
def flow_bifurcation(
    adata: AnnData,
    flow_result: dict,
    flow_model,
    *,
    n_timepoints: int = 10,
    evaluation_points: np.ndarray | None = None,
) -> dict:
    """Eigenvalue-based bifurcation detection along flow trajectories.

    Computes the Jacobian at multiple timepoints along each cell's transport
    trajectory and extracts:
    - divergence (trace of J): local volume change rate
    - eigenvalue structure: mixed signs indicate saddle/bifurcation
    - bifurcation_score: max divergence encountered along trajectory

    Parameters
    ----------
    adata : AnnData
    flow_result : dict
        From flow_within (with return_model=True).
    flow_model : FlowModel
    n_timepoints : int
        Number of time slices for Jacobian evaluation.
    evaluation_points : np.ndarray or None
        Default: source cell positions.

    Returns
    -------
    dict with divergence, bifurcation_score, eigenvalue_real, eigenvalue_imag,
         timepoints, n_saddle_points
    """
    if evaluation_points is None:
        evaluation_points = adata.obsm[flow_result["pca_key"]][flow_result["source_mask"]]

    n_cells = len(evaluation_points)
    dim = evaluation_points.shape[1]
    timepoints = np.linspace(0.05, 0.95, n_timepoints)  # avoid exact 0/1

    # Transport to get positions at each timepoint
    trajectory = flow_model.transport(
        evaluation_points, n_steps=n_timepoints - 1, return_trajectory=True
    )  # [n_timepoints, n_cells, dim]

    # Compute Jacobian at each timepoint
    divergence = np.zeros((n_timepoints, n_cells))
    eigenvalue_real = np.zeros((n_timepoints, n_cells, dim))
    eigenvalue_imag = np.zeros((n_timepoints, n_cells, dim))

    for ti, t_val in enumerate(timepoints):
        positions = trajectory[min(ti, len(trajectory)-1)]
        jac = flow_model.jacobian(positions, float(t_val))  # [n_cells, dim, dim]

        # Divergence = trace
        divergence[ti] = np.trace(jac, axis1=1, axis2=2)

        # Eigenvalues
        for ci in range(n_cells):
            eigvals = np.linalg.eigvals(jac[ci])
            eigenvalue_real[ti, ci] = eigvals.real
            eigenvalue_imag[ti, ci] = eigvals.imag

    # Bifurcation score: max absolute divergence along trajectory per cell
    bifurcation_score = np.max(np.abs(divergence), axis=0)  # [n_cells]

    # Count saddle points: timepoints where eigenvalue real parts have mixed signs
    n_saddle = np.zeros(n_cells, dtype=int)
    for ti in range(n_timepoints):
        for ci in range(n_cells):
            reals = eigenvalue_real[ti, ci]
            if np.any(reals > 0) and np.any(reals < 0):
                n_saddle[ci] += 1

    return {
        "divergence": divergence,  # [n_timepoints, n_cells]
        "bifurcation_score": bifurcation_score,  # [n_cells]
        "eigenvalue_real": eigenvalue_real,  # [n_timepoints, n_cells, dim]
        "eigenvalue_imag": eigenvalue_imag,
        "timepoints": timepoints,
        "n_saddle_points": n_saddle,  # [n_cells]
    }
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/honkala/Desktop/PEACH_public && conda run -n archetype pytest tests/test_statistical/test_flow_jacobian.py::test_bifurcation_scoring -v`

- [ ] **Step 4: Commit**

```bash
git add src/peach/tl/flow.py tests/test_statistical/test_flow_jacobian.py
git commit -m "feat: eigenvalue-based bifurcation scoring for flow trajectories"
```

---

## Chunk 6: Registry Updates

**Rationale:** After all code changes in Chunks 1-5, update types_index.py and tools_schema.py to reflect new parameters, return types, and removed fields.

**Depends on:** All previous chunks

**Files:**
- Modify: `src/peach/_core/types_index.py`
- Modify: `src/peach/_core/tools_schema.py`
- Modify: `src/peach/tl/__init__.py` (export `flow_bifurcation`)
- Modify: `src/peach/pl/comparison.py` (remove silhouette references from viz)

### Task 6.1: Update registries

- [ ] **Step 1: Update types_index.py**

Add/modify entries:
- `tl.classify_feature_patterns`: note new signature (fdr_threshold, no r2_threshold/cv_threshold)
- `tl.feature_simplex_decomposition`: add `model_type`, `model_selection` params; add `icl_values`, `model_type` to return
- `tl.archetype_mmd`: note weighted MMD (no hard assignments)
- `tl.archetype_feature_similarity`: remove silhouette fields, add `n_significant_features`
- `tl.flow_within`: add `holdout_fraction`, `use_ot` params; add `holdout_mmd` to return
- `tl.flow_gene_alignment`: add `per_cell` param; add `per_cell_alignment` to return
- `tl.flow_bifurcation`: new entry with return type
- `SimplexRegressionResult`: add `vertex_covariance`

- [ ] **Step 2: Update tools_schema.py**

Mirror the types_index changes in tool schemas.

- [ ] **Step 3: Export flow_bifurcation**

In `src/peach/tl/__init__.py`, add `flow_bifurcation` to imports and `__all__`.

- [ ] **Step 4: Update comparison visualization**

In `src/peach/pl/comparison.py`, remove any silhouette visualization functions or references. Keep Spearman heatmap, MMD heatmap, and volcano plots.

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/honkala/Desktop/PEACH_public && conda run -n archetype pytest tests/test_statistical/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/peach/_core/types_index.py src/peach/_core/tools_schema.py src/peach/tl/__init__.py src/peach/pl/comparison.py
git commit -m "chore: update registries and exports for v0.5.0 methodological revision"
```

---

## Summary

| Chunk | Description | Effort | Dependencies |
|-------|-------------|--------|-------------|
| 1 | Pattern classification: FDR-based, 3 categories | Low | None |
| 2 | Cache covariance, t-distribution Wald | Low | None |
| 3 | GMM: ICL + Dirichlet + pairwise NMI | Medium | None |
| 4 | Weighted MMD, unbiased estimator, drop silhouette | Medium | Chunk 2 |
| 5 | OT-CFM, dopri5, holdout, per-cell alignment, bifurcation | Medium-High | None |
| 6 | Registry updates | Low | All previous |

**New dependency:** `ot` (Python Optimal Transport) — `pip install POT` or `conda install -c conda-forge pot`

**Execution order:** Chunks 1, 2, 3, 5 can run in parallel. Chunk 4 after Chunk 2. Chunk 6 last.

---

## Chunk 7: Notebook Bug Fixes (from 12a/12b/12c/12c2 review)

**Rationale:** Bugs and correctness issues discovered during fine-toothed notebook review.

**Files:**
- Modify: `src/peach/pl/regression.py` (coefficient_heatmap, r2_barplot, regression_volcano, archetype_regression_dotplot)
- Modify: `src/peach/pl/comparison.py` (contrast_volcano_grid, mmd_heatmap, feature_similarity_heatmap)
- Modify: `src/peach/pl/decomposition.py` (component_neighborhood_graph)
- Modify: `src/peach/pl/flow.py` (velocity_quiver, soft_assignment_flow)
- Modify: `src/peach/_core/utils/feature_utils.py` (rename `prefer` arg)
- Modify: `src/peach/tl/flow.py` (temporal bins, flow_feature_graph)

### Task 7.1: Wald q-values all zero — investigate and fix

All Wald contrast FDR q-values showing as 0 is suspect. Likely causes:
- z-scores are enormous because HC3 SEs are tiny with large N → p-values underflow to 0 → FDR = 0
- This is partially addressed by Chunk 2 (switching to t-distribution), but may also need clamping

- [ ] **Step 1: Add diagnostic test**

```python
# tests/test_statistical/test_archetype_comparison.py
def test_wald_qvalues_not_all_zero(regression_adata):
    """Wald FDR q-values should NOT all be zero — that indicates numerical underflow."""
    import peach as pc
    pc.tl.feature_simplex_regression(regression_adata, n_bootstrap=0)
    result = pc.tl.archetype_contrasts(regression_adata)
    # At least some features should have non-zero FDR
    pairs = result["pairs"]
    all_fdr = np.concatenate([np.asarray(result["pvalues_fdr"][p]) for p in pairs])
    assert np.any(all_fdr > 0), "All Wald FDR q-values are zero — likely underflow"
    # Flat gene (gene_1) should have high FDR (not significant)
    for p in pairs:
        fdr_p = np.asarray(result["pvalues_fdr"][p])
        assert fdr_p[1] > 0.01, f"Flat gene FDR={fdr_p[1]} for pair {p}, expected >0.01"
```

- [ ] **Step 2: In compute_wald_contrasts, clamp p-values before FDR**

Verify the existing `np.clip(..., np.finfo(float).tiny, 1.0)` is working. The issue may be that `stats.norm.sf` returns actual 0.0 for very large z-scores. Add explicit clamping of raw p-values:
```python
pval = np.clip(2 * stats.t.sf(np.abs(z), df=df_contrast), np.finfo(float).tiny, 1.0)
```

- [ ] **Step 3: Run test, commit**

### Task 7.2: archetype_regression_dotplot not grouping by archetype

The dotplot should show features clustered by their dominant archetype. Currently showing feature sharing across archetypes even when `exclusive_only=True`.

- [ ] **Step 1: Fix dotplot to group features by dominant archetype**

In `src/peach/pl/regression.py:archetype_regression_dotplot`, change the feature sorting:
```python
# Sort features by dominant archetype, then by |beta| within each group
dominant_arch = np.argmax(np.abs(coefs[selected]), axis=1)
selected_with_sort = sorted(
    zip(selected, dominant_arch),
    key=lambda x: (x[1], -np.abs(coefs[x[0], x[1]]))
)
selected = [s[0] for s in selected_with_sort]
```

Add visual separators (horizontal lines) between archetype groups.

- [ ] **Step 2: Verify exclusive_only filter uses FDR-significant features only**

Pre-filter `selected` to features where `f_pvalue_fdr < 0.05` before applying the exclusive ratio check.

- [ ] **Step 3: Test and commit**

### Task 7.3: Per-component regression overwriting residuals

`simplex_regression.py:252` warns about overwriting `adata.obsm['peach_residuals']` during per-component regression. The `component_regression` function already passes `store_to_adata=False`, but residuals are stored separately.

- [ ] **Step 1: Fix residual storage in per-component regression**

In `src/peach/tl/feature_decomposition.py:component_regression`, pass `store_residuals=False` to `feature_simplex_regression`:
```python
reg = feature_simplex_regression(
    adata_sub, n_bootstrap=n_bootstrap, robust_se=robust_se,
    store_to_adata=False, store_residuals=False,
)
```

- [ ] **Step 2: Commit**

### Task 7.4: component_neighborhood_graph in PCA space instead of barycentric

- [ ] **Step 1: Modify component_neighborhood_graph to use PCA coordinates**

In `src/peach/pl/decomposition.py:component_neighborhood_graph`, change from barycentric simplex coordinates to PCA space. Use `adata.obsm['X_pca'][:, :2]` for node positions, with component centroids computed as mean PCA position of assigned cells.

- [ ] **Step 2: Commit**

### Task 7.5: Rename `prefer` argument

- [ ] **Step 1: Rename `prefer` to `feature_type` in resolve_regression_result**

In `src/peach/_core/utils/feature_utils.py:resolve_regression_result`, rename parameter from `prefer` to `feature_type` for clarity. Update all callers (grep for `prefer=`).

- [ ] **Step 2: Commit**

### Task 7.6: Jacobian determinant still zero

If Jacobian determinant is still returning 0 after the vectorized fix in the existing code, verify:
1. The `torch.func` imports are working (requires PyTorch >= 2.0)
2. The `functional_call` is correctly passing buffers
3. `self.velocity_net.eval()` with BatchNorm layers (if any) — but VelocityNetwork uses ReLU only, no BN

- [ ] **Step 1: Add explicit test that det(J) ≠ 0 after training**

```python
def test_jacobian_det_nonzero_after_training():
    from peach._core.utils.flow_matching import FlowModel
    model = FlowModel(dim=5, hidden_dims=(64, 64))
    source = np.random.randn(100, 5).astype(np.float32)
    target = source + np.random.randn(100, 5).astype(np.float32) * 2
    model.train(source, target, n_epochs=100, batch_size=64)
    jac = model.jacobian(source[:5], t=0.5)
    dets = np.linalg.det(jac)
    assert not np.allclose(dets, 0, atol=1e-6), f"Jacobian dets all ~0: {dets}"
    assert np.all(np.isfinite(dets))
```

- [ ] **Step 2: Debug if test fails — check torch version and functional_call**
- [ ] **Step 3: Commit**

---

## Chunk 8: Visualization Readability Fixes (from notebook review)

**Rationale:** Multiple plots have readability issues: cramped labels, missing axis annotations, alpha problems, missing significance filters.

**Files:**
- Modify: `src/peach/pl/regression.py`
- Modify: `src/peach/pl/comparison.py`
- Modify: `src/peach/pl/decomposition.py`
- Modify: `src/peach/pl/flow.py`

### Task 8.1: coefficient_heatmap — filter to significant features

- [ ] **Step 1: Add significance filter**

In `src/peach/pl/regression.py:coefficient_heatmap`, filter to features where `f_pvalue_fdr < 0.05` before ranking by R². Add `fdr_threshold: float = 0.05` parameter.

```python
f_fdr = np.asarray(reg.get("f_pvalue_fdr", np.zeros(len(r2))))
sig_mask = f_fdr < fdr_threshold
if sig_mask.any():
    sig_idx = np.where(sig_mask)[0]
    top_idx = sig_idx[np.argsort(r2[sig_idx])[-top_n:][::-1]]
else:
    top_idx = np.argsort(r2)[-top_n:][::-1]  # fallback if nothing significant
```

- [ ] **Step 2: Commit**

### Task 8.2: r2_barplot — fix readability

- [ ] **Step 1: Reduce to top_n=30 default, increase font size, horizontal bars**

Switch from vertical to horizontal bar chart for long gene names. Reduce default `top_n` from 50 to 30.

- [ ] **Step 2: Commit**

### Task 8.3: regression_volcano — investigate all-significant

- [ ] **Step 1: Verify volcano is using FDR-corrected p-values**

Check that `regression_volcano` uses `f_pvalue_fdr` not raw `f_pvalue` for the y-axis. If all features are significant after FDR correction, the volcano correctly shows them all above the threshold line — this may be expected for large N datasets. Add a note in the plot title: "N significant / N total".

- [ ] **Step 2: Commit**

### Task 8.4: contrast_volcano_grid — fix gene name cramming

- [ ] **Step 1: Reduce labels and improve spacing**

In `src/peach/pl/comparison.py:contrast_volcano_grid`:
- Reduce `n_labels` default from 5 to 3 per subplot
- Increase subplot spacing
- Use smaller font (size=6) for gene labels
- Add `textangle=-45` for rotated labels to prevent overlap
- Increase figure height proportional to number of subplot rows

- [ ] **Step 2: Commit**

### Task 8.5: feature_similarity — print Spearman rho alongside FDR

- [ ] **Step 1: Update feature_similarity_heatmap to show rho values**

In the feature similarity visualization, annotate the Spearman heatmap with both rho values (as text on cells) and significance stars. Currently only printing FDR without showing the actual correlation.

- [ ] **Step 2: Commit**

### Task 8.6: mmd_heatmap and feature_similarity_heatmap — axis labels for fits

- [ ] **Step 1: Add fit labels**

In `src/peach/pl/comparison.py:mmd_heatmap` and `feature_similarity_heatmap`:
- When `is_between_fit=True`, label x-axis as "Fit B archetypes" and y-axis as "Fit A archetypes"
- When within-fit, label both as "Archetypes"
- Pull fit names from result dict if available, else use "Fit A" / "Fit B"

- [ ] **Step 2: Commit**

### Task 8.7: archetype_radar_ridgeplot — fix radar, cut ridgeplot

- [ ] **Step 1: Remove ridgeplot from archetype_radar_ridgeplot**

Rename function to `archetype_radar` (drop `_ridgeplot`). Return single figure, not tuple. Remove all ridgeplot code.

- [ ] **Step 2: Fix radar plot angle ordering**

Radar spokes should be ordered by archetype similarity (e.g., by dendrogram ordering of Spearman correlation between archetype beta vectors). Currently using default ordering which doesn't reflect relationships.

- [ ] **Step 3: Update pl/__init__.py exports**
- [ ] **Step 4: Commit**

### Task 8.8: velocity_quiver — decrease arrow alpha

- [ ] **Step 1: Add alpha parameter, default 0.3**

In `src/peach/pl/flow.py:velocity_quiver`, add `arrow_alpha: float = 0.3` parameter and apply to arrow annotations:
```python
arrowcolor=f"rgba(31, 119, 180, {arrow_alpha})",  # COLOR_NEGATIVE with alpha
```

- [ ] **Step 2: Commit**

### Task 8.9: soft_assignment_flow — fix legend overlap, between-fit support

- [ ] **Step 1: Fix legend positioning**

Move legend to outside the plot area: `legend=dict(x=1.02, y=1, xanchor='left')`. Increase figure width to accommodate.

- [ ] **Step 2: Add between-fit mode**

Add `adata_b: AnnData | None = None` parameter. When provided, show fit A archetypes on left panel and fit B on right panel in a 1x2 subplot grid.

- [ ] **Step 3: Commit**

### Task 8.10: Flow graph improvements

- [ ] **Step 1: Temporal bins — change from 3 to 4**

In `src/peach/tl/flow.py:flow_temporal_feature_graph`, change phase definitions:
```python
early_mask = timepoints < 0.25
mid_early_mask = (timepoints >= 0.25) & (timepoints < 0.5)
mid_late_mask = (timepoints >= 0.5) & (timepoints < 0.75)
late_mask = timepoints >= 0.75
```
Return `top_early_genes`, `top_mid_early_genes`, `top_mid_late_genes`, `top_late_genes`.

- [ ] **Step 2: Early/late gene plots — label only top 20**

In flow graph visualization, limit gene labels to top 20 most important. Add `max_labels: int = 20` parameter.

- [ ] **Step 3: Hub genes by archetype**

In `flow_feature_graph` result, add per-archetype hub gene breakdown:
```python
hub_genes_per_archetype = {}
for k in range(K):
    arch_cells = np.argmax(weights, axis=1) == k
    # ... compute per-archetype centrality
```

- [ ] **Step 4: Commit**

---

## Chunk 9: Feature Additions from Notebook Review

**Rationale:** New capabilities identified during notebook review that enhance interpretability.

### Task 9.1: Pattern classification — archetype association readout

After classifying features, provide a structured readout of which features are associated with which archetypes.

- [ ] **Step 1: Add archetype_feature_map to classify_feature_patterns return**

```python
# In the classification result, add:
archetype_features = {}  # {archetype_idx: [feature_names]}
for i, c in enumerate(classifications):
    if c["pattern"] in ("archetype-exclusive", "structured"):
        dominant = c["details"].get("dominant_archetype")
        if dominant is not None:
            archetype_features.setdefault(dominant, []).append(feature_names[i])
result["archetype_features"] = archetype_features
```

- [ ] **Step 2: Commit**

### Task 9.2: Wrap soft_assignment_heatmap as pl function

- [ ] **Step 1: Create `pl.soft_assignment_heatmap`**

Move the inline soft assignment heatmap code from notebooks into `src/peach/pl/flow.py` as a proper function:
```python
def soft_assignment_heatmap(
    adata: AnnData,
    flow_result: dict,
    *,
    adata_b: AnnData | None = None,
    ...
) -> go.Figure:
```

- [ ] **Step 2: Commit**

### Task 9.3: Static flow graph — use igraph for adjacency

- [ ] **Step 1: Build igraph graph from adjacency matrix**

In `flow_feature_graph`, add optional igraph construction:
```python
try:
    import igraph as ig
    g = ig.Graph.Weighted_Adjacency(
        np.abs(G_sparse).tolist(), mode="directed"
    )
    g.vs["name"] = list(gene_names_sub)
    result["igraph"] = g
except ImportError:
    result["igraph"] = None
```

- [ ] **Step 2: Commit**

### Task 9.4: Temporal matching between specific archetype pairs

- [ ] **Step 1: Add archetype_pair parameter to flow_temporal_feature_graph**

```python
def flow_temporal_feature_graph(
    ...,
    archetype_pairs: list[tuple[int, int]] | None = None,
    ...
):
```

When `archetype_pairs` is provided, compute temporal feature graphs only for cells near those archetype pairs (using soft assignment scores from between-fit comparison).

- [ ] **Step 2: Commit**

### Task 9.5: Pathway handling — clearer 1st and 2nd degree separation

- [ ] **Step 1: In notebook generation, explicitly label pathway regression outputs**

When running `feature_simplex_regression` on pathways, ensure the result is stored at `peach_simplex_regression_pathways` and the coefficient heatmap title reflects "Pathway" not "Gene".

- [ ] **Step 2: Commit**

---

## Updated Summary

| Chunk | Description | Effort | Dependencies |
|-------|-------------|--------|-------------|
| 1 | Pattern classification: FDR-based, 3 categories | Low | None |
| 2 | Cache covariance, t-distribution Wald | Low | None |
| 3 | GMM: ICL + Dirichlet + pairwise NMI | Medium | None |
| 4 | Weighted MMD, unbiased estimator, drop silhouette | Medium | Chunk 2 |
| 5 | OT-CFM, dopri5, holdout, per-cell alignment, bifurcation | Medium-High | None |
| 6 | Registry updates | Low | Chunks 1-5 |
| 7 | Notebook bug fixes (Wald underflow, dotplot grouping, residual overwrite, Jacobian det) | Medium | Chunks 1-2 |
| 8 | Visualization readability (sig filter, labels, alpha, legends, radar) | Medium | Chunk 7 |
| 9 | Feature additions (archetype-feature map, soft assignment pl, igraph, temporal pairs) | Medium | Chunks 7-8 |

**Execution order:**
- **Phase 1** (parallel): Chunks 1, 2, 3, 5
- **Phase 2** (after Phase 1): Chunks 4, 7
- **Phase 3** (after Phase 2): Chunks 8, 9
- **Phase 4** (last): Chunk 6

**New dependencies:** `ot` (Python Optimal Transport), `igraph` (optional, for flow graph)

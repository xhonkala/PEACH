# Archetype Comparison API Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add three comparison tools to PEACH: (1) MMD-based archetype similarity with permutation p-values, (2) feature program comparison via silhouette + Spearman, (3) within-fit Wald contrasts on regression coefficients.

**Architecture:** Three new public functions in `tl/comparison.py`, backed by a thin utility module `_core/utils/archetype_comparison.py` for the compute kernels. New Pydantic result types. New plot functions in `pl/comparison.py`. Uses existing `compute_mmd` from `flow_matching.py` and existing `ols_fit`/`scheffe_design_matrix` from `simplex_regression.py`.

**Tech Stack:** numpy, scipy (spatial, stats, sparse), scikit-learn (silhouette_score), existing PEACH utilities.

---

## Overview

### Functions to implement

| Function | Purpose | Returns |
|----------|---------|---------|
| `pc.tl.archetype_mmd(adata, adata_b=None)` | K×K MMD similarity matrix with permutation p-values | `ArchetypeMMDResult` |
| `pc.tl.archetype_feature_similarity(adata, adata_b=None)` | Silhouette scores + Spearman rank correlation matrix | `ArchetypeFeatureSimilarityResult` |
| `pc.tl.archetype_contrasts(adata)` | Pairwise Wald contrasts β_k − β_j with SEs | `ArchetypeContrastsResult` |
| `pc.pl.mmd_heatmap(adata)` | Heatmap of MMD matrix | `go.Figure` |
| `pc.pl.contrast_volcano(adata, pair)` | Volcano plot for one archetype pair contrast | `go.Figure` |
| `pc.pl.feature_similarity_heatmap(adata)` | Spearman correlation heatmap | `go.Figure` |

### File plan

| Action | File |
|--------|------|
| Create | `src/peach/_core/utils/archetype_comparison.py` |
| Create | `src/peach/tl/comparison.py` |
| Create | `src/peach/pl/comparison.py` |
| Create | `tests/test_statistical/test_archetype_comparison.py` |
| Create | `tests/test_visualization/test_comparison_viz.py` |
| Modify | `src/peach/tl/__init__.py` |
| Modify | `src/peach/pl/__init__.py` |
| Modify | `src/peach/_core/types.py` (add 3 result types) |
| Modify | `src/peach/_core/types_index.py` (register new functions) |
| Modify | `src/peach/_core/utils/simplex_regression.py` (expose covariance) |

---

## Task 1: Result types in types.py

**Files:**
- Modify: `src/peach/_core/types.py` (append after DriverRegressionResult)

**Step 1: Write the three result types**

Add after the existing `DriverRegressionResult` class (around line 3470):

```python
class ArchetypeMMDResult(BaseModel):
    """MMD similarity matrix between archetypes."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mmd_matrix: np.ndarray          # [K, K] or [K_A, K_B]
    pvalue_matrix: np.ndarray       # [K, K] or [K_A, K_B]
    n_permutations: int
    is_between_fit: bool = False    # True if comparing two different fits
    archetype_names_a: list[str]
    archetype_names_b: list[str] | None = None  # None for within-fit

    def to_serializable(self) -> dict:
        d = {}
        for field_name, value in self:
            if value is None:
                continue
            if isinstance(value, np.ndarray):
                d[field_name] = value
            else:
                d[field_name] = value
        return d


class ArchetypeFeatureSimilarityResult(BaseModel):
    """Feature-level similarity between archetypes."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    silhouette_per_archetype: np.ndarray   # [K]
    silhouette_overall: float
    spearman_matrix: np.ndarray            # [K, K] or [K_A, K_B]
    spearman_pvalue_matrix: np.ndarray     # [K, K] or [K_A, K_B]
    n_shared_features: int
    is_between_fit: bool = False
    archetype_names_a: list[str]
    archetype_names_b: list[str] | None = None

    def to_serializable(self) -> dict:
        d = {}
        for field_name, value in self:
            if value is None:
                continue
            if isinstance(value, np.ndarray):
                d[field_name] = value
            else:
                d[field_name] = value
        return d


class ArchetypeContrastsResult(BaseModel):
    """Pairwise Wald contrasts between archetype regression coefficients."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    pairs: list[tuple[int, int]]                    # [(j, k), ...]
    delta_beta: dict[tuple[int, int], np.ndarray]   # {(j,k): [n_features]}
    delta_se: dict[tuple[int, int], np.ndarray]     # {(j,k): [n_features]}
    z_scores: dict[tuple[int, int], np.ndarray]     # {(j,k): [n_features]}
    pvalues: dict[tuple[int, int], np.ndarray]      # {(j,k): [n_features] two-sided}
    pvalues_fdr: dict[tuple[int, int], np.ndarray]  # {(j,k): [n_features] BH-corrected}
    feature_names: list[str]
    n_features: int
    n_archetypes: int

    def to_serializable(self) -> dict:
        d = {"pairs": self.pairs, "feature_names": self.feature_names,
             "n_features": self.n_features, "n_archetypes": self.n_archetypes}
        for key in ("delta_beta", "delta_se", "z_scores", "pvalues", "pvalues_fdr"):
            val = getattr(self, key)
            d[key] = {str(k): v for k, v in val.items()}
        return d
```

**Step 2: Verify import**

Run: `conda run -n archetype python -c "from peach._core.types import ArchetypeMMDResult, ArchetypeFeatureSimilarityResult, ArchetypeContrastsResult; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add src/peach/_core/types.py
git commit -m "Add result types for archetype comparison API"
```

---

## Task 2: Expose covariance from ols_fit

The Wald contrast needs the full covariance matrix `Var(β)` for each feature, not just the diagonal (SEs). We modify `ols_fit` to optionally return it.

**Files:**
- Modify: `src/peach/_core/utils/simplex_regression.py:59` (ols_fit signature + return)

**Step 1: Write failing test**

Create inline test in `tests/test_statistical/test_archetype_comparison.py` (we'll add more tests here later):

```python
# tests/test_statistical/test_archetype_comparison.py
import numpy as np
import pytest


class TestOlsFitCovariance:
    def test_returns_covariance_when_requested(self):
        from peach._core.utils.simplex_regression import ols_fit, scheffe_design_matrix

        rng = np.random.default_rng(42)
        K = 3
        n = 200
        weights = rng.dirichlet([1] * K, size=n)
        W, _ = scheffe_design_matrix(weights, degree=1)
        Y = weights @ rng.standard_normal((K, 10)) + rng.normal(0, 0.1, (n, 10))

        result = ols_fit(W, Y, robust_se=True, return_covariance=True)
        assert "covariance" in result
        # Should be a list of [p, p] matrices, one per feature
        assert len(result["covariance"]) == 10
        assert result["covariance"][0].shape == (K, K)

    def test_no_covariance_by_default(self):
        from peach._core.utils.simplex_regression import ols_fit, scheffe_design_matrix

        rng = np.random.default_rng(42)
        K = 3
        n = 200
        weights = rng.dirichlet([1] * K, size=n)
        W, _ = scheffe_design_matrix(weights, degree=1)
        Y = weights @ rng.standard_normal((K, 10)) + rng.normal(0, 0.1, (n, 10))

        result = ols_fit(W, Y, robust_se=True)
        assert "covariance" not in result
```

**Step 2: Run test to verify it fails**

Run: `conda run -n archetype python -m pytest tests/test_statistical/test_archetype_comparison.py::TestOlsFitCovariance -v`

Expected: FAIL (return_covariance kwarg doesn't exist yet)

**Step 3: Modify ols_fit**

In `src/peach/_core/utils/simplex_regression.py`, add `return_covariance=False` parameter to `ols_fit` signature (line 59), and:

1. Add parameter to docstring
2. In the dense path (after line 173), compute and store the full sandwich matrix per feature when `return_covariance=True`
3. In the sparse path (after line 155), same
4. Add `"covariance"` to the returned dict if requested

The key change in the dense path (after existing HC3 SE computation):

```python
    # Full covariance matrices (for Wald contrasts)
    covariance = None
    if return_covariance:
        if robust_se:
            covariance = _hc3_covariance(W, residuals, WtW_inv, H_diag)
        else:
            covariance = []
            for g in range(n_features):
                sigma2_g = ss_res[g] / max(n - p, 1)
                covariance.append(sigma2_g * WtW_inv)
```

And add the helper `_hc3_covariance`:

```python
def _hc3_covariance(W, residuals, WtW_inv, H_diag):
    """Full HC3 sandwich covariance per feature. Returns list of [p, p] matrices."""
    n, p = W.shape
    n_features = residuals.shape[1]
    adjustment = 1.0 / (1 - H_diag)
    cov_list = []
    for g in range(n_features):
        e_adj = residuals[:, g] * adjustment
        We = W * (e_adj ** 2)[:, np.newaxis]
        meat = W.T @ We
        sandwich = WtW_inv @ meat @ WtW_inv
        cov_list.append(sandwich)
    return cov_list
```

Add to the return dict:
```python
    result_dict = {
        "coefficients": beta,
        ...existing keys...
    }
    if return_covariance:
        result_dict["covariance"] = covariance
    return result_dict
```

**Step 4: Run test to verify it passes**

Run: `conda run -n archetype python -m pytest tests/test_statistical/test_archetype_comparison.py::TestOlsFitCovariance -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/peach/_core/utils/simplex_regression.py tests/test_statistical/test_archetype_comparison.py
git commit -m "Expose optional covariance matrix from ols_fit for Wald contrasts"
```

---

## Task 3: Core compute module

**Files:**
- Create: `src/peach/_core/utils/archetype_comparison.py`

**Step 1: Write failing tests**

Add to `tests/test_statistical/test_archetype_comparison.py`:

```python
from anndata import AnnData


@pytest.fixture
def comparison_adata():
    """AnnData with 4 archetypes, planted structure for comparison tests."""
    rng = np.random.default_rng(42)
    K = 4
    n = 600
    n_genes = 30

    weights = rng.dirichlet([1] * K, size=n)
    # Plant known structure: archetypes 0,1 are similar; 2,3 are different
    true_beta = np.zeros((n_genes, K))
    true_beta[:, 0] = rng.normal(5, 1, n_genes)
    true_beta[:, 1] = true_beta[:, 0] + rng.normal(0, 0.3, n_genes)  # similar to 0
    true_beta[:, 2] = rng.normal(-3, 1, n_genes)  # very different
    true_beta[:, 3] = rng.normal(0, 2, n_genes)   # different

    X = weights @ true_beta.T + rng.normal(0, 0.2, (n, n_genes))

    adata = AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obsm["cell_archetype_weights"] = weights
    adata.obsm["X_pca"] = rng.standard_normal((n, 10))
    # Hard assignments for silhouette
    adata.obs["archetype_assignment"] = np.argmax(weights, axis=1).astype(str)
    return adata


class TestArchetypeMMDCompute:
    def test_within_fit_symmetric(self, comparison_adata):
        from peach._core.utils.archetype_comparison import compute_archetype_mmd
        mmd_mat, pval_mat = compute_archetype_mmd(
            comparison_adata, n_permutations=20
        )
        K = 4
        assert mmd_mat.shape == (K, K)
        assert pval_mat.shape == (K, K)
        # Diagonal should be 0 (same vs same)
        np.testing.assert_allclose(np.diag(mmd_mat), 0, atol=1e-10)
        # Symmetric
        np.testing.assert_allclose(mmd_mat, mmd_mat.T, atol=1e-10)

    def test_similar_archetypes_low_mmd(self, comparison_adata):
        from peach._core.utils.archetype_comparison import compute_archetype_mmd
        mmd_mat, _ = compute_archetype_mmd(
            comparison_adata, n_permutations=10
        )
        # Archetypes 0 and 1 should have lower MMD than 0 and 2
        assert mmd_mat[0, 1] < mmd_mat[0, 2]


class TestArchetypeFeatureSimilarity:
    def test_spearman_symmetric(self, comparison_adata):
        from peach._core.utils.archetype_comparison import compute_feature_similarity
        import peach as pc
        # Need regression results first
        pc.tl.feature_simplex_regression(comparison_adata, n_bootstrap=0)
        result = compute_feature_similarity(comparison_adata)
        K = 4
        assert result["spearman_matrix"].shape == (K, K)
        # Diagonal should be 1.0 (perfect self-correlation)
        np.testing.assert_allclose(
            np.diag(result["spearman_matrix"]), 1.0, atol=1e-10
        )

    def test_similar_archetypes_high_spearman(self, comparison_adata):
        from peach._core.utils.archetype_comparison import compute_feature_similarity
        import peach as pc
        pc.tl.feature_simplex_regression(comparison_adata, n_bootstrap=0)
        result = compute_feature_similarity(comparison_adata)
        # Archetypes 0,1 have correlated coefficients → high Spearman
        assert result["spearman_matrix"][0, 1] > result["spearman_matrix"][0, 2]

    def test_silhouette_scores(self, comparison_adata):
        from peach._core.utils.archetype_comparison import compute_feature_similarity
        import peach as pc
        pc.tl.feature_simplex_regression(comparison_adata, n_bootstrap=0)
        result = compute_feature_similarity(comparison_adata)
        assert len(result["silhouette_per_archetype"]) == 4
        assert -1.0 <= result["silhouette_overall"] <= 1.0


class TestArchetypeContrasts:
    def test_all_pairs(self, comparison_adata):
        from peach._core.utils.archetype_comparison import compute_wald_contrasts
        import peach as pc
        pc.tl.feature_simplex_regression(comparison_adata, n_bootstrap=0)
        result = compute_wald_contrasts(comparison_adata)
        K = 4
        n_pairs = K * (K - 1) // 2
        assert len(result["pairs"]) == n_pairs
        for pair in result["pairs"]:
            assert result["delta_beta"][pair].shape == (30,)
            assert result["pvalues_fdr"][pair].shape == (30,)

    def test_similar_pair_fewer_significant(self, comparison_adata):
        from peach._core.utils.archetype_comparison import compute_wald_contrasts
        import peach as pc
        pc.tl.feature_simplex_regression(comparison_adata, n_bootstrap=0)
        result = compute_wald_contrasts(comparison_adata)
        # Pair (0,1) similar → fewer significant genes than (0,2)
        sig_01 = np.sum(result["pvalues_fdr"][(0, 1)] < 0.05)
        sig_02 = np.sum(result["pvalues_fdr"][(0, 2)] < 0.05)
        assert sig_01 < sig_02
```

**Step 2: Run tests to verify they fail**

Run: `conda run -n archetype python -m pytest tests/test_statistical/test_archetype_comparison.py -v -k "not OlsFit"`

Expected: FAIL (module doesn't exist)

**Step 3: Implement core module**

Create `src/peach/_core/utils/archetype_comparison.py`:

```python
"""Core compute functions for archetype comparison: MMD, feature similarity, Wald contrasts."""

import numpy as np
from itertools import combinations
from scipy import stats
from anndata import AnnData

from .feature_utils import get_archetype_weights


def compute_archetype_mmd(
    adata: AnnData,
    adata_b: AnnData | None = None,
    *,
    pca_key: str = "X_pca",
    assignment: str = "hard",
    n_permutations: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute K×K MMD matrix between archetype cell populations.

    Parameters
    ----------
    adata : AnnData
        Must have archetype weights and PCA coordinates.
    adata_b : AnnData or None
        If provided, compute K_A × K_B between-fit comparison.
    pca_key : str
        Key in obsm for cell coordinates.
    assignment : str
        'hard' (argmax) or 'soft' (weight-proportional sampling).
    n_permutations : int
        Permutations for p-value. 0 to skip.
    seed : int

    Returns
    -------
    (mmd_matrix, pvalue_matrix)
    """
    from .flow_matching import compute_mmd

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

    # Get cell sets per archetype (hard assignment)
    assign_a = np.argmax(weights_a, axis=1)
    assign_b = np.argmax(weights_b, axis=1)

    def _get_cells(pca, assignments, k):
        return pca[assignments == k]

    # Compute MMD matrix
    mmd_matrix = np.zeros((K_a, K_b))
    pvalue_matrix = np.ones((K_a, K_b))

    for i in range(K_a):
        cells_i = _get_cells(pca_a, assign_a, i)
        for j in range(K_b):
            if adata_b is None and j <= i:
                # Within-fit: fill lower triangle by symmetry (skip diagonal = 0)
                if j < i:
                    mmd_matrix[i, j] = mmd_matrix[j, i]
                    pvalue_matrix[i, j] = pvalue_matrix[j, i]
                continue

            cells_j = _get_cells(pca_b, assign_b, j)

            if len(cells_i) < 2 or len(cells_j) < 2:
                mmd_matrix[i, j] = np.nan
                pvalue_matrix[i, j] = np.nan
                continue

            observed_mmd = compute_mmd(cells_i, cells_j)
            mmd_matrix[i, j] = observed_mmd

            # Permutation test
            if n_permutations > 0:
                combined = np.vstack([cells_i, cells_j])
                n_i = len(cells_i)
                null_mmds = np.empty(n_permutations)
                for p in range(n_permutations):
                    perm = rng.permutation(len(combined))
                    null_mmds[p] = compute_mmd(
                        combined[perm[:n_i]], combined[perm[n_i:]]
                    )
                pvalue_matrix[i, j] = (
                    np.sum(null_mmds >= observed_mmd) + 1
                ) / (n_permutations + 1)

    # Fill lower triangle for within-fit
    if adata_b is None:
        mmd_matrix = np.maximum(mmd_matrix, mmd_matrix.T)
        # p-values: take the computed upper triangle
        for i in range(K_a):
            for j in range(i):
                pvalue_matrix[i, j] = pvalue_matrix[j, i]

    return mmd_matrix, pvalue_matrix


def compute_feature_similarity(
    adata: AnnData,
    adata_b: AnnData | None = None,
    *,
    pca_key: str = "X_pca",
) -> dict:
    """Silhouette scores + Spearman correlation on regression coefficients.

    Requires simplex regression results in adata.uns['peach_simplex_regression'].

    Parameters
    ----------
    adata : AnnData
    adata_b : AnnData or None
    pca_key : str

    Returns
    -------
    dict with silhouette_per_archetype, silhouette_overall,
         spearman_matrix, spearman_pvalue_matrix, n_shared_features
    """
    from sklearn.metrics import silhouette_score, silhouette_samples

    reg_a = adata.uns.get("peach_simplex_regression")
    if reg_a is None:
        raise ValueError(
            "No regression results. Run pc.tl.feature_simplex_regression() first."
        )

    coefs_a = np.asarray(reg_a["vertex_coefficients"])  # [n_features, K_a]
    names_a = list(reg_a["feature_names"])
    K_a = coefs_a.shape[1]

    if adata_b is not None:
        reg_b = adata_b.uns.get("peach_simplex_regression")
        if reg_b is None:
            raise ValueError("No regression results in adata_b.")
        coefs_b = np.asarray(reg_b["vertex_coefficients"])
        names_b = list(reg_b["feature_names"])
        K_b = coefs_b.shape[1]

        # Intersect features
        shared = sorted(set(names_a) & set(names_b))
        idx_a = [names_a.index(g) for g in shared]
        idx_b = [names_b.index(g) for g in shared]
        coefs_a_shared = coefs_a[idx_a]
        coefs_b_shared = coefs_b[idx_b]
    else:
        coefs_b_shared = coefs_a
        coefs_a_shared = coefs_a
        K_b = K_a
        shared = names_a

    n_shared = len(shared)

    # Spearman correlation: correlate β vectors between archetypes
    spearman_matrix = np.zeros((K_a, K_b))
    spearman_pvalue_matrix = np.ones((K_a, K_b))

    for i in range(K_a):
        for j in range(K_b):
            rho, pval = stats.spearmanr(coefs_a_shared[:, i], coefs_b_shared[:, j])
            spearman_matrix[i, j] = rho
            spearman_pvalue_matrix[i, j] = pval

    # Silhouette on expression space (within-fit only, needs hard assignments)
    weights_a = get_archetype_weights(adata)
    labels = np.argmax(weights_a, axis=1)
    pca = adata.obsm[pca_key]

    # Subsample for silhouette if large
    n = len(labels)
    if n > 10000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, 10000, replace=False)
        pca_sub = pca[idx]
        labels_sub = labels[idx]
    else:
        pca_sub = pca
        labels_sub = labels

    unique_labels = np.unique(labels_sub)
    if len(unique_labels) < 2:
        sil_overall = 0.0
        sil_per = np.zeros(K_a)
    else:
        sil_overall = silhouette_score(pca_sub, labels_sub)
        sil_samples = silhouette_samples(pca_sub, labels_sub)
        sil_per = np.array([
            sil_samples[labels_sub == k].mean() if np.any(labels_sub == k) else 0.0
            for k in range(K_a)
        ])

    return {
        "silhouette_per_archetype": sil_per,
        "silhouette_overall": sil_overall,
        "spearman_matrix": spearman_matrix,
        "spearman_pvalue_matrix": spearman_pvalue_matrix,
        "n_shared_features": n_shared,
    }


def compute_wald_contrasts(
    adata: AnnData,
    *,
    robust_se: bool = True,
) -> dict:
    """Pairwise Wald contrasts β_k − β_j with SEs from the regression covariance.

    Re-runs degree-1 regression with return_covariance=True to get the full
    sandwich covariance, then computes contrasts for all K*(K-1)/2 pairs.

    Parameters
    ----------
    adata : AnnData
        Must have archetype weights and regression results (for feature names).
    robust_se : bool
        Use HC3 covariance.

    Returns
    -------
    dict with pairs, delta_beta, delta_se, z_scores, pvalues, pvalues_fdr,
         feature_names, n_features, n_archetypes
    """
    from statsmodels.stats.multitest import multipletests
    from .simplex_regression import ols_fit, scheffe_design_matrix
    from .feature_utils import resolve_features

    reg = adata.uns.get("peach_simplex_regression")
    if reg is None:
        raise ValueError(
            "No regression results. Run pc.tl.feature_simplex_regression() first."
        )

    weights = get_archetype_weights(adata)
    feat_names = list(reg["feature_names"])
    K = weights.shape[1]
    n_features = len(feat_names)

    # Re-fit with covariance
    Y, _ = resolve_features(adata, None, feat_names)
    W, _ = scheffe_design_matrix(weights, degree=1)
    fit = ols_fit(W, Y, robust_se=robust_se, return_covariance=True)

    beta = fit["coefficients"]  # [n_features, K]
    cov_list = fit["covariance"]  # list of K×K matrices, length n_features

    pairs = list(combinations(range(K), 2))
    delta_beta = {}
    delta_se = {}
    z_scores = {}
    pvalues = {}
    pvalues_fdr = {}

    # Contrast vector: e_j - e_k (for β_j - β_k)
    for j, k in pairs:
        contrast = np.zeros(K)
        contrast[j] = 1.0
        contrast[k] = -1.0

        d_beta = beta[:, j] - beta[:, k]  # [n_features]
        d_se = np.array([
            np.sqrt(max(contrast @ cov_list[g] @ contrast, 0))
            for g in range(n_features)
        ])

        z = np.where(d_se > 0, d_beta / d_se, 0.0)
        pval = 2 * stats.norm.sf(np.abs(z))
        _, pval_fdr, _, _ = multipletests(pval, method="fdr_bh")

        delta_beta[(j, k)] = d_beta
        delta_se[(j, k)] = d_se
        z_scores[(j, k)] = z
        pvalues[(j, k)] = pval
        pvalues_fdr[(j, k)] = pval_fdr

    return {
        "pairs": pairs,
        "delta_beta": delta_beta,
        "delta_se": delta_se,
        "z_scores": z_scores,
        "pvalues": pvalues,
        "pvalues_fdr": pvalues_fdr,
        "feature_names": feat_names,
        "n_features": n_features,
        "n_archetypes": K,
    }
```

**Step 4: Run tests**

Run: `conda run -n archetype python -m pytest tests/test_statistical/test_archetype_comparison.py -v`

Expected: All PASS

**Step 5: Commit**

```bash
git add src/peach/_core/utils/archetype_comparison.py tests/test_statistical/test_archetype_comparison.py
git commit -m "Add core compute functions for archetype MMD, feature similarity, Wald contrasts"
```

---

## Task 4: Public API in tl/comparison.py

**Files:**
- Create: `src/peach/tl/comparison.py`
- Modify: `src/peach/tl/__init__.py`

**Step 1: Write failing test**

Add to `tests/test_statistical/test_archetype_comparison.py`:

```python
class TestPublicAPI:
    def test_archetype_mmd_api(self, comparison_adata):
        import peach as pc
        result = pc.tl.archetype_mmd(comparison_adata, n_permutations=10)
        assert hasattr(result, "mmd_matrix")
        assert hasattr(result, "pvalue_matrix")
        assert result.mmd_matrix.shape == (4, 4)
        # Stored in uns
        assert "peach_archetype_mmd" in comparison_adata.uns

    def test_archetype_feature_similarity_api(self, comparison_adata):
        import peach as pc
        pc.tl.feature_simplex_regression(comparison_adata, n_bootstrap=0)
        result = pc.tl.archetype_feature_similarity(comparison_adata)
        assert hasattr(result, "spearman_matrix")
        assert hasattr(result, "silhouette_overall")
        assert "peach_archetype_feature_similarity" in comparison_adata.uns

    def test_archetype_contrasts_api(self, comparison_adata):
        import peach as pc
        pc.tl.feature_simplex_regression(comparison_adata, n_bootstrap=0)
        result = pc.tl.archetype_contrasts(comparison_adata)
        assert hasattr(result, "pairs")
        assert hasattr(result, "delta_beta")
        assert "peach_archetype_contrasts" in comparison_adata.uns
```

**Step 2: Run to verify failure**

Run: `conda run -n archetype python -m pytest tests/test_statistical/test_archetype_comparison.py::TestPublicAPI -v`

Expected: FAIL (ImportError)

**Step 3: Implement tl/comparison.py**

```python
"""Archetype comparison: MMD similarity, feature similarity, Wald contrasts."""

from anndata import AnnData

from peach._core.types import (
    ArchetypeMMDResult,
    ArchetypeFeatureSimilarityResult,
    ArchetypeContrastsResult,
)
from peach._core.utils.archetype_comparison import (
    compute_archetype_mmd,
    compute_feature_similarity,
    compute_wald_contrasts,
)
from peach._core.utils.feature_utils import get_archetype_weights, store_result


def archetype_mmd(
    adata: AnnData,
    adata_b: AnnData | None = None,
    *,
    pca_key: str = "X_pca",
    n_permutations: int = 1000,
    seed: int = 42,
    copy: bool = False,
) -> ArchetypeMMDResult:
    """K x K MMD similarity matrix between archetype cell populations.

    Parameters
    ----------
    adata : AnnData
        Must have archetype weights in obsm['cell_archetype_weights']
        and PCA coordinates in obsm[pca_key].
    adata_b : AnnData or None
        If provided, compute K_A x K_B between-fit comparison.
    pca_key : str
        Key in obsm for cell coordinates.
    n_permutations : int
        Permutations for p-value computation.
    seed : int
    copy : bool

    Returns
    -------
    ArchetypeMMDResult
        Also stored in adata.uns['peach_archetype_mmd'].
    """
    if copy:
        adata = adata.copy()

    K_a = get_archetype_weights(adata).shape[1]
    mmd_matrix, pvalue_matrix = compute_archetype_mmd(
        adata, adata_b, pca_key=pca_key,
        n_permutations=n_permutations, seed=seed,
    )

    arch_names_a = [f"archetype_{i}" for i in range(K_a)]
    arch_names_b = None
    is_between = adata_b is not None
    if is_between:
        K_b = get_archetype_weights(adata_b).shape[1]
        arch_names_b = [f"archetype_{i}" for i in range(K_b)]

    result = ArchetypeMMDResult(
        mmd_matrix=mmd_matrix,
        pvalue_matrix=pvalue_matrix,
        n_permutations=n_permutations,
        is_between_fit=is_between,
        archetype_names_a=arch_names_a,
        archetype_names_b=arch_names_b,
    )
    store_result(adata, "archetype_mmd", result.to_serializable())
    return result


def archetype_feature_similarity(
    adata: AnnData,
    adata_b: AnnData | None = None,
    *,
    pca_key: str = "X_pca",
    copy: bool = False,
) -> ArchetypeFeatureSimilarityResult:
    """Feature-level archetype similarity: silhouette + Spearman on β vectors.

    Parameters
    ----------
    adata : AnnData
        Must have regression results in uns['peach_simplex_regression'].
    adata_b : AnnData or None
        If provided, compute between-fit Spearman on shared features.
    pca_key : str
    copy : bool

    Returns
    -------
    ArchetypeFeatureSimilarityResult
        Also stored in adata.uns['peach_archetype_feature_similarity'].
    """
    if copy:
        adata = adata.copy()

    K_a = get_archetype_weights(adata).shape[1]
    sim = compute_feature_similarity(adata, adata_b, pca_key=pca_key)

    arch_names_a = [f"archetype_{i}" for i in range(K_a)]
    is_between = adata_b is not None
    arch_names_b = None
    if is_between:
        K_b = get_archetype_weights(adata_b).shape[1]
        arch_names_b = [f"archetype_{i}" for i in range(K_b)]

    result = ArchetypeFeatureSimilarityResult(
        silhouette_per_archetype=sim["silhouette_per_archetype"],
        silhouette_overall=sim["silhouette_overall"],
        spearman_matrix=sim["spearman_matrix"],
        spearman_pvalue_matrix=sim["spearman_pvalue_matrix"],
        n_shared_features=sim["n_shared_features"],
        is_between_fit=is_between,
        archetype_names_a=arch_names_a,
        archetype_names_b=arch_names_b,
    )
    store_result(adata, "archetype_feature_similarity", result.to_serializable())
    return result


def archetype_contrasts(
    adata: AnnData,
    *,
    robust_se: bool = True,
    copy: bool = False,
) -> ArchetypeContrastsResult:
    """Pairwise Wald contrasts between archetype regression coefficients.

    For each pair (j, k), tests H0: β_j = β_k for every feature using
    the Wald statistic with HC3 covariance from the Scheffe regression.

    Parameters
    ----------
    adata : AnnData
        Must have archetype weights and regression results.
    robust_se : bool
        Use HC3 heteroscedasticity-consistent covariance.
    copy : bool

    Returns
    -------
    ArchetypeContrastsResult
        Also stored in adata.uns['peach_archetype_contrasts'].
    """
    if copy:
        adata = adata.copy()

    contrasts = compute_wald_contrasts(adata, robust_se=robust_se)

    result = ArchetypeContrastsResult(
        pairs=contrasts["pairs"],
        delta_beta=contrasts["delta_beta"],
        delta_se=contrasts["delta_se"],
        z_scores=contrasts["z_scores"],
        pvalues=contrasts["pvalues"],
        pvalues_fdr=contrasts["pvalues_fdr"],
        feature_names=contrasts["feature_names"],
        n_features=contrasts["n_features"],
        n_archetypes=contrasts["n_archetypes"],
    )
    store_result(adata, "archetype_contrasts", result.to_serializable())
    return result
```

**Step 4: Update tl/__init__.py**

Add after the flow imports:

```python
# v0.5.0: Archetype comparison
from .comparison import (
    archetype_mmd,
    archetype_feature_similarity,
    archetype_contrasts,
)
```

And add to `__all__`:
```python
    # v0.5.0: Archetype comparison
    "archetype_mmd",
    "archetype_feature_similarity",
    "archetype_contrasts",
```

**Step 5: Run tests**

Run: `conda run -n archetype python -m pytest tests/test_statistical/test_archetype_comparison.py -v`

Expected: All PASS

**Step 6: Commit**

```bash
git add src/peach/tl/comparison.py src/peach/tl/__init__.py
git commit -m "Add archetype comparison public API: MMD, feature similarity, Wald contrasts"
```

---

## Task 5: Visualization in pl/comparison.py

**Files:**
- Create: `src/peach/pl/comparison.py`
- Create: `tests/test_visualization/test_comparison_viz.py`
- Modify: `src/peach/pl/__init__.py`

**Step 1: Write failing tests**

Create `tests/test_visualization/test_comparison_viz.py`:

```python
import numpy as np
import pytest
from anndata import AnnData
import plotly.graph_objects as go


@pytest.fixture
def viz_adata():
    """AnnData with comparison results pre-computed."""
    rng = np.random.default_rng(42)
    K = 3
    n = 300
    n_genes = 20

    weights = rng.dirichlet([1] * K, size=n)
    true_beta = rng.standard_normal((n_genes, K)) * 5
    X = weights @ true_beta.T + rng.normal(0, 0.2, (n, n_genes))

    adata = AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obsm["cell_archetype_weights"] = weights
    adata.obsm["X_pca"] = rng.standard_normal((n, 10))

    import peach as pc
    pc.tl.feature_simplex_regression(adata, n_bootstrap=0)
    pc.tl.archetype_mmd(adata, n_permutations=10)
    pc.tl.archetype_feature_similarity(adata)
    pc.tl.archetype_contrasts(adata)
    return adata


class TestMMDHeatmap:
    def test_returns_figure(self, viz_adata):
        import peach as pc
        fig = pc.pl.mmd_heatmap(viz_adata, show=False)
        assert isinstance(fig, go.Figure)

    def test_has_heatmap_trace(self, viz_adata):
        import peach as pc
        fig = pc.pl.mmd_heatmap(viz_adata, show=False)
        assert any(isinstance(t, go.Heatmap) for t in fig.data)


class TestContrastVolcano:
    def test_returns_figure(self, viz_adata):
        import peach as pc
        fig = pc.pl.contrast_volcano(viz_adata, pair=(0, 1), show=False)
        assert isinstance(fig, go.Figure)

    def test_scatter_trace(self, viz_adata):
        import peach as pc
        fig = pc.pl.contrast_volcano(viz_adata, pair=(0, 1), show=False)
        assert any(isinstance(t, go.Scatter) for t in fig.data)


class TestFeatureSimilarityHeatmap:
    def test_returns_figure(self, viz_adata):
        import peach as pc
        fig = pc.pl.feature_similarity_heatmap(viz_adata, show=False)
        assert isinstance(fig, go.Figure)
```

**Step 2: Run to verify failure**

Run: `conda run -n archetype python -m pytest tests/test_visualization/test_comparison_viz.py -v`

Expected: FAIL

**Step 3: Implement pl/comparison.py**

```python
"""Archetype comparison visualizations: MMD heatmap, contrast volcano, similarity heatmap."""

import numpy as np
import plotly.graph_objects as go
from anndata import AnnData

from ._style import (
    COLOR_NEGATIVE,
    COLOR_PRIMARY,
    DIVERGING_COLORSCALE,
    SEQUENTIAL_COLORSCALE,
    apply_style,
    save_and_show,
)


def mmd_heatmap(
    adata: AnnData,
    *,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Heatmap of K x K MMD matrix between archetypes.

    Parameters
    ----------
    adata : AnnData
        Must have MMD results in uns['peach_archetype_mmd'].
    save_path : str or None
    show : bool

    Returns
    -------
    go.Figure
    """
    mmd_data = adata.uns.get("peach_archetype_mmd")
    if mmd_data is None:
        raise ValueError("No MMD results. Run pc.tl.archetype_mmd() first.")

    mmd_matrix = np.asarray(mmd_data["mmd_matrix"])
    K = mmd_matrix.shape[0]
    labels = [f"A{i}" for i in range(K)]

    fig = go.Figure(data=go.Heatmap(
        z=mmd_matrix,
        x=labels,
        y=labels,
        colorscale=SEQUENTIAL_COLORSCALE,
        colorbar=dict(title="MMD", thickness=12, len=0.6),
    ))
    apply_style(fig, title="Archetype MMD similarity",
                xaxis_title="Archetype", yaxis_title="Archetype")
    return save_and_show(fig, save_path=save_path, show=show)


def contrast_volcano(
    adata: AnnData,
    pair: tuple[int, int],
    *,
    fdr_threshold: float = 0.05,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Volcano plot for one archetype pair contrast: Δβ vs -log10(p).

    Parameters
    ----------
    adata : AnnData
        Must have contrast results in uns['peach_archetype_contrasts'].
    pair : tuple[int, int]
        Archetype pair (j, k).
    fdr_threshold : float
    save_path : str or None
    show : bool

    Returns
    -------
    go.Figure
    """
    contrast_data = adata.uns.get("peach_archetype_contrasts")
    if contrast_data is None:
        raise ValueError("No contrast results. Run pc.tl.archetype_contrasts() first.")

    pair_key = str(pair)
    delta = np.asarray(contrast_data["delta_beta"][pair_key])
    pvals = np.asarray(contrast_data["pvalues_fdr"][pair_key])
    names = list(contrast_data["feature_names"])

    neg_log_p = -np.log10(np.maximum(pvals, 1e-300))
    colors = [COLOR_NEGATIVE if p < fdr_threshold else COLOR_PRIMARY for p in pvals]

    fig = go.Figure(data=go.Scatter(
        x=delta,
        y=neg_log_p,
        mode="markers",
        text=names,
        hovertemplate="%{text}<br>Δβ=%{x:.3f}<br>-log10(q)=%{y:.1f}<extra></extra>",
        marker=dict(size=5, opacity=0.6, color=colors),
    ))

    # FDR threshold line
    fig.add_hline(y=-np.log10(fdr_threshold), line_dash="dot",
                  line_color="#999", line_width=1)

    j, k = pair
    apply_style(fig, title=f"Contrast: A{j} vs A{k}",
                xaxis_title=f"β_{j} − β_{k}",
                yaxis_title="-log₁₀(FDR q)")
    return save_and_show(fig, save_path=save_path, show=show)


def feature_similarity_heatmap(
    adata: AnnData,
    *,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Heatmap of Spearman correlation between archetype β vectors.

    Parameters
    ----------
    adata : AnnData
        Must have feature similarity results in
        uns['peach_archetype_feature_similarity'].
    save_path : str or None
    show : bool

    Returns
    -------
    go.Figure
    """
    sim_data = adata.uns.get("peach_archetype_feature_similarity")
    if sim_data is None:
        raise ValueError(
            "No feature similarity results. "
            "Run pc.tl.archetype_feature_similarity() first."
        )

    spearman = np.asarray(sim_data["spearman_matrix"])
    K = spearman.shape[0]
    labels = [f"A{i}" for i in range(K)]

    fig = go.Figure(data=go.Heatmap(
        z=spearman,
        x=labels,
        y=labels,
        colorscale=DIVERGING_COLORSCALE,
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(title="ρ", thickness=12, len=0.6),
    ))
    apply_style(fig, title="Archetype feature similarity (Spearman)",
                xaxis_title="Archetype", yaxis_title="Archetype")
    return save_and_show(fig, save_path=save_path, show=show)
```

**Step 4: Update pl/__init__.py**

Add after flow imports:

```python
# Archetype comparison visualization
from .comparison import (
    mmd_heatmap,
    contrast_volcano,
    feature_similarity_heatmap,
)
```

And add to `__all__`:
```python
    # Archetype comparison
    "mmd_heatmap",
    "contrast_volcano",
    "feature_similarity_heatmap",
```

**Step 5: Run tests**

Run: `conda run -n archetype python -m pytest tests/test_visualization/test_comparison_viz.py -v`

Expected: All PASS

**Step 6: Commit**

```bash
git add src/peach/pl/comparison.py src/peach/pl/__init__.py tests/test_visualization/test_comparison_viz.py
git commit -m "Add archetype comparison visualization: MMD heatmap, contrast volcano, similarity heatmap"
```

---

## Task 6: Update types_index.py and tools_schema.py

**Files:**
- Modify: `src/peach/_core/types_index.py`

**Step 1: Add new function entries to FUNCTION_RETURNS**

In the `tl` section, add:

```python
    "tl.archetype_mmd": (
        "ArchetypeMMDResult",
        [
            "mmd_matrix: [K, K] or [K_A, K_B] MMD values",
            "pvalue_matrix: permutation p-values",
            "stored in adata.uns['peach_archetype_mmd']",
        ],
    ),
    "tl.archetype_feature_similarity": (
        "ArchetypeFeatureSimilarityResult",
        [
            "silhouette_per_archetype: [K]",
            "silhouette_overall: float",
            "spearman_matrix: [K, K] rank correlation",
            "stored in adata.uns['peach_archetype_feature_similarity']",
        ],
    ),
    "tl.archetype_contrasts": (
        "ArchetypeContrastsResult",
        [
            "pairs: list of (j, k) tuples",
            "delta_beta: {(j,k): [n_features]}",
            "pvalues_fdr: {(j,k): [n_features]} BH-corrected",
            "stored in adata.uns['peach_archetype_contrasts']",
        ],
    ),
```

In the `pl` section, add:

```python
    "pl.mmd_heatmap": ("Figure", ["Heatmap of K×K MMD matrix"]),
    "pl.contrast_volcano": ("Figure", ["Volcano plot: Δβ vs -log10(FDR q)"]),
    "pl.feature_similarity_heatmap": ("Figure", ["Spearman correlation heatmap"]),
```

**Step 2: Verify import**

Run: `conda run -n archetype python -c "from peach._core.types_index import FUNCTION_RETURNS; print('archetype_mmd' in str(FUNCTION_RETURNS))"`

Expected: `True`

**Step 3: Commit**

```bash
git add src/peach/_core/types_index.py
git commit -m "Register archetype comparison functions in types_index"
```

---

## Task 7: Run full test suite

**Step 1: Run comparison tests**

Run: `conda run -n archetype python -m pytest tests/test_statistical/test_archetype_comparison.py tests/test_visualization/test_comparison_viz.py -v`

Expected: All PASS

**Step 2: Run existing tests to check for regressions**

Run: `conda run -n archetype python -m pytest tests/test_statistical/ tests/test_visualization/ -v --timeout=120`

Expected: All existing tests still PASS

**Step 3: Commit (if any fixes needed)**

---

## Summary

| Task | What | Tests |
|------|------|-------|
| 1 | Result types in types.py | Import check |
| 2 | Expose covariance from ols_fit | 2 tests |
| 3 | Core compute module | 8 tests (MMD: 2, Feature: 3, Contrasts: 2, planted signal: 1) |
| 4 | Public API (tl/comparison.py) | 3 tests |
| 5 | Visualization (pl/comparison.py) | 5 tests |
| 6 | types_index.py registration | Import check |
| 7 | Full regression test | Existing suite |

Total new tests: ~20

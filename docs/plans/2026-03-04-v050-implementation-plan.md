# v0.5.0 Continuous Archetype Characterization — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace binary archetype membership analysis with principled continuous regression, pattern classification, GMM decomposition, and cross-condition flow matching.

**Architecture:** Shared feature infrastructure (resolve_features, resampling) underpins four analysis modules (simplex regression, pattern classification, GMM, flow matching) each with core engine + public API + visualization. All follow PEACH's `(adata, *, kwargs) -> TypedResult` pattern with results stored in `adata.uns['peach_*']`.

**Tech Stack:** numpy/scipy (regression, ILR), sklearn (GMM), torch + fb flow_matching (flow), plotly (viz), python-ternary (optional), pydantic v2 (types).

**Design Reference:** `docs/plans/2026-03-04-v050-continuous-characterization-design.md`

**Existing Patterns:**
- API export: explicit imports + `__all__` in `tl/__init__.py`, `pl/__init__.py`
- Function signatures: `(adata, *, param=default, ...) -> Result`
- Optional deps: `_check_<dep>()` helper, import at call time
- Plotting: plotly `go.Figure`, optional `save_path`, `fig.show()` + return
- Core logic in `_core/utils/*.py`, thin wrappers in `tl/*.py`
- Tests: `conftest.py` has `sample_adata`, `small_adata`, `synthetic_adata`, `trained_small_adata`

---

## Phase 1: Shared Infrastructure

### Task 1.1: Feature Utilities (`feature_utils.py`)

**Files:**
- Create: `src/peach/_core/utils/feature_utils.py`
- Test: `tests/test_core/test_feature_utils.py`

**Step 1: Write failing tests**

```python
# tests/test_core/test_feature_utils.py
import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData


class TestResolveFeatures:
    """Test resolve_features() input resolution."""

    def test_default_uses_adata_x_dense(self):
        """feature_matrix=None -> adata.X, feature_names from var_names."""
        from peach._core.utils.feature_utils import resolve_features

        X = np.random.rand(100, 50)
        adata = AnnData(X, var={"gene": [f"gene_{i}" for i in range(50)]})
        adata.var_names = [f"gene_{i}" for i in range(50)]
        mat, names = resolve_features(adata)
        assert isinstance(mat, np.ndarray)
        assert mat.shape == (100, 50)
        assert len(names) == 50
        assert names[0] == "gene_0"

    def test_default_uses_adata_x_sparse(self):
        """Sparse adata.X stays sparse — never densified."""
        from peach._core.utils.feature_utils import resolve_features

        X = sp.random(100, 50, density=0.3, format="csr")
        adata = AnnData(X)
        mat, names = resolve_features(adata)
        assert sp.issparse(mat)
        assert mat.shape == (100, 50)

    def test_obsm_key_string(self):
        """feature_matrix='pathway_scores' -> adata.obsm['pathway_scores']."""
        from peach._core.utils.feature_utils import resolve_features

        adata = AnnData(np.zeros((100, 10)))
        adata.obsm["pathway_scores"] = np.random.rand(100, 20)
        mat, names = resolve_features(adata, feature_matrix="pathway_scores")
        assert mat.shape == (100, 20)
        assert len(names) == 20

    def test_direct_array_passthrough(self):
        """feature_matrix=np.ndarray -> use directly."""
        from peach._core.utils.feature_utils import resolve_features

        adata = AnnData(np.zeros((100, 10)))
        custom = np.random.rand(100, 30)
        custom_names = [f"feat_{i}" for i in range(30)]
        mat, names = resolve_features(adata, feature_matrix=custom, feature_names=custom_names)
        np.testing.assert_array_equal(mat, custom)
        assert names == custom_names

    def test_missing_obsm_key_raises(self):
        """Missing obsm key raises KeyError."""
        from peach._core.utils.feature_utils import resolve_features

        adata = AnnData(np.zeros((100, 10)))
        with pytest.raises(KeyError):
            resolve_features(adata, feature_matrix="nonexistent_key")

    def test_shape_mismatch_raises(self):
        """Array with wrong n_cells raises ValueError."""
        from peach._core.utils.feature_utils import resolve_features

        adata = AnnData(np.zeros((100, 10)))
        wrong_shape = np.random.rand(50, 10)
        with pytest.raises(ValueError, match="n_cells"):
            resolve_features(adata, feature_matrix=wrong_shape)


class TestGetArchetypeWeights:
    """Test get_archetype_weights() extraction + validation."""

    def test_valid_weights(self):
        """Weights summing to 1 are returned."""
        from peach._core.utils.feature_utils import get_archetype_weights

        adata = AnnData(np.zeros((100, 10)))
        weights = np.random.dirichlet([1] * 4, size=100)
        adata.obsm["cell_archetype_weights"] = weights
        result = get_archetype_weights(adata)
        np.testing.assert_array_almost_equal(result, weights)

    def test_missing_weights_raises(self):
        """Missing weights key raises KeyError."""
        from peach._core.utils.feature_utils import get_archetype_weights

        adata = AnnData(np.zeros((100, 10)))
        with pytest.raises(KeyError):
            get_archetype_weights(adata)

    def test_bad_sum_raises(self):
        """Weights not summing to 1 raises ValueError — never silently renormalizes."""
        from peach._core.utils.feature_utils import get_archetype_weights

        adata = AnnData(np.zeros((100, 10)))
        weights = np.random.rand(100, 4)  # won't sum to 1
        adata.obsm["cell_archetype_weights"] = weights
        with pytest.raises(ValueError, match="sum-to-1"):
            get_archetype_weights(adata)


class TestStoreResult:
    """Test store_result() storage helper."""

    def test_stores_in_uns(self):
        """Result stored with peach_ prefix in uns."""
        from peach._core.utils.feature_utils import store_result

        adata = AnnData(np.zeros((10, 5)))
        store_result(adata, "simplex_regression", {"r2": 0.5})
        assert "peach_simplex_regression" in adata.uns
        assert adata.uns["peach_simplex_regression"]["r2"] == 0.5

    def test_stores_in_obsm(self):
        """Result stored in obsm when domain='obsm'."""
        from peach._core.utils.feature_utils import store_result

        adata = AnnData(np.zeros((10, 5)))
        arr = np.zeros(10)
        store_result(adata, "gmm_labels", arr, domain="obsm")
        assert "peach_gmm_labels" in adata.obsm

    def test_overwrite_warns(self, caplog):
        """Overwriting existing result logs warning."""
        import logging
        from peach._core.utils.feature_utils import store_result

        adata = AnnData(np.zeros((10, 5)))
        store_result(adata, "test_key", {"v": 1})
        with caplog.at_level(logging.WARNING):
            store_result(adata, "test_key", {"v": 2})
        assert "overwriting" in caplog.text.lower()
```

**Step 2: Run tests to verify they fail**

Run: `conda run -n archetype pytest tests/test_core/test_feature_utils.py -v`
Expected: FAIL (ImportError — module doesn't exist yet)

**Step 3: Write implementation**

```python
# src/peach/_core/utils/feature_utils.py
"""Shared feature resolution and storage utilities for v0.5.0 analysis modules."""

import logging
from typing import Any

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

logger = logging.getLogger(__name__)

WEIGHTS_KEY = "cell_archetype_weights"
WEIGHTS_TOLERANCE = 1e-8


def resolve_features(
    adata: AnnData,
    feature_matrix=None,
    feature_names=None,
):
    """Resolve feature matrix and names from flexible input.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    feature_matrix : None, str, or array-like
        None -> adata.X (kept sparse if sparse).
        str -> adata.obsm[feature_matrix].
        array-like -> used directly.
    feature_names : list[str] or None
        Feature names. Inferred from adata.var_names if None and using adata.X.

    Returns
    -------
    tuple[np.ndarray | sp.spmatrix, list[str]]
        (matrix [n_cells, n_features], feature_names)
    """
    n_cells = adata.n_obs

    if feature_matrix is None:
        mat = adata.X
        names = list(adata.var_names) if feature_names is None else list(feature_names)
    elif isinstance(feature_matrix, str):
        mat = adata.obsm[feature_matrix]  # KeyError if missing
        n_feats = mat.shape[1]
        names = (
            list(feature_names)
            if feature_names is not None
            else [f"feature_{i}" for i in range(n_feats)]
        )
    else:
        mat = np.asarray(feature_matrix) if not sp.issparse(feature_matrix) else feature_matrix
        if mat.shape[0] != n_cells:
            raise ValueError(
                f"feature_matrix n_cells mismatch: got {mat.shape[0]}, expected {n_cells}"
            )
        n_feats = mat.shape[1]
        names = (
            list(feature_names)
            if feature_names is not None
            else [f"feature_{i}" for i in range(n_feats)]
        )

    return mat, names


def get_archetype_weights(adata: AnnData) -> np.ndarray:
    """Extract archetype weights from adata.obsm.

    Asserts sum-to-1 within tolerance. Raises ValueError if violated —
    never silently renormalizes.

    Returns
    -------
    np.ndarray
        Weights array [n_cells, K].
    """
    if WEIGHTS_KEY not in adata.obsm:
        raise KeyError(
            f"adata.obsm['{WEIGHTS_KEY}'] not found. "
            "Run pc.tl.extract_archetype_weights() first."
        )
    weights = np.asarray(adata.obsm[WEIGHTS_KEY])
    row_sums = weights.sum(axis=1)
    max_deviation = np.abs(row_sums - 1.0).max()
    if max_deviation > WEIGHTS_TOLERANCE:
        raise ValueError(
            f"Archetype weights violate sum-to-1 constraint: max deviation {max_deviation:.2e} "
            f"exceeds tolerance {WEIGHTS_TOLERANCE:.0e}. This indicates a bug upstream."
        )
    return weights


def store_result(
    adata: AnnData,
    key: str,
    result: Any,
    domain: str = "uns",
):
    """Store result in adata with peach_ prefix.

    Parameters
    ----------
    adata : AnnData
    key : str
        Key name (without peach_ prefix).
    result : Any
        Serializable result (dict, array, DataFrame).
    domain : str
        'uns' or 'obsm'.
    """
    full_key = f"peach_{key}"
    storage = getattr(adata, domain)

    if full_key in storage:
        logger.warning(f"Overwriting existing results at adata.{domain}['{full_key}']")

    storage[full_key] = result
```

**Step 4: Run tests to verify they pass**

Run: `conda run -n archetype pytest tests/test_core/test_feature_utils.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/peach/_core/utils/feature_utils.py tests/test_core/test_feature_utils.py
git commit -m "feat: add shared feature utilities for v0.5.0 modules"
```

---

### Task 1.2: Resampling Infrastructure (`resampling.py`)

**Files:**
- Create: `src/peach/_core/utils/resampling.py`
- Test: `tests/test_core/test_resampling.py`

**Step 1: Write failing tests**

```python
# tests/test_core/test_resampling.py
import numpy as np
import pytest


class TestPermutationTest:
    """Test generic permutation testing framework."""

    def test_significant_signal_detected(self):
        """Known signal should yield p < 0.05."""
        from peach._core.utils.resampling import permutation_test

        rng = np.random.default_rng(42)
        x = rng.normal(5.0, 1.0, size=200)  # strong signal away from 0

        def fit_fn(data):
            return np.mean(data)

        def stat_fn(model):
            return abs(model)

        def shuffle_fn(data, rng):
            return rng.choice([-1, 1], size=len(data)) * data  # sign-flip null

        result = permutation_test(
            fit_fn, stat_fn, x, shuffle_fn=shuffle_fn, n_permutations=199, seed=42
        )
        assert result["p_value"] < 0.05
        assert result["observed_stat"] > 4.0
        assert len(result["null_distribution"]) == 199

    def test_null_signal_not_detected(self):
        """No signal should yield p > 0.05."""
        from peach._core.utils.resampling import permutation_test

        rng = np.random.default_rng(42)
        x = rng.normal(0.0, 1.0, size=200)

        def fit_fn(data):
            return np.mean(data)

        def stat_fn(model):
            return abs(model)

        def shuffle_fn(data, rng):
            return rng.choice([-1, 1], size=len(data)) * data

        result = permutation_test(
            fit_fn, stat_fn, x, shuffle_fn=shuffle_fn, n_permutations=199, seed=42
        )
        assert result["p_value"] > 0.05


class TestBootstrapCI:
    """Test generic bootstrap CI framework."""

    def test_ci_covers_true_mean(self):
        """95% CI should cover the true mean for normal data."""
        from peach._core.utils.resampling import bootstrap_ci

        rng = np.random.default_rng(42)
        x = rng.normal(5.0, 1.0, size=500)

        def fit_fn(data):
            return np.mean(data)

        def stat_fn(model):
            return model

        result = bootstrap_ci(fit_fn, stat_fn, x, n_bootstrap=1000, seed=42)
        assert result["ci_lower"] < 5.0 < result["ci_upper"]
        assert len(result["bootstrap_distribution"]) == 1000

    def test_narrow_ci_with_low_variance(self):
        """Low-variance data should give narrow CIs."""
        from peach._core.utils.resampling import bootstrap_ci

        x = np.full(500, 3.0) + np.random.default_rng(42).normal(0, 0.01, 500)

        def fit_fn(data):
            return np.mean(data)

        def stat_fn(model):
            return model

        result = bootstrap_ci(fit_fn, stat_fn, x, n_bootstrap=500, seed=42)
        ci_width = result["ci_upper"] - result["ci_lower"]
        assert ci_width < 0.1

    def test_custom_ci_level(self):
        """90% CI should be narrower than 95% CI."""
        from peach._core.utils.resampling import bootstrap_ci

        rng = np.random.default_rng(42)
        x = rng.normal(0, 1.0, size=500)

        def fit_fn(data):
            return np.mean(data)

        def stat_fn(model):
            return model

        r95 = bootstrap_ci(fit_fn, stat_fn, x, ci_level=0.95, n_bootstrap=500, seed=42)
        r90 = bootstrap_ci(fit_fn, stat_fn, x, ci_level=0.90, n_bootstrap=500, seed=42)
        w95 = r95["ci_upper"] - r95["ci_lower"]
        w90 = r90["ci_upper"] - r90["ci_lower"]
        assert w90 < w95
```

**Step 2: Run tests to verify they fail**

Run: `conda run -n archetype pytest tests/test_core/test_resampling.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

```python
# src/peach/_core/utils/resampling.py
"""Shared permutation testing and bootstrap CI infrastructure."""

import numpy as np
from typing import Any, Callable


def permutation_test(
    fit_fn: Callable,
    stat_fn: Callable,
    data: Any,
    *,
    shuffle_fn: Callable,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """Generic permutation test.

    Parameters
    ----------
    fit_fn : callable
        fit_fn(data) -> model
    stat_fn : callable
        stat_fn(model) -> scalar test statistic
    data : Any
        Input data (array, tuple, etc.)
    shuffle_fn : callable
        shuffle_fn(data, rng) -> permuted data
    n_permutations : int
        Number of permutations.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: observed_stat, null_distribution, p_value
    """
    rng = np.random.default_rng(seed)
    observed_model = fit_fn(data)
    observed_stat = stat_fn(observed_model)

    null_distribution = np.empty(n_permutations)
    for i in range(n_permutations):
        perm_data = shuffle_fn(data, rng)
        perm_model = fit_fn(perm_data)
        null_distribution[i] = stat_fn(perm_model)

    # Two-sided p-value: fraction of null >= observed
    p_value = (np.sum(null_distribution >= observed_stat) + 1) / (n_permutations + 1)

    return {
        "observed_stat": observed_stat,
        "null_distribution": null_distribution,
        "p_value": p_value,
    }


def bootstrap_ci(
    fit_fn: Callable,
    stat_fn: Callable,
    data: Any,
    *,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Generic bootstrap confidence interval.

    Parameters
    ----------
    fit_fn : callable
        fit_fn(data) -> model
    stat_fn : callable
        stat_fn(model) -> scalar statistic
    data : Any
        Input data (must support integer indexing for resampling).
    n_bootstrap : int
        Number of bootstrap samples.
    ci_level : float
        Confidence level (0, 1).
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: point_estimate, ci_lower, ci_upper, bootstrap_distribution
    """
    rng = np.random.default_rng(seed)
    data_arr = np.asarray(data)
    n = len(data_arr)

    point_model = fit_fn(data_arr)
    point_estimate = stat_fn(point_model)

    bootstrap_distribution = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_sample = data_arr[idx]
        boot_model = fit_fn(boot_sample)
        bootstrap_distribution[i] = stat_fn(boot_model)

    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_distribution, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_distribution, 100 * (1 - alpha / 2))

    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_distribution": bootstrap_distribution,
    }
```

**Step 4: Run tests to verify they pass**

Run: `conda run -n archetype pytest tests/test_core/test_resampling.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/peach/_core/utils/resampling.py tests/test_core/test_resampling.py
git commit -m "feat: add shared resampling infrastructure (permutation + bootstrap)"
```

---

## Phase 2: Simplex Regression Core Engine

### Task 2.1: Scheffe Design Matrix + OLS Engine

**Files:**
- Create: `src/peach/_core/utils/simplex_regression.py`
- Test: `tests/test_core/test_simplex_regression.py`

**Step 1: Write failing tests**

```python
# tests/test_core/test_simplex_regression.py
import numpy as np
import pytest
import scipy.sparse as sp


class TestScheffeDesignMatrix:
    """Test Scheffe polynomial design matrix construction."""

    def test_degree1_is_weights(self):
        """Degree 1 design matrix = weights themselves (no intercept)."""
        from peach._core.utils.simplex_regression import scheffe_design_matrix

        W = np.random.dirichlet([1, 1, 1], size=100)
        X1, pairs = scheffe_design_matrix(W, degree=1)
        np.testing.assert_array_almost_equal(X1, W)
        assert pairs == []

    def test_degree2_adds_interactions(self):
        """Degree 2 appends K-choose-2 interaction columns."""
        from peach._core.utils.simplex_regression import scheffe_design_matrix

        K = 4
        W = np.random.dirichlet([1] * K, size=100)
        X2, pairs = scheffe_design_matrix(W, degree=2)
        n_interactions = K * (K - 1) // 2  # 6
        assert X2.shape == (100, K + n_interactions)
        assert len(pairs) == n_interactions
        # First K columns are weights
        np.testing.assert_array_almost_equal(X2[:, :K], W)
        # Interaction columns are products
        for idx, (j, k) in enumerate(pairs):
            np.testing.assert_array_almost_equal(
                X2[:, K + idx], W[:, j] * W[:, k]
            )

    def test_degree1_no_pairs(self):
        from peach._core.utils.simplex_regression import scheffe_design_matrix

        W = np.random.dirichlet([1, 1], size=50)
        _, pairs = scheffe_design_matrix(W, degree=1)
        assert pairs == []


class TestOLSFit:
    """Test OLS regression engine with known coefficients."""

    def test_recovers_known_coefficients_dense(self):
        """OLS recovers planted beta from noiseless simplex regression."""
        from peach._core.utils.simplex_regression import ols_fit

        rng = np.random.default_rng(42)
        K = 3
        W = np.random.dirichlet([1] * K, size=500)
        true_beta = np.array([[10.0, 2.0, 5.0]])  # 1 feature, 3 archetypes
        Y = W @ true_beta.T  # [500, 1]

        result = ols_fit(W, Y)
        np.testing.assert_array_almost_equal(result["coefficients"], true_beta, decimal=5)
        assert result["r_squared"][0] > 0.99

    def test_recovers_with_noise(self):
        """OLS R^2 is reasonable with moderate noise."""
        from peach._core.utils.simplex_regression import ols_fit

        rng = np.random.default_rng(42)
        K = 4
        n = 1000
        W = np.random.dirichlet([1] * K, size=n)
        true_beta = rng.standard_normal((5, K)) * 3
        noise = rng.normal(0, 0.5, size=(n, 5))
        Y = W @ true_beta.T + noise

        result = ols_fit(W, Y)
        assert result["coefficients"].shape == (5, K)
        assert all(r2 > 0.5 for r2 in result["r_squared"])

    def test_sparse_feature_matrix(self):
        """OLS handles sparse Y (densifies per-feature)."""
        from peach._core.utils.simplex_regression import ols_fit

        K = 3
        W = np.random.dirichlet([1] * K, size=200)
        Y_dense = np.random.rand(200, 10)
        Y_sparse = sp.csr_matrix(Y_dense)

        result_dense = ols_fit(W, Y_dense)
        result_sparse = ols_fit(W, Y_sparse)
        np.testing.assert_array_almost_equal(
            result_dense["coefficients"], result_sparse["coefficients"], decimal=5
        )


class TestHC3StandardErrors:
    """Test heteroscedasticity-consistent standard errors."""

    def test_hc3_se_shape(self):
        """HC3 SEs have correct shape [n_features, p]."""
        from peach._core.utils.simplex_regression import ols_fit

        K = 3
        W = np.random.dirichlet([1] * K, size=300)
        Y = np.random.rand(300, 10)

        result = ols_fit(W, Y, robust_se=True)
        assert result["standard_errors"].shape == (10, K)
        assert np.all(result["standard_errors"] > 0)

    def test_hc3_se_heteroscedastic_data(self):
        """HC3 SEs should be larger than OLS SEs with heteroscedastic noise."""
        from peach._core.utils.simplex_regression import ols_fit

        rng = np.random.default_rng(42)
        K = 3
        n = 500
        W = np.random.dirichlet([1] * K, size=n)
        true_beta = np.array([[10.0, 2.0, 5.0]])
        # Heteroscedastic noise: variance proportional to w_0
        noise = rng.normal(0, 1, size=(n, 1)) * W[:, 0:1] * 5
        Y = W @ true_beta.T + noise

        result_hc3 = ols_fit(W, Y, robust_se=True)
        result_ols = ols_fit(W, Y, robust_se=False)
        # HC3 SEs should generally be larger for heteroscedastic data
        assert np.mean(result_hc3["standard_errors"]) > np.mean(result_ols["standard_errors"]) * 0.8


class TestFTest:
    """Test overall model F-test."""

    def test_significant_model(self):
        """Strong signal -> small p-value."""
        from peach._core.utils.simplex_regression import ols_fit

        K = 3
        W = np.random.dirichlet([1] * K, size=500)
        true_beta = np.array([[10.0, 0.0, 0.0]])  # strong archetype-specific
        Y = W @ true_beta.T

        result = ols_fit(W, Y)
        assert result["f_pvalues"][0] < 1e-10

    def test_null_model(self):
        """Pure noise -> large p-value."""
        from peach._core.utils.simplex_regression import ols_fit

        K = 3
        W = np.random.dirichlet([1] * K, size=500)
        Y = np.random.default_rng(42).normal(5.0, 0.01, size=(500, 1))  # constant

        result = ols_fit(W, Y)
        assert result["f_pvalues"][0] > 0.01
```

**Step 2: Run tests to verify they fail**

Run: `conda run -n archetype pytest tests/test_core/test_simplex_regression.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

```python
# src/peach/_core/utils/simplex_regression.py
"""Simplex regression engine: Scheffe polynomials with OLS, HC3 SEs, F-tests."""

import numpy as np
import scipy.sparse as sp
from itertools import combinations
from scipy import stats


def scheffe_design_matrix(W, degree=1):
    """Build Scheffe polynomial design matrix from simplex weights.

    Parameters
    ----------
    W : np.ndarray
        Archetype weights [n_cells, K], rows sum to 1.
    degree : int
        1 = linear (weights only), 2 = with pairwise interactions.

    Returns
    -------
    tuple[np.ndarray, list[tuple]]
        Design matrix [n_cells, p] and list of interaction pairs (empty for degree=1).
    """
    if degree == 1:
        return W.copy(), []

    K = W.shape[1]
    pairs = list(combinations(range(K), 2))
    interactions = np.column_stack([W[:, j] * W[:, k] for j, k in pairs])
    X = np.column_stack([W, interactions])
    return X, pairs


def ols_fit(W, Y, robust_se=True):
    """Vectorized OLS: regress each feature on design matrix W (no intercept).

    Parameters
    ----------
    W : np.ndarray
        Design matrix [n_cells, p]. For Scheffe: p = K or K + K-choose-2.
    Y : np.ndarray or scipy.sparse matrix
        Feature matrix [n_cells, n_features]. Sparse is densified per-feature.
    robust_se : bool
        If True, compute HC3 heteroscedasticity-consistent standard errors.

    Returns
    -------
    dict
        coefficients: [n_features, p]
        r_squared: [n_features]
        residuals: [n_cells, n_features]
        standard_errors: [n_features, p] (HC3 if robust_se, else classical)
        t_statistics: [n_features, p]
        t_pvalues: [n_features, p]
        f_statistics: [n_features]
        f_pvalues: [n_features]
    """
    n, p = W.shape

    # Dense Y for regression
    if sp.issparse(Y):
        Y_dense = Y.toarray()
    else:
        Y_dense = np.asarray(Y)

    n_features = Y_dense.shape[1]

    # OLS: beta = (W'W)^{-1} W'Y
    WtW = W.T @ W  # [p, p]
    WtW_inv = np.linalg.inv(WtW)  # [p, p]
    beta = WtW_inv @ (W.T @ Y_dense)  # [p, n_features]
    beta = beta.T  # [n_features, p]

    # Residuals
    Y_hat = W @ beta.T  # [n, n_features]
    residuals = Y_dense - Y_hat  # [n, n_features]

    # R-squared (no-intercept version: 1 - SS_res / SS_total_from_zero)
    # For Scheffe on simplex, total SS is from zero since no intercept
    ss_res = np.sum(residuals ** 2, axis=0)  # [n_features]
    y_mean = Y_dense.mean(axis=0, keepdims=True)
    ss_tot = np.sum((Y_dense - y_mean) ** 2, axis=0)  # [n_features]
    r_squared = np.where(ss_tot > 0, 1 - ss_res / ss_tot, 0.0)

    # Standard errors
    if robust_se:
        # HC3: hat matrix diagonal only
        H_diag = np.sum((W @ WtW_inv) * W, axis=1)  # [n]
        se = _hc3_standard_errors(W, residuals, WtW_inv, H_diag)
    else:
        # Classical OLS SEs
        sigma2 = ss_res / max(n - p, 1)  # [n_features]
        # Var(beta) = sigma^2 * (W'W)^{-1}
        var_diag = np.diag(WtW_inv)  # [p]
        se = np.sqrt(np.outer(sigma2, var_diag))  # [n_features, p]

    # t-statistics and p-values
    t_stats = np.where(se > 0, beta / se, 0.0)
    df = max(n - p, 1)
    t_pvalues = 2 * stats.t.sf(np.abs(t_stats), df=df)

    # F-test (overall model significance)
    # F = (R^2 / p) / ((1 - R^2) / (n - p))
    # For features with R^2 = 1.0 or ss_tot = 0, set F to 0 and p to 1
    f_stats = np.zeros(n_features)
    f_pvalues = np.ones(n_features)
    valid = (r_squared > 0) & (r_squared < 1.0) & (ss_tot > 0)
    if np.any(valid):
        f_stats[valid] = (r_squared[valid] / p) / ((1 - r_squared[valid]) / max(n - p, 1))
        f_pvalues[valid] = stats.f.sf(f_stats[valid], dfn=p, dfd=max(n - p, 1))

    return {
        "coefficients": beta,
        "r_squared": r_squared,
        "residuals": residuals,
        "standard_errors": se,
        "t_statistics": t_stats,
        "t_pvalues": t_pvalues,
        "f_statistics": f_stats,
        "f_pvalues": f_pvalues,
    }


def _hc3_standard_errors(W, residuals, WtW_inv, H_diag):
    """Compute HC3 heteroscedasticity-consistent standard errors.

    HC3: Var(beta) = (W'W)^{-1} (sum_i w_i w_i' e_i^2 / (1-h_ii)^2) (W'W)^{-1}

    Parameters
    ----------
    W : np.ndarray [n, p]
    residuals : np.ndarray [n, n_features]
    WtW_inv : np.ndarray [p, p]
    H_diag : np.ndarray [n]

    Returns
    -------
    np.ndarray [n_features, p]
        Standard errors for each coefficient of each feature.
    """
    n, p = W.shape
    n_features = residuals.shape[1]

    # Adjusted residuals: e_i / (1 - h_ii)
    adjustment = 1.0 / (1 - H_diag)  # [n]
    adjustment = np.clip(adjustment, 0, 1e6)  # prevent division by zero at high-leverage points

    se = np.empty((n_features, p))
    for g in range(n_features):
        e_adj = residuals[:, g] * adjustment  # [n]
        # Meat: sum_i w_i w_i' * (e_i / (1-h_ii))^2
        We = W * (e_adj ** 2)[:, np.newaxis]  # [n, p]
        meat = W.T @ We  # [p, p]
        # Sandwich: (W'W)^{-1} meat (W'W)^{-1}
        sandwich = WtW_inv @ meat @ WtW_inv
        se[g] = np.sqrt(np.maximum(np.diag(sandwich), 0))

    return se
```

**Step 4: Run tests to verify they pass**

Run: `conda run -n archetype pytest tests/test_core/test_simplex_regression.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/peach/_core/utils/simplex_regression.py tests/test_core/test_simplex_regression.py
git commit -m "feat: add simplex regression OLS engine with Scheffe design + HC3 SEs"
```

---

### Task 2.2: Simplex Regression Public API + Pydantic Type

**Files:**
- Modify: `src/peach/_core/types.py` (append SimplexRegressionResult)
- Create: `src/peach/tl/feature_regression.py`
- Modify: `src/peach/tl/__init__.py` (add imports)
- Test: `tests/test_statistical/test_simplex_regression_api.py`

**Step 1: Write failing test**

```python
# tests/test_statistical/test_simplex_regression_api.py
import numpy as np
import pytest
from anndata import AnnData


@pytest.fixture
def regression_adata():
    """AnnData with known archetypal structure for regression testing."""
    rng = np.random.default_rng(42)
    K = 3
    n = 500
    n_genes = 50

    # Simplex weights
    weights = rng.dirichlet([1] * K, size=n)

    # Gene expression: known coefficients
    true_beta = rng.standard_normal((n_genes, K)) * 5
    # Gene 0: archetype-exclusive (high for arch 0 only)
    true_beta[0] = [10.0, 0.0, 0.0]
    # Gene 1: flat (same across all)
    true_beta[1] = [3.0, 3.0, 3.0]
    # Gene 2: gradient
    true_beta[2] = [8.0, 4.0, 1.0]

    noise = rng.normal(0, 0.3, size=(n, n_genes))
    X = weights @ true_beta.T + noise

    adata = AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obsm["cell_archetype_weights"] = weights
    return adata


class TestFeatureSimplex Regression:
    """Test pc.tl.feature_simplex_regression() public API."""

    def test_basic_run(self, regression_adata):
        """Runs without error, stores result in adata.uns."""
        import peach as pc

        result = pc.tl.feature_simplex_regression(regression_adata, n_bootstrap=0)
        assert "peach_simplex_regression" in regression_adata.uns
        assert result is not None
        assert result.vertex_coefficients.shape == (50, 3)
        assert len(result.r_squared_degree1) == 50

    def test_recovers_exclusive_gene(self, regression_adata):
        """Gene 0 (archetype-exclusive) should have high beta_0, low others."""
        import peach as pc

        result = pc.tl.feature_simplex_regression(regression_adata, n_bootstrap=0)
        gene0_betas = result.vertex_coefficients[0]
        assert gene0_betas[0] > 8.0
        assert gene0_betas[1] < 2.0
        assert gene0_betas[2] < 2.0

    def test_flat_gene_low_r2(self, regression_adata):
        """Gene 1 (flat) should have low R^2."""
        import peach as pc

        result = pc.tl.feature_simplex_regression(regression_adata, n_bootstrap=0)
        assert result.r_squared_degree1[1] < 0.1

    def test_degree2_adds_interactions(self, regression_adata):
        """max_degree=2 produces interaction coefficients."""
        import peach as pc

        result = pc.tl.feature_simplex_regression(
            regression_adata, max_degree=2, n_bootstrap=0
        )
        assert result.interaction_coefficients is not None
        K = 3
        n_interactions = K * (K - 1) // 2
        assert result.interaction_coefficients.shape == (50, n_interactions)

    def test_residuals_stored(self, regression_adata):
        """store_residuals=True puts residuals in obsm."""
        import peach as pc

        pc.tl.feature_simplex_regression(
            regression_adata, store_residuals=True, n_bootstrap=0
        )
        assert "peach_residuals" in regression_adata.obsm
        assert regression_adata.obsm["peach_residuals"].shape == (500, 50)

    def test_bootstrap_cis(self, regression_adata):
        """Bootstrap CIs are computed when n_bootstrap > 0."""
        import peach as pc

        result = pc.tl.feature_simplex_regression(
            regression_adata, n_bootstrap=50  # small for speed
        )
        assert result.vertex_ci_lower is not None
        assert result.vertex_ci_upper is not None
        assert result.vertex_ci_lower.shape == (50, 3)
        # CIs should bracket the point estimate
        assert np.all(result.vertex_ci_lower <= result.vertex_coefficients + 1e-6)
        assert np.all(result.vertex_ci_upper >= result.vertex_coefficients - 1e-6)

    def test_fdr_correction(self, regression_adata):
        """F-test p-values are FDR-corrected."""
        import peach as pc

        result = pc.tl.feature_simplex_regression(regression_adata, n_bootstrap=0)
        assert hasattr(result, "f_pvalue_fdr")
        assert len(result.f_pvalue_fdr) == 50
        # FDR-corrected p-values should be >= raw p-values
        assert np.all(result.f_pvalue_fdr >= result.f_pvalue - 1e-10)

    def test_convenience_gene_wrapper(self, regression_adata):
        """pc.tl.gene_simplex_regression() is a convenience wrapper."""
        import peach as pc

        result = pc.tl.gene_simplex_regression(regression_adata, n_bootstrap=0)
        assert result.vertex_coefficients.shape[0] == 50
```

**Step 2: Run tests to verify they fail**

Run: `conda run -n archetype pytest tests/test_statistical/test_simplex_regression_api.py -v`
Expected: FAIL

**Step 3: Write Pydantic type (append to types.py)**

Add to `src/peach/_core/types.py`:

```python
class SimplexRegressionResult(BaseModel):
    """Result of simplex regression (Scheffe polynomial) on archetype weights."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    feature_names: list[str]
    archetype_names: list[str]
    n_cells: int
    n_features: int
    n_archetypes: int

    # Degree 1 (linear)
    vertex_coefficients: np.ndarray  # [n_features, K]
    r_squared_degree1: np.ndarray  # [n_features]
    f_pvalue: np.ndarray  # [n_features] raw
    f_pvalue_fdr: np.ndarray  # [n_features] BH-corrected
    vertex_pvalues: np.ndarray  # [n_features, K]
    vertex_se: np.ndarray  # [n_features, K]

    # Degree 2 (interactions) — None if max_degree=1
    interaction_coefficients: np.ndarray | None = None  # [n_features, K-choose-2]
    interaction_pairs: list[tuple] | None = None
    interaction_pvalues: np.ndarray | None = None
    interaction_se: np.ndarray | None = None
    r_squared_degree2: np.ndarray | None = None

    # Bootstrap CIs — None if n_bootstrap=0
    vertex_ci_lower: np.ndarray | None = None  # [n_features, K]
    vertex_ci_upper: np.ndarray | None = None
    interaction_ci_lower: np.ndarray | None = None
    interaction_ci_upper: np.ndarray | None = None

    def to_serializable(self) -> dict:
        """Convert to h5ad-safe dict for adata.uns storage."""
        d = {}
        for field_name, value in self:
            if value is None:
                continue
            if isinstance(value, np.ndarray):
                d[field_name] = value
            elif isinstance(value, list):
                d[field_name] = value
            else:
                d[field_name] = value
        return d
```

**Step 4: Write public API**

```python
# src/peach/tl/feature_regression.py
"""Simplex regression and archetype driver regression public API."""

import numpy as np
from anndata import AnnData
from statsmodels.stats.multitest import multipletests

from peach._core.utils.feature_utils import (
    get_archetype_weights,
    resolve_features,
    store_result,
)
from peach._core.utils.simplex_regression import ols_fit, scheffe_design_matrix
from peach._core.types import SimplexRegressionResult


def feature_simplex_regression(
    adata: AnnData,
    *,
    feature_matrix=None,
    feature_names=None,
    max_degree: int = 2,
    permutation_test: bool = False,
    n_permutations: int = 1000,
    n_bootstrap: int = 1000,
    robust_se: bool = True,
    store_residuals: bool = True,
    copy: bool = False,
) -> SimplexRegressionResult:
    """Simplex regression of features on archetype weights (Scheffe polynomials).

    Parameters
    ----------
    adata : AnnData
        Must have archetype weights in obsm['cell_archetype_weights'].
    feature_matrix : None, str, or array-like
        Feature matrix to regress. None = adata.X.
    feature_names : list[str] or None
        Feature names. Inferred if None.
    max_degree : int
        1 = linear only, 2 = with pairwise interactions. Both degrees reported.
    permutation_test : bool
        If True, run permutation test for model significance.
    n_permutations : int
        Number of permutations (if permutation_test=True).
    n_bootstrap : int
        Number of bootstrap samples for CIs. 0 to disable.
    robust_se : bool
        If True, use HC3 heteroscedasticity-consistent SEs.
    store_residuals : bool
        If True, store residual matrix in adata.obsm['peach_residuals'].
    copy : bool
        If True, operate on a copy of adata.

    Returns
    -------
    SimplexRegressionResult
        Also stored in adata.uns['peach_simplex_regression'].
    """
    if copy:
        adata = adata.copy()

    weights = get_archetype_weights(adata)
    Y, feat_names = resolve_features(adata, feature_matrix, feature_names)
    K = weights.shape[1]
    n_cells = adata.n_obs
    n_features = len(feat_names)
    archetype_names = [f"archetype_{i}" for i in range(K)]

    # Degree 1
    W1, _ = scheffe_design_matrix(weights, degree=1)
    result1 = ols_fit(W1, Y, robust_se=robust_se)

    # FDR correction on F-test
    _, f_pvalue_fdr, _, _ = multipletests(result1["f_pvalues"], method="fdr_bh")

    # Degree 2 (if requested)
    interaction_coefficients = None
    interaction_pairs = None
    interaction_pvalues = None
    interaction_se = None
    r_squared_degree2 = None

    if max_degree >= 2:
        W2, pairs = scheffe_design_matrix(weights, degree=2)
        result2 = ols_fit(W2, Y, robust_se=robust_se)
        interaction_coefficients = result2["coefficients"][:, K:]
        interaction_pairs = pairs
        interaction_pvalues = result2["t_pvalues"][:, K:]
        interaction_se = result2["standard_errors"][:, K:]
        r_squared_degree2 = result2["r_squared"]

    # Bootstrap CIs
    vertex_ci_lower = None
    vertex_ci_upper = None
    interaction_ci_lower = None
    interaction_ci_upper = None

    if n_bootstrap > 0:
        vertex_ci_lower, vertex_ci_upper = _bootstrap_regression_cis(
            weights, Y, degree=1, n_bootstrap=n_bootstrap, K=K
        )
        if max_degree >= 2:
            int_ci_lo, int_ci_hi = _bootstrap_regression_cis(
                weights, Y, degree=2, n_bootstrap=n_bootstrap, K=K
            )
            interaction_ci_lower = int_ci_lo[:, K:]
            interaction_ci_upper = int_ci_hi[:, K:]
            vertex_ci_lower_d2 = int_ci_lo[:, :K]  # noqa: F841
            vertex_ci_upper_d2 = int_ci_hi[:, :K]  # noqa: F841

    # Store residuals
    if store_residuals:
        store_result(adata, "residuals", result1["residuals"], domain="obsm")

    # Build result
    result = SimplexRegressionResult(
        feature_names=feat_names,
        archetype_names=archetype_names,
        n_cells=n_cells,
        n_features=n_features,
        n_archetypes=K,
        vertex_coefficients=result1["coefficients"],
        r_squared_degree1=result1["r_squared"],
        f_pvalue=result1["f_pvalues"],
        f_pvalue_fdr=f_pvalue_fdr,
        vertex_pvalues=result1["t_pvalues"],
        vertex_se=result1["standard_errors"],
        interaction_coefficients=interaction_coefficients,
        interaction_pairs=interaction_pairs,
        interaction_pvalues=interaction_pvalues,
        interaction_se=interaction_se,
        r_squared_degree2=r_squared_degree2,
        vertex_ci_lower=vertex_ci_lower,
        vertex_ci_upper=vertex_ci_upper,
        interaction_ci_lower=interaction_ci_lower,
        interaction_ci_upper=interaction_ci_upper,
    )

    # Store serializable summary
    store_result(adata, "simplex_regression", result.to_serializable())

    return result


def gene_simplex_regression(adata: AnnData, **kwargs) -> SimplexRegressionResult:
    """Convenience: simplex regression on adata.X (gene expression)."""
    return feature_simplex_regression(adata, feature_matrix=None, **kwargs)


def pathway_simplex_regression(adata: AnnData, **kwargs) -> SimplexRegressionResult:
    """Convenience: simplex regression on adata.obsm['pathway_scores']."""
    return feature_simplex_regression(
        adata, feature_matrix="pathway_scores", **kwargs
    )


def _bootstrap_regression_cis(weights, Y, degree, n_bootstrap, K, ci_level=0.95, seed=42):
    """Bootstrap CIs for regression coefficients.

    Returns (ci_lower, ci_upper), each [n_features, p].
    """
    rng = np.random.default_rng(seed)
    n = weights.shape[0]

    W_design, _ = scheffe_design_matrix(weights, degree=degree)
    p = W_design.shape[1]
    n_features = Y.shape[1] if not hasattr(Y, 'toarray') else Y.shape[1]

    boot_coefs = np.empty((n_bootstrap, n_features, p))
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        W_boot = W_design[idx]
        if hasattr(Y, 'toarray'):
            Y_boot = Y[idx].toarray()
        else:
            Y_boot = np.asarray(Y)[idx]
        result = ols_fit(W_boot, Y_boot, robust_se=False)
        boot_coefs[b] = result["coefficients"]

    alpha = 1 - ci_level
    ci_lower = np.percentile(boot_coefs, 100 * alpha / 2, axis=0)
    ci_upper = np.percentile(boot_coefs, 100 * (1 - alpha / 2), axis=0)
    return ci_lower, ci_upper
```

**Step 5: Add to `tl/__init__.py`**

Add these imports and `__all__` entries:
```python
from .feature_regression import (
    feature_simplex_regression,
    gene_simplex_regression,
    pathway_simplex_regression,
)
```

**Step 6: Run tests to verify they pass**

Run: `conda run -n archetype pytest tests/test_statistical/test_simplex_regression_api.py -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add src/peach/_core/types.py src/peach/tl/feature_regression.py \
  src/peach/tl/__init__.py tests/test_statistical/test_simplex_regression_api.py
git commit -m "feat: add simplex regression public API with Scheffe polynomials"
```

---

## Phase 3: Pattern Classification

### Task 3.1: Classification Engine

**Files:**
- Create: `src/peach/_core/utils/pattern_classification.py`
- Test: `tests/test_statistical/test_pattern_classification.py`

**Step 1: Write failing tests**

```python
# tests/test_statistical/test_pattern_classification.py
import numpy as np
import pytest


class TestClassifyPatterns:
    """Test pattern classification from regression coefficients."""

    def test_exclusive_pattern(self):
        """One high beta, others near zero -> archetype-exclusive."""
        from peach._core.utils.pattern_classification import classify_single_feature

        betas = np.array([10.0, 0.5, 0.3])
        interactions = None
        r2 = 0.8
        p_betas = np.array([1e-10, 0.5, 0.7])
        p_interactions = None
        result = classify_single_feature(betas, interactions, r2, p_betas, p_interactions)
        assert result["pattern"] == "archetype-exclusive"

    def test_flat_pattern_low_r2(self):
        """Low R^2 -> flat/ubiquitous."""
        from peach._core.utils.pattern_classification import classify_single_feature

        betas = np.array([3.0, 3.1, 2.9])
        result = classify_single_feature(
            betas, None, r2=0.01, p_betas=np.array([0.5, 0.5, 0.5]),
            p_interactions=None, r2_threshold=0.05
        )
        assert result["pattern"] == "flat"

    def test_gradient_pattern(self):
        """Ordered coefficients, high R^2 -> monotonic gradient."""
        from peach._core.utils.pattern_classification import classify_single_feature

        betas = np.array([8.0, 4.0, 1.0])
        result = classify_single_feature(
            betas, None, r2=0.7,
            p_betas=np.array([1e-10, 1e-5, 0.01]),
            p_interactions=None
        )
        assert result["pattern"] == "monotonic-gradient"

    def test_shared_pattern(self):
        """2+ elevated betas -> multi-archetype shared."""
        from peach._core.utils.pattern_classification import classify_single_feature

        betas = np.array([8.0, 7.5, 1.0])
        result = classify_single_feature(
            betas, None, r2=0.6,
            p_betas=np.array([1e-10, 1e-10, 0.3]),
            p_interactions=None
        )
        assert result["pattern"] == "multi-archetype-shared"

    def test_ridge_pattern(self):
        """Positive significant interaction -> ridge/blend-enriched."""
        from peach._core.utils.pattern_classification import classify_single_feature

        betas = np.array([5.0, 5.0, 1.0])
        interactions = np.array([3.0, 0.1, 0.1])  # strong (0,1) interaction
        result = classify_single_feature(
            betas, interactions, r2=0.7,
            p_betas=np.array([1e-5, 1e-5, 0.3]),
            p_interactions=np.array([1e-5, 0.5, 0.5])
        )
        assert result["pattern"] == "ridge"

    def test_valley_pattern(self):
        """Negative significant interaction -> valley/blend-depleted."""
        from peach._core.utils.pattern_classification import classify_single_feature

        betas = np.array([5.0, 5.0, 1.0])
        interactions = np.array([-3.0, 0.1, 0.1])
        result = classify_single_feature(
            betas, interactions, r2=0.7,
            p_betas=np.array([1e-5, 1e-5, 0.3]),
            p_interactions=np.array([1e-5, 0.5, 0.5])
        )
        assert result["pattern"] == "valley"

    def test_antagonistic_pattern(self):
        """High spread, some high some low -> antagonistic."""
        from peach._core.utils.pattern_classification import classify_single_feature

        betas = np.array([10.0, -2.0, 8.0, -3.0])
        result = classify_single_feature(
            betas, None, r2=0.7,
            p_betas=np.array([1e-10, 1e-5, 1e-10, 1e-5]),
            p_interactions=None
        )
        assert result["pattern"] == "antagonistic"
```

**Step 2: Run tests to verify they fail**

Run: `conda run -n archetype pytest tests/test_statistical/test_pattern_classification.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

```python
# src/peach/_core/utils/pattern_classification.py
"""Feature pattern classification from simplex regression coefficients."""

import numpy as np


def classify_single_feature(
    vertex_betas,
    interaction_betas,
    r2,
    p_betas,
    p_interactions,
    r2_threshold=0.05,
    significance_threshold=0.05,
    effect_size_threshold=None,
):
    """Classify a single feature's regression pattern.

    Parameters
    ----------
    vertex_betas : np.ndarray [K]
    interaction_betas : np.ndarray [K-choose-2] or None
    r2 : float
    p_betas : np.ndarray [K]
    p_interactions : np.ndarray [K-choose-2] or None
    r2_threshold : float
    significance_threshold : float
    effect_size_threshold : float or None
        Auto-calibrated from data if None.

    Returns
    -------
    dict with keys: pattern, confidence, details
    """
    K = len(vertex_betas)

    # Auto-calibrate effect size threshold
    if effect_size_threshold is None:
        beta_range = np.ptp(vertex_betas)
        effect_size_threshold = max(beta_range * 0.2, 0.1)

    # Rule 1: Low R^2 -> flat
    if r2 < r2_threshold:
        return {"pattern": "flat", "confidence": 1.0 - r2 / r2_threshold, "details": {"reason": "low_r2"}}

    # Rule 2: Low effect size -> flat
    beta_range = np.ptp(vertex_betas)
    if beta_range < effect_size_threshold and not _has_significant_interactions(
        interaction_betas, p_interactions, significance_threshold
    ):
        return {"pattern": "flat", "confidence": 0.8, "details": {"reason": "low_effect_size"}}

    # Rule 3: Significant interactions -> ridge or valley
    if _has_significant_interactions(interaction_betas, p_interactions, significance_threshold):
        sig_mask = p_interactions < significance_threshold
        sig_interactions = interaction_betas[sig_mask]
        if np.mean(sig_interactions) > 0:
            return {"pattern": "ridge", "confidence": 0.8, "details": {"n_sig_interactions": int(sig_mask.sum())}}
        else:
            return {"pattern": "valley", "confidence": 0.8, "details": {"n_sig_interactions": int(sig_mask.sum())}}

    # Rule 4: Count significantly elevated betas
    sig_betas = p_betas < significance_threshold
    beta_mean = np.mean(vertex_betas)
    elevated = sig_betas & (vertex_betas > beta_mean + effect_size_threshold * 0.5)
    depressed = sig_betas & (vertex_betas < beta_mean - effect_size_threshold * 0.5)
    n_elevated = np.sum(elevated)
    n_depressed = np.sum(depressed)

    # Antagonistic: significant betas on both sides
    if n_elevated >= 1 and n_depressed >= 1 and (n_elevated + n_depressed) >= 3:
        return {"pattern": "antagonistic", "confidence": 0.7, "details": {"n_elevated": int(n_elevated), "n_depressed": int(n_depressed)}}

    # Exclusive: exactly 1 elevated
    if n_elevated == 1:
        # Check if it's a gradient (others are ordered) or truly exclusive
        sorted_betas = np.sort(vertex_betas)[::-1]
        ratio = sorted_betas[1] / max(sorted_betas[0], 1e-10)
        if ratio < 0.3:
            return {"pattern": "archetype-exclusive", "confidence": 0.9, "details": {"dominant_archetype": int(np.argmax(vertex_betas))}}
        else:
            return {"pattern": "monotonic-gradient", "confidence": 0.7, "details": {"dominant_archetype": int(np.argmax(vertex_betas))}}

    # Shared: 2+ elevated
    if n_elevated >= 2:
        return {"pattern": "multi-archetype-shared", "confidence": 0.7, "details": {"n_shared": int(n_elevated)}}

    # Gradient: ordered coefficients, 1 dominant
    sorted_betas = np.sort(vertex_betas)[::-1]
    if sorted_betas[0] > sorted_betas[1] * 1.5:
        return {"pattern": "monotonic-gradient", "confidence": 0.6, "details": {"dominant_archetype": int(np.argmax(vertex_betas))}}

    # Default: antagonistic if high spread, else gradient
    if beta_range > effect_size_threshold * 3:
        return {"pattern": "antagonistic", "confidence": 0.5, "details": {}}
    return {"pattern": "monotonic-gradient", "confidence": 0.5, "details": {}}


def classify_all_features(
    vertex_coefficients,
    interaction_coefficients,
    r_squared,
    vertex_pvalues,
    interaction_pvalues,
    r2_threshold=0.05,
    significance_threshold=0.05,
    effect_size_threshold=None,
):
    """Classify all features at once.

    Parameters
    ----------
    vertex_coefficients : np.ndarray [n_features, K]
    interaction_coefficients : np.ndarray [n_features, n_interactions] or None
    r_squared : np.ndarray [n_features]
    vertex_pvalues : np.ndarray [n_features, K]
    interaction_pvalues : np.ndarray [n_features, n_interactions] or None

    Returns
    -------
    list[dict]
        One classification dict per feature.
    """
    n_features = len(r_squared)
    results = []
    for i in range(n_features):
        int_betas = interaction_coefficients[i] if interaction_coefficients is not None else None
        int_pvals = interaction_pvalues[i] if interaction_pvalues is not None else None
        results.append(
            classify_single_feature(
                vertex_coefficients[i],
                int_betas,
                r_squared[i],
                vertex_pvalues[i],
                int_pvals,
                r2_threshold=r2_threshold,
                significance_threshold=significance_threshold,
                effect_size_threshold=effect_size_threshold,
            )
        )
    return results


def _has_significant_interactions(interaction_betas, p_interactions, threshold):
    """Check if any interaction terms are significant."""
    if interaction_betas is None or p_interactions is None:
        return False
    return np.any(p_interactions < threshold)
```

**Step 4: Run tests to verify they pass**

Run: `conda run -n archetype pytest tests/test_statistical/test_pattern_classification.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/peach/_core/utils/pattern_classification.py tests/test_statistical/test_pattern_classification.py
git commit -m "feat: add feature pattern classification engine"
```

---

### Task 3.2: Pattern Classification Public API

**Files:**
- Modify: `src/peach/_core/types.py` (append PatternClassificationResult)
- Create: `src/peach/tl/feature_patterns.py`
- Modify: `src/peach/tl/__init__.py`
- Test: `tests/test_statistical/test_pattern_api.py`

**Step 1: Write failing test**

```python
# tests/test_statistical/test_pattern_api.py
import numpy as np
import pytest
from anndata import AnnData


@pytest.fixture
def classified_adata():
    """AnnData with regression results ready for classification."""
    rng = np.random.default_rng(42)
    K = 3
    n = 500
    n_genes = 20

    weights = rng.dirichlet([1] * K, size=n)
    true_beta = rng.standard_normal((n_genes, K)) * 5
    true_beta[0] = [10.0, 0.0, 0.0]  # exclusive
    true_beta[1] = [3.0, 3.0, 3.0]  # flat
    true_beta[2] = [8.0, 4.0, 1.0]  # gradient

    noise = rng.normal(0, 0.3, size=(n, n_genes))
    X = weights @ true_beta.T + noise

    adata = AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obsm["cell_archetype_weights"] = weights
    return adata


class TestClassifyFeaturePatterns:
    def test_runs_after_regression(self, classified_adata):
        """classify_feature_patterns runs on regression output."""
        import peach as pc

        pc.tl.feature_simplex_regression(classified_adata, n_bootstrap=0)
        result = pc.tl.classify_feature_patterns(classified_adata)
        assert "peach_feature_patterns" in classified_adata.uns
        assert len(result.classifications) == 20

    def test_gene0_classified_exclusive(self, classified_adata):
        """Gene 0 should be classified as archetype-exclusive."""
        import peach as pc

        pc.tl.feature_simplex_regression(classified_adata, n_bootstrap=0)
        result = pc.tl.classify_feature_patterns(classified_adata)
        assert result.classifications[0]["pattern"] == "archetype-exclusive"

    def test_gene1_classified_flat(self, classified_adata):
        """Gene 1 should be classified as flat."""
        import peach as pc

        pc.tl.feature_simplex_regression(classified_adata, n_bootstrap=0)
        result = pc.tl.classify_feature_patterns(classified_adata)
        assert result.classifications[1]["pattern"] == "flat"

    def test_pattern_counts(self, classified_adata):
        """Pattern counts should sum to n_features."""
        import peach as pc

        pc.tl.feature_simplex_regression(classified_adata, n_bootstrap=0)
        result = pc.tl.classify_feature_patterns(classified_adata)
        assert sum(result.pattern_counts.values()) == 20
```

**Step 2-5:** Follow TDD cycle — implement PatternClassificationResult type, `classify_feature_patterns()` in `tl/feature_patterns.py`, add to `tl/__init__.py`, test, commit.

The public API function reads from `adata.uns['peach_simplex_regression']` (or accepts a `SimplexRegressionResult` directly), calls `classify_all_features()`, wraps in Pydantic type, stores in `adata.uns['peach_feature_patterns']`.

**Commit:** `git commit -m "feat: add pattern classification public API"`

---

## Phase 4: Archetype Driver Regression (Flipped)

### Task 4.1: ILR Transform Utilities

**Files:**
- Create: `src/peach/_core/utils/ilr_transform.py`
- Test: `tests/test_core/test_ilr_transform.py`

**Tests to write:**
- `test_ilr_roundtrip`: weights -> ILR -> inverse ILR recovers original weights
- `test_ilr_dimension`: K weights produce K-1 ILR coordinates
- `test_ilr_zero_handling`: epsilon smoothing for near-zero weights
- `test_ilr_preserves_ordering`: relative ordering maintained

**Implementation:** Helmert sub-composition basis using `scipy.special.logsumexp` for numerical stability. Add epsilon=1e-3 before log, renormalize.

**Commit:** `git commit -m "feat: add ILR transform utilities for simplex <-> unconstrained space"`

### Task 4.2: Driver Regression Engine + Public API

**Files:**
- Extend: `src/peach/_core/utils/simplex_regression.py` (add `driver_ols_fit`)
- Modify: `src/peach/_core/types.py` (append DriverRegressionResult)
- Extend: `src/peach/tl/feature_regression.py` (add `archetype_driver_regression`)
- Modify: `src/peach/tl/__init__.py`
- Test: `tests/test_statistical/test_driver_regression.py`

**Tests to write:**
- `test_recovers_known_driver`: planted geneset driving one archetype is recovered
- `test_ilr_space_regression`: K-1 regressions run
- `test_back_transform_to_simplex`: coefficients map back to per-archetype
- `test_max_interaction_features_guard`: error raised when n_features > 50 at degree=2
- `test_intercept_included`: intercept present (features are not compositional)

**Commit:** `git commit -m "feat: add archetype driver regression (flipped) with ILR transform"`

---

## Phase 5: GMM Decomposition

### Task 5.1: Simplex GMM Engine

**Files:**
- Create: `src/peach/_core/utils/simplex_gmm.py`
- Test: `tests/test_core/test_simplex_gmm.py`

**Tests to write:**
- `test_ilr_gmm_recovers_known_components`: 2 planted Gaussian blobs in weight space -> GMM finds 2
- `test_bic_selects_correct_k`: BIC minimum at planted K
- `test_stability_filtering`: unstable components removed
- `test_centroid_back_to_simplex`: ILR centroids map back to valid simplex points
- `test_feature_characterization`: per-component feature means computed correctly

**Implementation:** ILR transform weights, sklearn `GaussianMixture` scan over K..3K, multi-init stability via Hungarian algorithm on centroid distances.

**Commit:** `git commit -m "feat: add simplex GMM decomposition engine"`

### Task 5.2: GMM Public API

**Files:**
- Modify: `src/peach/_core/types.py` (append GMMResult)
- Create: `src/peach/tl/feature_decomposition.py`
- Modify: `src/peach/tl/__init__.py`
- Test: `tests/test_statistical/test_gmm_api.py`

**Tests to write:**
- `test_basic_run`: runs, stores in adata
- `test_component_labels_stored`: adata.obsm['peach_gmm_labels'] populated
- `test_stability_scores_bounded`: all scores in [0, 1]
- `test_component_simplex_means_valid`: centroids sum to 1

**Commit:** `git commit -m "feat: add GMM decomposition public API"`

---

## Phase 6: Flow Matching

### Task 6.1: Velocity Network + FlowModel

**Files:**
- Create: `src/peach/_core/utils/flow_matching.py`
- Test: `tests/test_core/test_flow_matching.py`

**Tests to write:**
- `test_velocity_network_output_shape`: (batch, dim) in -> (batch, dim) out
- `test_flow_model_training_loss_decreases`: loss goes down over epochs
- `test_transport_moves_source_toward_target`: MMD decreases
- `test_velocity_at_shape`: correct output dimensions
- `test_jacobian_shape`: [dim, dim] per point

**Implementation:** VelocityNetwork (MLP, ReLU), FlowModel wrapping fb `flow_matching.ConditionalProbPath`. Optional dep check for `flow_matching` package.

**Commit:** `git commit -m "feat: add flow matching core engine (velocity net + transport)"`

### Task 6.2: Flow Public API (flow_within, flow_between)

**Files:**
- Modify: `src/peach/_core/types.py` (append FlowWithinResult, FlowBetweenResult, GeneAlignmentResult, FlowJacobianResult)
- Create: `src/peach/tl/flow.py`
- Modify: `src/peach/tl/__init__.py`
- Test: `tests/test_integration/test_flow_api.py`

**Tests to write:**
- `test_flow_within_basic`: runs on single adata with obs-defined subsets
- `test_flow_within_mmd_improves`: mmd_after < mmd_before
- `test_flow_between_two_adatas`: concatenates, trains flows
- `test_flow_between_pca_dim_mismatch_errors`: different n_PCs raises ValueError
- `test_gene_alignment_returns_rankings`: top aligned/opposed genes returned
- `test_flow_jacobian_determinant`: volume change computed per-cell
- `test_flow_significance_permutation`: permutation test returns p-value

**Commit:** `git commit -m "feat: add flow matching public API (within, between, gene alignment, Jacobian)"`

---

## Phase 7: Ternary Facets + Regression Visualization

### Task 7.1: Ternary Facet Plots

**Files:**
- Create: `src/peach/pl/ternary.py`
- Modify: `src/peach/pl/__init__.py`
- Test: `tests/test_visualization/test_ternary.py`

**Tests to write:**
- `test_ternary_scatter_returns_figure`: valid plotly figure returned
- `test_ternary_contour_returns_figure`: contour mode works
- `test_ternary_facet_grid`: multiple facets generated
- `test_regression_overlay`: simplex surface overlaid correctly
- `test_invalid_archetypes_raises`: non-existent archetype indices raise error

**Implementation:** Use `python-ternary` if available (optional dep), fallback to custom matplotlib triangle. Renormalize 3 selected archetype weights to sum to 1. Optional `regression_overlay` zero-pads other weights.

**Commit:** `git commit -m "feat: add ternary facet plots for simplex visualization"`

### Task 7.2: Regression Visualization

**Files:**
- Create: `src/peach/pl/regression.py`
- Modify: `src/peach/pl/__init__.py`
- Test: `tests/test_visualization/test_regression_viz.py`

**Functions:** `coefficient_heatmap`, `interaction_heatmap`, `r2_barplot`, `vertex_radar`, `regression_volcano`, `pattern_summary`

**Tests:** Each function returns a plotly Figure and doesn't error on regression_adata.

**Commit:** `git commit -m "feat: add regression visualization plots"`

---

## Phase 8: GMM + Flow Visualization

### Task 8.1: GMM Visualization

**Files:**
- Create: `src/peach/pl/decomposition.py`
- Modify: `src/peach/pl/__init__.py`
- Test: `tests/test_visualization/test_decomposition_viz.py`

**Functions:** `component_scatter`, `gmm_bic_curve`, `component_heatmap`, `component_stability`

**Commit:** `git commit -m "feat: add GMM decomposition visualization plots"`

### Task 8.2: Flow Visualization

**Files:**
- Create: `src/peach/pl/flow.py`
- Modify: `src/peach/pl/__init__.py`
- Test: `tests/test_visualization/test_flow_viz.py`

**Functions:** `velocity_quiver`, `gene_alignment_barplot`, `jacobian_heatmap`, `trajectory_ribbon`, `flow_magnitude`, `density_comparison`, `archetype_correspondence`

**Commit:** `git commit -m "feat: add flow matching visualization plots"`

---

## Phase 9: Archetype Summary Query Layer

### Task 9.1: archetype_summary()

**Files:**
- Extend: `src/peach/tl/feature_patterns.py`
- Modify: `src/peach/tl/__init__.py`
- Test: `tests/test_statistical/test_archetype_summary.py`

**Tests to write:**
- `test_summary_single_archetype`: returns structured dict for one archetype
- `test_summary_all_archetypes`: returns list of dicts for all K
- `test_includes_regression_results`: top enriched/depleted features present
- `test_includes_driver_results`: driver genesets present when available
- `test_includes_gmm_components`: nearby GMM components listed when available
- `test_graceful_without_optional`: works with only regression (no GMM, no drivers)

**Commit:** `git commit -m "feat: add archetype summary query layer"`

---

## Phase 10: Spatial Pair Enrichment

### Task 10.1: Extend Spatial Module

**Files:**
- Modify: `src/peach/tl/spatial.py`
- Modify: `src/peach/pl/spatial.py`
- Test: `tests/test_integration/test_spatial_pair_enrichment.py`

**Tests to write:**
- `test_pair_enrichment_basic`: runs on spatial adata with archetype weights
- `test_permutation_pvalues`: p-values in [0, 1]
- `test_weight_threshold_filtering`: only cells above threshold participate
- `test_all_pairs_enumerated`: 'all' generates K*(K-1)/2 pairs

**Commit:** `git commit -m "feat: add archetype pair enrichment to spatial module"`

---

## Phase 11: Type Registry + Schema Updates

### Task 11.1: Update types_index.py

**Files:**
- Modify: `src/peach/_core/types_index.py`

Add entries for all new functions to `FUNCTION_RETURNS`, new adata keys to `ADATA_KEYS`, new optional fields to `USE_GET_FOR`.

**Commit:** `git commit -m "docs: update types_index.py with v0.5.0 function returns"`

### Task 11.2: Update tools_schema.py

**Files:**
- Modify: `src/peach/_core/tools_schema.py`

Add `ToolSchema` entries for all new public API functions.

**Commit:** `git commit -m "docs: update tools_schema.py with v0.5.0 tool definitions"`

### Task 11.3: Integration Test

**Files:**
- Create: `tests/test_integration/test_v050_integration.py`

End-to-end test running all v0.5.0 modules on `hsc_10k.h5ad`:
1. Load + prepare + train model
2. Run simplex regression
3. Classify patterns
4. Run driver regression (on pathway scores if available)
5. Run GMM decomposition
6. Generate archetype summaries
7. Create visualization suite

**Commit:** `git commit -m "test: add v0.5.0 end-to-end integration test"`

---

## Dependency Order Summary

```
Phase 1 (infrastructure)
  ├── Phase 2 (simplex regression) ← depends on Phase 1
  │   ├── Phase 3 (pattern classification) ← depends on Phase 2
  │   ├── Phase 4 (driver regression) ← depends on Phases 1, 2
  │   └── Phase 7 (ternary + regression viz) ← depends on Phase 2
  ├── Phase 5 (GMM) ← depends on Phase 1
  │   └── Phase 8.1 (GMM viz) ← depends on Phase 5
  └── Phase 6 (flow matching) ← depends on Phase 1
      └── Phase 8.2 (flow viz) ← depends on Phase 6
Phase 9 (archetype summary) ← depends on Phases 2-5
Phase 10 (spatial pair enrichment) ← depends on Phase 1
Phase 11 (registry + integration) ← depends on all
```

**Parallelizable groups:**
- After Phase 2: Phases 3, 4, 7 can run in parallel
- After Phase 1: Phases 5, 6, 10 can run in parallel with Phase 2
- Phases 8.1 and 8.2 are independent of each other

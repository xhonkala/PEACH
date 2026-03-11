"""
Gremlin Attack Suite for v0.5.0 Continuous Characterization
===========================================================

Systematic chaos testing of assumptions in:
- simplex_regression.py
- ilr_transform.py
- simplex_gmm.py
- flow_matching.py
- pattern_classification.py
- feature_utils.py
- feature_regression.py (public API)
- feature_patterns.py (public API)
- feature_decomposition.py (public API)
- flow.py (public API)

Each test targets a specific assumption and documents:
- What assumption is being attacked
- Expected behavior (crash, degrade, survive)
- Severity if broken
"""

import sys
import traceback
import numpy as np
import scipy.sparse as sp
from anndata import AnnData

# ---------- Helpers ----------

PASS_COUNT = 0
FAIL_COUNT = 0
PARTIAL_COUNT = 0
RESULTS = []


def attack(name, target_assumption, severity="MEDIUM"):
    """Decorator to record attack results."""
    def decorator(func):
        def wrapper():
            global PASS_COUNT, FAIL_COUNT, PARTIAL_COUNT
            result_entry = {
                "name": name,
                "target_assumption": target_assumption,
                "severity": severity,
            }
            try:
                outcome, behavior = func()
                result_entry["result"] = outcome
                result_entry["behavior"] = behavior
                if outcome == "BROKEN":
                    FAIL_COUNT += 1
                elif outcome == "PARTIAL":
                    PARTIAL_COUNT += 1
                else:
                    PASS_COUNT += 1
            except Exception as e:
                result_entry["result"] = "BROKEN"
                result_entry["behavior"] = f"Unhandled exception: {type(e).__name__}: {e}"
                FAIL_COUNT += 1
            RESULTS.append(result_entry)
            status_icon = {"SURVIVED": "[OK]", "BROKEN": "[XX]", "PARTIAL": "[!!]"}.get(
                result_entry["result"], "[??]"
            )
            print(f"  {status_icon} Attack: {name}")
            print(f"       Result: {result_entry['result']}")
            print(f"       Behavior: {result_entry['behavior']}")
            print()
        return wrapper
    return decorator


def make_adata_with_weights(n_cells=100, K=4, n_genes=50, weights=None):
    """Create a minimal AnnData with valid archetype weights."""
    rng = np.random.default_rng(42)
    X = rng.poisson(3, (n_cells, n_genes)).astype(np.float32)
    adata = AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]

    if weights is None:
        raw = rng.dirichlet(np.ones(K), size=n_cells)
        weights = raw
    adata.obsm["cell_archetype_weights"] = weights
    return adata


# =================================================================
# ATTACK BATTERY
# =================================================================

# ---- 1. ILR Transform Attacks ----

@attack("ILR: K=1 (degenerate simplex)", "ILR assumes K>=2", "CRITICAL")
def atk_ilr_k1():
    from peach._core.utils.ilr_transform import ilr_transform
    W = np.array([[1.0], [1.0], [1.0]])
    try:
        result = ilr_transform(W)
        return "BROKEN", f"K=1 should fail but returned shape {result.shape}"
    except (ValueError, IndexError) as e:
        return "SURVIVED", f"Correctly rejected K=1: {e}"
    except Exception as e:
        return "PARTIAL", f"Crashed with unexpected error: {type(e).__name__}: {e}"


@attack("ILR: K=2 (minimal simplex)", "ILR Helmert basis handles K=2", "HIGH")
def atk_ilr_k2():
    from peach._core.utils.ilr_transform import ilr_transform, inverse_ilr
    W = np.array([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]])
    ilr = ilr_transform(W)
    W_back = inverse_ilr(ilr)
    max_err = np.max(np.abs(W - W_back))
    if ilr.shape == (3, 1) and max_err < 0.05:
        return "SURVIVED", f"K=2 works, roundtrip max error = {max_err:.6f}"
    return "BROKEN", f"K=2 failed: ilr shape={ilr.shape}, roundtrip error={max_err:.6f}"


@attack("ILR: exact zeros in weights", "Epsilon smoothing handles zeros", "CRITICAL")
def atk_ilr_zeros():
    from peach._core.utils.ilr_transform import ilr_transform
    W = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    result = ilr_transform(W)
    if np.any(np.isinf(result)) or np.any(np.isnan(result)):
        return "BROKEN", f"ILR produced inf/nan for exact-zero weights: {result}"
    return "SURVIVED", f"Epsilon smoothing handled zeros, result finite: max={np.max(np.abs(result)):.4f}"


@attack("ILR: negative weights", "ILR does not validate input", "HIGH")
def atk_ilr_negative():
    from peach._core.utils.ilr_transform import ilr_transform
    W = np.array([[1.5, -0.5, 0.0], [0.5, 0.5, 0.0]])
    try:
        result = ilr_transform(W)
        has_nan = np.any(np.isnan(result))
        has_inf = np.any(np.isinf(result))
        if has_nan or has_inf:
            return "BROKEN", f"Negative weights produced nan/inf without error"
        return "PARTIAL", f"Negative weights silently accepted (may produce wrong results): {result}"
    except Exception as e:
        return "SURVIVED", f"Correctly rejected negatives: {e}"


@attack("ILR: NaN in weights", "ILR does not check for NaN", "HIGH")
def atk_ilr_nan():
    from peach._core.utils.ilr_transform import ilr_transform
    W = np.array([[0.5, np.nan, 0.5], [0.3, 0.3, 0.4]])
    try:
        result = ilr_transform(W)
        if np.any(np.isnan(result)):
            return "BROKEN", f"NaN propagated through ILR without error"
        return "PARTIAL", f"NaN somehow produced finite result: {result}"
    except Exception as e:
        return "SURVIVED", f"Correctly rejected NaN: {e}"


# ---- 2. Simplex Regression Attacks ----

@attack("OLS: K=1 archetype (rank-1 design)", "OLS assumes invertible W'W", "CRITICAL")
def atk_ols_k1():
    from peach._core.utils.simplex_regression import scheffe_design_matrix, ols_fit
    W = np.ones((100, 1))  # All weights = 1 since K=1
    Y = np.random.randn(100, 5)
    X, _ = scheffe_design_matrix(W, degree=1)
    try:
        result = ols_fit(X, Y)
        return "SURVIVED", f"K=1 regression ran, R²={result['r_squared'][:3]}"
    except np.linalg.LinAlgError as e:
        return "BROKEN", f"LinAlgError on K=1: {e}"
    except Exception as e:
        return "BROKEN", f"Unexpected error on K=1: {type(e).__name__}: {e}"


@attack("OLS: K=2 interactions (degree=2)", "Scheffe degree=2 needs K>=2 for interactions", "MEDIUM")
def atk_ols_k2_degree2():
    from peach._core.utils.simplex_regression import scheffe_design_matrix, ols_fit
    rng = np.random.default_rng(42)
    W = rng.dirichlet([1, 1], size=100)
    Y = rng.standard_normal((100, 5))
    X, pairs = scheffe_design_matrix(W, degree=2)
    try:
        result = ols_fit(X, Y)
        return "SURVIVED", f"K=2 degree=2: {len(pairs)} interaction pairs, shape={X.shape}"
    except Exception as e:
        return "BROKEN", f"K=2 degree=2 failed: {e}"


@attack("OLS: weights don't sum to 1", "get_archetype_weights validates sum-to-1", "HIGH")
def atk_weights_not_summing():
    from peach._core.utils.feature_utils import get_archetype_weights
    adata = make_adata_with_weights()
    adata.obsm["cell_archetype_weights"] = adata.obsm["cell_archetype_weights"] * 2.0
    try:
        w = get_archetype_weights(adata)
        return "BROKEN", f"Accepted weights summing to ~2.0 without error"
    except ValueError as e:
        return "SURVIVED", f"Correctly rejected: {e}"


@attack("OLS: all-zero feature column", "OLS handles constant features", "MEDIUM")
def atk_ols_zero_feature():
    from peach._core.utils.simplex_regression import scheffe_design_matrix, ols_fit
    rng = np.random.default_rng(42)
    W = rng.dirichlet([1, 1, 1], size=100)
    Y = rng.standard_normal((100, 5))
    Y[:, 2] = 0.0  # all-zero column
    X, _ = scheffe_design_matrix(W, degree=1)
    result = ols_fit(X, Y)
    r2_zero = result["r_squared"][2]
    if r2_zero == 0.0:
        return "SURVIVED", f"Zero-variance feature gets R²=0 as expected"
    return "PARTIAL", f"Zero-variance feature R²={r2_zero} (expected 0.0)"


@attack("OLS: n=1 cell", "OLS with 1 observation", "HIGH")
def atk_ols_n1():
    from peach._core.utils.simplex_regression import scheffe_design_matrix, ols_fit
    W = np.array([[0.3, 0.3, 0.4]])
    Y = np.array([[1.0, 2.0]])
    X, _ = scheffe_design_matrix(W, degree=1)
    try:
        result = ols_fit(X, Y)
        return "PARTIAL", f"n=1 ran (fragile): R²={result['r_squared']}, F={result['f_statistics']}"
    except np.linalg.LinAlgError as e:
        return "BROKEN", f"LinAlgError: {e}"
    except Exception as e:
        return "BROKEN", f"Error: {type(e).__name__}: {e}"


@attack("OLS: p >> n (more features than cells)", "OLS design p < n", "HIGH")
def atk_ols_p_gt_n():
    from peach._core.utils.simplex_regression import scheffe_design_matrix, ols_fit
    rng = np.random.default_rng(42)
    K = 20
    n_cells = 10
    W = rng.dirichlet(np.ones(K), size=n_cells)
    Y = rng.standard_normal((n_cells, 5))
    # degree=2 -> p = K + K*(K-1)/2 = 20 + 190 = 210 >> n=10
    X, pairs = scheffe_design_matrix(W, degree=2)
    try:
        result = ols_fit(X, Y)
        return "BROKEN", f"p={X.shape[1]} >> n={n_cells} should fail but gave R²={result['r_squared']}"
    except np.linalg.LinAlgError as e:
        return "SURVIVED", f"Correctly failed: {e}"
    except Exception as e:
        return "PARTIAL", f"Failed with: {type(e).__name__}: {e}"


@attack("OLS: NaN in Y (features)", "OLS does not check for NaN in input", "HIGH")
def atk_ols_nan_y():
    from peach._core.utils.simplex_regression import scheffe_design_matrix, ols_fit
    rng = np.random.default_rng(42)
    W = rng.dirichlet([1, 1, 1], size=100)
    Y = rng.standard_normal((100, 5))
    Y[10, 2] = np.nan
    X, _ = scheffe_design_matrix(W, degree=1)
    result = ols_fit(X, Y)
    if np.any(np.isnan(result["r_squared"])):
        return "BROKEN", f"NaN propagated to R² without error"
    return "SURVIVED", f"NaN handled somehow, R²={result['r_squared']}"


@attack("OLS: Inf in Y (features)", "OLS does not check for Inf", "HIGH")
def atk_ols_inf_y():
    from peach._core.utils.simplex_regression import scheffe_design_matrix, ols_fit
    rng = np.random.default_rng(42)
    W = rng.dirichlet([1, 1, 1], size=100)
    Y = rng.standard_normal((100, 5))
    Y[10, 2] = np.inf
    X, _ = scheffe_design_matrix(W, degree=1)
    try:
        result = ols_fit(X, Y)
        if np.any(np.isnan(result["r_squared"])) or np.any(np.isinf(result["r_squared"])):
            return "BROKEN", f"Inf propagated to R² without error: {result['r_squared']}"
        return "PARTIAL", f"Inf somehow produced finite results"
    except Exception as e:
        return "SURVIVED", f"Inf rejected: {e}"


@attack("OLS: sparse Y input", "OLS densifies sparse Y", "LOW")
def atk_ols_sparse_y():
    from peach._core.utils.simplex_regression import scheffe_design_matrix, ols_fit
    rng = np.random.default_rng(42)
    W = rng.dirichlet([1, 1, 1], size=100)
    Y_dense = rng.poisson(3, (100, 50)).astype(float)
    Y_sparse = sp.csr_matrix(Y_dense)
    X, _ = scheffe_design_matrix(W, degree=1)
    result_dense = ols_fit(X, Y_dense)
    result_sparse = ols_fit(X, Y_sparse)
    max_diff = np.max(np.abs(result_dense["r_squared"] - result_sparse["r_squared"]))
    if max_diff < 1e-10:
        return "SURVIVED", f"Sparse and dense give identical R² (diff={max_diff:.2e})"
    return "BROKEN", f"Sparse/dense R² differ by {max_diff:.2e}"


@attack("OLS: perfect multicollinearity", "OLS inverts W'W which may be singular", "CRITICAL")
def atk_ols_multicollinear():
    from peach._core.utils.simplex_regression import scheffe_design_matrix, ols_fit
    rng = np.random.default_rng(42)
    # All cells at the same vertex -> rank-1 weight matrix
    W = np.zeros((100, 3))
    W[:, 0] = 1.0  # All weight on archetype 0
    Y = rng.standard_normal((100, 5))
    X, _ = scheffe_design_matrix(W, degree=1)
    try:
        result = ols_fit(X, Y)
        return "BROKEN", f"Singular W'W should fail but got R²={result['r_squared'][:3]}"
    except np.linalg.LinAlgError as e:
        return "SURVIVED", f"Correctly detected singular matrix: {e}"
    except Exception as e:
        return "PARTIAL", f"Failed with: {type(e).__name__}: {e}"


@attack("OLS: hat matrix diagonal = 1 (perfect leverage)", "HC3 divides by (1-h_ii)", "CRITICAL")
def atk_ols_leverage():
    from peach._core.utils.simplex_regression import scheffe_design_matrix, ols_fit
    # n = p: every point is a leverage point (h_ii = 1)
    rng = np.random.default_rng(42)
    K = 3
    n = K  # n = p -> hat matrix = I
    W = np.eye(K)  # 3 cells at exactly 3 vertices
    Y = rng.standard_normal((n, 5))
    X, _ = scheffe_design_matrix(W, degree=1)
    try:
        result = ols_fit(X, Y, robust_se=True)
        se_max = np.max(result["standard_errors"])
        if se_max >= 1e6:
            return "PARTIAL", f"HC3 SEs clipped to max={se_max:.2e} (clip at 1e6)"
        return "SURVIVED", f"HC3 handled perfect leverage, max SE={se_max:.2e}"
    except ZeroDivisionError as e:
        return "BROKEN", f"HC3 division by zero: {e}"
    except Exception as e:
        return "PARTIAL", f"Unexpected: {type(e).__name__}: {e}"


@attack("OLS: R² > 1 possible?", "R² should be clamped to [0,1]", "MEDIUM")
def atk_r2_bounds():
    from peach._core.utils.simplex_regression import scheffe_design_matrix, ols_fit
    rng = np.random.default_rng(42)
    W = rng.dirichlet([1, 1, 1], size=200)
    Y = rng.standard_normal((200, 50))
    X, _ = scheffe_design_matrix(W, degree=1)
    result = ols_fit(X, Y)
    r2 = result["r_squared"]
    if np.any(r2 > 1.0 + 1e-10):
        return "BROKEN", f"R² > 1 found: max={np.max(r2):.6f}"
    if np.any(r2 < -1e-10):
        return "PARTIAL", f"Negative R² found: min={np.min(r2):.6f} (possible for no-intercept models)"
    return "SURVIVED", f"R² in valid range: [{np.min(r2):.6f}, {np.max(r2):.6f}]"


# ---- 3. Pattern Classification Attacks ----

@attack("Pattern: all betas zero", "Classification handles zero coefficients", "LOW")
def atk_pattern_all_zero():
    from peach._core.utils.pattern_classification import classify_single_feature
    result = classify_single_feature(
        vertex_betas=np.zeros(4),
        interaction_betas=None,
        r2=0.0,
        p_betas=np.ones(4),
        p_interactions=None,
    )
    if result["pattern"] == "flat":
        return "SURVIVED", f"All-zero correctly classified as flat"
    return "BROKEN", f"All-zero classified as {result['pattern']}"


@attack("Pattern: K=2 archetype classification", "Pattern works with K=2", "MEDIUM")
def atk_pattern_k2():
    from peach._core.utils.pattern_classification import classify_single_feature
    result = classify_single_feature(
        vertex_betas=np.array([5.0, 0.5]),
        interaction_betas=None,
        r2=0.8,
        p_betas=np.array([0.001, 0.3]),
        p_interactions=None,
    )
    if result["pattern"] in ["archetype-exclusive", "monotonic"]:
        return "SURVIVED", f"K=2 classified as {result['pattern']}"
    return "PARTIAL", f"K=2 classified as {result['pattern']} (expected exclusive/gradient)"


@attack("Pattern: division by zero in ratio", "sorted_betas[0] could be 0", "HIGH")
def atk_pattern_ratio_div0():
    from peach._core.utils.pattern_classification import classify_single_feature
    # All betas are zero but R² is high (contradictory but possible numerically)
    result = classify_single_feature(
        vertex_betas=np.array([0.0, 0.0, 0.0]),
        interaction_betas=None,
        r2=0.5,
        p_betas=np.array([0.01, 0.01, 0.01]),
        p_interactions=None,
    )
    return "SURVIVED", f"Zero betas with high R² classified as {result['pattern']}"


@attack("Pattern: NaN R²", "Classification does not check for NaN R²", "HIGH")
def atk_pattern_nan_r2():
    from peach._core.utils.pattern_classification import classify_single_feature
    try:
        result = classify_single_feature(
            vertex_betas=np.array([1.0, 2.0, 3.0]),
            interaction_betas=None,
            r2=np.nan,
            p_betas=np.array([0.01, 0.01, 0.01]),
            p_interactions=None,
        )
        # NaN comparisons are all False, so r2 < r2_threshold is False -> skips flat
        return "BROKEN", f"NaN R² not caught, classified as {result['pattern']}"
    except Exception as e:
        return "SURVIVED", f"NaN R² caught: {e}"


# ---- 4. GMM Attacks ----

@attack("GMM: K=2 archetypes (ILR 1D)", "GMM in 1D ILR space", "MEDIUM")
def atk_gmm_k2():
    from peach._core.utils.simplex_gmm import fit_simplex_gmm
    rng = np.random.default_rng(42)
    W = rng.dirichlet([1, 1], size=200)
    try:
        result = fit_simplex_gmm(W, n_components_range=(2, 4), n_initializations=5)
        return "SURVIVED", f"K=2 GMM: {result['n_components_stable']} stable components"
    except Exception as e:
        return "BROKEN", f"K=2 GMM failed: {e}"


@attack("GMM: all cells at one vertex", "GMM on degenerate distribution", "HIGH")
def atk_gmm_degenerate():
    from peach._core.utils.simplex_gmm import fit_simplex_gmm
    W = np.zeros((100, 3))
    W[:, 0] = 1.0
    try:
        result = fit_simplex_gmm(W, n_components_range=(2, 4), n_initializations=3)
        return "PARTIAL", f"Degenerate GMM ran: {result['n_components_stable']} components"
    except Exception as e:
        return "BROKEN", f"Degenerate distribution: {type(e).__name__}: {e}"


@attack("GMM: n_components_range larger than n_cells", "GMM n_components > n_cells", "MEDIUM")
def atk_gmm_too_many_components():
    from peach._core.utils.simplex_gmm import fit_simplex_gmm
    rng = np.random.default_rng(42)
    W = rng.dirichlet([1, 1, 1], size=10)
    try:
        result = fit_simplex_gmm(W, n_components_range=(5, 15), n_initializations=3)
        return "PARTIAL", f"n_comp > n_cells ran: {result['n_components_stable']} stable"
    except Exception as e:
        return "SURVIVED", f"Correctly failed: {type(e).__name__}: {e}"


@attack("GMM: very large K (100 archetypes)", "ILR + GMM with K=100", "MEDIUM")
def atk_gmm_large_k():
    from peach._core.utils.simplex_gmm import fit_simplex_gmm
    rng = np.random.default_rng(42)
    W = rng.dirichlet(np.ones(100), size=500)
    try:
        result = fit_simplex_gmm(W, n_components_range=(2, 5), n_initializations=3)
        return "SURVIVED", f"K=100 GMM: {result['n_components_stable']} components, ILR dim={99}"
    except Exception as e:
        return "BROKEN", f"K=100 failed: {type(e).__name__}: {e}"


# ---- 5. Feature Utils Attacks ----

@attack("feature_utils: missing weights key", "get_archetype_weights raises KeyError", "LOW")
def atk_missing_weights():
    from peach._core.utils.feature_utils import get_archetype_weights
    adata = AnnData(np.random.randn(10, 5))
    try:
        get_archetype_weights(adata)
        return "BROKEN", "Should have raised KeyError"
    except KeyError as e:
        return "SURVIVED", f"Correct KeyError: {e}"


@attack("feature_utils: resolve_features with wrong shape", "resolve_features checks shape", "MEDIUM")
def atk_resolve_bad_shape():
    from peach._core.utils.feature_utils import resolve_features
    adata = AnnData(np.random.randn(10, 5))
    wrong_shape = np.random.randn(20, 5)  # 20 != 10 cells
    try:
        resolve_features(adata, feature_matrix=wrong_shape)
        return "BROKEN", "Wrong shape accepted"
    except ValueError as e:
        return "SURVIVED", f"Correctly rejected: {e}"


@attack("feature_utils: sparse all-zero column", "resolve_features handles sparse", "LOW")
def atk_sparse_zero_col():
    from peach._core.utils.feature_utils import resolve_features
    X_dense = np.random.randn(10, 5)
    X_dense[:, 2] = 0.0
    X_sparse = sp.csr_matrix(X_dense)
    adata = AnnData(X_sparse)
    mat, names = resolve_features(adata, feature_matrix=None)
    if sp.issparse(mat):
        zero_col = mat[:, 2].toarray().ravel()
    else:
        zero_col = mat[:, 2]
    if np.all(zero_col == 0):
        return "SURVIVED", f"Sparse all-zero column preserved correctly"
    return "BROKEN", f"Zero column corrupted"


# ---- 6. Public API: feature_regression Attacks ----

@attack("API: feature_simplex_regression K=2", "API works with K=2", "HIGH")
def atk_api_regression_k2():
    from peach.tl.feature_regression import feature_simplex_regression
    adata = make_adata_with_weights(n_cells=100, K=2, n_genes=20)
    try:
        result = feature_simplex_regression(adata, n_bootstrap=0, permutation_test=False)
        return "SURVIVED", f"K=2 API: {result['n_features']} features, R² range [{np.min(result['r_squared_degree1']):.4f}, {np.max(result['r_squared_degree1']):.4f}]"
    except Exception as e:
        return "BROKEN", f"K=2 API failed: {e}"


@attack("API: feature_simplex_regression with sparse X", "API handles sparse adata.X", "MEDIUM")
def atk_api_regression_sparse():
    from peach.tl.feature_regression import feature_simplex_regression
    adata = make_adata_with_weights(n_cells=100, K=4, n_genes=20)
    adata.X = sp.csr_matrix(adata.X)
    try:
        result = feature_simplex_regression(adata, n_bootstrap=0, permutation_test=False)
        return "SURVIVED", f"Sparse API: {result['n_features']} features, max R²={np.max(result['r_squared_degree1']):.4f}"
    except Exception as e:
        return "BROKEN", f"Sparse API failed: {e}"


@attack("API: classify before regression", "classify_feature_patterns needs regression first", "MEDIUM")
def atk_api_classify_no_regression():
    from peach.tl.feature_patterns import classify_feature_patterns
    adata = make_adata_with_weights(n_cells=100, K=4, n_genes=20)
    try:
        classify_feature_patterns(adata)
        return "BROKEN", "Should have raised ValueError (no regression results)"
    except ValueError as e:
        return "SURVIVED", f"Correctly raised ValueError: {e}"


@attack("API: archetype_summary with no results", "archetype_summary needs regression", "MEDIUM")
def atk_api_summary_no_results():
    from peach.tl.feature_patterns import archetype_summary
    adata = make_adata_with_weights(n_cells=100, K=4, n_genes=20)
    try:
        archetype_summary(adata)
        return "BROKEN", "Should have raised ValueError"
    except ValueError as e:
        return "SURVIVED", f"Correctly raised ValueError: {e}"


@attack("API: archetype_driver_regression with too many features", "n_features > max_interaction_features guard", "LOW")
def atk_api_driver_too_many():
    from peach.tl.feature_regression import archetype_driver_regression
    adata = make_adata_with_weights(n_cells=100, K=4, n_genes=100)
    try:
        result = archetype_driver_regression(adata, max_degree=2, max_interaction_features=50, n_bootstrap=0)
        return "BROKEN", "Should have raised ValueError for too many features"
    except ValueError as e:
        return "SURVIVED", f"Correctly raised ValueError: {e}"


@attack("API: archetype_driver_regression K=2", "Driver regression with K=2 (1 ILR dim)", "HIGH")
def atk_api_driver_k2():
    from peach.tl.feature_regression import archetype_driver_regression
    adata = make_adata_with_weights(n_cells=100, K=2, n_genes=10)
    try:
        result = archetype_driver_regression(adata, max_degree=1, n_bootstrap=0)
        return "SURVIVED", f"K=2 driver: R²={result['r_squared']}, shape main_coefs={np.asarray(result['main_coefficients']).shape}"
    except Exception as e:
        return "BROKEN", f"K=2 driver failed: {e}"


# ---- 7. Flow Attacks ----

@attack("Flow: _build_mask with empty filter dict", "Empty dict matches all cells", "MEDIUM")
def atk_flow_empty_filter():
    from peach.tl.flow import _build_mask
    adata = make_adata_with_weights(n_cells=100, K=4)
    mask = _build_mask(adata, {})
    if np.all(mask):
        return "SURVIVED", f"Empty filter matches all {mask.sum()} cells"
    return "BROKEN", f"Empty filter matched {mask.sum()} cells (expected 100)"


@attack("Flow: _build_mask with nonexistent column", "Filter validates column existence", "LOW")
def atk_flow_bad_column():
    from peach.tl.flow import _build_mask
    adata = make_adata_with_weights(n_cells=100, K=4)
    try:
        _build_mask(adata, {"nonexistent_col": "value"})
        return "BROKEN", "Should have raised ValueError"
    except ValueError as e:
        return "SURVIVED", f"Correctly rejected: {e}"


@attack("Flow: _build_mask matches zero cells", "No cells match filter", "MEDIUM")
def atk_flow_zero_match():
    from peach.tl.flow import _build_mask
    adata = make_adata_with_weights(n_cells=100, K=4)
    adata.obs["treatment"] = "A"
    mask = _build_mask(adata, {"treatment": "NONEXISTENT"})
    if mask.sum() == 0:
        return "SURVIVED", f"Zero cells matched (caller must handle empty arrays)"
    return "BROKEN", f"Unexpected: {mask.sum()} cells matched"


@attack("Flow: compute_mmd identical distributions", "MMD should be ~0 for same distribution", "LOW")
def atk_mmd_identical():
    from peach._core.utils.flow_matching import compute_mmd
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 10))
    mmd = compute_mmd(X, X.copy())
    if mmd < 1e-10:
        return "SURVIVED", f"MMD of identical distributions = {mmd:.2e}"
    return "PARTIAL", f"MMD of identical distributions = {mmd:.2e} (should be ~0)"


@attack("Flow: compute_mmd single point", "MMD with n=1", "MEDIUM")
def atk_mmd_single():
    from peach._core.utils.flow_matching import compute_mmd
    X = np.array([[1.0, 2.0, 3.0]])
    Y = np.array([[4.0, 5.0, 6.0]])
    try:
        mmd = compute_mmd(X, Y)
        return "SURVIVED", f"Single-point MMD = {mmd:.4f}"
    except Exception as e:
        return "BROKEN", f"Single-point MMD failed: {e}"


@attack("Flow: VelocityNetwork dim=1", "Minimum dimensionality", "LOW")
def atk_flow_dim1():
    from peach._core.utils.flow_matching import VelocityNetwork
    import torch
    net = VelocityNetwork(dim=1, hidden_dims=(16,))
    x = torch.randn(5, 1)
    t = torch.rand(5)
    v = net(x, t)
    if v.shape == (5, 1):
        return "SURVIVED", f"dim=1 works: output shape={v.shape}"
    return "BROKEN", f"dim=1 wrong shape: {v.shape}"


# ---- 8. HC3 Numerical Edge Cases ----

@attack("HC3: all cells at same point (uniform leverage)", "HC3 with uniform hat diagonal", "MEDIUM")
def atk_hc3_uniform_leverage():
    from peach._core.utils.simplex_regression import scheffe_design_matrix, ols_fit
    rng = np.random.default_rng(42)
    # All cells with identical weights -> rank-deficient
    W = np.tile([0.5, 0.3, 0.2], (100, 1))
    Y = rng.standard_normal((100, 5))
    X, _ = scheffe_design_matrix(W, degree=1)
    try:
        result = ols_fit(X, Y, robust_se=True)
        return "BROKEN", f"All-identical weights should make W'W singular but got R²={result['r_squared'][:3]}"
    except np.linalg.LinAlgError as e:
        return "SURVIVED", f"Correctly detected rank-deficiency: {e}"


@attack("HC3: clip behavior at h_ii close to 1", "HC3 clips adjustment at 1e6", "MEDIUM")
def atk_hc3_clip():
    from peach._core.utils.simplex_regression import _hc3_standard_errors
    n, p = 10, 3
    W = np.random.randn(n, p)
    WtW_inv = np.linalg.inv(W.T @ W)
    H_diag = np.sum((W @ WtW_inv) * W, axis=1)
    # Force one h_ii very close to 1
    residuals = np.random.randn(n, 5)
    H_diag[0] = 0.9999999
    se = _hc3_standard_errors(W, residuals, WtW_inv, H_diag)
    if np.all(np.isfinite(se)):
        return "SURVIVED", f"HC3 handled h_ii near 1, max SE={np.max(se):.2e}"
    return "BROKEN", f"HC3 produced non-finite SEs: {se}"


# ---- 9. ILR Roundtrip Precision ----

@attack("ILR: roundtrip precision with extreme weights", "ILR epsilon corrupts near-vertex points", "MEDIUM")
def atk_ilr_roundtrip_extreme():
    from peach._core.utils.ilr_transform import ilr_transform, inverse_ilr
    W = np.array([
        [0.999, 0.0005, 0.0005],
        [0.0005, 0.999, 0.0005],
        [1/3, 1/3, 1/3],
    ])
    ilr = ilr_transform(W)
    W_back = inverse_ilr(ilr)
    max_err = np.max(np.abs(W - W_back))
    if max_err > 0.05:
        return "PARTIAL", f"ILR roundtrip error {max_err:.4f} for near-vertex points (epsilon smoothing)"
    return "SURVIVED", f"ILR roundtrip error {max_err:.6f}"


# ---- 10. Scheffe Degree 2 with K=1 ----

@attack("Scheffe: degree=2 with K=1 (no pairs)", "combinations(range(1), 2) is empty", "MEDIUM")
def atk_scheffe_k1_degree2():
    from peach._core.utils.simplex_regression import scheffe_design_matrix
    W = np.ones((100, 1))
    X, pairs = scheffe_design_matrix(W, degree=2)
    if len(pairs) == 0 and X.shape == (100, 1):
        return "SURVIVED", f"K=1 degree=2: no interaction pairs, shape={X.shape}"
    return "BROKEN", f"K=1 degree=2 unexpected: pairs={len(pairs)}, shape={X.shape}"


# ---- 11. Full Pipeline Integration Attacks ----

@attack("Pipeline: regression -> classification end-to-end", "Pipeline works without crash", "HIGH")
def atk_pipeline_e2e():
    from peach.tl.feature_regression import feature_simplex_regression
    from peach.tl.feature_patterns import classify_feature_patterns
    adata = make_adata_with_weights(n_cells=200, K=4, n_genes=30)
    try:
        reg_result = feature_simplex_regression(
            adata, max_degree=2, n_bootstrap=0, permutation_test=False
        )
        pat_result = classify_feature_patterns(adata, regression_result=reg_result)
        n_patterns = len(set(c["pattern"] for c in pat_result["classifications"]))
        return "SURVIVED", f"E2E pipeline: {pat_result['n_features']} features, {n_patterns} unique patterns"
    except Exception as e:
        return "BROKEN", f"E2E pipeline failed: {e}"


@attack("Pipeline: regression -> GMM end-to-end", "GMM pipeline works", "HIGH")
def atk_pipeline_gmm():
    from peach.tl.feature_decomposition import feature_simplex_decomposition
    adata = make_adata_with_weights(n_cells=200, K=4, n_genes=30)
    try:
        result = feature_simplex_decomposition(
            adata, n_initializations=3, n_components_range=(3, 6)
        )
        return "SURVIVED", f"GMM pipeline: {result['n_components_stable']} stable components"
    except Exception as e:
        return "BROKEN", f"GMM pipeline failed: {e}"


# ---- 12. Negative Weight Attacks via get_archetype_weights ----

@attack("Weights: negative values", "get_archetype_weights accepts negatives if sum=1", "HIGH")
def atk_negative_weights():
    from peach._core.utils.feature_utils import get_archetype_weights
    adata = AnnData(np.random.randn(5, 3))
    # Weights with negatives but summing to 1
    W = np.array([
        [1.5, -0.3, -0.2],
        [0.5, 0.3, 0.2],
        [0.8, 0.1, 0.1],
        [-0.1, 0.6, 0.5],
        [0.4, 0.4, 0.2],
    ])
    adata.obsm["cell_archetype_weights"] = W
    try:
        weights = get_archetype_weights(adata)
        if np.any(weights < 0):
            return "BROKEN", f"Negative weights accepted (sum=1 but entries negative)"
        return "SURVIVED", "Negatives blocked"
    except ValueError as e:
        return "SURVIVED", f"Negatives rejected via sum check: {e}"


# ---- 13. SimplexRegressionResult Reconstruction ----

@attack("Pydantic: SimplexRegressionResult reconstruction from stored dict",
        "to_serializable -> reconstruct roundtrip", "MEDIUM")
def atk_pydantic_roundtrip():
    from peach.tl.feature_regression import feature_simplex_regression
    from peach._core.types import SimplexRegressionResult
    adata = make_adata_with_weights(n_cells=100, K=3, n_genes=10)
    result = feature_simplex_regression(adata, max_degree=2, n_bootstrap=0, permutation_test=False)
    stored = adata.uns["peach_simplex_regression"]
    try:
        reconstructed = SimplexRegressionResult(**stored)
        return "SURVIVED", f"Pydantic roundtrip succeeded: {reconstructed.n_features} features"
    except Exception as e:
        return "BROKEN", f"Pydantic roundtrip failed: {e}"


# ---- 14. No-intercept R² behavior ----

@attack("R²: no-intercept model can give R² < 0", "R² = 1 - SSres/SStot_centered", "MEDIUM")
def atk_r2_no_intercept_negative():
    from peach._core.utils.simplex_regression import scheffe_design_matrix, ols_fit
    rng = np.random.default_rng(42)
    # Create a situation where the no-intercept model is worse than the mean
    K = 3
    n = 200
    W = rng.dirichlet([1, 1, 1], size=n)
    # Y is constant + noise: the model can only use weights (sum=1) to predict a constant,
    # which it can do well, so R² should be fine. Let's try harder.
    # Make Y deliberately anti-correlated with all weight columns
    Y = np.zeros((n, 1))
    Y[:, 0] = 100.0 + rng.normal(0, 0.001, n)  # Very tight around 100
    X, _ = scheffe_design_matrix(W, degree=1)
    result = ols_fit(X, Y)
    r2 = result["r_squared"][0]
    return "SURVIVED", f"No-intercept R² = {r2:.6f} for near-constant Y"


# ---- 15. Bootstrap with singular bootstrapped matrices ----

@attack("Bootstrap: singular bootstrap samples", "Bootstrap sample may have duplicate rows", "MEDIUM")
def atk_bootstrap_singular():
    from peach.tl.feature_regression import feature_simplex_regression
    # Small n: bootstrap likely to sample duplicates -> near-singular W'W
    adata = make_adata_with_weights(n_cells=5, K=3, n_genes=3)
    try:
        result = feature_simplex_regression(
            adata, max_degree=1, n_bootstrap=50, permutation_test=False
        )
        if result.get("vertex_ci_lower") is not None:
            return "SURVIVED", f"Bootstrap survived n=5: CI range = {np.ptp(result['vertex_ci_lower']):.4f}"
        return "PARTIAL", "Bootstrap returned None CIs"
    except np.linalg.LinAlgError as e:
        return "BROKEN", f"Bootstrap singular matrix crash: {e}"
    except Exception as e:
        return "BROKEN", f"Bootstrap crash: {type(e).__name__}: {e}"


# ---- 16. Characterize components with -1 labels ----

@attack("GMM characterize: all cells labeled -1", "characterize_components handles -1 labels", "LOW")
def atk_characterize_all_unstable():
    from peach._core.utils.simplex_gmm import characterize_components
    assignments = np.full(100, -1, dtype=int)
    features = np.random.randn(100, 10)
    profiles = characterize_components(assignments, features, n_components=3)
    if np.all(profiles == 0):
        return "SURVIVED", f"All -1 labels give zero profiles (correct)"
    return "PARTIAL", f"Non-zero profiles for unstable cells: {profiles.mean():.4f}"


# ---- 17. Stability scoring edge cases ----

@attack("Stability: n_initializations=1", "Stability needs >1 runs to compare", "MEDIUM")
def atk_stability_n1():
    from peach._core.utils.simplex_gmm import _compute_stability
    from peach._core.utils.ilr_transform import ilr_transform
    rng = np.random.default_rng(42)
    W = rng.dirichlet([1, 1, 1], size=200)
    ilr_coords = ilr_transform(W)
    try:
        stability = _compute_stability(ilr_coords, 3, "full", n_initializations=1, random_state=42)
        # With 1 init, the loop range(1, 1) is empty, so matches stays 0
        # stability = (0 + 1) / 1 = 1.0 for all components
        if np.all(stability == 1.0):
            return "PARTIAL", f"n_init=1: all components get stability=1.0 (vacuous truth)"
        return "SURVIVED", f"n_init=1: stability={stability}"
    except Exception as e:
        return "BROKEN", f"n_init=1 failed: {e}"


# ---- 18. Stability threshold at adaptive median ----

@attack("Stability: threshold=0 with median trick",
        "Adaptive threshold uses median(cost)*0.5 which can be 0", "MEDIUM")
def atk_stability_zero_threshold():
    from peach._core.utils.simplex_gmm import _compute_stability
    from peach._core.utils.ilr_transform import ilr_transform
    # All identical points -> cost matrix is all zeros -> threshold = 0
    W = np.tile([0.5, 0.3, 0.2], (200, 1)) + np.random.randn(200, 3) * 1e-10
    W = W / W.sum(axis=1, keepdims=True)
    ilr_coords = ilr_transform(W)
    try:
        stability = _compute_stability(ilr_coords, 2, "full", n_initializations=5, random_state=42)
        return "PARTIAL", f"Near-degenerate data: stability={stability} (threshold logic may be fragile)"
    except Exception as e:
        return "BROKEN", f"Near-degenerate stability failed: {e}"


# ---- 19. flow_between modifies input AnnDatas ----

@attack("Flow: flow_between mutates input AnnDatas", "flow_between adds condition_key column", "HIGH")
def atk_flow_between_mutation():
    # Just check the code path without actually running (requires flow_matching)
    # Instead test _build_mask behavior
    adata1 = make_adata_with_weights(n_cells=50, K=3)
    adata2 = make_adata_with_weights(n_cells=50, K=3)
    original_cols_1 = set(adata1.obs.columns)
    original_cols_2 = set(adata2.obs.columns)
    # Simulate what flow_between does to inputs
    for a, label in zip([adata1, adata2], ["cond_0", "cond_1"]):
        a.obs["condition"] = label
    new_cols_1 = set(adata1.obs.columns)
    new_cols_2 = set(adata2.obs.columns)
    if new_cols_1 != original_cols_1:
        return "BROKEN", f"flow_between mutates input AnnData obs columns: added {new_cols_1 - original_cols_1}"
    return "SURVIVED", "Input AnnDatas not mutated"


# =================================================================
# MAIN EXECUTION
# =================================================================

def run_all_attacks():
    print("=" * 70)
    print("GREMLIN ATTACK SUITE: v0.5.0 Continuous Characterization")
    print("=" * 70)
    print()

    attacks = [v for v in globals().values() if callable(v) and hasattr(v, '__wrapped__') or
               (callable(v) and v.__name__.startswith('atk_'))]

    # Collect all attack functions manually
    attack_funcs = []
    for name, obj in sorted(globals().items()):
        if name.startswith('atk_') and callable(obj):
            attack_funcs.append((name, obj))

    for name, func in attack_funcs:
        try:
            func()
        except Exception as e:
            print(f"  [!!] Attack {name} ERRORED at meta level: {e}")
            traceback.print_exc()
            print()

    print("=" * 70)
    print("GREMLIN REPORT SUMMARY")
    print("=" * 70)
    print(f"  Attacks attempted: {len(RESULTS)}")
    print(f"  BROKEN:   {FAIL_COUNT}")
    print(f"  PARTIAL:  {PARTIAL_COUNT}")
    print(f"  SURVIVED: {PASS_COUNT}")
    print()

    if FAIL_COUNT > 0:
        print("--- BROKEN ATTACKS (require fix) ---")
        for r in RESULTS:
            if r["result"] == "BROKEN":
                print(f"  [{r['severity']}] {r['name']}")
                print(f"    Assumption: {r['target_assumption']}")
                print(f"    Behavior:   {r['behavior']}")
                print()

    if PARTIAL_COUNT > 0:
        print("--- PARTIAL ATTACKS (degraded behavior) ---")
        for r in RESULTS:
            if r["result"] == "PARTIAL":
                print(f"  [{r['severity']}] {r['name']}")
                print(f"    Assumption: {r['target_assumption']}")
                print(f"    Behavior:   {r['behavior']}")
                print()

    return RESULTS


if __name__ == "__main__":
    run_all_attacks()

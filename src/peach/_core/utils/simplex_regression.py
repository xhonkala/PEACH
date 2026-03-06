"""Simplex regression engine: Scheffe polynomials with OLS, HC3 SEs, F-tests.

Implements intercept-free regression on simplex (barycentric) coordinates.
Because archetype weights sum to 1, an intercept is redundant in the Scheffe
polynomial basis. HC3 standard errors provide heteroscedasticity-robust
inference without assuming constant variance across the simplex.

References
----------
Scheffe, H. (1958). "Experiments with Mixtures." JRSS-B, 20(2), 344-360.
MacKinnon & White (1985). "Some heteroskedasticity-consistent covariance
    matrix estimators with improved finite sample properties." JoE, 29, 305-325.
"""

import warnings

import numpy as np
import scipy.sparse as sp
from itertools import combinations
from scipy import stats


def scheffe_design_matrix(W, degree=1):
    """Build Scheffe polynomial design matrix from simplex weights.

    The Scheffe polynomial basis is natural for mixture/simplex data:
    - Degree 1: X = W (linear effects, no intercept needed since sum=1)
    - Degree 2: X = [W | w_j * w_k for all j<k] (adds pairwise interactions)

    Parameters
    ----------
    W : np.ndarray
        Archetype weights [n_cells, K], rows sum to 1.
    degree : int
        1 = linear (weights only), 2 = with pairwise interactions.

    Returns
    -------
    X : np.ndarray
        Design matrix [n_cells, p] where p = K for degree=1,
        p = K + K*(K-1)/2 for degree=2.
    pairs : list[tuple[int, int]]
        List of (j, k) index pairs for interaction columns.
        Empty list for degree=1.
    """
    if degree == 1:
        return W.copy(), []

    K = W.shape[1]
    pairs = list(combinations(range(K), 2))
    if len(pairs) == 0:
        # K=1 with degree=2: no interaction pairs possible
        return W.copy(), []
    interactions = np.column_stack([W[:, j] * W[:, k] for j, k in pairs])
    X = np.column_stack([W, interactions])
    return X, pairs


def ols_fit(W, Y, robust_se=True, chunk_size=5000):
    """Vectorized OLS: regress each feature on design matrix W (no intercept).

    Fits Y = W @ beta + epsilon for each column of Y simultaneously.
    No intercept term because simplex weights sum to 1 (Scheffe basis).

    Handles sparse Y efficiently by processing features in chunks to avoid
    materializing the full dense matrix (which can be >60GB for real scRNA-seq).

    Parameters
    ----------
    W : np.ndarray
        Design matrix [n_cells, p]. For Scheffe: p = K or K + K*(K-1)/2.
    Y : np.ndarray or scipy.sparse matrix
        Feature matrix [n_cells, n_features].
    robust_se : bool
        If True, compute HC3 heteroscedasticity-consistent standard errors.
        If False, compute classical (homoscedastic) OLS standard errors.
    chunk_size : int
        Number of features to process at once when Y is sparse.

    Returns
    -------
    dict
        coefficients : np.ndarray [n_features, p]
        r_squared : np.ndarray [n_features]
        residuals : np.ndarray [n_cells, n_features] or None (sparse + large)
        standard_errors : np.ndarray [n_features, p]
        t_statistics : np.ndarray [n_features, p]
        t_pvalues : np.ndarray [n_features, p]
        f_statistics : np.ndarray [n_features]
        f_pvalues : np.ndarray [n_features]
    """
    n, p = W.shape
    is_sparse = sp.issparse(Y)
    n_features = Y.shape[1]

    # Input validation
    if n < p:
        raise ValueError(
            f"Underdetermined system: {n} cells < {p} design columns. "
            f"Need more observations than parameters."
        )

    # Solve normal equations with stability check
    WtW = W.T @ W
    try:
        WtW_inv = np.linalg.solve(WtW, np.eye(p))
    except np.linalg.LinAlgError:
        raise ValueError(
            "Design matrix W'W is singular. This usually means archetype weights "
            "are degenerate (e.g., all cells at one vertex, or n ≈ p)."
        )

    # Check condition number for near-singular warning
    cond = np.linalg.cond(WtW)
    if cond > 1e12:
        warnings.warn(
            f"Design matrix is near-singular (condition number {cond:.1e}). "
            "Results may be numerically unstable.",
            RuntimeWarning,
        )

    # Compute beta: sparse-safe (W.T @ sparse_Y works in scipy)
    if is_sparse:
        WtY = np.asarray((W.T @ Y).todense()) if sp.issparse(W.T @ Y) else W.T @ Y
    else:
        Y_dense = np.asarray(Y)
        WtY = W.T @ Y_dense
    beta = (WtW_inv @ WtY).T  # [n_features, p]

    # Hat matrix diagonal for HC3 — clip BEFORE division to avoid inf
    H_diag = None
    if robust_se:
        H_diag = np.sum((W @ WtW_inv) * W, axis=1)  # [n]
        H_diag = np.clip(H_diag, 0, 1 - 1e-10)

    # Compute residual statistics
    ss_res = np.zeros(n_features)
    ss_tot = np.zeros(n_features)
    se = np.zeros((n_features, p))

    if is_sparse:
        # Sparse path: process in chunks to avoid OOM
        residual_chunks = []
        for start in range(0, n_features, chunk_size):
            end = min(start + chunk_size, n_features)
            Y_chunk = Y[:, start:end].toarray()
            Y_hat_chunk = W @ beta[start:end].T
            res_chunk = Y_chunk - Y_hat_chunk

            ss_res[start:end] = np.sum(res_chunk ** 2, axis=0)
            y_mean = Y_chunk.mean(axis=0, keepdims=True)
            ss_tot[start:end] = np.sum((Y_chunk - y_mean) ** 2, axis=0)

            if robust_se:
                se[start:end] = _hc3_standard_errors(W, res_chunk, WtW_inv, H_diag)
            else:
                sigma2 = ss_res[start:end] / max(n - p, 1)
                var_diag = np.diag(WtW_inv)
                se[start:end] = np.sqrt(np.outer(sigma2, var_diag))

            residual_chunks.append(res_chunk)

        residuals = np.hstack(residual_chunks)
    else:
        # Dense path: compute all at once
        Y_hat = W @ beta.T
        residuals = Y_dense - Y_hat
        ss_res = np.sum(residuals ** 2, axis=0)
        y_mean = Y_dense.mean(axis=0, keepdims=True)
        ss_tot = np.sum((Y_dense - y_mean) ** 2, axis=0)

        if robust_se:
            se = _hc3_standard_errors(W, residuals, WtW_inv, H_diag)
        else:
            sigma2 = ss_res / max(n - p, 1)
            var_diag = np.diag(WtW_inv)
            se = np.sqrt(np.outer(sigma2, var_diag))

    r_squared = np.where(ss_tot > 0, 1 - ss_res / ss_tot, 0.0)

    # t-statistics and two-sided p-values
    t_stats = np.where(se > 0, beta / se, 0.0)
    df = max(n - p, 1)
    t_pvalues = 2 * stats.t.sf(np.abs(t_stats), df=df)

    # Overall model F-test: H0: all beta_j = 0
    ss_reg = ss_tot - ss_res
    df_reg = p
    df_res = max(n - p, 1)
    f_stats = np.zeros(n_features)
    f_pvalues = np.ones(n_features)
    valid = (ss_tot > 0) & (ss_reg > 0)
    if np.any(valid):
        ms_reg = ss_reg[valid] / df_reg
        ms_res = ss_res[valid] / df_res
        with np.errstate(divide="ignore", invalid="ignore"):
            f_stats[valid] = np.where(ms_res > 0, ms_reg / ms_res, np.inf)
        f_pvalues[valid] = stats.f.sf(f_stats[valid], dfn=df_reg, dfd=df_res)

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
    """HC3 heteroscedasticity-consistent standard errors.

    HC3 adjusts residuals by the leverage (hat matrix diagonal) to provide
    better finite-sample coverage than HC0/HC1. The sandwich estimator is:

        Var(beta) = (W'W)^{-1} [ sum_i w_i w_i' e_i^2 / (1 - h_ii)^2 ] (W'W)^{-1}

    Only the hat matrix diagonal h_ii is computed, never the full n x n matrix.

    Parameters
    ----------
    W : np.ndarray [n, p]
        Design matrix.
    residuals : np.ndarray [n, n_features]
        OLS residuals.
    WtW_inv : np.ndarray [p, p]
        Inverse of W'W, precomputed.
    H_diag : np.ndarray [n]
        Diagonal of the hat matrix, already clipped to [0, 1-eps].

    Returns
    -------
    se : np.ndarray [n_features, p]
        HC3 standard errors for each coefficient of each feature.
    """
    n, p = W.shape
    n_features = residuals.shape[1]

    # HC3 adjustment factor: 1 / (1 - h_ii)
    # H_diag is pre-clipped to [0, 1-1e-10] by caller, so no inf here
    adjustment = 1.0 / (1 - H_diag)

    se = np.empty((n_features, p))
    for g in range(n_features):
        # Adjusted residuals: e_i / (1 - h_ii)
        e_adj = residuals[:, g] * adjustment
        # Meat of the sandwich: W' diag(e_adj^2) W
        We = W * (e_adj ** 2)[:, np.newaxis]
        meat = W.T @ We
        # Full sandwich: (W'W)^{-1} meat (W'W)^{-1}
        sandwich = WtW_inv @ meat @ WtW_inv
        se[g] = np.sqrt(np.maximum(np.diag(sandwich), 0))

    return se

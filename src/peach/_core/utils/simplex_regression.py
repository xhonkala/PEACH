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
    interactions = np.column_stack([W[:, j] * W[:, k] for j, k in pairs])
    X = np.column_stack([W, interactions])
    return X, pairs


def ols_fit(W, Y, robust_se=True):
    """Vectorized OLS: regress each feature on design matrix W (no intercept).

    Fits Y = W @ beta + epsilon for each column of Y simultaneously.
    No intercept term because simplex weights sum to 1 (Scheffe basis).

    Parameters
    ----------
    W : np.ndarray
        Design matrix [n_cells, p]. For Scheffe: p = K or K + K*(K-1)/2.
    Y : np.ndarray or scipy.sparse matrix
        Feature matrix [n_cells, n_features]. Sparse matrices are densified.
    robust_se : bool
        If True, compute HC3 heteroscedasticity-consistent standard errors.
        If False, compute classical (homoscedastic) OLS standard errors.

    Returns
    -------
    dict
        coefficients : np.ndarray [n_features, p]
            OLS coefficient estimates.
        r_squared : np.ndarray [n_features]
            Coefficient of determination (centered, relative to mean).
        residuals : np.ndarray [n_cells, n_features]
            OLS residuals Y - W @ beta.
        standard_errors : np.ndarray [n_features, p]
            Standard errors (HC3 if robust_se=True, else classical).
        t_statistics : np.ndarray [n_features, p]
            t-statistics for each coefficient.
        t_pvalues : np.ndarray [n_features, p]
            Two-sided p-values for t-tests.
        f_statistics : np.ndarray [n_features]
            Overall model F-statistic.
        f_pvalues : np.ndarray [n_features]
            P-values for overall F-test.
    """
    n, p = W.shape

    if sp.issparse(Y):
        Y_dense = Y.toarray()
    else:
        Y_dense = np.asarray(Y)

    n_features = Y_dense.shape[1]

    # OLS normal equations: beta = (W'W)^{-1} W'Y
    WtW = W.T @ W
    WtW_inv = np.linalg.inv(WtW)
    beta = WtW_inv @ (W.T @ Y_dense)  # [p, n_features]
    beta = beta.T  # [n_features, p]

    # Residuals and fitted values
    Y_hat = W @ beta.T
    residuals = Y_dense - Y_hat

    # R-squared (centered total sum of squares)
    ss_res = np.sum(residuals ** 2, axis=0)
    y_mean = Y_dense.mean(axis=0, keepdims=True)
    ss_tot = np.sum((Y_dense - y_mean) ** 2, axis=0)
    r_squared = np.where(ss_tot > 0, 1 - ss_res / ss_tot, 0.0)

    # Standard errors
    if robust_se:
        H_diag = np.sum((W @ WtW_inv) * W, axis=1)
        se = _hc3_standard_errors(W, residuals, WtW_inv, H_diag)
    else:
        sigma2 = ss_res / max(n - p, 1)
        var_diag = np.diag(WtW_inv)
        se = np.sqrt(np.outer(sigma2, var_diag))

    # t-statistics and two-sided p-values
    t_stats = np.where(se > 0, beta / se, 0.0)
    df = max(n - p, 1)
    t_pvalues = 2 * stats.t.sf(np.abs(t_stats), df=df)

    # Overall model F-test: H0: all beta_j = 0
    # Use SS directly: F = (SS_reg / p) / (SS_res / (n - p))
    # This handles R^2 == 1.0 (perfect fit) gracefully via inf F-stat.
    ss_reg = ss_tot - ss_res
    df_reg = p
    df_res = max(n - p, 1)
    f_stats = np.zeros(n_features)
    f_pvalues = np.ones(n_features)
    valid = (ss_tot > 0) & (ss_reg > 0)
    if np.any(valid):
        ms_reg = ss_reg[valid] / df_reg
        ms_res = ss_res[valid] / df_res
        # Guard against zero residual SS (perfect fit) -> inf F-stat
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
        Diagonal of the hat matrix H = W (W'W)^{-1} W'.

    Returns
    -------
    se : np.ndarray [n_features, p]
        HC3 standard errors for each coefficient of each feature.
    """
    n, p = W.shape
    n_features = residuals.shape[1]

    # HC3 adjustment factor: 1 / (1 - h_ii)
    adjustment = 1.0 / (1 - H_diag)
    adjustment = np.clip(adjustment, 0, 1e6)

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

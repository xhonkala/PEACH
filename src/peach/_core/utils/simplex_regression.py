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

    Degree 1: X = W (linear, no intercept since sum=1)
    Degree 2: adds w_j * w_k for all j<k
    Degree d: adds products of d distinct weights for all d-subsets

    Parameters
    ----------
    W : np.ndarray [n_cells, K], rows sum to 1.
    degree : int, max polynomial degree. Must be <= K.

    Returns
    -------
    X : np.ndarray [n_cells, p]
    interaction_info : list[tuple] — index tuples for columns beyond K.
    """
    K = W.shape[1]
    if degree > K:
        raise ValueError(f"degree={degree} exceeds K={K}. Max meaningful degree on a {K}-simplex is {K}.")
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}.")
    if degree == 1:
        return W.copy(), []

    columns = [W]
    interaction_info = []
    for d in range(2, degree + 1):
        tuples = list(combinations(range(K), d))
        if not tuples:
            continue
        cols = np.column_stack([np.prod(W[:, list(t)], axis=1) for t in tuples])
        columns.append(cols)
        interaction_info.extend(tuples)

    X = np.column_stack(columns)
    return X, interaction_info


def ols_fit(W, Y, robust_se=True, chunk_size=5000, return_covariance=False, return_residuals=True):
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
    return_covariance : bool
        If True, include full covariance matrices in return dict.
        Needed for Wald contrasts. Default False (SEs only).
    return_residuals : bool
        If True, include residual matrix in return dict. If False, residuals
        entry is None. Memory savings are significant only for sparse Y
        (avoids accumulating dense chunks). For dense Y, residuals are still
        computed temporarily for SE/SS calculations but not returned.

    Returns
    -------
    dict
        coefficients : np.ndarray [n_features, p]
        r_squared : np.ndarray [n_features]
        residuals : np.ndarray [n_cells, n_features] or None
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
    cond = np.linalg.cond(WtW)

    # Report effective rank (SVD-based)
    _, s_vals, _ = np.linalg.svd(WtW)
    effective_rank = int(np.sum(s_vals > s_vals[0] * 1e-10))
    # Scheffe design (no intercept) has full rank p for well-sampled simplices.
    # The sum-to-1 constraint is affine, not linear, so columns are independent.
    # Flag if effective rank < p (indicates genuine collinearity).
    expected_rank = p
    extra_rank_deficient = effective_rank < expected_rank

    if cond > 1e12:
        warnings.warn(
            f"Design matrix is near-singular (condition number {cond:.1e}). "
            "Falling back to pseudoinverse for numerical stability.",
            RuntimeWarning,
        )
        WtW_inv = np.linalg.pinv(WtW)
    else:
        try:
            WtW_inv = np.linalg.solve(WtW, np.eye(p))
        except np.linalg.LinAlgError:
            warnings.warn(
                "W'W is singular; falling back to pseudoinverse.",
                RuntimeWarning,
            )
            WtW_inv = np.linalg.pinv(WtW)

    if extra_rank_deficient:
        warnings.warn(
            f"Design matrix has effective rank {effective_rank} but expected at "
            f"least {expected_rank}. This indicates collinearity beyond the "
            f"expected simplex sum-to-1 constraint.",
            RuntimeWarning,
        )

    # Compute beta: sparse-safe (W.T @ sparse_Y works in scipy)
    if is_sparse:
        WtY_raw = W.T @ Y
        WtY = np.asarray(WtY_raw.todense()) if sp.issparse(WtY_raw) else WtY_raw
    else:
        Y_dense = np.asarray(Y)
        WtY = W.T @ Y_dense
    beta = (WtW_inv @ WtY).T  # [n_features, p]

    # Hat matrix diagonal for HC3 — clip BEFORE division to avoid inf
    H_diag = None
    if robust_se:
        H_diag = np.sum((W @ WtW_inv) * W, axis=1)  # [n]
        H_diag = np.clip(H_diag, 0, 1 - 1e-10)
        n_extreme = np.sum(H_diag > 0.99)
        if n_extreme > 0:
            warnings.warn(
                f"{n_extreme} cells have leverage h_ii > 0.99 (near-saturated). "
                "HC3 standard errors for these cells may be unreliable. "
                "This usually means some cells sit exactly on an archetype vertex "
                "with no other cells nearby.",
                RuntimeWarning,
            )

    # Whether we need to materialize residuals
    need_residuals = return_residuals or return_covariance

    # Compute residual statistics
    ss_res = np.zeros(n_features)
    ss_tot = np.zeros(n_features)
    se = np.zeros((n_features, p))

    if is_sparse:
        # Sparse path: process in chunks to avoid OOM
        _residual_chunks = [] if need_residuals else None
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

            if need_residuals:
                _residual_chunks.append(res_chunk)

        residuals = np.hstack(_residual_chunks) if need_residuals else None
    else:
        # Dense path: compute all at once
        Y_hat = W @ beta.T
        if need_residuals:
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
        else:
            # Compute stats in-place without keeping full residual matrix
            diff = Y_dense - Y_hat
            ss_res = np.sum(diff ** 2, axis=0)
            y_mean = Y_dense.mean(axis=0, keepdims=True)
            ss_tot = np.sum((Y_dense - y_mean) ** 2, axis=0)

            if robust_se:
                se = _hc3_standard_errors(W, diff, WtW_inv, H_diag)
            else:
                sigma2 = ss_res / max(n - p, 1)
                var_diag = np.diag(WtW_inv)
                se = np.sqrt(np.outer(sigma2, var_diag))
            del diff
            residuals = None

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

    r_squared = np.where(ss_tot > 0, 1 - ss_res / ss_tot, 0.0)

    # t-statistics and two-sided p-values
    t_stats = np.where(se > 0, beta / se, 0.0)
    df = max(n - p, 1)
    t_pvalues = 2 * stats.t.sf(np.abs(t_stats), df=df)

    # Overall model F-test
    # On the simplex (sum w_i = 1), a constant model is always fit implicitly,
    # so the correct null is H0: all beta_j equal (not all zero).
    # This gives df_reg = p - 1, not p.
    ss_reg = ss_tot - ss_res
    df_reg = p - 1
    df_res = max(n - p, 1)
    f_stats = np.zeros(n_features)
    f_pvalues = np.ones(n_features)
    if df_reg >= 1:
        valid = (ss_tot > 0) & (ss_reg > 0)
        if np.any(valid):
            ms_reg = ss_reg[valid] / df_reg
            ms_res = ss_res[valid] / df_res
            with np.errstate(divide="ignore", invalid="ignore"):
                f_stats[valid] = np.where(ms_res > 0, ms_reg / ms_res, np.inf)
            f_pvalues[valid] = stats.f.sf(f_stats[valid], dfn=df_reg, dfd=df_res)

    result_dict = {
        "coefficients": beta,
        "r_squared": r_squared,
        "residuals": residuals if return_residuals else None,
        "standard_errors": se,
        "t_statistics": t_stats,
        "t_pvalues": t_pvalues,
        "f_statistics": f_stats,
        "f_pvalues": f_pvalues,
        "effective_rank": effective_rank,
        "expected_rank": expected_rank,
        "extra_rank_deficient": extra_rank_deficient,
    }
    if return_covariance:
        result_dict["covariance"] = covariance
    return result_dict


def _hc3_standard_errors(W, residuals, WtW_inv, H_diag):
    """HC3 heteroscedasticity-consistent standard errors (vectorized).

    HC3 adjusts residuals by the leverage (hat matrix diagonal) to provide
    better finite-sample coverage than HC0/HC1. The sandwich estimator is:

        Var(beta) = (W'W)^{-1} [ sum_i w_i w_i' e_i^2 / (1 - h_ii)^2 ] (W'W)^{-1}

    Only the hat matrix diagonal h_ii is computed, never the full n x n matrix.
    Uses einsum to compute all features simultaneously instead of a Python loop.

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
    # HC3 adjustment factor: 1 / (1 - h_ii)
    # H_diag is pre-clipped to [0, 1-1e-10] by caller, so no inf here
    adjustment = 1.0 / (1 - H_diag)

    # Adjusted residuals squared: [n, n_features]
    e_adj_sq = (residuals * adjustment[:, np.newaxis]) ** 2

    # Meat of sandwich for all features at once:
    # meat[g, a, b] = sum_i e_adj_sq[i, g] * W[i, a] * W[i, b]
    meat_all = np.einsum('ig,ia,ib->gab', e_adj_sq, W, W)  # [n_features, p, p]

    # Sandwich: (W'W)^{-1} @ meat @ (W'W)^{-1}
    sandwich_all = np.einsum('ab,gbc,cd->gad', WtW_inv, meat_all, WtW_inv)

    # SE = sqrt(diag(sandwich))
    se = np.sqrt(np.maximum(np.diagonal(sandwich_all, axis1=1, axis2=2), 0))

    return se


def _hc3_covariance(W, residuals, WtW_inv, H_diag):
    """Full HC3 sandwich covariance per feature (vectorized).

    Returns list of [p, p] matrices, one per feature.
    """
    adjustment = 1.0 / (1 - H_diag)
    e_adj_sq = (residuals * adjustment[:, np.newaxis]) ** 2
    meat_all = np.einsum('ig,ia,ib->gab', e_adj_sq, W, W)
    sandwich_all = np.einsum('ab,gbc,cd->gad', WtW_inv, meat_all, WtW_inv)
    return [sandwich_all[g] for g in range(sandwich_all.shape[0])]

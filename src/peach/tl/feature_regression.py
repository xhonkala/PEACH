"""Simplex regression and archetype driver regression public API."""

import warnings

import numpy as np
from anndata import AnnData
from scipy import stats
from statsmodels.stats.multitest import multipletests

from peach._core.utils.feature_utils import (
    get_archetype_weights,
    regression_storage_suffix,
    resolve_features,
    store_result,
)
from peach._core.utils.simplex_regression import ols_fit, scheffe_design_matrix
from peach._core.types import DriverRegressionResult, SimplexRegressionResult


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
    comprehensive_degree: bool = False,
    store_to_adata: bool = True,
    random_seed: int = 42,
    copy: bool = False,
) -> dict:
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
    comprehensive_degree : bool
        If True, run degree d=2..K-1 fits with incremental F-tests, storing
        results in serialized['degree_comparison'].
    store_to_adata : bool
        If True (default), store results in adata.uns. Set to False when
        calling in a loop (e.g., per-component regression) to avoid
        overwriting shared keys.
    random_seed : int
        Seed for bootstrap and permutation RNGs. Vary to check stability.
    copy : bool
        If True, operate on a copy of adata.

    Returns
    -------
    dict
        Serialized SimplexRegressionResult. Stored in namespaced key:
        adata.uns['peach_simplex_regression_genes'] (when feature_matrix=None),
        adata.uns['peach_simplex_regression_pathways'] (when feature_matrix='pathway_scores'),
        or adata.uns['peach_simplex_regression_{feature_matrix}'] for other obsm keys.
        Also stored at adata.uns['peach_simplex_regression'] for backward compat.
    """
    if copy:
        adata = adata.copy()

    weights = get_archetype_weights(adata)
    Y, feat_names = resolve_features(adata, feature_matrix, feature_names)
    K = weights.shape[1]
    n_cells = adata.n_obs
    n_features = len(feat_names)
    archetype_names = [f"archetype_{i+1}" for i in range(K)]

    # Degree 1
    W1, _ = scheffe_design_matrix(weights, degree=1)
    result1 = ols_fit(W1, Y, robust_se=robust_se, return_covariance=True)

    # Vertex contrasts: beta_j - mean(beta), consistent with F-test null H0: all equal.
    # SE via centering matrix L_K applied to the per-feature covariance sandwich.
    vertex_betas = result1["coefficients"]  # [n_features, K] (degree-1 p == K)
    vertex_contrasts = vertex_betas - vertex_betas.mean(axis=1, keepdims=True)

    cov_all = np.array(result1["covariance"])  # [n_features, K, K]
    L_K = np.eye(K) - np.ones((K, K)) / K
    contrast_cov_all = np.einsum('ab,gbc,cd->gad', L_K, cov_all, L_K.T)
    contrast_se = np.sqrt(np.maximum(np.diagonal(contrast_cov_all, axis1=1, axis2=2), 0))

    df1 = max(n_cells - K, 1)
    contrast_t_stats = np.where(contrast_se > 0, vertex_contrasts / contrast_se, 0.0)
    contrast_pvalues = 2 * stats.t.sf(np.abs(contrast_t_stats), df=df1)

    # FDR correction on F-test (clamp underflowed zeros for large-N datasets)
    f_pvals_clamped = np.clip(result1["f_pvalues"], np.finfo(float).tiny, 1.0)
    _, f_pvalue_fdr, _, _ = multipletests(f_pvals_clamped, method="fdr_bh")

    # FDR on vertex contrast p-values — per-archetype correction (one family per column).
    # Tests H0: beta_j = mean(beta), consistent with the df_reg=K-1 F-test null.
    vertex_pvalues_fdr = np.ones_like(contrast_pvalues)
    for col in range(contrast_pvalues.shape[1]):
        col_pvals = np.clip(contrast_pvalues[:, col], np.finfo(float).tiny, 1.0)
        _, col_fdr, _, _ = multipletests(col_pvals, method="fdr_bh")
        vertex_pvalues_fdr[:, col] = col_fdr

    # Degree 2 (if requested)
    interaction_coefficients = None
    interaction_pairs = None
    interaction_pvalues = None
    interaction_pvalues_fdr = None
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

        # FDR on interaction t-pvalues — per-pair correction (one family per column)
        interaction_pvalues_fdr = np.ones_like(interaction_pvalues)
        for col in range(interaction_pvalues.shape[1]):
            col_pvals = np.clip(interaction_pvalues[:, col], np.finfo(float).tiny, 1.0)
            _, col_fdr, _, _ = multipletests(col_pvals, method="fdr_bh")
            interaction_pvalues_fdr[:, col] = col_fdr

    # Permutation test for model significance
    permutation_pvalue = None
    permutation_pvalue_fdr = None

    if permutation_test:
        permutation_pvalue, permutation_pvalue_fdr = _permutation_test_regression(
            weights, Y, n_permutations=n_permutations,
            observed_r2=result1["r_squared"],
            seed=random_seed,
        )

    # Bootstrap CIs
    vertex_ci_lower = None
    vertex_ci_upper = None
    interaction_ci_lower = None
    interaction_ci_upper = None

    if n_bootstrap > 0:
        vertex_ci_lower, vertex_ci_upper = _bootstrap_regression_cis(
            weights, Y, degree=1, n_bootstrap=n_bootstrap, K=K, seed=random_seed,
        )
        if max_degree >= 2:
            int_ci_lo, int_ci_hi = _bootstrap_regression_cis(
                weights, Y, degree=2, n_bootstrap=n_bootstrap, K=K, seed=random_seed,
            )
            interaction_ci_lower = int_ci_lo[:, K:]
            interaction_ci_upper = int_ci_hi[:, K:]

    # Store residuals
    if store_residuals and result1["residuals"] is not None:
        store_result(adata, "residuals", result1["residuals"], domain="obsm")

    # Build result
    result = SimplexRegressionResult(
        feature_names=feat_names,
        archetype_names=archetype_names,
        n_cells=n_cells,
        n_features=n_features,
        n_archetypes=K,
        vertex_coefficients=result1["coefficients"],
        vertex_contrasts=vertex_contrasts,
        r_squared_degree1=result1["r_squared"],
        f_pvalue=result1["f_pvalues"],
        f_pvalue_fdr=f_pvalue_fdr,
        vertex_pvalues=contrast_pvalues,
        vertex_pvalues_fdr=vertex_pvalues_fdr,
        vertex_se=contrast_se,
        vertex_covariance=result1.get("covariance"),
        interaction_coefficients=interaction_coefficients,
        interaction_pairs=interaction_pairs,
        interaction_pvalues=interaction_pvalues,
        interaction_pvalues_fdr=interaction_pvalues_fdr,
        interaction_se=interaction_se,
        r_squared_degree2=r_squared_degree2,
        permutation_pvalue=permutation_pvalue,
        permutation_pvalue_fdr=permutation_pvalue_fdr,
        vertex_ci_lower=vertex_ci_lower,
        vertex_ci_upper=vertex_ci_upper,
        interaction_ci_lower=interaction_ci_lower,
        interaction_ci_upper=interaction_ci_upper,
    )

    # Store serializable summary at namespaced key + generic fallback
    serialized = result.to_serializable()

    # Track the feature matrix source for downstream consumers (e.g. Wald contrasts)
    serialized["feature_source"] = feature_matrix

    # Comprehensive degree comparison (Enhancement 4)
    if comprehensive_degree:
        serialized["degree_comparison"] = _comprehensive_degree_comparison(
            weights, Y, K, robust_se=robust_se,
            r_squared_degree1=result1["r_squared"],
        )

    if store_to_adata:
        suffix = regression_storage_suffix(feature_matrix)
        store_result(adata, f"simplex_regression_{suffix}", serialized)
        if suffix == "genes":
            store_result(adata, "simplex_regression", serialized)

    return serialized


def gene_simplex_regression(adata: AnnData, **kwargs) -> dict:
    """Convenience: simplex regression on adata.X (gene expression)."""
    return feature_simplex_regression(adata, feature_matrix=None, **kwargs)


def pathway_simplex_regression(adata: AnnData, **kwargs) -> dict:
    """Convenience: simplex regression on adata.obsm['pathway_scores']."""
    return feature_simplex_regression(
        adata, feature_matrix="pathway_scores", **kwargs
    )


def _permutation_test_regression(weights, Y, *, n_permutations, observed_r2, seed=42):
    """Vectorized permutation test for simplex regression R².

    Shuffles weight rows (breaking cell-weight correspondence), re-fits
    degree-1 Scheffe regression for all features simultaneously, collects
    null R² distribution per feature.

    Returns (permutation_pvalue, permutation_pvalue_fdr), each [n_features].
    """
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    n = weights.shape[0]
    n_features = Y.shape[1] if not sp.issparse(Y) else Y.shape[1]

    # Densify Y once for efficiency
    if sp.issparse(Y):
        Y_dense = Y.toarray()
    else:
        Y_dense = np.asarray(Y, dtype=np.float64)

    null_r2 = np.empty((n_permutations, n_features))

    for i in range(n_permutations):
        perm_idx = rng.permutation(n)
        W_perm, _ = scheffe_design_matrix(weights[perm_idx], degree=1)
        perm_result = ols_fit(W_perm, Y_dense, robust_se=False, return_residuals=False)
        null_r2[i] = perm_result["r_squared"]

    # Per-feature p-value: fraction of null >= observed
    permutation_pvalue = (
        np.sum(null_r2 >= observed_r2[np.newaxis, :], axis=0) + 1
    ) / (n_permutations + 1)

    # FDR correction
    _, permutation_pvalue_fdr, _, _ = multipletests(
        permutation_pvalue, method="fdr_bh"
    )

    return permutation_pvalue, permutation_pvalue_fdr


def _bootstrap_regression_cis(weights, Y, degree, n_bootstrap, K, ci_level=0.95, seed=42):
    """Bootstrap CIs for regression coefficients.

    Returns (ci_lower, ci_upper), each [n_features, p].
    """
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    n = weights.shape[0]

    W_design, _ = scheffe_design_matrix(weights, degree=degree)
    p = W_design.shape[1]
    n_features = Y.shape[1] if not sp.issparse(Y) else Y.shape[1]

    boot_coefs = np.empty((n_bootstrap, n_features, p))
    n_failed = 0
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        W_boot = W_design[idx]
        if sp.issparse(Y):
            Y_boot = Y[idx].toarray()
        else:
            Y_boot = np.asarray(Y)[idx]
        try:
            result = ols_fit(W_boot, Y_boot, robust_se=False, return_residuals=False)
            boot_coefs[b] = result["coefficients"]
        except (ValueError, np.linalg.LinAlgError):
            # Singular bootstrap sample (duplicate rows) — use NaN, filter later
            boot_coefs[b] = np.nan
            n_failed += 1

    if n_failed > 0:
        import logging
        logging.getLogger(__name__).warning(
            f"{n_failed}/{n_bootstrap} bootstrap samples were singular and skipped."
        )

    alpha = 1 - ci_level
    ci_lower = np.nanpercentile(boot_coefs, 100 * alpha / 2, axis=0)
    ci_upper = np.nanpercentile(boot_coefs, 100 * (1 - alpha / 2), axis=0)
    return ci_lower, ci_upper


def _comprehensive_degree_comparison(weights, Y, K, *, robust_se, r_squared_degree1,
                                      max_comparison_degree=3):
    """Run degree d=2..min(K-1, max_comparison_degree) fits with incremental F-tests.

    For each degree d:
    1. Fit regression at degree d
    2. Compute delta_R2 vs degree d-1
    3. Incremental F-test: F = ((SS_res_{d-1} - SS_res_d) / df_extra) / (SS_res_d / df_res)
    4. FDR correct the incremental p-values

    Returns dict mapping degree -> results dict.

    The default cap at degree 3 prevents runaway cost for large K: degrees
    4..K-1 rarely add interpretable signal beyond the cubic term, and at K=9
    the unbounded version would fit 7 regressions with up to 511 parameters
    each. Callers that explicitly want a full sweep can pass
    max_comparison_degree=K-1.
    """
    from math import comb
    import scipy.sparse as sp
    from scipy import stats

    n_cells = weights.shape[0]
    n_features = Y.shape[1]

    # Compute SS_tot once (handle sparse chunking)
    chunk_size = 5000
    if sp.issparse(Y):
        ss_tot = np.zeros(n_features)
        for start in range(0, n_features, chunk_size):
            end = min(start + chunk_size, n_features)
            Y_chunk = Y[:, start:end].toarray()
            y_mean = Y_chunk.mean(axis=0, keepdims=True)
            ss_tot[start:end] = np.sum((Y_chunk - y_mean) ** 2, axis=0)
    else:
        Y_dense = np.asarray(Y)
        y_mean = Y_dense.mean(axis=0, keepdims=True)
        ss_tot = np.sum((Y_dense - y_mean) ** 2, axis=0)

    def _n_params(K, degree):
        """Total parameter count for Scheffe polynomial of given degree on K-simplex."""
        return sum(comb(K, order) for order in range(1, degree + 1))

    # Previous degree info (start from degree 1)
    prev_r2 = r_squared_degree1
    prev_n_params = _n_params(K, 1)
    prev_ss_res = np.where(ss_tot > 0, (1 - prev_r2) * ss_tot, 0.0)

    # Cap at min(K-1, max_comparison_degree) to prevent runaway cost for large K
    max_degree = min(K - 1, max_comparison_degree)
    if max_degree < 2:
        return {}

    if max_degree > 5:
        from math import comb as _comb
        warnings.warn(
            f"comprehensive_degree with K={K}, max_comparison_degree={max_degree} "
            f"will fit {max_degree - 1} regressions with up to "
            f"{sum(_comb(K, d) for d in range(1, max_degree + 1))} parameters each. "
            f"This may be slow.",
            RuntimeWarning,
        )

    degree_results = {}
    for d in range(2, max_degree + 1):
        Wd, info = scheffe_design_matrix(weights, degree=d)
        result_d = ols_fit(Wd, Y, robust_se=robust_se, return_residuals=False)

        r2_d = result_d["r_squared"]
        n_params_d = _n_params(K, d)
        ss_res_d = np.where(ss_tot > 0, (1 - r2_d) * ss_tot, 0.0)

        delta_r2 = r2_d - prev_r2
        df_extra = n_params_d - prev_n_params
        df_res = max(n_cells - n_params_d, 1)

        # Incremental F-test
        incremental_f = np.zeros(n_features)
        incremental_p = np.ones(n_features)
        valid = (ss_tot > 0) & (df_extra > 0)
        if np.any(valid):
            num = (prev_ss_res[valid] - ss_res_d[valid]) / df_extra
            denom = ss_res_d[valid] / df_res
            with np.errstate(divide="ignore", invalid="ignore"):
                incremental_f[valid] = np.where(denom > 0, num / denom, np.inf)
            # Clamp negative F to 0 (can happen from numerical noise)
            incremental_f[valid] = np.maximum(incremental_f[valid], 0.0)
            incremental_p[valid] = stats.f.sf(incremental_f[valid], dfn=df_extra, dfd=df_res)

        # FDR correct incremental p-values (clamp underflowed zeros)
        incremental_p_clamped = np.clip(incremental_p, np.finfo(float).tiny, 1.0)
        _, incremental_p_fdr, _, _ = multipletests(incremental_p_clamped, method="fdr_bh")

        # Significant features at FDR < 0.05
        significant_features = np.sum(incremental_p_fdr < 0.05)

        degree_results[f"degree_{d}"] = {
            "r_squared": r2_d,
            "delta_r2": delta_r2,
            "incremental_f": incremental_f,
            "incremental_p_fdr": incremental_p_fdr,
            "significant_features": int(significant_features),
            "n_params": int(n_params_d),
            "df_extra": int(df_extra),
        }

        # Update previous for next iteration
        prev_r2 = r2_d
        prev_n_params = n_params_d
        prev_ss_res = ss_res_d

    return degree_results


def archetype_driver_regression(
    adata: AnnData,
    *,
    feature_matrix=None,
    feature_names=None,
    max_degree: int = 2,
    n_bootstrap: int = 1000,
    robust_se: bool = True,
    max_interaction_features: int = 50,
    copy: bool = False,
) -> dict:
    """Flipped regression: features predict archetype weights (in ILR space).

    Identifies which features (genesets, pathways) drive archetypal
    specialization. Instead of the standard simplex regression (weights
    predict features), this fits K-1 regressions predicting ILR-transformed
    archetype weights from feature values.

    Parameters
    ----------
    adata : AnnData
        Must have archetype weights in obsm['cell_archetype_weights'].
    feature_matrix : None, str, or array-like
        Feature matrix (predictors). Default: adata.obsm['pathway_scores']
        if available, else adata.X.
    feature_names : list[str] or None
        Feature names. Inferred if None.
    max_degree : int
        1 = main effects only, 2 = with pairwise interactions.
    n_bootstrap : int
        Number of bootstrap samples for CIs. 0 to disable.
    robust_se : bool
        If True, use HC3 heteroscedasticity-consistent standard errors.
    max_interaction_features : int
        Maximum number of features allowed for degree=2 interactions.
        Raises ValueError if n_features exceeds this threshold.
    copy : bool
        If True, operate on a copy of adata.

    Returns
    -------
    dict
        Serialized DriverRegressionResult. Also stored in adata.uns['peach_driver_regression'].
    """
    from itertools import combinations

    import scipy.sparse as sp
    from scipy import stats

    from peach._core.utils.ilr_transform import ilr_transform, inverse_ilr

    if copy:
        adata = adata.copy()

    weights = get_archetype_weights(adata)
    K = weights.shape[1]
    n_cells = adata.n_obs

    # Default to pathway scores if available — warn about upcoming behavior change
    if feature_matrix is None and "pathway_scores" in adata.obsm:
        warnings.warn(
            "archetype_driver_regression() auto-selects pathway_scores when "
            "available. This will change in v0.6.0 to always default to adata.X. "
            "Pass feature_matrix='pathway_scores' explicitly to keep current "
            "behavior and silence this warning.",
            FutureWarning,
            stacklevel=2,
        )
        feature_matrix = "pathway_scores"

    X_features, feat_names = resolve_features(adata, feature_matrix, feature_names)
    n_features = len(feat_names)

    # Safeguard for interaction terms
    if max_degree >= 2 and n_features > max_interaction_features:
        raise ValueError(
            f"n_features ({n_features}) exceeds max_interaction_features "
            f"({max_interaction_features}) for degree=2 interactions. "
            f"Reduce features or raise max_interaction_features."
        )

    # Transform weights to ILR space
    ilr_weights = ilr_transform(weights)  # [n_cells, K-1]

    # Densify features if sparse
    if sp.issparse(X_features):
        X_features = X_features.toarray()
    else:
        X_features = np.asarray(X_features, dtype=np.float64)

    # Build design matrix WITH intercept
    if max_degree >= 2:
        interaction_pairs = list(combinations(range(n_features), 2))
        interactions = np.column_stack(
            [X_features[:, j] * X_features[:, k] for j, k in interaction_pairs]
        )
        design = np.column_stack([np.ones((n_cells, 1)), X_features, interactions])
    else:
        interaction_pairs = []
        design = np.column_stack([np.ones((n_cells, 1)), X_features])

    # Fit K-1 regressions (one per ILR component)
    n_params = design.shape[1]
    main_coefs_ilr = np.zeros((K - 1, n_features))
    interaction_coefs_ilr = (
        np.zeros((K - 1, len(interaction_pairs))) if interaction_pairs else None
    )
    main_pvalues = np.zeros((K - 1, n_features))
    interaction_pvalues = (
        np.zeros((K - 1, len(interaction_pairs))) if interaction_pairs else None
    )
    r_squared = np.zeros(K - 1)
    intercepts = np.zeros(K - 1)

    # Invert design matrix with explicit condition number check.
    # np.linalg.solve does NOT raise on near-singular matrices (cond ~1e10 silently
    # returns garbage). Mirror the guard in ols_fit.
    DtD = design.T @ design
    _, s_dvals, _ = np.linalg.svd(DtD)
    cond_d = s_dvals[0] / max(s_dvals[-1], np.finfo(float).tiny)
    if cond_d > 1e12:
        warnings.warn(
            f"Driver regression design matrix is near-singular "
            f"(condition number {cond_d:.1e}). "
            "Falling back to pseudoinverse for numerical stability.",
            RuntimeWarning,
        )
        DtD_inv = np.linalg.pinv(DtD)
    else:
        try:
            DtD_inv = np.linalg.solve(DtD, np.eye(DtD.shape[0]))
        except np.linalg.LinAlgError:
            warnings.warn("DtD is singular; falling back to pseudoinverse.", RuntimeWarning)
            DtD_inv = np.linalg.pinv(DtD)

    for m in range(K - 1):
        y = ilr_weights[:, m]  # [n_cells]
        beta = DtD_inv @ (design.T @ y)  # [n_params]
        residuals = y - design @ beta
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared[m] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        intercepts[m] = beta[0]
        main_coefs_ilr[m] = beta[1 : n_features + 1]
        if interaction_pairs:
            interaction_coefs_ilr[m] = beta[n_features + 1 :]

        # Standard errors
        if robust_se:
            # HC3 sandwich estimator — clip H_diag BEFORE division
            H_diag = np.sum((design @ DtD_inv) * design, axis=1)
            H_diag = np.clip(H_diag, 0, 1 - 1e-10)
            adjustment = 1.0 / (1 - H_diag)
            e_adj = residuals * adjustment
            meat = design.T @ (design * (e_adj**2)[:, np.newaxis])
            sandwich = DtD_inv @ meat @ DtD_inv
            se = np.sqrt(np.maximum(np.diag(sandwich), 0))
        else:
            sigma2 = ss_res / max(n_cells - n_params, 1)
            se = np.sqrt(sigma2 * np.diag(DtD_inv))

        df = max(n_cells - n_params, 1)
        t_stats = np.where(se > 0, beta / se, 0.0)
        pvals = 2 * stats.t.sf(np.abs(t_stats), df=df)

        main_pvalues[m] = pvals[1 : n_features + 1]
        if interaction_pairs:
            interaction_pvalues[m] = pvals[n_features + 1 :]

    # Back-transform coefficients to simplex space
    # For each feature, the ILR coefficient vector describes a direction
    # in ILR space. We back-transform to see the per-archetype effect.
    center = inverse_ilr(np.zeros((1, K - 1)))[0]  # [K]

    main_coefs_simplex = np.zeros((K, n_features))
    for g in range(n_features):
        ilr_vec = main_coefs_ilr[:, g]  # [K-1]
        perturbed = inverse_ilr(ilr_vec.reshape(1, -1))[0]  # [K]
        main_coefs_simplex[:, g] = perturbed - center

    interaction_coefs_simplex = None
    if interaction_pairs:
        interaction_coefs_simplex = np.zeros((K, len(interaction_pairs)))
        for idx in range(len(interaction_pairs)):
            ilr_vec = interaction_coefs_ilr[:, idx]
            perturbed = inverse_ilr(ilr_vec.reshape(1, -1))[0]
            interaction_coefs_simplex[:, idx] = perturbed - center

    # Bootstrap CIs
    main_ci_lower = None
    main_ci_upper = None
    if n_bootstrap > 0:
        main_ci_lower, main_ci_upper = _bootstrap_driver_cis(
            ilr_weights,
            X_features,
            n_features=n_features,
            K=K,
            n_bootstrap=n_bootstrap,
            max_degree=1,  # CIs on main effects only
        )

    # Per-ILR-component FDR correction (one family per component column)
    main_pvalues_fdr = np.ones_like(main_pvalues)
    for col in range(main_pvalues.shape[0]):  # iterate ILR components
        col_pvals = np.clip(main_pvalues[col, :], np.finfo(float).tiny, 1.0)
        _, col_fdr, _, _ = multipletests(col_pvals, method="fdr_bh")
        main_pvalues_fdr[col, :] = col_fdr

    interaction_pvalues_fdr = None
    if interaction_pvalues is not None:
        interaction_pvalues_fdr = np.ones_like(interaction_pvalues)
        for col in range(interaction_pvalues.shape[0]):  # iterate ILR components
            col_pvals = np.clip(interaction_pvalues[col, :], np.finfo(float).tiny, 1.0)
            _, col_fdr, _, _ = multipletests(col_pvals, method="fdr_bh")
            interaction_pvalues_fdr[col, :] = col_fdr

    result = DriverRegressionResult(
        feature_names=feat_names,
        n_cells=n_cells,
        n_features=n_features,
        n_archetypes=K,
        main_coefficients_ilr=main_coefs_ilr,
        interaction_coefficients_ilr=interaction_coefs_ilr,
        main_coefficients=main_coefs_simplex,
        interaction_coefficients=interaction_coefs_simplex,
        main_pvalues=main_pvalues,
        main_pvalues_fdr=main_pvalues_fdr,
        interaction_pvalues=interaction_pvalues,
        interaction_pvalues_fdr=interaction_pvalues_fdr,
        main_ci_lower=main_ci_lower,
        main_ci_upper=main_ci_upper,
        r_squared=r_squared,
        intercepts=intercepts,
    )

    serialized = result.to_serializable()
    store_result(adata, "driver_regression", serialized)
    return serialized


def _bootstrap_driver_cis(
    ilr_weights, X_features, *, n_features, K, n_bootstrap, max_degree=1,
    ci_level=0.95, seed=42,
):
    """Bootstrap CIs for driver regression main-effect coefficients.

    Returns back-transformed simplex-space CIs: (ci_lower, ci_upper),
    each [K, n_features].
    """
    from peach._core.utils.ilr_transform import inverse_ilr

    rng = np.random.default_rng(seed)
    n = ilr_weights.shape[0]

    # Build design matrix (intercept + features, no interactions for CIs)
    design = np.column_stack([np.ones((n, 1)), X_features])

    center = inverse_ilr(np.zeros((1, K - 1)))[0]  # [K]

    boot_simplex_coefs = np.empty((n_bootstrap, K, n_features))
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        D_boot = design[idx]
        Y_boot = ilr_weights[idx]

        try:
            DtD_boot = D_boot.T @ D_boot
            DtD_inv_boot = np.linalg.solve(DtD_boot, np.eye(DtD_boot.shape[0]))
        except np.linalg.LinAlgError:
            boot_simplex_coefs[b] = np.nan
            continue

        # [K-1, n_features] main coefficients in ILR space
        main_ilr = np.zeros((K - 1, n_features))
        for m in range(K - 1):
            beta = DtD_inv_boot @ (D_boot.T @ Y_boot[:, m])
            main_ilr[m] = beta[1 : n_features + 1]

        # Back-transform each feature
        for g in range(n_features):
            perturbed = inverse_ilr(main_ilr[:, g].reshape(1, -1))[0]
            boot_simplex_coefs[b, :, g] = perturbed - center

    alpha = 1 - ci_level
    ci_lower = np.nanpercentile(boot_simplex_coefs, 100 * alpha / 2, axis=0)
    ci_upper = np.nanpercentile(boot_simplex_coefs, 100 * (1 - alpha / 2), axis=0)
    return ci_lower, ci_upper

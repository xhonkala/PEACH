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

    # Permutation test for model significance
    permutation_pvalue = None
    permutation_pvalue_fdr = None

    if permutation_test:
        permutation_pvalue, permutation_pvalue_fdr = _permutation_test_regression(
            weights, Y, n_permutations=n_permutations,
            observed_r2=result1["r_squared"],
        )

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
        permutation_pvalue=permutation_pvalue,
        permutation_pvalue_fdr=permutation_pvalue_fdr,
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
        perm_result = ols_fit(W_perm, Y_dense, robust_se=False)
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
            result = ols_fit(W_boot, Y_boot, robust_se=False)
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
) -> DriverRegressionResult:
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
    DriverRegressionResult
        Also stored in adata.uns['peach_driver_regression'].
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

    # Default to pathway scores if available
    if feature_matrix is None and "pathway_scores" in adata.obsm:
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

    DtD = design.T @ design
    try:
        DtD_inv = np.linalg.solve(DtD, np.eye(DtD.shape[0]))
    except np.linalg.LinAlgError:
        raise ValueError(
            "Feature design matrix is singular. This usually means features "
            "are perfectly collinear or n_cells < n_parameters."
        )

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
        interaction_pvalues=interaction_pvalues,
        main_ci_lower=main_ci_lower,
        main_ci_upper=main_ci_upper,
        r_squared=r_squared,
        intercepts=intercepts,
    )

    store_result(adata, "driver_regression", result.to_serializable())
    return result


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

"""Core compute functions for archetype comparison: MMD, feature similarity, Wald contrasts."""

import numpy as np
from itertools import combinations
from scipy import stats
from anndata import AnnData

from .feature_utils import get_archetype_weights, resolve_regression_result


def _get_assignments_and_k(adata, weights):
    """Get integer assignments and K from stored labels or argmax fallback.

    Uses stored labels only when their count matches the weight matrix
    dimension. Otherwise falls back to argmax to avoid shape mismatches
    (e.g., when central archetype inflates the label count beyond K).

    Returns (assign, K) where assign is 0-indexed integer array and K is
    the number of groups.
    """
    K_weights = weights.shape[1]
    if "archetypes" in adata.obs.columns:
        raw_labels = adata.obs["archetypes"].astype(str).values
        unique_labels = sorted(set(raw_labels) - {"no_archetype", "nan"})
        if len(unique_labels) == K_weights:
            label_to_idx = {l: i for i, l in enumerate(unique_labels)}
            assign = np.array([label_to_idx.get(str(l), -1) for l in raw_labels])
            return assign, K_weights

    # Fallback: argmax of weights (always consistent with weight matrix)
    assign = np.argmax(weights, axis=1)
    return assign, K_weights


def compute_archetype_mmd(
    adata: AnnData,
    adata_b: AnnData | None = None,
    *,
    pca_key: str = "X_pca",
    n_permutations: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute K x K MMD matrix between archetype cell populations.

    Uses hard assignments (argmax of weights) to define populations,
    then computes pairwise MMD with RBF kernel + permutation p-values.

    Parameters
    ----------
    adata : AnnData
        Must have archetype weights and PCA coordinates.
    adata_b : AnnData or None
        If provided, compute K_A x K_B between-fit comparison.
    pca_key : str
        Key in obsm for cell coordinates.
    n_permutations : int
        Permutations for p-value. 0 to skip.
    seed : int

    Returns
    -------
    (mmd_matrix, pvalue_matrix) : tuple of np.ndarray
    """
    from .flow_matching import compute_mmd

    rng = np.random.default_rng(seed)
    weights_a = get_archetype_weights(adata)
    pca_a = adata.obsm[pca_key]

    if adata_b is not None:
        weights_b = get_archetype_weights(adata_b)
        pca_b = adata_b.obsm[pca_key]
    else:
        weights_b = weights_a
        pca_b = pca_a

    assign_a, K_a = _get_assignments_and_k(adata, weights_a)
    if adata_b is not None:
        assign_b, K_b = _get_assignments_and_k(adata_b, weights_b)
    else:
        assign_b, K_b = assign_a, K_a

    mmd_matrix = np.zeros((K_a, K_b))
    pvalue_matrix = np.ones((K_a, K_b))

    for i in range(K_a):
        cells_i = pca_a[assign_a == i]
        for j in range(K_b):
            if adata_b is None and j <= i:
                if j < i:
                    mmd_matrix[i, j] = mmd_matrix[j, i]
                    pvalue_matrix[i, j] = pvalue_matrix[j, i]
                continue

            cells_j = pca_b[assign_b == j]

            if len(cells_i) < 2 or len(cells_j) < 2:
                mmd_matrix[i, j] = np.nan
                pvalue_matrix[i, j] = np.nan
                continue

            observed_mmd = compute_mmd(cells_i, cells_j)
            mmd_matrix[i, j] = observed_mmd

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

    if adata_b is None:
        mmd_matrix = np.maximum(mmd_matrix, mmd_matrix.T)
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

    Returns
    -------
    dict with silhouette_per_archetype, silhouette_overall,
         spearman_matrix, spearman_pvalue_matrix, n_shared_features
    """
    from sklearn.metrics import silhouette_score, silhouette_samples

    reg_a = resolve_regression_result(adata, prefer="genes")
    if reg_a is None:
        raise ValueError(
            "No regression results. Run pc.tl.feature_simplex_regression() first."
        )

    coefs_a = np.asarray(reg_a["vertex_coefficients"])
    names_a = list(reg_a["feature_names"])
    K_a = coefs_a.shape[1]

    if adata_b is not None:
        reg_b = resolve_regression_result(adata_b, prefer="genes")
        if reg_b is None:
            raise ValueError("No regression results in adata_b.")
        coefs_b = np.asarray(reg_b["vertex_coefficients"])
        names_b = list(reg_b["feature_names"])
        K_b = coefs_b.shape[1]

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

    spearman_matrix = np.zeros((K_a, K_b))
    spearman_pvalue_matrix = np.ones((K_a, K_b))

    for i in range(K_a):
        for j in range(K_b):
            rho, pval = stats.spearmanr(coefs_a_shared[:, i], coefs_b_shared[:, j])
            spearman_matrix[i, j] = rho
            spearman_pvalue_matrix[i, j] = pval

    # Global FDR correction across all K_a x K_b Spearman tests (clamp underflowed zeros)
    from statsmodels.stats.multitest import multipletests as _mt_spearman
    all_spearman_pvals = np.clip(spearman_pvalue_matrix.ravel(), np.finfo(float).tiny, 1.0)
    _, spearman_fdr_flat, _, _ = _mt_spearman(all_spearman_pvals, method="fdr_bh")
    spearman_pvalue_fdr_matrix = spearman_fdr_flat.reshape(spearman_pvalue_matrix.shape)

    # Silhouette on PCA space
    weights_a = get_archetype_weights(adata)
    labels, _ = _get_assignments_and_k(adata, weights_a)
    pca = adata.obsm[pca_key]

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
        sil_overall = float(silhouette_score(pca_sub, labels_sub))
        sil_samples = silhouette_samples(pca_sub, labels_sub)
        sil_per = np.array([
            float(sil_samples[labels_sub == k].mean()) if np.any(labels_sub == k) else 0.0
            for k in range(K_a)
        ])

    return {
        "silhouette_per_archetype": sil_per,
        "silhouette_overall": sil_overall,
        "spearman_matrix": spearman_matrix,
        "spearman_pvalue_matrix": spearman_pvalue_matrix,
        "spearman_pvalue_fdr_matrix": spearman_pvalue_fdr_matrix,
        "n_shared_features": n_shared,
    }


def compute_wald_contrasts(
    adata: AnnData,
    *,
    robust_se: bool = True,
) -> dict:
    """Pairwise Wald contrasts beta_k - beta_j with SEs from regression covariance.

    Re-runs degree-1 regression with return_covariance=True, then computes
    contrasts for all K*(K-1)/2 pairs.

    Returns
    -------
    dict with pairs, delta_beta, delta_se, z_scores, pvalues, pvalues_fdr,
         feature_names, n_features, n_archetypes
    """
    from statsmodels.stats.multitest import multipletests

    reg = resolve_regression_result(adata, prefer="genes")
    if reg is None:
        raise ValueError(
            "No regression results. Run pc.tl.feature_simplex_regression() first."
        )

    weights = get_archetype_weights(adata)
    feat_names = list(reg["feature_names"])
    K = weights.shape[1]
    n_features = len(feat_names)
    n_cells = adata.n_obs
    df = max(n_cells - K, 1)

    # Use cached covariance if available; fall back to re-running regression
    cached_cov = reg.get("vertex_covariance")
    if cached_cov is not None:
        beta = np.asarray(reg["vertex_coefficients"])
        cov_list = [np.asarray(c) for c in cached_cov]
    else:
        from .simplex_regression import ols_fit, scheffe_design_matrix
        from .feature_utils import resolve_features
        Y, _ = resolve_features(adata, None, feat_names)
        W, _ = scheffe_design_matrix(weights, degree=1)
        fit = ols_fit(W, Y, robust_se=robust_se, return_covariance=True)
        beta = fit["coefficients"]  # [n_features, K]
        cov_list = fit["covariance"]  # list of K x K matrices

    pairs = list(combinations(range(K), 2))
    delta_beta = {}
    delta_se = {}
    z_scores = {}
    pvalues = {}
    pvalues_fdr = {}

    # First pass: compute per-pair statistics, collect raw p-values
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
        pval = 2 * stats.t.sf(np.abs(z), df=df)

        delta_beta[(j, k)] = d_beta
        delta_se[(j, k)] = d_se
        z_scores[(j, k)] = z
        pvalues[(j, k)] = pval

        all_pvals.append(pval)
        pair_slices[(j, k)] = slice(offset, offset + n_features)
        offset += n_features

    # Global FDR correction across ALL pairs (not per-pair)
    # Clamp underflowed zeros, then filter trivial tests (SE=0 → p=1) to avoid diluting FDR
    all_pvals_flat = np.clip(np.concatenate(all_pvals), np.finfo(float).tiny, 1.0)
    testable = all_pvals_flat < 1.0
    all_fdr = np.ones_like(all_pvals_flat)
    if testable.any():
        _, fdr_vals, _, _ = multipletests(all_pvals_flat[testable], method="fdr_bh")
        all_fdr[testable] = fdr_vals

    for j, k in pairs:
        pvalues_fdr[(j, k)] = all_fdr[pair_slices[(j, k)]]

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

"""Core compute functions for archetype comparison: MMD, feature similarity, Wald contrasts."""

import numpy as np
from itertools import combinations
from scipy import stats
from anndata import AnnData

from .feature_utils import get_archetype_weights, resolve_regression_result


def _weighted_mmd_pair(pca_x, w_x, pca_y, w_y, bw):
    """Weighted MMD for one archetype between two populations.

    Each cell's contribution to the kernel sums is weighted by its archetype
    weight, giving an unbiased soft-assignment MMD instead of hard argmax.

    Parameters
    ----------
    pca_x : np.ndarray, shape [n, dim]
        PCA coordinates for population X.
    w_x : np.ndarray, shape [n]
        Archetype weights for population X (one archetype column).
    pca_y : np.ndarray, shape [m, dim]
        PCA coordinates for population Y.
    w_y : np.ndarray, shape [m]
        Archetype weights for population Y.
    bw : float
        RBF kernel bandwidth.

    Returns
    -------
    float
        Weighted MMD^2 value.
    """
    from scipy.spatial.distance import cdist

    K_xx = np.exp(-cdist(pca_x, pca_x, 'sqeuclidean') / (2 * bw**2))
    K_yy = np.exp(-cdist(pca_y, pca_y, 'sqeuclidean') / (2 * bw**2))
    K_xy = np.exp(-cdist(pca_x, pca_y, 'sqeuclidean') / (2 * bw**2))

    wx = w_x / (w_x.sum() + 1e-10)
    wy = w_y / (w_y.sum() + 1e-10)

    W_xx = np.outer(wx, wx)
    np.fill_diagonal(W_xx, 0)
    W_yy = np.outer(wy, wy)
    np.fill_diagonal(W_yy, 0)
    W_xy = np.outer(wx, wy)

    sum_wx2 = np.sum(wx**2)
    sum_wy2 = np.sum(wy**2)

    term1 = (W_xx * K_xx).sum() / max(1 - sum_wx2, 1e-10)
    term2 = (W_yy * K_yy).sum() / max(1 - sum_wy2, 1e-10)
    term3 = 2 * (W_xy * K_xy).sum()
    return float(term1 + term2 - term3)


def compute_archetype_mmd(
    adata: AnnData,
    adata_b: AnnData | None = None,
    *,
    pca_key: str = "X_pca",
    n_permutations: int = 1000,
    seed: int = 42,
    max_samples: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute K x K weighted MMD matrix between archetype populations.

    Uses full archetype weight vectors (soft assignment) rather than hard
    argmax. For each archetype pair (i, j), computes weighted MMD where
    each cell's contribution is scaled by its archetype weight.

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
    max_samples : int
        Subsample to this many cells for kernel computation.

    Returns
    -------
    (mmd_matrix, pvalue_matrix) : tuple of np.ndarray
    """
    from scipy.spatial.distance import pdist

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
    if len(pca_a) > max_samples:
        idx = rng.choice(len(pca_a), max_samples, replace=False)
        pca_a = pca_a[idx]
        weights_a = weights_a[idx]
    if len(pca_b) > max_samples:
        idx = rng.choice(len(pca_b), max_samples, replace=False)
        pca_b = pca_b[idx]
        weights_b = weights_b[idx]

    # Bandwidth via median heuristic on combined subsample
    combined_sub = np.vstack([pca_a[:min(500, len(pca_a))],
                              pca_b[:min(500, len(pca_b))]])
    bw = max(float(np.median(pdist(combined_sub))), 1e-6)

    mmd_matrix = np.zeros((K_a, K_b))
    pvalue_matrix = np.ones((K_a, K_b))

    for i in range(K_a):
        for j in range(K_b):
            if adata_b is None and j < i:
                mmd_matrix[i, j] = mmd_matrix[j, i]
                pvalue_matrix[i, j] = pvalue_matrix[j, i]
                continue

            # For within-fit diagonal (same archetype vs itself), MMD = 0
            if adata_b is None and i == j:
                mmd_matrix[i, j] = 0.0
                pvalue_matrix[i, j] = 1.0
                continue

            observed_mmd = _weighted_mmd_pair(
                pca_a, weights_a[:, i], pca_b, weights_b[:, j], bw
            )
            mmd_matrix[i, j] = observed_mmd

            if n_permutations > 0:
                n_a = len(pca_a)
                n_b = len(pca_b)
                if adata_b is None:
                    # Within-fit: shuffle weight columns i and j
                    null_mmds = np.empty(n_permutations)
                    for p in range(n_permutations):
                        # Permute cell identities
                        perm = rng.permutation(n_a)
                        null_mmds[p] = _weighted_mmd_pair(
                            pca_a, weights_a[perm, i],
                            pca_b, weights_b[:, j], bw
                        )
                else:
                    # Between-fit: permute cell identities between conditions
                    combined_pca = np.vstack([pca_a, pca_b])
                    combined_w_i = np.concatenate([weights_a[:, i], weights_b[:, i]
                                                   if i < K_b else np.zeros(n_b)])
                    combined_w_j = np.concatenate([weights_a[:, j]
                                                   if j < K_a else np.zeros(n_a),
                                                   weights_b[:, j]])
                    null_mmds = np.empty(n_permutations)
                    for p in range(n_permutations):
                        perm = rng.permutation(n_a + n_b)
                        null_mmds[p] = _weighted_mmd_pair(
                            combined_pca[perm[:n_a]], combined_w_i[perm[:n_a]],
                            combined_pca[perm[n_a:]], combined_w_j[perm[n_a:]], bw
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
) -> dict:
    """Spearman correlation on regression coefficients with FDR pre-filter.

    Only computes Spearman on features where at least one vertex has
    ``vertex_pvalues_fdr < 0.05`` in the regression result.

    Requires simplex regression results in adata.uns['peach_simplex_regression'].

    Returns
    -------
    dict with spearman_matrix, spearman_pvalue_matrix,
         spearman_pvalue_fdr_matrix, n_shared_features, n_significant_features
    """
    reg_a = resolve_regression_result(adata, feature_type="genes")
    if reg_a is None:
        raise ValueError(
            "No regression results. Run pc.tl.feature_simplex_regression() first."
        )

    coefs_a = np.asarray(reg_a["vertex_coefficients"])
    names_a = list(reg_a["feature_names"])
    K_a = coefs_a.shape[1]

    # FDR pre-filter: only keep features with at least one significant vertex
    fdr_a = reg_a.get("vertex_pvalues_fdr")
    if fdr_a is not None:
        fdr_a = np.asarray(fdr_a)
        sig_mask_a = np.any(fdr_a < 0.05, axis=1)  # [n_features] boolean
    else:
        # No FDR available — keep all features
        sig_mask_a = np.ones(len(names_a), dtype=bool)

    if adata_b is not None:
        reg_b = resolve_regression_result(adata_b, feature_type="genes")
        if reg_b is None:
            raise ValueError("No regression results in adata_b.")
        coefs_b = np.asarray(reg_b["vertex_coefficients"])
        names_b = list(reg_b["feature_names"])
        K_b = coefs_b.shape[1]

        fdr_b = reg_b.get("vertex_pvalues_fdr")
        if fdr_b is not None:
            fdr_b = np.asarray(fdr_b)
            sig_mask_b = np.any(fdr_b < 0.05, axis=1)
        else:
            sig_mask_b = np.ones(len(names_b), dtype=bool)

        # Find shared features, then apply FDR filter from either fit
        shared_all = sorted(set(names_a) & set(names_b))
        idx_a_all = [names_a.index(g) for g in shared_all]
        idx_b_all = [names_b.index(g) for g in shared_all]

        # A feature passes if significant in either fit
        sig_shared = [
            sig_mask_a[ia] or sig_mask_b[ib]
            for ia, ib in zip(idx_a_all, idx_b_all)
        ]
        shared = [g for g, s in zip(shared_all, sig_shared) if s]
        idx_a = [names_a.index(g) for g in shared]
        idx_b = [names_b.index(g) for g in shared]
        coefs_a_shared = coefs_a[idx_a]
        coefs_b_shared = coefs_b[idx_b]
        n_significant = len(shared)
    else:
        # Within-fit: filter to significant features
        sig_idx = np.where(sig_mask_a)[0]
        n_significant = len(sig_idx)
        if n_significant > 0:
            coefs_a_shared = coefs_a[sig_idx]
        else:
            coefs_a_shared = coefs_a  # fallback: use all if none significant
            n_significant = 0
        coefs_b_shared = coefs_a_shared
        K_b = K_a
        shared = [names_a[i] for i in sig_idx] if n_significant > 0 else names_a

    n_shared = len(shared)

    spearman_matrix = np.zeros((K_a, K_b))
    spearman_pvalue_matrix = np.ones((K_a, K_b))

    if n_shared >= 3:
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

    return {
        "spearman_matrix": spearman_matrix,
        "spearman_pvalue_matrix": spearman_pvalue_matrix,
        "spearman_pvalue_fdr_matrix": spearman_pvalue_fdr_matrix,
        "n_shared_features": n_shared,
        "n_significant_features": n_significant,
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

    reg = resolve_regression_result(adata, feature_type="genes")
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
        pval = np.clip(2 * stats.t.sf(np.abs(z), df=df), np.finfo(float).tiny, 1.0)

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

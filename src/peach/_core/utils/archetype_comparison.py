"""Core compute functions for archetype comparison: MMD, feature similarity, Wald contrasts."""

import logging
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
                    # Within-fit: permute BOTH columns independently
                    null_mmds = np.empty(n_permutations)
                    for p in range(n_permutations):
                        perm_i = rng.permutation(n_a)
                        perm_j = rng.permutation(n_a)
                        null_mmds[p] = _weighted_mmd_pair(
                            pca_a, weights_a[perm_i, i],
                            pca_b, weights_b[perm_j, j], bw
                        )
                else:
                    # Between-fit permutation: can only permute
                    # meaningfully when both i and j have real weight
                    # columns in both fits.
                    if i >= K_b or j >= K_a:
                        pvalue_matrix[i, j] = float("nan")
                        continue

                    combined_pca = np.vstack([pca_a, pca_b])
                    combined_w_i = np.concatenate([weights_a[:, i], weights_b[:, i]])
                    combined_w_j = np.concatenate([weights_a[:, j], weights_b[:, j]])
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

        # Pre-build index maps for O(1) lookup
        idx_map_a = {name: i for i, name in enumerate(names_a)}
        idx_map_b = {name: i for i, name in enumerate(names_b)}

        # Find shared features, then apply FDR filter from either fit
        shared_all = sorted(set(names_a) & set(names_b))
        idx_a_all = [idx_map_a[g] for g in shared_all]
        idx_b_all = [idx_map_b[g] for g in shared_all]

        # A feature passes if significant in either fit
        sig_shared = [
            sig_mask_a[ia] or sig_mask_b[ib]
            for ia, ib in zip(idx_a_all, idx_b_all)
        ]
        shared = [g for g, s in zip(shared_all, sig_shared) if s]
        idx_a = [idx_map_a[g] for g in shared]
        idx_b = [idx_map_b[g] for g in shared]
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
    feature_type: str = "genes",
) -> dict:
    """Pairwise Wald contrasts beta_k - beta_j with SEs from regression covariance.

    Re-runs degree-1 regression with return_covariance=True, then computes
    contrasts for all K*(K-1)/2 pairs.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with archetype weights and regression results.
    robust_se : bool
        Use HC3 heteroscedasticity-consistent standard errors.
    feature_type : str
        Which regression result to use: "genes" (default) or "pathways".

    Returns
    -------
    dict with pairs, delta_beta, delta_se, z_scores, pvalues, pvalues_fdr,
         feature_names, n_features, n_archetypes
    """
    from statsmodels.stats.multitest import multipletests

    reg = resolve_regression_result(adata, feature_type=feature_type)
    if reg is None:
        raise ValueError(
            f"No regression results for feature_type='{feature_type}'. "
            "Run pc.tl.feature_simplex_regression() first."
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
        feature_source = reg.get("feature_source")
        Y, _ = resolve_features(adata, feature_source, feat_names)
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

    # Per-pair FDR correction (one family per archetype pair)
    for j, k in pairs:
        pair_pvals = np.clip(pvalues[(j, k)], np.finfo(float).tiny, 1.0)
        testable = pair_pvals < 1.0
        pair_fdr = np.ones_like(pair_pvals)
        if testable.any():
            _, fdr_vals, _, _ = multipletests(pair_pvals[testable], method="fdr_bh")
            pair_fdr[testable] = fdr_vals
        pvalues_fdr[(j, k)] = pair_fdr

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


def compute_archetype_correspondence(
    source_weights: np.ndarray,
    source_coords: np.ndarray,
    target_weights: np.ndarray,
    target_coords: np.ndarray,
    *,
    k: int = 10,
    method: str = "hard",
) -> dict:
    """Compute archetype correspondence matrix between two Deep_AA fits.

    Answers the question: "Given a source cell in source archetype i, what
    fraction of its transported neighborhood mass lands in target archetype
    j?" The result is a K_src × K_tgt matrix that bridges two fits and
    supports cross-fit Sankey, per-pair flow_between, and Wald alignment.

    For each source cell, the k nearest neighbors in target coordinate space
    are found, their target-model archetype weights are averaged. How that
    per-cell average is then aggregated into a per-source-archetype row
    depends on ``method``.

    Decision guide: which method to use
    -----------------------------------
    Quick rule: **always start with "hard" on real data**. Only switch to
    "sharp" or "soft" if you have a specific reason (synthetic validation,
    backward compatibility, or mathematical smoothness requirement).

    +-------------+---------------------------+---------------------------+
    | Method      | Use when                  | Fails when                |
    +=============+===========================+===========================+
    | ``"hard"``  | Real scRNA-seq data where | Every source archetype is |
    | (default)   | source_weight_concentra-  | extrapolated (0-2 cells   |
    |             | tion < ~0.5 (diffuse      | hard-argmaxed to it) —    |
    |             | weights). Recommended for | then sparse_archetypes is |
    |             | almost all production     | non-empty and those rows  |
    |             | analyses.                 | are zeroed. Bad fit not   |
    |             |                           | a bad method — retrain.   |
    +-------------+---------------------------+---------------------------+
    | ``"sharp"`` | Moderately peaked source  | Still degrades to rank-1  |
    |             | (concentration 0.5-0.7)   | under severely diffuse    |
    |             | where you want a          | weights (worse than hard  |
    |             | smoothed / differentiable | on real HSC/CMP: column   |
    |             | signal. Empirically mid-  | CV ~0.12 vs hard's ~0.59  |
    |             | way between soft and hard.| in the r9 investigation). |
    +-------------+---------------------------+---------------------------+
    | ``"soft"``  | Synthetic tests with well-| Real data where source    |
    |             | peaked cells (max weight  | weights are diffuse.      |
    |             | >= 0.9). Original outer-  | Produces near-identical   |
    |             | product formulation —     | rows (rank-1 collapse),   |
    |             | smooth, differentiable,   | column CV < 0.1 = the     |
    |             | backward-compatible.      | correspondence carries    |
    |             |                           | essentially no signal.    |
    +-------------+---------------------------+---------------------------+

    Rank-1 collapse failure mode
    ----------------------------
    For ``method="soft"`` the mass computation reduces to

        mass[i, j] = Σ_cells source_w[c, i] * mean_target_w[c, j]

    which factorizes as ``source_weights.T @ nn_tw_mean``. When source
    weights are diffuse, every row of ``nn_tw_mean`` is similar (the k-NN
    averaging smooths out per-cell variation), so the matrix product
    degenerates to a rank-1 outer product of the source marginal and the
    target marginal. All rows become scaled copies of the same vector —
    the Markov matrix looks uniform and the correspondence carries no
    signal. On real HSC/CMP data this manifested as column CV ≈ 0.084 and
    visually identical Sankey rows (see the r9 investigation).

    The hard method avoids this by hard-assigning each source cell to ONE
    archetype (argmax) before aggregation. A diffuse weight vector still
    picks exactly one label, so the per-archetype-group aggregation
    preserves the true variation between source cells. Column CV jumped
    from ~0.084 to ~0.586 on the same HSC/CMP data.

    Sparse source archetype handling (hard method only)
    ---------------------------------------------------
    After hard argmax, some source archetypes may have very few or zero
    cells assigned to them (they sit outside the data cloud). These are
    "extrapolated" archetypes — see W-B10's ``compute_archetype_to_centroid
    _distance`` for a complementary diagnostic.

    Any source archetype with hard-argmax occupancy below
    ``max(2, int(0.02 * n_src))`` has its mass row zeroed and its index
    added to the returned ``sparse_archetypes`` list. A ``logging.warning``
    is emitted naming the zeroed archetypes. This prevents a handful of
    outlier cells from producing a noisy correspondence row, but it is a
    symptom-not-cure intervention: the real fix is to retrain with
    parameters that keep archetypes inside the data cloud (see W-A5
    inflation_factor range and W-A6 PCHA init diagnostic).

    Consistency with the rest of the analysis stack
    -----------------------------------------------
    The hard method uses ``source_weights.argmax(axis=1)`` for labeling.
    This matches the assignment produced by
    ``bin_cells_by_archetype(..., method="argmax")`` (W-B13) when the
    ``cell_archetype_weights`` obsm is used as the weight source. When both
    are used together, the correspondence matrix, dotplots, Sankey
    diagrams, flow models, and pattern analyses all read from the SAME
    source-cell-to-archetype label vector.

    If you use ``bin_cells_by_archetype(method="bin_prop")`` (the library
    default) AND ``compute_archetype_correspondence(method="hard")``, the
    two label vectors can disagree for individual cells. Prefer either:

    1. ``bin_cells_by_archetype(method="argmax")`` + hard correspondence
       (recommended for cross-analysis consistency), or
    2. ``bin_cells_by_archetype(method="bin_prop")`` + soft correspondence
       (if you need the backward-compatible proportional binning AND
       accept the rank-1 collapse risk on diffuse weights).

    Interpreting source_weight_concentration
    ----------------------------------------
    The returned ``source_weight_concentration`` is
    ``source_weights.max(axis=1).mean()`` — the average of each cell's
    peak archetype weight. Use it as follows:

    - ``>= 0.7``: well-peaked weights; any method works. Synthetic fixtures
      typically fall here.
    - ``0.4 - 0.7``: moderately diffuse; hard is safe, sharp is acceptable,
      soft may still give signal.
    - ``< 0.4``: diffuse. Soft WILL produce rank-1 collapse. Use hard.
      Also consider that the fit itself may be underpowered (W-A3 full
      data, W-A5 wider inflation grid).

    Sanity check: call the function twice with different methods
    ------------------------------------------------------------
    If you are unsure whether the correspondence you see is signal or
    rank-1 collapse, call this function twice on the same inputs —
    once with ``method="hard"`` and once with ``method="soft"``. Compare
    the column CVs of the Markov matrices. If soft's column CV is
    dramatically lower (e.g. < 0.2) while hard's is higher (e.g. > 0.4),
    the soft result is unreliable and hard is the truth. If both agree,
    either method is fine.

    Parameters
    ----------
    source_weights : ndarray, shape [n_src, K_src]
        Row-stochastic archetype weights in the source model.
    source_coords : ndarray, shape [n_src, d]
        Source cell coordinates in the common (target) coordinate space.
        For cross-fit use cases this is the post-transport (e.g.
        flow_matching transported) source coordinates.
    target_weights : ndarray, shape [n_tgt, K_tgt]
        Row-stochastic archetype weights in the target model.
    target_coords : ndarray, shape [n_tgt, d]
        Target cell coordinates in the same coordinate space as
        source_coords.
    k : int, default 10
        Number of nearest target neighbors to average per source cell.
        Clipped to ``n_tgt`` if larger.
    method : {"hard", "soft", "sharp"}, default "hard"
        Aggregation method (see above). Hard is the recommended default
        for real data; soft is kept for backward compatibility / synthetic
        well-peaked cases.

    Returns
    -------
    dict with keys:
        "mass" : ndarray [K_src, K_tgt]
            Raw correspondence mass. Non-negative, not normalized. Rows for
            empty / sparse source archetypes are zero.
        "markov" : ndarray [K_src, K_tgt]
            Row-normalized ("Markov-like") transition matrix. Each row sums
            to 1 (empty rows are set to zero, not NaN).
        "source_mass_per_archetype" : ndarray [K_src]
            Per-archetype marginal mass (row sums of the mass matrix).
        "method" : str
            Which method was used (echoes the input).
        "source_weight_concentration" : float
            ``source_weights.max(axis=1).mean()`` — average peakedness of
            source weights. Values < 0.4 indicate diffuse weights where
            ``method="soft"`` is unreliable.
        "target_weight_concentration" : float
            Same diagnostic for target weights.
        "source_archetype_occupancy_hard" : ndarray [K_src]
            Per-archetype cell count under hard argmax of source weights
            (``np.bincount(src_labels, minlength=K_src)``). Always
            populated regardless of method choice.
        "sparse_archetypes" : list[int]
            Indices of source archetypes whose hard-argmax occupancy is
            below ``max(2, int(0.02 * n_src))``. In ``method="hard"``
            their mass rows are zeroed.

    Notes
    -----
    The default switched from ``"soft"`` to ``"hard"`` after empirical
    investigation on real HSC/CMP data showed soft-soft column CV of
    ~0.084 (rows visually identical) vs hard-argmax column CV of ~0.586
    (~7× more signal). See tests/test_core/test_archetype_correspondence.py
    for the synthetic reproduction and tests/test_core/
    test_correspondence_real_data.py for the real-data integration check.
    """
    from scipy.spatial import cKDTree

    if method not in ("hard", "soft", "sharp"):
        raise ValueError(
            f"method must be one of {{'hard', 'soft', 'sharp'}}, got {method!r}"
        )

    source_weights = np.asarray(source_weights, dtype=float)
    source_coords = np.asarray(source_coords)
    target_weights = np.asarray(target_weights, dtype=float)
    target_coords = np.asarray(target_coords)

    K_src = source_weights.shape[1]
    K_tgt = target_weights.shape[1]
    n_src = source_weights.shape[0]
    n_tgt = target_weights.shape[0]

    # Early return for empty inputs — avoid cKDTree on zero rows / IndexErrors
    # downstream. All diagnostics return zero, sparse list = all archetypes.
    if n_src == 0 or n_tgt == 0:
        return {
            "mass": np.zeros((K_src, K_tgt)),
            "markov": np.zeros((K_src, K_tgt)),
            "source_mass_per_archetype": np.zeros(K_src),
            "method": method,
            "source_weight_concentration": 0.0,
            "target_weight_concentration": 0.0,
            "source_archetype_occupancy_hard": np.zeros(K_src, dtype=int),
            "sparse_archetypes": list(range(K_src)),
        }

    k_eff = max(1, min(k, n_tgt))
    tree = cKDTree(target_coords)
    _, nn_idx = tree.query(source_coords, k=k_eff)
    # k=1 returns 1D index array; normalize to 2D
    if k_eff == 1:
        nn_idx = nn_idx.reshape(-1, 1)

    # ------------------------------------------------------------------
    # Diagnostics computed for ALL methods (independent of method choice)
    # ------------------------------------------------------------------
    src_concentration = float(source_weights.max(axis=1).mean())
    tgt_concentration = float(target_weights.max(axis=1).mean())

    src_labels = source_weights.argmax(axis=1)
    occupancy_hard = np.bincount(src_labels, minlength=K_src)

    sparse_threshold = max(2, int(0.02 * n_src))
    sparse_archetypes = [
        int(i) for i in range(K_src) if occupancy_hard[i] < sparse_threshold
    ]

    # ------------------------------------------------------------------
    # Mass matrix construction (vectorized)
    #
    # Shared precomputation: per-source-cell mean of k-NN target weight
    # vectors. Shape [n_src, K_tgt]. All three methods reduce this matrix
    # into a [K_src, K_tgt] mass matrix; only the reduction differs.
    # ------------------------------------------------------------------
    nn_tw_mean = target_weights[nn_idx].mean(axis=1)  # [n_src, K_tgt]
    mass = np.zeros((K_src, K_tgt))

    if method == "hard":
        # Hard-assign each source cell to its argmax archetype, then sum
        # its k-NN target-weight average into the assigned row. Use
        # np.add.at for unbuffered scatter-add since multiple cells share
        # labels. Sparse archetypes are masked out so their rows stay 0.
        if sparse_archetypes:
            sparse_set = set(sparse_archetypes)
            non_sparse_mask = np.array(
                [lab not in sparse_set for lab in src_labels], dtype=bool
            )
        else:
            non_sparse_mask = np.ones(n_src, dtype=bool)
        np.add.at(
            mass,
            src_labels[non_sparse_mask],
            nn_tw_mean[non_sparse_mask],
        )

    elif method == "soft":
        # Original soft-soft outer-product construction (rank-1 prone).
        # Vectorized as a matmul: sum_n source_w[n].T @ nn_tw_mean[n].
        mass = source_weights.T @ nn_tw_mean

    elif method == "sharp":
        # Square then renormalize, then soft outer product (matmul).
        sw_sharp = source_weights ** 2
        row_norms = sw_sharp.sum(axis=1, keepdims=True)
        row_norms_safe = np.where(row_norms < 1e-12, 1.0, row_norms)
        sw_sharp = sw_sharp / row_norms_safe
        mass = sw_sharp.T @ nn_tw_mean

    # ------------------------------------------------------------------
    # Row normalization → markov matrix
    # ------------------------------------------------------------------
    row_sums = mass.sum(axis=1, keepdims=True)
    row_sums_safe = np.where(row_sums < 1e-10, 1.0, row_sums)
    markov = mass / row_sums_safe
    # Zero out rows that had no source mass (avoid spurious uniform fallback)
    markov[row_sums.flatten() < 1e-10] = 0.0

    # ------------------------------------------------------------------
    # Diagnostic warnings (one per call, never raise)
    # ------------------------------------------------------------------
    if method == "hard" and sparse_archetypes:
        sparse_info = ", ".join(
            f"arch {i}: {int(occupancy_hard[i])} cells"
            for i in sparse_archetypes
        )
        logging.warning(
            "compute_archetype_correspondence(method='hard'): %d sparse "
            "source archetype(s) have < %d cells under hard argmax and "
            "their mass rows are zeroed (%s).",
            len(sparse_archetypes),
            sparse_threshold,
            sparse_info,
        )

    if method == "soft" and n_src > 0 and src_concentration < 0.4:
        logging.warning(
            "compute_archetype_correspondence(method='soft'): source "
            "weight concentration is %.3f (mean per-cell max < 0.4). "
            "Soft-soft outer-product construction is prone to rank-1 "
            "collapse with diffuse source weights — consider "
            "method='hard'.",
            src_concentration,
        )

    return {
        "mass": mass,
        "markov": markov,
        "source_mass_per_archetype": row_sums.flatten(),
        "method": method,
        "source_weight_concentration": src_concentration,
        "target_weight_concentration": tgt_concentration,
        "source_archetype_occupancy_hard": occupancy_hard,
        "sparse_archetypes": sparse_archetypes,
    }


def compute_correspondence_permutation_null(
    source_weights: np.ndarray,
    source_coords: np.ndarray,
    target_weights: np.ndarray,
    target_coords: np.ndarray,
    *,
    k: int = 10,
    n_perms: int = 200,
    swap_fractions: tuple = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50),
    seed: int = 42,
) -> dict:
    """Empirical permutation curve null for cross-fit archetype correspondence.

    Replaces the previous Gaussian-z-score null with an empirical, rank-based
    p-value derived from a degradation curve over a swap-fraction grid. For
    each ``f`` in ``swap_fractions``, ``n_perms`` permutations swap a fraction
    ``f`` of cells between the source and target *spatial pools*, recompute
    ``compute_archetype_correspondence(method="hard")`` on the scrambled pair,
    and accumulate the resulting [K_src, K_tgt] correspondence matrices into
    a per-fraction null distribution. The empirical p-value for each
    source-target pair is computed via rank against the null at the largest
    swap fraction, then BH-corrected across all K_src * K_tgt pairs.

    DESIGN NOTE — Swap interpretation
    ---------------------------------
    Source and target have *different* archetype weight column counts in
    general (K_src != K_tgt), so we cannot swap weight rows directly between
    the two populations. We also do not have access to either model to
    re-project coordinates after a swap. The cleanest valid interpretation is
    a **coordinate-only swap**:

    1. Pick ``m = int(f * min(n_src, n_tgt))`` source-row indices uniformly
       at random and a matching ``m`` target-row indices.
    2. Exchange ``source_coords[src_idx]`` with ``target_coords[tgt_idx]``
       row-by-row.
    3. Leave ``source_weights`` and ``target_weights`` UNTOUCHED — each
       weight row stays bound to its original index in its own model.
    4. Recompute ``compute_archetype_correspondence`` on the scrambled
       coordinate arrays.

    Effect on the null:
        At ``f=0`` no rows are exchanged, so the result equals the
        observed correspondence exactly. As ``f`` grows, source cells
        increasingly query the target k-NN tree from positions that did not
        come from the source population, and target cells sitting at
        "source-typical" locations contaminate the k-NN neighborhoods of
        the remaining unswapped source cells. This tests the null:
        "does the spatial correspondence break down when we scramble cell
        identity by geography?". A genuine source archetype that maps to a
        specific target archetype will show degradation (its mass entry
        regresses toward the matrix mean) only as ``f`` becomes large; a
        spurious link degrades immediately.

    Why coordinate-only and not row-exchange:
        Row exchange would require K_src == K_tgt and would make the swap
        also reshuffle the source archetype assignment vector, which
        conflates two distinct nulls. Coordinate-only is the minimal
        intervention that breaks the spatial linkage between source-cell
        identity and target-cell neighborhood.

    Parameters
    ----------
    source_weights : ndarray, shape [n_src, K_src]
        Source archetype weight matrix.
    source_coords : ndarray, shape [n_src, d]
        Source cell coordinates in the common (target) coordinate space.
    target_weights : ndarray, shape [n_tgt, K_tgt]
        Target archetype weight matrix.
    target_coords : ndarray, shape [n_tgt, d]
        Target cell coordinates.
    k : int, default 10
        k-NN parameter forwarded to ``compute_archetype_correspondence``.
    n_perms : int, default 200
        Number of permutations PER swap fraction. Total compute is
        ``n_perms * len(swap_fractions)`` correspondence calls plus one
        baseline call.
    swap_fractions : tuple of float, default (0.0, 0.05, 0.10, 0.20, 0.35, 0.50)
        Fractions of cells to swap. Must include 0.0 as the no-shuffle
        baseline (the implementation enforces this implicitly: f=0 returns
        the observed matrix repeated n_perms times in the null
        distribution).
    seed : int, default 42
        RNG seed (``np.random.default_rng(seed)``) for full reproducibility.

    Returns
    -------
    dict with keys:
        observed_mass : ndarray [K_src, K_tgt]
            Observed correspondence mass matrix at f=0 (no swap baseline).
        null_distributions : dict[float, ndarray]
            Maps each swap fraction to a stacked array of shape
            [n_perms, K_src, K_tgt]. Entries at f=0 are all identical
            copies of observed_mass (no shuffling occurs).
        empirical_p : ndarray [K_src, K_tgt]
            Per-pair empirical p-value computed at the LARGEST swap
            fraction. ``p[i, j] = (1 + #perms where null_mass[i, j] >=
            observed_mass[i, j]) / (n_perms + 1)``. Values lie in
            ``[1/(n_perms+1), 1.0]``.
        empirical_fdr : ndarray [K_src, K_tgt]
            BH-corrected p-values across the full K_src * K_tgt family.
        swap_fractions : tuple of float
            Echo of the input swap fractions.
        n_perms : int
            Echo of n_perms.
        null_mean_curve : ndarray [n_f, K_src, K_tgt]
            Per-fraction mean of the null distribution.
        null_std_curve : ndarray [n_f, K_src, K_tgt]
            Per-fraction std of the null distribution.

    Notes
    -----
    Wall-clock expectations (single core, observed on macOS arm64):
        - Synthetic test sizes (100 cells, K=3-4, 200 perms x 6 fractions):
          under 1 second.
        - Mid scale (2000 cells, K=9-11, 50 perms x 6 fractions): ~12 seconds;
          extrapolates to ~50 seconds at 200 perms.
        - Real HSC/CMP scale (~20k cells per population, K=9-11, 200 perms
          x 6 fractions): approximately 5-10 minutes per fit. The dominant
          cost is the cKDTree query inside
          ``compute_archetype_correspondence`` — each permutation rebuilds
          the tree on the (lightly) scrambled target coordinates.
          If this is too slow for an iterative workflow, reduce n_perms to
          50 (resolution drops from 1/201 to 1/51).

    The function deliberately reuses ``compute_archetype_correspondence``
    rather than inlining a faster path so that any future fix or change to
    the correspondence logic propagates to the null automatically.
    """
    from scipy.stats import false_discovery_control

    source_weights = np.asarray(source_weights, dtype=float)
    source_coords = np.asarray(source_coords, dtype=float)
    target_weights = np.asarray(target_weights, dtype=float)
    target_coords = np.asarray(target_coords, dtype=float)

    n_src = source_weights.shape[0]
    n_tgt = target_weights.shape[0]
    K_src = source_weights.shape[1]
    K_tgt = target_weights.shape[1]

    rng = np.random.default_rng(seed)

    # Baseline (f=0) — call once and reuse for f=0 entries
    baseline_result = compute_archetype_correspondence(
        source_weights=source_weights,
        source_coords=source_coords,
        target_weights=target_weights,
        target_coords=target_coords,
        k=k,
        method="hard",
    )
    observed_mass = baseline_result["mass"].copy()

    # Storage
    null_distributions: dict = {}
    null_mean_curve = np.zeros((len(swap_fractions), K_src, K_tgt))
    null_std_curve = np.zeros((len(swap_fractions), K_src, K_tgt))

    n_swap_max = min(n_src, n_tgt)

    for fi, f in enumerate(swap_fractions):
        f_val = float(f)
        m = int(round(f_val * n_swap_max))

        # f=0 (or m == 0): no shuffle — null distribution is the constant
        # observed matrix. We still populate n_perms slices for shape
        # consistency with downstream consumers.
        if m == 0:
            null_arr = np.broadcast_to(
                observed_mass, (n_perms, K_src, K_tgt)
            ).copy()
            null_distributions[f_val] = null_arr
            null_mean_curve[fi] = observed_mass
            null_std_curve[fi] = np.zeros((K_src, K_tgt))
            continue

        null_arr = np.empty((n_perms, K_src, K_tgt))
        for p_i in range(n_perms):
            # Coordinate-only swap (DESIGN NOTE above): pick m source-row
            # indices and m target-row indices, swap their coordinate rows.
            src_idx = rng.choice(n_src, size=m, replace=False)
            tgt_idx = rng.choice(n_tgt, size=m, replace=False)

            sc_perm = source_coords.copy()
            tc_perm = target_coords.copy()
            tmp = sc_perm[src_idx].copy()
            sc_perm[src_idx] = tc_perm[tgt_idx]
            tc_perm[tgt_idx] = tmp

            perm_result = compute_archetype_correspondence(
                source_weights=source_weights,
                source_coords=sc_perm,
                target_weights=target_weights,
                target_coords=tc_perm,
                k=k,
                method="hard",
            )
            null_arr[p_i] = perm_result["mass"]

        null_distributions[f_val] = null_arr
        null_mean_curve[fi] = null_arr.mean(axis=0)
        null_std_curve[fi] = null_arr.std(axis=0)

    # Empirical p-values at the LARGEST swap fraction (hardest test).
    largest_f = max(swap_fractions)
    null_at_largest = null_distributions[float(largest_f)]
    # p[i, j] = (1 + #perms with null >= observed) / (n_perms + 1)
    ge_count = (null_at_largest >= observed_mass[None, :, :]).sum(axis=0)
    empirical_p = (1.0 + ge_count) / (n_perms + 1.0)

    # BH FDR correction across all K_src * K_tgt pairs.
    flat_p = np.clip(empirical_p.ravel(), 0.0, 1.0)
    flat_fdr = false_discovery_control(flat_p, method="bh")
    empirical_fdr = flat_fdr.reshape(empirical_p.shape)

    return {
        "observed_mass": observed_mass,
        "null_distributions": null_distributions,
        "empirical_p": empirical_p,
        "empirical_fdr": empirical_fdr,
        "swap_fractions": tuple(float(f) for f in swap_fractions),
        "n_perms": int(n_perms),
        "null_mean_curve": null_mean_curve,
        "null_std_curve": null_std_curve,
    }

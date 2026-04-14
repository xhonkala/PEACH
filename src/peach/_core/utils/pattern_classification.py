"""Feature pattern classification from simplex regression coefficients."""

import numpy as np


def _classify_interaction_detail(vertex_betas, interaction_betas, interaction_pairs,
                                  interaction_pvalues_fdr, fdr_threshold):
    """Sub-classify significant interaction terms by vertex + edge relationship.

    For each significant interaction pair (j, k):
    - Cooperative: beta_j and beta_k both high relative to other vertices
    - Tradeoff: beta_j high, beta_k low (or vice versa)
    - Transition-enriched: peaks in blending zone (modest vertices, large gamma)
    - Gradient: other significant interaction patterns

    Returns list of per-pair classification dicts.
    """
    if interaction_betas is None or interaction_pairs is None:
        return []

    betas = np.asarray(vertex_betas)
    int_betas = np.asarray(interaction_betas)

    # R2-based classification: use beta^2 as a proxy for per-archetype
    # variance explained. More interpretable than raw beta for pattern
    # classification because it reflects position dependence strength
    # rather than theoretical expression magnitude.
    r2_proxy = betas ** 2
    median_r2 = float(np.median(r2_proxy))

    detail = []
    for pair_idx, (j, k) in enumerate(interaction_pairs):
        # Skip non-significant pairs
        if interaction_pvalues_fdr is not None:
            if pair_idx < len(interaction_pvalues_fdr) and interaction_pvalues_fdr[pair_idx] >= fdr_threshold:
                continue

        int_gamma_raw = float(int_betas[pair_idx]) if pair_idx < len(int_betas) else 0.0
        beta_j = float(betas[j])
        beta_k = float(betas[k])

        # R2-based gamma: fraction of pair variance from archetype j.
        # Range [0, 1]: >0.5 = j-dominant, <0.5 = k-dominant, ~0.5 = balanced.
        r2_j = beta_j ** 2
        r2_k = beta_k ** 2
        gamma = r2_j / (r2_j + r2_k + 1e-10)

        # Classify using R2 proxy thresholds
        j_high = r2_j > median_r2
        k_high = r2_k > median_r2
        same_sign = (np.sign(beta_j) == np.sign(beta_k)
                     and beta_j != 0 and beta_k != 0)

        if j_high and k_high and same_sign:
            pair_type = "cooperative"
        elif (j_high and not k_high) or (not j_high and k_high):
            pair_type = "tradeoff"
        elif not j_high and not k_high and abs(int_gamma_raw) > np.sqrt(median_r2):
            pair_type = "transition-enriched"
        else:
            pair_type = "gradient"

        transition = "rising" if int_gamma_raw > 0 else "falling"

        detail.append({
            "pair": (int(j), int(k)),
            "pair_type": pair_type,
            "transition": transition,
            "gamma": gamma,
            "beta_j": beta_j,
            "beta_k": beta_k,
        })

    return detail


def classify_single_feature(
    vertex_betas,
    r2,
    f_pvalue_fdr,
    interaction_f_pvalue_fdr=None,
    fdr_threshold=0.05,
    exclusive_ratio=2.0,
    vertex_ses=None,
    interaction_betas=None,
    interaction_pairs=None,
    interaction_pvalues_fdr=None,
):
    """Classify a single feature into a biological pattern type.

    Parameters
    ----------
    vertex_betas : array-like, shape [K]
        Regression coefficients per archetype.
    r2 : float
        R-squared of the regression.
    f_pvalue_fdr : float
        FDR-corrected F-test p-value.
    interaction_f_pvalue_fdr : float or None
        FDR-corrected interaction F-test p-value.
    fdr_threshold : float
        Significance threshold.
    exclusive_ratio : float
        Min ratio of max |beta| to second max for exclusive classification.
    vertex_ses : array-like or None
        Standard errors per archetype. When provided, the dominant
        coefficient must satisfy ``|beta| > 2 * SE`` to be classified
        as exclusive.
    interaction_betas : array-like or None, shape [n_pairs]
        Per-pair interaction (gamma) coefficients.
    interaction_pairs : list of (j, k) tuples or None
        Archetype index pairs corresponding to interaction_betas.
    interaction_pvalues_fdr : array-like or None, shape [n_pairs]
        FDR-corrected p-values per interaction pair.
    """
    # NaN guard
    if np.isnan(r2):
        return {"pattern": "flat", "r2": float(r2), "details": {"reason": "nan_r2"}}
    if np.isnan(f_pvalue_fdr):
        return {"pattern": "flat", "r2": float(r2), "details": {"reason": "nan_pvalue"}}

    # Rule 1: nonsignificant F-test -> flat
    if f_pvalue_fdr > fdr_threshold:
        return {"pattern": "flat", "r2": float(r2), "details": {"reason": "nonsignificant"}}

    # Rule 2: archetype-exclusive (requires K >= 2 for a ratio comparison)
    abs_betas = np.abs(vertex_betas)
    sorted_abs = np.sort(abs_betas)[::-1]
    if len(sorted_abs) >= 2 and sorted_abs[0] > 0 and sorted_abs[0] / max(sorted_abs[1], 1e-10) >= exclusive_ratio:
        dominant = int(np.argmax(abs_betas))
        # SE filter: dominant coefficient must be > 2*SE to be reliably exclusive
        se_passes = True
        if vertex_ses is not None:
            dominant_se = vertex_ses[dominant]
            if dominant_se > 0 and abs_betas[dominant] < 2 * dominant_se:
                se_passes = False
        if se_passes:
            return {
                "pattern": "archetype-exclusive",
                "r2": float(r2),
                "details": {"dominant_archetype": dominant},
            }

    # Rule 3: interaction
    if interaction_f_pvalue_fdr is not None and interaction_f_pvalue_fdr < fdr_threshold:
        interaction_detail = _classify_interaction_detail(
            vertex_betas, interaction_betas, interaction_pairs,
            interaction_pvalues_fdr, fdr_threshold
        )
        return {
            "pattern": "interaction",
            "r2": float(r2),
            "details": {
                "dominant_archetype": int(np.argmax(abs_betas)),
                "interaction_detail": interaction_detail,
            },
        }

    # Rule 4: structured fallback
    return {
        "pattern": "structured",
        "r2": float(r2),
        "details": {"dominant_archetype": int(np.argmax(abs_betas))},
    }


def classify_all_features(
    vertex_coefficients,
    r_squared,
    f_pvalue_fdr,
    interaction_f_pvalue_fdr=None,
    fdr_threshold=0.05,
    exclusive_ratio=2.0,
    vertex_ses=None,
    interaction_coefficients=None,
    interaction_pairs=None,
    interaction_pvalues_fdr=None,
):
    """Classify all features into biological pattern types.

    Parameters
    ----------
    vertex_coefficients : array-like, shape [n_features, K]
    r_squared : array-like, shape [n_features]
    f_pvalue_fdr : array-like, shape [n_features]
    interaction_f_pvalue_fdr : array-like or None, shape [n_features]
    fdr_threshold : float
    exclusive_ratio : float
    vertex_ses : array-like or None, shape [n_features, K]
        Standard errors per feature per archetype. Passed to
        ``classify_single_feature`` for SE-aware exclusive filtering.
    interaction_coefficients : array-like or None, shape [n_features, n_pairs]
        Per-pair interaction (gamma) coefficients for each feature.
    interaction_pairs : list of (j, k) tuples or None
        Archetype index pairs corresponding to columns of interaction_coefficients.
    interaction_pvalues_fdr : array-like or None, shape [n_features, n_pairs]
        FDR-corrected p-values per interaction pair per feature.
    """
    n_features = len(r_squared)
    results = []
    for i in range(n_features):
        int_fdr = (
            interaction_f_pvalue_fdr[i]
            if interaction_f_pvalue_fdr is not None
            else None
        )
        feat_ses = vertex_ses[i] if vertex_ses is not None else None
        feat_int_betas = (
            interaction_coefficients[i]
            if interaction_coefficients is not None
            else None
        )
        feat_int_pvals = (
            interaction_pvalues_fdr[i]
            if interaction_pvalues_fdr is not None
            else None
        )
        results.append(
            classify_single_feature(
                vertex_coefficients[i],
                r_squared[i],
                f_pvalue_fdr[i],
                interaction_f_pvalue_fdr=int_fdr,
                fdr_threshold=fdr_threshold,
                exclusive_ratio=exclusive_ratio,
                vertex_ses=feat_ses,
                interaction_betas=feat_int_betas,
                interaction_pairs=interaction_pairs,
                interaction_pvalues_fdr=feat_int_pvals,
            )
        )
    return results

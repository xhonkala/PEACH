"""Feature pattern classification from simplex regression coefficients."""

import numpy as np


def classify_single_feature(
    vertex_betas,
    r2,
    f_pvalue_fdr,
    interaction_f_pvalue_fdr=None,
    fdr_threshold=0.05,
    exclusive_ratio=2.0,
    vertex_ses=None,
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
        return {
            "pattern": "interaction",
            "r2": float(r2),
            "details": {"dominant_archetype": int(np.argmax(abs_betas))},
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
        results.append(
            classify_single_feature(
                vertex_coefficients[i],
                r_squared[i],
                f_pvalue_fdr[i],
                interaction_f_pvalue_fdr=int_fdr,
                fdr_threshold=fdr_threshold,
                exclusive_ratio=exclusive_ratio,
                vertex_ses=feat_ses,
            )
        )
    return results

"""Feature pattern classification from simplex regression coefficients."""

import numpy as np


def classify_single_feature(
    vertex_betas,
    r2,
    f_pvalue_fdr,
    interaction_f_pvalue_fdr=None,
    fdr_threshold=0.05,
    exclusive_ratio=2.0,
):
    # NaN guard
    if np.isnan(r2):
        return {"pattern": "flat", "r2": float(r2), "details": {"reason": "nan_r2"}}
    if np.isnan(f_pvalue_fdr):
        return {"pattern": "flat", "r2": float(r2), "details": {"reason": "nan_pvalue"}}

    # Rule 1: nonsignificant F-test -> flat
    if f_pvalue_fdr > fdr_threshold:
        return {"pattern": "flat", "r2": float(r2), "details": {"reason": "nonsignificant"}}

    # Rule 2: archetype-exclusive
    abs_betas = np.abs(vertex_betas)
    sorted_abs = np.sort(abs_betas)[::-1]
    if sorted_abs[0] > 0 and sorted_abs[0] / max(sorted_abs[1], 1e-10) >= exclusive_ratio:
        dominant = int(np.argmax(abs_betas))
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
):
    n_features = len(r_squared)
    results = []
    for i in range(n_features):
        int_fdr = (
            interaction_f_pvalue_fdr[i]
            if interaction_f_pvalue_fdr is not None
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
            )
        )
    return results

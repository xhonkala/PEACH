"""Feature pattern classification from simplex regression coefficients."""

import numpy as np


def classify_single_feature(
    vertex_betas,
    interaction_betas,
    r2,
    p_betas,
    p_interactions=None,
    r2_threshold=0.05,
    cv_threshold=0.15,
    exclusive_ratio=2.0,
):
    """Classify a single feature's regression pattern.

    Patterns are checked in priority order:
    1. flat: R² < r2_threshold OR CV(beta) < cv_threshold
    2. archetype-exclusive: max(beta) / second_max(beta) >= exclusive_ratio
    3. gradient: 2+ betas in top tier separated by large gap (>30% of range)
    4. monotonic: fallback for structured but diffuse patterns

    Parameters
    ----------
    vertex_betas : np.ndarray [K]
        Regression coefficients for each archetype vertex.
    interaction_betas : np.ndarray or None
        Kept for API compatibility; not used in classification.
    r2 : float
        R-squared value from the regression.
    p_betas : np.ndarray [K]
        P-values for each vertex coefficient. Kept for API compatibility;
        not used in classification.
    p_interactions : np.ndarray or None
        Kept for API compatibility; not used in classification.
    r2_threshold : float
        Below this R², the feature is classified as "flat".
    cv_threshold : float
        Below this coefficient of variation, the feature is classified as "flat".
    exclusive_ratio : float
        Minimum ratio of max(beta) to second_max(beta) for "archetype-exclusive".

    Returns
    -------
    dict with keys: pattern, confidence, details
        pattern : str
            One of "flat", "archetype-exclusive", "gradient", "monotonic".
        confidence : float
            Confidence score in [0, 1].
        details : dict
            Additional classification metadata.
    """
    # Guard against NaN R²
    if np.isnan(r2):
        return {
            "pattern": "flat",
            "confidence": 0.0,
            "details": {"reason": "nan_r2"},
        }

    # Rule 1a: Low R² -> flat
    if r2 < r2_threshold:
        return {
            "pattern": "flat",
            "confidence": 1.0 - r2 / r2_threshold,
            "details": {"reason": "low_r2"},
        }

    # Rule 1b: Low coefficient of variation -> flat
    cv = np.std(vertex_betas) / max(abs(np.mean(vertex_betas)), 1e-10)
    if cv < cv_threshold:
        return {
            "pattern": "flat",
            "confidence": 0.9,
            "details": {"reason": "low_cv", "cv": float(cv)},
        }

    # Sort betas descending for subsequent rules
    sorted_betas = np.sort(vertex_betas)[::-1]

    # Rule 2: Archetype-exclusive — compare absolute magnitudes to handle negative betas
    sorted_abs = np.sort(np.abs(vertex_betas))[::-1]
    if sorted_abs[0] > 0 and sorted_abs[0] / max(sorted_abs[1], 1e-10) >= exclusive_ratio:
        return {
            "pattern": "archetype-exclusive",
            "confidence": 0.9,
            "details": {"dominant_archetype": int(np.argmax(vertex_betas))},
        }

    # Rule 3: Gradient — 2+ betas in top tier separated from rest by largest gap
    if len(sorted_betas) >= 3:
        gaps = np.diff(sorted_betas)  # negative since descending
        largest_gap_idx = np.argmin(gaps)  # most negative = largest drop
        gap_size = abs(gaps[largest_gap_idx])
        beta_range = sorted_betas[0] - sorted_betas[-1]
        n_high = largest_gap_idx + 1

        if n_high >= 2 and beta_range > 0 and gap_size > beta_range * 0.3:
            return {
                "pattern": "gradient",
                "confidence": 0.8,
                "details": {
                    "n_high": int(n_high),
                    "dominant_archetype": int(np.argmax(vertex_betas)),
                },
            }

    # Rule 4: Monotonic — fallback for everything else
    return {
        "pattern": "monotonic",
        "confidence": 0.6,
        "details": {"dominant_archetype": int(np.argmax(vertex_betas))},
    }


def classify_all_features(
    vertex_coefficients,
    interaction_coefficients,
    r_squared,
    vertex_pvalues,
    interaction_pvalues,
    r2_threshold=0.05,
    cv_threshold=0.15,
    exclusive_ratio=2.0,
):
    """Classify all features at once.

    Parameters
    ----------
    vertex_coefficients : np.ndarray [n_features, K]
        Regression coefficients matrix, one row per feature.
    interaction_coefficients : np.ndarray or None
        Kept for API compatibility; not used in classification.
    r_squared : np.ndarray [n_features]
        R-squared values, one per feature.
    vertex_pvalues : np.ndarray [n_features, K]
        P-values for vertex coefficients. Kept for API compatibility.
    interaction_pvalues : np.ndarray or None
        Kept for API compatibility; not used in classification.
    r2_threshold : float
        Below this R², the feature is classified as "flat".
    cv_threshold : float
        Below this coefficient of variation, the feature is classified as "flat".
    exclusive_ratio : float
        Minimum ratio of max(beta) to second_max(beta) for "archetype-exclusive".

    Returns
    -------
    list[dict]
        One classification dict per feature, each with keys:
        pattern, confidence, details.
    """
    n_features = len(r_squared)
    results = []
    for i in range(n_features):
        int_betas = (
            interaction_coefficients[i]
            if interaction_coefficients is not None
            else None
        )
        int_pvals = (
            interaction_pvalues[i] if interaction_pvalues is not None else None
        )
        results.append(
            classify_single_feature(
                vertex_coefficients[i],
                int_betas,
                r_squared[i],
                vertex_pvalues[i],
                int_pvals,
                r2_threshold=r2_threshold,
                cv_threshold=cv_threshold,
                exclusive_ratio=exclusive_ratio,
            )
        )
    return results

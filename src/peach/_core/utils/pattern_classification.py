"""Feature pattern classification from simplex regression coefficients."""

import numpy as np


def classify_single_feature(
    vertex_betas,
    interaction_betas,
    r2,
    p_betas,
    p_interactions,
    r2_threshold=0.05,
    significance_threshold=0.05,
    effect_size_threshold=None,
):
    """Classify a single feature's regression pattern.

    Parameters
    ----------
    vertex_betas : np.ndarray [K]
        Regression coefficients for each archetype vertex.
    interaction_betas : np.ndarray [K-choose-2] or None
        Interaction term coefficients, or None if no interaction model.
    r2 : float
        R-squared value from the regression.
    p_betas : np.ndarray [K]
        P-values for each vertex coefficient.
    p_interactions : np.ndarray [K-choose-2] or None
        P-values for interaction terms, or None if no interaction model.
    r2_threshold : float
        Below this R^2, the feature is classified as "flat".
    significance_threshold : float
        P-value cutoff for statistical significance.
    effect_size_threshold : float or None
        Minimum beta range to consider meaningful. Auto-calibrated from
        data if None.

    Returns
    -------
    dict with keys: pattern, confidence, details
        pattern : str
            One of "flat", "archetype-exclusive", "monotonic-gradient",
            "multi-archetype-shared", "antagonistic", "ridge", "valley".
        confidence : float
            Confidence score in [0, 1].
        details : dict
            Additional classification metadata.
    """
    K = len(vertex_betas)

    # Auto-calibrate effect size threshold
    if effect_size_threshold is None:
        beta_range = np.ptp(vertex_betas)
        effect_size_threshold = max(beta_range * 0.2, 0.1)

    # Rule 1: Low R^2 -> flat
    if r2 < r2_threshold:
        return {
            "pattern": "flat",
            "confidence": 1.0 - r2 / r2_threshold,
            "details": {"reason": "low_r2"},
        }

    # Rule 2: Low effect size -> flat
    beta_range = np.ptp(vertex_betas)
    if beta_range < effect_size_threshold and not _has_significant_interactions(
        interaction_betas, p_interactions, significance_threshold
    ):
        return {
            "pattern": "flat",
            "confidence": 0.8,
            "details": {"reason": "low_effect_size"},
        }

    # Rule 3: Significant interactions -> ridge or valley
    if _has_significant_interactions(
        interaction_betas, p_interactions, significance_threshold
    ):
        sig_mask = p_interactions < significance_threshold
        sig_interactions = interaction_betas[sig_mask]
        if np.mean(sig_interactions) > 0:
            return {
                "pattern": "ridge",
                "confidence": 0.8,
                "details": {"n_sig_interactions": int(sig_mask.sum())},
            }
        else:
            return {
                "pattern": "valley",
                "confidence": 0.8,
                "details": {"n_sig_interactions": int(sig_mask.sum())},
            }

    # Rule 4: Count significantly elevated betas
    sig_betas = p_betas < significance_threshold
    beta_mean = np.mean(vertex_betas)
    elevated = sig_betas & (vertex_betas > beta_mean + effect_size_threshold * 0.5)
    depressed = sig_betas & (vertex_betas < beta_mean - effect_size_threshold * 0.5)
    n_elevated = np.sum(elevated)
    n_depressed = np.sum(depressed)

    # Antagonistic: significant betas on both sides
    if n_elevated >= 1 and n_depressed >= 1 and (n_elevated + n_depressed) >= 3:
        return {
            "pattern": "antagonistic",
            "confidence": 0.7,
            "details": {
                "n_elevated": int(n_elevated),
                "n_depressed": int(n_depressed),
            },
        }

    # Exclusive: exactly 1 elevated
    if n_elevated == 1:
        sorted_betas = np.sort(vertex_betas)[::-1]
        ratio = sorted_betas[1] / max(sorted_betas[0], 1e-10)
        if ratio < 0.3:
            return {
                "pattern": "archetype-exclusive",
                "confidence": 0.9,
                "details": {"dominant_archetype": int(np.argmax(vertex_betas))},
            }
        else:
            return {
                "pattern": "monotonic-gradient",
                "confidence": 0.7,
                "details": {"dominant_archetype": int(np.argmax(vertex_betas))},
            }

    # Shared: 2+ elevated
    if n_elevated >= 2:
        return {
            "pattern": "multi-archetype-shared",
            "confidence": 0.7,
            "details": {"n_shared": int(n_elevated)},
        }

    # Gradient: ordered coefficients, 1 dominant
    sorted_betas = np.sort(vertex_betas)[::-1]
    if sorted_betas[0] > sorted_betas[1] * 1.5:
        return {
            "pattern": "monotonic-gradient",
            "confidence": 0.6,
            "details": {"dominant_archetype": int(np.argmax(vertex_betas))},
        }

    # Default
    if beta_range > effect_size_threshold * 3:
        return {"pattern": "antagonistic", "confidence": 0.5, "details": {}}
    return {"pattern": "monotonic-gradient", "confidence": 0.5, "details": {}}


def classify_all_features(
    vertex_coefficients,
    interaction_coefficients,
    r_squared,
    vertex_pvalues,
    interaction_pvalues,
    r2_threshold=0.05,
    significance_threshold=0.05,
    effect_size_threshold=None,
):
    """Classify all features at once.

    Parameters
    ----------
    vertex_coefficients : np.ndarray [n_features, K]
        Regression coefficients matrix, one row per feature.
    interaction_coefficients : np.ndarray [n_features, n_interactions] or None
        Interaction term coefficients, or None if no interaction model.
    r_squared : np.ndarray [n_features]
        R-squared values, one per feature.
    vertex_pvalues : np.ndarray [n_features, K]
        P-values for vertex coefficients.
    interaction_pvalues : np.ndarray [n_features, n_interactions] or None
        P-values for interaction terms, or None.
    r2_threshold : float
        Below this R^2, the feature is classified as "flat".
    significance_threshold : float
        P-value cutoff for statistical significance.
    effect_size_threshold : float or None
        Minimum beta range to consider meaningful. Auto-calibrated per
        feature if None.

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
                significance_threshold=significance_threshold,
                effect_size_threshold=effect_size_threshold,
            )
        )
    return results


def _has_significant_interactions(interaction_betas, p_interactions, threshold):
    """Check if any interaction terms are significant."""
    if interaction_betas is None or p_interactions is None:
        return False
    return np.any(p_interactions < threshold)

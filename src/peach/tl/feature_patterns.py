"""Feature pattern classification and archetype summary public API."""

import numpy as np
from anndata import AnnData

from peach._core.utils.feature_utils import resolve_regression_result, store_result
from peach._core.utils.pattern_classification import classify_all_features
from peach._core.types import PatternClassificationResult, SimplexRegressionResult


def classify_feature_patterns(
    adata: AnnData,
    *,
    regression_result: SimplexRegressionResult | None = None,
    r2_threshold: float = 0.05,
    cv_threshold: float = 0.15,
    exclusive_ratio: float = 2.0,
) -> dict:
    """Classify features into biological pattern types from regression coefficients.

    Parameters
    ----------
    adata : AnnData
        Must have simplex regression results in uns['peach_simplex_regression']
        or provide regression_result directly.
    regression_result : SimplexRegressionResult or None
        If None, reads from adata.uns['peach_simplex_regression'].
    r2_threshold : float
        Minimum R^2 to be classified as non-flat (default 0.05 = 5%).
    cv_threshold : float
        Below this coefficient of variation, the feature is classified as "flat".
    exclusive_ratio : float
        Minimum ratio of max(beta) to second_max(beta) for "archetype-exclusive".

    Returns
    -------
    dict
        Keys: feature_names, n_features, classifications, pattern_counts.
        Also stored in adata.uns['peach_feature_patterns'].
    """
    if regression_result is None:
        stored = resolve_regression_result(adata, prefer="genes")
        if stored is None:
            raise ValueError(
                "No regression results found. Run pc.tl.feature_simplex_regression() first "
                "or provide regression_result directly."
            )
        regression_result = SimplexRegressionResult(**stored)
    elif isinstance(regression_result, dict):
        # Accept serialized dict (returned by feature_simplex_regression)
        regression_result = SimplexRegressionResult(**regression_result)

    classifications = classify_all_features(
        vertex_coefficients=regression_result.vertex_coefficients,
        interaction_coefficients=regression_result.interaction_coefficients,
        r_squared=regression_result.r_squared_degree1,
        vertex_pvalues=regression_result.vertex_pvalues,
        interaction_pvalues=regression_result.interaction_pvalues,
        r2_threshold=r2_threshold,
        cv_threshold=cv_threshold,
        exclusive_ratio=exclusive_ratio,
    )

    # Count patterns
    pattern_counts = {}
    for c in classifications:
        p = c["pattern"]
        pattern_counts[p] = pattern_counts.get(p, 0) + 1

    result = PatternClassificationResult(
        feature_names=regression_result.feature_names,
        n_features=regression_result.n_features,
        classifications=classifications,
        pattern_counts=pattern_counts,
    )

    serialized = result.to_serializable()
    store_result(adata, "feature_patterns", serialized)
    return serialized


def archetype_summary(
    adata: AnnData,
    *,
    archetype_idx: int | None = None,
    top_n: int = 20,
    include_drivers: bool = True,
    include_gmm: bool = True,
) -> dict | list[dict]:
    """Generate structured summary for one or all archetypes.

    Aggregates results from simplex regression, pattern classification,
    driver regression, and GMM decomposition.

    Parameters
    ----------
    adata : AnnData
        Must have simplex regression results in uns['peach_simplex_regression'].
    archetype_idx : int or None
        Specific archetype index, or None for all.
    top_n : int
        Top enriched/depleted features to report.
    include_drivers : bool
        Include flipped regression results if available.
    include_gmm : bool
        Include GMM components near this archetype.

    Returns
    -------
    dict (single archetype) or list[dict] (all archetypes)
        Per-archetype structured summary.
    """
    # Check for regression results
    reg = resolve_regression_result(adata, prefer="genes")
    if reg is None:
        raise ValueError(
            "No simplex regression results found. "
            "Run pc.tl.feature_simplex_regression() first."
        )
    vertex_coefs = reg["vertex_coefficients"]  # [n_features, K]
    feature_names = reg["feature_names"]
    K = vertex_coefs.shape[1]

    # Pattern classifications
    patterns = None
    if "peach_feature_patterns" in adata.uns:
        patterns = adata.uns["peach_feature_patterns"]

    # Driver regression
    drivers = None
    if include_drivers and "peach_driver_regression" in adata.uns:
        drivers = adata.uns["peach_driver_regression"]

    # GMM
    gmm = None
    if include_gmm and "peach_gmm" in adata.uns:
        gmm = adata.uns["peach_gmm"]

    def _summary_for_archetype(k):
        """Build summary dict for archetype k."""
        betas_k = vertex_coefs[:, k]  # [n_features]

        # Top enriched/depleted
        sorted_idx = np.argsort(betas_k)
        top_enriched_idx = sorted_idx[-top_n:][::-1]
        top_depleted_idx = sorted_idx[:top_n]

        top_enriched = [
            {"feature": feature_names[i], "coefficient": float(betas_k[i])}
            for i in top_enriched_idx
        ]
        top_depleted = [
            {"feature": feature_names[i], "coefficient": float(betas_k[i])}
            for i in top_depleted_idx
        ]

        summary = {
            "archetype_idx": k,
            "top_enriched": top_enriched,
            "top_depleted": top_depleted,
        }

        # Interaction terms involving this archetype
        if (
            "interaction_coefficients" in reg
            and reg["interaction_coefficients"] is not None
        ):
            int_coefs = reg["interaction_coefficients"]
            int_pairs = reg.get("interaction_pairs", [])
            relevant = {}
            for idx, pair in enumerate(int_pairs):
                if k in pair:
                    other = pair[1] if pair[0] == k else pair[0]
                    # Mean interaction across features
                    mean_int = float(np.mean(int_coefs[:, idx]))
                    relevant[f"archetype_{other}"] = mean_int
            summary["interactions"] = relevant

        # Pattern counts
        if patterns is not None:
            classifications = patterns.get("classifications", [])
            pattern_counts = {}
            for i, c in enumerate(classifications):
                details = c.get("details", {})
                if details.get("dominant_archetype") == k or (
                    c["pattern"] == "archetype-exclusive"
                    and details.get("dominant_archetype") == k
                ):
                    p = c["pattern"]
                    pattern_counts[p] = pattern_counts.get(p, 0) + 1
            summary["pattern_counts"] = pattern_counts

        # Driver genesets
        if drivers is not None:
            driver_coefs = drivers.get("main_coefficients")
            if driver_coefs is not None:
                driver_names = drivers.get("feature_names", [])
                coefs_k = driver_coefs[k] if k < len(driver_coefs) else np.array([])
                if len(coefs_k) > 0:
                    sorted_d = np.argsort(np.abs(coefs_k))[::-1]
                    top_drivers = [
                        {
                            "feature": driver_names[i],
                            "coefficient": float(coefs_k[i]),
                        }
                        for i in sorted_d[:top_n]
                    ]
                    summary["driver_genesets"] = top_drivers

        # GMM components near this archetype
        if gmm is not None:
            archetype_map = gmm.get("component_archetype_map")
            if archetype_map is not None:
                nearby = np.where(np.asarray(archetype_map) == k)[0]
                summary["gmm_components"] = nearby.tolist()

        return summary

    if archetype_idx is not None:
        if archetype_idx < 0 or archetype_idx >= K:
            raise ValueError(
                f"archetype_idx {archetype_idx} out of range [0, {K})"
            )
        return _summary_for_archetype(archetype_idx)

    return [_summary_for_archetype(k) for k in range(K)]

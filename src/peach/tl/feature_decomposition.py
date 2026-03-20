"""Simplex density decomposition via GMM public API."""

import numpy as np
from anndata import AnnData

from peach._core.utils.feature_utils import (
    get_archetype_weights,
    resolve_features,
    store_result,
)
from peach._core.utils.simplex_gmm import fit_simplex_gmm, characterize_components
from peach._core.types import MixtureResult


def feature_simplex_decomposition(
    adata: AnnData,
    *,
    feature_matrix=None,
    feature_names=None,
    n_components_range=None,
    model_selection: str = "bic",
    model_type: str = "dirichlet",
    covariance_type: str = "full",
    n_initializations: int = 20,
    stability_threshold: float = 0.7,
    reassignment_confidence: float = 0.0,
    characterize_features: bool = True,
    ilr_epsilon: float = 1e-3,
    random_state: int = 42,
    copy: bool = False,
) -> dict:
    """Decompose cell populations by mixture model in weight space.

    Fits a Gaussian Mixture Model (in ILR space) or Dirichlet Mixture Model
    (directly on the simplex) to archetype weights, selects the number of
    components by BIC or ICL, and filters by multi-initialization pairwise
    stability analysis. Identifies sub-populations that occupy distinct regions
    of the archetype weight simplex.

    Parameters
    ----------
    adata : AnnData
        Must have archetype weights in obsm['cell_archetype_weights'].
    feature_matrix : None, str, or array-like
        Feature matrix for component characterization.
        None -> adata.X, str -> adata.obsm[feature_matrix],
        array-like -> used directly.
    feature_names : list[str] or None
        Feature names. Inferred from adata.var_names if None and using adata.X.
    n_components_range : tuple[int, int] or None
        (min_components, max_components). Default: (K, 3*K).
    model_selection : str
        'bic' or 'icl'. ICL = BIC + 2*entropy(posterior), which penalizes
        overlapping clusters more heavily.
    model_type : str
        'dirichlet' (default): Dirichlet mixture directly on the simplex.
        'gaussian': GMM in ILR-transformed space.
    covariance_type : str
        sklearn GMM covariance type. One of 'full', 'tied', 'diag', 'spherical'.
        Only used when model_type='gaussian'.
    n_initializations : int
        Number of random initializations for stability analysis.
    stability_threshold : float
        Minimum stability score to retain a component (fraction of runs where
        component is recovered).
    reassignment_confidence : float
        Minimum posterior probability required to reassign an unstable cell
        to a stable component. Default 0.0 means all unstable cells are
        reassigned (backward compatible).
    characterize_features : bool
        If True, compute per-component mean feature profiles.
    ilr_epsilon : float
        Smoothing epsilon for ILR transform. Only used when model_type='gaussian'.
    random_state : int
        Random seed for reproducibility.
    copy : bool
        If True, operate on a copy of adata.

    Returns
    -------
    dict
        Plain dict with mixture model results. Stored in adata.uns['peach_gmm'],
        labels in adata.obsm['peach_gmm_labels'].
    """
    if copy:
        adata = adata.copy()

    weights = get_archetype_weights(adata)

    gmm_result = fit_simplex_gmm(
        weights,
        n_components_range=n_components_range,
        covariance_type=covariance_type,
        n_initializations=n_initializations,
        stability_threshold=stability_threshold,
        random_state=random_state,
        ilr_epsilon=ilr_epsilon,
        reassignment_confidence=reassignment_confidence,
        model_selection=model_selection,
        model_type=model_type,
    )

    # Component characterization
    feature_profiles = None
    if characterize_features:
        Y, _ = resolve_features(adata, feature_matrix, feature_names)
        feature_profiles = characterize_components(
            gmm_result["component_assignments"],
            Y,
            gmm_result["n_components_stable"],
        )

    result_obj = MixtureResult(
        n_components_optimal=gmm_result["n_components_optimal"],
        n_components_stable=gmm_result["n_components_stable"],
        component_assignments=gmm_result["component_assignments"],
        component_simplex_means=gmm_result["component_simplex_means"],
        component_archetype_map=gmm_result["component_archetype_map"],
        component_stability_scores=gmm_result["component_stability_scores"],
        component_feature_profiles=feature_profiles,
        component_weight_means=gmm_result.get("component_weight_means"),
        component_probabilities=gmm_result.get("component_probabilities"),
        bic_values=gmm_result["bic_values"],
        n_components_tested=gmm_result["n_components_tested"],
    )

    # Serialize to plain dict (PEACH convention: public API returns dicts)
    serialized = result_obj.to_serializable()

    # Add fields not in MixtureResult Pydantic model
    serialized["icl_values"] = gmm_result.get("icl_values")
    serialized["model_type"] = gmm_result.get("model_type")

    # Store in adata
    store_result(adata, "gmm", serialized)
    store_result(adata, "gmm_labels", gmm_result["component_assignments"], domain="obsm")

    return serialized


def component_regression(
    adata: AnnData,
    *,
    feature_type: str = "genes",
    n_bootstrap: int = 100,
    robust_se: bool = True,
) -> dict:
    """Run simplex regression separately per GMM component.

    Fits an independent simplex regression within each stable GMM component,
    enabling detection of component-specific feature drivers that may be
    masked in the global regression.

    Parameters
    ----------
    adata : AnnData
        Must have archetype weights in obsm['cell_archetype_weights'] and
        GMM results in uns['peach_gmm'] (from feature_simplex_decomposition).
    feature_type : str
        'genes' uses adata.X, or name of an obsm key for other feature types.
    n_bootstrap : int
        Number of bootstrap replicates for confidence intervals.
    robust_se : bool
        If True, use HC3 heteroscedasticity-consistent standard errors.

    Returns
    -------
    dict with keys:
        component_regs : dict[int, dict]
            Per-component simplex regression results. Keys are component
            indices (0..n_stable-1), values are regression result dicts.
        n_components : int
            Number of stable GMM components.
    """
    from peach.tl.feature_regression import feature_simplex_regression

    gmm = adata.uns.get("peach_gmm")
    if gmm is None:
        raise ValueError("Run pc.tl.feature_simplex_decomposition() first.")

    assignments = np.asarray(gmm["component_assignments"])
    n_stable = gmm["n_components_stable"]

    component_regs = {}
    for c in range(n_stable):
        mask = assignments == c
        if mask.sum() < 20:
            continue
        adata_sub = adata[mask].copy()
        reg = feature_simplex_regression(
            adata_sub, n_bootstrap=n_bootstrap, robust_se=robust_se,
            store_to_adata=False, store_residuals=False,
        )
        component_regs[c] = reg

    return {"component_regs": component_regs, "n_components": n_stable}

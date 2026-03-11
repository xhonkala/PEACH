"""Simplex density decomposition via GMM public API."""

import numpy as np
from anndata import AnnData

from peach._core.utils.feature_utils import (
    get_archetype_weights,
    resolve_features,
    store_result,
)
from peach._core.utils.simplex_gmm import fit_simplex_gmm, characterize_components
from peach._core.types import GMMResult


def feature_simplex_decomposition(
    adata: AnnData,
    *,
    feature_matrix=None,
    feature_names=None,
    n_components_range=None,
    model_selection: str = "bic",
    covariance_type: str = "full",
    n_initializations: int = 20,
    stability_threshold: float = 0.7,
    characterize_features: bool = True,
    ilr_epsilon: float = 1e-3,
    random_state: int = 42,
    copy: bool = False,
) -> dict:
    """Decompose cell populations by GMM in ILR-transformed weight space.

    Fits a Gaussian Mixture Model to archetype weights after ILR transform,
    selects the number of components by BIC, and filters by multi-initialization
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
        'bic' (only supported option currently).
    covariance_type : str
        sklearn GMM covariance type. One of 'full', 'tied', 'diag', 'spherical'.
    n_initializations : int
        Number of random initializations for stability analysis.
    stability_threshold : float
        Minimum stability score to retain a component (fraction of runs where
        component is recovered).
    characterize_features : bool
        If True, compute per-component mean feature profiles.
    random_state : int
        Random seed for reproducibility.
    copy : bool
        If True, operate on a copy of adata.

    Returns
    -------
    dict
        Plain dict with GMM results. Stored in adata.uns['peach_gmm'],
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

    result_obj = GMMResult(
        n_components_optimal=gmm_result["n_components_optimal"],
        n_components_stable=gmm_result["n_components_stable"],
        component_assignments=gmm_result["component_assignments"],
        component_simplex_means=gmm_result["component_simplex_means"],
        component_archetype_map=gmm_result["component_archetype_map"],
        component_stability_scores=gmm_result["component_stability_scores"],
        component_feature_profiles=feature_profiles,
        component_weight_means=gmm_result.get("component_weight_means"),
        bic_values=gmm_result["bic_values"],
        n_components_tested=gmm_result["n_components_tested"],
    )

    # Serialize to plain dict (PEACH convention: public API returns dicts)
    serialized = result_obj.to_serializable()

    # Store in adata
    store_result(adata, "gmm", serialized)
    store_result(adata, "gmm_labels", gmm_result["component_assignments"], domain="obsm")

    return serialized

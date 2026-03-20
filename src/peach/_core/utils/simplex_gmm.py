"""Simplex density decomposition via GMM/Dirichlet in weight space.

Fits a Gaussian Mixture Model (in ILR space) or Dirichlet Mixture Model
(directly on the simplex) to archetype weights, selects the number of
components by BIC or ICL, and filters by multi-initialization pairwise
stability analysis using the Hungarian algorithm for component correspondence.

This module decomposes the cell population into sub-populations that occupy
distinct regions of the archetype weight simplex, enabling identification of
stable cell states and transitional populations.

References:
- McLachlan & Peel (2000), "Finite Mixture Models", Wiley.
- Biernacki et al. (2000), "Assessing a mixture model for clustering with the
  integrated completed likelihood", IEEE TPAMI 22(7).
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment

from peach._core.utils.ilr_transform import ilr_transform, inverse_ilr


def _compute_icl(gmm, X):
    """Integrated Completed Likelihood criterion.

    ICL = BIC + 2 * entropy(posterior probabilities).
    Penalizes overlapping clusters more than BIC alone.

    Parameters
    ----------
    gmm : GaussianMixture or object with bic() and predict_proba()
        Fitted mixture model.
    X : np.ndarray [n, d]
        Data the model was fitted on.

    Returns
    -------
    float
        ICL value (lower is better).
    """
    bic = gmm.bic(X)
    proba = gmm.predict_proba(X)
    entropy = -np.sum(proba * np.log(np.clip(proba, 1e-300, 1.0)))
    return bic + 2 * entropy


def fit_simplex_gmm(
    weights,
    n_components_range=None,
    covariance_type="full",
    n_initializations=20,
    stability_threshold=0.7,
    random_state=42,
    ilr_epsilon=1e-3,
    reassignment_confidence=0.0,
    model_selection="bic",
    model_type="dirichlet",
):
    """Fit mixture model to archetype weights with model selection and stability analysis.

    Parameters
    ----------
    weights : np.ndarray [n_cells, K]
        Archetype weights (rows sum to 1).
    n_components_range : tuple[int, int] or None
        (min_components, max_components). Default: (K, 3*K).
    covariance_type : str
        GMM covariance type. One of 'full', 'tied', 'diag', 'spherical'.
        Only used when model_type='gaussian'.
    n_initializations : int
        Number of random initializations for stability analysis.
    stability_threshold : float
        Minimum stability to report a component (fraction of runs where
        component is recovered).
    random_state : int
        Random seed for reproducibility.
    ilr_epsilon : float
        Smoothing epsilon for ILR transform. Only used when model_type='gaussian'.
    reassignment_confidence : float
        Minimum posterior probability (from predict_proba) required to
        reassign an unstable cell to a stable component. Default 0.0 means
        all unstable cells are reassigned (backward compatible). Higher values
        (e.g. 0.8) leave low-confidence cells as -1 (unassigned).
    model_selection : str
        Criterion for selecting optimal n_components. 'bic' or 'icl'.
        ICL = BIC + 2*entropy(posterior), which penalizes overlapping clusters.
    model_type : str
        'dirichlet' (default): Dirichlet mixture directly on the simplex.
        'gaussian': GMM in ILR-transformed space.

    Returns
    -------
    dict with keys:
        n_components_optimal : int
            Selected number of components.
        n_components_stable : int
            Number of components after stability filtering.
        component_assignments : np.ndarray [n_cells]
            Cluster labels for stable components. Cells assigned to unstable
            components are reassigned to the nearest stable component.
        component_simplex_means : np.ndarray [n_stable, K]
            Centroids mapped back to the weight simplex.
        component_archetype_map : np.ndarray [n_stable]
            Index of the nearest archetype for each component centroid
            (argmax of simplex centroid).
        component_stability_scores : np.ndarray [n_stable]
            Stability score for each retained component.
        bic_values : np.ndarray [n_tested]
            BIC values for each n_components tested.
        icl_values : np.ndarray [n_tested]
            ICL values for each n_components tested.
        n_components_tested : np.ndarray [n_tested]
            Array of n_components values tested.
        gmm_model : fitted model
            Fitted model for the optimal n_components.
        component_probabilities : np.ndarray or None [n_cells, n_stable]
            Posterior probabilities for stable components. None if no unstable
            cells exist.
        model_type : str
            'gaussian' or 'dirichlet'.
    """
    if model_selection not in ("bic", "icl"):
        raise ValueError(f"model_selection must be 'bic' or 'icl', got '{model_selection}'")
    if model_type not in ("gaussian", "dirichlet"):
        raise ValueError(f"model_type must be 'gaussian' or 'dirichlet', got '{model_type}'")

    weights = np.asarray(weights, dtype=np.float64)
    K = weights.shape[1]
    n_cells = weights.shape[0]

    # Determine range
    if n_components_range is None:
        n_components_range = (K, 3 * K)
    n_min, n_max = n_components_range

    if model_type == "dirichlet":
        return _fit_dirichlet(
            weights, K, n_cells, n_min, n_max,
            n_initializations, stability_threshold,
            random_state, reassignment_confidence, model_selection,
        )
    else:
        return _fit_gaussian(
            weights, K, n_cells, n_min, n_max,
            covariance_type, n_initializations, stability_threshold,
            random_state, ilr_epsilon, reassignment_confidence, model_selection,
        )


def _fit_gaussian(
    weights, K, n_cells, n_min, n_max,
    covariance_type, n_initializations, stability_threshold,
    random_state, ilr_epsilon, reassignment_confidence, model_selection,
):
    """Fit Gaussian mixture in ILR space."""
    # Transform to ILR space
    ilr_coords = ilr_transform(weights, epsilon=ilr_epsilon)  # [n_cells, K-1]

    # BIC/ICL scan
    n_range = np.arange(n_min, n_max + 1)
    bic_values = np.full(len(n_range), np.inf)
    icl_values = np.full(len(n_range), np.inf)
    best_model = None
    best_score = np.inf
    best_n = n_min

    for idx, n_comp in enumerate(n_range):
        gmm = GaussianMixture(
            n_components=n_comp,
            covariance_type=covariance_type,
            n_init=3,
            random_state=random_state,
        )
        gmm.fit(ilr_coords)
        bic = gmm.bic(ilr_coords)
        icl = _compute_icl(gmm, ilr_coords)
        bic_values[idx] = bic
        icl_values[idx] = icl

        score = icl if model_selection == "icl" else bic
        if score < best_score:
            best_score = score
            best_model = gmm
            best_n = n_comp

    # Stability analysis for optimal n_components
    stability_scores = _compute_stability(
        ilr_coords, best_n, covariance_type, n_initializations, random_state
    )

    # Filter stable components
    stable_mask = stability_scores >= stability_threshold
    n_stable = int(np.sum(stable_mask))

    if n_stable == 0:
        stable_mask = np.ones(best_n, dtype=bool)
        n_stable = best_n

    # Get assignments and centroids
    all_labels = best_model.predict(ilr_coords)
    ilr_centroids = best_model.means_  # [n_components, K-1]
    simplex_centroids = inverse_ilr(ilr_centroids)  # [n_components, K]

    # Map stable components
    stable_indices = np.where(stable_mask)[0]
    stable_centroids = simplex_centroids[stable_indices]
    stable_stability = stability_scores[stable_indices]

    # Remap labels
    label_map = {old: new for new, old in enumerate(stable_indices)}
    component_assignments = np.full(n_cells, -1, dtype=int)
    for old_label, new_label in label_map.items():
        component_assignments[all_labels == old_label] = new_label

    # Handle unstable cells using predict_proba
    unstable_mask = component_assignments == -1
    component_probabilities = None
    if np.any(unstable_mask) and n_stable > 0:
        all_proba = best_model.predict_proba(ilr_coords)
        stable_proba = all_proba[:, stable_indices]
        component_probabilities = stable_proba

        unstable_proba = stable_proba[unstable_mask]
        max_prob = unstable_proba.max(axis=1)
        confident_mask = max_prob >= reassignment_confidence
        confident_idx = np.where(unstable_mask)[0][confident_mask]
        component_assignments[confident_idx] = np.argmax(
            stable_proba[confident_idx], axis=1
        )

    # Nearest archetype per component
    archetype_map = np.argmax(stable_centroids, axis=1)

    # Arithmetic mean of archetype weights per component
    component_weight_means = np.zeros((n_stable, K))
    for c in range(n_stable):
        mask = component_assignments == c
        if np.any(mask):
            component_weight_means[c] = weights[mask].mean(axis=0)

    return {
        "n_components_optimal": best_n,
        "n_components_stable": n_stable,
        "component_assignments": component_assignments,
        "component_simplex_means": stable_centroids,
        "component_archetype_map": archetype_map,
        "component_stability_scores": stable_stability,
        "component_weight_means": component_weight_means,
        "bic_values": bic_values,
        "icl_values": icl_values,
        "n_components_tested": n_range,
        "gmm_model": best_model,
        "component_probabilities": component_probabilities,
        "model_type": "gaussian",
    }


def _fit_dirichlet(
    weights, K, n_cells, n_min, n_max,
    n_initializations, stability_threshold,
    random_state, reassignment_confidence, model_selection,
):
    """Fit Dirichlet mixture directly on the simplex."""
    from peach._core.utils.dirichlet_mixture import DirichletMixture

    n_range = np.arange(n_min, n_max + 1)
    bic_values = np.full(len(n_range), np.inf)
    icl_values = np.full(len(n_range), np.inf)
    best_model = None
    best_score = np.inf
    best_n = n_min

    for idx, n_comp in enumerate(n_range):
        dm = DirichletMixture(
            n_components=n_comp,
            n_init=3,
            random_state=random_state,
        )
        dm.fit(weights)
        bic = dm.bic(weights)
        icl = _compute_icl(dm, weights)
        bic_values[idx] = bic
        icl_values[idx] = icl

        score = icl if model_selection == "icl" else bic
        if score < best_score:
            best_score = score
            best_model = dm
            best_n = n_comp

    # Stability analysis for optimal n_components
    stability_scores = _compute_stability_dirichlet(
        weights, best_n, n_initializations, random_state
    )

    # Filter stable components
    stable_mask = stability_scores >= stability_threshold
    n_stable = int(np.sum(stable_mask))

    if n_stable == 0:
        stable_mask = np.ones(best_n, dtype=bool)
        n_stable = best_n

    # Get assignments and centroids
    all_labels = best_model.predict(weights)
    # Dirichlet centroids: alpha_k / sum(alpha_k), already on simplex
    simplex_centroids = best_model.means_  # [n_components, K]

    # Map stable components
    stable_indices = np.where(stable_mask)[0]
    stable_centroids = simplex_centroids[stable_indices]
    stable_stability = stability_scores[stable_indices]

    # Remap labels
    label_map = {old: new for new, old in enumerate(stable_indices)}
    component_assignments = np.full(n_cells, -1, dtype=int)
    for old_label, new_label in label_map.items():
        component_assignments[all_labels == old_label] = new_label

    # Handle unstable cells using predict_proba
    unstable_mask = component_assignments == -1
    component_probabilities = None
    if np.any(unstable_mask) and n_stable > 0:
        all_proba = best_model.predict_proba(weights)
        stable_proba = all_proba[:, stable_indices]
        component_probabilities = stable_proba

        unstable_proba = stable_proba[unstable_mask]
        max_prob = unstable_proba.max(axis=1)
        confident_mask = max_prob >= reassignment_confidence
        confident_idx = np.where(unstable_mask)[0][confident_mask]
        component_assignments[confident_idx] = np.argmax(
            stable_proba[confident_idx], axis=1
        )

    # Nearest archetype per component
    archetype_map = np.argmax(stable_centroids, axis=1)

    # Arithmetic mean of archetype weights per component
    component_weight_means = np.zeros((n_stable, K))
    for c in range(n_stable):
        mask = component_assignments == c
        if np.any(mask):
            component_weight_means[c] = weights[mask].mean(axis=0)

    return {
        "n_components_optimal": best_n,
        "n_components_stable": n_stable,
        "component_assignments": component_assignments,
        "component_simplex_means": stable_centroids,
        "component_archetype_map": archetype_map,
        "component_stability_scores": stable_stability,
        "component_weight_means": component_weight_means,
        "bic_values": bic_values,
        "icl_values": icl_values,
        "n_components_tested": n_range,
        "gmm_model": best_model,
        "component_probabilities": component_probabilities,
        "model_type": "dirichlet",
    }


def characterize_components(component_assignments, feature_matrix, n_components):
    """Compute per-component feature profiles.

    For each GMM component, computes the mean feature value across all cells
    assigned to that component.

    Parameters
    ----------
    component_assignments : np.ndarray [n_cells]
        Integer labels from fit_simplex_gmm. Cells with label -1 (assigned
        to unstable components) are excluded.
    feature_matrix : np.ndarray or sparse [n_cells, n_features]
        Feature values (e.g. gene expression, pathway scores).
    n_components : int
        Number of stable components.

    Returns
    -------
    np.ndarray [n_components, n_features]
        Mean feature values per component.
    """
    import scipy.sparse as sp

    if sp.issparse(feature_matrix):
        feature_matrix = feature_matrix.toarray()
    else:
        feature_matrix = np.asarray(feature_matrix)

    profiles = np.zeros((n_components, feature_matrix.shape[1]))
    for c in range(n_components):
        mask = component_assignments == c
        if np.sum(mask) > 0:
            profiles[c] = feature_matrix[mask].mean(axis=0)
    return profiles


def _compute_stability(
    ilr_coords, n_components, covariance_type, n_initializations, random_state
):
    """Compute per-component stability via pairwise cell recovery rate.

    Fits GMM n_initializations times with different seeds. Computes recovery
    rates across ALL pairwise comparisons of initializations using Hungarian
    matching, avoiding first-initialization bias.

    Parameters
    ----------
    ilr_coords : np.ndarray [n_cells, K-1]
        ILR-transformed weight coordinates.
    n_components : int
        Number of GMM components to fit.
    covariance_type : str
        GMM covariance type.
    n_initializations : int
        Number of independent random initializations.
    random_state : int
        Base random seed.

    Returns
    -------
    np.ndarray [n_components]
        Stability score per component (in [0, 1]).
    """
    rng = np.random.default_rng(random_state)
    all_labels = []

    for i in range(n_initializations):
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            n_init=1,
            random_state=int(rng.integers(0, 2**31)),
        )
        gmm.fit(ilr_coords)
        all_labels.append(gmm.predict(ilr_coords))

    return _pairwise_recovery(all_labels, n_components)


def _compute_stability_dirichlet(
    weights, n_components, n_initializations, random_state
):
    """Compute per-component stability for Dirichlet mixture via pairwise recovery.

    Parameters
    ----------
    weights : np.ndarray [n_cells, K]
        Simplex weights.
    n_components : int
        Number of Dirichlet components to fit.
    n_initializations : int
        Number of independent random initializations.
    random_state : int
        Base random seed.

    Returns
    -------
    np.ndarray [n_components]
        Stability score per component (in [0, 1]).
    """
    from peach._core.utils.dirichlet_mixture import DirichletMixture

    rng = np.random.default_rng(random_state)
    all_labels = []

    for i in range(n_initializations):
        dm = DirichletMixture(
            n_components=n_components,
            n_init=1,
            random_state=int(rng.integers(0, 2**31)),
        )
        dm.fit(weights)
        all_labels.append(dm.predict(weights))

    return _pairwise_recovery(all_labels, n_components)


def _pairwise_recovery(all_labels, n_components):
    """Compute pairwise cell recovery rates across all pairs of label assignments.

    Parameters
    ----------
    all_labels : list of np.ndarray [n_cells]
        Label assignments from each initialization.
    n_components : int
        Number of components.

    Returns
    -------
    np.ndarray [n_components]
        Mean recovery rate per component across all pairwise comparisons.
    """
    n_runs = len(all_labels)
    component_recovery = np.zeros(n_components)
    n_pairs = 0

    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            confusion = np.zeros((n_components, n_components))
            for r in range(n_components):
                for t in range(n_components):
                    confusion[r, t] = np.sum(
                        (all_labels[i] == r) & (all_labels[j] == t)
                    )
            row_ind, col_ind = linear_sum_assignment(-confusion)
            for c in range(n_components):
                count_c = np.sum(all_labels[i] == c)
                if count_c == 0:
                    continue
                matched = col_ind[c]
                recovered = np.sum(
                    (all_labels[i] == c) & (all_labels[j] == matched)
                )
                component_recovery[c] += recovered / count_c
            n_pairs += 1

    return component_recovery / max(n_pairs, 1)

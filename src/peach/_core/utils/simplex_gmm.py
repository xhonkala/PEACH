"""Simplex density decomposition via GMM in ILR-transformed weight space.

Fits a Gaussian Mixture Model to archetype weights after ILR transform,
selects the number of components by BIC, and filters by multi-initialization
stability analysis using the Hungarian algorithm for component correspondence.

This module decomposes the cell population into sub-populations that occupy
distinct regions of the archetype weight simplex, enabling identification of
stable cell states and transitional populations.

Reference: McLachlan & Peel (2000), "Finite Mixture Models", Wiley.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment

from peach._core.utils.ilr_transform import ilr_transform, inverse_ilr


def fit_simplex_gmm(
    weights,
    n_components_range=None,
    covariance_type="full",
    n_initializations=20,
    stability_threshold=0.7,
    random_state=42,
):
    """Fit GMM in ILR-transformed weight space with BIC selection and stability analysis.

    Parameters
    ----------
    weights : np.ndarray [n_cells, K]
        Archetype weights (rows sum to 1).
    n_components_range : tuple[int, int] or None
        (min_components, max_components). Default: (K, 3*K).
    covariance_type : str
        GMM covariance type. One of 'full', 'tied', 'diag', 'spherical'.
    n_initializations : int
        Number of random initializations for stability analysis.
    stability_threshold : float
        Minimum stability to report a component (fraction of runs where
        component is recovered).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        n_components_optimal : int
            BIC-selected number of components.
        n_components_stable : int
            Number of components after stability filtering.
        component_assignments : np.ndarray [n_cells]
            Cluster labels for stable components. Cells assigned to unstable
            components get label -1.
        component_simplex_means : np.ndarray [n_stable, K]
            Centroids mapped back to the weight simplex.
        component_archetype_map : np.ndarray [n_stable]
            Index of the nearest archetype for each component centroid
            (argmax of simplex centroid).
        component_stability_scores : np.ndarray [n_stable]
            Stability score for each retained component.
        bic_values : np.ndarray [n_tested]
            BIC values for each n_components tested.
        n_components_tested : np.ndarray [n_tested]
            Array of n_components values tested.
        gmm_model : GaussianMixture
            Fitted GaussianMixture model (in ILR space) for the BIC-optimal
            n_components.
    """
    weights = np.asarray(weights, dtype=np.float64)
    K = weights.shape[1]
    n_cells = weights.shape[0]

    # Transform to ILR space
    ilr_coords = ilr_transform(weights)  # [n_cells, K-1]

    # Determine range
    if n_components_range is None:
        n_components_range = (K, 3 * K)
    n_min, n_max = n_components_range

    # BIC scan
    n_range = np.arange(n_min, n_max + 1)
    bic_values = np.full(len(n_range), np.inf)
    best_gmm = None
    best_bic = np.inf
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
        bic_values[idx] = bic
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_n = n_comp

    # Stability analysis for BIC-optimal n_components
    stability_scores = _compute_stability(
        ilr_coords, best_n, covariance_type, n_initializations, random_state
    )

    # Filter stable components
    stable_mask = stability_scores >= stability_threshold
    n_stable = int(np.sum(stable_mask))

    if n_stable == 0:
        # Fall back to all components if none are stable
        stable_mask = np.ones(best_n, dtype=bool)
        n_stable = best_n

    # Get assignments and centroids
    all_labels = best_gmm.predict(ilr_coords)
    ilr_centroids = best_gmm.means_  # [n_components, K-1]
    simplex_centroids = inverse_ilr(ilr_centroids)  # [n_components, K]

    # Map stable components
    stable_indices = np.where(stable_mask)[0]
    stable_centroids = simplex_centroids[stable_indices]
    stable_stability = stability_scores[stable_indices]

    # Remap labels to only stable components
    label_map = {old: new for new, old in enumerate(stable_indices)}
    component_assignments = np.full(n_cells, -1, dtype=int)
    for old_label, new_label in label_map.items():
        component_assignments[all_labels == old_label] = new_label

    # Nearest archetype per component
    archetype_map = np.argmax(stable_centroids, axis=1)

    return {
        "n_components_optimal": best_n,
        "n_components_stable": n_stable,
        "component_assignments": component_assignments,
        "component_simplex_means": stable_centroids,
        "component_archetype_map": archetype_map,
        "component_stability_scores": stable_stability,
        "bic_values": bic_values,
        "n_components_tested": n_range,
        "gmm_model": best_gmm,
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
    """Compute component stability across multiple initializations.

    For each pair of runs, matches components via the Hungarian algorithm on
    centroid distances. A component in the reference run is "matched" if the
    corresponding component in the test run has centroid distance below an
    adaptive threshold. Stability = fraction of runs where each component
    appears.

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
    all_centroids = []

    for i in range(n_initializations):
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            n_init=1,
            random_state=int(rng.integers(0, 2**31)),
        )
        gmm.fit(ilr_coords)
        all_centroids.append(gmm.means_)

    # Use first run as reference
    ref = all_centroids[0]
    matches = np.zeros(n_components)

    for i in range(1, n_initializations):
        # Cost matrix: pairwise distances between ref and run i centroids
        cost = np.zeros((n_components, n_components))
        for r in range(n_components):
            for c in range(n_components):
                cost[r, c] = np.linalg.norm(ref[r] - all_centroids[i][c])

        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost)

        # Count matches within adaptive threshold
        threshold = np.median(cost) * 0.5
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < threshold:
                matches[r] += 1

    # Stability = fraction of (n_init - 1) runs with a match
    # +1 for reference run itself
    stability = (matches + 1) / n_initializations

    return stability

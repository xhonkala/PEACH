from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

"""
Archetypal Analysis Utilities
=============================

Independent analysis functions for extracting archetypal coordinates,
computing distances, and assigning cells to archetypes.

All functions work directly with AnnData objects and return clean 
DataFrames or store results in AnnData following scVerse conventions.

Main Functions
--------------
get_archetypal_coordinates : Extract coordinates from single batch (DEPRECATED)
extract_and_store_archetypal_coordinates : Extract and store all coordinates in AnnData
compute_archetype_distances : Compute cell-archetype distances in PCA space
bin_cells_by_archetype : Assign cells to archetypes based on distances
test_archetype_recovery : Compare learned vs true archetypes

Type Definitions
----------------
See ``peach._core.types`` for Pydantic models defining return structures:
    - ArchetypalCoordinates
    - ExtractedCoordinates
    - AnnDataKeys (reference for storage locations)

Examples
--------
>>> import peach as pc
>>> 
>>> # After training
>>> results = pc.tl.train_archetypal(adata, n_archetypes=5)
>>> 
>>> # Compute distances (stores in adata.obsm['archetype_distances'])
>>> distances_df = pc.tl.archetypal_coordinates(adata)
>>> 
>>> # Assign cells (stores in adata.obs['archetypes'])
>>> pc.tl.assign_archetypes(adata, percentage_per_archetype=0.1)
"""


def get_archetypal_coordinates(model, input: torch.Tensor, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Extract archetypal coordinates from trained model for a single batch.

    .. deprecated::
        Use :func:`extract_and_store_archetypal_coordinates` for AnnData integration.
        This function remains for internal use and backward compatibility.

    Parameters
    ----------
    model : torch.nn.Module | dict
        Trained Deep_AA model, or a results dict containing 'final_model' or 'model'.
    input : torch.Tensor
        Input data tensor of shape (batch_size, n_features).
    device : str, default: 'cpu'
        Computing device ('cpu', 'cuda', 'mps').

    Returns
    -------
    dict
        Dictionary containing coordinate tensors:

        - ``A`` : torch.Tensor
            Archetypal coordinates (cell weights), shape (batch_size, n_archetypes).
            Each row sums to 1 (barycentric coordinates).
        - ``B`` : torch.Tensor
            Dummy B matrix for compatibility, shape (batch_size, n_archetypes).
            Values are uniform: 1/batch_size.
        - ``Y`` : torch.Tensor
            Archetype positions in input space, shape (n_archetypes, n_features).
            These are the learned extreme points.
        - ``mu`` : torch.Tensor
            Encoder means, shape (batch_size, n_archetypes).
        - ``log_var`` : torch.Tensor
            Encoder log variances, shape (batch_size, n_archetypes).
        - ``z`` : torch.Tensor
            Reparameterized latent variables, shape (batch_size, n_archetypes).

    Raises
    ------
    ValueError
        If model is a dict but doesn't contain 'final_model' or 'model' keys.

    Notes
    -----
    This function sets the model to eval mode and uses torch.no_grad() context.

    For full dataset processing with AnnData integration, use
    :func:`extract_and_store_archetypal_coordinates` instead.

    Examples
    --------
    >>> # Internal use - prefer extract_and_store_archetypal_coordinates
    >>> coords = get_archetypal_coordinates(model, batch_tensor, device="cuda")
    >>> A_matrix = coords["A"]  # (batch_size, n_archetypes)
    >>> archetypes = coords["Y"]  # (n_archetypes, n_features)

    See Also
    --------
    extract_and_store_archetypal_coordinates : Recommended function for full datasets
    peach._core.types.ArchetypalCoordinates : Type definition for return structure
    """
    import warnings

    warnings.warn(
        "get_archetypal_coordinates is deprecated. Use extract_and_store_archetypal_coordinates() for AnnData integration.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Handle case where model might be a dict (from training results)
    if isinstance(model, dict):
        if "final_model" in model:
            actual_model = model["final_model"]
        elif "model" in model:
            actual_model = model["model"]
        else:
            raise ValueError(
                f"Model is a dict but doesn't contain 'final_model' or 'model' keys. Keys: {list(model.keys())}"
            )
        print(f" Warning: Model was a dict, extracted actual model of type {type(actual_model)}")
    else:
        actual_model = model

    actual_model.eval()

    # Detect model's device and move input there (fixes GPU support)
    model_device = next(actual_model.parameters()).device
    input = input.to(model_device)

    with torch.no_grad():
        outputs = actual_model.forward(input)

        # Create dummy B matrix for training loop compatibility
        batch_size = input.shape[0]
        dummy_B = torch.ones(batch_size, actual_model.n_archetypes, device=input.device) / batch_size

        return {
            "A": outputs["A"],
            "B": dummy_B,
            "Y": outputs["Y"].detach(),  # Detach archetype positions to remove grad tracking
            "mu": outputs["mu"],
            "log_var": outputs["log_var"],
            "z": outputs["z"],
        }


def test_archetype_recovery(
    model, true_archetypes: torch.Tensor | np.ndarray, dataloader, device: str = "cpu", tolerance: float = 0.5
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Test how well learned archetypes recover ground truth positions.

    Uses the Hungarian algorithm to find optimal assignment between
    learned and true archetypes, then computes recovery metrics.

    Parameters
    ----------
    model : torch.nn.Module
        Trained Deep_AA model.
    true_archetypes : torch.Tensor | np.ndarray
        Ground truth archetype positions, shape (n_archetypes, n_features).
    dataloader : torch.utils.data.DataLoader
        DataLoader for extracting learned archetypes.
    device : str, default: 'cpu'
        Computing device.
    tolerance : float, default: 0.5
        Tolerance multiplier for recovery success.
        Recovery is successful if mean_distance < tolerance * data_scale.

    Returns
    -------
    tuple
        Three-element tuple:

        - ``recovery_metrics`` : dict
            Recovery quality metrics:

            - ``mean_distance`` : float - Mean distance between matched archetypes
            - ``max_distance`` : float - Maximum distance among matches
            - ``normalized_mean_distance`` : float - Mean distance / data_scale
            - ``recovery_success`` : bool - Whether recovery meets tolerance
            - ``individual_distances`` : np.ndarray - Distance per archetype pair
            - ``assignment`` : list[tuple] - Optimal (learned_idx, true_idx) pairs

        - ``learned_Y`` : np.ndarray
            Learned archetype positions, shape (n_archetypes, n_features).
        - ``true_archetypes`` : np.ndarray
            True archetype positions (converted to numpy if needed).

    Notes
    -----
    The Hungarian algorithm finds the assignment that minimizes total
    distance between learned and true archetypes. This handles the
    permutation invariance of archetype discovery.

    ``data_scale`` is computed as ``np.std(true_archetypes)``.

    Examples
    --------
    >>> # With synthetic data where true archetypes are known
    >>> from peach._core.utils.convex_synth_data import generate_simplex_data
    >>> data, true_archetypes = generate_simplex_data(n_archetypes=4)
    >>> # ... train model ...
    >>> metrics, learned, true = test_archetype_recovery(model, true_archetypes, dataloader)
    >>> print(f"Recovery successful: {metrics['recovery_success']}")
    >>> print(f"Normalized distance: {metrics['normalized_mean_distance']:.4f}")
    >>> print(f"Assignment: {metrics['assignment']}")

    See Also
    --------
    compare_archetypal_recovery : Simpler comparison without model
    peach._core.types.ArchetypeRecoveryMetrics : Type definition
    """
    model.eval()

    data_batch = next(iter(dataloader))[0].to(device)
    coords = get_archetypal_coordinates(model, data_batch, device)
    learned_Y = coords["Y"].detach().cpu().numpy()  # Now [n_archetypes, input_dim] with my fix

    # Convert true archetypes to numpy if needed
    if isinstance(true_archetypes, torch.Tensor):
        true_archetypes = true_archetypes.detach().cpu().numpy()

    print(f"True archetypes shape: {true_archetypes.shape}")  # [4, 20]
    print(f"Learned archetypes shape: {learned_Y.shape}")  # [4, 20] - NOW MATCHES!

    # Calculate distances (no transpose needed now!)
    distances = cdist(learned_Y, true_archetypes)  # Both [n_archetypes, input_dim]

    # Find optimal assignment
    learned_idx, true_idx = linear_sum_assignment(distances)

    # Calculate recovery metrics
    matched_distances = distances[learned_idx, true_idx]
    data_scale = np.std(true_archetypes)  # Use true_archetypes directly

    recovery_metrics = {
        "mean_distance": np.mean(matched_distances),
        "max_distance": np.max(matched_distances),
        "normalized_mean_distance": np.mean(matched_distances) / data_scale,
        "recovery_success": np.mean(matched_distances)
        < tolerance * data_scale,  # More reasonable: mean distance < tolerance
        "individual_distances": matched_distances,
        "assignment": list(zip(learned_idx, true_idx, strict=False)),
    }

    print("\nArchetype Recovery Test Results:")
    print(f"Mean distance: {recovery_metrics['mean_distance']:.4f}")
    print(f"Normalized mean distance: {recovery_metrics['normalized_mean_distance']:.4f}")
    print(f"Recovery successful: {recovery_metrics['recovery_success']}")
    print(f"Archetype assignments (learned->true): {recovery_metrics['assignment']}")

    return recovery_metrics, learned_Y, true_archetypes  # Both now [n_archetypes, input_dim]


def compare_archetypal_recovery(
    true_archetypes: np.ndarray, estimated_archetypes: np.ndarray, tolerance: float = 0.5
) -> tuple[float, float]:
    """Compare learned vs true archetypes without model dependency.

    Handles dimensional mismatches by using only overlapping dimensions.
    Useful when true archetypes are in full feature space but estimated
    archetypes are in reduced PCA space.

    Parameters
    ----------
    true_archetypes : np.ndarray
        Ground truth archetypes, shape (n_archetypes, n_features).
    estimated_archetypes : np.ndarray
        Estimated archetypes, shape (n_archetypes, n_features_reduced).
    tolerance : float, default: 0.5
        Tolerance for assignment accuracy calculation.

    Returns
    -------
    tuple
        Two-element tuple:

        - ``recovery_score`` : float
            Normalized mean distance (lower is better).
            Computed as mean_distance / data_scale.
        - ``assignment_accuracy`` : float
            Fraction of archetypes recovered within tolerance (higher is better).

    Notes
    -----
    When dimensions don't match, only the first ``min(n_features, n_features_reduced)``
    dimensions are used for comparison.

    Examples
    --------
    >>> # Compare archetypes from different methods
    >>> score, accuracy = compare_archetypal_recovery(
    ...     true_archetypes,  # shape (4, 100)
    ...     pca_archetypes,  # shape (4, 30) - in PCA space
    ... )
    >>> print(f"Recovery score: {score:.4f} (lower is better)")
    >>> print(f"Accuracy: {accuracy:.1%}")
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    # Convert to numpy if needed
    if hasattr(true_archetypes, "detach"):
        true_archetypes = true_archetypes.detach().cpu().numpy()
    if hasattr(estimated_archetypes, "detach"):
        estimated_archetypes = estimated_archetypes.detach().cpu().numpy()

    # Handle dimensional mismatch: use minimum dimensions
    min_dims = min(true_archetypes.shape[1], estimated_archetypes.shape[1])
    true_subset = true_archetypes[:, :min_dims]
    estimated_subset = estimated_archetypes[:, :min_dims]

    print("Archetype recovery comparison:")
    print(f"  True archetypes: {true_archetypes.shape} → using first {min_dims} dims")
    print(f"  Estimated archetypes: {estimated_archetypes.shape} → using first {min_dims} dims")

    # Calculate distances between estimated and true archetypes
    distances = cdist(estimated_subset, true_subset)

    # Find optimal assignment using Hungarian algorithm
    estimated_idx, true_idx = linear_sum_assignment(distances)

    # Calculate recovery metrics
    matched_distances = distances[estimated_idx, true_idx]
    data_scale = np.std(true_subset)

    # Recovery score: normalized mean distance (lower is better)
    recovery_score = np.mean(matched_distances) / data_scale

    # Assignment accuracy: fraction within tolerance
    assignment_accuracy = np.mean(matched_distances < tolerance * data_scale)

    print(f"  Recovery score: {recovery_score:.4f} (lower is better)")
    print(f"  Assignment accuracy: {assignment_accuracy:.4f} (higher is better)")

    return recovery_score, assignment_accuracy


# def bin_cells_by_archetype(
#     adata,
#     percentage_per_archetype: float = 0.1,
#     obsm_key: str = 'archetype_distances',
#     obs_key: str = 'archetypes',
#     verbose: bool = True
# ) -> pd.DataFrame:
#     """
#     Assign cells to archetypes using AnnData-stored distance matrix.

#     CRITICAL: Now uses adata.obsm['archetype_distances'] as primary data source.
#     Updates adata.obs['archetypes'] with categorical archetype assignments.

#     WORKFLOW:
#     1. Extract distance matrix from adata.obsm['archetype_distances']
#     2. Sort each archetype column independently from smallest to largest distances
#     3. Identify cells at the top % specified percentage closest to each archetype
#     4. Store assignments in adata.obs['archetypes'] as categorical
#     5. Return DataFrame for downstream analysis

#     Args:
#         adata: AnnData object with distance matrix in .obsm
#         percentage_per_archetype: Percentage (0.0-1.0) of cells closest to each archetype
#         obsm_key: Key for distance matrix in adata.obsm
#         obs_key: Key for storing assignments in adata.obs
#         verbose: Whether to print assignment statistics

#     Returns:
#         DataFrame with columns:
#             - 'cell_id': adata.obs.index values (canonical cell IDs)
#             - 'cell_idx': 0-based position indices
#             - 'archetype_label': 'archetype_1', 'archetype_2', etc. or 'no_archetype'
#             - 'archetype_idx': Numeric archetype index (or -1 for no_archetype)
#             - 'distance': Distance to assigned archetype
#             - 'rank_in_archetype': Rank within archetype (0 = closest)
#     """
#     # CRITICAL: Use AnnData as primary data source
#     if obsm_key not in adata.obsm:
#         raise ValueError(f"Distance matrix not found in adata.obsm['{obsm_key}']. "
#                         f"Run compute_archetype_distances() first.")

#     distance_matrix = adata.obsm[obsm_key]  # [n_cells, n_archetypes]
#     n_cells, n_archetypes = distance_matrix.shape
#     n_cells_per_archetype = int(n_cells * percentage_per_archetype)

#     if verbose:
#         print(f" AnnData-centric archetype binning...")
#         print(f"   Distance matrix: {distance_matrix.shape} (from adata.obsm['{obsm_key}'])")
#         print(f"   Canonical cell reference: adata.obs.index ({len(adata.obs)} cells)")
#         print(f"   Selecting top {n_cells_per_archetype} cells ({percentage_per_archetype:.1%}) per archetype")

#     # Validate alignment
#     if len(adata.obs) != n_cells:
#         raise ValueError(f"Alignment error: adata.obs has {len(adata.obs)} cells but distance matrix has {n_cells}")

#     # Create assignment tracking
#     assignments = []
#     archetype_assignments = {}  # Track overlaps

#     # Process each archetype
#     for arch_idx in range(n_archetypes):
#         arch_distances = distance_matrix[:, arch_idx]

#         # Get indices of closest cells (smallest distances first)
#         sorted_indices = np.argsort(arch_distances)
#         closest_indices = sorted_indices[:n_cells_per_archetype]

#         # Record assignments using canonical cell IDs
#         for rank, cell_position in enumerate(closest_indices):
#             cell_id = adata.obs.index[cell_position]  # Canonical cell ID
#             cell_distance = arch_distances[cell_position]

#             assignments.append({
#                 'cell_id': cell_id,
#                 'cell_idx': cell_position,  # 0-based position
#                 'archetype_label': f'archetype_{arch_idx + 1}',  # 1-indexed for biology
#                 'archetype_idx': arch_idx,
#                 'distance': float(cell_distance),
#                 'rank_in_archetype': rank
#             })

#             # Track overlaps
#             if cell_position not in archetype_assignments:
#                 archetype_assignments[cell_position] = []
#             archetype_assignments[cell_position].append(arch_idx)

#         if verbose:
#             closest_distances = arch_distances[closest_indices]
#             print(f"   Archetype {arch_idx + 1}: {len(closest_indices)} cells, "
#                   f"distance range: [{closest_distances.min():.4f}, {closest_distances.max():.4f}], "
#                   f"mean: {closest_distances.mean():.4f}")

#     # Create assignments DataFrame
#     assignments_df = pd.DataFrame(assignments)

#     # Add unassigned cells
#     assigned_positions = set(assignments_df['cell_idx'].values)
#     all_positions = set(range(n_cells))
#     unassigned_positions = all_positions - assigned_positions

#     for cell_position in unassigned_positions:
#         cell_id = adata.obs.index[cell_position]
#         assignments_df = pd.concat([assignments_df, pd.DataFrame([{
#             'cell_id': cell_id,
#             'cell_idx': cell_position,
#             'archetype_label': 'no_archetype',
#             'archetype_idx': -1,
#             'distance': np.nan,
#             'rank_in_archetype': -1
#         }])], ignore_index=True)

#     # Sort by position for consistency
#     assignments_df = assignments_df.sort_values('cell_idx').reset_index(drop=True)

#     # Add overlap detection
#     assignments_df['multiple_archetypes'] = assignments_df['cell_idx'].apply(
#         lambda x: len(archetype_assignments.get(x, [])) > 1
#     )

#     # STORE IN ANNDATA.OBS (PRIMARY STORAGE)
#     # ======================================
#     # Create archetype categorical labels aligned with adata.obs.index
#     archetype_labels = ['no_archetype'] * len(adata.obs)

#     for _, row in assignments_df.iterrows():
#         if row['archetype_label'] != 'no_archetype':
#             position = int(row['cell_idx'])
#             archetype_labels[position] = row['archetype_label']

#     # Store as categorical in adata.obs
#     adata.obs[obs_key] = pd.Categorical(archetype_labels)

#     if verbose:
#         print(f"\n[STATS] Assignment Summary:")
#         print(f"   Total cells: {n_cells}")
#         for arch_idx in range(n_archetypes):
#             count = (assignments_df['archetype_idx'] == arch_idx).sum()
#             print(f"   Archetype {arch_idx + 1}: {count} cells ({100*count/n_cells:.1f}%)")

#         unassigned_count = (assignments_df['archetype_label'] == 'no_archetype').sum()
#         print(f"   No archetype: {unassigned_count} cells ({100*unassigned_count/n_cells:.1f}%)")

#         overlap_count = assignments_df['multiple_archetypes'].sum()
#         if overlap_count > 0:
#             print(f"   [WARNING]  Overlapping assignments: {overlap_count} cells")

#         print(f"\n[OK] Stored assignments in adata.obs['{obs_key}']:")
#         print(f"   Categories: {list(adata.obs[obs_key].cat.categories)}")
#         for cat, count in adata.obs[obs_key].value_counts().items():
#             print(f"   {cat}: {count} cells ({100*count/len(adata.obs):.1f}%)")

#     return assignments_df


def bin_cells_by_archetype(
    adata,
    percentage_per_archetype: float = 0.1,
    obsm_key: str = "archetype_distances",
    obs_key: str = "archetypes",
    include_central_archetype: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Assign cells to archetypes based on distance matrix.

    For each archetype, selects the closest N% of cells and assigns them
    to that archetype. Optionally creates a central "archetype_0" for
    generalist cells closest to the global centroid.

    Parameters
    ----------
    adata : AnnData
        AnnData object with distance matrix in ``adata.obsm[obsm_key]``.
        Run :func:`compute_archetype_distances` first.
    percentage_per_archetype : float, default: 0.1
        Fraction of cells (0.0-1.0) to assign to each archetype.
        E.g., 0.1 assigns the closest 10% of cells to each archetype.
    obsm_key : str, default: 'archetype_distances'
        Key for distance matrix in ``adata.obsm``.
    obs_key : str, default: 'archetypes'
        Key for storing assignments in ``adata.obs``.
    include_central_archetype : bool, default: True
        Whether to create archetype_0 for generalist cells closest to
        the global centroid (mean position across all archetypes).
        Useful for studying specialization trajectories.
    verbose : bool, default: True
        Whether to print assignment statistics.

    Returns
    -------
    pd.DataFrame
        DataFrame with assignment details:

        - ``cell_id`` : str
            Cell identifier from ``adata.obs.index``.
        - ``cell_idx`` : int
            0-based position index.
        - ``archetype_label`` : str
            Assignment label: 'archetype_0' (central), 'archetype_1', ...,
            or 'no_archetype' for unassigned cells.
        - ``archetype_idx`` : int
            Numeric index: 0 for central, 1+ for extremal, -1 for unassigned.
        - ``distance`` : float
            Distance to assigned archetype (mean distance for archetype_0).
            NaN for unassigned cells.
        - ``rank_in_archetype`` : int
            Rank within archetype (0 = closest). -1 for unassigned.
        - ``multiple_archetypes`` : bool
            True if cell was assigned to multiple archetypes (overlap).

    Stores
    ------
    The function stores the following in AnnData:

    - ``adata.obs[obs_key]`` : pd.Categorical
        Archetype assignments as categorical variable.
        Categories: 'archetype_0', 'archetype_1', ..., 'no_archetype'.
        Default key: 'archetypes'.

    Raises
    ------
    ValueError
        If distance matrix not found in ``adata.obsm[obsm_key]``.
        If ``adata.obs`` length doesn't match distance matrix rows.

    Notes
    -----
    **Overlap Handling**: A cell may be among the closest N% for multiple
    archetypes. Such cells appear multiple times in the returned DataFrame
    but get a single assignment in ``adata.obs`` (the first encountered).
    The ``multiple_archetypes`` column flags these overlapping cells.

    **Central Archetype**: When ``include_central_archetype=True``,
    archetype_0 represents cells with balanced contributions from all
    extremal archetypes. These are "generalist" cells useful for:

    - Studying specialization trajectories (center → extreme)
    - Identifying progenitor-like states
    - CellRank terminal state analysis

    **Archetype Numbering**:

    - With ``include_central_archetype=True``: 0=central, 1,2,3,...=extremal
    - With ``include_central_archetype=False``: 1,2,3,...=extremal (no 0)

    Examples
    --------
    >>> import peach as pc
    >>> # After computing distances
    >>> compute_archetype_distances(adata)
    >>> # Assign top 10% to each archetype, including central
    >>> assignments_df = bin_cells_by_archetype(adata, percentage_per_archetype=0.1, include_central_archetype=True)
    >>> # Access assignments from AnnData (preferred)
    >>> print(adata.obs["archetypes"].value_counts())
    >>> # Subset to cells assigned to archetype_1
    >>> arch1_cells = adata[adata.obs["archetypes"] == "archetype_1"]
    >>> # Check for overlapping assignments
    >>> overlapping = assignments_df[assignments_df["multiple_archetypes"]]
    >>> print(f"Cells assigned to multiple archetypes: {len(overlapping)}")
    >>> # Use with scanpy for visualization
    >>> import scanpy as sc
    >>> sc.pl.umap(adata, color="archetypes")

    See Also
    --------
    compute_archetype_distances : Must be run before this function
    peach.tl.assign_archetypes : User-facing wrapper
    select_cells : Advanced cell selection with weight criteria
    """
    # CRITICAL: Use AnnData as primary data source
    if obsm_key not in adata.obsm:
        raise ValueError(
            f"Distance matrix not found in adata.obsm['{obsm_key}']. Run compute_archetype_distances() first."
        )

    distance_matrix = adata.obsm[obsm_key]  # [n_cells, n_archetypes]
    n_cells, n_archetypes = distance_matrix.shape
    n_cells_per_archetype = int(n_cells * percentage_per_archetype)

    if verbose:
        print(" AnnData-centric archetype binning...")
        print(f"   Distance matrix: {distance_matrix.shape} (from adata.obsm['{obsm_key}'])")
        print(f"   Canonical cell reference: adata.obs.index ({len(adata.obs)} cells)")
        print(f"   Selecting top {n_cells_per_archetype} cells ({percentage_per_archetype:.1%}) per archetype")
        if include_central_archetype:
            print("   INCLUDING central archetype_0 (generalist cells)")

    # Validate alignment
    if len(adata.obs) != n_cells:
        raise ValueError(f"Alignment error: adata.obs has {len(adata.obs)} cells but distance matrix has {n_cells}")

    # Create assignment tracking
    assignments = []
    archetype_assignments = {}  # Track overlaps

    # NEW: Create central archetype_0 if requested
    # ============================================
    if include_central_archetype:
        # Calculate distance to global centroid (mean distance across all archetypes)
        centroid_distances = np.mean(distance_matrix, axis=1)

        # Get cells closest to centroid
        centroid_sorted_indices = np.argsort(centroid_distances)
        central_closest_indices = centroid_sorted_indices[:n_cells_per_archetype]

        # Record central archetype assignments
        for rank, cell_position in enumerate(central_closest_indices):
            cell_id = adata.obs.index[cell_position]
            cell_distance = centroid_distances[cell_position]

            assignments.append(
                {
                    "cell_id": cell_id,
                    "cell_idx": cell_position,
                    "archetype_label": "archetype_0",  # Central/generalist archetype
                    "archetype_idx": 0,  # Reserve 0 for central
                    "distance": float(cell_distance),
                    "rank_in_archetype": rank,
                }
            )

            # Track overlaps
            if cell_position not in archetype_assignments:
                archetype_assignments[cell_position] = []
            archetype_assignments[cell_position].append(0)  # 0 for central

        if verbose:
            central_distances = centroid_distances[central_closest_indices]
            print(
                f"   Archetype 0 (central): {len(central_closest_indices)} cells, "
                f"centroid distance range: [{central_distances.min():.4f}, {central_distances.max():.4f}], "
                f"mean: {central_distances.mean():.4f}"
            )

    # Process each extremal archetype
    # ===============================
    for arch_idx in range(n_archetypes):
        arch_distances = distance_matrix[:, arch_idx]

        # Get indices of closest cells (smallest distances first)
        sorted_indices = np.argsort(arch_distances)
        closest_indices = sorted_indices[:n_cells_per_archetype]

        # Record assignments using canonical cell IDs
        for rank, cell_position in enumerate(closest_indices):
            cell_id = adata.obs.index[cell_position]
            cell_distance = arch_distances[cell_position]

            # Adjust archetype numbering if central archetype exists
            archetype_num = arch_idx + 1 if include_central_archetype else arch_idx + 1
            archetype_storage_idx = arch_idx + 1 if include_central_archetype else arch_idx

            assignments.append(
                {
                    "cell_id": cell_id,
                    "cell_idx": cell_position,
                    "archetype_label": f"archetype_{archetype_num}",
                    "archetype_idx": archetype_storage_idx,
                    "distance": float(cell_distance),
                    "rank_in_archetype": rank,
                }
            )

            # Track overlaps
            if cell_position not in archetype_assignments:
                archetype_assignments[cell_position] = []
            archetype_assignments[cell_position].append(archetype_storage_idx)

        if verbose:
            closest_distances = arch_distances[closest_indices]
            print(
                f"   Archetype {archetype_num}: {len(closest_indices)} cells, "
                f"distance range: [{closest_distances.min():.4f}, {closest_distances.max():.4f}], "
                f"mean: {closest_distances.mean():.4f}"
            )

    # Create assignments DataFrame
    assignments_df = pd.DataFrame(assignments)

    # Add unassigned cells
    assigned_positions = set(assignments_df["cell_idx"].values)
    all_positions = set(range(n_cells))
    unassigned_positions = all_positions - assigned_positions

    for cell_position in unassigned_positions:
        cell_id = adata.obs.index[cell_position]
        assignments_df = pd.concat(
            [
                assignments_df,
                pd.DataFrame(
                    [
                        {
                            "cell_id": cell_id,
                            "cell_idx": cell_position,
                            "archetype_label": "no_archetype",
                            "archetype_idx": -1,
                            "distance": np.nan,
                            "rank_in_archetype": -1,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    # Sort by position for consistency
    assignments_df = assignments_df.sort_values("cell_idx").reset_index(drop=True)

    # Add overlap detection
    assignments_df["multiple_archetypes"] = assignments_df["cell_idx"].apply(
        lambda x: len(archetype_assignments.get(x, [])) > 1
    )

    # STORE IN ANNDATA.OBS (PRIMARY STORAGE)
    # ======================================
    # Create archetype categorical labels aligned with adata.obs.index
    archetype_labels = ["no_archetype"] * len(adata.obs)

    for _, row in assignments_df.iterrows():
        if row["archetype_label"] != "no_archetype":
            position = int(row["cell_idx"])
            archetype_labels[position] = row["archetype_label"]

    # Store as categorical in adata.obs
    adata.obs[obs_key] = pd.Categorical(archetype_labels)

    if verbose:
        print("\n[STATS] Assignment Summary:")
        print(f"   Total cells: {n_cells}")

        if include_central_archetype:
            central_count = (assignments_df["archetype_idx"] == 0).sum()
            print(f"   Archetype 0 (central): {central_count} cells ({100 * central_count / n_cells:.1f}%)")

        for arch_idx in range(n_archetypes):
            storage_idx = arch_idx + 1 if include_central_archetype else arch_idx
            display_num = arch_idx + 1 if include_central_archetype else arch_idx + 1
            count = (assignments_df["archetype_idx"] == storage_idx).sum()
            print(f"   Archetype {display_num}: {count} cells ({100 * count / n_cells:.1f}%)")

        unassigned_count = (assignments_df["archetype_label"] == "no_archetype").sum()
        print(f"   No archetype: {unassigned_count} cells ({100 * unassigned_count / n_cells:.1f}%)")

        overlap_count = assignments_df["multiple_archetypes"].sum()
        if overlap_count > 0:
            print(f"   [WARNING]  Overlapping assignments: {overlap_count} cells")

        print(f"\n[OK] Stored assignments in adata.obs['{obs_key}']:")
        print(f"   Categories: {list(adata.obs[obs_key].cat.categories)}")
        for cat, count in adata.obs[obs_key].value_counts().items():
            print(f"   {cat}: {count} cells ({100 * count / len(adata.obs):.1f}%)")

    return assignments_df


def select_cells(
    coordinates_df: pd.DataFrame,
    selection_criteria: dict[str, str],
    adata: Any | None = None,
    return_subset_adata: bool = True,
    verbose: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, Any]:
    """Select cells using percentage-based weight criteria.

    Allows complex inclusion/exclusion patterns using intuitive percentage
    strings. Perfect for identifying specific cell populations like
    transitional states or pure archetype representatives.

    Parameters
    ----------
    coordinates_df : pd.DataFrame
        DataFrame from :func:`get_all_archetypal_coordinates` with weight columns.
        Must have columns like 'archetype_0_weight', 'archetype_1_weight', etc.
    selection_criteria : dict[str, str]
        Dictionary mapping archetype names to percentage criteria.

        Format: ``{'archetype_N': 'OP X%'}`` where:

        - ``archetype_N`` : Archetype identifier (e.g., 'archetype_1' or just '1')
        - ``OP`` : Comparison operator ('>', '<', '>=', '<=', '==')
        - ``X%`` : Percentage threshold (e.g., '30%')

        All criteria are combined with AND logic.

    adata : AnnData | None, default: None
        Optional AnnData object to subset.
    return_subset_adata : bool, default: True
        If True and adata provided, return (DataFrame, subset_adata).
    verbose : bool, default: True
        Whether to print selection statistics.

    Returns
    -------
    pd.DataFrame | tuple
        If ``adata`` is None or ``return_subset_adata`` is False:
            DataFrame with selected cells and their weight profiles.
        If ``adata`` provided and ``return_subset_adata`` is True:
            Tuple of (DataFrame, subset_adata) or (DataFrame, None) if no cells selected.

    Raises
    ------
    ValueError
        If no weight columns found in coordinates_df.
        If specified archetype not found in available archetypes.
        If criteria string format is invalid.

    Notes
    -----
    **Criteria Examples**:

    - ``'>30%'`` : Weight greater than 0.30
    - ``'<10%'`` : Weight less than 0.10
    - ``'>=50%'`` : Weight greater than or equal to 0.50
    - ``'==25%'`` : Weight approximately equal to 0.25 (±1% tolerance)

    **Selection Patterns**:

    - Transitional cells: ``{'archetype_1': '>20%', 'archetype_3': '>20%'}``
    - Pure representatives: ``{'archetype_1': '>70%', 'archetype_2': '<15%'}``
    - Exclusion: ``{'archetype_4': '<5%'}`` (exclude cells near archetype_4)

    The subset AnnData includes selection metadata in ``.uns``:

    - ``uns['selection_criteria']`` : The criteria used
    - ``uns['selection_stats']`` : n_selected, n_total, selection_percentage

    Examples
    --------
    >>> # Select transitional cells between archetypes 1 and 3
    >>> transitional = select_cells(
    ...     coordinates_df, {"archetype_1": ">20%", "archetype_3": ">20%", "archetype_2": "<10%"}
    ... )
    >>> # Select pure archetype_1 representatives
    >>> pure_arch1 = select_cells(coordinates_df, {"archetype_1": ">70%", "archetype_2": "<15%", "archetype_3": "<15%"})
    >>> # Select with AnnData subsetting for downstream analysis
    >>> selected_df, selected_adata = select_cells(
    ...     coordinates_df, {"archetype_1": ">30%"}, adata=adata, return_subset_adata=True
    ... )
    >>> if selected_adata is not None:
    ...     print(f"Selected {selected_adata.n_obs} cells")
    ...     # Run differential expression on selected cells
    ...     sc.tl.rank_genes_groups(selected_adata, groupby="condition")

    See Also
    --------
    get_all_archetypal_coordinates : Generate input coordinates_df
    bin_cells_by_archetype : Simpler percentage-based assignment
    """
    if verbose:
        print(" Selecting cells based on archetype weight criteria...")
        print(f"   Input criteria: {selection_criteria}")

    # Parse selection criteria and apply filters
    def parse_percentage(criteria_str: str) -> tuple[str, float]:
        """Parse strings like '>30%' or '<10%' into operator and threshold."""
        criteria_str = criteria_str.strip()
        if criteria_str.startswith(">"):
            return ">", float(criteria_str[1:].rstrip("%")) / 100.0
        elif criteria_str.startswith("<"):
            return "<", float(criteria_str[1:].rstrip("%")) / 100.0
        elif criteria_str.startswith(">="):
            return ">=", float(criteria_str[2:].rstrip("%")) / 100.0
        elif criteria_str.startswith("<="):
            return "<=", float(criteria_str[2:].rstrip("%")) / 100.0
        elif criteria_str.startswith("=") or criteria_str.startswith("=="):
            threshold_str = criteria_str.lstrip("=").rstrip("%")
            return "==", float(threshold_str) / 100.0
        else:
            raise ValueError(
                f"Invalid criteria format: '{criteria_str}'. Use formats like '>30%', '<10%', '>=50%', etc."
            )

    # Get weight columns from coordinates DataFrame
    weight_cols = [col for col in coordinates_df.columns if col.endswith("_weight")]
    if not weight_cols:
        raise ValueError(
            "No weight columns found. Expected columns like 'archetype_0_weight', 'archetype_1_weight', etc."
        )

    # Create boolean mask for cell selection
    n_cells = len(coordinates_df)
    mask = pd.Series([True] * n_cells, index=coordinates_df.index)

    if verbose:
        print(f"   Found {len(weight_cols)} archetype weights, {n_cells} total cells")
        print(f"   Applying {len(selection_criteria)} selection criteria:")

    # Apply each selection criterion
    for archetype_label, criteria_str in selection_criteria.items():
        # Find the corresponding weight column
        if archetype_label.startswith("archetype_"):
            weight_col = f"{archetype_label}_weight"
        else:
            # Handle both 'archetype_1' and '1' formats
            if archetype_label.isdigit():
                weight_col = f"archetype_{archetype_label}_weight"
            else:
                weight_col = f"{archetype_label}_weight"

        if weight_col not in coordinates_df.columns:
            available_archetypes = [col.replace("_weight", "") for col in weight_cols]
            raise ValueError(f"Archetype '{archetype_label}' not found. Available archetypes: {available_archetypes}")

        # Parse and apply the criterion
        operator, threshold = parse_percentage(criteria_str)
        archetype_weights = coordinates_df[weight_col]

        if operator == ">":
            criterion_mask = archetype_weights > threshold
        elif operator == "<":
            criterion_mask = archetype_weights < threshold
        elif operator == ">=":
            criterion_mask = archetype_weights >= threshold
        elif operator == "<=":
            criterion_mask = archetype_weights <= threshold
        elif operator == "==":
            criterion_mask = np.isclose(archetype_weights, threshold, atol=0.01)  # 1% tolerance for equality

        # Apply this criterion to the overall mask (AND logic)
        mask = mask & criterion_mask

        if verbose:
            n_meeting_criterion = criterion_mask.sum()
            n_remaining = mask.sum()
            print(
                f"      {archetype_label} {criteria_str}: {n_meeting_criterion} cells meet criterion, {n_remaining} remain after AND"
            )

    # Select cells meeting all criteria
    selected_indices = coordinates_df.index[mask]
    selected_df = coordinates_df.loc[selected_indices].copy()

    if verbose:
        n_selected = len(selected_df)
        percentage = 100 * n_selected / n_cells
        print("\n[STATS] Selection Results:")
        print(f"   Selected {n_selected} cells ({percentage:.1f}% of total)")

        if n_selected > 0:
            print("\n   Weight statistics for selected cells:")
            for weight_col in weight_cols:
                archetype_name = weight_col.replace("_weight", "")
                weights = selected_df[weight_col]
                print(
                    f"      {archetype_name}: mean={weights.mean():.3f}, "
                    f"std={weights.std():.3f}, "
                    f"range=[{weights.min():.3f}, {weights.max():.3f}]"
                )

            # Show dominant archetype distribution
            dominant_archetypes = selected_df["dominant_archetype"].value_counts()
            print("\n   Dominant archetype distribution in selected cells:")
            for arch_idx, count in dominant_archetypes.items():
                arch_name = f"archetype_{arch_idx}"
                percentage_of_selected = 100 * count / n_selected
                print(f"      {arch_name}: {count} cells ({percentage_of_selected:.1f}%)")
        else:
            print("   [WARNING]  No cells met all selection criteria!")

    # Handle AnnData subsetting if requested
    if adata is not None and return_subset_adata:
        if verbose:
            print("\n Creating AnnData subset...")

        if len(selected_df) > 0:
            # Get cell indices for AnnData subsetting
            cell_indices = selected_df["cell_idx"].values

            # Ensure indices are within AnnData bounds
            valid_indices = cell_indices[cell_indices < adata.n_obs]
            if len(valid_indices) != len(cell_indices):
                if verbose:
                    print(
                        f"   [WARNING]  Warning: {len(cell_indices) - len(valid_indices)} cell indices out of AnnData bounds"
                    )

            # Create subset
            subset_adata = adata[valid_indices].copy()

            # Add selection criteria as metadata
            subset_adata.uns["selection_criteria"] = selection_criteria
            subset_adata.uns["selection_stats"] = {
                "n_selected": len(valid_indices),
                "n_total": n_cells,
                "selection_percentage": 100 * len(valid_indices) / n_cells,
            }

            if verbose:
                print(f"    Created AnnData subset: {subset_adata.n_obs} cells × {subset_adata.n_vars} genes")
                print("    Added selection criteria to .uns['selection_criteria']")

            return selected_df, subset_adata
        else:
            if verbose:
                print("   [WARNING]  Cannot create AnnData subset: no cells selected")
            return selected_df, None

    return selected_df


# DEPRECATED: archetypal_data_attributes removed - use independent functions instead
# Use get_all_archetypal_coordinates() and compute_archetype_distances() for clean DataFrame outputs


def get_all_archetypal_coordinates(model, dataloader, device: str = "cpu", verbose: bool = True) -> pd.DataFrame:
    """Extract archetypal coordinates for entire dataset as DataFrame.

    .. deprecated::
        Use :func:`extract_and_store_archetypal_coordinates` for AnnData integration.
        This function returns a DataFrame without storing in AnnData.

    Parameters
    ----------
    model : torch.nn.Module
        Trained Deep_AA model.
    dataloader : torch.utils.data.DataLoader
        DataLoader containing all data.
    device : str, default: 'cpu'
        Computing device.
    verbose : bool, default: True
        Whether to print progress.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:

        - ``cell_idx`` : int - Cell index (0 to n_cells-1)
        - ``archetype_0_weight``, ``archetype_1_weight``, ... : float
            A matrix weights (barycentric coordinates, sum to 1).
        - ``archetype_0_latent``, ``archetype_1_latent``, ... : float
            z latent variables.
        - ``archetype_0_mu``, ``archetype_1_mu``, ... : float
            Encoder means.
        - ``archetype_0_log_var``, ``archetype_1_log_var``, ... : float
            Encoder log variances.
        - ``max_weight`` : float - Maximum weight across archetypes.
        - ``dominant_archetype`` : int - Index of archetype with highest weight.
        - ``weight_entropy`` : float - Entropy of weight distribution.

    See Also
    --------
    extract_and_store_archetypal_coordinates : Recommended replacement
    """
    import warnings

    warnings.warn(
        "get_all_archetypal_coordinates is deprecated. Use extract_and_store_archetypal_coordinates() for AnnData integration.",
        DeprecationWarning,
        stacklevel=2,
    )
    model.eval()

    if verbose:
        print(" Extracting archetypal coordinates for entire dataset...")

    # Collect all coordinates
    all_A = []
    all_z = []
    all_mu = []
    all_log_var = []
    total_samples = 0
    n_archetypes = model.n_archetypes

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                data_batch = batch[0].to(device)
            else:
                data_batch = batch.to(device)

            # Get coordinates for this batch
            coords = get_archetypal_coordinates(model, data_batch, device)

            # Collect results
            all_A.append(coords["A"].cpu())
            all_z.append(coords["z"].cpu())
            all_mu.append(coords["mu"].cpu())
            all_log_var.append(coords["log_var"].cpu())

            total_samples += data_batch.shape[0]

            if verbose and (batch_idx + 1) % 10 == 0:
                print(f"   Processed {batch_idx + 1} batches, {total_samples} samples...")

    # Concatenate all results
    A_matrix = torch.cat(all_A, dim=0)  # [n_samples, n_archetypes]
    z_matrix = torch.cat(all_z, dim=0)  # [n_samples, n_archetypes]
    mu_matrix = torch.cat(all_mu, dim=0)  # [n_samples, n_archetypes]
    log_var_matrix = torch.cat(all_log_var, dim=0)  # [n_samples, n_archetypes]

    if verbose:
        print(f"[OK] Extracted coordinates for {total_samples} cells")
        print(f"   A matrix shape: {A_matrix.shape}")

    # Create DataFrame with clear column names
    data_dict = {"cell_idx": list(range(total_samples))}

    # Add archetype weights (A matrix)
    for arch_idx in range(n_archetypes):
        data_dict[f"archetype_{arch_idx}_weight"] = A_matrix[:, arch_idx].numpy()

    # Add latent variables (z matrix)
    for arch_idx in range(n_archetypes):
        data_dict[f"archetype_{arch_idx}_latent"] = z_matrix[:, arch_idx].numpy()

    # Add encoder outputs (optional, but useful for analysis)
    for arch_idx in range(n_archetypes):
        data_dict[f"archetype_{arch_idx}_mu"] = mu_matrix[:, arch_idx].numpy()
        data_dict[f"archetype_{arch_idx}_log_var"] = log_var_matrix[:, arch_idx].numpy()

    df = pd.DataFrame(data_dict)

    # Add summary statistics
    df["max_weight"] = A_matrix.max(dim=1)[0].numpy()
    df["dominant_archetype"] = A_matrix.argmax(dim=1).numpy()
    df["weight_entropy"] = -torch.sum(A_matrix * torch.log(A_matrix + 1e-8), dim=1).numpy()

    if verbose:
        print(f"[OK] Created coordinates DataFrame: {df.shape}")
        print(f"   Columns: {list(df.columns)[:5]}... (showing first 5)")
        print("   Dominant archetype distribution:")
        for arch_idx in range(n_archetypes):
            count = (df["dominant_archetype"] == arch_idx).sum()
            print(f"      Archetype {arch_idx}: {count} cells ({100 * count / len(df):.1f}%)")

    return df


def compute_archetype_distances(
    adata,
    pca_key: str = "X_pca",
    archetype_coords_key: str = "archetype_coordinates",
    obsm_key: str = "archetype_distances",
    uns_prefix: str = "archetype",
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute Euclidean distances between cells and archetypes in PCA space.

    This function computes the distance from each cell to each archetype,
    storing the distance matrix in ``adata.obsm`` for downstream analysis
    like archetype assignment and trajectory inference.

    Parameters
    ----------
    adata : AnnData
        AnnData object with PCA coordinates and archetype coordinates.
        Must have:

        - ``adata.obsm[pca_key]`` : Cell PCA coordinates
        - ``adata.uns[archetype_coords_key]`` : Archetype positions

    pca_key : str, default: 'X_pca'
        Key for PCA coordinates in ``adata.obsm``.
        Auto-detects: 'X_pca', 'X_PCA', 'PCA'.
    archetype_coords_key : str, default: 'archetype_coordinates'
        Key for archetype coordinates in ``adata.uns``.
    obsm_key : str, default: 'archetype_distances'
        Key for storing distance matrix in ``adata.obsm``.
    uns_prefix : str, default: 'archetype'
        Prefix for metadata keys in ``adata.uns``.
    verbose : bool, default: True
        Whether to print progress messages.

    Returns
    -------
    pd.DataFrame
        DataFrame with distance information (for backward compatibility):

        - ``cell_id`` : str
            Cell identifier from ``adata.obs.index``.
        - ``cell_idx`` : int
            0-based position index.
        - ``archetype_0_distance``, ``archetype_1_distance``, ... : float
            Distance to each archetype.
        - ``nearest_archetype`` : int
            Index of the nearest archetype.
        - ``nearest_distance`` : float
            Distance to the nearest archetype.
        - ``mean_distance`` : float
            Mean distance across all archetypes.
        - ``std_distance`` : float
            Standard deviation of distances.

    Stores
    ------
    The function stores the following in AnnData:

    - ``adata.obsm[obsm_key]`` : np.ndarray
        Distance matrix, shape (n_cells, n_archetypes).
        Default key: 'archetype_distances'.
    - ``adata.uns[uns_prefix + '_positions']`` : np.ndarray
        Copy of archetype positions.
    - ``adata.uns[uns_prefix + '_distance_info']`` : dict
        Computation metadata with keys:

        - ``n_archetypes`` : int
        - ``pca_key_used`` : str
        - ``archetype_coords_key`` : str
        - ``distance_metric`` : str ('euclidean')
        - ``distance_space`` : str ('PCA')
        - ``obsm_key`` : str

    Raises
    ------
    ValueError
        If PCA coordinates not found in ``adata.obsm``.
        If archetype coordinates not found in ``adata.uns``.

    Notes
    -----
    If the archetype coordinates have fewer dimensions than PCA coordinates,
    only the first N PCA dimensions are used for distance computation
    (where N = number of archetype dimensions).

    The distance matrix in ``adata.obsm`` is the primary output for
    downstream functions like :func:`bin_cells_by_archetype`.

    Examples
    --------
    >>> import peach as pc
    >>> # After training (which stores archetype_coordinates)
    >>> results = pc.tl.train_archetypal(adata, n_archetypes=5)
    >>> # Compute distances
    >>> distances_df = compute_archetype_distances(adata)
    >>> # Access distance matrix from AnnData (preferred)
    >>> distance_matrix = adata.obsm["archetype_distances"]
    >>> print(f"Shape: {distance_matrix.shape}")  # (n_cells, n_archetypes)
    >>> # Find cells closest to archetype 0
    >>> closest_to_arch0 = np.argsort(distance_matrix[:, 0])[:100]
    >>> # Or use the DataFrame
    >>> nearest_counts = distances_df["nearest_archetype"].value_counts()
    >>> print(nearest_counts)

    See Also
    --------
    bin_cells_by_archetype : Assign cells to archetypes based on distances
    peach.tl.archetypal_coordinates : User-facing wrapper
    peach._core.types.DistanceInfo : Type definition for metadata
    """
    if verbose:
        print(" Computing archetype distances in PCA space...")
        print(f"   Canonical reference: adata.obs.index ({len(adata.obs)} cells)")

    # Check for PCA coordinates in adata.obsm
    pca_coords = None
    for possible_key in [pca_key, "X_pca", "X_PCA", "PCA"]:
        if possible_key in adata.obsm:
            pca_coords = adata.obsm[possible_key]
            actual_pca_key = possible_key
            break

    if pca_coords is None:
        raise ValueError(
            f"No PCA coordinates found in adata.obsm. Expected keys: {pca_key}, 'X_pca', 'X_PCA', 'PCA'. "
            f"Available keys: {list(adata.obsm.keys())}"
        )

    # Check for archetype coordinates in adata.uns
    if archetype_coords_key not in adata.uns:
        raise ValueError(
            f"Archetype coordinates not found in adata.uns['{archetype_coords_key}']. "
            f"Run get_archetypal_coordinates() first and store results in adata.uns."
        )

    archetype_coords = adata.uns[archetype_coords_key]

    # Convert torch.Tensor to numpy if needed
    if hasattr(archetype_coords, "detach"):
        archetype_coords = archetype_coords.detach().cpu().numpy()

    if verbose:
        print(f"   Found PCA coordinates: {actual_pca_key} {pca_coords.shape}")
        print(f"   Found archetype coordinates: {archetype_coords_key} {archetype_coords.shape}")
        print(" Computing pairwise distances in PCA space...")

    # Ensure dimensional consistency for distance computation
    archetype_dims = archetype_coords.shape[1]  # Number of dimensions in archetype coords
    if pca_coords.shape[1] > archetype_dims:
        # Use only the first N dimensions that match the archetype dimensionality
        pca_coords_subset = pca_coords[:, :archetype_dims]
        if verbose:
            print(f"   Using first {archetype_dims} PCA components for distance computation")
            print(f"   PCA coordinates: {pca_coords.shape} → {pca_coords_subset.shape}")
    else:
        pca_coords_subset = pca_coords

    # Compute distance matrix using scipy
    from scipy.spatial.distance import cdist

    distance_matrix = cdist(pca_coords_subset, archetype_coords, metric="euclidean")
    # distance_matrix shape: [n_cells, n_archetypes]

    n_archetypes = archetype_coords.shape[0]

    if verbose:
        print("[OK] Distance computation complete")
        print(f"   Distance matrix shape: {distance_matrix.shape}")

    # STORE IN ANNDATA.OBSM (PRIMARY STORAGE)
    # ===================================
    adata.obsm[obsm_key] = distance_matrix  # [n_cells, n_archetypes]

    # Store archetype positions in adata.uns (already there, but update reference)
    adata.uns[f"{uns_prefix}_positions"] = archetype_coords

    # Store distance computation metadata
    adata.uns[f"{uns_prefix}_distance_info"] = {
        "n_archetypes": n_archetypes,
        "pca_key_used": actual_pca_key,
        "archetype_coords_key": archetype_coords_key,
        "distance_metric": "euclidean",
        "distance_space": "PCA",
        "obsm_key": obsm_key,
    }

    if verbose:
        print("[OK] Stored in AnnData:")
        print(f"   adata.obsm['{obsm_key}']: {distance_matrix.shape} distance matrix")
        print(f"   adata.uns['{uns_prefix}_positions']: {archetype_coords.shape} archetype positions")
        print(f"   adata.uns['{uns_prefix}_distance_info']: distance computation metadata")

    # CREATE DATAFRAME FOR BACKWARD COMPATIBILITY
    # ==========================================
    data_dict = {}

    # Use adata.obs.index as the canonical cell reference
    data_dict["cell_id"] = adata.obs.index.tolist()
    data_dict["cell_idx"] = list(range(len(adata.obs)))  # 0-based position indices

    # Add individual distances to each archetype
    for arch_idx in range(n_archetypes):
        data_dict[f"archetype_{arch_idx}_distance"] = distance_matrix[:, arch_idx]

    # Add summary statistics
    nearest_archetypes = np.argmin(distance_matrix, axis=1)
    nearest_distances = np.min(distance_matrix, axis=1)

    data_dict["nearest_archetype"] = nearest_archetypes
    data_dict["nearest_distance"] = nearest_distances
    data_dict["mean_distance"] = np.mean(distance_matrix, axis=1)
    data_dict["std_distance"] = np.std(distance_matrix, axis=1)

    # Create DataFrame with adata.obs.index alignment
    df = pd.DataFrame(data_dict)
    # Keep cell_id as a regular column instead of index to avoid _index issues
    # df.set_index('cell_id', inplace=True)  # Use canonical cell IDs as index

    if verbose:
        print("\n[STATS] Distance Statistics:")
        print("   Nearest archetype distribution:")
        for arch_idx in range(n_archetypes):
            count = (df["nearest_archetype"] == arch_idx).sum()
            mean_dist = df[df["nearest_archetype"] == arch_idx]["nearest_distance"].mean()
            print(
                f"      Archetype {arch_idx}: {count} cells ({100 * count / len(df):.1f}%), mean distance: {mean_dist:.4f}"
            )

        print("   Overall statistics:")
        print(f"      Mean nearest distance: {df['nearest_distance'].mean():.4f}")
        print(f"      Distance range: [{distance_matrix.min():.3f}, {distance_matrix.max():.3f}]")

    return df


def extract_and_store_archetypal_coordinates(
    model,
    dataloader,
    adata,
    pca_key: str = "X_pca",
    coords_key: str = "archetype_coordinates",
    cell_coords_key: str = "cell_archetype_weights",
    device: str = "cpu",
    verbose: bool = True,
) -> dict[str, Any]:
    """Extract archetypal coordinates for full dataset and store in AnnData.

    This is the primary function for coordinate extraction. It processes the
    entire dataset, extracts all archetypal information, and stores results
    in the appropriate AnnData locations.

    Replaces the deprecated functions:

    - ``get_archetypal_coordinates`` (single batch)
    - ``get_all_archetypal_coordinates`` (full dataset, no AnnData storage)
    - ``get_archetype_positions`` (Y matrix only)

    Parameters
    ----------
    model : torch.nn.Module
        Trained Deep_AA model.
    dataloader : torch.utils.data.DataLoader
        DataLoader for the full dataset.
    adata : AnnData
        AnnData object to store coordinates in.
    pca_key : str, default: 'X_pca'
        Key for PCA coordinates in ``adata.obsm``.
        Auto-detects: 'X_pca', 'X_PCA', 'PCA'.
    coords_key : str, default: 'archetype_coordinates'
        Key for storing archetype positions in ``adata.uns``.
    cell_coords_key : str, default: 'cell_archetype_weights'
        Key for storing cell weights in ``adata.obsm``.
    device : str, default: 'cpu'
        Computing device ('cpu', 'cuda', 'mps').
    verbose : bool, default: True
        Whether to print progress messages.

    Returns
    -------
    dict
        Dictionary containing extracted coordinates:

        - ``archetype_positions`` : np.ndarray
            Archetype positions in PCA space, shape (n_archetypes, n_pca_components).
        - ``cell_weights`` : np.ndarray
            A matrix weights (barycentric coords), shape (n_cells, n_archetypes).
            Each row sums to 1.
        - ``cell_latent`` : np.ndarray
            z latent variables, shape (n_cells, n_archetypes).
        - ``cell_mu`` : np.ndarray
            Encoder means, shape (n_cells, n_archetypes).
        - ``cell_log_var`` : np.ndarray
            Encoder log variances, shape (n_cells, n_archetypes).
        - ``n_cells`` : int
            Number of cells processed.
        - ``n_archetypes`` : int
            Number of archetypes in the model.
        - ``pca_key_used`` : str
            Actual PCA key used (may differ from input if auto-detected).

    Stores
    ------
    The function stores the following in AnnData:

    - ``adata.uns[coords_key]`` : np.ndarray
        Archetype positions, shape (n_archetypes, n_pca_components).
        Default key: 'archetype_coordinates'.
    - ``adata.obsm[cell_coords_key]`` : np.ndarray
        Cell weights (A matrix), shape (n_cells, n_archetypes).
        Default key: 'cell_archetype_weights'.
    - ``adata.obsm[cell_coords_key + '_latent']`` : np.ndarray
        z latent variables, shape (n_cells, n_archetypes).
    - ``adata.obsm[cell_coords_key + '_mu']`` : np.ndarray
        Encoder means, shape (n_cells, n_archetypes).
    - ``adata.obsm[cell_coords_key + '_log_var']`` : np.ndarray
        Encoder log variances, shape (n_cells, n_archetypes).

    Raises
    ------
    ValueError
        If no PCA coordinates found in ``adata.obsm``.

    Notes
    -----
    If PCA coordinates are not present, this function will compute them
    using ``scanpy.tl.pca(adata, n_comps=50)``.

    The archetype positions are projected to PCA space using the stored
    PCA components in ``adata.varm['PCs']`` if available.

    Examples
    --------
    >>> import peach as pc
    >>> # After training
    >>> results = pc.tl.train_archetypal(adata, n_archetypes=5)
    >>> model = results["final_model"]
    >>> # Create dataloader
    >>> from peach.pp import prepare_training
    >>> dataloader = prepare_training(adata)
    >>> # Extract and store all coordinates
    >>> coords = extract_and_store_archetypal_coordinates(model, dataloader, adata, device="cuda")
    >>> # Access from return value
    >>> print(f"Processed {coords['n_cells']} cells")
    >>> print(f"Cell weights shape: {coords['cell_weights'].shape}")
    >>> # Or access from AnnData (preferred for downstream analysis)
    >>> archetype_positions = adata.uns["archetype_coordinates"]
    >>> cell_weights = adata.obsm["cell_archetype_weights"]

    See Also
    --------
    compute_archetype_distances : Compute distances after extracting coordinates
    peach._core.types.ExtractedCoordinates : Type definition for return structure
    peach._core.types.AnnDataKeys : Reference for AnnData storage locations
    """
    if verbose:
        print(" Extracting and storing archetypal coordinates for all cells...")

    model.eval()

    # Extract coordinates for entire dataset
    all_A = []
    all_z = []
    all_mu = []
    all_log_var = []
    total_samples = 0
    n_archetypes = model.n_archetypes
    archetype_positions = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                data_batch = batch[0].to(device)
            else:
                data_batch = batch.to(device)

            # Get coordinates for this batch
            coords = get_archetypal_coordinates(model, data_batch, device)

            # Store archetype positions (same for all batches)
            if archetype_positions is None:
                archetype_positions = coords["Y"].detach().cpu().numpy()  # [n_archetypes, input_dim]

            # Collect cell coordinates
            all_A.append(coords["A"].cpu())
            all_z.append(coords["z"].cpu())
            all_mu.append(coords["mu"].cpu())
            all_log_var.append(coords["log_var"].cpu())

            total_samples += data_batch.shape[0]

            if verbose and (batch_idx + 1) % 10 == 0:
                print(f"   Processed {batch_idx + 1} batches, {total_samples} samples...")

    # Concatenate all results
    A_matrix = torch.cat(all_A, dim=0).numpy()  # [n_samples, n_archetypes]
    z_matrix = torch.cat(all_z, dim=0).numpy()  # [n_samples, n_archetypes]
    mu_matrix = torch.cat(all_mu, dim=0).numpy()  # [n_samples, n_archetypes]
    log_var_matrix = torch.cat(all_log_var, dim=0).numpy()  # [n_samples, n_archetypes]

    if verbose:
        print(f"[OK] Extracted coordinates for {total_samples} cells")
        print(f"   A matrix shape: {A_matrix.shape}")
        print(f"   Archetype positions shape: {archetype_positions.shape}")

    # Check if PCA coordinates exist
    pca_coords = None
    actual_pca_key = None
    for possible_key in [pca_key, "X_pca", "X_PCA", "PCA"]:
        if possible_key in adata.obsm:
            pca_coords = adata.obsm[possible_key]
            actual_pca_key = possible_key
            break

    if pca_coords is None:
        if verbose:
            print("   No PCA coordinates found, computing PCA...")
        # Compute PCA
        import scanpy as sc

        sc.tl.pca(adata, n_comps=min(50, adata.n_vars - 1))
        pca_coords = adata.obsm["X_pca"]
        actual_pca_key = "X_pca"
        if verbose:
            print(f"   [OK] PCA computed: {pca_coords.shape}")

    # Project archetype positions to PCA space
    # This requires the PCA transformation used for the cells
    if hasattr(adata, "varm") and "PCs" in adata.varm:
        # Use stored PCA components
        pca_components = adata.varm["PCs"]  # [n_features, n_components]
        pca_mean = adata.var["mean"] if "mean" in adata.var else adata.X.mean(axis=0)

        # Transform archetype positions to PCA space
        archetype_pca = (archetype_positions - pca_mean) @ pca_components
    else:
        # Fallback: assume archetype positions are already in the right space
        # This is a simplification - in practice, you'd want proper PCA transformation
        if verbose:
            print("   [WARNING]  Warning: No PCA components found, using archetype positions as-is")
        archetype_pca = archetype_positions[:, : pca_coords.shape[1]]  # Truncate to PCA dimensions

    # Store archetype positions in AnnData.uns
    adata.uns[coords_key] = archetype_pca

    # Store cell weights in AnnData.obsm with proper column names
    cell_weight_df = pd.DataFrame(A_matrix, columns=[f"archetype_{i}_weight" for i in range(n_archetypes)])
    cell_weight_df["cell_idx"] = range(total_samples)
    adata.obsm[cell_coords_key] = A_matrix

    # Also store latent variables
    adata.obsm[f"{cell_coords_key}_latent"] = z_matrix
    adata.obsm[f"{cell_coords_key}_mu"] = mu_matrix
    adata.obsm[f"{cell_coords_key}_log_var"] = log_var_matrix

    if verbose:
        print(f"   [OK] Stored archetype positions in adata.uns['{coords_key}']: {archetype_pca.shape}")
        print(f"   [OK] Stored cell weights in adata.obsm['{cell_coords_key}']: {A_matrix.shape}")
        print(f"   [OK] Stored latent variables in adata.obsm['{cell_coords_key}_latent']: {z_matrix.shape}")
        print(f"   PCA space: {actual_pca_key} {pca_coords.shape}")

    # Return comprehensive results
    return {
        "archetype_positions": archetype_pca,
        "cell_weights": A_matrix,
        "cell_latent": z_matrix,
        "cell_mu": mu_matrix,
        "cell_log_var": log_var_matrix,
        "n_cells": total_samples,
        "n_archetypes": n_archetypes,
        "pca_key_used": actual_pca_key,
    }


def get_archetype_positions(model, device: str = "cpu", verbose: bool = True) -> pd.DataFrame:
    """Extract archetype positions (Y matrix) as DataFrame.

    .. deprecated::
        Use :func:`extract_and_store_archetypal_coordinates` for AnnData integration.
        This function returns positions without AnnData context.

    Parameters
    ----------
    model : torch.nn.Module
        Trained Deep_AA model.
    device : str, default: 'cpu'
        Computing device (not used, archetypes are model parameters).
    verbose : bool, default: True
        Whether to print statistics.

    Returns
    -------
    pd.DataFrame
        DataFrame with archetype positions:

        - ``archetype_idx`` : int - Archetype index (0, 1, 2, ...).
        - ``feature_0``, ``feature_1``, ... : float - Position in each dimension.
        - ``mean`` : float - Mean across features.
        - ``std`` : float - Standard deviation across features.
        - ``min`` : float - Minimum value.
        - ``max`` : float - Maximum value.
        - ``range`` : float - max - min.

    See Also
    --------
    extract_and_store_archetypal_coordinates : Recommended replacement
    """
    import warnings

    warnings.warn(
        "get_archetype_positions is deprecated. Use extract_and_store_archetypal_coordinates() for AnnData integration.",
        DeprecationWarning,
        stacklevel=2,
    )

    model.eval()

    if verbose:
        print("[STATS] Extracting archetype positions...")

    # Get archetype positions from model
    archetype_positions = model.archetypes.detach().cpu().numpy()  # [n_archetypes, input_dim]
    n_archetypes, n_features = archetype_positions.shape

    # Create DataFrame with clear structure
    feature_columns = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(archetype_positions, columns=feature_columns)
    # Add archetype_idx as a proper column to avoid _index issues with AnnData
    df["archetype_idx"] = range(n_archetypes)
    # Reorder columns to put archetype_idx first
    df = df[["archetype_idx"] + feature_columns]

    # Add summary statistics for each archetype
    df["mean"] = archetype_positions.mean(axis=1)
    df["std"] = archetype_positions.std(axis=1)
    df["min"] = archetype_positions.min(axis=1)
    df["max"] = archetype_positions.max(axis=1)
    df["range"] = df["max"] - df["min"]

    if verbose:
        print(f"[OK] Extracted {n_archetypes} archetypes with {n_features} features")
        print("=" * 50)
        print("ARCHETYPE POSITIONS SUMMARY")
        print("=" * 50)

        for i in range(n_archetypes):
            row = df.iloc[i]
            print(f"\nArchetype {i}:")
            print(f"  First 5 features: {archetype_positions[i, :5]}")
            print(f"  Stats: mean={row['mean']:.3f}, std={row['std']:.3f}")
            print(f"  Range: [{row['min']:.3f}, {row['max']:.3f}] (span: {row['range']:.3f})")

        # Overall statistics
        print("\nOverall archetype statistics:")
        print(f"  Mean archetype span: {df['range'].mean():.3f}")
        print(f"  Data space span: [{archetype_positions.min():.3f}, {archetype_positions.max():.3f}]")

    return df


# =============================================================================
# CONDITIONAL CENTROIDS
# =============================================================================


def compute_conditional_centroids(
    adata,
    condition_column: str,
    pca_key: str = "X_pca",
    store_key: str = "conditional_centroids",
    exclude_archetypes: list | None = None,
    groupby: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Compute centroid positions in PCA space for each level of a categorical condition.

    Following R template patterns:
    - Uses ALL PCs for centroid calculation (equivalent to R's colMeans)
    - Stores full PC centroid but extracts first 3 for visualization
    - Excludes 'no_archetype' and 'archetype_0' cells by default

    Parameters
    ----------
    adata : AnnData
        Annotated data object with PCA coordinates in adata.obsm[pca_key].
    condition_column : str
        Name of categorical column in adata.obs to group by.
        Examples: 'treatment_phase', 'timepoint', 'batch'.
    pca_key : str, default: "X_pca"
        Key in adata.obsm containing PCA coordinates.
    store_key : str, default: "conditional_centroids"
        Key in adata.uns to store results.
    exclude_archetypes : list, optional
        Archetype labels to exclude from centroid calculation.
        Default: ['no_archetype', 'archetype_0'] (following R template).
        Set to empty list [] to include all cells.
    groupby : str, optional
        Second categorical column for multi-group trajectories.
        If provided, centroids are computed for each (group, level) combination.
        Example: groupby='response_group' to get separate trajectories per response.
    verbose : bool, default: True
        Whether to print progress messages.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``condition_column`` : str - name of the condition column
        - ``n_levels`` : int - number of unique levels
        - ``levels`` : List[str] - list of level names
        - ``centroids`` : Dict[str, List[float]] - level → full PCA coordinates
        - ``centroids_3d`` : Dict[str, List[float]] - level → [x, y, z] first 3 PCs
        - ``cell_counts`` : Dict[str, int] - level → cell count
        - ``pca_key`` : str - PCA key used
        - ``exclude_archetypes`` : List[str] - archetypes excluded
        - ``groupby`` : Optional[str] - groupby column if used
        - ``group_centroids`` : Optional[Dict] - if groupby: {group: {level: coords}}
        - ``group_centroids_3d`` : Optional[Dict] - if groupby: {group: {level: [x,y,z]}}
        - ``group_cell_counts`` : Optional[Dict] - if groupby: {group: {level: count}}

    Raises
    ------
    ValueError
        If condition_column not in adata.obs or PCA coordinates not found.

    Stores
    ------
    The function stores results in AnnData:

    - ``adata.uns[store_key][condition_column]`` : dict
        Full results dictionary as returned.

    Examples
    --------
    >>> # Simple centroid calculation
    >>> result = pc.tl.compute_conditional_centroids(adata, "treatment_phase")
    >>> print(result["centroids_3d"])
    {'chemo-naive': [1.2, 0.5, -0.3], 'IDS': [0.8, 1.1, 0.2]}

    >>> # Multi-group centroids for trajectory comparison
    >>> result = pc.tl.compute_conditional_centroids(adata, "treatment_phase", groupby="response_group")
    >>> for group, levels in result["group_centroids_3d"].items():
    ...     print(f"{group}: {levels}")

    See Also
    --------
    peach.pl.archetypal_space : Visualize with centroid trajectory overlay
    """
    # Set defaults
    if exclude_archetypes is None:
        exclude_archetypes = ["no_archetype", "archetype_0"]

    # Input validation - PCA coordinates
    pca_candidates = ["X_pca", "X_PCA", "PCA", "pca"]
    actual_pca_key = None
    for candidate in pca_candidates:
        if candidate in adata.obsm:
            actual_pca_key = candidate
            break
    if pca_key in adata.obsm:
        actual_pca_key = pca_key
    if actual_pca_key is None:
        raise ValueError(
            f"PCA coordinates not found. Tried: {pca_candidates + [pca_key]}. Available keys: {list(adata.obsm.keys())}"
        )

    # Input validation - condition column
    if condition_column not in adata.obs.columns:
        raise ValueError(
            f"Column '{condition_column}' not found in adata.obs. Available columns: {list(adata.obs.columns)}"
        )

    # Input validation - groupby column
    if groupby is not None and groupby not in adata.obs.columns:
        raise ValueError(
            f"Groupby column '{groupby}' not found in adata.obs. Available columns: {list(adata.obs.columns)}"
        )

    if verbose:
        print(f"[CENTROIDS] Computing centroids for '{condition_column}'")
        print(f"   PCA key: {actual_pca_key}")
        if groupby:
            print(f"   Groupby: {groupby}")

    # Get PCA coordinates (ALL dimensions)
    pca_coords = adata.obsm[actual_pca_key]
    n_pcs = pca_coords.shape[1]

    # Build cell mask for exclusions
    valid_mask = np.ones(adata.n_obs, dtype=bool)
    if exclude_archetypes and "archetypes" in adata.obs.columns:
        for excl in exclude_archetypes:
            valid_mask &= adata.obs["archetypes"] != excl
        n_excluded = (~valid_mask).sum()
        if verbose:
            print(f"   Excluding {n_excluded} cells with archetypes: {exclude_archetypes}")

    # Get condition values for valid cells
    condition_values = adata.obs[condition_column]
    levels = list(condition_values[valid_mask].unique())

    if verbose:
        print(f"   Condition levels: {levels}")
        print(f"   Valid cells: {valid_mask.sum()} / {adata.n_obs}")

    # Compute centroids (without groupby)
    centroids = {}
    centroids_3d = {}
    cell_counts = {}

    for level in levels:
        level_mask = valid_mask & (condition_values == level)
        level_coords = pca_coords[level_mask]

        if len(level_coords) == 0:
            if verbose:
                print(f"   [WARNING] No cells for level '{level}' after exclusions")
            continue

        # Compute mean position across ALL PCs (R: colMeans)
        centroid = level_coords.mean(axis=0)
        centroids[str(level)] = centroid.tolist()
        centroids_3d[str(level)] = centroid[:3].tolist()  # First 3 for viz
        cell_counts[str(level)] = int(level_mask.sum())

        if verbose:
            print(
                f"   {level}: {cell_counts[str(level)]} cells, "
                f"centroid_3d = [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]"
            )

    # Compute group centroids if groupby specified
    group_centroids = None
    group_centroids_3d = None
    group_cell_counts = None

    if groupby is not None:
        group_centroids = {}
        group_centroids_3d = {}
        group_cell_counts = {}

        group_values = adata.obs[groupby]
        groups = list(group_values[valid_mask].unique())

        if verbose:
            print(f"\n   [GROUPBY] Computing centroids per {groupby}")
            print(f"   Groups: {groups}")

        for group in groups:
            group_centroids[str(group)] = {}
            group_centroids_3d[str(group)] = {}
            group_cell_counts[str(group)] = {}

            for level in levels:
                group_level_mask = valid_mask & (condition_values == level) & (group_values == group)
                group_level_coords = pca_coords[group_level_mask]

                if len(group_level_coords) == 0:
                    if verbose:
                        print(f"   [WARNING] No cells for {group}/{level}")
                    continue

                centroid = group_level_coords.mean(axis=0)
                group_centroids[str(group)][str(level)] = centroid.tolist()
                group_centroids_3d[str(group)][str(level)] = centroid[:3].tolist()
                group_cell_counts[str(group)][str(level)] = int(group_level_mask.sum())

                if verbose:
                    print(
                        f"   {group}/{level}: {group_cell_counts[str(group)][str(level)]} cells, "
                        f"centroid_3d = [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]"
                    )

    # Build result dictionary
    result = {
        "condition_column": condition_column,
        "n_levels": len(levels),
        "levels": [str(l) for l in levels],
        "centroids": centroids,
        "centroids_3d": centroids_3d,
        "cell_counts": cell_counts,
        "pca_key": actual_pca_key,
        "exclude_archetypes": exclude_archetypes,
        "groupby": groupby,
        "group_centroids": group_centroids,
        "group_centroids_3d": group_centroids_3d,
        "group_cell_counts": group_cell_counts,
    }

    # Store in adata.uns
    if store_key not in adata.uns:
        adata.uns[store_key] = {}
    adata.uns[store_key][condition_column] = result

    if verbose:
        print(f"\n[OK] Stored centroids in adata.uns['{store_key}']['{condition_column}']")

    return result


def assign_to_centroids(
    adata,
    condition_column: str,
    pca_key: str = "X_pca",
    centroid_key: str = "conditional_centroids",
    bin_prop: float = 0.15,
    obs_key: str = "centroid_assignments",
    exclude_archetypes: list | None = None,
    verbose: bool = True,
) -> None:
    """Assign cells to nearest centroid based on distance (top bin_prop% closest).

    This function mirrors assign_archetypes but for condition-based centroids.
    It enables using treatment phase centroids as trajectory endpoints in
    single_trajectory_analysis.

    Parameters
    ----------
    adata : AnnData
        Annotated data object. Must have:
        - PCA coordinates in adata.obsm[pca_key]
        - Centroids computed via compute_conditional_centroids in adata.uns[centroid_key]
    condition_column : str
        Name of the condition column used in compute_conditional_centroids.
        This identifies which centroid set to use.
    pca_key : str, default: "X_pca"
        Key in adata.obsm containing PCA coordinates.
    centroid_key : str, default: "conditional_centroids"
        Key in adata.uns containing centroid results from compute_conditional_centroids.
    bin_prop : float, default: 0.15
        Proportion of cells to assign to each centroid (top 15% closest).
        Similar to percentage_per_archetype in assign_archetypes.
    obs_key : str, default: "centroid_assignments"
        Key in adata.obs to store assignments.
    exclude_archetypes : list, optional
        Archetype labels to exclude from assignment.
        Default: ['no_archetype'] - these cells get 'unassigned'.
    verbose : bool, default: True
        Whether to print progress messages.

    Returns
    -------
    None
        Modifies adata.obs[obs_key] with Categorical assignments.
        Values are condition levels (e.g., 'chemo_naive', 'IDS') or 'unassigned'.

    Raises
    ------
    ValueError
        If centroids not found or PCA coordinates missing.

    Examples
    --------
    >>> # First compute centroids
    >>> pc.tl.compute_conditional_centroids(adata, "treatment_stage")
    >>>
    >>> # Then assign cells to nearest centroid
    >>> pc.tl.assign_to_centroids(adata, "treatment_stage", bin_prop=0.15)
    >>>
    >>> # Check assignments
    >>> print(adata.obs["centroid_assignments"].value_counts())

    See Also
    --------
    compute_conditional_centroids : Compute centroids for condition levels
    assign_archetypes : Similar function for archetype assignments
    single_trajectory_analysis : Uses centroid assignments for trajectory analysis
    """
    # Defaults
    if exclude_archetypes is None:
        exclude_archetypes = ["no_archetype"]

    # Validate centroid data exists
    if centroid_key not in adata.uns:
        raise ValueError(
            f"Centroid data not found in adata.uns['{centroid_key}']. Run compute_conditional_centroids() first."
        )

    if condition_column not in adata.uns[centroid_key]:
        raise ValueError(
            f"No centroids for condition '{condition_column}' in adata.uns['{centroid_key}']. "
            f"Available: {list(adata.uns[centroid_key].keys())}"
        )

    centroid_result = adata.uns[centroid_key][condition_column]
    centroids = centroid_result["centroids"]
    levels = centroid_result["levels"]

    if verbose:
        print(f"[ASSIGN] Assigning cells to centroids for '{condition_column}'")
        print(f"   Levels: {levels}")
        print(f"   Bin proportion: {bin_prop:.1%}")

    # Validate PCA coordinates
    pca_candidates = ["X_pca", "X_PCA", "PCA", "pca"]
    actual_pca_key = None
    for candidate in pca_candidates:
        if candidate in adata.obsm:
            actual_pca_key = candidate
            break
    if pca_key in adata.obsm:
        actual_pca_key = pca_key
    if actual_pca_key is None:
        raise ValueError(
            f"PCA coordinates not found. Tried: {pca_candidates + [pca_key]}. Available keys: {list(adata.obsm.keys())}"
        )

    pca_coords = adata.obsm[actual_pca_key]

    # Build exclusion mask
    valid_mask = np.ones(adata.n_obs, dtype=bool)
    if exclude_archetypes and "archetypes" in adata.obs.columns:
        for excl in exclude_archetypes:
            valid_mask &= adata.obs["archetypes"] != excl
        if verbose:
            n_excluded = (~valid_mask).sum()
            print(f"   Excluding {n_excluded} cells with archetypes: {exclude_archetypes}")

    # Compute distances to each centroid
    n_cells = adata.n_obs
    n_centroids = len(levels)

    # Distance matrix: cells x centroids
    distances = np.zeros((n_cells, n_centroids))

    for i, level in enumerate(levels):
        centroid = np.array(centroids[level])
        # Match dimensions (centroids may have fewer PCs than data)
        n_pcs = min(pca_coords.shape[1], len(centroid))
        cell_coords = pca_coords[:, :n_pcs]
        centroid_trimmed = centroid[:n_pcs]

        # Euclidean distance
        distances[:, i] = np.linalg.norm(cell_coords - centroid_trimmed, axis=1)

    # Initialize assignments as 'unassigned'
    assignments = np.array(["unassigned"] * n_cells, dtype=object)

    # Assign top bin_prop% closest cells to each centroid
    for i, level in enumerate(levels):
        level_distances = distances[:, i].copy()

        # Set excluded cells to infinity so they won't be selected
        level_distances[~valid_mask] = np.inf

        # Also exclude cells already assigned to another centroid
        already_assigned = assignments != "unassigned"
        level_distances[already_assigned] = np.inf

        # Find threshold for top bin_prop%
        valid_distances = level_distances[level_distances < np.inf]
        if len(valid_distances) == 0:
            if verbose:
                print(f"   [WARNING] No valid cells for level '{level}'")
            continue

        n_to_assign = max(1, int(len(valid_distances) * bin_prop))
        threshold = np.partition(valid_distances, n_to_assign - 1)[n_to_assign - 1]

        # Assign cells below threshold
        assign_mask = (level_distances <= threshold) & (assignments == "unassigned")
        assignments[assign_mask] = level

        if verbose:
            n_assigned = assign_mask.sum()
            print(f"   {level}: {n_assigned} cells assigned (threshold={threshold:.3f})")

    # Store as categorical
    all_categories = levels + ["unassigned"]
    adata.obs[obs_key] = pd.Categorical(assignments, categories=all_categories)

    if verbose:
        counts = adata.obs[obs_key].value_counts()
        print(f"\n[OK] Stored assignments in adata.obs['{obs_key}']")
        print(f"   Distribution: {counts.to_dict()}")

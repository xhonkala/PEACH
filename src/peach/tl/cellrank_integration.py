"""CellRank integration for archetypal trajectory analysis."""

from .._core.utils.cellrank_helpers import (
    compute_lineage_pseudotimes as _compute_lineage_pseudotimes,
)
from .._core.utils.cellrank_helpers import (
    setup_cellrank_workflow,
)


def setup_cellrank(
    adata,
    high_purity_threshold=0.80,
    n_neighbors=30,
    n_pcs=11,
    compute_paga=True,
    solver="gmres",
    tol=1e-6,
    terminal_obs_key="archetypes",
    verbose=True,
):
    """
    Set up CellRank workflow for archetypal or centroid-based trajectory analysis.

    This function orchestrates the complete pipeline from terminal state assignments
    to fate probabilities, including neighbors computation, UMAP, PAGA, and GPCCA.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Must contain:
        - `adata.obs[terminal_obs_key]` : Terminal state assignments
        - `adata.obsm['X_pca']` : PCA coordinates
        If using archetypes (default):
        - `adata.obsm['cell_archetype_weights']` : Barycentric weights
        - `adata.obsm['archetype_distances']` : Distances to archetypes
    high_purity_threshold : float, optional (default: 0.80)
        Percentile threshold for defining high-purity cells.
        0.80 means top 20% of cells per archetype.
        Only used when terminal_obs_key='archetypes'.
    n_neighbors : int, optional (default: 30)
        Number of neighbors for k-NN graph construction
    n_pcs : int, optional (default: 11)
        Number of principal components to use
    compute_paga : bool, optional (default: True)
        Whether to compute PAGA connectivity
    solver : str, optional (default: 'gmres')
        Solver for fate probability computation ('gmres', 'direct', etc.)
    tol : float, optional (default: 1e-6)
        Tolerance for iterative solver
    terminal_obs_key : str, optional (default: 'archetypes')
        Key in adata.obs containing terminal state assignments.
        Use 'archetypes' for standard archetype-based analysis or
        'centroid_assignments' for treatment phase centroid trajectories
        (requires running pc.tl.assign_to_centroids() first).
    verbose : bool, optional (default: True)
        Print progress messages

    Returns
    -------
    ck : cellrank.kernels.ConnectivityKernel
        Computed transition kernel
    g : cellrank.estimators.GPCCA
        GPCCA estimator with fate probabilities

    Stores in adata
    ---------------
    adata.obs['terminal_states'] : pd.Series
        Terminal state assignments for high-purity cells
    adata.obsm['fate_probabilities'] : np.ndarray
        Fate probability matrix (n_obs × n_lineages)
    adata.uns['lineage_names'] : list
        List of lineage names (archetype or centroid names)
    adata.uns['cellrank_gpcca'] : GPCCA
        GPCCA estimator object for downstream functions
    adata.obsm['X_umap'] : np.ndarray
        UMAP coordinates (if not already present)
    adata.uns['neighbors'] : dict
        k-NN graph (if not already present)
    adata.uns['paga'] : dict
        PAGA results (if compute_paga=True)

    Examples
    --------
    Basic archetype-based analysis:

    >>> import peach as pc
    >>> ck, g = pc.tl.setup_cellrank(adata)

    Treatment phase centroid-based analysis:

    >>> # First compute centroids and assign cells
    >>> pc.tl.compute_conditional_centroids(adata, condition_column="treatment_phase")
    >>> pc.tl.assign_to_centroids(adata, condition_column="treatment_phase")
    >>> # Then run CellRank with centroid assignments
    >>> ck, g = pc.tl.setup_cellrank(adata, terminal_obs_key="centroid_assignments")

    Access results:

    >>> fate_probs = adata.obsm["fate_probabilities"]
    >>> lineages = adata.uns["lineage_names"]
    >>> terminal_states = adata.obs["terminal_states"]

    Customize parameters:

    >>> ck, g = pc.tl.setup_cellrank(
    ...     adata,
    ...     high_purity_threshold=0.90,  # Top 10% of cells
    ...     n_neighbors=50,
    ...     compute_paga=False,
    ... )

    Notes
    -----
    - Requires CellRank installation: `pip install cellrank`
    - For GAMR models, set R_HOME before importing cellrank:
      ```python
      import os

      os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources"
      ```

    See Also
    --------
    assign_to_centroids : Assign cells to treatment phase centroids
    compute_lineage_pseudotimes : Convert fate probabilities to pseudotime
    compute_lineage_drivers : Identify genes driving lineage commitment

    References
    ----------
    .. [1] Lange et al. (2022) "CellRank for directed single-cell fate mapping"
           Nature Methods. https://doi.org/10.1038/s41592-021-01346-6
    """
    ck, g = setup_cellrank_workflow(
        adata,
        high_purity_threshold=high_purity_threshold,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        compute_paga=compute_paga,
        kernel_type="connectivity",
        solver=solver,
        tol=tol,
        terminal_obs_key=terminal_obs_key,
        verbose=verbose,
    )

    # Store GPCCA object for downstream functions (compute_transition_frequencies, etc.)
    adata.uns["cellrank_gpcca"] = g

    return ck, g


def compute_lineage_pseudotimes(adata, lineage_names=None, fate_prob_key="fate_probabilities"):
    """
    Convert fate probabilities to lineage-specific pseudotimes.

    Creates continuous pseudotime variables for each lineage by using fate
    probabilities as progression measures. Stores results in adata.obs.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Must contain:
        - `adata.obsm['fate_probabilities']` : Fate probability matrix
        - `adata.uns['lineage_names']` : List of lineage names
    lineage_names : list of str, optional
        Specific lineages to compute pseudotime for.
        If None, computes for all lineages in adata.uns['lineage_names']
    fate_prob_key : str, optional (default: 'fate_probabilities')
        Key in adata.obsm containing fate probabilities

    Returns
    -------
    None
        Stores pseudotime variables in adata.obs with keys:
        'pseudotime_to_{lineage}' for each lineage

    Examples
    --------
    Compute pseudotimes for all lineages:

    >>> import peach as pc
    >>> pc.tl.compute_lineage_pseudotimes(adata)

    Access specific pseudotime:

    >>> pseudotime = adata.obs["pseudotime_to_archetype_5"]

    Compute for specific lineages:

    >>> pc.tl.compute_lineage_pseudotimes(adata, lineage_names=["archetype_3", "archetype_5"])

    Use for gene trend analysis:

    >>> import cellrank as cr
    >>> cr.pl.gene_trends(
    ...     adata, model=cr.models.GAMR(adata), genes=["RARRES1", "SOD2"], time_key="pseudotime_to_archetype_5"
    ... )

    Notes
    -----
    - Must run `setup_cellrank()` first to compute fate probabilities
    - Pseudotime values are simply the fate probabilities (range: 0-1)
    - Higher pseudotime = higher probability of committing to that lineage

    See Also
    --------
    setup_cellrank : Compute fate probabilities
    compute_lineage_drivers : Identify genes driving lineage commitment
    """
    return _compute_lineage_pseudotimes(adata, lineage_names=lineage_names, fate_prob_key=fate_prob_key)


def compute_lineage_drivers(adata, lineage, n_genes=100, method="cellrank", **kwargs):
    """
    Identify genes driving commitment to a specific lineage.

    Computes correlation between gene expression and fate probabilities to
    identify lineage-specific marker genes.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with fate probabilities computed
    lineage : str
        Target lineage name (e.g., 'archetype_5')
    n_genes : int, optional (default: 100)
        Number of top genes to return
    method : str, optional (default: 'cellrank')
        Method for computing drivers:
        - 'cellrank' : Use CellRank's compute_lineage_drivers (requires GPCCA object)
        - 'correlation' : Simple Spearman correlation (faster, works without GPCCA)
    **kwargs
        Additional arguments passed to method

    Returns
    -------
    drivers : pd.DataFrame
        Top driver genes with statistics:
        - 'gene' : Gene name
        - 'lineage' : Target lineage name
        - 'correlation' : Spearman correlation with fate probability
        - 'pvalue' : P-value from correlation test

    Examples
    --------
    Using CellRank method (GPCCA is automatically stored by setup_cellrank):

    >>> import peach as pc
    >>> ck, g = pc.tl.setup_cellrank(adata)
    >>> drivers = pc.tl.compute_lineage_drivers(adata, lineage="archetype_5", method="cellrank")

    Using correlation method (simpler, faster):

    >>> drivers = pc.tl.compute_lineage_drivers(adata, lineage="archetype_5", method="correlation", n_genes=50)

    Top genes:

    >>> print(drivers.head(10))

    Notes
    -----
    - 'cellrank' method is more sophisticated (uses GAM models)
    - 'correlation' method is faster and works without storing GPCCA object
    - For publication, recommend 'cellrank' method with GAMR models

    See Also
    --------
    setup_cellrank : Compute fate probabilities
    compute_lineage_pseudotimes : Create pseudotime variables
    """
    if method == "cellrank":
        # Use CellRank's method (requires GPCCA object)
        if "cellrank_gpcca" not in adata.uns:
            raise ValueError(
                "GPCCA object not found in adata.uns['cellrank_gpcca']. "
                "Store it after running setup_cellrank: adata.uns['cellrank_gpcca'] = g"
            )

        g = adata.uns["cellrank_gpcca"]
        drivers = g.compute_lineage_drivers(lineages=lineage, **kwargs)

        return drivers.head(n_genes)

    elif method == "correlation":
        # Simple correlation-based approach
        import pandas as pd
        from scipy.stats import spearmanr

        if "lineage_names" not in adata.uns:
            raise ValueError("Run setup_cellrank() first to compute lineages")

        if lineage not in adata.uns["lineage_names"]:
            raise ValueError(f"Lineage '{lineage}' not found in {adata.uns['lineage_names']}")

        # Get fate probability for target lineage
        lineage_idx = adata.uns["lineage_names"].index(lineage)
        fate_prob = adata.obsm["fate_probabilities"][:, lineage_idx]

        # Compute correlation for all genes
        results = []
        for gene in adata.var_names:
            expr = adata[:, gene].X.toarray().flatten() if hasattr(adata.X, "toarray") else adata[:, gene].X.flatten()
            corr, pval = spearmanr(fate_prob, expr)
            results.append({
                "gene": gene,
                "lineage": lineage,
                "correlation": corr,
                "pvalue": pval,
            })

        # Create DataFrame and sort by correlation (descending)
        drivers = pd.DataFrame(results)
        drivers = drivers.sort_values("correlation", ascending=False)

        return drivers.head(n_genes).reset_index(drop=True)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'cellrank' or 'correlation'")


def compute_transition_frequencies(adata, start_weight_threshold=0.5, fate_prob_threshold=0.3, lineages=None):
    """
    Compute frequency of transitions between archetypal states.

    Identifies cells transitioning from one archetype to another based on their
    starting archetypal weights and fate probabilities from CellRank.

    A transition is counted when a cell has:
    - High barycentric weight for source archetype (> start_weight_threshold)
    - High fate probability for target archetype (> fate_prob_threshold)

    Parameters
    ----------
    adata : AnnData
        Annotated data object with CellRank results.
        Must contain:
        - `adata.obsm['cell_archetype_weights']`: Barycentric weights [n_obs, n_archetypes]
        - `adata.obs['archetypes']`: Categorical archetype assignments
        - `adata.obsm['fate_probabilities']`: Fate probability matrix [n_obs, n_lineages]
        - `adata.uns['lineage_names']`: List of lineage/archetype names
        - `adata.uns['cellrank_gpcca']`: GPCCA estimator object (from setup_cellrank)

    start_weight_threshold : float, default=0.5
        Minimum barycentric weight to consider a cell as "starting" from an archetype.
        - 0.5 = top 50% cells per archetype (balanced)
        - 0.7 = top 30% cells (more stringent)
        - 0.3 = top 70% cells (more permissive)

    fate_prob_threshold : float, default=0.3
        Minimum fate probability to consider a cell as "transitioning to" an archetype.
        - 0.3 = 30% commitment probability (standard)
        - 0.5 = 50% commitment (stringent)
        - 0.2 = 20% commitment (permissive)

    lineages : list of str, optional
        Specific lineages/archetypes to analyze. If None, uses all lineages from
        `adata.uns['lineage_names']` that start with 'archetype_'.

    Returns
    -------
    pd.DataFrame
        Transition frequency matrix with shape [n_archetypes, n_archetypes].
        - Index: Source archetypes (starting weight)
        - Columns: Target archetypes (fate probability)
        - Values: Integer counts of cells satisfying both thresholds
        - Diagonal: Cells maintaining their archetype identity
        - Off-diagonal: Cross-archetype transitions

        Example:
                        archetype_0  archetype_1  archetype_2  archetype_3
        archetype_0            150           45           23           12
        archetype_1             12          200           67            8
        archetype_2              8           34          180           45
        archetype_3              5           15           30          190

    Raises
    ------
    ValueError
        If required CellRank results are missing (run `setup_cellrank()` first)

    Notes
    -----
    - **archetype_0 (centroid)** uses categorical assignment instead of weight threshold
    - Returns raw counts (not normalized probabilities)
    - Cells can appear in multiple transitions if they meet multiple criteria
    - Use with PAGA connectivity for complete trajectory analysis

    Examples
    --------
    Basic usage with default thresholds:

    >>> import peach as pc
    >>> # After running setup_cellrank()
    >>> transitions = pc.tl.compute_transition_frequencies(adata)
    >>> print(transitions)

    Stringent thresholds for high-confidence transitions:

    >>> transitions = pc.tl.compute_transition_frequencies(
    ...     adata,
    ...     start_weight_threshold=0.7,  # Top 30% cells
    ...     fate_prob_threshold=0.5,  # 50% commitment
    ... )

    Analyze specific archetypes only:

    >>> transitions = pc.tl.compute_transition_frequencies(
    ...     adata, lineages=["archetype_1", "archetype_2", "archetype_3"]
    ... )

    Visualize with seaborn:

    >>> import seaborn as sns
    >>> import matplotlib.pyplot as plt
    >>> sns.heatmap(transitions, annot=True, fmt="d", cmap="YlOrRd")
    >>> plt.title("Archetype Transition Frequencies")
    >>> plt.show()

    See Also
    --------
    setup_cellrank : Complete CellRank workflow setup
    compute_lineage_pseudotimes : Convert fate probabilities to pseudotime
    compute_lineage_drivers : Identify driver genes for lineage commitment
    """
    from .._core.utils.cellrank_helpers import compute_transition_frequencies as _compute_transition_frequencies

    # Validate required data
    if "cellrank_gpcca" not in adata.uns:
        raise ValueError("CellRank results not found. Run pc.tl.setup_cellrank() first.")

    if "lineage_names" not in adata.uns:
        raise ValueError("Lineage names not found in adata.uns['lineage_names']. Run pc.tl.setup_cellrank() first.")

    if "fate_probabilities" not in adata.obsm:
        raise ValueError(
            "Fate probabilities not found in adata.obsm['fate_probabilities']. Run pc.tl.setup_cellrank() first."
        )

    # Get GPCCA object and fate probabilities
    g = adata.uns["cellrank_gpcca"]
    fate_probs = g.fate_probabilities

    # Determine which lineages to analyze
    if lineages is None:
        # Default: all archetypes (exclude 'no_archetype' if present)
        lineages = [name for name in adata.uns["lineage_names"] if name.startswith("archetype_")]

    # Call core implementation
    transition_df = _compute_transition_frequencies(
        adata=adata,
        fate_probs=fate_probs,
        archetype_labels=lineages,
        start_weight_threshold=start_weight_threshold,
        fate_prob_threshold=fate_prob_threshold,
    )

    return transition_df


def single_trajectory_analysis(
    adata,
    trajectory: tuple,
    trajectories: list = None,
    selection_method: str = "discrete",
    source_weight_threshold: float = 0.4,
    target_fate_threshold: float = 0.4,
    verbose: bool = True,
):
    """
    Analyze single archetype-to-archetype trajectory.

    Filters cells based on source archetype assignment/weight and target fate
    probability. Returns a subset AnnData ready for CellRank gene trends analysis.

    **IMPORTANT**: This function requires CellRank setup to be run first:
        >>> ck, g = pc.tl.setup_cellrank(adata, high_purity_threshold=0.80)
        >>> pc.tl.compute_lineage_pseudotimes(adata)

    For driver genes, use CellRank directly:
        >>> drivers = g.compute_lineage_drivers(lineages="archetype_3")

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Must contain (from setup_cellrank):
        - `adata.obsm['fate_probabilities']` : Fate probability matrix
        - `adata.uns['lineage_names']` : List of lineage names
        - `adata.obs['pseudotime_to_{archetype}']` : Pseudotime from compute_lineage_pseudotimes
        - `adata.obs['archetypes']` : Discrete archetype assignments (for selection_method='discrete')
        - `adata.obsm['cell_archetype_weights']` : Barycentric weights (for selection_method='weight')
    trajectory : tuple
        Archetype pair as (source_idx, target_idx), e.g., (0, 3) for
        archetype_0 → archetype_3.
    trajectories : list of tuple, optional
        Multiple trajectory pairs to analyze sequentially.
        If provided, `trajectory` is ignored and returns list of results.
    selection_method : str, default: 'discrete'
        How to select source cells:
        - 'discrete' : Filter by adata.obs['archetypes'] == source_archetype
        - 'weight' : Filter by weights[:, source_idx] >= source_weight_threshold
        - 'both' : Compute both and report comparison (uses 'discrete' for subset)
    source_weight_threshold : float, default: 0.4
        Minimum barycentric weight for source archetype (only used if selection_method='weight').
    target_fate_threshold : float, default: 0.4
        Minimum fate probability for target archetype selection.
    verbose : bool, default: True
        Print progress messages.

    Returns
    -------
    Tuple[SingleTrajectoryResult, AnnData]
        - result : SingleTrajectoryResult with trajectory metadata
        - adata_traj : Subset AnnData containing only trajectory cells, ready for
          CellRank gene trends. If trajectories list provided, returns list of tuples.

    Stores in adata
    ---------------
    adata.obs['trajectory_{src}_to_{tgt}_cells'] : bool
        Boolean mask for cells in trajectory.
    adata.uns['trajectory_{src}_to_{tgt}'] : dict
        Trajectory analysis metadata.

    Examples
    --------
    Complete workflow with CellRank:

    >>> import peach as pc
    >>> import cellrank as cr
    >>>
    >>> # 1. Setup CellRank (computes fate probabilities)
    >>> ck, g = pc.tl.setup_cellrank(adata, high_purity_threshold=0.80)
    >>> pc.tl.compute_lineage_pseudotimes(adata)
    >>>
    >>> # 2. Analyze trajectory (returns subset AnnData)
    >>> result, adata_traj = pc.tl.single_trajectory_analysis(adata, trajectory=(4, 5), selection_method="discrete")
    >>> print(f"Found {result.n_trajectory_cells} cells")
    >>>
    >>> # 3. Get drivers from CellRank
    >>> drivers = g.compute_lineage_drivers(lineages="archetype_5")
    >>> top_genes = drivers.index[:5].tolist()
    >>>
    >>> # 4. Plot gene trends using subset
    >>> cr.pl.gene_trends(adata_traj, model=cr.models.GAMR(adata_traj), genes=top_genes, time_key=result.pseudotime_key)

    Compare selection methods:

    >>> result, adata_traj = pc.tl.single_trajectory_analysis(adata, trajectory=(1, 2), selection_method="both")
    >>> print(f"Discrete: {result.n_discrete_cells} cells")
    >>> print(f"Weight-based: {result.n_weight_cells} cells")

    Notes
    -----
    - Requires `setup_cellrank()` and `compute_lineage_pseudotimes()` to be run first
    - Driver computation is NOT included - use CellRank's g.compute_lineage_drivers() directly
    - Pseudotime uses CellRank-computed values from compute_lineage_pseudotimes()

    See Also
    --------
    setup_cellrank : Complete CellRank workflow setup (computes fate probabilities)
    compute_lineage_pseudotimes : Compute pseudotime to each lineage
    """
    from .._core.types import SingleTrajectoryResult

    # Handle multiple trajectories
    if trajectories is not None:
        results = []
        for traj in trajectories:
            result = single_trajectory_analysis(
                adata,
                trajectory=traj,
                selection_method=selection_method,
                source_weight_threshold=source_weight_threshold,
                target_fate_threshold=target_fate_threshold,
                verbose=verbose,
            )
            results.append(result)
        return results

    # Validate inputs
    source_idx, target_idx = trajectory

    if "fate_probabilities" not in adata.obsm:
        raise ValueError("Run pc.tl.setup_cellrank() first to compute fate probabilities")
    if "lineage_names" not in adata.uns:
        raise ValueError("Run pc.tl.setup_cellrank() first")

    # Get archetype names
    source_archetype = f"archetype_{source_idx}"
    target_archetype = f"archetype_{target_idx}"

    # Check for CellRank pseudotime
    cellrank_pseudotime_key = f"pseudotime_to_{target_archetype}"
    if cellrank_pseudotime_key not in adata.obs:
        raise ValueError(
            f"CellRank pseudotime '{cellrank_pseudotime_key}' not found. "
            f"Run pc.tl.compute_lineage_pseudotimes(adata) first."
        )

    # Validate lineage exists
    lineage_names = adata.uns["lineage_names"]
    if target_archetype not in lineage_names:
        raise ValueError(f"Target '{target_archetype}' not in lineage_names: {lineage_names}")

    if verbose:
        print(f"Analyzing trajectory: {source_archetype} → {target_archetype}")
        print(f"  Selection method: {selection_method}")

    # Get fate probabilities
    fate_probs = adata.obsm["fate_probabilities"]
    target_lineage_idx = lineage_names.index(target_archetype)

    # Target filter: cells with high fate probability to target
    target_cells = fate_probs[:, target_lineage_idx] >= target_fate_threshold

    # Source filter: depends on selection_method
    n_discrete_cells = None
    n_weight_cells = None

    # Discrete selection: adata.obs['archetypes'] == source_archetype
    if selection_method in ["discrete", "both"]:
        if "archetypes" not in adata.obs:
            raise ValueError(
                "Discrete selection requires adata.obs['archetypes']. "
                "Run pc.tl.assign_archetypes() first or use selection_method='weight'."
            )
        source_cells_discrete = adata.obs["archetypes"] == source_archetype
        trajectory_mask_discrete = source_cells_discrete & target_cells
        n_discrete_cells = int(trajectory_mask_discrete.sum())
        if verbose:
            print(
                f"  Discrete selection: {int(source_cells_discrete.sum())} source cells → {n_discrete_cells} trajectory cells"
            )

    # Weight-based selection: weights[:, source_idx] >= threshold
    if selection_method in ["weight", "both"]:
        if "cell_archetype_weights" not in adata.obsm:
            raise ValueError(
                "Weight selection requires adata.obsm['cell_archetype_weights']. "
                "Run pc.tl.extract_archetype_weights() or use selection_method='discrete'."
            )
        weights = adata.obsm["cell_archetype_weights"]
        n_archetypes = weights.shape[1]
        if source_idx >= n_archetypes:
            raise ValueError(f"source_idx {source_idx} out of range (max {n_archetypes - 1})")

        source_cells_weight = weights[:, source_idx] >= source_weight_threshold
        trajectory_mask_weight = source_cells_weight & target_cells
        n_weight_cells = int(trajectory_mask_weight.sum())
        if verbose:
            print(
                f"  Weight selection (>={source_weight_threshold}): {int(source_cells_weight.sum())} source cells → {n_weight_cells} trajectory cells"
            )

    # Determine final trajectory mask
    if selection_method == "discrete":
        trajectory_mask = trajectory_mask_discrete
        n_trajectory_cells = n_discrete_cells
    elif selection_method == "weight":
        trajectory_mask = trajectory_mask_weight
        n_trajectory_cells = n_weight_cells
    else:  # 'both' - use discrete for subset, report both counts
        trajectory_mask = trajectory_mask_discrete
        n_trajectory_cells = n_discrete_cells
        if verbose:
            print("  Using discrete selection for subset (both counts reported)")

    if verbose:
        print(f"  Target cells (fate >= {target_fate_threshold}): {int(target_cells.sum())}")
        print(f"  Final trajectory cells: {n_trajectory_cells}")

    # Define storage keys
    trajectory_key = f"trajectory_{source_idx}_to_{target_idx}"
    cell_mask_key = f"{trajectory_key}_cells"
    # Use CellRank pseudotime key
    pseudotime_key = cellrank_pseudotime_key

    # Store mask in adata.obs
    adata.obs[cell_mask_key] = trajectory_mask

    if verbose:
        print(f"  Stored mask: adata.obs['{cell_mask_key}']")
        print(f"  Using pseudotime: adata.obs['{pseudotime_key}']")

    # Create subset AnnData for trajectory cells
    adata_traj = adata[trajectory_mask].copy()
    adata_traj.uns["trajectory_info"] = {
        "source": source_archetype,
        "target": target_archetype,
        "selection_method": selection_method,
        "pseudotime_key": pseudotime_key,
    }

    if verbose:
        print(f"  Created subset: {adata_traj.n_obs} cells × {adata_traj.n_vars} genes")
        pt_values = adata_traj.obs[pseudotime_key]
        print(f"  Pseudotime range: [{pt_values.min():.3f}, {pt_values.max():.3f}]")

    # Store full results in uns
    adata.uns[trajectory_key] = {
        "source": source_archetype,
        "target": target_archetype,
        "source_idx": source_idx,
        "target_idx": target_idx,
        "n_cells": n_trajectory_cells,
        "selection_method": selection_method,
        "n_discrete_cells": n_discrete_cells,
        "n_weight_cells": n_weight_cells,
        "source_weight_threshold": source_weight_threshold,
        "target_fate_threshold": target_fate_threshold,
        "pseudotime_key": pseudotime_key,
    }

    if verbose:
        print(f"  Stored results: adata.uns['{trajectory_key}']")

    # Return Pydantic result and subset AnnData
    result = SingleTrajectoryResult(
        source_archetype=source_archetype,
        target_archetype=target_archetype,
        source_idx=source_idx,
        target_idx=target_idx,
        trajectory_key=trajectory_key,
        n_trajectory_cells=n_trajectory_cells,
        pseudotime_key=pseudotime_key,
        cell_mask_key=cell_mask_key,
        selection_method=selection_method,
        n_discrete_cells=n_discrete_cells,
        n_weight_cells=n_weight_cells,
        source_weight_threshold=source_weight_threshold,
        target_fate_threshold=target_fate_threshold,
    )

    return result, adata_traj

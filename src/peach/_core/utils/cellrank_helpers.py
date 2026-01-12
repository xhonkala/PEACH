"""
Helper functions for CellRank archetypal transition analysis.
"""

import numpy as np
import pandas as pd
import scanpy as sc


def define_high_purity_cells(adata, threshold=0.80, archetypes=None, exclude_zero=True, terminal_obs_key="archetypes"):
    """
    Identify high-purity cells for each archetype or centroid assignment.

    Parameters
    ----------
    adata : AnnData
        Must contain 'cell_archetype_weights' in obsm (unless using centroid mode)
    threshold : float
        Percentile threshold (e.g., 0.90 = top 10%)
    archetypes : list, optional
        List of archetype indices to process (default: 1-5)
    exclude_zero : bool
        Whether to exclude archetype_0 (centroid)
    terminal_obs_key : str, default='archetypes'
        Key in adata.obs containing terminal state assignments.
        Use 'archetypes' for archetype-based analysis or
        'centroid_assignments' for treatment phase centroids.

    Returns
    -------
    dict : {label: boolean_mask}
    """
    # Check if we're using centroid assignments (no weights available)
    using_centroids = terminal_obs_key != "archetypes"

    if using_centroids:
        # Centroid mode: use assignment categories directly
        if terminal_obs_key not in adata.obs:
            raise ValueError(f"'{terminal_obs_key}' not found in adata.obs")

        assignments = adata.obs[terminal_obs_key]
        high_purity_masks = {}

        # Get unique categories (excluding 'unassigned')
        categories = [c for c in assignments.cat.categories if c != "unassigned"]

        for cat in categories:
            mask = assignments == cat
            high_purity_masks[cat] = mask
            print(f"{cat}: {mask.sum()} terminal cells")

        return high_purity_masks

    # Standard archetype mode with barycentric weights
    weights = adata.obsm["cell_archetype_weights"]

    # Determine which archetypes to process
    if archetypes is None:
        # Weight indices 0-4 map to labeled archetypes 1-5
        archetypes = list(range(weights.shape[1]))

    high_purity_masks = {}

    for i in archetypes:
        # Map weight index to archetype label
        arch_label = f"archetype_{i + 1}"

        # Find cells with assigned archetype
        assigned = adata.obs[terminal_obs_key] == arch_label

        if not assigned.any():
            print(f"[WARNING]  No cells assigned to {arch_label}")
            continue

        # Calculate threshold for this archetype
        weight_threshold = np.percentile(weights[assigned, i], threshold * 100)

        # High purity = assigned AND above threshold
        high_purity = assigned & (weights[:, i] >= weight_threshold)

        high_purity_masks[arch_label] = high_purity

        print(f"{arch_label}: {high_purity.sum()} high-purity cells (threshold={weight_threshold:.3f})")

    # Handle archetype_0 separately if requested
    if not exclude_zero and "archetype_0" in adata.obs[terminal_obs_key].values:
        # archetype_0 is defined by proximity to centroid, not barycentric weight
        arch0_cells = adata.obs[terminal_obs_key] == "archetype_0"
        high_purity_masks["archetype_0"] = arch0_cells
        print(f"archetype_0: {arch0_cells.sum()} cells (centroid, all included)")

    return high_purity_masks


def compute_lineage_pseudotimes(adata, lineage_names=None, fate_prob_key="fate_probabilities"):
    """
    Convert fate probabilities to per-lineage pseudotimes.

    Creates pseudotime variables for each lineage by using fate probabilities
    as continuous progression measures. Stores results in adata.obs.

    Parameters
    ----------
    adata : AnnData
        Must contain fate_probabilities in obsm and lineage_names in uns
    lineage_names : list, optional
        List of lineage names. If None, uses all lineages in adata.uns['lineage_names']
    fate_prob_key : str
        Key in obsm containing fate probabilities (default: 'fate_probabilities')

    Returns
    -------
    None (stores pseudotime_to_{lineage} in adata.obs for each lineage)

    Examples
    --------
    >>> pc.tl.compute_lineage_pseudotimes(adata)
    >>> # Access pseudotimes
    >>> pseudotime = adata.obs["pseudotime_to_archetype_5"]
    """
    if fate_prob_key not in adata.obsm:
        raise ValueError(f"'{fate_prob_key}' not found in adata.obsm. Run setup_cellrank first.")

    if "lineage_names" not in adata.uns:
        raise ValueError("'lineage_names' not found in adata.uns. Run setup_cellrank first.")

    fate_probs = adata.obsm[fate_prob_key]

    if lineage_names is None:
        lineage_names = adata.uns["lineage_names"]

    print(f"Computing pseudotimes for {len(lineage_names)} lineages...")

    for i, lineage in enumerate(lineage_names):
        if i >= fate_probs.shape[1]:
            print(f"[WARNING]  Skipping {lineage}: index {i} out of bounds")
            continue

        # Use fate probability as pseudotime
        pseudotime_key = f"pseudotime_to_{lineage}"
        adata.obs[pseudotime_key] = fate_probs[:, i]

        print(f"  {lineage}: stored as '{pseudotime_key}'")

    print("[OK] Pseudotimes computed")


def setup_cellrank_workflow(
    adata,
    high_purity_threshold=0.80,
    n_neighbors=30,
    n_pcs=11,
    compute_paga=True,
    kernel_type="connectivity",
    solver="gmres",
    tol=1e-6,
    terminal_obs_key="archetypes",
    verbose=True,
):
    """
    Complete CellRank workflow: neighbors → UMAP → PAGA → ConnectivityKernel → GPCCA.

    This function orchestrates the full pipeline from raw archetypal assignments
    to fate probabilities, handling all intermediate steps.

    Parameters
    ----------
    adata : AnnData
        Must contain terminal_obs_key in obs. If using archetypes, also needs
        'cell_archetype_weights' in obsm.
    high_purity_threshold : float
        Percentile threshold for high-purity cells (0.80 = top 20%).
        Only used when terminal_obs_key='archetypes'.
    n_neighbors : int
        Number of neighbors for k-NN graph
    n_pcs : int
        Number of PCs to use for neighbors computation
    compute_paga : bool
        Whether to compute PAGA connectivity
    kernel_type : str
        Type of CellRank kernel ('connectivity' only for now)
    solver : str
        Solver for fate probability computation ('gmres', 'direct', etc.)
    tol : float
        Tolerance for iterative solver
    terminal_obs_key : str, default='archetypes'
        Key in adata.obs containing terminal state assignments.
        Use 'archetypes' for standard archetype-based analysis or
        'centroid_assignments' for treatment phase centroid trajectories.
    verbose : bool
        Print progress messages

    Returns
    -------
    ck : ConnectivityKernel
        Computed transition kernel
    g : GPCCA
        GPCCA estimator with computed fate probabilities

    Stores in adata
    ---------------
    adata.obs['terminal_states'] : Terminal state assignments
    adata.obsm['fate_probabilities'] : Fate probability matrix (n_obs × n_lineages)
    adata.uns['lineage_names'] : List of lineage names
    adata.obsm['X_umap'] : UMAP coordinates (if not present)
    adata.uns['neighbors'] : k-NN graph (if not present)
    adata.uns['paga'] : PAGA results (if compute_paga=True)

    Examples
    --------
    >>> # Standard archetype-based analysis
    >>> ck, g = pc.tl.setup_cellrank(adata, high_purity_threshold=0.80)

    >>> # Treatment phase centroid-based analysis
    >>> pc.tl.assign_to_centroids(adata, condition_column="treatment_phase")
    >>> ck, g = pc.tl.setup_cellrank(adata, terminal_obs_key="centroid_assignments")
    """
    try:
        from cellrank.estimators import GPCCA
        from cellrank.kernels import ConnectivityKernel
    except ImportError:
        raise ImportError("CellRank not installed. Install with: pip install cellrank")

    if verbose:
        print("=" * 60)
        print("CellRank Workflow Setup")
        print("=" * 60)

    # Step 1: Compute neighbors if not present
    if "neighbors" not in adata.uns:
        if verbose:
            print(f"\n1. Computing neighbors (n_neighbors={n_neighbors}, n_pcs={n_pcs})...")
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    elif verbose:
        print("\n1. [OK] Neighbors already computed")

    # Step 2: Compute UMAP if not present
    if "X_umap" not in adata.obsm:
        if verbose:
            print("\n2. Computing UMAP...")
        sc.tl.umap(adata)
    elif verbose:
        print("\n2. [OK] UMAP already computed")

    # Step 3: Define high-purity cells (or use centroid assignments)
    using_centroids = terminal_obs_key != "archetypes"
    if verbose:
        if using_centroids:
            print(f"\n3. Using centroid assignments from '{terminal_obs_key}'...")
        else:
            print(f"\n3. Defining high-purity cells (threshold={high_purity_threshold})...")
    high_purity_masks = define_high_purity_cells(
        adata, threshold=high_purity_threshold, terminal_obs_key=terminal_obs_key
    )
    terminal_labels = sorted([k for k in high_purity_masks.keys() if k not in ("no_archetype", "unassigned")])

    # Step 4: Compute PAGA
    if compute_paga:
        if verbose:
            print("\n4. Computing PAGA...")
        sc.tl.paga(adata, groups=terminal_obs_key)
        if verbose:
            print("   [OK] PAGA computed")
    elif verbose:
        print("\n4. Skipping PAGA (compute_paga=False)")

    # Step 5: Build ConnectivityKernel
    if verbose:
        print(f"\n5. Building {kernel_type} kernel...")

    if kernel_type == "connectivity":
        ck = ConnectivityKernel(adata).compute_transition_matrix()
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")

    if verbose:
        print(f"   [OK] Kernel shape: {ck.transition_matrix.shape}")

    # Step 6: Set terminal states
    if verbose:
        print("\n6. Setting terminal states...")

    terminal_states = pd.Series(None, index=adata.obs_names)
    for arch, mask in high_purity_masks.items():
        terminal_states[mask] = arch
        if verbose:
            print(f"   {arch}: {mask.sum()} terminal cells")

    # Convert to categorical (required by CellRank)
    terminal_states = terminal_states.astype(pd.CategoricalDtype(categories=terminal_labels))

    if verbose:
        print(f"   [OK] Total terminal cells: {terminal_states.notna().sum()}")

    # Step 7: Compute fate probabilities
    if verbose:
        print(f"\n7. Computing fate probabilities (solver='{solver}')...")

    g = GPCCA(ck)
    g.set_terminal_states(terminal_states)
    g.compute_fate_probabilities(solver=solver, tol=tol)

    if verbose:
        print(f"   [OK] Fate probabilities: {g.fate_probabilities.shape}")
        print(f"   Lineages: {list(g.fate_probabilities.names)}")

    # Step 8: Store results in AnnData
    adata.obs["terminal_states"] = terminal_states
    adata.obsm["fate_probabilities"] = g.fate_probabilities.X
    adata.uns["lineage_names"] = list(g.fate_probabilities.names)

    if verbose:
        print("\n" + "=" * 60)
        print("[OK] CellRank workflow complete")
        print("=" * 60)
        print("\nStored in AnnData:")
        print("  adata.obs['terminal_states']")
        print("  adata.obsm['fate_probabilities']")
        print("  adata.uns['lineage_names']")

    return ck, g


def diagnose_transition_matrix(T, name="Transition Matrix"):
    """
    Comprehensive diagnostics for transition matrix.
    """
    print(f"\n{'=' * 60}")
    print(f"{name} Diagnostics")
    print(f"{'=' * 60}")

    # Basic properties
    print(f"Shape: {T.shape}")
    print(f"Sparsity: {1 - T.nnz / (T.shape[0] * T.shape[1]):.4f}")
    print(f"Non-zero entries: {T.nnz:,}")

    # Stochasticity check
    row_sums = np.array(T.sum(axis=1)).flatten()
    print("\nStochasticity (row sums):")
    print(f"  Min: {row_sums.min():.8f}")
    print(f"  Max: {row_sums.max():.8f}")
    print(f"  Mean: {row_sums.mean():.8f}")
    print(f"  Std: {row_sums.std():.8f}")

    # Check if valid
    if np.allclose(row_sums, 1.0, atol=1e-3):
        print("  [OK] Valid stochastic matrix")
    else:
        print("  [WARNING]  Row sums deviate from 1.0")

    # Symmetry check
    T_dense = T.toarray() if hasattr(T, "toarray") else T
    symmetry = np.abs(T_dense - T_dense.T).max()
    print(f"\nSymmetry: max|T - T^T| = {symmetry:.6f}")
    if symmetry < 1e-10:
        print("  [OK] Symmetric matrix")
    else:
        print("  [OK] Asymmetric matrix (directional)")

    # Distribution of values
    values = T.data if hasattr(T, "data") else T_dense[T_dense > 0]
    print("\nValue distribution (non-zero):")
    print(f"  Min: {values.min():.6f}")
    print(f"  25%: {np.percentile(values, 25):.6f}")
    print(f"  50%: {np.percentile(values, 50):.6f}")
    print(f"  75%: {np.percentile(values, 75):.6f}")
    print(f"  Max: {values.max():.6f}")


def diagnose_fate_probabilities(g, adata):
    """
    Comprehensive diagnostics for CellRank fate probabilities.
    """
    fp = g.fate_probabilities

    print(f"\n{'=' * 60}")
    print("Fate Probabilities Diagnostics")
    print(f"{'=' * 60}")

    print(f"Shape: {fp.shape}")
    print(f"Lineages: {list(fp.names)}")

    # Row sum check
    row_sums = fp.X.sum(axis=1)
    print("\nRow sums (should be ~1.0):")
    print(f"  Min: {row_sums.min():.6f}")
    print(f"  Max: {row_sums.max():.6f}")
    print(f"  Mean: {row_sums.mean():.6f}")
    print(f"  Std: {row_sums.std():.6f}")

    if np.allclose(row_sums, 1.0, atol=1e-2):
        print("  [OK] Valid probability distribution")
    else:
        print("  [WARNING]  Row sums deviate from 1.0")

    # Per-lineage statistics
    print("\nPer-lineage statistics:")
    for i, lineage in enumerate(fp.names):
        probs = fp.X[:, i]
        print(f"\n  {lineage}:")
        print(f"    Mean: {probs.mean():.4f}")
        print(f"    Std: {probs.std():.4f}")
        print(f"    Max: {probs.max():.4f}")
        print(f"    Cells with >0.5 prob: {(probs > 0.5).sum()}")
        print(f"    Cells with >0.3 prob: {(probs > 0.3).sum()}")

    # Terminal state assignments
    print("\nTerminal states:")
    print(g.terminal_states.value_counts())


def compute_transition_frequencies(
    adata, fate_probs, archetype_labels, start_weight_threshold=0.5, fate_prob_threshold=0.3
):
    """
    Compute frequency of transitions between archetypes.

    Identifies cells transitioning from one archetype to another based on:
    - High starting weight for source archetype (> start_weight_threshold)
    - High fate probability for target archetype (> fate_prob_threshold)

    Parameters
    ----------
    adata : AnnData
        Annotated data object with archetypal analysis results.
        Must contain:
        - adata.obsm['cell_archetype_weights']: Barycentric weights [n_obs, n_archetypes]
        - adata.obs['archetypes']: Categorical archetype assignments

    fate_probs : CellRank FateProbs object or similar
        Object with attributes:
        - fate_probs.X: Fate probability matrix [n_obs, n_lineages]
        - fate_probs.names: List of lineage names

    archetype_labels : list of str
        List of archetype names to analyze (e.g., ['archetype_0', 'archetype_1', ...])

    start_weight_threshold : float, default=0.5
        Minimum barycentric weight to consider a cell as "starting" from an archetype.
        Typical values: 0.5 (top 50%), 0.7 (top 30%), 0.3 (top 70%)

    fate_prob_threshold : float, default=0.3
        Minimum fate probability to consider a cell as "transitioning to" an archetype.
        Typical values: 0.3 (30% commitment), 0.5 (50% commitment)

    Returns
    -------
    pd.DataFrame
        Transition frequency matrix with archetype labels as index and columns.
        Values represent number of cells satisfying both threshold conditions.
        Shape: [n_archetypes, n_archetypes]

        Example structure:
                        archetype_0  archetype_1  archetype_2  ...
        archetype_0            150           45           23
        archetype_1             12          200           67
        archetype_2              8           34          180
        ...

    Notes
    -----
    - Diagonal elements represent cells maintaining their archetype identity
    - Off-diagonal elements represent cross-archetype transitions
    - archetype_0 (centroid) uses categorical assignment instead of weight threshold
    - Returns integer counts (not normalized probabilities)

    Examples
    --------
    >>> from peach._core.utils.cellrank_helpers import compute_transition_frequencies
    >>> transitions = compute_transition_frequencies(
    ...     adata,
    ...     fate_probs=g.fate_probabilities,
    ...     archetype_labels=["archetype_0", "archetype_1", "archetype_2"],
    ...     start_weight_threshold=0.5,
    ...     fate_prob_threshold=0.3,
    ... )
    >>> print(transitions)
    """
    weights = adata.obsm["cell_archetype_weights"]
    n_arch = len(archetype_labels)

    # Initialize transition matrix
    transitions = np.zeros((n_arch, n_arch))

    for i, arch_i in enumerate(archetype_labels):
        # Find starting cells (high weight for archetype i)
        if "archetype_0" in arch_i:
            # archetype_0 is special (centroid)
            start_cells = adata.obs["archetypes"] == arch_i
        else:
            # Map label to weight index (archetype_1 → index 0, etc.)
            weight_idx = int(arch_i.split("_")[1]) - 1
            start_cells = weights[:, weight_idx] > start_weight_threshold

        for j, arch_j in enumerate(archetype_labels):
            # Find cells with high fate prob for archetype j
            try:
                lineage_idx = list(fate_probs.names).index(arch_j)
                fate_cells = fate_probs.X[:, lineage_idx] > fate_prob_threshold

                # Count cells satisfying both conditions
                transition_cells = start_cells & fate_cells
                transitions[i, j] = transition_cells.sum()
            except ValueError:
                # Lineage not in fate probabilities
                transitions[i, j] = 0

    # Return as DataFrame with proper labels
    return pd.DataFrame(transitions, index=archetype_labels, columns=archetype_labels, dtype=int)

"""Flow matching public API: within-model, between-model, gene alignment, Jacobian."""

import numpy as np
from anndata import AnnData

from peach._core.utils.feature_utils import store_result
from peach._core.utils.flow_matching import FlowModel, compute_mmd


def flow_within(
    adata: AnnData,
    source: dict,
    target: dict,
    *,
    pca_key: str = "X_pca",
    hidden_dims: tuple = (128, 128, 128),
    lr: float = 1e-3,
    n_epochs: int = 1000,
    batch_size: int = 256,
    n_steps: int = 50,
    device: str = "cpu",
    solver_method: str = "dopri5",
    name: str | None = None,
    random_state: int = 42,
    return_model: bool = False,
    use_ot: bool = False,
    holdout_fraction: float = 0.0,
    copy: bool = False,
) -> dict:
    """Intra-model flow between obs-defined cell subsets.

    Parameters
    ----------
    adata : AnnData
    source : dict
        Obs column filter, e.g. {'treatment': 'Base'}.
    target : dict
        Obs column filter, e.g. {'treatment': 'PD1'}.
    pca_key : str
        Key in adata.obsm for PCA coordinates.
    hidden_dims, lr, n_epochs, batch_size : model params
    n_steps : int
        ODE integration steps.
    device : str
    solver_method : str
        ODE solver method: 'dopri5' (default, adaptive), 'euler',
        'midpoint', 'heun3'.
    name : str or None
        Name for storage key.
    random_state : int
    return_model : bool
        If True, include the trained FlowModel in the result dict under
        key ``'model'``. Needed for Jacobian and trajectory analysis.
    use_ot : bool
        If True, use minibatch Sinkhorn OT coupling for training pairs.
        Requires the ``POT`` package.
    holdout_fraction : float
        Fraction of source cells to hold out for validation (0 to 1).
        If > 0, trains on the remaining source cells and evaluates MMD
        on the held-out set after transport.
    copy : bool
    """
    if copy:
        adata = adata.copy()

    if pca_key not in adata.obsm:
        raise ValueError(f"adata.obsm['{pca_key}'] not found.")

    # Build masks
    source_mask = _build_mask(adata, source)
    target_mask = _build_mask(adata, target)

    pca = adata.obsm[pca_key]
    source_pca = pca[source_mask]
    target_pca = pca[target_mask]
    dim = source_pca.shape[1]

    # Holdout split
    if holdout_fraction > 0:
        rng_ho = np.random.default_rng(random_state)
        n_source = source_pca.shape[0]
        n_holdout = max(int(n_source * holdout_fraction), 1)
        perm = rng_ho.permutation(n_source)
        holdout_idx = perm[:n_holdout]
        train_idx = perm[n_holdout:]
        source_train = source_pca[train_idx]
        source_holdout = source_pca[holdout_idx]
    else:
        source_train = source_pca
        source_holdout = None

    # Train flow model
    import torch
    torch.manual_seed(random_state)
    model = FlowModel(dim, hidden_dims=hidden_dims, lr=lr,
                      solver_method=solver_method, device=device)
    losses = model.train(source_train, target_pca, n_epochs=n_epochs,
                         batch_size=batch_size, use_ot=use_ot)

    # Transport full source (not just training subset)
    transported = model.transport(source_pca, n_steps=n_steps)

    # MMD
    mmd_before = compute_mmd(source_pca, target_pca)
    mmd_after = compute_mmd(transported, target_pca)

    result = {
        "source_mask": source_mask,
        "target_mask": target_mask,
        "transported": transported,
        "losses": losses,
        "mmd_before": mmd_before,
        "mmd_after": mmd_after,
        "pca_key": pca_key,
        "name": name,
    }
    if return_model:
        result["model"] = model

    # Holdout validation
    if source_holdout is not None:
        holdout_transported = model.transport(source_holdout, n_steps=n_steps)
        holdout_mmd = compute_mmd(holdout_transported, target_pca)
        result["holdout_mmd"] = holdout_mmd
        result["holdout_fraction"] = holdout_fraction

    # Store summary (not the model itself)
    storage_key = f"flow_{name}" if name else "flow_within"
    store_result(adata, storage_key, {
        "mmd_before": mmd_before,
        "mmd_after": mmd_after,
        "n_source": int(source_mask.sum()),
        "n_target": int(target_mask.sum()),
        "pca_key": pca_key,
    })

    return result


def flow_between(
    adatas: list[AnnData],
    *,
    condition_key: str = "condition",
    condition_labels: list[str] | None = None,
    pairs: list[tuple] | None = None,
    pca_key: str = "X_pca",
    hidden_dims: tuple = (128, 128, 128),
    lr: float = 1e-3,
    n_epochs: int = 1000,
    batch_size: int = 256,
    n_steps: int = 50,
    device: str = "cpu",
    solver_method: str = "dopri5",
    use_ot: bool = False,
    random_state: int = 42,
) -> dict:
    """Inter-model flow between separate AnnDatas.

    Parameters
    ----------
    adatas : list[AnnData]
    condition_key : str
    condition_labels : list[str] or None
    pairs : list[tuple] or None
        (source_label, target_label) pairs. Default: consecutive pairs.
    pca_key : str
    use_ot : bool
        If True, use minibatch Sinkhorn OT coupling for training pairs.
    """
    import anndata as ad

    # Validate PCA dimensions match
    dims = [a.obsm[pca_key].shape[1] for a in adatas]
    if len(set(dims)) > 1:
        raise ValueError(
            f"PCA dimensions don't match across AnnDatas: {dims}. "
            "All inputs must share the same PCA embedding."
        )

    # Labels
    if condition_labels is None:
        condition_labels = [f"condition_{i}" for i in range(len(adatas))]

    # Add condition labels to copies (never mutate caller's data)
    adatas_copy = []
    for a, label in zip(adatas, condition_labels):
        a_copy = a.copy()
        a_copy.obs[condition_key] = label
        adatas_copy.append(a_copy)

    adata_combined = ad.concat(adatas_copy, label=condition_key, keys=condition_labels)

    # Determine pairs
    if pairs is None:
        pairs = [(condition_labels[i], condition_labels[i + 1])
                 for i in range(len(condition_labels) - 1)]

    # Train flows for each pair
    flows = {}
    for pair_idx, (src_label, tgt_label) in enumerate(pairs):
        result = flow_within(
            adata_combined,
            source={condition_key: src_label},
            target={condition_key: tgt_label},
            pca_key=pca_key,
            hidden_dims=hidden_dims,
            lr=lr,
            n_epochs=n_epochs,
            batch_size=batch_size,
            n_steps=n_steps,
            device=device,
            solver_method=solver_method,
            use_ot=use_ot,
            name=f"{src_label}_to_{tgt_label}",
            random_state=random_state + pair_idx,
        )
        flows[(src_label, tgt_label)] = result

    return {
        "condition_key": condition_key,
        "condition_labels": condition_labels,
        "flows": flows,
        "archetype_correspondence": None,  # computed on demand
    }


def flow_gene_alignment(
    adata: AnnData,
    flow_result: dict,
    *,
    t: float = 0.5,
    n_top: int = 50,
    pca_loadings_key: str | None = None,
    n_permutations: int = 0,
    per_cell: bool = False,
    random_state: int = 42,
) -> dict:
    """Compute gene alignment with flow velocity.

    Parameters
    ----------
    adata : AnnData
    flow_result : FlowWithinResult
    t : float
        Time point to evaluate velocity.
    n_top : int
        Top aligned/opposed genes to report.
    pca_loadings_key : str or None
        Key in adata.varm for PCA loadings. Default: 'PCs'.
    per_cell : bool
        If True, also compute per-cell per-gene alignment scores.
        Returns an additional key ``'per_cell_alignment'`` with shape
        ``[n_source, n_genes]``.
    """
    # Get PCA loadings
    if pca_loadings_key is None:
        pca_loadings_key = "PCs"
    if pca_loadings_key not in adata.varm:
        raise ValueError(f"adata.varm['{pca_loadings_key}'] not found.")

    loadings = adata.varm[pca_loadings_key]  # [n_genes, n_PCs]
    gene_names = list(adata.var_names)

    # Use the transported - source difference as mean velocity
    source_pca = adata.obsm[flow_result["pca_key"]][flow_result["source_mask"]]
    mean_velocity = (flow_result["transported"] - source_pca).mean(axis=0)  # [dim]

    # Trim loadings to match PCA dims
    n_pcs = len(mean_velocity)
    loadings_trimmed = loadings[:, :n_pcs]

    # Alignment: dot product of each gene's loading with mean velocity
    alignment_scores = loadings_trimmed @ mean_velocity  # [n_genes]

    # Top aligned/opposed
    sorted_idx = np.argsort(alignment_scores)
    top_opposed = [gene_names[i] for i in sorted_idx[:n_top]]
    top_aligned = [gene_names[i] for i in sorted_idx[-n_top:][::-1]]

    result = {
        "alignment_scores": alignment_scores,
        "gene_names": gene_names,
        "top_aligned": top_aligned,
        "top_opposed": top_opposed,
        "t": t,
    }

    if per_cell:
        velocity_per_cell = flow_result["transported"] - source_pca  # [n_source, n_pcs]
        vel_norm = velocity_per_cell / (
            np.linalg.norm(velocity_per_cell, axis=1, keepdims=True) + 1e-10
        )
        load_norm = loadings_trimmed / (
            np.linalg.norm(loadings_trimmed, axis=1, keepdims=True) + 1e-10
        )
        per_cell_alignment = vel_norm @ load_norm.T  # [n_source, n_genes]
        result["per_cell_alignment"] = per_cell_alignment

    if n_permutations > 0:
        rng = np.random.default_rng(random_state)
        null_scores = np.zeros((n_permutations, len(alignment_scores)))
        for i in range(n_permutations):
            perm_loadings = loadings_trimmed[rng.permutation(len(loadings_trimmed))]
            null_scores[i] = perm_loadings @ mean_velocity

        pvalues = np.array([
            (np.sum(np.abs(null_scores[:, g]) >= np.abs(alignment_scores[g])) + 1)
            / (n_permutations + 1)
            for g in range(len(alignment_scores))
        ])
        from statsmodels.stats.multitest import multipletests
        _, pvalues_fdr, _, _ = multipletests(pvalues, method="fdr_bh")

        result["alignment_pvalues"] = pvalues
        result["alignment_pvalues_fdr"] = pvalues_fdr
        result["null_mean"] = null_scores.mean(axis=0)
        result["null_std"] = null_scores.std(axis=0)

    return result


def flow_jacobian(
    adata: AnnData,
    flow_result: dict,
    flow_model: "FlowModel",
    *,
    t: float = 0.5,
    evaluation_points: np.ndarray | None = None,
    pca_loadings_key: str | None = None,
    aggregate: str = "mean",
) -> dict:
    """Compute Jacobian of the flow velocity field.

    Parameters
    ----------
    adata : AnnData
    flow_result : FlowWithinResult
    flow_model : FlowModel
        The trained FlowModel (not stored in adata).
    t : float
    evaluation_points : np.ndarray or None
        Default: source cell positions.
    pca_loadings_key : str or None
    aggregate : str
        'mean', 'median', or None (per-cell).
    """
    if evaluation_points is None:
        evaluation_points = adata.obsm[flow_result["pca_key"]][flow_result["source_mask"]]

    # Compute Jacobian
    jac = flow_model.jacobian(evaluation_points, t)  # [n_points, dim, dim]

    # Jacobian determinant (local volume change)
    # Use slogdet to avoid underflow in high-dimensional spaces
    signs, logdets = np.linalg.slogdet(jac)
    jac_det = signs * np.exp(np.clip(logdets, -500, 500))  # Clipped exp for safety

    # Mean Jacobian
    if aggregate == "mean":
        mean_jac = jac.mean(axis=0)
    elif aggregate == "median":
        mean_jac = np.median(jac, axis=0)
    else:
        mean_jac = jac.mean(axis=0)

    # Per-gene expansion: project Jacobian onto PCA loadings
    if pca_loadings_key is None:
        pca_loadings_key = "PCs"
    if pca_loadings_key in adata.varm:
        loadings = adata.varm[pca_loadings_key]
        n_pcs = mean_jac.shape[0]
        loadings_trimmed = loadings[:, :n_pcs]
        # For each gene, compute how its PCA direction is expanded/contracted
        feature_expansion = np.array([
            np.dot(loadings_trimmed[g], mean_jac @ loadings_trimmed[g])
            for g in range(len(loadings_trimmed))
        ])
    else:
        feature_expansion = np.zeros(0)

    result = {
        "jacobian_det": jac_det,
        "jac_logdet": logdets,
        "jac_det_sign": signs,
        "feature_expansion": feature_expansion,
        "mean_jacobian": mean_jac,
        "t": t,
    }

    return result


def flow_significance(
    adata: AnnData,
    flow_result: dict | None = None,
    *,
    source: dict | None = None,
    target: dict | None = None,
    pca_key: str = "X_pca",
    n_permutations: int = 100,
    n_epochs_per_perm: int = 200,
    statistic: str = "mmd",
    hidden_dims: tuple = (128, 128, 128),
    lr: float = 1e-3,
    batch_size: int = 256,
    n_steps: int = 50,
    device: str = "cpu",
    solver_method: str = "euler",
    random_state: int = 42,
) -> dict:
    """Permutation test for flow significance.

    Permutes condition labels, retrains flow per permutation.

    Returns
    -------
    dict with p_value, observed_stat, null_distribution
    """
    import torch

    if flow_result is not None:
        source_mask = flow_result["source_mask"]
        target_mask = flow_result["target_mask"]
        pca_key = flow_result["pca_key"]
    elif source is not None and target is not None:
        source_mask = _build_mask(adata, source)
        target_mask = _build_mask(adata, target)
    else:
        raise ValueError("Provide flow_result or source/target dicts.")

    pca = adata.obsm[pca_key]
    source_pca = pca[source_mask]
    target_pca = pca[target_mask]
    dim = source_pca.shape[1]

    # Observed statistic
    observed_mmd = compute_mmd(source_pca, target_pca)

    # Null distribution
    rng = np.random.default_rng(random_state)
    combined = np.vstack([source_pca, target_pca])
    n_source = len(source_pca)
    null_stats = []

    for i in range(n_permutations):
        # Permute labels
        perm = rng.permutation(len(combined))
        perm_source = combined[perm[:n_source]]
        perm_target = combined[perm[n_source:]]

        # Train short flow
        torch.manual_seed(random_state + i)
        perm_model = FlowModel(dim, hidden_dims=hidden_dims, lr=lr,
                               solver_method=solver_method, device=device)
        perm_model.train(perm_source, perm_target, n_epochs=n_epochs_per_perm, batch_size=batch_size)
        perm_transported = perm_model.transport(perm_source, n_steps=n_steps)

        mmd_improvement = compute_mmd(perm_source, perm_target) - compute_mmd(perm_transported, perm_target)
        null_stats.append(mmd_improvement)

    # Observed improvement
    torch.manual_seed(random_state)
    obs_model = FlowModel(dim, hidden_dims=hidden_dims, lr=lr,
                          solver_method=solver_method, device=device)
    obs_model.train(source_pca, target_pca, n_epochs=n_epochs_per_perm, batch_size=batch_size)
    obs_transported = obs_model.transport(source_pca, n_steps=n_steps)
    observed_improvement = observed_mmd - compute_mmd(obs_transported, target_pca)

    null_stats = np.array(null_stats)
    p_value = (np.sum(null_stats >= observed_improvement) + 1) / (n_permutations + 1)

    return {
        "p_value": p_value,
        "observed_stat": observed_improvement,
        "null_distribution": null_stats,
    }


def flow_bifurcation(
    adata: AnnData,
    flow_result: dict,
    flow_model,
    *,
    n_timepoints: int = 10,
    evaluation_points: np.ndarray | None = None,
) -> dict:
    """Eigenvalue-based bifurcation scoring along the flow trajectory.

    Transports evaluation points along the learned flow and computes the
    Jacobian at each timepoint. Bifurcation is scored by the maximum
    absolute divergence (trace of Jacobian) along the trajectory, and
    saddle points are identified as timepoints with mixed-sign eigenvalue
    real parts.

    Parameters
    ----------
    adata : AnnData
    flow_result : dict
        Output of :func:`flow_within`.
    flow_model : FlowModel
        Trained flow model with ``.jacobian()`` and ``.transport()`` methods.
    n_timepoints : int
        Number of timepoints to evaluate along the trajectory.
    evaluation_points : np.ndarray or None
        Points to evaluate. Default: source cells from ``flow_result``.

    Returns
    -------
    dict
        Keys:

        - ``divergence``: ``[n_timepoints, n_cells]`` — trace of Jacobian
        - ``bifurcation_score``: ``[n_cells]`` — max |divergence| along trajectory
        - ``eigenvalue_real``: ``[n_timepoints, n_cells, dim]``
        - ``eigenvalue_imag``: ``[n_timepoints, n_cells, dim]``
        - ``timepoints``: ``[n_timepoints]``
        - ``n_saddle_points``: ``[n_cells]`` — count of timepoints with
          mixed-sign eigenvalue real parts
    """
    if evaluation_points is None:
        evaluation_points = adata.obsm[flow_result["pca_key"]][flow_result["source_mask"]]

    n_cells = evaluation_points.shape[0]
    dim = evaluation_points.shape[1]
    timepoints = np.linspace(0.05, 0.95, n_timepoints)

    # Transport to get positions at each timepoint
    trajectory = flow_model.transport(
        evaluation_points, n_steps=n_timepoints - 1, return_trajectory=True
    )  # [n_steps+1, n_cells, dim]

    # Map trajectory frames to timepoints: trajectory has n_timepoints frames
    # at np.linspace(0, 1, n_timepoints), but we want to evaluate Jacobian at
    # our custom timepoints. We use the trajectory positions that are closest
    # to each desired timepoint.
    traj_times = np.linspace(0, 1, trajectory.shape[0])

    divergence = np.zeros((n_timepoints, n_cells))
    eigenvalue_real = np.zeros((n_timepoints, n_cells, dim))
    eigenvalue_imag = np.zeros((n_timepoints, n_cells, dim))

    for ti, t_val in enumerate(timepoints):
        # Find closest trajectory frame
        frame_idx = np.argmin(np.abs(traj_times - t_val))
        positions = trajectory[frame_idx]  # [n_cells, dim]

        # Compute Jacobian at these positions and this time
        jac = flow_model.jacobian(positions, float(t_val))  # [n_cells, dim, dim]

        # Divergence = trace(J) per cell
        for ci in range(n_cells):
            divergence[ti, ci] = np.trace(jac[ci])

            # Eigenvalues
            eigvals = np.linalg.eigvals(jac[ci])
            eigenvalue_real[ti, ci] = eigvals.real
            eigenvalue_imag[ti, ci] = eigvals.imag

    # Bifurcation score: max |divergence| along trajectory per cell
    bifurcation_score = np.max(np.abs(divergence), axis=0)  # [n_cells]

    # Saddle points: timepoints with mixed-sign eigenvalue real parts
    n_saddle_points = np.zeros(n_cells, dtype=int)
    for ci in range(n_cells):
        for ti in range(n_timepoints):
            real_parts = eigenvalue_real[ti, ci]
            if np.any(real_parts > 0) and np.any(real_parts < 0):
                n_saddle_points[ci] += 1

    return {
        "divergence": divergence,
        "bifurcation_score": bifurcation_score,
        "eigenvalue_real": eigenvalue_real,
        "eigenvalue_imag": eigenvalue_imag,
        "timepoints": timepoints,
        "n_saddle_points": n_saddle_points,
    }


def _build_mask(adata, filters):
    """Build boolean mask from obs column filters.

    Parameters
    ----------
    adata : AnnData
    filters : dict
        {obs_column: value}

    Returns
    -------
    np.ndarray [n_cells] boolean
    """
    mask = np.ones(adata.n_obs, dtype=bool)
    for col, val in filters.items():
        if col not in adata.obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs")
        mask &= (adata.obs[col] == val).values

    if not mask.any():
        raise ValueError(
            f"No cells match filter {filters}. Check obs column values."
        )

    return mask


# ---------------------------------------------------------------------------
# Feature graph functions (Jacobian-based directed gene interaction graphs)
# ---------------------------------------------------------------------------


def flow_feature_graph(
    adata: AnnData,
    flow_result: dict,
    flow_model,
    *,
    n_top_genes: int = 200,
    n_timepoints: int = 20,
    n_eval_points: int = 300,
    edge_threshold: float | None = None,
    random_state: int = 42,
) -> dict:
    """Static feature coupling graph collapsed over time.

    Computes a directed gene interaction graph from the Jacobian of a trained
    flow model. For each timepoint, the mean Jacobian is projected from PCA
    space into gene space via the PCA loadings triple product
    ``L_sub @ J_mean(t) @ L_sub.T``. The per-timepoint gene-space matrices
    are averaged to produce a single directed adjacency matrix where entry
    ``G[i, j]`` quantifies how much gene j's direction drives gene i's
    expansion through the flow.

    Parameters
    ----------
    adata : AnnData
        Must contain ``adata.varm['PCs']`` (PCA loadings).
    flow_result : dict
        Output of :func:`flow_within` (must include ``source_mask``,
        ``pca_key``).
    flow_model : FlowModel
        Trained flow model with ``.jacobian(points, t)`` method.
    n_top_genes : int
        Number of top genes (by alignment score magnitude) to retain.
    n_timepoints : int
        Number of evenly spaced timepoints in [0, 1] for Jacobian evaluation.
    n_eval_points : int
        Number of source cells to subsample for Jacobian computation.
    edge_threshold : float or None
        Absolute threshold for sparsifying the adjacency matrix. If None,
        the top 5% of ``|G_total|`` entries are kept.
    random_state : int
        Seed for reproducible subsampling.

    Returns
    -------
    dict
        Keys: ``adjacency_matrix``, ``gene_names``, ``gene_indices``,
        ``out_centrality``, ``in_centrality``, ``flow_centrality``,
        ``top_hub_genes``, ``n_timepoints``, ``n_top_genes``,
        ``edge_threshold``, ``per_timepoint_jacobians``.
    """
    rng = np.random.default_rng(random_state)

    # --- PCA loadings and gene names ---
    if "PCs" not in adata.varm:
        raise ValueError("adata.varm['PCs'] not found.")
    loadings = adata.varm["PCs"]  # [n_genes, n_pcs]
    gene_names = np.array(adata.var_names)

    # --- Pre-filter to top genes by alignment score ---
    alignment = flow_gene_alignment(adata, flow_result)
    scores = np.abs(alignment["alignment_scores"])
    n_top = min(n_top_genes, len(scores))
    top_idx = np.argsort(scores)[-n_top:][::-1]  # descending by |score|
    top_idx = np.sort(top_idx)  # restore original ordering for consistency

    gene_names_sub = gene_names[top_idx]
    pca_key = flow_result["pca_key"]
    n_pcs = adata.obsm[pca_key].shape[1]
    L_sub = loadings[top_idx, :n_pcs]  # [n_top, n_pcs]

    # --- Subsample source cells for evaluation ---
    source_pca = adata.obsm[pca_key][flow_result["source_mask"]]
    n_source = len(source_pca)
    n_eval = min(n_eval_points, n_source)
    eval_idx = rng.choice(n_source, size=n_eval, replace=False)
    eval_points = source_pca[eval_idx]

    # --- Transport evaluation points along the flow trajectory ---
    # n_steps = n_timepoints - 1 so that trajectory has exactly n_timepoints
    # frames matching np.linspace(0, 1, n_timepoints).
    trajectory = flow_model.transport(
        eval_points, n_steps=n_timepoints - 1, return_trajectory=True
    )  # [n_timepoints, n_eval, dim]

    # --- Compute per-timepoint mean Jacobians and gene-space projections ---
    timepoints = np.linspace(0.0, 1.0, n_timepoints)
    per_tp_jacobians = np.zeros((n_timepoints, n_pcs, n_pcs))
    G_per_tp = np.zeros((n_timepoints, n_top, n_top))

    print(f"Computing Jacobians at {n_timepoints} timepoints "
          f"({n_eval} evaluation points, {n_top} genes)...")

    for ti, t_val in enumerate(timepoints):
        # Evaluate Jacobian at transported positions for this timepoint
        traj_positions = trajectory[ti]  # [n_eval, dim] — positions at time t
        jac = flow_model.jacobian(traj_positions, float(t_val))  # [n_eval, dim, dim]
        J_mean = jac.mean(axis=0)  # [dim, dim]
        per_tp_jacobians[ti] = J_mean

        # Project to gene space: G(t) = L_sub @ J_mean @ L_sub.T
        G_per_tp[ti] = L_sub @ J_mean @ L_sub.T

        print(f"  timepoint {ti + 1}/{n_timepoints} (t={t_val:.3f}) done")

    # --- Integrate over time ---
    G_total = G_per_tp.mean(axis=0)  # [n_top, n_top]

    # --- Sparsify ---
    abs_G = np.abs(G_total)
    if edge_threshold is None:
        # Top 5% of entries
        threshold = np.percentile(abs_G, 95)
    else:
        threshold = edge_threshold

    G_sparse = np.where(abs_G >= threshold, G_total, 0.0)

    # --- Centrality measures ---
    out_centrality = abs_G.sum(axis=1)      # row sum: gene i drives others
    in_centrality = abs_G.sum(axis=0)       # col sum: gene j is driven
    flow_centrality = out_centrality * in_centrality

    # Top hub genes
    hub_idx = np.argsort(flow_centrality)[-20:][::-1]
    top_hub_genes = list(gene_names_sub[hub_idx])

    print(f"Feature graph complete. Top 5 hub genes: {top_hub_genes[:5]}")

    result_dict = {
        "adjacency_matrix": G_sparse,
        "gene_names": list(gene_names_sub),
        "gene_indices": top_idx,
        "out_centrality": out_centrality,
        "in_centrality": in_centrality,
        "flow_centrality": flow_centrality,
        "top_hub_genes": top_hub_genes,
        "n_timepoints": n_timepoints,
        "n_top_genes": n_top,
        "edge_threshold": float(threshold),
        "per_timepoint_jacobians": per_tp_jacobians,
    }

    # Optional igraph construction
    try:
        import igraph as ig
        g = ig.Graph.Weighted_Adjacency(
            np.abs(G_sparse).tolist(), mode="directed"
        )
        g.vs["name"] = list(gene_names_sub)
        result_dict["igraph"] = g
    except ImportError:
        result_dict["igraph"] = None

    # Per-archetype hub genes via regression coefficient assignment
    hub_genes_per_archetype = {}
    reg = adata.uns.get("peach_simplex_regression") or adata.uns.get(
        "peach_simplex_regression_genes"
    )
    if reg is not None:
        coefs = np.asarray(reg["vertex_coefficients"])
        gene_names_all = list(reg["feature_names"])
        K = coefs.shape[1]
        for k in range(K):
            hub_genes_per_archetype[k] = []
        for gene in top_hub_genes:
            if gene in gene_names_all:
                gi = gene_names_all.index(gene)
                dominant_k = int(np.argmax(np.abs(coefs[gi])))
                hub_genes_per_archetype[dominant_k].append(gene)
    elif "cell_archetype_weights" in adata.obsm:
        K = adata.obsm["cell_archetype_weights"].shape[1]
        for k in range(K):
            hub_genes_per_archetype[k] = top_hub_genes[:10]
    result_dict["hub_genes_per_archetype"] = hub_genes_per_archetype

    return result_dict


def flow_temporal_feature_graph(
    adata: AnnData,
    flow_result: dict,
    flow_model,
    *,
    n_top_genes: int = 200,
    n_timepoints: int = 20,
    n_eval_points: int = 300,
    random_state: int = 42,
) -> dict:
    """Temporal feature graph with spatiotemporal nodes.

    Like :func:`flow_feature_graph`, but retains the full temporal structure
    rather than collapsing over time. Each node is a ``(gene, timepoint)``
    pair, with temporal backbone edges (gene self-expansion across consecutive
    timepoints) and cross-feature edges (inter-gene coupling at each
    timepoint).

    Parameters
    ----------
    adata : AnnData
        Must contain ``adata.varm['PCs']`` (PCA loadings).
    flow_result : dict
        Output of :func:`flow_within`.
    flow_model : FlowModel
        Trained flow model.
    n_top_genes : int
        Number of top genes (by alignment score magnitude) to retain.
    n_timepoints : int
        Number of evenly spaced timepoints in [0, 1].
    n_eval_points : int
        Number of source cells to subsample for Jacobian computation.
    random_state : int
        Seed for reproducible subsampling.

    Returns
    -------
    dict
        Keys: ``cross_matrices``, ``self_expansion``, ``gene_names``,
        ``timepoints``, ``temporal_centrality``, ``temporal_profile``,
        ``top_early_genes``, ``top_mid_early_genes``,
        ``top_mid_late_genes``, ``top_late_genes``,
        ``n_timepoints``, ``n_top_genes``.
    """
    rng = np.random.default_rng(random_state)

    # --- PCA loadings and gene names ---
    if "PCs" not in adata.varm:
        raise ValueError("adata.varm['PCs'] not found.")
    loadings = adata.varm["PCs"]
    gene_names = np.array(adata.var_names)

    # --- Pre-filter to top genes by alignment score ---
    alignment = flow_gene_alignment(adata, flow_result)
    scores = np.abs(alignment["alignment_scores"])
    n_top = min(n_top_genes, len(scores))
    top_idx = np.argsort(scores)[-n_top:][::-1]
    top_idx = np.sort(top_idx)

    gene_names_sub = gene_names[top_idx]
    pca_key = flow_result["pca_key"]
    n_pcs = adata.obsm[pca_key].shape[1]
    L_sub = loadings[top_idx, :n_pcs]  # [n_top, n_pcs]

    # --- Subsample source cells ---
    source_pca = adata.obsm[pca_key][flow_result["source_mask"]]
    n_source = len(source_pca)
    n_eval = min(n_eval_points, n_source)
    eval_idx = rng.choice(n_source, size=n_eval, replace=False)
    eval_points = source_pca[eval_idx]

    # --- Transport evaluation points along the flow trajectory ---
    # n_steps = n_timepoints - 1 so that trajectory has exactly n_timepoints
    # frames matching np.linspace(0, 1, n_timepoints).
    trajectory = flow_model.transport(
        eval_points, n_steps=n_timepoints - 1, return_trajectory=True
    )  # [n_timepoints, n_eval, dim]

    # --- Per-timepoint cross-term matrices and self-expansion ---
    timepoints = np.linspace(0.0, 1.0, n_timepoints)
    cross_matrices = np.zeros((n_timepoints, n_top, n_top))
    self_expansion = np.zeros((n_timepoints, n_top))

    print(f"Computing temporal feature graph at {n_timepoints} timepoints "
          f"({n_eval} evaluation points, {n_top} genes)...")

    for ti, t_val in enumerate(timepoints):
        # Evaluate Jacobian at transported positions for this timepoint
        traj_positions = trajectory[ti]  # [n_eval, dim] — positions at time t
        jac = flow_model.jacobian(traj_positions, float(t_val))  # [n_eval, dim, dim]
        J_mean = jac.mean(axis=0)  # [dim, dim]

        # Gene-space projection
        G_t = L_sub @ J_mean @ L_sub.T  # [n_top, n_top]
        cross_matrices[ti] = G_t
        self_expansion[ti] = np.diag(G_t)

        print(f"  timepoint {ti + 1}/{n_timepoints} (t={t_val:.3f}) done")

    # --- Temporal importance profile ---
    # Per-gene importance at each timepoint: |self-expansion| + mean |cross-terms|
    abs_cross = np.abs(cross_matrices)
    # For each gene g at timepoint t:
    #   backbone weight = |self_expansion[t, g]|
    #   cross weight = mean(|G_t[g, :]|) + mean(|G_t[:, g]|)  (outgoing + incoming)
    backbone_weight = np.abs(self_expansion)  # [n_timepoints, n_top]
    cross_out = abs_cross.sum(axis=2)  # [n_timepoints, n_top] — row sums
    cross_in = abs_cross.sum(axis=1)   # [n_timepoints, n_top] — col sums
    temporal_profile = backbone_weight + cross_out + cross_in  # [n_timepoints, n_top]

    # Overall temporal centrality: sum across all timepoints
    temporal_centrality = temporal_profile.sum(axis=0)  # [n_top]

    # --- Phase-specific top genes (4 temporal bins) ---
    early_mask = timepoints < 0.25
    mid_early_mask = (timepoints >= 0.25) & (timepoints < 0.5)
    mid_late_mask = (timepoints >= 0.5) & (timepoints < 0.75)
    late_mask = timepoints >= 0.75

    def _top_genes_for_phase(phase_mask, k=10):
        if not phase_mask.any():
            return []
        phase_importance = temporal_profile[phase_mask].sum(axis=0)
        n_return = min(k, n_top)
        idx = np.argsort(phase_importance)[-n_return:][::-1]
        return list(gene_names_sub[idx])

    top_early = _top_genes_for_phase(early_mask)
    top_mid_early = _top_genes_for_phase(mid_early_mask)
    top_mid_late = _top_genes_for_phase(mid_late_mask)
    top_late = _top_genes_for_phase(late_mask)

    print(f"Temporal feature graph complete.")
    print(f"  Top early genes     (t<0.25):       {top_early[:5]}")
    print(f"  Top mid-early genes (0.25<=t<0.5):  {top_mid_early[:5]}")
    print(f"  Top mid-late genes  (0.5<=t<0.75):  {top_mid_late[:5]}")
    print(f"  Top late genes      (t>=0.75):       {top_late[:5]}")

    return {
        "cross_matrices": cross_matrices,
        "self_expansion": self_expansion,
        "gene_names": list(gene_names_sub),
        "timepoints": timepoints,
        "temporal_centrality": temporal_centrality,
        "temporal_profile": temporal_profile,
        "top_early_genes": top_early,
        "top_mid_early_genes": top_mid_early,
        "top_mid_late_genes": top_mid_late,
        "top_late_genes": top_late,
        "n_timepoints": n_timepoints,
        "n_top_genes": n_top,
    }

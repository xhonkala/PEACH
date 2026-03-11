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
    solver_method: str = "euler",
    name: str | None = None,
    random_state: int = 42,
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
        ODE solver method: 'euler', 'midpoint', 'heun3', 'dopri5'.
    name : str or None
        Name for storage key.
    random_state : int
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

    # Train flow model
    import torch
    torch.manual_seed(random_state)
    model = FlowModel(dim, hidden_dims=hidden_dims, lr=lr,
                      solver_method=solver_method, device=device)
    losses = model.train(source_pca, target_pca, n_epochs=n_epochs, batch_size=batch_size)

    # Transport
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
    solver_method: str = "euler",
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

    return {
        "alignment_scores": alignment_scores,
        "gene_names": gene_names,
        "top_aligned": top_aligned,
        "top_opposed": top_opposed,
        "t": t,
    }


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
    jac_det = np.array([np.linalg.det(j) for j in jac])

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

    return {
        "jacobian_det": jac_det,
        "feature_expansion": feature_expansion,
        "mean_jacobian": mean_jac,
        "t": t,
    }


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

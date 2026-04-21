"""Flow matching public API: within-model, between-model, gene alignment, Jacobian."""

import logging

import numpy as np
from anndata import AnnData

from peach._core.utils.feature_utils import store_result
from peach._core.utils.flow_matching import FlowModel, compute_mmd

logger = logging.getLogger(__name__)


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
    model = FlowModel(dim, hidden_dims=hidden_dims, lr=lr,
                      solver_method=solver_method, device=device,
                      random_state=random_state)
    losses = model.train(source_train, target_pca, n_epochs=n_epochs,
                         batch_size=batch_size, use_ot=use_ot,
                         random_state=random_state)

    # Transport full source (not just training subset)
    transported = model.transport(source_pca, n_steps=n_steps)

    # MMD
    mmd_before = compute_mmd(source_pca, target_pca)
    mmd_after = compute_mmd(transported, target_pca)

    result = {
        "source_mask": source_mask,
        "target_mask": target_mask,
        "source_obs_names": adata.obs_names[source_mask].tolist(),
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

    # Add condition labels without copying expression matrices
    adatas_copy = []
    for a, label in zip(adatas, condition_labels):
        a_copy = a.copy(copy_X=False)
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
            random_state=hash((random_state, pair_idx)) % (2 ** 31),
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
    t: float | None = None,
    n_top: int = 50,
    pca_loadings_key: str | None = None,
    n_permutations: int = 0,
    null_type: str = "both",
    null_mode: str | None = None,
    per_cell: bool = True,
    n_top_features: int = 2500,
    normalize: bool = True,
    random_state: int = 42,
) -> dict:
    """Compute gene alignment with flow velocity.

    Parameters
    ----------
    adata : AnnData
    flow_result : FlowWithinResult
    t : float or None
        Time point to evaluate velocity. When ``t`` is not None and
        ``flow_result`` contains a trained model (``return_model=True``
        in ``flow_within``), the instantaneous velocity at time ``t``
        is used. When ``t`` is None (default), full-trajectory
        displacement is used (original behavior).
    n_top : int
        Top aligned/opposed genes to report.
    pca_loadings_key : str or None
        Key in adata.varm for PCA loadings. Default: 'PCs'.
    n_permutations : int
        Number of permutations for null distribution. Default: 0 (disabled).
    null_type : str
        Which null model(s) to run when ``n_permutations > 0``. One of
        ``"rotation"`` (random orthogonal rotation of the PCA loading matrix),
        ``"shuffle"`` (row-wise shuffle of gene-to-loading assignments), or
        ``"both"`` (run both and return results for each). Default: ``"both"``.

        - **Rotation null** tests: given the gene-gene correlation structure
          encoded in the loadings, is the observed alignment with the flow
          direction greater than chance? Preserves the covariance geometry of
          the loadings while randomising the coordinate frame. Produces an
          omnibus p-value (``rotation_omnibus_pvalue``).
        - **Shuffle null** tests: is THIS specific gene's loading aligned with
          flow, beyond what a random gene in its place would achieve? Breaks
          gene-gene correlation structure but enables per-gene testing. Produces
          per-gene p-values (``alignment_pvalues``, ``alignment_pvalues_fdr``).
    null_mode : str or None
        Alias for ``null_type``. When provided, overrides ``null_type``.
        Accepts the same values: ``"rotation"``, ``"shuffle"``, ``"both"``.
        Default: ``None`` (fall back to ``null_type``).
    per_cell : bool
        If True, also compute per-cell per-gene alignment scores for the top
        ``n_top_features`` genes (by absolute aggregated score). Returns
        additional keys ``'per_cell_alignment'`` with shape
        ``[n_source, n_top_features]``, ``'per_cell_gene_names'``, and
        ``'per_cell_gene_indices'``. Default: True.
    n_top_features : int
        Maximum number of genes to include in the per-cell alignment matrix.
        Genes are selected by absolute aggregated alignment score. Default: 2500.
    normalize : bool
        Deprecated. Alignment scores always use cosine similarity (unit-vector
        dot product) for both aggregate and per-cell scores. Passing
        ``normalize=False`` emits a DeprecationWarning and is ignored.

    Note
    ----
    Gene scores are mediated through PCA loadings. Genes with low variance
    explained by the top PCA components will have near-zero scores regardless
    of their biological relevance to the flow.
    random_state : int
        Random seed for permutation tests. Default: 42.
    """
    _validate_source_obs_names(adata, flow_result)
    # null_mode overrides null_type when provided (alias for API consistency)
    if null_mode is not None:
        null_type = null_mode

    if not normalize:
        import warnings
        warnings.warn(
            "normalize=False is deprecated in flow_gene_alignment. "
            "Alignment scores always use cosine similarity. "
            "This argument will be removed in v0.6.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Get PCA loadings
    if pca_loadings_key is None:
        pca_loadings_key = "PCs"
    if pca_loadings_key not in adata.varm:
        raise ValueError(f"adata.varm['{pca_loadings_key}'] not found.")

    loadings = adata.varm[pca_loadings_key]  # [n_genes, n_PCs]
    gene_names = list(adata.var_names)

    # Compute velocity vector(s)
    source_pca = adata.obsm[flow_result["pca_key"]][flow_result["source_mask"]]
    model = flow_result.get("model")

    if t is not None and model is not None:
        mean_velocity = model.velocity_at(source_pca, t).mean(axis=0)
        velocity_mode = "instantaneous"
    else:
        if t is not None and model is None:
            import warnings
            warnings.warn(
                f"t={t} specified but flow_result has no model (call flow_within "
                f"with return_model=True). Falling back to full-trajectory displacement.",
                UserWarning,
            )
        mean_velocity = (flow_result["transported"] - source_pca).mean(axis=0)
        velocity_mode = "displacement"

    # Trim loadings and normalize — always cosine similarity
    n_pcs = len(mean_velocity)
    loadings_trimmed = loadings[:, :n_pcs]
    loading_norms = np.linalg.norm(loadings_trimmed, axis=1, keepdims=True)
    loadings_for_agg = loadings_trimmed / np.maximum(loading_norms, 1e-10)
    vel_norm_agg = mean_velocity / (np.linalg.norm(mean_velocity) + 1e-10)
    alignment_scores = loadings_for_agg @ vel_norm_agg  # [n_genes]

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
        "velocity_mode": velocity_mode,
    }

    if per_cell:
        # Select top genes by aggregated alignment score
        n_top_feat = min(n_top_features, len(gene_names))
        top_feat_idx = np.argsort(np.abs(alignment_scores))[-n_top_feat:][::-1]
        top_feat_idx = np.sort(top_feat_idx)  # restore original ordering
        top_feat_names = [gene_names[i] for i in top_feat_idx]
        loadings_top = loadings_trimmed[top_feat_idx]  # [n_top_feat, n_pcs]

        if t is not None and model is not None:
            velocity_per_cell = model.velocity_at(source_pca, t)
        else:
            velocity_per_cell = flow_result["transported"] - source_pca

        # Always normalize per-cell (cosine similarity)
        vel_norm_pc = velocity_per_cell / (
            np.linalg.norm(velocity_per_cell, axis=1, keepdims=True) + 1e-10
        )
        load_norm_pc = loadings_top / (
            np.linalg.norm(loadings_top, axis=1, keepdims=True) + 1e-10
        )
        per_cell_alignment = vel_norm_pc @ load_norm_pc.T  # [n_source, n_top_feat]
        result["per_cell_alignment"] = per_cell_alignment
        result["per_cell_gene_names"] = top_feat_names
        result["per_cell_gene_indices"] = top_feat_idx

    if n_permutations > 0:
        from statsmodels.stats.multitest import multipletests
        rng = np.random.default_rng(random_state)
        n_pcs_perm = loadings_trimmed.shape[1]
        n_genes = len(alignment_scores)

        run_rotation = null_type in ("rotation", "both")
        run_shuffle = null_type in ("shuffle", "both")

        # --- Rotation null: random orthogonal rotation of loading matrix ---
        # Omnibus test — preserves gene-gene correlation, randomizes PC frame.
        if run_rotation:
            rot_null_scores = np.zeros((n_permutations, n_genes))
            for i in range(n_permutations):
                Z = rng.standard_normal((n_pcs_perm, n_pcs_perm))
                Q, _ = np.linalg.qr(Z)
                rotated = loadings_trimmed @ Q
                rot_norms = np.linalg.norm(rotated, axis=1, keepdims=True)
                rot_null_scores[i] = (rotated / np.maximum(rot_norms, 1e-10)) @ vel_norm_agg

            obs_max = np.max(np.abs(alignment_scores))
            null_maxes = np.max(np.abs(rot_null_scores), axis=1)
            omnibus_p = (np.sum(null_maxes >= obs_max) + 1) / (n_permutations + 1)

            result["rotation_omnibus_pvalue"] = omnibus_p
            result["rotation_null_mean"] = rot_null_scores.mean(axis=0)
            result["rotation_null_std"] = rot_null_scores.std(axis=0)

        # --- Shuffle null: permute gene-to-loading assignments ---
        # Per-gene test — breaks gene-gene correlation but tests gene identity.
        if run_shuffle:
            shuf_null_scores = np.zeros((n_permutations, n_genes))
            perm_indices = np.array([rng.permutation(n_genes) for _ in range(n_permutations)])
            perm_loadings_batch = loadings_for_agg[perm_indices]  # [n_perms, n_genes, n_pcs]
            shuf_null_scores = perm_loadings_batch @ vel_norm_agg  # [n_perms, n_genes]

            shuf_pvalues = np.array([
                (np.sum(np.abs(shuf_null_scores[:, g]) >= np.abs(alignment_scores[g])) + 1)
                / (n_permutations + 1)
                for g in range(n_genes)
            ])

            # BH FDR over the full gene family — pre-filtering introduces
            # selection bias and is not formally justified.
            _, shuf_pvalues_fdr, _, _ = multipletests(shuf_pvalues, method="fdr_bh")

            result["alignment_pvalues"] = shuf_pvalues
            result["alignment_pvalues_fdr"] = shuf_pvalues_fdr
            result["alignment_fdr_n_tested"] = n_genes
            result["null_mean"] = shuf_null_scores.mean(axis=0)
            result["null_std"] = shuf_null_scores.std(axis=0)

        # --- Rank-based null: only valid when rotation null is available ---
        # Shuffle null is degenerate for rank testing (produces all p=1.0).
        if run_shuffle and not run_rotation:
            import warnings
            warnings.warn(
                "Rank-based permutation null requires null_type='rotation' or 'both'. "
                "With null_type='shuffle' the rank null is degenerate (all p=1.0) "
                "and has been skipped.",
                UserWarning,
                stacklevel=3,
            )
        if run_rotation:
            abs_obs = np.abs(alignment_scores)
            obs_rank = (-abs_obs).argsort().argsort()
            sorted_null = np.sort(np.abs(rot_null_scores), axis=1)[:, ::-1]
            rank_pvalues = np.array([
                (np.sum(sorted_null[:, obs_rank[g]] >= abs_obs[g]) + 1)
                / (n_permutations + 1)
                for g in range(n_genes)
            ])
            _, rank_pvalues_fdr, _, _ = multipletests(rank_pvalues, method="fdr_bh")
            result["alignment_pvalues_rank"] = rank_pvalues
            result["alignment_pvalues_rank_fdr"] = rank_pvalues_fdr
            result["alignment_rank_null_mean"] = sorted_null.mean(axis=0)
            result["alignment_rank_null_std"] = sorted_null.std(axis=0)

        result["null_type"] = null_type

    return result


def flow_jacobian(
    adata: AnnData,
    flow_result: dict,
    flow_model: "FlowModel",
    *,
    t: "float | list[float]" = 0.5,
    n_steps: int = 100,
    pca_loadings_key: str | None = None,
    aggregate: str = "mean",
    per_cell_features: bool = True,
    n_top_features: int = 2500,
    n_permutations: int = 0,
    null_type: str = "both",
    null_mode: str | None = None,
    permutation_seed: int = 42,
) -> dict:
    """Compute flow map Jacobian ∂φ_t/∂x₀ via coupled ODE integration.

    Integrates the position ODE (dφ/dt = v(φ,t)) and variational ODE
    (dJ/dt = (∂v/∂x)|_φ @ J) simultaneously from t=0, starting each source
    cell at its PCA position. At each requested time t, J(t) gives the
    accumulated stretching/compression of gene-program neighborhoods by the
    full transport up to that point.

    Uses the midpoint (RK2) fixed-step solver. Permutation tests
    (``n_permutations > 0``) are run on the last (or only) requested timepoint.

    Parameters
    ----------
    adata : AnnData
    flow_result : dict
        Output of :func:`flow_within`. Must contain ``source_mask`` and
        ``pca_key``.
    flow_model : FlowModel
        The trained FlowModel. **Must** be the same model that produced
        ``flow_result`` — passing a mismatched model produces silently wrong
        results. Obtain via ``flow_within(..., return_model=True)`` and access
        as ``flow_result['model']``.
    t : float or list of float
        Time point(s) in (0, 1] at which to return J. When a single float,
        result keys have the same shape as before (no leading time dimension).
        When a list, result arrays gain a leading ``n_t`` dimension and the
        ``timepoints`` key is populated. Default: 0.5.
    n_steps : int
        Midpoint integration steps over [0, 1]. Step size = 1/n_steps.
        Default: 50.
    pca_loadings_key : str or None
        Key in ``adata.varm`` for PCA loadings used to project the Jacobian
        back to gene space. Default: ``'PCs'``.
    aggregate : str
        How to aggregate per-cell Jacobians into a single matrix for
        feature expansion: ``'mean'`` or ``'median'``. Default: ``'mean'``.
    per_cell_features : bool
        If True and PCA loadings are available, compute per-cell per-gene
        expansion (quadratic form L_g^T J_c L_g) for the top
        ``n_top_features`` genes. Returns ``'per_cell_expansion'``
        ``[n_cells, n_top_features]``, ``'per_cell_expansion_gene_names'``,
        and ``'per_cell_expansion_gene_indices'``. Default: True.
    n_top_features : int
        Maximum genes for per-cell expansion matrix. Selected by absolute
        aggregate feature expansion. Default: 2500.
    n_permutations : int
        Permutations for expansion significance (see ``null_type``). When > 0,
        the Jacobian J is fixed; only loading vectors are permuted, making
        this cheap. Applied to the last requested timepoint. Default: 0.
    null_type : str
        Which null(s) to run: ``'rotation'`` (omnibus, random orthogonal
        rotation of loadings), ``'shuffle'`` (per-gene, row-wise permutation),
        or ``'both'``. Default: ``'both'``.
    null_mode : str or None
        Alias for ``null_type``; overrides it when provided. Default: None.
    permutation_seed : int
        RNG seed for permutations. Default: 42.

    Note
    ----
    Gene scores are mediated through PCA loadings. Genes with low variance
    explained by the top PCA components will have near-zero scores regardless
    of their biological relevance to the flow.

    Returns
    -------
    dict
        When ``t`` is a float:

        - ``jacobian_det`` : [n_cells] — det(J) at t
        - ``jac_logdet`` : [n_cells] — log|det(J)| at t
        - ``jac_det_sign`` : [n_cells] — sign of det(J)
        - ``feature_expansion`` : [n_genes] — L^T mean_J L for each gene
        - ``mean_jacobian`` : [dim, dim] — aggregate J over cells
        - ``phi`` : [n_cells, dim] — transported positions at t
        - ``t`` : float

        When ``t`` is a list, all array keys gain a leading ``n_t``
        dimension and ``timepoints`` : list[float] is added.

        Optional keys (when ``per_cell_features=True`` and loadings present):

        - ``per_cell_expansion`` : [n_cells, n_top_features]
        - ``per_cell_expansion_gene_names`` : list[str]
        - ``per_cell_expansion_gene_indices`` : np.ndarray

        Optional keys (when ``n_permutations > 0``):

        - ``expansion_pvalues``, ``expansion_pvalues_fdr`` : [n_genes]
        - ``expansion_rotation_omnibus_pvalue`` : float
    """
    _validate_source_obs_names(adata, flow_result)
    scalar_t = isinstance(t, (int, float))
    t_eval = [float(t)] if scalar_t else [float(v) for v in t]

    if null_mode is not None:
        null_type = null_mode

    x0 = adata.obsm[flow_result["pca_key"]][flow_result["source_mask"]]

    # Integrate augmented ODE to get transported positions + flow map Jacobians
    fmj = flow_model.flow_map_jacobian(x0, t_eval, n_steps=n_steps)
    # fmj['phi']: [n_t, n_cells, dim]
    # fmj['J']:   [n_t, n_cells, dim, dim]

    if pca_loadings_key is None:
        pca_loadings_key = "PCs"
    has_loadings = pca_loadings_key in adata.varm

    if has_loadings:
        loadings = adata.varm[pca_loadings_key]
        n_pcs = fmj['J'].shape[-1]
        loadings_trimmed = loadings[:, :n_pcs]
        loading_norms = np.linalg.norm(loadings_trimmed, axis=1, keepdims=True)
        loading_norms = np.maximum(loading_norms, 1e-10)
        loadings_normalized = loadings_trimmed / loading_norms  # [n_genes, n_pcs]
        gene_names_all = list(adata.var_names) if hasattr(adata, 'var_names') else []

    # Compute feature expansion and per-cell gene selection from the LAST
    # (or only) timepoint's mean Jacobian. Using a single gene index set
    # across all timepoints ensures the stacked per_cell_expansion array has
    # a consistent gene axis.
    last_J = fmj['J'][-1]
    if aggregate == "median":
        last_mean_jac = np.median(last_J, axis=0)
    else:
        last_mean_jac = last_J.mean(axis=0)

    shared_top_feat_idx = None
    shared_top_feat_names = []
    if has_loadings and per_cell_features:
        last_feat_exp = np.einsum(
            'gi,ij,gj->g', loadings_normalized, last_mean_jac, loadings_normalized
        )
        n_top_feat = min(n_top_features, loadings.shape[0])
        shared_top_feat_idx = np.sort(
            np.argsort(np.abs(last_feat_exp))[-n_top_feat:]
        )
        shared_top_feat_names = [gene_names_all[i] for i in shared_top_feat_idx] if gene_names_all else []

    def _process_one_timepoint(jac_t):
        """Compute derived quantities for J at a single timepoint."""
        signs, logdets = np.linalg.slogdet(jac_t)
        jac_det = signs * np.exp(np.clip(logdets, -500, 500))

        if aggregate == "median":
            mean_jac = np.median(jac_t, axis=0)
        else:
            mean_jac = jac_t.mean(axis=0)

        out = {
            "jacobian_det": jac_det,
            "jac_logdet": logdets,
            "jac_det_sign": signs,
            "mean_jacobian": mean_jac,
        }

        if has_loadings:
            feature_expansion = np.einsum(
                'gi,ij,gj->g', loadings_normalized, mean_jac, loadings_normalized
            )
            out["feature_expansion"] = feature_expansion

            if per_cell_features and shared_top_feat_idx is not None:
                L_top = loadings_normalized[shared_top_feat_idx]
                per_cell_exp = np.einsum('gi,cij,gj->cg', L_top, jac_t, L_top)
                out["per_cell_expansion"] = per_cell_exp
                out["per_cell_expansion_gene_names"] = shared_top_feat_names
                out["per_cell_expansion_gene_indices"] = shared_top_feat_idx
        else:
            out["feature_expansion"] = np.zeros(0)

        return out

    # Process each timepoint
    per_t = [_process_one_timepoint(fmj['J'][i]) for i in range(len(t_eval))]

    # Permutation tests on the last (or only) timepoint
    if n_permutations > 0 and has_loadings:
        from peach._core.utils.permutation import fdr_correct, permutation_pvalue

        feature_expansion = per_t[-1].get("feature_expansion", np.zeros(0))
        mean_jac = per_t[-1]["mean_jacobian"]

        if len(feature_expansion) > 0:
            n_genes = loadings_normalized.shape[0]
            n_pcs_jac = loadings_normalized.shape[1]
            rng = np.random.default_rng(permutation_seed)

            run_rotation = null_type in ("rotation", "both")
            run_shuffle = null_type in ("shuffle", "both")

            perm_result = {}

            if run_rotation:
                rot_null = np.empty((n_permutations, n_genes))
                for p in range(n_permutations):
                    Z = rng.standard_normal((n_pcs_jac, n_pcs_jac))
                    Q, _ = np.linalg.qr(Z)
                    rotated = loadings_normalized @ Q
                    rot_null[p] = np.einsum('gi,ij,gj->g', rotated, mean_jac, rotated)
                obs_max = np.max(np.abs(feature_expansion))
                null_maxes = np.max(np.abs(rot_null), axis=1)
                perm_result["expansion_rotation_omnibus_pvalue"] = float(
                    (np.sum(null_maxes >= obs_max) + 1) / (n_permutations + 1)
                )

            if run_shuffle:
                shuf_null = np.empty((n_permutations, n_genes))
                perm_indices = np.array([rng.permutation(n_genes) for _ in range(n_permutations)])
                for p in range(n_permutations):
                    shuffled = loadings_normalized[perm_indices[p]]
                    shuf_null[p] = np.einsum('gi,ij,gj->g', shuffled, mean_jac, shuffled)

                perm_pvals = permutation_pvalue(
                    feature_expansion, shuf_null, alternative="two-sided"
                )
                # BH FDR over the full gene family — pre-filtering introduces
                # selection bias and is not formally justified.
                _, perm_fdr = fdr_correct(perm_pvals)
                n_raw_sig = int((perm_pvals < 0.01).sum())
                perm_result.update({
                    "expansion_pvalues": perm_pvals,
                    "expansion_pvalues_raw": perm_pvals,
                    "expansion_pvalues_fdr": perm_fdr,
                    "expansion_n_raw_significant": n_raw_sig,
                })

            perm_result["n_permutations"] = n_permutations
            perm_result["expansion_null_type"] = null_type
            per_t[-1].update(perm_result)

            logger.info(
                f"Jacobian permutation ({null_type}): "
                f"{perm_result.get('expansion_n_raw_significant', '?')}/{n_genes} "
                f"genes at raw p<0.01, "
                f"{(perm_result.get('expansion_pvalues_fdr', np.ones(1)) < 0.05).sum()}"
                f"/{n_genes} at FDR q<0.05 ({n_permutations} permutations)"
            )

    # Assemble result — scalar t: flatten to same structure as before (no time dim)
    if scalar_t:
        result = per_t[0]
        result["phi"] = fmj['phi'][0]
        result["t"] = t_eval[0]
    else:
        # Multi-timepoint: stack arrays along leading time dimension
        array_keys = ["jacobian_det", "jac_logdet", "jac_det_sign",
                      "mean_jacobian", "feature_expansion", "per_cell_expansion"]
        result = {"timepoints": t_eval}
        for k in array_keys:
            arrays = [pt[k] for pt in per_t if k in pt]
            if arrays:
                result[k] = np.stack(arrays, axis=0)
        # Non-array keys: take from last timepoint (gene names, indices, perm results)
        for k, v in per_t[-1].items():
            if k not in result and k not in array_keys:
                result[k] = v
        result["phi"] = fmj['phi']  # [n_t, n_cells, dim]
        result["t"] = t_eval

    return result


def flow_significance(
    adata: AnnData,
    flow_result: dict,
    *,
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

    Uses the original ``flow_result``'s MMD improvement as the observed
    statistic, then retrains flows on permuted condition labels to build
    a null distribution. This avoids retraining for the observed
    statistic, which would produce a different (and inconsistent) MMD.

    Parameters
    ----------
    adata : AnnData
    flow_result : dict
        Output of :func:`flow_within`. Must contain ``mmd_before`` and
        ``mmd_after`` keys.
    n_permutations : int
        Number of label-permuted null models to train.
    n_epochs_per_perm : int
        Epochs per null model.

    Returns
    -------
    dict with p_value, observed_stat, null_distribution
    """
    import torch

    # Use the original flow result's MMD improvement (no retraining)
    if "mmd_before" not in flow_result or "mmd_after" not in flow_result:
        raise ValueError(
            "flow_significance requires a flow_result dict with 'mmd_before' and "
            "'mmd_after' keys (from flow_within). Pass the flow_result directly."
        )
    observed_improvement = flow_result["mmd_before"] - flow_result["mmd_after"]

    source_mask = flow_result["source_mask"]
    target_mask = flow_result["target_mask"]
    pca_key = flow_result["pca_key"]

    pca = adata.obsm[pca_key]
    source_pca = pca[source_mask]
    target_pca = pca[target_mask]
    dim = source_pca.shape[1]

    # Null distribution: permute labels, retrain, measure improvement
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
        perm_model = FlowModel(dim, hidden_dims=hidden_dims, lr=lr,
                               solver_method=solver_method, device=device,
                               random_state=random_state + i)
        perm_model.train(perm_source, perm_target, n_epochs=n_epochs_per_perm, batch_size=batch_size)
        perm_transported = perm_model.transport(perm_source, n_steps=n_steps)

        mmd_improvement = compute_mmd(perm_source, perm_target) - compute_mmd(perm_transported, perm_target)
        null_stats.append(mmd_improvement)

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
        The trained FlowModel. **Must** be the same model that produced
        ``flow_result`` -- passing a mismatched model will produce silently
        wrong results. Use ``flow_within(..., return_model=True)`` and
        access via ``flow_result['model']``.
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

    # Generate a dense trajectory so each evaluation timepoint has a close
    # frame match. Using 10× the number of timepoints (min 100) keeps the
    # argmin error below 1/10 of the inter-timepoint spacing.
    n_traj_steps = max(n_timepoints * 10, 100)
    trajectory = flow_model.transport(
        evaluation_points, n_steps=n_traj_steps, return_trajectory=True
    )  # [n_traj_steps+1, n_cells, dim]
    traj_times = np.linspace(0, 1, trajectory.shape[0])

    divergence = np.zeros((n_timepoints, n_cells))
    eigenvalue_real = np.zeros((n_timepoints, n_cells, dim))
    eigenvalue_imag = np.zeros((n_timepoints, n_cells, dim))

    for ti, t_val in enumerate(timepoints):
        frame_idx = np.argmin(np.abs(traj_times - t_val))
        positions = trajectory[frame_idx]  # [n_cells, dim]

        jac = flow_model.jacobian(positions, float(t_val))  # [n_cells, dim, dim]
        divergence[ti] = np.trace(jac, axis1=1, axis2=2)

        eigvals = np.linalg.eigvals(jac)  # [n_cells, dim]
        eigenvalue_real[ti] = eigvals.real
        eigenvalue_imag[ti] = eigvals.imag

    bifurcation_score = np.max(np.abs(divergence), axis=0)  # [n_cells]

    return {
        "divergence": divergence,
        "bifurcation_score": bifurcation_score,
        "eigenvalue_real": eigenvalue_real,
        "eigenvalue_imag": eigenvalue_imag,
        "timepoints": timepoints,
    }


def _validate_source_obs_names(adata: "AnnData", flow_result: dict) -> None:
    """Raise if adata.obs_names don't match those used to build flow_result."""
    expected = flow_result.get("source_obs_names")
    if expected is None:
        return  # flow_result predates this guard — skip silently
    actual = adata.obs_names[flow_result["source_mask"]].tolist()
    if actual != expected:
        raise ValueError(
            "adata.obs_names do not match the AnnData used in flow_within. "
            "Pass the same adata object that was used to generate flow_result."
        )


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
        if not np.isscalar(val):
            raise ValueError(
                f"Filter value for '{col}' must be scalar; got {type(val).__name__}. "
                "For multi-value filtering, call flow_within separately per condition."
            )
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
    **kwargs,
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
        The trained FlowModel. **Must** be the same model that produced
        ``flow_result`` -- passing a mismatched model will produce silently
        wrong results. Use ``flow_within(..., return_model=True)`` and
        access via ``flow_result['model']``.
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
    import warnings
    warnings.warn(
        "flow_feature_graph is deprecated and will be removed in v0.6. "
        "Use flow_jacobian(per_cell_features=True) for gene expansion analysis. "
        "The velocity-Jacobian-based feature graph has been superseded by the "
        "flow map Jacobian approach in flow_jacobian.",
        DeprecationWarning,
        stacklevel=2,
    )
    raise NotImplementedError(
        "flow_feature_graph has been deprecated. "
        "Use pc.tl.flow_jacobian(per_cell_features=True) instead."
    )
    loadings = adata.varm["PCs"]  # [n_genes, n_pcs]
    gene_names = np.array(adata.var_names)

    # --- Pre-filter to top genes by alignment score ---
    alignment = flow_gene_alignment(adata, flow_result, per_cell=False)
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

    logger.info("Computing Jacobians at %d timepoints (%d eval points, %d genes)",
                n_timepoints, n_eval, n_top)

    for ti, t_val in enumerate(timepoints):
        # Evaluate Jacobian at transported positions for this timepoint
        traj_positions = trajectory[ti]  # [n_eval, dim] — positions at time t
        jac = flow_model.jacobian(traj_positions, float(t_val))  # [n_eval, dim, dim]
        J_mean = jac.mean(axis=0)  # [dim, dim]
        per_tp_jacobians[ti] = J_mean

        # Project to gene space: G(t) = L_sub @ J_mean @ L_sub.T
        G_per_tp[ti] = L_sub @ J_mean @ L_sub.T

        logger.debug("  timepoint %d/%d (t=%.3f) done", ti + 1, n_timepoints, t_val)

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

    # --- Centrality measures (from sparsified graph) ---
    abs_G_sparse = np.abs(G_sparse)
    out_centrality = abs_G_sparse.sum(axis=1)      # row sum: gene i drives others
    in_centrality = abs_G_sparse.sum(axis=0)       # col sum: gene j is driven
    flow_centrality = out_centrality * in_centrality

    # Top hub genes
    hub_idx = np.argsort(flow_centrality)[-20:][::-1]
    top_hub_genes = list(gene_names_sub[hub_idx])

    logger.info("Feature graph complete. Top 5 hub genes: %s", top_hub_genes[:5])

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
    archetype_pairs: list | None = None,
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
        The trained FlowModel. **Must** be the same model that produced
        ``flow_result`` -- passing a mismatched model will produce silently
        wrong results. Use ``flow_within(..., return_model=True)`` and
        access via ``flow_result['model']``.
    n_top_genes : int
        Number of top genes (by alignment score magnitude) to retain.
    n_timepoints : int
        Number of evenly spaced timepoints in [0, 1].
    n_eval_points : int
        Number of source cells to subsample for Jacobian computation.
    archetype_pairs : list of tuple[int, int] or None
        If provided, restrict evaluation to source cells whose two highest
        archetype weights correspond to one of the given pairs.  Requires
        ``adata.obsm['cell_archetype_weights']``.
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
    import warnings
    warnings.warn(
        "flow_temporal_feature_graph is deprecated and will be removed in v0.6. "
        "Use flow_jacobian(t=[...], per_cell_features=True) for multi-timepoint "
        "flow map Jacobian analysis.",
        DeprecationWarning,
        stacklevel=2,
    )
    raise NotImplementedError(
        "flow_temporal_feature_graph has been deprecated. "
        "Use pc.tl.flow_jacobian(t=[0.25, 0.5, 0.75], per_cell_features=True) instead."
    )
    rng = np.random.default_rng(random_state)

    # --- PCA loadings and gene names ---
    if "PCs" not in adata.varm:
        raise ValueError("adata.varm['PCs'] not found.")
    loadings = adata.varm["PCs"]
    gene_names = np.array(adata.var_names)

    # --- Pre-filter to top genes by alignment score ---
    alignment = flow_gene_alignment(adata, flow_result, per_cell=False)
    scores = np.abs(alignment["alignment_scores"])
    n_top = min(n_top_genes, len(scores))
    top_idx = np.argsort(scores)[-n_top:][::-1]
    top_idx = np.sort(top_idx)

    gene_names_sub = gene_names[top_idx]
    pca_key = flow_result["pca_key"]
    n_pcs = adata.obsm[pca_key].shape[1]
    L_sub = loadings[top_idx, :n_pcs]  # [n_top, n_pcs]

    # --- Subsample source cells (optionally filtered by archetype pairs) ---
    source_mask = flow_result["source_mask"]
    source_pca = adata.obsm[pca_key][source_mask]

    if archetype_pairs is not None:
        if "cell_archetype_weights" not in adata.obsm:
            raise ValueError(
                "archetype_pairs requires adata.obsm['cell_archetype_weights']"
            )
        weights_source = np.asarray(
            adata.obsm["cell_archetype_weights"][source_mask]
        )
        # For each cell, find its top-2 archetypes by weight
        top2 = np.argsort(weights_source, axis=1)[:, -2:]  # [n_source, 2]
        top2_sets = [frozenset(row) for row in top2]
        pair_sets = [frozenset(p) for p in archetype_pairs]
        keep = np.array([t2 in pair_sets for t2 in top2_sets])
        if keep.sum() < 10:
            import warnings
            warnings.warn(
                f"Only {keep.sum()} source cells match archetype_pairs "
                f"{archetype_pairs}; using all source cells instead.",
                UserWarning,
            )
        else:
            source_pca = source_pca[keep]
            logger.info("archetype_pairs filter: %d/%d source cells retained",
                        keep.sum(), len(keep))

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

    logger.info("Computing temporal feature graph at %d timepoints (%d eval points, %d genes)",
                n_timepoints, n_eval, n_top)

    for ti, t_val in enumerate(timepoints):
        # Evaluate Jacobian at transported positions for this timepoint
        traj_positions = trajectory[ti]  # [n_eval, dim] — positions at time t
        jac = flow_model.jacobian(traj_positions, float(t_val))  # [n_eval, dim, dim]
        J_mean = jac.mean(axis=0)  # [dim, dim]

        # Gene-space projection
        G_t = L_sub @ J_mean @ L_sub.T  # [n_top, n_top]
        cross_matrices[ti] = G_t
        self_expansion[ti] = np.diag(G_t)

        logger.debug("  timepoint %d/%d (t=%.3f) done", ti + 1, n_timepoints, t_val)

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

    logger.info("Temporal feature graph complete.")
    logger.info("  Top early genes     (t<0.25):       %s", top_early[:5])
    logger.info("  Top mid-early genes (0.25<=t<0.5):  %s", top_mid_early[:5])
    logger.info("  Top mid-late genes  (0.5<=t<0.75):  %s", top_mid_late[:5])
    logger.info("  Top late genes      (t>=0.75):       %s", top_late[:5])

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

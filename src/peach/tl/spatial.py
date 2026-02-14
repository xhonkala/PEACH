"""
Spatial archetype analysis functions.

Analyze spatial co-localization and neighborhood enrichment of archetypal
patterns using squidpy. Requires spatial transcriptomics data with tissue
coordinates in ``adata.obsm['spatial']`` and archetype assignments in
``adata.obs['archetypes']``.

Main Functions:
- spatial_neighbors(): Build spatial connectivity graph
- archetype_nhood_enrichment(): Test archetype co-localization via permutation
- archetype_co_occurrence(): Distance-dependent archetype co-occurrence
- archetype_spatial_autocorr(): Moran's I / Geary's C per archetype weight
- archetype_interaction_boundaries(): Detect cross-cell-type gradient fronts

Requires: ``pip install peach[spatial]`` (squidpy >= 1.3.0)
"""

from typing import Any

import numpy as np
from anndata import AnnData


def _check_squidpy():
    """Import squidpy with helpful error on failure."""
    try:
        import squidpy as sq

        return sq
    except ImportError:
        raise ImportError(
            "squidpy is required for spatial analysis. "
            "Install it with: pip install peach[spatial]  "
            "or: pip install squidpy"
        )


def _validate_spatial_data(adata: AnnData, spatial_key: str = "spatial", require_archetypes: bool = False):
    """Validate that adata has spatial coordinates and optionally archetype assignments."""
    if spatial_key not in adata.obsm:
        available = list(adata.obsm.keys())
        raise ValueError(
            f"No spatial coordinates found at adata.obsm['{spatial_key}']. "
            f"Available keys: {available}. "
            f"Spatial transcriptomics data must have 2D tissue coordinates."
        )

    coords = adata.obsm[spatial_key]
    if coords.shape[1] < 2:
        raise ValueError(f"Spatial coordinates must be at least 2D, got shape {coords.shape}")

    if require_archetypes:
        if "archetypes" not in adata.obs.columns:
            raise ValueError(
                "No archetype assignments found in adata.obs['archetypes']. "
                "Run pc.tl.assign_archetypes(adata) first."
            )


def _ensure_categorical(adata: AnnData, key: str):
    """Ensure an obs column is categorical (required by squidpy)."""
    if not hasattr(adata.obs[key], "cat"):
        adata.obs[key] = adata.obs[key].astype("category")


def spatial_neighbors(
    adata: AnnData,
    *,
    spatial_key: str = "spatial",
    n_neighs: int = 30,
    coord_type: str = "generic",
    **kwargs: Any,
) -> None:
    """Build spatial neighbor graph from tissue coordinates.

    Wrapper around ``squidpy.gr.spatial_neighbors()`` with PEACH-appropriate
    defaults for bead-based and imaging-based spatial transcriptomics.

    Parameters
    ----------
    adata : AnnData
        Annotated data with spatial coordinates in ``adata.obsm[spatial_key]``.
    spatial_key : str, default: "spatial"
        Key in ``adata.obsm`` containing 2D spatial coordinates.
    n_neighs : int, default: 30
        Number of nearest neighbors for the spatial graph.
    coord_type : str, default: "generic"
        Coordinate type. Use "generic" for Slide-seq/MERFISH (continuous coords)
        or "grid" for Visium (hexagonal grid). See squidpy docs for details.
    **kwargs
        Additional arguments passed to ``squidpy.gr.spatial_neighbors()``.

    Returns
    -------
    None
        Modifies ``adata`` in place:

        - ``adata.obsp['spatial_connectivities']``: sparse connectivity matrix
        - ``adata.obsp['spatial_distances']``: sparse distance matrix
    """
    sq = _check_squidpy()
    _validate_spatial_data(adata, spatial_key=spatial_key)

    n_cells = adata.n_obs
    print(f"  Building spatial neighbor graph: {n_cells} cells, {n_neighs} neighbors, coord_type='{coord_type}'")

    sq.gr.spatial_neighbors(
        adata,
        spatial_key=spatial_key,
        n_neighs=n_neighs,
        coord_type=coord_type,
        **kwargs,
    )

    print(f"  Spatial graph stored in adata.obsp['spatial_connectivities'] and adata.obsp['spatial_distances']")


def archetype_nhood_enrichment(
    adata: AnnData,
    *,
    cluster_key: str = "archetypes",
    n_perms: int = 1000,
    seed: int = 42,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    """Test spatial neighborhood enrichment between archetype groups.

    For each pair of archetypes, tests whether cells of one archetype are
    found more or less frequently in the spatial neighborhood of the other
    archetype than expected by chance (permutation test).

    Parameters
    ----------
    adata : AnnData
        Annotated data with spatial graph (run ``spatial_neighbors`` first)
        and archetype assignments in ``adata.obs[cluster_key]``.
    cluster_key : str, default: "archetypes"
        Column in ``adata.obs`` with archetype labels.
    n_perms : int, default: 1000
        Number of permutations for significance testing.
    seed : int, default: 42
        Random seed for permutation reproducibility.
    **kwargs
        Additional arguments passed to ``squidpy.gr.nhood_enrichment()``.

    Returns
    -------
    dict
        Dictionary with 'zscore' and 'count' arrays [n_archetypes x n_archetypes].
        Also stored in ``adata.uns['archetype_nhood_enrichment']``.

    Examples
    --------
    >>> pc.tl.spatial_neighbors(adata)
    >>> result = pc.tl.archetype_nhood_enrichment(adata)
    >>> print(result['zscore'])  # positive = enriched, negative = depleted
    """
    sq = _check_squidpy()

    if "spatial_connectivities" not in adata.obsp:
        raise ValueError("No spatial graph found. Run pc.tl.spatial_neighbors(adata) first.")

    if cluster_key not in adata.obs.columns:
        raise ValueError(
            f"'{cluster_key}' not found in adata.obs. "
            f"Run pc.tl.assign_archetypes(adata) first."
        )

    _ensure_categorical(adata, cluster_key)
    n_categories = adata.obs[cluster_key].nunique()
    print(f"  Computing neighborhood enrichment: {n_categories} archetype groups, {n_perms} permutations")

    # Default n_jobs=1 to avoid macOS multiprocessing spawn issues.
    # Users can override via kwargs if running on Linux or with proper guards.
    kwargs.setdefault("n_jobs", 1)

    sq.gr.nhood_enrichment(
        adata,
        cluster_key=cluster_key,
        seed=seed,
        n_perms=n_perms,
        **kwargs,
    )

    # squidpy stores results in adata.uns[f'{cluster_key}_nhood_enrichment']
    squidpy_key = f"{cluster_key}_nhood_enrichment"
    result = adata.uns.get(squidpy_key, {})

    # Also store under our standardized key
    adata.uns["archetype_nhood_enrichment"] = result

    if "zscore" in result:
        zscore = result["zscore"]
        print(f"  Enrichment z-scores range: [{zscore.min():.2f}, {zscore.max():.2f}]")

    return result


def archetype_co_occurrence(
    adata: AnnData,
    *,
    cluster_key: str = "archetypes",
    spatial_key: str = "spatial",
    interval: int = 50,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    """Compute distance-dependent co-occurrence of archetype groups.

    Measures how the co-occurrence ratio between archetype pairs varies
    with spatial distance. Useful for identifying distance-dependent
    spatial relationships (e.g., archetypes that co-occur at short but
    not long distances).

    Parameters
    ----------
    adata : AnnData
        Annotated data with spatial coordinates and archetype assignments.
    cluster_key : str, default: "archetypes"
        Column in ``adata.obs`` with archetype labels.
    spatial_key : str, default: "spatial"
        Key in ``adata.obsm`` with spatial coordinates.
    interval : int, default: 50
        Number of distance intervals to evaluate.
    **kwargs
        Additional arguments passed to ``squidpy.gr.co_occurrence()``.

    Returns
    -------
    dict
        Dictionary with 'occ' (co-occurrence ratios) and 'interval' (distance bins).
        Also stored in ``adata.uns['archetype_co_occurrence']``.

    Examples
    --------
    >>> pc.tl.spatial_neighbors(adata)
    >>> result = pc.tl.archetype_co_occurrence(adata)
    >>> print(result['occ'].shape)  # [n_archetypes, n_archetypes, n_intervals]
    """
    sq = _check_squidpy()
    _validate_spatial_data(adata, spatial_key=spatial_key)

    if cluster_key not in adata.obs.columns:
        raise ValueError(
            f"'{cluster_key}' not found in adata.obs. "
            f"Run pc.tl.assign_archetypes(adata) first."
        )

    _ensure_categorical(adata, cluster_key)
    n_categories = adata.obs[cluster_key].nunique()
    print(f"  Computing co-occurrence: {n_categories} archetype groups, {interval} distance intervals")

    # Default n_jobs=1 to avoid macOS multiprocessing spawn issues.
    kwargs.setdefault("n_jobs", 1)

    sq.gr.co_occurrence(
        adata,
        cluster_key=cluster_key,
        spatial_key=spatial_key,
        interval=interval,
        **kwargs,
    )

    # squidpy stores results in adata.uns[f'{cluster_key}_co_occurrence']
    squidpy_key = f"{cluster_key}_co_occurrence"
    result = adata.uns.get(squidpy_key, {})

    # Also store under our standardized key
    adata.uns["archetype_co_occurrence"] = result

    if "occ" in result:
        print(f"  Co-occurrence computed: {result['occ'].shape}")

    return result


def archetype_spatial_autocorr(
    adata: AnnData,
    *,
    weights_key: str = "cell_archetype_weights",
    mode: str = "moran",
    n_perms: int = 100,
    n_jobs: int = 1,
    **kwargs: Any,
):
    """Compute spatial autocorrelation (Moran's I or Geary's C) per archetype weight.

    Tests whether each archetype weight is spatially smooth (positive
    autocorrelation), spatially random, or spatially checkered (negative
    autocorrelation). Moran's I > 0 indicates spatial clustering of that
    archetype; I ~ 0 indicates random distribution; I < 0 indicates
    dispersal.

    Parameters
    ----------
    adata : AnnData
        Annotated data with spatial graph (run ``spatial_neighbors`` first)
        and archetype weights in ``adata.obsm[weights_key]``.
    weights_key : str, default: "archetype_weights"
        Key in ``adata.obsm`` with archetype weight matrix [n_cells, n_archetypes].
    mode : str, default: "moran"
        Autocorrelation statistic: "moran" (Moran's I) or "geary" (Geary's C).
    n_perms : int, default: 100
        Number of permutations for p-value estimation. Set to None for
        analytical p-values only.
    n_jobs : int, default: 1
        Number of parallel jobs. Default 1 for macOS compatibility.
    **kwargs
        Additional arguments passed to ``squidpy.gr.spatial_autocorr()``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with autocorrelation statistics per archetype weight.
        Also stored in ``adata.uns['archetype_spatial_autocorr']``.

    Examples
    --------
    >>> pc.tl.spatial_neighbors(adata)
    >>> autocorr = pc.tl.archetype_spatial_autocorr(adata)
    >>> print(autocorr)  # Moran's I per archetype
    """
    import pandas as pd

    sq = _check_squidpy()

    if "spatial_connectivities" not in adata.obsp:
        raise ValueError("No spatial graph found. Run pc.tl.spatial_neighbors(adata) first.")

    if weights_key not in adata.obsm:
        raise ValueError(
            f"No archetype weights found at adata.obsm['{weights_key}']. "
            f"Run pc.tl.extract_archetype_weights(adata) first."
        )

    W = adata.obsm[weights_key]
    n_arch = W.shape[1]
    print(f"  Computing spatial autocorrelation ({mode}): {n_arch} archetype weights")

    # Temporarily store archetype weights as obs columns for squidpy
    arch_cols = []
    for i in range(n_arch):
        col = f"_arch_weight_{i}"
        adata.obs[col] = W[:, i]
        arch_cols.append(col)

    try:
        sq.gr.spatial_autocorr(
            adata,
            mode=mode,
            attr="obs",
            genes=arch_cols,
            n_perms=n_perms,
            n_jobs=n_jobs,
            **kwargs,
        )

        # squidpy stores results in adata.uns['moranI'] or adata.uns['gearyC']
        result_key = "moranI" if mode == "moran" else "gearyC"
        if result_key in adata.uns:
            result_df = adata.uns[result_key].loc[arch_cols].copy()
            # Rename index from _arch_weight_0 to archetype_0
            result_df.index = [f"archetype_{i}" for i in range(n_arch)]
            adata.uns["archetype_spatial_autocorr"] = result_df

            stat_col = "I" if mode == "moran" else "C"
            print(f"\n  Spatial autocorrelation results ({mode}):")
            for idx, row in result_df.iterrows():
                val = row[stat_col]
                pval = row.get("pval_norm", row.get("pval_z_sim", np.nan))
                sig = "*" if pval < 0.05 else " "
                print(f"    {idx:20s}: {stat_col}={val:.4f}  p={pval:.2e} {sig}")

            return result_df
    finally:
        # Clean up temporary columns
        adata.obs.drop(columns=arch_cols, inplace=True, errors="ignore")


def archetype_interaction_boundaries(
    adata: AnnData,
    *,
    cell_type_col: str = "Cell_Type",
    weights_key: str = "cell_archetype_weights",
    cell_type_a: str | None = None,
    cell_type_b: str | None = None,
) -> dict:
    """Detect spatial fronts where archetype weight mixtures diverge between cell types.

    Uses the existing spatial neighbor graph (from ``pc.tl.spatial_neighbors()``)
    and continuous archetype weights (from ``pc.tl.extract_archetype_weights()``)
    to identify tissue regions where two cell types run different archetype
    programs in the same neighborhood.

    For each cell, computes the **mean archetype weight vector** of its
    spatial neighbors that belong to cell type A and cell type B separately,
    then measures divergence (Jensen-Shannon) between these two distributions.

    High-scoring cells sit at spatial fronts where the local archetype
    program of one cell type is shifting in a different direction from the
    other — e.g., macrophages trending toward inflammatory archetypes while
    neighboring fibroblasts trend toward fibrotic archetypes.

    Parameters
    ----------
    adata : AnnData
        Annotated data with:

        - Spatial neighbor graph in ``adata.obsp['spatial_connectivities']``
          (from ``pc.tl.spatial_neighbors()``)
        - Archetype weights in ``adata.obsm[weights_key]``
          (from ``pc.tl.extract_archetype_weights()``)
        - Cell type labels in ``adata.obs[cell_type_col]``
    cell_type_col : str, default: "Cell_Type"
        Column in ``adata.obs`` with cell type labels.
    weights_key : str, default: "archetype_weights"
        Key in ``adata.obsm`` with archetype weight matrix
        [n_cells, n_archetypes]. Each row sums to 1.
    cell_type_a : str | None, default: None
        First cell type for pairwise comparison. If None, uses the
        two most abundant cell types.
    cell_type_b : str | None, default: None
        Second cell type.

    Returns
    -------
    dict
        Dictionary with:

        - ``'boundary_scores'``: np.ndarray [n_cells] — per-cell JSD boundary score
        - ``'mean_weights_a'``: np.ndarray [n_cells, n_archetypes] — mean weight
          vector of type-A neighbors per cell
        - ``'mean_weights_b'``: np.ndarray [n_cells, n_archetypes] — mean weight
          vector of type-B neighbors per cell
        - ``'cross_correlations'``: pd.DataFrame — per-archetype Spearman r
          between cell types across space
        - ``'cell_type_a'``, ``'cell_type_b'``: str — cell type names
        - ``'n_archetypes'``: int — number of archetype dimensions

        Also stored in ``adata.uns['archetype_interaction_boundaries']``
        and boundary scores in ``adata.obs['boundary_score']``.

    Examples
    --------
    >>> pc.tl.spatial_neighbors(adata, n_neighs=30)
    >>> weights = pc.tl.extract_archetype_weights(adata)
    >>> result = pc.tl.archetype_interaction_boundaries(
    ...     adata, cell_type_col="Cell_Type",
    ...     cell_type_a="Myeloid", cell_type_b="Treg",
    ... )
    >>> pc.pl.interaction_boundaries(adata)
    """
    import pandas as pd
    import scipy.sparse as sp

    # --- validate inputs ---
    if "spatial_connectivities" not in adata.obsp:
        raise ValueError(
            "No spatial neighbor graph found. "
            "Run pc.tl.spatial_neighbors(adata) first."
        )
    if weights_key not in adata.obsm:
        raise ValueError(
            f"No archetype weights at adata.obsm['{weights_key}']. "
            f"Run pc.tl.extract_archetype_weights(adata) first."
        )
    if cell_type_col not in adata.obs.columns:
        raise ValueError(f"'{cell_type_col}' not found in adata.obs.")

    W = adata.obsm[weights_key]  # [n_cells, n_arch], rows sum to 1
    S = adata.obsp["spatial_connectivities"]  # [n_cells, n_cells], sparse
    n_cells, n_arch = W.shape

    # --- pick cell types ---
    if cell_type_a is None or cell_type_b is None:
        top2 = adata.obs[cell_type_col].value_counts().head(2).index.tolist()
        cell_type_a = cell_type_a or top2[0]
        cell_type_b = cell_type_b or top2[1]

    mask_a = (adata.obs[cell_type_col] == cell_type_a).values
    mask_b = (adata.obs[cell_type_col] == cell_type_b).values
    n_a, n_b = mask_a.sum(), mask_b.sum()

    if n_a == 0 or n_b == 0:
        raise ValueError(
            f"Cell types not found: {cell_type_a} ({n_a}), {cell_type_b} ({n_b})"
        )

    print(f"  Interaction boundaries: {cell_type_a} ({n_a}) vs {cell_type_b} ({n_b})")
    print(f"  Using {n_arch} archetype weight dimensions, {S.nnz // n_cells} avg neighbors")

    # --- vectorized mean-weight computation ---
    # S_a[i,j] = S[i,j] if cell j is type A, else 0
    # mean_w_a[i] = mean archetype weight vector of cell i's type-A neighbors
    diag_a = sp.diags(mask_a.astype(float))
    diag_b = sp.diags(mask_b.astype(float))

    S_a = S @ diag_a  # connectivity masked to type-A columns
    S_b = S @ diag_b

    count_a = np.asarray(S_a.sum(axis=1)).ravel()  # n type-A neighbors per cell
    count_b = np.asarray(S_b.sum(axis=1)).ravel()

    # Sum of neighbor weights, then divide by count for mean
    sum_w_a = np.asarray(S_a @ W)  # dense [n_cells, n_arch]
    sum_w_b = np.asarray(S_b @ W)

    safe_a = np.where(count_a > 0, count_a, 1.0)
    safe_b = np.where(count_b > 0, count_b, 1.0)

    mean_w_a = sum_w_a / safe_a[:, np.newaxis]
    mean_w_b = sum_w_b / safe_b[:, np.newaxis]

    # Zero out where no neighbors of that type exist
    mean_w_a[count_a == 0] = 0.0
    mean_w_b[count_b == 0] = 0.0

    # --- boundary score: Jensen-Shannon divergence ---
    # JSD(P, Q) = sqrt(0.5 * KL(P||M) + 0.5 * KL(Q||M)), M = (P+Q)/2
    # Vectorized over all cells at once
    both_present = (count_a > 0) & (count_b > 0)

    boundary_scores = np.zeros(n_cells)
    if both_present.any():
        P = mean_w_a[both_present]
        Q = mean_w_b[both_present]
        M = 0.5 * (P + Q)

        eps = 1e-12
        kl_pm = np.sum(np.where(P > eps, P * np.log(P / np.maximum(M, eps)), 0.0), axis=1)
        kl_qm = np.sum(np.where(Q > eps, Q * np.log(Q / np.maximum(M, eps)), 0.0), axis=1)
        jsd = np.sqrt(np.maximum(0.5 * kl_pm + 0.5 * kl_qm, 0.0))

        boundary_scores[both_present] = jsd

    adata.obs["boundary_score"] = boundary_scores

    # --- cross-correlations: per archetype dimension ---
    n_both = both_present.sum()
    cross_corr_data = []

    if n_both > 50:
        from scipy.stats import spearmanr

        for k in range(n_arch):
            r, p = spearmanr(mean_w_a[both_present, k], mean_w_b[both_present, k])
            cross_corr_data.append({
                "archetype": f"archetype_{k}",
                "spearman_r": r,
                "pvalue": p,
                "n_cells": int(n_both),
                "mean_weight_a": float(mean_w_a[both_present, k].mean()),
                "mean_weight_b": float(mean_w_b[both_present, k].mean()),
            })

    cross_corr_df = pd.DataFrame(cross_corr_data)

    # --- report ---
    print(f"  Cells with both types in neighborhood: {n_both}")
    bs_active = boundary_scores[both_present]
    if len(bs_active) > 0:
        print(f"  Boundary score: mean={bs_active.mean():.4f}, "
              f"median={np.median(bs_active):.4f}, max={bs_active.max():.4f}")

    if len(cross_corr_data) > 0:
        print(f"\n  Per-archetype cross-correlations ({cell_type_a} vs {cell_type_b}):")
        for _, row in cross_corr_df.iterrows():
            sig = "*" if row["pvalue"] < 0.05 else " "
            direction = "co-localize" if row["spearman_r"] > 0 else "anti-corr"
            print(f"    {row['archetype']:15s}: r={row['spearman_r']:+.3f}  "
                  f"p={row['pvalue']:.2e} {sig} ({direction})")

    result = {
        "boundary_scores": boundary_scores,
        "mean_weights_a": mean_w_a,
        "mean_weights_b": mean_w_b,
        "cross_correlations": cross_corr_df,
        "cell_type_a": cell_type_a,
        "cell_type_b": cell_type_b,
        "n_archetypes": n_arch,
    }
    adata.uns["archetype_interaction_boundaries"] = result

    return result

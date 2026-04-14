"""Shared preprocessing helpers for Paper Part 1 prep scripts.

Exposes:
    apply_mt_rb_mad_filter(adata, n_mads=3.0)
        3-MAD cell filter on MT/RB fractions + hard drop of MT,
        cytoplasmic ribosomal (RPL/RPS), mitochondrial ribosomal
        (MRPL/MRPS) and MALAT1 genes.

    select_n_pcs_by_cumvar(pca_matrix, threshold=0.95, min_pcs=2, max_pcs=50)
        Data-driven PCA dimension selection: smallest n where
        cumulative variance ratio reaches the threshold, clamped
        to [min_pcs, max_pcs].

Used by: scripts/prep_hsccmp.py, scripts/prep_bigov.py
"""

from __future__ import annotations

import re
import numpy as np
import scanpy as sc


_MT_PREFIXES = ("MT-", "mt-")
_RB_PREFIXES = ("RPS", "RPL", "Rps", "Rpl")
_MITORIBO_PREFIXES = ("MRPS", "MRPL", "Mrps", "Mrpl")
_CONTAM_GENE_REGEX = re.compile(
    r"^MT-|^mt-|^MALAT1$|^RPS|^RPL|^Rps|^Rpl|^MRPS|^MRPL|^Mrps|^Mrpl"
)


def _mad(values: np.ndarray) -> float:
    """Median absolute deviation (no scaling constant)."""
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


def select_n_pcs_by_cumvar(
    pca_matrix: np.ndarray,
    threshold: float = 0.95,
    min_pcs: int = 2,
    max_pcs: int = 50,
) -> int:
    """Select number of PCs to retain by cumulative explained variance.

    Computes the per-PC variance from the PCA matrix, converts to a
    variance ratio (variance / total variance), cumulatively sums, and
    returns the smallest n where ``cumvar[n-1] >= threshold``, clamped
    to ``[min_pcs, max_pcs]``.

    Parameters
    ----------
    pca_matrix : np.ndarray
        Cell x PC matrix, shape ``(n_cells, n_pcs_available)``.
    threshold : float, default 0.95
        Cumulative variance ratio target. Values in (0, 1].
    min_pcs : int, default 2
        Floor for the return value. Guards against degenerate
        one-component datasets being forced into archetypal analysis
        with k=1 which is meaningless.
    max_pcs : int, default 50
        Ceiling for the return value. Clamps runaway selection on
        near-isotropic data.

    Returns
    -------
    int
        Selected number of PCs in ``[min_pcs, min(max_pcs, n_pcs_available)]``.

    Raises
    ------
    ValueError
        If ``pca_matrix`` is not 2-D, is empty, or if the clamp range
        is invalid (``min_pcs > max_pcs``).

    Notes
    -----
    For ``threshold = 1.0``: cumulative variance ratio is bounded above
    by 1 but may not reach it exactly due to floating-point rounding.
    In that case the function returns ``min(max_pcs, n_pcs_available)``.

    For ``threshold = 0``: any positive cumulative variance satisfies
    the condition, so the function returns ``min_pcs``.
    """
    if pca_matrix.ndim != 2:
        raise ValueError(
            f"pca_matrix must be 2-D, got shape {pca_matrix.shape}"
        )
    n_pcs_available = pca_matrix.shape[1]
    if n_pcs_available == 0:
        raise ValueError("pca_matrix has zero PC columns")
    if min_pcs > max_pcs:
        raise ValueError(
            f"min_pcs ({min_pcs}) must be <= max_pcs ({max_pcs})"
        )

    variances = np.var(pca_matrix, axis=0)
    total = float(variances.sum())
    if total <= 0.0:
        # Pathological case: no variance at all. Return min_pcs as a
        # degenerate-safe default.
        return max(min_pcs, 1)

    var_ratio = variances / total
    cumvar = np.cumsum(var_ratio)

    # Smallest n where cumvar[n-1] >= threshold. argmax of boolean finds
    # the first True. If no value reaches threshold, argmax returns 0,
    # which we detect via the all-False check.
    reaches_threshold = cumvar >= threshold
    if reaches_threshold.any():
        n_selected = int(np.argmax(reaches_threshold)) + 1
    else:
        n_selected = n_pcs_available

    # Clamp to the configured range
    upper = min(max_pcs, n_pcs_available)
    n_selected = min(n_selected, upper)
    n_selected = max(n_selected, min_pcs)
    return n_selected


def apply_mt_rb_mad_filter(adata, n_mads: float = 3.0):
    """Drop high MT/RB cells and MT/RB/MALAT1/mitoribo genes.

    Steps:
      1. If ``pct_counts_mt`` / ``pct_counts_rb`` are absent, flag MT and RB
         genes in ``.var`` and compute QC metrics via
         ``scanpy.pp.calculate_qc_metrics``.
      2. Drop cells where ``pct_counts_mt > median + n_mads*MAD`` OR
         ``pct_counts_rb > median + n_mads*MAD``.
      3. Drop MT (MT-/mt-), cytoplasmic ribosomal (RPL/RPS), mitochondrial
         ribosomal (MRPL/MRPS) and MALAT1 genes. Human-only naming
         conventions (also matches the lowercase mouse variants
         Mt-/Rpl/Rps/Mrpl/Mrps).

    Logs cell and gene counts at each step and how many genes were removed
    per category. Raises RuntimeError if all cells are filtered out.

    Parameters
    ----------
    adata : AnnData
        Input AnnData. Not modified in place; a filtered copy is returned.
        If ``pct_counts_mt`` / ``pct_counts_rb`` already exist in
        ``adata.obs`` they are trusted as-is (caller is responsible for
        freshness with respect to the current gene set).
    n_mads : float, default 3.0
        MAD multiplier for the cell-level threshold. Uses raw (unscaled)
        MAD — more conservative than the scaled variant (x1.4826).

    Returns
    -------
    AnnData
        Filtered copy of the input. Original is unmodified.

    Raises
    ------
    RuntimeError
        If the cell filter drops every cell (sentinel of bad input).
    """
    n_cells_in = adata.shape[0]
    n_genes_in = adata.shape[1]
    print(
        f"  [mt_rb_mad_filter] input: {n_cells_in} cells x {n_genes_in} genes"
    )

    # --- QC metrics on MT + RB --------------------------------------------
    needs_metrics = (
        "pct_counts_mt" not in adata.obs.columns
        or "pct_counts_rb" not in adata.obs.columns
    )
    if needs_metrics:
        adata = adata.copy()
        adata.var["mt"] = adata.var_names.str.startswith(_MT_PREFIXES)
        adata.var["rb"] = adata.var_names.str.startswith(_RB_PREFIXES)
        sc.pp.calculate_qc_metrics(
            adata,
            qc_vars=["mt", "rb"],
            percent_top=None,
            log1p=False,
            inplace=True,
        )

    pct_mt = np.asarray(adata.obs["pct_counts_mt"].values, dtype=np.float64)
    pct_rb = np.asarray(adata.obs["pct_counts_rb"].values, dtype=np.float64)

    med_mt = float(np.median(pct_mt))
    mad_mt = _mad(pct_mt)
    med_rb = float(np.median(pct_rb))
    mad_rb = _mad(pct_rb)

    thr_mt = med_mt + n_mads * mad_mt
    thr_rb = med_rb + n_mads * mad_rb

    keep_cells = (pct_mt <= thr_mt) & (pct_rb <= thr_rb)
    n_drop_mt = int((pct_mt > thr_mt).sum())
    n_drop_rb = int((pct_rb > thr_rb).sum())
    n_drop_cells = int((~keep_cells).sum())

    print(
        f"  [mt_rb_mad_filter] MT  median={med_mt:.3f}  MAD={mad_mt:.3f}  "
        f"thr={thr_mt:.3f}  drop={n_drop_mt}"
    )
    print(
        f"  [mt_rb_mad_filter] RB  median={med_rb:.3f}  MAD={mad_rb:.3f}  "
        f"thr={thr_rb:.3f}  drop={n_drop_rb}"
    )

    adata = adata[keep_cells].copy()
    n_cells_after_cells = adata.shape[0]
    print(
        f"  [mt_rb_mad_filter] after cell filter: "
        f"{n_cells_in} -> {n_cells_after_cells} cells "
        f"(dropped {n_drop_cells}, "
        f"{(n_drop_cells / max(n_cells_in, 1)) * 100:.1f}%)"
    )

    if adata.shape[0] == 0:
        raise RuntimeError(
            "apply_mt_rb_mad_filter: all cells were dropped by MT/RB MAD "
            "filter. Check input pct_counts distributions."
        )

    # --- Gene-level filter -------------------------------------------------
    # Drops MT genes, cytoplasmic ribosomal (RPL/RPS) and mitochondrial
    # ribosomal (MRPL/MRPS), plus MALAT1. MRPL/MRPS matches the original
    # HSC prep filter, which was dropped in the r9 regression.
    var_names = adata.var_names
    mt_mask = var_names.str.startswith(_MT_PREFIXES)
    rpl_mask = var_names.str.startswith(("RPL", "Rpl"))
    rps_mask = var_names.str.startswith(("RPS", "Rps"))
    mrpl_mask = var_names.str.startswith(("MRPL", "Mrpl"))
    mrps_mask = var_names.str.startswith(("MRPS", "Mrps"))
    malat_mask = var_names.str.fullmatch("MALAT1")

    n_mt_genes = int(mt_mask.sum())
    n_rpl_genes = int(rpl_mask.sum())
    n_rps_genes = int(rps_mask.sum())
    n_mrpl_genes = int(mrpl_mask.sum())
    n_mrps_genes = int(mrps_mask.sum())
    n_malat_genes = int(malat_mask.sum())

    drop_mask = (
        mt_mask | rpl_mask | rps_mask | mrpl_mask | mrps_mask | malat_mask
    )
    n_drop_genes = int(drop_mask.sum())
    keep_gene_mask = ~drop_mask

    adata = adata[:, keep_gene_mask].copy()
    n_genes_after = adata.shape[1]
    print(
        f"  [mt_rb_mad_filter] gene filter: MT={n_mt_genes}  "
        f"RPL={n_rpl_genes}  RPS={n_rps_genes}  MRPL={n_mrpl_genes}  "
        f"MRPS={n_mrps_genes}  MALAT1={n_malat_genes}  (total {n_drop_genes})"
    )
    print(
        f"  [mt_rb_mad_filter] after gene filter: "
        f"{n_genes_in} -> {n_genes_after} genes"
    )

    # --- Regression guard --------------------------------------------------
    leftover = [g for g in adata.var_names if _CONTAM_GENE_REGEX.match(g)]
    if leftover:
        raise RuntimeError(
            "apply_mt_rb_mad_filter: contaminant genes survived filtering: "
            f"{leftover[:5]}"
        )

    return adata

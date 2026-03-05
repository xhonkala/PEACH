"""Shared feature resolution and storage utilities for v0.5.0 analysis modules."""

import logging
from typing import Any

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

logger = logging.getLogger(__name__)

WEIGHTS_KEY = "cell_archetype_weights"
WEIGHTS_TOLERANCE = 1e-8


def resolve_features(
    adata: AnnData,
    feature_matrix=None,
    feature_names=None,
):
    """Resolve feature matrix and names from flexible input.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    feature_matrix : None, str, or array-like
        None -> adata.X (kept sparse if sparse).
        str -> adata.obsm[feature_matrix].
        array-like -> used directly.
    feature_names : list[str] or None
        Feature names. Inferred from adata.var_names if None and using adata.X.

    Returns
    -------
    tuple[np.ndarray | sp.spmatrix, list[str]]
        (matrix [n_cells, n_features], feature_names)
    """
    n_cells = adata.n_obs

    if feature_matrix is None:
        mat = adata.X
        names = list(adata.var_names) if feature_names is None else list(feature_names)
    elif isinstance(feature_matrix, str):
        mat = adata.obsm[feature_matrix]  # KeyError if missing
        n_feats = mat.shape[1]
        names = (
            list(feature_names)
            if feature_names is not None
            else [f"feature_{i}" for i in range(n_feats)]
        )
    else:
        mat = np.asarray(feature_matrix) if not sp.issparse(feature_matrix) else feature_matrix
        if mat.shape[0] != n_cells:
            raise ValueError(
                f"feature_matrix n_cells mismatch: got {mat.shape[0]}, expected {n_cells}"
            )
        n_feats = mat.shape[1]
        names = (
            list(feature_names)
            if feature_names is not None
            else [f"feature_{i}" for i in range(n_feats)]
        )

    return mat, names


def get_archetype_weights(adata: AnnData) -> np.ndarray:
    """Extract archetype weights from adata.obsm.

    Asserts sum-to-1 within tolerance. Raises ValueError if violated --
    never silently renormalizes.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with archetype weights in obsm.

    Returns
    -------
    np.ndarray
        Weights array [n_cells, K].

    Raises
    ------
    KeyError
        If weights key not found in adata.obsm.
    ValueError
        If weights violate sum-to-1 constraint.
    """
    if WEIGHTS_KEY not in adata.obsm:
        raise KeyError(
            f"adata.obsm['{WEIGHTS_KEY}'] not found. "
            "Run pc.tl.extract_archetype_weights() first."
        )
    weights = np.asarray(adata.obsm[WEIGHTS_KEY])
    row_sums = weights.sum(axis=1)
    max_deviation = np.abs(row_sums - 1.0).max()
    if max_deviation > WEIGHTS_TOLERANCE:
        raise ValueError(
            f"Archetype weights violate sum-to-1 constraint: max deviation {max_deviation:.2e} "
            f"exceeds tolerance {WEIGHTS_TOLERANCE:.0e}. This indicates a bug upstream."
        )
    return weights


def store_result(
    adata: AnnData,
    key: str,
    result: Any,
    domain: str = "uns",
):
    """Store result in adata with peach_ prefix.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    key : str
        Key name (without peach_ prefix).
    result : Any
        Serializable result (dict, array, DataFrame).
    domain : str
        'uns' or 'obsm'.
    """
    full_key = f"peach_{key}"
    storage = getattr(adata, domain)

    if full_key in storage:
        logger.warning(f"Overwriting existing results at adata.{domain}['{full_key}']")

    storage[full_key] = result

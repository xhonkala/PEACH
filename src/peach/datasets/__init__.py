"""Example datasets for PEACH tutorials and testing.

This module provides preprocessed datasets for demonstrating PEACH workflows.

Functions
---------
hsc
    Human hematopoietic stem cell differentiation dataset (10k cells, 2500 HVGs)
synthetic_archetypes
    Generate synthetic data with known archetypal structure
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc
from anndata import AnnData

# Default cache directory
_CACHE_DIR = Path.home() / ".cache" / "peach" / "datasets"

# Zenodo hosting info - UPDATE AFTER UPLOAD
HSC_ZENODO_URL = "https://zenodo.org/record/XXXXXXX/files/hsc_10k.h5ad"
HSC_ZENODO_DOI = "10.5281/zenodo.XXXXXXX"


def _get_cache_dir() -> Path:
    """Get or create cache directory."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def hsc(
    path: Optional[str] = None,
    n_cells: Optional[int] = None,
    random_state: int = 42
) -> AnnData:
    """Load hematopoietic stem cell dataset (10k cells, 2500 HVGs).

    This dataset contains mouse hematopoietic stem and progenitor cells (HSPCs)
    from multiple studies, preprocessed with PCA for archetypal analysis.

    Parameters
    ----------
    path : str, optional
        Path to local h5ad file. If None, looks in ~/.cache/peach/datasets/
    n_cells : int, optional
        Subsample to this many cells (useful for quick testing).
        If None, return full 10k cell dataset.
    random_state : int, default: 42
        Random seed for subsampling reproducibility.

    Returns
    -------
    AnnData
        Annotated data object with:
        - `.X` : Normalized expression (2500 HVGs)
        - `.obs` : Cell metadata (AuthorCellType, Study, donor_id, etc.)
        - `.obsm['X_pca']` : PCA coordinates (50 PCs)

    Examples
    --------
    >>> import peach as pc
    >>> adata = pc.datasets.hsc()
    >>> print(adata)
    AnnData object with n_obs × n_vars = 10000 × 2500

    >>> # Quick test with subset
    >>> adata = pc.datasets.hsc(n_cells=1000)

    Notes
    -----
    Download the dataset from Zenodo if you don't have it:

        wget {url} -O ~/.cache/peach/datasets/hsc_10k.h5ad

    Or specify a custom path:

        adata = pc.datasets.hsc(path='/path/to/hsc_10k.h5ad')
    """.format(url=HSC_ZENODO_URL)

    # Find the file
    if path is not None:
        fpath = Path(path)
    else:
        fpath = _get_cache_dir() / "hsc_10k.h5ad"

    if not fpath.exists():
        raise FileNotFoundError(
            f"HSC dataset not found at {fpath}\n\n"
            f"Download it from Zenodo:\n"
            f"  wget {HSC_ZENODO_URL} -O {fpath}\n\n"
            f"Or visit: https://doi.org/{HSC_ZENODO_DOI}"
        )

    # Load dataset
    adata = sc.read_h5ad(fpath)

    # Subsample if requested
    if n_cells is not None and n_cells < adata.n_obs:
        sc.pp.subsample(adata, n_obs=n_cells, random_state=random_state)

    return adata


def synthetic_archetypes(
    n_cells: int = 5000,
    n_archetypes: int = 4,
    n_dims: int = 50,
    noise_level: float = 0.1,
    random_state: int = 42
) -> AnnData:
    """Generate synthetic data with known archetypal structure.

    Creates data points as convex combinations of archetypes plus noise.
    Useful for testing and validating archetypal analysis methods.

    Parameters
    ----------
    n_cells : int, default: 5000
        Number of synthetic cells to generate.
    n_archetypes : int, default: 4
        Number of archetypes (vertices of the simplex).
    n_dims : int, default: 50
        Dimensionality of the data (e.g., number of PCs).
    noise_level : float, default: 0.1
        Standard deviation of Gaussian noise added to data.
    random_state : int, default: 42
        Random seed for reproducibility.

    Returns
    -------
    AnnData
        Annotated data object with:
        - `.X` : Synthetic data matrix
        - `.obsm['X_pca']` : Same as .X (already in "PCA" space)
        - `.obsm['true_weights']` : True archetypal weights (A matrix)
        - `.uns['true_archetypes']` : True archetype positions

    Examples
    --------
    >>> import peach as pc
    >>> adata = pc.datasets.synthetic_archetypes(n_cells=1000, n_archetypes=3)
    >>>
    >>> # Train model and compare to ground truth
    >>> results = pc.tl.train_archetypal(adata, n_archetypes=3)
    >>> true_archetypes = adata.uns['true_archetypes']
    """
    rng = np.random.default_rng(random_state)

    # Generate archetypes as random points
    archetypes = rng.standard_normal((n_archetypes, n_dims))

    # Generate random convex weights (Dirichlet distribution)
    weights = rng.dirichlet(np.ones(n_archetypes), size=n_cells)

    # Generate data as convex combinations + noise
    data = weights @ archetypes + noise_level * rng.standard_normal((n_cells, n_dims))

    # Create AnnData
    adata = AnnData(data.astype(np.float32))
    adata.obsm["X_pca"] = data.astype(np.float32).copy()
    adata.obsm["true_weights"] = weights.astype(np.float32)
    adata.uns["true_archetypes"] = archetypes.astype(np.float32)
    adata.uns["synthetic_params"] = {
        "n_archetypes": n_archetypes,
        "noise_level": noise_level,
        "random_state": random_state,
    }

    # Add cell labels based on dominant archetype
    dominant = np.argmax(weights, axis=1)
    adata.obs["dominant_archetype"] = [f"archetype_{i}" for i in dominant]
    adata.obs["dominant_archetype"] = adata.obs["dominant_archetype"].astype("category")

    return adata


__all__ = [
    "hsc",
    "synthetic_archetypes",
]

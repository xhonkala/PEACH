"""
Basic preprocessing functions for archetypal analysis.

This module provides essential data loading, synthetic data generation,
and pathway analysis utilities for Deep Archetypal Analysis workflows.

Main Functions:
- load_data(): Load AnnData from file with proper format validation
- generate_synthetic(): Create realistic synthetic datasets for testing
- prepare_training(): Convert AnnData to PyTorch DataLoader
- load_pathway_networks(): Access MSigDB pathway collections
- compute_pathway_scores(): Calculate pathway activity scores

All functions follow scVerse conventions with AnnData-centric workflows.
"""

from anndata import AnnData
from torch.utils.data import DataLoader

from .._core.utils.convex_synth_data import generate_convex_data as _generate_convex_data
from .._core.utils.gene_analysis import compute_pathway_scores as _compute_pathway_scores
from .._core.utils.gene_analysis import load_pathway_networks as _load_pathway_networks
from .._core.utils.load_anndata import create_dataloader_from_anndata as _create_dataloader

# Import existing battle-tested functions
from .._core.utils.load_anndata import load_anndata as _load_anndata


def load_data(path: str, use_raw: bool = True, dim_reduction_key: str = "X_PCA", batch_size: int = 128) -> AnnData:
    """Load AnnData for archetypal analysis.

    Note: Use scanpy.pp.pca() to compute PCA coordinates after loading.

    Parameters
    ----------
    path : str
        Path to the data file (matches _core parameter name)
    use_raw : bool, default: True
        Whether to use raw data
    dim_reduction_key : str, default: "X_PCA"
        Key for dimension reduction in adata.obsm
    batch_size : int, default: 128
        Batch size for data loading

    Returns
    -------
    AnnData
        Loaded data. Use sc.pp.pca(adata) to add PCA coordinates.
    """
    return _load_anndata(path=path, use_raw=use_raw, dim_reduction_key=dim_reduction_key, batch_size=batch_size)


def generate_synthetic(
    n_points: int = 1000,
    n_dimensions: int = 50,
    n_archetypes: int = 4,
    noise: float = 0.1,
    *,
    seed: int = 1205,
    archetype_type: str = "random",
    scale: float = 20.0,
    return_torch: bool = True,
) -> AnnData:
    """Generate synthetic convex data for testing.

    Parameters
    ----------
    n_points : int, default: 1000
        Number of data points to generate (matches _core parameter)
    n_dimensions : int, default: 50
        Number of dimensions/features (matches _core parameter)
    n_archetypes : int, default: 4
        Number of archetypes
    noise : float, default: 0.1
        Noise level (matches _core parameter)
    seed : int, default: 1205
        Random seed for reproducibility
    archetype_type : str, default: "random"
        Type of archetype generation ('random', 'corners', 'sphere')
    scale : float, default: 20.0
        Scale factor for data generation
    return_torch : bool, default: True
        Whether to return PyTorch tensors

    Returns
    -------
    AnnData
        Synthetic data with ground truth archetypes in .uns
    """
    # Generate the synthetic data (returns tuple of tensors)
    data, archetypes = _generate_convex_data(
        n_points=n_points,
        n_dimensions=n_dimensions,
        n_archetypes=n_archetypes,
        noise=noise,
        seed=seed,
        archetype_type=archetype_type,
        scale=scale,
        return_torch=return_torch,
    )

    # Convert to AnnData
    from anndata import AnnData

    adata = AnnData(X=data.numpy() if return_torch else data)
    adata.var_names = [f"Feature_{i}" for i in range(n_dimensions)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_points)]

    # Store ground truth archetypes
    adata.uns["true_archetypes"] = archetypes.numpy() if return_torch else archetypes

    # Compute proper PCA coordinates (like real data workflow)
    import scanpy as sc

    sc.pp.pca(adata, n_comps=min(50, n_dimensions - 1))  # Standard PCA computation

    return adata


def prepare_training(
    adata: AnnData,
    batch_size: int = 128,
    shuffle: bool = True,
    pca_key: str = None,
    num_workers: int | str = "auto",
    pin_memory: bool | str = "auto",
    persistent_workers: bool | str = "auto",
    prefetch_factor: int = 2,
) -> DataLoader:
    """Create DataLoader from AnnData for training with HPC optimizations.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with PCA coordinates
    batch_size : int, default: 128
        Batch size for training
    shuffle : bool, default: True
        Whether to shuffle data in DataLoader
    pca_key : str, default: None
        Key in adata.obsm containing PCA coordinates (auto-detected if None)
    num_workers : int or 'auto', default: 'auto'
        Number of subprocesses for data loading. 'auto' detects optimal value
        based on environment (0 for Apple Silicon, 6 for HPC, 2 for local)
    pin_memory : bool or 'auto', default: 'auto'
        Use pinned memory for faster GPU transfer. 'auto' sets True if CUDA available
    persistent_workers : bool or 'auto', default: 'auto'
        Keep workers alive between epochs. 'auto' sets True if num_workers > 0
    prefetch_factor : int, default: 2
        Number of batches loaded in advance by each worker

    Returns
    -------
    DataLoader
        PyTorch DataLoader optimized for the execution environment

    Examples
    --------
    >>> # Auto-detect optimal settings
    >>> dataloader = peach.pp.prepare_training(adata)

    >>> # Force HPC settings
    >>> dataloader = peach.pp.prepare_training(adata, num_workers=8, pin_memory=True)

    >>> # Minimal settings for debugging
    >>> dataloader = peach.pp.prepare_training(adata, num_workers=0)
    """
    return _create_dataloader(
        adata=adata,
        batch_size=batch_size,
        shuffle=shuffle,
        pca_key=pca_key,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )


def load_pathway_networks(
    sources: list[str] = ["c5_bp"],
    *,
    organism: str = "human",
    geneset_repo: str = "msigdb",
    verbose: bool = True,
    **kwargs,
):
    """Load pathway networks from MSigDB or OmniPath.

    Parameters
    ----------
    sources : List[str], default: ["c5_bp"]
        Pathway sources to load. MSigDB collections: 'hallmark', 'c2_cp',
        'c2_cgp', 'c3_mir', 'c5_bp', 'c5_cc', 'c5_mf', 'c8'
    organism : str, default: "human"
        Organism to load pathways for: 'human' or 'mouse'
    geneset_repo : str, default: "msigdb"
        Repository to use: 'msigdb' (recommended) or 'omnipath'
    verbose : bool, default: True
        Whether to print loading progress
    **kwargs
        Additional arguments passed to load_pathway_networks

    Returns
    -------
    pd.DataFrame
        Pathway network with 'source', 'target', 'pathway' columns
    """
    return _load_pathway_networks(
        sources=sources, organism=organism, geneset_repo=geneset_repo, verbose=verbose, **kwargs
    )


def compute_pathway_scores(
    adata: AnnData, net=None, use_layer: str = None, obsm_key: str = "pathway_scores", verbose: bool = True
) -> None:
    """Compute pathway activity scores using MSigDB pathways.

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    net : pd.DataFrame, optional
        Pathway network dataframe. If None, will load using sources parameter
    use_layer : str, optional
        Layer in adata to use for scoring
    obsm_key : str, default: "pathway_scores"
        Key in adata.obsm to store pathway scores
    verbose : bool, default: True
        Whether to print progress
    """
    # Load pathway networks if not provided
    if net is None:
        net = _load_pathway_networks(sources=["c5_bp"], verbose=verbose)

    # Compute scores and store in AnnData
    _compute_pathway_scores(adata=adata, net=net, use_layer=use_layer, obsm_key=obsm_key, verbose=verbose)

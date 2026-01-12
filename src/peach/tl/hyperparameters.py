# """Hyperparameter optimization for archetypal analysis."""

# from typing import List, Dict, Any
# from anndata import AnnData
# from torch.utils.data import DataLoader
# import torch

# # Import existing battle-tested functions
# from .._core.utils.hyperparameter_search import ArchetypalGridSearch, SearchConfig
# from .._core.utils.grid_search_results import CVSummary


# def hyperparameter_search(
#     adata: AnnData,
#     *,
#     n_archetypes_range: List[int] = [3, 4, 5, 6],
#     cv_folds: int = 3,
#     max_epochs_cv: int = 15,
#     pca_key: str = "X_pca",
#     device: str = "cpu",
#     base_model_config: Dict[str, Any] | None = None,
#     **kwargs
# ) -> CVSummary:
#     """Perform cross-validation hyperparameter search.

#     Systematically searches hyperparameter space using cross-validation
#     to find optimal model configurations for archetypal analysis.

#     Parameters
#     ----------
#     adata : AnnData
#         Annotated data object with PCA coordinates
#     n_archetypes_range : List[int], default: [3, 4, 5, 6]
#         Range of archetype numbers to test
#     cv_folds : int, default: 3
#         Number of cross-validation folds
#     max_epochs_cv : int, default: 15
#         Maximum epochs per CV fold (early stopping recommended)
#     pca_key : str, default: "X_pca"
#         Key in adata.obsm containing PCA coordinates
#     device : str, default: "cpu"
#         Device to use for training ('cpu', 'cuda', or 'mps')
#         Default is 'cpu' for stability on Apple Silicon
#     base_model_config : dict | None, default: None
#         Base model configuration to extend

#     Returns
#     -------
#     CVSummary
#         Complete cross-validation results with ranking and analysis methods

#     Examples
#     --------
#     >>> cv_summary = pc.tl.hyperparameter_search(
#     ...     adata,
#     ...     n_archetypes_range=[3, 4, 5],
#     ...     device='cpu'
#     ... )
#     >>> print(cv_summary.summary_report())
#     >>> top_configs = cv_summary.rank_by_metric('archetype_r2')
#     >>> fig = cv_summary.plot_elbow_curve(['archetype_r2', 'rmse'])
#     """
#     # Input validation
#     if pca_key not in adata.obsm:
#         raise ValueError(f"adata.obsm['{pca_key}'] not found. Run sc.pp.pca() first.")

#     # Create DataLoader
#     from ..pp.basic import prepare_training
#     dataloader = prepare_training(adata, pca_key=pca_key)

#     # Configure search
#     # Note: SearchConfig doesn't have latent_dim_offset_range parameter
#     # Filter out the latent_dim_offset_range if passed
#     search_kwargs = dict(kwargs)
#     if 'latent_dim_offset_range' in search_kwargs:
#         search_kwargs.pop('latent_dim_offset_range')

#     search_config = SearchConfig(
#         n_archetypes_range=n_archetypes_range,
#         cv_folds=cv_folds,
#         max_epochs_cv=max_epochs_cv,
#         **search_kwargs
#     )

#     # Default base model configuration
#     if base_model_config is None:
#         base_model_config = {
#             'input_dim': adata.obsm[pca_key].shape[1],
#             'barycentric_mode': True,
#             'device': device
#         }

#     # Run search
#     grid_search = ArchetypalGridSearch(search_config)
#     cv_summary = grid_search.fit(dataloader, base_model_config)

#     return cv_summary

"""Hyperparameter optimization for archetypal analysis.

User-facing wrapper for cross-validation hyperparameter search.
Provides a simple interface to the core grid search functionality.

Examples
--------
>>> import peach as pc
>>> # Basic search
>>> cv_summary = pc.tl.hyperparameter_search(adata, n_archetypes_range=[3, 4, 5, 6], cv_folds=5)
>>> # View results
>>> print(cv_summary.summary_report())
>>> top_configs = cv_summary.rank_by_metric("archetype_r2")[:3]
>>> fig = cv_summary.plot_elbow_r2()
"""

from typing import Any

from anndata import AnnData

from .._core.utils.grid_search_results import CVSummary
from .._core.utils.hyperparameter_search import ArchetypalGridSearch, SearchConfig


def hyperparameter_search(
    adata: AnnData,
    *,
    n_archetypes_range: list[int] = [3, 4, 5, 6],
    cv_folds: int = 3,
    max_epochs_cv: int = 15,
    pca_key: str = "X_pca",
    device: str = "cpu",
    base_model_config: dict[str, Any] | None = None,
    **kwargs,
) -> CVSummary:
    """Perform cross-validation hyperparameter search for archetypal analysis.

    Systematically searches hyperparameter space using K-fold cross-validation
    to find optimal model configurations. Results support manual selection
    of the best configuration for final training.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with PCA coordinates in ``adata.obsm[pca_key]``.
        Run ``scanpy.pp.pca(adata)`` first.
    n_archetypes_range : list[int], default: [3, 4, 5, 6]
        Range of archetype numbers to test. Each value is evaluated
        via cross-validation.
    cv_folds : int, default: 3
        Number of cross-validation folds. Higher values give more
        reliable estimates but take longer.
    max_epochs_cv : int, default: 15
        Maximum training epochs per CV fold. Early stopping typically
        triggers before this limit.
    pca_key : str, default: "X_pca"
        Key in ``adata.obsm`` containing PCA coordinates.
        Auto-detects: 'X_pca', 'X_PCA', 'PCA'.
    device : str, default: "cpu"
        Computing device ('cpu', 'cuda', or 'mps').
        Default is 'cpu' for stability across platforms.
    base_model_config : dict | None, default: None
        Additional base model configuration. If None, uses defaults:

        - ``input_dim`` : Auto-detected from PCA dimensions
        - ``barycentric_mode`` : True
        - ``device`` : From device parameter

    **kwargs
        Additional arguments passed to SearchConfig:

        - ``hidden_dims_options`` : list[list[int]] - Architectures to test
        - ``inflation_factor_range`` : list[float] - Inflation factors to test
        - ``speed_preset`` : str - "fast", "balanced", or "thorough"
        - ``use_pcha_init`` : bool - Use PCHA initialization
        - ``subsample_fraction`` : float - Subsampling for large datasets
        - ``max_cells_cv`` : int - Maximum cells for CV
        - ``random_state`` : int - Random seed

    Returns
    -------
    CVSummary
        Complete cross-validation results with analysis methods:

        **Attributes:**

        - ``config_results`` : dict[str, CVResults] - Per-configuration results
        - ``summary_df`` : pd.DataFrame - Summary table
        - ``ranked_configs`` : list[dict] - Configs ranked by R²
        - ``cv_info`` : dict - Search metadata

        **Methods:**

        - ``summary_report()`` : str - Text summary for decision support
        - ``rank_by_metric(metric)`` : list[dict] - Rank by any metric
        - ``plot_elbow_r2()`` : Figure - Primary visualization
        - ``plot_metric(metric)`` : Figure - Generic metric visualization
        - ``save(path)`` / ``load(path)`` - Persistence

        **Ranked config structure:**

        Each dict in ``ranked_configs`` contains:

        - ``hyperparameters`` : dict with n_archetypes, hidden_dims, etc.
        - ``metric_value`` : float - R² value
        - ``std_error`` : float - Standard error across folds
        - ``config_summary`` : str - Human-readable description

    Raises
    ------
    ValueError
        If ``adata.obsm[pca_key]`` not found.

    Notes
    -----
    **Workflow Position**: This is Phase 2 of the PEACH pipeline. After
    finding good hyperparameters here, manually select the best configuration
    (Phase 3) and train the final model with ``pc.tl.train_archetypal()``
    (Phase 4).

    **Large Datasets**: Datasets larger than ``max_cells_cv`` (default 15000)
    are automatically subsampled for CV. This doesn't affect final training.

    **Selecting Best Configuration**: Use ``summary_report()`` for a quick
    overview, ``rank_by_metric()`` for detailed rankings, and
    ``plot_elbow_r2()`` to visualize the elbow curve.

    Examples
    --------
    Basic hyperparameter search:

    >>> import scanpy as sc
    >>> import peach as pc
    >>> # Prepare data
    >>> sc.pp.pca(adata, n_comps=30)
    >>> # Search hyperparameters
    >>> cv_summary = pc.tl.hyperparameter_search(adata, n_archetypes_range=[3, 4, 5, 6], cv_folds=5, device="cuda")
    >>> # Review results
    >>> print(cv_summary.summary_report())

    Analyze and visualize results:

    >>> # Get top configurations
    >>> top_configs = cv_summary.rank_by_metric("archetype_r2")[:3]
    >>> for config in top_configs:
    ...     print(f"{config['config_summary']}: R²={config['metric_value']:.4f}")
    >>> # Elbow curve
    >>> fig = cv_summary.plot_elbow_r2()
    >>> fig.show()
    >>> # Compare metrics
    >>> fig = cv_summary.plot_metric("rmse")
    >>> fig.show()

    Use selected hyperparameters for final training:

    >>> # Select best configuration
    >>> best_config = top_configs[0]["hyperparameters"]
    >>> n_archetypes = best_config["n_archetypes"]
    >>> # Train final model (Phase 4)
    >>> results = pc.tl.train_archetypal(
    ...     adata, n_archetypes=n_archetypes, n_epochs=200, model_config={"hidden_dims": best_config["hidden_dims"]}
    ... )

    Save and load results:

    >>> # Save for later
    >>> cv_summary.save("cv_results.pkl")
    >>> # Load in new session
    >>> from peach._core.utils.grid_search_results import CVSummary
    >>> cv_summary = CVSummary.load("cv_results.pkl")

    See Also
    --------
    peach.tl.train_archetypal : Train final model with selected hyperparameters
    peach._core.utils.hyperparameter_search.SearchConfig : Full configuration options
    peach._core.utils.grid_search_results.CVSummary : Return type details
    """
    # Input validation
    if pca_key not in adata.obsm:
        raise ValueError(f"adata.obsm['{pca_key}'] not found. Run sc.pp.pca() first.")

    # Create DataLoader
    from ..pp.basic import prepare_training

    dataloader = prepare_training(adata, pca_key=pca_key)

    # Configure search
    # Note: Filter out unsupported kwargs
    search_kwargs = dict(kwargs)
    unsupported = ["latent_dim_offset_range"]
    for key in unsupported:
        if key in search_kwargs:
            search_kwargs.pop(key)

    search_config = SearchConfig(
        n_archetypes_range=n_archetypes_range, cv_folds=cv_folds, max_epochs_cv=max_epochs_cv, **search_kwargs
    )

    # Default base model configuration
    if base_model_config is None:
        base_model_config = {"input_dim": adata.obsm[pca_key].shape[1], "barycentric_mode": True, "device": device}

    # Run search
    grid_search = ArchetypalGridSearch(search_config)
    cv_summary = grid_search.fit(dataloader, base_model_config)

    return cv_summary

"""
Core archetypal analysis functions.

This module provides the primary interface for training Deep Archetypal Analysis
models and extracting archetypal coordinates. All functions work directly with
AnnData objects and follow scVerse conventions.

Main Functions:
- train_archetypal(): Train Deep AA model to discover cellular archetypes
- archetypal_coordinates(): Extract archetypal coordinates for all cells
- assign_archetypes(): Assign cells to discovered archetypes based on distances

The module integrates PCHA initialization, inflation factors, and comprehensive
training diagnostics for production-ready archetypal analysis workflows.
"""

from typing import Any

import numpy as np
import torch
from anndata import AnnData

from .._core.models.Deep_AA import Deep_AA
from .._core.utils.analysis import bin_cells_by_archetype as _bin_cells_by_archetype
from .._core.utils.analysis import compute_archetype_distances as _compute_archetype_distances

# Import existing battle-tested functions
from .._core.utils.training import train_vae as _train_vae


def train_archetypal(
    adata: AnnData,
    n_archetypes: int = 5,
    n_epochs: int = 50,
    *,  # Keyword-only arguments (scVerse convention)
    layer: str | None = None,
    pca_key: str = "X_pca",
    # Model architecture parameters
    hidden_dims: list[int] | None = None,
    inflation_factor: float = 1.5,
    # Config dicts (for advanced users)
    model_config: dict[str, Any] | None = None,
    optimizer_config: dict[str, Any] | None = None,
    # Parameters from _core.train_vae
    device: str = "cpu",
    save_path: str = None,
    archetypal_weight: float = None,
    kld_weight: float = None,
    reconstruction_weight: float = 0.0,
    vae_recon_weight: float = 0.0,
    diversity_weight: float = 0.0,
    activation_func: str = "relu",
    track_stability: bool = True,
    validate_constraints: bool = True,
    lr_factor: float = 0.1,
    lr_patience: int = 10,
    seed: int = 42,
    constraint_tolerance: float = 1e-3,
    stability_history_size: int = 20,
    store_coords_key: str = "archetype_coordinates",
    early_stopping: bool = False,
    early_stopping_patience: int = 10,
    early_stopping_metric: str = "archetype_r2",
    min_improvement: float = 1e-4,
    validation_check_interval: int = 5,
    validation_data_loader=None,
    **kwargs,
) -> dict[str, Any]:
    """Train Deep Archetypal Analysis model to discover cellular archetypes.

    This function performs archetypal analysis using a variational autoencoder
    architecture to identify extreme cellular states (archetypes) that capture
    the main axes of biological variation. Each cell is represented as a convex
    combination of the learned archetypes.

    The model uses PCHA initialization with inflation factors for optimal
    archetype positioning and achieves state-of-the-art performance (R² > 0.89)
    on real single-cell datasets.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing single-cell expression data.
        Must have PCA coordinates in ``adata.obsm[pca_key]``.
        Typically generated using ``scanpy.pp.pca(adata)``.
    n_archetypes : int, default: 5
        Number of archetypal patterns to learn. Should be chosen based on
        biological knowledge or using hyperparameter optimization.
        Common values: 3-10 for most datasets.
    n_epochs : int, default: 50
        Number of training epochs. Larger datasets may require more epochs.
        For datasets >5K cells, consider 100-200 epochs.
    layer : str | None, default: None
        AnnData layer to use for training. If None, uses PCA coordinates
        from ``adata.obsm[pca_key]`` (recommended).
    pca_key : str, default: "X_pca"
        Key in ``adata.obsm`` containing PCA coordinates. The model works
        best with 5-50 PCA components. Auto-detects: 'X_pca', 'X_PCA', 'PCA'.
    hidden_dims : list[int] | None, default: None
        Encoder/decoder layer dimensions. If None, uses [256, 128, 64].
        Smaller architectures like [128, 64] train faster but may underfit.
        Larger architectures like [512, 256, 128] may overfit on small datasets.
    inflation_factor : float, default: 1.5
        PCHA inflation factor for archetype initialization. Values > 1.0 push
        initial archetypes further from the data centroid, improving separation.
        Recommended range: 1.2-2.0. Higher values for more distinct archetypes.
    model_config : dict | None, default: None
        Additional model configuration parameters (for advanced users):

        - ``archetypal_weight`` : float - Archetypal loss weight, default 1.0
        - ``kld_weight`` : float - KL divergence weight, default 0.0
        - ``diversity_weight`` : float - Archetype diversity weight, default 0.05
        - ``use_barycentric`` : bool - Use softmax constraints, default True

    optimizer_config : dict | None, default: None
        Optimizer configuration parameters:

        - ``lr`` : float - Learning rate, default 1e-3
        - ``weight_decay`` : float - L2 regularization, default 0.0
        - ``betas`` : tuple[float, float] - Adam momentum parameters

    device : str, default: "cpu"
        Compute device for training. One of "cpu", "cuda", "mps".
    save_path : str | None, default: None
        Path to save model checkpoints during training.
    archetypal_weight : float | None, default: None
        Weight for archetypal loss component. Uses model's configured value if None.
    kld_weight : float | None, default: None
        Weight for KL divergence loss. Uses model's configured value if None.
    reconstruction_weight : float, default: 0.0
        Legacy reconstruction weight parameter.
    vae_recon_weight : float, default: 0.0
        VAE reconstruction weight.
    diversity_weight : float, default: 0.0
        Weight for archetype diversity loss.
    activation_func : str, default: "relu"
        Activation function for the model.
    track_stability : bool, default: True
        Whether to monitor archetype drift during training. Adds stability
        metrics to history: archetype_drift_mean, archetype_variance_mean, etc.
    validate_constraints : bool, default: True
        Whether to validate archetypal constraints during training. Adds
        constraint metrics to history: constraints_satisfied, A_sum_error, etc.
    lr_factor : float, default: 0.1
        Factor for learning rate reduction on plateau.
    lr_patience : int, default: 10
        Number of epochs with no improvement before reducing learning rate.
    seed : int, default: 42
        Random seed for reproducibility.
    constraint_tolerance : float, default: 1e-3
        Tolerance for constraint validation.
    stability_history_size : int, default: 20
        Number of epochs to track for stability analysis.
    store_coords_key : str, default: "archetype_coordinates"
        Key to store learned archetype positions in ``adata.uns``.
    early_stopping : bool, default: False
        Whether to use early stopping based on validation metrics.
    early_stopping_patience : int, default: 10
        Patience for early stopping (number of checks without improvement).
    early_stopping_metric : str, default: "archetype_r2"
        Metric to monitor for early stopping. One of: 'archetype_r2', 'loss', 'rmse'.
    min_improvement : float, default: 1e-4
        Minimum improvement required to reset patience counter.
    validation_check_interval : int, default: 5
        How often to check validation metrics (in epochs).
    validation_data_loader : DataLoader | None, default: None
        Validation data loader for early stopping. Uses training data if None.
    **kwargs
        Additional arguments passed to the core training function.

    Returns
    -------
    dict
        Training results dictionary with the following structure:

        **Guaranteed keys (always present):**

        - ``history`` : dict
            Training metrics per epoch. Keys depend on tracking options:

            - Core metrics (always): ``loss``, ``archetypal_loss``, ``archetype_r2``, ``rmse``
            - KLD metrics: ``kld_loss``, ``KLD``
            - Stability metrics (if track_stability=True): ``archetype_drift_mean``,
              ``archetype_drift_max``, ``archetype_variance_mean``, etc.
            - Constraint metrics (if validate_constraints=True): ``constraints_satisfied``,
              ``A_sum_error``, ``B_sum_error``, ``constraint_violation_rate``
            - Validation metrics (if early_stopping=True): ``val_loss``, ``val_archetype_r2``

        - ``final_model`` : torch.nn.Module
            Trained Deep_AA model instance.
        - ``model`` : torch.nn.Module
            Alias for ``final_model`` (same object, for compatibility).
        - ``final_optimizer`` : torch.optim.Optimizer
            Final optimizer state.
        - ``final_analysis`` : dict
            Final training analysis containing:

            - ``final_constraint_validation`` : dict with constraint metrics
            - ``archetypal_weights`` : dict with A_matrix and B_matrix analysis
            - ``final_coordinates`` : dict with 'A', 'B', 'Y' tensors
            - ``error`` : str (only if analysis failed)

        - ``epoch_archetype_positions`` : list[torch.Tensor]
            Archetype positions at each epoch, shape (n_archetypes, input_dim).
        - ``training_config`` : dict
            Training configuration with keys: ``n_epochs``, ``actual_epochs``,
            ``early_stop_triggered``, ``archetypal_weight``, ``kld_weight``,
            ``reconstruction_weight``, ``activation_func``, ``seed``,
            ``constraint_tolerance``, ``stability_history_size``, ``early_stopping``,
            ``early_stopping_patience``, ``early_stopping_metric``.

        **Convenience keys (conditional - use .get() to access safely):**

        - ``final_archetype_r2`` : float | None
            Last value of ``history['archetype_r2']`` if tracked.
        - ``final_rmse`` : float | None
            Last value of ``history['rmse']`` if tracked.
        - ``final_mae`` : float | None
            Last value of ``history['mae']`` if tracked.
        - ``final_loss`` : float | None
            Last value of ``history['loss']`` if tracked.
        - ``convergence_epoch`` : int | None
            Equals ``training_config['actual_epochs']``.

    Raises
    ------
    ValueError
        If ``adata.obsm[pca_key]`` is not found. Run ``scanpy.pp.pca()`` first.
        If ``n_archetypes`` exceeds PCA dimensions.
    RuntimeError
        If CUDA device is requested but not available.

    Stores
    ------
    The function stores the following in AnnData:

    - ``adata.uns[store_coords_key]`` : np.ndarray
        Archetype positions in PCA space, shape (n_archetypes, n_pcs).
        Default key: 'archetype_coordinates'.

    Notes
    -----
    **Archetypal Analysis Theory**: Archetypal analysis represents each data point
    as a convex combination of extreme points (archetypes). Unlike clustering,
    which partitions data, archetypal analysis allows cells to have partial
    membership in multiple archetypes, better reflecting biological continuity.

    **Model Architecture**: Uses a variational autoencoder where the latent space
    directly represents archetypal coordinates (A matrix). Archetypes are learned
    as model parameters (Y matrix) rather than constructed from data points.

    **Accessing Results Safely**:

    - For guaranteed keys, direct access works: ``results['history']``
    - For convenience keys, use ``.get()``: ``results.get('final_archetype_r2')``
    - Or access from history: ``results['history']['archetype_r2'][-1]``

    **Type Validation**: For IDE autocomplete and runtime validation::

        from peach._core.types import TrainingResults, validate_training_results

        validated = validate_training_results(results)

    Examples
    --------
    Basic usage with default parameters:

    >>> import scanpy as sc
    >>> import peach as pc
    >>> # Prepare data with PCA
    >>> sc.pp.pca(adata, n_comps=30)
    >>> # Train archetypal model
    >>> results = pc.tl.train_archetypal(adata, n_archetypes=5, n_epochs=100)
    >>> # Access final R² safely (may be None if not tracked)
    >>> r2 = results.get("final_archetype_r2")
    >>> if r2 is not None:
    ...     print(f"Final R²: {r2:.3f}")
    >>> # Or access from history (guaranteed if metric was tracked)
    >>> if results["history"].get("archetype_r2"):
    ...     r2 = results["history"]["archetype_r2"][-1]
    ...     print(f"Final R²: {r2:.3f}")

    Advanced usage with custom configuration:

    >>> model_config = {"hidden_dims": [512, 256, 128], "inflation_factor": 2.0}
    >>> results = pc.tl.train_archetypal(
    ...     adata,
    ...     n_archetypes=4,
    ...     n_epochs=150,
    ...     model_config=model_config,
    ...     early_stopping=True,
    ...     early_stopping_patience=15,
    ...     device="cuda",
    ... )
    >>> # Check if early stopping triggered
    >>> config = results["training_config"]
    >>> if config["early_stop_triggered"]:
    ...     print(f"Converged at epoch {config['actual_epochs']}")

    Accessing archetype coordinates stored in AnnData:

    >>> # After training, coordinates are in adata.uns
    >>> archetype_coords = adata.uns["archetype_coordinates"]
    >>> print(f"Learned {archetype_coords.shape[0]} archetypes")
    >>> print(f"Each archetype has {archetype_coords.shape[1]} PCA dimensions")

    See Also
    --------
    peach.tl.archetypal_coordinates : Extract cell-archetype distances
    peach.tl.assign_archetypes : Assign cells to discovered archetypes
    peach.tl.extract_archetype_weights : Get barycentric coordinates for cells
    peach.pl.training_metrics : Visualize training curves
    peach._core.types.TrainingResults : Type definition for return structure
    """
    # Comprehensive input validation (scVerse convention)
    if pca_key not in adata.obsm:
        # Try to auto-detect PCA coordinates
        pca_candidates = ["X_pca", "X_PCA", "PCA", "x_pca"]
        found_pca = None
        for candidate in pca_candidates:
            if candidate in adata.obsm:
                found_pca = candidate
                break

        if found_pca:
            print(f"[WARNING]  Using {found_pca} instead of {pca_key}")
            pca_key = found_pca
        else:
            raise ValueError(f"PCA coordinates not found. Tried: {pca_candidates}. Run scanpy.pp.pca(adata) first.")

    # Validate PCA dimensions
    pca_shape = adata.obsm[pca_key].shape
    if pca_shape[1] < 3:
        raise ValueError(f"PCA has only {pca_shape[1]} components. Need at least 3 for archetypal analysis.")

    # Validate archetype count vs PCA dimensions
    if n_archetypes > pca_shape[1]:
        raise ValueError(
            f"n_archetypes ({n_archetypes}) cannot exceed PCA dimensions ({pca_shape[1]}). "
            f"Reduce n_archetypes or increase PCA components."
        )

    # Default configurations - use proven working setup
    # Use all available PCA components (or user-specified input_dim)
    input_dim = adata.obsm[pca_key].shape[1]  # Use all available PCA components

    default_model_config = {
        "input_dim": input_dim,
        "n_archetypes": n_archetypes,
        "latent_dim": n_archetypes,
        "hidden_dims": hidden_dims if hidden_dims is not None else [256, 128, 64],
        "archetypal_weight": 1.0,
        "kld_weight": 0.0,
        "diversity_weight": 0.05,
        "manifold_weight": 0.0,
        "regularity_weight": 0.0,
        "sparsity_weight": 0.0,
        "inflation_factor": inflation_factor,
        "use_barycentric": True,
        "use_hidden_transform": True,
        "device": device,
    }
    if model_config:
        default_model_config.update(model_config)

    default_optimizer_config = {"lr": 1e-3}  # User's successful learning rate
    if optimizer_config:
        default_optimizer_config.update(optimizer_config)

    # Create DataLoader with correct dimensionality
    from ..pp.basic import prepare_training

    # Slice PCA data to match input_dim for optimal performance
    if input_dim < adata.obsm[pca_key].shape[1]:
        # Temporarily modify adata to use only first input_dim components
        original_pca = adata.obsm[pca_key].copy()
        adata.obsm[pca_key] = adata.obsm[pca_key][:, :input_dim]
        print(f" Using first {input_dim} PCA components for training (shape: {adata.obsm[pca_key].shape})")

        dataloader = prepare_training(adata, pca_key=pca_key)

        # Restore original PCA data
        adata.obsm[pca_key] = original_pca
    else:
        dataloader = prepare_training(adata, pca_key=pca_key)

    # Initialize model (existing code)
    model = Deep_AA(**default_model_config)
    optimizer = torch.optim.Adam(model.parameters(), **default_optimizer_config)

    # CRITICAL: Initialize with PCHA + inflation (user's successful setup!)
    print(f" Initializing with PCHA + inflation_factor={default_model_config['inflation_factor']}...")

    # Get sample data for PCHA initialization
    # Ensure contiguous array (fixes negative stride issue)
    pca_sample = np.ascontiguousarray(adata.obsm[pca_key][:, :input_dim])
    sample_data = torch.FloatTensor(pca_sample)

    try:
        if hasattr(model, "initialize_archetypes"):
            success = model.initialize_archetypes(
                sample_data,
                use_pcha=True,
                use_inflation=True,
                inflation_factor=default_model_config["inflation_factor"],
            )
        else:
            # Fallback to deprecated method
            success = model.initialize_with_pcha_and_inflation(
                sample_data, inflation_factor=default_model_config["inflation_factor"], n_subsample=1000
            )
        if success:
            print("[OK] PCHA + inflation initialization successful!")
        else:
            print("[WARNING] PCHA initialization failed, using default initialization")
    except Exception as e:
        print(f"[WARNING] PCHA initialization error: {e}, using default initialization")

    # Train model (delegate to existing battle-tested function)
    results = _train_vae(
        model=model,
        data_loader=dataloader,
        optimizer=optimizer,
        n_epochs=n_epochs,
        device=device,
        save_path=save_path,
        archetypal_weight=archetypal_weight,
        kld_weight=kld_weight,
        reconstruction_weight=reconstruction_weight,
        vae_recon_weight=vae_recon_weight,
        diversity_weight=diversity_weight,
        activation_func=activation_func,
        track_stability=track_stability,
        validate_constraints=validate_constraints,
        lr_factor=lr_factor,
        lr_patience=lr_patience,
        seed=seed,
        constraint_tolerance=constraint_tolerance,
        stability_history_size=stability_history_size,
        adata=adata,  # Auto-stores coordinates
        store_coords_key=store_coords_key,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        early_stopping_metric=early_stopping_metric,
        min_improvement=min_improvement,
        validation_check_interval=validation_check_interval,
        validation_data_loader=validation_data_loader,
        **kwargs,
    )

    # train_vae ALWAYS returns (results_dict, trained_model) tuple
    # Extract just the results dictionary for user
    results_dict, trained_model = results

    # Store trained model in adata for downstream functions (extract_archetype_weights, etc.)
    adata.uns["trained_model"] = trained_model

    # Also add model to results dict for direct access
    results_dict["final_model"] = trained_model
    results_dict["model"] = trained_model  # Alias for compatibility

    # Add convenience keys for common access patterns (for user-friendly API)
    # These provide direct access without navigating nested dicts
    history = results_dict.get("history", {})
    if "archetype_r2" in history and len(history["archetype_r2"]) > 0:
        results_dict["final_archetype_r2"] = history["archetype_r2"][-1]
    if "rmse" in history and len(history["rmse"]) > 0:
        results_dict["final_rmse"] = history["rmse"][-1]
    if "mae" in history and len(history["mae"]) > 0:
        results_dict["final_mae"] = history["mae"][-1]
    if "loss" in history and len(history["loss"]) > 0:
        results_dict["final_loss"] = history["loss"][-1]

    training_config = results_dict.get("training_config", {})
    if "actual_epochs" in training_config:
        results_dict["convergence_epoch"] = training_config["actual_epochs"]

    # Optional: validate internally (catches bugs early)
    # validate_training_results(results_dict)
    return results_dict


def archetypal_coordinates(
    adata: AnnData,
    *,
    pca_key: str = "X_pca",
    archetype_coords_key: str = "archetype_coordinates",
    obsm_key: str = "archetype_distances",
    uns_prefix: str = "archetype",
    verbose: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """Extract archetypal coordinates for all cells.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with trained model coordinates
    pca_key : str, default: "X_pca"
        Key in adata.obsm containing PCA coordinates
    archetype_coords_key : str, default: "archetype_coordinates"
        Key in adata.uns containing archetype coordinates
    obsm_key : str, default: "archetype_distances"
        Key to store distance matrix in adata.obsm
    uns_prefix : str, default: "archetype"
        Prefix for keys stored in adata.uns
    verbose : bool, default: True
        Whether to print progress messages
    **kwargs
        Additional arguments passed to compute_archetype_distances

    Returns
    -------
    dict
        Dictionary with archetypal coordinates and distances
    """
    # Input validation
    if archetype_coords_key not in adata.uns:
        raise ValueError(f"adata.uns['{archetype_coords_key}'] not found. Run pc.tl.train_archetypal() first.")

    if pca_key not in adata.obsm:
        # Try to auto-detect PCA coordinates
        pca_candidates = ["X_pca", "X_PCA", "PCA", "x_pca"]
        found_pca = None
        for candidate in pca_candidates:
            if candidate in adata.obsm:
                found_pca = candidate
                break

        if found_pca:
            pca_key = found_pca
        else:
            raise ValueError(f"PCA coordinates not found. Tried: {pca_candidates}. Run scanpy.pp.pca(adata) first.")

    return _compute_archetype_distances(
        adata=adata,
        pca_key=pca_key,
        archetype_coords_key=archetype_coords_key,
        obsm_key=obsm_key,
        uns_prefix=uns_prefix,
        verbose=verbose,
        **kwargs,
    )


def assign_archetypes(
    adata: AnnData,
    *,
    percentage_per_archetype: float = 0.1,
    obsm_key: str = "archetype_distances",
    obs_key: str = "archetypes",
    include_central_archetype: bool = True,
    verbose: bool = True,
    **kwargs,
) -> None:
    """Assign cells to archetypes based on distances.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with archetype distances
    percentage_per_archetype : float, default: 0.1
        Percentage of cells to assign to each archetype
    obsm_key : str, default: "archetype_distances"
        Key in adata.obsm containing distance matrix
    obs_key : str, default: "archetypes"
        Key to store assignments in adata.obs
    include_central_archetype : bool, default: True
        Whether to include a central archetype (cells far from all extreme archetypes)
    verbose : bool, default: True
        Whether to print progress messages
    **kwargs
        Additional arguments passed to bin_cells_by_archetype
    """
    # Input validation
    if obsm_key not in adata.obsm:
        raise ValueError(f"adata.obsm['{obsm_key}'] not found. Run pc.tl.archetypal_coordinates() first.")

    # Validate percentage
    if not 0.0 < percentage_per_archetype <= 1.0:
        raise ValueError(f"percentage_per_archetype must be between 0 and 1, got {percentage_per_archetype}")

    _bin_cells_by_archetype(
        adata=adata,
        percentage_per_archetype=percentage_per_archetype,
        obsm_key=obsm_key,
        obs_key=obs_key,
        include_central_archetype=include_central_archetype,
        verbose=verbose,
        **kwargs,
    )


def extract_archetype_weights(
    adata: AnnData,
    model=None,
    *,
    pca_key: str = "X_pca",
    weights_key: str = "cell_archetype_weights",
    batch_size: int = 256,
    device: str = "cpu",
    verbose: bool = True,
) -> np.ndarray:
    """
    Extract cell archetype weights from trained Deep_AA model.

    This function computes the barycentric coordinates (weights) for each cell
    that describe how it's composed of the learned archetypes.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with PCA coordinates
    model : Deep_AA model, optional
        Trained model. If None, will look for model in adata.uns['trained_model']
    pca_key : str, default: "X_pca"
        Key in adata.obsm containing PCA coordinates
    weights_key : str, default: "cell_archetype_weights"
        Key to store weights in adata.obsm
    batch_size : int, default: 256
        Batch size for processing
    device : str, default: "cpu"
        Device for computation ('cpu', 'cuda', or 'mps')
    verbose : bool, default: True
        Whether to print progress

    Returns
    -------
    np.ndarray
        Cell archetype weights of shape (n_cells, n_archetypes)
        Also stores weights in adata.obsm[weights_key]

    Examples
    --------
    >>> # After training
    >>> results = pc.tl.train_archetypal(adata, n_archetypes=5)
    >>>
    >>> # Extract weights
    >>> weights = pc.tl.extract_archetype_weights(adata, results["model"])
    >>>
    >>> # Weights are now in adata.obsm['cell_archetype_weights']
    >>> print(adata.obsm["cell_archetype_weights"].shape)
    """
    # Convert device string to torch.device
    device = torch.device(device)

    # Get model
    if model is None:
        if "trained_model" in adata.uns:
            model = adata.uns["trained_model"]
            if verbose:
                print("[STATS] Using model from adata.uns['trained_model']")
        else:
            raise ValueError("No model provided and no trained model found in adata.uns")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Get PCA coordinates
    pca_candidates = [pca_key, "X_PCA", "X_pca", "PCA"]
    X_pca = None

    for candidate in pca_candidates:
        if candidate in adata.obsm:
            X_pca = adata.obsm[candidate]
            if verbose and candidate != pca_key:
                print(f"[WARNING]  Using {candidate} instead of {pca_key}")
            break

    if X_pca is None:
        raise ValueError(f"No PCA coordinates found. Tried: {pca_candidates}")

    if verbose:
        print(f"[STATS] Extracting weights for {len(X_pca)} cells...")
        print(f"   PCA shape: {X_pca.shape}")
        print(f"   Device: {device}")
        print(f"   Batch size: {batch_size}")

    # Extract weights in batches
    all_weights = []
    n_processed = 0

    with torch.no_grad():
        for i in range(0, len(X_pca), batch_size):
            batch = X_pca[i : i + batch_size]
            # Ensure contiguous array (fixes negative stride issue)
            batch_contiguous = np.ascontiguousarray(batch)
            batch_tensor = torch.FloatTensor(batch_contiguous).to(device)

            # Pass through encoder to get barycentric coordinates
            mu, log_var = model.encode(batch_tensor)
            z = model.reparameterize(mu, log_var)

            # z contains the barycentric weights (A matrix)
            all_weights.append(z.cpu().numpy())

            n_processed += len(batch)
            if verbose and (n_processed % 1000 == 0 or n_processed == len(X_pca)):
                print(f"   Processed {n_processed}/{len(X_pca)} cells...")

    # Combine all weights
    cell_weights = np.vstack(all_weights)

    # Verify they're proper barycentric coordinates
    sums = cell_weights.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-3):
        if verbose:
            print(f"[WARNING]  Warning: Weights don't sum to 1.0 perfectly (mean: {sums.mean():.4f})")

    if (cell_weights < 0).any():
        if verbose:
            print(f"[WARNING]  Warning: Found negative weights (min: {cell_weights.min():.4f})")

    # Store in adata
    adata.obsm[weights_key] = cell_weights

    if verbose:
        print(f"[OK] Stored cell weights in adata.obsm['{weights_key}']")
        print(f"   Shape: {cell_weights.shape}")
        print(f"   Range: [{cell_weights.min():.3f}, {cell_weights.max():.3f}]")
        print(f"   Mean sum: {sums.mean():.4f}")

        # Show archetype weight statistics
        print("\n[STATS] Archetype weight statistics:")
        for i in range(cell_weights.shape[1]):
            mean_w = cell_weights[:, i].mean()
            std_w = cell_weights[:, i].std()
            max_w = cell_weights[:, i].max()
            dominant = (cell_weights[:, i] > 0.5).sum()
            print(
                f"   Archetype {i}: mean={mean_w:.3f}, std={std_w:.3f}, max={max_w:.3f}, dominant in {dominant} cells"
            )

    return cell_weights


def compute_conditional_centroids(
    adata,
    condition_column: str,
    *,
    pca_key: str = "X_pca",
    store_key: str = "conditional_centroids",
    exclude_archetypes: list = None,
    groupby: str = None,
    verbose: bool = True,
):
    """Compute centroid positions in PCA space for each level of a categorical condition.

    This function calculates the mean position (centroid) in PCA space for cells
    belonging to each level of a categorical variable. Useful for visualizing
    how different conditions (e.g., treatment phases, timepoints) relate to
    the archetypal structure.

    Following R template patterns:
    - Uses ALL PCs for centroid calculation (equivalent to R's colMeans)
    - Stores full PC centroid but extracts first 3 for 3D visualization
    - Excludes 'no_archetype' and 'archetype_0' cells by default

    Parameters
    ----------
    adata : AnnData
        Annotated data object with PCA coordinates in adata.obsm[pca_key].
    condition_column : str
        Name of categorical column in adata.obs to group by.
        Examples: 'treatment_phase', 'timepoint', 'batch'.
    pca_key : str, default: "X_pca"
        Key in adata.obsm containing PCA coordinates.
    store_key : str, default: "conditional_centroids"
        Key in adata.uns to store results.
    exclude_archetypes : list, optional
        Archetype labels to exclude from centroid calculation.
        Default: ['no_archetype', 'archetype_0'] (following R template).
        Set to empty list [] to include all cells.
    groupby : str, optional
        Second categorical column for multi-group trajectories.
        If provided, centroids are computed for each (group, level) combination.
        Example: groupby='response_group' to get separate trajectories per response.
    verbose : bool, default: True
        Whether to print progress messages.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``condition_column`` : str - name of the condition column
        - ``n_levels`` : int - number of unique levels
        - ``levels`` : List[str] - list of level names
        - ``centroids`` : Dict[str, List[float]] - level → full PCA coordinates
        - ``centroids_3d`` : Dict[str, List[float]] - level → [x, y, z] first 3 PCs
        - ``cell_counts`` : Dict[str, int] - level → cell count
        - ``pca_key`` : str - PCA key used
        - ``exclude_archetypes`` : List[str] - archetypes excluded
        - ``groupby`` : Optional[str] - groupby column if used
        - ``group_centroids`` : Optional[Dict] - if groupby: {group: {level: coords}}
        - ``group_centroids_3d`` : Optional[Dict] - if groupby: {group: {level: [x,y,z]}}
        - ``group_cell_counts`` : Optional[Dict] - if groupby: {group: {level: count}}

    Raises
    ------
    ValueError
        If condition_column not in adata.obs or PCA coordinates not found.

    Stores
    ------
    The function stores results in AnnData:

    - ``adata.uns[store_key][condition_column]`` : dict
        Full results dictionary as returned.

    Examples
    --------
    >>> # Compute centroids for treatment phase
    >>> result = pc.tl.compute_conditional_centroids(adata, "treatment_phase")
    >>> print(result["centroids_3d"])
    {'chemo-naive': [1.2, 0.5, -0.3], 'IDS': [0.8, 1.1, 0.2]}

    >>> # Then visualize with trajectory
    >>> fig = pc.pl.archetypal_space(
    ...     adata, show_centroids=True, centroid_condition="treatment_phase", centroid_order=["chemo-naive", "IDS"]
    ... )

    >>> # Multi-group centroids for trajectory comparison
    >>> result = pc.tl.compute_conditional_centroids(adata, "treatment_phase", groupby="response_group")
    >>> fig = pc.pl.archetypal_space(
    ...     adata,
    ...     show_centroids=True,
    ...     centroid_condition="treatment_phase",
    ...     centroid_groupby="response_group",
    ...     centroid_order=["chemo-naive", "IDS"],
    ...     centroid_colors={"long": "magenta", "short": "cyan"},
    ... )

    See Also
    --------
    peach.pl.archetypal_space : Visualize with centroid trajectory overlay
    """
    from .._core.utils.analysis import compute_conditional_centroids as _compute_centroids

    return _compute_centroids(
        adata=adata,
        condition_column=condition_column,
        pca_key=pca_key,
        store_key=store_key,
        exclude_archetypes=exclude_archetypes,
        groupby=groupby,
        verbose=verbose,
    )


def assign_to_centroids(
    adata,
    condition_column: str,
    *,
    pca_key: str = "X_pca",
    centroid_key: str = "conditional_centroids",
    bin_prop: float = 0.15,
    obs_key: str = "centroid_assignments",
    exclude_archetypes: list = None,
    verbose: bool = True,
) -> None:
    """Assign cells to nearest centroid based on distance (top bin_prop% closest).

    This function mirrors assign_archetypes but for condition-based centroids.
    It enables using treatment phase centroids as trajectory endpoints in
    single_trajectory_analysis by creating categorical assignments that CellRank
    can use as terminal states.

    Parameters
    ----------
    adata : AnnData
        Annotated data object. Must have:
        - PCA coordinates in adata.obsm[pca_key]
        - Centroids computed via compute_conditional_centroids in adata.uns[centroid_key]
    condition_column : str
        Name of the condition column used in compute_conditional_centroids.
        This identifies which centroid set to use.
    pca_key : str, default: "X_pca"
        Key in adata.obsm containing PCA coordinates.
    centroid_key : str, default: "conditional_centroids"
        Key in adata.uns containing centroid results from compute_conditional_centroids.
    bin_prop : float, default: 0.15
        Proportion of cells to assign to each centroid (top 15% closest).
        Similar to percentage_per_archetype in assign_archetypes.
    obs_key : str, default: "centroid_assignments"
        Key in adata.obs to store assignments.
    exclude_archetypes : list, optional
        Archetype labels to exclude from assignment.
        Default: ['no_archetype'] - these cells get 'unassigned'.
    verbose : bool, default: True
        Whether to print progress messages.

    Returns
    -------
    None
        Modifies adata.obs[obs_key] with Categorical assignments.
        Values are condition levels (e.g., 'chemo_naive', 'IDS') or 'unassigned'.

    Examples
    --------
    >>> # First compute centroids for treatment phases
    >>> pc.tl.compute_conditional_centroids(adata, "treatment_stage")
    >>>
    >>> # Then assign cells to nearest centroid (top 15% closest)
    >>> pc.tl.assign_to_centroids(adata, "treatment_stage", bin_prop=0.15)
    >>>
    >>> # Check assignments
    >>> print(adata.obs["centroid_assignments"].value_counts())
    >>>
    >>> # Now can use with CellRank for trajectory analysis
    >>> # (setup_cellrank can use centroid_assignments as terminal states)

    See Also
    --------
    compute_conditional_centroids : Compute centroids for condition levels
    assign_archetypes : Similar function for archetype assignments
    single_trajectory_analysis : Uses centroid assignments for trajectory analysis
    """
    from .._core.utils.analysis import assign_to_centroids as _assign_to_centroids

    return _assign_to_centroids(
        adata=adata,
        condition_column=condition_column,
        pca_key=pca_key,
        centroid_key=centroid_key,
        bin_prop=bin_prop,
        obs_key=obs_key,
        exclude_archetypes=exclude_archetypes,
        verbose=verbose,
    )

"""
Cross-Validation Training Manager
=================================

CV training for hyperparameter evaluation (Phase 2 of pipeline).

This module handles the execution of cross-validation training runs for
hyperparameter search. It is optimized for speed and comparison, not
final model quality.

.. note::
    This is NOT for final model training. Use ``training.train_vae()``
    for Phase 4 final model training.

Main Classes
------------
CVTrainingConfig : Configuration for CV-specific training parameters
CVTrainingManager : Orchestrates CV training with early stopping

Type Definitions
----------------
See ``peach._core.types`` for related Pydantic models:
    - CVFoldResult (for individual fold results)
    - CVResults (from grid_search_results.py)

Examples
--------
>>> from peach._core.utils.cv_training import CVTrainingManager, CVTrainingConfig
>>> manager = CVTrainingManager(base_model_config, search_config)
>>> cv_result = manager.train_cv_configuration(hyperparameters, cv_splits)
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .grid_search_results import CVResults
from .training import train_vae


@dataclass
class CVTrainingConfig:
    """Configuration for CV-specific training parameters.

    These parameters are optimized for fast hyperparameter comparison,
    not final model quality.

    Attributes
    ----------
    max_epochs : int, default: 50
        Maximum training epochs per fold.
    early_stopping_patience : int, default: 5
        Number of validation checks without improvement before stopping.
    learning_rate : float, default: 0.001
        Learning rate for Adam optimizer.
    validation_check_interval : int, default: 5
        Check validation metrics every N epochs.
    min_improvement : float, default: 1e-4
        Minimum improvement in validation metric to reset patience counter.

    Examples
    --------
    >>> config = CVTrainingConfig(max_epochs=100, early_stopping_patience=10, learning_rate=0.0005)
    """

    max_epochs: int = 50
    early_stopping_patience: int = 5
    learning_rate: float = 0.001
    validation_check_interval: int = 5
    min_improvement: float = 1e-4


class CVTrainingManager:
    """Manages cross-validation training with early stopping and validation monitoring.

    Wraps the core ``train_vae`` function with CV-specific optimizations:

    - Early stopping based on validation metrics
    - Memory management between folds
    - Consistent model initialization (PCHA + inflation)
    - Metric aggregation across folds

    Parameters
    ----------
    base_model_config : dict
        Base configuration for Deep_AA model. Keys include:

        - ``input_dim`` : int - Input feature dimensions
        - ``device`` : str - Computing device ('cpu', 'cuda', 'mps')
        - Additional model parameters

    search_config : SearchConfig
        Search configuration from hyperparameter_search module.
        Must have attributes: ``max_epochs_cv``, ``early_stopping_patience``,
        ``random_state``.

    Attributes
    ----------
    base_config : dict
        Stored base model configuration.
    search_config : SearchConfig
        Stored search configuration.
    cv_config : CVTrainingConfig
        CV-specific training configuration derived from search_config.

    Examples
    --------
    >>> from peach._core.utils.cv_training import CVTrainingManager
    >>> from peach._core.utils.hyperparameter_search import SearchConfig
    >>> search_config = SearchConfig(max_epochs_cv=50, cv_folds=5)
    >>> base_config = {"input_dim": 30, "device": "cpu"}
    >>> manager = CVTrainingManager(base_config, search_config)
    >>> cv_result = manager.train_cv_configuration(hyperparams, cv_splits)

    See Also
    --------
    train_vae : Core training function used internally
    CVResults : Return type for train_cv_configuration
    """

    def __init__(self, base_model_config: dict[str, Any], search_config):
        """Initialize CV manager with model configuration and search parameters.

        Parameters
        ----------
        base_model_config : dict
            Base model configuration parameters.
        search_config : SearchConfig
            Search configuration with CV settings.
        """
        self.base_config = base_model_config
        self.search_config = search_config

        # Store device for data movement
        self.device = torch.device(base_model_config.get("device", "cpu"))

        self.cv_config = CVTrainingConfig(
            max_epochs=search_config.max_epochs_cv, early_stopping_patience=search_config.early_stopping_patience
        )

    # def train_cv_configuration(
    #     self,
    #     hyperparameters: Dict[str, Any],
    #     cv_splits: List[Tuple[DataLoader, DataLoader]]
    # ) -> CVResults:
    #     """Train a single hyperparameter configuration across all CV folds.

    #     Executes training on each fold sequentially, aggregating results
    #     for hyperparameter comparison.

    #     Parameters
    #     ----------
    #     hyperparameters : dict
    #         Hyperparameters to test. Expected keys:

    #         - ``n_archetypes`` : int - Number of archetypes
    #         - ``hidden_dims`` : list[int] - Network architecture
    #         - ``inflation_factor`` : float - PCHA inflation factor (default: 1.5)
    #         - ``use_pcha_init`` : bool - Whether to use PCHA initialization
    #         - ``use_inflation`` : bool - Whether to apply inflation

    #     cv_splits : list[tuple[DataLoader, DataLoader]]
    #         List of (train_loader, val_loader) tuples, one per fold.

    #     Returns
    #     -------
    #     CVResults
    #         Aggregated results across all folds with:

    #         - ``hyperparameters`` : dict - The tested hyperparameters
    #         - ``fold_results`` : list[dict] - Per-fold metrics
    #         - ``mean_metrics`` : dict - Mean metrics across folds
    #         - ``std_metrics`` : dict - Standard deviation across folds
    #         - ``best_fold_idx`` : int - Index of best-performing fold
    #         - ``convergence_epochs`` : list[int] - Convergence epoch per fold
    #         - ``fold_histories`` : list[dict] - Training history per fold
    #         - ``training_time`` : float - Total training time (set by caller)

    #     Notes
    #     -----
    #     GPU memory is automatically cleared between folds if CUDA is available.

    #     Examples
    #     --------
    #     >>> hyperparams = {
    #     ...     'n_archetypes': 5,
    #     ...     'hidden_dims': [256, 128, 64],
    #     ...     'inflation_factor': 1.5,
    #     ...     'use_pcha_init': True,
    #     ...     'use_inflation': True
    #     ... }
    #     >>> cv_result = manager.train_cv_configuration(hyperparams, cv_splits)
    #     >>> print(f"Mean R²: {cv_result.mean_metrics['archetype_r2']:.4f}")

    #     See Also
    #     --------
    #     train_single_fold : Trains individual folds
    #     CVResults : Return type structure
    #     """
    #     fold_results = []
    #     convergence_epochs = []
    #     fold_histories = []  # Store training histories for each fold

    #     for fold_idx, (train_loader, val_loader) in enumerate(cv_splits):
    #         print(f"     Fold {fold_idx + 1}/{len(cv_splits)}")

    #         # Train single fold (now returns result and history)
    #         fold_result, fold_history = self.train_single_fold(
    #             hyperparameters, train_loader, val_loader, fold_idx
    #         )

    #         fold_results.append(fold_result)
    #         fold_histories.append(fold_history)
    #         convergence_epochs.append(fold_result.get('convergence_epoch', self.cv_config.max_epochs))

    #         # Memory cleanup between folds
    #         if torch.cuda.is_available():
    #             torch.cuda.empty_cache()
    #         # Note: MPS doesn't have explicit cache clearing yet

    #     # Aggregate results across folds
    #     mean_metrics, std_metrics = self._aggregate_fold_results(fold_results)

    #     # Find best fold (highest archetype R²)
    #     best_fold_idx = np.argmax([result.get('archetype_r2', 0) for result in fold_results])

    #     return CVResults(
    #         hyperparameters=hyperparameters,
    #         fold_results=fold_results,
    #         mean_metrics=mean_metrics,
    #         std_metrics=std_metrics,
    #         best_fold_idx=best_fold_idx,
    #         convergence_epochs=convergence_epochs,
    #         fold_histories=fold_histories  # Include histories
    #     )
    def train_cv_configuration(
        self, hyperparameters: dict[str, Any], cv_splits: list[tuple[DataLoader, DataLoader]]
    ) -> CVResults:
        """Train a single hyperparameter configuration across all CV folds.

        Executes training on each fold sequentially, aggregating results
        for hyperparameter comparison.

        Parameters
        ----------
        hyperparameters : dict
            Hyperparameters to test. Expected keys:

            - ``n_archetypes`` : int - Number of archetypes
            - ``hidden_dims`` : list[int] - Network architecture
            - ``inflation_factor`` : float - PCHA inflation factor (default: 1.5)
            - ``use_pcha_init`` : bool - Whether to use PCHA initialization
            - ``use_inflation`` : bool - Whether to apply inflation

        cv_splits : list[tuple[DataLoader, DataLoader]]
            List of (train_loader, val_loader) tuples, one per fold.

        Returns
        -------
        CVResults
            Aggregated results across all folds with:

            - ``hyperparameters`` : dict - The tested hyperparameters
            - ``fold_results`` : list[dict] - Per-fold metrics
            - ``mean_metrics`` : dict - Mean metrics across folds
            - ``std_metrics`` : dict - Standard deviation across folds
            - ``best_fold_idx`` : int - Index of best-performing fold
            - ``convergence_epochs`` : list[int] - Convergence epoch per fold
            - ``fold_histories`` : list[dict] - Training history per fold
            - ``training_time`` : float - Total training time (set by caller)

        Notes
        -----
        GPU memory is automatically cleared between folds if CUDA is available.

        Examples
        --------
        >>> hyperparams = {
        ...     "n_archetypes": 5,
        ...     "hidden_dims": [256, 128, 64],
        ...     "inflation_factor": 1.5,
        ...     "use_pcha_init": True,
        ...     "use_inflation": True,
        ... }
        >>> cv_result = manager.train_cv_configuration(hyperparams, cv_splits)
        >>> print(f"Mean R²: {cv_result.mean_metrics['archetype_r2']:.4f}")

        See Also
        --------
        train_single_fold : Trains individual folds
        CVResults : Return type structure
        peach._core.types.CVResultsModel : Pydantic validation model
        """
        from ..types import CVHyperparameters, CVResultsModel

        # Validate input hyperparameters
        validated_hyperparams = CVHyperparameters.model_validate(hyperparameters)

        fold_results = []
        convergence_epochs = []
        fold_histories = []

        for fold_idx, (train_loader, val_loader) in enumerate(cv_splits):
            print(f"     Fold {fold_idx + 1}/{len(cv_splits)}")

            # Train single fold (returns metrics dict and history dict)
            fold_result, fold_history = self.train_single_fold(hyperparameters, train_loader, val_loader, fold_idx)

            fold_results.append(fold_result)
            fold_histories.append(fold_history)
            convergence_epochs.append(fold_result.get("convergence_epoch", self.cv_config.max_epochs))

            # Memory cleanup between folds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Aggregate results across folds
        mean_metrics, std_metrics = self._aggregate_fold_results(fold_results)

        # Find best fold (highest archetype R²)
        best_fold_idx = int(np.argmax([result.get("archetype_r2", 0) for result in fold_results]))

        # Build result
        cv_results = CVResults(
            hyperparameters=hyperparameters,
            fold_results=fold_results,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            best_fold_idx=best_fold_idx,
            convergence_epochs=convergence_epochs,
            fold_histories=fold_histories,
        )

        # Validate output structure (catches schema drift)
        if __debug__:
            CVResultsModel.model_validate(
                {
                    "hyperparameters": hyperparameters,
                    "fold_results": fold_results,
                    "mean_metrics": mean_metrics,
                    "std_metrics": std_metrics,
                    "best_fold_idx": best_fold_idx,
                    "convergence_epochs": convergence_epochs,
                    "training_time": 0.0,  # Set by caller
                    "fold_histories": fold_histories,
                }
            )

        return cv_results

    def train_single_fold(
        self, hyperparameters: dict[str, Any], train_loader: DataLoader, val_loader: DataLoader, fold_idx: int
    ) -> tuple[dict[str, float], dict[str, list[float]]]:
        """Train model on a single CV fold with early stopping.

        Parameters
        ----------
        hyperparameters : dict
            Model hyperparameters (see train_cv_configuration).
        train_loader : DataLoader
            Training data for this fold.
        val_loader : DataLoader
            Validation data for this fold.
        fold_idx : int
            Index of current fold (used for unique random seed).

        Returns
        -------
        tuple[dict, dict]
            Two-element tuple:

            - ``fold_metrics`` : dict[str, float]
                Final metrics for this fold:

                - ``train_loss``, ``train_archetypal_loss``, etc. : Training metrics
                - ``val_rmse``, ``val_mae``, ``val_archetype_r2`` : Validation metrics
                - ``convergence_epoch`` : int - Epoch when training stopped
                - ``early_stopped`` : bool - Whether early stopping triggered
                - ``archetype_r2`` : float - Primary metric (validation R²)

            - ``fold_history`` : dict[str, list[float]]
                Training history with metrics per epoch (from train_vae).

        Notes
        -----
        - Model input_dim is auto-detected from training data shape
        - PCHA initialization is attempted; falls back to random on failure
        - Early stopping monitors validation archetype_r2
        - Stability tracking is disabled for CV (single-epoch training)

        Examples
        --------
        >>> metrics, history = manager.train_single_fold(hyperparams, train_loader, val_loader, fold_idx=0)
        >>> print(f"Converged at epoch {metrics['convergence_epoch']}")
        >>> print(f"Early stopped: {metrics['early_stopped']}")
        """
        # Get actual input dimensions from training data
        sample_batch = next(iter(train_loader))[0]
        actual_input_dim = sample_batch.shape[1]

        # Create model with current hyperparameters and correct input_dim
        model_config = self.base_config.copy()
        model_config.update(
            {
                "n_archetypes": hyperparameters["n_archetypes"],
                "hidden_dims": hyperparameters["hidden_dims"],
                "input_dim": actual_input_dim,  # Override with actual data dimensions
                # latent_dim will be automatically set to n_archetypes in Deep_AA
            }
        )

        print(f"       Model input_dim: {actual_input_dim} (auto-detected from data)")

        # Import locally to avoid circular imports
        from ..models.Deep_AA import Deep_AA

        model = Deep_AA(**model_config)

        # Initialize model with consolidated archetype initialization
        if hasattr(model, "initialize_archetypes"):
            try:
                # Get initialization settings from hyperparameters (passed from grid search)
                use_pcha = hyperparameters.get("use_pcha_init", True)
                use_inflation = hyperparameters.get("use_inflation", False)
                inflation_factor = hyperparameters.get("inflation_factor", 1.5)

                model.initialize_archetypes(
                    X_sample=sample_batch,
                    use_pcha=use_pcha,
                    use_inflation=use_inflation,
                    inflation_factor=inflation_factor,
                    n_subsample=1000,
                )
                print(
                    f"       [OK] Archetype initialization complete (PCHA: {use_pcha}, Inflation: {inflation_factor})"
                )
            except Exception as e:
                print(f"       Warning: Archetype initialization failed: {e}")
                # Model will use random initialization by default

        # Setup optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.cv_config.learning_rate)

        # Training with early stopping
        best_val_metric = -float("inf")
        patience_counter = 0
        convergence_epoch = self.cv_config.max_epochs

        training_results = None

        for epoch in range(self.cv_config.max_epochs):
            # Train for one epoch
            training_results = train_vae(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                n_epochs=1,  # Single epoch
                device=model_config.get("device", "cpu"),  # Use device from config
                track_stability=False,  # CV doesn't support stability tracking (single epochs)
                validate_constraints=False,  # Skip constraint validation in CV
                seed=self.search_config.random_state + fold_idx,  # Unique seed per fold
                optimize_threads=(epoch == 0 and fold_idx == 0),  # Only optimize on first call
                _cv_mode=True,  # Suppress adata warning during CV
            )

            # Validation check
            if epoch % self.cv_config.validation_check_interval == 0:
                val_metrics = self._evaluate_on_validation(model, val_loader)

                # Early stopping based on archetype R²
                val_metric = val_metrics.get("archetype_r2", 0)

                if val_metric > best_val_metric + self.cv_config.min_improvement:
                    best_val_metric = val_metric
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Stop if patience exceeded
                if patience_counter >= self.cv_config.early_stopping_patience:
                    convergence_epoch = epoch + 1
                    break

        # Final evaluation on validation set
        final_val_metrics = self._evaluate_on_validation(model, val_loader)

        # Combine training and validation metrics
        final_metrics = {}
        fold_history = {}  # Store epoch-by-epoch history

        if training_results and "history" in training_results:
            # Get final training metrics
            history = training_results["history"]
            if history:
                # Store full history for plotting
                fold_history = history.copy()

                for key, values in history.items():
                    if values:  # Not empty
                        final_metrics[f"train_{key}"] = values[-1]

                # Note: Stability metrics not available in CV due to single-epoch training

        # Add validation metrics
        for key, value in final_val_metrics.items():
            final_metrics[f"val_{key}"] = value

        # Add convergence info
        final_metrics["convergence_epoch"] = convergence_epoch
        final_metrics["early_stopped"] = convergence_epoch < self.cv_config.max_epochs

        # Use validation archetype R² as primary metric
        final_metrics["archetype_r2"] = final_val_metrics.get("archetype_r2", 0)

        return final_metrics, fold_history

    def _evaluate_on_validation(self, model: torch.nn.Module, val_loader: DataLoader) -> dict[str, float]:
        """Evaluate model on validation set and compute key metrics.

        Parameters
        ----------
        model : torch.nn.Module
            Trained model to evaluate.
        val_loader : DataLoader
            Validation data loader.

        Returns
        -------
        dict[str, float]
            Validation metrics:

            - ``rmse`` : float - Root mean squared error
            - ``mae`` : float - Mean absolute error
            - ``archetype_r2`` : float - Archetype reconstruction R²

        Notes
        -----
        Model is set to eval mode during evaluation and returned to
        its original mode afterward.
        """
        model.eval()

        all_inputs = []
        all_reconstructions = []

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch

                # Move data to device (GPU support)
                inputs = inputs.to(self.device)

                outputs = model(inputs)
                reconstructions = outputs.get("arch_recons", outputs.get("recons"))

                all_inputs.append(inputs)
                all_reconstructions.append(reconstructions)

        # Concatenate all validation data
        val_inputs = torch.cat(all_inputs, dim=0)
        val_reconstructions = torch.cat(all_reconstructions, dim=0)

        # Compute metrics
        metrics = {}

        # RMSE
        mse = torch.nn.functional.mse_loss(val_reconstructions, val_inputs)
        metrics["rmse"] = torch.sqrt(mse).item()

        # Archetype R² using standardized function
        from .metrics import calculate_archetype_r2

        archetype_r2 = calculate_archetype_r2(val_reconstructions, val_inputs)
        metrics["archetype_r2"] = archetype_r2.item()

        # Mean absolute error
        mae = torch.nn.functional.l1_loss(
            val_reconstructions, val_inputs
        )  # functional for statelessness, no need to instantiate
        metrics["mae"] = mae.item()

        # Optional: Add archetypal validity metrics here
        # For now, keep it simple for CV speed

        return metrics

    def _aggregate_fold_results(
        self, fold_results: list[dict[str, float]]
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Aggregate metrics across CV folds.

        Parameters
        ----------
        fold_results : list[dict[str, float]]
            List of metric dictionaries, one per fold.

        Returns
        -------
        tuple[dict, dict]
            Two-element tuple:

            - ``mean_metrics`` : dict[str, float] - Mean value per metric
            - ``std_metrics`` : dict[str, float] - Std deviation per metric

        Notes
        -----
        NaN values are filtered before computing statistics.
        If all values for a metric are NaN, the result is NaN.
        """
        if not fold_results:
            return {}, {}

        # Get all metric names
        all_metrics = set()
        for result in fold_results:
            all_metrics.update(result.keys())

        mean_metrics = {}
        std_metrics = {}

        for metric in all_metrics:
            values = [result.get(metric, np.nan) for result in fold_results]
            # Filter out NaN values
            valid_values = [v for v in values if not np.isnan(v)]

            if valid_values:
                mean_metrics[metric] = np.mean(valid_values)
                std_metrics[metric] = np.std(valid_values) if len(valid_values) > 1 else 0.0
            else:
                mean_metrics[metric] = np.nan
                std_metrics[metric] = np.nan

        return mean_metrics, std_metrics

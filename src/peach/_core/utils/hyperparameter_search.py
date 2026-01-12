"""
Cross-Validation Hyperparameter Search for Archetypal Analysis
==============================================================

Phase 2 of the PEACH pipeline: systematic hyperparameter evaluation.

This module provides grid search over hyperparameter combinations using
cross-validation to estimate model performance. Results support manual
selection in Phase 3 - NO automatic selection is performed.

Pipeline Position
-----------------
Phase 1: Data Loading → **Phase 2: CV Search** → Phase 3: Manual Selection
→ Phase 4: Final Training → Phase 5: Evaluation

Main Classes
------------
SearchConfig : Configuration for hyperparameter search space and CV settings
ArchetypalGridSearch : Main orchestrator for grid search with cross-validation

Type Definitions
----------------
See ``peach._core.types`` for Pydantic models of return structures.

Examples
--------
>>> from peach._core.utils.hyperparameter_search import ArchetypalGridSearch, SearchConfig
>>> config = SearchConfig(n_archetypes_range=[3, 4, 5, 6], cv_folds=5, max_epochs_cv=50)
>>> grid_search = ArchetypalGridSearch(config)
>>> cv_summary = grid_search.fit(dataloader, base_model_config)
>>> print(cv_summary.summary_report())
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

from .cv_training import CVTrainingManager
from .grid_search_results import CVResults, CVSummary


@dataclass
class SearchConfig:
    """Configuration for hyperparameter search space and CV settings.

    Defines the hyperparameter grid to search and cross-validation parameters.
    Speed presets automatically adjust epochs and early stopping for different
    use cases.

    Parameters
    ----------
    n_archetypes_range : list[int] | None, default: None
        Range of archetype numbers to test. If None, defaults to [2, 3, 4, 5, 6, 7].
    hidden_dims_options : list[list[int]] | None, default: None
        Network architectures to test. If None, defaults to standard options:
        [[128, 64], [256, 128, 64], [128], [512, 256, 128]].
    inflation_factor_range : list[float] | None, default: None
        Inflation factors to test. If None, uses [1.5] (Helsinki optimal).
        Set to multiple values (e.g., [1.0, 1.5, 2.0]) to search inflation.
    cv_folds : int, default: 5
        Number of cross-validation folds.
    max_epochs_cv : int, default: 100
        Maximum epochs per CV fold (overridden by speed_preset).
    early_stopping_patience : int, default: 5
        Patience for early stopping (overridden by speed_preset).
    subsample_fraction : float, default: 0.5
        Fraction of data to use for CV when dataset > max_cells_cv.
    max_cells_cv : int, default: 15000
        Maximum cells for CV. Larger datasets are subsampled.
    speed_preset : str, default: "balanced"
        Training speed preset. Options:

        - ``"fast"`` : 25 epochs, patience=3 (quick exploration)
        - ``"balanced"`` : 50 epochs, patience=5 (recommended)
        - ``"thorough"`` : 100 epochs, patience=8 (comprehensive)

    use_pcha_init : bool, default: True
        Whether to use PCHA initialization for archetypes.
    random_state : int, default: 42
        Random seed for reproducibility.

    Attributes
    ----------
    _search_inflation : bool
        Internal flag indicating whether inflation is being searched
        (True if inflation_factor_range was explicitly provided).

    Raises
    ------
    ValueError
        If n_archetypes_range contains non-positive integers.
        If cv_folds <= 0.
        If max_epochs_cv <= 0.
        If subsample_fraction not in (0, 1].
        If max_cells_cv <= 0.
        If inflation_factor_range contains non-positive values.

    Examples
    --------
    >>> # Basic configuration
    >>> config = SearchConfig(n_archetypes_range=[3, 4, 5], cv_folds=3, speed_preset="fast")
    >>> # Search inflation factors
    >>> config = SearchConfig(n_archetypes_range=[4, 5, 6], inflation_factor_range=[1.0, 1.5, 2.0], cv_folds=5)
    >>> print(f"Searching inflation: {config._search_inflation}")  # True

    See Also
    --------
    ArchetypalGridSearch : Uses this configuration
    peach.tl.hyperparameter_search : User-facing wrapper
    """

    n_archetypes_range: list[int] = None
    hidden_dims_options: list[list[int]] = None
    cv_folds: int = 5
    max_epochs_cv: int = 100
    early_stopping_patience: int = 5
    subsample_fraction: float = 0.5
    max_cells_cv: int = 15000
    speed_preset: str = "balanced"
    use_pcha_init: bool = True
    inflation_factor_range: list[float] = None
    random_state: int = 42

    def __post_init__(self):
        """Initialize defaults and validate inputs."""
        # Set defaults first
        if self.n_archetypes_range is None:
            self.n_archetypes_range = [2, 3, 4, 5, 6, 7]
        if self.hidden_dims_options is None:
            self.hidden_dims_options = [
                [128, 64],  # Standard architecture
                [256, 128, 64],  # Deeper network
                [128],  # Simpler network
                [512, 256, 128],  # Larger capacity
            ]

        # NEW: Inflation factor handling
        # If None, use default optimal value (Helsinki breakthrough: 1.5)
        # If provided as list, test multiple values
        if self.inflation_factor_range is None:
            self.inflation_factor_range = [1.5]  # Default optimal
            self._search_inflation = False  # Flag: not searching inflation
        else:
            self._search_inflation = True  # Flag: actively searching inflation

        # Validate inputs
        if not self.n_archetypes_range or not all(n > 0 for n in self.n_archetypes_range):
            raise ValueError("n_archetypes_range must contain positive integers")

        if self.cv_folds <= 0:
            raise ValueError("cv_folds must be positive")

        if self.max_epochs_cv <= 0:
            raise ValueError("max_epochs_cv must be positive")

        if not (0 < self.subsample_fraction <= 1.0):
            raise ValueError("subsample_fraction must be between 0 and 1")

        if self.max_cells_cv <= 0:
            raise ValueError("max_cells_cv must be positive")

        # Validate inflation factors
        if not all(f > 0 for f in self.inflation_factor_range):
            raise ValueError("inflation_factor_range must contain positive values")


class ArchetypalGridSearch:
    """Main orchestrator for hyperparameter grid search with cross-validation.

    Performs systematic search over hyperparameter combinations, evaluating
    each configuration using K-fold cross-validation. Designed for large-scale
    single-cell datasets with intelligent subsampling and memory management.

    Parameters
    ----------
    search_config : SearchConfig | None, default: None
        Search configuration. If None, uses default SearchConfig().

    Attributes
    ----------
    config : SearchConfig
        Active search configuration.
    cv_manager : CVTrainingManager | None
        CV training manager (initialized during fit).
    results : CVSummary | None
        Search results (populated after fit).

    Examples
    --------
    >>> from peach._core.utils.hyperparameter_search import ArchetypalGridSearch, SearchConfig
    >>> # Configure search
    >>> config = SearchConfig(
    ...     n_archetypes_range=[3, 4, 5, 6],
    ...     hidden_dims_options=[[256, 128], [128, 64]],
    ...     cv_folds=5,
    ...     speed_preset="balanced",
    ... )
    >>> # Run search
    >>> grid_search = ArchetypalGridSearch(config)
    >>> cv_summary = grid_search.fit(dataloader, base_model_config)
    >>> # Analyze results
    >>> print(cv_summary.summary_report())
    >>> top_configs = cv_summary.rank_by_metric("archetype_r2")[:3]
    >>> fig = cv_summary.plot_elbow_r2()

    See Also
    --------
    SearchConfig : Configuration class
    CVSummary : Return type of fit()
    peach.tl.hyperparameter_search : User-facing wrapper
    """

    def __init__(self, search_config: SearchConfig = None):
        """Initialize grid search with configuration.

        Parameters
        ----------
        search_config : SearchConfig | None
            Search configuration. Defaults to SearchConfig() if None.
        """
        self.config = search_config or SearchConfig()
        self.cv_manager = None
        self.results = None
        self._best_model = None

        # Set random seeds for reproducibility
        torch.manual_seed(self.config.random_state)
        np.random.seed(self.config.random_state)

        # Setup speed preset configurations
        self.speed_presets = {
            "fast": {"max_epochs_cv": 25, "early_stopping_patience": 3},
            "balanced": {"max_epochs_cv": 50, "early_stopping_patience": 5},
            "thorough": {"max_epochs_cv": 100, "early_stopping_patience": 8},
        }

        # Apply speed preset
        preset_config = self.speed_presets.get(self.config.speed_preset, {})
        for key, value in preset_config.items():
            setattr(self.config, key, value)

    def fit(
        self, dataloader: DataLoader, base_model_config: dict[str, Any], compute_strategy: str = "sequential"
    ) -> CVSummary:
        """Execute hyperparameter grid search with cross-validation.

        Searches all combinations of hyperparameters, evaluating each with
        K-fold cross-validation. Results are organized for manual selection.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader containing full dataset. Will be subsampled if
            larger than ``config.max_cells_cv``.
        base_model_config : dict
            Base configuration for Deep_AA model:

            - ``input_dim`` : int - Input feature dimensions
            - ``device`` : str - Computing device ('cpu', 'cuda', 'mps')
            - Additional model parameters

        compute_strategy : str, default: "sequential"
            Execution strategy. Currently only "sequential" is implemented.
            "parallel" is reserved for future HPC support.

        Returns
        -------
        CVSummary
            Complete cross-validation results with:

            - ``config_results`` : dict[str, CVResults] - Results per configuration
            - ``summary_df`` : pd.DataFrame - Summary table for analysis
            - ``ranked_configs`` : list[dict] - Configurations ranked by R²
            - ``cv_info`` : dict - Search metadata

            Key methods on CVSummary:

            - ``summary_report()`` : Text summary for decision support
            - ``rank_by_metric(metric)`` : Rank configs by any metric
            - ``plot_elbow_r2()`` : Elbow curve visualization
            - ``plot_metric(metric)`` : Generic metric visualization
            - ``save(path)`` / ``load(path)`` : Persistence

        Notes
        -----
        - Large datasets (> max_cells_cv) are automatically subsampled
        - GPU memory is cleared between configurations
        - Results are also stored in ``self.results`` for later access

        Examples
        --------
        >>> base_config = {"input_dim": adata.obsm["X_pca"].shape[1], "device": "cuda"}
        >>> cv_summary = grid_search.fit(dataloader, base_config)
        >>> # Quick summary
        >>> print(cv_summary.summary_report())
        >>> # Get top 3 configurations
        >>> top3 = grid_search.get_top_configurations(top_k=3)
        >>> for config in top3:
        ...     print(f"{config['config_summary']}: R²={config['metric_value']:.4f}")

        See Also
        --------
        CVSummary : Return type with analysis methods
        get_top_configurations : Convenience method for top configs
        """
        print(" Starting Archetypal Hyperparameter Grid Search")
        if self.config._search_inflation:
            print(
                f"   Search space: {len(self.config.n_archetypes_range)} × {len(self.config.hidden_dims_options)} × {len(self.config.inflation_factor_range)} = {len(self._get_hyperparameter_combinations())} combinations"
            )
            print(f"   Searching inflation factors: {self.config.inflation_factor_range}")
        else:
            print(
                f"   Search space: {len(self.config.n_archetypes_range)} × {len(self.config.hidden_dims_options)} = {len(self._get_hyperparameter_combinations())} combinations"
            )
            print(f"   Using fixed inflation: {self.config.inflation_factor_range[0]}")
        print(f"   CV folds: {self.config.cv_folds}")
        print(f"   Total training runs: {len(self._get_hyperparameter_combinations()) * self.config.cv_folds}")

        # Prepare data with intelligent subsampling
        cv_splits = self._prepare_cv_data(dataloader)

        # Initialize CV training manager
        self.cv_manager = CVTrainingManager(base_model_config, self.config)

        # Generate hyperparameter combinations
        hyperparameter_combinations = self._get_hyperparameter_combinations()

        # Execute grid search
        cv_results = []
        total_combinations = len(hyperparameter_combinations)

        for i, hyperparams in enumerate(hyperparameter_combinations):
            print(f"\n🧪 Configuration {i + 1}/{total_combinations}: {hyperparams}")

            start_time = time.time()
            cv_result = self.cv_manager.train_cv_configuration(hyperparams, cv_splits)
            elapsed_time = time.time() - start_time

            cv_result.training_time = elapsed_time
            cv_results.append(cv_result)

            # Memory cleanup
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            print(f"   [OK] Completed in {elapsed_time:.1f}s")
            print(f"   [STATS] Mean archetype R²: {cv_result.mean_metrics.get('archetype_r2', 0):.4f}")
            print(f"   [STATS] Mean validation R²: {cv_result.mean_metrics.get('val_archetype_r2', 0):.4f}")

        # Compile results using new simplified architecture
        self.results = CVSummary.from_cv_results(
            cv_results=cv_results, search_config=self.config, data_info=self._get_data_info(dataloader)
        )

        print("\n Grid search completed!")
        print(f"   Best configuration: {self.results.ranked_configs[0]['config_summary']}")
        print(f"   Archetype R²: {self.results.ranked_configs[0]['metric_value']:.4f}")

        return self.results

    def _prepare_cv_data(self, dataloader: DataLoader) -> list[tuple[DataLoader, DataLoader]]:
        """Prepare cross-validation data splits with intelligent subsampling.

        Parameters
        ----------
        dataloader : DataLoader
            Original full dataset loader.

        Returns
        -------
        list[tuple[DataLoader, DataLoader]]
            List of (train_loader, val_loader) tuples, one per fold.

        Notes
        -----
        - Datasets > max_cells_cv are subsampled to max_cells_cv
        - KFold splitting with shuffle for randomization
        - DataLoader workers optimized for HPC environments
        """
        # Extract full dataset
        full_data = []
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                full_data.append(batch[0])
            else:
                full_data.append(batch)

        full_data = torch.cat(full_data, dim=0)
        n_total_cells = len(full_data)

        print(f"[STATS] Dataset info: {n_total_cells:,} cells, {full_data.shape[1]} features")

        # Determine subsampling strategy
        if n_total_cells > self.config.max_cells_cv:
            # Subsample for CV
            n_cv_cells = min(self.config.max_cells_cv, int(n_total_cells * self.config.subsample_fraction))

            # Stratified subsampling (simple random for now, could add PCA-based stratification)
            subsample_indices = torch.randperm(n_total_cells)[:n_cv_cells]
            cv_data = full_data[subsample_indices]

            print(f" Subsampled to {n_cv_cells:,} cells ({n_cv_cells / n_total_cells * 100:.1f}%) for CV")
        else:
            cv_data = full_data
            n_cv_cells = n_total_cells
            print(f" Using full dataset for CV ({n_cv_cells:,} cells)")

        # Create KFold splits
        kfold = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)

        cv_splits = []
        for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(cv_data)):
            # Create datasets
            train_data = cv_data[train_indices]
            val_data = cv_data[val_indices]

            # Create dataloaders
            train_dataset = TensorDataset(train_data)
            val_dataset = TensorDataset(val_data)

            # Get optimized DataLoader settings from original loader if available
            num_workers = getattr(dataloader, "num_workers", 0)
            pin_memory = getattr(dataloader, "pin_memory", False)

            # Auto-detect optimal settings if not provided
            if num_workers == 0 and not hasattr(torch.backends, "mps"):
                # Check for HPC environment
                if any([os.environ.get("SLURM_JOB_ID"), os.environ.get("PBS_JOBID"), (os.cpu_count() or 1) > 16]):
                    num_workers = min(6, max(4, (os.cpu_count() or 1) - 2))
                    if torch.cuda.is_available():
                        pin_memory = True

            # Build optimized DataLoader kwargs
            loader_kwargs = {
                "batch_size": dataloader.batch_size,
                "num_workers": num_workers,
                "pin_memory": pin_memory and torch.cuda.is_available(),
            }

            if num_workers > 0:
                loader_kwargs["persistent_workers"] = True
                loader_kwargs["prefetch_factor"] = 2

            train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
            val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

            cv_splits.append((train_loader, val_loader))

            print(f"   Fold {fold_idx + 1}: {len(train_data):,} train, {len(val_data):,} val")

        # Report DataLoader optimization
        if num_workers > 0:
            env_type = "HPC" if num_workers >= 4 else "Local"
            print(f"\n[STATS] DataLoader optimization: {env_type} mode with {num_workers} workers")
            if pin_memory:
                print("   GPU optimizations enabled (pin_memory=True)")

        return cv_splits

    def _get_hyperparameter_combinations(self) -> list[dict[str, Any]]:
        """Generate all hyperparameter combinations to test.

        Returns
        -------
        list[dict]
            Hyperparameter combinations, each dict containing:

            - ``n_archetypes`` : int
            - ``hidden_dims`` : list[int]
            - ``inflation_factor`` : float
            - ``use_pcha_init`` : bool
            - ``use_inflation`` : bool (True if inflation_factor > 1.0)

        Notes
        -----
        Generates Cartesian product of:
        n_archetypes × hidden_dims × inflation_factor
        """
        combinations = []

        for n_archetypes in self.config.n_archetypes_range:
            for hidden_dims in self.config.hidden_dims_options:
                for inflation_factor in self.config.inflation_factor_range:
                    combinations.append(
                        {
                            "n_archetypes": n_archetypes,
                            "hidden_dims": hidden_dims,
                            "inflation_factor": inflation_factor,
                            "use_pcha_init": self.config.use_pcha_init,
                            "use_inflation": inflation_factor > 1.0,  # Auto-enable if factor > 1.0
                            # latent_dim is automatically set to n_archetypes in Deep_AA
                        }
                    )

        return combinations

    def _get_data_info(self, dataloader: DataLoader) -> dict[str, Any]:
        """Extract dataset information for results metadata.

        Parameters
        ----------
        dataloader : DataLoader
            Data loader to analyze.

        Returns
        -------
        dict
            Dataset metadata:

            - ``n_total_samples`` : int - Total number of samples
            - ``n_features`` : int - Number of input features
            - ``batch_size`` : int - Batch size
            - ``device`` : str - Device of data tensors
        """
        # Get sample batch to determine data characteristics
        sample_batch = next(iter(dataloader))
        if isinstance(sample_batch, (list, tuple)):
            sample_data = sample_batch[0]
        else:
            sample_data = sample_batch

        return {
            "n_total_samples": len(dataloader.dataset),
            "n_features": sample_data.shape[1],
            "batch_size": dataloader.batch_size,
            "device": str(sample_data.device),
        }

    def get_top_configurations(self, metric: str = "archetype_r2", top_k: int = 5) -> list[dict[str, Any]]:
        """Get top-k configurations ranked by specified metric.

        Convenience method wrapping ``CVSummary.rank_by_metric()``.

        Parameters
        ----------
        metric : str, default: "archetype_r2"
            Metric to rank by. Common options:

            - ``"archetype_r2"`` : Reconstruction R² (higher is better)
            - ``"rmse"`` : Root mean squared error (lower is better)
            - ``"val_rmse"`` : Validation RMSE
            - ``"convergence_epoch"`` : Training convergence speed

        top_k : int, default: 5
            Number of top configurations to return.

        Returns
        -------
        list[dict]
            Top configurations, each dict containing:

            - ``hyperparameters`` : dict - Configuration parameters
            - ``metric_value`` : float - Value of ranking metric
            - ``std_error`` : float - Standard error across folds
            - ``config_summary`` : str - Human-readable summary

        Raises
        ------
        ValueError
            If ``fit()`` has not been called yet.

        Examples
        --------
        >>> top_configs = grid_search.get_top_configurations(metric="archetype_r2", top_k=3)
        >>> for i, config in enumerate(top_configs, 1):
        ...     print(f"{i}. {config['config_summary']}")
        ...     print(f"   R²: {config['metric_value']:.4f} ± {config['std_error']:.4f}")
        """
        if self.results is None:
            raise ValueError("Must run fit() before getting configurations")

        return self.results.rank_by_metric(metric)[:top_k]

    def save_results(self, path: str | Path) -> None:
        """Save CV summary to disk.

        Parameters
        ----------
        path : str | Path
            File path for saving (pickle format).

        Raises
        ------
        ValueError
            If ``fit()`` has not been called yet.

        Examples
        --------
        >>> grid_search.fit(dataloader, base_config)
        >>> grid_search.save_results("cv_results.pkl")
        """
        if self.results is None:
            raise ValueError("No results to save. Run fit() first.")

        self.results.save(path)

    def load_results(self, path: str | Path) -> CVSummary:
        """Load CV summary from disk.

        Parameters
        ----------
        path : str | Path
            File path to load from.

        Returns
        -------
        CVSummary
            Loaded results (also stored in self.results).

        Examples
        --------
        >>> cv_summary = grid_search.load_results("cv_results.pkl")
        >>> print(cv_summary.summary_report())
        """
        self.results = CVSummary.load(path)
        return self.results

    # Future extension points for parallelization
    def _execute_parallel(self, combinations: list[dict], cv_splits: list) -> list[CVResults]:
        """Future: Parallel execution of grid search."""
        # TODO: Implement when moving to HPC with GPU clusters
        # Could use multiprocessing, Ray, or joblib for parallelization
        raise NotImplementedError("Parallel execution not yet implemented")

    def _setup_gpu_strategy(self) -> None:
        """Future: Setup multi-GPU strategy for large-scale training."""
        # TODO: Implement GPU cluster support
        # Could use torch.distributed or similar for multi-GPU training
        raise NotImplementedError("GPU cluster support not yet implemented")

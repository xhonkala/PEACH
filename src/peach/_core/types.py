# src/peach/_core/types.py
"""
Canonical type definitions for PEACH return structures.

THIS IS THE SINGLE SOURCE OF TRUTH FOR ALL RETURN TYPES.

Developer Notes:
    1. ALWAYS check this file before accessing dict keys
    2. Use .get() for Optional fields marked with `= None`
    3. Required fields will raise ValidationError if missing
    4. Run: `python -c "from peach._core.types import TrainingResults; print(TrainingResults.model_fields.keys())"`

Usage:
    from peach._core.types import TrainingResults, TrainingHistory, ArchetypalCoordinates

    # Validate a results dict:
    validated = TrainingResults.model_validate(results_dict)

    # Access with autocomplete:
    validated.final_archetype_r2  # IDE knows this exists

Version: 0.3.0
Last Updated: 2025-01-XX (from code analysis)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

# =============================================================================
# NUMPY ARRAY HANDLING FOR PYDANTIC
# =============================================================================


class NumpyArrayModel(BaseModel):
    """Base model that allows numpy arrays."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


# =============================================================================
# TRAINING HISTORY METRICS
# =============================================================================


class TrainingHistory(BaseModel):
    """Training metrics tracked per epoch.

    All fields are optional because tracking depends on configuration:
    - Stability metrics: require track_stability=True
    - Constraint metrics: require validate_constraints=True
    - Validation metrics: require early_stopping=True
    - archetype_transform metrics: Deep_AA model-specific

    Core Metrics (almost always present):
        loss, archetypal_loss, archetype_r2, rmse

    Example:
        >>> history = results["history"]
        >>> if history.get("archetype_r2"):
        ...     final_r2 = history["archetype_r2"][-1]
    """

    model_config = ConfigDict(extra="allow")  # Allow additional metrics

    # Core loss metrics (always tracked)
    loss: list[float] | None = None
    archetypal_loss: list[float] | None = None
    kld_loss: list[float] | None = None
    reconstruction_loss: list[float] | None = None
    KLD: list[float] | None = None  # Alias used in some code paths

    # Performance metrics (always tracked)
    archetype_r2: list[float] | None = None
    rmse: list[float] | None = None
    mae: list[float] | None = None

    # Stability metrics (track_stability=True)
    archetype_drift_mean: list[float] | None = None
    archetype_drift_max: list[float] | None = None
    archetype_drift_std: list[float] | None = None
    archetype_variance_mean: list[float] | None = None
    archetype_variance_max: list[float] | None = None
    archetype_variance_std: list[float] | None = None

    # Constraint metrics (validate_constraints=True)
    constraint_violation_rate: list[float] | None = None
    constraints_satisfied: list[float] | None = None  # 0.0 or 1.0 per epoch
    A_sum_error: list[float] | None = None
    A_negative_fraction: list[float] | None = None
    B_sum_error: list[float] | None = None
    B_negative_fraction: list[float] | None = None

    # Archetype transform tracking (Deep_AA specific)
    archetype_transform_grad_norm: list[float] | None = None
    archetype_transform_grad_mean: list[float] | None = None
    archetype_transform_mean: list[float] | None = None
    archetype_transform_std: list[float] | None = None
    archetype_transform_norm: list[float] | None = None

    # Validation metrics (early_stopping=True)
    val_loss: list[float] | None = None
    val_archetype_r2: list[float] | None = None
    val_rmse: list[float] | None = None

    # Convergence tracking
    loss_delta: list[float] | None = None


class TrainingConfig(BaseModel):
    """Training configuration parameters.

    Returned in results['training_config'].
    All fields present after training completes.
    """

    n_epochs: int = Field(description="Requested number of epochs")
    actual_epochs: int = Field(description="Actual epochs run (may be less if early stopped)")
    early_stop_triggered: bool = Field(description="Whether early stopping was triggered")
    archetypal_weight: float
    kld_weight: float
    reconstruction_weight: float
    activation_func: str
    seed: int
    constraint_tolerance: float
    stability_history_size: int
    early_stopping: bool
    early_stopping_patience: int | None = Field(None, description="None if early_stopping=False")
    early_stopping_metric: str | None = Field(
        None, description="None if early_stopping=False. One of: 'archetype_r2', 'loss', 'rmse'"
    )


class ConstraintValidation(BaseModel):
    """Constraint validation results from model.validate_constraints().

    Returned in results['final_analysis']['final_constraint_validation'].
    """

    constraints_satisfied: bool | float = Field(description="Whether all constraints are satisfied (bool or 0.0/1.0)")
    A_sum_error: float = Field(description="Sum constraint error for A matrix")
    A_negative_fraction: float = Field(description="Fraction of negative values in A")
    B_sum_error: float = Field(description="Sum constraint error for B matrix")
    B_negative_fraction: float = Field(description="Fraction of negative values in B")


class WeightAnalysisMatrix(BaseModel):
    """Analysis of a single weight matrix (A or B)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # Fields vary by model implementation, allow extras


class WeightAnalysis(BaseModel):
    """Archetypal weight analysis for A and B matrices.

    Returned in results['final_analysis']['archetypal_weights'].
    """

    model_config = ConfigDict(extra="allow")

    A_matrix: dict[str, Any] = Field(description="A matrix analysis (numpy arrays)")
    B_matrix: dict[str, Any] = Field(description="B matrix analysis (numpy arrays)")


class FinalAnalysis(BaseModel):
    """Final training analysis.

    Returned in results['final_analysis'].
    May contain 'error' key if model doesn't support archetypal analysis.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    final_constraint_validation: ConstraintValidation | None = None
    archetypal_weights: WeightAnalysis | None = None
    final_coordinates: dict[str, Any] | None = Field(
        None, description="Contains 'A', 'B', 'Y' tensors from get_archetypal_coordinates()"
    )
    error: str | None = Field(None, description="Error message if analysis failed")


# =============================================================================
# ARCHETYPAL COORDINATES (from analysis.py)
# =============================================================================


class ArchetypalCoordinates(BaseModel):
    """Return type for get_archetypal_coordinates().

    Contains the core matrices from a single forward pass.

    Attributes
    ----------
        A: Archetypal coordinates [batch_size, n_archetypes]
        B: Dummy B matrix for compatibility [batch_size, n_archetypes]
        Y: Archetype positions [n_archetypes, input_dim]
        mu: Encoder means [batch_size, n_archetypes]
        log_var: Encoder log variances [batch_size, n_archetypes]
        z: Latent variables (reparameterized) [batch_size, n_archetypes]
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    A: Any = Field(description="Archetypal coordinates [batch_size, n_archetypes]")
    B: Any = Field(description="Dummy B matrix [batch_size, n_archetypes]")
    Y: Any = Field(description="Archetype positions [n_archetypes, input_dim]")
    mu: Any = Field(description="Encoder means [batch_size, n_archetypes]")
    log_var: Any = Field(description="Encoder log variances [batch_size, n_archetypes]")
    z: Any = Field(description="Latent variables [batch_size, n_archetypes]")


class ExtractedCoordinates(BaseModel):
    """Return type for extract_and_store_archetypal_coordinates().

    Comprehensive extraction results for full dataset.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    archetype_positions: Any = Field(description="Archetype positions in PCA space [n_archetypes, n_pca_components]")
    cell_weights: Any = Field(description="A matrix weights [n_cells, n_archetypes]")
    cell_latent: Any = Field(description="z latent variables [n_cells, n_archetypes]")
    cell_mu: Any = Field(description="Encoder means [n_cells, n_archetypes]")
    cell_log_var: Any = Field(description="Encoder log variances [n_cells, n_archetypes]")
    n_cells: int
    n_archetypes: int
    pca_key_used: str = Field(description="Key used for PCA coordinates in adata.obsm")


class ArchetypeDistancesResult(BaseModel):
    """Result structure for compute_archetype_distances()."""

    # DataFrame stored in adata.obs (columns documented)
    distance_columns: list[str]  # ['archetype_0_distance', ...]
    nearest_archetype_column: str = "nearest_archetype"
    nearest_distance_column: str = "nearest_archetype_distance"


class CellSelectionResult(BaseModel):
    """Result from select_cells()."""

    selected_mask: Any  # np.ndarray[bool]
    n_selected: int
    selection_criteria: dict[str, str]
    archetype_counts: dict[str, int]


# =============================================================================
# MAIN TRAINING RETURN TYPES
# =============================================================================


class CoreTrainingResults(BaseModel):
    """Return type for _core.utils.training.train_vae().

    NOTE: train_vae() returns Tuple[CoreTrainingResults, torch.nn.Module]
    The model is returned both in the dict AND as second tuple element.

    This is the internal return type. For user-facing API, see TrainingResults.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    history: TrainingHistory | dict[str, list[float]] = Field(description="Training metrics per epoch")
    final_model: Any = Field(description="Trained Deep_AA model (torch.nn.Module)")
    model: Any = Field(description="Alias for final_model (same object)")
    final_optimizer: Any = Field(description="Final optimizer state (torch.optim.Optimizer)")
    final_analysis: FinalAnalysis | dict[str, Any] = Field(description="Final training analysis and metrics")
    epoch_archetype_positions: list[Any] = Field(description="Archetype positions per epoch [List of Tensors]")
    training_config: TrainingConfig | dict[str, Any] = Field(description="Complete training configuration")


class TrainingResults(BaseModel):
    """Return type for tl.train_archetypal().

    This is the USER-FACING return type with convenience keys.

    GUARANTEED KEYS (always present):
        history, final_model, model, final_optimizer, final_analysis,
        epoch_archetype_positions, training_config

    CONVENIENCE KEYS (present if corresponding history metric exists):
        final_archetype_r2, final_rmse, final_mae, final_loss, convergence_epoch

    Example:
        >>> results = pc.tl.train_archetypal(adata, n_archetypes=5)
        >>> # SAFE: Use .get() for convenience keys
        >>> r2 = results.get("final_archetype_r2")
        >>> if r2 is not None:
        ...     print(f"R²: {r2:.4f}")
        >>> # OR: Access from history (always works if metric was tracked)
        >>> if results["history"].get("archetype_r2"):
        ...     r2 = results["history"]["archetype_r2"][-1]

    Validation:
        >>> from peach._core.types import TrainingResults
        >>> validated = TrainingResults.model_validate(results_dict)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # === GUARANTEED KEYS (always present) ===
    history: TrainingHistory | dict[str, Any] = Field(
        description="Training metrics per epoch. See TrainingHistory for keys."
    )
    final_model: Any = Field(description="Trained Deep_AA model (torch.nn.Module)")
    model: Any = Field(description="Alias for final_model (same object, for compatibility)")
    final_optimizer: Any = Field(description="Final optimizer state (torch.optim.Optimizer)")
    final_analysis: FinalAnalysis | dict[str, Any] = Field(
        description="Final analysis including constraint validation and weight analysis"
    )
    epoch_archetype_positions: list[Any] = Field(description="List of archetype position tensors, one per epoch")
    training_config: TrainingConfig | dict[str, Any] = Field(
        description="Training configuration parameters. See TrainingConfig."
    )

    # === CONVENIENCE KEYS (conditional - use .get() or check existence) ===
    final_archetype_r2: float | None = Field(
        None, description="history['archetype_r2'][-1] if exists. USE .get() TO ACCESS."
    )
    final_rmse: float | None = Field(None, description="history['rmse'][-1] if exists. USE .get() TO ACCESS.")
    final_mae: float | None = Field(None, description="history['mae'][-1] if exists. USE .get() TO ACCESS.")
    final_loss: float | None = Field(None, description="history['loss'][-1] if exists. USE .get() TO ACCESS.")
    convergence_epoch: int | None = Field(None, description="training_config['actual_epochs']. USE .get() TO ACCESS.")


# =============================================================================
# DISTANCE AND ASSIGNMENT RESULTS
# =============================================================================


class DistanceInfo(BaseModel):
    """Metadata stored in adata.uns['{prefix}_distance_info']."""

    n_archetypes: int
    pca_key_used: str
    archetype_coords_key: str
    distance_metric: str = "euclidean"
    distance_space: str = "PCA"
    obsm_key: str


class ArchetypeRecoveryMetrics(BaseModel):
    """Return type for test_archetype_recovery().

    Metrics for comparing learned vs true archetypes.
    """

    mean_distance: float
    max_distance: float
    normalized_mean_distance: float
    recovery_success: bool
    individual_distances: Any  # numpy array
    assignment: list[tuple]  # List of (learned_idx, true_idx) pairs


# =============================================================================
# ADATA MODIFICATIONS REFERENCE
# =============================================================================


class AnnDataKeys:
    """Reference for keys stored in AnnData by PEACH functions.

    This is a documentation class, not a runtime type.
    Use this to know where data is stored.

    Usage:
        >>> from peach._core.types import AnnDataKeys
        >>> print(AnnDataKeys.TRAIN_ARCHETYPAL)
    """

    # =========================================================================
    # PREREQUISITES (from scanpy preprocessing)
    # =========================================================================
    # NOTE: PCA key can vary between datasets. Common variants:
    # 'X_pca' (scanpy default), 'X_PCA', 'pca', 'x_pca'
    # Always check which variant exists in your data!
    X_PCA = "X_pca"  # adata.obsm - scanpy default
    # Alternative variants to check: 'X_PCA', 'pca', 'x_pca'

    # =========================================================================
    # PEACH OUTPUT KEYS
    # =========================================================================
    # train_archetypal() stores:
    ARCHETYPE_COORDINATES = "archetype_coordinates"  # adata.uns
    # Shape: (n_archetypes, n_pcs) - archetype positions in PCA space

    # archetypal_coordinates() / compute_archetype_distances() stores:
    ARCHETYPE_DISTANCES = "archetype_distances"  # adata.obsm
    # Shape: (n_cells, n_archetypes) - Euclidean distance to each archetype

    ARCHETYPE_POSITIONS = "archetype_positions"  # adata.uns
    # Shape: (n_archetypes, n_pcs) - copy of archetype positions

    ARCHETYPE_DISTANCE_INFO = "archetype_distance_info"  # adata.uns
    # Dict with computation metadata

    # assign_archetypes() / bin_cells_by_archetype() stores:
    ARCHETYPES = "archetypes"  # adata.obs
    # pd.Categorical with labels: 'archetype_0', 'archetype_1', ..., 'no_archetype'

    # extract_archetype_weights() stores:
    CELL_ARCHETYPE_WEIGHTS = "cell_archetype_weights"  # adata.obsm
    # Shape: (n_cells, n_archetypes) - barycentric coordinates

    # extract_and_store_archetypal_coordinates() stores:
    CELL_ARCHETYPE_WEIGHTS_LATENT = "cell_archetype_weights_latent"  # adata.obsm
    CELL_ARCHETYPE_WEIGHTS_MU = "cell_archetype_weights_mu"  # adata.obsm
    CELL_ARCHETYPE_WEIGHTS_LOG_VAR = "cell_archetype_weights_log_var"  # adata.obsm

    @classmethod
    def describe(cls) -> str:
        """Print all AnnData keys and their locations."""
        return """
AnnData Storage Locations:
==========================

adata.uns (unstructured):
  - 'archetype_coordinates': (n_archetypes, n_pcs) archetype positions
  - 'archetype_positions': copy of archetype positions  
  - 'archetype_distance_info': dict with computation metadata

adata.obsm (cell-level matrices):
  - 'archetype_distances': (n_cells, n_archetypes) distances
  - 'cell_archetype_weights': (n_cells, n_archetypes) barycentric coords
  - 'cell_archetype_weights_latent': (n_cells, n_archetypes) z values
  - 'cell_archetype_weights_mu': (n_cells, n_archetypes) encoder means
  - 'cell_archetype_weights_log_var': (n_cells, n_archetypes) encoder log vars

adata.obs (cell annotations):
  - 'archetypes': Categorical archetype assignments
"""


# =============================================================================
# VALIDATION HELPERS
# =============================================================================


def validate_training_results(results: dict[str, Any]) -> TrainingResults:
    """Validate and convert a results dict to TrainingResults.

    Use this to ensure results have expected structure.
    Raises ValidationError if required keys are missing.

    Example:
        >>> results = pc.tl.train_archetypal(adata, n_archetypes=5)
        >>> validated = validate_training_results(results)
        >>> print(validated.final_archetype_r2)  # With autocomplete!
    """
    return TrainingResults.model_validate(results)


def validate_archetypal_coordinates(coords: dict[str, Any]) -> ArchetypalCoordinates:
    """Validate coordinates dict from get_archetypal_coordinates()."""
    return ArchetypalCoordinates.model_validate(coords)


# =============================================================================
# TYPE ALIASES FOR CONVENIENCE
# =============================================================================

# For type hints in function signatures
HistoryDict = dict[str, list[float]]
ParameterDict = dict[str, Any]  # Renamed from ConfigDict to avoid shadowing pydantic.ConfigDict
CoordinatesDict = dict[str, Any]

# NOTE: These imports are duplicates of lines 26-29 but needed for backward compatibility
# TODO: Clean up by having downstream files import directly from typing/pydantic
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# =============================================================================
# CROSS-VALIDATION TYPES (Phase 2: Hyperparameter Search)
# =============================================================================


class CVFoldMetrics(BaseModel):
    """Metrics from a single CV fold.

    All fields are Optional because different metrics are tracked
    depending on training configuration and early stopping behavior.

    Attributes
    ----------
    train_loss : float | None
        Final training loss.
    train_archetypal_loss : float | None
        Final archetypal reconstruction loss.
    train_archetype_r2 : float | None
        Training set archetype R².
    train_rmse : float | None
        Training set RMSE.
    train_mae : float | None
        Training set MAE.
    val_rmse : float | None
        Validation set RMSE.
    val_mae : float | None
        Validation set MAE.
    val_archetype_r2 : float | None
        Validation set archetype R².
    archetype_r2 : float | None
        Primary metric (typically val_archetype_r2).
    convergence_epoch : int | None
        Epoch when training stopped (early stopping or max epochs).
    early_stopped : bool | None
        Whether early stopping was triggered.

    Examples
    --------
    >>> fold_metrics = CVFoldMetrics(
    ...     train_loss=0.0234, val_archetype_r2=0.891, archetype_r2=0.891, convergence_epoch=35, early_stopped=True
    ... )
    >>> fold_metrics.archetype_r2
    0.891
    """

    model_config = {"extra": "allow"}  # Allow additional metrics

    # Training metrics
    train_loss: float | None = None
    train_archetypal_loss: float | None = None
    train_archetype_r2: float | None = None
    train_rmse: float | None = None
    train_mae: float | None = None

    # Validation metrics
    val_rmse: float | None = None
    val_mae: float | None = None
    val_archetype_r2: float | None = None

    # Primary metric (convenience copy)
    archetype_r2: float | None = None

    # Convergence info
    convergence_epoch: int | None = None
    early_stopped: bool | None = None


class CVHyperparameters(BaseModel):
    """Hyperparameters tested during cross-validation.

    Attributes
    ----------
    n_archetypes : int
        Number of archetypes.
    hidden_dims : list[int]
        Network architecture as list of layer sizes.
    inflation_factor : float
        PCHA inflation factor (default: 1.5).
    use_pcha_init : bool
        Whether PCHA initialization was used.
    use_inflation : bool
        Whether inflation was applied (True if inflation_factor > 1.0).

    Examples
    --------
    >>> hyperparams = CVHyperparameters(
    ...     n_archetypes=5, hidden_dims=[256, 128, 64], inflation_factor=1.5, use_pcha_init=True, use_inflation=True
    ... )
    >>> hyperparams.n_archetypes
    5

    >>> # Validation catches errors
    >>> CVHyperparameters(n_archetypes=0, hidden_dims=[128])
    ValidationError: n_archetypes must be >= 1
    """

    n_archetypes: int = Field(..., ge=1, description="Number of archetypes")
    hidden_dims: list[int] = Field(..., min_length=1, description="Network layer sizes")
    inflation_factor: float = Field(default=1.5, gt=0, description="PCHA inflation factor")
    use_pcha_init: bool = Field(default=True, description="Use PCHA initialization")
    use_inflation: bool = Field(default=True, description="Apply inflation to archetypes")

    @field_validator("hidden_dims")
    @classmethod
    def validate_hidden_dims(cls, v: list[int]) -> list[int]:
        """Ensure all layer sizes are positive."""
        if not all(dim > 0 for dim in v):
            raise ValueError("All hidden_dims must be positive integers")
        return v


class CVMetricSummary(BaseModel):
    """Summary statistics for a metric across CV folds.

    Returned by CVResults.get_metric_summary().

    Attributes
    ----------
    mean : float
        Mean value across folds.
    std : float
        Standard deviation across folds.
    min : float
        Minimum value across folds.
    max : float
        Maximum value across folds.
    """

    mean: float
    std: float
    min: float
    max: float


class CVResultsModel(BaseModel):
    """Validated structure for cross-validation results of a single configuration.

    This Pydantic model mirrors the CVResults dataclass and provides
    runtime validation.

    Attributes
    ----------
    hyperparameters : CVHyperparameters
        The tested hyperparameters.
    fold_results : list[CVFoldMetrics]
        Metrics from each fold.
    mean_metrics : dict[str, float]
        Mean of each metric across folds.
    std_metrics : dict[str, float]
        Standard deviation across folds.
    best_fold_idx : int
        Index of best-performing fold (by archetype_r2).
    convergence_epochs : list[int]
        Convergence epoch for each fold.
    training_time : float
        Total training time in seconds.
    fold_histories : list[dict] | None
        Optional epoch-by-epoch history per fold.

    Examples
    --------
    >>> # Validate CV results
    >>> from peach._core.types import CVResultsModel
    >>> cv_result = manager.train_cv_configuration(hyperparams, cv_splits)
    >>> validated = CVResultsModel.model_validate(cv_result.__dict__)
    >>> print(f"Mean R²: {validated.mean_metrics.get('archetype_r2', 0):.4f}")

    See Also
    --------
    peach._core.utils.grid_search_results.CVResults : Dataclass implementation
    """

    hyperparameters: CVHyperparameters
    fold_results: list[CVFoldMetrics]
    mean_metrics: dict[str, float] = Field(default_factory=dict)
    std_metrics: dict[str, float] = Field(default_factory=dict)
    best_fold_idx: int = Field(..., ge=0)
    convergence_epochs: list[int]
    training_time: float = Field(default=0.0, ge=0)
    fold_histories: list[dict[str, list[float]]] | None = None

    def get_metric_summary(self, metric: str) -> CVMetricSummary:
        """Get summary statistics for a specific metric.

        Parameters
        ----------
        metric : str
            Metric name to summarize.

        Returns
        -------
        CVMetricSummary
            Summary with mean, std, min, max.
        """
        values = [fold.model_dump().get(metric, np.nan) for fold in self.fold_results]
        valid_values = [v for v in values if v is not None and not np.isnan(v)]

        if not valid_values:
            return CVMetricSummary(mean=float("nan"), std=float("nan"), min=float("nan"), max=float("nan"))

        return CVMetricSummary(
            mean=float(np.mean(valid_values)),
            std=float(np.std(valid_values)),
            min=float(np.min(valid_values)),
            max=float(np.max(valid_values)),
        )


class RankedConfig(BaseModel):
    """A single ranked configuration from CVSummary.rank_by_metric().

    Attributes
    ----------
    hyperparameters : CVHyperparameters
        The configuration's hyperparameters.
    metric_value : float
        Value of the ranking metric.
    std_error : float
        Standard error across CV folds.
    config_summary : str
        Human-readable summary string.
        Format: "{n} archetypes, {dims} hidden dims[, λ={factor}]"

    Examples
    --------
    >>> top_configs = cv_summary.rank_by_metric("archetype_r2")
    >>> best = RankedConfig.model_validate(top_configs[0])
    >>> print(f"{best.config_summary}: {best.metric_value:.4f}")
    >>> # Access hyperparameters
    >>> n_arch = best.hyperparameters.n_archetypes
    """

    hyperparameters: CVHyperparameters
    metric_value: float
    std_error: float = Field(default=0.0, ge=0)
    config_summary: str


class DatasetInfo(BaseModel):
    """Dataset metadata for CV search.

    Attributes
    ----------
    n_total_samples : int
        Total number of samples in dataset.
    n_features : int
        Number of input features (PCA dimensions).
    batch_size : int
        Batch size used for training.
    device : str
        Device tensors are on ('cpu', 'cuda', 'mps').
    """

    n_total_samples: int = Field(..., gt=0)
    n_features: int = Field(..., gt=0)
    batch_size: int = Field(..., gt=0)
    device: str = Field(..., pattern=r"^(cpu|cuda|mps)(:\d+)?$")


class CVInfo(BaseModel):
    """Metadata about a cross-validation search.

    Stored in CVSummary.cv_info.

    Attributes
    ----------
    n_configurations : int
        Number of hyperparameter configurations tested.
    cv_folds : int
        Number of CV folds per configuration.
    total_training_runs : int
        Total training runs (n_configurations × cv_folds).
    dataset_info : DatasetInfo
        Information about the dataset.
    total_training_time : float
        Total wall-clock time for search in seconds.

    Examples
    --------
    >>> info = CVInfo.model_validate(cv_summary.cv_info)
    >>> print(f"Tested {info.n_configurations} configs × {info.cv_folds} folds")
    >>> print(f"Total time: {info.total_training_time / 60:.1f} minutes")
    """

    n_configurations: int = Field(..., gt=0)
    cv_folds: int = Field(..., gt=0)
    total_training_runs: int = Field(..., gt=0)
    dataset_info: DatasetInfo
    total_training_time: float = Field(..., ge=0)


class SpeedPreset(str, Enum):
    """Speed presets for hyperparameter search.

    Attributes
    ----------
    FAST : str
        Quick exploration: 25 epochs, patience=3.
    BALANCED : str
        Recommended: 50 epochs, patience=5.
    THOROUGH : str
        Comprehensive: 100 epochs, patience=8.
    """

    FAST = "fast"
    BALANCED = "balanced"
    THOROUGH = "thorough"


class SearchConfigModel(BaseModel):
    """Validated configuration for hyperparameter search.

    Pydantic model version of SearchConfig dataclass.

    Attributes
    ----------
    n_archetypes_range : list[int]
        Range of archetype numbers to test.
    hidden_dims_options : list[list[int]]
        Network architectures to test.
    inflation_factor_range : list[float]
        Inflation factors to test.
    cv_folds : int
        Number of cross-validation folds.
    max_epochs_cv : int
        Maximum epochs per CV fold.
    early_stopping_patience : int
        Patience for early stopping.
    subsample_fraction : float
        Fraction of data to use for CV.
    max_cells_cv : int
        Maximum cells for CV.
    speed_preset : SpeedPreset
        Training speed preset.
    use_pcha_init : bool
        Whether to use PCHA initialization.
    random_state : int
        Random seed for reproducibility.

    Examples
    --------
    >>> config = SearchConfigModel(n_archetypes_range=[3, 4, 5, 6], cv_folds=5, speed_preset=SpeedPreset.BALANCED)
    >>> config.n_archetypes_range
    [3, 4, 5, 6]
    """

    n_archetypes_range: list[int] = Field(
        default=[2, 3, 4, 5, 6, 7], min_length=1, description="Archetype numbers to test"
    )
    hidden_dims_options: list[list[int]] = Field(
        default=[[128, 64], [256, 128, 64], [128], [512, 256, 128]],
        min_length=1,
        description="Network architectures to test",
    )
    inflation_factor_range: list[float] = Field(default=[1.5], min_length=1, description="Inflation factors to test")
    cv_folds: int = Field(default=5, ge=2, le=20)
    max_epochs_cv: int = Field(default=100, ge=1)
    early_stopping_patience: int = Field(default=5, ge=1)
    subsample_fraction: float = Field(default=0.5, gt=0, le=1.0)
    max_cells_cv: int = Field(default=15000, gt=0)
    speed_preset: SpeedPreset = Field(default=SpeedPreset.BALANCED)
    use_pcha_init: bool = Field(default=True)
    random_state: int = Field(default=42, ge=0)

    @field_validator("n_archetypes_range")
    @classmethod
    def validate_n_archetypes(cls, v: list[int]) -> list[int]:
        """Ensure all archetype counts are positive."""
        if not all(n > 0 for n in v):
            raise ValueError("All n_archetypes values must be positive")
        return sorted(set(v))  # Remove duplicates and sort

    @field_validator("inflation_factor_range")
    @classmethod
    def validate_inflation_factors(cls, v: list[float]) -> list[float]:
        """Ensure all inflation factors are positive."""
        if not all(f > 0 for f in v):
            raise ValueError("All inflation factors must be positive")
        return sorted(set(v))

    @property
    def search_inflation(self) -> bool:
        """Whether multiple inflation factors are being searched."""
        return len(self.inflation_factor_range) > 1

    @property
    def n_combinations(self) -> int:
        """Total number of hyperparameter combinations."""
        return len(self.n_archetypes_range) * len(self.hidden_dims_options) * len(self.inflation_factor_range)

    @property
    def total_training_runs(self) -> int:
        """Total number of training runs (combinations × folds)."""
        return self.n_combinations * self.cv_folds


# =============================================================================
# METRIC NAME MAPPING (Authoritative Reference)
# =============================================================================


class CVMetricName(str, Enum):
    """Valid metric names for CV ranking and plotting.

    Use these enum values for type-safe metric references.

    Examples
    --------
    >>> from peach._core.types import CVMetricName
    >>> # Type-safe metric access
    >>> top = cv_summary.rank_by_metric(CVMetricName.ARCHETYPE_R2.value)
    >>> # Or use string directly (validated against this enum)
    >>> top = cv_summary.rank_by_metric("archetype_r2")
    """

    # R² metrics (higher is better)
    R2 = "r2"
    ARCHETYPE_R2 = "archetype_r2"

    # RMSE metrics (lower is better)
    RMSE = "rmse"
    VAL_RMSE = "val_rmse"
    TRAIN_RMSE = "train_rmse"

    # MAE metrics (lower is better)
    MAE = "mae"
    VAL_MAE = "val_mae"
    TRAIN_MAE = "train_mae"

    # Loss metrics (lower is better)
    ARCHETYPAL_LOSS = "archetypal_loss"
    TRAIN_ARCHETYPAL_LOSS = "train_archetypal_loss"

    # Convergence (lower is better)
    CONVERGENCE_EPOCH = "convergence_epoch"


# User-friendly metric names → DataFrame column names
CV_METRIC_MAP: dict[str, str] = {
    # R² metrics
    "r2": "mean_archetype_r2",
    "archetype_r2": "mean_archetype_r2",
    # RMSE metrics
    "rmse": "mean_val_rmse",
    "val_rmse": "mean_val_rmse",
    "train_rmse": "mean_train_rmse",
    # MAE metrics
    "mae": "mean_val_mae",
    "val_mae": "mean_val_mae",
    "train_mae": "mean_train_mae",
    # Loss metrics
    "archetypal_loss": "mean_train_archetypal_loss",
    "train_archetypal_loss": "mean_train_archetypal_loss",
    # Convergence
    "convergence_epoch": "mean_convergence_epoch",
}


# Metrics where lower is better (affects sort order in rankings)
CV_LOWER_IS_BETTER: set[str] = {
    "rmse",
    "val_rmse",
    "train_rmse",
    "mae",
    "val_mae",
    "train_mae",
    "archetypal_loss",
    "train_archetypal_loss",
    "convergence_epoch",
}


# =============================================================================
# ANNDATA STORAGE KEYS (CV-related)
# =============================================================================


class CVSearchKeys:
    """Reference for CV search result storage locations in AnnData.

    CV results are typically stored in a CVSummary object (pickled to disk),
    not in AnnData. However, selected hyperparameters may be stored for
    reference after manual selection.

    Attributes
    ----------
    UNS_SELECTED_CONFIG : str
        Key for selected hyperparameters after Phase 3 manual selection.
        Location: ``adata.uns['peach_selected_config']``
    UNS_CV_SUMMARY_PATH : str
        Key for path to saved CVSummary pickle file.
        Location: ``adata.uns['peach_cv_summary_path']``

    Examples
    --------
    >>> # After manual selection (Phase 3)
    >>> from peach._core.types import CVSearchKeys, CVHyperparameters
    >>> selected = CVHyperparameters(n_archetypes=5, hidden_dims=[256, 128, 64], inflation_factor=1.5)
    >>> adata.uns[CVSearchKeys.UNS_SELECTED_CONFIG] = selected.model_dump()
    >>> adata.uns[CVSearchKeys.UNS_CV_SUMMARY_PATH] = "results/cv_summary.pkl"
    """

    UNS_SELECTED_CONFIG: str = "peach_selected_config"
    UNS_CV_SUMMARY_PATH: str = "peach_cv_summary_path"


from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# =============================================================================
# METRICS MODULE TYPES (_core/utils/metrics.py)
# =============================================================================


class VAEMetrics(BaseModel):
    """Metrics from calculate_vae_metrics().

    Standard VAE performance metrics including reconstruction quality,
    KL divergence, and evidence lower bound.

    Attributes
    ----------
    rmse : float
        Root mean squared error between reconstruction and input.
    kld : float
        KL divergence between encoder distribution and prior.
    elbo : float
        Evidence lower bound (negative, as loss quantity).

    Examples
    --------
    >>> from peach._core.types import VAEMetrics
    >>> metrics = calculate_vae_metrics(recons, input, mu, log_var)
    >>> validated = VAEMetrics.model_validate(metrics)
    >>> print(f"RMSE: {validated.rmse:.4f}, KLD: {validated.kld:.4f}")
    """

    rmse: float = Field(..., ge=0, description="Root mean squared error")
    kld: float = Field(..., description="KL divergence")
    elbo: float = Field(..., description="Evidence lower bound (as loss)")


class EpochMetrics(BaseModel):
    """Metrics from calculate_epoch_metrics().

    Epoch-level metrics extracted from loss dictionary. All fields are
    Optional because different metrics are tracked depending on training
    configuration.

    Attributes
    ----------
    loss : float | None
        Total training loss.
    archetypal_loss : float | None
        Archetypal reconstruction loss component.
    KLD : float | None
        KL divergence component.
    archetype_r2 : float | None
        Archetype reconstruction R².
    rmse : float | None
        Root mean squared error.
    loss_delta : float | None
        Change in loss from previous epoch.

    archetype_drift_mean : float | None
        Mean archetype position drift (if track_stability=True).
    archetype_drift_max : float | None
        Maximum archetype drift.
    archetype_drift_std : float | None
        Std of archetype drift.
    archetype_stability_mean : float | None
        Mean archetype stability (1 - normalized drift).
    archetype_stability_min : float | None
        Minimum archetype stability.
    archetype_variance_mean : float | None
        Mean variance in archetype positions.

    constraint_violation_rate : float | None
        Fraction of constraint violations (if validate_constraints=True).
    constraints_satisfied : bool | None
        Whether all constraints are satisfied.
    A_sum_error : float | None
        Sum constraint error for A matrix.
    A_negative_fraction : float | None
        Fraction of negative values in A.
    B_sum_error : float | None
        Sum constraint error for B matrix.
    B_negative_fraction : float | None
        Fraction of negative values in B.

    KLD_per_dim : float | None
        KLD normalized by latent dimension.
    archetypal_loss_per_dim : float | None
        Archetypal loss normalized by input dimension.
    convergence_rate : float | None
        Rate of loss decrease over recent window.
    loss_stability : float | None
        Coefficient of variation of recent losses.

    Examples
    --------
    >>> from peach._core.types import EpochMetrics
    >>> epoch_metrics = calculate_epoch_metrics(loss_dict)
    >>> validated = EpochMetrics.model_validate(epoch_metrics)
    >>> # Safe access to optional fields
    >>> if validated.archetype_drift_mean is not None:
    ...     print(f"Drift: {validated.archetype_drift_mean:.4f}")
    """

    model_config = {"extra": "allow"}  # Allow additional metrics

    # Core loss metrics
    loss: float | None = None
    archetypal_loss: float | None = None
    KLD: float | None = None
    archetype_r2: float | None = None
    rmse: float | None = None
    loss_delta: float | None = None

    # Stability metrics (require track_stability=True)
    archetype_drift_mean: float | None = None
    archetype_drift_max: float | None = None
    archetype_drift_std: float | None = None
    archetype_stability_mean: float | None = None
    archetype_stability_min: float | None = None
    archetype_variance_mean: float | None = None

    # Constraint metrics (require validate_constraints=True)
    constraint_violation_rate: float | None = Field(default=None, ge=0, le=1)
    constraints_satisfied: bool | None = None
    A_sum_error: float | None = Field(default=None, ge=0)
    A_negative_fraction: float | None = Field(default=None, ge=0, le=1)
    B_sum_error: float | None = Field(default=None, ge=0)
    B_negative_fraction: float | None = Field(default=None, ge=0, le=1)

    # Normalized metrics
    KLD_per_dim: float | None = None
    archetypal_loss_per_dim: float | None = None

    # Convergence metrics
    convergence_rate: float | None = None
    loss_stability: float | None = Field(default=None, ge=0)


class MetricSummary(BaseModel):
    """Summary statistics for a single metric.

    Returned as values in MetricsTracker.get_metric_summaries().

    Attributes
    ----------
    final : float
        Final (last epoch) value.
    min : float
        Minimum value across all epochs.
    max : float
        Maximum value across all epochs.
    range : float
        Total range (max - min).
    pct_improvement : float
        Percent improvement from worst to best value.
        For loss metrics: (max - final) / (max - min) * 100
        For improvement metrics: (final - min) / (max - min) * 100

    Examples
    --------
    >>> summaries = tracker.get_metric_summaries()
    >>> loss_summary = MetricSummary.model_validate(summaries["loss"])
    >>> print(f"Loss improved {loss_summary.pct_improvement:.1f}%")
    """

    final: float
    min: float
    max: float
    range: float = Field(..., ge=0)
    pct_improvement: float = Field(..., ge=0, le=100)


class MetricsHistory(BaseModel):
    """Complete training metrics history from MetricsTracker.

    Structure returned by MetricsTracker.get_history().
    All fields are lists of per-epoch values.

    Attributes
    ----------
    loss : list[float]
        Total loss per epoch.
    archetypal_loss : list[float]
        Archetypal loss per epoch.
    KLD : list[float]
        KL divergence per epoch.
    archetype_r2 : list[float]
        Archetype R² per epoch.
    rmse : list[float]
        RMSE per epoch.

    Additional stability and constraint metrics may be present
    depending on training configuration.

    Examples
    --------
    >>> history = tracker.get_history()
    >>> validated = MetricsHistory.model_validate(history)
    >>> # Plot loss curve
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(validated.loss)
    """

    model_config = {"extra": "allow"}  # Allow additional metrics

    # Core metrics (typically always present)
    loss: list[float] = Field(default_factory=list)
    archetypal_loss: list[float] = Field(default_factory=list)
    KLD: list[float] = Field(default_factory=list)
    archetype_r2: list[float] = Field(default_factory=list)
    rmse: list[float] = Field(default_factory=list)

    # Stability metrics (if track_stability=True)
    vertex_stability_latent: list[float] = Field(default_factory=list)
    mean_vertex_stability_latent: list[float] = Field(default_factory=list)
    max_vertex_stability_latent: list[float] = Field(default_factory=list)
    vertex_stability_pca: list[float] = Field(default_factory=list)
    mean_vertex_stability_pca: list[float] = Field(default_factory=list)
    max_vertex_stability_pca: list[float] = Field(default_factory=list)


class VertexMetrics(BaseModel):
    """Metrics from calculate_vertex_metrics().

    Archetype vertex position quality metrics.

    Attributes
    ----------
    vertex_range : list[float] | None
        [min, max] range of vertex values, or None if no vertices.
    vertex_mean : float | None
        Mean of vertex positions.
    vertex_std : float | None
        Standard deviation of vertex positions.
    reconstruction_error : float
        Reconstruction error passed to the function.
    weight_sparsity : float | None
        Sparsity of weight matrices (if provided).

    Examples
    --------
    >>> metrics = calculate_vertex_metrics(vertices, recon_error, sparsity)
    >>> validated = VertexMetrics.model_validate(metrics)
    >>> if validated.vertex_range is not None:
    ...     print(f"Vertex range: {validated.vertex_range}")
    """

    vertex_range: list[float] | None = Field(default=None, min_length=2, max_length=2)
    vertex_mean: float | None = None
    vertex_std: float | None = Field(default=None, ge=0)
    reconstruction_error: float
    weight_sparsity: float | None = Field(default=None, ge=0, le=1)


class MatrixConstraintStats(BaseModel):
    """Constraint statistics for a single matrix (A or B).

    Part of ConstraintDiagnostics returned by diagnose_constraint_violations().

    Attributes
    ----------
    sum_error_max : float
        Maximum sum constraint error.
    sum_error_mean : float
        Mean sum constraint error.
    sum_error_std : float
        Std of sum constraint errors.
    sum_error_median : float
        Median sum constraint error.
    negative_fraction : float
        Fraction of negative values (should be 0 for valid matrices).
    negative_count : int
        Total count of negative values.
    violating_samples : int
        Number of samples/archetypes violating constraints.
    violation_rate : float
        Fraction of samples/archetypes with violations.

    Examples
    --------
    >>> diagnostics = diagnose_constraint_violations(A, B)
    >>> a_stats = MatrixConstraintStats.model_validate(diagnostics["A_matrix"])
    >>> print(f"A violation rate: {a_stats.violation_rate:.2%}")
    """

    sum_error_max: float = Field(..., ge=0)
    sum_error_mean: float = Field(..., ge=0)
    sum_error_std: float = Field(..., ge=0)
    sum_error_median: float = Field(..., ge=0)
    negative_fraction: float = Field(..., ge=0, le=1)
    negative_count: int = Field(..., ge=0)
    violating_samples: int = Field(..., ge=0)  # For A: samples, for B: archetypes
    violation_rate: float = Field(..., ge=0, le=1)


class ConstraintSummary(BaseModel):
    """Summary of constraint satisfaction.

    Part of ConstraintDiagnostics returned by diagnose_constraint_violations().

    Attributes
    ----------
    constraints_satisfied_max : bool
        Whether constraints are satisfied using max error threshold.
    constraints_satisfied_mean : bool
        Whether constraints are satisfied using mean error threshold.
    max_vs_mean_discrepancy : dict
        Discrepancy between max and mean errors for debugging.
        Contains 'A_discrepancy' and 'B_discrepancy' floats.
    """

    constraints_satisfied_max: bool
    constraints_satisfied_mean: bool
    max_vs_mean_discrepancy: dict[str, float]


class ConstraintDiagnostics(BaseModel):
    """Complete constraint diagnostics from diagnose_constraint_violations().

    Comprehensive analysis of archetypal constraint satisfaction for
    A (reconstruction weights) and B (construction weights) matrices.

    Attributes
    ----------
    A_matrix : MatrixConstraintStats
        Statistics for A matrix (cell-to-archetype weights).
        A rows should sum to 1, values should be non-negative.
    B_matrix : MatrixConstraintStats
        Statistics for B matrix (archetype-to-cell weights).
        B columns should sum to 1, values should be non-negative.
    summary : ConstraintSummary
        Overall constraint satisfaction summary.

    Examples
    --------
    >>> from peach._core.types import ConstraintDiagnostics
    >>> diagnostics = diagnose_constraint_violations(A, B, tolerance=1e-3)
    >>> validated = ConstraintDiagnostics.model_validate(diagnostics)
    >>> if not validated.summary.constraints_satisfied_max:
    ...     print(f"A max error: {validated.A_matrix.sum_error_max:.4f}")
    ...     print(f"B max error: {validated.B_matrix.sum_error_max:.4f}")
    >>> # Check for numerical issues
    >>> if validated.A_matrix.negative_count > 0:
    ...     print(f"Warning: {validated.A_matrix.negative_count} negative values in A")

    See Also
    --------
    peach._core.utils.metrics.diagnose_constraint_violations : Function that returns this
    """

    A_matrix: MatrixConstraintStats
    B_matrix: MatrixConstraintStats
    summary: ConstraintSummary


class ArchetypeR2Result(BaseModel):
    """Result from calculate_archetype_r2().

    Single scalar R² value with metadata.

    Attributes
    ----------
    value : float
        The R² value. Range is (-inf, 1.0].
        - 1.0 = perfect reconstruction
        - 0.0 = reconstruction equals mean (no better than baseline)
        - < 0 = reconstruction worse than mean

    Notes
    -----
    R² = 1 - (SS_res / SS_tot) where:
    - SS_res = ||X - X_reconstructed||²_F
    - SS_tot = ||X - X_mean||²_F

    Examples
    --------
    >>> r2 = calculate_archetype_r2(reconstructions, original)
    >>> print(f"Archetype R²: {r2.item():.4f}")
    """

    value: float = Field(..., le=1.0, description="R² value (≤1.0, can be negative)")


# =============================================================================
# METRIC CLASSIFICATION (for MetricsTracker)
# =============================================================================


class MetricType(str, Enum):
    """Classification of metrics by optimization direction.

    Used by MetricsTracker to determine improvement calculation.
    """

    LOSS = "loss"  # Lower is better
    IMPROVEMENT = "improvement"  # Higher is better
    NEUTRAL = "neutral"  # No direction preference


# Metrics where lower values are better
LOSS_METRICS: set[str] = {
    "loss",
    "archetypal_loss",
    "KLD",
    "rmse",
    "loss_delta",
    "A_sum_error",
    "B_sum_error",
    "constraint_violation_rate",
    "archetype_drift_mean",
    "archetype_drift_max",
    "archetype_variance_mean",
}

# Metrics where higher values are better
IMPROVEMENT_METRICS: set[str] = {
    "archetype_r2",
    "archetype_stability_mean",
    "archetype_stability_min",
    "vertex_stability_latent",
    "mean_vertex_stability_latent",
    "max_vertex_stability_latent",
    "vertex_stability_pca",
    "mean_vertex_stability_pca",
    "max_vertex_stability_pca",
    "constraints_satisfied",
}


def get_metric_type(metric_name: str) -> MetricType:
    """Determine whether a metric should be minimized or maximized.

    Parameters
    ----------
    metric_name : str
        Name of the metric.

    Returns
    -------
    MetricType
        LOSS if lower is better, IMPROVEMENT if higher is better,
        NEUTRAL if unknown.

    Examples
    --------
    >>> from peach._core.types import get_metric_type, MetricType
    >>> get_metric_type("loss")
    MetricType.LOSS
    >>> get_metric_type("archetype_r2")
    MetricType.IMPROVEMENT
    """
    if metric_name in LOSS_METRICS:
        return MetricType.LOSS
    elif metric_name in IMPROVEMENT_METRICS:
        return MetricType.IMPROVEMENT
    else:
        return MetricType.NEUTRAL


from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# =============================================================================
# MODEL TYPES (_core/models/)
# =============================================================================


# -----------------------------------------------------------------------------
# VAE_Base Types
# -----------------------------------------------------------------------------


class VAEForwardOutput(BaseModel):
    """Output structure from VAE_Base.forward().

    Standard VAE forward pass returns reconstruction and latent parameters.

    Attributes
    ----------
    recons : Any
        Reconstructed data tensor [batch_size, input_dim].
    input : Any
        Original input tensor (passed through for loss computation).
    mu : Any
        Latent space mean [batch_size, latent_dim].
    log_var : Any
        Latent space log variance [batch_size, latent_dim].

    Notes
    -----
    VAE_Base.forward() returns a List[Tensor], not a dict.
    This model documents the list order: [recons, input, mu, log_var].

    Examples
    --------
    >>> outputs = vae_model(input_data)
    >>> recons, input, mu, log_var = outputs  # Unpack list
    """

    recons: Any
    input: Any
    mu: Any
    log_var: Any


class VAELossOutput(BaseModel):
    """Output structure from VAE_Base.loss_function().

    Attributes
    ----------
    loss : Any
        Total VAE loss (reconstruction + KLD).
    reconstruction_loss : Any
        MSE reconstruction loss (detached).
    KLD : Any
        KL divergence loss (detached).

    Examples
    --------
    >>> loss_dict = vae_model.loss_function(*outputs, kld_weight=1.0)
    >>> total_loss = loss_dict["loss"]
    >>> total_loss.backward()
    """

    loss: Any  # torch.Tensor (requires grad)
    reconstruction_loss: Any  # torch.Tensor (detached)
    KLD: Any  # torch.Tensor (detached)


class VAEConfig(BaseModel):
    """Configuration for VAE_Base model.

    Attributes
    ----------
    input_dim : int
        Input feature dimension.
    latent_dim : int
        Latent space dimension.
    hidden_dims : list[int] | None
        Hidden layer sizes for encoder/decoder.
        Default: [128, 64, 32].
    n_archetypes : int | None
        Number of archetypes (not used in base VAE).
    archetypal_weight : float
        Weight for archetypal loss (not used in base VAE).

    Examples
    --------
    >>> config = VAEConfig(input_dim=30, latent_dim=10, hidden_dims=[256, 128, 64])
    """

    input_dim: int = Field(..., gt=0)
    latent_dim: int = Field(..., gt=0)
    hidden_dims: list[int] | None = Field(default=None)
    n_archetypes: int | None = Field(default=None, gt=0)
    archetypal_weight: float = Field(default=1.0, ge=0)

    @field_validator("hidden_dims")
    @classmethod
    def validate_hidden_dims(cls, v: list[int] | None) -> list[int] | None:
        """Ensure all hidden dimensions are positive."""
        if v is not None and not all(d > 0 for d in v):
            raise ValueError("All hidden_dims must be positive")
        return v


# -----------------------------------------------------------------------------
# Deep_AA Types
# -----------------------------------------------------------------------------


class DeepAAForwardOutput(BaseModel):
    """Output structure from Deep_AA.forward().

    Deep_AA returns a dictionary with archetypal analysis outputs.

    Attributes
    ----------
    arch_recons : Any
        Archetypal reconstruction [batch_size, input_dim].
        Computed as A @ Y (coordinates @ archetypes).
    mu : Any
        Encoder output mean [batch_size, n_archetypes].
    log_var : Any
        Encoder output log variance [batch_size, n_archetypes].
    z : Any
        Archetypal coordinates after reparameterization [batch_size, n_archetypes].
        This IS the A matrix (cell-to-archetype weights).
    archetypes : Any
        Learned archetype positions [n_archetypes, input_dim].
        This is the Y matrix.
    input : Any
        Original input tensor (passed through for loss computation).

    recons : Any
        Alias for arch_recons (legacy compatibility).
    archetypal_coordinates : Any
        Alias for z (legacy compatibility).
    A : Any
        Alias for z - the cell-to-archetype weight matrix.
    Y : Any
        Alias for archetypes - the archetype position matrix.

    Notes
    -----
    In Deep_AA, z represents archetypal coordinates (the A matrix),
    NOT a traditional VAE latent space. Each row of z sums to 1 and
    represents how much each archetype contributes to reconstructing
    that sample.

    Examples
    --------
    >>> outputs = model(input_data)
    >>> # Primary outputs
    >>> reconstruction = outputs["arch_recons"]
    >>> coordinates = outputs["z"]  # Same as outputs['A']
    >>> archetypes = outputs["archetypes"]  # Same as outputs['Y']
    >>> # Verify archetypal constraint
    >>> assert torch.allclose(outputs["z"].sum(dim=1), torch.ones(batch_size))
    """

    # Primary outputs
    arch_recons: Any
    mu: Any
    log_var: Any
    z: Any
    archetypes: Any
    input: Any

    # Legacy compatibility aliases
    recons: Any
    archetypal_coordinates: Any
    A: Any
    Y: Any

    model_config = {"extra": "allow"}


class DeepAALossOutput(BaseModel):
    """Output structure from Deep_AA.loss_function().

    Comprehensive loss output with all components and metrics.

    Attributes
    ----------
    loss : Any
        Total weighted loss (requires grad for backprop).

    kld_loss : Any
        KL divergence loss component (detached).
    archetypal_loss : Any
        Archetypal reconstruction loss (detached).
    diversity_loss : Any
        Archetype diversity/separation loss (detached).
    regularity_loss : Any
        Archetype usage regularity loss (detached).
    sparsity_loss : Any
        Coordinate sparsity loss (detached).
    manifold_loss : Any
        Manifold regularization loss (detached).

    rmse : Any
        Root mean squared error (detached).
    archetype_r2 : Any
        Archetype reconstruction R² (detached).

    archetype_entropy : Any
        Entropy of archetype usage distribution (detached).
    max_archetype_usage : Any
        Maximum mean usage across archetypes (detached).
    min_archetype_usage : Any
        Minimum mean usage across archetypes (detached).
    active_archetypes_per_sample : Any
        Mean number of archetypes with weight > 0.01 per sample.

    mean_archetype_data_distance : float
        Mean distance from archetypes to nearest data point.
    max_archetype_data_distance : float
        Maximum distance from any archetype to nearest data point.

    loss_delta : Any
        Change in loss from previous call (for convergence tracking).
    loss_history : list[float]
        Recent loss history (last 100 values).

    input_dim : int
        Model input dimension.
    latent_dim : int
        Model latent dimension (= n_archetypes).
    n_archetypes : int
        Number of archetypes.

    KLD : Any
        Alias for kld_loss (legacy compatibility).
    reconstruction_loss : Any
        Alias for archetypal_loss (legacy compatibility).

    Examples
    --------
    >>> loss_dict = model.loss_function(outputs)
    >>> # Backpropagation
    >>> loss_dict["loss"].backward()
    >>> # Monitoring
    >>> print(f"R²: {loss_dict['archetype_r2'].item():.4f}")
    >>> print(f"RMSE: {loss_dict['rmse'].item():.4f}")
    >>> # Check archetype health
    >>> if loss_dict["min_archetype_usage"] < 0.01:
    ...     print("Warning: Some archetypes underutilized")
    """

    # Primary loss (requires grad)
    loss: Any

    # Loss components (all detached)
    kld_loss: Any
    archetypal_loss: Any
    diversity_loss: Any
    regularity_loss: Any
    sparsity_loss: Any
    manifold_loss: Any

    # Performance metrics (detached)
    rmse: Any
    archetype_r2: Any

    # Archetype usage metrics (detached)
    archetype_entropy: Any
    max_archetype_usage: Any
    min_archetype_usage: Any
    active_archetypes_per_sample: Any

    # Manifold quality metrics
    mean_archetype_data_distance: float
    max_archetype_data_distance: float

    # Convergence tracking
    loss_delta: Any
    loss_history: list[float]

    # Model info
    input_dim: int
    latent_dim: int
    n_archetypes: int

    # Legacy compatibility
    KLD: Any
    reconstruction_loss: Any

    model_config = {"extra": "allow"}


class DeepAAConfig(BaseModel):
    """Configuration for Deep_AA model.

    Attributes
    ----------
    input_dim : int
        Input feature dimension (e.g., number of PCA components).
    n_archetypes : int
        Number of archetypes to learn. Must be >= 2.
    latent_dim : int | None
        Latent dimension. If None or different from n_archetypes,
        will be set to n_archetypes (archetypal constraint).
    hidden_dims : list[int] | None
        Encoder/decoder hidden layer sizes.
        Default: [128, 64, 32].

    archetypal_weight : float
        Weight for archetypal reconstruction loss. Default: 1.0.
        **Recommended: Keep at 1.0** (proven optimal).
    kld_weight : float
        Weight for KL divergence loss. Default: 0.0.
        **Warning: Non-zero values hurt performance (-3.0%)**.
    diversity_weight : float
        Weight for archetype diversity loss. Default: 0.0.
        **Warning: Non-zero values hurt performance (-2.4%)**.
    regularity_weight : float
        Weight for archetype usage regularity loss. Default: 0.0.
        **Warning: Non-zero values hurt performance (-3.7%)**.
    sparsity_weight : float
        Weight for coordinate sparsity loss. Default: 0.0.
        **Warning: Not tested, likely harmful**.
    manifold_weight : float
        Weight for manifold regularization loss. Default: 0.0.
        **Warning: Non-zero values hurt performance (-3.2%)**.

    inflation_factor : float
        Scalar inflation factor for PCHA initialization. Default: 1.5.
        This is the "Helsinki breakthrough" parameter.
    use_barycentric : bool
        Use softmax for strict barycentric coordinates. Default: True.
    use_hidden_transform : bool
        Apply learned transformation to archetypes. Default: True.

    device : str
        Device for model parameters ('cpu', 'cuda', 'mps').

    Notes
    -----
    The default loss weights (all auxiliary losses = 0.0) are the result
    of extensive ablation studies. The minimalist configuration with only
    archetypal_weight=1.0 consistently outperforms configurations with
    additional loss terms.

    Examples
    --------
    >>> # Recommended configuration
    >>> config = DeepAAConfig(input_dim=30, n_archetypes=5, hidden_dims=[256, 128, 64], inflation_factor=1.5)
    >>> # Create model
    >>> model = Deep_AA(**config.model_dump())

    See Also
    --------
    peach.tl.train_archetypal : User-facing training function
    """

    input_dim: int = Field(..., gt=0, description="Input feature dimension")
    n_archetypes: int = Field(..., ge=2, description="Number of archetypes")
    latent_dim: int | None = Field(default=None, description="Latent dim (defaults to n_archetypes)")
    hidden_dims: list[int] | None = Field(default=None, description="Hidden layer sizes")

    # Loss weights
    archetypal_weight: float = Field(default=0.9, ge=0)
    kld_weight: float = Field(default=0.1, ge=0)
    diversity_weight: float = Field(default=0.0, ge=0)
    regularity_weight: float = Field(default=0.0, ge=0)
    sparsity_weight: float = Field(default=0.0, ge=0)
    manifold_weight: float = Field(default=0.0, ge=0)

    # Inflation and behavior
    inflation_factor: float = Field(default=1.5, gt=0)
    use_barycentric: bool = Field(default=True)
    use_hidden_transform: bool = Field(default=True)

    # Device
    device: str = Field(default="cpu", pattern=r"^(cpu|cuda|mps)(:\d+)?$")

    @field_validator("hidden_dims")
    @classmethod
    def validate_hidden_dims(cls, v: list[int] | None) -> list[int] | None:
        if v is not None and not all(d > 0 for d in v):
            raise ValueError("All hidden_dims must be positive")
        return v

    @model_validator(mode="after")
    def set_latent_dim(self) -> DeepAAConfig:
        """Ensure latent_dim equals n_archetypes."""
        if self.latent_dim is None or self.latent_dim != self.n_archetypes:
            object.__setattr__(self, "latent_dim", self.n_archetypes)
        return self


class ArchetypeInitConfig(BaseModel):
    """Configuration for Deep_AA.initialize_archetypes().

    Attributes
    ----------
    use_pcha : bool
        Use PCHA initialization (vs furthest-sum fallback). Default: True.
    use_inflation : bool
        Apply scalar inflation after initialization. Default: False.
    inflation_factor : float | None
        Inflation factor. If None, uses model's inflation_factor.
    n_subsample : int
        Max samples for PCHA efficiency. Default: 1000.
    test_inflation_factors : bool
        Test multiple inflation factors and select best. Default: False.
    inflation_test_range : list[float] | None
        Factors to test. Default: [1.0, 1.2, 1.5, 2.0, 3.0].

    Examples
    --------
    >>> # Standard initialization
    >>> model.initialize_archetypes(X_sample, use_pcha=True, use_inflation=True)
    >>> # Test multiple factors
    >>> model.initialize_archetypes(X_sample, test_inflation_factors=True, inflation_test_range=[1.0, 1.5, 2.0, 2.5])
    """

    use_pcha: bool = Field(default=True)
    use_inflation: bool = Field(default=False)
    inflation_factor: float | None = Field(default=None, gt=0)
    n_subsample: int = Field(default=1000, gt=0)
    test_inflation_factors: bool = Field(default=False)
    inflation_test_range: list[float] | None = Field(default=None)

    @field_validator("inflation_test_range")
    @classmethod
    def validate_test_range(cls, v: list[float] | None) -> list[float] | None:
        if v is not None and not all(f > 0 for f in v):
            raise ValueError("All inflation factors must be positive")
        return v


class InflationTestResult(BaseModel):
    """Result for a single inflation factor test.

    Returned as values in test_inflation_factors() results dict.

    Attributes
    ----------
    archetype_r2 : float
        Initial reconstruction R² with this factor.
    arch_loss : float
        Archetypal reconstruction loss.
    outside_data_count : int
        Number of archetypes positioned outside data radius.
    mean_arch_distance : float
        Mean distance of archetypes from data centroid.
    min_arch_distance : float
        Minimum archetype distance from centroid.
    max_arch_distance : float
        Maximum archetype distance from centroid.
    error : str | None
        Error message if test failed, None otherwise.
    """

    archetype_r2: float | None = None
    arch_loss: float | None = None
    outside_data_count: int | None = None
    mean_arch_distance: float | None = None
    min_arch_distance: float | None = None
    max_arch_distance: float | None = None
    error: str | None = None


class ArchetypalWeightAnalysis(BaseModel):
    """Result from Deep_AA.analyze_archetypal_weights().

    Attributes
    ----------
    A_matrix : MatrixWeightStats
        Statistics for A (cell-to-archetype) weights.
    B_matrix : MatrixWeightStats
        Statistics for B (archetype-to-cell) weights.
        Note: In Deep_AA, B is a dummy matrix.
    """

    A_matrix: MatrixWeightStats
    B_matrix: MatrixWeightStats


class MatrixWeightStats(BaseModel):
    """Weight statistics for a single matrix (A or B).

    Attributes
    ----------
    mean_weights : Any
        Mean weight per archetype [n_archetypes].
    std_weights : Any
        Std of weights per archetype [n_archetypes].
    max_weights : Any
        Max weight per archetype [n_archetypes].
    min_weights : Any
        Min weight per archetype [n_archetypes].
    dominant_archetype : Any
        Fraction of samples where each archetype is dominant [n_archetypes].
        Only present for A_matrix.
    """

    mean_weights: Any
    std_weights: Any
    max_weights: Any
    min_weights: Any
    dominant_archetype: Any | None = None


class ConstraintValidation(BaseModel):
    """Result from Deep_AA.validate_constraints().

    Attributes
    ----------
    A_sum_error : float
        Mean deviation of A row sums from 1.0.
    A_negative_fraction : float
        Fraction of negative values in A.
    B_sum_error : float
        Mean deviation of B column sums from 1.0.
    B_negative_fraction : float
        Fraction of negative values in B.
    constraints_satisfied : float
        1.0 if all constraints satisfied, 0.0 otherwise.

    Examples
    --------
    >>> validation = model.validate_constraints(A, B, tolerance=1e-3)
    >>> validated = ConstraintValidation.model_validate(validation)
    >>> if validated.constraints_satisfied < 1.0:
    ...     print(f"A sum error: {validated.A_sum_error:.4f}")
    """

    A_sum_error: float = Field(..., ge=0)
    A_negative_fraction: float = Field(..., ge=0, le=1)
    B_sum_error: float = Field(..., ge=0)
    B_negative_fraction: float = Field(..., ge=0, le=1)
    constraints_satisfied: float = Field(..., ge=0, le=1)


# -----------------------------------------------------------------------------
# PCHA Types (used by Deep_AA initialization)
# -----------------------------------------------------------------------------


class PCHAResults(BaseModel):
    """Results from PCHA analysis.

    Returned by run_pcha_analysis() and stored in model.pcha_results.

    Attributes
    ----------
    archetypes : Any
        Archetype positions [n_archetypes, n_features].
    archetype_r2 : float
        Explained variance ratio (R²).
    A : Any
        Cell-to-archetype weights [n_cells, n_archetypes].
    B : Any
        Archetype-to-cell weights [n_archetypes, n_cells].

    Examples
    --------
    >>> from peach._core.utils.PCHA import run_pcha_analysis
    >>> results = run_pcha_analysis(data, n_archetypes=5)
    >>> validated = PCHAResults.model_validate(results)
    >>> print(f"PCHA R²: {validated.archetype_r2:.4f}")
    """

    archetypes: Any
    archetype_r2: float = Field(..., le=1.0)
    A: Any
    B: Any

    model_config = {"extra": "allow"}


from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# =============================================================================
# STATISTICAL TESTING TYPES (_core/utils/statistical_tests.py)
# =============================================================================


# -----------------------------------------------------------------------------
# Enums and Constants
# -----------------------------------------------------------------------------


class FDRMethod(str, Enum):
    """FDR correction methods."""

    BENJAMINI_HOCHBERG = "benjamini_hochberg"
    BONFERRONI = "bonferroni"
    # HOLM = "holm"  # Commented out in implementation


class FDRScope(str, Enum):
    """Scope for FDR correction."""

    GLOBAL = "global"
    PER_ARCHETYPE = "per_archetype"
    NONE = "none"


class TestDirection(str, Enum):
    """Statistical test direction."""

    TWO_SIDED = "two-sided"
    GREATER = "greater"
    LESS = "less"


class ComparisonGroup(str, Enum):
    """Comparison group for statistical tests."""

    ALL = "all"
    ARCHETYPES_ONLY = "archetypes_only"


class PatternType(str, Enum):
    """Types of archetypal patterns."""

    EXCLUSIVE = "exclusive"
    EXCLUSIVE_PAIRWISE = "exclusive_pairwise"
    SPECIALIZATION = "specialization"
    TRADEOFF = "tradeoff"
    TRADEOFF_PAIR = "tradeoff_pair"
    TRADEOFF_PATTERN = "tradeoff_pattern"
    CUSTOM = "custom"


class TradeoffMode(str, Enum):
    """Mode for tradeoff pattern analysis."""

    PAIRS = "pairs"
    PATTERNS = "patterns"


# -----------------------------------------------------------------------------
# Gene Association Types
# -----------------------------------------------------------------------------


class GeneAssociationResult(BaseModel):
    """Single gene-archetype association test result.

    Returned as rows in test_archetype_gene_associations() DataFrame.

    Attributes
    ----------
    gene : str
        Gene symbol/identifier.
    archetype : str
        Archetype identifier (e.g., 'archetype_1').
    n_archetype_cells : int
        Number of cells in the archetype bin.
    n_other_cells : int
        Number of cells in the comparison group.
    mean_archetype : float
        Mean expression in archetype cells.
    mean_other : float
        Mean expression in other cells.
    log_fold_change : float
        Log fold change (archetype vs others).
        For log-transformed data: simple difference.
        For raw counts: log2((mean_arch + 1) / (mean_other + 1)).
    statistic : float
        Mann-Whitney U test statistic.
    pvalue : float
        Raw p-value from statistical test.
    test_direction : str
        Direction of test performed ('two-sided', 'greater', 'less').
    direction : str
        Effect direction ('higher' or 'lower' in archetype).
    passes_lfc_threshold : bool
        Whether the result passes the log fold change threshold.
    fdr_pvalue : float | None
        FDR-corrected p-value (after apply_fdr_correction).
    significant : bool | None
        Whether statistically significant (FDR < alpha).

    Examples
    --------
    >>> results = test_archetype_gene_associations(adata)
    >>> for _, row in results.iterrows():
    ...     result = GeneAssociationResult.model_validate(row.to_dict())
    ...     if result.significant and result.direction == "higher":
    ...         print(f"{result.gene}: LFC={result.log_fold_change:.2f}")
    """

    gene: str
    archetype: str
    n_archetype_cells: int = Field(..., ge=0)
    n_other_cells: int = Field(..., ge=0)
    mean_archetype: float
    mean_other: float
    log_fold_change: float
    statistic: float
    pvalue: float = Field(..., ge=0, le=1)
    test_direction: str = Field(default="two-sided")
    direction: str = Field(..., pattern=r"^(higher|lower)$")
    passes_lfc_threshold: bool = Field(default=True)
    fdr_pvalue: float | None = Field(default=None, ge=0, le=1)
    significant: bool | None = Field(default=None)

    model_config = {"extra": "allow"}


class GeneAssociationConfig(BaseModel):
    """Configuration for test_archetype_gene_associations().

    Attributes
    ----------
    bin_prop : float
        Proportion of cells closest to each archetype to use.
        Default: 0.1 (10% of cells).
    obsm_key : str
        Key for distance matrix in adata.obsm.
        Default: 'archetype_distances'.
    obs_key : str
        Key for archetype assignments in adata.obs.
        Default: 'archetypes'.
    use_layer : str | None
        AnnData layer to use. None uses adata.X.
        Auto-selects 'logcounts' or 'log1p' if available.
    test_method : str
        Statistical test method. Default: 'mannwhitneyu'.
    fdr_method : str
        FDR correction method. Default: 'benjamini_hochberg'.
    fdr_scope : str
        FDR scope: 'global', 'per_archetype', or 'none'.
    test_direction : str
        Test direction: 'two-sided', 'greater', or 'less'.
    min_logfc : float
        Minimum log fold change threshold. Default: 0.01.
    min_cells : int
        Minimum cells per archetype. Default: 10.
    comparison_group : str
        'all' or 'archetypes_only'. Default: 'all'.

    Examples
    --------
    >>> config = GeneAssociationConfig(bin_prop=0.15, fdr_scope="per_archetype", min_cells=20)
    """

    bin_prop: float = Field(default=0.1, gt=0, le=1)
    obsm_key: str = Field(default="archetype_distances")
    obs_key: str = Field(default="archetypes")
    use_layer: str | None = Field(default=None)
    test_method: str = Field(default="mannwhitneyu")
    fdr_method: str = Field(default="benjamini_hochberg")
    fdr_scope: str = Field(default="global", pattern=r"^(global|per_archetype|none)$")
    test_direction: str = Field(default="two-sided")
    min_logfc: float = Field(default=0.01, ge=0)
    min_cells: int = Field(default=10, ge=1)
    comparison_group: str = Field(default="all", pattern=r"^(all|archetypes_only)$")


# -----------------------------------------------------------------------------
# Pathway Association Types
# -----------------------------------------------------------------------------


class PathwayAssociationResult(BaseModel):
    """Single pathway-archetype association test result.

    Returned as rows in test_archetype_pathway_associations() DataFrame.

    Attributes
    ----------
    pathway : str
        Pathway name/identifier.
    archetype : str
        Archetype identifier.
    n_archetype_cells : int
        Number of cells in archetype.
    n_other_cells : int
        Number of cells in comparison group.
    mean_archetype : float
        Mean pathway score in archetype.
    mean_other : float
        Mean pathway score in other cells.
    mean_diff : float
        Mean difference (primary effect size for pathways).
        More appropriate than log fold change for activity scores.
    log_fold_change : float
        Alias for mean_diff (backward compatibility).
    statistic : float
        Mann-Whitney U test statistic.
    pvalue : float
        Raw p-value.
    test_direction : str
        Direction of test performed.
    direction : str
        Effect direction ('higher' or 'lower').
    passes_lfc_threshold : bool
        Whether result passes effect size threshold.
    fdr_pvalue : float | None
        FDR-corrected p-value.
    significant : bool | None
        Statistical significance.

    Notes
    -----
    Pathway scores represent activity levels (e.g., AUCell, pySCENIC),
    not expression counts. Mean difference is more interpretable than
    log fold change for these scores.
    """

    pathway: str
    archetype: str
    n_archetype_cells: int = Field(..., ge=0)
    n_other_cells: int = Field(..., ge=0)
    mean_archetype: float
    mean_other: float
    mean_diff: float  # Primary effect size for pathways
    log_fold_change: float  # Backward compatibility alias
    statistic: float
    pvalue: float = Field(..., ge=0, le=1)
    test_direction: str = Field(default="two-sided")
    direction: str = Field(..., pattern=r"^(higher|lower)$")
    passes_lfc_threshold: bool | None = Field(default=True)
    fdr_pvalue: float | None = Field(default=None, ge=0, le=1)
    significant: bool | None = Field(default=None)

    model_config = {"extra": "allow"}


# -----------------------------------------------------------------------------
# Conditional Association Types
# -----------------------------------------------------------------------------


class ConditionalAssociationResult(BaseModel):
    """Single archetype-condition enrichment test result.

    Returned as rows in test_archetype_conditional_associations() DataFrame.

    Attributes
    ----------
    archetype : str
        Archetype identifier.
    condition : str
        Condition value from obs_column.
    observed : int
        Observed count of archetype cells in condition.
    expected : float
        Expected count under null hypothesis.
    total_archetype : int
        Total cells in archetype.
    total_condition : int
        Total cells in condition.
    odds_ratio : float
        Odds ratio (enrichment measure). >1 = enriched, <1 = depleted.
    ci_lower : float
        Lower 95% confidence interval for odds ratio.
    ci_upper : float
        Upper 95% confidence interval for odds ratio.
    pvalue : float
        Hypergeometric p-value.
    fdr_pvalue : float | None
        FDR-corrected p-value.
    significant : bool | None
        Statistical significance.

    Examples
    --------
    >>> results = test_archetype_conditional_associations(adata, obs_column="sample")
    >>> enriched = results[(results["significant"]) & (results["odds_ratio"] > 2)]
    """

    archetype: str
    condition: str
    observed: int = Field(..., ge=0)
    expected: float = Field(..., ge=0)
    total_archetype: int = Field(..., ge=0)
    total_condition: int = Field(..., ge=0)
    odds_ratio: float = Field(..., ge=0)
    ci_lower: float
    ci_upper: float
    pvalue: float = Field(..., ge=0, le=1)
    fdr_pvalue: float | None = Field(default=None, ge=0, le=1)
    significant: bool | None = Field(default=None)

    model_config = {"extra": "allow"}


# -----------------------------------------------------------------------------
# Pattern Analysis Types
# -----------------------------------------------------------------------------


class ArchetypePattern(BaseModel):
    """Definition of an archetype pattern for testing.

    Generated by generate_archetype_patterns().

    Attributes
    ----------
    high_archetypes : list[str]
        Archetypes expected to have high values.
    low_archetypes : list[str]
        Archetypes expected to have low values.
    pattern_name : str
        Descriptive pattern name (e.g., 'specialist_arch1_1xxxx_0xxxx').
    pattern_code : str
        Visual pattern code (e.g., '12xxx_xx345').
        Position = archetype number, numbers = high, 'x' = low.
    pattern_type : str
        Type: 'specialization', 'tradeoff', or 'custom'.
    pattern_set : str | None
        Pattern set description (e.g., 'non-zero archetypes').

    Notes
    -----
    Pattern code format: "12xxx_xx345"
    - Position corresponds to archetype number (0, 1, 2, 3, 4, 5...)
    - Numbers = high archetypes, 'x' = low archetypes
    - Underscore separates high group from low group

    Examples
    --------
    >>> patterns = generate_archetype_patterns(unique_archetypes)
    >>> for p in patterns:
    ...     pattern = ArchetypePattern.model_validate(p)
    ...     print(f"{pattern.pattern_code}: {pattern.high_archetypes} vs {pattern.low_archetypes}")
    """

    high_archetypes: list[str]
    low_archetypes: list[str]
    pattern_name: str
    pattern_code: str
    pattern_type: str = Field(..., pattern=r"^(specialization|tradeoff|custom)$")
    pattern_set: str | None = Field(default=None)


class PatternAssociationResult(BaseModel):
    """Single pattern-feature association test result.

    Returned as rows in test_archetype_pattern_associations() DataFrame.

    Attributes
    ----------
    pathway : str | None
        Pathway name (if pathway data).
    gene : str | None
        Gene name (if gene data).
    pattern_name : str
        Interpretable pattern name.
    pattern_code : str
        Visual pattern code.
    pattern_type : str
        Pattern type ('specialization', 'tradeoff', 'custom').
    high_archetypes : str
        Comma-separated high archetype names.
    low_archetypes : str
        Comma-separated low archetype names.
    n_high_cells : int
        Number of cells in high group.
    n_low_cells : int
        Number of cells in low group.
    mean_high : float
        Mean value in high group.
    mean_low : float
        Mean value in low group.
    log_fold_change : float
        Effect size (log fold change or mean diff).
    primary_effect_size : float
        Standardized effect size (mean_diff for pathways, lfc for genes).
    effect_size_col : str
        Which effect size column was used ('mean_diff' or 'log_fold_change').
    statistic : float
        Test statistic.
    pvalue : float
        Raw p-value.
    test_direction : str
        Test direction.
    direction : str
        Effect direction.
    passes_lfc_threshold : bool
        Whether passes threshold.
    fdr_pvalue : float | None
        FDR-corrected p-value.
    significant : bool | None
        Statistical significance.

    archetype : str
        Alias for pattern_name (plotting compatibility).
    mean_archetype : float
        Alias for mean_high (plotting compatibility).
    mean_other : float
        Alias for mean_low (plotting compatibility).
    n_archetype_cells : int
        Alias for n_high_cells (plotting compatibility).
    n_other_cells : int
        Alias for n_low_cells (plotting compatibility).
    """

    # Feature identifier (one of these)
    pathway: str | None = Field(default=None)
    gene: str | None = Field(default=None)

    # Pattern information
    pattern_name: str
    pattern_code: str
    pattern_type: str
    high_archetypes: str
    low_archetypes: str

    # Group statistics
    n_high_cells: int = Field(..., ge=0)
    n_low_cells: int = Field(..., ge=0)
    mean_high: float
    mean_low: float

    # Effect sizes
    log_fold_change: float
    primary_effect_size: float | None = Field(default=None)
    effect_size_col: str | None = Field(default=None)
    mean_diff: float | None = Field(default=None)

    # Test results
    statistic: float
    pvalue: float = Field(..., ge=0, le=1)
    test_direction: str = Field(default="two-sided")
    direction: str
    passes_lfc_threshold: bool = Field(default=True)
    fdr_pvalue: float | None = Field(default=None, ge=0, le=1)
    significant: bool | None = Field(default=None)

    # Compatibility aliases
    archetype: str | None = Field(default=None)
    mean_archetype: float | None = Field(default=None)
    mean_other: float | None = Field(default=None)
    n_archetype_cells: int | None = Field(default=None)
    n_other_cells: int | None = Field(default=None)

    model_config = {"extra": "allow"}


class ExclusivePatternResult(BaseModel):
    """Result for archetype-exclusive feature.

    Returned by identify_archetype_exclusive_patterns().

    Attributes
    ----------
    pathway : str | None
        Pathway name (if pathway data).
    gene : str | None
        Gene name (if gene data).
    archetype : str
        The archetype where this feature is exclusively high.
    n_archetype_cells : int
        Cells in exclusive archetype.
    n_other_cells : int
        Cells in all other archetypes.
    mean_archetype : float
        Mean in exclusive archetype.
    mean_other : float
        Mean in other archetypes.
    mean_diff : float | None
        Effect size (pathways).
    log_fold_change : float | None
        Effect size (genes).
    min_pairwise_effect : float | None
        Minimum effect vs any other archetype (pairwise mode).
    max_pairwise_pvalue : float | None
        Maximum p-value vs any other archetype (pairwise mode).
    pvalue : float
        P-value (most conservative if pairwise).
    statistic : float
        Test statistic.
    direction : str
        Always 'higher' for exclusive patterns.
    pattern_type : str
        'exclusive' or 'exclusive_pairwise'.
    exclusivity_score : float
        Ratio of expression in target vs max other archetype.
        Higher = more exclusive.
    fdr_pvalue : float | None
        FDR-corrected p-value.
    significant : bool | None
        Statistical significance.

    Examples
    --------
    >>> exclusive = identify_archetype_exclusive_patterns(adata)
    >>> # Find highly exclusive features
    >>> top_exclusive = exclusive.nlargest(20, "exclusivity_score")
    """

    # Feature identifier
    pathway: str | None = Field(default=None)
    gene: str | None = Field(default=None)

    # Archetype info
    archetype: str
    n_archetype_cells: int = Field(..., ge=0)
    n_other_cells: int = Field(..., ge=0)
    mean_archetype: float
    mean_other: float

    # Effect sizes
    mean_diff: float | None = Field(default=None)
    log_fold_change: float | None = Field(default=None)
    min_pairwise_effect: float | None = Field(default=None)
    max_pairwise_pvalue: float | None = Field(default=None)

    # Test results
    pvalue: float = Field(..., ge=0, le=1)
    statistic: float = Field(default=0)
    direction: str = Field(default="higher")
    pattern_type: str = Field(..., pattern=r"^(exclusive|exclusive_pairwise)$")
    exclusivity_score: float = Field(..., ge=0)
    pattern_code: str | None = Field(default=None)

    # FDR results
    fdr_pvalue: float | None = Field(default=None, ge=0, le=1)
    significant: bool | None = Field(default=None)

    model_config = {"extra": "allow"}


class MutualExclusivityResult(BaseModel):
    """Result for mutual exclusivity pattern.

    Returned by identify_mutual_exclusivity_patterns().

    Attributes
    ----------
    pathway : str | None
        Pathway name (if pathway data).
    gene : str | None
        Gene name (if gene data).
    positive_patterns : list[str]
        Patterns where feature is high.
    negative_patterns : list[str]
        Patterns where feature is low.
    positive_pattern_codes : list[str]
        Visual codes for positive patterns.
    negative_pattern_codes : list[str]
        Visual codes for negative patterns.
    tradeoff_score : int
        Number of patterns involved (higher = more complex tradeoff).
    max_positive_effect : float
        Maximum positive effect size.
    min_negative_effect : float
        Minimum negative effect size.
    effect_range : float
        Range of effects (max_positive - min_negative).
    primary_effect_size : float
        Standardized effect size.
    effect_size_col : str
        Effect size column used.
    effect_range_name : str
        Name for effect range ('mean_diff_range' or 'lfc_range').

    archetype : str
        Pattern identifier (for plotting compatibility).
    pattern_name : str
        Pattern identifier.
    pattern_code : str
        Visual pattern code.
    log_fold_change : float
        For plotting compatibility.
    mean_archetype : float
        For plotting compatibility (= tradeoff_score).
    pvalue : float
        Mock p-value for patterns.
    fdr_pvalue : float
        Mock FDR p-value.
    significant : bool
        Always True (pre-filtered).
    """

    # Feature identifier
    pathway: str | None = Field(default=None)
    gene: str | None = Field(default=None)

    # Pattern lists
    positive_patterns: list[str]
    negative_patterns: list[str]
    positive_pattern_codes: list[str]
    negative_pattern_codes: list[str]

    # Scores
    tradeoff_score: int = Field(..., ge=2)
    max_positive_effect: float
    min_negative_effect: float
    effect_range: float = Field(..., ge=0)
    primary_effect_size: float
    effect_size_col: str
    effect_range_name: str | None = Field(default=None)
    mean_diff: float | None = Field(default=None)

    # Compatibility fields
    archetype: str
    pattern_name: str
    pattern_code: str
    log_fold_change: float
    mean_archetype: float  # = tradeoff_score
    pvalue: float = Field(default=0.01, ge=0, le=1)
    fdr_pvalue: float = Field(default=0.05, ge=0, le=1)
    significant: bool = Field(default=True)

    model_config = {"extra": "allow"}


# -----------------------------------------------------------------------------
# Comprehensive Analysis Types
# -----------------------------------------------------------------------------


class ComprehensivePatternResults(BaseModel):
    """Results from analyze_archetypal_patterns_comprehensive().

    Attributes
    ----------
    individual : Any
        Individual archetype characterization results (DataFrame).
        Standard 1-vs-all archetype tests.
    patterns : Any
        Pattern-based test results (DataFrame).
        Specialization, tradeoff, and complex patterns.
    exclusivity : Any
        Mutual exclusivity analysis results (DataFrame).
        Features with opposing patterns across archetypes.

    Notes
    -----
    Each attribute is a pandas DataFrame. The Pydantic model uses `Any`
    because DataFrames aren't directly serializable, but the structure
    documents what each contains.

    Examples
    --------
    >>> results = analyze_archetypal_patterns_comprehensive(adata)
    >>> # Access individual results
    >>> individual_df = results["individual"]
    >>> # Access pattern results
    >>> patterns_df = results["patterns"]
    >>> specialists = patterns_df[patterns_df["pattern_type"] == "specialization"]
    >>> # Access exclusivity results
    >>> if not results["exclusivity"].empty:
    ...     exclusive_features = results["exclusivity"]
    """

    individual: Any  # pd.DataFrame
    patterns: Any  # pd.DataFrame
    exclusivity: Any  # pd.DataFrame

    model_config = {"extra": "allow"}


# -----------------------------------------------------------------------------
# FDR Correction Types
# -----------------------------------------------------------------------------


class FDRCorrectionConfig(BaseModel):
    """Configuration for apply_fdr_correction().

    Attributes
    ----------
    pvalue_column : str
        Column name containing raw p-values. Default: 'pvalue'.
    method : str
        FDR method: 'benjamini_hochberg' or 'bonferroni'.
    alpha : float
        Significance threshold. Default: 0.05.
    validate_assumptions : bool
        Whether to validate FDR assumptions. Default: True.

    Examples
    --------
    >>> config = FDRCorrectionConfig(method="bonferroni", alpha=0.01)
    """

    pvalue_column: str = Field(default="pvalue")
    method: str = Field(default="benjamini_hochberg", pattern=r"^(benjamini_hochberg|bonferroni)$")
    alpha: float = Field(default=0.05, gt=0, lt=1)
    validate_assumptions: bool = Field(default=True)


# -----------------------------------------------------------------------------
# Reference Classes for Keys
# -----------------------------------------------------------------------------


class StatisticalTestKeys:
    """Standard keys used in statistical testing results.

    Use these constants to avoid typos when accessing result columns.

    Examples
    --------
    >>> results = test_archetype_gene_associations(adata)
    >>> sig_mask = results[StatisticalTestKeys.SIGNIFICANT]
    >>> genes = results.loc[sig_mask, StatisticalTestKeys.GENE]
    """

    # Common columns
    GENE = "gene"
    PATHWAY = "pathway"
    ARCHETYPE = "archetype"
    PVALUE = "pvalue"
    FDR_PVALUE = "fdr_pvalue"
    SIGNIFICANT = "significant"
    DIRECTION = "direction"

    # Effect sizes
    LOG_FOLD_CHANGE = "log_fold_change"
    MEAN_DIFF = "mean_diff"
    PRIMARY_EFFECT_SIZE = "primary_effect_size"

    # Statistics
    STATISTIC = "statistic"
    N_ARCHETYPE_CELLS = "n_archetype_cells"
    N_OTHER_CELLS = "n_other_cells"
    MEAN_ARCHETYPE = "mean_archetype"
    MEAN_OTHER = "mean_other"

    # Pattern-specific
    PATTERN_NAME = "pattern_name"
    PATTERN_CODE = "pattern_code"
    PATTERN_TYPE = "pattern_type"
    HIGH_ARCHETYPES = "high_archetypes"
    LOW_ARCHETYPES = "low_archetypes"

    # Exclusivity-specific
    EXCLUSIVITY_SCORE = "exclusivity_score"
    TRADEOFF_SCORE = "tradeoff_score"
    EFFECT_RANGE = "effect_range"

    # Conditional-specific
    CONDITION = "condition"
    OBSERVED = "observed"
    EXPECTED = "expected"
    ODDS_RATIO = "odds_ratio"
    CI_LOWER = "ci_lower"
    CI_UPPER = "ci_upper"


# -----------------------------------------------------------------------------
# Validation Helpers
# -----------------------------------------------------------------------------


def validate_statistical_results(df: Any, result_type: str = "gene") -> bool:
    """Validate statistical results DataFrame has expected columns.

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame to validate.
    result_type : str
        Type of results: 'gene', 'pathway', 'conditional', 'pattern', 'exclusive'.

    Returns
    -------
    bool
        True if validation passes.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    required_columns = {
        "gene": ["gene", "archetype", "pvalue", "log_fold_change", "direction"],
        "pathway": ["pathway", "archetype", "pvalue", "mean_diff", "direction"],
        "conditional": ["archetype", "condition", "pvalue", "odds_ratio"],
        "pattern": ["pattern_name", "pattern_code", "pvalue", "log_fold_change"],
        "exclusive": ["archetype", "pvalue", "exclusivity_score", "direction"],
    }

    if result_type not in required_columns:
        raise ValueError(f"Unknown result_type: {result_type}")

    missing = set(required_columns[result_type]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for {result_type} results: {missing}")

    return True


from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# =============================================================================
# GENE ANALYSIS / PATHWAY SCORING TYPES (_core/utils/gene_analysis.py)
# =============================================================================


# -----------------------------------------------------------------------------
# Enums and Constants
# -----------------------------------------------------------------------------


class MSigDBCollection(str, Enum):
    """MSigDB gene set collections.

    See https://www.gsea-msigdb.org/gsea/msigdb/collections.jsp
    """

    HALLMARK = "hallmark"
    C2_CP = "c2_cp"  # Canonical pathways
    C2_CGP = "c2_cgp"  # Chemical/genetic perturbations
    C3_MIR = "c3_mir"  # microRNA targets
    C5_BP = "c5_bp"  # GO Biological Process
    C5_CC = "c5_cc"  # GO Cellular Component
    C5_MF = "c5_mf"  # GO Molecular Function
    C8 = "c8"  # Cell type signatures


class GenesetRepository(str, Enum):
    """Gene set repository options."""

    MSIGDB = "msigdb"
    # OMNIPATH = "omnipath"  # Commented out in implementation


class Organism(str, Enum):
    """Supported organisms for pathway analysis."""

    HUMAN = "human"
    MOUSE = "mouse"


class PathwayScoringMethod(str, Enum):
    """Pathway scoring methods."""

    AUCELL = "aucell"
    # ULM = "ulm"  # Commented out - not currently used


# -----------------------------------------------------------------------------
# Pathway Network Types
# -----------------------------------------------------------------------------


class PathwayNetworkRow(BaseModel):
    """Single row in pathway network DataFrame.

    Represents a gene-pathway membership relationship.

    Attributes
    ----------
    source : str
        Pathway/gene set name.
    target : str
        Gene symbol.
    weight : float
        Membership weight. Always 1.0 for binary membership.
    pathway : str
        Pathway database source (e.g., 'HALLMARK', 'C5_BP').

    Examples
    --------
    >>> row = PathwayNetworkRow(source="HALLMARK_HYPOXIA", target="VEGFA", weight=1.0, pathway="HALLMARK")
    """

    source: str  # Pathway name
    target: str  # Gene symbol
    weight: float = Field(default=1.0, ge=0)
    pathway: str  # Database source


class PathwayNetworkConfig(BaseModel):
    """Configuration for load_pathway_networks().

    Attributes
    ----------
    sources : list[str]
        Pathway databases to load.
        MSigDB options: 'hallmark', 'c2_cp', 'c2_cgp', 'c3_mir',
        'c5_bp', 'c5_cc', 'c5_mf', 'c8'.
    organism : str
        Species: 'human' or 'mouse'.
    geneset_repo : str
        Repository: 'msigdb' (recommended).

    Examples
    --------
    >>> config = PathwayNetworkConfig(sources=["hallmark", "c5_bp"], organism="human")
    """

    sources: list[str] = Field(default=["c5_bp"])
    organism: str = Field(default="human", pattern=r"^(human|mouse)$")
    geneset_repo: str = Field(default="msigdb", pattern=r"^(msigdb)$")

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v: list[str]) -> list[str]:
        """Validate source names."""
        valid_sources = {"hallmark", "c2_cp", "c2_cgp", "c3_mir", "c5_bp", "c5_cc", "c5_mf", "c8"}
        for source in v:
            if source.lower() not in valid_sources:
                raise ValueError(f"Unknown source '{source}'. Valid options: {valid_sources}")
        return v


class PathwayNetworkSummary(BaseModel):
    """Summary statistics for loaded pathway network.

    Attributes
    ----------
    total_pathways : int
        Total number of unique pathways.
    total_genes : int
        Total number of unique genes.
    total_interactions : int
        Total gene-pathway pairs.
    pathways_per_database : dict[str, int]
        Number of pathways per database source.

    Examples
    --------
    >>> net = load_pathway_networks(["hallmark", "c5_bp"])
    >>> summary = PathwayNetworkSummary(
    ...     total_pathways=net["source"].nunique(),
    ...     total_genes=net["target"].nunique(),
    ...     total_interactions=len(net),
    ...     pathways_per_database=net.groupby("pathway")["source"].nunique().to_dict(),
    ... )
    """

    total_pathways: int = Field(..., ge=0)
    total_genes: int = Field(..., ge=0)
    total_interactions: int = Field(..., ge=0)
    pathways_per_database: dict[str, int]


# -----------------------------------------------------------------------------
# Pathway Scoring Types
# -----------------------------------------------------------------------------


class PathwayScoreConfig(BaseModel):
    """Configuration for compute_pathway_scores().

    Attributes
    ----------
    use_layer : str | None
        AnnData layer to use. None uses adata.X.
    obsm_key : str
        Key for storing results in adata.obsm.
        Default: 'pathway_scores'.
    method : str
        Scoring method. Currently only 'aucell' supported.

    Examples
    --------
    >>> config = PathwayScoreConfig(use_layer="logcounts", obsm_key="hallmark_scores")
    """

    use_layer: str | None = Field(default=None)
    obsm_key: str = Field(default="pathway_scores")
    method: str = Field(default="aucell", pattern=r"^(aucell)$")


class GeneOverlapStats(BaseModel):
    """Statistics for gene overlap between expression data and pathway network.

    Attributes
    ----------
    expression_genes : int
        Number of genes in expression data.
    pathway_genes : int
        Number of genes in pathway network.
    overlapping_genes : int
        Number of genes in both datasets.
    overlap_percentage : float
        Percentage of expression genes with pathway annotations.
    case_insensitive_overlap : int | None
        Overlap after case normalization (if different).
    normalized_improvement : int | None
        Additional genes matched after normalization.

    Examples
    --------
    >>> stats = GeneOverlapStats(
    ...     expression_genes=15000, pathway_genes=8000, overlapping_genes=6500, overlap_percentage=43.3
    ... )
    >>> if stats.overlap_percentage < 30:
    ...     print("Warning: Low gene overlap")
    """

    expression_genes: int = Field(..., ge=0)
    pathway_genes: int = Field(..., ge=0)
    overlapping_genes: int = Field(..., ge=0)
    overlap_percentage: float = Field(..., ge=0, le=100)
    case_insensitive_overlap: int | None = Field(default=None, ge=0)
    normalized_improvement: int | None = Field(default=None, ge=0)


class PathwayScoreSummary(BaseModel):
    """Summary statistics for computed pathway scores.

    Attributes
    ----------
    n_cells : int
        Number of cells scored.
    n_pathways : int
        Number of pathways with scores.
    score_min : float
        Minimum score across all cells and pathways.
    score_max : float
        Maximum score across all cells and pathways.
    score_mean : float
        Mean score across all cells and pathways.
    score_std : float
        Standard deviation of scores.
    method : str
        Scoring method used ('aucell').
    most_variable_pathways : list[str]
        Top 5 most variable pathways by variance.

    Examples
    --------
    >>> summary = PathwayScoreSummary(
    ...     n_cells=5000,
    ...     n_pathways=50,
    ...     score_min=0.0,
    ...     score_max=0.45,
    ...     score_mean=0.12,
    ...     score_std=0.08,
    ...     method="aucell",
    ...     most_variable_pathways=["HALLMARK_HYPOXIA", "HALLMARK_GLYCOLYSIS", ...],
    ... )
    """

    n_cells: int = Field(..., ge=0)
    n_pathways: int = Field(..., ge=0)
    score_min: float
    score_max: float
    score_mean: float
    score_std: float = Field(..., ge=0)
    method: str
    most_variable_pathways: list[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# AnnData Storage Types
# -----------------------------------------------------------------------------


class PathwayScoreAnnDataStorage(BaseModel):
    """Documents pathway score storage in AnnData.

    Attributes
    ----------
    obsm_key : str
        Key in adata.obsm for score matrix.
        Default: 'pathway_scores'.
    obsm_shape : tuple[int, int]
        Shape of score matrix [n_cells, n_pathways].
    uns_pathways_key : str
        Key in adata.uns for pathway names list.
        Default: '{obsm_key}_pathways'.
    uns_method_key : str
        Key in adata.uns for scoring method.
        Default: '{obsm_key}_method'.

    Notes
    -----
    After compute_pathway_scores(), access scores as:

    - ``adata.obsm['pathway_scores']`` : Score matrix [n_cells, n_pathways]
    - ``adata.uns['pathway_scores_pathways']`` : List of pathway names
    - ``adata.uns['pathway_scores_method']`` : Scoring method ('aucell')

    Examples
    --------
    >>> # After compute_pathway_scores(adata, net)
    >>> scores = adata.obsm["pathway_scores"]  # [n_cells, n_pathways]
    >>> pathway_names = adata.uns["pathway_scores_pathways"]
    >>> # Get score for specific pathway
    >>> hypoxia_idx = pathway_names.index("HALLMARK_HYPOXIA")
    >>> hypoxia_scores = scores[:, hypoxia_idx]
    """

    obsm_key: str = Field(default="pathway_scores")
    obsm_shape: tuple = Field(...)  # (n_cells, n_pathways)
    uns_pathways_key: str = Field(default="pathway_scores_pathways")
    uns_method_key: str = Field(default="pathway_scores_method")


# -----------------------------------------------------------------------------
# Reference Classes for Keys
# -----------------------------------------------------------------------------


class PathwayAnalysisKeys:
    """Standard keys used in pathway analysis.

    Use these constants for consistent key access across the codebase.

    Examples
    --------
    >>> scores = adata.obsm[PathwayAnalysisKeys.PATHWAY_SCORES]
    >>> names = adata.uns[PathwayAnalysisKeys.PATHWAY_NAMES]
    """

    # AnnData.obsm keys
    PATHWAY_SCORES = "pathway_scores"

    # AnnData.uns keys
    PATHWAY_NAMES = "pathway_scores_pathways"
    SCORING_METHOD = "pathway_scores_method"

    # Network DataFrame columns
    NET_SOURCE = "source"  # Pathway name
    NET_TARGET = "target"  # Gene symbol
    NET_WEIGHT = "weight"  # Membership weight
    NET_PATHWAY = "pathway"  # Database source


# -----------------------------------------------------------------------------
# MSigDB Collection Mapping
# -----------------------------------------------------------------------------


MSIGDB_COLLECTION_MAP: dict[str, str] = {
    "hallmark": "h.all",
    "c2_cp": "c2.cp",
    "c2_cgp": "c2.cgp",
    "c3_mir": "c3.mir",
    "c5_bp": "c5.go.bp",
    "c5_cc": "c5.go.cc",
    "c5_mf": "c5.go.mf",
    "c8": "c8.all",
}
"""Mapping from short source names to MSigDB category prefixes."""


MSIGDB_COLLECTION_DESCRIPTIONS: dict[str, str] = {
    "hallmark": "Hallmark gene sets (50 curated sets)",
    "c2_cp": "Canonical pathways (KEGG, Reactome, BioCarta, etc.)",
    "c2_cgp": "Chemical and genetic perturbations",
    "c3_mir": "microRNA targets",
    "c5_bp": "GO Biological Process",
    "c5_cc": "GO Cellular Component",
    "c5_mf": "GO Molecular Function",
    "c8": "Cell type signature gene sets",
}
"""Human-readable descriptions of MSigDB collections."""

# =============================================================================
# GENERIC DATAFRAME VALIDATION
# =============================================================================


import pandas as pd


def validate_dataframe_schema(
    df: pd.DataFrame,
    model: type[BaseModel],
    *,
    required_only: bool = True,
    sample_n: int = 10,
    strict: bool = False,
) -> bool:
    """Validate DataFrame rows against a Pydantic model schema.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    model : type[BaseModel]
        Pydantic model defining expected schema.
    required_only : bool, default True
        If True, only check required columns are present.
        If False, check all model fields exist as columns.
    sample_n : int, default 10
        Number of rows to validate (for performance). Ignored if strict=True.
    strict : bool, default False
        If True, validate all rows. Use for small DataFrames or CI.

    Returns
    -------
    bool
        True if validation passes.

    Raises
    ------
    ValueError
        If required columns are missing or row validation fails.

    Examples
    --------
    >>> from peach._core.types import validate_dataframe_schema, GeneAssociationResult
    >>> results = pc.tl.gene_associations(adata)
    >>> validate_dataframe_schema(results, GeneAssociationResult)
    True
    >>> # Strict validation in tests
    >>> validate_dataframe_schema(results, GeneAssociationResult, strict=True)
    """
    if df.empty:
        return True  # Empty DataFrame trivially valid

    # Get required vs optional fields from model
    required_fields: set[str] = set()
    optional_fields: set[str] = set()

    for name, field_info in model.model_fields.items():
        if field_info.is_required():
            required_fields.add(name)
        else:
            optional_fields.add(name)

    # Determine which fields to check
    check_fields = required_fields if required_only else (required_fields | optional_fields)
    missing = check_fields - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns for {model.__name__}: {sorted(missing)}\nDataFrame has: {sorted(df.columns)}"
        )

    # Validate row contents
    rows_to_check = df if strict else df.head(sample_n)
    all_model_fields = required_fields | optional_fields

    for idx, row in rows_to_check.iterrows():
        try:
            # Only pass columns that exist in the model (ignore extras)
            row_dict = {k: v for k, v in row.to_dict().items() if k in all_model_fields}
            model.model_validate(row_dict)
        except Exception as e:
            raise ValueError(
                f"Row {idx} failed validation for {model.__name__}: {e}\nRow data (model fields only): {row_dict}"
            )

    return True


# Convenience mapping for string-based validation
RESULT_TYPE_MODELS: dict[str, type[BaseModel]] = {
    # Statistical testing results
    "gene_association": GeneAssociationResult,
    "pathway_association": PathwayAssociationResult,
    "conditional_association": ConditionalAssociationResult,
    "pattern_association": PatternAssociationResult,
    "exclusive_pattern": ExclusivePatternResult,
    "mutual_exclusivity": MutualExclusivityResult,
}


def validate_results(df: pd.DataFrame, result_type: str, **kwargs) -> bool:
    """Validate results DataFrame by type name.

    Convenience wrapper around validate_dataframe_schema().

    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame.
    result_type : str
        One of: 'gene_association', 'pathway_association',
        'conditional_association', 'pattern_association',
        'exclusive_pattern', 'mutual_exclusivity'.
    **kwargs
        Passed to validate_dataframe_schema (required_only, sample_n, strict).

    Returns
    -------
    bool
        True if valid.

    Raises
    ------
    ValueError
        If result_type unknown or validation fails.

    Examples
    --------
    >>> from peach._core.types import validate_results
    >>> results = pc.tl.gene_associations(adata)
    >>> validate_results(results, "gene_association")
    True
    >>> # In pytest
    >>> validate_results(results, "gene_association", strict=True)
    """
    if result_type not in RESULT_TYPE_MODELS:
        raise ValueError(f"Unknown result_type: '{result_type}'. Valid options: {list(RESULT_TYPE_MODELS.keys())}")

    return validate_dataframe_schema(df, RESULT_TYPE_MODELS[result_type], **kwargs)


def get_required_columns(result_type: str) -> set[str]:
    """Get required columns for a result type.

    Useful for debugging schema mismatches.

    Parameters
    ----------
    result_type : str
        Result type name (see validate_results for options).

    Returns
    -------
    set[str]
        Required column names.

    Examples
    --------
    >>> from peach._core.types import get_required_columns
    >>> get_required_columns("gene_association")
    {'gene', 'archetype', 'pvalue', 'log_fold_change', 'direction', ...}
    """
    if result_type not in RESULT_TYPE_MODELS:
        raise ValueError(f"Unknown result_type: '{result_type}'")

    model = RESULT_TYPE_MODELS[result_type]
    return {name for name, field_info in model.model_fields.items() if field_info.is_required()}



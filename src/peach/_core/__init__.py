# Core implementation modules
# These are the actual implementations that power the high-level API

# Import submodules to make them accessible
from peach._core import models, types, utils, viz

# Expose commonly used items for convenience
from peach._core.models import Deep_AA, VAE_Base

# Expose validation utilities
from peach._core.types import (
    AnnDataKeys,
    CVResultsModel,
    GeneAssociationResult,
    PathwayAssociationResult,
    TrainingResults,
    validate_dataframe_schema,
    validate_results,
    validate_training_results,
)
from peach._core.utils import (
    SearchConfig,
    bin_cells_by_archetype,
    calculate_archetype_r2,
    compute_archetype_distances,
    train_vae,
)

__all__ = [
    "models",
    "utils",
    "viz",
    "types",
    # Commonly used classes
    "Deep_AA",
    "VAE_Base",
    # Commonly used functions
    "train_vae",
    "calculate_archetype_r2",
    "compute_archetype_distances",
    "bin_cells_by_archetype",
    "SearchConfig",
    # Validation utilities
    "validate_results",
    "validate_training_results",
    "validate_dataframe_schema",
    # Key types
    "TrainingResults",
    "CVResultsModel",
    "GeneAssociationResult",
    "PathwayAssociationResult",
    "AnnDataKeys",
]

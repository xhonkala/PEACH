"""Peach: Archetypal analysis for single-cell genomics.

This package provides archetypal analysis capabilities for single-cell data,
following scverse ecosystem conventions.

The package can be imported as `import peach as pc` and provides:
- High-level API: pc.pp, pc.tl, pc.pl (preprocessing, tools, plotting)
- Core implementations: pc._core (models, utils, viz)
- Direct access to commonly used items: pc.Deep_AA, pc.train_vae, pc.calculate_archetype_r2, etc.
"""

# High-level API modules
# Core implementation modules
from . import _core, datasets, pl, pp, tl

# Expose commonly used core items for convenience
from ._core.models import (
    Deep_AA,
    VAE_Base,
)
from ._core.utils import (
    PCHA,
    ArchetypalGridSearch,
    CVTrainingManager,
    SearchConfig,
    archetypal_R2,
    bin_cells_by_archetype,
    calculate_archetype_r2,
    compute_archetype_distances,
    get_archetype_positions,
    train_vae,
)

__version__ = "0.4.0"
__author__ = "Alexander Honkala"

__all__ = [
    # High-level API
    "pp",
    "tl",
    "pl",
    "datasets",
    # Core modules
    "_core",
    # Models
    "Deep_AA",
    "VAE_Base",
    # Utils
    "train_vae",
    "calculate_archetype_r2",
    "archetypal_R2",
    "compute_archetype_distances",
    "bin_cells_by_archetype",
    "get_archetype_positions",
    "SearchConfig",
    "ArchetypalGridSearch",
    "CVTrainingManager",
    "PCHA",
]

# Type hints and validation available via pc._core.types
# Example: from peach._core.types import TrainingResults, validate_results

# make utils visible
# Note: model_selection.py removed - use CVSummary methods instead
# Note: GridSearchResults deprecated - use CVSummary instead
# making metrics visible
from peach._core.utils.metrics import (
    MetricsTracker,
    calculate_archetype_r2,
    calculate_epoch_metrics,
    calculate_vae_metrics,
)

from .analysis import *
from .convex_synth_data import generate_convex_data
from .cv_training import CVTrainingConfig, CVTrainingManager
from .grid_search_results import CVResults, CVSummary

# Hyperparameter search modules - SIMPLIFIED ARCHITECTURE
from .hyperparameter_search import ArchetypalGridSearch, SearchConfig
from .load_anndata import load_anndata
from .metrics import MetricsTracker, calculate_epoch_metrics, calculate_vae_metrics
from .PCHA import *
from .training import train_vae

# Alias for compatibility/cleaner API
archetypal_R2 = calculate_archetype_r2

__all__ = [
    # Metrics
    "calculate_archetype_r2",
    "archetypal_R2",
    "MetricsTracker",
    "calculate_vae_metrics",
    "calculate_epoch_metrics",
    # Training
    "train_vae",
    # Data generation
    "generate_convex_data",
    # Data loading
    "load_anndata",
    # Hyperparameter search
    "ArchetypalGridSearch",
    "SearchConfig",
    "CVTrainingManager",
    "CVTrainingConfig",
    "CVSummary",
    "CVResults",
    # Analysis functions (from analysis.py)
    "get_archetypal_coordinates",
    "test_archetype_recovery",
    "compare_archetypal_recovery",
    "bin_cells_by_archetype",
    "select_cells",
    "get_all_archetypal_coordinates",
    "compute_archetype_distances",
    "extract_and_store_archetypal_coordinates",
    "get_archetype_positions",
    # PCHA functions
    "furthest_sum",
    "PCHA",
    "run_pcha_analysis",
]

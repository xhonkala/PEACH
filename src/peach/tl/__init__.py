"""Tools for archetypal analysis."""

# Import SearchConfig from core for API access
from .._core.utils.hyperparameter_search import SearchConfig
from .archetypal import (
    archetypal_coordinates,
    assign_archetypes,
    assign_to_centroids,
    compute_conditional_centroids,
    extract_archetype_weights,
    train_archetypal,
)

# CellRank integration
from .cellrank_integration import (
    compute_lineage_drivers,
    compute_lineage_pseudotimes,
    compute_transition_frequencies,
    setup_cellrank,
    single_trajectory_analysis,
)
from .hyperparameters import hyperparameter_search
from .statistical import (
    archetype_exclusive_patterns,
    conditional_associations,
    gene_associations,
    pathway_associations,
    pattern_analysis,
    specialization_patterns,
    tradeoff_patterns,
)

__all__ = [
    "train_archetypal",
    "archetypal_coordinates",
    "assign_archetypes",
    "extract_archetype_weights",
    "compute_conditional_centroids",
    "assign_to_centroids",
    "gene_associations",
    "pathway_associations",
    "pattern_analysis",
    "conditional_associations",
    "archetype_exclusive_patterns",
    "specialization_patterns",
    "tradeoff_patterns",
    "hyperparameter_search",
    "SearchConfig",
    "setup_cellrank",
    "compute_lineage_pseudotimes",
    "compute_lineage_drivers",
    "compute_transition_frequencies",
    "single_trajectory_analysis",
]

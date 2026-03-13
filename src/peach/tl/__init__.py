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

# Spatial analysis (requires squidpy)
from .spatial import (
    archetype_co_occurrence,
    archetype_interaction_boundaries,
    archetype_nhood_enrichment,
    archetype_pair_enrichment,
    archetype_spatial_autocorr,
    spatial_neighbors,
)
from .statistical import (
    archetype_exclusive_patterns,
    conditional_associations,
    gene_associations,
    pathway_associations,
    pattern_analysis,
    specialization_patterns,
    tradeoff_patterns,
)

# v0.5.0: Simplex regression + driver regression
from .feature_regression import (
    archetype_driver_regression,
    feature_simplex_regression,
    gene_simplex_regression,
    pathway_simplex_regression,
)

# v0.5.0: Pattern classification + archetype summary
from .feature_patterns import archetype_summary, classify_feature_patterns

# v0.5.0: Simplex density decomposition
from .feature_decomposition import feature_simplex_decomposition, component_regression

# v0.5.0: Flow matching
from .flow import (
    flow_within,
    flow_between,
    flow_gene_alignment,
    flow_jacobian,
    flow_significance,
    flow_feature_graph,
    flow_temporal_feature_graph,
)

# v0.5.0: Archetype comparison
from .comparison import (
    archetype_mmd,
    archetype_feature_similarity,
    archetype_contrasts,
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
    # Spatial
    "spatial_neighbors",
    "archetype_nhood_enrichment",
    "archetype_co_occurrence",
    "archetype_spatial_autocorr",
    "archetype_interaction_boundaries",
    "archetype_pair_enrichment",
    # v0.5.0: Regression
    "archetype_driver_regression",
    "feature_simplex_regression",
    "gene_simplex_regression",
    "pathway_simplex_regression",
    # v0.5.0: Pattern classification + archetype summary
    "classify_feature_patterns",
    "archetype_summary",
    # v0.5.0: Simplex density decomposition
    "feature_simplex_decomposition",
    "component_regression",
    # v0.5.0: Flow matching
    "flow_within",
    "flow_between",
    "flow_gene_alignment",
    "flow_jacobian",
    "flow_significance",
    "flow_feature_graph",
    "flow_temporal_feature_graph",
    # v0.5.0: Archetype comparison
    "archetype_mmd",
    "archetype_feature_similarity",
    "archetype_contrasts",
]

"""Plotting functions for archetypal analysis."""

from .archetypal import (
    archetypal_space,
    archetypal_space_multi,
    archetype_positions,
    archetype_positions_3d,
    archetype_statistics,
    elbow_curve,
    training_metrics,
)

# CellRank visualization
# Note: gene_trends removed - use cellrank.pl.gene_trends() directly
from .cellrank_viz import fate_probabilities, lineage_drivers
from .pattern_visualization import pattern_dotplot, pattern_heatmap, pattern_summary_barplot
from .results import dotplot

# Spatial visualization (requires squidpy for analysis, plotly for plots)
from .spatial import (
    co_occurrence,
    cross_correlations,
    interaction_boundaries,
    nhood_enrichment,
    spatial_archetypes,
    spatial_autocorr,
)

# Regression visualization
from .regression import (
    coefficient_heatmap,
    interaction_heatmap,
    r2_barplot,
    vertex_radar,
    regression_volcano,
    archetype_regression_dotplot,
    archetype_radar_ridgeplot,
    pattern_summary,
)

# Ternary simplex visualization
from .ternary import ternary_facet, ternary_facet_grid

# GMM decomposition visualization
from .decomposition import (
    component_scatter,
    gmm_bic_curve,
    component_heatmap,
    component_archetype_summary,
    component_neighborhood_graph,
    component_stability,
)

# Flow matching visualization
from .flow import (
    velocity_quiver,
    gene_alignment_barplot,
    jacobian_heatmap,
    trajectory_ribbon,
    flow_magnitude,
    density_comparison,
    archetype_correspondence,
    soft_assignment_flow,
    flow_topo_landscape,
)

# Archetype comparison visualization
from .comparison import (
    mmd_heatmap,
    contrast_volcano,
    contrast_volcano_grid,
    feature_similarity_heatmap,
)

__all__ = [
    "archetypal_space",
    "archetypal_space_multi",
    "training_metrics",
    "elbow_curve",
    "dotplot",
    "archetype_positions",
    "archetype_positions_3d",
    "archetype_statistics",
    "pattern_dotplot",
    "pattern_summary_barplot",
    "pattern_heatmap",
    "fate_probabilities",
    "lineage_drivers",
    # Spatial
    "nhood_enrichment",
    "co_occurrence",
    "spatial_archetypes",
    "interaction_boundaries",
    "spatial_autocorr",
    "cross_correlations",
    # Regression
    "coefficient_heatmap",
    "interaction_heatmap",
    "r2_barplot",
    "vertex_radar",
    "regression_volcano",
    "archetype_regression_dotplot",
    "archetype_radar_ridgeplot",
    "pattern_summary",
    # Ternary
    "ternary_facet",
    "ternary_facet_grid",
    # GMM Decomposition
    "component_scatter",
    "gmm_bic_curve",
    "component_heatmap",
    "component_archetype_summary",
    "component_neighborhood_graph",
    "component_stability",
    # Flow Matching
    "velocity_quiver",
    "gene_alignment_barplot",
    "jacobian_heatmap",
    "trajectory_ribbon",
    "flow_magnitude",
    "density_comparison",
    "archetype_correspondence",
    "soft_assignment_flow",
    "flow_topo_landscape",
    # Archetype comparison
    "mmd_heatmap",
    "contrast_volcano",
    "contrast_volcano_grid",
    "feature_similarity_heatmap",
]

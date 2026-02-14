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
]

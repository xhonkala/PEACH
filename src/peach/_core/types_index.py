"""PEACH types index — AUTO-GENERATED from inspect + docstring parsing.

DO NOT EDIT MANUALLY. Regenerate with: python scripts/_regenerate_types_index.py

Maps every public PEACH function to its return type and key fields.
"""

from __future__ import annotations

# Function -> (return_type, [key_fields])
RETURN_TYPES: dict[str, tuple[str, list[str]]] = {
    "pl.archetypal_space": ("Figure", ['plotly.graph_objects.Figure', 'Interactive 3D scatter plot containing:', '- Cell points colored by color_by (with colorbar if continuous)', '- Archetype positions as diamond markers', '- Archetype labels (if show_archetype_labels=True)', '...']),
    "pl.archetypal_space_multi": ("Figure", ['plotly.graph_objects.Figure', 'Interactive 3D comparison plot']),
    "pl.archetype_correspondence": ("Figure", ['go.Figure']),
    "pl.archetype_positions": ("Any", ['matplotlib.figure.Figure', 'Figure with archetype position visualizations']),
    "pl.archetype_positions_3d": ("Any", ['matplotlib.figure.Figure', '3D visualization of archetypes']),
    "pl.archetype_radar": ("Figure", ['go.Figure', 'Radar plot figure.']),
    "pl.archetype_regression_dotplot": ("Figure", ['go.Figure']),
    "pl.archetype_statistics": ("dict", ['dict', 'Statistics dictionary with keys:', '- n_archetypes : int - Number of archetypes', '- n_dimensions : int - Embedding dimensions', '- mean_distance : float - Mean pairwise Euclidean distance', '...']),
    "pl.co_occurrence": ("Figure", ['plotly.graph_objects.Figure']),
    "pl.coefficient_heatmap": ("Figure", ['go.Figure']),
    "pl.component_archetype_summary": ("Figure", ['go.Figure']),
    "pl.component_heatmap": ("Figure", ['go.Figure']),
    "pl.component_neighborhood_graph": ("Figure", ['go.Figure', 'Interactive 2D plotly figure.']),
    "pl.component_scatter": ("Figure", ['go.Figure']),
    "pl.component_stability": ("Figure", ['go.Figure']),
    "pl.contrast_volcano": ("Figure", ['go.Figure']),
    "pl.contrast_volcano_grid": ("Figure", ['go.Figure']),
    "pl.cross_correlations": ("Figure", ['plotly.graph_objects.Figure']),
    "pl.density_comparison": ("Figure", ['go.Figure']),
    "pl.dotplot": ("Figure", ['matplotlib.figure.Figure', 'Dotplot figure with:', '- X-axis: Groups (archetypes/patterns)', '- Y-axis: Features (genes/pathways), sorted by effect size', '- Dot size: Effect magnitude (with legend)', '...']),
    "pl.elbow_curve": ("Figure", ['plotly.graph_objects.Figure', 'Interactive elbow curve plot']),
    "pl.fate_probabilities": ("unspecified", ['None', "Displays matplotlib figure. CellRank's plotting functions", 'display directly rather than returning figure objects.']),
    "pl.feature_similarity_heatmap": ("Figure", ['go.Figure']),
    "pl.flow_magnitude": ("Figure", ['go.Figure']),
    "pl.flow_topo_landscape": ("Figure", ['matplotlib.figure.Figure', 'The topographic landscape figure.']),
    "pl.gene_alignment_barplot": ("Figure", ['go.Figure']),
    "pl.gmm_bic_curve": ("Figure", ['go.Figure']),
    "pl.interaction_boundaries": ("Figure", ['plotly.graph_objects.Figure']),
    "pl.interaction_heatmap": ("Figure", ['go.Figure']),
    "pl.jacobian_heatmap": ("Figure", ['go.Figure']),
    "pl.lineage_drivers": ("unspecified", ['fig : matplotlib.figure.Figure', 'Figure object']),
    "pl.mmd_heatmap": ("Figure", ['go.Figure']),
    "pl.nhood_enrichment": ("Figure", ['plotly.graph_objects.Figure']),
    "pl.pattern_dotplot": ("Figure", ['matplotlib.figure.Figure', 'Dotplot figure with:', '- X-axis: Pattern codes or archetype names', '- Y-axis: Feature names (genes/pathways)', '- Dot size: Effect size magnitude', '...']),
    "pl.pattern_heatmap": ("Figure", ['plt.Figure', 'The heatmap figure']),
    "pl.pattern_summary": ("Figure", ['go.Figure']),
    "pl.pattern_summary_barplot": ("Figure", ['plt.Figure', 'The summary barplot figure']),
    "pl.r2_barplot": ("Figure", ['go.Figure']),
    "pl.regression_volcano": ("Figure", ['go.Figure']),
    "pl.soft_assignment_flow": ("Figure", ['go.Figure', 'Plotly Sankey figure.']),
    "pl.soft_assignment_heatmap": ("Figure", ['go.Figure']),
    "pl.spatial_archetypes": ("Figure", ['plotly.graph_objects.Figure']),
    "pl.spatial_autocorr": ("Figure", ['plotly.graph_objects.Figure']),
    "pl.ternary_facet": ("Figure", ['plotly.graph_objects.Figure', 'The ternary scatter figure.']),
    "pl.ternary_facet_grid": ("Figure", ['list of plotly.graph_objects.Figure', 'One figure per archetype triple.']),
    "pl.training_metrics": ("Figure", ['plotly.graph_objects.Figure or None', 'Interactive training metrics plot with 3-row layout:', '- Row 1 (40%): Loss metrics (loss, archetypal_loss, KLD, rmse)', '- Row 2 (30%): Stability metrics (vertex_stability_latent/pca)', '- Row 3 (30%): Convergence (loss_delta with rolling mean)', '...']),
    "pl.trajectory_ribbon": ("Figure", ['go.Figure']),
    "pl.velocity_quiver": ("Figure", ['go.Figure']),
    "pl.vertex_radar": ("Figure", ['go.Figure']),
    "pp.compute_pathway_scores": ("None", []),
    "pp.generate_synthetic": ("AnnData", ['AnnData', 'Synthetic data with ground truth archetypes in .uns']),
    "pp.load_data": ("AnnData", ['AnnData', 'Loaded data. Use sc.pp.pca(adata) to add PCA coordinates.']),
    "pp.load_pathway_networks": ("unspecified", ['pd.DataFrame', "Pathway network with 'source', 'target', 'pathway' columns"]),
    "pp.prepare_atacseq": ("None", ['None', 'Modifies ``adata`` in place:', '- ``adata.obsm[store_key]``: LSI embeddings [n_cells, n_components]', "- ``adata.uns['lsi']``: dict with 'variance_ratio' and 'components'"]),
    "pp.prepare_training": ("DataLoader", ['DataLoader', 'PyTorch DataLoader optimized for the execution environment']),
    "tl.archetypal_coordinates": ("dict", ['dict', 'Dictionary with archetypal coordinates and distances']),
    "tl.archetype_co_occurrence": ("dict", ['dict', "Dictionary with 'occ' (co-occurrence ratios) and 'interval' (distance bins).", "Also stored in ``adata.uns['archetype_co_occurrence']``."]),
    "tl.archetype_contrasts": ("dict", ['dict', 'Serialized ArchetypeContrastsResult. Also stored in', "adata.uns['peach_archetype_contrasts_{feature_type}']."]),
    "tl.archetype_driver_regression": ("dict", ['dict', "Serialized DriverRegressionResult. Also stored in adata.uns['peach_driver_regression']."]),
    "tl.archetype_exclusive_patterns": ("DataFrame", ['pd.DataFrame', 'Results with columns:', '- ``pathway``/``gene`` : Feature identifier', '- ``archetype`` : Exclusive archetype', '- ``mean_archetype`` : Mean in exclusive archetype', '...']),
    "tl.archetype_feature_similarity": ("dict", ['dict', "Serialized ArchetypeFeatureSimilarityResult. Also stored in adata.uns['peach_archetype_feature_similarity']."]),
    "tl.archetype_interaction_boundaries": ("dict", ['dict', 'Dictionary with:', "- ``'boundary_scores'``: np.ndarray [n_cells] — per-cell JSD boundary score", "- ``'mean_weights_a'``: np.ndarray [n_cells, n_archetypes] — mean weight", 'vector of type-A neighbors per cell', '...']),
    "tl.archetype_mmd": ("dict", ['dict', "Serialized ArchetypeMMDResult. Also stored in adata.uns['peach_archetype_mmd']."]),
    "tl.archetype_nhood_enrichment": ("dict", ['dict', "Dictionary with 'zscore' and 'count' arrays [n_archetypes x n_archetypes].", "Also stored in ``adata.uns['archetype_nhood_enrichment']``."]),
    "tl.archetype_pair_enrichment": ("dict", ['dict', "Per-pair results: {(i,j): {'enrichment_score': float, 'p_value': float,", "'n_cells_i': int, 'n_cells_j': int}}", "Also stored in adata.uns['peach_pair_enrichment']."]),
    "tl.archetype_spatial_autocorr": ("unspecified", ['pandas.DataFrame', 'DataFrame with autocorrelation statistics per archetype weight.', "Also stored in ``adata.uns['archetype_spatial_autocorr']``."]),
    "tl.archetype_summary": ("dict", ['dict (single archetype) or list[dict] (all archetypes)', 'Per-archetype structured summary.']),
    "tl.assign_archetypes": ("None", []),
    "tl.assign_to_centroids": ("None", ['None', 'Modifies adata.obs[obs_key] with Categorical assignments.', "Values are condition levels (e.g., 'chemo_naive', 'IDS') or 'unassigned'."]),
    "tl.classify_feature_patterns": ("dict", ['dict', 'Keys: feature_names, n_features, classifications, pattern_counts.', "Also stored in adata.uns['peach_feature_patterns']."]),
    "tl.component_regression": ("dict", ['dict with keys:', 'component_regs : dict[int, dict]', 'Per-component simplex regression results. Keys are component', 'indices (0..n_stable-1), values are regression result dicts.', 'n_components : int', '...']),
    "tl.compute_conditional_centroids": ("unspecified", ['dict', 'Dictionary with keys:', '- ``condition_column`` : str - name of the condition column', '- ``n_levels`` : int - number of unique levels', '- ``levels`` : List[str] - list of level names', '...']),
    "tl.compute_lineage_drivers": ("unspecified", ['drivers : pd.DataFrame', 'Top driver genes with statistics:', "- 'gene' : Gene name", "- 'lineage' : Target lineage name", "- 'correlation' : Spearman correlation with fate probability", '...']),
    "tl.compute_lineage_pseudotimes": ("unspecified", ['None', 'Stores pseudotime variables in adata.obs with keys:', "'pseudotime_to_{lineage}' for each lineage"]),
    "tl.compute_transition_frequencies": ("unspecified", ['pd.DataFrame', 'Transition frequency matrix with shape [n_archetypes, n_archetypes].', '- Index: Source archetypes (starting weight)', '- Columns: Target archetypes (fate probability)', '- Values: Integer counts of cells satisfying both thresholds', '...']),
    "tl.conditional_associations": ("DataFrame", ['pd.DataFrame', 'Results with columns:', '- ``archetype`` : str - Archetype identifier', '- ``condition`` : str - Condition value from obs_column', '- ``observed`` : int - Observed count in overlap', '...']),
    "tl.extract_archetype_weights": ("np.ndarray", ['np.ndarray', 'Cell archetype weights of shape (n_cells, n_archetypes)', 'Also stores weights in adata.obsm[weights_key]']),
    "tl.feature_simplex_decomposition": ("dict", ['dict', "Plain dict with mixture model results. Stored in adata.uns['peach_gmm'],", "labels in adata.obsm['peach_gmm_labels']."]),
    "tl.feature_simplex_regression": ("dict", ['dict', 'Serialized SimplexRegressionResult. Stored in namespaced key:', "adata.uns['peach_simplex_regression_genes'] (when feature_matrix=None),", "adata.uns['peach_simplex_regression_pathways'] (when feature_matrix='pathway_scores'),", "or adata.uns['peach_simplex_regression_{feature_matrix}'] for other obsm keys.", '...']),
    "tl.flow_between": ("dict", []),
    "tl.flow_bifurcation": ("dict", ['dict', 'Keys:', '- ``divergence``: ``[n_timepoints, n_cells]`` — trace of Jacobian', '- ``bifurcation_score``: ``[n_cells]`` — max |divergence| along trajectory', '- ``eigenvalue_real``: ``[n_timepoints, n_cells, dim]``', '...']),
    "tl.flow_feature_graph": ("dict", ['dict', 'Keys: ``adjacency_matrix``, ``gene_names``, ``gene_indices``,', '``out_centrality``, ``in_centrality``, ``flow_centrality``,', '``top_hub_genes``, ``n_timepoints``, ``n_top_genes``,', '``edge_threshold``, ``per_timepoint_jacobians``.']),
    "tl.flow_gene_alignment": ("dict", []),
    "tl.flow_jacobian": ("dict", []),
    "tl.flow_significance": ("dict", ['dict with p_value, observed_stat, null_distribution']),
    "tl.flow_temporal_feature_graph": ("dict", ['dict', 'Keys: ``cross_matrices``, ``self_expansion``, ``gene_names``,', '``timepoints``, ``temporal_centrality``, ``temporal_profile``,', '``top_early_genes``, ``top_mid_early_genes``,', '``top_mid_late_genes``, ``top_late_genes``,', '...']),
    "tl.flow_within": ("dict", []),
    "tl.gene_associations": ("DataFrame", ['pd.DataFrame', 'Results with columns:', '- ``gene`` : str - Gene symbol/identifier', '- ``archetype`` : str - Archetype identifier', '- ``n_archetype_cells`` : int - Cells in archetype', '...']),
    "tl.gene_simplex_regression": ("dict", []),
    "tl.hyperparameter_search": ("CVSummary", ['CVSummary', 'Complete cross-validation results with analysis methods:', '**Attributes:**', '- ``config_results`` : dict[str, CVResults] - Per-configuration results', '- ``summary_df`` : pd.DataFrame - Summary table', '...']),
    "tl.pathway_associations": ("DataFrame", ['pd.DataFrame', 'Results with columns:', '- ``pathway`` : str - Pathway name', '- ``archetype`` : str - Archetype identifier', '- ``n_archetype_cells`` : int - Cells in archetype', '...']),
    "tl.pathway_simplex_regression": ("dict", []),
    "tl.pattern_analysis": ("dict", ['dict[str, pd.DataFrame]', 'Dictionary with keys:', "- ``'individual'`` : Individual archetype results", "- ``'patterns'`` : Pattern-based test results", "- ``'exclusivity'`` : Mutual exclusivity results"]),
    "tl.setup_cellrank": ("unspecified", ['ck : cellrank.kernels.ConnectivityKernel', 'Computed transition kernel', 'g : cellrank.estimators.GPCCA', 'GPCCA estimator with fate probabilities', 'Stores in adata', '...']),
    "tl.single_trajectory_analysis": ("unspecified", ['Tuple[SingleTrajectoryResult, AnnData]', '- result : SingleTrajectoryResult with trajectory metadata', '- adata_traj : Subset AnnData containing only trajectory cells, ready for', 'CellRank gene trends. If trajectories list provided, returns list of tuples.', 'Stores in adata', '...']),
    "tl.spatial_neighbors": ("None", ['None', 'Modifies ``adata`` in place:', "- ``adata.obsp['spatial_connectivities']``: sparse connectivity matrix", "- ``adata.obsp['spatial_distances']``: sparse distance matrix"]),
    "tl.specialization_patterns": ("DataFrame", ['pd.DataFrame', 'Results showing specialization from archetype_0.']),
    "tl.tradeoff_patterns": ("DataFrame", ['pd.DataFrame', 'Results with tradeoff patterns:', '- ``pattern_code`` : Visual pattern code', '- ``high_archetypes``, ``low_archetypes`` : Groups', '- ``mean_high``, ``mean_low`` : Group means', '...']),
    "tl.train_archetypal": ("dict", ['dict', 'Training results dictionary with the following structure:', '**Guaranteed keys (always present):**', '- ``history`` : dict', 'Training metrics per epoch. Keys depend on tracking options:', '...']),
}


def get_return_type(func_name: str) -> tuple[str, list[str]]:
    """Get return type and key fields for a PEACH function.

    Parameters
    ----------
    func_name : str
        Function name, e.g. 'tl.train_archetypal'

    Returns
    -------
    tuple of (type_name, key_fields)
    """
    if func_name in RETURN_TYPES:
        return RETURN_TYPES[func_name]
    raise KeyError(f"No return type for '{func_name}'. Available: {list(RETURN_TYPES.keys())}")


# adata storage keys extracted from docstrings
ADATA_KEYS = {
    "obsm": {
        "X_pca": "Set by tl.setup_cellrank",
        "X_umap": "Set by tl.setup_cellrank",
        "archetype_distances": "Set by tl.setup_cellrank",
        "cell_archetype_weights": "Set by tl.single_trajectory_analysis",
        "fate_probabilities": "Set by tl.single_trajectory_analysis",
        "pathway_scores": "Set by tl.pathway_simplex_regression",
        "peach_gmm_labels": "Set by tl.feature_simplex_decomposition",
        "peach_residuals": "Set by tl.feature_simplex_regression",
    },
    "obs": {
        "archetypes": "Set by tl.single_trajectory_analysis",
        "boundary_score": "Set by tl.archetype_interaction_boundaries",
        "pseudotime_to_{archetype}": "Set by tl.single_trajectory_analysis",
        "terminal_states": "Set by tl.setup_cellrank",
        "trajectory_{src}_to_{tgt}_cells": "Set by tl.single_trajectory_analysis",
    },
    "uns": {
        "archetype_co_occurrence": "Set by tl.archetype_co_occurrence",
        "archetype_interaction_boundaries": "Set by tl.archetype_interaction_boundaries",
        "archetype_nhood_enrichment": "Set by tl.archetype_nhood_enrichment",
        "archetype_spatial_autocorr": "Set by tl.archetype_spatial_autocorr",
        "cellrank_gpcca": "Set by tl.setup_cellrank",
        "lineage_names": "Set by tl.single_trajectory_analysis",
        "lsi": "Set by pp.prepare_atacseq",
        "neighbors": "Set by tl.setup_cellrank",
        "paga": "Set by tl.setup_cellrank",
        "peach_archetype_contrasts_{feature_type}": "Set by tl.archetype_contrasts",
        "peach_archetype_feature_similarity": "Set by tl.archetype_feature_similarity",
        "peach_archetype_mmd": "Set by tl.archetype_mmd",
        "peach_driver_regression": "Set by tl.archetype_driver_regression",
        "peach_feature_patterns": "Set by tl.classify_feature_patterns",
        "peach_gmm": "Set by tl.feature_simplex_decomposition",
        "peach_pair_enrichment": "Set by tl.archetype_pair_enrichment",
        "peach_simplex_regression": "Set by tl.feature_simplex_regression",
        "peach_simplex_regression_genes": "Set by tl.feature_simplex_regression",
        "peach_simplex_regression_pathways": "Set by tl.feature_simplex_regression",
        "peach_simplex_regression_{feature_matrix}": "Set by tl.feature_simplex_regression",
        "trained_model": "Set by tl.extract_archetype_weights",
        "trajectory_{src}_to_{tgt}": "Set by tl.single_trajectory_analysis",
    },
    "obsp": {
        "spatial_connectivities": "Set by tl.spatial_neighbors",
        "spatial_distances": "Set by tl.spatial_neighbors",
    },
}

# Optional fields that may not be present in return dicts.
# Always use .get() for these.
USE_GET_FOR = {
    "alignment_pvalues",
    "alignment_pvalues_fdr",
    "bootstrap_ci_lower",
    "bootstrap_ci_upper",
    "component_feature_profiles",
    "convergence_epoch",
    "degree_comparison",
    "expansion_pvalues",
    "expansion_pvalues_fdr",
    "final_archetype_r2",
    "final_model",
    "holdout_fraction",
    "holdout_mmd",
    "interaction_coefficients",
    "interaction_pairs",
    "interaction_pvalues",
    "interaction_pvalues_fdr",
    "model",
    "per_cell_alignment",
    "per_cell_expansion",
    "per_cell_expansion_gene_names",
    "per_cell_gene_names",
    "residuals",
    "vertex_covariance",
}

PITFALLS = {
    "final_archetype_r2": "Optional in TrainingResults - use .get(). Train-mode R2 includes reparameterization noise; eval-mode is the real metric.",
    "X_pca": "Check for X_pca, X_PCA, or pca variants in adata.obsm.",
    "CVSummary_ranking": "Use ranked[i]['metric_value'] NOT ranked[i].mean_archetype_r2.",
    "distance_vs_weight": "Distance-based and weight-based archetype assignment disagree for ~60% of cells - this is expected.",
    "FRGeom_returns": "torch.Tensor, NOT numpy arrays.",
}


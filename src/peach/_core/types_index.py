# src/peach/_core/types_index.py
"""
PEACH Type Index - Compact function→type mapping for programmatic access.

=============================================================================
READ THIS FILE FIRST before writing ANY PEACH code.
Only read full types.py definitions on-demand for specific types.
=============================================================================

This file provides:
1. FUNCTION_RETURNS: function → (return_type, required_keys)
2. ADATA_KEYS: What PEACH stores in AnnData
3. USE_GET_FOR: Keys that require .get() access
4. DATAFRAME_SCHEMAS: DataFrame column specifications

For full Pydantic type definitions: grep "class TypeName" types.py
For full docstrings: read the specific module function

Version: 0.4.0
"""


# =============================================================================
# FUNCTION → RETURN TYPE MAPPING
# =============================================================================
# Format: "module.function": (ReturnType, [required_keys_or_description])
# ReturnType options: Pydantic class name, "DataFrame", "Figure", "None", etc.

FUNCTION_RETURNS: dict[str, tuple[str, list[str]]] = {
    # =========================================================================
    # pp (PREPROCESSING) - 5 functions
    # =========================================================================
    "pp.load_data": ("AnnData", ["adata with .X, .obs, .var loaded"]),
    "pp.generate_synthetic": (
        "AnnData",
        [
            "adata.X: synthetic expression",
            "adata.obsm['X_pca']: PCA coords",
            "adata.uns['true_archetypes']: ground truth positions",
        ],
    ),
    "pp.prepare_training": (
        "Tuple[DataLoader, AnnData]",
        ["dataloader: PyTorch DataLoader", "adata: with .obsm['X_pca'] ensured"],
    ),
    "pp.load_pathway_networks": ("Dict[str, Set[str]]", ["pathway_name → gene_set"]),
    "pp.compute_pathway_scores": ("AnnData", ["adata.obsm['pathway_scores'] added"]),
    "pp.prepare_atacseq": (
        "None",
        [
            "adata.obsm['X_lsi']: LSI embeddings [n_cells, n_components]",
            "adata.uns['lsi']: Dict with 'variance_ratio' and 'components'",
        ],
    ),
    # =========================================================================
    # tl (TOOLS) - 19 functions
    # =========================================================================
    # --- Training ---
    "tl.train_archetypal": (
        "TrainingResults",
        [
            # GUARANTEED (always present)
            "history",  # Dict[str, List[float]] - per-epoch metrics
            "final_model",  # Deep_AA torch.nn.Module
            "model",  # Alias for final_model
            "final_optimizer",  # torch.optim.Optimizer
            "final_analysis",  # Dict with constraint validation
            "epoch_archetype_positions",  # List[Tensor]
            "training_config",  # Dict with n_epochs, early_stop_triggered, etc.
            # CONVENIENCE (use .get() - may be None)
            # final_archetype_r2, final_rmse, final_mae, final_loss, convergence_epoch
        ],
    ),
    "tl.hyperparameter_search": (
        "CVSummary",
        [
            "summary_df",  # DataFrame with all configs + metrics
            "config_results",  # List[CVResults] per config
            "best_config",  # CVHyperparameters
            "cv_info",  # Dict with n_configurations, cv_folds, etc.
            # Methods: .rank_by_metric(metric) → List[RankedConfig]
            # Access: ranked[0].metric_value, ranked[0].hyperparameters.n_archetypes
        ],
    ),
    # --- Coordinates & Assignment ---
    "tl.archetypal_coordinates": (
        "DataFrame",
        [
            "archetype_0_distance",
            "archetype_1_distance",
            "...",  # Per-archetype distances
            "nearest_archetype",  # str: 'archetype_0', etc.
            "nearest_archetype_distance",  # float
            # Also stores in adata.obsm['archetype_distances']
        ],
    ),
    "tl.assign_archetypes": (
        "None",
        [
            "MODIFIES: adata.obs['archetypes']",  # Categorical
            # Values: 'archetype_0', 'archetype_1', ..., 'no_archetype'
        ],
    ),
    "tl.extract_archetype_weights": (
        "np.ndarray",
        ["shape: (n_cells, n_archetypes)", "Also stores in adata.obsm['cell_archetype_weights']"],
    ),
    # --- Statistical Testing ---
    "tl.gene_associations": (
        "DataFrame[GeneAssociationResult]",
        [
            "gene",
            "archetype",
            "n_archetype_cells",
            "n_other_cells",
            "mean_archetype",
            "mean_other",
            "log_fold_change",
            "statistic",
            "pvalue",
            "direction",  # 'higher' or 'lower'
            "fdr_pvalue",
            "significant",  # After FDR correction
        ],
    ),
    "tl.pathway_associations": (
        "DataFrame[PathwayAssociationResult]",
        [
            "pathway",
            "archetype",
            "n_archetype_cells",
            "n_other_cells",
            "mean_archetype",
            "mean_other",
            "mean_diff",  # Primary effect size
            "log_fold_change",  # Alias for mean_diff
            "statistic",
            "pvalue",
            "direction",
            "fdr_pvalue",
            "significant",
        ],
    ),
    "tl.conditional_associations": (
        "DataFrame[ConditionalAssociationResult]",
        [
            "archetype",
            "condition",
            "observed",
            "expected",
            "total_archetype",
            "total_condition",
            "odds_ratio",
            "ci_lower",
            "ci_upper",  # >1 = enriched
            "pvalue",
            "fdr_pvalue",
            "significant",
        ],
    ),
    "tl.pattern_analysis": (
        "DataFrame[PatternAssociationResult]",
        [
            "pathway|gene",
            "pattern_name",
            "pattern_code",
            "pattern_type",
            "high_archetypes",
            "low_archetypes",
            "n_high_cells",
            "n_low_cells",
            "mean_high",
            "mean_low",
            "log_fold_change",
            "primary_effect_size",
            "statistic",
            "pvalue",
            "direction",
            "fdr_pvalue",
            "significant",
        ],
    ),
    "tl.archetype_exclusive_patterns": (
        "DataFrame[PatternAssociationResult]",
        ["Same as tl.pattern_analysis, pattern_type='exclusive'"],
    ),
    "tl.specialization_patterns": (
        "DataFrame[PatternAssociationResult]",
        ["Same as tl.pattern_analysis, pattern_type='specialization'"],
    ),
    "tl.tradeoff_patterns": (
        "DataFrame[PatternAssociationResult]",
        ["Same as tl.pattern_analysis, pattern_type='tradeoff'"],
    ),
    # --- Conditional Centroids ---
    "tl.compute_conditional_centroids": (
        "ConditionalCentroidResult",
        [
            "condition_column",  # str - name of condition column
            "n_levels",  # int - number of unique levels
            "levels",  # List[str] - level names
            "centroids",  # Dict[str, List[float]] - level → full PCA coords
            "centroids_3d",  # Dict[str, List[float]] - level → [x, y, z] first 3 PCs
            "cell_counts",  # Dict[str, int] - level → cell count
            "pca_key",  # str - PCA key used
            "exclude_archetypes",  # List[str] - archetypes excluded
            "groupby",  # Optional[str] - groupby column if used
            "group_centroids",  # Optional[Dict] - if groupby: {group: {level: coords}}
            "group_centroids_3d",  # Optional[Dict] - if groupby: {group: {level: [x,y,z]}}
            "group_cell_counts",  # Optional[Dict] - if groupby: {group: {level: count}}
            # MODIFIES: adata.uns['conditional_centroids'][condition_column]
        ],
    ),
    "tl.assign_to_centroids": (
        "None",
        [
            "MODIFIES: adata.obs[obs_key]",  # Categorical: condition levels + 'unassigned'
            # Assigns top bin_prop% closest cells to each centroid
            # Mirrors assign_archetypes but for condition-based centroids
            # Enables using centroids as trajectory endpoints in single_trajectory_analysis
        ],
    ),
    # --- CellRank Integration ---
    "tl.setup_cellrank": (
        "VelocityKernel",
        ["CellRank VelocityKernel object", "MODIFIES: adata.obsp['T_forward'] transition matrix"],
    ),
    "tl.compute_lineage_pseudotimes": (
        "Dict[str, np.ndarray]",
        ["lineage_name → pseudotime array", "MODIFIES: adata.obs['dpt_pseudotime'], adata.obs['{lineage}_pseudotime']"],
    ),
    "tl.compute_lineage_drivers": ("DataFrame", ["gene", "lineage", "correlation", "pvalue", "qvalue"]),
    "tl.compute_transition_frequencies": (
        "DataFrame",
        ["source_archetype", "target_archetype", "frequency", "normalized_freq"],
    ),
    # --- Single Trajectory Analysis ---
    # Returns: Tuple[SingleTrajectoryResult, AnnData] - (result, adata_traj subset)
    "tl.single_trajectory_analysis": (
        "Tuple[SingleTrajectoryResult, AnnData]",
        [
            "source_archetype",  # str - e.g., 'archetype_0'
            "target_archetype",  # str - e.g., 'archetype_3'
            "source_idx",  # int - source archetype index
            "target_idx",  # int - target archetype index
            "trajectory_key",  # str - key in adata.uns (e.g., 'trajectory_0_to_3')
            "n_trajectory_cells",  # int - number of cells in trajectory
            "pseudotime_key",  # str - CellRank pseudotime key (e.g., 'pseudotime_to_archetype_3')
            "cell_mask_key",  # str - key in adata.obs for boolean mask
            "selection_method",  # str - 'discrete', 'weight', or 'both'
            "n_discrete_cells",  # Optional[int] - cell count with discrete selection
            "n_weight_cells",  # Optional[int] - cell count with weight selection
            # MODIFIES: adata.obs[trajectory_key + '_cells'], adata.uns[trajectory_key]
            # RETURNS: adata_traj subset ready for cr.pl.gene_trends()
        ],
    ),
    # --- Spatial Analysis ---
    "tl.spatial_neighbors": (
        "None",
        [
            "adata.obsp['spatial_connectivities']: sparse connectivity matrix",
            "adata.obsp['spatial_distances']: sparse distance matrix",
        ],
    ),
    "tl.archetype_nhood_enrichment": (
        "Dict",
        [
            "'zscore': ndarray [n_archetypes, n_archetypes] enrichment z-scores",
            "'count': ndarray [n_archetypes, n_archetypes] neighbor counts",
            "ALSO: adata.uns['archetype_nhood_enrichment']",
        ],
    ),
    "tl.archetype_co_occurrence": (
        "Dict",
        [
            "'occ': ndarray [n_archetypes, n_archetypes, n_steps] co-occurrence ratios",
            "'interval': ndarray distance bin edges",
            "ALSO: adata.uns['archetype_co_occurrence']",
        ],
    ),
    "tl.archetype_spatial_autocorr": (
        "DataFrame",
        [
            "Index: archetype_0, archetype_1, ...",
            "Columns: I (or C), pval_norm, var_norm, pval_z_sim (if n_perms > 0)",
            "ALSO: adata.uns['archetype_spatial_autocorr']",
        ],
    ),
    "tl.archetype_interaction_boundaries": (
        "Dict",
        [
            "'boundary_scores': ndarray [n_cells] JSD between mean weight vectors of cell type neighbors",
            "'mean_weights_a': ndarray [n_cells, n_arch] mean archetype weight of type-A neighbors",
            "'mean_weights_b': ndarray [n_cells, n_arch] mean archetype weight of type-B neighbors",
            "'cross_correlations': DataFrame with per-archetype Spearman r between cell types",
            "ALSO: adata.obs['boundary_score'], adata.uns['archetype_interaction_boundaries']",
        ],
    ),
    # =========================================================================
    # pl (PLOTTING) - 17 functions
    # =========================================================================
    # All return matplotlib Figure or Axes (unless show=True returns None)
    "pl.archetypal_space": ("Figure|Axes", ["2D simplex projection with cells colored"]),
    "pl.archetypal_space_multi": ("Figure", ["Multi-panel simplex by condition"]),
    "pl.training_metrics": ("Figure", ["Loss curves, R², constraint satisfaction"]),
    "pl.elbow_curve": ("Figure", ["n_archetypes vs metric for selection"]),
    "pl.dotplot": ("Figure", ["Gene/pathway expression dotplot by archetype"]),
    "pl.archetype_positions": ("Figure", ["2D PCA with archetype positions marked"]),
    "pl.archetype_positions_3d": ("Figure", ["3D PCA with archetypes"]),
    "pl.archetype_statistics": ("Figure", ["Archetype usage, distances, weights"]),
    "pl.pattern_dotplot": ("Figure", ["Pattern results as dotplot"]),
    "pl.pattern_summary_barplot": ("Figure", ["Summary of significant patterns"]),
    "pl.pattern_heatmap": ("Figure", ["Pattern effect sizes as heatmap"]),
    "pl.fate_probabilities": ("Figure", ["CellRank fate probability UMAP"]),
    # Note: gene_trends removed - use cellrank.pl.gene_trends() directly
    "pl.lineage_drivers": ("Figure", ["Top driver genes per lineage"]),
    # --- Spatial Plots ---
    "pl.nhood_enrichment": ("Figure", ["Heatmap of archetype neighborhood enrichment z-scores"]),
    "pl.co_occurrence": ("Figure", ["Line plot of distance-dependent archetype co-occurrence"]),
    "pl.spatial_archetypes": ("Figure", ["Scatter of cells on spatial coords colored by archetype"]),
    "pl.interaction_boundaries": ("Figure", ["Spatial map of boundary scores between cell types"]),
    "pl.spatial_autocorr": ("Figure", ["Lollipop chart of Moran's I / Geary's C per archetype"]),
    "pl.cross_correlations": ("Figure", ["Diverging dot plot of per-archetype Spearman r between cell types"]),
    # =========================================================================
    # _core (INTERNAL) - Key functions
    # =========================================================================
    "_core.train_vae": (
        "Tuple[CoreTrainingResults, Module]",
        ["results_dict: Same keys as TrainingResults", "model: trained Deep_AA Module"],
    ),
    "_core.calculate_archetype_r2": (
        "float|Tensor",
        ["R² value (1.0 = perfect, 0.0 = mean baseline, <0 = worse than mean)"],
    ),
    "_core.get_archetypal_coordinates": (
        "ArchetypalCoordinates",
        [
            "A",  # [batch, n_archetypes] - cell weights (rows sum to 1)
            "B",  # [batch, n_archetypes] - dummy matrix
            "Y",  # [n_archetypes, n_features] - archetype positions
            "mu",  # [batch, n_archetypes] - encoder means
            "log_var",  # [batch, n_archetypes] - encoder log variances
            "z",  # [batch, n_archetypes] - latent (same as A)
        ],
    ),
    "_core.extract_and_store_archetypal_coordinates": (
        "ExtractedCoordinates",
        [
            "archetype_positions",  # [n_archetypes, n_pcs]
            "cell_weights",  # [n_cells, n_archetypes]
            "cell_latent",  # [n_cells, n_archetypes]
            "cell_mu",
            "cell_log_var",
            "n_cells",
            "n_archetypes",
            "pca_key_used",
        ],
    ),
    "_core.compute_archetype_distances": ("DataFrame", ["Same as tl.archetypal_coordinates"]),
    "_core.bin_cells_by_archetype": ("None", ["MODIFIES: adata.obs['archetypes']"]),
    "_core.test_archetype_recovery": (
        "ArchetypeRecoveryMetrics",
        [
            "mean_distance",
            "max_distance",
            "normalized_mean_distance",
            "recovery_success",  # bool
            "individual_distances",  # np.ndarray
            "assignment",  # List[(learned_idx, true_idx)]
        ],
    ),
    "_core.generate_convex_data": (
        "Dict",
        [
            "data",  # [n_samples, n_features] synthetic data
            "archetypes",  # [n_archetypes, n_features] true archetypes
            "weights",  # [n_samples, n_archetypes] true mixing weights
            "labels",  # [n_samples] cluster assignments
        ],
    ),
    "_core.PCHA": (
        "PCHAResults",
        [
            "archetypes",  # [n_archetypes, n_features]
            "archetype_r2",  # float - explained variance
            "A",  # [n_cells, n_archetypes]
            "B",  # [n_archetypes, n_cells]
        ],
    ),
}


# =============================================================================
# FUNCTION PARAMETERS (Comprehensive - from inspect.signature)
# =============================================================================
# Format: "module.function": {"param": (type_hint, default)}
# REQUIRED = no default (must be provided)
# Generated from inspect.signature() on all 44 public functions

REQUIRED = "REQUIRED"  # Sentinel for required parameters

FUNCTION_PARAMS = {
    # =========================================================================
    # PP MODULE (6 functions)
    # =========================================================================
    "pp.compute_pathway_scores": {
        "adata": ("AnnData", REQUIRED),
        "net": ("dict|None", None),  # Pathway network dict
        "use_layer": ("str|None", None),
        "obsm_key": ("str", "pathway_scores"),
        "verbose": ("bool", True),
    },
    "pp.generate_synthetic": {
        "n_points": ("int", 1000),
        "n_dimensions": ("int", 50),
        "n_archetypes": ("int", 4),
        "noise": ("float", 0.1),
        "seed": ("int", 1205),
        "archetype_type": ("str", "random"),  # "random", "simplex"
        "scale": ("float", 20.0),
        "return_torch": ("bool", True),
    },
    "pp.load_data": {
        "path": ("str", REQUIRED),
        "use_raw": ("bool", True),
        "dim_reduction_key": ("str", "X_PCA"),
        "batch_size": ("int", 128),
    },
    "pp.load_pathway_networks": {
        "sources": ("list[str]", ["c5_bp"]),
        "organism": ("str", "human"),
        "geneset_repo": ("str", "msigdb"),
        "verbose": ("bool", True),
    },
    "pp.prepare_training": {
        "adata": ("AnnData", REQUIRED),
        "batch_size": ("int", 128),
        "shuffle": ("bool", True),
        "pca_key": ("str|None", None),
        "num_workers": ("int|str", "auto"),
        "pin_memory": ("bool|str", "auto"),
        "persistent_workers": ("bool|str", "auto"),
        "prefetch_factor": ("int", 2),
    },
    "pp.prepare_atacseq": {
        "adata": ("AnnData", REQUIRED),  # Peak count matrix in .X (sparse)
        "n_components": ("int", 50),  # 30-50 standard for scATAC-seq
        "drop_first": ("bool", True),  # First LSI component ≈ sequencing depth
        "log_tf": ("bool", True),  # log(1 + TF) variant
        "store_key": ("str", "X_lsi"),
        "random_state": ("int", 42),
    },
    # =========================================================================
    # TL MODULE (22 functions)
    # =========================================================================
    "tl.train_archetypal": {
        "adata": ("AnnData", REQUIRED),
        "n_archetypes": ("int", 5),
        "n_epochs": ("int", 50),
        "layer": ("str|None", None),
        "pca_key": ("str", "X_pca"),
        "hidden_dims": ("list[int]|None", None),  # e.g. [256, 128, 64], None uses default
        "inflation_factor": ("float", 1.5),  # PCHA inflation, 1.2-2.0 recommended
        "model_config": ("dict|None", None),  # Advanced: {archetypal_weight, kld_weight, ...}
        "optimizer_config": ("dict|None", None),  # {lr, weight_decay}
        "device": ("str", "cpu"),  # "cpu", "cuda", "mps"
        "save_path": ("str|None", None),
        "archetypal_weight": ("float|None", None),  # Default 0.9
        "kld_weight": ("float|None", None),  # Default 0.1 (regularizes encoder variance)
        "reconstruction_weight": ("float", 0.0),
        "vae_recon_weight": ("float", 0.0),
        "diversity_weight": ("float", 0.0),
        "activation_func": ("str", "relu"),
        "track_stability": ("bool", True),
        "validate_constraints": ("bool", True),
        "lr_factor": ("float", 0.1),
        "lr_patience": ("int", 10),
        "seed": ("int", 42),
        "constraint_tolerance": ("float", 0.001),
        "stability_history_size": ("int", 20),
        "store_coords_key": ("str", "archetype_coordinates"),
        "early_stopping": ("bool", False),
        "early_stopping_patience": ("int", 10),
        "early_stopping_metric": ("str", "archetype_r2"),
        "min_improvement": ("float", 0.0001),
        "validation_check_interval": ("int", 5),
        "validation_data_loader": ("DataLoader|None", None),
    },
    "tl.hyperparameter_search": {
        "adata": ("AnnData", REQUIRED),
        "n_archetypes_range": ("list[int]", [3, 4, 5, 6]),
        "cv_folds": ("int", 3),
        "max_epochs_cv": ("int", 15),
        "pca_key": ("str", "X_pca"),
        "device": ("str", "cpu"),
        "base_model_config": ("dict|None", None),
        # --- kwargs passed to SearchConfig ---
        "hidden_dims_options": ("list[list[int]]|None", None),  # [[128,64], [256,128,64]]
        "inflation_factor_range": ("list[float]|None", None),  # [1.5]
        "speed_preset": ("str", "fast"),  # "fast", "balanced", "thorough"
        "use_pcha_init": ("bool", True),
        "subsample_fraction": ("float", 0.5),
        "max_cells_cv": ("int", 5000),
        "random_state": ("int", 42),
        "early_stopping_patience": ("int", 5),
    },
    "tl.archetypal_coordinates": {
        "adata": ("AnnData", REQUIRED),
        "pca_key": ("str", "X_pca"),
        "archetype_coords_key": ("str", "archetype_coordinates"),
        "obsm_key": ("str", "archetype_distances"),
        "uns_prefix": ("str", "archetype"),
        "verbose": ("bool", True),
    },
    "tl.assign_archetypes": {
        "adata": ("AnnData", REQUIRED),
        "percentage_per_archetype": ("float", 0.1),
        "obsm_key": ("str", "archetype_distances"),
        "obs_key": ("str", "archetypes"),
        "include_central_archetype": ("bool", True),
        "verbose": ("bool", True),
    },
    "tl.extract_archetype_weights": {
        "adata": ("AnnData", REQUIRED),
        "model": ("Module|None", None),
        "pca_key": ("str", "X_pca"),
        "weights_key": ("str", "cell_archetype_weights"),
        "batch_size": ("int", 256),
        "device": ("str", "cpu"),  # "cpu", "cuda", "mps"
        "verbose": ("bool", True),
    },
    "tl.gene_associations": {
        "adata": ("AnnData", REQUIRED),
        "bin_prop": ("float", 0.1),
        "obsm_key": ("str", "archetype_distances"),
        "obs_key": ("str", "archetypes"),
        "use_layer": ("str|None", None),
        "test_method": ("str", "mannwhitneyu"),
        "fdr_method": ("str", "benjamini_hochberg"),
        "fdr_scope": ("str", "global"),  # "global", "per_archetype"
        "test_direction": ("str", "two-sided"),
        "min_logfc": ("float", 0.01),
        "min_cells": ("int", 10),
        "comparison_group": ("str", "all"),
        "verbose": ("bool", True),
    },
    "tl.pathway_associations": {
        "adata": ("AnnData", REQUIRED),
        "pathway_obsm_key": ("str", "pathway_scores"),
        "obsm_key": ("str", "archetype_distances"),
        "obs_key": ("str", "archetypes"),
        "test_method": ("str", "mannwhitneyu"),
        "fdr_method": ("str", "benjamini_hochberg"),
        "fdr_scope": ("str", "global"),
        "test_direction": ("str", "two-sided"),
        "min_logfc": ("float", 0.01),
        "min_cells": ("int", 10),
        "comparison_group": ("str", "all"),
        "verbose": ("bool", True),
    },
    "tl.pattern_analysis": {
        "adata": ("AnnData", REQUIRED),
        "data_obsm_key": ("str", "pathway_scores"),
        "obs_key": ("str", "archetypes"),
        "include_individual_tests": ("bool", True),  # → 'individual' key
        "include_pattern_tests": ("bool", True),  # → 'patterns' key
        "include_exclusivity_analysis": ("bool", True),  # → 'exclusivity' key
        "verbose": ("bool", True),
    },
    "tl.conditional_associations": {
        "adata": ("AnnData", REQUIRED),
        "obs_column": ("str", REQUIRED),
        "archetype_assignments": ("array|None", None),
        "obs_key": ("str", "archetypes"),
        "test_method": ("str", "hypergeometric"),
        "fdr_method": ("str", "benjamini_hochberg"),
        "min_cells": ("int", 5),
        "verbose": ("bool", True),
    },
    "tl.archetype_exclusive_patterns": {
        "adata": ("AnnData", REQUIRED),
        "data_obsm_key": ("str", "pathway_scores"),
        "obs_key": ("str", "archetypes"),
        "test_method": ("str", "mannwhitneyu"),
        "fdr_method": ("str", "benjamini_hochberg"),
        "fdr_scope": ("str", "global"),
        "min_effect_size": ("float", 0.05),
        "min_cells": ("int", 10),
        "use_pairwise": ("bool", True),
        "verbose": ("bool", True),
    },
    "tl.specialization_patterns": {
        "adata": ("AnnData", REQUIRED),
        "data_obsm_key": ("str", "pathway_scores"),
        "obs_key": ("str", "archetypes"),
        "test_method": ("str", "mannwhitneyu"),
        "fdr_method": ("str", "benjamini_hochberg"),
        "fdr_scope": ("str", "global"),
        "min_cells": ("int", 10),
        "verbose": ("bool", True),
    },
    "tl.tradeoff_patterns": {
        "adata": ("AnnData", REQUIRED),
        "data_obsm_key": ("str", "pathway_scores"),
        "obs_key": ("str", "archetypes"),
        "tradeoffs": ("str", "pairs"),  # "pairs", "patterns"
        "test_method": ("str", "mannwhitneyu"),
        "fdr_method": ("str", "benjamini_hochberg"),
        "fdr_scope": ("str", "global"),
        "min_cells": ("int", 10),
        "min_effect_size": ("float", 0.1),
        "verbose": ("bool", True),
    },
    "tl.compute_conditional_centroids": {
        "adata": ("AnnData", REQUIRED),
        "condition_column": ("str", REQUIRED),
        "pca_key": ("str", "X_pca"),
        "store_key": ("str", "conditional_centroids"),
        "exclude_archetypes": ("list|None", None),
        "groupby": ("str|None", None),
        "verbose": ("bool", True),
    },
    "tl.assign_to_centroids": {
        "adata": ("AnnData", REQUIRED),
        "condition_column": ("str", REQUIRED),
        "pca_key": ("str", "X_pca"),
        "centroid_key": ("str", "conditional_centroids"),
        "bin_prop": ("float", 0.15),
        "obs_key": ("str", "centroid_assignments"),
        "exclude_archetypes": ("list|None", None),
        "verbose": ("bool", True),
    },
    "tl.setup_cellrank": {
        "adata": ("AnnData", REQUIRED),
        "high_purity_threshold": ("float", 0.8),
        "n_neighbors": ("int", 30),
        "n_pcs": ("int", 11),
        "compute_paga": ("bool", True),
        "solver": ("str", "gmres"),
        "tol": ("float", 1e-6),
        "terminal_obs_key": ("str", "archetypes"),
        "verbose": ("bool", True),
    },
    "tl.compute_lineage_pseudotimes": {
        "adata": ("AnnData", REQUIRED),
        "lineage_names": ("list|None", None),
        "fate_prob_key": ("str", "fate_probabilities"),
    },
    "tl.compute_lineage_drivers": {
        "adata": ("AnnData", REQUIRED),
        "lineage": ("str", REQUIRED),
        "n_genes": ("int", 100),
        "method": ("str", "cellrank"),
    },
    "tl.compute_transition_frequencies": {
        "adata": ("AnnData", REQUIRED),
        "start_weight_threshold": ("float", 0.5),
        "fate_prob_threshold": ("float", 0.3),
        "lineages": ("list|None", None),
    },
    "tl.single_trajectory_analysis": {
        "adata": ("AnnData", REQUIRED),
        "trajectory": ("tuple", REQUIRED),  # (source, target)
        "trajectories": ("list|None", None),  # Multiple trajectories
        "selection_method": ("str", "discrete"),  # "discrete", "weight", "both"
        "source_weight_threshold": ("float", 0.4),
        "target_fate_threshold": ("float", 0.4),
        "verbose": ("bool", True),
    },
    # --- Spatial Analysis (requires squidpy) ---
    "tl.spatial_neighbors": {
        "adata": ("AnnData", REQUIRED),
        "spatial_key": ("str", "spatial"),  # Key in obsm for 2D coords
        "n_neighs": ("int", 30),
        "coord_type": ("str", "generic"),  # "generic" for Slide-seq/MERFISH, "grid" for Visium
    },
    "tl.archetype_nhood_enrichment": {
        "adata": ("AnnData", REQUIRED),
        "cluster_key": ("str", "archetypes"),
        "n_perms": ("int", 1000),
        "seed": ("int", 42),
    },
    "tl.archetype_co_occurrence": {
        "adata": ("AnnData", REQUIRED),
        "cluster_key": ("str", "archetypes"),
        "spatial_key": ("str", "spatial"),
        "interval": ("int", 50),  # Number of distance intervals
    },
    "tl.archetype_spatial_autocorr": {
        "adata": ("AnnData", REQUIRED),
        "weights_key": ("str", "cell_archetype_weights"),  # Key in obsm with weight matrix
        "mode": ("str", "moran"),  # "moran" or "geary"
        "n_perms": ("int", 100),
        "n_jobs": ("int", 1),  # Default 1 for macOS
    },
    "tl.archetype_interaction_boundaries": {
        "adata": ("AnnData", REQUIRED),
        "cell_type_col": ("str", "Cell_Type"),
        "weights_key": ("str", "cell_archetype_weights"),  # Continuous weights in obsm
        "cell_type_a": ("str|None", None),  # Auto-picks top 2 if None
        "cell_type_b": ("str|None", None),
    },
    # =========================================================================
    # PL MODULE (16 functions)
    # =========================================================================
    "pl.archetypal_space": {
        "adata": ("AnnData", REQUIRED),
        "archetype_coords_key": ("str", "archetype_coordinates"),
        "pca_key": ("str", "X_pca"),
        "color_by": ("str|None", None),
        "use_layer": ("str", "logcounts"),
        "cell_size": ("float", 2.0),
        "cell_opacity": ("float", 0.6),
        "archetype_size": ("float", 8.0),
        "archetype_color": ("str", "red"),
        "show_archetype_labels": ("bool", True),
        "show_connections": ("bool", True),
        "color_scale": ("str", "viridis"),
        "categorical_colors": ("dict|None", None),
        "title": ("str", "Archetypal Space Visualization"),
        "auto_scale": ("bool", True),
        "save_path": ("str|None", None),
        "fixed_ranges": ("dict|None", None),
        "legend_marker_scale": ("float", 1.0),
        "legend_font_size": ("int", 12),
        "show_centroids": ("bool", False),
        "centroid_condition": ("str|None", None),
        "centroid_order": ("list|None", None),
        "centroid_groupby": ("str|None", None),
        "centroid_size": ("float", 20.0),
        "centroid_start_symbol": ("str", "circle"),
        "centroid_end_symbol": ("str", "diamond"),
        "centroid_line_width": ("float", 6.0),
        "centroid_colors": ("dict|None", None),
    },
    "pl.archetypal_space_multi": {
        "adata_list": ("list[AnnData]", REQUIRED),
        "archetype_coords_key": ("str", "archetype_coordinates"),
        "pca_key": ("str", "X_pca"),
        "labels_list": ("list[str]|None", None),
        "color_by": ("str|None", None),
        "color_values": ("array|None", None),
        "cell_size": ("float", 2.0),
        "cell_opacity": ("float", 0.6),
        "archetype_size": ("float", 8.0),
        "archetype_colors": ("list[str]|None", None),
        "show_labels": ("bool|list[int]", True),
        "auto_scale": ("bool", True),
        "range_reference": ("int|AnnData|None", None),
        "fixed_ranges": ("dict|None", None),
        "color_scale": ("str", "viridis"),
        "categorical_colors": ("dict|None", None),
        "title": ("str", "Multi-Archetypal Space Comparison"),
        "save_path": ("str|None", None),
    },
    "pl.archetype_positions": {
        "adata": ("AnnData", REQUIRED),
        "coords_key": ("str", "archetype_coordinates"),
        "title": ("str", "Archetype Positions in PCA Space"),
        "figsize": ("tuple", (15, 6)),
        "cmap": ("str", "tab10"),
        "show_distances": ("bool", True),
        "save_path": ("str|None", None),
    },
    "pl.archetype_positions_3d": {
        "adata": ("AnnData", REQUIRED),
        "coords_key": ("str", "archetype_coordinates"),
        "title": ("str", "Archetype Positions in 3D PCA Space"),
        "figsize": ("tuple", (12, 10)),
        "cmap": ("str", "tab10"),
        "save_path": ("str|None", None),
    },
    "pl.archetype_statistics": {
        "adata": ("AnnData", REQUIRED),
        "coords_key": ("str", "archetype_coordinates"),
        "verbose": ("bool", True),
    },
    "pl.training_metrics": {
        "history": ("dict", REQUIRED),
        "height": ("int", 400),
        "width": ("int", 800),
        "display": ("bool", True),  # False to get Figure object
    },
    "pl.elbow_curve": {
        "cv_summary": ("CVSummary", REQUIRED),
        "metrics": ("list[str]", ["archetype_r2", "rmse"]),
    },
    "pl.dotplot": {
        "results_df": ("DataFrame", REQUIRED),
        "x_col": ("str", "archetype"),
        "y_col": ("str", "gene"),
        "size_col": ("str", "mean_archetype"),
        "color_col": ("str", "pvalue"),
        "top_n_per_group": ("int", 10),
        "filter_zero_p": ("bool", True),
        "log_transform_p": ("bool", True),
        "max_log_p": ("float", 300.0),
        "title": ("str", "Gene-Archetype Associations"),
        "figsize": ("tuple", (12, 8)),
        "color_palette": ("str", "plasma"),
        "save_path": ("str|None", None),
    },
    "pl.pattern_dotplot": {
        "pattern_df": ("DataFrame", REQUIRED),
        "pattern_type": ("str|None", None),
        "top_n": ("int", 20),
        "min_effect_size": ("float", 0.5),
        "max_pvalue": ("float", 0.05),
        "figsize": ("tuple", (12, 8)),
        "title": ("str|None", None),
        "save_path": ("str|None", None),
    },
    "pl.pattern_heatmap": {
        "pattern_df": ("DataFrame", REQUIRED),
        "adata": ("AnnData", REQUIRED),
        "top_n": ("int", 30),
        "cluster_patterns": ("bool", True),
        "cluster_features": ("bool", True),
        "figsize": ("tuple", (10, 12)),
        "cmap": ("str", "RdBu_r"),
        "save_path": ("str|None", None),
    },
    "pl.pattern_summary_barplot": {
        "pattern_results": ("dict", REQUIRED),  # Dict[str, DataFrame]
        "figsize": ("tuple", (14, 6)),
        "save_path": ("str|None", None),
    },
    "pl.fate_probabilities": {
        "adata": ("AnnData", REQUIRED),
        "lineages": ("list|None", None),
        "basis": ("str", "X_umap"),
        "same_plot": ("bool", False),
        "ncols": ("int", 3),
        "figsize": ("tuple|None", None),
    },
    "pl.lineage_drivers": {
        "adata": ("AnnData", REQUIRED),
        "lineage": ("str", REQUIRED),
        "n_genes": ("int", 20),
        "driver_key": ("str|None", None),
        "figsize": ("tuple", (10, 8)),
    },
    # --- Spatial Plots (plotly-based) ---
    "pl.nhood_enrichment": {
        "adata": ("AnnData", REQUIRED),
        "uns_key": ("str", "archetype_nhood_enrichment"),
        "cluster_key": ("str", "archetypes"),
        "title": ("str", "Archetype Neighborhood Enrichment"),
        "colorscale": ("str", "RdBu_r"),
        "save_path": ("str|None", None),
    },
    "pl.co_occurrence": {
        "adata": ("AnnData", REQUIRED),
        "uns_key": ("str", "archetype_co_occurrence"),
        "cluster_key": ("str", "archetypes"),
        "title": ("str", "Archetype Spatial Co-occurrence"),
        "save_path": ("str|None", None),
    },
    "pl.spatial_archetypes": {
        "adata": ("AnnData", REQUIRED),
        "spatial_key": ("str", "spatial"),
        "color_key": ("str", "archetypes"),
        "point_size": ("float", 2.0),
        "opacity": ("float", 0.7),
        "title": ("str", "Spatial Archetype Map"),
        "save_path": ("str|None", None),
        "colors": ("list[str]|None", None),  # Custom color palette
        "legend_marker_size": ("float", 12.0),
    },
    "pl.interaction_boundaries": {
        "adata": ("AnnData", REQUIRED),
        "spatial_key": ("str", "spatial"),
        "score_key": ("str", "boundary_score"),
        "point_size": ("float", 2.0),
        "colorscale": ("str", "Inferno"),
        "title": ("str|None", None),  # Auto-generated if None
        "save_path": ("str|None", None),
    },
    "pl.spatial_autocorr": {
        "adata": ("AnnData", REQUIRED),
        "uns_key": ("str", "archetype_spatial_autocorr"),
        "title": ("str|None", None),
        "save_path": ("str|None", None),
    },
    "pl.cross_correlations": {
        "adata": ("AnnData", REQUIRED),
        "uns_key": ("str", "archetype_interaction_boundaries"),
        "title": ("str|None", None),
        "save_path": ("str|None", None),
    },
}


def get_params(func_name: str) -> dict:
    """Get parameter dict for a function.

    Usage:
        >>> params = get_params("tl.train_archetypal")
        >>> print(params["n_epochs"])  # ('int', 50)
    """
    if func_name in FUNCTION_PARAMS:
        return FUNCTION_PARAMS[func_name]
    for key in FUNCTION_PARAMS:
        if key.endswith(f".{func_name}"):
            return FUNCTION_PARAMS[key]
    raise KeyError(f"Unknown function: {func_name}")


# =============================================================================
# ANNDATA STORAGE KEYS
# =============================================================================
# What PEACH stores in AnnData objects

ADATA_KEYS = {
    "obsm": {
        "X_pca": "PCA coordinates (REQUIRED INPUT) - check variants: X_pca, X_PCA, pca",
        "archetype_distances": "[n_cells, n_archetypes] Euclidean distances",
        "cell_archetype_weights": "[n_cells, n_archetypes] barycentric coordinates (A matrix)",
        "cell_archetype_weights_latent": "[n_cells, n_archetypes] z latent values",
        "cell_archetype_weights_mu": "[n_cells, n_archetypes] encoder means",
        "cell_archetype_weights_log_var": "[n_cells, n_archetypes] encoder log vars",
        "pathway_scores": "[n_cells, n_pathways] pathway activity scores",
        "X_lsi": "[n_cells, n_components] LSI embeddings from scATAC-seq (pc.pp.prepare_atacseq)",
    },
    "uns": {
        "archetype_coordinates": "[n_archetypes, n_pcs] archetype positions in PCA space",
        "archetype_positions": "Copy of archetype_coordinates",
        "archetype_distance_info": "Dict with n_archetypes, pca_key_used, etc.",
        "true_archetypes": "[n_archetypes, n_features] ground truth (synthetic data only)",
        "conditional_centroids": "Dict[condition_column, ConditionalCentroidResult] centroid positions",
        "trajectory_{src}_to_{tgt}": "Dict with trajectory analysis results (source, target, driver_genes, etc.)",
        "pathway_scores_names": "List[str] pathway names for columns in obsm['pathway_scores']",
        "lsi": "Dict with 'variance_ratio' and 'components' from TruncatedSVD (pc.pp.prepare_atacseq)",
        "archetype_nhood_enrichment": "Dict with 'zscore' and 'count' arrays (pc.tl.archetype_nhood_enrichment)",
        "archetype_co_occurrence": "Dict with 'occ' and 'interval' arrays (pc.tl.archetype_co_occurrence)",
        "archetype_spatial_autocorr": "DataFrame with Moran's I / Geary's C per archetype weight",
        "archetype_interaction_boundaries": "Dict with boundary_scores, mean_weights_a/b, cross-correlations",
    },
    "obs": {
        "archetypes": "Categorical: 'archetype_0', 'archetype_1', ..., 'no_archetype'",
        "dpt_pseudotime": "Diffusion pseudotime (CellRank)",
        "{lineage}_pseudotime": "Per-lineage pseudotime (CellRank)",
        "trajectory_{src}_to_{tgt}_cells": "Boolean mask for cells in trajectory",
        "pseudotime_{src}_to_{tgt}": "Float trajectory-specific pseudotime",
        "boundary_score": "Float per-cell boundary score (pc.tl.archetype_interaction_boundaries)",
    },
    "obsp": {
        "T_forward": "Transition probability matrix (CellRank)",
        "spatial_connectivities": "Spatial neighbor graph (pc.tl.spatial_neighbors via squidpy)",
        "spatial_distances": "Spatial distance matrix (pc.tl.spatial_neighbors via squidpy)",
    },
}


# =============================================================================
# FIELDS REQUIRING .get() ACCESS
# =============================================================================
# These are OPTIONAL fields - direct access may raise KeyError

USE_GET_FOR: set[str] = {
    # TrainingResults convenience keys
    "final_archetype_r2",
    "final_rmse",
    "final_mae",
    "final_loss",
    "convergence_epoch",
    # TrainingHistory conditional metrics
    "val_loss",
    "val_archetype_r2",
    "val_rmse",
    "archetype_drift_mean",
    "archetype_drift_max",
    "constraint_violation_rate",
    "archetype_transform_grad_norm",
    # DataFrame columns (may be absent depending on FDR settings)
    "fdr_pvalue",
    "significant",
}


# =============================================================================
# DATAFRAME SCHEMAS (Column specifications for DataFrame-returning functions)
# =============================================================================

DATAFRAME_SCHEMAS = {
    "gene_associations": [
        "gene",  # str: gene name
        "archetype",  # str: "archetype_0", "archetype_1", ...
        "n_archetype_cells",  # int: cells in archetype bin
        "n_other_cells",  # int: cells outside archetype bin
        "mean_archetype",  # float: mean expression in archetype
        "mean_other",  # float: mean expression outside
        "log_fold_change",  # float: log2(mean_archetype / mean_other)
        "statistic",  # float: test statistic
        "pvalue",  # float: raw p-value
        "test_direction",  # str: "two-sided", "greater", "less"
        "direction",  # str: "higher" or "lower"
        "passes_lfc_threshold",  # bool: |log_fold_change| > min_logfc
        "fdr_pvalue",  # float: FDR-corrected p-value
        "significant",  # bool: fdr_pvalue < 0.05 AND passes_lfc
    ],
    "pathway_associations": [
        # Same as gene_associations + mean_diff
        "gene",  # str: pathway name (not gene!)
        "archetype",
        "n_archetype_cells",
        "n_other_cells",
        "mean_archetype",
        "mean_other",
        "mean_diff",  # float: pathway-specific effect
        "log_fold_change",
        "statistic",
        "pvalue",
        "test_direction",
        "direction",
        "passes_lfc_threshold",
        "fdr_pvalue",
        "significant",
    ],
    "conditional_associations": [
        "archetype",  # str: archetype name
        "condition",  # str: condition level
        "observed",  # int: observed count
        "expected",  # float: expected count under null
        "total_archetype",  # int: total cells in archetype
        "total_condition",  # int: total cells in condition
        "odds_ratio",  # float: >1 = enriched, <1 = depleted
        "ci_lower",  # float: 95% CI lower bound
        "ci_upper",  # float: 95% CI upper bound
        "pvalue",  # float: raw p-value
        "fdr_pvalue",  # float: FDR-corrected
        "significant",  # bool
    ],
    "pattern_analysis_individual": [
        # 15 columns - individual gene/pathway tests
        "gene",
        "archetype",
        "n_archetype_cells",
        "n_other_cells",
        "mean_archetype",
        "mean_other",
        "log_fold_change",
        "statistic",
        "pvalue",
        "test_direction",
        "direction",
        "passes_lfc_threshold",
        "fdr_pvalue",
        "significant",
        "primary_effect_size",
    ],
    "pattern_analysis_patterns": [
        # 26 columns - multi-archetype pattern tests
        "pathway",
        "pattern_name",
        "pattern_code",
        "pattern_type",
        "high_archetypes",
        "low_archetypes",
        "n_high_cells",
        "n_low_cells",
        "mean_high",
        "mean_low",
        "log_fold_change",
        "primary_effect_size",
        "statistic",
        "pvalue",
        "test_direction",
        "direction",
        "passes_lfc_threshold",
        "fdr_pvalue",
        "significant",
        # Additional pattern columns
        "high_archetype_count",
        "low_archetype_count",
        "high_archetype_names",
        "low_archetype_names",
        "pattern_strength",
        "pattern_balance",
        "pattern_complexity",
    ],
    "pattern_analysis_exclusivity": [
        # 21 columns - exclusivity analysis
        "pathway",
        "pattern_name",
        "pattern_code",
        "exclusive_high",
        "exclusive_low",
        "shared_high",
        "shared_low",
        "exclusivity_score",  # float: 0-1, higher = more exclusive
        "overlap_coefficient",
        "jaccard_index",
        # Plus standard columns
        "pvalue",
        "fdr_pvalue",
        "significant",
        "n_cells_high",
        "n_cells_low",
        "mean_high",
        "mean_low",
        "log_fold_change",
        "statistic",
        "direction",
        "passes_lfc_threshold",
    ],
}


# =============================================================================
# COMMON PITFALLS (INLINE REFERENCE)
# =============================================================================

PITFALLS = {
    "PCA_KEY": "Check for 'X_pca' vs 'X_PCA' vs 'pca' - varies by dataset",
    "CV_RANKING": "Use ranked[i].metric_value, NOT ranked[i].mean_archetype_r2",
    "DISTANCE_VS_WEIGHT": "Distance-based assignment ≠ weight-based for ~60% of cells",
    "SUM_CONSTRAINT": "A matrix rows sum to 1.0 (barycentric), verify with A.sum(dim=1)",
}


# =============================================================================
# QUICK TYPE LOOKUP HELPER
# =============================================================================


def get_return_type(func_name: str) -> tuple[str, list[str]]:
    """Get return type and keys for a function.

    Usage:
        >>> ret_type, keys = get_return_type("tl.train_archetypal")
        >>> print(f"Returns: {ret_type}")
        >>> print(f"Keys: {keys}")
    """
    if func_name in FUNCTION_RETURNS:
        return FUNCTION_RETURNS[func_name]
    # Try without module prefix
    for key, val in FUNCTION_RETURNS.items():
        if key.endswith(f".{func_name}"):
            return val
    raise KeyError(f"Unknown function: {func_name}. Check FUNCTION_RETURNS.keys()")


def print_api_summary():
    """Print compact API summary."""
    print("=" * 70)
    print("PEACH API SUMMARY")
    print("=" * 70)
    for module in ["pp", "tl", "pl", "_core"]:
        funcs = [k for k in FUNCTION_RETURNS.keys() if k.startswith(module)]
        if funcs:
            print(f"\n{module} ({len(funcs)} functions):")
            for f in funcs:
                ret_type, _ = FUNCTION_RETURNS[f]
                print(f"  {f.split('.')[-1]:35} → {ret_type}")


# =============================================================================
# MODULE DOCSTRING FOR AGENTS
# =============================================================================

__doc__ = """
PEACH Type Index - READ THIS FIRST

LOOKUP PATTERN:
1. Find function in FUNCTION_RETURNS → get return type name
2. If Pydantic type needed: grep "class {TypeName}" types.py
3. If still unclear: read specific function docstring

EXAMPLE:
    >>> from peach._core.types_index import get_return_type
    >>> ret_type, keys = get_return_type("tl.train_archetypal")
    >>> # Returns: ("TrainingResults", ["history", "final_model", ...])
    >>>
    >>> # Need full TrainingResults definition?
    >>> # grep "class TrainingResults" src/peach/_core/types.py

COMMON TASKS:
- Training results: FUNCTION_RETURNS["tl.train_archetypal"]
- CV search results: FUNCTION_RETURNS["tl.hyperparameter_search"]
- What's in adata: ADATA_KEYS
- Optional fields: USE_GET_FOR
"""

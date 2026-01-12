# src/peach/_core/tools_schema.py
"""
PEACH Tools Schema - Function signatures for programmatic use.

This module provides complete input/output schemas for all PEACH functions,
enabling programmatic access and tool integrations.

Usage:
    from peach._core.tools_schema import get_tool_schema, TOOL_SCHEMAS

    # Get schema for a specific function
    schema = get_tool_schema("tl.train_archetypal")

    # Generate tool definitions for an agent
    tools = generate_tool_definitions(["tl.train_archetypal", "tl.archetypal_coordinates"])

Key Concepts:
    - All functions operate on AnnData objects referenced by `adata_key`
    - Session state maintains loaded datasets in ADATA_REGISTRY
    - Results are stored back in the AnnData object (adata.obs, adata.obsm, adata.uns)

Version: 0.3.0
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# =============================================================================
# PARAMETER TYPE DEFINITIONS
# =============================================================================


class ParamType(str, Enum):
    """Parameter types for tool schemas."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    ADATA_REF = "adata_reference"  # Special: reference to loaded AnnData
    MODEL_REF = "model_reference"  # Special: reference to trained model


@dataclass
class Parameter:
    """Tool parameter definition."""

    name: str
    type: ParamType
    description: str
    required: bool = True
    default: Any = None
    enum: list[Any] | None = None  # For constrained choices
    items_type: ParamType | None = None  # For arrays

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema format."""
        schema = {
            "type": self.type.value if self.type != ParamType.ADATA_REF else "string",
            "description": self.description,
        }
        if self.type == ParamType.ADATA_REF:
            schema["description"] += " (AnnData reference key)"
        if self.type == ParamType.MODEL_REF:
            schema["description"] += " (trained model reference key)"
        if self.default is not None:
            schema["default"] = self.default
        if self.enum:
            schema["enum"] = self.enum
        if self.items_type and self.type == ParamType.ARRAY:
            schema["items"] = {"type": self.items_type.value}
        return schema


@dataclass
class ToolSchema:
    """Complete tool schema for a PEACH function."""

    name: str
    description: str
    parameters: list[Parameter]
    returns: str  # Return type name from types_index.py
    returns_description: str
    modifies_adata: list[str] = field(default_factory=list)  # Keys modified in adata
    requires: list[str] = field(default_factory=list)  # Prerequisites (e.g., "X_pca in adata.obsm")

    def to_tool_definition(self) -> dict[str, Any]:
        """Convert to tool definition format."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        return {
            "name": self.name.replace(".", "_"),  # tl.train_archetypal → tl_train_archetypal
            "description": self._build_description(),
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def _build_description(self) -> str:
        """Build complete description including requirements and outputs."""
        desc = self.description
        if self.requires:
            desc += f"\n\nRequires: {', '.join(self.requires)}"
        if self.modifies_adata:
            desc += f"\n\nModifies AnnData: {', '.join(self.modifies_adata)}"
        desc += f"\n\nReturns: {self.returns} - {self.returns_description}"
        return desc


# =============================================================================
# TOOL SCHEMAS - Complete parameter definitions
# =============================================================================

TOOL_SCHEMAS: dict[str, ToolSchema] = {
    # =========================================================================
    # pp (PREPROCESSING)
    # =========================================================================
    "pp.load_data": ToolSchema(
        name="pp.load_data",
        description="Load single-cell data from file into AnnData format.",
        parameters=[
            Parameter("filepath", ParamType.STRING, "Path to data file (.h5ad, .loom, .csv)"),
            Parameter("adata_key", ParamType.STRING, "Key to store loaded AnnData in registry", default="adata"),
        ],
        returns="AnnData",
        returns_description="Loaded AnnData object stored in registry",
        modifies_adata=[],
    ),
    "pp.generate_synthetic": ToolSchema(
        name="pp.generate_synthetic",
        description="Generate synthetic data with known archetypes for testing.",
        parameters=[
            Parameter("n_points", ParamType.INTEGER, "Number of samples (cells) to generate", default=1000),
            Parameter("n_dimensions", ParamType.INTEGER, "Number of features (genes)", default=50),
            Parameter("n_archetypes", ParamType.INTEGER, "Number of true archetypes", default=4),
            Parameter("noise", ParamType.FLOAT, "Noise standard deviation", default=0.1),
            Parameter("seed", ParamType.INTEGER, "Random seed", default=1205),
            Parameter(
                "archetype_type",
                ParamType.STRING,
                "How to generate archetypes",
                default="random",
                enum=["random", "simplex"],
            ),
            Parameter("scale", ParamType.FLOAT, "Scale of archetype positions", default=20.0),
        ],
        returns="AnnData",
        returns_description="Synthetic AnnData with true archetypes in .uns['true_archetypes']",
        modifies_adata=["uns['true_archetypes']", "obsm['X_pca']"],
    ),
    "pp.prepare_training": ToolSchema(
        name="pp.prepare_training",
        description="Prepare data for training (ensure PCA, create DataLoader).",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to loaded AnnData"),
            Parameter("n_pcs", ParamType.INTEGER, "Number of PCA components", default=30),
            Parameter("batch_size", ParamType.INTEGER, "Training batch size", default=256),
            Parameter("pca_key", ParamType.STRING, "Key for PCA in obsm", default="X_pca"),
        ],
        returns="Tuple[DataLoader, AnnData]",
        returns_description="PyTorch DataLoader and updated AnnData",
        requires=["adata loaded"],
        modifies_adata=["obsm['X_pca'] if not present"],
    ),
    # =========================================================================
    # tl (TOOLS) - Training
    # =========================================================================
    "tl.train_archetypal": ToolSchema(
        name="tl.train_archetypal",
        description="Train Deep Archetypal Analysis model. Main training function.",
        parameters=[
            # --- CORE (commonly used) ---
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with PCA"),
            Parameter("n_archetypes", ParamType.INTEGER, "Number of archetypes to learn", default=5),
            Parameter("n_epochs", ParamType.INTEGER, "Maximum training epochs", default=50),
            Parameter(
                "hidden_dims",
                ParamType.ARRAY,
                "Encoder/decoder layer dimensions, e.g. [256, 128, 64]",
                default=None,
                items_type=ParamType.INTEGER,
            ),
            Parameter(
                "inflation_factor",
                ParamType.FLOAT,
                "PCHA inflation factor for initialization (1.2-2.0 recommended)",
                default=1.5,
            ),
            Parameter("early_stopping", ParamType.BOOLEAN, "Enable early stopping", default=False),
            Parameter("early_stopping_patience", ParamType.INTEGER, "Patience for early stopping", default=10),
            Parameter("seed", ParamType.INTEGER, "Random seed", default=42),
            Parameter(
                "device", ParamType.STRING, "Computing device", default="auto", enum=["cpu", "cuda", "mps", "auto"]
            ),
            # --- DATA SELECTION ---
            Parameter("layer", ParamType.STRING, "Expression layer to use", required=False, default=None),
            Parameter("pca_key", ParamType.STRING, "Key for PCA coordinates", default="X_pca"),
            Parameter(
                "store_coords_key", ParamType.STRING, "Key for archetype coords in uns", default="archetype_coordinates"
            ),
            # --- ADVANCED (model_config for other options) ---
            Parameter(
                "model_config",
                ParamType.OBJECT,
                "Additional model config: {archetypal_weight, kld_weight, diversity_weight, use_barycentric}",
                default=None,
            ),
            # --- LOSS WEIGHTS (advanced - defaults are optimal) ---
            Parameter(
                "archetypal_weight",
                ParamType.FLOAT,
                "Archetypal loss weight (default 1.0 in model)",
                required=False,
                default=None,
            ),
            Parameter(
                "kld_weight",
                ParamType.FLOAT,
                "KL divergence weight (0.0 optimal, non-zero hurts R²)",
                required=False,
                default=None,
            ),
            Parameter("reconstruction_weight", ParamType.FLOAT, "Reconstruction loss weight", default=0.0),
            Parameter("diversity_weight", ParamType.FLOAT, "Archetype diversity weight", default=0.0),
            # --- OPTIMIZER (advanced) ---
            Parameter(
                "optimizer_config",
                ParamType.OBJECT,
                "Optimizer config: {lr: float, weight_decay: float}",
                required=False,
                default=None,
            ),
            Parameter("lr_factor", ParamType.FLOAT, "LR reduction factor on plateau", default=0.1),
            Parameter("lr_patience", ParamType.INTEGER, "LR scheduler patience", default=10),
            # --- TRAINING BEHAVIOR (advanced) ---
            Parameter("activation_func", ParamType.STRING, "Activation function", default="relu"),
            Parameter("track_stability", ParamType.BOOLEAN, "Track archetype stability metrics", default=True),
            Parameter("validate_constraints", ParamType.BOOLEAN, "Validate archetypal constraints", default=True),
            Parameter("constraint_tolerance", ParamType.FLOAT, "Constraint violation tolerance", default=0.001),
            Parameter("stability_history_size", ParamType.INTEGER, "Window size for stability tracking", default=20),
            # --- EARLY STOPPING (advanced) ---
            Parameter(
                "early_stopping_metric",
                ParamType.STRING,
                "Metric for early stopping",
                default="archetype_r2",
                enum=["archetype_r2", "loss", "rmse"],
            ),
            Parameter("min_improvement", ParamType.FLOAT, "Min improvement for early stopping", default=0.0001),
            Parameter("validation_check_interval", ParamType.INTEGER, "Epochs between validation checks", default=5),
        ],
        returns="TrainingResults",
        returns_description="Dict with history, final_model, model, training_config. Use .get() for final_archetype_r2",
        requires=["X_pca in adata.obsm"],
        modifies_adata=["uns['archetype_coordinates']"],
    ),
    "tl.hyperparameter_search": ToolSchema(
        name="tl.hyperparameter_search",
        description="Grid search over hyperparameters with cross-validation.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter(
                "n_archetypes_range",
                ParamType.ARRAY,
                "Archetype numbers to test",
                default=[3, 4, 5, 6],
                items_type=ParamType.INTEGER,
            ),
            Parameter(
                "hidden_dims_options",
                ParamType.ARRAY,
                "Network architectures to test",
                default=[[128, 64], [256, 128, 64]],
            ),
            Parameter("cv_folds", ParamType.INTEGER, "Number of CV folds", default=5),
            Parameter("max_epochs_cv", ParamType.INTEGER, "Max epochs per fold", default=50),
            Parameter("subsample_fraction", ParamType.FLOAT, "Fraction of data for CV", default=0.5),
        ],
        returns="CVSummary",
        returns_description="Use .rank_by_metric('archetype_r2') → ranked[i].metric_value for best config",
        requires=["X_pca in adata.obsm"],
        modifies_adata=[],
    ),
    # =========================================================================
    # tl (TOOLS) - Coordinates & Assignment
    # =========================================================================
    "tl.archetypal_coordinates": ToolSchema(
        name="tl.archetypal_coordinates",
        description="Compute distances from cells to archetypes in PCA space.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("model_key", ParamType.MODEL_REF, "Reference to trained model", default="model"),
            Parameter("pca_key", ParamType.STRING, "Key for PCA coordinates", default="X_pca"),
        ],
        returns="DataFrame",
        returns_description="Columns: archetype_0_distance, ..., nearest_archetype, nearest_archetype_distance",
        requires=["archetype_coordinates in adata.uns", "trained model"],
        modifies_adata=["obsm['archetype_distances']"],
    ),
    "tl.assign_archetypes": ToolSchema(
        name="tl.assign_archetypes",
        description="Assign cells to nearest archetype based on distance.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter(
                "percentage_per_archetype",
                ParamType.FLOAT,
                "Top percentage of cells per archetype (0.1 = 10%)",
                default=0.1,
            ),
            Parameter("obsm_key", ParamType.STRING, "Key for distances", default="archetype_distances"),
        ],
        returns="None",
        returns_description="Modifies adata.obs['archetypes'] with Categorical assignments",
        requires=["archetype_distances in adata.obsm"],
        modifies_adata=["obs['archetypes']"],
    ),
    "tl.extract_archetype_weights": ToolSchema(
        name="tl.extract_archetype_weights",
        description="Extract cell-archetype weight matrix (A matrix / barycentric coordinates).",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("model_key", ParamType.MODEL_REF, "Reference to trained model"),
        ],
        returns="np.ndarray",
        returns_description="Shape (n_cells, n_archetypes), rows sum to 1",
        requires=["trained model"],
        modifies_adata=["obsm['cell_archetype_weights']"],
    ),
    "tl.compute_conditional_centroids": ToolSchema(
        name="tl.compute_conditional_centroids",
        description="Compute centroid positions in PCA space for each level of a categorical condition. "
        "Enables trajectory visualization of condition changes (e.g., treatment phases) in archetypal space.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with PCA coordinates"),
            Parameter("condition_column", ParamType.STRING, "Categorical column in adata.obs to compute centroids for"),
            Parameter("pca_key", ParamType.STRING, "Key for PCA coordinates in obsm", default="X_pca"),
            Parameter(
                "store_key", ParamType.STRING, "Key to store results in adata.uns", default="conditional_centroids"
            ),
            Parameter(
                "exclude_archetypes",
                ParamType.ARRAY,
                "Archetype labels to exclude from calculation",
                default=["no_archetype", "archetype_0"],
                items_type=ParamType.STRING,
            ),
            Parameter(
                "groupby",
                ParamType.STRING,
                "Second categorical column for multi-group trajectories",
                required=False,
                default=None,
            ),
            Parameter("verbose", ParamType.BOOLEAN, "Print progress messages", default=True),
        ],
        returns="ConditionalCentroidResult",
        returns_description="Dict with centroids, centroids_3d, cell_counts, levels. Also stores in adata.uns['conditional_centroids']",
        requires=["X_pca in adata.obsm", "condition_column in adata.obs"],
        modifies_adata=["uns['conditional_centroids']"],
    ),
    "tl.assign_to_centroids": ToolSchema(
        name="tl.assign_to_centroids",
        description="Assign cells to nearest centroid based on distance (top bin_prop% closest). "
        "Mirrors assign_archetypes but for condition-based centroids. "
        "Enables using treatment phase centroids as trajectory endpoints in single_trajectory_analysis.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with centroids computed"),
            Parameter("condition_column", ParamType.STRING, "Condition column used in compute_conditional_centroids"),
            Parameter("pca_key", ParamType.STRING, "Key for PCA coordinates in obsm", default="X_pca"),
            Parameter(
                "centroid_key",
                ParamType.STRING,
                "Key in adata.uns containing centroid results",
                default="conditional_centroids",
            ),
            Parameter(
                "bin_prop", ParamType.FLOAT, "Proportion of cells to assign to each centroid (0.15 = 15%)", default=0.15
            ),
            Parameter(
                "obs_key", ParamType.STRING, "Key in adata.obs to store assignments", default="centroid_assignments"
            ),
            Parameter(
                "exclude_archetypes",
                ParamType.ARRAY,
                "Archetype labels to exclude from assignment",
                default=["no_archetype"],
                items_type=ParamType.STRING,
            ),
            Parameter("verbose", ParamType.BOOLEAN, "Print progress messages", default=True),
        ],
        returns="None",
        returns_description="Modifies adata.obs[obs_key] with Categorical assignments (condition levels + 'unassigned')",
        requires=["conditional_centroids in adata.uns (from compute_conditional_centroids)", "X_pca in adata.obsm"],
        modifies_adata=["obs['centroid_assignments']"],
    ),
    # =========================================================================
    # tl (TOOLS) - Statistical Testing
    # =========================================================================
    "tl.gene_associations": ToolSchema(
        name="tl.gene_associations",
        description="Test gene expression associations with archetypes (Mann-Whitney U by default). "
        "Returns 14-column DataFrame with gene, archetype, log_fold_change, pvalue, fdr_pvalue, etc.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("obs_key", ParamType.STRING, "Key for archetype assignments in obs", default="archetypes"),
            Parameter("bin_prop", ParamType.FLOAT, "Proportion of cells per archetype bin", default=0.1),
            Parameter("obsm_key", ParamType.STRING, "Key for distances in obsm", default="archetype_distances"),
            Parameter(
                "use_layer", ParamType.STRING, "Expression layer to use (None = .X)", required=False, default=None
            ),
            # --- Statistical testing ---
            Parameter(
                "test_method",
                ParamType.STRING,
                "Statistical test method",
                default="mannwhitneyu",
                enum=["mannwhitneyu", "ttest"],
            ),
            Parameter(
                "test_direction",
                ParamType.STRING,
                "Test direction",
                default="two-sided",
                enum=["two-sided", "greater", "less"],
            ),
            # --- FDR correction ---
            Parameter(
                "fdr_method",
                ParamType.STRING,
                "FDR correction method",
                default="benjamini_hochberg",
                enum=["benjamini_hochberg", "bonferroni"],
            ),
            Parameter(
                "fdr_scope",
                ParamType.STRING,
                "FDR scope: global (all tests) or per_archetype",
                default="global",
                enum=["global", "per_archetype"],
            ),
            # --- Thresholds ---
            Parameter("min_logfc", ParamType.FLOAT, "Minimum |log_fold_change| threshold", default=0.01),
            Parameter("min_cells", ParamType.INTEGER, "Minimum cells per group for valid test", default=10),
            Parameter(
                "comparison_group",
                ParamType.STRING,
                "Comparison group: 'all' other cells or specific archetype",
                default="all",
            ),
            Parameter("verbose", ParamType.BOOLEAN, "Print progress", default=True),
        ],
        returns="DataFrame[GeneAssociationResult]",
        returns_description="14 cols: gene, archetype, n_archetype_cells, n_other_cells, mean_archetype, mean_other, "
        "log_fold_change, statistic, pvalue, test_direction, direction, passes_lfc_threshold, fdr_pvalue, significant",
        requires=["archetypes in adata.obs", "archetype_distances in adata.obsm"],
        modifies_adata=[],
    ),
    "tl.pathway_associations": ToolSchema(
        name="tl.pathway_associations",
        description="Test pathway activity associations with archetypes. Requires pp.compute_pathway_scores() first. "
        "Returns 15-column DataFrame (gene_associations columns + mean_diff).",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("obs_key", ParamType.STRING, "Key for archetype assignments in obs", default="archetypes"),
            Parameter("pathway_obsm_key", ParamType.STRING, "Key for pathway scores in obsm", default="pathway_scores"),
            Parameter("obsm_key", ParamType.STRING, "Key for distances in obsm", default="archetype_distances"),
            # --- Statistical testing ---
            Parameter(
                "test_method",
                ParamType.STRING,
                "Statistical test method",
                default="mannwhitneyu",
                enum=["mannwhitneyu", "ttest"],
            ),
            Parameter(
                "test_direction",
                ParamType.STRING,
                "Test direction",
                default="two-sided",
                enum=["two-sided", "greater", "less"],
            ),
            # --- FDR correction ---
            Parameter("fdr_method", ParamType.STRING, "FDR correction method", default="benjamini_hochberg"),
            Parameter("fdr_scope", ParamType.STRING, "FDR scope", default="global", enum=["global", "per_archetype"]),
            # --- Thresholds ---
            Parameter("min_logfc", ParamType.FLOAT, "Minimum effect size threshold", default=0.01),
            Parameter("min_cells", ParamType.INTEGER, "Minimum cells per group", default=10),
            Parameter("comparison_group", ParamType.STRING, "Comparison group", default="all"),
            Parameter("verbose", ParamType.BOOLEAN, "Print progress", default=True),
        ],
        returns="DataFrame[PathwayAssociationResult]",
        returns_description="15 cols: gene (pathway name), archetype, mean_diff, + 12 cols from gene_associations",
        requires=["archetypes in adata.obs", "pathway_scores in adata.obsm (from pp.compute_pathway_scores)"],
        modifies_adata=[],
    ),
    "tl.conditional_associations": ToolSchema(
        name="tl.conditional_associations",
        description="Test archetype enrichment for categorical conditions (hypergeometric test). "
        "Returns 12-column DataFrame with odds ratios and confidence intervals.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("obs_column", ParamType.STRING, "Categorical column in adata.obs to test"),
            Parameter("obs_key", ParamType.STRING, "Key for archetype assignments", default="archetypes"),
            Parameter(
                "archetype_assignments",
                ParamType.ARRAY,
                "Override archetype assignments (array)",
                required=False,
                default=None,
            ),
            # --- Testing ---
            Parameter("test_method", ParamType.STRING, "Test method", default="hypergeometric"),
            Parameter("fdr_method", ParamType.STRING, "FDR correction method", default="benjamini_hochberg"),
            Parameter("min_cells", ParamType.INTEGER, "Minimum cells per group", default=5),
            Parameter("verbose", ParamType.BOOLEAN, "Print progress", default=True),
        ],
        returns="DataFrame[ConditionalAssociationResult]",
        returns_description="12 cols: archetype, condition, observed, expected, total_archetype, total_condition, "
        "odds_ratio, ci_lower, ci_upper, pvalue, fdr_pvalue, significant",
        requires=["archetypes in adata.obs", "obs_column in adata.obs"],
        modifies_adata=[],
    ),
    "tl.pattern_analysis": ToolSchema(
        name="tl.pattern_analysis",
        description="Test multi-archetype patterns. Returns dict with conditional keys: "
        "'individual' (15 cols), 'patterns' (26 cols), 'exclusivity' (21 cols, requires patterns).",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("obs_key", ParamType.STRING, "Key for archetype assignments", default="archetypes"),
            Parameter(
                "data_obsm_key",
                ParamType.STRING,
                "Key for data (pathway_scores or gene expression)",
                default="pathway_scores",
            ),
            # --- Control which analyses to run ---
            Parameter(
                "include_individual_tests",
                ParamType.BOOLEAN,
                "Include individual gene/pathway tests → 'individual' key",
                default=True,
            ),
            Parameter(
                "include_pattern_tests",
                ParamType.BOOLEAN,
                "Include multi-archetype pattern tests → 'patterns' key",
                default=True,
            ),
            Parameter(
                "include_exclusivity_analysis",
                ParamType.BOOLEAN,
                "Include exclusivity analysis → 'exclusivity' key (requires patterns)",
                default=True,
            ),
            Parameter("verbose", ParamType.BOOLEAN, "Print progress", default=True),
        ],
        returns="Dict[str, DataFrame]",
        returns_description="Conditional dict: 'individual' (if include_individual_tests), 'patterns' (if include_pattern_tests), "
        "'exclusivity' (if include_exclusivity_analysis AND include_pattern_tests)",
        requires=["archetypes in adata.obs"],
        modifies_adata=[],
    ),
    # =========================================================================
    # tl (TOOLS) - CellRank Integration
    # =========================================================================
    "tl.setup_cellrank": ToolSchema(
        name="tl.setup_cellrank",
        description="Set up CellRank workflow for trajectory analysis with archetypes or centroids as terminal states.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter(
                "high_purity_threshold",
                ParamType.FLOAT,
                "Percentile threshold for high-purity cells (only for archetypes)",
                default=0.80,
            ),
            Parameter("n_neighbors", ParamType.INTEGER, "Number of neighbors for k-NN graph", default=30),
            Parameter("n_pcs", ParamType.INTEGER, "Number of PCs to use", default=11),
            Parameter("compute_paga", ParamType.BOOLEAN, "Compute PAGA connectivity", default=True),
            Parameter("solver", ParamType.STRING, "Solver for fate probabilities", default="gmres"),
            Parameter("tol", ParamType.FLOAT, "Tolerance for solver", default=1e-6),
            Parameter(
                "terminal_obs_key",
                ParamType.STRING,
                "Key in obs for terminal states ('archetypes' or 'centroid_assignments')",
                default="archetypes",
            ),
            Parameter("verbose", ParamType.BOOLEAN, "Print progress", default=True),
        ],
        returns="Tuple[ConnectivityKernel, GPCCA]",
        returns_description="CellRank kernel and GPCCA estimator with fate probabilities",
        requires=["terminal_obs_key in adata.obs", "X_pca in adata.obsm"],
        modifies_adata=["obs['terminal_states']", "obsm['fate_probabilities']", "uns['lineage_names']"],
    ),
    "tl.compute_lineage_pseudotimes": ToolSchema(
        name="tl.compute_lineage_pseudotimes",
        description="Compute pseudotime for each lineage.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("root_cells", ParamType.STRING, "Obs column or cell IDs for root"),
        ],
        returns="Dict[str, np.ndarray]",
        returns_description="lineage_name → pseudotime array",
        requires=["T_forward in adata.obsp"],
        modifies_adata=["obs['dpt_pseudotime']", "obs['{lineage}_pseudotime']"],
    ),
    # =========================================================================
    # pl (PLOTTING)
    # =========================================================================
    "pl.archetypal_space": ToolSchema(
        name="pl.archetypal_space",
        description="Plot cells in 2D archetypal simplex projection.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("color", ParamType.STRING, "Column to color by", default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
            Parameter("save", ParamType.STRING, "Path to save figure", default=None),
        ],
        returns="Figure",
        returns_description="Matplotlib Figure (None if show=True)",
        requires=["cell_archetype_weights in adata.obsm"],
        modifies_adata=[],
    ),
    "pl.training_metrics": ToolSchema(
        name="pl.training_metrics",
        description="Plot training loss curves and metrics.",
        parameters=[
            Parameter("results", ParamType.OBJECT, "TrainingResults dict from tl.train_archetypal"),
            Parameter(
                "metrics",
                ParamType.ARRAY,
                "Metrics to plot",
                default=["loss", "archetype_r2"],
                items_type=ParamType.STRING,
            ),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="Figure",
        returns_description="Multi-panel training metrics figure",
        requires=["TrainingResults from training"],
        modifies_adata=[],
    ),
    "pl.dotplot": ToolSchema(
        name="pl.dotplot",
        description="Create dotplot of gene/pathway expression by archetype.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("var_names", ParamType.ARRAY, "Genes or pathways to plot", items_type=ParamType.STRING),
            Parameter("groupby", ParamType.STRING, "Grouping column", default="archetypes"),
            Parameter("use_raw", ParamType.BOOLEAN, "Use raw expression", default=False),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="Figure",
        returns_description="Dotplot figure with size=fraction expressing, color=mean expression",
        requires=["archetypes in adata.obs"],
        modifies_adata=[],
    ),
    "pl.archetype_positions": ToolSchema(
        name="pl.archetype_positions",
        description="Plot archetype positions in PCA space.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("dims", ParamType.ARRAY, "PCA dimensions to plot", default=[0, 1], items_type=ParamType.INTEGER),
            Parameter("show_cells", ParamType.BOOLEAN, "Show cell scatter", default=True),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="Figure",
        returns_description="2D PCA scatter with archetype positions marked",
        requires=["archetype_coordinates in adata.uns", "X_pca in adata.obsm"],
        modifies_adata=[],
    ),
    # =========================================================================
    # _core (Advanced)
    # =========================================================================
    "_core.calculate_archetype_r2": ToolSchema(
        name="_core.calculate_archetype_r2",
        description="Calculate R² for archetypal reconstruction.",
        parameters=[
            Parameter("reconstructions", ParamType.OBJECT, "Reconstructed data tensor"),
            Parameter("original", ParamType.OBJECT, "Original data tensor"),
        ],
        returns="float",
        returns_description="R² value (1.0=perfect, 0.0=mean baseline, <0=worse than mean)",
        requires=[],
        modifies_adata=[],
    ),
    # =========================================================================
    # pp (PREPROCESSING) - Remaining
    # =========================================================================
    "pp.load_pathway_networks": ToolSchema(
        name="pp.load_pathway_networks",
        description="Load pathway gene sets from MSigDB or custom GMT files.",
        parameters=[
            Parameter("pathway_source", ParamType.STRING, "Source: 'msigdb', 'reactome', or GMT file path"),
            Parameter(
                "collection", ParamType.STRING, "MSigDB collection", default="H", enum=["H", "C2", "C5", "C6", "C7"]
            ),
            Parameter(
                "species", ParamType.STRING, "Species for gene symbols", default="human", enum=["human", "mouse"]
            ),
            Parameter("min_genes", ParamType.INTEGER, "Minimum genes per pathway", default=10),
            Parameter("max_genes", ParamType.INTEGER, "Maximum genes per pathway", default=500),
        ],
        returns="Dict[str, Set[str]]",
        returns_description="pathway_name → set of gene symbols",
        requires=[],
        modifies_adata=[],
    ),
    "pp.compute_pathway_scores": ToolSchema(
        name="pp.compute_pathway_scores",
        description="Compute pathway activity scores per cell (AUCell-like scoring).",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("pathways", ParamType.OBJECT, "Dict of pathway → gene sets"),
            Parameter("method", ParamType.STRING, "Scoring method", default="mean", enum=["mean", "sum", "aucell"]),
            Parameter("use_raw", ParamType.BOOLEAN, "Use raw counts", default=False),
        ],
        returns="AnnData",
        returns_description="AnnData with pathway_scores added to obsm",
        requires=["gene symbols in adata.var_names"],
        modifies_adata=["obsm['pathway_scores']"],
    ),
    # =========================================================================
    # tl (TOOLS) - Pattern Analysis Variants
    # =========================================================================
    "tl.archetype_exclusive_patterns": ToolSchema(
        name="tl.archetype_exclusive_patterns",
        description="Test genes/pathways exclusive to single archetypes (high in one, low in all others).",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter(
                "feature_type", ParamType.STRING, "Features to test", default="genes", enum=["genes", "pathways"]
            ),
            Parameter("bin_prop", ParamType.FLOAT, "Proportion of cells per archetype", default=0.1),
            Parameter("fdr_method", ParamType.STRING, "FDR correction method", default="benjamini_hochberg"),
            Parameter("min_logfc", ParamType.FLOAT, "Minimum log fold change", default=0.5),
        ],
        returns="DataFrame[PatternAssociationResult]",
        returns_description="pattern_type='exclusive'. Columns: gene/pathway, pattern_code, pvalue, significant",
        requires=["archetypes in adata.obs"],
        modifies_adata=[],
    ),
    "tl.specialization_patterns": ToolSchema(
        name="tl.specialization_patterns",
        description="Test genes/pathways showing specialization (high in subset, low in complement).",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter(
                "feature_type", ParamType.STRING, "Features to test", default="genes", enum=["genes", "pathways"]
            ),
            Parameter("bin_prop", ParamType.FLOAT, "Proportion of cells per group", default=0.1),
            Parameter("fdr_method", ParamType.STRING, "FDR correction method", default="benjamini_hochberg"),
            Parameter("min_logfc", ParamType.FLOAT, "Minimum log fold change", default=0.5),
        ],
        returns="DataFrame[PatternAssociationResult]",
        returns_description="pattern_type='specialization'. Tests all subsets of archetypes",
        requires=["archetypes in adata.obs"],
        modifies_adata=[],
    ),
    "tl.tradeoff_patterns": ToolSchema(
        name="tl.tradeoff_patterns",
        description="Test genes/pathways showing tradeoffs (high in one group, low in another).",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter(
                "feature_type", ParamType.STRING, "Features to test", default="genes", enum=["genes", "pathways"]
            ),
            Parameter("bin_prop", ParamType.FLOAT, "Proportion of cells per group", default=0.1),
            Parameter("mode", ParamType.STRING, "Tradeoff mode", default="pairs", enum=["pairs", "patterns"]),
            Parameter("fdr_method", ParamType.STRING, "FDR correction method", default="benjamini_hochberg"),
        ],
        returns="DataFrame[PatternAssociationResult]",
        returns_description="pattern_type='tradeoff'. Tests pairwise or multi-archetype tradeoffs",
        requires=["archetypes in adata.obs"],
        modifies_adata=[],
    ),
    # =========================================================================
    # tl (TOOLS) - CellRank Remaining
    # =========================================================================
    "tl.compute_lineage_drivers": ToolSchema(
        name="tl.compute_lineage_drivers",
        description="Identify driver genes for each lineage using correlation with fate probabilities.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("lineages", ParamType.ARRAY, "Lineage names to analyze", items_type=ParamType.STRING),
            Parameter("n_top_genes", ParamType.INTEGER, "Number of top drivers per lineage", default=100),
            Parameter("use_raw", ParamType.BOOLEAN, "Use raw counts for correlation", default=False),
        ],
        returns="DataFrame",
        returns_description="Columns: gene, lineage, correlation, pvalue, qvalue",
        requires=["fate probabilities computed"],
        modifies_adata=[],
    ),
    "tl.compute_transition_frequencies": ToolSchema(
        name="tl.compute_transition_frequencies",
        description="Compute transition frequencies between archetypes from transition matrix.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("transition_key", ParamType.STRING, "Key for transition matrix in obsp", default="T_forward"),
        ],
        returns="DataFrame",
        returns_description="Columns: source_archetype, target_archetype, frequency, normalized_freq",
        requires=["T_forward in adata.obsp", "archetypes in adata.obs"],
        modifies_adata=[],
    ),
    "tl.single_trajectory_analysis": ToolSchema(
        name="tl.single_trajectory_analysis",
        description="Analyze single archetype-to-archetype trajectory. Filters cells by source archetype and target "
        "fate probability, returns subset AnnData ready for CellRank gene_trends. "
        "REQUIRES: setup_cellrank() and compute_lineage_pseudotimes() to be run first. "
        "For driver genes, use CellRank's g.compute_lineage_drivers() directly.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with CellRank setup"),
            Parameter(
                "trajectory",
                ParamType.ARRAY,
                "Archetype pair as [source_idx, target_idx], e.g., [0, 3]",
                items_type=ParamType.INTEGER,
            ),
            Parameter(
                "trajectories",
                ParamType.ARRAY,
                "Multiple trajectory pairs to analyze sequentially",
                required=False,
                default=None,
            ),
            Parameter(
                "selection_method",
                ParamType.STRING,
                "How to select source cells: 'discrete' (archetypes column), 'weight' (threshold), 'both' (compare)",
                default="discrete",
                enum=["discrete", "weight", "both"],
            ),
            Parameter(
                "source_weight_threshold",
                ParamType.FLOAT,
                "Minimum barycentric weight for weight-based selection",
                default=0.4,
            ),
            Parameter(
                "target_fate_threshold", ParamType.FLOAT, "Minimum fate probability for target archetype", default=0.4
            ),
            Parameter("verbose", ParamType.BOOLEAN, "Print progress", default=True),
        ],
        returns="Tuple[SingleTrajectoryResult, AnnData]",
        returns_description="(result, adata_traj) - Result metadata and subset AnnData for trajectory cells. "
        "Use adata_traj directly with cr.pl.gene_trends(). List if trajectories provided.",
        requires=[
            "fate_probabilities in adata.obsm",
            "lineage_names in adata.uns",
            "pseudotime_to_{archetype} in adata.obs (from compute_lineage_pseudotimes)",
            "archetypes in adata.obs (for selection_method='discrete')",
        ],
        modifies_adata=["obs['trajectory_{src}_to_{tgt}_cells']", "uns['trajectory_{src}_to_{tgt}']"],
    ),
    # =========================================================================
    # pl (PLOTTING) - Remaining
    # =========================================================================
    "pl.archetypal_space_multi": ToolSchema(
        name="pl.archetypal_space_multi",
        description="Plot multiple archetypal space panels, one per condition.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("groupby", ParamType.STRING, "Column to split panels by"),
            Parameter("color", ParamType.STRING, "Column to color cells by", default=None),
            Parameter("ncols", ParamType.INTEGER, "Number of columns in grid", default=3),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
            Parameter("save", ParamType.STRING, "Path to save figure", default=None),
        ],
        returns="Figure",
        returns_description="Multi-panel figure with one simplex per group",
        requires=["cell_archetype_weights in adata.obsm", "groupby column in adata.obs"],
        modifies_adata=[],
    ),
    "pl.elbow_curve": ToolSchema(
        name="pl.elbow_curve",
        description="Plot metric vs n_archetypes for model selection (elbow method).",
        parameters=[
            Parameter("cv_summary", ParamType.OBJECT, "CVSummary from hyperparameter_search"),
            Parameter("metric", ParamType.STRING, "Metric to plot", default="archetype_r2"),
            Parameter("show_std", ParamType.BOOLEAN, "Show standard deviation bands", default=True),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="Figure",
        returns_description="Elbow curve with error bars",
        requires=["CVSummary from hyperparameter_search"],
        modifies_adata=[],
    ),
    "pl.archetype_positions_3d": ToolSchema(
        name="pl.archetype_positions_3d",
        description="Plot archetype positions in 3D PCA space.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("dims", ParamType.ARRAY, "PCA dimensions (3)", default=[0, 1, 2], items_type=ParamType.INTEGER),
            Parameter("show_cells", ParamType.BOOLEAN, "Show cell scatter", default=True),
            Parameter("alpha", ParamType.FLOAT, "Cell point transparency", default=0.3),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="Figure",
        returns_description="3D scatter plot with archetype positions",
        requires=["archetype_coordinates in adata.uns", "X_pca in adata.obsm"],
        modifies_adata=[],
    ),
    "pl.archetype_statistics": ToolSchema(
        name="pl.archetype_statistics",
        description="Plot summary statistics for archetypes (usage, distances, weights).",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("show_usage", ParamType.BOOLEAN, "Show archetype usage histogram", default=True),
            Parameter("show_distances", ParamType.BOOLEAN, "Show distance distributions", default=True),
            Parameter("show_weights", ParamType.BOOLEAN, "Show weight distributions", default=True),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="Figure",
        returns_description="Multi-panel statistics figure",
        requires=["archetypes in adata.obs", "cell_archetype_weights in adata.obsm"],
        modifies_adata=[],
    ),
    "pl.pattern_dotplot": ToolSchema(
        name="pl.pattern_dotplot",
        description="Dotplot visualization of pattern analysis results.",
        parameters=[
            Parameter("results_df", ParamType.OBJECT, "DataFrame from pattern_analysis/specialization/tradeoff"),
            Parameter("top_n", ParamType.INTEGER, "Number of top patterns per type", default=20),
            Parameter("significance_threshold", ParamType.FLOAT, "FDR threshold", default=0.05),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="Figure",
        returns_description="Dotplot with size=significance, color=effect size",
        requires=["PatternAssociationResult DataFrame"],
        modifies_adata=[],
    ),
    "pl.pattern_summary_barplot": ToolSchema(
        name="pl.pattern_summary_barplot",
        description="Bar plot summarizing number of significant patterns per archetype.",
        parameters=[
            Parameter("results_df", ParamType.OBJECT, "DataFrame from pattern_analysis"),
            Parameter(
                "pattern_types",
                ParamType.ARRAY,
                "Pattern types to include",
                default=["exclusive", "specialization", "tradeoff"],
                items_type=ParamType.STRING,
            ),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="Figure",
        returns_description="Grouped bar chart",
        requires=["PatternAssociationResult DataFrame"],
        modifies_adata=[],
    ),
    "pl.pattern_heatmap": ToolSchema(
        name="pl.pattern_heatmap",
        description="Heatmap of pattern effect sizes across archetypes.",
        parameters=[
            Parameter("results_df", ParamType.OBJECT, "DataFrame from pattern_analysis"),
            Parameter("value_col", ParamType.STRING, "Column for heatmap values", default="log_fold_change"),
            Parameter("top_n", ParamType.INTEGER, "Number of top patterns to show", default=50),
            Parameter("cluster", ParamType.BOOLEAN, "Hierarchically cluster rows/cols", default=True),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="Figure",
        returns_description="Clustered heatmap",
        requires=["PatternAssociationResult DataFrame"],
        modifies_adata=[],
    ),
    "pl.fate_probabilities": ToolSchema(
        name="pl.fate_probabilities",
        description="Plot CellRank fate probabilities on UMAP or embedding.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("lineages", ParamType.ARRAY, "Lineages to plot", items_type=ParamType.STRING),
            Parameter("basis", ParamType.STRING, "Embedding key in obsm", default="X_umap"),
            Parameter("ncols", ParamType.INTEGER, "Columns in subplot grid", default=3),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="Figure",
        returns_description="UMAP colored by fate probability per lineage",
        requires=["fate probabilities computed", "embedding in adata.obsm"],
        modifies_adata=[],
    ),
    # Note: gene_trends removed - use cellrank.pl.gene_trends() directly
    "pl.lineage_drivers": ToolSchema(
        name="pl.lineage_drivers",
        description="Plot top driver genes for each lineage.",
        parameters=[
            Parameter("drivers_df", ParamType.OBJECT, "DataFrame from compute_lineage_drivers"),
            Parameter("top_n", ParamType.INTEGER, "Number of top drivers per lineage", default=10),
            Parameter("show_correlation", ParamType.BOOLEAN, "Show correlation values", default=True),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="Figure",
        returns_description="Horizontal bar chart of driver genes",
        requires=["lineage_drivers DataFrame"],
        modifies_adata=[],
    ),
    # =========================================================================
    # _core (ADVANCED) - Remaining Core Functions
    # =========================================================================
    "_core.train_vae": ToolSchema(
        name="_core.train_vae",
        description="Low-level VAE training function with full control over training loop.",
        parameters=[
            Parameter("model", ParamType.OBJECT, "Deep_AA model instance"),
            Parameter("dataloader", ParamType.OBJECT, "PyTorch DataLoader"),
            Parameter("n_epochs", ParamType.INTEGER, "Number of training epochs", default=100),
            Parameter("lr", ParamType.FLOAT, "Learning rate", default=1e-3),
            Parameter("early_stopping", ParamType.BOOLEAN, "Enable early stopping", default=True),
            Parameter("early_stopping_patience", ParamType.INTEGER, "Early stopping patience", default=10),
            Parameter("track_stability", ParamType.BOOLEAN, "Track archetype stability", default=True),
            Parameter("validate_constraints", ParamType.BOOLEAN, "Validate simplex constraints", default=True),
            Parameter("device", ParamType.STRING, "Computing device", default="cpu"),
            Parameter("_cv_mode", ParamType.BOOLEAN, "Internal: suppress adata warning during CV", default=False),
        ],
        returns="Tuple[CoreTrainingResults, Module]",
        returns_description="(results_dict, trained_model). Results has same structure as TrainingResults",
        requires=["initialized model", "DataLoader"],
        modifies_adata=[],
    ),
    "_core.get_archetypal_coordinates": ToolSchema(
        name="_core.get_archetypal_coordinates",
        description="Extract archetypal coordinates from model for a single batch (internal use).",
        parameters=[
            Parameter("model", ParamType.OBJECT, "Trained Deep_AA model"),
            Parameter("input", ParamType.OBJECT, "Input tensor [batch_size, n_features]"),
            Parameter("device", ParamType.STRING, "Computing device", default="cpu"),
        ],
        returns="ArchetypalCoordinates",
        returns_description="Dict with A, B, Y, mu, log_var, z tensors",
        requires=["trained model"],
        modifies_adata=[],
    ),
    "_core.extract_and_store_archetypal_coordinates": ToolSchema(
        name="_core.extract_and_store_archetypal_coordinates",
        description="Extract coordinates for full dataset and store in AnnData.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("model_key", ParamType.MODEL_REF, "Reference to trained model"),
            Parameter("pca_key", ParamType.STRING, "Key for PCA coordinates", default="X_pca"),
            Parameter("batch_size", ParamType.INTEGER, "Batch size for extraction", default=256),
        ],
        returns="ExtractedCoordinates",
        returns_description="archetype_positions, cell_weights, cell_latent, cell_mu, cell_log_var",
        requires=["trained model", "X_pca in adata.obsm"],
        modifies_adata=[
            "obsm['cell_archetype_weights']",
            "obsm['cell_archetype_weights_latent']",
            "obsm['cell_archetype_weights_mu']",
            "obsm['cell_archetype_weights_log_var']",
        ],
    ),
    "_core.compute_archetype_distances": ToolSchema(
        name="_core.compute_archetype_distances",
        description="Compute Euclidean distances from cells to archetypes in PCA space.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("pca_key", ParamType.STRING, "Key for PCA coordinates", default="X_pca"),
            Parameter(
                "archetype_key", ParamType.STRING, "Key for archetype positions", default="archetype_coordinates"
            ),
        ],
        returns="DataFrame",
        returns_description="Columns: archetype_0_distance, ..., nearest_archetype, nearest_archetype_distance",
        requires=["X_pca in adata.obsm", "archetype_coordinates in adata.uns"],
        modifies_adata=["obsm['archetype_distances']"],
    ),
    "_core.bin_cells_by_archetype": ToolSchema(
        name="_core.bin_cells_by_archetype",
        description="Assign cells to archetypes based on distance thresholds.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("percentage_per_archetype", ParamType.FLOAT, "Top percentage per archetype", default=0.1),
            Parameter("obsm_key", ParamType.STRING, "Key for distance matrix", default="archetype_distances"),
            Parameter("obs_key", ParamType.STRING, "Key for assignments in obs", default="archetypes"),
        ],
        returns="None",
        returns_description="Modifies adata.obs with Categorical assignments",
        requires=["archetype_distances in adata.obsm"],
        modifies_adata=["obs['archetypes']"],
    ),
    "_core.test_archetype_recovery": ToolSchema(
        name="_core.test_archetype_recovery",
        description="Test recovery of true archetypes (for synthetic data validation).",
        parameters=[
            Parameter("model", ParamType.OBJECT, "Trained Deep_AA model"),
            Parameter("true_archetypes", ParamType.OBJECT, "True archetype positions [n_arch, n_features]"),
            Parameter("dataloader", ParamType.OBJECT, "DataLoader for computing learned positions"),
            Parameter("tolerance", ParamType.FLOAT, "Distance tolerance for success", default=0.1),
        ],
        returns="ArchetypeRecoveryMetrics",
        returns_description="mean_distance, max_distance, normalized_mean_distance, recovery_success, assignment",
        requires=["true archetypes (synthetic data)"],
        modifies_adata=[],
    ),
    "_core.generate_convex_data": ToolSchema(
        name="_core.generate_convex_data",
        description="Generate synthetic data with known convex hull structure.",
        parameters=[
            Parameter("n_samples", ParamType.INTEGER, "Number of samples", default=1000),
            Parameter("n_archetypes", ParamType.INTEGER, "Number of archetypes", default=4),
            Parameter("n_features", ParamType.INTEGER, "Number of features", default=100),
            Parameter("noise_level", ParamType.FLOAT, "Noise standard deviation", default=0.1),
            Parameter("archetype_scale", ParamType.FLOAT, "Scale of archetype positions", default=1.0),
            Parameter("seed", ParamType.INTEGER, "Random seed", default=42),
        ],
        returns="Dict",
        returns_description="data, archetypes, weights, labels arrays",
        requires=[],
        modifies_adata=[],
    ),
    "_core.PCHA": ToolSchema(
        name="_core.PCHA",
        description="Principal Convex Hull Analysis - find archetypes as convex hull vertices.",
        parameters=[
            Parameter("X", ParamType.OBJECT, "Data matrix [n_samples, n_features]"),
            Parameter("n_archetypes", ParamType.INTEGER, "Number of archetypes to find"),
            Parameter("n_iter", ParamType.INTEGER, "Number of iterations", default=100),
            Parameter("delta", ParamType.FLOAT, "Convergence threshold", default=1e-6),
        ],
        returns="PCHAResults",
        returns_description="archetypes [n_arch, n_feat], A [n_cells, n_arch], B [n_arch, n_cells], archetype_r2",
        requires=[],
        modifies_adata=[],
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_tool_schema(func_name: str) -> ToolSchema:
    """Get tool schema for a function.

    Args:
        func_name: Function name (e.g., "tl.train_archetypal")

    Returns
    -------
        ToolSchema with parameters and return info

    Raises
    ------
        KeyError: If function not found
    """
    if func_name in TOOL_SCHEMAS:
        return TOOL_SCHEMAS[func_name]
    raise KeyError(f"No schema for '{func_name}'. Available: {list(TOOL_SCHEMAS.keys())}")


def generate_tool_definitions(func_names: list[str] | None = None) -> list[dict[str, Any]]:
    """Generate tool definitions for specified functions.

    Args:
        func_names: List of function names, or None for all functions

    Returns
    -------
        List of tool definitions in JSON schema format
    """
    if func_names is None:
        func_names = list(TOOL_SCHEMAS.keys())

    return [TOOL_SCHEMAS[name].to_tool_definition() for name in func_names]


def print_tool_summary():
    """Print summary of all available tools."""
    print("=" * 70)
    print("PEACH TOOLS SUMMARY")
    print("=" * 70)

    for module in ["pp", "tl", "pl", "_core"]:
        tools = [k for k in TOOL_SCHEMAS.keys() if k.startswith(module)]
        if tools:
            print(f"\n{module.upper()} ({len(tools)} tools):")
            for name in tools:
                schema = TOOL_SCHEMAS[name]
                n_required = sum(1 for p in schema.parameters if p.required)
                n_optional = len(schema.parameters) - n_required
                print(f"  {name.split('.')[-1]:30} → {schema.returns:20} ({n_required} req, {n_optional} opt)")


# =============================================================================
# SESSION STATE MANAGEMENT
# =============================================================================


class PeachSession:
    """Session state for PEACH tool execution.

    Maintains loaded AnnData objects and trained models across tool calls.

    Usage:
        session = PeachSession()
        session.load_adata("my_data", adata)
        session.store_model("my_model", trained_model)

        # Later calls can reference by key
        adata = session.get_adata("my_data")
    """

    def __init__(self):
        self._adata_registry: dict[str, Any] = {}
        self._model_registry: dict[str, Any] = {}
        self._results_registry: dict[str, Any] = {}

    def load_adata(self, key: str, adata: Any) -> None:
        """Register an AnnData object."""
        self._adata_registry[key] = adata

    def get_adata(self, key: str) -> Any:
        """Get AnnData by key."""
        if key not in self._adata_registry:
            raise KeyError(f"AnnData '{key}' not found. Available: {list(self._adata_registry.keys())}")
        return self._adata_registry[key]

    def store_model(self, key: str, model: Any) -> None:
        """Register a trained model."""
        self._model_registry[key] = model

    def get_model(self, key: str) -> Any:
        """Get model by key."""
        if key not in self._model_registry:
            raise KeyError(f"Model '{key}' not found. Available: {list(self._model_registry.keys())}")
        return self._model_registry[key]

    def store_results(self, key: str, results: Any) -> None:
        """Store results (e.g., TrainingResults, CVSummary)."""
        self._results_registry[key] = results

    def get_results(self, key: str) -> Any:
        """Get stored results."""
        if key not in self._results_registry:
            raise KeyError(f"Results '{key}' not found. Available: {list(self._results_registry.keys())}")
        return self._results_registry[key]

    def list_all(self) -> dict[str, list[str]]:
        """List all registered objects."""
        return {
            "adata": list(self._adata_registry.keys()),
            "models": list(self._model_registry.keys()),
            "results": list(self._results_registry.keys()),
        }


# Global session instance (for simple use cases)
_default_session = PeachSession()


def get_session() -> PeachSession:
    """Get the default session instance."""
    return _default_session

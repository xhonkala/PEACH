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

NOTE ON adata_key vs adata
--------------------------
Schema entries use ``adata_key`` (ParamType.ADATA_REF) referencing AnnData objects
by name in a PeachSession registry. The actual Python API functions accept ``adata``
(an AnnData instance) directly. When using schemas programmatically via PeachSession,
call ``session.get_adata(adata_key)`` to resolve the reference before passing to
the function. When calling functions directly, ignore adata_key and pass adata.

Version: 0.5.0
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
    "pp.prepare_atacseq": ToolSchema(
        name="pp.prepare_atacseq",
        description="TF-IDF + LSI preprocessing for scATAC-seq peak count data. "
        "Produces embeddings usable with pc.tl.train_archetypal(adata, pca_key='X_lsi').",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with peak count matrix in .X"),
            Parameter("n_components", ParamType.INTEGER, "Number of LSI components to compute (30-50 standard)", default=50),
            Parameter(
                "drop_first",
                ParamType.BOOLEAN,
                "Drop first SVD component (captures sequencing depth, not biology)",
                default=True,
            ),
            Parameter("log_tf", ParamType.BOOLEAN, "Use log(1 + TF) variant of term frequency", default=True),
            Parameter("store_key", ParamType.STRING, "Key in adata.obsm to store LSI embeddings", default="X_lsi"),
            Parameter("random_state", ParamType.INTEGER, "Random seed for truncated SVD", default=42),
        ],
        returns="None",
        returns_description="Modifies adata in place: obsm[store_key] = LSI embeddings, uns['lsi'] = variance info",
        requires=["sparse peak count matrix in adata.X"],
        modifies_adata=["obsm['X_lsi']", "uns['lsi']"],
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
                "device", ParamType.STRING, "Computing device", default="cpu", enum=["cpu", "cuda", "mps"]
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
                "KL divergence weight (0.1 default, regularizes encoder variance)",
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
        returns_description="Columns: archetype_1_distance, ..., nearest_archetype, nearest_archetype_distance (1-indexed)",
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
    # tl (TOOLS) - Spatial Analysis (requires squidpy)
    # =========================================================================
    "tl.spatial_neighbors": ToolSchema(
        name="tl.spatial_neighbors",
        description="Build spatial neighbor graph from tissue coordinates. "
        "Wrapper around squidpy.gr.spatial_neighbors() with PEACH-appropriate defaults.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with spatial coordinates"),
            Parameter("spatial_key", ParamType.STRING, "Key in adata.obsm for 2D spatial coordinates", default="spatial"),
            Parameter("n_neighs", ParamType.INTEGER, "Number of nearest neighbors", default=10),
            Parameter(
                "coord_type",
                ParamType.STRING,
                "Coordinate type: 'generic' for Slide-seq/MERFISH, 'grid' for Visium",
                default="generic",
                enum=["generic", "grid"],
            ),
        ],
        returns="None",
        returns_description="Modifies adata in place: obsp['spatial_connectivities'] and obsp['spatial_distances']",
        requires=["spatial coordinates in adata.obsm['spatial']", "pip install peach[spatial]"],
        modifies_adata=["obsp['spatial_connectivities']", "obsp['spatial_distances']"],
    ),
    "tl.archetype_nhood_enrichment": ToolSchema(
        name="tl.archetype_nhood_enrichment",
        description="Test spatial neighborhood enrichment between archetype groups via permutation test. "
        "For each archetype pair, tests whether cells co-localize more/less than expected by chance.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with spatial graph"),
            Parameter("cluster_key", ParamType.STRING, "Column in adata.obs with archetype labels", default="archetypes"),
            Parameter("n_perms", ParamType.INTEGER, "Number of permutations for significance testing", default=1000),
            Parameter("seed", ParamType.INTEGER, "Random seed for permutation reproducibility", default=42),
        ],
        returns="Dict",
        returns_description="Dict with 'zscore' and 'count' arrays [n_archetypes x n_archetypes]. "
        "Positive z-score = enriched (co-localized), negative = depleted (separated). "
        "Also stored in adata.uns['archetype_nhood_enrichment'].",
        requires=["spatial_connectivities in adata.obsp (from spatial_neighbors)", "archetypes in adata.obs"],
        modifies_adata=["uns['archetype_nhood_enrichment']"],
    ),
    "tl.archetype_co_occurrence": ToolSchema(
        name="tl.archetype_co_occurrence",
        description="Compute distance-dependent co-occurrence of archetype groups. "
        "Measures how co-occurrence ratio varies with spatial distance.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with spatial coordinates"),
            Parameter("cluster_key", ParamType.STRING, "Column in adata.obs with archetype labels", default="archetypes"),
            Parameter("spatial_key", ParamType.STRING, "Key in adata.obsm with spatial coordinates", default="spatial"),
            Parameter("interval", ParamType.INTEGER, "Number of distance intervals to evaluate", default=50),
        ],
        returns="Dict",
        returns_description="Dict with 'occ' (ratios [n_arch, n_arch, n_intervals]) and 'interval' (distance bins). "
        "Also stored in adata.uns['archetype_co_occurrence'].",
        requires=["spatial coordinates in adata.obsm", "archetypes in adata.obs", "pip install peach[spatial]"],
        modifies_adata=["uns['archetype_co_occurrence']"],
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
        description="Convert fate probabilities to lineage-specific pseudotimes. Stores in adata.obs.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("lineage_names", ParamType.ARRAY, "Lineage names to compute. None = all from uns['lineage_names']", required=False, default=None),
            Parameter("fate_prob_key", ParamType.STRING, "Key in obsm for fate probabilities", default="fate_probabilities"),
        ],
        returns="None",
        returns_description="Modifies adata.obs in-place with pseudotime_to_{lineage} columns",
        requires=["fate_probabilities in adata.obsm", "lineage_names in adata.uns"],
        modifies_adata=["obs['pseudotime_to_{lineage}']"],
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
    # =========================================================================
    # pl (PLOTTING) - Spatial (requires squidpy for analysis, plotly for plots)
    # =========================================================================
    "pl.nhood_enrichment": ToolSchema(
        name="pl.nhood_enrichment",
        description="Plotly heatmap of archetype neighborhood enrichment z-scores. "
        "Red = co-localized, blue = spatially separated.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with enrichment results"),
            Parameter(
                "uns_key", ParamType.STRING, "Key in adata.uns for enrichment results", default="archetype_nhood_enrichment"
            ),
            Parameter("cluster_key", ParamType.STRING, "Column in adata.obs for axis labels", default="archetypes"),
            Parameter("title", ParamType.STRING, "Plot title", default="Archetype Neighborhood Enrichment"),
            Parameter("colorscale", ParamType.STRING, "Plotly colorscale", default="RdBu_r"),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
        ],
        returns="Figure",
        returns_description="Plotly Figure with z-score heatmap (symmetric around 0)",
        requires=["archetype_nhood_enrichment in adata.uns (from tl.archetype_nhood_enrichment)"],
        modifies_adata=[],
    ),
    "pl.co_occurrence": ToolSchema(
        name="pl.co_occurrence",
        description="Plotly line plot of distance-dependent archetype co-occurrence ratios. "
        "Values > 1 = co-occurrence above chance, < 1 = avoidance.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with co-occurrence results"),
            Parameter(
                "uns_key", ParamType.STRING, "Key in adata.uns for co-occurrence results", default="archetype_co_occurrence"
            ),
            Parameter("cluster_key", ParamType.STRING, "Column in adata.obs for legend labels", default="archetypes"),
            Parameter("title", ParamType.STRING, "Plot title", default="Archetype Spatial Co-occurrence"),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
        ],
        returns="Figure",
        returns_description="Plotly Figure with co-occurrence ratio vs distance lines per archetype pair",
        requires=["archetype_co_occurrence in adata.uns (from tl.archetype_co_occurrence)"],
        modifies_adata=[],
    ),
    "pl.spatial_archetypes": ToolSchema(
        name="pl.spatial_archetypes",
        description="ScatterGL plot of cells on spatial coordinates colored by archetype assignment.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with spatial coords"),
            Parameter("spatial_key", ParamType.STRING, "Key in adata.obsm with 2D spatial coordinates", default="spatial"),
            Parameter("color_key", ParamType.STRING, "Column in adata.obs to color by", default="archetypes"),
            Parameter("point_size", ParamType.FLOAT, "Size of scatter points", default=2.0),
            Parameter("opacity", ParamType.FLOAT, "Point opacity", default=0.7),
            Parameter("title", ParamType.STRING, "Plot title", default="Spatial Archetype Map"),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
        ],
        returns="Figure",
        returns_description="Plotly Figure with cells plotted at spatial positions, colored by archetype",
        requires=["spatial coordinates in adata.obsm", "color_key in adata.obs"],
        modifies_adata=[],
    ),
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
    # v0.5.0: Continuous Characterization (tl)
    # =========================================================================
    "tl.feature_simplex_regression": ToolSchema(
        name="tl.feature_simplex_regression",
        description="Simplex regression of features on archetype weights using Scheffe polynomials. "
        "Fits linear (degree 1) and optionally interaction (degree 2) models. "
        "Stores results in adata.uns['peach_simplex_regression'].",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with archetype weights"),
            Parameter(
                "feature_matrix",
                ParamType.STRING,
                "Feature matrix to regress. None = adata.X, str = obsm key, array = direct",
                required=False,
                default=None,
            ),
            Parameter("feature_names", ParamType.ARRAY, "Feature names. Inferred if None", required=False, default=None),
            Parameter("max_degree", ParamType.INTEGER, "1 = linear only, 2 = with pairwise interactions", default=2),
            Parameter("permutation_test", ParamType.BOOLEAN, "Run permutation test for model significance", default=False),
            Parameter("n_permutations", ParamType.INTEGER, "Number of permutations", default=1000),
            Parameter("n_bootstrap", ParamType.INTEGER, "Bootstrap samples for CIs (0 to disable)", default=1000),
            Parameter("robust_se", ParamType.BOOLEAN, "Use HC3 heteroscedasticity-consistent SEs", default=True),
            Parameter("store_residuals", ParamType.BOOLEAN, "Store residuals in adata.obsm", default=True),
            Parameter("comprehensive_degree", ParamType.BOOLEAN, "Run degree 2..K-1 fits with incremental F-tests", default=False),
            Parameter("store_to_adata", ParamType.BOOLEAN, "Store results in adata.uns", default=True),
            Parameter("copy", ParamType.BOOLEAN, "Operate on a copy of adata", default=False),
        ],
        returns="dict (serialized SimplexRegressionResult)",
        returns_description="vertex_coefficients [n_features, K], vertex_covariance [n_features] list of [K,K], "
        "r_squared_degree1, f_pvalue, vertex_pvalues, "
        "interaction_coefficients (optional), CIs (optional), effective_rank, expected_rank, extra_rank_deficient",
        requires=["cell_archetype_weights in adata.obsm"],
        modifies_adata=[
            "uns['peach_simplex_regression']",
            "obsm['peach_residuals'] (if store_residuals=True)",
        ],
    ),
    "tl.classify_feature_patterns": ToolSchema(
        name="tl.classify_feature_patterns",
        description="Classify features into biological pattern types (flat, archetype-exclusive, "
        "interaction, structured) based on FDR-corrected regression p-values.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with regression results"),
            Parameter(
                "regression_result",
                ParamType.OBJECT,
                "SimplexRegressionResult. If None, reads from adata.uns",
                required=False,
                default=None,
            ),
            Parameter("fdr_threshold", ParamType.FLOAT, "FDR p-value threshold for significance", default=0.05),
            Parameter("exclusive_ratio", ParamType.FLOAT, "Min fold-change for exclusive pattern", default=2.0),
        ],
        returns="PatternClassificationResult",
        returns_description="classifications [n_features] with pattern type and details, pattern_counts dict, "
        "archetype_features {archetype_idx: [feature_names]}",
        requires=["peach_simplex_regression in adata.uns (or regression_result)"],
        modifies_adata=["uns['peach_feature_patterns']"],
    ),
    "tl.archetype_driver_regression": ToolSchema(
        name="tl.archetype_driver_regression",
        description="Flipped regression: features predict archetype weights (ILR space). "
        "Identifies which features drive archetypal specialization.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with archetype weights"),
            Parameter(
                "feature_matrix",
                ParamType.STRING,
                "Feature matrix (predictors). Default: pathway_scores if available, else adata.X",
                required=False,
                default=None,
            ),
            Parameter("feature_names", ParamType.ARRAY, "Feature names", required=False, default=None),
            Parameter("max_degree", ParamType.INTEGER, "1 = main effects only, 2 = with interactions", default=2),
            Parameter("n_bootstrap", ParamType.INTEGER, "Bootstrap samples for CIs (0 to disable)", default=1000),
            Parameter("robust_se", ParamType.BOOLEAN, "Use HC3 SEs", default=True),
            Parameter(
                "max_interaction_features",
                ParamType.INTEGER,
                "Max features allowed for degree=2",
                default=50,
            ),
            Parameter("copy", ParamType.BOOLEAN, "Operate on a copy of adata", default=False),
        ],
        returns="dict (serialized DriverRegressionResult)",
        returns_description="main_coefficients_ilr [K-1, n_features], main_coefficients [K, n_features], "
        "main_pvalues, main_pvalues_fdr [K, n_features], r_squared [K-1]",
        requires=["cell_archetype_weights in adata.obsm"],
        modifies_adata=["uns['peach_driver_regression']"],
    ),
    "tl.feature_simplex_decomposition": ToolSchema(
        name="tl.feature_simplex_decomposition",
        description="Decompose cell populations by mixture model in archetype weight space. "
        "Supports Gaussian (ILR) or Dirichlet mixture. Selects component count by BIC or ICL "
        "and filters by multi-initialization pairwise stability.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with archetype weights"),
            Parameter(
                "feature_matrix",
                ParamType.STRING,
                "Feature matrix for component characterization",
                required=False,
                default=None,
            ),
            Parameter("feature_names", ParamType.ARRAY, "Feature names", required=False, default=None),
            Parameter(
                "n_components_range",
                ParamType.ARRAY,
                "(min, max) components to test. Default: (K, 3*K)",
                required=False,
                default=None,
            ),
            Parameter(
                "model_type",
                ParamType.STRING,
                "Mixture model type: 'gaussian' (GMM in ILR space) or 'dirichlet'",
                default="gaussian",
                enum=["gaussian", "dirichlet"],
            ),
            Parameter(
                "model_selection",
                ParamType.STRING,
                "Model selection criterion: 'bic' or 'icl' (integrated classification likelihood)",
                default="bic",
                enum=["bic", "icl"],
            ),
            Parameter("covariance_type", ParamType.STRING, "GMM covariance type", default="full"),
            Parameter("n_initializations", ParamType.INTEGER, "Random inits for stability", default=20),
            Parameter("stability_threshold", ParamType.FLOAT, "Min stability score to retain", default=0.7),
            Parameter("ilr_epsilon", ParamType.FLOAT, "ILR zero smoothing constant", default=0.001),
            Parameter("characterize_features", ParamType.BOOLEAN, "Compute per-component feature profiles", default=True),
            Parameter(
                "reassignment_confidence",
                ParamType.FLOAT,
                "Min posterior probability to reassign unstable cells. 0.0 = always reassign, 1.0 = never",
                default=0.0,
            ),
            Parameter("random_state", ParamType.INTEGER, "Random seed", default=42),
            Parameter("copy", ParamType.BOOLEAN, "Operate on a copy of adata", default=False),
        ],
        returns="dict",
        returns_description="n_components_optimal, n_components_stable, component_assignments, "
        "component_simplex_means, component_probabilities, bic_values, "
        "icl_values (when model_selection='icl'), model_type",
        requires=["cell_archetype_weights in adata.obsm"],
        modifies_adata=["uns['peach_gmm']", "obsm['peach_gmm_labels']"],
    ),
    "tl.flow_within": ToolSchema(
        name="tl.flow_within",
        description="Train a neural ODE flow model to transport source cells to target cells "
        "within a single AnnData. Measures transport quality via MMD. Supports OT-CFM "
        "training and holdout validation.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("source", ParamType.OBJECT, "Obs column filter dict, e.g. {'treatment': 'Base'}"),
            Parameter("target", ParamType.OBJECT, "Obs column filter dict, e.g. {'treatment': 'PD1'}"),
            Parameter("pca_key", ParamType.STRING, "Key in obsm for PCA coordinates", default="X_pca"),
            Parameter("hidden_dims", ParamType.ARRAY, "MLP hidden dimensions", default=[128, 128, 128]),
            Parameter("lr", ParamType.FLOAT, "Learning rate", default=1e-3),
            Parameter("n_epochs", ParamType.INTEGER, "Training epochs", default=1000),
            Parameter("batch_size", ParamType.INTEGER, "Batch size", default=256),
            Parameter("n_steps", ParamType.INTEGER, "ODE integration steps", default=50),
            Parameter("device", ParamType.STRING, "Computing device", default="cpu"),
            Parameter("solver_method", ParamType.STRING, "ODE solver: 'euler', 'midpoint', 'heun3', 'dopri5'", default="dopri5"),
            Parameter("name", ParamType.STRING, "Name for storage key", required=False, default=None),
            Parameter("random_state", ParamType.INTEGER, "Random seed", default=42),
            Parameter("return_model", ParamType.BOOLEAN, "Return FlowModel in result (needed for Jacobian/trajectory)", default=False),
            Parameter("use_ot", ParamType.BOOLEAN, "Use minibatch Sinkhorn OT coupling for training pairs (requires POT)", default=False),
            Parameter("holdout_fraction", ParamType.FLOAT, "Fraction of source cells held out for validation MMD (0 to 1)", default=0.0),
            Parameter("copy", ParamType.BOOLEAN, "Operate on a copy of adata", default=False),
        ],
        returns="FlowWithinResult",
        returns_description="transported [n_source, dim], losses, mmd_before, mmd_after, source/target masks, "
        "model (if return_model=True), holdout_mmd (if holdout_fraction > 0)",
        requires=["X_pca in adata.obsm", "source/target columns in adata.obs"],
        modifies_adata=["uns['peach_flow_*']"],
    ),
    "tl.archetype_summary": ToolSchema(
        name="tl.archetype_summary",
        description="Generate structured summary for one or all archetypes. "
        "Aggregates results from regression, patterns, drivers, and GMM.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with regression results"),
            Parameter(
                "archetype_idx",
                ParamType.INTEGER,
                "Specific archetype index, or None for all",
                required=False,
                default=None,
            ),
            Parameter("top_n", ParamType.INTEGER, "Top enriched/depleted features to report", default=20),
            Parameter("include_drivers", ParamType.BOOLEAN, "Include driver regression results", default=True),
            Parameter("include_gmm", ParamType.BOOLEAN, "Include GMM components", default=True),
        ],
        returns="dict | list[dict]",
        returns_description="Per-archetype summary with top_enriched, top_depleted, interactions, pattern_counts, "
        "driver_genesets (optional), gmm_components (optional)",
        requires=["peach_simplex_regression in adata.uns"],
        modifies_adata=[],
    ),
    "tl.archetype_mmd": ToolSchema(
        name="tl.archetype_mmd",
        description="K x K MMD similarity matrix between archetype cell populations. "
        "Uses soft archetype weights (not hard argmax) and permutation-based p-values.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with archetype weights"),
            Parameter(
                "adata_b_key",
                ParamType.ADATA_REF,
                "Second AnnData for between-fit comparison. None for within-fit.",
                required=False,
                default=None,
            ),
            Parameter("pca_key", ParamType.STRING, "Key in obsm for PCA coordinates", default="X_pca"),
            Parameter("n_permutations", ParamType.INTEGER, "Permutations for p-value computation", default=1000),
            Parameter("seed", ParamType.INTEGER, "Random seed", default=42),
            Parameter("copy", ParamType.BOOLEAN, "Operate on a copy of adata", default=False),
        ],
        returns="dict (serialized ArchetypeMMDResult)",
        returns_description="mmd_matrix [K, K], pvalue_matrix [K, K], n_permutations, is_between_fit, "
        "archetype_names_a, archetype_names_b",
        requires=["cell_archetype_weights in adata.obsm", "X_pca in adata.obsm"],
        modifies_adata=["uns['peach_archetype_mmd']"],
    ),
    "tl.archetype_feature_similarity": ToolSchema(
        name="tl.archetype_feature_similarity",
        description="Feature-level archetype similarity: Spearman correlation on FDR-significant "
        "regression coefficient vectors. Requires simplex regression results.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with regression results"),
            Parameter(
                "adata_b_key",
                ParamType.ADATA_REF,
                "Second AnnData for between-fit Spearman on shared features",
                required=False,
                default=None,
            ),
            Parameter("copy", ParamType.BOOLEAN, "Operate on a copy of adata", default=False),
        ],
        returns="dict (serialized ArchetypeFeatureSimilarityResult)",
        returns_description="spearman_matrix [K, K], spearman_pvalue_matrix [K, K], "
        "spearman_pvalue_fdr_matrix [K, K], n_shared_features, n_significant_features",
        requires=["peach_simplex_regression in adata.uns"],
        modifies_adata=["uns['peach_archetype_feature_similarity']"],
    ),
    "tl.archetype_contrasts": ToolSchema(
        name="tl.archetype_contrasts",
        description="Pairwise Wald contrasts between archetype regression coefficients. "
        "Tests H0: beta_j = beta_k using t-distribution (n - K df) with global BH FDR correction.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with archetype weights and regression results"),
            Parameter("robust_se", ParamType.BOOLEAN, "Use HC3 heteroscedasticity-consistent covariance", default=True),
            Parameter("copy", ParamType.BOOLEAN, "Operate on a copy of adata", default=False),
        ],
        returns="dict (serialized ArchetypeContrastsResult)",
        returns_description="pairs [(j,k)], delta_beta {pair: [n_features]}, delta_se, t_scores, "
        "pvalues (t-distribution), pvalues_fdr (global BH), feature_names, n_features, n_archetypes",
        requires=["cell_archetype_weights in adata.obsm", "peach_simplex_regression in adata.uns"],
        modifies_adata=["uns['peach_archetype_contrasts']"],
    ),
    "tl.component_regression": ToolSchema(
        name="tl.component_regression",
        description="Run simplex regression independently per GMM component. "
        "Detects component-specific feature drivers masked in the global regression.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with archetype weights and GMM results"),
            Parameter("feature_type", ParamType.STRING, "'genes' for adata.X, or obsm key", default="genes"),
            Parameter("n_bootstrap", ParamType.INTEGER, "Bootstrap replicates for CIs", default=100),
            Parameter("robust_se", ParamType.BOOLEAN, "Use HC3 SEs", default=True),
        ],
        returns="dict",
        returns_description="component_regs: dict[int, regression_result], n_components: int",
        requires=["cell_archetype_weights in adata.obsm", "peach_gmm in adata.uns"],
        modifies_adata=[],
    ),
    "tl.flow_gene_alignment": ToolSchema(
        name="tl.flow_gene_alignment",
        description="Compute gene alignment with flow velocity. Projects mean transport "
        "direction onto PCA loadings to identify genes aligned/opposed to the flow.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with PCA loadings"),
            Parameter("flow_result", ParamType.OBJECT, "FlowWithinResult from pc.tl.flow_within()"),
            Parameter("t", ParamType.FLOAT, "Time point for instantaneous velocity (None=displacement)", required=False, default=None),
            Parameter("n_top", ParamType.INTEGER, "Number of top aligned/opposed genes to report", default=50),
            Parameter(
                "pca_loadings_key",
                ParamType.STRING,
                "Key in adata.varm for PCA loadings",
                required=False,
                default=None,
            ),
            Parameter("n_permutations", ParamType.INTEGER, "Permutations for significance. 0 to skip", default=0),
            Parameter("per_cell", ParamType.BOOLEAN, "Compute per-cell per-gene alignment scores", default=False),
            Parameter("random_state", ParamType.INTEGER, "Random seed for permutations", default=42),
        ],
        returns="dict",
        returns_description="alignment_scores [n_genes], gene_names, top_aligned, top_opposed, t, "
        "velocity_mode ('displacement' or 'instantaneous'), "
        "alignment_pvalues (optional), alignment_pvalues_fdr (optional), "
        "per_cell_alignment [n_cells, n_genes] (if per_cell=True)",
        requires=["PCs in adata.varm", "FlowWithinResult"],
        modifies_adata=[],
    ),
    "tl.flow_jacobian": ToolSchema(
        name="tl.flow_jacobian",
        description="Compute Jacobian of the flow velocity field. Returns determinants, "
        "mean Jacobian, and feature expansion scores via PCA loadings.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with PCA loadings"),
            Parameter("flow_result", ParamType.OBJECT, "FlowWithinResult from pc.tl.flow_within()"),
            Parameter("flow_model", ParamType.OBJECT, "FlowModel (from flow_result['model'] when return_model=True)"),
            Parameter("t", ParamType.FLOAT, "Time point to evaluate Jacobian", default=0.5),
            Parameter("evaluation_points", ParamType.ARRAY, "Points to evaluate at [n_points, dim]. None = subsample source", required=False, default=None),
            Parameter("pca_loadings_key", ParamType.STRING, "Key in adata.varm for PCA loadings", required=False, default=None),
            Parameter("aggregate", ParamType.STRING, "Aggregation: 'mean' or 'none'", default="mean"),
        ],
        returns="dict",
        returns_description="jacobian_det [n_points], mean_jacobian [dim, dim], feature_expansion [n_genes], t",
        requires=["FlowModel from flow_within(return_model=True)", "PCs in adata.varm (for feature_expansion)"],
        modifies_adata=[],
    ),
    "tl.flow_bifurcation": ToolSchema(
        name="tl.flow_bifurcation",
        description="Eigenvalue-based bifurcation scoring along the flow trajectory. "
        "Computes Jacobian eigenvalues at multiple timepoints to detect saddle points "
        "and divergent dynamics.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("flow_result", ParamType.OBJECT, "FlowWithinResult from pc.tl.flow_within()"),
            Parameter("flow_model", ParamType.OBJECT, "FlowModel (from flow_result['model'] when return_model=True)"),
            Parameter("n_timepoints", ParamType.INTEGER, "Number of timepoints to evaluate along trajectory", default=10),
            Parameter("evaluation_points", ParamType.ARRAY, "Points to evaluate [n_points, dim]. None = source cells",
                      required=False, default=None),
        ],
        returns="dict",
        returns_description="divergence [n_eval, n_t], bifurcation_score [n_eval], "
        "eigenvalue_real [n_eval, n_t, dim], eigenvalue_imag [n_eval, n_t, dim], "
        "timepoints [n_t], n_saddle_points [n_eval]",
        requires=["FlowModel from flow_within(return_model=True)"],
        modifies_adata=[],
    ),
    "tl.flow_feature_graph": ToolSchema(
        name="tl.flow_feature_graph",
        description="Static feature coupling graph from flow Jacobian. Builds a directed "
        "gene interaction graph by projecting the mean Jacobian into gene space "
        "via PCA loadings, collapsed over time.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with PCA loadings"),
            Parameter("flow_result", ParamType.OBJECT, "FlowWithinResult from pc.tl.flow_within()"),
            Parameter("flow_model", ParamType.OBJECT, "FlowModel (from flow_result['model'] when return_model=True)"),
            Parameter("n_top_genes", ParamType.INTEGER, "Number of top genes by PCA loading", default=200),
            Parameter("n_timepoints", ParamType.INTEGER, "Timepoints for Jacobian averaging", default=20),
            Parameter("n_eval_points", ParamType.INTEGER, "Points to evaluate Jacobian", default=300),
            Parameter("edge_threshold", ParamType.FLOAT, "Min |weight| for edges. Auto = mean + 2*std",
                      required=False, default=None),
            Parameter("random_state", ParamType.INTEGER, "Random seed", default=42),
        ],
        returns="dict",
        returns_description="adjacency_matrix [n_top, n_top], gene_names, gene_indices, "
        "out_centrality, in_centrality, flow_centrality, top_hub_genes, "
        "igraph (optional), hub_genes_per_archetype {k: [genes]}",
        requires=["PCs in adata.varm", "FlowModel from flow_within(return_model=True)"],
        modifies_adata=[],
    ),
    "tl.flow_temporal_feature_graph": ToolSchema(
        name="tl.flow_temporal_feature_graph",
        description="Temporal feature graph with (gene, timepoint) nodes. Retains full "
        "temporal structure with backbone edges and cross-feature edges per timepoint.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with PCA loadings"),
            Parameter("flow_result", ParamType.OBJECT, "FlowWithinResult from pc.tl.flow_within()"),
            Parameter("flow_model", ParamType.OBJECT, "FlowModel (from flow_result['model'] when return_model=True)"),
            Parameter("n_top_genes", ParamType.INTEGER, "Number of top genes by PCA loading", default=200),
            Parameter("n_timepoints", ParamType.INTEGER, "Timepoints for temporal nodes", default=20),
            Parameter("n_eval_points", ParamType.INTEGER, "Points to evaluate Jacobian", default=300),
            Parameter("archetype_pairs", ParamType.ARRAY, "Restrict to cells whose top-2 weights match these pairs", default=None),
            Parameter("random_state", ParamType.INTEGER, "Random seed", default=42),
        ],
        returns="dict",
        returns_description="cross_matrices [n_t], self_expansion [n_top, n_t], gene_names, timepoints, "
        "temporal_centrality [n_top], temporal_profile [n_top, 4], "
        "top_early_genes, top_mid_early_genes, top_mid_late_genes, top_late_genes",
        requires=["PCs in adata.varm", "FlowModel from flow_within(return_model=True)"],
        modifies_adata=[],
    ),
    "tl.flow_significance": ToolSchema(
        name="tl.flow_significance",
        description="Permutation test for flow significance. Uses the original flow_result's "
        "MMD improvement as the observed statistic, then retrains flows on permuted "
        "condition labels to build a null distribution.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("flow_result", ParamType.OBJECT, "FlowWithinResult from pc.tl.flow_within(). Must contain mmd_before and mmd_after"),
            Parameter("n_permutations", ParamType.INTEGER, "Number of label-permuted null models to train", default=100),
            Parameter("n_epochs_per_perm", ParamType.INTEGER, "Epochs per null model", default=200),
            Parameter("statistic", ParamType.STRING, "Test statistic to use", default="mmd"),
            Parameter("solver_method", ParamType.STRING, "ODE solver: 'euler', 'midpoint', 'heun3', 'dopri5'", default="dopri5"),
            Parameter("random_state", ParamType.INTEGER, "Random seed", default=42),
        ],
        returns="dict",
        returns_description="p_value, observed_stat, null_distribution",
        requires=["FlowWithinResult from flow_within()"],
        modifies_adata=[],
    ),
    # =========================================================================
    # v0.5.0: Continuous Characterization (pl)
    # =========================================================================
    "pl.ternary_facet": ToolSchema(
        name="pl.ternary_facet",
        description="Ternary plot for 3 selected archetypes on triangular axes.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with archetype weights"),
            Parameter("archetypes", ParamType.ARRAY, "Tuple of 3 archetype indices", default=[0, 1, 2]),
            Parameter("color_by", ParamType.STRING, "Color by gene, obs column, or obsm column", required=False, default=None),
            Parameter("style", ParamType.STRING, "Plot style: 'scatter' or 'density'", default="scatter"),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="go.Figure",
        returns_description="Ternary scatter or density plot for 3 archetypes",
        requires=["cell_archetype_weights in adata.obsm"],
        modifies_adata=[],
    ),
    "pl.coefficient_heatmap": ToolSchema(
        name="pl.coefficient_heatmap",
        description="Heatmap of vertex coefficients for top features by R^2.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with regression results"),
            Parameter("top_n", ParamType.INTEGER, "Number of top features to display", default=50),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="go.Figure",
        returns_description="Plotly heatmap of features x archetypes",
        requires=["peach_simplex_regression in adata.uns"],
        modifies_adata=[],
    ),
    "pl.r2_barplot": ToolSchema(
        name="pl.r2_barplot",
        description="Bar plot of features ranked by R^2, with optional per-archetype |beta| breakdown.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with regression results"),
            Parameter("top_n", ParamType.INTEGER, "Number of top features to display", default=50),
            Parameter("per_archetype", ParamType.BOOLEAN, "Show per-archetype |beta| grouped bars + global R^2 overlay", default=True),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="go.Figure",
        returns_description="Ranked bar plot of R^2 values with per-archetype coefficient magnitudes",
        requires=["peach_simplex_regression in adata.uns"],
        modifies_adata=[],
    ),
    "pl.pattern_summary": ToolSchema(
        name="pl.pattern_summary",
        description="Bar plot summarizing pattern type counts from feature classification.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with pattern results"),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="go.Figure",
        returns_description="Bar chart of pattern type counts",
        requires=["peach_feature_patterns in adata.uns"],
        modifies_adata=[],
    ),
    "pl.component_scatter": ToolSchema(
        name="pl.component_scatter",
        description="2D PCA scatter colored by GMM component assignment.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with GMM results"),
            Parameter("pca_key", ParamType.STRING, "Key in obsm for PCA coordinates", default="X_pca"),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="go.Figure",
        returns_description="PCA scatter with cells colored by GMM component",
        requires=["peach_gmm in adata.uns", "peach_gmm_labels in adata.obsm"],
        modifies_adata=[],
    ),
    "pl.velocity_quiver": ToolSchema(
        name="pl.velocity_quiver",
        description="2D quiver plot of flow transport directions in PCA space.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("flow_result", ParamType.OBJECT, "FlowWithinResult from pc.tl.flow_within()"),
            Parameter("pca_key", ParamType.STRING, "Key in obsm for PCA coordinates", default="X_pca"),
            Parameter("n_arrows", ParamType.INTEGER, "Number of arrows to draw", default=200),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="go.Figure",
        returns_description="2D quiver plot with arrows from source to transported positions",
        requires=["X_pca in adata.obsm", "FlowWithinResult"],
        modifies_adata=[],
    ),
    "pl.mmd_heatmap": ToolSchema(
        name="pl.mmd_heatmap",
        description="Heatmap of K x K MMD matrix between archetypes.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with MMD results"),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="go.Figure",
        returns_description="Heatmap of MMD values between archetype populations",
        requires=["peach_archetype_mmd in adata.uns"],
        modifies_adata=[],
    ),
    "pl.contrast_volcano": ToolSchema(
        name="pl.contrast_volcano",
        description="Volcano plot for one archetype pair: delta-beta vs -log10(FDR q-value). "
        "Includes 95% CI error bars from Wald SE.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with contrast results"),
            Parameter("pair", ParamType.ARRAY, "Archetype pair (j, k) as [j, k]"),
            Parameter("fdr_threshold", ParamType.FLOAT, "FDR threshold for significance coloring", default=0.05),
            Parameter("n_labels", ParamType.INTEGER, "Number of top features to label on plot", default=10),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="go.Figure",
        returns_description="Volcano plot with delta-beta on x-axis, -log10(q) on y-axis, error bars for 95% CI, top feature labels",
        requires=["peach_archetype_contrasts in adata.uns"],
        modifies_adata=[],
    ),
    "pl.contrast_volcano_grid": ToolSchema(
        name="pl.contrast_volcano_grid",
        description="Small-multiple grid of volcano plots for all pairwise Wald contrasts.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with contrast results"),
            Parameter("fdr_threshold", ParamType.FLOAT, "FDR threshold for significance coloring", default=0.05),
            Parameter("n_labels", ParamType.INTEGER, "Number of top features to label per subplot", default=5),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="go.Figure",
        returns_description="Grid of volcano subplots, one per archetype pair, with top feature labels",
        requires=["peach_archetype_contrasts in adata.uns"],
        modifies_adata=[],
    ),
    "pl.feature_similarity_heatmap": ToolSchema(
        name="pl.feature_similarity_heatmap",
        description="Heatmap of Spearman correlation between archetype beta vectors.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with feature similarity results"),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="go.Figure",
        returns_description="Diverging heatmap of Spearman rho between archetype coefficient vectors",
        requires=["peach_archetype_feature_similarity in adata.uns"],
        modifies_adata=[],
    ),
    "pl.archetype_regression_dotplot": ToolSchema(
        name="pl.archetype_regression_dotplot",
        description="Dotplot of top genes per archetype: dot size = |beta|, dot color = -log10(p). Supports exclusivity filter and degree-2 interactions.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with regression results"),
            Parameter("top_n", ParamType.INTEGER, "Number of top features per archetype", default=10),
            Parameter("exclusive_only", ParamType.BOOLEAN, "Only show features where max coef >= 2x second-highest", default=False),
            Parameter("degree", ParamType.INTEGER, "1 = vertex only, 2 = also show interaction columns", default=1),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="go.Figure",
        returns_description="Dotplot with genes (rows) x archetypes+interactions (cols), size=|beta|, color=-log10(p)",
        requires=["peach_simplex_regression in adata.uns"],
        modifies_adata=[],
    ),
    "pl.component_archetype_summary": ToolSchema(
        name="pl.component_archetype_summary",
        description="2x2 panel: component sizes, weight profiles, archetype proximity, entropy.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with GMM results"),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="go.Figure",
        returns_description="2x2 subplot: bar sizes, weight heatmap, proximity bars, entropy boxes",
        requires=["peach_gmm in adata.uns", "cell_archetype_weights in adata.obsm"],
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
        returns_description="Columns: archetype_1_distance, ..., nearest_archetype, nearest_archetype_distance (1-indexed)",
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
    # =========================================================================
    # tl (TOOLS) - Flow Between & Simplex Regression Wrappers
    # =========================================================================
    "tl.flow_between": ToolSchema(
        name="tl.flow_between",
        description="Inter-model flow between separate AnnDatas. Trains a neural ODE to "
        "transport source cells from one AnnData to target cells in another.",
        parameters=[
            Parameter(
                "adatas",
                ParamType.ARRAY,
                "List of AnnData objects (at least 2) to compute flow between",
            ),
            Parameter("condition_key", ParamType.STRING, "Column in obs identifying conditions", default="condition"),
            Parameter(
                "condition_labels",
                ParamType.ARRAY,
                "Ordered condition labels. None = infer from condition_key",
                required=False,
                default=None,
                items_type=ParamType.STRING,
            ),
            Parameter(
                "pairs",
                ParamType.ARRAY,
                "Specific (source, target) pairs to compute. None = consecutive pairs",
                required=False,
                default=None,
            ),
            Parameter("pca_key", ParamType.STRING, "Key in obsm for PCA coordinates", default="X_pca"),
            Parameter(
                "hidden_dims",
                ParamType.ARRAY,
                "MLP hidden dimensions",
                default=[128, 128, 128],
                items_type=ParamType.INTEGER,
            ),
            Parameter("lr", ParamType.FLOAT, "Learning rate", default=1e-3),
            Parameter("n_epochs", ParamType.INTEGER, "Training epochs", default=1000),
            Parameter("batch_size", ParamType.INTEGER, "Batch size", default=256),
            Parameter("n_steps", ParamType.INTEGER, "ODE integration steps", default=50),
            Parameter("device", ParamType.STRING, "Computing device", default="cpu"),
            Parameter(
                "solver_method",
                ParamType.STRING,
                "ODE solver: 'euler', 'midpoint', 'heun3', 'dopri5'",
                default="dopri5",
            ),
            Parameter("use_ot", ParamType.BOOLEAN, "Use minibatch Sinkhorn OT coupling for training pairs", default=False),
            Parameter("random_state", ParamType.INTEGER, "Random seed", default=42),
        ],
        returns="dict",
        returns_description="Per-pair flow results with transported cells, losses, MMD before/after, "
        "archetype correspondence matrix (K_src x K_tgt)",
        requires=["X_pca in each adata.obsm"],
        modifies_adata=[],
    ),
    "tl.gene_simplex_regression": ToolSchema(
        name="tl.gene_simplex_regression",
        description="Convenience wrapper: simplex regression on adata.X (gene expression). "
        "Calls tl.feature_simplex_regression with feature_matrix=None.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with archetype weights"),
        ],
        returns="dict (serialized SimplexRegressionResult)",
        returns_description="vertex_coefficients [n_genes, K], r_squared_degree1, vertex_pvalues, "
        "interaction_coefficients (optional). Same as feature_simplex_regression.",
        requires=["cell_archetype_weights in adata.obsm"],
        modifies_adata=["uns['peach_simplex_regression']"],
    ),
    "tl.pathway_simplex_regression": ToolSchema(
        name="tl.pathway_simplex_regression",
        description="Convenience wrapper: simplex regression on pathway scores. "
        "Calls tl.feature_simplex_regression with feature_matrix='pathway_scores'.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with archetype weights and pathway scores"),
        ],
        returns="dict (serialized SimplexRegressionResult)",
        returns_description="vertex_coefficients [n_pathways, K], r_squared_degree1, vertex_pvalues, "
        "interaction_coefficients (optional). Same as feature_simplex_regression.",
        requires=["cell_archetype_weights in adata.obsm", "pathway_scores in adata.obsm"],
        modifies_adata=["uns['peach_simplex_regression']"],
    ),
    # =========================================================================
    # tl (TOOLS) - Spatial Analysis (v0.5.0 additions)
    # =========================================================================
    "tl.archetype_spatial_autocorr": ToolSchema(
        name="tl.archetype_spatial_autocorr",
        description="Spatial autocorrelation (Moran's I / Geary's C) per archetype weight. "
        "Measures whether archetype weight distributions are spatially clustered.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with spatial graph and archetype weights"),
            Parameter("weights_key", ParamType.STRING, "Key in obsm for archetype weights", default="cell_archetype_weights"),
            Parameter(
                "mode",
                ParamType.STRING,
                "Autocorrelation measure: 'moran' (Moran's I) or 'geary' (Geary's C)",
                default="moran",
                enum=["moran", "geary"],
            ),
            Parameter("n_perms", ParamType.INTEGER, "Number of permutations for significance", default=100),
            Parameter("n_jobs", ParamType.INTEGER, "Number of parallel jobs", default=1),
        ],
        returns="DataFrame",
        returns_description="Per-archetype spatial autocorrelation statistics (I/C, p-value, z-score)",
        requires=["spatial_connectivities in adata.obsp", "cell_archetype_weights in adata.obsm"],
        modifies_adata=["uns['archetype_spatial_autocorr']"],
    ),
    "tl.archetype_interaction_boundaries": ToolSchema(
        name="tl.archetype_interaction_boundaries",
        description="Detect spatial fronts where archetype compositions diverge between cell types "
        "using Jensen-Shannon Divergence (JSD) on archetype weight vectors.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with spatial coords and archetype weights"),
            Parameter("cell_type_col", ParamType.STRING, "Column in adata.obs with cell type labels", default="Cell_Type"),
            Parameter("weights_key", ParamType.STRING, "Key in obsm for archetype weights", default="cell_archetype_weights"),
            Parameter(
                "cell_type_a",
                ParamType.STRING,
                "First cell type. None = auto-detect",
                required=False,
                default=None,
            ),
            Parameter(
                "cell_type_b",
                ParamType.STRING,
                "Second cell type. None = auto-detect",
                required=False,
                default=None,
            ),
        ],
        returns="dict",
        returns_description="boundary_scores, per-archetype cross-correlations, cell type pair metadata",
        requires=["cell_archetype_weights in adata.obsm", "cell_type_col in adata.obs"],
        modifies_adata=["uns['archetype_interaction_boundaries']"],
    ),
    "tl.archetype_pair_enrichment": ToolSchema(
        name="tl.archetype_pair_enrichment",
        description="Permutation test for spatial co-localization of archetype weight pairs. "
        "Tests whether cells with high weights for two archetypes are spatially proximate.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with spatial coords and archetype weights"),
            Parameter(
                "archetype_pairs",
                ParamType.ARRAY,
                "Specific archetype pairs to test. None = all pairs",
                required=False,
                default=None,
            ),
            Parameter("weight_threshold", ParamType.FLOAT, "Weight threshold for 'high' assignment", default=0.3),
            Parameter("n_permutations", ParamType.INTEGER, "Number of permutations for significance", default=1000),
            Parameter("spatial_key", ParamType.STRING, "Key in obsm for spatial coordinates", default="spatial"),
        ],
        returns="dict",
        returns_description="Per-pair enrichment scores, p-values, observed vs expected co-localization",
        requires=["cell_archetype_weights in adata.obsm", "spatial coordinates in adata.obsm"],
        modifies_adata=[],
    ),
    # =========================================================================
    # pl (PLOTTING) - Flow Visualization (v0.5.0)
    # =========================================================================
    "pl.archetype_correspondence": ToolSchema(
        name="pl.archetype_correspondence",
        description="K_src x K_tgt heatmap for archetype correspondence from flow_between results.",
        parameters=[
            Parameter("flow_between_result", ParamType.OBJECT, "Result dict from tl.flow_between()"),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="go.Figure",
        returns_description="Plotly heatmap of archetype correspondence matrix",
        requires=["flow_between result"],
        modifies_adata=[],
    ),
    "pl.density_comparison": ToolSchema(
        name="pl.density_comparison",
        description="KDE comparison of source vs transported vs target distributions in PC1.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("flow_result", ParamType.OBJECT, "FlowWithinResult from pc.tl.flow_within()"),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="go.Figure",
        returns_description="KDE density curves for source, transported, and target in PC1",
        requires=["X_pca in adata.obsm", "FlowWithinResult"],
        modifies_adata=[],
    ),
    "pl.flow_magnitude": ToolSchema(
        name="pl.flow_magnitude",
        description="2D scatter colored by transport magnitude (displacement norm).",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("flow_result", ParamType.OBJECT, "FlowWithinResult from pc.tl.flow_within()"),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="go.Figure",
        returns_description="2D PCA scatter with cells colored by flow displacement magnitude",
        requires=["X_pca in adata.obsm", "FlowWithinResult"],
        modifies_adata=[],
    ),
    "pl.flow_topo_landscape": ToolSchema(
        name="pl.flow_topo_landscape",
        description="Topographic contour map of feature expression and Jacobian expansion over the flow.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("flow_result", ParamType.OBJECT, "FlowWithinResult from pc.tl.flow_within()"),
            Parameter("flow_model", ParamType.OBJECT, "FlowModel (from flow_result['model'] when return_model=True)"),
            Parameter(
                "features",
                ParamType.ARRAY,
                "Specific feature names to plot. None = auto-select top features",
                required=False,
                default=None,
                items_type=ParamType.STRING,
            ),
            Parameter("n_features", ParamType.INTEGER, "Number of top features to display (if features=None)", default=5),
            Parameter("n_timepoints", ParamType.INTEGER, "Number of timepoints along trajectory", default=20),
            Parameter("n_eval_points", ParamType.INTEGER, "Number of points for Jacobian evaluation", default=300),
            Parameter("n_grid", ParamType.INTEGER, "Grid resolution for contour map", default=80),
            Parameter(
                "feature_type",
                ParamType.STRING,
                "Feature source: 'genes' for adata.X, or obsm key",
                default="genes",
            ),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
            Parameter("save", ParamType.STRING, "Path to save figure", required=False, default=None),
        ],
        returns="matplotlib.figure.Figure",
        returns_description="Topographic contour map with feature expression and Jacobian expansion overlays",
        requires=["FlowModel from flow_within(return_model=True)", "PCs in adata.varm"],
        modifies_adata=[],
    ),
    "pl.gene_alignment_barplot": ToolSchema(
        name="pl.gene_alignment_barplot",
        description="Top aligned/opposed genes bar plot from flow gene alignment analysis.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("alignment_result", ParamType.OBJECT, "Result dict from tl.flow_gene_alignment()"),
            Parameter("n_top", ParamType.INTEGER, "Number of top aligned/opposed genes to show", default=20),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="go.Figure",
        returns_description="Horizontal bar plot of top aligned and opposed genes by alignment score",
        requires=["flow_gene_alignment result"],
        modifies_adata=[],
    ),
    "pl.jacobian_heatmap": ToolSchema(
        name="pl.jacobian_heatmap",
        description="Mean Jacobian matrix heatmap from flow Jacobian analysis.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("jacobian_result", ParamType.OBJECT, "Result dict from tl.flow_jacobian()"),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="go.Figure",
        returns_description="Plotly heatmap of mean Jacobian matrix",
        requires=["flow_jacobian result"],
        modifies_adata=[],
    ),
    "pl.soft_assignment_flow": ToolSchema(
        name="pl.soft_assignment_flow",
        description="Sankey diagram of feature flow between archetype pairs based on soft assignment.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with regression results"),
            Parameter("top_n", ParamType.INTEGER, "Number of top features to include", default=15),
            Parameter(
                "feature_type",
                ParamType.STRING,
                "Feature source: 'genes' for adata.X, or obsm key",
                default="genes",
            ),
            Parameter(
                "pairs",
                ParamType.ARRAY,
                "Specific archetype pairs to show. None = all pairs",
                required=False,
                default=None,
            ),
            Parameter("alpha", ParamType.FLOAT, "FDR significance threshold", default=0.05),
            Parameter("degree", ParamType.INTEGER, "Regression degree for feature selection", default=1),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
            Parameter("save", ParamType.STRING, "Path to save figure", required=False, default=None),
        ],
        returns="go.Figure",
        returns_description="Sankey diagram showing feature flow between archetype pairs",
        requires=["peach_simplex_regression in adata.uns"],
        modifies_adata=[],
    ),
    "pl.soft_assignment_heatmap": ToolSchema(
        name="pl.soft_assignment_heatmap",
        description="Soft archetype assignment correspondence heatmap via kNN matching between two conditions.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData (source)"),
            Parameter("flow_result", ParamType.OBJECT, "FlowWithinResult from pc.tl.flow_within()"),
            Parameter(
                "adata_b_key",
                ParamType.ADATA_REF,
                "Second AnnData (target). None = same adata",
                required=False,
                default=None,
            ),
            Parameter("pca_key", ParamType.STRING, "Key in obsm for PCA coordinates", default="X_pca"),
            Parameter("n_neighbors", ParamType.INTEGER, "Number of nearest neighbors for matching", default=10),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="go.Figure",
        returns_description="Heatmap of soft archetype assignment correspondence between conditions",
        requires=["cell_archetype_weights in adata.obsm", "FlowWithinResult"],
        modifies_adata=[],
    ),
    "pl.trajectory_ribbon": ToolSchema(
        name="pl.trajectory_ribbon",
        description="Transported cells colored by time step, showing flow trajectory evolution.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData"),
            Parameter("flow_result", ParamType.OBJECT, "FlowWithinResult from pc.tl.flow_within()"),
            Parameter(
                "flow_model",
                ParamType.OBJECT,
                "FlowModel for intermediate steps. None = linear interpolation",
                required=False,
                default=None,
            ),
            Parameter("n_sample", ParamType.INTEGER, "Number of cells to sample for trajectory", default=100),
            Parameter("n_steps", ParamType.INTEGER, "Number of time steps to visualize", default=20),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="go.Figure",
        returns_description="2D scatter of sampled cells at multiple time steps, colored by t",
        requires=["X_pca in adata.obsm", "FlowWithinResult"],
        modifies_adata=[],
    ),
    # =========================================================================
    # pl (PLOTTING) - GMM / Decomposition (v0.5.0)
    # =========================================================================
    "pl.component_heatmap": ToolSchema(
        name="pl.component_heatmap",
        description="Per-component feature profiles heatmap from GMM decomposition.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with GMM results"),
            Parameter("top_n", ParamType.INTEGER, "Number of top features per component", default=20),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="go.Figure",
        returns_description="Heatmap of top features per GMM component",
        requires=["peach_gmm in adata.uns"],
        modifies_adata=[],
    ),
    "pl.component_neighborhood_graph": ToolSchema(
        name="pl.component_neighborhood_graph",
        description="2D network graph of GMM components based on PCA proximity.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with GMM results"),
            Parameter("pca_key", ParamType.STRING, "Key in obsm for PCA coordinates", default="X_pca"),
            Parameter(
                "edge_threshold",
                ParamType.FLOAT,
                "Min edge weight to display. None = auto-threshold",
                required=False,
                default=None,
            ),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
            Parameter("save", ParamType.STRING, "Path to save figure", required=False, default=None),
        ],
        returns="go.Figure",
        returns_description="Network graph of GMM components with edges weighted by proximity",
        requires=["peach_gmm in adata.uns", "X_pca in adata.obsm"],
        modifies_adata=[],
    ),
    "pl.component_stability": ToolSchema(
        name="pl.component_stability",
        description="Bar plot of component stability scores from GMM decomposition.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with GMM results"),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="go.Figure",
        returns_description="Bar plot of per-component stability scores",
        requires=["peach_gmm in adata.uns"],
        modifies_adata=[],
    ),
    "pl.gmm_bic_curve": ToolSchema(
        name="pl.gmm_bic_curve",
        description="BIC vs number of components line plot for GMM model selection.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with GMM results"),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="go.Figure",
        returns_description="Line plot of BIC values vs number of components",
        requires=["peach_gmm in adata.uns"],
        modifies_adata=[],
    ),
    # =========================================================================
    # pl (PLOTTING) - Regression & Characterization (v0.5.0)
    # =========================================================================
    "pl.archetype_radar": ToolSchema(
        name="pl.archetype_radar",
        description="Radar plot of archetype phenotype characterization from regression coefficients.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with regression results"),
            Parameter("top_n", ParamType.INTEGER, "Number of top features per archetype to display", default=10),
            Parameter(
                "feature_type",
                ParamType.STRING,
                "Feature source: 'genes' for adata.X, or obsm key",
                default="genes",
            ),
            Parameter("min_degree", ParamType.INTEGER, "Minimum regression degree for feature inclusion", default=1),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
            Parameter("save", ParamType.STRING, "Path to save figure", required=False, default=None),
        ],
        returns="go.Figure",
        returns_description="Radar plot with per-archetype phenotype profiles",
        requires=["peach_simplex_regression in adata.uns"],
        modifies_adata=[],
    ),
    "pl.interaction_heatmap": ToolSchema(
        name="pl.interaction_heatmap",
        description="Interaction coefficients heatmap from degree-2 simplex regression.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with regression results"),
            Parameter("top_n", ParamType.INTEGER, "Number of top interaction features to display", default=50),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="go.Figure",
        returns_description="Heatmap of interaction coefficients (features x archetype pairs)",
        requires=["peach_simplex_regression in adata.uns with max_degree >= 2"],
        modifies_adata=[],
    ),
    "pl.regression_volcano": ToolSchema(
        name="pl.regression_volcano",
        description="R-squared vs max vertex contrast scatter (volcano-style) for regression features.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with regression results"),
            Parameter("alpha", ParamType.FLOAT, "Significance threshold for coloring", default=0.05),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="go.Figure",
        returns_description="Scatter of R^2 vs max vertex contrast, colored by significance",
        requires=["peach_simplex_regression in adata.uns"],
        modifies_adata=[],
    ),
    "pl.vertex_radar": ToolSchema(
        name="pl.vertex_radar",
        description="Spider plot of vertex coefficients for a single feature across all archetypes.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with regression results"),
            Parameter("feature", ParamType.STRING, "Feature name to plot"),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="go.Figure",
        returns_description="Spider/radar plot of vertex regression coefficients for a single feature",
        requires=["peach_simplex_regression in adata.uns"],
        modifies_adata=[],
    ),
    "pl.ternary_facet_grid": ToolSchema(
        name="pl.ternary_facet_grid",
        description="Multiple ternary facets for all or selected archetype triples.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with archetype weights"),
            Parameter(
                "color_by",
                ParamType.STRING,
                "Color by gene, obs column, or obsm column",
                required=False,
                default=None,
            ),
            Parameter(
                "facets",
                ParamType.STRING,
                "Which triples to show: 'all' or specific indices",
                default="all",
            ),
            Parameter("ncols", ParamType.INTEGER, "Number of columns in facet grid", default=3),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="list[go.Figure]",
        returns_description="List of ternary Plotly figures, one per archetype triple",
        requires=["cell_archetype_weights in adata.obsm"],
        modifies_adata=[],
    ),
    # =========================================================================
    # pl (PLOTTING) - Spatial Visualization (v0.5.0)
    # =========================================================================
    "pl.cross_correlations": ToolSchema(
        name="pl.cross_correlations",
        description="Diverging dot plot of per-archetype Spearman cross-correlations "
        "from interaction boundary analysis.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with boundary results"),
            Parameter(
                "uns_key",
                ParamType.STRING,
                "Key in adata.uns for interaction boundary results",
                default="archetype_interaction_boundaries",
            ),
            Parameter("title", ParamType.STRING, "Plot title", required=False, default=None),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="go.Figure",
        returns_description="Diverging dot plot of Spearman rho per archetype weight",
        requires=["archetype_interaction_boundaries in adata.uns"],
        modifies_adata=[],
    ),
    "pl.interaction_boundaries": ToolSchema(
        name="pl.interaction_boundaries",
        description="Spatial map of interaction boundary scores overlaid on tissue coordinates.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with spatial coords and boundary scores"),
            Parameter("spatial_key", ParamType.STRING, "Key in obsm for spatial coordinates", default="spatial"),
            Parameter("score_key", ParamType.STRING, "Key in obs for boundary score", default="boundary_score"),
            Parameter("point_size", ParamType.FLOAT, "Size of scatter points", default=2.0),
            Parameter("colorscale", ParamType.STRING, "Plotly colorscale", default="Inferno"),
            Parameter("title", ParamType.STRING, "Plot title", required=False, default=None),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="go.Figure",
        returns_description="Plotly scatter of cells at spatial positions colored by boundary score",
        requires=["spatial coordinates in adata.obsm", "boundary_score in adata.obs"],
        modifies_adata=[],
    ),
    "pl.spatial_autocorr": ToolSchema(
        name="pl.spatial_autocorr",
        description="Lollipop plot of spatial autocorrelation per archetype weight.",
        parameters=[
            Parameter("adata_key", ParamType.ADATA_REF, "Reference to AnnData with spatial autocorrelation results"),
            Parameter(
                "uns_key",
                ParamType.STRING,
                "Key in adata.uns for spatial autocorrelation results",
                default="archetype_spatial_autocorr",
            ),
            Parameter("title", ParamType.STRING, "Plot title", required=False, default=None),
            Parameter("save_path", ParamType.STRING, "Path to save as HTML", required=False, default=None),
            Parameter("show", ParamType.BOOLEAN, "Display plot", default=True),
        ],
        returns="go.Figure",
        returns_description="Lollipop plot of Moran's I / Geary's C per archetype weight",
        requires=["archetype_spatial_autocorr in adata.uns"],
        modifies_adata=[],
    ),
    # =========================================================================
    # _core (ADVANCED) - Remaining Core Functions
    # =========================================================================
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

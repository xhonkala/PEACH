"""
PHASE 6B: Statistical Testing Framework for Archetypal Analysis
==============================================================

PURPOSE: Comprehensive statistical testing of archetypal associations with genes, pathways, and conditional factors.

ARCHITECTURAL ROLE:
- Extension of Phase 6A pathway analysis with rigorous statistical validation
- Implements 1-vs-all testing framework for archetype characterization
- Provides standardized statistical reporting with effect sizes and FDR correction
- Bridges archetypal discovery with biological hypothesis testing

DESIGN PRINCIPLES:
- 1-vs-all testing paradigm: each archetype vs all other cells
- Multiple testing correction with various FDR methods
- Comprehensive statistical reporting (effect sizes, confidence intervals)
- DataFrame-based API consistent with existing analysis functions
- Scalable implementation for large single-cell datasets

=== MODULE API INVENTORY ===

MAIN FUNCTIONS:
 test_archetype_gene_associations(adata, archetype_assignments, use_layer=None, test_method='wilcoxon', fdr_method='benjamini_hochberg', min_cells=10, verbose=True) -> pd.DataFrame
    Purpose: 1-vs-all Wilcoxon rank-sum tests for gene expression differences per archetype
    Inputs: adata(AnnData), archetype_assignments(DataFrame from bin_cells_by_archetype), use_layer(str optional), test_method(str), fdr_method(str), min_cells(int), verbose(bool)
    Outputs: DataFrame with 'gene', 'archetype', 'n_archetype_cells', 'n_other_cells', 'mean_archetype', 'mean_other', 'log_fold_change', 'statistic', 'pvalue', 'fdr_pvalue', 'significant', 'direction'
    Side Effects: Full dataset processing, statistical computation, multiple testing correction

 test_archetype_pathway_associations(pathway_scores_df, archetype_assignments, test_method='wilcoxon', fdr_method='benjamini_hochberg', min_cells=10, verbose=True) -> pd.DataFrame
    Purpose: 1-vs-all Wilcoxon rank-sum tests for pathway activity differences per archetype
    Inputs: pathway_scores_df(DataFrame from compute_pathway_scores), archetype_assignments(DataFrame), test_method(str), fdr_method(str), min_cells(int), verbose(bool)
    Outputs: DataFrame with 'pathway', 'archetype', 'n_archetype_cells', 'n_other_cells', 'mean_archetype', 'mean_other', 'log_fold_change', 'statistic', 'pvalue', 'fdr_pvalue', 'significant', 'direction'
    Side Effects: Pathway score statistical testing, log fold change computation, FDR correction

 test_archetype_conditional_associations(adata, archetype_assignments, obs_column, test_method='hypergeometric', fdr_method='benjamini_hochberg', min_cells=5, verbose=True) -> pd.DataFrame
    Purpose: Hypergeometric tests for enrichment of archetypes in different conditions/metadata categories using boolean matrix approach
    Inputs: adata(AnnData), archetype_assignments(DataFrame), obs_column(str AnnData.obs column name), test_method(str), fdr_method(str), min_cells(int), verbose(bool)
    Outputs: DataFrame with 'archetype', 'condition', 'observed', 'expected', 'total_archetype', 'total_condition', 'odds_ratio', 'ci_lower', 'ci_upper', 'pvalue', 'fdr_pvalue', 'significant'
    Side Effects: Boolean matrix construction, vectorized overlap computation, hypergeometric testing, odds ratio computation with confidence intervals

 apply_fdr_correction(results_df, pvalue_column='pvalue', method='benjamini_hochberg', alpha=0.05) -> pd.DataFrame
    Purpose: Apply false discovery rate correction to any results DataFrame with p-values
    Inputs: results_df(DataFrame with p-values), pvalue_column(str column name), method(str FDR method), alpha(float significance threshold)
    Outputs: DataFrame with added 'fdr_pvalue' and 'significant' columns
    Side Effects: Multiple testing correction, significance determination

STATISTICAL METHODS:
 Wilcoxon Rank-Sum Test: Non-parametric test for comparing distributions between two groups
 Hypergeometric Test: Exact test for categorical enrichment analysis
 Effect Size Calculation: Log fold change for gene/pathway expression, odds ratios for categorical enrichment
 FDR Correction: Benjamini-Hochberg, Bonferroni, Holm methods available
 Confidence Intervals: Bootstrap-based CIs for robust uncertainty quantification

EXTERNAL DEPENDENCIES:
 From scipy.stats: ranksum (Wilcoxon), hypergeom (hypergeometric), false_discovery_control (FDR)
 From numpy: Statistical functions, array operations, mathematical computations
 From pandas: DataFrame operations, groupby aggregations, statistical summaries
 From typing: Type hints for function signatures and return values

DATA FLOW PATTERNS:
 Gene Testing: AnnData + Archetype assignments → Expression extraction → 1-vs-all tests → FDR correction → Results DataFrame
 Pathway Testing: Pathway scores + Archetype assignments → Score extraction → 1-vs-all tests → FDR correction → Results DataFrame
 Conditional Testing: AnnData.obs + Archetype assignments → Boolean matrices → Vectorized hypergeometric tests → FDR correction → Results DataFrame
 Summary Integration: Multiple test results → Cross-archetype profiling → Biological interpretation → Summary DataFrames

ERROR HANDLING:
 Minimum cell count validation for statistical power
 Missing data handling in expression and metadata
 Degenerate test cases (no variance, empty groups)
 Multiple testing correction edge cases
 Memory management for large-scale testing

BIOLOGICAL INTERPRETATION:
 Effect sizes provide practical significance beyond statistical significance
 1-vs-all framework identifies archetype-specific vs shared features
 Conditional enrichment reveals external factor associations
 Integrated summaries enable systems-level archetype characterization

IMPLEMENTATION NOTES:
 Exclusive patterns: Uses 1-vs-all filtering (NOT pairwise comparisons)
    Features exclusive if significant in only ONE archetype's 1-vs-all test
 Tradeoff patterns: True pairwise comparisons between archetype groups
 Specialization patterns: Each archetype vs archetype_0 (centroid)
 Pattern analysis: Combines multiple testing approaches for comprehensive characterization
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Import statistical testing functions
try:
    from scipy.stats import false_discovery_control, hypergeom, mannwhitneyu
    from scipy.stats.contingency import odds_ratio

    SCIPY_AVAILABLE = True
except ImportError as e:
    SCIPY_AVAILABLE = False
    warnings.warn(f"scipy.stats not available: {e}. Statistical testing functions will not work.")


# =============================================================================
# SHARED UTILITY FUNCTIONS
# =============================================================================


def _standardize_results_dataframe(
    results_df: pd.DataFrame,
    pattern_type: str = None,
    data_type: str = "pathway",  # 'pathway' or 'gene'
) -> pd.DataFrame:
    """
    Standardize results DataFrame to ensure consistent column structure across all pattern analysis functions.

    This ensures all results work with existing visualization functions (dotplots, heatmaps, etc.)
    and enables easy downstream analysis.

    Args:
        results_df: Raw results DataFrame from any pattern analysis
        pattern_type: Type of pattern ('exclusive', 'specialization', 'tradeoff_pair', 'tradeoff_pattern')
        data_type: Whether analyzing 'pathway' or 'gene' data

    Returns
    -------
        DataFrame with standardized column structure
    """
    # Ensure required columns exist
    required_cols = {
        "n_archetype_cells",
        "n_other_cells",
        "mean_archetype",
        "mean_other",
        "statistic",
        "pvalue",
        "fdr_pvalue",
        "significant",
        "direction",
    }

    # Create standardized dataframe
    standardized = results_df.copy()

    # Ensure feature column name is consistent
    if "gene" not in standardized.columns and "pathway" not in standardized.columns:
        if "feature" in standardized.columns:
            if data_type == "gene":
                standardized["gene"] = standardized["feature"]
            else:
                standardized["pathway"] = standardized["feature"]
            standardized = standardized.drop(columns=["feature"])

    # Ensure archetype/pattern column exists
    if "archetype" not in standardized.columns and "pattern" not in standardized.columns:
        if "pattern_name" in standardized.columns:
            standardized["pattern"] = standardized["pattern_name"]
        elif "pattern_code" in standardized.columns:
            standardized["pattern"] = standardized["pattern_code"]

    # Add pattern_type if specified
    if pattern_type is not None:
        standardized["pattern_type"] = pattern_type

    # Ensure effect size columns are properly named
    # IMPORTANT: Always provide BOTH columns for compatibility with existing code
    if data_type == "gene":
        # For genes, primary is log_fold_change
        if "log_fold_change" not in standardized.columns:
            if "effect_size" in standardized.columns:
                standardized["log_fold_change"] = standardized["effect_size"]
            elif "mean_diff" in standardized.columns:
                standardized["log_fold_change"] = standardized["mean_diff"]
        # Also provide mean_diff as alias
        if "mean_diff" not in standardized.columns:
            standardized["mean_diff"] = standardized["log_fold_change"]
    else:  # pathway
        # For pathways, primary is mean_diff
        if "mean_diff" not in standardized.columns:
            if "effect_size" in standardized.columns:
                standardized["mean_diff"] = standardized["effect_size"]
            elif "log_fold_change" in standardized.columns:
                standardized["mean_diff"] = standardized["log_fold_change"]
        # Also provide log_fold_change as alias for backward compatibility
        if "log_fold_change" not in standardized.columns:
            standardized["log_fold_change"] = standardized["mean_diff"]

    # Fill missing standard columns with defaults if needed
    if "n_archetype_cells" not in standardized.columns:
        standardized["n_archetype_cells"] = 0
    if "n_other_cells" not in standardized.columns:
        standardized["n_other_cells"] = 0
    if "mean_archetype" not in standardized.columns:
        standardized["mean_archetype"] = 0.0
    if "mean_other" not in standardized.columns:
        standardized["mean_other"] = 0.0
    if "statistic" not in standardized.columns:
        standardized["statistic"] = 0.0
    if "pvalue" not in standardized.columns:
        standardized["pvalue"] = 1.0
    if "fdr_pvalue" not in standardized.columns:
        standardized["fdr_pvalue"] = 1.0
    if "significant" not in standardized.columns:
        standardized["significant"] = standardized["fdr_pvalue"] < 0.05
    if "direction" not in standardized.columns:
        # Infer direction from effect size (use whichever column exists)
        if "log_fold_change" in standardized.columns:
            standardized["direction"] = standardized["log_fold_change"].apply(lambda x: "higher" if x > 0 else "lower")
        elif "mean_diff" in standardized.columns:
            standardized["direction"] = standardized["mean_diff"].apply(lambda x: "higher" if x > 0 else "lower")
        else:
            standardized["direction"] = "unknown"

    return standardized


def robust_mannwhitneyu_test(
    group1: np.ndarray,
    group2: np.ndarray,
    alternative: str = "two-sided",
    min_nonzero_frac: float = 0.05,
    tie_breaking: bool = True,
    feature_name: str = "feature",
) -> tuple[float, float]:
    """
    Robust Mann-Whitney U test with tie-breaking for sparse single-cell data.

    Args:
        group1: Expression/activity values for group 1 (archetype cells)
        group2: Expression/activity values for group 2 (other cells)
        alternative: Test direction ('greater', 'less', 'two-sided')
        min_nonzero_frac: Minimum fraction of non-zero values required
        tie_breaking: Whether to add small noise to break ties
        feature_name: Name of feature being tested (for debugging)

    Returns
    -------
        Tuple of (statistic, pvalue)

    Raises
    ------
        ValueError: If test fails or data is insufficient
    """
    # Validate inputs
    if len(group1) == 0 or len(group2) == 0:
        raise ValueError(f"Empty groups: group1={len(group1)}, group2={len(group2)}")

    # Check for sufficient non-zero values (important for sparse single-cell data)
    nonzero_frac1 = np.count_nonzero(group1) / len(group1)
    nonzero_frac2 = np.count_nonzero(group2) / len(group2)

    if nonzero_frac1 < min_nonzero_frac and nonzero_frac2 < min_nonzero_frac:
        # Both groups are too sparse - return non-significant result
        return 0.0, 1.0

    # Skip features with no variance in both groups
    if np.var(group1) == 0 and np.var(group2) == 0:
        return 0.0, 1.0

    # Apply tie-breaking if requested
    if tie_breaking:
        # Calculate appropriate noise scale
        combined_data = np.concatenate([group1, group2])
        data_std = np.std(combined_data)

        if data_std > 0:
            # Add very small noise (1e-8 of data standard deviation)
            noise_scale = data_std * 1e-8

            # Generate reproducible noise based on data content for consistency
            np.random.seed(hash(feature_name) % 2**32)
            noise1 = np.random.normal(0, noise_scale, len(group1))
            noise2 = np.random.normal(0, noise_scale, len(group2))

            group1_jittered = group1 + noise1
            group2_jittered = group2 + noise2
        else:
            # If no variance, use original data
            group1_jittered = group1
            group2_jittered = group2
    else:
        group1_jittered = group1
        group2_jittered = group2

    # Perform Mann-Whitney U test
    try:
        statistic, pvalue = mannwhitneyu(group1_jittered, group2_jittered, alternative=alternative)
        return float(statistic), float(pvalue)
    except Exception as e:
        raise ValueError(f"Mann-Whitney U test failed for {feature_name}: {e}")


def test_archetype_gene_associations(
    adata,
    bin_prop: float = 0.1,
    obsm_key: str = "archetype_distances",
    obs_key: str = "archetypes",
    use_layer: str | None = None,
    test_method: str = "mannwhitneyu",
    fdr_method: str = "benjamini_hochberg",
    fdr_scope: str = "global",
    test_direction: str = "two-sided",
    min_logfc: float = 0.01,
    min_cells: int = 10,
    comparison_group: str = "all",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Perform 1-vs-all statistical tests for gene expression differences per archetype.

    Tests whether genes are differentially expressed between cells assigned to
    each archetype versus all other cells using Mann-Whitney U tests.

    Parameters
    ----------
    adata : AnnData
        AnnData object with gene expression data, archetype distances, and assignments.

        Required keys:

        - ``obsm[obsm_key]`` : Distance matrix [n_cells, n_archetypes]
        - ``obs[obs_key]`` : Archetype assignments (from bin_cells_by_archetype)
        - ``X`` or ``layers[use_layer]`` : Gene expression data
    bin_prop : float, default: 0.1
        Proportion of cells closest to each archetype to use (e.g., 0.1 = 10%).
    obsm_key : str, default: 'archetype_distances'
        Key for distance matrix in adata.obsm.
    obs_key : str, default: 'archetypes'
        Key for archetype assignments in adata.obs.
    use_layer : str | None, default: None
        AnnData layer to use for expression data. If None, uses adata.X.
        Auto-selects 'logcounts' or 'log1p' if available for optimal statistics.
    test_method : str, default: 'mannwhitneyu'
        Statistical test method. Currently supports 'mannwhitneyu'.
    fdr_method : str, default: 'benjamini_hochberg'
        Multiple testing correction method: 'benjamini_hochberg' or 'bonferroni'.
    fdr_scope : str, default: 'global'
        FDR correction scope:

        - ``'global'`` : Correct across all tests (most stringent)
        - ``'per_archetype'`` : Correct within each archetype
        - ``'none'`` : No FDR correction (use raw p-values)
    test_direction : str, default: 'two-sided'
        Test direction: 'two-sided' (recommended), 'greater', or 'less'.
    min_logfc : float, default: 0.01
        Minimum absolute log fold change threshold for filtering results.
    min_cells : int, default: 10
        Minimum cells per archetype required for testing.
    comparison_group : str, default: 'all'
        Comparison group for statistical tests:

        - ``'all'`` : Compare archetype cells vs ALL other cells (default)
        - ``'archetypes_only'`` : Compare vs cells assigned to other archetypes
          (excludes archetype_0 and no_archetype cells)
    verbose : bool, default: True
        Whether to print testing progress.

    Returns
    -------
    pd.DataFrame
        Gene association results with columns:

        - ``gene`` : str - Gene symbol/identifier
        - ``archetype`` : str - Archetype identifier (e.g., 'archetype_1')
        - ``n_archetype_cells`` : int - Number of cells in archetype bin
        - ``n_other_cells`` : int - Number of cells in comparison group
        - ``mean_archetype`` : float - Mean expression in archetype cells
        - ``mean_other`` : float - Mean expression in other cells
        - ``log_fold_change`` : float - Log fold change (archetype vs others).
          For log-transformed data: simple difference.
          For raw counts: log2((mean_arch + 1) / (mean_other + 1)).
        - ``statistic`` : float - Mann-Whitney U test statistic
        - ``pvalue`` : float - Raw p-value from statistical test
        - ``test_direction`` : str - Direction of test performed ('two-sided')
        - ``direction`` : str - Effect direction ('higher' or 'lower' in archetype)
        - ``passes_lfc_threshold`` : bool - Whether result passes min_logfc threshold
        - ``fdr_pvalue`` : float - FDR-corrected p-value (after correction)
        - ``significant`` : bool - Whether statistically significant (FDR < 0.05)

    Raises
    ------
    ValueError
        If required keys not found in adata or no valid tests performed.
    ImportError
        If scipy.stats is not available.

    See Also
    --------
    test_archetype_pathway_associations : Pathway-level testing
    peach.tl.gene_associations : User-facing wrapper
    peach._core.types.GeneAssociationResult : Result row type definition

    Examples
    --------
    >>> # Basic workflow
    >>> compute_archetype_distances(model, adata, dataloader)
    >>> bin_cells_by_archetype(adata, percentage_per_archetype=0.1)
    >>> gene_results = test_archetype_gene_associations(adata)
    >>> # Use specific layer
    >>> gene_results = test_archetype_gene_associations(adata, use_layer="logcounts")
    >>> # Top genes per archetype
    >>> for archetype in gene_results["archetype"].unique():
    ...     arch_genes = gene_results[
    ...         (gene_results["archetype"] == archetype)
    ...         & (gene_results["significant"])
    ...         & (gene_results["direction"] == "higher")
    ...     ].nlargest(10, "log_fold_change")
    ...     print(f"{archetype}: {arch_genes['gene'].tolist()}")
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy.stats is required for statistical testing.")

    # Validate comparison_group parameter
    valid_comparison_groups = ["all", "archetypes_only"]
    if comparison_group not in valid_comparison_groups:
        raise ValueError(f"comparison_group must be one of {valid_comparison_groups}, got '{comparison_group}'")

    if verbose:
        print("🧪 Testing archetype-gene associations (AnnData-centric)...")
        print(f"   Method: {test_method}")
        print(f"   FDR correction: {fdr_method} ({fdr_scope} scope)")
        print(f"   Test direction: {test_direction}")
        print(f"   Bin proportion: {bin_prop} (closest cells to each archetype)")
        print(f"   Minimum cells per archetype: {min_cells}")
        print(f"   Comparison group: {comparison_group}")

    # SMART LAYER SELECTION: Prefer log-transformed data for statistical rigor
    if use_layer is None:
        if "logcounts" in adata.layers:
            use_layer = "logcounts"
            if verbose:
                print("    Auto-selected 'logcounts' layer for optimal statistical testing")
        elif "log1p" in adata.layers:
            use_layer = "log1p"
            if verbose:
                print("    Auto-selected 'log1p' layer for statistical testing")
        else:
            if verbose:
                print("   [WARNING]  Using adata.X - assuming data is already log-transformed")
    else:
        if verbose:
            print(f"   [STATS] Using specified layer: {use_layer}")

    # CRITICAL: Validate AnnData contains required data
    if obsm_key not in adata.obsm:
        raise ValueError(
            f"Distance matrix not found in adata.obsm['{obsm_key}']. Run compute_archetype_distances() first."
        )

    if obs_key not in adata.obs.columns:
        raise ValueError(
            f"Archetype assignments not found in adata.obs['{obs_key}']. Run bin_cells_by_archetype() first."
        )

    distance_matrix = adata.obsm[obsm_key]  # [n_cells, n_archetypes]
    archetype_assignments = adata.obs[obs_key]  # Categorical series
    n_cells, n_archetypes = distance_matrix.shape

    if verbose:
        print("   [OK] AnnData validation passed:")
        print(f"      Distance matrix: {distance_matrix.shape} from adata.obsm['{obsm_key}']")
        print(f"      Archetype assignments: {len(archetype_assignments)} cells from adata.obs['{obs_key}']")
        if hasattr(archetype_assignments, "cat"):
            print(f"      Assignment categories: {list(archetype_assignments.cat.categories)}")
        else:
            print(f"      Assignment unique values: {list(archetype_assignments.unique())}")

    # Validate alignment
    if len(adata.obs) != n_cells:
        raise ValueError(
            f"Data alignment error: adata.obs has {len(adata.obs)} cells but distance matrix has {n_cells} cells"
        )

    if len(archetype_assignments) != n_cells:
        raise ValueError(
            f"Assignment alignment error: archetype assignments have {len(archetype_assignments)} cells "
            f"but distance matrix has {n_cells} cells"
        )

    # Get expression data and optimize for row access
    if use_layer is None:
        expr_data = adata.X
        if verbose:
            print("   Using adata.X")
    else:
        if use_layer not in adata.layers:
            raise ValueError(f"Layer '{use_layer}' not found in adata.layers. Available: {list(adata.layers.keys())}")
        expr_data = adata.layers[use_layer]
        if verbose:
            print(f"   Using layer: {use_layer}")

    # Check if sparse and optimize format for row access
    is_sparse = sp.issparse(expr_data)
    n_expr_cells, n_genes = expr_data.shape

    if verbose:
        print(f"   Expression data: {n_expr_cells} cells × {n_genes} genes")
        if is_sparse:
            print(f"   Sparse matrix: {100 * (1 - expr_data.nnz / (n_expr_cells * n_genes)):.1f}% zeros")
            print(f"   Original format: {expr_data.format}")

    # Final validation: expression data must match AnnData cell count
    if n_expr_cells != n_cells:
        raise ValueError(f"Expression data has {n_expr_cells} cells but AnnData has {n_cells} cells")

    # OPTIMIZATION: Convert to CSR format for efficient row access
    if is_sparse and expr_data.format != "csr":
        if verbose:
            print("    Converting to CSR format for efficient row access...")
        import time

        start_time = time.time()
        expr_data = expr_data.tocsr()
        conversion_time = time.time() - start_time
        if verbose:
            print(f"   [OK] CSR conversion completed in {conversion_time:.2f} seconds")

    # ANNDATA-CENTRIC BINNING: Use archetype assignments
    # ================================================
    # Get unique archetype labels (excluding 'no_archetype')
    # Handle both categorical and non-categorical columns
    if hasattr(archetype_assignments, "cat"):
        unique_archetypes = [cat for cat in archetype_assignments.cat.categories if cat != "no_archetype"]
    else:
        unique_archetypes = [
            arch for arch in archetype_assignments.unique() if arch != "no_archetype" and pd.notna(arch)
        ]

    if verbose:
        print("    Using AnnData archetype assignments for binning...")
        print(f"      Found {len(unique_archetypes)} archetype categories: {unique_archetypes}")

        # Show assignment distribution
        assignment_counts = archetype_assignments.value_counts()
        for archetype, count in assignment_counts.items():
            print(f"         {archetype}: {count} cells ({100 * count / len(archetype_assignments):.1f}%)")

    # Create cell bins based on assignments
    archetype_bins = {}
    for archetype_label in unique_archetypes:
        # Get all cells assigned to this archetype
        archetype_mask = archetype_assignments == archetype_label
        archetype_indices = np.where(archetype_mask)[0]  # 0-based positions

        if len(archetype_indices) >= min_cells:
            archetype_bins[archetype_label] = archetype_indices
            if verbose:
                print(f"      {archetype_label}: {len(archetype_indices)} assigned cells")
        else:
            if verbose:
                print(f"      {archetype_label}: {len(archetype_indices)} cells (< min_cells={min_cells}) - SKIPPING")

    if not archetype_bins:
        raise ValueError(f"No archetype bins meet minimum cell requirement ({min_cells} cells)")

    if verbose:
        print(f"   [OK] Assignment-based binning completed ({len(archetype_bins)} valid archetypes)")

    # Perform statistical tests
    results = []

    for archetype_name, arch_cell_indices in archetype_bins.items():
        if verbose:
            print(f"\n   Testing {archetype_name}...")

        # Get comparison cells based on comparison_group parameter
        if comparison_group == "all":
            # Get ALL other cells (1-vs-all approach - default behavior)
            all_cell_indices = np.arange(n_cells)
            other_cell_indices = np.setdiff1d(all_cell_indices, arch_cell_indices)
        elif comparison_group == "archetypes_only":
            # Get only cells assigned to OTHER archetypes (exclude archetype_0 and no_archetype)
            other_archetype_mask = (
                (archetype_assignments != archetype_name)
                & (archetype_assignments != "archetype_0")
                & (archetype_assignments != "no_archetype")
            )
            other_cell_indices = np.where(other_archetype_mask)[0]

        if verbose:
            print(f"      Archetype cells: {len(arch_cell_indices)}, Comparison cells: {len(other_cell_indices)}")

        if len(arch_cell_indices) < min_cells or len(other_cell_indices) < min_cells:
            if verbose:
                print(
                    f"      Skipping {archetype_name}: arch={len(arch_cell_indices)}, other={len(other_cell_indices)} (min: {min_cells})"
                )
            continue

        # PERFORMANCE FIX #1: Extract ALL genes at once (massive speedup!)
        if verbose:
            print(
                f"       Extracting expression data for {len(arch_cell_indices)} archetype + {len(other_cell_indices)} other cells..."
            )

        if is_sparse:
            # Extract ALL genes for archetype cells at once (single sparse operation)
            arch_data = expr_data[arch_cell_indices, :].toarray()  # Shape: [n_arch_cells, n_genes]
            other_data = expr_data[other_cell_indices, :].toarray()  # Shape: [n_other_cells, n_genes]
        else:
            # For dense matrices, direct slicing
            arch_data = expr_data[arch_cell_indices, :]
            other_data = expr_data[other_cell_indices, :]

        if verbose:
            print(f"      [OK] Expression extraction completed - testing {n_genes} genes...")

        # Test each gene using pre-extracted data
        n_genes_tested = 0
        for gene_idx in range(n_genes):
            # Extract expression values for this gene (now just array indexing!)
            arch_expr = arch_data[:, gene_idx]  # Simple column access
            other_expr = other_data[:, gene_idx]  # Simple column access

            gene_name = adata.var.index[gene_idx]

            # Skip genes with no variance
            if np.var(arch_expr) == 0 and np.var(other_expr) == 0:
                continue

            # Perform statistical test
            try:
                # SIMPLIFIED: All methods use the same robust Mann-Whitney U test
                # if test_method.lower() in ['wilcoxon', 'wilcox', 'mannwhitneyu', 'ranksum']:
                if test_method.lower() in ["mannwhitneyu"]:
                    # All aliases use the same two-sided Mann-Whitney U test with tie-breaking
                    pass  # Continue to unified testing below
                else:
                    raise ValueError(
                        f"Unknown test method: {test_method}. Supported: 'wilcoxon', 'wilcox', 'mannwhitneyu', 'ranksum'"
                    )

                # Calculate log fold change (now the only effect size measure)
                mean_arch = np.mean(arch_expr)
                mean_other = np.mean(other_expr)

                # SMART LOG FOLD CHANGE CALCULATION
                # For known log-transformed layers, always use simple difference
                if use_layer in ["logcounts", "log1p"]:
                    # Data is already in log space - simple difference is correct
                    log_fold_change = mean_arch - mean_other
                else:
                    # For unknown data types, detect transformation status
                    combined_expr = np.concatenate([arch_expr, other_expr])
                    max_val = np.max(combined_expr)
                    min_val = np.min(combined_expr[combined_expr > 0]) if np.any(combined_expr > 0) else 0

                    # If data appears to be raw counts (large values), use log2 ratio
                    # If data appears log-transformed (small values, negative values), use difference
                    if max_val > 50 or (min_val > 0 and max_val / min_val > 1000):
                        # Raw count data - use log2 ratio
                        if mean_other > 0:
                            log_fold_change = np.log2((mean_arch + 1) / (mean_other + 1))
                        else:
                            log_fold_change = np.log2(mean_arch + 1)
                    else:
                        # Log-transformed data - use simple difference
                        log_fold_change = mean_arch - mean_other

                # UNIFIED APPROACH: Always use two-sided test (most robust)
                # Convert legacy directions to two-sided for consistency
                if test_direction in ["two-sided", "both", "greater", "less"]:
                    # All directions now use two-sided test for robustness
                    statistic, pvalue = robust_mannwhitneyu_test(
                        arch_expr, other_expr, alternative="two-sided", tie_breaking=True, feature_name=gene_name
                    )

                    # Apply LFC threshold filter
                    passes_lfc = abs(log_fold_change) >= min_logfc

                    if passes_lfc:
                        # Determine direction based on mean comparison
                        direction = "higher" if mean_arch > mean_other else "lower"

                        results.append(
                            {
                                "gene": gene_name,
                                "archetype": archetype_name,
                                "n_archetype_cells": len(arch_cell_indices),
                                "n_other_cells": len(other_cell_indices),
                                "mean_archetype": mean_arch,
                                "mean_other": mean_other,
                                "log_fold_change": log_fold_change,
                                "statistic": statistic,
                                "pvalue": pvalue,
                                "test_direction": "two-sided",
                                "direction": direction,
                                "passes_lfc_threshold": True,
                            }
                        )

                # Legacy handling (deprecated but preserved for compatibility)
                elif test_direction == "both_legacy":
                    # Test greater direction with robust test
                    stat_greater, pval_greater = robust_mannwhitneyu_test(
                        arch_expr, other_expr, alternative="greater", tie_breaking=True, feature_name=gene_name
                    )
                    results.append(
                        {
                            "gene": gene_name,
                            "archetype": archetype_name,
                            "n_archetype_cells": len(arch_cell_indices),
                            "n_other_cells": len(other_cell_indices),
                            "mean_archetype": mean_arch,
                            "mean_other": mean_other,
                            "log_fold_change": log_fold_change,
                            "statistic": stat_greater,
                            "pvalue": pval_greater,
                            "test_direction": "greater",
                            "direction": "higher",  # Greater test always means higher
                        }
                    )

                    # Test less direction with robust test
                    stat_less, pval_less = robust_mannwhitneyu_test(
                        arch_expr, other_expr, alternative="less", tie_breaking=True, feature_name=gene_name
                    )
                    results.append(
                        {
                            "gene": gene_name,
                            "archetype": archetype_name,
                            "n_archetype_cells": len(arch_cell_indices),
                            "n_other_cells": len(other_cell_indices),
                            "mean_archetype": mean_arch,
                            "mean_other": mean_other,
                            "log_fold_change": log_fold_change,
                            "statistic": stat_less,
                            "pvalue": pval_less,
                            "test_direction": "less",
                            "direction": "lower",  # Less test always means lower
                        }
                    )
                else:
                    # Single direction test with LFC filtering
                    statistic, pvalue = robust_mannwhitneyu_test(
                        arch_expr, other_expr, alternative=test_direction, tie_breaking=True, feature_name=gene_name
                    )

                    # Apply appropriate LFC threshold filter
                    passes_lfc = False
                    if test_direction == "greater" and log_fold_change >= min_logfc:
                        passes_lfc = True
                    elif test_direction == "less" and log_fold_change <= -min_logfc:
                        passes_lfc = True

                    if passes_lfc:
                        # Determine direction based on test direction
                        direction = "higher" if test_direction == "greater" else "lower"

                        results.append(
                            {
                                "gene": gene_name,
                                "archetype": archetype_name,
                                "n_archetype_cells": len(arch_cell_indices),
                                "n_other_cells": len(other_cell_indices),
                                "mean_archetype": mean_arch,
                                "mean_other": mean_other,
                                "log_fold_change": log_fold_change,
                                "statistic": statistic,
                                "pvalue": pvalue,
                                "test_direction": test_direction
                                if test_direction in ["greater", "less"]
                                else "greater",
                                "direction": direction,
                                "passes_lfc_threshold": True,
                            }
                        )

                n_genes_tested += 1

            except Exception as e:
                if verbose and n_genes_tested == 0:  # Only warn once per archetype
                    print(f"      Warning: Statistical test failed for some genes: {e}")
                continue

        if verbose:
            print(f"      Tested {n_genes_tested} genes")

    if not results:
        raise ValueError("No valid tests performed. Check minimum cell requirements and data quality.")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Apply FDR correction based on scope
    if verbose:
        print(f"\n   Applying FDR correction ({fdr_method}, {fdr_scope} scope)...")
        print(f"   Total tests performed: {len(results_df)}")

    # Apply FDR correction based on scope
    if fdr_scope == "global":
        # Global FDR correction across all tests
        results_df = apply_fdr_correction(
            results_df, pvalue_column="pvalue", method=fdr_method, alpha=0.05, verbose=verbose
        )
        if verbose:
            n_significant = results_df["significant"].sum()
            print(f"    Applied global FDR correction: {n_significant}/{len(results_df)} significant")

    elif fdr_scope == "per_archetype":
        # Per-archetype FDR correction
        results_df["fdr_pvalue"] = np.nan
        results_df["significant"] = False

        for archetype in results_df["archetype"].unique():
            arch_mask = results_df["archetype"] == archetype
            arch_results = results_df[arch_mask].copy()

            if len(arch_results) > 0:
                arch_results = apply_fdr_correction(
                    arch_results, pvalue_column="pvalue", method=fdr_method, alpha=0.05, verbose=False
                )
                results_df.loc[arch_mask, "fdr_pvalue"] = arch_results["fdr_pvalue"]
                results_df.loc[arch_mask, "significant"] = arch_results["significant"]

        if verbose:
            n_significant = results_df["significant"].sum()
            print(f"    Applied per-archetype FDR correction: {n_significant}/{len(results_df)} significant")

    elif fdr_scope == "none":
        # No FDR correction - use raw p-values
        results_df["fdr_pvalue"] = results_df["pvalue"]
        results_df["significant"] = results_df["pvalue"] < 0.05

        if verbose:
            n_significant = results_df["significant"].sum()
            print(f"    No FDR correction applied: {n_significant}/{len(results_df)} significant (raw p<0.05)")

    else:
        raise ValueError(f"Invalid fdr_scope: {fdr_scope}. Must be 'global', 'per_archetype', or 'none'")

    # Sort by significance and log fold change (absolute value for effect size)
    # Create a temporary column for absolute log fold change sorting
    results_df["abs_log_fold_change"] = results_df["log_fold_change"].abs()
    results_df = results_df.sort_values(["fdr_pvalue", "pvalue", "abs_log_fold_change"], ascending=[True, True, False])
    results_df = results_df.drop(columns=["abs_log_fold_change"])

    if verbose:
        n_significant = results_df["significant"].sum()
        total_tests = len(results_df)
        print("[OK] Gene association testing completed!")
        print(f"   Total tests: {total_tests}")
        print(f"   Significant associations: {n_significant} ({100 * n_significant / total_tests:.1f}%)")

        # Show significance breakdown
        if "test_direction" in results_df.columns:
            direction_counts = results_df[results_df["significant"]].groupby("test_direction").size()
            if len(direction_counts) > 0:
                print(f"   Significant by direction: {dict(direction_counts)}")

        # Show FDR correction impact
        raw_significant = (results_df["pvalue"] < 0.05).sum()
        print(f"   Raw significant (p<0.05): {raw_significant}")
        if fdr_scope != "none":
            print(f"   FDR correction impact: {raw_significant} → {n_significant}")

        # Show top associations per archetype
        print("\n[STATS] Top significant associations per archetype:")
        for archetype in results_df["archetype"].unique():
            arch_results = results_df[(results_df["archetype"] == archetype) & (results_df["significant"])].head(3)

            print(f"   {archetype}:")
            for _, row in arch_results.iterrows():
                # Use the new direction column if available, otherwise fall back to log_fold_change
                if "direction" in row:
                    direction_arrow = "↑" if row["direction"] == "higher" else "↓"
                    direction_text = f" ({row['direction']} in archetype)"
                else:
                    direction_arrow = "↑" if row["log_fold_change"] > 0 else "↓"
                    direction_text = ""

                test_dir = f" ({row['test_direction']})" if "test_direction" in row else ""
                print(
                    f"      {direction_arrow} {row['gene']}: FC={row['log_fold_change']:.2f}, "
                    f"p={row['fdr_pvalue']:.2e}{direction_text}{test_dir}"
                )

    return results_df


def test_archetype_pathway_associations(
    adata,
    pathway_obsm_key: str = "pathway_scores",
    obsm_key: str = "archetype_distances",
    obs_key: str = "archetypes",
    test_method: str = "mannwhitneyu",
    fdr_method: str = "benjamini_hochberg",
    fdr_scope: str = "global",
    test_direction: str = "two-sided",
    min_logfc: float = 0.01,
    min_cells: int = 10,
    comparison_group: str = "all",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Perform 1-vs-all statistical tests for pathway activity differences per archetype.

    Tests whether pathway scores are differentially active between cells assigned
    to each archetype versus all other cells using Mann-Whitney U tests.

    Parameters
    ----------
    adata : AnnData
        AnnData object with pathway scores, archetype distances, and assignments.

        Required keys:

        - ``obsm[pathway_obsm_key]`` : Pathway scores [n_cells, n_pathways]
        - ``obs[obs_key]`` : Archetype assignments
        - ``uns[pathway_obsm_key + '_pathways']`` : Pathway names (optional)
    pathway_obsm_key : str, default: 'pathway_scores'
        Key for pathway scores in adata.obsm.
    obsm_key : str, default: 'archetype_distances'
        Key for distance matrix in adata.obsm.
    obs_key : str, default: 'archetypes'
        Key for archetype assignments in adata.obs.
    test_method : str, default: 'mannwhitneyu'
        Statistical test method. Currently supports 'mannwhitneyu'.
    fdr_method : str, default: 'benjamini_hochberg'
        Multiple testing correction method.
    fdr_scope : str, default: 'global'
        FDR correction scope: 'global', 'per_archetype', or 'none'.
    test_direction : str, default: 'two-sided'
        Test direction: 'two-sided' (recommended), 'greater', or 'less'.
    min_logfc : float, default: 0.01
        Minimum effect size threshold (mean_diff for pathways).
    min_cells : int, default: 10
        Minimum cells per archetype for testing.
    comparison_group : str, default: 'all'
        Comparison group: 'all' or 'archetypes_only'.
    verbose : bool, default: True
        Whether to print testing progress.

    Returns
    -------
    pd.DataFrame
        Pathway association results with columns:

        - ``pathway`` : str - Pathway name/identifier
        - ``archetype`` : str - Archetype identifier
        - ``n_archetype_cells`` : int - Number of cells in archetype
        - ``n_other_cells`` : int - Number of cells in comparison group
        - ``mean_archetype`` : float - Mean pathway score in archetype
        - ``mean_other`` : float - Mean pathway score in other cells
        - ``mean_diff`` : float - Mean difference (primary effect size for pathways).
          More appropriate than log fold change for activity scores.
        - ``log_fold_change`` : float - Alias for mean_diff (backward compatibility)
        - ``statistic`` : float - Mann-Whitney U test statistic
        - ``pvalue`` : float - Raw p-value
        - ``test_direction`` : str - Direction of test performed ('two-sided')
        - ``direction`` : str - Effect direction ('higher' or 'lower')
        - ``passes_lfc_threshold`` : bool - Whether result passes effect size threshold
        - ``fdr_pvalue`` : float - FDR-corrected p-value
        - ``significant`` : bool - Whether statistically significant

    Raises
    ------
    ValueError
        If required keys not found in adata or no valid tests performed.

    Notes
    -----
    Pathway scores (from AUCell, pySCENIC, etc.) represent activity levels,
    not expression counts. Mean difference (``mean_diff``) is more interpretable
    than log fold change for these scores. The ``log_fold_change`` column is
    provided as an alias for backward compatibility.

    See Also
    --------
    test_archetype_gene_associations : Gene-level testing
    peach.tl.pathway_associations : User-facing wrapper
    peach._core.types.PathwayAssociationResult : Result row type definition

    Examples
    --------
    >>> pathway_results = test_archetype_pathway_associations(adata)
    >>> # Filter by pathway category
    >>> metabolism = pathway_results[pathway_results["pathway"].str.contains("METABOLISM", case=False)]
    >>> # Top pathways per archetype
    >>> for archetype in pathway_results["archetype"].unique():
    ...     top = pathway_results[
    ...         (pathway_results["archetype"] == archetype) & (pathway_results["significant"])
    ...     ].nlargest(5, "mean_diff")
    ...     print(f"{archetype}: {top['pathway'].tolist()}")
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy.stats is required for statistical testing.")

    # Validate comparison_group parameter
    valid_comparison_groups = ["all", "archetypes_only"]
    if comparison_group not in valid_comparison_groups:
        raise ValueError(f"comparison_group must be one of {valid_comparison_groups}, got '{comparison_group}'")

    if verbose:
        print("🧪 Testing archetype-pathway associations (AnnData-centric)...")
        print("   Method: mannwhitneyu")  # Accurate method name
        print(f"   FDR correction: {fdr_method} ({fdr_scope} scope)")
        print(f"   Test direction: {test_direction}")
        print(f"   Minimum cells per archetype: {min_cells}")
        print(f"   Comparison group: {comparison_group}")

    # CRITICAL: Validate AnnData contains required data
    if pathway_obsm_key not in adata.obsm:
        raise ValueError(
            f"Pathway scores not found in adata.obsm['{pathway_obsm_key}']. Run compute_pathway_scores() first."
        )

    if obs_key not in adata.obs.columns:
        raise ValueError(
            f"Archetype assignments not found in adata.obs['{obs_key}']. Run bin_cells_by_archetype() first."
        )

    pathway_scores = adata.obsm[pathway_obsm_key]  # [n_cells, n_pathways]
    archetype_assignments = adata.obs[obs_key]  # Categorical series
    n_cells, n_pathways = pathway_scores.shape

    if verbose:
        print("   [OK] AnnData validation passed:")
        print(f"      Pathway scores: {pathway_scores.shape} from adata.obsm['{pathway_obsm_key}']")
        print(f"      Archetype assignments: {len(archetype_assignments)} cells from adata.obs['{obs_key}']")
        if hasattr(archetype_assignments, "cat"):
            print(f"      Assignment categories: {list(archetype_assignments.cat.categories)}")
        else:
            print(f"      Assignment unique values: {list(archetype_assignments.unique())}")

    # Validate alignment
    if len(adata.obs) != n_cells:
        raise ValueError(
            f"Data alignment error: adata.obs has {len(adata.obs)} cells but pathway scores have {n_cells} cells"
        )

    if len(archetype_assignments) != n_cells:
        raise ValueError(
            f"Assignment alignment error: archetype assignments have {len(archetype_assignments)} cells "
            f"but pathway scores have {n_cells} cells"
        )

    # ANNDATA-CENTRIC BINNING: Use archetype assignments (same as gene testing)
    # ================================================
    # Get unique archetype labels (excluding 'no_archetype')
    # Handle both categorical and non-categorical columns
    if hasattr(archetype_assignments, "cat"):
        unique_archetypes = [cat for cat in archetype_assignments.cat.categories if cat != "no_archetype"]
    else:
        unique_archetypes = [
            arch for arch in archetype_assignments.unique() if arch != "no_archetype" and pd.notna(arch)
        ]

    if verbose:
        print("    Using AnnData archetype assignments for binning...")
        print(f"      Found {len(unique_archetypes)} archetype categories: {unique_archetypes}")

        # Show assignment distribution
        assignment_counts = archetype_assignments.value_counts()
        for archetype, count in assignment_counts.items():
            print(f"         {archetype}: {count} cells ({100 * count / len(archetype_assignments):.1f}%)")

    # Create cell bins based on assignments
    archetype_bins = {}
    for archetype_label in unique_archetypes:
        # Get all cells assigned to this archetype
        archetype_mask = archetype_assignments == archetype_label
        archetype_indices = np.where(archetype_mask)[0]  # 0-based positions

        if len(archetype_indices) >= min_cells:
            archetype_bins[archetype_label] = archetype_indices
            if verbose:
                print(f"      {archetype_label}: {len(archetype_indices)} assigned cells")
        else:
            if verbose:
                print(f"      {archetype_label}: {len(archetype_indices)} cells (< min_cells={min_cells}) - SKIPPING")

    if not archetype_bins:
        raise ValueError(f"No archetype bins meet minimum cell requirement ({min_cells} cells)")

    if verbose:
        print(f"   [OK] Assignment-based binning completed ({len(archetype_bins)} valid archetypes)")

    # Get pathway names from adata.uns if available
    pathway_names_key = f"{pathway_obsm_key}_pathways"
    if pathway_names_key in adata.uns:
        pathway_names = adata.uns[pathway_names_key]
    else:
        # Generate generic pathway names
        pathway_names = [f"pathway_{i}" for i in range(n_pathways)]
        if verbose:
            print(f"   [WARNING]  Pathway names not found in adata.uns['{pathway_names_key}'], using generic names")

    # Perform statistical tests using assignment-based binning
    results = []

    for archetype_name, arch_cell_indices in archetype_bins.items():
        if verbose:
            print(f"\n   Testing {archetype_name}...")

        # Get comparison cells based on comparison_group parameter
        if comparison_group == "all":
            # Get ALL other cells (1-vs-all approach - default behavior)
            all_cell_indices = np.arange(n_cells)
            other_cell_indices = np.setdiff1d(all_cell_indices, arch_cell_indices)
        elif comparison_group == "archetypes_only":
            # Get only cells assigned to OTHER archetypes (exclude archetype_0 and no_archetype)
            other_archetype_mask = (
                (archetype_assignments != archetype_name)
                & (archetype_assignments != "archetype_0")
                & (archetype_assignments != "no_archetype")
            )
            other_cell_indices = np.where(other_archetype_mask)[0]

        if verbose:
            print(f"      Archetype cells: {len(arch_cell_indices)}, Comparison cells: {len(other_cell_indices)}")

        if len(arch_cell_indices) < min_cells or len(other_cell_indices) < min_cells:
            if verbose:
                print(
                    f"      Skipping {archetype_name}: arch={len(arch_cell_indices)}, other={len(other_cell_indices)} (min: {min_cells})"
                )
            continue

        # Extract pathway scores for both groups
        arch_pathway_scores = pathway_scores[arch_cell_indices, :]  # [n_arch_cells, n_pathways]
        other_pathway_scores = pathway_scores[other_cell_indices, :]  # [n_other_cells, n_pathways]

        if verbose:
            print(f"       Extracted pathway scores: {arch_pathway_scores.shape} vs {other_pathway_scores.shape}")

        # Test each pathway
        n_pathways_tested = 0
        for pathway_idx, pathway_name in enumerate(pathway_names):
            arch_scores = arch_pathway_scores[:, pathway_idx]
            other_scores = other_pathway_scores[:, pathway_idx]

            # Skip pathways with no variance
            if np.var(arch_scores) == 0 and np.var(other_scores) == 0:
                continue

            try:
                # if test_method.lower() in ['wilcoxon', 'wilcox', 'mannwhitneyu', 'ranksum']:
                if test_method.lower() in ["mannwhitneyu"]:
                    # Use robust Mann-Whitney U test with tie-breaking for pathways too
                    # Determine test direction(s)
                    test_alternatives = []
                    if test_direction in ["greater", "both"]:
                        test_alternatives.append("greater")
                    if test_direction in ["less", "both"]:
                        test_alternatives.append("less")

                    # For backward compatibility, default to 'greater' if invalid direction
                    if not test_alternatives:
                        test_alternatives = ["greater"]

                    alternative = test_alternatives[0]
                    statistic, pvalue = robust_mannwhitneyu_test(
                        arch_scores, other_scores, alternative=alternative, tie_breaking=True, feature_name=pathway_name
                    )
                else:
                    raise ValueError(f"Unknown test method: {test_method}")

                # Calculate effect size for pathway scores
                mean_arch = np.mean(arch_scores)
                mean_other = np.mean(other_scores)

                # PATHWAY SCORES: Use mean difference, not log fold change
                # Pathway scores are activity measures (AUCell, pySCENIC), not expression levels
                # Simple difference is more interpretable for activity scores
                mean_diff = mean_arch - mean_other
                # Keep log_fold_change for backward compatibility but it's actually mean_diff
                log_fold_change = mean_diff

                # Handle different test directions with LFC filtering
                if test_direction == "two-sided":
                    # Two-sided test: most robust approach
                    statistic, pvalue = robust_mannwhitneyu_test(
                        arch_scores, other_scores, alternative="two-sided", tie_breaking=True, feature_name=pathway_name
                    )

                    # Apply LFC threshold filter (higher threshold for gene sets)
                    passes_lfc = abs(log_fold_change) >= min_logfc

                    if passes_lfc:
                        # Determine direction based on mean comparison
                        direction = "higher" if mean_arch > mean_other else "lower"

                        results.append(
                            {
                                "pathway": pathway_name,
                                "archetype": archetype_name,
                                "n_archetype_cells": len(arch_scores),
                                "n_other_cells": len(other_scores),
                                "mean_archetype": mean_arch,
                                "mean_other": mean_other,
                                "mean_diff": mean_diff,  # NEW: More appropriate for pathway scores
                                "log_fold_change": log_fold_change,  # Kept for compatibility
                                "statistic": statistic,
                                "pvalue": pvalue,
                                "test_direction": "two-sided",
                                "direction": direction,
                                "passes_lfc_threshold": True,
                            }
                        )

                # Handle both directions if requested (legacy mode)
                elif test_direction == "both":
                    # Test greater direction
                    stat_greater, pval_greater = robust_mannwhitneyu_test(
                        arch_scores, other_scores, alternative="greater", tie_breaking=True, feature_name=pathway_name
                    )
                    results.append(
                        {
                            "pathway": pathway_name,
                            "archetype": archetype_name,
                            "n_archetype_cells": len(arch_scores),
                            "n_other_cells": len(other_scores),
                            "mean_archetype": mean_arch,
                            "mean_other": mean_other,
                            "log_fold_change": log_fold_change,
                            "statistic": stat_greater,
                            "pvalue": pval_greater,
                            "test_direction": "greater",
                            "direction": "higher",  # Greater test always means higher
                        }
                    )

                    # Test less direction
                    stat_less, pval_less = robust_mannwhitneyu_test(
                        arch_scores, other_scores, alternative="less", tie_breaking=True, feature_name=pathway_name
                    )
                    results.append(
                        {
                            "pathway": pathway_name,
                            "archetype": archetype_name,
                            "n_archetype_cells": len(arch_scores),
                            "n_other_cells": len(other_scores),
                            "mean_archetype": mean_arch,
                            "mean_other": mean_other,
                            "log_fold_change": log_fold_change,
                            "statistic": stat_less,
                            "pvalue": pval_less,
                            "test_direction": "less",
                            "direction": "lower",  # Less test always means lower
                        }
                    )
                else:
                    # Single direction test
                    # Determine direction based on test direction
                    direction = "higher" if test_direction == "greater" else "lower"

                    results.append(
                        {
                            "pathway": pathway_name,
                            "archetype": archetype_name,
                            "n_archetype_cells": len(arch_scores),
                            "n_other_cells": len(other_scores),
                            "mean_archetype": mean_arch,
                            "mean_other": mean_other,
                            "log_fold_change": log_fold_change,
                            "statistic": statistic,
                            "pvalue": pvalue,
                            "test_direction": test_direction if test_direction in ["greater", "less"] else "greater",
                            "direction": direction,
                        }
                    )

                n_pathways_tested += 1

            except Exception:
                continue

        if verbose:
            print(f"      Tested {n_pathways_tested} pathways")

    if not results:
        raise ValueError("No valid pathway tests performed.")

    # Create results DataFrame and apply FDR correction based on scope
    results_df = pd.DataFrame(results)

    if verbose:
        print(f"\n   Applying FDR correction ({fdr_method}, {fdr_scope} scope)...")
        print(f"   Total tests performed: {len(results_df)}")

    # Apply FDR correction based on scope
    if fdr_scope == "global":
        # Global FDR correction across all tests
        results_df = apply_fdr_correction(
            results_df, pvalue_column="pvalue", method=fdr_method, alpha=0.05, verbose=verbose
        )
        if verbose:
            n_significant = results_df["significant"].sum()
            print(f"    Applied global FDR correction: {n_significant}/{len(results_df)} significant")

    elif fdr_scope == "per_archetype":
        # Per-archetype FDR correction
        results_df["fdr_pvalue"] = np.nan
        results_df["significant"] = False

        for archetype in results_df["archetype"].unique():
            arch_mask = results_df["archetype"] == archetype
            arch_results = results_df[arch_mask].copy()

            if len(arch_results) > 0:
                arch_results = apply_fdr_correction(
                    arch_results, pvalue_column="pvalue", method=fdr_method, alpha=0.05, verbose=False
                )
                results_df.loc[arch_mask, "fdr_pvalue"] = arch_results["fdr_pvalue"]
                results_df.loc[arch_mask, "significant"] = arch_results["significant"]

        if verbose:
            n_significant = results_df["significant"].sum()
            print(f"    Applied per-archetype FDR correction: {n_significant}/{len(results_df)} significant")

    elif fdr_scope == "none":
        # No FDR correction - use raw p-values
        results_df["fdr_pvalue"] = results_df["pvalue"]
        results_df["significant"] = results_df["pvalue"] < 0.05

        if verbose:
            n_significant = results_df["significant"].sum()
            print(f"    No FDR correction applied: {n_significant}/{len(results_df)} significant (raw p<0.05)")

    else:
        raise ValueError(f"Invalid fdr_scope: {fdr_scope}. Must be 'global', 'per_archetype', or 'none'")

    # Sort by significance and log fold change (absolute value for effect size)
    results_df["abs_log_fold_change"] = results_df["log_fold_change"].abs()
    results_df = results_df.sort_values(["fdr_pvalue", "pvalue", "abs_log_fold_change"], ascending=[True, True, False])
    results_df = results_df.drop(columns=["abs_log_fold_change"])

    if verbose:
        n_significant = results_df["significant"].sum()
        total_tests = len(results_df)
        print("[OK] Pathway association testing completed!")
        print(f"   Total tests: {total_tests}")
        print(f"   Significant associations: {n_significant} ({100 * n_significant / total_tests:.1f}%)")

        # Show significance breakdown
        if "test_direction" in results_df.columns:
            direction_counts = results_df[results_df["significant"]].groupby("test_direction").size()
            if len(direction_counts) > 0:
                print(f"   Significant by direction: {dict(direction_counts)}")

        # Show FDR correction impact
        raw_significant = (results_df["pvalue"] < 0.05).sum()
        print(f"   Raw significant (p<0.05): {raw_significant}")
        if fdr_scope != "none":
            print(f"   FDR correction impact: {raw_significant} → {n_significant}")

    return results_df


def test_archetype_conditional_associations(
    adata,
    obs_column: str,
    archetype_assignments=None,  # For backward compatibility - will be ignored
    obs_key: str = "archetypes",
    test_method: str = "hypergeometric",
    fdr_method: str = "benjamini_hochberg",
    min_cells: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Test for enrichment of archetypes in different conditions using hypergeometric tests.

    Uses efficient boolean matrix approach to test whether specific archetypes are
    significantly enriched in particular experimental conditions, treatments, cell types, etc.

    This implementation follows genomics best practices by:
    1. Creating boolean matrices for archetypes and conditions upfront
    2. Using vectorized operations for overlap computations
    3. Avoiding redundant contingency table construction
    4. Scaling efficiently to large datasets

    Args:
        adata: AnnData object with metadata in .obs
        obs_column: Column name in adata.obs to test (e.g., 'treatment', 'patient_id', 'timepoint')
        archetype_assignments: DEPRECATED - archetype assignments are now read from adata.obs[obs_key]
        obs_key: Key for archetype assignments in adata.obs (default: 'archetypes')
        test_method: Statistical test method ('hypergeometric')
        fdr_method: Multiple testing correction method
        min_cells: Minimum cells per condition for testing
        verbose: Whether to print testing progress

    Returns
    -------
        pd.DataFrame with columns:
            - archetype: Archetype label
            - condition: Condition value from obs_column
            - observed: Observed count of archetype in condition
            - expected: Expected count under null hypothesis
            - total_archetype: Total cells in archetype
            - total_condition: Total cells in condition
            - odds_ratio: Odds ratio (observed/expected enrichment)
            - ci_lower: Lower 95% confidence interval for odds ratio
            - ci_upper: Upper 95% confidence interval for odds ratio
            - pvalue: Hypergeometric p-value
            - fdr_pvalue: FDR-corrected p-value
            - significant: Boolean significance
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy.stats is required for statistical testing.")

    if verbose:
        print("🧪 Testing archetype-conditional associations...")
        print(f"   Condition column: {obs_column}")
        print(f"   Method: {test_method} (boolean matrix approach)")

    # Check if obs_column exists
    if obs_column not in adata.obs.columns:
        raise ValueError(f"Column '{obs_column}' not found in adata.obs. Available: {list(adata.obs.columns)}")

    # CRITICAL: Use AnnData archetype assignments directly
    if obs_key not in adata.obs.columns:
        raise ValueError(
            f"Archetype assignments not found in adata.obs['{obs_key}']. Run bin_cells_by_archetype() first."
        )

    archetype_assignments = adata.obs[obs_key]  # Categorical series
    conditions_series = adata.obs[obs_column]  # Condition series

    # Filter to cells with both archetype assignments and condition values
    # Exclude 'no_archetype' cells and missing conditions
    valid_mask = (archetype_assignments != "no_archetype") & conditions_series.notna()

    if valid_mask.sum() == 0:
        raise ValueError(f"No cells have both archetype assignments and non-null values in {obs_column}")

    # Extract valid data
    valid_archetype_assignments = archetype_assignments[valid_mask]
    valid_conditions = conditions_series[valid_mask]

    # Get unique archetypes and conditions
    # Handle both categorical and non-categorical columns
    if hasattr(valid_archetype_assignments, "cat"):
        unique_archetypes = [cat for cat in valid_archetype_assignments.cat.categories if cat != "no_archetype"]
    else:
        unique_archetypes = [
            arch for arch in valid_archetype_assignments.unique() if arch != "no_archetype" and pd.notna(arch)
        ]
    unique_conditions = valid_conditions.unique()

    if verbose:
        print(f"   Testing {len(unique_archetypes)} archetypes × {len(unique_conditions)} conditions")
        print(f"   Total cells in analysis: {len(valid_archetype_assignments)}")

        # Show condition distribution
        condition_counts = valid_conditions.value_counts()
        print("   Condition distribution:")
        for condition, count in condition_counts.head(5).items():
            print(f"      {condition}: {count} cells")

    # CREATE BOOLEAN MATRICES (GENOMICS BEST PRACTICE)
    # ================================================

    # 1. Create archetype boolean matrix: (n_cells × n_archetypes)
    n_valid_cells = len(valid_archetype_assignments)
    archetype_matrix = np.zeros((n_valid_cells, len(unique_archetypes)), dtype=bool)
    archetype_to_idx = {arch: i for i, arch in enumerate(unique_archetypes)}

    for i, arch in enumerate(valid_archetype_assignments):
        if arch in archetype_to_idx:
            archetype_matrix[i, archetype_to_idx[arch]] = True

    # 2. Create condition boolean matrix: (n_cells × n_conditions)
    condition_matrix = np.zeros((n_valid_cells, len(unique_conditions)), dtype=bool)
    condition_to_idx = {cond: i for i, cond in enumerate(unique_conditions)}

    for i, cond in enumerate(valid_conditions):
        condition_matrix[i, condition_to_idx[cond]] = True

    if verbose:
        print("   Boolean matrices created:")
        print(f"      Archetype matrix: {archetype_matrix.shape} (cells × archetypes)")
        print(f"      Condition matrix: {condition_matrix.shape} (cells × conditions)")

    # VECTORIZED HYPERGEOMETRIC TESTING
    # ==================================

    results = []
    total_cells = n_valid_cells

    # For each archetype-condition combination
    for arch_idx, archetype in enumerate(unique_archetypes):
        # Get boolean vector for this archetype
        arch_vector = archetype_matrix[:, arch_idx]
        total_archetype = np.sum(arch_vector)

        # Skip archetypes with too few cells
        if total_archetype < min_cells:
            continue

        for cond_idx, condition in enumerate(unique_conditions):
            # Get boolean vector for this condition
            cond_vector = condition_matrix[:, cond_idx]
            total_condition = np.sum(cond_vector)

            # Skip conditions with too few cells
            if total_condition < min_cells:
                continue

            # VECTORIZED OVERLAP COMPUTATION
            observed = np.sum(arch_vector & cond_vector)  # Boolean AND for overlap
            expected = (total_archetype * total_condition) / total_cells

            # Skip if no overlap or too few cells in overlap
            if observed < min_cells:
                continue

            try:
                if test_method.lower() == "hypergeometric":
                    # Hypergeometric test: P(X >= observed) where X ~ Hypergeom(N, K, n)
                    # N = total_cells, K = total_condition (successes in population)
                    # n = total_archetype (draws), X = observed (successes in sample)
                    pvalue = hypergeom.sf(observed - 1, total_cells, total_condition, total_archetype)

                    # Calculate 2x2 contingency table for odds ratio
                    arch_and_cond = observed
                    arch_not_cond = total_archetype - observed
                    not_arch_cond = total_condition - observed
                    not_arch_not_cond = total_cells - total_archetype - total_condition + observed

                    # Compute odds ratio with confidence interval
                    if arch_not_cond > 0 and not_arch_cond > 0:
                        contingency_table = np.array(
                            [[arch_and_cond, arch_not_cond], [not_arch_cond, not_arch_not_cond]]
                        )
                        or_result = odds_ratio(contingency_table)
                        odds_ratio_val = or_result.statistic
                        ci_lower, ci_upper = or_result.confidence_interval()
                    else:
                        # Handle edge cases
                        odds_ratio_val = np.inf if observed > expected else 0.0
                        ci_lower, ci_upper = np.nan, np.nan

                else:
                    raise ValueError(f"Unknown test method: {test_method}")

                results.append(
                    {
                        "archetype": archetype,
                        "condition": condition,
                        "observed": int(observed),
                        "expected": expected,
                        "total_archetype": int(total_archetype),
                        "total_condition": int(total_condition),
                        "odds_ratio": odds_ratio_val,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "pvalue": pvalue,
                    }
                )

            except Exception as e:
                if verbose:
                    print(f"      Warning: Test failed for {archetype} × {condition}: {e}")
                continue

    if not results:
        raise ValueError("No valid conditional tests performed. Check minimum cell requirements.")

    # Create results DataFrame and apply FDR correction
    results_df = pd.DataFrame(results)
    results_df = apply_fdr_correction(results_df, method=fdr_method)

    # Sort by significance and effect size (fix the lambda key issue)
    results_df = results_df.sort_values(["fdr_pvalue", "pvalue", "odds_ratio"], ascending=[True, True, False])

    if verbose:
        n_significant = results_df["significant"].sum()
        total_tests = len(results_df)
        print("[OK] Conditional association testing completed!")
        print(f"   Total tests: {total_tests}")
        print(f"   Significant associations: {n_significant} ({100 * n_significant / total_tests:.1f}%)")

        # Show efficiency gain
        theoretical_tests = len(unique_archetypes) * len(unique_conditions)
        print(f"   Efficiency: {total_tests}/{theoretical_tests} tests performed (filtered by min_cells)")

        # Show top enrichments
        if n_significant > 0:
            print("\n[STATS] Top significant enrichments:")
            sig_results = results_df[results_df["significant"]].head(5)
            for _, row in sig_results.iterrows():
                enrichment = "enriched" if row["odds_ratio"] > 1 else "depleted"
                print(
                    f"   {row['archetype']} {enrichment} in {row['condition']}: "
                    f"OR={row['odds_ratio']:.2f}, p={row['fdr_pvalue']:.2e}"
                )

    return results_df


def apply_fdr_correction(
    results_df: pd.DataFrame,
    pvalue_column: str = "pvalue",
    method: str = "benjamini_hochberg",
    alpha: float = 0.05,
    validate_assumptions: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Apply robust false discovery rate correction to p-values.

    IMPORTANT: FDR methods assume test independence. In single-cell analysis,
    genes are often correlated, which can lead to conservative results.

    Args:
        results_df: DataFrame containing p-values
        pvalue_column: Name of column containing raw p-values
        method: FDR correction method ('benjamini_hochberg', 'bonferroni', 'holm')
        alpha: Significance threshold
        validate_assumptions: Whether to validate FDR assumptions and warn about violations
        verbose: Whether to print detailed correction statistics

    Returns
    -------
        DataFrame with added 'fdr_pvalue' and 'significant' columns

    Notes
    -----
        - Benjamini-Hochberg (BH): Controls FDR under independence/positive dependence
        - Bonferroni: More conservative, controls FWER (Family-Wise Error Rate)
        - Holm: Step-down Bonferroni, more powerful than Bonferroni but still conservative
    """
    if pvalue_column not in results_df.columns:
        raise ValueError(f"Column '{pvalue_column}' not found in results DataFrame")

    results_df = results_df.copy()
    pvalues = results_df[pvalue_column].values

    # Validate p-values and assumptions
    if validate_assumptions:
        # Check for valid p-values
        invalid_pvals = (pvalues < 0) | (pvalues > 1) | np.isnan(pvalues)
        if np.any(invalid_pvals):
            n_invalid = np.sum(invalid_pvals)
            if verbose:
                print(f"[WARNING]  Found {n_invalid} invalid p-values (NaN, <0, or >1). Setting to 1.0.")
            pvalues[invalid_pvals] = 1.0

        # Check for uniform distribution (should not be perfectly uniform)
        n_bins = min(20, len(pvalues) // 5)
        if n_bins >= 5:
            hist, _ = np.histogram(pvalues, bins=n_bins)
            # Chi-square test for uniformity would be too strict
            # Instead, just check for obvious patterns
            if np.std(hist) / np.mean(hist) < 0.1:  # Very uniform
                if verbose:
                    print("[WARNING]  P-value distribution appears very uniform. Check test validity.")

    if verbose:
        print("[STATS] FDR Correction Statistics:")
        print(f"   Method: {method}")
        print(f"   Number of tests: {len(pvalues)}")
        print(f"   Raw p-values < {alpha}: {np.sum(pvalues < alpha)}")
        print(f"   Min p-value: {np.min(pvalues):.2e}")
        print(f"   Median p-value: {np.median(pvalues):.2e}")

    # Apply correction
    if method.lower() == "benjamini_hochberg":
        corrected_pvalues = false_discovery_control(pvalues, method="bh")
    elif method.lower() == "bonferroni":
        corrected_pvalues = np.minimum(pvalues * len(pvalues), 1.0)
    # elif method.lower() == 'holm':
    #     # Holm-Bonferroni method
    #     sorted_indices = np.argsort(pvalues)
    #     corrected_pvalues = np.zeros_like(pvalues)

    #     for i, idx in enumerate(sorted_indices):
    #         corrected_pvalues[idx] = min(pvalues[idx] * (len(pvalues) - i), 1.0)

    #     # Ensure monotonicity
    #     for i in range(1, len(sorted_indices)):
    #         idx = sorted_indices[i]
    #         prev_idx = sorted_indices[i-1]
    #         corrected_pvalues[idx] = max(corrected_pvalues[idx], corrected_pvalues[prev_idx])
    else:
        raise ValueError(f"Unknown FDR method: {method}")

    results_df["fdr_pvalue"] = corrected_pvalues
    results_df["significant"] = corrected_pvalues < alpha

    if verbose:
        n_significant_raw = np.sum(pvalues < alpha)
        n_significant_corrected = np.sum(corrected_pvalues < alpha)
        print(f"   FDR-corrected p-values < {alpha}: {n_significant_corrected}")
        print(
            f"   Correction ratio: {n_significant_corrected}/{n_significant_raw} = {n_significant_corrected / max(n_significant_raw, 1):.3f}"
        )

        if n_significant_corrected == 0 and n_significant_raw > 0:
            print("   [WARNING]  No significant results after FDR correction!")
            print("   Consider: (1) Relaxing alpha, (2) Using per-archetype scope, or (3) Check test assumptions")

        # Report by method-specific information
        if method.lower() == "benjamini_hochberg":
            print("   Note: BH assumes independence or positive dependence among tests")
            if n_significant_corrected < n_significant_raw * 0.1:
                print("   NOTE:  Large reduction suggests potential test dependence or weak signals")

    return results_df


def generate_archetype_patterns(
    unique_archetypes: list[str],
    include_specialization_patterns: bool = True,
    include_tradeoff_patterns: bool = True,
    max_pattern_size: int = None,
    exclude_archetype_0: bool = False,
    specific_patterns: list[str] = None,
) -> list[dict[str, Any]]:
    """
    Generate systematic archetype pattern combinations for testing.

    Creates pattern definitions with interpretable naming for statistical testing.
    Patterns define which archetypes should be "high" vs "low" for a given feature.

    Parameters
    ----------
    unique_archetypes : list[str]
        List of archetype labels (e.g., ['archetype_0', 'archetype_1', ...]).
    include_specialization_patterns : bool, default: True
        Include archetype_0 vs individual archetype patterns.
    include_tradeoff_patterns : bool, default: True
        Include complex multi-archetype high vs low patterns.
    max_pattern_size : int | None, default: None
        Maximum number of archetypes in high/low groups.
        If None, defaults to min(n_archetypes // 2, 3).
    exclude_archetype_0 : bool, default: False
        Exclude archetype_0 from tradeoff patterns (for clarity).
    specific_patterns : list[str] | None, default: None
        Generate only specific patterns (e.g., ['2v3', '1v45']).
        Format: digits for high archetypes, 'v', digits for low archetypes.

    Returns
    -------
    list[dict]
        List of pattern dictionaries, each containing:

        - ``high_archetypes`` : list[str] - Archetypes expected to have high values
        - ``low_archetypes`` : list[str] - Archetypes expected to have low values
        - ``pattern_name`` : str - Descriptive name (e.g., 'specialist_arch1_1xxxx_0xxxx')
        - ``pattern_code`` : str - Visual pattern code (e.g., '12xxx_xx345')
        - ``pattern_type`` : str - Type: 'specialization', 'tradeoff', or 'custom'
        - ``pattern_set`` : str | None - Pattern set description (e.g., 'non-zero archetypes').
          Only present for tradeoff patterns.

    Notes
    -----
    **Pattern Code Format**: "12xxx_xx345"

    - Position corresponds to archetype number (0, 1, 2, 3, 4, 5...)
    - Numbers = high archetypes, 'x' = low/uninvolved archetypes
    - Underscore separates high group from low group

    **Examples of pattern codes**:

    - "1xxxx_0xxxx" = archetype_1 high, archetype_0 low (specialist)
    - "12xxx_xx345" = archetypes 1,2 high vs archetypes 3,4,5 low (tradeoff)
    - "0xxxx_x1234" = archetype_0 high vs archetypes 1,2,3,4 low (binary)

    **Systematic Generation**:

    - Generates BOTH directions: "0xxxx_x1234" AND "x1234_0xxxx"
    - Creates patterns with archetype_0 and without (suffix: _nonzero)
    - Organized by complexity: specialists → binary → 3-way → complex

    See Also
    --------
    test_archetype_pattern_associations : Tests patterns against features
    peach._core.types.ArchetypePattern : Pattern type definition

    Examples
    --------
    >>> unique_archetypes = ["archetype_0", "archetype_1", "archetype_2", "archetype_3"]
    >>> # Generate all patterns
    >>> patterns = generate_archetype_patterns(unique_archetypes)
    >>> for p in patterns[:5]:
    ...     print(f"{p['pattern_code']}: {p['high_archetypes']} vs {p['low_archetypes']}")
    >>> # Generate specific patterns only
    >>> specific = generate_archetype_patterns(unique_archetypes, specific_patterns=["1v2", "1v3", "2v3"])
    """
    from itertools import combinations

    def create_pattern_code(high_archetypes: list[str], low_archetypes: list[str], all_archetypes: list[str]) -> str:
        """
        Create interpretable pattern code like '12xxx_xx345'.

        Args:
            high_archetypes: List of archetype names (e.g., ['archetype_1', 'archetype_2'])
            low_archetypes: List of archetype names (e.g., ['archetype_3', 'archetype_4'])
            all_archetypes: Complete list of archetype names for positioning

        Returns
        -------
            Pattern code string (e.g., '12xxx_xx345')
        """
        # Create a mapping from archetype names to indices/codes
        sorted_archetypes = sorted(all_archetypes)

        # Try to extract numeric suffixes if they exist
        try:
            # Check if all archetypes have numeric suffixes after underscore
            if all("_" in arch for arch in all_archetypes):
                # Try to extract numbers
                nums = [int(arch.split("_")[-1]) for arch in all_archetypes]
                # If successful, use numeric codes
                arch_to_code = {arch: str(int(arch.split("_")[-1])) for arch in all_archetypes}
                use_numeric = True
            else:
                raise ValueError("Not all archetypes have numeric suffixes")
        except (ValueError, AttributeError):
            # Use alphabetic codes for arbitrary names (A, B, C, ...)
            # or use indices (0, 1, 2, ...) if more than 26 archetypes
            if len(sorted_archetypes) <= 26:
                # Use letters A-Z
                arch_to_code = {arch: chr(65 + i) for i, arch in enumerate(sorted_archetypes)}
            else:
                # Use numeric indices
                arch_to_code = {arch: str(i) for i, arch in enumerate(sorted_archetypes)}
            use_numeric = False

        # Build pattern strings
        high_str = ""
        low_str = ""

        for arch in sorted_archetypes:
            code = arch_to_code[arch]
            if arch in high_archetypes:
                high_str += code
                low_str += "x"
            elif arch in low_archetypes:
                high_str += "x"
                low_str += code
            else:
                # Archetype not involved in this pattern
                high_str += "x"
                low_str += "x"

        return f"{high_str}_{low_str}"

    def create_pattern_name(
        pattern_code: str, pattern_type: str, high_archetypes: list[str], low_archetypes: list[str]
    ) -> str:
        """Create descriptive pattern name with interpretable code."""
        if pattern_type == "specialization":
            specialist_arch = high_archetypes[0].split("_")[-1]
            return f"specialist_arch{specialist_arch}_{pattern_code}"
        elif pattern_type == "tradeoff":
            return f"tradeoff_{pattern_code}"
        else:
            return f"custom_{pattern_code}"

    patterns = []

    # SPECIFIC PATTERNS: If specified, generate only those patterns
    if specific_patterns:
        for pattern_spec in specific_patterns:
            if "v" in pattern_spec:
                # Parse patterns like '2v3', '1v45', '12v34'
                high_part, low_part = pattern_spec.split("v")
                high_archetypes = [f"archetype_{i}" for i in high_part]
                low_archetypes = [f"archetype_{i}" for i in low_part]

                # Validate that archetypes exist
                high_valid = all(arch in unique_archetypes for arch in high_archetypes)
                low_valid = all(arch in unique_archetypes for arch in low_archetypes)

                if high_valid and low_valid:
                    pattern_code = create_pattern_code(high_archetypes, low_archetypes, unique_archetypes)
                    pattern_name = create_pattern_name(pattern_code, "custom", high_archetypes, low_archetypes)

                    patterns.append(
                        {
                            "high_archetypes": high_archetypes,
                            "low_archetypes": low_archetypes,
                            "pattern_name": pattern_name,
                            "pattern_code": pattern_code,
                            "pattern_type": "custom",
                        }
                    )

        return patterns  # Return only specific patterns if requested

    # 1. SPECIALIZATION PATTERNS (archetype_0 vs individual archetypes)
    if include_specialization_patterns and "archetype_0" in unique_archetypes:
        other_archetypes = [a for a in unique_archetypes if a != "archetype_0"]

        for arch in other_archetypes:
            pattern_code = create_pattern_code([arch], ["archetype_0"], unique_archetypes)
            pattern_name = create_pattern_name(pattern_code, "specialization", [arch], ["archetype_0"])

            patterns.append(
                {
                    "high_archetypes": [arch],
                    "low_archetypes": ["archetype_0"],
                    "pattern_name": pattern_name,
                    "pattern_code": pattern_code,
                    "pattern_type": "specialization",
                }
            )

    # 2. TRADEOFF PATTERNS (systematic high/low combinations)
    if include_tradeoff_patterns:
        # Generate two sets: with and without archetype_0
        pattern_sets = []

        # Detect archetype_0 - could be 'archetype_0', 'Archetype_0', '0', etc.
        # Also check for patterns like 'Arch_0', 'arch0', etc.
        detected_arch0 = None
        for arch in unique_archetypes:
            # Direct matches
            if arch in ["archetype_0", "Archetype_0", "0"]:
                detected_arch0 = arch
                break
            # Pattern matches (anything ending with _0 or 0)
            if arch.lower().endswith("_0") or arch.lower() == "arch0":
                detected_arch0 = arch
                break

        # Determine which archetype sets to use
        if detected_arch0 and exclude_archetype_0:
            # Exclude archetype_0 from patterns
            non_zero_archetypes = [a for a in unique_archetypes if a != detected_arch0]
            if len(non_zero_archetypes) >= 2:
                pattern_sets.append(
                    {"archetypes": non_zero_archetypes, "name_suffix": "_nonzero", "description": "non-zero archetypes"}
                )
        else:
            # Include all archetypes (either no archetype_0 exists or we're including it)
            pattern_sets.append({"archetypes": unique_archetypes, "name_suffix": "", "description": "all archetypes"})

        # If we have archetype_0 but aren't excluding it, also generate patterns without it
        if detected_arch0 and not exclude_archetype_0:
            non_zero_archetypes = [a for a in unique_archetypes if a != detected_arch0]
            if len(non_zero_archetypes) >= 2:
                pattern_sets.append(
                    {"archetypes": non_zero_archetypes, "name_suffix": "_nonzero", "description": "non-zero archetypes"}
                )

        # Make sure we have at least one pattern set for complex patterns
        if not pattern_sets and len(unique_archetypes) >= 2:
            # Fallback: use all archetypes
            pattern_sets.append({"archetypes": unique_archetypes, "name_suffix": "", "description": "all archetypes"})

        for pattern_set in pattern_sets:
            tradeoff_archetypes = pattern_set["archetypes"]
            n_archetypes = len(tradeoff_archetypes)

            # Set reasonable max pattern size
            if max_pattern_size is None:
                max_pattern_size = min(n_archetypes // 2, 3)  # Max 3 archetypes per group

            # Generate patterns based on max_pattern_size
            if max_pattern_size == 1:
                # Special case for pairwise (1v1) patterns
                # Generate all unique pairs
                for i, arch1 in enumerate(tradeoff_archetypes):
                    for arch2 in tradeoff_archetypes[i + 1 :]:
                        # Create both directions of the pair
                        for high_arch, low_arch in [(arch1, arch2), (arch2, arch1)]:
                            high_archetypes = [high_arch]
                            low_archetypes = [low_arch]

                            pattern_code = create_pattern_code(high_archetypes, low_archetypes, unique_archetypes)
                            pattern_name = create_pattern_name(
                                pattern_code, "tradeoff", high_archetypes, low_archetypes
                            )

                            patterns.append(
                                {
                                    "high_archetypes": high_archetypes,
                                    "low_archetypes": low_archetypes,
                                    "pattern_name": pattern_name + pattern_set["name_suffix"],
                                    "pattern_code": pattern_code,
                                    "pattern_type": "tradeoff",
                                    "pattern_set": pattern_set["description"],
                                }
                            )
            else:
                # General case for complex patterns
                # Generate patterns where each side has at most max_pattern_size archetypes
                for high_size in range(1, min(max_pattern_size + 1, n_archetypes)):
                    # For each possible low_size
                    for low_size in range(1, min(max_pattern_size + 1, n_archetypes - high_size + 1)):
                        # Generate all combinations of high_size archetypes
                        for high_archetypes in combinations(tradeoff_archetypes, high_size):
                            # Get remaining archetypes
                            remaining = [a for a in tradeoff_archetypes if a not in high_archetypes]
                            # Generate all combinations of low_size from remaining
                            for low_archetypes in combinations(remaining, low_size):
                                # Now we have a valid pattern with controlled sizes on both sides
                                low_archetypes = list(low_archetypes)  # Convert to list

                                # Create pattern code and name
                                pattern_code = create_pattern_code(
                                    list(high_archetypes), low_archetypes, unique_archetypes
                                )
                                pattern_name = create_pattern_name(
                                    pattern_code, "tradeoff", list(high_archetypes), low_archetypes
                                )

                                patterns.append(
                                    {
                                        "high_archetypes": list(high_archetypes),
                                        "low_archetypes": low_archetypes,
                                        "pattern_name": pattern_name + pattern_set["name_suffix"],
                                        "pattern_code": pattern_code,
                                        "pattern_type": "tradeoff",
                                        "pattern_set": pattern_set["description"],
                                    }
                                )

    # ORGANIZE PATTERNS: Sort by complexity and then by pattern code for clean output
    def pattern_sort_key(pattern):
        """Sort patterns by complexity, then by archetype order."""
        code = pattern["pattern_code"]

        # Count number of archetypes involved (numbers in the pattern)
        n_high = len([c for c in code.split("_")[0] if c.isdigit()])
        n_low = len([c for c in code.split("_")[1] if c.isdigit()])
        complexity = n_high + n_low

        # Sort by: 1) pattern type, 2) complexity, 3) alphabetical pattern code
        type_order = {"specialization": 0, "tradeoff": 1, "custom": 2}
        return (type_order.get(pattern["pattern_type"], 3), complexity, code)

    patterns.sort(key=pattern_sort_key)
    return patterns


def test_archetype_pattern_associations(
    adata,
    data_obsm_key: str = "pathway_scores",
    obs_key: str = "archetypes",
    pattern_types: list[str] = ["specialization", "tradeoff"],
    test_method: str = "mannwhitneyu",
    fdr_method: str = "benjamini_hochberg",
    fdr_scope: str = "global",
    min_logfc: float = 0.01,
    min_cells: int = 10,
    max_pattern_size: int = 3,
    exclude_archetype_0: bool = False,
    specific_patterns: list[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Test systematic archetype patterns for gene/pathway associations.

    Generates and tests all meaningful archetype combinations:

    1. **Specialization patterns**: Individual archetypes vs archetype_0 (generalist)
    2. **Tradeoff patterns**: Multi-archetype high vs low groups

    Parameters
    ----------
    adata : AnnData
        AnnData object with gene/pathway scores and archetype assignments.
    data_obsm_key : str, default: 'pathway_scores'
        Key for gene/pathway scores in adata.obsm.
    obs_key : str, default: 'archetypes'
        Key for archetype assignments in adata.obs.
    pattern_types : list[str], default: ['specialization', 'tradeoff']
        Types of patterns to test. Options: 'specialization', 'tradeoff', 'all'.
    test_method : str, default: 'mannwhitneyu'
        Statistical test method.
    fdr_method : str, default: 'benjamini_hochberg'
        Multiple testing correction method.
    fdr_scope : str, default: 'global'
        FDR correction scope: 'global', 'per_archetype', or 'none'.
    min_logfc : float, default: 0.01
        Minimum effect size threshold (log_fold_change for genes, mean_diff for pathways).
    min_cells : int, default: 10
        Minimum cells per group for testing.
    max_pattern_size : int, default: 3
        Maximum number of archetypes in high/low groups.
    exclude_archetype_0 : bool, default: False
        Exclude archetype_0 from tradeoff patterns.
    specific_patterns : list[str] | None, default: None
        Test only specific patterns (e.g., ['2v3', '1v45']).
    verbose : bool, default: True
        Whether to print testing progress.

    Returns
    -------
    pd.DataFrame
        Pattern association results with columns:

        **Feature identifier** (one present based on data type):

        - ``pathway`` : str - Pathway name (if pathway data)
        - ``gene`` : str - Gene name (if gene data)

        **Pattern information**:

        - ``pattern_name`` : str - Interpretable pattern name
        - ``pattern_code`` : str - Visual pattern code (e.g., '12xxx_xx345')
        - ``pattern_type`` : str - 'specialization', 'tradeoff', or 'custom'
        - ``high_archetypes`` : str - Comma-separated high archetype names
        - ``low_archetypes`` : str - Comma-separated low archetype names

        **Group statistics**:

        - ``n_high_cells`` : int - Number of cells in high group
        - ``n_low_cells`` : int - Number of cells in low group
        - ``mean_high`` : float - Mean value in high group
        - ``mean_low`` : float - Mean value in low group

        **Effect sizes**:

        - ``log_fold_change`` : float - Effect size (backward compatible)
        - ``primary_effect_size`` : float - Standardized effect size
        - ``effect_size_col`` : str - Which column was used ('mean_diff' or 'log_fold_change')
        - ``mean_diff`` : float - Mean difference (for pathways only)

        **Test results**:

        - ``statistic`` : float - Test statistic
        - ``pvalue`` : float - Raw p-value
        - ``test_direction`` : str - Test direction ('two-sided')
        - ``direction`` : str - Effect direction ('higher' or 'lower')
        - ``passes_lfc_threshold`` : bool - Whether passes effect size threshold
        - ``fdr_pvalue`` : float - FDR-corrected p-value
        - ``significant`` : bool - Whether statistically significant

        **Compatibility aliases** (for plotting functions):

        - ``archetype`` : str - Alias for pattern_name
        - ``mean_archetype`` : float - Alias for mean_high
        - ``mean_other`` : float - Alias for mean_low
        - ``n_archetype_cells`` : int - Alias for n_high_cells
        - ``n_other_cells`` : int - Alias for n_low_cells

    Raises
    ------
    ValueError
        If required keys not found or no valid tests performed.

    See Also
    --------
    generate_archetype_patterns : Pattern generation
    identify_mutual_exclusivity_patterns : Find opposing patterns
    peach.tl.pattern_analysis : User-facing comprehensive analysis
    peach._core.types.PatternAssociationResult : Result row type definition

    Examples
    --------
    >>> # Test specialization patterns only
    >>> spec_results = test_archetype_pattern_associations(adata, pattern_types=["specialization"])
    >>> # Test specific patterns
    >>> specific_results = test_archetype_pattern_associations(adata, specific_patterns=["2v3", "1v45", "12v34"])
    >>> # Filter results
    >>> specialists = results[results["pattern_type"] == "specialization"]
    >>> strong_effects = results[(results["significant"]) & (results["log_fold_change"].abs() > 0.5)]
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy.stats is required for statistical testing.")

    if verbose:
        print("🧪 Testing archetypal pattern associations...")
        print(f"   Pattern types: {pattern_types}")
        print(f"   Method: {test_method}")
        print(f"   FDR correction: {fdr_method} ({fdr_scope} scope)")
        print(f"   Max pattern size: {max_pattern_size}")

    # Validate AnnData contains required data
    if data_obsm_key not in adata.obsm:
        raise ValueError(f"Data not found in adata.obsm['{data_obsm_key}']")

    if obs_key not in adata.obs.columns:
        raise ValueError(f"Archetype assignments not found in adata.obs['{obs_key}']")

    feature_scores = adata.obsm[data_obsm_key]
    archetype_assignments = adata.obs[obs_key]
    n_cells, n_features = feature_scores.shape

    # SMART DATA TYPE DETECTION: Determine if we're working with genes or pathways
    is_pathway_data = "pathway" in data_obsm_key.lower()
    data_type = "pathway" if is_pathway_data else "gene"

    if verbose:
        print(f"   Detected data type: {data_type} (from key: {data_obsm_key})")

    # Get unique archetypes (excluding 'no_archetype')
    # Handle both categorical and non-categorical columns
    if hasattr(archetype_assignments, "cat"):
        # Categorical column
        unique_archetypes = [cat for cat in archetype_assignments.cat.categories if cat != "no_archetype"]
    else:
        # Regular column (string, object, etc.)
        unique_archetypes = [
            arch for arch in archetype_assignments.unique() if arch != "no_archetype" and pd.notna(arch)
        ]

    if verbose:
        print(f"   Found archetypes: {unique_archetypes}")

    # Generate patterns to test
    include_specialization = "specialization" in pattern_types or "all" in pattern_types
    include_tradeoff = "tradeoff" in pattern_types or "all" in pattern_types

    patterns = generate_archetype_patterns(
        unique_archetypes=unique_archetypes,
        include_specialization_patterns=include_specialization,
        include_tradeoff_patterns=include_tradeoff,
        max_pattern_size=max_pattern_size,
        exclude_archetype_0=exclude_archetype_0,
        specific_patterns=specific_patterns,
    )

    if verbose:
        print(f"   Generated {len(patterns)} patterns to test:")
        for pattern in patterns[:5]:  # Show first 5
            print(f"      {pattern['pattern_name']}: {pattern['high_archetypes']} vs {pattern['low_archetypes']}")
        if len(patterns) > 5:
            print(f"      ... and {len(patterns) - 5} more patterns")

    # Get feature names using generalized approach
    feature_names_key = f"{data_obsm_key}_{data_type}s"  # e.g., 'pathway_scores_pathways' or 'gene_scores_genes'

    if feature_names_key in adata.uns:
        feature_names = adata.uns[feature_names_key]
    else:
        # Try alternative naming conventions
        alt_keys = [f"{data_obsm_key}_features", f"{data_type}_names", f"{data_type}s"]
        feature_names = None
        for alt_key in alt_keys:
            if alt_key in adata.uns:
                feature_names = adata.uns[alt_key]
                if verbose:
                    print(f"    Using alternative feature names key: {alt_key}")
                break

        if feature_names is None:
            feature_names = [f"{data_type}_{i}" for i in range(n_features)]
            if verbose:
                print("   [WARNING]  Feature names not found, using generic names")

    # Create cell index mappings
    archetype_cell_maps = {}
    for archetype in unique_archetypes:
        mask = archetype_assignments == archetype
        indices = np.where(mask)[0]
        if len(indices) >= min_cells:
            archetype_cell_maps[archetype] = indices
            if verbose and len(archetype_cell_maps) <= 6:  # Don't spam output
                print(f"   {archetype}: {len(indices)} cells")

    # Test each pattern
    results = []

    for pattern in patterns:
        if verbose:
            print(f"\n   Testing pattern: {pattern['pattern_name']}")

        # Get cell indices for high and low groups
        high_indices = []
        low_indices = []

        for arch in pattern["high_archetypes"]:
            if arch in archetype_cell_maps:
                high_indices.extend(archetype_cell_maps[arch])

        for arch in pattern["low_archetypes"]:
            if arch in archetype_cell_maps:
                low_indices.extend(archetype_cell_maps[arch])

        high_indices = np.array(high_indices)
        low_indices = np.array(low_indices)

        if len(high_indices) < min_cells or len(low_indices) < min_cells:
            if verbose:
                print(f"      Skipping: high={len(high_indices)}, low={len(low_indices)} (min: {min_cells})")
            continue

        if verbose:
            print(f"      High group: {len(high_indices)} cells, Low group: {len(low_indices)} cells")

        # Extract feature scores for both groups
        high_scores = feature_scores[high_indices, :]
        low_scores = feature_scores[low_indices, :]

        # Test each feature
        n_features_tested = 0
        for feature_idx, feature_name in enumerate(feature_names):
            high_feature_scores = high_scores[:, feature_idx]
            low_feature_scores = low_scores[:, feature_idx]

            # Skip features with no variance
            if np.var(high_feature_scores) == 0 and np.var(low_feature_scores) == 0:
                continue

            try:
                # Use two-sided test for pattern detection
                statistic, pvalue = robust_mannwhitneyu_test(
                    high_feature_scores,
                    low_feature_scores,
                    alternative="two-sided",
                    tie_breaking=True,
                    feature_name=feature_name,
                )

                # STANDARDIZED EFFECT SIZE CALCULATION
                mean_high = np.mean(high_feature_scores)
                mean_low = np.mean(low_feature_scores)

                # CRITICAL: Standardized effect size handling
                if is_pathway_data:
                    # PATHWAY SCORES: Always use mean difference (activity scores)
                    primary_effect_size = mean_high - mean_low
                    effect_size_col = "mean_diff"
                    # Set log_fold_change to mean_diff for backward compatibility
                    log_fold_change = primary_effect_size
                else:
                    # GENE DATA: Use smart log fold change detection
                    combined_scores = np.concatenate([high_feature_scores, low_feature_scores])
                    max_val = np.max(combined_scores)
                    min_val = np.min(combined_scores[combined_scores > 0]) if np.any(combined_scores > 0) else 0

                    # Detect if data appears to be log-transformed
                    if max_val > 50 or (min_val > 0 and max_val / min_val > 1000):
                        # Raw count data - use log2 ratio
                        if mean_low > 0:
                            primary_effect_size = np.log2((mean_high + 1) / (mean_low + 1))
                        else:
                            primary_effect_size = np.log2(mean_high + 1)
                    else:
                        # Log-transformed data - use simple difference
                        primary_effect_size = mean_high - mean_low

                    effect_size_col = "log_fold_change"
                    log_fold_change = primary_effect_size

                # Apply effect size threshold filter using the appropriate column
                if abs(primary_effect_size) >= min_logfc:
                    direction = "higher" if mean_high > mean_low else "lower"

                    # Create result compatible with existing plotting functions
                    result = {
                        data_type: feature_name,  # Dynamic key: 'gene' or 'pathway'
                        "pattern_name": pattern["pattern_name"],  # NEW: Interpretable pattern name
                        "pattern_code": pattern["pattern_code"],  # NEW: Visual pattern code
                        "pattern_type": pattern["pattern_type"],
                        "high_archetypes": ",".join(pattern["high_archetypes"]),
                        "low_archetypes": ",".join(pattern["low_archetypes"]),
                        "n_high_cells": len(high_indices),
                        "n_low_cells": len(low_indices),
                        "mean_high": mean_high,
                        "mean_low": mean_low,
                        "log_fold_change": log_fold_change,  # For backward compatibility
                        "statistic": statistic,
                        "pvalue": pvalue,
                        "test_direction": "two-sided",
                        "direction": direction,
                        "passes_lfc_threshold": True,
                        "effect_size_col": effect_size_col,  # NEW: Track which effect size to use
                    }

                    # STANDARDIZED: Add the proper effect size column based on data type
                    if is_pathway_data:
                        result["mean_diff"] = primary_effect_size
                        result["primary_effect_size"] = primary_effect_size  # Standard name
                    else:
                        result["log_fold_change"] = primary_effect_size  # Already set above
                        result["primary_effect_size"] = primary_effect_size  # Standard name

                    # Add compatibility aliases for plotting functions
                    result["archetype"] = pattern["pattern_name"]  # For x-axis in plots
                    result["mean_archetype"] = mean_high  # For size in dotplots
                    result["mean_other"] = mean_low
                    result["n_archetype_cells"] = len(high_indices)
                    result["n_other_cells"] = len(low_indices)

                    results.append(result)

                n_features_tested += 1

            except Exception:
                continue

        if verbose:
            print(f"      Tested {n_features_tested} features")

    if not results:
        # Provide detailed error message
        error_msg = "No valid pattern tests performed."
        if verbose:
            error_msg += f"\n  Generated {len(patterns)} patterns but none were tested."
            error_msg += "\n  Possible causes:"
            error_msg += f"\n    - Insufficient cells per group (min_cells={min_cells})"
            error_msg += "\n    - No variation in data for the features"
            error_msg += "\n    - All patterns filtered out during testing"
            if len(patterns) == 0:
                error_msg += "\n    - No patterns generated! Check archetype naming and parameters."
        raise ValueError(error_msg)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Apply FDR correction with scope support (like other functions)
    if verbose:
        print(f"\n   Applying FDR correction ({fdr_method}, {fdr_scope} scope)...")
        print(f"   Total tests performed: {len(results_df)}")

    if fdr_scope == "global":
        # Global FDR correction across all tests
        results_df = apply_fdr_correction(
            results_df, pvalue_column="pvalue", method=fdr_method, alpha=0.05, verbose=verbose
        )
        if verbose:
            n_significant = results_df["significant"].sum()
            print(f"    Applied global FDR correction: {n_significant}/{len(results_df)} significant")

    elif fdr_scope == "per_archetype":
        # Per-pattern FDR correction (adapted for patterns instead of archetypes)
        results_df["fdr_pvalue"] = np.nan
        results_df["significant"] = False

        for pattern_name in results_df["pattern_name"].unique():
            pattern_mask = results_df["pattern_name"] == pattern_name
            pattern_results = results_df[pattern_mask].copy()

            if len(pattern_results) > 0:
                pattern_results = apply_fdr_correction(
                    pattern_results, pvalue_column="pvalue", method=fdr_method, alpha=0.05, verbose=False
                )
                results_df.loc[pattern_mask, "fdr_pvalue"] = pattern_results["fdr_pvalue"]
                results_df.loc[pattern_mask, "significant"] = pattern_results["significant"]

        if verbose:
            n_significant = results_df["significant"].sum()
            print(f"    Applied per-pattern FDR correction: {n_significant}/{len(results_df)} significant")

    elif fdr_scope == "none":
        # No FDR correction - use raw p-values
        results_df["fdr_pvalue"] = results_df["pvalue"]
        results_df["significant"] = results_df["pvalue"] < 0.05

        if verbose:
            n_significant = results_df["significant"].sum()
            print(f"    No FDR correction applied: {n_significant}/{len(results_df)} significant (raw p<0.05)")

    else:
        raise ValueError(f"Invalid fdr_scope: {fdr_scope}. Must be 'global', 'per_archetype', or 'none'")

    # Sort by significance and effect size
    results_df["abs_log_fold_change"] = results_df["log_fold_change"].abs()
    results_df = results_df.sort_values(["fdr_pvalue", "pvalue", "abs_log_fold_change"], ascending=[True, True, False])
    results_df = results_df.drop(columns=["abs_log_fold_change"])

    if verbose:
        n_significant = results_df["significant"].sum()
        total_tests = len(results_df)
        print("[OK] Pattern association testing completed!")
        print(f"   Total tests: {total_tests}")
        print(f"   Significant associations: {n_significant} ({100 * n_significant / total_tests:.1f}%)")

        # Show breakdown by pattern type
        if "pattern_type" in results_df.columns:
            type_counts = results_df[results_df["significant"]].groupby("pattern_type").size()
            if len(type_counts) > 0:
                print(f"   Significant by pattern type: {dict(type_counts)}")

    return results_df


def identify_mutual_exclusivity_patterns(
    pattern_results_df: pd.DataFrame,
    specialization_results_df: pd.DataFrame = None,
    min_effect_size: float = 0.05,
    significance_threshold: float = 0.05,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Identify genes/pathways showing mutual exclusivity between archetype patterns.

    Finds features that have significant positive effects in some patterns and
    significant negative effects in others, indicating biological tradeoffs.

    Parameters
    ----------
    pattern_results_df : pd.DataFrame
        Results from test_archetype_pattern_associations().
        Must contain 'primary_effect_size' or 'mean_diff'/'log_fold_change' columns.
    specialization_results_df : pd.DataFrame | None, default: None
        Optional individual archetype results for comparison.
    min_effect_size : float, default: 0.05
        Minimum absolute effect size for inclusion.
        Uses 'primary_effect_size' column (mean_diff for pathways, log_fc for genes).
    significance_threshold : float, default: 0.05
        P-value threshold for significance.
    verbose : bool, default: True
        Whether to print analysis summary.

    Returns
    -------
    pd.DataFrame
        Mutual exclusivity results with columns:

        **Feature identifier** (one present based on data type):

        - ``pathway`` : str - Pathway name (if pathway data)
        - ``gene`` : str - Gene name (if gene data)

        **Pattern lists**:

        - ``positive_patterns`` : list[str] - Patterns where feature is high
        - ``negative_patterns`` : list[str] - Patterns where feature is low
        - ``positive_pattern_codes`` : list[str] - Visual codes for positive patterns
        - ``negative_pattern_codes`` : list[str] - Visual codes for negative patterns

        **Scores**:

        - ``tradeoff_score`` : int - Number of patterns involved (higher = more complex)
        - ``max_positive_effect`` : float - Maximum positive effect size
        - ``min_negative_effect`` : float - Minimum negative effect size
        - ``effect_range`` : float - Range of effects (max_positive - min_negative)
        - ``primary_effect_size`` : float - Standardized effect size
        - ``effect_size_col`` : str - Effect size column used
        - ``effect_range_name`` : str - Name for effect range ('mean_diff_range' or 'lfc_range')
        - ``mean_diff`` : float - Mean difference (for pathways)

        **Compatibility fields** (for plotting functions):

        - ``archetype`` : str - Pattern identifier
        - ``pattern_name`` : str - Pattern identifier
        - ``pattern_code`` : str - Visual pattern code
        - ``log_fold_change`` : float - For plotting compatibility
        - ``mean_archetype`` : float - Equals tradeoff_score (size in plots)
        - ``pvalue`` : float - Mock p-value (0.01)
        - ``fdr_pvalue`` : float - Mock FDR p-value (0.05)
        - ``significant`` : bool - Always True (pre-filtered)

    Notes
    -----
    **Standardized Effect Size Handling**:

    Uses the 'primary_effect_size' column from pattern results, which contains:

    - ``mean_diff`` for pathway data (activity scores)
    - ``log_fold_change`` for gene data (expression levels)

    The function auto-detects data type from the 'effect_size_col' column.

    See Also
    --------
    test_archetype_pattern_associations : Generate pattern results
    analyze_archetypal_patterns_comprehensive : Full analysis pipeline
    peach._core.types.MutualExclusivityResult : Result row type definition

    Examples
    --------
    >>> # Run pattern analysis first
    >>> pattern_results = test_archetype_pattern_associations(adata)
    >>> # Find mutual exclusivity
    >>> exclusivity = identify_mutual_exclusivity_patterns(pattern_results, min_effect_size=0.1)
    >>> # Top tradeoff features
    >>> top_tradeoffs = exclusivity.nlargest(10, "effect_range")
    """
    # STANDARDIZED EFFECT SIZE DETECTION: Use the new standardized columns
    if "primary_effect_size" in pattern_results_df.columns and "effect_size_col" in pattern_results_df.columns:
        # NEW: Use standardized approach
        effect_size_col = "primary_effect_size"
        is_pathway_data = pattern_results_df["effect_size_col"].iloc[0] == "mean_diff"
    else:
        # FALLBACK: Use legacy detection method
        is_pathway_data = "mean_diff" in pattern_results_df.columns
        effect_size_col = "mean_diff" if is_pathway_data else "log_fold_change"

    if verbose:
        data_type = "pathway" if is_pathway_data else "gene"
        print(f"   Using effect size column: {effect_size_col} (detected: {data_type} data)")
        print(f"   Input data shape: {pattern_results_df.shape}")
        print(f"   Significance threshold: {significance_threshold}")
        print(f"   Min effect size: {min_effect_size}")

    # DEBUG: Check filtering steps
    significant_mask = pattern_results_df["fdr_pvalue"] < significance_threshold
    strong_effect_mask = pattern_results_df[effect_size_col].abs() > min_effect_size

    if verbose:
        print(f"   Significant results: {significant_mask.sum()}/{len(pattern_results_df)}")
        print(f"   Strong effects (>{min_effect_size}): {strong_effect_mask.sum()}/{len(pattern_results_df)}")
        print(
            f"   Effect size range: [{pattern_results_df[effect_size_col].min():.3f}, {pattern_results_df[effect_size_col].max():.3f}]"
        )

        # AUTO-SUGGEST better thresholds if current ones are too restrictive
        if strong_effect_mask.sum() == 0 and significant_mask.sum() > 0:
            suggested_threshold = pattern_results_df[significant_mask][effect_size_col].abs().quantile(0.75)
            print(
                f"   NOTE:  Suggestion: Try min_effect_size={suggested_threshold:.3f} (75th percentile of significant effects)"
            )
        elif strong_effect_mask.sum() < 10 and significant_mask.sum() > 10:
            suggested_threshold = pattern_results_df[significant_mask][effect_size_col].abs().quantile(0.5)
            print(
                f"   NOTE:  Suggestion: Try min_effect_size={suggested_threshold:.3f} (median of significant effects)"
            )

    # Filter for significant associations with strong effects
    sig_strong = pattern_results_df[significant_mask & strong_effect_mask].copy()

    if sig_strong.empty:
        if verbose:
            print("No significant strong associations found for mutual exclusivity analysis.")
        return pd.DataFrame()

    # Identify features with opposing patterns
    mutual_exclusivity_features = []

    feature_col = "pathway" if "pathway" in sig_strong.columns else "gene"

    for feature in sig_strong[feature_col].unique():
        feature_results = sig_strong[sig_strong[feature_col] == feature]

        # Look for opposing directions using the appropriate effect size column
        has_positive = (feature_results[effect_size_col] > min_effect_size).any()
        has_negative = (feature_results[effect_size_col] < -min_effect_size).any()

        if has_positive and has_negative:
            positive_patterns = feature_results[feature_results[effect_size_col] > min_effect_size][
                "archetype"
            ].tolist()

            negative_patterns = feature_results[feature_results[effect_size_col] < -min_effect_size][
                "archetype"
            ].tolist()

            # Calculate trade-off score
            tradeoff_score = len(positive_patterns) + len(negative_patterns)
            max_positive_effect = feature_results[effect_size_col].max()
            min_negative_effect = feature_results[effect_size_col].min()

            # Extract pattern codes for better visualization
            positive_pattern_codes = []
            negative_pattern_codes = []

            for pattern_name in positive_patterns:
                pattern_match = feature_results[feature_results["archetype"] == pattern_name]
                if not pattern_match.empty and "pattern_code" in pattern_match.columns:
                    positive_pattern_codes.append(pattern_match["pattern_code"].iloc[0])
                else:
                    positive_pattern_codes.append(pattern_name)  # Fallback

            for pattern_name in negative_patterns:
                pattern_match = feature_results[feature_results["archetype"] == pattern_name]
                if not pattern_match.empty and "pattern_code" in pattern_match.columns:
                    negative_pattern_codes.append(pattern_match["pattern_code"].iloc[0])
                else:
                    negative_pattern_codes.append(pattern_name)  # Fallback

            # Create interpretable exclusivity pattern name (simplified to avoid redundancy)
            exclusivity_pattern_code = f"exclusivity_{tradeoff_score}_patterns"
            if positive_pattern_codes and negative_pattern_codes:
                # Use only the first pattern code to avoid redundancy like "01xxx_xx234_vs_xx234_01xxx"
                # Since mutual exclusivity is bidirectional, we only need one direction
                main_positive = positive_pattern_codes[0] if positive_pattern_codes else "pos"
                # For mutual exclusivity, the pattern is self-descriptive, no need for "vs" redundancy
                exclusivity_pattern_code = f"exclusivity_{main_positive}"

            # Add compatibility columns for dotplot visualization
            exclusivity_result = {
                feature_col: feature,
                "positive_patterns": positive_patterns,
                "negative_patterns": negative_patterns,
                "positive_pattern_codes": positive_pattern_codes,  # NEW: Visual codes
                "negative_pattern_codes": negative_pattern_codes,  # NEW: Visual codes
                "tradeoff_score": tradeoff_score,
                "max_positive_effect": max_positive_effect,
                "min_negative_effect": min_negative_effect,
                "effect_range": max_positive_effect - min_negative_effect,
                "primary_effect_size": max_positive_effect,  # NEW: Standardized effect size
                "effect_size_col": "mean_diff" if is_pathway_data else "log_fold_change",  # NEW: Track type
                # Add missing statistical columns for dotplot compatibility
                "log_fold_change": max_positive_effect,  # For backward compatibility
                "mean_archetype": tradeoff_score,  # Use tradeoff score as size proxy
                "pvalue": 0.01,  # Mock p-value for patterns
                "fdr_pvalue": 0.05,  # Mock FDR-corrected p-value
                "significant": True,  # Patterns are pre-filtered
                "archetype": exclusivity_pattern_code,  # NEW: Interpretable pattern code
                "pattern_name": exclusivity_pattern_code,  # NEW: For consistency
                "pattern_code": exclusivity_pattern_code,  # NEW: Visual pattern code
            }

            # STANDARDIZED: Add the proper effect size column for the detected data type
            if is_pathway_data:
                exclusivity_result["mean_diff"] = max_positive_effect
                exclusivity_result["effect_range_name"] = "mean_diff_range"
            else:
                exclusivity_result["log_fold_change"] = max_positive_effect  # Already added above
                exclusivity_result["effect_range_name"] = "lfc_range"

            mutual_exclusivity_features.append(exclusivity_result)

    if mutual_exclusivity_features:
        exclusivity_df = pd.DataFrame(mutual_exclusivity_features)
        exclusivity_df = exclusivity_df.sort_values("effect_range", ascending=False)

        if verbose:
            effect_name = "mean_diff" if is_pathway_data else "log_fold_change"
            print(" Mutual Exclusivity Analysis:")
            print(f"   Found {len(exclusivity_df)} features with opposing patterns")
            print("   Top trade-off features:")
            for _, row in exclusivity_df.head(5).iterrows():
                print(f"      {row[feature_col]}: {effect_name} range = {row['effect_range']:.2f}")

        return exclusivity_df
    else:
        if verbose:
            print("No mutual exclusivity patterns identified.")
        return pd.DataFrame()


# Convenience function for comprehensive pattern analysis
def analyze_archetypal_patterns_comprehensive(
    adata,
    data_obsm_key: str = "pathway_scores",  # GENERALIZED: Now accepts any data type
    obs_key: str = "archetypes",
    include_individual_tests: bool = True,
    include_pattern_tests: bool = True,
    include_exclusivity_analysis: bool = True,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Run comprehensive archetypal pattern analysis combining multiple approaches.

    WHAT THIS FUNCTION PROVIDES:
    ===========================
    Complete archetypal characterization through three complementary analyses:

    1. INDIVIDUAL TESTS (all_results['individual']):
       - Standard 1-vs-all archetype characterization
       - Broad feature associations for each archetype vs population average
       - Most comprehensive results (baseline characterization)

    2. PATTERN TESTS (all_results['patterns']):
       - Systematic archetype combination testing
       - Pattern_1_archetypes: Specialists (exclusive to one archetype)
       - Pattern_2_archetypes: Binary trade-offs (high in some, low in others)
       - Pattern_3+_archetypes: Complex multi-way trade-offs

    3. EXCLUSIVITY ANALYSIS (all_results['exclusivity']):
       - Features with mutually exclusive patterns across archetypes
       - True archetype specialists with opposing activity patterns
       - Subset of pattern results meeting strict exclusivity criteria

    UNDERSTANDING PATTERN_N_ARCHETYPES:
    ==================================
    The pattern naming in exclusivity results indicates the complexity of trade-offs:

    Pattern_1_archetypes: SPECIALISTS
    - Features high in ONE archetype, low in ALL others
    - Example: "Pattern_archetype_2" = features exclusive to archetype 2
    - Biological meaning: Unique cellular programs

    Pattern_2_archetypes: BINARY TRADE-OFFS
    - Features high in one archetype, low in one other (2-way competition)
    - Example: "Pattern_archetype_1_vs_archetype_3" = 1 high, 3 low
    - Biological meaning: Competing cellular programs

    Pattern_3+_archetypes: COMPLEX TRADE-OFFS
    - Features with complex multi-archetype opposing patterns
    - Example: "Pattern_archetype_1_vs_archetype_2_vs_archetype_4"
    - Biological meaning: Multi-dimensional cellular specialization

    Args:
        adata: AnnData object with gene/pathway scores and archetype assignments
        data_obsm_key: Key for gene/pathway scores in adata.obsm
        obs_key: Key for archetype assignments in adata.obs
        include_individual_tests: Run individual archetype 1-vs-all tests
        include_pattern_tests: Run systematic pattern tests
        include_exclusivity_analysis: Analyze mutual exclusivity patterns
        verbose: Print progress information

    Returns
    -------
        Dictionary with keys: 'individual', 'patterns', 'exclusivity'
        Each contains a pandas DataFrame with test results

    Examples
    --------
        # Complete workflow
        all_results = analyze_archetypal_patterns_comprehensive(
            adata,
            data_obsm_key='pathway_scores',
            verbose=True
        )

        # SUBSETTING EXAMPLES FOR COMPREHENSIBLE ANALYSIS (UPDATED FOR NEW NAMING):
        # ========================================================================

        # 1. Focus on specialists only (using pattern codes)
        specialists_only = all_results['patterns'][
            all_results['patterns']['pattern_type'] == 'specialization'
        ]

        # 2. Focus on binary trade-offs (2-way patterns in pattern_code)
        binary_tradeoffs = all_results['patterns'][
            (all_results['patterns']['pattern_type'] == 'tradeoff') &
            (all_results['patterns']['pattern_code'].str.count('[0-9]') == 2)  # Exactly 2 numbers
        ]

        # 3. Visual pattern filtering using interpretable codes
        # Pattern codes like '12xxx_xx345' make filtering intuitive:

        # Archetype 1 involved in any pattern:
        arch1_patterns = all_results['patterns'][
            all_results['patterns']['pattern_code'].str.contains('1')
        ]

        # Archetype 0 vs any other (generalist vs specialist):
        generalist_patterns = all_results['patterns'][
            all_results['patterns']['pattern_code'].str.match(r'0xxxx_x[1-9]xxx|x[1-9]xxx_0xxxx')
        ]

        # Simple 2-archetype comparisons:
        simple_comparisons = all_results['patterns'][
            all_results['patterns']['pattern_code'].str.match(r'[0-9]xxxx_x[0-9]xxx')
        ]

        # 4. Focus on specific archetype interactions (NEW: using visual codes)
        archetype_1_vs_2 = all_results['patterns'][
            all_results['patterns']['pattern_code'].str.contains('1xxxx_x2xxx|x1xxx_2xxxx')
        ]

        # 5. Filter by biological relevance (using standardized effect sizes)
        strong_effects = all_results['patterns'][
            (all_results['patterns']['significant']) &
            (all_results['patterns']['primary_effect_size'].abs() > 0.1)  # Strong effect
        ]

        # 6. Focus on exclusivity patterns with interpretable codes
        if not all_results['exclusivity'].empty:
            # Strong exclusivity patterns
            strong_exclusivity = all_results['exclusivity'][
                (all_results['exclusivity']['effect_range'] > 0.2) &  # Strong effect range
                (all_results['exclusivity']['tradeoff_score'] >= 2)    # Multiple opposing patterns
            ]

            # Patterns involving specific archetypes (using pattern codes)
            arch1_exclusivity = all_results['exclusivity'][
                all_results['exclusivity']['pattern_code'].str.contains('1')
            ]

        # 7. Focus on specific biological processes (pathways)
        if 'pathway' in all_results.get('individual', pd.DataFrame()).columns:
            metabolism_patterns = all_results['individual'][
                all_results['individual']['pathway'].str.contains('METABOLISM|GLYCOLYSIS', case=False)
            ]

        # PLOTTING WORKFLOW WITH SMART FILTERING (UPDATED FOR NEW CODES):
        # ==============================================================

        # Plot 1: Individual archetype characteristics (broadest view)
        fig_individual = create_dotplot_visualization(
            all_results['individual'].head(100),  # Limit for clarity
            title='Individual Archetype Associations'
        )

        # Plot 2: Focus on specialists (clearest biological interpretation)
        if not specialists_only.empty:
            fig_specialists = create_dotplot_visualization(
                specialists_only,
                x_col='pattern_name',  # Shows interpretable names like 'specialist_arch1_1xxxx_0xxxx'
                title='Archetype Specialists (Exclusive Features)'
            )

        # Plot 3: Binary trade-offs (interpretable competition)
        if not binary_tradeoffs.empty:
            fig_binary = create_dotplot_visualization(
                binary_tradeoffs.head(50),  # Limit complexity
                x_col='pattern_name',  # Shows codes like 'tradeoff_12xxx_xx345'
                title='Binary Archetype Trade-offs'
            )

        # Plot 4: Strong effects using standardized effect sizes
        if not strong_effects.empty:
            fig_strong = create_dotplot_visualization(
                strong_effects,
                x_col='pattern_name',
                size_col='primary_effect_size',  # NEW: Standardized effect size
                title='Strong Pattern Associations'
            )

        # Plot 5: Exclusivity patterns with interpretable naming
        if not all_results['exclusivity'].empty:
            fig_exclusivity = create_dotplot_visualization(
                all_results['exclusivity'],
                title='Mutual Exclusivity Patterns'
                # Auto-detects columns, now includes pattern_code for legend clarity
            )
    """
    results = {}

    # AUTO-DETECT DATA TYPE for appropriate function selection
    is_pathway_data = "pathway" in data_obsm_key.lower()

    if include_individual_tests:
        if verbose:
            print("=" * 60)
            print("1. INDIVIDUAL ARCHETYPE TESTING (1-vs-all)")
            print("=" * 60)

        # CRITICAL FIX: Use the appropriate function based on data type
        if is_pathway_data:
            individual_results = test_archetype_pathway_associations(
                adata=adata,
                pathway_obsm_key=data_obsm_key,  # Use the generalized parameter
                obs_key=obs_key,
                verbose=verbose,
            )
        else:
            # For gene data, use the gene testing function
            individual_results = test_archetype_gene_associations(
                adata=adata,
                obsm_key="archetype_distances",  # Standard distance key
                obs_key=obs_key,
                verbose=verbose,
            )
        results["individual"] = individual_results

    if include_pattern_tests:
        if verbose:
            print("\n" + "=" * 60)
            print("2. PATTERN TESTING (systematic combinations)")
            print("=" * 60)

        # CRITICAL FIX: Use the generalized function with updated parameter name
        pattern_results = test_archetype_pattern_associations(
            adata=adata,
            data_obsm_key=data_obsm_key,  # Use the generalized parameter
            obs_key=obs_key,
            pattern_types=["specialization", "tradeoff"],
            verbose=verbose,
        )
        results["patterns"] = pattern_results

    if include_exclusivity_analysis and len(results) > 0:
        if verbose:
            print("\n" + "=" * 60)
            print("3. MUTUAL EXCLUSIVITY ANALYSIS")
            print("=" * 60)

        # Combine results for exclusivity analysis
        main_results = results.get("patterns", results.get("individual"))
        if main_results is not None:
            exclusivity_results = identify_mutual_exclusivity_patterns(pattern_results_df=main_results, verbose=verbose)
            results["exclusivity"] = exclusivity_results

    return results


# =============================================================================
# NEW FOCUSED PATTERN ANALYSIS FUNCTIONS
# =============================================================================


def _identify_exclusive_patterns_pairwise(
    adata,
    data_obsm_key: str = "pathway_scores",
    obs_key: str = "archetypes",
    test_method: str = "mannwhitneyu",
    fdr_method: str = "benjamini_hochberg",
    fdr_scope: str = "global",
    min_cells: int = 10,
    min_effect_size: float = 0.05,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Helper function for true pairwise exclusive pattern identification.

    Tests each archetype against EVERY other archetype individually.
    A feature is exclusive if significantly higher in one archetype vs ALL others.
    """
    # Get archetype assignments
    archetype_assignments = adata.obs[obs_key]

    # Handle both categorical and non-categorical columns
    if hasattr(archetype_assignments, "cat"):
        unique_archetypes = [cat for cat in archetype_assignments.cat.categories if cat != "no_archetype"]
    else:
        unique_archetypes = [
            arch for arch in archetype_assignments.unique() if arch != "no_archetype" and pd.notna(arch)
        ]

    if verbose:
        print(f"   Found {len(unique_archetypes)} archetypes for pairwise testing")
        n_comparisons = len(unique_archetypes) * (len(unique_archetypes) - 1) // 2
        print(f"   Will perform {n_comparisons} pairwise comparisons per feature")

    # Determine data type and get feature data
    is_pathway_data = data_obsm_key is not None and data_obsm_key in adata.obsm
    data_type = "pathway" if is_pathway_data else "gene"
    effect_col = "mean_diff" if is_pathway_data else "log_fold_change"

    # Get feature data and names
    if is_pathway_data:
        feature_data = adata.obsm[data_obsm_key]
        feature_names_key = f"{data_obsm_key}_pathways"
        if feature_names_key in adata.uns:
            feature_names = adata.uns[feature_names_key]
        else:
            feature_names = [f"pathway_{i}" for i in range(feature_data.shape[1])]
    else:
        # Gene expression data
        feature_data = adata.X
        feature_names = adata.var_names.tolist()
        # Convert sparse to dense if needed
        if hasattr(feature_data, "toarray"):
            feature_data = feature_data.toarray()

    # Create cell index maps for each archetype
    archetype_indices = {}
    for arch in unique_archetypes:
        mask = archetype_assignments == arch
        indices = np.where(mask)[0]
        if len(indices) >= min_cells:
            archetype_indices[arch] = indices
        elif verbose:
            print(f"   Skipping {arch}: only {len(indices)} cells (min: {min_cells})")

    if len(archetype_indices) < 2:
        if verbose:
            print("   Need at least 2 archetypes with sufficient cells for exclusivity testing")
        return pd.DataFrame()

    # For each feature, test all pairwise comparisons
    exclusive_results = []
    n_features = feature_data.shape[1]

    if verbose:
        print(f"   Testing {n_features} features across all archetype pairs...")

    for feat_idx in range(n_features):
        feature_name = feature_names[feat_idx]

        # Track which archetypes are significantly higher than ALL others
        archetype_wins = {}  # arch -> list of archs it beats

        for test_arch in archetype_indices:
            test_indices = archetype_indices[test_arch]
            test_scores = feature_data[test_indices, feat_idx]

            wins_against_all = True
            min_effect = 0.05  # float('inf')
            max_pvalue = 0.05

            # Test against each other archetype
            for other_arch in archetype_indices:
                if other_arch == test_arch:
                    continue

                other_indices = archetype_indices[other_arch]
                other_scores = feature_data[other_indices, feat_idx]

                # Skip if no variance
                if np.var(test_scores) == 0 and np.var(other_scores) == 0:
                    wins_against_all = False
                    break

                # Perform pairwise test
                try:
                    statistic, pvalue = robust_mannwhitneyu_test(
                        test_scores,
                        other_scores,
                        alternative="greater",  # Test if test_arch > other_arch
                        tie_breaking=True,
                        feature_name=f"{feature_name}:{test_arch}_vs_{other_arch}",
                    )

                    # Calculate effect size
                    mean_test = np.mean(test_scores)
                    mean_other = np.mean(other_scores)

                    if is_pathway_data:
                        # Mean difference for pathway scores
                        effect_size = mean_test - mean_other
                    else:
                        # Log fold change for gene expression
                        if mean_other > 0:
                            effect_size = np.log2((mean_test + 1) / (mean_other + 1))
                        else:
                            effect_size = np.log2(mean_test + 1) if mean_test > 0 else 0

                    # Check if this archetype beats the other
                    if pvalue >= 0.05 or effect_size < min_effect_size:
                        wins_against_all = False
                        break

                    # Track minimum effect and maximum p-value
                    min_effect = min(min_effect, effect_size)
                    max_pvalue = max(max_pvalue, pvalue)

                except Exception:
                    wins_against_all = False
                    break

            if wins_against_all:
                archetype_wins[test_arch] = {
                    "min_effect": min_effect,
                    "max_pvalue": max_pvalue,
                    "mean": np.mean(feature_data[test_indices, feat_idx]),
                }

        # Check if exactly one archetype wins against all others
        if len(archetype_wins) == 1:
            winner_arch = list(archetype_wins.keys())[0]
            winner_info = archetype_wins[winner_arch]

            # Calculate mean across all other cells for comparison
            all_other_indices = []
            for arch in archetype_indices:
                if arch != winner_arch:
                    all_other_indices.extend(archetype_indices[arch])

            other_mean = np.mean(feature_data[all_other_indices, feat_idx])

            # Create result row
            result = {
                data_type: feature_name,
                "archetype": winner_arch,
                "n_archetype_cells": len(archetype_indices[winner_arch]),
                "n_other_cells": len(all_other_indices),
                "mean_archetype": winner_info["mean"],
                "mean_other": other_mean,
                effect_col: winner_info["min_effect"],
                "min_pairwise_effect": winner_info["min_effect"],
                "max_pairwise_pvalue": winner_info["max_pvalue"],
                "pvalue": winner_info["max_pvalue"],  # Use most conservative p-value
                "statistic": 0,  # Placeholder
                "direction": "higher",
                "pattern_type": "exclusive_pairwise",
                "exclusivity_score": winner_info["min_effect"] / (min_effect_size + 1e-6),
            }

            exclusive_results.append(result)

    if not exclusive_results:
        if verbose:
            print("   No features showed true pairwise exclusivity")
        return pd.DataFrame()

    # Create DataFrame and apply FDR correction
    results_df = pd.DataFrame(exclusive_results)

    # Apply FDR correction
    if fdr_scope != "none":
        results_df = apply_fdr_correction(
            results_df, pvalue_column="pvalue", method=fdr_method, alpha=0.05, verbose=False
        )
    else:
        results_df["fdr_pvalue"] = results_df["pvalue"]
        results_df["significant"] = results_df["pvalue"] < 0.05

    # Standardize results
    results_df = _standardize_results_dataframe(results_df, pattern_type="exclusive", data_type=data_type)

    if verbose:
        print(f"   Found {len(results_df)} features with pairwise exclusivity")
        if len(results_df) > 0:
            print("   Distribution by archetype:")
            for arch, count in results_df["archetype"].value_counts().items():
                print(f"      {arch}: {count} features")

    return results_df.sort_values("exclusivity_score", ascending=False)


def identify_archetype_exclusive_patterns(
    adata,
    data_obsm_key: str = "pathway_scores",
    obs_key: str = "archetypes",
    test_method: str = "mannwhitneyu",
    fdr_method: str = "benjamini_hochberg",
    fdr_scope: str = "global",
    min_cells: int = 10,
    min_effect_size: float = 0.05,
    use_pairwise: bool = True,  # NEW: Choose between pairwise and 1-vs-all
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Identify genes or pathways that are exclusively high in only one archetype.

    Supports two methods:
    1. **Pairwise method (use_pairwise=True, DEFAULT)**:
       - Tests each archetype against EVERY other archetype individually
       - A feature is exclusive if significantly higher in one archetype vs ALL others
       - More stringent but computationally intensive

    2. **1-vs-all filtering (use_pairwise=False)**:
       - Tests each archetype vs all other cells combined
       - Features exclusive if significant in only ONE archetype's 1-vs-all test
       - More statistical power but may miss subtle patterns

    Args:
        adata: AnnData object with archetype assignments
        data_obsm_key: Key for data in adata.obsm ('pathway_scores' for pathways, use None for genes)
        obs_key: Key for archetype assignments in adata.obs
        test_method: Statistical test method ('mannwhitneyu')
        fdr_method: Multiple testing correction method
        fdr_scope: FDR correction scope ('global', 'per_archetype', 'none')
        min_cells: Minimum cells per archetype for testing
        min_effect_size: Minimum effect size for exclusivity (mean_diff for pathways, log_fc for genes)
        use_pairwise: If True, use pairwise comparisons; if False, use 1-vs-all filtering
        verbose: Whether to print progress

    Returns
    -------
        Standardized DataFrame with columns:
        - pathway/gene: Feature identifier
        - archetype: Exclusive archetype
        - n_archetype_cells: Number of cells in exclusive archetype
        - n_other_cells: Number of cells in all other archetypes
        - mean_archetype: Mean in exclusive archetype
        - mean_other: Mean in other archetypes
        - mean_diff/log_fold_change: Effect size
        - statistic: Test statistic
        - pvalue: Raw p-value
        - fdr_pvalue: FDR-corrected p-value
        - significant: Boolean significance
        - direction: Always 'higher' for exclusive patterns
        - pattern_type: Always 'exclusive'
        - exclusivity_score: Ratio of expression in target vs max other archetype

    Example:
        >>> exclusive = identify_archetype_exclusive_patterns(adata)
        >>> # Find CD8+ T cell exclusive markers
        >>> cd8_markers = exclusive[exclusive["archetype"] == "archetype_3"]
    """
    if verbose:
        print(" Identifying Archetype-Exclusive Patterns...")
        method_name = "pairwise comparisons" if use_pairwise else "1-vs-all filtering"
        print(f"   Method: {method_name}")
        print(f"   Min effect size: {min_effect_size}")

    # Determine data type
    is_pathway_data = data_obsm_key is not None and data_obsm_key in adata.obsm
    data_type = "pathway" if is_pathway_data else "gene"

    # Branch based on method
    if use_pairwise:
        return _identify_exclusive_patterns_pairwise(
            adata=adata,
            data_obsm_key=data_obsm_key,
            obs_key=obs_key,
            test_method=test_method,
            fdr_method=fdr_method,
            fdr_scope=fdr_scope,
            min_cells=min_cells,
            min_effect_size=min_effect_size,
            verbose=verbose,
        )

    # Otherwise use 1-vs-all filtering approach
    # First get individual archetype associations
    if is_pathway_data:
        individual_results = test_archetype_pathway_associations(
            adata=adata,
            pathway_obsm_key=data_obsm_key,
            obs_key=obs_key,
            test_method=test_method,
            fdr_method=fdr_method,
            fdr_scope=fdr_scope,
            min_cells=min_cells,
            verbose=False,
        )
    else:
        individual_results = test_archetype_gene_associations(
            adata=adata,
            obs_key=obs_key,
            test_method=test_method,
            fdr_method=fdr_method,
            fdr_scope=fdr_scope,
            min_cells=min_cells,
            verbose=False,
        )

    # Identify features that are exclusive to one archetype
    exclusive_results = []
    feature_col = "pathway" if is_pathway_data else "gene"
    effect_col = "mean_diff" if is_pathway_data else "log_fold_change"

    # Group by feature to check exclusivity
    for feature, feature_df in individual_results.groupby(feature_col):
        # Get significant positive associations
        significant_positive = feature_df[(feature_df["significant"]) & (feature_df["direction"] == "higher")]

        # Check if exactly one archetype shows significant positive association
        if len(significant_positive) == 1:
            exclusive_row = significant_positive.iloc[0].copy()

            # Calculate exclusivity score (ratio to next highest)
            all_effects = feature_df[effect_col].values
            target_effect = exclusive_row[effect_col]
            other_effects = all_effects[all_effects != target_effect]

            if len(other_effects) > 0:
                max_other = np.max(other_effects)
                exclusivity_score = target_effect / (max_other + 1e-6)  # Avoid division by zero
            else:
                exclusivity_score = float("inf")

            # Add exclusive pattern metadata
            exclusive_row["pattern_type"] = "exclusive"
            exclusive_row["exclusivity_score"] = exclusivity_score
            exclusive_row["pattern_code"] = f"exclusive_{exclusive_row['archetype'].split('_')[-1]}"

            exclusive_results.append(exclusive_row)

    if exclusive_results:
        results_df = pd.DataFrame(exclusive_results)
        # Standardize the results
        results_df = _standardize_results_dataframe(results_df, pattern_type="exclusive", data_type=data_type)

        if verbose:
            print(f"   Found {len(results_df)} exclusive features")
            print("   Distribution by archetype:")
            for arch, count in results_df["archetype"].value_counts().items():
                print(f"      {arch}: {count} features")

        return results_df.sort_values("exclusivity_score", ascending=False)
    else:
        if verbose:
            print("   No exclusive patterns found")
        return pd.DataFrame()


def identify_specialization_patterns(
    adata,
    data_obsm_key: str = "pathway_scores",
    obs_key: str = "archetypes",
    test_method: str = "mannwhitneyu",
    fdr_method: str = "benjamini_hochberg",
    fdr_scope: str = "global",
    min_cells: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Identify specialization patterns: features high in specialized archetypes vs archetype_0.

    Archetype_0 typically represents the centroid/generalist state. This function
    identifies features that distinguish each specialized archetype from this
    undifferentiated baseline.

    Parameters
    ----------
    adata : AnnData
        AnnData object with archetype assignments.
    data_obsm_key : str, default: 'pathway_scores'
        Key for data in adata.obsm. Use None for gene expression from adata.X.
    obs_key : str, default: 'archetypes'
        Key for archetype assignments in adata.obs.
    test_method : str, default: 'mannwhitneyu'
        Statistical test method.
    fdr_method : str, default: 'benjamini_hochberg'
        Multiple testing correction method.
    fdr_scope : str, default: 'global'
        FDR correction scope: 'global', 'per_archetype', or 'none'.
    min_cells : int, default: 10
        Minimum cells per archetype for testing.
    verbose : bool, default: True
        Whether to print progress.

    Returns
    -------
    pd.DataFrame
        Specialization results with standardized columns:

        **Feature identifier**:

        - ``pathway`` : str - Pathway name (if pathway data)
        - ``gene`` : str - Gene name (if gene data)

        **Archetype information**:

        - ``archetype`` : str - Specialized archetype (e.g., 'archetype_1')
        - ``specialized_archetype`` : str - Archetype number extracted from pattern

        **Statistics** (from _standardize_results_dataframe):

        - ``n_archetype_cells`` : int - Cells in specialized archetype
        - ``n_other_cells`` : int - Cells in archetype_0
        - ``mean_archetype`` : float - Mean in specialized archetype
        - ``mean_other`` : float - Mean in archetype_0
        - ``mean_diff`` : float - Effect size (for pathways)
        - ``log_fold_change`` : float - Effect size (for genes or backward compat)
        - ``statistic`` : float - Test statistic
        - ``pvalue`` : float - Raw p-value
        - ``fdr_pvalue`` : float - FDR-corrected p-value
        - ``significant`` : bool - Whether significant
        - ``direction`` : str - 'higher' or 'lower'

        **Pattern metadata**:

        - ``pattern_type`` : str - Always 'specialization'
        - ``pattern_name`` : str - Pattern name
        - ``pattern_code`` : str - Visual pattern code

    Notes
    -----
    Archetype_0 represents the centroid where cells have balanced contributions
    from all archetypes. Features elevated in other archetypes relative to
    archetype_0 represent specialized cellular programs.

    See Also
    --------
    identify_archetype_exclusive_patterns : Exclusive pattern analysis
    identify_tradeoff_patterns : Mutual exclusivity analysis
    peach.tl.specialization_patterns : User-facing wrapper

    Examples
    --------
    >>> specialized = identify_specialization_patterns(adata)
    >>> # Find archetype_2 specialization features
    >>> arch2_spec = specialized[(specialized["archetype"] == "archetype_2") & (specialized["significant"])]
    """
    if verbose:
        print(" Identifying Specialization Patterns...")
        print("   Testing each archetype vs archetype_0 (centroid)")

    # Check if archetype_0 exists
    archetypes = adata.obs[obs_key].unique()
    if "archetype_0" not in archetypes:
        if verbose:
            print("   Warning: archetype_0 not found. Using most central archetype.")
        # Could implement logic to find most central archetype
        return pd.DataFrame()

    # Use pattern testing with specialization patterns only
    results = test_archetype_pattern_associations(
        adata=adata,
        data_obsm_key=data_obsm_key,
        obs_key=obs_key,
        pattern_types=["specialization"],
        test_method=test_method,
        fdr_method=fdr_method,
        fdr_scope=fdr_scope,
        min_cells=min_cells,
        verbose=False,
    )

    if not results.empty:
        # Standardize results
        data_type = "pathway" if data_obsm_key in adata.obsm else "gene"
        results = _standardize_results_dataframe(results, pattern_type="specialization", data_type=data_type)

        # Add clear archetype identification
        results["specialized_archetype"] = results["pattern_name"].str.extract(r"arch(\d+)")
        results["archetype"] = "archetype_" + results["specialized_archetype"].astype(str)

        if verbose:
            print(f"   Found {len(results)} specialization patterns")
            significant = results[results["significant"]]
            print(f"   Significant patterns: {len(significant)}")
            if len(significant) > 0:
                print("   Distribution by archetype:")
                for arch, count in significant["archetype"].value_counts().items():
                    print(f"      {arch}: {count} features")

        return results
    else:
        if verbose:
            print("   No specialization patterns found")
        return pd.DataFrame()


def identify_tradeoff_patterns(
    adata,
    data_obsm_key: str = "pathway_scores",
    obs_key: str = "archetypes",
    tradeoffs: str = "pairs",
    test_method: str = "mannwhitneyu",
    fdr_method: str = "benjamini_hochberg",
    fdr_scope: str = "global",
    min_cells: int = 10,
    min_effect_size: float = 0.1,
    max_pattern_size: int = 2,
    exclude_archetype_0: bool = True,
    specific_patterns: list[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Identify tradeoff patterns: mutually exclusive features between archetypes.

    Finds features showing opposing patterns between archetypes, indicating
    biological tradeoffs or mutually exclusive cellular states.

    Parameters
    ----------
    adata : AnnData
        AnnData object with archetype assignments.
    data_obsm_key : str, default: 'pathway_scores'
        Key for data in adata.obsm. Use None for gene expression.
    obs_key : str, default: 'archetypes'
        Key for archetype assignments in adata.obs.
    tradeoffs : str, default: 'pairs'
        Type of tradeoffs to identify:

        - ``'pairs'`` : Binary tradeoffs between archetype pairs (1v1)
        - ``'patterns'`` : Complex multi-archetype mutual exclusivity
    test_method : str, default: 'mannwhitneyu'
        Statistical test method.
    fdr_method : str, default: 'benjamini_hochberg'
        Multiple testing correction method.
    fdr_scope : str, default: 'global'
        FDR correction scope.
    min_cells : int, default: 10
        Minimum cells per group for testing.
    min_effect_size : float, default: 0.1
        Minimum effect size for pairwise tradeoffs.
    max_pattern_size : int, default: 2
        Maximum archetypes per group (for 'patterns' mode).
    exclude_archetype_0 : bool, default: True
        Exclude archetype_0 from tradeoff analysis.
    specific_patterns : list[str] | None, default: None
        Test only specific patterns (e.g., ['2v3', '1v45']).
    verbose : bool, default: True
        Whether to print progress.

    Returns
    -------
    pd.DataFrame
        Tradeoff results with standardized columns:

        **Feature identifier**:

        - ``pathway`` : str - Pathway name (if pathway data)
        - ``gene`` : str - Gene name (if gene data)

        **Pattern information**:

        - ``pattern_name`` : str - Pattern name
        - ``pattern_code`` : str - Visual pattern code (e.g., '1xxxx_x2xxx')
        - ``pattern_type`` : str - 'tradeoff_pair' or 'tradeoff_pattern'
        - ``high_archetypes`` : str - Comma-separated high archetypes
        - ``low_archetypes`` : str - Comma-separated low archetypes
        - ``pattern_complexity`` : int - Number of archetypes involved

        **Statistics** (from _standardize_results_dataframe):

        - ``n_archetype_cells`` : int - Cells in high group
        - ``n_other_cells`` : int - Cells in low group
        - ``mean_archetype`` : float - Mean in high group
        - ``mean_other`` : float - Mean in low group
        - ``mean_diff`` : float - Effect size (for pathways)
        - ``log_fold_change`` : float - Effect size
        - ``statistic`` : float - Test statistic
        - ``pvalue`` : float - Raw p-value
        - ``fdr_pvalue`` : float - FDR-corrected p-value
        - ``significant`` : bool - Whether significant
        - ``direction`` : str - Always 'higher' after direction correction

    Notes
    -----
    **Direction Standardization**:

    Results are post-processed by ``_standardize_tradeoff_direction()`` to ensure
    'high' groups always have higher expression. When effect_size < 0, groups
    are swapped and pattern codes flipped to maintain consistency.

    **Duplicate Removal**:

    After direction correction, duplicate patterns (A>B and B<A become the same)
    are removed to avoid redundancy.

    See Also
    --------
    identify_archetype_exclusive_patterns : Exclusive pattern analysis
    identify_specialization_patterns : Centroid comparison
    peach.tl.tradeoff_patterns : User-facing wrapper

    Examples
    --------
    >>> # Find pairwise tradeoffs
    >>> pairs = identify_tradeoff_patterns(adata, tradeoffs="pairs")
    >>> # Find complex patterns
    >>> patterns = identify_tradeoff_patterns(adata, tradeoffs="patterns", max_pattern_size=3)
    >>> # Test specific hypothesis
    >>> specific = identify_tradeoff_patterns(adata, specific_patterns=["2v3", "1v4"])
    >>> # Filter by complexity
    >>> simple = patterns[patterns["pattern_complexity"] == 2]
    """
    if verbose:
        mode_desc = "binary archetype pairs" if tradeoffs == "pairs" else "complex patterns"
        print(f" Identifying Tradeoff Patterns ({mode_desc})...")

    # Configure pattern generation based on mode
    if tradeoffs == "pairs":
        # For pairs mode, limit to size 1 vs 1
        actual_max_size = 1
        pattern_types = ["tradeoff"]
    else:  # patterns mode
        actual_max_size = max_pattern_size
        pattern_types = ["tradeoff"]

    # Run pattern analysis with effect size threshold
    results = test_archetype_pattern_associations(
        adata=adata,
        data_obsm_key=data_obsm_key,
        obs_key=obs_key,
        pattern_types=pattern_types,
        test_method=test_method,
        fdr_method=fdr_method,
        fdr_scope=fdr_scope,
        min_logfc=min_effect_size,  # Pass effect size threshold for filtering
        min_cells=min_cells,
        max_pattern_size=actual_max_size,
        exclude_archetype_0=exclude_archetype_0,
        specific_patterns=specific_patterns,
        verbose=False,
    )

    if not results.empty:
        # Filter based on tradeoff type
        if tradeoffs == "pairs":
            # Filter for binary patterns (1v1)
            # Count non-x characters (archetypes) in the pattern code
            results = results[results["pattern_code"].str.count("[^x_]") == 2]
            pattern_type = "tradeoff_pair"
        else:
            # Keep all complex patterns
            pattern_type = "tradeoff_pattern"

        # Standardize results
        data_type = "pathway" if data_obsm_key in adata.obsm else "gene"
        results = _standardize_results_dataframe(results, pattern_type=pattern_type, data_type=data_type)

        # Apply direction standardization to ensure 'high' groups always have higher expression
        results = _standardize_tradeoff_direction(results)

        # Remove duplicate patterns after direction correction
        # After flipping, A>B and B<A become the same pattern
        feature_col = "pathway" if data_type == "pathway" else "gene"
        if "pattern_code" in results.columns and feature_col in results.columns:
            before_dedup = len(results)
            results = results.drop_duplicates(subset=[feature_col, "pattern_code"], keep="first")
            after_dedup = len(results)
            if verbose and before_dedup != after_dedup:
                print(f"   Removed {before_dedup - after_dedup} duplicate patterns after direction correction")

        # Add pattern complexity score (count non-x archetype characters)
        results["pattern_complexity"] = results["pattern_code"].str.count("[^x_]")

        if verbose:
            print(f"   Found {len(results)} tradeoff patterns")
            significant = results[results["significant"]]
            print(f"   Significant patterns: {len(significant)}")

            if tradeoffs == "pairs" and len(significant) > 0:
                # Show archetype pair distribution
                print("   Top archetype pairs with tradeoffs:")
                pair_counts = significant["pattern_code"].value_counts().head(5)
                for pattern, count in pair_counts.items():
                    print(f"      {pattern}: {count} features")
            elif len(significant) > 0:
                # Show complexity distribution
                print("   Pattern complexity distribution:")
                for complexity, count in significant["pattern_complexity"].value_counts().sort_index().items():
                    print(f"      {complexity} archetypes: {count} patterns")

        return results.sort_values(["significant", "pattern_complexity"], ascending=[False, True])
    else:
        if verbose:
            print("   No tradeoff patterns found")
        return pd.DataFrame()


def _standardize_tradeoff_direction(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-hoc correction to ensure 'high' groups always have higher expression.

    When effect_size < 0, it means the designated 'high' group actually has
    lower expression. This function swaps the groups and flips the pattern code
    to maintain consistency.

    Args:
        results_df: DataFrame with tradeoff pattern results

    Returns
    -------
        DataFrame with corrected directions
    """
    if results_df.empty:
        return results_df

    # Make a copy to avoid modifying original
    df = results_df.copy()

    # Identify effect size column
    effect_col = None
    for col in ["mean_diff", "log_fold_change", "primary_effect_size"]:
        if col in df.columns:
            effect_col = col
            break

    if not effect_col or len(df) == 0:
        return df

    # Find patterns with negative effects (reversed direction)
    reversed_mask = df[effect_col] < 0
    n_reversed = reversed_mask.sum()

    if n_reversed > 0:
        # For reversed patterns, swap high/low groups
        if "high_archetypes" in df.columns and "low_archetypes" in df.columns:
            # Store originals
            orig_high = df.loc[reversed_mask, "high_archetypes"].copy()
            orig_low = df.loc[reversed_mask, "low_archetypes"].copy()

            # Swap
            df.loc[reversed_mask, "high_archetypes"] = orig_low
            df.loc[reversed_mask, "low_archetypes"] = orig_high

        # Flip the pattern code
        if "pattern_code" in df.columns:

            def flip_pattern_code(code):
                """Flip a pattern code like 'Axx_xBC' to 'xBC_Axx'"""
                if "_" in str(code):
                    parts = str(code).split("_")
                    if len(parts) == 2:
                        return f"{parts[1]}_{parts[0]}"
                return code

            df.loc[reversed_mask, "pattern_code"] = df.loc[reversed_mask, "pattern_code"].apply(flip_pattern_code)

        # Make effect size positive
        df.loc[reversed_mask, effect_col] = -df.loc[reversed_mask, effect_col]

        # Update other related columns if present
        if "mean_high" in df.columns and "mean_low" in df.columns:
            orig_mean_high = df.loc[reversed_mask, "mean_high"].copy()
            orig_mean_low = df.loc[reversed_mask, "mean_low"].copy()
            df.loc[reversed_mask, "mean_high"] = orig_mean_low
            df.loc[reversed_mask, "mean_low"] = orig_mean_high

        # Update direction column if present
        if "direction" in df.columns:
            # All should now be 'higher' after correction
            df.loc[reversed_mask, "direction"] = "higher"

    return df

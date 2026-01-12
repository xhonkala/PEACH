# """
# Statistical testing functions for archetypal analysis.

# This module provides comprehensive statistical analysis tools for archetype
# characterization and biological interpretation. All functions implement robust
# statistical methods with proper multiple testing correction.

# Main Functions:
# - gene_associations(): Mann-Whitney U tests for gene-archetype associations
# - pathway_associations(): Pathway activity testing with MSigDB integration
# - conditional_associations(): Hypergeometric tests for metadata enrichment
# - pattern_analysis(): Advanced pattern testing for archetype interactions

# Features:
# - Multiple testing correction (global, per-archetype, or none)
# - Effect size calculation and direction reporting
# - AnnData-centric workflows with guaranteed cell alignment
# - Production-ready statistical validation
# """

# from typing import Literal, Dict
# import pandas as pd
# from anndata import AnnData

# # Import existing battle-tested functions
# from .._core.utils.statistical_tests import test_archetype_gene_associations as _test_genes
# from .._core.utils.statistical_tests import test_archetype_pathway_associations as _test_pathways
# from .._core.utils.statistical_tests import test_archetype_conditional_associations as _test_conditional
# from .._core.utils.statistical_tests import analyze_archetypal_patterns_comprehensive as _analyze_patterns
# # Import new focused pattern analysis functions
# from .._core.utils.statistical_tests import identify_archetype_exclusive_patterns as _exclusive_patterns
# from .._core.utils.statistical_tests import identify_specialization_patterns as _specialization_patterns
# from .._core.utils.statistical_tests import identify_tradeoff_patterns as _tradeoff_patterns


# def gene_associations(
#     adata: AnnData,
#     *,  # All arguments keyword-only
#     bin_prop: float = 0.1,
#     obsm_key: str = "archetype_distances",
#     obs_key: str = "archetypes",
#     use_layer: str | None = None,
#     test_method: str = "mannwhitneyu",
#     fdr_method: str = "benjamini_hochberg",
#     fdr_scope: Literal["global", "per_archetype", "none"] = "global",
#     test_direction: str = "two-sided",
#     min_logfc: float = 0.01,
#     min_cells: int = 10,
#     verbose: bool = True,
#     **kwargs
# ) -> pd.DataFrame:
#     """Test gene expression associations with archetypal assignments.

#     Performs Mann-Whitney U tests to identify genes with significantly
#     different expression between each archetype and all other cells.

#     Parameters
#     ----------
#     adata : AnnData
#         Annotated data object with archetypal assignments
#     bin_prop : float, default: 0.1
#         Proportion of cells to bin into each archetype
#     obsm_key : str, default: "archetype_distances"
#         Key in adata.obsm containing archetype distance matrix
#     obs_key : str, default: "archetypes"
#         Column in adata.obs containing archetypal assignments
#     use_layer : str | None, default: None
#         Layer for gene expression. If None, uses adata.X
#     test_method : str, default: "mannwhitneyu"
#         Statistical test method to use
#     fdr_method : str, default: "benjamini_hochberg"
#         FDR correction method. Options: "benjamini_hochberg", "bonferroni"
#     fdr_scope : {"global", "per_archetype", "none"}, default: "global"
#         Scope of FDR correction:
#         - "global": FDR correction across all tests
#         - "per_archetype": FDR correction within each archetype
#         - "none": No FDR correction
#     test_direction : str, default: "two-sided"
#         Direction of statistical test
#     min_logfc : float, default: 0.01
#         Minimum log fold change threshold
#     min_cells : int, default: 10
#         Minimum number of cells required per archetype for testing
#     comparison_group : str, default: 'all'
#         Comparison group for statistical tests:

#         - 'all': Compare archetype cells vs ALL other cells (default)
#         - 'archetypes_only': Compare vs cells assigned to other archetypes only
#           (excludes archetype_0 and no_archetype cells)
#     verbose : bool, default: True
#         Whether to print progress messages

#     Returns
#     -------
#     pd.DataFrame
#         Results with columns:

#         - `gene` : str - Gene symbol/identifier
#         - `archetype` : str - Archetype identifier
#         - `n_archetype_cells` : int - Number of cells in archetype
#         - `n_other_cells` : int - Number of cells in comparison group
#         - `mean_archetype` : float - Mean expression in archetype
#         - `mean_other` : float - Mean expression in other cells
#         - `log_fold_change` : float - Log fold change (archetype vs others)
#         - `statistic` : float - Mann-Whitney U test statistic
#         - `pvalue` : float - Raw p-value from statistical test
#         - `test_direction` : str - Direction of test ('two-sided')
#         - `passes_lfc_threshold` : bool - Whether meets log fold change threshold
#         - `fdr_pvalue` : float - FDR-corrected p-value
#         - `significant` : bool - Whether statistically significant (FDR < 0.05)
#         - `direction` : str - Effect direction ('higher'/'lower')

#     Examples
#     --------
#     >>> results = pc.tl.gene_associations(adata, fdr_scope="global")
#     >>> sig_genes = results[results.significant]
#     >>> print(f"Found {len(sig_genes)} significant associations")
#     """
#     # Input validation
#     if obs_key not in adata.obs.columns:
#         raise ValueError(f"adata.obs['{obs_key}'] not found. Run pc.tl.assign_archetypes() first.")

#     # Delegate to existing function (preserves all battle-tested logic)
#     results = _test_genes(
#         adata=adata,
#         bin_prop=bin_prop,
#         obsm_key=obsm_key,
#         obs_key=obs_key,
#         use_layer=use_layer,
#         test_method=test_method,
#         fdr_method=fdr_method,
#         fdr_scope=fdr_scope,
#         test_direction=test_direction,
#         min_logfc=min_logfc,
#         min_cells=min_cells,
#         verbose=verbose,
#         **kwargs
#     )

#     return results


# def pathway_associations(
#     adata: AnnData,
#     *,
#     pathway_obsm_key: str = "pathway_scores",
#     obsm_key: str = "archetype_distances",
#     obs_key: str = "archetypes",
#     test_method: str = "mannwhitneyu",
#     fdr_method: str = "benjamini_hochberg",
#     fdr_scope: Literal["global", "per_archetype", "none"] = "global",
#     test_direction: str = "two-sided",
#     min_logfc: float = 0.01,
#     min_cells: int = 10,
#     verbose: bool = True,
#     **kwargs
# ) -> pd.DataFrame:
#     """Test pathway activity associations with archetypal assignments.

#     Parameters
#     ----------
#     adata : AnnData
#         Annotated data object with pathway scores and archetypal assignments
#     pathway_obsm_key : str, default: "pathway_scores"
#         Key in adata.obsm containing pathway activity scores
#     obsm_key : str, default: "archetype_distances"
#         Key in adata.obsm containing archetype distance matrix
#     obs_key : str, default: "archetypes"
#         Column in adata.obs containing archetypal assignments
#     test_method : str, default: "mannwhitneyu"
#         Statistical test method to use
#     fdr_method : str, default: "benjamini_hochberg"
#         FDR correction method. Options: "benjamini_hochberg", "bonferroni"
#     fdr_scope : {"global", "per_archetype", "none"}, default: "global"
#         Scope of FDR correction
#     test_direction : str, default: "two-sided"
#         Direction of statistical test
#     min_logfc : float, default: 0.01
#         Minimum log fold change threshold
#     min_cells : int, default: 10
#         Minimum number of cells required per archetype for testing
#     comparison_group : str, default: 'all'
#         Comparison group for statistical tests:

#         - 'all': Compare archetype cells vs ALL other cells (default)
#         - 'archetypes_only': Compare vs cells assigned to other archetypes only
#           (excludes archetype_0 and no_archetype cells)
#     verbose : bool, default: True
#         Whether to print progress messages

#     Returns
#     -------
#     pd.DataFrame
#         Results with columns: pathway, archetype, pvalue, fdr_pvalue,
#         mean_diff, effect_size, direction, significant
#     """
#     # Input validation
#     if obs_key not in adata.obs.columns:
#         raise ValueError(f"adata.obs['{obs_key}'] not found. Run pc.tl.assign_archetypes() first.")
#     if pathway_obsm_key not in adata.obsm:
#         raise ValueError(f"adata.obsm['{pathway_obsm_key}'] not found. Run pc.pp.compute_pathway_scores() first.")

#     # Delegate to existing function
#     results = _test_pathways(
#         adata=adata,
#         pathway_obsm_key=pathway_obsm_key,
#         obsm_key=obsm_key,
#         obs_key=obs_key,
#         test_method=test_method,
#         fdr_method=fdr_method,
#         fdr_scope=fdr_scope,
#         test_direction=test_direction,
#         min_logfc=min_logfc,
#         min_cells=min_cells,
#         verbose=verbose,
#         **kwargs
#     )

#     return results


# def pattern_analysis(
#     adata: AnnData,
#     *,
#     data_obsm_key: str = "pathway_scores",
#     obs_key: str = "archetypes",
#     include_individual_tests: bool = True,
#     include_pattern_tests: bool = True,
#     include_exclusivity_analysis: bool = True,
#     verbose: bool = True,
#     **kwargs
# ) -> Dict[str, pd.DataFrame]:
#     """Comprehensive archetypal pattern analysis.

#     Performs systematic analysis of archetypal patterns including
#     gene expression and pathway activity associations.

#     Parameters
#     ----------
#     adata : AnnData
#         Annotated data object with archetypal assignments
#     data_obsm_key : str, default: "pathway_scores"
#         Key in adata.obsm containing scores/data for pattern analysis
#     obs_key : str, default: "archetypes"
#         Column in adata.obs containing archetypal assignments
#     include_individual_tests : bool, default: True
#         Whether to include individual archetype characterization tests
#     include_pattern_tests : bool, default: True
#         Whether to include pattern-based tests (specialists, trade-offs)
#     include_exclusivity_analysis : bool, default: True
#         Whether to include mutual exclusivity analysis
#     verbose : bool, default: True
#         Whether to print analysis progress

#     Returns
#     -------
#     dict
#         Dictionary containing pattern analysis results:
#         - 'individual': Individual archetype characterization results
#         - 'patterns': Pattern-based test results
#         - 'exclusivity': Mutual exclusivity analysis results
#     """
#     # Input validation
#     if obs_key not in adata.obs.columns:
#         raise ValueError(f"adata.obs['{obs_key}'] not found. Run pc.tl.assign_archetypes() first.")

#     # Delegate to existing comprehensive analysis function
#     results = _analyze_patterns(
#         adata=adata,
#         data_obsm_key=data_obsm_key,
#         obs_key=obs_key,
#         include_individual_tests=include_individual_tests,
#         include_pattern_tests=include_pattern_tests,
#         include_exclusivity_analysis=include_exclusivity_analysis,
#         verbose=verbose,
#         **kwargs
#     )

#     return results

# def conditional_associations(
#     adata: AnnData,
#     *,
#     obs_column: str,
#     archetype_assignments = None,  # For backward compatibility - will be ignored
#     obs_key: str = "archetypes",
#     test_method: str = "hypergeometric",
#     fdr_method: str = "benjamini_hochberg",
#     min_cells: int = 5,
#     verbose: bool = True,
#     **kwargs
# ) -> pd.DataFrame:
#     """Test associations between archetypes and categorical metadata.

#     Performs hypergeometric tests to identify significant enrichment of
#     archetypes within different categorical conditions (e.g., samples,
#     treatments, cell types).

#     Parameters
#     ----------
#     adata : AnnData
#         Annotated data object with archetypal assignments and metadata
#     obs_column : str
#         Column name in adata.obs containing categorical variable to test
#     archetype_assignments : None, optional
#         For backward compatibility - will be ignored
#     obs_key : str, default: "archetypes"
#         Column in adata.obs containing archetypal assignments
#     test_method : str, default: "hypergeometric"
#         Statistical test method (currently only hypergeometric supported)
#     fdr_method : str, default: "benjamini_hochberg"
#         FDR correction method
#     min_cells : int, default: 5
#         Minimum number of cells required per archetype for testing
#     verbose : bool, default: True
#         Whether to print progress messages

#     Returns
#     -------
#     pd.DataFrame
#         Results with columns: archetype, condition, observed, expected,
#         total_archetype, total_condition, odds_ratio, ci_lower, ci_upper,
#         pvalue, fdr_pvalue, significant

#     Examples
#     --------
#     >>> results = pc.tl.conditional_associations(adata, obs_column="sample")
#     >>> enriched = results[results.significant]
#     >>> print(f"Found {len(enriched)} significant enrichments")
#     """
#     # Input validation
#     if obs_key not in adata.obs.columns:
#         raise ValueError(f"adata.obs['{obs_key}'] not found. Run pc.tl.assign_archetypes() first.")
#     if obs_column not in adata.obs.columns:
#         raise ValueError(f"adata.obs['{obs_column}'] not found.")

#     # Delegate to existing function
#     results = _test_conditional(
#         adata=adata,
#         obs_column=obs_column,
#         archetype_assignments=archetype_assignments,
#         obs_key=obs_key,
#         test_method=test_method,
#         fdr_method=fdr_method,
#         min_cells=min_cells,
#         verbose=verbose,
#         **kwargs
#     )

#     return results

# def archetype_exclusive_patterns(
#     adata: AnnData,
#     *,
#     data_obsm_key: str = "pathway_scores",
#     obs_key: str = "archetypes",
#     test_method: str = "mannwhitneyu",
#     fdr_method: str = "benjamini_hochberg",
#     fdr_scope: Literal["global", "per_archetype", "none"] = "global",
#     min_effect_size: float = 0.05,
#     min_cells: int = 10,
#     use_pairwise: bool = True,
#     verbose: bool = True,
#     **kwargs
# ) -> pd.DataFrame:
#     """Identify features exclusively high in single archetypes.

#     Finds genes or pathways that are specifically elevated in only one
#     archetype compared to all others, using either pairwise comparisons
#     or 1-vs-all filtering.

#     Parameters
#     ----------
#     adata : AnnData
#         Annotated data object with archetypal assignments
#     data_obsm_key : str, default: "pathway_scores"
#         Key in adata.obsm containing scores/data for analysis
#     obs_key : str, default: "archetypes"
#         Column in adata.obs containing archetypal assignments
#     test_method : str, default: "mannwhitneyu"
#         Statistical test method to use
#     fdr_method : str, default: "benjamini_hochberg"
#         FDR correction method
#     fdr_scope : {"global", "per_archetype", "none"}, default: "global"
#         Scope of FDR correction
#     min_effect_size : float, default: 0.05
#         Minimum effect size for exclusivity (mean_diff for pathways, log_fc for genes)
#     min_cells : int, default: 10
#         Minimum cells required per archetype
#     use_pairwise : bool, default: True
#         If True, use rigorous pairwise comparisons; if False, use 1-vs-all filtering
#     verbose : bool, default: True
#         Whether to print progress

#     Returns
#     -------
#     pd.DataFrame
#         Standardized results with exclusive features

#     Examples
#     --------
#     >>> # Use pairwise method (more stringent)
#     >>> exclusive = pc.tl.archetype_exclusive_patterns(adata, use_pairwise=True)
#     >>>
#     >>> # Use 1-vs-all method (more permissive)
#     >>> exclusive = pc.tl.archetype_exclusive_patterns(adata, use_pairwise=False)
#     >>>
#     >>> arch3_exclusive = exclusive[exclusive.archetype == "Archetype_3"]
#     >>> print(f"Archetype 3 exclusive features: {len(arch3_exclusive)}")
#     """
#     # Input validation
#     if obs_key not in adata.obs.columns:
#         raise ValueError(f"adata.obs['{obs_key}'] not found. Run pc.tl.assign_archetypes() first.")

#     # Delegate to core function with new parameters
#     results = _exclusive_patterns(
#         adata=adata,
#         data_obsm_key=data_obsm_key,
#         obs_key=obs_key,
#         test_method=test_method,
#         fdr_method=fdr_method,
#         fdr_scope=fdr_scope,
#         min_cells=min_cells,
#         min_effect_size=min_effect_size,
#         use_pairwise=use_pairwise,
#         verbose=verbose
#     )

#     return results

# def specialization_patterns(
#     adata: AnnData,
#     *,
#     data_obsm_key: str = "pathway_scores",
#     obs_key: str = "archetypes",
#     test_method: str = "mannwhitneyu",
#     fdr_method: str = "benjamini_hochberg",
#     fdr_scope: Literal["global", "per_archetype", "none"] = "global",
#     test_direction: str = "two-sided",
#     min_logfc: float = 0.01,
#     min_cells: int = 10,
#     verbose: bool = True,
#     **kwargs
# ) -> pd.DataFrame:
#     """Identify specialization features relative to centroid archetype.

#     Compares each archetype to archetype_0 (centroid/generalist) to find
#     features that represent specialized states or differentiation away
#     from the central cellular state.

#     Parameters
#     ----------
#     adata : AnnData
#         Annotated data object with archetypal assignments
#     data_obsm_key : str, default: "pathway_scores"
#         Key in adata.obsm containing scores/data for analysis
#     obs_key : str, default: "archetypes"
#         Column in adata.obs containing archetypal assignments
#     test_method : str, default: "mannwhitneyu"
#         Statistical test method to use
#     fdr_method : str, default: "benjamini_hochberg"
#         FDR correction method
#     fdr_scope : {"global", "per_archetype", "none"}, default: "global"
#         Scope of FDR correction
#     test_direction : str, default: "two-sided"
#         Direction of statistical test
#     min_logfc : float, default: 0.01
#         Minimum log fold change threshold
#     min_cells : int, default: 10
#         Minimum cells required per archetype
#     verbose : bool, default: True
#         Whether to print progress

#     Returns
#     -------
#     pd.DataFrame
#         Standardized results showing specialization from centroid

#     Examples
#     --------
#     >>> spec = pc.tl.specialization_patterns(adata)
#     >>> arch4_spec = spec[(spec.archetype == "Archetype_4") & spec.significant]
#     >>> print(f"Archetype 4 specializations: {len(arch4_spec)}")
#     """
#     # Input validation
#     if obs_key not in adata.obs.columns:
#         raise ValueError(f"adata.obs['{obs_key}'] not found. Run pc.tl.assign_archetypes() first.")

#     # Delegate to core function (only pass parameters it accepts)
#     results = _specialization_patterns(
#         adata=adata,
#         data_obsm_key=data_obsm_key,
#         obs_key=obs_key,
#         test_method=test_method,
#         fdr_method=fdr_method,
#         fdr_scope=fdr_scope,
#         min_cells=min_cells,
#         verbose=verbose
#     )

#     return results

# def tradeoff_patterns(
#     adata: AnnData,
#     *,
#     data_obsm_key: str = "pathway_scores",
#     obs_key: str = "archetypes",
#     tradeoffs: Literal["pairs", "patterns"] = "pairs",
#     test_method: str = "mannwhitneyu",
#     fdr_method: str = "benjamini_hochberg",
#     fdr_scope: Literal["global", "per_archetype", "none"] = "global",
#     test_direction: str = "two-sided",
#     min_logfc: float = 0.01,
#     min_cells: int = 10,
#     verbose: bool = True,
#     **kwargs
# ) -> pd.DataFrame:
#     """Identify mutual exclusivity and tradeoff patterns.

#     Finds features showing opposing patterns between archetypes, indicating
#     biological tradeoffs or mutually exclusive cellular states.

#     Parameters
#     ----------
#     adata : AnnData
#         Annotated data object with archetypal assignments
#     data_obsm_key : str, default: "pathway_scores"
#         Key in adata.obsm containing scores/data for analysis
#     obs_key : str, default: "archetypes"
#         Column in adata.obs containing archetypal assignments
#     tradeoffs : {"pairs", "patterns"}, default: "pairs"
#         Type of tradeoff analysis:
#         - "pairs": Simple pairwise mutual exclusivity (A high, B low)
#         - "patterns": Complex multi-archetype patterns (AB high, CD low)
#     test_method : str, default: "mannwhitneyu"
#         Statistical test method to use
#     fdr_method : str, default: "benjamini_hochberg"
#         FDR correction method
#     fdr_scope : {"global", "per_archetype", "none"}, default: "global"
#         Scope of FDR correction
#     test_direction : str, default: "two-sided"
#         Direction of statistical test
#     min_logfc : float, default: 0.01
#         Minimum log fold change threshold
#     min_cells : int, default: 10
#         Minimum cells required per archetype
#     verbose : bool, default: True
#         Whether to print progress
#     **kwargs
#         Additional parameters passed to core function:
#         - max_pattern_size : int, default: 2
#             Maximum archetypes per group for complex patterns
#         - exclude_archetype_0 : bool, default: True
#             Whether to exclude archetype_0 from tradeoff patterns
#         - specific_patterns : List[str], optional
#             Test only specific patterns (e.g., ['2v3', '1v45'])

#     Returns
#     -------
#     pd.DataFrame
#         Standardized results with tradeoff patterns identified

#     Examples
#     --------
#     >>> # Find simple pairwise tradeoffs
#     >>> pairs = pc.tl.tradeoff_patterns(adata, tradeoffs="pairs")
#     >>> arch1_2_tradeoffs = pairs[pairs.pattern.str.contains("1.*2|2.*1")]

#     >>> # Find complex multi-archetype patterns
#     >>> patterns = pc.tl.tradeoff_patterns(adata, tradeoffs="patterns")
#     >>> complex = patterns[patterns.pattern.str.count("x") >= 2]
#     """
#     # Input validation
#     if obs_key not in adata.obs.columns:
#         raise ValueError(f"adata.obs['{obs_key}'] not found. Run pc.tl.assign_archetypes() first.")

#     # Extract only the parameters the core function accepts from kwargs
#     core_params = {}
#     allowed_kwargs = ['max_pattern_size', 'exclude_archetype_0', 'specific_patterns']
#     for key in allowed_kwargs:
#         if key in kwargs:
#             core_params[key] = kwargs[key]

#     # Delegate to core function (only pass parameters it accepts)
#     results = _tradeoff_patterns(
#         adata=adata,
#         data_obsm_key=data_obsm_key,
#         obs_key=obs_key,
#         tradeoffs=tradeoffs,
#         test_method=test_method,
#         fdr_method=fdr_method,
#         fdr_scope=fdr_scope,
#         min_cells=min_cells,
#         verbose=verbose,
#         **core_params  # Only pass allowed parameters
#     )

#     return results

"""
Statistical Testing for Archetypal Analysis
============================================

User-facing API for statistical analysis of archetype-feature associations.

This module provides comprehensive statistical testing tools for characterizing
archetypes by their gene expression, pathway activity, and metadata associations.
All functions implement robust statistical methods with proper multiple testing
correction.

Main Functions
--------------
gene_associations
    Mann-Whitney U tests for gene-archetype associations
pathway_associations
    Pathway activity testing for archetype characterization
conditional_associations
    Hypergeometric tests for metadata enrichment
pattern_analysis
    Comprehensive archetypal pattern analysis
archetype_exclusive_patterns
    Identify features exclusively high in single archetypes
specialization_patterns
    Compare archetypes to centroid (archetype_0)
tradeoff_patterns
    Identify mutual exclusivity between archetypes

Type Definitions
----------------
See ``peach._core.types`` for Pydantic models:

- ``GeneAssociationResult`` : gene_associations() row structure
- ``PathwayAssociationResult`` : pathway_associations() row structure
- ``ConditionalAssociationResult`` : conditional_associations() row structure
- ``PatternAssociationResult`` : Pattern test result structure
- ``ExclusivePatternResult`` : Exclusive pattern result structure
- ``ComprehensivePatternResults`` : pattern_analysis() return structure

Examples
--------
>>> import peach as pc
>>> # Gene associations
>>> gene_results = pc.tl.gene_associations(adata)
>>> sig_genes = gene_results[gene_results.significant]
>>> # Pathway associations
>>> pathway_results = pc.tl.pathway_associations(adata)
>>> # Comprehensive pattern analysis
>>> patterns = pc.tl.pattern_analysis(adata)
>>> specialists = patterns["patterns"][patterns["patterns"]["pattern_type"] == "specialization"]

See Also
--------
peach._core.utils.statistical_tests : Core implementation
peach._core.types : Type definitions
"""

from typing import Literal

import pandas as pd
from anndata import AnnData

from .._core.utils.statistical_tests import (
    analyze_archetypal_patterns_comprehensive as _analyze_patterns,
)
from .._core.utils.statistical_tests import (
    identify_archetype_exclusive_patterns as _exclusive_patterns,
)
from .._core.utils.statistical_tests import (
    identify_specialization_patterns as _specialization_patterns,
)
from .._core.utils.statistical_tests import (
    identify_tradeoff_patterns as _tradeoff_patterns,
)
from .._core.utils.statistical_tests import (
    test_archetype_conditional_associations as _test_conditional,
)
from .._core.utils.statistical_tests import (
    test_archetype_gene_associations as _test_genes,
)
from .._core.utils.statistical_tests import (
    test_archetype_pathway_associations as _test_pathways,
)


def gene_associations(
    adata: AnnData,
    *,
    bin_prop: float = 0.1,
    obsm_key: str = "archetype_distances",
    obs_key: str = "archetypes",
    use_layer: str | None = None,
    test_method: str = "mannwhitneyu",
    fdr_method: str = "benjamini_hochberg",
    fdr_scope: Literal["global", "per_archetype", "none"] = "global",
    test_direction: str = "two-sided",
    min_logfc: float = 0.01,
    min_cells: int = 10,
    comparison_group: str = "all",
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Test gene expression associations with archetypal assignments.

    Performs Mann-Whitney U tests to identify genes with significantly
    different expression between each archetype and all other cells
    (1-vs-all testing paradigm).

    Parameters
    ----------
    adata : AnnData
        Annotated data object with:

        - ``obsm[obsm_key]`` : Archetype distance matrix [n_cells, n_archetypes]
        - ``obs[obs_key]`` : Archetype assignments (from bin_cells_by_archetype)
        - ``X`` or ``layers[use_layer]`` : Gene expression data
    bin_prop : float, default: 0.1
        Proportion of cells closest to each archetype to use for binning.
    obsm_key : str, default: "archetype_distances"
        Key in adata.obsm containing archetype distance matrix.
    obs_key : str, default: "archetypes"
        Column in adata.obs containing archetypal assignments.
    use_layer : str | None, default: None
        Layer for gene expression. If None, uses adata.X.
        Auto-selects 'logcounts' or 'log1p' if available.
    test_method : str, default: "mannwhitneyu"
        Statistical test method. Currently supports 'mannwhitneyu'.
    fdr_method : str, default: "benjamini_hochberg"
        FDR correction method: 'benjamini_hochberg' or 'bonferroni'.
    fdr_scope : {'global', 'per_archetype', 'none'}, default: 'global'
        Scope of FDR correction:

        - ``'global'`` : Correct across all tests (most stringent)
        - ``'per_archetype'`` : Correct within each archetype
        - ``'none'`` : No FDR correction (raw p-values)
    test_direction : str, default: "two-sided"
        Direction of statistical test: 'two-sided', 'greater', or 'less'.
    min_logfc : float, default: 0.01
        Minimum absolute log fold change threshold for filtering.
    min_cells : int, default: 10
        Minimum cells required per archetype for testing.
    comparison_group : str, default: 'all'
        Comparison group for statistical tests:

        - ``'all'`` : Compare archetype cells vs ALL other cells
        - ``'archetypes_only'`` : Compare vs cells in other archetypes only
          (excludes archetype_0 and unassigned cells)
    verbose : bool, default: True
        Whether to print progress messages.

    Returns
    -------
    pd.DataFrame
        Results with columns:

        - ``gene`` : str - Gene symbol/identifier
        - ``archetype`` : str - Archetype identifier
        - ``n_archetype_cells`` : int - Cells in archetype
        - ``n_other_cells`` : int - Cells in comparison group
        - ``mean_archetype`` : float - Mean expression in archetype
        - ``mean_other`` : float - Mean expression in others
        - ``log_fold_change`` : float - Log fold change
        - ``statistic`` : float - Mann-Whitney U statistic
        - ``pvalue`` : float - Raw p-value
        - ``fdr_pvalue`` : float - FDR-corrected p-value
        - ``significant`` : bool - Whether FDR < 0.05
        - ``direction`` : str - 'higher' or 'lower' in archetype

    Raises
    ------
    ValueError
        If required keys not found in adata.

    Examples
    --------
    >>> # Basic usage
    >>> results = pc.tl.gene_associations(adata)
    >>> sig_genes = results[results.significant]
    >>> # Per-archetype FDR correction (less stringent)
    >>> results = pc.tl.gene_associations(adata, fdr_scope="per_archetype")
    >>> # Top markers per archetype
    >>> for arch in results["archetype"].unique():
    ...     arch_genes = results[
    ...         (results["archetype"] == arch) & (results["significant"]) & (results["direction"] == "higher")
    ...     ].nlargest(10, "log_fold_change")
    ...     print(f"{arch}: {arch_genes['gene'].tolist()}")

    See Also
    --------
    peach.tl.pathway_associations : Pathway-level testing
    peach.tl.pattern_analysis : Comprehensive pattern analysis
    peach._core.types.GeneAssociationResult : Result row structure
    """
    if obs_key not in adata.obs.columns:
        raise ValueError(f"adata.obs['{obs_key}'] not found. Run pc.tl.assign_archetypes() first.")

    return _test_genes(
        adata=adata,
        bin_prop=bin_prop,
        obsm_key=obsm_key,
        obs_key=obs_key,
        use_layer=use_layer,
        test_method=test_method,
        fdr_method=fdr_method,
        fdr_scope=fdr_scope,
        test_direction=test_direction,
        min_logfc=min_logfc,
        min_cells=min_cells,
        comparison_group=comparison_group,
        verbose=verbose,
        **kwargs,
    )


def pathway_associations(
    adata: AnnData,
    *,
    pathway_obsm_key: str = "pathway_scores",
    obsm_key: str = "archetype_distances",
    obs_key: str = "archetypes",
    test_method: str = "mannwhitneyu",
    fdr_method: str = "benjamini_hochberg",
    fdr_scope: Literal["global", "per_archetype", "none"] = "global",
    test_direction: str = "two-sided",
    min_logfc: float = 0.01,
    min_cells: int = 10,
    comparison_group: str = "all",
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Test pathway activity associations with archetypal assignments.

    Performs Mann-Whitney U tests to identify pathways with significantly
    different activity between each archetype and all other cells.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with:

        - ``obsm[pathway_obsm_key]`` : Pathway scores [n_cells, n_pathways]
        - ``obsm[obsm_key]`` : Archetype distance matrix
        - ``obs[obs_key]`` : Archetype assignments
        - ``uns[pathway_obsm_key + '_pathways']`` : Pathway names (optional)
    pathway_obsm_key : str, default: "pathway_scores"
        Key in adata.obsm containing pathway activity scores.
    obsm_key : str, default: "archetype_distances"
        Key in adata.obsm containing archetype distance matrix.
    obs_key : str, default: "archetypes"
        Column in adata.obs containing archetypal assignments.
    test_method : str, default: "mannwhitneyu"
        Statistical test method.
    fdr_method : str, default: "benjamini_hochberg"
        FDR correction method.
    fdr_scope : {'global', 'per_archetype', 'none'}, default: 'global'
        Scope of FDR correction.
    test_direction : str, default: "two-sided"
        Direction of statistical test.
    min_logfc : float, default: 0.01
        Minimum effect size threshold (mean_diff for pathways).
    min_cells : int, default: 10
        Minimum cells required per archetype.
    comparison_group : str, default: 'all'
        Comparison group: 'all' or 'archetypes_only'.
    verbose : bool, default: True
        Whether to print progress.

    Returns
    -------
    pd.DataFrame
        Results with columns:

        - ``pathway`` : str - Pathway name
        - ``archetype`` : str - Archetype identifier
        - ``n_archetype_cells`` : int - Cells in archetype
        - ``n_other_cells`` : int - Cells in comparison
        - ``mean_archetype`` : float - Mean score in archetype
        - ``mean_other`` : float - Mean score in others
        - ``mean_diff`` : float - Mean difference (primary effect size)
        - ``log_fold_change`` : float - Alias for mean_diff
        - ``statistic`` : float - Test statistic
        - ``pvalue`` : float - Raw p-value
        - ``fdr_pvalue`` : float - FDR-corrected p-value
        - ``significant`` : bool - Whether significant
        - ``direction`` : str - 'higher' or 'lower'

    Notes
    -----
    Pathway scores (from AUCell, pySCENIC, etc.) represent activity levels,
    not expression counts. Mean difference is more interpretable than
    log fold change for these scores.

    Examples
    --------
    >>> # Basic usage
    >>> results = pc.tl.pathway_associations(adata)
    >>> # Filter for specific pathway categories
    >>> metabolism = results[results["pathway"].str.contains("METABOLISM", case=False)]
    >>> # Top pathways per archetype
    >>> for arch in results["archetype"].unique():
    ...     top = results[(results["archetype"] == arch) & (results["significant"])].nlargest(5, "mean_diff")
    ...     print(f"{arch}: {top['pathway'].tolist()}")

    See Also
    --------
    peach.tl.gene_associations : Gene-level testing
    peach._core.types.PathwayAssociationResult : Result row structure
    """
    if obs_key not in adata.obs.columns:
        raise ValueError(f"adata.obs['{obs_key}'] not found. Run pc.tl.assign_archetypes() first.")
    if pathway_obsm_key not in adata.obsm:
        raise ValueError(f"adata.obsm['{pathway_obsm_key}'] not found. Run pc.pp.compute_pathway_scores() first.")

    return _test_pathways(
        adata=adata,
        pathway_obsm_key=pathway_obsm_key,
        obsm_key=obsm_key,
        obs_key=obs_key,
        test_method=test_method,
        fdr_method=fdr_method,
        fdr_scope=fdr_scope,
        test_direction=test_direction,
        min_logfc=min_logfc,
        min_cells=min_cells,
        comparison_group=comparison_group,
        verbose=verbose,
        **kwargs,
    )


def conditional_associations(
    adata: AnnData,
    *,
    obs_column: str,
    archetype_assignments=None,
    obs_key: str = "archetypes",
    test_method: str = "hypergeometric",
    fdr_method: str = "benjamini_hochberg",
    min_cells: int = 5,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Test associations between archetypes and categorical metadata.

    Performs hypergeometric tests to identify significant enrichment of
    archetypes within different categorical conditions (samples, treatments,
    cell types, etc.).

    Parameters
    ----------
    adata : AnnData
        Annotated data object with:

        - ``obs[obs_key]`` : Archetype assignments
        - ``obs[obs_column]`` : Categorical variable to test
    obs_column : str
        Column name in adata.obs containing categorical variable.
    archetype_assignments : None, optional
        Deprecated. Archetype assignments now read from adata.obs[obs_key].
    obs_key : str, default: "archetypes"
        Column in adata.obs containing archetypal assignments.
    test_method : str, default: "hypergeometric"
        Statistical test method (currently only 'hypergeometric').
    fdr_method : str, default: "benjamini_hochberg"
        FDR correction method.
    min_cells : int, default: 5
        Minimum cells required per archetype-condition combination.
    verbose : bool, default: True
        Whether to print progress.

    Returns
    -------
    pd.DataFrame
        Results with columns:

        - ``archetype`` : str - Archetype identifier
        - ``condition`` : str - Condition value from obs_column
        - ``observed`` : int - Observed count in overlap
        - ``expected`` : float - Expected count under null
        - ``total_archetype`` : int - Total cells in archetype
        - ``total_condition`` : int - Total cells in condition
        - ``odds_ratio`` : float - Enrichment measure (>1 = enriched)
        - ``ci_lower`` : float - Lower 95% CI for odds ratio
        - ``ci_upper`` : float - Upper 95% CI for odds ratio
        - ``pvalue`` : float - Hypergeometric p-value
        - ``fdr_pvalue`` : float - FDR-corrected p-value
        - ``significant`` : bool - Whether significant

    Examples
    --------
    >>> # Test sample associations
    >>> results = pc.tl.conditional_associations(adata, obs_column="sample")
    >>> # Find enriched archetypes per condition
    >>> enriched = results[(results["significant"]) & (results["odds_ratio"] > 2)]
    >>> # Test treatment effects
    >>> treatment_results = pc.tl.conditional_associations(adata, obs_column="treatment")

    See Also
    --------
    peach._core.types.ConditionalAssociationResult : Result row structure
    """
    if obs_key not in adata.obs.columns:
        raise ValueError(f"adata.obs['{obs_key}'] not found. Run pc.tl.assign_archetypes() first.")
    if obs_column not in adata.obs.columns:
        raise ValueError(f"adata.obs['{obs_column}'] not found.")

    return _test_conditional(
        adata=adata,
        obs_column=obs_column,
        archetype_assignments=archetype_assignments,
        obs_key=obs_key,
        test_method=test_method,
        fdr_method=fdr_method,
        min_cells=min_cells,
        verbose=verbose,
        **kwargs,
    )


def pattern_analysis(
    adata: AnnData,
    *,
    data_obsm_key: str = "pathway_scores",
    obs_key: str = "archetypes",
    include_individual_tests: bool = True,
    include_pattern_tests: bool = True,
    include_exclusivity_analysis: bool = True,
    verbose: bool = True,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Comprehensive archetypal pattern analysis.

    Performs systematic analysis combining three complementary approaches:

    1. **Individual tests**: Standard 1-vs-all archetype characterization
    2. **Pattern tests**: Systematic archetype combination testing
       (specialists, binary tradeoffs, complex patterns)
    3. **Exclusivity analysis**: Features with opposing patterns

    Parameters
    ----------
    adata : AnnData
        Annotated data object with archetypal assignments and scores.
    data_obsm_key : str, default: "pathway_scores"
        Key in adata.obsm containing scores for pattern analysis.
    obs_key : str, default: "archetypes"
        Column in adata.obs containing archetypal assignments.
    include_individual_tests : bool, default: True
        Run individual archetype 1-vs-all tests.
    include_pattern_tests : bool, default: True
        Run systematic pattern tests (specialists, tradeoffs).
    include_exclusivity_analysis : bool, default: True
        Analyze mutual exclusivity patterns.
    verbose : bool, default: True
        Print analysis progress.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with keys:

        - ``'individual'`` : Individual archetype results
        - ``'patterns'`` : Pattern-based test results
        - ``'exclusivity'`` : Mutual exclusivity results

    Notes
    -----
    **Pattern Types in 'patterns' DataFrame**:

    - ``specialization`` : Archetype vs archetype_0 (centroid)
    - ``tradeoff`` : Multi-archetype high vs low groups

    **Pattern Code Format**: "12xxx_xx345"

    - Position = archetype number (0, 1, 2...)
    - Numbers = high archetypes
    - 'x' = low archetypes
    - Underscore separates high from low group

    Examples
    --------
    >>> # Run comprehensive analysis
    >>> results = pc.tl.pattern_analysis(adata)
    >>> # Access individual results
    >>> individual = results["individual"]
    >>> # Find specialists (exclusive to one archetype)
    >>> patterns = results["patterns"]
    >>> specialists = patterns[patterns["pattern_type"] == "specialization"]
    >>> # Find mutual exclusivity patterns
    >>> if not results["exclusivity"].empty:
    ...     exclusive = results["exclusivity"]
    ...     top_tradeoffs = exclusive.nlargest(10, "effect_range")

    See Also
    --------
    peach.tl.archetype_exclusive_patterns : Focused exclusive pattern analysis
    peach.tl.specialization_patterns : Centroid comparison analysis
    peach.tl.tradeoff_patterns : Mutual exclusivity analysis
    peach._core.types.ComprehensivePatternResults : Return type structure
    """
    if obs_key not in adata.obs.columns:
        raise ValueError(f"adata.obs['{obs_key}'] not found. Run pc.tl.assign_archetypes() first.")

    return _analyze_patterns(
        adata=adata,
        data_obsm_key=data_obsm_key,
        obs_key=obs_key,
        include_individual_tests=include_individual_tests,
        include_pattern_tests=include_pattern_tests,
        include_exclusivity_analysis=include_exclusivity_analysis,
        verbose=verbose,
        **kwargs,
    )


def archetype_exclusive_patterns(
    adata: AnnData,
    *,
    data_obsm_key: str = "pathway_scores",
    obs_key: str = "archetypes",
    test_method: str = "mannwhitneyu",
    fdr_method: str = "benjamini_hochberg",
    fdr_scope: Literal["global", "per_archetype", "none"] = "global",
    min_effect_size: float = 0.05,
    min_cells: int = 10,
    use_pairwise: bool = True,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Identify features exclusively high in single archetypes.

    Finds genes or pathways specifically elevated in only one archetype
    compared to all others. Supports two methods:

    1. **Pairwise** (default): Tests each archetype vs every other
       archetype individually. Feature is exclusive if significantly
       higher vs ALL others. More stringent.

    2. **1-vs-all filtering**: Tests each archetype vs all other cells.
       Feature is exclusive if significant in only ONE archetype's test.
       More permissive, higher statistical power.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with archetypal assignments.
    data_obsm_key : str, default: "pathway_scores"
        Key in adata.obsm for scores. Use None for gene expression.
    obs_key : str, default: "archetypes"
        Column in adata.obs with archetypal assignments.
    test_method : str, default: "mannwhitneyu"
        Statistical test method.
    fdr_method : str, default: "benjamini_hochberg"
        FDR correction method.
    fdr_scope : {'global', 'per_archetype', 'none'}, default: 'global'
        Scope of FDR correction.
    min_effect_size : float, default: 0.05
        Minimum effect size (mean_diff for pathways, log_fc for genes).
    min_cells : int, default: 10
        Minimum cells per archetype.
    use_pairwise : bool, default: True
        If True, use rigorous pairwise comparisons.
        If False, use 1-vs-all filtering.
    verbose : bool, default: True
        Print progress.

    Returns
    -------
    pd.DataFrame
        Results with columns:

        - ``pathway``/``gene`` : Feature identifier
        - ``archetype`` : Exclusive archetype
        - ``mean_archetype`` : Mean in exclusive archetype
        - ``mean_other`` : Mean in other archetypes
        - ``mean_diff``/``log_fold_change`` : Effect size
        - ``exclusivity_score`` : Ratio vs max other archetype
        - ``pvalue``, ``fdr_pvalue``, ``significant``
        - ``pattern_type`` : 'exclusive' or 'exclusive_pairwise'

    Examples
    --------
    >>> # Pairwise method (more stringent)
    >>> exclusive = pc.tl.archetype_exclusive_patterns(adata)
    >>> # 1-vs-all method (more permissive)
    >>> exclusive = pc.tl.archetype_exclusive_patterns(adata, use_pairwise=False)
    >>> # Find markers for specific archetype
    >>> arch3_markers = exclusive[exclusive["archetype"] == "archetype_3"]
    >>> top_markers = arch3_markers.nlargest(10, "exclusivity_score")

    See Also
    --------
    peach.tl.pattern_analysis : Comprehensive pattern analysis
    peach._core.types.ExclusivePatternResult : Result row structure
    """
    if obs_key not in adata.obs.columns:
        raise ValueError(f"adata.obs['{obs_key}'] not found. Run pc.tl.assign_archetypes() first.")

    return _exclusive_patterns(
        adata=adata,
        data_obsm_key=data_obsm_key,
        obs_key=obs_key,
        test_method=test_method,
        fdr_method=fdr_method,
        fdr_scope=fdr_scope,
        min_cells=min_cells,
        min_effect_size=min_effect_size,
        use_pairwise=use_pairwise,
        verbose=verbose,
    )


def specialization_patterns(
    adata: AnnData,
    *,
    data_obsm_key: str = "pathway_scores",
    obs_key: str = "archetypes",
    test_method: str = "mannwhitneyu",
    fdr_method: str = "benjamini_hochberg",
    fdr_scope: Literal["global", "per_archetype", "none"] = "global",
    min_cells: int = 10,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Identify specialization features relative to centroid archetype.

    Compares each archetype to archetype_0 (centroid/generalist) to find
    features representing specialized states or differentiation away from
    the central cellular state.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with archetypal assignments.
    data_obsm_key : str, default: "pathway_scores"
        Key in adata.obsm for scores.
    obs_key : str, default: "archetypes"
        Column in adata.obs with archetypal assignments.
    test_method : str, default: "mannwhitneyu"
        Statistical test method.
    fdr_method : str, default: "benjamini_hochberg"
        FDR correction method.
    fdr_scope : {'global', 'per_archetype', 'none'}, default: 'global'
        Scope of FDR correction.
    min_cells : int, default: 10
        Minimum cells per archetype.
    verbose : bool, default: True
        Print progress.

    Returns
    -------
    pd.DataFrame
        Results showing specialization from archetype_0.

    Notes
    -----
    Archetype_0 typically represents the centroid or generalist state
    where cells have balanced contributions from all archetypes.
    Features elevated in other archetypes relative to archetype_0
    represent specialized cellular programs.

    Examples
    --------
    >>> spec = pc.tl.specialization_patterns(adata)
    >>> # Find archetype_4 specialization features
    >>> arch4_spec = spec[(spec["archetype"] == "archetype_4") & (spec["significant"])]

    See Also
    --------
    peach.tl.archetype_exclusive_patterns : Exclusive pattern analysis
    """
    if obs_key not in adata.obs.columns:
        raise ValueError(f"adata.obs['{obs_key}'] not found. Run pc.tl.assign_archetypes() first.")

    return _specialization_patterns(
        adata=adata,
        data_obsm_key=data_obsm_key,
        obs_key=obs_key,
        test_method=test_method,
        fdr_method=fdr_method,
        fdr_scope=fdr_scope,
        min_cells=min_cells,
        verbose=verbose,
    )


def tradeoff_patterns(
    adata: AnnData,
    *,
    data_obsm_key: str = "pathway_scores",
    obs_key: str = "archetypes",
    tradeoffs: Literal["pairs", "patterns"] = "pairs",
    test_method: str = "mannwhitneyu",
    fdr_method: str = "benjamini_hochberg",
    fdr_scope: Literal["global", "per_archetype", "none"] = "global",
    min_cells: int = 10,
    min_effect_size: float = 0.1,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Identify mutual exclusivity and tradeoff patterns.

    Finds features showing opposing patterns between archetypes,
    indicating biological tradeoffs or mutually exclusive states.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with archetypal assignments.
    data_obsm_key : str, default: "pathway_scores"
        Key in adata.obsm for scores.
    obs_key : str, default: "archetypes"
        Column in adata.obs with archetypal assignments.
    tradeoffs : {'pairs', 'patterns'}, default: 'pairs'
        Type of tradeoff analysis:

        - ``'pairs'`` : Binary pairwise (A high, B low)
        - ``'patterns'`` : Complex multi-archetype (AB high, CD low)
    test_method : str, default: "mannwhitneyu"
        Statistical test method.
    fdr_method : str, default: "benjamini_hochberg"
        FDR correction method.
    fdr_scope : {'global', 'per_archetype', 'none'}, default: 'global'
        Scope of FDR correction.
    min_cells : int, default: 10
        Minimum cells per group.
    min_effect_size : float, default: 0.1
        Minimum effect size for tradeoffs.
    verbose : bool, default: True
        Print progress.
    **kwargs
        Additional parameters:

        - ``max_pattern_size`` : int, default: 2
            Maximum archetypes per group for complex patterns.
        - ``exclude_archetype_0`` : bool, default: True
            Exclude archetype_0 from tradeoff patterns.
        - ``specific_patterns`` : List[str], optional
            Test only specific patterns (e.g., ['2v3', '1v45']).

    Returns
    -------
    pd.DataFrame
        Results with tradeoff patterns:

        - ``pattern_code`` : Visual pattern code
        - ``high_archetypes``, ``low_archetypes`` : Groups
        - ``mean_high``, ``mean_low`` : Group means
        - ``log_fold_change`` : Effect size
        - ``pattern_complexity`` : Number of archetypes involved

    Examples
    --------
    >>> # Find pairwise tradeoffs
    >>> pairs = pc.tl.tradeoff_patterns(adata, tradeoffs="pairs")
    >>> # Find complex patterns
    >>> patterns = pc.tl.tradeoff_patterns(adata, tradeoffs="patterns", max_pattern_size=3)
    >>> # Test specific hypothesis
    >>> specific = pc.tl.tradeoff_patterns(adata, specific_patterns=["2v3", "1v4"])

    See Also
    --------
    peach.tl.archetype_exclusive_patterns : Exclusive pattern analysis
    peach._core.types.PatternAssociationResult : Result row structure
    """
    if obs_key not in adata.obs.columns:
        raise ValueError(f"adata.obs['{obs_key}'] not found. Run pc.tl.assign_archetypes() first.")

    # Extract allowed kwargs
    core_params = {}
    for key in ["max_pattern_size", "exclude_archetype_0", "specific_patterns"]:
        if key in kwargs:
            core_params[key] = kwargs[key]

    return _tradeoff_patterns(
        adata=adata,
        data_obsm_key=data_obsm_key,
        obs_key=obs_key,
        tradeoffs=tradeoffs,
        test_method=test_method,
        fdr_method=fdr_method,
        fdr_scope=fdr_scope,
        min_cells=min_cells,
        min_effect_size=min_effect_size,
        verbose=verbose,
        **core_params,
    )

# # Suppress OmniPath messages before any imports
# import os
# import logging
# os.environ['OMNIPATH_SILENT'] = '1'
# logging.getLogger('omnipath').setLevel(logging.ERROR)

# """
# PHASE 6A: Gene Set Enrichment and Pathway Analysis with Archetypal Integration
# =============================================================================

# PURPOSE: Gene set enrichment analysis using MSigDB/OmniPath integration with archetypal analysis results.

# ARCHITECTURAL ROLE:
# - Extension of Phase 5 analysis with biological pathway interpretation
# - Integrates AUCell pathway scoring for robust gene set activity estimation
# - Bridges unsupervised archetypal discovery with supervised biological interpretation
# - Provides pathway scores for downstream statistical testing

# DESIGN PRINCIPLES:
# - Use MSigDB for comprehensive gene set collections (preferred over OmniPath)
# - AUCell scoring for robust, non-parametric pathway activity estimates
# - DataFrame-based API consistent with existing analysis functions
# - AnnData.obsm integration for scverse ecosystem compatibility
# - Clear separation between pathway scoring and archetypal association testing
# - Optimized sparse matrix handling for large-scale single-cell data

# PERFORMANCE OPTIMIZATIONS:
# - Gene symbol normalization for better pathway-expression overlap
# - Efficient sparse matrix processing with format conversion recommendations
# - Memory-efficient pathway scoring with progress reporting

# === MODULE API INVENTORY ===

# MAIN FUNCTIONS:
#  load_pathway_networks(sources=['kegg'], organism='human', verbose=True) -> pd.DataFrame
#     Purpose: Load pathway gene sets from OmniPath databases using decoupler
#     Inputs: sources(List[str] pathway databases), organism(str species), verbose(bool)
#     Outputs: DataFrame with 'source', 'target', 'weight', 'pathway' columns (decoupler net format)
#     Side Effects: Downloads pathway data from OmniPath, caches locally, progress reporting

#  compute_pathway_scores(adata, net, method='ulm', use_layer=None, obsm_key='pathway_scores', verbose=True) -> pd.DataFrame
#     Purpose: Compute pathway activity scores using decoupler methods (ULM, etc.)
#     Inputs: adata(AnnData), net(DataFrame from load_pathway_networks), method(str), use_layer(str optional), obsm_key(str), verbose(bool)
#     Outputs: DataFrame with cells as rows, pathways as columns, activity scores as values
#     Side Effects: Updates adata.obsm with pathway scores, full dataset processing

#  integrate_pathways_with_archetypes(pathway_scores_df, distances_df, verbose=True) -> pd.DataFrame
#      Purpose: Combine pathway scores with archetypal distances for integrated analysis
#      Inputs: pathway_scores_df(DataFrame from compute_pathway_scores), distances_df(DataFrame from get_all_archetypal_distances), verbose(bool)
#      Outputs: DataFrame with cell_idx, archetype distances, pathway scores, and metadata
#      Side Effects: Merges datasets on cell indices, prepares for downstream statistical testing

# EXTERNAL DEPENDENCIES:
#  From decoupler: OmniPath network loading, ULM pathway scoring, standardized pathway databases
#  From omnipath: Direct API access for pathway data retrieval and caching
#  From pandas: DataFrame operations, merging, statistical functions
#  From numpy: Numerical operations, array handling
#  From typing: Type hints for function signatures

# DATA FLOW PATTERNS:
#  Pathway Loading: OmniPath/MSigDB API → Network DataFrame → Cached locally → Ready for scoring
#  Pathway Scoring: AnnData + Network → AUCell computation → Pathway scores DataFrame → AnnData.obsm storage
#  Integration: Pathway scores + Archetype distances → Merged DataFrame → Ready for statistical testing
#  Output: Clean DataFrame format for downstream 1-vs-all statistical testing

# ERROR HANDLING:
#  Network loading failures with OmniPath API connectivity
#  Gene symbol matching between datasets and pathway networks
#  Missing data handling in pathway score computation
#  Cell index alignment between different DataFrame sources
#  Memory management for large pathway score matrices

# BIOLOGICAL INTERPRETATION:
#  AUCell scores represent pathway activity: higher scores = higher pathway activity
#  Integration with archetype distances enables statistical testing across cell states
#  Downstream 1-vs-all testing identifies biological functions of each archetype
#  Prepared for Wilcoxon rank-sum tests and hypergeometric enrichment analysis
# """

# import pandas as pd
# import numpy as np
# from typing import List, Dict, Union, Optional, Tuple
# import warnings
# import logging

# # Suppress OmniPath logging messages that come from pyscenic/decoupler dependencies
# # This must be done before importing pyscenic
# import os
# os.environ['OMNIPATH_SILENT'] = '1'  # Suppress OmniPath messages

# # Set logging levels to suppress INFO messages
# omnipath_logger = logging.getLogger('omnipath')
# omnipath_logger.setLevel(logging.ERROR)  # Only show errors

# # Also suppress decoupler logging if present
# decoupler_logger = logging.getLogger('decoupler')
# decoupler_logger.setLevel(logging.ERROR)

# # Suppress other related loggers
# for logger_name in ['omnipath.requests', 'omnipath.interactions', 'omnipath.cache']:
#     logger = logging.getLogger(logger_name)
#     logger.setLevel(logging.ERROR)

# # Import decoupler and omnipath
# # DECOUPLER_AVAILABLE = False
# # try:
# #     import decoupler as dc
# #     import omnipath as op
# #     # from omnipath.requests import annotations
# #     DECOUPLER_AVAILABLE = True
# # except ImportError as e:
# #     warnings.warn(f"decoupler/omnipath not available at import time: {e}. Will check at runtime.")


# def load_pathway_networks(
#     sources: List[str] = ['c5_bp'],
#     organism: str = 'human',
#     geneset_repo: str = 'msigdb',  # 'msigdb' or 'omnipath'
#     verbose: bool = True
# ) -> pd.DataFrame:
#     """
#     Load pathway gene sets from MSigDB or OmniPath databases.

#     Downloads standardized pathway networks for gene set enrichment analysis.
#     MSigDB provides much richer collections than OmniPath.

#     Args:
#         sources: List of pathway databases to load.
#                  MSigDB options: ['hallmark', 'c2_cp', 'c2_cgp', 'c3_mir', 'c5_bp', 'c5_cc', 'c5_mf', 'c8']
#                  OmniPath options: ['hallmark', 'progeny', 'dorothea']
#         organism: Species for pathway data ('human', 'mouse')
#         geneset_repo: Repository to use ('msigdb' or 'omnipath')
#         verbose: Whether to print loading progress and statistics

#     Returns:
#         pd.DataFrame with columns ['source', 'target', 'weight', 'pathway']:
#             - source: Pathway/gene set name
#             - target: Gene symbol
#             - weight: Always 1.0 for binary membership
#             - pathway: Pathway database source

#     Examples:
#         # Load MSigDB Hallmark gene sets (recommended)
#         hallmark = load_pathway_networks(['hallmark'], geneset_repo='msigdb')

#         # Load multiple MSigDB collections
#         multi = load_pathway_networks(['hallmark', 'c2_cp', 'c5_bp'], geneset_repo='msigdb')

#         # Load for mouse
#         mouse = load_pathway_networks(['hallmark'], organism='mouse', geneset_repo='msigdb')

#         # Use OmniPath (smaller collection)
#         omni = load_pathway_networks(['hallmark'], geneset_repo='omnipath')
#     """
#     if verbose:
#         print(f" Loading pathway networks from {geneset_repo.upper()}...")
#         print(f"   Sources: {sources}")
#         print(f"   Organism: {organism}")

#     if geneset_repo.lower() == 'msigdb':
#         return _load_msigdb_pathways(sources, organism, verbose)
#     # elif geneset_repo.lower() == 'omnipath':
#     #     return _load_omnipath_pathways(sources, organism, verbose)
#     # else:
#     #     raise ValueError(f"Unknown geneset_repo: {geneset_repo}. Use 'msigdb' or 'omnipath'")


# def _load_msigdb_pathways(sources: List[str], organism: str, verbose: bool) -> pd.DataFrame:
#     """Load pathway networks from MSigDB using gseapy."""
#     try:
#         from gseapy import Msigdb
#     except ImportError:
#         raise ImportError("gseapy is required for MSigDB access. Install with: pip install gseapy")

#     # Map organism to MSigDB version suffix
#     org_map = {
#         'human': 'Hs',
#         'mouse': 'Mm'
#     }

#     if organism.lower() not in org_map:
#         raise ValueError(f"Organism '{organism}' not supported for MSigDB. Use 'human' or 'mouse'")

#     org_suffix = org_map[organism.lower()]
#     msig = Msigdb()

#     # Map source names to MSigDB categories
#     source_map = {
#         'hallmark': f'h.all.v2025.1.{org_suffix}',
#         'c2_cp': f'c2.cp.v2025.1.{org_suffix}',  # Canonical pathways
#         'c2_cgp': f'c2.cgp.v2025.1.{org_suffix}',  # Chemical/genetic perturbations
#         'c3_mir': f'c3.mir.v2025.1.{org_suffix}',  # microRNA targets
#         'c5_bp': f'c5.go.bp.v2025.1.{org_suffix}',  # GO Biological Process
#         'c5_cc': f'c5.go.cc.v2025.1.{org_suffix}',  # GO Cellular Component
#         'c5_mf': f'c5.go.mf.v2025.1.{org_suffix}',  # GO Molecular Function
#         'c8': f'c8.all.v2025.1.{org_suffix}',  # Cell type signatures
#     }

#     pathway_networks = []

#     for source in sources:
#         if verbose:
#             print(f"\n Loading {source.upper()} from MSigDB...")

#         try:
#             if source.lower() not in source_map:
#                 if verbose:
#                     print(f"   [WARNING]  Warning: Unknown MSigDB source '{source}', skipping...")
#                 continue

#             category = source_map[source.lower()]

#             # Load GMT data from MSigDB
#             gmt_data = msig.get_gmt(category=category.split('.v')[0], dbver=f"2025.1.{org_suffix}")

#             # Convert to network format
#             pathway_rows = []
#             for pathway_name, gene_list in gmt_data.items():
#                 for gene in gene_list:
#                     pathway_rows.append({
#                         'source': pathway_name,
#                         'target': gene,
#                         'weight': 1.0,
#                         'pathway': source.upper()
#                     })

#             if pathway_rows:
#                 net = pd.DataFrame(pathway_rows)
#                 pathway_networks.append(net)

#                 if verbose:
#                     n_pathways = len(gmt_data)
#                     n_genes = len(set([row['target'] for row in pathway_rows]))
#                     n_interactions = len(pathway_rows)
#                     print(f"    Loaded {n_pathways} pathways, {n_genes} unique genes, {n_interactions} gene-pathway pairs")

#         except Exception as e:
#             if verbose:
#                 print(f"   [ERROR] Failed to load {source}: {e}")
#             continue

#     if not pathway_networks:
#         raise ValueError(f"Failed to load any MSigDB pathway networks from sources: {sources}")

#     # Combine all pathway networks
#     combined_net = pd.concat(pathway_networks, ignore_index=True)

#     if verbose:
#         total_pathways = combined_net['source'].nunique()
#         total_genes = combined_net['target'].nunique()
#         total_interactions = len(combined_net)

#         print(f"\n[STATS] Combined MSigDB Network Summary:")
#         print(f"   Total pathways: {total_pathways}")
#         print(f"   Total unique genes: {total_genes}")
#         print(f"   Total gene-pathway pairs: {total_interactions}")

#         # Show pathway distribution by source
#         pathway_counts = combined_net.groupby('pathway')['source'].nunique()
#         print(f"\n   Pathways per database:")
#         for db, count in pathway_counts.items():
#             print(f"      {db}: {count} pathways")

#     return combined_net

# def compute_pathway_scores(
#     adata,
#     net: pd.DataFrame,
#     use_layer: Optional[str] = None,
#     obsm_key: str = 'pathway_scores',
#     verbose: bool = True
# ) -> pd.DataFrame:
#     """
#     Compute pathway activity scores using AUCell.

#     Uses pySCENIC's AUCell to compute pathway activity scores from gene expression data.
#     AUCell provides robust, rank-based pathway activity estimates.

#     Args:
#         adata: AnnData object with gene expression data
#         net: Pathway network DataFrame from load_pathway_networks()
#         use_layer: AnnData layer to use (None = X, 'logcounts', 'raw', etc.)
#         obsm_key: Key for storing results in adata.obsm
#         verbose: Whether to print computation progress

#     Returns:
#         pd.DataFrame with cells as rows, pathways as columns, activity scores as values

#     Examples:
#         # Compute AUCell pathway scores
#         pathway_scores = compute_pathway_scores(adata, net)

#         # Use log-transformed counts
#         pathway_scores = compute_pathway_scores(adata, net, use_layer='logcounts')
#     """
#     if verbose:
#         print(f"🧮 Computing pathway activity scores using AUCell...")
#         print(f"   Data layer: {use_layer if use_layer else 'X (default)'}")
#         print(f"   Network: {net['source'].nunique()} pathways, {net['target'].nunique()} genes")

#     # Prepare expression data
#     if use_layer is None:
#         expr_data = adata.X
#         if hasattr(expr_data, 'toarray'):  # Handle sparse matrices
#             expr_data = expr_data.toarray()
#         expr_df = pd.DataFrame(expr_data.T, index=adata.var.index, columns=adata.obs.index)
#     else:
#         if use_layer not in adata.layers:
#             raise ValueError(f"Layer '{use_layer}' not found in adata.layers. Available: {list(adata.layers.keys())}")
#         layer_data = adata.layers[use_layer]
#         if hasattr(layer_data, 'toarray'):
#             layer_data = layer_data.toarray()
#         expr_df = pd.DataFrame(layer_data.T, index=adata.var.index, columns=adata.obs.index)

#     if verbose:
#         print(f"   Expression data: {expr_df.shape[0]} genes × {expr_df.shape[1]} cells")

#         # Check gene overlap between expression data and pathway network
#         expr_genes = set(expr_df.index)
#         pathway_genes = set(net['target'].unique())
#         overlap_genes = expr_genes & pathway_genes

#         print(f"   Gene overlap check:")
#         print(f"      Expression genes: {len(expr_genes)}")
#         print(f"      Pathway genes: {len(pathway_genes)}")
#         print(f"      Overlapping genes: {len(overlap_genes)} ({100*len(overlap_genes)/len(expr_genes):.1f}% of expression genes)")

#         # Show sample gene symbols from each dataset for format comparison
#         sample_expr_genes = list(expr_genes)[:5]
#         sample_pathway_genes = list(pathway_genes)[:5]
#         print(f"      Sample expression genes: {sample_expr_genes}")
#         print(f"      Sample pathway genes: {sample_pathway_genes}")

#         # TYPE DEBUGGING: Check if expression genes are actually integers
#         if sample_expr_genes and isinstance(sample_expr_genes[0], int):
#             print(f"   [WARNING]  WARNING: Expression gene identifiers are integers, not gene symbols!")
#             print(f"      This suggests adata.var.index contains numeric indices instead of gene names.")
#             print(f"      Check if adata.var['gene_symbols'] or similar column contains actual gene names.")

#             # Check for potential gene symbol columns
#             if hasattr(adata, 'var') and len(adata.var.columns) > 0:
#                 print(f"      Available adata.var columns: {list(adata.var.columns)}")
#                 gene_symbol_candidates = [col for col in adata.var.columns if 'gene' in col.lower() or 'symbol' in col.lower()]
#                 if gene_symbol_candidates:
#                     print(f"      Potential gene symbol columns: {gene_symbol_candidates}")
#             print(f"      Proceeding with numeric conversion, but pathway overlap will likely be 0.")

#         if len(overlap_genes) < 50:
#             print(f"   [WARNING]  Warning: Low gene overlap may affect pathway scoring quality")
#             if len(overlap_genes) > 0:
#                 print(f"      Sample overlapping genes: {list(overlap_genes)[:10]}")

#         # Gene symbol normalization attempt (with type safety)
#         # Convert all gene identifiers to strings first, then normalize
#         expr_genes_str = {str(gene) for gene in expr_genes}
#         pathway_genes_str = {str(gene) for gene in pathway_genes}

#         expr_genes_upper = {gene.upper().strip() for gene in expr_genes_str}
#         pathway_genes_upper = {gene.upper().strip() for gene in pathway_genes_str}
#         overlap_upper = expr_genes_upper.intersection(pathway_genes_upper)

#         if len(overlap_upper) > len(overlap_genes):
#             print(f"   NOTE:  Case-insensitive overlap: {len(overlap_upper)} genes (improvement: +{len(overlap_upper) - len(overlap_genes)})")
#             print(f"      Consider normalizing gene symbols to uppercase")

#             # Optionally normalize gene symbols for better overlap
#             print(f"    Attempting gene symbol normalization...")

#             # Create normalized expression DataFrame (with type safety)
#             expr_df_norm = expr_df.copy()
#             expr_df_norm.index = expr_df_norm.index.astype(str).str.upper().str.strip()

#             # Create normalized network (with type safety)
#             net_norm = net.copy()
#             net_norm['target'] = net_norm['target'].astype(str).str.upper().str.strip()

#             # Recalculate overlap with normalized data
#             overlap_norm = set(expr_df_norm.index) & set(net_norm['target'].unique())
#             print(f"      Normalized overlap: {len(overlap_norm)} genes (+{len(overlap_norm) - len(overlap_genes)} improvement)")

#             # Use normalized data if significantly better
#             if len(overlap_norm) > len(overlap_genes) * 1.2:  # 20% improvement threshold
#                 print(f"   [OK] Using normalized gene symbols for analysis")
#                 expr_df = expr_df_norm
#                 net = net_norm
#             else:
#                 print(f"   [WARNING]  Normalization didn't provide sufficient improvement, using original symbols")

#     # Debug pathway network structure before running decoupler
#     if verbose:
#         print(f"    Pathway network debugging:")
#         pathway_counts = net.groupby('source').size()
#         print(f"      Total pathways in network: {len(pathway_counts)}")
#         print(f"      Genes per pathway stats: min={pathway_counts.min()}, max={pathway_counts.max()}, median={pathway_counts.median():.1f}")

#         # Check which pathways have genes in the expression data
#         expr_genes_set = set(expr_df.index)
#         filtered_net = net[net['target'].isin(expr_genes_set)]
#         filtered_pathway_counts = filtered_net.groupby('source').size()

#         print(f"      After filtering to expression genes:")
#         print(f"         Pathways remaining: {len(filtered_pathway_counts)}")
#         if len(filtered_pathway_counts) > 0:
#             print(f"         Genes per pathway: min={filtered_pathway_counts.min()}, max={filtered_pathway_counts.max()}, median={filtered_pathway_counts.median():.1f}")
#             print(f"         Pathways with ≥1 gene: {(filtered_pathway_counts >= 1).sum()}")
#             print(f"         Pathways with ≥2 genes: {(filtered_pathway_counts >= 2).sum()}")
#             print(f"         Pathways with ≥5 genes: {(filtered_pathway_counts >= 5).sum()}")
#         else:
#             print(f"         [ERROR] No pathways have any genes in common with expression data!")

#     # Run AUCell pathway analysis
#     try:
#         # AUCell from pySCENIC - robust gene set scoring
#         if verbose:
#             print(f"    Running AUCell (pySCENIC) - robust gene set scoring")

#         try:
#             from pyscenic.aucell import aucell
#             from ctxcore.genesig import GeneSignature
#         except ImportError:
#             raise ImportError("pySCENIC and ctxcore are required for AUCell method. Install with: pip install pyscenic")

#         # Prepare expression data (cells × genes format for AUCell)
#         # Note: AUCell expects (n_cells x n_genes), opposite of what we had before
#         if use_layer is None:
#             expr_data = adata.X
#             if hasattr(expr_data, 'toarray'):
#                 expr_data = expr_data.toarray()
#             expr_df = pd.DataFrame(expr_data, index=adata.obs.index, columns=adata.var.index)
#         else:
#             layer_data = adata.layers[use_layer]
#             if hasattr(layer_data, 'toarray'):
#                 layer_data = layer_data.toarray()
#             expr_df = pd.DataFrame(layer_data, index=adata.obs.index, columns=adata.var.index)

#         if verbose:
#             print(f"      Expression matrix: {expr_df.shape[0]} cells × {expr_df.shape[1]} genes")

#         # Convert network to GeneSignature objects for AUCell
#         # Use simple binary gene sets (no complex weight handling)
#         gene_signatures = []

#         for pathway in net['source'].unique():
#             pathway_data = net[net['source'] == pathway]

#             # Get genes for this pathway that are present in expression data
#             pathway_genes = [
#                 row['target'] for _, row in pathway_data.iterrows()
#                 if row['target'] in expr_df.columns
#             ]

#             # Create simple binary gene signature
#             if len(pathway_genes) > 0:
#                 # All genes get weight 1.0 for binary scoring
#                 gene2weight = {gene: 1.0 for gene in pathway_genes}
#                 signature = GeneSignature(name=pathway, gene2weight=gene2weight)
#                 gene_signatures.append(signature)

#         if verbose:
#             print(f"      Converted to {len(gene_signatures)} binary gene signatures")
#             avg_genes = np.mean([len(sig.genes) for sig in gene_signatures]) if gene_signatures else 0
#             print(f"      Average genes per signature: {avg_genes:.1f}")

#         if len(gene_signatures) == 0:
#             raise ValueError("No pathways have genes present in the expression data")

#         # Run AUCell scoring - use noweights=True for binary scoring
#         auc_scores = aucell(
#             expr_df,  # cells × genes DataFrame
#             gene_signatures,  # List of GeneSignature objects
#             auc_threshold=0.05,  # Standard threshold (0.05 = top 5% of genes)
#             noweights=True,     # Ignore weights - use binary scoring
#             normalize=False,    # Don't normalize (we want raw AUC scores)
#             num_workers=1
#         )

#         # AUCell returns DataFrame with cells as rows, signatures as columns
#         # Column names are automatically set to signature names by AUCell
#         if verbose:
#             print(f"      AUCell result shape: {auc_scores.shape[0]} cells × {auc_scores.shape[1]} signatures")
#             print(f"      Signature names: {list(auc_scores.columns[:3])}...")  # Show first 3

#         # Convert to our expected format (pathways × cells)
#         pathway_scores = auc_scores.T
#         pathway_pvals = None  # AUCell doesn't provide p-values

#         if verbose:
#             print(f"   [OK] AUCell completed: {pathway_scores.shape[0]} pathways × {pathway_scores.shape[1]} cells")
#             print(f"      Score range: {pathway_scores.values.min():.3f} to {pathway_scores.values.max():.3f}")

#     except Exception as e:
#         raise RuntimeError(f"Pathway scoring failed with AUCell: {e}")

#     # Convert to DataFrame with cells as rows, pathways as columns
#     pathway_scores_df = pathway_scores.T  # Transpose: pathways x cells -> cells x pathways

#     # CRITICAL FIX: Ensure perfect alignment with adata.obs.index
#     # The scores should already be aligned since we used adata.obs.index in AUCell
#     # But we need to verify and ensure consistent ordering
#     if not pathway_scores_df.index.equals(adata.obs.index):
#         if verbose:
#             print(f"    Realigning pathway scores with adata.obs.index...")

#         # Reindex to match adata.obs.index exactly
#         pathway_scores_df = pathway_scores_df.reindex(adata.obs.index, fill_value=0.0)

#         if verbose:
#             print(f"   [OK] Pathway scores realigned to adata.obs.index")

#     # Reset to 0-based integer index for backward compatibility
#     # pathway_scores_df = pathway_scores_df.reset_index(drop=True)
#     # pathway_scores_df.index.name = 'cell_idx'

#     # # Add explicit cell_idx column (0-based positions)
#     # pathway_scores_df = pathway_scores_df.reset_index()

#     if verbose:
#         print(f"[OK] Pathway scoring completed!")
#         print(f"   Scores shape: {pathway_scores_df.shape}")
#         print(f"   Pathways scored: {pathway_scores_df.shape[1]}")
#         print(f"   Score range: [{pathway_scores_df.values.min():.3f}, {pathway_scores_df.values.max():.3f}]")

#         # Show some statistics
#         print(f"\n[STATS] Pathway Score Statistics:")
#         print(f"   Mean pathway activity: {pathway_scores_df.mean().mean():.3f}")
#         print(f"   Std pathway activity: {pathway_scores_df.std().mean():.3f}")

#         # Top variable pathways
#         pathway_vars = pathway_scores_df.var().sort_values(ascending=False)
#         print(f"\n   Most variable pathways:")
#         for i, (pathway, var) in enumerate(pathway_vars.head(5).items()):
#             print(f"      {i+1}. {pathway}: variance = {var:.3f}")

#     # Store in AnnData.obsm (PRIMARY STORAGE)
#     try:
#         # pathway_scores_df should already be aligned with adata.obs.index
#         # No need to remove 'cell_idx' column since it doesn't exist
#         pathway_matrix = pathway_scores_df.values  # [n_cells, n_pathways]
#         pathway_names = list(pathway_scores_df.columns)

#         # Store pathway scores matrix (aligned with adata.obs row order)
#         adata.obsm[obsm_key] = pathway_matrix

#         # Store pathway names as metadata
#         adata.uns[f'{obsm_key}_pathways'] = pathway_names
#         adata.uns[f'{obsm_key}_method'] = 'aucell'

#         if verbose:
#             print(f"    Stored in AnnData:")
#             print(f"      adata.obsm['{obsm_key}']: {pathway_matrix.shape} pathway scores")
#             print(f"      adata.uns['{obsm_key}_pathways']: {len(pathway_names)} pathway names")
#             print(f"      adata.uns['{obsm_key}_method']: aucell")

#     except Exception as e:
#         if verbose:
#             print(f"   [WARNING]  Warning: Failed to store in adata.obsm: {e}")

#     return pathway_scores_df

"""
Gene Set Enrichment and Pathway Analysis
=========================================

MSigDB/AUCell integration for pathway activity scoring in archetypal analysis.

This module provides pathway activity scoring using AUCell (pySCENIC) with
gene sets from MSigDB. Pathway scores are stored in AnnData.obsm for seamless
integration with downstream statistical testing.

Main Functions
--------------
load_pathway_networks
    Load gene sets from MSigDB
compute_pathway_scores
    Compute AUCell pathway activity scores

Type Definitions
----------------
See ``peach._core.types`` for Pydantic models:

- ``PathwayNetworkRow`` : Network DataFrame row structure
- ``PathwayNetworkConfig`` : load_pathway_networks() configuration
- ``PathwayScoreConfig`` : compute_pathway_scores() configuration
- ``GeneOverlapStats`` : Gene overlap statistics
- ``PathwayScoreSummary`` : Score summary statistics
- ``PathwayScoreAnnDataStorage`` : AnnData storage documentation

Workflow
--------
1. Load pathway networks from MSigDB
2. Compute AUCell scores for each pathway
3. Store scores in adata.obsm for statistical testing
4. Use with statistical testing functions (gene_associations, pathway_associations)

Examples
--------
>>> import peach as pc
>>> # Load Hallmark gene sets
>>> net = pc.pp.load_pathway_networks(["hallmark"])
>>> # Compute pathway scores
>>> scores = pc.pp.compute_pathway_scores(adata, net)
>>> # Scores are stored in AnnData
>>> print(adata.obsm["pathway_scores"].shape)  # [n_cells, n_pathways]
>>> print(adata.uns["pathway_scores_pathways"])  # Pathway names
>>> # Use with statistical testing
>>> results = pc.tl.pathway_associations(adata)

See Also
--------
peach.tl.pathway_associations : Statistical testing of pathway-archetype associations
peach.tl.pattern_analysis : Comprehensive pattern analysis
"""

# Suppress OmniPath messages before any imports
import logging
import os

os.environ["OMNIPATH_SILENT"] = "1"
logging.getLogger("omnipath").setLevel(logging.ERROR)


import numpy as np
import pandas as pd


def load_pathway_networks(
    sources: list[str] = ["c5_bp"], organism: str = "human", geneset_repo: str = "msigdb", verbose: bool = True
) -> pd.DataFrame:
    """
    Load pathway gene sets from MSigDB.

    Downloads standardized pathway networks for gene set enrichment analysis.
    MSigDB provides comprehensive, curated gene set collections.

    Parameters
    ----------
    sources : list[str], default: ['c5_bp']
        Pathway databases to load. Available MSigDB collections:

        - ``'hallmark'`` : Hallmark gene sets (50 curated biological processes)
        - ``'c2_cp'`` : Canonical pathways (KEGG, Reactome, BioCarta, etc.)
        - ``'c2_cgp'`` : Chemical and genetic perturbations
        - ``'c3_mir'`` : microRNA targets
        - ``'c5_bp'`` : GO Biological Process (recommended for functional analysis)
        - ``'c5_cc'`` : GO Cellular Component
        - ``'c5_mf'`` : GO Molecular Function
        - ``'c8'`` : Cell type signature gene sets
    organism : str, default: 'human'
        Species for pathway data: 'human' or 'mouse'.
    geneset_repo : str, default: 'msigdb'
        Repository to use. Currently only 'msigdb' supported.
    verbose : bool, default: True
        Whether to print loading progress and statistics.

    Returns
    -------
    pd.DataFrame
        Pathway network with columns:

        - ``source`` : str - Pathway/gene set name (e.g., 'HALLMARK_HYPOXIA')
        - ``target`` : str - Gene symbol (e.g., 'VEGFA')
        - ``weight`` : float - Membership weight (always 1.0 for binary)
        - ``pathway`` : str - Database source (e.g., 'HALLMARK', 'C5_BP')

    Raises
    ------
    ImportError
        If gseapy is not installed.
    ValueError
        If organism not supported or no networks loaded.

    See Also
    --------
    compute_pathway_scores : Compute AUCell scores from network
    peach._core.types.PathwayNetworkRow : Row type definition
    peach._core.types.PathwayNetworkConfig : Configuration type

    Examples
    --------
    >>> # Load Hallmark gene sets (recommended starting point)
    >>> hallmark = load_pathway_networks(["hallmark"])
    >>> print(f"Loaded {hallmark['source'].nunique()} pathways")
    >>> # Load multiple collections
    >>> multi = load_pathway_networks(["hallmark", "c2_cp", "c5_bp"])
    >>> # Load for mouse
    >>> mouse_net = load_pathway_networks(["hallmark"], organism="mouse")
    >>> # Check network structure
    >>> print(hallmark.columns.tolist())
    ['source', 'target', 'weight', 'pathway']
    >>> print(hallmark.head())
    """
    if verbose:
        print(f"📂 Loading pathway networks from {geneset_repo.upper()}...")
        print(f"   Sources: {sources}")
        print(f"   Organism: {organism}")

    if geneset_repo.lower() == "msigdb":
        return _load_msigdb_pathways(sources, organism, verbose)
    else:
        raise ValueError(f"Unknown geneset_repo: {geneset_repo}. Use 'msigdb'.")


def _load_msigdb_pathways(sources: list[str], organism: str, verbose: bool) -> pd.DataFrame:
    """
    Load pathway networks from MSigDB using gseapy.

    Internal function called by load_pathway_networks().

    Parameters
    ----------
    sources : list[str]
        MSigDB collection names.
    organism : str
        Species: 'human' or 'mouse'.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Combined pathway network with columns:
        ['source', 'target', 'weight', 'pathway'].

    Raises
    ------
    ImportError
        If gseapy not available.
    ValueError
        If organism not supported or no networks loaded.
    """
    try:
        from gseapy import Msigdb
    except ImportError:
        raise ImportError("gseapy is required for MSigDB access. Install with: pip install gseapy")

    # Map organism to MSigDB version suffix
    org_map = {"human": "Hs", "mouse": "Mm"}

    if organism.lower() not in org_map:
        raise ValueError(f"Organism '{organism}' not supported. Use 'human' or 'mouse'.")

    org_suffix = org_map[organism.lower()]
    msig = Msigdb()

    # Map source names to MSigDB categories
    source_map = {
        "hallmark": f"h.all.v2025.1.{org_suffix}",
        "c2_cp": f"c2.cp.v2025.1.{org_suffix}",
        "c2_cgp": f"c2.cgp.v2025.1.{org_suffix}",
        "c3_mir": f"c3.mir.v2025.1.{org_suffix}",
        "c5_bp": f"c5.go.bp.v2025.1.{org_suffix}",
        "c5_cc": f"c5.go.cc.v2025.1.{org_suffix}",
        "c5_mf": f"c5.go.mf.v2025.1.{org_suffix}",
        "c8": f"c8.all.v2025.1.{org_suffix}",
    }

    pathway_networks = []

    for source in sources:
        if verbose:
            print(f"\n📦 Loading {source.upper()} from MSigDB...")

        try:
            if source.lower() not in source_map:
                if verbose:
                    print(f"   ⚠️ Warning: Unknown MSigDB source '{source}', skipping...")
                continue

            category = source_map[source.lower()]

            # Load GMT data from MSigDB
            gmt_data = msig.get_gmt(category=category.split(".v")[0], dbver=f"2025.1.{org_suffix}")

            # Convert to network format
            pathway_rows = []
            for pathway_name, gene_list in gmt_data.items():
                for gene in gene_list:
                    pathway_rows.append(
                        {"source": pathway_name, "target": gene, "weight": 1.0, "pathway": source.upper()}
                    )

            if pathway_rows:
                net = pd.DataFrame(pathway_rows)
                pathway_networks.append(net)

                if verbose:
                    n_pathways = len(gmt_data)
                    n_genes = len(set([row["target"] for row in pathway_rows]))
                    n_interactions = len(pathway_rows)
                    print(
                        f"   ✓ Loaded {n_pathways} pathways, "
                        f"{n_genes} unique genes, "
                        f"{n_interactions} gene-pathway pairs"
                    )

        except Exception as e:
            if verbose:
                print(f"   ❌ Failed to load {source}: {e}")
            continue

    if not pathway_networks:
        raise ValueError(f"Failed to load any MSigDB pathway networks from sources: {sources}")

    # Combine all pathway networks
    combined_net = pd.concat(pathway_networks, ignore_index=True)

    if verbose:
        total_pathways = combined_net["source"].nunique()
        total_genes = combined_net["target"].nunique()
        total_interactions = len(combined_net)

        print("\n📊 Combined MSigDB Network Summary:")
        print(f"   Total pathways: {total_pathways}")
        print(f"   Total unique genes: {total_genes}")
        print(f"   Total gene-pathway pairs: {total_interactions}")

        # Show pathway distribution by source
        pathway_counts = combined_net.groupby("pathway")["source"].nunique()
        print("\n   Pathways per database:")
        for db, count in pathway_counts.items():
            print(f"      {db}: {count} pathways")

    return combined_net


def compute_pathway_scores(
    adata, net: pd.DataFrame, use_layer: str | None = None, obsm_key: str = "pathway_scores", verbose: bool = True
) -> pd.DataFrame:
    """
    Compute pathway activity scores using AUCell.

    Uses pySCENIC's AUCell algorithm to compute robust, rank-based pathway
    activity scores from gene expression data. Scores are stored in
    adata.obsm for downstream statistical testing.

    Parameters
    ----------
    adata : AnnData
        AnnData object with gene expression data.

        Requirements:

        - ``X`` or ``layers[use_layer]`` : Expression matrix [n_cells, n_genes]
        - ``var.index`` : Gene symbols matching pathway network
    net : pd.DataFrame
        Pathway network from load_pathway_networks().
        Must have columns: ['source', 'target', 'weight', 'pathway'].
    use_layer : str | None, default: None
        AnnData layer to use for expression data.
        If None, uses adata.X.
    obsm_key : str, default: 'pathway_scores'
        Key for storing results in adata.obsm.
    verbose : bool, default: True
        Whether to print computation progress.

    Returns
    -------
    pd.DataFrame
        Pathway scores with:

        - **Index**: Cell identifiers (matching adata.obs.index)
        - **Columns**: Pathway names
        - **Values**: AUCell activity scores (0.0 to ~0.5 typically)

    Raises
    ------
    ImportError
        If pySCENIC or ctxcore not installed.
    ValueError
        If use_layer not found or no pathways have overlapping genes.
    RuntimeError
        If AUCell scoring fails.

    Notes
    -----
    **AnnData Storage**:

    After calling this function, scores are stored in:

    - ``adata.obsm[obsm_key]`` : Score matrix [n_cells, n_pathways]
    - ``adata.uns[f'{obsm_key}_pathways']`` : List of pathway names
    - ``adata.uns[f'{obsm_key}_method']`` : 'aucell'

    **Gene Symbol Matching**:

    The function attempts case-insensitive gene symbol normalization if
    overlap is low. Check verbose output for gene overlap statistics.

    **AUCell Scores**:

    - Scores represent pathway activity (higher = more active)
    - Range typically 0.0 to ~0.5 (depends on gene set size)
    - Robust to outliers due to rank-based computation

    See Also
    --------
    load_pathway_networks : Load pathway gene sets
    peach.tl.pathway_associations : Statistical testing
    peach._core.types.PathwayScoreSummary : Summary statistics type
    peach._core.types.PathwayScoreAnnDataStorage : Storage documentation

    Examples
    --------
    >>> # Basic usage
    >>> net = load_pathway_networks(["hallmark"])
    >>> scores = compute_pathway_scores(adata, net)
    >>> # Use log-transformed counts
    >>> scores = compute_pathway_scores(adata, net, use_layer="logcounts")
    >>> # Access stored scores
    >>> pathway_matrix = adata.obsm["pathway_scores"]
    >>> pathway_names = adata.uns["pathway_scores_pathways"]
    >>> # Get score for specific pathway
    >>> hypoxia_idx = pathway_names.index("HALLMARK_HYPOXIA")
    >>> hypoxia_scores = pathway_matrix[:, hypoxia_idx]
    >>> # Check score statistics
    >>> print(f"Score range: {pathway_matrix.min():.3f} to {pathway_matrix.max():.3f}")
    >>> print(f"Most variable pathway: {pathway_names[pathway_matrix.var(axis=0).argmax()]}")
    """
    if verbose:
        print("🧮 Computing pathway activity scores using AUCell...")
        print(f"   Data layer: {use_layer if use_layer else 'X (default)'}")
        print(f"   Network: {net['source'].nunique()} pathways, {net['target'].nunique()} genes")

    # Prepare expression data
    if use_layer is None:
        expr_data = adata.X
        if hasattr(expr_data, "toarray"):
            expr_data = expr_data.toarray()
        expr_df = pd.DataFrame(expr_data.T, index=adata.var.index, columns=adata.obs.index)
    else:
        if use_layer not in adata.layers:
            raise ValueError(f"Layer '{use_layer}' not found in adata.layers. Available: {list(adata.layers.keys())}")
        layer_data = adata.layers[use_layer]
        if hasattr(layer_data, "toarray"):
            layer_data = layer_data.toarray()
        expr_df = pd.DataFrame(layer_data.T, index=adata.var.index, columns=adata.obs.index)

    if verbose:
        print(f"   Expression data: {expr_df.shape[0]} genes × {expr_df.shape[1]} cells")

        # Check gene overlap
        expr_genes = set(expr_df.index)
        pathway_genes = set(net["target"].unique())
        overlap_genes = expr_genes & pathway_genes

        print("   Gene overlap check:")
        print(f"      Expression genes: {len(expr_genes)}")
        print(f"      Pathway genes: {len(pathway_genes)}")
        print(
            f"      Overlapping genes: {len(overlap_genes)} "
            f"({100 * len(overlap_genes) / len(expr_genes):.1f}% of expression genes)"
        )

        # Check for integer gene indices (common error)
        sample_expr_genes = list(expr_genes)[:5]
        if sample_expr_genes and isinstance(sample_expr_genes[0], int):
            print("   ⚠️ WARNING: Expression gene identifiers are integers!")
            print("      Check if adata.var has a gene symbol column.")
            if hasattr(adata, "var") and len(adata.var.columns) > 0:
                print(f"      Available adata.var columns: {list(adata.var.columns)}")

        if len(overlap_genes) < 50:
            print("   ⚠️ Warning: Low gene overlap may affect scoring quality")

        # Try case-insensitive normalization
        expr_genes_str = {str(gene) for gene in expr_genes}
        pathway_genes_str = {str(gene) for gene in pathway_genes}
        expr_genes_upper = {gene.upper().strip() for gene in expr_genes_str}
        pathway_genes_upper = {gene.upper().strip() for gene in pathway_genes_str}
        overlap_upper = expr_genes_upper & pathway_genes_upper

        if len(overlap_upper) > len(overlap_genes):
            print(
                f"   💡 Case-insensitive overlap: {len(overlap_upper)} genes "
                f"(+{len(overlap_upper) - len(overlap_genes)})"
            )
            print("   🔄 Attempting gene symbol normalization...")

            # Normalize
            expr_df_norm = expr_df.copy()
            expr_df_norm.index = expr_df_norm.index.astype(str).str.upper().str.strip()

            net_norm = net.copy()
            net_norm["target"] = net_norm["target"].astype(str).str.upper().str.strip()

            overlap_norm = set(expr_df_norm.index) & set(net_norm["target"].unique())

            if len(overlap_norm) > len(overlap_genes) * 1.2:
                print(f"   ✓ Using normalized gene symbols (+{len(overlap_norm) - len(overlap_genes)} genes)")
                expr_df = expr_df_norm
                net = net_norm

    # Run AUCell
    try:
        if verbose:
            print("   🔬 Running AUCell (pySCENIC)...")

        try:
            from ctxcore.genesig import GeneSignature
            from pyscenic.aucell import aucell
        except ImportError:
            raise ImportError("pySCENIC and ctxcore are required for AUCell. Install with: pip install pyscenic")

        # Prepare expression data (cells × genes for AUCell)
        if use_layer is None:
            expr_data = adata.X
            if hasattr(expr_data, "toarray"):
                expr_data = expr_data.toarray()
            expr_df = pd.DataFrame(expr_data, index=adata.obs.index, columns=adata.var.index)
        else:
            layer_data = adata.layers[use_layer]
            if hasattr(layer_data, "toarray"):
                layer_data = layer_data.toarray()
            expr_df = pd.DataFrame(layer_data, index=adata.obs.index, columns=adata.var.index)

        if verbose:
            print(f"      Expression matrix: {expr_df.shape[0]} cells × {expr_df.shape[1]} genes")

        # Convert network to GeneSignature objects
        gene_signatures = []

        for pathway in net["source"].unique():
            pathway_data = net[net["source"] == pathway]

            pathway_genes = [row["target"] for _, row in pathway_data.iterrows() if row["target"] in expr_df.columns]

            if len(pathway_genes) > 0:
                gene2weight = dict.fromkeys(pathway_genes, 1.0)
                signature = GeneSignature(name=pathway, gene2weight=gene2weight)
                gene_signatures.append(signature)

        if verbose:
            print(f"      Created {len(gene_signatures)} gene signatures")
            if gene_signatures:
                avg_genes = np.mean([len(sig.genes) for sig in gene_signatures])
                print(f"      Average genes per signature: {avg_genes:.1f}")

        if len(gene_signatures) == 0:
            raise ValueError("No pathways have genes present in expression data. Check gene symbol format.")

        # Run AUCell scoring
        auc_scores = aucell(
            expr_df, gene_signatures, auc_threshold=0.05, noweights=True, normalize=False, num_workers=1
        )

        if verbose:
            print(f"      AUCell result: {auc_scores.shape[0]} cells × {auc_scores.shape[1]} signatures")

        pathway_scores = auc_scores.T

        if verbose:
            print(f"   ✓ AUCell completed: {pathway_scores.shape[0]} pathways × {pathway_scores.shape[1]} cells")
            print(f"      Score range: {pathway_scores.values.min():.3f} to {pathway_scores.values.max():.3f}")

    except Exception as e:
        raise RuntimeError(f"Pathway scoring failed: {e}")

    # Convert to DataFrame (cells as rows)
    pathway_scores_df = pathway_scores.T

    # Ensure alignment with adata.obs.index
    if not pathway_scores_df.index.equals(adata.obs.index):
        if verbose:
            print("   🔄 Realigning pathway scores with adata.obs.index...")
        pathway_scores_df = pathway_scores_df.reindex(adata.obs.index, fill_value=0.0)

    if verbose:
        print("✓ Pathway scoring completed!")
        print(f"   Scores shape: {pathway_scores_df.shape}")
        print(f"   Pathways scored: {pathway_scores_df.shape[1]}")
        print(f"   Score range: [{pathway_scores_df.values.min():.3f}, {pathway_scores_df.values.max():.3f}]")

        print("\n📊 Pathway Score Statistics:")
        print(f"   Mean activity: {pathway_scores_df.mean().mean():.3f}")
        print(f"   Std activity: {pathway_scores_df.std().mean():.3f}")

        pathway_vars = pathway_scores_df.var().sort_values(ascending=False)
        print("\n   Most variable pathways:")
        for i, (pathway, var) in enumerate(pathway_vars.head(5).items()):
            print(f"      {i + 1}. {pathway}: variance = {var:.3f}")

    # Store in AnnData.obsm
    try:
        pathway_matrix = pathway_scores_df.values
        pathway_names = list(pathway_scores_df.columns)

        adata.obsm[obsm_key] = pathway_matrix
        adata.uns[f"{obsm_key}_pathways"] = pathway_names
        adata.uns[f"{obsm_key}_method"] = "aucell"

        if verbose:
            print("   📦 Stored in AnnData:")
            print(f"      adata.obsm['{obsm_key}']: {pathway_matrix.shape}")
            print(f"      adata.uns['{obsm_key}_pathways']: {len(pathway_names)} names")
            print(f"      adata.uns['{obsm_key}_method']: 'aucell'")

    except Exception as e:
        if verbose:
            print(f"   ⚠️ Warning: Failed to store in adata.obsm: {e}")

    return pathway_scores_df

#!/usr/bin/env python
"""
WORKFLOW 05: Gene & Pathway Enrichment Analysis
================================================

This workflow demonstrates how to identify genes and pathways associated with each archetype:
1. Compute gene associations (differential expression per archetype)
2. Compute pathway associations (enrichment analysis)

The gene_associations function returns a DataFrame with columns:
- gene: Gene identifier
- archetype: Which archetype (archetype_0, archetype_1, ...)
- n_archetype_cells: Number of cells in this archetype
- n_other_cells: Number of cells in other archetypes
- mean_archetype: Mean expression in archetype cells
- mean_other: Mean expression in other cells
- log_fold_change: Log fold change (archetype vs others)
- statistic: Test statistic
- pvalue: Raw p-value
- fdr_pvalue: FDR-adjusted p-value
- significant: Boolean significance flag

Example usage:
    python WORKFLOW_05.py

Requirements:
    - peach
    - scanpy
    - Trained model with archetype assignments (from WORKFLOW_04)
"""

import scanpy as sc
import peach as pc
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

# Data path
data_path = Path("data/HSC.h5ad")

# Training parameters (from WORKFLOW_03-04)
n_archetypes = 5
hidden_dims = [256, 128, 64]
max_epochs = 50
random_state = 42

# Gene association parameters
top_n_genes = 50  # Number of top genes to display per archetype
p_value_threshold = 0.05  # Significance threshold

# =============================================================================
# Step 1: Prepare Data with Model and Assignments (Prerequisites)
# =============================================================================

print("Preparing data (loading, training, assigning)...")
adata = sc.read_h5ad(data_path)
print(f"  Shape: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

# Ensure PCA exists
if 'X_pca' not in adata.obsm:
    print("  Running PCA...")
    sc.tl.pca(adata, n_comps=13)

# Train model
print(f"  Training model ({n_archetypes} archetypes)...")
pc.tl.train_archetypal(
    adata,
    n_archetypes=n_archetypes,
    n_epochs=max_epochs,
    model_config={
        'hidden_dims': hidden_dims,
    },
    seed=random_state,
    device='cpu',
)

# Compute coordinates and assign cells to archetypes
print("  Computing coordinates and assigning cells to archetypes...")
pc.tl.archetypal_coordinates(adata)
pc.tl.assign_archetypes(adata, percentage_per_archetype=0.15)
print(f"  Assignments created: adata.obs['archetypes']")

# =============================================================================
# Step 2: Compute Gene Associations
# =============================================================================

print("\nComputing gene associations...")
print("  (Differential expression per archetype)")

gene_assoc = pc.tl.gene_associations(
    adata,
    obs_key='archetypes',  # Use archetype assignments
    fdr_scope='global',
    verbose=False,
)

print(f"\n  Results DataFrame shape: {gene_assoc.shape}")
print(f"  Columns: {list(gene_assoc.columns)}")

# Show top genes for each archetype
print(f"\nTop {top_n_genes} genes per archetype:")
for archetype in sorted(gene_assoc['archetype'].unique()):
    arch_genes = gene_assoc[gene_assoc['archetype'] == archetype]

    # Filter by significance (using actual column names)
    sig_genes = arch_genes[arch_genes['fdr_pvalue'] < p_value_threshold]

    # Sort by fold change (using actual column name)
    top_genes = sig_genes.nlargest(5, 'log_fold_change')

    print(f"\n  Archetype {archetype} ({len(sig_genes)} significant genes):")
    if len(top_genes) > 0:
        for idx, row in top_genes.iterrows():
            gene = row.get('gene', 'unknown')
            fc = row['log_fold_change']
            pval = row['fdr_pvalue']
            print(f"    {gene}: logFC={fc:.2f}, fdr_p={pval:.2e}")
    else:
        print(f"    No significant genes found")

# =============================================================================
# Step 3: Compute Pathway Associations (Optional)
# =============================================================================

print("\nComputing pathway associations (if pathway data available)...")

# Note: This requires pathway networks to be loaded first
# For demonstration, we'll show the function call pattern

try:
    # Load pathway networks (example: MSigDB or GO terms)
    # pc.pp.load_pathway_networks(adata, pathway_database='GO_BP')

    # Compute pathway associations
    # pathway_assoc = pc.tl.pathway_associations(
    #     adata,
    #     groupby='archetypes',
    #     method='gsea',  # or 'ora' for over-representation
    # )

    # For now, just note the pattern
    print("  Pathway analysis requires pathway database to be loaded")
    print("  See docs for pc.pp.load_pathway_networks()")
    print("  Then use pc.tl.pathway_associations()")

except Exception as e:
    print(f"  Pathway analysis skipped: {e}")

# =============================================================================
# Step 4: Export Results
# =============================================================================

print("\nExporting results...")

# Save gene associations to CSV
output_file = "gene_associations.csv"
gene_assoc.to_csv(output_file, index=False)
print(f"  Saved: {output_file}")

# Summary statistics
print(f"\nSummary statistics:")
print(f"  Total genes tested: {gene_assoc['gene'].nunique()}")
print(f"  Archetypes analyzed: {gene_assoc['archetype'].nunique()}")
print(f"  Significant associations (p < {p_value_threshold}): {(gene_assoc['fdr_pvalue'] < p_value_threshold).sum()}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*70)
print("WORKFLOW 05 COMPLETE")
print("="*70)
print(f"Gene associations computed:")
print(f"  • DataFrame with {len(gene_assoc):,} gene-archetype associations")
print(f"  • Columns: gene, archetype, log_fold_change, pvalue, fdr_pvalue, significant")
print(f"  • Exported to: {output_file}")
print(f"\nNext workflows:")
print(f"  • WORKFLOW_06: CellRank Integration (requires RNA velocity)")
print(f"  • WORKFLOW_08: Comprehensive Visualization")
print("="*70)

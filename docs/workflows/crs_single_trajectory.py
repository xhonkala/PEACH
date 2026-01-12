#!/usr/bin/env python
"""
WORKFLOW 06.1: Single Trajectory Analysis
==========================================

This workflow demonstrates focused analysis of a single archetype-to-archetype
trajectory using PEACH's single_trajectory_analysis() with real CellRank integration.

Key Steps:
1. Data Setup: Load data, prepare archetype weights
2. CellRank Setup: Run setup_cellrank() to compute real fate probabilities
3. Pseudotime: Compute lineage pseudotimes via CellRank
4. Single Trajectory: Filter cells and create subset AnnData
5. Driver Genes: Use CellRank's compute_lineage_drivers()
6. Gene Trends: Plot expression along trajectory with cr.pl.gene_trends()

Test Case: Trajectory from archetype_1 → archetype_2

Requirements:
    - peach
    - scanpy
    - cellrank
    - matplotlib

Example usage:
    python WORKFLOW_06.1_SINGLE_TRAJECTORY.py

3/3 scripts used in preparing the preprint at: https://www.biorxiv.org/content/10.64898/2025.12.29.696912v1
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script execution

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import peach as pc
import cellrank as cr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def main():
    """Main workflow function - wrapped for multiprocessing compatibility."""

    # =============================================================================
    # Configuration
    # =============================================================================

    # Data path
    data_path = Path("data/helsinki_fit.h5ad")
    output_dir = Path("tests")
    output_dir.mkdir(exist_ok=True)

    # Trajectory to analyze
    SOURCE_ARCHETYPE = 1
    TARGET_ARCHETYPE = 2

    # Thresholds
    SOURCE_WEIGHT_THRESHOLD = 0.4  # For weight-based selection
    TARGET_FATE_THRESHOLD = 0.4    # Cells with >= 40% fate probability for target

    # CellRank settings
    HIGH_PURITY_THRESHOLD = 0.80   # For terminal state identification

    # =============================================================================
    # Step 1: Data Preparation
    # =============================================================================

    print("=" * 70)
    print("WORKFLOW 06.1: Single Trajectory Analysis (CellRank Integration)")
    print(f"Trajectory: archetype_{SOURCE_ARCHETYPE} → archetype_{TARGET_ARCHETYPE}")
    print("=" * 70)

    print("\nStep 1: Data Preparation")
    print("-" * 70)

    # Load data
    adata = sc.read_h5ad(data_path)
    print(f"  Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # Ensure PCA exists with enough components for CellRank (default n_pcs=11)
    pca_key = 'X_pca' if 'X_pca' in adata.obsm else 'X_PCA'
    n_pcs_available = adata.obsm[pca_key].shape[1] if pca_key in adata.obsm else 0
    print(f"  PCA key: {pca_key}, N PCs available: {n_pcs_available}")

    # Rerun PCA if not enough components for CellRank (needs at least 11 for default n_pcs)
    MIN_PCS_FOR_CELLRANK = 30
    if n_pcs_available < MIN_PCS_FOR_CELLRANK:
        print(f"  Recomputing PCA with {MIN_PCS_FOR_CELLRANK} components for CellRank...")
        sc.pp.pca(adata, n_comps=MIN_PCS_FOR_CELLRANK, svd_solver='arpack')
        print(f"  New PCA shape: {adata.obsm['X_pca'].shape}")

    # Get number of archetypes from existing data
    n_archetypes = adata.obsm['archetype_distances'].shape[1]
    print(f"  N archetypes: {n_archetypes}")

    # Compute archetype weights from distances if not present
    if 'cell_archetype_weights' not in adata.obsm:
        print("  Computing archetype weights from distances...")
        distances = adata.obsm['archetype_distances']
        weights = np.exp(-distances)
        weights = weights / weights.sum(axis=1, keepdims=True)
        adata.obsm['cell_archetype_weights'] = weights
    weights = adata.obsm['cell_archetype_weights']
    print(f"  Weights shape: {weights.shape}")

    # Ensure archetypes assignment exists (for discrete selection)
    if 'archetypes' not in adata.obs:
        print("  Computing archetype assignments...")
        pc.tl.assign_archetypes(adata)
    print(f"  Archetype distribution: {adata.obs['archetypes'].value_counts().to_dict()}")

    # =============================================================================
    # Step 2: CellRank Setup (Real Fate Probabilities)
    # =============================================================================

    print("\nStep 2: CellRank Setup")
    print("-" * 70)

    print("  Running setup_cellrank() with ConnectivityKernel...")
    print("  (This computes neighbors, UMAP, PAGA, and fate probabilities)")

    # This is the key step - computes REAL fate probabilities via GPCCA
    ck, g = pc.tl.setup_cellrank(adata, high_purity_threshold=HIGH_PURITY_THRESHOLD, verbose=True)

    # Store GPCCA object for later use with driver genes
    adata.uns['cellrank_gpcca'] = g

    print(f"\n  Fate probabilities shape: {adata.obsm['fate_probabilities'].shape}")
    print(f"  Lineage names: {adata.uns['lineage_names']}")

    # =============================================================================
    # Step 3: Compute Lineage Pseudotimes
    # =============================================================================

    print("\nStep 3: Compute Lineage Pseudotimes")
    print("-" * 70)

    pc.tl.compute_lineage_pseudotimes(adata)

    # Show available pseudotime keys
    pt_keys = [k for k in adata.obs.columns if k.startswith('pseudotime_to_')]
    print(f"  Created pseudotime columns: {pt_keys}")

    target_archetype = f'archetype_{TARGET_ARCHETYPE}'
    pt_key = f'pseudotime_to_{target_archetype}'
    pt_stats = adata.obs[pt_key].describe()
    print(f"\n  {pt_key} statistics:")
    print(f"    Min: {pt_stats['min']:.3f}")
    print(f"    Max: {pt_stats['max']:.3f}")
    print(f"    Mean: {pt_stats['mean']:.3f}")

    # =============================================================================
    # Step 4: Single Trajectory Analysis
    # =============================================================================

    print("\nStep 4: Single Trajectory Analysis")
    print("-" * 70)

    print(f"\n  Analyzing trajectory: archetype_{SOURCE_ARCHETYPE} → archetype_{TARGET_ARCHETYPE}")
    print("  Using selection_method='both' to compare discrete vs weight-based")

    # Run with 'both' to see comparison
    result, adata_traj = pc.tl.single_trajectory_analysis(
        adata,
        trajectory=(SOURCE_ARCHETYPE, TARGET_ARCHETYPE),
        selection_method='both',  # Compare discrete vs weight-based
        source_weight_threshold=SOURCE_WEIGHT_THRESHOLD,
        target_fate_threshold=TARGET_FATE_THRESHOLD,
        verbose=True
    )

    print(f"\n  Result Summary:")
    print(f"    Source archetype: {result.source_archetype}")
    print(f"    Target archetype: {result.target_archetype}")
    print(f"    Selection method used: {result.selection_method}")
    print(f"    Discrete selection cells: {result.n_discrete_cells}")
    print(f"    Weight selection cells: {result.n_weight_cells}")
    print(f"    Final trajectory cells: {result.n_trajectory_cells}")
    print(f"    Pseudotime key: {result.pseudotime_key}")

    # Show subset info
    print(f"\n  Subset AnnData:")
    print(f"    Cells: {adata_traj.n_obs}")
    print(f"    Genes: {adata_traj.n_vars}")
    print(f"    Trajectory info: {adata_traj.uns.get('trajectory_info', {})}")

    # =============================================================================
    # Step 5: Driver Genes via CellRank
    # =============================================================================

    print("\nStep 5: Driver Genes via CellRank")
    print("-" * 70)

    print(f"\n  Computing drivers for lineage: {target_archetype}")
    print("  Using g.compute_lineage_drivers() from CellRank GPCCA...")

    # Use CellRank's driver computation directly
    drivers_df = None
    top_genes = None
    try:
        drivers_df = g.compute_lineage_drivers(lineages=target_archetype)

        print(f"\n  Found {len(drivers_df)} driver genes")
        print(f"\n  Top 10 Driver Genes for {target_archetype}:")

        # Display top drivers
        top_drivers = drivers_df.head(10)
        for i, (gene, row) in enumerate(top_drivers.iterrows()):
            corr_col = f'{target_archetype}_corr'
            corr = row.get(corr_col, row.iloc[0]) if corr_col in row.index else row.iloc[0]
            print(f"    {i+1}. {gene}: corr = {corr:.3f}")

        # Get top 5 genes for visualization
        top_genes = drivers_df.index[:5].tolist()
        print(f"\n  Selected for visualization: {top_genes}")

    except Exception as e:
        print(f"  Warning: Driver computation failed: {e}")

    # Save drivers to CSV
    if drivers_df is not None:
        drivers_output = output_dir / f'trajectory_{SOURCE_ARCHETYPE}_to_{TARGET_ARCHETYPE}_drivers.csv'
        drivers_df.to_csv(drivers_output)
        print(f"  Saved drivers: {drivers_output}")

    # =============================================================================
    # Step 6: Gene Trends Visualization
    # =============================================================================

    print("\nStep 6: Gene Trends Visualization")
    print("-" * 70)

    trends_output = None
    if top_genes and adata_traj.n_obs > 10:
        print(f"\n  Plotting gene trends for: {top_genes}")
        print(f"  Using subset AnnData with {adata_traj.n_obs} trajectory cells")
        print(f"  Pseudotime key: {result.pseudotime_key}")

        try:
            # Use CellRank's gene_trends with the trajectory subset
            fig = cr.pl.gene_trends(
                adata_traj,
                model=cr.models.GAMR(adata_traj),
                genes=top_genes,
                time_key=result.pseudotime_key,
                ncols=3,
                figsize=(15, 5),
                return_figure=True
            )

            trends_output = output_dir / f'trajectory_{SOURCE_ARCHETYPE}_to_{TARGET_ARCHETYPE}_gene_trends.png'
            if fig is not None:
                fig.savefig(trends_output, dpi=150, bbox_inches='tight')
                print(f"  Saved: {trends_output}")
            else:
                plt.savefig(trends_output, dpi=150, bbox_inches='tight')
                print(f"  Saved: {trends_output}")
            plt.close('all')

        except Exception as e:
            print(f"  Warning: Gene trends failed: {e}")
            print("  Falling back to simple scatter plot...")

            # Fallback: simple scatter plot
            fig, axes = plt.subplots(1, len(top_genes), figsize=(4*len(top_genes), 4))
            if len(top_genes) == 1:
                axes = [axes]

            pseudotime = adata_traj.obs[result.pseudotime_key].values

            for ax, gene in zip(axes, top_genes):
                if gene in adata_traj.var_names:
                    gene_idx = adata_traj.var_names.get_loc(gene)
                    if hasattr(adata_traj.X, 'toarray'):
                        expr = adata_traj.X[:, gene_idx].toarray().flatten()
                    else:
                        expr = adata_traj.X[:, gene_idx].flatten()

                    ax.scatter(pseudotime, expr, alpha=0.5, s=10, c='steelblue')
                    ax.set_xlabel('Pseudotime')
                    ax.set_ylabel('Expression')
                    ax.set_title(gene)

            plt.suptitle(f'Gene Expression: {result.source_archetype} → {result.target_archetype}')
            plt.tight_layout()

            trends_output = output_dir / f'trajectory_{SOURCE_ARCHETYPE}_to_{TARGET_ARCHETYPE}_gene_trends.png'
            plt.savefig(trends_output, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved fallback: {trends_output}")

    else:
        print("  Skipping gene trends (not enough cells or no drivers)")

    # =============================================================================
    # Step 7: Trajectory Visualization in Archetypal Space
    # =============================================================================

    print("\nStep 7: Trajectory Visualization")
    print("-" * 70)

    # 7a: Color all cells by trajectory pseudotime
    print("\n  7a: Archetypal space colored by trajectory pseudotime...")

    # Create pseudotime column for all cells (NaN for non-trajectory)
    pt_col = f'trajectory_{SOURCE_ARCHETYPE}_to_{TARGET_ARCHETYPE}_pseudotime'
    adata.obs[pt_col] = np.nan
    adata.obs.loc[adata.obs[result.cell_mask_key], pt_col] = \
        adata.obs.loc[adata.obs[result.cell_mask_key], result.pseudotime_key]

    fig_pt = pc.pl.archetypal_space(
        adata,
        color_by=pt_col,
        color_scale='viridis',
        title=f'Trajectory Pseudotime: {result.source_archetype} → {result.target_archetype}',
        cell_opacity=0.7
    )
    pt_output = output_dir / f'trajectory_{SOURCE_ARCHETYPE}_to_{TARGET_ARCHETYPE}_pseudotime.html'
    fig_pt.write_html(str(pt_output))
    print(f"      Saved: {pt_output}")

    # 7b: Highlight trajectory cells
    print("\n  7b: Trajectory cells highlighted...")
    adata.obs['trajectory_membership'] = 'Other cells'
    adata.obs.loc[adata.obs[result.cell_mask_key], 'trajectory_membership'] = 'Trajectory cells'

    fig_mask = pc.pl.archetypal_space(
        adata,
        color_by='trajectory_membership',
        categorical_colors={'Trajectory cells': 'red', 'Other cells': 'lightgray'},
        title=f'Cells in Trajectory: {result.source_archetype} → {result.target_archetype}',
        cell_opacity=0.6
    )
    mask_output = output_dir / f'trajectory_{SOURCE_ARCHETYPE}_to_{TARGET_ARCHETYPE}_cells.html'
    fig_mask.write_html(str(mask_output))
    print(f"      Saved: {mask_output}")

    # =============================================================================
    # Step 8: Statistics Summary
    # =============================================================================

    print("\nStep 8: Statistics Summary")
    print("-" * 70)

    import pandas as pd

    # Get pseudotime stats for trajectory cells
    pt_traj = adata_traj.obs[result.pseudotime_key]

    summary_data = {
        'Metric': [
            'Source Archetype',
            'Target Archetype',
            'Selection Method',
            'Source Weight Threshold',
            'Target Fate Threshold',
            'Total Cells in Dataset',
            'Discrete Selection Cells',
            'Weight Selection Cells',
            'Final Trajectory Cells',
            'Pseudotime Min',
            'Pseudotime Max',
            'Pseudotime Mean',
            'N Driver Genes'
        ],
        'Value': [
            result.source_archetype,
            result.target_archetype,
            result.selection_method,
            f'{SOURCE_WEIGHT_THRESHOLD:.2f}',
            f'{TARGET_FATE_THRESHOLD:.2f}',
            str(adata.n_obs),
            str(result.n_discrete_cells),
            str(result.n_weight_cells),
            str(result.n_trajectory_cells),
            f'{pt_traj.min():.3f}',
            f'{pt_traj.max():.3f}',
            f'{pt_traj.mean():.3f}',
            str(len(drivers_df) if drivers_df is not None else 0)
        ]
    }

    print("\n  Trajectory Statistics:")
    for metric, value in zip(summary_data['Metric'], summary_data['Value']):
        print(f"    {metric}: {value}")

    # Save statistics
    stats_df = pd.DataFrame(summary_data)
    stats_output = output_dir / f'trajectory_{SOURCE_ARCHETYPE}_to_{TARGET_ARCHETYPE}_stats.csv'
    stats_df.to_csv(stats_output, index=False)
    print(f"\n  Saved statistics: {stats_output}")

    # =============================================================================
    # Summary
    # =============================================================================

    print("\n" + "=" * 70)
    print("WORKFLOW 06.1 COMPLETE")
    print("=" * 70)

    print("\nOutput Files:")
    print(f"  - {pt_output}")
    print(f"  - {mask_output}")
    if trends_output:
        print(f"  - {trends_output}")
    if drivers_df is not None:
        print(f"  - {output_dir / f'trajectory_{SOURCE_ARCHETYPE}_to_{TARGET_ARCHETYPE}_drivers.csv'}")
    print(f"  - {stats_output}")

    print("\nKey Results:")
    print(f"  - Trajectory: {result.source_archetype} → {result.target_archetype}")
    print(f"  - Cells in trajectory: {result.n_trajectory_cells}")
    print(f"  - Selection comparison: discrete={result.n_discrete_cells}, weight={result.n_weight_cells}")
    print(f"  - Pseudotime range: [{pt_traj.min():.3f}, {pt_traj.max():.3f}]")
    if top_genes:
        print(f"  - Top driver gene: {top_genes[0]}")

    print("\nUsage Pattern for Custom Analysis:")
    print("""
  # 1. Setup CellRank
  ck, g = pc.tl.setup_cellrank(adata, high_purity_threshold=0.80)
  pc.tl.compute_lineage_pseudotimes(adata)

  # 2. Analyze trajectory (returns subset AnnData)
  result, adata_traj = pc.tl.single_trajectory_analysis(
      adata, trajectory=(src, tgt), selection_method='discrete'
  )

  # 3. Get drivers from CellRank
  drivers = g.compute_lineage_drivers(lineages='archetype_{tgt}')

  # 4. Plot gene trends with subset
  cr.pl.gene_trends(
      adata_traj,
      model=cr.models.GAMR(adata_traj),
      genes=drivers.index[:5].tolist(),
      time_key=result.pseudotime_key
  )
""")
    print("=" * 70)


if __name__ == '__main__':
    # Required for CellRank multiprocessing on macOS
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)
    main()

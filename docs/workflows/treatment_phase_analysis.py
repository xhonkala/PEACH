#!/usr/bin/env python3
"""
Treatment Phase Archetypal Analysis Workflow
=============================================

Complete PEACH workflow for analyzing treatment phase transitions in archetypal space.

This workflow:
1. Loads pre-processed data (with PCA)
2. Runs hyperparameter search (k=3-5 archetypes)
3. Trains final model with best configuration
4. Computes full archetypal analysis (coordinates, weights, distances, assignments)
5. Runs pathway scoring and statistical associations
6. Generates treatment phase trajectory visualizations with CRS stratification

Inputs:
    - Pre-processed .h5ad with PCA computed
    - Pathway database CSV (source, target, weight, pathway columns)

Outputs:
    - Trained model (.pt)
    - Processed AnnData (.h5ad)
    - Training metrics (.csv, .png)
    - Association results (.csv, .png per analysis type)
    - Archetypal space visualizations (.png)

Usage:
    python treatment_phase_analysis.py \\
        --input_adata /path/to/data.h5ad \\
        --output_dir /path/to/output \\
        --pathway_file /path/to/pathways.csv

1/2 scripts used in preparing the preprint at: https://www.biorxiv.org/content/10.64898/2025.12.29.696912v1
"""

import os
import sys
import argparse
import warnings
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import torch
import traceback

import peach as pc
from peach._core.utils.hyperparameter_search import ArchetypalGridSearch
from peach.tl import SearchConfig


# =============================================================================
# CONFIGURATION
# =============================================================================

# Hyperparameter search ranges
N_ARCHETYPES_RANGE = [3, 4, 5]
HIDDEN_DIMS_OPTIONS = [
    [128, 64],
    [256, 128, 64],
    [512, 256, 128],
]
INFLATION_FACTOR_RANGE = [1.0, 1.25, 1.5, 1.75, 2.0]

# Training settings
CV_FOLDS = 3
MAX_EPOCHS_CV = 30
MAX_EPOCHS_FINAL = 200
LEARNING_RATE = 0.001
BATCH_SIZE = 256

# Association testing settings
BIN_PROP = 0.15  # Top 15% cells per archetype
FDR_THRESHOLD = 0.05
MIN_LOGFC = 0.5


# =============================================================================
# DATA VALIDATION UTILITIES
# =============================================================================

def detect_pca_key(adata) -> str:
    """Detect PCA key in adata.obsm (handles X_pca vs X_PCA vs PCA variants)."""
    for key in ['X_pca', 'X_PCA', 'PCA', 'pca']:
        if key in adata.obsm:
            print(f"  Found PCA coordinates: adata.obsm['{key}']")
            return key
    raise KeyError("No PCA coordinates found. Expected 'X_pca', 'X_PCA', 'PCA', or 'pca' in adata.obsm")


def ensure_logcounts_layer(adata) -> str:
    """Ensure a suitable normalized layer exists for downstream assays.

    Returns:
        str: The layer key to use ('logcounts', 'scvi_normalized', or 'log1p')
    """
    # Check for existing normalized layers in order of preference
    preferred_layers = ['logcounts', 'scvi_normalized', 'log1p', 'normalized']
    for layer in preferred_layers:
        if layer in adata.layers:
            print(f"  Found normalized layer: adata.layers['{layer}']")
            return layer

    # Create logcounts from adata.X if no normalized layer exists
    print("  No normalized layer found. Creating logcounts layer from adata.X")
    adata.layers['logcounts'] = adata.X.copy()
    return 'logcounts'


def check_gene_names(adata) -> Tuple[str, Optional[str]]:
    """
    Check gene name format and find conversion column if needed.

    Returns:
        Tuple of (gene_format, conversion_column)
        - gene_format: 'hgnc', 'ensembl', or 'unknown'
        - conversion_column: column name in adata.var with HGNC symbols, or None
    """
    gene_names = adata.var_names.tolist()
    sample = gene_names[:100]  # Check first 100 genes

    # Check for Ensembl IDs (ENSG...)
    ensembl_count = sum(1 for g in sample if str(g).startswith('ENSG'))
    ensembl_fraction = ensembl_count / len(sample)

    if ensembl_fraction > 0.5:
        print(f"  Gene names appear to be Ensembl IDs ({ensembl_fraction:.0%} start with ENSG)")

        # Look for conversion column in adata.var
        possible_cols = ['gene_symbol', 'gene_name', 'symbol', 'hgnc_symbol',
                        'external_gene_name', 'gene_short_name', 'SYMBOL']
        for col in possible_cols:
            if col in adata.var.columns:
                # Verify it contains different values (not just copying index)
                unique_vals = adata.var[col].nunique()
                if unique_vals > 100:  # Reasonable number of unique gene symbols
                    print(f"  Found gene symbol conversion column: adata.var['{col}']")
                    return 'ensembl', col

        warnings.warn("Ensembl IDs detected but no conversion column found in adata.var. "
                     "Gene-level analyses may have limited interpretability.")
        return 'ensembl', None

    # Assume HGNC if not Ensembl
    print("  Gene names appear to be HGNC symbols")
    return 'hgnc', None


def detect_obs_column(adata, candidates: List[str], description: str) -> str:
    """Detect which candidate column exists in adata.obs."""
    for col in candidates:
        if col in adata.obs.columns:
            unique_vals = adata.obs[col].unique().tolist()
            print(f"  Found {description}: adata.obs['{col}'] with values: {unique_vals}")
            return col
    raise KeyError(f"None of {candidates} found in adata.obs for {description}")


def validate_input_data(adata) -> Dict[str, Any]:
    """
    Comprehensive input data validation.

    Returns dict with:
        - pca_key: str
        - gene_format: str
        - gene_symbol_col: Optional[str]
        - condition_col: str (patient/sample identifier for conditional analysis)
        - treatment_col: str (treatment_stage or treatment_phase)
        - crs_col: str (always 'CRS')
    """
    print("\n" + "="*70)
    print("VALIDATING INPUT DATA")
    print("="*70)

    validation = {}

    # 1. PCA coordinates
    validation['pca_key'] = detect_pca_key(adata)

    # 2. Normalized layer for gene expression analysis
    validation['expression_layer'] = ensure_logcounts_layer(adata)

    # 3. Gene names
    gene_format, symbol_col = check_gene_names(adata)
    validation['gene_format'] = gene_format
    validation['gene_symbol_col'] = symbol_col

    # 4. Patient/sample column for conditional associations
    # (tissue site not available in this dataset subset)
    validation['condition_col'] = detect_obs_column(
        adata,
        ['publication_patient_code_final', 'patient_id', 'sample_id',
         'patient', 'sample', 'donor_id'],
        'patient/sample identifier'
    )

    # 5. Treatment column (treatment_stage or treatment_phase)
    validation['treatment_col'] = detect_obs_column(
        adata,
        ['treatment_stage', 'treatment_phase'],
        'treatment phase'
    )

    # 6. CRS column (required for stratified trajectories)
    if 'CRS' not in adata.obs.columns:
        raise KeyError("'CRS' column not found in adata.obs")
    # Get non-null unique values
    crs_series = adata.obs['CRS'].dropna()
    crs_vals = crs_series.unique().tolist()
    # Filter out string representations of NA
    crs_vals = [v for v in crs_vals if v not in ['<NA>', 'NA', 'nan', None, '']]
    if len(crs_vals) < 2:
        raise ValueError(f"CRS column must have at least 2 valid values, found: {crs_vals}")
    print(f"  Found CRS column with values: {crs_vals}")
    validation['crs_col'] = 'CRS'
    validation['crs_values'] = crs_vals

    print("\n  Data validation complete")
    print("="*70 + "\n")

    return validation


# =============================================================================
# HYPERPARAMETER SEARCH
# =============================================================================

def run_hyperparameter_search(
    adata,
    pca_key: str,
    output_dir: str,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Run hyperparameter search to find optimal model configuration.

    Returns:
        Best configuration dict with n_archetypes, hidden_dims, inflation_factor
    """
    print("\n" + "="*70)
    print("HYPERPARAMETER SEARCH")
    print("="*70)

    # Create dataloader
    dataloader = pc.pp.prepare_training(adata, batch_size=BATCH_SIZE, pca_key=pca_key)

    # Get input dimensions
    n_pcs = adata.obsm[pca_key].shape[1]
    print(f"  Input dimensions: {n_pcs} PCs")
    print(f"  Archetype range: {N_ARCHETYPES_RANGE}")
    print(f"  Hidden dims options: {HIDDEN_DIMS_OPTIONS}")
    print(f"  Inflation factors: {INFLATION_FACTOR_RANGE}")

    # Configure search
    search_config = SearchConfig(
        n_archetypes_range=N_ARCHETYPES_RANGE,
        hidden_dims_options=HIDDEN_DIMS_OPTIONS,
        inflation_factor_range=INFLATION_FACTOR_RANGE,
        cv_folds=CV_FOLDS,
        max_epochs_cv=MAX_EPOCHS_CV,
    )

    # Base model config
    base_model_config = {
        'hidden_dims': [256, 128, 64],
        'use_barycentric': True,
        'use_hidden_transform': True,
        'diversity_weight': 0.05,
        'initialize_with_pcha': True,
        'device': device,
    }

    # Run grid search
    print("\nRunning grid search...")
    grid_search = ArchetypalGridSearch(search_config)
    cv_summary = grid_search.fit(
        dataloader=dataloader,
        base_model_config=base_model_config
    )

    # Get best configuration
    summary_df = cv_summary.summary_df
    top_configs = summary_df.sort_values('mean_archetype_r2', ascending=False)
    best_config = top_configs.iloc[0]

    # Extract hyperparameters
    best_params = {
        'n_archetypes': int(best_config['n_archetypes']),
        'hidden_dims': eval(best_config['hidden_dims']),  # String to list
        'inflation_factor': float(best_config.get('inflation_factor', 1.0)),
        'mean_r2': float(best_config['mean_archetype_r2']),
    }

    print(f"\n  Best configuration:")
    print(f"    n_archetypes: {best_params['n_archetypes']}")
    print(f"    hidden_dims: {best_params['hidden_dims']}")
    print(f"    inflation_factor: {best_params['inflation_factor']}")
    print(f"    CV R²: {best_params['mean_r2']:.4f}")

    # Save CV results
    cv_results_path = os.path.join(output_dir, 'cv_results.csv')
    top_configs.to_csv(cv_results_path, index=False)
    print(f"\n  Saved CV results to: {cv_results_path}")

    # Generate elbow plot
    try:
        fig = pc.pl.elbow_curve(cv_summary, metrics=['archetype_r2'])
        elbow_path = os.path.join(output_dir, 'elbow_curve.png')
        fig.write_image(elbow_path, scale=2)  # Plotly figure
        print(f"  Saved elbow curve to: {elbow_path}")
    except Exception as e:
        print(f"  Warning: Could not generate elbow curve: {e}")

    print("="*70 + "\n")

    return best_params


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_final_model(
    adata,
    best_params: Dict[str, Any],
    pca_key: str,
    output_dir: str,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Train final model with best hyperparameters using pc.tl.train_archetypal().

    Returns:
        Training results dict
    """
    print("\n" + "="*70)
    print("TRAINING FINAL MODEL")
    print("="*70)

    print(f"  n_archetypes: {best_params['n_archetypes']}")
    print(f"  hidden_dims: {best_params['hidden_dims']}")
    print(f"  inflation_factor: {best_params['inflation_factor']}")
    print(f"  Training for {MAX_EPOCHS_FINAL} epochs on {device}...")

    # Model config with best hyperparameters
    model_config = {
        'hidden_dims': best_params['hidden_dims'],
        'inflation_factor': best_params['inflation_factor'],
        'use_barycentric': True,
        'use_hidden_transform': True,
    }

    # Train using pc.tl.train_archetypal() which handles PCHA initialization automatically
    results = pc.tl.train_archetypal(
        adata,
        n_archetypes=best_params['n_archetypes'],
        n_epochs=MAX_EPOCHS_FINAL,
        pca_key=pca_key,
        model_config=model_config,
        device=device,
        archetypal_weight=0.95,
        diversity_weight=0.05,
        track_stability=True,
        validate_constraints=True,
        store_coords_key='archetype_coordinates',
        early_stopping=True,
        early_stopping_patience=15,
    )

    # Extract final metrics
    final_r2 = results['history'].get('archetype_r2', [0])[-1]
    final_loss = results['history'].get('loss', [0])[-1]

    print(f"\n  Training complete:")
    print(f"    Final R²: {final_r2:.4f}")
    print(f"    Final loss: {final_loss:.4f}")

    # Save model (get from results)
    n_pcs = adata.obsm[pca_key].shape[1]
    model = results.get('final_model')
    if model is not None:
        model_path = os.path.join(output_dir, 'model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'hyperparameters': best_params,
            'n_pcs': n_pcs,
        }, model_path)
        print(f"  Saved model to: {model_path}")

    # Save training history (handle unequal array lengths)
    history_dict = {
        k: v for k, v in results['history'].items()
        if isinstance(v, list) and len(v) > 0
    }
    if history_dict:
        # Find the most common length (usually the number of epochs)
        lengths = [len(v) for v in history_dict.values()]
        target_len = max(set(lengths), key=lengths.count)
        # Only include metrics with matching length
        history_dict = {k: v for k, v in history_dict.items() if len(v) == target_len}
        history_df = pd.DataFrame(history_dict)
        history_path = os.path.join(output_dir, 'training_history.csv')
        history_df.to_csv(history_path, index=False)
        print(f"  Saved training history to: {history_path}")
    else:
        print("  Warning: No training history to save")

    # Plot training metrics (returns plotly figure)
    try:
        fig = pc.pl.training_metrics(results['history'], display=False)
        if fig is not None:
            metrics_path = os.path.join(output_dir, 'training_metrics.png')
            fig.write_image(metrics_path, scale=2)
            print(f"  Saved training metrics plot to: {metrics_path}")
    except Exception as e:
        print(f"  Warning: Could not generate training metrics plot: {e}")

    print("="*70 + "\n")

    return results


# =============================================================================
# ARCHETYPAL ANALYSIS
# =============================================================================

def compute_archetypal_analysis(
    adata,
    training_results: Dict[str, Any],
    pca_key: str,
    output_dir: str
) -> None:
    """
    Compute full archetypal analysis: coordinates, weights, distances, assignments.
    """
    print("\n" + "="*70)
    print("COMPUTING ARCHETYPAL ANALYSIS")
    print("="*70)

    # 1. Archetypal coordinates (distances to archetypes)
    print("  Computing archetypal coordinates (distances)...")
    pc.tl.archetypal_coordinates(adata, pca_key=pca_key)

    # 2. Extract archetype weights (A matrix / barycentric coordinates)
    print("  Extracting archetype weights...")
    model = training_results.get('final_model')
    if model is not None:
        pc.tl.extract_archetype_weights(
            adata,
            model=model,
            pca_key=pca_key,
            weights_key='cell_archetype_weights'
        )

    # 3. Assign cells to archetypes
    print("  Assigning cells to archetypes...")
    pc.tl.assign_archetypes(adata, percentage_per_archetype=BIN_PROP)

    # Report assignments
    archetype_counts = adata.obs['archetypes'].value_counts()
    print(f"\n  Archetype assignments:")
    for arch, count in archetype_counts.items():
        pct = 100 * count / len(adata)
        print(f"    {arch}: {count:,} cells ({pct:.1f}%)")

    print("="*70 + "\n")


# =============================================================================
# PATHWAY SCORING
# =============================================================================

def compute_pathway_scores(
    adata,
    pathway_file: str,
    expression_layer: str = 'logcounts',
    gene_symbol_col: Optional[str] = None
) -> None:
    """
    Load pathways and compute pathway activity scores per cell.
    """
    print("\n" + "="*70)
    print("COMPUTING PATHWAY SCORES")
    print("="*70)

    # Load pathway database
    print(f"  Loading pathways from: {pathway_file}")
    pathway_df = pd.read_csv(pathway_file)

    print(f"  Pathway DataFrame: {len(pathway_df):,} rows")
    print(f"  Columns: {pathway_df.columns.tolist()}")
    print(f"  Unique pathways: {pathway_df['pathway'].nunique():,}")
    print(f"  Unique genes: {pathway_df['target'].nunique():,}")

    # Compute pathway scores using the network DataFrame directly
    # The function expects columns: source, target, weight, pathway
    print(f"\n  Computing pathway activity scores using layer '{expression_layer}'...")
    pc.pp.compute_pathway_scores(
        adata,
        net=pathway_df,
        use_layer=expression_layer,
        obsm_key='pathway_scores',
        verbose=True
    )

    # Store pathway names for downstream use
    pathway_names = pathway_df['pathway'].unique().tolist()
    adata.uns['pathway_scores_names'] = pathway_names

    print(f"  Stored scores in adata.obsm['pathway_scores']")
    print(f"  Shape: {adata.obsm['pathway_scores'].shape}")

    print("="*70 + "\n")


# =============================================================================
# STATISTICAL ASSOCIATIONS
# =============================================================================

def run_gene_associations(adata, output_dir: str, expression_layer: Optional[str] = None) -> pd.DataFrame:
    """Run gene-archetype association testing with dotplot."""
    print("\n" + "="*70)
    print("GENE ASSOCIATIONS")
    print("="*70)

    print("  Running gene association tests...")
    results = pc.tl.gene_associations(
        adata,
        bin_prop=BIN_PROP,
        use_layer=expression_layer,
        fdr_method='benjamini_hochberg',
        fdr_scope='global',
        min_logfc=MIN_LOGFC,
        verbose=True
    )

    # Filter significant
    sig_results = results[results['significant'] == True]
    print(f"\n  Total tests: {len(results):,}")
    print(f"  Significant (FDR < {FDR_THRESHOLD}): {len(sig_results):,}")

    # Save
    results_path = os.path.join(output_dir, 'gene_associations.csv')
    results.to_csv(results_path, index=False)
    print(f"  Saved to: {results_path}")

    # Dotplot - pass FULL results, let dotplot handle filtering
    if len(results) > 0:
        try:
            print(f"\n  Creating gene dotplot with {len(results)} results...")
            fig = pc.pl.dotplot(
                results,
                top_n_per_group=10,
                filter_zero_p=False,
                title='Gene-Archetype Associations'
            )
            if fig is not None:
                fig_path = os.path.join(output_dir, 'gene_associations_dotplot.png')
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved dotplot to: {fig_path}")
        except Exception as e:
            print(f"  Warning: Could not generate gene dotplot: {e}")
            traceback.print_exc()

    print("="*70 + "\n")
    return results


def run_pathway_associations(adata, output_dir: str) -> Optional[pd.DataFrame]:
    """Run pathway-archetype association testing with dotplot."""
    print("\n" + "="*70)
    print("PATHWAY ASSOCIATIONS")
    print("="*70)

    if 'pathway_scores' not in adata.obsm:
        print("  Skipping - no pathway_scores in adata.obsm")
        print("="*70 + "\n")
        return None

    print("  Running pathway association tests...")
    results = pc.tl.pathway_associations(
        adata,
        pathway_obsm_key='pathway_scores',
        fdr_method='benjamini_hochberg',
        fdr_scope='global',
        verbose=True
    )

    # Filter significant
    sig_results = results[results['significant'] == True]
    print(f"\n  Total tests: {len(results):,}")
    print(f"  Significant (FDR < {FDR_THRESHOLD}): {len(sig_results):,}")

    # Save
    results_path = os.path.join(output_dir, 'pathway_associations.csv')
    results.to_csv(results_path, index=False)
    print(f"  Saved to: {results_path}")

    # Dotplot - pass FULL results, let dotplot handle filtering
    # Use y_col='pathway' since pathway_associations returns 'pathway' column
    if len(results) > 0:
        try:
            print(f"\n  Creating pathway dotplot with {len(results)} results...")
            fig = pc.pl.dotplot(
                results,
                y_col='pathway',
                top_n_per_group=10,
                filter_zero_p=False,
                title='Pathway-Archetype Associations'
            )
            if fig is not None:
                fig_path = os.path.join(output_dir, 'pathway_associations_dotplot.png')
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved dotplot to: {fig_path}")
        except Exception as e:
            print(f"  Warning: Could not generate pathway dotplot: {e}")
            traceback.print_exc()

    print("="*70 + "\n")
    return results


def run_conditional_associations(
    adata,
    condition_col: str,
    output_dir: str
) -> Optional[pd.DataFrame]:
    """Run conditional (patient/sample) associations with hypergeometric test."""
    print("\n" + "="*70)
    print(f"CONDITIONAL ASSOCIATIONS ({condition_col})")
    print("="*70)

    print(f"  Running hypergeometric tests for '{condition_col}'...")
    try:
        results = pc.tl.conditional_associations(
            adata,
            obs_column=condition_col,
            obs_key='archetypes',
            test_method='hypergeometric',
            fdr_method='benjamini_hochberg'
        )

        # Filter significant
        sig_results = results[results['significant'] == True]
        print(f"\n  Total tests: {len(results):,}")
        print(f"  Significant (FDR < {FDR_THRESHOLD}): {len(sig_results):,}")

        # Save
        results_path = os.path.join(output_dir, 'conditional_associations.csv')
        results.to_csv(results_path, index=False)
        print(f"  Saved to: {results_path}")

        # Heatmap of odds ratios
        if len(results) > 0 and 'odds_ratio' in results.columns:
            try:
                fig, ax = plt.subplots(figsize=(12, 8))
                pivot = results.pivot(index='archetype', columns='condition', values='odds_ratio')
                sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdBu_r', center=1, ax=ax)
                ax.set_title(f'Archetype Enrichment by {condition_col}', fontsize=14, fontweight='bold')
                fig_path = os.path.join(output_dir, 'conditional_associations_heatmap.png')
                fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved heatmap to: {fig_path}")
            except Exception as e:
                print(f"  Warning: Could not generate heatmap: {e}")

        print("="*70 + "\n")
        return results

    except Exception as e:
        print(f"  Warning: Conditional association tests failed: {e}")
        traceback.print_exc()
        print("="*70 + "\n")
        return None


def run_pattern_analyses(adata, output_dir: str) -> Dict[str, pd.DataFrame]:
    """Run exclusivity, specialization, and tradeoff pattern analyses."""
    print("\n" + "="*70)
    print("PATTERN ANALYSES")
    print("="*70)

    if 'pathway_scores' not in adata.obsm:
        print("  Skipping - no pathway_scores in adata.obsm")
        print("="*70 + "\n")
        return {}

    results = {}

    # Pattern analysis settings
    PATTERN_MIN_EFFECT_SIZE = 0.05  # Lower threshold for pathway mean_diff

    # 1. Exclusivity patterns
    print("\n  Running exclusivity patterns...")
    try:
        exclusive_results = pc.tl.archetype_exclusive_patterns(
            adata,
            data_obsm_key='pathway_scores',
            min_effect_size=PATTERN_MIN_EFFECT_SIZE,
            fdr_scope='global',
            verbose=True
        )
        exclusive_results.to_csv(os.path.join(output_dir, 'exclusive_patterns.csv'), index=False)
        results['exclusive'] = exclusive_results
        print(f"    Found {len(exclusive_results)} exclusive patterns")

        # Dotplot for exclusivity
        if len(exclusive_results) > 0:
            try:
                fig = pc.pl.pattern_dotplot(
                    exclusive_results,
                    pattern_type='exclusive',
                    top_n=30,
                    min_effect_size=0.0,  # Don't filter again, already filtered
                    max_pvalue=1.0
                )
                if fig is not None:
                    fig.savefig(os.path.join(output_dir, 'exclusive_patterns_dotplot.png'),
                               dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print(f"    Saved exclusivity dotplot")
            except Exception as e:
                print(f"    Warning: Could not generate exclusivity dotplot: {e}")
    except Exception as e:
        print(f"    Warning: Exclusivity analysis failed: {e}")
        traceback.print_exc()

    # 2. Specialization patterns
    print("\n  Running specialization patterns...")
    try:
        spec_results = pc.tl.specialization_patterns(
            adata,
            data_obsm_key='pathway_scores',
            fdr_scope='per_archetype',
            verbose=True
        )
        spec_results.to_csv(os.path.join(output_dir, 'specialization_patterns.csv'), index=False)
        results['specialization'] = spec_results
        print(f"    Found {len(spec_results)} specialization patterns")

        # Dotplot for specialization
        if len(spec_results) > 0:
            try:
                fig = pc.pl.pattern_dotplot(
                    spec_results,
                    pattern_type='specialization',
                    top_n=30,
                    min_effect_size=0.0,
                    max_pvalue=1.0
                )
                if fig is not None:
                    fig.savefig(os.path.join(output_dir, 'specialization_patterns_dotplot.png'),
                               dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print(f"    Saved specialization dotplot")
            except Exception as e:
                print(f"    Warning: Could not generate specialization dotplot: {e}")
    except Exception as e:
        print(f"    Warning: Specialization analysis failed: {e}")
        traceback.print_exc()

    # 3. Tradeoff patterns
    print("\n  Running tradeoff patterns...")
    try:
        tradeoff_results = pc.tl.tradeoff_patterns(
            adata,
            data_obsm_key='pathway_scores',
            tradeoffs='pairs',
            min_effect_size=0.1,
            fdr_scope='global',
            verbose=True
        )
        tradeoff_results.to_csv(os.path.join(output_dir, 'tradeoff_patterns.csv'), index=False)
        results['tradeoff'] = tradeoff_results
        print(f"    Found {len(tradeoff_results)} tradeoff patterns")

        # Dotplot for tradeoffs
        if len(tradeoff_results) > 0:
            try:
                fig = pc.pl.pattern_dotplot(
                    tradeoff_results,
                    pattern_type='tradeoff',
                    top_n=30,
                    min_effect_size=0.0,
                    max_pvalue=1.0
                )
                if fig is not None:
                    fig.savefig(os.path.join(output_dir, 'tradeoff_patterns_dotplot.png'),
                               dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print(f"    Saved tradeoff dotplot")
            except Exception as e:
                print(f"    Warning: Could not generate tradeoff dotplot: {e}")
    except Exception as e:
        print(f"    Warning: Tradeoff analysis failed: {e}")
        traceback.print_exc()

    print("="*70 + "\n")
    return results


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def generate_archetypal_space_plots(
    adata,
    condition_col: str,
    pca_key: str,
    output_dir: str
) -> None:
    """Generate 3D archetypal space visualization colored by archetype."""
    print("\n" + "="*70)
    print("3D ARCHETYPAL SPACE VISUALIZATION")
    print("="*70)

    print("  Generating 3D plot colored by archetype...")
    try:
        fig = pc.pl.archetypal_space(
            adata,
            pca_key=pca_key,
            color_by='archetypes',
            title='Archetypal Space by Assignment'
        )
        if fig is not None:
            # Save as interactive HTML (avoids Kaleido/Chrome issues on HPC)
            html_path = os.path.join(output_dir, 'archetypal_space_by_archetype.html')
            fig.write_html(html_path)
            print(f"  Saved interactive plot to: {html_path}")
    except Exception as e:
        print(f"  Warning: Could not generate archetypal space plot: {e}")
        traceback.print_exc()

    print("="*70 + "\n")


def compute_treatment_centroids(
    adata,
    treatment_col: str,
    pca_key: str,
    output_dir: str,
    groupby_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compute treatment phase centroids, optionally stratified by a grouping variable.

    If groupby_col is provided (e.g., 'CRS'), generates separate trajectories per group.
    """
    print("\n" + "="*70)
    print("COMPUTING TREATMENT PHASE CENTROIDS")
    print("="*70)

    print(f"  Treatment column: {treatment_col}")
    if groupby_col:
        print(f"  Stratifying by: {groupby_col}")
    else:
        print("  No stratification (single trajectory)")

    # Compute centroids
    centroid_result = pc.tl.compute_conditional_centroids(
        adata,
        condition_column=treatment_col,
        pca_key=pca_key,
        store_key='conditional_centroids',
        exclude_archetypes=['no_archetype'],
        groupby=groupby_col,
        verbose=True
    )

    # Report results
    print(f"\n  Computed centroids:")
    print(f"    Treatment levels: {centroid_result['levels']}")

    if groupby_col and centroid_result.get('group_centroids'):
        print(f"    Groups: {list(centroid_result['group_centroids'].keys())}")
        for group_name, group_data in centroid_result['group_centroids'].items():
            print(f"\n    {group_name} centroids:")
            for level, coords in group_data.items():
                print(f"      {level}: [{coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f}]")
    else:
        # Non-grouped centroids
        for level, coords in centroid_result.get('centroids_3d', {}).items():
            print(f"    {level}: [{coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f}]")

    # Save centroid data
    centroid_df_rows = []
    if groupby_col and centroid_result.get('group_centroids_3d'):
        for group_name, group_data in centroid_result['group_centroids_3d'].items():
            for treatment_level, coords in group_data.items():
                centroid_df_rows.append({
                    'group': group_name,
                    'treatment': treatment_level,
                    'PC1': coords[0],
                    'PC2': coords[1],
                    'PC3': coords[2],
                    'n_cells': centroid_result['group_cell_counts'][group_name][treatment_level]
                })
    else:
        # Non-grouped centroids
        for treatment_level, coords in centroid_result.get('centroids_3d', {}).items():
            centroid_df_rows.append({
                'treatment': treatment_level,
                'PC1': coords[0],
                'PC2': coords[1],
                'PC3': coords[2],
                'n_cells': centroid_result['cell_counts'][treatment_level]
            })

    if centroid_df_rows:
        centroid_df = pd.DataFrame(centroid_df_rows)
        centroid_path = os.path.join(output_dir, 'treatment_centroids.csv')
        centroid_df.to_csv(centroid_path, index=False)
        print(f"\n  Saved centroids to: {centroid_path}")

    print("="*70 + "\n")

    return centroid_result


def plot_treatment_trajectories(
    adata,
    centroid_result,
    treatment_col: str,
    pca_key: str,
    output_dir: str,
    groupby_col: Optional[str] = None,
    group_colors: Optional[Dict[str, str]] = None
) -> None:
    """
    Plot archetypal space with treatment phase trajectories.

    If groupby_col is provided, plots separate trajectories per group.
    """
    print("\n" + "="*70)
    print("PLOTTING TREATMENT TRAJECTORIES")
    print("="*70)

    # Use provided colors or defaults
    if group_colors is None:
        group_colors = {'short': '#E74C3C', 'long': '#3498DB'}

    try:
        if groupby_col:
            # Grouped trajectories (e.g., by CRS)
            fig = pc.pl.archetypal_space(
                adata,
                pca_key=pca_key,
                color_by=groupby_col,
                categorical_colors=group_colors,
                title=f'Treatment Trajectories by {groupby_col} ({treatment_col})',
                show_centroids=True,
                centroid_condition=treatment_col,
                centroid_order=centroid_result['levels'],
                centroid_groupby=groupby_col,
                centroid_colors=group_colors,
                centroid_size=20.0,
                centroid_line_width=6.0,
                cell_opacity=0.4,
                cell_size=3.0,
            )
            filename_suffix = f'by_{groupby_col}'
        else:
            # Single trajectory (no grouping)
            fig = pc.pl.archetypal_space(
                adata,
                pca_key=pca_key,
                color_by=treatment_col,
                title=f'Treatment Trajectories ({treatment_col})',
                show_centroids=True,
                centroid_condition=treatment_col,
                centroid_order=centroid_result['levels'],
                centroid_size=20.0,
                centroid_line_width=6.0,
                cell_opacity=0.4,
                cell_size=3.0,
            )
            filename_suffix = 'by_treatment'

        if fig is not None:
            # Save as interactive HTML (avoids Kaleido/Chrome issues on HPC)
            html_path = os.path.join(output_dir, f'treatment_trajectories_{filename_suffix}.html')
            fig.write_html(html_path)
            print(f"  Saved interactive trajectory plot to: {html_path}")

    except Exception as e:
        print(f"  Warning: Could not generate trajectory plot: {e}")
        print("  Falling back to manual trajectory plotting...")

        try:
            weights = adata.obsm.get('cell_archetype_weights')
            if weights is None:
                print("  Warning: No cell_archetype_weights found, skipping fallback plot")
            else:
                fig, ax = plt.subplots(figsize=(12, 10))

                if groupby_col:
                    # Plot cells colored by group
                    for group_val, color in group_colors.items():
                        mask = adata.obs[groupby_col] == group_val
                        if mask.sum() > 0:
                            cell_weights = weights[mask]
                            ax.scatter(cell_weights[:, 0], cell_weights[:, 1],
                                      c=color, alpha=0.2, s=5, label=f'{group_val}')

                    # Plot grouped centroid trajectories
                    if centroid_result.get('group_centroids_3d'):
                        for group_val, color in group_colors.items():
                            if group_val in centroid_result['group_centroids_3d']:
                                group_centroids = centroid_result['group_centroids_3d'][group_val]
                                coords = [group_centroids[lvl][:2] for lvl in centroid_result['levels']
                                         if lvl in group_centroids]
                                if len(coords) >= 2:
                                    coords = np.array(coords)
                                    ax.plot(coords[:, 0], coords[:, 1], c=color, linewidth=3,
                                           label=f'{group_val} trajectory')
                                    ax.scatter(coords[:, 0], coords[:, 1], c=color, s=150,
                                              edgecolors='white', linewidths=2, zorder=5)
                else:
                    # Single trajectory
                    ax.scatter(weights[:, 0], weights[:, 1], alpha=0.2, s=5, c='gray')
                    if centroid_result.get('centroids_3d'):
                        coords = [centroid_result['centroids_3d'][lvl][:2]
                                 for lvl in centroid_result['levels']
                                 if lvl in centroid_result['centroids_3d']]
                        if len(coords) >= 2:
                            coords = np.array(coords)
                            ax.plot(coords[:, 0], coords[:, 1], c='blue', linewidth=3)
                            ax.scatter(coords[:, 0], coords[:, 1], c='blue', s=150,
                                      edgecolors='white', linewidths=2, zorder=5)

                ax.set_xlabel('Weight 1', fontsize=12)
                ax.set_ylabel('Weight 2', fontsize=12)
                ax.set_title(f'Treatment Trajectories ({treatment_col})', fontsize=14)
                ax.legend(loc='upper right')

                fig_path = os.path.join(output_dir, 'treatment_trajectories_fallback.png')
                fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved fallback trajectory plot to: {fig_path}")

        except Exception as e2:
            print(f"  Warning: Fallback plotting also failed: {e2}")

    print("="*70 + "\n")


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Treatment Phase Archetypal Analysis Workflow',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input_adata',
        type=str,
        required=True,
        help='Path to input .h5ad file (pre-processed with PCA)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for all results'
    )

    parser.add_argument(
        '--pathway_file',
        type=str,
        required=True,
        help='Path to pathway CSV (columns: source, target, weight, pathway)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help='Computing device'
    )

    return parser.parse_args()


def get_device(requested: str) -> str:
    """Determine computing device."""
    if requested == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    return requested


def main():
    """Main workflow execution."""
    args = parse_args()

    # Setup device
    device = get_device(args.device)

    # Print header
    print("\n" + "="*70)
    print("TREATMENT PHASE ARCHETYPAL ANALYSIS WORKFLOW")
    print("="*70)
    print(f"Input: {args.input_adata}")
    print(f"Output: {args.output_dir}")
    print(f"Pathways: {args.pathway_file}")
    print(f"Device: {device}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # =========================================================================
    # STEP 1: LOAD AND VALIDATE DATA
    # =========================================================================

    print("Loading AnnData...")
    adata = sc.read_h5ad(args.input_adata)
    print(f"  Loaded: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

    # Filter to Helsinki subset (dataset == 1) which has valid CRS scores
    if 'dataset' in adata.obs.columns:
        original_n = adata.shape[0]
        # Handle categorical, string, or numeric dataset values
        dataset_col = adata.obs['dataset']
        # Convert to string for reliable comparison (handles categorical, int, float)
        mask = dataset_col.astype(str) == '1'
        if mask.sum() == 0:
            # Try numeric comparison as fallback
            try:
                mask = dataset_col == 1
            except:
                pass
        if mask.sum() == 0:
            raise ValueError(f"No cells found with dataset==1. Unique values: {dataset_col.unique().tolist()}")
        adata = adata[mask].copy()
        print(f"  Filtered to Helsinki subset (dataset==1): {adata.shape[0]:,} cells "
              f"(from {original_n:,}, {100*adata.shape[0]/original_n:.1f}%)")

    # Validate input data
    validation = validate_input_data(adata)
    pca_key = validation['pca_key']
    condition_col = validation['condition_col']
    treatment_col = validation['treatment_col']

    # =========================================================================
    # STEP 2: HYPERPARAMETER SEARCH
    # =========================================================================

    best_params = run_hyperparameter_search(adata, pca_key, args.output_dir, device)

    # =========================================================================
    # STEP 3: TRAIN FINAL MODEL
    # =========================================================================

    training_results = train_final_model(
        adata, best_params, pca_key, args.output_dir, device
    )

    # =========================================================================
    # STEP 4: ARCHETYPAL ANALYSIS
    # =========================================================================

    compute_archetypal_analysis(adata, training_results, pca_key, args.output_dir)

    # =========================================================================
    # STEP 5: PATHWAY SCORING
    # =========================================================================

    compute_pathway_scores(
        adata,
        args.pathway_file,
        expression_layer=validation['expression_layer'],
        gene_symbol_col=validation.get('gene_symbol_col')
    )

    # =========================================================================
    # STEP 6: GENE ASSOCIATIONS
    # =========================================================================

    gene_results = run_gene_associations(
        adata, args.output_dir,
        expression_layer=validation.get('expression_layer')
    )

    # =========================================================================
    # STEP 7: PATHWAY ASSOCIATIONS
    # =========================================================================

    pathway_results = run_pathway_associations(adata, args.output_dir)

    # =========================================================================
    # STEP 8: PATTERN ANALYSES
    # =========================================================================

    pattern_results = run_pattern_analyses(adata, args.output_dir)

    # =========================================================================
    # STEP 9: CONDITIONAL ASSOCIATIONS
    # =========================================================================

    if condition_col:
        conditional_results = run_conditional_associations(
            adata, condition_col, args.output_dir
        )

    # =========================================================================
    # STEP 10: 3D ARCHETYPAL SPACE VISUALIZATION
    # =========================================================================

    generate_archetypal_space_plots(adata, condition_col, pca_key, args.output_dir)

    # =========================================================================
    # STEP 11: TREATMENT PHASE TRAJECTORIES (stratified by CRS)
    # =========================================================================

    centroid_result = compute_treatment_centroids(
        adata, treatment_col, pca_key, args.output_dir,
        groupby_col='CRS'
    )

    plot_treatment_trajectories(
        adata, centroid_result, treatment_col, pca_key, args.output_dir,
        groupby_col='CRS',
        group_colors={'short': '#E74C3C', 'long': '#3498DB'}
    )

    # =========================================================================
    # STEP 12: SAVE FINAL ADATA
    # =========================================================================

    print("\n" + "="*70)
    print("SAVING FINAL RESULTS")
    print("="*70)

    final_adata_path = os.path.join(args.output_dir, 'processed_adata.h5ad')
    print(f"  Saving processed AnnData to: {final_adata_path}")
    adata.write_h5ad(final_adata_path)
    print("  Done!")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\n" + "="*70)
    print("WORKFLOW COMPLETE")
    print("="*70)
    print(f"\nOutputs saved to: {args.output_dir}")
    print("\nGenerated files:")
    for f in sorted(os.listdir(args.output_dir)):
        fpath = os.path.join(args.output_dir, f)
        size = os.path.getsize(fpath) / 1024  # KB
        print(f"  {f}: {size:.1f} KB")

    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()

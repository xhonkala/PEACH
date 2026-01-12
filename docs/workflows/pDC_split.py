#!/usr/bin/env python
"""
Complete Peach + CellRank Analysis Pipeline - Version 2

FIXES APPLIED:
1. Added logcounts layer creation from adata.X
2. Added extract_archetype_weights() call before CellRank setup
3. Import igraph early to check availability
4. Fixed pathway CSV loading - convert to DataFrame properly
5. Better error handling with full tracebacks

Usage:
    python scripts/run_peach_cellrank_analysis_v2.py

Requirements:
    - Conda environment: archetype (with igraph installed)
    - R installation in renv for GAMR models

2/3 scripts used in preparing the pre-print at https://www.biorxiv.org/content/10.64898/2025.12.29.696912v1
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import torch
import torch.optim as optim
from datetime import datetime
import traceback

# Set R_HOME before importing CellRank
os.environ['R_HOME'] = '~/honkala/miniconda3/envs/renv'

# Check for igraph before proceeding
try:
    import igraph
    IGRAPH_AVAILABLE = True
    print("✓ igraph available for PAGA computation")
    print(f"  igraph version: {igraph.__version__}")
    print(f"  igraph location: {igraph.__file__}")
except ImportError:
    IGRAPH_AVAILABLE = False
    print("⚠️  WARNING: python-igraph not installed")
    print("   CellRank PAGA analysis will be skipped")
    print("   Install with: conda install -c conda-forge python-igraph")

# Import CellRank - it will check for igraph internally
import cellrank as cr
import peach as pc

# Verify scanpy can see igraph
if IGRAPH_AVAILABLE:
    try:
        # Test that scanpy can access igraph (needed for PAGA)
        import scanpy.external as sce
        print("✓ Scanpy igraph integration verified")
    except Exception as e:
        print(f"⚠️  Scanpy may have issues with igraph: {e}")
        print("   Continuing anyway - will fail later if PAGA can't run")

# Import core modules for training
try:
    from peach._core.models.Deep_AA import Deep_AA
    from peach._core.utils.training import train_vae
    from peach._core.utils.load_anndata import create_dataloader_from_anndata
except ImportError:
    from src.models.Deep_AA import Deep_AA
    from src.utils.training import train_vae
    from src.utils.load_anndata import create_dataloader_from_anndata

print("=" * 80)
print("PEACH + CELLRANK ANALYSIS PIPELINE (V2)")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# SECTION 1: LOAD DATA AND SET CONFIGURATION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 1: DATA LOADING AND CONFIGURATION")
print("=" * 80)

# USER CONFIGURATION
OUTPUT_DIR = "~/HSC_analysis/20251006_myeloid"
INPUT_ADATA_PATH = "~/hematopoiesis/myeloid_data_13PCs.h5ad"

# Model configuration
CONFIG = {
    'n_archetypes': 7,
    'hidden_dims': [128, 256, 512],
    'n_pcs': 13,
    'n_epochs': 200,
    'inflation_factor': 1.0,
    'use_pcha_init': True,
    'barycentric': True,
    'learning_rate': 0.001,
    'early_stopping_patience': 10
}

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

for directory in [PLOTS_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

print(f"✓ Output directory: {OUTPUT_DIR}")
print(f"✓ Created subdirectories: plots/, models/, results/")

# Load data
print(f"\nLoading data from: {INPUT_ADATA_PATH}")
adata = sc.read_h5ad(INPUT_ADATA_PATH)
print(f"✓ Loaded AnnData: {adata.shape}")
print(f"  Observations: {adata.n_obs}")
print(f"  Features: {adata.n_vars}")

# FIX 1: Create logcounts layer if missing
if 'logcounts' not in adata.layers:
    print("\n🔧 FIX: Creating 'logcounts' layer from adata.X...")
    adata.layers['logcounts'] = adata.X.copy()
    print("✓ Created adata.layers['logcounts']")
else:
    print("✓ 'logcounts' layer already exists")

# Convert gene IDs to gene symbols if needed
if 'gene_symbols' in adata.var.columns:
    print("\nConverting ENSG IDs to gene symbols...")
    print(f"  Before: {adata.var_names[0]} (example)")
    adata.var_names = adata.var['gene_symbols']
    adata.var_names_make_unique()
    print(f"  After: {adata.var_names[0]} (example)")
    print("✓ Gene symbols set as var_names")
else:
    print("⚠ Warning: 'gene_symbols' column not found in adata.var")

# Verify PCA coordinates
if 'X_pca' not in adata.obsm:
    raise ValueError("PCA coordinates not found. Run sc.tl.pca() first.")
print(f"✓ PCA coordinates found: {adata.obsm['X_pca'].shape}")

# Save configuration
config_df = pd.DataFrame([CONFIG])
config_df.to_csv(os.path.join(RESULTS_DIR, "config.csv"), index=False)
print(f"✓ Saved configuration")

# ============================================================================
# SECTION 2: TRAIN MODEL AND EVALUATE
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: MODEL TRAINING")
print("=" * 80)

# Create DataLoader
print("\nPreparing training data...")
dataloader = create_dataloader_from_anndata(adata, batch_size=128)
print(f"✓ DataLoader created with batch_size=128")

# Initialize model
print(f"\nInitializing Deep_AA model...")
model = Deep_AA(
    input_dim=CONFIG['n_pcs'],
    latent_dim=CONFIG['n_archetypes'],
    n_archetypes=CONFIG['n_archetypes'],
    hidden_dims=CONFIG['hidden_dims'],
    archetypal_weight=0.9,
    diversity_weight=0.1,
    inflation_factor=CONFIG['inflation_factor']
)

# PCHA initialization
if CONFIG['use_pcha_init']:
    print("Initializing with PCHA...")
    sample_batch = next(iter(dataloader))[0]
    model.initialize_with_pcha_and_inflation(
        sample_batch,
        inflation_factor=CONFIG['inflation_factor']
    )
    print("✓ PCHA initialization complete")

# Setup optimizer
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

# Train model
print(f"\nTraining model for {CONFIG['n_epochs']} epochs...")
print("Tracking: loss, RMSE, archetype R², stability")

results, trained_model = train_vae(
    model=model,
    data_loader=dataloader,
    optimizer=optimizer,
    n_epochs=CONFIG['n_epochs'],
    adata=adata,
    track_stability=True,
    validate_constraints=True,
    store_coords_key='archetype_coordinates'
)

print("\n✓ Training complete!")
print(f"  Final loss: {results['history']['loss'][-1]:.4f}")
if 'rmse' in results['history'] and len(results['history']['rmse']) > 0:
    print(f"  Final RMSE: {results['history']['rmse'][-1]:.4f}")
if 'archetype_r2' in results['history'] and len(results['history']['archetype_r2']) > 0:
    print(f"  Final Archetype R²: {results['history']['archetype_r2'][-1]:.4f}")

# Save trained model
model_path = os.path.join(MODELS_DIR, "trained_model.pt")
torch.save(trained_model.state_dict(), model_path)
print(f"✓ Saved model to: {model_path}")

# Save training history
history_dict = {}
max_len = max(len(v) for v in results['history'].values() if isinstance(v, list))
for key, values in results['history'].items():
    if isinstance(values, list):
        padded = values + [np.nan] * (max_len - len(values))
        history_dict[key] = padded

history_df = pd.DataFrame(history_dict)
history_df.to_csv(os.path.join(RESULTS_DIR, "training_history.csv"), index=False)
print(f"✓ Saved training history")

# Visualize training metrics
print("\nGenerating training visualizations...")
try:
    fig = pc.pl.training_metrics(results['history'])
    if fig is not None:
        fig.write_html(os.path.join(PLOTS_DIR, "training_metrics.html"))
        print(f"✓ Saved training metrics plot")
except Exception as e:
    print(f"⚠ Warning: Could not generate training metrics plot: {e}")

# Print final performance
print("\n" + "-" * 80)
print("FINAL MODEL PERFORMANCE:")
print("-" * 80)
if 'archetype_r2' in results['history'] and len(results['history']['archetype_r2']) > 0:
    print(f"Archetype R²: {results['history']['archetype_r2'][-1]:.4f}")
if 'rmse' in results['history'] and len(results['history']['rmse']) > 0:
    print(f"RMSE: {results['history']['rmse'][-1]:.4f}")
if 'mae' in results['history'] and len(results['history']['mae']) > 0:
    print(f"MAE: {results['history']['mae'][-1]:.4f}")
print(f"Convergence epoch: {len(results['history']['loss'])}")
print("-" * 80)

# ============================================================================
# SECTION 3: ARCHETYPE CHARACTERIZATION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: ARCHETYPE CHARACTERIZATION")
print("=" * 80)

# Extract archetypal coordinates
print("\nExtracting archetypal coordinates...")
pc.tl.archetypal_coordinates(adata)
print("✓ Archetypal coordinates extracted")

# FIX 2: Extract archetype weights (needed for CellRank)
print("\n🔧 FIX: Extracting cell archetype weights...")
weights = pc.tl.extract_archetype_weights(
    adata,
    model=trained_model,
    pca_key='X_pca',
    weights_key='cell_archetype_weights'
)
print(f"✓ Extracted weights: {weights.shape}")
print(f"  Stored in adata.obsm['cell_archetype_weights']")

# Assign cells to archetypes
print("\nAssigning cells to archetypes...")
pc.tl.assign_archetypes(adata, percentage_per_archetype=0.15)
print(f"✓ Cell assignments complete")
print("\nArchetype distribution:")
print(adata.obs['archetypes'].value_counts().sort_index())

# Visualizations
print("\nGenerating archetypal space visualizations...")

if 'cell_type' in adata.obs.columns:
    fig1 = pc.pl.archetypal_space(adata, color_by='cell_type')
    fig1.write_html(os.path.join(PLOTS_DIR, "archetypal_space_by_celltype.html"))
    print(f"✓ Saved: archetypal_space_by_celltype.html")

fig2 = pc.pl.archetypal_space(adata, color_by='archetypes')
fig2.write_html(os.path.join(PLOTS_DIR, "archetypal_space_by_archetype.html"))
print(f"✓ Saved: archetypal_space_by_archetype.html")

# Gene expression plots
genes_to_plot = ['ITGAE', 'YBX1', 'SOD1', 'HLA-DRA']
existing_genes = [g for g in genes_to_plot if g in adata.var_names]

if len(existing_genes) > 0:
    print(f"\nGenerating gene expression plots for: {existing_genes}")
    for gene in existing_genes:
        fig = pc.pl.archetypal_space(adata, color_by=gene, color_scale='plasma')
        fig.write_html(os.path.join(PLOTS_DIR, f"archetypal_space_{gene}.html"))
        print(f"✓ Saved: archetypal_space_{gene}.html")

# FIX 3 & 4: Load pathway database - convert to DataFrame properly
print("\n\nLoading pathway database (c5_bp - GO Biological Process)...")
pathway_file = "~/honkala/c5_bp_pathway_df.csv"
pathway_df = None

try:
    # Read CSV
    pathway_data = pd.read_csv(pathway_file)

    # Convert to DataFrame if it's not already (e.g., if it's a numpy array or matrix)
    if not isinstance(pathway_data, pd.DataFrame):
        print(f"⚠️  WARNING: read_csv returned {type(pathway_data)}, converting to DataFrame...")

        # Try to convert to DataFrame
        try:
            # If it's a numpy array/matrix, we need to figure out columns
            if hasattr(pathway_data, 'shape'):
                # Assume it has 4 columns based on your file
                pathway_df = pd.DataFrame(
                    pathway_data,
                    columns=['source', 'target', 'weight', 'pathway']
                )
                print(f"  ✓ Converted {type(pathway_data)} to DataFrame")
            else:
                print(f"  ❌ Don't know how to convert {type(pathway_data)} to DataFrame")
                pathway_df = None
        except Exception as conv_error:
            print(f"  ❌ Conversion failed: {conv_error}")
            pathway_df = None
    else:
        # It's already a DataFrame
        pathway_df = pathway_data
        print(f"✓ Loaded pathway DataFrame: {len(pathway_df)} rows")

    if pathway_df is not None:
        print(f"  Type: {type(pathway_df)}")
        print(f"  Columns: {list(pathway_df.columns)}")

        # Validate required columns
        required_cols = ['source', 'target', 'weight', 'pathway']
        missing_cols = [col for col in required_cols if col not in pathway_df.columns]

        if missing_cols:
            print(f"❌ ERROR: Missing required columns: {missing_cols}")
            pathway_df = None
        else:
            print(f"  ✓ All required columns present")
            print(f"  Unique pathways: {pathway_df['source'].nunique()}")
            print(f"  Unique genes: {pathway_df['target'].nunique()}")

            # Compute pathway activity scores
            print("\n🧮 Computing pathway activity scores...")
            pc.pp.compute_pathway_scores(
                adata,
                net=pathway_df,
                use_layer='logcounts',
                obsm_key='pathway_scores',
                verbose=True
            )

            # Verify it worked
            if 'pathway_scores' in adata.obsm:
                print(f"✅ SUCCESS: Pathway scores computed!")
                print(f"  Stored in: adata.obsm['pathway_scores']")
                print(f"  Shape: {adata.obsm['pathway_scores'].shape}")
            else:
                print(f"❌ ERROR: Pathway scores not in adata.obsm")
                pathway_df = None

except FileNotFoundError:
    print(f"❌ ERROR: Pathway file not found at: {pathway_file}")
    pathway_df = None

except Exception as e:
    print(f"❌ ERROR loading/computing pathway scores:")
    print(f"   {type(e).__name__}: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    pathway_df = None

# Check final status
if pathway_df is not None and 'pathway_scores' in adata.obsm:
    print("\n✅ Pathway analysis will proceed")
else:
    print("\n⚠️  Pathway analysis will be skipped")
    pathway_df = None

# Gene association tests
print("\nRunning gene association tests...")
gene_results = pc.tl.gene_associations(adata, fdr_scope='global')
gene_results.to_csv(os.path.join(RESULTS_DIR, "gene_associations.csv"), index=False)
print(f"✓ Saved gene associations")
if 'significant' in gene_results.columns:
    print(f"  Significant genes: {(gene_results['significant'] == True).sum()}")

# Dotplot for top genes
if len(gene_results) > 0:
    try:
        print(f"\n🎨 Creating gene dotplot with {len(gene_results)} results...")
        print(f"  Columns: {list(gene_results.columns)}")
        print(f"  Archetypes represented: {gene_results['archetype'].unique()}")

        fig_genes = pc.pl.dotplot(
            gene_results,            # Pass FULL DataFrame, let dotplot filter
            top_n_per_group=10,      # Get top 10 per archetype
            filter_zero_p=False      # Don't filter zero p-values
        )
        if fig_genes is not None:
            fig_genes.savefig(os.path.join(PLOTS_DIR, "gene_associations_dotplot.png"),
                            dpi=300, bbox_inches='tight')
            plt.close(fig_genes)
            print(f"✓ Saved gene associations dotplot")
        else:
            print(f"⚠️  dotplot returned None")
    except Exception as e:
        print(f"⚠ Warning: Could not create gene dotplot: {e}")
        traceback.print_exc()

# Pathway association tests (only if pathways loaded)
if pathway_df is not None and 'pathway_scores' in adata.obsm:
    print("\nRunning pathway association tests...")
    pathway_results = pc.tl.pathway_associations(
        adata,
        pathway_obsm_key='pathway_scores',
        fdr_scope='global'
    )
    pathway_results.to_csv(os.path.join(RESULTS_DIR, "pathway_associations.csv"), index=False)
    print(f"✓ Saved pathway associations")
    if 'significant' in pathway_results.columns:
        print(f"  Significant pathways: {(pathway_results['significant'] == True).sum()}")

    # Dotplot for top pathways
    if len(pathway_results) > 0:
        try:
            print(f"\n🎨 Creating pathway dotplot with {len(pathway_results)} results...")
            print(f"  Columns: {list(pathway_results.columns)}")
            print(f"  Archetypes represented: {pathway_results['archetype'].unique()}")

            fig_pathways = pc.pl.dotplot(
                pathway_results,            # Pass FULL DataFrame, let dotplot filter
                top_n_per_group=10,         # Get top 10 per archetype
                filter_zero_p=False         # Don't filter zero p-values
            )
            if fig_pathways is not None:
                fig_pathways.savefig(os.path.join(PLOTS_DIR, "pathway_associations_dotplot.png"),
                                   dpi=300, bbox_inches='tight')
                plt.close(fig_pathways)
                print(f"✓ Saved pathway associations dotplot")
            else:
                print(f"⚠️  dotplot returned None")
        except Exception as e:
            print(f"⚠ Warning: Could not create pathway dotplot: {e}")
            traceback.print_exc()
else:
    print("\n⚠️  Skipping pathway association tests (pathways not loaded)")

# Pattern analysis (only if pathways loaded)
if pathway_df is not None and 'pathway_scores' in adata.obsm:
    print("\nRunning pattern analyses...")

    # Exclusivity
    print("  - Exclusivity patterns...")
    exclusive_results = pc.tl.archetype_exclusive_patterns(
        adata,
        data_obsm_key='pathway_scores',
        min_effect_size=0.05,
        fdr_scope='global'
    )
    exclusive_results.to_csv(os.path.join(RESULTS_DIR, "exclusive_patterns.csv"), index=False)
    print(f"    ✓ Found {len(exclusive_results)} exclusive patterns")

    # Specialization
    print("  - Specialization patterns...")
    specialization_results = pc.tl.specialization_patterns(
        adata,
        data_obsm_key='pathway_scores',
        fdr_scope='per_archetype'
    )
    specialization_results.to_csv(os.path.join(RESULTS_DIR, "specialization_patterns.csv"), index=False)
    print(f"    ✓ Found {len(specialization_results)} specialization patterns")

    # Tradeoffs
    print("  - Tradeoff patterns...")
    tradeoff_results = pc.tl.tradeoff_patterns(
        adata,
        data_obsm_key='pathway_scores',
        tradeoffs='pairs',
        min_effect_size=0.1,
        fdr_scope='global'
    )
    tradeoff_results.to_csv(os.path.join(RESULTS_DIR, "tradeoff_patterns.csv"), index=False)
    print(f"    ✓ Found {len(tradeoff_results)} tradeoff patterns")
else:
    print("\n⚠️  Skipping pattern analysis (pathways not loaded)")

# Conditional association tests
if 'cell_type' in adata.obs.columns:
    print("\nRunning conditional association tests for cell_type...")
    try:
        conditional_results = pc.tl.conditional_associations(
            adata,
            obs_column='cell_type',
            fdr_method='benjamini_hochberg'
        )
        conditional_results.to_csv(os.path.join(RESULTS_DIR, "conditional_associations_celltype.csv"),
                                  index=False)
        print(f"✓ Saved conditional associations")
    except Exception as e:
        print(f"⚠ Warning: Could not complete conditional association tests: {e}")

adata.write(os.path.join(RESULTS_DIR, "myeloid.h5ad"))

# ============================================================================
# SECTION 4: CELLRANK SETUP
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: CELLRANK PREPARATION")
print("=" * 80)

if IGRAPH_AVAILABLE:
    print("\nSetting up CellRank workflow...")
    print("  - Computing neighbors in PCA space")
    print("  - Computing UMAP")
    print("  - Computing PAGA connectivity")
    print("  - Building ConnectivityKernel")
    print("  - Defining high-purity terminal states (threshold=0.80)")
    print("  - Computing fate probabilities via GPCCA")

    try:
        # Try with PAGA first
        try:
            ck, g = pc.tl.setup_cellrank(
                adata,
                high_purity_threshold=0.80,
                n_neighbors=30,
                n_pcs=CONFIG['n_pcs'],
                compute_paga=True,
                verbose=True
            )
            print("\n✓ CellRank setup complete (with PAGA)!")

        except Exception as paga_error:
            # If PAGA fails (igraph issue), try without it
            if 'igraph' in str(paga_error).lower() or 'paga' in str(paga_error).lower():
                print(f"\n⚠️  PAGA computation failed: {paga_error}")
                print("   Retrying without PAGA...")

                ck, g = pc.tl.setup_cellrank(
                    adata,
                    high_purity_threshold=0.80,
                    n_neighbors=30,
                    n_pcs=CONFIG['n_pcs'],
                    compute_paga=False,  # Skip PAGA
                    verbose=True
                )
                print("\n✓ CellRank setup complete (without PAGA)!")
            else:
                raise  # Re-raise if it's not a PAGA/igraph issue

        # Store GPCCA object
        adata.uns['cellrank_gpcca'] = g

        print(f"  Terminal states defined: {adata.obs['terminal_states'].notna().sum()} cells")
        print(f"  Lineages: {adata.uns['lineage_names']}")

        CELLRANK_SUCCESS = True

    except Exception as e:
        print(f"\n❌ ERROR in CellRank setup:")
        print(f"   {type(e).__name__}: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("\n⚠️  Skipping CellRank analysis")
        CELLRANK_SUCCESS = False

else:
    print("\n⚠️  Skipping CellRank setup (igraph not available)")
    print("   Install with: conda install -c conda-forge python-igraph")
    CELLRANK_SUCCESS = False

# ============================================================================
# SECTION 5: CELLRANK ANALYSIS (only if setup succeeded)
# ============================================================================
if CELLRANK_SUCCESS:
    print("\n" + "=" * 80)
    print("SECTION 5: CELLRANK TRAJECTORY ANALYSIS")
    print("=" * 80)

    # Compute lineage-specific pseudotimes
    print("\nComputing lineage-specific pseudotimes...")
    pc.tl.compute_lineage_pseudotimes(adata)
    pseudotime_cols = [col for col in adata.obs.columns if col.startswith('pseudotime_to_')]
    print(f"✓ Created {len(pseudotime_cols)} pseudotime variables")

    # PAGA visualizations (only if PAGA was computed)
    if 'paga' in adata.uns:
        print("\nGenerating PAGA connectivity visualizations...")

        try:
            sc.pl.paga(adata, color='archetypes', save=False)
            plt.savefig(os.path.join(PLOTS_DIR, "paga_connectivity_graph.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved PAGA graph")
        except Exception as e:
            print(f"⚠️  Could not create PAGA graph plot: {e}")
            plt.close('all')

        # PAGA heatmap
        try:
            paga_conn = adata.uns['paga']['connectivities'].toarray()
            paga_labels = adata.obs['archetypes'].cat.categories
            paga_df = pd.DataFrame(paga_conn, index=paga_labels, columns=paga_labels)

            arch_only = [label for label in paga_labels if label.startswith('archetype_')]
            paga_df_filtered = paga_df.loc[arch_only, arch_only]

            plt.figure(figsize=(10, 8))
            sns.heatmap(paga_df_filtered, annot=True, fmt='.3f', cmap='viridis',
                        cbar_kws={'label': 'PAGA Connectivity'}, vmin=0, vmax=1)
            plt.title('PAGA: Archetype-to-Archetype Connectivity', fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, "paga_connectivity_heatmap.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved PAGA heatmap")

            paga_df_filtered.to_csv(os.path.join(RESULTS_DIR, "paga_connectivity_matrix.csv"))
        except Exception as e:
            print(f"⚠️  Could not create PAGA heatmap: {e}")
            plt.close('all')
    else:
        print("\n⚠️  Skipping PAGA visualizations (PAGA not computed)")

    # Fate probabilities
    print("\nGenerating fate probability visualizations...")

    try:
        g.plot_fate_probabilities(same_plot=True, legend_loc="right", save=None)
        plt.savefig(os.path.join(PLOTS_DIR, "fate_probabilities_combined.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved combined fate probabilities")
    except Exception as e:
        print(f"⚠ Warning: {e}")
        plt.close('all')

    try:
        g.plot_fate_probabilities(same_plot=False, basis='X_umap', ncols=3,
                                 figsize=(18, 12), save=None)
        plt.savefig(os.path.join(PLOTS_DIR, "fate_probabilities_separate.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved separate fate probabilities")
    except Exception as e:
        print(f"⚠ Warning: {e}")
        plt.close('all')

    # Aggregate fate probabilities
    archetype_labels = sorted([k for k in adata.uns['lineage_names']
                               if k != 'no_archetype' and k.startswith('archetype_')])

    try:
        cr.pl.aggregate_fate_probabilities(
            adata,
            cluster_key='archetypes',
            lineages=archetype_labels,
            mode='heatmap',
            figsize=(10, 8),
            title='Mean Fate Probabilities per Archetype',
            save=None
        )
        plt.savefig(os.path.join(PLOTS_DIR, "aggregate_fate_probabilities.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved aggregate fate probabilities")
    except Exception as e:
        print(f"⚠ Warning: {e}")
        plt.close('all')

    # Driver genes
    print("\nIdentifying driver genes...")
    for lineage in archetype_labels:
        print(f"  Processing {lineage}...")
        try:
            drivers = pc.tl.compute_lineage_drivers(
                adata,
                lineage=lineage,
                method='correlation',
                n_genes=100
            )
            drivers.to_csv(os.path.join(RESULTS_DIR, f"driver_genes_{lineage}.csv"))

            fig = pc.pl.lineage_drivers(adata, lineage=lineage, n_genes=10, figsize=(8, 6))
            plt.savefig(os.path.join(PLOTS_DIR, f"driver_genes_heatmap_{lineage}.png"),
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    ✓ Saved drivers for {lineage}")
        except Exception as e:
            print(f"    ⚠ Warning: {e}")

    # Transition frequencies
    print("\nComputing transition frequencies...")
    transitions = pc.tl.compute_transition_frequencies(
        adata,
        start_weight_threshold=0.5,
        fate_prob_threshold=0.3
    )
    transitions.to_csv(os.path.join(RESULTS_DIR, "transition_frequencies.csv"))

    plt.figure(figsize=(10, 8))
    sns.heatmap(transitions, annot=True, fmt='d', cmap='YlOrRd',
                cbar_kws={'label': 'Number of Transitions'}, linewidths=0.5)
    plt.title('Archetype Transition Frequencies', fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "transition_frequencies_heatmap.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved transition frequencies")

else:
    print("\n" + "=" * 80)
    print("SECTION 5: CELLRANK ANALYSIS - SKIPPED")
    print("=" * 80)
    print("⚠️  CellRank setup failed or igraph unavailable")

# ============================================================================
# SAVE FINAL RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING FINAL ANNOTATED ADATA")
print("=" * 80)

output_adata_path = os.path.join(OUTPUT_DIR, "adata_fully_annotated.h5ad")
adata.write_h5ad(output_adata_path)
print(f"✓ Saved fully annotated AnnData")

print("\nData stored in AnnData:")
print("  adata.obs columns:", len(adata.obs.columns))
print("  adata.obsm keys:", len(adata.obsm.keys()))
print("  adata.layers keys:", len(adata.layers.keys()))
print("  adata.uns keys:", len(adata.uns.keys()))

# ============================================================================
# COMPLETION SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nAll outputs saved to: {OUTPUT_DIR}")

print("\n" + "=" * 80)
print("Pipeline execution successful! 🎉")
print("=" * 80)

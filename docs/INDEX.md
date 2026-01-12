# 🍑 PEACH Documentation

**Version**: 0.3.0

---

## Quick Start

```python
import peach as pc
import scanpy as sc

# Load and prepare data
adata = sc.read_h5ad('your_data.h5ad')
sc.pp.pca(adata, n_comps=50) # recommend using lowest n PCs that makes sense for your dataset for better results

# Train archetypal model
results = pc.tl.train_archetypal(adata, n_archetypes=5, n_epochs=100)

# Extract coordinates and assign cells
pc.tl.archetypal_coordinates(adata)
pc.tl.assign_archetypes(adata, percentage_per_archetype=0.15)

# Statistical analysis
gene_results = pc.tl.gene_associations(adata, fdr_scope='global')

# Visualize
pc.pl.archetypal_space(adata, color_by='archetypes').show()
```

See [Installation](installation.md) for setup instructions.

---

## Tutorials

Interactive Jupyter notebooks walking through each analysis step:

| Tutorial | Description |
|----------|-------------|
| [01_data_loading](tutorials/01_data_loading.ipynb) | Data loading and preprocessing |
| [02_hyperparameter_search](tutorials/02_hyperparameter_search.ipynb) | Cross-validation for optimal archetypes |
| [03_model_training](tutorials/03_model_training.ipynb) | Training archetypal models |
| [04_archetype_coordinates](tutorials/04_archetype_coordinates.ipynb) | Coordinate extraction and cell assignment |
| [05_gene_enrichment](tutorials/05_gene_enrichment.ipynb) | Gene and pathway associations |
| [06_cellrank_integration](tutorials/06_cellrank_integration.ipynb) | Trajectory analysis with CellRank |
| 07 pointed to a set of experimental features that are not yet ready for release, check back later |
| [08_visualization](tutorials/08_visualization.ipynb) | Comprehensive visualization guide |

---

## Workflow Scripts

Standalone Python scripts for each workflow stage:

| Script | Purpose |
|--------|---------|
| [WORKFLOW_01_DATA_LOAD.py](workflows/WORKFLOW_01_DATA_LOAD.py) | Data loading, QC, PCA |
| [WORKFLOW_02_HYPERPARAM_SEARCH.py](workflows/WORKFLOW_02_HYPERPARAM_SEARCH.py) | Hyperparameter optimization |
| [WORKFLOW_03_MODEL_TRAINING.py](workflows/WORKFLOW_03_MODEL_TRAINING.py) | Model training |
| [WORKFLOW_04_COORDINATES.py](workflows/WORKFLOW_04_COORDINATES.py) | Coordinates and assignment |
| [WORKFLOW_05_ENRICHMENT.py](workflows/WORKFLOW_05_ENRICHMENT.py) | Gene/pathway enrichment |
| [WORKFLOW_06_CELLRANK.py](workflows/WORKFLOW_06_CELLRANK.py) | CellRank integration |
| [WORKFLOW_08_VISUALIZATION.py](workflows/WORKFLOW_08_VISUALIZATION.py) | Visualization |

---

## API Structure

```
pp (5):  load_data, generate_synthetic, prepare_training,
         load_pathway_networks, compute_pathway_scores

tl (16): train_archetypal, hyperparameter_search, archetypal_coordinates,
         assign_archetypes, extract_archetype_weights, gene_associations,
         pathway_associations, pattern_analysis, conditional_associations,
         archetype_exclusive_patterns, specialization_patterns, tradeoff_patterns,
         setup_cellrank, compute_lineage_pseudotimes, compute_lineage_drivers,
         compute_transition_frequencies

pl (14): archetypal_space, archetypal_space_multi, training_metrics,
         elbow_curve, dotplot, archetype_positions, archetype_positions_3d,
         archetype_statistics, pattern_dotplot, pattern_summary_barplot,
         pattern_heatmap, fate_probabilities, gene_trends, lineage_drivers
```

---

## Developer Reference

For function signatures and return types, consult (in order):

1. `src/peach/_core/types_index.py` (~270 lines) - Function → return type mapping
2. `src/peach/_core/tools_schema.py` (~1000 lines) - Function → input parameters
3. `src/peach/_core/types.py` - Full Pydantic type definitions (grep as needed)

---

## Additional Resources

- [Installation Guide](installation.md)
- [Contributing](https://github.com/xhonkala/PEACH/blob/main/CONTRIBUTING.md)
- [API Reference](api/index.rst)

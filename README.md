# PEACH: Deep Archetypal Analysis for Single-Cell Biology

**Production-ready PyTorch implementation of Deep Archetypal Analysis for single-cell data with comprehensive statistical testing and visualization.**

[![Documentation Status](https://readthedocs.org/projects/scpeach/badge/?version=latest)](https://scpeach.readthedocs.io/en/latest/?badge=latest)

[![Tests](https://github.com/xhonkala/PEACH/actions/workflows/test.yaml/badge.svg)](https://github.com/xhonkala/PEACH/actions/workflows/test.yaml)
[![Documentation](https://readthedocs.org/projects/scpeach/badge/?version=latest)](https://scpeach.readthedocs.io/en/latest/?badge=latest)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-red)]()
[![scverse](https://img.shields.io/badge/scverse-compatible-orange)]()

---

## Overview

**PEACH** identifies cellular archetypes that represent extremal states in biological systems. It provides a complete workflow from data preprocessing through statistical analysis to publication-ready visualizations.

### Key Capabilities

- **Cell state identification** - Discover extreme cellular states and transitions
- **Statistical testing** - Comprehensive gene and pathway association testing
- **Pattern discovery** - Advanced pattern analysis for archetype interactions
- **Interactive visualization** - Publication-ready 3D plots and dotplots
- **Trajectory analysis** - CellRank integration for lineage analysis

---

## Quick Start

### Installation

**Option 1: Conda environment (recommended)**
```bash
git clone https://github.com/xhonkala/PEACH.git
cd PEACH
conda env create -f environment.yaml
conda activate peach
pip install -e .
```

**Option 2: pip only**
```bash
git clone https://github.com/xhonkala/PEACH.git
cd PEACH
pip install -e .
```

### Basic Workflow

```python
import peach as pc
import scanpy as sc

# 1. Load and prepare data
adata = sc.read_h5ad('your_data.h5ad')
sc.pp.pca(adata, n_comps=50)  # PCA required!

# 2. Train archetypal model
results = pc.tl.train_archetypal(adata, n_archetypes=5, n_epochs=100)
print(f"Final R2: {results.get('final_archetype_r2', 'N/A')}")

# 3. Extract coordinates and assign cells
pc.tl.archetypal_coordinates(adata)
pc.tl.assign_archetypes(adata, percentage_per_archetype=0.15)

# 4. Statistical analysis
gene_results = pc.tl.gene_associations(adata, fdr_scope='global')

# 5. Visualize
pc.pl.archetypal_space(adata, color_by='archetypes').show()
pc.pl.dotplot(gene_results, top_n_per_group=10).show()
```

**Complete workflows**: See [docs/INDEX.md](docs/INDEX.md) for detailed workflow guides.

---

## Key Features

### Advanced Deep Learning
- **Deep_AA** - Archetypal VAE with PCHA initialization and inflation scaling
- **Multiple loss functions** - Archetypal, diversity, regularity, sparsity, manifold
- **Hyperparameter optimization** - Cross-validation with intelligent grid search

### Statistical Framework
- **Gene/pathway associations** - Mann-Whitney U tests with FDR correction
- **Pattern analysis** - Exclusive, specialization, and tradeoff patterns
- **MSigDB integration** - 8 collections + OmniPath support
- **Conditional associations** - Metadata enrichment testing

### Visualization
- **3D archetypal space** - Interactive Plotly plots with gene expression coloring
- **Publication-ready dotplots** - Statistical results visualization
- **Training diagnostics** - Comprehensive metrics and stability analysis

### Trajectory Analysis
- **CellRank integration** - One-line setup with `pc.tl.setup_cellrank()`
- **Lineage pseudotimes** - Automatic conversion from fate probabilities
- **Driver gene identification** - Correlation-based methods

---

## Documentation

### Tutorials

| Tutorial | Description |
|----------|-------------|
| [01_data_loading](docs/tutorials/01_data_loading.ipynb) | Data loading and preprocessing |
| [02_hyperparameter_search](docs/tutorials/02_hyperparameter_search.ipynb) | Cross-validation and model selection |
| [03_model_training](docs/tutorials/03_model_training.ipynb) | Training archetypal models |
| [04_archetype_coordinates](docs/tutorials/04_archetype_coordinates.ipynb) | Coordinate extraction and cell assignment |
| [05_gene_enrichment](docs/tutorials/05_gene_enrichment.ipynb) | Gene and pathway associations |
| [06_cellrank_integration](docs/tutorials/06_cellrank_integration.ipynb) | Trajectory analysis with CellRank |
| [08_visualization](docs/tutorials/08_visualization.ipynb) | Comprehensive visualization guide |

### API Reference

See the [full documentation](https://scpeach.readthedocs.io/) for complete API reference.

---

## Agentic Usage & Tool Schemas

PEACH provides structured schemas for AI agent integration and programmatic tool discovery.

### Ground Truth Files

| File | Purpose | Use Case |
|------|---------|----------|
| `src/peach/_core/types_index.py` | Function → return types | Know what a function returns |
| `src/peach/_core/tools_schema.py` | Function → parameters | Know how to call a function |
| `src/peach/_core/types.py` | Pydantic type definitions | Full type validation |

### Programmatic Lookup

```python
from peach._core.types_index import get_return_type, ADATA_KEYS, USE_GET_FOR
from peach._core.tools_schema import get_tool_schema

# Get return type and keys for any function
ret_type, keys = get_return_type("tl.train_archetypal")
# → ("TrainingResults", ["history", "model", "archetype_coordinates", ...])

# Get full parameter schema
schema = get_tool_schema("tl.train_archetypal")
print([p.name for p in schema.parameters])
# → ["adata", "n_archetypes", "n_epochs", "device", ...]

# Check which keys require .get() access (optional fields)
print(USE_GET_FOR)
# → {"final_archetype_r2", "best_epoch", ...}
```

### Composable Workflows

The workflow scripts in `docs/workflows/` demonstrate validated, composable patterns:

```
WORKFLOW_01 → WORKFLOW_02 → WORKFLOW_03 → WORKFLOW_05 → WORKFLOW_06
(data load)   (hyperparam)   (training)    (enrichment)   (cellrank)
```

Each workflow script can be adapted for custom pipelines. See [docs/INDEX.md](docs/INDEX.md) for the complete list.

---

## Dependencies

**Core:**
- `torch >= 2.0.0`
- `anndata >= 0.8.0`
- `scanpy >= 1.9.0`
- `cellrank >= 2.0.0`
- `pandas >= 2.0.0`
- `numpy >= 1.24.0`
- `plotly >= 5.0.0`
- `pydantic >= 2.0.0`

---

## Examples

### Gene Association Testing

```python
# Test gene-archetype associations
gene_results = pc.tl.gene_associations(
    adata,
    fdr_scope='global',  # Global FDR correction
    min_logfc=0.1        # Minimum log-fold change
)

# Filter significant results
sig_genes = gene_results[gene_results['significant']]

# Visualize
pc.pl.dotplot(gene_results, top_n_per_group=10).show()
```

### Pathway Analysis

```python
# Load pathways
pathways = pc.pp.load_pathway_networks(
    sources=['hallmark', 'c5_bp'],
    organism='human'
)

# Compute activity scores
pc.pp.compute_pathway_scores(adata, net=pathways)

# Test associations
pathway_results = pc.tl.pathway_associations(adata, fdr_scope='global')

# Visualize
pc.pl.dotplot(pathway_results, top_n_per_group=10).show()
```

### Pattern Analysis

```python
# Find archetype-exclusive features
exclusive = pc.tl.archetype_exclusive_patterns(
    adata,
    min_effect_size=0.05
)

# Find mutual exclusivity
tradeoffs = pc.tl.tradeoff_patterns(
    adata,
    tradeoffs='pairs',
    min_effect_size=0.1
)
```

### Hyperparameter Optimization

```python
# Automated grid search
cv_summary = pc.tl.hyperparameter_search(
    adata,
    n_archetypes_range=[3, 4, 5, 6],
    cv_folds=3,
    max_epochs_cv=15
)

# Visualize results
pc.pl.elbow_curve(cv_summary)

# Get best configuration
top_configs = cv_summary.rank_by_metric('archetype_r2')
print(top_configs[0])
```

---

## Known Issues

### Apple Silicon (MPS) GPU Support

PyTorch's MPS backend can be unstable. Explicitly use CPU for reliable execution:

```python
results = pc.tl.train_archetypal(
    adata,
    n_archetypes=5,
    device='cpu'  # Explicitly use CPU
)
```

---

## Citation

If you use PEACH in your research, please cite:

```bibtex
@article{peach2025,
  title={Python Encoders for Archetypal Convex Hulls (PEACH): PyTorch-Based Archetypal Analysis},
  author={Honkala, Alexander; Malhotra, Sanjay},
  journal={bioRxiv},
  year={2025}
}
```

**Theoretical foundation:**
```bibtex
@article{cutler1994archetypal,
  title={Archetypal analysis},
  author={Cutler, Adele and Breiman, Leo},
  journal={Technometrics},
  volume={36},
  number={4},
  pages={338--347},
  year={1994}
}
```

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Support

- **Documentation**: [scpeach.readthedocs.io](https://scpeach.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/xhonkala/PEACH/issues)
- **Discussions**: [GitHub Discussions](https://github.com/xhonkala/PEACH/discussions)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Version**: 0.3.0 | **Author**: Alexander Honkala


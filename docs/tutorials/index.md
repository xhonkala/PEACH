# Tutorials

Learn how to use PEACH for archetypal analysis of single-cell data through comprehensive tutorials.

## Core Workflows

```{toctree}
:maxdepth: 1

01_data_loading
02_hyperparameter_search
03_model_training
04_archetype_coordinates
05_gene_enrichment
06_cellrank_integration
08_visualization
```

## Tutorial Overview

### 01 - Data Loading & Preprocessing
Load single-cell data, perform quality control, and prepare for archetypal analysis.

### 02 - Hyperparameter Search
Optimize model architecture using cross-validation and grid search.

### 03 - Model Training
Train the Deep Archetypal Analysis model with optimal parameters.

### 04 - Archetype Coordinates
Extract cell-archetype distances, weights, and assign cells to archetypes.

### 05 - Gene Enrichment
Perform differential expression and pathway enrichment analysis per archetype.

### 06 - CellRank Integration
Integrate with CellRank for lineage tracing and trajectory analysis.

### 08 - Visualization
Create publication-ready 3D archetypal space plots and statistical visualizations.

## Getting Started

Each tutorial is self-contained and can be run as a Jupyter notebook. Start with Tutorial 01 for a complete walkthrough, or jump to specific topics as needed.

```bash
# Install peach
pip install peach

# Run tutorials
jupyter lab docs/tutorials/
```

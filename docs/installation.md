# Installation

## Requirements

Peach requires Python 3.10 or later and is compatible with the scverse ecosystem.

### Core Dependencies

- **anndata** >=0.8.0 - Single-cell data structures
- **scanpy** >=1.9.0 - Single-cell analysis toolkit  
- **torch** >=2.0.0 - Deep learning framework
- **scikit-learn** >=1.0.0 - Machine learning utilities
- **pandas** >=2.0.0 - Data manipulation
- **numpy** >=1.24.0 - Numerical computing
- **plotly** >=5.0.0 - Interactive visualization
- **scipy** >=1.10.0 - Scientific computing
- **statsmodels** >=0.14.0 - Statistical analysis

## Installation Options

### Option 1: PyPI (Recommended)

```bash
pip install peach
```

See included environment.yaml file to create a working conda env for this package

### Option 2: Development Installation

For the latest features and development:

```bash
git clone https://github.com/xhonkala/PEACH.git
cd peach
pip install -e ".[dev]"
```

### Option 3: Conda Environment (Recommended)

Use the provided environment file:

```bash
git clone https://github.com/xhonkala/PEACH.git
cd PEACH
conda env create -f environment.yaml
conda activate peach
pip install -e .
```

## Verify Installation

Test your installation:

```python
import peach as pc
print(pc.__version__)

# Quick test with synthetic data
adata = pc.pp.generate_synthetic(n_points=100, n_dimensions=50, n_archetypes=3)
results = pc.tl.train_archetypal(adata, n_archetypes=3, n_epochs=5)
print(f"Test R²: {results['history']['archetype_r2'][-1]:.3f}")
```

## GPU Support (Optional)

For GPU acceleration with large datasets:

NB: this is not yet fully supported but will be included in the next release.

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed with correct versions
2. **CUDA errors**: Make sure PyTorch CUDA version matches your system
3. **Memory errors**: Consider using CPU for large datasets or reducing batch size

### Getting Help

- **GitHub Issues**: [Report bugs](https://github.com/yourusername/peach/issues)
- **Discussions**: [Ask questions](https://github.com/yourusername/peach/discussions)
- **scverse Discord**: Join the broader community

### System Requirements

- **Memory**: Minimum 4GB RAM, 16GB+ recommended for large datasets
- **Storage**: ~1GB for installation plus data storage
- **CPU**: Multi-core recommended for optimal performance
- **GPU**: Optional but recommended for datasets >10k cells
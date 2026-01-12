"""Shared fixtures for peach tests."""

import pytest
import numpy as np
import pandas as pd
from anndata import AnnData


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")


@pytest.fixture
def sample_adata():
    """Create sample AnnData for testing."""
    np.random.seed(42)
    n_obs, n_vars = 1000, 2000
    X = np.random.negative_binomial(5, 0.3, (n_obs, n_vars))
    
    adata = AnnData(X=X)
    # Use real gene symbols that exist in pathway databases
    real_genes = [
        'ACTB', 'GAPDH', 'TP53', 'BRCA1', 'EGFR', 'MYC', 'PIK3CA', 'KRAS', 'AKT1', 'MTOR',
        'ERBB2', 'CDK2', 'CCND1', 'RB1', 'CDKN1A', 'BCL2', 'BAX', 'APAF1', 'CASP3', 'PTEN',
        'TNF', 'IL6', 'IFNG', 'STAT3', 'NF1', 'MDM2', 'CHEK2', 'ATM', 'FOXO1', 'ESR1',
        'AR', 'VEGFA', 'HIF1A', 'TGFB1', 'SMAD4', 'WNT3A', 'CTNNB1', 'APC', 'GSK3B', 'DVL1',
        'NOTCH1', 'HES1', 'JAG1', 'DLL1', 'RBPJ', 'MAML1', 'FGF2', 'FGFR1', 'SHH', 'GLI1'
    ]
    # Repeat gene list to fill n_vars, cycling through real genes
    gene_names = [real_genes[i % len(real_genes)] + f"_{i//len(real_genes)}" if i >= len(real_genes) else real_genes[i] for i in range(n_vars)]
    adata.var_names = gene_names
    adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]
    
    # Add PCA coordinates
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50)
    adata.obsm['X_pca'] = pca.fit_transform(X.astype(np.float32))
    
    return adata


@pytest.fixture
def small_adata():
    """Create small AnnData for quick tests."""
    np.random.seed(42)
    n_obs, n_vars = 100, 50
    X = np.random.negative_binomial(3, 0.3, (n_obs, n_vars))
    
    adata = AnnData(X=X)
    # Use real gene symbols for pathway testing compatibility
    real_genes = [
        'ACTB', 'GAPDH', 'TP53', 'BRCA1', 'EGFR', 'MYC', 'PIK3CA', 'KRAS', 'AKT1', 'MTOR',
        'ERBB2', 'CDK2', 'CCND1', 'RB1', 'CDKN1A', 'BCL2', 'BAX', 'APAF1', 'CASP3', 'PTEN',
        'TNF', 'IL6', 'IFNG', 'STAT3', 'NF1', 'MDM2', 'CHEK2', 'ATM', 'FOXO1', 'ESR1',
        'AR', 'VEGFA', 'HIF1A', 'TGFB1', 'SMAD4', 'WNT3A', 'CTNNB1', 'APC', 'GSK3B', 'DVL1',
        'NOTCH1', 'HES1', 'JAG1', 'DLL1', 'RBPJ', 'MAML1', 'FGF2', 'FGFR1', 'SHH', 'GLI1'
    ]
    # Use real genes for small dataset, cycling through if needed
    gene_names = [real_genes[i % len(real_genes)] for i in range(n_vars)]
    adata.var_names = gene_names
    adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]
    
    # Add PCA coordinates (just use the data itself for small example)
    adata.obsm['X_pca'] = X.astype(np.float32)
    
    return adata


@pytest.fixture
def synthetic_adata():
    """Create synthetic archetypal data for recovery tests."""
    import peach as pc
    
    # Generate synthetic data with known archetypal structure
    return pc.pp.generate_synthetic(
        n_samples=150,
        n_features=25,
        n_archetypes=3,
        noise_std=0.05
    )


@pytest.fixture
def trained_small_adata(small_adata):
    """Create small AnnData with trained model for testing downstream functions."""
    import peach as pc
    
    # Train model
    pc.tl.train_archetypal(small_adata, n_archetypes=3, n_epochs=5)
    pc.tl.archetypal_coordinates(small_adata)
    pc.tl.assign_archetypes(small_adata, percentage_per_archetype=0.2)
    
    return small_adata
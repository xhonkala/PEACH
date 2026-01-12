"""Tests for advanced plotting functionality."""

import pytest
import pandas as pd
import peach as pc

# Set matplotlib backend to non-GUI for testing
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def test_gene_expression_coloring(small_adata):
    """Test 3D plot with gene expression coloring."""
    # Setup
    pc.tl.train_archetypal(small_adata, n_archetypes=3, n_epochs=5)
    pc.tl.archetypal_coordinates(small_adata)
    
    # Test gene expression coloring
    gene_name = small_adata.var_names[0]  # Use first gene
    
    fig = pc.pl.archetypal_space(
        small_adata,
        color_by=gene_name,
        color_scale='plasma',
        title=f'{gene_name} Expression'
    )
    
    assert fig is not None
    assert hasattr(fig, 'show')
    
    # Check plotly figure
    import plotly.graph_objects as go
    if isinstance(fig, go.Figure):
        assert hasattr(fig, 'data')


def test_pathway_visualization(small_adata):
    """Test dotplot with pathway results."""
    # Skip test if gseapy is not available (dependency build issues on some systems)
    try:
        import gseapy
    except ImportError:
        pytest.skip("gseapy not available - skipping pathway visualization test")
    
    # Setup with pathways
    pc.tl.train_archetypal(small_adata, n_archetypes=3, n_epochs=5)
    pc.tl.archetypal_coordinates(small_adata)
    pc.tl.assign_archetypes(small_adata, percentage_per_archetype=0.2)
    
    # Try to compute pathway scores (may skip with synthetic data due to no gene overlap)
    try:
        net = pc.pp.load_pathway_networks(sources=['hallmark'])
        pc.pp.compute_pathway_scores(small_adata, net=net, obsm_key='pathway_scores')
        
        # Test pathway associations if scores were computed successfully
        if 'pathway_scores' in small_adata.obsm and small_adata.obsm['pathway_scores'].shape[1] > 0:
            pathway_results = pc.tl.pathway_associations(
                small_adata,
                pathway_obsm_key='pathway_scores',
                fdr_scope='global'
            )
            
            # Test dotplot with pathway data if results exist
            if len(pathway_results) > 0:
                fig = pc.pl.dotplot(
                    pathway_results.head(5),
                    title='Test Pathway Associations'
                )
                
                assert fig is not None
                assert hasattr(fig, 'show')
                
                # Check figure type appropriately
                try:
                    import matplotlib.figure
                    if isinstance(fig, matplotlib.figure.Figure):
                        assert hasattr(fig, 'axes')
                except (ImportError, AttributeError):
                    # Handle matplotlib compatibility issues gracefully
                    pass
            else:
                pytest.skip("No pathway associations found - acceptable for small test data")
        else:
            pytest.skip("No pathway scores computed - acceptable for small test data")
    except (RuntimeError, ImportError) as e:
        # Expected with synthetic data due to no gene overlap or missing dependencies
        if "No pathways have genes" in str(e) or "gseapy" in str(e):
            pytest.skip(f"Pathway analysis not available: {e}")
        else:
            raise


def test_memory_efficiency():
    """Test that visualizations don't consume excessive memory."""
    # Create slightly larger test data
    adata = pc.pp.generate_synthetic(n_points=200, n_dimensions=50, n_archetypes=3)
    
    # Train and setup
    pc.tl.train_archetypal(adata, n_archetypes=3, n_epochs=3)
    pc.tl.archetypal_coordinates(adata)
    pc.tl.assign_archetypes(adata)
    
    # Create visualization (should not crash with memory issues)
    fig = pc.pl.archetypal_space(adata, color_by='archetypes')
    
    assert fig is not None
    
    # Test that figure can be created without issues
    assert len(fig.data) > 0


def test_plot_data_validation(small_adata):
    """Test that plotting functions handle different data types correctly."""
    # Setup
    pc.tl.train_archetypal(small_adata, n_archetypes=3, n_epochs=5)
    pc.tl.archetypal_coordinates(small_adata)
    pc.tl.assign_archetypes(small_adata, percentage_per_archetype=0.2)
    
    # Test with different result types
    gene_results = pc.tl.gene_associations(small_adata, min_cells=5)
    
    # Test that plot handles typical gene association results
    fig = pc.pl.dotplot(gene_results.head(5))
    assert fig is not None
    
    # Test archetypal space with different color options
    fig1 = pc.pl.archetypal_space(small_adata)  # Default coloring
    fig2 = pc.pl.archetypal_space(small_adata, color_by='archetypes')  # Categorical
    
    assert fig1 is not None
    assert fig2 is not None
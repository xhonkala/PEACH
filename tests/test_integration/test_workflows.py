"""Integration tests for complete end-to-end workflows."""

import pytest
import numpy as np
import peach as pc

# Set matplotlib backend to non-GUI for testing
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


@pytest.mark.integration
def test_basic_archetypal_analysis_workflow(small_adata):
    """Test complete basic archetypal analysis workflow."""
    # Step 1: Train archetypal model
    results = pc.tl.train_archetypal(
        small_adata,
        n_archetypes=3,
        n_epochs=8,
        model_config={'archetypal_weight': 1.0, 'kld_weight': 0.0}
    )
    
    # Validate training results
    assert 'history' in results
    assert 'archetype_coordinates' in small_adata.uns
    
    # Step 2: Extract archetypal coordinates
    pc.tl.archetypal_coordinates(small_adata)
    assert 'archetype_distances' in small_adata.obsm
    
    # Step 3: Assign cells to archetypes
    pc.tl.assign_archetypes(small_adata, percentage_per_archetype=0.15)
    assert 'archetypes' in small_adata.obs
    
    # Step 4: Gene association analysis
    gene_results = pc.tl.gene_associations(
        small_adata,
        fdr_scope='global',
        min_cells=5
    )
    assert len(gene_results) > 0
    assert 'gene' in gene_results.columns
    assert 'significant' in gene_results.columns
    
    # Step 5: Create visualization
    fig = pc.pl.archetypal_space(small_adata, color_by='archetypes')
    assert fig is not None
    assert hasattr(fig, 'show')


@pytest.mark.integration
def test_pathway_analysis_workflow(small_adata):
    """Test complete pathway analysis workflow with real gene names."""
    # Skip test if gseapy is not available (dependency build issues on some systems)
    try:
        import gseapy
    except ImportError:
        pytest.skip("gseapy not available - skipping pathway workflow test")
        
    # Step 1: Train archetypal model
    pc.tl.train_archetypal(small_adata, n_archetypes=3, n_epochs=8)
    pc.tl.archetypal_coordinates(small_adata)
    pc.tl.assign_archetypes(small_adata, percentage_per_archetype=0.15)
    
    # Step 2: Pathway analysis (with graceful handling of gene overlap)
    try:
        # Load and compute pathway scores
        # Load hallmark pathways first
        net = pc.pp.load_pathway_networks(sources=['hallmark'])
        pc.pp.compute_pathway_scores(
            small_adata,
            net=net,
            obsm_key='pathway_scores'
        )
        
        # Only proceed if pathway scores were computed
        if 'pathway_scores' in small_adata.obsm and small_adata.obsm['pathway_scores'].shape[1] > 0:
            # Step 3: Pathway association testing
            pathway_results = pc.tl.pathway_associations(
                small_adata,
                pathway_obsm_key='pathway_scores',
                fdr_scope='global'
            )
            
            assert len(pathway_results) > 0
            assert 'pathway' in pathway_results.columns
            assert 'pvalue' in pathway_results.columns
            
            # Step 4: Visualization
            if len(pathway_results) > 0:
                fig = pc.pl.dotplot(pathway_results.head(5))
                assert fig is not None
        else:
            pytest.skip("No pathway overlap with dataset - acceptable for test data")
            
    except (RuntimeError, ImportError) as e:
        if "No pathways have genes" in str(e) or "gseapy" in str(e):
            pytest.skip(f"Pathway analysis not available: {e}")
        else:
            raise


@pytest.mark.integration
def test_hyperparameter_search_workflow():
    """Test complete hyperparameter search workflow."""
    # Step 1: Generate test data
    adata = pc.pp.generate_synthetic(
        n_points=150,
        n_dimensions=25,
        n_archetypes=3,
        noise=0.1
    )
    
    # Step 2: Prepare training data
    dataloader = pc.pp.prepare_training(adata, batch_size=32)
    
    # Step 3: Configure and run hyperparameter search
    from peach._core.utils.hyperparameter_search import SearchConfig, ArchetypalGridSearch
    
    search_config = SearchConfig(
        n_archetypes_range=[2, 3],
        cv_folds=2,
        max_epochs_cv=3,
        max_cells_cv=100  # Limit for speed
    )
    
    base_model_config = {
        'archetypal_weight': 1.0,
        'kld_weight': 0.0
    }
    
    # Step 4: Run search
    grid_search = ArchetypalGridSearch(search_config)
    cv_summary = grid_search.fit(dataloader, base_model_config)
    
    # Step 5: Validate results and decision support
    assert hasattr(cv_summary, 'summary_df')
    assert len(cv_summary.summary_df) > 0
    
    # Test ranking functionality
    ranked = cv_summary.rank_by_metric('archetype_r2')
    assert len(ranked) > 0
    assert 'config_summary' in ranked[0]
    
    # Test summary report
    report = cv_summary.summary_report()
    assert isinstance(report, str)
    assert len(report) > 50  # Should be substantial


@pytest.mark.integration
@pytest.mark.slow
def test_complete_analysis_with_visualization(small_adata):
    """Test complete analysis pipeline with all visualization types."""
    # Skip pathway components if gseapy not available
    try:
        import gseapy
        gseapy_available = True
    except ImportError:
        gseapy_available = False
        
    # Step 1: Complete archetypal analysis
    results = pc.tl.train_archetypal(
        small_adata,
        n_archetypes=3,
        n_epochs=10
    )
    
    pc.tl.archetypal_coordinates(small_adata)
    pc.tl.assign_archetypes(small_adata, percentage_per_archetype=0.1)
    
    # Step 2: Statistical analyses
    gene_results = pc.tl.gene_associations(small_adata, min_cells=5)
    
    # Step 3: Pattern analysis (add pathway scores first)
    small_adata.obsm['pathway_scores'] = np.random.rand(small_adata.n_obs, 20)  # More features
    try:
        pattern_results = pc.tl.pattern_analysis(
            small_adata,
            data_obsm_key='pathway_scores',
            verbose=False
        )
        assert isinstance(pattern_results, dict)
        assert 'individual' in pattern_results
    except ValueError as e:
        # May fail with "No valid pattern tests" or "minimum cell requirement" on small synthetic data
        acceptable_errors = [
            "No valid pattern tests",
            "minimum cell requirement",
            "No archetype bins meet",
        ]
        if any(err in str(e) for err in acceptable_errors):
            pass  # Acceptable for small test data
        else:
            raise

    # Step 4: Multiple visualization types

    # Training metrics (display=False to get figure object)
    training_fig = pc.pl.training_metrics(results['history'], display=False)
    assert training_fig is not None
    
    # Archetypal space (default)
    space_fig1 = pc.pl.archetypal_space(small_adata)
    assert space_fig1 is not None
    
    # Archetypal space with archetype coloring
    space_fig2 = pc.pl.archetypal_space(small_adata, color_by='archetypes')
    assert space_fig2 is not None
    
    # Gene expression coloring
    gene_name = small_adata.var_names[0]
    space_fig3 = pc.pl.archetypal_space(
        small_adata,
        color_by=gene_name,
        color_scale='viridis'
    )
    assert space_fig3 is not None
    
    # Statistical results dotplot
    dotplot_fig = pc.pl.dotplot(gene_results.head(8))
    assert dotplot_fig is not None
    
    # Step 5: Verify all results are coherent
    # Check that archetypes are consistent across analyses
    n_assigned_cells = (small_adata.obs['archetypes'] != 'unassigned').sum()
    assert n_assigned_cells > 0, "No cells were assigned to archetypes"
    
    # Check that gene results include assigned archetypes
    result_archetypes = set(gene_results['archetype'].unique())
    obs_archetypes = set(small_adata.obs['archetypes'].unique())
    # Should have overlap (minus 'unassigned')
    assigned_archetypes = obs_archetypes - {'unassigned'}
    assert len(result_archetypes & assigned_archetypes) > 0, "No archetype overlap between results"
    
    # All visualizations should be created successfully
    figures = [training_fig, space_fig1, space_fig2, space_fig3, dotplot_fig]
    for fig in figures:
        assert fig is not None
        assert hasattr(fig, 'show')
"""Tests for hyperparameter search functionality."""

import pytest
import numpy as np
import peach as pc
from peach._core.utils.hyperparameter_search import SearchConfig


def test_search_config_validation():
    """Test SearchConfig parameter validation."""
    # Test valid configuration
    config = SearchConfig(
        n_archetypes_range=[2, 3, 4],
        cv_folds=3,
        max_epochs_cv=10
    )
    
    # Validate required attributes exist
    assert hasattr(config, 'n_archetypes_range')
    assert hasattr(config, 'cv_folds')
    assert hasattr(config, 'max_epochs_cv')
    
    # Test invalid configuration should raise error
    with pytest.raises(ValueError):
        SearchConfig(
            n_archetypes_range=[],  # Empty range
            cv_folds=1  # Too few folds
        )


def test_grid_search_small(small_adata):
    """Test basic grid search functionality with small dataset."""
    from peach._core.utils.hyperparameter_search import ArchetypalGridSearch
    
    # Configure search
    search_config = SearchConfig(
        n_archetypes_range=[2, 3],
        hidden_dims_options=[[32, 16]],  # Single architecture for controlled testing
        cv_folds=2,
        max_epochs_cv=3  # Very short for testing
    )
    
    # Create grid search
    grid_search = ArchetypalGridSearch(search_config)
    
    # Prepare data
    from peach.pp import prepare_training
    dataloader = prepare_training(small_adata, batch_size=32)
    
    # Run search (this will take a moment)
    base_model_config = {
        'archetypal_weight': 1.0,
        'kld_weight': 0.0
    }
    
    cv_summary = grid_search.fit(dataloader, base_model_config)
    
    # Validate results
    assert cv_summary is not None
    assert hasattr(cv_summary, 'summary_df')
    assert len(cv_summary.summary_df) == 2  # 2 configurations tested
    
    # Check that results contain expected metrics
    assert 'mean_archetype_r2' in cv_summary.summary_df.columns
    assert 'std_archetype_r2' in cv_summary.summary_df.columns
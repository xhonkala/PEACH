"""
Tests for CV results visualization functions in grid_search_results.py

Tests the new modular plotting functions:
- plot_elbow_r2() - Primary R² visualization
- plot_metric() - Generic metric visualization
- _plot_metric_elbow() - Core implementation
- _map_metric_name() - Metric name mapping
- _detect_facet_strategy() - Faceting strategy detection
"""

import pytest
import numpy as np
import pandas as pd
import warnings
from peach._core.utils.grid_search_results import CVResults, CVSummary, METRIC_MAP


@pytest.fixture
def sample_cv_results_basic():
    """Create basic CV results without inflation factor."""
    cv_results = []

    for n_arch in [3, 4, 5]:
        for hidden_dims in [[128, 64], [256, 128, 64]]:
            # Simulate fold results
            fold_results = []
            for fold in range(3):
                fold_results.append({
                    'train_rmse': 0.5 + np.random.rand() * 0.1,
                    'val_rmse': 0.6 + np.random.rand() * 0.1,
                    'archetype_r2': 0.75 + np.random.rand() * 0.15,
                    'convergence_epoch': 20 + np.random.randint(10),
                    'early_stopped': False
                })

            # Aggregate metrics
            mean_metrics = {
                'train_rmse': np.mean([f['train_rmse'] for f in fold_results]),
                'val_rmse': np.mean([f['val_rmse'] for f in fold_results]),
                'archetype_r2': np.mean([f['archetype_r2'] for f in fold_results])
            }
            std_metrics = {
                'train_rmse': np.std([f['train_rmse'] for f in fold_results]),
                'val_rmse': np.std([f['val_rmse'] for f in fold_results]),
                'archetype_r2': np.std([f['archetype_r2'] for f in fold_results])
            }

            cv_result = CVResults(
                hyperparameters={
                    'n_archetypes': n_arch,
                    'hidden_dims': hidden_dims
                },
                fold_results=fold_results,
                mean_metrics=mean_metrics,
                std_metrics=std_metrics,
                best_fold_idx=0,
                convergence_epochs=[20, 22, 21]
            )
            cv_results.append(cv_result)

    return cv_results


@pytest.fixture
def sample_cv_results_with_inflation():
    """Create CV results WITH inflation factor (3D search)."""
    cv_results = []

    for n_arch in [3, 4, 5]:
        for hidden_dims in [[128, 64], [256, 128, 64]]:
            for inflation in [1.0, 1.5, 2.0]:
                # Simulate fold results (R² increases with inflation up to 1.5)
                base_r2 = 0.75 + (n_arch - 3) * 0.03
                inflation_bonus = 0.05 if inflation == 1.5 else 0.0

                fold_results = []
                for fold in range(3):
                    fold_results.append({
                        'train_rmse': 0.5 + np.random.rand() * 0.1,
                        'val_rmse': 0.6 + np.random.rand() * 0.1,
                        'archetype_r2': base_r2 + inflation_bonus + np.random.rand() * 0.05,
                        'convergence_epoch': 20 + np.random.randint(10),
                        'early_stopped': False
                    })

                # Aggregate metrics
                mean_metrics = {
                    'train_rmse': np.mean([f['train_rmse'] for f in fold_results]),
                    'val_rmse': np.mean([f['val_rmse'] for f in fold_results]),
                    'archetype_r2': np.mean([f['archetype_r2'] for f in fold_results])
                }
                std_metrics = {
                    'train_rmse': np.std([f['train_rmse'] for f in fold_results]),
                    'val_rmse': np.std([f['val_rmse'] for f in fold_results]),
                    'archetype_r2': np.std([f['archetype_r2'] for f in fold_results])
                }

                cv_result = CVResults(
                    hyperparameters={
                        'n_archetypes': n_arch,
                        'hidden_dims': hidden_dims,
                        'inflation_factor': inflation
                    },
                    fold_results=fold_results,
                    mean_metrics=mean_metrics,
                    std_metrics=std_metrics,
                    best_fold_idx=0,
                    convergence_epochs=[20, 22, 21]
                )
                cv_results.append(cv_result)

    return cv_results


@pytest.fixture
def cv_summary_basic(sample_cv_results_basic):
    """Create CVSummary from basic CV results."""
    from dataclasses import dataclass

    @dataclass
    class MockSearchConfig:
        cv_folds: int = 3

    return CVSummary.from_cv_results(
        sample_cv_results_basic,
        search_config=MockSearchConfig(),
        data_info={'n_total_samples': 1000, 'n_features': 2000}
    )


@pytest.fixture
def cv_summary_with_inflation(sample_cv_results_with_inflation):
    """Create CVSummary with inflation factor."""
    from dataclasses import dataclass

    @dataclass
    class MockSearchConfig:
        cv_folds: int = 3

    return CVSummary.from_cv_results(
        sample_cv_results_with_inflation,
        search_config=MockSearchConfig(),
        data_info={'n_total_samples': 1000, 'n_features': 2000}
    )


# ==================== Test METRIC_MAP ====================

def test_metric_map_exists():
    """Test that METRIC_MAP constant exists and has expected mappings."""
    assert 'rmse' in METRIC_MAP
    assert 'r2' in METRIC_MAP
    assert 'archetype_r2' in METRIC_MAP
    assert METRIC_MAP['rmse'] == 'mean_val_rmse'
    assert METRIC_MAP['r2'] == 'mean_archetype_r2'
    assert METRIC_MAP['archetype_r2'] == 'mean_archetype_r2'


# ==================== Test _map_metric_name ====================

def test_map_metric_name_exact_match(cv_summary_basic):
    """Test metric name mapping with exact column match."""
    # Exact match should return the same name
    result = cv_summary_basic._map_metric_name('mean_archetype_r2')
    assert result == 'mean_archetype_r2'


def test_map_metric_name_with_mapping(cv_summary_basic):
    """Test metric name mapping using METRIC_MAP."""
    # User-friendly name should map to DataFrame column
    result = cv_summary_basic._map_metric_name('rmse')
    assert result == 'mean_val_rmse'

    result = cv_summary_basic._map_metric_name('r2')
    assert result == 'mean_archetype_r2'


def test_map_metric_name_with_prefix(cv_summary_basic):
    """Test metric name mapping by adding mean_ prefix."""
    result = cv_summary_basic._map_metric_name('val_rmse')
    assert result == 'mean_val_rmse'


def test_map_metric_name_not_found(cv_summary_basic):
    """Test metric name mapping returns original if not found."""
    # Should return original name (will error downstream)
    result = cv_summary_basic._map_metric_name('nonexistent_metric')
    assert result == 'nonexistent_metric'


# ==================== Test _detect_facet_strategy ====================

def test_detect_facet_strategy_both(cv_summary_with_inflation):
    """Test facet strategy detection with both inflation and hidden_dims."""
    plot_data = cv_summary_with_inflation.get_plot_data()
    elbow_df = plot_data['elbow']

    strategy = cv_summary_with_inflation._detect_facet_strategy(elbow_df)

    assert strategy['facet_by'] == 'inflation_factor'
    assert strategy['color_by'] == 'hidden_dims'
    assert strategy['facet_label'] == 'Inflation Factor (λ)'
    assert strategy['color_label'] == 'Architecture'
    assert strategy['n_facets'] == 3  # Three inflation values


def test_detect_facet_strategy_hidden_only(cv_summary_basic):
    """Test facet strategy detection with only hidden_dims."""
    plot_data = cv_summary_basic.get_plot_data()
    elbow_df = plot_data['elbow']

    strategy = cv_summary_basic._detect_facet_strategy(elbow_df)

    assert strategy['facet_by'] is None
    assert strategy['color_by'] == 'hidden_dims'
    assert strategy['color_label'] == 'Architecture'
    assert strategy['n_facets'] == 1


def test_detect_facet_strategy_inflation_only():
    """Test facet strategy detection with only inflation_factor."""
    # Create DataFrame with only inflation (no hidden_dims variation)
    elbow_df = pd.DataFrame({
        'n_archetypes': [3, 4, 5] * 3,
        'inflation_factor': [1.0] * 3 + [1.5] * 3 + [2.0] * 3,
        'mean_archetype_r2': np.random.rand(9)
    })

    from dataclasses import dataclass
    @dataclass
    class MockSearchConfig:
        cv_folds: int = 3

    # Create minimal CVSummary
    cv_results = []
    cv_summary = CVSummary(
        config_results={},
        summary_df=elbow_df,
        ranked_configs=[],
        cv_info={},
        search_config=MockSearchConfig()
    )

    strategy = cv_summary._detect_facet_strategy(elbow_df)

    assert strategy['facet_by'] == 'inflation_factor'
    assert strategy['color_by'] is None
    assert strategy['n_facets'] == 3


# ==================== Test plot_elbow_r2 ====================

def test_plot_elbow_r2_basic(cv_summary_basic):
    """Test plot_elbow_r2() with basic CV results (no inflation)."""
    fig = cv_summary_basic.plot_elbow_r2()

    assert fig is not None
    assert len(fig.data) > 0  # Should have traces
    assert fig.layout.title.text == "Archetype R² vs Number of Archetypes"


def test_plot_elbow_r2_with_inflation(cv_summary_with_inflation):
    """Test plot_elbow_r2() with inflation factor (faceted plot)."""
    fig = cv_summary_with_inflation.plot_elbow_r2()

    assert fig is not None
    assert len(fig.data) > 0
    # Should have 3 facets (3 inflation values) × 2 hidden_dims = 6 traces
    assert len(fig.data) == 6


def test_plot_elbow_r2_custom_dimensions(cv_summary_basic):
    """Test plot_elbow_r2() with custom height/width."""
    fig = cv_summary_basic.plot_elbow_r2(height=800, width=1000)

    assert fig.layout.height == 800
    assert fig.layout.width == 1000


# ==================== Test plot_metric ====================

def test_plot_metric_rmse(cv_summary_basic):
    """Test plot_metric() with RMSE metric."""
    fig = cv_summary_basic.plot_metric('rmse')

    assert fig is not None
    assert len(fig.data) > 0


def test_plot_metric_with_inflation(cv_summary_with_inflation):
    """Test plot_metric() with inflation factor."""
    fig = cv_summary_with_inflation.plot_metric('val_rmse')

    assert fig is not None
    assert len(fig.data) == 6  # 3 inflation × 2 hidden_dims


def test_plot_metric_friendly_names(cv_summary_basic):
    """Test plot_metric() accepts user-friendly metric names."""
    # All these should work via mapping
    fig1 = cv_summary_basic.plot_metric('rmse')
    fig2 = cv_summary_basic.plot_metric('val_rmse')
    fig3 = cv_summary_basic.plot_metric('r2')

    assert fig1 is not None
    assert fig2 is not None
    assert fig3 is not None


def test_plot_metric_invalid_metric(cv_summary_basic):
    """Test plot_metric() raises error for invalid metric."""
    with pytest.raises(ValueError, match="not found in columns"):
        cv_summary_basic.plot_metric('nonexistent_metric')


# ==================== Test _plot_metric_elbow ====================

def test_plot_metric_elbow_core(cv_summary_basic):
    """Test core _plot_metric_elbow() implementation."""
    fig = cv_summary_basic._plot_metric_elbow('archetype_r2')

    assert fig is not None
    assert len(fig.data) > 0


def test_plot_metric_elbow_custom_title(cv_summary_basic):
    """Test _plot_metric_elbow() with custom title."""
    fig = cv_summary_basic._plot_metric_elbow(
        'archetype_r2',
        title="Custom Title"
    )

    assert fig.layout.title.text == "Custom Title"


def test_plot_metric_elbow_empty_data():
    """Test _plot_metric_elbow() with empty data."""
    from dataclasses import dataclass

    @dataclass
    class MockSearchConfig:
        cv_folds: int = 3

    # Create CVSummary with empty data
    cv_summary = CVSummary(
        config_results={},
        summary_df=pd.DataFrame(),
        ranked_configs=[],
        cv_info={},
        search_config=MockSearchConfig()
    )

    fig = cv_summary._plot_metric_elbow('archetype_r2')

    # Should return empty figure with warning
    assert fig is not None
    assert len(fig.data) == 0


# ==================== Test plot_elbow_curve ====================

def test_plot_elbow_curve_still_works(cv_summary_basic):
    """Test that deprecated plot_elbow_curve() still functions."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = cv_summary_basic.plot_elbow_curve()

        assert fig is not None
        assert len(fig.data) > 0


# ==================== Integration tests ====================

def test_inflation_factor_in_plot_data(cv_summary_with_inflation):
    """Test that inflation_factor appears in plot data."""
    plot_data = cv_summary_with_inflation.get_plot_data()
    elbow_df = plot_data['elbow']

    assert 'inflation_factor' in elbow_df.columns
    assert elbow_df['inflation_factor'].nunique() == 3
    assert set(elbow_df['inflation_factor'].unique()) == {1.0, 1.5, 2.0}


def test_plot_data_grouping_with_inflation(cv_summary_with_inflation):
    """Test that plot data properly groups by inflation_factor."""
    plot_data = cv_summary_with_inflation.get_plot_data()
    elbow_df = plot_data['elbow']

    # Should have: 3 n_archetypes × 2 hidden_dims × 3 inflation = 18 rows
    assert len(elbow_df) == 18


def test_plot_data_grouping_without_inflation(cv_summary_basic):
    """Test that plot data groups without inflation_factor."""
    plot_data = cv_summary_basic.get_plot_data()
    elbow_df = plot_data['elbow']

    # Should NOT have inflation_factor column
    assert 'inflation_factor' not in elbow_df.columns

    # Should have: 3 n_archetypes × 2 hidden_dims = 6 rows
    assert len(elbow_df) == 6


def test_faceting_produces_correct_number_of_traces(cv_summary_with_inflation):
    """Test that faceting produces expected number of traces."""
    fig = cv_summary_with_inflation.plot_elbow_r2()

    # 3 inflation facets × 2 hidden_dims = 6 traces
    assert len(fig.data) == 6

    # Check legend groups (should show each hidden_dims only once)
    legend_names = [trace.name for trace in fig.data if trace.showlegend]
    assert len(legend_names) == 2  # Two hidden_dims architectures


def test_trace_colors_consistent(cv_summary_with_inflation):
    """Test that traces for same hidden_dims have same color across facets."""
    fig = cv_summary_with_inflation.plot_elbow_r2()

    # Get colors for each hidden_dims
    colors_by_hidden = {}
    for trace in fig.data:
        hidden = trace.name.split('=')[1]  # Extract from "Architecture=[...]"
        if hidden not in colors_by_hidden:
            colors_by_hidden[hidden] = trace.line.color
        else:
            # Same hidden_dims should have same color
            assert colors_by_hidden[hidden] == trace.line.color


# ==================== Edge cases ====================

def test_single_archetype_value():
    """Test plotting with only single n_archetypes value."""
    cv_results = []

    for hidden_dims in [[128, 64], [256, 128, 64]]:
        fold_results = [{'archetype_r2': 0.85, 'val_rmse': 0.5} for _ in range(3)]
        mean_metrics = {'archetype_r2': 0.85, 'val_rmse': 0.5}
        std_metrics = {'archetype_r2': 0.01, 'val_rmse': 0.02}

        cv_result = CVResults(
            hyperparameters={'n_archetypes': 5, 'hidden_dims': hidden_dims},
            fold_results=fold_results,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            best_fold_idx=0,
            convergence_epochs=[20, 20, 20]
        )
        cv_results.append(cv_result)

    from dataclasses import dataclass
    @dataclass
    class MockSearchConfig:
        cv_folds: int = 3

    cv_summary = CVSummary.from_cv_results(
        cv_results,
        search_config=MockSearchConfig(),
        data_info={'n_total_samples': 1000, 'n_features': 2000}
    )

    fig = cv_summary.plot_elbow_r2()
    assert fig is not None
    assert len(fig.data) > 0


def test_single_hidden_dims():
    """Test plotting with only single hidden_dims value."""
    cv_results = []

    for n_arch in [3, 4, 5]:
        fold_results = [{'archetype_r2': 0.75 + n_arch * 0.03, 'val_rmse': 0.5} for _ in range(3)]
        mean_metrics = {'archetype_r2': 0.75 + n_arch * 0.03, 'val_rmse': 0.5}
        std_metrics = {'archetype_r2': 0.01, 'val_rmse': 0.02}

        cv_result = CVResults(
            hyperparameters={'n_archetypes': n_arch, 'hidden_dims': [256, 128, 64]},
            fold_results=fold_results,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            best_fold_idx=0,
            convergence_epochs=[20, 20, 20]
        )
        cv_results.append(cv_result)

    from dataclasses import dataclass
    @dataclass
    class MockSearchConfig:
        cv_folds: int = 3

    cv_summary = CVSummary.from_cv_results(
        cv_results,
        search_config=MockSearchConfig(),
        data_info={'n_total_samples': 1000, 'n_features': 2000}
    )

    fig = cv_summary.plot_elbow_r2()
    assert fig is not None
    # Should have single trace
    assert len(fig.data) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

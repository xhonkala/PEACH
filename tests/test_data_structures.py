"""
Test to verify all data structures documented in docs/data_structures.md
This ensures the reference stays accurate as the codebase evolves.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from anndata import AnnData

import peach as pc
from peach import Deep_AA, calculate_archetype_r2, archetypal_R2


def test_model_forward_return_structure():
    """Verify Deep_AA.forward() returns correct dictionary structure."""
    model = Deep_AA(input_dim=13, n_archetypes=5, hidden_dims=[128, 64])
    X = torch.randn(10, 13)

    outputs = model.forward(X)

    # Check it's a dictionary
    assert isinstance(outputs, dict), "forward() should return dict"

    # Check primary keys
    assert 'arch_recons' in outputs
    assert 'mu' in outputs
    assert 'log_var' in outputs
    assert 'z' in outputs
    assert 'archetypes' in outputs
    assert 'input' in outputs

    # Check legacy keys
    assert 'recons' in outputs
    assert 'archetypal_coordinates' in outputs
    assert 'A' in outputs
    assert 'Y' in outputs

    # Check shapes
    assert outputs['arch_recons'].shape == (10, 13)
    assert outputs['mu'].shape == (10, 5)
    assert outputs['log_var'].shape == (10, 5)
    assert outputs['z'].shape == (10, 5)
    assert outputs['archetypes'].shape == (5, 13)

    # Check aliases point to same tensors
    assert torch.equal(outputs['arch_recons'], outputs['recons'])
    assert torch.equal(outputs['z'], outputs['A'])
    assert torch.equal(outputs['z'], outputs['archetypal_coordinates'])
    assert torch.equal(outputs['archetypes'], outputs['Y'])


def test_model_loss_function_return_structure():
    """Verify Deep_AA.loss_function() returns correct dictionary structure."""
    model = Deep_AA(input_dim=13, n_archetypes=5, hidden_dims=[128, 64])
    X = torch.randn(10, 13)

    outputs = model.forward(X)
    loss_dict = model.loss_function(outputs)

    # Check it's a dictionary
    assert isinstance(loss_dict, dict), "loss_function() should return dict"

    # Check primary loss (has gradients)
    assert 'loss' in loss_dict
    assert loss_dict['loss'].requires_grad, "Total loss should require gradients"

    # Check loss components (all detached)
    loss_components = ['kld_loss', 'archetypal_loss', 'diversity_loss',
                       'regularity_loss', 'sparsity_loss', 'manifold_loss']
    for component in loss_components:
        assert component in loss_dict
        assert not loss_dict[component].requires_grad, f"{component} should be detached"

    # Check performance metrics (all detached)
    assert 'rmse' in loss_dict
    assert 'archetype_r2' in loss_dict
    assert not loss_dict['rmse'].requires_grad
    assert not loss_dict['archetype_r2'].requires_grad

    # Check archetype usage metrics
    usage_metrics = ['archetype_entropy', 'max_archetype_usage',
                     'min_archetype_usage', 'active_archetypes_per_sample']
    for metric in usage_metrics:
        assert metric in loss_dict

    # Check manifold quality metrics
    assert 'mean_archetype_data_distance' in loss_dict
    assert 'max_archetype_data_distance' in loss_dict

    # Check convergence tracking
    assert 'loss_delta' in loss_dict
    assert 'loss_history' in loss_dict
    assert isinstance(loss_dict['loss_history'], list)

    # Check model info
    assert loss_dict['input_dim'] == 13
    assert loss_dict['latent_dim'] == 5
    assert loss_dict['n_archetypes'] == 5

    # Check legacy aliases
    assert 'KLD' in loss_dict
    assert 'reconstruction_loss' in loss_dict


def test_calculate_archetype_r2_return_type():
    """Verify calculate_archetype_r2() returns scalar tensor."""
    X_original = torch.randn(100, 13)
    X_recon = torch.randn(100, 13)

    # Test both function names (should be identical)
    r2_1 = calculate_archetype_r2(X_recon, X_original)
    r2_2 = archetypal_R2(X_recon, X_original)

    # Check return type
    assert isinstance(r2_1, torch.Tensor), "Should return tensor"
    assert r2_1.dim() == 0, "Should be scalar tensor"

    # Check functions are aliases
    assert torch.equal(r2_1, r2_2), "Both functions should give same result"

    # Check .item() works
    r2_float = r2_1.item()
    assert isinstance(r2_float, float), ".item() should convert to Python float"


def test_train_vae_return_structure():
    """Verify train_vae() returns (dict, model) tuple."""
    from peach._core.utils.training import train_vae

    # Create simple model and data
    model = Deep_AA(input_dim=5, n_archetypes=3, hidden_dims=[16])
    X = torch.randn(50, 5)
    dataset = torch.utils.data.TensorDataset(X)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train for just 2 epochs
    result = train_vae(model, dataloader, optimizer, n_epochs=2)

    # Check it returns tuple
    assert isinstance(result, tuple), "train_vae() should return tuple"
    assert len(result) == 2, "Should return (results, model)"

    results_dict, trained_model = result

    # Check results dictionary structure
    assert isinstance(results_dict, dict)
    assert 'history' in results_dict
    assert 'final_model' in results_dict
    assert 'model' in results_dict
    assert 'final_optimizer' in results_dict
    assert 'final_analysis' in results_dict
    assert 'epoch_archetype_positions' in results_dict
    assert 'training_config' in results_dict

    # Check history structure (no 'epoch' key - metrics stored directly)
    history = results_dict['history']
    assert 'loss' in history
    assert 'archetype_r2' in history
    assert len(history['loss']) == 2, "Should have 2 epochs"
    assert len(history['archetype_r2']) == 2, "Should have 2 epochs"

    # Check training config
    config = results_dict['training_config']
    assert config['n_epochs'] == 2
    assert 'actual_epochs' in config
    assert 'early_stop_triggered' in config

    # Check trained model
    assert isinstance(trained_model, torch.nn.Module)


def test_common_pitfall_forward_returns_dict():
    """Test that users must extract tensors from forward() output."""
    model = Deep_AA(input_dim=13, n_archetypes=5, hidden_dims=[128, 64])
    X = torch.randn(10, 13)

    # This is the pitfall - forward returns dict
    outputs = model.forward(X)

    # This should fail if someone tries to use outputs directly
    with pytest.raises(TypeError):
        # This would fail: "unsupported operand type for -: 'Tensor' and 'dict'"
        r2 = calculate_archetype_r2(outputs, X)

    # Correct usage
    X_recon = outputs['arch_recons']
    r2 = calculate_archetype_r2(X_recon, X)
    assert isinstance(r2, torch.Tensor)


def test_common_pitfall_r2_needs_item():
    """Test that R² functions return tensors that need .item()."""
    X_original = torch.randn(100, 13)
    X_recon = torch.randn(100, 13)

    r2_tensor = calculate_archetype_r2(X_recon, X_original)

    # Verify it's a tensor
    assert isinstance(r2_tensor, torch.Tensor)
    assert r2_tensor.dim() == 0, "Should be scalar tensor"

    # Correct usage - convert to float
    r2_float = r2_tensor.item()
    assert isinstance(r2_float, float)

    # This should work fine
    df = pd.DataFrame({'r2': [r2_float]})
    assert df['r2'].iloc[0] == r2_float

    # Storing in dict for later use
    results = {'archetype_r2': r2_float}
    assert isinstance(results['archetype_r2'], float)


def test_common_pitfall_model_device():
    """Test getting device from model parameters."""
    model = Deep_AA(input_dim=13, n_archetypes=5, hidden_dims=[128, 64])

    # Models don't have .device attribute
    assert not hasattr(model, 'device'), "Models don't have .device"

    # Correct way to get device
    device = next(model.parameters()).device
    assert isinstance(device, torch.device)

    # Should work to move tensors to that device
    X = torch.randn(10, 13)
    X_on_device = X.to(device)
    assert X_on_device.device == device


def test_anndata_storage_conventions():
    """Test AnnData storage locations after training."""
    # Create synthetic AnnData
    n_cells, n_genes = 100, 50
    X = np.random.randn(n_cells, n_genes)
    adata = AnnData(X=X)

    # Add PCA (required for training)
    adata.obsm['X_pca'] = np.random.randn(n_cells, 13)

    # After training, archetype coordinates should be in .uns
    adata.uns['archetype_coordinates'] = np.random.randn(5, 13)

    # After coordinate extraction, distances should be in .obsm
    adata.obsm['archetype_distances'] = np.random.randn(n_cells, 5)

    # After assignment, categorical should be in .obs
    adata.obs['archetypes'] = pd.Categorical(
        np.random.choice(['archetype_0', 'archetype_1', 'None'], size=n_cells)
    )

    # After pathway scoring
    adata.obsm['pathway_scores'] = np.random.randn(n_cells, 20)
    adata.uns['pathway_scores_pathways'] = [f"pathway_{i}" for i in range(20)]

    # Verify all storage locations
    assert 'archetype_coordinates' in adata.uns
    assert 'archetype_distances' in adata.obsm
    assert 'archetypes' in adata.obs
    assert 'pathway_scores' in adata.obsm
    assert 'pathway_scores_pathways' in adata.uns

    # Verify shapes
    assert adata.obsm['archetype_distances'].shape == (n_cells, 5)
    assert adata.obsm['pathway_scores'].shape == (n_cells, 20)
    assert len(adata.uns['pathway_scores_pathways']) == 20


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

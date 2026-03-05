"""Tests for flow matching: velocity network, training, transport, Jacobian, MMD."""

import numpy as np
import pytest
import torch


class TestVelocityNetwork:
    def test_output_shape(self):
        """(batch, dim) in -> (batch, dim) out."""
        from peach._core.utils.flow_matching import VelocityNetwork

        dim = 10
        net = VelocityNetwork(dim, hidden_dims=(32, 32))
        x = torch.randn(16, dim)
        t = torch.rand(16, 1)
        v = net(x, t)
        assert v.shape == (16, dim)

    def test_1d_time(self):
        """Handles 1D time input (no trailing dim)."""
        from peach._core.utils.flow_matching import VelocityNetwork

        dim = 5
        net = VelocityNetwork(dim, hidden_dims=(16,))
        x = torch.randn(8, dim)
        t = torch.rand(8)  # 1D
        v = net(x, t)
        assert v.shape == (8, dim)

    def test_single_sample(self):
        """Works with a single sample."""
        from peach._core.utils.flow_matching import VelocityNetwork

        dim = 4
        net = VelocityNetwork(dim, hidden_dims=(16,))
        x = torch.randn(1, dim)
        t = torch.tensor([0.5])
        v = net(x, t)
        assert v.shape == (1, dim)

    def test_gradient_flows(self):
        """Gradients propagate through the network."""
        from peach._core.utils.flow_matching import VelocityNetwork

        dim = 4
        net = VelocityNetwork(dim, hidden_dims=(16, 16))
        x = torch.randn(8, dim, requires_grad=True)
        t = torch.rand(8, 1)
        v = net(x, t)
        loss = v.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (8, dim)


class TestFlowModel:
    def test_training_loss_decreases(self):
        """Loss should decrease over training epochs."""
        from peach._core.utils.flow_matching import FlowModel

        rng = np.random.default_rng(42)
        dim = 5
        source = rng.normal(0, 1, (200, dim))
        target = rng.normal(3, 1, (200, dim))  # shifted

        model = FlowModel(dim, hidden_dims=(32, 32), lr=1e-3)
        losses = model.train(source, target, n_epochs=100, batch_size=64)

        assert len(losses) == 100
        # Last 10 avg should be less than first 10 avg
        assert np.mean(losses[-10:]) < np.mean(losses[:10])

    def test_transport_moves_toward_target(self):
        """MMD(transported, target) < MMD(source, target)."""
        from peach._core.utils.flow_matching import FlowModel, compute_mmd

        rng = np.random.default_rng(42)
        dim = 3
        source = rng.normal(0, 1, (100, dim))
        target = rng.normal(5, 1, (100, dim))

        model = FlowModel(dim, hidden_dims=(64, 64), lr=1e-3)
        model.train(source, target, n_epochs=200, batch_size=64)
        transported = model.transport(source, n_steps=50)

        mmd_before = compute_mmd(source, target)
        mmd_after = compute_mmd(transported, target)
        assert mmd_after < mmd_before

    def test_transport_output_shape(self):
        """Transport returns correct shape without trajectory."""
        from peach._core.utils.flow_matching import FlowModel

        rng = np.random.default_rng(42)
        dim = 4
        source = rng.normal(0, 1, (30, dim))
        target = rng.normal(1, 1, (30, dim))

        model = FlowModel(dim, hidden_dims=(16, 16))
        model.train(source, target, n_epochs=10, batch_size=16)
        result = model.transport(source, n_steps=5)
        assert result.shape == (30, dim)
        assert isinstance(result, np.ndarray)

    def test_transport_trajectory(self):
        """return_trajectory=True returns full path with correct shape."""
        from peach._core.utils.flow_matching import FlowModel

        rng = np.random.default_rng(42)
        dim = 3
        source = rng.normal(0, 1, (50, dim))
        target = rng.normal(3, 1, (50, dim))

        model = FlowModel(dim, hidden_dims=(32, 32))
        model.train(source, target, n_epochs=50, batch_size=32)
        n_steps = 10
        traj = model.transport(source, n_steps=n_steps, return_trajectory=True)
        assert traj.shape == (n_steps + 1, 50, dim)
        # First frame should be the source
        np.testing.assert_array_almost_equal(traj[0], source)

    def test_velocity_at(self):
        """velocity_at returns correct shape."""
        from peach._core.utils.flow_matching import FlowModel

        rng = np.random.default_rng(42)
        dim = 4
        source = rng.normal(0, 1, (50, dim))
        target = rng.normal(2, 1, (50, dim))

        model = FlowModel(dim, hidden_dims=(32, 32))
        model.train(source, target, n_epochs=20, batch_size=32)
        v = model.velocity_at(source, t=0.5)
        assert v.shape == (50, dim)
        assert isinstance(v, np.ndarray)

    def test_velocity_at_different_times(self):
        """Velocity at t=0 and t=1 can differ."""
        from peach._core.utils.flow_matching import FlowModel

        rng = np.random.default_rng(42)
        dim = 3
        source = rng.normal(0, 1, (30, dim))
        target = rng.normal(5, 1, (30, dim))

        model = FlowModel(dim, hidden_dims=(32, 32))
        model.train(source, target, n_epochs=50, batch_size=32)
        v0 = model.velocity_at(source, t=0.0)
        v1 = model.velocity_at(source, t=1.0)
        # They should not be identical (network is time-dependent)
        assert not np.allclose(v0, v1, atol=1e-3)

    def test_jacobian_shape(self):
        """Jacobian returns [n_points, dim, dim]."""
        from peach._core.utils.flow_matching import FlowModel

        rng = np.random.default_rng(42)
        dim = 3
        source = rng.normal(0, 1, (20, dim))
        target = rng.normal(2, 1, (20, dim))

        model = FlowModel(dim, hidden_dims=(16, 16))
        model.train(source, target, n_epochs=20, batch_size=16)
        # Test on small subset for speed
        jac = model.jacobian(source[:5], t=0.5)
        assert jac.shape == (5, dim, dim)
        assert isinstance(jac, np.ndarray)

    def test_jacobian_finite_difference(self):
        """Jacobian should approximately match finite differences."""
        from peach._core.utils.flow_matching import FlowModel

        rng = np.random.default_rng(42)
        dim = 3
        source = rng.normal(0, 1, (50, dim))
        target = rng.normal(2, 1, (50, dim))

        model = FlowModel(dim, hidden_dims=(32, 32))
        model.train(source, target, n_epochs=50, batch_size=32)

        # Compare autograd Jacobian with finite differences at one point
        x0 = source[:1]
        t_val = 0.5
        jac_auto = model.jacobian(x0, t=t_val)  # [1, dim, dim]

        eps = 1e-4
        jac_fd = np.zeros((dim, dim))
        v0 = model.velocity_at(x0, t=t_val)[0]
        for k in range(dim):
            x_plus = x0.copy()
            x_plus[0, k] += eps
            v_plus = model.velocity_at(x_plus, t=t_val)[0]
            jac_fd[:, k] = (v_plus - v0) / eps

        np.testing.assert_allclose(jac_auto[0], jac_fd, atol=1e-2, rtol=0.1)

    def test_different_source_target_sizes(self):
        """Training works when source and target have different sizes."""
        from peach._core.utils.flow_matching import FlowModel

        rng = np.random.default_rng(42)
        dim = 3
        source = rng.normal(0, 1, (80, dim))
        target = rng.normal(3, 1, (120, dim))

        model = FlowModel(dim, hidden_dims=(32, 32))
        losses = model.train(source, target, n_epochs=20, batch_size=32)
        assert len(losses) == 20
        result = model.transport(source, n_steps=10)
        assert result.shape == (80, dim)


class TestMMD:
    def test_identical_distributions(self):
        """MMD between identical sets should be ~0."""
        from peach._core.utils.flow_matching import compute_mmd

        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 5))
        mmd = compute_mmd(X, X)
        assert mmd < 0.01

    def test_different_distributions(self):
        """MMD between well-separated distributions should be large."""
        from peach._core.utils.flow_matching import compute_mmd

        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 5))
        Y = rng.normal(5, 1, (200, 5))
        mmd = compute_mmd(X, Y)
        assert mmd > 0.1

    def test_mmd_symmetry(self):
        """MMD(X, Y) == MMD(Y, X)."""
        from peach._core.utils.flow_matching import compute_mmd

        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 3))
        Y = rng.normal(2, 1, (100, 3))
        mmd_xy = compute_mmd(X, Y)
        mmd_yx = compute_mmd(Y, X)
        np.testing.assert_almost_equal(mmd_xy, mmd_yx, decimal=10)

    def test_custom_bandwidth(self):
        """Custom bandwidth parameter is respected."""
        from peach._core.utils.flow_matching import compute_mmd

        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 3))
        Y = rng.normal(2, 1, (100, 3))
        mmd_small_bw = compute_mmd(X, Y, bandwidth=0.1)
        mmd_large_bw = compute_mmd(X, Y, bandwidth=10.0)
        # Different bandwidths should give different values
        assert mmd_small_bw != mmd_large_bw

    def test_different_sizes(self):
        """MMD works with different-sized point sets."""
        from peach._core.utils.flow_matching import compute_mmd

        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (80, 4))
        Y = rng.normal(0, 1, (120, 4))
        mmd = compute_mmd(X, Y)
        # Same distribution, different sizes -- should still be small
        assert mmd < 0.1

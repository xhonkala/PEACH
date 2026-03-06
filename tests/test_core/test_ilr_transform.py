import numpy as np
import pytest


class TestILRTransform:
    """Test ILR transform for simplex compositions."""

    def test_roundtrip(self):
        """weights -> ILR -> inverse ILR recovers epsilon-smoothed weights.

        The forward transform applies epsilon smoothing, so the roundtrip
        recovers W_smooth (not the original W). We verify this exactly.
        """
        from peach._core.utils.ilr_transform import (
            ILR_EPSILON,
            ilr_transform,
            inverse_ilr,
        )

        rng = np.random.default_rng(42)
        W = rng.dirichlet([2, 3, 1, 4], size=100)

        # Compute the expected smoothed weights
        W_smooth = W + ILR_EPSILON
        W_smooth = W_smooth / W_smooth.sum(axis=1, keepdims=True)

        ilr_coords = ilr_transform(W)
        W_recovered = inverse_ilr(ilr_coords)
        np.testing.assert_array_almost_equal(W_smooth, W_recovered, decimal=10)

    def test_roundtrip_interior_points(self):
        """Interior simplex points roundtrip close to original weights.

        With large Dirichlet alphas, weights stay far from boundaries,
        so epsilon smoothing has negligible effect and the roundtrip
        approximately recovers the original weights.
        """
        from peach._core.utils.ilr_transform import ilr_transform, inverse_ilr

        rng = np.random.default_rng(42)
        # Large alphas keep weights well away from 0
        W = rng.dirichlet([20, 30, 10, 40], size=100)
        ilr_coords = ilr_transform(W)
        W_recovered = inverse_ilr(ilr_coords)
        # epsilon=1e-3 introduces ~1e-3 distortion even for interior points
        np.testing.assert_array_almost_equal(W, W_recovered, decimal=3)

    def test_dimension_reduction(self):
        """K weights produce K-1 ILR coordinates."""
        from peach._core.utils.ilr_transform import ilr_transform

        rng = np.random.default_rng(42)
        K = 5
        W = rng.dirichlet([1] * K, size=50)
        ilr_coords = ilr_transform(W)
        assert ilr_coords.shape == (50, K - 1)

    def test_zero_handling(self):
        """Near-zero weights are smoothed with epsilon before log."""
        from peach._core.utils.ilr_transform import ilr_transform

        W = np.array([[1.0, 0.0, 0.0],  # vertex -- has zeros
                       [0.5, 0.5, 0.0]])  # edge -- has zero
        ilr_coords = ilr_transform(W)
        assert np.all(np.isfinite(ilr_coords))  # no -inf from log(0)
        assert ilr_coords.shape == (2, 2)

    def test_preserves_ordering(self):
        """Points closer on simplex should be closer in ILR space."""
        from peach._core.utils.ilr_transform import ilr_transform

        W1 = np.array([[0.8, 0.1, 0.1]])
        W2 = np.array([[0.7, 0.2, 0.1]])  # close to W1
        W3 = np.array([[0.1, 0.1, 0.8]])  # far from W1

        ilr1 = ilr_transform(W1)[0]
        ilr2 = ilr_transform(W2)[0]
        ilr3 = ilr_transform(W3)[0]

        dist_12 = np.linalg.norm(ilr1 - ilr2)
        dist_13 = np.linalg.norm(ilr1 - ilr3)
        assert dist_12 < dist_13

    def test_inverse_ilr_sums_to_one(self):
        """Inverse ILR output should sum to 1."""
        from peach._core.utils.ilr_transform import inverse_ilr

        # Random ILR coordinates
        ilr_coords = np.random.default_rng(42).standard_normal((100, 3))
        W = inverse_ilr(ilr_coords)
        assert W.shape == (100, 4)  # K = ILR_dim + 1
        np.testing.assert_array_almost_equal(W.sum(axis=1), 1.0)
        assert np.all(W > 0)  # compositions are positive

    def test_single_sample(self):
        """Works for a single sample."""
        from peach._core.utils.ilr_transform import ilr_transform, inverse_ilr

        W = np.array([[0.25, 0.25, 0.25, 0.25]])
        ilr_coords = ilr_transform(W)
        assert ilr_coords.shape == (1, 3)
        W_back = inverse_ilr(ilr_coords)
        np.testing.assert_array_almost_equal(W, W_back, decimal=4)

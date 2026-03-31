"""Tests for BIC elbow detection in simplex_gmm."""

import numpy as np
import pytest


def test_bic_elbow_finds_correct_elbow():
    """BIC curve with clear elbow at n=5 should return 5."""
    from peach._core.utils.simplex_gmm import _find_bic_elbow
    # Simulate BIC curve: steep drop then plateau
    n_range = list(range(3, 15))
    bic = [1000, 800, 600, 500, 480, 475, 473, 472, 471.5, 471.2, 471.0, 470.9]
    result = _find_bic_elbow(n_range, bic)
    # Elbow should be around n=5-8 (where improvement flattens)
    assert 5 <= result <= 8, f"Expected elbow at 5-8, got {result}"


def test_bic_elbow_monotonic_returns_argmin():
    """Monotonically decreasing BIC with no elbow should return last value."""
    from peach._core.utils.simplex_gmm import _find_bic_elbow
    n_range = list(range(3, 10))
    bic = [100, 90, 80, 70, 60, 50, 40]
    result = _find_bic_elbow(n_range, bic)
    assert result == 9  # argmin is the last value


def test_bic_elbow_short_range():
    """With only 2 values, should return argmin."""
    from peach._core.utils.simplex_gmm import _find_bic_elbow
    result = _find_bic_elbow([3, 4], [100, 90])
    assert result == 4

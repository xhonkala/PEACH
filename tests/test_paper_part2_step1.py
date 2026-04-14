"""Tests for Part 2 Step 1 pipeline (prep + run + structural regex checks).

Mirrors tests/test_paper_part1_fixes.py structure:
- Computation tests for new helpers
- Structural regex tests for the Part 2 scripts
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make scripts/ importable for helper tests
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ============================================================================
# Task 1 — safe_stratified_split
# ============================================================================


def test_safe_stratified_split_simple_case():
    """With uniformly large strata, behaves like normal stratified split."""
    from _paper_part1_prep import safe_stratified_split

    primary = pd.Series(["A"] * 50 + ["B"] * 50)
    fallback = pd.Series(["X"] * 100)
    train_idx, holdout_idx, diag = safe_stratified_split(
        primary, fallback, test_size=0.20, min_stratum_size=10, random_state=42
    )
    assert len(train_idx) + len(holdout_idx) == 100
    assert len(set(train_idx) & set(holdout_idx)) == 0
    # Each primary stratum should be ~80/20
    for label in ("A", "B"):
        mask = primary == label
        in_train = mask.values[train_idx].sum()
        in_holdout = mask.values[holdout_idx].sum()
        assert 7 <= in_holdout <= 13  # 20% of 50 = 10 ± noise
        assert in_train + in_holdout == 50
    assert diag["n_fallback_cells"] == 0
    assert diag["n_random_fallback_cells"] == 0


def test_safe_stratified_split_fallback_triggers_on_tiny_strata():
    """Strata below min_stratum_size collapse to fallback key."""
    from _paper_part1_prep import safe_stratified_split

    # 3 big strata (40 each) + 2 tiny strata (3 each) that share a fallback key
    primary = pd.Series(["A"] * 40 + ["B"] * 40 + ["C"] * 40 + ["D"] * 3 + ["E"] * 3)
    fallback = pd.Series(["big"] * 120 + ["tiny_group"] * 6)
    train_idx, holdout_idx, diag = safe_stratified_split(
        primary, fallback, test_size=0.20, min_stratum_size=10, random_state=42
    )
    assert len(train_idx) + len(holdout_idx) == 126
    # Tiny strata (D, E) count as fallback cells
    assert diag["n_fallback_cells"] == 6


def test_safe_stratified_split_random_fallback_when_fallback_also_tiny():
    """If both primary and fallback strata are <2, fall back to random per-cell split."""
    from _paper_part1_prep import safe_stratified_split

    # One stratum of size 1 — can't stratify even after fallback collapse
    primary = pd.Series(["A"] * 40 + ["B"] * 40 + ["Z"] * 1)
    fallback = pd.Series(["big"] * 80 + ["orphan"] * 1)
    train_idx, holdout_idx, diag = safe_stratified_split(
        primary, fallback, test_size=0.20, min_stratum_size=10, random_state=42
    )
    assert len(train_idx) + len(holdout_idx) == 81
    assert diag["n_random_fallback_cells"] >= 1


def test_safe_stratified_split_deterministic():
    """Same random_state → same split."""
    from _paper_part1_prep import safe_stratified_split

    primary = pd.Series(["A"] * 50 + ["B"] * 50)
    fallback = pd.Series(["X"] * 100)
    t1, h1, _ = safe_stratified_split(primary, fallback, random_state=42)
    t2, h2, _ = safe_stratified_split(primary, fallback, random_state=42)
    assert np.array_equal(t1, t2)
    assert np.array_equal(h1, h2)


# ============================================================================
# Task 2 — build_response_timepoint_colormap
# ============================================================================


def test_response_timepoint_colormap_default_shape():
    from _paper_part1_viz import build_response_timepoint_colormap
    cmap = build_response_timepoint_colormap()
    # Default: 3 responses × 3 treatments
    assert len(cmap) == 9
    for key in [("NR", "Base"), ("R1", "PD1"), ("R2", "RTPD1")]:
        assert key in cmap
        assert cmap[key].startswith("#") and len(cmap[key]) == 7


def test_response_timepoint_colormap_hue_by_response():
    """Same response, different timepoints → same hue family."""
    from _paper_part1_viz import build_response_timepoint_colormap
    cmap = build_response_timepoint_colormap()
    # Crude check: red hex starts with high R (first 2 hex chars high)
    assert int(cmap[("NR", "Base")][1:3], 16) > 200  # light red
    assert int(cmap[("NR", "RTPD1")][1:3], 16) < 200  # darker red
    # R1 — orange (red + green)
    assert int(cmap[("R1", "Base")][1:3], 16) > 200  # light orange
    # R2 — blue (low red, high blue)
    assert int(cmap[("R2", "Base")][1:3], 16) < 200   # blue has low red
    assert int(cmap[("R2", "Base")][5:7], 16) > 200   # blue has high blue


# ============================================================================
# Task 3 — compute_w2_archetype_distance (Bures–Wasserstein)
# ============================================================================


def test_w2_identical_distributions_is_zero():
    from _paper_part1_viz import compute_w2_archetype_distance

    rng = np.random.default_rng(0)
    weights = rng.dirichlet(alpha=[1.0, 1.0, 1.0], size=200)
    d = compute_w2_archetype_distance(weights, weights)
    assert d == pytest.approx(0.0, abs=1e-6)


def test_w2_symmetric():
    from _paper_part1_viz import compute_w2_archetype_distance

    rng = np.random.default_rng(0)
    a = rng.dirichlet(alpha=[5.0, 1.0, 1.0], size=100)
    b = rng.dirichlet(alpha=[1.0, 1.0, 5.0], size=100)
    d_ab = compute_w2_archetype_distance(a, b)
    d_ba = compute_w2_archetype_distance(b, a)
    assert d_ab == pytest.approx(d_ba, rel=1e-5)


def test_w2_nonnegative():
    from _paper_part1_viz import compute_w2_archetype_distance
    rng = np.random.default_rng(0)
    a = rng.dirichlet(alpha=[1.0, 1.0, 1.0], size=50)
    b = rng.dirichlet(alpha=[10.0, 1.0, 1.0], size=50)
    assert compute_w2_archetype_distance(a, b) > 0


def test_w2_ordering_makes_sense():
    """Distribution far apart in mean should have larger W2 than similar ones."""
    from _paper_part1_viz import compute_w2_archetype_distance
    rng = np.random.default_rng(42)
    near = rng.dirichlet(alpha=[5.0, 1.0, 1.0], size=200)
    mid = rng.dirichlet(alpha=[1.0, 5.0, 1.0], size=200)
    far = rng.dirichlet(alpha=[1.0, 1.0, 5.0], size=200)
    anchor = rng.dirichlet(alpha=[5.0, 1.0, 1.0], size=200)
    d_near = compute_w2_archetype_distance(anchor, near)
    d_mid = compute_w2_archetype_distance(anchor, mid)
    d_far = compute_w2_archetype_distance(anchor, far)
    # anchor and near share the Dirichlet — distance small
    assert d_near < d_mid
    assert d_near < d_far

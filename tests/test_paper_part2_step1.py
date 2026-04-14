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


# ============================================================================
# Task 4 — build_segregation_ratio
# ============================================================================


def _make_synthetic_groups(adata_like_dict: dict, weights_by_group: dict,
                            n_per_group: int = 100, seed: int = 0):
    """Build a small fake adata.obs + weights for group-pair testing."""
    import pandas as pd
    import numpy as np

    rng = np.random.default_rng(seed)
    rows = []
    weights = []
    for (resp, tx), alpha in weights_by_group.items():
        rows.extend([(resp, tx)] * n_per_group)
        weights.append(rng.dirichlet(alpha=alpha, size=n_per_group))
    obs = pd.DataFrame(rows, columns=["response_group", "treatment"])
    W = np.vstack(weights)
    return obs, W


def test_segregation_ratio_identity_case_is_one():
    """If every group has the same distribution, within == between → ratio ≈ 1."""
    from _paper_part1_viz import build_segregation_ratio

    # All 9 groups share the same Dirichlet → same distribution
    alpha = [1.0, 1.0, 1.0]
    groups = {(r, t): alpha for r in ("NR", "R1", "R2")
              for t in ("Base", "PD1", "RTPD1")}
    obs, W = _make_synthetic_groups(None, groups, n_per_group=120, seed=42)
    out = build_segregation_ratio(obs, W,
                                   response_col="response_group",
                                   treatment_col="treatment")
    assert 0.7 <= out["ratio"] <= 1.3
    assert out["n_within_pairs"] == 9   # 3 responses × C(3,2)
    assert out["n_between_pairs"] == 27 # 3 resp_pairs × 3 × 3 tx combos


def test_segregation_ratio_strong_separation():
    """R2 lives on archetype 3 only; NR on archetype 1 — expect ratio > 1.3."""
    from _paper_part1_viz import build_segregation_ratio
    groups = {
        ("NR", "Base"): [10.0, 1.0, 1.0],  ("NR", "PD1"): [10.0, 1.0, 1.0],  ("NR", "RTPD1"): [10.0, 1.0, 1.0],
        ("R1", "Base"): [1.0, 10.0, 1.0],  ("R1", "PD1"): [1.0, 10.0, 1.0],  ("R1", "RTPD1"): [1.0, 10.0, 1.0],
        ("R2", "Base"): [1.0, 1.0, 10.0],  ("R2", "PD1"): [1.0, 1.0, 10.0],  ("R2", "RTPD1"): [1.0, 1.0, 10.0],
    }
    obs, W = _make_synthetic_groups(None, groups, n_per_group=150, seed=7)
    out = build_segregation_ratio(obs, W, "response_group", "treatment")
    assert out["ratio"] >= 1.3
    assert out["within"] < out["between"]


# ============================================================================
# Task 5 — build_archetype_char_table
# ============================================================================


def test_archetype_char_table_schema():
    from _paper_part1_viz import build_archetype_char_table
    import pandas as pd
    rng = np.random.default_rng(0)
    n_cells, K = 300, 4
    obs = pd.DataFrame({
        "archetypes": rng.integers(0, K, n_cells),
        "response_group": rng.choice(["NR", "R1", "R2"], n_cells),
        "treatment": rng.choice(["Base", "PD1", "RTPD1"], n_cells),
        "cohort": ["P" + str(i % 5) for i in range(n_cells)],
        "majority_voting": rng.choice(["tumor", "luminal_2"], n_cells),
    })
    df = build_archetype_char_table(
        obs,
        archetypes_col="archetypes",
        covariate_cols=["response_group", "treatment", "cohort", "majority_voting"],
        top_genes_by_archetype=None,  # omit genes for this unit test
    )
    assert list(df.columns) >= [
        "archetype", "n_cells", "pct_cells",
        "dom_response_group", "dom_treatment", "dom_majority_voting",
        "top_cohorts",
    ]
    assert len(df) == K
    assert df["pct_cells"].sum() == pytest.approx(100.0, abs=0.01)


def test_archetype_char_table_top_genes_populated():
    from _paper_part1_viz import build_archetype_char_table
    import pandas as pd
    obs = pd.DataFrame({
        "archetypes": [0, 0, 1, 1],
        "response_group": ["NR", "NR", "R1", "R2"],
        "treatment": ["Base", "PD1", "Base", "PD1"],
        "cohort": ["P1", "P2", "P3", "P4"],
        "majority_voting": ["tumor", "tumor", "tumor", "tumor"],
    })
    top_genes = {0: ["GeneA", "GeneB"], 1: ["GeneC"]}
    df = build_archetype_char_table(
        obs, "archetypes",
        ["response_group", "treatment", "cohort", "majority_voting"],
        top_genes_by_archetype=top_genes,
    )
    assert df.loc[df["archetype"] == 0, "top_genes"].iat[0] == "GeneA, GeneB"
    assert df.loc[df["archetype"] == 1, "top_genes"].iat[0] == "GeneC"


# ============================================================================
# Task 6 — build_archetype_hypergeometric_tables
# ============================================================================


def test_hypergeometric_tables_schema_and_bh_monotonic():
    from _paper_part1_viz import build_archetype_hypergeometric_tables
    import pandas as pd

    rng = np.random.default_rng(7)
    n = 600
    K = 4
    # Strong enrichment: archetype 0 is 80% NR
    arch = rng.integers(0, K, n)
    resp = np.array([
        "NR" if (a == 0 and rng.random() < 0.8) else rng.choice(["R1", "R2", "NR"])
        for a in arch
    ])
    obs = pd.DataFrame({
        "archetypes": arch,
        "response_group": resp,
        "treatment": rng.choice(["Base", "PD1"], n),
    })
    tables = build_archetype_hypergeometric_tables(
        obs, archetypes_col="archetypes",
        covariate_cols=["response_group", "treatment"],
    )
    assert "response_group" in tables and "treatment" in tables
    for name, df in tables.items():
        # columns: archetype + for each level: OR, p, q
        # Check monotonicity: sorted-by-p q-values are ≥ sorted p-values
        # (simple BH check: every q-value is ≥ corresponding p-value)
        p_cols = [c for c in df.columns if c.startswith("p_")]
        q_cols = [c.replace("p_", "q_") for c in p_cols]
        for pc, qc in zip(p_cols, q_cols):
            assert (df[qc].values >= df[pc].values - 1e-12).all(), \
                f"BH q<p violation in {name}.{qc}"


def test_hypergeometric_enriched_archetype_has_low_q():
    """Archetype 0 enriched for NR should have q < 0.05 on NR column."""
    from _paper_part1_viz import build_archetype_hypergeometric_tables
    import pandas as pd

    rng = np.random.default_rng(3)
    n = 1200
    arch = rng.integers(0, 4, n)
    resp = np.where((arch == 0) & (rng.random(n) < 0.9), "NR",
                     rng.choice(["R1", "R2", "NR"], n))
    obs = pd.DataFrame({"archetypes": arch, "response_group": resp})
    tables = build_archetype_hypergeometric_tables(
        obs, "archetypes", ["response_group"]
    )
    df = tables["response_group"]
    q_NR = df.loc[df["archetype"] == 0, "q_NR"].iat[0]
    assert q_NR < 0.05


# ============================================================================
# Task 7 — build_holdout_projection_qc
# ============================================================================


def test_holdout_projection_qc_identical_inputs_match():
    """When train == holdout, R²s should match and NN distance mean should be low."""
    from _paper_part1_viz import build_holdout_projection_qc

    rng = np.random.default_rng(0)
    K, D = 4, 12
    archetypes = rng.normal(size=(K, D))
    cells = rng.normal(size=(300, D))
    # archetypal R² uses: original vs reconstruction = weights @ archetypes
    weights = rng.dirichlet(alpha=[1.0] * K, size=300)
    reconstruction = weights @ archetypes

    qc = build_holdout_projection_qc(
        cells_train=cells,
        reconstruction_train=reconstruction,
        cells_holdout=cells,
        reconstruction_holdout=reconstruction,
        archetype_positions=archetypes,
    )
    assert "train_r2" in qc and "holdout_r2" in qc
    assert abs(qc["train_r2"] - qc["holdout_r2"]) < 1e-6
    assert qc["holdout_mean_nn_dist"] >= 0.0


def test_holdout_projection_qc_worse_when_holdout_is_noise():
    """Train R² >> holdout R² when train cells are near their reconstruction but
    holdout cells are unrelated noise — the archetypes explain train well, not holdout."""
    from _paper_part1_viz import build_holdout_projection_qc

    rng = np.random.default_rng(0)
    K, D = 4, 10
    archetypes = rng.normal(size=(K, D))

    # Train: cells ≈ reconstruction + tiny noise → R² close to 1
    weights_train = rng.dirichlet([1.0] * K, size=200)
    recon_train = weights_train @ archetypes
    cells_train = recon_train + rng.normal(size=(200, D)) * 0.05

    # Holdout: pure noise cells, archetype reconstruction is unrelated → R² << 0
    cells_holdout = rng.normal(size=(200, D)) * 3.0
    weights_holdout = rng.dirichlet([1.0] * K, size=200)
    recon_holdout = weights_holdout @ archetypes

    qc = build_holdout_projection_qc(
        cells_train, recon_train, cells_holdout, recon_holdout, archetypes
    )
    # Holdout R² should be much worse than train
    assert qc["holdout_r2"] < qc["train_r2"]

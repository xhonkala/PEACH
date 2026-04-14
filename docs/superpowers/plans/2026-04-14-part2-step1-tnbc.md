# Paper Part 2 — Step 1 TNBC Global Fit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the Part 2 Step 1 pipeline — prep + global archetypal fit on TNBC tumor cells — producing `part2_report_YYYYMMDD_r1.html` with Figures 3A, 3B, and gated 3C.

**Architecture:** Fork Part 1's three-phase skeleton (training → Fig 3A → Fig 3B/3C). Add 8 new helpers to `_paper_part1_{prep,viz}.py` so both Part 1 and Part 2 can use them. New prep script + new run script + new test file, all committed to the current branch (`feature/v050-continuous-characterization`).

**Tech Stack:** Python 3.11 + `archetype` conda env, `scanpy`, `anndata`, `peach as pc` (editable install at `~/Desktop/peach`), `matplotlib`, `plotly`, `scipy`, `pandas`, `numpy`, `sklearn`, `scipy.linalg.sqrtm` (Bures–Wasserstein).

**Spec reference:** `docs/superpowers/specs/2026-04-14-part2-step1-design.md`

**Conventions inherited from Part 1:**
- No Claude attribution in commits/code/docstrings
- `conda run -n archetype python ...` for all runs
- Run tests via `conda run -n archetype python -m pytest ...`
- Prefer `.get()` for keys in `USE_GET_FOR` (see `peach._core.types_index`)
- Commit messages: dated or short topical (e.g., `Part 2: add safe_stratified_split`)

---

## File Structure

### Created
| Path | Responsibility |
|------|-----------------|
| `scripts/prep_tnbcrad.py` | TNBC prep script (MAD filter → PCA → stratified holdout split) |
| `scripts/run_paper_part2_tnbc.py` | Main Part 2 Step 1 run script (3 phases) |
| `tests/test_paper_part2_step1.py` | All tests for Part 2 Step 1 — computation + structural |
| `data/paper_part2/` | Output of prep script (3 h5ad files + provenance HTML) |
| `outputs/paper_part2/` | Output of run script (reports + logs) |

### Modified
| Path | Change |
|------|--------|
| `scripts/_paper_part1_prep.py` | Add `safe_stratified_split` |
| `scripts/_paper_part1_viz.py` | Add 7 new helpers (colormap, W2, segregation ratio, char table, hypergeometric, holdout QC, distance heatmaps, diversity block) |

### Untouched
- `src/peach/` — no package changes for Step 1
- `scripts/_paper_part1_prep.py: apply_mt_rb_mad_filter` + `select_n_pcs_by_cumvar` (reused as-is)
- All `run_paper_part1_*.py` — Part 1 keeps iterating independently
- `stress_genes/` — imported read-only

---

## Task 1: Add `safe_stratified_split` to `_paper_part1_prep.py`

**Files:**
- Modify: `scripts/_paper_part1_prep.py` (append new function; update module docstring)
- Test: `tests/test_paper_part2_step1.py` (create file)

**Purpose:** Provide fallback-aware stratified 80/20 split. Some strata will be tiny (Patient06 has 6 cells across 3 strata) — naive `StratifiedShuffleSplit` errors. We collapse tiny strata to a fallback key, then random-split if still too small.

- [ ] **Step 1: Create `tests/test_paper_part2_step1.py` with the failing test**

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
conda run -n archetype python -m pytest tests/test_paper_part2_step1.py -v
```
Expected: all four tests FAIL with `ImportError: cannot import name 'safe_stratified_split' from '_paper_part1_prep'`.

- [ ] **Step 3: Implement `safe_stratified_split` — append to `scripts/_paper_part1_prep.py`**

Add this function at the bottom of the file (after `apply_mt_rb_mad_filter`):

```python
def safe_stratified_split(
    primary_stratum: "pd.Series",
    fallback_stratum: "pd.Series",
    test_size: float = 0.20,
    min_stratum_size: int = 10,
    random_state: int = 42,
):
    """Fallback-aware stratified 80/20 (or custom test_size) split.

    Strata in ``primary_stratum`` smaller than ``min_stratum_size`` are
    collapsed into groups defined by ``fallback_stratum`` for the split.
    If a fallback group itself has <2 cells, those cells are randomly
    assigned (honouring ``test_size``).

    Parameters
    ----------
    primary_stratum, fallback_stratum : pd.Series, same length
        Primary stratum tag per cell, and a coarser fallback tag.
        Typical: primary = f"{cohort}|{response}|{treatment}",
                 fallback = cohort.
    test_size : float
        Fraction of cells in the holdout set.
    min_stratum_size : int
        Primary strata with fewer than this many cells collapse to
        their fallback.
    random_state : int
        RNG seed.

    Returns
    -------
    train_idx, holdout_idx : np.ndarray of int
        Positional row indices (0..N-1).
    diag : dict
        {
            "n_cells": int,
            "n_primary_strata": int,
            "n_collapsed_strata": int,
            "n_fallback_cells": int,          # cells routed through fallback
            "n_random_fallback_cells": int,   # cells ultimately random-split
            "test_size": float,
        }
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedShuffleSplit

    assert len(primary_stratum) == len(fallback_stratum), "length mismatch"
    n = len(primary_stratum)
    primary = pd.Series(primary_stratum).reset_index(drop=True).astype(str)
    fallback = pd.Series(fallback_stratum).reset_index(drop=True).astype(str)

    # 1. Identify tiny primary strata and collapse to fallback
    primary_sizes = primary.value_counts()
    tiny_labels = set(primary_sizes[primary_sizes < min_stratum_size].index)

    effective = primary.copy()
    effective.loc[primary.isin(tiny_labels)] = (
        "__FB__" + fallback.loc[primary.isin(tiny_labels)]
    )
    n_fallback_cells = int(primary.isin(tiny_labels).sum())
    n_collapsed_strata = len(tiny_labels)

    # 2. Identify remaining strata that are still too small to stratified-split
    #    (sklearn needs at least 2 samples per class AND test_size * n ≥ 1)
    eff_sizes = effective.value_counts()
    random_labels = set(eff_sizes[eff_sizes < 2].index)
    random_mask = effective.isin(random_labels).values

    n_random_fallback_cells = int(random_mask.sum())

    rng = np.random.default_rng(random_state)
    train_parts: list[np.ndarray] = []
    holdout_parts: list[np.ndarray] = []

    # 3. Stratified split on the good strata
    good_idx = np.where(~random_mask)[0]
    if len(good_idx) > 0:
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        good_effective = effective.iloc[good_idx].values
        # StratifiedShuffleSplit needs at least 2 classes; if only one class, fall through
        if len(set(good_effective)) >= 2:
            for tr, ho in sss.split(np.zeros(len(good_idx)), good_effective):
                train_parts.append(good_idx[tr])
                holdout_parts.append(good_idx[ho])
        else:
            # Single class — random split
            shuffled = good_idx.copy()
            rng.shuffle(shuffled)
            n_ho = int(round(test_size * len(shuffled)))
            holdout_parts.append(shuffled[:n_ho])
            train_parts.append(shuffled[n_ho:])

    # 4. Random fallback for the truly orphan cells
    if n_random_fallback_cells > 0:
        rnd_idx = np.where(random_mask)[0]
        shuffled = rnd_idx.copy()
        rng.shuffle(shuffled)
        n_ho = int(round(test_size * len(shuffled)))
        holdout_parts.append(shuffled[:n_ho])
        train_parts.append(shuffled[n_ho:])

    train_idx = np.sort(np.concatenate(train_parts)) if train_parts else np.array([], dtype=int)
    holdout_idx = np.sort(np.concatenate(holdout_parts)) if holdout_parts else np.array([], dtype=int)

    diag = {
        "n_cells": n,
        "n_primary_strata": int(primary.nunique()),
        "n_collapsed_strata": n_collapsed_strata,
        "n_fallback_cells": n_fallback_cells,
        "n_random_fallback_cells": n_random_fallback_cells,
        "test_size": test_size,
    }
    return train_idx, holdout_idx, diag
```

Also update the module docstring at the top of `_paper_part1_prep.py` to list the new function (look for the existing "Exposes:" block and append).

- [ ] **Step 4: Run the tests to verify they pass**

```bash
conda run -n archetype python -m pytest tests/test_paper_part2_step1.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_paper_part2_step1.py scripts/_paper_part1_prep.py
git commit -m "Part 2: add safe_stratified_split with fallback"
```

---

## Task 2: Add `build_response_timepoint_colormap` to `_paper_part1_viz.py`

**Files:**
- Modify: `scripts/_paper_part1_viz.py`
- Test: `tests/test_paper_part2_step1.py` (append)

- [ ] **Step 1: Append the failing test**

Add below Task 1's tests in `tests/test_paper_part2_step1.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
conda run -n archetype python -m pytest tests/test_paper_part2_step1.py::test_response_timepoint_colormap_default_shape tests/test_paper_part2_step1.py::test_response_timepoint_colormap_hue_by_response -v
```
Expected: 2 FAIL with ImportError.

- [ ] **Step 3: Implement — append to `scripts/_paper_part1_viz.py`**

Add at the end of the file:

```python
# ============================================================================
# Part 2 helpers (shared with Part 1 when relevant)
# ============================================================================


def build_response_timepoint_colormap(
    responses: Sequence[str] = ("NR", "R1", "R2"),
    treatments: Sequence[str] = ("Base", "PD1", "RTPD1"),
) -> dict:
    """Return a ``(response, treatment) -> hex color`` map.

    Hue = response lineage (NR=reds, R1=oranges, R2=blues); lightness =
    timepoint (lightest at the first treatment, darkest at the last).

    Raises ValueError if an unknown response is passed.
    """
    ramps = {
        "NR": ["#fca5a5", "#ef4444", "#991b1b"],
        "R1": ["#fed7aa", "#f97316", "#9a3412"],
        "R2": ["#93c5fd", "#2563eb", "#1e3a8a"],
    }
    unknown = set(responses) - set(ramps)
    if unknown:
        raise ValueError(f"Unknown response groups: {sorted(unknown)}. "
                         f"Expected subset of {sorted(ramps)}.")
    if len(treatments) > 3:
        raise ValueError("Only up to 3 treatments supported by the ramp width.")

    out: dict = {}
    for r in responses:
        ramp = ramps[r]
        for i, t in enumerate(treatments):
            out[(r, t)] = ramp[i]
    return out
```

- [ ] **Step 4: Verify pass**

```bash
conda run -n archetype python -m pytest tests/test_paper_part2_step1.py -v
```
Expected: 6 passed total (4 from Task 1 + 2 new).

- [ ] **Step 5: Commit**

```bash
git add scripts/_paper_part1_viz.py tests/test_paper_part2_step1.py
git commit -m "Part 2: add build_response_timepoint_colormap"
```

---

## Task 3: Add `compute_w2_archetype_distance` (Bures–Wasserstein) to `_paper_part1_viz.py`

**Files:**
- Modify: `scripts/_paper_part1_viz.py`
- Test: `tests/test_paper_part2_step1.py` (append)

**Approach:** 2-Wasserstein between two multivariate Gaussian approximations of cell-group archetype weight distributions. Formula:

```
W2²(N(μ1,Σ1), N(μ2,Σ2)) = ‖μ1-μ2‖² + Tr(Σ1 + Σ2 - 2·(Σ1^½ · Σ2 · Σ1^½)^½)
```

Using `scipy.linalg.sqrtm`. Fast for K≤20 (archetype simplex dims).

- [ ] **Step 1: Append tests**

```python
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
```

- [ ] **Step 2: Run tests, expect failure**

```bash
conda run -n archetype python -m pytest tests/test_paper_part2_step1.py -k w2 -v
```
Expected: 4 FAIL.

- [ ] **Step 3: Implement — append to `scripts/_paper_part1_viz.py`**

```python
def compute_w2_archetype_distance(weights_a, weights_b):
    """2-Wasserstein distance between two cell groups' archetype-weight
    distributions, via the Bures–Wasserstein closed form on Gaussian
    approximations.

    Parameters
    ----------
    weights_a, weights_b : np.ndarray
        Shape ``(n_cells_*, n_archetypes)``. Each row is a simplex
        point. Groups may have different cell counts.

    Returns
    -------
    float
        Non-negative W2 distance. Returns 0.0 when both inputs are
        identical (bit-exact).
    """
    import numpy as np
    from scipy.linalg import sqrtm

    a = np.asarray(weights_a, dtype=np.float64)
    b = np.asarray(weights_b, dtype=np.float64)
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"Archetype dim mismatch: a={a.shape[1]}, b={b.shape[1]}"
        )
    if a.shape[0] < 2 or b.shape[0] < 2:
        raise ValueError("Each group needs at least 2 cells for a covariance.")

    mu_a = a.mean(axis=0)
    mu_b = b.mean(axis=0)
    Sig_a = np.cov(a, rowvar=False)
    Sig_b = np.cov(b, rowvar=False)

    # Numerical floor on the diagonals (archetype weights can be near-degenerate)
    eps = 1e-10
    Sig_a = Sig_a + eps * np.eye(Sig_a.shape[0])
    Sig_b = Sig_b + eps * np.eye(Sig_b.shape[0])

    # Mean-distance term
    mean_term = float(np.sum((mu_a - mu_b) ** 2))

    # Bures term: Tr(Σa + Σb - 2 * (Σa^½ Σb Σa^½)^½)
    sqrt_Sa = sqrtm(Sig_a)
    # sqrtm may return complex due to floating round-off; drop imaginary
    sqrt_Sa = np.asarray(sqrt_Sa).real
    middle = sqrt_Sa @ Sig_b @ sqrt_Sa
    sqrt_middle = sqrtm(middle)
    sqrt_middle = np.asarray(sqrt_middle).real
    bures = float(np.trace(Sig_a) + np.trace(Sig_b) - 2.0 * np.trace(sqrt_middle))

    # Numerical clamp (tiny negatives from sqrtm roundoff)
    w2_sq = max(0.0, mean_term + bures)
    return float(np.sqrt(w2_sq))
```

- [ ] **Step 4: Verify pass**

```bash
conda run -n archetype python -m pytest tests/test_paper_part2_step1.py -k w2 -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/_paper_part1_viz.py tests/test_paper_part2_step1.py
git commit -m "Part 2: add Bures-Wasserstein compute_w2_archetype_distance"
```

---

## Task 4: Add `build_segregation_ratio` to `_paper_part1_viz.py`

**Files:**
- Modify: `scripts/_paper_part1_viz.py`
- Test: `tests/test_paper_part2_step1.py` (append)

**Purpose:** Compute the Fig 3C gate value per spec §5.4.2. Returns within / between / ratio scalars with the exact pair-counting rules from the spec.

- [ ] **Step 1: Append tests**

```python
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
```

- [ ] **Step 2: Run tests, expect failure**

```bash
conda run -n archetype python -m pytest tests/test_paper_part2_step1.py -k segregation -v
```
Expected: 2 FAIL.

- [ ] **Step 3: Implement**

Append to `scripts/_paper_part1_viz.py`:

```python
def build_segregation_ratio(obs, weights, response_col: str,
                              treatment_col: str) -> dict:
    """Within- vs between-response 2-Wasserstein ratio for Fig 3C gating.

    Parameters
    ----------
    obs : pd.DataFrame
        One row per cell; must contain ``response_col`` and ``treatment_col``.
    weights : np.ndarray
        Shape ``(n_cells, n_archetypes)``. Same row order as ``obs``.
    response_col, treatment_col : str
        Column names in ``obs``.

    Returns
    -------
    dict with keys: ``within`` (mean W2, same-response different-treatment),
                    ``between`` (mean W2, different-response any-treatment),
                    ``ratio`` = between/within,
                    ``n_within_pairs``, ``n_between_pairs``,
                    ``pair_distances`` (list of {i,j,kind,w2}).
    """
    import numpy as np

    obs = obs.reset_index(drop=True)
    groups = (
        obs[[response_col, treatment_col]]
        .apply(tuple, axis=1)
        .tolist()
    )
    # Collect per-group cell indices
    unique = sorted(set(groups))
    idx_of = {g: [] for g in unique}
    for i, g in enumerate(groups):
        idx_of[g].append(i)
    idx_of = {g: np.array(v) for g, v in idx_of.items() if len(v) >= 2}

    unique_ok = sorted(idx_of.keys())
    pair_dists = []
    within_vals = []
    between_vals = []

    for i, g1 in enumerate(unique_ok):
        for g2 in unique_ok[i + 1:]:
            w1 = weights[idx_of[g1]]
            w2 = weights[idx_of[g2]]
            d = compute_w2_archetype_distance(w1, w2)
            kind = "within" if g1[0] == g2[0] else "between"
            pair_dists.append({"group_a": g1, "group_b": g2, "kind": kind, "w2": d})
            if kind == "within":
                within_vals.append(d)
            else:
                between_vals.append(d)

    within_mean = float(np.mean(within_vals)) if within_vals else float("nan")
    between_mean = float(np.mean(between_vals)) if between_vals else float("nan")
    ratio = (between_mean / within_mean) if within_mean > 0 else float("nan")

    return {
        "within": within_mean,
        "between": between_mean,
        "ratio": ratio,
        "n_within_pairs": len(within_vals),
        "n_between_pairs": len(between_vals),
        "pair_distances": pair_dists,
    }
```

- [ ] **Step 4: Verify pass**

```bash
conda run -n archetype python -m pytest tests/test_paper_part2_step1.py -k segregation -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/_paper_part1_viz.py tests/test_paper_part2_step1.py
git commit -m "Part 2: add build_segregation_ratio for Fig 3C gate"
```

---

## Task 5: Add `build_archetype_char_table` to `_paper_part1_viz.py`

**Files:**
- Modify: `scripts/_paper_part1_viz.py`
- Test: `tests/test_paper_part2_step1.py` (append)

**Purpose:** Per-archetype "quick look" characterization table (spec §5.3.3).

- [ ] **Step 1: Append tests**

```python
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
```

- [ ] **Step 2: Run tests, expect failure**

```bash
conda run -n archetype python -m pytest tests/test_paper_part2_step1.py -k char_table -v
```
Expected: 2 FAIL.

- [ ] **Step 3: Implement**

Append to `scripts/_paper_part1_viz.py`:

```python
def build_archetype_char_table(
    obs,
    archetypes_col: str,
    covariate_cols: Sequence[str],
    top_genes_by_archetype: dict | None = None,
    top_k_cohorts: int = 3,
) -> "pd.DataFrame":
    """One-row-per-archetype quick-look characterization table.

    Columns (fixed order):
        archetype, n_cells, pct_cells,
        dom_{covariate} for each covariate in ``covariate_cols``,
        top_cohorts (if ``cohort`` or similar patient-like col is present),
        top_genes (if ``top_genes_by_archetype`` provided).
    """
    import pandas as pd

    # Filter rows with a non-NaN archetype assignment
    obs = obs.loc[obs[archetypes_col].notna()].copy()
    total_cells = len(obs)

    rows = []
    archetype_ids = sorted(obs[archetypes_col].unique())
    cohort_col_candidate = next(
        (c for c in ("cohort", "patient", "donor") if c in covariate_cols),
        None,
    )

    for a in archetype_ids:
        sub = obs.loc[obs[archetypes_col] == a]
        n = len(sub)
        row = {
            "archetype": int(a),
            "n_cells": n,
            "pct_cells": 100.0 * n / total_cells if total_cells else 0.0,
        }
        for cov in covariate_cols:
            if cov == cohort_col_candidate:
                # emit as top-k string
                vc = sub[cov].value_counts().head(top_k_cohorts)
                row["top_cohorts"] = ", ".join(
                    f"{k} ({v})" for k, v in vc.items()
                )
            else:
                vc = sub[cov].value_counts()
                dom = vc.index[0] if len(vc) else None
                dom_frac = vc.iloc[0] / n if n and len(vc) else 0.0
                row[f"dom_{cov}"] = f"{dom} ({100*dom_frac:.0f}%)" if dom is not None else ""
        if top_genes_by_archetype is not None:
            row["top_genes"] = ", ".join(top_genes_by_archetype.get(int(a), []))
        rows.append(row)

    df = pd.DataFrame(rows)
    return df
```

- [ ] **Step 4: Verify**

```bash
conda run -n archetype python -m pytest tests/test_paper_part2_step1.py -k char_table -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/_paper_part1_viz.py tests/test_paper_part2_step1.py
git commit -m "Part 2: add build_archetype_char_table"
```

---

## Task 6: Add `build_archetype_hypergeometric_tables` to `_paper_part1_viz.py`

**Files:**
- Modify: `scripts/_paper_part1_viz.py`
- Test: `tests/test_paper_part2_step1.py` (append)

**Purpose:** Spec §5.3.4 — per-covariate OR / p / q tables, BH-corrected per covariate.

- [ ] **Step 1: Append tests**

```python
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
```

- [ ] **Step 2: Run, expect failure**

```bash
conda run -n archetype python -m pytest tests/test_paper_part2_step1.py -k hypergeometric -v
```
Expected: 2 FAIL.

- [ ] **Step 3: Implement**

Append to `scripts/_paper_part1_viz.py`:

```python
def build_archetype_hypergeometric_tables(
    obs,
    archetypes_col: str,
    covariate_cols: Sequence[str],
    min_level_cells: int = 50,
) -> dict:
    """Per-covariate archetype enrichment tables.

    For each covariate (e.g., ``response_group``), build a K × L table
    where L = number of levels with ≥ ``min_level_cells``. Each cell
    reports ``OR (p, q)`` for the 2x2 Fisher test of
    "cells in archetype ∩ cells in level" vs margins.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keyed by covariate name. Each df has columns:
        ``archetype``, ``{OR|p|q}_{level}`` for each surviving level.
    """
    import pandas as pd
    from scipy.stats import fisher_exact
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError as e:
        raise ImportError("statsmodels required for BH correction") from e

    obs = obs.loc[obs[archetypes_col].notna()].copy()
    archetype_ids = sorted(obs[archetypes_col].unique())
    K = len(archetype_ids)
    n_total = len(obs)
    out: dict = {}

    for cov in covariate_cols:
        vc = obs[cov].value_counts()
        keep_levels = vc[vc >= min_level_cells].index.tolist()
        if not keep_levels:
            out[cov] = pd.DataFrame({"archetype": archetype_ids})
            continue

        # Compute OR + p per (archetype × level)
        ors = np.full((K, len(keep_levels)), np.nan)
        ps = np.full((K, len(keep_levels)), np.nan)
        for ai, a in enumerate(archetype_ids):
            in_arch = obs[archetypes_col] == a
            n_arch = int(in_arch.sum())
            for li, lv in enumerate(keep_levels):
                in_lv = obs[cov] == lv
                a11 = int((in_arch & in_lv).sum())
                a12 = int((in_arch & ~in_lv).sum())
                a21 = int((~in_arch & in_lv).sum())
                a22 = int((~in_arch & ~in_lv).sum())
                table = [[a11, a12], [a21, a22]]
                or_val, pval = fisher_exact(table, alternative="two-sided")
                ors[ai, li] = or_val
                ps[ai, li] = pval

        # BH correction across all (archetype × level) tests in this covariate
        flat_p = ps.flatten()
        _, qs, _, _ = multipletests(flat_p, method="fdr_bh")
        qs = qs.reshape(ps.shape)

        df = pd.DataFrame({"archetype": archetype_ids})
        for li, lv in enumerate(keep_levels):
            df[f"OR_{lv}"] = ors[:, li]
            df[f"p_{lv}"] = ps[:, li]
            df[f"q_{lv}"] = qs[:, li]
        out[cov] = df

    return out
```

- [ ] **Step 4: Verify**

```bash
conda run -n archetype python -m pytest tests/test_paper_part2_step1.py -k hypergeometric -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/_paper_part1_viz.py tests/test_paper_part2_step1.py
git commit -m "Part 2: add build_archetype_hypergeometric_tables"
```

---

## Task 7: Add `build_holdout_projection_qc` to `_paper_part1_viz.py`

**Files:**
- Modify: `scripts/_paper_part1_viz.py`
- Test: `tests/test_paper_part2_step1.py` (append)

**Purpose:** Spec §5.2 — compute train vs holdout archetypal R² and per-cell NN distance to nearest archetype.

- [ ] **Step 1: Append tests**

```python
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
    from _paper_part1_viz import build_holdout_projection_qc

    rng = np.random.default_rng(0)
    K, D = 4, 10
    archetypes = rng.normal(size=(K, D))
    cells_train = rng.normal(size=(200, D))
    weights_train = rng.dirichlet([1.0] * K, size=200)
    recon_train = weights_train @ archetypes

    cells_holdout = rng.normal(size=(200, D)) * 5.0   # 5x noisier
    weights_holdout = rng.dirichlet([1.0] * K, size=200)
    recon_holdout = weights_holdout @ archetypes

    qc = build_holdout_projection_qc(
        cells_train, recon_train, cells_holdout, recon_holdout, archetypes
    )
    # Holdout R² should be much worse than train because reconstruction
    # doesn't match high-noise cells
    assert qc["holdout_r2"] < qc["train_r2"]
```

- [ ] **Step 2: Run, expect failure**

```bash
conda run -n archetype python -m pytest tests/test_paper_part2_step1.py -k holdout_projection -v
```
Expected: 2 FAIL.

- [ ] **Step 3: Implement**

Append to `scripts/_paper_part1_viz.py`:

```python
def build_holdout_projection_qc(
    cells_train,
    reconstruction_train,
    cells_holdout,
    reconstruction_holdout,
    archetype_positions,
) -> dict:
    """Archetypal R² on train + holdout, plus per-cell NN distance to the
    nearest archetype position for the holdout set.

    Parameters
    ----------
    cells_train, cells_holdout : np.ndarray
        Shape ``(n_cells, n_dims)`` in the same coord space as
        ``archetype_positions`` (typically PCA or a learned latent).
    reconstruction_train, reconstruction_holdout : np.ndarray
        Same shape — the archetype-weighted reconstructions
        (``weights @ archetype_positions``).
    archetype_positions : np.ndarray
        Shape ``(K, n_dims)``.

    Returns
    -------
    dict : ``train_r2``, ``holdout_r2``, ``holdout_mean_nn_dist``,
           ``holdout_median_nn_dist``.
    """
    import numpy as np

    def _arch_r2(original, recon):
        # Mirrors peach.calculate_archetype_r2 semantics — per-feature mean
        # centering, scalar ss_tot if >1D.
        ss_res = float(np.sum((original - recon) ** 2))
        ss_tot = float(np.sum((original - original.mean(axis=0)) ** 2))
        return 1.0 - ss_res / max(ss_tot, 1e-12)

    train_r2 = _arch_r2(cells_train, reconstruction_train)
    holdout_r2 = _arch_r2(cells_holdout, reconstruction_holdout)

    # NN distance from each holdout cell to nearest archetype
    diff = cells_holdout[:, None, :] - archetype_positions[None, :, :]
    dists = np.linalg.norm(diff, axis=2)  # (n_holdout, K)
    nn = dists.min(axis=1)

    return {
        "train_r2": train_r2,
        "holdout_r2": holdout_r2,
        "holdout_mean_nn_dist": float(nn.mean()),
        "holdout_median_nn_dist": float(np.median(nn)),
    }
```

- [ ] **Step 4: Verify**

```bash
conda run -n archetype python -m pytest tests/test_paper_part2_step1.py -k holdout_projection -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/_paper_part1_viz.py tests/test_paper_part2_step1.py
git commit -m "Part 2: add build_holdout_projection_qc"
```

---

## Task 8: Add `build_distance_heatmaps` to `_paper_part1_viz.py`

**Files:**
- Modify: `scripts/_paper_part1_viz.py`
- Test: `tests/test_paper_part2_step1.py` (append)

**Purpose:** Spec §5.4.2 — render the gated (3K × 3K) W2 and Euclidean-centroid heatmaps side-by-side, return Spearman agreement.

- [ ] **Step 1: Append test**

```python
# ============================================================================
# Task 8 — build_distance_heatmaps
# ============================================================================


def test_distance_heatmaps_returns_figure_and_spearman():
    from _paper_part1_viz import build_distance_heatmaps
    import pandas as pd
    import numpy as np

    rng = np.random.default_rng(0)
    n = 900
    K = 3
    obs = pd.DataFrame({
        "response_group": rng.choice(["NR", "R1", "R2"], n),
        "archetypes": rng.integers(0, K, n),
    })
    weights = rng.dirichlet([1.0] * K, n)
    pca = rng.normal(size=(n, 8))

    fig, spearman_rho = build_distance_heatmaps(
        obs, weights, pca,
        response_col="response_group",
        archetypes_col="archetypes",
    )
    assert fig is not None
    assert -1.0 <= spearman_rho <= 1.0
```

- [ ] **Step 2: Run, expect failure**

```bash
conda run -n archetype python -m pytest tests/test_paper_part2_step1.py -k distance_heatmaps -v
```
Expected: 1 FAIL.

- [ ] **Step 3: Implement**

Append to `scripts/_paper_part1_viz.py`:

```python
def build_distance_heatmaps(
    obs,
    weights,
    pca,
    response_col: str,
    archetypes_col: str,
):
    """Return (plotly.go.Figure, spearman_rho) for Fig 3C-i.

    Two side-by-side heatmaps of size (3K × 3K), rows/cols =
    (response_group, archetype) pairs:
      - Left: W2 in archetype-weight simplex
      - Right: Euclidean centroid distance in PCA space
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.stats import spearmanr

    obs = obs.reset_index(drop=True)
    # Build group index: (response, archetype)
    obs["_grp"] = list(zip(obs[response_col], obs[archetypes_col]))
    groups = sorted(obs["_grp"].unique(), key=lambda t: (str(t[0]), int(t[1])))
    idx_of = {g: np.where(obs["_grp"].values == g)[0] for g in groups}
    # Drop groups with <2 cells (can't compute W2)
    groups = [g for g in groups if len(idx_of[g]) >= 2]
    G = len(groups)

    w2_mat = np.zeros((G, G))
    eu_mat = np.zeros((G, G))
    for i, gi in enumerate(groups):
        for j, gj in enumerate(groups):
            if j <= i:
                continue
            wi = weights[idx_of[gi]]
            wj = weights[idx_of[gj]]
            w2 = compute_w2_archetype_distance(wi, wj)
            w2_mat[i, j] = w2_mat[j, i] = w2
            pi = pca[idx_of[gi]].mean(axis=0)
            pj = pca[idx_of[gj]].mean(axis=0)
            d_eu = float(np.linalg.norm(pi - pj))
            eu_mat[i, j] = eu_mat[j, i] = d_eu

    # Spearman on the upper triangle
    iu = np.triu_indices(G, k=1)
    if len(iu[0]) >= 3:
        rho, _ = spearmanr(w2_mat[iu], eu_mat[iu])
    else:
        rho = float("nan")

    labels = [f"{r}/A{int(a)}" for r, a in groups]
    fig = make_subplots(rows=1, cols=2,
                         subplot_titles=("W2 (archetype weights)",
                                         "Euclidean centroid (PCA space)"))
    fig.add_trace(
        go.Heatmap(z=w2_mat, x=labels, y=labels, colorscale="Viridis",
                     showscale=True, colorbar=dict(x=0.43, len=0.75)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(z=eu_mat, x=labels, y=labels, colorscale="Plasma",
                     showscale=True, colorbar=dict(x=1.02, len=0.75)),
        row=1, col=2,
    )
    fig.update_layout(
        title=f"(response × archetype) pairwise distances — Spearman ρ = {rho:.3f}",
        height=520, width=1200,
    )
    return fig, float(rho) if not np.isnan(rho) else rho
```

- [ ] **Step 4: Verify**

```bash
conda run -n archetype python -m pytest tests/test_paper_part2_step1.py -k distance_heatmaps -v
```
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/_paper_part1_viz.py tests/test_paper_part2_step1.py
git commit -m "Part 2: add build_distance_heatmaps"
```

---

## Task 9: Add `build_diversity_block` to `_paper_part1_viz.py`

**Files:**
- Modify: `scripts/_paper_part1_viz.py`
- Test: `tests/test_paper_part2_step1.py` (append)

**Purpose:** Spec §5.4.3 — three-panel diversity figure with Kruskal-Wallis + pairwise Dunn + bootstrap CI bar.

- [ ] **Step 1: Append tests**

```python
# ============================================================================
# Task 9 — build_diversity_block
# ============================================================================


def test_diversity_block_returns_figure_and_summary():
    from _paper_part1_viz import build_diversity_block
    import pandas as pd
    import numpy as np

    rng = np.random.default_rng(0)
    n = 600
    K = 4
    obs = pd.DataFrame({"response_group": rng.choice(["NR", "R1", "R2"], n)})
    # Uneven diversity: NR has high entropy Dirichlet, R2 has peaked
    weights = np.vstack([
        rng.dirichlet([1.0] * K) if r == "NR"
        else rng.dirichlet([5.0, 1.0, 1.0, 1.0]) if r == "R1"
        else rng.dirichlet([10.0, 1.0, 1.0, 1.0])
        for r in obs["response_group"]
    ])
    pca = rng.normal(size=(n, 8))

    fig, summary = build_diversity_block(
        obs, weights, pca,
        group_col="response_group",
        bootstrap_n=50, subsample=100, random_state=0,
    )
    assert fig is not None
    for key in ["per_cell_shannon_kw_stat", "per_cell_shannon_kw_p",
                "per_group_pca_dispersion", "per_group_archetype_entropy"]:
        assert key in summary
    assert set(summary["per_group_pca_dispersion"].keys()) == {"NR", "R1", "R2"}
    assert set(summary["per_group_archetype_entropy"].keys()) == {"NR", "R1", "R2"}
```

- [ ] **Step 2: Run, expect failure**

```bash
conda run -n archetype python -m pytest tests/test_paper_part2_step1.py -k diversity_block -v
```
Expected: 1 FAIL.

- [ ] **Step 3: Implement**

Append to `scripts/_paper_part1_viz.py`:

```python
def build_diversity_block(
    obs,
    weights,
    pca,
    group_col: str,
    bootstrap_n: int = 200,
    subsample: int = 500,
    random_state: int = 42,
):
    """Fig 3C-ii — three-panel diversity block (spec §5.4.3).

    Returns
    -------
    fig : plotly.graph_objects.Figure (3 panels)
    summary : dict with KW stat + per-group scalars for panels 2 and 3.
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.stats import kruskal, entropy
    try:
        import scikit_posthocs as sp
        have_dunn = True
    except ImportError:
        have_dunn = False

    rng = np.random.default_rng(random_state)
    obs = obs.reset_index(drop=True)
    groups = sorted(obs[group_col].unique())

    # Panel 1 — per-cell Shannon H of weights
    per_cell_H = np.array([entropy(w + 1e-12) for w in weights])

    # Panel 2 — per-group PCA median pairwise dispersion with bootstrap CI
    def _median_pairwise_dist(X, max_cells):
        if X.shape[0] > max_cells:
            idx = rng.choice(X.shape[0], size=max_cells, replace=False)
            X = X[idx]
        from scipy.spatial.distance import pdist
        dists = pdist(X, metric="euclidean")
        return float(np.median(dists))

    per_group_disp = {}
    disp_ci = {}
    for g in groups:
        mask = obs[group_col].values == g
        Xg = pca[mask]
        est = _median_pairwise_dist(Xg, subsample)
        boot = []
        for _ in range(bootstrap_n):
            if Xg.shape[0] < 2:
                boot.append(float("nan"))
                continue
            sample_idx = rng.integers(0, Xg.shape[0], size=Xg.shape[0])
            boot.append(_median_pairwise_dist(Xg[sample_idx], subsample))
        boot_arr = np.asarray([b for b in boot if not np.isnan(b)])
        ci_lo = float(np.percentile(boot_arr, 2.5)) if len(boot_arr) else float("nan")
        ci_hi = float(np.percentile(boot_arr, 97.5)) if len(boot_arr) else float("nan")
        per_group_disp[g] = est
        disp_ci[g] = (ci_lo, ci_hi)

    # Panel 3 — entropy of pooled mean weight vector
    per_group_arch_H = {}
    for g in groups:
        mask = obs[group_col].values == g
        mu = weights[mask].mean(axis=0)
        per_group_arch_H[g] = float(entropy(mu + 1e-12))

    # Stats — Kruskal-Wallis on per-cell Shannon
    kw_stat, kw_p = kruskal(*[per_cell_H[obs[group_col].values == g] for g in groups])

    # Dunn pairwise post-hoc (optional)
    dunn_df = None
    if have_dunn:
        df_long = pd.DataFrame({"entropy": per_cell_H, "group": obs[group_col].values})
        dunn_df = sp.posthoc_dunn(df_long, val_col="entropy", group_col="group",
                                   p_adjust="fdr_bh")

    # Build figure — 3 panels
    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        "Per-cell Shannon H (weights)",
        "Per-group PCA dispersion (bootstrap)",
        "Per-group entropy of mean archetype profile",
    ))
    # Panel 1: violin
    for g in groups:
        mask = obs[group_col].values == g
        fig.add_trace(go.Violin(
            y=per_cell_H[mask], name=str(g), points="outliers",
            box_visible=True, showlegend=False,
        ), row=1, col=1)
    # Panel 2: bar with CI
    xs = list(groups)
    ys = [per_group_disp[g] for g in xs]
    err_lo = [per_group_disp[g] - disp_ci[g][0] for g in xs]
    err_hi = [disp_ci[g][1] - per_group_disp[g] for g in xs]
    fig.add_trace(go.Bar(
        x=xs, y=ys,
        error_y=dict(type="data", array=err_hi, arrayminus=err_lo, visible=True),
        showlegend=False,
    ), row=1, col=2)
    # Panel 3: bar
    fig.add_trace(go.Bar(
        x=xs, y=[per_group_arch_H[g] for g in xs], showlegend=False,
    ), row=1, col=3)
    fig.update_layout(
        title=f"Diversity block — KW H={kw_stat:.2f}, p={kw_p:.2e}",
        height=480, width=1400,
    )

    summary = {
        "per_cell_shannon_kw_stat": float(kw_stat),
        "per_cell_shannon_kw_p": float(kw_p),
        "per_group_pca_dispersion": per_group_disp,
        "per_group_pca_dispersion_ci": disp_ci,
        "per_group_archetype_entropy": per_group_arch_H,
        "dunn_posthoc": dunn_df.to_dict() if dunn_df is not None else None,
    }
    return fig, summary
```

- [ ] **Step 4: Verify**

```bash
conda run -n archetype python -m pytest tests/test_paper_part2_step1.py -k diversity_block -v
```
Expected: 1 passed. If `scikit_posthocs` is missing, test still passes (Dunn is optional).

- [ ] **Step 5: Commit**

```bash
git add scripts/_paper_part1_viz.py tests/test_paper_part2_step1.py
git commit -m "Part 2: add build_diversity_block (three-panel)"
```

---

## Task 10: Write `prep_tnbcrad.py` and run it

**Files:**
- Create: `scripts/prep_tnbcrad.py`
- Test: `tests/test_paper_part2_step1.py` (append structural tests)

- [ ] **Step 1: Append structural tests**

```python
# ============================================================================
# Task 10 — prep_tnbcrad.py structural tests
# ============================================================================


def _read_prep_source() -> str:
    path = REPO_ROOT / "scripts" / "prep_tnbcrad.py"
    return path.read_text() if path.exists() else ""


def test_prep_tnbcrad_no_forbidden_calls():
    src = _read_prep_source()
    assert src, "prep_tnbcrad.py missing"
    assert "sc.pp.normalize_total" not in src
    assert "sc.pp.scale" not in src
    assert "highly_variable_genes" not in src


def test_prep_tnbcrad_required_call_sites():
    src = _read_prep_source()
    assert "apply_mt_rb_mad_filter" in src
    assert "n_mads=3.0" in src
    assert "sc.pp.pca" in src
    assert "n_comps=50" in src
    assert "zero_center=False" in src
    assert "safe_stratified_split" in src
    assert 'adata.layers["logcounts"]' in src
    assert "N_PCS = 12" in src


def test_prep_tnbcrad_writes_three_outputs():
    src = _read_prep_source()
    # full_prepped + train + holdout
    for name in ("adata_tnbc_full_prepped.h5ad",
                 "adata_tnbc_train.h5ad",
                 "adata_tnbc_holdout.h5ad"):
        assert name in src, f"prep must write {name}"
```

- [ ] **Step 2: Run, expect failure**

```bash
conda run -n archetype python -m pytest tests/test_paper_part2_step1.py -k prep_tnbcrad -v
```
Expected: 3 FAIL (file missing).

- [ ] **Step 3: Create `scripts/prep_tnbcrad.py`**

```python
"""TNBC prep pipeline — Part 2 Step 1 input.

Mirrors prep_hsccmp.py recipe. Input: data/GSE246613_TNBC_ONLY_TRAIN.h5ad
(31,503 malignant cells by inferCNV). Output: three h5ad files in
data/paper_part2/ + a lightweight prep_report.html logging diagnostics.

Recipe (no deviations):
  1. adata.X = adata.layers["logcounts"].copy()  (no normalize, no scale)
  2. Coerce cohort/treatment/response_group/majority_voting -> category
  3. apply_mt_rb_mad_filter(n_mads=3.0)
  4. sc.pp.pca(n_comps=50, zero_center=False, use_highly_variable=False)
  5. Slice to N_PCS=12  (scree-locked 2026-04-14)
  6. safe_stratified_split on cohort × response × treatment (fallback: cohort)
  7. Write full_prepped / train / holdout + prep_report.html
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display on remote runs
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _paper_part1_prep import apply_mt_rb_mad_filter, safe_stratified_split  # noqa: E402

INPUT_PATH = REPO_ROOT / "data" / "GSE246613_TNBC_ONLY_TRAIN.h5ad"
OUT_DIR = REPO_ROOT / "data" / "paper_part2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PCS = 12                    # scree-locked 2026-04-14
MAD_N_MADS = 3.0
TEST_SIZE = 0.20
MIN_STRATUM_SIZE = 10
RANDOM_STATE = 42


def main() -> None:
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Loading {INPUT_PATH} ...")
    adata = sc.read_h5ad(INPUT_PATH)
    print(f"  Loaded {adata.shape}")

    # 1. logcounts -> X (no normalize, no scale)
    adata.X = adata.layers["logcounts"].copy()

    # 2. Categorical coercion
    for col in ("cohort", "treatment", "response_group", "majority_voting"):
        if col in adata.obs.columns:
            adata.obs[col] = adata.obs[col].astype("category")

    # 3. MAD filter
    n_in = adata.n_obs
    print(f"[{time.strftime('%H:%M:%S')}] MAD filter ...")
    adata = apply_mt_rb_mad_filter(adata, n_mads=MAD_N_MADS)
    n_dropped = n_in - adata.n_obs
    print(f"  After MAD filter: {adata.shape}  (dropped {n_dropped} cells)")

    # 4. PCA
    print(f"[{time.strftime('%H:%M:%S')}] Running sc.pp.pca(n_comps=50) ...")
    sc.pp.pca(adata, n_comps=50, zero_center=False, use_highly_variable=False)

    # 5. Slice to N_PCS=12
    full_var_ratio = np.asarray(adata.uns["pca"]["variance_ratio"])
    full_var = np.asarray(adata.uns["pca"]["variance"])
    adata.obsm["X_pca"] = adata.obsm["X_pca"][:, :N_PCS]
    adata.uns["pca"]["variance_ratio"] = full_var_ratio[:N_PCS]
    adata.uns["pca"]["variance"] = full_var[:N_PCS]

    # 6. Stratified 80/20 split
    primary = (adata.obs["cohort"].astype(str) + "|" +
               adata.obs["response_group"].astype(str) + "|" +
               adata.obs["treatment"].astype(str))
    fallback = adata.obs["cohort"].astype(str)
    train_idx, holdout_idx, diag = safe_stratified_split(
        primary, fallback,
        test_size=TEST_SIZE, min_stratum_size=MIN_STRATUM_SIZE,
        random_state=RANDOM_STATE,
    )
    print(f"[{time.strftime('%H:%M:%S')}] Split: "
          f"train={len(train_idx)} holdout={len(holdout_idx)} "
          f"fallback_cells={diag['n_fallback_cells']} "
          f"random_fallback={diag['n_random_fallback_cells']}")

    # 7. Write outputs
    full_path = OUT_DIR / "adata_tnbc_full_prepped.h5ad"
    train_path = OUT_DIR / "adata_tnbc_train.h5ad"
    holdout_path = OUT_DIR / "adata_tnbc_holdout.h5ad"
    print(f"[{time.strftime('%H:%M:%S')}] Writing {full_path}")
    adata.write_h5ad(full_path)
    print(f"[{time.strftime('%H:%M:%S')}] Writing {train_path}")
    adata[train_idx].copy().write_h5ad(train_path)
    print(f"[{time.strftime('%H:%M:%S')}] Writing {holdout_path}")
    adata[holdout_idx].copy().write_h5ad(holdout_path)

    # Scree sanity plot in output dir
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, 51), full_var_ratio, "o-")
    ax.axvline(N_PCS, ls="--", color="red", label=f"N_PCS={N_PCS}")
    ax.set_xlabel("PC")
    ax.set_ylabel("variance ratio")
    ax.set_title("TNBC prep PCA scree (50 PCs)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "prep_scree.png", dpi=140)
    plt.close(fig)

    # Minimal provenance HTML
    cumvar_at_k = float(full_var_ratio[:N_PCS].sum())
    prep_html = OUT_DIR / "prep_report.html"
    prep_html.write_text(f"""<!DOCTYPE html><html><body>
<h1>TNBC Prep Report</h1>
<p>Date: {time.strftime('%Y-%m-%d %H:%M')}</p>
<h2>Pipeline</h2>
<ul>
  <li>Input: {INPUT_PATH.name}</li>
  <li>MAD filter (n_mads={MAD_N_MADS}): {n_in} -> {adata.n_obs} cells (dropped {n_dropped})</li>
  <li>PCA: 50 components, no scale, no HVG, zero_center=False</li>
  <li>N_PCS sliced to {N_PCS}, cumulative variance = {cumvar_at_k:.3f}</li>
  <li>Stratified split (cohort × response × treatment, fallback cohort):
      train={len(train_idx)} holdout={len(holdout_idx)}
      fallback_cells={diag['n_fallback_cells']}
      random_fallback_cells={diag['n_random_fallback_cells']}
      collapsed_strata={diag['n_collapsed_strata']}
  </li>
</ul>
<p>Elapsed: {time.time() - t0:.1f}s</p>
</body></html>
""")
    print(f"[{time.strftime('%H:%M:%S')}] Done. Elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run structural tests**

```bash
conda run -n archetype python -m pytest tests/test_paper_part2_step1.py -k prep_tnbcrad -v
```
Expected: 3 passed.

- [ ] **Step 5: Actually run the prep pipeline**

```bash
conda run -n archetype python -u scripts/prep_tnbcrad.py
```
Expected: writes 3 h5ad files + prep_scree.png + prep_report.html into `data/paper_part2/`. Takes ~5 minutes. Look for the final "Done. Elapsed..." line.

- [ ] **Step 6: Sanity check the outputs**

```bash
conda run -n archetype python -c "
import anndata
for name in ['adata_tnbc_full_prepped.h5ad', 'adata_tnbc_train.h5ad', 'adata_tnbc_holdout.h5ad']:
    a = anndata.read_h5ad(f'data/paper_part2/{name}', backed='r')
    print(name, '->', a.shape, 'X_pca shape:', a.obsm['X_pca'].shape)
"
```
Expected: full_prepped = ~(26395, 36403), train ≈ 80% of that, holdout ≈ 20%, all with `X_pca` shape (N, 12).

- [ ] **Step 7: Commit (script + data artefacts if user wants provenance; otherwise just script)**

```bash
git add scripts/prep_tnbcrad.py tests/test_paper_part2_step1.py
git commit -m "Part 2: prep_tnbcrad.py (MAD filter + PCA-12 + stratified holdout)"
```

Note: `data/paper_part2/` outputs may be too large to commit — check with user. If small (<100MB total) and `data/paper_part1/` precedent includes them, commit; otherwise `.gitignore` it.

---

## Task 11: `run_paper_part2_tnbc.py` — scaffold + Phase 1 (training + QC)

**Files:**
- Create: `scripts/run_paper_part2_tnbc.py`
- Test: `tests/test_paper_part2_step1.py` (append structural tests)

This task builds the script scaffolding and the training phase. Phases 2 and 3 are Tasks 12 and 13.

- [ ] **Step 1: Append structural tests**

```python
# ============================================================================
# Task 11 — run_paper_part2_tnbc.py scaffold + Phase 1
# ============================================================================


def _read_run_source() -> str:
    path = REPO_ROOT / "scripts" / "run_paper_part2_tnbc.py"
    return path.read_text() if path.exists() else ""


def test_run_part2_prototype_banner():
    src = _read_run_source()
    assert src, "run_paper_part2_tnbc.py missing"
    assert "Step 1 of 5" in src
    # Must reference the downstream steps explicitly so forks see the map
    for step in ("Step 2", "Step 3", "Step 4", "Step 5"):
        assert step in src


def test_run_part2_config_constants():
    src = _read_run_source()
    for const in (
        "SUBSAMPLE_FRACTION = 0.2",
        "MAX_EPOCHS_FINAL = 200",
        "N_PCS = 12",
        'SUBSAMPLE_STRATIFY = "response_group"',
        "EARLY_STOP_PATIENCE = 15",
        "K_RANGE = list(range(3, 11))",
    ):
        assert const in src, f"missing config constant: {const}"


def test_run_part2_phase1_calls():
    src = _read_run_source()
    assert "phase1_train_model" in src
    assert "pc.tl.hyperparameter_search" in src
    assert "pc.tl.train_archetypal" in src
    assert "pc.tl.archetypal_coordinates" in src
    assert "pc.tl.assign_archetypas" not in src  # typo check
    assert "pc.tl.assign_archetypes" in src
    assert "build_drift_qc_panel" in src
    assert "build_holdout_projection_qc" in src
```

- [ ] **Step 2: Run, expect failure**

```bash
conda run -n archetype python -m pytest tests/test_paper_part2_step1.py -k run_part2 -v
```
Expected: 3 FAIL.

- [ ] **Step 3: Create `scripts/run_paper_part2_tnbc.py`**

```python
"""Paper Part 2 — Step 1: Global archetypal fit on TNBC tumor cells.

*** PROTOTYPE — Step 1 of 5 in Paper Part 2 ***

This script stands up the "does R1/R2/NR segregate in a single global
archetype space?" analysis only. It does NOT cover:

  Step 2 — Per-timepoint × per-response models (6 total): specialist
           selection via archetype relatedness (MMD, Spearman, Wald).
           Uses the global fit from this script as a reference frame.
  Step 3 — Flow along treatments (Base -> PD1 -> RTPD1): expanding /
           contracting features, stress-gene subset (Fig 4C).
  Step 4 — R vs NR contrasts per treatment + per-patient centroid
           trajectories in global space (Fig 5).
  Step 5 — Held-out-patient prediction via LOPO lasso (separate script).

Structure mirrors run_paper_part1_hsc.py: Phase 1 training, Phase 2 Fig 3A,
Phase 3 Fig 3B+3C. Helpers imported from _paper_part1_{prep,viz}.py.

SUBSAMPLE_FRACTION = 0.2 for prototyping; flip to 1.0 for production.
"""
from __future__ import annotations

import glob
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import peach as pc  # noqa: E402

from _paper_part1_viz import (  # noqa: E402
    build_archetype_char_table,
    build_archetype_hypergeometric_tables,
    build_distance_heatmaps,
    build_diversity_block,
    build_drift_qc_panel,
    build_holdout_projection_qc,
    build_response_timepoint_colormap,
    build_segregation_ratio,
    compute_w2_archetype_distance,
    convergence_status,
    dotplot_figsize,
)
from stress_genes.load_stress_genes import STRESS_GENES_FLAT  # noqa: E402


# ============================================================================
# Config
# ============================================================================
DATA_DIR = REPO_ROOT / "data" / "paper_part2"
OUTPUT_DIR = REPO_ROOT / "outputs" / "paper_part2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUBSAMPLE_FRACTION = 0.2
SUBSAMPLE_STRATIFY = "response_group"

MAX_EPOCHS_FINAL = 200
EARLY_STOP_PATIENCE = 15
N_PCS = 12

K_RANGE = list(range(3, 11))
HIDDEN_DIMS_OPTIONS = [[64, 128], [128, 256], [256, 128, 64]]
INFLATION_FACTOR_RANGE = [0.75, 1.0, 1.25, 1.5]

MODEL_CONFIG = {
    "manifold_weight": 0.005,
    "kld_weight": 0.01,
    "sparsity_weight": 0.0,
    "archetypal_weight": 0.9,
}

FDR_THRESHOLD = 0.05
EXCLUSIVE_RATIO_THRESHOLD = 2.5
FIG3C_GATE_THRESHOLD = 1.3

# Auto-increment rev number for today
_DATE_TAG = time.strftime("%Y%m%d")
_existing = sorted(glob.glob(str(OUTPUT_DIR / f"part2_report_{_DATE_TAG}_r*.html")))
_REV = len(_existing) + 1
REPORT_PATH = OUTPUT_DIR / f"part2_report_{_DATE_TAG}_r{_REV}.html"


# ============================================================================
# HTMLReport (inlined; mirrors run_paper_part1_hsc.py)
# ============================================================================


class HTMLReport:
    """Minimal single-file HTML report builder with <details> sections."""

    def __init__(self, title: str):
        self.title = title
        self.sections: list[str] = []

    def text(self, s: str) -> None:
        self.sections.append(f"<p>{s}</p>")

    def add_section(self, title: str, html: str, step_num: int | None = None, open_by_default: bool = False) -> None:
        num = f"{step_num}. " if step_num is not None else ""
        attr = "open" if open_by_default else ""
        self.sections.append(
            f'<details {attr}><summary><h2 style="display:inline">{num}{title}</h2></summary>{html}</details>'
        )

    def fig_to_img(self, fig, caption: str, dpi: int = 150) -> str:
        import base64, io
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f'<figure><img src="data:image/png;base64,{b64}"/><figcaption>{caption}</figcaption></figure>'

    def plotly_to_div(self, fig, caption: str) -> str:
        inline = fig.to_html(full_html=False, include_plotlyjs="cdn")
        return f"<figure>{inline}<figcaption>{caption}</figcaption></figure>"

    def df_to_html(self, df, caption: str, max_rows: int = 50) -> str:
        try:
            html = df.head(max_rows).to_html(index=False, float_format=lambda x: f"{x:.3g}")
        except Exception as e:
            html = f"<em>Error rendering table: {e}</em>"
        return f"<figure>{html}<figcaption>{caption}</figcaption></figure>"

    def save(self, path: Path) -> None:
        body = "".join(self.sections)
        html = f"""<!DOCTYPE html><html><head>
<meta charset="utf-8"><title>{self.title}</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;max-width:1200px}}
h1{{border-bottom:2px solid #333;padding-bottom:4px}}
details{{margin:12px 0;padding:8px;border:1px solid #ddd;border-radius:6px}}
figure{{margin:12px 0}}img{{max-width:100%;height:auto}}
table{{border-collapse:collapse;margin:8px 0}}
th,td{{border:1px solid #ccc;padding:4px 8px;font-size:0.9em}}
th{{background:#f3f4f6}}</style>
</head><body><h1>{self.title}</h1>{body}</body></html>"""
        Path(path).write_text(html)


def error_html(msg: str) -> str:
    return f'<div style="padding:8px;background:#fee;border-left:4px solid #c33;color:#900">{msg}</div>'


def metric_card(label: str, value, fmt: str = ".3g") -> str:
    try:
        rendered = format(value, fmt)
    except (TypeError, ValueError):
        rendered = str(value)
    return (f'<div style="display:inline-block;padding:8px;margin:4px;'
            f'border:1px solid #ddd;border-radius:6px;min-width:140px">'
            f'<div style="color:#666;font-size:0.8em">{label}</div>'
            f'<div style="font-weight:bold;font-size:1.2em">{rendered}</div></div>')


def metric_grid(cards: list[str]) -> str:
    return f'<div style="display:flex;flex-wrap:wrap">{"".join(cards)}</div>'


# ============================================================================
# Helpers (script-local; don't promote)
# ============================================================================


def _stratified_subsample(adata, frac: float, stratify_col: str, seed: int = 0):
    """Per-stratum subsample. Copies the Part 1 helper so this script is standalone."""
    rng = np.random.default_rng(seed)
    keep_idx = []
    for level in adata.obs[stratify_col].unique():
        pool = np.where(adata.obs[stratify_col].values == level)[0]
        n = max(1, int(round(len(pool) * frac)))
        keep_idx.append(rng.choice(pool, size=min(n, len(pool)), replace=False))
    keep_idx = np.sort(np.concatenate(keep_idx))
    return adata[keep_idx].copy()


def _smallest_k_above_threshold(cv_summary, threshold: float) -> int | None:
    """Return smallest K in CV summary with mean_archetype_r2 >= threshold,
    else the K with best R². Mirrors Part 1 convention."""
    rows = cv_summary.summary_df.copy()
    rows = rows.sort_values("n_archetypes")
    above = rows[rows["mean_archetype_r2"] >= threshold]
    if len(above):
        return int(above.iloc[0]["n_archetypes"])
    return int(rows.sort_values("mean_archetype_r2", ascending=False).iloc[0]["n_archetypes"])


# ============================================================================
# Phase 1
# ============================================================================


def phase1_train_model(report: HTMLReport):
    """Load train/holdout, CV search, final fit, drift+holdout QC."""
    t_phase = time.time()
    adata_train = sc.read_h5ad(DATA_DIR / "adata_tnbc_train.h5ad")
    adata_holdout = sc.read_h5ad(DATA_DIR / "adata_tnbc_holdout.h5ad")
    print(f"  phase1: loaded train={adata_train.shape} holdout={adata_holdout.shape}")

    if SUBSAMPLE_FRACTION < 1.0:
        adata_train = _stratified_subsample(
            adata_train, SUBSAMPLE_FRACTION, SUBSAMPLE_STRATIFY, seed=42
        )
        print(f"  phase1: subsampled train -> {adata_train.shape}")

    # 1a — CV search
    cv = pc.tl.hyperparameter_search(
        adata_train,
        pca_key="X_pca",
        n_archetypes_range=K_RANGE,
        hidden_dims_options=HIDDEN_DIMS_OPTIONS,
        inflation_factor_range=INFLATION_FACTOR_RANGE,
        use_pcha_init=False,
        max_epochs=20,
        n_folds=5,
    )
    K_pick = _smallest_k_above_threshold(cv, threshold=0.9)
    best_df = cv.summary_df.sort_values("mean_archetype_r2", ascending=False)
    best_row = best_df[best_df["n_archetypes"] == K_pick].iloc[0]
    print(f"  phase1: CV picked K={K_pick} from {K_RANGE}")

    # 1b — final fit
    res = pc.tl.train_archetypal(
        adata_train,
        n_archetypes=K_pick,
        pca_key="X_pca",
        n_epochs=MAX_EPOCHS_FINAL,
        early_stop_patience=EARLY_STOP_PATIENCE,
        pcha_init=True,
        model_config={
            **MODEL_CONFIG,
            "hidden_dims": list(best_row["hidden_dims"]),
            "inflation_factor": float(best_row["inflation_factor"]),
        },
    )

    # 1c — extract coords + assignments on train
    pc.tl.archetypal_coordinates(adata_train, training_results=res)
    pc.tl.assign_archetypes(adata_train, percentage_per_archetype=0.15)

    # 1d — project holdout through trained model
    model = res.get("model") or res.get("final_model")
    pc.tl.extract_archetype_weights(adata_holdout, model=model, pca_key="X_pca")
    pc.tl.archetypal_coordinates(adata_holdout, training_results=res)

    # 1e — Phase 1 section
    html_parts: list[str] = []

    # Config card grid
    cards = [
        metric_card("K (picked)", K_pick, "d"),
        metric_card("N cells (train)", adata_train.n_obs, "d"),
        metric_card("N cells (holdout)", adata_holdout.n_obs, "d"),
        metric_card("N PCs", N_PCS, "d"),
        metric_card("Max epochs", MAX_EPOCHS_FINAL, "d"),
        metric_card("Subsample", SUBSAMPLE_FRACTION, ".2f"),
        metric_card("Train R²", res.get("final_archetype_r2", float("nan")), ".3f"),
    ]
    html_parts.append(metric_grid(cards))

    # CV summary
    html_parts.append(report.df_to_html(cv.summary_df, "CV search summary", max_rows=50))

    # Drift / stability
    try:
        drift_html = build_drift_qc_panel([res], drift_threshold=0.05, converged_window=10)
        html_parts.append(drift_html)
    except Exception as e:
        html_parts.append(error_html(f"drift QC failed: {e}"))

    # Convergence flag
    try:
        status = convergence_status(
            res["history"], max_epochs=MAX_EPOCHS_FINAL,
            early_stop_triggered=res.get("early_stopped", False),
            actual_epochs=len(res["history"].get("loss", [])),
        )
        html_parts.append(
            f"<p><strong>Convergence status:</strong> {status['status']} "
            f"(Δloss={status.get('delta_mean', float('nan')):.4g})</p>"
        )
    except Exception as e:
        html_parts.append(error_html(f"convergence_status failed: {e}"))

    # Holdout projection QC
    try:
        archetype_positions = np.asarray(res["archetype_coords"])
        weights_train = adata_train.obsm["cell_archetype_weights"]
        weights_holdout = adata_holdout.obsm["cell_archetype_weights"]
        recon_train = weights_train @ archetype_positions
        recon_holdout = weights_holdout @ archetype_positions
        qc = build_holdout_projection_qc(
            adata_train.obsm["X_pca"], recon_train,
            adata_holdout.obsm["X_pca"], recon_holdout,
            archetype_positions,
        )
        html_parts.append(metric_grid([
            metric_card("Train R² (manual)", qc["train_r2"], ".3f"),
            metric_card("Holdout R²", qc["holdout_r2"], ".3f"),
            metric_card("Holdout mean NN dist", qc["holdout_mean_nn_dist"], ".3f"),
            metric_card("Holdout median NN dist", qc["holdout_median_nn_dist"], ".3f"),
        ]))
    except Exception as e:
        html_parts.append(error_html(f"holdout projection QC failed: {e}"))

    # PC1 correlation scan
    try:
        pc1 = np.asarray(adata_train.obsm["X_pca"][:, 0])
        from scipy.stats import spearmanr
        rows_pc1 = []
        for col in ("cohort", "treatment", "response_group"):
            codes = adata_train.obs[col].astype("category").cat.codes.values
            rho, p = spearmanr(pc1, codes)
            rows_pc1.append({"covariate": col, "spearman_rho": rho, "p": p})
        for col in ("total_counts", "percent_mito", "percent_ribo"):
            if col in adata_train.obs.columns:
                rho, p = spearmanr(pc1, adata_train.obs[col].values)
                rows_pc1.append({"covariate": col, "spearman_rho": rho, "p": p})
        pc1_df = pd.DataFrame(rows_pc1).sort_values("spearman_rho",
                                                     key=lambda s: s.abs(), ascending=False)
        html_parts.append(report.df_to_html(
            pc1_df, "PC1 correlation scan (Spearman). Large |ρ| on cohort = batch-like PC1."
        ))
    except Exception as e:
        html_parts.append(error_html(f"PC1 scan failed: {e}"))

    report.add_section("Phase 1: Training + QC",
                        "\n".join(html_parts),
                        step_num=1, open_by_default=True)
    print(f"  phase1: done in {time.time() - t_phase:.1f}s")
    return adata_train, adata_holdout, res


# ============================================================================
# Phase 2 and Phase 3 stubs — filled in Tasks 12 and 13
# ============================================================================


def phase2_figure3a(adata_train, adata_holdout, res, report: HTMLReport):
    """Fig 3A — global archetype space. Implemented in Task 12."""
    report.add_section("Phase 2: Fig 3A (stub)",
                        "<p><em>Pending Task 12.</em></p>", step_num=2)


def phase3_figure3bc(adata_train, res, report: HTMLReport):
    """Fig 3B + gated Fig 3C. Implemented in Task 13."""
    report.add_section("Phase 3: Fig 3B/3C (stub)",
                        "<p><em>Pending Task 13.</em></p>", step_num=3)


# ============================================================================
# main
# ============================================================================


def main() -> None:
    report = HTMLReport(f"Paper Part 2 Step 1 — TNBC Global Fit ({_DATE_TAG} r{_REV})")
    report.text(
        "<strong>Prototype run — Step 1 of 5 in Paper Part 2.</strong> "
        "This report covers the global archetypal fit on TNBC tumor cells. "
        "Steps 2–4 (per-timepoint models, flow, R vs NR contrasts) and "
        "Step 5 (held-out prediction) are separate scripts and not yet implemented. "
        f"SUBSAMPLE_FRACTION = {SUBSAMPLE_FRACTION}; flip to 1.0 for production."
    )
    t0 = time.time()

    adata_train, adata_holdout, res = phase1_train_model(report)
    phase2_figure3a(adata_train, adata_holdout, res, report)
    phase3_figure3bc(adata_train, res, report)

    report.save(REPORT_PATH)
    print(f"\n[main] total elapsed: {time.time() - t0:.1f}s")
    print(f"[main] report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify structural tests pass**

```bash
conda run -n archetype python -m pytest tests/test_paper_part2_step1.py -k run_part2 -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_paper_part2_tnbc.py tests/test_paper_part2_step1.py
git commit -m "Part 2: run script scaffold + Phase 1 training"
```

---

## Task 12: Phase 2 — Fig 3A (global archetype space)

**Files:**
- Modify: `scripts/run_paper_part2_tnbc.py` — replace `phase2_figure3a` stub
- Test: `tests/test_paper_part2_step1.py` (append)

- [ ] **Step 1: Append test**

```python
# ============================================================================
# Task 12 — Phase 2 (Fig 3A)
# ============================================================================


def test_run_part2_phase2_calls():
    src = _read_run_source()
    assert "build_response_timepoint_colormap" in src
    assert "build_archetype_char_table" in src
    assert "build_archetype_hypergeometric_tables" in src
    assert "Fig 3A" in src or "Figure 3A" in src
    assert "pc.pl.archetypal_space" in src
```

- [ ] **Step 2: Run, expect failure**

```bash
conda run -n archetype python -m pytest tests/test_paper_part2_step1.py::test_run_part2_phase2_calls -v
```
Expected: FAIL.

- [ ] **Step 3: Replace the Phase 2 stub**

In `scripts/run_paper_part2_tnbc.py`, replace the entire `phase2_figure3a` body with:

```python
def phase2_figure3a(adata_train, adata_holdout, res, report: HTMLReport):
    """Fig 3A — global archetype space + char table + covariate OR tables."""
    t_phase = time.time()
    html_parts: list[str] = []

    # 2.1 — 9-color (response × treatment) map applied via a synthetic obs column.
    # pc.pl.archetypal_space's categorical_colors expects flat {level: color},
    # so we build a combined 'response_treatment' column and a flat cmap.
    cmap_tuple = build_response_timepoint_colormap()
    cmap_flat = {f"{r}|{t}": c for (r, t), c in cmap_tuple.items()}
    adata_train.obs["response_treatment"] = (
        adata_train.obs["response_group"].astype(str) + "|" +
        adata_train.obs["treatment"].astype(str)
    ).astype("category")

    # 2.2 — main 3D plot
    try:
        fig_main = pc.pl.archetypal_space(
            adata_train,
            color_by="response_treatment",
            cell_opacity=0.55,
            show_archetype_labels=True,
            title="Fig 3A — Global archetypal space (response × timepoint)",
            categorical_colors=cmap_flat,
        )
        html_parts.append(report.plotly_to_div(
            fig_main, "Fig 3A — main: 9-combo ramp (hue=response, lightness=timepoint)."
        ))
    except Exception as e:
        html_parts.append(error_html(f"Fig 3A main plot failed: {e}"))

    # 2.3 — per-timepoint facet panel (3 subplots)
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig_facet = make_subplots(rows=1, cols=3,
                                    specs=[[{"type": "scatter3d"}] * 3],
                                    subplot_titles=("Base", "PD1", "RTPD1"))
        archetype_pos = np.asarray(res["archetype_coords"])[:, :3]
        for col, tp in enumerate(("Base", "PD1", "RTPD1"), start=1):
            mask = (adata_train.obs["treatment"].astype(str) == tp).values
            pts = adata_train.obsm["X_pca"][mask, :3]
            resp = adata_train.obs.loc[mask, "response_group"].astype(str).values
            colors = [cmap[(r, tp)] for r in resp]
            fig_facet.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers", marker=dict(size=2.0, color=colors, opacity=0.55),
                showlegend=False,
            ), row=1, col=col)
            fig_facet.add_trace(go.Scatter3d(
                x=archetype_pos[:, 0], y=archetype_pos[:, 1], z=archetype_pos[:, 2],
                mode="markers+text",
                marker=dict(size=6, color="black", symbol="diamond"),
                text=[f"A{i}" for i in range(archetype_pos.shape[0])],
                showlegend=False,
            ), row=1, col=col)
        fig_facet.update_layout(height=500, width=1300,
                                  title="Fig 3A-ii — Per-timepoint facets")
        html_parts.append(report.plotly_to_div(
            fig_facet, "Fig 3A-ii — 3 facets (Base / PD1 / RTPD1)."
        ))
    except Exception as e:
        html_parts.append(error_html(f"Fig 3A facet panel failed: {e}"))

    # 2.4 — holdout projection visualization
    try:
        import plotly.graph_objects as go
        archetype_pos = np.asarray(res["archetype_coords"])[:, :3]
        fig_ho = go.Figure()
        fig_ho.add_trace(go.Scatter3d(
            x=adata_train.obsm["X_pca"][:, 0],
            y=adata_train.obsm["X_pca"][:, 1],
            z=adata_train.obsm["X_pca"][:, 2],
            mode="markers", marker=dict(size=1.5, color="lightgray", opacity=0.4),
            name="train",
        ))
        ho_resp = adata_holdout.obs["response_group"].astype(str).values
        ho_tx = adata_holdout.obs["treatment"].astype(str).values
        ho_colors = [cmap[(r, t)] for r, t in zip(ho_resp, ho_tx)]
        fig_ho.add_trace(go.Scatter3d(
            x=adata_holdout.obsm["X_pca"][:, 0],
            y=adata_holdout.obsm["X_pca"][:, 1],
            z=adata_holdout.obsm["X_pca"][:, 2],
            mode="markers", marker=dict(size=2.5, color=ho_colors, opacity=0.85),
            name="holdout",
        ))
        fig_ho.add_trace(go.Scatter3d(
            x=archetype_pos[:, 0], y=archetype_pos[:, 1], z=archetype_pos[:, 2],
            mode="markers+text",
            marker=dict(size=8, color="black", symbol="diamond"),
            text=[f"A{i}" for i in range(archetype_pos.shape[0])],
        ))
        fig_ho.update_layout(title="Fig 3A-iii — Holdout cells projected", height=560)
        html_parts.append(report.plotly_to_div(
            fig_ho, "Fig 3A-iii — held-out cells (colored) over train cells (grey)."
        ))
    except Exception as e:
        html_parts.append(error_html(f"Fig 3A holdout projection failed: {e}"))

    # 2.5 — characterization table
    try:
        # Top genes per archetype: quick peek via simplex regression if available
        top_genes_by_archetype = {}
        try:
            reg = pc.tl.feature_simplex_regression(adata_train, degrees=(1,))
            # reg is a serialized dict; pull per-archetype top features
            # The exact structure varies across v0.5.0 — be defensive.
            top_df = (reg.get("degree_1") or {}).get("feature_results")
            if top_df is not None:
                tdf = pd.DataFrame(top_df)
                for a in sorted(adata_train.obs["archetypes"].dropna().unique()):
                    sub = tdf[tdf["archetype"] == a].sort_values(
                        "coef", ascending=False
                    ).head(5)
                    top_genes_by_archetype[int(a)] = sub["feature"].tolist()
        except Exception as e_inner:
            html_parts.append(error_html(
                f"simplex regression for top-genes unavailable: {e_inner}. "
                "Characterization table will omit top_genes column."
            ))
            top_genes_by_archetype = None

        char_df = build_archetype_char_table(
            adata_train.obs,
            archetypes_col="archetypes",
            covariate_cols=["response_group", "treatment", "cohort", "majority_voting"],
            top_genes_by_archetype=top_genes_by_archetype,
        )
        html_parts.append(report.df_to_html(
            char_df, "Archetype characterization — quick-look table."
        ))
    except Exception as e:
        html_parts.append(error_html(f"characterization table failed: {e}"))

    # 2.6 — hypergeometric OR tables
    try:
        or_tables = build_archetype_hypergeometric_tables(
            adata_train.obs,
            archetypes_col="archetypes",
            covariate_cols=["response_group", "treatment", "majority_voting", "cohort"],
            min_level_cells=50,
        )
        for cov, df in or_tables.items():
            cap = f"Hypergeometric enrichment — {cov} × archetype (BH q-values within covariate)."
            html_parts.append(report.df_to_html(df, cap, max_rows=60))
    except Exception as e:
        html_parts.append(error_html(f"hypergeometric tables failed: {e}"))

    report.add_section("Fig 3A — Global archetype space (response × timepoint)",
                        "\n".join(html_parts), step_num=2, open_by_default=True)
    print(f"  phase2: done in {time.time() - t_phase:.1f}s")
```

- [ ] **Step 4: Verify structural tests**

```bash
conda run -n archetype python -m pytest tests/test_paper_part2_step1.py -k run_part2 -v
```
Expected: 4 passed (3 prior + the new one).

- [ ] **Step 5: Smoke-run the script (SUBSAMPLE=0.2 is fast)**

```bash
conda run -n archetype python -u scripts/run_paper_part2_tnbc.py 2>&1 | tail -40
```
Expected: runs Phase 1 + Phase 2, writes HTML; Phase 3 still stub (one section). Expect ~30-60 min at SUBSAMPLE=0.2. If it crashes in Phase 2 on a PEACH call, capture the error and iterate — do not commit a broken script. Address any crash before moving on.

- [ ] **Step 6: Commit once the smoke run produces a valid HTML**

```bash
git add scripts/run_paper_part2_tnbc.py tests/test_paper_part2_step1.py
git commit -m "Part 2: Phase 2 Fig 3A (9-combo ramp, facets, holdout, char+OR tables)"
```

---

## Task 13: Phase 3 — Fig 3B + gated Fig 3C

**Files:**
- Modify: `scripts/run_paper_part2_tnbc.py` — replace `phase3_figure3bc` stub
- Test: `tests/test_paper_part2_step1.py` (append)

- [ ] **Step 1: Append test**

```python
# ============================================================================
# Task 13 — Phase 3 (Fig 3B/3C)
# ============================================================================


def test_run_part2_phase3_calls():
    src = _read_run_source()
    assert "build_segregation_ratio" in src
    assert "build_distance_heatmaps" in src
    assert "build_diversity_block" in src
    assert "STRESS_GENES_FLAT" in src
    assert "FIG3C_GATE_THRESHOLD" in src
    assert "pc.tl.feature_simplex_regression" in src
    assert "Fig 3B" in src or "Figure 3B" in src
    assert "Fig 3C" in src or "Figure 3C" in src
```

- [ ] **Step 2: Run, expect failure**

```bash
conda run -n archetype python -m pytest tests/test_paper_part2_step1.py::test_run_part2_phase3_calls -v
```
Expected: FAIL.

- [ ] **Step 3: Replace the Phase 3 stub**

```python
def phase3_figure3bc(adata_train, res, report: HTMLReport):
    """Fig 3B (gene/pathway/stress dotplots) + gated Fig 3C (heatmaps + diversity)."""
    t_phase = time.time()

    # ------ Fig 3B ---------------------------------------------------------
    fig3b_parts: list[str] = []

    # 3B.1 — degree-1 gene simplex regression
    try:
        reg_genes = pc.tl.feature_simplex_regression(
            adata_train, degrees=(1,), fdr_threshold=FDR_THRESHOLD,
        )
        # Convert to long dataframe for dotplot. Be defensive about schema.
        gene_df = None
        d1 = (reg_genes.get("degree_1") or {})
        if "feature_results" in d1:
            gene_df = pd.DataFrame(d1["feature_results"])
        if gene_df is not None and len(gene_df):
            # Filter for archetype-exclusive (ratio >= 2.5) and FDR <= 0.05
            gene_df = gene_df[gene_df.get("q", gene_df.get("fdr", 1.0)) <= FDR_THRESHOLD]
            if "exclusive_ratio" in gene_df.columns:
                gene_df = gene_df[gene_df["exclusive_ratio"] >= EXCLUSIVE_RATIO_THRESHOLD]
            figsize = dotplot_figsize(gene_df, y_col="feature")
            fig_b1 = pc.pl.dotplot(
                adata_train, gene_df, group_col="archetype", feature_col="feature",
                size_col="coef", color_col="q", figsize=figsize,
            )
            fig3b_parts.append(report.fig_to_img(
                fig_b1, "Fig 3B-1 — archetype-exclusive genes (deg-1 simplex regression)."
            ))
        else:
            fig3b_parts.append(error_html(
                "No archetype-exclusive genes survived filters (FDR ≤ 0.05, ratio ≥ 2.5)."
            ))
    except Exception as e:
        fig3b_parts.append(error_html(f"Fig 3B gene dotplot failed: {e}"))

    # 3B.2 — pathway simplex regression (if pathway_scores present)
    if "pathway_scores" in adata_train.obsm:
        try:
            reg_pw = pc.tl.feature_simplex_regression(
                adata_train, degrees=(1,),
                feature_matrix="pathway_scores",
                fdr_threshold=FDR_THRESHOLD,
            )
            pw_df = pd.DataFrame((reg_pw.get("degree_1") or {}).get("feature_results") or [])
            if len(pw_df):
                pw_df = pw_df[pw_df.get("q", pw_df.get("fdr", 1.0)) <= FDR_THRESHOLD]
                figsize = dotplot_figsize(pw_df, y_col="feature")
                fig_b2 = pc.pl.dotplot(
                    adata_train, pw_df, group_col="archetype", feature_col="feature",
                    size_col="coef", color_col="q", figsize=figsize,
                )
                fig3b_parts.append(report.fig_to_img(
                    fig_b2, "Fig 3B-2 — archetype pathway enrichment."
                ))
        except Exception as e:
            fig3b_parts.append(error_html(f"Fig 3B pathway dotplot failed: {e}"))
    else:
        fig3b_parts.append(error_html(
            "adata.obsm['pathway_scores'] absent — Fig 3B-2 pathway dotplot skipped. "
            "Upstream prep must add this via pp.compute_pathway_scores."
        ))

    # 3B.3 — stress-gene subset
    try:
        stress_in_data = [g for g in STRESS_GENES_FLAT if g in adata_train.var_names]
        if not stress_in_data:
            raise ValueError("No stress genes found in adata.var_names.")
        adata_stress = adata_train[:, stress_in_data].copy()
        # Re-propagate archetypes col which we need for dotplot grouping
        adata_stress.obs = adata_train.obs.copy()
        adata_stress.obsm = adata_train.obsm.copy()
        adata_stress.uns = adata_train.uns.copy()
        reg_stress = pc.tl.feature_simplex_regression(
            adata_stress, degrees=(1,), fdr_threshold=FDR_THRESHOLD,
        )
        s_df = pd.DataFrame((reg_stress.get("degree_1") or {}).get("feature_results") or [])
        if len(s_df):
            figsize = dotplot_figsize(s_df, y_col="feature")
            fig_b3 = pc.pl.dotplot(
                adata_stress, s_df, group_col="archetype", feature_col="feature",
                size_col="coef", color_col="q", figsize=figsize,
            )
            fig3b_parts.append(report.fig_to_img(
                fig_b3, f"Fig 3B-3 — stress genes (n={len(stress_in_data)} overlap)."
            ))
        else:
            fig3b_parts.append(error_html(
                "No stress genes reached FDR ≤ 0.05 — negative control confirmed."
            ))
    except Exception as e:
        fig3b_parts.append(error_html(f"Fig 3B stress dotplot failed: {e}"))

    report.add_section("Fig 3B — Archetype molecular characterization",
                        "\n".join(fig3b_parts), step_num=3, open_by_default=True)

    # ------ Fig 3C ---------------------------------------------------------
    fig3c_parts: list[str] = []

    weights = adata_train.obsm["cell_archetype_weights"]

    # 3C gate computation
    try:
        seg = build_segregation_ratio(
            adata_train.obs, weights,
            response_col="response_group", treatment_col="treatment",
        )
        fig3c_parts.append(metric_grid([
            metric_card("Within (mean W2)", seg["within"], ".3f"),
            metric_card("Between (mean W2)", seg["between"], ".3f"),
            metric_card("Segregation ratio", seg["ratio"], ".3f"),
            metric_card("Gate (≥ 1.3)", "PASS" if seg["ratio"] >= FIG3C_GATE_THRESHOLD else "FAIL", "s"),
            metric_card("N within pairs", seg["n_within_pairs"], "d"),
            metric_card("N between pairs", seg["n_between_pairs"], "d"),
        ]))
        gate_passed = seg["ratio"] >= FIG3C_GATE_THRESHOLD
    except Exception as e:
        fig3c_parts.append(error_html(f"segregation ratio failed: {e}"))
        gate_passed = False

    # 3C-i — gated heatmaps
    if gate_passed:
        try:
            fig_heat, rho = build_distance_heatmaps(
                adata_train.obs, weights, adata_train.obsm["X_pca"],
                response_col="response_group", archetypes_col="archetypes",
            )
            fig3c_parts.append(report.plotly_to_div(
                fig_heat,
                f"Fig 3C-i — (response × archetype) distances: W2 vs Euclidean "
                f"centroid. Spearman ρ = {rho:.3f}."
            ))
        except Exception as e:
            fig3c_parts.append(error_html(f"Fig 3C-i heatmaps failed: {e}"))
    else:
        fig3c_parts.append(
            '<div style="padding:8px;background:#fffbeb;border-left:4px solid #ca8a04">'
            "<strong>Fig 3C-i skipped.</strong> Segregation ratio below threshold "
            f"({FIG3C_GATE_THRESHOLD}). Consider K±1, per-timepoint modeling, or "
            "batch correction on PC1 (see Phase 1 PC1 scan)."
            "</div>"
        )

    # 3C-ii — always rendered
    try:
        fig_div, summary = build_diversity_block(
            adata_train.obs, weights, adata_train.obsm["X_pca"],
            group_col="response_group", bootstrap_n=200, subsample=500,
            random_state=42,
        )
        fig3c_parts.append(report.plotly_to_div(
            fig_div,
            f"Fig 3C-ii — Diversity block. "
            f"KW H={summary['per_cell_shannon_kw_stat']:.2f}, "
            f"p={summary['per_cell_shannon_kw_p']:.2e}. "
            f"Per-group PCA dispersion (pre-registered test for R2 &lt; NR): "
            f"{summary['per_group_pca_dispersion']}."
        ))
    except Exception as e:
        fig3c_parts.append(error_html(f"Fig 3C-ii diversity block failed: {e}"))

    report.add_section("Fig 3C — Segregation distances + diversity",
                        "\n".join(fig3c_parts), step_num=4, open_by_default=True)
    print(f"  phase3: done in {time.time() - t_phase:.1f}s")
```

- [ ] **Step 4: Verify structural test passes**

```bash
conda run -n archetype python -m pytest tests/test_paper_part2_step1.py -k run_part2 -v
```
Expected: 5 passed.

- [ ] **Step 5: Run the full script end-to-end**

```bash
conda run -n archetype python -u scripts/run_paper_part2_tnbc.py 2>&1 \
  | tee "outputs/paper_part2/run_log_$(date +%Y%m%d)_r1.txt"
```
Expected: Phase 1 → Phase 2 → Phase 3 → writes `outputs/paper_part2/part2_report_YYYYMMDD_r1.html`. Inspect the HTML for each section. Expect ~2h at SUBSAMPLE=0.2.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_paper_part2_tnbc.py tests/test_paper_part2_step1.py
git commit -m "Part 2: Phase 3 Fig 3B/3C (genes+pathways+stress, gated heatmaps, diversity)"
```

---

## Task 14: Final test sweep + clean run log

**Files:**
- No code changes

- [ ] **Step 1: Run the full test file**

```bash
conda run -n archetype python -m pytest tests/test_paper_part2_step1.py -v
```
Expected: all tests pass.

- [ ] **Step 2: Run the full repo test suite regression check**

```bash
conda run -n archetype python -m pytest tests/test_paper_part1_fixes.py \
  tests/test_paper_part2_step1.py \
  tests/test_core/test_archetype_correspondence.py -q
```
Expected: no regressions in Part 1 or core correspondence tests. (We added helpers to `_paper_part1_viz.py` — make sure the Part 1 structural regexes still match.)

- [ ] **Step 3: Triage the r1 report HTML**

Open `outputs/paper_part2/part2_report_YYYYMMDD_r1.html` in a browser. Check:
- Prototype banner text is present.
- Phase 1 metric grid has sane values (K between 3-10; train R² > 0.5; holdout R² within 0.05 of train).
- PC1 correlation scan — if `cohort` or `batch` is the top hit, log a feedback item for r2.
- Fig 3A main plot renders; 9-combo colors visible; archetype diamonds present.
- Fig 3A facet panel: all 3 timepoint subplots render.
- Fig 3A holdout projection visible.
- Characterization table and OR tables populated.
- Fig 3B: at least the gene dotplot renders; pathway + stress may degrade gracefully.
- Fig 3C: segregation_ratio value + gate status visible; either heatmaps or skip-box; diversity block renders.

- [ ] **Step 4: Write `r1-feedback.md` in `docs/plans/`** (optional, for human review handoff)

Capture anything that surprised you: PC1 correlations, gate pass/fail, visual issues, convergence warnings. Not required — just a good habit carried forward from Part 1's iteration loop.

- [ ] **Step 5: Commit anything final + tag the rev**

```bash
git add -A
git commit -m "Part 2: r1 end-to-end run complete" --allow-empty
```

---

## Self-review — spec coverage map

| Spec requirement | Task |
|------------------|------|
| §4.1 prep recipe (MAD → PCA → split) | Task 10 |
| §4.2 `safe_stratified_split` helper | Task 1 |
| §4.3 prep report HTML | Task 10 |
| §4.4 explicit non-steps (no scale, no HVG, no normalize) | Task 10 structural tests |
| §5.1 config constants | Task 11 |
| §5.2 Phase 1 training + drift QC + PC1 scan + holdout R² | Task 11 |
| §5.3.1 colormap | Task 2 |
| §5.3.2 3D layout (main + facet + holdout) | Task 12 |
| §5.3.3 characterization table | Task 5, used in Task 12 |
| §5.3.4 hypergeometric OR tables | Task 6, used in Task 12 |
| §5.4.1 Fig 3B genes + pathways + stress | Task 13 |
| §5.4.2 segregation gate | Task 4, used in Task 13 |
| §5.4.2 W2 + Euclidean heatmaps | Task 3 (W2), Task 8 (heatmaps), Task 13 |
| §5.4.3 diversity block (3 panels) | Task 9, used in Task 13 |
| §6 reused + new helpers | Tasks 2, 3, 4, 5, 6, 7, 8, 9 |
| §7.1 computation tests | Tasks 1–9 (each has TDD cycle) |
| §7.2 structural tests | Tasks 10, 11, 12, 13 |
| §8 run commands + rev numbering | Task 11 (REPORT_PATH) + Task 14 |
| §9 open risks flagged in report | Task 11 (PC1 scan), Task 13 (gate fallback) |

All spec sections have a task. No gaps found.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-14-part2-step1-tnbc.md`.**

Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?

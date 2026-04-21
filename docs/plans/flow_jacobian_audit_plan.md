# Flow Jacobian Audit Plan

**Date**: 2026-04-20
**Branch**: feature/v050-continuous-characterization
**Files**: `src/peach/_core/utils/flow_matching.py`, `src/peach/tl/flow.py`, `tests/test_statistical/test_flow_jacobian.py`

---

## Jacobian Type Taxonomy (Reference)

| Type | Notation | Method | Measures |
|---|---|---|---|
| **Velocity Jacobian** | `dv/dx` | `FlowModel.jacobian(x, t)` | Instantaneous local structure of velocity field at a point. `trace = divergence`. |
| **Flow Map Jacobian** | `∂φ_t/∂x₀` | `FlowModel.flow_map_jacobian(x0, t_eval)` | Accumulated deformation of cell neighborhoods over transport [0, t]. |

**Scientific priority**: Flow map Jacobian is the primary tool for understanding mass transit between archetype pairs and gene expansion/contraction.

---

## Decisions Log

### Item 4 — Velocity vs. Flow Map Jacobian (Architectural)

**`flow_feature_graph`** → **Deprecate.** Route users to `flow_jacobian`, which already correctly computes flow map Jacobian + feature expansion.

**`flow_temporal_feature_graph`** → **Deprecate**, mark for future development. Multi-timepoint `flow_jacobian(t=[...])` covers the primary use case.

**`flow_bifurcation`** → **Keep** for divergence/eigenvalue analysis. Remove saddle point detection (see Item 1).

### Item 1 — `flow_bifurcation` Saddle Point Detection

**Decision**: Remove `n_saddle_points` from output entirely.

**Reason**: Mixed-sign eigenvalues of `dv/dx` at transported cell positions (non-fixed-points) is nearly universal and biologically meaningless. Saddle detection requires locating fixed points first. Keep `divergence`, `eigenvalue_real`, `eigenvalue_imag`, `bifurcation_score`, `timepoints`.

**Also fix**: Frame alignment in `flow_bifurcation` — `timepoints = np.linspace(0.05, 0.95, n)` doesn't match `traj_times = np.linspace(0, 1, n)`, causing boundary frames to evaluate Jacobian at wrong positions. Fix by using the same linspace for both or reintegrating to exact times.

### Item 2 — FDR Pre-Filtering (Statistical Correctness)

**Decision**: BH correction over **all genes** (full family). Remove top-5% pre-filter from both `flow_gene_alignment` and `flow_jacobian`.

**Reason**: Pre-selecting top 5% by the same data used to compute p-values introduces selection bias — q-values are understated. BH over the full family is conservative but correct and defensible in a paper.

**Affects**: `flow.py:443-450` and `flow.py:728-733`.

### Item 3 — Degenerate Rank-Based Null

**Decision**: Gate rank-based permutation null behind `if run_rotation`. When `null_type='shuffle'`, emit a `UserWarning` explaining that the rank null is skipped (shuffle null is degenerate for rank-based testing, producing all p=1.0) and store nothing.

**Affects**: `flow.py:483-503`.

### Item 5 — `flow_between` Memory + Seed Issues

**Decision**: Fix both.

- Replace `a.copy()` with a lightweight approach (add label column to a copy of just `obs`, or use `copy_X=False`) to avoid duplicating expression matrices.
- Replace `random_state + pair_idx` with `hash((random_state, pair_idx)) % (2**31)` so pair seeds are independent and don't shift together when `random_state` changes.

**Affects**: `flow.py:190-196`, `flow.py:221`.

### Item 6 — `normalize` Flag Inconsistency

**Decision**: Always use cosine similarity for both aggregate and per-cell alignment. Deprecate `normalize=False`.

**Implementation**:
- Remove the `normalize` parameter branch for per-cell (it was already always cosine).
- Make aggregate also always cosine; deprecate `normalize=False` with a `DeprecationWarning`.
- Document clearly: both aggregate and per-cell use cosine similarity.

**Affects**: `flow.py:337-344`, `flow.py:373-380`, docstring.

### Item 7 — Archetype Hub Gene Fallback

**Not moot — deprecation must gut the function body.**

`flow_feature_graph` will emit a `DeprecationWarning` and return early. The body — including the fallback at `flow.py:1171-1174` that assigns `top_hub_genes[:10]` identically to every archetype — must be removed or replaced with `raise NotImplementedError` after the warning. A deprecated function that still silently returns wrong output (same gene list for every archetype) is worse than a hard error.

```python
# flow_feature_graph implementation:
def flow_feature_graph(...):
    import warnings
    warnings.warn(
        "flow_feature_graph is deprecated. Use flow_jacobian with per_cell_features=True "
        "for gene expansion analysis. flow_feature_graph will be removed in v0.6.",
        DeprecationWarning,
        stacklevel=2,
    )
    raise NotImplementedError(
        "flow_feature_graph has been deprecated. See flow_jacobian."
    )
```

**Future design note**: If archetype-specific hub genes are ever needed, compute them from
per-archetype mean Jacobians (subset source cells by dominant archetype weight, compute
`flow_jacobian` per subset) — NOT by copying a global top-10 list to every archetype.

### Item 8 — ODE Solver Mismatch (`midpoint` vs `dopri5`)

**Decision**: Document the mismatch; increase default `n_steps` to 100; fix the rigged test.

**Details**:
- `flow_map_jacobian` must use `torchdiffeq midpoint` (adaptive solvers cause autograd nesting errors with `jacrev` inside RHS). This is an inherent constraint.
- Add to `flow_map_jacobian` docstring: "Uses torchdiffeq midpoint (fixed-step RK2). This differs from the dopri5 solver used by `transport()`. At n_steps=100 the discretization error in phi at t=1 is small but not zero relative to `flow_result['transported']`."
- Change default `n_steps` from 50 → 100 in `flow_map_jacobian`.
- Fix `test_flow_map_jacobian_phi_matches_transport`: add dopri5 variant that checks phi within an honest tolerance (e.g., `atol=0.05`) documenting expected solver discrepancy.

### Item 9 — Gene Pre-Filter in `flow_feature_graph`

**Moot** — deprecated.

### Item 10 — Liouville Equation Consistency Test

**Decision**: Add test.

**Test logic**: For a trained model, verify that `integrate(trace(dv/dx(φ(t), t)) dt, 0→T) ≈ log|det J(T)|`. Uses `flow_model.jacobian()` for the integrand and `flow_map_jacobian()` for the left-hand side. This cross-validates the two Jacobian computation paths.

### Item 11 — MMD Bandwidth Scaling

**Decision**: Divide median-heuristic bandwidth by `sqrt(n_dims)`.

**Implementation**: In `compute_mmd`, after computing `bandwidth = np.median(...)`, add `bandwidth /= np.sqrt(XY.shape[1])`. Better-calibrated kernel for 10+ PCA dimensions.

**Affects**: `flow_matching.py:528-535`.

### Item 12 — PCA Visibility Docstring Warning

**Decision**: Add to docstrings of `flow_gene_alignment` and `flow_jacobian`.

**Text**: "Note: Gene scores are mediated through PCA loadings. Genes with low variance explained by the top PCA components will have near-zero scores regardless of their biological relevance to the flow. Consider this when interpreting low-scoring genes."

### Item T5 — Source Mask Alignment (Production Bug Guard)

**Decision**: Store `source_obs_names` in `flow_result` at `flow_within` time. Validate in `flow_jacobian` and `flow_gene_alignment`.

**Implementation**:
```python
# In flow_within, after building source_mask:
result["source_obs_names"] = adata.obs_names[source_mask].tolist()

# In flow_jacobian / flow_gene_alignment, add:
if "source_obs_names" in flow_result:
    expected = flow_result["source_obs_names"]
    actual = adata.obs_names[flow_result["source_mask"]].tolist()
    if expected != actual:
        raise ValueError(
            "adata.obs_names do not match the AnnData used in flow_within. "
            "Pass the same adata object used to generate flow_result."
        )
```

### Item S1 — `_build_mask` Non-Scalar Filter Values

**Decision**: Raise `ValueError` on non-scalar `val` immediately.

```python
if not np.isscalar(val):
    raise ValueError(
        f"Filter value for '{col}' must be scalar; got {type(val).__name__}. "
        "For multi-value filtering, call flow_within separately per condition."
    )
```

**Affects**: `flow.py:980-983`.

---

## Test Fixes

### T1 — Finite-Difference Jacobian Tolerance Too Loose

Increase training epochs to 200, tighten `rtol` from 0.1 → 0.01. Smoother velocity field after more training makes the FD comparison meaningful.

**Affects**: `test_flow_jacobian.py:81`.

### T2 — Rigged Transport Consistency Test

Add a separate test using the default dopri5 solver that documents the expected solver discrepancy between `flow_map_jacobian` and `transport()`. Use `atol=0.1` and a comment explaining the midpoint vs. dopri5 gap.

**Affects**: `test_flow_jacobian.py:563`.

### T3 — J-at-t0 Tolerance Too Loose

Increase `n_steps` in test to 200, tighten `atol` from 0.3 → 0.05. J at t=0.02 with 200 midpoint steps should be very close to identity.

**Affects**: `test_flow_jacobian.py:536`.

---

## Minor Code Quality (Include in Same Pass)

- **S3**: Remove `vel_norm_agg = None` initialization; scope it inside the `if normalize:` block (now moot after deprecating `normalize=False`, but still clean up the dead init).
- **Item 13**: Vectorize shuffle null permutation loops in `flow_gene_alignment` and `flow_jacobian` (batch permutation indices, single matmul) for performance at large `n_genes` × `n_permutations`.

---

## Not Addressed Now (Future Work)

- `flow.py` structural refactor (S4): 1367-line junk drawer. Split into `flow_core.py`, `flow_jacobian.py`, `flow_graphs.py`. Deferred.
- `flow_temporal_feature_graph`: deprecate now, redesign later with flow map Jacobian if temporal gene coupling is needed.
- Gene set / pathway flow association as first-class API function (currently Spearman correlation in scripts). Candidate for v0.6.

---

## Implementation Order

1. **Deprecations** (`flow_feature_graph`, `flow_temporal_feature_graph`) — emit `DeprecationWarning` then `raise NotImplementedError`; function bodies fully removed so broken fallback logic cannot run
2. **Source mask guard** (T5/S1) — prevents production bugs, low risk
3. **`flow_bifurcation` saddle removal + frame fix** — removes wrong output
4. **FDR + rank null fix** (Items 2+3) — statistical correctness, critical for paper
5. **`normalize` consolidation** (Item 6) — cleanup + deprecation warning
6. **`flow_map_jacobian` n_steps + docstring** (Item 8) — low risk
7. **`flow_between` memory + seed** (Item 5) — low risk
8. **`_build_mask` scalar guard** (S1) — low risk
9. **MMD bandwidth scaling** (Item 11) — one line
10. **Docstring additions** (Item 12) — documentation only
11. **Test fixes** (T1, T2, T3) + Liouville test (Item 10)
12. **Vectorize permutation loops** (Item 13)

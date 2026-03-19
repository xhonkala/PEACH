# v0.5.0 Redteam Fixes: Statistical & Implementation Hardening

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 23 issues identified by Reviewer 2 / Senior / Gremlin redteam audit of v0.5.0 features — silent wrong results, incorrect statistics, numerical hazards, performance, and test gaps.

**Architecture:** Fixes are grouped into 4 implementation chunks (flow, regression/comparison, statistics/cleanup, adversarial tests). Each chunk is independently testable and committable. Tasks within each chunk must be executed **sequentially** (multiple tasks modify the same files, e.g., `flow.py` is touched by Tasks 1.1, 1.4, 1.5, 1.7, 2.3, 3.6). Chunk 4 (tests) depends on all prior chunks being complete.

**Tech Stack:** Python 3.10+, numpy, scipy, torch, statsmodels, anndata, pytest

**Conda env:** `archetype` (activate before all commands)

---

## File Map

### Files to Modify

| File | Fixes | Summary |
|------|-------|---------|
| `src/peach/tl/flow.py` | 1,6,9,11,15,17,22 | t param, significance, centrality, vectorize, logging |
| `src/peach/_core/utils/flow_matching.py` | 9,17 | OT-CFM seed, MMD degenerate guard |
| `src/peach/_core/utils/simplex_regression.py` | 13,14 | Sparse dedup, HC3 vectorize |
| `src/peach/_core/utils/archetype_comparison.py` | 2,8,10,16 | Feature tracking, symmetric perm, cross-K, index dict |
| `src/peach/_core/utils/pattern_classification.py` | 19 | SE-aware classification |
| `src/peach/_core/utils/ilr_transform.py` | 5 | Clip clr before exp |
| `src/peach/_core/utils/statistical_tests.py` | 12 | Remove global RNG mutation |
| `src/peach/_core/utils/dirichlet_mixture.py` | 18,23 | Vertex smoothing, dedup LL |
| `src/peach/tl/feature_regression.py` | 3,7 | Explicit feature_matrix, normalize loadings |
| `src/peach/tl/comparison.py` | 4 | Docstring warning for model/result matching |

### Files to Create (Tests)

| File | Tests |
|------|-------|
| `tests/test_core/test_dirichlet_mixture.py` | Dirichlet convergence, recovery, vertex data |
| `tests/test_statistical/test_redteam_adversarial.py` | Degenerate inputs, K=2, silent corruption, type-I error |

---

## Chunk 0: Shared Test Helpers

Before starting any chunk, add these helpers to the relevant test files. They are referenced by multiple tasks.

### Task 0.1: Add flow test helpers

**Files:**
- Modify: `tests/test_statistical/test_flow_jacobian.py` (add at top, after imports)

- [ ] **Step 1: Add helper functions**

```python
def _make_flow_fixture(n_cells=100, n_genes=50, K=3, return_model=True, seed=42):
    """Create minimal AnnData + trained flow for testing."""
    import anndata as ad
    from peach._core.utils.flow_matching import FlowModel
    import torch

    rng = np.random.default_rng(seed)
    X = rng.randn(n_cells, n_genes).astype(np.float32)
    adata = ad.AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obsm["X_pca"] = rng.randn(n_cells, 10).astype(np.float32)
    adata.obsm["cell_archetype_weights"] = rng.dirichlet(np.ones(K), n_cells).astype(np.float32)
    adata.varm["PCs"] = rng.randn(n_genes, 10).astype(np.float32)
    adata.obs["condition"] = ["source"] * (n_cells // 2) + ["target"] * (n_cells - n_cells // 2)

    import peach as pc
    flow_result = pc.tl.flow_within(
        adata,
        source={"condition": "source"},
        target={"condition": "target"},
        n_epochs=50,
        hidden_dims=(32, 32),
        return_model=return_model,
        random_state=seed,
    )
    return adata, flow_result


def _make_feature_graph_result(seed=42):
    """Create a flow feature graph result for testing."""
    import peach as pc
    adata, flow_result = _make_flow_fixture(return_model=True, seed=seed)
    model = flow_result["model"]
    result = pc.tl.flow_feature_graph(
        adata, flow_result, model,
        n_top_genes=20, n_timepoints=5, n_eval_points=30,
        random_state=seed,
    )
    return result
```

- [ ] **Step 2: Add comparison test helpers**

Add to `tests/test_statistical/test_archetype_comparison.py` (after imports):

```python
def _make_adata_with_weights(rng=None, n_cells=200, K=3, n_genes=50):
    """Create AnnData with weights and PCA for comparison tests."""
    import anndata as ad
    if rng is None:
        rng = np.random.default_rng(42)
    X = rng.randn(n_cells, n_genes).astype(np.float32)
    adata = ad.AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obsm["X_pca"] = rng.randn(n_cells, 10).astype(np.float32)
    adata.obsm["cell_archetype_weights"] = rng.dirichlet(np.ones(K), n_cells)
    return adata
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_statistical/test_flow_jacobian.py tests/test_statistical/test_archetype_comparison.py
git commit -m "Add shared test helpers for flow and comparison tests"
```

---

## Chunk 1: Flow Module Fixes

### Task 1.1: Make `t` parameter functional in `flow_gene_alignment`

**Files:**
- Modify: `src/peach/tl/flow.py:227-319`
- Test: `tests/test_statistical/test_flow_jacobian.py`

The `t` parameter is accepted but ignored — the function always computes full-trajectory displacement. Fix: when `t` is provided AND `flow_result` contains `"model"`, use `model.velocity_at(source_pca, t)` to compute instantaneous velocity. When no model is available, fall back to displacement with a warning.

**Signature change:** `t: float = 0.5` → `t: float | None = None`. This is a **behavior change**: previously `t=0.5` was stored but ignored (displacement always). After the fix, passing `t=0.5` with a model present switches to instantaneous velocity mode. `t=None` (new default) preserves the old displacement behavior.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_statistical/test_flow_jacobian.py`:

```python
def test_gene_alignment_t_parameter_changes_output():
    """Verify that different t values produce different alignment scores
    when a model is available."""
    import peach as pc
    from peach._core.utils.flow_matching import FlowModel

    # Use the shared synthetic flow fixture
    adata, flow_result = _make_flow_fixture(return_model=True)

    result_t01 = pc.tl.flow_gene_alignment(adata, flow_result, t=0.1)
    result_t09 = pc.tl.flow_gene_alignment(adata, flow_result, t=0.9)

    # Different t values MUST produce different alignment scores
    assert not np.allclose(
        result_t01["alignment_scores"],
        result_t09["alignment_scores"],
        atol=1e-6,
    ), "t parameter had no effect on alignment scores"


def test_gene_alignment_no_model_uses_displacement():
    """Without a model in flow_result, t is ignored and displacement is used."""
    import peach as pc

    adata, flow_result = _make_flow_fixture(return_model=False)

    result_default = pc.tl.flow_gene_alignment(adata, flow_result)
    result_with_t = pc.tl.flow_gene_alignment(adata, flow_result, t=0.3)

    # Without model, t has no effect (both use displacement)
    np.testing.assert_array_equal(
        result_default["alignment_scores"],
        result_with_t["alignment_scores"],
    )
    assert result_default.get("velocity_mode") == "displacement"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n archetype pytest tests/test_statistical/test_flow_jacobian.py::test_gene_alignment_t_parameter_changes_output -v`
Expected: FAIL (currently t is ignored, so scores are identical)

- [ ] **Step 3: Implement the fix**

In `src/peach/tl/flow.py`, modify `flow_gene_alignment`:

```python
def flow_gene_alignment(
    adata: AnnData,
    flow_result: dict,
    *,
    t: float | None = None,  # Changed: None means displacement, float means velocity at t
    n_top: int = 50,
    pca_loadings_key: str | None = None,
    n_permutations: int = 0,
    per_cell: bool = False,
    random_state: int = 42,
) -> dict:
    # ... (keep existing loadings setup through line 270) ...

    # Compute velocity vector(s)
    source_pca = adata.obsm[flow_result["pca_key"]][flow_result["source_mask"]]
    model = flow_result.get("model")

    if t is not None and model is not None:
        # Instantaneous velocity at time t via the trained model
        mean_velocity = model.velocity_at(source_pca, t).mean(axis=0)
        velocity_mode = "instantaneous"
    else:
        if t is not None and model is None:
            import warnings
            warnings.warn(
                f"t={t} specified but flow_result has no model (call flow_within "
                f"with return_model=True). Falling back to full-trajectory displacement.",
                UserWarning,
            )
        # Full trajectory displacement (original behavior)
        mean_velocity = (flow_result["transported"] - source_pca).mean(axis=0)
        velocity_mode = "displacement"

    # ... (rest of function unchanged, but add velocity_mode to result) ...

    result = {
        "alignment_scores": alignment_scores,
        "gene_names": gene_names,
        "top_aligned": top_aligned,
        "top_opposed": top_opposed,
        "t": t,
        "velocity_mode": velocity_mode,
    }
    # ... (per_cell and permutation blocks unchanged) ...
```

Also update the per_cell block to use instantaneous velocity when available:

```python
    if per_cell:
        if t is not None and model is not None:
            velocity_per_cell = model.velocity_at(source_pca, t)
        else:
            velocity_per_cell = flow_result["transported"] - source_pca
        # ... rest unchanged ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n archetype pytest tests/test_statistical/test_flow_jacobian.py::test_gene_alignment_t_parameter_changes_output tests/test_statistical/test_flow_jacobian.py::test_gene_alignment_no_model_uses_displacement -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/peach/tl/flow.py tests/test_statistical/test_flow_jacobian.py
git commit -m "Fix flow_gene_alignment: use instantaneous velocity when t is specified"
```

---

### Task 1.2: Fix `flow_significance` to use original model's MMD

**Files:**
- Modify: `src/peach/tl/flow.py:393-476`
- Test: `tests/test_integration/test_flow_api.py`

Remove the retraining step. The observed improvement should come directly from the already-computed `flow_result["mmd_before"]` and `flow_result["mmd_after"]`. Only the null models are retrained.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_integration/test_flow_api.py`:

```python
def test_flow_significance_uses_original_mmd():
    """flow_significance should use the original flow_result's MMD,
    not retrain a new model."""
    import peach as pc

    adata, flow_result = _make_flow_fixture(return_model=False)

    sig = pc.tl.flow_significance(
        adata, flow_result, n_permutations=5, n_epochs_per_perm=50
    )

    # The observed stat should match flow_result's actual improvement
    expected_improvement = flow_result["mmd_before"] - flow_result["mmd_after"]
    assert abs(sig["observed_stat"] - expected_improvement) < 1e-10, (
        f"Expected observed_stat={expected_improvement}, got {sig['observed_stat']}. "
        "flow_significance should use the original model's MMD, not retrain."
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n archetype pytest tests/test_integration/test_flow_api.py::test_flow_significance_uses_original_mmd -v`
Expected: FAIL (current code retrains, gets different MMD)

- [ ] **Step 3: Implement the fix**

Replace lines 461-467 of `flow_significance` with:

```python
    # Use the original flow result's improvement (no retraining)
    if flow_result is not None and "mmd_before" in flow_result and "mmd_after" in flow_result:
        observed_improvement = flow_result["mmd_before"] - flow_result["mmd_after"]
    else:
        # No pre-computed flow: require flow_result with MMD values
        raise ValueError(
            "flow_significance requires a flow_result dict with 'mmd_before' and "
            "'mmd_after' keys (from flow_within). Pass the flow_result directly "
            "instead of source/target dicts."
        )
```

Remove the `obs_model` training block entirely (lines 462-467) and the `source/target` dict path (lines 425-428). The function should now **require** `flow_result` as its sole input (not optional). Update the signature:

```python
def flow_significance(
    adata: AnnData,
    flow_result: dict,        # CHANGED: no longer optional
    *,
    n_permutations: int = 100,
    n_epochs_per_perm: int = 200,
    # ... (remove source/target params) ...
```

This simplifies the API: train your flow first, then test its significance. The null models are still retrained with `n_epochs_per_perm`.

- [ ] **Step 4: Run tests**

Run: `conda run -n archetype pytest tests/test_integration/test_flow_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/peach/tl/flow.py tests/test_integration/test_flow_api.py
git commit -m "Fix flow_significance: use original model's MMD instead of retraining"
```

---

### Task 1.3: Seed OT-CFM `np.random.choice`

**Files:**
- Modify: `src/peach/_core/utils/flow_matching.py:140,216-218`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_core/test_flow_matching.py`:

```python
def test_ot_cfm_reproducible():
    """OT-CFM training should be reproducible with same random_state."""
    pytest.importorskip("ot")
    source = np.random.randn(50, 5).astype(np.float32)
    target = np.random.randn(50, 5).astype(np.float32) + 2

    torch.manual_seed(42)
    np.random.seed(99)  # pollute global state
    m1 = FlowModel(5, hidden_dims=(32, 32), lr=1e-3)
    losses1 = m1.train(source, target, n_epochs=20, batch_size=32, use_ot=True, random_state=42)

    torch.manual_seed(42)
    np.random.seed(77)  # different global state
    m2 = FlowModel(5, hidden_dims=(32, 32), lr=1e-3)
    losses2 = m2.train(source, target, n_epochs=20, batch_size=32, use_ot=True, random_state=42)

    np.testing.assert_allclose(losses1, losses2, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n archetype pytest tests/test_core/test_flow_matching.py::test_ot_cfm_reproducible -v`
Expected: FAIL (global np.random state differs)

- [ ] **Step 3: Implement the fix**

In `FlowModel.train`, add `random_state` parameter and seed both torch and numpy:

```python
def train(self, source, target, n_epochs=1000, batch_size=256, use_ot=False, random_state=None):
    # ... existing setup ...

    # Seed both RNGs for full reproducibility
    if random_state is not None:
        torch.manual_seed(random_state)
    ot_rng = np.random.default_rng(random_state)

    for epoch in range(n_epochs):
        # ... existing sampling (torch.randint is now seeded via torch.manual_seed) ...

        if use_ot:
            # ... existing Sinkhorn computation ...
            coupling_flat = coupling.ravel()
            coupling_flat /= coupling_flat.sum()
            pair_idx = ot_rng.choice(            # CHANGED: ot_rng instead of np.random
                len(x0) * len(x1), size=len(x0), p=coupling_flat
            )
            # ... rest unchanged ...
```

Update `flow_within` to pass `random_state` through:

```python
losses = model.train(source_train, target_pca, n_epochs=n_epochs,
                     batch_size=batch_size, use_ot=use_ot, random_state=random_state)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n archetype pytest tests/test_core/test_flow_matching.py::test_ot_cfm_reproducible -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/peach/_core/utils/flow_matching.py src/peach/tl/flow.py tests/test_core/test_flow_matching.py
git commit -m "Fix OT-CFM reproducibility: use seeded RNG instead of np.random.choice"
```

---

### Task 1.4: Fix centrality from sparsified adjacency matrix

**Files:**
- Modify: `src/peach/tl/flow.py:720-732`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_statistical/test_flow_jacobian.py`:

```python
def test_centrality_matches_sparse_adjacency():
    """out_centrality and in_centrality must be computed from the sparsified
    adjacency_matrix, not the dense unthresholded matrix."""
    # This test verifies internal consistency
    result = _make_feature_graph_result()  # helper that calls flow_feature_graph

    adj = np.abs(result["adjacency_matrix"])
    expected_out = adj.sum(axis=1)
    expected_in = adj.sum(axis=0)

    np.testing.assert_array_almost_equal(result["out_centrality"], expected_out)
    np.testing.assert_array_almost_equal(result["in_centrality"], expected_in)
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL (centrality currently from dense abs_G, not G_sparse)

- [ ] **Step 3: Implement the fix**

In `flow_feature_graph`, move centrality computation AFTER sparsification (line 727+):

```python
    G_sparse = np.where(abs_G >= threshold, G_total, 0.0)

    # --- Centrality measures (from sparsified graph) ---
    abs_G_sparse = np.abs(G_sparse)
    out_centrality = abs_G_sparse.sum(axis=1)
    in_centrality = abs_G_sparse.sum(axis=0)
    flow_centrality = out_centrality * in_centrality
```

- [ ] **Step 4: Run test, verify pass**
- [ ] **Step 5: Commit**

```bash
git add src/peach/tl/flow.py tests/test_statistical/test_flow_jacobian.py
git commit -m "Fix centrality: compute from sparsified adjacency, not dense matrix"
```

---

### Task 1.5: Vectorize bifurcation eigendecomposition

**Files:**
- Modify: `src/peach/tl/flow.py:550-568`

- [ ] **Step 1: No new test needed** — existing `test_flow_bifurcation_basic` covers correctness; this is a pure performance refactor.

- [ ] **Step 2: Implement the vectorization**

Replace the Python loops at lines 550-568:

```python
    for ti, t_val in enumerate(timepoints):
        frame_idx = np.argmin(np.abs(traj_times - t_val))
        positions = trajectory[frame_idx]

        jac = flow_model.jacobian(positions, float(t_val))  # [n_cells, dim, dim]

        # Vectorized trace (no Python loop)
        divergence[ti] = np.trace(jac, axis1=1, axis2=2)

        # Vectorized eigenvalues (numpy batches over first dimension)
        eigvals = np.linalg.eigvals(jac)  # [n_cells, dim]
        eigenvalue_real[ti] = eigvals.real
        eigenvalue_imag[ti] = eigvals.imag

    bifurcation_score = np.max(np.abs(divergence), axis=0)

    # Vectorized saddle point detection
    has_positive = np.any(eigenvalue_real > 0, axis=2)  # [n_timepoints, n_cells]
    has_negative = np.any(eigenvalue_real < 0, axis=2)
    n_saddle_points = np.sum(has_positive & has_negative, axis=0)  # [n_cells]
```

- [ ] **Step 3: Run existing tests to verify no regression**

Run: `conda run -n archetype pytest tests/test_statistical/test_flow_jacobian.py -v`
Expected: PASS (same results, faster)

- [ ] **Step 4: Commit**

```bash
git add src/peach/tl/flow.py
git commit -m "Vectorize bifurcation: replace Python loops with batched numpy ops"
```

---

### Task 1.6: Guard `compute_mmd` against degenerate inputs

**Files:**
- Modify: `src/peach/_core/utils/flow_matching.py:398-400`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_core/test_flow_matching.py`:

```python
def test_mmd_single_cell_returns_nan():
    """compute_mmd with < 2 points should return NaN, not 0."""
    from peach._core.utils.flow_matching import compute_mmd
    result = compute_mmd(np.zeros((1, 5)), np.ones((100, 5)))
    assert np.isnan(result), f"Expected NaN for single-cell input, got {result}"
```

- [ ] **Step 2: Run test to verify it fails** (currently returns 0.0)
- [ ] **Step 3: Fix**

```python
    if len(X) < 2 or len(Y) < 2:
        return float("nan")
```

- [ ] **Step 4: Run tests, check no downstream breakage**

Run: `conda run -n archetype pytest tests/test_core/test_flow_matching.py tests/test_integration/test_flow_api.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/peach/_core/utils/flow_matching.py tests/test_core/test_flow_matching.py
git commit -m "Fix compute_mmd: return NaN for degenerate inputs instead of 0"
```

---

### Task 1.7: Replace `print()` with `logging` in flow modules

**Files:**
- Modify: `src/peach/tl/flow.py` (10 print calls: lines 701, 714, 738, 885, 905, 919, 954-958)
- Modify: `src/peach/pl/flow.py` (2 print calls: lines 585, 828)

- [ ] **Step 1: No new test needed**

- [ ] **Step 2: Add logger at top of each file**

```python
import logging
logger = logging.getLogger(__name__)
```

Replace all `print(...)` with `logger.info(...)`. Example:

```python
# Before
print(f"Computing Jacobians at {n_timepoints} timepoints ...")
# After
logger.info("Computing Jacobians at %d timepoints (%d eval points, %d genes)",
            n_timepoints, n_eval, n_top)
```

- [ ] **Step 3: Run existing tests**

Run: `conda run -n archetype pytest tests/test_statistical/test_flow_jacobian.py tests/test_visualization/test_flow_viz.py -v`

- [ ] **Step 4: Commit**

```bash
git add src/peach/tl/flow.py src/peach/pl/flow.py
git commit -m "Replace print() with logging in flow modules"
```

---

## Chunk 2: Regression, Comparison & Classification Fixes

### Task 2.1: Track feature matrix source in Wald contrasts

**Files:**
- Modify: `src/peach/_core/utils/archetype_comparison.py:292-334`
- Modify: `src/peach/tl/feature_regression.py` (store `feature_source` in regression result)

The bug: `compute_wald_contrasts` falls back to re-running regression with `resolve_features(adata, None, feat_names)`, which always returns `adata.X` even when the original regression used pathway scores. Fix: store the feature matrix key in the regression result and use it in the fallback.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_statistical/test_archetype_comparison.py`:

```python
def test_wald_contrasts_respects_feature_source():
    """Wald contrasts must use the same feature matrix as the original regression."""
    # Run regression on a subset of genes (not full adata.X)
    adata = _make_adata_with_weights()  # helper
    # Store a custom feature matrix
    n_cells = adata.n_obs
    custom_features = np.random.randn(n_cells, 10)
    adata.obsm["test_features"] = custom_features

    import peach as pc
    pc.tl.feature_simplex_regression(adata, feature_matrix="test_features")

    # Wald contrasts should use the same feature matrix
    result = pc.tl.archetype_contrasts(adata)
    assert result["n_features"] == 10, (
        f"Expected 10 features (from test_features), got {result['n_features']}"
    )
```

- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement the fix**

In `feature_simplex_regression` (feature_regression.py), add `feature_source` to the stored result:

```python
result["feature_source"] = feature_matrix  # str key or None for adata.X
```

In `compute_wald_contrasts` (archetype_comparison.py), use the stored source:

```python
    cached_cov = reg.get("vertex_covariance")
    if cached_cov is not None:
        beta = np.asarray(reg["vertex_coefficients"])
        cov_list = [np.asarray(c) for c in cached_cov]
    else:
        from .simplex_regression import ols_fit, scheffe_design_matrix
        from .feature_utils import resolve_features
        feature_source = reg.get("feature_source")  # NEW: use stored source
        Y, _ = resolve_features(adata, feature_source, feat_names)
        W, _ = scheffe_design_matrix(weights, degree=1)
        fit = ols_fit(W, Y, robust_se=robust_se, return_covariance=True)
        beta = fit["coefficients"]
        cov_list = fit["covariance"]
```

- [ ] **Step 4: Run tests**
- [ ] **Step 5: Commit**

```bash
git add src/peach/_core/utils/archetype_comparison.py src/peach/tl/feature_regression.py tests/test_statistical/test_archetype_comparison.py
git commit -m "Fix Wald contrasts: track and reuse original feature matrix source"
```

---

### Task 2.2: Make `archetype_driver_regression` explicit about feature matrix

**Files:**
- Modify: `src/peach/tl/feature_regression.py:456-458`

- [ ] **Step 1: No new test — this is a behavior change with warning**

- [ ] **Step 2: Replace silent fallback with explicit warning**

```python
    # Default to pathway scores if available — but warn the user
    if feature_matrix is None and "pathway_scores" in adata.obsm:
        import warnings
        warnings.warn(
            "feature_matrix not specified and adata.obsm['pathway_scores'] exists. "
            "Using pathway scores. Pass feature_matrix='pathway_scores' explicitly "
            "to silence this warning, or feature_matrix=None to use adata.X.",
            UserWarning,
        )
        feature_matrix = "pathway_scores"
```

Per user's note: need to add clarity. Replace the silent fallback with an explicit `FutureWarning` for one release, then remove in the next:

```python
    if feature_matrix is None and "pathway_scores" in adata.obsm:
        import warnings
        warnings.warn(
            "archetype_driver_regression() auto-selects pathway_scores when "
            "available. This will change in v0.6.0 to always default to adata.X. "
            "Pass feature_matrix='pathway_scores' explicitly to keep current "
            "behavior and silence this warning.",
            FutureWarning,
            stacklevel=2,
        )
        feature_matrix = "pathway_scores"
```

Update the docstring to note that `None` currently means "pathway_scores if available, else adata.X" but will change to always mean `adata.X` in v0.6.0.

- [ ] **Step 3: Run existing tests, update any that relied on the implicit fallback**

Run: `conda run -n archetype pytest tests/test_statistical/test_driver_regression.py -v`

- [ ] **Step 4: Commit**

```bash
git add src/peach/tl/feature_regression.py
git commit -m "Remove implicit pathway fallback in archetype_driver_regression"
```

---

### Task 2.3: Normalize PCA loadings in `feature_expansion`

**Files:**
- Modify: `src/peach/tl/flow.py:372-377`

The metric `l_g^T J l_g` conflates PCA loading magnitude with flow-induced expansion. Normalize each gene's loading vector to unit L2 norm so the metric measures pure directional expansion.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_statistical/test_flow_jacobian.py`:

```python
def test_feature_expansion_invariant_to_loading_scale():
    """feature_expansion should depend on flow direction, not PCA loading magnitude."""
    adata, flow_result = _make_flow_fixture(return_model=True)
    model = flow_result["model"]

    # Compute feature expansion
    jac_result = pc.tl.flow_jacobian(adata, flow_result, model)

    # Scale PCA loadings by 10x — should NOT change feature_expansion
    adata2 = adata.copy()
    adata2.varm["PCs"] = adata.varm["PCs"] * 10.0
    jac_result2 = pc.tl.flow_jacobian(adata2, flow_result, model)

    np.testing.assert_allclose(
        jac_result["feature_expansion"],
        jac_result2["feature_expansion"],
        atol=1e-6,
        err_msg="feature_expansion should be invariant to PCA loading scale",
    )
```

- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement the fix**

In `flow_jacobian`, normalize loadings before computing expansion:

```python
        # Normalize each gene's loading to unit norm for scale-invariant expansion
        loading_norms = np.linalg.norm(loadings_trimmed, axis=1, keepdims=True)
        loading_norms = np.maximum(loading_norms, 1e-10)  # avoid division by zero
        loadings_normalized = loadings_trimmed / loading_norms

        # For each gene, compute how its PCA direction is expanded/contracted
        # Vectorized: diag(L_norm @ J @ L_norm.T)
        feature_expansion = np.einsum(
            'gi,ij,gj->g', loadings_normalized, mean_jac, loadings_normalized
        )
```

This also vectorizes the per-gene loop into a single einsum.

- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit**

```bash
git add src/peach/tl/flow.py tests/test_statistical/test_flow_jacobian.py
git commit -m "Normalize PCA loadings in feature_expansion for scale invariance"
```

---

### Task 2.4: Fix sparse `W.T @ Y` computed twice

**Files:**
- Modify: `src/peach/_core/utils/simplex_regression.py:153-154`

- [ ] **Step 1: No new test — performance fix**

- [ ] **Step 2: Implement**

```python
    if is_sparse:
        WtY_raw = W.T @ Y
        WtY = np.asarray(WtY_raw.todense()) if sp.issparse(WtY_raw) else WtY_raw
    else:
```

- [ ] **Step 3: Run tests**

Run: `conda run -n archetype pytest tests/test_core/test_simplex_regression.py -v`

- [ ] **Step 4: Commit**

```bash
git add src/peach/_core/utils/simplex_regression.py
git commit -m "Fix sparse regression: avoid computing W.T @ Y twice"
```

---

### Task 2.5: Vectorize HC3 standard errors

**Files:**
- Modify: `src/peach/_core/utils/simplex_regression.py:292-354`

Replace the per-gene Python loop with vectorized einsum. Also refactor `_hc3_covariance` to call `_hc3_standard_errors` core.

- [ ] **Step 1: Write a correctness-preserving test**

Add to `tests/test_core/test_simplex_regression.py`:

**IMPORTANT:** Before replacing the loop-based implementation, first capture its output as ground truth for comparison.

```python
def test_hc3_vectorized_matches_loop():
    """Vectorized HC3 must produce identical results to the original loop."""
    from peach._core.utils.simplex_regression import (
        ols_fit, scheffe_design_matrix, _hc3_standard_errors
    )
    rng = np.random.default_rng(42)
    n, K, n_features = 200, 4, 50
    W_raw = rng.dirichlet(np.ones(K), n)
    W_design, _ = scheffe_design_matrix(W_raw)
    beta_true = rng.randn(n_features, K)
    Y = W_design @ beta_true.T + rng.randn(n, n_features) * 0.5

    # Compute HC3 SEs via full regression
    result = ols_fit(W_design, Y, robust_se=True)

    # Verify shape, finite, and non-negative
    assert result["standard_errors"].shape == (n_features, K)
    assert np.all(np.isfinite(result["standard_errors"]))
    assert np.all(result["standard_errors"] >= 0)

    # Cross-check: manually compute HC3 for first 3 features via explicit loop
    WtW_inv = np.linalg.solve(W_design.T @ W_design, np.eye(K))
    H_diag = np.clip(np.sum((W_design @ WtW_inv) * W_design, axis=1), 0, 1 - 1e-10)
    residuals = Y - W_design @ result["coefficients"].T
    adjustment = 1.0 / (1 - H_diag)

    for g in range(3):
        e_adj = residuals[:, g] * adjustment
        We = W_design * (e_adj ** 2)[:, np.newaxis]
        meat = W_design.T @ We
        sandwich = WtW_inv @ meat @ WtW_inv
        se_loop = np.sqrt(np.maximum(np.diag(sandwich), 0))
        np.testing.assert_allclose(
            result["standard_errors"][g], se_loop, atol=1e-10,
            err_msg=f"HC3 SE mismatch for feature {g}"
        )
```

- [ ] **Step 2: Implement vectorized HC3**

```python
def _hc3_standard_errors(W, residuals, WtW_inv, H_diag):
    n, p = W.shape
    n_features = residuals.shape[1]

    # HC3 adjustment: e_i / (1 - h_ii)
    adjustment = 1.0 / (1 - H_diag)  # [n], pre-clipped by caller

    # Adjusted residuals squared: [n, n_features]
    e_adj_sq = (residuals * adjustment[:, np.newaxis]) ** 2

    # Meat of sandwich for all features at once:
    # meat_g = W.T @ diag(e_adj_sq[:, g]) @ W
    # = sum_i e_adj_sq[i, g] * outer(W[i], W[i])
    # Vectorized: meat[g] = (W * sqrt(e_adj_sq[:, g:g+1])).T @ (W * sqrt(e_adj_sq[:, g:g+1]))
    # More efficient: compute W.T @ (e_adj_sq[:, g:g+1] * W) for all g
    # Using einsum: meat[g, a, b] = sum_i e_adj_sq[i, g] * W[i, a] * W[i, b]
    # = einsum('ig,ia,ib->gab', e_adj_sq, W, W)
    meat_all = np.einsum('ig,ia,ib->gab', e_adj_sq, W, W)  # [n_features, p, p]

    # Sandwich: (W'W)^{-1} @ meat @ (W'W)^{-1}
    # sandwich[g] = WtW_inv @ meat_all[g] @ WtW_inv
    sandwich_all = np.einsum('ab,gbc,cd->gad', WtW_inv, meat_all, WtW_inv)

    # SE = sqrt(diag(sandwich))
    se = np.sqrt(np.maximum(np.diagonal(sandwich_all, axis1=1, axis2=2), 0))

    return se


def _hc3_covariance(W, residuals, WtW_inv, H_diag):
    """Full HC3 sandwich covariance per feature. Returns list of [p, p] matrices."""
    n, p = W.shape
    adjustment = 1.0 / (1 - H_diag)
    e_adj_sq = (residuals * adjustment[:, np.newaxis]) ** 2
    meat_all = np.einsum('ig,ia,ib->gab', e_adj_sq, W, W)
    sandwich_all = np.einsum('ab,gbc,cd->gad', WtW_inv, meat_all, WtW_inv)
    return [sandwich_all[g] for g in range(sandwich_all.shape[0])]
```

- [ ] **Step 3: Run tests**

Run: `conda run -n archetype pytest tests/test_core/test_simplex_regression.py -v`

- [ ] **Step 4: Commit**

```bash
git add src/peach/_core/utils/simplex_regression.py tests/test_core/test_simplex_regression.py
git commit -m "Vectorize HC3 standard errors: einsum replaces per-gene Python loop"
```

---

### Task 2.6: Pre-build index dict in `compute_feature_similarity`

**Files:**
- Modify: `src/peach/_core/utils/archetype_comparison.py:237-248`

- [ ] **Step 1: No new test — performance fix**

- [ ] **Step 2: Implement**

Replace `names_a.index(g)` with pre-built dicts:

```python
        # Pre-build index maps for O(1) lookup
        idx_map_a = {name: i for i, name in enumerate(names_a)}
        idx_map_b = {name: i for i, name in enumerate(names_b)}

        shared_all = sorted(set(names_a) & set(names_b))
        idx_a_all = [idx_map_a[g] for g in shared_all]
        idx_b_all = [idx_map_b[g] for g in shared_all]

        # ... (sig filter) ...

        shared = [g for g, s in zip(shared_all, sig_shared) if s]
        idx_a = [idx_map_a[g] for g in shared]
        idx_b = [idx_map_b[g] for g in shared]
```

- [ ] **Step 3: Run tests**

Run: `conda run -n archetype pytest tests/test_statistical/test_archetype_comparison.py -v`

- [ ] **Step 4: Commit**

```bash
git add src/peach/_core/utils/archetype_comparison.py
git commit -m "Optimize feature_similarity: pre-built index dicts replace O(n) list.index"
```

---

### Task 2.7: Add SE-aware filtering to pattern classification

**Files:**
- Modify: `src/peach/_core/utils/pattern_classification.py:6-33`
- Test: `tests/test_statistical/test_pattern_classification.py`

Add optional `vertex_ses` parameter. When provided, the exclusive ratio check requires the dominant coefficient to be outside the noise band (|beta| > 2*SE).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_statistical/test_pattern_classification.py`:

```python
def test_exclusive_requires_significance_when_ses_provided():
    """A large but noisy coefficient should NOT be classified as exclusive."""
    from peach._core.utils.pattern_classification import classify_single_feature

    # Large coefficient but huge SE → not reliably exclusive
    result = classify_single_feature(
        vertex_betas=np.array([10.0, 1.0, 0.5]),
        r2=0.5,
        f_pvalue_fdr=0.001,
        vertex_ses=np.array([50.0, 0.1, 0.01]),  # SE on dominant is 50!
    )
    assert result["pattern"] != "archetype-exclusive", (
        "Should not classify as exclusive when dominant coefficient SE is huge"
    )
```

- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement**

```python
def classify_single_feature(
    vertex_betas,
    r2,
    f_pvalue_fdr,
    interaction_f_pvalue_fdr=None,
    fdr_threshold=0.05,
    exclusive_ratio=2.0,
    vertex_ses=None,  # NEW: optional standard errors
):
    # ... (existing NaN and F-test gates) ...

    # Rule 2: archetype-exclusive
    abs_betas = np.abs(vertex_betas)
    sorted_abs = np.sort(abs_betas)[::-1]
    if sorted_abs[0] > 0 and sorted_abs[0] / max(sorted_abs[1], 1e-10) >= exclusive_ratio:
        dominant = int(np.argmax(abs_betas))
        # SE filter: dominant coefficient must be > 2*SE to be reliably exclusive
        if vertex_ses is not None:
            dominant_se = vertex_ses[dominant]
            if dominant_se > 0 and abs_betas[dominant] < 2 * dominant_se:
                # Coefficient is within noise — don't classify as exclusive
                pass  # fall through to interaction/structured
            else:
                return {
                    "pattern": "archetype-exclusive",
                    "r2": float(r2),
                    "details": {"dominant_archetype": dominant},
                }
        else:
            return {
                "pattern": "archetype-exclusive",
                "r2": float(r2),
                "details": {"dominant_archetype": dominant},
            }

    # ... (rest unchanged) ...
```

Also update `classify_all_features` to accept and pass through `vertex_ses`.

- [ ] **Step 4: Run tests**

Run: `conda run -n archetype pytest tests/test_statistical/test_pattern_classification.py -v`

- [ ] **Step 5: Wire SEs through from `feature_simplex_regression`**

In `feature_regression.py`, pass `standard_errors` to `classify_all_features`:

```python
classifications = classify_all_features(
    vertex_coefficients, r_squared, f_pvalue_fdr,
    interaction_f_pvalue_fdr=interaction_f_pvalue_fdr,
    vertex_ses=standard_errors if standard_errors is not None else None,
)
```

- [ ] **Step 6: Commit**

```bash
git add src/peach/_core/utils/pattern_classification.py src/peach/tl/feature_regression.py tests/test_statistical/test_pattern_classification.py
git commit -m "Add SE-aware filtering to pattern classification"
```

---

## Chunk 3: Statistical & Cleanup Fixes

### Task 3.1: Symmetric MMD permutation test

**Files:**
- Modify: `src/peach/_core/utils/archetype_comparison.py:147-156`

The current test permutes only archetype i's weights, creating an asymmetric null. Fix: permute cell indices for BOTH archetypes independently.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_statistical/test_archetype_comparison.py`:

```python
def test_mmd_permutation_symmetric():
    """Permuting (i,j) and (j,i) should give the same p-value distribution."""
    from peach._core.utils.archetype_comparison import compute_archetype_mmd

    rng = np.random.default_rng(42)
    adata = _make_adata_with_weights(rng, n_cells=200, K=3)

    # Should be symmetric for within-fit
    mmd, pval = compute_archetype_mmd(adata, n_permutations=200, seed=42)

    # p-value matrix should be symmetric
    np.testing.assert_array_almost_equal(
        pval, pval.T, decimal=1,
        err_msg="Within-fit MMD p-values should be symmetric"
    )
```

- [ ] **Step 2: Implement the fix**

Replace lines 148-156:

```python
                    # Within-fit: permute BOTH columns independently
                    null_mmds = np.empty(n_permutations)
                    for p in range(n_permutations):
                        perm_i = rng.permutation(n_a)
                        perm_j = rng.permutation(n_a)
                        null_mmds[p] = _weighted_mmd_pair(
                            pca_a, weights_a[perm_i, i],
                            pca_b, weights_b[perm_j, j], bw
                        )
```

- [ ] **Step 3: Run tests**

Run: `conda run -n archetype pytest tests/test_statistical/test_archetype_comparison.py -v`

- [ ] **Step 4: Commit**

```bash
git add src/peach/_core/utils/archetype_comparison.py tests/test_statistical/test_archetype_comparison.py
git commit -m "Fix MMD permutation: symmetric null by permuting both weight columns"
```

---

### Task 3.2: Fix between-fit MMD permutation for mismatched K

**Files:**
- Modify: `src/peach/_core/utils/archetype_comparison.py:157-171`

The bug is in the weight concatenation logic: when `i >= K_b`, the code uses `np.zeros(n_b)` as a stand-in for a nonexistent archetype column, making the permutation test meaningless for that entry. All `K_a x K_b` MMD values are valid comparisons, but the permutation test can only meaningfully shuffle weights that actually exist in both fits.

Fix: for entries where `i >= K_b` or `j >= K_a`, the observed MMD is still valid (it compares archetype i's weighted distribution from A against archetype j's from B), but the permutation test uses a degenerate zero-weight null. Mark these p-values as NaN and document the limitation.

- [ ] **Step 1: Write the test**

```python
def test_between_fit_mmd_mismatched_K():
    """Between-fit MMD with different K: MMD valid everywhere, but p-values
    only meaningful when both archetype indices exist in both fits."""
    from peach._core.utils.archetype_comparison import compute_archetype_mmd

    adata_a = _make_adata_with_weights(K=3)
    adata_b = _make_adata_with_weights(K=5)

    mmd, pval = compute_archetype_mmd(adata_a, adata_b, n_permutations=50)

    assert mmd.shape == (3, 5)
    # All MMD entries should be finite (comparison is always valid)
    assert np.all(np.isfinite(mmd))
    # p-values for i < K_a=3 and j < K_b=5 where i < K_b=5 and j < K_a=3:
    # valid block is [0:3, 0:3]
    assert np.all(np.isfinite(pval[:3, :3]))
    # p-values outside the valid block should be NaN
    assert np.all(np.isnan(pval[:3, 3:])), (
        "p-values for j >= K_a should be NaN (no matching archetype in fit A)"
    )
```

- [ ] **Step 2: Implement**

In the between-fit permutation block (lines 157-171), add a guard:

```python
                else:
                    # Between-fit permutation:
                    # Can only permute meaningfully when both i and j have
                    # real weight columns in both fits.
                    if i >= K_b or j >= K_a:
                        pvalue_matrix[i, j] = float("nan")
                        continue

                    combined_pca = np.vstack([pca_a, pca_b])
                    combined_w_i = np.concatenate([weights_a[:, i], weights_b[:, i]])
                    combined_w_j = np.concatenate([weights_a[:, j], weights_b[:, j]])
                    null_mmds = np.empty(n_permutations)
                    for p in range(n_permutations):
                        perm = rng.permutation(n_a + n_b)
                        null_mmds[p] = _weighted_mmd_pair(
                            combined_pca[perm[:n_a]], combined_w_i[perm[:n_a]],
                            combined_pca[perm[n_a:]], combined_w_j[perm[n_a:]], bw
                        )
```

- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit**

```bash
git add src/peach/_core/utils/archetype_comparison.py tests/test_statistical/test_archetype_comparison.py
git commit -m "Fix between-fit MMD: NaN p-values for cross-K archetype entries"
```

---

### Task 3.3: Clip `inverse_ilr` to prevent overflow

**Files:**
- Modify: `src/peach/_core/utils/ilr_transform.py:120-122`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_core/test_ilr_transform.py`:

```python
def test_inverse_ilr_extreme_coordinates():
    """Extreme ILR coordinates should produce valid simplex weights, not inf/nan."""
    from peach._core.utils.ilr_transform import inverse_ilr
    extreme = np.array([[100.0, -100.0], [-50.0, 200.0]])
    result = inverse_ilr(extreme)
    assert np.all(np.isfinite(result)), f"inverse_ilr produced non-finite values: {result}"
    np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-10)
```

- [ ] **Step 2: Run test to verify it fails** (currently produces inf/nan)
- [ ] **Step 3: Implement**

```python
    # CLR -> composition: exp and normalize (clip to prevent overflow)
    clr = np.clip(clr, -500, 500)
    W = np.exp(clr)
    W = W / W.sum(axis=1, keepdims=True)
```

- [ ] **Step 4: Run tests**

Run: `conda run -n archetype pytest tests/test_core/test_ilr_transform.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/peach/_core/utils/ilr_transform.py tests/test_core/test_ilr_transform.py
git commit -m "Fix inverse_ilr: clip CLR values to prevent exp overflow"
```

---

### Task 3.4: Remove global RNG mutation in `robust_mannwhitneyu_test`

**Files:**
- Modify: `src/peach/_core/utils/statistical_tests.py:272-274`

- [ ] **Step 1: No new test — removing a bad pattern**

- [ ] **Step 2: Implement**

Replace the global seed mutation with a local RNG seeded deterministically from the feature name (using hashlib, not `hash()`, since Python's `hash()` is randomized per-session):

```python
        if data_std > 0:
            noise_scale = data_std * 1e-8
            # Use local RNG with deterministic seed (no global state mutation)
            import hashlib
            seed_val = int(hashlib.sha256(feature_name.encode()).hexdigest()[:8], 16)
            local_rng = np.random.default_rng(seed_val)
            noise1 = local_rng.normal(0, noise_scale, len(group1))
            noise2 = local_rng.normal(0, noise_scale, len(group2))
```

- [ ] **Step 3: Run tests**

Run: `conda run -n archetype pytest tests/test_statistical/test_gene_associations.py -v`

- [ ] **Step 4: Commit**

```bash
git add src/peach/_core/utils/statistical_tests.py
git commit -m "Fix mannwhitneyu: use local RNG instead of mutating global np.random.seed"
```

---

### Task 3.5: Remove redundant log-likelihood in Dirichlet EM

**Files:**
- Modify: `src/peach/_core/utils/dirichlet_mixture.py:137-148`

- [ ] **Step 1: No new test — performance fix**

- [ ] **Step 2: Implement**

Compute log-likelihood from the log-responsibilities (they share the same computation):

```python
        for iteration in range(self.max_iter):
            # E-step: compute responsibilities and log-likelihood together
            log_resp_unnorm = self._log_joint(W, alphas, mix_weights)  # [n, C]
            log_resp_max = log_resp_unnorm.max(axis=1, keepdims=True)
            log_sum_exp = log_resp_max.squeeze() + np.log(
                np.exp(log_resp_unnorm - log_resp_max).sum(axis=1)
            )
            ll = log_sum_exp.sum()
            log_resp = log_resp_unnorm - log_sum_exp[:, np.newaxis]
            resp = np.exp(log_resp)

            if abs(ll - prev_ll) < self.tol and iteration > 0:
                converged = True
                break
            prev_ll = ll
            # ... M-step unchanged ...
```

Add a `_log_joint` method that both `_log_responsibilities` and `_log_likelihood` delegate to:

```python
    def _log_joint(self, W, alphas, mix_weights):
        """Log joint probabilities: log(pi_c * p(w|alpha_c)). [n, n_components]"""
        n = W.shape[0]
        log_joint = np.zeros((n, self.n_components))
        for c in range(self.n_components):
            log_joint[:, c] = (
                np.log(np.clip(mix_weights[c], 1e-300, None))
                + self._log_dirichlet_pdf(W, alphas[c])
            )
        return log_joint
```

Then simplify `_log_responsibilities` and `_log_likelihood` to use `_log_joint`.

- [ ] **Step 3: Run tests**

Run: `conda run -n archetype pytest tests/test_statistical/test_gmm_api.py -v`

- [ ] **Step 4: Commit**

```bash
git add src/peach/_core/utils/dirichlet_mixture.py
git commit -m "Dirichlet EM: deduplicate log-likelihood computation"
```

---

### Task 3.6: Add `flow_model/result` consistency docstring warning

**Files:**
- Modify: `src/peach/tl/flow.py` (docstrings for `flow_jacobian`, `flow_bifurcation`, `flow_feature_graph`, `flow_temporal_feature_graph`)

- [ ] **Step 1: Add warnings to all 4 docstrings**

Add to each function's `flow_model` parameter doc:

```
    flow_model : FlowModel
        The trained FlowModel. **Must** be the same model that produced
        ``flow_result`` — passing a mismatched model will produce silently
        wrong results. Use ``flow_within(..., return_model=True)`` and
        access via ``flow_result['model']``.
```

- [ ] **Step 2: Commit**

```bash
git add src/peach/tl/flow.py
git commit -m "Add docstring warnings: flow_model must match flow_result"
```

---

### Task 3.7: Dirichlet mixture vertex smoothing (documentation + guard)

**Files:**
- Modify: `src/peach/_core/utils/dirichlet_mixture.py:70-73`

Per user note: unclear if Dirichlet is fully tested as an alternative to ILR. Add a vertex fraction warning and document the limitation.

- [ ] **Step 1: Add vertex detection and warning**

In `DirichletMixture.fit`, after line 73:

```python
        # Warn about vertex-heavy data
        vertex_fraction = np.mean(np.any(W < 1e-6, axis=1))
        if vertex_fraction > 0.1:
            import warnings
            warnings.warn(
                f"{vertex_fraction:.0%} of cells are near simplex vertices "
                f"(at least one weight < 1e-6). Dirichlet mixture fitting may "
                f"be unstable for boundary-heavy data. Consider using "
                f"model_type='gaussian' (GMM on ILR coordinates) instead.",
                UserWarning,
            )
```

- [ ] **Step 2: Commit**

```bash
git add src/peach/_core/utils/dirichlet_mixture.py
git commit -m "Add vertex-heavy data warning to Dirichlet mixture fitting"
```

---

## Chunk 4: Deep Adversarial Tests

### Task 4.1: Create Dirichlet mixture test suite

**Files:**
- Create: `tests/test_core/test_dirichlet_mixture.py`

- [ ] **Step 1: Write test file**

```python
"""Tests for DirichletMixture: convergence, recovery, edge cases."""

import numpy as np
import pytest
from peach._core.utils.dirichlet_mixture import DirichletMixture


@pytest.fixture
def dirichlet_rng():
    return np.random.default_rng(42)


class TestDirichletConvergence:
    """Test that the EM algorithm converges."""

    def test_converges_on_well_separated_components(self, dirichlet_rng):
        """Two clearly distinct Dirichlet components should converge."""
        K = 3
        # Component 1: concentrated near vertex 0
        alpha1 = np.array([10.0, 0.5, 0.5])
        # Component 2: concentrated near vertex 2
        alpha2 = np.array([0.5, 0.5, 10.0])

        W1 = dirichlet_rng.dirichlet(alpha1, 200)
        W2 = dirichlet_rng.dirichlet(alpha2, 200)
        W = np.vstack([W1, W2])

        model = DirichletMixture(n_components=2, max_iter=200, random_state=42)
        model.fit(W)
        assert model.converged_, "Dirichlet EM should converge on well-separated data"

    def test_bic_selects_correct_n_components(self, dirichlet_rng):
        """BIC should prefer 2 components for 2-component data."""
        K = 3
        W1 = dirichlet_rng.dirichlet([8, 1, 1], 300)
        W2 = dirichlet_rng.dirichlet([1, 1, 8], 300)
        W = np.vstack([W1, W2])

        bics = {}
        for nc in [1, 2, 3, 4]:
            model = DirichletMixture(n_components=nc, max_iter=200, random_state=42)
            model.fit(W)
            bics[nc] = model.bic(W)

        assert bics[2] < bics[1], "BIC should prefer 2 over 1 components"
        assert bics[2] < bics[4], "BIC should prefer 2 over 4 components"


class TestDirichletRecovery:
    """Test parameter recovery."""

    def test_recovers_alpha_direction(self, dirichlet_rng):
        """Fitted alphas should point in the same direction as ground truth."""
        alpha_true = np.array([10.0, 2.0, 0.5])
        W = dirichlet_rng.dirichlet(alpha_true, 500)

        model = DirichletMixture(n_components=1, max_iter=200, random_state=42)
        model.fit(W)

        # Check that the fitted mean direction matches ground truth
        fitted_mean = model.means_[0]
        true_mean = alpha_true / alpha_true.sum()
        correlation = np.corrcoef(fitted_mean, true_mean)[0, 1]
        assert correlation > 0.9, f"Fitted mean direction correlation = {correlation}"


class TestDirichletEdgeCases:
    """Edge cases and robustness."""

    def test_single_component(self, dirichlet_rng):
        """Single component should work."""
        W = dirichlet_rng.dirichlet([2, 2, 2], 100)
        model = DirichletMixture(n_components=1, max_iter=100, random_state=42)
        model.fit(W)
        assert model.alphas_.shape == (1, 3)

    def test_vertex_data_warns(self, dirichlet_rng):
        """Data with many vertex cells should trigger a warning."""
        W = np.zeros((100, 3))
        # 80% at vertex 0
        W[:80, 0] = 1.0
        # 20% at vertex 1
        W[80:, 1] = 1.0

        model = DirichletMixture(n_components=2, max_iter=50, random_state=42)
        with pytest.warns(UserWarning, match="near simplex vertices"):
            model.fit(W)

    def test_k_equals_2(self, dirichlet_rng):
        """K=2 (line simplex) should work."""
        W = dirichlet_rng.dirichlet([5, 2], 200)
        model = DirichletMixture(n_components=2, max_iter=100, random_state=42)
        model.fit(W)
        assert model.alphas_.shape == (2, 2)

    def test_predict_returns_valid_labels(self, dirichlet_rng):
        """Predictions should be valid component indices."""
        W = dirichlet_rng.dirichlet([3, 3, 3], 100)
        model = DirichletMixture(n_components=3, max_iter=100, random_state=42)
        model.fit(W)
        labels = model.predict(W)
        assert set(labels).issubset({0, 1, 2})
        assert len(labels) == 100
```

- [ ] **Step 2: Run tests**

Run: `conda run -n archetype pytest tests/test_core/test_dirichlet_mixture.py -v`

- [ ] **Step 3: Commit**

```bash
git add tests/test_core/test_dirichlet_mixture.py
git commit -m "Add Dirichlet mixture test suite: convergence, recovery, edge cases"
```

---

### Task 4.2: Create adversarial redteam test suite

**Files:**
- Create: `tests/test_statistical/test_redteam_adversarial.py`

This is the deep, harsh test suite that covers all the attack surfaces identified by the Gremlin that aren't addressed by the code fixes above.

- [ ] **Step 1: Write test file**

```python
"""Adversarial redteam tests for v0.5.0 features.

These tests exercise degenerate inputs, boundary conditions, and silent
corruption scenarios identified by the Gremlin/Reviewer2 audit.
"""

import numpy as np
import pytest
import anndata as ad
import scipy.sparse as sp


# =====================================================================
# Helpers
# =====================================================================

def _make_synthetic_adata(n_cells=200, n_genes=100, K=4, seed=42):
    """Minimal AnnData with PCA, weights, and var_names."""
    rng = np.random.default_rng(seed)
    X = rng.randn(n_cells, n_genes).astype(np.float32)
    adata = ad.AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obsm["X_pca"] = rng.randn(n_cells, 20).astype(np.float32)
    adata.obsm["cell_archetype_weights"] = rng.dirichlet(np.ones(K), n_cells)
    # PCA loadings
    adata.varm["PCs"] = rng.randn(n_genes, 20).astype(np.float32)
    return adata


def _make_flow_pair(adata, seed=42):
    """Train a minimal flow model for testing."""
    from peach._core.utils.flow_matching import FlowModel
    rng = np.random.default_rng(seed)
    n = adata.n_obs
    # Split into source/target by simple partition
    adata.obs["condition"] = ["source"] * (n // 2) + ["target"] * (n - n // 2)

    import peach as pc
    result = pc.tl.flow_within(
        adata,
        source={"condition": "source"},
        target={"condition": "target"},
        n_epochs=50,
        return_model=True,
        random_state=seed,
    )
    return result


# =====================================================================
# Degenerate Data Attacks
# =====================================================================

class TestDegenerateData:
    """Test behavior with degenerate or extreme inputs."""

    def test_zero_variance_features_regression(self):
        """Regression on constant features should return flat classifications."""
        import peach as pc
        adata = _make_synthetic_adata(n_cells=100, n_genes=10, K=3)
        # Make ALL features constant (zero variance)
        adata.X = np.ones_like(adata.X)

        result = pc.tl.feature_simplex_regression(adata)
        # All features should be flat (R2 = 0, F-test nonsig)
        for cls in result["classifications"]:
            assert cls["pattern"] == "flat", (
                f"Constant feature classified as {cls['pattern']}, expected flat"
            )

    def test_identical_cells_mmd(self):
        """MMD between identical distributions should be ~0."""
        from peach._core.utils.flow_matching import compute_mmd
        X = np.ones((100, 5))  # all identical
        Y = np.ones((100, 5))
        mmd = compute_mmd(X, Y)
        assert mmd < 1e-6, f"MMD between identical points = {mmd}, expected ~0"

    def test_sparse_all_zeros_regression(self):
        """Sparse matrix of all zeros should not crash."""
        from peach._core.utils.simplex_regression import ols_fit, scheffe_design_matrix
        rng = np.random.default_rng(42)
        W = rng.dirichlet(np.ones(3), 100)
        W_design, _ = scheffe_design_matrix(W)
        Y = sp.csr_matrix((100, 50))  # all zeros
        result = ols_fit(W_design, Y)
        assert np.all(result["r_squared"] == 0)

    def test_single_archetype_weight_concentrated(self):
        """Weighted MMD with single-cell dominance should not produce inf."""
        from peach._core.utils.archetype_comparison import _weighted_mmd_pair
        pca = np.random.randn(100, 5)
        w_x = np.zeros(100)
        w_x[0] = 1.0  # single cell dominates
        w_y = np.ones(100) / 100
        bw = 1.0

        result = _weighted_mmd_pair(pca, w_x, pca, w_y, bw)
        assert np.isfinite(result), f"Concentrated weights produced {result}"


# =====================================================================
# K=2 Edge Cases
# =====================================================================

class TestKEquals2:
    """Test K=2 archetypes (line simplex) — all modules."""

    def test_regression_k2(self):
        """Regression with K=2 should work and produce 2 coefficients."""
        import peach as pc
        adata = _make_synthetic_adata(K=2, n_genes=20)
        result = pc.tl.feature_simplex_regression(adata)
        assert result["vertex_coefficients"].shape[1] == 2

    def test_ilr_k2(self):
        """ILR transform with K=2 produces 1-dimensional coordinates."""
        from peach._core.utils.ilr_transform import ilr_transform
        W = np.column_stack([np.linspace(0.1, 0.9, 100),
                             np.linspace(0.9, 0.1, 100)])
        ilr = ilr_transform(W)
        assert ilr.shape == (100, 1)

    def test_gmm_k2(self):
        """GMM decomposition with K=2 should work."""
        import peach as pc
        adata = _make_synthetic_adata(K=2, n_genes=20)
        result = pc.tl.feature_simplex_decomposition(adata, n_components_range=(2, 4))
        assert result["n_components"] >= 2


# =====================================================================
# Flow Silent Corruption
# =====================================================================

class TestFlowSilentCorruption:
    """Tests that would silently give wrong results without fixes."""

    def test_gene_alignment_t_has_effect(self):
        """Different t values must produce different alignment scores."""
        import peach as pc
        adata = _make_synthetic_adata(n_cells=100, n_genes=50, K=3)
        flow_result = _make_flow_pair(adata)

        if "model" not in flow_result:
            pytest.skip("No model in flow_result")

        r1 = pc.tl.flow_gene_alignment(adata, flow_result, t=0.1)
        r2 = pc.tl.flow_gene_alignment(adata, flow_result, t=0.9)

        # They should differ (unless model is degenerate)
        diff = np.abs(r1["alignment_scores"] - r2["alignment_scores"]).max()
        assert diff > 1e-8 or r1.get("velocity_mode") == "displacement"

    def test_inverse_ilr_no_nan(self):
        """inverse_ilr should never return NaN even with extreme inputs."""
        from peach._core.utils.ilr_transform import inverse_ilr
        extreme = np.array([
            [500, -500],
            [-1000, 1000],
            [0, 0],
            [1, -1],
        ])
        result = inverse_ilr(extreme)
        assert np.all(np.isfinite(result)), f"Got non-finite: {result}"
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-10)

    def test_mmd_degenerate_returns_nan(self):
        """compute_mmd with <2 samples should return NaN."""
        from peach._core.utils.flow_matching import compute_mmd
        assert np.isnan(compute_mmd(np.zeros((1, 3)), np.ones((50, 3))))
        assert np.isnan(compute_mmd(np.zeros((50, 3)), np.ones((0, 3))))


# =====================================================================
# Permutation Test Validity
# =====================================================================

class TestPermutationValidity:
    """Verify permutation tests have correct type-I error."""

    @pytest.mark.slow
    def test_mmd_permutation_uniform_under_null(self):
        """Under the null (same distribution), p-values should be ~uniform."""
        from peach._core.utils.archetype_comparison import compute_archetype_mmd

        rng = np.random.default_rng(42)
        p_values = []

        for trial in range(20):
            adata = ad.AnnData(np.zeros((200, 1)))
            # Same Dirichlet for all cells — archetype distributions identical
            W = rng.dirichlet(np.ones(3), 200)
            adata.obsm["cell_archetype_weights"] = W
            adata.obsm["X_pca"] = rng.randn(200, 5)

            _, pval = compute_archetype_mmd(adata, n_permutations=100, seed=trial)
            # Off-diagonal p-values (within-fit)
            for i in range(3):
                for j in range(i + 1, 3):
                    p_values.append(pval[i, j])

        # Under H0, p-values should be roughly uniform [0, 1]
        # At alpha=0.05, expect ~5% rejections (with some tolerance)
        rejection_rate = np.mean(np.array(p_values) < 0.05)
        assert rejection_rate < 0.20, (
            f"Type-I error rate = {rejection_rate:.2f}, expected < 0.20 "
            f"(some inflation expected due to soft weights)"
        )


# =====================================================================
# Regression Edge Cases
# =====================================================================

class TestRegressionEdgeCases:
    """Edge cases in simplex regression."""

    def test_collinear_design_gives_warning(self):
        """Near-singular design matrix should warn, not crash."""
        from peach._core.utils.simplex_regression import ols_fit, scheffe_design_matrix
        rng = np.random.default_rng(42)
        # Degenerate: all cells at same point on simplex
        W = np.tile([0.5, 0.3, 0.2], (100, 1))
        W += rng.randn(100, 3) * 1e-10  # tiny perturbation
        W = W / W.sum(axis=1, keepdims=True)
        W_design, _ = scheffe_design_matrix(W)
        Y = rng.randn(100, 10)

        with pytest.warns(RuntimeWarning):
            result = ols_fit(W_design, Y)
        assert np.all(np.isfinite(result["r_squared"]))

    def test_n_equals_p_regression(self):
        """Exactly determined system (n == p) should work."""
        from peach._core.utils.simplex_regression import ols_fit, scheffe_design_matrix
        rng = np.random.default_rng(42)
        K = 4
        n = K  # exactly determined
        W = rng.dirichlet(np.ones(K), n)
        W_design, _ = scheffe_design_matrix(W)
        Y = rng.randn(n, 5)

        result = ols_fit(W_design, Y, robust_se=False)
        # Should produce exact fit (R2 = 1)
        np.testing.assert_allclose(result["r_squared"], 1.0, atol=1e-6)


# =====================================================================
# Feature Expansion Normalization
# =====================================================================

class TestFeatureExpansion:
    """Verify feature_expansion is scale-invariant."""

    def test_expansion_invariant_to_loading_scale(self):
        """Scaling PCA loadings should not change feature_expansion ranking."""
        import peach as pc
        adata = _make_synthetic_adata(n_cells=100, n_genes=50, K=3)
        flow_result = _make_flow_pair(adata)
        if "model" not in flow_result:
            pytest.skip("No model")

        r1 = pc.tl.flow_jacobian(adata, flow_result, flow_result["model"])

        adata2 = adata.copy()
        adata2.varm["PCs"] = adata.varm["PCs"] * 100
        r2 = pc.tl.flow_jacobian(adata2, flow_result, flow_result["model"])

        np.testing.assert_allclose(
            r1["feature_expansion"], r2["feature_expansion"], atol=1e-4,
            err_msg="feature_expansion changed with loading scale"
        )
```

- [ ] **Step 2: Run all adversarial tests**

Run: `conda run -n archetype pytest tests/test_statistical/test_redteam_adversarial.py -v`

- [ ] **Step 3: Fix any failures exposed by the tests** (iterative)

- [ ] **Step 4: Commit**

```bash
git add tests/test_statistical/test_redteam_adversarial.py
git commit -m "Add adversarial redteam test suite: degenerate data, K=2, type-I error, corruption"
```

---

## Implementation Notes

### Open Design Decisions (resolved in user notes)

| # | Decision | Resolution |
|---|----------|------------|
| 4 | flow_model/result mismatch | Docstring warning only — no runtime check |
| 7 | Normalize l_g to what? | Unit L2 norm: `l_g / ||l_g||` |
| 8 | Asymmetric permutation fix | Permute BOTH weight columns independently |
| 10 | Between-fit K mismatch | NaN p-values for cross-K entries |
| 18 | Dirichlet vertex handling | Warning + documentation (not a full fix) |
| 20 | K=2 pattern classification | OK as is (accepted) |
| 21 | component_regression K > 20 | Keep max 20 (accepted) |

### Execution Order

**Chunk 0** (helpers) must run first. Then **Chunks 1-3** sequentially.
**Chunk 4** (adversarial tests) must run LAST — it depends on code changes from all prior chunks (e.g., Task 4.1's `test_vertex_data_warns` needs Task 3.7's warning code).

**Within each chunk, tasks must be sequential** — multiple tasks modify the same files:
- Chunk 1: Tasks 1.1, 1.4, 1.5, 1.7 all modify `flow.py`. Do 1.7 (print→logging) LAST.
- Chunk 2: Tasks 2.1 and 2.2 both modify `feature_regression.py`.
- Chunk 3: No conflicts, but sequential is still safer.

### Verification

After all chunks complete, run the full test suite:

```bash
conda run -n archetype pytest tests/ -v --tb=short -x
```

Also run the comprehensive e2e test to verify no visual regressions:

```bash
conda run -n archetype pytest tests/test_e2e_v050_comprehensive.py -v
```

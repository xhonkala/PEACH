# Per-Cell Flow Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-cell gene alignment (default on, top-2500 cap) and per-cell Jacobian feature expansion to PEACH flow analysis, with normalization consistency fix and stress tests.

**Architecture:** Two function modifications in `src/peach/tl/flow.py` — `flow_gene_alignment` gets `per_cell=True` default with `n_top_features=2500` cap and `normalize` flag; `flow_jacobian` gets `per_cell_features=True` with matching cap. Internal callers updated to pass `per_cell=False`. Stress tests validate math (PCA reconstruction) and biology (HSC myeloid TFs).

**Tech Stack:** numpy (einsum, linalg), scipy (stats for Spearman), existing FlowModel/flow_matching infrastructure

**Spec:** `docs/superpowers/specs/2026-03-19-per-cell-flow-features-design.md`

**Conda env:** `archetype`

---

### Task 1: Update `flow_gene_alignment` signature and normalization

**Files:**
- Modify: `src/peach/tl/flow.py:232-348`

- [ ] **Step 1: Update function signature**

Add `n_top_features` and `normalize` parameters, change `per_cell` default:

```python
def flow_gene_alignment(
    adata: AnnData,
    flow_result: dict,
    *,
    t: float | None = None,
    n_top: int = 50,
    pca_loadings_key: str | None = None,
    n_permutations: int = 0,
    per_cell: bool = True,           # CHANGED from False
    n_top_features: int = 2500,      # NEW
    normalize: bool = True,          # NEW
    random_state: int = 42,
) -> dict:
```

Update docstring to document new parameters and changed default.

- [ ] **Step 2: Implement normalized aggregated alignment**

Replace the aggregated alignment computation (line 298) with normalization-aware version. Insert after `loadings_trimmed = loadings[:, :n_pcs]` (line 295):

```python
    # Normalize loadings for direction-only alignment (cosine-like)
    vel_norm_agg = None  # defined unconditionally for permutation null safety
    if normalize:
        loading_norms = np.linalg.norm(loadings_trimmed, axis=1, keepdims=True)
        loadings_for_agg = loadings_trimmed / np.maximum(loading_norms, 1e-10)
        vel_norm_agg = mean_velocity / (np.linalg.norm(mean_velocity) + 1e-10)
        alignment_scores = loadings_for_agg @ vel_norm_agg  # [n_genes]
    else:
        alignment_scores = loadings_trimmed @ mean_velocity  # [n_genes] — raw dot product
```

- [ ] **Step 3: Implement top-N per-cell computation**

Replace the existing per-cell block (lines 314-326) with top-N capped version:

```python
    if per_cell:
        # Select top genes by aggregated alignment score
        n_top_feat = min(n_top_features, len(gene_names))
        top_feat_idx = np.argsort(np.abs(alignment_scores))[-n_top_feat:][::-1]
        top_feat_idx = np.sort(top_feat_idx)  # restore original ordering
        top_feat_names = [gene_names[i] for i in top_feat_idx]
        loadings_top = loadings_trimmed[top_feat_idx]  # [n_top_feat, n_pcs]

        if t is not None and model is not None:
            velocity_per_cell = model.velocity_at(source_pca, t)
        else:
            velocity_per_cell = flow_result["transported"] - source_pca

        # Always normalize per-cell (cosine similarity)
        vel_norm_pc = velocity_per_cell / (
            np.linalg.norm(velocity_per_cell, axis=1, keepdims=True) + 1e-10
        )
        load_norm_pc = loadings_top / (
            np.linalg.norm(loadings_top, axis=1, keepdims=True) + 1e-10
        )
        per_cell_alignment = vel_norm_pc @ load_norm_pc.T  # [n_source, n_top_feat]
        result["per_cell_alignment"] = per_cell_alignment
        result["per_cell_gene_names"] = top_feat_names
        result["per_cell_gene_indices"] = top_feat_idx
```

- [ ] **Step 4: Fix permutation null to respect normalize flag**

Replace the permutation loop (lines 328-333):

```python
    if n_permutations > 0:
        rng = np.random.default_rng(random_state)
        null_scores = np.zeros((n_permutations, len(alignment_scores)))
        for i in range(n_permutations):
            perm_loadings = loadings_trimmed[rng.permutation(len(loadings_trimmed))]
            if normalize:
                perm_norms = np.linalg.norm(perm_loadings, axis=1, keepdims=True)
                perm_loadings_norm = perm_loadings / np.maximum(perm_norms, 1e-10)
                null_scores[i] = perm_loadings_norm @ vel_norm_agg
            else:
                null_scores[i] = perm_loadings @ mean_velocity
```

Note: `vel_norm_agg` is defined in the normalize block from Step 2. When `normalize=False`, use raw `mean_velocity`. Ensure `vel_norm_agg` is available when `n_permutations > 0` and `normalize=True` by defining it before the permutation block.

- [ ] **Step 5: Verify existing tests still pass**

Run: `conda run -n archetype python -m pytest tests/test_statistical/test_flow_jacobian.py -v -x 2>&1 | head -80`

Expected: some shape assertion failures in `test_per_cell_gene_alignment` (line 228) — that's expected and will be fixed in Task 3.

---

### Task 2: Update `flow_jacobian` with per-cell feature expansion

**Files:**
- Modify: `src/peach/tl/flow.py:351-426`

- [ ] **Step 1: Update function signature**

```python
def flow_jacobian(
    adata: AnnData,
    flow_result: dict,
    flow_model: "FlowModel",
    *,
    t: float = 0.5,
    evaluation_points: np.ndarray | None = None,
    pca_loadings_key: str | None = None,
    aggregate: str = "mean",
    per_cell_features: bool = True,   # NEW
    n_top_features: int = 2500,       # NEW
) -> dict:
```

Update docstring to document new parameters.

- [ ] **Step 2: Add per-cell feature expansion computation**

After the existing `feature_expansion` computation (line 413) and before the `else:` at line 414, add the per-cell block **inside the same `if pca_loadings_key in adata.varm:` branch** (line 401). This is critical — `loadings_normalized` and `jac` are only valid inside this branch:

```python
        # Per-cell feature expansion for top genes (inside the if-branch)
        if per_cell_features:
            n_top_feat = min(n_top_features, loadings.shape[0])
            top_feat_idx = np.argsort(np.abs(feature_expansion))[-n_top_feat:][::-1]
            top_feat_idx = np.sort(top_feat_idx)
            L_top = loadings_normalized[top_feat_idx]  # [n_top_feat, n_pcs]

            # Per-cell quadratic form: L_g^T J_c L_g for each cell c, gene g
            per_cell_exp = np.einsum(
                'gi,cij,gj->cg', L_top, jac, L_top
            )  # [n_points, n_top_feat]

            gene_names_all = list(adata.var_names) if hasattr(adata, 'var_names') else []
            top_feat_names = [gene_names_all[i] for i in top_feat_idx] if gene_names_all else []
```

Then in the result dict construction (after line 424), add conditionally:
```python
    if pca_loadings_key in adata.varm and per_cell_features:
        result["per_cell_expansion"] = per_cell_exp
        result["per_cell_expansion_gene_names"] = top_feat_names
        result["per_cell_expansion_gene_indices"] = top_feat_idx
```

When PCA loadings are absent, per-cell features are silently skipped (matching existing `feature_expansion = np.zeros(0)` fallback).

- [ ] **Step 3: Verify feature_expansion_invariant_to_loading_scale test still passes**

Run: `conda run -n archetype python -m pytest tests/test_statistical/test_flow_jacobian.py::test_feature_expansion_invariant_to_loading_scale -v`

Expected: PASS (aggregated feature_expansion unchanged)

---

### Task 3: Update internal callers and fix bugs

**Files:**
- Modify: `src/peach/tl/flow.py:711,892` (internal `flow_gene_alignment` calls)
- Modify: `src/peach/pl/flow.py:581` (internal `flow_gene_alignment` call)
- Modify: `scripts/generate_12c_flow.py:464` (bug fix)

- [ ] **Step 1: Add `per_cell=False` to internal callers in flow.py**

At line 711 inside `flow_feature_graph()`:
```python
    alignment = flow_gene_alignment(adata, flow_result, per_cell=False)
```

At line 892 inside `flow_temporal_feature_graph()`:
```python
    alignment = flow_gene_alignment(adata, flow_result, per_cell=False)
```

- [ ] **Step 2: Add `per_cell=False` to internal caller in pl/flow.py**

At line 581 inside the auto-select features block:
```python
        align = flow_gene_alignment(adata, flow_result, n_permutations=0, per_cell=False)
```

- [ ] **Step 3: Fix generate_12c_flow.py per-cell key bugs (lines 464-475)**

Three lines need fixing — the key name, the print string, and the gene name lookup:

At line 464, change:
```python
pc_scores = percell_align["per_cell_scores"]
```
to:
```python
pc_scores = percell_align["per_cell_alignment"]
```

At line 466, change:
```python
print(f"  (n_source_cells x n_genes)")
```
to:
```python
print(f"  (n_source_cells x n_top_features)")
```

At line 470, change:
```python
gene_names_pc = percell_align["gene_names"]
```
to:
```python
gene_names_pc = percell_align["per_cell_gene_names"]
```

This is critical: without the line 470 fix, column indices into the subsetted `[n_source, n_top_feat]` matrix would be mapped to the full gene name list, producing silently wrong gene labels.

- [ ] **Step 4: Run flow tests to verify no regressions from internal caller changes**

Run: `conda run -n archetype python -m pytest tests/test_statistical/test_flow_jacobian.py -v -x 2>&1 | head -80`

Expected: `test_per_cell_gene_alignment` may still fail on shape (fixed in Task 4). All other tests should pass.

---

### Task 4: Update existing test assertions

**Files:**
- Modify: `tests/test_statistical/test_flow_jacobian.py:226-228`
- Modify: `tests/test_statistical/test_chunk1_senior.py:331,342,350-367`

- [ ] **Step 1: Fix shape assertion in test_flow_jacobian.py**

At line 226-228, change:
```python
    result = flow_gene_alignment(adata, flow_result, per_cell=True)
    assert "per_cell_alignment" in result
    assert result["per_cell_alignment"].shape == (50, n_genes)
```
to:
```python
    result = flow_gene_alignment(adata, flow_result, per_cell=True)
    assert "per_cell_alignment" in result
    assert "per_cell_gene_names" in result
    assert "per_cell_gene_indices" in result
    n_top_feat = min(2500, n_genes)
    assert result["per_cell_alignment"].shape == (50, n_top_feat)
    assert len(result["per_cell_gene_names"]) == n_top_feat
    assert len(result["per_cell_gene_indices"]) == n_top_feat

    # Also test with explicit cap smaller than n_genes
    result_capped = flow_gene_alignment(
        adata, flow_result, per_cell=True, n_top_features=10
    )
    assert result_capped["per_cell_alignment"].shape == (50, 10)
    assert len(result_capped["per_cell_gene_names"]) == 10
    assert len(result_capped["per_cell_gene_indices"]) == 10
```

- [ ] **Step 2: Fix shape assertions in test_chunk1_senior.py**

At line 331, change:
```python
        assert result["per_cell_alignment"].shape == (n_source, n_genes)
```
to:
```python
        n_top_feat = min(2500, n_genes)
        assert result["per_cell_alignment"].shape == (n_source, n_top_feat)
```

At line 342, same change:
```python
        n_top_feat = min(2500, n_genes)
        assert result["per_cell_alignment"].shape == (n_source, n_top_feat)
```

Lines 350-353 (value bounds) and 361-367 (displacement vs instantaneous): unchanged — cosine bounds and inequality checks still valid on the subsetted matrix.

- [ ] **Step 3: Run all updated tests**

Run: `conda run -n archetype python -m pytest tests/test_statistical/test_flow_jacobian.py tests/test_statistical/test_chunk1_senior.py -v -x 2>&1 | tail -30`

Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add src/peach/tl/flow.py src/peach/pl/flow.py scripts/generate_12c_flow.py tests/test_statistical/test_flow_jacobian.py tests/test_statistical/test_chunk1_senior.py
git commit -m "$(cat <<'EOF'
Add per-cell gene alignment (default on, top-2500 cap) and per-cell Jacobian expansion

- flow_gene_alignment: per_cell=True default, n_top_features=2500 cap,
  normalize flag fixing aggregated/per-cell inconsistency, permutation
  null respects normalize
- flow_jacobian: per_cell_features=True, per-cell einsum quadratic form
  for top 2500 genes by |feature_expansion|
- Internal callers (flow_feature_graph, flow_temporal_feature_graph,
  pl.flow_heatmap) pass per_cell=False to avoid unnecessary compute
- Fix per_cell_scores -> per_cell_alignment bug in generate_12c_flow.py
- Update shape assertions in test_flow_jacobian.py and test_chunk1_senior.py
EOF
)"
```

---

### Task 5: Update registry files (tools_schema.py, types_index.py)

**Files:**
- Modify: `src/peach/_core/tools_schema.py:1430-1475`
- Modify: `src/peach/_core/types_index.py:417-437,986-1004`

- [ ] **Step 1: Update tools_schema.py for flow_gene_alignment**

In the `tl.flow_gene_alignment` ToolSchema (line 1447), change:
```python
            Parameter("per_cell", ParamType.BOOLEAN, "Compute per-cell per-gene alignment scores", default=False),
```
to:
```python
            Parameter("per_cell", ParamType.BOOLEAN, "Compute per-cell per-gene alignment scores (top n_top_features genes)", default=True),
            Parameter("n_top_features", ParamType.INTEGER, "Max genes in per-cell matrix (by |alignment_score|)", default=2500),
            Parameter("normalize", ParamType.BOOLEAN, "Normalize loadings to unit norm (cosine-like scores)", default=True),
```

Update `returns_description` (line 1451-1454):
```python
        returns_description="alignment_scores [n_genes], gene_names, top_aligned, top_opposed, t, "
        "velocity_mode ('displacement' or 'instantaneous'), "
        "alignment_pvalues (optional), alignment_pvalues_fdr (optional), "
        "per_cell_alignment [n_source, n_top_feat] (if per_cell=True), "
        "per_cell_gene_names, per_cell_gene_indices (if per_cell=True)",
```

- [ ] **Step 2: Update tools_schema.py for flow_jacobian**

In the `tl.flow_jacobian` ToolSchema (after line 1469), add:
```python
            Parameter("per_cell_features", ParamType.BOOLEAN, "Compute per-cell feature expansion for top genes", default=True),
            Parameter("n_top_features", ParamType.INTEGER, "Max genes in per-cell expansion matrix", default=2500),
```

Update `returns_description` (line 1472):
```python
        returns_description="jacobian_det [n_points], mean_jacobian [dim, dim], feature_expansion [n_genes], t, "
        "per_cell_expansion [n_points, n_top_feat] (if per_cell_features=True), "
        "per_cell_expansion_gene_names, per_cell_expansion_gene_indices (if per_cell_features=True)",
```

- [ ] **Step 3: Update types_index.py return type docs**

At line 425, change:
```python
            "per_cell_alignment [n_cells, n_genes] (if per_cell=True)",
```
to:
```python
            "per_cell_alignment [n_source, n_top_feat] (if per_cell=True, default)",
            "per_cell_gene_names [n_top_feat] (if per_cell=True)",
            "per_cell_gene_indices [n_top_feat] (if per_cell=True)",
```

At line 436, after `"t (evaluation time)"`, add:
```python
            "per_cell_expansion [n_points, n_top_feat] (if per_cell_features=True, default)",
            "per_cell_expansion_gene_names [n_top_feat] (if per_cell_features=True)",
            "per_cell_expansion_gene_indices [n_top_feat] (if per_cell_features=True)",
```

- [ ] **Step 4: Update types_index.py FUNCTION_PARAMS**

At line 993, change:
```python
        "per_cell": ("bool", False),  # per-cell per-gene alignment scores
```
to:
```python
        "per_cell": ("bool", True),  # per-cell per-gene alignment scores (top n_top_features)
        "n_top_features": ("int", 2500),  # max genes in per-cell matrix
        "normalize": ("bool", True),  # normalize loadings to unit norm
```

At line 1003, after `"aggregate": ("str", "mean"),`, add:
```python
        "per_cell_features": ("bool", True),  # per-cell feature expansion
        "n_top_features": ("int", 2500),  # max genes in per-cell expansion
```

- [ ] **Step 5: Commit**

```bash
git add src/peach/_core/tools_schema.py src/peach/_core/types_index.py
git commit -m "$(cat <<'EOF'
Update registries for per-cell flow feature changes

- tools_schema.py: new params (n_top_features, normalize, per_cell_features),
  updated defaults (per_cell=True), updated return type docs
- types_index.py: per_cell_alignment shape [n_source, n_top_feat],
  new return keys (per_cell_gene_names/indices, per_cell_expansion*),
  FUNCTION_PARAMS updated for both functions
EOF
)"
```

---

### Task 6: Stress test — mathematical validation

**Files:**
- Create: `tests/test_statistical/test_flow_alignment.py`

- [ ] **Step 1: Write PCA reconstruction test**

```python
"""Stress tests for flow gene alignment: mathematical and biological validation."""

import numpy as np
import pytest


def _make_alignment_fixture(n_cells=200, n_genes=100, n_pcs=20, seed=42):
    """Create synthetic AnnData with trained flow for alignment testing."""
    import anndata as ad
    from peach.tl.flow import flow_within

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
    adata = ad.AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    # Generate PCA via actual decomposition for realistic loadings
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_pcs)
    pca_coords = pca.fit_transform(X).astype(np.float32)
    adata.obsm["X_pca"] = pca_coords
    adata.varm["PCs"] = pca.components_.T.astype(np.float32)  # [n_genes, n_pcs]
    adata.uns["pca_mean"] = pca.mean_.astype(np.float32)

    adata.obs["condition"] = (
        ["source"] * (n_cells // 2) + ["target"] * (n_cells - n_cells // 2)
    )

    flow_result = flow_within(
        adata,
        source={"condition": "source"},
        target={"condition": "target"},
        n_epochs=200,
        hidden_dims=(64, 64),
        return_model=True,
        random_state=seed,
    )
    return adata, flow_result


class TestPCAReconstructionAlignment:
    """Validate that alignment scores predict gene expression change via PCA."""

    def test_raw_alignment_equals_reconstructed_displacement(self):
        """With normalize=False, alignment score = mean gene expression change."""
        from peach.tl.flow import flow_gene_alignment

        adata, flow_result = _make_alignment_fixture()
        loadings = adata.varm["PCs"]  # [n_genes, n_pcs]
        source_pca = adata.obsm["X_pca"][flow_result["source_mask"]]

        # Alignment scores (raw, no normalization)
        result = flow_gene_alignment(
            adata, flow_result, normalize=False, per_cell=False
        )
        alignment_scores = result["alignment_scores"]

        # PCA-reconstructed gene expression change
        delta_pca = flow_result["transported"] - source_pca  # [n_source, n_pcs]
        mean_delta_pca = delta_pca.mean(axis=0)  # [n_pcs]
        mean_delta_expr = loadings @ mean_delta_pca  # [n_genes]

        # These should be identical (both = loadings @ mean_velocity)
        np.testing.assert_allclose(
            alignment_scores, mean_delta_expr, atol=1e-5,
            err_msg="Raw alignment scores should equal PCA-reconstructed expression change",
        )

    def test_normalized_alignment_ranking_matches_reconstruction(self):
        """With normalize=True, ranking should match reconstructed displacement."""
        from peach.tl.flow import flow_gene_alignment
        from scipy.stats import spearmanr

        adata, flow_result = _make_alignment_fixture()
        loadings = adata.varm["PCs"]
        source_pca = adata.obsm["X_pca"][flow_result["source_mask"]]

        result = flow_gene_alignment(
            adata, flow_result, normalize=True, per_cell=False
        )
        alignment_scores = result["alignment_scores"]

        delta_pca = flow_result["transported"] - source_pca
        mean_delta_expr = loadings @ delta_pca.mean(axis=0)

        # Filter to non-trivial genes (avoid near-zero scores dominating rank)
        threshold = np.percentile(np.abs(mean_delta_expr), 25)
        mask = np.abs(mean_delta_expr) > threshold

        rho, _ = spearmanr(alignment_scores[mask], mean_delta_expr[mask])
        assert rho > 0.99, (
            f"Normalized alignment ranking should match reconstruction: Spearman={rho:.4f}"
        )


class TestNormalizationConsistency:
    """Verify aggregated and per-cell paths agree after normalization fix."""

    def test_aggregated_and_percell_mean_agree(self):
        """Mean of per-cell alignment should rank-match aggregated scores."""
        from peach.tl.flow import flow_gene_alignment
        from scipy.stats import spearmanr

        adata, flow_result = _make_alignment_fixture(n_genes=50)

        result = flow_gene_alignment(
            adata, flow_result, normalize=True, per_cell=True,
            n_top_features=50,  # all genes
        )

        agg_scores = result["alignment_scores"]
        pc_mean = result["per_cell_alignment"].mean(axis=0)  # [n_top_feat]
        pc_gene_idx = result["per_cell_gene_indices"]

        # Compare rankings on the genes present in per_cell
        rho, _ = spearmanr(agg_scores[pc_gene_idx], pc_mean)
        assert rho > 0.95, (
            f"Aggregated and per-cell-mean rankings should agree: Spearman={rho:.4f}"
        )

    def test_normalize_false_matches_legacy(self):
        """normalize=False should produce identical scores to the old code path."""
        from peach.tl.flow import flow_gene_alignment

        adata, flow_result = _make_alignment_fixture()
        loadings = adata.varm["PCs"]
        source_pca = adata.obsm["X_pca"][flow_result["source_mask"]]
        mean_velocity = (flow_result["transported"] - source_pca).mean(axis=0)

        result = flow_gene_alignment(
            adata, flow_result, normalize=False, per_cell=False
        )
        expected = loadings[:, :len(mean_velocity)] @ mean_velocity
        np.testing.assert_allclose(
            result["alignment_scores"], expected, atol=1e-6,
            err_msg="normalize=False should match raw loadings @ velocity",
        )
```

- [ ] **Step 2: Run to verify tests pass**

Run: `conda run -n archetype python -m pytest tests/test_statistical/test_flow_alignment.py -v -x 2>&1 | tail -30`

Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_statistical/test_flow_alignment.py
git commit -m "$(cat <<'EOF'
Add stress tests: PCA reconstruction validates alignment math

- test_raw_alignment_equals_reconstructed_displacement: proves
  alignment_score = loadings @ mean_velocity = mean(gene_expr_change)
- test_normalized_alignment_ranking_matches_reconstruction: Spearman > 0.99
- test_aggregated_and_percell_mean_agree: normalization consistency
- test_normalize_false_matches_legacy: backward compat
EOF
)"
```

---

### Task 7: Stress test — PCA truncation sensitivity

**Files:**
- Modify: `tests/test_statistical/test_flow_alignment.py`

- [ ] **Step 1: Write PCA truncation test**

Append to `test_flow_alignment.py`:

```python
class TestPCATruncationSensitivity:
    """Verify alignment ranking stability across PCA dimensionalities."""

    def test_top_genes_stable_across_pc_counts(self):
        """Top aligned genes should be largely stable between 30 and 50 PCs."""
        import anndata as ad
        from sklearn.decomposition import PCA
        from peach.tl.flow import flow_within, flow_gene_alignment
        from scipy.stats import spearmanr

        rng = np.random.default_rng(42)
        n_cells, n_genes = 300, 200
        X = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
        adata = ad.AnnData(X)
        adata.var_names = [f"gene_{i}" for i in range(n_genes)]
        adata.obs["condition"] = (
            ["source"] * (n_cells // 2) + ["target"] * (n_cells - n_cells // 2)
        )

        pc_counts = [10, 30, 50]
        scores_by_npc = {}

        for n_pcs in pc_counts:
            pca = PCA(n_components=n_pcs)
            pca_coords = pca.fit_transform(X).astype(np.float32)
            adata.obsm["X_pca"] = pca_coords
            adata.varm["PCs"] = pca.components_.T.astype(np.float32)

            flow_result = flow_within(
                adata,
                source={"condition": "source"},
                target={"condition": "target"},
                n_epochs=150,
                hidden_dims=(64, 64),
                return_model=True,
                random_state=42,
            )
            result = flow_gene_alignment(
                adata, flow_result, normalize=True, per_cell=False
            )
            scores_by_npc[n_pcs] = result["alignment_scores"]

        # Compare 30-PC vs 50-PC: top 100 genes by |score| in 50-PC version
        scores_50 = scores_by_npc[50]
        scores_30 = scores_by_npc[30]
        top100_idx = np.argsort(np.abs(scores_50))[-100:]

        rho, _ = spearmanr(scores_50[top100_idx], scores_30[top100_idx])
        assert rho > 0.85, (
            f"Top 100 gene rankings between 30-PC and 50-PC should be stable: "
            f"Spearman={rho:.4f}"
        )

        # 10-PC vs 50-PC can be lower but should still be positive
        scores_10 = scores_by_npc[10]
        rho_10, _ = spearmanr(scores_50[top100_idx], scores_10[top100_idx])
        assert rho_10 > 0.5, (
            f"10-PC vs 50-PC should have moderate concordance: Spearman={rho_10:.4f}"
        )
```

- [ ] **Step 2: Run test**

Run: `conda run -n archetype python -m pytest tests/test_statistical/test_flow_alignment.py::TestPCATruncationSensitivity -v -x`

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_statistical/test_flow_alignment.py
git commit -m "Add PCA truncation sensitivity stress test (Spearman > 0.85 for 30 vs 50 PCs)"
```

---

### Task 8: Stress test — biological validation (HSC CMP->Mono)

**Files:**
- Modify: `tests/test_statistical/test_flow_alignment.py`

**Prerequisites:** Requires HSC data at `~/Desktop/peach/data/hsc_10k.h5ad`. Test is marked with `@pytest.mark.slow` since it loads real data and trains a flow model.

- [ ] **Step 1: Write biological validation test**

Append to `test_flow_alignment.py`:

```python
@pytest.mark.slow
class TestBiologicalValidationHSC:
    """Validate alignment on real HSC CMP->Mono transition.

    Requires hsc_10k.h5ad in ~/Desktop/peach/data/.
    Run with: pytest -m slow
    """

    @pytest.fixture(scope="class")
    def hsc_flow(self):
        """Load HSC data, prepare, train archetype model, train flow."""
        import os
        import peach as pc

        data_path = os.path.expanduser("~/Desktop/peach/data/hsc_10k.h5ad")
        if not os.path.exists(data_path):
            pytest.skip(f"HSC data not found at {data_path}")

        adata = pc.pp.load_data(data_path)

        # Ensure PCA exists
        if "X_pca" not in adata.obsm:
            import scanpy as sc
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=2000)
            sc.pp.pca(adata, n_comps=50)

        # Check for cell type annotations
        celltype_col = None
        for col in ["cell_type", "celltype", "CellType", "label"]:
            if col in adata.obs.columns:
                celltype_col = col
                break
        if celltype_col is None:
            pytest.skip("No cell type column found in HSC data")

        cell_types = adata.obs[celltype_col].unique()
        has_cmp = any("CMP" in str(ct) for ct in cell_types)
        has_mono = any("Mono" in str(ct) for ct in cell_types)
        if not (has_cmp and has_mono):
            pytest.skip(f"Need CMP and Mono cell types, found: {cell_types}")

        # Train flow CMP -> Mono
        flow_result = pc.tl.flow_within(
            adata,
            source={celltype_col: [ct for ct in cell_types if "CMP" in str(ct)][0]},
            target={celltype_col: [ct for ct in cell_types if "Mono" in str(ct)][0]},
            n_epochs=500,
            hidden_dims=(128, 128, 128),
            return_model=True,
            random_state=42,
        )

        return adata, flow_result

    def test_myeloid_tfs_top_aligned(self, hsc_flow):
        """Canonical myeloid TFs should appear in top 100 aligned genes."""
        from peach.tl.flow import flow_gene_alignment

        adata, flow_result = hsc_flow
        result = flow_gene_alignment(
            adata, flow_result, normalize=True, per_cell=False, n_top=100
        )

        top_aligned = set(result["top_aligned"])
        myeloid_tfs = {"SPI1", "CEBPA", "CEBPB", "CSF1R", "IRF8"}
        present_myeloid = myeloid_tfs & set(adata.var_names)
        found = present_myeloid & top_aligned

        # Soft assertion: document what we found
        print(f"Myeloid TFs in dataset: {present_myeloid}")
        print(f"Myeloid TFs in top 100 aligned: {found}")
        print(f"Top 10 aligned: {result['top_aligned'][:10]}")

        assert len(found) >= 2, (
            f"Expected >= 2 myeloid TFs in top 100 aligned, found {len(found)}: {found}. "
            f"Present in data: {present_myeloid}. Top 10: {result['top_aligned'][:10]}"
        )

    def test_erythroid_markers_top_opposed(self, hsc_flow):
        """Erythroid markers should appear in top 100 opposed genes."""
        from peach.tl.flow import flow_gene_alignment

        adata, flow_result = hsc_flow
        result = flow_gene_alignment(
            adata, flow_result, normalize=True, per_cell=False, n_top=100
        )

        top_opposed = set(result["top_opposed"])
        erythroid_markers = {"GATA1", "KLF1", "EPOR", "HBB", "HBA1"}
        present_ery = erythroid_markers & set(adata.var_names)
        found = present_ery & top_opposed

        print(f"Erythroid markers in dataset: {present_ery}")
        print(f"Erythroid markers in top 100 opposed: {found}")
        print(f"Top 10 opposed: {result['top_opposed'][:10]}")

        assert len(found) >= 1, (
            f"Expected >= 1 erythroid marker in top 100 opposed, found {len(found)}: {found}. "
            f"Present in data: {present_ery}. Top 10: {result['top_opposed'][:10]}"
        )

    def test_jacobian_expansion_concordance(self, hsc_flow):
        """Jacobian expansion scores should positively correlate with alignment."""
        from peach.tl.flow import flow_gene_alignment, flow_jacobian
        from scipy.stats import spearmanr

        adata, flow_result = hsc_flow
        model = flow_result["model"]

        align_result = flow_gene_alignment(
            adata, flow_result, normalize=True, per_cell=False
        )
        jac_result = flow_jacobian(
            adata, flow_result, model, per_cell_features=False
        )

        alignment = align_result["alignment_scores"]
        expansion = jac_result["feature_expansion"]

        if len(expansion) == 0:
            pytest.skip("No feature expansion (PCA loadings missing)")

        # Both should agree on direction
        rho, pval = spearmanr(alignment, expansion)
        print(f"Alignment vs expansion Spearman: rho={rho:.4f}, p={pval:.2e}")

        assert rho > 0, (
            f"Alignment and Jacobian expansion should be positively correlated: "
            f"Spearman={rho:.4f}, p={pval:.2e}"
        )
```

- [ ] **Step 2: Run biological tests (slow)**

Run: `conda run -n archetype python -m pytest tests/test_statistical/test_flow_alignment.py::TestBiologicalValidationHSC -v -m slow --timeout=300 2>&1 | tail -30`

Expected: PASS (soft criteria — document results even on failure)

- [ ] **Step 3: Commit**

```bash
git add tests/test_statistical/test_flow_alignment.py
git commit -m "$(cat <<'EOF'
Add biological validation: HSC CMP->Mono alignment stress test

- Myeloid TFs (SPI1, CEBPA, etc.) should be top-aligned
- Erythroid markers (GATA1, KLF1) should be top-opposed
- Jacobian expansion should positively correlate with alignment scores
- Marked @pytest.mark.slow, requires hsc_10k.h5ad
EOF
)"
```

---

### Task 9: Per-cell Jacobian expansion test

**Files:**
- Modify: `tests/test_statistical/test_flow_jacobian.py`

- [ ] **Step 1: Add per-cell expansion tests**

Append to `test_flow_jacobian.py`:

```python
def test_per_cell_jacobian_expansion_shape():
    """flow_jacobian with per_cell_features=True returns correct shape."""
    import peach as pc

    adata, flow_result = _make_flow_fixture(n_genes=50, return_model=True)
    model = flow_result["model"]

    result = pc.tl.flow_jacobian(adata, flow_result, model, per_cell_features=True)

    assert "per_cell_expansion" in result
    assert "per_cell_expansion_gene_names" in result
    assert "per_cell_expansion_gene_indices" in result

    n_source = flow_result["source_mask"].sum()
    n_top_feat = min(2500, 50)  # 50 genes in fixture
    assert result["per_cell_expansion"].shape == (n_source, n_top_feat)
    assert len(result["per_cell_expansion_gene_names"]) == n_top_feat
    assert np.all(np.isfinite(result["per_cell_expansion"]))


def test_per_cell_expansion_mean_matches_aggregated():
    """Mean of per-cell expansion should approximate aggregated feature_expansion."""
    import peach as pc

    adata, flow_result = _make_flow_fixture(n_genes=30, return_model=True)
    model = flow_result["model"]

    result = pc.tl.flow_jacobian(
        adata, flow_result, model,
        per_cell_features=True, n_top_features=30,  # all genes
    )

    # Aggregated feature_expansion uses mean Jacobian
    agg = result["feature_expansion"]
    # Per-cell mean should be close (both use same loadings normalization)
    pc_mean = result["per_cell_expansion"].mean(axis=0)
    pc_idx = result["per_cell_expansion_gene_indices"]

    # Not exact (mean of quadratic forms != quadratic form of mean),
    # but should be correlated
    from scipy.stats import spearmanr
    rho, _ = spearmanr(agg[pc_idx], pc_mean)
    assert rho > 0.8, (
        f"Per-cell expansion mean should correlate with aggregated: Spearman={rho:.4f}"
    )


def test_per_cell_expansion_disabled():
    """per_cell_features=False should not include per-cell keys."""
    import peach as pc

    adata, flow_result = _make_flow_fixture(return_model=True)
    model = flow_result["model"]

    result = pc.tl.flow_jacobian(
        adata, flow_result, model, per_cell_features=False
    )
    assert "per_cell_expansion" not in result
    assert "per_cell_expansion_gene_names" not in result
    # Aggregated feature_expansion should still be present
    assert "feature_expansion" in result


def test_per_cell_expansion_invariant_to_loading_scale():
    """Per-cell expansion should not change when loadings are scaled."""
    import peach as pc

    adata, flow_result = _make_flow_fixture(n_genes=30, return_model=True)
    model = flow_result["model"]

    result1 = pc.tl.flow_jacobian(
        adata, flow_result, model,
        per_cell_features=True, n_top_features=30,
    )

    adata2 = adata.copy()
    adata2.varm["PCs"] = adata.varm["PCs"] * 10.0
    result2 = pc.tl.flow_jacobian(
        adata2, flow_result, model,
        per_cell_features=True, n_top_features=30,
    )

    np.testing.assert_allclose(
        result1["per_cell_expansion"],
        result2["per_cell_expansion"],
        atol=1e-5,
        err_msg="Per-cell expansion should be invariant to loading scale",
    )
```

- [ ] **Step 2: Run tests**

Run: `conda run -n archetype python -m pytest tests/test_statistical/test_flow_jacobian.py -v -x -k "per_cell_expansion or per_cell_jacobian" 2>&1 | tail -20`

Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_statistical/test_flow_jacobian.py
git commit -m "$(cat <<'EOF'
Add per-cell Jacobian expansion tests

- Shape validation (n_source x n_top_feat)
- Mean of per-cell approx matches aggregated (Spearman > 0.8)
- per_cell_features=False excludes per-cell keys
- Scale invariance to loading magnitude
EOF
)"
```

---

### Task 10: Final integration test

**Files:** None new — runs existing test suite

- [ ] **Step 1: Run full flow test suite**

Run: `conda run -n archetype python -m pytest tests/test_statistical/test_flow_jacobian.py tests/test_statistical/test_flow_alignment.py tests/test_statistical/test_chunk1_senior.py -v --timeout=120 2>&1 | tail -40`

Expected: ALL PASS

- [ ] **Step 2: Verify flow_feature_graph and flow_temporal_feature_graph still work**

Run: `conda run -n archetype python -m pytest tests/test_statistical/test_flow_jacobian.py -v -k "feature_graph or temporal_graph or centrality" --timeout=120 2>&1 | tail -20`

Expected: ALL PASS (internal callers now pass `per_cell=False`, should be faster)

- [ ] **Step 3: Spot check: default per_cell=True behavior**

Run a quick Python check that the new defaults work end-to-end:

```bash
conda run -n archetype python -c "
import numpy as np, anndata as ad
from peach.tl.flow import flow_within, flow_gene_alignment, flow_jacobian

rng = np.random.default_rng(42)
n, g, d = 100, 50, 10
adata = ad.AnnData(rng.standard_normal((n, g)).astype('f'))
adata.obsm['X_pca'] = rng.standard_normal((n, d)).astype('f')
adata.varm['PCs'] = rng.standard_normal((g, d)).astype('f')
adata.obs['grp'] = ['A']*50 + ['B']*50

fr = flow_within(adata, {'grp':'A'}, {'grp':'B'}, n_epochs=30, hidden_dims=(32,32), return_model=True)

# Gene alignment: per_cell=True by default now
ga = flow_gene_alignment(adata, fr)
assert 'per_cell_alignment' in ga, 'per_cell should be default'
assert 'per_cell_gene_names' in ga
print(f'gene_alignment per_cell shape: {ga[\"per_cell_alignment\"].shape}')

# Jacobian: per_cell_features=True by default now
jac = flow_jacobian(adata, fr, fr['model'])
assert 'per_cell_expansion' in jac, 'per_cell_features should be default'
print(f'jacobian per_cell_expansion shape: {jac[\"per_cell_expansion\"].shape}')
print('All default behaviors verified.')
"
```

Expected: prints shapes and "All default behaviors verified."

- [ ] **Step 4: Final commit (if any cleanup needed)**

If all tests pass and no cleanup needed, no commit required. Otherwise:

```bash
git add -u
git commit -m "Fix integration issues from per-cell flow features"
```

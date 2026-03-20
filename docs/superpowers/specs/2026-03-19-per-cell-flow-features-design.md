# Per-Cell Flow Features: Gene Alignment + Jacobian Expansion

**Date**: 2026-03-19
**Status**: Approved
**Scope**: Two targeted changes to `flow_gene_alignment` and `flow_jacobian` in `src/peach/tl/flow.py`, plus stress tests

---

## Motivation

The v0.5.0 blinded analysis design requires per-cell flow features for post-hoc R/NR stratification. Currently:
- `flow_gene_alignment` computes per-cell scores only when `per_cell=True` (default False)
- `flow_jacobian` has no per-cell feature expansion at all — only aggregated from mean Jacobian

Both need per-cell outputs as the default, with a memory cap to prevent OOM on large datasets.

Additionally, the aggregated gene alignment scores have a normalization inconsistency: per-cell path normalizes loadings to unit norm, but aggregated path uses raw loadings. This inflates scores for high-variance genes in the aggregated case.

---

## Change 1: `flow_gene_alignment` — per-cell default with top-2500 cap

### Signature

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
    n_top_features: int = 2500,      # NEW — caps per_cell matrix
    normalize: bool = True,          # NEW — unified normalization
    random_state: int = 42,
) -> dict:
```

### Behavior

1. Aggregated `alignment_scores [n_genes]` always computed for ALL genes
2. When `normalize=True` (default), both aggregated and per-cell paths normalize gene loadings to unit norm. Scores become cosine-like (direction-only). When `normalize=False`, raw dot product (magnitude-weighted)
3. Per-cell matrix `per_cell_alignment [n_source, min(n_top_features, n_genes)]` computed for top genes by `|alignment_scores|`
4. Gene identity tracked via `per_cell_gene_names` and `per_cell_gene_indices`
5. When `n_permutations > 0` and `normalize=True`, the permutation null also uses normalized loadings (so observed and null statistics are computed on the same scale)

### New return keys

```python
"per_cell_alignment":    np.ndarray [n_source, n_top_feat]  # when per_cell=True
"per_cell_gene_names":   list[str]                          # which genes, in order
"per_cell_gene_indices": np.ndarray [n_top_feat]            # indices into adata.var_names
```

### Normalization fix

Current code (per-cell path, lines 319-325):
```python
vel_norm = velocity_per_cell / (np.linalg.norm(...) + 1e-10)
load_norm = loadings_trimmed / (np.linalg.norm(...) + 1e-10)
per_cell_alignment = vel_norm @ load_norm.T
```

Current code (aggregated path, line 298):
```python
alignment_scores = loadings_trimmed @ mean_velocity  # NO normalization
```

Fix: when `normalize=True`, the aggregated path also normalizes:
```python
loading_norms = np.linalg.norm(loadings_trimmed, axis=1, keepdims=True)
loadings_normalized = loadings_trimmed / np.maximum(loading_norms, 1e-10)
vel_norm = mean_velocity / (np.linalg.norm(mean_velocity) + 1e-10)
alignment_scores = loadings_normalized @ vel_norm
```

When `normalize=False`, use raw loadings and raw velocity (current behavior).

The permutation null (lines 330-333) must also respect the `normalize` flag:
```python
if normalize:
    perm_loadings_norm = perm_loadings / np.maximum(
        np.linalg.norm(perm_loadings, axis=1, keepdims=True), 1e-10
    )
    null_scores[i] = perm_loadings_norm @ vel_norm  # normalized velocity
else:
    null_scores[i] = perm_loadings @ mean_velocity  # raw (current behavior)
```

### Internal callers must pass `per_cell=False`

Three internal call sites use `flow_gene_alignment` only for gene selection (top genes by alignment score). They don't need per-cell output and should not pay the memory/compute cost:

- `flow_feature_graph()` at line 711: `alignment = flow_gene_alignment(adata, flow_result, per_cell=False)`
- `flow_temporal_feature_graph()` at line 892: `alignment = flow_gene_alignment(adata, flow_result, per_cell=False)`
- `flow_heatmap()` in `src/peach/pl/flow.py` at line 581: `align = flow_gene_alignment(adata, flow_result, n_permutations=0, per_cell=False)`

---

## Change 2: `flow_jacobian` — per-cell feature expansion

### Signature

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
    n_top_features: int = 2500,       # NEW — caps per_cell matrix
) -> dict:
```

### Behavior

1. Existing aggregated `feature_expansion [n_genes]` unchanged (from mean/median Jacobian)
2. When `per_cell_features=True`: compute per-cell gene-space Jacobian quadratic form for top 2500 genes by `|feature_expansion|`
3. Computation: `einsum('gi,cij,gj->cg', L_norm, jac, L_norm)` where `L_norm` is unit-normalized PCA loadings for the top genes, `jac` is per-cell Jacobian `[n_points, dim, dim]`
4. Cap: `n_top_feat = min(n_top_features, n_genes)`
5. When PCA loadings are absent from `adata.varm`, per-cell features are silently skipped (matching the existing aggregated fallback where `feature_expansion = np.zeros(0)`)

### New return keys (consistent naming with Change 1)

```python
"per_cell_expansion":        np.ndarray [n_points, n_top_feat]
"per_cell_expansion_gene_names":   list[str]
"per_cell_expansion_gene_indices": np.ndarray [n_top_feat]
```

### Memory estimate

At 10k eval points x 2500 genes x 8 bytes = ~200 MB. Acceptable. The intermediate Jacobian tensor `[n_points, dim, dim]` at 10k x 50 x 50 x 8 = ~200 MB is already in memory during computation.

### Known limitation

Top gene selection is based on aggregated (mean Jacobian) feature expansion scores. A gene with near-zero mean expansion but high inter-cell variance (heterogeneous expansion) will be excluded. This is acceptable for the initial implementation — a future `selection_method` parameter could add variance-based selection.

---

## Bug fixes (opportunistic)

### Fix `per_cell_scores` key bug in generate_12c_flow.py

Line 464 references `percell_align["per_cell_scores"]` but the actual key is `per_cell_alignment`. Fix to match.

---

## Existing test updates

The following test assertions reference the old `[n_source, n_genes]` shape and must be updated to `[n_source, min(n_top_features, n_genes)]`:

| File | Line | Current assertion | New assertion |
|------|------|-------------------|---------------|
| `tests/test_statistical/test_flow_jacobian.py` | 228 | `shape == (50, n_genes)` | `shape[0] == 50` and `shape[1] <= n_genes` |
| `tests/test_statistical/test_chunk1_senior.py` | 331 | `shape == (n_source, n_genes)` | `shape == (n_source, min(2500, n_genes))` |
| `tests/test_statistical/test_chunk1_senior.py` | 342 | `shape == (n_source, n_genes)` | `shape == (n_source, min(2500, n_genes))` |
| `tests/test_statistical/test_chunk1_senior.py` | 351-353 | value bounds on `per_cell_alignment` | unchanged (cosine bounds still hold) |
| `tests/test_statistical/test_chunk1_senior.py` | 361-367 | displacement vs instantaneous differ | unchanged (still valid) |

Also update `types_index.py` line 425 and `tools_schema.py` line 1454 to document new shape `[n_cells, n_top_feat]` and new return keys.

---

## Stress Tests

### A. Mathematical validation

**Test: PCA-reconstructed expression change matches alignment scores**
- Train CMP->Mono flow, compute alignment scores
- For source cells, compute displacement: `delta_pca = transported - source_pca`
- Reconstruct gene-space displacement: `delta_expr = delta_pca @ loadings.T` → `[n_source, n_genes]`
- Mean gene expression change: `mean_delta_expr = delta_expr.mean(axis=0)` → `[n_genes]`
- With `normalize=False`, alignment scores should equal `mean_delta_expr` exactly (both are `loadings @ mean_velocity`)
- With `normalize=True`, alignment score ranking should match `mean_delta_expr` ranking (Spearman > 0.99 on non-zero genes)

**Test: PCA truncation sensitivity**
- Compute alignment scores using 10, 30, 50 PCs on HSC CMP->Mono flow
- Compare rank stability: Spearman correlation on top 100 genes between each pair of PC counts
- Pass criterion: Spearman rho > 0.85 between 30-PC and 50-PC versions

**Test: normalization consistency**
- Verify that with `normalize=True`, aggregated score ranking and mean(per_cell) ranking agree (Spearman > 0.99)
- Verify that with `normalize=False`, aggregated scores equal `loadings @ mean_velocity` exactly

### B. Biological validation (HSC CMP->Mono)

**Test: known myeloid TF alignment**
- Train flow on HSC data: CMP -> Monocyte
- Top-aligned genes should include canonical myeloid TFs: SPI1 (PU.1), CEBPA, CEBPB, CSF1R, IRF8
- Top-opposed genes should include erythroid/other lineage markers: GATA1, KLF1, EPOR (if present in var_names)
- Pass criterion: at least 2 of {SPI1, CEBPA, CEBPB, CSF1R, IRF8} in top 100 aligned; at least 1 of {GATA1, KLF1} in top 100 opposed
- Note: soft criteria — test documents results rather than hard-failing, since gene presence depends on dataset filtering

**Test: Jacobian expansion vs alignment concordance**
- Top expanding genes (positive feature_expansion) should overlap substantially with top aligned genes
- Spearman correlation between alignment scores and feature expansion scores should be positive and significant
- Validates that velocity dot loading and Jacobian quadratic form agree on directionality

---

## Files modified

| File | Changes |
|------|---------|
| `src/peach/tl/flow.py` | `flow_gene_alignment`: new defaults, normalize param, top-N per-cell. `flow_jacobian`: per_cell_features param. Internal callers: add `per_cell=False` |
| `src/peach/pl/flow.py` | `flow_heatmap`: add `per_cell=False` to internal `flow_gene_alignment` call |
| `src/peach/_core/tools_schema.py` | Add `n_top_features`, `normalize` params to alignment schema; `per_cell_features`, `n_top_features` to Jacobian schema; change `per_cell` default from `False` to `True` (line ~1447); update return type docs |
| `src/peach/_core/types_index.py` | Update per_cell_alignment shape doc, add new return keys. Update `FUNCTION_PARAMS` entries: `flow_gene_alignment` per_cell default `False`→`True`, add `n_top_features`/`normalize` params; `flow_jacobian` add `per_cell_features`/`n_top_features` params |
| `scripts/generate_12c_flow.py` | Fix `per_cell_scores` → `per_cell_alignment` bug (line 464) |
| `tests/test_statistical/test_flow_jacobian.py` | Update shape assertions, add stress tests |
| `tests/test_statistical/test_chunk1_senior.py` | Update shape assertions |
| `tests/test_statistical/test_flow_alignment.py` | NEW — alignment-specific stress tests (math + bio validation) |

## Files NOT modified

- `src/peach/_core/utils/flow_matching.py` — no changes to FlowModel internals
- Plotting functions other than `flow_heatmap` — they consume result dicts, no signature changes needed

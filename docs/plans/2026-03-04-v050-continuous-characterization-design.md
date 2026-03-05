# PEACH v0.5.0 Design: Continuous Archetype Characterization + Cross-Condition Flow Matching

**Status**: Design approved 2026-03-04
**Branch from**: `main` (commit 90e98ba)
**Scope**: Simplex regression, pattern classification, flipped regression, GMM decomposition, flow matching, ternary visualization, spatial pair enrichment

---

## Problem Statement

Current archetypal analysis in the field uses binary archetype membership + 1-vs-all Wilcoxon rank-sum tests. This discards the continuous geometry (barycentric weights, distances) that PEACH computes. Features that associate with multiple archetypes can't be distinguished from bimodal patterns. There's no principled way to compare conditions within or between archetype fits. And the combinatorial post-hoc pattern tests (exclusivity, specialization, tradeoff) reconstruct structure already implicit in the continuous coordinates.

v0.5.0 replaces this with:
- Principled regression on compositional archetype weights (Scheffe polynomials)
- Automatic feature pattern classification from regression coefficients
- Flipped regression to identify geneset interactions driving archetypal specialization
- Sub-archetype population structure via GMM in ILR-transformed weight space
- Cross-condition flow matching in PCA space with gene alignment and Jacobian analysis
- Ternary facet visualization for simplex surfaces

---

## Design Decisions (Locked)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Flow geometry | PCA/latent space, Euclidean only | Proven in FRTNBC POC, no simplex constraint headaches |
| Flow dependency | fb `flow_matching` library | Citable in papers, delegates ODE/path interpolation to tested infrastructure |
| Flow API shape | Two functions: `flow_within()` + `flow_between()` | Explicit intent, no type-dispatch magic |
| `flow_between()` output | Concatenated AnnData with condition labels in `.obs` | Follows scverse conventions |
| Simplex regression degrees | Run both 1st and 2nd order, compare in interpretation layer | Richer than BIC-selecting one; interaction terms are biologically informative |
| Residual storage | Full residuals default-on (`store_residuals=True`) | Memory cost acceptable; enables downstream diagnostics |
| Result storage | Centralized convention in `types_index.py`, `peach_` prefix in `uns`/`obsm` | Consistent, discoverable |
| Level sets (Module 1) | Dropped — absorbed by simplex regression interaction terms | 2nd-order Scheffe terms detect ridges/blend zones; separate binned profiles add complexity without insight |
| Adaptive regression (Module 3) | Struck | High complexity, may not add beyond 2nd-degree Scheffe |
| Ternary facet plots | In scope | Natural visualization for simplex regression surfaces |
| GMM stability | Multi-initialization, keep only stable components | Equivalent to bootstrap for mixture models; BIC overestimates components at large n |
| Spatial front alignment | Out (v0.6) | Edge detection problem requiring more research |
| Uncertainty quantification | Per-module (see below) | Layered: analytic always, resampling opt-in |

---

## Architecture

### File Layout

```
src/peach/
├── _core/utils/
│   ├── feature_utils.py            # NEW: resolve_features(), storage helpers, weight extraction
│   ├── resampling.py               # NEW: shared permutation/bootstrap infrastructure
│   ├── simplex_regression.py       # NEW: OLS engine, HC3 SEs, Scheffe design matrices
│   ├── pattern_classification.py   # NEW: classify regression coefficients into biological patterns
│   ├── simplex_gmm.py             # NEW: ILR transform, GMM fitting, stability analysis
│   └── flow_matching.py           # NEW: velocity net, train loop, transport, Jacobian
├── tl/
│   ├── feature_regression.py       # NEW: public API for regression modules
│   ├── feature_decomposition.py    # NEW: public API for GMM
│   ├── feature_patterns.py         # NEW: public API for pattern classification + archetype summary
│   ├── flow.py                     # NEW: public API for flow matching
│   └── spatial.py                  # EXTEND: pair enrichment
├── pl/
│   ├── regression.py               # NEW: regression visualizations
│   ├── ternary.py                  # NEW: ternary facet plots
│   ├── decomposition.py            # NEW: GMM visualizations
│   ├── flow.py                     # NEW: flow visualizations
│   └── spatial.py                  # EXTEND: pair plots
```

### Shared Feature Infrastructure

**`_core/utils/feature_utils.py`**

All analysis modules accept the same `(feature_matrix, feature_names)` pattern:

```python
def resolve_features(adata, feature_matrix=None, feature_names=None):
    """
    Returns (np.ndarray[n_cells, n_features], list[str]).

    Resolution:
      feature_matrix=None       -> adata.X (densified if sparse)
      feature_matrix='key'      -> adata.obsm['key']
      feature_matrix=np.ndarray -> use directly
      feature_names=None        -> infer from adata.var_names or generate
    """

def get_archetype_weights(adata, renormalize=True):
    """Extract archetype weights from adata.obsm, assert/enforce sum-to-1."""

def store_result(adata, key, result, domain='uns'):
    """Store result in adata with peach_ prefix."""
```

**Storage convention** (registered in `types_index.py`):

```
adata.uns['peach_simplex_regression']     -> SimplexRegressionResult
adata.uns['peach_driver_regression']      -> DriverRegressionResult
adata.uns['peach_feature_patterns']       -> PatternClassificationResult
adata.uns['peach_gmm']                    -> GMMResult
adata.uns['peach_flow_{name}']            -> FlowResult (per named flow)
adata.obsm['peach_gmm_labels']            -> component assignments (n_cells,)
adata.obsm['peach_residuals']             -> residual matrix (n_cells, n_features)
```

All results also returned from the function call for users who prefer functional style.

### Shared Resampling Infrastructure

**`_core/utils/resampling.py`**

```python
def permutation_test(fit_fn, stat_fn, data, n_permutations=1000, seed=42):
    """
    Generic permutation framework.

    fit_fn(data) -> model
    stat_fn(model) -> scalar test statistic

    Returns: observed_stat, null_distribution, p_value
    """

def bootstrap_ci(fit_fn, stat_fn, data, n_bootstrap=1000, ci_level=0.95, seed=42):
    """
    Generic bootstrap CI framework.

    Returns: point_estimate, ci_lower, ci_upper, bootstrap_distribution
    """
```

Each module provides its own callables. Resampling logic stays DRY.

---

## Module 2: Simplex Regression (Scheffe Polynomials)

### Method

**First degree** (linear on simplex):
```
E[feature_g] = Sigma_k beta_gk * w_k    (no intercept)
```
- beta_gk = expected feature value at archetype vertex k
- Fit via OLS, no intercept. Vectorized across all features simultaneously.

**Second degree** (with interactions):
```
E[feature_g] = Sigma_k beta_gk * w_k + Sigma_{j<k} beta_g,jk * w_j * w_k
```
- beta_g,jk = blend-zone interaction effect for archetype pair (j,k)
- Positive = synergistic blend, negative = antagonistic

Both degrees fit simultaneously. Results include both for comparison in the interpretation layer.

### Implementation Notes

**HC3 standard errors** (default-on): Compute hat matrix diagonal only, never the full hat matrix.
```python
WtW_inv = np.linalg.inv(W.T @ W)            # [p x p], p = K or K + K-choose-2
H_diag = np.sum((W @ WtW_inv) * W, axis=1)  # [n], O(np^2)
# HC3: Var(beta) = (WtW)^-1 (Sigma_i w_i w_i^T * e_i^2 / (1 - h_ii)^2) (WtW)^-1
```

**No-intercept assertion**: Assert `abs(weights.sum(axis=1) - 1.0).max() < 1e-6` or renormalize.

**Multiple testing**: Global FDR (Benjamini-Hochberg) across all genes for the F-test. Per-gene FDR for coefficient-level t-tests.

### Uncertainty Quantification

| Method | Default | Purpose |
|--------|---------|---------|
| HC3 SEs | On | Analytic coefficient uncertainty, heteroscedasticity-robust |
| Bootstrap CIs | On (`n_bootstrap=1000`) | 95% CIs on each beta_k and beta_{jk} |
| Permutation test | Off (`permutation_test=False`) | Null distribution of R^2 for model significance |

### Public API

```python
pc.tl.feature_simplex_regression(
    adata,
    feature_matrix=None,        # default: adata.X
    feature_names=None,
    max_degree=2,               # 1 = linear only, 2 = with interactions (both reported)
    permutation_test=False,
    n_permutations=1000,
    n_bootstrap=1000,           # 0 to disable bootstrap CIs
    robust_se=True,             # HC3
    store_residuals=True,       # store full residual matrix in obsm
    copy=False,
)
# Returns: SimplexRegressionResult (also stored in adata.uns['peach_simplex_regression'])
```

**Convenience wrappers**:
- `pc.tl.gene_simplex_regression(adata, **kwargs)` -> `feature_matrix=adata.X`
- `pc.tl.pathway_simplex_regression(adata, **kwargs)` -> `feature_matrix=adata.obsm['pathway_scores']`

### Key Outputs (SimplexRegressionResult)

Per feature:
- `vertex_coefficients`: array of beta_k [n_features x K]
- `interaction_coefficients`: array of beta_{jk} [n_features x K-choose-2] (degree 2 only)
- `vertex_ci_lower`, `vertex_ci_upper`: bootstrap CIs [n_features x K]
- `interaction_ci_lower`, `interaction_ci_upper`: bootstrap CIs
- `r_squared_degree1`, `r_squared_degree2`: per-feature R^2 at each degree
- `f_pvalue`: overall model significance (FDR-corrected)
- `vertex_pvalues`: per-coefficient significance [n_features x K]
- `interaction_pvalues`: per-interaction significance [n_features x K-choose-2]
- `residuals`: optional [n_cells x n_features] in obsm

Global:
- `feature_names`: list[str]
- `archetype_names`: list[str] (or indices)
- `interaction_pairs`: list[tuple] — which (j,k) pairs correspond to interaction columns
- `n_cells`, `n_features`, `n_archetypes`

---

## Archetype Driver Regression (Flipped)

### Method

Standard simplex regression: weights predict features.
Flipped: features predict weights.

```
E[w_k] = alpha_k + Sigma_g gamma_kg * GS_g + Sigma_{g<h} gamma_k,gh * GS_g * GS_h
```

- gamma_kg = how much geneset g contributes to archetype k membership
- gamma_k,gh = joint effect of genesets g and h on archetype k (interaction driving specialization)
- Intercept required (geneset scores are not compositional)
- K separate OLS regressions (one per archetype weight)

### Design Matrix Scaling

With G genesets, the interaction design matrix has G-choose-2 columns. For G=50 pathways, that's 1,225 interaction terms — manageable. For G=200, it's 19,900 — still feasible for OLS but getting large. For G=2000+, interaction terms become impractical.

**Safeguard**: If n_interactions > 5000, warn and suggest pre-filtering genesets or disabling interactions. The function should not silently build a 100k-column design matrix.

### Public API

```python
pc.tl.archetype_driver_regression(
    adata,
    feature_matrix=None,        # default: adata.obsm['pathway_scores']
    feature_names=None,
    max_degree=2,               # 1 = main effects only, 2 = with interactions
    n_bootstrap=1000,
    robust_se=True,
    max_interaction_features=200,  # safeguard: error if n_features exceeds this for degree=2
    copy=False,
)
# Returns: DriverRegressionResult (stored in adata.uns['peach_driver_regression'])
```

### Key Outputs (DriverRegressionResult)

Per archetype:
- `main_coefficients`: [K x n_features] — gamma_kg values
- `interaction_coefficients`: [K x n_features-choose-2] — gamma_k,gh values
- `main_pvalues`, `interaction_pvalues`: significance
- `main_ci_lower`, `main_ci_upper`: bootstrap CIs
- `r_squared`: per-archetype R^2 (how well genesets explain this archetype's weight)

---

## Feature Pattern Classification

### Method

Interpretation layer on simplex regression coefficients. No new statistical tests — pure classification of coefficient structure.

### Pattern Taxonomy

| Pattern | Detection Rule |
|---------|---------------|
| **Archetype-exclusive** | One beta_k significantly above all others; no significant interactions |
| **Monotonic gradient** | High R^2; one dominant beta_k but others non-negligible; ordered coefficient profile |
| **Multi-archetype shared** | 2+ beta_k values significantly elevated above the rest |
| **Antagonistic** | Some beta_k high, others low; large spread in vertex coefficients |
| **Ridge / blend-enriched** | Significant positive beta_{jk} interaction terms |
| **Valley / blend-depleted** | Significant negative beta_{jk} interaction terms |
| **Flat / ubiquitous** | Low R^2; all beta_k approximately equal |

### Classification Logic

Rule-based on coefficient magnitudes, significance, and R^2 thresholds:
1. If R^2 < threshold (default 0.01) -> flat
2. If max(beta_k) - min(beta_k) < effect_threshold AND no significant interactions -> flat
3. If any significant interaction terms -> ridge or valley (by sign)
4. Count how many beta_k are significantly above the mean -> exclusive (1), gradient (1 dominant + others), shared (2+), antagonistic (high spread)

Each feature gets a primary classification + confidence score + detailed breakdown.

### Public API

```python
pc.tl.classify_feature_patterns(
    adata,
    regression_result=None,     # default: read from adata.uns['peach_simplex_regression']
    r2_threshold=0.01,
    significance_threshold=0.05,
    effect_size_threshold=None, # auto-calibrated from data if None
)
# Returns: PatternClassificationResult (stored in adata.uns['peach_feature_patterns'])
```

### Archetype Summary (Query Layer)

```python
pc.tl.archetype_summary(
    adata,
    archetype_idx=None,         # int or None for all
    top_n=20,
    include_drivers=True,       # include flipped regression results if available
    include_gmm=True,           # include GMM components near this archetype
)
# Returns dict (or list of dicts) with per-archetype structured summary:
# {
#   'archetype_idx': 3,
#   'top_enriched': [...],      # highest beta_k features with CIs
#   'top_depleted': [...],
#   'interactions': {...},       # significant beta_{jk} involving this archetype
#   'pattern_counts': {...},     # how many features per pattern type
#   'residual_stats': {...},     # mean, std, outliers for cells near this archetype
#   'gmm_components': [...],    # nearby GMM components
#   'driver_genesets': [...],   # from flipped regression
# }
```

---

## Module 4: Simplex Density Decomposition (GMM)

### Method

1. **ILR transform**: Archetype weights (simplex) -> unconstrained R^{K-1} via isometric log-ratio
   - Helmert sub-composition basis (choice doesn't matter for full-covariance GMM — rotation-invariant)
   - Handle zeros: add small epsilon before log (standard compositional data practice)

2. **GMM fitting**: sklearn GaussianMixture on ILR-transformed weights
   - Fit for n_components = K through K_max (default: 3K)
   - Select by BIC

3. **Multi-initialization stability analysis**:
   - Run 20 initializations per n_components
   - For BIC-selected n_components, compute component correspondence across runs via Hungarian algorithm on centroid distances
   - Stability score per component = fraction of runs where it appears (centroid within threshold)
   - Only report components with stability > 0.7 (configurable)

4. **Component characterization**: For each stable component:
   - Map centroid back to simplex (inverse ILR) -> sub-archetype weight profile
   - Compute feature means/medians per component
   - Map to nearest archetype vertex

### Public API

```python
pc.tl.feature_simplex_decomposition(
    adata,
    feature_matrix=None,
    feature_names=None,
    n_components_range=None,    # default: (K, 3*K)
    model_selection='bic',
    covariance_type='full',
    n_initializations=20,       # for stability analysis
    stability_threshold=0.7,    # min stability to report a component
    characterize_features=True,
    random_state=42,
    copy=False,
)
# Returns: GMMResult (stored in adata.uns['peach_gmm'])
# Also stores: adata.obsm['peach_gmm_labels'] = component assignments
```

### Key Outputs (GMMResult)

- `n_components_optimal`: BIC-selected
- `n_components_stable`: after stability filtering
- `component_assignments`: [n_cells] — labels for stable components only
- `component_simplex_means`: [n_stable x K] — centroids in weight space
- `component_archetype_map`: [n_stable] — nearest archetype per component
- `component_stability_scores`: [n_stable] — stability across initializations
- `component_feature_profiles`: [n_stable x n_features] — mean feature per component
- `bic_values`: [n_tested] — BIC curve for elbow plot
- `gmm_model`: fitted sklearn GMM (in ILR space)

---

## Flow Matching

### Dependencies

- `flow_matching` (Facebook Research) — required, new dependency
- `torch` — already a dependency

### Core Implementation (`_core/utils/flow_matching.py`)

**Velocity network**: MLP with tangent projection (sum-to-zero not needed in Euclidean PCA space — just a standard MLP).

```python
class VelocityNetwork(nn.Module):
    """MLP: (x, t) -> v. Concatenates position and scalar time."""
    def __init__(self, dim, hidden_dims=(128, 128, 128)):
        # Standard MLP with ReLU activations

class FlowModel:
    """Wraps velocity net + fb flow_matching training infrastructure."""
    def __init__(self, dim, hidden_dims, lr, device):
        self.velocity_net = VelocityNetwork(dim, hidden_dims)
        self._prob_path = ConditionalProbPath(...)  # fb library

    def train(self, source, target, n_epochs, batch_size):
        """Train velocity field. Returns loss history."""

    def transport(self, x0, n_steps=50, return_trajectory=False):
        """Transport points from source to target. Euler integration."""

    def velocity_at(self, x, t):
        """Evaluate learned velocity at points and time."""

    def jacobian(self, x, t):
        """Compute Jacobian of velocity field via torch.autograd.functional.jacobian."""
```

### flow_within()

Intra-model flow between obs-defined cell subsets.

```python
pc.tl.flow_within(
    adata,
    source,                     # dict: {obs_key: value} e.g. {'treatment': 'Base'}
    target,                     # dict: {obs_key: value} e.g. {'treatment': 'PD1'}
    pca_key='X_pca',
    hidden_dims=(128, 128, 128),
    lr=1e-3,
    n_epochs=1000,
    batch_size=256,
    n_steps=50,                 # transport integration steps
    device='cpu',
    name=None,                  # stored as adata.uns[f'peach_flow_{name}']
    random_state=42,
    copy=False,
)
# Returns: FlowWithinResult
```

**FlowWithinResult**:
- `model`: trained FlowModel (for downstream Jacobian, re-transport)
- `losses`: training loss history
- `source_mask`, `target_mask`: boolean masks into adata
- `transported`: [n_source x dim] — transported source cells
- `mmd_before`, `mmd_after`: MMD between source/target before and after transport
- `pca_key`: which embedding was used

### flow_between()

Inter-model flow between separate AnnDatas.

```python
pc.tl.flow_between(
    adatas,                     # list of AnnData objects (2+)
    condition_key='condition',  # obs column name for labeling in output
    condition_labels=None,      # list of labels; default: ['condition_0', 'condition_1', ...]
    pairs=None,                 # list of (source_label, target_label); default: all ordered pairs
    pca_key='X_pca',
    hidden_dims=(128, 128, 128),
    lr=1e-3,
    n_epochs=1000,
    batch_size=256,
    n_steps=50,
    device='cpu',
    random_state=42,
)
# Returns: FlowBetweenResult
```

**FlowBetweenResult**:
- `adata`: concatenated AnnData with condition labels in `.obs[condition_key]`
- `flows`: dict of {(source_label, target_label): FlowWithinResult}
- `archetype_correspondence`: dict of {(src, tgt): np.ndarray[K_src x K_tgt]} — soft mapping

**Archetype correspondence**: For each pair, transport source archetype positions (from `adata.uns['peach_results']`) through the learned flow, compute normalized inverse-distance to target archetypes.

### flow_gene_alignment()

```python
pc.tl.flow_gene_alignment(
    adata,
    flow_result,                # FlowWithinResult or FlowBetweenResult
    t=0.5,                      # time point to evaluate velocity
    n_top=50,                   # top aligned/opposed genes to report
    pca_loadings_key=None,      # default: infer from adata.varm['PCs']
)
# Returns: GeneAlignmentResult
# - alignment_scores: [n_genes] — dot product of gene loading with mean velocity
# - top_aligned: top N genes moving WITH the flow
# - top_opposed: top N genes moving AGAINST the flow
```

### flow_jacobian()

```python
pc.tl.flow_jacobian(
    adata,
    flow_result,
    t=0.5,
    evaluation_points=None,     # default: source cell positions
    pca_loadings_key=None,
    aggregate='mean',           # 'mean', 'median', or None (per-cell)
)
# Returns: FlowJacobianResult
# - jacobian_det: [n_points] — local volume change (>1 = expanding, <1 = contracting)
# - feature_expansion: [n_genes] — per-gene expansion/contraction score
#   (projection of Jacobian onto each PCA loading direction)
# - mean_jacobian: [dim x dim] — averaged Jacobian matrix
```

### Flow Permutation Test

```python
pc.tl.flow_significance(
    adata,
    flow_result,                # or source/target specification
    n_permutations=100,         # fewer than regression (each is expensive)
    statistic='mmd',            # test statistic: 'mmd' or 'mean_velocity_norm'
)
# Returns: p_value, observed_stat, null_distribution
```

Permutes condition labels, retrains flow per permutation, compares observed MMD improvement against null. Default 100 permutations (not 1000) because each requires full training.

---

## Ternary Facet Plots

### Method

For any 3 archetypes: extract those 3 barycentric weights, renormalize to sum to 1, plot on triangular axes.

### Public API

```python
pc.pl.ternary_facet(
    adata,
    archetypes=(i, j, k),       # which 3 archetypes
    color_by='feature_name',     # gene, pathway, obs column, or 'density'
    feature_matrix=None,
    style='scatter',             # 'scatter', 'contour', 'relief'
    resolution=50,               # grid resolution for contour/relief
    regression_overlay=False,    # overlay simplex regression predicted surface
    ax=None,
    **kwargs,
)

pc.pl.ternary_facet_grid(
    adata,
    color_by='feature_name',
    facets='all',                # 'all' or list of (i,j,k) tuples
    ncols=3,
    **kwargs,
)
```

**Dependency**: `python-ternary` (optional, like squidpy). Fallback to custom matplotlib if not installed.

### Visualization Options

The `regression_overlay=True` option plots the Scheffe polynomial predicted surface as contour lines on top of the cell scatter. This directly connects Module 2 output to the ternary visualization.

---

## Additional Visualizations

### Regression Plots (`pl/regression.py`)

```python
pc.pl.coefficient_heatmap(adata, top_n=50)           # features x archetypes, beta_k values
pc.pl.interaction_heatmap(adata, top_n=50)            # features x archetype-pairs, beta_{jk}
pc.pl.r2_barplot(adata, top_n=50)                     # ranked features by R^2
pc.pl.vertex_radar(adata, feature)                    # spider plot of beta_k for one feature
pc.pl.regression_volcano(adata)                       # R^2 vs max vertex contrast
pc.pl.pattern_summary(adata)                          # stacked bar: feature counts per pattern
```

### GMM Plots (`pl/decomposition.py`)

```python
pc.pl.component_scatter(adata)                        # cells colored by GMM component
pc.pl.gmm_bic_curve(adata)                           # BIC vs n_components
pc.pl.component_heatmap(adata, top_n=20)             # features x components
pc.pl.component_stability(adata)                      # stability scores barplot
```

### Flow Plots (`pl/flow.py`)

```python
pc.pl.velocity_quiver(adata, flow_result)             # 2D PCA projection with arrows
pc.pl.gene_alignment_barplot(adata, alignment_result) # top/bottom aligned genes
pc.pl.jacobian_heatmap(adata, jacobian_result)        # per-feature expansion/contraction
pc.pl.trajectory_ribbon(adata, flow_result)           # transported cells colored by time
pc.pl.flow_magnitude(adata, flow_result)              # where is flow strongest
pc.pl.density_comparison(adata, flow_result)          # source vs transported vs target KDE
pc.pl.archetype_correspondence(flow_between_result)   # K_src x K_tgt heatmap or chord diagram
```

---

## Spatial Extension

### Archetype Pair Enrichment

Extend existing `tl/spatial.py` to support weight-based pair analysis.

```python
pc.tl.archetype_pair_enrichment(
    adata,
    archetype_pairs=None,       # list of (i,j) or 'all'
    weight_threshold=0.3,       # min weight to consider a cell "participating" in an archetype
    n_permutations=1000,
    spatial_key='spatial',
)
# Returns: PairEnrichmentResult
# Per pair: enrichment score, p-value, spatial autocorrelation
```

Uses squidpy infrastructure under the hood. Extends the existing `_ensure_categorical()` pattern to handle weight-derived pair labels.

---

## Implementation Order

| Phase | Component | Depends On | Est. Files |
|-------|-----------|------------|------------|
| 1 | Feature infrastructure + resampling | None | 2 |
| 2 | Simplex regression (core engine) | Phase 1 | 2 (core + tl) |
| 3 | Pattern classification | Phase 2 | 2 (core + tl) |
| 4 | Archetype driver regression (flipped) | Phase 1 | extends Phase 2 files |
| 5 | GMM decomposition | Phase 1 | 2 (core + tl) |
| 6 | Flow matching | Phase 1 | 2 (core + tl) |
| 7 | Ternary facets + regression viz | Phase 2 | 2 (pl/) |
| 8 | GMM + flow visualizations | Phases 5, 6 | 2 (pl/) |
| 9 | Archetype summary query layer | Phases 2-5 | extends Phase 3 tl file |
| 10 | Spatial pair enrichment | Phase 1 | extends existing spatial.py |
| 11 | types_index.py + tools_schema.py updates | All | 2 existing files |

### Testing Strategy

Per module:
- Synthetic data with planted ground truth (known coefficients, known components, known flow)
- Real data: hsc_10k.h5ad for quick e2e
- Positive controls: features with known patterns should be correctly classified
- Negative controls: permuted features should yield no signal (low R^2, no significant coefficients)
- Each module gets its own test file + integration test running all modules on same dataset

### Dependencies (new)

| Package | Required/Optional | Used By |
|---------|-------------------|---------|
| `flow_matching` (fb) | Optional | Flow matching module |
| `python-ternary` | Optional | Ternary facet plots |
| `torch` | Already required | Flow matching (velocity net) |
| `sklearn` | Already required | GMM fitting |
| `scipy` | Already required | ILR transform, statistical tests |

---

## Scope Boundary

**In scope (v0.5.0)**: Everything above.

**Out of scope (v0.6+)**:
- Level set profiles (absorbed by regression interaction terms)
- Adaptive complexity regression / GAMs
- Flow matching in archetype weight space (Riemannian)
- Bootstrap CIs on flow gene alignment
- Spatial cross-cell-type Pareto front alignment
- Higher-order Scheffe polynomials (degree 3+)
- GMM per-component archetype subspace re-fitting
- Dirichlet mixture model (ILR+GMM is sufficient)

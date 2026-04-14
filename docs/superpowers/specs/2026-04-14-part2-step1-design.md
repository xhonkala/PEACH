# Paper Part 2 — Step 1: Global archetypal fit on TNBC tumor cells

**Status**: design approved 2026-04-14, implementation pending
**Scope**: Script 2 Step 1 only (Fig 3A, 3B, 3C). Steps 2–4 (per-timepoint models, flow, R vs NR contrasts) and Step 5 (held-out prediction) are out of scope.
**Parent plan**: `docs/plans/2026-04-14-paper-part1-script-architecture.md`
**Companion memory**: RNRflow paper plan v3

---

## 1. Goal

Establish whether R1, R2, and NR patients' TNBC tumor cells occupy distinct territories in a single globally-fit archetype space, across three treatment timepoints (Base, PD1, RTPD1). Produces three figures:

- **Fig 3A** — global archetype space, 3-class × 3-timepoint coloring, with held-out cell projection QC.
- **Fig 3B** — archetype molecular characterization via simplex regression (genes + pathways + stress-gene subset).
- **Fig 3C** — gated phenotype distance heatmaps (W2 + Euclidean centroid) + ungated three-panel diversity block.

Deliverables: `prep_tnbcrad.py`, `run_paper_part2_tnbc.py`, `tests/test_paper_part2_step1.py`, populated `data/paper_part2/`, one `part2_report_YYYYMMDD_rN.html` per iteration.

## 2. Philosophy decisions (pinned)

1. **Approach 1 — full Part 1 fork.** Copy `run_paper_part1_hsc.py`, keep three-phase skeleton, reuse `_paper_part1_{prep,viz}.py` by import (no rename — Part 1 is still iterating at r13; shared-code refactor is wrong time).
2. **Cell subset = all 31,503 malignant cells** (inferCNV-called). Trust `subtype_new == 'cancer cells'`; ignore `predicted_labels` (CellTypist leakage). Use `majority_voting` as a downstream covariate, not as a filter.
3. **Response kept 3-class** (R1 / R2 / NR). Do not collapse R1∪R2. The biology payoff is whether R1/R2/NR show three distinct or branching territories.
4. **Stratification for 80/20 cell-level holdout** = `cohort × response_group × treatment`, with safety fallback to cohort-only for strata < 10 cells.
5. **N_PCS = 12**, locked by scree inspection of `outputs/diagnostic/tnbc_pca_scree.png` on 2026-04-14. Do not data-drive; hardcode like Part 1.
6. **2-Wasserstein (W2) everywhere** — consistent with the flow module codebase convention.
7. **MAX_EPOCHS_FINAL = 200** (Part 1 default). If convergence cap is hit in r1, bump in a later iteration; do not preempt.
8. **SUBSAMPLE_FRACTION = 0.2** for prototyping, flip to 1.0 for production runs.

## 3. Files and data flow

### 3.1 Directory layout

```
scripts/
├── _paper_part1_prep.py        # (existing, reused — no changes)
├── _paper_part1_viz.py         # (existing, augmented with new helpers — see §6)
├── prep_tnbcrad.py             # NEW
└── run_paper_part2_tnbc.py     # NEW

data/paper_part2/                # NEW directory
├── adata_tnbc_full_prepped.h5ad     # MAD-filtered + PCA'd, pre-split (provenance)
├── adata_tnbc_train.h5ad            # 80% train split
└── adata_tnbc_holdout.h5ad          # 20% cell-level holdout

outputs/paper_part2/             # NEW directory
├── part2_report_YYYYMMDD_rN.html
└── run_log_YYYYMMDD_rN.txt

tests/
└── test_paper_part2_step1.py   # NEW — mirrors test_paper_part1_fixes.py structure

docs/superpowers/specs/
└── 2026-04-14-part2-step1-design.md   # this document
```

### 3.2 Data flow

```
GSE246613_TNBC_ONLY_TRAIN.h5ad  (31,503 cells, all malignant per inferCNV)
  │
  ├─ prep_tnbcrad.py
  │   1. adata.X = adata.layers["logcounts"].copy()  (no further normalization)
  │   2. Coerce cohort / treatment / response_group / majority_voting to category
  │   3. apply_mt_rb_mad_filter(n_mads=3.0)           (reused from _paper_part1_prep)
  │   4. sc.pp.pca(n_comps=50, zero_center=False, use_highly_variable=False)
  │   5. Slice to first 12 PCs (locked by scree)
  │   6. Stratified 80/20 split on cohort × response_group × treatment
  │      with cohort-only fallback for strata < 10 cells
  │   7. Write full_prepped / train / holdout h5ad files
  │
  └─ run_paper_part2_tnbc.py
      ├─ Phase 1 — training + QC
      ├─ Phase 2 — Fig 3A (global archetype space + 3×3 coloring + holdout QC)
      └─ Phase 3 — Fig 3B + gated Fig 3C
      → outputs/paper_part2/part2_report_YYYYMMDD_rN.html
```

### 3.3 Prototype banner (top of run script)

The main script opens with a module docstring declaring prototype status, identifying this as Step 1 of 5 in Part 2, and enumerating Steps 2–5 so future collaborators understand what is *not* here. The report's first text block reiterates the prototype status and `SUBSAMPLE_FRACTION` value.

## 4. Prep pipeline (`prep_tnbcrad.py`)

### 4.1 Recipe (ordered, no deviations)

```python
adata = sc.read_h5ad("data/GSE246613_TNBC_ONLY_TRAIN.h5ad")

# Step 2 — logcounts → X, no scaling, no re-normalization
adata.X = adata.layers["logcounts"].copy()

# Step 3 — categorical dtype coercion
for col in ["cohort", "treatment", "response_group", "majority_voting"]:
    adata.obs[col] = adata.obs[col].astype("category")

# Step 4 — MAD filter (reused)
adata = apply_mt_rb_mad_filter(adata, n_mads=3.0)

# Step 5 — fresh PCA
sc.pp.pca(adata, n_comps=50, zero_center=False, use_highly_variable=False)

# Step 6 — slice to 12 PCs (scree-locked)
N_PCS = 12
adata.obsm["X_pca"] = adata.obsm["X_pca"][:, :N_PCS]
adata.uns["pca"]["variance"] = adata.uns["pca"]["variance"][:N_PCS]
adata.uns["pca"]["variance_ratio"] = adata.uns["pca"]["variance_ratio"][:N_PCS]

# Step 7 — stratified split with safety valve
stratum = (adata.obs["cohort"].astype(str) + "|" +
           adata.obs["response_group"].astype(str) + "|" +
           adata.obs["treatment"].astype(str))
train_idx, holdout_idx = safe_stratified_split(
    stratum, test_size=0.20, min_stratum_size=10, random_state=42
)

# Step 8 — write outputs
adata.write_h5ad("data/paper_part2/adata_tnbc_full_prepped.h5ad")
adata[train_idx].write_h5ad("data/paper_part2/adata_tnbc_train.h5ad")
adata[holdout_idx].write_h5ad("data/paper_part2/adata_tnbc_holdout.h5ad")
```

### 4.2 `safe_stratified_split` (new helper, lives in `_paper_part1_prep.py`)

Wraps `sklearn.model_selection.StratifiedShuffleSplit`. For each unique stratum value, if the stratum has fewer than `min_stratum_size` cells, replace that stratum label with `cohort` only (stripping the response/treatment subdivision). If a cohort-only stratum is still < `n_splits + 1` (i.e., 2), fall back to random assignment for those cells. Log fallback count (cells / strata) so the prep report can surface it.

Rationale: Part 2 data has severe per-patient imbalance (e.g., Patient06 = 6 cells across 3 strata). Naive `StratifiedShuffleSplit` would error. The safety valve preserves the stratification intent wherever possible while not crashing on tiny strata.

### 4.3 Prep report output

A lightweight HTML at `data/paper_part2/prep_report.html` logging:
- Input cell and gene counts
- MAD filter diagnostics (MT/RB thresholds, drop counts)
- PCA variance ratio at chosen N_PCS=12 (for provenance — we already know this)
- Stratified split counts per stratum
- Fallback count (how many cells fell back to cohort-only / random)

### 4.4 Explicit non-steps

- **No** `sc.pp.normalize_total` (already log-normalized)
- **No** `sc.pp.scale` (destroys convex hull — Part 1 §10 gotcha)
- **No** HVG subsetting before PCA (kills archetypal signal)
- **No** batch correction / scVI (data-driven archetypes should learn batch structure if it is real; batch correction defeats the archetypes)
- **No** data-driven PC selection at prep time (scree-locked to 12)

## 5. Run script (`run_paper_part2_tnbc.py`)

### 5.1 Config constants

```python
DATA_DIR = "data/paper_part2/"
OUTPUT_DIR = "outputs/paper_part2/"

SUBSAMPLE_FRACTION = 0.2          # prototyping; flip to 1.0 for production
SUBSAMPLE_STRATIFY = "response_group"

MAX_EPOCHS_FINAL = 200
EARLY_STOP_PATIENCE = 15
N_PCS = 12                         # scree-locked 2026-04-14

K_RANGE = list(range(3, 11))       # 3..10 archetypes
HIDDEN_DIMS_OPTIONS = [[64,128], [128,256], [256,128,64]]
INFLATION_FACTOR_RANGE = [0.75, 1.0, 1.25, 1.5]

MODEL_CONFIG = {
    "manifold_weight": 0.005,
    "kld_weight": 0.01,
    "sparsity_weight": 0.0,
    "archetypal_weight": 0.9,
}
```

`use_hidden_transform` is left at `Deep_AA`'s default of `True`. All three loss terms (manifold, diversity, monitoring) now use `get_effective_archetypes()` per the 2026-04-14 Deep_AA patch — this spec explicitly relies on that patch being in place.

### 5.2 Phase 1 — training + QC

Reads train + holdout h5ad files. Optionally subsamples train by `SUBSAMPLE_STRATIFY`. Runs the standard Part 1 CV → final-fit → drift QC pattern:

1. `pc.tl.hyperparameter_search` over `K_RANGE × HIDDEN_DIMS_OPTIONS × INFLATION_FACTOR_RANGE` with 5-fold CV, `max_epochs=20`, `use_pcha_init=False`.
2. Pick K as smallest value with CV mean_archetype_r2 ≥ 0.9. Pick hidden_dims and inflation_factor from CV winner at that K.
3. `pc.tl.train_archetypal` with `pcha_init=True`, winner config, `MODEL_CONFIG`, `MAX_EPOCHS_FINAL`, `EARLY_STOP_PATIENCE`.
4. `pc.tl.archetypal_coordinates` + `pc.tl.assign_archetypes(percentage_per_archetype=0.15)` on train.
5. Project holdout via `pc.tl.extract_archetype_weights(model=res["model"])` + `pc.tl.archetypal_coordinates`.

**Phase 1 report section contents** (reused from Part 1):
- Config card grid
- CV summary table
- Elbow curve (K vs R²)
- `build_drift_qc_panel` drift/stability output
- `convergence_status` plateau check
- 3D archetype space preview

**Phase 1 NEW for Part 2:**
- **PC1 correlation scan**: Spearman of `X_pca[:, 0]` against `cohort`, `treatment`, `response_group`, `total_counts`, `percent_mito`. Top-3 correlations reported inline. Rationale: the scree diagnostic showed PC1 eats 40.4% of variance — a flag for batch-like structure that we want visible in the report. If PC1 is dominantly `cohort`, the Fig 3C gate becomes more important as a guardrail.
- **Holdout projection R²**: archetypal R² on the 20% held-out cells against the trained model, reported alongside train R². Large train-vs-holdout gap = overfit warning.

### 5.3 Phase 2 — Fig 3A (global archetype space)

**5.3.1 Coloring scheme — 9-combo ramp**

| response_group | hue family | Base / PD1 / RTPD1 |
|----------------|------------|--------------------|
| **NR** | Reds | `#fca5a5` / `#ef4444` / `#991b1b` |
| **R1** | Oranges | `#fed7aa` / `#f97316` / `#9a3412` |
| **R2** | Blues | `#93c5fd` / `#2563eb` / `#1e3a8a` |

Hue encodes response lineage; lightness encodes timepoint. Rationale: intra-group lineage identity (R1 stays R1 across time) is the primary story; timepoint is secondary ordering. Rejected alternative: three-panel facet with 3 colors each (loses the "does R1's trajectory bend toward R2 over time?" signal in a single frame).

**5.3.2 Layout**

Main section contains three 3D-plotly panels plus characterization + hypergeometric tables:

1. **Main scatter** — `pc.pl.archetypal_space` with 9-combo coloring, archetype diamonds overlaid. Sidebar: N cells per (response × treatment), K archetypes, dominant-archetype-per-response bar chart, holdout R², PC1 correlation scan.
2. **Per-timepoint facet** — 3 side-by-side 3D views (Base / PD1 / RTPD1), each showing 3 response-colored clouds. Answers "where does each arm start vs end."
3. **Holdout projection** — same space, holdout cells at 50% alpha + distinct marker. Caption reports mean per-cell distance from holdout cells to nearest archetype.

**5.3.3 Archetype characterization table**

One row per archetype, columns:

| archetype | % cells | dom. response | dom. treatment | top-3 patients | dom. `majority_voting` | top-5 genes |
|-----------|---------|----------------|-----------------|----------------|------------------------|--------------|

Cells assigned per `assign_archetypes(percentage_per_archetype=0.15)`. Top-5 genes from `pc.tl.feature_simplex_regression` degree-1 exclusive coefficients.

**5.3.4 Companion hypergeometric table (new)**

Rendered alongside §5.3.3 as its "friend." One sub-table per covariate:

| Sub-table | Rows × Cols | Cell format | FDR correction |
|-----------|-------------|-------------|-----------------|
| Response | K × {NR, R1, R2} | `OR (p, q)` | BH across 3K tests |
| Treatment | K × {Base, PD1, RTPD1} | `OR (p, q)` | BH across 3K tests |
| `majority_voting` | K × subtypes with ≥50 cells | `OR (p, q)` | BH across live-subtype × K tests |
| Cohort | per archetype: top-3 enriched patients with `OR (p, q)` | row-wise truncation (23×K unreadable); rank by ascending p-value, break ties by descending OR | per-archetype BH |

Binned assignment = cells captured by `assign_archetypes(percentage_per_archetype=0.15)`. Test = two-tailed Fisher's exact on each 2×2 contingency. Cells in overlapping 15% windows are counted once per window (matches Part 1 convention).

### 5.4 Phase 3 — Fig 3B + (gated) Fig 3C

**5.4.1 Fig 3B — archetype molecular characterization (3 sub-plots)**

1. **Gene dotplot** — `pc.tl.feature_simplex_regression` degree-1 results, filtered to exclusive ratio ≥ 2.5 and FDR ≤ 0.05, visualized via `pc.pl.dotplot` with `dotplot_figsize` sizing.
2. **Pathway dotplot** — same degree-1 call with `feature_matrix='pathway_scores'` (requires `adata.obsm['pathway_scores']`; if missing, render an explicit error block and skip this sub-plot without killing Phase 3).
3. **Stress-gene subset dotplot** — re-run degree-1 simplex regression restricted to genes in `STRESS_GENES_FLAT` (from `stress_genes/load_stress_genes.py`). First peek at stress axis; negative results are informative too.

**No pathway hypergeometric table** — with >3000 pathways it would be pandemonium. The pathway simplex regression already handles archetype-specific pathway signal.

**5.4.2 Fig 3C-i — gated distance heatmaps**

**Gate**:
```python
segregation_ratio = between_response_mean_w2 / within_response_mean_w2
FIG3C_GATE = segregation_ratio >= 1.3   # prototype threshold — tune after r1
```

Definitions:
- Unit of grouping = `(response_group, treatment)` — one of 9 groups. The archetype weight distribution of a group is the distribution over K archetypes for all cells in that group.
- `within_response_mean_w2`: mean W2 over all pairs `(gi, gj)` where `gi` and `gj` share the same `response_group` but differ in `treatment` (3 responses × C(3,2)=3 treatment pairs = 9 pairs).
- `between_response_mean_w2`: mean W2 over all pairs `(gi, gj)` where `gi` and `gj` differ in `response_group` (any treatments: 3 response pairs × 3×3 = 27 pairs).

The 1.3 threshold is a prototype choice — expect to tune it in r1 feedback once we see the empirical distribution. It is **not** a hypothesis test; it is a gating heuristic for whether rendering 3C-i tells a coherent story.

**If gate passes** — render two `(3K × 3K)` heatmaps side by side; rows/cols are `(response_group, archetype)` pairs:

| Panel | Metric | Space |
|-------|--------|-------|
| Left | 2-Wasserstein (W2) | K-dim archetype weight simplex |
| Right | Euclidean centroid distance | N_PCS=12 PCA space |

Agreement scalar in caption: Spearman correlation between paired distances. ρ > 0.8 = robust segregation; lower = investigate which metric is misleading.

**If gate fails**: info box with the metric values and a handoff note — "segregation not clear at this K — consider K±1 or earlier per-timepoint split." 3C-ii still renders.

**5.4.3 Fig 3C-ii — diversity block (always rendered, three panels)**

| Panel | Metric | Input | Output |
|-------|--------|-------|--------|
| **1** | Per-cell Shannon entropy of archetype weights | `adata.obsm[weights_key]` | Violin × 3 groups, Kruskal-Wallis + pairwise Dunn (BH) |
| **2** | **Per-group PCA dispersion** (pre-registered test) | `adata.obsm['X_pca']` (12 dims), full cell profiles | Median pairwise Euclidean distance per group, bar with bootstrap CI (200 resamples, up to 500 cells per group per resample) |
| **3** | Per-group entropy of *pooled* mean archetype weight vector | `.mean(axis=0)` of `adata.obsm[weights_key]` within each group | Shannon H per group, bar |

Why three panels:
- Panel 1 asks "are individual cells internally mixed?"
- Panel 2 asks "are cells feature-space-heterogeneous?" (full-profile diversity, not weight-only) — **pre-registered**: R2 < NR predicted.
- Panel 3 asks "does the group collectively use a diverse archetype set?"

Disagreements are diagnostic: high panel 2 + low panel 1 = cells are heterogeneous in feature space but each one is purity-1 on an archetype (distinct sub-populations, not fuzzy mixing).

## 6. Reused vs new helpers

### Reused verbatim (imports only)
- `_paper_part1_prep.apply_mt_rb_mad_filter`
- `_paper_part1_viz.build_drift_qc_panel`, `convergence_status`, `build_r2_vs_fdr_scatter`, `build_overlapping_ridgeplot`, `compute_cross_model_r2`, `compute_archetype_to_centroid_distance`, `dotplot_figsize`, `_adjust_labels`
- `stress_genes.load_stress_genes.STRESS_GENES_FLAT`

### New helpers (land in `_paper_part1_{prep,viz}.py` so Part 1 can use them too when it fits)

In `_paper_part1_prep.py`:
- `safe_stratified_split(stratum, test_size, min_stratum_size, random_state) -> (train_idx, holdout_idx)` — fallback-aware stratified split.

In `_paper_part1_viz.py`:
- `build_response_timepoint_colormap(responses, treatments) -> dict` — returns the 9-combo hue × lightness mapping.
- `build_archetype_char_table(adata, archetypes_col, covariates, top_k_genes=5) -> pd.DataFrame` — §5.3.3 table.
- `build_archetype_hypergeometric_tables(adata, archetypes_col, covariates, min_subtype_cells=50) -> dict[str, pd.DataFrame]` — §5.3.4 OR tables, BH-corrected per covariate.
- `build_holdout_projection_qc(adata_train, adata_holdout, archetype_coords_col) -> dict` — train vs holdout R² + per-cell NN distance.
- `compute_w2_archetype_distance(weights_a, weights_b) -> float` — 2-Wasserstein in K-dim simplex for two cell groups.
- `build_segregation_ratio(adata, group_col, archetypes_col) -> dict[str, float]` — within / between / ratio.
- `build_distance_heatmaps(adata, group_col, archetypes_col, pca_key, assign_key) -> tuple[Figure, float]` — W2 + Euclidean `(3K × 3K)` heatmaps + Spearman agreement scalar.
- `build_diversity_block(adata, group_col, weights_key, pca_key, bootstrap_n=200, subsample=500) -> Figure` — the three-panel §5.4.3 block.

Script-local (do **not** promote to `_paper_part1_viz.py`; these are Part 2 Step 1-specific):
- `_stratified_subsample(adata, fraction, stratify_col)` — already exists script-local in Part 1; copy into Part 2 script.
- `_smallest_k_above_threshold(cv, threshold)` — K-selection rule; mirrors Part 1.

## 7. Testing (`tests/test_paper_part2_step1.py`)

Mirrors Part 1's `test_paper_part1_fixes.py` structure: synthetic-data computation checks + regex-based structural checks on `run_paper_part2_tnbc.py` and `prep_tnbcrad.py`.

### 7.1 Computation tests
- `test_safe_stratified_split_fallback` — synthetic dataframe with one-cell strata, verify fallback triggers and cells are assigned.
- `test_prep_tnbcrad_categorical_coercion` — verifies `cohort/treatment/response_group/majority_voting` are `CategoricalDtype` after prep.
- `test_response_timepoint_colormap_shape` — returns 9 colors for 3 responses × 3 treatments.
- `test_archetype_hypergeometric_tables_bh_correction` — synthetic cells with known response enrichment → BH-corrected q-values are monotone in p-values.
- `test_w2_archetype_distance_symmetric_nonneg` — `W2(a,b) == W2(b,a)` and ≥ 0.
- `test_build_segregation_ratio_identity_case` — identical group distributions give ratio ≈ 1.0.
- `test_diversity_block_computed` — builds all three panels on synthetic data without crashing; outputs are finite.
- `test_fig3c_gate_threshold` — synthetic high-segregation vs low-segregation inputs cross the gate in expected direction.

### 7.2 Structural tests
- `test_part2_config_constants` — regex for `N_PCS = 12`, `MAX_EPOCHS_FINAL = 200`, `K_RANGE = list(range(3, 11))`, `SUBSAMPLE_FRACTION = 0.2`.
- `test_part2_required_call_sites` — regex for `apply_mt_rb_mad_filter`, `sc.pp.pca(n_comps=50, zero_center=False`, `safe_stratified_split`, `build_response_timepoint_colormap`.
- `test_part2_section_titles` — run script contains Fig 3A, Fig 3B, Fig 3C section adds.
- `test_part2_no_forbidden_calls` — no `sc.pp.normalize_total`, no `sc.pp.scale`, no `highly_variable_genes` in `prep_tnbcrad.py`.
- `test_part2_prototype_banner_present` — module docstring mentions "Step 1 of 5" + references Steps 2–5.

### 7.3 Skipped tests (known, match Part 1 pattern)
- `test_part2_subsample_fraction_is_full` — skipped while `SUBSAMPLE_FRACTION = 0.2` for prototyping.

## 8. Run commands + iteration protocol

### Prep
```bash
conda run -n archetype python scripts/prep_tnbcrad.py
```

### Main run (unbuffered stdout — Part 1 §10 gotcha)
```bash
conda run -n archetype python -u scripts/run_paper_part2_tnbc.py \
  > outputs/paper_part2/run_log_$(date +%Y%m%d)_r1.txt 2>&1 &
```
Or `PYTHONUNBUFFERED=1`. Log stays 0 bytes without one of these.

### Rev numbering

Same pattern as Part 1: `REPORT_PATH = f"outputs/paper_part2/part2_report_{DATE_TAG}_r{rev}.html"`. Auto-increment `rev` from existing file count. Log filenames chosen manually.

### Iteration cadence
1. Start with `SUBSAMPLE_FRACTION = 0.2` (~2 h/run).
2. Run in background, poll every 30–60 min. `/loop` skill handles this.
3. After each run: triage HTML, keep `r{N}-feedback.md` alongside the plan doc, batch fixes.
4. Run `pytest tests/test_paper_part2_step1.py -q` after each batch.
5. Flip `SUBSAMPLE_FRACTION = 1.0` only once structure + content are converged.

## 9. Open risks (flagged, not blockers)

1. **PC1 = 40.4% variance.** If the PC1-correlation scan shows PC1 ≈ cohort/patient, the archetype space carries a batch axis. Discuss with author on first report — may require ComBat or Harmony pre-PCA (carefully, to avoid destroying the convex hull). Not a blocker for prototype.
2. **Severe per-patient cell imbalance.** Patient53 has 4,835 cells, Patient06 has 6. Cohort enrichment OR tests for tiny patients will be noisy; flag this in the hypergeometric table caption.
3. **R1/R2/NR meaning.** The paper plan memo described 2-class (R/NR, 20R/18NR) but the data file has 3-class. We kept the 3-class interpretation. Author to confirm R1/R2 definitions match their clinical stratification before final production runs.
4. **Held-out patients.** Script 3's held-out-patient file is not yet in `data/`. Step 1 does not need it — cell-level 20% holdout is sufficient for Fig 3A projection QC. Blocking only when Script 3 lands.
5. **Pathway scores not guaranteed.** `adata.obsm['pathway_scores']` may not be present on the TNBC file. Phase 3 Fig 3B sub-plot 2 (pathway dotplot) renders an explicit error block and continues if absent; this is a degradation mode, not a crash.

## 10. Deliverables checklist

- [ ] `scripts/prep_tnbcrad.py`
- [ ] `scripts/run_paper_part2_tnbc.py`
- [ ] `scripts/_paper_part1_prep.py` — add `safe_stratified_split`
- [ ] `scripts/_paper_part1_viz.py` — add 7 new helpers per §6
- [ ] `data/paper_part2/` populated by prep script
- [ ] `outputs/paper_part2/` populated by first r1 run
- [ ] `tests/test_paper_part2_step1.py`
- [ ] First `part2_report_YYYYMMDD_r1.html` triaged for r2 feedback

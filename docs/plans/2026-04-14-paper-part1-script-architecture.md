# Paper Part 1 Script Architecture

**Purpose**: Reference for forking the Part 1 pipeline into a Part 2
prototyping loop. Documents the directory layout, shared helpers, phase
boundaries, report rendering, config surface, and common gotchas
discovered in the r1–r13 review cycle.

**Audience**: you, a future collaborator, or a coding agent picking up
the pipeline cold. Everything here is empirically validated against the
actual scripts as of 2026-04-14 (r13 complete).

---

## 1. Directory layout

```
scripts/
├── _paper_part1_prep.py      # Shared QC / PCA helpers (prep step)
├── _paper_part1_viz.py       # Shared report / figure helpers (main run)
├── prep_hsccmp.py            # Prep pipeline for HSC + CMP → .h5ad files
├── prep_bigov.py             # Prep pipeline for bigOV (ovarian) → .h5ad
├── run_paper_part1_hsc.py    # Main run: HSC/CMP Part 1 report
├── run_paper_part1_ov.py     # Main run: Ovarian Part 1 report
├── run_diagnostic_fits.py    # Ad hoc diagnostic (PCA + search + fit)
├── run_manifold_sweep.py     # manifold_weight grid sweep
├── run_loss_sweep.py         # kld × sparsity sweep
├── run_fit_diagnostic.py     # Full-data fit diagnostic
└── run_viz_prototypes.py     # Sandbox for new viz ideas

data/paper_part1/              # Outputs of prep step
├── adata_hsc_train.h5ad       # 80 % HSC train split
├── adata_holdout.h5ad         # 20 % HSC holdout split
├── adata_cmp.h5ad             # All CMP cells (target population)
└── (analogous OV files)

outputs/paper_part1/            # Outputs of main run
├── part1_report_YYYYMMDD_rN.html   # Report (auto-increments rN per day)
└── run_log_YYYYMMDD_rN.txt          # Stdout / stderr (if launched with tee)
```

**Convention**: anything starting with `_` is a library module (imported
by the scripts beside it). Anything starting with `run_` or `prep_` is
an executable entry point.

---

## 2. Shared helper modules

### `_paper_part1_prep.py`

Re-usable preprocessing primitives. Functions:

| Function                          | Purpose |
|-----------------------------------|---------|
| `apply_mt_rb_mad_filter(adata, n_mads=3.0)` | 3-MAD cell filter on MT + RB fractions + hard drop of `^MT-`, `^RPS/RPL`, `^MRPS/MRPL`, `MALAT1` genes. Single mandatory QC step. |
| `select_n_pcs_by_cumvar(pca_matrix, threshold=0.95, min_pcs=2, max_pcs=50)` | Data-driven PCA dim selection. Not currently used by prep scripts (see §4 — PCs hardcoded to 13 for HSC, 11 for OV). Kept for future use. |

### `_paper_part1_viz.py`

Report / figure helpers shared across HSC + OV runs. Grouped by concern:

**Drift / convergence QC (W-A7, W-A8)**
- `build_drift_qc_panel(results_list, drift_threshold=0.05, converged_window=10)` → full HTML fragment with STABLE / DRIFTING badges, drift curve figure, and summary table. Uses `_window_median` (r12 fix) for robustness to early-training drift spikes.
- `convergence_status(history, max_epochs, early_stop_triggered, actual_epochs, window=10, delta_threshold=0.01)` → three-tier status (CONVERGED / NON_CONVERGED_HIT_CAP / NOT_CONVERGED_INSUFFICIENT_HISTORY) + Δloss mean. Orthogonal to drift; checks loss plateau.

**Figure helpers**
- `build_r2_vs_fdr_scatter(r2, fdr, names, …)` → R² vs −log10(min FDR) scatter with thresholds + adjustText label repulsion.
- `build_tricolor_gene_scatter(expression, expansion, flow_strength, names, …)` → expression (x) × expansion (y) coloured by flow strength + sized by |flow|.
- `build_absence_plot(expression, flow_strength, names, …)` → highlights high-expression / low-flow genes in red. Caption explicitly notes these are candidates for *shared* source/target baseline, not hidden flow drivers (r12 clarification).
- `build_lollipop_chart(flow, expansion, expression, names, top_n=25, …)` → ranked bars; colour = expansion sign, head size = expression. Has a composite legend (r12).
- `build_overlapping_ridgeplot(data, overlap=0.5, max_groups=12, …)` → Seurat-style stacked KDEs.
- `build_pseudotime_expansion_plot(pseudotime, per_cell_gene_matrix, names, max_genes=10, …)` → binned gene traces along a scalar pseudotime. Reused for pathway traces in r13.
- `build_permutation_curve_figure(swap_fractions, null_mean_curve, null_std_curve, observed)` → permutation degradation curve per (source, target) pair.

**Computational utilities**
- `compute_cross_model_r2(weights, archetypes, original_coords)` → pure numpy archetypal R² on reconstructed coords. Used for the cross-model generalisation check.
- `compute_archetype_to_centroid_distance(archetypes, centroids, bin_radius)` → extrapolation_ratio per archetype; used by drift panel.
- `dotplot_figsize(long_df, y_col, *, base=(12,8), per_row=0.30, floor=6.0, ceiling=48.0, width=None)` → dynamic `figsize` tuple so long y-axis dotplots auto-expand vertically. Pass to `pc.pl.dotplot(..., figsize=dotplot_figsize(df))`.
- `_adjust_labels(ax, texts, x=None, y=None)` → adjustText wrapper with safe fallback (silent no-op if adjustText missing).

**HTML atoms** (replicated inline in each `run_*` script, not here)
- `fmt_pval`, `error_html`, `metric_card`, `metric_grid`, `safe_plotly_html`
- `HTMLReport` class (minimal report builder with sections as `<details>` blocks)

---

## 3. Main run script structure

Both `run_paper_part1_hsc.py` and `run_paper_part1_ov.py` follow the
same three-phase skeleton, documented here from the HSC script.

```
run_paper_part1_hsc.py
├─ Paths + run config (DATA_DIR, OUTPUT_DIR, SUBSAMPLE_FRACTION, …)
├─ Helpers (fmt_pval, _stratified_subsample, …)
├─ HTMLReport class
├─ regression_to_long_df(reg_result, …)    # simplex reg → dotplot DF
│
├─ phase1_train_models(report)                        ← §3.1
│    - Loads adata_hsc_train + adata_cmp
│    - Hyperparameter search (CV over K × hidden × inflation)
│    - Trains final HSC + CMP models
│    - Returns (adata_hsc, adata_cmp, res_hsc, res_cmp)
│    - Writes sections: "CMP Model (Fig 1C)", "HSC Model (Supplemental)"
│
├─ phase2_figure1(adata_hsc, adata_cmp, res_hsc, report)  ← §3.2
│    - Figure 1: Deep_AA introduction
│    - Writes sections: "Figure 1: Deep_AA Introduction"
│
├─ phase3_figure2(adata_hsc, adata_cmp, report)           ← §3.3
│    - Figures 2A–F: simplex regression, Wald, flow, gene alignment
│    - Writes sections: "Figure 2A-C", "Figure 2D", "Figure 2E", "Figure 2F"
│
└─ main()
     1. Phase 1 → returns adatas + training results
     2. Drift / Stability QC panel (build_drift_qc_panel)
     3. Phase 2
     4. Phase 3
     5. report.save(REPORT_PATH)
```

### 3.1 Phase 1 — training

- Data: reads two h5ad files from `data/paper_part1/`.
- Optional stratified subsample via `_stratified_subsample` (stratify by
  `cell_type`). Controlled by `SUBSAMPLE_FRACTION` at the top of the
  file. **r13 is running with 0.2 for prototyping speed; production
  runs should set this back to 1.0.**
- Hyperparameter search: `pc.tl.hyperparameter_search(...)` with
  - `n_archetypes_range=[2, 3, …]`, up to a K ceiling chosen by the script
  - `hidden_dims_options=[[64,128], [128,256], [256,128,64]]`
  - `inflation_factor_range=[0.75, 1.0, 1.25, 1.5]` (r12 shortened)
  - `use_pcha_init=False` at CV time, True at final-fit time
  - `max_epochs_cv=20`
- Final fit: `pc.tl.train_archetypal(...)` with `model_config={"manifold_weight": 0.005, "kld_weight": 0.01, "sparsity_weight": 0.0, "archetypal_weight": 0.9}`, `pcha_init=True`, `inflation_factor` and hidden dims from CV winner, K chosen as smallest K where CV R² ≥ 0.9.
- Writes: config card grid, hyperparameter search tables, final
  archetype coordinates, elbow curve, PCA 3-D with archetype positions.

### 3.2 Phase 2 — Figure 1

- Uses `res_hsc` training history for the convergence narrative.
- Renders the "introduction" figure set: PCA projection with archetype
  positions, weights heatmap sorted by K-cluster, archetype R² tracker.

### 3.3 Phase 3 — Figure 2

The biggest phase (~100 min of a 130 min total run). Sub-figures each
render an independent `report.add_section`:

| Section | Content |
|---------|---------|
| **Fig 2A–C** | Simplex regression (HSC + CMP), per-degree dotplots (deg1 per-archetype, deg2 per-interaction-pair, deg3 summary table), pattern classification, nesting table, UpSet plot, permutation null |
| **Fig 2D** | Within-HSC Wald contrasts, R² vs FDR scatter, volcano plots for all pairs, cross-fit β-based Spearman, **cross-fit per-degree R² Spearman** (r12-item-9) |
| **Fig 2E** | Flow HSC → CMP: full-population Markov transition (r12 removed the k=10 variant), raw pairwise DataFrame with FDR, Sankey, permutation-curve null |
| **Fig 2F** | Per-pair flow analysis (capped at 20 significant pairs): gene alignment, Jacobian expansion, tricolor scatter, absence plot, lollipop, flow-associated pathways (r13 new — see §3.4), pseudotime × expansion / pathway traces |

### 3.4 Per-pair Fig 2F anatomy (reusable template)

This loop is the backbone for per-pair biology. Each iteration:

```python
for hi, ci in significant_pairs[:MAX_SIG_PAIRS]:
    # 1. Re-run flow_within() on the source-bin cells of this (hi, ci)
    #    pair to get transported PCA + per-cell model (fr_pp).
    fr_pp = pc.tl.flow_within(adata_full, source_mask=src_pair_mask_2f, …)

    # 2a. Gene-level: alignment + Jacobian + per-cell expansion
    align_pp = pc.tl.flow_gene_alignment(adata_full, fr_pp, per_cell=False, …)
    jac_pp   = pc.tl.flow_jacobian(adata_full, fr_pp, fr_pp["model"], per_cell_features=True, …)

    # 2a-i. Permutation curve for this pair
    # 2a-ii. Tricolor gene scatter
    # 2a-iii. Absence plot
    # 2a-iv. Lollipop chart
    # 2a-v. Pseudotime × expansion (top genes)
    # 2a-vi. Seurat ridgeplot

    # 2b. Pathway-level: flow-associated pathway Spearman (r13)
    #     Per-cell AUCell pathway score vs signed flow + |flow|.
    #     Pseudotime × pathway score trace for top 8 by |ρ_signed|.

    # 2c. Summary metrics (W2 before/after, n_sig_genes_FDR05)
```

Key invariants (r13 bug taught us these):

1. **`fr_pp["source_mask"]` is sized for `adata_full`**, not any subset. If you subset, always subset using `adata_full.obs_names[fr_pp["source_mask"]]`.
2. **`adata_full` and `adata_hsc` may have different cell counts** when subsampling is on. If a feature matrix (e.g. `adata_hsc.obsm["pathway_scores"]`) only exists on the smaller adata, you must map source cell names → positional indices in the smaller adata and drop non-intersecting cells before aligning rows.
3. **Per-cell flow vectors**: `disp = fr_pp["transported"] - adata_full.obsm[pca_key][fr_pp["source_mask"]]` gives `(n_source, n_pcs)`. `flow_mag = ||disp||`, `flow_signed = disp @ mean_dir`.

---

## 4. Prep pipeline

`prep_hsccmp.py` and `prep_bigov.py` follow the same 7-step recipe:

```
1. Load .h5ad (logcounts from the raw dataset)
2. Scrublet filter (threshold 0.25)
3. ENSG → gene symbols (if needed)
4. apply_mt_rb_mad_filter(adata, n_mads=3.0)      # from _paper_part1_prep
5. sc.pp.pca(adata, n_comps=N_PCS, zero_center=False, use_highly_variable=False)
   #   HSC: 13 PCs  (validated scree elbow)
   #   OV : 11 PCs
   #   ALL genes, unscaled logcounts. DO NOT sc.pp.scale() — destroys convex hull.
   #   DO NOT subset to HVGs before PCA — kills the archetypal signal.
6. 80/20 stratified holdout split by cell_type (HSC) or tissue (OV)
7. Save paper_part1/*.h5ad
```

**PCA choices are load-bearing**: scaling or HVG-subsetting before PCA
concentrates data into a spherical cloud, making archetypal hulls
meaningless. This was the r10 → r11 discovery. Copy this recipe
verbatim for Part 2.

---

## 5. Config surface (what to tweak when forking)

| Variable | File | Default | What to tune |
|----------|------|---------|--------------|
| `SUBSAMPLE_FRACTION` | `run_paper_part1_*.py` | `0.2` (r13 prototype) / `1.0` (production) | Stratified cell subsample before training. Keep 0.2 for fast iteration; set 1.0 before promoting. |
| `MAX_EPOCHS_FINAL`   | `run_paper_part1_*.py` | `200` | Upper bound for final-model training. CMP hit the cap in r13; bump to 300-400 for convergence margin. |
| `EARLY_STOP_PATIENCE`| `run_paper_part1_*.py` | `15` | Consecutive non-improving val checks before early stop. |
| `model_config.manifold_weight` | `run_paper_part1_*.py` | `0.005` (r12) | Penalty for archetypes drifting away from data hull. Sweep 0.001–0.05 if hull fit is poor. |
| `model_config.kld_weight`       | `run_paper_part1_*.py` | `0.01` | KLD floor; prevents encoder variance blowup. |
| `model_config.archetypal_weight`| `run_paper_part1_*.py` | `0.9` | Archetypal loss relative weight. |
| `N_PCS` | `prep_*.py` | 13 (HSC) / 11 (OV) | Number of PCs. Validated by scree elbow. Do not auto-select. |
| `MT_RB_N_MADS` | `prep_*.py` | `3.0` | 3-MAD cell filter cutoff. |
| `HOLDOUT_FRACTION` | `prep_*.py` | `0.20` | Train/test split. |
| Hyperparameter grid | `phase1_train_models` | see §3.1 | Narrow if prototyping; widen for production. |
| `MAX_SIG_PAIRS` | `phase3_figure2` (Fig 2F) | `20` | Cap on per-pair flow analyses. Each pair adds ~2-5 min. |
| `fdr_threshold` | everywhere | `0.05` | BH FDR cutoff. r11 tightened from 0.10. |
| `exclusive_threshold` | dotplot calls | `2.5` | max|β| / second-max|β| ratio for archetype-exclusive features. r11 tightened from 1.5. |

---

## 6. Report model (HTMLReport)

Each `run_paper_part1_*.py` inlines its own `HTMLReport` class (~90 LoC).
Single-file HTML output with base64-embedded PNGs and inline Plotly.
Core methods:

- `report.add_section(title, html, step_num=None)` — appends a
  `<details>` block. First three sections are open, rest collapsed.
- `report.fig_to_img(fig, caption, dpi=150)` → matplotlib Figure → PNG.
- `report.plotly_to_div(fig, caption)` → Plotly figure → inline div.
- `report.df_to_html(df, caption, max_rows=50)` → styled DataFrame.
- `report.text(str)` → `<p>…</p>` wrapper.
- `report.save(path)` → writes full HTML to disk.

Every section wraps its body in `try: … except Exception as e: html +=
error_html(f"...: {e}")` so a single failure doesn't abort the run.
When forking, preserve this pattern.

---

## 7. Run commands + rev numbering

### Prep (run once per dataset)

```bash
conda run -n archetype python scripts/prep_hsccmp.py
conda run -n archetype python scripts/prep_bigov.py
```

Writes into `data/paper_part1/`.

### Main run

```bash
# Preferred: unbuffered stdout so the log streams live.
# (r13 taught us: conda run + > logfile fully-buffers stdout.)
conda run -n archetype python -u scripts/run_paper_part1_hsc.py \
  > outputs/paper_part1/run_log_$(date +%Y%m%d)_r13.txt 2>&1 &
```

Or set `PYTHONUNBUFFERED=1` before launch. Without one of these, the
log file stays empty until the Python process exits.

**Rev numbering**: `REPORT_PATH` in the main scripts is:

```python
_DATE_TAG = time.strftime("%Y%m%d")
_existing = sorted(glob.glob(f"part1_report_{_DATE_TAG}*.html"))
_REV = len(_existing) + 1
REPORT_PATH = f"outputs/paper_part1/part1_report_{_DATE_TAG}_r{_REV}.html"
```

So same-day re-runs auto-increment `_rN`. Log filenames are chosen
manually by the launcher (convention: keep them numbered by major
iteration, e.g. `run_log_20260413_r13.txt`).

### Diagnostic scripts

```bash
conda run -n archetype python scripts/run_manifold_sweep.py   # manifold_weight grid
conda run -n archetype python scripts/run_loss_sweep.py       # kld × sparsity
conda run -n archetype python scripts/run_fit_diagnostic.py   # full-data fit
conda run -n archetype python scripts/run_viz_prototypes.py   # viz sandbox
```

---

## 8. Tests

`tests/test_paper_part1_fixes.py` is the regression harness. It does
both:

- **Code parses** (synthetic data training checks, e.g. `test_training_with_kld_and_manifold`).
- **Script structure** (regex-based sanity checks on `run_paper_part1_*.py`: section titles, config values, required call sites).

When refactoring, run:

```bash
conda run -n archetype python -m pytest tests/test_paper_part1_fixes.py -q
conda run -n archetype python -m pytest tests/test_core/test_archetype_correspondence.py -q
```

Skipped tests to know about:

- `test_subsample_fraction_is_full` — skipped while `SUBSAMPLE_FRACTION=0.2` for prototyping.
- `test_report_section_parity` — skipped while HSC iterates ahead of OV.

---

## 9. Forking into Part 2

Recommended template for `run_paper_part2_*.py`:

1. **Copy `run_paper_part1_hsc.py` to `run_paper_part2_hsc.py`** (or
   whatever dataset Part 2 uses). Keep the HTMLReport class and helper
   imports intact.
2. **Keep Phase 1 (training) as-is** if you're reusing the same
   trained models. Otherwise fork the CV grid and final-fit config for
   your dataset.
3. **Replace Phase 2 + Phase 3** with your Part 2 figure definitions.
   Each figure = one `report.add_section(title, html, step_num=...)`.
   Wrap every block in `try/except` so one failure doesn't abort.
4. **Reuse `_paper_part1_viz.py`** wholesale — don't fork it. Add new
   figure helpers *there* so both Part 1 and Part 2 can share them.
   Name them `build_<figure>_<purpose>()`.
5. **Reuse `_paper_part1_prep.py`** wholesale. If Part 2 needs a new
   dataset, write `prep_part2_<dataset>.py` that imports
   `apply_mt_rb_mad_filter` and writes into a new subfolder under
   `data/` (e.g. `data/paper_part2/`).
6. **Update `REPORT_PATH`** to point at `outputs/paper_part2/` so the
   two runs don't cross-contaminate.
7. **Add matching tests** under `tests/test_paper_part2_*.py` mirroring
   the structural checks in `test_paper_part1_fixes.py` (section
   parity, config values, presence of required call sites).

### Filter with the curated stress-gene list

New as of 2026-04-14: the `stress_genes/` folder contains
`stress_signatures.json` and `stress_genes_flat.txt` (529 unique
symbols across HSR / OSR / UPR / HySR / DDR). To filter them out in
Part 2 prep:

```python
from stress_genes.load_stress_genes import STRESS_GENES_FLAT
adata = adata[:, ~adata.var_names.isin(STRESS_GENES_FLAT)].copy()
```

Or for a per-signature pathway score, use `STRESS_SIGNATURES`
(`{HSR: [...], DDR: [...], ...}`).

---

## 10. Gotchas that bit us in r1–r13

- **Stdout buffering**: `conda run ... > logfile 2>&1 &` fully-buffers
  stdout. Log stays 0 bytes until the process exits. Launch with
  `python -u` or `PYTHONUNBUFFERED=1`.
- **`.reverse()` vs `[::-1]`**: `hidden_dims.reverse()` mutates the
  caller's list; cross-CV-fold interference. Always use `[::-1]` to
  copy. Fixed in commit `90e98ba`.
- **`adata_full` ≠ `adata_hsc`** when subsampling is on. Boolean masks
  from `fr_pp["source_mask"]` are `adata_full`-sized; feature matrices
  from `adata_hsc.obsm[...]` are `adata_hsc`-sized. Aligning them
  requires an `obs_names` round-trip. Bit us in r13 on
  flow-associated pathways.
- **`sc.pp.scale()`** before PCA destroys the convex hull. Do not
  scale. Do not subset to HVGs before PCA either.
- **`manifold_regularization_loss`** must operate on *effective*
  archetype positions (post-transform), not raw `self.archetypes`.
  Fixed earlier; still the single most impactful training-time change.
- **CMP hitting epoch cap**: 200 epochs is tight for CMP. When `status
  == NON_CONVERGED_HIT_CAP`, bump `MAX_EPOCHS_FINAL` to 300–400.
- **Degree-3 has no per-triple coefficients**:
  `_comprehensive_degree_comparison` stores only R² / Δ R² / FDR for
  degree 3 — no interaction-triple matrix. Any per-archetype or
  per-triple dotplot at degree 3 is misleading; use a summary table
  instead (r12 decision).
- **Drift badge**: `_is_converged` uses **median** of last 10 drift
  values (not mean) so transient spikes early in training don't drag
  the flag (r12).
- **FDR threshold**: 0.05 is project-wide. Older sections may still
  reference 0.10 in strings or regexes — update as you touch them.
- **Exclusive ratio**: 2.5× (r11 tightened from 1.5×). Same caveat.

---

## 11. Parallel prototyping loop suggested protocol

When you set up the Part 2 loop:

1. Start with `SUBSAMPLE_FRACTION=0.2` for fast iteration (~2h/run).
2. Run in background with `python -u ... > log 2>&1 &`, then poll the
   log / process every 30–60 min. The `/loop` skill in Claude Code
   handles this well.
3. After each run, triage the report HTML, keep a running review
   document (like `r12-feedback.md`), and land fixes in batches.
4. Run `pytest tests/` after each batch. Most failures will be in the
   structural regex tests — update them to match new intent rather
   than rolling back the code.
5. Flip `SUBSAMPLE_FRACTION=1.0` only when the prototype is converged
   on structure + content. Full runs take ~8-10 h on local hardware;
   plan accordingly.

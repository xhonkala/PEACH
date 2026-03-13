# PEACH v0.5.0 Paper: Experimental Plan

**Working title**: Flow Shepherds and Wolves: Continuous Archetypal Geometry Reveals Stress Canalization Networks in Treatment Response

**Status**: Draft plan, 2026-03-04
**Paper type**: Methods + application hybrid
**Pre-print context**: v0.2 PEACH pre-print already published on HSC data

---

## Thesis

Binary archetype membership discards the continuous geometry that archetypal analysis computes. Simplex regression, flow matching, and temporal Jacobian profiling recover this geometry and reveal a new class of regulatory structure: **flow shepherds** (genes that canalize cell state transitions through bottlenecks) and **flow wolves** (genes that expand state diversity post-transition). Genes that transition from shepherd to wolf mark **tipping points** — bifurcations where cells commit to new state basins. In TNBC treatment response, the timing and identity of these tipping genes differ between responders and non-responders, and their baseline expression patterns predict treatment benefit.

---

## Dataset

**Primary application**: TNBC cohort
- ~50 patients
- 3 treatment timepoints: baseline, pembrolizumab, radiotherapy
- 2 response labels: responder (R), non-responder (NR)
- Modalities: scRNA-seq, spatial CODEX, TCR (TCR deprioritized for this paper)
- Cell compartments: tumor cells, immune microenvironment (analyzed independently)

**Holdout design**: 15 patients reserved for prediction validation (stratified by R/NR). All method development and discovery on remaining ~35 patients.

**Methods demo**: hsc_10k.h5ad (existing PEACH test data) for quick HSC validation figure.

---

## Experimental Steps

### Phase 0: Data Preparation and Archetype Fitting

**Step 0.1**: Load and QC TNBC scRNA-seq data. Separate tumor and immune compartments.

**Step 0.2**: PCA on combined data (all patients, all timepoints) to establish a shared embedding space. This is critical — all downstream flow matching requires a common PCA space.

**Step 0.3**: Fit archetypes on each compartment.
- **Design A (main)**: Single archetype fit per compartment across all patients/timepoints. Flow matching by condition within shared archetype space.
- **Design C (supplemental)**: Separate archetype fits split by condition. Use `flow_between()` + archetype correspondence.
- Run hyperparameter search (`pc.tl.hyperparameter_search`) to select K for each compartment.

**Controls**:
- (+) Archetype R^2 should exceed 0.7 for chosen K
- (+) Known marker genes should segregate to biologically interpretable archetypes
- (-) Permuted expression matrix should yield low R^2 and no interpretable archetype structure

---

### Phase 1: Simplex Regression — Continuous Feature-Archetype Associations

**Step 1.1**: Run `pc.tl.feature_simplex_regression()` on tumor and immune compartments. Both degree 1 (linear) and degree 2 (with interactions).

**Step 1.2**: Run `pc.tl.classify_feature_patterns()` to categorize genes into archetype-exclusive, gradient, shared, antagonistic, ridge, valley, and flat patterns.
+ **classification patterns currently underspecified**
	+ may need to see some prelim results here to get a sense for how they look before having a more nuanced opinion here

**Step 1.3**: Run `pc.tl.archetype_driver_regression()` (flipped) on pathway scores to identify geneset interactions driving archetypal specialization.

**Controls**:
- (+) Known lineage markers (e.g., CD8A for cytotoxic T cells) should classify as archetype-exclusive with high R^2
- (+) Housekeeping genes should classify as flat/ubiquitous with low R^2
- (-) Permuted weight vectors should yield uniformly low R^2 and no significant coefficients
- (-) Bootstrap CIs on permuted data should be wide and centered on zero

**HSC demo (Figure 2)**: Run simplex regression on HSC data. Show a gene that Wilcoxon calls "differentially expressed in archetype 3" but simplex regression reveals as a gradient across archetypes 2-3-5 with a significant interaction term for the 3-5 blend zone. Concrete demonstration that binary membership misses continuous structure.

---

### Phase 2: Flow Matching — Cross-Condition Transport

**Step 2.1**: Define flow pairs for Design A:
- Within each compartment, split cells by timepoint:
  - base -> pembro (treatment initiation)
  - pembro -> RT (treatment escalation)
  - base -> RT (full treatment arc, supplemental)
- Split each flow by response group: 6 flows per compartment (3 transitions x 2 groups)

**Step 2.2**: Train flow models (`pc.tl.flow_within()`) for each pair. 1000 epochs, GPU. Reduce epochs if needed.

**Step 2.3**: Validate flows.
- Compute MMD before and after transport — significant reduction expected
- Run `pc.tl.flow_significance()` permutation test (100 permutations, 200 epochs each) on each flow. Profile how long 1 permutation takes and reduce epochs if needed.

**Step 2.4**: Run `pc.tl.flow_gene_alignment()` at t=0.5 for initial gene ranking.

**Controls**:
- (+) MMD after transport should be significantly lower than before (flow learned real structure)
- (+) Known treatment-response genes (e.g., PD-L1 pathway for pembrolizumab) should appear in top aligned genes
- (-) Flow trained on permuted condition labels should show no MMD improvement
- (-) Flow between two random subsets of the same condition should show minimal transport (no real signal)

---

### Phase 3: Flow Shepherds and Wolves — Temporal Jacobian Profiling

**Step 3.1**: For each of the 6 flows (per compartment), evaluate temporal Jacobian:
```
t_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
pc.tl.flow_jacobian(adata, flow_result, t=t_values, n_subsample=1000, n_bootstrap=200)
```

**Step 3.2**: At each t, compute the three shepherd/wolf axes per gene:

| Axis | Symbol | Definition | Shepherd | Wolf |
|------|--------|-----------|----------|------|
| Directional alignment | A_g(t) | dot(PCA_loading_g, mean_velocity(t)) via `flow_gene_alignment(t=t)` | A_g > 0 | A_g > 0 |
| Bottleneck association | B_g(t) | -spearman_corr(expression_g[subsampled], jacobian_det[subsampled]) across n_subsample cells | B_g > 0 (high expr in contracting regions) | B_g < 0 (high expr in expanding regions) |
| Canalization | C_g(t) | -feature_expansion_g from `flow_jacobian()` (sign-flipped Jacobian projection onto PCA loading) | C_g > 0 (variance collapsing) | C_g < 0 (variance expanding) |

**Composite scores** (z-scored per axis across all genes at each t):
```
Shepherd_g(t) = z(A_g) + z(B_g) + z(C_g)       # all three oriented so positive = shepherd
Wolf_g(t)     = z(A_g) + z(-B_g) + z(-C_g)      # alignment same, bottleneck/canalization flipped
```

**Conjunctive requirement**: a gene is classified shepherd only if ALL THREE individual z-scores > 0 (not just the sum). Same for wolf. This prevents a gene with enormous alignment but anti-canalizing behavior from being called a shepherd. A gene can be neither (most genes) but never both simultaneously.

**Step 3.3**: Classify each gene at each time point: shepherd, wolf, or neutral (fails conjunctive gate).

**Step 3.4**: Identify **tipping genes** — genes that transition shepherd -> wolf across flow time. Record tipping time t* for each.

Tipping detection: for each gene, track B_g(t) and C_g(t) across t. Tipping time t* = the t where both B_g and C_g cross zero (contracting → expanding). Operationally:
- Fit a sigmoid or find the zero-crossing of a smoothed B_g(t) trajectory
- Require that the gene passes the conjunctive shepherd gate for at least 2 consecutive t values before t*, and the conjunctive wolf gate for at least 2 consecutive t values after t*
- Genes that are shepherd across ALL t values = **full canalizers** (interesting in their own right)
- Genes that are wolf across ALL t values = **full dispersers**
- Genes that tip = **bifurcation markers**

**Step 3.5**: Compare shepherd lists, wolf lists, and t* distributions between R and NR flows.

**Controls**:
- (+) Bootstrap CIs on Jacobian-derived gene scores should be tight for top-ranked shepherds/wolves (robust to cell subsampling)
- (+) Shepherd genes should enrich for known stress response / canalization pathways (DDR, UPR, autophagy)
- (+) Wolf genes should enrich for differentiation / state diversification pathways
- (-) Genes classified as flat/ubiquitous in simplex regression (Phase 1) should not appear as shepherds or wolves (no archetype association = no flow role)
- (-) Tipping genes identified from permuted condition labels should not reproduce the real t* distribution

---

### Phase 4: Sequential Validation via CellRank

**Step 4.1**: For each identified tipping gene, fit CellRank connectivity kernel and compute pseudotime (`pc.tl.compute_lineage_pseudotimes()`).

**Step 4.2**: Fit GAM of each tipping gene's expression along pseudotime. Record GAM peak location along normalized pseudotime.

**Step 4.3**: Correlate GAM peak position with flow tipping time t*. Two independent methods should agree on temporal ordering. Calculate first derivatives at each time t and compare: should help ID sharpest slope. Second derivative should ID if there are multiple peaks/valleys.

**Step 4.4**: Compare GAM coefficient profiles and peak ordering between R and NR.

**Controls**:
- (+) GAM peak position and flow t* should show significant positive correlation (Spearman rho > 0.5) for tipping genes
- (-) Non-tipping genes (pure shepherds or pure wolves) should not show the same correlation pattern
- (-) GAM peaks on permuted pseudotime should not correlate with t*

---

### Phase 5: Spatial Co-Localization (CODEX)

**Step 5.0 (prerequisite)**: Inspect CODEX antibody panel. Determine which flow shepherd/wolf genes have corresponding protein markers. This gates how deep the spatial analysis can go.

**Step 5.1**: Run `pc.tl.archetype_gradient_colocalization()` on CODEX data. For each archetype pair with a clear gradient in tumor cells, test which immune cell types are enriched in spatial neighborhoods of high archetype spread.

**Spatial gradient estimation**: Compute discrete gradient of archetype weights on the spatial neighbor graph. For each edge (cell_i, cell_j) in the squidpy spatial graph, delta_w = w_j - w_i per archetype. Average over local neighborhoods to get a smoothed per-cell gradient vector. Gradient magnitude |nabla w| identifies cells at the steepest part of the archetype transition. Test whether specific immune cell types are enriched near high-gradient-magnitude cancer cells.

Open question: full spatial front detection (Sobel/Canny on rasterized weight surfaces) deferred to v0.6. The graph-based discrete gradient is sufficient for co-localization testing.

**Step 5.2**: For significant co-localization hits, run `flow_within()` on tumor cells between those two archetypes. Cross-reference flow shepherds with ligand-receptor databases (CellPhoneDB, NicheNet).

**Step 5.3**: Nominate interaction candidates: genes that are flow shepherds in tumor cells AND whose cognate ligands are expressed by the co-localized immune cell type.

**Controls**:
- (+) Known spatial interactions (e.g., PD-L1/PD-1 between tumor/T cells) should appear in co-localization results
- (+) Nominated ligand-receptor pairs should have higher spatial co-expression than random gene pairs
- (-) Permuted spatial coordinates should abolish co-localization enrichment
- (-) Cell types with no biological role in tumor regulation (e.g., erythrocytes) should not show enrichment

**Scope note**: Depth of this phase depends entirely on CODEX panel coverage. If panel doesn't cover key shepherd genes, this becomes a supporting analysis ("spatial proximity supports the computationally nominated interaction") rather than a main figure.

---

### Phase 6: Baseline Prediction — The Clinical Punchline

**Step 6.1**: From Phases 3-5, compile the final candidate gene set:
- Top tipping genes with different t* in R vs NR
- Shepherd genes with R/NR differential composite scores
- Spatially-validated interaction partners (if available from Phase 5)

**Step 6.2**: Prediction via flow-field projection (preferred approach):
1. For each held-out patient's baseline cells, compute archetype weights using the fitted model
2. Project baseline cells into the learned R and NR flow fields from the discovery cohort
3. Evaluate where each flow predicts these cells would move (transport to t=1.0)
4. Compare predicted endpoint distributions to actual R and NR target distributions (MMD or KL divergence)
5. Prediction: patient is predicted R if their cells' predicted trajectory is closer to the R target distribution
6. For dual-compartment prediction: compute tumor and immune flow projections independently, combine (e.g., patient is predicted R if BOTH compartments' flows point toward response, or weighted average)

**Step 6.2b** (fallback): If flow-based prediction lacks power at N=15, fall back to elastic net on baseline expression of candidate genes (shepherd/tipping gene set from Phases 3-5). Interpretability over performance.

**Step 6.3**: Evaluate on **held-out 15 patients**. Report AUROC, AUPRC, and calibration. Report honest CIs given small N.

**Step 6.4**: Compare predictive features to existing pembrolizumab response signatures in the literature. Are we recovering known biology (validation) or finding new markers (discovery)?

**Step 6.5**: Repeat prediction from tumor cells and immune cells independently. Does one compartment predict better? Does combining them improve?

**Controls**:
- (+) AUROC on held-out patients should exceed 0.65 (above chance, acknowledging small N)
- (+) Top predictive features should overlap with flow shepherd/tipping genes (the biology-driven features should be the predictive ones)
- (-) Classifier trained on random gene sets of the same size should perform at chance (AUROC ~0.5)
- (-) Classifier trained on permuted R/NR labels should perform at chance
- (-) Leave-one-out cross-validation on discovery set should show stable feature selection (same genes appearing across folds)

---

## Figure Plan

### Main Figures

| Fig | Content | Key message |
|-----|---------|-------------|
| 1 | Methods schematic: simplex regression + flow matching + shepherd/wolf/tipping concept | "Here's what we built and why" |
| 2 | HSC demo: simplex regression reveals continuous structure Wilcoxon misses | "Binary membership wastes information" |
| 3 | TNBC archetype landscape: tumor + immune archetypes, simplex regression gradients, pattern classification | "Continuous associations reveal richer biology" |
| 4 | Shepherd and wolf gene maps across treatment flows, R vs NR | "Flow geometry identifies genes that control state transitions" |
| 5 | Tipping point analysis: shepherd->wolf transitions, t* distributions, CellRank GAM validation | "Bifurcation timing differs between R and NR" — centerpiece |
| 6 | Baseline prediction: held-out validation + spatial interaction candidates | "These regulatory circuits are visible before treatment and predict response" |

### Supplemental

- Design A vs C archetype fitting comparison
- Spatial CODEX co-localization details (or main figure if panel coverage is strong)
- GMM sub-archetype populations
- All permutation/bootstrap validation panels
- Full gene lists and pathway enrichments
- Flow training diagnostics (loss curves, MMD)
- Sensitivity analyses (varying K, varying t* detection thresholds)

---

## Key Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| CODEX panel doesn't cover shepherd genes | Spatial capstone weakened | Fall back to spatial proximity of cell types + scRNA ligand-receptor inference |
| 50 patients too few for robust prediction | Weak holdout AUROC | Use elastic net with strong regularization; report honest CIs; frame as "nominated candidates" not "validated biomarker" |
| Flow shepherds don't differ between R and NR | No clinical story | Pivot to "shepherds are conserved, wolves differ" or "tipping timing differs even if gene identity is shared" |
| Temporal Jacobian profiling is noisy | Tipping times unreliable | Bootstrap CIs; require CellRank GAM concordance as independent validation |
| Stress response hypothesis doesn't pan out | Biological framing weakened | The methods contribution stands regardless; reframe around whatever pathway structure emerges |
| Compute cost of 12 flows x 9 time points x bootstrap | Wall time | GPU-accelerate; subsample to 1000 cells; reduce bootstrap to 100 if needed |

---

## Scope Boundaries

**In scope for this paper**:
- Simplex regression (forward + flipped)
- Pattern classification
- Flow matching (within-model, Design A)
- Temporal Jacobian profiling with shepherd/wolf/tipping classification
- CellRank GAM validation of tipping times
- Spatial co-localization (depth depends on CODEX panel)
- Baseline prediction holdout

**Out of scope (future work)**:
- TCR analysis
- Intra-patient heterogeneity decomposition
- GMM sub-archetype populations (supplemental only)
- Design C flow_between() (supplemental comparison only)
- Dirichlet mixture models
- Higher-order Scheffe polynomials
- Riemannian flow matching on the simplex
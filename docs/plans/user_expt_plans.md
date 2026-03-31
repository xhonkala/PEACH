### RNRflow Experiment Planning

20260317
+ overarching hypotheses
	+ method hypothesis: current uses of archetype analysis in the literature leave information on the table and lack a principled way to compare conditions, makes it much harder for other groups to pick up the tool and use it for their projects
		+ information left on the table = continuous distances from archetypes and tradeoffs between them
			+ from initial Alon papers to recent Krishnaswamy papers, they're using 1-vs-all Wilcoxon rank sum tests to characterize archetypes and manually curating phenotypes--not practical for applied use
		+ comparing between conditions is a dual problem: there's comparing conditions within a single fit and there's comparing conditions between fits
		+ more of a field-based hypothesis on tool usage and utility
		+ solution is to develop additional methods to close these gaps and bring archetype analysis method coverage up to that seen in traditional clustering and DEG analyses like Seurat or Scanpy
			+ simplex regression/Scheffé polynomials = find features that have strong associations with one or more archetype distances simultaneously
				+ 1st term = linear interactions with archetypes
				+ 2nd term = feature interactions
				+ can go higher as well
			+ that then sets up Wald contrasts on the simplex regression coefficients to do principled pairwise comparisons of feature differences
			+ then there's the question: do different subgroups near an archetype have different tradeoff levels that a simple distance-based bin runs over? GMM decomposition for that
			+ how to compare conditions within a fit or between fits? flow matching with Sinkhorn so that source and pair cells are directly matched and we get a specific flow field, not an averaged one
			+ finally having a flow field where we can calculate how much a given gene's expression is aligned with the flow field scores, we can get out which genes are expanding and which are contracting via "timepoint" Jacobians
				+ want to at least test, may be out of scope: setting up a trajectory-spanning gene network based on flow connectivity: calculate Jacobian at multiple timepoints/distances between source and target and measure gene contribution/alignment at each point to rank how each gene contributes to the learned flow
				+ will need to note that this is a learned *hypothetical* flow that we can't say is real real unless we have transitional cells to validate, but we can at least hold out some fraction of cells from source and then transport them in the field afterwards to measure MMD against the field quality
	+ biological hypothesis -- thinking through this in a per step of the paper perspective: **NB focus first on TNBC cells, leave TME until later to avoid running into the multiplicity of cell types**
		+ overarching generalized 1: archetype analysis reveals biological features, such as specialization gradient or tradeoffs, that are invisible in standard clustering + DEG workflows
			+ *upon review:* strike this one, it's preemptive defense and I don't need to justify using these tools--including it would invite a whole set of criticisms and detailed comparisons to standard clustering to do
		+ overarching generalized 2: stress response regulators will be significantly overrepresented in flow fields between response timepoints and R vs NR will segregrate into distinct stress response regimes where R will have to traverse a greater phenotypic distance between timepoints than NR, whose stress response regimes will be more chaotic
			+ corollary: NR stress response regimes are visible in pre-tx timepoints in a minority of cells, showing selection of pre-existing stress response capacity through treatment
			+ cocorollary: these stress response segregation patterns will be visible in immune cells as well $\to$ *paper 2*
			+ cococorollary: we'll see some spatial co-occurence of interacting stress response regimes between cancer cells, TC, and myeloid cells based on archetype-associated enriched markers (since CODEX data is kinda shallow) $\to$ *paper 2*
	+ OK then that leads into some per-technique biological hypotheses that I can pre-register--overall on TNBCrad data: hold out some patients from each arm (R and NR) and then cap the paper by predicting their response conditional from how they fit into the models (which will already be trained and which all need to be saved as I go, includes the PEACH models, the regression models, the GMM models, and the flow models))
		+ on HSC data mainly need to show that simplex regression interaction terms capture some HSC TF fate regulators, that a GMM component is enriched for fate commitment markers, and that that component flows towards a 2nd cell type with flow-aligned genes that recap those TF fate regulators
			+ *upon review:* strike this too
		+ then getting into R vs NR in TNBC focus on cancer cells first with simplex regression
			+ if there is within-fit dose association, can compare first- and second-order features and their differences in response arms: can start pre-registering associations here as major flow-aligned genes for later
			+ introduce Wald contrasts here
		+ next I can get into R vs NR on GMM decomposition
			+ use GMM components and treatment associations in pre- and post-treatment timepoints to compare component similarity: can we find a pre-existing highly-related component before a treatment that expands or splits specialization in a post-treatment component? here can just use Spearman's correlation on feature enrichment or Wilcoxon rank sum
			+ also look for and feature components that are not in global centroid and poised between 2 archetypes: transitional populations actively navigating a tradeoff
		+ then there's flow matching within a fit and between fits
			+ flow-aligned genes vs flow-opposed genes, goodness of fit, and emphasize that this is a theoretical trajectory not an observed one--interpolating between observations
			+ bring in the held-out source cells here for validation of transport map quality
			+ can do a head-to-head of within vs between, will matter more if each treatment timepoint is better fit by a different k archetype set
			+ **this may be enough for the paper**
			+ do need to benchmark this against just comparing the significant features found by simplex regression (honestly should include that as a method as well, that's somewhere where the Wilcoxon rank sum test actually is the appropriate test)
		+ next there's introducing the Jacobian term on flow: there's Jacobian on the flow field itself calculated across all genes and all cells that are being transported but it can also be broken out per cell, which then introduces the per-gene or -geneset expanding and contracting idea
			+ show that we can ID top expanding genes and show how they map to soft assignments between archetypes--will require doing flow matching between archetype pairs across fits, particularly interesting if some highly expanding genes are associated with a pre-tx archetype flowing mostly to 2+ different post-tx archetypes in taking on a new specialization
			+ same for contracting genes
		+ finally can do the multi-timepoint Jacobian and the network we can derive from it
			+ simplest way is counting flow alignment/blocking at each timepoint
			+ more complicated is calculating peak widths or necks and working backwards into a betweenness score for each associated gene

20260318
+ thinking about experimental flow here
+ 3 different paths that I can see here, not necessarily mutually exclusive:
	1) pure methods + a dataset vignette
	2) top-down stepwise TNBCrad RNRflow analysis: introduce features and what can be recovered from them
	3) predictive power flex: mostly blinded analysis forward pass through methods, then treat unblinding as the dependent variable in a logistic regression model for different feature sets from the new methods
	4) can hold out some patients from top-down stepwise approach to get some predictive power
	5) predictive power flex runs into some tricky framing issues that I've been chewing on all evening of how to do progressive unblinding for different categoricals (mainly timepoint and response) with refitting and comparison, very possible that the story winds up muddled quickly
+ let's outline what the narrative flow for options 4 and 5 would look like
	+ top-down stepwise + holdout prediction
		+ talked it out with Angelica somewhat
		+ think that this is the way to go--pre-register that NR will maintain greater heterogeneity, particularly in stress-related pathways, throughout treatment
			+ more storytelling needed in introducing new methods to explain how and why they work
			+ then the TNBCrad analysis becomes about showing what we can do with them
				+ so for example, simplex regression becomes
					+ needed a more principled way to figure out archetypal phenotypes than 1-vs-all Wilcoxon rank sum tests that throw away all distance data for a binary rank
					+ so introduces simplex regression
					+ identify flat features: disregard
					+ significant first-degree terms lead to archetype-exclusive phenotypes: works for both genes and genesets
					+ second-degree terms for nonlinear interactions
					+ can sort second-degree terms for cooperative gradients across multiple archetypes vs tradeoff fronts
					+ can also compare per-cell residuals from different fits to quantify difference between fits
				+ then show what R+NR TNBC phenotypes are like and introduce pairwise Wald contrast
				+ next use the reverse simplex regression for archetype drivers
				+ maybe I should exclude the GMM part from the paper itself? I think that depends on initial results: if a GMM component in a pre-tx timepoint is then selected for in the next timepoint, that's worth including--otherwise put it in Supplemental or just online tutorial
					+ also a question of how archetypal some components are: if there's strong enrichment there then that's worth including since the GMMs are formed from barycentric archetype weights anyway
				+ multi-timepoint analysis then for flow matching soft assignment
					+ lets us compare pre- and post-tx populations and which are most related
					+ brings back to the Wald contrasts on simplex regression parts to quantify how related archetypes are similar or differ across conditions--can we detect a pre-tx subpop that expands post-tx
					+ can qualify things like archetype or component heterogeneity and whether R have collapsed diversity relative to NR
				+ flow matching via Sinkhorn for learning the transport maps and genes in them by alignment
				+ lastly Jacobian terms on the flow matching for expanding vs contracting factors
					+ I think multi-timepoint Jacobian and temporal backbone GRN are beyond the scope of this paper
	+ blinded flex with progressive unmasking
		+ advantage that multiple different feature types can be compared for their predictive power is also a disadvantage: while it's cool that it's feasible, it's also a lot to explain and make understood by the audience
			+ pro in having lots of feature predictions is in hypothesis generation and deciding which level of result you want to use for further investigation, some freedom of choice
			+ con is in figuring out how to relate a simplex regression coefficient term or residuals comparison in terms of their predictive power
		+ let's think through all the different predictors that could be compared
			+ simplex regression coefficients: archetype loadings for features vs its flip in drivers for archetypes $\to$ these mainly become helpful after unmasking with the phenotype interpretation and separation of categoricals with archetypes via hypergeometric test
			+ simplex regression residuals: per-cell goodness-of-model-fit for simplex regression, look at distribution of R vs NR after unmasking
			+ Wald contrast $\chi^2$ distribution comparisons
			+ flow soft assignments through the fit and space $\to$ these come in after timepoint unmasking in contrasting the global model to the per-timepoint fit soft assignments
			+ flow matching feature alignment, separation of features by unmasked conditional
			+ flow matching Jacobians per feature, expansion vs contraction profiles/distributions after unmasking
			+ temporal sequences from Jacobian timepoints, again looking for separation
		+ overall this becomes more a paper about the underlying statistics than it is about the biology or what the methods mean for biology
		+ paper flow would be
			+ introduce the feature upgrades at a high level
			+ global blinded fit
			+ global simplex regression
			+ global Wald contrasts
			+ global flow soft assignment
			+ timepoint unmasking and treatment segregation
			+ flow between timepoints
			+ flow alignment features
			+ responder unmasking
			+ contrasting each of the above by responder status
			+ fitting in held out data
		+ I think I'm talking myself out of this: it's valid, it's interesting ML, and it might be a fine scope to tackle if I were willing to spend another year on it
			+ can always take this progressive-unblinding approach to another project later and run it across a few datasets to ask which predictive features generalize best for predicting treatment status


20260319
# End to End Prototype Analysis Flow

## Output format: rich HTML report
+ single self-contained .html file with embedded plotly figures and styled tables
+ collapsible sections per analysis step (HTML details/summary)
+ each section includes:
	+ section header with step number and title
	+ narrative summary (auto-generated text: what was run, key findings, QC flags)
	+ embedded visualizations (plotly for interactive, matplotlib for static → base64 PNG)
	+ result tables (styled pandas DataFrames → HTML)
	+ timing for each step
+ report also saves all intermediate results to output directory as .h5ad, .pkl, and .csv for downstream reuse
+ viz spec per step listed below with [VIZ] tags

+ scope:
	+ 10k myeloid cell subset all conditional data included
	+ work through each step in the top-down feature introduction & biological story analysis: goal is primarily to work through all analysis steps and data interpretation to iron out details before finalizing for later TNBC cell runs
	+ questions/interpretation to address:
		+ archetype feature characterizations: exclusive archetype-associated features vs interaction terms (co-drivers vs tradeoffs)
		+ separation of conditional variables in archetype space (hypergeometric tests on treatment and response): seeing this in a myeloid test set gives confidence in same being seen in TNBC cells
		+ how to report results from
			+ first- vs second-order simplex regression features
			+ differentiation of co-activation vs tradeoff second-order simplex regression features
			+ Wald contrasts: between which pairs, how to rank, and which features
			+ driver regression results
			+ comparison of simplex regression vs driver regression results--should theoretically be a broad overlap
			+ mixture model component stability
			+ mixture model component characterization: regression model results
			+ comparison of mixture model component characteristics to archetypes: similarity, which components are closest to which archetypes, which components are archetype-enriched, poised between 2 archetypes as a transitional population, or anchored to the global centroid as undifferentiated/non-specialized cells
			+ soft assignment pairwise matching and fit quality
			+ flow matching model QC
			+ flow matching gene or gene set alignment
			+ gene or gene set feature expansion vs contraction: magnitude, direction, alignment
1) dataset prep
	+ load and register what's in adata.obs and adata.var_names (should be gene symbols)
	+ already pre-processed, adata.X is logcounts
	+ go ahead and run pc.pp.load_pathway_networks let's use HALLMARKS and then pc.pp.compute_pathway_scores--might as well have that metadata already ready to go for the global object
	+ prepare dataset splits that will be used later--min split size 1000 cells
		+ global--no filter no split
		+ per-dose: baseline, PD1, RT
		+ per-response: R, NR
		+ per-response per-dose: for each R and NR, baseline, PD1, RT
		+ set up dataset and trained models output directory
	+ set up pc.pp.prepare_training for global--only load the ones being currently used to avoid memory pressure crashing with OOM
		+ swap later as needed
	+ [VIZ] dataset overview table: n_cells, n_genes, obs columns, split sizes
	+ [VIZ] split summary table: cells per dose, per response, per dose×response with min-cell-count flags
	+ [VIZ] pathway score distribution: histogram of pathway scores across cells for top 10 pathways by variance
2) global hyperparameter fit
	+ pc.tl.hyperparameter_search on k 3:9, hidden_dims [64,128], [128, 256], [128, 256, 512], and inflation factors 1.0, 1.5, and 1.75
	+ report results in run summary
	+ but select best results from slicing returned results to pass to training model
	+ set up train model with auto-selected best model config with pc.tl.train_archetypal
	+ report training metrics: QC, convergence, archetypal $R^2$, RMSE
	+ save trained model
	+ run annotation functions to save archetype-annotated AnnData: pc.tl.archetypal_coordinates, pc.tl.assign_archetypes, pc.tl.extract_archetype_weights
	+ [VIZ] CV results table: ranked configs with K, hidden_dims, inflation, mean R², SE (styled: best row highlighted)
	+ [VIZ] pc.pl.elbow_curve: R² vs K with error bars
	+ [VIZ] pc.pl.training_metrics: loss curve, R² convergence for selected model
	+ [VIZ] training QC summary table: final R², RMSE, n_epochs, convergence flag
	+ [VIZ] pc.pl.archetypal_space: 2D PCA scatter colored by archetype assignment
	+ [VIZ] pc.pl.archetype_positions: archetype positions in PCA space with cell density
	+ [VIZ] weight distribution: histogram of max archetype weight per cell (commitment vs hedging)
3) global simplex regression
	+ pc.tl.feature_simplex_regression for both genes and genesets, save covariance matrices, can use convenience functions if needed
	+ need to report simplex regression overall fit quality
	+ need to report per-feature coefficients and residuals for processing
		+ report should just include structure of results for now
		+ include summary statistics: % flat, % 1st-degree, % 2nd-degree
	+ pc.tl.classify_feature_patterns
		+ discard flat features
		+ per-archetype 1st-degree terms
		+ per-archetype 1st-degree *exclusive* terms
		+ top 2nd degree terms and their associations
		+ deconvolve 2nd degree terms by types at the archetypes they're involved in
			+ cooperative/co-activation: matching pattern of coefficients with archetypes (increasing or decreasing)--this is a new classification to prototype
			+ tradeoff: opposing pattern of coefficients with archetypes (one increasing and one decreasing)--this is a new classification to prototype
			+ report top per interaction pairs--will be compared to Wald contrasts later
	+ [VIZ] regression fit summary table: n_features tested, n_significant (FDR<0.05), median R², effective rank
	+ [VIZ] pc.pl.r2_barplot: top 30 genes by R², with per-archetype |β| breakdown
	+ [VIZ] pc.pl.coefficient_heatmap: top 50 genes, K columns
	+ [VIZ] pc.pl.regression_volcano: R² vs max contrast for genes, colored by pattern type
	+ [VIZ] pattern classification summary: stacked bar of flat/exclusive/interaction/structured (genes + pathways side by side)
	+ [VIZ] pc.pl.archetype_regression_dotplot: top 10 exclusive genes per archetype
	+ [VIZ] pc.pl.archetype_radar: one per archetype, top features as spokes
	+ [VIZ] interaction term table: top 20 2nd-degree terms with pair labels, β, FDR, co-activation vs tradeoff flag
	+ [VIZ] pc.pl.interaction_heatmap: degree-2 interaction coefficients
	+ [VIZ] pathway regression: repeat r2_barplot + coefficient_heatmap for pathway scores
4) global hypergeometric tests
	+ pc.tl.conditional_associations for which archetypes associate with which treatment timepoints, patients, and response categories
		+ this is a prelim test to see if cells from this dataset exhibit conditional separation in archetypal space
	+ summarize in report
	+ [VIZ] enrichment heatmap: archetypes × conditions (dose, response), colored by -log10(FDR), sized by odds ratio
	+ [VIZ] mosaic/proportion bar: cell proportion per archetype, stacked by condition
	+ [VIZ] significance table: archetype × condition with enrichment/depletion direction, OR, FDR
5) global Wald contrasts
	+ pc.tl.archetype_contrasts run all Wald tests between all pairs
	+ report return structure
	+ summarize top features across all pairwise contrasts
	+ compare pairwise contrasts to 2nd-degree simplex regression
		+ report overlapping features vs differentiating features
	+ Wald contrast QC metrics report
	+ [VIZ] pc.pl.contrast_volcano_grid: all K(K-1)/2 pairs as subplot grid
	+ [VIZ] top contrasts table: per pair, top 10 genes by |Δβ| with z-scores and FDR
	+ [VIZ] Wald vs interaction overlap: Venn or UpSet plot of top 50 Wald genes vs top 50 interaction genes per pair
	+ [VIZ] QC table: n_tests, n_significant per pair, median |z|, FDR distribution histogram
6) global within-fit comparisons and soft assignments
	+ pc.tl.archetype_mmd and pc.tl.archetype_feature_similarity
		+ contrast to pairwise pc.tl.flow_within results
	+ needs to include report heatmap and summary of MMD results vs Spearman's ρ
	+ [VIZ] pc.pl.mmd_heatmap: K×K within-fit MMD with p-values
	+ [VIZ] pc.pl.feature_similarity_heatmap: K×K Spearman ρ on regression β vectors
	+ [VIZ] side-by-side: MMD heatmap vs feature similarity heatmap (are distance-similar archetypes also feature-similar?)
	+ [VIZ] summary table: per-pair MMD, ρ, and concordance flag
7) global driver regression
	+ pc.tl.archetype_driver_regression
	+ only do this for global for now, go through results and interpretability before bringing to all the per-dose fits
	+ compare results to simplex regression results: feature overlap vs exclusive contrasts
	+ [VIZ] driver regression R² per ILR component: bar plot
	+ [VIZ] driver coefficients heatmap: top 30 features × K archetypes (back-transformed from ILR)
	+ [VIZ] simplex vs driver concordance: scatter of per-gene R² (simplex) vs per-gene max |β| (driver), Spearman annotated
	+ [VIZ] concordance table: top 20 shared features + top 10 simplex-only + top 10 driver-only
8) global mixture models
	+ pc.tl.feature_simplex_decomposition use Dirichlet backend
	+ report n components and stability
	+ visualize relative to archetypes in Euclidean space, not barycentric space
	+ [VIZ] pc.pl.gmm_bic_curve: BIC/ICL vs n_components
	+ [VIZ] pc.pl.component_stability: stability scores bar plot with threshold line
	+ [VIZ] pc.pl.component_scatter: 2D PCA colored by component assignment
	+ [VIZ] pc.pl.component_archetype_summary: 2×2 panel (sizes, weight profiles, proximity, entropy)
	+ [VIZ] component summary table: n_cells, dominant archetype, stability score, mean entropy per component
9) global component characterization and soft assignment
	+ pc.tl.component_regression -- is this even the right metric here anymore? these components aren't necessarily in a simplex anymore
	+ report per-component characterization
	+ re-use pc.tl.archetype_mmd and pc.tl.archetype_feature_similarity on components to compare them
		+ For MMD: filter adata by component, compute pairwise MMD on PCA coordinates directly via peach._core.utils.flow_matching.compute_mmd
		+ For feature similarity: run simplex regression per component (already have component_regression), then Spearman on the resulting coefficient vectors manually
	+ re-use pc.tl.conditional_associations for component-conditional associations
	+ component relatedness graph network
	+ [VIZ] pc.pl.component_heatmap: per-component top features from component_regression
	+ [VIZ] component-archetype mapping table: which component → which archetype(s), with weight centroid, distance to vertices, and classification (vertex-anchored / transitional / centroid)
	+ [VIZ] component MMD heatmap: pairwise MMD between components (computed via compute_mmd)
	+ [VIZ] component feature similarity heatmap: Spearman ρ between component regression β vectors
	+ [VIZ] component-condition enrichment heatmap: components × conditions (dose, response)
	+ [VIZ] pc.pl.component_neighborhood_graph: component relatedness network in PCA space
10) per-dose breakout hyperparameter fit
	+ repeat (2) for each dose split: baseline, PD1, RT
	+ train model with best config and save
	+ annotate AnnDatas as above in (2)
	+ [VIZ] per-dose K comparison table: selected K, R², hidden_dims for each dose vs global
	+ [VIZ] per-dose archetypal space panels: 3 side-by-side PCA scatters (baseline, PD1, RT) with archetype positions
	+ [VIZ] K stability assessment: does optimal K change across doses? table + note
11) per-dose simplex regression, hypergeometric tests, Wald contrast
	+ repeat (3)-(5) for each timepoint/dose
	+ run Wald contrasts between doses (how timepoints will be referred to henceforth)
	+ [VIZ] per-dose regression summary table: n_significant genes per dose, median R², pattern distribution comparison
	+ [VIZ] cross-dose feature stability: Spearman of per-gene R² across doses (are the same genes important at each timepoint?)
	+ [VIZ] per-dose Wald volcano grids (one grid per dose)
	+ [VIZ] cross-dose contrast summary: which archetype pairs show the largest Δβ shifts between doses
12) between-fit flow for soft assignments between doses
	+ pairwise pc.tl.flow_between for baseline, PD1, RT
	+ report flow QC & fit metrics
	+ report soft assignments as above in (6)
	+ re-run (6) on each dose
	+ run pc.tl.archetype_feature_similarity between each dose pair
		+ compare both archetypes and Dirichlet components across doses: how much of baseline component 3 is present in PD1 component 7 (for example)?
	+ [VIZ] pc.pl.archetype_correspondence: K_src × K_tgt heatmap for each dose pair (base→PD1, PD1→RT, base→RT)
	+ [VIZ] between-fit MMD heatmaps: K_a × K_b for each dose pair
	+ [VIZ] between-fit feature similarity heatmaps: Spearman ρ between dose-pair archetypes
	+ [VIZ] pc.pl.soft_assignment_heatmap: kNN-based soft assignment correspondence per dose pair
	+ [VIZ] component correspondence table: for each dose pair, which components map to which (Spearman on regression β vectors)
13) between-fit Sinkhorn flow matching for flow fields and gene/geneset association between doses
	+ reuse pc.tl.flow_between for baseline, PD1, RT
		+ pc.tl.flow_significance is critical here
	+ run pc.tl.flow_gene_alignment--should be extensible to geneset score magnitudes
		+ for now transport source cells, reconstruct gene expression at sorce and transposed positions via loadings @ PCA_coords, and recompute pathway scores at both ends to measure change per pathway
		+ if that works wrap that workflow to flow_pathway_alignment()
	+ report QC & fit metrics
	+ report return dataset structures
	+ summarize top feature associations, both flow-aligned and flow-opposing
	+ [VIZ] flow training QC table per dose pair: n_epochs, final loss, MMD before/after, MMD improvement, significance p-value
	+ [VIZ] loss curves: per dose pair training loss over epochs
	+ [VIZ] pc.pl.velocity_quiver: flow vectors in PCA space per dose pair
	+ [VIZ] pc.pl.density_comparison: source vs transported vs target KDE per dose pair
	+ [VIZ] pc.pl.gene_alignment_barplot: top 20 aligned + top 20 opposed genes per dose pair
	+ [VIZ] gene alignment table: per dose pair, top 30 aligned and opposed genes with scores, FDR (if permutation run)
	+ [VIZ] pathway alignment table: per dose pair, Δ pathway score for each Hallmark pathway, ranked by |Δ|
	+ [VIZ] pathway alignment bar plot: Hallmark pathways ranked by flow-direction Δ score per dose pair
14) per-dose transition Jacobian: gene and geneset expansion/contraction
	+ run pc.tl.flow_jacobian for each dose pair, with per_cell_features=True
		+ for gene sets, aggregate per-gene expansion scores by gene set membership by mean
	+ report as above in (13) for each, pairwise between doses
	+ compare flow-aligned genes to simplex regression results, Wald contrasts, and Dirichlet components
	+ [VIZ] pc.pl.jacobian_heatmap: mean Jacobian matrix per dose pair
	+ [VIZ] gene expansion table: per dose pair, top 30 expanding + top 30 contracting genes with per-cell variance
	+ [VIZ] pathway expansion table: per dose pair, Hallmark pathways ranked by mean expansion score
	+ [VIZ] expansion vs alignment scatter: per-gene expansion score vs alignment score, Spearman annotated (validates concordance)
	+ [VIZ] cross-method concordance table: per dose pair, Jaccard overlap of top 50 genes from alignment, expansion, simplex regression R², and Wald |Δβ|
	+ [VIZ] pc.pl.flow_magnitude: transport displacement magnitude per cell in PCA space
15) per-dose transition: top gene deep dive
	+ for top 20 aligned and top 20 opposed genes per flow field, rank expansion and flow across the transport map
	+ [VIZ] per-cell expansion distributions: violin plots of per-cell expansion for top 10 expanding vs top 10 contracting genes per dose pair
	+ [VIZ] pc.pl.trajectory_ribbon: transported cells colored by time step per dose pair
	+ [VIZ] pc.pl.flow_topo_landscape: topographic contour for top expanding gene overlaid on flow per dose pair
16) finally per treatment response grouped across all treatment timepoints
	+ re-run per grouped treatment response branches for (10) through (15)
	+ [VIZ] R vs NR comparison dashboard per step: side-by-side versions of key viz from (10)-(15) for R and NR
	+ [VIZ] R vs NR archetype space overlay: same PCA axes, R cells vs NR cells colored differently
	+ [VIZ] R vs NR summary table: per-method comparison (regression R² correlation, MMD distance, flow alignment overlap, expansion concordance)
17) then per treatment response per dose
	+ re-run (10) through (15) per-treatment response per-dose
	+ [VIZ] 2×3 grid of archetypal spaces: rows = R/NR, columns = baseline/PD1/RT
	+ [VIZ] response×dose feature stability: heatmap of Spearman ρ between all 6 condition pairs on regression β vectors
	+ [VIZ] flow comparison: R base→PD1 vs NR base→PD1 top aligned genes side by side (same for PD1→RT)
	+ [VIZ] expansion divergence: per-gene expansion in R vs NR for same dose transition, scatter with divergent genes highlighted

## Report structure
+ HTML report sections map 1:1 to steps 1-17 above
+ each section is a collapsible `<details>` block
+ executive summary at top: pipeline runtime, K selected, n_features significant, top-line findings per section
+ cross-reference index at bottom: gene name → which sections it appears in as significant
+ all plotly figures are interactive (hover for gene names, zoom)
+ all tables are sortable via DataTables.js or equivalent lightweight JS
+ total expected report size: ~5-15 MB depending on plotly figure complexity

20260326 Update -- Simplifying a Bit
+ technical hypothesis: archetype analysis previously limited by data scale and feature parity with other techniques (e.g., disambiguating archetype-exclusive features, separating linear from interaction terms, comparing between conditions)
+ biological hypothesis: responder lineage exhibits lower overall stress diversity across all treatment points while non-responder lineage's higher stress diversity gives more successful resistance strategies to exploit in growing through the next treatment timepoint
	+ dispositive signal = change in stress diversity per timepoints, compared across lineages
		+ measurements
			+ change in the expression of stress-related genes
			+ changes in the scores of stress pathways
			+ changes in stress-enriched archetype composition
	+ secondary signal is for subpopulations with pre-existing stress upregulation in pre-treatment timepoints in the NR lineage
		+ that means detecting those subpopulations via Spearman's correlation and MMD and investigating what about highly related subpops changes using Wald contrasts
+ NB: exclude Dirichlet component decomposition from this paper--good to introduce as a feature but introduces more complexity than it solves for the current paper
	+ good to keep in the mix to get it working
	+ also exclude driver regression from this paper
+ sets up a 2 part paper: part 1 answers the technical hypothesis in a limited subset of data and then part 2 uses the new tools to dig into the biological hypothesis
+ paper setup
	+ part 1 -- technical hypotheses, run all this on just the R base to PD1 timepoint data subset (patients already held out)
		+ have to introduce Deep_AA, just because it's been published in a pre-print doesn't mean I can just cite it out here
		+ then introduce hyperparameter search for finding the right k archetypes
		+ show ParetoTI/ParTI feature parity on visualization, binning, Wilcoxon rank sum tests, gene set scores, and hypergeometric tests
		+ introduce simplex regression: what it is, how it works, why it's the right solution on this simplex (use the geometry we just learned for biological interpretability)
			+ 1st degree terms: all and archetype-exclusive
			+ 2nd degree terms: patterns in them, cooperative vs tradeoff
		+ Wald contrasts: structured pairwise comparisons both within a singular fit (baseline) or between separate fits (baseline vs PD1)
		+ flow MMD fits comparisons to ID which archetypes are most similar to each other (contrast to cross-fit Spearman correlation)
		+ introduce flow fields and gene alignment @ Jacobian timepoints
	+ part 2 -- biological hypotheses (patients already held out)
		1) global fit: do R/NR and treatment segregate in global archetype space? (archetype features, hypergeometric test results)
		2) detection of related archetypes across treatment timepoints (MMD, Spearman's correlation, Wald contrasts)
			+ will need to also characterize the per-tx archetypes if there are major shifts, especially in interaction terms and especially especially in tradeoff terms
			+ goal is to ID, say, archetype 2 in baseline and that it is most closely related to archetypes 3 and 4 in PD1 in the NR lineage (and how its features compare) vs archetype 3 in baseline going to archetype 1 in the R lineage
		3) flow along treatments to ID top expanding and contracting features
			+ show how different genes and gene classes expand and contract along the treatments
			+ set up qualitative measurement of more contraction in stress genes in R lineage than in NR
			+ ID which stress genes/pathways contributing--pre-registering hypothesis that will see more different arms of the stress responses in NR
		4) R vs NR contrast at each treatment against global fits
			+ need to ID citeable stress genes subset I can reuse in all of this
			+ meause change in stress feature diversity (alpha and beta diversity measures)
				+ contrast to 2nd degree interaction features and flow features, especially top expanding/contracting features
				+ represent differences in contrasting confusion matrices?
			+ **here's where to actually answer the biological hypothesis: clear measurements of diversity at each timepoint and across each lineage**
				+ can then back it up with which stress features are most involved in those changes from results from (3)
				+ can emphasize which stress features are popping up across 2nd degree simplex regression results and flow expansion/contraction
				+ control is permutation/label swapping cells between R and NR lineages to mix them up
					+ should actually obscure the R vs NR label in even running this
					+ only bring it in later and then ask how cells predict label by their segregation in archetype space
		5) finally: held-out data at the archetype fits (global, per-lineage, per-treatment) and at flow fields
			+ predict on the R vs NR label here

20260327
+ thinking about RNRflow and the goal of having all primary data run by end of next week
    + partially worrying that I've missed something in planning that's gonna be an issue
    + thinking through where it is vs where it needs to go
        + where it is
            + triangulating towards interpretable analyses
            + major functions working for the most part
            + testing soft assignment Gaussian RBF on PCA coordinates vs flow matching on gene expression vectors, suspect latter will work better or more clearly justify phenotypic similarity
        + where it needs to go
            + a bunch of current analysis chunks are templates that'll need to be repeated for comparisons—composable layers
            + soft assignment needs to be clearer for relating archetype similarity across fit—especially for IDing a pre-tx population
            + need citeable stress subset to focus on
            + need to set up diversity terms for overall phenotypic similarity and at the gene expression and gene set levels
            + each statistical control needs to be double checked (eg is a label permutation test really the right one for simplex regression?)
    + no barrier to breaking down each step's input, controls, output structure, interpretation, and relation to other steps
        + useful to ID composable chunk characteristics and double check analysis strategy
        + relation to other steps mainly comparing phenotypic features after relatedness scoring
    + label prediction for held out cells currently a bit underspecified
        + going to need to compare predictive power of each feature on each outcome: ideally standardize models as much as possible for a fair comparison
        + instinct is to use lasso/ridge regression but need to think about this hard

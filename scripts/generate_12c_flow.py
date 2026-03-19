#!/usr/bin/env python
"""Generate 12c_flow.ipynb — flow matching, Jacobian, trajectories, gene alignment, CellRank."""

import nbformat

nb = nbformat.v4.new_notebook()
nb.metadata.kernelspec = {
    "display_name": "Python 3 (archetype)",
    "language": "python",
    "name": "python3",
}

DATA_DIR = "/Users/honkala/Desktop/PEACH_public/data/trained"


def md(text):
    nb.cells.append(nbformat.v4.new_markdown_cell(text))


def code(text):
    nb.cells.append(nbformat.v4.new_code_cell(text))


# ===========================================================================
# TITLE
# ===========================================================================
md("""\
# 12c. Flow Matching & Trajectory Analysis

Conditional optimal transport flow from CMP to Monocyte in PCA space.
Includes Jacobian analysis (local expansion/contraction), trajectory sampling,
gene alignment with permutation significance, and CellRank comparison.

**Loads pre-trained AnnDatas** with archetype weights, pathway scores, and
PCA coordinates already computed.""")

# ===========================================================================
# SETUP
# ===========================================================================
code("""\
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != "fork":
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass

import numpy as np
import pandas as pd
import anndata as ad
import torch
import peach as pc

ALPHA = 0.05
K = 4
N_PERMUTATIONS = 200
FLOW_EPOCHS = 300""")

code(f"""\
adata_cmp = ad.read_h5ad("{DATA_DIR}/hsc_cmp_v050.h5ad")
adata_mono = ad.read_h5ad("{DATA_DIR}/hsc_mono_v050.h5ad")
print(f"CMP: {{adata_cmp.shape}}")
print(f"Mono: {{adata_mono.shape}}")""")

# ===========================================================================
# 1. FLOW TRAINING
# ===========================================================================
md("""\
## 1. Flow Matching (CMP -> Mono)

Conditional flow matching with optional Sinkhorn OT coupling (`use_ot=True`)
and adaptive ODE integration (`dopri5` default). Learns a velocity field v(x, t)
mapping the CMP distribution to the Monocyte distribution.""")

# flow_within expects source=dict and target=dict that filter on obs columns.
# After concat, cell_type distinguishes "common myeloid progenitor" vs
# "CD14-positive monocyte".  ad.concat drops varm, so we restore PCs manually.
code("""\
import time

# Build combined adata for flow
adata_combined = ad.concat([adata_cmp, adata_mono], join="inner")
adata_combined.obs_names_make_unique()

# ad.concat drops varm — restore PCA loadings (identical across both adatas)
adata_combined.varm["PCs"] = adata_cmp.varm["PCs"].copy()

print(f"Combined: {adata_combined.n_obs:,} cells")
print(f"  cell_type values: {adata_combined.obs['cell_type'].unique().tolist()}")

t1 = time.time()
flow_result = pc.tl.flow_within(
    adata_combined,
    source={"cell_type": "common myeloid progenitor"},
    target={"cell_type": "CD14-positive monocyte"},
    pca_key="X_pca", n_epochs=FLOW_EPOCHS,
    return_model=True,
    holdout_fraction=0.1,
)
elapsed = time.time() - t1

# Extract source_mask for later use
source_mask = flow_result["source_mask"]

mmd_before = flow_result.get("mmd_before", "?")
mmd_after = flow_result.get("mmd_after", "?")
holdout_mmd = flow_result.get("holdout_mmd", "N/A")
print(f"Done ({elapsed:.1f}s): MMD {mmd_before:.4f} -> {mmd_after:.4f} "
      f"({100*(1-mmd_after/mmd_before):.1f}% reduction)")
print(f"Holdout MMD: {holdout_mmd}")""")

# ===========================================================================
# 2. FLOW VISUALIZATIONS
# ===========================================================================
md("""\
## 2. Flow Visualizations""")

code("""\
_ = pc.pl.velocity_quiver(adata_combined, flow_result, n_arrows=300, show=True)
_ = pc.pl.density_comparison(adata_combined, flow_result, show=True)""")

# ===========================================================================
# 3. JACOBIAN ANALYSIS
# ===========================================================================
md("""\
## 3. Jacobian Analysis

The Jacobian dv/dx at each point reveals local expansion (det > 0) or
contraction (det < 0) of the flow field. Computed with `torch.func.jacrev + vmap`
for vectorized efficiency.""")

# Use pc.tl.flow_jacobian for the structured result, then also do manual
# exploration for educational value.
code("""\
# Compute Jacobian at t=0.5 via the public API
model = flow_result["model"]

# Subsample source cells for Jacobian computation
source_pca = adata_combined.obsm["X_pca"][source_mask]
n_jac = min(500, len(source_pca))
rng = np.random.default_rng(42)
jac_idx = rng.choice(len(source_pca), size=n_jac, replace=False)
jac_points = source_pca[jac_idx]

t1 = time.time()
jac_result = pc.tl.flow_jacobian(
    adata_combined, flow_result, model,
    t=0.5, evaluation_points=jac_points,
)
elapsed = time.time() - t1
print(f"Jacobian computed in {elapsed:.1f}s on {n_jac} points")

# Determinants
dets = jac_result["jacobian_det"]
print(f"Determinant stats: mean={dets.mean():.4f}, std={dets.std():.4f}")
print(f"  range=[{dets.min():.4f}, {dets.max():.4f}]")

# Mean Jacobian
mean_jac = jac_result["mean_jacobian"]
print(f"Mean Jacobian trace: {np.trace(mean_jac):.4f}")
print(f"Mean Jacobian Frobenius norm: {np.linalg.norm(mean_jac, 'fro'):.4f}")""")

code("""\
# Feature expansion scores from Jacobian result
feature_expansion = jac_result["feature_expansion"]
if len(feature_expansion) > 0:
    gene_names_jac = list(adata_combined.var_names)
    print(f"\\nFeature expansion: {len(gene_names_jac)} genes scored")
    sorted_idx = np.argsort(feature_expansion)
    print("\\nTop 5 expanding genes:")
    for i in sorted_idx[-5:][::-1]:
        print(f"  {gene_names_jac[i]:20s} expansion={feature_expansion[i]:.4f}")
    print("\\nTop 5 contracting genes:")
    for i in sorted_idx[:5]:
        print(f"  {gene_names_jac[i]:20s} expansion={feature_expansion[i]:.4f}")
else:
    print("No feature expansion scores (PCA loadings not available)")""")

code("""\
# Jacobian heatmap — expects the jacobian result dict (with 'mean_jacobian' key)
_ = pc.pl.jacobian_heatmap(adata_combined, jac_result, show=True)""")

# ===========================================================================
# 4. TRAJECTORY SAMPLING
# ===========================================================================
md("""\
## 4. Trajectory Sampling

ODE integration from t=0 to t=1 at multiple timepoints. Track how cells
move through PCA space and how distributions change along the trajectory.""")

# model.transport(..., return_trajectory=True) returns [n_steps+1, n_cells, dim]
code("""\
# Compute trajectory via model.transport with return_trajectory=True
n_traj = 500
traj_idx = rng.choice(np.where(source_mask)[0], size=n_traj, replace=False)
traj_points = adata_combined.obsm["X_pca"][traj_idx]
n_steps = 20

t1 = time.time()
trajectory = model.transport(traj_points, n_steps=n_steps, return_trajectory=True)
elapsed = time.time() - t1
print(f"Trajectory: {trajectory.shape} (n_steps+1 x n_cells x dim), {elapsed:.1f}s")

# Statistics per timepoint
from peach._core.utils.flow_matching import compute_mmd
target_pca = adata_combined.obsm["X_pca"][~source_mask]
source_pca_all = adata_combined.obsm["X_pca"][source_mask]

print("\\nTimepoint statistics:")
for step_idx in [0, n_steps//4, n_steps//2, 3*n_steps//4, n_steps]:
    t_val = step_idx / n_steps
    pts = trajectory[step_idx]
    mmd_src = compute_mmd(pts, source_pca_all[:n_traj])
    mmd_tgt = compute_mmd(pts, target_pca[:n_traj])
    pc1_mean = pts[:, 0].mean()
    print(f"  t={t_val:.2f}: MMD_src={mmd_src:.4f}, MMD_tgt={mmd_tgt:.4f}, "
          f"PC1_mean={pc1_mean:.3f}")""")

# model.velocity_at(x, t) is the correct method name (not model.velocity)
code("""\
# Velocity magnitude along trajectory
print("\\nVelocity magnitude along trajectory:")
for step_idx in [0, n_steps//4, n_steps//2, 3*n_steps//4, n_steps]:
    t_val = step_idx / n_steps
    pts = trajectory[step_idx]
    vel = model.velocity_at(pts, t=t_val)
    mag = np.linalg.norm(vel, axis=1)
    print(f"  t={t_val:.2f}: |v| mean={mag.mean():.4f}, std={mag.std():.4f}")""")

# trajectory_ribbon accepts optional flow_model for real ODE trajectory
code("""\
# Trajectory ribbon — pass the model for real ODE integration
_ = pc.pl.trajectory_ribbon(adata_combined, flow_result, flow_model=model, show=True)""")

# ===========================================================================
# 5. ARCHETYPE CORRESPONDENCE (SOFT ASSIGNMENT)
# ===========================================================================
md("""\
## 5. Soft Archetype Correspondence

Transport CMP archetype-dominant cells to Mono space and compute soft assignment
to Mono archetypes. Reveals which CMP archetypes map to which Mono archetypes.""")

code("""\
weights_cmp = np.asarray(adata_cmp.obsm["cell_archetype_weights"])
weights_mono = np.asarray(adata_mono.obsm["cell_archetype_weights"])

# Archetype centroids: mean of top-20% purity cells
correspondence = np.zeros((K, K))
for k_src in range(K):
    purity = weights_cmp[:, k_src]
    threshold = np.percentile(purity, 80)
    high_purity = np.where(purity >= threshold)[0]

    src_pca = adata_cmp.obsm["X_pca"][high_purity]
    transported = model.transport(src_pca)

    # Soft assignment: find nearest mono archetype via weight interpolation
    from scipy.spatial import cKDTree
    tree = cKDTree(adata_mono.obsm["X_pca"])
    _, nn_idx = tree.query(transported, k=10)
    nn_weights = weights_mono[nn_idx].mean(axis=1)
    correspondence[k_src] = nn_weights.mean(axis=0)

# Normalize rows
row_sums = correspondence.sum(axis=1, keepdims=True)
correspondence /= np.maximum(row_sums, 1e-10)

print("Correspondence matrix (CMP -> Mono):")
header = "         " + "  ".join(f"Mono_A{k}" for k in range(K))
print(header)
for k_src in range(K):
    row = "  ".join(f"{correspondence[k_src, k]:.3f}" for k in range(K))
    best = int(np.argmax(correspondence[k_src]))
    print(f"  CMP_A{k_src}: {row}   -> A{best}")""")

# archetype_correspondence in pl.flow expects a flow_between_result dict, not
# flow_within. Plot the correspondence matrix directly with plotly instead.
code("""\
import plotly.graph_objects as go

fig = go.Figure(data=go.Heatmap(
    z=correspondence,
    x=[f"Mono A{k}" for k in range(K)],
    y=[f"CMP A{k}" for k in range(K)],
    colorscale="Blues",
    colorbar=dict(title="Weight", thickness=12, len=0.6),
    text=np.round(correspondence, 3),
    texttemplate="%{text}",
))
fig.update_layout(
    title="Archetype correspondence (CMP -> Mono)",
    xaxis_title="Mono archetypes",
    yaxis_title="CMP archetypes",
    width=500, height=400,
)
fig.show()""")

# ===========================================================================
# 6. GENE ALIGNMENT WITH PERMUTATION STATS
# ===========================================================================
md("""\
## 6. Gene-Level Flow Alignment

Project mean transport velocity onto PCA loadings to identify genes aligned
with the CMP->Mono transition. Permutation test (shuffle PCA loadings) provides
null distribution and FDR-corrected p-values.""")

code("""\
t1 = time.time()
gene_align = pc.tl.flow_gene_alignment(
    adata_combined, flow_result,
    n_permutations=N_PERMUTATIONS, random_state=42,
)
elapsed = time.time() - t1

scores = gene_align["alignment_scores"]
gene_names = gene_align["gene_names"]
print(f"Gene alignment: {len(gene_names)} genes scored in {elapsed:.1f}s")
print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")

# Permutation results
pvals = gene_align.get("alignment_pvalues")
fdr = gene_align.get("alignment_pvalues_fdr")
if fdr is not None:
    n_sig = int(np.sum(fdr < ALPHA))
    print(f"Significant genes (FDR < 0.05): {n_sig}/{len(fdr)}")

print(f"\\nTop 10 aligned:")
for g in gene_align["top_aligned"][:10]:
    i = gene_names.index(g)
    p_str = f"  q={fdr[i]:.3f}" if fdr is not None else ""
    print(f"  {g:20s} score={scores[i]:.4f}{p_str}")

print(f"\\nTop 10 opposed:")
for g in gene_align["top_opposed"][:10]:
    i = gene_names.index(g)
    p_str = f"  q={fdr[i]:.3f}" if fdr is not None else ""
    print(f"  {g:20s} score={scores[i]:.4f}{p_str}")""")

# gene_alignment_barplot expects the alignment result dict, not flow_result
code("""\
_ = pc.pl.gene_alignment_barplot(adata_combined, gene_align, show=True)""")

# ===========================================================================
# 7. GENESET FLOW ALIGNMENT (HALLMARK)
# ===========================================================================
md("""\
### Geneset Flow Alignment

Aggregate gene-level alignment scores per HALLMARK pathway to identify
pathway-level transitions.""")

code("""\
pw_names = list(adata_cmp.uns.get("pathway_scores_pathways", []))
if pw_names:
    net = pc.pp.load_pathway_networks(["hallmark"], verbose=False)
    scores = gene_align["alignment_scores"]
    gene_names = gene_align["gene_names"]
    gene_to_score = dict(zip(gene_names, scores))

    from scipy.stats import ttest_1samp
    from statsmodels.stats.multitest import multipletests

    pw_results = []
    for pw in pw_names:
        pw_genes = set(net[net["source"] == pw]["target"])
        pw_scores = [gene_to_score[g] for g in pw_genes if g in gene_to_score]
        if len(pw_scores) >= 5:
            mean_score = np.mean(pw_scores)
            _, pval = ttest_1samp(pw_scores, 0)
            pw_results.append((pw, mean_score, pval, len(pw_scores)))

    if pw_results:
        pw_results.sort(key=lambda x: x[1])
        pvals_all = [r[2] for r in pw_results]
        _, fdr_all, _, _ = multipletests(pvals_all, method="fdr_bh")

        print(f"Pathways scored: {len(pw_results)}, "
              f"significant: {int(np.sum(np.array(fdr_all) < ALPHA))}")

        print(f"\\nTop 5 aligned:")
        for r, q in sorted(zip(pw_results, fdr_all), key=lambda x: -x[0][1])[:5]:
            print(f"  {r[0]:50s} mean={r[1]:.4f} (q={q:.3f}, n={r[3]})")

        print(f"\\nTop 5 opposed:")
        for r, q in sorted(zip(pw_results, fdr_all), key=lambda x: x[0][1])[:5]:
            print(f"  {r[0]:50s} mean={r[1]:.4f} (q={q:.3f}, n={r[3]})")""")

# ===========================================================================
# 8. CELLRANK COMPARISON
# ===========================================================================
md("""\
## 8. CellRank Comparison

Compare flow transport direction with CellRank pseudotimes. Positive Spearman
correlation indicates agreement between continuous OT flow and graph-based
pseudotime.""")

code("""\
try:
    import cellrank
    print(f"CellRank {cellrank.__version__} available")
except ImportError:
    print("CellRank not available, skipping")
    cellrank = None""")

code("""\
if cellrank is not None:
    print("Setting up CellRank on CMP data...")
    t1 = time.time()

    try:
        cr_result = pc.tl.setup_cellrank(adata_cmp)
        print(f"CellRank setup: {time.time()-t1:.1f}s")

        print("\\nComputing pseudotimes...")
        pc.tl.compute_lineage_pseudotimes(adata_cmp)
        lineages = adata_cmp.uns["lineage_names"]
        print(f"  Lineages: {lineages}")

        # Compare flow trajectory with pseudotimes
        from scipy.stats import spearmanr
        transported = flow_result["transported"]
        source_pca = adata_cmp.obsm["X_pca"]
        flow_displacement = np.linalg.norm(transported[:len(source_pca)] - source_pca, axis=1)

        print("\\nFlow vs CellRank pseudotime:")
        for lin in lineages:
            pt_key = f"pseudotime_to_{lin}"
            if pt_key in adata_cmp.obs:
                pt_vals = adata_cmp.obs[pt_key].values
                valid = np.isfinite(pt_vals) & np.isfinite(flow_displacement)
                if valid.sum() > 100:
                    rho, pval = spearmanr(flow_displacement[valid], pt_vals[valid])
                    print(f"  {lin}: rho={rho:.3f} (p={pval:.2e})")
    except Exception as e:
        print(f"CellRank failed: {e}")""")


# ===========================================================================
# 9. PER-CELL GENE ALIGNMENT
# ===========================================================================
md("""\
## 9. Per-Cell Gene Alignment

Instead of a single global alignment score per gene, compute per-cell alignment
by correlating each cell's velocity direction with PCA loadings. Reveals spatial
heterogeneity in gene-level flow contributions.""")

code("""\
t1 = time.time()
percell_align = pc.tl.flow_gene_alignment(
    adata_combined, flow_result,
    per_cell=True, random_state=42,
)
elapsed = time.time() - t1

pc_scores = percell_align["per_cell_alignment"]
print(f"Per-cell alignment: {pc_scores.shape} ({elapsed:.1f}s)")
print(f"  (n_source_cells x n_top_features)")

# Top genes by variance across cells (heterogeneous alignment)
var_scores = np.var(pc_scores, axis=0)
gene_names_pc = percell_align["per_cell_gene_names"]
top_var = np.argsort(var_scores)[-10:][::-1]
print(f"\\nTop 10 genes by alignment variance (spatially heterogeneous):")
for i in top_var:
    print(f"  {gene_names_pc[i]:20s} var={var_scores[i]:.4f} "
          f"mean={pc_scores[:, i].mean():.4f}")""")

# ===========================================================================
# 10. BIFURCATION SCORING
# ===========================================================================
md("""\
## 10. Bifurcation Scoring

Eigenvalue decomposition of the Jacobian at multiple timepoints identifies
where the flow field transitions from convergent to divergent — potential
bifurcation points where cell fate decisions occur.""")

code("""\
t1 = time.time()
bif = pc.tl.flow_bifurcation(
    adata_combined, flow_result, model,
    n_timepoints=10,
)
elapsed = time.time() - t1

print(f"Bifurcation analysis ({elapsed:.1f}s):")
print(f"  Timepoints: {len(bif['timepoints'])}")
print(f"  Per-cell scores: {bif['bifurcation_score'].shape}")
print(f"  Saddle points detected: {bif['n_saddle_points']}")

# Divergence profile over time
div_mean = bif["divergence"].mean(axis=1)
for i, t_val in enumerate(bif["timepoints"]):
    print(f"  t={t_val:.2f}: mean_divergence={div_mean[i]:.4f}")""")

# ===========================================================================
# WRITE
# ===========================================================================
output_path = "docs/tutorials/12c_flow.ipynb"
nbformat.write(nb, output_path)
print(f"Wrote {len(nb.cells)} cells to {output_path}")

#!/usr/bin/env python
"""Generate 12c2_flow_graph.ipynb — feature coupling graphs from Jacobian, centrality analysis."""

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
# 12c2. Feature Coupling Graphs from Flow Jacobian

Construct directed gene interaction graphs from the Jacobian of the CMP->Mono
flow field. The cross-term `L[g_i] @ J(t) @ L[g_j]` measures how gene j's
direction feeds into gene i's expansion at time t.

**Approach C**: Static graph collapsed over time — identifies overall hub genes.
**Approach A**: Temporal graph — distinguishes early, bridge, and late mediators.

**Requires**: Pre-trained AnnDatas + a trained flow model (from 12c).""")

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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

K = 4
FLOW_EPOCHS = 300""")

code(f"""\
adata_cmp = ad.read_h5ad("{DATA_DIR}/hsc_cmp_v050.h5ad")
adata_mono = ad.read_h5ad("{DATA_DIR}/hsc_mono_v050.h5ad")
print(f"CMP: {{adata_cmp.shape}}")
print(f"Mono: {{adata_mono.shape}}")""")

# ===========================================================================
# 1. TRAIN FLOW MODEL
# ===========================================================================
md("""\
## 1. Flow Model Training

Need a trained FlowModel for Jacobian computation. Re-train with `return_model=True`.""")

code("""\
import time

adata_combined = ad.concat([adata_cmp, adata_mono], join="inner")
adata_combined.obs_names_make_unique()
adata_combined.varm["PCs"] = adata_cmp.varm["PCs"].copy()

print(f"Combined: {adata_combined.n_obs:,} cells")

t1 = time.time()
flow_result = pc.tl.flow_within(
    adata_combined,
    source={"cell_type": "common myeloid progenitor"},
    target={"cell_type": "CD14-positive monocyte"},
    pca_key="X_pca", n_epochs=FLOW_EPOCHS,
    return_model=True,
)
elapsed = time.time() - t1

source_mask = flow_result["source_mask"]
model = flow_result["model"]
mmd_before = flow_result.get("mmd_before", 0)
mmd_after = flow_result.get("mmd_after", 0)
print(f"Done ({elapsed:.1f}s): MMD {mmd_before:.4f} -> {mmd_after:.4f}")""")

# ===========================================================================
# 2. APPROACH C: STATIC FEATURE COUPLING GRAPH
# ===========================================================================
md("""\
## 2. Static Feature Coupling Graph (Approach C)

Integrate Jacobian cross-terms over 20 timepoints to build a single directed
gene-gene coupling graph. `G[i,j] = L[g_i] @ J_mean(t) @ L[g_j]` averaged
over time.""")

code("""\
t1 = time.time()
graph_c = pc.tl.flow_feature_graph(
    adata_combined, flow_result, model,
    n_top_genes=200,
    n_timepoints=20,
    n_eval_points=300,
)
elapsed = time.time() - t1

adj = graph_c["adjacency_matrix"]
gene_names = graph_c["gene_names"]
print(f"Static graph: {len(gene_names)} genes, {elapsed:.1f}s")
print(f"Adjacency matrix: {adj.shape}, non-zero: {np.count_nonzero(adj)}")
print(f"Edge threshold: {graph_c['edge_threshold']:.6f}")""")

code("""\
# Centrality results
out_c = graph_c["out_centrality"]
in_c = graph_c["in_centrality"]
flow_c = graph_c["flow_centrality"]

print("Top 20 hub genes (by flow centrality = out * in):")
for g in graph_c["top_hub_genes"]:
    i = gene_names.index(g)
    print(f"  {g:20s}  out={out_c[i]:.4f}  in={in_c[i]:.4f}  flow={flow_c[i]:.4f}")

# igraph graph object (if igraph is installed)
ig_graph = graph_c.get("igraph")
if ig_graph is not None:
    print(f"\\nigraph graph: {ig_graph.vcount()} vertices, {ig_graph.ecount()} edges")
    pr = ig_graph.pagerank()
    btw = ig_graph.betweenness()
    top_pr = sorted(range(len(pr)), key=lambda i: -pr[i])[:10]
    print("Top 10 by PageRank:")
    for i in top_pr:
        print(f"  {ig_graph.vs[i]['name']:20s} PR={pr[i]:.4f} btw={btw[i]:.1f}")
else:
    print("\\nigraph not installed — skipping graph metrics")

# Hub genes per archetype
hub_per_arch = graph_c.get("hub_genes_per_archetype", {})
if hub_per_arch:
    print(f"\\nHub genes per archetype:")
    for k, genes in sorted(hub_per_arch.items()):
        print(f"  A{k}: {genes[:10]}")""")

# ===========================================================================
# 2b. STATIC GRAPH VISUALIZATION
# ===========================================================================
md("""\
### Static Graph Visualization""")

code("""\
# Adjacency heatmap (top 50 by flow centrality)
top50_idx = np.argsort(flow_c)[-50:][::-1]
top50_names = [gene_names[i] for i in top50_idx]
sub_adj = adj[np.ix_(top50_idx, top50_idx)]

fig = go.Figure(go.Heatmap(
    z=sub_adj,
    x=top50_names,
    y=top50_names,
    colorscale="RdBu_r",
    zmid=0,
    colorbar=dict(title="Coupling", thickness=12, len=0.6),
))
fig.update_layout(
    title="Feature Coupling Matrix (top 50 hub genes)",
    width=900, height=800,
    xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
    yaxis=dict(tickfont=dict(size=8)),
)
fig.show()""")

code("""\
# Centrality bar plot
top30_idx = np.argsort(flow_c)[-30:][::-1]
top30_names = [gene_names[i] for i in top30_idx]

fig = make_subplots(rows=1, cols=3, subplot_titles=["Out-centrality", "In-centrality", "Flow centrality"])
for col, (vals, label) in enumerate([
    (out_c[top30_idx], "Out"), (in_c[top30_idx], "In"), (flow_c[top30_idx], "Flow")
], 1):
    fig.add_trace(go.Bar(
        x=vals, y=top30_names, orientation="h", name=label,
        marker_color=["#0072B2", "#D55E00", "#009E73"][col-1],
    ), row=1, col=col)
fig.update_layout(height=700, width=1200, showlegend=False,
                  title="Gene Centrality in Flow Coupling Graph")
fig.update_yaxes(autorange="reversed")
fig.show()""")

# ===========================================================================
# 3. APPROACH A: TEMPORAL FEATURE GRAPH
# ===========================================================================
md("""\
## 3. Temporal Feature Graph (Approach A)

Same Jacobian computation but retaining temporal structure. Uses 4 temporal
bins (early t<0.25, mid-early, mid-late, late t≥0.75) to identify phase-specific
mediators. Can be restricted to specific archetype pairs via `archetype_pairs`.""")

code("""\
t1 = time.time()
graph_a = pc.tl.flow_temporal_feature_graph(
    adata_combined, flow_result, model,
    n_top_genes=200,
    n_timepoints=20,
    n_eval_points=300,
)
elapsed = time.time() - t1

print(f"Temporal graph: {len(graph_a['gene_names'])} genes x "
      f"{graph_a['n_timepoints']} timepoints, {elapsed:.1f}s")
print(f"Cross matrices: {graph_a['cross_matrices'].shape}")
print(f"Self expansion: {graph_a['self_expansion'].shape}")""")

md("""\
### Archetype-Pair Focused Temporal Graph

Restrict temporal analysis to cells transitioning between specific archetype
pairs. Here we focus on (A0, A1) — cells whose top two weight allocations are
to archetypes 0 and 1.""")

code("""\
t1 = time.time()
graph_a_pair = pc.tl.flow_temporal_feature_graph(
    adata_combined, flow_result, model,
    n_top_genes=200,
    n_timepoints=20,
    n_eval_points=300,
    archetype_pairs=[(0, 1)],
)
elapsed = time.time() - t1
print(f"Pair-focused graph ({elapsed:.1f}s): {len(graph_a_pair['gene_names'])} genes")
print(f"  Top early (A0-A1): {graph_a_pair['top_early_genes'][:5]}")
print(f"  Top late  (A0-A1): {graph_a_pair['top_late_genes'][:5]}")""")

code("""\
# Phase-specific genes (4 temporal bins)
print("\\nTop 10 EARLY genes (t < 0.25):")
for g in graph_a["top_early_genes"]:
    print(f"  {g}")

print("\\nTop 10 MID-EARLY genes (0.25 <= t < 0.5):")
for g in graph_a["top_mid_early_genes"]:
    print(f"  {g}")

print("\\nTop 10 MID-LATE genes (0.5 <= t < 0.75):")
for g in graph_a["top_mid_late_genes"]:
    print(f"  {g}")

print("\\nTop 10 LATE genes (t >= 0.75):")
for g in graph_a["top_late_genes"]:
    print(f"  {g}")""")

# ===========================================================================
# 3b. TEMPORAL VISUALIZATION
# ===========================================================================
md("""\
### Temporal Profile Visualization""")

code("""\
# Temporal importance heatmap
profile = graph_a["temporal_profile"]  # [n_timepoints, n_top]
timepoints = graph_a["timepoints"]
t_gene_names = graph_a["gene_names"]

# Top 40 genes by total temporal centrality
tc = graph_a["temporal_centrality"]
top40_idx = np.argsort(tc)[-40:][::-1]
top40_names = [t_gene_names[i] for i in top40_idx]

fig = go.Figure(go.Heatmap(
    z=profile[:, top40_idx].T,
    x=[f"{t:.2f}" for t in timepoints],
    y=top40_names,
    colorscale="Viridis",
    colorbar=dict(title="Importance", thickness=12, len=0.6),
))
fig.update_layout(
    title="Temporal Gene Importance Profile (top 40)",
    xaxis_title="Flow time (t)",
    yaxis_title="Gene",
    width=1000, height=700,
    yaxis=dict(tickfont=dict(size=9)),
)
fig.show()""")

code("""\
# Self-expansion ribbon for top 20 genes
self_exp = graph_a["self_expansion"]  # [n_timepoints, n_top]
top20_idx = np.argsort(tc)[-20:][::-1]

fig = go.Figure()
colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
          "#56B4E9", "#F0E442", "#999999", "#882255", "#332288",
          "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#AA4499",
          "#661100", "#6699CC", "#AA4466", "#997700", "#994455"]

for rank, gi in enumerate(top20_idx):
    fig.add_trace(go.Scatter(
        x=timepoints, y=self_exp[:, gi],
        mode="lines", name=t_gene_names[gi],
        line=dict(color=colors[rank % len(colors)], width=2),
    ))

fig.update_layout(
    title="Self-Expansion Along Flow (top 20 genes)",
    xaxis_title="Flow time (t)",
    yaxis_title="Self-expansion (L[g] @ J(t) @ L[g])",
    width=900, height=500,
    legend=dict(font=dict(size=9)),
)
fig.show()""")

code("""\
# Compare early vs late: scatter plot
early_mask = timepoints < 0.25
late_mask = timepoints >= 0.75

early_importance = profile[early_mask].mean(axis=0) if early_mask.any() else np.zeros(len(t_gene_names))
late_importance = profile[late_mask].mean(axis=0) if late_mask.any() else np.zeros(len(t_gene_names))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=early_importance, y=late_importance,
    mode="markers+text",
    text=t_gene_names,
    textposition="top center",
    textfont=dict(size=7),
    marker=dict(size=6, color=tc, colorscale="Viridis",
                colorbar=dict(title="Total centrality", thickness=12)),
))
fig.add_shape(type="line", x0=0, x1=max(early_importance.max(), late_importance.max()),
              y0=0, y1=max(early_importance.max(), late_importance.max()),
              line=dict(dash="dash", color="gray"))
fig.update_layout(
    title="Early vs Late Gene Importance",
    xaxis_title="Early importance (t < 0.25)",
    yaxis_title="Late importance (t ≥ 0.75)",
    width=700, height=600,
)
fig.show()""")

# ===========================================================================
# 4. CROSS-REFERENCE WITH REGRESSION
# ===========================================================================
md("""\
## 4. Cross-Reference: Hub Genes vs Regression Drivers

Compare flow centrality hub genes with top simplex regression features.
Genes that are both regression drivers AND flow hubs are strong candidates
for key transition regulators.""")

code("""\
from peach._core.utils.feature_utils import resolve_regression_result

reg = resolve_regression_result(adata_cmp, feature_type="genes")
reg_r2 = np.asarray(reg["r_squared_degree1"])
reg_names = list(reg["feature_names"])

# Map hub genes to their regression R²
print("Hub genes vs regression R²:")
print(f"{'Gene':20s} {'Flow centrality':>16s} {'Regression R²':>14s}")
print("-" * 54)
for g in graph_c["top_hub_genes"][:20]:
    fc_idx = gene_names.index(g)
    if g in reg_names:
        reg_idx = reg_names.index(g)
        r2 = reg_r2[reg_idx]
    else:
        r2 = float("nan")
    print(f"{g:20s} {flow_c[fc_idx]:16.4f} {r2:14.4f}")""")

# ===========================================================================
# 5. ADDITIONAL VISUALIZATIONS
# ===========================================================================
md("""\
## 5. Additional Visualizations

Archetype radar, GMM neighborhood graph, soft assignment flow, and
soft assignment heatmap.""")

code("""\
# Archetype radar for CMP archetypes
print("--- CMP Archetype Radar ---")
_ = pc.pl.archetype_radar(adata_cmp, top_n=8, show=True)""")

code("""\
# GMM component neighborhood graph
print("--- CMP GMM Neighborhood Graph ---")
_ = pc.pl.component_neighborhood_graph(adata_cmp, show=True)""")

code("""\
# Soft assignment feature flow
print("--- CMP Soft Assignment Flow ---")
_ = pc.pl.soft_assignment_flow(adata_cmp, top_n=10, show=True)""")

code("""\
# Soft assignment heatmap (between-fit correspondence)
print("--- Soft Assignment Heatmap (CMP -> Mono) ---")
_ = pc.pl.soft_assignment_heatmap(adata_cmp, flow_result, adata_b=adata_mono, show=True)""")

# ===========================================================================
# 6. FLOW TOPO LANDSCAPE
# ===========================================================================
md("""\
## 6. Flow Topographic Landscape

Topographic contour map showing feature expression at source/target and
Jacobian expansion/contraction through the flow field. PC1 x PC2 projection.""")

code("""\
import matplotlib
matplotlib.use("Agg")

print("--- Flow Topo Landscape ---")
fig = pc.pl.flow_topo_landscape(
    adata_combined, flow_result, model,
    n_features=5,
    n_timepoints=20,
    n_eval_points=300,
    show=False,
)
print(f"Figure size: {fig.get_size_inches()}")
print("Topo landscape generated successfully.")""")


# ===========================================================================
# WRITE
# ===========================================================================
output_path = "docs/tutorials/12c2_flow_graph.ipynb"
nbformat.write(nb, output_path)
print(f"Wrote {len(nb.cells)} cells to {output_path}")

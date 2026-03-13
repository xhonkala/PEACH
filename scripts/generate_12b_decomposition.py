#!/usr/bin/env python
"""Generate 12b_decomposition.ipynb — GMM decomposition, component analysis, per-component regression."""

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
# 12b. GMM Simplex Decomposition & Component Analysis

Gaussian Mixture Models in ILR-transformed simplex space identify subpopulations
with distinct archetype weight profiles. Each component maps to a dominant archetype
and can be analyzed independently.

**Loads pre-trained AnnDatas** with archetype weights, GMM results, and
regression results already computed.""")

# ===========================================================================
# SETUP
# ===========================================================================
code("""\
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import anndata as ad
import plotly.graph_objects as go
import peach as pc

ALPHA = 0.05
K = 4""")

code(f"""\
adata_cmp = ad.read_h5ad("{DATA_DIR}/hsc_cmp_v050.h5ad")
adata_mono = ad.read_h5ad("{DATA_DIR}/hsc_mono_v050.h5ad")
print(f"CMP: {{adata_cmp.shape}}")
print(f"Mono: {{adata_mono.shape}}")""")

# ===========================================================================
# 1. GMM OVERVIEW
# ===========================================================================
md("""\
## 1. GMM Decomposition Overview

Fit in ILR-transformed weight space (K-1 dimensions). BIC selects optimal number
of components; stability filtering removes unreliable components.""")

code("""\
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    gmm = ad_obj.uns.get("peach_gmm")
    if gmm is None:
        print(f"{name}: no GMM results, computing...")
        gmm = pc.tl.feature_simplex_decomposition(ad_obj, characterize_features=True)

    print(f"\\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Optimal: {gmm['n_components_optimal']}, Stable: {gmm['n_components_stable']}")
    print(f"  Archetype map: {gmm['component_archetype_map']}")

    # Component probabilities
    probs = gmm.get("component_probabilities")
    if probs is not None:
        probs = np.asarray(probs)
        print(f"  Posterior probabilities: {probs.shape}")
        print(f"    Max prob: median={np.max(probs, axis=1).mean():.3f}")""")

# ===========================================================================
# 2. GMM VISUALIZATIONS
# ===========================================================================
md("""\
## 2. GMM Visualizations""")

code("""\
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    print(f"\\n--- {name} ---")
    _ = pc.pl.component_scatter(ad_obj, show=True)
    _ = pc.pl.gmm_bic_curve(ad_obj, show=True)
    _ = pc.pl.component_heatmap(ad_obj, top_n=30, show=True)
    _ = pc.pl.component_stability(ad_obj, show=True)
    _ = pc.pl.component_archetype_summary(ad_obj, show=True)""")

# ===========================================================================
# 3. PER-COMPONENT ARCHETYPE DOMINANCE
# ===========================================================================
md("""\
## 3. Per-Component Archetype Dominance

Each component's centroid in weight space reveals which archetype(s) it is
closest to. Low entropy = component dominated by a single archetype.""")

code("""\
archetype_vertices = np.eye(K)

for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    gmm = ad_obj.uns["peach_gmm"]
    weight_means = gmm.get("component_weight_means")
    if weight_means is None:
        weight_means = np.asarray(gmm["component_simplex_means"])

    assignments = np.asarray(gmm["component_assignments"])
    n_comp = weight_means.shape[0]
    weights = np.asarray(ad_obj.obsm["cell_archetype_weights"])

    print(f"\\n{'='*60}")
    print(f"  {name}: {n_comp} components")
    print(f"{'='*60}")

    for c in range(n_comp):
        dominant = int(np.argmax(weight_means[c]))
        dom_w = weight_means[c, dominant]
        n_cells = int(np.sum(assignments == c))

        # Distances to vertices
        dists = np.linalg.norm(archetype_vertices - weight_means[c], axis=1)
        closest = int(np.argmin(dists))

        # Entropy
        w_safe = np.clip(weight_means[c], 1e-10, 1.0)
        entropy = -np.sum(w_safe * np.log2(w_safe))
        max_entropy = np.log2(K)

        w_str = ", ".join(f"{w:.3f}" for w in weight_means[c])
        d_str = ", ".join(f"{d:.3f}" for d in dists)
        print(f"  C{c}: dominant=A{dominant} (w={dom_w:.3f}), "
              f"n={n_cells}, H={entropy:.2f}/{max_entropy:.2f}")
        print(f"    weights: [{w_str}]")
        print(f"    dists:   [{d_str}]")""")

code("""\
# Stacked bar: component weight profiles
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    gmm = ad_obj.uns["peach_gmm"]
    weight_means = gmm.get("component_weight_means")
    if weight_means is None:
        weight_means = np.asarray(gmm["component_simplex_means"])
    n_comp = weight_means.shape[0]

    fig = go.Figure()
    for k in range(K):
        fig.add_trace(go.Bar(
            name=f"A{k}",
            x=[f"C{c}" for c in range(n_comp)],
            y=weight_means[:, k],
        ))
    fig.update_layout(
        barmode="stack", title=f"{name}: Component Weight Profiles",
        yaxis_range=[0, 1.05], height=400, width=600,
    )
    fig.show()""")

# ===========================================================================
# 4. 3D ARCHETYPAL SPACE BY COMPONENT
# ===========================================================================
md("""\
## 4. Archetypal Space Colored by GMM Component""")

code("""\
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    gmm = ad_obj.uns["peach_gmm"]
    assignments = np.asarray(gmm["component_assignments"])
    ad_obj.obs["gmm_component"] = pd.Categorical(
        [f"C{c}" if c >= 0 else "unassigned" for c in assignments]
    )
    fig = pc.pl.archetypal_space(
        ad_obj, color_by="gmm_component",
        title=f"{name}: Archetypal Space by GMM Component",
    )""")

# ===========================================================================
# 5. PER-COMPONENT REGRESSION
# ===========================================================================
md("""\
## 5. Per-Component Regression

Run simplex regression independently per GMM component. Detects component-specific
gene drivers that are masked in the global (pooled) regression.""")

code("""\
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    print(f"\\n{'='*60}")
    print(f"  {name}: Per-Component Regression")
    print(f"{'='*60}")
    comp_regs = pc.tl.component_regression(ad_obj, n_bootstrap=0, robust_se=True)
    n_comp = comp_regs["n_components"]

    for c_idx, c_reg in comp_regs["component_regs"].items():
        n_cells = c_reg.get("n_cells", "?")
        r2_d1 = np.asarray(c_reg["r_squared_degree1"])
        f_pval = np.asarray(c_reg.get("f_pvalue", np.ones(len(r2_d1))))
        n_sig = int(np.sum(f_pval < ALPHA))
        feat_names = c_reg["feature_names"]

        print(f"\\n  Component {c_idx}: {n_cells} cells, "
              f"median R²={np.median(r2_d1):.4f}, {n_sig} significant")

        # Top 5 by R²
        top5 = np.argsort(r2_d1)[-5:][::-1]
        for i in top5:
            print(f"    {feat_names[i]:20s} R²={r2_d1[i]:.4f}")""")

# ===========================================================================
# 6. COMPONENT-SPECIFIC vs GLOBAL DRIVERS
# ===========================================================================
md("""\
### Component-Specific vs Global Drivers

Compare the top drivers per component against the global regression to identify
genes uniquely important in specific subpopulations.""")

code("""\
from peach._core.utils.feature_utils import resolve_regression_result

for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    global_reg = resolve_regression_result(ad_obj, prefer="genes")
    global_r2 = np.asarray(global_reg["r_squared_degree1"])
    global_names = global_reg["feature_names"]

    comp_regs = pc.tl.component_regression(ad_obj, n_bootstrap=0, robust_se=True)

    print(f"\\n--- {name}: Component-specific discoveries ---")
    for c_idx, c_reg in comp_regs["component_regs"].items():
        c_r2 = np.asarray(c_reg["r_squared_degree1"])
        c_names = c_reg["feature_names"]

        # Find genes with high component R² but low global R²
        # Match by name since gene order may differ
        name_to_global_r2 = dict(zip(global_names, global_r2))
        discoveries = []
        for i in range(len(c_names)):
            g_r2 = name_to_global_r2.get(c_names[i], 0)
            if c_r2[i] > 0.05 and c_r2[i] > g_r2 * 2:
                discoveries.append((c_names[i], c_r2[i], g_r2))

        discoveries.sort(key=lambda x: -x[1])
        if discoveries:
            print(f"  Component {c_idx}: {len(discoveries)} component-enriched genes "
                  f"(comp R² > 2x global R²)")
            for gene, cr2, gr2 in discoveries[:5]:
                print(f"    {gene:20s} comp_R²={cr2:.4f}  global_R²={gr2:.4f}  "
                      f"ratio={cr2/max(gr2,1e-6):.1f}x")
        else:
            print(f"  Component {c_idx}: no component-enriched genes found")""")

# ===========================================================================
# 7. GMM COMPONENT NEIGHBORHOOD GRAPH
# ===========================================================================
md("""\
## 7. GMM Component Neighborhood Graph

3D network in archetypal weight space. Nodes = GMM components sized by cell count,
colored by dominant archetype. Edges connect nearby components.""")

code("""\
for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
    print(f"\\n--- {name} ---")
    _ = pc.pl.component_neighborhood_graph(ad_obj, show=True)""")


# ===========================================================================
# WRITE
# ===========================================================================
output_path = "docs/tutorials/12b_decomposition.ipynb"
nbformat.write(nb, output_path)
print(f"Wrote {len(nb.cells)} cells to {output_path}")

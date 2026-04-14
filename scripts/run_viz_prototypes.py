#!/usr/bin/env python
"""Prototype visualization comparison for items 18 and 19.

18: gamma = beta_j/beta_k vs R2_j/R2_k side-by-side
19: Per-pair gene viz options from the brainstorm shortlist

Uses synthetic data that mimics real per-pair gene attributes so we can
iterate on viz choices without waiting for a full training run.
"""
import matplotlib
matplotlib.use("Agg")

import base64
import io
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "diagnostic")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fig_to_b64(fig, dpi=130):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;">'


# =========================================================================
# Synthetic data generation
# =========================================================================

def make_synthetic_gene_data(n_genes=150, seed=42):
    """Synthetic per-pair gene attributes mimicking HSC->CMP flow results.

    Returns a DataFrame with columns:
      gene, expression, expansion, flow_strength, beta_j, beta_k, r2_j, r2_k
    """
    rng = np.random.default_rng(seed)

    genes = [f"GENE_{i}" for i in range(n_genes)]
    # Some real-looking gene names for the top hits
    real_names = [
        "HBA1", "HBB", "MPO", "ELANE", "AZU1", "CTSG", "PRTN3",
        "LYZ", "CST3", "FLT3", "CD34", "KIT", "GATA2", "RUNX1",
        "SPI1", "CEBPA", "IRF8", "HOXA9", "MEIS1", "PBX1",
        "CDK6", "CCND1", "MCM2", "PCNA", "MKI67", "TOP2A",
        "HLA-DRA", "HLA-DRB1", "CD74", "B2M", "AREG", "XIST",
    ]
    for i, name in enumerate(real_names):
        if i < n_genes:
            genes[i] = name

    # Expression: log-normal, range ~0.1 to 5
    expression = rng.lognormal(mean=0.5, sigma=0.8, size=n_genes)

    # Expansion: signed, most near 0, some strong expanders/contractors
    expansion = rng.normal(0, 0.3, size=n_genes)
    # Inject some strong signals
    expansion[:5] = rng.uniform(0.5, 1.5, size=5)   # strong expanders
    expansion[5:10] = rng.uniform(-1.5, -0.5, size=5)  # strong contractors

    # Flow strength: 0-1, most weak, some strong
    flow_strength = np.abs(rng.normal(0, 0.2, size=n_genes))
    flow_strength = np.clip(flow_strength, 0, 1)
    flow_strength[:8] = rng.uniform(0.6, 0.95, size=8)  # strong flow genes

    # Beta coefficients for archetypes j and k
    beta_j = rng.normal(1.0, 0.5, size=n_genes)
    beta_k = rng.normal(0.8, 0.5, size=n_genes)
    # Some tradeoff genes: high beta_j, low beta_k
    beta_j[:5] = rng.uniform(2.0, 4.0, size=5)
    beta_k[:5] = rng.uniform(0.1, 0.5, size=5)
    # Some cooperative genes
    beta_j[5:10] = rng.uniform(2.0, 3.0, size=5)
    beta_k[5:10] = rng.uniform(1.5, 3.0, size=5)

    # R2 per archetype
    r2_j = np.clip(rng.beta(2, 5, size=n_genes), 0.01, 0.99)
    r2_k = np.clip(rng.beta(2, 5, size=n_genes), 0.01, 0.99)
    r2_j[:5] = rng.uniform(0.5, 0.9, size=5)
    r2_k[:5] = rng.uniform(0.05, 0.2, size=5)

    # Pseudotime: 0-1 per cell, n_cells cells
    n_cells = 500
    pseudotime = np.sort(rng.uniform(0, 1, size=n_cells))
    # Per-cell expansion for top genes: varies along pseudotime
    cell_expansion = {}
    for i in range(min(20, n_genes)):
        base = expansion[i]
        noise = rng.normal(0, 0.15, size=n_cells)
        trend = base * (1 + 0.5 * np.sin(2 * np.pi * pseudotime))
        cell_expansion[genes[i]] = trend + noise

    df = pd.DataFrame({
        "gene": genes,
        "expression": expression,
        "expansion": expansion,
        "flow_strength": flow_strength,
        "beta_j": beta_j,
        "beta_k": beta_k,
        "r2_j": r2_j,
        "r2_k": r2_k,
    })

    return df, pseudotime, cell_expansion


# =========================================================================
# Item 18: gamma comparison
# =========================================================================

def section_18_gamma_comparison(df):
    """Side-by-side comparison of gamma = beta_j/beta_k vs R2_j/R2_k."""
    html = "<h2>Item 18: gamma definition comparison</h2>"
    html += "<p>Comparing two ways to define the interaction coefficient gamma "
    html += "for tradeoff/cooperative pattern classification:</p>"

    gamma_beta = df["beta_j"] / np.clip(df["beta_k"], 0.01, None)
    gamma_r2 = df["r2_j"] / np.clip(df["r2_k"], 0.01, None)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: scatter of gamma_beta vs gamma_r2
    ax = axes[0]
    ax.scatter(gamma_beta, gamma_r2, s=15, alpha=0.6, c="steelblue")
    ax.set_xlabel("gamma (beta_j / beta_k)")
    ax.set_ylabel("gamma (R2_j / R2_k)")
    ax.set_title("Gamma: beta-ratio vs R2-ratio")
    ax.axhline(1, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(1, color="gray", linestyle="--", alpha=0.5)
    # Label outliers
    for idx in range(min(8, len(df))):
        ax.annotate(df["gene"].iloc[idx], (gamma_beta.iloc[idx], gamma_r2.iloc[idx]),
                     fontsize=7, alpha=0.8)
    ax.set_xlim(-1, max(10, gamma_beta.quantile(0.99)))
    ax.set_ylim(-1, max(10, gamma_r2.quantile(0.99)))
    ax.grid(True, alpha=0.3)

    # Panel 2: classification comparison
    ax = axes[1]
    # Tradeoff = gamma >> 1 (j dominates k), cooperative = gamma ~ 1
    beta_class = pd.cut(gamma_beta, bins=[-np.inf, 0.5, 1.5, np.inf],
                        labels=["k-dominant", "cooperative", "j-dominant"])
    r2_class = pd.cut(gamma_r2, bins=[-np.inf, 0.5, 1.5, np.inf],
                      labels=["k-dominant", "cooperative", "j-dominant"])
    agree = (beta_class == r2_class).sum()
    total = len(df)

    confusion = pd.crosstab(beta_class, r2_class, margins=True)
    ax.axis("off")
    ax.set_title(f"Classification agreement: {agree}/{total} ({agree/total:.0%})")
    tbl = ax.table(cellText=confusion.values, colLabels=confusion.columns,
                   rowLabels=confusion.index, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.4)

    # Panel 3: distribution of each gamma
    ax = axes[2]
    bins = np.linspace(-1, 10, 50)
    ax.hist(gamma_beta.clip(-1, 10), bins=bins, alpha=0.5, label="beta_j/beta_k", color="steelblue")
    ax.hist(gamma_r2.clip(-1, 10), bins=bins, alpha=0.5, label="R2_j/R2_k", color="coral")
    ax.set_xlabel("gamma value")
    ax.set_ylabel("count")
    ax.set_title("Distribution of gamma under each definition")
    ax.legend()
    ax.axvline(1, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Item 18: gamma = beta_j/beta_k vs R2_j/R2_k", fontsize=13)
    fig.tight_layout()
    html += fig_to_b64(fig)

    html += "<p><b>Key observations:</b></p><ul>"
    html += f"<li>Agreement on 3-class classification: {agree}/{total} ({agree/total:.0%})</li>"
    html += "<li>beta-ratio: sensitive to coefficient magnitude (theoretical max expression at archetype)</li>"
    html += "<li>R2-ratio: sensitive to variance explained (position dependence strength)</li>"
    html += "<li>R2-ratio may be more interpretable: 'how much of this gene's variance does archetype j explain relative to k?'</li>"
    html += "<li>beta-ratio can be extreme when beta_k is near zero (division instability)</li>"
    html += "</ul>"

    return html


# =========================================================================
# Item 19: Per-pair gene viz prototypes
# =========================================================================

def viz_tricolor_scatter(df, title="Tricolor Scatter (signed y-axis)"):
    """Already implemented as R12-17. Replicate here for comparison."""
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(df["expression"], df["expansion"],
                    c=df["flow_strength"], cmap="viridis",
                    s=20 + 80 * df["flow_strength"], alpha=0.7,
                    edgecolors="none")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    plt.colorbar(sc, ax=ax, label="Flow association strength")
    ax.set_xlabel("Mean expression")
    ax.set_ylabel("Expansion (+) / Contraction (-)")
    ax.set_title(title)
    # Label top genes
    score = np.abs(df["expansion"] * df["flow_strength"])
    top = score.nlargest(12).index
    for i in top:
        ax.annotate(df["gene"].iloc[i], (df["expression"].iloc[i], df["expansion"].iloc[i]),
                     fontsize=7, alpha=0.85, xytext=(3, 3), textcoords="offset points")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def viz_ternary(df, title="Ternary Plot"):
    """Map three attributes to barycentric coordinates on a triangle."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Normalize each attribute to [0, 1]
    e = (df["expression"] - df["expression"].min()) / (df["expression"].max() - df["expression"].min() + 1e-10)
    x = (np.abs(df["expansion"]) - np.abs(df["expansion"]).min()) / (np.abs(df["expansion"]).max() - np.abs(df["expansion"]).min() + 1e-10)
    f = (df["flow_strength"] - df["flow_strength"].min()) / (df["flow_strength"].max() - df["flow_strength"].min() + 1e-10)
    total = e + x + f + 1e-10
    e, x, f = e / total, x / total, f / total

    # Triangle vertices
    v0 = np.array([0, 0])       # expression
    v1 = np.array([1, 0])       # expansion
    v2 = np.array([0.5, 0.866]) # flow

    px = e * v0[0] + x * v1[0] + f * v2[0]
    py = e * v0[1] + x * v1[1] + f * v2[1]

    sign_color = np.where(df["expansion"] > 0, "red", "blue")
    ax.scatter(px, py, s=15, c=sign_color, alpha=0.6)

    # Draw triangle
    tri = plt.Polygon([v0, v1, v2], fill=False, edgecolor="black", linewidth=2)
    ax.add_patch(tri)
    ax.text(v0[0] - 0.05, v0[1] - 0.04, "Expression", ha="center", fontsize=10, fontweight="bold")
    ax.text(v1[0] + 0.05, v1[1] - 0.04, "|Expansion|", ha="center", fontsize=10, fontweight="bold")
    ax.text(v2[0], v2[1] + 0.04, "Flow strength", ha="center", fontsize=10, fontweight="bold")

    # Label top genes
    score = np.abs(df["expansion"] * df["flow_strength"])
    top = score.nlargest(10).index
    for i in top:
        ax.annotate(df["gene"].iloc[i], (px.iloc[i], py.iloc[i]),
                     fontsize=7, alpha=0.8)

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.1, 1.0)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def viz_dot_bar_hybrid(df, title="Dot-Bar Hybrid", top_n=25):
    """Dotplot-style grid: left = expression dot, right = stacked bars."""
    top = df.nlargest(top_n, "flow_strength")

    fig, axes = plt.subplots(1, 3, figsize=(14, max(6, top_n * 0.3)),
                              gridspec_kw={"width_ratios": [1, 1, 1]})

    y = np.arange(top_n)
    gene_names = top["gene"].values

    # Panel 1: expression (dot size)
    ax = axes[0]
    sizes = 20 + 150 * (top["expression"] / top["expression"].max())
    ax.scatter(np.zeros(top_n), y, s=sizes, c="steelblue", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(gene_names, fontsize=8)
    ax.set_title("Expression", fontsize=10)
    ax.set_xlim(-0.5, 0.5)
    ax.invert_yaxis()
    ax.set_xticks([])

    # Panel 2: expansion (horizontal bars, signed)
    ax = axes[1]
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in top["expansion"]]
    ax.barh(y, top["expansion"].values, color=colors, alpha=0.7, height=0.6)
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_title("Expansion/Contraction", fontsize=10)
    ax.set_yticks([])
    ax.invert_yaxis()

    # Panel 3: flow strength (horizontal bars)
    ax = axes[2]
    ax.barh(y, top["flow_strength"].values, color="green", alpha=0.6, height=0.6)
    ax.set_title("Flow Strength", fontsize=10)
    ax.set_yticks([])
    ax.invert_yaxis()

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def viz_3d_scatter(df, title="3D Scatter with Projection Shadows"):
    """x=expression, y=expansion, z=flow_strength with wall projections."""
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    x = df["expression"].values
    y = df["expansion"].values
    z = df["flow_strength"].values

    ax.scatter(x, y, z, s=15, c=z, cmap="viridis", alpha=0.7)

    # Projection shadows
    ax.scatter(x, y, np.zeros_like(z), s=5, c="gray", alpha=0.1)  # floor
    ax.scatter(x, np.full_like(y, y.min()), z, s=5, c="gray", alpha=0.1)  # back wall
    ax.scatter(np.full_like(x, x.max()), y, z, s=5, c="gray", alpha=0.1)  # right wall

    # Label top genes
    score = np.abs(y * z)
    top = np.argsort(-score)[:10]
    for i in top:
        ax.text(x[i], y[i], z[i], df["gene"].iloc[i], fontsize=7)

    ax.set_xlabel("Expression")
    ax.set_ylabel("Expansion")
    ax.set_zlabel("Flow Strength")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def viz_absence_plot(df, title="Absence: high-expression, low-flow genes"):
    """Which genes are NOT flow-associated despite high expression?"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color by whether the gene is a "surprise non-hit"
    high_expr = df["expression"] > df["expression"].quantile(0.7)
    low_flow = df["flow_strength"] < df["flow_strength"].quantile(0.3)
    absent = high_expr & low_flow

    colors = np.where(absent, "#d62728", np.where(df["flow_strength"] > 0.5, "#2ca02c", "#cccccc"))
    sizes = np.where(absent, 40, 12)

    ax.scatter(df["expression"], df["flow_strength"], s=sizes, c=colors, alpha=0.7)
    ax.axhline(df["flow_strength"].quantile(0.3), color="gray", linestyle="--", alpha=0.5,
               label=f"flow p30={df['flow_strength'].quantile(0.3):.2f}")
    ax.axvline(df["expression"].quantile(0.7), color="gray", linestyle=":", alpha=0.5,
               label=f"expr p70={df['expression'].quantile(0.7):.2f}")

    # Label the absent genes
    for i in df[absent].index:
        ax.annotate(df["gene"].iloc[i], (df["expression"].iloc[i], df["flow_strength"].iloc[i]),
                     fontsize=7, color="#d62728", fontweight="bold")

    # Label some high-flow genes for contrast
    high_flow = df.nlargest(5, "flow_strength")
    for i in high_flow.index:
        ax.annotate(df["gene"].iloc[i], (df["expression"].iloc[i], df["flow_strength"].iloc[i]),
                     fontsize=7, color="#2ca02c")

    ax.set_xlabel("Mean Expression")
    ax.set_ylabel("Flow Association Strength")
    ax.set_title(f"{title}\nRed = high expression but NOT flow-associated. Why?")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def viz_gene_embedding(df, title="Gene Space (t-SNE on gene attributes)"):
    """Embed genes in 2D using their (expression, expansion, flow) vectors."""
    from sklearn.manifold import TSNE

    features = df[["expression", "expansion", "flow_strength"]].values
    # Standardize
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-10)

    tsne = TSNE(n_components=2, perplexity=min(30, len(df) - 1), random_state=42)
    coords = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=df["flow_strength"], cmap="viridis",
                    s=10 + 50 * np.abs(df["expansion"]), alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Flow strength")

    # Label top genes
    score = np.abs(df["expansion"] * df["flow_strength"])
    top = score.nlargest(15).index
    for i in top:
        ax.annotate(df["gene"].iloc[i], (coords[i, 0], coords[i, 1]),
                     fontsize=7, alpha=0.85)

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"{title}\nColor=flow strength, size=|expansion|")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def viz_pseudotime_expansion(pseudotime, cell_expansion, title="Pseudotime x Expansion"):
    """x = pseudotime, y = expansion per cell, one trace per gene."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(cell_expansion))))

    for i, (gene, values) in enumerate(list(cell_expansion.items())[:10]):
        # Bin along pseudotime for cleaner lines
        n_bins = 30
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_idx = np.digitize(pseudotime, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        binned = np.array([values[bin_idx == b].mean() if (bin_idx == b).any() else np.nan
                           for b in range(n_bins)])
        ax.plot(bin_centers, binned, color=colors[i], linewidth=1.5, label=gene, alpha=0.8)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Flow pseudotime (transport distance)")
    ax.set_ylabel("Expansion (+) / Contraction (-)")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def viz_lollipop(df, title="Lollipop Chart", top_n=25):
    """Gene lollipop: length=flow, head color=expansion sign, head size=expression."""
    top = df.nlargest(top_n, "flow_strength")

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    y = np.arange(top_n)

    # Stems
    for i, (_, row) in enumerate(top.iterrows()):
        ax.plot([0, row["flow_strength"]], [i, i], color="gray", linewidth=1, alpha=0.5)

    # Heads
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in top["expansion"]]
    sizes = 30 + 120 * (top["expression"] / top["expression"].max())
    ax.scatter(top["flow_strength"], y, s=sizes, c=colors, zorder=5, edgecolors="black", linewidths=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(top["gene"].values, fontsize=8)
    ax.set_xlabel("Flow Association Strength")
    ax.set_title(f"{title}\nHead color: red=expanding, blue=contracting. Head size=expression.")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


# =========================================================================
# Main
# =========================================================================

def main():
    print("Generating synthetic data...")
    df, pseudotime, cell_expansion = make_synthetic_gene_data()

    html = """<!DOCTYPE html><html><head><meta charset="UTF-8">
    <title>Viz Prototypes: Items 18 & 19</title>
    <style>
    body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px 40px;
           background: #fafafa; color: #222; max-width: 1400px; }
    h2 { color: #2c3e50; border-bottom: 2px solid #4a90d9; padding-bottom: 8px; margin-top: 30px; }
    h3 { color: #34495e; }
    img { max-width: 100%; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; }
    .verdict { background: #f0f4f8; padding: 12px; border-radius: 8px; margin: 10px 0; }
    </style></head><body>
    <h1>Visualization Prototypes: Items 18 and 19</h1>
    <p>All plots use the same synthetic gene data (150 genes with injected
    signal patterns) so the visual comparisons are fair.</p>
    """

    # Item 18
    print("Item 18: gamma comparison...")
    html += section_18_gamma_comparison(df)

    # Item 19: each viz option
    viz_options = [
        ("19a", "Tricolor Scatter (signed y-axis)", viz_tricolor_scatter,
         "Primary option (already implemented as R12-17). x=expression, y=signed expansion, color=flow strength. "
         "Intuitive axes, good for identifying genes that are both expanding AND flow-aligned."),

        ("19b", "Ternary Plot", viz_ternary,
         "Maps three attributes to barycentric coords on a triangle. Visually striking but harder to read "
         "exact values. Red=expanding, blue=contracting. Best for showing which attribute DOMINATES each gene."),

        ("19c", "Dot-Bar Hybrid", viz_dot_bar_hybrid,
         "Three aligned panels: dot size=expression, signed bar=expansion, bar=flow strength. "
         "Easy to compare across genes. Reads like a table but with visual encoding. Good for top-N lists."),

        ("19d", "3D Scatter with Projection Shadows", viz_3d_scatter,
         "Full 3D encoding. Good for exploration in interactive plotly. Static matplotlib version shown here "
         "loses interactivity. Wall projections show 2D relationships."),

        ("19e", "Absence Plot", viz_absence_plot,
         "Highlights genes that are highly expressed but NOT flow-associated (red). Answers: 'what genes "
         "are the flow ignoring?' This contrast is often more biologically interesting than the positive hits."),

        ("19f", "Gene Space (t-SNE embedding)", viz_gene_embedding,
         "Embeds genes (not cells) in 2D using their attribute vectors. Reveals gene clusters by functional "
         "profile: 'these 40 genes form a contracting, low-expression, high-flow cluster.' Color=flow, size=|expansion|."),

        ("19g", "Pseudotime x Expansion", viz_pseudotime_expansion,
         "x=pseudotime, y=expansion per cell, one trace per gene. Shows HOW expansion changes along the "
         "flow trajectory. Intuitive: read left-to-right as 'along the transition'. Already implemented as R12-16."),

        ("19h", "Lollipop Chart", viz_lollipop,
         "Ranked by flow strength. Head color=expansion sign, head size=expression. Simple, scannable, "
         "good for supplementary figures. Easy to read top-N at a glance."),
    ]

    html += "<h2>Item 19: Per-pair gene visualization options</h2>"
    html += "<p>Each option encodes the same three attributes (expression, expansion, flow strength) differently. "
    html += "Evaluate by: (1) can you identify the top flow genes? (2) can you see expansion/contraction? "
    html += "(3) is the plot readable at 150 genes? (4) would a biologist understand it?</p>"

    for tag, name, func, description in viz_options:
        print(f"  {tag}: {name}...")
        html += f"<h3>{tag}: {name}</h3>"
        html += f"<p><i>{description}</i></p>"

        if func == viz_pseudotime_expansion:
            fig = func(pseudotime, cell_expansion, title=name)
        else:
            fig = func(df, title=name)

        html += fig_to_b64(fig)
        html += "<div class='verdict'><b>Verdict:</b> [review after viewing]</div>"

    html += """
    <h2>Summary: Recommended Combination</h2>
    <p>Based on the prototypes, consider using:</p>
    <ul>
    <li><b>Primary</b>: Tricolor scatter (19a) for the main figure — it's the most information-dense single plot</li>
    <li><b>Supplementary</b>: Dot-bar hybrid (19c) for ranked gene lists in tables</li>
    <li><b>Discovery</b>: Absence plot (19e) for identifying surprising non-associations</li>
    <li><b>Trajectory</b>: Pseudotime x expansion (19g) for dynamic patterns along the flow</li>
    <li><b>Overview</b>: Gene embedding (19f) for clustering genes by functional profile</li>
    </ul>
    """

    html += "</body></html>"

    report_path = os.path.join(OUTPUT_DIR, "viz_prototypes.html")
    with open(report_path, "w") as f:
        f.write(html)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()

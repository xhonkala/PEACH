#!/usr/bin/env python
"""PEACH v0.5 End-to-End HSC Analysis Pipeline.

Runs global characterization steps (1-9) on pooled hematopoietic data,
per-lineage breakout (10-11), biological transition flows (12-15),
and lineage comparison (16-17). Generates a self-contained HTML report.

Data: hsc_10k.h5ad (10k cells, Ensembl IDs in var_names, gene symbols in var["gene_symbols"])
Subset: 3 cell types (HSC, CMP, Mono) -> ~1,721 cells — focused myeloid trajectory
PCA: Recomputed on subset, sliced to 13 components

Usage: conda run -n archetype python scripts/run_e2e_hsc.py
"""

import matplotlib
matplotlib.use("Agg")

import base64
import io
import logging
import os
import pickle
import re
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="plotly")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CT_SHORT = {
    "hematopoietic stem cell": "HSC",
    "common myeloid progenitor": "CMP",
    "CD14-positive monocyte": "Mono",
}
CT_LONG = {v: k for k, v in CT_SHORT.items()}

FLOW_PAIRS = [
    ("HSC", "CMP"),    # progenitor commitment
    ("CMP", "Mono"),   # myeloid differentiation
    ("HSC", "Mono"),   # full trajectory
]

LINEAGE_GROUPS = {
    "progenitors": ["HSC", "CMP"],
    "myeloid": ["CMP", "Mono"],
    "full_trajectory": ["HSC", "CMP", "Mono"],
}

DATA_PATH = "/Users/honkala/Desktop/peach/data/hsc_10k.h5ad"
_RUN_ID = time.strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"outputs/e2e_hsc_myeloid/{_RUN_ID}"
REPORT_PATH = os.path.join(OUTPUT_DIR, "e2e_hsc_myeloid_report.html")

N_PCS = 13
MIN_CELLS_MODEL = 400
MIN_CELLS_FLOW = 200


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def safe_save_h5ad(adata, path):
    """Save adata to h5ad, stripping non-serializable objects from uns."""
    max_retries = 10
    stash = {}
    for attempt in range(max_retries):
        try:
            adata.write_h5ad(path)
            break
        except Exception as e:
            msg = str(e)
            match = re.search(r"key ['\"]/?uns/([^'\"]+)['\"]", msg)
            if match:
                bad_key = match.group(1).split("/")[0]
                if bad_key not in stash and bad_key in adata.uns:
                    log.warning(f"h5ad save: removing non-serializable uns['{bad_key}']")
                    stash[bad_key] = adata.uns.pop(bad_key)
                    continue
            removed_any = False
            for k in list(adata.uns.keys()):
                type_name = type(adata.uns[k]).__name__
                if type_name in ("Deep_AA", "VAE_Base", "FlowModel", "Module"):
                    stash[k] = adata.uns.pop(k)
                    removed_any = True
            if removed_any:
                continue
            log.error(f"h5ad save failed after {attempt+1} attempts: {e}")
            break
    adata.uns.update(stash)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("e2e_hsc")


# ---------------------------------------------------------------------------
# Gene name helper
# ---------------------------------------------------------------------------

def ensembl_to_symbol(adata, ids):
    """Map gene IDs to display symbols.

    After var_names are converted to symbols in step1, this is an identity
    mapping (kept for backward compatibility with downstream display code).
    Falls back to adata.var['gene_symbols'] lookup if var_names are still
    Ensembl IDs (e.g. when called on subsets created before the conversion).
    """
    # Fast path: check if first id looks like an Ensembl ID
    first = ids[0] if len(ids) > 0 else ""
    if first.startswith("ENSG") and "gene_symbols" in adata.var.columns:
        mapping = dict(zip(adata.var_names, adata.var["gene_symbols"]))
        return [mapping.get(g, g) for g in ids]
    # var_names are already symbols — identity mapping
    return list(ids)


def _display_genes(adata, gene_list, join=True):
    """Convert a list of gene IDs to symbols, optionally join as string."""
    symbols = ensembl_to_symbol(adata, gene_list)
    return ", ".join(symbols) if join else symbols


def short_ct(name):
    """Convert long cell type name to short label."""
    return CT_SHORT.get(name, name)


def long_ct(short):
    """Convert short label back to full cell type name."""
    return CT_LONG.get(short, short)


# ============================================================================
# HTMLReport class (identical to myeloid pipeline)
# ============================================================================

class HTMLReport:
    """Accumulates sections and writes a self-contained HTML file."""

    def __init__(self, title: str):
        self.title = title
        self.sections: list[dict] = []
        self.start_time = time.time()

    def fig_to_img(self, fig, caption: str = "", dpi: int = 150) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        html = f'<img src="data:image/png;base64,{b64}" style="max-width:100%;">'
        if caption:
            html += f"<p class='caption'>{caption}</p>"
        return html

    def plotly_to_div(self, fig, caption: str = "") -> str:
        import plotly.io as pio
        div = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
        html = f"<div class='plotly-wrap'>{div}</div>"
        if caption:
            html += f"<p class='caption'>{caption}</p>"
        return html

    def df_to_html(self, df: pd.DataFrame, caption: str = "", max_rows: int = 50) -> str:
        orig_len = len(df)
        if len(df) > max_rows:
            df = df.head(max_rows)
            note = f"<p><em>Showing first {max_rows} of {orig_len} rows.</em></p>"
        else:
            note = ""
        table = df.to_html(
            classes="styled-table", index=True,
            float_format=lambda x: f"{x:.4g}", border=0,
        )
        html = ""
        if caption:
            html += f"<p class='caption'><strong>{caption}</strong></p>"
        html += note + table
        return html

    def text(self, txt: str) -> str:
        return f"<p>{txt}</p>"

    def add_section(self, title: str, content_html: str, step_num: int | None = None):
        self.sections.append({"title": title, "content": content_html, "step": step_num})

    def save(self, path: str):
        elapsed = time.time() - self.start_time
        minutes, seconds = divmod(int(elapsed), 60)
        time_str = f"{minutes}m {seconds}s"
        sections_html = ""
        for i, s in enumerate(self.sections):
            step_tag = f"Step {s['step']}: " if s["step"] is not None else ""
            sections_html += f"""
            <details {'open' if i < 2 else ''}>
                <summary>{step_tag}{s['title']}</summary>
                <div class="section-body">{s['content']}</div>
            </details>
            """
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{self.title}</title>
<style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px 40px;
           background: #fafafa; color: #222; line-height: 1.6; max-width: 1400px; margin: 0 auto; }}
    h1 {{ border-bottom: 3px solid #0072B2; padding-bottom: 10px; color: #0072B2; }}
    .meta {{ color: #666; font-size: 0.9em; margin-bottom: 20px; }}
    details {{ margin-bottom: 12px; background: #fff; border: 1px solid #ddd;
              border-radius: 6px; overflow: hidden; }}
    summary {{ cursor: pointer; padding: 12px 16px; background: #f0f4f8; font-weight: 600;
              font-size: 1.05em; border-bottom: 1px solid #ddd; }}
    summary:hover {{ background: #e2eaf2; }}
    .section-body {{ padding: 16px; }}
    .caption {{ color: #555; font-style: italic; font-size: 0.92em; margin-top: 4px; }}
    .styled-table {{ border-collapse: collapse; margin: 1em 0; font-size: 0.9em; width: auto; }}
    .styled-table th {{ background: #2c3e50; color: white; padding: 8px 12px; text-align: left; font-weight: 600; }}
    .styled-table td {{ padding: 6px 12px; border-bottom: 1px solid #e0e0e0; }}
    .styled-table tr:nth-child(even) {{ background: #f8f9fa; }}
    .styled-table tr:hover {{ background: #eef2f7; }}
    .plotly-wrap {{ margin: 10px 0; }}
    .error {{ background: #fff0f0; border-left: 4px solid #c0392b; padding: 10px 14px;
             margin: 10px 0; border-radius: 4px; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                   gap: 10px; margin: 12px 0; }}
    .metric-card {{ background: #f0f4f8; padding: 12px; border-radius: 6px; text-align: center; }}
    .metric-card .value {{ font-size: 1.6em; font-weight: 700; color: #0072B2; }}
    .metric-card .label {{ font-size: 0.85em; color: #666; }}
</style>
</head>
<body>
<h1>{self.title}</h1>
<p class="meta">Generated: {time.strftime('%Y-%m-%d %H:%M')} | Total runtime: {time_str}
   | {len(self.sections)} sections</p>
{sections_html}
</body>
</html>"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(html)
        log.info(f"Report saved to {path}")


# ============================================================================
# Helper utilities
# ============================================================================

def metric_card(value, label: str) -> str:
    if isinstance(value, float):
        value = f"{value:.4f}"
    return f'<div class="metric-card"><div class="value">{value}</div><div class="label">{label}</div></div>'


def metric_grid(cards: list[str]) -> str:
    return '<div class="metric-grid">' + "".join(cards) + "</div>"


def error_html(msg: str) -> str:
    return f'<div class="error">{msg}</div>'


def safe_plotly_html(report: "HTMLReport", fig, caption: str = "") -> str:
    """Safely convert plotly figure; return error html on failure."""
    try:
        return report.plotly_to_div(fig, caption)
    except Exception as e:
        return error_html(f"Plotly render failed: {e}")


def safe_mpl_html(report: "HTMLReport", caption: str = "", dpi: int = 150) -> str:
    """Capture current matplotlib figure; return error html on failure."""
    try:
        fig = plt.gcf()
        html = report.fig_to_img(fig, caption=caption, dpi=dpi)
        plt.close("all")
        return html
    except Exception as e:
        plt.close("all")
        return error_html(f"Matplotlib render failed: {e}")


def _dense_X(adata):
    """Get dense X matrix."""
    return adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X


# ============================================================================
# Step functions
# ============================================================================

def step1_dataset_prep(adata, report):
    """Step 1: Dataset preparation -- subset, PCA recompute, pathways, overview."""
    import peach as pc
    import scanpy as sc

    html = ""

    # -- Subset to 6 cell types -----------------------------------------------
    ct_mask = adata.obs["cell_type"].isin(list(CT_SHORT.keys()))
    n_before = adata.shape[0]
    adata_sub = adata[ct_mask].copy()
    log.info(f"Subset: {n_before} -> {adata_sub.shape[0]} cells ({len(CT_SHORT)} cell types)")

    # Add short cell type labels
    adata_sub.obs["cell_type_short"] = adata_sub.obs["cell_type"].map(CT_SHORT)

    # Cell type counts
    ct_counts = adata_sub.obs["cell_type_short"].value_counts()
    ct_rows = [{"Cell type": ct, "N cells": int(n)} for ct, n in ct_counts.items()]
    html += report.df_to_html(pd.DataFrame(ct_rows), caption="Cell type counts (subset)")

    # -- Recompute PCA on subset -----------------------------------------------
    log.info(f"Recomputing PCA on subset ({adata_sub.shape[0]} cells, {N_PCS} components)...")
    sc.pp.pca(adata_sub, n_comps=N_PCS)
    html += report.text(f"PCA recomputed on {adata_sub.shape[0]} cells with {N_PCS} components.")

    # -- Convert var_names from Ensembl IDs to gene symbols --------------------
    # Required for pathway scoring — MSigDB uses symbols. Do this after PCA so
    # loadings are already stored under the Ensembl-named axis.
    if "gene_symbols" in adata_sub.var.columns:
        symbols = adata_sub.var["gene_symbols"].values.copy()
        # Handle duplicates by appending suffix
        seen = {}
        for i, s in enumerate(symbols):
            if s in seen:
                seen[s] += 1
                symbols[i] = f"{s}_{seen[s]}"
            else:
                seen[s] = 0
        adata_sub.var_names = pd.Index(symbols)
        log.info(f"Converted var_names to gene symbols ({adata_sub.n_vars} genes)")
        html += report.text(
            f"var_names converted from Ensembl IDs to gene symbols ({adata_sub.n_vars} genes). "
            "Duplicate symbols disambiguated with numeric suffix."
        )

    # Overview cards
    n_cells, n_genes = adata_sub.shape
    cards = [
        metric_card(n_cells, "Cells (subset)"),
        metric_card(n_genes, "Genes"),
        metric_card(N_PCS, "PCA dims"),
        metric_card(len(CT_SHORT), "Cell types"),
    ]
    if "Study" in adata_sub.obs.columns:
        cards.append(metric_card(adata_sub.obs["Study"].nunique(), "Studies"))
    if "CyclePhase" in adata_sub.obs.columns:
        cards.append(metric_card(adata_sub.obs["CyclePhase"].nunique(), "Cell cycle phases"))
    html += metric_grid(cards)

    # X data summary
    X = _dense_X(adata_sub)
    html += report.text(
        f"X range: [{X.min():.2f}, {X.max():.2f}] | "
        f"X mean: {X.mean():.3f} | X std: {X.std():.3f} | "
        f"Note: X is sparse logcounts (not z-scored)."
    )

    # Gene name check
    example_vars = list(adata_sub.var_names[:5])
    html += report.text(
        f"Gene ID format after conversion: symbol (e.g. {example_vars[0]}). "
        f"var_names are now gene symbols; display functions use them directly."
    )

    # Pathway scores
    try:
        log.info("Loading C5:BP (GO Biological Process) pathway networks...")
        net = pc.pp.load_pathway_networks(sources=["c5_bp"], verbose=False)
        log.info("Computing pathway scores...")
        pc.pp.compute_pathway_scores(adata_sub, net=net, verbose=False)
        n_pathways = adata_sub.obsm["pathway_scores"].shape[1]
        pathway_names = adata_sub.uns.get("pathway_scores_pathways", [])
        html += report.text(f"Pathway scores computed: {n_pathways} C5:BP (GO Biological Process) pathways.")
        if len(pathway_names) > 0:
            html += report.text(f"Example pathways: {', '.join(pathway_names[:5])}...")

        fig, ax = plt.subplots(figsize=(10, 3))
        scores = np.asarray(adata_sub.obsm["pathway_scores"])
        ax.hist(scores.ravel(), bins=80, color="#0072B2", alpha=0.7, edgecolor="none")
        ax.set_xlabel("Pathway score")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of all pathway scores")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        html += report.fig_to_img(fig, caption="Pathway score distribution (all pathways pooled)")
        plt.close("all")
    except Exception as e:
        html += error_html(f"Pathway scoring failed: {e}. Continuing without pathways.")
        log.warning(f"Pathway scoring failed: {e}")

    # Prepare training data
    log.info("Preparing training data...")
    pc.pp.prepare_training(adata_sub, batch_size=128)

    report.add_section("Dataset Preparation", html, step_num=1)
    return adata_sub


def step2_hyperparameter_fit(adata, report):
    """Step 2: Hyperparameter search + final model training."""
    import peach as pc

    html = ""

    # Hyperparameter search
    log.info("Running hyperparameter search (K=3..8)...")
    cv_summary = pc.tl.hyperparameter_search(
        adata,
        n_archetypes_range=[3, 4, 5, 6, 7, 8],
        hidden_dims_options=[[64, 128], [128, 256]],
        inflation_factor_range=[1.0, 1.25, 1.5, 1.75],
        cv_folds=3,
        max_epochs_cv=10,
        subsample_fraction=0.8,
    )

    # CV results table -- FIX #1: debug raw ranked list, check for 0s and -inf
    ranked = cv_summary.rank_by_metric("archetype_r2")
    html += report.text(f"Raw CV results: {len(ranked)} configurations evaluated.")

    # Debug: show raw top 10 before filtering
    debug_rows = []
    for i, r in enumerate(ranked[:10]):
        hp = r["hyperparameters"]
        debug_rows.append({
            "Rank": i + 1,
            "K": hp["n_archetypes"],
            "hidden_dims": str(hp.get("hidden_dims", "N/A")),
            "metric_value": r["metric_value"],
            "is_inf": r["metric_value"] <= -1e6,
            "is_zero": r["metric_value"] == 0.0,
        })
    html += report.df_to_html(pd.DataFrame(debug_rows),
                              caption="Raw CV ranked list (top 10, before filtering)")

    # Filter out -inf and flag 0s
    ranked_clean = [r for r in ranked if r["metric_value"] > -1e6]
    n_zero = sum(1 for r in ranked_clean if r["metric_value"] == 0.0)
    if n_zero > 0:
        html += report.text(f"WARNING: {n_zero} configurations have R-squared = 0.0 exactly. "
                            "This may indicate failed training or degenerate fits.")

    if not ranked_clean:
        html += error_html("All CV configurations failed or returned -inf. Using fallback K=4.")
        ranked_clean = [{"hyperparameters": {"n_archetypes": 4, "hidden_dims": [128, 256]},
                         "metric_value": 0.0, "std_error": 0.0}]

    cv_rows = []
    for r in ranked_clean:
        hp = r["hyperparameters"]
        cv_rows.append({
            "K": hp["n_archetypes"],
            "hidden_dims": str(hp.get("hidden_dims", "N/A")),
            "inflation": hp.get("inflation_factor", "N/A"),
            "R2": f"{r['metric_value']:.4f}",
            "SE": f"{r.get('std_error', 0):.4f}",
        })
    cv_df = pd.DataFrame(cv_rows)
    html += report.df_to_html(cv_df, caption="CV search results (ranked by R-squared, filtered)")

    # Elbow curve (plotly) -- FIX #2: wrap in try/except
    try:
        fig_elbow = pc.pl.elbow_curve(cv_summary, metrics=["archetype_r2", "rmse"])
        html += safe_plotly_html(report, fig_elbow, "Elbow curve: R-squared and RMSE vs K")
    except Exception as e:
        html += error_html(f"Elbow curve failed: {e}")

    # Select best R² configuration (not elbow — want maximum R² for archetypes)
    best = ranked_clean[0]  # ranked_clean is sorted by metric_value descending

    best_hp = best["hyperparameters"]
    best_K = best_hp["n_archetypes"]
    best_hidden = best_hp.get("hidden_dims", [128, 256])
    best_r2 = best["metric_value"]

    html += report.text(f"Best R² selection: K={best_K}")
    html += metric_grid([
        metric_card(best_K, "Best K"),
        metric_card(f"{best_r2:.4f}", "Best CV R-squared"),
        metric_card(str(best_hidden), "Hidden dims"),
    ])

    # Train final model
    log.info(f"Training final model: K={best_K}, hidden_dims={best_hidden}...")
    results = pc.tl.train_archetypal(
        adata,
        n_archetypes=best_K,
        n_epochs=100,
        hidden_dims=best_hidden,
        kld_weight=0.1,
        early_stopping=True,
        early_stopping_patience=15,
    )
    final_r2 = results.get("final_archetype_r2", "N/A")
    html += metric_grid([
        metric_card(f"{final_r2:.4f}" if isinstance(final_r2, float) else final_r2,
                    "Final R-squared"),
    ])

    # Training metrics (plotly) -- FIX #2: try/except
    try:
        fig_train = pc.pl.training_metrics(results["history"], display=False)
        if fig_train is not None:
            html += safe_plotly_html(report, fig_train, "Training metrics over epochs")
    except Exception as e:
        html += error_html(f"Training metrics plot failed: {e}")

    # Run annotation
    log.info("Running coordinate + assignment annotation...")
    pc.tl.archetypal_coordinates(adata, verbose=False)
    pc.tl.assign_archetypes(adata, verbose=False)
    pc.tl.extract_archetype_weights(adata, verbose=False)

    # FIX #13: Print stored adata attributes after annotation
    peach_obs = [c for c in adata.obs.columns if "archetype" in c.lower() or "peach" in c.lower()]
    peach_obsm = [k for k in adata.obsm.keys() if "archetype" in k.lower() or "peach" in k.lower()
                  or k in ("cell_archetype_weights", "archetype_coordinates", "archetype_distances")]
    peach_uns = [k for k in adata.uns.keys() if k.startswith("peach_")]
    html += report.text(
        f"<strong>Stored adata attributes after annotation:</strong><br>"
        f"obs columns: {', '.join(peach_obs) if peach_obs else 'none'}<br>"
        f"obsm keys: {', '.join(peach_obsm) if peach_obsm else 'none'}<br>"
        f"uns keys (peach_): {', '.join(peach_uns) if peach_uns else 'none'}"
    )

    # Archetypal space scatter colored by cell type
    try:
        fig_space = pc.pl.archetypal_space(adata, color_by="cell_type_short",
                                           title="Archetypal space (cell type)")
        html += safe_plotly_html(report, fig_space, "Archetypal space colored by cell type")
    except Exception as e:
        html += error_html(f"pc.pl.archetypal_space failed: {e}")

    # Also by Study if available
    if "Study" in adata.obs.columns:
        try:
            fig_space2 = pc.pl.archetypal_space(adata, color_by="Study",
                                                title="Archetypal space (study)")
            html += safe_plotly_html(report, fig_space2, "Archetypal space colored by study")
        except Exception as e:
            html += error_html(f"pc.pl.archetypal_space (Study) failed: {e}")

    # Archetype statistics (returns a dict of numeric stats, not a figure)
    try:
        arch_stats = pc.pl.archetype_statistics(adata, verbose=False)
        if arch_stats is not None and isinstance(arch_stats, dict):
            stat_rows = [
                {"Statistic": k, "Value": (f"{v:.4f}" if isinstance(v, float) else str(v))}
                for k, v in arch_stats.items()
                if k not in ("distance_matrix", "nearest_pair", "farthest_pair")
                   and not isinstance(v, np.ndarray)
            ]
            if arch_stats.get("nearest_pair") is not None:
                j, k = arch_stats["nearest_pair"]
                stat_rows.append({"Statistic": "nearest_pair", "Value": f"A{j+1}-A{k+1}"})
            if arch_stats.get("farthest_pair") is not None:
                j, k = arch_stats["farthest_pair"]
                stat_rows.append({"Statistic": "farthest_pair", "Value": f"A{j+1}-A{k+1}"})
            html += report.df_to_html(pd.DataFrame(stat_rows), caption="Archetype position statistics")
    except Exception as e:
        html += error_html(f"pc.pl.archetype_statistics failed: {e}")

    # Weight distribution
    try:
        weights = adata.obsm["cell_archetype_weights"]
        K = weights.shape[1]
        fig, axes = plt.subplots(1, K, figsize=(3 * K, 3), sharey=True)
        if K == 1:
            axes = [axes]
        for k in range(K):
            axes[k].hist(weights[:, k], bins=50, color="#0072B2", alpha=0.7, edgecolor="none")
            axes[k].set_title(f"A{k+1}")
            axes[k].set_xlabel("Weight")
            axes[k].spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Count")
        fig.suptitle("Cell-archetype weight distributions", y=1.02)
        fig.tight_layout()
        html += report.fig_to_img(fig, caption="Per-archetype weight distributions")
        plt.close("all")
    except Exception as e:
        html += error_html(f"Weight distribution plot failed: {e}")

    report.add_section("Hyperparameter Search & Model Training", html, step_num=2)
    return results


def step3_simplex_regression(adata, report):
    """Step 3: Simplex regression (gene + pathway), pattern classification."""
    import peach as pc

    html = ""

    # Gene simplex regression (with permutation test for model significance)
    log.info("Running gene simplex regression (degree 1+2) with permutation test...")
    gene_reg = pc.tl.gene_simplex_regression(adata, max_degree=2, robust_se=True,
                                              store_residuals=True,
                                              permutation_test=True,
                                              n_permutations=200)

    n_features = len(gene_reg["feature_names"])
    r2_d1 = np.asarray(gene_reg["r_squared_degree1"])
    f_fdr = np.asarray(gene_reg.get("f_pvalue_fdr", np.ones(n_features)))
    n_sig = int((f_fdr < 0.05).sum())

    # Permutation-significant count
    perm_fdr = np.asarray(gene_reg.get("permutation_pvalue_fdr", np.ones(n_features)))
    n_perm_sig = int((perm_fdr < 0.05).sum())

    html += metric_grid([
        metric_card(n_features, "Genes tested"),
        metric_card(n_sig, "F-test FDR-significant (q<0.05)"),
        metric_card(n_perm_sig, "Permutation FDR-significant (q<0.05)"),
        metric_card(f"{r2_d1.mean():.4f}", "Mean R-squared (degree 1)"),
        metric_card(f"{np.median(r2_d1):.4f}", "Median R-squared"),
        metric_card(f"{r2_d1.max():.4f}", "Max R-squared"),
    ])

    # FIX #14: Interpretation context
    html += report.text(
        "Note: Simplex regression R-squared measures how well archetype weights predict "
        "individual gene expression (per-gene fit quality). This differs from archetypal R-squared "
        "(step 2), which measures how well the model reconstructs the full PCA space. "
        "Simplex R-squared captures the gene-specific signal in the archetypal coordinate system."
    )

    # --- Section 2: Pattern classification counts (moved up) ---
    log.info("Classifying feature patterns...")
    pattern_result = None
    try:
        pattern_result = pc.tl.classify_feature_patterns(adata)
        pattern_counts = pattern_result.get("pattern_counts", {})
        html += report.text("Pattern classification counts:")
        pattern_df = pd.DataFrame(
            [{"Pattern": k, "Count": v} for k, v in pattern_counts.items()]
        )
        html += report.df_to_html(pattern_df, caption="Feature pattern counts")

        try:
            fig_pat = pc.pl.pattern_summary(adata, show=False)
            html += safe_plotly_html(report, fig_pat, "Pattern type summary")
        except Exception as e:
            html += error_html(f"Pattern summary plot failed: {e}")
    except Exception as e:
        html += error_html(f"Pattern classification failed: {e}")

    # --- Section 3: Archetype-exclusive features dotplot ---
    try:
        fig_dot = pc.pl.archetype_regression_dotplot(adata, top_n=10, exclusive_only=True, show=False)
        html += safe_plotly_html(report, fig_dot,
                                 "Archetype regression dotplot (top 10 exclusive genes per archetype)")
    except Exception as e:
        html += error_html(f"Regression dotplot (exclusive) failed: {e}")

    # --- Section 4: Archetype-exclusive features enumerated per archetype ---
    if pattern_result is not None:
        arch_feat_map = pattern_result.get("archetype_features", {})
        if arch_feat_map:
            map_rows = []
            vertex_coefs_map = np.asarray(gene_reg["vertex_coefficients"])
            feat_names_map = list(gene_reg["feature_names"])
            for k, feats in arch_feat_map.items():
                display_k = f"A{k+1}" if isinstance(k, int) else str(k)
                # Sort by |coefficient| for this archetype
                feat_coefs = []
                for f in feats:
                    if f in feat_names_map:
                        idx = feat_names_map.index(f)
                        coef_k = vertex_coefs_map[idx, k] if isinstance(k, int) and k < vertex_coefs_map.shape[1] else 0
                        feat_coefs.append((f, abs(coef_k)))
                    else:
                        feat_coefs.append((f, 0))
                feat_coefs.sort(key=lambda x: -x[1])
                sorted_feats = [f for f, _ in feat_coefs]
                feat_symbols = ensembl_to_symbol(adata, sorted_feats[:10])
                map_rows.append({
                    "Archetype": display_k,
                    "N features": len(feats),
                    "Top features (by |β|)": ", ".join(feat_symbols),
                })
            map_df = pd.DataFrame(map_rows)
            html += report.df_to_html(map_df, caption="Archetype-exclusive feature map (sorted by |coefficient|)")

    # --- Section 5: Gene set (pathway) exclusive features dotplot + enumeration ---
    # (computed after pathway regression below, added here as placeholder variable)
    pw_pattern_result = None

    # --- Section 6: Structured/interaction features (cooperative vs tradeoff) ---
    try:
        int_coefs = gene_reg.get("interaction_coefficients")
        int_pairs = gene_reg.get("interaction_pairs", [])
        int_fdr = gene_reg.get("interaction_pvalues_fdr")
        vertex_coefs = np.asarray(gene_reg["vertex_coefficients"])
        feat_names = list(gene_reg["feature_names"])

        if int_coefs is not None and len(int_pairs) > 0 and int_fdr is not None:
            int_coefs = np.asarray(int_coefs)
            int_fdr = np.asarray(int_fdr)

            coop_rows = []
            for feat_idx in range(len(feat_names)):
                for pair_idx, (j, k) in enumerate(int_pairs):
                    if int_fdr[feat_idx, pair_idx] < 0.05:
                        beta_j = vertex_coefs[feat_idx, j]
                        beta_k = vertex_coefs[feat_idx, k]
                        if np.sign(beta_j) == np.sign(beta_k):
                            itype = "cooperative"
                        else:
                            itype = "tradeoff"
                        coop_rows.append({
                            "Feature": feat_names[feat_idx],
                            "Feature_symbol": ensembl_to_symbol(adata, [feat_names[feat_idx]])[0],
                            "Pair": f"A{j+1}-A{k+1}",
                            "Type": itype,
                            "beta_j": f"{beta_j:.3f}",
                            "beta_k": f"{beta_k:.3f}",
                            "int_coef": f"{int_coefs[feat_idx, pair_idx]:.3f}",
                            "FDR q": f"{int_fdr[feat_idx, pair_idx]:.2e}",
                        })

            if coop_rows:
                coop_df = pd.DataFrame(coop_rows)
                n_coop = (coop_df["Type"] == "cooperative").sum()
                n_trade = (coop_df["Type"] == "tradeoff").sum()
                html += metric_grid([
                    metric_card(len(coop_rows), "Significant interactions"),
                    metric_card(n_coop, "Cooperative"),
                    metric_card(n_trade, "Tradeoff"),
                ])

                html += report.text(
                    "Interaction coefficients (beta_int): the product term w_j*w_k in the Scheffe "
                    "polynomial. Positive beta_int = synergistic effect (gene upregulated when cell has "
                    "high weight on both archetypes). Negative beta_int = antagonistic (gene suppressed "
                    "in the blending zone). <br>"
                    "<strong>Cooperative:</strong> both archetypes increase/decrease the gene together "
                    "(same-sign vertex beta). <br>"
                    "<strong>Tradeoff:</strong> one archetype increases while the other decreases the gene "
                    "(opposite-sign vertex beta)."
                )

                # Nested groupby table sorted by |int_coef| within each pair x type
                int_df_b2 = coop_df.copy()
                int_df_b2["abs_coef"] = pd.to_numeric(int_df_b2["int_coef"], errors="coerce").abs()
                int_df_b2_sorted = int_df_b2.sort_values(
                    ["Pair", "Type", "abs_coef"], ascending=[True, True, False]
                )
                grouped_b2 = int_df_b2_sorted.groupby(["Pair", "Type"]).head(5)
                display_cols_b2 = ["Feature_symbol", "Pair", "Type", "beta_j", "beta_k",
                                   "int_coef", "FDR q"]
                html += report.df_to_html(
                    grouped_b2[display_cols_b2].drop_duplicates(),
                    caption="Top 2nd-degree interactions (grouped by pair x type, ranked by |coef|)"
                )
                html += report.text(
                    "Interpretation: 'cooperative' = both vertex coefficients have the same sign "
                    "(features increase/decrease together near both archetypes). "
                    "'tradeoff' = opposing signs (feature increases near one archetype, decreases near the other). "
                    "int_coef > 0 = synergistic amplification; int_coef < 0 = mutual suppression."
                )

                # Features with significant interactions across multiple archetype pairs
                gene_pair_counts = int_df_b2.groupby("Feature_symbol")["Pair"].nunique()
                multi_pair = gene_pair_counts[gene_pair_counts > 1].sort_values(ascending=False)
                if len(multi_pair) > 0:
                    mp_df = pd.DataFrame({"Gene": multi_pair.index, "N_pairs": multi_pair.values})
                    pair_details = (int_df_b2.groupby("Feature_symbol")["Pair"]
                                    .apply(lambda x: ", ".join(sorted(set(x)))).to_dict())
                    mp_df["Pairs"] = mp_df["Gene"].map(pair_details)
                    html += report.df_to_html(mp_df.head(20),
                                              caption="Features with interactions across multiple archetype pairs")
            else:
                html += report.text("No significant interaction terms at FDR < 0.05.")
        else:
            html += report.text("Interaction terms not available in regression results.")
    except Exception as e:
        html += error_html(f"Cooperative/tradeoff classification failed: {e}")

    # --- Section 7: Interaction dotplot (degree 2) ---
    try:
        fig_int_dot = pc.pl.archetype_regression_dotplot(adata, top_n=10, degree=2, show=False)
        html += safe_plotly_html(report, fig_int_dot,
                                 "Interaction regression dotplot (top 10 per archetype pair, degree 2)")
    except Exception as e:
        html += error_html(f"Interaction dotplot failed: {e}")

    # --- Section 8: Vertex coefficient heatmap grouped by archetype ---
    try:
        vertex_coefs_all = np.asarray(gene_reg["vertex_coefficients"])
        feat_names_all = list(gene_reg["feature_names"])
        n_show_hm = min(50, vertex_coefs_all.shape[0])
        max_abs = np.abs(vertex_coefs_all).max(axis=1)
        top_idx = np.argsort(max_abs)[-n_show_hm:]
        dom_arch = np.argmax(np.abs(vertex_coefs_all[top_idx]), axis=1) + 1
        sort_order = np.argsort(dom_arch)
        top_idx_sorted = top_idx[sort_order]
        dom_arch_sorted = dom_arch[sort_order]
        K_hm = vertex_coefs_all.shape[1]
        fig_hm, ax_hm = plt.subplots(figsize=(max(5, K_hm * 1.5), max(5, n_show_hm * 0.35)))
        im_hm = ax_hm.imshow(vertex_coefs_all[top_idx_sorted], aspect="auto", cmap="RdBu_r")
        ax_hm.set_xticks(range(K_hm))
        ax_hm.set_xticklabels([f"A{k+1}" for k in range(K_hm)])
        ax_hm.set_yticks(range(n_show_hm))
        ylabels = [f"{ensembl_to_symbol(adata, [feat_names_all[i]])[0]} [A{dom_arch_sorted[r]}]"
                   for r, i in enumerate(top_idx_sorted)]
        ax_hm.set_yticklabels(ylabels, fontsize=7)
        plt.colorbar(im_hm, ax=ax_hm, label="Coefficient", shrink=0.6)
        ax_hm.set_title("Vertex coefficients (grouped by dominant archetype)")
        fig_hm.tight_layout()
        html += report.fig_to_img(fig_hm, caption="Coefficient heatmap grouped by dominant archetype (bracket = dominant)")
        plt.close("all")
    except Exception as e:
        html += error_html(f"Grouped coefficient heatmap failed: {e}")
        plt.close("all")

    # --- Section 9: Archetype radar ---
    try:
        fig_radar = pc.pl.archetype_radar(adata, top_n=8, order_by_similarity=True, show=False)
        html += safe_plotly_html(report, fig_radar, "Archetype radar (top features)")
        html += report.text(
            "Radar axes ordered by archetype similarity (Spearman correlation on regression coefficients, "
            "spectral 1D embedding). Adjacent archetypes on the radar have the most similar feature profiles."
        )
    except Exception as e:
        html += error_html(f"Archetype radar failed: {e}")

    # --- Section 10: Pathway simplex regression ---
    if "pathway_scores" in adata.obsm:
        log.info("Running pathway simplex regression...")
        try:
            pw_reg = pc.tl.pathway_simplex_regression(adata, max_degree=2, robust_se=True)
            pw_r2 = np.asarray(pw_reg["r_squared_degree1"])
            pw_fdr = np.asarray(pw_reg.get("f_pvalue_fdr", np.ones(len(pw_r2))))
            n_pw_sig = int((pw_fdr < 0.05).sum())
            pw_names = list(pw_reg.get("feature_names", []))
            html += metric_grid([
                metric_card(len(pw_r2), "Pathways tested"),
                metric_card(n_pw_sig, "FDR-significant pathways"),
                metric_card(f"{pw_r2.mean():.4f}", "Mean pathway R-squared"),
                metric_card(f"{pw_r2.max():.4f}", "Max pathway R-squared"),
            ])

            if len(pw_names) > 0:
                pw_df = pd.DataFrame({"Pathway": pw_names, "R2": pw_r2, "FDR_q": pw_fdr})
                pw_df = pw_df.sort_values("R2", ascending=False).head(20)
                html += report.df_to_html(pw_df, caption="Top 20 pathways by R-squared")

            # Pathway coefficient heatmap
            try:
                pw_coefs = pw_reg.get("vertex_coefficients")
                if pw_coefs is not None and len(pw_names) > 0:
                    pw_coefs = np.asarray(pw_coefs)
                    n_show = min(25, pw_coefs.shape[0])
                    mean_abs = np.abs(pw_coefs).max(axis=1)
                    top_idx = np.argsort(mean_abs)[-n_show:][::-1]
                    K_pw = pw_coefs.shape[1]
                    fig, ax = plt.subplots(figsize=(max(5, K_pw * 1.5), max(4, n_show * 0.35)))
                    im = ax.imshow(pw_coefs[top_idx], aspect="auto", cmap="RdBu_r")
                    ax.set_xticks(range(K_pw))
                    ax.set_xticklabels([f"A{k+1}" for k in range(K_pw)])
                    ax.set_yticks(range(n_show))
                    ax.set_yticklabels([pw_names[i] for i in top_idx], fontsize=8)
                    plt.colorbar(im, ax=ax, label="Coefficient", shrink=0.6)
                    ax.set_title("Pathway coefficients (top by max |beta|)")
                    fig.tight_layout()
                    html += report.fig_to_img(fig, caption="Pathway coefficient heatmap")
                    plt.close("all")
            except Exception as e:
                html += error_html(f"Pathway coefficient heatmap failed: {e}")
                plt.close("all")

            # Pathway pattern classification + exclusive pathways per archetype
            try:
                _gene_patterns_stash = adata.uns.pop("peach_feature_patterns", None)
                pw_pattern_result = pc.tl.classify_feature_patterns(adata, regression_result=pw_reg)
                if pw_pattern_result is not None:
                    pw_pattern_counts = pw_pattern_result.get("pattern_counts", {})
                    if pw_pattern_counts:
                        pw_pat_df = pd.DataFrame(
                            [{"Pattern": k, "Count": v} for k, v in pw_pattern_counts.items()]
                        )
                        html += report.df_to_html(pw_pat_df, caption="Pathway pattern classification")

                    # Gene set exclusive features: enumerate exclusive pathways per archetype
                    pw_arch_map = pw_pattern_result.get("archetype_features", {})
                    if pw_arch_map:
                        try:
                            fig_pw_dot = pc.pl.archetype_regression_dotplot(
                                adata, top_n=10, exclusive_only=True, show=False)
                            html += safe_plotly_html(report, fig_pw_dot,
                                                     "Pathway regression dotplot (exclusive pathways)")
                        except Exception as e:
                            html += error_html(f"Pathway dotplot failed: {e}")

                        for k, pathways in pw_arch_map.items():
                            if pathways:
                                display_k = f"A{k+1}" if isinstance(k, int) else str(k)
                                html += report.text(
                                    f"{display_k} exclusive pathways: {', '.join(pathways[:10])}"
                                    + ("..." if len(pathways) > 10 else "")
                                )

                # Pathway exclusive dotplot (swap generic key to pathway data)
                try:
                    _gene_reg_stash = adata.uns.pop("peach_simplex_regression", None)
                    adata.uns["peach_simplex_regression"] = pw_reg
                    fig_pw_dot = pc.pl.archetype_regression_dotplot(
                        adata, top_n=10, exclusive_only=True, show=False
                    )
                    html += safe_plotly_html(report, fig_pw_dot, "Pathway exclusive dotplot")
                except Exception as e:
                    html += error_html(f"Pathway exclusive dotplot failed: {e}")
                finally:
                    adata.uns.pop("peach_simplex_regression", None)
                    if _gene_reg_stash is not None:
                        adata.uns["peach_simplex_regression"] = _gene_reg_stash

            except Exception as e:
                html += error_html(f"Pathway pattern classification failed: {e}")
            finally:
                if _gene_patterns_stash is not None:
                    adata.uns["peach_feature_patterns"] = _gene_patterns_stash

        except Exception as e:
            html += error_html(f"Pathway simplex regression failed: {e}")

    report.add_section("Simplex Regression & Pattern Classification", html, step_num=3)
    return gene_reg


def step4_hypergeometric(adata, report):
    """Step 4: Conditional associations -- cell_type is the primary condition."""
    import peach as pc

    html = ""
    cond_results = {}

    # Use cell_type_short, Study, CyclePhase as condition columns
    for col in ["cell_type_short", "Study", "CyclePhase"]:
        if col not in adata.obs.columns:
            continue
        log.info(f"Conditional associations: {col}...")
        try:
            cond_df = pc.tl.conditional_associations(adata, obs_column=col, verbose=False)
            cond_results[col] = cond_df

            sig_col = "significant" if "significant" in cond_df.columns else None
            if sig_col:
                sig_df = cond_df[cond_df[sig_col] == True]
            else:
                sig_df = cond_df[cond_df["fdr_pvalue"] < 0.05]
            html += report.text(f"<strong>{col}</strong>: {len(sig_df)} significant associations "
                                f"out of {len(cond_df)} tests.")

            display_cols = ["archetype", "condition", "observed", "expected",
                            "odds_ratio", "fdr_pvalue", "significant"]
            display_cols = [c for c in display_cols if c in cond_df.columns]
            html += report.df_to_html(cond_df[display_cols], caption=f"Conditional associations: {col}")

            # Enrichment heatmap
            try:
                pivot = cond_df.pivot_table(
                    index="archetype", columns="condition",
                    values="odds_ratio", aggfunc="first"
                )
                fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 1.2),
                                                max(4, pivot.shape[0] * 0.8)))
                import matplotlib.colors as mcolors
                finite_vals = pivot.values[np.isfinite(pivot.values) & (pivot.values > 0)]
                if len(finite_vals) == 0:
                    raise ValueError("No finite positive odds ratios")
                vmin = max(0.1, float(finite_vals.min()))
                vmax = max(vmin * 10, float(finite_vals.max()))
                # Replace inf values with vmax for display
                display_vals = np.where(np.isfinite(pivot.values) & (pivot.values > 0), pivot.values, vmax)
                norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
                im = ax.imshow(display_vals, aspect="auto", cmap="RdBu_r", norm=norm)
                ax.set_xticks(range(pivot.shape[1]))
                ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
                ax.set_yticks(range(pivot.shape[0]))
                ax.set_yticklabels(pivot.index)
                plt.colorbar(im, ax=ax, label="Odds ratio (log scale)")
                ax.set_title(f"Enrichment: {col}")
                fig.tight_layout()
                html += report.fig_to_img(fig, caption=f"Odds ratio heatmap: {col}")
                plt.close("all")
            except Exception as e:
                html += error_html(f"Enrichment heatmap ({col}) failed: {e}")
                plt.close("all")

        except Exception as e:
            html += error_html(f"Conditional associations for {col} failed: {e}")

    # Proportion bars: cell type composition per archetype
    if "archetypes" in adata.obs.columns and "cell_type_short" in adata.obs.columns:
        try:
            ct = pd.crosstab(adata.obs["archetypes"], adata.obs["cell_type_short"], normalize="index")
            fig, ax = plt.subplots(figsize=(10, 5))
            ct.plot(kind="bar", stacked=True, ax=ax, colormap="Set2", edgecolor="none")
            ax.set_ylabel("Proportion")
            ax.set_title("Cell type composition per archetype")
            ax.legend(title="Cell type", bbox_to_anchor=(1.02, 1), loc="upper left")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            html += report.fig_to_img(fig, caption="Stacked bar: cell type proportions per archetype")
            plt.close("all")
        except Exception as e:
            html += error_html(f"Proportion bar failed: {e}")
            plt.close("all")

    report.add_section("Hypergeometric Conditional Associations", html, step_num=4)
    return cond_results


def step5_wald_contrasts(adata, report, gene_reg):
    """Step 5: Wald contrasts between archetypes."""
    import peach as pc

    html = ""

    log.info("Computing Wald contrasts...")
    contrast_result = pc.tl.archetype_contrasts(adata)

    pairs = contrast_result.get("pairs", [])
    feature_names = list(contrast_result.get("feature_names", []))
    html += report.text(f"Tested {len(pairs)} archetype pairs across {len(feature_names)} features.")

    html += report.text(
        "Null hypothesis: beta_j = beta_k (no coefficient difference between archetype pair). "
        "FDR via Benjamini-Hochberg across all features x pairs."
    )

    # Summary per pair
    summary_rows = []
    for pair in pairs:
        pair_key = str(tuple(pair) if isinstance(pair, list) else pair)
        pvals = np.asarray(contrast_result["pvalues_fdr"][pair_key])
        n_sig = int((pvals < 0.05).sum())
        delta = np.asarray(contrast_result["delta_beta"][pair_key])
        j, k = pair if isinstance(pair, (list, tuple)) else (pair[0], pair[1])
        summary_rows.append({
            "Pair": f"A{j+1} vs A{k+1}",
            "N significant (FDR<0.05)": n_sig,
            "Mean |delta-beta|": f"{np.abs(delta).mean():.4f}",
            "Max |delta-beta|": f"{np.abs(delta).max():.4f}",
        })
    summary_df = pd.DataFrame(summary_rows)
    html += report.df_to_html(summary_df, caption="Pairwise Wald contrast summary")

    # Volcano grid
    try:
        fig_vg = pc.pl.contrast_volcano_grid(adata, show=False)
        # Make gene labels more readable
        fig_vg.update_traces(textfont_size=18)
        fig_vg.update_layout(
            font=dict(size=13),
            width=1800,
            height=1200,
        )
        # Update all subplot axes
        for i in range(1, 20):
            fig_vg.update_xaxes(title_text="Δβ", row=None, col=None)
            fig_vg.update_yaxes(title_text="-log10(FDR)", row=None, col=None)
        html += safe_plotly_html(report, fig_vg, "Pairwise Wald contrast volcano grid")
    except Exception as e:
        html += error_html(f"Contrast volcano grid failed: {e}")

    # FIX #8: Top contrasts table grouped by Pair with delta_beta sign interpretation
    top_rows = []
    for pair in pairs:
        pair_key = str(tuple(pair) if isinstance(pair, list) else pair)
        pvals = np.asarray(contrast_result["pvalues_fdr"][pair_key])
        delta = np.asarray(contrast_result["delta_beta"][pair_key])
        j, k = pair if isinstance(pair, (list, tuple)) else (pair[0], pair[1])
        for feat_idx in range(len(feature_names)):
            if pvals[feat_idx] < 0.05:
                sign_interp = (f"higher in A{j+1}" if delta[feat_idx] > 0
                               else f"higher in A{k+1}")
                top_rows.append({
                    "Feature": feature_names[feat_idx],
                    "Feature_symbol": ensembl_to_symbol(adata, [feature_names[feat_idx]])[0],
                    "Pair": f"A{j+1} vs A{k+1}",
                    "delta_beta": delta[feat_idx],
                    "Direction": sign_interp,
                    "FDR q": pvals[feat_idx],
                })
    if top_rows:
        top_df = pd.DataFrame(top_rows)
        # B6: Group by pair, rank by |Δβ| within each pair
        top_df["abs_delta"] = top_df["delta_beta"].abs()
        for pair_name, pair_group in top_df.groupby("Pair"):
            top_pair = pair_group.nlargest(10, "abs_delta")
            display_cols = ["Feature_symbol", "Pair", "delta_beta", "Direction", "FDR q"]
            html += report.df_to_html(
                top_pair[display_cols],
                caption=f"Top 10 contrasts: {pair_name} (ranked by |Δβ|)"
            )
        html += report.text(
            "delta_beta > 0: gene has stronger association with the first archetype in the pair. "
            "delta_beta < 0: stronger association with the second archetype."
        )
    else:
        html += report.text("No significant contrasts at FDR < 0.05.")

    # C8: Shared contrast genes across multiple archetype pairs
    if top_rows:
        all_contrast_df = pd.DataFrame(top_rows)
        if "Feature_symbol" in all_contrast_df.columns:
            gene_pair_counts = all_contrast_df.groupby("Feature_symbol")["Pair"].nunique()
            shared_genes = gene_pair_counts[gene_pair_counts > 1].sort_values(ascending=False)
            if len(shared_genes) > 0:
                html += "<h4>Genes in multiple pairwise contrasts</h4>"
                shared_detail = []
                for gene, n_pairs in shared_genes.head(20).items():
                    gene_rows = all_contrast_df[all_contrast_df["Feature_symbol"] == gene]
                    pairs_detail = "; ".join(
                        f"{r['Pair']}: {'higher in first' if r['delta_beta'] > 0 else 'higher in second'}"
                        for _, r in gene_rows.iterrows()
                    )
                    shared_detail.append({
                        "Gene": gene,
                        "N pairs": n_pairs,
                        "Direction by pair": pairs_detail,
                    })
                html += report.df_to_html(
                    pd.DataFrame(shared_detail),
                    caption="Genes significant in multiple archetype pair contrasts"
                )

    # B7: Cross-method feature overlap (interaction terms vs Wald contrasts)
    try:
        int_coefs_b7 = gene_reg.get("interaction_coefficients")
        int_fdr_b7 = gene_reg.get("interaction_pvalues_fdr")
        if int_coefs_b7 is not None and int_fdr_b7 is not None:
            int_fdr_b7 = np.asarray(int_fdr_b7)
            reg_feat_names_b7 = list(gene_reg["feature_names"])
            sig_int_mask_b7 = (int_fdr_b7 < 0.05).any(axis=1)
            sig_int_features_b7 = set(
                reg_feat_names_b7[i] for i in range(len(reg_feat_names_b7)) if sig_int_mask_b7[i]
            )
            sig_contrast_features_b7 = set()
            for pair in pairs:
                pair_key = str(tuple(pair) if isinstance(pair, list) else pair)
                pvals = np.asarray(contrast_result["pvalues_fdr"][pair_key])
                for i, p in enumerate(pvals):
                    if p < 0.05:
                        sig_contrast_features_b7.add(feature_names[i])
            overlap_b7 = sig_int_features_b7 & sig_contrast_features_b7
            html += metric_grid([
                metric_card(len(sig_int_features_b7), "Sig interaction features"),
                metric_card(len(sig_contrast_features_b7), "Sig contrast features"),
                metric_card(len(overlap_b7), "Interaction ∩ Wald"),
            ])
            html += report.text("See cross-method concordance in step 14 for full overlap analysis.")
    except Exception as e:
        html += error_html(f"Overlap analysis failed: {e}")

    # C1: Cross-reference: top-50 interaction genes vs top-50 Wald genes
    try:
        reg_result_c1 = adata.uns.get("peach_simplex_regression_genes") or adata.uns.get("peach_simplex_regression")
        if reg_result_c1 and "interaction_coefficients" in reg_result_c1:
            int_coefs_c1 = np.asarray(reg_result_c1["interaction_coefficients"])
            feat_names_c1 = list(reg_result_c1.get("feature_names", adata.var_names))
            # Top 50 by max |interaction coefficient|
            max_int_c1 = np.max(np.abs(int_coefs_c1), axis=1) if int_coefs_c1.ndim == 2 else np.abs(int_coefs_c1)
            top50_int_c1 = set(np.array(feat_names_c1)[np.argsort(max_int_c1)[-50:]])
            # Top 50 by Wald |Δβ| (uses abs_delta computed above)
            top50_wald_c1 = set()
            if top_rows:
                top50_wald_c1 = set(top_df.nlargest(50, "abs_delta")["Feature"].values) if "Feature" in top_df.columns else set()

            shared_c1 = top50_int_c1 & top50_wald_c1
            int_only_c1 = top50_int_c1 - top50_wald_c1
            wald_only_c1 = top50_wald_c1 - top50_int_c1

            html += metric_grid([
                metric_card(len(shared_c1), "Shared (interaction ∩ Wald top-50)"),
                metric_card(len(int_only_c1), "Interaction-only"),
                metric_card(len(wald_only_c1), "Wald-only"),
            ])
            if shared_c1:
                shared_symbols_c1 = ensembl_to_symbol(adata, sorted(list(shared_c1))[:20])
                html += report.text(f"Shared features: {', '.join(shared_symbols_c1)}"
                                    + ("..." if len(shared_c1) > 20 else ""))
        elif gene_reg.get("interaction_coefficients") is not None:
            # Fall back to gene_reg directly
            int_coefs_c1 = np.asarray(gene_reg["interaction_coefficients"])
            feat_names_c1 = list(gene_reg["feature_names"])
            max_int_c1 = np.max(np.abs(int_coefs_c1), axis=1) if int_coefs_c1.ndim == 2 else np.abs(int_coefs_c1)
            top50_int_c1 = set(np.array(feat_names_c1)[np.argsort(max_int_c1)[-50:]])
            top50_wald_c1 = set()
            if top_rows and "abs_delta" in top_df.columns:
                top50_wald_c1 = set(top_df.nlargest(50, "abs_delta")["Feature"].values)

            shared_c1 = top50_int_c1 & top50_wald_c1
            int_only_c1 = top50_int_c1 - top50_wald_c1
            wald_only_c1 = top50_wald_c1 - top50_int_c1

            html += metric_grid([
                metric_card(len(shared_c1), "Shared (interaction ∩ Wald top-50)"),
                metric_card(len(int_only_c1), "Interaction-only"),
                metric_card(len(wald_only_c1), "Wald-only"),
            ])
            if shared_c1:
                shared_symbols_c1 = ensembl_to_symbol(adata, sorted(list(shared_c1))[:20])
                html += report.text(f"Shared features: {', '.join(shared_symbols_c1)}"
                                    + ("..." if len(shared_c1) > 20 else ""))
    except Exception as e:
        html += error_html(f"Wald vs interaction comparison failed: {e}")

    # --- Pathway-level contrasts ---
    pw_reg_data = adata.uns.get("peach_simplex_regression_pathways")
    if pw_reg_data is not None:
        html += "<h4>Pathway-level Wald contrasts</h4>"
        try:
            # Temporarily swap pathway regression into the generic key so
            # archetype_contrasts picks it up
            _gene_stash_wald = adata.uns.pop("peach_simplex_regression", None)
            adata.uns["peach_simplex_regression"] = pw_reg_data
            pw_contrast = pc.tl.archetype_contrasts(adata)

            pw_pairs = pw_contrast.get("pairs", [])
            pw_feat_names = list(pw_contrast.get("feature_names", []))
            for pw_pair in pw_pairs:
                pw_pair_key = str(tuple(pw_pair) if isinstance(pw_pair, list) else pw_pair)
                pw_pvals = np.asarray(pw_contrast["pvalues_fdr"][pw_pair_key])
                pw_delta = np.asarray(pw_contrast["delta_beta"][pw_pair_key])
                j, k = pw_pair if isinstance(pw_pair, (list, tuple)) else (pw_pair[0], pw_pair[1])
                pw_top_rows = []
                for feat_idx in range(len(pw_feat_names)):
                    if pw_pvals[feat_idx] < 0.05:
                        sign_interp = (f"higher in A{j+1}" if pw_delta[feat_idx] > 0
                                       else f"higher in A{k+1}")
                        pw_top_rows.append({
                            "Pathway": pw_feat_names[feat_idx],
                            "Pair": f"A{j+1} vs A{k+1}",
                            "delta_beta": pw_delta[feat_idx],
                            "Direction": sign_interp,
                            "FDR q": pw_pvals[feat_idx],
                        })
                if pw_top_rows:
                    pw_top_df = pd.DataFrame(pw_top_rows)
                    pw_top_df["abs_delta"] = pw_top_df["delta_beta"].abs()
                    top10_pw = pw_top_df.nlargest(10, "abs_delta")
                    display_cols = ["Pathway", "Pair", "delta_beta", "Direction", "FDR q"]
                    html += report.df_to_html(
                        top10_pw[display_cols],
                        caption=f"Top 10 pathway contrasts: A{j+1} vs A{k+1} (ranked by |Δβ|)"
                    )
                else:
                    html += report.text(f"No significant pathway contrasts for A{j+1} vs A{k+1}.")
        except Exception as e:
            html += error_html(f"Pathway-level contrasts failed: {e}")
        finally:
            # Restore gene regression to generic key
            adata.uns.pop("peach_simplex_regression", None)
            if _gene_stash_wald is not None:
                adata.uns["peach_simplex_regression"] = _gene_stash_wald
    else:
        html += report.text("Pathway-level contrasts skipped: no pathway regression results found.")

    report.add_section("Wald Contrasts", html, step_num=5)
    return contrast_result


def step6_within_fit_comparisons(adata, report):
    """Step 6: Within-fit MMD and feature similarity."""
    import peach as pc

    html = ""

    # MMD
    log.info("Computing within-fit MMD...")
    try:
        mmd_result = pc.tl.archetype_mmd(adata, n_permutations=100)
        mmd_matrix = np.asarray(mmd_result["mmd_matrix"])
        K = mmd_matrix.shape[0]
        html += report.text(f"MMD matrix: {K}x{K} archetypes.")

        # Display p-value significance
        pval_matrix = np.asarray(mmd_result.get("pvalue_matrix", np.ones_like(mmd_matrix)))
        n_sig_mmd = int((pval_matrix[np.triu_indices(K, k=1)] < 0.05).sum())
        total_pairs = K * (K - 1) // 2
        html += metric_card(f"{n_sig_mmd}/{total_pairs}", "Significant MMD pairs (p<0.05)")

        try:
            fig_mmd = pc.pl.mmd_heatmap(adata, show=False)
            html += safe_plotly_html(report, fig_mmd, "Within-fit MMD heatmap")
        except Exception as e:
            html += error_html(f"MMD heatmap plot failed: {e}")

        html += report.text(
            "Note: A Sankey diagram of archetype-to-archetype flow would be informative here. "
            "This is available via pc.pl.soft_assignment_flow() in notebook mode."
        )

        # MMD statistical basis note
        html += report.text(
            "MMD (Maximum Mean Discrepancy): measures distributional distance between "
            "cell populations assigned to each archetype in PCA space, using a Gaussian "
            "kernel. Low MMD = similar distributions; high MMD = distinct cell populations. "
            "p-values from permutation test (H0: populations are exchangeable)."
        )
        html += report.text(
            "<strong>Statistical basis:</strong> MMD is computed as the squared difference "
            "between kernel mean embeddings of the two cell populations in a reproducing kernel "
            "Hilbert space (RKHS). The Gaussian kernel k(x,y) = exp(-||x-y||^2 / (2*sigma^2)) "
            "is used, with sigma set to the median pairwise distance (median heuristic). "
            "The unbiased two-sample estimator is used. "
            "The permutation p-value is computed by randomly shuffling archetype labels "
            "across cells and recomputing MMD to form a null distribution. "
            "<strong>Spearman rho</strong> in the feature similarity section is computed by "
            "ranking the FDR-significant (q&lt;0.05) vertex beta coefficients from simplex "
            "regression for each pair of archetypes, then computing Spearman rank correlation "
            "across those ranked coefficient vectors. High rho means two archetypes share "
            "similar gene-weight profiles; low rho means they are molecularly distinct."
        )
    except Exception as e:
        html += error_html(f"MMD computation failed: {e}")

    # Feature similarity
    log.info("Computing within-fit feature similarity...")
    try:
        sim_result = pc.tl.archetype_feature_similarity(adata)
        spearman = np.asarray(sim_result["spearman_matrix"])
        n_shared = sim_result.get("n_shared_features", "?")
        n_sig_feat = sim_result.get("n_significant_features", "?")
        html += metric_grid([
            metric_card(n_shared, "Shared features"),
            metric_card(n_sig_feat, "Significant features used"),
        ])

        try:
            fig_sim = pc.pl.feature_similarity_heatmap(adata, show=False)
            html += safe_plotly_html(report, fig_sim, "Feature similarity (Spearman) heatmap")
        except Exception as e:
            html += error_html(f"Feature similarity heatmap plot failed: {e}")
        html += report.text(
            "Feature similarity: Spearman rho computed on FDR-significant (q<0.05) "
            "vertex beta coefficients from simplex regression. Higher rho = archetypes "
            "share similar gene-archetype associations."
        )
    except Exception as e:
        html += error_html(f"Feature similarity failed: {e}")

    # Summary table
    try:
        arch_labels = [f"A{i+1}" for i in range(K)]
        rows = []
        for i in range(K):
            for j in range(i + 1, K):
                row = {"Pair": f"A{i+1} vs A{j+1}", "MMD": f"{mmd_matrix[i, j]:.4f}"}
                try:
                    row["Spearman r"] = f"{spearman[i, j]:.4f}"
                except Exception:
                    pass
                rows.append(row)
        if rows:
            html += report.df_to_html(pd.DataFrame(rows), caption="Pairwise comparison summary")
    except Exception:
        pass

    report.add_section("Within-Fit Comparisons (MMD & Feature Similarity)", html, step_num=6)


def step7_driver_regression(adata, report):
    """Step 7: Driver regression -- FIX #9: test on BOTH pathways and genes."""
    import peach as pc

    html = ""

    # -- Pathway driver regression (if available) --
    has_pathways = "pathway_scores" in adata.obsm
    if has_pathways:
        log.info("Running driver regression on pathway scores (degree=1)...")
        try:
            pw_driver = pc.tl.archetype_driver_regression(
                adata, feature_matrix="pathway_scores", max_degree=1,
                n_bootstrap=500, robust_se=True,
            )
            r2_pw = np.asarray(pw_driver["r_squared"])
            pw_feat_names = list(pw_driver.get("feature_names", []))
            html += report.text("<strong>Pathway driver regression</strong>")
            html += metric_grid([
                metric_card(f"{r2_pw.mean():.4f}", "Mean R-sq (ILR, pathways)"),
                metric_card(len(pw_feat_names), "Pathway features"),
            ])

            r2_rows = [{"ILR component": i+1, "R-squared": f"{r2_pw[i]:.4f}"}
                       for i in range(len(r2_pw))]
            html += report.df_to_html(pd.DataFrame(r2_rows),
                                      caption="Pathway driver R-squared per ILR component")

            # Coefficient heatmap
            try:
                main_coefs = np.asarray(pw_driver["main_coefficients"])
                n_show = min(25, len(pw_feat_names))
                mean_abs = np.abs(main_coefs).mean(axis=0)
                top_idx = np.argsort(mean_abs)[-n_show:][::-1]
                fig, ax = plt.subplots(figsize=(max(6, main_coefs.shape[0] * 1.2),
                                                max(5, n_show * 0.35)))
                im = ax.imshow(main_coefs[:, top_idx].T, aspect="auto", cmap="RdBu_r")
                ax.set_xticks(range(main_coefs.shape[0]))
                ax.set_xticklabels([f"A{k+1}" for k in range(main_coefs.shape[0])])
                ax.set_yticks(range(n_show))
                ax.set_yticklabels([pw_feat_names[i] for i in top_idx], fontsize=8)
                plt.colorbar(im, ax=ax, label="Coefficient", shrink=0.6)
                ax.set_title("Pathway driver coefficients")
                fig.tight_layout()
                html += report.fig_to_img(fig, caption="Pathway driver coefficient heatmap")
                plt.close("all")
            except Exception as e:
                html += error_html(f"Pathway driver heatmap failed: {e}")
                plt.close("all")

        except Exception as e:
            html += error_html(f"Pathway driver regression failed: {e}")

    # -- Gene driver regression (subset to top 200 by variance) -- FIX #9
    log.info("Running driver regression on top-200 genes by variance...")
    try:
        X = _dense_X(adata)
        gene_var = np.var(X, axis=0)
        top200_idx = np.argsort(gene_var)[-200:]
        # Create a custom feature matrix
        adata.obsm["_driver_genes_top200"] = X[:, top200_idx]
        top200_names = list(adata.var_names[top200_idx])
        top200_symbols = ensembl_to_symbol(adata, top200_names)

        gene_driver = pc.tl.archetype_driver_regression(
            adata, feature_matrix="_driver_genes_top200", max_degree=1,
            n_bootstrap=500, robust_se=True,
        )
        r2_gene = np.asarray(gene_driver["r_squared"])
        html += report.text("<strong>Gene driver regression (top 200 by variance)</strong>")
        html += metric_grid([
            metric_card(f"{r2_gene.mean():.4f}", "Mean R-sq (ILR, genes)"),
            metric_card(200, "Gene features"),
        ])

        r2_rows = [{"ILR component": i+1, "R-squared": f"{r2_gene[i]:.4f}"}
                   for i in range(len(r2_gene))]
        html += report.df_to_html(pd.DataFrame(r2_rows),
                                  caption="Gene driver R-squared per ILR component")

        # B8: Coefficient heatmap grouped by dominant archetype
        try:
            main_coefs = np.asarray(gene_driver["main_coefficients"])  # [K_ilr, n_features]
            n_show = min(30, main_coefs.shape[1])
            mean_abs = np.abs(main_coefs).mean(axis=0)
            top_idx = np.argsort(mean_abs)[-n_show:]
            # Sort by dominant archetype for grouped display
            # main_coefs shape: [n_ilr, n_features]; use max |coef| across ILR components
            # Approximate dominant archetype: use the ILR component with max |coef|
            dom_arch_driver = np.argmax(np.abs(main_coefs[:, top_idx]), axis=0) + 1  # per-feature dominant ILR
            sort_order_driver = np.argsort(dom_arch_driver)
            top_idx_sorted_driver = top_idx[sort_order_driver]
            dom_arch_sorted_driver = dom_arch_driver[sort_order_driver]
            fig, ax = plt.subplots(figsize=(max(6, main_coefs.shape[0] * 1.2),
                                            max(5, n_show * 0.35)))
            im = ax.imshow(main_coefs[:, top_idx_sorted_driver].T, aspect="auto", cmap="RdBu_r")
            ax.set_xticks(range(main_coefs.shape[0]))
            ax.set_xticklabels([f"ILR{k+1}" for k in range(main_coefs.shape[0])])
            ax.set_yticks(range(n_show))
            ylabels_driver = [
                f"{top200_symbols[top_idx_sorted_driver[r]]} [ILR{dom_arch_sorted_driver[r]}]"
                for r in range(len(top_idx_sorted_driver))
            ]
            ax.set_yticklabels(ylabels_driver, fontsize=7)
            plt.colorbar(im, ax=ax, label="Coefficient", shrink=0.6)
            ax.set_title("Gene driver coefficients (grouped by dominant ILR component)")
            fig.tight_layout()
            html += report.fig_to_img(fig, caption="Gene driver coefficient heatmap (grouped by dominant ILR)")
            plt.close("all")
        except Exception as e:
            html += error_html(f"Gene driver heatmap failed: {e}")
            plt.close("all")

        # B9: Top 5 drivers per ILR component
        try:
            main_coefs_b9 = np.asarray(gene_driver["main_coefficients"])
            feat_names_b9 = top200_symbols  # already resolved above
            for k in range(main_coefs_b9.shape[0]):
                top5_idx_b9 = np.argsort(np.abs(main_coefs_b9[k]))[-5:][::-1]
                top5_rows = [{"Gene": feat_names_b9[i], "β": f"{main_coefs_b9[k, i]:.4f}"}
                             for i in top5_idx_b9]
                html += report.df_to_html(pd.DataFrame(top5_rows),
                                          caption=f"Top 5 drivers for ILR{k+1}")
        except Exception as e:
            html += error_html(f"Top drivers per ILR failed: {e}")

        # Clean up temp obsm key
        del adata.obsm["_driver_genes_top200"]

        # Concordance: pathway vs gene driver
        if has_pathways:
            try:
                html += report.text(
                    f"Pathway driver mean R-sq: {r2_pw.mean():.4f} vs "
                    f"Gene driver mean R-sq: {r2_gene.mean():.4f}"
                )
            except Exception:
                pass

        # C2: Gene driver vs simplex regression overlap (top-50 by max |coef|)
        try:
            reg_result_c2 = (adata.uns.get("peach_simplex_regression_genes")
                             or adata.uns.get("peach_simplex_regression"))
            if reg_result_c2 and "feature_names" in reg_result_c2:
                sreg_names_c2 = list(reg_result_c2["feature_names"])
                sreg_r2_c2 = np.asarray(reg_result_c2["r_squared_degree1"])
                top50_sreg_c2 = set(np.array(sreg_names_c2)[np.argsort(sreg_r2_c2)[-50:]])

                main_coefs_c2 = np.asarray(gene_driver["main_coefficients"])
                driver_max_c2 = np.max(np.abs(main_coefs_c2), axis=0)
                # Map back to original var_names for comparison
                top50_driver_c2 = set(np.array(top200_names)[np.argsort(driver_max_c2)[-50:]])

                shared_c2 = top50_sreg_c2 & top50_driver_c2
                sreg_only_c2 = top50_sreg_c2 - top50_driver_c2
                driv_only_c2 = top50_driver_c2 - top50_sreg_c2

                html += metric_grid([
                    metric_card(len(shared_c2), "Shared top-50 (driver ∩ simplex)"),
                    metric_card(len(sreg_only_c2), "Simplex-only"),
                    metric_card(len(driv_only_c2), "Driver-only"),
                ])
                if shared_c2:
                    shared_syms_c2 = ensembl_to_symbol(adata, sorted(list(shared_c2))[:20])
                    html += report.text(f"Shared genes: {', '.join(shared_syms_c2)}"
                                        + ("..." if len(shared_c2) > 20 else ""))
        except Exception as e:
            html += error_html(f"Driver vs simplex regression comparison failed: {e}")

    except Exception as e:
        html += error_html(f"Gene driver regression failed: {e}")

    report.add_section("Driver Regression", html, step_num=7)


def step8_mixture_models(adata, report):
    """Step 8: Simplex density decomposition (Dirichlet). FIX #16: add archetypal space by component."""
    import peach as pc

    html = ""

    log.info("Running simplex decomposition (Dirichlet)...")
    try:
        K = adata.obsm["cell_archetype_weights"].shape[1]
        decomp_result = pc.tl.feature_simplex_decomposition(
            adata, model_type="dirichlet",
            n_components_range=(K, 3 * K),
            model_selection="bic_elbow",
            n_initializations=20,
            stability_threshold=0.7,
        )

        n_opt = decomp_result.get("n_components_optimal", "?")
        n_stable = decomp_result.get("n_components_stable", "?")
        bic = decomp_result.get("bic_values", [])
        stab_scores = decomp_result.get("component_stability_scores", [])

        # Boundary warning: if elbow lands at max, the range may be too narrow
        max_c = 3 * K
        if isinstance(n_opt, int) and n_opt >= max_c:
            html += error_html(
                f"WARNING: Optimal components ({n_opt}) at search boundary ({max_c}). "
                "Consider widening n_components_range."
            )

        html += metric_grid([
            metric_card(n_opt, "Optimal components (BIC elbow)"),
            metric_card(n_stable, "Stable components"),
            metric_card(decomp_result.get("model_type", "?"), "Model type"),
        ])

        try:
            fig_bic = pc.pl.gmm_bic_curve(adata, show=False)
            html += safe_plotly_html(report, fig_bic, "BIC curve vs number of components")
        except Exception as e:
            html += error_html(f"BIC curve plot failed: {e}")

        try:
            fig_stab = pc.pl.component_stability(adata, show=False)
            html += safe_plotly_html(report, fig_stab, "Component stability scores")
        except Exception as e:
            html += error_html(f"Component stability plot failed: {e}")
        html += report.text(
            "Stability: fraction of n_initializations (20 random starts) where this "
            "component is recovered via Hungarian matching of component centroids. "
            "Higher = more robust to initialization. Threshold: 0.7."
        )

        # PCA scatter colored by component
        try:
            fig_scat = pc.pl.component_scatter(adata, show=False)
            html += safe_plotly_html(report, fig_scat, "PCA scatter colored by component")
        except Exception as e:
            html += error_html(f"Component scatter failed: {e}")

        # FIX #16: Archetypal space colored by component
        try:
            gmm_data = adata.uns.get("peach_gmm")
            if gmm_data is not None:
                assignments = np.asarray(gmm_data["component_assignments"])
                adata.obs["_gmm_comp_temp"] = pd.Categorical(
                    [f"C{int(a)}" if a >= 0 else "unassigned" for a in assignments]
                )
                fig_arch_comp = pc.pl.archetypal_space(adata, color_by="_gmm_comp_temp",
                                                       title="Archetypal space (component)")
                html += safe_plotly_html(report, fig_arch_comp,
                                         "Archetypal space colored by mixture component")
                del adata.obs["_gmm_comp_temp"]
        except Exception as e:
            html += error_html(f"Archetypal space by component failed: {e}")

        # Component-archetype summary
        try:
            fig_summary = pc.pl.component_archetype_summary(adata, show=False)
            fig_summary.update_layout(showlegend=True)
            html += safe_plotly_html(report, fig_summary, "Component-archetype summary (4-panel)")
        except Exception as e:
            html += error_html(f"Component-archetype summary failed: {e}")

        # Component weight means table
        weight_means = decomp_result.get("component_simplex_means")
        if weight_means is not None:
            weight_means = np.asarray(weight_means)
            K = weight_means.shape[1]
            rows = []
            for c in range(weight_means.shape[0]):
                row = {"Component": c}
                for k in range(K):
                    row[f"A{k+1} weight"] = f"{weight_means[c, k]:.3f}"
                if isinstance(stab_scores, (list, np.ndarray)) and c < len(stab_scores):
                    row["Stability"] = f"{stab_scores[c]:.3f}"
                rows.append(row)
            html += report.df_to_html(pd.DataFrame(rows), caption="Component simplex means + stability")

    except Exception as e:
        html += error_html(f"Simplex decomposition failed: {e}")

    report.add_section("Mixture Model Decomposition", html, step_num=8)


def step9_component_characterization(adata, report):
    """Step 9: Component regression, enrichment. FIX #17: component-archetype proximity table."""
    import peach as pc

    html = ""

    # Component regression
    log.info("Running component regression...")
    try:
        comp_reg = pc.tl.component_regression(adata)
        n_comps = comp_reg.get("n_components", 0)
        comp_regs = comp_reg.get("component_regs", {})

        html += metric_grid([
            metric_card(n_comps, "Mixture components"),
            metric_card(len(comp_regs), "Components with regression"),
        ])

        comp_rows = []
        for c, reg in comp_regs.items():
            r2 = np.asarray(reg["r_squared_degree1"])
            f_fdr = np.asarray(reg.get("f_pvalue_fdr", np.ones(len(r2))))
            n_sig = int((f_fdr < 0.05).sum())
            top_feat = reg["feature_names"][np.argmax(r2)] if len(reg["feature_names"]) > 0 else "N/A"
            top_symbol = ensembl_to_symbol(adata, [top_feat])[0] if top_feat != "N/A" else "N/A"
            comp_rows.append({
                "Component": c,
                "N features": len(r2),
                "Mean R-squared": f"{r2.mean():.4f}",
                "N significant": n_sig,
                "Top feature": top_symbol,
                "Top R-squared": f"{r2.max():.4f}",
            })
        if comp_rows:
            html += report.df_to_html(pd.DataFrame(comp_rows), caption="Per-component regression summary")

        try:
            fig_heat = pc.pl.component_heatmap(adata, show=False)
            html += safe_plotly_html(report, fig_heat, "Component feature profiles heatmap")
        except Exception as e:
            html += error_html(f"Component heatmap failed: {e}")

        # Per-component top genes
        if comp_regs:
            for c_id, creg in comp_regs.items():
                if creg and "feature_names" in creg and "r_squared_degree1" in creg:
                    c_names = list(creg["feature_names"])
                    c_r2 = np.asarray(creg["r_squared_degree1"])
                    top10_idx = np.argsort(c_r2)[-10:][::-1]
                    top10 = [{"Gene": ensembl_to_symbol(adata, [c_names[i]])[0], "R2": f"{c_r2[i]:.4f}"} for i in top10_idx]
                    html += report.df_to_html(pd.DataFrame(top10),
                        caption=f"Component {c_id}: top 10 genes by R²")

    except Exception as e:
        html += error_html(f"Component regression failed: {e}")

    # FIX #17: Component-archetype proximity table
    gmm_data = adata.uns.get("peach_gmm")
    if gmm_data is not None:
        try:
            weight_means = gmm_data.get("component_simplex_means")
            if weight_means is not None:
                weight_means = np.asarray(weight_means)
                K = weight_means.shape[1]
                prox_rows = []
                for c in range(weight_means.shape[0]):
                    nearest_arch = int(np.argmax(weight_means[c])) + 1
                    max_weight = weight_means[c].max()
                    prox_rows.append({
                        "Component": c,
                        "Nearest archetype": f"A{nearest_arch}",
                        "Max weight": f"{max_weight:.3f}",
                        "Weight vector": ", ".join([f"A{k+1}={weight_means[c, k]:.3f}" for k in range(K)]),
                    })
                html += report.df_to_html(pd.DataFrame(prox_rows),
                                          caption="Component-archetype proximity (nearest archetype by mean weight)")
        except Exception as e:
            html += error_html(f"Component-archetype proximity table failed: {e}")

    # Component conditional associations
    if gmm_data is not None:
        try:
            assignments = np.asarray(gmm_data["component_assignments"])
            adata.obs["gmm_component"] = pd.Categorical(
                [f"comp_{int(a)}" if a >= 0 else "unassigned" for a in assignments]
            )

            for col in ["cell_type_short", "Study"]:
                if col not in adata.obs.columns:
                    continue
                log.info(f"Component conditional associations: {col}...")
                try:
                    comp_cond = pc.tl.conditional_associations(
                        adata, obs_column=col, obs_key="gmm_component", verbose=False
                    )
                    sig_col = "significant" if "significant" in comp_cond.columns else None
                    if sig_col:
                        sig_comp = comp_cond[comp_cond[sig_col] == True]
                    else:
                        sig_comp = comp_cond[comp_cond["fdr_pvalue"] < 0.05]
                    html += report.text(
                        f"Component x {col}: {len(sig_comp)} significant / {len(comp_cond)} tests"
                    )
                    display_cols = ["archetype", "condition", "odds_ratio", "fdr_pvalue", "significant"]
                    display_cols = [c for c in display_cols if c in comp_cond.columns]
                    html += report.df_to_html(comp_cond[display_cols],
                                              caption=f"Component enrichment: {col}")
                except Exception as e:
                    html += error_html(f"Component conditional ({col}) failed: {e}")

            if "gmm_component" in adata.obs.columns:
                del adata.obs["gmm_component"]

        except Exception as e:
            html += error_html(f"Component enrichment analysis failed: {e}")

    # Component neighborhood graph
    try:
        fig_net = pc.pl.component_neighborhood_graph(adata, show=False)
        html += safe_plotly_html(report, fig_net, "Component neighborhood graph")
    except Exception as e:
        html += error_html(f"Component neighborhood graph failed: {e}")

    # C9: Component-archetype similarity comparison (ARI + majority vote + MMD)
    try:
        from sklearn.metrics import adjusted_rand_score
        from peach._core.utils.flow_matching import compute_mmd

        gmm = adata.uns.get("peach_gmm")
        if gmm is not None and "archetypes" in adata.obs.columns:
            comp_assignments = np.asarray(gmm["component_assignments"])
            arch_assignments = adata.obs["archetypes"].values

            ari = adjusted_rand_score(arch_assignments, comp_assignments)
            html += "<h3>Component-Archetype Similarity</h3>"
            html += metric_card(f"{ari:.4f}", "Adjusted Rand Index (components vs archetypes)")

            n_comp = gmm.get("n_components_stable", gmm.get("n_components_optimal", 0))
            pca = adata.obsm["X_pca"]

            similarity_rows = []
            for c in range(n_comp):
                c_mask = comp_assignments == c
                if c_mask.sum() < 5:
                    continue
                c_pca = pca[c_mask]
                c_archs = arch_assignments[c_mask]
                dom_arch = pd.Series(c_archs).mode().iloc[0] if len(c_archs) > 0 else "N/A"
                dom_frac = float((c_archs == dom_arch).mean()) if len(c_archs) > 0 else 0

                mmd_vals = {}
                for arch_label in sorted(set(arch_assignments)):
                    a_mask = arch_assignments == arch_label
                    if a_mask.sum() >= 5:
                        mmd_val = compute_mmd(c_pca, pca[a_mask])
                        mmd_vals[str(arch_label)] = float(mmd_val)

                nearest_arch = min(mmd_vals, key=mmd_vals.get) if mmd_vals else "N/A"
                nearest_mmd = mmd_vals.get(nearest_arch, float('nan'))

                similarity_rows.append({
                    "Component": c,
                    "N cells": int(c_mask.sum()),
                    "Dominant archetype": str(dom_arch),
                    "Dominant fraction": f"{dom_frac:.0%}",
                    "Nearest by MMD": str(nearest_arch),
                    "Nearest MMD": f"{nearest_mmd:.4f}",
                    "Agreement": "yes" if str(dom_arch) == str(nearest_arch) else "NO",
                })

            if similarity_rows:
                html += report.df_to_html(
                    pd.DataFrame(similarity_rows),
                    caption="Component-archetype similarity (majority vote vs MMD nearest)"
                )
                html += report.text(
                    "Three metrics compared: ARI (global partition agreement), "
                    "majority-vote (which archetype dominates each component), and MMD "
                    "(distributional distance in PCA space). 'yes' = vote and MMD agree."
                )
    except Exception as e:
        html += error_html(f"Component-archetype similarity failed: {e}")

    report.add_section("Component Characterization", html, step_num=9)


# ============================================================================
# Per-lineage helpers (shared by steps 10-17)
# ============================================================================

def _subset_adata(adata, mask, label: str):
    """Subset adata by boolean mask, preserving obsm/varm references."""
    sub = adata[mask].copy()
    log.info(f"  Subset '{label}': {sub.shape[0]} cells")
    return sub


def _get_ct_mask(adata, ct_short_list):
    """Get boolean mask for a list of short cell type names."""
    long_names = [long_ct(s) for s in ct_short_list]
    return adata.obs["cell_type"].isin(long_names).values


def run_subset_model(adata_sub, K: int, hidden_dims: list, label: str) -> dict | None:
    """Train model on a subset adata. Returns training results dict or None."""
    import peach as pc

    if adata_sub.shape[0] < MIN_CELLS_MODEL:
        log.warning(f"  {label}: only {adata_sub.shape[0]} cells, skipping model training.")
        return None

    pc.pp.prepare_training(adata_sub, batch_size=min(128, adata_sub.shape[0] // 4))
    results = pc.tl.train_archetypal(
        adata_sub, n_archetypes=K, n_epochs=80, hidden_dims=hidden_dims,
        kld_weight=0.1,
        early_stopping=True, early_stopping_patience=12,
    )
    pc.tl.archetypal_coordinates(adata_sub, verbose=False)
    pc.tl.assign_archetypes(adata_sub, verbose=False)
    pc.tl.extract_archetype_weights(adata_sub, verbose=False)
    return results


def run_subset_regression(adata_sub, label: str) -> dict | None:
    """Run simplex regression + patterns + conditional + contrasts on subset."""
    import peach as pc

    if "cell_archetype_weights" not in adata_sub.obsm:
        log.warning(f"  {label}: no archetype weights, skipping regression.")
        return None

    gene_reg = pc.tl.gene_simplex_regression(adata_sub, max_degree=1, robust_se=True)

    try:
        pc.tl.classify_feature_patterns(adata_sub)
    except Exception as e:
        log.warning(f"  {label}: pattern classification failed: {e}")

    for col in ["cell_type_short", "Study"]:
        if col in adata_sub.obs.columns and adata_sub.obs[col].nunique() > 1:
            try:
                pc.tl.conditional_associations(adata_sub, obs_column=col, verbose=False)
            except Exception:
                pass

    try:
        pc.tl.archetype_contrasts(adata_sub)
    except Exception as e:
        log.warning(f"  {label}: contrasts failed: {e}")

    return gene_reg


def _reg_summary_row(gene_reg: dict, label: str) -> dict:
    r2 = np.asarray(gene_reg["r_squared_degree1"])
    f_fdr = np.asarray(gene_reg.get("f_pvalue_fdr", np.ones(len(r2))))
    return {
        "Subset": label,
        "N features": len(r2),
        "N sig (FDR<0.05)": int((f_fdr < 0.05).sum()),
        "Mean R2": f"{r2.mean():.4f}",
        "Max R2": f"{r2.max():.4f}",
    }


# ============================================================================
# Steps 10-11: Per-lineage breakout
# ============================================================================

def step10_per_lineage_models(adata, report):
    """Step 10: Per-lineage hyperparameter search, model training, annotation."""
    import peach as pc

    html = ""
    lineage_adatas = {}

    # Define lineage groups for per-subset models
    model_groups = {
        "progenitors": ["HSC", "CMP"],
        "myeloid": ["CMP", "Mono"],
    }

    K_range = [3, 4, 5, 6]
    hidden_opts = [[64, 128], [128, 256]]

    summary_rows = []
    for group_name, ct_list in model_groups.items():
        mask = _get_ct_mask(adata, ct_list)
        sub = _subset_adata(adata, mask, group_name)

        if sub.shape[0] < MIN_CELLS_MODEL:
            html += error_html(f"{group_name}: {sub.shape[0]} cells < {MIN_CELLS_MODEL}, skipped.")
            continue

        # Hyperparameter search
        log.info(f"  {group_name}: hyperparameter search...")
        try:
            pc.pp.prepare_training(sub, batch_size=min(128, sub.shape[0] // 4))
            cv = pc.tl.hyperparameter_search(
                sub, n_archetypes_range=K_range, hidden_dims_options=hidden_opts,
                inflation_factor_range=[1.0, 1.25, 1.5, 1.75], cv_folds=3,
                max_epochs_cv=10, subsample_fraction=0.8,
            )
            ranked = cv.rank_by_metric("archetype_r2")
            ranked = [r for r in ranked if r["metric_value"] > -1e6]
            best = ranked[0] if ranked else {"hyperparameters": {"n_archetypes": 4, "hidden_dims": [128, 256]}, "metric_value": 0.0}
            best_hp = best["hyperparameters"]
            best_K = best_hp["n_archetypes"]
            best_hd = best_hp.get("hidden_dims", [128, 256])
        except Exception as e:
            html += error_html(f"{group_name}: CV search failed ({e}), using K=4 fallback.")
            best_K, best_hd = 4, [128, 256]

        log.info(f"  {group_name}: training K={best_K}, hidden={best_hd}...")
        res = run_subset_model(sub, K=best_K, hidden_dims=best_hd, label=group_name)
        if res is not None:
            lineage_adatas[group_name] = sub
            r2 = res.get("final_archetype_r2", float("nan"))
            summary_rows.append({
                "Group": group_name, "Cell types": ", ".join(ct_list),
                "N cells": sub.shape[0], "Best K": best_K,
                "Hidden": str(best_hd),
                "Final R2": f"{r2:.4f}" if isinstance(r2, float) else str(r2),
            })

    if summary_rows:
        html += report.df_to_html(pd.DataFrame(summary_rows),
                                  caption="Per-lineage model comparison")

    # Per-lineage archetypal space using PEACH interactive plots
    for gname, sub in lineage_adatas.items():
        try:
            fig_arch = pc.pl.archetypal_space(sub, color_by="cell_type_short",
                                               title=f"Archetypal space: {gname}",
                                               )
            html += safe_plotly_html(report, fig_arch, f"Archetypal space: {gname}")
        except Exception as e:
            html += error_html(f"Archetypal space ({gname}) failed: {e}")

    report.add_section("Per-Lineage Models", html, step_num=10)
    return lineage_adatas


def step11_per_lineage_regression(adata, lineage_adatas, report):
    """Step 11: Per-lineage regression, pattern classification, feature stability.
    FIX #10: Replace scatter with table of top 20 most conserved features."""
    import peach as pc
    from scipy.stats import spearmanr

    html = ""

    if not lineage_adatas:
        html += error_html("No per-lineage adatas available, skipping.")
        report.add_section("Per-Lineage Regression", html, step_num=11)
        return

    summary_rows = []
    lineage_regs = {}
    for gname, sub in lineage_adatas.items():
        log.info(f"  {gname}: regression + patterns...")
        try:
            reg = run_subset_regression(sub, label=gname)
            if reg is not None:
                lineage_regs[gname] = reg
                summary_rows.append(_reg_summary_row(reg, gname))
                try:
                    fig_dot = pc.pl.archetype_regression_dotplot(sub, top_n=10, show=False)
                    html += safe_plotly_html(report, fig_dot, f"Regression dotplot: {gname}")
                except Exception as e_dot:
                    html += error_html(f"Regression dotplot ({gname}) failed: {e_dot}")
        except Exception as e:
            html += error_html(f"{gname} regression failed: {e}")

    if summary_rows:
        html += report.df_to_html(pd.DataFrame(summary_rows),
                                  caption="Per-lineage regression summary")

    # FIX #10: Cross-lineage feature stability -- top 20 most conserved features
    group_labels = list(lineage_regs.keys())
    if len(group_labels) >= 2:
        try:
            # Build R2 matrix aligned by shared features
            all_feat_sets = [set(lineage_regs[g]["feature_names"]) for g in group_labels]
            shared_feats = sorted(set.intersection(*all_feat_sets))
            if len(shared_feats) >= 10:
                # Get R2 per feature per group
                r2_per_group = {}
                for g in group_labels:
                    feat_list = list(lineage_regs[g]["feature_names"])
                    r2_arr = np.asarray(lineage_regs[g]["r_squared_degree1"])
                    idx_map = {f: i for i, f in enumerate(feat_list)}
                    r2_per_group[g] = np.array([r2_arr[idx_map[f]] for f in shared_feats])

                # Stability table: top 20 by minimum R2 across all groups
                min_r2 = np.min([r2_per_group[g] for g in group_labels], axis=0)
                top20_idx = np.argsort(min_r2)[-20:][::-1]

                conserved_rows = []
                for rank, idx in enumerate(top20_idx):
                    feat = shared_feats[idx]
                    row = {
                        "Rank": rank + 1,
                        "Feature": ensembl_to_symbol(adata, [feat])[0],
                        "Min R2": f"{min_r2[idx]:.4f}",
                    }
                    for g in group_labels:
                        row[f"R2 ({g})"] = f"{r2_per_group[g][idx]:.4f}"
                    conserved_rows.append(row)
                html += report.df_to_html(
                    pd.DataFrame(conserved_rows),
                    caption="Top 20 most conserved features (highest min-R-squared across lineage groups)"
                )

                # Also show cross-group Spearman
                stab_rows = []
                for i in range(len(group_labels)):
                    for j in range(i + 1, len(group_labels)):
                        g1, g2 = group_labels[i], group_labels[j]
                        rho, p = spearmanr(r2_per_group[g1], r2_per_group[g2])
                        stab_rows.append({
                            "Pair": f"{g1} vs {g2}",
                            "Shared features": len(shared_feats),
                            "Spearman rho": f"{rho:.3f}",
                            "p-value": f"{p:.2e}",
                        })
                if stab_rows:
                    html += report.df_to_html(pd.DataFrame(stab_rows),
                                              caption="Cross-lineage feature stability (R-squared Spearman)")
        except Exception as e:
            html += error_html(f"Feature stability analysis failed: {e}")

    # Lineage-exclusive features: significant in one lineage but not others
    try:
        all_sig_sets = {}
        for gname, greg in lineage_regs.items():
            if greg and "feature_names" in greg and "f_pvalue_fdr" in greg:
                fdr = np.asarray(greg["f_pvalue_fdr"])
                names = list(greg["feature_names"])
                sig = set(np.array(names)[fdr < 0.05])
                all_sig_sets[gname] = sig

        if len(all_sig_sets) >= 2:
            for gname, sig in all_sig_sets.items():
                others = set()
                for other_name, other_sig in all_sig_sets.items():
                    if other_name != gname:
                        others |= other_sig
                exclusive = sig - others
                if exclusive:
                    exc_syms = ensembl_to_symbol(adata, sorted(list(exclusive))[:20])
                    exc_list = exc_syms
                    html += report.text(f"{gname}-exclusive features ({len(exclusive)} total): {', '.join(exc_list)}")
    except Exception as e:
        html += error_html(f"Lineage-exclusive features failed: {e}")

    report.add_section("Per-Lineage Regression", html, step_num=11)


# ============================================================================
# Steps 12-15: Biological transition flows
# ============================================================================

# === FUTURE: R/NR Label-Swap Permutation Protocol ===
# 1. Pool R + NR cells at each timepoint
# 2. For n=1000 permutations: shuffle R/NR labels (preserving per-timepoint counts)
# 3. Fit archetype model, compute stress diversity (Shannon on stress pathways,
#    Bray-Curtis on archetype composition between timepoints)
# 4. Compare observed R-NR diversity difference to null
# 5. FDR correct across timepoints
# Implementation deferred to Part 2 data preparation


def step12_biological_flow(adata, report):
    """Step 12: Between-cell-type flow (soft assignment), biological transitions.
    FIX #18: Add soft assignment calculation method note."""
    import peach as pc

    html = ""
    flow_results = {}

    # FIX #18: Method note
    html += report.text(
        "<strong>Method:</strong> Soft assignment computes how source cells are mapped to target "
        "archetype neighborhoods by the flow model. For each source cell, the transported position "
        "is projected onto the target archetype simplex. The resulting matrix shows the expected "
        "archetype composition of transported cells, revealing differentiation trajectory structure."
    )

    for src_short, tgt_short in FLOW_PAIRS:
        src_long, tgt_long = long_ct(src_short), long_ct(tgt_short)
        src_mask = (adata.obs["cell_type"] == src_long).values
        tgt_mask = (adata.obs["cell_type"] == tgt_long).values
        n_src = int(src_mask.sum())
        n_tgt = int(tgt_mask.sum())

        if n_src < MIN_CELLS_FLOW or n_tgt < MIN_CELLS_FLOW:
            html += error_html(f"{src_short}->{tgt_short}: insufficient cells ({n_src}, {n_tgt}), skipping.")
            continue

        pair_key = f"{src_short}_to_{tgt_short}"
        log.info(f"  Flow: {src_short} -> {tgt_short} ({n_src} -> {n_tgt} cells)...")
        html += f"<h4>Flow: {src_short} -> {tgt_short}</h4>"

        # FIX #4: Source is progenitor (earlier), target is differentiated (later)
        try:
            fr = pc.tl.flow_within(
                adata,
                source={"cell_type": src_long},
                target={"cell_type": tgt_long},
                n_epochs=300,
                hidden_dims=(128, 128, 128),
                batch_size=256,
                return_model=True,
                name=pair_key,
            )
            flow_results[pair_key] = fr

            html += metric_grid([
                metric_card(pair_key, "Flow pair"),
                metric_card(f"{fr['mmd_before']:.4f}", "MMD before"),
                metric_card(f"{fr['mmd_after']:.4f}", "MMD after"),
                metric_card(f"{1 - fr['mmd_after']/max(fr['mmd_before'], 1e-10):.1%}", "MMD reduction"),
            ])
        except Exception as e:
            html += error_html(f"Flow {pair_key} failed: {e}")

    # Soft assignment heatmaps
    for pair_key, fr in flow_results.items():
        html += f"<h4>Soft assignment: {pair_key.replace('_to_', ' -> ')}</h4>"
        try:
            fig_sa = pc.pl.soft_assignment_heatmap(adata, fr, show=False)
            html += safe_plotly_html(report, fig_sa,
                                    f"Soft assignment (rows: source archetypes, cols: target archetypes): {pair_key}")
        except Exception as e:
            html += error_html(f"Soft assignment heatmap ({pair_key}) failed: {e}")

    # Discrete archetype transition matrices (more robust than soft assignment)
    for pair_key, fr in flow_results.items():
        try:
            html += f"<h4>Discrete transition: {pair_key.replace('_to_', ' -> ')}</h4>"
            src_archs = adata.obs.loc[fr["source_mask"], "archetypes"].values
            transported = fr["transported"]
            arch_positions = np.asarray(adata.uns["archetype_coordinates"])
            n_pcs = min(transported.shape[1], arch_positions.shape[1])
            # dists[i, k] = ||transported[i] - arch_positions[k]||
            dists = np.linalg.norm(
                transported[:, None, :n_pcs] - arch_positions[None, :, :n_pcs],
                axis=2
            )
            tgt_archs = np.argmin(dists, axis=1)  # nearest archetype for each transported cell

            K = arch_positions.shape[0]
            trans_matrix = np.zeros((K, K))
            for sa, ta in zip(src_archs, tgt_archs):
                # Handle both "A2" and "archetype_2" label formats
                if isinstance(sa, str):
                    sa_clean = str(sa).replace("archetype_", "").replace("A", "")
                    try:
                        si = int(sa_clean) - 1
                    except ValueError:
                        continue
                else:
                    si = int(sa)
                if 0 <= si < K:
                    trans_matrix[si, ta] += 1

            # Normalize rows
            row_sums = trans_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            trans_prop = trans_matrix / row_sums

            fig_trans, ax_trans = plt.subplots(figsize=(6, 5))
            im_trans = ax_trans.imshow(trans_prop, cmap="YlOrRd", vmin=0, vmax=1)
            ax_trans.set_xticks(range(K))
            ax_trans.set_xticklabels([f"A{k+1}\n(target)" for k in range(K)])
            ax_trans.set_yticks(range(K))
            ax_trans.set_yticklabels([f"A{k+1}\n(source)" for k in range(K)])
            for i in range(K):
                for j in range(K):
                    ax_trans.text(j, i, f"{trans_prop[i,j]:.0%}", ha="center", va="center",
                                 fontsize=9, color="white" if trans_prop[i, j] > 0.5 else "black")
            plt.colorbar(im_trans, ax=ax_trans, label="Proportion", shrink=0.8)
            ax_trans.set_title(f"Discrete archetype transition: {pair_key.replace('_to_', ' -> ')}")
            fig_trans.tight_layout()
            html += report.fig_to_img(fig_trans,
                                      caption=f"Discrete transition matrix: {pair_key} (row=source arch, col=target after transport)")
            plt.close("all")
        except Exception as e:
            html += error_html(f"Discrete transition matrix ({pair_key}) failed: {e}")
            plt.close("all")

    html += report.text(
        "Discrete transition matrix: source cells are transported via the learned flow field, "
        "then assigned to the nearest archetype by Euclidean distance in PCA space. "
        "Row = source archetype (pre-transport assignment), column = target archetype (post-transport nearest)."
    )

    report.add_section("Biological Transition Flow", html, step_num=12)
    return flow_results


def step13_gene_alignment(adata, flow_results, report):
    """Step 13: Gene alignment along flow, gene expression change.
    FIX #19: Color bars by archetype association, note on flow-aligned vs upregulated."""
    import peach as pc

    html = ""
    alignment_results = {}

    if not flow_results:
        html += error_html("No flow results available for gene alignment.")
        report.add_section("Gene Alignment Along Flow", html, step_num=13)
        return alignment_results

    # FIX #19: Note distinguishing "flow-aligned" from "upregulated"
    html += report.text(
        "Gene-flow alignment: cosine similarity between each gene's PCA loading vector "
        "and the mean flow velocity. Positive = gene expression changes in the same direction "
        "as the flow (not necessarily upregulated -- it means the gene's variation axis aligns "
        "with the transport direction). Negative = expression changes in the opposite direction. "
        "<strong>Flow-aligned != upregulated:</strong> a gene can be flow-aligned because it "
        "decreases monotonically along the trajectory, if that decrease is the dominant direction."
    )

    # Flow QC summary
    qc_rows = []
    for pair_key, fr in flow_results.items():
        qc_rows.append({
            "Pair": pair_key.replace("_to_", " -> "),
            "N source": int(fr["source_mask"].sum()),
            "N target": int(fr["target_mask"].sum()),
            "MMD before": f"{fr['mmd_before']:.4f}",
            "MMD after": f"{fr['mmd_after']:.4f}",
            "Reduction": f"{1 - fr['mmd_after']/max(fr['mmd_before'], 1e-10):.1%}",
        })
    if qc_rows:
        html += report.df_to_html(pd.DataFrame(qc_rows), caption="Flow QC summary")

    # Loss curves
    for pair_key, fr in flow_results.items():
        try:
            losses = fr.get("losses", [])
            if len(losses) > 0:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.plot(losses, color="#0072B2", linewidth=0.8)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title(f"Flow loss: {pair_key}")
                ax.spines[["top", "right"]].set_visible(False)
                fig.tight_layout()
                html += report.fig_to_img(fig, caption=f"Loss curve: {pair_key}")
                plt.close("all")
        except Exception:
            plt.close("all")

    # Accumulators for cross-transition comparison (C3+C5)
    all_aligned = {}
    all_opposed = {}

    # Gene alignment + velocity quiver
    # Get archetype associations for coloring (from simplex regression)
    arch_assoc = {}
    gene_reg = adata.uns.get("peach_simplex_regression_genes") or adata.uns.get("peach_simplex_regression")
    if gene_reg is not None:
        try:
            vertex_coefs = np.asarray(gene_reg["vertex_coefficients"])
            feat_names = list(gene_reg["feature_names"])
            for i, fname in enumerate(feat_names):
                arch_assoc[fname] = int(np.argmax(np.abs(vertex_coefs[i]))) + 1
        except Exception:
            pass

    for pair_key, fr in flow_results.items():
        _flow_label = pair_key.replace("_to_", " -> ")
        html += f"<h4>Flow: {_flow_label}</h4>"
        log.info(f"  Gene alignment: {pair_key}...")
        try:
            align = pc.tl.flow_gene_alignment(adata, fr, n_top=30, per_cell=False,
                                                n_permutations=200)
            alignment_results[pair_key] = align

            # Report FDR-significant gene count
            if "alignment_pvalues_fdr" in align:
                n_sig = int((np.asarray(align["alignment_pvalues_fdr"]) < 0.05).sum())
                html += report.text(f"FDR-significant aligned genes: {n_sig}")

            # Accumulate top aligned/opposed for cross-transition comparison (C3+C5)
            top_aligned_names = align.get("top_aligned", [])
            top_opposed_names = align.get("top_opposed", [])
            all_aligned[pair_key] = set(top_aligned_names[:100])
            all_opposed[pair_key] = set(top_opposed_names[:100])

            # FIX #19: Bar plot with archetype association coloring
            try:
                fig_bar = pc.pl.gene_alignment_barplot(adata, align, n_top=20, show=False)
                html += safe_plotly_html(report, fig_bar, f"Gene alignment: {pair_key}")
            except Exception as e:
                html += error_html(f"Gene alignment barplot ({pair_key}) failed: {e}")

            # Detailed alignment table: top 20 per direction with scores,
            # FDR p-value, and archetype association
            align_scores = align.get("alignment_scores", [])
            align_fdr = align.get("alignment_pvalues_fdr", [])
            if len(align_scores) > 0:
                fdr_arr = np.asarray(align_fdr) if len(align_fdr) > 0 else np.ones(len(align_scores))
                # Top 20 aligned (most positive)
                aligned_idx = np.argsort(align_scores)[::-1][:20]
                # Top 20 opposed (most negative)
                opposed_idx = np.argsort(align_scores)[:20]
                align_detail_rows = []
                for gi in aligned_idx:
                    gname = adata.var_names[gi]
                    gsymbol = ensembl_to_symbol(adata, [gname])[0]
                    assoc = arch_assoc.get(gname, "?")
                    fdr_val = float(fdr_arr[gi]) if gi < len(fdr_arr) else float("nan")
                    align_detail_rows.append({
                        "Gene": gsymbol,
                        "Direction": "aligned",
                        "Alignment score": f"{align_scores[gi]:.4f}",
                        "FDR q": f"{fdr_val:.3e}",
                        "Archetype assoc": f"A{assoc}" if isinstance(assoc, int) else str(assoc),
                    })
                for gi in opposed_idx:
                    gname = adata.var_names[gi]
                    gsymbol = ensembl_to_symbol(adata, [gname])[0]
                    assoc = arch_assoc.get(gname, "?")
                    fdr_val = float(fdr_arr[gi]) if gi < len(fdr_arr) else float("nan")
                    align_detail_rows.append({
                        "Gene": gsymbol,
                        "Direction": "opposed",
                        "Alignment score": f"{align_scores[gi]:.4f}",
                        "FDR q": f"{fdr_val:.3e}",
                        "Archetype assoc": f"A{assoc}" if isinstance(assoc, int) else str(assoc),
                    })
                html += report.df_to_html(
                    pd.DataFrame(align_detail_rows),
                    caption=f"Gene-flow alignment (top 20 per direction): {pair_key}"
                )
        except Exception as e:
            html += error_html(f"Gene alignment ({pair_key}) failed: {e}")

        try:
            fig_quiv = pc.pl.velocity_quiver(adata, fr, show=False)
            # Make cells more visible, arrows less dominant
            try:
                for trace in fig_quiv.data:
                    if hasattr(trace, 'marker') and trace.marker is not None:
                        trace.marker.opacity = 0.8
                        trace.marker.size = 5
                    if hasattr(trace, 'line') and trace.line is not None:
                        trace.line.width = 0.5
                        trace.opacity = 0.3
            except Exception:
                pass  # plotly trace structure varies
            html += safe_plotly_html(report, fig_quiv, f"Velocity quiver: {pair_key}")
        except Exception as e:
            html += error_html(f"Quiver ({pair_key}) failed: {e}")

    # C3+C5: Cross-transition gene alignment comparison
    if len(all_aligned) >= 2:
        html += "<h3>Cross-transition gene alignment comparison</h3>"
        pair_names = list(all_aligned.keys())
        for i, p1 in enumerate(pair_names):
            for p2 in pair_names[i+1:]:
                shared = all_aligned[p1] & all_aligned[p2]
                html += report.text(f"{p1} ∩ {p2}: {len(shared)} shared aligned genes")
                if shared:
                    html += report.text(f"  -> {', '.join(sorted(list(shared))[:15])}")

    # C5: Compare flow-aligned genes to simplex regression top genes
    _reg_c5 = adata.uns.get("peach_simplex_regression_genes") or adata.uns.get("peach_simplex_regression")
    if _reg_c5 and "feature_names" in _reg_c5 and len(all_aligned) > 0:
        try:
            reg_names_c5 = list(_reg_c5["feature_names"])
            reg_r2_c5 = np.asarray(_reg_c5["r_squared_degree1"])
            top100_reg_c5 = set(np.array(reg_names_c5)[np.argsort(reg_r2_c5)[-100:]])

            for pair_key_c5, aligned_set_c5 in all_aligned.items():
                overlap_c5 = aligned_set_c5 & top100_reg_c5
                html += report.text(f"Flow-aligned ({pair_key_c5}) ∩ simplex top-100: {len(overlap_c5)} genes")
        except Exception as e:
            html += error_html(f"Flow-aligned vs simplex comparison failed: {e}")

    # Gene expression change along flow (PCA reconstruction)
    if "PCs" in adata.varm:
        for pair_key, fr in flow_results.items():
            html += f"<h4>Expression change along flow: {pair_key.replace('_to_', ' -> ')}</h4>"
            try:
                source_pca = adata.obsm["X_pca"][fr["source_mask"]]
                transported = fr["transported"]
                delta_pca = transported - source_pca
                loadings = adata.varm["PCs"]
                n_pcs = delta_pca.shape[1]
                delta_expr = delta_pca @ loadings[:, :n_pcs].T
                mean_delta = delta_expr.mean(axis=0)

                sorted_idx = np.argsort(np.abs(mean_delta))[::-1]
                pw_rows = []
                for rank, gi in enumerate(sorted_idx[:20]):
                    gname = adata.var_names[gi]
                    gsymbol = ensembl_to_symbol(adata, [gname])[0]
                    pw_rows.append({
                        "Rank": rank + 1,
                        "Gene": gsymbol,
                        "Mean delta": f"{mean_delta[gi]:.4f}",
                        "Abs delta": f"{abs(mean_delta[gi]):.4f}",
                    })
                html += report.df_to_html(
                    pd.DataFrame(pw_rows),
                    caption=f"Gene expression delta along flow (PCA reconstruction): {pair_key}",
                )
            except Exception as e:
                html += error_html(f"Gene expression change ({pair_key}) failed: {e}")

    report.add_section("Gene Alignment Along Flow", html, step_num=13)
    return alignment_results


def step14_jacobian(adata, flow_results, alignment_results, report):
    """Step 14: Jacobian expansion/contraction, cross-method concordance.
    FIX #11: viridis colormap for transport magnitude, archetype associations for genes."""
    import peach as pc

    html = ""
    jac_results = {}

    for pair_key, fr in flow_results.items():
        html += f"<h4>Flow: {pair_key.replace('_to_', ' -> ')}</h4>"
        model = fr.get("model")
        if model is None:
            html += error_html(f"{pair_key}: no model in flow result, skipping Jacobian.")
            continue

        log.info(f"  Jacobian: {pair_key}...")
        try:
            jac = pc.tl.flow_jacobian(
                adata, fr, model, per_cell_features=True, n_top_features=500,
                n_permutations=1000,
            )
            jac_results[pair_key] = jac

            det = jac["jacobian_det"]
            expansion = jac["feature_expansion"]

            det_finite = det[np.isfinite(det)]
            html += metric_grid([
                metric_card(pair_key, "Flow pair"),
                metric_card(f"{np.median(det_finite):.4f}", "Median det(J)"),
                metric_card(f"{np.mean(det_finite):.4f}", "Mean det(J)"),
                metric_card(f"{np.std(det_finite):.4f}", "Std det(J)"),
                metric_card(f"{(det_finite > 1).mean():.1%}", "% expanding (det>1)"),
                metric_card(f"{(det_finite < 1).mean():.1%}", "% contracting (det<1)"),
            ])

            if np.std(det_finite) < 0.01:
                html += report.text(
                    "WARNING: Jacobian determinants are near-constant (std < 0.01). "
                    "This suggests the flow field has minimal local expansion/contraction. "
                    "Check flow training convergence and MMD improvement."
                )

            html += report.text(
                "Jacobian feature expansion: quadratic form L^T * J * L for each gene's normalized "
                "PCA loading L and the Jacobian J of the velocity field. Values >0 indicate the "
                "gene's PCA direction is locally expanding (diverging trajectories); <0 indicates "
                "contraction (converging). Evaluated at t=0.5 (midpoint of learned flow)."
            )

            # FIX #11: Top expanded/contracted genes with archetype associations
            if len(expansion) > 0:
                gene_names = list(adata.var_names)
                gene_reg = adata.uns.get("peach_simplex_regression_genes") or adata.uns.get("peach_simplex_regression")
                arch_assoc = {}
                if gene_reg is not None:
                    try:
                        vertex_coefs = np.asarray(gene_reg["vertex_coefficients"])
                        feat_names = list(gene_reg["feature_names"])
                        for i, fname in enumerate(feat_names):
                            arch_assoc[fname] = int(np.argmax(np.abs(vertex_coefs[i]))) + 1
                        log.info(f"  Archetype associations mapped: {len(arch_assoc)}/{len(gene_names)} genes")
                    except Exception as e:
                        log.warning(f"  Archetype association mapping failed: {e}")
                else:
                    log.warning("  No simplex regression results in adata.uns for archetype mapping")

                sorted_exp = np.argsort(expansion)
                top_expand = sorted_exp[-15:][::-1]
                top_contract = sorted_exp[:15]
                exp_rows = []
                for gi in top_expand:
                    gsymbol = ensembl_to_symbol(adata, [gene_names[gi]])[0]
                    assoc = arch_assoc.get(gene_names[gi], "?")
                    exp_rows.append({
                        "Gene": gsymbol,
                        "Expansion": f"{expansion[gi]:.4f}",
                        "Direction": "expanding",
                        "Archetype": f"A{assoc}" if isinstance(assoc, int) else str(assoc),
                    })
                for gi in top_contract:
                    gsymbol = ensembl_to_symbol(adata, [gene_names[gi]])[0]
                    assoc = arch_assoc.get(gene_names[gi], "?")
                    exp_rows.append({
                        "Gene": gsymbol,
                        "Expansion": f"{expansion[gi]:.4f}",
                        "Direction": "contracting",
                        "Archetype": f"A{assoc}" if isinstance(assoc, int) else str(assoc),
                    })
                html += report.df_to_html(pd.DataFrame(exp_rows),
                                          caption=f"Top expanded/contracted genes: {pair_key}")

                # B15: Top expanding genes with archetype associations from simplex regression
                try:
                    reg_result_b15 = adata.uns.get("peach_simplex_regression_genes") or adata.uns.get("peach_simplex_regression")
                    if reg_result_b15 and "vertex_coefficients" in reg_result_b15:
                        vcoefs_b15 = np.asarray(reg_result_b15["vertex_coefficients"])
                        reg_names_b15 = list(reg_result_b15.get("feature_names", adata.var_names))
                        name_to_idx_b15 = {n: i for i, n in enumerate(reg_names_b15)}

                        gene_names_b15 = list(adata.var_names)
                        top_exp_idx_b15 = np.argsort(expansion)[-20:][::-1]
                        exp_arch_rows = []
                        for idx_b15 in top_exp_idx_b15:
                            gname_b15 = gene_names_b15[idx_b15] if idx_b15 < len(gene_names_b15) else f"gene_{idx_b15}"
                            reg_idx_b15 = name_to_idx_b15.get(gname_b15)
                            if reg_idx_b15 is not None:
                                arch_label_b15 = f"A{np.argmax(np.abs(vcoefs_b15[reg_idx_b15])) + 1}"
                            else:
                                arch_label_b15 = "N/A"
                            exp_arch_rows.append({
                                "Gene": ensembl_to_symbol(adata, [gname_b15])[0],
                                "Expansion": f"{expansion[idx_b15]:.4f}",
                                "Dominant_archetype": arch_label_b15,
                            })
                        html += report.df_to_html(pd.DataFrame(exp_arch_rows),
                            caption=f"Top 20 expanding genes with archetype associations: {pair_key}")
                except Exception as e_b15:
                    html += error_html(f"Expansion-archetype table ({pair_key}) failed: {e_b15}")

        except Exception as e:
            html += error_html(f"Jacobian ({pair_key}) failed: {e}")

    # Report permutation-significant counts across all flow pairs
    for pair_key, jac in jac_results.items():
        n_raw_sig = jac.get("expansion_n_raw_significant", 0)
        n_fdr_sig = int((np.asarray(jac.get("expansion_pvalues_fdr", [])) < 0.05).sum()) if "expansion_pvalues_fdr" in jac else 0
        n_perms_used = jac.get("n_permutations", "N/A")
        html += report.text(
            f"Jacobian permutation ({pair_key}): {n_raw_sig} genes at raw p<0.01, "
            f"{n_fdr_sig} at FDR q<0.05 ({n_perms_used} permutations). "
            "Note: FDR is conservative for loading-shuffle nulls (correlated null distributions). "
            "Raw p-values are more appropriate for ranking."
        )

    html += report.text(
        "Jacobian permutation null: gene-to-PCA-loading assignments are shuffled while "
        "the Jacobian matrix is held fixed. This tests whether a gene's expansion score "
        "is specific to its PCA loading direction or could arise by chance."
    )

    # Cross-method concordance (alignment vs expansion)
    for pair_key in set(jac_results.keys()) & set(alignment_results.keys()):
        try:
            from scipy.stats import spearmanr

            jac = jac_results[pair_key]
            align = alignment_results[pair_key]
            expansion = jac["feature_expansion"]
            align_scores = align["alignment_scores"]

            if len(expansion) == len(align_scores) and len(expansion) > 10:
                rho, p = spearmanr(np.abs(expansion), np.abs(align_scores))
                html += report.text(
                    f"Concordance |expansion| vs |alignment| for {pair_key}: "
                    f"Spearman rho={rho:.3f}, p={p:.2e}"
                )

                top_exp = set(np.argsort(np.abs(expansion))[-50:])
                top_ali = set(np.argsort(np.abs(align_scores))[-50:])
                jaccard = len(top_exp & top_ali) / max(len(top_exp | top_ali), 1)
                html += metric_grid([
                    metric_card(f"{jaccard:.2f}", "Jaccard top-50 (expansion vs alignment)"),
                ])

                fig, ax = plt.subplots(figsize=(5, 5))
                ax.scatter(np.abs(expansion), np.abs(align_scores), s=4, alpha=0.2, c="#0072B2")
                ax.set_xlabel("|Feature expansion|")
                ax.set_ylabel("|Gene alignment|")
                ax.set_title(f"Concordance: {pair_key} (rho={rho:.3f})")
                ax.spines[["top", "right"]].set_visible(False)
                fig.tight_layout()
                html += report.fig_to_img(fig, caption=f"Expansion vs alignment: {pair_key}")
                plt.close("all")
        except Exception as e:
            html += error_html(f"Concordance ({pair_key}) failed: {e}")
            plt.close("all")

    # FIX #11: Flow magnitude with viridis colormap
    for pair_key, fr in flow_results.items():
        try:
            fig_mag = pc.pl.flow_magnitude(adata, fr, show=False)
            # Try to update colorscale to viridis
            try:
                fig_mag.update_traces(marker=dict(opacity=0.6, colorscale="Viridis"))
            except Exception:
                fig_mag.update_traces(marker=dict(opacity=0.6))
            html += safe_plotly_html(report, fig_mag, f"Flow magnitude (viridis): {pair_key}")
        except Exception as e:
            html += error_html(f"Flow magnitude ({pair_key}) failed: {e}")

    report.add_section("Jacobian Expansion/Contraction", html, step_num=14)
    return jac_results


def step15_gene_deep_dive(adata, flow_results, jac_results, report):
    """Step 15: Trajectory ribbon, topo landscape, per-cell expansion violins.
    FIX #12: Label violin axes, add expansion score context."""
    import peach as pc

    html = ""

    # Trajectory ribbon for each flow
    html += report.text(
        "Trajectory ribbon: cells transported from source (t=0) to target (t=1) "
        "through the learned flow field, colored by transport time step. "
        "Tighter ribbons indicate more coherent transport; spread indicates divergence."
    )
    for pair_key, fr in flow_results.items():
        html += f"<h4>Flow: {pair_key.replace('_to_', ' -> ')}</h4>"
        model = fr.get("model")
        try:
            fig_rib = pc.pl.trajectory_ribbon(adata, fr, flow_model=model, show=False)
            html += safe_plotly_html(report, fig_rib, f"Trajectory ribbon: {pair_key}")
        except Exception as e:
            html += error_html(f"Trajectory ribbon ({pair_key}) failed: {e}")

    # Topo landscape for first flow pair with model
    for pair_key, fr in flow_results.items():
        html += f"<h4>Topo landscape: {pair_key.replace('_to_', ' -> ')}</h4>"
        model = fr.get("model")
        if model is None:
            continue
        try:
            fig_topo = pc.pl.flow_topo_landscape(
                adata, fr, model, n_features=5, n_eval_points=200,
                show_velocity=False, show=False,
                save=os.path.join(OUTPUT_DIR, f"topo_{pair_key}.png"),
            )
            topo_path = os.path.join(OUTPUT_DIR, f"topo_{pair_key}.png")
            if os.path.exists(topo_path):
                with open(topo_path, "rb") as fh:
                    b64 = base64.b64encode(fh.read()).decode("utf-8")
                html += f'<img src="data:image/png;base64,{b64}" style="max-width:100%;">'
                html += f"<p class='caption'>Topographic landscape: {pair_key}</p>"
        except Exception as e:
            html += error_html(f"Topo landscape ({pair_key}) failed: {e}")
            plt.close("all")
        break  # Only first pair

    # Expression-vs-flow-change scatter plots (top 5 expanding genes per flow pair)
    for pair_key, jac in jac_results.items():
        fr = flow_results.get(pair_key)
        if fr is None:
            continue
        per_cell = jac.get("per_cell_expansion")
        gene_names_jac = jac.get("per_cell_expansion_gene_names", [])
        if per_cell is None or len(gene_names_jac) == 0:
            continue
        if "PCs" not in adata.varm:
            html += error_html(f"Expression scatter ({pair_key}): adata.varm['PCs'] not found.")
            continue

        html += f"<h4>Expression vs flow-change scatter: {pair_key.replace('_to_', ' -> ')}</h4>"
        html += report.text(
            "Scatter plots: for each of the top 5 expanding genes, "
            "x-axis = source cell expression (PCA reconstructed), "
            "y-axis = expression change along flow (delta from OT transport). "
            "Reveals whether high-expressing cells show systematically larger expression changes."
        )

        try:
            source_pca = adata.obsm["X_pca"][fr["source_mask"]]
            transported = fr["transported"]
            n_pcs = min(source_pca.shape[1], transported.shape[1])
            delta_pca = transported[:, :n_pcs] - source_pca[:, :n_pcs]
            loadings = adata.varm["PCs"]  # shape: (n_genes, n_pcs)

            # Get the top 5 expanding genes by mean per-cell expansion score
            mean_expansion = per_cell.mean(axis=0)
            n_top = min(5, len(gene_names_jac))
            top5_idx = np.argsort(mean_expansion)[-n_top:][::-1]
            top5_names = [gene_names_jac[i] for i in top5_idx]
            top5_symbols = ensembl_to_symbol(adata, top5_names)

            # For each top gene, compute source expression and delta expression
            fig, axes = plt.subplots(1, n_top, figsize=(5 * n_top, 4))
            if n_top == 1:
                axes = [axes]
            for plot_i, (gene_name, gene_sym) in enumerate(zip(top5_names, top5_symbols)):
                ax = axes[plot_i]
                # Find gene index in adata.var_names
                if gene_name in adata.var_names:
                    gene_idx = list(adata.var_names).index(gene_name)
                else:
                    ax.set_visible(False)
                    continue
                # Source expression: PCA reconstructed
                gene_loading = loadings[gene_idx, :n_pcs]  # shape (n_pcs,)
                src_expr = source_pca @ gene_loading  # shape (n_cells,)
                # Expression change along flow: delta_pca projected onto gene loading
                delta_expr = delta_pca @ gene_loading  # shape (n_cells,)
                ax.scatter(src_expr, delta_expr, s=4, alpha=0.3, c="#0072B2")
                ax.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
                # Pearson correlation + regression line
                from scipy.stats import pearsonr
                r_val, p_val = pearsonr(src_expr, delta_expr)
                x_range = np.linspace(src_expr.min(), src_expr.max(), 100)
                slope = r_val * delta_expr.std() / max(src_expr.std(), 1e-10)
                intercept = delta_expr.mean() - slope * src_expr.mean()
                ax.plot(x_range, slope * x_range + intercept, 'r-', alpha=0.7, linewidth=1.5)
                ax.set_xlabel("Source expression (PCA reconstructed)")
                ax.set_ylabel("Expression change (delta)")
                ax.set_title(f"{gene_sym} (r={r_val:.3f}, p={p_val:.1e})", fontsize=10)
                ax.spines[["top", "right"]].set_visible(False)
            fig.suptitle(f"Expression vs flow-change: {pair_key}", y=1.02)
            fig.tight_layout()
            html += report.fig_to_img(
                fig,
                caption=f"Expression vs flow-change scatter (top 5 expanding genes): {pair_key}"
            )
            plt.close("all")
        except Exception as e:
            html += error_html(f"Expression scatter ({pair_key}) failed: {e}")
            plt.close("all")
        break  # Only first pair

    report.add_section("Gene Deep Dive", html, step_num=15)


def step_synthesis(adata, flow_results, alignment_results, jac_results, report):
    """Cross-step synthesis: connect flow, alignment, Jacobian, and regression results."""
    html = ""

    if not flow_results:
        html += report.text("No flow results available for synthesis.")
        report.add_section("Cross-Step Synthesis", html, step_num=None)
        return

    # Get archetype assignments and regression exclusive features
    arch_assignments = adata.obs.get("archetypes")
    gene_reg = (adata.uns.get("peach_simplex_regression_genes")
                or adata.uns.get("peach_simplex_regression"))

    # Build archetype-exclusive feature map from pattern classification
    pattern_data = adata.uns.get("peach_feature_patterns")
    exclusive_map = {}  # arch_idx -> set of feature names
    if pattern_data is not None:
        arch_feat = pattern_data.get("archetype_features", {})
        for k, feats in arch_feat.items():
            exclusive_map[k] = set(feats)

    # Build gene → dominant archetype lookup from simplex regression coefficients
    gene_to_arch = {}
    if gene_reg:
        feat_names = list(gene_reg.get("feature_names", []))
        vertex_coefs = gene_reg.get("vertex_coefficients")
        if feat_names and vertex_coefs is not None:
            vertex_coefs = np.asarray(vertex_coefs)
            for i, fname in enumerate(feat_names):
                gene_to_arch[fname] = f"A{int(np.argmax(np.abs(vertex_coefs[i]))) + 1}"

    # --- Per-flow-pair synthesis ---
    for pair_key, fr in flow_results.items():
        parts = pair_key.split("_to_")
        if len(parts) != 2:
            continue
        src_short, tgt_short = parts
        src_long, tgt_long = long_ct(src_short), long_ct(tgt_short)
        src_mask = fr.get("source_mask")
        tgt_mask = fr.get("target_mask")
        if src_mask is None or tgt_mask is None:
            continue

        html += f"<h4>Synthesis: {src_short} -> {tgt_short}</h4>"

        # (a) Archetype transition matrix
        try:
            if arch_assignments is not None:
                src_archs = arch_assignments[src_mask].values
                tgt_archs = arch_assignments[tgt_mask].values
                all_archs = sorted(set(src_archs) | set(tgt_archs))

                # Use transported positions to find nearest target archetype
                transported = fr.get("transported")
                if transported is not None and "archetype_coordinates" in adata.obsm:
                    from scipy.spatial.distance import cdist
                    # Get archetype positions (mean coordinates per archetype)
                    arch_coords = adata.obsm["archetype_coordinates"]
                    arch_means = {}
                    for a in all_archs:
                        a_mask = arch_assignments.values == a
                        if a_mask.sum() > 0:
                            arch_means[a] = arch_coords[a_mask].mean(axis=0)

                    if arch_means:
                        arch_names = sorted(arch_means.keys())
                        arch_centers = np.array([arch_means[a] for a in arch_names])

                        # Map transported source cells to nearest archetype
                        # transported is in PCA space; project to archetype coordinates
                        n_pcs = transported.shape[1]
                        src_pca = adata.obsm["X_pca"][src_mask][:, :n_pcs]
                        # Distances from transported positions to archetype centers in PCA
                        # Use archetype coord means for distance
                        dists = cdist(transported, arch_centers[:, :min(n_pcs, arch_centers.shape[1])])
                        nearest_tgt = np.array(arch_names)[np.argmin(dists, axis=1)]

                        # Build transition matrix
                        trans_rows = []
                        for sa in sorted(set(src_archs)):
                            sa_mask_local = src_archs == sa
                            if sa_mask_local.sum() == 0:
                                continue
                            nearest_for_sa = nearest_tgt[sa_mask_local]
                            for ta in sorted(set(nearest_for_sa)):
                                frac = (nearest_for_sa == ta).sum() / sa_mask_local.sum()
                                if frac > 0.01:
                                    trans_rows.append({
                                        "Source Arch": str(sa),
                                        "Target Arch": str(ta),
                                        "% of flow": f"{frac:.1%}",
                                        "N cells": int((nearest_for_sa == ta).sum()),
                                    })
                        if trans_rows:
                            html += report.df_to_html(pd.DataFrame(trans_rows),
                                                      caption=f"Archetype transition matrix: {pair_key}")
        except Exception as e:
            html += error_html(f"Archetype transition matrix ({pair_key}) failed: {e}")

        # (d-f) Top aligned, expanding, and exclusive genes per flow pair
        try:
            top_aligned = []
            top_expanding = []
            if pair_key in alignment_results:
                align = alignment_results[pair_key]
                top_aligned = align.get("top_aligned", [])[:15]
            if pair_key in jac_results:
                jac = jac_results[pair_key]
                expansion = jac.get("feature_expansion")
                if expansion is not None and len(expansion) > 0:
                    gene_names = list(adata.var_names)
                    top_exp_idx = np.argsort(expansion)[-15:][::-1]
                    top_expanding = [gene_names[i] for i in top_exp_idx if i < len(gene_names)]

            # Find shared between aligned and expanding
            shared_genes = set(top_aligned) & set(top_expanding)

            # Find which exclusive features overlap
            shared_exclusive = set()
            for k, excl_set in exclusive_map.items():
                shared_exclusive |= (set(top_aligned[:30]) | set(top_expanding[:30])) & excl_set

            summary_rows = []
            for g in sorted(shared_genes):
                gsymbol = ensembl_to_symbol(adata, [g])[0]
                in_exclusive = "Yes" if g in shared_exclusive else "No"
                dominant_arch = gene_to_arch.get(g, "n/a")
                summary_rows.append({
                    "Gene": gsymbol,
                    "Flow-aligned": "Yes",
                    "Expanding": "Yes",
                    "Archetype-exclusive": in_exclusive,
                    "Archetype": dominant_arch,
                })
            if summary_rows:
                html += report.df_to_html(pd.DataFrame(summary_rows),
                                          caption=f"Genes both flow-aligned and expanding: {pair_key}")
            else:
                html += report.text(f"No overlap between top-15 aligned and top-15 expanding genes for {pair_key}.")

        except Exception as e:
            html += error_html(f"Gene overlap synthesis ({pair_key}) failed: {e}")

    # --- Cross-transition shared genes ---
    if len(alignment_results) >= 2 or len(jac_results) >= 2:
        html += "<h4>Cross-transition universal drivers</h4>"
        try:
            all_aligned_sets = {}
            all_expanding_sets = {}
            for pk in flow_results:
                if pk in alignment_results:
                    al = alignment_results[pk]
                    all_aligned_sets[pk] = set(al.get("top_aligned", [])[:50])
                if pk in jac_results:
                    jac = jac_results[pk]
                    exp = jac.get("feature_expansion")
                    if exp is not None and len(exp) > 0:
                        gnames = list(adata.var_names)
                        top_idx = np.argsort(exp)[-50:][::-1]
                        all_expanding_sets[pk] = set(gnames[i] for i in top_idx if i < len(gnames))

            # Genes aligned in ALL transitions
            if len(all_aligned_sets) >= 2:
                universal_aligned = set.intersection(*all_aligned_sets.values())
                if universal_aligned:
                    syms = ensembl_to_symbol(adata, sorted(list(universal_aligned))[:20])
                    html += report.text(
                        f"Universally aligned genes (top-50, all {len(all_aligned_sets)} transitions): "
                        f"{', '.join(syms)}"
                    )
                else:
                    html += report.text("No genes are in the top-50 aligned set for all transitions.")

            if len(all_expanding_sets) >= 2:
                universal_expanding = set.intersection(*all_expanding_sets.values())
                if universal_expanding:
                    syms = ensembl_to_symbol(adata, sorted(list(universal_expanding))[:20])
                    html += report.text(
                        f"Universally expanding genes (top-50, all {len(all_expanding_sets)} transitions): "
                        f"{', '.join(syms)}"
                    )
                else:
                    html += report.text("No genes are in the top-50 expanding set for all transitions.")
        except Exception as e:
            html += error_html(f"Cross-transition analysis failed: {e}")

    # --- Component expansion tracking ---
    gmm = adata.uns.get("peach_gmm")
    if gmm is not None and flow_results:
        html += "<h4>Component distribution shifts along flow</h4>"
        try:
            assignments = np.asarray(gmm["component_assignments"])
            for pair_key, fr in flow_results.items():
                src_mask = fr.get("source_mask")
                tgt_mask = fr.get("target_mask")
                if src_mask is None or tgt_mask is None:
                    continue
                src_dist = pd.Series(assignments[src_mask]).value_counts(normalize=True)
                tgt_dist = pd.Series(assignments[tgt_mask]).value_counts(normalize=True)
                # Align indices so both have the same components (missing → 0.0); ensures sums to 100%
                all_comps = sorted(set(src_dist.index) | set(tgt_dist.index))
                src_dist = src_dist.reindex(all_comps, fill_value=0.0)
                tgt_dist = tgt_dist.reindex(all_comps, fill_value=0.0)
                shift_rows = []
                for c in all_comps:
                    s = src_dist.get(c, 0.0)
                    t = tgt_dist.get(c, 0.0)
                    change = t - s
                    direction = "expanding" if change > 0.02 else ("contracting" if change < -0.02 else "stable")
                    shift_rows.append({
                        "Component": int(c),
                        "Source %": f"{s:.1%}",
                        "Target %": f"{t:.1%}",
                        "Change": f"{change:+.1%}",
                        "Direction": direction,
                    })
                if shift_rows:
                    html += report.df_to_html(pd.DataFrame(shift_rows),
                                              caption=f"Component distribution shift: {pair_key}")
            html += report.text(
                "Note: Component indices from independent archetype fits (source vs target cell types) "
                "are not guaranteed to correspond. Interpret component-level comparisons cautiously. "
                "A future iteration will add Hungarian matching by centroid distance to align components."
            )
        except Exception as e:
            html += error_html(f"Component expansion tracking failed: {e}")

    report.add_section("Cross-Step Synthesis", html, step_num=None)


# ============================================================================
# Steps 16-17: Lineage comparison
# ============================================================================

def step16_lineage_comparison(adata, report):
    """Step 16: Compare progenitor vs differentiated branch models."""
    import peach as pc

    html = ""

    # Define lineage branches
    branches = {
        "progenitors": ["HSC", "CMP"],
        "differentiated": ["CMP", "Mono"],
    }

    global_K = adata.obsm.get("cell_archetype_weights")
    K = global_K.shape[1] if global_K is not None else 5
    hidden_dims = [128, 256]

    summary_rows = []
    reg_rows = []
    branch_adatas = {}

    for branch, ct_list in branches.items():
        mask = _get_ct_mask(adata, ct_list)
        sub = _subset_adata(adata, mask, branch)

        if sub.shape[0] < MIN_CELLS_MODEL:
            html += error_html(f"{branch}: {sub.shape[0]} cells < {MIN_CELLS_MODEL}, skipped.")
            continue

        k_use = min(K, max(3, sub.shape[0] // 200))
        log.info(f"  {branch}: training K={k_use}...")
        res = run_subset_model(sub, K=k_use, hidden_dims=hidden_dims, label=branch)
        if res is None:
            continue

        branch_adatas[branch] = sub
        r2 = res.get("final_archetype_r2", float("nan"))
        summary_rows.append({
            "Branch": branch, "Cell types": ", ".join(ct_list),
            "N cells": sub.shape[0], "K": k_use,
            "Final R2": f"{r2:.4f}" if isinstance(r2, float) else str(r2),
        })

        try:
            reg = run_subset_regression(sub, label=branch)
            if reg is not None:
                reg_rows.append(_reg_summary_row(reg, branch))
                try:
                    fig_dot = pc.pl.archetype_regression_dotplot(sub, top_n=10, show=False)
                    html += safe_plotly_html(report, fig_dot, f"Regression dotplot: {branch}")
                except Exception as e_dot:
                    html += error_html(f"Regression dotplot ({branch}) failed: {e_dot}")
        except Exception as e:
            html += error_html(f"{branch} regression failed: {e}")

    if summary_rows:
        html += report.df_to_html(pd.DataFrame(summary_rows), caption="Branch model summary")
    if reg_rows:
        html += report.df_to_html(pd.DataFrame(reg_rows), caption="Branch regression summary")

    # Per-branch archetypal space using PEACH interactive plots
    for branch, sub in branch_adatas.items():
        try:
            fig_arch = pc.pl.archetypal_space(sub, color_by="cell_type_short",
                                               title=f"Archetypal space: {branch}",
                                               )
            html += safe_plotly_html(report, fig_arch, f"Archetypal space: {branch}")
        except Exception as e:
            html += error_html(f"Archetypal space ({branch}) failed: {e}")

    # Cross-branch feature similarity
    if len(branch_adatas) >= 2:
        br_list = list(branch_adatas.keys())
        try:
            sim = pc.tl.archetype_feature_similarity(
                branch_adatas[br_list[0]], adata_b=branch_adatas[br_list[1]]
            )
            spear = np.asarray(sim["spearman_matrix"])
            html += metric_grid([
                metric_card(f"{spear.mean():.3f}", "Mean cross-branch Spearman"),
                metric_card(f"{spear.max():.3f}", "Max cross-branch Spearman"),
            ])
        except Exception as e:
            html += error_html(f"Cross-branch feature similarity failed: {e}")

    # Clean up
    for sub in branch_adatas.values():
        del sub
    branch_adatas.clear()

    report.add_section("Lineage Branch Comparison", html, step_num=16)


def step17_full_differentiation(adata, report):
    """Step 17: Full differentiation trajectory analysis: HSC -> CMP -> Mono."""
    import peach as pc

    html = ""

    # Single myeloid trajectory
    trajectories = {
        "myeloid_full": ["HSC", "CMP", "Mono"],
    }

    for traj_name, ct_list in trajectories.items():
        html += f"<h4>Trajectory: {' -> '.join(ct_list)}</h4>"
        mask = _get_ct_mask(adata, ct_list)
        sub = _subset_adata(adata, mask, traj_name)

        if sub.shape[0] < MIN_CELLS_MODEL:
            html += error_html(f"{traj_name}: {sub.shape[0]} cells < {MIN_CELLS_MODEL}, skipped.")
            continue

        # Train model
        k_use = min(5, max(3, sub.shape[0] // 200))
        log.info(f"  {traj_name}: training K={k_use}...")
        try:
            res = run_subset_model(sub, K=k_use, hidden_dims=[128, 256], label=traj_name)
            if res is None:
                continue

            r2 = res.get("final_archetype_r2", float("nan"))
            # B17: K selection QC
            html += report.text(f"K={k_use} selected (cell count: {sub.shape[0]})")
            html += metric_grid([
                metric_card(sub.shape[0], "Cells"),
                metric_card(k_use, "K"),
                metric_card(f"{r2:.4f}" if isinstance(r2, float) else str(r2), "Final R2"),
            ])

            # Regression
            reg = run_subset_regression(sub, label=traj_name)
            if reg is not None:
                r2_d1 = np.asarray(reg["r_squared_degree1"])
                f_fdr = np.asarray(reg.get("f_pvalue_fdr", np.ones(len(r2_d1))))
                html += metric_grid([
                    metric_card(f"{r2_d1.mean():.4f}", "Mean gene R2"),
                    metric_card(int((f_fdr < 0.05).sum()), "N sig genes"),
                ])

                # Top genes by R2 (with symbols)
                feat_names = list(reg["feature_names"])
                top_idx = np.argsort(r2_d1)[-20:][::-1]
                top_rows = []
                for i, idx in enumerate(top_idx):
                    gsymbol = ensembl_to_symbol(adata, [feat_names[idx]])[0]
                    top_rows.append({
                        "Rank": i + 1,
                        "Gene": gsymbol,
                        "R2": f"{r2_d1[idx]:.4f}",
                        "FDR q": f"{f_fdr[idx]:.2e}",
                    })
                html += report.df_to_html(pd.DataFrame(top_rows),
                                          caption=f"Top 20 genes by R2: {traj_name}")

                try:
                    fig_dot = pc.pl.archetype_regression_dotplot(sub, top_n=10, show=False)
                    html += safe_plotly_html(report, fig_dot, f"Regression dotplot: {traj_name}")
                except Exception as e_dot:
                    html += error_html(f"Regression dotplot ({traj_name}) failed: {e_dot}")

            # Conditional associations
            if "cell_type_short" in sub.obs.columns:
                try:
                    cond_df = pc.tl.conditional_associations(sub, obs_column="cell_type_short",
                                                             verbose=False)
                    display_cols = ["archetype", "condition", "odds_ratio", "fdr_pvalue"]
                    display_cols = [c for c in display_cols if c in cond_df.columns]
                    html += report.df_to_html(cond_df[display_cols],
                                              caption=f"Cell type enrichment: {traj_name}")
                except Exception as e:
                    html += error_html(f"Conditional associations ({traj_name}) failed: {e}")

            # Archetypal space colored by cell type
            try:
                fig_arch = pc.pl.archetypal_space(sub, color_by="cell_type_short",
                                                   title=f"Archetypal space: {traj_name}")
                html += safe_plotly_html(report, fig_arch,
                                         f"Archetypal space: {traj_name}")
            except Exception as e:
                html += error_html(f"pc.pl.archetypal_space ({traj_name}) failed: {e}")

        except Exception as e:
            html += error_html(f"Trajectory {traj_name} failed: {e}")
        finally:
            del sub

    report.add_section("Full Differentiation Trajectories", html, step_num=17)


# ============================================================================
# Main runner
# ============================================================================

def main():
    import scanpy as sc
    import peach as pc

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report = HTMLReport("PEACH v0.5 -- HSC End-to-End Analysis")

    # Load data
    log.info(f"Loading data from {DATA_PATH}...")
    adata = sc.read_h5ad(DATA_PATH)
    log.info(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # -- Step 1: Dataset prep + subset -----------------------------------------
    t0 = time.time()
    adata_sub = None
    try:
        adata_sub = step1_dataset_prep(adata, report)
        log.info(f"Step 1 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 1 failed: {e}", exc_info=True)
        report.add_section("Dataset Preparation", error_html(f"Step 1 failed: {e}"), step_num=1)

    if adata_sub is None:
        log.error("Step 1 failed to produce subset. Cannot continue.")
        report.save(REPORT_PATH)
        return

    # From here on, use adata_sub (the 6-cell-type subset)
    adata = adata_sub
    del adata_sub

    safe_save_h5ad(adata, os.path.join(OUTPUT_DIR, "adata_step1.h5ad"))

    # -- Step 2: Hyperparameter search + model training ------------------------
    t0 = time.time()
    results = None
    try:
        results = step2_hyperparameter_fit(adata, report)
        log.info(f"Step 2 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 2 failed: {e}", exc_info=True)
        report.add_section("Hyperparameter Search & Model Training",
                           error_html(f"Step 2 failed: {e}"), step_num=2)

    safe_save_h5ad(adata, os.path.join(OUTPUT_DIR, "adata_step2.h5ad"))
    if results is not None:
        try:
            model = results.get("final_model") or results.get("model")
            if model is not None:
                import torch
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_state_dict.pt"))
        except Exception:
            pass

    # -- Step 3: Simplex regression + patterns ---------------------------------
    t0 = time.time()
    gene_reg = None
    try:
        gene_reg = step3_simplex_regression(adata, report)
        log.info(f"Step 3 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 3 failed: {e}", exc_info=True)
        report.add_section("Simplex Regression & Pattern Classification",
                           error_html(f"Step 3 failed: {e}"), step_num=3)

    # -- Step 4: Conditional associations --------------------------------------
    t0 = time.time()
    try:
        step4_hypergeometric(adata, report)
        log.info(f"Step 4 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 4 failed: {e}", exc_info=True)
        report.add_section("Hypergeometric Conditional Associations",
                           error_html(f"Step 4 failed: {e}"), step_num=4)

    # -- Step 5: Wald contrasts ------------------------------------------------
    t0 = time.time()
    try:
        if gene_reg is None:
            gene_reg = adata.uns.get("peach_simplex_regression_genes") or adata.uns.get("peach_simplex_regression", {})
        step5_wald_contrasts(adata, report, gene_reg)
        log.info(f"Step 5 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 5 failed: {e}", exc_info=True)
        report.add_section("Wald Contrasts",
                           error_html(f"Step 5 failed: {e}"), step_num=5)

    # -- Step 6: Within-fit comparisons ----------------------------------------
    t0 = time.time()
    try:
        step6_within_fit_comparisons(adata, report)
        log.info(f"Step 6 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 6 failed: {e}", exc_info=True)
        report.add_section("Within-Fit Comparisons (MMD & Feature Similarity)",
                           error_html(f"Step 6 failed: {e}"), step_num=6)

    safe_save_h5ad(adata, os.path.join(OUTPUT_DIR, "adata_step6.h5ad"))

    # -- Step 7: Driver regression ---------------------------------------------
    t0 = time.time()
    try:
        step7_driver_regression(adata, report)
        log.info(f"Step 7 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 7 failed: {e}", exc_info=True)
        report.add_section("Driver Regression",
                           error_html(f"Step 7 failed: {e}"), step_num=7)

    # -- Step 8: Mixture models ------------------------------------------------
    t0 = time.time()
    try:
        step8_mixture_models(adata, report)
        log.info(f"Step 8 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 8 failed: {e}", exc_info=True)
        report.add_section("Mixture Model Decomposition",
                           error_html(f"Step 8 failed: {e}"), step_num=8)

    # -- Step 9: Component characterization ------------------------------------
    t0 = time.time()
    try:
        step9_component_characterization(adata, report)
        log.info(f"Step 9 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 9 failed: {e}", exc_info=True)
        report.add_section("Component Characterization",
                           error_html(f"Step 9 failed: {e}"), step_num=9)

    safe_save_h5ad(adata, os.path.join(OUTPUT_DIR, "adata_global.h5ad"))
    log.info("Global adata saved.")

    # ==== Per-lineage analyses (steps 10-11) ====

    lineage_adatas = {}

    # -- Step 10: Per-lineage models -------------------------------------------
    t0 = time.time()
    try:
        lineage_adatas = step10_per_lineage_models(adata, report)
        log.info(f"Step 10 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 10 failed: {e}", exc_info=True)
        report.add_section("Per-Lineage Models",
                           error_html(f"Step 10 failed: {e}"), step_num=10)

    # -- Step 11: Per-lineage regression ---------------------------------------
    t0 = time.time()
    try:
        step11_per_lineage_regression(adata, lineage_adatas, report)
        log.info(f"Step 11 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 11 failed: {e}", exc_info=True)
        report.add_section("Per-Lineage Regression",
                           error_html(f"Step 11 failed: {e}"), step_num=11)

    # Free lineage adatas
    for sub in lineage_adatas.values():
        del sub
    lineage_adatas.clear()

    # ==== Biological transition flows (steps 12-15) ====

    flow_results = {}
    alignment_results = {}
    jac_results = {}

    # -- Step 12: Biological flow ----------------------------------------------
    t0 = time.time()
    try:
        flow_results = step12_biological_flow(adata, report)
        log.info(f"Step 12 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 12 failed: {e}", exc_info=True)
        report.add_section("Biological Transition Flow",
                           error_html(f"Step 12 failed: {e}"), step_num=12)

    # -- Step 13: Gene alignment -----------------------------------------------
    t0 = time.time()
    try:
        alignment_results = step13_gene_alignment(adata, flow_results, report)
        log.info(f"Step 13 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 13 failed: {e}", exc_info=True)
        report.add_section("Gene Alignment Along Flow",
                           error_html(f"Step 13 failed: {e}"), step_num=13)

    # -- Step 14: Jacobian -----------------------------------------------------
    t0 = time.time()
    try:
        jac_results = step14_jacobian(adata, flow_results, alignment_results, report)
        log.info(f"Step 14 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 14 failed: {e}", exc_info=True)
        report.add_section("Jacobian Expansion/Contraction",
                           error_html(f"Step 14 failed: {e}"), step_num=14)

    # -- Step 15: Gene deep dive -----------------------------------------------
    t0 = time.time()
    try:
        step15_gene_deep_dive(adata, flow_results, jac_results, report)
        log.info(f"Step 15 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 15 failed: {e}", exc_info=True)
        report.add_section("Gene Deep Dive",
                           error_html(f"Step 15 failed: {e}"), step_num=15)

    # -- Synthesis: cross-step integration ----------------------------------------
    t0 = time.time()
    try:
        step_synthesis(adata, flow_results, alignment_results, jac_results, report)
        log.info(f"Synthesis done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Synthesis failed: {e}", exc_info=True)
        report.add_section("Cross-Step Synthesis",
                           error_html(f"Synthesis failed: {e}"), step_num=None)

    # Free flow models
    for fr in flow_results.values():
        fr.pop("model", None)
    flow_results.clear()
    alignment_results.clear()
    jac_results.clear()

    # ==== Lineage comparison (steps 16-17) ====

    # -- Step 16: Lineage branch comparison ------------------------------------
    t0 = time.time()
    try:
        step16_lineage_comparison(adata, report)
        log.info(f"Step 16 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 16 failed: {e}", exc_info=True)
        report.add_section("Lineage Branch Comparison",
                           error_html(f"Step 16 failed: {e}"), step_num=16)

    # -- Step 17: Full differentiation trajectories ----------------------------
    t0 = time.time()
    try:
        step17_full_differentiation(adata, report)
        log.info(f"Step 17 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 17 failed: {e}", exc_info=True)
        report.add_section("Full Differentiation Trajectories",
                           error_html(f"Step 17 failed: {e}"), step_num=17)

    # Final save
    safe_save_h5ad(adata, os.path.join(OUTPUT_DIR, "adata_final.h5ad"))
    log.info("Final adata saved.")

    # Save report
    report.save(REPORT_PATH)

    # Symlink to latest run
    latest_link = "outputs/e2e_hsc_myeloid/latest"
    try:
        if os.path.islink(latest_link):
            os.unlink(latest_link)
        os.symlink(_RUN_ID, latest_link)
        log.info(f"Latest symlink: {latest_link} -> {_RUN_ID}")
    except Exception as e:
        log.warning(f"Could not create latest symlink: {e}")

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()

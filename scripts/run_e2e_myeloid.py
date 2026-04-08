#!/usr/bin/env python
"""PEACH v0.5 End-to-End Myeloid Analysis Pipeline.

Runs global characterization steps (1-9) headless and generates a rich HTML report.
Usage: conda run -n archetype python scripts/run_e2e_myeloid.py
"""

import matplotlib
matplotlib.use("Agg")

import base64
import io
import logging
import os
import pickle
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="plotly")


def safe_save_h5ad(adata, path):
    """Save adata to h5ad, stripping non-serializable objects from uns.

    h5ad can't handle PyTorch modules or mixed-type dicts. Rather than
    trying to detect every problematic object, just try the write and
    iteratively remove offending keys until it succeeds.
    """
    max_retries = 10
    stash = {}
    for attempt in range(max_retries):
        try:
            adata.write_h5ad(path)
            break
        except Exception as e:
            # Extract the offending key from the error message
            msg = str(e)
            # Look for pattern: "key 'X' of" or "key '/uns/X'"
            import re
            match = re.search(r"key ['\"]/?uns/([^'\"]+)['\"]", msg)
            if match:
                bad_key = match.group(1).split("/")[0]  # top-level uns key
                if bad_key not in stash and bad_key in adata.uns:
                    log.warning(f"h5ad save: removing non-serializable uns['{bad_key}']")
                    stash[bad_key] = adata.uns.pop(bad_key)
                    continue
            # Also handle torch modules by type name
            removed_any = False
            for k in list(adata.uns.keys()):
                type_name = type(adata.uns[k]).__name__
                if type_name in ("Deep_AA", "VAE_Base", "FlowModel", "Module"):
                    stash[k] = adata.uns.pop(k)
                    removed_any = True
            if removed_any:
                continue
            # Can't fix it — log and skip save
            log.error(f"h5ad save failed after {attempt+1} attempts: {e}")
            break
    # Restore stashed keys
    adata.uns.update(stash)


def fmt_pval(p, threshold=1e-300):
    """Format p-value, replacing machine-epsilon floor with readable text."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "NA"
    if p < threshold:
        return "< 1e-300"
    if p > 0.99:
        return f"{p:.3f}"
    if p > 0.01:
        return f"{p:.3f}"
    return f"{p:.2e}"


def display_arch(label):
    """Convert 0-indexed obs label (archetype_0) to 1-indexed display string (A1)."""
    if isinstance(label, str) and label.startswith("archetype_"):
        try:
            idx = int(label.split("_")[1])
            return f"A{idx + 1}"
        except (ValueError, IndexError):
            pass
    return str(label)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("e2e_myeloid")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = "/Users/honkala/Desktop/peach/data/hsc_10k.h5ad"
OUTPUT_DIR = "outputs/e2e_hsc"
_DATE_TAG = time.strftime("%Y%m%d")

# Auto-increment revision: find existing reports for today and bump
import glob as _glob
_existing = sorted(_glob.glob(os.path.join(OUTPUT_DIR, f"e2e_hsc_report_{_DATE_TAG}*.html")))
_REV = len(_existing) + 1
_REV_TAG = f"{_DATE_TAG}_r{_REV}"
REPORT_PATH = os.path.join(OUTPUT_DIR, f"e2e_hsc_report_{_REV_TAG}.html")

# ---------------------------------------------------------------------------
# Cell type constants (HSC myeloid trajectory)
# ---------------------------------------------------------------------------
CT_SHORT = {
    "hematopoietic stem cell": "HSC",
    "common myeloid progenitor": "CMP",
    "CD14-positive monocyte": "Mono",
}
CT_LONG = {v: k for k, v in CT_SHORT.items()}

FLOW_PAIRS = [
    ("HSC", "CMP"),
    ("CMP", "Mono"),
    ("HSC", "Mono"),
]

LINEAGE_GROUPS = {
    "progenitors": ["HSC", "CMP"],
    "myeloid": ["CMP", "Mono"],
    "full_trajectory": ["HSC", "CMP", "Mono"],
}

N_PCS = 13
MIN_CELLS_MODEL = 400
MIN_CELLS_FLOW = 200


# ============================================================================
# HTMLReport class
# ============================================================================

class HTMLReport:
    """Accumulates sections and writes a self-contained HTML file."""

    def __init__(self, title: str):
        self.title = title
        self.sections: list[dict] = []
        self.start_time = time.time()

    # -- primitives ----------------------------------------------------------

    def fig_to_img(self, fig, caption: str = "", dpi: int = 150) -> str:
        """Convert matplotlib figure to base64 <img> tag."""
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
        """Convert plotly figure to embedded HTML div."""
        import plotly.io as pio
        div = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
        html = f"<div class='plotly-wrap'>{div}</div>"
        if caption:
            html += f"<p class='caption'>{caption}</p>"
        return html

    def df_to_html(self, df: pd.DataFrame, caption: str = "", max_rows: int = 50) -> str:
        """Convert DataFrame to styled HTML table."""
        if len(df) > max_rows:
            df = df.head(max_rows)
            note = f"<p><em>Showing first {max_rows} of {len(df)} rows.</em></p>"
        else:
            note = ""
        table = df.to_html(
            classes="styled-table",
            index=True,
            float_format=lambda x: f"{x:.4g}",
            border=0,
        )
        html = ""
        if caption:
            html += f"<p class='caption'><strong>{caption}</strong></p>"
        html += f'<div style="overflow-x: auto; max-width: 100%;">{note}{table}</div>'
        return html

    def text(self, txt: str) -> str:
        """Wrap text in a paragraph."""
        return f"<p>{txt}</p>"

    # -- section management --------------------------------------------------

    def add_section(self, title: str, content_html: str, step_num: int | None = None):
        self.sections.append({
            "title": title,
            "content": content_html,
            "step": step_num,
        })

    # -- save ----------------------------------------------------------------

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
    """Small metric card for dashboard-style display."""
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


def define_splits(adata) -> dict:
    """Define boolean masks for dose/response splits."""
    splits = {}
    for col in ["treatment", "pCR"]:
        if col in adata.obs.columns:
            for val in adata.obs[col].unique():
                key = f"{col}={val}"
                splits[key] = (adata.obs[col] == val).values
    # Interaction splits
    if "treatment" in adata.obs.columns and "pCR" in adata.obs.columns:
        for tx in adata.obs["treatment"].unique():
            for resp in adata.obs["pCR"].unique():
                key = f"{tx}_{resp}"
                splits[key] = (
                    (adata.obs["treatment"] == tx) & (adata.obs["pCR"] == resp)
                ).values
    return splits


# ============================================================================
# Step functions
# ============================================================================

def _dense_X(adata):
    """Get dense X matrix."""
    return adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X


def step1_dataset_prep(adata, report):
    """Step 1: Dataset preparation -- subset, PCA recompute, pathways, overview."""
    import peach as pc
    import scanpy as sc

    html = ""

    # -- Subset to 3 cell types -----------------------------------------------
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
    # Required for pathway scoring -- MSigDB uses symbols. Do this after PCA so
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

        # Pathway score distribution
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
        inflation_factor_range=[1.0, 1.25, 1.5],
        cv_folds=3,
        max_epochs_cv=15,
        subsample_fraction=0.8,
    )

    # CV results table
    ranked = cv_summary.rank_by_metric("archetype_r2")
    # Filter out failed configs (-inf R²)
    ranked = [r for r in ranked if r["metric_value"] > -1e6]
    if not ranked:
        html += error_html("All CV configurations failed. Using fallback K=4.")
        ranked = [{"hyperparameters": {"n_archetypes": 4, "hidden_dims": [128, 256]}, "metric_value": 0.0, "std_error": 0.0}]
    cv_rows = []
    for r in ranked:
        hp = r["hyperparameters"]
        cv_rows.append({
            "K": hp["n_archetypes"],
            "hidden_dims": str(hp.get("hidden_dims", "N/A")),
            "inflation": hp.get("inflation_factor", "N/A"),
            "R2": f"{r['metric_value']:.4f}",
            "SE": f"{r.get('std_error', 0):.4f}",
        })
    cv_df = pd.DataFrame(cv_rows)
    html += report.df_to_html(cv_df, caption="CV search results (ranked by R-squared)")

    # Elbow curve (plotly)
    try:
        fig_elbow = pc.pl.elbow_curve(cv_summary, metrics=["archetype_r2", "rmse"])
        html += safe_plotly_html(report, fig_elbow, "Elbow curve: R-squared and RMSE vs K")
    except Exception as e:
        html += error_html(f"Elbow curve failed: {e}")

    # Use best CV-ranked configuration directly (no elbow heuristic override)
    best = ranked[0]
    html += report.text(f"Best CV configuration: K={best['hyperparameters']['n_archetypes']}, "
                        f"R²={best['metric_value']:.4f}")

    best_hp = best["hyperparameters"]
    best_K = best_hp["n_archetypes"]
    best_hidden = best_hp.get("hidden_dims", [128, 256])
    best_r2 = best["metric_value"]
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
        n_epochs=200,
        hidden_dims=best_hidden,
        kld_weight=0.0,
        archetypal_weight=1.0,
        early_stopping=True,
        early_stopping_patience=15,
    )
    final_r2 = results.get("final_archetype_r2", "N/A")
    html += metric_grid([
        metric_card(f"{final_r2:.4f}" if isinstance(final_r2, float) else final_r2,
                    "Final R-squared"),
    ])

    # Training metrics (plotly)
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

    # Archetypal space scatter (plotly)
    try:
        fig_space = pc.pl.archetypal_space(adata, color_by="cell_type_short",
                                           title="Archetypal space (cell type)")
        html += safe_plotly_html(report, fig_space, "Archetypal space colored by cell type")
    except Exception as e:
        html += error_html(f"Archetypal space plot failed: {e}")

    try:
        fig_space2 = pc.pl.archetypal_space(adata, color_by="archetypes",
                                            title="Archetypal space (archetypes)")
        html += safe_plotly_html(report, fig_space2, "Archetypal space colored by archetype assignment")
    except Exception as e:
        html += error_html(f"Archetypal space (pCR) failed: {e}")

    # Archetype statistics (returns a dict, NOT a figure)
    try:
        stats = pc.pl.archetype_statistics(adata, verbose=False)
        if stats is not None:
            stat_cards = [
                metric_card(stats.get("n_archetypes", "?"), "Archetypes"),
                metric_card(f"{stats.get('mean_distance', 0):.3f}", "Mean pairwise dist"),
                metric_card(f"{stats.get('min_distance', 0):.3f}", "Min pairwise dist"),
                metric_card(f"{stats.get('max_distance', 0):.3f}", "Max pairwise dist"),
            ]
            hull_vol = stats.get("hull_volume")
            if hull_vol is not None:
                stat_cards.append(metric_card(f"{hull_vol:.4f}", "Hull volume"))
            html += metric_grid(stat_cards)
    except Exception:
        pass

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

    # Gene simplex regression
    log.info("Running gene simplex regression (degree 1+2)...")
    gene_reg = pc.tl.gene_simplex_regression(adata, max_degree=2, robust_se=True,
                                              store_residuals=True)

    n_features = len(gene_reg["feature_names"])
    r2_d1 = np.asarray(gene_reg["r_squared_degree1"])
    f_fdr = np.asarray(gene_reg.get("f_pvalue_fdr", np.ones(n_features)))
    n_sig = int((f_fdr < 0.05).sum())

    html += metric_grid([
        metric_card(n_features, "Genes tested"),
        metric_card(n_sig, "FDR-significant (q<0.05)"),
        metric_card(f"{r2_d1.mean():.4f}", "Mean R-squared (degree 1)"),
        metric_card(f"{np.median(r2_d1):.4f}", "Median R-squared"),
        metric_card(f"{r2_d1.max():.4f}", "Max R-squared"),
    ])
    html += report.text("Note: Simplex regression R\u00b2 measures how well archetype weights predict "
                        "individual gene expression (per-gene fit quality). This differs from archetypal R\u00b2 "
                        "(step 2), which measures how well the model reconstructs the full PCA space.")

    # --- 1. Archetype-exclusive features ---
    try:
        fig_excl = pc.pl.archetype_regression_dotplot(
            adata, top_n=10, exclusive_only=True, exclusive_threshold=2.5,
            rank_by="r2", show=False)
        html += safe_plotly_html(report, fig_excl,
            "Archetype-exclusive genes (top 10 per archetype, ranked by R², "
            "exclusive = max|β| ≥ 2.5× second-highest)")
    except Exception as e:
        html += error_html(f"Exclusive dotplot failed: {e}")

    # --- 2. Top features by R² ---
    try:
        fig_top = pc.pl.archetype_regression_dotplot(
            adata, top_n=10, exclusive_only=False, rank_by="r2", show=False)
        html += safe_plotly_html(report, fig_top,
            "Top genes per archetype (all features, ranked by R²)")
    except Exception as e:
        html += error_html(f"Top features dotplot failed: {e}")

    # (Radar plot moved to step 6 where Spearman similarity data drives spoke angles)

    # Pattern classification
    log.info("Classifying feature patterns...")
    try:
        pattern_result = pc.tl.classify_feature_patterns(adata)
        pattern_counts = pattern_result.get("pattern_counts", {})
        html += report.text("Pattern classification counts:")
        pattern_df = pd.DataFrame(
            [{"Pattern": k, "Count": v} for k, v in pattern_counts.items()]
        )
        html += report.df_to_html(pattern_df, caption="Feature pattern counts")

        # Pattern summary bar chart
        try:
            fig_pat = pc.pl.pattern_summary(adata, show=False)
            html += safe_plotly_html(report, fig_pat, "Pattern type summary")
        except Exception as e:
            html += error_html(f"Pattern summary plot failed: {e}")

        # Archetype-feature map — filter for FDR significance, rank by R²
        arch_feat_map = pattern_result.get("archetype_features", {})
        if arch_feat_map and "feature_names" in gene_reg:
            feat_names_reg = list(gene_reg["feature_names"])
            r2_d1_arr = np.asarray(gene_reg["r_squared_degree1"])
            f_fdr_arr = np.asarray(gene_reg.get("f_pvalue_fdr", np.ones(len(r2_d1_arr))))
            vertex_coefs = np.asarray(gene_reg["vertex_coefficients"])

            # Build expanded table with one row per feature
            map_rows = []
            for k, feats in arch_feat_map.items():
                arch_label = f"A{k+1}" if isinstance(k, int) else str(k)
                for f in feats:
                    if f in feat_names_reg:
                        fi = feat_names_reg.index(f)
                        if f_fdr_arr[fi] < 0.05:
                            map_rows.append({
                                "Archetype": arch_label,
                                "Feature": f,
                                "R²": f"{r2_d1_arr[fi]:.4f}",
                                "FDR q": fmt_pval(f_fdr_arr[fi]),
                                "β (dominant)": f"{vertex_coefs[fi, k if isinstance(k, int) else 0]:.3f}",
                            })
            if map_rows:
                map_df = pd.DataFrame(map_rows)
                map_df = map_df.sort_values("R²", ascending=False)
                html += report.df_to_html(map_df, caption="Archetype-exclusive features (FDR < 0.05, ranked by R²)")
            else:
                html += report.text("No archetype-exclusive features pass FDR < 0.05.")
    except Exception as e:
        html += error_html(f"Pattern classification failed: {e}")

    # Interaction term reclassification
    try:
        int_coefs = gene_reg.get("interaction_coefficients")
        int_pairs = gene_reg.get("interaction_pairs", [])
        int_fdr = gene_reg.get("interaction_pvalues_fdr")
        vertex_coefs = np.asarray(gene_reg["vertex_coefficients"])
        feat_names = list(gene_reg["feature_names"])

        if int_coefs is not None and len(int_pairs) > 0 and int_fdr is not None:
            int_coefs = np.asarray(int_coefs)
            int_fdr = np.asarray(int_fdr)

            interaction_rows = []
            for feat_idx in range(len(feat_names)):
                for pair_idx, (j, k) in enumerate(int_pairs):
                    if int_fdr[feat_idx, pair_idx] < 0.05:
                        gamma = int_coefs[feat_idx, pair_idx]
                        beta_j = vertex_coefs[feat_idx, j]
                        beta_k = vertex_coefs[feat_idx, k]

                        # Classify vertex relationship
                        median_abs = np.median(np.abs(vertex_coefs[feat_idx]))
                        j_high = abs(beta_j) > median_abs
                        k_high = abs(beta_k) > median_abs
                        same_sign = (np.sign(beta_j) == np.sign(beta_k)
                                     and beta_j != 0 and beta_k != 0)

                        if j_high and k_high and same_sign:
                            pair_type = "cooperative"
                        elif (j_high and not k_high) or (not j_high and k_high):
                            pair_type = "tradeoff"
                        elif not j_high and not k_high and abs(gamma) > median_abs:
                            pair_type = "transition-enriched"
                        else:
                            pair_type = "structured"

                        transition = f"rising (A{j+1}\u2192A{k+1})" if gamma > 0 else f"falling (A{j+1}\u2192A{k+1})"

                        interaction_rows.append({
                            "Feature": feat_names[feat_idx],
                            "Pair": f"A{j+1}-A{k+1}",
                            "Type": pair_type,
                            "Transition": transition,
                            "beta_j": f"{beta_j:.3f}",
                            "beta_k": f"{beta_k:.3f}",
                            "Edge \u03b3": f"{gamma:.3f}",
                            "FDR q": fmt_pval(int_fdr[feat_idx, pair_idx]),
                        })

            if interaction_rows:
                int_df = pd.DataFrame(interaction_rows)
                # Sort by |γ| descending within each type
                int_df["abs_gamma"] = int_df["Edge \u03b3"].apply(lambda x: abs(float(x)))
                int_df = int_df.sort_values("abs_gamma", ascending=False)

                type_counts = int_df["Type"].value_counts()
                cards = [metric_card(len(interaction_rows), "Significant interactions")]
                for t in ["tradeoff", "cooperative", "transition-enriched", "structured"]:
                    cards.append(metric_card(int(type_counts.get(t, 0)), t.capitalize()))
                html += metric_grid(cards)

                html += report.text(
                    "Interaction classification: <b>Cooperative</b> = high at both archetypes, same sign "
                    "(shared program). <b>Tradeoff</b> = high at one, low at other (distinguishes "
                    "archetypes). <b>Transition-enriched</b> = peaks in blending zone, not at either vertex. "
                    "<b>Structured</b> = significant interaction with mixed pattern (e.g., both modest + opposite signs, "
                    "or one high with unclear contrast). "
                    "Edge \u03b3 direction: rising (\u03b3>0) = gene increases along archetype edge; "
                    "falling (\u03b3<0) = gene decreases. "
                    "Tables sorted by |\u03b3| descending.")

                for itype in ["tradeoff", "cooperative", "transition-enriched", "structured"]:
                    sub = int_df[int_df["Type"] == itype].drop(columns=["abs_gamma"]).head(20)
                    if len(sub) > 0:
                        html += report.df_to_html(sub, caption=f"Top {itype} interactions (ranked by |γ|)")
            else:
                html += report.text("No significant interaction terms at FDR < 0.05.")
        else:
            html += report.text("Interaction terms not available in regression results.")
    except Exception as e:
        html += error_html(f"Interaction classification failed: {e}")

    # Mutual exclusivity: pairwise tradeoff accounting
    try:
        if int_coefs is not None and len(int_pairs) > 0 and int_fdr is not None:
            me_rows = []
            for pair_idx, (j, k) in enumerate(int_pairs):
                for feat_idx in range(len(feat_names)):
                    if int_fdr[feat_idx, pair_idx] >= 0.05:
                        continue
                    beta_j = vertex_coefs[feat_idx, j]
                    beta_k = vertex_coefs[feat_idx, k]
                    median_abs = np.median(np.abs(vertex_coefs[feat_idx]))
                    # Tradeoff: one high, one low
                    if (abs(beta_j) > median_abs) != (abs(beta_k) > median_abs):
                        high_arch = f"A{j+1}" if abs(beta_j) > abs(beta_k) else f"A{k+1}"
                        low_arch = f"A{k+1}" if abs(beta_j) > abs(beta_k) else f"A{j+1}"
                        me_rows.append({
                            "archetype_high": high_arch,
                            "archetype_low": low_arch,
                            "gene": feat_names[feat_idx],
                            "direction": f"{high_arch}\u2192{low_arch}",
                            "beta_high": f"{max(abs(beta_j), abs(beta_k)):.3f}",
                            "beta_low": f"{min(abs(beta_j), abs(beta_k)):.3f}",
                            "gamma": f"{int_coefs[feat_idx, pair_idx]:.3f}",
                            "fdr": fmt_pval(int_fdr[feat_idx, pair_idx]),
                        })

            if me_rows:
                me_df = pd.DataFrame(me_rows)
                pair_summary = me_df.groupby(["archetype_high", "archetype_low"]).size().reset_index(name="n_genes")
                html += report.df_to_html(pair_summary,
                                          caption="Mutual exclusivity: tradeoff gene counts per archetype pair")
                html += report.df_to_html(me_df.head(40),
                                          caption="Mutual exclusivity: tradeoff genes (top 40)")

                # Mutual exclusivity dotplot: top genes per direction pair
                try:
                    top_me = me_df.copy()
                    top_me["abs_gamma"] = top_me["gamma"].apply(lambda x: abs(float(x)))
                    # Get top 5 per direction pair by |gamma|
                    top_per_dir = (
                        top_me.sort_values("abs_gamma", ascending=False)
                        .groupby("direction", sort=False)
                        .head(5)
                        .reset_index(drop=True)
                    )

                    if len(top_per_dir) > 0:
                        # Dotplot: genes on y-axis, archetype pairs on x-axis
                        # Dot size = |gamma|, color = direction (which archetype is high)
                        directions = sorted(top_per_dir["direction"].unique())
                        genes_shown = top_per_dir["gene"].unique()
                        n_genes_show = len(genes_shown)
                        n_dirs = len(directions)

                        fig_me, ax_me = plt.subplots(
                            figsize=(max(6, n_dirs * 1.2), max(4, n_genes_show * 0.35 + 1)))
                        cmap_dirs = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
                                     "#E69F00", "#56B4E9", "#F0E442", "#999999"]
                        for di, direction in enumerate(directions):
                            sub = top_per_dir[top_per_dir["direction"] == direction]
                            for _, row in sub.iterrows():
                                gi = list(genes_shown).index(row["gene"])
                                gamma_abs = abs(float(row["gamma"]))
                                ax_me.scatter(di, gi, s=gamma_abs * 300 + 20,
                                              c=cmap_dirs[di % len(cmap_dirs)],
                                              alpha=0.7, edgecolors="black", linewidth=0.3)

                        ax_me.set_xticks(range(n_dirs))
                        ax_me.set_xticklabels(directions, rotation=45, ha="right", fontsize=8)
                        ax_me.set_yticks(range(n_genes_show))
                        ax_me.set_yticklabels(genes_shown, fontsize=8)
                        ax_me.set_title("Mutual exclusivity: tradeoff genes by archetype pair")
                        ax_me.set_xlabel("Archetype pair (high→low)")
                        ax_me.spines[["top", "right"]].set_visible(False)
                        fig_me.tight_layout()
                        html += report.fig_to_img(fig_me,
                            caption="Mutual exclusivity dotplot: dot size = |γ| interaction strength")
                        plt.close("all")
                except Exception as e_me:
                    html += error_html(f"Mutual exclusivity dotplot failed: {e_me}")
                    plt.close("all")
    except Exception as e:
        html += error_html(f"Mutual exclusivity table failed: {e}")

    # Pathway simplex regression (if pathway scores exist)
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

            # Top 20 pathways by R² with per-archetype β values
            if len(pw_names) > 0:
                pw_vertex_coefs = np.asarray(pw_reg.get("vertex_coefficients", []))
                pw_K = pw_vertex_coefs.shape[1] if pw_vertex_coefs.ndim == 2 else 0
                pw_data = {"Pathway": pw_names, "R2": pw_r2, "FDR_q": pw_fdr}
                for k in range(pw_K):
                    pw_data[f"A{k+1} β"] = pw_vertex_coefs[:, k]
                pw_df = pd.DataFrame(pw_data)
                pw_df = pw_df.sort_values("R2", ascending=False).head(20)
                pw_df["FDR_q"] = pw_df["FDR_q"].apply(fmt_pval)
                for k in range(pw_K):
                    pw_df[f"A{k+1} β"] = pw_df[f"A{k+1} β"].apply(lambda x: f"{x:.3f}")
                html += report.df_to_html(pw_df, caption="Top 20 pathways by R² (pathway simplex regression)")

            # Pathway exclusive dotplot (fewer per archetype — pathway names are long)
            try:
                fig_pw_excl = pc.pl.archetype_regression_dotplot(
                    adata, top_n=5, exclusive_only=True, exclusive_threshold=2.5,
                    rank_by="r2", feature_type="pathways", show=False)
                # Calculate height from number of features, not traces
                n_features = len(set(d.y[0] for d in fig_pw_excl.data if hasattr(d, 'y') and len(d.y) > 0)) if fig_pw_excl.data else 10
                fig_pw_excl.update_layout(height=max(400, n_features * 25 + 80),
                                          margin=dict(l=350))
                html += safe_plotly_html(report, fig_pw_excl,
                    "Archetype-exclusive pathways (top 5 per archetype, ranked by R², "
                    "exclusive = max|β| ≥ 2.5× second-highest)")
            except Exception as e:
                html += error_html(f"Pathway exclusive dotplot failed: {e}")

            # Pathway top features dotplot
            try:
                fig_pw_top = pc.pl.archetype_regression_dotplot(
                    adata, top_n=5, exclusive_only=False, rank_by="r2",
                    feature_type="pathways", show=False)
                n_features = len(set(d.y[0] for d in fig_pw_top.data if hasattr(d, 'y') and len(d.y) > 0)) if fig_pw_top.data else 10
                fig_pw_top.update_layout(height=max(400, n_features * 25 + 80),
                                         margin=dict(l=350))
                html += safe_plotly_html(report, fig_pw_top,
                    "Top pathways per archetype (all, ranked by R²)")
            except Exception as e:
                html += error_html(f"Pathway dotplot failed: {e}")

            # (Pathway radar moved to step 6 where Spearman similarity data drives spoke angles)

            # Pattern classification summary for pathways
            # Stash gene patterns to avoid overwrite
            try:
                _gene_patterns_stash = adata.uns.pop("peach_feature_patterns", None)
                pw_pattern_result = pc.tl.classify_feature_patterns(
                    adata, regression_result=pw_reg
                )
                if pw_pattern_result is not None:
                    pw_pattern_counts = pw_pattern_result.get("pattern_counts", {})
                    if pw_pattern_counts:
                        pw_pat_df = pd.DataFrame(
                            [{"Pattern": k, "Count": v} for k, v in pw_pattern_counts.items()]
                        )
                        html += report.df_to_html(pw_pat_df, caption="Pathway pattern classification counts")
            except Exception as e:
                html += error_html(f"Pathway pattern classification failed: {e}")
            finally:
                # Restore gene patterns
                if _gene_patterns_stash is not None:
                    adata.uns["peach_feature_patterns"] = _gene_patterns_stash

        except Exception as e:
            html += error_html(f"Pathway simplex regression failed: {e}")

    report.add_section("Simplex Regression & Pattern Classification", html, step_num=3)
    return gene_reg


def step4_hypergeometric(adata, report):
    """Step 4: Conditional associations (hypergeometric tests)."""
    import peach as pc

    html = ""
    cond_results = {}

    for col in ["cell_type_short"]:
        if col not in adata.obs.columns:
            continue
        log.info(f"Conditional associations: {col}...")
        try:
            cond_df = pc.tl.conditional_associations(adata, obs_column=col, verbose=False)
            # Cap inf odds_ratios for display (inf = perfect enrichment, mathematically correct)
            if "odds_ratio" in cond_df.columns:
                cond_df["odds_ratio"] = cond_df["odds_ratio"].replace([np.inf], 999.0)
            cond_results[col] = cond_df

            sig_df = cond_df[cond_df["significant"] == True] if "significant" in cond_df.columns else cond_df[cond_df["fdr_pvalue"] < 0.05]
            html += report.text(f"<strong>{col}</strong>: {len(sig_df)} significant associations "
                                f"out of {len(cond_df)} tests.")

            # Show full table
            display_cols = ["archetype", "condition", "observed", "expected",
                            "odds_ratio", "fdr_pvalue", "significant"]
            display_cols = [c for c in display_cols if c in cond_df.columns]
            html += report.df_to_html(cond_df[display_cols], caption=f"Conditional associations: {col}")

            # Enrichment heatmap (matplotlib)
            try:
                pivot = cond_df.pivot_table(
                    index="archetype", columns="condition",
                    values="odds_ratio", aggfunc="first"
                )
                fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 1.2), max(4, pivot.shape[0] * 0.8)))
                import matplotlib.colors as mcolors
                # Clip inf odds ratios before plotting
                plot_vals = np.where(np.isfinite(pivot.values), pivot.values, np.nan)
                finite_positive = plot_vals[np.isfinite(plot_vals) & (plot_vals > 0)]
                if len(finite_positive) == 0:
                    finite_positive = np.array([0.1, 10.0])
                max_or = min(finite_positive.max() * 2, 1000)  # cap at 1000
                plot_vals = np.where(np.isnan(plot_vals), max_or, plot_vals)
                plot_vals = np.clip(plot_vals, 0.01, max_or)
                norm = mcolors.LogNorm(vmin=max(0.01, finite_positive.min()),
                                       vmax=max(max_or, finite_positive.min() * 10))
                im = ax.imshow(plot_vals, aspect="auto", cmap="RdBu_r", norm=norm)
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

        except Exception as e:
            html += error_html(f"Conditional associations for {col} failed: {e}")

    # Proportion bars
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

    report.add_section("Hypergeometric Conditional Associations", html, step_num=4)
    return cond_results


def step5_wald_contrasts(adata, report, gene_reg):
    """Step 5: Wald contrasts between archetypes."""
    import peach as pc

    html = ""

    log.info("Computing Wald contrasts...")
    contrast_result = pc.tl.archetype_contrasts(adata)
    # contrast_volcano_grid looks for 'peach_archetype_contrasts' (no _genes suffix),
    # but archetype_contrasts stores under 'peach_archetype_contrasts_genes'.
    # Mirror the result so the plot function can find it.
    adata.uns["peach_archetype_contrasts"] = contrast_result

    pairs = contrast_result.get("pairs", [])
    feature_names = list(contrast_result.get("feature_names", []))
    html += report.text(f"Tested {len(pairs)} archetype pairs across {len(feature_names)} features.")

    # Summary: count significant per pair
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
        fig_vg.update_layout(
            font=dict(size=12),
            xaxis_title="Δβ (effect size)",
            yaxis_title="-log10(FDR q-value)",
        )
        fig_vg.update_traces(textfont_size=10)
        html += safe_plotly_html(report, fig_vg, "Pairwise Wald contrast volcano grid")
    except Exception as e:
        html += error_html(f"Contrast volcano grid failed: {e}")

    # Top contrasts table (top 30 across all pairs, grouped by Direction)
    top_rows = []
    # Also collect per-gene per-pair directions for multi-pair confusion matrix
    gene_pair_directions = {}  # gene -> {pair_label -> "+"/"-"}
    all_pair_labels = [
        f"A{(pair[0] if isinstance(pair, (list, tuple)) else pair[0])+1}-"
        f"A{(pair[1] if isinstance(pair, (list, tuple)) else pair[1])+1}"
        for pair in pairs
    ]
    for pair in pairs:
        pair_key = str(tuple(pair) if isinstance(pair, list) else pair)
        pvals = np.asarray(contrast_result["pvalues_fdr"][pair_key])
        delta = np.asarray(contrast_result["delta_beta"][pair_key])
        j, k = pair if isinstance(pair, (list, tuple)) else (pair[0], pair[1])
        pair_label = f"A{j+1}-A{k+1}"
        for feat_idx in range(len(feature_names)):
            if pvals[feat_idx] < 0.05:
                direction = f"Up in A{j+1}" if delta[feat_idx] > 0 else f"Up in A{k+1}"
                top_rows.append({
                    "Feature": feature_names[feat_idx],
                    "Pair": f"A{j+1} vs A{k+1}",
                    "Direction": direction,
                    "delta_beta": delta[feat_idx],
                    "FDR q": pvals[feat_idx],
                })
                gene = feature_names[feat_idx]
                if gene not in gene_pair_directions:
                    gene_pair_directions[gene] = {}
                gene_pair_directions[gene][pair_label] = direction
    if top_rows:
        top_df = pd.DataFrame(top_rows)
        # Sort by Direction (ascending: + before -) then by |delta_beta| descending
        top_df["abs_delta"] = top_df["delta_beta"].abs()
        top_df = top_df.sort_values(["Direction", "abs_delta"], ascending=[True, False])
        top_df = top_df.drop(columns=["abs_delta"]).head(30)
        top_df["FDR q"] = top_df["FDR q"].apply(fmt_pval)
        html += report.df_to_html(top_df, caption="Top 30 significant contrasts (grouped by Direction, sorted by |delta-beta|)")
    else:
        html += report.text("No significant contrasts at FDR < 0.05.")

    # Multi-pair confusion matrix: genes significant in ≥2 pairs
    n_multi = sum(1 for d in gene_pair_directions.values() if len(d) >= 2)
    log.info(f"Multi-pair genes (significant in ≥2 pairs): {n_multi}")
    multi_genes = {
        gene: directions
        for gene, directions in gene_pair_directions.items()
        if len(directions) >= 2
    }
    if multi_genes:
        # Get R² for ranking
        r2_vals = np.asarray(gene_reg.get("r_squared_degree1", [])) if gene_reg else np.array([])
        reg_feat_names = list(gene_reg.get("feature_names", [])) if gene_reg else []
        feat_to_r2 = dict(zip(reg_feat_names, r2_vals)) if len(r2_vals) == len(reg_feat_names) else {}

        matrix_rows = []
        for gene_name, gene_dir in sorted(multi_genes.items()):
            row = {"Gene": gene_name, "R2": feat_to_r2.get(gene_name, np.nan)}
            for pair_label in all_pair_labels:
                row[pair_label] = gene_dir.get(pair_label, "")
            row["N pairs"] = len(gene_dir)
            matrix_rows.append(row)
        conf_df = pd.DataFrame(matrix_rows).set_index("Gene")
        # Sort by R² descending (most explanatory genes first)
        conf_df = conf_df.sort_values("R2", ascending=False)
        conf_df["R2"] = conf_df["R2"].apply(lambda x: f"{x:.4f}" if np.isfinite(x) else "N/A")
        html += report.df_to_html(
            conf_df,
            caption=(
                f"Multi-pair contrast direction matrix ({len(multi_genes)} genes significant in ≥2 pairs; "
                "direction = which archetype has higher expression)"
            ),
        )
        # UpSet plot: which gene sets overlap across archetype pairs
        try:
            sig_genes_per_pair = {}
            for pair_label in all_pair_labels:
                sig_genes_per_pair[pair_label] = set(
                    g for g, dirs in gene_pair_directions.items() if pair_label in dirs
                )
            # Also get genes sig in only 1 pair (not yet in gene_pair_directions)
            all_sig_genes_set = set(gene_pair_directions.keys())
            for pair in pairs:
                pair_key = str(tuple(pair) if isinstance(pair, list) else pair)
                pvals = np.asarray(contrast_result["pvalues_fdr"][pair_key])
                j, k = pair if isinstance(pair, (list, tuple)) else (pair[0], pair[1])
                pair_label = f"A{j+1}-A{k+1}"
                for feat_idx in range(len(feature_names)):
                    if pvals[feat_idx] < 0.05:
                        all_sig_genes_set.add(feature_names[feat_idx])
                        if pair_label not in sig_genes_per_pair:
                            sig_genes_per_pair[pair_label] = set()
                        sig_genes_per_pair[pair_label].add(feature_names[feat_idx])

            # Build intersection counts for UpSet
            from itertools import combinations as _combinations
            pair_labels_list = [pl for pl in all_pair_labels if pl in sig_genes_per_pair]
            n_pairs = len(pair_labels_list)

            # If too many pairs for full UpSet (2^n subsets), use top pairs by sig gene count
            if n_pairs > 12:
                pair_sizes = [(pl, len(sig_genes_per_pair.get(pl, set()))) for pl in pair_labels_list]
                pair_sizes.sort(key=lambda x: x[1], reverse=True)
                pair_labels_list = [ps[0] for ps in pair_sizes[:12]]
                n_pairs = len(pair_labels_list)
            if n_pairs >= 2:
                intersection_counts = []
                for r in range(1, n_pairs + 1):
                    for combo in _combinations(range(n_pairs), r):
                        in_sets = [sig_genes_per_pair[pair_labels_list[i]] for i in combo]
                        out_sets = [sig_genes_per_pair[pair_labels_list[i]]
                                    for i in range(n_pairs) if i not in combo]
                        members = set.intersection(*in_sets) if in_sets else set()
                        for os in out_sets:
                            members = members - os
                        if len(members) > 0:
                            intersection_counts.append((combo, len(members)))

                # Sort by count descending
                intersection_counts.sort(key=lambda x: x[1], reverse=True)
                top_intersections = intersection_counts[:15]  # top 15

                if top_intersections:
                    fig_upset, (ax_bar, ax_dots) = plt.subplots(
                        2, 1, figsize=(max(8, len(top_intersections) * 0.7), 5),
                        gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

                    x = range(len(top_intersections))
                    counts = [c for _, c in top_intersections]
                    ax_bar.bar(x, counts, color="#0072B2", edgecolor="none")
                    ax_bar.set_ylabel("Gene count")
                    ax_bar.set_title("UpSet: Gene overlap across archetype pairs")
                    ax_bar.spines[["top", "right"]].set_visible(False)
                    for xi, ci in zip(x, counts):
                        ax_bar.text(xi, ci + 0.5, str(ci), ha="center", va="bottom", fontsize=8)

                    # Dot matrix
                    for xi, (combo, _) in enumerate(top_intersections):
                        for yi in range(n_pairs):
                            if yi in combo:
                                ax_dots.plot(xi, yi, 'o', color="#0072B2", markersize=8)
                            else:
                                ax_dots.plot(xi, yi, 'o', color="#CCCCCC", markersize=5)
                        # Connect dots in combo
                        ys = [y for y in combo]
                        if len(ys) > 1:
                            ax_dots.plot([xi] * len(ys), ys, '-', color="#0072B2", linewidth=2)

                    ax_dots.set_yticks(range(n_pairs))
                    ax_dots.set_yticklabels(pair_labels_list, fontsize=9)
                    ax_dots.set_xlim(-0.5, len(top_intersections) - 0.5)
                    ax_dots.set_ylim(-0.5, n_pairs - 0.5)
                    ax_dots.invert_yaxis()
                    ax_dots.spines[["top", "right", "bottom"]].set_visible(False)
                    ax_dots.set_xticks([])

                    fig_upset.tight_layout()
                    html += report.fig_to_img(fig_upset,
                        caption="UpSet plot: gene overlap across archetype contrast pairs")
                    plt.close("all")
        except Exception as e:
            html += error_html(f"UpSet plot failed: {e}")
            plt.close("all")

    else:
        html += report.text("No genes were significant across multiple archetype pairs.")

    # Overlap with 2nd-degree interaction terms
    try:
        int_coefs = gene_reg.get("interaction_coefficients")
        int_pairs_reg = gene_reg.get("interaction_pairs", [])
        int_fdr = gene_reg.get("interaction_pvalues_fdr")
        if int_coefs is not None and int_fdr is not None:
            int_fdr = np.asarray(int_fdr)
            reg_feat_names = list(gene_reg["feature_names"])

            # Significant interaction features (any pair)
            sig_int_mask = (int_fdr < 0.05).any(axis=1)
            sig_int_features = set(
                reg_feat_names[i] for i in range(len(reg_feat_names)) if sig_int_mask[i]
            )

            # Significant contrast features (any pair)
            sig_contrast_features = set()
            for pair in pairs:
                pair_key = str(tuple(pair) if isinstance(pair, list) else pair)
                pvals = np.asarray(contrast_result["pvalues_fdr"][pair_key])
                for i, p in enumerate(pvals):
                    if p < 0.05:
                        sig_contrast_features.add(feature_names[i])

            overlap = sig_int_features & sig_contrast_features
            html += metric_grid([
                metric_card(len(sig_int_features), "Sig interaction features"),
                metric_card(len(sig_contrast_features), "Sig contrast features"),
                metric_card(len(overlap), "Overlap"),
            ])
            if overlap:
                html += report.text(
                    f"Overlap features: {', '.join(sorted(overlap)[:20])}"
                    + ("..." if len(overlap) > 20 else "")
                )
    except Exception as e:
        html += error_html(f"Overlap analysis failed: {e}")

    # Pathway contrasts (if pathway regression results exist)
    has_pathway_reg = "peach_simplex_regression_pathways" in adata.uns
    if has_pathway_reg:
        try:
            log.info("Computing Wald contrasts for pathways...")
            pathway_contrast = pc.tl.archetype_contrasts(adata, feature_type="pathways")
            pw_pairs = pathway_contrast.get("pairs", [])
            pw_feature_names = list(pathway_contrast.get("feature_names", []))
            html += report.text(
                f"Pathway Wald contrasts: {len(pw_pairs)} pairs across {len(pw_feature_names)} pathway features."
            )
            # Summary table
            pw_summary_rows = []
            for pair in pw_pairs:
                pair_key = str(tuple(pair) if isinstance(pair, list) else pair)
                pw_pvals = np.asarray(pathway_contrast["pvalues_fdr"][pair_key])
                pw_delta = np.asarray(pathway_contrast["delta_beta"][pair_key])
                j, k = pair if isinstance(pair, (list, tuple)) else (pair[0], pair[1])
                pw_summary_rows.append({
                    "Pair": f"A{j+1} vs A{k+1}",
                    "N significant (FDR<0.05)": int((pw_pvals < 0.05).sum()),
                    "Mean |delta-beta|": f"{np.abs(pw_delta).mean():.4f}",
                    "Max |delta-beta|": f"{np.abs(pw_delta).max():.4f}",
                })
            if pw_summary_rows:
                html += report.df_to_html(
                    pd.DataFrame(pw_summary_rows),
                    caption="Pairwise Wald contrast summary — pathways",
                )
            # Top pathway contrasts grouped by Direction
            pw_top_rows = []
            for pair in pw_pairs:
                pair_key = str(tuple(pair) if isinstance(pair, list) else pair)
                pw_pvals = np.asarray(pathway_contrast["pvalues_fdr"][pair_key])
                pw_delta = np.asarray(pathway_contrast["delta_beta"][pair_key])
                j, k = pair if isinstance(pair, (list, tuple)) else (pair[0], pair[1])
                for feat_idx in range(len(pw_feature_names)):
                    if pw_pvals[feat_idx] < 0.05:
                        direction = f"Up in A{j+1}" if pw_delta[feat_idx] > 0 else f"Up in A{k+1}"
                        pw_top_rows.append({
                            "Pathway": pw_feature_names[feat_idx],
                            "Pair": f"A{j+1} vs A{k+1}",
                            "Direction": direction,
                            "delta_beta": pw_delta[feat_idx],
                            "FDR q": pw_pvals[feat_idx],
                        })
            if pw_top_rows:
                pw_top_df = pd.DataFrame(pw_top_rows)
                pw_top_df["abs_delta"] = pw_top_df["delta_beta"].abs()
                pw_top_df = pw_top_df.sort_values(
                    ["Direction", "abs_delta"], ascending=[True, False]
                )
                pw_top_df = pw_top_df.drop(columns=["abs_delta"]).head(30)
                pw_top_df["FDR q"] = pw_top_df["FDR q"].apply(fmt_pval)
                html += report.df_to_html(
                    pw_top_df,
                    caption="Top 30 significant pathway contrasts (grouped by direction, sorted by |delta-beta|)",
                )
            else:
                html += report.text("No significant pathway contrasts at FDR < 0.05.")

            # Pathway UpSet plot: overlap across archetype pairs
            try:
                pw_sig_per_pair = {}
                for pair in pw_pairs:
                    pair_key = str(tuple(pair) if isinstance(pair, list) else pair)
                    pvals_arr = np.asarray(pathway_contrast["pvalues_fdr"][pair_key])
                    j, k = pair if isinstance(pair, (list, tuple)) else (pair[0], pair[1])
                    pl_label = f"A{j+1}-A{k+1}"
                    pw_sig_per_pair[pl_label] = set(
                        pw_feature_names[i] for i in range(len(pw_feature_names))
                        if pvals_arr[i] < 0.05
                    )
                pw_pair_labels = [pl for pl in pw_sig_per_pair if pw_sig_per_pair[pl]]
                n_pw_pairs = len(pw_pair_labels)
                if n_pw_pairs >= 2 and n_pw_pairs <= 12:
                    from itertools import combinations as _combinations
                    pw_intersections = []
                    for r in range(1, n_pw_pairs + 1):
                        for combo in _combinations(range(n_pw_pairs), r):
                            in_sets = [pw_sig_per_pair[pw_pair_labels[i]] for i in combo]
                            out_sets = [pw_sig_per_pair[pw_pair_labels[i]]
                                        for i in range(n_pw_pairs) if i not in combo]
                            members = set.intersection(*in_sets) if in_sets else set()
                            for os in out_sets:
                                members -= os
                            if members:
                                pw_intersections.append((combo, len(members)))
                    pw_intersections.sort(key=lambda x: x[1], reverse=True)
                    pw_top_int = pw_intersections[:15]
                    if pw_top_int:
                        fig_upset_pw, (ax_bar_pw, ax_dots_pw) = plt.subplots(
                            2, 1, figsize=(max(8, len(pw_top_int) * 0.7), 5),
                            gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
                        x_pw = range(len(pw_top_int))
                        counts_pw = [c for _, c in pw_top_int]
                        ax_bar_pw.bar(x_pw, counts_pw, color="#D55E00", edgecolor="none")
                        ax_bar_pw.set_ylabel("Pathway count")
                        ax_bar_pw.set_title("UpSet: Pathway overlap across archetype pairs")
                        ax_bar_pw.spines[["top", "right"]].set_visible(False)
                        for xi, ci in zip(x_pw, counts_pw):
                            ax_bar_pw.text(xi, ci + 0.3, str(ci), ha="center", va="bottom", fontsize=8)
                        for xi, (combo, _) in enumerate(pw_top_int):
                            for yi in range(n_pw_pairs):
                                if yi in combo:
                                    ax_dots_pw.plot(xi, yi, 'o', color="#D55E00", markersize=8)
                                else:
                                    ax_dots_pw.plot(xi, yi, 'o', color="#CCCCCC", markersize=5)
                            ys = list(combo)
                            if len(ys) > 1:
                                ax_dots_pw.plot([xi] * len(ys), ys, '-', color="#D55E00", linewidth=2)
                        ax_dots_pw.set_yticks(range(n_pw_pairs))
                        ax_dots_pw.set_yticklabels(pw_pair_labels, fontsize=9)
                        ax_dots_pw.set_xlim(-0.5, len(pw_top_int) - 0.5)
                        ax_dots_pw.set_ylim(-0.5, n_pw_pairs - 0.5)
                        ax_dots_pw.invert_yaxis()
                        ax_dots_pw.spines[["top", "right", "bottom"]].set_visible(False)
                        ax_dots_pw.set_xticks([])
                        fig_upset_pw.tight_layout()
                        html += report.fig_to_img(fig_upset_pw,
                            caption="UpSet plot: pathway overlap across archetype contrast pairs")
                        plt.close("all")
            except Exception as e:
                html += error_html(f"Pathway UpSet plot failed: {e}")
                plt.close("all")
        except Exception as e:
            html += error_html(f"Pathway contrasts failed: {e}")
    else:
        html += report.text(
            "Pathway Wald contrasts skipped: no pathway regression results found "
            "(peach_simplex_regression_pathways not in adata.uns)."
        )

    report.add_section("Wald Contrasts", html, step_num=5)
    return contrast_result


def step6_within_fit_comparisons(adata, report):
    """Step 6: Flow-based within-fit pairwise comparison + feature similarity."""
    import peach as pc

    html = ""

    # Get archetype labels
    if "archetypes" not in adata.obs.columns:
        html += error_html("No 'archetypes' column — skipping within-fit comparisons.")
        report.add_section("Within-Fit Comparisons", html, step_num=6)
        return None

    arch_labels = sorted([a for a in adata.obs["archetypes"].unique()
                          if a != "no_archetype" and not pd.isna(a)])
    K = len(arch_labels)
    html += report.text(f"Pairwise flow comparison for {K} archetypes: {', '.join(arch_labels)}")

    # Pairwise flow_within between archetype groups
    flow_pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            flow_pairs.append((arch_labels[i], arch_labels[j]))

    flow_rows = []
    flow_raw = {}  # Keep raw flow results (with source_mask) for permutation tests
    for src_label, tgt_label in flow_pairs:
        n_src = int((adata.obs["archetypes"] == src_label).sum())
        n_tgt = int((adata.obs["archetypes"] == tgt_label).sum())

        if n_src < 50 or n_tgt < 50:
            flow_rows.append({
                "Source": display_arch(src_label), "Target": display_arch(tgt_label),
                "N_source": n_src, "N_target": n_tgt,
                "MMD_before": float("nan"), "MMD_after": float("nan"),
                "MMD_reduction": float("nan"),
                "Status": "Skipped (<50 cells)",
            })
            continue

        log.info(f"  Flow: {src_label} -> {tgt_label} ({n_src} -> {n_tgt} cells)...")
        try:
            fr = pc.tl.flow_within(
                adata,
                source={"archetypes": src_label},
                target={"archetypes": tgt_label},
                n_epochs=200,
                hidden_dims=(128, 128),
                batch_size=min(128, min(n_src, n_tgt) // 2),
                return_model=False,
                name=f"wf_{src_label}_to_{tgt_label}",
            )
            mmd_b = fr["mmd_before"]
            mmd_a = fr["mmd_after"]
            reduction = 1 - mmd_a / max(mmd_b, 1e-10)
            flow_raw[f"{src_label}_to_{tgt_label}"] = fr  # keep raw result with source_mask
            flow_rows.append({
                "Source": display_arch(src_label), "Target": display_arch(tgt_label),
                "N_source": n_src, "N_target": n_tgt,
                "MMD_before": round(mmd_b, 4), "MMD_after": round(mmd_a, 4),
                "MMD_reduction": round(reduction, 4),
                "Status": "OK",
            })
        except Exception as e:
            log.warning(f"Flow {src_label}->{tgt_label} failed: {e}")
            flow_rows.append({
                "Source": display_arch(src_label), "Target": display_arch(tgt_label),
                "N_source": n_src, "N_target": n_tgt,
                "MMD_before": float("nan"), "MMD_after": float("nan"),
                "MMD_reduction": float("nan"),
                "Status": f"Failed: {e}",
            })

    flow_df = pd.DataFrame(flow_rows)
    html += report.df_to_html(flow_df, caption="Pairwise flow-based archetype comparison")

    # Build K×K dissimilarity matrix (1 - MMD reduction)
    # Higher MMD reduction = easier to transport = more similar; so 1 - MMD_reduction = dissimilarity
    # DataFrame Source/Target use display labels (A1, A2, ...) via display_arch()
    arch_short = [f"A{i+1}" for i in range(K)]
    sim_matrix = np.full((K, K), np.nan)
    for _, row in flow_df.iterrows():
        if row["Status"] == "OK":
            # Source/Target are display labels (A1, A2, ...) — look up by display index
            try:
                i = arch_short.index(row["Source"])
                j = arch_short.index(row["Target"])
            except ValueError:
                continue
            val = row["MMD_reduction"]
            sim_matrix[i, j] = 1.0 - val  # dissimilarity: higher = more different
            sim_matrix[j, i] = 1.0 - val
    np.fill_diagonal(sim_matrix, 0.0)  # self-dissimilarity = 0

    # Heatmap
    try:
        import plotly.graph_objects as go
        fig = go.Figure(data=go.Heatmap(
            z=sim_matrix, x=arch_short, y=arch_short,
            colorscale="Blues",
            text=np.where(np.isnan(sim_matrix), "", np.round(sim_matrix, 3).astype(str)),
            texttemplate="%{text}", textfont_size=10,
        ))
        fig.update_layout(title="Flow-based archetype dissimilarity (1 - MMD reduction)",
                          xaxis_title="Target", yaxis_title="Source",
                          width=500, height=450)
        html += safe_plotly_html(report, fig,
                                 "Higher values = more phenotypically distinct archetype pairs")
    except Exception as e:
        html += error_html(f"Flow similarity heatmap failed: {e}")

    # Permutation null: use flow_significance() with proper label-permuted retraining
    # Run on ALL pairs with successful flow fits
    sig_results = []
    for src_label, tgt_label in flow_pairs:
        # Use raw flow result (has source_mask, target_mask, pca_key)
        fr_stored = flow_raw.get(f"{src_label}_to_{tgt_label}")
        if fr_stored is None:
            continue
        try:
            log.info(f"  Permutation test: {src_label} -> {tgt_label} (20 perms)...")
            sig = pc.tl.flow_significance(
                adata, fr_stored,
                n_permutations=20,
                n_epochs_per_perm=100,
            )
            sig_results.append({
                "Pair": f"{display_arch(src_label)}→{display_arch(tgt_label)}",
                "Observed improvement": round(sig["observed_stat"], 4),
                "Null mean": round(np.mean(sig["null_distribution"]), 4),
                "Null std": round(np.std(sig["null_distribution"]), 4),
                "p-value": f"{sig['p_value']:.3f}",
            })
        except Exception as e:
            log.warning(f"Permutation test {src_label}->{tgt_label} failed: {e}")

    if sig_results:
        sig_df = pd.DataFrame(sig_results)
        html += report.df_to_html(sig_df,
            caption="Permutation test: observed MMD improvement vs label-permuted null")
        html += report.text(
            "Null model: pool source + target cells, randomly re-assign to groups of same size, "
            "retrain flow model, measure MMD improvement. "
            "p-value = fraction of null improvements ≥ observed improvement. "
            "Significant p-values confirm archetype assignment captures real phenotype structure.")
    else:
        html += report.text("<em>No flow results available for permutation testing.</em>")

    # Feature similarity (Spearman on regression coefficients) — keep existing
    log.info("Computing within-fit feature similarity...")
    try:
        sim_result = pc.tl.archetype_feature_similarity(adata)
        n_sig_feat = sim_result.get("n_significant_features", "?")
        html += report.text(f"Spearman \u03c1 computed on {n_sig_feat} FDR-significant (q<0.05) "
                            "vertex \u03b2 coefficients from simplex regression. Only features significant "
                            "in at least one archetype are included.")

        fig_sim = pc.pl.feature_similarity_heatmap(adata, show=False)
        html += safe_plotly_html(report, fig_sim, "Feature similarity (Spearman \u03c1) heatmap")
    except Exception as e:
        html += error_html(f"Feature similarity failed: {e}")

    # Radar plots — placed here so Spearman similarity can inform spoke angles
    try:
        fig_radar = pc.pl.archetype_radar(adata, top_n=8, order_by_similarity=True, show=False)
        html += safe_plotly_html(report, fig_radar,
            "Archetype radar (top 8 features, spokes ordered by coefficient similarity)")
    except Exception as e:
        html += error_html(f"Archetype radar failed: {e}")
    if "pathway_scores" in adata.obsm:
        try:
            fig_pw_radar = pc.pl.archetype_radar(
                adata, top_n=8, feature_type="pathways",
                order_by_similarity=True, show=False)
            html += safe_plotly_html(report, fig_pw_radar,
                "Pathway radar (similarity-ordered)")
        except Exception as e:
            html += error_html(f"Pathway radar failed: {e}")

    report.add_section("Within-Fit Comparisons (Flow + Feature Similarity)", html, step_num=6)
    return flow_df


def step6b_diversity_metrics(adata, report):
    """Step 6b: Alpha and beta diversity of gene expression per archetype."""
    from scipy.stats import entropy as shannon_entropy
    from scipy.spatial.distance import braycurtis, pdist, squareform

    html = ""

    if "archetypes" not in adata.obs.columns:
        html += error_html("No 'archetypes' column — skipping diversity metrics.")
        report.add_section("Diversity Metrics", html, step_num="6b")
        return

    arch_labels = sorted([a for a in adata.obs["archetypes"].unique()
                          if a != "no_archetype" and not pd.isna(a)])

    # Get expression matrix (dense)
    X = np.asarray(adata.X.todense()) if hasattr(adata.X, "todense") else np.asarray(adata.X)
    # Shift to non-negative for entropy (X is log-normalized, may have negatives)
    X_shifted = X - X.min(axis=1, keepdims=True) + 1e-10

    # Alpha diversity: Shannon entropy per cell, averaged per archetype
    html += report.text(
        "<b>Alpha diversity (Shannon entropy)</b>: Computed per-cell on the full gene expression "
        f"matrix ({adata.shape[1]} genes, log-normalized, shifted to non-negative). "
        "Higher entropy = more uniform expression across genes = less specialized.")
    alpha_rows = []
    for label in arch_labels:
        mask = (adata.obs["archetypes"] == label).values
        cells = X_shifted[mask]
        per_cell_ent = np.array([shannon_entropy(c / c.sum()) for c in cells])
        alpha_rows.append({
            "Archetype": display_arch(label),
            "N cells": int(mask.sum()),
            "Mean Shannon H": f"{per_cell_ent.mean():.4f}",
            "Median Shannon H": f"{np.median(per_cell_ent):.4f}",
            "Std Shannon H": f"{per_cell_ent.std():.4f}",
        })
    html += report.df_to_html(pd.DataFrame(alpha_rows),
                              caption="Alpha diversity: Shannon entropy of expression per archetype")

    # Null baseline: shuffle archetype labels
    try:
        rng = np.random.RandomState(42)
        null_entropies = []
        for _ in range(5):
            shuffled_labels = rng.permutation(adata.obs["archetypes"].values)
            for label in arch_labels:
                mask = shuffled_labels == label
                cells = X_shifted[mask]
                per_cell_ent = np.array([shannon_entropy(c / c.sum()) for c in cells])
                null_entropies.append(per_cell_ent.mean())
        null_mean = np.mean(null_entropies)
        null_std = np.std(null_entropies)
        # Max possible Shannon entropy for this gene count
        max_ent = np.log(X_shifted.shape[1])
        # Observed mean across all archetypes
        obs_entropies = [float(r["Mean Shannon H"]) for r in alpha_rows]
        obs_mean_ent = np.mean(obs_entropies)
        html += metric_grid([
            metric_card(f"{obs_mean_ent:.4f}", "Observed mean H"),
            metric_card(f"{null_mean:.4f}", f"Shuffled null mean H"),
            metric_card(f"{max_ent:.4f}", f"Max H (uniform, {X_shifted.shape[1]} genes)"),
        ])
        html += report.text(
            f"Null: shuffled archetype labels (5 repeats): mean H = {null_mean:.4f} ± {null_std:.4f}. "
            f"Max possible H = {max_ent:.4f} (uniform over {X_shifted.shape[1]} genes). "
            "Observed deviating from shuffled null = archetype assignment captures diversity structure.")
    except Exception as e:
        html += report.text(f"<em>Null baseline computation failed: {e}</em>")

    # Beta diversity: Bray-Curtis between archetype mean profiles
    html += report.text(
        "<b>Beta diversity (Bray-Curtis)</b>: Dissimilarity between archetype mean expression "
        f"profiles ({adata.shape[1]} genes). Range 0-1; 0 = identical profiles, "
        "1 = completely different.")
    mean_profiles = []
    for label in arch_labels:
        mask = (adata.obs["archetypes"] == label).values
        mean_profiles.append(X_shifted[mask].mean(axis=0))
    mean_profiles = np.array(mean_profiles)

    bc_matrix = squareform(pdist(mean_profiles, metric="braycurtis"))
    bc_df = pd.DataFrame(bc_matrix,
                         index=[f"A{i+1}" for i in range(len(arch_labels))],
                         columns=[f"A{i+1}" for i in range(len(arch_labels))])
    html += report.df_to_html(bc_df.round(4),
                              caption="Beta diversity: Bray-Curtis between archetype mean profiles")

    # Bray-Curtis heatmap
    try:
        import plotly.graph_objects as go
        arch_short = [f"A{i+1}" for i in range(len(arch_labels))]
        fig = go.Figure(data=go.Heatmap(
            z=bc_matrix, x=arch_short, y=arch_short,
            colorscale="Viridis",
            text=np.round(bc_matrix, 3).astype(str),
            texttemplate="%{text}", textfont_size=10,
        ))
        fig.update_layout(title="Bray-Curtis dissimilarity between archetypes",
                          width=500, height=450)
        html += safe_plotly_html(report, fig,
                                 "Bray-Curtis (0=identical, 1=completely different)")
    except Exception as e:
        html += error_html(f"Bray-Curtis heatmap failed: {e}")

    # Gene set diversity (if pathway scores available)
    if "pathway_scores" in adata.obsm:
        pw_scores = np.asarray(adata.obsm["pathway_scores"])
        pw_names = adata.uns.get("pathway_scores_pathways", [])
        html += report.text(
            f"<b>Pathway score diversity</b>: Variance of {len(pw_names)} pathway scores "
            "per archetype. Higher variance = more heterogeneous pathway activity within the archetype.")
        pw_alpha_rows = []
        for label in arch_labels:
            mask = (adata.obs["archetypes"] == label).values
            pw_cells = pw_scores[mask]
            pw_var = pw_cells.var(axis=0).mean()
            pw_alpha_rows.append({
                "Archetype": display_arch(label),
                "Mean pathway score variance": f"{pw_var:.4f}",
            })
        html += report.df_to_html(pd.DataFrame(pw_alpha_rows),
                                  caption="Pathway score diversity per archetype")

    # Weight entropy: how committed are cells to one archetype
    weights = np.asarray(adata.obsm.get("cell_archetype_weights", np.array([])))
    if weights.size > 0:
        K = weights.shape[1] if weights.ndim == 2 else None
        K_label = str(K) if K is not None else "?"
        max_entropy_str = f"{np.log(K):.2f}" if K is not None else "?"
        html += report.text(
            f"<b>Weight entropy</b>: -\u2211 w_i \u00b7 log(w_i) across {K_label} archetypes per cell. "
            f"Max possible = {max_entropy_str} (uniform weights). "
            "Higher = cell is distributed across archetypes; lower = strongly committed to one.")
        w_clipped = np.clip(weights, 1e-10, 1.0)
        weight_entropy = -np.sum(w_clipped * np.log(w_clipped), axis=1)
        entropy_rows = []
        for label in arch_labels:
            mask = (adata.obs["archetypes"] == label).values
            ent = weight_entropy[mask]
            entropy_rows.append({
                "Archetype": display_arch(label),
                "Mean weight entropy": f"{ent.mean():.4f}",
                "Median": f"{np.median(ent):.4f}",
            })
        html += report.df_to_html(pd.DataFrame(entropy_rows),
                                  caption="Archetype weight entropy (higher = less committed)")

        # Global weight entropy for comparison
        if weights.size > 0:
            global_ent = weight_entropy.mean()
            K_val = weights.shape[1] if weights.ndim == 2 else 0
            max_ent = np.log(K_val) if K_val > 0 else 0
            html += report.text(
                f"<em>Global mean weight entropy: {global_ent:.4f} (max possible: {max_ent:.2f} for uniform weights). "
                f"Per-archetype values below global mean indicate more committed cells.</em>")

    # Per-condition diversity
    for condition_col in ["cell_type_short"]:
        if condition_col not in adata.obs.columns:
            continue
        html += report.text(
            f"<b>Per-{condition_col} diversity</b>: Shannon entropy computed on "
            f"{adata.shape[1]} genes per cell, averaged across cells in each {condition_col} group.")
        groups = sorted(adata.obs[condition_col].unique())
        cond_rows = []
        for grp in groups:
            mask = (adata.obs[condition_col] == grp).values
            cells = X_shifted[mask]
            per_cell_ent = np.array([shannon_entropy(c / c.sum()) for c in cells])
            cond_rows.append({
                "Group": str(grp), "N cells": int(mask.sum()),
                "Mean Shannon H": f"{per_cell_ent.mean():.4f}",
            })
        html += report.df_to_html(pd.DataFrame(cond_rows),
                                  caption=f"Expression diversity by {condition_col}")

    report.add_section("Diversity Metrics (Prototype)", html, step_num="6b")


def step7_driver_regression(adata, report):
    """Step 7: Driver regression (features predict archetype weights).

    Runs gene driver regression for concordance with simplex regression (same
    feature space). If pathway scores are available, also runs a separate
    pathway driver regression and reports it independently.
    """
    import peach as pc

    html = ""

    # -----------------------------------------------------------------------
    # PRIMARY: Gene driver regression (same feature space as simplex regression)
    # This is the one used for concordance comparison.
    # -----------------------------------------------------------------------
    log.info("Running gene driver regression (for concordance with simplex regression)...")
    try:
        # Pre-select top variable genes to avoid singular design matrix
        # (n_cells must exceed n_features for OLS)
        X_dense = _dense_X(adata)
        gene_var = np.var(X_dense, axis=0)
        K_arch = adata.obsm.get("cell_archetype_weights", np.empty((0, 5))).shape[1]
        max_feat = min(X_dense.shape[1], max(50, adata.n_obs // 4))
        top_gene_idx = np.argsort(gene_var)[-max_feat:]
        gene_feat_matrix = X_dense[:, top_gene_idx]
        gene_feat_names = [adata.var_names[i] for i in top_gene_idx]
        log.info(f"  Gene driver: {max_feat}/{X_dense.shape[1]} genes by variance "
                 f"(n_cells={adata.n_obs})")
        html += report.text(f"Gene driver regression: {max_feat}/{X_dense.shape[1]} genes "
                            f"selected by variance (n_cells={adata.n_obs}, need n_cells > n_features).")

        gene_driver_result = pc.tl.archetype_driver_regression(
            adata,
            feature_matrix=gene_feat_matrix,
            feature_names=gene_feat_names,
            max_degree=1,
            n_bootstrap=500,
            robust_se=True,
        )
        driver_result = gene_driver_result  # used in concordance section below
        feat_label = "genes"
    except Exception as e:
        gene_driver_result = None
        driver_result = None
        feat_label = "genes"
        html += error_html(f"Gene driver regression failed: {e}")

    # -----------------------------------------------------------------------
    # SECONDARY: Pathway driver regression (separate analysis, not used for concordance)
    # -----------------------------------------------------------------------
    has_pathways = "pathway_scores" in adata.obsm
    if has_pathways:
        pw_scores = adata.obsm.get("pathway_scores")
        if pw_scores is not None:
            pw_var = np.var(pw_scores, axis=0)
            K_arch = adata.obsm.get("cell_archetype_weights", np.empty((0, 5))).shape[1]
            n_keep = min(pw_scores.shape[1], max(20, K_arch * 4))
            top_pw_idx = np.argsort(pw_var)[-n_keep:]
            pw_names_all = adata.uns.get("pathway_scores_pathways",
                                         [f"pw_{i}" for i in range(pw_scores.shape[1])])
            pw_feat_names = [pw_names_all[i] for i in top_pw_idx]
            pw_feat_matrix = pw_scores[:, top_pw_idx]
            log.info(f"  Pathway driver regression: keeping {n_keep}/{pw_scores.shape[1]} "
                     f"pathways by variance (K={K_arch})")
            html += report.text(f"Pathway driver regression (separate analysis): "
                                f"{n_keep}/{pw_scores.shape[1]} pathways retained "
                                f"(top by variance, max(20, K×4)={n_keep}).")
            try:
                pathway_driver_result = pc.tl.archetype_driver_regression(
                    adata,
                    feature_matrix=pw_feat_matrix,
                    feature_names=pw_feat_names,
                    max_degree=1,
                    n_bootstrap=500,
                    robust_se=True,
                )
                pw_r2 = np.asarray(pathway_driver_result["r_squared"])
                html += report.text(
                    f"Pathway driver regression R² (ILR components): "
                    f"mean = {pw_r2.mean():.4f}, "
                    f"range [{pw_r2.min():.4f}, {pw_r2.max():.4f}]. "
                    "(Note: concordance comparison below uses gene driver regression.)")

                # Pathway driver coefficient heatmap
                try:
                    pw_main_coefs = np.asarray(pathway_driver_result["main_coefficients"])
                    n_show_pw = min(20, len(pw_feat_names))
                    pw_mean_abs = np.abs(pw_main_coefs).mean(axis=0)
                    pw_top_idx = np.argsort(pw_mean_abs)[-n_show_pw:][::-1]
                    fig_pw, ax_pw = plt.subplots(
                        figsize=(max(6, pw_main_coefs.shape[0] * 1.2),
                                 max(4, n_show_pw * 0.4)))
                    im = ax_pw.imshow(pw_main_coefs[:, pw_top_idx].T, aspect="auto", cmap="RdBu_r")
                    ax_pw.set_xticks(range(pw_main_coefs.shape[0]))
                    ax_pw.set_xticklabels([f"A{k+1}" for k in range(pw_main_coefs.shape[0])])
                    ax_pw.set_yticks(range(n_show_pw))
                    ax_pw.set_yticklabels([pw_feat_names[i] for i in pw_top_idx], fontsize=7)
                    plt.colorbar(im, ax=ax_pw, label="Coefficient", shrink=0.6)
                    ax_pw.set_title("Pathway driver coefficients")
                    fig_pw.tight_layout()
                    html += report.fig_to_img(fig_pw,
                        caption="Pathway driver coefficients (top pathways by mean |coef|)")
                    plt.close("all")
                except Exception as e_pwh:
                    html += error_html(f"Pathway coefficient heatmap failed: {e_pwh}")
                    plt.close("all")

            except Exception as e:
                html += error_html(f"Pathway driver regression failed: {e}")

    if driver_result is None:
        report.add_section("Driver Regression", html, step_num=7)
        return

    log.info(f"Reporting gene driver regression results (feature_matrix={feat_label}, degree=1)...")
    try:
        r2_vals = np.asarray(driver_result["r_squared"])
        K_minus_1 = len(r2_vals)
        main_coefs = np.asarray(driver_result["main_coefficients"])  # [K, n_feat]
        feat_names = list(driver_result.get("feature_names", []))

        K_arch = K_minus_1 + 1
        html += metric_grid([
            metric_card(f"{r2_vals.mean():.4f}", "Mean R-squared (ILR components)"),
            metric_card(f"{K_minus_1} (K−1 for K={K_arch})", "ILR dimensions"),
            metric_card(len(feat_names), "Features"),
        ])
        html += report.text(
            f"Driver regression uses {K_minus_1} ILR components (K−1 = {K_arch}−1). "
            "ILR transform maps K-simplex archetype weights to K−1 unconstrained coordinates. "
            "Coefficients are back-transformed to simplex space for interpretation.")

        # R-squared per ILR component
        r2_rows = [{"ILR component": i + 1, "R-squared": f"{r2_vals[i]:.4f}"} for i in range(K_minus_1)]
        html += report.df_to_html(pd.DataFrame(r2_rows), caption="Driver regression R-squared per ILR component")

        # Coefficient heatmap (matplotlib)
        try:
            n_show = min(30, len(feat_names))
            mean_abs = np.abs(main_coefs).mean(axis=0)
            top_idx = np.argsort(mean_abs)[-n_show:][::-1]
            fig, ax = plt.subplots(figsize=(max(6, main_coefs.shape[0] * 1.2), max(5, n_show * 0.35)))
            im = ax.imshow(main_coefs[:, top_idx].T, aspect="auto", cmap="RdBu_r")
            ax.set_xticks(range(main_coefs.shape[0]))
            ax.set_xticklabels([f"A{k+1}" for k in range(main_coefs.shape[0])])
            ax.set_yticks(range(n_show))
            ax.set_yticklabels([feat_names[i] for i in top_idx], fontsize=8)
            plt.colorbar(im, ax=ax, label="Coefficient", shrink=0.6)
            ax.set_title("Driver regression coefficients (simplex space)")
            fig.tight_layout()
            html += report.fig_to_img(fig, caption="Driver coefficients heatmap (top features by mean |coef|)")
            plt.close("all")
        except Exception as e:
            html += error_html(f"Driver coefficient heatmap failed: {e}")

        # Concordance with simplex regression
        try:
            simplex_reg = adata.uns.get("peach_simplex_regression") or adata.uns.get("peach_simplex_regression_genes")
            if simplex_reg is not None:
                simplex_coefs = np.asarray(simplex_reg["vertex_coefficients"])
                simplex_names = list(simplex_reg["feature_names"])

                # Match feature names
                shared = set(feat_names) & set(simplex_names)
                if len(shared) > 10:
                    from scipy.stats import spearmanr
                    # Mean absolute coefficient per feature in each
                    driver_mean = np.abs(main_coefs).mean(axis=0)
                    simplex_mean = np.abs(simplex_coefs).max(axis=1)

                    d_idx = [feat_names.index(f) for f in shared]
                    s_idx = [simplex_names.index(f) for f in shared]

                    r, p = spearmanr(driver_mean[d_idx], simplex_mean[s_idx])
                    html += report.text(
                        f"Concordance between driver and simplex regression (mean |coef|): "
                        f"Spearman r = {r:.3f}, p = {fmt_pval(p)} ({len(shared)} shared features)"
                    )

                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.scatter(driver_mean[d_idx], simplex_mean[s_idx],
                               alpha=0.3, s=8, color="#0072B2")
                    ax.set_xlabel("Driver regression mean |coef|")
                    ax.set_ylabel("Simplex regression max |beta|")
                    ax.set_title(f"Concordance: r={r:.3f}")
                    ax.spines[["top", "right"]].set_visible(False)
                    fig.tight_layout()
                    html += report.fig_to_img(fig, caption="Driver vs simplex regression concordance")
                    plt.close("all")
        except Exception as e:
            html += error_html(f"Concordance analysis failed: {e}")

        # Feature-level overlap: top 50 from each method
        try:
            gene_reg = adata.uns.get("peach_simplex_regression") or adata.uns.get("peach_simplex_regression_genes")
            if gene_reg is not None and "feature_names" in gene_reg:
                simplex_names = list(gene_reg["feature_names"])
                simplex_r2 = np.asarray(gene_reg["r_squared_degree1"])
                top50_simplex = set(np.array(simplex_names)[np.argsort(simplex_r2)[-50:]])

                driver_names = list(driver_result.get("feature_names", []))
                driver_max_beta = np.abs(np.asarray(driver_result["main_coefficients"])).max(axis=0)
                top50_driver = set(np.array(driver_names)[np.argsort(driver_max_beta)[-50:]])

                shared = top50_simplex & top50_driver
                simplex_only = top50_simplex - top50_driver
                driver_only = top50_driver - top50_simplex

                html += report.text(
                    "<b>Method comparison</b>: 'Simplex-only' = features ranked highly by simplex regression R\u00b2 "
                    "but NOT in the driver regression top list. 'Driver-only' = features with high ILR driver "
                    "coefficients but low simplex R\u00b2. 'Shared' = top features in both methods. "
                    "Concordance indicates robust archetype-feature associations.")
                html += metric_grid([
                    metric_card(len(shared), "Shared top-50"),
                    metric_card(len(simplex_only), "Simplex-only"),
                    metric_card(len(driver_only), "Driver-only"),
                ])
                # Filter shared features for significance (FDR < 0.05 in simplex regression)
                simplex_fdr = np.asarray(gene_reg.get("f_pvalue_fdr", np.ones(len(simplex_r2))))
                sig_shared = set()
                for g in shared:
                    if g in simplex_names:
                        gi = simplex_names.index(g)
                        if simplex_fdr[gi] < 0.05:
                            sig_shared.add(g)

                overlap_rows = [{"Feature": g, "Source": "shared (FDR<0.05)"} for g in sorted(sig_shared)[:20]]
                overlap_rows += [{"Feature": g, "Source": "shared (not sig)"} for g in sorted(shared - sig_shared)[:5]]
                overlap_rows += [{"Feature": g, "Source": "simplex-only"} for g in sorted(simplex_only)[:10]]
                overlap_rows += [{"Feature": g, "Source": "driver-only"} for g in sorted(driver_only)[:10]]
                html += report.df_to_html(pd.DataFrame(overlap_rows),
                    caption=f"Feature overlap: simplex R² top 50 vs driver |β| top 50 ({len(sig_shared)} shared + FDR-significant)")

                # Shared features: coefficient comparison table
                if shared:
                    # Build lookup dicts: feature -> scalar summary
                    simplex_r2_lookup = dict(zip(simplex_names, simplex_r2.tolist()))
                    driver_max_beta = np.abs(np.asarray(driver_result["main_coefficients"])).max(axis=0)
                    driver_beta_lookup = dict(zip(driver_names, driver_max_beta.tolist()))
                    shared_rows = []
                    for feat in sorted(shared)[:30]:
                        row = {"Feature": feat}
                        if feat in simplex_r2_lookup:
                            row["Simplex R\u00b2"] = f"{simplex_r2_lookup[feat]:.3f}"
                        if feat in driver_beta_lookup:
                            row["Driver max|\u03b2|"] = f"{driver_beta_lookup[feat]:.3f}"
                        shared_rows.append(row)
                    html += report.df_to_html(
                        pd.DataFrame(shared_rows),
                        caption="Shared top features: simplex R\u00b2 vs driver regression max|\u03b2|"
                    )
        except Exception as e:
            html += error_html(f"Feature overlap analysis failed: {e}")

    except Exception as e:
        html += error_html(f"Driver regression failed: {e}")

    report.add_section("Driver Regression", html, step_num=7)


def step8_mixture_models(adata, report):
    """Step 8: Simplex density decomposition (Dirichlet)."""
    import peach as pc

    html = ""

    log.info("Running simplex decomposition (Dirichlet)...")
    try:
        decomp_result = pc.tl.feature_simplex_decomposition(
            adata,
            model_type="dirichlet",
            model_selection="bic_elbow",
            n_initializations=20,
            stability_threshold=0.7,
        )

        n_opt = decomp_result.get("n_components_optimal", "?")
        n_stable = decomp_result.get("n_components_stable", "?")
        bic = decomp_result.get("bic_values", [])
        stab_scores = decomp_result.get("component_stability_scores", [])

        html += metric_grid([
            metric_card(n_opt, "Optimal components (BIC)"),
            metric_card(n_stable, "Stable components"),
            metric_card(decomp_result.get("model_type", "?"), "Model type"),
        ])

        # BIC curve
        try:
            fig_bic = pc.pl.gmm_bic_curve(adata, show=False)
            html += safe_plotly_html(report, fig_bic, "BIC curve vs number of components")
        except Exception as e:
            html += error_html(f"BIC curve plot failed: {e}")

        # Component stability
        try:
            fig_stab = pc.pl.component_stability(adata, show=False)
            html += safe_plotly_html(report, fig_stab, "Component stability scores")
        except Exception as e:
            html += error_html(f"Component stability plot failed: {e}")
        html += report.text("Stability: fraction of n_initializations (20 random starts) where this "
                            "component is recovered via Hungarian matching of component centroids. "
                            "Higher = more robust to initialization. Threshold: 0.7.")

        # Component scatter
        try:
            fig_scat = pc.pl.component_scatter(adata, show=False)
            html += safe_plotly_html(report, fig_scat, "PCA scatter colored by component")
        except Exception as e:
            html += error_html(f"Component scatter failed: {e}")

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
    """Step 9: Component regression, component MMD, component enrichment."""
    import peach as pc

    html = ""

    # Component regression
    log.info("Running component regression...")
    try:
        comp_reg = pc.tl.component_regression(adata)
        n_comps = comp_reg.get("n_components", 0)
        comp_regs = comp_reg.get("component_regs", {})

        html += metric_grid([
            metric_card(n_comps, "GMM components"),
            metric_card(len(comp_regs), "Components with regression"),
        ])

        # Per-component R2 summary
        comp_rows = []
        for c, reg in comp_regs.items():
            r2 = np.asarray(reg["r_squared_degree1"])
            f_fdr = np.asarray(reg.get("f_pvalue_fdr", np.ones(len(r2))))
            n_sig = int((f_fdr < 0.05).sum())
            comp_rows.append({
                "Component": c,
                "N features": len(r2),
                "Mean R-squared": f"{r2.mean():.4f}",
                "N significant": n_sig,
                "Top feature": reg["feature_names"][np.argmax(r2)] if len(reg["feature_names"]) > 0 else "N/A",
                "Top R-squared": f"{r2.max():.4f}",
            })
        if comp_rows:
            html += report.df_to_html(pd.DataFrame(comp_rows), caption="Per-component regression summary")

        # Component heatmap
        try:
            fig_heat = pc.pl.component_heatmap(adata, show=False)
            html += safe_plotly_html(report, fig_heat, "Component feature profiles heatmap")
        except Exception as e:
            html += error_html(f"Component heatmap failed: {e}")

        # Pathway characterization per component
        # Note: component_regression ignores feature_type in its current implementation
        # (always uses adata.X). We call feature_simplex_regression directly per component
        # with feature_matrix="pathway_scores" to correctly use pathway features.
        if "pathway_scores" in adata.obsm:
            try:
                log.info("  Component pathway characterization...")
                from peach.tl.feature_regression import feature_simplex_regression as _fsr
                gmm_for_pw = adata.uns.get("peach_gmm")
                pw_comp_regs = {}
                if gmm_for_pw is not None:
                    _assignments = np.asarray(gmm_for_pw["component_assignments"])
                    _n_stable = gmm_for_pw["n_components_stable"]
                    for _c in range(_n_stable):
                        _mask = _assignments == _c
                        if _mask.sum() < 20:
                            continue
                        _adata_sub = adata[_mask].copy()
                        _reg = _fsr(
                            _adata_sub,
                            feature_matrix="pathway_scores",
                            n_bootstrap=100,
                            robust_se=True,
                            store_to_adata=False,
                            store_residuals=False,
                        )
                        pw_comp_regs[_c] = _reg
                pw_comp = {"component_regs": pw_comp_regs, "n_components": _n_stable if gmm_for_pw else 0}
                if pw_comp_regs:
                    html += report.text("Pathway characterization per component:")
                    pw_comp_rows = []
                    for c, reg in pw_comp_regs.items():
                        r2_pw = np.asarray(reg["r_squared_degree1"])
                        fn_pw = reg.get("feature_names", [])
                        if len(fn_pw) == 0:
                            continue
                        top_idx_pw = np.argmax(r2_pw)
                        pw_comp_rows.append({
                            "Component": c,
                            "N pathways": len(r2_pw),
                            "Mean R\u00b2": f"{r2_pw.mean():.4f}",
                            "Top pathway": fn_pw[top_idx_pw],
                            "Top R\u00b2": f"{r2_pw[top_idx_pw]:.4f}",
                        })
                    if pw_comp_rows:
                        html += report.df_to_html(
                            pd.DataFrame(pw_comp_rows),
                            caption="Per-component pathway regression summary"
                        )
            except Exception as e:
                html += error_html(f"Pathway component characterization failed: {e}")

    except Exception as e:
        html += error_html(f"Component regression failed: {e}")

    # Component-archetype similarity: raw metrics
    gmm_raw = adata.uns.get("peach_gmm")
    if gmm_raw is not None:
        try:
            from sklearn.metrics import adjusted_rand_score

            assignments_raw = np.asarray(gmm_raw["component_assignments"])
            weights_raw = adata.obsm.get("cell_archetype_weights")
            arch_labels_raw = adata.obs.get("archetypes")

            ari_value = None
            majority_acc = None
            mean_mmd = None

            # ARI between GMM components and hard archetype assignments
            if arch_labels_raw is not None:
                arch_int = pd.Categorical(arch_labels_raw).codes
                valid_mask = (assignments_raw >= 0) & (arch_int >= 0)
                if valid_mask.sum() > 10:
                    ari_value = adjusted_rand_score(
                        arch_int[valid_mask], assignments_raw[valid_mask]
                    )

            # Majority-vote accuracy (for each component, how pure is its archetype?)
            if weights_raw is not None and gmm_raw.get("n_components_stable", 0) > 0:
                n_stable_raw = int(gmm_raw["n_components_stable"])
                weights_arr = np.asarray(weights_raw)
                hard_arch = np.argmax(weights_arr, axis=1)
                vote_accs = []
                for c in range(n_stable_raw):
                    comp_mask = assignments_raw == c
                    if comp_mask.sum() < 5:
                        continue
                    arch_in_comp = hard_arch[comp_mask]
                    majority_frac = np.bincount(arch_in_comp).max() / comp_mask.sum()
                    vote_accs.append(majority_frac)
                if vote_accs:
                    majority_acc = float(np.mean(vote_accs))

            # Mean pairwise MMD in archetype weight space
            if weights_raw is not None and gmm_raw.get("n_components_stable", 0) > 1:
                n_stable_raw = int(gmm_raw["n_components_stable"])
                weights_arr = np.asarray(weights_raw)
                mmd_vals = []
                for ca in range(n_stable_raw):
                    for cb in range(ca + 1, n_stable_raw):
                        xa = weights_arr[assignments_raw == ca]
                        xb = weights_arr[assignments_raw == cb]
                        if len(xa) < 5 or len(xb) < 5:
                            continue
                        # Unbiased MMD^2 with RBF kernel (bandwidth = median heuristic)
                        n_sub = min(500, len(xa), len(xb))
                        rng_mmd = np.random.default_rng(42)
                        xa_sub = xa[rng_mmd.choice(len(xa), n_sub, replace=False)]
                        xb_sub = xb[rng_mmd.choice(len(xb), n_sub, replace=False)]
                        all_x = np.vstack([xa_sub, xb_sub])
                        sq_dists = np.sum((all_x[:, None] - all_x[None, :]) ** 2, axis=-1)
                        bw = np.median(sq_dists[sq_dists > 0]) or 1.0
                        K_aa = np.exp(-sq_dists[:n_sub, :n_sub] / bw)
                        K_bb = np.exp(-sq_dists[n_sub:, n_sub:] / bw)
                        K_ab = np.exp(-sq_dists[:n_sub, n_sub:] / bw)
                        np.fill_diagonal(K_aa, 0)
                        np.fill_diagonal(K_bb, 0)
                        mmd2 = (K_aa.sum() / (n_sub * (n_sub - 1))
                                + K_bb.sum() / (n_sub * (n_sub - 1))
                                - 2 * K_ab.mean())
                        mmd_vals.append(float(np.sqrt(max(mmd2, 0))))
                if mmd_vals:
                    mean_mmd = float(np.mean(mmd_vals))

            raw_metrics = []
            if ari_value is not None:
                raw_metrics.append({"Metric": "Adjusted Rand Index (GMM vs archetype)",
                                    "Value": f"{ari_value:.4f}"})
            if majority_acc is not None:
                raw_metrics.append({"Metric": "Mean majority-vote archetype purity",
                                    "Value": f"{majority_acc:.4f}"})
            if mean_mmd is not None:
                raw_metrics.append({"Metric": "Mean pairwise MMD (archetype weight space)",
                                    "Value": f"{mean_mmd:.4f}"})
            if raw_metrics:
                html += report.df_to_html(
                    pd.DataFrame(raw_metrics),
                    caption="Component-archetype similarity: raw metrics"
                )

            # Per-archetype ARI breakout: within each archetype's cells, how well
            # do GMM components agree with intra-archetype structure?
            if arch_labels_raw is not None and weights_raw is not None:
                from sklearn.metrics import adjusted_rand_score as _ari
                unique_archs = sorted(set(arch_labels_raw.dropna()))
                per_arch_rows = []
                for arch_label in unique_archs:
                    arch_mask = (arch_labels_raw == arch_label).values
                    n_cells_arch = int(arch_mask.sum())
                    if n_cells_arch < 20:
                        continue
                    gmm_in_arch = assignments_raw[arch_mask]
                    arch_codes_in = pd.Categorical(arch_labels_raw[arch_mask]).codes
                    # Number of distinct GMM components in this archetype
                    n_comps_in = len(set(gmm_in_arch[gmm_in_arch >= 0]))
                    # Majority purity: what fraction of cells have the most common component?
                    if len(gmm_in_arch[gmm_in_arch >= 0]) > 0:
                        comp_counts = np.bincount(gmm_in_arch[gmm_in_arch >= 0])
                        purity = comp_counts.max() / comp_counts.sum()
                    else:
                        purity = 0
                    per_arch_rows.append({
                        "Archetype": display_arch(arch_label),
                        "N cells": n_cells_arch,
                        "N GMM components": n_comps_in,
                        "Majority purity": f"{purity:.3f}",
                    })
                if per_arch_rows:
                    html += report.df_to_html(
                        pd.DataFrame(per_arch_rows),
                        caption="Per-archetype GMM component breakdown (higher purity = archetype maps to fewer components)"
                    )
        except Exception as e:
            html += error_html(f"Component-archetype raw metrics failed: {e}")

    # Component conditional associations
    gmm_data = adata.uns.get("peach_gmm")
    if gmm_data is not None:
        try:
            assignments = np.asarray(gmm_data["component_assignments"])
            adata.obs["gmm_component"] = pd.Categorical(
                [f"comp_{int(a)}" if a >= 0 else "unassigned" for a in assignments]
            )

            for col in ["cell_type_short"]:
                if col not in adata.obs.columns:
                    continue
                log.info(f"Component conditional associations: {col}...")
                try:
                    comp_cond = pc.tl.conditional_associations(
                        adata, obs_column=col, obs_key="gmm_component", verbose=False
                    )
                    sig_comp = comp_cond[comp_cond.get("significant", comp_cond["fdr_pvalue"] < 0.05) == True]
                    html += report.text(
                        f"Component x {col}: {len(sig_comp)} significant / {len(comp_cond)} tests"
                    )
                    display_cols = ["archetype", "condition", "odds_ratio", "fdr_pvalue", "significant"]
                    display_cols = [c for c in display_cols if c in comp_cond.columns]
                    html += report.df_to_html(comp_cond[display_cols],
                                              caption=f"Component enrichment: {col}")
                except Exception as e:
                    html += error_html(f"Component conditional ({col}) failed: {e}")

            # Clean up temporary column
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

    report.add_section("Component Characterization", html, step_num=9)


# ============================================================================
# Per-subset helpers (shared by steps 10-17)
# ============================================================================


def _subset_adata(adata, mask, label: str):
    """Subset adata by boolean mask, preserving obsm/varm references."""
    sub = adata[mask].copy()
    log.info(f"  Subset '{label}': {sub.shape[0]} cells")
    return sub


def run_subset_model(adata_sub, K: int, hidden_dims: list, label: str) -> dict | None:
    """Train model on a subset adata. Returns training results dict or None."""
    import peach as pc

    if adata_sub.shape[0] < MIN_CELLS_MODEL:
        log.warning(f"  {label}: only {adata_sub.shape[0]} cells, skipping model training.")
        return None

    pc.pp.prepare_training(adata_sub, batch_size=min(128, adata_sub.shape[0] // 4))
    results = pc.tl.train_archetypal(
        adata_sub,
        n_archetypes=K,
        n_epochs=150,
        hidden_dims=hidden_dims,
        early_stopping=True,
        early_stopping_patience=12,
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

    for col in ["cell_type_short"]:
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
    """Build a summary row dict from a regression result."""
    r2 = np.asarray(gene_reg["r_squared_degree1"])
    f_fdr = np.asarray(gene_reg.get("f_pvalue_fdr", np.ones(len(r2))))
    return {
        "Subset": label,
        "N features": len(r2),
        "N sig (FDR<0.05)": int((f_fdr < 0.05).sum()),
        "Mean R2": f"{r2.mean():.4f}",
        "Max R2": f"{r2.max():.4f}",
    }


def _make_dose_pairs():
    """Return the three canonical dose pairs."""
    return [("Base", "PD1"), ("PD1", "RTPD1"), ("Base", "RTPD1")]


# ============================================================================
# Steps 10-17: Per-subset analyses
# ============================================================================


def step10_per_dose_models(adata, report):
    """Step 10: Per-condition models. Uses 'treatment' if available, else 'Study'."""
    import peach as pc

    html = ""
    dose_adatas = {}

    # Determine condition column: treatment > Study > skip
    if "treatment" in adata.obs.columns:
        condition_col = "treatment"
    elif "Study" in adata.obs.columns:
        condition_col = "Study"
    else:
        html += error_html("No 'treatment' or 'Study' column -- skipping per-condition models.")
        report.add_section("Per-Condition Models", html, step_num=10)
        return dose_adatas

    doses = sorted(adata.obs[condition_col].unique())
    html += report.text(f"Condition column: <b>{condition_col}</b> ({len(doses)} groups: "
                        f"{', '.join(str(d) for d in doses)})")
    # Smaller search grid for subsets
    K_range = [3, 4, 5, 6, 7]
    hidden_opts = [[64, 128], [128, 256]]

    summary_rows = []
    for dose in doses:
        mask = (adata.obs[condition_col] == dose).values
        sub = _subset_adata(adata, mask, dose)

        if sub.shape[0] < MIN_CELLS_MODEL:
            html += error_html(f"{dose}: {sub.shape[0]} cells < {MIN_CELLS_MODEL}, skipped.")
            continue

        # Hyperparameter search
        log.info(f"  {dose}: hyperparameter search...")
        try:
            pc.pp.prepare_training(sub, batch_size=min(128, sub.shape[0] // 4))
            cv = pc.tl.hyperparameter_search(
                sub,
                n_archetypes_range=K_range,
                hidden_dims_options=hidden_opts,
                inflation_factor_range=[1.0, 1.5, 2.0],
                cv_folds=3,
                max_epochs_cv=12,
                subsample_fraction=0.8,
            )
            ranked = cv.rank_by_metric("archetype_r2")
            best = ranked[0]
            # Guard against -inf R² (all configs failed)
            if best["metric_value"] == float("-inf") or np.isnan(best["metric_value"]):
                html += error_html(f"{dose}: all CV configs returned -inf R², using K=4 fallback.")
                best_K, best_hd = 4, [128, 256]
            else:
                best_hp = best["hyperparameters"]
                best_K = best_hp["n_archetypes"]
                best_hd = best_hp.get("hidden_dims", [128, 256])

            # CV search QC
            try:
                cv_rows = []
                for r in ranked[:5]:
                    hp = r["hyperparameters"]
                    cv_rows.append({
                        "K": hp["n_archetypes"],
                        "Hidden": str(hp.get("hidden_dims", "?")),
                        "Mean R\u00b2": f"{r['metric_value']:.4f}",
                    })
                html += report.df_to_html(pd.DataFrame(cv_rows),
                                          caption=f"{dose}: top 5 CV configurations")
            except Exception:
                pass
        except Exception as e:
            html += error_html(f"{dose}: CV search failed ({e}), using K=5 fallback.")
            best_K, best_hd = 5, [128, 256]

        # Train final model
        log.info(f"  {dose}: training K={best_K}, hidden={best_hd}...")
        res = run_subset_model(sub, K=best_K, hidden_dims=best_hd, label=dose)
        if res is not None:
            dose_adatas[dose] = sub
            r2 = res.get("final_archetype_r2", float("nan"))
            summary_rows.append({
                "Dose": dose, "N cells": sub.shape[0],
                "Best K": best_K, "Hidden": str(best_hd),
                "Final R2": f"{r2:.4f}" if isinstance(r2, float) else str(r2),
            })

    if summary_rows:
        html += report.df_to_html(pd.DataFrame(summary_rows),
                                  caption=f"Per-condition model comparison ({condition_col})")

    # Side-by-side archetypal space scatters (limit to 6 panels max)
    n_dose = len(dose_adatas)
    if n_dose > 0:
        n_show = min(n_dose, 6)
        try:
            fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 4))
            if n_show == 1:
                axes = [axes]
            for ax, (dose, sub) in zip(axes, list(dose_adatas.items())[:n_show]):
                coords = sub.obsm.get("archetype_coordinates")
                if coords is not None and coords.shape[1] >= 2:
                    ax.scatter(coords[:, 0], coords[:, 1], s=2, alpha=0.3, c="#0072B2")
                    # Mark archetype vertices and expand limits
                    arch_pos = sub.uns.get("archetype_coordinates")
                    if arch_pos is not None:
                        arch_pos = np.asarray(arch_pos)
                        if arch_pos.ndim == 2 and arch_pos.shape[1] >= 2:
                            ax.scatter(arch_pos[:, 0], arch_pos[:, 1], s=80, c="red",
                                       marker="^", zorder=5, edgecolors="black", linewidth=0.5)
                            all_pts = np.vstack([coords[:, :2], arch_pos[:, :2]])
                            margin = 0.1 * np.ptp(all_pts, axis=0)
                            margin = np.maximum(margin, 0.1)
                            ax.set_xlim(all_pts[:, 0].min() - margin[0], all_pts[:, 0].max() + margin[0])
                            ax.set_ylim(all_pts[:, 1].min() - margin[1], all_pts[:, 1].max() + margin[1])
                    ax.set_title(f"{dose} (K={sub.obsm['cell_archetype_weights'].shape[1]})")
                    ax.set_xlabel("Arch coord 1")
                    ax.set_ylabel("Arch coord 2")
                    ax.spines[["top", "right"]].set_visible(False)
            fig.suptitle(f"Per-condition archetypal space ({condition_col})", y=1.02)
            fig.tight_layout()
            html += report.fig_to_img(fig, caption=f"Per-condition archetypal space ({condition_col})")
            plt.close("all")
        except Exception as e:
            html += error_html(f"Per-condition scatter failed: {e}")
            plt.close("all")

    report.add_section("Per-Condition Models", html, step_num=10)
    return dose_adatas


def step11_per_dose_regression(dose_adatas, report):
    """Step 11: Per-dose regression, pattern classification, feature stability."""
    import peach as pc
    from scipy.stats import spearmanr

    html = ""

    if not dose_adatas:
        html += error_html("No per-dose adatas available, skipping.")
        report.add_section("Per-Dose Regression", html, step_num=11)
        return

    summary_rows = []
    dose_regs = {}
    for dose, sub in dose_adatas.items():
        log.info(f"  {dose}: regression + patterns...")
        try:
            reg = run_subset_regression(sub, label=dose)
            if reg is not None:
                dose_regs[dose] = reg
                summary_rows.append(_reg_summary_row(reg, dose))
        except Exception as e:
            html += error_html(f"{dose} regression failed: {e}")

    if summary_rows:
        html += report.df_to_html(pd.DataFrame(summary_rows),
                                  caption="Per-dose regression summary")

    # Per-dose exclusive dotplots
    for dose, sub in dose_adatas.items():
        if "peach_simplex_regression" not in sub.uns:
            continue
        try:
            fig_dot = pc.pl.archetype_regression_dotplot(sub, top_n=10, exclusive_only=True, show=False)
            html += report.plotly_to_div(fig_dot,
                                         caption=f"Exclusive gene features: {dose}, top 10 per archetype (ranked by |β|)")
        except Exception as e:
            html += error_html(f"{dose}: dotplot failed: {e}")

    # Cross-dose feature stability (Spearman of R2 vectors)
    dose_labels = list(dose_regs.keys())
    if len(dose_labels) >= 2:
        try:
            stab_rows = []
            for i in range(len(dose_labels)):
                for j in range(i + 1, len(dose_labels)):
                    d1, d2 = dose_labels[i], dose_labels[j]
                    r1 = dose_regs[d1]
                    r2_reg = dose_regs[d2]
                    names1 = set(r1["feature_names"])
                    names2 = set(r2_reg["feature_names"])
                    shared = sorted(names1 & names2)
                    if len(shared) < 10:
                        continue
                    idx1 = [list(r1["feature_names"]).index(f) for f in shared]
                    idx2 = [list(r2_reg["feature_names"]).index(f) for f in shared]
                    r2_v1 = np.asarray(r1["r_squared_degree1"])[idx1]
                    r2_v2 = np.asarray(r2_reg["r_squared_degree1"])[idx2]
                    rho, p = spearmanr(r2_v1, r2_v2)
                    stab_rows.append({
                        "Pair": f"{d1} vs {d2}",
                        "Shared features": len(shared),
                        "Spearman rho": f"{rho:.3f}",
                        "p-value": fmt_pval(p),
                    })
            if stab_rows:
                html += report.df_to_html(pd.DataFrame(stab_rows),
                                          caption="Cross-dose feature stability (R-squared Spearman)")

            # Scatter of R2 for first pair
            if len(dose_labels) >= 2:
                d1, d2 = dose_labels[0], dose_labels[1]
                r1 = dose_regs[d1]
                r2_reg = dose_regs[d2]
                shared = sorted(set(r1["feature_names"]) & set(r2_reg["feature_names"]))
                if len(shared) >= 10:
                    idx1 = [list(r1["feature_names"]).index(f) for f in shared]
                    idx2 = [list(r2_reg["feature_names"]).index(f) for f in shared]
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.scatter(
                        np.asarray(r1["r_squared_degree1"])[idx1],
                        np.asarray(r2_reg["r_squared_degree1"])[idx2],
                        s=6, alpha=0.3, c="#0072B2",
                    )
                    ax.set_xlabel(f"R2 ({d1})")
                    ax.set_ylabel(f"R2 ({d2})")
                    ax.set_title("Cross-dose R2 stability")
                    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
                    ax.spines[["top", "right"]].set_visible(False)
                    fig.tight_layout()
                    html += report.fig_to_img(fig, caption=f"R2 scatter: {d1} vs {d2}")
                    plt.close("all")
        except Exception as e:
            html += error_html(f"Feature stability analysis failed: {e}")
            plt.close("all")

    report.add_section("Per-Dose Regression", html, step_num=11)


def step12_between_dose_flow(adata, dose_adatas, report):
    """Step 12: Between-condition flow (soft assignment), MMD, feature similarity."""
    import peach as pc

    html = ""
    flow_results = {}

    # Determine flow pairs: treatment > cell_type_short
    if "treatment" in adata.obs.columns:
        dose_pairs = _make_dose_pairs()
        obs_key = "treatment"
    elif "cell_type_short" in adata.obs.columns:
        # Use biological transition pairs (cell type) instead of dose pairs
        dose_pairs = FLOW_PAIRS
        obs_key = "cell_type_short"
        html += report.text(
            "Using biological transition pairs (cell type): "
            + ", ".join(f"{s}→{t}" for s, t in dose_pairs)
        )
    else:
        html += error_html("No 'treatment' or 'cell_type_short' column -- skipping.")
        report.add_section("Between-Condition Flow", html, step_num=12)
        return flow_results

    available_doses = set(adata.obs[obs_key].unique())

    for src, tgt in dose_pairs:
        if src not in available_doses or tgt not in available_doses:
            html += report.text(f"Skipping {src}->{tgt}: group not present.")
            continue

        n_src = int((adata.obs[obs_key] == src).sum())
        n_tgt = int((adata.obs[obs_key] == tgt).sum())
        if n_src < MIN_CELLS_FLOW or n_tgt < MIN_CELLS_FLOW:
            html += error_html(f"{src}->{tgt}: insufficient cells ({n_src}, {n_tgt}), skipping.")
            continue

        pair_key = f"{src}_to_{tgt}"
        log.info(f"  Flow: {src} -> {tgt} ({n_src} -> {n_tgt} cells)...")
        html += f"<h4>Flow: {pair_key.replace('_to_', ' → ')}</h4>"
        try:
            fr = pc.tl.flow_within(
                adata,
                source={obs_key: src},
                target={obs_key: tgt},
                n_epochs=600,
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

    # Flow significance permutation test (expensive — retrains n_permutations models)
    html += "<h3>Flow Significance (Permutation Test)</h3>"
    html += report.text(
        "Permutation test: shuffles source/target labels and retrains flow models to build "
        "a null distribution of MMD improvement. Tests whether the observed flow transport "
        "is significantly better than random label assignment. This is expensive (retrains "
        "n_permutations=50 models per pair)."
    )
    for pair_key, fr in flow_results.items():
        _label = pair_key.replace("_to_", " → ")
        log.info(f"  Flow significance: {pair_key} (200 permutations, may take a while)...")
        try:
            sig_result = pc.tl.flow_significance(
                adata, fr,
                n_permutations=200,
                n_epochs_per_perm=200,
                hidden_dims=(128, 128, 128),
                batch_size=256,
                solver_method="euler",
            )
            p_val = sig_result["p_value"]
            obs_stat = sig_result["observed_stat"]
            null_dist = sig_result["null_distribution"]

            html += metric_grid([
                metric_card(_label, "Flow pair"),
                metric_card(f"{obs_stat:.4f}", "Observed MMD improvement"),
                metric_card(f"{np.mean(null_dist):.4f}", "Null mean improvement"),
                metric_card(fmt_pval(p_val), "p-value"),
            ])

            # Null distribution histogram
            try:
                fig_sig, ax_sig = plt.subplots(figsize=(6, 3))
                ax_sig.hist(null_dist, bins=20, color="#999999", alpha=0.7,
                            edgecolor="none", label="Null distribution")
                ax_sig.axvline(obs_stat, color="#D55E00", linewidth=2,
                               linestyle="--", label=f"Observed ({obs_stat:.4f})")
                ax_sig.set_xlabel("MMD improvement")
                ax_sig.set_ylabel("Count")
                ax_sig.set_title(f"Flow significance: {_label} (p={fmt_pval(p_val)})")
                ax_sig.legend()
                ax_sig.spines[["top", "right"]].set_visible(False)
                fig_sig.tight_layout()
                html += report.fig_to_img(fig_sig, caption=f"Flow significance null distribution: {_label}")
                plt.close("all")
            except Exception:
                plt.close("all")

        except Exception as e:
            html += error_html(f"Flow significance ({pair_key}) failed: {e}")

    # Soft assignment heatmaps
    for pair_key, fr in flow_results.items():
        html += f"<h4>Flow: {pair_key.replace('_to_', ' → ')}</h4>"
        try:
            fig_sa = pc.pl.soft_assignment_heatmap(adata, fr, show=False)
            html += safe_plotly_html(report, fig_sa,
                                    f"Soft assignment heatmap (Rows: source archetypes, Columns: target archetypes): {pair_key}")
            html += report.text(
                "Soft assignment correspondence: source cells are transported via the learned flow field, "
                "then matched to their k-nearest target neighbors in PCA space. Entry [i,j] shows "
                "the fraction of source archetype i's transported mass landing near target archetype j. "
                "Uniform rows = diffuse transitions; concentrated rows = canalization to specific target archetypes.")
        except Exception as e:
            html += error_html(f"Soft assignment heatmap ({pair_key}) failed: {e}")

    # Feature similarity between dose pairs (if per-dose adatas available)
    if len(dose_adatas) >= 2:
        dose_labels = list(dose_adatas.keys())
        sim_rows = []
        for i in range(len(dose_labels)):
            for j in range(i + 1, len(dose_labels)):
                d1, d2 = dose_labels[i], dose_labels[j]
                sub1, sub2 = dose_adatas[d1], dose_adatas[d2]
                try:
                    sim = pc.tl.archetype_feature_similarity(sub1, adata_b=sub2)
                    spear = np.asarray(sim["spearman_matrix"])
                    sim_rows.append({
                        "Pair": f"{d1} vs {d2}",
                        "Mean Spearman": f"{spear.mean():.3f}",
                        "Max Spearman": f"{spear.max():.3f}",
                    })
                except Exception as e:
                    html += error_html(f"Feature sim {d1} vs {d2} failed: {e}")

        if sim_rows:
            html += report.df_to_html(pd.DataFrame(sim_rows),
                                      caption="Cross-dose feature similarity (archetype Spearman)")
            html += report.text("Feature similarity: Spearman ρ computed on FDR-significant (q<0.05) "
                                "vertex β coefficients from simplex regression across dose-pair fits. "
                                "Higher ρ = archetypes share similar gene-archetype associations between doses.")

    report.add_section("Between-Condition Flow", html, step_num=12)
    return flow_results


def step12b_soft_assignment_interpretation(adata, flow_results, report):
    """Step 12b: Soft assignment correspondence — rate archetype relatedness across conditions."""
    import peach as pc

    html = ""
    top_pairs = {}  # pair_key -> list of (src_arch, tgt_arch, correspondence_score)

    if not flow_results:
        html += report.text("No flow results from step 12 — skipping soft assignment interpretation.")
        report.add_section("Soft Assignment Interpretation", html, step_num="12b")
        return top_pairs

    html += report.text(
        "Soft assignment correspondence: for each flow pair, transported source cells are matched to "
        "their k-nearest target neighbors. The K_source × K_target correspondence matrix shows how "
        "archetype membership maps between conditions. High off-diagonal entries indicate archetype "
        "reorganization along the flow.")

    for pair_key, fr in flow_results.items():
        _label = pair_key.replace("_to_", " → ")
        html += f"<h4>{_label}: Archetype correspondence</h4>"
        try:
            # Compute soft assignment heatmap — this builds the correspondence matrix internally
            fig_sa = pc.pl.soft_assignment_heatmap(adata, fr, show=False)
            html += safe_plotly_html(report, fig_sa, f"Correspondence heatmap: {_label}")

            # Extract the correspondence matrix for ranking
            # Recompute it explicitly for the ranking
            from scipy.spatial import cKDTree
            weights = adata.obsm.get("cell_archetype_weights")
            if weights is None:
                html += error_html(f"{pair_key}: no archetype weights, skipping ranking.")
                continue

            source_mask = fr["source_mask"]
            target_mask = fr["target_mask"]
            transported = fr["transported"]
            pca_key = fr.get("pca_key", "X_pca")

            target_pca = adata.obsm[pca_key][target_mask]
            weights_source = weights[source_mask]
            weights_target = weights[target_mask]
            K_src = weights_source.shape[1]
            K_tgt = weights_target.shape[1]

            tree = cKDTree(target_pca)
            nn_dists, nn_idx = tree.query(transported, k=min(10, len(target_pca)))

            correspondence = np.zeros((K_src, K_tgt))
            for i in range(len(transported)):
                w_src = weights_source[i]
                w_tgt = weights_target[nn_idx[i]].mean(axis=0)
                correspondence += np.outer(w_src, w_tgt)
            correspondence /= len(transported)
            # Row-normalize
            row_sums = correspondence.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            corr_norm = correspondence / row_sums

            # Rank all (src_arch, tgt_arch) pairs by RAW correspondence mass
            # (not row-normalized — normalization amplifies noise for low-mass archetypes)
            pair_scores = []
            for i in range(K_src):
                for j in range(K_tgt):
                    pair_scores.append({
                        "Source archetype": f"A{i+1}",
                        "Target archetype": f"A{j+1}",
                        "Row-normalized": f"{corr_norm[i, j]:.3f}",
                        "Raw mass": f"{correspondence[i, j]:.4f}",
                        "raw_value": correspondence[i, j],
                    })
            pair_scores.sort(key=lambda x: x["raw_value"], reverse=True)

            # Store top pairs for step 12c — filter by minimum raw mass
            # to avoid selecting archetype pairs with negligible cell membership
            MIN_RAW_MASS = 0.01
            top_candidates = [
                ps for ps in pair_scores if ps["raw_value"] > MIN_RAW_MASS
            ]
            top_k = min(3, len(top_candidates))
            top_pairs[pair_key] = [
                (int(ps["Source archetype"][1:]) - 1,
                 int(ps["Target archetype"][1:]) - 1,
                 ps["raw_value"])
                for ps in top_candidates[:top_k]
            ]

            # Display table
            display_scores = [{k: v for k, v in ps.items() if k != "raw_value"}
                              for ps in pair_scores]
            html += report.df_to_html(
                pd.DataFrame(display_scores),
                caption=f"Archetype correspondence ranking: {_label} (row-normalized)")

            # Highlight top pairs
            top_desc = ", ".join(
                f"A{s+1}→A{t+1} ({c:.2f})"
                for s, t, c in top_pairs[pair_key]
            )
            html += report.text(f"<b>Top correspondence pairs for zoomed flow:</b> {top_desc}")

            # Sankey diagram: archetype correspondence flow
            try:
                import plotly.graph_objects as go
                src_labels = [f"{src_label} A{i+1}" for i in range(K_src)]
                tgt_labels = [f"{tgt_label} A{j+1}" for j in range(K_tgt)]
                all_labels = src_labels + tgt_labels

                sankey_src, sankey_tgt, sankey_val = [], [], []
                for i in range(K_src):
                    for j in range(K_tgt):
                        if corr_norm[i, j] > 0.05:  # only show flows > 5%
                            sankey_src.append(i)
                            sankey_tgt.append(K_src + j)
                            sankey_val.append(float(corr_norm[i, j]))

                if sankey_val:
                    fig_sankey = go.Figure(data=[go.Sankey(
                        node=dict(label=all_labels, pad=15, thickness=20),
                        link=dict(source=sankey_src, target=sankey_tgt, value=sankey_val),
                    )])
                    fig_sankey.update_layout(
                        title=f"Archetype correspondence flow: {_label}",
                        width=700, height=400)
                    html += safe_plotly_html(report, fig_sankey,
                        f"Sankey: archetype correspondence {_label} (flows > 5% shown)")
            except Exception as e_sankey:
                html += error_html(f"Sankey diagram failed: {e_sankey}")

        except Exception as e:
            html += error_html(f"Soft assignment ({pair_key}) failed: {e}")

    report.add_section("Soft Assignment Interpretation", html, step_num="12b")
    return top_pairs


def step12c_zoomed_flow(adata, flow_results, top_pairs, report):
    """Step 12c: Zoomed-in flow matching on top correspondence pairs.

    Subsets to cells assigned to the top archetype pairs from 12b,
    trains focused flow_within, and compares gene alignment against
    the full-population results.
    """
    import peach as pc

    html = ""
    zoomed_results = {}

    if not top_pairs:
        html += report.text("No top pairs from step 12b — skipping zoomed flow.")
        report.add_section("Zoomed Flow (Top Correspondence Pairs)", html, step_num="12c")
        return zoomed_results

    # Determine obs_key: check if pair labels match cell_type_short values
    first_pair_key = next(iter(top_pairs.keys()))
    src_label = first_pair_key.split("_to_")[0]
    if "cell_type_short" in adata.obs.columns and src_label in adata.obs["cell_type_short"].values:
        obs_key = "cell_type_short"
    elif "treatment" in adata.obs.columns:
        obs_key = "treatment"
    else:
        obs_key = "cell_type_short"

    for pair_key, arch_pairs in top_pairs.items():
        src_label, tgt_label = pair_key.split("_to_")
        _label = pair_key.replace("_to_", " → ")
        fr_full = flow_results.get(pair_key)
        if fr_full is None:
            continue

        html += f"<h3>Zoomed flow: {_label}</h3>"

        for src_arch, tgt_arch, corr_score in arch_pairs[:2]:  # Top 2 pairs
            zoom_label = f"A{src_arch+1}({src_label})→A{tgt_arch+1}({tgt_label})"
            html += f"<h4>{zoom_label} (correspondence={corr_score:.2f})</h4>"
            try:
                # Subset to cells with high weight for the specific archetypes
                weights = adata.obsm.get("cell_archetype_weights")
                if weights is None:
                    html += error_html("No archetype weights for subsetting.")
                    continue

                source_mask = (adata.obs[obs_key] == src_label).values
                target_mask = (adata.obs[obs_key] == tgt_label).values

                # Select cells with substantial weight for this archetype (soft threshold)
                # Using soft weights instead of argmax avoids discarding cells
                # that the correspondence matrix counted via their soft membership.
                # Adaptive threshold: try 0.3 first, fall back to 0.15 if too few cells
                for SOFT_THRESH in (0.3, 0.15, 0.05):
                    src_cells = source_mask & (weights[:, src_arch] > SOFT_THRESH)
                    tgt_cells = target_mask & (weights[:, tgt_arch] > SOFT_THRESH)
                    if int(src_cells.sum()) >= 30 and int(tgt_cells.sum()) >= 30:
                        break

                n_src = int(src_cells.sum())
                n_tgt = int(tgt_cells.sum())
                html += report.text(f"Zoomed subset: {n_src} source cells (A{src_arch+1} in {src_label}), "
                                    f"{n_tgt} target cells (A{tgt_arch+1} in {tgt_label})")

                if n_src < 30 or n_tgt < 30:
                    html += error_html(f"Insufficient cells for zoomed flow ({n_src}, {n_tgt}), skipping.")
                    continue

                # Create subset adata with custom obs labels for flow_within
                adata_zoom = adata.copy()
                adata_zoom.obs["_zoom_group"] = "other"
                adata_zoom.obs.loc[src_cells, "_zoom_group"] = "zoom_src"
                adata_zoom.obs.loc[tgt_cells, "_zoom_group"] = "zoom_tgt"

                log.info(f"  Zoomed flow: {zoom_label} ({n_src} → {n_tgt} cells)...")
                zoom_fr = pc.tl.flow_within(
                    adata_zoom,
                    source={"_zoom_group": "zoom_src"},
                    target={"_zoom_group": "zoom_tgt"},
                    n_epochs=600,
                    hidden_dims=(128, 128, 128),
                    batch_size=max(1, min(256, min(n_src, n_tgt) // 2)),
                    return_model=True,
                    name=f"zoom_{pair_key}_A{src_arch+1}_A{tgt_arch+1}",
                )

                zoomed_results[f"{pair_key}_A{src_arch+1}_A{tgt_arch+1}"] = zoom_fr

                html += metric_grid([
                    metric_card(zoom_label, "Zoomed pair"),
                    metric_card(f"{zoom_fr['mmd_before']:.4f}", "MMD before"),
                    metric_card(f"{zoom_fr['mmd_after']:.4f}", "MMD after"),
                    metric_card(f"{1 - zoom_fr['mmd_after']/max(zoom_fr['mmd_before'], 1e-10):.1%}",
                                "MMD reduction"),
                ])

                # Gene alignment with permutation
                log.info(f"  Zoomed gene alignment: {zoom_label}...")
                zoom_align = pc.tl.flow_gene_alignment(
                    adata_zoom, zoom_fr, n_top=20, per_cell=False,
                    n_permutations=200, null_type="both")

                # Top aligned genes table
                align_scores = zoom_align["alignment_scores"]
                gene_names = list(adata_zoom.var_names)
                sorted_idx = np.argsort(np.abs(align_scores))[::-1]

                align_rows = []
                align_pvals = zoom_align.get("alignment_pvalues")
                align_fdr = zoom_align.get("alignment_pvalues_fdr")
                null_mean_z = zoom_align.get("null_mean")
                for rank, gi in enumerate(sorted_idx[:20]):
                    row = {
                        "Rank": rank + 1,
                        "Gene": gene_names[gi],
                        "Alignment": f"{align_scores[gi]:.4f}",
                        "Direction": "increasing" if align_scores[gi] > 0 else "decreasing",
                    }
                    if align_pvals is not None:
                        row["p-value"] = fmt_pval(align_pvals[gi])
                    if align_fdr is not None:
                        row["FDR q"] = fmt_pval(align_fdr[gi])
                    align_rows.append(row)
                html += report.df_to_html(
                    pd.DataFrame(align_rows),
                    caption=f"Zoomed gene alignment: {zoom_label} (rotation null, n_perm=200)")

                # Null distribution comparison
                if align_pvals is not None and null_mean_z is not None:
                    n_sig = int((np.asarray(align_pvals) < 0.05).sum())
                    html += report.text(
                        f"Genes with raw p < 0.05: {n_sig}/{len(align_scores)}, "
                        f"observed mean |alignment| = {np.mean(np.abs(align_scores)):.4f}, "
                        f"null mean |alignment| = {np.mean(np.abs(null_mean_z)):.4f}")

                # Pathway alignment (if pathway scores available)
                if "pathway_scores" in adata.obsm and "PCs" in adata.varm:
                    try:
                        # Pathway-level flow alignment via pathway score change
                        source_mask_z = zoom_fr["source_mask"]
                        transported_z = zoom_fr["transported"]
                        source_pca_z = adata_zoom.obsm["X_pca"][source_mask_z]
                        delta_pca_z = transported_z - source_pca_z
                        pw_scores_src = adata_zoom.obsm["pathway_scores"][source_mask_z]
                        # Mean pathway score change along flow
                        pw_names = list(adata_zoom.uns.get("pathway_scores_pathways", []))
                        if pw_names:
                            mean_delta_pw = delta_pca_z.mean(axis=0)  # mean PCA delta
                            # Pathway change = correlation of pathway scores with flow magnitude
                            from scipy.stats import spearmanr as _sp
                            flow_mag = np.linalg.norm(delta_pca_z, axis=1)
                            pw_flow_rows = []
                            for pi, pname in enumerate(pw_names[:100]):  # limit
                                rho, pval = _sp(pw_scores_src[:, pi], flow_mag)
                                if not np.isnan(rho):
                                    pw_flow_rows.append({
                                        "Pathway": pname,
                                        "Flow correlation": f"{rho:.3f}",
                                        "p-value": fmt_pval(pval),
                                    })
                            if pw_flow_rows:
                                pw_df = pd.DataFrame(pw_flow_rows)
                                pw_df = pw_df.sort_values("Flow correlation",
                                    key=lambda x: x.apply(lambda v: abs(float(v))),
                                    ascending=False).head(15)
                                html += report.df_to_html(pw_df,
                                    caption=f"Zoomed pathway-flow correlation: {zoom_label}")
                    except Exception as e_pw:
                        html += error_html(f"Zoomed pathway alignment ({zoom_label}) failed: {e_pw}")

                # Compare zoomed vs full-population alignment
                full_align = None
                # Try to get full-population alignment (computed in step 13)
                # We'll compute it here for comparison
                try:
                    full_align_result = pc.tl.flow_gene_alignment(
                        adata, fr_full, n_top=20, per_cell=False, n_permutations=0)
                    full_scores = full_align_result["alignment_scores"]
                    if len(full_scores) == len(align_scores):
                        from scipy.stats import spearmanr
                        rho, p = spearmanr(np.abs(align_scores), np.abs(full_scores))
                        html += report.text(
                            f"Concordance zoomed vs full-population |alignment|: "
                            f"Spearman ρ={rho:.3f}, p={fmt_pval(p)}")

                        # Jaccard of top 30
                        top_zoom = set(sorted_idx[:30])
                        top_full = set(np.argsort(np.abs(full_scores))[::-1][:30])
                        jaccard = len(top_zoom & top_full) / max(len(top_zoom | top_full), 1)
                        html += metric_grid([
                            metric_card(f"{rho:.2f}", "Spearman ρ (zoom vs full)"),
                            metric_card(f"{jaccard:.2f}", "Jaccard top-30"),
                        ])

                        # Genes unique to zoomed analysis
                        zoom_only = top_zoom - top_full
                        if zoom_only:
                            zoom_only_genes = [gene_names[gi] for gi in zoom_only]
                            html += report.text(
                                f"Genes in zoomed top-30 but NOT full top-30 ({len(zoom_only)}): "
                                + ", ".join(zoom_only_genes[:15]))
                except Exception as e_comp:
                    html += error_html(f"Zoom vs full comparison failed: {e_comp}")

                # Jacobian with permutation
                zoom_model = zoom_fr.get("model")
                if zoom_model is not None:
                    try:
                        log.info(f"  Zoomed Jacobian: {zoom_label}...")
                        zoom_jac = pc.tl.flow_jacobian(
                            adata_zoom, zoom_fr, zoom_model,
                            per_cell_features=True, n_top_features=500,
                            n_permutations=200)

                        expansion = zoom_jac["feature_expansion"]
                        exp_pvals = zoom_jac.get("expansion_pvalues")
                        exp_fdr = zoom_jac.get("expansion_pvalues_fdr")

                        # Report expanding/contracting genes
                        jac_rows = []
                        sorted_exp = np.argsort(expansion)
                        top_expand = sorted_exp[-10:][::-1]
                        top_contract = sorted_exp[:10]

                        for gi in top_expand:
                            row = {"Gene": gene_names[gi], "Expansion": f"{expansion[gi]:.4f}",
                                   "Direction": "expanding"}
                            if exp_pvals is not None:
                                row["p-value"] = fmt_pval(exp_pvals[gi])
                            if exp_fdr is not None:
                                row["FDR q"] = fmt_pval(exp_fdr[gi])
                            jac_rows.append(row)
                        for gi in top_contract:
                            row = {"Gene": gene_names[gi], "Expansion": f"{expansion[gi]:.4f}",
                                   "Direction": "contracting"}
                            if exp_pvals is not None:
                                row["p-value"] = fmt_pval(exp_pvals[gi])
                            if exp_fdr is not None:
                                row["FDR q"] = fmt_pval(exp_fdr[gi])
                            jac_rows.append(row)
                        html += report.df_to_html(
                            pd.DataFrame(jac_rows),
                            caption=f"Zoomed Jacobian expansion/contraction: {zoom_label}")

                        if exp_pvals is not None:
                            n_sig = int((np.asarray(exp_pvals) < 0.05).sum())
                            html += report.text(f"Genes with raw p < 0.05: {n_sig} / {len(expansion)}")
                        if exp_fdr is not None:
                            n_fdr = int((np.asarray(exp_fdr) < 0.05).sum())
                            html += report.text(f"Genes with FDR q < 0.05: {n_fdr} / {len(expansion)}")

                    except Exception as e_jac:
                        html += error_html(f"Zoomed Jacobian ({zoom_label}) failed: {e_jac}")

                del adata_zoom  # free memory

            except Exception as e:
                html += error_html(f"Zoomed flow ({zoom_label}) failed: {e}")

    report.add_section("Zoomed Flow (Top Correspondence Pairs)", html, step_num="12c")
    return zoomed_results


def step13_sinkhorn_flow(adata, flow_results, report):
    """Step 13: OT flow, gene alignment, gene expression change along flow."""
    import peach as pc

    html = ""
    alignment_results = {}

    if not flow_results:
        # Try training new OT flows using cell type or treatment pairs
        if "treatment" in adata.obs.columns:
            fallback_pairs = _make_dose_pairs()
            fallback_key = "treatment"
        elif "cell_type_short" in adata.obs.columns:
            fallback_pairs = FLOW_PAIRS
            fallback_key = "cell_type_short"
        else:
            fallback_pairs = []
            fallback_key = None

        for src, tgt in fallback_pairs:
            available = set(adata.obs[fallback_key].unique())
            if src not in available or tgt not in available:
                continue
            n_src = int((adata.obs[fallback_key] == src).sum())
            n_tgt = int((adata.obs[fallback_key] == tgt).sum())
            if n_src < MIN_CELLS_FLOW or n_tgt < MIN_CELLS_FLOW:
                continue
            pair_key = f"{src}_to_{tgt}"
            log.info(f"  OT flow: {pair_key} ...")
            try:
                fr = pc.tl.flow_within(
                    adata,
                    source={fallback_key: src},
                    target={fallback_key: tgt},
                    n_epochs=600,
                    use_ot=True,
                    return_model=True,
                    name=f"{pair_key}_ot",
                )
                flow_results[pair_key] = fr
            except Exception as e:
                html += error_html(f"OT flow {pair_key} failed: {e}")

    # Flow QC summary
    qc_rows = []
    for pair_key, fr in flow_results.items():
        qc_rows.append({
            "Pair": pair_key,
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
        html += f"<h4>Flow: {pair_key.replace('_to_', ' → ')}</h4>"
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

    # Gene alignment + velocity quiver
    html += report.text("Gene-flow alignment: cosine similarity between each gene's PCA loading vector "
                        "and the mean flow velocity. Positive = gene expression increases along the flow "
                        "direction. Negative = expression decreases. Scores are direction-only (normalized).")
    html += report.text(
        "Null: permutation of gene-to-PCA-loading assignments. Tests whether a specific gene's "
        "alignment with the flow field is stronger than expected for a random gene-loading pairing.")
    for pair_key, fr in flow_results.items():
        _flow_label = pair_key.replace("_to_", " \u2192 ")
        html += f"<h4>Flow: {_flow_label}</h4>"
        log.info(f"  Gene alignment: {pair_key}...")
        try:
            align = pc.tl.flow_gene_alignment(adata, fr, n_top=30, per_cell=False,
                                              n_permutations=200, null_type="both")
            alignment_results[pair_key] = align

            # Dual-null permutation statistics
            align_scores = align["alignment_scores"]
            align_pvals = align.get("alignment_pvalues")  # shuffle null (per-gene)
            align_fdr = align.get("alignment_pvalues_fdr")
            null_mean = align.get("null_mean")  # shuffle null mean
            omnibus_p = align.get("rotation_omnibus_pvalue")  # rotation null (global)
            n_total = len(align_scores)

            # Omnibus test (rotation null)
            if omnibus_p is not None:
                html += metric_grid([
                    metric_card(fmt_pval(omnibus_p), "Rotation omnibus p"),
                    metric_card(n_total, "Genes tested"),
                ])
                html += report.text(
                    f"<b>Rotation null</b> (omnibus): tests whether the loading manifold's "
                    f"orientation relative to velocity is globally significant. p={fmt_pval(omnibus_p)}")

            # Per-gene test (shuffle null)
            if align_pvals is not None:
                raw_p = np.asarray(align_pvals)
                n_shuf_sig = int((raw_p < 0.05).sum())
                n_fdr_sig = int((np.asarray(align_fdr) < 0.05).sum()) if align_fdr is not None else 0
                html += metric_grid([
                    metric_card(n_shuf_sig, "Shuffle p < 0.05"),
                    metric_card(n_fdr_sig, "Shuffle FDR < 0.05"),
                    metric_card(f"{n_shuf_sig/n_total:.1%}", "% significant"),
                ])
                html += report.text(
                    f"<b>Shuffle null</b> (per-gene): tests whether each gene is specifically "
                    f"aligned vs a random gene's loading. {n_shuf_sig}/{n_total} at p<0.05, "
                    f"{n_fdr_sig}/{n_total} at FDR<0.05.")

                # Histogram: observed vs shuffle null
                if null_mean is not None:
                    try:
                        fig_null, ax_null = plt.subplots(figsize=(8, 4))
                        ax_null.hist(np.abs(align_scores), bins=50, alpha=0.6, color="#0072B2",
                                     label="Observed |alignment|", density=True)
                        ax_null.hist(np.abs(null_mean), bins=50, alpha=0.6, color="#D55E00",
                                     label="Shuffle null mean", density=True)
                        ax_null.set_xlabel("|Gene-flow alignment score|")
                        ax_null.set_ylabel("Density")
                        ax_null.set_title(f"Observed vs shuffle null: {pair_key}")
                        ax_null.legend()
                        ax_null.spines[["top", "right"]].set_visible(False)
                        fig_null.tight_layout()
                        html += report.fig_to_img(fig_null, caption=f"Alignment: observed vs shuffle null ({pair_key})")
                        plt.close("all")
                    except Exception:
                        plt.close("all")

                # Top genes table with shuffle p-values
                sorted_idx = np.argsort(np.abs(align_scores))[::-1]
                sig_rows = []
                for rank, gi in enumerate(sorted_idx[:30]):
                    row = {
                        "Rank": rank + 1,
                        "Gene": adata.var_names[gi],
                        "Alignment": f"{align_scores[gi]:.4f}",
                        "Direction": "increasing" if align_scores[gi] > 0 else "decreasing",
                        "Shuffle p": fmt_pval(raw_p[gi]),
                    }
                    if align_fdr is not None:
                        row["Shuffle FDR"] = fmt_pval(align_fdr[gi])
                    sig_rows.append(row)
                html += report.df_to_html(
                    pd.DataFrame(sig_rows),
                    caption=f"Top 30 genes by |alignment| with shuffle p-values: {pair_key}")
            else:
                html += report.text("No permutation p-values available.")

            fig_bar = pc.pl.gene_alignment_barplot(adata, align, n_top=20, show=False)
            html += safe_plotly_html(report, fig_bar, f"Gene alignment barplot: {pair_key}")
        except Exception as e:
            html += error_html(f"Gene alignment ({pair_key}) failed: {e}")

        try:
            fig_quiv = pc.pl.velocity_quiver(adata, fr, show=False)
            fig_quiv.update_layout(title=f"Flow velocity field: {pair_key.replace('_to_', ' → ')}")
            # Raise cell scatter alpha for better visibility
            fig_quiv.update_traces(marker=dict(opacity=0.75), selector=dict(mode="markers"))
            html += safe_plotly_html(report, fig_quiv, f"Velocity quiver: {pair_key}")
        except Exception as e:
            html += error_html(f"Quiver ({pair_key}) failed: {e}")

    # Significance-filtered gene alignment summary with dual-null comparison
    for pair_key, align in alignment_results.items():
        _flow_label = pair_key.replace("_to_", " → ")
        html += f"<h4>Significance-filtered genes: {_flow_label}</h4>"
        align_pvals = align.get("alignment_pvalues")  # shuffle null
        align_fdr = align.get("alignment_pvalues_fdr")
        align_scores = align["alignment_scores"]
        omnibus_p = align.get("rotation_omnibus_pvalue")

        # Report omnibus first
        if omnibus_p is not None:
            omnibus_sig = "YES" if omnibus_p < 0.05 else "NO"
            html += report.text(
                f"Rotation omnibus: p={fmt_pval(omnibus_p)} — global alignment "
                f"{'is' if omnibus_p < 0.05 else 'is NOT'} significant.")

        if align_fdr is not None:
            fdr_mask = np.asarray(align_fdr) < 0.05
            n_fdr = int(fdr_mask.sum())
            if n_fdr > 0:
                fdr_idx = np.where(fdr_mask)[0]
                sorted_fdr = fdr_idx[np.argsort(np.abs(align_scores[fdr_idx]))[::-1]]
                sig_rows = []
                for rank, gi in enumerate(sorted_fdr[:30]):
                    sig_rows.append({
                        "Rank": rank + 1,
                        "Gene": adata.var_names[gi],
                        "Alignment": f"{align_scores[gi]:.4f}",
                        "Direction": "increasing" if align_scores[gi] > 0 else "decreasing",
                        "Shuffle FDR q": fmt_pval(align_fdr[gi]),
                    })
                html += report.df_to_html(
                    pd.DataFrame(sig_rows),
                    caption=f"FDR-significant genes (shuffle null, q<0.05): {pair_key} ({n_fdr} total)")

                html += report.text(
                    f"<b>Interpretation</b>: rotation omnibus confirms global alignment is "
                    f"{'real' if omnibus_p is not None and omnibus_p < 0.05 else 'not confirmed'}; "
                    f"shuffle null identifies {n_fdr} genes with gene-specific alignment. "
                    "Genes significant under the shuffle null are specifically aligned with "
                    "the flow beyond what's expected from a random gene in this loading space.")
            else:
                html += report.text(f"No genes survive FDR correction for {pair_key}. "
                                    "Showing top 20 by raw p-value above.")
        elif align_pvals is not None:
            html += report.text("FDR correction not available; raw p-values shown in table above.")

    report.add_section("Sinkhorn Flow & Gene Alignment", html, step_num=13)
    return alignment_results


def step14_jacobian(adata, flow_results, alignment_results, report):
    """Step 14: Jacobian expansion/contraction, cross-method concordance."""
    import peach as pc

    html = ""
    jac_results = {}

    for pair_key, fr in flow_results.items():
        html += f"<h4>Flow: {pair_key.replace('_to_', ' → ')}</h4>"
        model = fr.get("model")
        if model is None:
            html += error_html(f"{pair_key}: no model in flow result, skipping Jacobian.")
            continue

        log.info(f"  Jacobian: {pair_key}...")
        try:
            jac = pc.tl.flow_jacobian(
                adata, fr, model, per_cell_features=True, n_top_features=500,
                n_permutations=200, null_type="both",
            )
            jac_results[pair_key] = jac

            det = jac["jacobian_det"]
            expansion = jac["feature_expansion"]

            # Jacobian diagnostic: det>1 = expanding volume, det<1 = contracting
            # If det ≈ 1 everywhere, flow is near-identity (weak transport)
            det_finite = det[np.isfinite(det)]
            html += metric_grid([
                metric_card(pair_key, "Flow pair"),
                metric_card(f"{np.median(det_finite):.4f}", "Median det(J)"),
                metric_card(f"{np.mean(det_finite):.4f}", "Mean det(J)"),
                metric_card(f"{np.std(det_finite):.4f}", "Std det(J)"),
                metric_card(f"{(det_finite > 1).mean():.1%}", "% expanding (det>1)"),
                metric_card(f"{(det_finite < 1).mean():.1%}", "% contracting (det<1)"),
            ])

            # Flag if Jacobian is near-identity (weak flow)
            if np.std(det_finite) < 0.01:
                html += report.text(
                    "⚠️ Jacobian determinants are near-constant (std < 0.01). "
                    "This suggests the flow field has minimal local expansion/contraction. "
                    "Check flow training convergence and MMD improvement."
                )

            # Top expanded/contracted genes
            html += report.text("Jacobian feature expansion: quadratic form L^T\u00b7J\u00b7L for each gene's normalized "
                                "PCA loading L and the Jacobian J of the velocity field. Values >0 indicate the "
                                "gene's PCA direction is locally expanding (diverging trajectories); <0 indicates "
                                "contraction (converging). Evaluated at t=0.5 (midpoint of learned flow).")
            if len(expansion) > 0:
                gene_names = list(adata.var_names)

                # Significance diagnostics (dual null)
                exp_pvals = jac.get("expansion_pvalues")  # shuffle null (per-gene)
                exp_fdr = jac.get("expansion_pvalues_fdr")
                exp_omnibus = jac.get("expansion_rotation_omnibus_pvalue")  # rotation null
                n_raw_sig = 0

                if exp_omnibus is not None:
                    html += report.text(
                        f"<b>Rotation omnibus</b>: expansion/contraction structure globally "
                        f"significant? p={fmt_pval(exp_omnibus)}")

                if exp_pvals is not None:
                    raw_p = np.asarray(exp_pvals)
                    n_raw_sig = int((raw_p < 0.05).sum())
                    html += report.text(f"<b>Shuffle null</b> (per-gene): {n_raw_sig}/{len(raw_p)} genes at p<0.05")

                if exp_fdr is not None:
                    fdr_p = np.asarray(exp_fdr)
                    n_fdr_sig = int((fdr_p < 0.05).sum())
                    html += report.text(f"Shuffle FDR q < 0.05: {n_fdr_sig}/{len(fdr_p)}")
                    if n_fdr_sig == 0 and n_raw_sig > 0:
                        html += report.text(
                            "<em>No genes survive FDR correction. Showing raw-p significant genes below.</em>")

                # Filter to significant genes when p-values are available
                if exp_pvals is not None and n_raw_sig > 0:
                    sig_mask = np.asarray(exp_pvals) < 0.05
                    expansion_display = np.where(sig_mask, expansion, np.nan)
                    sorted_exp = np.argsort(expansion_display)
                    # Remove NaN (non-significant) entries from sorted indices
                    sorted_exp = [gi for gi in sorted_exp if not np.isnan(expansion_display[gi])]
                else:
                    sorted_exp = list(np.argsort(expansion))

                top_expand = sorted_exp[-15:][::-1] if len(sorted_exp) >= 15 else sorted_exp[::-1]
                top_contract = sorted_exp[:15] if len(sorted_exp) >= 15 else sorted_exp
                exp_rows = []
                for gi in top_expand:
                    row = {"Gene": gene_names[gi], "Expansion": f"{expansion[gi]:.4f}", "Direction": "expanding"}
                    if exp_pvals is not None:
                        row["p-value"] = fmt_pval(exp_pvals[gi])
                    if exp_fdr is not None:
                        row["FDR q"] = fmt_pval(exp_fdr[gi])
                    exp_rows.append(row)
                for gi in top_contract:
                    if gi not in top_expand:
                        row = {"Gene": gene_names[gi], "Expansion": f"{expansion[gi]:.4f}", "Direction": "contracting"}
                        if exp_pvals is not None:
                            row["p-value"] = fmt_pval(exp_pvals[gi])
                        if exp_fdr is not None:
                            row["FDR q"] = fmt_pval(exp_fdr[gi])
                        exp_rows.append(row)
                html += report.df_to_html(pd.DataFrame(exp_rows),
                                          caption=f"Top expanded/contracted genes (with rotation-null p-values): {pair_key}")

                # Null distribution histogram for Jacobian expansion
                if exp_pvals is not None:
                    try:
                        fig_jnull, ax_jnull = plt.subplots(figsize=(8, 4))
                        ax_jnull.hist(np.abs(expansion), bins=50, alpha=0.6, color="#0072B2",
                                      label="Observed |expansion|", density=True)
                        # For null comparison: compute mean |expansion| across all permutations
                        n_perm = jac.get("n_permutations", 200)
                        ax_jnull.set_xlabel("|Feature expansion score|")
                        ax_jnull.set_ylabel("Density")
                        ax_jnull.set_title(f"Jacobian expansion: observed distribution ({pair_key})")
                        n_raw = int((np.asarray(exp_pvals) < 0.05).sum())
                        n_fdr = int((np.asarray(exp_fdr) < 0.05).sum()) if exp_fdr is not None else 0
                        ax_jnull.legend(title=f"raw p<0.05: {n_raw}, FDR q<0.05: {n_fdr}")
                        ax_jnull.spines[["top", "right"]].set_visible(False)
                        fig_jnull.tight_layout()
                        html += report.fig_to_img(fig_jnull,
                            caption=f"Jacobian expansion distribution ({n_perm} rotation permutations): {pair_key}")
                        plt.close("all")
                    except Exception:
                        plt.close("all")

                # Per-archetype expansion breakdown
                per_cell_exp = jac.get("per_cell_expansion")
                per_cell_gene_names = jac.get("per_cell_expansion_gene_names", [])
                if per_cell_exp is not None and len(per_cell_gene_names) > 0 and "archetypes" in adata.obs.columns:
                    try:
                        source_mask = fr["source_mask"]
                        arch_labels = adata.obs["archetypes"].values[source_mask]
                        unique_archs = sorted(set(arch_labels))
                        arch_exp_rows = []
                        for arch in unique_archs:
                            arch_mask = arch_labels == arch
                            if arch_mask.sum() < 5:
                                continue
                            if per_cell_exp.ndim == 2:
                                arch_mean_exp = per_cell_exp[arch_mask].mean(axis=0)
                            else:
                                arch_mean_exp = per_cell_exp[arch_mask]
                            # Top 5 expanding and contracting for this archetype
                            sorted_ae = np.argsort(arch_mean_exp)
                            for gi in sorted_ae[-5:][::-1]:
                                arch_exp_rows.append({
                                    "Archetype": display_arch(arch),
                                    "Gene": per_cell_gene_names[gi],
                                    "Mean expansion": f"{arch_mean_exp[gi]:.4f}",
                                    "Direction": "expanding",
                                })
                            for gi in sorted_ae[:5]:
                                arch_exp_rows.append({
                                    "Archetype": display_arch(arch),
                                    "Gene": per_cell_gene_names[gi],
                                    "Mean expansion": f"{arch_mean_exp[gi]:.4f}",
                                    "Direction": "contracting",
                                })
                        if arch_exp_rows:
                            html += report.df_to_html(
                                pd.DataFrame(arch_exp_rows),
                                caption=f"Per-archetype top expanding/contracting genes: {pair_key}")
                    except Exception as e_arch:
                        html += error_html(f"Per-archetype expansion ({pair_key}) failed: {e_arch}")

                # Pathway expansion: correlate pathway scores with per-cell expansion
                if "pathway_scores" in adata.obsm:
                    try:
                        from scipy.stats import spearmanr as _spearmanr
                        pw_names_all = list(adata.uns.get("pathway_names", []))
                        if not pw_names_all:
                            pw_names_all = list(adata.uns.get("pathway_scores_pathways", []))
                        pw_scores = adata.obsm["pathway_scores"][fr["source_mask"]]
                        per_cell_exp = jac.get("per_cell_expansion")
                        if per_cell_exp is not None and len(pw_names_all) > 0:
                            # per_cell_expansion is [n_source, n_features]; take mean across features
                            if per_cell_exp.ndim == 2:
                                cell_exp_scalar = per_cell_exp.mean(axis=1)
                            else:
                                cell_exp_scalar = per_cell_exp
                            pw_expansion = []
                            for pi, pname in enumerate(pw_names_all):
                                rho, _ = _spearmanr(pw_scores[:, pi], cell_exp_scalar)
                                pw_expansion.append({
                                    "Pathway": pname,
                                    "Expansion_corr": float(rho),
                                })
                            pw_exp_df = pd.DataFrame(pw_expansion)
                            pw_exp_df = pw_exp_df.sort_values(
                                "Expansion_corr", key=lambda x: x.abs(), ascending=False
                            ).head(15)
                            pw_exp_df["Expansion_corr"] = pw_exp_df["Expansion_corr"].map(lambda x: f"{x:.4f}")
                            html += report.df_to_html(
                                pw_exp_df,
                                caption=f"Top pathways by expansion correlation: {pair_key}",
                            )
                    except Exception as e_pw:
                        html += error_html(f"Pathway expansion ({pair_key}) failed: {e_pw}")

        except Exception as e:
            html += error_html(f"Jacobian ({pair_key}) failed: {e}")

    # Cross-method concordance (alignment vs expansion vs regression)
    for pair_key in set(jac_results.keys()) & set(alignment_results.keys()):
        try:
            from scipy.stats import spearmanr

            jac = jac_results[pair_key]
            align = alignment_results[pair_key]
            expansion = jac["feature_expansion"]
            align_scores = align["alignment_scores"]

            # Both are per-gene vectors
            if len(expansion) == len(align_scores) and len(expansion) > 10:
                rho, p = spearmanr(np.abs(expansion), np.abs(align_scores))
                html += report.text(
                    f"Concordance |expansion| vs |alignment| for {pair_key}: "
                    f"Spearman rho={rho:.3f}, p={fmt_pval(p)}"
                )

                # Jaccard of top 50
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

    # Flow magnitude for each pair
    for pair_key, fr in flow_results.items():
        try:
            fig_mag = pc.pl.flow_magnitude(adata, fr, show=False)
            fig_mag.update_traces(marker=dict(opacity=0.6))
            html += safe_plotly_html(report, fig_mag, f"Flow magnitude: {pair_key}")
        except Exception as e:
            html += error_html(f"Flow magnitude ({pair_key}) failed: {e}")

    report.add_section("Jacobian Expansion/Contraction", html, step_num=14)
    return jac_results


def step15_gene_deep_dive(adata, flow_results, jac_results, report):
    """Step 15: Trajectory ribbon, topo landscape for top gene deep dive."""
    import peach as pc

    html = ""

    # Trajectory ribbon for each flow
    for pair_key, fr in flow_results.items():
        html += f"<h4>Flow: {pair_key.replace('_to_', ' → ')}</h4>"
        model = fr.get("model")
        try:
            fig_rib = pc.pl.trajectory_ribbon(adata, fr, flow_model=model, show=False)
            html += safe_plotly_html(report, fig_rib, f"Trajectory ribbon: {pair_key}")
        except Exception as e:
            html += error_html(f"Trajectory ribbon ({pair_key}) failed: {e}")

    # Topo landscape for first flow pair with model + Jacobian
    for pair_key, fr in flow_results.items():
        html += f"<h4>Flow: {pair_key.replace('_to_', ' → ')}</h4>"
        model = fr.get("model")
        if model is None:
            continue
        try:
            pc.pl.flow_topo_landscape(
                adata, fr, model, n_features=5, n_eval_points=200, show=False,
                show_velocity=False,
                save=os.path.join(OUTPUT_DIR, f"topo_{pair_key}.png"),
            )
            # Read it back for the report
            topo_path = os.path.join(OUTPUT_DIR, f"topo_{pair_key}.png")
            if os.path.exists(topo_path):
                with open(topo_path, "rb") as fh:
                    b64 = base64.b64encode(fh.read()).decode("utf-8")
                html += f'<img src="data:image/png;base64,{b64}" style="max-width:100%;">'
                html += f"<p class='caption'>Topographic landscape: {pair_key}</p>"
        except Exception as e:
            html += error_html(f"Topo landscape ({pair_key}) failed: {e}")
            plt.close("all")
        break  # Only first pair to save time

    # Per-cell expansion ridgeplots: expansion vs flow pseudotime
    log.info(f"Ridgeplots: {len(jac_results)} Jacobian results available")
    for pair_key, jac in jac_results.items():
        per_cell = jac.get("per_cell_expansion")
        gene_names = jac.get("per_cell_expansion_gene_names", [])
        log.info(f"  {pair_key}: per_cell={'present' if per_cell is not None else 'MISSING'}, "
                 f"n_genes={len(gene_names)}")
    for pair_key, jac in jac_results.items():
        per_cell = jac.get("per_cell_expansion")
        gene_names = jac.get("per_cell_expansion_gene_names", [])
        if per_cell is None or len(gene_names) == 0:
            continue

        fr = flow_results.get(pair_key)
        if fr is None:
            continue

        try:
            # Compute flow pseudotime: project source cells onto flow axis (t=0→1)
            source_pca = adata.obsm["X_pca"][fr["source_mask"]]
            transported = fr["transported"]
            flow_direction = (transported - source_pca)
            # Pseudotime = projection of cell position onto mean flow direction
            mean_flow = flow_direction.mean(axis=0)
            mean_flow_norm = mean_flow / (np.linalg.norm(mean_flow) + 1e-10)
            pseudotime = (source_pca - source_pca.mean(axis=0)) @ mean_flow_norm
            # Normalize to [0, 1]
            pt_min, pt_max = pseudotime.min(), pseudotime.max()
            if pt_max - pt_min > 1e-10:
                pseudotime = (pseudotime - pt_min) / (pt_max - pt_min)
            else:
                pseudotime = np.zeros_like(pseudotime) + 0.5

            # Select genes ranked by effect size — work within per_cell's index space (500 genes)
            # per_cell shape: [n_cells, n_top_features], gene_names has n_top_features entries
            n_per_cell = per_cell.shape[1]
            per_cell_mean_abs = np.abs(per_cell.mean(axis=0))  # [n_top_features]

            # Map significance from full feature_expansion to per_cell indices
            exp_pvals_full = jac.get("expansion_pvalues")
            per_cell_gene_idx = jac.get("per_cell_expansion_gene_indices")
            used_sig_filter = False
            if exp_pvals_full is not None and per_cell_gene_idx is not None:
                exp_pvals_sub = np.asarray(exp_pvals_full)[per_cell_gene_idx]
                sig_mask = exp_pvals_sub < 0.05
                if sig_mask.sum() >= 5:
                    gene_order = np.where(sig_mask)[0]
                    gene_order = gene_order[np.argsort(per_cell_mean_abs[gene_order])[::-1]]
                    used_sig_filter = True
                else:
                    gene_order = np.argsort(per_cell_mean_abs)[::-1]
            else:
                gene_order = np.argsort(per_cell_mean_abs)[::-1]

            n_total = min(20, len(gene_order))
            gene_order = gene_order[:n_total]

            # Bin cells by pseudotime
            n_bins = 20
            pt_bins = np.linspace(0, 1, n_bins + 1)
            bin_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])
            bin_idx = np.clip(np.digitize(pseudotime, pt_bins) - 1, 0, n_bins - 1)

            # Overlay up to 5 genes per panel with transparency
            COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
            n_panels = (n_total + 4) // 5
            fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3.0 * n_panels), sharex=True)
            if n_panels == 1:
                axes = [axes]

            for panel_i in range(n_panels):
                ax = axes[panel_i]
                panel_genes = gene_order[panel_i * 5 : (panel_i + 1) * 5]

                for ci, gi in enumerate(panel_genes):
                    vals = per_cell[:, gi]
                    color = COLORS[ci % len(COLORS)]
                    bin_means = np.array([vals[bin_idx == b].mean() if (bin_idx == b).any() else np.nan
                                          for b in range(n_bins)])
                    bin_stds = np.array([vals[bin_idx == b].std() if (bin_idx == b).sum() > 1 else 0
                                         for b in range(n_bins)])
                    valid = ~np.isnan(bin_means)
                    ax.fill_between(bin_centers[valid],
                                    bin_means[valid] - bin_stds[valid],
                                    bin_means[valid] + bin_stds[valid],
                                    alpha=0.15, color=color)
                    ax.plot(bin_centers[valid], bin_means[valid], color=color, linewidth=1.5,
                            label=gene_names[gi], alpha=0.85)

                ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
                ax.legend(fontsize=8, loc="upper right", framealpha=0.7)
                ax.spines[["top", "right"]].set_visible(False)
                if panel_i == n_panels - 1:
                    ax.set_xlabel("Flow pseudotime (0 = source, 1 = target)")
                ax.set_ylabel("Expansion")

            sig_note = "(significant genes only)" if used_sig_filter else "(top by effect size)"
            fig.suptitle(
                f"Expansion along flow: {pair_key.replace('_to_', ' → ')} {sig_note}\n"
                "(>1 = expanding, <1 = contracting; shaded = ±1 std, 5 genes overlaid per panel)",
                y=1.02, fontsize=11)
            fig.tight_layout()
            html += report.fig_to_img(fig, caption=f"Expansion ridgeplots along flow pseudotime: {pair_key}")
            plt.close("all")
        except Exception as e:
            html += error_html(f"Expansion ridgeplots ({pair_key}) failed: {e}")
            plt.close("all")
        break  # Only first pair

    report.add_section("Gene Deep Dive", html, step_num=15)


def step16_per_response(adata, report):
    """Step 16: Per treatment response (R vs NR pooled across doses)."""
    import peach as pc

    html = ""

    if "pCR" not in adata.obs.columns:
        html += error_html("No 'pCR' column -- skipping per-response analysis.")
        report.add_section("Per-Response Analysis", html, step_num=16)
        return

    responses = sorted(adata.obs["pCR"].unique())
    html += report.text(f"Response groups: {', '.join(str(r) for r in responses)}")

    # Use global K (from already-trained model)
    global_K = adata.obsm.get("cell_archetype_weights")
    if global_K is not None:
        K = global_K.shape[1]
    else:
        K = 5
    html += report.text(
        f"<b>Note:</b> Per-response models use global K={K} (from step 2 model) "
        "rather than running separate hyperparameter searches. This ensures archetype "
        "comparability across response groups.")

    hidden_dims = [128, 256]

    summary_rows = []
    reg_rows = []
    response_adatas = {}

    for resp in responses:
        mask = (adata.obs["pCR"] == resp).values
        sub = _subset_adata(adata, mask, f"pCR={resp}")

        if sub.shape[0] < MIN_CELLS_MODEL:
            html += error_html(f"pCR={resp}: {sub.shape[0]} cells < {MIN_CELLS_MODEL}, skipped.")
            continue

        # Train model (skip CV, use global K)
        log.info(f"  pCR={resp}: training K={K}...")
        res = run_subset_model(sub, K=K, hidden_dims=hidden_dims, label=f"pCR={resp}")
        if res is None:
            continue

        # Ensure archetype coordinates are computed for plotting
        try:
            if "archetype_coordinates" not in sub.obsm:
                pc.tl.archetypal_coordinates(sub)
        except Exception:
            pass

        response_adatas[resp] = sub
        r2 = res.get("final_archetype_r2", float("nan"))
        summary_rows.append({
            "Response": resp, "N cells": sub.shape[0],
            "K": K,
            "Final R2": f"{r2:.4f}" if isinstance(r2, float) else str(r2),
        })

        # Regression
        try:
            reg = run_subset_regression(sub, label=f"pCR={resp}")
            if reg is not None:
                reg_rows.append(_reg_summary_row(reg, f"pCR={resp}"))
        except Exception as e:
            html += error_html(f"pCR={resp} regression failed: {e}")

    if summary_rows:
        html += report.df_to_html(pd.DataFrame(summary_rows), caption="Per-response model summary")
    if reg_rows:
        html += report.df_to_html(pd.DataFrame(reg_rows), caption="Per-response regression summary")

    # Exclusive dotplots per response group
    for resp, sub in response_adatas.items():
        try:
            fig_dot = pc.pl.archetype_regression_dotplot(
                sub, top_n=10, exclusive_only=True, show=False)
            html += safe_plotly_html(report, fig_dot,
                                     f"Exclusive gene features: pCR={resp}, top 10 per archetype (ranked by |β|)")
        except Exception as e:
            html += error_html(f"Dotplot pCR={resp} failed: {e}")

        if "pathway_scores" in sub.obsm:
            try:
                fig_pw = pc.pl.archetype_regression_dotplot(
                    sub, top_n=10, exclusive_only=True,
                    feature_type="pathways", show=False)
                html += safe_plotly_html(report, fig_pw,
                                         f"Exclusive pathway features: pCR={resp}, top 10 per archetype (ranked by |β|)")
            except Exception as e:
                html += error_html(f"Pathway dotplot pCR={resp} failed: {e}")

    # Archetype overlay: R vs NR side by side
    if len(response_adatas) >= 2:
        try:
            n_resp = len(response_adatas)
            fig, axes = plt.subplots(1, n_resp, figsize=(5 * n_resp, 4))
            if n_resp == 1:
                axes = [axes]
            for ax, (resp, sub) in zip(axes, response_adatas.items()):
                coords = sub.obsm.get("archetype_coordinates")
                if coords is not None and coords.shape[1] >= 2:
                    ax.scatter(coords[:, 0], coords[:, 1], s=2, alpha=0.3, c="#0072B2")
                    ax.set_title(f"pCR={resp} (n={sub.shape[0]})")
                    ax.set_xlabel("Arch coord 1")
                    ax.set_ylabel("Arch coord 2")
                    ax.spines[["top", "right"]].set_visible(False)
                    # Add archetype vertex markers and expand limits
                    arch_pos = sub.uns.get("archetype_coordinates")
                    if arch_pos is not None:
                        arch_pos = np.asarray(arch_pos)
                        if arch_pos.ndim == 2 and arch_pos.shape[1] >= 2:
                            ax.scatter(arch_pos[:, 0], arch_pos[:, 1], s=80, c="red",
                                       marker="^", zorder=5, edgecolors="black", linewidth=0.5)
                            all_pts = np.vstack([coords[:, :2], arch_pos[:, :2]])
                            margin = 0.1 * np.ptp(all_pts, axis=0)
                            margin = np.maximum(margin, 0.1)
                            ax.set_xlim(all_pts[:, 0].min() - margin[0], all_pts[:, 0].max() + margin[0])
                            ax.set_ylim(all_pts[:, 1].min() - margin[1], all_pts[:, 1].max() + margin[1])
            fig.suptitle("R vs NR archetypal space", y=1.02)
            fig.tight_layout()
            html += report.fig_to_img(fig, caption="Per-response archetypal space")
            plt.close("all")
        except Exception as e:
            html += error_html(f"Response overlay failed: {e}")
            plt.close("all")

    # Flow: R vs NR within each dose (if treatment column exists)
    if "treatment" in adata.obs.columns and len(response_adatas) >= 2:
        resp_list = list(response_adatas.keys())
        if len(resp_list) == 2:
            src_resp, tgt_resp = resp_list[0], resp_list[1]
            n_src = int((adata.obs["pCR"] == src_resp).sum())
            n_tgt = int((adata.obs["pCR"] == tgt_resp).sum())
            if n_src >= MIN_CELLS_FLOW and n_tgt >= MIN_CELLS_FLOW:
                try:
                    log.info(f"  Flow: pCR={src_resp} -> pCR={tgt_resp}...")
                    fr_resp = pc.tl.flow_within(
                        adata,
                        source={"pCR": src_resp},
                        target={"pCR": tgt_resp},
                        n_epochs=400,
                        return_model=True,
                        name=f"resp_{src_resp}_to_{tgt_resp}",
                    )
                    html += metric_grid([
                        metric_card(f"pCR {src_resp}->{tgt_resp}", "Flow"),
                        metric_card(f"{fr_resp['mmd_before']:.4f}", "MMD before"),
                        metric_card(f"{fr_resp['mmd_after']:.4f}", "MMD after"),
                    ])
                except Exception as e:
                    html += error_html(f"Response flow failed: {e}")

    # Clean up
    for sub in response_adatas.values():
        del sub
    response_adatas.clear()

    report.add_section("Per-Response Analysis", html, step_num=16)


def step17_per_response_per_dose(adata, report):
    """Step 17: Per treatment response per dose (6 groups)."""
    import peach as pc

    html = ""

    if "treatment" not in adata.obs.columns or "pCR" not in adata.obs.columns:
        html += error_html("Need both 'treatment' and 'pCR' columns -- skipping.")
        report.add_section("Per-Response-Per-Dose Analysis", html, step_num=17)
        return

    doses = sorted(adata.obs["treatment"].unique())
    responses = sorted(adata.obs["pCR"].unique())

    global_K = adata.obsm.get("cell_archetype_weights")
    K = global_K.shape[1] if global_K is not None else 5
    hidden_dims = [64, 128]

    summary_rows = []
    reg_rows = []
    group_r2 = {}  # (dose, resp) -> R2 vector for stability

    for dose in doses:
        for resp in responses:
            label = f"{dose}_{resp}"
            mask = ((adata.obs["treatment"] == dose) & (adata.obs["pCR"] == resp)).values
            n_cells = int(mask.sum())

            if n_cells < MIN_CELLS_MODEL:
                summary_rows.append({
                    "Dose": dose, "Response": resp, "N cells": n_cells,
                    "K": "skipped", "R2": "N/A",
                })
                continue

            sub = _subset_adata(adata, mask, label)

            # Use smaller K for small subsets
            k_use = min(K, max(3, n_cells // 200))
            log.info(f"  {label}: training K={k_use}...")
            try:
                res = run_subset_model(sub, K=k_use, hidden_dims=hidden_dims, label=label)
                if res is None:
                    summary_rows.append({
                        "Dose": dose, "Response": resp, "N cells": n_cells,
                        "K": k_use, "R2": "failed",
                    })
                    del sub
                    continue

                r2_val = res.get("final_archetype_r2", float("nan"))
                summary_rows.append({
                    "Dose": dose, "Response": resp, "N cells": n_cells,
                    "K": k_use,
                    "R2": f"{r2_val:.4f}" if isinstance(r2_val, float) else str(r2_val),
                })

                # Regression only (no flow -- too few cells)
                reg = run_subset_regression(sub, label=label)
                if reg is not None:
                    reg_rows.append(_reg_summary_row(reg, label))
                    group_r2[(dose, resp)] = np.asarray(reg["r_squared_degree1"])

                # Exclusive dotplot
                try:
                    fig_dot = pc.pl.archetype_regression_dotplot(
                        sub, top_n=10, exclusive_only=True, show=False)
                    html += safe_plotly_html(report, fig_dot,
                                             f"Exclusive gene features: {label}, top 10 per archetype (ranked by |β|)")
                except Exception as e_dot:
                    html += error_html(f"Dotplot {label} failed: {e_dot}")

                if "pathway_scores" in sub.obsm:
                    try:
                        fig_pw = pc.pl.archetype_regression_dotplot(
                            sub, top_n=10, exclusive_only=True,
                            feature_type="pathways", show=False)
                        html += safe_plotly_html(report, fig_pw,
                                                 f"Exclusive pathway features: {label}, top 10 per archetype (ranked by |β|)")
                    except Exception as e_pw:
                        html += error_html(f"Pathway dotplot {label} failed: {e_pw}")

            except Exception as e:
                html += error_html(f"{label} failed: {e}")
                summary_rows.append({
                    "Dose": dose, "Response": resp, "N cells": n_cells,
                    "K": k_use, "R2": f"error: {e}",
                })
            finally:
                del sub

    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        html += report.df_to_html(sdf, caption="Per-response-per-dose model summary (2x3 grid)")

    if reg_rows:
        html += report.df_to_html(pd.DataFrame(reg_rows),
                                  caption="Per-response-per-dose regression summary")

    # Feature stability heatmap: Spearman between all group pairs
    if len(group_r2) >= 2:
        try:
            from scipy.stats import spearmanr
            groups = list(group_r2.keys())
            # All R2 vectors need same feature order; they should since they come
            # from the same adata.var_names through gene_simplex_regression
            n_g = len(groups)
            stab_matrix = np.zeros((n_g, n_g))
            for i in range(n_g):
                for j in range(n_g):
                    v1 = group_r2[groups[i]]
                    v2 = group_r2[groups[j]]
                    min_len = min(len(v1), len(v2))
                    if min_len > 10:
                        rho, _ = spearmanr(v1[:min_len], v2[:min_len])
                        stab_matrix[i, j] = rho
                    else:
                        stab_matrix[i, j] = float("nan")

            labels = [f"{d}_{r}" for d, r in groups]
            fig, ax = plt.subplots(figsize=(max(6, n_g * 1.2), max(5, n_g * 1.0)))
            im = ax.imshow(stab_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(n_g))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(n_g))
            ax.set_yticklabels(labels, fontsize=8)
            plt.colorbar(im, ax=ax, label="Spearman rho", shrink=0.7)
            ax.set_title("Feature stability: R2 Spearman across dose x response")
            fig.tight_layout()
            html += report.fig_to_img(fig, caption="Feature stability heatmap (dose x response groups)")
            plt.close("all")
        except Exception as e:
            html += error_html(f"Feature stability heatmap failed: {e}")
            plt.close("all")

    # Flow between R and NR per dose
    html += "<h3>Flow: R → NR per dose</h3>"
    for dose in doses:
        r_mask = (adata.obs["treatment"] == dose) & (adata.obs["pCR"] == "R")
        nr_mask = (adata.obs["treatment"] == dose) & (adata.obs["pCR"] == "NR")
        n_r, n_nr = r_mask.sum(), nr_mask.sum()
        if n_r < 100 or n_nr < 100:
            html += report.text(f"{dose}: too few cells for flow (R={n_r}, NR={n_nr}), skipping.")
            continue
        try:
            fr = pc.tl.flow_within(
                adata,
                source={"treatment": dose, "pCR": "R"},
                target={"treatment": dose, "pCR": "NR"},
                n_epochs=400, hidden_dims=(128, 128), batch_size=128,
                return_model=True, use_ot=True, random_state=42,
            )
            html += metric_grid([
                metric_card(dose, "Dose"),
                metric_card(f"{fr['mmd_before']:.4f}", "MMD before"),
                metric_card(f"{fr['mmd_after']:.4f}", "MMD after"),
            ])
            # Gene alignment
            ga = pc.tl.flow_gene_alignment(adata, fr, normalize=True, per_cell=False)
            top_aligned = ga["top_aligned"][:10]
            top_opposed = ga["top_opposed"][:10]
            align_rows = [{"Gene": g, "Direction": "aligned"} for g in top_aligned]
            align_rows += [{"Gene": g, "Direction": "opposed"} for g in top_opposed]
            html += report.df_to_html(pd.DataFrame(align_rows),
                                      caption=f"R\u2192NR flow: top aligned/opposed genes ({dose})")
        except Exception as e:
            html += error_html(f"R\u2192NR flow ({dose}) failed: {e}")

    report.add_section("Per-Response-Per-Dose Analysis", html, step_num=17)


# ============================================================================
# Main runner
# ============================================================================

def main():
    import scanpy as sc
    import peach as pc

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report = HTMLReport(f"PEACH v0.5 -- HSC Myeloid Trajectory ({_REV_TAG})")

    # Load data
    log.info(f"Loading data from {DATA_PATH}...")
    adata = sc.read_h5ad(DATA_PATH)
    log.info(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # Define splits (safe -- HSC data won't have treatment/pCR)
    try:
        splits = define_splits(adata)
        log.info(f"Defined {len(splits)} splits")
    except Exception as e:
        log.warning(f"define_splits failed (expected for HSC data): {e}")
        splits = {}

    # Storage for cross-step results
    gene_reg = None
    flow_comparison_df = None

    # -- Step 1 --------------------------------------------------------------
    t0 = time.time()
    try:
        adata = step1_dataset_prep(adata, report)
        log.info(f"Step 1 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 1 failed: {e}", exc_info=True)
        report.add_section("Dataset Preparation", error_html(f"Step 1 failed: {e}"), step_num=1)

    # Save checkpoint
    safe_save_h5ad(adata, os.path.join(OUTPUT_DIR, "adata_step1.h5ad"))

    # -- Step 2 --------------------------------------------------------------
    t0 = time.time()
    results = None
    try:
        results = step2_hyperparameter_fit(adata, report)
        log.info(f"Step 2 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 2 failed: {e}", exc_info=True)
        report.add_section("Hyperparameter Search & Model Training",
                           error_html(f"Step 2 failed: {e}"), step_num=2)

    # Save checkpoint
    safe_save_h5ad(adata,os.path.join(OUTPUT_DIR, "adata_step2.h5ad"))
    if results is not None:
        # Save model separately
        try:
            model = results.get("final_model") or results.get("model")
            if model is not None:
                import torch
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_state_dict.pt"))
        except Exception:
            pass

    # -- Step 3 --------------------------------------------------------------
    t0 = time.time()
    try:
        gene_reg = step3_simplex_regression(adata, report)
        log.info(f"Step 3 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 3 failed: {e}", exc_info=True)
        report.add_section("Simplex Regression & Pattern Classification",
                           error_html(f"Step 3 failed: {e}"), step_num=3)

    # -- Step 4 --------------------------------------------------------------
    t0 = time.time()
    try:
        step4_hypergeometric(adata, report)
        log.info(f"Step 4 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 4 failed: {e}", exc_info=True)
        report.add_section("Hypergeometric Conditional Associations",
                           error_html(f"Step 4 failed: {e}"), step_num=4)

    # -- Step 5 --------------------------------------------------------------
    t0 = time.time()
    try:
        if gene_reg is None:
            # Try to recover from adata
            gene_reg = adata.uns.get("peach_simplex_regression") or adata.uns.get("peach_simplex_regression_genes", {})
        step5_wald_contrasts(adata, report, gene_reg)
        log.info(f"Step 5 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 5 failed: {e}", exc_info=True)
        report.add_section("Wald Contrasts",
                           error_html(f"Step 5 failed: {e}"), step_num=5)

    # -- Step 6 --------------------------------------------------------------
    t0 = time.time()
    try:
        flow_comparison_df = step6_within_fit_comparisons(adata, report)
        log.info(f"Step 6 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 6 failed: {e}", exc_info=True)
        report.add_section("Within-Fit Comparisons (Flow + Feature Similarity)",
                           error_html(f"Step 6 failed: {e}"), step_num=6)

    # -- Step 6b (diversity) --------------------------------------------------
    t0 = time.time()
    try:
        step6b_diversity_metrics(adata, report)
        log.info(f"Step 6b done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 6b failed: {e}", exc_info=True)
        report.add_section("Diversity Metrics", error_html(f"Step 6b failed: {e}"), step_num="6b")

    # Save checkpoint
    safe_save_h5ad(adata,os.path.join(OUTPUT_DIR, "adata_step6.h5ad"))

    # -- Step 7 --------------------------------------------------------------
    t0 = time.time()
    try:
        step7_driver_regression(adata, report)
        log.info(f"Step 7 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 7 failed: {e}", exc_info=True)
        report.add_section("Driver Regression",
                           error_html(f"Step 7 failed: {e}"), step_num=7)

    # -- Step 8 --------------------------------------------------------------
    t0 = time.time()
    try:
        step8_mixture_models(adata, report)
        log.info(f"Step 8 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 8 failed: {e}", exc_info=True)
        report.add_section("Mixture Model Decomposition",
                           error_html(f"Step 8 failed: {e}"), step_num=8)

    # -- Step 9 --------------------------------------------------------------
    t0 = time.time()
    try:
        step9_component_characterization(adata, report)
        log.info(f"Step 9 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 9 failed: {e}", exc_info=True)
        report.add_section("Component Characterization",
                           error_html(f"Step 9 failed: {e}"), step_num=9)

    # Save checkpoint after global analyses
    safe_save_h5ad(adata,os.path.join(OUTPUT_DIR, "adata_global.h5ad"))
    log.info("Global adata saved.")

    # ==== Per-subset analyses (steps 10-17) ====

    dose_adatas = {}
    flow_results = {}
    alignment_results = {}
    jac_results = {}

    # -- Step 10 -------------------------------------------------------------
    t0 = time.time()
    try:
        dose_adatas = step10_per_dose_models(adata, report)
        log.info(f"Step 10 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 10 failed: {e}", exc_info=True)
        report.add_section("Per-Condition Models",
                           error_html(f"Step 10 failed: {e}"), step_num=10)

    # -- Step 11 -------------------------------------------------------------
    t0 = time.time()
    try:
        step11_per_dose_regression(dose_adatas, report)
        log.info(f"Step 11 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 11 failed: {e}", exc_info=True)
        report.add_section("Per-Dose Regression",
                           error_html(f"Step 11 failed: {e}"), step_num=11)

    # -- Step 12 -------------------------------------------------------------
    t0 = time.time()
    try:
        flow_results = step12_between_dose_flow(adata, dose_adatas, report)
        log.info(f"Step 12 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 12 failed: {e}", exc_info=True)
        report.add_section("Between-Condition Flow",
                           error_html(f"Step 12 failed: {e}"), step_num=12)

    # -- Step 12b (soft assignment interpretation) ----------------------------
    top_pairs = {}
    t0 = time.time()
    try:
        top_pairs = step12b_soft_assignment_interpretation(adata, flow_results, report)
        log.info(f"Step 12b done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 12b failed: {e}", exc_info=True)
        report.add_section("Soft Assignment Interpretation",
                           error_html(f"Step 12b failed: {e}"), step_num="12b")

    # -- Step 12c (zoomed flow on top pairs) ----------------------------------
    zoomed_results = {}
    t0 = time.time()
    try:
        zoomed_results = step12c_zoomed_flow(adata, flow_results, top_pairs, report)
        log.info(f"Step 12c done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 12c failed: {e}", exc_info=True)
        report.add_section("Zoomed Flow (Top Correspondence Pairs)",
                           error_html(f"Step 12c failed: {e}"), step_num="12c")

    # Free per-dose adatas to save memory
    for sub in dose_adatas.values():
        del sub
    dose_adatas.clear()

    # -- Step 13 -------------------------------------------------------------
    t0 = time.time()
    try:
        alignment_results = step13_sinkhorn_flow(adata, flow_results, report)
        log.info(f"Step 13 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 13 failed: {e}", exc_info=True)
        report.add_section("Sinkhorn Flow & Gene Alignment",
                           error_html(f"Step 13 failed: {e}"), step_num=13)

    # -- Step 14 -------------------------------------------------------------
    t0 = time.time()
    try:
        jac_results = step14_jacobian(adata, flow_results, alignment_results, report)
        log.info(f"Step 14 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 14 failed: {e}", exc_info=True)
        report.add_section("Jacobian Expansion/Contraction",
                           error_html(f"Step 14 failed: {e}"), step_num=14)

    # -- Step 15 -------------------------------------------------------------
    t0 = time.time()
    try:
        step15_gene_deep_dive(adata, flow_results, jac_results, report)
        log.info(f"Step 15 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 15 failed: {e}", exc_info=True)
        report.add_section("Gene Deep Dive",
                           error_html(f"Step 15 failed: {e}"), step_num=15)

    # Free flow models to save memory
    for fr in flow_results.values():
        fr.pop("model", None)
    flow_results.clear()
    alignment_results.clear()
    jac_results.clear()
    for zfr in zoomed_results.values():
        zfr.pop("model", None)
    zoomed_results.clear()
    top_pairs.clear()

    # -- Step 16 -------------------------------------------------------------
    t0 = time.time()
    try:
        step16_per_response(adata, report)
        log.info(f"Step 16 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 16 failed: {e}", exc_info=True)
        report.add_section("Per-Response Analysis",
                           error_html(f"Step 16 failed: {e}"), step_num=16)

    # -- Step 17 -------------------------------------------------------------
    t0 = time.time()
    try:
        step17_per_response_per_dose(adata, report)
        log.info(f"Step 17 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 17 failed: {e}", exc_info=True)
        report.add_section("Per-Response-Per-Dose Analysis",
                           error_html(f"Step 17 failed: {e}"), step_num=17)

    # Final save
    safe_save_h5ad(adata,os.path.join(OUTPUT_DIR, "adata_final.h5ad"))
    log.info("Final adata saved.")

    # Save report
    report.save(REPORT_PATH)
    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()

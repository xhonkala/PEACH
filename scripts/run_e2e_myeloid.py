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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("e2e_myeloid")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = "/Users/honkala/Desktop/FRTNBC/data/tnbc_myeloid_preprocessed.h5ad"
OUTPUT_DIR = "outputs/e2e_myeloid"
REPORT_PATH = os.path.join(OUTPUT_DIR, "e2e_myeloid_report.html")


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
        html += note + table
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

def step1_dataset_prep(adata, report, splits):
    """Step 1: Dataset preparation -- pathways, splits, overview."""
    import peach as pc

    html = ""

    # Overview cards
    n_cells, n_genes = adata.shape
    cards = [
        metric_card(n_cells, "Cells"),
        metric_card(n_genes, "Genes (HVG)"),
        metric_card(adata.obsm["X_pca"].shape[1], "PCA dims"),
    ]
    if "treatment" in adata.obs.columns:
        cards.append(metric_card(adata.obs["treatment"].nunique(), "Treatment levels"))
    if "pCR" in adata.obs.columns:
        cards.append(metric_card(adata.obs["pCR"].nunique(), "Response levels"))
    if "subcluster" in adata.obs.columns:
        cards.append(metric_card(adata.obs["subcluster"].nunique(), "Subclusters"))
    html += metric_grid(cards)

    # Split sizes table
    split_rows = []
    for name, mask in splits.items():
        split_rows.append({"Split": name, "N cells": int(mask.sum())})
    split_df = pd.DataFrame(split_rows)
    html += report.df_to_html(split_df, caption="Split sizes")

    # X data summary
    X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    html += report.text(
        f"X range: [{X.min():.2f}, {X.max():.2f}] | "
        f"X mean: {X.mean():.3f} | X std: {X.std():.3f} | "
        f"Note: X is scaled (not raw logcounts)."
    )

    # Pathway scores
    try:
        log.info("Loading Hallmark pathway networks...")
        net = pc.pp.load_pathway_networks(sources=["hallmark"], verbose=False)
        log.info("Computing pathway scores...")
        pc.pp.compute_pathway_scores(adata, net=net, verbose=False)
        n_pathways = adata.obsm["pathway_scores"].shape[1]
        pathway_names = adata.uns.get("pathway_scores_pathways", [])
        html += report.text(f"Pathway scores computed: {n_pathways} Hallmark pathways.")
        if len(pathway_names) > 0:
            html += report.text(f"Example pathways: {', '.join(pathway_names[:5])}...")

        # Pathway score distribution
        fig, ax = plt.subplots(figsize=(10, 3))
        scores = np.asarray(adata.obsm["pathway_scores"])
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

    # Slice PCA to 15 components — z-scored data spreads variance broadly
    # across PCs; PEACH works better with fewer dimensions
    n_pcs_use = 15
    if adata.obsm["X_pca"].shape[1] > n_pcs_use:
        log.info(f"Slicing PCA: {adata.obsm['X_pca'].shape[1]} -> {n_pcs_use} components")
        adata.obsm["X_pca"] = adata.obsm["X_pca"][:, :n_pcs_use]
        adata.varm["PCs"] = adata.varm["PCs"][:, :n_pcs_use]
        html += report.text(f"PCA sliced to {n_pcs_use} components (z-scored data spreads variance broadly).")

    # Prepare training data
    log.info("Preparing training data...")
    pc.pp.prepare_training(adata, batch_size=128)

    report.add_section("Dataset Preparation", html, step_num=1)


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
        inflation_factor_range=[1.0, 1.5, 2.0],
        cv_folds=3,
        max_epochs_cv=15,
        subsample_fraction=0.8,
    )

    # CV results table
    ranked = cv_summary.rank_by_metric("r2")
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

    # Elbow selection: find K where R² improvement drops below threshold
    # Group by K, take best R² per K
    k_to_best = {}
    for r in ranked:
        k = r["hyperparameters"]["n_archetypes"]
        if k not in k_to_best or r["metric_value"] > k_to_best[k]["metric_value"]:
            k_to_best[k] = r
    sorted_ks = sorted(k_to_best.keys())
    k_r2 = [(k, k_to_best[k]["metric_value"]) for k in sorted_ks]

    # Find elbow: first K where marginal R² gain drops below 10% of initial gain
    threshold = 0.01  # default if only one K available
    if len(k_r2) > 1:
        initial_gain = k_r2[1][1] - k_r2[0][1]
        threshold = max(0.01, initial_gain * 0.10)  # 10% of first step
        best_k_val = k_r2[0][0]
        for i in range(1, len(k_r2)):
            marginal = k_r2[i][1] - k_r2[i-1][1]
            if marginal < threshold:
                best_k_val = k_r2[i-1][0]
                break
            best_k_val = k_r2[i][0]
        best = k_to_best[best_k_val]
    else:
        best = ranked[0]

    html += report.text(f"Elbow selection: K={best['hyperparameters']['n_archetypes']} "
                        f"(marginal R² gain threshold: {threshold:.4f})")

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
        fig_space = pc.pl.archetypal_space(adata, color_by="treatment",
                                           title="Archetypal space (treatment)")
        html += safe_plotly_html(report, fig_space, "Archetypal space colored by treatment")
    except Exception as e:
        html += error_html(f"Archetypal space plot failed: {e}")

    try:
        fig_space2 = pc.pl.archetypal_space(adata, color_by="pCR",
                                            title="Archetypal space (response)")
        html += safe_plotly_html(report, fig_space2, "Archetypal space colored by pCR response")
    except Exception as e:
        html += error_html(f"Archetypal space (pCR) failed: {e}")

    # Archetype statistics
    try:
        fig_stats = pc.pl.archetype_statistics(adata)
        if fig_stats is not None:
            html += safe_plotly_html(report, fig_stats, "Archetype statistics")
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

    # R2 barplot (plotly)
    try:
        fig_r2 = pc.pl.r2_barplot(adata, top_n=30, show=False)
        html += safe_plotly_html(report, fig_r2, "Top 30 genes by R-squared (per-archetype |beta|)")
    except Exception as e:
        html += error_html(f"R2 barplot failed: {e}")

    # Coefficient heatmap
    try:
        fig_coef = pc.pl.coefficient_heatmap(adata, top_n=50, show=False)
        html += safe_plotly_html(report, fig_coef, "Vertex coefficient heatmap (top 50 by R-squared)")
    except Exception as e:
        html += error_html(f"Coefficient heatmap failed: {e}")

    # Regression volcano
    try:
        fig_volc = pc.pl.regression_volcano(adata, show=False)
        html += safe_plotly_html(report, fig_volc, "Regression volcano: R-squared vs vertex contrast")
    except Exception as e:
        html += error_html(f"Regression volcano failed: {e}")

    # Interaction heatmap (degree 2)
    try:
        fig_int = pc.pl.interaction_heatmap(adata, top_n=30, show=False)
        html += safe_plotly_html(report, fig_int, "Interaction coefficient heatmap (degree 2)")
    except Exception as e:
        html += error_html(f"Interaction heatmap failed: {e}")

    # Archetype regression dotplot
    try:
        fig_dot = pc.pl.archetype_regression_dotplot(adata, top_n=10, show=False)
        html += safe_plotly_html(report, fig_dot, "Archetype regression dotplot (top 10 per archetype)")
    except Exception as e:
        html += error_html(f"Regression dotplot failed: {e}")

    # Archetype radar
    try:
        fig_radar = pc.pl.archetype_radar(adata, top_n=8, order_by_similarity=True, show=False)
        html += safe_plotly_html(report, fig_radar, "Archetype radar (top features)")
    except Exception as e:
        html += error_html(f"Archetype radar failed: {e}")

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

        # Archetype-feature map
        arch_feat_map = pattern_result.get("archetype_features", {})
        if arch_feat_map:
            map_rows = []
            for k, feats in arch_feat_map.items():
                map_rows.append({
                    "Archetype": f"A{k+1}" if isinstance(k, int) else str(k),
                    "N features": len(feats),
                    "Top features": ", ".join(feats[:8]),
                })
            map_df = pd.DataFrame(map_rows)
            html += report.df_to_html(map_df, caption="Archetype-exclusive feature map")
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
                            pair_type = "gradient"

                        transition = "rising" if gamma > 0 else "falling"

                        interaction_rows.append({
                            "Feature": feat_names[feat_idx],
                            "Pair": f"A{j+1}-A{k+1}",
                            "Type": pair_type,
                            "Transition": transition,
                            "beta_j": f"{beta_j:.3f}",
                            "beta_k": f"{beta_k:.3f}",
                            "gamma": f"{gamma:.3f}",
                            "FDR q": fmt_pval(int_fdr[feat_idx, pair_idx]),
                        })

            if interaction_rows:
                int_df = pd.DataFrame(interaction_rows)
                type_counts = int_df["Type"].value_counts()
                cards = [metric_card(len(interaction_rows), "Significant interactions")]
                for t in ["tradeoff", "cooperative", "transition-enriched", "gradient"]:
                    cards.append(metric_card(int(type_counts.get(t, 0)), t.capitalize()))
                html += metric_grid(cards)

                html += report.text(
                    "Interaction classification: <b>Cooperative</b> = high at both archetypes "
                    "(shared program). <b>Tradeoff</b> = high at one, low at other (distinguishes "
                    "archetypes). <b>Transition-enriched</b> = peaks in blending zone. "
                    "<b>Gradient</b> = moderate signal. Transition direction: rising (\u03b3>0) = gene "
                    "increases along edge; falling (\u03b3<0) = gene decreases.")

                for itype in ["tradeoff", "cooperative", "transition-enriched", "gradient"]:
                    sub = int_df[int_df["Type"] == itype].head(20)
                    if len(sub) > 0:
                        html += report.df_to_html(sub, caption=f"Top {itype} interactions")
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

            # Top 20 pathways by R²
            if len(pw_names) > 0:
                pw_df = pd.DataFrame({"Pathway": pw_names, "R2": pw_r2,
                                      "FDR_q": pw_fdr})
                pw_df = pw_df.sort_values("R2", ascending=False).head(20)
                pw_df["FDR_q"] = pw_df["FDR_q"].apply(fmt_pval)
                html += report.df_to_html(pw_df, caption="Top 20 pathways by R² (pathway simplex regression)")

            # Pathway coefficient heatmap (matplotlib)
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
                    html += report.fig_to_img(fig, caption="Pathway coefficient heatmap (top 25 by max |beta|)")
                    plt.close("all")
            except Exception as e:
                html += error_html(f"Pathway coefficient heatmap failed: {e}")
                plt.close("all")

            # Pathway regression dotplot (exclusive pathways)
            try:
                fig_pw_dot = pc.pl.archetype_regression_dotplot(
                    adata, top_n=10, exclusive_only=True,
                    feature_type="pathways", show=False)
                html += safe_plotly_html(report, fig_pw_dot,
                                         "Pathway regression dotplot (exclusive pathways)")
            except Exception as e:
                html += error_html(f"Pathway dotplot failed: {e}")

            # Pathway radar
            try:
                fig_pw_radar = pc.pl.archetype_radar(
                    adata, top_n=8, feature_type="pathways",
                    order_by_similarity=True, show=False)
                html += safe_plotly_html(report, fig_pw_radar,
                                         "Pathway radar (similarity-ordered)")
            except Exception as e:
                html += error_html(f"Pathway radar failed: {e}")

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

    for col in ["treatment", "pCR", "subcluster"]:
        if col not in adata.obs.columns:
            continue
        log.info(f"Conditional associations: {col}...")
        try:
            cond_df = pc.tl.conditional_associations(adata, obs_column=col, verbose=False)
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
                norm = mcolors.LogNorm(vmin=max(0.1, pivot.min().min()),
                                       vmax=max(10, pivot.max().max()))
                im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", norm=norm)
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
    if "archetypes" in adata.obs.columns and "treatment" in adata.obs.columns:
        try:
            ct = pd.crosstab(adata.obs["archetypes"], adata.obs["treatment"], normalize="index")
            fig, ax = plt.subplots(figsize=(10, 5))
            ct.plot(kind="bar", stacked=True, ax=ax, colormap="Set2", edgecolor="none")
            ax.set_ylabel("Proportion")
            ax.set_title("Treatment composition per archetype")
            ax.legend(title="Treatment", bbox_to_anchor=(1.02, 1), loc="upper left")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            html += report.fig_to_img(fig, caption="Stacked bar: treatment proportions per archetype")
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
                direction = "+" if delta[feat_idx] > 0 else "-"
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
    multi_genes = {
        gene: directions
        for gene, directions in gene_pair_directions.items()
        if len(directions) >= 2
    }
    if multi_genes:
        matrix_rows = []
        for gene_name, gene_dir in sorted(multi_genes.items()):
            row = {"Gene": gene_name}
            for pair_label in all_pair_labels:
                row[pair_label] = gene_dir.get(pair_label, "")
            # Count pairs where this gene is significant
            row["N pairs"] = len(gene_dir)
            matrix_rows.append(row)
        conf_df = pd.DataFrame(matrix_rows).set_index("Gene")
        # Sort by number of significant pairs descending
        conf_df = conf_df.sort_values("N pairs", ascending=False)
        html += report.df_to_html(
            conf_df,
            caption=(
                f"Multi-pair contrast direction matrix ({len(multi_genes)} genes significant in ≥2 pairs; "
                "+ = up in first archetype, - = down)"
            ),
        )
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
                        direction = "+" if pw_delta[feat_idx] > 0 else "-"
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
                    caption="Top 30 significant pathway contrasts (grouped by Direction, sorted by |delta-beta|)",
                )
            else:
                html += report.text("No significant pathway contrasts at FDR < 0.05.")
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
    for src_label, tgt_label in flow_pairs:
        n_src = int((adata.obs["archetypes"] == src_label).sum())
        n_tgt = int((adata.obs["archetypes"] == tgt_label).sum())

        if n_src < 50 or n_tgt < 50:
            flow_rows.append({
                "Source": src_label, "Target": tgt_label,
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
            flow_rows.append({
                "Source": src_label, "Target": tgt_label,
                "N_source": n_src, "N_target": n_tgt,
                "MMD_before": round(mmd_b, 4), "MMD_after": round(mmd_a, 4),
                "MMD_reduction": round(reduction, 4),
                "Status": "OK",
            })
        except Exception as e:
            log.warning(f"Flow {src_label}->{tgt_label} failed: {e}")
            flow_rows.append({
                "Source": src_label, "Target": tgt_label,
                "N_source": n_src, "N_target": n_tgt,
                "MMD_before": float("nan"), "MMD_after": float("nan"),
                "MMD_reduction": float("nan"),
                "Status": f"Failed: {e}",
            })

    flow_df = pd.DataFrame(flow_rows)
    html += report.df_to_html(flow_df, caption="Pairwise flow-based archetype comparison")

    # Build K×K dissimilarity matrix (1 - MMD reduction)
    # Higher MMD reduction = easier to transport = more similar; so 1 - MMD_reduction = dissimilarity
    sim_matrix = np.full((K, K), np.nan)
    for _, row in flow_df.iterrows():
        if row["Status"] == "OK":
            i = arch_labels.index(row["Source"])
            j = arch_labels.index(row["Target"])
            val = row["MMD_reduction"]
            sim_matrix[i, j] = 1.0 - val  # dissimilarity: higher = more different
            sim_matrix[j, i] = 1.0 - val
    np.fill_diagonal(sim_matrix, 0.0)  # self-dissimilarity = 0

    # Heatmap
    try:
        import plotly.graph_objects as go
        arch_short = [f"A{i+1}" for i in range(K)]
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

    # Feature similarity (Spearman on regression coefficients) — keep existing
    log.info("Computing within-fit feature similarity...")
    try:
        sim_result = pc.tl.archetype_feature_similarity(adata)
        n_sig_feat = sim_result.get("n_significant_features", "?")
        html += report.text(f"Spearman \u03c1 computed on {n_sig_feat} FDR-significant (q<0.05) "
                            "vertex \u03b2 coefficients from simplex regression.")

        fig_sim = pc.pl.feature_similarity_heatmap(adata, show=False)
        html += safe_plotly_html(report, fig_sim, "Feature similarity (Spearman \u03c1) heatmap")
    except Exception as e:
        html += error_html(f"Feature similarity failed: {e}")

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
    alpha_rows = []
    for label in arch_labels:
        mask = (adata.obs["archetypes"] == label).values
        cells = X_shifted[mask]
        per_cell_ent = np.array([shannon_entropy(c / c.sum()) for c in cells])
        alpha_rows.append({
            "Archetype": label,
            "N cells": int(mask.sum()),
            "Mean Shannon H": f"{per_cell_ent.mean():.4f}",
            "Median Shannon H": f"{np.median(per_cell_ent):.4f}",
            "Std Shannon H": f"{per_cell_ent.std():.4f}",
        })
    html += report.df_to_html(pd.DataFrame(alpha_rows),
                              caption="Alpha diversity: Shannon entropy of expression per archetype")

    # Beta diversity: Bray-Curtis between archetype mean profiles
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
        pw_alpha_rows = []
        for label in arch_labels:
            mask = (adata.obs["archetypes"] == label).values
            pw_cells = pw_scores[mask]
            pw_var = pw_cells.var(axis=0).mean()
            pw_alpha_rows.append({
                "Archetype": label,
                "Mean pathway score variance": f"{pw_var:.4f}",
            })
        html += report.df_to_html(pd.DataFrame(pw_alpha_rows),
                                  caption="Pathway score diversity per archetype")

    # Weight entropy: how committed are cells to one archetype
    weights = np.asarray(adata.obsm.get("cell_archetype_weights", np.array([])))
    if weights.size > 0:
        w_clipped = np.clip(weights, 1e-10, 1.0)
        weight_entropy = -np.sum(w_clipped * np.log(w_clipped), axis=1)
        entropy_rows = []
        for label in arch_labels:
            mask = (adata.obs["archetypes"] == label).values
            ent = weight_entropy[mask]
            entropy_rows.append({
                "Archetype": label,
                "Mean weight entropy": f"{ent.mean():.4f}",
                "Median": f"{np.median(ent):.4f}",
            })
        html += report.df_to_html(pd.DataFrame(entropy_rows),
                                  caption="Archetype weight entropy (higher = less committed)")

    # Per-condition diversity
    for condition_col in ["treatment", "pCR"]:
        if condition_col not in adata.obs.columns:
            continue
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
    """Step 7: Driver regression (features predict archetype weights)."""
    import peach as pc

    html = ""

    # Use pathway scores if available, else genes
    has_pathways = "pathway_scores" in adata.obsm
    feat_label = "pathway_scores" if has_pathways else None
    feat_matrix_arg = feat_label  # default: pass key string
    feat_names_arg = None  # default: infer from adata

    if not has_pathways:
        html += report.text("Warning: No pathway scores available. "
                            "Driver regression on full gene set may be underdetermined.")

    # Filter pathways by variance to avoid singular design matrix
    if has_pathways:
        pw_scores = adata.obsm.get("pathway_scores")
        if pw_scores is not None:
            pw_var = np.var(pw_scores, axis=0)
            K_arch = adata.obsm.get("cell_archetype_weights", np.empty((0, 5))).shape[1]
            n_keep = min(pw_scores.shape[1], max(20, K_arch * 4))
            top_pw_idx = np.argsort(pw_var)[-n_keep:]
            pw_names_all = adata.uns.get("pathway_scores_pathways",
                                         [f"pw_{i}" for i in range(pw_scores.shape[1])])
            feat_names_arg = [pw_names_all[i] for i in top_pw_idx]
            feat_matrix_arg = pw_scores[:, top_pw_idx]
            log.info(f"  Pathway driver regression: keeping {n_keep}/{pw_scores.shape[1]} "
                     f"pathways by variance (K={K_arch})")
            html += report.text(f"Pathway variance filter: {n_keep}/{pw_scores.shape[1]} pathways "
                                f"retained (top by variance, max(20, K×4)={n_keep}).")

    log.info(f"Running driver regression (feature_matrix={feat_label}, degree=1)...")
    try:
        driver_result = pc.tl.archetype_driver_regression(
            adata,
            feature_matrix=feat_matrix_arg,
            feature_names=feat_names_arg,
            max_degree=1,  # degree=2 creates too many interaction terms
            n_bootstrap=500,
            robust_se=True,
        )

        r2_vals = np.asarray(driver_result["r_squared"])
        K_minus_1 = len(r2_vals)
        main_coefs = np.asarray(driver_result["main_coefficients"])  # [K, n_feat]
        feat_names = list(driver_result.get("feature_names", []))

        html += metric_grid([
            metric_card(f"{r2_vals.mean():.4f}", "Mean R-squared (ILR components)"),
            metric_card(K_minus_1, "ILR dimensions"),
            metric_card(len(feat_names), "Features"),
        ])

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

                html += metric_grid([
                    metric_card(len(shared), "Shared top-50"),
                    metric_card(len(simplex_only), "Simplex-only"),
                    metric_card(len(driver_only), "Driver-only"),
                ])
                overlap_rows = [{"Feature": g, "Source": "shared"} for g in sorted(shared)[:20]]
                overlap_rows += [{"Feature": g, "Source": "simplex-only"} for g in sorted(simplex_only)[:10]]
                overlap_rows += [{"Feature": g, "Source": "driver-only"} for g in sorted(driver_only)[:10]]
                html += report.df_to_html(pd.DataFrame(overlap_rows), caption="Feature overlap: simplex R² top 50 vs driver |β| top 50")

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
        if "pathway_scores" in adata.obsm:
            try:
                log.info("  Component pathway characterization...")
                pw_comp = pc.tl.component_regression(adata, feature_type="pathway_scores")
                pw_comp_regs = pw_comp.get("component_regs", {})
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

            for col in ["treatment", "pCR"]:
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

MIN_CELLS_MODEL = 500
MIN_CELLS_FLOW = 300


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

    for col in ["treatment", "pCR", "subcluster"]:
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
    """Step 10: Per-dose hyperparameter search, model training, annotation."""
    import peach as pc

    html = ""
    dose_adatas = {}

    if "treatment" not in adata.obs.columns:
        html += error_html("No 'treatment' column -- skipping per-dose models.")
        report.add_section("Per-Dose Models", html, step_num=10)
        return dose_adatas

    doses = sorted(adata.obs["treatment"].unique())
    html += report.text(f"Doses: {', '.join(doses)}")

    # Smaller search grid for subsets
    K_range = [3, 4, 5, 6, 7]
    hidden_opts = [[64, 128], [128, 256]]

    summary_rows = []
    for dose in doses:
        mask = (adata.obs["treatment"] == dose).values
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
            ranked = cv.rank_by_metric("r2")
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
                                  caption="Per-dose model comparison")

    # Side-by-side archetypal space scatters
    n_dose = len(dose_adatas)
    if n_dose > 0:
        try:
            fig, axes = plt.subplots(1, n_dose, figsize=(5 * n_dose, 4))
            if n_dose == 1:
                axes = [axes]
            for ax, (dose, sub) in zip(axes, dose_adatas.items()):
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
            fig.suptitle("Per-dose archetypal space", y=1.02)
            fig.tight_layout()
            html += report.fig_to_img(fig, caption="Per-dose archetypal space (first 2 coords)")
            plt.close("all")
        except Exception as e:
            html += error_html(f"Per-dose scatter failed: {e}")
            plt.close("all")

    report.add_section("Per-Dose Models", html, step_num=10)
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
                                         caption=f"{dose}: archetype-exclusive top features (|beta| size, -log10p color)")
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
    """Step 12: Between-dose flow (soft assignment), MMD, feature similarity."""
    import peach as pc

    html = ""
    flow_results = {}

    if "treatment" not in adata.obs.columns:
        html += error_html("No 'treatment' column -- skipping between-dose flow.")
        report.add_section("Between-Dose Flow", html, step_num=12)
        return flow_results

    dose_pairs = _make_dose_pairs()
    available_doses = set(adata.obs["treatment"].unique())

    for src, tgt in dose_pairs:
        if src not in available_doses or tgt not in available_doses:
            html += report.text(f"Skipping {src}->{tgt}: dose not present.")
            continue

        n_src = int((adata.obs["treatment"] == src).sum())
        n_tgt = int((adata.obs["treatment"] == tgt).sum())
        if n_src < MIN_CELLS_FLOW or n_tgt < MIN_CELLS_FLOW:
            html += error_html(f"{src}->{tgt}: insufficient cells ({n_src}, {n_tgt}), skipping.")
            continue

        pair_key = f"{src}_to_{tgt}"
        log.info(f"  Flow: {src} -> {tgt} ({n_src} -> {n_tgt} cells)...")
        html += f"<h4>Flow: {pair_key.replace('_to_', ' → ')}</h4>"
        try:
            fr = pc.tl.flow_within(
                adata,
                source={"treatment": src},
                target={"treatment": tgt},
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

    report.add_section("Between-Dose Flow", html, step_num=12)
    return flow_results


def step13_sinkhorn_flow(adata, flow_results, report):
    """Step 13: OT flow, gene alignment, gene expression change along flow."""
    import peach as pc

    html = ""
    alignment_results = {}

    if not flow_results:
        # Try training new OT flows for each dose pair
        dose_pairs = _make_dose_pairs()
        available_doses = set(adata.obs["treatment"].unique()) if "treatment" in adata.obs.columns else set()
        for src, tgt in dose_pairs:
            if src not in available_doses or tgt not in available_doses:
                continue
            n_src = int((adata.obs["treatment"] == src).sum())
            n_tgt = int((adata.obs["treatment"] == tgt).sum())
            if n_src < MIN_CELLS_FLOW or n_tgt < MIN_CELLS_FLOW:
                continue
            pair_key = f"{src}_to_{tgt}"
            log.info(f"  OT flow: {pair_key} ...")
            try:
                fr = pc.tl.flow_within(
                    adata,
                    source={"treatment": src},
                    target={"treatment": tgt},
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
            align = pc.tl.flow_gene_alignment(adata, fr, n_top=30, per_cell=False, n_permutations=200)
            alignment_results[pair_key] = align

            # Significance filtering
            align_pvals = align.get("alignment_pvalues")
            if align_pvals is not None:
                raw_p = np.asarray(align_pvals)
                sig_mask = raw_p < 0.05
                n_raw_sig = int(sig_mask.sum())
                html += report.text(f"Genes with raw p < 0.05: {n_raw_sig} / {len(raw_p)}")

            fig_bar = pc.pl.gene_alignment_barplot(adata, align, n_top=20, show=False)
            html += safe_plotly_html(report, fig_bar, f"Gene alignment: {pair_key}")
        except Exception as e:
            html += error_html(f"Gene alignment ({pair_key}) failed: {e}")

        try:
            fig_quiv = pc.pl.velocity_quiver(adata, fr, show=False)
            fig_quiv.update_layout(title=f"Flow velocity field: {pair_key.replace('_to_', ' → ')}")
            html += safe_plotly_html(report, fig_quiv, f"Velocity quiver: {pair_key}")
        except Exception as e:
            html += error_html(f"Quiver ({pair_key}) failed: {e}")

    # Gene expression change along flow (gene-space reconstruction)
    if "PCs" in adata.varm:
        for pair_key, fr in flow_results.items():
            html += f"<h4>Flow: {pair_key.replace('_to_', ' → ')}</h4>"
            try:
                source_pca = adata.obsm["X_pca"][fr["source_mask"]]
                transported = fr["transported"]
                delta_pca = transported - source_pca
                loadings = adata.varm["PCs"]
                n_pcs = delta_pca.shape[1]
                delta_expr = delta_pca @ loadings[:, :n_pcs].T  # [n_source, n_genes]
                mean_delta = delta_expr.mean(axis=0)

                # Top genes by expression change along flow
                html += report.text(
                    "Gene expression delta: PCA-reconstructed expression change computed as "
                    "(PCA loadings) \u00d7 (\u0394 PCA coordinates) between transported and source positions. "
                    "Units are log-normalized expression change.")
                sorted_idx = np.argsort(np.abs(mean_delta))[::-1]
                pw_rows = []
                for rank, gi in enumerate(sorted_idx[:20]):
                    pw_rows.append({
                        "Rank": rank + 1,
                        "Gene": adata.var_names[gi],
                        "Mean delta": f"{mean_delta[gi]:.4f}",
                        "Abs delta": f"{abs(mean_delta[gi]):.4f}",
                    })
                html += report.df_to_html(
                    pd.DataFrame(pw_rows),
                    caption=f"Gene expression \u0394 along flow (PCA reconstruction): {pair_key}",
                )

                # Pathway alignment: top/bottom expression change along flow mapped to pathway names
                if "pathway_scores" in adata.obsm:
                    try:
                        pw_names = list(adata.uns.get("pathway_names", []))
                        if not pw_names:
                            # Fall back to pathway_scores_pathways key used by compute_pathway_scores
                            pw_names = list(adata.uns.get("pathway_scores_pathways", []))
                        if pw_names:
                            html += report.text(f"Pathway alignment via PCA reconstruction ({pair_key}):")
                            sorted_genes = np.argsort(mean_delta)
                            top_up = set(adata.var_names[sorted_genes[-100:]])
                            top_down = set(adata.var_names[sorted_genes[:100]])
                            html += report.text(
                                f"  Top 100 upregulated genes along flow: "
                                f"{', '.join(list(top_up)[:8])}..."
                            )
                            html += report.text(
                                f"  Top 100 downregulated genes along flow: "
                                f"{', '.join(list(top_down)[:8])}..."
                            )
                    except Exception as e_pw:
                        html += error_html(f"Pathway alignment ({pair_key}) failed: {e_pw}")

            except Exception as e:
                html += error_html(f"Gene expression change along flow ({pair_key}) failed: {e}")

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

                # Significance diagnostics
                exp_pvals = jac.get("expansion_pvalues")
                exp_fdr = jac.get("expansion_pvalues_fdr")
                n_raw_sig = 0

                if exp_pvals is not None:
                    raw_p = np.asarray(exp_pvals)
                    n_raw_sig = int((raw_p < 0.05).sum())
                    html += report.text(f"Genes with raw p < 0.05: {n_raw_sig} / {len(raw_p)}")

                if exp_fdr is not None:
                    fdr_p = np.asarray(exp_fdr)
                    n_fdr_sig = int((fdr_p < 0.05).sum())
                    html += report.text(f"Genes with FDR q < 0.05: {n_fdr_sig} / {len(fdr_p)}")
                    if n_fdr_sig == 0 and n_raw_sig > 0:
                        html += report.text(
                            "<em>Note: no genes survive FDR correction. Showing raw-p significant genes below.</em>")

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
                    exp_rows.append({"Gene": gene_names[gi], "Expansion": f"{expansion[gi]:.4f}", "Direction": "expanding"})
                for gi in top_contract:
                    if gi not in top_expand:
                        exp_rows.append({"Gene": gene_names[gi], "Expansion": f"{expansion[gi]:.4f}", "Direction": "contracting"})
                html += report.df_to_html(pd.DataFrame(exp_rows),
                                          caption=f"Top expanded/contracted genes: {pair_key}")

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

    # Per-cell expansion violin for top genes from first Jacobian
    log.info(f"Violin plots: {len(jac_results)} Jacobian results available")
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

        try:
            n_show = min(10, len(gene_names))
            fig, axes = plt.subplots(2, 5, figsize=(20, 6))
            axes = axes.flatten()
            for gi in range(n_show):
                ax = axes[gi]
                vals = per_cell[:, gi]
                ax.violinplot(vals, showmedians=True)
                ax.set_title(gene_names[gi], fontsize=9)
                ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
                ax.set_ylabel("Expansion score")
                ax.spines[["top", "right"]].set_visible(False)
            for gi in range(n_show, len(axes)):
                axes[gi].set_visible(False)
            fig.suptitle(f"Per-cell expansion: {pair_key} (>1 = expanding, <1 = contracting)", y=1.02)
            fig.tight_layout()
            html += report.fig_to_img(fig, caption=f"Per-cell expansion violins: {pair_key}")
            plt.close("all")
        except Exception as e:
            html += error_html(f"Expansion violins ({pair_key}) failed: {e}")
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
                                     f"Exclusive features: pCR={resp}")
        except Exception as e:
            html += error_html(f"Dotplot pCR={resp} failed: {e}")

        if "pathway_scores" in sub.obsm:
            try:
                fig_pw = pc.pl.archetype_regression_dotplot(
                    sub, top_n=10, exclusive_only=True,
                    feature_type="pathways", show=False)
                html += safe_plotly_html(report, fig_pw,
                                         f"Exclusive pathways: pCR={resp}")
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
                                             f"Exclusive features: {label}")
                except Exception as e_dot:
                    html += error_html(f"Dotplot {label} failed: {e_dot}")

                if "pathway_scores" in sub.obsm:
                    try:
                        fig_pw = pc.pl.archetype_regression_dotplot(
                            sub, top_n=10, exclusive_only=True,
                            feature_type="pathways", show=False)
                        html += safe_plotly_html(report, fig_pw,
                                                 f"Exclusive pathways: {label}")
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
    report = HTMLReport("PEACH v0.5 -- Myeloid End-to-End Analysis")

    # Load data
    log.info(f"Loading data from {DATA_PATH}...")
    adata = sc.read_h5ad(DATA_PATH)
    log.info(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # Define splits
    splits = define_splits(adata)
    log.info(f"Defined {len(splits)} splits")

    # Storage for cross-step results
    gene_reg = None
    flow_comparison_df = None

    # -- Step 1 --------------------------------------------------------------
    t0 = time.time()
    try:
        step1_dataset_prep(adata, report, splits)
        log.info(f"Step 1 done in {time.time() - t0:.1f}s")
    except Exception as e:
        log.error(f"Step 1 failed: {e}", exc_info=True)
        report.add_section("Dataset Preparation", error_html(f"Step 1 failed: {e}"), step_num=1)

    # Save checkpoint
    safe_save_h5ad(adata,os.path.join(OUTPUT_DIR, "adata_step1.h5ad"))

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
        report.add_section("Per-Dose Models",
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
        report.add_section("Between-Dose Flow",
                           error_html(f"Step 12 failed: {e}"), step_num=12)

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

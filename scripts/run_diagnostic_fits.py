#!/usr/bin/env python
"""Diagnostic PCA-to-archetypal-fit report for HSC and OV datasets.

Lightweight verification that cleaned PCA preprocessing gives reasonable
archetypal fits. Produces a single HTML report with 5 sections per dataset.

Usage: conda run -n archetype python scripts/run_diagnostic_fits.py
"""

import matplotlib
matplotlib.use("Agg")

import base64
import io
import logging
import os
import sys
import time
import traceback
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="plotly")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("diagnostic_fits")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "diagnostic")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HSC_TRAIN = os.path.join(PROJECT_DIR, "data", "paper_part1", "adata_hsc_train.h5ad")
OV_TRAIN = os.path.join(PROJECT_DIR, "data", "paper_part1_ov", "adata_primary_train.h5ad")
OV_META_TRAIN = os.path.join(PROJECT_DIR, "data", "paper_part1_ov", "adata_metastatic_train.h5ad")

# Import shared viz helpers
sys.path.insert(0, SCRIPT_DIR)
from _paper_part1_viz import (
    build_drift_qc_panel,
    convergence_status,
    compute_archetype_to_centroid_distance,
)

# Training config
MAX_EPOCHS_FINAL = 200
EARLY_STOP_PATIENCE = 15

# Dataset-specific KLD weights (from paper scripts)
KLD_WEIGHTS = {"HSC": 0.09, "OV": 0.15}

# Hyperparameter search ranges
HP_RANGES = {
    "HSC": dict(
        n_archetypes_range=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        inflation_factor_range=[0.5, 0.75, 1.0, 1.25, 1.5],
        hidden_dims_options=[[64, 128], [128, 256]],
        cv_folds=3, max_epochs_cv=15,
    ),
    "OV": dict(
        n_archetypes_range=[2, 3, 4, 5, 6, 7, 8, 9],
        inflation_factor_range=[0.5, 0.75, 1.0, 1.25, 1.5],
        hidden_dims_options=[[64, 128], [128, 256]],
        cv_folds=3, max_epochs_cv=15,
    ),
}


# ---------------------------------------------------------------------------
# HTMLReport (same pattern as paper scripts)
# ---------------------------------------------------------------------------
class HTMLReport:
    """Minimal HTML report builder."""

    def __init__(self, title):
        self.title = title
        self.sections = []
        self.start_time = time.time()

    def fig_to_img(self, fig, caption="", dpi=150):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        html = f'<img src="data:image/png;base64,{b64}" style="max-width:100%;">'
        if caption:
            html += f"<p class='caption'>{caption}</p>"
        return html

    def plotly_to_div(self, fig, caption=""):
        import plotly.io as pio
        div = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
        html = f"<div class='plotly-wrap'>{div}</div>"
        if caption:
            html += f"<p class='caption'>{caption}</p>"
        return html

    def df_to_html(self, df, caption="", max_rows=50):
        if len(df) > max_rows:
            df = df.head(max_rows)
        table = df.to_html(classes="styled-table", index=True,
                           float_format=lambda x: f"{x:.4g}", border=0)
        html = ""
        if caption:
            html += f"<p class='caption'><strong>{caption}</strong></p>"
        html += f'<div style="overflow-x: auto; max-width: 100%;">{table}</div>'
        return html

    def text(self, txt):
        return f"<p>{txt}</p>"

    def add_section(self, title, content_html, step_num=None):
        self.sections.append({"title": title, "content": content_html, "step": step_num})

    def save(self, path):
        elapsed = time.time() - self.start_time
        m, s = divmod(int(elapsed), 60)
        sections_html = ""
        for i, sec in enumerate(self.sections):
            tag = f"Fig {sec['step']}: " if sec["step"] else ""
            sections_html += f"""
            <details {'open' if i < 6 else ''}>
                <summary>{tag}{sec['title']}</summary>
                <div class="section-body">{sec['content']}</div>
            </details>"""
        html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>{self.title}</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0 auto; padding: 20px 40px;
       background: #fafafa; color: #222; line-height: 1.6; max-width: 1400px; }}
h1 {{ border-bottom: 3px solid #0072B2; padding-bottom: 10px; color: #0072B2; }}
.meta {{ color: #666; font-size: 0.9em; margin-bottom: 20px; }}
details {{ margin-bottom: 12px; background: #fff; border: 1px solid #ddd; border-radius: 6px; overflow: hidden; }}
summary {{ cursor: pointer; padding: 12px 16px; background: #f0f4f8; font-weight: 600; font-size: 1.05em; border-bottom: 1px solid #ddd; }}
summary:hover {{ background: #e2eaf2; }}
.section-body {{ padding: 16px; }}
.caption {{ color: #555; font-style: italic; font-size: 0.92em; margin-top: 4px; }}
.styled-table {{ border-collapse: collapse; margin: 1em 0; font-size: 0.9em; width: auto; }}
.styled-table th {{ background: #2c3e50; color: white; padding: 8px 12px; text-align: left; }}
.styled-table td {{ padding: 6px 12px; border-bottom: 1px solid #e0e0e0; }}
.styled-table tr:nth-child(even) {{ background: #f8f9fa; }}
.styled-table tr:hover {{ background: #eef2f7; }}
.plotly-wrap {{ margin: 10px 0; }}
.error {{ background: #fff0f0; border-left: 4px solid #c0392b; padding: 10px 14px; margin: 10px 0; border-radius: 4px; }}
.metric-grid {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 12px 0; }}
.metric-card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px 18px; min-width: 120px; text-align: center; }}
.metric-value {{ font-size: 1.4em; font-weight: 700; color: #0072B2; }}
.metric-label {{ font-size: 0.85em; color: #666; margin-top: 4px; }}
</style></head><body>
<h1>{self.title}</h1>
<p class="meta">Generated {time.strftime('%Y-%m-%d %H:%M')} | Runtime: {m}m {s}s</p>
{sections_html}
</body></html>"""
        with open(path, "w") as f:
            f.write(html)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def error_html(msg):
    return f'<div class="error">{msg}</div>'


def metric_card(value, label):
    return (f'<div class="metric-card"><div class="metric-value">{value}</div>'
            f'<div class="metric-label">{label}</div></div>')


def metric_grid(cards):
    return '<div class="metric-grid">' + ''.join(cards) + '</div>'


def safe_plotly_html(report, fig, caption=""):
    try:
        return report.plotly_to_div(fig, caption=caption)
    except Exception as e:
        return error_html(f"Plotly render failed: {e}")


def _get_variance_ratio(adata):
    """Extract or compute PCA variance ratio."""
    # Try adata.uns['pca']
    if "pca" in adata.uns and "variance_ratio" in adata.uns["pca"]:
        return np.array(adata.uns["pca"]["variance_ratio"])
    # Fallback: compute from X_pca column variance
    if "X_pca" in adata.obsm:
        pca = adata.obsm["X_pca"]
        col_var = np.var(pca, axis=0)
        total = col_var.sum()
        if total > 0:
            return col_var / total
    return None


def _cell_type_col(adata):
    """Find the cell type column name."""
    for col in ["cell_type", "Cell_Type", "celltype", "cell_ontology_class"]:
        if col in adata.obs.columns:
            return col
    return None


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def section1_raw_pca(report, adata, dataset_name):
    """Section 1: Raw PCA projection (3D scatter of first 3 PCs)."""
    html = ""
    try:
        pca = adata.obsm["X_pca"]
        n_pcs = pca.shape[1]
        ct_col = _cell_type_col(adata)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        if ct_col is not None:
            categories = adata.obs[ct_col].astype("category")
            cat_codes = categories.cat.codes.values
            unique_cats = categories.cat.categories
            cmap = plt.cm.get_cmap("tab20", len(unique_cats))
            scatter = ax.scatter(
                pca[:, 0], pca[:, 1], pca[:, 2],
                c=cat_codes, cmap=cmap, s=1, alpha=0.4,
            )
            # Legend
            handles = [
                plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=cmap(i), markersize=6, label=str(c))
                for i, c in enumerate(unique_cats)
            ]
            ax.legend(handles=handles, loc="upper left", fontsize=7,
                      bbox_to_anchor=(1.05, 1.0), frameon=True)
        else:
            ax.scatter(pca[:, 0], pca[:, 1], pca[:, 2], s=1, alpha=0.4, c="#0072B2")

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(f"{dataset_name}: Raw PCA embedding (first 3 of {n_pcs} PCs)")
        fig.tight_layout()
        html += report.fig_to_img(fig, caption=f"Raw PCA embedding (first 3 of {n_pcs} PCs), no archetypes yet")
        plt.close(fig)

    except Exception as e:
        html += error_html(f"Section 1 failed: {e}\n{traceback.format_exc()}")
    return html


def section2_pca_summary(report, adata, dataset_name):
    """Section 2: PCA summary -- scree plot, data stats, cell type composition."""
    html = ""
    try:
        pca = adata.obsm["X_pca"]
        n_cells, n_pcs = pca.shape
        n_genes = adata.shape[1]
        vr = _get_variance_ratio(adata)

        # Metric cards
        cards = [
            metric_card(f"{n_cells:,}", "Cells"),
            metric_card(f"{n_genes:,}", "Genes"),
            metric_card(str(n_pcs), "PCs"),
        ]
        if vr is not None:
            cumvar = vr.sum()
            pc1_var = vr[0]
            cards.append(metric_card(f"{cumvar:.3f}", "Cumulative Variance"))
            cards.append(metric_card(f"{pc1_var:.3f}", "PC1 Variance Fraction"))
        html += metric_grid(cards)

        # Scree plot
        if vr is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.bar(range(1, len(vr) + 1), vr, color="#0072B2", alpha=0.7)
            ax1.set_xlabel("PC")
            ax1.set_ylabel("Variance Ratio")
            ax1.set_title(f"{dataset_name}: Scree Plot")

            cumsum = np.cumsum(vr)
            ax2.plot(range(1, len(cumsum) + 1), cumsum, "o-", color="#D55E00")
            ax2.axhline(0.9, linestyle="--", color="gray", alpha=0.5)
            ax2.set_xlabel("Number of PCs")
            ax2.set_ylabel("Cumulative Variance")
            ax2.set_title(f"{dataset_name}: Cumulative Variance")
            fig.tight_layout()
            html += report.fig_to_img(fig, caption="PCA variance explained")
            plt.close(fig)

        # Cell type composition
        ct_col = _cell_type_col(adata)
        if ct_col is not None:
            counts = adata.obs[ct_col].value_counts()
            comp_df = pd.DataFrame({
                "cell_type": counts.index,
                "count": counts.values,
                "fraction": (counts.values / counts.values.sum()),
            })
            html += report.df_to_html(comp_df, caption=f"{dataset_name} cell type composition")

    except Exception as e:
        html += error_html(f"Section 2 failed: {e}\n{traceback.format_exc()}")
    return html


def section3_hyperparam_search(report, adata, dataset_name):
    """Section 3: Hyperparameter search. Returns (html, cv_summary, best_config)."""
    import peach as pc

    html = ""
    cv_summary = None
    best_config = None
    try:
        log.info(f"[{dataset_name}] Preparing training data...")
        pc.pp.prepare_training(adata, batch_size=min(128, adata.shape[0] // 4))

        hp = HP_RANGES[dataset_name]
        log.info(f"[{dataset_name}] Running hyperparameter search: "
                 f"K={hp['n_archetypes_range']}, "
                 f"inflation={hp['inflation_factor_range']}...")

        cv_summary = pc.tl.hyperparameter_search(
            adata,
            n_archetypes_range=hp["n_archetypes_range"],
            hidden_dims_options=hp["hidden_dims_options"],
            inflation_factor_range=hp["inflation_factor_range"],
            cv_folds=hp["cv_folds"],
            max_epochs_cv=hp["max_epochs_cv"],
            subsample_fraction=0.8,
            use_pcha_init=False,
        )

        ranked = cv_summary.rank_by_metric("archetype_r2")
        ranked = [r for r in ranked if r["metric_value"] > -1e6]

        if not ranked:
            html += error_html("No valid configurations found in hyperparameter search.")
            return html, None, None

        # K selection: min K where mean CV R2 > 0.9 (avoids overfitting
        # for marginal R2 gain at higher K). If nothing reaches 0.9,
        # fall back to the top-ranked config.
        R2_THRESHOLD = 0.9
        candidates_above = [
            r for r in ranked if r["metric_value"] >= R2_THRESHOLD
        ]
        if candidates_above:
            # Among configs above threshold, pick the one with smallest K
            # (ties broken by highest R2)
            candidates_above.sort(
                key=lambda r: (r["hyperparameters"]["n_archetypes"], -r["metric_value"])
            )
            best_config = candidates_above[0]
            selection_method = f"min K with R2 >= {R2_THRESHOLD}"
        else:
            best_config = ranked[0]
            selection_method = f"max R2 (none reached {R2_THRESHOLD})"

        best_hp = best_config["hyperparameters"]
        best_k = best_hp["n_archetypes"]
        best_hd = best_hp.get("hidden_dims", [128, 256])
        best_inf = best_hp.get("inflation_factor", 1.0)

        # Ranked results table (top 10)
        cv_rows = []
        for i, r in enumerate(ranked[:10]):
            hp_dict = r["hyperparameters"]
            cv_rows.append({
                "rank": i + 1,
                "K": hp_dict["n_archetypes"],
                "hidden_dims": str(hp_dict.get("hidden_dims", "?")),
                "inflation": hp_dict.get("inflation_factor", "?"),
                "R2": f"{r['metric_value']:.4f}",
                "SE": f"{r.get('std_error', 0):.4f}",
            })
        html += report.df_to_html(pd.DataFrame(cv_rows), caption=f"{dataset_name} CV search (top 10)")

        html += report.text(
            f"<b>Selected ({selection_method})</b>: K={best_k}, "
            f"hidden_dims={best_hd}, inflation={best_inf}. "
            f"R2={best_config['metric_value']:.4f} "
            f"(SE={best_config.get('std_error', 0):.4f}) across "
            f"{hp['cv_folds']} CV folds."
        )

        # Elbow curve
        try:
            fig_elbow = pc.pl.elbow_curve(cv_summary, metrics=["archetype_r2", "rmse"])
            html += safe_plotly_html(report, fig_elbow, f"{dataset_name} elbow curve")
        except Exception as e:
            html += error_html(f"Elbow curve failed: {e}")

    except Exception as e:
        html += error_html(f"Section 3 failed: {e}\n{traceback.format_exc()}")

    return html, cv_summary, best_config


def section4_train_model(report, adata, dataset_name, best_config):
    """Section 4: Train best model + PCHA-off comparison. Returns (html, results)."""
    import peach as pc

    html = ""
    results = None
    try:
        best_hp = best_config["hyperparameters"]
        best_k = best_hp["n_archetypes"]
        best_hd = best_hp.get("hidden_dims", [128, 256])
        best_inf = best_hp.get("inflation_factor", 1.0)
        kld_w = KLD_WEIGHTS[dataset_name]

        # --- Main model: PCHA-off (furthest-sum init) ---
        # PCHA-off consistently matched or outperformed PCHA-on in the r10
        # diagnostic, so we use furthest-sum init as the primary model.
        log.info(f"[{dataset_name}] Training main model (pcha_init=False): K={best_k}, "
                 f"hidden={best_hd}, inflation={best_inf}, kld={kld_w}...")
        results = pc.tl.train_archetypal(
            adata, n_archetypes=best_k, n_epochs=MAX_EPOCHS_FINAL,
            hidden_dims=best_hd, kld_weight=kld_w, archetypal_weight=1.0,
            inflation_factor=best_inf, pcha_init=False,
            model_config={"manifold_weight": 0.001},
            early_stopping=True, early_stopping_patience=EARLY_STOP_PATIENCE,
        )

        r2_main = results.get("final_archetype_r2", float("nan"))
        history = results.get("history", {})
        tc = results.get("training_config", {})
        actual_epochs = tc.get("actual_epochs", MAX_EPOCHS_FINAL)
        early_triggered = tc.get("early_stop_triggered", False)

        html += metric_grid([
            metric_card(f"{r2_main:.4f}", "Archetype R2 (main, no PCHA)"),
            metric_card(str(actual_epochs), "Actual Epochs"),
            metric_card(str(best_k), "K archetypes"),
        ])

        # Training metrics plot
        try:
            fig_tm = pc.pl.training_metrics(history, display=False)
            html += safe_plotly_html(report, fig_tm, f"{dataset_name} training metrics")
        except Exception as e:
            html += error_html(f"Training metrics plot failed: {e}")

        # Drift QC panel (W-A7)
        try:
            results.setdefault("training_config", {}).update({
                "n_archetypes": best_k,
                "hidden_dims": best_hd,
                "inflation_factor": best_inf,
                "use_pcha_init": False,
            })
            drift_html = build_drift_qc_panel(
                [(f"{dataset_name} main (no PCHA)", results)],
                drift_threshold=0.01, converged_window=10,
            )
            html += drift_html
        except Exception as e:
            html += error_html(f"Drift QC panel failed: {e}")

        # Convergence status (W-A8)
        try:
            status, delta_mean = convergence_status(
                history=history, max_epochs=MAX_EPOCHS_FINAL,
                early_stop_triggered=early_triggered,
                actual_epochs=actual_epochs,
            )
            html += report.text(
                f"<b>Convergence status</b>: {status} "
                f"(delta_loss_mean={delta_mean:.5f}, actual_epochs={actual_epochs}, "
                f"early_stop={early_triggered})"
            )
        except Exception as e:
            html += error_html(f"Convergence status failed: {e}")

        # --- PCHA-on comparison (main model is already PCHA-off) ---
        try:
            log.info(f"[{dataset_name}] Training PCHA-on comparison...")
            adata_pcha = adata.copy()
            pc.pp.prepare_training(adata_pcha, batch_size=min(128, adata_pcha.shape[0] // 4))
            res_on = pc.tl.train_archetypal(
                adata_pcha, n_archetypes=best_k, n_epochs=MAX_EPOCHS_FINAL,
                hidden_dims=best_hd, kld_weight=kld_w, archetypal_weight=1.0,
                inflation_factor=best_inf, pcha_init=True,
                model_config={"manifold_weight": 0.001},
                early_stopping=True, early_stopping_patience=EARLY_STOP_PATIENCE,
            )
            r2_on = res_on.get("final_archetype_r2", float("nan"))
            diff = r2_main - r2_on

            comp_rows = [
                {"init": "Furthest-sum (main)", "R2": f"{r2_main:.4f}"},
                {"init": "PCHA (comparison)", "R2": f"{r2_on:.4f}"},
                {"init": "Delta (main - PCHA)", "R2": f"{diff:+.4f}"},
            ]
            html += report.df_to_html(
                pd.DataFrame(comp_rows),
                caption=f"{dataset_name}: furthest-sum vs PCHA init R2 comparison"
            )
            log.info(f"[{dataset_name}] Init comparison: main(no-pcha)={r2_main:.4f}, "
                     f"pcha={r2_on:.4f}, delta={diff:+.4f}")
        except Exception as e:
            html += error_html(f"PCHA-off comparison failed: {e}")

    except Exception as e:
        html += error_html(f"Section 4 failed: {e}\n{traceback.format_exc()}")

    return html, results


def section5_archetypal_space(report, adata, dataset_name, results):
    """Section 5: Archetypal space visualization + centroid distance table."""
    import peach as pc

    html = ""
    try:
        # Coordinates, weights, assignment
        log.info(f"[{dataset_name}] Computing archetypal coordinates and assignments...")
        pc.tl.archetypal_coordinates(adata)
        pc.tl.extract_archetype_weights(adata)
        pc.tl.assign_archetypes(adata)

        # Archetypal space 3D scatter
        try:
            fig_space = pc.pl.archetypal_space(
                adata, color_by="archetypes",
                title=f"{dataset_name}: Archetypal Space (argmax assignments)",
            )
            html += safe_plotly_html(report, fig_space,
                                     f"{dataset_name} archetypal space colored by assignment")
        except Exception as e:
            html += error_html(f"Archetypal space plot failed: {e}")

        # Centroid-to-archetype distance table (W-B10)
        try:
            dist_df = compute_archetype_to_centroid_distance(adata)
            html += report.df_to_html(
                dist_df,
                caption=f"{dataset_name}: Centroid-to-archetype distance (W-B10)"
            )

            # Interpretation
            max_ratio = dist_df["extrapolation_ratio"].max()
            mean_ratio = dist_df["extrapolation_ratio"].mean()
            html += report.text(
                f"<b>Extrapolation check</b>: mean ratio={mean_ratio:.3f}, "
                f"max ratio={max_ratio:.3f}. "
                f"Values < 1 mean archetypes sit among their cells; "
                f"values >> 1 mean archetypes are extrapolated beyond the cell cloud."
            )
        except Exception as e:
            html += error_html(f"Centroid distance table failed: {e}")

        # Archetype positions plot
        try:
            fig_pos = pc.pl.archetype_positions(adata, save_path=None)
            html += report.fig_to_img(fig_pos,
                                       caption=f"{dataset_name} archetype positions in PCA space")
            plt.close(fig_pos)
        except Exception as e:
            html += error_html(f"Archetype positions plot failed: {e}")

        # Archetype assignment counts
        try:
            if "archetypes" in adata.obs.columns:
                counts = adata.obs["archetypes"].value_counts().sort_index()
                counts_df = pd.DataFrame({
                    "archetype": counts.index,
                    "n_cells": counts.values,
                    "fraction": counts.values / counts.values.sum(),
                })
                html += report.df_to_html(counts_df,
                                           caption=f"{dataset_name} archetype assignment counts")
        except Exception as e:
            html += error_html(f"Assignment counts failed: {e}")

    except Exception as e:
        html += error_html(f"Section 5 failed: {e}\n{traceback.format_exc()}")
    return html


# ---------------------------------------------------------------------------
# Main: run all sections for a dataset
# ---------------------------------------------------------------------------

def run_dataset(report, adata, dataset_name):
    """Run all 5 diagnostic sections for one dataset."""
    log.info(f"{'='*60}")
    log.info(f"  {dataset_name} dataset: {adata.shape[0]} cells x {adata.shape[1]} genes")
    log.info(f"{'='*60}")

    # Section 1
    log.info(f"[{dataset_name}] Section 1: Raw PCA projection...")
    html1 = section1_raw_pca(report, adata, dataset_name)
    report.add_section(f"{dataset_name}: Raw PCA Projection", html1, step_num=f"{dataset_name}-1")

    # Section 2
    log.info(f"[{dataset_name}] Section 2: PCA summary...")
    html2 = section2_pca_summary(report, adata, dataset_name)
    report.add_section(f"{dataset_name}: PCA Summary", html2, step_num=f"{dataset_name}-2")

    # Section 3
    log.info(f"[{dataset_name}] Section 3: Hyperparameter search...")
    html3, cv_summary, best_config = section3_hyperparam_search(report, adata, dataset_name)
    report.add_section(f"{dataset_name}: Hyperparameter Search", html3, step_num=f"{dataset_name}-3")

    if best_config is None:
        log.warning(f"[{dataset_name}] No best config found, skipping sections 4-5.")
        report.add_section(f"{dataset_name}: Training (skipped)", error_html("No best config."),
                           step_num=f"{dataset_name}-4")
        report.add_section(f"{dataset_name}: Archetypal Space (skipped)", error_html("No model."),
                           step_num=f"{dataset_name}-5")
        return

    # Section 4
    log.info(f"[{dataset_name}] Section 4: Train best model...")
    html4, results = section4_train_model(report, adata, dataset_name, best_config)
    report.add_section(f"{dataset_name}: Train Best Model", html4, step_num=f"{dataset_name}-4")

    if results is None:
        log.warning(f"[{dataset_name}] Training failed, skipping section 5.")
        report.add_section(f"{dataset_name}: Archetypal Space (skipped)",
                           error_html("Training failed."), step_num=f"{dataset_name}-5")
        return

    # Section 5
    log.info(f"[{dataset_name}] Section 5: Archetypal space visualization...")
    html5 = section5_archetypal_space(report, adata, dataset_name, results)
    report.add_section(f"{dataset_name}: Archetypal Space", html5, step_num=f"{dataset_name}-5")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import anndata as ad

    report = HTMLReport("Diagnostic Archetypal Fits: HSC + OV")
    timestamp = time.strftime("%Y%m%d")
    report_path = os.path.join(OUTPUT_DIR, f"diagnostic_report_{timestamp}.html")

    # --- HSC dataset ---
    try:
        log.info(f"Loading HSC data from {HSC_TRAIN}...")
        adata_hsc = ad.read_h5ad(HSC_TRAIN)
        log.info(f"  HSC: {adata_hsc.shape}")
        run_dataset(report, adata_hsc, "HSC")
    except Exception as e:
        log.error(f"HSC dataset failed: {e}")
        report.add_section("HSC: FATAL ERROR", error_html(f"HSC failed: {e}\n{traceback.format_exc()}"),
                           step_num="HSC-ERR")

    # --- OV (primary) dataset ---
    try:
        log.info(f"Loading OV (primary) data from {OV_TRAIN}...")
        adata_ov = ad.read_h5ad(OV_TRAIN)
        log.info(f"  OV primary: {adata_ov.shape}")
        run_dataset(report, adata_ov, "OV")
    except Exception as e:
        log.error(f"OV dataset failed: {e}")
        report.add_section("OV: FATAL ERROR", error_html(f"OV failed: {e}\n{traceback.format_exc()}"),
                           step_num="OV-ERR")

    # Save report
    report.save(report_path)
    log.info(f"Report saved to: {report_path}")
    log.info("Done.")


if __name__ == "__main__":
    main()

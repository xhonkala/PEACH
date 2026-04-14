#!/usr/bin/env python
"""Full-data fit diagnostic with hyperparameter search.

Uses the loss config validated in the sweep (manifold=0.001, kld=0.01,
sparsity=0.0, archetypal=0.9, pcha_init=False) and runs a hyperparameter
search over inflation_factor and hidden_dims on full HSC + OV data.

Produces an HTML report with:
  1. Hyperparameter search results (ranked table + elbow curve)
  2. Train best model (min K >= 0.9 R2)
  3. Archetypal space via pc.pl.archetypal_space (plotly 3D)
  4. Centroid-to-archetype distance table
  5. PCHA on/off comparison
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fit_diagnostic")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "diagnostic")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from _paper_part1_viz import (
    build_drift_qc_panel,
    convergence_status,
    compute_archetype_to_centroid_distance,
)

# ---- Validated loss config from sweep ----
MANIFOLD_WEIGHT = 0.001
KLD_WEIGHT = 0.01
SPARSITY_WEIGHT = 0.0
ARCHETYPAL_WEIGHT = 0.9
N_EPOCHS = 200
EARLY_STOP_PATIENCE = 20

# ---- Search grid ----
DATASETS = {
    "HSC": {
        "path": os.path.join(PROJECT_DIR, "data", "paper_part1", "adata_hsc_train.h5ad"),
        "k_range": [3, 4, 5, 6, 7, 8, 9],
        "kld_weight": 0.01,
    },
}
INFLATION_RANGE = [0.75, 1.0, 1.25, 1.5]
HIDDEN_DIMS_OPTIONS = [[64, 128], [128, 256], [256, 128, 64]]
CV_FOLDS = 3
MAX_EPOCHS_CV = 20


# ---- HTML helpers ----
class HTMLReport:
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
        return (f'<img src="data:image/png;base64,{b64}" style="max-width:100%;">'
                + (f"<p class='caption'>{caption}</p>" if caption else ""))

    def plotly_to_html(self, fig, caption=""):
        html = fig.to_html(full_html=False, include_plotlyjs="cdn")
        if caption:
            html += f"<p class='caption'>{caption}</p>"
        return html

    def text(self, t):
        return f"<p>{t}</p>"

    def df_to_html(self, df, caption=""):
        h = df.to_html(classes="styled-table", index=False, border=0, escape=False)
        if caption:
            h = f"<p class='caption'><b>{caption}</b></p>" + h
        return h

    def add_section(self, title, content):
        self.sections.append({"title": title, "content": content})

    def save(self, path):
        elapsed = time.time() - self.start_time
        m, s = divmod(int(elapsed), 60)
        sections_html = ""
        for i, sec in enumerate(self.sections):
            sections_html += f"""
            <details {'open' if i < 12 else ''}>
                <summary>{sec['title']}</summary>
                <div class="section-body">{sec['content']}</div>
            </details>"""
        html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>{self.title}</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0 auto; padding: 20px 40px;
       background: #fafafa; color: #222; line-height: 1.6; max-width: 1400px; }}
h1 {{ color: #2c3e50; border-bottom: 2px solid #4a90d9; padding-bottom: 10px; }}
details {{ margin: 10px 0; border: 1px solid #ddd; border-radius: 8px; background: white; }}
summary {{ padding: 12px 16px; cursor: pointer; font-weight: 600; font-size: 1.05em;
           background: #f8f9fa; border-radius: 8px; }}
.section-body {{ padding: 16px; }}
table.styled-table {{ border-collapse: collapse; margin: 10px 0; width: 100%; }}
table.styled-table th {{ background: #4a90d9; color: white; padding: 8px 12px; text-align: left; }}
table.styled-table td {{ border: 1px solid #ddd; padding: 6px 10px; }}
table.styled-table tr:nth-child(even) {{ background: #f2f2f2; }}
.caption {{ color: #666; font-size: 0.9em; margin-top: 4px; }}
.metric-grid {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 12px 0; }}
.metric-card {{ background: #f0f4f8; border-radius: 8px; padding: 12px 16px; min-width: 120px; text-align: center; }}
.metric-card .value {{ font-size: 1.4em; font-weight: 700; color: #2c3e50; }}
.metric-card .label {{ font-size: 0.8em; color: #666; }}
.good {{ background: #d4edda !important; }}
.bad {{ background: #f8d7da !important; }}
</style></head><body>
<h1>{self.title}</h1>
<p class="meta">Runtime: {m}m {s}s</p>
{sections_html}
</body></html>"""
        with open(path, "w") as f:
            f.write(html)


def metric_card(value, label):
    return f'<div class="metric-card"><div class="value">{value}</div><div class="label">{label}</div></div>'

def metric_grid(cards):
    return '<div class="metric-grid">' + "".join(cards) + '</div>'

def error_html(msg):
    return f'<div style="color:red;padding:10px;background:#fff0f0;border-radius:6px;margin:8px 0;">{msg}</div>'


def run_dataset(report, dataset_name, config):
    """Run full diagnostic for one dataset."""
    import anndata as ad
    import peach as pc
    import torch

    log.info(f"{'='*60}")
    log.info(f"  {dataset_name}")
    log.info(f"{'='*60}")

    adata = ad.read_h5ad(config["path"])
    log.info(f"  Loaded: {adata.shape}")
    pc.pp.prepare_training(adata, batch_size=min(128, adata.shape[0] // 4))

    X = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
    data_centroid = X.mean(axis=0)
    data_max_dist = np.linalg.norm(X - data_centroid, axis=1).max()
    kld_w = config["kld_weight"]

    # ---- Section 1: Hyperparameter search ----
    html_search = ""
    best_config = None
    try:
        log.info(f"  Hyperparameter search: K={config['k_range']}, "
                 f"inflation={INFLATION_RANGE}, hidden_dims={HIDDEN_DIMS_OPTIONS}")

        cv_summary = pc.tl.hyperparameter_search(
            adata,
            n_archetypes_range=config["k_range"],
            hidden_dims_options=HIDDEN_DIMS_OPTIONS,
            inflation_factor_range=INFLATION_RANGE,
            cv_folds=CV_FOLDS,
            max_epochs_cv=MAX_EPOCHS_CV,
            subsample_fraction=0.8,
            use_pcha_init=False,
        )

        ranked = cv_summary.rank_by_metric("archetype_r2")
        ranked = [r for r in ranked if r["metric_value"] > -1e6]

        if not ranked:
            html_search += error_html("No valid configs found.")
            report.add_section(f"{dataset_name}: Hyperparameter Search", html_search)
            return

        # Min K >= 0.9 R2
        R2_THRESHOLD = 0.9
        above = [r for r in ranked if r["metric_value"] >= R2_THRESHOLD]
        if above:
            above.sort(key=lambda r: (r["hyperparameters"]["n_archetypes"], -r["metric_value"]))
            best_config = above[0]
            method = f"min K with R2 >= {R2_THRESHOLD}"
        else:
            best_config = ranked[0]
            method = f"max R2 (none reached {R2_THRESHOLD})"

        best_hp = best_config["hyperparameters"]
        best_k = best_hp["n_archetypes"]
        best_hd = best_hp.get("hidden_dims", [128, 256])
        best_inf = best_hp.get("inflation_factor", 1.0)

        # Table
        rows = []
        for i, r in enumerate(ranked[:15]):
            hp = r["hyperparameters"]
            rows.append({
                "rank": i + 1,
                "K": hp["n_archetypes"],
                "hidden_dims": str(hp.get("hidden_dims", "?")),
                "inflation": hp.get("inflation_factor", "?"),
                "R2": f"{r['metric_value']:.4f}",
                "SE": f"{r.get('std_error', 0):.4f}",
            })
        html_search += report.df_to_html(pd.DataFrame(rows),
            caption=f"{dataset_name} CV search (top 15, {CV_FOLDS}-fold, {MAX_EPOCHS_CV} epochs/fold)")
        html_search += report.text(
            f"<b>Selected ({method})</b>: K={best_k}, hidden_dims={best_hd}, "
            f"inflation={best_inf}, R2={best_config['metric_value']:.4f}")

        # Elbow curve
        try:
            fig_elbow = pc.pl.elbow_curve(cv_summary, metrics=["archetype_r2", "rmse"])
            html_search += report.plotly_to_html(fig_elbow,
                caption=f"{dataset_name} elbow curve")
        except Exception as e:
            html_search += error_html(f"Elbow curve failed: {e}")

        log.info(f"  Selected: K={best_k}, hd={best_hd}, inf={best_inf}, "
                 f"R2={best_config['metric_value']:.4f} ({method})")

    except Exception as e:
        html_search += error_html(f"Search failed: {e}\n{traceback.format_exc()}")

    report.add_section(f"{dataset_name}: Hyperparameter Search", html_search)

    if best_config is None:
        return

    # ---- Section 2: Train best model ----
    html_train = ""
    results = None
    best_hp = best_config["hyperparameters"]
    best_k = best_hp["n_archetypes"]
    best_hd = best_hp.get("hidden_dims", [128, 256])
    best_inf = best_hp.get("inflation_factor", 1.0)

    try:
        log.info(f"  Training main model: K={best_k}, hd={best_hd}, inf={best_inf}")
        results = pc.tl.train_archetypal(
            adata, n_archetypes=best_k, n_epochs=N_EPOCHS,
            hidden_dims=best_hd, kld_weight=kld_w, archetypal_weight=ARCHETYPAL_WEIGHT,
            inflation_factor=best_inf, pcha_init=False,
            model_config={"manifold_weight": MANIFOLD_WEIGHT, "sparsity_weight": SPARSITY_WEIGHT},
            early_stopping=True, early_stopping_patience=EARLY_STOP_PATIENCE,
        )

        r2 = results.get("final_archetype_r2", float("nan"))
        history = results.get("history", {})
        tc = results.get("training_config", {})
        actual_epochs = tc.get("actual_epochs", N_EPOCHS)

        html_train += metric_grid([
            metric_card(f"{r2:.4f}", "R2"),
            metric_card(str(best_k), "K"),
            metric_card(str(best_hd), "hidden_dims"),
            metric_card(str(best_inf), "inflation"),
            metric_card(str(actual_epochs), "epochs"),
        ])

        # Training metrics
        try:
            fig_tm = pc.pl.training_metrics(history, display=False)
            html_train += report.plotly_to_html(fig_tm, f"{dataset_name} training curves")
        except Exception as e:
            html_train += error_html(f"Training metrics failed: {e}")

        # Drift QC
        try:
            drift_html = build_drift_qc_panel(
                [(dataset_name, results)], drift_threshold=0.01, converged_window=10)
            html_train += drift_html
        except Exception as e:
            html_train += error_html(f"Drift QC failed: {e}")

        # Convergence
        try:
            status, delta = convergence_status(
                history=history, max_epochs=N_EPOCHS,
                early_stop_triggered=tc.get("early_stop_triggered", False),
                actual_epochs=actual_epochs)
            html_train += report.text(f"<b>Convergence</b>: {status} (delta={delta:.5f})")
        except Exception as e:
            html_train += error_html(f"Convergence check failed: {e}")

        # Hull ratio
        model = results["model"]
        model.eval()
        with torch.no_grad():
            out = model(torch.FloatTensor(X[:min(128, len(X))]))
            Y = out["Y"].cpu().numpy()
        arch_dists = np.linalg.norm(Y - data_centroid, axis=1)
        ratio = arch_dists.max() / data_max_dist
        n_outside = int((arch_dists > data_max_dist).sum())
        html_train += report.text(
            f"<b>Hull check</b>: max arch/data ratio = {ratio:.2f}, "
            f"outside hull = {n_outside}/{best_k}")

        log.info(f"  R2={r2:.4f}, hull ratio={ratio:.2f}, outside={n_outside}/{best_k}")

    except Exception as e:
        html_train += error_html(f"Training failed: {e}\n{traceback.format_exc()}")

    report.add_section(f"{dataset_name}: Train Best Model", html_train)

    if results is None:
        return

    # ---- Section 3: Archetypal space (plotly) ----
    html_arch = ""
    try:
        pc.tl.archetypal_coordinates(adata, verbose=False)
        pc.tl.extract_archetype_weights(adata, verbose=False)
        pc.tl.assign_archetypes(adata, verbose=False)

        fig_space = pc.pl.archetypal_space(
            adata, color_by="archetypes",
            title=f"{dataset_name}: Archetypal Space (K={best_k})")
        html_arch += report.plotly_to_html(fig_space,
            f"{dataset_name} archetypal space — archetypes should sit on/near the data cloud")

        # Assignment counts
        if "archetypes" in adata.obs:
            counts = adata.obs["archetypes"].value_counts()
            html_arch += report.df_to_html(
                counts.reset_index().rename(columns={"index": "archetype", "archetypes": "archetype", "count": "n_cells"}),
                caption="Cell assignment counts (bin_prop)")

    except Exception as e:
        html_arch += error_html(f"Archetypal space failed: {e}\n{traceback.format_exc()}")

    report.add_section(f"{dataset_name}: Archetypal Space", html_arch)

    # ---- Section 4: Centroid distance table ----
    html_cd = ""
    try:
        cd_df = compute_archetype_to_centroid_distance(
            adata, obs_key="archetypes", pca_key="X_pca")
        display_cd = cd_df.copy()
        for col in ["archetype_position_norm", "centroid_distance",
                     "data_mean_distance", "bin_radius", "extrapolation_ratio"]:
            if col in display_cd:
                display_cd[col] = display_cd[col].apply(
                    lambda x: "NaN" if pd.isna(x) else f"{x:.4f}")
        html_cd += report.df_to_html(display_cd,
            caption="Archetype-to-centroid distance (centroid of binned cells)")
    except Exception as e:
        html_cd += error_html(f"Centroid distance failed: {e}")

    report.add_section(f"{dataset_name}: Centroid Distance", html_cd)

    # ---- Section 5: PCHA comparison ----
    html_comp = ""
    try:
        log.info(f"  Training PCHA-on comparison...")
        adata_pcha = adata.copy()
        pc.pp.prepare_training(adata_pcha, batch_size=min(128, adata_pcha.shape[0] // 4))
        res_pcha = pc.tl.train_archetypal(
            adata_pcha, n_archetypes=best_k, n_epochs=N_EPOCHS,
            hidden_dims=best_hd, kld_weight=kld_w, archetypal_weight=ARCHETYPAL_WEIGHT,
            inflation_factor=best_inf, pcha_init=True,
            model_config={"manifold_weight": MANIFOLD_WEIGHT, "sparsity_weight": SPARSITY_WEIGHT},
            early_stopping=True, early_stopping_patience=EARLY_STOP_PATIENCE,
        )
        r2_pcha = res_pcha.get("final_archetype_r2", float("nan"))
        r2_main = results.get("final_archetype_r2", float("nan"))
        diff = r2_main - r2_pcha
        comp_rows = [
            {"init": "Furthest-sum (main)", "R2": f"{r2_main:.4f}"},
            {"init": "PCHA", "R2": f"{r2_pcha:.4f}"},
            {"init": "Delta", "R2": f"{diff:+.4f}"},
        ]
        html_comp += report.df_to_html(pd.DataFrame(comp_rows),
            caption=f"{dataset_name}: init comparison")
        log.info(f"  PCHA comparison: main={r2_main:.4f}, pcha={r2_pcha:.4f}, delta={diff:+.4f}")
    except Exception as e:
        html_comp += error_html(f"PCHA comparison failed: {e}")

    report.add_section(f"{dataset_name}: PCHA Comparison", html_comp)


def main():
    from datetime import datetime
    tag = datetime.now().strftime("%Y%m%d")
    report = HTMLReport(f"Fit Diagnostic: HSC + OV ({tag})")

    # Config summary
    report.add_section("Config", f"""<pre>
Loss weights (from sweep):
  manifold_weight = {MANIFOLD_WEIGHT}
  kld_weight      = {KLD_WEIGHT} (HSC) / {KLD_WEIGHT} (OV)
  sparsity_weight = {SPARSITY_WEIGHT}
  archetypal_weight = {ARCHETYPAL_WEIGHT}

Search grid:
  inflation_factor: {INFLATION_RANGE}
  hidden_dims:      {HIDDEN_DIMS_OPTIONS}
  cv_folds:         {CV_FOLDS}
  max_epochs_cv:    {MAX_EPOCHS_CV}

Training:
  n_epochs:         {N_EPOCHS}
  early_stopping:   patience={EARLY_STOP_PATIENCE}
  pcha_init:        False (main), True (comparison)
  K selection:      min K >= 0.9 R2
</pre>""")

    for name, cfg in DATASETS.items():
        try:
            run_dataset(report, name, cfg)
        except Exception as e:
            log.exception(f"{name} failed")
            report.add_section(f"{name}: FATAL ERROR",
                error_html(f"{e}\n{traceback.format_exc()}"))

    report_path = os.path.join(OUTPUT_DIR, f"fit_diagnostic_{tag}.html")
    report.save(report_path)
    log.info(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()

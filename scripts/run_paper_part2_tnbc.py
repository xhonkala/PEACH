"""Paper Part 2 — Step 1: Global archetypal fit on TNBC tumor cells.

*** PROTOTYPE — Step 1 of 5 in Paper Part 2 ***

This script stands up the "does R1/R2/NR segregate in a single global
archetype space?" analysis only. It does NOT cover:

  Step 2 — Per-timepoint × per-response models (6 total): specialist
           selection via archetype relatedness (MMD, Spearman, Wald).
           Uses the global fit from this script as a reference frame.
  Step 3 — Flow along treatments (Base -> PD1 -> RTPD1): expanding /
           contracting features, stress-gene subset (Fig 4C).
  Step 4 — R vs NR contrasts per treatment + per-patient centroid
           trajectories in global space (Fig 5).
  Step 5 — Held-out-patient prediction via LOPO lasso (separate script).

Structure mirrors run_paper_part1_hsc.py: Phase 1 training, Phase 2 Fig 3A,
Phase 3 Fig 3B+3C. Helpers imported from _paper_part1_{prep,viz}.py.

SUBSAMPLE_FRACTION = 0.2 for prototyping; flip to 1.0 for production.
"""
from __future__ import annotations

import glob
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import peach as pc  # noqa: E402

from _paper_part1_viz import (  # noqa: E402
    build_archetype_char_table,
    build_archetype_hypergeometric_tables,
    build_distance_heatmaps,
    build_diversity_block,
    build_drift_qc_panel,
    build_holdout_projection_qc,
    build_response_timepoint_colormap,
    build_segregation_ratio,
    compute_w2_archetype_distance,
    convergence_status,
    dotplot_figsize,
)
from stress_genes.load_stress_genes import STRESS_GENES_FLAT  # noqa: E402


# ============================================================================
# Config
# ============================================================================
DATA_DIR = REPO_ROOT / "data" / "paper_part2"
OUTPUT_DIR = REPO_ROOT / "outputs" / "paper_part2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUBSAMPLE_FRACTION = 0.2
SUBSAMPLE_STRATIFY = "response_group"

MAX_EPOCHS_FINAL = 200
EARLY_STOP_PATIENCE = 15
N_PCS = 12

K_RANGE = list(range(3, 11))
HIDDEN_DIMS_OPTIONS = [[64, 128], [128, 256], [256, 128, 64]]
INFLATION_FACTOR_RANGE = [0.75, 1.0, 1.25, 1.5]

MODEL_CONFIG = {
    "manifold_weight": 0.005,
    "kld_weight": 0.01,
    "sparsity_weight": 0.0,
    "archetypal_weight": 0.9,
}

FDR_THRESHOLD = 0.05
EXCLUSIVE_RATIO_THRESHOLD = 2.5
FIG3C_GATE_THRESHOLD = 1.3

# Auto-increment rev number for today
_DATE_TAG = time.strftime("%Y%m%d")
_existing = sorted(glob.glob(str(OUTPUT_DIR / f"part2_report_{_DATE_TAG}_r*.html")))
_REV = len(_existing) + 1
REPORT_PATH = OUTPUT_DIR / f"part2_report_{_DATE_TAG}_r{_REV}.html"


# ============================================================================
# HTMLReport (inlined; mirrors run_paper_part1_hsc.py)
# ============================================================================


class HTMLReport:
    """Minimal single-file HTML report builder with <details> sections."""

    def __init__(self, title: str):
        self.title = title
        self.sections: list[str] = []

    def text(self, s: str) -> None:
        self.sections.append(f"<p>{s}</p>")

    def add_section(self, title: str, html: str, step_num: int | None = None, open_by_default: bool = False) -> None:
        num = f"{step_num}. " if step_num is not None else ""
        attr = "open" if open_by_default else ""
        self.sections.append(
            f'<details {attr}><summary><h2 style="display:inline">{num}{title}</h2></summary>{html}</details>'
        )

    def fig_to_img(self, fig, caption: str, dpi: int = 150) -> str:
        import base64, io
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f'<figure><img src="data:image/png;base64,{b64}"/><figcaption>{caption}</figcaption></figure>'

    def plotly_to_div(self, fig, caption: str) -> str:
        inline = fig.to_html(full_html=False, include_plotlyjs="cdn")
        return f"<figure>{inline}<figcaption>{caption}</figcaption></figure>"

    def df_to_html(self, df, caption: str, max_rows: int = 50) -> str:
        try:
            html = df.head(max_rows).to_html(index=False, float_format=lambda x: f"{x:.3g}")
        except Exception as e:
            html = f"<em>Error rendering table: {e}</em>"
        return f"<figure>{html}<figcaption>{caption}</figcaption></figure>"

    def save(self, path: "Path") -> None:
        body = "".join(self.sections)
        html = f"""<!DOCTYPE html><html><head>
<meta charset="utf-8"><title>{self.title}</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;max-width:1200px}}
h1{{border-bottom:2px solid #333;padding-bottom:4px}}
details{{margin:12px 0;padding:8px;border:1px solid #ddd;border-radius:6px}}
figure{{margin:12px 0}}img{{max-width:100%;height:auto}}
table{{border-collapse:collapse;margin:8px 0}}
th,td{{border:1px solid #ccc;padding:4px 8px;font-size:0.9em}}
th{{background:#f3f4f6}}</style>
</head><body><h1>{self.title}</h1>{body}</body></html>"""
        Path(path).write_text(html)


def error_html(msg: str) -> str:
    return f'<div style="padding:8px;background:#fee;border-left:4px solid #c33;color:#900">{msg}</div>'


def metric_card(label: str, value, fmt: str = ".3g") -> str:
    try:
        rendered = format(value, fmt)
    except (TypeError, ValueError):
        rendered = str(value)
    return (f'<div style="display:inline-block;padding:8px;margin:4px;'
            f'border:1px solid #ddd;border-radius:6px;min-width:140px">'
            f'<div style="color:#666;font-size:0.8em">{label}</div>'
            f'<div style="font-weight:bold;font-size:1.2em">{rendered}</div></div>')


def metric_grid(cards: list) -> str:
    return f'<div style="display:flex;flex-wrap:wrap">{"".join(cards)}</div>'


# ============================================================================
# Helpers (script-local; don't promote)
# ============================================================================


def _stratified_subsample(adata, frac: float, stratify_col: str, seed: int = 0):
    """Per-stratum subsample. Copies the Part 1 helper so this script is standalone."""
    rng = np.random.default_rng(seed)
    keep_idx = []
    for level in adata.obs[stratify_col].unique():
        pool = np.where(adata.obs[stratify_col].values == level)[0]
        n = max(1, int(round(len(pool) * frac)))
        keep_idx.append(rng.choice(pool, size=min(n, len(pool)), replace=False))
    keep_idx = np.sort(np.concatenate(keep_idx))
    return adata[keep_idx].copy()


def _smallest_k_above_threshold(cv_summary, threshold: float) -> int:
    """Return smallest K in CV summary with mean_archetype_r2 >= threshold,
    else the K with best R². Mirrors Part 1 convention."""
    rows = cv_summary.summary_df.copy()
    rows = rows.sort_values("n_archetypes")
    above = rows[rows["mean_archetype_r2"] >= threshold]
    if len(above):
        return int(above.iloc[0]["n_archetypes"])
    return int(rows.sort_values("mean_archetype_r2", ascending=False).iloc[0]["n_archetypes"])


# ============================================================================
# Phase 1
# ============================================================================


def phase1_train_model(report: HTMLReport):
    """Load train/holdout, CV search, final fit, drift+holdout QC."""
    t_phase = time.time()
    adata_train = sc.read_h5ad(DATA_DIR / "adata_tnbc_train.h5ad")
    adata_holdout = sc.read_h5ad(DATA_DIR / "adata_tnbc_holdout.h5ad")
    print(f"  phase1: loaded train={adata_train.shape} holdout={adata_holdout.shape}")

    if SUBSAMPLE_FRACTION < 1.0:
        adata_train = _stratified_subsample(
            adata_train, SUBSAMPLE_FRACTION, SUBSAMPLE_STRATIFY, seed=42
        )
        print(f"  phase1: subsampled train -> {adata_train.shape}")

    # 1a — CV search
    cv = pc.tl.hyperparameter_search(
        adata_train,
        pca_key="X_pca",
        n_archetypes_range=K_RANGE,
        hidden_dims_options=HIDDEN_DIMS_OPTIONS,
        inflation_factor_range=INFLATION_FACTOR_RANGE,
        use_pcha_init=False,
        max_epochs=20,
        n_folds=5,
    )
    K_pick = _smallest_k_above_threshold(cv, threshold=0.9)
    best_df = cv.summary_df.sort_values("mean_archetype_r2", ascending=False)
    best_row = best_df[best_df["n_archetypes"] == K_pick].iloc[0]
    print(f"  phase1: CV picked K={K_pick} from {K_RANGE}")

    # 1b — final fit
    res = pc.tl.train_archetypal(
        adata_train,
        n_archetypes=K_pick,
        pca_key="X_pca",
        n_epochs=MAX_EPOCHS_FINAL,
        early_stop_patience=EARLY_STOP_PATIENCE,
        pcha_init=True,
        model_config={
            **MODEL_CONFIG,
            "hidden_dims": list(best_row["hidden_dims"]),
            "inflation_factor": float(best_row["inflation_factor"]),
        },
    )

    # 1c — extract coords + assignments on train
    pc.tl.archetypal_coordinates(adata_train, training_results=res)
    pc.tl.assign_archetypes(adata_train, percentage_per_archetype=0.15)

    # 1d — project holdout through trained model
    model = res.get("model") or res.get("final_model")
    pc.tl.extract_archetype_weights(adata_holdout, model=model, pca_key="X_pca")
    pc.tl.archetypal_coordinates(adata_holdout, training_results=res)

    # 1e — Phase 1 section
    html_parts: list = []

    # Config card grid
    cards = [
        metric_card("K (picked)", K_pick, "d"),
        metric_card("N cells (train)", adata_train.n_obs, "d"),
        metric_card("N cells (holdout)", adata_holdout.n_obs, "d"),
        metric_card("N PCs", N_PCS, "d"),
        metric_card("Max epochs", MAX_EPOCHS_FINAL, "d"),
        metric_card("Subsample", SUBSAMPLE_FRACTION, ".2f"),
        metric_card("Train R²", res.get("final_archetype_r2", float("nan")), ".3f"),
    ]
    html_parts.append(metric_grid(cards))

    # CV summary
    html_parts.append(report.df_to_html(cv.summary_df, "CV search summary", max_rows=50))

    # Drift / stability
    try:
        drift_html = build_drift_qc_panel([res], drift_threshold=0.05, converged_window=10)
        html_parts.append(drift_html)
    except Exception as e:
        html_parts.append(error_html(f"drift QC failed: {e}"))

    # Convergence flag
    try:
        status = convergence_status(
            res["history"], max_epochs=MAX_EPOCHS_FINAL,
            early_stop_triggered=res.get("early_stopped", False),
            actual_epochs=len(res["history"].get("loss", [])),
        )
        html_parts.append(
            f"<p><strong>Convergence status:</strong> {status['status']} "
            f"(Δloss={status.get('delta_mean', float('nan')):.4g})</p>"
        )
    except Exception as e:
        html_parts.append(error_html(f"convergence_status failed: {e}"))

    # Holdout projection QC
    try:
        archetype_positions = np.asarray(res["archetype_coords"])
        weights_train = adata_train.obsm["cell_archetype_weights"]
        weights_holdout = adata_holdout.obsm["cell_archetype_weights"]
        recon_train = weights_train @ archetype_positions
        recon_holdout = weights_holdout @ archetype_positions
        qc = build_holdout_projection_qc(
            adata_train.obsm["X_pca"], recon_train,
            adata_holdout.obsm["X_pca"], recon_holdout,
            archetype_positions,
        )
        html_parts.append(metric_grid([
            metric_card("Train R² (manual)", qc["train_r2"], ".3f"),
            metric_card("Holdout R²", qc["holdout_r2"], ".3f"),
            metric_card("Holdout mean NN dist", qc["holdout_mean_nn_dist"], ".3f"),
            metric_card("Holdout median NN dist", qc["holdout_median_nn_dist"], ".3f"),
        ]))
    except Exception as e:
        html_parts.append(error_html(f"holdout projection QC failed: {e}"))

    # PC1 correlation scan
    try:
        pc1 = np.asarray(adata_train.obsm["X_pca"][:, 0])
        from scipy.stats import spearmanr
        rows_pc1 = []
        for col in ("cohort", "treatment", "response_group"):
            codes = adata_train.obs[col].astype("category").cat.codes.values
            rho, p = spearmanr(pc1, codes)
            rows_pc1.append({"covariate": col, "spearman_rho": rho, "p": p})
        for col in ("total_counts", "percent_mito", "percent_ribo"):
            if col in adata_train.obs.columns:
                rho, p = spearmanr(pc1, adata_train.obs[col].values)
                rows_pc1.append({"covariate": col, "spearman_rho": rho, "p": p})
        pc1_df = pd.DataFrame(rows_pc1).sort_values("spearman_rho",
                                                     key=lambda s: s.abs(), ascending=False)
        html_parts.append(report.df_to_html(
            pc1_df, "PC1 correlation scan (Spearman). Large |ρ| on cohort = batch-like PC1."
        ))
    except Exception as e:
        html_parts.append(error_html(f"PC1 scan failed: {e}"))

    report.add_section("Phase 1: Training + QC",
                        "\n".join(html_parts),
                        step_num=1, open_by_default=True)
    print(f"  phase1: done in {time.time() - t_phase:.1f}s")
    return adata_train, adata_holdout, res


# ============================================================================
# Phase 2 and Phase 3 stubs — filled in Tasks 12 and 13
# ============================================================================


def phase2_figure3a(adata_train, adata_holdout, res, report: HTMLReport):
    """Fig 3A — global archetype space. Implemented in Task 12."""
    report.add_section("Phase 2: Fig 3A (stub)",
                        "<p><em>Pending Task 12.</em></p>", step_num=2)


def phase3_figure3bc(adata_train, res, report: HTMLReport):
    """Fig 3B + gated Fig 3C. Implemented in Task 13."""
    report.add_section("Phase 3: Fig 3B/3C (stub)",
                        "<p><em>Pending Task 13.</em></p>", step_num=3)


# ============================================================================
# main
# ============================================================================


def main() -> None:
    report = HTMLReport(f"Paper Part 2 Step 1 — TNBC Global Fit ({_DATE_TAG} r{_REV})")
    report.text(
        "<strong>Prototype run — Step 1 of 5 in Paper Part 2.</strong> "
        "This report covers the global archetypal fit on TNBC tumor cells. "
        "Steps 2–4 (per-timepoint models, flow, R vs NR contrasts) and "
        "Step 5 (held-out prediction) are separate scripts and not yet implemented. "
        f"SUBSAMPLE_FRACTION = {SUBSAMPLE_FRACTION}; flip to 1.0 for production."
    )
    t0 = time.time()

    adata_train, adata_holdout, res = phase1_train_model(report)
    phase2_figure3a(adata_train, adata_holdout, res, report)
    phase3_figure3bc(adata_train, res, report)

    report.save(REPORT_PATH)
    print(f"\n[main] total elapsed: {time.time() - t0:.1f}s")
    print(f"[main] report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()

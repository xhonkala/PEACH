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
sys.path.insert(0, str(REPO_ROOT))
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


def regression_to_long_df(reg_result, *, y_col="gene", exclusive_only=False,
                            exclusive_threshold=2.5, top_n_per_archetype=10,
                            fdr_threshold=0.05, degree=1):
    """Convert simplex regression dict → long-format DataFrame for pc.pl.dotplot.

    Copied verbatim from run_paper_part1_hsc.py to avoid importing that
    active-iteration module. Keep in sync if the Part 1 helper changes —
    or promote to _paper_part1_viz.py in a later iteration.
    """
    feat_names = list(reg_result.get("feature_names", []))
    coefs = np.asarray(reg_result.get("vertex_coefficients", []))
    pvals = np.asarray(reg_result.get("vertex_pvalues", []))
    fdrs = np.asarray(reg_result.get("vertex_pvalues_fdr", []))

    if degree == 1:
        r2 = np.asarray(reg_result.get("r_squared_degree1", []))
        per_arch_sig_fdr = fdrs
        per_feat_sig_fdr = None
    elif degree == 2:
        dc = reg_result.get("degree_comparison", {}) or {}
        d2 = dc.get("degree_2", {}) or {}
        r2 = np.asarray(d2.get("delta_r2", []))
        if r2.size == 0:
            r2_total = np.asarray(reg_result.get("r_squared_degree2", []))
            r2_d1 = np.asarray(reg_result.get("r_squared_degree1", []))
            if r2_total.size > 0 and r2_d1.size > 0:
                r2 = r2_total - r2_d1
            else:
                r2 = np.asarray(d2.get("r_squared", []))
        per_feat_sig_fdr = np.asarray(d2.get("incremental_p_fdr", []))
        per_arch_sig_fdr = None
    else:
        raise ValueError(f"degree must be 1 or 2 — got {degree}")

    if coefs.size == 0 or len(feat_names) == 0 or r2.size == 0:
        return pd.DataFrame()

    n_feat, K = coefs.shape

    if exclusive_only:
        abs_coefs = np.abs(coefs)
        sorted_abs = np.sort(abs_coefs, axis=1)[:, ::-1]
        max_c = sorted_abs[:, 0]
        second_c = sorted_abs[:, 1] if K > 1 else np.zeros(n_feat)
        second_c_safe = np.where(second_c < 1e-10, 1e-10, second_c)
        ratio = max_c / second_c_safe
        keep_feat_mask = ratio >= exclusive_threshold
    else:
        keep_feat_mask = np.ones(n_feat, dtype=bool)

    argmax_arch = np.argmax(np.abs(coefs), axis=1)

    rows = []
    for fi in range(n_feat):
        if not keep_feat_mask[fi]:
            continue
        if per_feat_sig_fdr is not None:
            if fi >= per_feat_sig_fdr.size:
                continue
            feat_fdr = float(per_feat_sig_fdr[fi])
            if feat_fdr > fdr_threshold:
                continue
        for a in range(K):
            if per_arch_sig_fdr is not None:
                p = float(pvals[fi, a]) if pvals.size else 1.0
                f = float(per_arch_sig_fdr[fi, a]) if per_arch_sig_fdr.size else 1.0
                if f > fdr_threshold:
                    continue
            else:
                f = float(per_feat_sig_fdr[fi])
                p = f
            rows.append({
                y_col: feat_names[fi],
                "archetype": f"archetype_{a}",
                "mean_archetype": float(np.abs(coefs[fi, a])),
                "signed_coef": float(coefs[fi, a]),
                "pvalue": p,
                "pvalue_fdr": f,
                "r_squared": float(r2[fi]),
                "argmax_archetype": int(argmax_arch[fi]),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    keep_mask = np.zeros(len(df), dtype=bool)
    for a in range(K):
        is_argmax = df["argmax_archetype"] == a
        if not is_argmax.any():
            continue
        sub = df[is_argmax].drop_duplicates(subset=[y_col])
        top_feats = sub.nlargest(top_n_per_archetype, "r_squared")[y_col].values
        keep_mask |= df[y_col].isin(top_feats)
    df = df[keep_mask].reset_index(drop=True)
    return df


def _pick_best_from_cv(cv_summary, r2_threshold: float = 0.9):
    """Return (best_entry, ranked_list) mirroring Part 1 convention.

    best_entry is the smallest-K entry with metric_value >= r2_threshold;
    otherwise the top ranked entry. Each entry is a dict with keys
    'hyperparameters' and 'metric_value'.
    """
    ranked = cv_summary.rank_by_metric("archetype_r2")
    ranked = [r for r in ranked if r.get("metric_value", float("-inf")) > -1e6]
    above = [r for r in ranked if r["metric_value"] >= r2_threshold]
    if above:
        above.sort(key=lambda r: (r["hyperparameters"]["n_archetypes"],
                                    -r["metric_value"]))
        return above[0], ranked
    return ranked[0], ranked


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

    # 1a — CV search (matches Part 1 conventions: prepare_training first,
    # cv_folds + max_epochs_cv, pick via rank_by_metric)
    pc.pp.prepare_training(adata_train, batch_size=min(128, adata_train.shape[0] // 4))
    cv = pc.tl.hyperparameter_search(
        adata_train,
        pca_key="X_pca",
        n_archetypes_range=K_RANGE,
        hidden_dims_options=HIDDEN_DIMS_OPTIONS,
        inflation_factor_range=INFLATION_FACTOR_RANGE,
        use_pcha_init=False,
        cv_folds=3,
        max_epochs_cv=20,
        subsample_fraction=0.8,
    )
    best_entry, ranked = _pick_best_from_cv(cv, r2_threshold=0.9)
    K_pick = int(best_entry["hyperparameters"]["n_archetypes"])
    hd_pick = list(best_entry["hyperparameters"].get("hidden_dims", [128, 256]))
    inf_pick = float(best_entry["hyperparameters"].get("inflation_factor", 1.0))
    print(f"  phase1: CV picked K={K_pick} hidden={hd_pick} "
          f"inflation={inf_pick} (R²={best_entry['metric_value']:.3f})")

    # 1b — final fit (Part 1 conventions: explicit kwargs, hidden_dims + kld
    # + archetypal + inflation at top level, only manifold_weight inside model_config)
    res = pc.tl.train_archetypal(
        adata_train,
        n_archetypes=K_pick,
        pca_key="X_pca",
        n_epochs=MAX_EPOCHS_FINAL,
        hidden_dims=hd_pick,
        inflation_factor=inf_pick,
        kld_weight=MODEL_CONFIG["kld_weight"],
        archetypal_weight=MODEL_CONFIG["archetypal_weight"],
        pcha_init=True,
        early_stopping=True,
        early_stopping_patience=EARLY_STOP_PATIENCE,
        model_config={"manifold_weight": MODEL_CONFIG["manifold_weight"]},
    )

    # 1c — extract coords + weights + assignments on train.
    # `train_archetypal` has already stored the model at adata.uns['trained_model']
    # and archetype_coordinates is read from adata.uns['archetype_coordinates'].
    pc.tl.archetypal_coordinates(adata_train, verbose=False)
    pc.tl.extract_archetype_weights(adata_train, verbose=False)
    pc.tl.assign_archetypes(adata_train, percentage_per_archetype=0.15, verbose=False)

    # 1d — project holdout through trained model. Transfer the trained_model +
    # archetype_coordinates entries so the same helpers work on the holdout adata.
    model = res.get("model") or res.get("final_model")
    for key in ("trained_model", "archetype_coordinates"):
        if key in adata_train.uns:
            adata_holdout.uns[key] = adata_train.uns[key]
    pc.tl.extract_archetype_weights(adata_holdout, model=model,
                                      pca_key="X_pca", verbose=False)
    pc.tl.archetypal_coordinates(adata_holdout, verbose=False)
    pc.tl.assign_archetypes(adata_holdout, percentage_per_archetype=0.15, verbose=False)

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

    # CV summary — build from ranked list (CVSummary has no .summary_df attr)
    cv_rows = []
    for r in ranked[:50]:
        hp = r["hyperparameters"]
        cv_rows.append({
            "K": hp["n_archetypes"],
            "hidden_dims": str(hp.get("hidden_dims", "?")),
            "inflation": hp.get("inflation_factor", "?"),
            "R²": r.get("metric_value", float("nan")),
        })
    html_parts.append(report.df_to_html(
        pd.DataFrame(cv_rows), "CV search (ranked by archetype_r2)", max_rows=50))

    # Drift / stability — helper expects Sequence[Tuple[str, dict]]
    try:
        drift_html = build_drift_qc_panel(
            [("Global TNBC fit", res)],
            drift_threshold=0.05, converged_window=10,
        )
        html_parts.append(drift_html)
    except Exception as e:
        html_parts.append(error_html(f"drift QC failed: {e}"))

    # Convergence flag — convergence_status returns (status_str, delta_mean_float)
    try:
        tc = res.get("training_config", {})
        hist = res.get("history", {})
        actual_epochs = tc.get("actual_epochs", len(hist.get("loss", [])))
        early_stop = tc.get("early_stop_triggered", res.get("early_stopped", False))
        status_str, delta_mean = convergence_status(
            history=hist, max_epochs=MAX_EPOCHS_FINAL,
            early_stop_triggered=bool(early_stop),
            actual_epochs=int(actual_epochs),
        )
        html_parts.append(
            f"<p><strong>Convergence status:</strong> {status_str} "
            f"(Δloss={delta_mean:.4g}, actual_epochs={actual_epochs})</p>"
        )
    except Exception as e:
        html_parts.append(error_html(f"convergence_status failed: {e}"))

    # Holdout projection QC
    try:
        archetype_positions = np.asarray(adata_train.uns["archetype_coordinates"])
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
    """Fig 3A — global archetype space + char table + covariate OR tables."""
    t_phase = time.time()
    html_parts: list = []

    # 2.1 — 9-color (response × treatment) map applied via a synthetic obs column.
    # pc.pl.archetypal_space's categorical_colors expects flat {level: color},
    # so we build a combined 'response_treatment' column and a flat cmap.
    cmap_tuple = build_response_timepoint_colormap()
    cmap_flat = {f"{r}|{t}": c for (r, t), c in cmap_tuple.items()}
    adata_train.obs["response_treatment"] = (
        adata_train.obs["response_group"].astype(str) + "|" +
        adata_train.obs["treatment"].astype(str)
    ).astype("category")

    # 2.2 — main 3D plot
    try:
        fig_main = pc.pl.archetypal_space(
            adata_train,
            color_by="response_treatment",
            cell_opacity=0.55,
            show_archetype_labels=True,
            title="Fig 3A — Global archetypal space (response × timepoint)",
            categorical_colors=cmap_flat,
        )
        html_parts.append(report.plotly_to_div(
            fig_main, "Fig 3A — main: 9-combo ramp (hue=response, lightness=timepoint)."
        ))
    except Exception as e:
        html_parts.append(error_html(f"Fig 3A main plot failed: {e}"))

    # 2.3 — per-timepoint facet panel (3 subplots)
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig_facet = make_subplots(rows=1, cols=3,
                                    specs=[[{"type": "scatter3d"}] * 3],
                                    subplot_titles=("Base", "PD1", "RTPD1"))
        archetype_pos = np.asarray(adata_train.uns["archetype_coordinates"])[:, :3]
        for col, tp in enumerate(("Base", "PD1", "RTPD1"), start=1):
            mask = (adata_train.obs["treatment"].astype(str) == tp).values
            pts = adata_train.obsm["X_pca"][mask, :3]
            resp = adata_train.obs.loc[mask, "response_group"].astype(str).values
            colors = [cmap_tuple[(r, tp)] for r in resp]
            fig_facet.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers", marker=dict(size=2.0, color=colors, opacity=0.55),
                showlegend=False,
            ), row=1, col=col)
            fig_facet.add_trace(go.Scatter3d(
                x=archetype_pos[:, 0], y=archetype_pos[:, 1], z=archetype_pos[:, 2],
                mode="markers+text",
                marker=dict(size=6, color="black", symbol="diamond"),
                text=[f"A{i}" for i in range(archetype_pos.shape[0])],
                showlegend=False,
            ), row=1, col=col)
        fig_facet.update_layout(height=500, width=1300,
                                  title="Fig 3A-ii — Per-timepoint facets")
        html_parts.append(report.plotly_to_div(
            fig_facet, "Fig 3A-ii — 3 facets (Base / PD1 / RTPD1)."
        ))
    except Exception as e:
        html_parts.append(error_html(f"Fig 3A facet panel failed: {e}"))

    # 2.4 — holdout projection visualization
    try:
        import plotly.graph_objects as go
        archetype_pos = np.asarray(adata_train.uns["archetype_coordinates"])[:, :3]
        fig_ho = go.Figure()
        fig_ho.add_trace(go.Scatter3d(
            x=adata_train.obsm["X_pca"][:, 0],
            y=adata_train.obsm["X_pca"][:, 1],
            z=adata_train.obsm["X_pca"][:, 2],
            mode="markers", marker=dict(size=1.5, color="lightgray", opacity=0.4),
            name="train",
        ))
        ho_resp = adata_holdout.obs["response_group"].astype(str).values
        ho_tx = adata_holdout.obs["treatment"].astype(str).values
        ho_colors = [cmap_tuple[(r, t)] for r, t in zip(ho_resp, ho_tx)]
        fig_ho.add_trace(go.Scatter3d(
            x=adata_holdout.obsm["X_pca"][:, 0],
            y=adata_holdout.obsm["X_pca"][:, 1],
            z=adata_holdout.obsm["X_pca"][:, 2],
            mode="markers", marker=dict(size=2.5, color=ho_colors, opacity=0.85),
            name="holdout",
        ))
        fig_ho.add_trace(go.Scatter3d(
            x=archetype_pos[:, 0], y=archetype_pos[:, 1], z=archetype_pos[:, 2],
            mode="markers+text",
            marker=dict(size=8, color="black", symbol="diamond"),
            text=[f"A{i}" for i in range(archetype_pos.shape[0])],
        ))
        fig_ho.update_layout(title="Fig 3A-iii — Holdout cells projected", height=560)
        html_parts.append(report.plotly_to_div(
            fig_ho, "Fig 3A-iii — held-out cells (colored) over train cells (grey)."
        ))
    except Exception as e:
        html_parts.append(error_html(f"Fig 3A holdout projection failed: {e}"))

    # 2.5 — characterization table.
    # Run simplex regression ONCE here; store into adata.uns so Phase 3 can reuse.
    try:
        pc.tl.feature_simplex_regression(adata_train, max_degree=1, robust_se=True)
        reg_result = adata_train.uns.get("peach_simplex_regression_genes") or \
                      adata_train.uns.get("peach_simplex_regression", {})
    except Exception as e_reg:
        html_parts.append(error_html(
            f"simplex regression failed: {e_reg}. "
            "Fig 3B dotplots and characterization top_genes will be empty."
        ))
        reg_result = {}

    # Build top_genes_by_archetype from regression result
    top_genes_by_archetype = None
    try:
        if reg_result:
            # regression_to_long_df → pick top 5 per archetype by R² (no exclusive filter)
            ldf = regression_to_long_df(
                reg_result, y_col="gene", exclusive_only=False,
                top_n_per_archetype=5, fdr_threshold=1.0, degree=1,
            )
            if len(ldf):
                # Extract per-archetype top genes where that archetype is argmax
                top_genes_by_archetype = {}
                for a_idx in sorted(ldf["argmax_archetype"].unique()):
                    sub = ldf[ldf["argmax_archetype"] == a_idx]
                    sub = sub.drop_duplicates(subset=["gene"]).nlargest(5, "r_squared")
                    top_genes_by_archetype[int(a_idx)] = sub["gene"].tolist()
    except Exception as e_top:
        html_parts.append(error_html(f"top_genes extraction failed: {e_top}"))
        top_genes_by_archetype = None

    try:
        char_df = build_archetype_char_table(
            adata_train.obs,
            archetypes_col="archetypes",
            covariate_cols=["response_group", "treatment", "cohort", "majority_voting"],
            top_genes_by_archetype=top_genes_by_archetype,
        )
        html_parts.append(report.df_to_html(
            char_df, "Archetype characterization — quick-look table."
        ))
    except Exception as e:
        html_parts.append(error_html(f"characterization table failed: {e}"))

    # 2.6 — hypergeometric OR tables
    try:
        or_tables = build_archetype_hypergeometric_tables(
            adata_train.obs,
            archetypes_col="archetypes",
            covariate_cols=["response_group", "treatment", "majority_voting", "cohort"],
            min_level_cells=50,
        )
        for cov, df in or_tables.items():
            cap = f"Hypergeometric enrichment — {cov} × archetype (BH q-values within covariate)."
            html_parts.append(report.df_to_html(df, cap, max_rows=60))
    except Exception as e:
        html_parts.append(error_html(f"hypergeometric tables failed: {e}"))

    report.add_section("Fig 3A — Global archetype space (response × timepoint)",
                        "\n".join(html_parts), step_num=2, open_by_default=True)
    print(f"  phase2: done in {time.time() - t_phase:.1f}s")


def phase3_figure3bc(adata_train, res, report: HTMLReport):
    """Fig 3B (gene/pathway/stress dotplots) + gated Fig 3C (heatmaps + diversity)."""
    t_phase = time.time()

    # ------ Fig 3B ---------------------------------------------------------
    fig3b_parts: list = []

    # 3B.1 — gene dotplot from the regression result Phase 2 already stored.
    try:
        reg_result = adata_train.uns.get("peach_simplex_regression_genes") or \
                      adata_train.uns.get("peach_simplex_regression", {})
        if not reg_result:
            raise ValueError("No simplex regression result in adata.uns — "
                             "Phase 2 may have failed to run it.")
        gene_long = regression_to_long_df(
            reg_result, y_col="gene",
            exclusive_only=True, exclusive_threshold=EXCLUSIVE_RATIO_THRESHOLD,
            fdr_threshold=FDR_THRESHOLD, top_n_per_archetype=10, degree=1,
        )
        if len(gene_long):
            fig_b1 = pc.pl.dotplot(
                gene_long, x_col="archetype", y_col="gene",
                size_col="mean_archetype", color_col="pvalue",
                top_n_per_group=10,
                figsize=dotplot_figsize(gene_long, y_col="gene"),
                title=f"Fig 3B-1 — archetype-exclusive genes (deg-1, excl ≥{EXCLUSIVE_RATIO_THRESHOLD}, FDR≤{FDR_THRESHOLD})",
            )
            fig3b_parts.append(report.fig_to_img(
                fig_b1,
                f"Fig 3B-1 — archetype-exclusive genes. "
                f"n_unique={gene_long['gene'].nunique()}, rows={len(gene_long)}."
            ))
        else:
            fig3b_parts.append(error_html(
                f"No archetype-exclusive genes survived filters "
                f"(exclusive_ratio ≥ {EXCLUSIVE_RATIO_THRESHOLD}, FDR ≤ {FDR_THRESHOLD})."
            ))
    except Exception as e:
        fig3b_parts.append(error_html(f"Fig 3B gene dotplot failed: {e}"))

    # 3B.2 — pathway simplex regression (if pathway_scores present)
    if "pathway_scores" in adata_train.obsm:
        try:
            pc.tl.feature_simplex_regression(
                adata_train, max_degree=1, feature_matrix="pathway_scores",
                robust_se=True,
            )
            pw_reg = adata_train.uns.get("peach_simplex_regression_pathways") or \
                      adata_train.uns.get("peach_simplex_regression_pathway_scores", {})
            pw_long = regression_to_long_df(
                pw_reg, y_col="pathway", exclusive_only=False,
                fdr_threshold=FDR_THRESHOLD, top_n_per_archetype=5, degree=1,
            )
            if len(pw_long):
                fig_b2 = pc.pl.dotplot(
                    pw_long, x_col="archetype", y_col="pathway",
                    size_col="mean_archetype", color_col="pvalue",
                    top_n_per_group=5,
                    figsize=dotplot_figsize(pw_long, y_col="pathway"),
                    title=f"Fig 3B-2 — archetype pathway enrichment (top 5/archetype, FDR≤{FDR_THRESHOLD})",
                )
                fig3b_parts.append(report.fig_to_img(
                    fig_b2, "Fig 3B-2 — archetype pathway enrichment."
                ))
            else:
                fig3b_parts.append(error_html(
                    f"No pathways reached FDR ≤ {FDR_THRESHOLD}."
                ))
        except Exception as e:
            fig3b_parts.append(error_html(f"Fig 3B pathway dotplot failed: {e}"))
    else:
        fig3b_parts.append(error_html(
            "adata.obsm['pathway_scores'] absent — Fig 3B-2 pathway dotplot skipped. "
            "Upstream prep must add this via pp.compute_pathway_scores."
        ))

    # 3B.3 — stress-gene subset (re-run regression restricted to stress genes).
    try:
        stress_in_data = [g for g in STRESS_GENES_FLAT if g in adata_train.var_names]
        if not stress_in_data:
            raise ValueError("No stress genes found in adata.var_names.")
        adata_stress = adata_train[:, stress_in_data].copy()
        # Copy over the archetypes / weights so regression can run on this subset
        adata_stress.obs = adata_train.obs.copy()
        adata_stress.obsm = adata_train.obsm.copy()
        adata_stress.uns = adata_train.uns.copy()
        pc.tl.feature_simplex_regression(adata_stress, max_degree=1, robust_se=True)
        stress_reg = adata_stress.uns.get("peach_simplex_regression_genes") or \
                      adata_stress.uns.get("peach_simplex_regression", {})
        stress_long = regression_to_long_df(
            stress_reg, y_col="gene", exclusive_only=False,
            fdr_threshold=FDR_THRESHOLD, top_n_per_archetype=5, degree=1,
        )
        if len(stress_long):
            fig_b3 = pc.pl.dotplot(
                stress_long, x_col="archetype", y_col="gene",
                size_col="mean_archetype", color_col="pvalue",
                top_n_per_group=5,
                figsize=dotplot_figsize(stress_long, y_col="gene"),
                title=f"Fig 3B-3 — stress genes ({len(stress_in_data)} overlap, FDR≤{FDR_THRESHOLD})",
            )
            fig3b_parts.append(report.fig_to_img(
                fig_b3,
                f"Fig 3B-3 — stress-gene subset. "
                f"n_overlap={len(stress_in_data)}, n_unique_plotted={stress_long['gene'].nunique()}."
            ))
        else:
            fig3b_parts.append(error_html(
                f"No stress genes reached FDR ≤ {FDR_THRESHOLD} — negative control confirmed."
            ))
    except Exception as e:
        fig3b_parts.append(error_html(f"Fig 3B stress dotplot failed: {e}"))

    report.add_section("Fig 3B — Archetype molecular characterization",
                        "\n".join(fig3b_parts), step_num=3, open_by_default=True)

    # ------ Fig 3C ---------------------------------------------------------
    fig3c_parts: list = []

    weights = adata_train.obsm["cell_archetype_weights"]

    # 3C gate computation
    try:
        seg = build_segregation_ratio(
            adata_train.obs, weights,
            response_col="response_group", treatment_col="treatment",
        )
        fig3c_parts.append(metric_grid([
            metric_card("Within (mean W2)", seg["within"], ".3f"),
            metric_card("Between (mean W2)", seg["between"], ".3f"),
            metric_card("Segregation ratio", seg["ratio"], ".3f"),
            metric_card("Gate (≥ 1.3)", "PASS" if seg["ratio"] >= FIG3C_GATE_THRESHOLD else "FAIL", "s"),
            metric_card("N within pairs", seg["n_within_pairs"], "d"),
            metric_card("N between pairs", seg["n_between_pairs"], "d"),
        ]))
        gate_passed = seg["ratio"] >= FIG3C_GATE_THRESHOLD
    except Exception as e:
        fig3c_parts.append(error_html(f"segregation ratio failed: {e}"))
        gate_passed = False

    # 3C-i — gated heatmaps
    if gate_passed:
        try:
            fig_heat, rho = build_distance_heatmaps(
                adata_train.obs, weights, adata_train.obsm["X_pca"],
                response_col="response_group", archetypes_col="archetypes",
            )
            fig3c_parts.append(report.plotly_to_div(
                fig_heat,
                f"Fig 3C-i — (response × archetype) distances: W2 vs Euclidean "
                f"centroid. Spearman ρ = {rho:.3f}."
            ))
        except Exception as e:
            fig3c_parts.append(error_html(f"Fig 3C-i heatmaps failed: {e}"))
    else:
        fig3c_parts.append(
            '<div style="padding:8px;background:#fffbeb;border-left:4px solid #ca8a04">'
            "<strong>Fig 3C-i skipped.</strong> Segregation ratio below threshold "
            f"({FIG3C_GATE_THRESHOLD}). Consider K±1, per-timepoint modeling, or "
            "batch correction on PC1 (see Phase 1 PC1 scan)."
            "</div>"
        )

    # 3C-ii — always rendered
    try:
        fig_div, summary = build_diversity_block(
            adata_train.obs, weights, adata_train.obsm["X_pca"],
            group_col="response_group", bootstrap_n=200, subsample=500,
            random_state=42,
        )
        fig3c_parts.append(report.plotly_to_div(
            fig_div,
            f"Fig 3C-ii — Diversity block. "
            f"KW H={summary['per_cell_shannon_kw_stat']:.2f}, "
            f"p={summary['per_cell_shannon_kw_p']:.2e}. "
            f"Per-group PCA dispersion (pre-registered test for R2 &lt; NR): "
            f"{summary['per_group_pca_dispersion']}."
        ))
    except Exception as e:
        fig3c_parts.append(error_html(f"Fig 3C-ii diversity block failed: {e}"))

    report.add_section("Fig 3C — Segregation distances + diversity",
                        "\n".join(fig3c_parts), step_num=4, open_by_default=True)
    print(f"  phase3: done in {time.time() - t_phase:.1f}s")


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

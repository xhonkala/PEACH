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
    build_absence_plot,
    build_archetype_char_table,
    build_archetype_hypergeometric_tables,
    build_distance_heatmaps,
    build_diversity_block,
    build_drift_qc_panel,
    build_holdout_projection_qc,
    build_lollipop_chart,
    build_overlapping_ridgeplot,
    build_permutation_curve_figure,
    build_pseudotime_expansion_plot,
    build_response_timepoint_colormap,
    build_segregation_ratio,
    build_tricolor_gene_scatter,
    convergence_status,
    dotplot_figsize,
    wasserstein2_distance,
)

# Reuse Part 1's helpers verbatim — do NOT re-implement (per shared memory rule
# "Always check tools_schema.py and types_index.py + Part 1 helpers first").
from run_paper_part1_hsc import regression_to_long_df  # noqa: E402

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

# r4 Step 1 cache + Step 2/3 toggles
STEP1_CACHE_PATH = DATA_DIR / "step1_cache.h5ad"
STEP1_HOLDOUT_CACHE_PATH = DATA_DIR / "step1_holdout_cache.h5ad"
STEP1_FORCE_RECOMPUTE = False   # set True to invalidate cache
RUN_STEP2 = True                # Phase 4 — per-condition fits + Fig 4A/B
RUN_STEP3 = True                # Phase 5 — flow_between on selected pairs

# Phase 4 (Step 2) config
STEP2_CONDITIONS = [
    ("NR", "Base"), ("NR", "PD1"), ("NR", "RTPD1"),
    ("R1", "Base"), ("R1", "PD1"),                   # R1/RTPD1 excluded (46 cells)
    ("R2", "Base"), ("R2", "PD1"), ("R2", "RTPD1"),
]   # 8 conditions
STEP2_K_RANGE = list(range(3, 8))
STEP2_HIDDEN_DIMS_OPTIONS = [[64, 128], [128, 256]]
STEP2_INFLATION_FACTOR_RANGE = [1.0, 1.25, 1.5]
STEP2_R2_THRESHOLD = 0.85               # looser than Step 1 (smaller per-condition n)
STEP2_MIN_CELLS = 500                   # below this, skip the fit
STEP2_FLOW_EPOCHS_SCORING = 200         # cheap pairwise flow for Fig 4B scoring
STEP2_FLOW_PERMUTATIONS_SCORING = 50    # short null for scoring
STEP2_FLOW_PERM_EPOCHS_SCORING = 50

# Phase 5 (Step 3) pair selection + full flow config
STEP3_MASS_PCT_THRESHOLD = 0.05         # mass ≥ 5% of cumulative
STEP3_PERM_P_THRESHOLD = 0.05           # significant by permutation null
STEP3_FLOW_EPOCHS_FULL = 1000           # full epochs for selected pairs
STEP3_FLOW_PERMUTATIONS_FULL = 100
STEP3_FLOW_GENE_TOP = 50
STEP3_FLOW_JAC_TOP = 2500

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


# NOTE: regression_to_long_df is imported from run_paper_part1_hsc.py — see
# the top-of-file import block. Do NOT define a local copy here; re-using the
# Part 1 implementation guarantees Part 1 / Part 2 dotplots stay in sync.


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
        # r4 fix A — apply matched subsample to BOTH train and holdout so the
        # 80/20 ratio is preserved post-subsample. r3 review flagged that
        # train shrunk to 4223 while holdout stayed 5279 — inverted ratio.
        adata_train = _stratified_subsample(
            adata_train, SUBSAMPLE_FRACTION, SUBSAMPLE_STRATIFY, seed=42
        )
        adata_holdout = _stratified_subsample(
            adata_holdout, SUBSAMPLE_FRACTION, SUBSAMPLE_STRATIFY, seed=42
        )
        print(f"  phase1: subsampled train -> {adata_train.shape}, "
              f"holdout -> {adata_holdout.shape}")

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

    # 1d-bis — compute pathway scores so Fig 3B-2 can render.
    # Loads MSigDB c5_bp (GO:BP) networks; decoupler + tqdm are required.
    try:
        pathway_net = pc.pp.load_pathway_networks(
            sources=["c5_bp"], organism="human", verbose=False,
        )
        pc.pp.compute_pathway_scores(
            adata_train, net=pathway_net,
            obsm_key="pathway_scores", verbose=False,
        )
        print(f"  phase1: pathway_scores computed, shape="
              f"{adata_train.obsm['pathway_scores'].shape}")
    except Exception as e:
        print(f"  phase1: pathway_scores FAILED ({e}); Fig 3B-2 will skip.")

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

    # r4 fix B — training metrics curves (loss / archetypal_loss / KLD / R² /
    # stability) via PEACH-native pc.pl.training_metrics. Three-row layout
    # courtesy of the helper.
    try:
        fig_train = pc.pl.training_metrics(res["history"], display=False,
                                              height=520, width=1100)
        if fig_train is not None:
            html_parts.append(report.plotly_to_div(
                fig_train,
                "Phase 1 training metrics — loss, archetypal/KLD/recon, "
                "stability (latent + PCA), and Δloss convergence."
            ))
    except Exception as e:
        html_parts.append(error_html(f"training_metrics failed: {e}"))

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

    # 2.2 — main 3D plot (cell_opacity=1.0 per r3 review)
    try:
        fig_main = pc.pl.archetypal_space(
            adata_train,
            color_by="response_treatment",
            cell_opacity=1.0,
            show_archetype_labels=True,
            title="Fig 3A — Global archetypal space (response × timepoint)",
            categorical_colors=cmap_flat,
        )
        html_parts.append(report.plotly_to_div(
            fig_main,
            "Fig 3A — main: 9-combo ramp (hue=response: NR=reds / R1=greens / "
            "R2=blues; lightness=timepoint: Base→PD1→RTPD1)."
        ))
    except Exception as e:
        html_parts.append(error_html(f"Fig 3A main plot failed: {e}"))

    # 2.3 — per-timepoint facet panel (3 subplots).
    # r3 adds: archetype diamonds + full connecting edges on each facet, alpha=1.0.
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig_facet = make_subplots(rows=1, cols=3,
                                    specs=[[{"type": "scatter3d"}] * 3],
                                    subplot_titles=("Base", "PD1", "RTPD1"))
        archetype_pos = np.asarray(adata_train.uns["archetype_coordinates"])[:, :3]
        K = archetype_pos.shape[0]
        # Per-facet cell counts — warn user if RTPD1 is severely imbalanced
        facet_n = {}
        for col, tp in enumerate(("Base", "PD1", "RTPD1"), start=1):
            mask = (adata_train.obs["treatment"].astype(str) == tp).values
            facet_n[tp] = int(mask.sum())
            pts = adata_train.obsm["X_pca"][mask, :3]
            resp = adata_train.obs.loc[mask, "response_group"].astype(str).values
            colors = [cmap_tuple[(r, tp)] for r in resp]
            fig_facet.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers", marker=dict(size=2.0, color=colors, opacity=1.0),
                showlegend=False,
            ), row=1, col=col)
            # Archetype-archetype connecting edges (simplex edges)
            for i in range(K):
                for j in range(i + 1, K):
                    fig_facet.add_trace(go.Scatter3d(
                        x=[archetype_pos[i, 0], archetype_pos[j, 0]],
                        y=[archetype_pos[i, 1], archetype_pos[j, 1]],
                        z=[archetype_pos[i, 2], archetype_pos[j, 2]],
                        mode="lines",
                        line=dict(color="black", width=1.5),
                        showlegend=False, hoverinfo="skip",
                    ), row=1, col=col)
            # Archetype diamonds + labels on top
            fig_facet.add_trace(go.Scatter3d(
                x=archetype_pos[:, 0], y=archetype_pos[:, 1], z=archetype_pos[:, 2],
                mode="markers+text",
                marker=dict(size=8, color="black", symbol="diamond"),
                text=[f"A{i}" for i in range(K)],
                textposition="top center",
                showlegend=False, hoverinfo="skip",
            ), row=1, col=col)
        fig_facet.update_layout(height=560, width=1400,
                                  title="Fig 3A-ii — Per-timepoint facets")
        html_parts.append(report.plotly_to_div(
            fig_facet,
            f"Fig 3A-ii — 3 facets (Base / PD1 / RTPD1). Cell counts: "
            f"Base={facet_n.get('Base', 0)}, PD1={facet_n.get('PD1', 0)}, "
            f"RTPD1={facet_n.get('RTPD1', 0)}. "
            f"Note RTPD1 imbalance if counts differ by ≥5× — may affect "
            f"response-group separation visibility."
        ))
    except Exception as e:
        html_parts.append(error_html(f"Fig 3A facet panel failed: {e}"))

    # 2.4 — holdout projection visualization.
    # r3 fix: make train background clearly visible grey, holdout smaller/brighter
    # so the grey backdrop reads properly.
    try:
        import plotly.graph_objects as go
        archetype_pos = np.asarray(adata_train.uns["archetype_coordinates"])[:, :3]
        fig_ho = go.Figure()
        # Train cells — deliberate grey backdrop. Drawn first so holdout renders on top.
        fig_ho.add_trace(go.Scatter3d(
            x=adata_train.obsm["X_pca"][:, 0],
            y=adata_train.obsm["X_pca"][:, 1],
            z=adata_train.obsm["X_pca"][:, 2],
            mode="markers",
            marker=dict(size=1.6, color="rgb(200,200,200)", opacity=0.7),
            name=f"train (n={adata_train.n_obs})",
        ))
        ho_resp = adata_holdout.obs["response_group"].astype(str).values
        ho_tx = adata_holdout.obs["treatment"].astype(str).values
        ho_colors = [cmap_tuple[(r, t)] for r, t in zip(ho_resp, ho_tx)]
        fig_ho.add_trace(go.Scatter3d(
            x=adata_holdout.obsm["X_pca"][:, 0],
            y=adata_holdout.obsm["X_pca"][:, 1],
            z=adata_holdout.obsm["X_pca"][:, 2],
            mode="markers", marker=dict(size=3.2, color=ho_colors, opacity=1.0,
                                         line=dict(color="black", width=0.3)),
            name=f"holdout (n={adata_holdout.n_obs})",
        ))
        # Archetype simplex edges for reference
        K = archetype_pos.shape[0]
        for i in range(K):
            for j in range(i + 1, K):
                fig_ho.add_trace(go.Scatter3d(
                    x=[archetype_pos[i, 0], archetype_pos[j, 0]],
                    y=[archetype_pos[i, 1], archetype_pos[j, 1]],
                    z=[archetype_pos[i, 2], archetype_pos[j, 2]],
                    mode="lines",
                    line=dict(color="black", width=1.5),
                    showlegend=False, hoverinfo="skip",
                ))
        fig_ho.add_trace(go.Scatter3d(
            x=archetype_pos[:, 0], y=archetype_pos[:, 1], z=archetype_pos[:, 2],
            mode="markers+text",
            marker=dict(size=8, color="black", symbol="diamond"),
            text=[f"A{i}" for i in range(K)],
            textposition="top center",
            showlegend=False,
        ))
        fig_ho.update_layout(title="Fig 3A-iii — Holdout cells projected", height=620)
        html_parts.append(report.plotly_to_div(
            fig_ho,
            "Fig 3A-iii — train cells (grey backdrop) + held-out cells "
            "(9-combo ramp, black outline) + archetype simplex edges."
        ))
    except Exception as e:
        html_parts.append(error_html(f"Fig 3A holdout projection failed: {e}"))

    # 2.4b — Holdout projection stats (new in r3): per-archetype KS + Hotelling T².
    try:
        from scipy.stats import ks_2samp, chi2
        try:
            from statsmodels.stats.multitest import multipletests
        except ImportError:
            multipletests = None

        weights_tr = adata_train.obsm["cell_archetype_weights"]
        weights_ho = adata_holdout.obsm["cell_archetype_weights"]
        K_ = weights_tr.shape[1]

        # Per-archetype KS test (train vs holdout weight on that archetype)
        ks_rows = []
        for k in range(K_):
            stat, p = ks_2samp(weights_tr[:, k], weights_ho[:, k])
            ks_rows.append({"archetype": f"archetype_{k}",
                             "ks_statistic": float(stat),
                             "p": float(p)})
        ks_df = pd.DataFrame(ks_rows)
        if multipletests is not None and len(ks_df):
            _, qs, _, _ = multipletests(ks_df["p"].values, method="fdr_bh")
            ks_df["q (BH)"] = qs
        html_parts.append(report.df_to_html(
            ks_df,
            "Holdout projection QC — per-archetype KS test on archetype "
            "weights (train vs holdout). BH-corrected q. Large p/q = holdout "
            "weight distribution matches train for that archetype."
        ))

        # Hotelling's T² on mean weight vector (multivariate)
        mu_tr = weights_tr.mean(axis=0)
        mu_ho = weights_ho.mean(axis=0)
        n1, n2 = weights_tr.shape[0], weights_ho.shape[0]
        # Pooled covariance
        cov_tr = np.cov(weights_tr, rowvar=False)
        cov_ho = np.cov(weights_ho, rowvar=False)
        pooled_cov = ((n1 - 1) * cov_tr + (n2 - 1) * cov_ho) / max(n1 + n2 - 2, 1)
        # Tikhonov-regularize for invertibility
        pooled_cov = pooled_cov + 1e-8 * np.eye(K_)
        diff = (mu_tr - mu_ho).reshape(-1, 1)
        inv_cov = np.linalg.pinv(pooled_cov)
        t2 = float((n1 * n2 / (n1 + n2)) * (diff.T @ inv_cov @ diff)[0, 0])
        # F approximation: T² ~ (n1+n2-2)*p / (n1+n2-p-1) * F(p, n1+n2-p-1)
        p_dim = K_
        df1, df2 = p_dim, n1 + n2 - p_dim - 1
        F_stat = t2 * df2 / (df1 * (n1 + n2 - 2))
        from scipy.stats import f as f_dist
        ho_p = float(1 - f_dist.cdf(F_stat, df1, df2)) if df2 > 0 else float("nan")
        html_parts.append(metric_grid([
            metric_card("Hotelling T²", t2, ".3f"),
            metric_card("F (approx)", F_stat, ".3f"),
            metric_card("df1 / df2", f"{df1} / {df2}", "s"),
            metric_card("Hotelling p", ho_p, ".3e"),
        ]))
    except Exception as e:
        html_parts.append(error_html(f"holdout KS/Hotelling failed: {e}"))

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

    # 2.6 — hypergeometric OR tables. r3: drop majority_voting (not useful —
    # nearly all cells are 'tumor' by design, so no contrast).
    try:
        or_tables = build_archetype_hypergeometric_tables(
            adata_train.obs,
            archetypes_col="archetypes",
            covariate_cols=["response_group", "treatment", "cohort"],
            min_level_cells=50,
        )
        for cov, df in or_tables.items():
            cap = (f"Hypergeometric enrichment — {cov} × archetype "
                   f"(BH q-values within covariate).")
            html_parts.append(report.df_to_html(df, cap, max_rows=60))
    except Exception as e:
        html_parts.append(error_html(f"hypergeometric tables failed: {e}"))

    # 2.6b — per-(archetype × level) cell count contingency tables (r3 diagnostic).
    # Helps investigate surprising OR + q results (e.g. r2 reported OR=0.029 with
    # q=5e-80 on archetype 1 × R2) by exposing the raw observed cell counts.
    try:
        rows_diag = []
        for cov in ("response_group", "treatment"):
            for a in sorted(adata_train.obs["archetypes"].dropna().unique()):
                sub = adata_train.obs.loc[adata_train.obs["archetypes"] == a]
                counts = sub[cov].value_counts()
                row = {"archetype": str(a), "covariate": cov,
                        "archetype_total": int(len(sub))}
                for lv in sorted(adata_train.obs[cov].dropna().unique()):
                    row[f"n_{lv}"] = int(counts.get(lv, 0))
                rows_diag.append(row)
        if rows_diag:
            html_parts.append(report.df_to_html(
                pd.DataFrame(rows_diag),
                "Diagnostic — observed cell counts per (archetype × level). "
                "Compare with OR tables above to sanity-check surprising enrichments.",
                max_rows=60,
            ))
    except Exception as e:
        html_parts.append(error_html(f"count-contingency diagnostic failed: {e}"))

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
            # r4 fix E — loosen FDR 0.05 → 0.10 and top_n 5 → 10. r3 review
            # showed the panel was nearly empty due to total_counts-dominated
            # PC1 crowding out pathway signal at the tighter threshold.
            pw_long = regression_to_long_df(
                pw_reg, y_col="pathway", exclusive_only=False,
                fdr_threshold=0.10, top_n_per_archetype=10, degree=1,
            )
            if len(pw_long):
                fig_b2 = pc.pl.dotplot(
                    pw_long, x_col="archetype", y_col="pathway",
                    size_col="mean_archetype", color_col="pvalue",
                    top_n_per_group=10,
                    figsize=dotplot_figsize(pw_long, y_col="pathway"),
                    title="Fig 3B-2 — archetype pathway enrichment (top 10/archetype, FDR≤0.10)",
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
                f"Fig 3B-3 — stress-gene expression *within* global archetypes "
                f"(n_overlap={len(stress_in_data)}, n_unique_plotted="
                f"{stress_long['gene'].nunique()}). NOTE: shows how stress genes "
                f"score against archetypes derived from full-gene PCA. "
                f"Stress-specific archetype geometry (own PCA → own archetypes) "
                f"is a Step 2 question and lives in a future Phase."
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

    # 3C gate computation. r4 fix C — explain what this measures:
    # We split cells into 9 (response × treatment) groups, then take all C(9,2)=36
    # pair-wise 2-Wasserstein distances between groups in K-dim archetype-weight
    # space. "Within" = pairs that share the same response_group (e.g. NR-Base
    # vs NR-PD1). "Between" = pairs with different response_groups (e.g. NR-Base
    # vs R2-PD1). Segregation ratio = mean(between) / mean(within). >1 means
    # response-group identity drives more variation in archetype weights than
    # treatment timepoint does. Gate at 1.3 = "between cluster spread ≥ 30%
    # bigger than within-response spread"; below this we don't render the per-
    # cell-pop distance heatmap (Fig 3C-i) because the signal is too weak to
    # be readable. Diversity block (3C-ii/iii) renders unconditionally.
    try:
        seg = build_segregation_ratio(
            adata_train.obs, weights,
            response_col="response_group", treatment_col="treatment",
        )
        fig3c_parts.append(metric_grid([
            metric_card("Within-response mean W2", seg["within"], ".3f"),
            metric_card("Between-response mean W2", seg["between"], ".3f"),
            metric_card("Segregation ratio (between/within)", seg["ratio"], ".3f"),
            metric_card("Gate (≥ 1.3)", "PASS" if seg["ratio"] >= FIG3C_GATE_THRESHOLD else "FAIL", "s"),
            metric_card("# within-response pairs", seg["n_within_pairs"], "d"),
            metric_card("# between-response pairs", seg["n_between_pairs"], "d"),
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

    # 3C-iii — gene-profile-based diversity per response group (r3 addition).
    # Three complementary gene-level metrics, all aggregated per group:
    #   1. Mean per-gene CV across top HVGs — intra-group gene-level
    #      heterogeneity.
    #   2. Mean pairwise cell-cell Pearson correlation in HVG space —
    #      inverse diversity (high r = homogeneous).
    #   3. Pairwise Jensen-Shannon divergence between group pseudobulks —
    #      inter-group separation in gene space.
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from scipy.spatial.distance import jensenshannon

        # Pick top 500 HVGs by variance in .X (log1p counts)
        X = adata_train.X
        if hasattr(X, "toarray"):
            X_full = X.toarray()
        else:
            X_full = np.asarray(X)
        gene_var = X_full.var(axis=0)
        hvg_idx = np.argsort(-gene_var)[:500]
        X_hvg = X_full[:, hvg_idx]

        groups = sorted(adata_train.obs["response_group"].dropna().unique())
        rng = np.random.default_rng(42)

        per_group_cv = {}
        per_group_mean_corr = {}
        pseudobulks = {}
        for g in groups:
            mask = (adata_train.obs["response_group"].values == g)
            Xg = X_hvg[mask]
            if Xg.shape[0] < 2:
                per_group_cv[g] = float("nan")
                per_group_mean_corr[g] = float("nan")
                pseudobulks[g] = np.zeros(X_hvg.shape[1])
                continue
            # Metric 1: mean CV across HVGs (stdev / |mean|)
            means = Xg.mean(axis=0)
            stds = Xg.std(axis=0)
            cv = np.where(np.abs(means) > 1e-8, stds / np.abs(means), 0.0)
            per_group_cv[g] = float(cv.mean())
            # Metric 2: mean pairwise Pearson in HVG space (subsample for speed)
            n_sample = min(300, Xg.shape[0])
            idx = rng.choice(Xg.shape[0], size=n_sample, replace=False)
            Xs = Xg[idx]
            corr = np.corrcoef(Xs)
            iu = np.triu_indices_from(corr, k=1)
            per_group_mean_corr[g] = float(corr[iu].mean())
            # Metric 3 input: pseudobulk gene profile (sum-normalized to probability)
            pb = Xg.sum(axis=0) + 1e-10
            pseudobulks[g] = pb / pb.sum()

        # Inter-group JSD heatmap (symmetric)
        js_mat = np.zeros((len(groups), len(groups)))
        for i, gi in enumerate(groups):
            for j, gj in enumerate(groups):
                js_mat[i, j] = jensenshannon(pseudobulks[gi], pseudobulks[gj])

        # Three-panel figure
        fig_gp = make_subplots(rows=1, cols=3, subplot_titles=(
            "Mean per-gene CV (top 500 HVGs)",
            "Mean pairwise cell-cell Pearson (HVG space)",
            "Pairwise JSD between group pseudobulks",
        ))
        fig_gp.add_trace(go.Bar(
            x=list(groups), y=[per_group_cv[g] for g in groups],
            showlegend=False,
        ), row=1, col=1)
        fig_gp.add_trace(go.Bar(
            x=list(groups), y=[per_group_mean_corr[g] for g in groups],
            showlegend=False,
        ), row=1, col=2)
        fig_gp.add_trace(go.Heatmap(
            z=js_mat, x=list(groups), y=list(groups),
            colorscale="Viridis", showscale=True,
            colorbar=dict(x=1.02, len=0.75, title="JSD"),
        ), row=1, col=3)
        fig_gp.update_layout(
            height=440, width=1400,
            title="Fig 3C-iii — Gene-profile diversity (top 500 HVGs)",
        )
        fig3c_parts.append(report.plotly_to_div(
            fig_gp,
            f"Fig 3C-iii — gene-profile diversity per response group. "
            f"CV: higher = more gene-level heterogeneity within group. "
            f"Pearson: lower = more diverse cell profiles within group. "
            f"JSD: higher = more different between groups. "
            f"CV={ {g: f'{per_group_cv[g]:.3f}' for g in groups} }, "
            f"meanCorr={ {g: f'{per_group_mean_corr[g]:.3f}' for g in groups} }."
        ))
    except Exception as e:
        fig3c_parts.append(error_html(f"Fig 3C-iii gene-profile diversity failed: {e}"))

    report.add_section("Fig 3C — Segregation distances + diversity",
                        "\n".join(fig3c_parts), step_num=4, open_by_default=True)
    print(f"  phase3: done in {time.time() - t_phase:.1f}s")


# ============================================================================
# Step 1 cache helpers (r4)
# ============================================================================


def _save_step1_cache(adata_train, adata_holdout, res):
    """Persist Phase 1-3 artifacts so Step 2/3 iterations skip retraining."""
    print(f"  [cache] writing {STEP1_CACHE_PATH.name} ({adata_train.n_obs} cells)")
    adata_train.write_h5ad(STEP1_CACHE_PATH)
    print(f"  [cache] writing {STEP1_HOLDOUT_CACHE_PATH.name} ({adata_holdout.n_obs} cells)")
    adata_holdout.write_h5ad(STEP1_HOLDOUT_CACHE_PATH)


def _load_step1_cache():
    """Returns (adata_train, adata_holdout) from cache, or (None, None) if missing."""
    if not STEP1_CACHE_PATH.exists() or not STEP1_HOLDOUT_CACHE_PATH.exists():
        return None, None
    print(f"  [cache] loading {STEP1_CACHE_PATH.name}")
    adata_train = sc.read_h5ad(STEP1_CACHE_PATH)
    print(f"  [cache] loading {STEP1_HOLDOUT_CACHE_PATH.name}")
    adata_holdout = sc.read_h5ad(STEP1_HOLDOUT_CACHE_PATH)
    return adata_train, adata_holdout


# ============================================================================
# Phase 4 — Step 2: per-(response, treatment) fits + Fig 4A + Fig 4B
# ============================================================================


def _fit_one_condition(adata_full, response, treatment,
                         k_range=STEP2_K_RANGE,
                         n_epochs=MAX_EPOCHS_FINAL):
    """Fit one per-condition archetype model. Returns dict with status/results."""
    mask = (
        (adata_full.obs["response_group"].astype(str) == response) &
        (adata_full.obs["treatment"].astype(str) == treatment)
    ).values
    n_cells = int(mask.sum())
    if n_cells < STEP2_MIN_CELLS:
        return {"status": "skipped_insufficient_cells", "n_cells": n_cells,
                "response": response, "treatment": treatment}

    adata_sub = adata_full[mask].copy()
    pc.pp.prepare_training(adata_sub, batch_size=min(128, n_cells // 4))

    try:
        cv = pc.tl.hyperparameter_search(
            adata_sub, pca_key="X_pca",
            n_archetypes_range=list(k_range),
            hidden_dims_options=STEP2_HIDDEN_DIMS_OPTIONS,
            inflation_factor_range=STEP2_INFLATION_FACTOR_RANGE,
            use_pcha_init=False,
            cv_folds=3, max_epochs_cv=20, subsample_fraction=0.8,
        )
        best, _ = _pick_best_from_cv(cv, r2_threshold=STEP2_R2_THRESHOLD)
        K = int(best["hyperparameters"]["n_archetypes"])
        hd = list(best["hyperparameters"].get("hidden_dims", [128, 256]))
        inf = float(best["hyperparameters"].get("inflation_factor", 1.0))

        res = pc.tl.train_archetypal(
            adata_sub, n_archetypes=K, pca_key="X_pca",
            n_epochs=n_epochs, hidden_dims=hd, inflation_factor=inf,
            kld_weight=MODEL_CONFIG["kld_weight"],
            archetypal_weight=MODEL_CONFIG["archetypal_weight"],
            pcha_init=True, early_stopping=True,
            early_stopping_patience=EARLY_STOP_PATIENCE,
            model_config={"manifold_weight": MODEL_CONFIG["manifold_weight"]},
        )
        pc.tl.archetypal_coordinates(adata_sub, verbose=False)
        pc.tl.extract_archetype_weights(adata_sub, verbose=False)
        pc.tl.assign_archetypes(adata_sub, percentage_per_archetype=0.15, verbose=False)

        return {"status": "ok", "n_cells": n_cells,
                "response": response, "treatment": treatment,
                "adata_sub": adata_sub, "res": res,
                "K": K, "hidden_dims": hd, "inflation": inf,
                "final_r2": res.get("final_archetype_r2", float("nan"))}
    except Exception as e:
        return {"status": "failed", "error": str(e),
                "n_cells": n_cells,
                "response": response, "treatment": treatment}


def _build_fig4a_table(fits):
    """Per-model diversity metrics table (Fig 4A)."""
    rows = []
    for (resp, tx), entry in fits.items():
        if entry["status"] != "ok":
            rows.append({"response": resp, "treatment": tx,
                          "n_cells": entry.get("n_cells", 0),
                          "status": entry["status"],
                          "K": "—", "R²": "—",
                          "Shannon_H": "—", "PCA_disp": "—",
                          "arch_prof_H": "—"})
            continue
        adata_sub = entry["adata_sub"]
        weights = adata_sub.obsm.get("cell_archetype_weights")
        from scipy.stats import entropy as scipy_entropy
        from scipy.spatial.distance import pdist
        # Per-cell Shannon mean
        shannon = float(np.mean([scipy_entropy(w + 1e-12) for w in weights]))
        # PCA dispersion (median pairwise euclidean, subsample to 500 for speed)
        rng = np.random.default_rng(42)
        Xp = adata_sub.obsm["X_pca"]
        if Xp.shape[0] > 500:
            Xp = Xp[rng.choice(Xp.shape[0], 500, replace=False)]
        pca_disp = float(np.median(pdist(Xp)))
        # Mean weight vector entropy
        mu = weights.mean(axis=0)
        arch_prof_H = float(scipy_entropy(mu + 1e-12))
        rows.append({"response": resp, "treatment": tx,
                      "n_cells": entry["n_cells"], "status": "ok",
                      "K": entry["K"], "R²": entry.get("final_r2"),
                      "Shannon_H": shannon, "PCA_disp": pca_disp,
                      "arch_prof_H": arch_prof_H})
    return pd.DataFrame(rows)


def _score_pair_w2(entry_a, entry_b):
    """W2 between two condition models' archetype weight distributions."""
    wa = entry_a["adata_sub"].obsm["cell_archetype_weights"]
    wb = entry_b["adata_sub"].obsm["cell_archetype_weights"]
    # Pad to common K via zero-padding so wasserstein_distance_nd accepts both
    K_max = max(wa.shape[1], wb.shape[1])
    if wa.shape[1] < K_max:
        wa = np.hstack([wa, np.zeros((wa.shape[0], K_max - wa.shape[1]))])
    if wb.shape[1] < K_max:
        wb = np.hstack([wb, np.zeros((wb.shape[0], K_max - wb.shape[1]))])
    return wasserstein2_distance(wa, wb)


def _score_pair_flow(entry_a, entry_b, label_a, label_b,
                       n_epochs=STEP2_FLOW_EPOCHS_SCORING,
                       n_perms=STEP2_FLOW_PERMUTATIONS_SCORING,
                       n_perm_epochs=STEP2_FLOW_PERM_EPOCHS_SCORING):
    """Run a cheap flow_between + flow_significance to score (transport mass, perm p)."""
    adata_a = entry_a["adata_sub"].copy()
    adata_b = entry_b["adata_sub"].copy()
    adata_a.obs["__flow_cond__"] = label_a
    adata_b.obs["__flow_cond__"] = label_b
    try:
        fr = pc.tl.flow_between(
            [adata_a, adata_b],
            condition_key="__flow_cond__",
            condition_labels=[label_a, label_b],
            pairs=[(label_a, label_b)],
            pca_key="X_pca",
            n_epochs=n_epochs,
        )
    except Exception as e:
        return {"transport_mass": float("nan"), "perm_p": float("nan"),
                "error": f"flow_between failed: {e}"}
    # Transport mass extraction varies by PEACH version — try common keys
    mass = float("nan")
    for k in ("total_mass", "transported_mass", "mass"):
        if k in fr:
            mass = float(fr[k])
            break
    if np.isnan(mass):
        # Approximate from pair_results if present
        pr = fr.get("pair_results", {}) or {}
        for v in pr.values():
            if isinstance(v, dict) and "mass" in v:
                mass = float(v["mass"])
                break
    # Permutation null
    perm_p = float("nan")
    try:
        sig = pc.tl.flow_significance(
            adata_a, fr,
            n_permutations=n_perms,
            n_epochs_per_perm=n_perm_epochs,
        )
        perm_p = float(sig.get("p_value", float("nan")))
    except Exception as e:
        return {"transport_mass": mass, "perm_p": perm_p,
                "error": f"flow_significance failed: {e}",
                "flow_result": fr}
    return {"transport_mass": mass, "perm_p": perm_p, "flow_result": fr}


def _build_fig4b_heatmaps(fits, scores, report):
    """Three 8×8 heatmaps: W2, transport mass, -log10(perm p)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    cond_keys = [k for k in fits.keys() if fits[k]["status"] == "ok"]
    labels = [f"{r}|{t}" for (r, t) in cond_keys]
    G = len(cond_keys)
    w2_mat = np.full((G, G), np.nan)
    mass_mat = np.full((G, G), np.nan)
    p_mat = np.full((G, G), np.nan)
    for (i_idx, j_idx), s in scores.items():
        w2_mat[i_idx, j_idx] = w2_mat[j_idx, i_idx] = s.get("w2", np.nan)
        mass_mat[i_idx, j_idx] = mass_mat[j_idx, i_idx] = s.get("transport_mass", np.nan)
        p = s.get("perm_p", np.nan)
        if not np.isnan(p):
            log_p = -np.log10(max(p, 1e-300))
            p_mat[i_idx, j_idx] = p_mat[j_idx, i_idx] = log_p
    fig = make_subplots(rows=1, cols=3,
                          subplot_titles=("2-Wasserstein (weight space)",
                                          "Transport mass (flow_between)",
                                          "−log10(perm p)"))
    fig.add_trace(go.Heatmap(z=w2_mat, x=labels, y=labels, colorscale="Viridis",
                              showscale=True, colorbar=dict(x=0.30, len=0.75)),
                    row=1, col=1)
    fig.add_trace(go.Heatmap(z=mass_mat, x=labels, y=labels, colorscale="Plasma",
                              showscale=True, colorbar=dict(x=0.66, len=0.75)),
                    row=1, col=2)
    fig.add_trace(go.Heatmap(z=p_mat, x=labels, y=labels, colorscale="Inferno",
                              showscale=True, colorbar=dict(x=1.02, len=0.75)),
                    row=1, col=3)
    fig.update_layout(height=540, width=1500,
                        title="Fig 4B — pairwise condition relatedness")
    return fig


def _select_step3_pairs(scores):
    """Filter pairs by significance ∧ mass ≥ 5% of cumulative mass."""
    valid = [(k, s) for k, s in scores.items()
             if not np.isnan(s.get("transport_mass", np.nan))
             and s.get("perm_p", 1.0) <= STEP3_PERM_P_THRESHOLD]
    if not valid:
        return []
    total_mass = sum(s["transport_mass"] for _, s in valid)
    if total_mass <= 0:
        return []
    threshold = STEP3_MASS_PCT_THRESHOLD * total_mass
    selected = [k for k, s in valid if s["transport_mass"] >= threshold]
    return selected


def phase4_step2(adata_train, report: HTMLReport):
    """Train 8 per-(response, treatment) main-PCA models + Fig 4A + Fig 4B."""
    if not RUN_STEP2:
        return None, None
    t_phase = time.time()
    fits = {}
    for (resp, tx) in STEP2_CONDITIONS:
        print(f"  phase4: fitting ({resp}, {tx}) ...")
        fits[(resp, tx)] = _fit_one_condition(adata_train, resp, tx)
        e = fits[(resp, tx)]
        print(f"    -> status={e['status']}, n_cells={e.get('n_cells', 0)}, "
              f"K={e.get('K', '—')}, R²={e.get('final_r2', '—')}")

    # Fig 4A diversity table
    html_parts = []
    try:
        df_4a = _build_fig4a_table(fits)
        html_parts.append(report.df_to_html(df_4a,
            "Fig 4A — per-condition diversity metrics. Skipped rows had "
            f"<{STEP2_MIN_CELLS} cells; failed rows hit a training error."))
    except Exception as e:
        html_parts.append(error_html(f"Fig 4A failed: {e}"))

    # Pairwise scoring (W2 + transport mass + perm p)
    cond_keys = [k for k in fits.keys() if fits[k]["status"] == "ok"]
    scores = {}
    print(f"  phase4: scoring {len(cond_keys)*(len(cond_keys)-1)//2} pairs ...")
    for i, ki in enumerate(cond_keys):
        for j, kj in enumerate(cond_keys):
            if j <= i:
                continue
            label_i = f"{ki[0]}_{ki[1]}"
            label_j = f"{kj[0]}_{kj[1]}"
            print(f"    pair ({label_i}, {label_j}) ...")
            try:
                w2 = _score_pair_w2(fits[ki], fits[kj])
            except Exception as e:
                w2 = float("nan"); print(f"      W2 failed: {e}")
            try:
                fl = _score_pair_flow(fits[ki], fits[kj], label_i, label_j)
                mass = fl["transport_mass"]
                p = fl["perm_p"]
            except Exception as e:
                mass = p = float("nan"); fl = {"error": str(e)}
                print(f"      flow scoring failed: {e}")
            scores[(i, j)] = {"w2": w2, "transport_mass": mass,
                                "perm_p": p, "key_a": ki, "key_b": kj}

    # Fig 4B three-panel heatmap
    try:
        fig_4b = _build_fig4b_heatmaps(fits, scores, report)
        html_parts.append(report.plotly_to_div(
            fig_4b,
            "Fig 4B — pairwise condition-model relatedness. W2 = 2-Wasserstein "
            "in archetype-weight simplex (lower = more similar). Transport "
            "mass = total probability mass moved by flow_between (higher = "
            "more cross-flow). −log10(p) = significance vs label-shuffle null."
        ))
    except Exception as e:
        html_parts.append(error_html(f"Fig 4B heatmaps failed: {e}"))

    # Pair selection table
    selected = _select_step3_pairs(scores)
    sel_rows = []
    for (i_idx, j_idx) in scores.keys():
        s = scores[(i_idx, j_idx)]
        sel_rows.append({
            "pair": f"{s['key_a'][0]}_{s['key_a'][1]} ↔ {s['key_b'][0]}_{s['key_b'][1]}",
            "W2": s["w2"], "transport_mass": s["transport_mass"],
            "perm_p": s["perm_p"],
            "selected_for_step3": (i_idx, j_idx) in selected,
        })
    html_parts.append(report.df_to_html(
        pd.DataFrame(sel_rows).sort_values("transport_mass", ascending=False),
        f"Pair selection — {len(selected)} of {len(scores)} pairs passed "
        f"(perm p ≤ {STEP3_PERM_P_THRESHOLD} ∧ mass ≥ "
        f"{int(STEP3_MASS_PCT_THRESHOLD*100)}% cumulative). Selected pairs "
        "feed Phase 5 / Fig 4C flow analyses.",
        max_rows=60,
    ))

    report.add_section(
        "Phase 4 — Step 2: per-condition fits + Fig 4A/B",
        "\n".join(html_parts), step_num=5, open_by_default=True,
    )
    print(f"  phase4: done in {time.time() - t_phase:.1f}s; "
          f"selected {len(selected)} pairs for Phase 5")
    return fits, [(scores[k]["key_a"], scores[k]["key_b"]) for k in selected]


# ============================================================================
# Phase 5 — Step 3: full flow_between on selected pairs + per-pair detail figs
# ============================================================================


def _phase5_one_pair(fits, key_a, key_b, report):
    """Full flow + per-pair detail figs for one selected pair."""
    label_a = f"{key_a[0]}_{key_a[1]}"
    label_b = f"{key_b[0]}_{key_b[1]}"
    print(f"  phase5: detailing pair ({label_a} → {label_b}) ...")
    parts = []

    adata_a = fits[key_a]["adata_sub"].copy()
    adata_b = fits[key_b]["adata_sub"].copy()
    adata_a.obs["__flow_cond__"] = label_a
    adata_b.obs["__flow_cond__"] = label_b
    try:
        fr = pc.tl.flow_between(
            [adata_a, adata_b],
            condition_key="__flow_cond__",
            condition_labels=[label_a, label_b],
            pairs=[(label_a, label_b)],
            pca_key="X_pca",
            n_epochs=STEP3_FLOW_EPOCHS_FULL,
        )
    except Exception as e:
        parts.append(error_html(f"flow_between failed: {e}"))
        return "\n".join(parts)

    # Combined adata for downstream analyses (concat source + target)
    import anndata as ad
    adata_pair = ad.concat([adata_a, adata_b], join="outer", merge="same")
    adata_pair.obsm["X_pca"] = np.vstack([
        adata_a.obsm["X_pca"], adata_b.obsm["X_pca"]
    ])

    # Gene alignment
    try:
        align = pc.tl.flow_gene_alignment(
            adata_pair, fr, per_cell=False,
            n_top=STEP3_FLOW_GENE_TOP,
            n_permutations=STEP3_FLOW_PERMUTATIONS_FULL,
        )
    except Exception as e:
        parts.append(error_html(f"flow_gene_alignment failed: {e}"))
        align = None

    # Jacobian
    try:
        jac = pc.tl.flow_jacobian(
            adata_pair, fr, fr.get("model"),
            per_cell_features=True,
            n_top_features=STEP3_FLOW_JAC_TOP,
        )
    except Exception as e:
        parts.append(error_html(f"flow_jacobian failed: {e}"))
        jac = None

    # Permutation curve figure
    try:
        sig = pc.tl.flow_significance(
            adata_pair, fr,
            n_permutations=STEP3_FLOW_PERMUTATIONS_FULL,
            n_epochs_per_perm=200,
        )
        if "null_distribution" in sig and "observed_stat" in sig:
            null_arr = np.asarray(sig["null_distribution"])
            obs = float(sig["observed_stat"])
            sw = np.linspace(0, 1, len(null_arr))
            null_mean = np.full_like(sw, null_arr.mean())
            null_std = np.full_like(sw, null_arr.std())
            fig_pc = build_permutation_curve_figure(sw, null_mean, null_std, obs)
            parts.append(report.fig_to_img(fig_pc,
                f"Pair {label_a}→{label_b}: flow_significance permutation curve. "
                f"observed={obs:.4f}, null mean={null_arr.mean():.4f}, "
                f"p_value={sig.get('p_value', float('nan')):.3e}"))
    except Exception as e:
        parts.append(error_html(f"flow_significance failed: {e}"))

    # Per-cell expansion + alignment helpers (for tricolor / absence / lollipop)
    per_cell_align = align.get("per_cell_alignment") if align else None
    per_cell_expansion = jac.get("per_cell_expansion") if jac else None
    expansion_genes = (jac.get("per_cell_expansion_gene_names") if jac else None) or []
    align_genes = (align.get("per_cell_gene_names") if align else None) or []

    # Tricolor scatter (gene level): expression vs expansion vs flow alignment
    try:
        if per_cell_align is not None and per_cell_expansion is not None:
            # Aggregate per-gene means across cells
            top_n_viz = min(200, per_cell_expansion.shape[1])
            mean_expr = adata_pair.X.mean(axis=0)
            mean_expr = np.asarray(mean_expr).flatten()
            # Map expansion genes to indices
            gene_idx = [adata_pair.var_names.get_loc(g) for g in expansion_genes[:top_n_viz]
                         if g in adata_pair.var_names]
            if gene_idx:
                fig_tri = build_tricolor_gene_scatter(
                    mean_expr[gene_idx],
                    per_cell_expansion[:, :len(gene_idx)].mean(axis=0),
                    np.abs(per_cell_align).mean(axis=0)[:len(gene_idx)] if per_cell_align.ndim == 2 else np.zeros(len(gene_idx)),
                    [adata_pair.var_names[i] for i in gene_idx],
                )
                parts.append(report.fig_to_img(fig_tri,
                    f"Pair {label_a}→{label_b}: tricolor scatter "
                    "(expression × expansion × |alignment|)."))
    except Exception as e:
        parts.append(error_html(f"tricolor scatter failed: {e}"))

    # Absence plot
    try:
        if per_cell_align is not None:
            mean_expr_full = np.asarray(adata_pair.X.mean(axis=0)).flatten()
            mean_align = np.abs(per_cell_align).mean(axis=0) if per_cell_align.ndim == 2 else per_cell_align
            top_n_viz = min(200, len(align_genes))
            gene_idx = [adata_pair.var_names.get_loc(g) for g in align_genes[:top_n_viz]
                         if g in adata_pair.var_names]
            if gene_idx:
                fig_abs = build_absence_plot(
                    mean_expr_full[gene_idx],
                    mean_align[:len(gene_idx)],
                    [adata_pair.var_names[i] for i in gene_idx],
                )
                parts.append(report.fig_to_img(fig_abs,
                    f"Pair {label_a}→{label_b}: absence plot — "
                    "high-expression genes with low flow alignment."))
    except Exception as e:
        parts.append(error_html(f"absence plot failed: {e}"))

    # Lollipop chart
    try:
        if per_cell_align is not None and per_cell_expansion is not None:
            top_n_viz = min(25, per_cell_expansion.shape[1])
            mean_expr_full = np.asarray(adata_pair.X.mean(axis=0)).flatten()
            mean_align_g = np.abs(per_cell_align).mean(axis=0) if per_cell_align.ndim == 2 else per_cell_align
            mean_exp_g = per_cell_expansion.mean(axis=0)
            gene_idx = [adata_pair.var_names.get_loc(g) for g in expansion_genes[:top_n_viz]
                         if g in adata_pair.var_names]
            if gene_idx:
                fig_lol = build_lollipop_chart(
                    mean_align_g[:len(gene_idx)],
                    mean_exp_g[:len(gene_idx)],
                    mean_expr_full[gene_idx],
                    [adata_pair.var_names[i] for i in gene_idx],
                    top_n=top_n_viz,
                )
                parts.append(report.fig_to_img(fig_lol,
                    f"Pair {label_a}→{label_b}: lollipop — top genes by "
                    "alignment, colored by expansion sign, sized by mean expr."))
    except Exception as e:
        parts.append(error_html(f"lollipop chart failed: {e}"))

    # Ridgeplot of top expanding/contracting genes
    try:
        if per_cell_expansion is not None and len(expansion_genes):
            # Pick top 5 expanding + top 5 contracting by mean per-cell expansion
            mean_exp_g = per_cell_expansion.mean(axis=0)
            n_use = min(len(mean_exp_g), len(expansion_genes))
            order_exp = np.argsort(-mean_exp_g[:n_use])
            top_exp_idx = order_exp[:5]
            top_con_idx = order_exp[-5:]
            top_idx = np.concatenate([top_exp_idx, top_con_idx])
            ridge_data = {}
            for i in top_idx:
                gname = expansion_genes[i] if i < len(expansion_genes) else f"feat_{i}"
                if gname in adata_pair.var_names:
                    j = adata_pair.var_names.get_loc(gname)
                    col = adata_pair.X[:, j]
                    if hasattr(col, "toarray"):
                        col = col.toarray().flatten()
                    ridge_data[gname] = np.asarray(col).flatten()
            if ridge_data:
                fig_ridge = build_overlapping_ridgeplot(ridge_data, max_groups=12)
                top_genes_str = ", ".join(list(ridge_data.keys())[:10])
                parts.append(report.fig_to_img(fig_ridge,
                    f"Pair {label_a}→{label_b}: ridgeplot of top expanding + "
                    f"contracting genes. Top: {top_genes_str}"))
    except Exception as e:
        parts.append(error_html(f"ridgeplot failed: {e}"))

    # Pathway dotplot — per-pair flow-associated pathway scores via simplex regression
    # on pathway_scores, restricted to the source-archetype dimension
    try:
        if "pathway_scores" in adata_a.obsm:
            pc.tl.feature_simplex_regression(
                adata_a, max_degree=1, feature_matrix="pathway_scores",
                robust_se=True,
            )
            pw_reg = adata_a.uns.get("peach_simplex_regression_pathways") or \
                      adata_a.uns.get("peach_simplex_regression_pathway_scores", {})
            pw_long = regression_to_long_df(
                pw_reg, y_col="pathway", exclusive_only=False,
                fdr_threshold=0.10, top_n_per_archetype=10, degree=1,
            )
            if len(pw_long):
                fig_pwd = pc.pl.dotplot(
                    pw_long, x_col="archetype", y_col="pathway",
                    size_col="mean_archetype", color_col="pvalue",
                    top_n_per_group=10,
                    figsize=dotplot_figsize(pw_long, y_col="pathway"),
                    title=f"Flow-associated pathways for source ({label_a})",
                )
                top_pw_str = ", ".join(pw_long["pathway"].head(10).tolist())
                parts.append(report.fig_to_img(fig_pwd,
                    f"Pair {label_a}→{label_b}: source-side flow-associated "
                    f"pathways. Top 10: {top_pw_str}"))
    except Exception as e:
        parts.append(error_html(f"per-pair pathway dotplot failed: {e}"))

    return "\n".join(parts) if parts else "<em>(no detail content rendered)</em>"


def phase5_step3(adata_train, fits, selected_pairs, report: HTMLReport):
    if not RUN_STEP3 or fits is None or not selected_pairs:
        return
    t_phase = time.time()
    print(f"  phase5: detailing {len(selected_pairs)} selected pairs ...")
    sec_html = ""
    for (key_a, key_b) in selected_pairs:
        try:
            pair_html = _phase5_one_pair(fits, key_a, key_b, report)
        except Exception as e:
            pair_html = error_html(
                f"Pair {key_a} → {key_b} failed entirely: {e}"
            )
        sec_html += (
            f'<details><summary><h3 style="display:inline">'
            f'Pair {key_a[0]}/{key_a[1]} → {key_b[0]}/{key_b[1]}</h3></summary>'
            f'{pair_html}</details>'
        )
    report.add_section(
        "Phase 5 — Step 3: per-pair flow analyses (Fig 4C surface)",
        sec_html, step_num=6, open_by_default=False,
    )
    print(f"  phase5: done in {time.time() - t_phase:.1f}s")


# ============================================================================
# main
# ============================================================================


def main() -> None:
    report = HTMLReport(f"Paper Part 2 Steps 1–3 — TNBC ({_DATE_TAG} r{_REV})")
    report.text(
        "<strong>Run includes Step 1 (global fit, Figs 3A/B/C) + Step 2 "
        "(per-condition models, Figs 4A/B) + Step 3 (selected-pair flow "
        "analyses, Fig 4C surface).</strong> "
        f"SUBSAMPLE_FRACTION = {SUBSAMPLE_FRACTION}; flip to 1.0 for production. "
        f"Step 1 caches at {STEP1_CACHE_PATH.name}; subsequent runs skip Phase "
        f"1-3 compute unless STEP1_FORCE_RECOMPUTE is True."
    )
    t0 = time.time()

    # Phase 1-3 with cache
    cached_train, cached_holdout = _load_step1_cache() if not STEP1_FORCE_RECOMPUTE else (None, None)
    if cached_train is not None and cached_holdout is not None:
        print("  [cache hit] Step 1 loaded from cache; skipping Phase 1-3 compute.")
        adata_train, adata_holdout = cached_train, cached_holdout
        # Re-render Phase 1-3 sections in the report from cached adata (best
        # effort — some sections need the original `res` dict which isn't
        # cached, so they may render minimally).
        report.text(
            f"<em>Phase 1-3 loaded from cache "
            f"({STEP1_CACHE_PATH.name}, {STEP1_HOLDOUT_CACHE_PATH.name}); full "
            f"Phase 1-3 report sections skipped to save compute. Delete the "
            f"cache or set STEP1_FORCE_RECOMPUTE=True for full Phase 1-3 "
            f"re-render.</em>"
        )
        res = None
    else:
        adata_train, adata_holdout, res = phase1_train_model(report)
        phase2_figure3a(adata_train, adata_holdout, res, report)
        phase3_figure3bc(adata_train, res, report)
        try:
            _save_step1_cache(adata_train, adata_holdout, res)
        except Exception as e:
            print(f"  [cache] save failed: {e}")

    fits = None
    selected_pairs = []
    if RUN_STEP2:
        try:
            fits, selected_pairs = phase4_step2(adata_train, report)
        except Exception as e:
            print(f"  phase4 crashed: {e}")
            report.add_section("Phase 4 (failed)", error_html(str(e)), step_num=5)

    if RUN_STEP3 and fits and selected_pairs:
        try:
            phase5_step3(adata_train, fits, selected_pairs, report)
        except Exception as e:
            print(f"  phase5 crashed: {e}")
            report.add_section("Phase 5 (failed)", error_html(str(e)), step_num=6)

    report.save(REPORT_PATH)
    print(f"\n[main] total elapsed: {time.time() - t0:.1f}s")
    print(f"[main] report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()

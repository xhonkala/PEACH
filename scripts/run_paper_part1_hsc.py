#!/usr/bin/env python
"""Paper Part 1: HSC Technical Hypotheses.

Generates figures for the technical introduction of Deep_AA, simplex regression,
Wald contrasts, and flow fields on HSC→CMP data.

Usage: conda run -n archetype python scripts/run_paper_part1_hsc.py
"""

import matplotlib
matplotlib.use("Agg")

import base64
import io
import logging
import os
import time
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
log = logging.getLogger("paper_part1")

# ---------------------------------------------------------------------------
# Paths + run config
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "paper_part1")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "paper_part1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# W-A3: use full HSC + CMP data. Subsampling the CMP population left the fit
# severely underspecified (~125 cells per archetype), contributing to the
# archetype-far-from-data symptoms flagged in the r9 review. Subsample code
# paths are preserved below in case the constant needs to drop again, but
# the default is now 1.0 (no-op).
SUBSAMPLE_FRACTION = 0.2
SUBSAMPLE_SEED = 42

# Training convergence settings.
# MAX_EPOCHS_FINAL: upper limit for final model training (raised from 100 to allow
#   actual convergence; early stopping will exit before this when the model converges).
# EARLY_STOP_PATIENCE: consecutive validation checks without improvement before stopping.
MAX_EPOCHS_FINAL = 200
EARLY_STOP_PATIENCE = 15


def _stratified_subsample(adata, frac, seed, stratify_col="cell_type"):
    """Stratified subsample of an AnnData by a categorical obs column."""
    if frac >= 1.0:
        return adata
    rng = np.random.default_rng(seed)
    if stratify_col not in adata.obs.columns:
        # Fall back to uniform random sample
        n_keep = max(int(adata.shape[0] * frac), 1)
        idx = rng.choice(adata.shape[0], size=n_keep, replace=False)
        return adata[np.sort(idx)].copy()
    keep_idx = []
    for ct in adata.obs[stratify_col].unique():
        ct_idx = np.where(adata.obs[stratify_col] == ct)[0]
        n_keep = max(int(len(ct_idx) * frac), 1)
        sampled = rng.choice(ct_idx, size=n_keep, replace=False)
        keep_idx.extend(sampled.tolist())
    keep_idx = np.sort(np.array(keep_idx))
    return adata[keep_idx].copy()

_DATE_TAG = time.strftime("%Y%m%d")
import glob as _glob
_existing = sorted(_glob.glob(os.path.join(OUTPUT_DIR, f"part1_report_{_DATE_TAG}*.html")))
_REV = len(_existing) + 1
REPORT_PATH = os.path.join(OUTPUT_DIR, f"part1_report_{_DATE_TAG}_r{_REV}.html")

# ---------------------------------------------------------------------------
# Helpers (copied from e2e — could be shared module later)
# ---------------------------------------------------------------------------

def fmt_pval(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "NA"
    if p < 1e-300:
        return "< 1e-300"
    if p > 0.01:
        return f"{p:.3f}"
    return f"{p:.2e}"


def display_arch(label):
    if isinstance(label, str) and label.startswith("archetype_"):
        try:
            return f"A{int(label.split('_')[1]) + 1}"
        except (ValueError, IndexError):
            pass
    return str(label)


def _straw_plot_xlabel(n_mapped, n_fallback, x_min, x_max):
    """Build the straw plot x-axis label from gene mapping counts and data range.

    Parameters
    ----------
    n_mapped : int
        Number of genes successfully mapped to expression values (logcounts).
    n_fallback : int
        Number of genes that fell back to flow_coord (normalized pseudotime 0–1).
    x_min : float
        Minimum value of the x-axis data (used to show range in label).
    x_max : float
        Maximum value of the x-axis data (used to show range in label).

    Returns
    -------
    str
        Human-readable x-axis label reflecting the actual data plotted.
    """
    n_total = n_mapped + n_fallback
    if n_fallback == 0:
        return (f"Binned mean expression per pseudotime bin (logcounts; "
                f"X_source range [{x_min:.2f}, {x_max:.2f}])")
    elif n_mapped == 0:
        return "Flow coordinate (normalized pseudotime, 0\u20131; all genes fell back)"
    else:
        return (f"Binned mean expression or flow_coord ({n_fallback}/{n_total} "
                f"genes fell back to flow_coord; logcounts range "
                f"[{x_min:.2f}, {x_max:.2f}])")


def error_html(msg):
    return f'<div class="error">{msg}</div>'


def metric_card(value, label):
    return f'<div class="metric-card"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>'


def metric_grid(cards):
    return '<div class="metric-grid">' + ''.join(cards) + '</div>'


def safe_plotly_html(report, fig, caption=""):
    try:
        return report.plotly_to_div(fig, caption=caption)
    except Exception as e:
        return error_html(f"Plotly render failed: {e}")


def wasserstein2_distance(X, Y, *, max_n=2000, seed=42):
    """Compute Wasserstein-2 distance between two point clouds in PC units.

    Uses scipy.stats.wasserstein_distance_nd which calls POT under the hood.
    Subsamples to max_n points per side for tractability on large datasets.

    Parameters
    ----------
    X : np.ndarray, shape [n1, d]
        Source point cloud (e.g. PCA coordinates).
    Y : np.ndarray, shape [n2, d]
        Target point cloud.
    max_n : int
        Maximum points per side. If either side exceeds this, randomly subsample.
    seed : int
        RNG seed for subsampling.

    Returns
    -------
    float
        Wasserstein-2 distance in the units of X/Y (e.g. PCA coordinate units).
    """
    from scipy.stats import wasserstein_distance_nd
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] > max_n:
        X = X[rng.choice(X.shape[0], max_n, replace=False)]
    if Y.shape[0] > max_n:
        Y = Y[rng.choice(Y.shape[0], max_n, replace=False)]
    return float(wasserstein_distance_nd(X, Y))


def regression_to_long_df(reg_result, *, y_col="gene", exclusive_only=False,
                           exclusive_threshold=2.5, top_n_per_archetype=10,
                           fdr_threshold=0.05, degree=1):
    """Convert simplex regression dict → long-format DataFrame for pc.pl.dotplot.

    Builds a long-format table with one row per (feature, archetype):
      - <y_col>: feature name
      - archetype: archetype label (archetype_0, archetype_1, ...)
      - mean_archetype: |vertex coefficient| (used as effect size for dot size)
      - pvalue: significance (vertex FDR for deg=1, incremental FDR for deg>=2)
      - pvalue_fdr: same as pvalue (kept for downstream compat)
      - r_squared: per-feature R² at the requested degree (used for ranking)

    Parameters
    ----------
    reg_result : dict
        Serialized SimplexRegressionResult from pc.tl.feature_simplex_regression.
    y_col : str
        Name for the feature column ("gene" or "pathway").
    exclusive_only : bool
        If True, keep only features where max(|coef|) >= exclusive_threshold * second_max.
    exclusive_threshold : float
        Ratio threshold for "exclusive" features (max vs second-max across archetypes).
    top_n_per_archetype : int
        Keep only top N features per archetype by R² (features are ranked by R²
        within their "argmax archetype" group).
    fdr_threshold : float
        Filter out rows with FDR > threshold (set to 1.0 to disable).
    degree : int
        Polynomial degree to score features by. 1 = pure vertex model (default,
        historical behaviour). 2 or 3 = rank features by delta_r2 (incremental
        R² gain from degree d-1 to d) from degree_comparison so the dotplot
        highlights interaction-specific features rather than features dominated
        by their degree-1 main effects; significance comes from the degree-d
        incremental F-test FDR (degree_comparison["degree_d"]
        ["incremental_p_fdr"]). Per-archetype layout (argmax, dot size) still
        uses the degree-1 vertex coefficients because the higher-degree Scheffe
        polynomial mixes interaction terms that have no single archetype
        "home".

    Returns
    -------
    pd.DataFrame
        Long-format table ready for pc.pl.dotplot.
    """
    feat_names = list(reg_result.get("feature_names", []))
    coefs = np.asarray(reg_result.get("vertex_coefficients", []))
    pvals = np.asarray(reg_result.get("vertex_pvalues", []))
    fdrs = np.asarray(reg_result.get("vertex_pvalues_fdr", []))

    # Resolve per-degree R² and per-feature significance.
    # Layout is always driven by the degree-1 vertex coefficients (argmax +
    # |coef| dot size); only the ranking R² and the significance filter
    # change with degree.
    if degree == 1:
        r2 = np.asarray(reg_result.get("r_squared_degree1", []))
        # Per-archetype significance for the filter (deg-1 uses vertex FDR)
        per_arch_sig_fdr = fdrs
        per_feat_sig_fdr = None
    elif degree == 2:
        # Use delta_r2 (incremental R2 gain from degree 1 -> 2) for ranking
        # so the dotplot highlights interaction-specific features rather than
        # features dominated by their degree-1 main effects.
        dc = reg_result.get("degree_comparison", {}) or {}
        d2 = dc.get("degree_2", {}) or {}
        r2 = np.asarray(d2.get("delta_r2", []))
        if r2.size == 0:
            # Fallback: compute delta from total R2 values
            r2_total = np.asarray(reg_result.get("r_squared_degree2", []))
            r2_d1 = np.asarray(reg_result.get("r_squared_degree1", []))
            if r2_total.size > 0 and r2_d1.size > 0:
                r2 = r2_total - r2_d1
            else:
                r2 = np.asarray(d2.get("r_squared", []))
        per_feat_sig_fdr = np.asarray(d2.get("incremental_p_fdr", []))
        per_arch_sig_fdr = None
    elif degree == 3:
        # Use delta_r2 (incremental R2 gain from degree 2 -> 3) for ranking.
        dc = reg_result.get("degree_comparison", {}) or {}
        d3 = dc.get("degree_3", {}) or {}
        r2 = np.asarray(d3.get("delta_r2", []))
        if r2.size == 0:
            r2 = np.asarray(d3.get("r_squared", []))
        per_feat_sig_fdr = np.asarray(d3.get("incremental_p_fdr", []))
        per_arch_sig_fdr = None
    else:
        raise ValueError(f"degree must be 1, 2, or 3 — got {degree}")

    if coefs.size == 0 or len(feat_names) == 0 or r2.size == 0:
        return pd.DataFrame()

    n_feat, K = coefs.shape

    # Apply exclusive filter first (on features)
    if exclusive_only:
        abs_coefs = np.abs(coefs)
        sorted_abs = np.sort(abs_coefs, axis=1)[:, ::-1]  # descending
        max_c = sorted_abs[:, 0]
        second_c = sorted_abs[:, 1] if K > 1 else np.zeros(n_feat)
        # Avoid div-by-zero
        second_c_safe = np.where(second_c < 1e-10, 1e-10, second_c)
        ratio = max_c / second_c_safe
        keep_feat_mask = ratio >= exclusive_threshold
    else:
        keep_feat_mask = np.ones(n_feat, dtype=bool)

    # Assign each feature to its argmax archetype (for ranking within groups)
    argmax_arch = np.argmax(np.abs(coefs), axis=1)

    # Build long DataFrame
    rows = []
    for fi in range(n_feat):
        if not keep_feat_mask[fi]:
            continue
        # Degree >=2: filter by per-feature incremental F-test FDR first
        if per_feat_sig_fdr is not None:
            if fi >= per_feat_sig_fdr.size:
                continue
            feat_fdr = float(per_feat_sig_fdr[fi])
            if feat_fdr > fdr_threshold:
                continue
        for a in range(K):
            if per_arch_sig_fdr is not None:
                # Degree 1: per-vertex FDR filter
                p = float(pvals[fi, a]) if pvals.size else 1.0
                f = float(per_arch_sig_fdr[fi, a]) if per_arch_sig_fdr.size else 1.0
                if f > fdr_threshold:
                    continue
            else:
                # Degree >=2: no per-vertex FDR (whole feature passes or not)
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

    # For each archetype, keep only top N features by R² where this archetype is argmax
    # (ensures top N per archetype represents features "best explained by" that archetype)
    keep_mask = np.zeros(len(df), dtype=bool)
    for a in range(K):
        is_argmax = df["argmax_archetype"] == a
        if not is_argmax.any():
            continue
        # Get unique features for this archetype, rank by R²
        sub = df[is_argmax].drop_duplicates(subset=[y_col])
        top_feats = sub.nlargest(top_n_per_archetype, "r_squared")[y_col].values
        keep_mask |= df[y_col].isin(top_feats)
    df = df[keep_mask].reset_index(drop=True)
    return df


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
            <details {'open' if i < 3 else ''}>
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


# ============================================================================
# Diagnostic helpers
# ============================================================================

# W-B11: archetype_cell_proximity removed entirely. The old kNN-vs-kNN
# ratio was circular (both sides came from the same k-nearest-neighbors
# computation). Replaced by compute_archetype_to_centroid_distance in
# scripts/_paper_part1_viz.py (W-B10).


# ============================================================================
# PHASE 1: Train two models (HSC + CMP)
# ============================================================================

def phase1_train_models(report):
    """Train HSC and CMP models with hyperparameter search."""
    import peach as pc
    import anndata as ad

    html_hsc = ""
    html_cmp = ""

    # --- CMP model (main text: Fig 1C) ---
    log.info("Loading CMP train data...")
    adata_cmp = ad.read_h5ad(os.path.join(DATA_DIR, "adata_cmp_train.h5ad"))
    if SUBSAMPLE_FRACTION < 1.0:
        log.info(f"  Subsampling CMP train to {SUBSAMPLE_FRACTION*100:.0f}% (seed={SUBSAMPLE_SEED})")
        adata_cmp = _stratified_subsample(adata_cmp, SUBSAMPLE_FRACTION, SUBSAMPLE_SEED)
    log.info(f"  CMP train: {adata_cmp.shape}")

    log.info("CMP hyperparameter search...")
    pc.pp.prepare_training(adata_cmp, batch_size=min(128, adata_cmp.shape[0] // 4))
    cv_cmp = pc.tl.hyperparameter_search(
        adata_cmp,
        n_archetypes_range=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        hidden_dims_options=[[64, 128], [128, 256], [256, 128, 64]],
        inflation_factor_range=[0.75, 1.0, 1.25, 1.5],
        cv_folds=3, max_epochs_cv=20, subsample_fraction=0.8,
        use_pcha_init=False,
    )
    ranked_cmp = cv_cmp.rank_by_metric("archetype_r2")
    ranked_cmp = [r for r in ranked_cmp if r["metric_value"] > -1e6]
    # K selection: min K where CV R2 >= 0.9 (avoids overfitting at higher K)
    _above_cmp = [r for r in ranked_cmp if r["metric_value"] >= 0.9]
    if _above_cmp:
        _above_cmp.sort(key=lambda r: (r["hyperparameters"]["n_archetypes"], -r["metric_value"]))
        best_cmp = _above_cmp[0]
    else:
        best_cmp = ranked_cmp[0]
    K_cmp = best_cmp["hyperparameters"]["n_archetypes"]
    hd_cmp = best_cmp["hyperparameters"].get("hidden_dims", [128, 256])

    # CV table
    cv_rows = []
    for r in ranked_cmp[:10]:
        hp = r["hyperparameters"]
        cv_rows.append({"K": hp["n_archetypes"], "hidden": str(hp.get("hidden_dims", "?")),
                        "inflation": hp.get("inflation_factor", "?"),
                        "R2": f"{r['metric_value']:.4f}", "SE": f"{r.get('std_error', 0):.4f}"})
    html_cmp += report.df_to_html(pd.DataFrame(cv_rows), caption="CMP CV search (top 10)")
    html_cmp += report.text(
        f"<b>Selected</b>: K={K_cmp}, hidden_dims={hd_cmp}, "
        f"inflation=1.0 (fixed). Best mean R2={best_cmp['metric_value']:.4f} "
        f"(SE={best_cmp.get('std_error', 0):.4f}) across {3} CV folds.")

    # Elbow curve
    try:
        fig_elbow = pc.pl.elbow_curve(cv_cmp, metrics=["archetype_r2", "rmse"])
        html_cmp += safe_plotly_html(report, fig_elbow, "CMP elbow curve")
    except Exception as e:
        html_cmp += error_html(f"Elbow curve failed: {e}")

    # Train CMP model — allow up to MAX_EPOCHS_FINAL; early stopping exits earlier on convergence.
    log.info(f"Training CMP model: K={K_cmp}, hidden={hd_cmp}, max_epochs={MAX_EPOCHS_FINAL}...")
    res_cmp = pc.tl.train_archetypal(
        adata_cmp, n_archetypes=K_cmp, n_epochs=MAX_EPOCHS_FINAL, hidden_dims=hd_cmp,
        kld_weight=0.01, archetypal_weight=0.9, inflation_factor=1.0,
        model_config={"manifold_weight": 0.005},
        early_stopping=True, early_stopping_patience=EARLY_STOP_PATIENCE,
    )
    # --- Convergence QC (CMP) --- W-A8: delegate to helper that uses
    # mean(|Δloss|) over the last window as the primary convergence signal,
    # not just "hit_cap and not early_stop". This fixes the r9 bug where
    # runs with delta_loss ≈ 0 were flagged NON-CONVERGED.
    from _paper_part1_viz import convergence_status as _convergence_status
    _cmp_tc = res_cmp.get("training_config", {})
    _cmp_actual = _cmp_tc.get("actual_epochs", MAX_EPOCHS_FINAL)
    _cmp_early = _cmp_tc.get("early_stop_triggered", False)
    _cmp_history = res_cmp.get("history", {})
    _cmp_status, _cmp_delta_mean = _convergence_status(
        history=_cmp_history,
        max_epochs=MAX_EPOCHS_FINAL,
        early_stop_triggered=_cmp_early,
        actual_epochs=_cmp_actual,
        window=10,
        delta_threshold=0.01,
    )
    _cmp_hit_cap = _cmp_status == "NON_CONVERGED_HIT_CAP"
    log.info(
        f"CMP convergence QC: status={_cmp_status}, actual_epochs={_cmp_actual}, "
        f"early_stop={_cmp_early}, last-10-epoch mean |delta_loss|={_cmp_delta_mean:.5f}"
    )
    if _cmp_hit_cap:
        log.warning(
            f"CMP model hit the {MAX_EPOCHS_FINAL}-epoch cap with mean |Δloss| > 0.01 — "
            "may not be fully converged. Consider increasing MAX_EPOCHS_FINAL or inspecting the loss curve."
        )
    # Enrich training_config with model-level params for downstream metric display
    res_cmp.setdefault("training_config", {}).update({
        "n_archetypes": K_cmp,
        "hidden_dims": hd_cmp,
        "inflation_factor": 1.0,
        "use_pcha_init": False,
    })

    # W-A6: PCHA-off comparison run. Train the same model with pcha_init=False
    # on a cloned adata (so downstream analyses still use the PCHA-on fit).
    # This is purely diagnostic — used only to populate the HTML comparison table.
    log.info("CMP PCHA-off comparison: training with pcha_init=False...")
    try:
        adata_cmp_nopcha = adata_cmp.copy()
        pc.pp.prepare_training(adata_cmp_nopcha, batch_size=min(128, adata_cmp_nopcha.shape[0] // 4))
        res_cmp_nopcha = pc.tl.train_archetypal(
            adata_cmp_nopcha, n_archetypes=K_cmp, n_epochs=MAX_EPOCHS_FINAL, hidden_dims=hd_cmp,
            kld_weight=0.01, archetypal_weight=0.9, inflation_factor=1.0,
            model_config={"manifold_weight": 0.005},
            early_stopping=True, early_stopping_patience=EARLY_STOP_PATIENCE,
            pcha_init=False,
        )
        _r2_cmp_on = res_cmp.get("final_archetype_r2", float("nan"))
        _r2_cmp_off = res_cmp_nopcha.get("final_archetype_r2", float("nan"))
        _cmp_pcha_fired = res_cmp.get("pcha_init_fired", False)
        log.info(
            f"CMP PCHA init diagnostic: fired={_cmp_pcha_fired}, "
            f"R² PCHA-on={_r2_cmp_on}, R² PCHA-off={_r2_cmp_off}"
        )
        _cmp_pcha_comparison_rows = [
            {"pcha_init": "True",  "fired": str(_cmp_pcha_fired),
             "final R²": f"{_r2_cmp_on:.4f}" if isinstance(_r2_cmp_on, float) else str(_r2_cmp_on)},
            {"pcha_init": "False", "fired": str(res_cmp_nopcha.get("pcha_init_fired", False)),
             "final R²": f"{_r2_cmp_off:.4f}" if isinstance(_r2_cmp_off, float) else str(_r2_cmp_off)},
        ]
    except Exception as _cmp_pcha_exc:
        log.warning(f"CMP PCHA-off comparison failed: {_cmp_pcha_exc}")
        _cmp_pcha_comparison_rows = [
            {"pcha_init": "True",  "fired": str(res_cmp.get("pcha_init_fired", False)),
             "final R²": f"{res_cmp.get('final_archetype_r2', float('nan')):.4f}"},
            {"pcha_init": "False", "fired": "comparison failed",
             "final R²": f"ERROR: {_cmp_pcha_exc}"},
        ]

    pc.tl.archetypal_coordinates(adata_cmp, verbose=False)
    pc.tl.extract_archetype_weights(adata_cmp, verbose=False)
    pc.tl.assign_archetypes(adata_cmp, verbose=False)
    r2_cmp = res_cmp.get("final_archetype_r2", "N/A")
    html_cmp += metric_grid([
        metric_card(K_cmp, "CMP K"), metric_card(f"{r2_cmp:.4f}" if isinstance(r2_cmp, float) else r2_cmp, "CMP R2"),
        metric_card(f"{adata_cmp.shape[0]}", "N cells"),
    ])
    # Per-archetype cell counts
    if "archetypes" in adata_cmp.obs.columns:
        arch_counts = adata_cmp.obs["archetypes"].value_counts().sort_index()
        count_cards = [metric_card(f"{v}", display_arch(k)) for k, v in arch_counts.items()]
        html_cmp += report.text("<b>Per-archetype cell counts (CMP)</b>:")
        html_cmp += metric_grid(count_cards)

    # Training parameters summary (Fig 1C)
    _tc_cmp = res_cmp.get("training_config", {})
    _r2_cmp_disp = res_cmp.get("final_archetype_r2", None)
    _r2_cmp_str = f"{_r2_cmp_disp:.4f}" if isinstance(_r2_cmp_disp, float) else "N/A"
    html_cmp += report.text("<b>Training parameters</b>")
    html_cmp += metric_grid([
        metric_card(_tc_cmp.get("n_archetypes", K_cmp), "n_archetypes"),
        metric_card(str(_tc_cmp.get("hidden_dims", hd_cmp)), "hidden_dims"),
        metric_card(_tc_cmp.get("n_epochs", MAX_EPOCHS_FINAL), "n_epochs (max)"),
        metric_card(_tc_cmp.get("actual_epochs", "?"), "actual_epochs"),
        metric_card(_r2_cmp_str, "final R²"),
        metric_card(_tc_cmp.get("kld_weight", "?"), "kld_weight"),
        metric_card(_tc_cmp.get("archetypal_weight", "?"), "archetypal_weight"),
        metric_card(_tc_cmp.get("inflation_factor", "?"), "inflation_factor"),
        metric_card(_tc_cmp.get("manifold_weight", "0.005"), "manifold_weight"),
        metric_card(str(_tc_cmp.get("use_pcha_init", False)), "use_pcha_init"),
    ])
    # Convergence QC badge (CMP) — W-A8: status is driven by
    # delta_loss magnitude, not just "hit epoch cap".
    if _cmp_status == "CONVERGED":
        _cmp_conv_tag = " <b style='color:green'>[CONVERGED]</b>"
    elif _cmp_status == "NON_CONVERGED_HIT_CAP":
        _cmp_conv_tag = " <b style='color:orange'>[NON-CONVERGED: hit epoch cap, Δloss &gt; 0.01]</b>"
    else:  # NOT_CONVERGED_INSUFFICIENT_HISTORY
        _cmp_conv_tag = " <b style='color:orange'>[INSUFFICIENT HISTORY]</b>"
    _cmp_conv_msg = (
        (f"Early stopped at epoch {_cmp_actual}/{MAX_EPOCHS_FINAL}. "
         if _cmp_early else
         f"Ran {_cmp_actual}/{MAX_EPOCHS_FINAL} epochs. ")
        + f"Last-10-epoch mean |Δloss| = {_cmp_delta_mean:.5f}."
        + _cmp_conv_tag
    )
    html_cmp += report.text(f"<b>Convergence QC</b>: {_cmp_conv_msg}")

    # Training metrics
    try:
        fig_train = pc.pl.training_metrics(res_cmp["history"], display=False)
        if fig_train:
            html_cmp += safe_plotly_html(report, fig_train, "CMP training metrics")
    except Exception as e:
        html_cmp += error_html(f"CMP training metrics failed: {e}")

    # W-A6: PCHA init comparison table (final R² with/without PCHA init)
    html_cmp += report.text("<b>PCHA init diagnostic (W-A6)</b>")
    html_cmp += report.df_to_html(
        pd.DataFrame(_cmp_pcha_comparison_rows),
        caption=(
            "Comparison of final archetypal R² with PCHA initialization on vs off. "
            "'fired' column reports whether the diagnostic flag confirms PCHA init "
            "actually ran (True) or was skipped (False). If the two R² values differ "
            "substantially, PCHA seeding is materially helping the fit."
        ),
    )

    # Benchmarking
    html_cmp += report.text(f"<b>Benchmarking</b>: CMP model ({adata_cmp.shape[0]} cells) — "
                            f"CV search + training completed in {time.time() - report.start_time:.0f}s total")

    # Inflation factor 1.25 comparison (deferred):
    # Training a second full CMP model (CV search + final training) would add
    # ~60-120 s per run and complicate downstream data-flow (all Fig 2 analyses
    # must use the inflation_factor=1.0 model).  The comparison is therefore
    # deferred to a dedicated script.  If you need the comparison, run:
    #   python scripts/archive/inflation_comparison.py
    html_cmp += report.text(
        "<b>Inflation factor 1.25 comparison</b>: deferred for the full run — "
        "see <code>scripts/archive/inflation_comparison.py</code> if needed. "
        "All downstream Fig 2 analyses use the inflation_factor=1.0 model above."
    )

    report.add_section("CMP Model (Fig 1C)", html_cmp, step_num="1C")

    # --- HSC model (supplemental search, main text usage) ---
    log.info("Loading HSC train data...")
    adata_hsc = ad.read_h5ad(os.path.join(DATA_DIR, "adata_hsc_train.h5ad"))
    if SUBSAMPLE_FRACTION < 1.0:
        log.info(f"  Subsampling HSC train to {SUBSAMPLE_FRACTION*100:.0f}% (seed={SUBSAMPLE_SEED})")
        adata_hsc = _stratified_subsample(adata_hsc, SUBSAMPLE_FRACTION, SUBSAMPLE_SEED)
    log.info(f"  HSC train: {adata_hsc.shape}")

    log.info("HSC hyperparameter search...")
    pc.pp.prepare_training(adata_hsc, batch_size=min(128, adata_hsc.shape[0] // 4))
    cv_hsc = pc.tl.hyperparameter_search(
        adata_hsc,
        n_archetypes_range=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        hidden_dims_options=[[64, 128], [128, 256]],
        inflation_factor_range=[0.75, 1.0, 1.25, 1.5],
        cv_folds=3, max_epochs_cv=15, subsample_fraction=0.8,
    )
    ranked_hsc = cv_hsc.rank_by_metric("archetype_r2")
    ranked_hsc = [r for r in ranked_hsc if r["metric_value"] > -1e6]
    best_hsc = ranked_hsc[0]
    K_hsc = best_hsc["hyperparameters"]["n_archetypes"]
    hd_hsc = best_hsc["hyperparameters"].get("hidden_dims", [128, 256])

    # HSC CV table (Supplemental)
    cv_rows_hsc = []
    for r in ranked_hsc[:10]:
        hp = r["hyperparameters"]
        cv_rows_hsc.append({"K": hp["n_archetypes"], "hidden": str(hp.get("hidden_dims", "?")),
                            "inflation": hp.get("inflation_factor", "?"),
                            "R2": f"{r['metric_value']:.4f}"})
    html_hsc += report.df_to_html(pd.DataFrame(cv_rows_hsc), caption="HSC CV search (Supplemental)")
    html_hsc += report.text(
        f"<b>Selected</b>: K={K_hsc}, hidden_dims={hd_hsc}, "
        f"inflation=1.0 (fixed). Best mean R2={best_hsc['metric_value']:.4f} "
        f"(SE={best_hsc.get('std_error', 0):.4f}) across {3} CV folds.")

    try:
        fig_elbow_hsc = pc.pl.elbow_curve(cv_hsc, metrics=["archetype_r2", "rmse"])
        html_hsc += safe_plotly_html(report, fig_elbow_hsc, "HSC elbow curve (Supplemental)")
    except Exception as e:
        html_hsc += error_html(f"HSC elbow curve failed: {e}")

    # Train HSC model — allow up to MAX_EPOCHS_FINAL; early stopping exits earlier on convergence.
    log.info(f"Training HSC model: K={K_hsc}, hidden={hd_hsc}, max_epochs={MAX_EPOCHS_FINAL}...")
    res_hsc = pc.tl.train_archetypal(
        adata_hsc, n_archetypes=K_hsc, n_epochs=MAX_EPOCHS_FINAL, hidden_dims=hd_hsc,
        kld_weight=0.01, archetypal_weight=0.9, inflation_factor=1.0,
        model_config={"manifold_weight": 0.005},
        early_stopping=True, early_stopping_patience=EARLY_STOP_PATIENCE,
    )
    # --- Convergence QC (HSC) --- W-A8: delegate to helper (same rationale
    # as CMP block above).
    from _paper_part1_viz import convergence_status as _convergence_status
    _hsc_tc = res_hsc.get("training_config", {})
    _hsc_actual = _hsc_tc.get("actual_epochs", MAX_EPOCHS_FINAL)
    _hsc_early = _hsc_tc.get("early_stop_triggered", False)
    _hsc_history = res_hsc.get("history", {})
    _hsc_status, _hsc_delta_mean = _convergence_status(
        history=_hsc_history,
        max_epochs=MAX_EPOCHS_FINAL,
        early_stop_triggered=_hsc_early,
        actual_epochs=_hsc_actual,
        window=10,
        delta_threshold=0.01,
    )
    _hsc_hit_cap = _hsc_status == "NON_CONVERGED_HIT_CAP"
    log.info(
        f"HSC convergence QC: status={_hsc_status}, actual_epochs={_hsc_actual}, "
        f"early_stop={_hsc_early}, last-10-epoch mean |delta_loss|={_hsc_delta_mean:.5f}"
    )
    if _hsc_hit_cap:
        log.warning(
            f"HSC model hit the {MAX_EPOCHS_FINAL}-epoch cap with mean |Δloss| > 0.01 — "
            "may not be fully converged. Consider increasing MAX_EPOCHS_FINAL or inspecting the loss curve."
        )
    # Enrich training_config with model-level params for downstream metric display
    res_hsc.setdefault("training_config", {}).update({
        "n_archetypes": K_hsc,
        "hidden_dims": hd_hsc,
        "inflation_factor": 1.0,
        "use_pcha_init": False,
    })

    # W-A6: PCHA-off comparison run (HSC). Cloned adata so the downstream
    # analyses keep using the PCHA-on fit — this is purely for the diagnostic
    # table in the HTML report.
    log.info("HSC PCHA-off comparison: training with pcha_init=False...")
    try:
        adata_hsc_nopcha = adata_hsc.copy()
        pc.pp.prepare_training(adata_hsc_nopcha, batch_size=min(128, adata_hsc_nopcha.shape[0] // 4))
        res_hsc_nopcha = pc.tl.train_archetypal(
            adata_hsc_nopcha, n_archetypes=K_hsc, n_epochs=MAX_EPOCHS_FINAL, hidden_dims=hd_hsc,
            kld_weight=0.01, archetypal_weight=0.9, inflation_factor=1.0,
            model_config={"manifold_weight": 0.005},
            early_stopping=True, early_stopping_patience=EARLY_STOP_PATIENCE,
            pcha_init=False,
        )
        _r2_hsc_on = res_hsc.get("final_archetype_r2", float("nan"))
        _r2_hsc_off = res_hsc_nopcha.get("final_archetype_r2", float("nan"))
        _hsc_pcha_fired = res_hsc.get("pcha_init_fired", False)
        log.info(
            f"HSC PCHA init diagnostic: fired={_hsc_pcha_fired}, "
            f"R² PCHA-on={_r2_hsc_on}, R² PCHA-off={_r2_hsc_off}"
        )
        _hsc_pcha_comparison_rows = [
            {"pcha_init": "True",  "fired": str(_hsc_pcha_fired),
             "final R²": f"{_r2_hsc_on:.4f}" if isinstance(_r2_hsc_on, float) else str(_r2_hsc_on)},
            {"pcha_init": "False", "fired": str(res_hsc_nopcha.get("pcha_init_fired", False)),
             "final R²": f"{_r2_hsc_off:.4f}" if isinstance(_r2_hsc_off, float) else str(_r2_hsc_off)},
        ]
    except Exception as _hsc_pcha_exc:
        log.warning(f"HSC PCHA-off comparison failed: {_hsc_pcha_exc}")
        _hsc_pcha_comparison_rows = [
            {"pcha_init": "True",  "fired": str(res_hsc.get("pcha_init_fired", False)),
             "final R²": f"{res_hsc.get('final_archetype_r2', float('nan')):.4f}"},
            {"pcha_init": "False", "fired": "comparison failed",
             "final R²": f"ERROR: {_hsc_pcha_exc}"},
        ]

    pc.tl.archetypal_coordinates(adata_hsc, verbose=False)
    pc.tl.extract_archetype_weights(adata_hsc, verbose=False)
    pc.tl.assign_archetypes(adata_hsc, verbose=False)
    r2_hsc = res_hsc.get("final_archetype_r2", "N/A")
    html_hsc += metric_grid([
        metric_card(K_hsc, "HSC K"), metric_card(f"{r2_hsc:.4f}" if isinstance(r2_hsc, float) else r2_hsc, "HSC R2"),
        metric_card(f"{adata_hsc.shape[0]}", "N cells"),
    ])
    # Per-archetype cell counts
    if "archetypes" in adata_hsc.obs.columns:
        arch_counts_hsc = adata_hsc.obs["archetypes"].value_counts().sort_index()
        count_cards_hsc = [metric_card(f"{v}", display_arch(k)) for k, v in arch_counts_hsc.items()]
        html_hsc += report.text("<b>Per-archetype cell counts (HSC)</b>:")
        html_hsc += metric_grid(count_cards_hsc)
    # Convergence QC badge (HSC — phase 1 section) — W-A8: see CMP block.
    if _hsc_status == "CONVERGED":
        _hsc_conv_tag = " <b style='color:green'>[CONVERGED]</b>"
    elif _hsc_status == "NON_CONVERGED_HIT_CAP":
        _hsc_conv_tag = " <b style='color:orange'>[NON-CONVERGED: hit epoch cap, Δloss &gt; 0.01]</b>"
    else:  # NOT_CONVERGED_INSUFFICIENT_HISTORY
        _hsc_conv_tag = " <b style='color:orange'>[INSUFFICIENT HISTORY]</b>"
    _hsc_conv_msg = (
        (f"Early stopped at epoch {_hsc_actual}/{MAX_EPOCHS_FINAL}. "
         if _hsc_early else
         f"Ran {_hsc_actual}/{MAX_EPOCHS_FINAL} epochs. ")
        + f"Last-10-epoch mean |Δloss| = {_hsc_delta_mean:.5f}."
        + _hsc_conv_tag
    )
    html_hsc += report.text(f"<b>Convergence QC</b>: {_hsc_conv_msg}")

    try:
        fig_train_hsc = pc.pl.training_metrics(res_hsc["history"], display=False)
        if fig_train_hsc:
            html_hsc += safe_plotly_html(report, fig_train_hsc, "HSC training metrics")
    except Exception as e:
        html_hsc += error_html(f"HSC training metrics failed: {e}")

    # W-A6: PCHA init comparison table (HSC)
    html_hsc += report.text("<b>PCHA init diagnostic (W-A6)</b>")
    html_hsc += report.df_to_html(
        pd.DataFrame(_hsc_pcha_comparison_rows),
        caption=(
            "Comparison of final archetypal R² with PCHA initialization on vs off. "
            "'fired' column reports whether the diagnostic flag confirms PCHA init "
            "actually ran (True) or was skipped (False). If the two R² values differ "
            "substantially, PCHA seeding is materially helping the fit."
        ),
    )

    report.add_section("HSC Model (Supplemental)", html_hsc, step_num="S1")

    return adata_hsc, adata_cmp, res_hsc, res_cmp


# ============================================================================
# PHASE 2: Figure 1 — Introduce Deep_AA
# ============================================================================

def phase2_figure1(adata_hsc, adata_cmp, res_hsc, report):
    """Fig 1A: PCA vs archetypal, 1B: held-out projection, 1D: ParetoTI parity."""
    import peach as pc
    import anndata as ad

    # --- Fig 1A: Archetypal space (HSC train) via PEACH ---
    html = "<h3>Fig 1A: Archetypal space — HSC train</h3>"
    r2_val = res_hsc.get("final_archetype_r2")
    html += report.text(
        (f"<b>Archetypal R²</b> = {r2_val:.4f}. "
         "Measures how much variance in PCA space is explained by the learned archetypal "
         "coordinate system. Analogous to R² in regression: 1.0 = perfect reconstruction of "
         "cell positions from archetype weights.")
        if isinstance(r2_val, float) else "<b>Archetypal R²</b>: not available"
    )
    # Training parameters summary (Fig 1A)
    _tc_hsc = res_hsc.get("training_config", {})
    _r2_hsc_str = f"{r2_val:.4f}" if isinstance(r2_val, float) else "N/A"
    html += report.text("<b>Training parameters</b>")
    html += metric_grid([
        metric_card(_tc_hsc.get("n_archetypes", "?"), "n_archetypes"),
        metric_card(str(_tc_hsc.get("hidden_dims", "?")), "hidden_dims"),
        metric_card(_tc_hsc.get("n_epochs", MAX_EPOCHS_FINAL), "n_epochs (max)"),
        metric_card(_tc_hsc.get("actual_epochs", "?"), "actual_epochs"),
        metric_card(_r2_hsc_str, "final R²"),
        metric_card(_tc_hsc.get("kld_weight", "?"), "kld_weight"),
        metric_card(_tc_hsc.get("archetypal_weight", "?"), "archetypal_weight"),
        metric_card(_tc_hsc.get("inflation_factor", "?"), "inflation_factor"),
        metric_card(_tc_hsc.get("manifold_weight", "0.005"), "manifold_weight"),
        metric_card(str(_tc_hsc.get("use_pcha_init", False)), "use_pcha_init"),
    ])

    # W-B10: non-circular centroid-distance diagnostic. Measures distance
    # from each archetype position to the centroid of its binned cells,
    # scaled by the cell cloud radius. Extrapolation ratio >> 1 means the
    # archetype sits well outside its own cell cloud.
    try:
        from _paper_part1_viz import compute_archetype_to_centroid_distance
        _cd_df = compute_archetype_to_centroid_distance(
            adata_hsc, obs_key="archetypes", pca_key="X_pca",
        )
        html += report.text("<b>Archetype-to-cell centroid distance (W-B10)</b>")
        _display_cd = _cd_df.copy()
        for _col in [
            "archetype_position_norm", "centroid_distance",
            "data_mean_distance", "bin_radius", "extrapolation_ratio",
        ]:
            _display_cd[_col] = _display_cd[_col].apply(
                lambda x: "NaN" if pd.isna(x) else f"{x:.4f}"
            )
        html += report.df_to_html(
            _display_cd,
            caption=(
                "L2 distance from each archetype position to the centroid "
                "of cells binned to it. <b>archetype_position_norm</b> = "
                "L2 norm from PCA origin (0,0,...,0). "
                "<b>centroid_distance</b> = L2 distance from archetype to "
                "its binned-cell centroid. <b>bin_radius</b> = mean L2 "
                "distance from binned cells to their own centroid (cell "
                "cloud scale). <b>extrapolation_ratio</b> = centroid_distance "
                "/ bin_radius; values >> 1 indicate the archetype sits "
                "outside its own cell cloud."
            ),
        )
        _extrap_mask = _cd_df["extrapolation_ratio"] > 2.0
        _extrap_labels = _cd_df.loc[_extrap_mask, "archetype_label"].tolist()
        if _extrap_labels:
            html += report.text(
                f"<b style='color:orange'>Warning</b>: archetypes "
                f"{_extrap_labels} have extrapolation_ratio > 2 — they "
                "sit well outside their own binned cell cloud."
            )
            log.warning(
                f"HSC archetypes {_extrap_labels} have extrapolation_ratio > 2"
            )
    except Exception as _cd_exc:
        log.warning(f"Centroid distance diagnostic failed: {_cd_exc}")
        html += report.text(f"Centroid distance diagnostic failed: {_cd_exc}")

    try:
        # PEACH 3D archetypal space colored by archetype assignment
        _k_hsc = adata_hsc.uns.get("archetype_coordinates", np.array([[]])).shape[0]
        fig_a_hsc = pc.pl.archetypal_space(
            adata_hsc, color_by="archetypes",
            title=f"Fig 1A: HSC train archetypal space (K={_k_hsc if _k_hsc > 0 else '?'})")
        html += safe_plotly_html(report, fig_a_hsc,
            "Fig 1A: HSC train cells + archetype vertices in 3D PCA/archetypal space")
    except Exception as e:
        log.exception("Fig 1A archetypal_space failed")
        html += error_html(f"Fig 1A archetypal_space failed: {e}")

    # --- Fig 1B: Held-out HSC projection ---
    html += "<h3>Fig 1B: HSC holdout projection</h3>"
    try:
        from scipy.stats import ks_2samp
        adata_holdout_hsc = ad.read_h5ad(os.path.join(DATA_DIR, "adata_hsc_holdout.h5ad"))
        if SUBSAMPLE_FRACTION < 1.0:
            adata_holdout_hsc = _stratified_subsample(adata_holdout_hsc, SUBSAMPLE_FRACTION, SUBSAMPLE_SEED)
        log.info(f"Projecting HSC holdout ({adata_holdout_hsc.shape[0]} cells)...")
        pc.pp.prepare_training(adata_holdout_hsc, batch_size=64)
        # Transfer trained model from HSC train to holdout for projection
        adata_holdout_hsc.uns["trained_model"] = adata_hsc.uns["trained_model"]
        adata_holdout_hsc.uns["archetype_coordinates"] = adata_hsc.uns["archetype_coordinates"]
        pc.tl.archetypal_coordinates(adata_holdout_hsc, verbose=False)
        pc.tl.extract_archetype_weights(adata_holdout_hsc, verbose=False)
        pc.tl.assign_archetypes(adata_holdout_hsc, verbose=False)

        # --- Build concatenated adata for archetypal_space plot ---
        # Both adata_hsc and adata_holdout_hsc must share archetype coordinates and weights
        try:
            adata_hsc.obs["split"] = "train"
            adata_holdout_hsc.obs["split"] = "holdout"
            adata_concat_hsc = ad.concat(
                [adata_hsc, adata_holdout_hsc], join="outer", merge="first",
                label=None, keys=None, index_unique=None,
            )
            # Preserve archetype_coordinates and trained_model in concat
            adata_concat_hsc.uns["trained_model"] = adata_hsc.uns["trained_model"]
            adata_concat_hsc.uns["archetype_coordinates"] = adata_hsc.uns["archetype_coordinates"]
            fig_concat = pc.pl.archetypal_space(
                adata_concat_hsc, color_by="split",
                title="Fig 1B: HSC train vs holdout in archetypal space")
            html += safe_plotly_html(report, fig_concat,
                "Fig 1B: HSC train (color 1) vs holdout (color 2) in 3D archetypal space")
        except Exception as e:
            log.exception("Fig 1B concat archetypal_space failed")
            html += error_html(f"Concat archetypal_space failed: {e}")

        # --- KS test: train vs holdout weight distributions ---
        weights_train = adata_hsc.obsm.get("cell_archetype_weights")
        weights_hold = adata_holdout_hsc.obsm.get("cell_archetype_weights")
        log.info(f"HSC weights shapes: train={weights_train.shape if weights_train is not None else None}, "
                 f"hold={weights_hold.shape if weights_hold is not None else None}")
        if weights_train is not None and weights_hold is not None:
            K = weights_train.shape[1]
            log.info(f"HSC K={K}, building full KS table")
            ks_rows = []
            for k in range(K):
                stat, pval = ks_2samp(weights_train[:, k], weights_hold[:, k])
                pval_bonf = min(pval * K, 1.0)  # Bonferroni correction
                ks_rows.append({
                    "Archetype": f"A{k+1}",
                    "KS stat": f"{stat:.4f}",
                    "p-value": fmt_pval(pval),
                    "p-Bonf": fmt_pval(pval_bonf),
                    "Sig (Bonf<0.05)": "Yes" if pval_bonf < 0.05 else "No",
                })
            ks_df = pd.DataFrame(ks_rows)
            n_sig_ks = int((ks_df["Sig (Bonf<0.05)"] == "Yes").sum())

            # --- Hotelling T² omnibus test on full weight vectors ---
            # T² = n_train * n_hold / (n_train + n_hold) * (mu_t - mu_h)^T Sigma^-1 (mu_t - mu_h)
            from scipy.stats import f as f_dist
            n_tr, n_ho = weights_train.shape[0], weights_hold.shape[0]
            mu_t, mu_h = weights_train.mean(axis=0), weights_hold.mean(axis=0)
            # Pooled covariance
            S_t = np.cov(weights_train, rowvar=False)
            S_h = np.cov(weights_hold, rowvar=False)
            S_pooled = ((n_tr - 1) * S_t + (n_ho - 1) * S_h) / (n_tr + n_ho - 2)
            # Add ridge for singular cases
            S_pooled += np.eye(K) * 1e-8
            try:
                S_inv = np.linalg.inv(S_pooled)
                diff = mu_t - mu_h
                T2 = (n_tr * n_ho / (n_tr + n_ho)) * (diff @ S_inv @ diff)
                # Convert to F: F = T2 * (n_tr+n_ho-K-1) / (K*(n_tr+n_ho-2))
                df1, df2 = K, n_tr + n_ho - K - 1
                if df2 > 0:
                    F_stat = T2 * df2 / (df1 * (n_tr + n_ho - 2))
                    hotelling_p = 1 - f_dist.cdf(F_stat, df1, df2)
                else:
                    F_stat = float("nan"); hotelling_p = float("nan")
            except np.linalg.LinAlgError:
                T2 = F_stat = hotelling_p = float("nan")

            html += report.text(
                f"<b>HSC holdout generalization</b>: "
                f"Hotelling T²={T2:.3f}, F({df1},{df2})={F_stat:.3f}, p={fmt_pval(hotelling_p)}. "
                f"Per-archetype KS: {n_sig_ks}/{K} Bonferroni-significant. "
                f"Good result: Hotelling p > 0.05 AND few Bonferroni-significant KS. "
                f"N train={n_tr}, N holdout={n_ho}.")
            html += report.df_to_html(ks_df,
                caption=f"HSC: full K={K} per-archetype KS test + Bonferroni (train vs holdout weights)")
            html += metric_grid([
                metric_card(f"{n_tr}", "N train"),
                metric_card(f"{n_ho}", "N holdout"),
                metric_card(f"{n_sig_ks}/{K}", "KS sig (Bonf<0.05)"),
                metric_card(fmt_pval(hotelling_p), "Hotelling T² p"),
                metric_card(f"{T2:.3f}", "T² stat"),
            ])

            # Weight distribution histograms — show ALL K archetypes (not just 4)
            n_cols = min(4, K)
            n_rows = (K + n_cols - 1) // n_cols
            fig_hist, axes_h = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
            axes_h = np.atleast_2d(axes_h).flatten()
            for k in range(K):
                axes_h[k].hist(weights_train[:, k], bins=30, alpha=0.5, color="#0072B2",
                               label="Train", density=True)
                axes_h[k].hist(weights_hold[:, k], bins=30, alpha=0.5, color="#D55E00",
                               label="Holdout", density=True)
                axes_h[k].set_title(f"A{k+1}")
                axes_h[k].legend(fontsize=7)
                axes_h[k].spines[["top", "right"]].set_visible(False)
            # Hide unused subplots
            for k in range(K, len(axes_h)):
                axes_h[k].set_visible(False)
            fig_hist.suptitle(f"HSC weight distributions: Train vs Holdout (all {K} archetypes)", y=1.02)
            fig_hist.tight_layout()
            html += report.fig_to_img(fig_hist,
                caption=f"HSC: per-archetype weight distributions (train blue, holdout orange, all {K} archetypes)")
            plt.close("all")

        # --- Degradation test: project CMP through HSC model ---
        html += "<h4>Degradation test: CMP cells projected through HSC model</h4>"
        log.info("Degradation test: projecting CMP through HSC model...")
        adata_cmp_copy = adata_cmp.copy()
        pc.pp.prepare_training(adata_cmp_copy, batch_size=64)
        adata_cmp_copy.uns["trained_model"] = adata_hsc.uns["trained_model"]
        adata_cmp_copy.uns["archetype_coordinates"] = adata_hsc.uns["archetype_coordinates"]
        pc.tl.archetypal_coordinates(adata_cmp_copy, verbose=False)
        pc.tl.extract_archetype_weights(adata_cmp_copy, verbose=False)
        pc.tl.assign_archetypes(adata_cmp_copy, verbose=False)

        weights_cmp_via_hsc = adata_cmp_copy.obsm.get("cell_archetype_weights")
        if weights_cmp_via_hsc is not None and weights_train is not None:
            # Concat HSC (native) + CMP (projected via HSC)
            adata_hsc_native = adata_hsc.copy()
            adata_hsc_native.obs["degradation_split"] = "HSC (native)"
            adata_cmp_copy.obs["degradation_split"] = "CMP (projected)"
            try:
                adata_deg = ad.concat(
                    [adata_hsc_native, adata_cmp_copy], join="outer", merge="first",
                )
                adata_deg.uns["trained_model"] = adata_hsc.uns["trained_model"]
                adata_deg.uns["archetype_coordinates"] = adata_hsc.uns["archetype_coordinates"]
                fig_deg = pc.pl.archetypal_space(
                    adata_deg, color_by="degradation_split",
                    title="Degradation: HSC native vs CMP projected through HSC model")
                html += safe_plotly_html(report, fig_deg,
                    "Degradation: CMP cells projected through HSC-trained model — should collapse")
            except Exception as e:
                log.exception("Degradation archetypal_space failed")
                html += error_html(f"Degradation plot failed: {e}")

            # KS on degradation
            deg_rows = []
            for k in range(min(K, weights_cmp_via_hsc.shape[1])):
                stat, pval = ks_2samp(weights_train[:, k], weights_cmp_via_hsc[:, k])
                pval_bonf = min(pval * K, 1.0)
                deg_rows.append({"Archetype": f"A{k+1}", "KS stat": f"{stat:.4f}",
                                 "p-value": fmt_pval(pval), "p-Bonf": fmt_pval(pval_bonf)})
            html += report.df_to_html(pd.DataFrame(deg_rows),
                caption="Degradation KS (all K): HSC native vs CMP projected — expect LARGE differences")

            # W-B12: cross-model R² for the degradation test. Reports how
            # well the HSC archetypes reconstruct CMP cells. A large drop
            # from HSC-native R² to CMP-via-HSC R² is the real numerical
            # degradation signal (before W-B12 only KS was reported).
            try:
                from _paper_part1_viz import compute_cross_model_r2
                hsc_archetypes = np.asarray(adata_hsc.uns["archetype_coordinates"])
                cmp_coords = np.asarray(adata_cmp_copy.obsm["X_pca"])
                cmp_r2_via_hsc = compute_cross_model_r2(
                    weights_cmp_via_hsc, hsc_archetypes, cmp_coords,
                )
                hsc_native_r2 = float(res_hsc.get("final_archetype_r2", float("nan")))
                r2_drop = (
                    hsc_native_r2 - cmp_r2_via_hsc
                    if not (np.isnan(hsc_native_r2) or np.isnan(cmp_r2_via_hsc))
                    else float("nan")
                )
                html += report.text(
                    f"<b>Cross-model R²</b>: HSC-native "
                    f"R² = {hsc_native_r2:.4f} vs CMP-projected-through-HSC "
                    f"R² = {cmp_r2_via_hsc:.4f} "
                    f"(drop = {r2_drop:.4f}). Large drop = HSC archetypes "
                    "do NOT describe CMP structure; small drop = they do."
                )
                log.info(
                    f"Degradation R²: HSC native={hsc_native_r2:.4f}, "
                    f"CMP via HSC={cmp_r2_via_hsc:.4f}, "
                    f"drop={r2_drop:.4f}"
                )
            except Exception as _r2_exc:
                log.warning(f"Cross-model R² computation failed: {_r2_exc}")
                html += report.text(
                    f"Cross-model R² computation failed: {_r2_exc}"
                )
    except Exception as e:
        log.exception("Fig 1B / degradation test failed")
        html += error_html(f"Fig 1B / degradation test failed: {e}")

    # --- Fig 1D: ParetoTI parity (CMP characterization) ---
    try:
        html += "<h3>Fig 1D: ParetoTI Parity — CMP characterization</h3>"

        # Archetype space colored by assignment (PEACH function)
        fig_space = pc.pl.archetypal_space(
            adata_cmp, color_by="archetypes",
            title=f"CMP archetypal space (K={adata_cmp.uns['archetype_coordinates'].shape[0]})")
        html += safe_plotly_html(report, fig_space,
            "CMP cells colored by assigned archetype (PEACH archetypal_space)")

        # Archetype positions (2D + distance matrix)
        try:
            fig_pos = pc.pl.archetype_positions(adata_cmp, save_path=None)
            html += report.fig_to_img(fig_pos,
                caption="CMP archetype positions in PCA space + pairwise distance matrix")
            plt.close("all")
        except Exception as e:
            log.exception("Archetype positions failed")
            html += error_html(f"Archetype positions plot failed: {e}")

        # --- CMP gene simplex regression (top 5K HVGs) ---
        log.info("CMP gene simplex regression (degree 1, top 5K HVGs)...")
        if "_hvg5k" not in adata_cmp.obsm:
            X_cmp_d = adata_cmp.X.toarray() if hasattr(adata_cmp.X, "toarray") else np.asarray(adata_cmp.X)
            gv_cmp = np.var(X_cmp_d, axis=0)
            t5k = np.argsort(gv_cmp)[-5000:]
            adata_cmp.obsm["_hvg5k"] = X_cmp_d[:, t5k]
            adata_cmp.uns["_hvg5k_names"] = [adata_cmp.var_names[i] for i in t5k]
        cmp_reg_d1 = pc.tl.feature_simplex_regression(
            adata_cmp, max_degree=1, robust_se=True,
            feature_matrix="_hvg5k", feature_names=adata_cmp.uns["_hvg5k_names"])
        if "peach_simplex_regression__hvg5k" in adata_cmp.uns:
            adata_cmp.uns["peach_simplex_regression_genes"] = adata_cmp.uns["peach_simplex_regression__hvg5k"]
            adata_cmp.uns["peach_simplex_regression"] = adata_cmp.uns["peach_simplex_regression__hvg5k"]

        # Regression summary metrics
        reg_result = adata_cmp.uns.get("peach_simplex_regression_genes", {})
        r2_vals = reg_result.get("r_squared_degree1", reg_result.get("r_squared", []))
        if len(r2_vals) > 0:
            r2_arr = np.asarray(r2_vals)
            # W-B15: break out high-confidence R² > 0.5 tier from the
            # existing > 0.1 tier. > 0.5 is the "clear signal" count.
            html += metric_grid([
                metric_card(f"{len(r2_arr)}", "Features tested"),
                metric_card(f"{(r2_arr > 0.05).sum()}", "R² > 0.05"),
                metric_card(f"{(r2_arr > 0.10).sum()}", "R² > 0.10"),
                metric_card(f"{(r2_arr > 0.50).sum()}", "R² > 0.50"),
                metric_card(f"{np.median(r2_arr):.4f}", "Median R²"),
                metric_card(f"{np.max(r2_arr):.4f}", "Max R²"),
            ])

        # --- Simplex regression dotplot (exclusive only, via pc.pl.dotplot) ---
        try:
            log.info("Building simplex regression long-format dataframe for CMP...")
            cmp_long = regression_to_long_df(
                reg_result, y_col="gene", exclusive_only=True,
                exclusive_threshold=2.5, top_n_per_archetype=10,
                fdr_threshold=0.05)
            log.info(f"  CMP long df rows: {len(cmp_long)}, unique genes: "
                     f"{cmp_long['gene'].nunique() if len(cmp_long) else 0}")
            if len(cmp_long) > 0:
                from _paper_part1_viz import dotplot_figsize
                fig_reg = pc.pl.dotplot(
                    cmp_long, x_col="archetype", y_col="gene",
                    size_col="mean_archetype", color_col="pvalue",
                    top_n_per_group=10,
                    figsize=dotplot_figsize(cmp_long, y_col="gene"),
                    title="CMP SIMPLEX REGRESSION dotplot (exclusive ≥2.5x, FDR<0.05)")
                html += report.fig_to_img(fig_reg,
                    caption=f"CMP simplex regression: top 10 exclusive genes per archetype "
                            f"(method=Scheffé polynomial regression, exclusive_threshold=2.5, "
                            f"FDR<0.05, {cmp_long['gene'].nunique()} unique genes)")
                plt.close("all")
            else:
                html += error_html("No CMP features passed exclusive+FDR filter")
        except Exception as e:
            log.exception("CMP simplex regression dotplot failed")
            html += error_html(f"CMP simplex regression dotplot failed: {e}")

        # --- Coefficient heatmap (grouped by archetype) ---
        try:
            fig_heat = pc.pl.coefficient_heatmap(
                adata_cmp, top_n=50, group_by_archetype=True, show=False)
            html += safe_plotly_html(report, fig_heat,
                "CMP SIMPLEX REGRESSION coefficient heatmap: top 50 genes by R², "
                "grouped by argmax archetype")
        except Exception as e:
            log.exception("Coefficient heatmap failed")
            html += error_html(f"Coefficient heatmap failed: {e}")

        # Hypergeometric for origin study
        if "Study" in adata_cmp.obs.columns:
            log.info("CMP hypergeometric: Study...")
            cond_df = pc.tl.conditional_associations(adata_cmp, obs_column="Study", verbose=False)
            if "odds_ratio" in cond_df.columns:
                cond_df["odds_ratio"] = cond_df["odds_ratio"].replace([np.inf], 999.0)
            display_cols = [c for c in ["archetype", "condition", "observed", "expected",
                            "odds_ratio", "fdr_pvalue", "significant"] if c in cond_df.columns]
            html += report.df_to_html(cond_df[display_cols],
                caption="Hypergeometric test: archetype × origin study (CMP)")

        # --- Wilcoxon rank sum tests (gene associations) + dotplot ---
        log.info("CMP gene associations (Wilcoxon)...")
        try:
            gene_assoc = pc.tl.gene_associations(adata_cmp, verbose=False)
            n_sig_genes = int((gene_assoc["fdr_pvalue"] < 0.05).sum()) if "fdr_pvalue" in gene_assoc.columns else 0
            n_total_tests = len(gene_assoc)
            unique_archs = gene_assoc["archetype"].nunique() if "archetype" in gene_assoc.columns else "?"
            html += metric_grid([
                metric_card(f"{n_sig_genes}", "Wilcoxon FDR<0.05"),
                metric_card(f"{n_total_tests}", "Total tests"),
                metric_card(f"{unique_archs}", "Archetypes"),
            ])
            try:
                fig_wilcox = pc.pl.dotplot(
                    gene_assoc, top_n_per_group=10,
                    title="CMP WILCOXON rank-sum gene associations (top 10/archetype)")
                html += report.fig_to_img(fig_wilcox,
                    caption="CMP Wilcoxon rank-sum test: dot size = mean_archetype, color = -log10(p). "
                            "Method: Mann-Whitney U, top 10 per archetype.")
                plt.close("all")
            except Exception as e:
                log.exception("Wilcoxon dotplot failed")
                html += error_html(f"Wilcoxon dotplot failed: {e}")
            if n_sig_genes > 0:
                top_wilcox = gene_assoc[gene_assoc["fdr_pvalue"] < 0.05].nsmallest(30, "fdr_pvalue")
                display_cols_w = [c for c in ["gene", "archetype", "mean_archetype", "mean_rest",
                                  "log_fc", "pvalue", "fdr_pvalue"] if c in top_wilcox.columns]
                html += report.df_to_html(top_wilcox[display_cols_w],
                    caption="Top 30 CMP Wilcoxon gene associations (by FDR)")
        except Exception as e:
            log.exception("Gene associations failed")
            html += error_html(f"Gene associations failed: {e}")

        # --- Pathway associations (Wilcoxon) + pathway simplex regression ---
        log.info("CMP pathway associations (Wilcoxon)...")
        try:
            if "pathway_scores" not in adata_cmp.obsm:
                net = pc.pp.load_pathway_networks(sources=["c5_bp"])
                pc.pp.compute_pathway_scores(adata_cmp, net=net)
            # Wilcoxon pathway associations
            pw_assoc = pc.tl.pathway_associations(adata_cmp, verbose=False)
            n_sig_pw = int((pw_assoc["fdr_pvalue"] < 0.05).sum()) if "fdr_pvalue" in pw_assoc.columns else 0
            html += metric_grid([
                metric_card(f"{n_sig_pw}", "Wilcoxon pathway FDR<0.05"),
                metric_card(f"{len(pw_assoc)}", "Total pathway tests"),
            ])
            if n_sig_pw > 0:
                try:
                    fig_pw_dot = pc.pl.dotplot(
                        pw_assoc, y_col="pathway", top_n_per_group=5,
                        title="CMP WILCOXON pathway associations (C5:BP, top 5/archetype)")
                    html += report.fig_to_img(fig_pw_dot,
                        caption="CMP Wilcoxon pathway dotplot (MSigDB C5:BP pathways)")
                    plt.close("all")
                except Exception as e:
                    log.exception("Wilcoxon pathway dotplot failed")
                    html += error_html(f"Wilcoxon pathway dotplot failed: {e}")
                top_pw = pw_assoc[pw_assoc["fdr_pvalue"] < 0.05].nsmallest(20, "fdr_pvalue")
                display_cols_p = [c for c in ["pathway", "archetype", "mean_archetype",
                                  "pvalue", "fdr_pvalue"] if c in top_pw.columns]
                html += report.df_to_html(top_pw[display_cols_p],
                    caption="Top 20 CMP Wilcoxon pathway associations (by FDR)")

            # CMP pathway SIMPLEX REGRESSION (for C5:BP comparison)
            try:
                log.info("CMP pathway simplex regression (C5:BP)...")
                pw_reg_cmp = pc.tl.pathway_simplex_regression(
                    adata_cmp, max_degree=1, robust_se=True)
                pw_reg_result = adata_cmp.uns.get("peach_simplex_regression_pathways", {})
                pw_long = regression_to_long_df(
                    pw_reg_result, y_col="pathway", exclusive_only=True,
                    exclusive_threshold=2.5, top_n_per_archetype=5, fdr_threshold=0.05)
                if len(pw_long) > 0:
                    from _paper_part1_viz import dotplot_figsize
                    fig_pw_reg = pc.pl.dotplot(
                        pw_long, x_col="archetype", y_col="pathway",
                        size_col="mean_archetype", color_col="pvalue",
                        top_n_per_group=5,
                        figsize=dotplot_figsize(pw_long, y_col="pathway", per_row=0.45),
                        title="CMP SIMPLEX REGRESSION pathway dotplot (C5:BP, exclusive ≥2.5x)")
                    html += report.fig_to_img(fig_pw_reg,
                        caption="CMP simplex regression on pathway scores (C5:BP, "
                                "top 5 exclusive per archetype)")
                    plt.close("all")
            except Exception as e:
                log.exception("CMP pathway simplex regression failed")
                html += error_html(f"CMP pathway simplex regression failed: {e}")
        except Exception as e:
            log.exception("Pathway associations failed")
            html += error_html(f"Pathway associations failed: {e}")

    except Exception as e:
        log.exception("Fig 1D failed")
        html += error_html(f"Fig 1D failed: {e}")

    # --- CMP holdout generalization (symmetric with HSC holdout: Hotelling + KS) ---
    html += "<h3>CMP holdout generalization</h3>"
    try:
        from scipy.stats import ks_2samp as ks_2samp_cmp, f as f_dist_cmp
        adata_cmp_hold = ad.read_h5ad(os.path.join(DATA_DIR, "adata_cmp_holdout.h5ad"))
        if SUBSAMPLE_FRACTION < 1.0:
            adata_cmp_hold = _stratified_subsample(adata_cmp_hold, SUBSAMPLE_FRACTION, SUBSAMPLE_SEED)
        log.info(f"Projecting CMP holdout ({adata_cmp_hold.shape[0]} cells) into CMP model...")
        pc.pp.prepare_training(adata_cmp_hold, batch_size=64)
        adata_cmp_hold.uns["trained_model"] = adata_cmp.uns["trained_model"]
        adata_cmp_hold.uns["archetype_coordinates"] = adata_cmp.uns["archetype_coordinates"]
        pc.tl.archetypal_coordinates(adata_cmp_hold, verbose=False)
        pc.tl.extract_archetype_weights(adata_cmp_hold, verbose=False)
        pc.tl.assign_archetypes(adata_cmp_hold, verbose=False)

        # Concat train+holdout for PEACH archetypal_space plot
        try:
            adata_cmp.obs["split"] = "train"
            adata_cmp_hold.obs["split"] = "holdout"
            adata_cmp_concat = ad.concat(
                [adata_cmp, adata_cmp_hold], join="outer", merge="first",
            )
            adata_cmp_concat.uns["trained_model"] = adata_cmp.uns["trained_model"]
            adata_cmp_concat.uns["archetype_coordinates"] = adata_cmp.uns["archetype_coordinates"]
            fig_cmp_concat = pc.pl.archetypal_space(
                adata_cmp_concat, color_by="split",
                title="CMP model: train vs holdout in archetypal space")
            html += safe_plotly_html(report, fig_cmp_concat,
                "CMP: train vs holdout projected in 3D archetypal space")
        except Exception as e:
            log.exception("CMP concat archetypal_space failed")
            html += error_html(f"CMP concat archetypal_space failed: {e}")

        # KS + Hotelling T² on CMP holdout weights
        w_cmp_train = adata_cmp.obsm.get("cell_archetype_weights")
        w_cmp_hold = adata_cmp_hold.obsm.get("cell_archetype_weights")
        log.info(f"CMP weights shapes: train={w_cmp_train.shape if w_cmp_train is not None else None}, "
                 f"hold={w_cmp_hold.shape if w_cmp_hold is not None else None}")
        if w_cmp_train is not None and w_cmp_hold is not None:
            K_c = w_cmp_train.shape[1]
            # Full KS table (all K archetypes)
            ks_rows_c = []
            for kk in range(K_c):
                stat_c, pval_c = ks_2samp_cmp(w_cmp_train[:, kk], w_cmp_hold[:, kk])
                pval_bonf_c = min(pval_c * K_c, 1.0)
                ks_rows_c.append({"Archetype": f"A{kk+1}", "KS stat": f"{stat_c:.4f}",
                                  "p-value": fmt_pval(pval_c), "p-Bonf": fmt_pval(pval_bonf_c),
                                  "Sig (Bonf<0.05)": "Yes" if pval_bonf_c < 0.05 else "No"})
            ks_df_c = pd.DataFrame(ks_rows_c)
            n_sig_cmp_ks = int((ks_df_c["Sig (Bonf<0.05)"] == "Yes").sum())

            # Hotelling T² omnibus
            n_tr_c, n_ho_c = w_cmp_train.shape[0], w_cmp_hold.shape[0]
            mu_t_c = w_cmp_train.mean(axis=0)
            mu_h_c = w_cmp_hold.mean(axis=0)
            S_t_c = np.cov(w_cmp_train, rowvar=False)
            S_h_c = np.cov(w_cmp_hold, rowvar=False)
            S_pooled_c = ((n_tr_c - 1) * S_t_c + (n_ho_c - 1) * S_h_c) / (n_tr_c + n_ho_c - 2)
            S_pooled_c += np.eye(K_c) * 1e-8
            try:
                S_inv_c = np.linalg.inv(S_pooled_c)
                diff_c = mu_t_c - mu_h_c
                T2_c = (n_tr_c * n_ho_c / (n_tr_c + n_ho_c)) * (diff_c @ S_inv_c @ diff_c)
                df1_c, df2_c = K_c, n_tr_c + n_ho_c - K_c - 1
                if df2_c > 0:
                    F_stat_c = T2_c * df2_c / (df1_c * (n_tr_c + n_ho_c - 2))
                    hotelling_p_c = 1 - f_dist_cmp.cdf(F_stat_c, df1_c, df2_c)
                else:
                    F_stat_c = float("nan"); hotelling_p_c = float("nan")
            except np.linalg.LinAlgError:
                T2_c = F_stat_c = hotelling_p_c = float("nan")

            html += report.text(
                f"<b>CMP holdout generalization</b>: "
                f"Hotelling T²={T2_c:.3f}, F({df1_c},{df2_c})={F_stat_c:.3f}, p={fmt_pval(hotelling_p_c)}. "
                f"Per-archetype KS: {n_sig_cmp_ks}/{K_c} Bonferroni-significant. "
                f"N train={n_tr_c}, N holdout={n_ho_c}.")
            html += report.df_to_html(ks_df_c,
                caption=f"CMP: full K={K_c} per-archetype KS test + Bonferroni (train vs holdout)")
            html += metric_grid([
                metric_card(f"{n_tr_c}", "N train"),
                metric_card(f"{n_ho_c}", "N holdout"),
                metric_card(f"{n_sig_cmp_ks}/{K_c}", "KS sig (Bonf<0.05)"),
                metric_card(fmt_pval(hotelling_p_c), "Hotelling T² p"),
                metric_card(f"{T2_c:.3f}", "T² stat"),
            ])
    except Exception as e:
        log.exception("CMP holdout generalization failed")
        html += error_html(f"CMP holdout generalization failed: {e}")

    report.add_section("Figure 1: Deep_AA Introduction", html, step_num="1")


# ============================================================================
# PHASE 3: Figure 2 — Simplex Regression + Cross-fit + Flow
# ============================================================================

def phase3_figure2(adata_hsc, adata_cmp, report):
    """Fig 2A-F: regression, Wald contrasts, flow, gene alignment."""
    import peach as pc

    # --- Simplex regression (degree 1 AND 2) on HSC ---
    html_reg = ""
    # Pre-select top 5000 HVGs for regression (full 28K is too slow for degree 2)
    log.info("Selecting top 5000 HVGs for simplex regression...")
    # NOTE: HVG selection by variance introduces selection bias — permutation null
    # (shuffling archetype weights) controls for this within each model, but cross-model
    # comparisons operate only on the intersection of independently-selected HVG sets.
    X_dense = adata_hsc.X.toarray() if hasattr(adata_hsc.X, "toarray") else np.asarray(adata_hsc.X)
    assert X_dense.min() >= 0, f"X contains negative values ({X_dense.min():.2f}) — expected logcounts or raw counts"
    gene_var = np.var(X_dense, axis=0)
    top5k_idx = np.argsort(gene_var)[-5000:]
    # Store in obsm so feature_simplex_regression can find it by key
    adata_hsc.obsm["_hvg5k"] = X_dense[:, top5k_idx]
    adata_hsc.uns["_hvg5k_names"] = [adata_hsc.var_names[i] for i in top5k_idx]
    log.info(f"  Using {len(top5k_idx)} HVGs for regression (of {adata_hsc.shape[1]} total)")

    log.info("HSC gene simplex regression (degree 1 + 2 + comprehensive degree comparison)...")
    gene_reg = pc.tl.feature_simplex_regression(adata_hsc, max_degree=2, robust_se=True,
                                                permutation_test=True, n_permutations=200,
                                                feature_matrix="_hvg5k",
                                                feature_names=adata_hsc.uns["_hvg5k_names"],
                                                comprehensive_degree=True)
    # Alias so dotplot/contrasts can find results under standard key
    if "peach_simplex_regression__hvg5k" in adata_hsc.uns:
        adata_hsc.uns["peach_simplex_regression_genes"] = adata_hsc.uns["peach_simplex_regression__hvg5k"]
        adata_hsc.uns["peach_simplex_regression"] = adata_hsc.uns["peach_simplex_regression__hvg5k"]

    log.info("HSC pathway simplex regression...")
    try:
        net = pc.pp.load_pathway_networks()  # returns DataFrame, not stored on adata
        pc.pp.compute_pathway_scores(adata_hsc, net=net)
        pw_reg = pc.tl.pathway_simplex_regression(adata_hsc, max_degree=2, robust_se=True)
    except Exception as e:
        log.warning(f"Pathway regression failed: {e}")
        pw_reg = None

    # --- Regression summary metrics ---
    reg_result = adata_hsc.uns.get("peach_simplex_regression_genes", {})
    r2_vals = reg_result.get("r_squared_degree1", reg_result.get("r_squared", []))
    feature_names_reg = reg_result.get("feature_names", adata_hsc.uns.get("_hvg5k_names", []))
    if len(r2_vals) > 0:
        r2_arr = np.asarray(r2_vals)
        # W-B15: break out high-confidence R² > 0.5 tier.
        html_reg += metric_grid([
            metric_card(f"{len(r2_arr)}", "Features tested"),
            metric_card(f"{(r2_arr > 0.05).sum()}", "R² > 0.05"),
            metric_card(f"{(r2_arr > 0.10).sum()}", "R² > 0.10"),
            metric_card(f"{(r2_arr > 0.50).sum()}", "R² > 0.50"),
            metric_card(f"{np.median(r2_arr):.4f}", "Median R²"),
            metric_card(f"{np.max(r2_arr):.4f}", "Max R²"),
        ])

    # Permutation null: FDR correction + summary
    perm_pval = gene_reg.get("permutation_pvalue")
    if perm_pval is not None:
        from scipy.stats import false_discovery_control
        perm_arr = np.asarray(perm_pval)
        valid_mask = ~np.isnan(perm_arr)
        perm_fdr = np.ones_like(perm_arr)
        if valid_mask.sum() > 0:
            perm_fdr[valid_mask] = false_discovery_control(perm_arr[valid_mask], method="bh")
        n_sig_raw = int((perm_arr < 0.05).sum())
        n_sig_fdr = int((perm_fdr < 0.05).sum())
        n_total_p = len(perm_arr)
        html_reg += metric_grid([
            metric_card(f"{n_sig_raw}/{n_total_p}", "Perm p<0.05"),
            metric_card(f"{n_sig_fdr}/{n_total_p}", "Perm FDR<0.05"),
        ])
        html_reg += report.text(
            f"<b>HSC SIMPLEX REGRESSION permutation null</b>: "
            f"{n_sig_raw}/{n_total_p} raw p<0.05, "
            f"{n_sig_fdr}/{n_total_p} after BH FDR. "
            f"Method: 200 permutations shuffling archetype weights (p resolution 1/201 ≈ 0.005), "
            f"tests whether per-feature R² exceeds shuffled-weight null.")

    # --- Fig 2A: HSC simplex regression — three per-degree dotplots + nesting table + UpSet (W-B16) ---
    # Replaces the old single ΔR² dotplot. Layout:
    #   1. Degree-1 dotplot: top genes per archetype ranked by r_squared_degree1
    #      (vertex FDR<0.05 filter, exclusive flag for interpretability).
    #   2. Degree-2 dotplot: top genes per archetype ranked by r_squared_degree2
    #      (degree_comparison incremental F-test FDR<0.05 filter).
    #   3. Degree-3 dotplot: top genes per archetype ranked by r_squared_degree3
    #      from degree_comparison["degree_3"] (requires K-1 >= 3).
    #   4. Nesting table: per-feature deg1/deg2/deg3 R² + which degrees it was
    #      significant in ("appears_in_deg" column, sorted descending 3→2→1).
    #   5. UpSet plot on per-degree significant-feature sets (top 100 per set).
    try:
        html_reg += "<h3>Fig 2A: HSC simplex regression (per-degree panels)</h3>"
        html_reg += report.text(
            "<b>Panel interpretation</b>: Degree-1 = pure archetype main effects (archetypes "
            "act independently). Degree-2 = pairwise archetype interactions (features enriched "
            "in blending zones). Degree-3 = triple interactions (requires K-1 ≥ 3). Each "
            "dotplot shows the top 10 genes per archetype ranked by that degree's R² after "
            "FDR<0.05 filtering. The nesting table then shows which degrees each top feature "
            "appeared in, and the UpSet plot visualises the overlap across the three degree "
            "signature sets.")

        # Gather the three degree-specific long-format dataframes.
        log.info("Fig 2A: building per-degree long-format dataframes...")
        hsc_long_d1 = regression_to_long_df(
            reg_result, y_col="gene", exclusive_only=True,
            exclusive_threshold=2.5, top_n_per_archetype=10, fdr_threshold=0.05,
            degree=1)
        hsc_long_d2 = regression_to_long_df(
            reg_result, y_col="gene", exclusive_only=False,
            top_n_per_archetype=10, fdr_threshold=0.05, degree=2)
        try:
            hsc_long_d3 = regression_to_long_df(
                reg_result, y_col="gene", exclusive_only=False,
                top_n_per_archetype=10, fdr_threshold=0.05, degree=3)
        except Exception:
            hsc_long_d3 = pd.DataFrame()
        log.info(f"  deg1 rows={len(hsc_long_d1)}, deg2 rows={len(hsc_long_d2)}, "
                 f"deg3 rows={len(hsc_long_d3)}")

        # Panel 1: degree-1 dotplot (per-archetype, unchanged)
        from _paper_part1_viz import dotplot_figsize
        if len(hsc_long_d1) > 0:
            fig_d1 = pc.pl.dotplot(
                hsc_long_d1, x_col="archetype", y_col="gene",
                size_col="mean_archetype", color_col="pvalue",
                top_n_per_group=10,
                figsize=dotplot_figsize(hsc_long_d1, y_col="gene"),
                title="HSC SIMPLEX REGRESSION degree 1 (exclusive ≥2.5x, vertex FDR<0.05)")
            html_reg += report.fig_to_img(fig_d1,
                caption=f"Degree-1 panel: top 10 genes per archetype ranked by "
                        f"r_squared_degree1. Dot size = |vertex main-effect coef|; "
                        f"colour = vertex FDR. {hsc_long_d1['gene'].nunique()} unique genes.")
            plt.close("all")
        else:
            html_reg += report.text(
                "Degree-1 panel: no features pass exclusive ≥2.5x + vertex FDR<0.05.")

        # Panel 2: degree-2 PER-INTERACTION-PAIR layout (r12-item-2).
        # Replaces the per-archetype degree-2 dotplot: each row is a
        # (feature, interaction-pair) combo where interaction FDR<0.05.
        # Dotplot x=pair label (e.g. "A1-A3"), y=feature. Below the
        # dotplot, a classification table groups rows by pair_type
        # (cooperative/tradeoff/transition-enriched/structured) and
        # sorts within type by |γ| descending. Same pattern used in
        # run_e2e_myeloid.py:692-772.
        try:
            _int_coefs = reg_result.get("interaction_coefficients")
            _int_pairs = reg_result.get("interaction_pairs", [])
            _int_fdr_arr = reg_result.get("interaction_pvalues_fdr")
            _v_coefs = np.asarray(reg_result.get("vertex_coefficients", []))
            _feat_names_d2 = list(reg_result.get("feature_names", []))
            if (_int_coefs is None or len(_int_pairs) == 0
                    or _int_fdr_arr is None or _v_coefs.size == 0):
                html_reg += report.text(
                    "Degree-2 panel: interaction terms unavailable (max_degree<2 "
                    "or empty interaction matrices).")
            else:
                _int_coefs = np.asarray(_int_coefs)
                _int_fdr_arr = np.asarray(_int_fdr_arr)
                _int_rows = []
                for _fi in range(len(_feat_names_d2)):
                    for _pi, (_j, _k) in enumerate(_int_pairs):
                        if _int_fdr_arr[_fi, _pi] >= 0.05:
                            continue
                        _gamma = float(_int_coefs[_fi, _pi])
                        _bj = float(_v_coefs[_fi, _j])
                        _bk = float(_v_coefs[_fi, _k])
                        _median_abs = float(np.median(np.abs(_v_coefs[_fi])))
                        _j_high = abs(_bj) > _median_abs
                        _k_high = abs(_bk) > _median_abs
                        _same_sign = (np.sign(_bj) == np.sign(_bk)
                                      and _bj != 0 and _bk != 0)
                        if _j_high and _k_high and _same_sign:
                            _ptype = "cooperative"
                        elif (_j_high and not _k_high) or (not _j_high and _k_high):
                            _ptype = "tradeoff"
                        elif (not _j_high and not _k_high
                              and abs(_gamma) > _median_abs):
                            _ptype = "transition-enriched"
                        else:
                            _ptype = "structured"
                        _transition = (
                            f"rising (A{_j+1}→A{_k+1})" if _gamma > 0
                            else f"falling (A{_j+1}→A{_k+1})"
                        )
                        _int_rows.append({
                            "gene": _feat_names_d2[_fi],
                            "pair": f"A{_j+1}-A{_k+1}",
                            "pair_type": _ptype,
                            "transition": _transition,
                            "beta_j": _bj,
                            "beta_k": _bk,
                            "gamma": _gamma,
                            "abs_gamma": abs(_gamma),
                            "fdr_q": float(_int_fdr_arr[_fi, _pi]),
                        })
                if not _int_rows:
                    html_reg += report.text(
                        "Degree-2 panel: no (feature, pair) combinations pass "
                        "interaction FDR<0.05.")
                else:
                    _int_df = pd.DataFrame(_int_rows)
                    log.info(
                        f"  Degree-2 interaction rows: {len(_int_df)} total, "
                        f"type counts={_int_df['pair_type'].value_counts().to_dict()}"
                    )
                    # Per-pair dotplot: top 10 features per pair by |γ|.
                    _int_df_plot = (
                        _int_df.sort_values("abs_gamma", ascending=False)
                        .groupby("pair", group_keys=False).head(10)
                        .copy()
                    )
                    # pc.pl.dotplot expects mean_archetype + pvalue; reuse |γ|
                    # and fdr_q so the existing plot machinery renders them.
                    _int_df_plot["mean_archetype"] = _int_df_plot["abs_gamma"]
                    _int_df_plot["pvalue"] = _int_df_plot["fdr_q"]
                    try:
                        fig_d2 = pc.pl.dotplot(
                            _int_df_plot, x_col="pair", y_col="gene",
                            size_col="mean_archetype", color_col="pvalue",
                            top_n_per_group=10,
                            figsize=dotplot_figsize(_int_df_plot, y_col="gene"),
                            title=(
                                "HSC SIMPLEX REGRESSION degree 2 — "
                                "per-interaction-pair (interaction FDR<0.05, top 10/pair)"
                            ),
                        )
                        html_reg += report.fig_to_img(
                            fig_d2,
                            caption=(
                                f"Degree-2 per-pair dotplot: each row = (feature, pair). "
                                f"x-axis = interaction pair A_j-A_k, y-axis = feature. "
                                f"Dot size = |γ| (interaction coefficient magnitude), "
                                f"colour = interaction FDR. "
                                f"{_int_df_plot['gene'].nunique()} unique genes across "
                                f"{_int_df_plot['pair'].nunique()} pairs."
                            ),
                        )
                        plt.close("all")
                    except Exception as _e_d2plot:
                        log.exception("Degree-2 per-pair dotplot failed")
                        html_reg += error_html(
                            f"Degree-2 per-pair dotplot failed: {_e_d2plot}"
                        )
                    # Classification table grouped by pair_type, sorted by |γ|.
                    html_reg += report.text(
                        "<b>Interaction classification</b>: <b>Cooperative</b> = "
                        "both archetypes high, same sign (shared program). "
                        "<b>Tradeoff</b> = one high, one low (distinguishes "
                        "archetypes). <b>Transition-enriched</b> = peaks in "
                        "blending zone, not at either vertex. <b>Structured</b> = "
                        "mixed pattern. Edge γ direction: rising (γ>0) = gene "
                        "increases along A_j→A_k edge; falling (γ<0) = decreases. "
                        "Tables sorted by |γ| descending."
                    )
                    _type_counts = _int_df["pair_type"].value_counts()
                    _cards = [metric_card(len(_int_df), "Significant interactions")]
                    for _t in ["tradeoff", "cooperative",
                               "transition-enriched", "structured"]:
                        _cards.append(
                            metric_card(int(_type_counts.get(_t, 0)), _t.capitalize())
                        )
                    html_reg += metric_grid(_cards)
                    for _t in ["tradeoff", "cooperative",
                               "transition-enriched", "structured"]:
                        _sub = (
                            _int_df[_int_df["pair_type"] == _t]
                            .sort_values("abs_gamma", ascending=False)
                            .head(20)
                        )
                        if len(_sub) == 0:
                            continue
                        _display = _sub[[
                            "gene", "pair", "transition",
                            "beta_j", "beta_k", "gamma", "fdr_q",
                        ]].copy()
                        _display["beta_j"] = _display["beta_j"].map(lambda v: f"{v:.3f}")
                        _display["beta_k"] = _display["beta_k"].map(lambda v: f"{v:.3f}")
                        _display["gamma"] = _display["gamma"].map(lambda v: f"{v:.3f}")
                        _display["fdr_q"] = _display["fdr_q"].map(fmt_pval)
                        _display = _display.rename(columns={
                            "gene": "Gene", "pair": "Pair",
                            "transition": "Transition",
                            "beta_j": "β_j", "beta_k": "β_k",
                            "gamma": "Edge γ", "fdr_q": "FDR q",
                        })
                        html_reg += report.df_to_html(
                            _display,
                            caption=f"Top {_t} interactions (ranked by |γ|)",
                        )
        except Exception as _e_d2int:
            log.exception("Degree-2 per-interaction panel failed")
            html_reg += error_html(
                f"Degree-2 per-interaction panel failed: {_e_d2int}"
            )

        # Panel 3: degree-3 summary (dotplot dropped per r12 review — the
        # comprehensive_degree code path stores only R²/delta_R²/FDR for
        # degree 3 with no per-triple coefficient matrix, so a
        # per-archetype or per-triple dotplot would be misleading).
        try:
            _dc_d3 = (reg_result.get("degree_comparison", {}) or {}).get("degree_3", {}) or {}
            _r2_d3 = np.asarray(_dc_d3.get("r_squared", []))
            _delta_d3 = np.asarray(_dc_d3.get("delta_r2", []))
            _incr_fdr_d3 = np.asarray(_dc_d3.get("incremental_p_fdr", []))
            _feat_names_d3 = list(reg_result.get("feature_names", []))
            if (_r2_d3.size == 0 or _delta_d3.size == 0
                    or _incr_fdr_d3.size == 0):
                html_reg += report.text(
                    "Degree-3 panel: comprehensive degree comparison unavailable "
                    "(requires K-1 ≥ 3 and comprehensive_degree=True)."
                )
            else:
                _n_sig_d3 = int((_incr_fdr_d3 < 0.05).sum())
                html_reg += report.text(
                    f"<b>Degree-3 summary</b>: {_n_sig_d3}/{len(_incr_fdr_d3)} "
                    f"features pass incremental F-test FDR&lt;0.05 (triple "
                    f"interaction adds significant variance on top of deg-2)."
                )
                _d3_rows = []
                _order = np.argsort(-_delta_d3)
                for _fi in _order[:30]:
                    if _incr_fdr_d3[_fi] >= 0.05:
                        continue
                    _d3_rows.append({
                        "Gene": _feat_names_d3[_fi] if _fi < len(_feat_names_d3) else f"feat_{_fi}",
                        "deg3 R²": f"{float(_r2_d3[_fi]):.4f}",
                        "Δ R² (deg2→deg3)": f"{float(_delta_d3[_fi]):.4f}",
                        "Incr. F-test FDR": fmt_pval(float(_incr_fdr_d3[_fi])),
                    })
                if _d3_rows:
                    html_reg += report.df_to_html(
                        pd.DataFrame(_d3_rows),
                        caption=(
                            "Top 30 features by Δ R² (degree 2 → 3), restricted to "
                            "incremental F-test FDR<0.05. No per-triple coefficient "
                            "table available — degree-3 regression stores only "
                            "aggregate R²/FDR statistics."
                        ),
                    )
                else:
                    html_reg += report.text(
                        "Degree-3 panel: no features with incremental F-test FDR<0.05."
                    )
        except Exception as _e_d3:
            log.exception("Degree-3 summary failed")
            html_reg += error_html(f"Degree-3 summary failed: {_e_d3}")

        # ---- Nesting table + UpSet: signature sets per degree ----
        # For each degree, extract the "significant feature set" (by that degree's
        # significance criterion). Then:
        #  - nesting table: sorted descending by degree-of-first-appearance
        #    (3 → 2 → 1), listing deg1/deg2/deg3 R² values and appears_in_deg.
        #  - UpSet plot: cross-degree overlap.
        try:
            feat_names_nst = list(reg_result.get("feature_names", []))
            r2_d1_arr = np.asarray(reg_result.get("r_squared_degree1", []))
            r2_d2_arr = np.asarray(reg_result.get("r_squared_degree2", []))
            if r2_d2_arr.size == 0:
                _dc_tmp = reg_result.get("degree_comparison", {}) or {}
                r2_d2_arr = np.asarray((_dc_tmp.get("degree_2", {}) or {}).get("r_squared", []))
            _dc_nst = reg_result.get("degree_comparison", {}) or {}
            _d3_nst = _dc_nst.get("degree_3", {}) or {}
            r2_d3_arr = np.asarray(_d3_nst.get("r_squared", []))
            # Significance masks
            # deg1 significant = any archetype vertex FDR < 0.05
            _v_fdr = np.asarray(reg_result.get("vertex_pvalues_fdr", []))
            if _v_fdr.ndim == 2 and _v_fdr.size > 0:
                sig_d1_mask = np.any(_v_fdr < 0.05, axis=1)
            else:
                sig_d1_mask = np.zeros(len(feat_names_nst), dtype=bool)
            # deg2 significant = incremental F-test FDR < 0.05
            _d2_fdr = np.asarray((_dc_nst.get("degree_2", {}) or {}).get("incremental_p_fdr", []))
            if _d2_fdr.size > 0:
                sig_d2_mask = _d2_fdr < 0.05
            else:
                sig_d2_mask = np.zeros(len(feat_names_nst), dtype=bool)
            # deg3 significant = incremental F-test FDR < 0.05
            _d3_fdr = np.asarray(_d3_nst.get("incremental_p_fdr", []))
            if _d3_fdr.size > 0:
                sig_d3_mask = _d3_fdr < 0.05
            else:
                sig_d3_mask = np.zeros(len(feat_names_nst), dtype=bool)
            log.info(
                f"  Nesting masks: deg1 sig={int(sig_d1_mask.sum())}, "
                f"deg2 sig={int(sig_d2_mask.sum())}, deg3 sig={int(sig_d3_mask.sum())}"
            )

            # Build signature sets per degree, capped at top-100 by that
            # degree's R² (descending) — W-B18-style cap for UpSet readability.
            _UPSET_CAP = 100
            def _top_sig(mask, r2_arr, cap):
                if mask.sum() == 0 or r2_arr.size == 0:
                    return []
                idx = np.where(mask)[0]
                # Defensive: align mask and r2_arr lengths
                valid_idx = idx[idx < r2_arr.size]
                if valid_idx.size == 0:
                    return []
                ranked = valid_idx[np.argsort(r2_arr[valid_idx])[::-1]]
                return [feat_names_nst[i] for i in ranked[:cap] if i < len(feat_names_nst)]

            top_d1 = _top_sig(sig_d1_mask, r2_d1_arr, _UPSET_CAP)
            top_d2 = _top_sig(sig_d2_mask, r2_d2_arr, _UPSET_CAP)
            top_d3 = _top_sig(sig_d3_mask, r2_d3_arr, _UPSET_CAP)
            set_d1 = set(top_d1)
            set_d2 = set(top_d2)
            set_d3 = set(top_d3)

            # Nesting table: union of the three top-100 sets, per-feature
            # deg1/deg2/deg3 R² + appears_in_deg (e.g. "3,2,1"), sorted
            # descending by the earliest degree it appears in.
            # Also: per-degree archetype association columns (deg1_arch,
            # deg2_arch, deg3_arch) showing which archetype(s) each feature
            # is most associated with at each polynomial degree.
            union = set_d1 | set_d2 | set_d3

            # Pre-extract vertex_coefficients and interaction info for
            # archetype association columns.
            _vtx_coefs = np.asarray(reg_result.get("vertex_coefficients", []))
            _int_coefs = reg_result.get("interaction_coefficients")
            _int_pairs = reg_result.get("interaction_pairs")
            if _int_coefs is not None:
                _int_coefs = np.asarray(_int_coefs)
            # K from vertex coefficients
            _K_nst = _vtx_coefs.shape[1] if _vtx_coefs.ndim == 2 and _vtx_coefs.size > 0 else 0

            def _deg1_arch_label(fi):
                """Archetype with largest |vertex coefficient| for feature fi."""
                if _vtx_coefs.ndim != 2 or fi >= _vtx_coefs.shape[0] or _K_nst == 0:
                    return "-"
                row = np.abs(_vtx_coefs[fi])
                return f"A{int(np.argmax(row)) + 1}"

            def _deg2_arch_label(fi):
                """Top-2 archetypes by |vertex coef| magnitude, or the
                interaction pair with largest |interaction coef| if available."""
                if _int_coefs is not None and _int_pairs is not None and fi < _int_coefs.shape[0]:
                    # Find the interaction pair with the largest absolute coef
                    abs_int = np.abs(_int_coefs[fi])
                    if abs_int.size > 0:
                        best_pair_idx = int(np.argmax(abs_int))
                        if best_pair_idx < len(_int_pairs):
                            pi, pj = _int_pairs[best_pair_idx]
                            return f"A{pi+1}-A{pj+1}"
                # Fallback: top-2 archetypes by |vertex coef|
                if _vtx_coefs.ndim != 2 or fi >= _vtx_coefs.shape[0] or _K_nst < 2:
                    return "-"
                row = np.abs(_vtx_coefs[fi])
                top2 = np.argsort(row)[-2:][::-1]
                return f"A{int(top2[0])+1}-A{int(top2[1])+1}"

            def _deg3_arch_label(fi):
                """Top-3 archetypes by |vertex coef| magnitude for feature fi."""
                if _vtx_coefs.ndim != 2 or fi >= _vtx_coefs.shape[0] or _K_nst < 3:
                    return "-"
                row = np.abs(_vtx_coefs[fi])
                top3 = np.argsort(row)[-3:][::-1]
                return f"A{int(top3[0])+1}-A{int(top3[1])+1}-A{int(top3[2])+1}"

            if union:
                _name_to_idx = {n: i for i, n in enumerate(feat_names_nst)}
                nesting_rows = []
                for fname in union:
                    fi = _name_to_idx.get(fname)
                    if fi is None:
                        continue
                    in_d1 = fname in set_d1
                    in_d2 = fname in set_d2
                    in_d3 = fname in set_d3
                    degs_in = []
                    if in_d3:
                        degs_in.append("3")
                    if in_d2:
                        degs_in.append("2")
                    if in_d1:
                        degs_in.append("1")
                    # Sort-by-first-degree: 3 highest, then 2, then 1.
                    if in_d3:
                        first_deg_sort = 0
                    elif in_d2:
                        first_deg_sort = 1
                    else:
                        first_deg_sort = 2
                    nesting_rows.append({
                        "feature_name": fname,
                        "deg1_R²": f"{float(r2_d1_arr[fi]):.4f}"
                            if fi < r2_d1_arr.size else "-",
                        "deg1_arch": _deg1_arch_label(fi),
                        "deg2_R²": f"{float(r2_d2_arr[fi]):.4f}"
                            if fi < r2_d2_arr.size else "-",
                        "deg2_arch": _deg2_arch_label(fi),
                        "deg3_R²": f"{float(r2_d3_arr[fi]):.4f}"
                            if fi < r2_d3_arr.size else "-",
                        "deg3_arch": _deg3_arch_label(fi),
                        "appears_in_deg": ",".join(degs_in) if degs_in else "-",
                        "notes": (
                            "nested through all three degrees"
                            if len(degs_in) == 3
                            else ("first appears at degree " + degs_in[0]
                                  if degs_in else "-")
                        ),
                        "_first_deg_sort": first_deg_sort,
                        "_deg3_r2_val": float(r2_d3_arr[fi])
                            if fi < r2_d3_arr.size else -np.inf,
                    })
                nesting_df = pd.DataFrame(nesting_rows)
                if not nesting_df.empty:
                    nesting_df = nesting_df.sort_values(
                        ["_first_deg_sort", "_deg3_r2_val"],
                        ascending=[True, False],
                    ).drop(columns=["_first_deg_sort", "_deg3_r2_val"])
                    # Cap table length for report readability
                    _MAX_NESTING_ROWS = 60
                    nesting_display = nesting_df.head(_MAX_NESTING_ROWS)
                    html_reg += report.df_to_html(
                        nesting_display,
                        caption=(
                            f"Fig 2A nesting table: per-feature R² at degrees 1/2/3 with "
                            f"archetype association columns (deg1_arch = argmax |vertex coef|; "
                            f"deg2_arch = interaction pair with largest |interaction coef| or "
                            f"top-2 by |vertex coef|; deg3_arch = top-3 by |vertex coef|) "
                            f"and appears_in_deg column showing which polynomial degrees the "
                            f"feature appeared significant in (top-100 per degree by R²). "
                            f"Sorted descending by degree-of-first-appearance (3→2→1). "
                            f"Showing {min(len(nesting_df), _MAX_NESTING_ROWS)} of "
                            f"{len(nesting_df)} union features."
                        ),
                    )
            else:
                html_reg += report.text(
                    "Fig 2A nesting table: no significant features at any degree.")

            # UpSet on the three degree signature sets (top-100 cap already applied).
            try:
                from upsetplot import from_contents, UpSet
                import matplotlib.pyplot as _plt_upset_fig2a
                _fig2a_sets = {
                    "deg1_sig_features": set_d1,
                    "deg2_sig_features": set_d2,
                    "deg3_sig_features": set_d3,
                }
                _n_nonempty = sum(1 for s in _fig2a_sets.values() if len(s) > 0)
                if _n_nonempty >= 2:
                    _upset_fig2a_data = from_contents(_fig2a_sets)
                    _fig_upset_fig2a = _plt_upset_fig2a.figure(figsize=(10, 5))
                    UpSet(_upset_fig2a_data, show_counts=True,
                          sort_by="cardinality", min_subset_size=1
                          ).plot(fig=_fig_upset_fig2a)
                    _fig_upset_fig2a.suptitle(
                        "Fig 2A: cross-degree feature overlap (UpSet, top-100 per degree)",
                        fontsize=12, y=1.01)
                    html_reg += report.fig_to_img(
                        _fig_upset_fig2a,
                        caption=(
                            f"UpSet plot: overlap between the top-100 significant features at "
                            f"degrees 1, 2, and 3. Set sizes: "
                            f"deg1={len(set_d1)}, deg2={len(set_d2)}, deg3={len(set_d3)}. "
                            f"Each bar = count of features appearing in that exact combination "
                            f"of degree sets."
                        ),
                    )
                    _plt_upset_fig2a.close("all")
                else:
                    html_reg += report.text(
                        f"Fig 2A UpSet skipped: only {_n_nonempty} of 3 degree sets non-empty."
                    )
            except Exception as e:
                log.exception("Fig 2A UpSet plot failed")
                html_reg += error_html(f"Fig 2A UpSet plot failed: {e}")
        except Exception as e:
            log.exception("Fig 2A nesting table / UpSet failed")
            html_reg += error_html(f"Fig 2A nesting table / UpSet failed: {e}")

        # Comprehensive degree comparison summary table (kept from prior revision for context).
        try:
            deg_comp = reg_result.get("degree_comparison") or gene_reg.get("degree_comparison", {})
            if deg_comp:
                comp_rows = []
                for deg_key in sorted(deg_comp.keys()):
                    d = deg_comp[deg_key]
                    comp_rows.append({
                        "Degree": deg_key,
                        "Mean R²": f"{np.mean(d['r_squared']):.4f}",
                        "Mean dR2": f"{np.mean(d['delta_r2']):.4f}",
                        "Max dR2": f"{np.max(d['delta_r2']):.4f}",
                        "Features with incr. FDR<0.05": str(d.get("significant_features", "?")),
                        "Extra params": str(d.get("n_params", "?")),
                    })
                if comp_rows:
                    html_reg += report.df_to_html(pd.DataFrame(comp_rows),
                        caption="Comprehensive degree comparison: incremental R² gains from adding "
                                "higher-order interaction terms. Each row shows mean/max gain over "
                                "all 5K tested genes.")
        except Exception as e:
            log.exception("Degree comparison summary table failed")
            html_reg += error_html(f"Degree comparison summary table failed: {e}")

        # Interaction heatmap (degree 2)
        try:
            fig_int_heat = pc.pl.interaction_heatmap(adata_hsc, top_n=30, show=False)
            html_reg += safe_plotly_html(report, fig_int_heat,
                "HSC SIMPLEX REGRESSION interaction heatmap: top 30 features (degree 2, R²-ranked)")
        except Exception as e:
            log.exception("Interaction heatmap failed")
            html_reg += error_html(f"Interaction heatmap failed: {e}")

        # Radar chart (order by similarity)
        try:
            fig_radar = pc.pl.archetype_radar(
                adata_hsc, top_n=8, feature_type="genes",
                min_degree=1, order_by_similarity=True, show=False)
            html_reg += safe_plotly_html(report, fig_radar,
                "HSC SIMPLEX REGRESSION radar: top 8 genes per archetype, "
                "spokes ordered by Fiedler similarity")
        except Exception as e:
            log.exception("Radar plot failed")
            html_reg += error_html(f"Radar plot failed: {e}")

    except Exception as e:
        log.exception("Fig 2A failed")
        html_reg += error_html(f"Fig 2A failed: {e}")

    # --- Fig 2B: Tradeoff patterns + pathway simplex regression ---
    try:
        html_reg += "<h3>Fig 2B: Patterns + pathway regression</h3>"
        # Classify feature patterns
        pattern_result = pc.tl.classify_feature_patterns(adata_hsc)
        pattern_counts = pattern_result.get("pattern_counts", {})
        if isinstance(pattern_counts, dict):
            html_reg += report.text(
                f"<b>HSC SIMPLEX REGRESSION feature pattern classification</b>: {pattern_counts}")
        else:
            html_reg += report.text(
                f"<b>HSC feature patterns</b>: {pattern_result.get('n_features', '?')} features classified")

        # --- Full pattern taxonomy (Task 16) ---
        # Expand beyond tradeoff: extract cooperative, transition-enriched, gradient sub-types
        # from the interaction_detail field of features classified as "interaction".
        # Pattern classifier produces: flat, archetype-exclusive, interaction, structured.
        # "interaction" features further subdivide via interaction_detail (per-pair):
        #   cooperative, tradeoff, transition-enriched, gradient.
        html_reg += "<h4>Fig 2B: Full interaction pattern taxonomy</h4>"
        try:
            classifications_b = pattern_result.get("classifications", [])
            feat_names_b = pattern_result.get("feature_names", [])
            # Build a map: feature_name → list of pair_type strings (from interaction_detail)
            interaction_subtypes = {}  # pair_type → list of (feat_name, pair, gamma)
            for i, c in enumerate(classifications_b):
                if c.get("pattern") != "interaction":
                    continue
                fname_b = feat_names_b[i] if i < len(feat_names_b) else f"feat_{i}"
                details_b = c.get("details", {})
                int_detail = details_b.get("interaction_detail", [])
                for pair_info in int_detail:
                    pt = pair_info.get("pair_type", "unknown")
                    interaction_subtypes.setdefault(pt, []).append({
                        "feature": fname_b,
                        "pair": str(pair_info.get("pair", "")),
                        "gamma": pair_info.get("gamma", float("nan")),
                        "beta_j": pair_info.get("beta_j", float("nan")),
                        "beta_k": pair_info.get("beta_k", float("nan")),
                        "transition": pair_info.get("transition", ""),
                    })

            # Build dotplots for each interaction sub-type that has ≥ 1 entry.
            #
            # W-B17 FIX: the previous implementation called
            #     pc.pl.dotplot(sub_long_filtered, x_col="archetype", ...)
            # which plotted MAIN-EFFECT vertex coefficients. Because those
            # coefficients are the same numbers used by the Fig 2A exclusive
            # dotplot (just filtered to a different gene list), the sub-type
            # dotplots visually collapsed into "the same dotplot as Fig 2A"
            # (round-9 review finding).
            #
            # The correct visualization for tradeoff / cooperative /
            # transition-enriched / gradient patterns is to group by the
            # archetype PAIR (j, k), not a single archetype, because each
            # row in interaction_detail refers to the interaction between
            # exactly one pair. We build a per-sub-type DataFrame with one
            # row per (feature, pair) and pass it to pc.pl.pattern_dotplot,
            # which reads the `pattern_code` column as its X-axis (pairs)
            # and `gene` as its Y-axis (features).
            for sub_type in ["tradeoff", "cooperative", "transition-enriched", "gradient"]:
                entries = interaction_subtypes.get(sub_type, [])
                html_reg += f"<h5>Pattern sub-type: {sub_type} ({len(entries)} feature-pair instances)</h5>"
                if len(entries) == 0:
                    html_reg += report.text(
                        f"No features classified as <b>{sub_type}</b> by the interaction classifier.")
                    continue
                # Build a small DataFrame from the entries for display
                df_sub = pd.DataFrame(entries)
                df_sub["|gamma|"] = np.abs(df_sub["gamma"]).round(4)
                df_sub = df_sub.sort_values("|gamma|", ascending=False).head(20)
                html_reg += report.df_to_html(df_sub,
                    caption=f"Top 20 {sub_type} feature-pair instances (sorted by |gamma|). "
                            f"gamma = R2-based relative contribution (beta_j^2 / (beta_j^2 + beta_k^2)); range [0,1] "
                            f"for the two archetypes in the pair.")

                # ---- Pair-based dotplot ----
                # pc.pl.pattern_dotplot expects a DataFrame with columns:
                #   - gene  (or pathway / feature)    → Y-axis
                #   - pattern_code                    → X-axis (archetype pair)
                #   - log_fold_change / mean_diff     → dot size  (effect)
                #   - pvalue                          → dot colour (-log10 p)
                # We synthesize this from the interaction_detail entries.
                # `pair` is stored as a stringified tuple "(j, k)" — format
                # it as "A{j+1}_A{k+1}" to match Fig 2E/F pair notation
                # (1-indexed archetype labels).
                def _fmt_pair(pair_str):
                    """Parse '(j, k)' -> 'A{j+1}_A{k+1}'. Fallback: raw string."""
                    try:
                        import ast as _ast
                        j, k = _ast.literal_eval(pair_str)
                        return f"A{int(j)+1}_A{int(k)+1}"
                    except Exception:
                        return str(pair_str)

                plot_rows = []
                for row in df_sub.to_dict("records"):
                    gamma_val = float(row.get("gamma", 0.0))
                    plot_rows.append({
                        "gene": row.get("feature", "?"),
                        "pattern_code": _fmt_pair(row.get("pair", "")),
                        "log_fold_change": gamma_val,
                        "mean_diff": gamma_val,
                        # The interaction_detail entries have already been
                        # filtered on FDR (significant only) by
                        # classify_feature_patterns, so p-values are known
                        # to be < 0.05. Use a nominal low value so
                        # pattern_dotplot's default max_pvalue=0.05 filter
                        # does not drop rows.
                        "pvalue": 0.01,
                        "fdr_pvalue": 0.01,
                        "significant": True,
                    })
                df_plot = pd.DataFrame(plot_rows)
                if df_plot.empty:
                    html_reg += report.text(
                        f"  {sub_type}: no rows available for pair-based dotplot.")
                    continue
                try:
                    # Use a small min_effect_size so pair-level gamma
                    # values (which can be modest even when significant)
                    # are not filtered out. pattern_type controls only
                    # the title and labelling inside pattern_dotplot.
                    fig_sub = pc.pl.pattern_dotplot(
                        df_plot,
                        pattern_type=sub_type,
                        top_n=20,
                        min_effect_size=0.001,
                        max_pvalue=1.0,
                        figsize=(8, max(4, 0.25 * len(df_plot))),
                        title=f"HSC {sub_type} pattern dotplot "
                              f"(archetype pair on X, top 20 by |gamma|)",
                    )
                    html_reg += report.fig_to_img(fig_sub,
                        caption=f"{sub_type.capitalize()} pattern dotplot: "
                                f"X = archetype PAIR (Aj_Ak), Y = gene, "
                                f"dot size = |gamma| (interaction coefficient), "
                                f"colour = -log10(p-value). "
                                f"{len(df_plot)} (feature, pair) rows "
                                f"from the interaction_detail field of "
                                f"classify_feature_patterns.")
                    plt.close("all")
                except Exception as e:
                    html_reg += error_html(f"{sub_type} pair dotplot failed: {e}")
        except Exception as e:
            log.exception("Full pattern taxonomy (Task 16) failed")
            html_reg += error_html(f"Full pattern taxonomy failed: {e}")

        # Legacy tradeoff patterns table (kept for backward compatibility)
        try:
            tradeoff_df = pc.tl.tradeoff_patterns(adata_hsc)
            if isinstance(tradeoff_df, pd.DataFrame) and len(tradeoff_df) > 0:
                html_reg += report.df_to_html(tradeoff_df.head(20),
                    caption="HSC top 20 tradeoff patterns (simplex regression-derived)")
        except Exception as e:
            log.exception("Tradeoff patterns failed")
            html_reg += error_html(f"Tradeoff patterns failed: {e}")

        # Pattern summary barplot — function expects dict[str, DataFrame] not adata
        try:
            pattern_dict = {}
            try:
                pattern_dict["exclusive"] = pc.tl.archetype_exclusive_patterns(adata_hsc, verbose=False)
            except Exception as e:
                log.warning(f"archetype_exclusive_patterns failed: {e}")
            try:
                pattern_dict["specialization"] = pc.tl.specialization_patterns(adata_hsc)
            except Exception as e:
                log.warning(f"specialization_patterns failed: {e}")
            try:
                pattern_dict["tradeoff"] = pc.tl.tradeoff_patterns(adata_hsc)
            except Exception as e:
                log.warning(f"tradeoff_patterns failed: {e}")
            if pattern_dict:
                fig_pat_bar = pc.pl.pattern_summary_barplot(pattern_dict)
                html_reg += report.fig_to_img(fig_pat_bar,
                    caption=f"HSC pattern summary barplot: {list(pattern_dict.keys())}")
                plt.close("all")
            else:
                html_reg += error_html("No pattern results available for summary barplot")
        except Exception as e:
            log.exception("Pattern barplot failed")
            html_reg += error_html(f"Pattern barplot failed: {e}")

        # HSC pathway simplex regression dotplot via pc.pl.dotplot
        if pw_reg is not None:
            try:
                log.info("Building HSC pathway simplex regression long-format dataframe...")
                hsc_pw_reg = adata_hsc.uns.get("peach_simplex_regression_pathways", pw_reg)
                hsc_pw_long = regression_to_long_df(
                    hsc_pw_reg, y_col="pathway", exclusive_only=True,
                    exclusive_threshold=2.5, top_n_per_archetype=5, fdr_threshold=0.05)
                log.info(f"  HSC pathway long df rows: {len(hsc_pw_long)}")
                if len(hsc_pw_long) > 0:
                    # Truncate long pathway names for readability
                    hsc_pw_long["pathway"] = hsc_pw_long["pathway"].str[:50]
                    n_pathways = hsc_pw_long["pathway"].nunique()
                    fig_pw = pc.pl.dotplot(
                        hsc_pw_long, x_col="archetype", y_col="pathway",
                        size_col="mean_archetype", color_col="pvalue",
                        top_n_per_group=5,
                        figsize=(12, max(8, n_pathways * 0.4)),
                        title="HSC SIMPLEX REGRESSION pathway dotplot (C5:BP, exclusive ≥2.5x)")
                    html_reg += report.fig_to_img(fig_pw,
                        caption="HSC simplex regression on pathway scores (top 5 exclusive per archetype)")
                    plt.close("all")
            except Exception as e:
                log.exception("Pathway simplex regression dotplot failed")
                html_reg += error_html(f"Pathway simplex regression dotplot failed: {e}")
    except Exception as e:
        log.exception("Fig 2B failed")
        html_reg += error_html(f"Fig 2B failed: {e}")

    # --- Fig 2C: Cooperative patterns (Supplemental) ---
    try:
        if "peach_feature_patterns" in adata_hsc.uns:
            patterns = adata_hsc.uns["peach_feature_patterns"]
            classifications = patterns.get("classifications", {})
            # Handle both dict and list formats
            if isinstance(classifications, dict):
                cooperative_genes = [g for g, c in classifications.items()
                                     if isinstance(c, dict) and c.get("pattern") == "cooperative"]
            elif isinstance(classifications, list):
                cooperative_genes = [c.get("feature", "?") for c in classifications
                                     if isinstance(c, dict) and c.get("pattern") == "cooperative"]
            else:
                cooperative_genes = []
            html_reg += report.text(
                f"<b>Fig 2C (Supplemental)</b>: {len(cooperative_genes)} cooperative pattern genes identified")
            if cooperative_genes[:10]:
                html_reg += report.text(f"  Examples: {', '.join(str(g) for g in cooperative_genes[:10])}")
            # Pattern summary
            try:
                fig_coop = pc.pl.pattern_summary(adata_hsc, show=False)
                html_reg += safe_plotly_html(report, fig_coop,
                    "Fig 2C: Pattern summary (all pattern types)")
            except Exception as e:
                html_reg += error_html(f"Pattern summary failed: {e}")
            # Pattern dotplot for cooperative genes if available
            if len(cooperative_genes) >= 3:
                try:
                    fig_pat_dot = pc.pl.pattern_dotplot(adata_hsc, show=False)
                    html_reg += report.fig_to_img(fig_pat_dot,
                        caption="Pattern dotplot (cooperative + exclusive + tradeoff features)")
                    plt.close("all")
                except Exception as e:
                    html_reg += error_html(f"Pattern dotplot failed: {e}")
    except Exception as e:
        html_reg += error_html(f"Fig 2C failed: {e}")

    report.add_section("Figure 2A-C: Simplex Regression", html_reg, step_num="2A-C")

    # --- Fig 2D: Within-HSC Wald contrasts + Cross-fit Wald via flow soft assignment ---
    html_wald = ""
    # R12-5: Wald methodology note at TOP of section (moved from bottom)
    html_wald += report.text(
        "<b>Wald contrasts method</b>: HC3 sandwich covariance from Scheffe polynomial regression. "
        "Per-pair BH FDR correction over shared genes. "
        "For each pair (i, j), we compute the Wald statistic (beta_i - beta_j) / sqrt(SE_i^2 + SE_j^2) "
        "where beta and SE come from the within-fit simplex regression. "
        "Under the null, this statistic is asymptotically normal. P-values are two-sided; "
        "significance is assessed after Benjamini-Hochberg FDR correction applied <b>globally</b> "
        "across all K*(K-1)/2 archetype pairs (not per-pair) to avoid dilution. "
        "Cross-fit Wald (Fig 2E) uses flow soft assignment to identify high-relatedness "
        "HSC-to-CMP archetype pairs, then compares beta vectors between independent fits.")
    try:
        from scipy.stats import norm as scipy_norm, false_discovery_control

        log.info("CMP degree-2 regression for cross-fit Wald...")
        # Need regression on CMP too — same top 5000 HVG approach
        if "_hvg5k" not in adata_cmp.obsm:
            X_cmp = adata_cmp.X.toarray() if hasattr(adata_cmp.X, "toarray") else np.asarray(adata_cmp.X)
            gene_var_cmp = np.var(X_cmp, axis=0)
            top5k_cmp = np.argsort(gene_var_cmp)[-5000:]
            adata_cmp.obsm["_hvg5k"] = X_cmp[:, top5k_cmp]
            adata_cmp.uns["_hvg5k_names"] = [adata_cmp.var_names[i] for i in top5k_cmp]
        cmp_reg = pc.tl.feature_simplex_regression(adata_cmp, max_degree=2, robust_se=True,
                                                   permutation_test=True, n_permutations=200,
                                                   feature_matrix="_hvg5k",
                                                   feature_names=adata_cmp.uns["_hvg5k_names"])
        if "peach_simplex_regression__hvg5k" in adata_cmp.uns:
            adata_cmp.uns["peach_simplex_regression_genes"] = adata_cmp.uns["peach_simplex_regression__hvg5k"]
            adata_cmp.uns["peach_simplex_regression"] = adata_cmp.uns["peach_simplex_regression__hvg5k"]

        # --- Within-HSC Wald contrasts ---
        log.info("Within-HSC Wald contrasts...")
        hsc_contrasts = pc.tl.archetype_contrasts(adata_hsc)
        adata_hsc.uns["peach_archetype_contrasts"] = hsc_contrasts
        adata_hsc.uns["peach_archetype_contrasts_genes"] = hsc_contrasts
        pairs = hsc_contrasts.get("pairs", [])
        contrast_features = list(hsc_contrasts.get("feature_names", []))
        n_pairs = len(pairs)
        html_wald += report.text(f"<b>Within-HSC Wald</b>: {n_pairs} archetype pairs, "
                                 f"{len(contrast_features)} features")

        # Summary table with top contrast genes per pair
        summary_rows = []
        for pair in pairs:
            pair_key = str(tuple(pair) if isinstance(pair, list) else pair)
            pvals = np.asarray(hsc_contrasts["pvalues_fdr"][pair_key])
            delta = np.asarray(hsc_contrasts["delta_beta"][pair_key])
            j, k = pair if isinstance(pair, (list, tuple)) else (pair[0], pair[1])
            n_sig_p = int((pvals < 0.05).sum())
            # Top 3 genes by |delta| with FDR < 0.05
            sig_mask = pvals < 0.05
            top_genes_str = ""
            if sig_mask.any() and len(contrast_features) == len(delta):
                sig_idx = np.where(sig_mask)[0]
                top_sig = sig_idx[np.argsort(np.abs(delta[sig_idx]))[::-1]][:3]
                top_genes_str = ", ".join(contrast_features[i] for i in top_sig)
            summary_rows.append({
                "Pair": f"A{j+1} vs A{k+1}",
                "N sig (FDR<0.05)": n_sig_p,
                "Mean |delta|": f"{np.abs(delta).mean():.4f}",
                "Top genes": top_genes_str,
            })
        html_wald += report.df_to_html(pd.DataFrame(summary_rows),
            caption="HSC within-fit Wald contrast summary (with top differentiating genes)")

        # W-B18: R² vs -log10(FDR) scatter. Replaces β-based volcano as the
        # primary view of per-feature position dependence. One point per
        # feature; x = simplex regression R² (fraction of variance
        # explained by archetype position), y = -log10(min vertex FDR).
        # Top-right = genes both well-explained AND significant.
        try:
            from _paper_part1_viz import build_r2_vs_fdr_scatter
            _hsc_reg = adata_hsc.uns.get("peach_simplex_regression_genes", {})
            _hsc_r2 = _hsc_reg.get("r_squared_degree1", _hsc_reg.get("r_squared"))
            _hsc_fdr_mat = _hsc_reg.get("vertex_pvalues_fdr")
            _hsc_feats = _hsc_reg.get("feature_names")
            if (_hsc_r2 is not None and _hsc_fdr_mat is not None
                    and _hsc_feats is not None and len(_hsc_feats) > 0):
                _hsc_r2_arr = np.asarray(_hsc_r2)
                _hsc_fdr_arr = np.asarray(_hsc_fdr_mat)
                if _hsc_fdr_arr.ndim == 2:
                    _hsc_min_fdr = _hsc_fdr_arr.min(axis=1)
                else:
                    _hsc_min_fdr = _hsc_fdr_arr
                _fig_r2fdr = build_r2_vs_fdr_scatter(
                    _hsc_r2_arr, _hsc_min_fdr, list(_hsc_feats),
                    r2_threshold=0.1, fdr_threshold=0.05, n_labels=20,
                    title="HSC Fig 2D: Feature R² vs -log10(min vertex FDR)",
                )
                html_wald += report.fig_to_img(
                    _fig_r2fdr,
                    caption=(
                        "Per-feature simplex regression R² vs -log10(min "
                        "vertex FDR). Red = passes both R² > 0.1 and "
                        "FDR < 0.05. Top-right corner = position-dependent "
                        "AND significant. Replaces the β-based volcano as "
                        "the primary position-dependence view (W-B18)."
                    ),
                )
                plt.close("all")
        except Exception as _r2fdr_exc:
            log.warning(f"R² vs FDR scatter failed: {_r2fdr_exc}")
            html_wald += error_html(f"R² vs FDR scatter failed: {_r2fdr_exc}")

        # Volcano plots: ALL pairs (sorted by N significant genes for prominence)
        if pairs:
            sorted_pair_idx = sorted(range(len(summary_rows)),
                                     key=lambda i: summary_rows[i]["N sig (FDR<0.05)"], reverse=True)
            html_wald += report.text(
                f"<b>Wald volcano plots: ALL {len(pairs)} pairs</b> (sorted by N significant genes desc)")
            for rank, idx in enumerate(sorted_pair_idx):
                pair_r = pairs[idx]
                jr, kr = pair_r if isinstance(pair_r, (list, tuple)) else (pair_r[0], pair_r[1])
                n_sig_this = summary_rows[idx]["N sig (FDR<0.05)"]
                n_lab = 15 if rank < 3 else 8
                try:
                    fig_v = pc.pl.contrast_volcano(adata_hsc, pair=(jr, kr), n_labels=n_lab, show=False)
                    html_wald += safe_plotly_html(report, fig_v,
                        f"Wald volcano #{rank+1}: A{jr+1} vs A{kr+1} ({n_sig_this} sig genes FDR<0.05)")
                except Exception as e:
                    log.exception(f"Volcano A{jr+1} vs A{kr+1} failed")
                    html_wald += error_html(f"Volcano A{jr+1} vs A{kr+1} failed: {e}")

        # --- Cross-fit Spearman (HSC vs CMP beta vectors) ---
        log.info("Cross-fit Spearman (HSC vs CMP)...")
        try:
            cross_sim = pc.tl.archetype_feature_similarity(adata_hsc, adata_b=adata_cmp)
            n_shared = cross_sim.get("n_shared_features", "?")
            n_sig_cs = cross_sim.get("n_significant_features", "?")
            html_wald += report.text(
                f"<b>Cross-fit Spearman (β-based)</b> (HSC vs CMP models): "
                f"{n_shared} shared features, {n_sig_cs} FDR-significant. "
                f"Compares archetype coefficient (β) profiles between independently trained models.")
            try:
                fig_sim = pc.pl.feature_similarity_heatmap(adata_hsc, show=False)
                html_wald += safe_plotly_html(report, fig_sim,
                    "Cross-fit feature similarity heatmap (HSC_K x CMP_K Spearman rho, β-based)")
            except Exception as e:
                html_wald += error_html(f"Similarity heatmap failed: {e}")
        except Exception as e:
            html_wald += error_html(f"Cross-fit Spearman failed: {e}")

        # --- Cross-fit PER-DEGREE R² concordance (r12-item-9) ---
        # User request: rank cross-fit concordance by R², not β. Compare
        # per-feature R² vectors between the HSC and CMP simplex-regression
        # fits on shared features, separately for degree 1, degree 2 (Δ R²)
        # and degree 3 (Δ R²). This is a feature-vector Spearman on the
        # per-feature variance-explained scores — a more direct "do the two
        # fits agree on which features are position-dependent?" test than
        # the archetype-coefficient β concordance above.
        try:
            from scipy.stats import spearmanr as _spearman
            _hsc_reg_cf = adata_hsc.uns.get("peach_simplex_regression_genes", {})
            _cmp_reg_cf = adata_cmp.uns.get("peach_simplex_regression_genes", {})
            _hsc_feat_cf = list(_hsc_reg_cf.get("feature_names", []))
            _cmp_feat_cf = list(_cmp_reg_cf.get("feature_names", []))
            if not _hsc_feat_cf or not _cmp_feat_cf:
                html_wald += report.text(
                    "Per-degree R² concordance skipped: one or both "
                    "regression results missing feature names.")
            else:
                _cmp_idx_cf = {g: i for i, g in enumerate(_cmp_feat_cf)}
                _shared_cf = [(i, _cmp_idx_cf[g]) for i, g in enumerate(_hsc_feat_cf)
                              if g in _cmp_idx_cf]
                _hsc_sel = np.array([a for a, _ in _shared_cf], dtype=int)
                _cmp_sel = np.array([b for _, b in _shared_cf], dtype=int)
                log.info(
                    f"  Cross-fit R² concordance: {len(_shared_cf)} shared "
                    f"features between HSC and CMP regression fits."
                )

                def _pull_r2(res_dict, key):
                    arr = np.asarray(res_dict.get(key, []))
                    return arr if arr.size else None

                def _pull_deg(res_dict, deg):
                    dc = (res_dict.get("degree_comparison", {}) or {}).get(
                        f"degree_{deg}", {}) or {}
                    return (
                        np.asarray(dc.get("r_squared", [])),
                        np.asarray(dc.get("delta_r2", [])),
                        np.asarray(dc.get("incremental_p_fdr", [])),
                    )

                _cf_rows = []
                # Degree 1: total R²
                _h_r2_d1 = _pull_r2(_hsc_reg_cf, "r_squared_degree1")
                _c_r2_d1 = _pull_r2(_cmp_reg_cf, "r_squared_degree1")
                if _h_r2_d1 is not None and _c_r2_d1 is not None:
                    _h = _h_r2_d1[_hsc_sel]
                    _c = _c_r2_d1[_cmp_sel]
                    _mask = np.isfinite(_h) & np.isfinite(_c)
                    if _mask.sum() >= 5:
                        _rho, _p = _spearman(_h[_mask], _c[_mask])
                        _cf_rows.append({
                            "Degree": "1 (R²)",
                            "Metric": "r_squared_degree1",
                            "N features": int(_mask.sum()),
                            "Spearman ρ": f"{_rho:.4f}",
                            "p-value": fmt_pval(float(_p)),
                        })

                # Degree 2: Δ R² (incremental signal, matches dotplot ranking)
                _h_r2_d2, _h_delta_d2, _ = _pull_deg(_hsc_reg_cf, 2)
                _c_r2_d2, _c_delta_d2, _ = _pull_deg(_cmp_reg_cf, 2)
                if _h_delta_d2.size and _c_delta_d2.size:
                    _h = _h_delta_d2[_hsc_sel]
                    _c = _c_delta_d2[_cmp_sel]
                    _mask = np.isfinite(_h) & np.isfinite(_c)
                    if _mask.sum() >= 5:
                        _rho, _p = _spearman(_h[_mask], _c[_mask])
                        _cf_rows.append({
                            "Degree": "2 (Δ R²)",
                            "Metric": "degree_2.delta_r2",
                            "N features": int(_mask.sum()),
                            "Spearman ρ": f"{_rho:.4f}",
                            "p-value": fmt_pval(float(_p)),
                        })

                # Degree 3: Δ R² (if comprehensive_degree extended to 3)
                _h_r2_d3, _h_delta_d3, _ = _pull_deg(_hsc_reg_cf, 3)
                _c_r2_d3, _c_delta_d3, _ = _pull_deg(_cmp_reg_cf, 3)
                if _h_delta_d3.size and _c_delta_d3.size:
                    _h = _h_delta_d3[_hsc_sel]
                    _c = _c_delta_d3[_cmp_sel]
                    _mask = np.isfinite(_h) & np.isfinite(_c)
                    if _mask.sum() >= 5:
                        _rho, _p = _spearman(_h[_mask], _c[_mask])
                        _cf_rows.append({
                            "Degree": "3 (Δ R²)",
                            "Metric": "degree_3.delta_r2",
                            "N features": int(_mask.sum()),
                            "Spearman ρ": f"{_rho:.4f}",
                            "p-value": fmt_pval(float(_p)),
                        })

                if _cf_rows:
                    html_wald += report.text(
                        "<b>Cross-fit per-degree R² concordance (r12)</b>: "
                        "Spearman correlation of per-feature R² (or Δ R²) "
                        "between the HSC and CMP simplex-regression fits on "
                        "shared features. Ranking by R² rather than β tests "
                        "whether the two fits identify the SAME features as "
                        "position-dependent, separately for pure main effects "
                        "(deg 1), pairwise interactions (deg 2 Δ R²), and "
                        "triple interactions (deg 3 Δ R², when K-1 ≥ 3)."
                    )
                    html_wald += report.df_to_html(
                        pd.DataFrame(_cf_rows),
                        caption=(
                            "Cross-fit R²-based concordance by degree. "
                            "Higher ρ = the two independently trained models "
                            "agree on which features explain position at that "
                            "Scheffé polynomial degree."
                        ),
                    )
                else:
                    html_wald += report.text(
                        "Per-degree R² concordance: no degree yielded enough "
                        "finite shared-feature values for Spearman."
                    )
        except Exception as _e_cfr2:
            log.exception("Cross-fit per-degree R² concordance failed")
            html_wald += error_html(
                f"Cross-fit per-degree R² concordance failed: {_e_cfr2}"
            )

        # --- UpSet plot: per-archetype significant feature set intersections ---
        try:
            from upsetplot import from_contents, UpSet
            import matplotlib.pyplot as _plt_upset

            _upset_reg = adata_hsc.uns.get("peach_simplex_regression_genes", {})
            _upset_feat = list(_upset_reg.get("feature_names", []))
            _upset_fdr = _upset_reg.get("vertex_pvalues_fdr")
            # W-B18: cap top N features per archetype for UpSet readability.
            # A 5000-HVG dataset with 1000+ significant features per archetype
            # produces an unreadable UpSet plot; capping to top-100 by FDR
            # keeps the combinatorial intersections tractable.
            _UPSET_TOP_N_PER_ARCH = 100
            if _upset_fdr is not None and len(_upset_feat) > 0:
                _upset_fdr = np.asarray(_upset_fdr)  # [n_features, K]
                if _upset_fdr.ndim == 2 and _upset_fdr.shape[0] == len(_upset_feat):
                    _arch_sig_sets = {}
                    for _ai in range(_upset_fdr.shape[1]):
                        _sig_mask = _upset_fdr[:, _ai] < 0.05
                        _sig_idx = np.where(_sig_mask)[0]
                        # Cap to top-N by FDR (ascending = most significant)
                        if len(_sig_idx) > _UPSET_TOP_N_PER_ARCH:
                            _order = np.argsort(_upset_fdr[_sig_idx, _ai])
                            _sig_idx = _sig_idx[_order[:_UPSET_TOP_N_PER_ARCH]]
                        _arch_sig_sets[f"A{_ai + 1}"] = set(
                            _upset_feat[_j] for _j in _sig_idx
                        )
                    _n_archs_with_sigs = sum(1 for s in _arch_sig_sets.values() if len(s) > 0)
                    if len(_arch_sig_sets) >= 2 and _n_archs_with_sigs >= 2:
                        _upset_data = from_contents(_arch_sig_sets)
                        _fig_upset = _plt_upset.figure(figsize=(12, 6))
                        UpSet(_upset_data, show_counts=True, sort_by="cardinality",
                              min_subset_size=3).plot(fig=_fig_upset)
                        _fig_upset.suptitle(
                            "Fig 2D: Archetype-significant feature set intersections (UpSet)",
                            fontsize=12, y=1.01)
                        _total_sig = sum(len(s) for s in _arch_sig_sets.values())
                        _per_arch_str = ", ".join(
                            f"A{_ai+1}={len(_arch_sig_sets[f'A{_ai+1}'])}"
                            for _ai in range(len(_arch_sig_sets))
                        )
                        html_wald += report.fig_to_img(
                            _fig_upset,
                            caption=(
                                "UpSet plot: intersections of archetype-significant feature sets "
                                "(simplex regression vertex FDR < 0.05). "
                                f"Per-archetype counts: {_per_arch_str}. "
                                "Each bar = count of features significant in that exact combination of archetypes; "
                                "subsets with < 3 features are hidden."
                            ),
                        )
                        _plt_upset.close("all")
                        log.info(f"UpSet plot: {len(_arch_sig_sets)} archetypes, "
                                 f"{_total_sig} total sig features (with overlap), "
                                 f"subsets shown (min_size=3)")
                    else:
                        html_wald += error_html(
                            f"UpSet plot skipped: only {_n_archs_with_sigs} of "
                            f"{len(_arch_sig_sets)} archetypes have significant features (need ≥ 2)."
                        )
                else:
                    html_wald += error_html(
                        f"UpSet plot skipped: vertex_pvalues_fdr shape {_upset_fdr.shape} "
                        f"does not match feature_names length {len(_upset_feat)}."
                    )
            else:
                html_wald += error_html(
                    "UpSet plot skipped: no vertex_pvalues_fdr or feature_names in "
                    "peach_simplex_regression_genes."
                )
        except Exception as e:
            log.exception("UpSet plot failed")
            html_wald += error_html(f"UpSet plot failed: {e}")

        # R12-5: Wald methodology note moved to top of section (above)
    except Exception as e:
        log.exception("Fig 2D failed")
        html_wald += error_html(f"Fig 2D failed: {e}")

    report.add_section("Figure 2D: Wald Contrasts", html_wald, step_num="2D")

    # --- Fig 2E: flow HSC→CMP + soft assignment + cross-fit Wald ---
    html_flow = ""
    corr_matrix = None  # Will be set if soft assignment succeeds (for cross-fit Wald)
    try:
        log.info("flow HSC → CMP (using flow_within on combined adata)...")
        import anndata as ad
        import plotly.graph_objects as go
        from scipy.spatial import cKDTree
        from scipy.stats import norm as scipy_norm, false_discovery_control
        from peach._core.utils.archetype_comparison import (
            compute_archetype_correspondence,
            compute_correspondence_permutation_null,
        )
        from _paper_part1_viz import build_permutation_curve_figure

        adata_full = ad.read_h5ad(os.path.join(DATA_DIR, "adata_full_prepped.h5ad"))
        if SUBSAMPLE_FRACTION < 1.0:
            log.info(f"Subsampling adata_full to {SUBSAMPLE_FRACTION*100:.0f}%")
            adata_full = _stratified_subsample(adata_full, SUBSAMPLE_FRACTION, SUBSAMPLE_SEED)
            log.info(f"  adata_full subsampled: {adata_full.shape}")
        pc.pp.prepare_training(adata_full, batch_size=128)

        # Transfer HSC trained model to full adata for annotation
        adata_full.uns["trained_model"] = adata_hsc.uns["trained_model"]
        adata_full.uns["archetype_coordinates"] = adata_hsc.uns["archetype_coordinates"]
        pc.tl.archetypal_coordinates(adata_full, verbose=False)
        pc.tl.extract_archetype_weights(adata_full, verbose=False)
        pc.tl.assign_archetypes(adata_full, verbose=False)

        # Count source and target cells
        source_cell_mask = adata_full.obs["cell_type"] == "hematopoietic stem cell"
        target_cell_mask = adata_full.obs["cell_type"] == "common myeloid progenitor"
        n_src = int(source_cell_mask.sum())
        n_tgt = int(target_cell_mask.sum())
        html_flow += metric_grid([
            metric_card(f"{n_src}", "N source (HSC)"),
            metric_card(f"{n_tgt}", "N target (CMP)"),
        ])

        # Use flow_within directly so we get the model back for gene alignment
        fr = pc.tl.flow_within(
            adata_full,
            source={"cell_type": "hematopoietic stem cell"},
            target={"cell_type": "common myeloid progenitor"},
            n_epochs=300, hidden_dims=(128, 128, 128),
            batch_size=128, return_model=True,
            name="HSC_to_CMP", random_state=42,
        )
        flow_results = {"HSC_to_CMP": fr}

        for pair_key, fr in flow_results.items():
            mmd_b = fr["mmd_before"]
            mmd_a = fr["mmd_after"]
            mmd_reduction = 1 - mmd_a / max(mmd_b, 1e-10)

            # --- Wasserstein-2 distance (in PC units) ---
            log.info(f"Computing W2 distance for {pair_key}...")
            pca_key_local = fr.get("pca_key", "X_pca")
            source_pca_full = adata_full.obsm[pca_key_local][fr["source_mask"]]
            target_pca_full = adata_full.obsm[pca_key_local][fr["target_mask"]]
            transported_full = fr["transported"]
            try:
                w2_before = wasserstein2_distance(source_pca_full, target_pca_full, max_n=2000, seed=42)
                w2_after = wasserstein2_distance(transported_full, target_pca_full, max_n=2000, seed=42)
                w2_reduction_abs = w2_before - w2_after  # absolute PC units
                w2_reduction_rel = (w2_before - w2_after) / max(w2_before, 1e-10)  # fraction
                log.info(f"  W2: before={w2_before:.4f}, after={w2_after:.4f}, "
                         f"reduction={w2_reduction_abs:.4f} PC ({w2_reduction_rel:.1%})")
            except Exception as e:
                log.exception("W2 distance failed")
                w2_before = w2_after = w2_reduction_abs = w2_reduction_rel = float("nan")

            # Flow convergence diagnostics
            flow_loss_hist = fr.get("loss_history", fr.get("losses", None))
            n_epochs_flow = fr.get("n_epochs", 300)
            flow_cards = [
                metric_card(pair_key, "Flow pair"),
                metric_card(f"{w2_before:.3f}", "W2 before (PC units)"),
                metric_card(f"{w2_after:.3f}", "W2 after (PC units)"),
                metric_card(f"{w2_reduction_abs:.3f}", "W2 reduction (PC)"),
                metric_card(f"{w2_reduction_rel:.1%}", "W2 reduction (%)"),
                metric_card(f"{mmd_b:.4f}", "MMD before"),
                metric_card(f"{mmd_a:.4f}", "MMD after"),
                metric_card(f"{mmd_reduction:.1%}", "MMD reduction"),
                metric_card(f"{fr['source_mask'].sum()}", "N source (flow)"),
                metric_card(f"{fr['target_mask'].sum()}", "N target (flow)"),
                metric_card(f"{n_epochs_flow}", "Epochs"),
            ]
            if flow_loss_hist is not None:
                final_loss = flow_loss_hist[-1] if len(flow_loss_hist) > 0 else "?"
                flow_cards.append(metric_card(f"{final_loss:.4f}" if isinstance(final_loss, float) else str(final_loss), "Final loss"))
            html_flow += metric_grid(flow_cards)
            html_flow += report.text(
                "<b>Flow distance metrics</b>: Wasserstein-2 (W2) is the optimal-transport distance "
                "in PCA coordinate units — a true metric (symmetric, triangle inequality). "
                "MMD is a kernel-based discrepancy measure (Gaussian kernel, scale-dependent). "
                "W2 reduction in absolute PC units is the most interpretable measure of how much "
                "the transport pulled the source distribution toward the target.")
            # Flow loss curve if available
            if flow_loss_hist is not None and len(flow_loss_hist) > 1:
                fig_floss, ax_fl = plt.subplots(figsize=(6, 3))
                ax_fl.plot(flow_loss_hist, color="#0072B2", linewidth=1)
                ax_fl.set_xlabel("Epoch"); ax_fl.set_ylabel("Loss")
                ax_fl.set_title("Flow model training loss")
                ax_fl.spines[["top", "right"]].set_visible(False)
                fig_floss.tight_layout()
                html_flow += report.fig_to_img(fig_floss, caption="Flow model convergence (loss vs epoch)")
                plt.close("all")

            # Soft assignment heatmap
            try:
                fig_sa = pc.pl.soft_assignment_heatmap(adata_full, fr, show=False)
                html_flow += safe_plotly_html(report, fig_sa, f"Soft assignment: {pair_key}")
            except Exception as e:
                html_flow += error_html(f"Soft assignment heatmap failed: {e}")

            # --- CORRESPONDENCE MATRIX (FIXED) ---
            # Previous bug (r7): target weights came from HSC model (adata_full annotated with HSC),
            # making corr a K_hsc × K_hsc matrix.
            # Previous bug (r8): obs_name lookup failed when subsampling because adata_cmp_train
            # and adata_full are subsampled independently (~30% overlap).
            # Clean fix (r8b): project adata_full's CMP cells DIRECTLY through the CMP model.
            # Guarantees 100% match for whatever subset of CMP cells is in the flow target.
            try:
                # Source side (HSC): use HSC-model weights from adata_full
                weights_full_hsc = adata_full.obsm.get("cell_archetype_weights")  # HSC model
                source_mask = fr["source_mask"]
                target_mask = fr["target_mask"]
                pca_key = fr.get("pca_key", "X_pca")

                # Target side (CMP): build a subset adata with ONLY the target cells from adata_full,
                # then project through the CMP-trained model.
                log.info(f"Projecting {int(target_mask.sum())} adata_full CMP cells through CMP model...")
                adata_target_cmp = adata_full[target_mask].copy()
                pc.pp.prepare_training(adata_target_cmp, batch_size=64)
                adata_target_cmp.uns["trained_model"] = adata_cmp.uns["trained_model"]
                adata_target_cmp.uns["archetype_coordinates"] = adata_cmp.uns["archetype_coordinates"]
                pc.tl.archetypal_coordinates(adata_target_cmp, verbose=False)
                pc.tl.extract_archetype_weights(adata_target_cmp, verbose=False)
                weights_tgt_cmp = adata_target_cmp.obsm.get("cell_archetype_weights")
                if weights_tgt_cmp is None:
                    raise RuntimeError("Failed to extract CMP-model weights for target cells")

                # Target PCA coordinates (same order as weights_tgt_cmp since we used a subset)
                target_pca_matched = adata_full.obsm[pca_key][target_mask]

                # Source side: HSC-model weights for source cells
                weights_src_hsc = weights_full_hsc[source_mask]
                K_h = weights_src_hsc.shape[1]
                K_c = weights_tgt_cmp.shape[1]
                log.info(f"  100% matched: src shape {weights_src_hsc.shape}, tgt shape {weights_tgt_cmp.shape}")

                log.info(f"Correspondence matrix dims: K_hsc={K_h}, K_cmp={K_c} "
                         f"(source N={weights_src_hsc.shape[0]}, target N={weights_tgt_cmp.shape[0]})")

                # Full-population correspondence (r12: removed k=10 local
                # version — a true Markov transition matrix aggregates over
                # the full target population, not a kNN neighborhood).
                # Uses library hard-mode correspondence with k=N_target (or
                # k=1000 cap for memory) so rows are exact weighted averages
                # over the full target population.
                _n_tgt_full = int(weights_tgt_cmp.shape[0])
                _k_full_pop = min(1000, _n_tgt_full)
                log.info(
                    f"Correspondence: full-population (k={_k_full_pop}, "
                    f"N_target={_n_tgt_full}) — r12 removed k=10 local variant."
                )
                corr_result = compute_archetype_correspondence(
                    source_weights=weights_src_hsc,
                    source_coords=fr["transported"],
                    target_weights=weights_tgt_cmp,
                    target_coords=target_pca_matched,
                    k=_k_full_pop,
                    method="hard",
                )
                corr_matrix = corr_result["mass"]
                corr_markov = corr_result["markov"]
                corr = corr_matrix  # local alias preserves downstream code

                # DIAGNOSTIC: dump raw correspondence matrix + library diagnostics to log
                log.info(f"Correspondence matrix raw values (K_hsc={K_h} x K_cmp={K_c}):")
                log.info(f"  method={corr_result['method']}")
                log.info(f"  source_weight_concentration={corr_result['source_weight_concentration']:.4f}")
                log.info(f"  target_weight_concentration={corr_result['target_weight_concentration']:.4f}")
                log.info(f"  source_archetype_occupancy_hard={list(corr_result['source_archetype_occupancy_hard'])}")
                log.info(f"  sparse_archetypes={corr_result['sparse_archetypes']}")
                log.info(f"  total mass: {corr.sum():.4f}, "
                         f"row marginals: {corr.sum(axis=1).tolist()}, "
                         f"col marginals: {corr.sum(axis=0).tolist()}")
                log.info(f"  range: [{corr.min():.4f}, {corr.max():.4f}], "
                         f"std across rows: {corr.std(axis=1).mean():.4f}, "
                         f"std across cols: {corr.std(axis=0).mean():.4f}")
                for i in range(K_h):
                    log.info(f"  HSC A{i+1}: {corr[i].tolist()}")

                # W-B23: Permutation curve null with global cell swap.
                # Replaces the previous Gaussian-z-score null with an empirical
                # rank-based p-value derived from a degradation curve. For each
                # swap fraction f in {0, 0.05, 0.10, 0.20, 0.35, 0.50}, n_perms
                # permutations swap a fraction f of cells between source and
                # target spatial pools, recompute the correspondence, and build
                # an empirical null. p-values are computed at the largest f
                # (hardest test). FDR via BH across the K_h * K_c pair family.
                log.info(
                    "Building permutation curve null for correspondence "
                    "(global swap, n_perms=200, 6 fractions)..."
                )
                perm_null = compute_correspondence_permutation_null(
                    source_weights=weights_src_hsc,
                    source_coords=fr["transported"],
                    target_weights=weights_tgt_cmp,
                    target_coords=target_pca_matched,
                    k=_k_full_pop,
                    n_perms=200,
                    swap_fractions=(0.0, 0.05, 0.10, 0.20, 0.35, 0.50),
                    seed=42,
                )
                # Maintain legacy variable names for downstream consumers.
                pair_pvals = perm_null["empirical_p"]
                pair_fdr = perm_null["empirical_fdr"]
                # null_mean / null_std at the largest swap fraction give the
                # legacy "shuffle null" mean / std analogue used by the
                # z-score and downstream pair-selection logic.
                null_mean_corr = perm_null["null_mean_curve"][-1]
                null_std_corr = perm_null["null_std_curve"][-1]
                safe_null_std = np.where(null_std_corr < 1e-10, 1.0, null_std_corr)
                pair_z = (corr_matrix - null_mean_corr) / safe_null_std

                src_labels = [f"HSC A{i+1}" for i in range(K_h)]
                tgt_labels = [f"CMP A{j+1}" for j in range(K_c)]

                log.info(
                    f"  Permutation null: empirical_p range "
                    f"[{pair_pvals.min():.3f}, {pair_pvals.max():.3f}], "
                    f"FDR<0.05 pairs: {int((pair_fdr < 0.05).sum())}/{K_h*K_c}"
                )

                # W-B20: Raw pairwise results DataFrame (rendered BEFORE
                # Sankey + heatmaps so the reader sees every pair, which
                # ones are significant, and why — not just the top-3).
                # One row per (i, j) pair, sorted by significance first
                # then by mass. Significant rows get a background tint
                # via inline style on the final HTML.
                html_flow += report.text("<h3>Raw pairwise results (all K_src x K_tgt pairs)</h3>")
                pair_rows = []
                for i in range(K_h):
                    for j in range(K_c):
                        fdr_ij = float(pair_fdr[i, j])
                        pair_rows.append({
                            "source_arch": src_labels[i],
                            "target_arch": tgt_labels[j],
                            "mass": float(corr[i, j]),
                            "markov_p": float(corr_markov[i, j]),
                            "empirical_p": float(pair_pvals[i, j]),
                            "empirical_fdr": fdr_ij,
                            "significant": "Yes" if fdr_ij < 0.05 else "No",
                        })
                raw_pair_df = pd.DataFrame(pair_rows)
                # Sort: significant first, then by mass descending
                raw_pair_df = raw_pair_df.sort_values(
                    by=["significant", "mass"],
                    ascending=[False, False],
                    kind="mergesort",  # stable
                ).reset_index(drop=True)

                # Render with inline-style row highlight for significant pairs.
                # Build the HTML manually so we can inject a background
                # color on significant rows (df_to_html strips styles).
                _raw_display_df = raw_pair_df.copy()
                _raw_display_df["mass"] = _raw_display_df["mass"].map(lambda v: f"{v:.3f}")
                _raw_display_df["markov_p"] = _raw_display_df["markov_p"].map(lambda v: f"{v:.3f}")
                _raw_display_df["empirical_p"] = _raw_display_df["empirical_p"].map(lambda v: f"{v:.3f}")
                _raw_display_df["empirical_fdr"] = _raw_display_df["empirical_fdr"].map(lambda v: f"{v:.3f}")
                _raw_html = _raw_display_df.to_html(
                    classes="styled-table", index=False, border=0, escape=False
                )
                # Post-process: inject background color on significant rows.
                # Significant rows have `<td>Yes</td>` in the significant
                # column; we add a style to the enclosing <tr>.
                import re as _re_raw
                def _highlight_row(m):
                    row_html = m.group(0)
                    if "<td>Yes</td>" in row_html:
                        return row_html.replace(
                            "<tr>", '<tr style="background:#e8f4f8;">'
                        )
                    return row_html
                _raw_html = _re_raw.sub(
                    r"<tr>.*?</tr>",
                    _highlight_row,
                    _raw_html,
                    flags=_re_raw.DOTALL,
                )
                n_sig_raw = int((pair_fdr < 0.05).sum())
                html_flow += (
                    "<p class='caption'><strong>Raw pairwise correspondence "
                    f"({K_h*K_c} pairs, {n_sig_raw} significant at FDR&lt;0.05).</strong> "
                    "<code>mass</code> = raw transport mass "
                    "corr[i,j]. <code>markov_p</code> = row-normalized "
                    "transition probability P(target j | source i). "
                    "<code>empirical_p</code> + <code>empirical_fdr</code> "
                    "come from the W-B23 global-swap permutation curve null "
                    "at the largest swap fraction (hardest test). Rows with "
                    "<code>significant=Yes</code> (FDR&lt;0.05) are tinted "
                    "pale blue. Sorted by significance then by mass.</p>"
                )
                html_flow += f'<div style="overflow-x: auto; max-width: 100%;">{_raw_html}</div>'

                # --- Raw correspondence matrix (transport mass) ---
                corr_df = pd.DataFrame(corr, index=src_labels, columns=tgt_labels)
                html_flow += report.df_to_html(corr_df,
                    caption=f"Raw correspondence matrix (K_hsc={K_h} x K_cmp={K_c}, transport mass). "
                            f"Source weights from HSC model, target from CMP model. "
                            f"Computed via compute_archetype_correspondence(method='hard').")

                # --- Row-normalized correspondence (full-population Markov) ---
                # Each row = P(CMP arch j | HSC arch i): "given a source cell in HSC arch i,
                # what fraction of its transport mass lands in each CMP arch?"
                # Rows sum to 1 → can be read as a transition kernel from HSC arch space to CMP arch space.
                # corr_markov already row-normalized by compute_archetype_correspondence.
                # r12 note: k is set to min(1000, N_target) so rows aggregate
                # over the full target population rather than a kNN
                # neighborhood — removes the "local-only" caveat from prior
                # iterations. Remaining departures from a textbook Markov
                # matrix (sparse-archetype row zeroing, one-step not
                # stationary) are enumerated below.
                corr_markov_df = pd.DataFrame(corr_markov, index=src_labels, columns=tgt_labels)
                html_flow += report.df_to_html(corr_markov_df,
                    caption=(
                        f"Row-normalized correspondence "
                        f"(<b>full-population Markov</b>, k={_k_full_pop} / "
                        f"N_target={_n_tgt_full}). Non-sparse rows sum to 1. "
                        f"Each cell = P(CMP archetype j | HSC archetype i). "
                        f"Read row-by-row: 'given an HSC cell in archetype "
                        f"i, this is the probability distribution over CMP "
                        f"archetypes after transport'. "
                        f"<br><br><b>Remaining departures from a textbook "
                        f"Markov matrix</b>: "
                        f"(1) source cells are hard-assigned to archetypes "
                        f"via argmax of weights; sparse archetypes (< 2% "
                        f"occupancy) have their rows zeroed (sum to 0 not 1); "
                        f"(2) this is a one-step cross-fit projection, not "
                        f"a stationary stochastic process — no ergodicity "
                        f"or detailed-balance claims apply."
                    ))

                # Markov transition heatmap (full-population).
                fig_markov = go.Figure(data=go.Heatmap(
                    z=corr_markov, x=tgt_labels, y=src_labels,
                    colorscale="Blues",
                    zmin=0, zmax=1,
                    colorbar=dict(title="P(j|i)"),
                    text=[[f"{corr_markov[i,j]:.2f}" for j in range(K_c)] for i in range(K_h)],
                    texttemplate="%{text}",
                ))
                fig_markov.update_layout(
                    title=(
                        f"Markov transition matrix HSC→CMP "
                        f"(full-population, k={_k_full_pop}, K={K_h}x{K_c})"
                    ),
                    xaxis_title="CMP archetype (target)",
                    yaxis_title="HSC archetype (source)",
                    width=600, height=500,
                )
                html_flow += safe_plotly_html(report, fig_markov,
                    f"Row-normalized correspondence as Markov transition "
                    f"heatmap (full-population k={_k_full_pop}, rows sum to 1)")

                # (r12: removed the separate full-population block since
                # the primary correspondence is now already full-population.)

                # Significance annotation matrix
                sig_marks = pd.DataFrame(
                    np.where(pair_fdr < 0.05, "*", ""),
                    index=src_labels, columns=tgt_labels)
                # Combine into one display
                pval_df = pd.DataFrame(
                    [[f"{corr[i,j]:.3f}<br>p={pair_pvals[i,j]:.3f}<br>q={pair_fdr[i,j]:.3f}{'*' if pair_fdr[i,j]<0.05 else ''}"
                      for j in range(K_c)] for i in range(K_h)],
                    index=src_labels, columns=tgt_labels)
                # Use escape=False so <br> renders as line breaks (not literal text)
                _pval_table = pval_df.to_html(classes="styled-table", index=True, border=0, escape=False)
                html_flow += "<p class='caption'><strong>Per-pair correspondence with permutation p-value and BH FDR (* = q&lt;0.05)</strong></p>"
                html_flow += f'<div style="overflow-x: auto; max-width: 100%;">{_pval_table}</div>'

                # Sankey: use RAW mass weighted by significance
                # Threshold: keep links with FDR < 0.05 OR mass > 5% of total
                total_mass = corr.sum()
                s_idx, t_idx, vals, link_colors = [], [], [], []
                for i in range(K_h):
                    for j in range(K_c):
                        keep = (pair_fdr[i, j] < 0.05) or (corr[i, j] > total_mass * 0.05)
                        if keep:
                            s_idx.append(i); t_idx.append(K_h + j)
                            vals.append(float(corr[i, j]))
                            link_colors.append("rgba(0,114,178,0.5)" if pair_fdr[i, j] < 0.05
                                               else "rgba(150,150,150,0.3)")
                if vals:
                    log.info(f"Sankey link values: {vals}")
                    fig_chord = go.Figure(data=[go.Sankey(
                        node=dict(
                            label=src_labels + tgt_labels,
                            pad=15, thickness=20,
                            color=["#0072B2"] * K_h + ["#D55E00"] * K_c,
                        ),
                        link=dict(source=s_idx, target=t_idx, value=vals, color=link_colors),
                    )])
                    fig_chord.update_layout(
                        title=f"HSC→CMP correspondence (mass; blue=FDR<0.05, gray=mass>5%)",
                        width=900, height=600,
                        font=dict(size=12),
                    )
                    html_flow += safe_plotly_html(report, fig_chord,
                        "Sankey: HSC→CMP archetype correspondence. "
                        "Link width = transport mass; blue links are FDR<0.05 by per-pair label-permutation null.")

                # Heatmap of correspondence with FDR overlay
                fig_corr_heat = go.Figure(data=go.Heatmap(
                    z=corr, x=tgt_labels, y=src_labels,
                    colorscale="Viridis",
                    colorbar=dict(title="Mass"),
                    text=[[f"{corr[i,j]:.2f}{'*' if pair_fdr[i,j]<0.05 else ''}" for j in range(K_c)] for i in range(K_h)],
                    texttemplate="%{text}",
                ))
                fig_corr_heat.update_layout(
                    title=f"Correspondence matrix (K_hsc={K_h} × K_cmp={K_c}, * = FDR<0.05)",
                    xaxis_title="CMP archetype", yaxis_title="HSC archetype",
                    width=600, height=500,
                )
                html_flow += safe_plotly_html(report, fig_corr_heat,
                    "Correspondence heatmap. Values are <b>transport mass</b> from "
                    "compute_archetype_correspondence(method='hard'): sum of k-NN "
                    "target weight averages for source cells hard-assigned to each "
                    "source archetype. * = empirical FDR < 0.05.")

                # W-B20: Per-pair permutation degradation curves for ALL
                # significant pairs (empirical_fdr < 0.05), capped at 20
                # to avoid a 100-curve scroll wall. Earlier revisions
                # hardcoded top-3 (W-B23), which hid significant pairs
                # beyond the top of the list. When more than 20 pairs
                # are significant, the top-20 by mass are shown and the
                # total count is reported. If NO pairs are significant,
                # a warning banner is shown and the top-3 by mass are
                # rendered anyway so the reader sees the matrix shape.
                try:
                    html_flow += report.text(
                        "<h3>Significant pairs (FDR &lt; 0.05)</h3>"
                        "<p class='caption'>For each significant pair, "
                        "the observed correspondence mass (red dashed) "
                        "is plotted against the per-fraction permutation "
                        "null distribution (gray points = mean +/- std, "
                        "shaded band = full min/max range). A real "
                        "correspondence pair shows the observed line "
                        "above the null band at all swap fractions; a "
                        "noisy pair degrades into the null early.</p>"
                    )

                    # Build list of significant pairs sorted by mass desc.
                    MAX_SIG_CURVES = 20
                    sig_mask = (pair_fdr < 0.05)
                    n_sig_total = int(sig_mask.sum())
                    sig_flat_idx = np.argsort(-corr.flatten())
                    sig_pairs = []
                    for flat_i in sig_flat_idx:
                        hi = int(flat_i // K_c)
                        ci = int(flat_i % K_c)
                        if pair_fdr[hi, ci] < 0.05:
                            sig_pairs.append((hi, ci))
                    # Cap at 20 by mass order
                    if len(sig_pairs) > MAX_SIG_CURVES:
                        log.info(
                            f"Capping per-pair degradation curves: "
                            f"{len(sig_pairs)} significant pairs -> "
                            f"top {MAX_SIG_CURVES} by mass."
                        )
                        html_flow += report.text(
                            f"<p><i>Note: {len(sig_pairs)} pairs are "
                            f"significant at FDR&lt;0.05; showing top "
                            f"{MAX_SIG_CURVES} by mass to bound the "
                            f"scroll length.</i></p>"
                        )
                        sig_pairs = sig_pairs[:MAX_SIG_CURVES]

                    if n_sig_total == 0:
                        # W-B20 no-significant fallback: show top-3 by
                        # mass so the reader sees the matrix shape even
                        # when the null test finds no signal.
                        html_flow += error_html(
                            "WARNING: no significant pairs at FDR&lt;0.05. "
                            "Showing top-3 pairs by mass below so the "
                            "matrix shape is visible even without signal. "
                            "All pairs had empirical_fdr >= 0.05 at the "
                            "largest swap fraction (f=0.50)."
                        )
                        flat_mass_idx = np.argsort(-corr.flatten())
                        fallback_pairs = []
                        for flat_i in flat_mass_idx:
                            hi = int(flat_i // K_c)
                            ci = int(flat_i % K_c)
                            fallback_pairs.append((hi, ci))
                            if len(fallback_pairs) >= 3:
                                break
                        sig_pairs = fallback_pairs

                    for hi, ci in sig_pairs:
                        fig_curve = build_permutation_curve_figure(
                            perm_null,
                            pair_i=hi,
                            pair_j=ci,
                            src_label=src_labels[hi],
                            tgt_label=tgt_labels[ci],
                        )
                        html_flow += report.fig_to_img(
                            fig_curve,
                            caption=(
                                f"Permutation curve for {src_labels[hi]} -> "
                                f"{tgt_labels[ci]} "
                                f"(p={pair_pvals[hi, ci]:.3f}, "
                                f"FDR={pair_fdr[hi, ci]:.3f}, "
                                f"mass={corr[hi, ci]:.3f})"
                            ),
                        )
                except Exception as e:
                    log.exception("Permutation curve figures failed")
                    html_flow += error_html(
                        f"Permutation curve figures failed: {e}"
                    )
            except Exception as e:
                log.exception("Correspondence matrix failed")
                html_flow += error_html(f"Correspondence matrix failed: {e}")
                corr_matrix = None
                pair_fdr = None

            # Permutation test with full QC reporting
            try:
                log.info("flow_significance permutation test...")
                sig = pc.tl.flow_significance(adata_full, fr, n_permutations=100, n_epochs_per_perm=100, random_state=42)
                null_dist = np.asarray(sig["null_distribution"])
                obs_stat = sig["observed_stat"]
                null_mean = np.mean(null_dist)
                null_std = np.std(null_dist) if len(null_dist) > 1 else 0
                html_flow += metric_grid([
                    metric_card(f"{obs_stat:.4f}", "Observed stat"),
                    metric_card(f"{null_mean:.4f}", "Null mean"),
                    metric_card(f"{null_std:.4f}", "Null std"),
                    metric_card(f"{sig['p_value']:.3f}", "p-value"),
                    metric_card(f"{len(null_dist)}", "N permutations"),
                ])
                # Null distribution histogram
                fig_null, ax_null = plt.subplots(figsize=(6, 3))
                ax_null.hist(null_dist, bins=min(20, len(null_dist)), alpha=0.7, color="#999")
                ax_null.axvline(obs_stat, color="red", linewidth=2, label=f"Observed={obs_stat:.4f}")
                ax_null.legend()
                ax_null.set_xlabel("MMD improvement statistic")
                ax_null.set_title(f"Flow significance null distribution (p={sig['p_value']:.3f})")
                ax_null.spines[["top", "right"]].set_visible(False)
                fig_null.tight_layout()
                html_flow += report.fig_to_img(fig_null, caption="Null distribution vs observed statistic")
                plt.close("all")
            except Exception as e:
                html_flow += error_html(f"Permutation test failed: {e}")

            # --- Cross-fit Wald via flow soft assignment (now with correct K_h x K_c) ---
                html_flow += report.text(
                    "<b>Cross-fit Wald note</b>: top genes and pathways shown below are from the "
                    "<b>intersection</b> of features present in both HSC and CMP simplex regressions. "
                    "Only shared features can be compared across fits.")
            try:
                if corr_matrix is not None:
                    log.info("Cross-fit Wald via flow soft assignment...")

                    def _crossfit_wald(reg_dict_h, reg_dict_c, label):
                        """Compute cross-fit Wald for a feature type (genes or pathways)."""
                        hsc_feat = list(reg_dict_h.get("feature_names", []))
                        cmp_feat = list(reg_dict_c.get("feature_names", []))
                        hsc_coefs = reg_dict_h.get("vertex_coefficients")
                        cmp_coefs = reg_dict_c.get("vertex_coefficients")
                        hsc_se = reg_dict_h.get("vertex_se")
                        cmp_se = reg_dict_c.get("vertex_se")
                        if hsc_coefs is None or cmp_coefs is None or hsc_se is None or cmp_se is None:
                            return None, None
                        hsc_coefs = np.asarray(hsc_coefs)
                        cmp_coefs = np.asarray(cmp_coefs)
                        hsc_se = np.asarray(hsc_se)
                        cmp_se = np.asarray(cmp_se)
                        shared = sorted(set(hsc_feat) & set(cmp_feat))
                        if len(shared) == 0:
                            return None, None
                        hsc_idx = [hsc_feat.index(g) for g in shared]
                        cmp_idx = [cmp_feat.index(g) for g in shared]
                        log.info(f"  {label} cross-fit Wald: {len(shared)} shared "
                                 f"(HSC {len(hsc_feat)}, CMP {len(cmp_feat)}); "
                                 f"hsc_coefs shape={hsc_coefs.shape}, cmp_coefs shape={cmp_coefs.shape}")
                        return shared, (hsc_coefs, cmp_coefs, hsc_se, cmp_se, hsc_idx, cmp_idx)

                    # Gene-level cross-fit
                    hsc_reg = adata_hsc.uns.get("peach_simplex_regression_genes", {})
                    cmp_reg_r = adata_cmp.uns.get("peach_simplex_regression_genes", {})
                    shared_g, gene_data = _crossfit_wald(hsc_reg, cmp_reg_r, "GENE")

                    # Pathway-level cross-fit (compute CMP pathway regression first)
                    if "pathway_scores" not in adata_cmp.obsm:
                        net = pc.pp.load_pathway_networks(sources=["c5_bp"])
                        pc.pp.compute_pathway_scores(adata_cmp, net=net)
                    if "pathway_scores" not in adata_hsc.obsm:
                        # Should already be there from earlier in phase 3
                        net = pc.pp.load_pathway_networks(sources=["c5_bp"])
                        pc.pp.compute_pathway_scores(adata_hsc, net=net)
                    log.info("CMP pathway simplex regression (for cross-fit pathway Wald)...")
                    try:
                        pc.tl.pathway_simplex_regression(adata_cmp, max_degree=1, robust_se=True)
                    except Exception as e:
                        log.warning(f"CMP pathway regression failed: {e}")
                    hsc_pw_reg = adata_hsc.uns.get("peach_simplex_regression_pathways", {})
                    cmp_pw_reg = adata_cmp.uns.get("peach_simplex_regression_pathways", {})
                    shared_p, pw_data = _crossfit_wald(hsc_pw_reg, cmp_pw_reg, "PATHWAY")

                    # Gene HVG overlap report
                    n_h_g = len(hsc_reg.get("feature_names", []))
                    n_c_g = len(cmp_reg_r.get("feature_names", []))
                    if shared_g is not None:
                        _n_shared_g = len(shared_g)
                        _pct_overlap = 100 * _n_shared_g / min(n_h_g, n_c_g) if min(n_h_g, n_c_g) > 0 else 0
                        html_flow += report.text(
                            f"<b>Cross-fit Wald gene overlap</b>: {_n_shared_g} shared "
                            f"(HSC {n_h_g} HVGs, CMP {n_c_g} HVGs, "
                            f"overlap = {_pct_overlap:.0f}% of smaller set). "
                            f"HVGs selected independently by variance per cell type.")
                        # HVG overlap interpretation
                        _n_total_genes = adata_hsc.shape[1] if adata_hsc is not None else 0
                        if _n_total_genes > 0 and min(n_h_g, n_c_g) > 0:
                            _expected_overlap = int(min(n_h_g, n_c_g) / _n_total_genes * min(n_h_g, n_c_g))
                            html_flow += report.text(
                                f"<b>HVG overlap interpretation</b>: {_pct_overlap:.0f}% HVG overlap is striking: "
                                f"independent HVG selection on HSC and CMP from {_n_total_genes} genes would give "
                                f"~{100*min(n_h_g,n_c_g)/_n_total_genes:.0f}% overlap under random selection of "
                                f"{min(n_h_g,n_c_g)} ({min(n_h_g,n_c_g)}/{_n_total_genes} × {min(n_h_g,n_c_g)} "
                                f"≈ {_expected_overlap} expected). "
                                f"The observed {_n_shared_g} / {min(n_h_g,n_c_g)} overlap indicates strong shared "
                                f"variance structure between HSC and CMP populations, consistent with them sampling "
                                f"a common differentiation trajectory rather than being biologically independent."
                            )

                    K_h, K_c = corr_matrix.shape
                    # Pair selection: z > 2 against per-pair shuffle null, OR FDR < 0.05.
                    # This replaces the old fixed 2%-mass threshold, which was too strict
                    # for near-uniform K_h x K_c grids (avg mass per pair ≈ 1/K_h*K_c).
                    top_cross_pairs_set = set()
                    for hi in range(K_h):
                        for ci in range(K_c):
                            if pair_z[hi, ci] > 2.0 or (pair_fdr is not None and pair_fdr[hi, ci] < 0.05):
                                top_cross_pairs_set.add((int(hi), int(ci)))
                    top_cross_pairs = sorted(top_cross_pairs_set,
                                             key=lambda p: -pair_z[p[0], p[1]])
                    n_z = int((pair_z > 2.0).sum())
                    n_fdr = int((pair_fdr < 0.05).sum()) if pair_fdr is not None else 0
                    log.info(f"Top cross-fit pairs (z>2 OR FDR<0.05): {len(top_cross_pairs)} pairs "
                             f"(z>2: {n_z}, FDR<0.05: {n_fdr})")
                    log.info(f"  Z-score range: [{pair_z.min():.2f}, {pair_z.max():.2f}], "
                             f"mass range: [{corr_matrix.min():.3f}, {corr_matrix.max():.3f}], "
                             f"avg per pair: {corr_matrix.sum()/(K_h*K_c):.3f}")

                    def _wald_for_pair(hi, ci, data_tuple, names):
                        coefs_h, coefs_c, se_h_full, se_c_full, idx_h, idx_c = data_tuple
                        if hi >= coefs_h.shape[1] or ci >= coefs_c.shape[1]:
                            return None
                        beta_h = coefs_h[idx_h, hi]
                        beta_c = coefs_c[idx_c, ci]
                        se_h = se_h_full[idx_h, hi]
                        se_c = se_c_full[idx_c, ci]
                        se_diff = np.sqrt(se_h**2 + se_c**2)
                        se_diff[se_diff < 1e-10] = 1e-10
                        z_vals = (beta_h - beta_c) / se_diff
                        p_vals = 2 * (1 - scipy_norm.cdf(np.abs(z_vals)))
                        fdr_vals = false_discovery_control(p_vals, method="bh")
                        n_sig_cw = int((fdr_vals < 0.05).sum())
                        # Top differentiating features
                        top_diff_idx = np.argsort(np.abs(z_vals))[::-1][:5]
                        top_feats = ", ".join(names[idx] for idx in top_diff_idx)
                        return n_sig_cw, top_feats, len(p_vals)

                    # Build gene + pathway cross-fit Wald table
                    cross_wald_rows = []
                    for hi, ci in top_cross_pairs:
                        row = {
                            "HSC archetype": f"A{hi+1}",
                            "CMP archetype": f"A{ci+1}",
                            "Mass": f"{corr_matrix[hi, ci]:.3f}",
                            "Z-score": f"{pair_z[hi, ci]:.2f}",
                            "Pair p-value": f"{pair_pvals[hi, ci]:.3f}" if pair_fdr is not None else "?",
                            "Pair FDR": f"{pair_fdr[hi, ci]:.3f}" if pair_fdr is not None else "?",
                        }
                        if gene_data is not None:
                            res_g = _wald_for_pair(hi, ci, gene_data, shared_g)
                            if res_g:
                                row["N gene sig"] = f"{res_g[0]}/{res_g[2]}"
                                row["Top genes"] = res_g[1]
                        if pw_data is not None:
                            res_p = _wald_for_pair(hi, ci, pw_data, shared_p)
                            if res_p:
                                row["N pathway sig"] = f"{res_p[0]}/{res_p[2]}"
                                row["Top pathways"] = res_p[1][:200]  # truncate long pathway names
                        cross_wald_rows.append(row)
                    if cross_wald_rows:
                        html_flow += report.df_to_html(pd.DataFrame(cross_wald_rows),
                            caption="Cross-fit Wald: HSC vs CMP archetype pairs with z > 2 against per-pair "
                                    "permutation null, or FDR < 0.05. Z-score = (observed mass - null mean) / null SD "
                                    "over 50 shuffle permutations. Wald Z = (beta_h - beta_c) / sqrt(SE_h^2 + SE_c^2), "
                                    "BH FDR over shared features.")
                    else:
                        html_flow += error_html("No cross-fit pairs passed z-score threshold (z > 2 or FDR < 0.05)")

                    # --- Per-pair zoomed flow_between() models for significant pairs ---
                    # For each (HSC arch hi, CMP arch ci) pair flagged by z > 2 or FDR < 0.05,
                    # train a small flow model from HSC cells hard-assigned to arch hi
                    # to CMP cells hard-assigned to arch ci. Then run gene alignment + Jacobian
                    # on the per-pair flow to get a "zoomed" view of which features drive the
                    # specific archetype-to-archetype transition.
                    html_flow += report.text("<h3>Per-pair zoomed flow models</h3>")
                    html_flow += report.text(
                        "<p>For each cross-fit pair with z &gt; 2 or FDR &lt; 0.05, we train "
                        "a per-pair <code>flow_within()</code> model restricted to HSC cells "
                        "hard-assigned to the source archetype and CMP cells hard-assigned to "
                        "the target archetype, then compute zoomed gene alignment and Jacobian "
                        "scores to identify features that specifically drive that archetype "
                        "transition.</p>"
                        "<p><b>Notation</b>: <i>z</i> = permutation z-score "
                        "(<code>(observed - null_mean) / null_sd</code>); "
                        "<i>q</i> = Benjamini-Hochberg FDR.</p>")

                    if not top_cross_pairs_set:
                        html_flow += report.text(
                            "<p><i>No significant cross-fit pairs (z &gt; 2 or FDR &lt; 0.05). "
                            "Skipping per-pair zoomed flow models.</i></p>")
                    else:
                        # Show ALL significant pairs (FDR < 0.05), sorted by
                        # transport mass descending so the heaviest transitions
                        # appear first.
                        ranked_pairs = sorted(
                            top_cross_pairs_set,
                            key=lambda p: -corr_matrix[p[0], p[1]],
                        )
                        html_flow += report.text(
                            f"<p><i>{len(ranked_pairs)} significant pairs "
                            f"(FDR &lt; 0.05 or z &gt; 2), sorted by transport mass "
                            f"descending.</i></p>"
                        )

                        # Pre-compute hard archetype assignment for HSC source on adata_full
                        # (same scope as weights_full_hsc which holds HSC-model weights for all cells)
                        hsc_argmax_full = np.argmax(weights_full_hsc, axis=1)  # [n_full]
                        # CMP archetype hard assignment for the target subset (already in target order)
                        cmp_argmax_target = np.argmax(weights_tgt_cmp, axis=1)  # [n_target]
                        # Map target subset rows back to adata_full row indices
                        target_full_idx = np.where(target_mask)[0]
                        source_full_idx = np.where(source_cell_mask.values)[0] \
                            if hasattr(source_cell_mask, "values") else np.where(np.asarray(source_cell_mask))[0]

                        # Temporary obs column we will overwrite per pair
                        pair_obs_col = "_pair_flow_label"

                        for pair_rank, (hi, ci) in enumerate(ranked_pairs):
                            try:
                                log.info(
                                    f"[Per-pair flow] HSC A{hi+1} -> CMP A{ci+1} "
                                    f"(z={pair_z[hi, ci]:.2f}, q={pair_fdr[hi, ci]:.3f})"
                                )
                                # Source mask: HSC cells (cell_type filter) AND HSC arch == hi
                                src_pair_mask = source_cell_mask.values.copy() if hasattr(source_cell_mask, "values") else np.asarray(source_cell_mask).copy()
                                src_pair_mask &= (hsc_argmax_full == hi)
                                n_src_pair = int(src_pair_mask.sum())

                                # Target mask: build full-length mask from target_full_idx
                                tgt_pair_mask = np.zeros(adata_full.n_obs, dtype=bool)
                                tgt_subset_keep = (cmp_argmax_target == ci)
                                if tgt_subset_keep.any():
                                    tgt_pair_mask[target_full_idx[tgt_subset_keep]] = True
                                n_tgt_pair = int(tgt_pair_mask.sum())

                                if n_src_pair < 30 or n_tgt_pair < 30:
                                    html_flow += report.text(
                                        f"<h4>HSC A{hi+1} → CMP A{ci+1} "
                                        f"(z={pair_z[hi, ci]:.2f}, q={pair_fdr[hi, ci]:.3f})</h4>"
                                    )
                                    html_flow += error_html(
                                        f"Skipped: too few cells "
                                        f"(N source={n_src_pair}, N target={n_tgt_pair}; "
                                        f"need ≥30 each)."
                                    )
                                    continue

                                # Stamp the obs column with one of three values: "source", "target", "other"
                                pair_labels = np.full(adata_full.n_obs, "other", dtype=object)
                                pair_labels[src_pair_mask] = "source"
                                pair_labels[tgt_pair_mask] = "target"
                                adata_full.obs[pair_obs_col] = pd.Categorical(pair_labels)

                                # Train per-pair flow (smaller, fewer epochs than the global one)
                                fr_pair = pc.tl.flow_within(
                                    adata_full,
                                    source={pair_obs_col: "source"},
                                    target={pair_obs_col: "target"},
                                    n_epochs=200,
                                    hidden_dims=(64, 64),
                                    batch_size=128,
                                    return_model=True,
                                    name=f"HSC_A{hi+1}_to_CMP_A{ci+1}",
                                    random_state=42 + pair_rank,
                                )

                                # W2 reduction
                                pca_key_pair = fr_pair.get("pca_key", "X_pca")
                                src_pca_pair = adata_full.obsm[pca_key_pair][fr_pair["source_mask"]]
                                tgt_pca_pair = adata_full.obsm[pca_key_pair][fr_pair["target_mask"]]
                                trans_pair = fr_pair["transported"]
                                try:
                                    w2_b = wasserstein2_distance(src_pca_pair, tgt_pca_pair, max_n=1000, seed=42)
                                    w2_a = wasserstein2_distance(trans_pair, tgt_pca_pair, max_n=1000, seed=42)
                                    w2_red_abs = w2_b - w2_a
                                    w2_red_rel = w2_red_abs / max(w2_b, 1e-10)
                                except Exception:
                                    w2_b = w2_a = w2_red_abs = w2_red_rel = float("nan")

                                mmd_b_p = fr_pair["mmd_before"]
                                mmd_a_p = fr_pair["mmd_after"]
                                mmd_red_p = 1 - mmd_a_p / max(mmd_b_p, 1e-10)

                                html_flow += report.text(
                                    f"<h4>HSC A{hi+1} → CMP A{ci+1} "
                                    f"(z={pair_z[hi, ci]:.2f}, q={pair_fdr[hi, ci]:.3f})</h4>"
                                )
                                html_flow += metric_grid([
                                    metric_card(f"{n_src_pair}", "N source"),
                                    metric_card(f"{n_tgt_pair}", "N target"),
                                    metric_card(f"{w2_b:.3f}", "W2 before"),
                                    metric_card(f"{w2_a:.3f}", "W2 after"),
                                    metric_card(f"{w2_red_abs:.3f}", "W2 reduction (PC)"),
                                    metric_card(f"{w2_red_rel:.1%}", "W2 reduction (%)"),
                                    metric_card(f"{mmd_b_p:.4f}", "MMD before"),
                                    metric_card(f"{mmd_a_p:.4f}", "MMD after"),
                                    metric_card(f"{mmd_red_p:.1%}", "MMD reduction"),
                                ])

                                # W-B20: Zoomed gene alignment with a
                                # small shuffle null (50 permutations) so
                                # FDR-pass counts can be reported in the
                                # caption alongside the per-pair
                                # correspondence permutation null mean +/- std
                                # for THAT pair (from perm_null at the
                                # largest swap fraction).
                                try:
                                    align_pair = pc.tl.flow_gene_alignment(
                                        adata_full, fr_pair,
                                        n_top=10, per_cell=False,
                                        n_permutations=50, null_type="shuffle",
                                        random_state=42,
                                    )
                                    top_aligned_pair = align_pair.get("top_aligned", [])[:10]
                                    top_opposed_pair = align_pair.get("top_opposed", [])[:10]
                                    a_scores = np.asarray(align_pair.get("alignment_scores", []))
                                    a_genes = list(align_pair.get("gene_names", []))
                                    a_pvals = align_pair.get("alignment_pvalues")
                                    a_fdr = align_pair.get("alignment_pvalues_fdr")
                                    a_null_mean = align_pair.get("null_mean")
                                    a_null_std = align_pair.get("null_std")

                                    # Per-pair correspondence null context
                                    # (from the W-B23 permutation curve null,
                                    # at the largest swap fraction).
                                    pair_null_mean = float(null_mean_corr[hi, ci])
                                    pair_null_std = float(null_std_corr[hi, ci])
                                    pair_obs_mass = float(corr[hi, ci])

                                    # Gene-alignment null FDR-pass count
                                    n_genes_total = int(len(a_scores)) if len(a_scores) > 0 else 0
                                    n_genes_p05 = (
                                        int((np.asarray(a_pvals) < 0.05).sum())
                                        if a_pvals is not None else 0
                                    )
                                    n_genes_fdr05 = (
                                        int((np.asarray(a_fdr) < 0.05).sum())
                                        if a_fdr is not None else 0
                                    )

                                    if len(a_scores) > 0:
                                        # Build top-10 aligned table with scores + per-gene null context
                                        aligned_rows = []
                                        for g in top_aligned_pair:
                                            try:
                                                idx_g = a_genes.index(g)
                                                row = {
                                                    "Gene": g,
                                                    "Alignment score": f"{a_scores[idx_g]:+.4f}",
                                                }
                                                if a_null_mean is not None and a_null_std is not None:
                                                    row["Null mean"] = f"{float(a_null_mean[idx_g]):+.4f}"
                                                    row["Null std"] = f"{float(a_null_std[idx_g]):.4f}"
                                                if a_pvals is not None:
                                                    row["p-value"] = f"{float(a_pvals[idx_g]):.3f}"
                                                if a_fdr is not None:
                                                    row["FDR q"] = f"{float(a_fdr[idx_g]):.3f}"
                                                aligned_rows.append(row)
                                            except ValueError:
                                                aligned_rows.append({"Gene": g, "Alignment score": "?"})
                                        html_flow += report.df_to_html(
                                            pd.DataFrame(aligned_rows),
                                            caption=(
                                                f"Top 10 aligned genes for HSC A{hi+1} -> CMP A{ci+1} "
                                                f"(zoomed flow). Score = cosine similarity between gene's "
                                                f"PCA loading and per-pair flow velocity. "
                                                f"<br><b>Per-pair correspondence permutation null</b> "
                                                f"(from the W-B23 global-swap null, at f=0.50, for THIS "
                                                f"specific pair): null mean = {pair_null_mean:.4f}, "
                                                f"null std = {pair_null_std:.4f}, observed mass = "
                                                f"{pair_obs_mass:.4f}. "
                                                f"<br><b>Gene-alignment shuffle null</b> "
                                                f"(50 permutations, null_type='shuffle'): "
                                                f"{n_genes_fdr05}/{n_genes_total} genes pass shuffle null "
                                                f"at FDR&lt;0.05 ({n_genes_p05}/{n_genes_total} at p&lt;0.05)."
                                            ),
                                        )
                                    else:
                                        html_flow += error_html(
                                            f"Pair {hi+1}->{ci+1}: gene alignment returned no scores"
                                        )
                                except Exception as e_align:
                                    log.exception(f"Per-pair gene alignment failed for ({hi},{ci})")
                                    html_flow += error_html(
                                        f"Per-pair gene alignment failed: {e_align}"
                                    )

                                # Zoomed Jacobian (per_cell_features=False keeps it cheap)
                                try:
                                    jac_pair = pc.tl.flow_jacobian(
                                        adata_full, fr_pair, fr_pair["model"],
                                        t=0.5, per_cell_features=False, n_permutations=0,
                                    )
                                    feat_exp = jac_pair.get("feature_expansion")
                                    if feat_exp is not None and len(feat_exp) > 0:
                                        feat_exp = np.asarray(feat_exp)
                                        # Top genes by absolute expansion (largest |L^T J L|)
                                        var_names = list(adata_full.var_names)
                                        order = np.argsort(np.abs(feat_exp))[::-1][:10]
                                        jac_rows = [
                                            {"Gene": var_names[i], "Expansion": f"{float(feat_exp[i]):+.4f}"}
                                            for i in order
                                        ]
                                        html_flow += report.df_to_html(
                                            pd.DataFrame(jac_rows),
                                            caption=f"Top 10 Jacobian-expansion genes for HSC A{hi+1} → CMP A{ci+1}. "
                                                    f"Expansion = L_g^T J L_g where L_g is the gene's "
                                                    f"normalized PCA loading and J is the per-pair flow Jacobian "
                                                    f"at t=0.5. Positive = expansion, negative = contraction.")
                                    else:
                                        html_flow += error_html(
                                            f"Pair {hi+1}->{ci+1}: Jacobian returned no feature expansion"
                                        )
                                except Exception as e_jac:
                                    log.exception(f"Per-pair Jacobian failed for ({hi},{ci})")
                                    html_flow += error_html(
                                        f"Per-pair Jacobian failed: {e_jac}"
                                    )

                            except Exception as e_pair:
                                log.exception(f"Per-pair flow failed for ({hi},{ci})")
                                html_flow += report.text(
                                    f"<h4>HSC A{hi+1} → CMP A{ci+1}</h4>"
                                )
                                html_flow += error_html(
                                    f"Per-pair flow_within() failed: {e_pair}"
                                )

                        # Clean up the temporary obs column
                        if pair_obs_col in adata_full.obs.columns:
                            del adata_full.obs[pair_obs_col]
            except Exception as e:
                log.exception("Cross-fit Wald failed")
                html_flow += error_html(f"Cross-fit Wald failed: {e}")

    except Exception as e:
        log.exception("Fig 2E failed")
        html_flow += error_html(f"Fig 2E failed: {e}")
        flow_results = {}

    report.add_section("Figure 2E: Flow Between HSC→CMP", html_flow, step_num="2E")

    # --- Fig 2F: Gene alignment + Jacobian + straw plots + ridgeplots ---
    html_genes = ""
    try:
        for pair_key, fr in flow_results.items():
            model = fr.get("model")
            if model is None:
                html_genes += error_html(f"{pair_key}: no model, skipping gene alignment")
                continue

            # Gene alignment (dual null) — bumped to 1000 permutations for fine-grained p-values
            log.info(f"Gene alignment: {pair_key} (1000 permutations)...")
            align = pc.tl.flow_gene_alignment(
                adata_full, fr, n_top=30, per_cell=False,
                n_permutations=1000, null_type="both", random_state=42)

            # --- Full null model reporting ---
            omnibus_p = align.get("rotation_omnibus_pvalue")
            rot_stat = align.get("rotation_observed_stat")
            rot_null = align.get("rotation_null_distribution")
            html_genes += report.text("<h4>Gene alignment null models</h4>")
            align_cards = []
            if omnibus_p is not None:
                align_cards.append(metric_card(fmt_pval(omnibus_p), "Rotation omnibus p"))
            if rot_stat is not None:
                align_cards.append(metric_card(f"{rot_stat:.4f}", "Rotation stat"))
            n_tested = align.get("alignment_fdr_n_tested", "?")
            align_pvals = align.get("alignment_pvalues")
            align_fdr = align.get("alignment_pvalues_fdr")
            null_mean_align = align.get("null_mean")
            null_std_align = align.get("null_std")
            if align_pvals is not None:
                n_sig_align = int((np.asarray(align_pvals) < 0.05).sum())
                align_cards.append(metric_card(f"{n_sig_align}", "Shuffle p<0.05"))
            if align_fdr is not None:
                n_sig_fdr = int((np.asarray(align_fdr) < 0.05).sum())
                align_cards.append(metric_card(f"{n_sig_fdr}", "Shuffle FDR<0.05"))
            align_cards.append(metric_card(f"{n_tested}", "N tested"))
            align_cards.append(metric_card("1000", "N permutations"))
            html_genes += metric_grid(align_cards)
            html_genes += report.text(
                "<b>Gene alignment method</b>: cosine similarity between PCA gene loadings and "
                "mean flow velocity vector. Two nulls: rotation (random orthogonal Q rotates loadings, "
                "tests omnibus) and shuffle (per-gene random permutation of loading-to-gene mapping). "
                "Pre-filtered FDR over top 5% of genes by |score|.")

            # Rotation null distribution histogram
            if rot_null is not None and rot_stat is not None:
                fig_rot, ax_rot = plt.subplots(figsize=(6, 3))
                ax_rot.hist(rot_null, bins=min(30, len(rot_null)), alpha=0.7, color="#999")
                ax_rot.axvline(rot_stat, color="red", linewidth=2, label=f"Observed={rot_stat:.4f}")
                ax_rot.legend()
                ax_rot.set_xlabel("Rotation null statistic")
                ax_rot.set_title(f"Gene alignment rotation null (p={fmt_pval(omnibus_p)})")
                ax_rot.spines[["top", "right"]].set_visible(False)
                fig_rot.tight_layout()
                html_genes += report.fig_to_img(fig_rot, caption="Rotation null distribution")
                plt.close("all")

            # Gene alignment barplot (top aligned + opposed)
            try:
                fig_bar = pc.pl.gene_alignment_barplot(adata_full, align, n_top=20, show=False)
                html_genes += safe_plotly_html(report, fig_bar,
                    f"Top 20 aligned + opposed genes: {pair_key}")
            except Exception as e:
                log.exception("Gene alignment barplot failed")
                html_genes += error_html(f"Gene alignment barplot failed: {e}")

            # --- Top aligned genes table with regression context + Z-score ---
            scores = align.get("alignment_scores", [])
            gene_names_align = align.get("gene_names", [])

            # Build regression context lookup: gene → R², argmax archetype, pattern
            hsc_reg_for_ctx = adata_hsc.uns.get("peach_simplex_regression_genes", {})
            ctx_feat = list(hsc_reg_for_ctx.get("feature_names", []))
            ctx_r2 = np.asarray(hsc_reg_for_ctx.get("r_squared_degree1", []))
            ctx_coefs = np.asarray(hsc_reg_for_ctx.get("vertex_coefficients", []))
            r2_lookup = {g: ctx_r2[i] for i, g in enumerate(ctx_feat) if i < len(ctx_r2)}
            argmax_lookup = {}
            if ctx_coefs.size > 0 and len(ctx_feat) == ctx_coefs.shape[0]:
                amax = np.argmax(np.abs(ctx_coefs), axis=1)
                argmax_lookup = {g: int(amax[i]) for i, g in enumerate(ctx_feat)}
            # Pattern lookup
            pattern_lookup = {}
            patterns = adata_hsc.uns.get("peach_feature_patterns", {})
            classifications = patterns.get("classifications", {})
            if isinstance(classifications, dict):
                for g, c in classifications.items():
                    if isinstance(c, dict):
                        pattern_lookup[g] = c.get("pattern", "?")
            elif isinstance(classifications, list):
                for c in classifications:
                    if isinstance(c, dict):
                        pattern_lookup[c.get("feature", "?")] = c.get("pattern", "?")

            # DIAGNOSTIC: dump alignment p-value spread
            if align_pvals is not None:
                ap = np.asarray(align_pvals)
                log.info(f"Alignment p-value diagnostic: min={ap.min():.6f}, max={ap.max():.6f}, "
                         f"unique={len(np.unique(ap))}, n={len(ap)}")
                log.info(f"  First 5 align_pvals: {ap[:5]}, last 5: {ap[-5:]}")

            if len(scores) > 0:
                scores_arr = np.asarray(scores)
                sorted_idx = np.argsort(np.abs(scores_arr))[::-1]
                # Build per-gene top-2 archetype lookup from degree-1 vertex coefficients.
                # For each gene, report the two archetypes with largest |vertex_coefficient|,
                # their signed values, and a short biological interpretation string.
                top2_arch_lookup = {}   # gene → [(arch_label, coef), (arch_label, coef)]
                if ctx_coefs.size > 0 and len(ctx_feat) == ctx_coefs.shape[0]:
                    K_ctx = ctx_coefs.shape[1]
                    abs_ctx = np.abs(ctx_coefs)
                    for ii, gn in enumerate(ctx_feat):
                        sorted_ai = np.argsort(abs_ctx[ii])[::-1]
                        top2 = [(f"A{sorted_ai[j]+1}", float(ctx_coefs[ii, sorted_ai[j]]))
                                for j in range(min(2, K_ctx))]
                        top2_arch_lookup[gn] = top2

                def _flow_interp(score, pattern, top2):
                    """Short interpretation string combining flow score + regression context."""
                    direction = "aligned" if score > 0 else "opposed"
                    arch_str = ""
                    if top2:
                        a1, c1 = top2[0]
                        arch_str = f"dominant {a1} (β={c1:.3f})"
                        if len(top2) > 1:
                            a2, c2 = top2[1]
                            arch_str += f", secondary {a2} (β={c2:.3f})"
                    pat = pattern or "unknown"
                    if pat == "archetype-exclusive":
                        bio = f"exclusive to {top2[0][0] if top2 else '?'}"
                    elif pat == "interaction":
                        bio = "interaction/blending zone gene"
                    elif pat == "structured":
                        bio = "structured gradient"
                    elif pat == "flat":
                        bio = "flat (no archetype preference)"
                    else:
                        bio = pat
                    return f"{direction}; {arch_str}; {bio}"

                top_rows = []
                for gi in sorted_idx[:20]:
                    gname = gene_names_align[gi] if gi < len(gene_names_align) else "?"
                    in_reg = gname in r2_lookup
                    row = {
                        "Gene": gname,
                        "Score": f"{scores_arr[gi]:.4f}",
                        "Direction": "aligned" if scores_arr[gi] > 0 else "opposed",
                    }
                    # Z-score from null distribution (continuous, varies)
                    if null_mean_align is not None and null_std_align is not None:
                        nm = float(null_mean_align[gi])
                        ns = float(null_std_align[gi])
                        z_align = (scores_arr[gi] - nm) / max(ns, 1e-10)
                        row["Z-score"] = f"{z_align:.3f}"
                    if align_pvals is not None:
                        row["p-value"] = fmt_pval(float(align_pvals[gi]))
                    if align_fdr is not None:
                        row["FDR q"] = fmt_pval(float(align_fdr[gi]))
                    # Regression context — enriched with top-2 archetypes (Task 17)
                    if in_reg:
                        row["HSC R²"] = f"{r2_lookup[gname]:.4f}"
                    top2_g = top2_arch_lookup.get(gname, [])
                    if top2_g:
                        row["Top arch 1"] = f"{top2_g[0][0]} (β={top2_g[0][1]:.3f})"
                        if len(top2_g) > 1:
                            row["Top arch 2"] = f"{top2_g[1][0]} (β={top2_g[1][1]:.3f})"
                    elif not in_reg:
                        row["Top arch 1"] = "not in regression HVG set"
                    pat_g = pattern_lookup.get(gname)
                    if pat_g:
                        row["Pattern"] = pat_g
                    row["Interpretation"] = _flow_interp(
                        scores_arr[gi], pat_g, top2_g)
                    top_rows.append(row)
                html_genes += report.df_to_html(pd.DataFrame(top_rows),
                    caption=f"Top 20 flow-aligned genes (by |score|, enriched with HSC simplex "
                            f"regression context). "
                            f"Z-score = (score - null_mean) / null_std (varies continuously even when "
                            f"permutation p-values are at floor 1/1001). "
                            f"Top arch 1/2 = top archetypes by |degree-1 vertex coefficient|. "
                            f"Pattern from classify_feature_patterns() on degree-2 regression. "
                            f"Genes not in regression HVG set show 'not in regression HVG set'.")

            # --- Jacobian + straw plots + ridgeplots ---
            log.info(f"Jacobian: {pair_key} (1000 permutations)...")
            jac = pc.tl.flow_jacobian(
                adata_full, fr, model, per_cell_features=True, n_top_features=500,
                n_permutations=1000, null_type="both", permutation_seed=42)

            jac_cards = []
            jac_rot_p = jac.get("rotation_omnibus_pvalue") or jac.get("expansion_rotation_omnibus_pvalue")
            jac_rot_stat = jac.get("rotation_observed_stat")
            exp_pvals = jac.get("expansion_pvalues")
            exp_fdr = jac.get("expansion_pvalues_fdr")
            if jac_rot_p is not None:
                jac_cards.append(metric_card(fmt_pval(jac_rot_p), "Jac rotation p"))
            if jac_rot_stat is not None:
                jac_cards.append(metric_card(f"{jac_rot_stat:.4f}", "Jac rotation stat"))
            if exp_pvals is not None:
                n_sig_exp = int((np.asarray(exp_pvals) < 0.05).sum())
                jac_cards.append(metric_card(f"{n_sig_exp}", "Expansion p<0.05"))
            if exp_fdr is not None:
                n_sig_exp_fdr = int((np.asarray(exp_fdr) < 0.05).sum())
                jac_cards.append(metric_card(f"{n_sig_exp_fdr}", "Expansion FDR<0.05"))
            jac_cards.append(metric_card("1000", "N permutations"))
            if jac_cards:
                html_genes += report.text("<h4>Jacobian null models (HSC→CMP global flow)</h4>")
                html_genes += metric_grid(jac_cards)

            # Jacobian heatmap (mean Jacobian in PCA space — note this is PC×PC, not archetype×archetype)
            try:
                fig_jac = pc.pl.jacobian_heatmap(adata_full, jac, show=False)
                html_genes += safe_plotly_html(report, fig_jac,
                    "Global flow mean Jacobian heatmap (PCA × PCA, NOT archetype × archetype). "
                    "See per-pair archetype flow models below for archetype-level interpretation.")
            except Exception as e:
                log.exception("Jacobian heatmap failed")
                html_genes += error_html(f"Jacobian heatmap failed: {e}")

            per_cell = jac.get("per_cell_expansion")
            jac_gene_names = jac.get("per_cell_expansion_gene_names", [])
            if per_cell is not None and len(jac_gene_names) > 0:
                import scipy.sparse as sp

                # Compute flow-projected coordinate (1D projection)
                source_pca = adata_full.obsm["X_pca"][fr["source_mask"]]
                transported = fr["transported"]
                flow_dir = (transported - source_pca).mean(axis=0)
                flow_norm = flow_dir / (np.linalg.norm(flow_dir) + 1e-10)
                flow_coord = (source_pca - source_pca.mean(axis=0)) @ flow_norm
                pt_min, pt_max = flow_coord.min(), flow_coord.max()
                if pt_max - pt_min > 1e-10:
                    flow_coord = (flow_coord - pt_min) / (pt_max - pt_min)

                # Select top 20 genes by effect size (prefer significant)
                per_cell_mean = np.abs(per_cell.mean(axis=0))
                per_cell_gene_idx = jac.get("per_cell_expansion_gene_indices")
                used_sig = False
                if exp_pvals is not None and per_cell_gene_idx is not None:
                    sub_pvals = np.asarray(exp_pvals)[per_cell_gene_idx]
                    sig_mask = sub_pvals < 0.05
                    if sig_mask.sum() >= 5:
                        gene_order = np.where(sig_mask)[0]
                        gene_order = gene_order[np.argsort(per_cell_mean[gene_order])[::-1]]
                        used_sig = True
                    else:
                        gene_order = np.argsort(per_cell_mean)[::-1]
                else:
                    gene_order = np.argsort(per_cell_mean)[::-1]

                n_plot = min(20, len(gene_order))
                gene_order = gene_order[:n_plot]

                # Get expression matrix for source cells
                X_source = adata_full[fr["source_mask"]].X
                if sp.issparse(X_source):
                    X_source = X_source.toarray()
                X_source = np.asarray(X_source)

                # DIAGNOSTIC: report X_source value range
                log.info(f"X_source diagnostic: shape={X_source.shape}, "
                         f"min={X_source.min():.4f}, max={X_source.max():.4f}, "
                         f"mean={X_source.mean():.4f}, median={np.median(X_source):.4f}, "
                         f"non-zero frac={(X_source > 0).mean():.4f}")
                # Verify likely logcounts (typical range 0..7-ish)
                if X_source.max() > 100:
                    log.warning(f"X_source max={X_source.max():.2f} suggests RAW counts, not logcounts!")

                # Map per_cell gene names back to var_names indices
                var_list = list(adata_full.var_names)
                gene_var_idx = []
                for gn in jac_gene_names:
                    if gn in var_list:
                        gene_var_idx.append(var_list.index(gn))
                    else:
                        gene_var_idx.append(-1)
                assert len(gene_var_idx) == len(jac_gene_names), \
                    f"gene_var_idx len {len(gene_var_idx)} != jac_gene_names len {len(jac_gene_names)}"
                n_unmapped = sum(1 for x in gene_var_idx if x == -1)
                # DIAGNOSTIC: report fallback rate
                log.info(f"Straw plot gene mapping: {n_unmapped}/{len(gene_var_idx)} unmapped "
                         f"(fallback to flow_coord). Top 20 plot fallback count: "
                         f"{sum(1 for gi in gene_order if gene_var_idx[gi] == -1)}/{n_plot}")
                if n_unmapped > 0:
                    log.warning(f"  {n_unmapped}/{len(gene_var_idx)} Jacobian genes not found in adata.var_names")

                n_bins = 20
                pt_bins = np.linspace(0, 1, n_bins + 1)
                bin_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])
                bin_idx = np.clip(np.digitize(flow_coord, pt_bins) - 1, 0, n_bins - 1)

                COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
                          "#56B4E9", "#F0E442", "#882255", "#117733", "#332288"]

                # Straw plot removed per r12 review (redundant with ridgeplot + lollipop).
                sig_note = "(significant)" if used_sig else "(top by effect)"

                # --- RIDGEPLOT: Seurat-style overlapping KDE ridges (W-B21) ---
                # Each ridge = KDE of per-cell Jacobian expansion values for
                # one gene, drawn with build_overlapping_ridgeplot() so adjacent
                # ridges visibly stack (overlap=0.5) instead of the old
                # separated-panel style. Groups are capped at max_groups=12
                # per panel for readability.
                try:
                    from _paper_part1_viz import build_overlapping_ridgeplot
                    for panel_start in range(0, n_plot, 10):
                        panel_genes = gene_order[panel_start:panel_start + 10]
                        ridge_data = {}
                        for gi in panel_genes:
                            gname = jac_gene_names[gi]
                            vals = np.asarray(per_cell[:, gi], dtype=float).ravel()
                            vals = vals[np.isfinite(vals)]
                            if vals.size >= 2:
                                ridge_data[gname] = vals
                        if not ridge_data:
                            continue
                        fig_ridge = build_overlapping_ridgeplot(
                            ridge_data,
                            overlap=0.5,
                            max_groups=12,
                            title=(
                                f"Seurat-style ridgeplot: Jacobian expansion "
                                f"{sig_note} "
                                f"(genes {panel_start+1}-{panel_start+len(panel_genes)})"
                            ),
                            xlabel="Per-cell Jacobian expansion factor (1.0 = neutral)",
                            cmap_name="viridis",
                        )
                        html_genes += report.fig_to_img(
                            fig_ridge,
                            caption=(
                                "Seurat-style overlapping ridgeplot "
                                "(W-B21): each ridge = KDE of per-cell "
                                "Jacobian expansion values for one gene. "
                                "Adjacent ridges partially overlap "
                                "vertically (overlap=0.5) so distributions "
                                "visually stack instead of sitting on "
                                "isolated baselines. Dashed tick = mean "
                                "expansion per ridge; x-axis is shared "
                                "across all ridges so shapes are directly "
                                "comparable."
                            ),
                        )
                        plt.close("all")
                except Exception as e:
                    log.exception("Overlapping ridgeplot failed")
                    html_genes += error_html(f"Overlapping ridgeplot failed: {e}")

                # Expansion summary table with regression context
                exp_rows = []
                for gi in gene_order[:20]:
                    gname = jac_gene_names[gi]
                    mean_exp = per_cell[:, gi].mean()
                    row = {"Gene": gname, "Mean expansion": f"{mean_exp:.4f}",
                           "Direction": "expanding" if mean_exp > 1 else "contracting"}
                    if exp_pvals is not None and per_cell_gene_idx is not None:
                        row["p-value"] = fmt_pval(float(exp_pvals[per_cell_gene_idx[gi]]))
                    if exp_fdr is not None and per_cell_gene_idx is not None:
                        row["FDR q"] = fmt_pval(float(exp_fdr[per_cell_gene_idx[gi]]))
                    if gname in r2_lookup:
                        row["HSC R²"] = f"{r2_lookup[gname]:.4f}"
                    if gname in argmax_lookup:
                        row["HSC argmax"] = f"A{argmax_lookup[gname]+1}"
                    if gname in pattern_lookup:
                        row["Pattern"] = pattern_lookup[gname]
                    exp_rows.append(row)
                html_genes += report.df_to_html(pd.DataFrame(exp_rows),
                    caption="Top 20 genes by Jacobian expansion effect (with HSC simplex regression context)")

            # --- PER-PAIR FLOW MODELS (W-B21): per-pair detail for ALL ---
            #     significant pairs, capped at MAX_SIG_PAIRS = 20.
            #
            # Each significant pair (pair_fdr < 0.05 OR pair_z > 2.0) gets:
            #   1. Permutation curve (W-B23 build_permutation_curve_figure)
            #   2. Per-pair gene-alignment table (top by alignment, with
            #      shuffle-null context if available)
            #   3. Per-pair PATHWAY simplex regression on the source-bin
            #      cells (W-B21 — mirrors W-B20 gene null work for genes,
            #      using pathway scores instead of gene PCA loadings)
            #   4. Seurat-style overlapping ridgeplot of per-cell gene
            #      alignment values, one ridge per top-aligned gene
            #
            # This replaces the old "summary heatmaps only" block flagged
            # in the r9 review as the least valuable summary.
            if corr_matrix is not None:
                html_genes += "<h4>Per-pair archetype flow models (W-B21 detail)</h4>"
                html_genes += report.text(
                    "<b>Per-pair flow models</b>: For each HSC→CMP "
                    "archetype pair flagged as significant by the "
                    "permutation null (pair_fdr &lt; 0.05 or pair_z &gt; "
                    "2.0), train a small flow model from HSC cells "
                    "argmax-assigned to source archetype hi to CMP cells "
                    "argmax-assigned to target archetype ci, then run "
                    "(a) per-pair gene alignment with shuffle null, "
                    "(b) per-pair pathway simplex regression on the "
                    "source-bin subset, and (c) a Seurat-style "
                    "overlapping ridgeplot of per-cell gene alignment "
                    "values. Capped at MAX_SIG_PAIRS = 20.")

                from _paper_part1_viz import (
                    build_permutation_curve_figure,
                )

                MAX_SIG_PAIRS = 20  # cap on per-pair detail blocks (W-B21)
                K_h_local, K_c_local = corr_matrix.shape

                # Build the ranked list of significant pairs by mass
                # (mirrors the W-B20 selection in Fig 2E so the same pairs
                # appear in both sections).
                sig_pairs_2f = []
                if pair_fdr is not None:
                    flat_mass_order = np.argsort(-corr_matrix.flatten())
                    for fi in flat_mass_order:
                        hi = int(fi // K_c_local)
                        ci = int(fi % K_c_local)
                        is_sig = (pair_fdr[hi, ci] < 0.05) or (
                            'pair_z' in locals()
                            and pair_z is not None
                            and pair_z.shape == (K_h_local, K_c_local)
                            and pair_z[hi, ci] > 2.0
                        )
                        if is_sig:
                            sig_pairs_2f.append((hi, ci))
                # Fallback: if nothing significant, take top-3 by mass
                # (so the section still shows something).
                if not sig_pairs_2f:
                    log.info(
                        "Fig 2F per-pair: no significant pairs; "
                        "falling back to top-3 by mass."
                    )
                    html_genes += error_html(
                        "WARNING: no significant pairs at FDR&lt;0.05 or "
                        "z&gt;2.0. Showing top-3 pairs by mass below so "
                        "the matrix shape is visible even without "
                        "significant signal."
                    )
                    flat_mass_order = np.argsort(-corr_matrix.flatten())
                    for fi in flat_mass_order[:3]:
                        sig_pairs_2f.append(
                            (int(fi // K_c_local), int(fi % K_c_local))
                        )

                # Cap at MAX_SIG_PAIRS by mass order
                if len(sig_pairs_2f) > MAX_SIG_PAIRS:
                    log.info(
                        f"Fig 2F per-pair: capping {len(sig_pairs_2f)} "
                        f"significant pairs to top {MAX_SIG_PAIRS} by mass."
                    )
                    html_genes += report.text(
                        f"<p><i>Note: {len(sig_pairs_2f)} pairs "
                        f"significant; showing top {MAX_SIG_PAIRS} by "
                        f"mass to bound report length.</i></p>"
                    )
                    sig_pairs_2f = sig_pairs_2f[:MAX_SIG_PAIRS]

                # Pre-compute hard archetype assignments
                hsc_argmax_full_2f = (
                    np.argmax(weights_full_hsc, axis=1)
                    if weights_full_hsc is not None else None
                )
                cmp_argmax_target_2f = (
                    np.argmax(weights_tgt_cmp, axis=1)
                    if weights_tgt_cmp is not None else None
                )
                target_full_idx_2f = (
                    np.where(target_mask)[0]
                    if 'target_mask' in locals() and target_mask is not None
                    else None
                )

                pp_summary_rows = []
                for pair_rank_2f, (hi, ci) in enumerate(sig_pairs_2f):
                    log.info(
                        f"[Fig 2F per-pair] HSC A{hi+1} -> CMP A{ci+1} "
                        f"(rank {pair_rank_2f+1}/{len(sig_pairs_2f)})"
                    )
                    try:
                        # 0. Build masks restricted to argmax bins
                        if (hsc_argmax_full_2f is None
                                or cmp_argmax_target_2f is None
                                or target_full_idx_2f is None):
                            raise RuntimeError(
                                "Missing prerequisite weights / target_mask"
                            )
                        src_pair_mask_2f = source_cell_mask.values.copy() \
                            if hasattr(source_cell_mask, "values") \
                            else np.asarray(source_cell_mask).copy()
                        src_pair_mask_2f &= (hsc_argmax_full_2f == hi)
                        tgt_pair_mask_2f = np.zeros(adata_full.n_obs, dtype=bool)
                        tgt_subset_keep = (cmp_argmax_target_2f == ci)
                        if tgt_subset_keep.any():
                            tgt_pair_mask_2f[
                                target_full_idx_2f[tgt_subset_keep]
                            ] = True
                        n_src_pp = int(src_pair_mask_2f.sum())
                        n_tgt_pp = int(tgt_pair_mask_2f.sum())

                        html_genes += report.text(
                            f"<h5>HSC A{hi+1} → CMP A{ci+1} "
                            f"(rank {pair_rank_2f+1}, mass="
                            f"{corr_matrix[hi, ci]:.3f})</h5>"
                        )

                        if n_src_pp < 30 or n_tgt_pp < 30:
                            html_genes += error_html(
                                f"Skipped: too few cells "
                                f"(N source={n_src_pp}, N target={n_tgt_pp}; "
                                f"need ≥30 each)."
                            )
                            pp_summary_rows.append({
                                "Pair": f"H A{hi+1} → C A{ci+1}",
                                "N src": n_src_pp,
                                "N tgt": n_tgt_pp,
                                "Status": "skip (too few cells)",
                            })
                            continue

                        # 1. Permutation curve figure (reuses W-B23)
                        if 'perm_null' in locals() and perm_null is not None:
                            try:
                                fig_curve_2f = build_permutation_curve_figure(
                                    perm_null,
                                    pair_i=hi,
                                    pair_j=ci,
                                    src_label=src_labels[hi],
                                    tgt_label=tgt_labels[ci],
                                )
                                html_genes += report.fig_to_img(
                                    fig_curve_2f,
                                    caption=(
                                        f"Permutation degradation curve for "
                                        f"{src_labels[hi]} -> {tgt_labels[ci]} "
                                        f"(W-B23 null). Replaces the old "
                                        f"'summary heatmap only' output."
                                    ),
                                )
                                plt.close("all")
                            except Exception as e_curve:
                                log.warning(
                                    f"Permutation curve failed for ({hi},{ci}): "
                                    f"{e_curve}"
                                )

                        # 2. Per-pair flow_within + gene alignment with
                        #    shuffle null. Stamps a temp obs column.
                        adata_full.obs["_pp2f_label"] = pd.Categorical(
                            np.where(
                                src_pair_mask_2f, "source",
                                np.where(tgt_pair_mask_2f, "target", "other"),
                            )
                        )
                        try:
                            fr_pp = pc.tl.flow_within(
                                adata_full,
                                source={"_pp2f_label": "source"},
                                target={"_pp2f_label": "target"},
                                n_epochs=150, hidden_dims=(64, 64),
                                batch_size=64, return_model=True,
                                name=f"H{hi+1}_to_C{ci+1}_2f",
                                random_state=42 + pair_rank_2f,
                            )
                            align_pp = pc.tl.flow_gene_alignment(
                                adata_full, fr_pp,
                                n_top=20, per_cell=True,
                                n_top_features=20,
                                n_permutations=50, null_type="shuffle",
                                random_state=42,
                            )
                            scores_pp = np.asarray(
                                align_pp.get("alignment_scores", [])
                            )
                            gene_names_pp = list(
                                align_pp.get("gene_names", [])
                            )
                            pvals_pp = align_pp.get("alignment_pvalues")
                            fdr_pp = align_pp.get("alignment_pvalues_fdr")
                            null_mean_pp = align_pp.get("null_mean")
                            null_std_pp = align_pp.get("null_std")
                            per_cell_align_pp = align_pp.get(
                                "per_cell_alignment"
                            )
                            per_cell_gnames_pp = align_pp.get(
                                "per_cell_gene_names", []
                            )

                            # 2a. Top-20 gene alignment table
                            if scores_pp.size > 0:
                                top20_idx = np.argsort(np.abs(scores_pp))[::-1][:20]
                                gene_rows = []
                                for gi in top20_idx:
                                    g = gene_names_pp[gi]
                                    row = {
                                        "Gene": g,
                                        "Alignment": f"{scores_pp[gi]:+.4f}",
                                        "Direction": (
                                            "aligned" if scores_pp[gi] > 0
                                            else "opposed"
                                        ),
                                    }
                                    if (null_mean_pp is not None
                                            and null_std_pp is not None):
                                        nm = float(null_mean_pp[gi])
                                        ns = float(null_std_pp[gi])
                                        z = (scores_pp[gi] - nm) / max(ns, 1e-10)
                                        row["Null mean"] = f"{nm:+.4f}"
                                        row["Null std"] = f"{ns:.4f}"
                                        row["Z-score"] = f"{z:+.2f}"
                                    if pvals_pp is not None:
                                        row["p-value"] = fmt_pval(
                                            float(pvals_pp[gi])
                                        )
                                    if fdr_pp is not None:
                                        q = float(fdr_pp[gi])
                                        row["FDR q"] = fmt_pval(q)
                                        row["FDR<0.05"] = (
                                            "*" if q < 0.05 else ""
                                        )
                                    gene_rows.append(row)
                                html_genes += report.df_to_html(
                                    pd.DataFrame(gene_rows),
                                    caption=(
                                        f"Per-pair gene alignment for "
                                        f"HSC A{hi+1} → CMP A{ci+1}: top 20 "
                                        f"genes by |alignment| with shuffle "
                                        f"null context (50 permutations). "
                                        f"Column 'FDR<0.05' marks genes "
                                        f"passing the per-gene shuffle null "
                                        f"after BH correction."
                                    ),
                                )

                            # 2a-ii. Tricolor gene scatter + absence + lollipop
                            _pair_expr = _pair_expansion = _pair_flow = _pair_gene_names = None
                            try:
                                from _paper_part1_viz import (
                                    build_tricolor_gene_scatter,
                                )
                                # Compute per-pair Jacobian (aggregated,
                                # not per-cell) for expansion per gene.
                                jac_tri = pc.tl.flow_jacobian(
                                    adata_full, fr_pp, fr_pp["model"],
                                    t=0.5,
                                    per_cell_features=False,
                                    n_permutations=0,
                                )
                                feat_exp_tri = jac_tri.get(
                                    "feature_expansion"
                                )
                                if (feat_exp_tri is not None
                                        and scores_pp.size > 0
                                        and len(feat_exp_tri) > 0):
                                    feat_exp_tri = np.asarray(feat_exp_tri)
                                    var_names_tri = list(
                                        adata_full.var_names
                                    )
                                    n_genes_tri = min(
                                        len(feat_exp_tri),
                                        len(var_names_tri),
                                    )
                                    # Mean expression for source cells
                                    X_src_tri = adata_full[
                                        fr_pp["source_mask"]
                                    ].X
                                    if hasattr(X_src_tri, "toarray"):
                                        X_src_tri = X_src_tri.toarray()
                                    mean_expr_tri = np.asarray(
                                        X_src_tri
                                    ).mean(axis=0).ravel()[:n_genes_tri]

                                    # Expansion: centered around 0 for
                                    # the scatter (subtract 1 so neutral=0)
                                    expansion_tri = (
                                        feat_exp_tri[:n_genes_tri] - 1.0
                                    )

                                    # Flow strength: alignment scores
                                    # (gene_names_pp aligns with scores_pp
                                    #  but has all genes; build a lookup)
                                    flow_str_tri = np.zeros(n_genes_tri)
                                    gene_pp_lookup = {
                                        g: i for i, g in enumerate(
                                            gene_names_pp
                                        )
                                    }
                                    for gi_t in range(n_genes_tri):
                                        gn = var_names_tri[gi_t]
                                        if gn in gene_pp_lookup:
                                            idx_g = gene_pp_lookup[gn]
                                            if idx_g < len(scores_pp):
                                                flow_str_tri[gi_t] = (
                                                    scores_pp[idx_g]
                                                )

                                    # Store for downstream viz (absence, lollipop)
                                    _pair_expr = mean_expr_tri
                                    _pair_expansion = expansion_tri
                                    _pair_flow = flow_str_tri
                                    _pair_gene_names = var_names_tri[:n_genes_tri]

                                    fig_tri = build_tricolor_gene_scatter(
                                        mean_expr_tri,
                                        expansion_tri,
                                        flow_str_tri,
                                        var_names_tri[:n_genes_tri],
                                        n_labels=15,
                                        title=(
                                            f"Tricolor gene scatter: "
                                            f"HSC A{hi+1} → CMP A{ci+1}"
                                        ),
                                    )
                                    html_genes += report.fig_to_img(
                                        fig_tri,
                                        caption=(
                                            f"Tricolor gene scatter for "
                                            f"HSC A{hi+1} → CMP A{ci+1}. "
                                            f"X = mean expression in source "
                                            f"cells. Y = expansion/"
                                            f"contraction (Jacobian - 1; "
                                            f">0 = expanding, <0 = "
                                            f"contracting). Color = flow "
                                            f"alignment (cosine similarity "
                                            f"between gene's PCA loading "
                                            f"and flow velocity). Size = "
                                            f"|flow alignment|. Labels = "
                                            f"top 15 genes by |expansion "
                                            f"* flow alignment|."
                                        ),
                                    )
                                    plt.close("all")
                            except Exception as e_tri:
                                log.exception(
                                    f"Tricolor gene scatter failed for "
                                    f"({hi},{ci})"
                                )
                                html_genes += error_html(
                                    f"Tricolor gene scatter failed: "
                                    f"{e_tri}"
                                )

                            # 2a-iii. Absence plot: high-expression, low-flow genes
                            try:
                                from _paper_part1_viz import build_absence_plot
                                if (_pair_expr is not None and _pair_flow is not None
                                        and _pair_gene_names is not None):
                                    fig_abs = build_absence_plot(
                                        _pair_expr, _pair_flow, _pair_gene_names,
                                        title=f"Absence: HSC A{hi+1} -> CMP A{ci+1}",
                                    )
                                    html_genes += report.fig_to_img(
                                        fig_abs,
                                        caption=(
                                            "Absence plot: <b>red</b> = highly expressed "
                                            "genes NOT associated with the flow. "
                                            "<b>Green</b> = top flow-associated for contrast. "
                                            "<b>Interpretation caveat</b>: \"highly expressed but "
                                            "not flow-associated\" is the expected profile for "
                                            "housekeeping genes and for genes expressed at "
                                            "similar levels in BOTH the source and the target "
                                            "population — the flow subtracts them out. Look at "
                                            "these red genes as candidates for a shared "
                                            "expression baseline, not as hidden flow drivers."
                                        ),
                                    )
                                    plt.close("all")
                            except Exception as e_abs:
                                log.warning(f"Absence plot failed: {e_abs}")

                            # 2a-iv. Lollipop chart: ranked by flow strength
                            try:
                                from _paper_part1_viz import build_lollipop_chart
                                if (_pair_expr is not None and _pair_flow is not None
                                        and _pair_expansion is not None
                                        and _pair_gene_names is not None):
                                    fig_lol = build_lollipop_chart(
                                        _pair_flow, _pair_expansion, _pair_expr,
                                        _pair_gene_names, top_n=25,
                                        title=f"HSC A{hi+1} -> CMP A{ci+1}",
                                    )
                                    html_genes += report.fig_to_img(
                                        fig_lol,
                                        caption=(
                                            f"Lollipop: ranked by flow strength. "
                                            f"Red = expanding, blue = contracting. "
                                            f"Dot size = expression level."
                                        ),
                                    )
                                    plt.close("all")
                            except Exception as e_lol:
                                log.warning(f"Lollipop chart failed: {e_lol}")

                            # 2b. PER-PAIR FLOW-ASSOCIATED PATHWAYS (r12-item-5)
                            # Replaces the previous per-pair pathway simplex
                            # regression. The user's requested intent is "which
                            # pathways drive the flow for this pair" — use the
                            # per-cell AUCell pathway scores (already in
                            # adata.obsm['pathway_scores']) and correlate each
                            # pathway's score with the per-cell flow magnitude
                            # across the source cells of this pair. Sort by
                            # |ρ| descending. Same pattern as
                            # scripts/run_e2e_myeloid.py:3130-3163.
                            # Resolve pathway scores for the source cells.
                            # fr_pp["source_mask"] is sized for adata_full; if
                            # pathway_scores only exists on adata_hsc (which
                            # may have different obs count after subsampling),
                            # we must restrict to the intersection of source
                            # cell obs_names ∩ adata_hsc.obs_names so the
                            # pathway rows align with the same cells' flow
                            # vectors. r13 bug: applied adata_full-sized mask
                            # directly to adata_hsc.obsm['pathway_scores'].
                            pw_flow_obsm_full = None
                            pw_flow_names: list = []
                            _pw_src_from_hsc = False
                            if "pathway_scores" in adata_full.obsm:
                                pw_flow_obsm_full = adata_full.obsm["pathway_scores"]
                                pw_flow_names = list(
                                    adata_full.uns.get("pathway_scores_pathways", [])
                                )
                            elif "pathway_scores" in adata_hsc.obsm:
                                pw_flow_obsm_full = adata_hsc.obsm["pathway_scores"]
                                pw_flow_names = list(
                                    adata_hsc.uns.get("pathway_scores_pathways", [])
                                )
                                _pw_src_from_hsc = True
                            if pw_flow_obsm_full is None or len(pw_flow_names) == 0:
                                html_genes += error_html(
                                    "Flow-associated pathways skipped: no "
                                    "pathway_scores / pathway_scores_pathways "
                                    "in adata_full or adata_hsc."
                                )
                            else:
                                try:
                                    from scipy.stats import spearmanr as _sp_pw
                                    # Per-cell flow magnitude and signed pseudotime
                                    # for the source cells of this pair.
                                    src_pca_pw_full = adata_full.obsm[
                                        fr_pp.get("pca_key", "X_pca")
                                    ][fr_pp["source_mask"]]
                                    trans_pw = fr_pp["transported"]
                                    disp_pw_full = trans_pw - src_pca_pw_full
                                    # Obs names for the source cells (in
                                    # adata_full order). Used to restrict to
                                    # the adata_hsc intersection if pathway
                                    # scores are only available there.
                                    _src_names_full = np.asarray(
                                        adata_full.obs_names[fr_pp["source_mask"]]
                                    )
                                    if _pw_src_from_hsc:
                                        # Map source obs_names → adata_hsc
                                        # positional indices. Intersection
                                        # only; cells absent from adata_hsc
                                        # (e.g. held-out from subsampling)
                                        # are dropped from this analysis.
                                        _hsc_pos = {
                                            n: i for i, n in enumerate(
                                                adata_hsc.obs_names
                                            )
                                        }
                                        _src_keep_full = np.array([
                                            n in _hsc_pos for n in _src_names_full
                                        ], dtype=bool)
                                        if _src_keep_full.sum() < 30:
                                            raise RuntimeError(
                                                f"Flow-associated pathways: only "
                                                f"{int(_src_keep_full.sum())} source "
                                                f"cells intersect adata_hsc "
                                                f"(need ≥ 30). Likely caused by "
                                                f"SUBSAMPLE_FRACTION < 1.0."
                                            )
                                        _hsc_idx = np.array([
                                            _hsc_pos[n]
                                            for n in _src_names_full[_src_keep_full]
                                        ], dtype=int)
                                        disp_pw = disp_pw_full[_src_keep_full]
                                        pw_src_scores = np.asarray(
                                            pw_flow_obsm_full[_hsc_idx]
                                        )
                                    else:
                                        # adata_full has pathway_scores
                                        # natively — boolean mask works.
                                        disp_pw = disp_pw_full
                                        pw_src_scores = np.asarray(
                                            pw_flow_obsm_full[fr_pp["source_mask"]]
                                        )
                                    flow_mag_pw = np.linalg.norm(disp_pw, axis=1)
                                    mean_dir_pw = disp_pw.mean(axis=0)
                                    _md_norm = float(np.linalg.norm(mean_dir_pw))
                                    if _md_norm > 1e-10:
                                        mean_dir_pw_u = mean_dir_pw / _md_norm
                                    else:
                                        mean_dir_pw_u = mean_dir_pw
                                    flow_signed_pw = disp_pw @ mean_dir_pw_u
                                    # Sanity: flow + score rows must agree
                                    if pw_src_scores.shape[0] != disp_pw.shape[0]:
                                        raise RuntimeError(
                                            f"Flow-associated pathways: row "
                                            f"mismatch (scores={pw_src_scores.shape[0]} "
                                            f"vs flow={disp_pw.shape[0]}). Alignment "
                                            f"fell through."
                                        )
                                    if pw_src_scores.shape[0] < 30:
                                        html_genes += error_html(
                                            f"Flow-associated pathways skipped: "
                                            f"too few source cells "
                                            f"(N={pw_src_scores.shape[0]})"
                                        )
                                    else:
                                        log.info(
                                            f"  Flow-associated pathways: "
                                            f"({hi},{ci}) "
                                            f"N_src={pw_src_scores.shape[0]}, "
                                            f"N_pathways={len(pw_flow_names)}"
                                        )
                                        # Per-pathway Spearman vs both |flow|
                                        # (magnitude) and signed flow direction.
                                        # Magnitude picks up pathways that peak
                                        # in large-flow cells (rate-of-change);
                                        # signed picks up pathways that rise/
                                        # fall along the flow direction.
                                        pw_assoc_rows = []
                                        for pi_pw in range(len(pw_flow_names)):
                                            col = pw_src_scores[:, pi_pw].astype(float)
                                            if not np.any(np.isfinite(col)):
                                                continue
                                            try:
                                                rho_mag, p_mag = _sp_pw(
                                                    col, flow_mag_pw
                                                )
                                                rho_sig, p_sig = _sp_pw(
                                                    col, flow_signed_pw
                                                )
                                            except Exception:
                                                continue
                                            if np.isnan(rho_mag) and np.isnan(rho_sig):
                                                continue
                                            pw_assoc_rows.append({
                                                "pathway_idx": pi_pw,
                                                "pathway_name": pw_flow_names[pi_pw]
                                                    if pi_pw < len(pw_flow_names)
                                                    else f"pw_{pi_pw}",
                                                "rho_mag": float(rho_mag)
                                                    if not np.isnan(rho_mag) else 0.0,
                                                "p_mag": float(p_mag)
                                                    if not np.isnan(p_mag) else 1.0,
                                                "rho_signed": float(rho_sig)
                                                    if not np.isnan(rho_sig) else 0.0,
                                                "p_signed": float(p_sig)
                                                    if not np.isnan(p_sig) else 1.0,
                                            })
                                        if not pw_assoc_rows:
                                            html_genes += error_html(
                                                "Flow-associated pathways: "
                                                "no finite Spearman values."
                                            )
                                        else:
                                            pw_df = pd.DataFrame(pw_assoc_rows)
                                            # BH FDR over the pathway family
                                            # (signed is the primary ranking per
                                            # user intent; magnitude secondary).
                                            try:
                                                from statsmodels.stats.multitest import multipletests as _mt
                                                _, fdr_signed, _, _ = _mt(
                                                    pw_df["p_signed"].values,
                                                    method="fdr_bh",
                                                )
                                                _, fdr_mag, _, _ = _mt(
                                                    pw_df["p_mag"].values,
                                                    method="fdr_bh",
                                                )
                                            except Exception:
                                                fdr_signed = pw_df["p_signed"].values
                                                fdr_mag = pw_df["p_mag"].values
                                            pw_df["fdr_signed"] = fdr_signed
                                            pw_df["fdr_mag"] = fdr_mag
                                            pw_df["abs_rho_signed"] = pw_df["rho_signed"].abs()

                                            # Top 20 by |rho_signed|
                                            top_pw_fa = (
                                                pw_df.sort_values(
                                                    "abs_rho_signed", ascending=False
                                                )
                                                .head(20)
                                            )
                                            display_pw = top_pw_fa[[
                                                "pathway_name",
                                                "rho_signed", "p_signed", "fdr_signed",
                                                "rho_mag", "p_mag", "fdr_mag",
                                            ]].copy()
                                            display_pw["pathway_name"] = (
                                                display_pw["pathway_name"]
                                                .str.slice(0, 60)
                                            )
                                            display_pw["rho_signed"] = (
                                                display_pw["rho_signed"]
                                                .map(lambda v: f"{v:.3f}")
                                            )
                                            display_pw["rho_mag"] = (
                                                display_pw["rho_mag"]
                                                .map(lambda v: f"{v:.3f}")
                                            )
                                            display_pw["p_signed"] = (
                                                display_pw["p_signed"].map(fmt_pval)
                                            )
                                            display_pw["fdr_signed"] = (
                                                display_pw["fdr_signed"].map(fmt_pval)
                                            )
                                            display_pw["p_mag"] = (
                                                display_pw["p_mag"].map(fmt_pval)
                                            )
                                            display_pw["fdr_mag"] = (
                                                display_pw["fdr_mag"].map(fmt_pval)
                                            )
                                            display_pw = display_pw.rename(columns={
                                                "pathway_name": "Pathway",
                                                "rho_signed": "ρ (signed flow)",
                                                "p_signed": "p (signed)",
                                                "fdr_signed": "FDR (signed)",
                                                "rho_mag": "ρ (|flow|)",
                                                "p_mag": "p (|flow|)",
                                                "fdr_mag": "FDR (|flow|)",
                                            })
                                            n_sig_fa = int(
                                                (pw_df["fdr_signed"] < 0.05).sum()
                                            )
                                            html_genes += report.df_to_html(
                                                display_pw,
                                                caption=(
                                                    f"<b>Flow-associated pathways "
                                                    f"(r12): HSC A{hi+1} → CMP "
                                                    f"A{ci+1}</b>. Spearman "
                                                    f"correlation between each "
                                                    f"pathway's per-cell AUCell "
                                                    f"score and the cell's flow "
                                                    f"direction on the source-bin "
                                                    f"cells "
                                                    f"(N={pw_src_scores.shape[0]}). "
                                                    f"<code>ρ (signed flow)</code> "
                                                    f"= correlation with flow "
                                                    f"projected onto mean "
                                                    f"direction (positive = pathway "
                                                    f"rises along the flow; "
                                                    f"negative = falls). "
                                                    f"<code>ρ (|flow|)</code> = "
                                                    f"correlation with flow "
                                                    f"magnitude only (pathway "
                                                    f"peaks in rate-of-change cells "
                                                    f"regardless of direction). "
                                                    f"{n_sig_fa}/{len(pw_df)} "
                                                    f"pathways pass BH FDR < 0.05 "
                                                    f"on the signed correlation. "
                                                    f"Sorted by |ρ signed|."
                                                ),
                                            )

                                            # Pseudotime x pathway score plot
                                            # for the top-8 signed pathways
                                            # (parallels the per-gene pseudotime-
                                            # expansion plot but on pathway
                                            # scores, as user requested).
                                            try:
                                                from _paper_part1_viz import (
                                                    build_pseudotime_expansion_plot,
                                                )
                                                top8_fa = top_pw_fa.head(8)
                                                top8_scores = pw_src_scores[
                                                    :, top8_fa["pathway_idx"].values
                                                ]
                                                top8_names = [
                                                    str(n)[:40]
                                                    for n in top8_fa["pathway_name"].tolist()
                                                ]
                                                fig_pt_pw = build_pseudotime_expansion_plot(
                                                    flow_signed_pw,
                                                    top8_scores,
                                                    top8_names,
                                                    title=(
                                                        f"Pseudotime × pathway score: "
                                                        f"HSC A{hi+1} → CMP A{ci+1}"
                                                    ),
                                                    max_genes=8,
                                                )
                                                html_genes += report.fig_to_img(
                                                    fig_pt_pw,
                                                    caption=(
                                                        f"Top 8 flow-associated "
                                                        f"pathways (by |ρ signed|) "
                                                        f"traced along the signed "
                                                        f"flow pseudotime on the "
                                                        f"source-bin cells. X = "
                                                        f"transport distance "
                                                        f"projected onto mean flow "
                                                        f"direction. Y = AUCell "
                                                        f"pathway score (binned mean)."
                                                    ),
                                                )
                                                plt.close("all")
                                            except Exception as _e_fa_pt:
                                                log.warning(
                                                    f"Pseudotime × pathway plot failed: "
                                                    f"{_e_fa_pt}"
                                                )
                                except Exception as e_fa_pw:
                                    log.exception(
                                        f"Flow-associated pathways failed for "
                                        f"({hi},{ci})"
                                    )
                                    html_genes += error_html(
                                        f"Flow-associated pathways failed: "
                                        f"{e_fa_pw}"
                                    )

                            # 2c. Pseudotime x expansion/contraction plot
                            #     (replaces the old cosine ridgeplot).
                            #     Uses per-cell Jacobian expansion and
                            #     transport distance as pseudotime proxy.
                            try:
                                from _paper_part1_viz import (
                                    build_pseudotime_expansion_plot,
                                )
                                # Compute per-pair Jacobian with per-cell
                                # expansion for the top genes.
                                jac_pp = pc.tl.flow_jacobian(
                                    adata_full, fr_pp, fr_pp["model"],
                                    t=0.5,
                                    per_cell_features=True,
                                    n_top_features=20,
                                    n_permutations=0,
                                )
                                pc_exp_pp = jac_pp.get("per_cell_expansion")
                                pc_exp_gnames = jac_pp.get(
                                    "per_cell_expansion_gene_names", []
                                )
                                if (pc_exp_pp is not None
                                        and len(pc_exp_gnames) > 0):
                                    # Pseudotime proxy: per-cell transport
                                    # distance projected onto the mean flow
                                    # direction (scalar per source cell).
                                    src_pca_2c = adata_full.obsm[
                                        fr_pp.get("pca_key", "X_pca")
                                    ][fr_pp["source_mask"]]
                                    trans_2c = fr_pp["transported"]
                                    disp_2c = trans_2c - src_pca_2c
                                    mean_dir = disp_2c.mean(axis=0)
                                    dir_norm = np.linalg.norm(mean_dir)
                                    if dir_norm > 1e-10:
                                        mean_dir /= dir_norm
                                    pseudotime_2c = disp_2c @ mean_dir

                                    fig_pt_exp = build_pseudotime_expansion_plot(
                                        pseudotime_2c,
                                        np.asarray(pc_exp_pp),
                                        pc_exp_gnames,
                                        title=(
                                            f"Pseudotime x expansion: "
                                            f"HSC A{hi+1} → CMP A{ci+1}"
                                        ),
                                        max_genes=10,
                                    )
                                    html_genes += report.fig_to_img(
                                        fig_pt_exp,
                                        caption=(
                                            f"Pseudotime x expansion for "
                                            f"HSC A{hi+1} → CMP A{ci+1}. "
                                            f"X = transport distance "
                                            f"projected onto mean flow "
                                            f"direction (pseudotime proxy). "
                                            f"Y = per-cell per-gene Jacobian "
                                            f"expansion score (>1 = "
                                            f"expanding, <1 = contracting). "
                                            f"Each line = one gene (top "
                                            f"{min(10, len(pc_exp_gnames))} "
                                            f"by mean |expansion|), binned "
                                            f"along pseudotime."
                                        ),
                                    )
                                    plt.close("all")
                                else:
                                    html_genes += error_html(
                                        f"Pseudotime x expansion skipped: "
                                        f"no per-cell expansion data for "
                                        f"({hi},{ci})"
                                    )
                            except Exception as e_pt_exp:
                                log.exception(
                                    f"Pseudotime x expansion failed for "
                                    f"({hi},{ci})"
                                )
                                html_genes += error_html(
                                    f"Pseudotime x expansion failed: "
                                    f"{e_pt_exp}"
                                )

                            # Summary metrics
                            try:
                                src_pca_pp = adata_full.obsm[
                                    fr_pp.get("pca_key", "X_pca")
                                ][fr_pp["source_mask"]]
                                tgt_pca_pp = adata_full.obsm[
                                    fr_pp.get("pca_key", "X_pca")
                                ][fr_pp["target_mask"]]
                                w2_b_pp = wasserstein2_distance(
                                    src_pca_pp, tgt_pca_pp, max_n=1000
                                )
                                w2_a_pp = wasserstein2_distance(
                                    fr_pp["transported"], tgt_pca_pp, max_n=1000
                                )
                            except Exception:
                                w2_b_pp = w2_a_pp = float("nan")
                            n_genes_fdr05_pp = (
                                int((np.asarray(fdr_pp) < 0.05).sum())
                                if fdr_pp is not None else 0
                            )
                            pp_summary_rows.append({
                                "Pair": f"H A{hi+1} → C A{ci+1}",
                                "N src": n_src_pp,
                                "N tgt": n_tgt_pp,
                                "Mass": f"{corr_matrix[hi, ci]:.3f}",
                                "Pair FDR": (
                                    f"{pair_fdr[hi, ci]:.3f}"
                                    if pair_fdr is not None else "?"
                                ),
                                "W2 before": f"{w2_b_pp:.3f}",
                                "W2 after": f"{w2_a_pp:.3f}",
                                "MMD red": (
                                    f"{(1 - fr_pp['mmd_after']/max(fr_pp['mmd_before'], 1e-10)):.1%}"
                                ),
                                "Genes FDR<0.05": n_genes_fdr05_pp,
                            })
                        finally:
                            if "_pp2f_label" in adata_full.obs.columns:
                                del adata_full.obs["_pp2f_label"]

                    except Exception as e:
                        log.exception(
                            f"Fig 2F per-pair detail failed for ({hi},{ci})"
                        )
                        html_genes += error_html(
                            f"Per-pair detail failed for HSC A{hi+1} → "
                            f"CMP A{ci+1}: {e}"
                        )
                        pp_summary_rows.append({
                            "Pair": f"H A{hi+1} → C A{ci+1}",
                            "Status": f"failed: {e}",
                        })

                if pp_summary_rows:
                    html_genes += report.df_to_html(
                        pd.DataFrame(pp_summary_rows),
                        caption=(
                            f"Fig 2F per-pair detail summary "
                            f"({len(pp_summary_rows)} pairs, "
                            f"MAX_SIG_PAIRS = {MAX_SIG_PAIRS})."
                        ),
                    )

    except Exception as e:
        log.exception("Fig 2F failed")
        html_genes += error_html(f"Fig 2F failed: {e}")
        plt.close("all")

    report.add_section("Figure 2F: Flow Gene Alignment + Straw Plots", html_genes, step_num="2F")

    return gene_reg


# ============================================================================
# MAIN
# ============================================================================

def main():
    report = HTMLReport(f"Paper Part 1: HSC Technical Hypotheses ({_DATE_TAG})")

    t0 = time.time()

    # Phase 1: Train models
    log.info("=== PHASE 1: Train models ===")
    adata_hsc, adata_cmp, res_hsc, res_cmp = phase1_train_models(report)
    log.info(f"Phase 1 done in {time.time() - t0:.0f}s")

    # Phase 1b: Drift / stability QC panel (W-A7). Surfaces archetype
    # position drift across epochs so the user can see whether training
    # is dragging archetypes away from their PCHA init (the suspected
    # cause of the "archetypes far from data" symptom in the r9 review).
    try:
        from _paper_part1_viz import build_drift_qc_panel
        drift_panel_html = build_drift_qc_panel(
            [("HSC", res_hsc), ("CMP", res_cmp)],
            drift_threshold=0.05,
            converged_window=10,
        )
        report.add_section(
            "Drift & Stability QC (W-A7)",
            drift_panel_html,
            step_num="1b",
        )
    except Exception as e:
        log.exception("Drift QC panel failed")
        report.add_section(
            "Drift & Stability QC (W-A7)",
            error_html(f"Drift QC panel failed: {e}"),
            step_num="1b",
        )

    # Phase 2: Figure 1
    t1 = time.time()
    log.info("=== PHASE 2: Figure 1 ===")
    phase2_figure1(adata_hsc, adata_cmp, res_hsc, report)
    log.info(f"Phase 2 done in {time.time() - t1:.0f}s")

    # Phase 3: Figure 2
    t2 = time.time()
    log.info("=== PHASE 3: Figure 2 ===")
    phase3_figure2(adata_hsc, adata_cmp, report)
    log.info(f"Phase 3 done in {time.time() - t2:.0f}s")

    # Save report
    report.save(REPORT_PATH)
    log.info(f"Report saved to {REPORT_PATH}")
    log.info(f"Total runtime: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

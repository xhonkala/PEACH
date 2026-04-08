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

# Subsample fraction for fast iteration. Set to 1.0 for the production paper run.
# 0.3 = ~6.4K cells (vs 21K full), runs in ~30-60min instead of 3-4h.
SUBSAMPLE_FRACTION = 0.3
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
                           exclusive_threshold=1.5, top_n_per_archetype=10,
                           fdr_threshold=0.05):
    """Convert simplex regression dict → long-format DataFrame for pc.pl.dotplot.

    Builds a long-format table with one row per (feature, archetype):
      - <y_col>: feature name
      - archetype: archetype label (archetype_0, archetype_1, ...)
      - mean_archetype: |vertex coefficient| (used as effect size for dot size)
      - pvalue: vertex t-test p-value
      - pvalue_fdr: FDR-corrected vertex p-value
      - r_squared: per-feature R² (used for ranking top N per archetype)

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

    Returns
    -------
    pd.DataFrame
        Long-format table ready for pc.pl.dotplot.
    """
    feat_names = list(reg_result.get("feature_names", []))
    coefs = np.asarray(reg_result.get("vertex_coefficients", []))
    pvals = np.asarray(reg_result.get("vertex_pvalues", []))
    fdrs = np.asarray(reg_result.get("vertex_pvalues_fdr", []))
    r2 = np.asarray(reg_result.get("r_squared_degree1", []))

    if coefs.size == 0 or len(feat_names) == 0:
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
        for a in range(K):
            p = float(pvals[fi, a]) if pvals.size else 1.0
            f = float(fdrs[fi, a]) if fdrs.size else 1.0
            if f > fdr_threshold:
                continue
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

def archetype_cell_proximity(adata, k=5):
    """For each archetype, return median distance to its k nearest cells in PCA space.

    Parameters
    ----------
    adata : AnnData
        Must have ``uns['archetype_coordinates']`` and ``obsm['X_pca']``.
    k : int
        Number of nearest cells per archetype to query.

    Returns
    -------
    dict with keys:
        ``median_dist_per_archetype`` — ndarray shape (n_archetypes,)
        ``median_dist_overall``       — float, median of all queried distances
        ``max_dist_per_archetype``    — ndarray shape (n_archetypes,)
    None if required data is missing.
    """
    from scipy.spatial import cKDTree

    archetypes = adata.uns.get("archetype_coordinates", None)
    pca = adata.obsm.get("X_pca", None)
    if archetypes is None or pca is None:
        return None
    archetypes = np.asarray(archetypes)
    pca = np.asarray(pca)
    # Restrict to the same number of dimensions used by archetypes
    n_dims = min(archetypes.shape[1], pca.shape[1])
    archetypes = archetypes[:, :n_dims]
    pca = pca[:, :n_dims]
    tree = cKDTree(pca)
    dists, _ = tree.query(archetypes, k=min(k, pca.shape[0]))
    if dists.ndim == 1:
        dists = dists.reshape(-1, 1)
    return {
        "median_dist_per_archetype": np.median(dists, axis=1),
        "median_dist_overall": float(np.median(dists)),
        "max_dist_per_archetype": dists.max(axis=1),
    }


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
        n_archetypes_range=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        hidden_dims_options=[[64, 128], [128, 256]],
        inflation_factor_range=[1.0],
        cv_folds=3, max_epochs_cv=15, subsample_fraction=0.8,
    )
    ranked_cmp = cv_cmp.rank_by_metric("archetype_r2")
    ranked_cmp = [r for r in ranked_cmp if r["metric_value"] > -1e6]
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
        kld_weight=0.09, archetypal_weight=1.0, inflation_factor=1.0,
        model_config={"manifold_weight": 0.001},
        early_stopping=True, early_stopping_patience=EARLY_STOP_PATIENCE,
    )
    # --- Convergence QC (CMP) ---
    _cmp_tc = res_cmp.get("training_config", {})
    _cmp_actual = _cmp_tc.get("actual_epochs", MAX_EPOCHS_FINAL)
    _cmp_early = _cmp_tc.get("early_stop_triggered", False)
    _cmp_history = res_cmp.get("history", {})
    _cmp_losses = _cmp_history.get("loss", [])
    if len(_cmp_losses) >= 10:
        _last10 = _cmp_losses[-10:]
        _cmp_delta_mean = float(np.mean(np.abs(np.diff(_last10))))
    else:
        _cmp_delta_mean = float("nan")
    _cmp_hit_cap = _cmp_actual >= MAX_EPOCHS_FINAL and not _cmp_early
    log.info(
        f"CMP convergence QC: actual_epochs={_cmp_actual}, early_stop={_cmp_early}, "
        f"last-10-epoch mean |delta_loss|={_cmp_delta_mean:.5f}, hit_cap={_cmp_hit_cap}"
    )
    if _cmp_hit_cap:
        log.warning(
            f"CMP model hit the {MAX_EPOCHS_FINAL}-epoch cap without early stopping — "
            "may not be fully converged. Consider increasing MAX_EPOCHS_FINAL or inspecting the loss curve."
        )
    # Enrich training_config with model-level params for downstream metric display
    res_cmp.setdefault("training_config", {}).update({
        "n_archetypes": K_cmp,
        "hidden_dims": hd_cmp,
        "inflation_factor": 1.0,
        "use_pcha_init": True,
    })
    pc.tl.archetypal_coordinates(adata_cmp, verbose=False)
    pc.tl.assign_archetypes(adata_cmp, verbose=False)
    pc.tl.extract_archetype_weights(adata_cmp, verbose=False)
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
        metric_card(str(_tc_cmp.get("use_pcha_init", True)), "use_pcha_init"),
    ])
    # Convergence QC badge (CMP)
    _cmp_conv_msg = (
        f"Early stopped at epoch {_cmp_actual}/{MAX_EPOCHS_FINAL}. "
        f"Last-10-epoch mean |Δloss| = {_cmp_delta_mean:.5f}."
        if _cmp_early else
        f"Ran {_cmp_actual}/{MAX_EPOCHS_FINAL} epochs. "
        f"Last-10-epoch mean |Δloss| = {_cmp_delta_mean:.5f}."
        + (" <b style='color:orange'>[NON-CONVERGED: hit epoch cap]</b>" if _cmp_hit_cap else "")
    )
    html_cmp += report.text(f"<b>Convergence QC</b>: {_cmp_conv_msg}")

    # Training metrics
    try:
        fig_train = pc.pl.training_metrics(res_cmp["history"], display=False)
        if fig_train:
            html_cmp += safe_plotly_html(report, fig_train, "CMP training metrics")
    except Exception as e:
        html_cmp += error_html(f"CMP training metrics failed: {e}")

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
        n_archetypes_range=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        hidden_dims_options=[[64, 128], [128, 256]],
        inflation_factor_range=[1.0],
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
        kld_weight=0.09, archetypal_weight=1.0, inflation_factor=1.0,
        model_config={"manifold_weight": 0.001},
        early_stopping=True, early_stopping_patience=EARLY_STOP_PATIENCE,
    )
    # --- Convergence QC (HSC) ---
    _hsc_tc = res_hsc.get("training_config", {})
    _hsc_actual = _hsc_tc.get("actual_epochs", MAX_EPOCHS_FINAL)
    _hsc_early = _hsc_tc.get("early_stop_triggered", False)
    _hsc_history = res_hsc.get("history", {})
    _hsc_losses = _hsc_history.get("loss", [])
    if len(_hsc_losses) >= 10:
        _last10_hsc = _hsc_losses[-10:]
        _hsc_delta_mean = float(np.mean(np.abs(np.diff(_last10_hsc))))
    else:
        _hsc_delta_mean = float("nan")
    _hsc_hit_cap = _hsc_actual >= MAX_EPOCHS_FINAL and not _hsc_early
    log.info(
        f"HSC convergence QC: actual_epochs={_hsc_actual}, early_stop={_hsc_early}, "
        f"last-10-epoch mean |delta_loss|={_hsc_delta_mean:.5f}, hit_cap={_hsc_hit_cap}"
    )
    if _hsc_hit_cap:
        log.warning(
            f"HSC model hit the {MAX_EPOCHS_FINAL}-epoch cap without early stopping — "
            "may not be fully converged. Consider increasing MAX_EPOCHS_FINAL or inspecting the loss curve."
        )
    # Enrich training_config with model-level params for downstream metric display
    res_hsc.setdefault("training_config", {}).update({
        "n_archetypes": K_hsc,
        "hidden_dims": hd_hsc,
        "inflation_factor": 1.0,
        "use_pcha_init": True,
    })
    pc.tl.archetypal_coordinates(adata_hsc, verbose=False)
    pc.tl.assign_archetypes(adata_hsc, verbose=False)
    pc.tl.extract_archetype_weights(adata_hsc, verbose=False)
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
    # Convergence QC badge (HSC — phase 1 section)
    _hsc_conv_msg = (
        f"Early stopped at epoch {_hsc_actual}/{MAX_EPOCHS_FINAL}. "
        f"Last-10-epoch mean |Δloss| = {_hsc_delta_mean:.5f}."
        if _hsc_early else
        f"Ran {_hsc_actual}/{MAX_EPOCHS_FINAL} epochs. "
        f"Last-10-epoch mean |Δloss| = {_hsc_delta_mean:.5f}."
        + (" <b style='color:orange'>[NON-CONVERGED: hit epoch cap]</b>" if _hsc_hit_cap else "")
    )
    html_hsc += report.text(f"<b>Convergence QC</b>: {_hsc_conv_msg}")

    try:
        fig_train_hsc = pc.pl.training_metrics(res_hsc["history"], display=False)
        if fig_train_hsc:
            html_hsc += safe_plotly_html(report, fig_train_hsc, "HSC training metrics")
    except Exception as e:
        html_hsc += error_html(f"HSC training metrics failed: {e}")

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
        metric_card(str(_tc_hsc.get("use_pcha_init", True)), "use_pcha_init"),
    ])

    # --- Archetype-to-cell proximity diagnostic (Task 12) ---
    # Addresses the concern that outlier removal causes archetypes to be learned at convex-hull
    # extremes that are then depopulated, making the simplex an extrapolation of removed cells.
    try:
        _K_hsc_prox = adata_hsc.uns.get("archetype_coordinates", np.array([[]])).shape[0]
        proximity = archetype_cell_proximity(adata_hsc, k=5)
        if proximity:
            # Ratio: archetype median dist / overall median dist.  > 2.0 = likely extrapolated.
            _prox_ratios = proximity["median_dist_per_archetype"] / proximity["median_dist_overall"]
            html += report.text("<b>Archetype-to-cell proximity diagnostic</b>")
            prox_df = pd.DataFrame({
                "Archetype": [f"A{i+1}" for i in range(_K_hsc_prox)],
                "Median dist to 5 nearest cells": [f"{v:.4f}" for v in proximity["median_dist_per_archetype"]],
                "Max dist to 5 nearest cells": [f"{v:.4f}" for v in proximity["max_dist_per_archetype"]],
                "Proximity ratio (vs overall)": [f"{r:.3f}" for r in _prox_ratios],
            })
            html += report.df_to_html(
                prox_df,
                caption=(
                    f"Overall median cell-to-cell 5-NN distance = {proximity['median_dist_overall']:.4f}. "
                    "Proximity ratio > 2.0 may indicate archetype is extrapolated / sits in sparse region "
                    "vacated by outlier removal."
                ),
            )
            _sparse_arch = [i for i, r in enumerate(_prox_ratios) if r > 2.0]
            if _sparse_arch:
                _sparse_labels = [f"A{i+1}" for i in _sparse_arch]
                html += report.text(
                    f"<b style='color:orange'>Warning</b>: HSC archetypes {_sparse_labels} have proximity "
                    "ratio > 2x global median — may be extrapolated from outlier removal."
                )
                log.warning(
                    f"HSC archetypes {_sparse_arch} have proximity ratio > 2x global median — "
                    "may be extrapolated from outlier removal."
                )
            else:
                html += report.text("All archetypes within 2x proximity ratio — no extrapolation concern.")
        else:
            html += report.text("Proximity diagnostic: archetype_coordinates or X_pca not found, skipped.")
    except Exception as _prox_exc:
        log.warning(f"Archetype-to-cell proximity diagnostic failed: {_prox_exc}")
        html += report.text(f"Proximity diagnostic failed: {_prox_exc}")

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
        pc.tl.assign_archetypes(adata_holdout_hsc, verbose=False)
        pc.tl.extract_archetype_weights(adata_holdout_hsc, verbose=False)

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
        pc.tl.assign_archetypes(adata_cmp_copy, verbose=False)
        pc.tl.extract_archetype_weights(adata_cmp_copy, verbose=False)

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
            html += metric_grid([
                metric_card(f"{len(r2_arr)}", "Features tested"),
                metric_card(f"{(r2_arr > 0.05).sum()}", "R² > 0.05"),
                metric_card(f"{(r2_arr > 0.10).sum()}", "R² > 0.10"),
                metric_card(f"{np.median(r2_arr):.4f}", "Median R²"),
                metric_card(f"{np.max(r2_arr):.4f}", "Max R²"),
            ])

        # --- Simplex regression dotplot (exclusive only, via pc.pl.dotplot) ---
        try:
            log.info("Building simplex regression long-format dataframe for CMP...")
            cmp_long = regression_to_long_df(
                reg_result, y_col="gene", exclusive_only=True,
                exclusive_threshold=1.5, top_n_per_archetype=10,
                fdr_threshold=0.05)
            log.info(f"  CMP long df rows: {len(cmp_long)}, unique genes: "
                     f"{cmp_long['gene'].nunique() if len(cmp_long) else 0}")
            if len(cmp_long) > 0:
                fig_reg = pc.pl.dotplot(
                    cmp_long, x_col="archetype", y_col="gene",
                    size_col="mean_archetype", color_col="pvalue",
                    top_n_per_group=10,
                    title="CMP SIMPLEX REGRESSION dotplot (exclusive ≥1.5x, FDR<0.05)")
                html += report.fig_to_img(fig_reg,
                    caption=f"CMP simplex regression: top 10 exclusive genes per archetype "
                            f"(method=Scheffé polynomial regression, exclusive_threshold=1.5, "
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
                    exclusive_threshold=1.5, top_n_per_archetype=5, fdr_threshold=0.05)
                if len(pw_long) > 0:
                    fig_pw_reg = pc.pl.dotplot(
                        pw_long, x_col="archetype", y_col="pathway",
                        size_col="mean_archetype", color_col="pvalue",
                        top_n_per_group=5,
                        title="CMP SIMPLEX REGRESSION pathway dotplot (C5:BP, exclusive ≥1.5x)")
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
        pc.tl.assign_archetypes(adata_cmp_hold, verbose=False)
        pc.tl.extract_archetype_weights(adata_cmp_hold, verbose=False)

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
        html_reg += metric_grid([
            metric_card(f"{len(r2_arr)}", "Features tested"),
            metric_card(f"{(r2_arr > 0.05).sum()}", "R² > 0.05"),
            metric_card(f"{(r2_arr > 0.10).sum()}", "R² > 0.10"),
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

    # --- Fig 2A: HSC simplex regression dotplot (exclusive, degree 1) ---
    try:
        html_reg += "<h3>Fig 2A: HSC simplex regression</h3>"
        log.info("Building HSC simplex regression long-format dataframe...")
        hsc_long = regression_to_long_df(
            reg_result, y_col="gene", exclusive_only=True,
            exclusive_threshold=1.5, top_n_per_archetype=10, fdr_threshold=0.05)
        log.info(f"  HSC long df rows: {len(hsc_long)}, unique genes: "
                 f"{hsc_long['gene'].nunique() if len(hsc_long) else 0}")
        if len(hsc_long) > 0:
            fig_d1 = pc.pl.dotplot(
                hsc_long, x_col="archetype", y_col="gene",
                size_col="mean_archetype", color_col="pvalue",
                top_n_per_group=10,
                title="HSC SIMPLEX REGRESSION (degree 1, exclusive ≥1.5x, FDR<0.05)")
            html_reg += report.fig_to_img(fig_d1,
                caption=f"HSC simplex regression: top 10 exclusive genes per archetype "
                        f"(Scheffé polynomial, HC3 robust SE, {hsc_long['gene'].nunique()} unique genes)")
            plt.close("all")

        # --- Degree-1 / degree-2 / degree-3 side-by-side comparison (Task 15) ---
        # Degree-1: pure archetype main effects (vertex betas from Scheffé degree-1 design).
        # Degree-2: adds pairwise interaction terms; features ranked by ΔR²(d2−d1) showing
        #           which genes gain the most by modelling archetype blending.
        # Degree-3+: incremental R² gains from higher-order terms; vertex coefs are not
        #             separately extracted — shown as a gain table from comprehensive_degree.
        html_reg += "<h4>Fig 2A comparison: degree-1 vs degree-2 vs degree-3 feature sets</h4>"
        html_reg += report.text(
            "<b>Panel interpretation</b>: Degree-1 = pure archetype main effects (archetypes act "
            "independently). Degree-2 = pairwise archetype interactions (enriched in mixed states). "
            "Degree-3 = triple interactions (incremental R² gain shown; vertex coefs not separately "
            "fitted here — requires K≥4 for non-trivial gains). Each panel uses the same 5000 HVGs.")

        # Degree-2 dotplot: rank features by ΔR² = r2_d2 - r2_d1, show top N per archetype.
        try:
            r2_d1 = np.asarray(reg_result.get("r_squared_degree1", []))
            r2_d2 = np.asarray(reg_result.get("r_squared_degree2", []))
            if r2_d2.size > 0 and r2_d1.size == r2_d2.size:
                delta_r2 = r2_d2 - r2_d1
                # Build long-df with delta_r2 as sorting key but same vertex coefs for visual
                # (vertex coefs from degree-1 model remain the main-effect interpretable part)
                feat_names_reg = list(reg_result.get("feature_names", []))
                coefs_d1 = np.asarray(reg_result.get("vertex_coefficients", []))
                fdrs_d1 = np.asarray(reg_result.get("vertex_pvalues_fdr", []))
                n_feat_d2 = coefs_d1.shape[0] if coefs_d1.ndim == 2 else 0
                K_d = coefs_d1.shape[1] if coefs_d1.ndim == 2 else 0
                if n_feat_d2 > 0 and K_d > 0:
                    # Filter to features with meaningful interaction gain (ΔR² > 0.01)
                    gain_mask = delta_r2 > 0.01
                    log.info(f"  Degree-2 gain: {gain_mask.sum()} features with ΔR²>0.01")
                    rows_d2 = []
                    for fi in range(n_feat_d2):
                        if not gain_mask[fi]:
                            continue
                        am = int(np.argmax(np.abs(coefs_d1[fi])))
                        for a in range(K_d):
                            fdr_v = float(fdrs_d1[fi, a]) if fdrs_d1.size else 1.0
                            if fdr_v > 0.05:
                                continue
                            rows_d2.append({
                                "gene": feat_names_reg[fi],
                                "archetype": f"archetype_{a}",
                                "mean_archetype": float(np.abs(coefs_d1[fi, a])),
                                "pvalue": fdr_v,
                                "pvalue_fdr": fdr_v,
                                "r_squared": float(delta_r2[fi]),   # rank by ΔR²
                                "argmax_archetype": am,
                            })
                    df_d2 = pd.DataFrame(rows_d2) if rows_d2 else pd.DataFrame()
                    # Trim to top 10 per archetype by ΔR²
                    if not df_d2.empty:
                        keep_d2 = np.zeros(len(df_d2), dtype=bool)
                        for a in range(K_d):
                            is_am = df_d2["argmax_archetype"] == a
                            if is_am.any():
                                top_feats_d2 = (df_d2[is_am]
                                                .drop_duplicates(subset=["gene"])
                                                .nlargest(10, "r_squared")["gene"].values)
                                keep_d2 |= df_d2["gene"].isin(top_feats_d2)
                        df_d2 = df_d2[keep_d2].reset_index(drop=True)
                    if len(df_d2) > 0:
                        fig_d2 = pc.pl.dotplot(
                            df_d2, x_col="archetype", y_col="gene",
                            size_col="mean_archetype", color_col="pvalue",
                            top_n_per_group=10,
                            title="HSC degree-2 interaction gain (ΔR²>0.01, vertex main effects shown)")
                        html_reg += report.fig_to_img(fig_d2,
                            caption="Degree-2 panel: top 10 genes per archetype ranked by ΔR² (degree-2 "
                                    "minus degree-1 R²). Dot size = |vertex main-effect coef|; colour = "
                                    "vertex FDR q. These genes are most enriched in archetype blending zones.")
                        plt.close("all")
                    else:
                        html_reg += report.text(
                            "Degree-2 panel: no features with ΔR²>0.01 and vertex FDR<0.05.")
            else:
                html_reg += report.text("Degree-2 panel skipped: r_squared_degree2 not available.")
        except Exception as e:
            log.exception("Degree-2 side-by-side panel failed")
            html_reg += error_html(f"Degree-2 side-by-side panel failed: {e}")

        # Degree-3+ incremental gains table (from comprehensive_degree comparison).
        try:
            deg_comp = reg_result.get("degree_comparison") or gene_reg.get("degree_comparison", {})
            if deg_comp:
                comp_rows = []
                for deg_key in sorted(deg_comp.keys()):
                    d = deg_comp[deg_key]
                    comp_rows.append({
                        "Degree": deg_key,
                        "Mean R²": f"{np.mean(d['r_squared']):.4f}",
                        "Mean ΔR²": f"{np.mean(d['delta_r2']):.4f}",
                        "Max ΔR²": f"{np.max(d['delta_r2']):.4f}",
                        "Features with incr. FDR<0.05": str(d.get("significant_features", "?")),
                        "Extra params": str(d.get("n_params", "?")),
                    })
                if comp_rows:
                    html_reg += report.df_to_html(pd.DataFrame(comp_rows),
                        caption="Comprehensive degree comparison: incremental R² gains from adding "
                                "higher-order interaction terms. Degree-3 = triple archetype interactions. "
                                "Each row shows mean/max gain over all 5K tested genes.")
            else:
                html_reg += report.text("Degree-3 table: comprehensive_degree comparison not available.")
        except Exception as e:
            log.exception("Degree-3 comprehensive comparison table failed")
            html_reg += error_html(f"Degree-3 comprehensive comparison table failed: {e}")

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

            # Build dotplots for each interaction sub-type that has ≥ 1 entry
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
                            f"gamma = interaction coefficient; beta_j/beta_k = main effects "
                            f"for the two archetypes in the pair.")
                # Build dotplot for this sub-type: use the long-df of degree-2 interaction features
                # limited to features in this sub-type's set
                sub_feat_names = list(df_sub["feature"].unique())
                sub_long = regression_to_long_df(
                    reg_result, y_col="gene", exclusive_only=False,
                    exclusive_threshold=1.0, top_n_per_archetype=50, fdr_threshold=0.05)
                if not sub_long.empty and len(sub_feat_names) > 0:
                    sub_long_filtered = sub_long[sub_long["gene"].isin(sub_feat_names)]
                    if len(sub_long_filtered) > 0:
                        try:
                            fig_sub = pc.pl.dotplot(
                                sub_long_filtered, x_col="archetype", y_col="gene",
                                size_col="mean_archetype", color_col="pvalue",
                                top_n_per_group=10,
                                title=f"HSC {sub_type} pattern genes (top 10 per archetype)")
                            html_reg += report.fig_to_img(fig_sub,
                                caption=f"{sub_type.capitalize()} pattern dotplot: "
                                        f"{len(sub_long_filtered['gene'].unique())} unique genes. "
                                        f"Dot size = |vertex main-effect coef|, colour = vertex FDR q.")
                            plt.close("all")
                        except Exception as e:
                            html_reg += error_html(f"{sub_type} dotplot failed: {e}")
                    else:
                        html_reg += report.text(
                            f"  {sub_type}: features found in interaction_detail but none "
                            f"passed vertex FDR<0.05 filter for dotplot.")
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
                    exclusive_threshold=1.5, top_n_per_archetype=5, fdr_threshold=0.05)
                log.info(f"  HSC pathway long df rows: {len(hsc_pw_long)}")
                if len(hsc_pw_long) > 0:
                    fig_pw = pc.pl.dotplot(
                        hsc_pw_long, x_col="archetype", y_col="pathway",
                        size_col="mean_archetype", color_col="pvalue",
                        top_n_per_group=5,
                        title="HSC SIMPLEX REGRESSION pathway dotplot (C5:BP, exclusive ≥1.5x)")
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
                f"<b>Cross-fit Spearman</b> (HSC vs CMP models): "
                f"{n_shared} shared features, {n_sig_cs} FDR-significant. "
                f"Compares archetype coefficient profiles between independently trained models.")
            try:
                fig_sim = pc.pl.feature_similarity_heatmap(adata_hsc, show=False)
                html_wald += safe_plotly_html(report, fig_sim,
                    "Cross-fit feature similarity heatmap (HSC_K x CMP_K Spearman rho)")
            except Exception as e:
                html_wald += error_html(f"Similarity heatmap failed: {e}")
        except Exception as e:
            html_wald += error_html(f"Cross-fit Spearman failed: {e}")

        # --- UpSet plot: per-archetype significant feature set intersections ---
        try:
            from upsetplot import from_contents, UpSet
            import matplotlib.pyplot as _plt_upset

            _upset_reg = adata_hsc.uns.get("peach_simplex_regression_genes", {})
            _upset_feat = list(_upset_reg.get("feature_names", []))
            _upset_fdr = _upset_reg.get("vertex_pvalues_fdr")
            if _upset_fdr is not None and len(_upset_feat) > 0:
                _upset_fdr = np.asarray(_upset_fdr)  # [n_features, K]
                if _upset_fdr.ndim == 2 and _upset_fdr.shape[0] == len(_upset_feat):
                    _arch_sig_sets = {}
                    for _ai in range(_upset_fdr.shape[1]):
                        _sig_mask = _upset_fdr[:, _ai] < 0.05
                        _arch_sig_sets[f"A{_ai + 1}"] = set(
                            _upset_feat[_j] for _j in range(len(_upset_feat)) if _sig_mask[_j]
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

        html_wald += report.text(
            "<b>Wald contrasts method</b>: HC3 sandwich covariance from Scheffe polynomial regression. "
            "Per-pair BH FDR correction over shared genes. "
            "Cross-fit Wald (below in Fig 2E) uses flow soft assignment to identify high-relatedness "
            "HSC→CMP archetype pairs, then compares beta vectors between independent fits.")
        html_wald += report.text(
            "<b>Wald null model (Fig 2D)</b>: Wald contrasts between archetype pairs test whether "
            "per-feature coefficients differ between two archetypes. "
            "For each pair (i, j), we compute the Wald statistic (β_i − β_j) / sqrt(SE_i² + SE_j²) "
            "where β and SE come from the within-fit simplex regression. "
            "Under the null hypothesis that features have identical effects in both archetypes, "
            "this statistic is asymptotically normal. "
            "P-values are two-sided normal p-values; significance is assessed after Benjamini-Hochberg "
            "FDR correction applied <b>globally</b> across all K*(K−1)/2 archetype pairs (not per-pair) "
            "to avoid dilution.")
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
        from peach._core.utils.archetype_comparison import compute_archetype_correspondence

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
        pc.tl.assign_archetypes(adata_full, verbose=False)
        pc.tl.extract_archetype_weights(adata_full, verbose=False)

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

                # Use library function (hard mode) instead of inline rank-1-prone soft outer product.
                # See src/peach/_core/utils/archetype_comparison.py:compute_archetype_correspondence
                corr_result = compute_archetype_correspondence(
                    source_weights=weights_src_hsc,
                    source_coords=fr["transported"],
                    target_weights=weights_tgt_cmp,
                    target_coords=target_pca_matched,
                    k=10,
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

                # Per-pair null model: permute source weights row-wise (breaks source-cell ↔
                # source-archetype binding) and recompute the full correspondence matrix via
                # the library function. This isolates the null from the k-NN computation and
                # tests the entire pipeline rather than just the nn lookup.
                log.info("Building per-pair null model for correspondence matrix...")
                rng_corr = np.random.default_rng(42)
                n_perm_corr = 50
                null_corrs = np.zeros((n_perm_corr, K_h, K_c))
                n_src_corr = weights_src_hsc.shape[0]
                for perm_i in range(n_perm_corr):
                    perm_idx = rng_corr.permutation(n_src_corr)
                    null_result = compute_archetype_correspondence(
                        source_weights=weights_src_hsc[perm_idx],
                        source_coords=fr["transported"],
                        target_weights=weights_tgt_cmp,
                        target_coords=target_pca_matched,
                        k=10,
                        method="hard",
                    )
                    null_corrs[perm_i] = null_result["mass"]

                null_mean_corr = null_corrs.mean(axis=0)
                null_std_corr = null_corrs.std(axis=0)
                # Per-pair p-value: (#null >= observed + 1) / (n_perm + 1)
                pair_pvals = np.zeros((K_h, K_c))
                for i in range(K_h):
                    for j in range(K_c):
                        pair_pvals[i, j] = (np.sum(null_corrs[:, i, j] >= corr[i, j]) + 1) / (n_perm_corr + 1)
                # Per-pair FDR (BH on the K_h*K_c family)
                pair_pvals_flat = pair_pvals.flatten()
                pair_fdr_flat = false_discovery_control(pair_pvals_flat, method="bh")
                pair_fdr = pair_fdr_flat.reshape(K_h, K_c)

                # Per-pair z-score vs shuffle null
                safe_null_std = np.where(null_std_corr < 1e-10, 1.0, null_std_corr)
                pair_z = (corr_matrix - null_mean_corr) / safe_null_std

                src_labels = [f"HSC A{i+1}" for i in range(K_h)]
                tgt_labels = [f"CMP A{j+1}" for j in range(K_c)]

                # --- Raw correspondence matrix (transport mass) ---
                corr_df = pd.DataFrame(corr, index=src_labels, columns=tgt_labels)
                html_flow += report.df_to_html(corr_df,
                    caption=f"Raw correspondence matrix (K_hsc={K_h} x K_cmp={K_c}, transport mass). "
                            f"Source weights from HSC model, target from CMP model. "
                            f"Computed via compute_archetype_correspondence(method='hard').")

                # --- Row-normalized correspondence (Markov transition interpretation) ---
                # Each row = P(CMP arch j | HSC arch i): "given a source cell in HSC arch i,
                # what fraction of its transport mass lands in each CMP arch?"
                # Rows sum to 1 → can be read as a transition kernel from HSC arch space to CMP arch space.
                # corr_markov already row-normalized by compute_archetype_correspondence.
                corr_markov_df = pd.DataFrame(corr_markov, index=src_labels, columns=tgt_labels)
                html_flow += report.df_to_html(corr_markov_df,
                    caption="Row-normalized correspondence (≈ Markov transition matrix). "
                            "Row sums = 1. Each cell = P(CMP archetype j | HSC archetype i). "
                            "Read row-by-row: 'given an HSC cell in archetype i, this is the probability "
                            "distribution over CMP archetypes after transport'.")

                # Markov transition heatmap (often more readable than the raw mass heatmap)
                fig_markov = go.Figure(data=go.Heatmap(
                    z=corr_markov, x=tgt_labels, y=src_labels,
                    colorscale="Blues",
                    zmin=0, zmax=1,
                    colorbar=dict(title="P(j|i)"),
                    text=[[f"{corr_markov[i,j]:.2f}" for j in range(K_c)] for i in range(K_h)],
                    texttemplate="%{text}",
                ))
                fig_markov.update_layout(
                    title=f"Markov transition matrix HSC→CMP (row-normalized, K={K_h}x{K_c})",
                    xaxis_title="CMP archetype (target)",
                    yaxis_title="HSC archetype (source)",
                    width=600, height=500,
                )
                html_flow += safe_plotly_html(report, fig_markov,
                    "Row-normalized correspondence as Markov transition heatmap (rows sum to 1)")

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
                # Threshold: keep links with FDR < 0.10 OR mass > 5% of total
                total_mass = corr.sum()
                s_idx, t_idx, vals, link_colors = [], [], [], []
                for i in range(K_h):
                    for j in range(K_c):
                        keep = (pair_fdr[i, j] < 0.10) or (corr[i, j] > total_mass * 0.05)
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
                    "Correspondence heatmap with FDR significance markers")
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
                    # Pair selection: z > 2 against per-pair shuffle null, OR FDR < 0.10.
                    # This replaces the old fixed 2%-mass threshold, which was too strict
                    # for near-uniform K_h x K_c grids (avg mass per pair ≈ 1/K_h*K_c).
                    top_cross_pairs_set = set()
                    for hi in range(K_h):
                        for ci in range(K_c):
                            if pair_z[hi, ci] > 2.0 or (pair_fdr is not None and pair_fdr[hi, ci] < 0.10):
                                top_cross_pairs_set.add((int(hi), int(ci)))
                    top_cross_pairs = sorted(top_cross_pairs_set,
                                             key=lambda p: -pair_z[p[0], p[1]])
                    n_z = int((pair_z > 2.0).sum())
                    n_fdr = int((pair_fdr < 0.10).sum()) if pair_fdr is not None else 0
                    log.info(f"Top cross-fit pairs (z>2 OR FDR<0.10): {len(top_cross_pairs)} pairs "
                             f"(z>2: {n_z}, FDR<0.10: {n_fdr})")
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
                                    "permutation null, or FDR < 0.10. Z-score = (observed mass - null mean) / null SD "
                                    "over 50 shuffle permutations. Wald Z = (beta_h - beta_c) / sqrt(SE_h^2 + SE_c^2), "
                                    "BH FDR over shared features.")
                    else:
                        html_flow += error_html("No cross-fit pairs passed z-score threshold (z > 2 or FDR < 0.10)")

                    # --- Per-pair zoomed flow_between() models for significant pairs ---
                    # For each (HSC arch hi, CMP arch ci) pair flagged by z > 2 or FDR < 0.10,
                    # train a small flow model from HSC cells hard-assigned to arch hi
                    # to CMP cells hard-assigned to arch ci. Then run gene alignment + Jacobian
                    # on the per-pair flow to get a "zoomed" view of which features drive the
                    # specific archetype-to-archetype transition.
                    html_flow += report.text("<h3>Per-pair zoomed flow models</h3>")
                    html_flow += report.text(
                        "<p>For each cross-fit pair with z &gt; 2 or FDR &lt; 0.10, we train "
                        "a per-pair <code>flow_within()</code> model restricted to HSC cells "
                        "hard-assigned to the source archetype and CMP cells hard-assigned to "
                        "the target archetype, then compute zoomed gene alignment and Jacobian "
                        "scores to identify features that specifically drive that archetype "
                        "transition.</p>")

                    if not top_cross_pairs_set:
                        html_flow += report.text(
                            "<p><i>No significant cross-fit pairs (z &gt; 2 or FDR &lt; 0.10). "
                            "Skipping per-pair zoomed flow models.</i></p>")
                    else:
                        # Limit to 3 pairs by z-score to keep runtime bounded.
                        # 6 was the original spec but per-pair flow + Jacobian is the heaviest
                        # block in the script; 3 keeps the additional cost under ~5 min.
                        MAX_PAIRS = 3
                        ranked_pairs = sorted(
                            top_cross_pairs_set,
                            key=lambda p: -pair_z[p[0], p[1]],
                        )
                        if len(ranked_pairs) > MAX_PAIRS:
                            log.info(
                                f"Per-pair flow_between: capping {len(ranked_pairs)} significant "
                                f"pairs to top {MAX_PAIRS} by z-score (runtime guard)."
                            )
                            html_flow += report.text(
                                f"<p><i>Note: {len(ranked_pairs)} pairs were significant; "
                                f"showing only the top {MAX_PAIRS} by z-score to bound runtime.</i></p>"
                            )
                            ranked_pairs = ranked_pairs[:MAX_PAIRS]

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

                                # Zoomed gene alignment (no permutation null — would be too slow)
                                try:
                                    align_pair = pc.tl.flow_gene_alignment(
                                        adata_full, fr_pair,
                                        n_top=10, per_cell=False,
                                        n_permutations=0, random_state=42,
                                    )
                                    top_aligned_pair = align_pair.get("top_aligned", [])[:10]
                                    top_opposed_pair = align_pair.get("top_opposed", [])[:10]
                                    a_scores = np.asarray(align_pair.get("alignment_scores", []))
                                    a_genes = list(align_pair.get("gene_names", []))
                                    if len(a_scores) > 0:
                                        # Build top-10 aligned table with scores
                                        aligned_rows = []
                                        for g in top_aligned_pair:
                                            try:
                                                idx_g = a_genes.index(g)
                                                aligned_rows.append({"Gene": g, "Alignment score": f"{a_scores[idx_g]:+.4f}"})
                                            except ValueError:
                                                aligned_rows.append({"Gene": g, "Alignment score": "?"})
                                        html_flow += report.df_to_html(
                                            pd.DataFrame(aligned_rows),
                                            caption=f"Top 10 aligned genes for HSC A{hi+1} → CMP A{ci+1} "
                                                    f"(zoomed flow). Score = cosine similarity between "
                                                    f"gene's PCA loading and per-pair flow velocity.")
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

                # --- STRAW PLOT: expansion (y) vs expression-along-flow (x) ---
                for panel_start in range(0, n_plot, 10):
                    panel_genes = gene_order[panel_start:panel_start + 10]
                    fig_straw, ax_s = plt.subplots(figsize=(10, 6))
                    n_fb_panel = 0
                    for ci, gi in enumerate(panel_genes):
                        gname = jac_gene_names[gi]
                        expansion_vals = per_cell[:, gi]
                        gvi = gene_var_idx[gi]
                        if gvi >= 0:
                            expr_vals = X_source[:, gvi]
                        else:
                            n_fb_panel += 1
                            log.warning(f"  Straw plot fallback for {gname} (panel {panel_start//10+1})")
                            expr_vals = flow_coord
                        bx = np.array([expr_vals[bin_idx == b].mean() if (bin_idx == b).any()
                                       else np.nan for b in range(n_bins)])
                        by = np.array([expansion_vals[bin_idx == b].mean() if (bin_idx == b).any()
                                       else np.nan for b in range(n_bins)])
                        valid = ~(np.isnan(bx) | np.isnan(by))
                        color = COLORS[ci % len(COLORS)]
                        if valid.sum() >= 2:
                            ax_s.plot(bx[valid], by[valid], color=color, linewidth=1.8,
                                      alpha=0.85, label=gname)
                            ax_s.scatter(bx[valid][0], by[valid][0], color=color, s=30,
                                         marker="o", zorder=5)
                            ax_s.scatter(bx[valid][-1], by[valid][-1], color=color, s=30,
                                         marker="^", zorder=5)
                    ax_s.axhline(1.0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
                    # Label x-axis based on what data is actually plotted.
                    # Delegates to module-level helper so the logic can be unit-tested.
                    n_mapped_panel = len(panel_genes) - n_fb_panel
                    _xlabel = _straw_plot_xlabel(
                        n_mapped_panel, n_fb_panel,
                        float(X_source.min()), float(X_source.max()),
                    )
                    ax_s.set_xlabel(_xlabel)
                    ax_s.set_ylabel("Expansion factor (Jacobian diagonal)")
                    ax_s.legend(fontsize=7, loc="best", ncol=2, framealpha=0.7)
                    ax_s.spines[["top", "right"]].set_visible(False)
                    sig_note = "(significant)" if used_sig else "(top by effect)"
                    ax_s.set_title(f"Straw plot: expansion vs expression {sig_note} "
                                   f"(genes {panel_start+1}-{panel_start+len(panel_genes)}, "
                                   f"{n_fb_panel} fallback to flow_coord)")
                    fig_straw.tight_layout()
                    html_genes += report.fig_to_img(fig_straw,
                        caption=f"Straw plot: each line=gene, ○=flow start, ▲=flow end. "
                                f"y=1 is neutral; above=expansion, below=contraction. "
                                f"X-axis is binned mean logcounts within flow_coord bins.")
                    plt.close("all")

                # --- RIDGEPLOT: KDE-filled stacked ridges (Task 18) ---
                # Each ridge shows the distribution of per-cell Jacobian expansion values
                # for one gene, stacked vertically with y-offsets. Uses scipy KDE + fill_between.
                # Falls back to line plot if fewer than 5 values are available per ridge.
                from scipy.stats import gaussian_kde
                for panel_start in range(0, n_plot, 10):
                    panel_genes = gene_order[panel_start:panel_start + 10]
                    n_ridges = len(panel_genes)
                    fig_height = max(4, n_ridges * 0.9 + 1.5)
                    fig_ridge, ax_r = plt.subplots(figsize=(10, fig_height))
                    ridge_spacing = 1.0   # vertical spacing between ridge baselines
                    x_kde = np.linspace(0.0, 1.0, 200)  # x axis = flow coord (0→1)
                    any_kde_drawn = False
                    for ci, gi in enumerate(panel_genes):
                        gname = jac_gene_names[gi]
                        expansion_vals = per_cell[:, gi]
                        # Use flow_coord bins to get mean expansion per position
                        by = np.array([expansion_vals[bin_idx == b].mean() if (bin_idx == b).any()
                                       else np.nan for b in range(n_bins)])
                        valid_mask = ~np.isnan(by)
                        valid_vals = by[valid_mask]
                        valid_x = bin_centers[valid_mask]
                        color = COLORS[ci % len(COLORS)]
                        y_offset = (n_ridges - 1 - ci) * ridge_spacing
                        if valid_vals.size >= 5:
                            # KDE over the expansion values; evaluate at uniform x_kde grid.
                            # Bandwidth via Scott's rule (default).
                            try:
                                kde = gaussian_kde(valid_vals, bw_method="scott")
                                # Map KDE density to y-range by evaluating at uniformly spaced
                                # expansion values; scale to fit within ridge_spacing * 0.8.
                                exp_grid = np.linspace(valid_vals.min(), valid_vals.max(), 200)
                                density = kde(exp_grid)
                                density_norm = density / max(density.max(), 1e-10) * ridge_spacing * 0.8
                                # x-axis = expansion value range, shifted/scaled to [0,1] flow coord
                                # for display; we map exp_grid → [0,1] for side-by-side comparison.
                                exp_min, exp_max = valid_vals.min(), valid_vals.max()
                                if exp_max - exp_min > 1e-6:
                                    x_plot = (exp_grid - exp_min) / (exp_max - exp_min)
                                else:
                                    x_plot = np.linspace(0, 1, len(exp_grid))
                                ax_r.fill_between(x_plot, y_offset, y_offset + density_norm,
                                                  color=color, alpha=0.55)
                                ax_r.plot(x_plot, y_offset + density_norm, color=color,
                                          linewidth=1.2, alpha=0.9)
                                # Mark the mean expansion with a vertical tick
                                mean_exp = valid_vals.mean()
                                mean_x = (mean_exp - exp_min) / max(exp_max - exp_min, 1e-6)
                                ax_r.vlines(mean_x, y_offset, y_offset + ridge_spacing * 0.5,
                                            color=color, linewidth=1.0, linestyle="--", alpha=0.7)
                                any_kde_drawn = True
                            except Exception:
                                # KDE failed (e.g., constant expansion) → fall back to line
                                if valid_vals.size >= 2:
                                    ax_r.plot(valid_x, y_offset + valid_vals - valid_vals.mean(),
                                              color=color, linewidth=1.5, alpha=0.8)
                        elif valid_vals.size >= 2:
                            # Too few points for KDE — fall back to simple line
                            ax_r.plot(valid_x, y_offset + valid_vals - valid_vals.mean(),
                                      color=color, linewidth=1.5, alpha=0.8)
                        # Gene label on the left
                        ax_r.text(-0.02, y_offset + ridge_spacing * 0.3, gname,
                                  ha="right", va="center", fontsize=8, color=color,
                                  transform=ax_r.get_yaxis_transform())
                    # x-axis is normalised expansion value (0=min, 1=max per gene)
                    ax_r.set_xlabel("Normalised expansion value (0=min, 1=max per gene)")
                    ax_r.set_ylabel("")
                    ax_r.set_yticks([])
                    ax_r.spines[["top", "right", "left"]].set_visible(False)
                    ax_r.set_title(f"Filled ridgeplot: Jacobian expansion distribution {sig_note} "
                                   f"(genes {panel_start+1}-{panel_start+len(panel_genes)})")
                    fig_ridge.tight_layout()
                    html_genes += report.fig_to_img(fig_ridge,
                        caption="Filled ridgeplot: each ridge = KDE of per-cell Jacobian expansion "
                                "values for one gene. Dashed tick = mean expansion. "
                                "x-axis is normalised to [0,1] per gene for shape comparison. "
                                "Falls back to line trace if <5 binned values available.")
                    plt.close("all")

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

            # --- PER-PAIR FLOW MODELS: top 5 archetype pairs by mass ---
            if corr_matrix is not None:
                html_genes += "<h4>Per-pair archetype flow models</h4>"
                html_genes += report.text(
                    "<b>Per-pair flow models</b>: For HSC→CMP archetype pairs with z > 2 against the "
                    "per-pair shuffle null, or FDR < 0.10 (up to 5 pairs), train a small flow model "
                    "between source cells (HSC archetype assigned) and target cells (CMP archetype assigned). "
                    "Per-pair Jacobian gives archetype-level interpretation that the global flow lacks.")

                # Get top pairs by z-score vs shuffle null (same criterion as cross-fit Wald section),
                # capped at 5 to limit compute. Falls back to top 5 by mass if pair_z unavailable.
                K_h_local, K_c_local = corr_matrix.shape
                if 'pair_z' in locals() and pair_z is not None and pair_z.shape == (K_h_local, K_c_local):
                    top_pp_pairs = sorted(
                        [(hi, ci) for hi in range(K_h_local) for ci in range(K_c_local)
                         if pair_z[hi, ci] > 2.0 or (pair_fdr is not None and pair_fdr[hi, ci] < 0.10)],
                        key=lambda p: -pair_z[p[0], p[1]]
                    )[:5]
                    log.info(f"Per-pair Jacobian: {len(top_pp_pairs)} pairs selected (z>2 OR FDR<0.10)")
                else:
                    flat_idx = np.argsort(corr_matrix.ravel())[::-1]
                    top_pp_pairs = [(int(hi), int(ci)) for fi in flat_idx[:5]
                                    for hi, ci in [divmod(fi, K_c_local)]]
                    log.info(f"Per-pair Jacobian: {len(top_pp_pairs)} pairs selected (top-5 mass fallback)")

                pp_summary_rows = []
                for hi, ci in top_pp_pairs:
                    log.info(f"Per-pair flow model: HSC A{hi+1} -> CMP A{ci+1}")
                    try:
                        # Build masks: source = HSC cells assigned to archetype hi, target = CMP cells in CMP archetype ci
                        adata_pair = adata_full.copy()
                        # HSC source: uses HSC-model assignment
                        hsc_arch_label = f"archetype_{hi}"
                        # We need CMP-model archetype assignments for CMP cells.
                        # adata_target_cmp (built earlier for correspondence matrix) has these.
                        # Build a lookup: obs_name → CMP archetype argmax
                        cmp_arch_assign_lookup = {}
                        if 'adata_target_cmp' in locals() and weights_tgt_cmp is not None:
                            for cn, wv in zip(adata_target_cmp.obs_names, weights_tgt_cmp):
                                cmp_arch_assign_lookup[cn] = int(np.argmax(wv))
                        cmp_assign = []
                        for cn in adata_pair.obs_names:
                            if cn in cmp_arch_assign_lookup:
                                cmp_assign.append(f"cmp_arch_{cmp_arch_assign_lookup[cn]}")
                            else:
                                cmp_assign.append("not_in_cmp")
                        adata_pair.obs["cmp_archetype"] = cmp_assign
                        adata_pair.obs["pair_role"] = "neither"
                        src_role_mask = (adata_pair.obs["cell_type"] == "hematopoietic stem cell") & \
                                        (adata_pair.obs["archetypes"] == hsc_arch_label)
                        tgt_role_mask = (adata_pair.obs["cell_type"] == "common myeloid progenitor") & \
                                        (adata_pair.obs["cmp_archetype"] == f"cmp_arch_{ci}")
                        adata_pair.obs.loc[src_role_mask, "pair_role"] = "source"
                        adata_pair.obs.loc[tgt_role_mask, "pair_role"] = "target"

                        n_src_pp = int(src_role_mask.sum())
                        n_tgt_pp = int(tgt_role_mask.sum())
                        log.info(f"  N source: {n_src_pp}, N target: {n_tgt_pp}")
                        if n_src_pp < 30 or n_tgt_pp < 30:
                            pp_summary_rows.append({
                                "Pair": f"H A{hi+1} → C A{ci+1}",
                                "N src": n_src_pp,
                                "N tgt": n_tgt_pp,
                                "Status": "skip (too few cells)",
                            })
                            continue

                        fr_pp = pc.tl.flow_within(
                            adata_pair,
                            source={"pair_role": "source"},
                            target={"pair_role": "target"},
                            n_epochs=150, hidden_dims=(64, 64),
                            batch_size=64, return_model=True,
                            name=f"H{hi+1}_to_C{ci+1}", random_state=42,
                        )
                        # Per-pair Jacobian (smaller, fewer perms)
                        jac_pp = pc.tl.flow_jacobian(
                            adata_pair, fr_pp, fr_pp.get("model"),
                            per_cell_features=True, n_top_features=200,
                            n_permutations=100, null_type="both", permutation_seed=42)
                        per_cell_pp = jac_pp.get("per_cell_expansion")
                        gnames_pp = jac_pp.get("per_cell_expansion_gene_names", [])
                        exp_p_pp = jac_pp.get("expansion_pvalues_fdr")
                        n_sig_exp_pp = 0
                        top_genes_pp_str = ""
                        if per_cell_pp is not None and len(gnames_pp) > 0:
                            mean_exp_pp = per_cell_pp.mean(axis=0)
                            order_pp = np.argsort(np.abs(mean_exp_pp - 1))[::-1][:5]
                            top_genes_pp_str = ", ".join(gnames_pp[idx] for idx in order_pp)
                            if exp_p_pp is not None:
                                gene_idx_pp = jac_pp.get("per_cell_expansion_gene_indices")
                                if gene_idx_pp is not None:
                                    sub_pp = np.asarray(exp_p_pp)[gene_idx_pp]
                                    n_sig_exp_pp = int((sub_pp < 0.05).sum())
                        # Per-pair W2 distance
                        try:
                            src_pca_pp = adata_pair.obsm[fr_pp.get("pca_key", "X_pca")][fr_pp["source_mask"]]
                            tgt_pca_pp = adata_pair.obsm[fr_pp.get("pca_key", "X_pca")][fr_pp["target_mask"]]
                            w2_b_pp = wasserstein2_distance(src_pca_pp, tgt_pca_pp, max_n=1000)
                            w2_a_pp = wasserstein2_distance(fr_pp["transported"], tgt_pca_pp, max_n=1000)
                        except Exception:
                            w2_b_pp = w2_a_pp = float("nan")
                        pp_summary_rows.append({
                            "Pair": f"H A{hi+1} → C A{ci+1}",
                            "N src": n_src_pp,
                            "N tgt": n_tgt_pp,
                            "W2 before (PC)": f"{w2_b_pp:.3f}",
                            "W2 after (PC)": f"{w2_a_pp:.3f}",
                            "W2 reduction": f"{(w2_b_pp - w2_a_pp):.3f}",
                            "MMD before": f"{fr_pp['mmd_before']:.4f}",
                            "MMD after": f"{fr_pp['mmd_after']:.4f}",
                            "MMD reduction": f"{(1 - fr_pp['mmd_after']/max(fr_pp['mmd_before'], 1e-10)):.1%}",
                            "Exp FDR<0.05": n_sig_exp_pp,
                            "Top expanding/contracting": top_genes_pp_str,
                        })

                        # Per-pair Jacobian heatmap
                        try:
                            fig_jac_pp = pc.pl.jacobian_heatmap(adata_pair, jac_pp, show=False)
                            html_genes += safe_plotly_html(report, fig_jac_pp,
                                f"Per-pair Jacobian heatmap: HSC A{hi+1} → CMP A{ci+1}")
                        except Exception as e:
                            log.exception(f"Per-pair jacobian heatmap failed for {hi},{ci}")
                    except Exception as e:
                        log.exception(f"Per-pair flow model failed: H A{hi+1} -> C A{ci+1}")
                        pp_summary_rows.append({
                            "Pair": f"H A{hi+1} → C A{ci+1}",
                            "Status": f"failed: {e}",
                        })

                if pp_summary_rows:
                    html_genes += report.df_to_html(pd.DataFrame(pp_summary_rows),
                        caption="Per-pair archetype flow model summary (top 5 by transport mass)")

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

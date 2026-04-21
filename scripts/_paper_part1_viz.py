"""Shared visualization helpers for Paper Part 1 reports.

Exposes:
    build_drift_qc_panel(results_list, drift_threshold=0.01, converged_window=10)
        Render an HTML fragment (drift curve figure + summary table + badge)
        surfacing archetype drift/stability metrics already tracked in
        TrainingResults['history'] by train_archetypal().

    convergence_status(history, max_epochs, early_stop_triggered,
                       actual_epochs, window=10, delta_threshold=0.01)
        Classify a final training run as CONVERGED / NON_CONVERGED_HIT_CAP /
        NOT_CONVERGED_INSUFFICIENT_HISTORY from its loss history. Used by the
        paper scripts to render the Convergence QC badge (W-A8 fix).

Used by:
    scripts/run_paper_part1_hsc.py  (HSC + CMP models)
    scripts/run_paper_part1_ov.py   (primary + metastatic models)

Background:
    PEACH training logs per-epoch archetype position drift under keys:
        - archetype_drift_mean / _max / _std  (L2 movement epoch-over-epoch)
        - archetype_stability_mean / _min     (inverse-drift stability score)
        - archetype_variance_mean             (variance over sliding window)
    These metrics are visible in stdout but not in the HTML QC report.
    W-A7 surfaces them so W-A6's "archetypes drifting away from PCHA init"
    hypothesis is immediately visible at-a-glance.
"""

from __future__ import annotations

import base64
import io
from typing import Iterable, List, Sequence, Tuple


def _final_value(seq, default=float("nan")):
    """Return the last entry of a history sequence, or default if empty/None."""
    if seq is None:
        return default
    try:
        if len(seq) == 0:
            return default
        return float(seq[-1])
    except (TypeError, ValueError):
        return default


def dotplot_figsize(long_df, y_col="gene", *, base=(12.0, 8.0),
                     per_row=0.30, floor=6.0, ceiling=48.0, width=None):
    """Compute a ``(width, height)`` figsize scaling with number of y rows.

    The Wilcoxon path through ``pc.pl.dotplot`` auto-expands when the row
    count is large, but the simplex-regression path crams many rows into
    the default height. Passing an explicit ``figsize`` derived from this
    helper prevents y-axis label crowding while preserving the base width.

    Parameters
    ----------
    long_df : pandas.DataFrame
        Long-format dataframe being passed to ``pc.pl.dotplot``.
    y_col : str
        Name of the row column (dotplot y-axis).
    base : (float, float)
        Default base ``(width, height)``.
    per_row : float
        Vertical inches allocated per unique y-row.
    floor : float
        Minimum height in inches.
    ceiling : float
        Maximum height in inches (safety cap for pathway names).
    width : float or None
        Override width. ``None`` uses ``base[0]``.

    Returns
    -------
    (width, height) tuple ready for ``pc.pl.dotplot(..., figsize=...)``.
    """
    if long_df is None or len(long_df) == 0 or y_col not in long_df.columns:
        return base
    n_rows = int(long_df[y_col].nunique())
    h = max(floor, min(ceiling, n_rows * per_row + 2.0))
    w = float(width) if width is not None else float(base[0])
    return (w, h)


def _adjust_labels(ax, texts, *, x=None, y=None):
    """Repel overlapping matplotlib ``Text`` labels with adjustText.

    Silently no-ops if adjustText is not installed or errors out — labels
    remain at their initial positions (still legible for small N, just
    potentially overlapping for large N). ``x`` / ``y`` may be passed so
    adjustText can draw leader lines back to the original points.
    """
    if not texts:
        return
    try:
        from adjustText import adjust_text  # type: ignore
        kw = dict(ax=ax,
                  arrowprops=dict(arrowstyle="-", color="#888888",
                                  lw=0.5, alpha=0.5),
                  expand_text=(1.05, 1.15),
                  expand_points=(1.05, 1.15),
                  force_text=(0.3, 0.5),
                  force_points=(0.2, 0.4),
                  only_move={"points": "xy", "text": "xy"})
        if x is not None and y is not None:
            adjust_text(texts, x=list(x), y=list(y), **kw)
        else:
            adjust_text(texts, **kw)
    except Exception:
        # adjustText missing or failed — leave labels where they are.
        pass


def _window_mean(seq, window):
    """Mean of the last `window` entries of seq, or NaN if insufficient data."""
    import math
    if seq is None:
        return float("nan")
    try:
        n = len(seq)
    except TypeError:
        return float("nan")
    if n == 0:
        return float("nan")
    tail = list(seq[-min(window, n):])
    if not tail:
        return float("nan")
    return float(sum(tail) / len(tail))


def _window_median(seq, window):
    """Median of the last `window` entries of seq, or NaN if insufficient data."""
    import math
    if seq is None:
        return float("nan")
    try:
        n = len(seq)
    except TypeError:
        return float("nan")
    if n == 0:
        return float("nan")
    tail = sorted(list(seq[-min(window, n):]))
    if not tail:
        return float("nan")
    m = len(tail)
    if m % 2 == 1:
        return float(tail[m // 2])
    return float(0.5 * (tail[m // 2 - 1] + tail[m // 2]))


def _is_converged(drift_mean_history, threshold, window):
    """Converged iff median of last `window` drift_mean values <= threshold.

    Uses median (not mean) to be robust to transient spikes earlier in the
    window — HSC training can have a single large drift epoch that lingers
    in the window but does not reflect the final stability.

    Returns False for empty / NaN histories so we never claim convergence
    on missing data.
    """
    import math
    med = _window_median(drift_mean_history, window)
    if med is None or math.isnan(med):
        return False
    return med <= threshold


def _build_drift_figure(results_list, dpi=130):
    """Matplotlib drift curve figure: one line per model, with min/max band.

    X axis: epoch index (1-based).
    Y axis: archetype_drift_mean (shaded envelope from drift_max).
    """
    # Lazy import so tests do not pay for matplotlib setup unless they hit
    # this function (they always do, but keeping the import local follows
    # the pattern in scripts/run_paper_part1_hsc.py).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=dpi)

    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"]
    any_data = False
    for idx, (label, res) in enumerate(results_list):
        history = (res or {}).get("history", {}) or {}
        drift_mean = list(history.get("archetype_drift_mean", []) or [])
        drift_max = list(history.get("archetype_drift_max", []) or [])
        if not drift_mean:
            continue
        any_data = True
        epochs = list(range(1, len(drift_mean) + 1))
        color = colors[idx % len(colors)]
        ax.plot(epochs, drift_mean, label=f"{label} drift_mean",
                color=color, linewidth=2.0)
        if drift_max and len(drift_max) == len(drift_mean):
            ax.fill_between(epochs, drift_mean, drift_max,
                            color=color, alpha=0.18,
                            label=f"{label} drift_max envelope")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Archetype drift (L2 per epoch)")
    ax.set_title("Archetype position drift per epoch")
    if any_data:
        ax.legend(loc="best", fontsize=8, frameon=True)
    else:
        ax.text(0.5, 0.5, "No drift history recorded",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="#888888")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    return fig


def _fig_to_base64_img(fig, dpi=130):
    """Serialize a matplotlib figure to an HTML <img> tag (base64 PNG)."""
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return (
        f'<img src="data:image/png;base64,{b64}" '
        f'style="max-width:100%;" '
        f'alt="Archetype drift curve">'
    )


def _badge_html(text, color):
    """Small inline badge with background color."""
    return (
        f'<span style="display:inline-block; padding:4px 12px; '
        f'border-radius:12px; background:{color}; color:white; '
        f'font-weight:700; font-size:0.9em; margin-right:8px;">{text}</span>'
    )


def _summary_table_html(rows):
    """Minimal summary HTML table matching the styled-table class used by
    the HTMLReport helper in the paper scripts."""
    cols = [
        "model",
        "n_epochs",
        "final_drift_mean",
        "final_drift_max",
        "last10_drift_median",
        "final_stability_mean",
        "converged",
    ]
    header = "".join(f"<th>{c}</th>" for c in cols)
    body_rows = []
    for r in rows:
        cells = "".join(f"<td>{r.get(c, '')}</td>" for c in cols)
        body_rows.append(f"<tr>{cells}</tr>")
    body = "".join(body_rows) or "<tr><td colspan='5'>No data</td></tr>"
    return (
        f'<table class="styled-table">'
        f'<thead><tr>{header}</tr></thead>'
        f'<tbody>{body}</tbody>'
        f'</table>'
    )


def build_drift_qc_panel(
    results_list: Sequence[Tuple[str, dict]],
    *,
    drift_threshold: float = 0.01,
    converged_window: int = 10,
) -> str:
    """Build the drift/stability QC panel HTML fragment.

    Parameters
    ----------
    results_list : sequence of (label, TrainingResults-like dict)
        Each dict is expected to have a ``history`` sub-dict containing the
        keys ``archetype_drift_mean``, ``archetype_drift_max``,
        ``archetype_stability_mean`` (and optionally drift_std,
        stability_min, variance_mean). Missing keys degrade gracefully to
        NaN cells and an empty curve.
    drift_threshold : float, default 0.01
        A model is flagged "STABLE" iff its median drift_mean over the last
        ``converged_window`` epochs falls at or below this threshold.
        Using the median (not mean) makes the badge robust to transient
        drift spikes that can occur earlier in the window but do not
        reflect the final training state.
        Otherwise it is flagged "DRIFTING".
    converged_window : int, default 10
        How many trailing epochs to average for the convergence test.

    Returns
    -------
    html : str
        Self-contained HTML fragment:
          - One or more stability badges (STABLE / DRIFTING) per model
          - Base64-embedded drift curve figure
          - Summary table with columns:
            model, final_drift_mean, final_drift_max, final_stability_mean,
            converged
        Intended to be dropped directly into an HTMLReport section body.
    """
    import math

    # ---- Figure ----
    fig = _build_drift_figure(results_list)
    img_html = _fig_to_base64_img(fig)

    # ---- Badges + summary rows ----
    badges_html = ""
    rows = []
    for label, res in results_list:
        history = (res or {}).get("history", {}) or {}
        dmean_hist = history.get("archetype_drift_mean", [])
        dmax_hist = history.get("archetype_drift_max", [])
        smean_hist = history.get("archetype_stability_mean", [])

        final_dmean = _final_value(dmean_hist)
        final_dmax = _final_value(dmax_hist)
        final_smean = _final_value(smean_hist)
        # Fallback: archetype_stability_mean is in the training metrics whitelist
        # but never computed in the training loop (only drift_mean is tracked).
        # Derive stability as the inverse-drift approximation so the summary
        # table always shows a meaningful value instead of "N/A".
        if math.isnan(final_smean) and not math.isnan(final_dmean):
            final_smean = 1.0 / (1.0 + final_dmean)

        # Expose the actual median value so we can see what the badge is
        # comparing against (r14 bug: CMP final_drift_mean=0.00001 but flag
        # still DRIFTING — suggests the median over the window is getting
        # dragged by a spike within the window, not just an outlier outside
        # it).
        last10_median = _window_median(dmean_hist, converged_window)
        n_epochs = len(dmean_hist) if dmean_hist is not None else 0
        converged = _is_converged(dmean_hist, drift_threshold, converged_window)
        if converged:
            badges_html += _badge_html(f"{label}: STABLE", "#2ca02c")
        else:
            badges_html += _badge_html(f"{label}: DRIFTING", "#c0392b")

        # Diagnostic stderr line — helps reconcile the badge with the raw
        # history when the answer is surprising.
        import sys as _sys
        try:
            _tail_preview = list(dmean_hist[-converged_window:])
            print(
                f"[drift-qc] {label}: n_epochs={n_epochs}, "
                f"last10_median={last10_median:.6f}, "
                f"threshold={drift_threshold}, converged={converged}, "
                f"last10_values={[f'{v:.5f}' for v in _tail_preview]}",
                file=_sys.stderr, flush=True,
            )
        except Exception:
            pass

        def _fmt(v):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "N/A"
            return f"{v:.5f}"

        rows.append({
            "model": label,
            "n_epochs": str(n_epochs),
            "final_drift_mean": _fmt(final_dmean),
            "final_drift_max": _fmt(final_dmax),
            "last10_drift_median": _fmt(last10_median),
            "final_stability_mean": _fmt(final_smean),
            "converged": "Y" if converged else "N",
        })

    table_html = _summary_table_html(rows)

    caption = (
        f"Archetype drift per epoch (L2 movement of archetype positions). "
        f"A model is flagged STABLE when the <b>median</b> of its last "
        f"{converged_window} drift_mean values is at or below "
        f"{drift_threshold}. Using the median (not mean) makes the badge "
        f"robust to transient drift spikes that can linger in the window "
        f"but do not reflect the final training state. Otherwise it is "
        f"DRIFTING — archetypes are still moving and the fit may be "
        f"dragging them away from their PCHA init."
    )

    panel = (
        '<div class="drift-qc-panel">'
        '<h3>Archetype drift &amp; stability QC (W-A7)</h3>'
        f'<div class="badges" style="margin:8px 0 12px 0;">{badges_html}</div>'
        f'{img_html}'
        f'<p class="caption">{caption}</p>'
        f'{table_html}'
        '</div>'
    )
    return panel


# ---------------------------------------------------------------------------
# W-A8: convergence flag helper
# ---------------------------------------------------------------------------

def convergence_status(
    history: dict,
    max_epochs: int,
    early_stop_triggered: bool,
    actual_epochs: int,
    *,
    window: int = 10,
    delta_threshold: float = 0.01,
    loss_key: str = "archetypal_loss",
) -> Tuple[str, float]:
    """Classify a final training run as converged or not from its loss history.

    This fixes the r9 bug where training runs with near-zero ``delta_loss``
    were still flagged NON-CONVERGED solely because ``actual_epochs ==
    max_epochs`` and the built-in early stopper did not fire. The textbook
    definition of "converged" is "loss is no longer decreasing"; if the loss
    has plateaued, the model is converged regardless of whether the epoch
    cap was hit.

    The classification rules (evaluated in order):

    1. If ``len(history['loss']) < 2`` (cannot form a diff) →
       ``"NOT_CONVERGED_INSUFFICIENT_HISTORY"`` with ``delta_loss_mean=NaN``.
    2. Compute ``delta_loss_mean`` = mean of ``|diff(loss[-window:])|``
       (last ``window`` epochs, or fewer if the run was shorter).
    3. If ``delta_loss_mean <= delta_threshold`` → ``"CONVERGED"`` (loss has
       plateaued).
    4. Elif ``early_stop_triggered`` is True → ``"CONVERGED"`` (trust the
       training-loop early stopper even if the window mean is slightly
       above threshold, because it saw the full history).
    5. Elif ``actual_epochs >= max_epochs`` → ``"NON_CONVERGED_HIT_CAP"``
       (ran to the cap and loss is still dropping faster than the
       threshold).
    6. Else → ``"CONVERGED"`` (training finished short of the cap without
       early stopping — unusual but we treat a clean exit as converged).

    Parameters
    ----------
    history : dict
        TrainingResults-style history dict. Must contain key ``"loss"``
        mapping to a sequence of per-epoch loss values. Missing key is
        treated as an empty list → insufficient history.
    max_epochs : int
        The epoch cap passed to ``train_archetypal`` (``n_epochs``).
    early_stop_triggered : bool
        Whether the training loop's built-in early stopping fired.
    actual_epochs : int
        Number of epochs the run actually executed (may be less than
        ``max_epochs`` if early stopping fired).
    window : int, default 10
        Number of trailing loss entries to use for the delta calculation.
    delta_threshold : float, default 0.01
        If ``mean(|Δloss|)`` over the last ``window`` entries falls at or
        below this threshold, the run is considered converged.

    Returns
    -------
    status : str
        One of ``"CONVERGED"``, ``"NON_CONVERGED_HIT_CAP"``, or
        ``"NOT_CONVERGED_INSUFFICIENT_HISTORY"``.
    delta_loss_mean : float
        The mean absolute per-epoch loss change over the last ``window``
        epochs (or NaN when history has fewer than 2 entries).
    """
    import math

    # r14-item-82: default to the ``archetypal_loss`` (reconstruction)
    # history instead of the combined ``loss`` so convergence is judged on
    # the reconstruction signal alone. The combined loss includes the KLD
    # term, which can continue to drift as the encoder variance settles
    # even after archetypal reconstruction has plateaued — producing
    # spurious NON_CONVERGED_HIT_CAP flags. Fallback to the combined
    # ``loss`` if the model was trained without per-component history or
    # if the requested key is absent.
    hist = history or {}
    losses = list(hist.get(loss_key, []) or [])
    if len(losses) < 2:
        losses = list(hist.get("loss", []) or [])

    # Rule 1: need at least 2 loss entries to compute any delta.
    if len(losses) < 2:
        return ("NOT_CONVERGED_INSUFFICIENT_HISTORY", float("nan"))

    # Rule 2: compute mean |Δloss| over the last `window` entries (or fewer).
    tail = losses[-window:] if len(losses) >= window else losses
    diffs = [abs(tail[i + 1] - tail[i]) for i in range(len(tail) - 1)]
    if not diffs:
        # Shouldn't happen given len(losses) >= 2, but guard defensively.
        return ("NOT_CONVERGED_INSUFFICIENT_HISTORY", float("nan"))
    delta_loss_mean = float(sum(diffs) / len(diffs))

    if math.isnan(delta_loss_mean):
        return ("NOT_CONVERGED_INSUFFICIENT_HISTORY", float("nan"))

    # Rule 3: loss has plateaued → converged.
    if delta_loss_mean <= delta_threshold:
        return ("CONVERGED", delta_loss_mean)

    # Rule 4: early stopper fired → trust it.
    if early_stop_triggered:
        return ("CONVERGED", delta_loss_mean)

    # Rule 5: hit the epoch cap without plateauing or early-stopping.
    if actual_epochs >= max_epochs:
        return ("NON_CONVERGED_HIT_CAP", delta_loss_mean)

    # Rule 6: clean early exit short of cap, no early stopper, but delta
    # above threshold. This is an unusual edge case — treat as converged
    # (the training loop terminated normally).
    return ("CONVERGED", delta_loss_mean)


def build_r2_vs_fdr_scatter(
    r2_values,
    fdr_values,
    feature_names,
    *,
    r2_threshold: float = 0.1,
    fdr_threshold: float = 0.05,
    n_labels: int = 20,
    title: str = "Feature R² vs -log10(FDR)",
):
    """R²-based position-dependence scatter for feature simplex regression (W-B18).

    Replaces the Wald delta-beta volcano's x-axis with per-feature R².
    Each point is one feature; x = simplex regression R² (fraction of
    variance explained by archetype position), y = -log10(minimum FDR
    across the archetype vertex contrasts). Top-right corner = genes
    that are both significantly position-dependent AND well-explained.

    Parameters
    ----------
    r2_values : array-like, shape [n_features]
        Per-feature simplex regression R² (e.g.,
        ``reg_result["r_squared_degree1"]`` or ``reg_result["r_squared"]``).
    fdr_values : array-like, shape [n_features]
        Per-feature minimum vertex FDR across archetypes (use
        ``reg_result["vertex_pvalues_fdr"].min(axis=1)``).
    feature_names : list[str]
        Feature labels aligned with r2_values / fdr_values.
    r2_threshold : float, default 0.1
        Vertical dashed line at this R².
    fdr_threshold : float, default 0.05
        Horizontal dashed line at -log10(this FDR).
    n_labels : int, default 20
        Number of top-right (high R² and low FDR) features to label.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r2 = np.asarray(r2_values, dtype=np.float64)
    fdr = np.asarray(fdr_values, dtype=np.float64)
    fdr_safe = np.clip(fdr, 1e-300, 1.0)
    neg_log_fdr = -np.log10(fdr_safe)
    names = list(feature_names)

    if len(r2) != len(fdr) or len(r2) != len(names):
        raise ValueError(
            f"Length mismatch: r2={len(r2)}, fdr={len(fdr)}, "
            f"names={len(names)}"
        )

    # Color points by whether they cross both thresholds
    sig_mask = (r2 > r2_threshold) & (fdr < fdr_threshold)
    colors = np.where(sig_mask, "#d62728", "#888888")

    fig, ax = plt.subplots(figsize=(7.2, 5.5), dpi=130)
    ax.scatter(r2, neg_log_fdr, s=14, c=colors, alpha=0.55,
               edgecolor="none")
    ax.axvline(r2_threshold, linestyle="--", color="#666666",
               linewidth=0.9, alpha=0.7)
    ax.axhline(-np.log10(fdr_threshold), linestyle="--",
               color="#666666", linewidth=0.9, alpha=0.7)

    # Label top-right points: sort by R² * -log10(fdr) score among
    # passing features, take top n_labels. adjustText repels overlapping
    # labels so crowded top-right corners stay legible.
    if n_labels > 0 and sig_mask.any():
        score = r2 * neg_log_fdr
        score[~sig_mask] = -np.inf
        top_idx = np.argsort(-score)[: min(n_labels, int(sig_mask.sum()))]
        texts = []
        tx = []
        ty = []
        for i in top_idx:
            tx.append(float(r2[i]))
            ty.append(float(neg_log_fdr[i]))
            texts.append(ax.text(
                float(r2[i]), float(neg_log_fdr[i]), names[i],
                fontsize=7, color="#1f1f1f", alpha=0.85,
            ))
        _adjust_labels(ax, texts, x=tx, y=ty)

    ax.set_xlabel(
        "Feature R² (simplex regression: variance explained by "
        "archetype position)"
    )
    ax.set_ylabel("-log10(min vertex FDR)")
    n_sig = int(sig_mask.sum())
    ax.set_title(
        f"{title}\n{n_sig} features with R²>{r2_threshold} and "
        f"FDR<{fdr_threshold}"
    )
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    return fig


def compute_cross_model_r2(
    weights,
    archetypes,
    original_coords,
) -> float:
    """Cross-model archetypal R² — "how well does model A explain data B?" (W-B12).

    Computes:

        reconstruction = weights @ archetypes
        R² = 1 - ||original - reconstruction||² / ||original - mean(original)||²

    This is the standardized archetypal R² formulation (same as
    peach._core.utils.metrics.calculate_archetype_r2) but in pure numpy
    so it can be called on held-out / projected cells without requiring
    a torch forward pass.

    The HSC script's degradation test projects CMP cells through the HSC
    model; this function then quantifies how poorly the HSC archetypes
    explain the CMP reconstruction. A large drop from HSC-native R² to
    CMP-via-HSC R² is the real "degradation" signal — before W-B12 the
    degradation section only printed KS distributions and never reported
    the numerical drop.

    Parameters
    ----------
    weights : np.ndarray [n_cells, K]
        Barycentric archetype weights (rows should sum to ~1 but no
        constraint is enforced here).
    archetypes : np.ndarray [K, D]
        Archetype coordinates in the same space as ``original_coords``.
    original_coords : np.ndarray [n_cells, D]
        The ground-truth cell coordinates (e.g., ``adata.obsm['X_pca']``).

    Returns
    -------
    float
        R² value. 1.0 = perfect reconstruction, 0.0 = no better than
        the per-feature mean, negative = worse than the mean.

    Raises
    ------
    ValueError
        If shapes don't line up.
    """
    import numpy as np

    weights = np.asarray(weights, dtype=np.float64)
    archetypes = np.asarray(archetypes, dtype=np.float64)
    original = np.asarray(original_coords, dtype=np.float64)

    if weights.ndim != 2 or archetypes.ndim != 2 or original.ndim != 2:
        raise ValueError(
            "All inputs must be 2-D ndarrays; got shapes "
            f"weights={weights.shape}, archetypes={archetypes.shape}, "
            f"original={original.shape}"
        )
    if weights.shape[0] != original.shape[0]:
        raise ValueError(
            f"weights rows ({weights.shape[0]}) must match original rows "
            f"({original.shape[0]})"
        )
    if weights.shape[1] != archetypes.shape[0]:
        raise ValueError(
            f"weights cols ({weights.shape[1]}) must match archetypes rows "
            f"({archetypes.shape[0]})"
        )
    # archetypes.shape[1] must match original.shape[1]; trim to min if off.
    D = min(archetypes.shape[1], original.shape[1])
    archetypes_trim = archetypes[:, :D]
    original_trim = original[:, :D]

    reconstruction = weights @ archetypes_trim
    ss_res = float(np.sum((original_trim - reconstruction) ** 2))
    ss_tot = float(
        np.sum((original_trim - original_trim.mean(axis=0)) ** 2)
    )
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def compute_archetype_to_centroid_distance(
    adata,
    obs_key: str = "archetypes",
    pca_key: str = "X_pca",
):
    """Non-circular 'are archetypes near their cells?' diagnostic (W-B10).

    Replaces the old ``archetype_cell_proximity`` ratio, which was
    circular: both sides of the ratio came from the same kNN computation,
    so it couldn't actually distinguish "archetype near its cells" from
    "archetype far from its cells".

    This function instead reports, for each archetype:

    - ``n_binned``: how many cells are assigned to that archetype
      (reads from ``adata.obs[obs_key]``).
    - ``archetype_position_norm``: L2 norm of the archetype coordinate
      vector.
    - ``centroid_distance``: L2 distance from the archetype position to
      the centroid of its binned cells. Small = archetype sits among its
      cells. Large = archetype is extrapolated beyond its cell cloud.
    - ``data_mean_distance``: L2 distance from the archetype position to
      the global data mean. Reference scale for reading
      ``archetype_position_norm`` in context.
    - ``bin_radius``: mean L2 distance from cells in this bin to their
      own centroid — the characteristic size of the cell cloud.
    - ``extrapolation_ratio``: ``centroid_distance / bin_radius``.
      Values >> 1 mean the archetype sits well outside its own cell
      cloud; values < 1 mean it sits among its cells. Using bin_radius
      as the denominator (instead of data_mean_distance) avoids the
      pathology where displacing an archetype far scales BOTH the
      numerator and naive denominators proportionally.

    Parameters
    ----------
    adata : AnnData
        Must contain ``adata.obsm[pca_key]`` (N x D) and
        ``adata.uns['archetype_coordinates']`` (K x D). Archetype
        assignments must be a categorical in ``adata.obs[obs_key]`` with
        labels like ``"archetype_1"``, ``"archetype_2"``, ...
    obs_key : str, default 'archetypes'
        Key in ``adata.obs`` holding the archetype assignment column.
    pca_key : str, default 'X_pca'
        Key in ``adata.obsm`` holding the coordinate matrix used for the
        L2 computations.

    Returns
    -------
    pd.DataFrame
        One row per archetype with columns
        ``archetype_label``, ``n_binned``, ``archetype_position_norm``,
        ``centroid_distance``, ``data_mean_distance``, ``bin_radius``,
        ``extrapolation_ratio``. Empty bins get ``n_binned=0``,
        ``centroid_distance=NaN``, and ``bin_radius=NaN``; the ratio is
        also NaN.
    """
    import numpy as np
    import pandas as pd

    if pca_key not in adata.obsm:
        raise ValueError(f"adata.obsm['{pca_key}'] not found")
    if "archetype_coordinates" not in adata.uns:
        raise ValueError("adata.uns['archetype_coordinates'] not found")
    if obs_key not in adata.obs:
        raise ValueError(f"adata.obs['{obs_key}'] not found")

    coords = np.asarray(adata.obsm[pca_key], dtype=np.float64)
    arch_positions = np.asarray(
        adata.uns["archetype_coordinates"], dtype=np.float64
    )
    K, D_arch = arch_positions.shape
    _, D_coords = coords.shape
    if D_arch != D_coords:
        raise ValueError(
            f"Dimension mismatch: archetypes have {D_arch} dims, "
            f"coords have {D_coords} dims"
        )

    assignments = adata.obs[obs_key]
    data_mean = coords.mean(axis=0)

    labels = [f"archetype_{k + 1}" for k in range(K)]
    rows = []
    for k, label in enumerate(labels):
        pos = arch_positions[k]
        mask = (assignments == label).values
        n_binned = int(mask.sum())
        arch_norm = float(np.linalg.norm(pos))
        data_mean_dist = float(np.linalg.norm(pos - data_mean))

        if n_binned == 0:
            centroid_dist = float("nan")
            bin_radius = float("nan")
            ratio = float("nan")
        else:
            bin_coords = coords[mask]
            centroid = bin_coords.mean(axis=0)
            centroid_dist = float(np.linalg.norm(pos - centroid))
            # Characteristic cloud size: mean L2 distance from cells to
            # their bin's centroid. Robust fallback when only 1 cell is
            # binned (radius = 0 → ratio undefined, report as NaN).
            cell_to_centroid = np.linalg.norm(bin_coords - centroid, axis=1)
            bin_radius = float(cell_to_centroid.mean())
            if bin_radius > 0:
                ratio = centroid_dist / bin_radius
            else:
                ratio = float("nan")

        rows.append({
            "archetype_label": label,
            "n_binned": n_binned,
            "archetype_position_norm": arch_norm,
            "centroid_distance": centroid_dist,
            "data_mean_distance": data_mean_dist,
            "bin_radius": bin_radius,
            "extrapolation_ratio": ratio,
        })

    return pd.DataFrame(rows)


def build_permutation_curve_figure(
    perm_null_result: dict,
    pair_i: int,
    pair_j: int,
    *,
    src_label: str | None = None,
    tgt_label: str | None = None,
    dpi: int = 130,
):
    """Single-pair permutation curve degradation figure (W-B23).

    Renders the permutation null curve for one (source, target) archetype
    pair: x-axis = swap fraction, y-axis = correspondence mass. The
    observed mass (constant across f because it does not depend on the
    swap) is drawn as a horizontal red line; the per-fraction null
    distribution is shown as mean +/- std error bars (with the full mass
    range shaded behind for context).

    A real correspondence pair shows the observed line above the null
    band at all swap fractions; a noisy pair has the observed line
    inside the null band even at f=0 (zero discrimination).

    Parameters
    ----------
    perm_null_result : dict
        Output of ``compute_correspondence_permutation_null``.
    pair_i : int
        Source archetype row index.
    pair_j : int
        Target archetype column index.
    src_label : str, optional
        Display label for the source archetype (e.g. ``"HSC A1"``).
    tgt_label : str, optional
        Display label for the target archetype.
    dpi : int, default 130
        Figure DPI.

    Returns
    -------
    matplotlib.figure.Figure
        Single-axis figure with the degradation curve.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as _np

    swap_fractions = list(perm_null_result["swap_fractions"])
    null_mean_curve = _np.asarray(perm_null_result["null_mean_curve"])
    null_std_curve = _np.asarray(perm_null_result["null_std_curve"])
    observed_mass = _np.asarray(perm_null_result["observed_mass"])
    null_distributions = perm_null_result["null_distributions"]
    empirical_p = _np.asarray(perm_null_result["empirical_p"])
    empirical_fdr = _np.asarray(perm_null_result["empirical_fdr"])

    obs_val = float(observed_mass[pair_i, pair_j])
    null_means = null_mean_curve[:, pair_i, pair_j]
    null_stds = null_std_curve[:, pair_i, pair_j]

    # Per-fraction min/max for the shaded band
    null_min = []
    null_max = []
    for f in swap_fractions:
        arr = _np.asarray(null_distributions[float(f)])[:, pair_i, pair_j]
        null_min.append(float(arr.min()))
        null_max.append(float(arr.max()))

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=dpi)

    # Range band
    ax.fill_between(
        swap_fractions, null_min, null_max,
        color="#999999", alpha=0.18, label="Null min/max range"
    )
    # Mean +/- std error bars
    ax.errorbar(
        swap_fractions, null_means, yerr=null_stds,
        fmt="o-", color="#666666", linewidth=1.5, capsize=4,
        label="Null mean +/- std",
    )
    # Observed horizontal line (does not depend on f)
    ax.axhline(
        obs_val, color="#D55E00", linewidth=2.0, linestyle="--",
        label=f"Observed = {obs_val:.3f}",
    )

    src_disp = src_label if src_label is not None else f"src arch {pair_i}"
    tgt_disp = tgt_label if tgt_label is not None else f"tgt arch {pair_j}"
    ax.set_xlabel("Swap fraction f")
    ax.set_ylabel("Correspondence mass")
    ax.set_title(
        f"Permutation curve: {src_disp} -> {tgt_disp}\n"
        f"empirical p={empirical_p[pair_i, pair_j]:.3f}, "
        f"FDR={empirical_fdr[pair_i, pair_j]:.3f}"
    )
    ax.legend(loc="best", fontsize=8, frameon=True)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def build_overlapping_ridgeplot(
    data_per_group: dict,
    *,
    bandwidth=None,
    overlap: float = 0.5,
    max_groups: int = 12,
    title: str = "",
    xlabel: str = "",
    cmap_name: str = "viridis",
    figsize: tuple | None = None,
    dpi: int = 130,
):
    """Seurat-style overlapping KDE ridgeplot (W-B21).

    Each group's values are smoothed with a Gaussian KDE and drawn as a
    filled ridge. Ridges are stacked vertically with a baseline spacing
    of 1.0, but each ridge is scaled to occupy ``(1 + overlap)`` units of
    vertical space, so adjacent ridges visually overlap. This mimics
    Seurat's ``RidgePlot`` default aesthetics.

    Parameters
    ----------
    data_per_group : dict[str, np.ndarray]
        Ordered mapping ``group_label -> 1-D values``. Ridges are drawn
        bottom-to-top in iteration order. Groups with fewer than 2 points
        or zero variance are skipped with a text marker at their baseline.
    bandwidth : float | str | None, default None
        Bandwidth passed to ``scipy.stats.gaussian_kde``. When None,
        Scott's rule is used.
    overlap : float, default 0.5
        Controls vertical stacking density. 0 = no overlap (ridges
        touch their baselines); 0.5 = ridges occupy 1.5 * baseline
        spacing, creating visible stacking; 0.8 = aggressive Seurat
        default stacking. Clamped to [0, 2].
    max_groups : int, default 12
        Hard cap on number of ridges rendered. Extra groups are dropped
        with a caption annotation. Reader-defined ordering is preserved.
    title : str, default ""
        Figure title.
    xlabel : str, default ""
        X-axis label.
    cmap_name : str, default "viridis"
        Matplotlib colormap name for ridge fill colors. ``"tab10"`` is
        a good discrete alternative.
    figsize : (w, h) or None
        Figure size. Auto-scales with n_ridges when None.
    dpi : int, default 130
        Figure DPI.

    Returns
    -------
    matplotlib.figure.Figure
        Single-axes figure with stacked ridges. Y-tick labels are the
        group names at each ridge's baseline position.

    Notes
    -----
    All ridges share the same x-axis (the union range over all group
    values). KDE density is normalised per-ridge so the tallest point
    of each ridge is ``1 + overlap`` units of vertical space; this keeps
    visual stacking uniform regardless of per-group sample size.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as _np
    from scipy.stats import gaussian_kde

    # Clamp overlap to a sensible range
    overlap = float(max(0.0, min(2.0, overlap)))

    # Keep iteration order; cap at max_groups
    items = list(data_per_group.items())
    n_total = len(items)
    dropped = []
    if n_total > max_groups:
        dropped = [k for k, _ in items[max_groups:]]
        items = items[:max_groups]
    n_ridges = len(items)

    if n_ridges == 0:
        # Empty plot with a text marker
        fig, ax = plt.subplots(figsize=figsize or (8, 2), dpi=dpi)
        ax.text(0.5, 0.5, "No groups to plot", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    # Auto figsize: 8 wide, height scales with ridge count
    if figsize is None:
        fig_height = max(3.0, 0.6 * n_ridges + 1.2)
        figsize = (8.0, fig_height)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Global x-range across all groups, with a 2% pad on each side
    all_values = []
    for _, vals in items:
        v = _np.asarray(vals, dtype=float).ravel()
        v = v[_np.isfinite(v)]
        if v.size > 0:
            all_values.append(v)
    if not all_values:
        ax.text(0.5, 0.5, "No finite values across groups",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        return fig
    x_lo = float(min(v.min() for v in all_values))
    x_hi = float(max(v.max() for v in all_values))
    if x_hi - x_lo < 1e-10:
        x_lo -= 0.5
        x_hi += 0.5
    pad = 0.02 * (x_hi - x_lo)
    x_lo -= pad
    x_hi += pad
    x_grid = _np.linspace(x_lo, x_hi, 400)

    # Ridge vertical spacing: baselines are 1 unit apart, each ridge's
    # maximum height is (1 + overlap) so adjacent ridges visually stack.
    base_spacing = 1.0
    ridge_height = base_spacing * (1.0 + overlap)

    cmap = plt.get_cmap(cmap_name)

    baseline_ys = []
    labels = []
    for ci, (label, vals) in enumerate(items):
        # Baselines drawn bottom-to-top: group 0 at the bottom.
        y_offset = ci * base_spacing
        baseline_ys.append(y_offset)
        labels.append(str(label))

        # Compute color from colormap: discrete maps (tab10, Set1) use
        # integer indexing; continuous maps use normalized floats.
        if cmap_name in ("tab10", "tab20", "Set1", "Set2", "Set3", "Paired", "Dark2"):
            color = cmap(ci % cmap.N)
        else:
            denom = max(n_ridges - 1, 1)
            color = cmap(ci / denom)

        v = _np.asarray(vals, dtype=float).ravel()
        v = v[_np.isfinite(v)]
        if v.size < 2 or _np.std(v) < 1e-12:
            # Too few points or zero variance — just draw a tick at the
            # mean and a horizontal baseline line so the reader sees the
            # ridge exists.
            ax.hlines(y_offset, x_lo, x_hi, color=color, linewidth=0.6,
                      alpha=0.6)
            if v.size >= 1:
                ax.plot([float(v.mean())], [y_offset], marker="o",
                        color=color, markersize=3)
            continue

        try:
            if bandwidth is None:
                kde = gaussian_kde(v, bw_method="scott")
            else:
                kde = gaussian_kde(v, bw_method=bandwidth)
            density = kde(x_grid)
            dmax = float(density.max())
            if dmax <= 0:
                continue
            density_norm = density / dmax * ridge_height
            # Fill + outline
            ax.fill_between(
                x_grid, y_offset, y_offset + density_norm,
                color=color, alpha=0.55, linewidth=0,
            )
            ax.plot(
                x_grid, y_offset + density_norm,
                color=color, linewidth=1.1, alpha=0.95,
            )
            # Mean tick
            mean_val = float(v.mean())
            # Find density at mean for tick top
            tick_top = y_offset + ridge_height * 0.5
            ax.vlines(mean_val, y_offset, tick_top, color=color,
                      linewidth=0.9, linestyle="--", alpha=0.75)
        except Exception:
            # KDE failed (rare) — fall back to histogram outline
            hist, edges = _np.histogram(v, bins=30, range=(x_lo, x_hi),
                                        density=True)
            hmax = hist.max() if hist.max() > 0 else 1.0
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.plot(centers, y_offset + hist / hmax * ridge_height,
                    color=color, linewidth=1.0, alpha=0.9)

    ax.set_xlim(x_lo, x_hi)
    # Top of the plot must include the tallest point of the topmost
    # ridge.
    ax.set_ylim(-0.1 * base_spacing,
                (n_ridges - 1) * base_spacing + ridge_height + 0.1 * base_spacing)
    ax.set_yticks(baseline_ys)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    if title:
        ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.4)

    if dropped:
        # Annotate dropped groups at the bottom
        ax.text(
            0.99, -0.08,
            f"(+ {len(dropped)} groups dropped: {', '.join(dropped[:3])}"
            + ("..." if len(dropped) > 3 else "") + ")",
            ha="right", va="top", fontsize=8, color="#666666",
            transform=ax.transAxes,
        )

    fig.tight_layout()
    return fig


def build_pseudotime_expansion_plot(
    pseudotime,
    expansion_scores,
    gene_names,
    *,
    title: str = "",
    xlabel: str = "Flow pseudotime (transport distance)",
    ylabel: str = "Expansion / contraction score",
    n_bins: int = 30,
    max_genes: int = 10,
    figsize: tuple | None = None,
    dpi: int = 130,
):
    """Pseudotime vs expansion/contraction scatter with binned trend lines.

    For each gene, cells are binned along the pseudotime axis and the mean
    expansion score per bin is plotted as a trend line, giving a
    "pseudotime x expansion" view that replaces the cosine-similarity
    ridgeplot.

    Parameters
    ----------
    pseudotime : np.ndarray, shape [n_cells]
        Per-cell pseudotime values (e.g., transport distance from source
        to transported position along the flow direction).
    expansion_scores : np.ndarray, shape [n_cells, n_genes]
        Per-cell per-gene expansion/contraction scores (e.g., from
        ``flow_jacobian`` per_cell_expansion).
    gene_names : list[str]
        Gene names aligned with columns of ``expansion_scores``.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    n_bins : int
        Number of pseudotime bins for the trend lines.
    max_genes : int
        Maximum number of genes to show. Genes are ranked by absolute
        mean expansion score across cells.
    figsize : tuple or None
        Figure size. Auto-computed if None.
    dpi : int
        Figure DPI.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import numpy as _np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pt = _np.asarray(pseudotime, dtype=float).ravel()
    exp = _np.asarray(expansion_scores, dtype=float)
    names = list(gene_names)

    if exp.ndim == 1:
        exp = exp[:, _np.newaxis]
    n_cells, n_genes = exp.shape

    # Rank genes by mean |expansion| and pick top max_genes
    mean_abs = _np.nanmean(_np.abs(exp), axis=0)
    n_show = min(max_genes, n_genes)
    top_idx = _np.argsort(mean_abs)[-n_show:][::-1]

    if figsize is None:
        figsize = (8.0, max(4.0, 0.5 * n_show + 2.0))
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    cmap = plt.get_cmap("tab10")

    # Filter to finite pseudotime
    valid = _np.isfinite(pt)
    pt_v = pt[valid]
    exp_v = exp[valid]

    if pt_v.size < 10:
        ax.text(0.5, 0.5, "Too few cells with valid pseudotime",
                ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig

    bin_edges = _np.linspace(pt_v.min(), pt_v.max(), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = _np.digitize(pt_v, bin_edges) - 1
    bin_idx = _np.clip(bin_idx, 0, n_bins - 1)

    for rank, gi in enumerate(top_idx):
        color = cmap(rank % 10)
        gname = names[gi] if gi < len(names) else f"gene_{gi}"
        gene_vals = exp_v[:, gi]

        # Binned mean
        bin_means = _np.full(n_bins, _np.nan)
        for b in range(n_bins):
            mask_b = bin_idx == b
            if mask_b.sum() >= 2:
                bin_means[b] = _np.nanmean(gene_vals[mask_b])

        valid_bins = _np.isfinite(bin_means)
        if valid_bins.sum() >= 2:
            ax.plot(bin_centers[valid_bins], bin_means[valid_bins],
                    color=color, linewidth=1.8, alpha=0.85, label=gname)
            # Mark start/end
            ax.scatter(bin_centers[valid_bins][0], bin_means[valid_bins][0],
                       color=color, s=25, marker="o", zorder=5)
            ax.scatter(bin_centers[valid_bins][-1], bin_means[valid_bins][-1],
                       color=color, s=25, marker="^", zorder=5)

    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="best", ncol=2, framealpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    return fig


def build_tricolor_gene_scatter(
    expression,
    expansion,
    flow_strength,
    gene_names,
    *,
    n_labels: int = 15,
    title: str = "",
    figsize: tuple | None = None,
    dpi: int = 130,
):
    """Tricolor scatter: expression vs expansion, colored by flow strength.

    Visualizes per-gene statistics from a per-pair flow analysis:
    - x-axis: mean expression level
    - y-axis: expansion/contraction (signed: expanding genes above 0,
      contracting below 0)
    - color: flow association strength (viridis colormap)
    - point size: proportional to abs(flow_strength)
    - Labels: top ``n_labels`` genes by abs(expansion * flow_strength)

    Parameters
    ----------
    expression : array-like, shape [n_genes]
        Mean expression per gene.
    expansion : array-like, shape [n_genes]
        Signed expansion/contraction score per gene. Positive = expanding.
    flow_strength : array-like, shape [n_genes]
        Flow association score per gene (e.g., cosine alignment or R²).
    gene_names : list[str]
        Gene labels aligned with the arrays.
    n_labels : int
        Number of top genes to label (by |expansion * flow_strength|).
    title : str
        Plot title.
    figsize : tuple or None
        Figure size.
    dpi : int
        Figure DPI.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import numpy as _np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    expr = _np.asarray(expression, dtype=float).ravel()
    expan = _np.asarray(expansion, dtype=float).ravel()
    flow = _np.asarray(flow_strength, dtype=float).ravel()
    names = list(gene_names)

    n = min(len(expr), len(expan), len(flow), len(names))
    expr = expr[:n]
    expan = expan[:n]
    flow = flow[:n]
    names = names[:n]

    # Filter non-finite values
    valid = _np.isfinite(expr) & _np.isfinite(expan) & _np.isfinite(flow)
    if valid.sum() < 3:
        fig, ax = plt.subplots(figsize=figsize or (8, 6), dpi=dpi)
        ax.text(0.5, 0.5, "Too few valid data points",
                ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig

    expr_v = expr[valid]
    expan_v = expan[valid]
    flow_v = flow[valid]
    names_v = [names[i] for i in range(n) if valid[i]]

    # Point sizes proportional to |flow_strength|
    abs_flow = _np.abs(flow_v)
    if abs_flow.max() > 0:
        sizes = 10 + 80 * (abs_flow / abs_flow.max())
    else:
        sizes = _np.full(len(abs_flow), 20.0)

    if figsize is None:
        figsize = (8.0, 6.0)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    norm = Normalize(vmin=float(flow_v.min()), vmax=float(flow_v.max()))
    sc = ax.scatter(
        expr_v, expan_v,
        c=flow_v, cmap="viridis", norm=norm,
        s=sizes, alpha=0.6, edgecolor="none",
    )
    fig.colorbar(sc, ax=ax, label="Flow association strength")

    ax.axhline(0.0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    # Label top genes by |expansion * flow_strength| — adjustText repels
    # overlapping labels so crowded regions remain legible.
    if n_labels > 0:
        composite = _np.abs(expan_v * flow_v)
        top_idx = _np.argsort(composite)[-min(n_labels, len(composite)):]
        texts = []
        tx = []
        ty = []
        for i in top_idx:
            tx.append(float(expr_v[i]))
            ty.append(float(expan_v[i]))
            texts.append(ax.text(
                float(expr_v[i]), float(expan_v[i]), names_v[i],
                fontsize=7, alpha=0.85, color="#1f1f1f",
            ))
        _adjust_labels(ax, texts, x=tx, y=ty)

    ax.set_xlabel("Mean expression level")
    ax.set_ylabel("Expansion / contraction (signed)")
    if title:
        ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    return fig


def build_absence_plot(
    expression,
    flow_strength,
    gene_names,
    *,
    expr_quantile: float = 0.7,
    flow_quantile: float = 0.3,
    n_labels_absent: int = 15,
    n_labels_positive: int = 5,
    title: str = "",
):
    """Highlight genes with high expression but LOW flow association.

    These 'absent' genes are often more biologically interesting than
    the positive hits — why is the flow ignoring a highly-expressed gene?

    Parameters
    ----------
    expression : array-like [n_genes]
    flow_strength : array-like [n_genes]
    gene_names : list[str]
    expr_quantile : float
        Expression threshold percentile (genes above this are 'high').
    flow_quantile : float
        Flow threshold percentile (genes below this are 'low').
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    expression = np.asarray(expression, dtype=np.float64)
    flow_strength = np.asarray(flow_strength, dtype=np.float64)
    names = list(gene_names)

    expr_thr = float(np.percentile(expression, expr_quantile * 100))
    flow_thr = float(np.percentile(flow_strength, flow_quantile * 100))

    high_expr = expression > expr_thr
    low_flow = flow_strength < flow_thr
    absent = high_expr & low_flow
    positive = flow_strength > np.percentile(flow_strength, 90)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=130)
    colors = np.where(absent, "#d62728",
             np.where(positive, "#2ca02c", "#cccccc"))
    sizes = np.where(absent, 40, np.where(positive, 25, 10))

    ax.scatter(expression, flow_strength, s=sizes, c=colors, alpha=0.7)
    ax.axhline(flow_thr, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(expr_thr, color="gray", linestyle=":", alpha=0.5)

    # Label absent + top positive genes with adjustText repulsion so
    # crowded corners stay legible.
    texts = []
    tx = []
    ty = []
    absent_idx = np.where(absent)[0]
    if len(absent_idx) > n_labels_absent:
        absent_idx = absent_idx[np.argsort(-expression[absent_idx])][:n_labels_absent]
    for i in absent_idx:
        tx.append(float(expression[i]))
        ty.append(float(flow_strength[i]))
        texts.append(ax.text(
            float(expression[i]), float(flow_strength[i]), names[i],
            fontsize=7, color="#d62728", fontweight="bold",
        ))
    pos_idx = np.argsort(-flow_strength)[:n_labels_positive]
    for i in pos_idx:
        tx.append(float(expression[i]))
        ty.append(float(flow_strength[i]))
        texts.append(ax.text(
            float(expression[i]), float(flow_strength[i]), names[i],
            fontsize=7, color="#2ca02c",
        ))
    _adjust_labels(ax, texts, x=tx, y=ty)

    n_absent = int(absent.sum())
    ax.set_xlabel("Mean Expression")
    ax.set_ylabel("Flow Association Strength")
    subtitle = (
        f"{n_absent} genes high-expr / low-flow (red) — likely shared "
        f"between source and target"
    )
    ax.set_title(f"{title}\n{subtitle}" if title else subtitle)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def build_lollipop_chart(
    flow_strength,
    expansion,
    expression,
    gene_names,
    *,
    top_n: int = 25,
    title: str = "",
):
    """Lollipop chart ranked by flow strength.

    Stem length = flow association. Head color = expansion sign
    (red = expanding, blue = contracting). Head size = expression level.

    Parameters
    ----------
    flow_strength : array-like [n_genes]
    expansion : array-like [n_genes] signed
    expression : array-like [n_genes]
    gene_names : list[str]
    top_n : int
        Show top N genes by flow strength.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flow_strength = np.asarray(flow_strength, dtype=np.float64)
    expansion = np.asarray(expansion, dtype=np.float64)
    expression = np.asarray(expression, dtype=np.float64)
    names = list(gene_names)

    # Top N by flow strength
    order = np.argsort(-flow_strength)[:min(top_n, len(flow_strength))]
    n_show = len(order)

    fig, ax = plt.subplots(figsize=(10, max(5, n_show * 0.28)), dpi=130)
    y = np.arange(n_show)

    for i, idx in enumerate(order):
        ax.plot([0, flow_strength[idx]], [i, i], color="gray",
                linewidth=1, alpha=0.5)

    head_colors = ["#d62728" if expansion[idx] > 0 else "#1f77b4"
                   for idx in order]
    expr_max = float(expression[order].max()) if n_show else 1.0
    expr_min = float(expression[order].min()) if n_show else 0.0
    head_sizes = 30 + 120 * (expression[order] / (expr_max + 1e-10))

    ax.scatter(flow_strength[order], y, s=head_sizes, c=head_colors,
               zorder=5, edgecolors="black", linewidths=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels([names[idx] for idx in order], fontsize=8)
    ax.set_xlabel("Flow Association Strength")
    ax.set_title(
        f"{title}\nRed = expanding, blue = contracting. Size = expression."
        if title else "Red = expanding, blue = contracting. Size = expression.")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    # Composite legend: expansion direction (color) + expression level (size).
    from matplotlib.lines import Line2D
    # Three size anchors: min / mid / max of plotted expression
    expr_mid = 0.5 * (expr_min + expr_max)
    size_anchors = [
        ("min", expr_min, 30 + 120 * (expr_min / (expr_max + 1e-10))),
        ("mid", expr_mid, 30 + 120 * (expr_mid / (expr_max + 1e-10))),
        ("max", expr_max, 30 + 120 * (expr_max / (expr_max + 1e-10))),
    ]
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
               markeredgecolor="black", markersize=8, label="expanding (expan > 0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
               markeredgecolor="black", markersize=8, label="contracting (expan < 0)"),
    ]
    for anchor_label, expr_val, s in size_anchors:
        # Line2D markersize is point-diameter; scatter 's' is area in pt^2.
        ms = max(3.0, (s ** 0.5))
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
                   markeredgecolor="black", markersize=ms,
                   label=f"expr {anchor_label} ({expr_val:.2f})"),
        )
    ax.legend(
        handles=legend_handles,
        loc="lower right", fontsize=7, framealpha=0.85,
        title="Color = expansion sign • Size = expression",
        title_fontsize=7,
    )
    fig.tight_layout()
    return fig


# ============================================================================
# Part 2 helpers (shared with Part 1 when relevant)
# ============================================================================


def build_response_timepoint_colormap(
    responses: Sequence[str] = ("NR", "R1", "R2"),
    treatments: Sequence[str] = ("Base", "PD1", "RTPD1"),
) -> dict:
    """Return a ``(response, treatment) -> hex color`` map.

    Hue = response lineage (NR=reds, R1=oranges, R2=blues); lightness =
    timepoint (lightest at the first treatment, darkest at the last).

    Raises ValueError if an unknown response is passed.
    """
    # R1 moved from oranges to greens (r3 review) — NR warm reds are otherwise
    # hard to distinguish from R1 oranges at cell-level opacity. Three hue
    # families now stay visually distinct even at alpha=1.0.
    ramps = {
        "NR": ["#fca5a5", "#ef4444", "#991b1b"],   # reds: light → dark
        "R1": ["#86efac", "#16a34a", "#14532d"],   # greens: light → dark
        "R2": ["#93c5fd", "#2563eb", "#1e3a8a"],   # blues: light → dark
    }
    unknown = set(responses) - set(ramps)
    if unknown:
        raise ValueError(f"Unknown response groups: {sorted(unknown)}. "
                         f"Expected subset of {sorted(ramps)}.")
    if len(treatments) > 3:
        raise ValueError("Only up to 3 treatments supported by the ramp width.")

    out: dict = {}
    for r in responses:
        ramp = ramps[r]
        for i, t in enumerate(treatments):
            out[(r, t)] = ramp[i]
    return out


def wasserstein2_distance(X, Y, *, max_n=2000, seed=42):
    """2-Wasserstein distance between two point clouds (sample-based via POT).

    Mirrors the helper in scripts/run_paper_part1_hsc.py so Part 2 can
    reuse the same implementation. Uses scipy.stats.wasserstein_distance_nd
    (which calls POT). Subsamples to ``max_n`` per side for tractability.

    Parameters
    ----------
    X : np.ndarray, shape [n1, d]
        Source point cloud (e.g. archetype weights or PCA coords).
    Y : np.ndarray, shape [n2, d]
        Target point cloud, same dim.
    max_n : int
        Maximum points per side. If either side exceeds this, randomly subsample.
    seed : int
        RNG seed for subsampling.

    Returns
    -------
    float
        Wasserstein-2 distance in input units.
    """
    import numpy as np
    from scipy.stats import wasserstein_distance_nd
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] > max_n:
        X = X[rng.choice(X.shape[0], max_n, replace=False)]
    if Y.shape[0] > max_n:
        Y = Y[rng.choice(Y.shape[0], max_n, replace=False)]
    return float(wasserstein_distance_nd(X, Y))


def build_segregation_ratio(obs, weights, response_col: str,
                              treatment_col: str) -> dict:
    """Within- vs between-response 2-Wasserstein ratio for Fig 3C gating.

    Parameters
    ----------
    obs : pd.DataFrame
        One row per cell; must contain ``response_col`` and ``treatment_col``.
    weights : np.ndarray
        Shape ``(n_cells, n_archetypes)``. Same row order as ``obs``.
    response_col, treatment_col : str
        Column names in ``obs``.

    Returns
    -------
    dict with keys: ``within`` (mean W2, same-response different-treatment),
                    ``between`` (mean W2, different-response any-treatment),
                    ``ratio`` = between/within,
                    ``n_within_pairs``, ``n_between_pairs``,
                    ``pair_distances`` (list of {group_a, group_b, kind, w2}).
    """
    import numpy as np

    obs = obs.reset_index(drop=True)
    groups = (
        obs[[response_col, treatment_col]]
        .apply(tuple, axis=1)
        .tolist()
    )
    # Collect per-group cell indices
    unique = sorted(set(groups))
    idx_of = {g: [] for g in unique}
    for i, g in enumerate(groups):
        idx_of[g].append(i)
    idx_of = {g: np.array(v) for g, v in idx_of.items() if len(v) >= 2}

    unique_ok = sorted(idx_of.keys())
    pair_dists = []
    within_vals = []
    between_vals = []

    for i, g1 in enumerate(unique_ok):
        for g2 in unique_ok[i + 1:]:
            w1 = weights[idx_of[g1]]
            w2 = weights[idx_of[g2]]
            d = wasserstein2_distance(w1, w2)
            kind = "within" if g1[0] == g2[0] else "between"
            pair_dists.append({"group_a": g1, "group_b": g2, "kind": kind, "w2": d})
            if kind == "within":
                within_vals.append(d)
            else:
                between_vals.append(d)

    within_mean = float(np.mean(within_vals)) if within_vals else float("nan")
    between_mean = float(np.mean(between_vals)) if between_vals else float("nan")
    ratio = (between_mean / within_mean) if within_mean > 0 else float("nan")

    return {
        "within": within_mean,
        "between": between_mean,
        "ratio": ratio,
        "n_within_pairs": len(within_vals),
        "n_between_pairs": len(between_vals),
        "pair_distances": pair_dists,
    }


def build_archetype_char_table(
    obs,
    archetypes_col: str,
    covariate_cols: Sequence[str],
    top_genes_by_archetype: dict | None = None,
    top_k_cohorts: int = 3,
) -> "pd.DataFrame":
    """One-row-per-archetype quick-look characterization table.

    Columns (fixed order):
        archetype, n_cells, pct_cells,
        dom_{covariate} for each covariate in ``covariate_cols``,
        top_cohorts (if ``cohort`` or similar patient-like col is present),
        top_genes (if ``top_genes_by_archetype`` provided).
    """
    import pandas as pd

    # Filter rows with a non-NaN archetype assignment
    obs = obs.loc[obs[archetypes_col].notna()].copy()
    total_cells = len(obs)

    rows = []
    archetype_ids = sorted(obs[archetypes_col].unique())
    cohort_col_candidate = next(
        (c for c in ("cohort", "patient", "donor") if c in covariate_cols),
        None,
    )

    def _label_to_int(label):
        """Return integer index for an archetype label (int, '0', 'archetype_0')."""
        import re
        try:
            return int(label)
        except (TypeError, ValueError):
            pass
        m = re.search(r"\d+$", str(label))
        return int(m.group()) if m else None

    for a in archetype_ids:
        sub = obs.loc[obs[archetypes_col] == a]
        n = len(sub)
        a_int = _label_to_int(a)
        row = {
            # Preserve original label (int or string like "archetype_0") so the
            # column matches adata.obs values for downstream joins.
            "archetype": a if a_int is None else a_int,
            "n_cells": n,
            "pct_cells": 100.0 * n / total_cells if total_cells else 0.0,
        }
        for cov in covariate_cols:
            if cov == cohort_col_candidate:
                # emit as top-k string
                vc = sub[cov].value_counts().head(top_k_cohorts)
                row["top_cohorts"] = ", ".join(
                    f"{k} ({v})" for k, v in vc.items()
                )
            else:
                vc = sub[cov].value_counts()
                dom = vc.index[0] if len(vc) else None
                dom_frac = vc.iloc[0] / n if n and len(vc) else 0.0
                row[f"dom_{cov}"] = f"{dom} ({100*dom_frac:.0f}%)" if dom is not None else ""
        if top_genes_by_archetype is not None:
            # Accept int-keyed OR str-keyed dicts
            genes = top_genes_by_archetype.get(a) or top_genes_by_archetype.get(a_int, [])
            row["top_genes"] = ", ".join(genes) if genes else ""
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def build_archetype_hypergeometric_tables(
    obs,
    archetypes_col: str,
    covariate_cols: Sequence[str],
    min_level_cells: int = 50,
) -> dict:
    """Per-covariate archetype enrichment tables.

    For each covariate (e.g., ``response_group``), build a K × L table
    where L = number of levels with ≥ ``min_level_cells``. Each cell
    reports ``OR (p, q)`` for the 2x2 Fisher test of
    "cells in archetype ∩ cells in level" vs margins.

    Parameters
    ----------
    obs : pd.DataFrame
        Cell-level metadata with at least ``archetypes_col`` and each column
        in ``covariate_cols``.
    archetypes_col : str
        Column name holding archetype assignments (integer or string labels).
    covariate_cols : Sequence[str]
        Categorical covariate columns to test (e.g. ``["response_group",
        "treatment"]``).
    min_level_cells : int, optional
        Levels with fewer than this many cells are dropped. Default 50.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keyed by covariate name. Each DataFrame has columns:
        ``archetype``, ``OR_{level}``, ``p_{level}``, ``q_{level}``
        for each surviving level. ``q`` values are BH-corrected across all
        archetype × level tests within that covariate.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import fisher_exact
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError as e:
        raise ImportError("statsmodels required for BH correction") from e

    obs = obs.loc[obs[archetypes_col].notna()].copy()
    archetype_ids = sorted(obs[archetypes_col].unique())
    K = len(archetype_ids)
    out: dict = {}

    for cov in covariate_cols:
        vc = obs[cov].value_counts()
        keep_levels = vc[vc >= min_level_cells].index.tolist()
        if not keep_levels:
            out[cov] = pd.DataFrame({"archetype": archetype_ids})
            continue

        # Compute OR + p per (archetype × level)
        ors = np.full((K, len(keep_levels)), np.nan)
        ps = np.full((K, len(keep_levels)), np.nan)
        for ai, a in enumerate(archetype_ids):
            in_arch = obs[archetypes_col] == a
            for li, lv in enumerate(keep_levels):
                in_lv = obs[cov] == lv
                a11 = int((in_arch & in_lv).sum())
                a12 = int((in_arch & ~in_lv).sum())
                a21 = int((~in_arch & in_lv).sum())
                a22 = int((~in_arch & ~in_lv).sum())
                table = [[a11, a12], [a21, a22]]
                or_val, pval = fisher_exact(table, alternative="two-sided")
                ors[ai, li] = or_val
                ps[ai, li] = pval

        # BH correction across all (archetype × level) tests in this covariate
        flat_p = ps.flatten()
        _, qs, _, _ = multipletests(flat_p, method="fdr_bh")
        qs = qs.reshape(ps.shape)

        df = pd.DataFrame({"archetype": archetype_ids})
        for li, lv in enumerate(keep_levels):
            df[f"OR_{lv}"] = ors[:, li]
            df[f"p_{lv}"] = ps[:, li]
            df[f"q_{lv}"] = qs[:, li]
        out[cov] = df

    return out


def build_holdout_projection_qc(
    cells_train,
    reconstruction_train,
    cells_holdout,
    reconstruction_holdout,
    archetype_positions,
) -> dict:
    """Archetypal R² on train + holdout, plus per-cell NN distance to the
    nearest archetype position for the holdout set.

    Parameters
    ----------
    cells_train, cells_holdout : np.ndarray
        Shape ``(n_cells, n_dims)`` in the same coord space as
        ``archetype_positions`` (typically PCA or a learned latent).
    reconstruction_train, reconstruction_holdout : np.ndarray
        Same shape — the archetype-weighted reconstructions
        (``weights @ archetype_positions``).
    archetype_positions : np.ndarray
        Shape ``(K, n_dims)``.

    Returns
    -------
    dict : ``train_r2``, ``holdout_r2``, ``holdout_mean_nn_dist``,
           ``holdout_median_nn_dist``.
    """
    import numpy as np

    def _arch_r2(original, recon):
        # Mirrors peach.calculate_archetype_r2 semantics — per-feature mean
        # centering, scalar ss_tot if >1D.
        ss_res = float(np.sum((original - recon) ** 2))
        ss_tot = float(np.sum((original - original.mean(axis=0)) ** 2))
        return 1.0 - ss_res / max(ss_tot, 1e-12)

    train_r2 = _arch_r2(cells_train, reconstruction_train)
    holdout_r2 = _arch_r2(cells_holdout, reconstruction_holdout)

    # NN distance from each holdout cell to nearest archetype
    diff = cells_holdout[:, None, :] - archetype_positions[None, :, :]
    dists = np.linalg.norm(diff, axis=2)  # (n_holdout, K)
    nn = dists.min(axis=1)

    return {
        "train_r2": train_r2,
        "holdout_r2": holdout_r2,
        "holdout_mean_nn_dist": float(nn.mean()),
        "holdout_median_nn_dist": float(np.median(nn)),
    }


def build_distance_heatmaps(
    obs,
    weights,
    pca,
    response_col: str,
    archetypes_col: str,
):
    """Return (plotly.go.Figure, spearman_rho) for Fig 3C-i.

    Two side-by-side heatmaps of size (3K × 3K), rows/cols =
    (response_group, archetype) pairs:
      - Left: W2 in archetype-weight simplex
      - Right: Euclidean centroid distance in PCA space

    Parameters
    ----------
    obs : pd.DataFrame
        Cell-level metadata containing ``response_col`` and ``archetypes_col``.
    weights : np.ndarray, shape (n_cells, K)
        Per-cell archetype weights (rows sum to 1).
    pca : np.ndarray, shape (n_cells, n_dims)
        Per-cell coordinates in PCA (or other embedding) space.
    response_col : str
        Column in ``obs`` identifying response group (e.g. "NR", "R1", "R2").
    archetypes_col : str
        Column in ``obs`` with integer archetype assignments (0-indexed).

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Side-by-side heatmaps.
    spearman_rho : float
        Spearman correlation between W2 and Euclidean upper-triangle entries.
        NaN when fewer than 3 unique group pairs exist.
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.stats import spearmanr

    obs = obs.reset_index(drop=True)
    # Build group index: (response, archetype)
    resp_vals = obs[response_col].values
    arch_vals = obs[archetypes_col].values

    def _arch_to_int(v):
        """Convert archetype label to sortable int. Handles 'archetype_1' or plain int."""
        if isinstance(v, (int, np.integer)):
            return int(v)
        s = str(v)
        # assign_archetypes stores "archetype_1", "archetype_2", ... (1-indexed)
        return int(s.split("_")[-1]) if "_" in s else int(s)

    # Filter out "no_archetype" cells (assign_archetypes with percentage_per_archetype
    # leaves cells that don't rank in the top P% of any archetype unassigned).
    _assigned_mask = np.array([str(v) != "no_archetype" for v in arch_vals])
    if not _assigned_mask.all():
        obs = obs[_assigned_mask].reset_index(drop=True)
        resp_vals = resp_vals[_assigned_mask]
        arch_vals = arch_vals[_assigned_mask]

    groups_raw = sorted(
        set(zip(resp_vals, arch_vals)),
        key=lambda t: (str(t[0]), _arch_to_int(t[1])),
    )
    # Build index arrays per group using column equality (avoids tuple broadcasting)
    idx_of = {
        g: np.where((resp_vals == g[0]) & (arch_vals == g[1]))[0]
        for g in groups_raw
    }
    # Drop groups with <2 cells (can't compute W2)
    groups = [g for g in groups_raw if len(idx_of[g]) >= 2]
    G = len(groups)

    w2_mat = np.zeros((G, G))
    eu_mat = np.zeros((G, G))
    for i, gi in enumerate(groups):
        for j, gj in enumerate(groups):
            if j <= i:
                continue
            wi = weights[idx_of[gi]]
            wj = weights[idx_of[gj]]
            w2 = wasserstein2_distance(wi, wj)
            w2_mat[i, j] = w2_mat[j, i] = w2
            pi = pca[idx_of[gi]].mean(axis=0)
            pj = pca[idx_of[gj]].mean(axis=0)
            d_eu = float(np.linalg.norm(pi - pj))
            eu_mat[i, j] = eu_mat[j, i] = d_eu

    # Spearman on the upper triangle
    iu = np.triu_indices(G, k=1)
    if len(iu[0]) >= 3:
        rho, _ = spearmanr(w2_mat[iu], eu_mat[iu])
    else:
        rho = float("nan")

    labels = [f"{r}/A{_arch_to_int(a)}" for r, a in groups]
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("W2 (archetype weights)",
                                        "Euclidean centroid (PCA space)"))
    fig.add_trace(
        go.Heatmap(z=w2_mat, x=labels, y=labels, colorscale="Viridis",
                   showscale=True, colorbar=dict(x=0.43, len=0.75)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(z=eu_mat, x=labels, y=labels, colorscale="Plasma",
                   showscale=True, colorbar=dict(x=1.02, len=0.75)),
        row=1, col=2,
    )
    fig.update_layout(
        title=f"(response × archetype) pairwise distances — Spearman ρ = {rho:.3f}",
        height=520, width=1200,
    )
    return fig, float(rho) if not np.isnan(rho) else rho


def build_diversity_block(
    obs,
    weights,
    pca,
    group_col: str,
    bootstrap_n: int = 200,
    subsample: int = 500,
    random_state: int = 42,
):
    """Fig 3C-ii — three-panel diversity block (spec §5.4.3).

    Parameters
    ----------
    obs : pd.DataFrame
        Per-cell metadata; must contain ``group_col``.
    weights : np.ndarray, shape (n_cells, K)
        Archetype weight vectors (rows sum to ≈1).
    pca : np.ndarray, shape (n_cells, D)
        PCA embedding (or any low-dim embedding) of cells.
    group_col : str
        Column in ``obs`` that defines groups (e.g. "response_group").
    bootstrap_n : int
        Number of bootstrap resamples used to compute 95 % CI for PCA
        dispersion (panel 2).
    subsample : int
        Max cells to use when computing median pairwise distance inside each
        bootstrap replicate; keeps runtime manageable for large groups.
    random_state : int
        Seed for the internal RNG.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Three-panel figure:
          1. Violin — per-cell Shannon entropy of archetype weights.
          2. Bar + CI — median pairwise Euclidean distance in PCA space per group.
          3. Bar — Shannon entropy of the group-mean archetype profile.
    summary : dict
        Keys:
          ``per_cell_shannon_kw_stat`` — Kruskal-Wallis H statistic.
          ``per_cell_shannon_kw_p`` — corresponding p-value.
          ``per_group_pca_dispersion`` — {group: float} point estimates.
          ``per_group_pca_dispersion_ci`` — {group: (lo, hi)} bootstrap 95 % CI.
          ``per_group_archetype_entropy`` — {group: float} entropy of mean weight.
          ``dunn_posthoc`` — dict (from DataFrame) if scikit-posthocs installed,
                             else None.
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.stats import kruskal, entropy
    try:
        import scikit_posthocs as sp
        have_dunn = True
    except ImportError:
        have_dunn = False

    rng = np.random.default_rng(random_state)
    obs = obs.reset_index(drop=True)
    groups = sorted(obs[group_col].unique())

    # Panel 1 — per-cell Shannon H of weights
    per_cell_H = np.array([entropy(w + 1e-12) for w in weights])

    # Panel 2 — per-group PCA median pairwise dispersion with bootstrap CI
    def _median_pairwise_dist(X, max_cells):
        if X.shape[0] > max_cells:
            idx = rng.choice(X.shape[0], size=max_cells, replace=False)
            X = X[idx]
        from scipy.spatial.distance import pdist
        dists = pdist(X, metric="euclidean")
        return float(np.median(dists))

    per_group_disp = {}
    disp_ci = {}
    for g in groups:
        mask = obs[group_col].values == g
        Xg = pca[mask]
        est = _median_pairwise_dist(Xg, subsample)
        boot = []
        for _ in range(bootstrap_n):
            if Xg.shape[0] < 2:
                boot.append(float("nan"))
                continue
            sample_idx = rng.integers(0, Xg.shape[0], size=Xg.shape[0])
            boot.append(_median_pairwise_dist(Xg[sample_idx], subsample))
        boot_arr = np.asarray([b for b in boot if not np.isnan(b)])
        ci_lo = float(np.percentile(boot_arr, 2.5)) if len(boot_arr) else float("nan")
        ci_hi = float(np.percentile(boot_arr, 97.5)) if len(boot_arr) else float("nan")
        per_group_disp[g] = est
        disp_ci[g] = (ci_lo, ci_hi)

    # Panel 3 — entropy of pooled mean weight vector
    per_group_arch_H = {}
    for g in groups:
        mask = obs[group_col].values == g
        mu = weights[mask].mean(axis=0)
        per_group_arch_H[g] = float(entropy(mu + 1e-12))

    # Stats — Kruskal-Wallis on per-cell Shannon
    kw_stat, kw_p = kruskal(*[per_cell_H[obs[group_col].values == g] for g in groups])

    # Dunn pairwise post-hoc (optional)
    dunn_df = None
    if have_dunn:
        df_long = pd.DataFrame({"entropy": per_cell_H, "group": obs[group_col].values})
        dunn_df = sp.posthoc_dunn(df_long, val_col="entropy", group_col="group",
                                   p_adjust="fdr_bh")

    # Build figure — 3 panels
    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        "Per-cell Shannon H (weights)",
        "Per-group PCA dispersion (bootstrap)",
        "Per-group entropy of mean archetype profile",
    ))
    # Panel 1: violin
    for g in groups:
        mask = obs[group_col].values == g
        fig.add_trace(go.Violin(
            y=per_cell_H[mask], name=str(g), points="outliers",
            box_visible=True, showlegend=False,
        ), row=1, col=1)
    # Panel 2: bar with CI
    xs = list(groups)
    ys = [per_group_disp[g] for g in xs]
    err_lo = [per_group_disp[g] - disp_ci[g][0] for g in xs]
    err_hi = [disp_ci[g][1] - per_group_disp[g] for g in xs]
    fig.add_trace(go.Bar(
        x=xs, y=ys,
        error_y=dict(type="data", array=err_hi, arrayminus=err_lo, visible=True),
        showlegend=False,
    ), row=1, col=2)
    # Panel 3: bar
    fig.add_trace(go.Bar(
        x=xs, y=[per_group_arch_H[g] for g in xs], showlegend=False,
    ), row=1, col=3)
    fig.update_layout(
        title=f"Diversity block — KW H={kw_stat:.2f}, p={kw_p:.2e}",
        height=480, width=1400,
    )

    summary = {
        "per_cell_shannon_kw_stat": float(kw_stat),
        "per_cell_shannon_kw_p": float(kw_p),
        "per_group_pca_dispersion": per_group_disp,
        "per_group_pca_dispersion_ci": disp_ci,
        "per_group_archetype_entropy": per_group_arch_H,
        "dunn_posthoc": dunn_df.to_dict() if dunn_df is not None else None,
    }
    return fig, summary


def build_chord_diagram(
    entries: list,
    K: int,
    *,
    title: str = "",
    node_labels=None,
    color_pos: str = "#d62728",
    color_neg: str = "#1f77b4",
    alpha_min: float = 0.15,
    width_min: float = 1.0,
    width_max: float = 4.0,
):
    """Build a plotly chord-style diagram connecting archetype pairs.

    Parameters
    ----------
    entries : list of dict, each with keys:
        j         int   0-indexed source archetype
        k         int   0-indexed target archetype
        weight    float chord thickness scaling (e.g., |gamma| or N sig genes)
        direction float sign: +1 → color_pos (red/rising), -1 → color_neg (blue/falling)
    K : int
        Total number of archetypes (nodes on the circle).
    title : str
        Figure title.
    node_labels : list[str] or None
        Labels for each node.  Defaults to ["A1", …, "AK"].
    color_pos, color_neg : str
        Hex colors for positive / negative direction chords.
    alpha_min : float
        Minimum chord opacity (prevents near-invisible thin chords).
    width_min, width_max : float
        Line-width range mapped to the normalized weight.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import math as _math
    import numpy as _np
    import plotly.graph_objects as go

    if node_labels is None:
        node_labels = [f"A{i + 1}" for i in range(K)]

    # Node positions on unit circle (top = A1, clockwise)
    angles = [2 * _math.pi * i / K - _math.pi / 2 for i in range(K)]
    node_x = [_math.cos(a) for a in angles]
    node_y = [_math.sin(a) for a in angles]

    # Normalize weights
    weights = [abs(float(e.get("weight", 1.0))) for e in entries]
    max_w = max(weights) if weights else 1.0

    def _hex_to_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    rgb_pos = _hex_to_rgb(color_pos)
    rgb_neg = _hex_to_rgb(color_neg)

    traces = []

    for e, w in zip(entries, weights):
        j = int(e["j"])
        k = int(e["k"])
        d = float(e.get("direction", 1.0))

        if j == k or j >= K or k >= K:
            continue

        # Quadratic bezier through origin as control point
        n_pts = 60
        t = _np.linspace(0, 1, n_pts)
        P0x, P0y = node_x[j], node_y[j]
        P1x, P1y = node_x[k], node_y[k]
        bx = (1 - t) ** 2 * P0x + t ** 2 * P1x  # Pcx = 0
        by = (1 - t) ** 2 * P0y + t ** 2 * P1y  # Pcy = 0

        w_norm = w / max_w if max_w > 0 else 0.5
        alpha = max(alpha_min, 0.85 * w_norm)
        lw = width_min + (width_max - width_min) * w_norm
        r, g_c, b_c = rgb_pos if d >= 0 else rgb_neg
        color_str = f"rgba({r},{g_c},{b_c},{alpha:.2f})"

        lbl = (
            f"{node_labels[j]}→{node_labels[k]}<br>"
            f"weight={w:.3f}<br>"
            f"{'rising' if d >= 0 else 'falling'}"
        )
        traces.append(go.Scatter(
            x=list(bx) + [None],
            y=list(by) + [None],
            mode="lines",
            line=dict(color=color_str, width=lw),
            hoverinfo="text",
            text=[lbl] * n_pts + [None],
            showlegend=False,
        ))

    # Node markers
    traces.append(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=16, color="#444444", line=dict(width=1.5, color="white")),
        text=node_labels,
        textposition="top center",
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))

    # Legend proxies
    traces.append(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color=color_pos, width=2.5),
        name="rising (γ>0) / δβ>0",
        showlegend=True,
    ))
    traces.append(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color=color_neg, width=2.5),
        name="falling (γ<0) / δβ<0",
        showlegend=True,
    ))

    layout = go.Layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False,
                   visible=False, fixedrange=True),
        yaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False,
                   visible=False, scaleanchor="x", fixedrange=True),
        width=520, height=520,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(x=1.0, y=1.0),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return go.Figure(data=traces, layout=layout)


def build_feature_chord_diagram(
    entries: list,
    K: int,
    *,
    title: str = "",
    max_nodes: int = 16,
    width_min: float = 1.0,
    width_max: float = 5.0,
):
    """Feature chord diagram: source genes on the left arc, target on the right arc.

    Position encodes role (source vs target), so each archetype needs only one
    colour. Chords are coloured by from_arch; nodes are coloured by their
    archetype (from_arch for left, to_arch for right).

    Parameters
    ----------
    entries : list of dict, each with:
        from_feature  str   gene/pathway on the source (rising) side
        to_feature    str   gene/pathway on the target (falling) side
        from_arch     int   0-indexed source archetype
        to_arch       int   0-indexed target archetype (falls back to from_arch)
        weight        float chord thickness (e.g., |gamma| or |delta_beta|)
    K : int
        Total number of archetypes (for colour palette sizing).
    max_nodes : int
        Cap on unique nodes per side.
    """
    import math as _math
    import numpy as _np
    import plotly.graph_objects as go
    import plotly.colors as _pc_colors

    palette = _pc_colors.qualitative.Plotly
    _arch_rgb = {}

    def _rgb(arch_idx):
        if arch_idx not in _arch_rgb:
            hex_c = palette[arch_idx % len(palette)].lstrip("#")
            _arch_rgb[arch_idx] = (int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16))
        return _arch_rgb[arch_idx]

    def _arch_color(arch_idx, alpha=0.65):
        r, g, b = _rgb(arch_idx)
        return f"rgba({r},{g},{b},{alpha:.2f})"

    # Collect unique nodes per side, preserving insertion order
    src_nodes, src_arch = [], {}  # from_feature → from_arch
    tgt_nodes, tgt_arch = [], {}  # to_feature   → to_arch
    for e in entries:
        ff = str(e.get("from_feature", ""))
        tf = str(e.get("to_feature", ""))
        fa = int(e.get("from_arch", 0))
        ta = int(e.get("to_arch", fa))
        if ff and ff not in src_arch and len(src_nodes) < max_nodes:
            src_nodes.append(ff)
            src_arch[ff] = fa
        if tf and tf not in tgt_arch and len(tgt_nodes) < max_nodes:
            tgt_nodes.append(tf)
            tgt_arch[tf] = ta

    if not src_nodes or not tgt_nodes:
        return go.Figure()

    # Circular arc layout: source nodes on the LEFT arc (108°–252°),
    # target nodes on the RIGHT arc (72°–(−72°)).
    # Bezier chords curve inward through the origin.
    src_angles = [
        _math.radians(108 + 144 * i / max(len(src_nodes) - 1, 1))
        for i in range(len(src_nodes))
    ]
    tgt_angles = [
        _math.radians(72 - 144 * j / max(len(tgt_nodes) - 1, 1))
        for j in range(len(tgt_nodes))
    ]

    R = 1.0   # node radius
    src_pos = {n: (R * _math.cos(a), R * _math.sin(a))
               for n, a in zip(src_nodes, src_angles)}
    tgt_pos = {n: (R * _math.cos(a), R * _math.sin(a))
               for n, a in zip(tgt_nodes, tgt_angles)}

    def _text_pos(angle_rad):
        x, y = _math.cos(angle_rad), _math.sin(angle_rad)
        if abs(x) >= abs(y):
            return "middle left" if x < 0 else "middle right"
        return "top center" if y > 0 else "bottom center"

    weights = [abs(float(e.get("weight", 1.0))) for e in entries]
    max_w = max(weights) if weights else 1.0

    traces = []
    legend_archs = set()
    n_pts = 60
    t_arr = _np.linspace(0, 1, n_pts)

    for e, w in zip(entries, weights):
        f1 = str(e.get("from_feature", ""))
        f2 = str(e.get("to_feature", ""))
        from_arch = int(e.get("from_arch", 0))
        to_arch = int(e.get("to_arch", from_arch))
        if f1 not in src_pos or f2 not in tgt_pos:
            continue
        sx, sy = src_pos[f1]
        tx, ty = tgt_pos[f2]
        w_norm = w / max_w if max_w > 0 else 0.5
        lw = width_min + (width_max - width_min) * w_norm
        alpha = 0.45 + 0.45 * w_norm
        color_str = _arch_color(from_arch, alpha=alpha)
        # Quadratic bezier with control point at origin → chord bows inward.
        bx = (1 - t_arr) ** 2 * sx + t_arr ** 2 * tx
        by = (1 - t_arr) ** 2 * sy + t_arr ** 2 * ty
        lbl = f"{f1} (A{from_arch+1}) → {f2} (A{to_arch+1})<br>weight={w:.3f}"
        traces.append(go.Scatter(
            x=list(bx) + [None], y=list(by) + [None],
            mode="lines",
            line=dict(color=color_str, width=lw),
            hoverinfo="text",
            text=[lbl] * n_pts + [None],
            showlegend=False,
        ))
        if from_arch not in legend_archs:
            legend_archs.add(from_arch)
            traces.append(go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color=_arch_color(from_arch, 0.9), width=2.5),
                name=f"A{from_arch+1}",
                showlegend=True,
            ))

    # Individual node markers — one scatter per node so each gets its own
    # legend entry and colour (avoids plotly collapsing to single-colour legend).
    for n, a in zip(src_nodes, src_angles):
        nx, ny = src_pos[n]
        traces.append(go.Scatter(
            x=[nx], y=[ny], mode="markers+text",
            marker=dict(size=12, color=_arch_color(src_arch[n], 0.9),
                        symbol="circle", line=dict(width=2, color="white")),
            text=[n], textposition=_text_pos(a),
            hovertemplate=f"{n} (source, A{src_arch[n]+1})<extra></extra>",
            showlegend=False,
        ))
    for n, a in zip(tgt_nodes, tgt_angles):
        nx, ny = tgt_pos[n]
        traces.append(go.Scatter(
            x=[nx], y=[ny], mode="markers+text",
            marker=dict(size=12, color=_arch_color(tgt_arch[n], 0.9),
                        symbol="diamond", line=dict(width=2, color="white")),
            text=[n], textposition=_text_pos(a),
            hovertemplate=f"{n} (target, A{tgt_arch[n]+1})<extra></extra>",
            showlegend=False,
        ))

    # Compact legend: shape key for source/target role
    traces.append(go.Scatter(x=[None], y=[None], mode="markers",
        marker=dict(size=10, symbol="circle",  color="#444"),
        name="source (●)", showlegend=True))
    traces.append(go.Scatter(x=[None], y=[None], mode="markers",
        marker=dict(size=10, symbol="diamond", color="#444"),
        name="target (◆)", showlegend=True))

    layout = go.Layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(range=[-1.65, 1.65], showgrid=False, zeroline=False,
                   visible=False, fixedrange=True),
        yaxis=dict(range=[-1.45, 1.45], showgrid=False, zeroline=False,
                   visible=False, scaleanchor="x", fixedrange=True),
        width=620, height=600,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(x=1.0, y=1.0),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return go.Figure(data=traces, layout=layout)


def build_bipartite_chord_diagram(
    corr_matrix,
    src_labels: list,
    tgt_labels: list,
    *,
    title: str = "",
    top_n: int = 15,
    src_color: str = "#1f77b4",
    tgt_color: str = "#ff7f0e",
    width_min: float = 1.0,
    width_max: float = 6.0,
):
    """Bipartite chord diagram for archetype correspondence.

    Source archetypes placed on the left semicircle, target on the right.
    Chords are drawn for the top_n pairs by transport mass.

    Parameters
    ----------
    corr_matrix : array-like, shape (K_src, K_tgt)
        Raw transport mass matrix.
    src_labels, tgt_labels : list[str]
        Archetype labels for source (HSC) and target (CMP).
    top_n : int
        Number of highest-mass pairs to draw.
    src_color, tgt_color : str
        Hex colours for source and target node markers.
    """
    import math as _math
    import numpy as _np
    import plotly.graph_objects as go

    corr_matrix = _np.asarray(corr_matrix)
    K_src, K_tgt = corr_matrix.shape

    def _left_pos(i, n):
        ang = _math.pi / 2 - _math.pi * i / max(n - 1, 1) if n > 1 else 0.0
        return -0.85 * _math.cos(ang), 0.85 * _math.sin(ang)

    def _right_pos(j, m):
        ang = _math.pi / 2 - _math.pi * j / max(m - 1, 1) if m > 1 else 0.0
        return 0.85 * _math.cos(ang), 0.85 * _math.sin(ang)

    src_pos = [_left_pos(i, K_src) for i in range(K_src)]
    tgt_pos = [_right_pos(j, K_tgt) for j in range(K_tgt)]

    flat = sorted(
        [(float(corr_matrix[i, j]), i, j) for i in range(K_src) for j in range(K_tgt)],
        reverse=True,
    )
    top_pairs = flat[:top_n]
    max_m = top_pairs[0][0] if top_pairs else 1.0

    def _hex_rgba(h, a):
        h = h.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{a:.2f})"

    traces = []
    n_pts = 60
    t_arr = _np.linspace(0, 1, n_pts)
    for mass, i, j in top_pairs:
        w_norm = mass / max_m if max_m > 0 else 0.5
        lw = width_min + (width_max - width_min) * w_norm
        alpha = 0.25 + 0.65 * w_norm
        color = _hex_rgba(src_color, alpha)
        sx, sy = src_pos[i]
        tx, ty = tgt_pos[j]
        bx = (1 - t_arr) ** 2 * sx + t_arr ** 2 * tx
        by = (1 - t_arr) ** 2 * sy + t_arr ** 2 * ty
        lbl = f"{src_labels[i]}→{tgt_labels[j]}<br>mass={mass:.3f}"
        traces.append(go.Scatter(
            x=list(bx) + [None], y=list(by) + [None],
            mode="lines",
            line=dict(color=color, width=lw),
            hoverinfo="text",
            text=[lbl] * n_pts + [None],
            showlegend=False,
        ))

    traces.append(go.Scatter(
        x=[p[0] for p in src_pos], y=[p[1] for p in src_pos],
        mode="markers+text",
        marker=dict(size=14, color=src_color, line=dict(width=1.5, color="white")),
        text=src_labels, textposition="middle left",
        hovertemplate="%{text}<extra></extra>",
        name="HSC archetypes", showlegend=True,
    ))
    traces.append(go.Scatter(
        x=[p[0] for p in tgt_pos], y=[p[1] for p in tgt_pos],
        mode="markers+text",
        marker=dict(size=14, color=tgt_color, line=dict(width=1.5, color="white")),
        text=tgt_labels, textposition="middle right",
        hovertemplate="%{text}<extra></extra>",
        name="CMP archetypes", showlegend=True,
    ))

    layout = go.Layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False, visible=False, fixedrange=True),
        yaxis=dict(range=[-1.2, 1.2], showgrid=False, zeroline=False, visible=False,
                   scaleanchor="x", fixedrange=True),
        width=640, height=520,
        margin=dict(l=90, r=90, t=50, b=40),
        legend=dict(x=0.35, y=-0.05, orientation="h"),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return go.Figure(data=traces, layout=layout)

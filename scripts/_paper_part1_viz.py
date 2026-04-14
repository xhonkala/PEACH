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
        "final_drift_mean",
        "final_drift_max",
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

        converged = _is_converged(dmean_hist, drift_threshold, converged_window)
        if converged:
            badges_html += _badge_html(f"{label}: STABLE", "#2ca02c")
        else:
            badges_html += _badge_html(f"{label}: DRIFTING", "#c0392b")

        def _fmt(v):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "N/A"
            return f"{v:.5f}"

        rows.append({
            "model": label,
            "final_drift_mean": _fmt(final_dmean),
            "final_drift_max": _fmt(final_dmax),
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

    losses = list((history or {}).get("loss", []) or [])

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
    ramps = {
        "NR": ["#fca5a5", "#ef4444", "#991b1b"],
        "R1": ["#fed7aa", "#f97316", "#9a3412"],
        "R2": ["#93c5fd", "#2563eb", "#1e3a8a"],
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


def compute_w2_archetype_distance(weights_a, weights_b):
    """2-Wasserstein distance between two cell groups' archetype-weight
    distributions, via the Bures–Wasserstein closed form on Gaussian
    approximations.

    Parameters
    ----------
    weights_a, weights_b : np.ndarray
        Shape ``(n_cells_*, n_archetypes)``. Each row is a simplex
        point. Groups may have different cell counts.

    Returns
    -------
    float
        Non-negative W2 distance. Returns 0.0 when both inputs are
        identical (bit-exact).
    """
    import numpy as np
    from scipy.linalg import sqrtm

    a = np.asarray(weights_a, dtype=np.float64)
    b = np.asarray(weights_b, dtype=np.float64)

    # Short-circuit: W2(P, P) = 0 by definition; avoids sqrtm roundoff.
    if a.shape == b.shape and np.array_equal(a, b):
        return 0.0

    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"Archetype dim mismatch: a={a.shape[1]}, b={b.shape[1]}"
        )
    if a.shape[0] < 2 or b.shape[0] < 2:
        raise ValueError("Each group needs at least 2 cells for a covariance.")

    mu_a = a.mean(axis=0)
    mu_b = b.mean(axis=0)
    Sig_a = np.cov(a, rowvar=False)
    Sig_b = np.cov(b, rowvar=False)

    # Numerical floor on the diagonals (archetype weights can be near-degenerate)
    eps = 1e-10
    Sig_a = Sig_a + eps * np.eye(Sig_a.shape[0])
    Sig_b = Sig_b + eps * np.eye(Sig_b.shape[0])

    # Mean-distance term
    mean_term = float(np.sum((mu_a - mu_b) ** 2))

    # Bures term: Tr(Σa + Σb - 2 * (Σa^½ Σb Σa^½)^½)
    sqrt_Sa = sqrtm(Sig_a)
    # sqrtm may return complex due to floating round-off; drop imaginary
    sqrt_Sa = np.asarray(sqrt_Sa).real
    middle = sqrt_Sa @ Sig_b @ sqrt_Sa
    sqrt_middle = sqrtm(middle)
    sqrt_middle = np.asarray(sqrt_middle).real
    bures = float(np.trace(Sig_a) + np.trace(Sig_b) - 2.0 * np.trace(sqrt_middle))

    # Numerical clamp (tiny negatives from sqrtm roundoff)
    w2_sq = max(0.0, mean_term + bures)
    return float(np.sqrt(w2_sq))


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
            d = compute_w2_archetype_distance(w1, w2)
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

    for a in archetype_ids:
        sub = obs.loc[obs[archetypes_col] == a]
        n = len(sub)
        row = {
            "archetype": int(a),
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
            row["top_genes"] = ", ".join(top_genes_by_archetype.get(int(a), []))
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

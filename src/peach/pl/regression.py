"""Regression visualization: coefficient heatmaps, R² plots, pattern summaries."""

import numpy as np
import plotly.graph_objects as go
from anndata import AnnData

from peach._core.utils.feature_utils import resolve_regression_result

from ._style import (
    CATEGORICAL_PALETTE,
    COLOR_MUTED,
    COLOR_PRIMARY,
    DIVERGING_COLORSCALE,
    SEQUENTIAL_COLORSCALE,
    apply_style,
    save_and_show,
)


def _get_regression_data(adata, feature_type="genes"):
    """Helper to extract regression results from adata.uns.

    Tries namespaced key first (e.g. peach_simplex_regression_genes),
    falls back to generic peach_simplex_regression.
    """
    result = resolve_regression_result(adata, feature_type=feature_type)
    if result is None:
        raise ValueError(
            "No regression results found. Run pc.tl.feature_simplex_regression() first."
        )
    return result


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a hex color string to an rgba() CSS string."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def coefficient_heatmap(
    adata: AnnData,
    *,
    top_n: int = 50,
    fdr_threshold: float = 0.05,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Heatmap of vertex coefficients (beta_k) for top features by R-squared.

    Parameters
    ----------
    adata : AnnData
        Must have regression results in ``uns['peach_simplex_regression']``.
    top_n : int
        Number of top features to display, ranked by R-squared.
    fdr_threshold : float
        FDR significance threshold. Only features with FDR q < threshold
        are shown (ranked by R-squared). Falls back to all features ranked
        by R-squared if none are significant.
    save_path : str or None
        If provided, save figure as HTML to this path.
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    reg = _get_regression_data(adata)
    coefs = np.asarray(reg["vertex_coefficients"])  # [n_features, K]
    names = list(reg["feature_names"])
    r2 = np.asarray(reg["r_squared_degree1"])

    # Filter to significant features before ranking by R-squared
    f_fdr = np.asarray(reg.get("f_pvalue_fdr", np.zeros(len(r2))))
    sig_mask = f_fdr < fdr_threshold
    if sig_mask.any():
        sig_idx = np.where(sig_mask)[0]
        top_idx = sig_idx[np.argsort(r2[sig_idx])[-top_n:][::-1]]
    else:
        top_idx = np.argsort(r2)[-top_n:][::-1]  # fallback
    top_coefs = coefs[top_idx]
    top_names = [names[i] for i in top_idx]

    K = coefs.shape[1]
    arch_names = [f"Archetype {k+1}" for k in range(K)]

    fig = go.Figure(data=go.Heatmap(
        z=top_coefs,
        x=arch_names,
        y=top_names,
        colorscale=DIVERGING_COLORSCALE,
        zmid=0,
        colorbar=dict(title="beta", thickness=12, len=0.6),
    ))
    n_shown = min(top_n, len(top_names))
    apply_style(fig, title=f"Vertex coefficients -- top {n_shown} by R-squared",
                height=max(400, n_shown * 18))

    return save_and_show(fig, save_path=save_path, show=show)


def interaction_heatmap(
    adata: AnnData,
    *,
    top_n: int = 50,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Heatmap of interaction coefficients (beta_{jk}) for top features.

    Parameters
    ----------
    adata : AnnData
        Must have degree-2 regression results in ``uns['peach_simplex_regression']``.
    top_n : int
        Number of top features to display, ranked by R-squared.
    save_path : str or None
        If provided, save figure as HTML to this path.
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    reg = _get_regression_data(adata)

    if reg.get("interaction_coefficients") is None:
        raise ValueError("No interaction coefficients found. Run with max_degree=2.")

    int_coefs = np.asarray(reg["interaction_coefficients"])
    names = list(reg["feature_names"])
    pairs = reg.get("interaction_pairs", [])
    r2 = np.asarray(reg.get("r_squared_degree2", reg["r_squared_degree1"]))

    top_idx = np.argsort(r2)[-top_n:][::-1]
    top_coefs = int_coefs[top_idx]
    top_names = [names[i] for i in top_idx]
    pair_names = [f"({p[0]},{p[1]})" for p in pairs]

    fig = go.Figure(data=go.Heatmap(
        z=top_coefs,
        x=pair_names,
        y=top_names,
        colorscale=DIVERGING_COLORSCALE,
        zmid=0,
        colorbar=dict(title="beta_int", thickness=12, len=0.6),
    ))
    n_shown = min(top_n, len(top_names))
    apply_style(fig, title=f"Interaction coefficients -- top {n_shown} by R-squared",
                height=max(400, n_shown * 18))

    return save_and_show(fig, save_path=save_path, show=show)


def r2_barplot(
    adata: AnnData,
    *,
    top_n: int = 30,
    per_archetype: bool = True,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Horizontal bar plot of features ranked by R-squared, with per-archetype
    coefficient magnitude breakdown.

    When ``per_archetype=True`` (default), each feature shows grouped bars --
    one bar per archetype colored by archetype color, sized by the absolute
    vertex coefficient for that archetype. A thin overlay bar shows the global
    R-squared for reference. When ``per_archetype=False``, falls back to the
    original single-bar global R-squared display.

    Parameters
    ----------
    adata : AnnData
        Must have regression results in ``uns['peach_simplex_regression']``.
    top_n : int
        Number of top features to display.
    per_archetype : bool
        If True (default), show grouped bars with per-archetype coefficient
        magnitudes alongside global R-squared. If False, show only global
        R-squared bars.
    save_path : str or None
        If provided, save figure as HTML to this path.
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    reg = _get_regression_data(adata)
    r2 = np.asarray(reg["r_squared_degree1"])
    names = list(reg["feature_names"])
    coefs = np.asarray(reg["vertex_coefficients"])  # [n_features, K]

    top_idx = np.argsort(r2)[-top_n:][::-1]
    top_r2 = r2[top_idx]
    top_names = [names[i] for i in top_idx]
    n_shown = min(top_n, len(top_names))

    if not per_archetype:
        # Original single-bar display
        fig = go.Figure(data=go.Bar(
            x=top_r2[::-1],
            y=top_names[::-1],
            orientation="h",
            marker_color=COLOR_PRIMARY,
        ))
        apply_style(fig, title=f"Top {n_shown} features by R-squared",
                    xaxis_title="R-squared",
                    height=max(400, n_shown * 18))
        return save_and_show(fig, save_path=save_path, show=show)

    # Per-archetype grouped bars: show |beta_k| for each archetype
    K = coefs.shape[1]
    top_coefs = coefs[top_idx]  # [n_shown, K]
    arch_labels = [f"A{k+1}" for k in range(K)]

    # Reverse for plotly horizontal bar (highest at top)
    display_names = top_names[::-1]
    display_coefs = top_coefs[::-1]
    display_r2 = top_r2[::-1]

    fig = go.Figure()

    # Add per-archetype bars (grouped, colored by archetype)
    for k in range(K):
        color = CATEGORICAL_PALETTE[k % len(CATEGORICAL_PALETTE)]
        fig.add_trace(go.Bar(
            x=np.abs(display_coefs[:, k]),
            y=display_names,
            orientation="h",
            name=arch_labels[k],
            marker_color=color,
            legendgroup=arch_labels[k],
            hovertemplate=(
                "%{y}<br>"
                + arch_labels[k]
                + " |beta|=%{x:.3f}<extra></extra>"
            ),
        ))

    # Overlay global R-squared as scatter markers on a secondary x-axis
    fig.add_trace(go.Scatter(
        x=display_r2,
        y=display_names,
        mode="markers",
        name="R-squared (global)",
        marker=dict(
            symbol="diamond",
            size=7,
            color="#333",
            line=dict(width=1, color="white"),
        ),
        xaxis="x2",
        hovertemplate="%{y}<br>R-squared=%{x:.3f}<extra></extra>",
    ))

    height = max(400, n_shown * (18 + 4 * K))
    apply_style(fig, title=f"Top {n_shown} features -- per-archetype |beta| + global R-squared",
                height=height)
    fig.update_layout(
        barmode="group",
        xaxis_title="|beta|",
        xaxis2=dict(
            title="R-squared",
            overlaying="x",
            side="top",
            showgrid=False,
            zeroline=False,
            linecolor="#aaa",
            linewidth=0.5,
            ticks="outside",
            ticklen=3,
            tickwidth=0.5,
            tickcolor="#aaa",
            tickfont=dict(size=10),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    return save_and_show(fig, save_path=save_path, show=show)


def vertex_radar(
    adata: AnnData,
    feature: str,
    *,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Spider/radar plot of vertex coefficients for one feature.

    Parameters
    ----------
    adata : AnnData
        Must have regression results in ``uns['peach_simplex_regression']``.
    feature : str
        Feature name to visualize.
    save_path : str or None
        If provided, save figure as HTML to this path.
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    reg = _get_regression_data(adata)
    names = list(reg["feature_names"])
    coefs = np.asarray(reg["vertex_coefficients"])

    if feature not in names:
        raise ValueError(f"Feature '{feature}' not found in regression results.")

    idx = names.index(feature)
    betas = coefs[idx]
    K = len(betas)
    arch_names = [f"Archetype {k+1}" for k in range(K)]

    fig = go.Figure(data=go.Scatterpolar(
        r=list(betas) + [betas[0]],  # close the polygon
        theta=arch_names + [arch_names[0]],
        fill="toself",
        fillcolor=f"rgba(0, 114, 178, 0.15)",
        line=dict(color=COLOR_PRIMARY, width=2),
        name=feature,
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, gridcolor="#ddd", linewidth=0),
            angularaxis=dict(linewidth=0, gridcolor="#ddd"),
            bgcolor="white",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=60, t=40, b=40),
        font=dict(family="Arial, Helvetica, sans-serif", size=12),
    )
    apply_style(fig, title=feature)
    # Radar plots need the polar layout preserved, so re-apply polar specifics
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, gridcolor="#eee", linewidth=0),
            angularaxis=dict(linewidth=0, gridcolor="#eee"),
            bgcolor="white",
        ),
    )

    return save_and_show(fig, save_path=save_path, show=show)


def regression_volcano(
    adata: AnnData,
    *,
    alpha: float = 0.05,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Scatter plot: R-squared vs max vertex contrast (max beta - min beta),
    colored by FDR significance.

    Features with FDR-corrected F-test q-value < ``alpha`` are highlighted
    in the primary color; non-significant features are shown in muted gray.

    Parameters
    ----------
    adata : AnnData
        Must have regression results in ``uns['peach_simplex_regression']``.
    alpha : float
        Significance threshold for FDR-corrected p-values (default 0.05).
        Features with q < alpha are colored as significant.
    save_path : str or None
        If provided, save figure as HTML to this path.
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    reg = _get_regression_data(adata)
    r2 = np.asarray(reg["r_squared_degree1"])
    coefs = np.asarray(reg["vertex_coefficients"])
    names = list(reg["feature_names"])

    contrast = np.ptp(coefs, axis=1)  # max - min per feature

    # Determine significance from FDR-corrected F-test p-values
    # Fall back to raw f_pvalue, then to vertex_pvalues_fdr min across archetypes
    qvals = None
    if reg.get("f_pvalue_fdr") is not None:
        qvals = np.asarray(reg["f_pvalue_fdr"])
    elif reg.get("f_pvalue") is not None:
        qvals = np.asarray(reg["f_pvalue"])
    elif reg.get("vertex_pvalues_fdr") is not None:
        # Use minimum FDR q-value across archetypes per feature
        qvals = np.min(np.asarray(reg["vertex_pvalues_fdr"]), axis=1)

    n_total = len(r2)

    if qvals is not None:
        sig_mask = qvals < alpha
        n_sig = int(np.sum(sig_mask))
        n_nonsig = len(sig_mask) - n_sig

        fig = go.Figure()

        # Non-significant points (muted gray, behind)
        if n_nonsig > 0:
            ns_idx = ~sig_mask
            fig.add_trace(go.Scatter(
                x=contrast[ns_idx],
                y=r2[ns_idx],
                mode="markers",
                text=[names[i] for i in range(len(names)) if ns_idx[i]],
                hovertemplate=(
                    "%{text}<br>R-squared=%{y:.3f}<br>Contrast=%{x:.2f}"
                    f"<br>q>={alpha:.2g}<extra></extra>"
                ),
                marker=dict(size=4, opacity=0.3, color=COLOR_MUTED),
                name=f"q >= {alpha} (n={n_nonsig})",
            ))

        # Significant points (primary color, on top)
        if n_sig > 0:
            fig.add_trace(go.Scatter(
                x=contrast[sig_mask],
                y=r2[sig_mask],
                mode="markers",
                text=[names[i] for i in range(len(names)) if sig_mask[i]],
                hovertemplate=(
                    "%{text}<br>R-squared=%{y:.3f}<br>Contrast=%{x:.2f}"
                    f"<br>q<{alpha:.2g}<extra></extra>"
                ),
                marker=dict(size=5, opacity=0.7, color=COLOR_PRIMARY),
                name=f"q < {alpha} (n={n_sig})",
            ))
    else:
        # No q-values available -- fall back to uniform coloring
        n_sig = None
        fig = go.Figure(data=go.Scatter(
            x=contrast,
            y=r2,
            mode="markers",
            text=names,
            hovertemplate="%{text}<br>R-squared=%{y:.3f}<br>Contrast=%{x:.2f}<extra></extra>",
            marker=dict(size=4, opacity=0.5, color=COLOR_PRIMARY),
        ))

    if n_sig is not None:
        volcano_title = f"Regression volcano (N={n_sig} significant / {n_total} total)"
    else:
        volcano_title = f"Regression volcano (N={n_total} features)"
    apply_style(fig, title=volcano_title,
                xaxis_title="max beta - min beta",
                yaxis_title="R-squared")

    return save_and_show(fig, save_path=save_path, show=show)


def archetype_regression_dotplot(
    adata: AnnData,
    *,
    top_n: int = 10,
    exclusive_only: bool = False,
    degree: int = 1,
    feature_type: str = "genes",
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Dotplot of top genes per archetype from regression coefficients.

    Rows: top genes per archetype (by |beta|, union across archetypes).
    Columns: archetypes (and optionally interaction pairs for degree=2).
    Dot size: |beta coefficient|.
    Dot color: -log10(vertex p-value).

    Parameters
    ----------
    adata : AnnData
        Must have regression results in ``uns['peach_simplex_regression']``.
    top_n : int
        Number of top features per archetype to include.
    exclusive_only : bool
        If True, only show features where the max coefficient is at least
        2x the second-highest coefficient across archetypes. This filters
        to archetype-exclusive features.
    degree : int
        1 = show only degree-1 (vertex) coefficients.
        2 = also show interaction term coefficients from degree-2 regression
        as additional columns.
    save_path : str or None
    show : bool

    Returns
    -------
    go.Figure
    """
    reg = _get_regression_data(adata, feature_type=feature_type)
    coefs = np.asarray(reg["vertex_coefficients"])  # [n_features, K]
    names = list(reg["feature_names"])
    pvals = np.asarray(reg.get("vertex_pvalues", np.ones_like(coefs)))

    K = coefs.shape[1]

    # Collect union of top_n genes per archetype (by |beta|)
    selected = set()
    for k in range(K):
        top_idx = np.argsort(np.abs(coefs[:, k]))[-top_n:]
        selected.update(top_idx)
    # Group features by dominant archetype, then rank by |beta| within group
    selected = sorted(selected, key=lambda i: (np.argmax(np.abs(coefs[i])), -np.max(np.abs(coefs[i]))))

    # Filter to exclusive features if requested
    if exclusive_only:
        exclusive = []
        for i in selected:
            abs_betas = np.sort(np.abs(coefs[i]))[::-1]
            if len(abs_betas) >= 2 and abs_betas[1] > 0:
                if abs_betas[0] / abs_betas[1] >= 2.0:
                    exclusive.append(i)
            elif len(abs_betas) >= 1 and abs_betas[0] > 0:
                # Only one non-zero -- trivially exclusive
                exclusive.append(i)
        selected = exclusive

    if len(selected) == 0:
        # Empty plot with message
        fig = go.Figure()
        fig.add_annotation(
            text="No features pass the exclusivity filter (max/2nd >= 2x).",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14),
        )
        apply_style(fig, title="Regression dotplot -- no exclusive features")
        return save_and_show(fig, save_path=save_path, show=show)

    # Group by dominant archetype for visual ordering (no prefix on labels)
    dom_archs = [np.argmax(np.abs(coefs[i])) for i in selected]
    gene_labels = [names[i] for i in selected]
    arch_labels = [f"A{k+1}" for k in range(K)]

    # Build column labels: archetypes + optionally interaction pairs
    col_labels = list(arch_labels)
    int_coefs = None
    int_pvals = None
    if degree >= 2 and reg.get("interaction_coefficients") is not None:
        int_coefs = np.asarray(reg["interaction_coefficients"])  # [n_features, K-choose-2]
        int_pvals_raw = reg.get("interaction_pvalues")
        int_pvals = (
            np.asarray(int_pvals_raw) if int_pvals_raw is not None
            else np.ones_like(int_coefs)
        )
        pairs = reg.get("interaction_pairs", [])
        # Pair labels use 1-indexed archetype names
        pair_labels = [f"A{p[0]+1}xA{p[1]+1}" for p in pairs]
        col_labels.extend(pair_labels)

    # Build dot arrays
    x_vals, y_vals, sizes, colors, hover = [], [], [], [], []

    # Compute max absolute coefficient for size scaling (across both vertex + interaction)
    abs_vertex = np.abs(coefs[selected])
    max_abs = abs_vertex.max() if abs_vertex.max() > 0 else 1.0
    if int_coefs is not None:
        abs_int = np.abs(int_coefs[selected])
        max_abs = max(max_abs, abs_int.max() if abs_int.max() > 0 else 0)

    for gi, gene_idx in enumerate(selected):
        # Vertex coefficients
        for k in range(K):
            x_vals.append(arch_labels[k])
            y_vals.append(gene_labels[gi])
            beta = coefs[gene_idx, k]
            pval = max(pvals[gene_idx, k], 1e-300)
            sizes.append(np.abs(beta) / max_abs * 20 + 2)
            colors.append(-np.log10(pval))
            hover.append(f"{names[gene_idx]}<br>beta={beta:.3f}<br>p={pval:.2e}")

        # Interaction coefficients (degree 2)
        if int_coefs is not None:
            for pi in range(int_coefs.shape[1]):
                x_vals.append(col_labels[K + pi])
                y_vals.append(gene_labels[gi])
                beta = int_coefs[gene_idx, pi]
                pval = max(int_pvals[gene_idx, pi], 1e-300)
                sizes.append(np.abs(beta) / max_abs * 20 + 2)
                colors.append(-np.log10(pval))
                hover.append(
                    f"{names[gene_idx]}<br>"
                    f"{col_labels[K + pi]} beta={beta:.3f}<br>p={pval:.2e}"
                )

    fig = go.Figure(data=go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="markers",
        marker=dict(
            size=sizes,
            color=colors,
            colorscale=SEQUENTIAL_COLORSCALE,
            colorbar=dict(title="-log10(p)", thickness=12, len=0.6),
            line=dict(width=0.5, color="#999"),
        ),
        text=hover,
        hovertemplate="%{text}<extra></extra>",
    ))
    n_genes = len(gene_labels)
    n_cols = len(col_labels)

    # Proportional width: scale by columns shown, capped at 1000px
    width = min(1000, max(600, n_cols * 25))

    title_parts = [f"Regression dotplot -- top {top_n} per archetype"]
    if exclusive_only:
        title_parts.append("(exclusive only)")
    if degree >= 2 and int_coefs is not None:
        title_parts.append("+ interactions")

    apply_style(fig, title=" ".join(title_parts),
                height=max(400, n_genes * 18 + 80),
                width=width)

    return save_and_show(fig, save_path=save_path, show=show)


def archetype_radar(
    adata: AnnData,
    *,
    top_n: int = 10,
    feature_type: str = "genes",
    min_degree: int = 1,
    order_by_similarity: bool = False,
    show: bool = True,
    save: str | None = None,
) -> go.Figure:
    """Radar/spider plot for archetype phenotype characterization.

    Each polygon represents a feature and each angular spoke an archetype.
    Polygon vertices are ``|vertex_coefficient|`` -- features shared across
    archetypes produce round polygons; exclusive features produce spiky ones.

    Parameters
    ----------
    adata : AnnData
        Must contain regression results (run ``pc.tl.feature_simplex_regression``
        first).
    top_n : int
        Number of top features per archetype (by ``|vertex_coefficient|``)
        to include. The union across archetypes is displayed, truncated to
        ``top_n`` total features.
    feature_type : str
        ``"genes"`` or ``"pathways"``.
    min_degree : int
        When set to 2, only include features that have a significant
        degree-2 (interaction) coefficient (FDR q < 0.05 for at least one
        interaction term). Default 1 (no interaction filter).
    order_by_similarity : bool
        If True, reorder the archetype spokes using a Fiedler vector
        (spectral 1D embedding) derived from Spearman correlation between
        archetype regression coefficient profiles. Adjacent spokes on the
        radar will have the most similar feature profiles. Default False
        (uniform angular spacing in archetype index order).
    show : bool
        Whether to display the figure interactively.
    save : str or None
        If provided, save figure to this path (format inferred from extension).

    Returns
    -------
    go.Figure
        Radar plot figure.
    """
    # ------------------------------------------------------------------
    # 1. Resolve regression result
    # ------------------------------------------------------------------
    reg = _get_regression_data(adata, feature_type=feature_type)
    coefs = np.asarray(reg["vertex_coefficients"])  # [n_features, K]
    feat_names = list(reg["feature_names"])
    K = coefs.shape[1]

    # ------------------------------------------------------------------
    # 1b. Optionally reorder archetype spokes by similarity
    #     (Spearman correlation → Fiedler vector → 1D ordering)
    # ------------------------------------------------------------------
    if order_by_similarity and K > 2:
        from scipy.stats import spearmanr
        from scipy.sparse.csgraph import laplacian

        # Spearman correlation between archetype coefficient profiles
        corr_matrix = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                corr_matrix[i, j], _ = spearmanr(coefs[:, i], coefs[:, j])
        # Similarity-based Laplacian → Fiedler vector for 1D embedding
        sim_matrix = np.maximum(0, corr_matrix)  # clip negatives for Laplacian
        np.fill_diagonal(sim_matrix, 0)
        L = laplacian(sim_matrix, normed=True)
        _eigenvalues, eigenvectors = np.linalg.eigh(L)
        fiedler = eigenvectors[:, 1]  # second smallest eigenvalue
        order = np.argsort(fiedler)
        # Reorder archetype columns and update labels
        coefs = coefs[:, order]
        arch_labels_ordered = [f"A{order[k]+1}" for k in range(K)]
    else:
        arch_labels_ordered = [f"A{k+1}" for k in range(K)]

    # ------------------------------------------------------------------
    # 2. Select top_n features per archetype (union), then truncate
    # ------------------------------------------------------------------
    selected_idx = set()
    for k in range(K):
        top_idx = np.argsort(np.abs(coefs[:, k]))[-top_n:]
        selected_idx.update(top_idx.tolist())
    # Sort by max absolute coefficient across any archetype (descending)
    selected_idx = sorted(selected_idx, key=lambda i: -np.max(np.abs(coefs[i])))

    # Filter by min_degree=2: only keep features with significant interaction terms
    if min_degree >= 2:
        int_fdr = reg.get("interaction_pvalues_fdr")
        if int_fdr is not None:
            int_fdr_arr = np.asarray(int_fdr)
            # Keep features where at least one interaction term has FDR < 0.05
            sig_mask = np.any(int_fdr_arr < 0.05, axis=1)
            selected_idx = [i for i in selected_idx if sig_mask[i]]
        # If no FDR available, skip the filter with a warning
        # (better than silently dropping everything)

    # Truncate to top_n total features (the bug fix: union can exceed top_n)
    selected_idx = selected_idx[:top_n]

    if len(selected_idx) == 0:
        # Return empty figure with annotation
        fig = go.Figure()
        fig.add_annotation(
            text="No features pass the filters.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14),
        )
        apply_style(fig, title="Radar -- no features")
        return save_and_show(fig, save_path=save, show=show)

    sel_names = []
    for i in selected_idx:
        name = feat_names[i]
        # Strip HALLMARK_ prefix for pathway readability
        if feature_type == "pathways" and name.startswith("HALLMARK_"):
            name = name[len("HALLMARK_"):]
        # Truncate long names
        if len(name) > 20:
            name = name[:18] + ".."
        sel_names.append(name)

    n_features = len(selected_idx)
    arch_labels = arch_labels_ordered

    # ------------------------------------------------------------------
    # 3. Build radar plot
    # ------------------------------------------------------------------
    abs_coefs = np.abs(coefs[selected_idx])  # [n_sel, K]
    theta_labels = arch_labels + [arch_labels[0]]  # close polygon

    fig = go.Figure()
    for fi, feat_label in enumerate(sel_names):
        r_vals = abs_coefs[fi].tolist()
        r_vals_closed = r_vals + [r_vals[0]]
        color = CATEGORICAL_PALETTE[fi % len(CATEGORICAL_PALETTE)]

        fig.add_trace(
            go.Scatterpolar(
                r=r_vals_closed,
                theta=theta_labels,
                fill="toself",
                fillcolor=_hex_to_rgba(color, 0.08),
                line=dict(color=color, width=1.5),
                name=feat_label,
                legendgroup=feat_label,
                showlegend=True,
                hovertemplate=(
                    feat_label + "<br>%{theta}: %{r:.3f}<extra></extra>"
                ),
            ),
        )

    radar_size = max(450, 350 + n_features * 10)
    apply_style(fig, title="Archetype phenotype radar",
                height=radar_size, width=radar_size)
    # Re-apply polar styling (apply_style resets to cartesian defaults)
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, gridcolor="#eee", linewidth=0),
            angularaxis=dict(linewidth=0, gridcolor="#eee"),
            bgcolor="white",
        ),
        legend=dict(
            font=dict(size=9),
            bgcolor="rgba(255,255,255,0.85)",
            borderwidth=0,
        ),
        margin=dict(l=60, r=60, t=50, b=40),
    )

    return save_and_show(fig, save_path=save, show=show)


def pattern_summary(
    adata: AnnData,
    *,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Bar chart of pattern counts from classification results.

    Parameters
    ----------
    adata : AnnData
        Must have pattern classification results in ``uns['peach_feature_patterns']``.
    save_path : str or None
        If provided, save figure as HTML to this path.
    show : bool
        Whether to call ``fig.show()``.

    Returns
    -------
    go.Figure
    """
    if "peach_feature_patterns" not in adata.uns:
        raise ValueError(
            "No pattern classification results found. "
            "Run pc.tl.classify_feature_patterns() first."
        )

    patterns = adata.uns["peach_feature_patterns"]
    counts = patterns["pattern_counts"]
    labels = list(counts.keys())
    values = list(counts.values())

    # One color per pattern type
    colors = [CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]
              for i in range(len(labels))]

    fig = go.Figure(data=go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
    ))
    apply_style(fig, title="Feature pattern distribution",
                yaxis_title="Count")

    return save_and_show(fig, save_path=save_path, show=show)

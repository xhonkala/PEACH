"""Ternary facet plots for simplex visualization.

Extracts barycentric weights for any 3 archetypes, renormalizes to sum to 1,
and plots on triangular axes using plotly's built-in ternary support.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import scipy.sparse as sp
from anndata import AnnData
from itertools import combinations


def ternary_facet(
    adata: AnnData,
    archetypes: tuple[int, int, int] = (0, 1, 2),
    *,
    color_by: str | None = None,
    style: str = "scatter",
    resolution: int = 50,
    save_path: str | None = None,
    show: bool = True,
    **kwargs,
) -> go.Figure:
    """Ternary plot for 3 selected archetypes.

    Extracts the 3 barycentric weights from the full simplex, renormalizes
    them to sum to 1, and plots on a triangular axis.

    Parameters
    ----------
    adata : AnnData
        Must have archetype weights in ``obsm['cell_archetype_weights']``.
    archetypes : tuple of 3 ints
        Which archetypes to show. Default ``(0, 1, 2)``.
    color_by : str or None
        Gene name (from ``var_names``), obs column name, or ``'density'``
        for kernel density coloring. If None, uniform coloring.
    style : str
        ``'scatter'`` (default), ``'contour'``, or ``'relief'``.
        Currently only scatter is implemented.
    resolution : int
        Grid resolution for contour/relief styles.
    save_path : str or None
        If provided, save figure as HTML to this path.
    show : bool
        Whether to call ``fig.show()``. Set False for non-interactive use.
    **kwargs
        Additional keyword arguments passed to scatter marker styling.

    Returns
    -------
    plotly.graph_objects.Figure
        The ternary scatter figure.

    Raises
    ------
    ValueError
        If any archetype index is out of range for the number of archetypes.
    KeyError
        If archetype weights are not found in ``adata.obsm``.
    """
    from peach._core.utils.feature_utils import get_archetype_weights

    weights = get_archetype_weights(adata)
    K = weights.shape[1]
    i, j, k = archetypes

    if max(archetypes) >= K:
        raise ValueError(
            f"Archetype indices {archetypes} out of range for K={K} archetypes."
        )

    # Extract and renormalize 3 weights to the sub-simplex
    w3 = weights[:, [i, j, k]]
    w3_sum = w3.sum(axis=1, keepdims=True)
    w3_norm = w3 / np.maximum(w3_sum, 1e-10)

    # Resolve color values
    color_values = None
    colorbar_title = ""
    if color_by is not None:
        if color_by == "density":
            # Density coloring: use sum of original 3 weights as a proxy
            # (cells with higher combined weight for these 3 archetypes are
            # more "relevant" to this sub-simplex)
            color_values = w3_sum.ravel()
            colorbar_title = "Sub-simplex relevance"
        elif color_by in adata.obs.columns:
            color_values = adata.obs[color_by].values
            colorbar_title = color_by
        elif color_by in adata.var_names:
            gene_idx = list(adata.var_names).index(color_by)
            if sp.issparse(adata.X):
                color_values = np.asarray(adata.X[:, gene_idx].todense()).ravel()
            else:
                color_values = np.asarray(adata.X[:, gene_idx]).ravel()
            colorbar_title = color_by

    # Build plotly ternary scatter
    fig = go.Figure()

    marker_opts = {
        "size": kwargs.pop("marker_size", 3),
        "opacity": kwargs.pop("marker_opacity", 0.6),
    }

    # Numeric continuous coloring
    if color_values is not None and np.issubdtype(
        np.asarray(color_values).dtype, np.number
    ):
        marker_opts["color"] = color_values
        marker_opts["colorscale"] = kwargs.pop("colorscale", "Viridis")
        marker_opts["colorbar"] = {"title": colorbar_title}

    fig.add_trace(
        go.Scatterternary(
            a=w3_norm[:, 0],
            b=w3_norm[:, 1],
            c=w3_norm[:, 2],
            mode="markers",
            marker=marker_opts,
        )
    )

    fig.update_layout(
        ternary={
            "sum": 1,
            "aaxis": {"title": f"Archetype {i}", "min": 0, "linewidth": 2},
            "baxis": {"title": f"Archetype {j}", "min": 0, "linewidth": 2},
            "caxis": {"title": f"Archetype {k}", "min": 0, "linewidth": 2},
        },
        title=f"Ternary: Archetypes ({i}, {j}, {k})",
        width=600,
        height=500,
    )

    if save_path:
        fig.write_html(save_path)

    if show:
        fig.show()

    return fig


def ternary_facet_grid(
    adata: AnnData,
    *,
    color_by: str | None = None,
    facets: str | list[tuple[int, int, int]] = "all",
    ncols: int = 3,
    save_path: str | None = None,
    show: bool = True,
    **kwargs,
) -> list[go.Figure]:
    """Generate multiple ternary facets for all or selected archetype triples.

    Parameters
    ----------
    adata : AnnData
        Must have archetype weights in ``obsm['cell_archetype_weights']``.
    color_by : str or None
        Passed to each :func:`ternary_facet` call.
    facets : ``'all'`` or list of (i, j, k) tuples
        If ``'all'``, generates C(K, 3) plots for all archetype triples.
    ncols : int
        Columns in grid layout (reserved for future subplot support).
    save_path : str or None
        If provided, saves a combined HTML. Individual figures are not saved
        separately.
    show : bool
        Whether to call ``fig.show()`` on each figure. Set False for
        non-interactive use.
    **kwargs
        Additional keyword arguments passed to :func:`ternary_facet`.

    Returns
    -------
    list of plotly.graph_objects.Figure
        One figure per archetype triple.
    """
    from peach._core.utils.feature_utils import get_archetype_weights

    weights = get_archetype_weights(adata)
    K = weights.shape[1]

    if facets == "all":
        triples = list(combinations(range(K), 3))
    else:
        triples = facets

    figures = []
    for triple in triples:
        fig = ternary_facet(
            adata, archetypes=triple, color_by=color_by, show=show, **kwargs
        )
        figures.append(fig)

    return figures

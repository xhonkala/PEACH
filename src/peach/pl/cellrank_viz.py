"""Visualization functions for CellRank archetypal trajectories."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def fate_probabilities(adata, lineages=None, basis="X_umap", same_plot=False, ncols=3, figsize=None, **kwargs):
    """
    Plot fate probabilities on embedding.

    Visualizes the probability of cells committing to each lineage/archetype
    using CellRank's plotting functionality.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with fate probabilities computed.
    lineages : list of str, optional
        Specific lineages to plot. If None, plots all lineages.
    basis : str, default: 'X_umap'
        Embedding to use ('X_umap', 'X_pca', etc.).
    same_plot : bool, default: False
        If True, plots all lineages on same axes (pie chart style).
        If False, creates separate panel per lineage.
    ncols : int, default: 3
        Number of columns for multi-panel plot (if same_plot=False).
    figsize : tuple, optional
        Figure size. If None, automatically determined.
    **kwargs
        Additional arguments passed to CellRank's plot function.

    Returns
    -------
    None
        Displays matplotlib figure. CellRank's plotting functions
        display directly rather than returning figure objects.

    Raises
    ------
    ImportError
        If CellRank is not installed.
    NotImplementedError
        If GPCCA object not stored in adata.uns['cellrank_gpcca'].

    Examples
    --------
    >>> pc.pl.fate_probabilities(adata, same_plot=False, ncols=3)

    >>> pc.pl.fate_probabilities(adata, lineages=["archetype_3", "archetype_5"], same_plot=True)
    """
    try:
        import cellrank as cr
    except ImportError:
        raise ImportError("CellRank not installed. Install with: pip install cellrank")

    if "cellrank_gpcca" not in adata.uns:
        # Reconstruct GPCCA object from stored fate probabilities
        raise NotImplementedError(
            "Direct plotting from stored fate probabilities not yet implemented. "
            "Please store GPCCA object: adata.uns['cellrank_gpcca'] = g"
        )

    g = adata.uns["cellrank_gpcca"]

    # Use CellRank's plotting function
    # Note: CellRank uses 'states' parameter, not 'lineages'
    g.plot_fate_probabilities(
        states=lineages, same_plot=same_plot, basis=basis, ncols=ncols, figsize=figsize, **kwargs
    )


def gene_trends(adata, genes, lineage, time_key, model=None, data_key="X", ncols=3, figsize=None, **kwargs):
    """
    Plot gene expression trends along lineage pseudotime.

    Visualizes how gene expression changes as cells progress toward
    a specific lineage/archetype.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    genes : str or list of str
        Gene name(s) to plot.
    lineage : str
        Target lineage name (e.g., 'archetype_5').
    time_key : str
        Key in adata.obs containing pseudotime
        (e.g., 'pseudotime_to_archetype_5').
    model : cellrank.models.BaseModel, optional
        Model for fitting trends (e.g., cr.models.GAMR(adata)).
        If None, plots raw scatter without smoothed fit.
    data_key : str, default: 'X'
        Key in adata layers for expression data.
    ncols : int, default: 3
        Number of columns for multi-gene plot.
    figsize : tuple, optional
        Figure size.
    **kwargs
        Additional arguments passed to CellRank's gene_trends.

    Returns
    -------
    None
        Displays matplotlib figure directly. CellRank's plotting
        functions display rather than returning figure objects.

    Raises
    ------
    ImportError
        If CellRank is not installed.
    ValueError
        If time_key not found in adata.obs.

    Notes
    -----
    GAMR models require R installation with mgcv package and R_HOME set:

    >>> import os
    >>> os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources"

    Examples
    --------
    >>> pc.pl.gene_trends(
    ...     adata,
    ...     genes=["RARRES1", "SOD2"],
    ...     lineage="archetype_5",
    ...     time_key="pseudotime_to_archetype_5",
    ...     model=cr.models.GAMR(adata),
    ... )
    """
    try:
        import cellrank as cr
    except ImportError:
        raise ImportError("CellRank not installed. Install with: pip install cellrank")

    # Ensure genes is a list
    if isinstance(genes, str):
        genes = [genes]

    # Check pseudotime exists
    if time_key not in adata.obs.columns:
        raise ValueError(f"Pseudotime '{time_key}' not found in adata.obs. Run compute_lineage_pseudotimes() first.")

    # Use CellRank's plotting function
    cr.pl.gene_trends(
        adata, model=model, genes=genes, data_key=data_key, time_key=time_key, ncols=ncols, figsize=figsize, **kwargs
    )


def lineage_drivers(adata, lineage, n_genes=20, driver_key=None, figsize=(10, 8), **kwargs):
    """
    Plot heatmap of top driver genes for a lineage.

    Visualizes expression of genes most correlated with commitment to a specific
    lineage/archetype.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    lineage : str
        Target lineage name (e.g., 'archetype_5')
    n_genes : int, optional (default: 20)
        Number of top genes to plot
    driver_key : str, optional
        Key in adata.varm containing driver gene scores.
        If None, computes drivers on-the-fly using correlation method
    figsize : tuple, optional (default: (10, 8))
        Figure size
    **kwargs
        Additional arguments passed to seaborn.heatmap

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object

    Examples
    --------
    Plot top 20 driver genes:

    >>> import peach as pc
    >>> pc.pl.lineage_drivers(adata, lineage="archetype_5", n_genes=20)

    Custom number of genes:

    >>> pc.pl.lineage_drivers(adata, lineage="archetype_3", n_genes=30, figsize=(12, 10))

    Using pre-computed drivers:

    >>> drivers = pc.tl.compute_lineage_drivers(adata, lineage="archetype_5")
    >>> adata.var["driver_scores"] = drivers[f"archetype_5_corr"]
    >>> pc.pl.lineage_drivers(adata, lineage="archetype_5", driver_key="driver_scores")

    Notes
    -----
    - If driver_key is None, uses simple correlation method
    - Heatmap rows are cells ordered by fate probability
    - Heatmap columns are top driver genes

    See Also
    --------
    gene_trends : Plot expression trends along pseudotime
    fate_probabilities : Visualize fate probability distributions
    """
    import pandas as pd
    from scipy.stats import spearmanr

    # Get or compute driver genes
    if driver_key is not None:
        if driver_key not in adata.var.columns:
            raise ValueError(f"Driver key '{driver_key}' not found in adata.var")
        driver_scores = adata.var[driver_key]
        top_genes = driver_scores.nlargest(n_genes).index.tolist()
    else:
        # Compute on-the-fly using correlation
        if "lineage_names" not in adata.uns:
            raise ValueError("Run setup_cellrank() first")

        if lineage not in adata.uns["lineage_names"]:
            raise ValueError(f"Lineage '{lineage}' not found")

        lineage_idx = adata.uns["lineage_names"].index(lineage)
        fate_prob = adata.obsm["fate_probabilities"][:, lineage_idx]

        # Compute correlations
        results = []
        for gene in adata.var_names:
            expr = adata[:, gene].X.toarray().flatten() if hasattr(adata.X, "toarray") else adata[:, gene].X.flatten()
            corr, _ = spearmanr(fate_prob, expr)
            results.append({"gene": gene, "corr": corr})

        df = pd.DataFrame(results)
        top_genes = df.nlargest(n_genes, "corr")["gene"].tolist()

    # Get expression matrix for top genes
    expr_matrix = adata[:, top_genes].X
    if hasattr(expr_matrix, "toarray"):
        expr_matrix = expr_matrix.toarray()

    # Order cells by fate probability
    lineage_idx = adata.uns["lineage_names"].index(lineage)
    fate_prob = adata.obsm["fate_probabilities"][:, lineage_idx]
    order = np.argsort(fate_prob)

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        expr_matrix[order, :].T,
        xticklabels=False,
        yticklabels=top_genes,
        cmap="viridis",
        cbar_kws={"label": "Expression"},
        ax=ax,
        **kwargs,
    )

    ax.set_xlabel(f"Cells (ordered by {lineage} fate probability)", fontsize=12)
    ax.set_ylabel("Genes", fontsize=12)
    ax.set_title(f"Top {n_genes} Driver Genes for {lineage}", fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig

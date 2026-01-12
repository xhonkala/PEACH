"""
CV Results for Hyperparameter Selection
=======================================

Bridge between Phase 2 (CV Search) and Phase 3 (Manual Selection).

This module provides data structures for organizing, analyzing, and
visualizing cross-validation results. It supports user decision-making
but does NOT perform automatic model selection.

Main Classes
------------
CVResults : Results for a single hyperparameter configuration
CVSummary : Complete search results with analysis and visualization

Type Definitions
----------------
See ``peach._core.types`` for Pydantic models that can validate these structures.

Examples
--------
>>> # CVSummary is returned by ArchetypalGridSearch.fit()
>>> cv_summary = grid_search.fit(dataloader, base_config)
>>> # Text summary for decision support
>>> print(cv_summary.summary_report())
>>> # Rank by different metrics
>>> by_r2 = cv_summary.rank_by_metric("archetype_r2")
>>> by_rmse = cv_summary.rank_by_metric("rmse")
>>> # Visualize
>>> fig = cv_summary.plot_elbow_r2()
>>> fig.show()
"""

import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Metric name mapping: user-friendly names → DataFrame column names
METRIC_MAP = {
    "r2": "mean_archetype_r2",
    "archetype_r2": "mean_archetype_r2",
    "rmse": "mean_val_rmse",
    "val_rmse": "mean_val_rmse",
    "train_rmse": "mean_train_rmse",
    "mae": "mean_val_mae",
    "val_mae": "mean_val_mae",
    "train_mae": "mean_train_mae",
    "archetypal_loss": "mean_train_archetypal_loss",
    "train_archetypal_loss": "mean_train_archetypal_loss",
    "convergence_epoch": "mean_convergence_epoch",
}


@dataclass
class CVResults:
    """Results from cross-validation for a single hyperparameter configuration.

    Contains per-fold results and aggregated statistics for one set of
    hyperparameters tested during grid search.

    Attributes
    ----------
    hyperparameters : dict
        Tested hyperparameters:

        - ``n_archetypes`` : int
        - ``hidden_dims`` : list[int]
        - ``inflation_factor`` : float
        - ``use_pcha_init`` : bool
        - ``use_inflation`` : bool

    fold_results : list[dict[str, float]]
        Per-fold metrics. Each dict contains:

        - ``train_loss``, ``train_archetypal_loss``, etc. : Training metrics
        - ``val_rmse``, ``val_mae``, ``val_archetype_r2`` : Validation metrics
        - ``convergence_epoch`` : int
        - ``early_stopped`` : bool
        - ``archetype_r2`` : float (primary metric)

    mean_metrics : dict[str, float]
        Mean value of each metric across folds.
    std_metrics : dict[str, float]
        Standard deviation of each metric across folds.
    best_fold_idx : int
        Index of fold with highest archetype_r2.
    convergence_epochs : list[int]
        Convergence epoch for each fold.
    training_time : float, default: 0.0
        Total training time in seconds (set by caller).
    fold_histories : list[dict[str, list[float]]] | None, default: None
        Optional epoch-by-epoch training history per fold.

    Examples
    --------
    >>> # CVResults is created by CVTrainingManager
    >>> cv_result = manager.train_cv_configuration(hyperparams, cv_splits)
    >>> # Access aggregated metrics
    >>> print(f"Mean R²: {cv_result.mean_metrics['archetype_r2']:.4f}")
    >>> print(f"Std R²: {cv_result.std_metrics['archetype_r2']:.4f}")
    >>> # Get summary for specific metric
    >>> summary = cv_result.get_metric_summary("val_rmse")
    >>> print(f"RMSE: {summary['mean']:.4f} ± {summary['std']:.4f}")
    """

    hyperparameters: dict[str, Any]
    fold_results: list[dict[str, float]]
    mean_metrics: dict[str, float]
    std_metrics: dict[str, float]
    best_fold_idx: int
    convergence_epochs: list[int]
    training_time: float = 0.0
    fold_histories: list[dict[str, list[float]]] = None

    def get_metric_summary(self, metric: str) -> dict[str, float]:
        """Get summary statistics for a specific metric across folds.

        Parameters
        ----------
        metric : str
            Metric name (e.g., 'archetype_r2', 'val_rmse').

        Returns
        -------
        dict[str, float]
            Summary statistics:

            - ``mean`` : float - Mean across folds
            - ``std`` : float - Standard deviation
            - ``min`` : float - Minimum value
            - ``max`` : float - Maximum value

            Returns NaN values if metric not found or all values are NaN.

        Examples
        --------
        >>> summary = cv_result.get_metric_summary("archetype_r2")
        >>> print(f"R²: {summary['mean']:.4f} (range: {summary['min']:.4f}-{summary['max']:.4f})")
        """
        values = [fold.get(metric, np.nan) for fold in self.fold_results]
        valid_values = [v for v in values if not np.isnan(v)]

        if not valid_values:
            return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}

        return {
            "mean": np.mean(valid_values),
            "std": np.std(valid_values),
            "min": np.min(valid_values),
            "max": np.max(valid_values),
        }

    def get_aggregated_history(self, metric: str = "rmse") -> dict[str, list[float]]:
        """Get aggregated training history across folds for visualization.

        Parameters
        ----------
        metric : str, default: 'rmse'
            Metric to aggregate (must be in fold_histories).

        Returns
        -------
        dict[str, list[float]]
            Aggregated history:

            - ``{metric}_mean`` : list[float] - Mean at each epoch
            - ``{metric}_std`` : list[float] - Std at each epoch

            Returns empty dict if fold_histories not available.

        Notes
        -----
        Histories are truncated to the minimum length across folds
        (to handle early stopping at different epochs).

        Examples
        --------
        >>> history = cv_result.get_aggregated_history("loss")
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(history["loss_mean"])
        >>> plt.fill_between(
        ...     range(len(history["loss_mean"])),
        ...     np.array(history["loss_mean"]) - np.array(history["loss_std"]),
        ...     np.array(history["loss_mean"]) + np.array(history["loss_std"]),
        ...     alpha=0.3,
        ... )
        """
        if not self.fold_histories:
            return {}

        # Collect histories for the metric
        metric_histories = []
        for fold_hist in self.fold_histories:
            if fold_hist and metric in fold_hist:
                metric_histories.append(fold_hist[metric])

        if not metric_histories:
            return {}

        # Find minimum length to handle early stopping
        min_length = min(len(h) for h in metric_histories)

        # Truncate all histories to same length
        aligned_histories = [h[:min_length] for h in metric_histories]

        # Calculate mean and std at each epoch
        mean_history = []
        std_history = []
        for epoch in range(min_length):
            values = [h[epoch] for h in aligned_histories]
            mean_history.append(np.mean(values))
            std_history.append(np.std(values))

        return {f"{metric}_mean": mean_history, f"{metric}_std": std_history}


@dataclass
class CVSummary:
    """Complete cross-validation results with analysis and visualization.

    The definitive format for CV results. Provides methods for ranking
    configurations, generating visualizations, and creating decision
    support reports.

    Attributes
    ----------
    config_results : dict[str, CVResults]
        Results per configuration. Keys are formatted as
        "n_arch={N}_hidden={dims}".
    summary_df : pd.DataFrame
        Summary table with one row per configuration, columns for
        hyperparameters and all metrics (mean_ and std_ prefixed).
    ranked_configs : list[dict]
        Configurations ranked by archetype_r2. Each dict has:

        - ``hyperparameters`` : dict
        - ``metric_value`` : float
        - ``std_error`` : float
        - ``config_summary`` : str

    cv_info : dict
        Search metadata:

        - ``n_configurations`` : int
        - ``cv_folds`` : int
        - ``total_training_runs`` : int
        - ``dataset_info`` : dict
        - ``total_training_time`` : float

    search_config : SearchConfig | None
        Original search configuration (if available).
    timestamp : str
        ISO format timestamp of when results were created.

    Examples
    --------
    >>> # CVSummary is returned by ArchetypalGridSearch.fit()
    >>> cv_summary = grid_search.fit(dataloader, base_config)
    >>> # Text summary
    >>> print(cv_summary.summary_report())
    >>> # Rank by different metrics
    >>> top_by_r2 = cv_summary.rank_by_metric("archetype_r2")[:5]
    >>> top_by_rmse = cv_summary.rank_by_metric("rmse")[:5]
    >>> # Visualizations
    >>> fig = cv_summary.plot_elbow_r2()  # Primary visualization
    >>> fig = cv_summary.plot_metric("rmse")  # Any metric
    >>> # Save/load
    >>> cv_summary.save("results.pkl")
    >>> loaded = CVSummary.load("results.pkl")

    See Also
    --------
    CVResults : Per-configuration results
    ArchetypalGridSearch : Creates CVSummary via fit()
    peach.tl.hyperparameter_search : User-facing wrapper
    """

    config_results: dict[str, CVResults]
    summary_df: pd.DataFrame
    ranked_configs: list[dict]
    cv_info: dict[str, Any]
    search_config: Any = None
    timestamp: str = field(default_factory=lambda: pd.Timestamp.now().isoformat())

    @classmethod
    def from_cv_results(cls, cv_results: list[CVResults], search_config: Any, data_info: dict[str, Any]) -> "CVSummary":
        """Create CVSummary from list of CVResults.

        Factory method that organizes raw CV results into the summary format.

        Parameters
        ----------
        cv_results : list[CVResults]
            Results from hyperparameter search.
        search_config : SearchConfig
            Search configuration used.
        data_info : dict
            Dataset information from _get_data_info().

        Returns
        -------
        CVSummary
            Organized results with rankings and metadata.

        Examples
        --------
        >>> # Usually called internally by ArchetypalGridSearch
        >>> cv_summary = CVSummary.from_cv_results(cv_results, search_config, data_info)
        """
        # Build config_results dict
        config_results = {}
        for cv_result in cv_results:
            config_key = (
                f"n_arch={cv_result.hyperparameters['n_archetypes']}_hidden={cv_result.hyperparameters['hidden_dims']}"
            )
            config_results[config_key] = cv_result

        # Build summary DataFrame
        summary_df = cls._build_summary_dataframe(cv_results)

        # Build ranked configs (default: archetype_r2)
        ranked_configs = cls._rank_configurations(cv_results, "archetype_r2")

        # Build CV info
        cv_info = {
            "n_configurations": len(cv_results),
            "cv_folds": search_config.cv_folds if search_config else "Unknown",
            "total_training_runs": len(cv_results) * (search_config.cv_folds if search_config else "Unknown"),
            "dataset_info": data_info,
            "total_training_time": sum(cv.training_time for cv in cv_results),
        }

        return cls(
            config_results=config_results,
            summary_df=summary_df,
            ranked_configs=ranked_configs,
            cv_info=cv_info,
            search_config=search_config,
        )

    @staticmethod
    def _build_summary_dataframe(cv_results: list[CVResults]) -> pd.DataFrame:
        """Build the summary DataFrame from CV results."""
        rows = []

        for cv_result in cv_results:
            row = cv_result.hyperparameters.copy()

            # Convert hidden_dims list to string for pandas compatibility
            if "hidden_dims" in row:
                row["hidden_dims"] = str(row["hidden_dims"])

            # Add mean metrics
            for metric, value in cv_result.mean_metrics.items():
                row[f"mean_{metric}"] = value

            # Add std metrics
            for metric, value in cv_result.std_metrics.items():
                row[f"std_{metric}"] = value

            # Add summary statistics
            row["training_time"] = cv_result.training_time
            row["mean_convergence_epoch"] = np.mean(cv_result.convergence_epochs)
            row["early_stopping_rate"] = np.mean([fold.get("early_stopped", False) for fold in cv_result.fold_results])

            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def _rank_configurations(cv_results: list[CVResults], metric: str = "archetype_r2") -> list[dict]:
        """Rank configurations by specified metric."""
        rankings = []

        for cv_result in cv_results:
            value = cv_result.mean_metrics.get(metric, -float("inf"))

            # Build config summary including inflation factor if present
            n_arch = cv_result.hyperparameters["n_archetypes"]
            hidden = cv_result.hyperparameters["hidden_dims"]
            inflation = cv_result.hyperparameters.get("inflation_factor")

            if inflation is not None:
                config_str = f"{n_arch} archetypes, {hidden} hidden dims, λ={inflation}"
            else:
                config_str = f"{n_arch} archetypes, {hidden} hidden dims"

            rankings.append(
                {
                    "hyperparameters": cv_result.hyperparameters,
                    "metric_value": value,
                    "std_error": cv_result.std_metrics.get(metric, 0),
                    "config_summary": config_str,
                }
            )

        # Sort by metric value (descending for most metrics)
        ascending = metric in ["rmse", "val_rmse", "train_archetypal_loss", "convergence_epoch"]
        rankings.sort(key=lambda x: x["metric_value"], reverse=not ascending)

        return rankings

    def rank_by_metric(self, metric: str) -> list[dict]:
        """Rank configurations by specified metric.

        Parameters
        ----------
        metric : str
            Metric name to rank by. Accepts user-friendly names:

            - ``'r2'``, ``'archetype_r2'`` : Archetype R² (higher is better)
            - ``'rmse'``, ``'val_rmse'`` : Validation RMSE (lower is better)
            - ``'mae'``, ``'val_mae'`` : Validation MAE (lower is better)
            - ``'convergence_epoch'`` : Convergence speed (lower is better)

        Returns
        -------
        list[dict]
            Configurations sorted by metric. Each dict contains:

            - ``hyperparameters`` : dict - Configuration parameters
            - ``metric_value`` : float - Value of ranking metric
            - ``std_error`` : float - Standard error across folds
            - ``config_summary`` : str - Human-readable summary

        Examples
        --------
        >>> # Rank by R² (default)
        >>> top_configs = cv_summary.rank_by_metric("archetype_r2")
        >>> print(f"Best: {top_configs[0]['config_summary']}")
        >>> # Rank by RMSE (lower is better)
        >>> by_rmse = cv_summary.rank_by_metric("rmse")
        """
        cv_results_list = list(self.config_results.values())
        return self._rank_configurations(cv_results_list, metric)

    def get_plot_data(self) -> dict[str, pd.DataFrame]:
        """Get data formatted for plotting.

        Returns
        -------
        dict[str, pd.DataFrame]
            DataFrames for different plot types:

            - ``'summary'`` : Full summary table
            - ``'elbow'`` : Aggregated by n_archetypes (for elbow plots)
            - ``'convergence'`` : Convergence analysis data
            - ``'fold_consistency'`` : Per-fold data (for box plots)

        Examples
        --------
        >>> plot_data = cv_summary.get_plot_data()
        >>> elbow_df = plot_data["elbow"]
        >>> # Custom plotting with elbow_df
        """
        plot_data = {}

        # Main summary table
        plot_data["summary"] = self.summary_df.copy()

        # Elbow plot data (all metrics vs n_archetypes)
        # Find all mean_ and std_ columns for aggregation
        mean_cols = [col for col in self.summary_df.columns if col.startswith("mean_")]
        std_cols = [col for col in self.summary_df.columns if col.startswith("std_")]

        if mean_cols:
            agg_dict = {}
            for col in mean_cols:
                agg_dict[col] = "mean"
            for col in std_cols:
                agg_dict[col] = "mean"

            # Group by inflation_factor if present in columns
            group_cols = ["n_archetypes", "hidden_dims"]
            if "inflation_factor" in self.summary_df.columns:
                group_cols.append("inflation_factor")

            elbow_data = self.summary_df.groupby(group_cols).agg(agg_dict).reset_index()
            plot_data["elbow"] = elbow_data

        # Convergence analysis data
        if "mean_convergence_epoch" in self.summary_df.columns:
            conv_cols = ["n_archetypes", "hidden_dims", "mean_convergence_epoch", "early_stopping_rate"]
            # Include inflation_factor if present
            if "inflation_factor" in self.summary_df.columns:
                conv_cols.insert(2, "inflation_factor")  # Add after hidden_dims
            conv_data = self.summary_df[conv_cols].copy()
            plot_data["convergence"] = conv_data

        # Fold consistency data (for box plots)
        fold_data = []
        for config_key, cv_result in self.config_results.items():
            for fold_idx, fold_result in enumerate(cv_result.fold_results):
                for metric, value in fold_result.items():
                    fold_data.append(
                        {
                            "config": config_key,
                            "n_archetypes": cv_result.hyperparameters["n_archetypes"],
                            "hidden_dims": str(cv_result.hyperparameters["hidden_dims"]),
                            "inflation_factor": cv_result.hyperparameters.get("inflation_factor", 1.5),
                            "fold_idx": fold_idx,
                            "metric": metric,
                            "value": value,
                        }
                    )
        plot_data["fold_consistency"] = pd.DataFrame(fold_data)

        return plot_data

    def _map_metric_name(self, metric: str) -> str:
        """
        Map user-friendly metric name to DataFrame column name.

        Args:
            metric: User-friendly metric name (e.g., 'rmse', 'r2')

        Returns
        -------
            DataFrame column name (e.g., 'mean_val_rmse', 'mean_archetype_r2')
        """
        # Try exact match first
        if metric in self.summary_df.columns:
            return metric

        # Try mapping
        if metric in METRIC_MAP:
            mapped = METRIC_MAP[metric]
            if mapped in self.summary_df.columns:
                return mapped

        # Try adding 'mean_' prefix
        if f"mean_{metric}" in self.summary_df.columns:
            return f"mean_{metric}"

        # Return original if no mapping found (will error later with clear message)
        return metric

    def _detect_facet_strategy(self, elbow_df: pd.DataFrame) -> dict[str, str]:
        """
        Detect optimal faceting strategy based on available dimensions.

        Args:
            elbow_df: Elbow plot DataFrame

        Returns
        -------
            Dict with 'facet_by', 'color_by', 'facet_label', 'color_label' keys
        """
        has_inflation = "inflation_factor" in elbow_df.columns
        has_hidden = "hidden_dims" in elbow_df.columns

        if has_inflation and has_hidden:
            # Both available: facet by inflation (rows), color by hidden_dims
            n_inflation = elbow_df["inflation_factor"].nunique()
            return {
                "facet_by": "inflation_factor",
                "color_by": "hidden_dims",
                "facet_label": "Inflation Factor (λ)",
                "color_label": "Architecture",
                "n_facets": n_inflation,
            }
        elif has_inflation:
            # Only inflation: facet by inflation
            n_inflation = elbow_df["inflation_factor"].nunique()
            return {
                "facet_by": "inflation_factor",
                "color_by": None,
                "facet_label": "Inflation Factor (λ)",
                "color_label": None,
                "n_facets": n_inflation,
            }
        elif has_hidden:
            # Only hidden_dims: color by hidden_dims (no facets)
            return {
                "facet_by": None,
                "color_by": "hidden_dims",
                "facet_label": None,
                "color_label": "Architecture",
                "n_facets": 1,
            }
        else:
            # Neither: single line
            return {"facet_by": None, "color_by": None, "facet_label": None, "color_label": None, "n_facets": 1}

    def _plot_metric_elbow(
        self, metric: str, height: int = 600, width: int = 800, title: str | None = None
    ) -> go.Figure:
        """
        Core shared implementation for elbow curve plotting.

        Args:
            metric: Metric to plot (will be mapped to DataFrame column)
            height, width: Plot dimensions
            title: Optional custom title

        Returns
        -------
            Plotly figure with faceted elbow curve
        """
        plot_data = self.get_plot_data()
        elbow_df = plot_data.get("elbow")

        if elbow_df is None or elbow_df.empty:
            print("[WARNING] No elbow plot data available")
            return go.Figure()

        # Map metric name to DataFrame column
        metric_col = self._map_metric_name(metric)

        if metric_col not in elbow_df.columns:
            raise ValueError(
                f"Metric '{metric}' (mapped to '{metric_col}') not found in columns: {elbow_df.columns.tolist()}"
            )

        # Detect faceting strategy
        strategy = self._detect_facet_strategy(elbow_df)
        n_rows = strategy["n_facets"]

        # Create subplots
        if n_rows > 1:
            fig = make_subplots(
                rows=n_rows,
                cols=1,
                subplot_titles=[
                    f"{strategy['facet_label']} = {val}" for val in sorted(elbow_df[strategy["facet_by"]].unique())
                ],
                vertical_spacing=0.12,
                row_heights=[1 / n_rows] * n_rows,
            )
        else:
            fig = go.Figure()

        colors = px.colors.qualitative.Set1

        # Plot by facets
        if strategy["facet_by"]:
            facet_values = sorted(elbow_df[strategy["facet_by"]].unique())
            for row_idx, facet_val in enumerate(facet_values, start=1):
                facet_subset = elbow_df[elbow_df[strategy["facet_by"]] == facet_val]

                if strategy["color_by"]:
                    # Color by groups within facet
                    for color_idx, color_val in enumerate(sorted(facet_subset[strategy["color_by"]].unique())):
                        subset = facet_subset[facet_subset[strategy["color_by"]] == color_val].sort_values(
                            "n_archetypes"
                        )

                        trace = go.Scatter(
                            x=subset["n_archetypes"],
                            y=subset[metric_col],
                            mode="lines+markers",
                            name=f"{strategy['color_label']}={color_val}",
                            line=dict(color=colors[color_idx % len(colors)], width=2),
                            marker=dict(size=8),
                            showlegend=(row_idx == 1),
                            legendgroup=f"group_{color_val}",
                        )

                        if n_rows > 1:
                            fig.add_trace(trace, row=row_idx, col=1)
                        else:
                            fig.add_trace(trace)
                else:
                    # Single line per facet
                    subset = facet_subset.sort_values("n_archetypes")
                    trace = go.Scatter(
                        x=subset["n_archetypes"],
                        y=subset[metric_col],
                        mode="lines+markers",
                        name=f"λ={facet_val}",
                        line=dict(color=colors[row_idx % len(colors)], width=2),
                        marker=dict(size=8),
                        showlegend=True,
                    )

                    if n_rows > 1:
                        fig.add_trace(trace, row=row_idx, col=1)
                    else:
                        fig.add_trace(trace)
        else:
            # No facets, just color by groups
            if strategy["color_by"]:
                for color_idx, color_val in enumerate(sorted(elbow_df[strategy["color_by"]].unique())):
                    subset = elbow_df[elbow_df[strategy["color_by"]] == color_val].sort_values("n_archetypes")

                    trace = go.Scatter(
                        x=subset["n_archetypes"],
                        y=subset[metric_col],
                        mode="lines+markers",
                        name=f"{strategy['color_label']}={color_val}",
                        line=dict(color=colors[color_idx % len(colors)], width=2),
                        marker=dict(size=8),
                    )
                    fig.add_trace(trace)
            else:
                # Single line only
                subset = elbow_df.sort_values("n_archetypes")
                trace = go.Scatter(
                    x=subset["n_archetypes"],
                    y=subset[metric_col],
                    mode="lines+markers",
                    line=dict(width=2),
                    marker=dict(size=8),
                )
                fig.add_trace(trace)

        # Format title
        if title is None:
            title = f"{metric.replace('_', ' ').title()} vs Number of Archetypes"

        # Update layout
        fig.update_layout(title=title, height=height, width=width, showlegend=True, hovermode="closest")

        # Update axes
        if n_rows > 1:
            for row in range(1, n_rows + 1):
                fig.update_xaxes(title_text="Number of Archetypes", row=row, col=1)
                fig.update_yaxes(title_text=metric.replace("_", " ").title(), row=row, col=1)
        else:
            fig.update_xaxes(title_text="Number of Archetypes")
            fig.update_yaxes(title_text=metric.replace("_", " ").title())

        return fig

    def plot_elbow_r2(self, height: int = 600, width: int = 800) -> go.Figure:
        """Generate elbow curve for Archetype R² vs number of archetypes.

        Primary visualization for hyperparameter selection. Automatically
        handles multi-dimensional search spaces with smart faceting:

        - Rows: Inflation factors (if multiple tested)
        - Colors: Hidden dimensions architectures

        Parameters
        ----------
        height : int, default: 600
            Plot height in pixels.
        width : int, default: 800
            Plot width in pixels.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive elbow curve plot.

        Examples
        --------
        >>> fig = cv_summary.plot_elbow_r2()
        >>> fig.show()
        >>> # Save to file
        >>> fig.write_html("elbow_curve.html")
        >>> fig.write_image("elbow_curve.png")

        See Also
        --------
        plot_metric : Generic version for any metric
        """
        return self._plot_metric_elbow(
            metric="archetype_r2", height=height, width=width, title="Archetype R² vs Number of Archetypes"
        )

    def plot_metric(self, metric: str, height: int = 600, width: int = 800) -> go.Figure:
        """Generate elbow curve for any CV metric.

        Generic version of plot_elbow_r2() for exploring different metrics.

        Parameters
        ----------
        metric : str
            Metric to plot. Accepts user-friendly names (see rank_by_metric).
        height : int, default: 600
            Plot height in pixels.
        width : int, default: 800
            Plot width in pixels.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive elbow curve plot.

        Examples
        --------
        >>> # Plot RMSE
        >>> fig = cv_summary.plot_metric("rmse")
        >>> fig.show()
        >>> # Plot convergence epochs
        >>> fig = cv_summary.plot_metric("convergence_epoch")
        """
        return self._plot_metric_elbow(metric=metric, height=height, width=width)

    def summary_report(self) -> str:
        """Generate user-friendly text summary for decision support.

        Returns
        -------
        str
            Formatted report including:

            - Search space summary
            - Dataset information
            - Top 3 configurations with metrics
            - Performance overview
            - Training efficiency stats

        Examples
        --------
        >>> report = cv_summary.summary_report()
        >>> print(report)

        >>> # Or in Jupyter
        >>> from IPython.display import Markdown
        >>> display(Markdown(report.replace(" ", "📊")))
        """
        report = []
        report.append(" CROSS-VALIDATION SUMMARY")
        report.append("=" * 40)

        # Search space summary
        report.append("Search Space:")
        n_archetypes = sorted(self.summary_df["n_archetypes"].unique())
        hidden_dims_options = sorted(self.summary_df["hidden_dims"].unique())
        report.append(f"  • n_archetypes tested: {n_archetypes}")
        report.append(f"  • hidden_dims_options: {hidden_dims_options}")
        report.append(f"  • Total configurations: {self.cv_info['n_configurations']}")
        report.append(f"  • CV folds: {self.cv_info['cv_folds']}")

        # Dataset info
        dataset_info = self.cv_info.get("dataset_info", {})
        report.append("\nDataset:")
        report.append(f"  • Samples: {dataset_info.get('n_total_samples', 'Unknown'):,}")
        report.append(f"  • Features: {dataset_info.get('n_features', 'Unknown')}")

        # Top 3 configurations
        report.append("\n TOP 3 CONFIGURATIONS:")
        for i, config in enumerate(self.ranked_configs[:3], 1):
            report.append(f"   {i}. {config['config_summary']}")
            report.append(f"      Archetype R²: {config['metric_value']:.4f}")
            report.append(f"      Std Error: ±{config['std_error']:.4f}")

        # Performance overview
        if "mean_archetype_r2" in self.summary_df.columns:
            r2 = self.summary_df["mean_archetype_r2"]
            report.append("\n[STATS] Performance Overview:")
            report.append(f"  • Best archetype R²: {r2.max():.4f}")
            report.append(f"  • Average across configs: {r2.mean():.4f} ± {r2.std():.4f}")
            report.append(f"  • Performance range: [{r2.min():.4f}, {r2.max():.4f}]")

        # Training efficiency
        total_time = self.cv_info.get("total_training_time", 0)
        if total_time > 0:
            report.append("\n⏱  Training Efficiency:")
            report.append(f"  • Total time: {total_time:.1f}s ({total_time / 60:.1f}min)")
            report.append(f"  • Average per config: {total_time / self.cv_info['n_configurations']:.1f}s")

        return "\n".join(report)

    def plot_elbow_curve(self, metrics: list[str] = None, height: int = 500, width: int = 800) -> go.Figure:
        """Generate elbow plots for multiple metrics.

        Creates side-by-side elbow curves for comparing different metrics
        across hyperparameter configurations.

        Parameters
        ----------
        metrics : list[str] | None
            Metrics to plot. Defaults to ['archetype_r2', 'rmse'].
            Accepts user-friendly names like 'r2', 'rmse', 'mae'.
        height, width : int
            Plot dimensions.

        Returns
        -------
        plotly.graph_objects.Figure
            Multi-panel elbow curve plot with all hyperparameter combinations.

        Examples
        --------
        >>> fig = cv_summary.plot_elbow_curve(metrics=['archetype_r2', 'rmse'])
        >>> fig.show()
        """
        if metrics is None:
            metrics = ["archetype_r2", "rmse"]

        plot_data = self.get_plot_data()
        elbow_df = plot_data.get("elbow")

        if elbow_df is None or elbow_df.empty:
            print("[WARNING] No elbow plot data available")
            return go.Figure()

        # Map metric names to actual column names
        metric_cols = []
        valid_metrics = []
        for metric in metrics:
            mapped_col = self._map_metric_name(metric)
            if mapped_col in elbow_df.columns:
                metric_cols.append(mapped_col)
                valid_metrics.append(metric)
            else:
                print(f"[WARNING] Metric '{metric}' (mapped to '{mapped_col}') not found, skipping")

        if not valid_metrics:
            print("[WARNING] No valid metrics found to plot")
            return go.Figure()

        fig = make_subplots(
            rows=1,
            cols=len(valid_metrics),
            subplot_titles=[f"{m.replace('_', ' ').title()}" for m in valid_metrics],
            horizontal_spacing=0.15,
        )

        colors = px.colors.qualitative.Set1

        # Determine grouping: create composite key from inflation_factor + hidden_dims
        has_inflation = "inflation_factor" in elbow_df.columns
        has_hidden = "hidden_dims" in elbow_df.columns

        # Create composite group key for proper separation
        if has_inflation and has_hidden:
            elbow_df = elbow_df.copy()
            elbow_df["_group_key"] = elbow_df.apply(
                lambda r: f"λ={r['inflation_factor']}, arch={r['hidden_dims']}", axis=1
            )
            group_var = "_group_key"
        elif has_inflation:
            group_var = "inflation_factor"
        elif has_hidden:
            group_var = "hidden_dims"
        else:
            group_var = None

        for i, (metric, mean_col) in enumerate(zip(valid_metrics, metric_cols)):
            col = i + 1

            if group_var:
                # Plot each configuration as separate line
                for j, group_val in enumerate(sorted(elbow_df[group_var].unique())):
                    subset = elbow_df[elbow_df[group_var] == group_val].sort_values("n_archetypes")

                    # Format legend label
                    if group_var == "_group_key":
                        label = str(group_val)
                    elif group_var == "inflation_factor":
                        label = f"λ={group_val}"
                    else:
                        label = f"hidden={group_val}"

                    trace = go.Scatter(
                        x=subset["n_archetypes"],
                        y=subset[mean_col],
                        mode="lines+markers",
                        name=label,
                        line=dict(color=colors[j % len(colors)], width=2),
                        marker=dict(size=8),
                        showlegend=(i == 0),
                        legendgroup=f"group_{j}",
                    )
                    fig.add_trace(trace, row=1, col=col)
            else:
                # Single line (no grouping)
                subset = elbow_df.sort_values("n_archetypes")
                trace = go.Scatter(
                    x=subset["n_archetypes"],
                    y=subset[mean_col],
                    mode="lines+markers",
                    line=dict(width=2),
                    marker=dict(size=8),
                )
                fig.add_trace(trace, row=1, col=col)

        fig.update_layout(
            title="Elbow Curves: Performance vs n_archetypes",
            height=height,
            width=width,
            showlegend=True,
        )

        # Update x-axis labels
        for i in range(1, len(valid_metrics) + 1):
            fig.update_xaxes(title_text="Number of Archetypes", row=1, col=i)

        return fig

    def save(self, path: str | Path) -> None:
        """Save CVSummary to pickle file.

        Parameters
        ----------
        path : str | Path
            Save location. Parent directories created if needed.

        Examples
        --------
        >>> cv_summary.save("results/cv_summary.pkl")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)

        print(f" CV results saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "CVSummary":
        """Load CVSummary from pickle file.

        Parameters
        ----------
        path : str | Path
            File to load.

        Returns
        -------
        CVSummary
            Loaded results.

        Examples
        --------
        >>> cv_summary = CVSummary.load("results/cv_summary.pkl")
        >>> print(cv_summary.summary_report())
        """
        with open(path, "rb") as f:
            results = pickle.load(f)

        print(f" CV results loaded from {path}")
        return results


# Keep GridSearchResults for backward compatibility (deprecated)
@dataclass
class GridSearchResults(CVSummary):
    """DEPRECATED: Use CVSummary instead. Kept for backward compatibility."""

    def __post_init__(self):
        print("[WARNING] GridSearchResults is deprecated. Use CVSummary instead.")

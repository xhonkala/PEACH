# Training and model performance visualization


import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from ..utils.analysis import get_archetypal_coordinates

"""
Training and Model Performance Visualization
Comprehensive visualization suite for archetypal analysis training metrics, convergence analysis, and hyperparameter optimization results.

=== MODULE API INVENTORY ===

MAIN FUNCTIONS:
 plot_training_metrics(history: Dict[str, List[float]], height: int = 400, width: int = 800, display: bool = True) -> Optional[go.Figure]
    Purpose: Interactive plotly visualization of training metrics organized by category (loss, stability, convergence)
    Inputs: history(Dict with metric lists), height/width(int, plot dimensions), display(bool, show plot)
    Outputs: go.Figure or None, organized subplots with loss/stability/convergence sections
    Side Effects: Displays interactive plot if display=True
    Features: Rolling mean for convergence, color-coded metrics, spline smoothing

 plot_convergence_analysis(history: Dict[str, List[float]], window_size: int = 5) -> None
    Purpose: Detailed convergence analysis with loss trends and delta distribution
    Inputs: history(Dict with 'loss' and 'loss_delta' keys), window_size(int, rolling window)
    Outputs: None (displays plot directly)
    Side Effects: Shows dual-panel plotly figure with scatter and box plots
    Features: Loss trajectory and delta distribution analysis

 print_space_stats(pca: np.ndarray, latent: np.ndarray, recons: np.ndarray) -> None
    Purpose: Print statistical summary of different data spaces for debugging
    Inputs: pca/latent/recons(np.ndarray, data in different spaces)
    Outputs: None (prints to console)
    Side Effects: Console output with shape, mean, std, range statistics
    Use Case: Debugging space transformations and scale verification

 plot_basic_metrics(history: Dict[str, List[float]], display: bool = True) -> Optional[plt.Figure]
    Purpose: Simple matplotlib fallback for training metrics visualization
    Inputs: history(Dict with metric lists), display(bool, show plot)
    Outputs: plt.Figure or None, dual-panel matplotlib figure
    Side Effects: Displays matplotlib plot if display=True
    Use Case: Debugging when plotly fails or for static plots

 plot_model_performance(training_results: Dict) -> plt.Figure
    Purpose: Comprehensive 4-panel performance analysis for Deep_AA model training
    Inputs: training_results(Dict with 'history' key containing metric lists)
    Outputs: plt.Figure with 2x2 subplot layout
    Panels: Loss curves, performance metrics, constraint violations, stability analysis
    Features: Handles missing metrics gracefully, dual y-axes for different scales

 plot_hull_metrics(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str = 'cpu') -> plt.Figure
    Purpose: Archetypal hull quality assessment with constraint validation and manifold distances
    Inputs: model(trained Deep_AA), dataloader(torch.DataLoader), device(str, 'cpu'|'cuda')
    Outputs: plt.Figure bar chart with constraint and quality metrics
    Features: Real-time constraint checking, archetype spacing, manifold proximity analysis
    Metrics: Constraint satisfaction, weight distributions, archetype distances, manifold distances

 save_metrics_plot(fig, path: str) -> None
    Purpose: Save plotly figure to HTML file
    Inputs: fig(go.Figure), path(str, output file path)
    Outputs: None
    Side Effects: Writes HTML file to disk
    Use Case: Export interactive plots for reports/sharing

HYPERPARAMETER OPTIMIZATION VISUALIZATION:
 plot_hyperparameter_elbow(cv_summary, metrics: List[str] = None, height: int = 500, width: int = 1200) -> go.Figure
    Purpose: Multi-metric elbow curves for optimal hyperparameter selection with cross-validation error bars
    Inputs: cv_summary(CVSummary object), metrics(List[str], default ['archetype_r2', 'rmse']), dimensions
    Outputs: go.Figure with subplots for each metric showing elbow curves by latent_offset
    Features: Error bars from CV std, grouping by latent_offset, interactive hover
    Use Case: Identifying optimal n_archetypes and hyperparameter combinations

 plot_cv_fold_consistency(cv_summary, metric: str = 'archetype_r2', height: int = 600, width: int = 1000) -> go.Figure
    Purpose: Cross-validation fold consistency analysis via box plots
    Inputs: cv_summary(CVSummary object), metric(str, specific metric to analyze), dimensions
    Outputs: go.Figure box plot showing metric distribution across CV folds
    Features: Individual fold points, outlier detection, configuration comparison
    Use Case: Assessing hyperparameter stability and overfitting detection

EXTERNAL DEPENDENCIES:
 plotly.graph_objects + plotly.subplots: Interactive web-based visualizations
 matplotlib.pyplot: Static publication-quality plots
 numpy: Numerical array operations and statistics
 torch: GPU tensor handling and model evaluation
 ..utils.analysis.get_archetypal_coordinates: Model coordinate extraction
"""


def plot_training_metrics(
    history: dict[str, list[float]], height: int = 400, width: int = 800, display: bool = True
) -> go.Figure | None:
    """
    Plot training metrics over epochs using Plotly.

    Creates organized 3-row subplot with loss, stability, and
    convergence metrics.

    Parameters
    ----------
    history : dict[str, list[float]]
        Dictionary of metric lists from training.
    height : int, default: 400
        Base plot height (actual height is 2x this for 3 rows).
    width : int, default: 800
        Plot width.
    display : bool, default: True
        Whether to display plot immediately via fig.show().

    Returns
    -------
    plotly.graph_objects.Figure or None
        The plotly Figure object. Returns None only if history is empty.

    Notes
    -----
    Subplot layout:
    - Row 1 (40%): Loss metrics (loss, archetypal_loss, KLD, rmse)
    - Row 2 (30%): Stability metrics (vertex_stability_latent/pca, hull_stability)
    - Row 3 (30%): Convergence (loss_delta with rolling mean)

    Metrics are color-coded and smoothed with spline interpolation.
    """
    # Check if history is empty
    if not history or not any(history.values()):
        print("Warning: No metrics to plot - training history is empty")
        return None

    # Organize metrics into categories
    loss_metrics = ["loss", "archetypal_loss", "KLD", "rmse"]
    stability_metrics = ["vertex_stability_latent", "vertex_stability_pca", "hull_stability"]
    performance_metrics = ["archetype_r2"]
    convergence_metrics = ["loss_delta"]

    # Create subplots with better organization
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Loss Metrics", "Stability Metrics", "Convergence Analysis"),
        vertical_spacing=0.1,
        row_heights=[0.4, 0.3, 0.3],
    )

    # Color scheme
    colors = {
        "loss": "#1f77b4",  # blue
        "archetypal_loss": "#ff7f0e",  # orange
        "KLD": "#2ca02c",  # green
        "rmse": "#d62728",  # red
        "vertex_stability_latent": "#9467bd",  # purple
        "vertex_stability_pca": "#8c564b",  # brown
        "hull_stability": "#e377c2",  # pink
        "archetype_r2": "#7f7f7f",  # gray
        "loss_delta": "#bcbd22",  # yellow-green
    }

    # Plot loss metrics (top subplot)
    for metric_name in loss_metrics:
        if metric_name in history and len(history[metric_name]) > 1:
            values = history[metric_name]
            epochs = np.arange(len(values))

            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=values,
                    name=metric_name.replace("_", " ").title(),
                    mode="lines",
                    line=dict(width=2, color=colors.get(metric_name, "#1f77b4"), shape="spline", smoothing=0.3),
                    hovertemplate="Epoch %{x}<br>" + f"{metric_name}: " + "%{y:.4f}<br>",
                ),
                row=1,
                col=1,
            )

    # Plot stability metrics (middle subplot)
    for metric_name in stability_metrics:
        if metric_name in history and len(history[metric_name]) > 1:
            values = history[metric_name]
            epochs = np.arange(len(values))

            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=values,
                    name=metric_name.replace("_", " ").title(),
                    mode="lines",
                    line=dict(width=2, color=colors.get(metric_name, "#1f77b4"), shape="spline", smoothing=0.3),
                    hovertemplate="Epoch %{x}<br>" + f"{metric_name}: " + "%{y:.4f}<br>",
                ),
                row=2,
                col=1,
            )

    # Plot convergence metrics (bottom subplot)
    if "loss_delta" in history and len(history["loss_delta"]) > 5:
        epochs = np.arange(len(history["loss_delta"]))

        # Plot raw loss delta
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["loss_delta"],
                name="Loss Δ",
                mode="lines",
                line=dict(
                    width=1,
                    color="rgba(188, 189, 34, 0.3)",  # Transparent yellow-green
                ),
                showlegend=False,
            ),
            row=3,
            col=1,
        )

        # Calculate and plot rolling mean
        window = min(5, len(epochs))
        rolling_mean = np.convolve(history["loss_delta"], np.ones(window) / window, mode="valid")

        fig.add_trace(
            go.Scatter(
                x=epochs[window - 1 :],
                y=rolling_mean,
                name="Loss Δ (Rolling Mean)",
                mode="lines",
                line=dict(
                    width=2,
                    color="#bcbd22",  # Solid yellow-green
                    shape="spline",
                    smoothing=0.3,
                ),
                hovertemplate="Epoch %{x}<br>Mean Δ: %{y:.4f}<br>",
            ),
            row=3,
            col=1,
        )

    # Update layout for better readability
    fig.update_layout(
        height=height * 2,
        width=width,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        template="plotly_white",  # Clean white template
        font=dict(size=12),
    )

    # Update axes labels and styling
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)", row=1, col=1)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)", row=3, col=1)

    fig.update_yaxes(
        title_text="Loss Value", showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)", row=1, col=1
    )
    fig.update_yaxes(
        title_text="Stability", showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)", row=2, col=1
    )
    fig.update_yaxes(title_text="Loss Δ", showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)", row=3, col=1)

    if display:
        fig.show()

    return fig


def plot_convergence_analysis(history: dict[str, list[float]], window_size: int = 5):
    """
    Detailed convergence analysis plot
    """
    if not history or "loss_delta" not in history:
        print("Warning: No convergence metrics to plot")
        return

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Loss Convergence", "Loss Delta Distribution"),
        specs=[[{"type": "scatter"}, {"type": "box"}]],
    )

    # Loss convergence plot
    epochs = np.arange(len(history["loss"]))
    fig.add_trace(
        go.Scatter(x=epochs, y=history["loss"], name="Loss", mode="lines+markers", marker=dict(size=4)), row=1, col=1
    )

    # Loss delta box plot
    fig.add_trace(
        go.Box(y=history["loss_delta"], name="Loss Δ Distribution", boxpoints="all", jitter=0.3, pointpos=-1.8),
        row=1,
        col=2,
    )

    fig.update_layout(height=400, width=800, showlegend=True, title_text="Convergence Analysis")

    fig.show()


def save_metrics_plot(fig, path: str):
    """Save metrics plot to file"""
    if fig is not None:
        fig.write_html(path)


def print_space_stats(pca: np.ndarray, latent: np.ndarray, recons: np.ndarray) -> None:
    """Print basic statistics about each space"""
    spaces = {"PCA": pca, "Latent": latent, "Reconstructed": recons}

    print("Space Statistics:")
    print("-" * 50)
    for name, data in spaces.items():
        print(f"\n{name} Space:")
        print(f"Shape: {data.shape}")
        print(f"Mean: {data.mean():.3f}")
        print(f"Std: {data.std():.3f}")
        print(f"Range: [{data.min():.3f}, {data.max():.3f}]")


def plot_basic_metrics(history: dict[str, list[float]], display: bool = True):
    """Simple matplotlib plot for debugging training metrics"""
    if not history or not any(history.values()):
        print("Warning: No metrics to plot")
        return

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot main metrics
    for metric_name, values in history.items():
        if metric_name != "loss_delta" and values:  # Plot everything except loss_delta
            epochs = range(len(values))
            ax1.plot(epochs, values, label=metric_name)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Value")
    ax1.set_title("Training Metrics")
    ax1.legend()

    # Plot convergence
    if "loss_delta" in history and history["loss_delta"]:
        epochs = range(len(history["loss_delta"]))
        ax2.plot(epochs, history["loss_delta"], "r-", label="Loss Δ")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss Δ")
        ax2.set_title("Convergence")
        ax2.legend()

    plt.tight_layout()

    if display:
        plt.show()

    return fig


def plot_model_performance(training_results: dict) -> plt.Figure:
    """
    Plot training performance metrics for Deep_AA model.

    Creates 4-panel visualization covering loss, performance,
    constraints, and stability.

    Parameters
    ----------
    training_results : dict
        Dictionary from train_vae() containing 'history' key.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with 2x2 subplot layout:
        - [0,0] Training Loss: total loss, archetypal_loss, kld_loss
        - [0,1] Performance Metrics: archetype_r2 (left axis), rmse (right axis)
        - [1,0] Constraint Violations: A/B_sum_error, A/B_negative_fraction
        - [1,1] Stability & Manifold: archetype_drift/stability metrics,
                mean/max_archetype_data_distance

    Notes
    -----
    Performance panel uses dual y-axes since R² and RMSE have different scales.
    Stability panel shows placeholder text if no stability metrics found.
    """
    history = training_results["history"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot loss curves
    ax1 = axes[0, 0]
    if "loss" in history:
        ax1.plot(history["loss"], label="Total Loss", linewidth=2)
    if "archetypal_loss" in history:
        ax1.plot(history["archetypal_loss"], label="Archetypal Loss", linewidth=2)
    if "kld_loss" in history:
        ax1.plot(history["kld_loss"], label="KLD Loss", linewidth=2)
    # Note: reconstruction_loss now maps to archetypal_loss
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot performance metrics
    ax2 = axes[0, 1]
    if "archetype_r2" in history:
        ax2.plot(history["archetype_r2"], label="Archetype R²", linewidth=2, color="green")
    if "rmse" in history:
        ax2_twin = ax2.twinx()
        ax2_twin.plot(history["rmse"], label="RMSE", linewidth=2, color="orange")
        ax2_twin.set_ylabel("RMSE", color="orange")
        ax2_twin.tick_params(axis="y", labelcolor="orange")
    ax2.set_title("Performance Metrics")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Archetype R²", color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.grid(True, alpha=0.3)

    # Plot constraint violations
    ax3 = axes[1, 0]
    constraint_metrics = ["A_sum_error", "B_sum_error", "A_negative_fraction", "B_negative_fraction"]
    for metric in constraint_metrics:
        if metric in history:
            ax3.plot(history[metric], label=metric, linewidth=2)
    ax3.set_title("Constraint Violations")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Error")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot stability metrics
    ax4 = axes[1, 1]

    # Try all possible stability metric names
    all_stability_metrics = [
        "archetype_drift_mean",
        "archetype_stability_mean",
        "archetype_drift_max",
        "archetype_drift_std",
        "archetype_stability_min",
        "archetype_variance_mean",
        "mean_archetype_data_distance",
        "max_archetype_data_distance",  # New manifold metrics
    ]

    stability_found = False
    for metric in all_stability_metrics:
        if metric in history and len(history[metric]) > 0:
            ax4.plot(history[metric], label=metric, linewidth=2)
            stability_found = True

    if stability_found:
        ax4.set_title("Archetype Stability & Manifold")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Stability Score")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        # No stability metrics found - show placeholder
        ax4.text(
            0.5,
            0.5,
            "No stability metrics found\n\nAvailable metrics:\n"
            + "\n".join([k for k in history.keys() if "drift" in k or "stability" in k or "variance" in k]),
            ha="center",
            va="center",
            transform=ax4.transAxes,
            fontsize=10,
        )
        ax4.set_title("Archetype Stability (Not Available)")
        ax4.set_xticks([])
        ax4.set_yticks([])

    plt.tight_layout()
    return fig


def plot_hull_metrics(
    model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str = "cpu"
) -> plt.Figure:
    """
    Plot archetypal hull quality metrics.

    Evaluates constraint satisfaction and manifold positioning
    quality for a trained Deep_AA model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained Deep_AA model.
    dataloader : torch.utils.data.DataLoader
        DataLoader containing dataset.
    device : str, default: 'cpu'
        Device for computations ('cpu', 'cuda', 'mps').

    Returns
    -------
    matplotlib.figure.Figure
        Bar chart with 9 metrics:

        Constraint metrics (red bars):
        - Constraint Satisfaction: 1.0 if all satisfied, 0.0 otherwise
        - Mean A Sum Error: deviation of A row sums from 1.0
        - Mean B Sum Error: deviation of B column sums from 1.0
        - A Negative Fraction: fraction of negative values in A
        - B Negative Fraction: fraction of negative values in B

        Quality metrics (blue bars):
        - Mean Archetype Distance: mean pairwise distance between archetypes
        - Weight Sparsity (A): fraction of A weights > 0.1
        - Mean Manifold Distance: mean distance from archetypes to nearest data
        - Max Manifold Distance: max distance from any archetype to nearest data

    Notes
    -----
    Lower manifold distances indicate archetypes are well-positioned
    within the data cloud rather than in empty space. B matrix metrics
    will be approximately uniform for Deep_AA (dummy B matrix).
    """
    model.eval()

    # Get a representative batch
    data_batch = next(iter(dataloader))[0].to(device)

    with torch.no_grad():
        # Get coordinates and validate constraints
        coords = get_archetypal_coordinates(model, data_batch)
        constraints = model.validate_constraints(coords["A"], coords["B"])

        # Calculate basic hull metrics
        A, B, Y = coords["A"], coords["B"], coords["Y"]

        # Archetype spread (distances between archetypes)
        archetype_distances = []
        for i in range(Y.shape[0]):  # Y is [n_archetypes, input_dim]
            for j in range(i + 1, Y.shape[0]):
                dist = torch.norm(Y[i] - Y[j]).item()
                archetype_distances.append(dist)

        # Manifold distance metrics
        manifold_distances = []
        for archetype in Y:
            distances = torch.norm(data_batch - archetype.unsqueeze(0), dim=1)
            min_distance = torch.min(distances).item()
            manifold_distances.append(min_distance)

        metrics = {
            "Constraint Satisfaction": float(constraints["constraints_satisfied"]),
            "Mean A Sum Error": constraints["A_sum_error"],
            "Mean B Sum Error": constraints["B_sum_error"],  # Should be ~0 for dummy B
            "A Negative Fraction": constraints["A_negative_fraction"],
            "B Negative Fraction": constraints["B_negative_fraction"],
            "Mean Archetype Distance": np.mean(archetype_distances),
            "Weight Sparsity (A)": (A > 0.1).float().mean().item(),
            "Mean Manifold Distance": np.mean(manifold_distances),  # How close archetypes are to data
            "Max Manifold Distance": np.max(manifold_distances),  # Worst archetype position
        }

    fig, ax = plt.subplots(figsize=(12, 6))

    # Split metrics into constraint and quality metrics
    constraint_metrics = [
        "Constraint Satisfaction",
        "Mean A Sum Error",
        "Mean B Sum Error",
        "A Negative Fraction",
        "B Negative Fraction",
    ]
    quality_metrics = [
        "Mean Archetype Distance",
        "Weight Sparsity (A)",
        "Mean Manifold Distance",
        "Max Manifold Distance",
    ]

    x_pos = np.arange(len(metrics))
    colors = ["red" if k in constraint_metrics else "blue" for k in metrics.keys()]

    bars = ax.bar(x_pos, list(metrics.values()), color=colors, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(metrics.keys()), rotation=45, ha="right")
    ax.set_title("Archetypal Hull Quality Metrics")
    ax.set_ylabel("Metric Value")

    # Add value labels on bars
    for bar, value in zip(bars, metrics.values(), strict=False):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{value:.4f}", ha="center", va="bottom")

    plt.tight_layout()
    return fig


def plot_cv_training_histories(
    cv_summary, metric: str = "rmse", config_key: str = None, height: int = 400, width: int = 800
) -> go.Figure:
    """
    Plot training history trends from CV results.

    Args:
        cv_summary: CVSummary object with fold_histories
        metric: Metric to plot ('rmse', 'loss', 'archetype_r2')
        config_key: Specific configuration to plot (if None, plots best config)
        height, width: Plot dimensions

    Returns
    -------
        Plotly figure with training trends
    """
    if not cv_summary or not cv_summary.config_results:
        print("[WARNING] No CV results available")
        return go.Figure()

    # Get specific configuration or best one
    if config_key is None:
        # Get best configuration by archetype_r2
        best_config = cv_summary.ranked_configs[0] if cv_summary.ranked_configs else None
        if not best_config:
            print("[WARNING] No configurations found")
            return go.Figure()
        config_key = f"n_arch={best_config['hyperparameters']['n_archetypes']}_hidden={best_config['hyperparameters']['hidden_dims']}"

    if config_key not in cv_summary.config_results:
        print(f"[WARNING] Configuration '{config_key}' not found")
        return go.Figure()

    cv_result = cv_summary.config_results[config_key]

    # Check if fold histories exist
    if not hasattr(cv_result, "fold_histories") or not cv_result.fold_histories:
        print(f"[WARNING] No training histories stored for {config_key}. Only final metrics available.")
        print(f"   Available metrics: {list(cv_result.mean_metrics.keys())}")
        return go.Figure()

    # Get aggregated history
    aggregated = cv_result.get_aggregated_history(metric)

    if not aggregated:
        print(f"[WARNING] Metric '{metric}' not found in training histories")
        return go.Figure()

    mean_key = f"{metric}_mean"
    std_key = f"{metric}_std"

    if mean_key not in aggregated:
        print(f"[WARNING] No mean history found for {metric}")
        return go.Figure()

    epochs = list(range(len(aggregated[mean_key])))
    mean_values = aggregated[mean_key]
    std_values = aggregated.get(std_key, [0] * len(mean_values))

    # Create plot
    fig = go.Figure()

    # Add mean line with std band
    fig.add_trace(
        go.Scatter(x=epochs, y=mean_values, mode="lines", name=f"Mean {metric}", line=dict(color="blue", width=2))
    )

    # Add confidence band
    upper_bound = [m + s for m, s in zip(mean_values, std_values, strict=False)]
    lower_bound = [m - s for m, s in zip(mean_values, std_values, strict=False)]

    fig.add_trace(
        go.Scatter(
            x=epochs + epochs[::-1],
            y=upper_bound + lower_bound[::-1],
            fill="toself",
            fillcolor="rgba(0,100,250,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
            name="Confidence",
        )
    )

    fig.update_layout(
        title=f"{metric.upper()} Training History - {config_key}",
        xaxis_title="Epoch",
        yaxis_title=metric.upper(),
        height=height,
        width=width,
        showlegend=True,
    )

    return fig


def plot_hyperparameter_elbow(cv_summary, metrics: list[str] = None, height: int = 500, width: int = 1200) -> go.Figure:
    """
    Plot elbow curves for hyperparameter selection.

    Creates multi-panel visualization showing how each metric varies
    with number of archetypes, useful for identifying optimal k.

    Parameters
    ----------
    cv_summary : CVSummary
        CVSummary object from hyperparameter search.
    metrics : list[str] | None, default: None
        Metrics to plot. If None, defaults to ['archetype_r2', 'rmse'].
    height : int, default: 500
        Plot height in pixels.
    width : int, default: 1200
        Plot width in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        Figure with one subplot per metric, each showing:
        - X-axis: Number of archetypes
        - Y-axis: Metric value (mean across CV folds)
        - Error bars: Standard deviation across CV folds
        - Multiple lines if latent_offset varies (grouped by offset)

    Notes
    -----
    Look for "elbow" points where adding more archetypes yields
    diminishing returns. For R² metrics, look for where the curve
    plateaus. For RMSE, look for where decrease slows.

    Examples
    --------
    >>> cv_summary = pc.tl.hyperparameter_search(adata, n_archetypes_range=[3, 4, 5, 6, 7])
    >>> fig = plot_hyperparameter_elbow(cv_summary)
    >>> fig.show()
    """
    if metrics is None:
        metrics = ["archetype_r2", "rmse"]

    # Get plot data from CVSummary
    plot_data = cv_summary.get_plot_data()
    summary_df = plot_data.get("summary")

    if summary_df is None or summary_df.empty:
        print("[WARNING] No summary data available for elbow plots")
        return go.Figure()

    # Create subplots for each metric
    fig = make_subplots(
        rows=1,
        cols=len(metrics),
        subplot_titles=[f"{metric.replace('_', ' ').title()}" for metric in metrics],
        horizontal_spacing=0.15,
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, metric in enumerate(metrics):
        col = i + 1
        mean_col = f"mean_{metric}"
        std_col = f"std_{metric}"

        if mean_col not in summary_df.columns:
            print(f"[WARNING] Metric {metric} not found in results")
            continue

        # Group by latent_offset if available
        if "latent_offset" in summary_df.columns:
            # Plot by latent offset
            for j, offset in enumerate(sorted(summary_df["latent_offset"].unique())):
                subset = summary_df[summary_df["latent_offset"] == offset].sort_values("n_archetypes")

                # Add error bars if std data is available
                error_y = None
                if std_col in summary_df.columns:
                    error_y = dict(type="data", array=subset[std_col], visible=True)

                trace = go.Scatter(
                    x=subset["n_archetypes"],
                    y=subset[mean_col],
                    error_y=error_y,
                    mode="lines+markers",
                    name=f"latent_offset +{offset}",
                    line=dict(color=colors[j % len(colors)]),
                    marker=dict(size=8),
                    showlegend=(i == 0),  # Only show legend for first subplot
                )

                fig.add_trace(trace, row=1, col=col)
        else:
            # Simple plot without latent offset grouping
            subset = summary_df.sort_values("n_archetypes")

            error_y = None
            if std_col in summary_df.columns:
                error_y = dict(type="data", array=subset[std_col], visible=True)

            trace = go.Scatter(
                x=subset["n_archetypes"],
                y=subset[mean_col],
                error_y=error_y,
                mode="lines+markers",
                name=metric,
                marker=dict(size=8),
                showlegend=(i == 0),
            )

            fig.add_trace(trace, row=1, col=col)

    # Update layout
    fig.update_layout(title="Hyperparameter Elbow Curves", height=height, width=width, hovermode="x unified")

    # Update x-axis labels
    for i in range(1, len(metrics) + 1):
        fig.update_xaxes(title_text="Number of Archetypes", row=1, col=i)

    return fig


def plot_cv_fold_consistency(
    cv_summary, metric: str = "archetype_r2", height: int = 600, width: int = 1000
) -> go.Figure:
    """
    Plot cross-validation fold consistency for a specific metric.

    Args:
        cv_summary: CVSummary object from hyperparameter search
        metric: Metric to analyze for consistency
        height, width: Plot dimensions

    Returns
    -------
        Plotly box plot showing fold consistency
    """
    plot_data = cv_summary.get_plot_data()
    fold_data = plot_data.get("fold_consistency")

    if fold_data is None or fold_data.empty:
        print("[WARNING] No fold consistency data available")
        return go.Figure()

    # Filter for the specific metric
    metric_data = fold_data[fold_data["metric"] == metric]

    if metric_data.empty:
        print(f"[WARNING] No data found for metric: {metric}")
        return go.Figure()

    fig = go.Figure()

    # Create box plots for each configuration
    for config in sorted(metric_data["config"].unique()):
        config_data = metric_data[metric_data["config"] == config]

        fig.add_trace(
            go.Box(y=config_data["value"], name=config.replace("_", " "), boxpoints="all", jitter=0.3, pointpos=-1.8)
        )

    fig.update_layout(
        title=f"Cross-Validation Fold Consistency: {metric.replace('_', ' ').title()}",
        xaxis_title="Configuration",
        yaxis_title=f"{metric.replace('_', ' ').title()}",
        height=height,
        width=width,
        showlegend=False,
    )

    return fig


def plot_archetype_positions(
    archetype_coordinates: np.ndarray,
    title: str = "Archetype Positions in PCA Space",
    figsize: tuple = (15, 6),
    cmap: str = "tab10",
    show_distances: bool = True,
    save_path: str = None,
) -> plt.Figure:
    """
    Visualize archetype positions in PCA space with distance matrix heatmap.

    Creates a two-panel visualization showing:
    1. Archetype positions in first 2 principal components
    2. Pairwise distance matrix heatmap

    Args:
        archetype_coordinates: Array of archetype coordinates [n_archetypes, n_dims]
        title: Main figure title
        figsize: Figure size as (width, height)
        cmap: Colormap for archetype points
        show_distances: Whether to show distance matrix panel
        save_path: Optional path to save figure

    Returns
    -------
        matplotlib.Figure: Figure with archetype visualizations

    Raises
    ------
        ValueError: If coordinates have less than 2 dimensions

    Examples
    --------
        >>> coords = adata.uns["archetype_coordinates"]
        >>> fig = plot_archetype_positions(coords)
        >>> plt.show()
    """
    # Handle torch tensors
    if hasattr(archetype_coordinates, "detach"):
        archetype_coordinates = archetype_coordinates.detach().cpu().numpy()

    # Validate dimensions
    if archetype_coordinates.shape[1] < 2:
        raise ValueError(f"Need at least 2 dimensions for visualization, got {archetype_coordinates.shape[1]}")

    n_archetypes = archetype_coordinates.shape[0]

    # Calculate pairwise distances
    from scipy.spatial.distance import pdist, squareform

    distance_matrix = squareform(pdist(archetype_coordinates, metric="euclidean"))

    # Create figure
    if show_distances:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0] // 2, figsize[1]))

    # Plot 1: Archetype positions in first 2 PCs
    scatter = ax1.scatter(
        archetype_coordinates[:, 0],
        archetype_coordinates[:, 1],
        s=200,
        c=range(n_archetypes),
        cmap=cmap,
        alpha=0.8,
        edgecolors="black",
        linewidth=2,
    )

    # Add archetype labels
    for i, (x, y) in enumerate(archetype_coordinates[:, :2]):
        ax1.annotate(f"A{i + 1}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=12, fontweight="bold")

    ax1.set_xlabel("PC 1", fontsize=12)
    ax1.set_ylabel("PC 2", fontsize=12)
    ax1.set_title("Archetype Positions", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Add hull connecting archetypes
    if n_archetypes > 2:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(archetype_coordinates[:, :2])
        for simplex in hull.simplices:
            ax1.plot(archetype_coordinates[simplex, 0], archetype_coordinates[simplex, 1], "k-", alpha=0.3, linewidth=1)

    # Plot 2: Distance matrix heatmap
    if show_distances:
        im = ax2.imshow(distance_matrix, cmap="viridis", alpha=0.8, aspect="auto")
        ax2.set_xlabel("Archetype", fontsize=12)
        ax2.set_ylabel("Archetype", fontsize=12)
        ax2.set_title("Pairwise Archetype Distances", fontsize=14)

        # Set ticks
        ax2.set_xticks(range(n_archetypes))
        ax2.set_yticks(range(n_archetypes))
        ax2.set_xticklabels([f"A{i + 1}" for i in range(n_archetypes)])
        ax2.set_yticklabels([f"A{i + 1}" for i in range(n_archetypes)])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label("Euclidean Distance", fontsize=11)

        # Add distance values to heatmap (if not too many archetypes)
        if n_archetypes <= 10:
            for i in range(n_archetypes):
                for j in range(n_archetypes):
                    text_color = "white" if distance_matrix[i, j] > distance_matrix.max() * 0.5 else "black"
                    ax2.text(
                        j,
                        i,
                        f"{distance_matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontweight="bold",
                        fontsize=9,
                    )

    # Overall title
    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Figure saved to: {save_path}")

    return fig


def plot_archetype_distances_3d(
    archetype_coordinates: np.ndarray,
    title: str = "Archetype Positions in 3D PCA Space",
    figsize: tuple = (12, 10),
    cmap: str = "tab10",
    save_path: str = None,
) -> plt.Figure:
    """
    Visualize archetype positions in 3D PCA space.

    Args:
        archetype_coordinates: Array of archetype coordinates [n_archetypes, n_dims]
        title: Figure title
        figsize: Figure size as (width, height)
        cmap: Colormap for archetype points
        save_path: Optional path to save figure

    Returns
    -------
        matplotlib.Figure: 3D visualization of archetypes

    Raises
    ------
        ValueError: If coordinates have less than 3 dimensions

    Examples
    --------
        >>> coords = adata.uns["archetype_coordinates"]
        >>> fig = plot_archetype_distances_3d(coords)
        >>> plt.show()
    """
    # Handle torch tensors
    if hasattr(archetype_coordinates, "detach"):
        archetype_coordinates = archetype_coordinates.detach().cpu().numpy()

    # Validate dimensions
    if archetype_coordinates.shape[1] < 3:
        raise ValueError(f"Need at least 3 dimensions for 3D visualization, got {archetype_coordinates.shape[1]}")

    n_archetypes = archetype_coordinates.shape[0]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Plot archetype positions
    scatter = ax.scatter(
        archetype_coordinates[:, 0],
        archetype_coordinates[:, 1],
        archetype_coordinates[:, 2],
        s=300,
        c=range(n_archetypes),
        cmap=cmap,
        alpha=0.8,
        edgecolors="black",
        linewidth=2,
        depthshade=True,
    )

    # Add archetype labels
    for i, (x, y, z) in enumerate(archetype_coordinates[:, :3]):
        ax.text(x, y, z, f"  A{i + 1}", fontsize=12, fontweight="bold")

    # Draw edges between all archetypes (convex hull edges in 3D)
    if n_archetypes > 3:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(archetype_coordinates[:, :3])

        # Draw hull edges
        for simplex in hull.simplices:
            # Each simplex in 3D is a triangle, draw its edges
            triangle = archetype_coordinates[simplex, :3]
            for i in range(3):
                ax.plot(
                    [triangle[i, 0], triangle[(i + 1) % 3, 0]],
                    [triangle[i, 1], triangle[(i + 1) % 3, 1]],
                    [triangle[i, 2], triangle[(i + 1) % 3, 2]],
                    "k-",
                    alpha=0.2,
                    linewidth=1,
                )

    ax.set_xlabel("PC 1", fontsize=12, labelpad=10)
    ax.set_ylabel("PC 2", fontsize=12, labelpad=10)
    ax.set_zlabel("PC 3", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, pad=20)

    # Improve viewing angle
    ax.view_init(elev=20, azim=45)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.1)
    cbar.set_label("Archetype Index", fontsize=11)
    cbar.set_ticks(range(n_archetypes))
    cbar.set_ticklabels([f"A{i + 1}" for i in range(n_archetypes)])

    plt.tight_layout()

    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Figure saved to: {save_path}")

    return fig


def compute_archetype_statistics(archetype_coordinates: np.ndarray, verbose: bool = True) -> dict:
    """
    Compute statistics about archetype positions.

    Parameters
    ----------
    archetype_coordinates : np.ndarray
        Array of archetype coordinates [n_archetypes, n_dims].
        Can also be a torch.Tensor (will be converted).
    verbose : bool, default: True
        Whether to print statistics to console.

    Returns
    -------
    dict
        Statistics dictionary with keys:
        - n_archetypes : int - Number of archetypes
        - n_dimensions : int - Number of embedding dimensions
        - mean_distance : float - Mean pairwise Euclidean distance
        - std_distance : float - Std of pairwise distances
        - min_distance : float - Minimum pairwise distance
        - max_distance : float - Maximum pairwise distance
        - distance_range : float - max - min distance
        - nearest_pair : tuple[int, int] - Indices of nearest pair
        - farthest_pair : tuple[int, int] - Indices of farthest pair
        - distance_matrix : np.ndarray - Full pairwise distance matrix
        - hull_volume : float | None - Convex hull volume (3D+ only)
        - hull_area : float | None - Convex hull surface area (3D+ only)

    Examples
    --------
    >>> coords = adata.uns["archetype_coordinates"]
    >>> stats = compute_archetype_statistics(coords)
    >>> print(f"Archetypes span {stats['distance_range']:.2f} units")
    """
    # Handle torch tensors
    if hasattr(archetype_coordinates, "detach"):
        archetype_coordinates = archetype_coordinates.detach().cpu().numpy()

    from scipy.spatial.distance import pdist, squareform

    # Compute pairwise distances
    distances = pdist(archetype_coordinates, metric="euclidean")
    distance_matrix = squareform(distances)

    # Compute statistics (convert to Python native types for JSON serialization)
    stats = {
        "n_archetypes": int(archetype_coordinates.shape[0]),
        "n_dimensions": int(archetype_coordinates.shape[1]),
        "mean_distance": float(np.mean(distances)),
        "std_distance": float(np.std(distances)),
        "min_distance": float(np.min(distances)),
        "max_distance": float(np.max(distances)),
        "distance_range": float(np.max(distances) - np.min(distances)),
        "distance_matrix": distance_matrix,
    }

    # Find nearest and farthest archetype pairs
    min_idx = np.unravel_index(np.argmin(distance_matrix + np.eye(len(distance_matrix)) * 1e10), distance_matrix.shape)
    max_idx = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)

    stats["nearest_pair"] = (int(min_idx[0]), int(min_idx[1]))
    stats["farthest_pair"] = (int(max_idx[0]), int(max_idx[1]))

    # Compute hull volume (for 3D+)
    if archetype_coordinates.shape[1] >= 3:
        from scipy.spatial import ConvexHull

        try:
            hull = ConvexHull(archetype_coordinates)
            stats["hull_volume"] = float(hull.volume)
            stats["hull_area"] = float(hull.area)
        except:
            stats["hull_volume"] = None
            stats["hull_area"] = None

    if verbose:
        print("[STATS] Archetype Statistics")
        print("=" * 50)
        print(f"Number of archetypes: {stats['n_archetypes']}")
        print(f"Embedding dimensions: {stats['n_dimensions']}")
        print("\nDistance statistics:")
        print(f"  Mean distance: {stats['mean_distance']:.4f}")
        print(f"  Std distance:  {stats['std_distance']:.4f}")
        print(f"  Min distance:  {stats['min_distance']:.4f}")
        print(f"  Max distance:  {stats['max_distance']:.4f}")
        print(f"  Range:         {stats['distance_range']:.4f}")
        print(f"\nNearest archetypes:  A{stats['nearest_pair'][0] + 1} - A{stats['nearest_pair'][1] + 1}")
        print(f"Farthest archetypes: A{stats['farthest_pair'][0] + 1} - A{stats['farthest_pair'][1] + 1}")

        if "hull_volume" in stats and stats["hull_volume"] is not None:
            print(f"\nConvex hull volume: {stats['hull_volume']:.4f}")
            print(f"Convex hull area:   {stats['hull_area']:.4f}")

    return stats

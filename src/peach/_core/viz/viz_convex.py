# Convex data and archetypal space visualization
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from ..utils.analysis import get_archetypal_coordinates

"""
Convex Data and Archetypal Space Visualization
Specialized visualization tools for archetypal analysis results, convex hulls, and high-dimensional data exploration.

=== MODULE API INVENTORY ===

CORE VISUALIZATION FUNCTIONS:
├── visualize_convex_data(points: np.ndarray, archetypes: np.ndarray = None, dims_to_plot: list = None, title: str = None) -> go.Figure
│   └── Purpose: Interactive 2D/3D visualization of data points with archetypal convex hull overlay
│   └── Inputs: points(np.ndarray [n_points, n_dimensions]), archetypes(optional np.ndarray [n_archetypes, n_dimensions]), dims_to_plot(list of 2-3 dimension indices), title(str)
│   └── Outputs: go.Figure with data points as scatter plot, archetypes as red markers with hull edges
│   └── Features: Automatic 2D/3D detection, convex hull edge connections, archetypal labeling
│   └── Use Case: Visualizing synthetic convex data generation and archetypal fits

├── plot_archetype_weights(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str = 'cpu') -> plt.Figure
│   └── Purpose: Comprehensive analysis of archetypal weight matrices (A and B) with distribution and dominance patterns
│   └── Inputs: model(trained Deep_AA), dataloader(torch.DataLoader), device(str, 'cpu'|'cuda')
│   └── Outputs: plt.Figure with 2x2 subplot layout analyzing weight patterns
│   └── Panels: A matrix distributions, B matrix distributions (dummy), dominant archetype per sample, mean weights comparison
│   └── Features: Real-time weight analysis using model.analyze_archetypal_weights(), histogram overlays
│   └── Use Case: Understanding sample-archetype associations and weight sparsity patterns

ADVANCED ARCHETYPAL SPACE ANALYSIS:
├── visualize_archetypal_space(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str = 'cpu', dims_to_plot: List[int] = [0, 1, 2], title: str = "Archetypal Space Visualization", max_points: int = 1000, show_reconstructions: bool = True, use_full_dataset: bool = True) -> go.Figure
│   └── Purpose: Advanced comparative visualization of original data vs archetypal reconstructions in latent/PCA space
│   └── Inputs: model(trained Deep_AA), dataloader(torch.DataLoader), device(str), dims_to_plot(List[int], 2-3 dimensions), title(str), max_points(int, performance limit), show_reconstructions(bool, dual-panel mode), use_full_dataset(bool, full vs single batch)
│   └── Outputs: go.Figure with side-by-side comparison plots (if show_reconstructions=True)
│   └── Features: Full dataset loading with intelligent subsampling, dual-panel original vs reconstructed comparison, 2D/3D automatic adaptation, archetypal hull overlay
│   └── Performance: Handles large datasets via max_points subsampling, GPU-compatible processing
│   └── Use Case: Model quality assessment, reconstruction fidelity analysis, archetypal space exploration

ADVANCED FEATURES:
├── Data Loading: Full dataset processing with memory-efficient batching
├── Intelligent Subsampling: Random sampling when dataset exceeds max_points threshold
├── Multi-dimensional Support: Automatic 2D/3D plotting with consistent interfaces  
├── Interactive Elements: Plotly-based hover information, zoom, and pan capabilities
├── Performance Optimization: GPU tensor processing with CPU visualization data preparation
├── Hull Visualization: Convex hull edge rendering between all archetype pairs
├── Comparative Analysis: Side-by-side original vs reconstructed data visualization

EXTERNAL DEPENDENCIES:
├── plotly.graph_objects + plotly.subplots: Interactive web-based visualizations
├── matplotlib.pyplot: Static publication-quality plots  
├── numpy: Numerical array operations and data handling
├── torch: GPU tensor processing and model evaluation
├── itertools.combinations: Efficient archetype pair enumeration for hull edges
├── ..utils.analysis.get_archetypal_coordinates: Model coordinate extraction and archetype computation

COORDINATE SPACE HANDLING:
├── Input Space: Original high-dimensional single-cell data
├── Latent Space: VAE-encoded representations
├── PCA Space: Principal component projections for visualization
├── Archetypal Space: Convex combinations of archetypal coordinates
├── Reconstruction Space: Model-generated archetypal reconstructions
"""


# update 20241109
def visualize_convex_data(
    points: np.ndarray, archetypes: np.ndarray = None, dims_to_plot: list = None, title: str = None
):
    """
    Visualize data in 2D or 3D using Plotly

    Args:
        points: Array of shape (n_points, n_dimensions)
        archetypes: Optional array of shape (n_archetypes, n_dimensions)
        dims_to_plot: List of 2 or 3 dimension indices to plot
        title: Optional title for the plot
    """
    n_dimensions = points.shape[1]
    if dims_to_plot is None:
        dims_to_plot = list(range(min(3, n_dimensions)))

    if len(dims_to_plot) == 2:
        fig = go.Figure()

        # Add data points
        fig.add_trace(
            go.Scatter(
                x=points[:, dims_to_plot[0]],
                y=points[:, dims_to_plot[1]],
                mode="markers",
                marker=dict(size=3, color="blue", opacity=0.5),
                name="Data Points",
            )
        )

        # Add archetypes if provided
        if archetypes is not None:
            fig.add_trace(
                go.Scatter(
                    x=archetypes[:, dims_to_plot[0]],
                    y=archetypes[:, dims_to_plot[1]],
                    mode="markers+text",
                    marker=dict(size=10, color="red"),
                    text=[f"A{i + 1}" for i in range(len(archetypes))],
                    textposition="top center",
                    name="Archetypes",
                )
            )

            # Add lines between archetypes
            for i, j in combinations(range(len(archetypes)), 2):
                fig.add_trace(
                    go.Scatter(
                        x=[archetypes[i, dims_to_plot[0]], archetypes[j, dims_to_plot[0]]],
                        y=[archetypes[i, dims_to_plot[1]], archetypes[j, dims_to_plot[1]]],
                        mode="lines",
                        line=dict(color="red", width=1),
                        opacity=0.5,
                        showlegend=False,
                    )
                )

    elif len(dims_to_plot) == 3:
        fig = go.Figure()

        # Add data points
        fig.add_trace(
            go.Scatter3d(
                x=points[:, dims_to_plot[0]],
                y=points[:, dims_to_plot[1]],
                z=points[:, dims_to_plot[2]],
                mode="markers",
                marker=dict(size=2, color="blue", opacity=0.5),
                name="Data Points",
            )
        )

        # Add archetypes if provided
        if archetypes is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=archetypes[:, dims_to_plot[0]],
                    y=archetypes[:, dims_to_plot[1]],
                    z=archetypes[:, dims_to_plot[2]],
                    mode="markers+text",
                    marker=dict(size=5, color="red"),
                    text=[f"A{i + 1}" for i in range(len(archetypes))],
                    name="Archetypes",
                )
            )

            # Add lines between archetypes
            for i, j in combinations(range(len(archetypes)), 2):
                fig.add_trace(
                    go.Scatter3d(
                        x=[archetypes[i, dims_to_plot[0]], archetypes[j, dims_to_plot[0]]],
                        y=[archetypes[i, dims_to_plot[1]], archetypes[j, dims_to_plot[1]]],
                        z=[archetypes[i, dims_to_plot[2]], archetypes[j, dims_to_plot[2]]],
                        mode="lines",
                        line=dict(color="red", width=2),
                        opacity=0.5,
                        showlegend=False,
                    )
                )

    # Update layout
    fig.update_layout(
        title=title or "Data Visualization",
        scene=dict(
            xaxis_title=f"Dimension {dims_to_plot[0]}",
            yaxis_title=f"Dimension {dims_to_plot[1]}",
            zaxis_title=f"Dimension {dims_to_plot[2]}" if len(dims_to_plot) == 3 else None,
            aspectmode="cube",
        ),
        height=700,
        width=900,
    )

    fig.show()
    return fig


def plot_archetype_weights(
    model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str = "cpu"
) -> plt.Figure:
    """
    Plot distribution of archetypal weights (A matrix only for Deep_AA).

    Args:
        model: Trained Deep_AA model
        dataloader: DataLoader containing dataset
        device: Device to run computations on

    Returns
    -------
        Matplotlib figure
    """
    model.eval()

    # Get a representative batch
    data_batch = next(iter(dataloader))[0].to(device)

    with torch.no_grad():
        # Get weight analysis
        weight_analysis = model.analyze_archetypal_weights(data_batch)

        # Get raw coordinates for distribution plots
        coords = get_archetypal_coordinates(model, data_batch)
        A = coords["A"].cpu().numpy()  # [batch_size, n_archetypes]
        B = coords["B"].cpu().numpy()  # [batch_size, n_archetypes] - dummy uniform weights

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot A matrix weight distributions
    ax1 = axes[0, 0]
    for i in range(A.shape[1]):
        ax1.hist(A[:, i], alpha=0.6, label=f"Archetype {i}", bins=30)
    ax1.set_title("A Matrix: Sample Weight Distributions (Primary)")
    ax1.set_xlabel("Weight Value")
    ax1.set_ylabel("Frequency")
    ax1.legend()

    # Plot B matrix weight distributions (dummy - should be uniform)
    ax2 = axes[0, 1]
    for i in range(B.shape[1]):
        ax2.hist(B[:, i], alpha=0.6, label=f"Archetype {i}", bins=30)
    ax2.set_title("B Matrix: Dummy Weights (Uniform)")
    ax2.set_xlabel("Weight Value")
    ax2.set_ylabel("Frequency")
    ax2.legend()

    # Plot dominant archetype per sample
    ax3 = axes[1, 0]
    dominant_archetypes = np.argmax(A, axis=1)
    unique, counts = np.unique(dominant_archetypes, return_counts=True)
    ax3.bar(unique, counts)
    ax3.set_title("Dominant Archetype Distribution")
    ax3.set_xlabel("Archetype Index")
    ax3.set_ylabel("Number of Samples")
    ax3.set_xticks(range(A.shape[1]))

    # Plot mean weights
    ax4 = axes[1, 1]
    A_means = weight_analysis["A_matrix"]["mean_weights"].cpu().numpy()
    B_means = weight_analysis["B_matrix"]["mean_weights"].cpu().numpy()

    x = np.arange(len(A_means))
    width = 0.35

    ax4.bar(x - width / 2, A_means, width, label="A Matrix (Active)", alpha=0.8)
    ax4.bar(x + width / 2, B_means, width, label="B Matrix (Dummy)", alpha=0.8)
    ax4.set_title("Mean Weights by Archetype")
    ax4.set_xlabel("Archetype Index")
    ax4.set_ylabel("Mean Weight")
    ax4.set_xticks(x)
    ax4.legend()

    plt.tight_layout()
    return fig


def visualize_archetypal_space(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
    dims_to_plot: list[int] = [0, 1, 2],
    title: str = "Archetypal Space Visualization",
    max_points: int = 1000,
    show_reconstructions: bool = True,
    use_full_dataset: bool = True,
) -> go.Figure:
    """
    Visualize archetypal space with optional reconstruction comparison.

    Creates interactive 2D/3D scatter plot showing data points,
    archetype positions, and optionally archetypal reconstructions.

    Parameters
    ----------
    model : torch.nn.Module
        Trained Deep_AA model.
    dataloader : torch.utils.data.DataLoader
        DataLoader containing dataset.
    device : str, default: 'cpu'
        Device for computations.
    dims_to_plot : list[int], default: [0, 1, 2]
        Dimension indices to plot (2 or 3 dimensions).
    title : str, default: 'Archetypal Space Visualization'
        Plot title.
    max_points : int, default: 1000
        Maximum points to plot (randomly subsampled if exceeded).
    show_reconstructions : bool, default: True
        Whether to show side-by-side reconstruction comparison.
    use_full_dataset : bool, default: True
        Whether to load full dataset or use single batch.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive figure with:

        If show_reconstructions=True (default):
        - Left panel: Original data points + archetype positions + hull edges
        - Right panel: Archetypal reconstructions + archetype positions + hull edges

        If show_reconstructions=False:
        - Single panel: Original data + archetypes

        Archetype positions shown as red markers with labels (A1, A2, ...).
        Hull edges connect all archetype pairs.

    Notes
    -----
    Full dataset mode (use_full_dataset=True) loads all batches into
    memory before visualization. For very large datasets, consider
    setting use_full_dataset=False or reducing max_points.

    Examples
    --------
    >>> fig = visualize_archetypal_space(model, dataloader, dims_to_plot=[0, 1, 2], show_reconstructions=True)
    >>> fig.show()
    """
    model.eval()

    if use_full_dataset:
        # Collect all data from dataloader
        print("Loading full dataset...")
        all_data = []
        all_arch_reconstructions = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                data_batch = batch[0].to(device)

                # Get model outputs for this batch
                outputs = model(data_batch)

                all_data.append(data_batch.cpu())
                all_arch_reconstructions.append(outputs["arch_recons"].cpu())

        # Concatenate all batches
        full_data = torch.cat(all_data, dim=0)
        full_arch_reconstructions = torch.cat(all_arch_reconstructions, dim=0)

        print(f"Full dataset shape: {full_data.shape}")

        # Sample points if dataset is too large
        if full_data.shape[0] > max_points:
            indices = torch.randperm(full_data.shape[0])[:max_points]
            data_batch = full_data[indices]
            arch_reconstructed_data = full_arch_reconstructions[indices].numpy()
            print(f"Sampled {max_points} points from {full_data.shape[0]} total")
        else:
            data_batch = full_data
            arch_reconstructed_data = full_arch_reconstructions.numpy()

        # Get archetypes using existing model method (archetypes are model parameters, not data-dependent)
        sample_batch = full_data[: min(128, full_data.shape[0])].to(device)
        with torch.no_grad():
            coords = get_archetypal_coordinates(model, sample_batch)
            archetypes = coords["Y"].cpu().numpy()  # [n_archetypes, input_dim]

        original_data = data_batch.numpy()

    else:
        # Use just one batch (original behavior)
        data_batch = next(iter(dataloader))[0].to(device)

        # Limit points for visualization performance
        if data_batch.shape[0] > max_points:
            indices = torch.randperm(data_batch.shape[0])[:max_points]
            data_batch = data_batch[indices]

        with torch.no_grad():
            # Get model outputs
            outputs = model(data_batch)
            coords = get_archetypal_coordinates(model, data_batch)

            # Extract data
            original_data = data_batch.cpu().numpy()
            arch_reconstructed_data = outputs["arch_recons"].cpu().numpy()
            archetypes = coords["Y"].cpu().numpy()  # [n_archetypes, input_dim]

    print(f"Plotting data shape: {original_data.shape}")
    print(f"Archetypal reconstructed data shape: {arch_reconstructed_data.shape}")
    print(f"Archetypes shape: {archetypes.shape}")

    # Create subplot layout
    if show_reconstructions:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Original Data + Archetypes", "Archetypal Reconstructions + Archetypes"),
            specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]]
            if len(dims_to_plot) == 3
            else [[{"type": "scatter"}, {"type": "scatter"}]],
        )
    else:
        fig = go.Figure()

    # Helper function to add traces
    def add_data_and_archetypes(fig, data, archetypes, dims, title_suffix="", row=None, col=None):
        if len(dims) == 2:
            # 2D plot
            fig.add_trace(
                go.Scatter(
                    x=data[:, dims[0]],
                    y=data[:, dims[1]],
                    mode="markers",
                    marker=dict(size=2, color="blue", opacity=0.25),
                    name=f"Data Points{title_suffix}",
                    showlegend=(row is None or row == 1) and (col is None or col == 1),
                ),
                row=row,
                col=col,
            )

            # Add archetypes
            fig.add_trace(
                go.Scatter(
                    x=archetypes[:, dims[0]],
                    y=archetypes[:, dims[1]],
                    mode="markers+text",
                    marker=dict(size=7, color="red"),
                    text=[f"A{i + 1}" for i in range(len(archetypes))],
                    textposition="top center",
                    name=f"Archetypes{title_suffix}",
                    showlegend=(row is None or row == 1) and (col is None or col == 1),
                ),
                row=row,
                col=col,
            )

            # Add lines between archetypes
            for i, j in combinations(range(len(archetypes)), 2):
                fig.add_trace(
                    go.Scatter(
                        x=[archetypes[i, dims[0]], archetypes[j, dims[0]]],
                        y=[archetypes[i, dims[1]], archetypes[j, dims[1]]],
                        mode="lines",
                        line=dict(color="red", width=5),
                        opacity=0.5,
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

        elif len(dims) == 3:
            # 3D plot
            fig.add_trace(
                go.Scatter3d(
                    x=data[:, dims[0]],
                    y=data[:, dims[1]],
                    z=data[:, dims[2]],
                    mode="markers",
                    marker=dict(size=2, color="blue", opacity=0.25),
                    name=f"Data Points{title_suffix}",
                    showlegend=(row is None or row == 1) and (col is None or col == 1),
                ),
                row=row,
                col=col,
            )

            # Add archetypes
            fig.add_trace(
                go.Scatter3d(
                    x=archetypes[:, dims[0]],
                    y=archetypes[:, dims[1]],
                    z=archetypes[:, dims[2]],
                    mode="markers+text",
                    marker=dict(size=5, color="red"),
                    text=[f"A{i + 1}" for i in range(len(archetypes))],
                    name=f"Archetypes{title_suffix}",
                    showlegend=(row is None or row == 1) and (col is None or col == 1),
                ),
                row=row,
                col=col,
            )

            # Add lines between archetypes
            for i, j in combinations(range(len(archetypes)), 2):
                fig.add_trace(
                    go.Scatter3d(
                        x=[archetypes[i, dims[0]], archetypes[j, dims[0]]],
                        y=[archetypes[i, dims[1]], archetypes[j, dims[1]]],
                        z=[archetypes[i, dims[2]], archetypes[j, dims[2]]],
                        mode="lines",
                        line=dict(color="red", width=2),
                        opacity=0.5,
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

    # Add traces to figure
    if show_reconstructions:
        # Original data (left subplot)
        add_data_and_archetypes(fig, original_data, archetypes, dims_to_plot, title_suffix=" (Original)", row=1, col=1)

        # Archetypal reconstructions (right subplot)
        add_data_and_archetypes(
            fig, arch_reconstructed_data, archetypes, dims_to_plot, title_suffix=" (Archetypal)", row=1, col=2
        )
    else:
        # Single plot with original data
        add_data_and_archetypes(fig, original_data, archetypes, dims_to_plot)

    # Update layout
    if len(dims_to_plot) == 3:
        scene_layout = dict(
            xaxis_title=f"PC {dims_to_plot[0]}",
            yaxis_title=f"PC {dims_to_plot[1]}",
            zaxis_title=f"PC {dims_to_plot[2]}",
            aspectmode="cube",
        )
        if show_reconstructions:
            fig.update_layout(
                scene=scene_layout,
                scene2=scene_layout,
                title=title,
                height=700,
                width=1400,  # Adjusted for 2 subplots
            )
        else:
            fig.update_layout(scene=scene_layout, title=title, height=700, width=900)
    else:
        axis_layout = dict(title=f"PC {dims_to_plot[0]}" if "PC" in title else f"Dimension {dims_to_plot[0]}")
        yaxis_layout = dict(title=f"PC {dims_to_plot[1]}" if "PC" in title else f"Dimension {dims_to_plot[1]}")

        if show_reconstructions:
            fig.update_layout(
                xaxis=axis_layout,
                yaxis=yaxis_layout,
                xaxis2=axis_layout,
                yaxis2=yaxis_layout,
                title=title,
                height=600,
                width=1400,  # Adjusted for 2 subplots
            )
        else:
            fig.update_layout(xaxis=axis_layout, yaxis=yaxis_layout, title=title, height=600, width=900)

    return fig

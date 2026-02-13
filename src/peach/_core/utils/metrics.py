# metrics playground


import numpy as np
import torch
from torch.nn import functional as F

"""
Training Metrics and Performance Tracking
=========================================

Metrics calculation, tracking, and constraint validation for archetypal analysis.

This module provides:

- VAE performance metrics (RMSE, KLD, ELBO)
- Epoch-level metric extraction from loss dictionaries
- Training history tracking with range analysis
- Archetypal constraint validation and diagnostics
- Archetype R² calculation

Main Functions
--------------
calculate_vae_metrics : VAE performance metrics
calculate_epoch_metrics : Extract scalar metrics from loss dict
calculate_archetype_r2 : Standardized R² computation
calculate_vertex_metrics : Archetype position quality
diagnose_constraint_violations : Constraint satisfaction analysis
print_final_metrics : Formatted metrics summary

Main Classes
------------
MetricsTracker : Track metrics over training epochs

Type Definitions
----------------
See ``peach._core.types`` for Pydantic models:

- ``VAEMetrics`` : Return type of calculate_vae_metrics()
- ``EpochMetrics`` : Return type of calculate_epoch_metrics()
- ``MetricSummary`` : Per-metric summary from MetricsTracker
- ``MetricsHistory`` : Full history from MetricsTracker.get_history()
- ``VertexMetrics`` : Return type of calculate_vertex_metrics()
- ``ConstraintDiagnostics`` : Return type of diagnose_constraint_violations()
- ``LOSS_METRICS`` : Set of metrics where lower is better
- ``IMPROVEMENT_METRICS`` : Set of metrics where higher is better

Examples
--------
>>> from peach._core.utils.metrics import (
...     calculate_vae_metrics,
...     calculate_archetype_r2,
...     MetricsTracker,
...     diagnose_constraint_violations
... )
>>> 
>>> # VAE metrics
>>> metrics = calculate_vae_metrics(recons, inputs, mu, log_var)
>>> print(f"RMSE: {metrics['rmse']:.4f}")
>>> 
>>> # Track training
>>> tracker = MetricsTracker()
>>> for epoch in range(n_epochs):
...     epoch_metrics = calculate_epoch_metrics(loss_dict)
...     tracker.update(epoch_metrics)
>>> summaries = tracker.get_metric_summaries()
>>> 
>>> # Constraint checking
>>> diagnostics = diagnose_constraint_violations(A, B)
>>> if not diagnostics['summary']['constraints_satisfied_max']:
...     print("Constraint violations detected!")
"""


def calculate_vae_metrics(
    recons: torch.Tensor, input: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor, beta: float = 1.0
) -> dict[str, float]:
    """
    Calculate VAE metrics
    args:
        recons: reconstructed data tensor
        input: original input data tensor
        mu: mean tensor from encoder
        log_var: log variance tensor from encoder
        beta: weight for KLD term

    Returns
    -------
        dictionary containing metrics
    """
    metrics = {}  # initialize dict

    # RMSE
    metrics["rmse"] = torch.sqrt(F.mse_loss(recons, input)).item()

    # KLD
    kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1)).item()
    metrics["kld"] = kld

    # ELBO
    reconstruction_loss = F.mse_loss(recons, input, reduction="sum") / input.size(0)  # normalizing to batch size
    elbo = (
        -reconstruction_loss - beta * kld
    )  # this tracks ELBO as a loss quantity itself, convert to elbo = reconstruction_loss + beta * kld for use in training to minimize it
    metrics["elbo"] = elbo.item()

    return metrics


class MetricsTracker:
    """Track metrics over epochs"""

    def __init__(self):
        # Initialize with expected metric names from loss_dict
        self.metrics_history = {
            # Loss metrics
            "loss": [],
            "archetypal_loss": [],
            "KLD": [],
            # Stability metrics (all scaled 0-1, higher = more stable)
            "vertex_stability_latent": [],
            "mean_vertex_stability_latent": [],
            "max_vertex_stability_latent": [],
            "vertex_stability_pca": [],
            "mean_vertex_stability_pca": [],
            "max_vertex_stability_pca": [],
            # Performance metrics
            "archetype_r2": [],
            "rmse": [],
        }
        # Track min/max for each metric
        self.metrics_history = {}
        self.metrics_ranges = {}

    def update(self, epoch_metrics: dict[str, float]):
        # Update metrics after each epoch
        for name, value in epoch_metrics.items():
            # Update history
            if name not in self.metrics_history:
                self.metrics_history[name] = []
            self.metrics_history[name].append(value)

            # Update ranges
            if name not in self.metrics_ranges:
                self.metrics_ranges[name] = {"min": value, "max": value}
            else:
                self.metrics_ranges[name]["min"] = min(self.metrics_ranges[name]["min"], value)
                self.metrics_ranges[name]["max"] = max(self.metrics_ranges[name]["max"], value)

    def get_history(self) -> dict[str, list]:
        return self.metrics_history

    def get_metric_summaries(self) -> dict[str, dict[str, float]]:
        """
        Get summary statistics for each metric including:
        - Final value
        - Range (min, max)
        - Total change (max - min)
        - Percent improvement from worst to best

        Returns
        -------
            Dictionary of metric summaries
        """
        summaries = {}
        # Define metrics where lower values are better
        loss_metrics = ["loss", "archetypal_loss", "KLD", "rmse", "loss_delta"]
        # Define metrics where higher values are better (all stability metrics)
        improvement_metrics = [
            "archetype_r2",
            "vertex_stability_latent",
            "mean_vertex_stability_latent",
            "max_vertex_stability_latent",
            "vertex_stability_pca",
            "mean_vertex_stability_pca",
            "max_vertex_stability_pca",
        ]

        for name, history in self.metrics_history.items():
            if not history:  # Skip empty metrics
                continue

            final_value = history[-1]
            metric_range = self.metrics_ranges[name]
            total_range = metric_range["max"] - metric_range["min"]

            # Calculate percent improvement (handle both minimization and maximization metrics)
            if name in loss_metrics:
                # For loss metrics (lower is better)
                pct_improvement = (
                    (metric_range["max"] - final_value) / (metric_range["max"] - metric_range["min"] + 1e-8)
                ) * 100
            else:
                # For improvement metrics (higher is better)
                pct_improvement = (
                    (final_value - metric_range["min"]) / (metric_range["max"] - metric_range["min"] + 1e-8)
                ) * 100

            summaries[name] = {
                "final": final_value,
                "min": metric_range["min"],
                "max": metric_range["max"],
                "range": total_range,
                "pct_improvement": pct_improvement,
            }

        return summaries


def calculate_epoch_metrics(loss_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    """
    Calculate epoch-level metrics from loss dictionary

    Args:
        loss_dict: Dictionary containing loss components and metrics

    Returns
    -------
        Dictionary of scalar metrics for the epoch
    """
    metrics = {}

    # Handle scalar metrics - UPDATED LIST
    scalar_metrics = [
        "loss",
        "archetypal_loss",
        "kld_loss",
        "reconstruction_loss",
        "KLD",
        "archetype_r2",
        "rmse",
        "loss_delta",
        # STABILITY METRICS
        "archetype_drift_mean",
        "archetype_drift_max",
        "archetype_drift_std",
        "archetype_stability_mean",
        "archetype_stability_min",
        "archetype_variance_mean",
        # CONSTRAINT METRICS
        "constraint_violation_rate",
        "constraints_satisfied",
        "A_sum_error",
        "A_negative_fraction",
        "B_sum_error",
        "B_negative_fraction",
        # ARCHETYPE TRANSFORM METRICS
        "archetype_transform_grad_norm",
        "archetype_transform_grad_mean",
        "archetype_transform_mean",
        "archetype_transform_std",
        "archetype_transform_norm",
    ]

    for key in scalar_metrics:
        if key in loss_dict:
            val = loss_dict[key]
            if torch.is_tensor(val):
                metrics[key] = val.item()

                # Add normalized versions of key metrics
                if key == "KLD":
                    if "latent_dim" in loss_dict:
                        metrics["KLD_per_dim"] = val.item() / loss_dict["latent_dim"]
                elif key == "archetypal_loss":
                    if "input_dim" in loss_dict:
                        metrics["archetypal_loss_per_dim"] = val.item() / loss_dict["input_dim"]
            else:
                metrics[key] = val

    # Handle per-dimension metrics by taking means
    per_dim_metrics = [k for k in loss_dict.keys() if k.endswith("_per_dim")]
    for key in per_dim_metrics:
        val = loss_dict[key]
        if torch.is_tensor(val):
            metrics[f"mean_{key}"] = val.mean().item()
            metrics[f"std_{key}"] = val.std().item()

    # Add convergence rate metrics if we have loss history
    if "loss_history" in loss_dict and len(loss_dict["loss_history"]) > 1:
        history = loss_dict["loss_history"]
        window_size = min(5, len(history))
        recent_losses = history[-window_size:]

        if len(recent_losses) > 1:
            metrics["convergence_rate"] = (recent_losses[-1] - recent_losses[0]) / window_size

        if len(recent_losses) > 2:
            metrics["loss_stability"] = np.std(recent_losses) / (np.mean(recent_losses) + 1e-8)

    return metrics


def print_final_metrics(history: dict, include_emojis: bool = True) -> dict:
    """
    Pretty print and return summarized metrics from training history with ranges.

    Args:
        history: Dictionary of metric histories
        include_emojis: Whether to include emoji icons in output

    Returns
    -------
        Dictionary of summarized metrics
    """
    icons = {
        "header": "[STATS] " if include_emojis else "",
        "loss": " " if include_emojis else "",
        "stability": " " if include_emojis else "",
        "performance": "[STATS] " if include_emojis else "",
    }

    print(f"\n{icons['header']}Final Metrics")
    print("=" * 60)

    # Calculate final values and ranges
    metrics_summary = {}
    for k, v in history.items():
        if len(v) > 0:
            metrics_summary[k] = {
                "final": v[-1],
                "min": min(v),
                "max": max(v),
                "range": max(v) - min(v),
                "pct_improvement": ((max(v) - v[-1]) / (max(v) - min(v) + 1e-8) * 100)
                if k in ["loss", "archetypal_loss", "KLD", "rmse"]
                else ((v[-1] - min(v)) / (max(v) - min(v) + 1e-8) * 100),
            }

    # Group metrics by type for cleaner display
    metric_groups = {
        f"{icons['loss']}Loss Metrics:": ["loss", "archetypal_loss", "KLD", "rmse"],
        f"{icons['stability']}Stability Metrics:": [
            "vertex_stability_latent_mean",
            "vertex_stability_latent",
            "mean_vertex_drift_latent",
            "max_vertex_drift_latent",
        ],
        f"{icons['performance']}Performance Metrics:": ["archetype_r2"],
    }

    # Print each group
    for group_name, metrics_list in metric_groups.items():
        print(f"\n{group_name}")
        print("-" * 40)
        for metric in metrics_list:
            if metric in metrics_summary:
                summary = metrics_summary[metric]
                print(f"{metric}:")
                print(f"  Final: {summary['final']:.4f}")
                print(f"  Range: [{summary['min']:.4f}, {summary['max']:.4f}]")
                print(f"  Change: {summary['range']:.4f}")
                print(f"  Improvement: {summary['pct_improvement']:.1f}%")

    return metrics_summary


def calculate_vertex_metrics(vertices, reconstruction_error, weight_sparsity=None):
    """Calculate metrics for vertex positions and quality."""
    if vertices is None:
        print("Warning: No vertices provided to calculate metrics")
        return {
            "vertex_range": None,
            "vertex_mean": None,
            "vertex_std": None,
            "reconstruction_error": reconstruction_error,
            "weight_sparsity": weight_sparsity if weight_sparsity is not None else None,
        }

    metrics = {
        "vertex_range": [vertices.min().item(), vertices.max().item()],
        "vertex_mean": vertices.mean().item(),
        "vertex_std": vertices.std().item(),
        "reconstruction_error": reconstruction_error,
    }

    if weight_sparsity is not None:
        metrics["weight_sparsity"] = weight_sparsity

    return metrics


def diagnose_constraint_violations(
    A: torch.Tensor, B: torch.Tensor, tolerance: float = 1e-3
) -> dict[str, dict[str, float]]:
    """
    Comprehensive constraint violation diagnosis to help debug discrepancies.

    Args:
        A: Reconstruction weights [batch_size, n_archetypes]
        B: Construction weights [batch_size, n_archetypes]
        tolerance: Tolerance for constraint violations

    Returns
    -------
        Dictionary with both max and mean violation statistics
    """
    with torch.no_grad():
        # A constraints: rows sum to 1, non-negative
        A_row_sums = A.sum(dim=1)  # Should be all 1s
        A_sum_violations = torch.abs(A_row_sums - 1.0)
        A_negative_mask = A < 0

        # B constraints: columns sum to 1, non-negative
        B_col_sums = B.sum(dim=0)  # Should be all 1s
        B_sum_violations = torch.abs(B_col_sums - 1.0)
        B_negative_mask = B < 0

        # Comprehensive statistics
        A_stats = {
            "sum_error_max": A_sum_violations.max().item(),
            "sum_error_mean": A_sum_violations.mean().item(),
            "sum_error_std": A_sum_violations.std().item(),
            "sum_error_median": A_sum_violations.median().item(),
            "negative_fraction": A_negative_mask.float().mean().item(),
            "negative_count": A_negative_mask.sum().item(),
            "violating_samples": (A_sum_violations > tolerance).sum().item(),
            "violation_rate": (A_sum_violations > tolerance).float().mean().item(),
        }

        B_stats = {
            "sum_error_max": B_sum_violations.max().item(),
            "sum_error_mean": B_sum_violations.mean().item(),
            "sum_error_std": B_sum_violations.std().item(),
            "sum_error_median": B_sum_violations.median().item(),
            "negative_fraction": B_negative_mask.float().mean().item(),
            "negative_count": B_negative_mask.sum().item(),
            "violating_archetypes": (B_sum_violations > tolerance).sum().item(),
            "violation_rate": (B_sum_violations > tolerance).float().mean().item(),
        }

        return {
            "A_matrix": A_stats,
            "B_matrix": B_stats,
            "summary": {
                "constraints_satisfied_max": (
                    A_stats["sum_error_max"] < tolerance
                    and A_stats["negative_fraction"] < tolerance
                    and B_stats["sum_error_max"] < tolerance
                    and B_stats["negative_fraction"] < tolerance
                ),
                "constraints_satisfied_mean": (
                    A_stats["sum_error_mean"] < tolerance
                    and A_stats["negative_fraction"] < tolerance
                    and B_stats["sum_error_mean"] < tolerance
                    and B_stats["negative_fraction"] < tolerance
                ),
                "max_vs_mean_discrepancy": {
                    "A_discrepancy": A_stats["sum_error_max"] - A_stats["sum_error_mean"],
                    "B_discrepancy": B_stats["sum_error_max"] - B_stats["sum_error_mean"],
                },
            },
        }


def calculate_archetype_r2(reconstructions: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """
    Calculate archetype R² using standardized Frobenius norm formulation.

    Based on the mathematical formulation from README.md:
    R² = 1 - (SS_res / SS_tot)
    where:
    - SS_res = ||X - X_reconstructed||²_F (sum of squared residuals)
    - SS_tot = ||X - X_mean||²_F (total sum of squares)

    Args:
        reconstructions: Reconstructed data tensor [batch_size, features]
        original: Original input data tensor [batch_size, features]

    Returns
    -------
        torch.Tensor: Archetype R² value (scalar tensor)

    Note:
        - Higher values indicate better reconstruction quality
        - R² = 1.0 means perfect reconstruction
        - R² = 0.0 means reconstruction is no better than using the mean
        - R² can be negative if reconstruction is worse than using the mean
    """
    # Calculate sum of squared residuals (Frobenius norm squared)
    ss_res = torch.sum((original - reconstructions) ** 2)

    # Calculate total sum of squares (variance around per-feature mean)
    ss_tot = torch.sum((original - original.mean(dim=0)) ** 2)

    # Calculate R² with numerical stability
    archetype_r2 = 1 - (ss_res / ss_tot.clamp(min=1e-8))

    return archetype_r2

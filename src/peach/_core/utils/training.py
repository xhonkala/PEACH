import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .analysis import get_archetypal_coordinates
from .metrics import MetricsTracker, calculate_epoch_metrics
from .performance_optimization import calculate_monitoring_frequency, optimize_torch_threads, should_monitor

"""
PHASE 4: Final Model Training Pipeline
=====================================

PURPOSE: Comprehensive training for final model with full evaluation and monitoring.

ARCHITECTURAL ROLE:
- Primary training component for Phase 4 final model training
- Complete training with stability tracking, constraint validation, evolution monitoring
- Generates final performance metrics for reporting
- Independent from CV training (Phase 2)

DESIGN PRINCIPLES:
- Comprehensive training with full monitoring
- Final model quality optimization (vs CV speed optimization)
- Complete state isolation from CV training
- Results used for final model evaluation and reporting

WORKFLOW INTEGRATION:
Phase 1: Data Loading → Phase 2: CV Search → Phase 3: Manual Selection → **Phase 4: Final Training** → Phase 5: Evaluation

IMPORTANT: This is for final model training only - CV training uses optimized cv_training.py.

=== MODULE API INVENTORY ===

MAIN FUNCTIONS:
 train_vae(model, data_loader, optimizer, n_epochs, device='cuda'|'cpu', save_path=None, **training_params) -> Dict[str, Any]
    Purpose: Primary training loop for archetypal analysis models with comprehensive monitoring and early stopping
    Inputs: model(torch.nn.Module), data_loader(DataLoader), optimizer(torch.optim.Optimizer), n_epochs(int), device(str), save_path(str), weight parameters(float), tracking flags(bool), seed(int)
    Outputs: Dict[str, Any] with 'history', 'final_model', 'final_optimizer', 'final_analysis', 'epoch_archetype_positions', 'training_config'
    Side Effects: Model training, checkpoint saving, metrics tracking, archetype evolution monitoring, GPU memory management

 track_archetype_stability(model: torch.nn.Module, Y: torch.Tensor, history_size: int = 20) -> Dict[str, float]
    Purpose: Track stability of archetype positions during training with drift and variance analysis
    Inputs: model(torch.nn.Module for history storage), Y(torch.Tensor [input_dim, n_archetypes] current positions), history_size(int tracking window)
    Outputs: Dict[str, float] with drift metrics, stability scores, and variance measures
    Side Effects: Updates model.archetype_positions_history, calculates stability metrics over time window

 save_checkpoint(state: Dict[str, Any], path: str)
    Purpose: Save comprehensive model checkpoint with all training state and metadata
    Inputs: state(Dict with model state, optimizer state, metrics, config), path(str save location)
    Outputs: None (saves to disk)
    Side Effects: Creates directory structure, saves checkpoint file, prints confirmation

 load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None) -> Dict[str, Any]
     Purpose: Load model checkpoint and restore complete training state
     Inputs: path(str checkpoint location), model(torch.nn.Module to restore), optimizer(torch.optim.Optimizer optional)
     Outputs: Dict[str, Any] loaded checkpoint data
     Side Effects: Restores model state, optimizer state if provided, prints confirmation

TRAINING COMPONENTS:
 Epoch-Level Processing: Batch iteration → Loss computation → Gradient updates → Metric aggregation
 Stability Tracking: Archetype position monitoring → Drift calculation → Stability scoring → Evolution analysis
 Constraint Validation: A/B matrix checking → Violation detection → Rate calculation → Compliance monitoring
 Early Stopping: Learning rate scheduling → Performance monitoring → Checkpoint management
 Memory Management: GPU cache cleanup → Batch processing → State management
 Comprehensive Logging: Progress reporting → Debug information → Final summaries

METRICS AND MONITORING:
 Loss Components: Total loss, archetypal loss, KLD loss, reconstruction loss breakdown
 Performance Metrics: Archetype R², RMSE, MAE for reconstruction quality
 Stability Metrics: Archetype drift (mean/max/std), stability scores, position variance
 Constraint Metrics: Violation rates, sum errors, constraint satisfaction status
 Convergence Metrics: Loss delta, convergence rates, learning progress
 Model-Specific: archetype_transform evolution tracking, gradient monitoring, parameter analysis

EXTERNAL DEPENDENCIES:
 From .metrics: MetricsTracker, calculate_epoch_metrics - Performance tracking and aggregation
 From .analysis: get_archetypal_coordinates - Coordinate extraction for validation
 From torch: Tensor operations, autograd, model state management
 From torch.nn.functional: Loss computations, constraint validation
 From torch.optim.lr_scheduler: ReduceLROnPlateau for adaptive learning rates
 From numpy: Statistical operations, random seed management
 From os: File operations, directory management for checkpoints

DATA FLOW PATTERNS:
 Input: DataLoader → Batch processing → Model forward → Loss computation → Backpropagation
 Monitoring: Model state → Coordinate extraction → Stability analysis → Constraint validation → Metric aggregation
 Evolution: Archetype positions → Historical tracking → Drift calculation → Stability scoring
 Checkpointing: Training state → Serialization → File storage → Recovery capability
 Reporting: Epoch metrics → Progress logging → Final analysis → Summary generation

ERROR HANDLING:
 GPU availability → Automatic device detection and fallback to CPU
 Missing model methods → Graceful degradation for optional features (archetype analysis, evolution tracking)
 Checkpoint operations → Directory creation, file handling with error propagation
 Memory constraints → Cache cleanup, batch size management
 Convergence issues → Maximum iteration limits, learning rate adaptation
 Numerical stability → Seed management, deterministic training, gradient monitoring
"""


def track_archetype_stability(model: torch.nn.Module, Y: torch.Tensor, history_size: int = 20) -> dict[str, float]:
    """
    Track stability of archetype positions during training.

    Args:
        model: Model instance (used to store history)
        Y: Current archetype positions [input_dim, n_archetypes]
        history_size: Number of past positions to track

    Returns
    -------
        Dictionary of stability metrics
    """
    # Initialize history if needed
    if not hasattr(model, "archetype_positions_history"):
        from collections import deque

        model.archetype_positions_history = deque(maxlen=history_size)

    # Add current position
    model.archetype_positions_history.append(Y.detach().clone())

    stability_metrics = {}

    if len(model.archetype_positions_history) > 1:
        # Calculate drift from previous position
        prev_Y = model.archetype_positions_history[-2]

        # Per-archetype drift (L2 norm across features)
        archetype_drift = torch.norm(Y - prev_Y, dim=0)  # [n_archetypes]

        stability_metrics.update(
            {
                "archetype_drift_mean": archetype_drift.mean().item(),
                "archetype_drift_max": archetype_drift.max().item(),
                "archetype_drift_std": archetype_drift.std().item(),
            }
        )

        # Calculate stability over longer window if we have enough history
        if len(model.archetype_positions_history) >= 5:
            # Calculate variance of positions over last 5 steps
            recent_positions = list(model.archetype_positions_history)[-5:]
            position_stack = torch.stack(recent_positions, dim=0)  # [5, input_dim, n_archetypes]

            # Variance across time for each archetype
            position_variance = torch.var(position_stack, dim=0)  # [input_dim, n_archetypes]
            archetype_variance = position_variance.mean(dim=0)  # [n_archetypes]

            # Use variance directly - lower variance = more stable
            stability_metrics.update(
                {
                    "archetype_variance_mean": archetype_variance.mean().item(),
                    "archetype_variance_max": archetype_variance.max().item(),
                    "archetype_variance_std": archetype_variance.std().item(),
                }
            )
    else:
        # First epoch - initialize with zeros
        n_archetypes = Y.shape[1]
        stability_metrics.update(
            {
                "archetype_drift_mean": 0.0,
                "archetype_drift_max": 0.0,
                "archetype_drift_std": 0.0,
                "archetype_variance_mean": 0.0,
                "archetype_variance_max": 0.0,
                "archetype_variance_std": 0.0,
            }
        )

    return stability_metrics


def _evaluate_validation_metrics(
    model: torch.nn.Module,
    validation_loader: torch.utils.data.DataLoader,
    device: str,
    archetypal_weight: float,
    kld_weight: float,
    reconstruction_weight: float,
) -> dict[str, float]:
    """
    Simple validation evaluation for early stopping - uses single batch only.

    Args:
        model: Model to evaluate
        validation_loader: DataLoader for validation data
        device: Device to run evaluation on
        archetypal_weight, kld_weight, reconstruction_weight: Loss weights

    Returns
    -------
        Dictionary of validation metrics
    """
    model.eval()

    try:
        with torch.no_grad():
            # Use only first batch for speed and memory efficiency
            data = next(iter(validation_loader))[0].to(device)

            # Forward pass
            outputs = model(data)

            # Calculate loss (reuse existing loss function)
            loss_dict = model.loss_function(
                outputs,
                archetypal_weight=archetypal_weight,
                kld_weight=kld_weight,
                reconstruction_weight=reconstruction_weight,
            )

            # Extract simple metrics that already exist in loss_dict
            val_metrics = {
                "loss": loss_dict["loss"].item(),
                "archetype_r2": loss_dict.get("archetype_r2", 0.0),
                "rmse": loss_dict.get("rmse", 0.0),
            }

            # Ensure all values are numbers
            for key, value in val_metrics.items():
                if torch.is_tensor(value):
                    val_metrics[key] = value.item()
                elif not isinstance(value, (int, float)):
                    val_metrics[key] = 0.0

    except Exception as e:
        print(f"   [WARNING]  Validation failed: {e}")
        val_metrics = {"loss": float("inf"), "archetype_r2": 0.0, "rmse": float("inf")}

    model.train()  # Switch back to training mode
    return val_metrics


def train_vae(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    device: str = "cpu",
    save_path: str = None,
    archetypal_weight: float = None,  # Use model's configured weight if None
    kld_weight: float = None,  # Use model's configured weight if None
    reconstruction_weight: float = 0.0,  # Legacy parameter (ignored by Deep_3)
    vae_recon_weight: float = 0.0,
    diversity_weight: float = 0.0,
    activation_func: str = "relu",
    track_stability: bool = True,
    validate_constraints: bool = True,
    lr_factor: float = 0.1,
    lr_patience: int = 10,
    seed: int = 42,
    constraint_tolerance: float = 1e-3,
    stability_history_size: int = 20,
    adata=None,  # Optional AnnData object to store coordinates
    store_coords_key: str = "archetype_coordinates",
    # NEW: Early stopping parameters
    early_stopping: bool = False,
    early_stopping_patience: int = 10,
    early_stopping_metric: str = "archetype_r2",  # 'archetype_r2', 'loss', 'rmse'
    min_improvement: float = 1e-4,
    validation_check_interval: int = 5,
    validation_data_loader: torch.utils.data.DataLoader = None,
    # PERFORMANCE: Monitoring frequency control
    monitor_frequency: int | None = None,  # Auto-calculate if None
    detailed_monitoring: bool = False,  # Enable all monitoring (overrides monitor_frequency)
    optimize_threads: bool = True,  # Auto-optimize PyTorch threads (SLURM-aware)
    _cv_mode: bool = False,  # Internal: suppress adata warning during CV
) -> tuple[dict[str, Any], torch.nn.Module]:
    """
    Comprehensive VAE training loop for archetypal analysis with optional early stopping

    Args:
        model: VAE model instance
        data_loader: PyTorch DataLoader for training data
        optimizer: PyTorch optimizer
        n_epochs: Maximum number of training epochs
        device: Device to train on
        save_path: Path to save checkpoints
        archetypal_weight: Weight for archetypal loss (None = use model's configured weight)
        kld_weight: Weight for KL divergence loss (None = use model's configured weight)
        reconstruction_weight: Weight for traditional VAE reconstruction loss (ignored by Deep_3)
        activation_func: Activation function to use ("relu", "softmax", "leakyrelu", etc.)
        track_stability: Whether to track archetype stability
        validate_constraints: Whether to validate archetypal constraints
        lr_factor: Factor to reduce learning rate by when plateauing
        lr_patience: Number of epochs to wait before reducing LR
        seed: Random seed for reproducibility
        constraint_tolerance: Tolerance for constraint violations
        stability_history_size: Number of past archetype positions to track
        adata: Optional AnnData object to store coordinates
        store_coords_key: Key for storing archetype coordinates in adata.uns
        early_stopping: Whether to enable early stopping based on validation metrics
        early_stopping_patience: Number of validation checks without improvement before stopping
        early_stopping_metric: Metric to monitor ('archetype_r2', 'loss', 'rmse')
        min_improvement: Minimum improvement required to reset patience counter
        validation_check_interval: Check validation every N epochs
        validation_data_loader: Optional validation DataLoader (if None, uses training data)
        monitor_frequency: Monitoring frequency in epochs (None = auto-calculate ~20 times)
        detailed_monitoring: Force monitoring every epoch (backward compatibility)
        optimize_threads: Auto-optimize PyTorch threads for HPC environments
    """
    # PERFORMANCE: Optimize PyTorch threads for HPC if requested
    if optimize_threads:
        n_cores = optimize_torch_threads(verbose=False)

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # PERFORMANCE: Calculate monitoring frequency
    if detailed_monitoring:
        monitor_frequency = 1  # Backward compatibility
    elif monitor_frequency is None:
        monitor_frequency = calculate_monitoring_frequency(n_epochs, target_checkpoints=20)

    # Set activation function if model supports it
    if hasattr(model, "set_activation"):
        model.set_activation(activation_func)

    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience
    )

    # DEFENSIVE: Validate device availability
    if device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                # Test MPS with dummy operation
                test_tensor = torch.zeros(1).to("mps")
                del test_tensor  # Clean up
                print("[INFO] Using MPS (Apple Silicon GPU)")
            except RuntimeError as e:
                raise RuntimeError(
                    f"MPS device requested but initialization failed: {e}. "
                    "Use device='cpu' instead."
                )
        else:
            raise RuntimeError(
                "MPS device requested but not available. "
                "Use device='cpu' instead."
            )
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but not available. "
                "Use device='cpu' or device='mps' instead."
            )

    model = model.to(device)
    tracker = MetricsTracker()
    best_loss = float("inf")

    # EARLY STOPPING INITIALIZATION
    early_stop_triggered = False
    convergence_epoch = n_epochs
    if early_stopping:
        # Use validation data if provided, otherwise use training data for early stopping
        validation_loader = validation_data_loader if validation_data_loader is not None else data_loader

        # Initialize early stopping variables
        if early_stopping_metric == "archetype_r2":
            best_val_metric = -float("inf")  # Higher is better
            improvement_direction = 1
        else:  # 'loss' or 'rmse'
            best_val_metric = float("inf")  # Lower is better
            improvement_direction = -1

        patience_counter = 0
        print(f"Early stopping enabled: monitoring {early_stopping_metric} with patience {early_stopping_patience}")

    # EPOCH-LEVEL STABILITY TRACKING INITIALIZATION
    epoch_archetype_positions = []  # Store archetype positions from each epoch

    # Use model's configured weights if not specified
    if archetypal_weight is None:
        archetypal_weight = getattr(model, "archetypal_weight", 0.9)  # Fallback to 0.9
    if kld_weight is None:
        kld_weight = getattr(model, "kld_weight", 0.1)  # Fallback to 0.1

    print(f"Starting training for {n_epochs} epochs...")
    print(f"Device: {device}")
    print(
        f"Archetypal weight: {archetypal_weight}, KLD weight: {kld_weight}, Reconstruction weight: {reconstruction_weight}"
    )
    print(
        f"  (Model configured: arch={getattr(model, 'archetypal_weight', 'N/A')}, kld={getattr(model, 'kld_weight', 'N/A')})"
    )
    print(f"Tracking stability: {track_stability}, Validating constraints: {validate_constraints}")

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []
        epoch_constraint_violations = []
        epoch_loss_dicts = []  # Collect loss components from all batches

        # BATCH PROCESSING LOOP
        for batch_idx, data_tuple in enumerate(data_loader):
            data = data_tuple[0].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(data)

            # DEBUGGING - Only run if monitoring this epoch
            # In training.py, around line 170, after outputs = model(data)
            if batch_idx == 0 and should_monitor(epoch, n_epochs, monitor_frequency):
                with torch.no_grad():
                    z = outputs["z"]
                    arch_recons = outputs["arch_recons"]

                    # Check archetypal coordinate constraints
                    z_sums = z.sum(dim=1)
                    print(f"\nEpoch {epoch} Debug:")
                    print(f"z row sums (should be ~1.0): {z_sums.mean():.4f} ± {z_sums.std():.4f}")
                    print(f"z stats: min={z.min():.4f}, max={z.max():.4f}, mean={z.mean():.4f}")

                    # Check reconstruction quality
                    recon_mse = F.mse_loss(arch_recons, data)
                    print(f"Batch reconstruction MSE: {recon_mse:.4f}")

                    # Check archetype positions
                    archetypes = outputs["archetypes"]
                    print(f"Archetype stats: min={archetypes.min():.4f}, max={archetypes.max():.4f}")

                    # Check if archetypes are learning
                    if hasattr(model, "_previous_archetypes"):
                        archetype_change = torch.norm(archetypes - model._previous_archetypes)
                        print(f"Archetype change since last debug: {archetype_change:.6f}")
                    model._previous_archetypes = archetypes.clone()

            # Calculate loss with reconstruction weight
            loss_dict = model.loss_function(
                outputs,
                archetypal_weight=archetypal_weight,
                kld_weight=kld_weight,
                reconstruction_weight=reconstruction_weight,
                current_epoch=epoch,  # ,
                # batch_contains_pretraining_data=batch_contains_pretraining_data
            )

            # Add archetypal-specific constraint tracking (per batch)
            try:
                coords = get_archetypal_coordinates(model, data)
                A, B, Y = coords["A"], coords["B"], coords["Y"]

                # PERFORMANCE: Constraint validation only during monitoring epochs
                if validate_constraints and should_monitor(epoch, n_epochs, monitor_frequency):
                    constraint_metrics = model.validate_constraints(A, B, tolerance=constraint_tolerance)
                    loss_dict.update(constraint_metrics)
                    epoch_constraint_violations.append(not bool(constraint_metrics["constraints_satisfied"]))
            except (AttributeError, Exception):
                # Model doesn't support archetypal coordinates or error occurred
                pass

            # Backward pass
            loss = loss_dict["loss"]
            loss.backward()

            # DEBUGGING - Only run if monitoring this epoch
            if batch_idx == 0 and should_monitor(epoch, n_epochs, monitor_frequency):
                archetype_grad = model.archetypes.grad
                if archetype_grad is not None:
                    grad_norm = torch.norm(archetype_grad).item()
                    grad_mean = torch.mean(torch.abs(archetype_grad)).item()
                    print(f"Archetype gradients: norm={grad_norm:.6f}, mean={grad_mean:.6f}")
                else:
                    print("[ERROR] Archetype gradients are None!")

            # Check archetype_transform gradients AFTER backward pass
            if hasattr(model, "archetype_transform"):
                transform_grads = [p.grad for p in model.archetype_transform.parameters() if p.grad is not None]
                if transform_grads:
                    loss_dict["archetype_transform_grad_norm"] = sum(torch.norm(g).item() for g in transform_grads)
                    loss_dict["archetype_transform_grad_mean"] = sum(torch.mean(torch.abs(g)).item() for g in transform_grads) / len(transform_grads)
                else:
                    loss_dict["archetype_transform_grad_norm"] = 0.0
                    loss_dict["archetype_transform_grad_mean"] = 0.0
                # Track transform parameter stats
                transform_params = torch.cat([p.detach().flatten() for p in model.archetype_transform.parameters()])
                loss_dict["archetype_transform_mean"] = transform_params.mean().item()
                loss_dict["archetype_transform_std"] = transform_params.std().item()
                loss_dict["archetype_transform_norm"] = torch.norm(transform_params).item()
            else:
                loss_dict["archetype_transform_grad_norm"] = 0.0
                loss_dict["archetype_transform_grad_mean"] = 0.0

            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_loss_dicts.append(loss_dict)

        # END OF EPOCH PROCESSING
        avg_loss = sum(epoch_losses) / len(epoch_losses)

        # Aggregate loss components across batches for this epoch
        epoch_aggregated_loss = {}
        if epoch_loss_dicts:
            # Average numeric loss components across batches
            numeric_keys = [
                k for k in epoch_loss_dicts[0].keys() if isinstance(epoch_loss_dicts[0][k], (int, float, torch.Tensor))
            ]

            # Also include boolean metrics without averaging
            boolean_keys = [k for k in epoch_loss_dicts[0].keys() if isinstance(epoch_loss_dicts[0][k], bool)]

            # Process numeric keys (average across batches)
            for key in numeric_keys:
                values = []
                for batch_loss_dict in epoch_loss_dicts:
                    if key in batch_loss_dict:
                        val = batch_loss_dict[key]
                        if torch.is_tensor(val):
                            val = val.item()
                        values.append(val)
                if values:
                    epoch_aggregated_loss[key] = sum(values) / len(values)

            # Process boolean keys (take last value)
            for key in boolean_keys:
                if key in epoch_loss_dicts[-1]:  # Use last batch value
                    epoch_aggregated_loss[key] = epoch_loss_dicts[-1][key]

        # Calculate constraint violation rate for this epoch
        if validate_constraints and epoch_constraint_violations:
            constraint_violation_rate = sum(epoch_constraint_violations) / len(epoch_constraint_violations)
            epoch_aggregated_loss["constraint_violation_rate"] = constraint_violation_rate

        # PERFORMANCE: EPOCH-LEVEL STABILITY TRACKING - Only during monitoring epochs
        if track_stability and should_monitor(epoch, n_epochs, monitor_frequency):
            model.eval()  # Switch to eval mode for stability calculation
            with torch.no_grad():
                # Get representative batch for archetype extraction
                sample_data = next(iter(data_loader))[0].to(device)
                coords = get_archetypal_coordinates(model, sample_data)
                current_Y = coords["Y"].detach().cpu()  # [n_archetypes, input_dim]

                # Store current epoch's archetype positions
                epoch_archetype_positions.append(current_Y.clone())

                # Deep_3 specific: Track archetype evolution using Hungarian algorithm
                evolution_metrics = {}
                if hasattr(model, "track_archetype_evolution"):
                    try:
                        evolution_result = model.track_archetype_evolution(epoch)
                        if evolution_result is not None:
                            evolution_metrics.update(evolution_result)
                            print(
                                f"    Archetype evolution: mean_drift={evolution_result['mean_drift']:.4f}, "
                                f"max_drift={evolution_result['max_drift']:.4f}"
                            )
                    except Exception as e:
                        print(f"   [WARNING]  Archetype evolution tracking failed: {e}")

                # Calculate stability metrics based on archetype evolution
                stability_metrics = {}
                if len(epoch_archetype_positions) >= 2:
                    # Calculate drift from previous epoch
                    prev_Y = epoch_archetype_positions[-2]

                    # Per-archetype drift (L2 norm across input dimensions)
                    archetype_drift = torch.norm(current_Y - prev_Y, dim=1)  # [n_archetypes]

                    stability_metrics.update(
                        {
                            "archetype_drift_mean": archetype_drift.mean().item(),
                            "archetype_drift_max": archetype_drift.max().item(),
                            "archetype_drift_std": archetype_drift.std().item(),
                        }
                    )

                    # Calculate stability over longer window if we have enough history
                    if len(epoch_archetype_positions) >= 5:
                        # Use last 5 epochs for stability calculation
                        recent_positions = epoch_archetype_positions[-5:]
                        position_stack = torch.stack(recent_positions, dim=0)  # [5, n_archetypes, input_dim]

                        # Variance across epochs for each archetype
                        position_variance = torch.var(position_stack, dim=0)  # [n_archetypes, input_dim]
                        archetype_variance = position_variance.mean(
                            dim=1
                        )  # [n_archetypes] - avg variance per archetype

                        # Use variance directly - lower variance = more stable
                        stability_metrics.update(
                            {
                                "archetype_variance_mean": archetype_variance.mean().item(),
                                "archetype_variance_max": archetype_variance.max().item(),
                                "archetype_variance_std": archetype_variance.std().item(),
                            }
                        )

                    # Keep history size manageable
                    if len(epoch_archetype_positions) > stability_history_size:
                        epoch_archetype_positions.pop(0)

                else:
                    # First epoch - initialize with neutral values
                    stability_metrics = {
                        "archetype_drift_mean": 0.0,
                        "archetype_drift_max": 0.0,
                        "archetype_drift_std": 0.0,
                        "archetype_variance_mean": 0.0,
                        "archetype_variance_max": 0.0,
                        "archetype_variance_std": 0.0,
                    }

                # Add stability metrics to epoch aggregated loss
                epoch_aggregated_loss.update(stability_metrics)

            model.train()  # Back to training mode

        # EARLY STOPPING CHECK
        # PERFORMANCE: Align with monitor_frequency when possible
        effective_val_interval = (
            validation_check_interval if detailed_monitoring else max(validation_check_interval, monitor_frequency)
        )
        if early_stopping and epoch % effective_val_interval == 0:
            try:
                val_metrics = _evaluate_validation_metrics(
                    model, validation_loader, device, archetypal_weight, kld_weight, reconstruction_weight
                )

                # Get the monitored metric
                current_val_metric = val_metrics.get(early_stopping_metric, float("inf"))

                # Check for improvement
                improved = False
                if improvement_direction == 1:  # Higher is better (e.g., archetype_r2)
                    if current_val_metric > best_val_metric + min_improvement:
                        improved = True
                else:  # Lower is better (e.g., loss, rmse)
                    if current_val_metric < best_val_metric - min_improvement:
                        improved = True

                if improved:
                    best_val_metric = current_val_metric
                    patience_counter = 0
                    print(f"   [OK] Validation {early_stopping_metric} improved to {current_val_metric:.6f}")
                else:
                    patience_counter += 1
                    print(
                        f"   ⏳ Validation {early_stopping_metric}: {current_val_metric:.6f} (patience {patience_counter}/{early_stopping_patience})"
                    )

                # Add validation metrics to epoch aggregated loss for tracking
                for key, value in val_metrics.items():
                    epoch_aggregated_loss[f"val_{key}"] = value

                # Check if early stopping should trigger
                if patience_counter >= early_stopping_patience:
                    convergence_epoch = epoch + 1
                    early_stop_triggered = True
                    print(f"    Early stopping triggered at epoch {convergence_epoch}")
                    break

            except Exception as e:
                print(f"   [ERROR] Early stopping validation failed: {e}")
                print("    Continuing training without early stopping for this epoch...")

        # Update metrics tracker with epoch-level aggregated metrics
        epoch_metrics = calculate_epoch_metrics(epoch_aggregated_loss)
        tracker.update(epoch_metrics)

        # Step learning rate scheduler
        scheduler.step(avg_loss)

        # Save best model
        if save_path and avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "metrics_history": tracker.get_history(),
                    "epoch_archetype_positions": epoch_archetype_positions,  # Include archetype evolution
                    "model_config": {
                        "input_dim": model.input_dim,
                        "latent_dim": model.latent_dim,
                        "n_archetypes": model.n_archetypes,
                        "archetypal_weight": archetypal_weight,
                        "kld_weight": kld_weight,
                        "reconstruction_weight": reconstruction_weight,
                    },
                },
                save_path,
            )

        # Print progress
        if epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1:
            print(f"\nEpoch {epoch + 1}/{n_epochs}")
            print(f"Average loss: {avg_loss:.4f}")

            if "archetypal_loss" in epoch_aggregated_loss:
                print(f"Archetypal loss: {epoch_aggregated_loss['archetypal_loss']:.4f}")
            if "kld_loss" in epoch_aggregated_loss:
                print(f"KLD loss: {epoch_aggregated_loss['kld_loss']:.4f}")
            if "reconstruction_loss" in epoch_aggregated_loss:
                print(f"Reconstruction loss: {epoch_aggregated_loss['reconstruction_loss']:.4f}")
            if "archetype_r2" in epoch_aggregated_loss:
                print(f"Archetype R²: {epoch_aggregated_loss['archetype_r2']:.4f}")

            # Print archetype transform monitoring metrics
            if "archetype_transform_grad_norm" in epoch_aggregated_loss:
                print(f"Archetype transform grad norm: {epoch_aggregated_loss['archetype_transform_grad_norm']:.6f}")
            if "archetype_transform_norm" in epoch_aggregated_loss:
                print(f"Archetype transform param norm: {epoch_aggregated_loss['archetype_transform_norm']:.4f}")

            # Print constraint validation
            if validate_constraints and "constraints_satisfied" in epoch_aggregated_loss:
                print(f"Constraints satisfied: {bool(epoch_aggregated_loss['constraints_satisfied'])}")
                if "constraint_violation_rate" in epoch_aggregated_loss:
                    print(f"Constraint violation rate: {epoch_aggregated_loss['constraint_violation_rate']:.3f}")

            # Print stability metrics
            if track_stability and "archetype_drift_mean" in epoch_aggregated_loss:
                print(f"Archetype drift (mean): {epoch_aggregated_loss['archetype_drift_mean']:.6f}")
                if "archetype_stability_mean" in epoch_aggregated_loss:
                    print(f"Archetype stability (mean): {epoch_aggregated_loss['archetype_stability_mean']:.4f}")

    # Final analysis
    print("\n" + "=" * 60)
    if early_stop_triggered:
        print(f"TRAINING COMPLETED (EARLY STOPPED AT EPOCH {convergence_epoch})")
    else:
        print("TRAINING COMPLETED")
    print("=" * 60)

    final_history = tracker.get_history()
    valid_history = {k: v for k, v in final_history.items() if len(v) > 0}

    # Final model analysis
    model.eval()
    with torch.no_grad():
        # Get final batch for analysis
        final_batch = next(iter(data_loader))[0].to(device)

        # Final constraint validation and weight analysis
        final_analysis = {}
        try:
            coords = get_archetypal_coordinates(model, final_batch)
            weight_analysis = model.analyze_archetypal_weights(final_batch)

            # Store coordinates in AnnData if provided
            if adata is not None:
                # Store the raw archetype coordinates from get_archetypal_coordinates
                adata.uns[store_coords_key] = coords["Y"].detach().cpu().numpy()  # [n_archetypes, input_dim]
                print(f"   [OK] Stored archetype coordinates in adata.uns['{store_coords_key}']: {coords['Y'].shape}")
            elif not _cv_mode:
                print("   [WARNING] AnnData missing - archetype coordinates not stored")

            final_analysis = {
                "final_constraint_validation": model.validate_constraints(coords["A"], coords["B"]),
                "archetypal_weights": {
                    "A_matrix": {
                        k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in weight_analysis["A_matrix"].items()
                    },
                    "B_matrix": {
                        k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in weight_analysis["B_matrix"].items()
                    },
                },
                "final_coordinates": coords,  # Store coordinates in results too
            }
        except (AttributeError, Exception):
            # Model doesn't support archetypal coordinates
            final_analysis = {"error": "Model does not support archetypal analysis"}

    # Print final summary
    print("\nFinal Performance:")
    for metric_name in ["loss", "archetypal_loss", "kld_loss", "reconstruction_loss", "archetype_r2"]:
        if metric_name in valid_history and len(valid_history[metric_name]) > 0:
            values = valid_history[metric_name]
            print(f"  {metric_name}: {values[-1]:.4f} (range: {min(values):.4f} - {max(values):.4f})")

    # Print archetype transform evolution summary
    print("\nArchetype Transform Summary:")
    transform_metrics = ["archetype_transform_mean", "archetype_transform_std", "archetype_transform_norm", "archetype_transform_grad_norm"]
    for metric_name in transform_metrics:
        if metric_name in valid_history and len(valid_history[metric_name]) > 0:
            values = valid_history[metric_name]
            print(f"  {metric_name}: {values[-1]:.6f} (range: {min(values):.6f} - {max(values):.6f})")

    # Check if archetype transform is learning
    if "archetype_transform_mean" in valid_history and len(valid_history["archetype_transform_mean"]) > 1:
        initial_mean = valid_history["archetype_transform_mean"][0]
        final_mean = valid_history["archetype_transform_mean"][-1]
        mean_change = abs(final_mean - initial_mean)
        print(f"  Transform mean change: {mean_change:.6f}")

        if mean_change > 0.01:
            print("  [OK] Archetype transform is learning!")
        else:
            print("  [WARNING] Archetype transform might not be learning enough")

    # Print final stability summary
    if track_stability:
        print("\nFinal Stability Metrics:")
        stability_names = ["archetype_drift_mean", "archetype_stability_mean", "archetype_variance_mean"]
        for metric_name in stability_names:
            if metric_name in valid_history and len(valid_history[metric_name]) > 0:
                values = valid_history[metric_name]
                print(f"  {metric_name}: {values[-1]:.6f} (range: {min(values):.6f} - {max(values):.6f})")

    if validate_constraints and final_analysis and "final_constraint_validation" in final_analysis:
        print("\nFinal Constraint Status:")
        final_constraints = final_analysis["final_constraint_validation"]
        print(f"  A matrix sum error: {final_constraints['A_sum_error']:.6f}")
        print(f"  B matrix sum error: {final_constraints['B_sum_error']:.6f}")
        print(f"  Constraints satisfied: {bool(final_constraints['constraints_satisfied'])}")

    # Create results dict with standard structure for compatibility
    results = {
        "history": valid_history,
        "final_model": model,  # PRESERVE existing key for compatibility
        "final_optimizer": optimizer,
        "final_analysis": final_analysis,
        "epoch_archetype_positions": epoch_archetype_positions,  # NEW: Full archetype evolution history
        "training_config": {
            "n_epochs": n_epochs,
            "actual_epochs": convergence_epoch,
            "early_stop_triggered": early_stop_triggered,
            "archetypal_weight": archetypal_weight,
            "kld_weight": kld_weight,
            "reconstruction_weight": reconstruction_weight,
            "activation_func": activation_func,
            "seed": seed,
            "constraint_tolerance": constraint_tolerance,
            "stability_history_size": stability_history_size,
            "early_stopping": early_stopping,
            "early_stopping_patience": early_stopping_patience if early_stopping else None,
            "early_stopping_metric": early_stopping_metric if early_stopping else None,
        },
    }

    # Also provide direct model access as an additional key
    results["model"] = model  # Alternative access method

    # Return both results dict AND model.pt separately for guaranteed model access
    return results, model


def save_checkpoint(state: dict[str, Any], path: str):
    """Save model checkpoint with all training state"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None) -> dict[str, Any]:
    """Load model checkpoint and restore training state"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Checkpoint loaded from {path}")
    return checkpoint

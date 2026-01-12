"""
Performance Optimization Utilities for HPC Environments

This module provides performance optimizations for training on HPC systems,
particularly those with older CPUs. All optimizations are optional and
backwards-compatible with sensible defaults.

"""

import os
from typing import Any

import torch
from torch.utils.data import DataLoader

# Module-level flag to track if threads have been optimized
_threads_optimized = False


def optimize_torch_threads(n_physical_cores: int | None = None, verbose: bool = True) -> int:
    """
    Configure PyTorch threading for optimal HPC performance.
    Safely handles cases where threads have already been configured.

    Args:
        n_physical_cores: Number of physical cores to use. If None, auto-detect.
        verbose: Print configuration details

    Returns
    -------
        Number of cores configured (or current setting if already configured)
    """
    global _threads_optimized

    # Skip if already optimized
    if _threads_optimized:
        current_threads = torch.get_num_threads()
        if verbose:
            print(f"ℹ  PyTorch threads already optimized: {current_threads} threads")
        return current_threads
    # Check if we're on Apple Silicon (unified memory architecture)
    is_apple_silicon = False
    try:
        import platform

        is_apple_silicon = platform.system() == "Darwin" and platform.machine() in ["arm64", "arm64e"]
    except:
        pass

    if is_apple_silicon:
        # Apple Silicon benefits from different threading strategy
        if verbose:
            print(" Apple Silicon detected - using default threading")
        return torch.get_num_threads()  # Return current setting

    # Auto-detect cores if not specified
    if n_physical_cores is None:
        # PRIORITY 1: Use SLURM allocation (most accurate)
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_JOB_CPUS_PER_NODE")
        if slurm_cpus:
            n_physical_cores = int(slurm_cpus)
            if verbose:
                print(f" SLURM allocation: {n_physical_cores} CPUs")
        # PRIORITY 2: Use PBS allocation
        elif os.environ.get("PBS_JOBID"):
            # PBS: try to get from environment or use nproc
            n_physical_cores = 4  # Safe PBS default
            if verbose:
                print(f" PBS detected: using {n_physical_cores} CPUs")
        # PRIORITY 3: Local machine
        else:
            n_cores = os.cpu_count()
            if n_cores:
                # Local: use half (assume hyperthreading)
                n_physical_cores = max(1, n_cores // 2)
            else:
                n_physical_cores = 4  # Fallback

    # Try to set threads, but handle if already configured
    try:
        torch.set_num_threads(n_physical_cores)
        # Only try to set interop threads if we successfully set num_threads
        try:
            torch.set_num_interop_threads(1)  # Avoid thread oversubscription
            interop_set = True
        except RuntimeError as e:
            if "interop" in str(e).lower():
                # Already configured, that's okay
                interop_set = False
            else:
                raise

        if verbose:
            if interop_set:
                print(f"[OK] PyTorch threads optimized: {n_physical_cores} threads (interop: 1)")
            else:
                print(f"[OK] PyTorch threads set: {n_physical_cores} (interop already configured)")
            print(f"   CPU count: {os.cpu_count()}, Configured threads: {n_physical_cores}")
    except RuntimeError as e:
        if "parallel" in str(e).lower() or "threads" in str(e).lower():
            # Threads already configured by PyTorch
            current_threads = torch.get_num_threads()
            if verbose:
                print(f"ℹ  PyTorch threads already configured: {current_threads} threads")
            return current_threads
        else:
            raise

    # Mark as optimized
    _threads_optimized = True
    return n_physical_cores


def create_optimized_dataloader(
    dataset,
    batch_size: int = 256,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    shuffle: bool = True,
    verbose: bool = False,
) -> DataLoader:
    """
    Create an optimized DataLoader for HPC and local environments.
    Automatically detects Apple Silicon and HPC environments.

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size (default: 256 for better throughput)
        num_workers: Number of worker processes. Auto-configured if None.
        pin_memory: Pin memory for CUDA. Auto-detected if None.
        persistent_workers: Keep workers alive between epochs. Auto if None.
        prefetch_factor: Number of batches to prefetch per worker.
        shuffle: Whether to shuffle data
        verbose: Print configuration details

    Returns
    -------
        Optimized DataLoader instance
    """
    # Detect environment
    is_apple_silicon = False
    is_hpc = False
    try:
        import platform

        is_apple_silicon = platform.system() == "Darwin" and platform.machine() in ["arm64", "arm64e"]
        is_hpc = any([os.environ.get("SLURM_JOB_ID"), os.environ.get("PBS_JOBID"), (os.cpu_count() or 1) > 16])
    except:
        pass

    # Smart defaults based on environment
    if num_workers is None:
        # PRIORITY 1: SLURM allocation
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_JOB_CPUS_PER_NODE")
        if slurm_cpus:
            allocated_cpus = int(slurm_cpus)
            # Small allocations: no workers (avoid contention)
            # Large allocations: conservative workers
            if allocated_cpus <= 4:
                num_workers = 0
            elif allocated_cpus <= 8:
                num_workers = 2
            else:
                num_workers = min(4, allocated_cpus // 4)
        elif is_apple_silicon:
            # Apple Silicon: unified memory
            num_workers = 0
        elif is_hpc:
            # HPC without SLURM info: no workers (safe)
            num_workers = 0
        else:
            # Local machine
            cpu_count = os.cpu_count()
            if cpu_count and cpu_count > 4:
                num_workers = min(2, cpu_count // 4)
            else:
                num_workers = 0

    if pin_memory is None:
        # Don't pin memory on Apple Silicon (unified memory)
        pin_memory = torch.cuda.is_available() and not is_apple_silicon

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    if prefetch_factor is None and num_workers > 0:
        # More aggressive prefetching on Apple Silicon due to fast memory
        prefetch_factor = 4 if is_apple_silicon else 2

    # Create DataLoader with optimized settings
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False,
    )

    if verbose:
        # Determine environment for display
        env_type = "Apple Silicon" if is_apple_silicon else ("HPC" if is_hpc else "Local")
        print(f"[OK] Optimized DataLoader created ({env_type}):")
        print(f"   Batch size: {batch_size}")
        print(f"   Workers: {num_workers}")
        print(f"   Pin memory: {pin_memory}")
        print(f"   Persistent workers: {persistent_workers}")
        if num_workers > 0:
            print(f"   Prefetch factor: {prefetch_factor}")

    return loader


def calculate_monitoring_frequency(
    n_epochs: int, target_checkpoints: int = 20, min_frequency: int = 1, max_frequency: int = 100
) -> int:
    """
    Calculate optimal monitoring frequency based on epoch count.

    Args:
        n_epochs: Total number of epochs
        target_checkpoints: Desired number of monitoring points
        min_frequency: Minimum monitoring frequency
        max_frequency: Maximum monitoring frequency

    Returns
    -------
        Monitoring frequency (monitor every N epochs)
    """
    if n_epochs <= target_checkpoints:
        return min_frequency

    frequency = max(min_frequency, n_epochs // target_checkpoints)
    frequency = min(frequency, max_frequency)

    return frequency


def should_monitor(
    epoch: int, n_epochs: int, monitor_frequency: int | None = None, always_monitor_last: bool = True
) -> bool:
    """
    Determine if monitoring should occur at this epoch.

    Args:
        epoch: Current epoch (0-indexed)
        n_epochs: Total number of epochs
        monitor_frequency: Monitor every N epochs. Auto if None.
        always_monitor_last: Always monitor the last epoch

    Returns
    -------
        True if monitoring should occur
    """
    if monitor_frequency is None:
        monitor_frequency = calculate_monitoring_frequency(n_epochs)

    # Always monitor first epoch
    if epoch == 0:
        return True

    # Always monitor last epoch if requested
    if always_monitor_last and epoch == n_epochs - 1:
        return True

    # Regular monitoring interval
    return epoch % monitor_frequency == 0


def get_training_config(
    n_epochs: int = 100,
    batch_size: int = 256,
    monitor_frequency: int | None = None,
    num_workers: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """
    Get optimized training configuration for current environment.

    Args:
        n_epochs: Number of training epochs
        batch_size: Training batch size
        monitor_frequency: Monitoring frequency (auto if None)
        num_workers: DataLoader workers (auto if None)
        device: Training device (auto if None)

    Returns
    -------
        Dictionary of optimized training parameters
    """
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Auto-configure monitoring
    if monitor_frequency is None:
        monitor_frequency = calculate_monitoring_frequency(n_epochs)

    # Auto-configure workers
    if num_workers is None:
        cpu_count = os.cpu_count()
        if cpu_count:
            num_workers = min(4, max(0, cpu_count // 4))
        else:
            num_workers = 0

    config = {
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "monitor_frequency": monitor_frequency,
        "num_workers": num_workers,
        "device": device,
        "pin_memory": device == "cuda",
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 2 if num_workers > 0 else None,
    }

    return config


def print_performance_config():
    """Print current performance configuration."""
    print("=" * 60)
    print("PERFORMANCE CONFIGURATION")
    print("=" * 60)

    # System info
    print(f"CPU count: {os.cpu_count()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # PyTorch threads
    print(f"\nPyTorch threads: {torch.get_num_threads()}")
    print(f"Interop threads: {torch.get_num_interop_threads()}")

    # Recommended settings
    config = get_training_config()
    print("\nRecommended settings:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("=" * 60)

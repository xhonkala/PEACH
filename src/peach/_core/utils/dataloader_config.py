"""
DataLoader configuration for optimized performance on different hardware.

This module provides smart defaults for DataLoader parameters based on the
execution environment (local vs HPC) and hardware capabilities.
"""

import os
from dataclasses import dataclass
from typing import Literal

import torch


def detect_environment() -> Literal["hpc", "local", "apple_silicon"]:
    """Detect execution environment for optimal DataLoader settings.

    Returns
    -------
        Environment type: 'hpc', 'local', or 'apple_silicon'
    """
    # Check for Apple Silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "apple_silicon"

    # Check for HPC environment indicators
    is_hpc = any(
        [
            os.environ.get("SLURM_JOB_ID"),
            os.environ.get("PBS_JOBID"),
            os.environ.get("LSF_JOBID"),
            os.environ.get("SGE_TASK_ID"),
            os.path.exists("/etc/slurm"),
            (os.cpu_count() or 1) > 16,  # Many cores usually indicates HPC
        ]
    )

    return "hpc" if is_hpc else "local"


def get_optimal_num_workers(environment: str | None = None) -> int:
    """Get optimal number of workers for DataLoader based on environment.

    Parameters
    ----------
    environment : str, optional
        Override environment detection ('hpc', 'local', 'apple_silicon')

    Returns
    -------
    int
        Optimal number of worker processes
    """
    if environment is None:
        environment = detect_environment()

    cpu_count = os.cpu_count() or 1

    if environment == "apple_silicon":
        # Apple Silicon has efficient unified memory, workers often slow it down
        return 0
    elif environment == "hpc":
        # HPC nodes benefit from multiple workers, leave some cores for system
        return min(8, max(4, cpu_count - 2))
    else:
        # Local machines: moderate number of workers
        return min(4, max(0, cpu_count - 2))


@dataclass
class DataLoaderConfig:
    """Configuration for optimized DataLoader creation.

    Parameters
    ----------
    batch_size : int, default: 128
        Number of samples per batch
    shuffle : bool, default: True
        Whether to shuffle data each epoch
    num_workers : int or 'auto', default: 'auto'
        Number of subprocesses for data loading.
        'auto' will detect optimal value based on environment
    pin_memory : bool or 'auto', default: 'auto'
        Use pinned memory for faster GPU transfer.
        'auto' will set True if CUDA is available
    persistent_workers : bool or 'auto', default: 'auto'
        Keep workers alive between epochs.
        'auto' will set True if num_workers > 0
    prefetch_factor : int, default: 2
        Number of batches loaded in advance by each worker
    drop_last : bool, default: False
        Drop the last incomplete batch if dataset size is not divisible by batch size
    """

    batch_size: int = 128
    shuffle: bool = True
    num_workers: int | None = None
    pin_memory: bool | None = None
    persistent_workers: bool | None = None
    prefetch_factor: int = 2
    drop_last: bool = False

    # Environment override
    environment: str | None = None

    def __post_init__(self):
        """Auto-configure optimal settings based on environment."""
        # Auto-detect number of workers
        if self.num_workers is None or self.num_workers == "auto":
            self.num_workers = get_optimal_num_workers(self.environment)

        # Auto-configure pin_memory for GPU
        if self.pin_memory is None or self.pin_memory == "auto":
            self.pin_memory = torch.cuda.is_available() and self.num_workers > 0

        # Auto-configure persistent_workers
        if self.persistent_workers is None or self.persistent_workers == "auto":
            self.persistent_workers = self.num_workers > 0

        # Validate settings
        if self.num_workers == 0:
            # Can't use persistent_workers without workers
            self.persistent_workers = False
            # Pin memory less useful without workers
            if not torch.cuda.is_available():
                self.pin_memory = False

    def to_dict(self) -> dict:
        """Convert to dictionary for DataLoader constructor."""
        return {
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": self.prefetch_factor if self.num_workers > 0 else None,
            "drop_last": self.drop_last,
        }

    def get_dataloader_kwargs(self) -> dict:
        """Get kwargs for DataLoader, filtering out None values."""
        kwargs = self.to_dict()
        # Remove None values and prefetch_factor if num_workers is 0
        if self.num_workers == 0:
            kwargs.pop("prefetch_factor", None)
            kwargs.pop("persistent_workers", None)
        return {k: v for k, v in kwargs.items() if v is not None}

    def summary(self) -> str:
        """Get human-readable summary of configuration."""
        env = self.environment or detect_environment()
        lines = [
            f"DataLoader Configuration ({env})",
            f"  Batch size: {self.batch_size}",
            f"  Workers: {self.num_workers}",
            f"  Pin memory: {self.pin_memory}",
            f"  Persistent: {self.persistent_workers}",
            f"  Prefetch: {self.prefetch_factor if self.num_workers > 0 else 'N/A'}",
        ]
        return "\n".join(lines)


# Preset configurations for common scenarios
PRESETS = {
    "apple_silicon": DataLoaderConfig(
        num_workers=0,  # Unified memory is fast
        pin_memory=False,
        persistent_workers=False,
    ),
    "hpc_gpu": DataLoaderConfig(num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=3),
    "hpc_cpu": DataLoaderConfig(num_workers=8, pin_memory=False, persistent_workers=True, prefetch_factor=2),
    "local_gpu": DataLoaderConfig(num_workers=4, pin_memory=True, persistent_workers=True),
    "local_cpu": DataLoaderConfig(num_workers=2, pin_memory=False, persistent_workers=True),
    "minimal": DataLoaderConfig(num_workers=0, pin_memory=False, persistent_workers=False),
}


def get_preset_config(preset_name: str, **overrides) -> DataLoaderConfig:
    """Get a preset configuration with optional overrides.

    Parameters
    ----------
    preset_name : str
        Name of preset ('apple_silicon', 'hpc_gpu', 'hpc_cpu', 'local_gpu', 'local_cpu', 'minimal')
    **overrides
        Override specific parameters

    Returns
    -------
    DataLoaderConfig
        Configuration object
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")

    config = PRESETS[preset_name]
    for key, value in overrides.items():
        setattr(config, key, value)

    return config

# load AnnData
import os

import anndata as ad
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

"""
AnnData Loading and PyTorch Integration
Utilities for loading single-cell data from AnnData format and creating PyTorch DataLoaders.

=== MODULE API INVENTORY ===

MAIN FUNCTIONS:
 load_anndata(path: str, use_raw: bool = True, dim_reduction_key: str = 'X_PCA', batch_size: int = 128) -> ad.AnnData
    Purpose: Load single-cell data from .h5ad file with flexible data source selection
    Inputs: path(str, .h5ad file path), use_raw(bool, use adata.X vs dimensionally reduced), dim_reduction_key(str, obsm key for reduced data), batch_size(int, backward compatibility)
    Outputs: ad.AnnData object with expression data in .X attribute
    Side Effects: Reads file from disk, modifies adata.X if using dimensionally reduced data, prints warnings for missing keys

 create_dataloader_from_anndata(adata: ad.AnnData, batch_size: int = 128, shuffle: bool = True, pca_key: Optional[str] = None) -> DataLoader
     Purpose: Convert AnnData PCA coordinates to PyTorch DataLoader for archetypal analysis training
     Inputs: adata(ad.AnnData with PCA in .obsm), batch_size(int), shuffle(bool), pca_key(Optional[str] for specific PCA key)
     Outputs: DataLoader with PCA coordinates (NOT raw gene expression)
     Side Effects: Auto-detects PCA from ['X_pca', 'X_PCA', 'PCA', 'x_pca'], converts to torch.FloatTensor, validation & error handling

DATA SOURCE OPTIONS:
 Raw Expression Data (use_raw=True): Uses adata.X directly for full gene expression matrix
    Best for: Archetypal analysis on complete expression profiles
    Data characteristics: High-dimensional, sparse, cell x gene matrix

 Dimensionally Reduced Data (use_raw=False): Uses adata.obsm[dim_reduction_key]
     Best for: Analysis on PCA, UMAP, or other reduced representations
     Data characteristics: Lower-dimensional, dense, typically 10-50 components

EXTERNAL DEPENDENCIES:
 From torch: FloatTensor conversion, tensor operations
 From torch.utils.data: TensorDataset, DataLoader for batch processing
 From anndata: AnnData format reading and manipulation
 From typing: Type hints for function signatures

DATA FLOW PATTERNS:
 Input: .h5ad file → AnnData loading → PCA coordinate detection → PCA matrix extraction
 Integration: PCA coordinates → Auto-detection → Tensor conversion → TensorDataset creation → DataLoader configuration
 Pipeline: File I/O → PCA processing → PyTorch integration → Archetypal training compatibility

CRITICAL DESIGN CHANGE:
 BEFORE: Used adata.X (raw gene expression) - INCORRECT for archetypal analysis
 AFTER: Uses adata.obsm[PCA_key] (PCA coordinates) - CORRECT for archetypal analysis
 RATIONALE: Archetypal analysis identifies extremal states in lower-dimensional PCA space
 VALIDATION: Auto-detects common PCA key variations, provides clear error messages

ERROR HANDLING:
 Missing PCA coordinates → Clear error with candidates and guidance
 File loading errors → Propagated from anndata.read_h5ad()
 Tensor conversion issues → Handled by torch.FloatTensor() with automatic dtype conversion
 DataLoader creation → Standard PyTorch error handling for batch size and data compatibility
 PCA key validation → Explicit error messages with available keys and suggestions
"""


def load_anndata(
    path: str, use_raw: bool = True, dim_reduction_key: str = "X_PCA", batch_size: int = 128
) -> ad.AnnData:
    """
    Load data from AnnData file.

    Args:
        path: path to .h5ad file
        use_raw: if True, return adata.X (raw expression), if False use dim_reduction_key
        dim_reduction_key: obsm dimensionality reduction key (used when use_raw=False)
        batch_size: DataLoader batch size (for backwards compatibility)

    Returns
    -------
        adata: AnnData object with expression data in .X
    """
    # Fix: use read_h5ad instead of head_h5ad
    adata = ad.read_h5ad(path)

    # For archetypal analysis, we typically want raw expression data
    if use_raw:
        # adata.X already contains the expression matrix we want
        pass
    else:
        # Use dimensionally reduced data if requested
        if dim_reduction_key in adata.obsm:
            adata.X = adata.obsm[dim_reduction_key]
        else:
            print(f"Warning: {dim_reduction_key} not found in adata.obsm, using adata.X")

    return adata


def create_dataloader_from_anndata(
    adata: ad.AnnData,
    batch_size: int = 128,
    shuffle: bool = True,
    pca_key: str | None = None,
    num_workers: int | str | None = "auto",
    pin_memory: bool | str | None = "auto",
    persistent_workers: bool | str | None = "auto",
    prefetch_factor: int = 2,
) -> DataLoader:
    """
    Create PyTorch DataLoader from AnnData object using PCA coordinates with HPC optimizations.

    CRITICAL: Archetypal analysis works on PCA coordinates, not raw gene expression.
    This function automatically detects and uses PCA coordinates from adata.obsm.

    Args:
        adata: AnnData object with PCA coordinates
        batch_size: batch size for DataLoader
        shuffle: whether to shuffle data
        pca_key: specific PCA key to use (if None, auto-detects)
        num_workers: number of worker processes ('auto' to detect optimal)
        pin_memory: use pinned memory for GPU ('auto' to detect)
        persistent_workers: keep workers alive between epochs ('auto' to detect)
        prefetch_factor: number of batches to prefetch per worker

    Returns
    -------
        dataloader: PyTorch DataLoader with PCA coordinates

    Raises
    ------
        ValueError: If no PCA coordinates found in adata.obsm
    """
    # AUTO-DETECT PCA COORDINATES
    # Check common PCA key variations in priority order
    pca_candidates = ["X_pca", "X_PCA", "PCA", "x_pca", "X_lsi"]

    if pca_key is not None:
        # User specified exact key
        if pca_key in adata.obsm:
            pca_data = adata.obsm[pca_key]
            print(f"[OK] Using specified PCA coordinates: adata.obsm['{pca_key}'] {pca_data.shape}")
        else:
            raise ValueError(
                f"Specified PCA key '{pca_key}' not found in adata.obsm. Available keys: {list(adata.obsm.keys())}"
            )
    else:
        # Auto-detect PCA coordinates
        pca_data = None
        found_key = None

        for candidate in pca_candidates:
            if candidate in adata.obsm:
                pca_data = adata.obsm[candidate]
                found_key = candidate
                print(f"[OK] Auto-detected PCA coordinates: adata.obsm['{candidate}'] {pca_data.shape}")
                break

        if pca_data is None:
            available_keys = list(adata.obsm.keys())
            raise ValueError(
                f"No PCA coordinates found in adata.obsm. "
                f"Expected one of: {pca_candidates}. "
                f"Available keys: {available_keys}. "
                f"Run PCA first: sc.pp.pca(adata), or for scATAC-seq: pc.pp.prepare_atacseq(adata), or specify pca_key parameter."
            )

    # Convert PCA coordinates to tensor
    # X = torch.FloatTensor(pca_data)
    # Ensure array is contiguous (fixes negative stride issue with some AnnData formats)
    pca_data_contiguous = np.ascontiguousarray(pca_data)
    X = torch.FloatTensor(pca_data_contiguous)
    dataset = TensorDataset(X)

    # Auto-configure DataLoader settings for HPC
    if num_workers == "auto":
        # PRIORITY 1: Check SLURM allocation (critical!)
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_JOB_CPUS_PER_NODE")
        if slurm_cpus:
            allocated_cpus = int(slurm_cpus)
            # Small allocations: no workers to avoid contention
            if allocated_cpus <= 4:
                num_workers = 0
            elif allocated_cpus <= 8:
                num_workers = 2
            else:
                num_workers = min(4, allocated_cpus // 4)
        # Apple Silicon - unified memory is fast
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            num_workers = 0
        # HPC without SLURM info (fallback)
        elif any([os.environ.get("SLURM_JOB_ID"), os.environ.get("PBS_JOBID")]):
            # In HPC but no CPU info - be conservative
            num_workers = 0
        # Local machine
        else:
            cpu_count = os.cpu_count() or 1
            if cpu_count > 4:
                num_workers = min(2, cpu_count // 4)
            else:
                num_workers = 0

    if pin_memory == "auto":
        pin_memory = torch.cuda.is_available() and num_workers > 0

    if persistent_workers == "auto":
        persistent_workers = num_workers > 0

    # Build DataLoader kwargs
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    # Add optional kwargs based on configuration
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        dataloader_kwargs["prefetch_factor"] = prefetch_factor

    dataloader = DataLoader(dataset, **dataloader_kwargs)

    # Report configuration
    env_type = (
        "HPC"
        if num_workers >= 4
        else ("Apple Silicon" if num_workers == 0 and hasattr(torch.backends, "mps") else "Local")
    )
    print(f"[STATS] DataLoader created: {len(dataset)} cells × {X.shape[1]} PCA components")
    print(f"   Config: batch_size={batch_size}, workers={num_workers} ({env_type})")
    if num_workers > 0:
        print(f"   Optimizations: pin_memory={pin_memory}, persistent={persistent_workers}")

    return dataloader

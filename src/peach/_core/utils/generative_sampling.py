"""
Generative Sampling for Archetypal Analysis
Functions for generating synthetic samples using trained archetypal models.

=== MODULE API INVENTORY ===

MAIN FUNCTIONS (MODULAR DESIGN):

SAMPLING LAYER:
 sample_pca_coordinates(adata, n_samples, strategy, random_seed) -> Dict
    Purpose: PURE SAMPLER - Generate random PCA coordinates within data bounds (no model)
    Inputs: adata with PCA, n_samples, sampling strategy ('uniform', 'normal', 'poisson')
    Outputs: Dict with 'pca_coordinates', 'sampling_bounds', 'sampling_info'

GENERATION LAYER:
 generate_synthetic_data(model, pca_coordinates, device) -> Dict
    Purpose: PURE GENERATOR - Transform PCA coords through archetypal model
    Inputs: trained model, PCA coordinates array, device
    Outputs: Dict with 'archetype_weights', 'reconstructed_pca', 'archetypal_r2', 'generation_metrics'

TRANSFORMATION LAYER:
 transform_to_gene_space(pca_coordinates, adata) -> Dict
    Purpose: STANDALONE INVERSE PCA - Transform PCA coords to gene expression
    Inputs: PCA coordinates array, adata with PCA components
    Outputs: Dict with 'gene_expression', 'transformation_metrics', 'transformation_info'

VALIDATION LAYER:
 validate_pca_distributions(generated_pca, real_pca_data, archetype_coords, comparison_type, nearest_percentile) -> Dict
    Purpose: MODULAR PCA VALIDATION - Compare PCA coordinate distributions and archetype distances
    Inputs: generated PCA coords, real PCA data, optional archetype coords, comparison type ('full'/'regional'), nearest neighbor settings
    Outputs: Dict with cosine similarities, distribution stats, archetype distance analysis, comparison info

 validate_gene_correlations(generated_genes, real_gene_data, gene_names, max_genes, max_samples, correlation_method) -> Dict
    Purpose: MODULAR GENE VALIDATION - Compare gene expression correlations and distributions
    Inputs: generated gene expression, real gene data, optional gene names, computational limits, correlation method ('spearman'/'pearson')
    Outputs: Dict with gene correlations, sample correlations, expression stats, correlation summary

 validate_archetypal_distributions(generated_weights, real_weights) -> Dict
    Purpose: SIMPLIFIED KS TESTS - Compare real vs generated archetype weight distributions
    Inputs: generated weights, real weights (from .obsm after training)
    Outputs: Dict with KS test results per archetype weight distribution

DATA MANAGEMENT LAYER:
 create_volumetric_holdout(adata, holdout_strategy, target_fraction, dimension_ranges) -> Tuple[AnnData, AnnData]
    Purpose: Create holdout region with improved cell count targeting
    Inputs: adata, holdout strategy, target fraction, dimension ranges
    Outputs: (training_adata, holdout_adata) with reasonable cell counts

 create_generated_samples_anndata(pca_coords, archetype_weights, gene_expr, source_adata_uns) -> AnnData
    Purpose: Create AnnData object from generated samples for visualization
    Inputs: generated coordinates, weights, expression data, source metadata (.uns)
    Outputs: AnnData with proper metadata transfer for multi-archetype visualization

DATA FLOW:
 PCA sampling: PCA bounds → Poisson sampling → PCA coordinates
 Model generation: PCA coords → model.encode() → reparameterize → model.decode() → reconstructed PCA
 Inverse transform: Generated PCA → adata.varm['PCs'] → Gene expression
 Validation: Generated vs Real → KS tests + Cosine similarities

CONSTRAINT HANDLING:
 Archetype weights: Flag violations but retain for analysis
 PCA bounds: Clip to input data range (no extrapolation)
 Convex hull: Natural constraint via model's softmax/normalization

VALIDATION METRICS:
 KS Tests: Joint distributions of archetype weights and distances
 Cosine Similarities: Cell-wise comparisons in PCA space
 Gene Correlations: Feature-wise and sample-wise correlations (Spearman/Pearson)
 Constraint Analysis: Violation rates and spatial distribution
"""

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.spatial.distance import cdist


def sample_pca_coordinates(
    adata: ad.AnnData,
    n_samples: int = 1000,
    pca_key: str = "X_pca",
    strategy: str = "uniform",
    random_seed: int | None = None,
    match_real_size: bool = False,
    hull_clip_factor: float = 0.8,  # NEW: Clip to 80% of max distance from centroid
    verbose: bool = True,
) -> dict:
    """
    PURE SAMPLER: Generate random PCA coordinates within data bounds with hull-aware clipping.

    CRITICAL FIX: Addresses the "rotating ball" problem where uniform sampling across mismatched
    PCA ranges creates coordinates outside the convex hull, degrading decode performance.

    Args:
        adata: AnnData object with PCA coordinates
        n_samples: Number of coordinate samples to generate
        pca_key: Key for PCA coordinates in adata.obsm
        strategy: Sampling strategy ('uniform', 'normal', 'poisson', 'dirichlet')
        random_seed: Random seed for reproducibility
        match_real_size: If True, override n_samples to match real data size for balanced comparison
        hull_clip_factor: Factor (0-1) to clip coordinates to fraction of max distance from centroid
        verbose: Print progress information

    Returns
    -------
        Dict containing:
        - 'pca_coordinates': Generated PCA coordinates [n_samples, n_components]
        - 'sampling_bounds': PCA space bounds used for sampling
        - 'sampling_info': Metadata about sampling process
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if verbose:
        print(f" Sampling {n_samples} PCA coordinates via {strategy} strategy...")

    # Get PCA coordinates
    pca_coords = None
    actual_pca_key = None
    for possible_key in [pca_key, "X_pca", "X_PCA", "PCA"]:
        if possible_key in adata.obsm:
            pca_coords = adata.obsm[possible_key]
            actual_pca_key = possible_key
            break

    if pca_coords is None:
        raise ValueError(f"No PCA coordinates found in adata.obsm. Available keys: {list(adata.obsm.keys())}")

    n_cells, n_components = pca_coords.shape

    # Optionally match real data size for balanced comparison
    if match_real_size:
        n_samples = n_cells
        if verbose:
            print(f"    Matching real data size: generating {n_samples} samples")

    # Compute PCA space bounds
    pca_min = pca_coords.min(axis=0)
    pca_max = pca_coords.max(axis=0)
    pca_range = pca_max - pca_min
    pca_center = pca_coords.mean(axis=0)
    pca_std = pca_coords.std(axis=0)

    if verbose:
        print(f"   [STATS] PCA space: {n_cells} cells × {n_components} components")
        print(f"    PCA bounds: [{pca_min.min():.3f}, {pca_max.max():.3f}]")

    # Calculate hull-aware bounds to prevent "rotating ball" problem
    # Compute max distance from centroid for each cell
    distances_from_center = np.linalg.norm(pca_coords - pca_center, axis=1)
    max_distance = np.max(distances_from_center)

    # Create conservative bounds: clip to hull_clip_factor of max distance
    effective_radius = max_distance * hull_clip_factor

    if verbose:
        print(f"    Hull-aware clipping: max distance {max_distance:.3f} → effective radius {effective_radius:.3f}")
        print(f"   [STATS] Clipping factor: {hull_clip_factor} (prevents convex hull violations)")

    # Generate samples based on strategy
    if strategy == "dirichlet":
        # OPTION D: REJECTION SAMPLING - Combine archetypal validity + spatial constraints
        if "archetype_coordinates" in adata.uns:
            if verbose:
                print("    Using spatial-constrained Dirichlet sampling (rejection method)")
            archetype_coords = adata.uns["archetype_coordinates"]

            # Check if we need spatial constraints (holdout info available)
            spatial_constraints = None
            if "holdout_info" in adata.uns and "volume_bounds" in adata.uns["holdout_info"]:
                spatial_constraints = adata.uns["holdout_info"]["volume_bounds"]
                if verbose:
                    print(f"    Found spatial constraints: {len(spatial_constraints)} dimensions")

            # Generate barycentric coordinates using Dirichlet distribution
            n_archetypes = archetype_coords.shape[0]
            n_components = archetype_coords.shape[1]
            alpha = np.ones(n_archetypes)  # Uniform preference

            # REJECTION SAMPLING LOOP
            sampled_coords = []
            barycentric_weights_list = []
            n_generated = 0
            n_rejected = 0
            max_attempts = n_samples * 10  # Safety limit

            if verbose and spatial_constraints:
                print(f"    Starting rejection sampling (target: {n_samples} valid samples)")

            while len(sampled_coords) < n_samples and (n_generated + n_rejected) < max_attempts:
                # Generate single archetypal coordinate
                barycentric_weight = np.random.dirichlet(alpha)
                candidate_coord = barycentric_weight @ archetype_coords

                # Check spatial constraints if available
                is_valid = True
                if spatial_constraints:
                    for pc_idx in range(n_components):
                        pc_key = f"PC{pc_idx}"
                        if pc_key in spatial_constraints:
                            bounds = spatial_constraints[pc_key]["bounds"]
                            if not (bounds[0] <= candidate_coord[pc_idx] <= bounds[1]):
                                is_valid = False
                                break

                if is_valid:
                    sampled_coords.append(candidate_coord)
                    barycentric_weights_list.append(barycentric_weight)
                    n_generated += 1
                else:
                    n_rejected += 1

            # Convert to arrays
            sampled_coords = np.array(sampled_coords)
            barycentric_weights = np.array(barycentric_weights_list) if barycentric_weights_list else None

            # Report results
            if verbose:
                if spatial_constraints:
                    rejection_rate = n_rejected / (n_generated + n_rejected) if (n_generated + n_rejected) > 0 else 0
                    print(f"   [OK] Rejection sampling complete: {n_generated} valid, {n_rejected} rejected")
                    print(f"   [STATS] Rejection rate: {rejection_rate * 100:.1f}%")
                    print(f"   [STATS] Efficiency: {(1 - rejection_rate) * 100:.1f}% (higher is better)")
                else:
                    print(f"   [OK] Generated {n_generated} archetypal coordinates (no spatial constraints)")

                if barycentric_weights is not None:
                    print(f"     Weight range: [{barycentric_weights.min():.4f}, {barycentric_weights.max():.4f}]")

            # Handle case where we couldn't generate enough samples
            if len(sampled_coords) < n_samples:
                if verbose:
                    print(f"   [WARNING]  Only generated {len(sampled_coords)}/{n_samples} samples within constraints")
                    print(
                        "   NOTE:  Consider: (1) increasing dimension ranges, (2) using fewer constraints, or (3) fallback sampling"
                    )
        else:
            if verbose:
                print("   [WARNING]  No archetype coordinates found, falling back to uniform sampling")
            strategy = "uniform"  # Fallback

    if strategy == "uniform":
        # Uniform sampling within bounds
        sampled_coords = np.random.uniform(low=pca_min, high=pca_max, size=(n_samples, n_components))

    elif strategy == "normal":
        # Normal sampling centered on data center
        sampled_coords = np.random.normal(
            loc=pca_center,
            scale=pca_std * 0.5,  # Conservative scaling
            size=(n_samples, n_components),
        )
        # Clip to bounds to avoid extrapolation
        sampled_coords = np.clip(sampled_coords, pca_min, pca_max)

    elif strategy == "poisson":
        # Exponential-like sampling (Poisson-inspired)
        sampled_coords = np.random.exponential(scale=pca_range / 4, size=(n_samples, n_components))
        sampled_coords = sampled_coords + pca_min
        # Clip to bounds
        sampled_coords = np.clip(sampled_coords, pca_min, pca_max)

    elif strategy != "dirichlet":  # Already handled above
        raise ValueError(f"Unknown sampling strategy: {strategy}. Use 'uniform', 'normal', 'poisson', or 'dirichlet'")

    # Apply hull-aware clipping ONLY for non-Dirichlet strategies
    if strategy != "dirichlet":
        # CRITICAL FIX: Apply hull-aware clipping to prevent convex hull violations
        # Check distances from center and clip points that are too far out
        sample_distances = np.linalg.norm(sampled_coords - pca_center, axis=1)
        outside_hull_mask = sample_distances > effective_radius
        n_outside = outside_hull_mask.sum()

        if n_outside > 0:
            # Clip outlying points to the effective radius boundary
            for i in np.where(outside_hull_mask)[0]:
                direction = sampled_coords[i] - pca_center
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 0:  # Avoid division by zero
                    sampled_coords[i] = pca_center + direction * (effective_radius / direction_norm)

            if verbose:
                print(f"    Hull clipping: {n_outside}/{n_samples} points clipped to hull boundary")
    else:
        if verbose:
            print("   [OK] Dirichlet sampling: No hull clipping needed (guaranteed within convex hull)")

    # Final validation of coordinates
    final_distances = np.linalg.norm(sampled_coords - pca_center, axis=1)
    max_final_distance = np.max(final_distances)

    if verbose:
        print(f"   [OK] Generated {strategy} samples")
        print(f"   [STATS] Sample range: [{sampled_coords.min():.3f}, {sampled_coords.max():.3f}]")
        print(f"    Max distance from center: {max_final_distance:.3f} (target ≤ {effective_radius:.3f})")

    # Compile results
    results = {
        "pca_coordinates": sampled_coords,
        "sampling_bounds": {"min": pca_min, "max": pca_max, "center": pca_center, "std": pca_std, "range": pca_range},
        "sampling_info": {
            "n_samples": n_samples,
            "n_components": n_components,
            "strategy": strategy,
            "pca_key_used": actual_pca_key,
            "random_seed": random_seed,
            "source_data_shape": pca_coords.shape,
        },
    }

    if verbose:
        print(f"   [OK] Sampling complete: {n_samples} PCA coordinates ready for generation")

    return results


def sample_archetypal_specialists(
    archetype_coordinates: np.ndarray,
    archetype_index: int,
    n_samples: int = 100,
    specialist_radius: float = 0.1,
    random_seed: int | None = None,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """
    Generate PCA coordinates for pure archetypal specialists near a specific archetype vertex.

    This function samples coordinates in a small sphere around an archetype to generate
    what pure specialist cells would look like - addressing the biological reality that
    most real cells are mixtures and don't reach the pure archetypal state.

    Args:
        archetype_coordinates: Archetype positions in PCA space [n_archetypes, n_components]
        archetype_index: Which archetype to generate specialists for (0-indexed)
        n_samples: Number of specialist cells to generate
        specialist_radius: Radius around archetype vertex (distance in PCA space)
        random_seed: Random seed for reproducibility
        verbose: Print progress information

    Returns
    -------
        Dict containing:
        - 'pca_coordinates': Generated specialist coordinates [n_samples, n_components]
        - 'target_archetype': Index of target archetype
        - 'specialist_radius': Radius used for generation
        - 'distances_to_archetype': Distance from each sample to target archetype

    Example:
        >>> # Generate 100 pure specialists for archetype 0
        >>> specialists = sample_archetypal_specialists(
        ...     archetype_coordinates=model_archetypes, archetype_index=0, n_samples=100, specialist_radius=0.05
        ... )
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n_archetypes, n_components = archetype_coordinates.shape

    if archetype_index >= n_archetypes or archetype_index < 0:
        raise ValueError(f"archetype_index {archetype_index} out of range [0, {n_archetypes - 1}]")

    target_archetype = archetype_coordinates[archetype_index]

    if verbose:
        print(f" Generating {n_samples} archetypal specialists for archetype {archetype_index}")
        print(f"    Target archetype position: {target_archetype}")
        print(f"    Specialist radius: {specialist_radius}")

    # Generate random points in a sphere around the archetype
    # Use normal distribution, then normalize and scale to get uniform distribution in sphere
    specialists = []
    distances = []

    for _ in range(n_samples):
        # Generate random direction (normal distribution, then normalize)
        direction = np.random.normal(0, 1, n_components)
        direction = direction / np.linalg.norm(direction)

        # Generate random radius (uniform in sphere volume)
        # For uniform distribution in n-dimensional sphere: r = R * (random^(1/n))
        radius = specialist_radius * (np.random.random() ** (1.0 / n_components))

        # Generate specialist coordinate
        specialist_coord = target_archetype + radius * direction
        specialists.append(specialist_coord)
        distances.append(radius)

    specialists = np.array(specialists)
    distances = np.array(distances)

    if verbose:
        print(f"   [OK] Generated {n_samples} specialists")
        print(f"   [STATS] Distance range: [{distances.min():.4f}, {distances.max():.4f}]")
        print(f"   [STATS] Mean distance: {distances.mean():.4f} ± {distances.std():.4f}")
        print(f"    All specialists within radius {specialist_radius}")

    return {
        "pca_coordinates": specialists,
        "target_archetype": archetype_index,
        "specialist_radius": specialist_radius,
        "distances_to_archetype": distances,
        "archetype_center": target_archetype,
    }


def sample_within_convex_hull(
    archetype_coordinates: np.ndarray, n_samples: int = 1000, random_seed: int | None = None, verbose: bool = True
) -> dict[str, np.ndarray]:
    """
    Generate PCA coordinates guaranteed to be within the convex hull of archetypes.

    This addresses the issue where rectangular sampling creates coordinates outside
    the valid cellular state space defined by the archetypal convex hull.

    Args:
        archetype_coordinates: Archetype positions [n_archetypes, n_components]
        n_samples: Number of samples to generate
        random_seed: Random seed for reproducibility
        verbose: Print progress information

    Returns
    -------
        Dict containing:
        - 'pca_coordinates': Generated coordinates [n_samples, n_components]
        - 'barycentric_weights': Convex weights used [n_samples, n_archetypes]
        - 'generation_method': 'convex_hull_sampling'

    Note:
        Generated coordinates are guaranteed to be within the convex hull by construction.
        This uses Dirichlet distribution to generate valid barycentric coordinates.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n_archetypes, n_components = archetype_coordinates.shape

    if verbose:
        print(f" Generating {n_samples} coordinates within archetypal convex hull")
        print(f"    Using {n_archetypes} archetypes in {n_components}D PCA space")

    # Generate random barycentric coordinates using Dirichlet distribution
    # This ensures all weights are non-negative and sum to 1 (valid convex combinations)
    alpha = np.ones(n_archetypes)  # Uniform Dirichlet (equal preference for all archetypes)
    barycentric_weights = np.random.dirichlet(alpha, size=n_samples)

    # Generate coordinates as convex combinations of archetypes
    coordinates = barycentric_weights @ archetype_coordinates

    if verbose:
        print(f"   [OK] Generated {n_samples} valid coordinates")
        print(f"   [STATS] Coordinate range: [{coordinates.min():.3f}, {coordinates.max():.3f}]")
        print(f"     Weight range: [{barycentric_weights.min():.4f}, {barycentric_weights.max():.4f}]")
        print("    All coordinates guaranteed within convex hull")

        # Validate convex combination (sanity check)
        weight_sums = barycentric_weights.sum(axis=1)
        print(f"   [OK] Weight sum validation: [{weight_sums.min():.6f}, {weight_sums.max():.6f}] (should be 1.0)")

    return {
        "pca_coordinates": coordinates,
        "barycentric_weights": barycentric_weights,
        "generation_method": "convex_hull_sampling",
        "archetype_coordinates": archetype_coordinates,
    }


def generate_synthetic_data(model, pca_coordinates: np.ndarray, device: str = "cpu", verbose: bool = True) -> dict:
    """
    PURE GENERATOR: Transform PCA coordinates through archetypal model.

    Args:
        model: Trained archetypal model (Deep_AA or similar)
        pca_coordinates: PCA coordinates to transform [n_samples, n_components]
        device: PyTorch device for computation
        verbose: Print progress information

    Returns
    -------
        Dict containing:
        - 'archetype_weights': Barycentric coordinates/archetypal weights [n_samples, n_archetypes]
        - 'reconstructed_pca': Model reconstruction [n_samples, n_components]
        - 'archetypal_loss': Frobenius loss (for optimization comparison with training)
        - 'archetypal_r2': R² score (for reconstruction quality assessment)
        - 'constraint_violations': Analysis of constraint satisfaction
        - 'generation_metrics': Both loss metrics plus usage statistics

    Note:
        - Model training state is preserved (eval mode used temporarily)
        - Provides BOTH Frobenius loss and R² for complete validation
        - Matches Deep_AA.py metric structure (after Option A implementation)
        - 'archetype_weights' are barycentric coordinates satisfying convexity constraints
    """
    if verbose:
        print(f"🧠 Generating archetypal samples from {pca_coordinates.shape[0]} PCA coordinates...")

    n_samples, n_components = pca_coordinates.shape

    # Verify model compatibility
    if hasattr(model, "input_dim"):
        expected_dims = model.input_dim
        if n_components != expected_dims:
            raise ValueError(
                f"PCA dimension mismatch! Model expects {expected_dims} components "
                f"but provided coordinates have {n_components} components."
            )

    # Convert to tensor and process through model
    # Store original training state to restore later
    was_training = model.training
    model.eval()  # Set to eval mode for inference

    try:
        pca_tensor = torch.FloatTensor(pca_coordinates).to(device)

        with torch.no_grad():
            # Encode PCA coordinates to archetypal parameter space
            mu, log_var = model.encode(pca_tensor)

            # Reparameterize to get barycentric coordinates (archetypal weights)
            barycentric_coords = model.reparameterize(mu, log_var)

            # Decode back to PCA space
            reconstructed_pca = model.decode(barycentric_coords)

            # Convert back to numpy
            barycentric_coords_np = barycentric_coords.cpu().numpy()
            reconstructed_pca_np = reconstructed_pca.cpu().numpy()
    finally:
        # Restore original training state
        model.train(was_training)

    if verbose:
        print("   [STATS] Model processing complete")
        print(f"   [STATS] Barycentric coordinates (archetype weights): {barycentric_coords_np.shape}")
        print(f"   [STATS] Reconstructed PCA: {reconstructed_pca_np.shape}")

    # Calculate BOTH metrics to match Deep_AA.py
    from .metrics import calculate_archetype_r2

    pca_tensor_orig = torch.FloatTensor(pca_coordinates).to(device)
    reconstructed_tensor = torch.FloatTensor(reconstructed_pca_np).to(device)

    # 1. Frobenius loss (for optimization comparison)
    archetypal_loss = (
        torch.norm(reconstructed_tensor - pca_tensor_orig, p="fro") ** 2 / pca_tensor_orig.numel()
    ).item()

    # 2. R² (for reconstruction quality assessment)
    archetypal_r2 = calculate_archetype_r2(reconstructed_tensor, pca_tensor_orig).item()

    # Analyze constraint violations
    constraint_violations = analyze_archetype_constraints(barycentric_coords_np, verbose=verbose)

    # Calculate additional metrics using STANDARD terms from Deep_AA.py
    # RMSE calculation (standard metric)
    rmse = np.sqrt(np.mean((pca_coordinates - reconstructed_pca_np) ** 2))

    # Archetype usage statistics (for analysis only)
    mean_weights = barycentric_coords_np.mean(axis=0)
    weight_entropy = -np.sum(mean_weights * np.log(mean_weights + 1e-10))  # Shannon entropy

    generation_metrics = {
        "archetypal_loss": archetypal_loss,  # Frobenius loss (for optimization comparison)
        "archetypal_r2": archetypal_r2,  # R² (for reconstruction quality)
        "rmse": rmse,  # Standard RMSE metric
        "archetype_usage_entropy": weight_entropy,
        "mean_archetype_weights": mean_weights,
        "n_effective_archetypes": np.sum(mean_weights > 0.01),  # Archetypes with >1% average usage
    }

    if verbose:
        print("   [STATS] Generation Quality Metrics:")
        print(f"     - Archetypal Loss (Frobenius): {archetypal_loss:.3f}")
        print(f"     - Archetypal R²: {archetypal_r2:.3f}")
        print(f"     - RMSE: {rmse:.3f}")
        print(f"     - Effective archetypes: {generation_metrics['n_effective_archetypes']}")
        print(f"     - Usage entropy: {weight_entropy:.3f}")

    # Compile results
    results = {
        "archetype_weights": barycentric_coords_np,  # These are barycentric coordinates/weights
        "reconstructed_pca": reconstructed_pca_np,
        "archetypal_loss": archetypal_loss,  # Frobenius loss (for optimization comparison)
        "archetypal_r2": archetypal_r2,  # R² (for reconstruction quality)
        "constraint_violations": constraint_violations,
        "generation_metrics": generation_metrics,
        "generation_info": {
            "model_type": type(model).__name__,
            "n_samples": n_samples,
            "n_components": n_components,
            "n_archetypes": getattr(model, "n_archetypes", None),
            "device": device,
        },
    }

    if verbose:
        print("   [OK] Archetypal generation complete")

    return results


def create_volumetric_holdout(
    adata: ad.AnnData,
    pca_key: str = "X_pca",
    holdout_strategy: str = "percentage",
    target_holdout_fraction: float = 0.15,
    dimension_ranges: dict[int, float] = None,
    center_bias: bool = True,
    verbose: bool = True,
) -> tuple[ad.AnnData, ad.AnnData]:
    """
    Create volumetric holdout by specifying % ranges for each PCA dimension.

    Args:
        adata: AnnData object with PCA coordinates
        pca_key: Key for PCA coordinates in adata.obsm
        holdout_strategy: Strategy for holdout creation ('percentage', 'geometric')
        target_holdout_fraction: Target fraction of cells for holdout (e.g., 0.15 = 15%)
        dimension_ranges: Dict mapping dimension index to % range (0.0-1.0)
                         e.g., {0: 1.0, 1: 1.0, 2: 0.3} = full range on PC1&2, 30% on PC3+
        center_bias: Whether to center ranges around data center
        verbose: Print progress information

    Returns
    -------
        Tuple of (training_adata, holdout_adata) where holdout contains cells in specified volume
    """
    if dimension_ranges is None:
        # Default: full range on first 2 PCs, 30% range on remaining for bigger holdout
        dimension_ranges = {0: 1.0, 1: 1.0}  # Will default remaining to 0.3

    if verbose:
        print(f" Creating volumetric holdout with dimension ranges: {dimension_ranges}")

    # Get PCA coordinates
    pca_coords = None
    actual_pca_key = None
    for possible_key in [pca_key, "X_pca", "X_PCA", "PCA"]:
        if possible_key in adata.obsm:
            pca_coords = adata.obsm[possible_key]
            actual_pca_key = possible_key
            break

    if pca_coords is None:
        raise ValueError(f"No PCA coordinates found in adata.obsm. Available keys: {list(adata.obsm.keys())}")

    n_cells, n_components = pca_coords.shape

    if verbose:
        print(f"   [STATS] PCA data: {n_cells} cells × {n_components} components")

    # Compute bounds for each dimension
    pca_min = pca_coords.min(axis=0)
    pca_max = pca_coords.max(axis=0)
    pca_center = (pca_min + pca_max) / 2
    pca_range = pca_max - pca_min

    # Create mask for cells in holdout volume
    holdout_mask = np.ones(n_cells, dtype=bool)

    for dim_idx in range(n_components):
        # Get range percentage for this dimension
        range_pct = dimension_ranges.get(dim_idx, 0.3)  # Default 30% for unspecified (bigger holdout)

        # Calculate bounds for this dimension
        half_range = (pca_range[dim_idx] * range_pct) / 2
        dim_min = pca_center[dim_idx] - half_range
        dim_max = pca_center[dim_idx] + half_range

        # Update mask: cells must be within bounds for ALL dimensions
        dim_mask = (pca_coords[:, dim_idx] >= dim_min) & (pca_coords[:, dim_idx] <= dim_max)
        holdout_mask = holdout_mask & dim_mask

        if verbose:
            range_actual = pca_range[dim_idx]
            range_holdout = dim_max - dim_min
            cells_in_range = dim_mask.sum()
            print(
                f"    PC{dim_idx}: {range_pct * 100:.1f}% range = [{dim_min:.3f}, {dim_max:.3f}] "
                f"({range_holdout:.3f}/{range_actual:.3f}), {cells_in_range} cells"
            )

    n_holdout = holdout_mask.sum()
    n_training = n_cells - n_holdout

    if verbose:
        print("   [OK] Volumetric split complete:")
        print(f"     - Training: {n_training} cells ({n_training / n_cells * 100:.1f}%)")
        print(f"     - Holdout: {n_holdout} cells ({n_holdout / n_cells * 100:.1f}%)")

    if n_holdout == 0:
        warnings.warn(
            "No cells found in holdout volume. The intersection of all dimension constraints is empty. "
            "Consider: (1) increasing dimension ranges, (2) using fewer constrained dimensions, "
            "or (3) using random sampling instead of geometric intersection."
        )
        # Return copies with empty holdout
        return adata.copy(), adata[[]].copy()

    if n_training == 0:
        warnings.warn("No cells found for training. Consider decreasing dimension ranges.")
        # Return empty training with full holdout
        return adata[[]].copy(), adata.copy()

    # Create training and holdout AnnData objects
    training_adata = adata[~holdout_mask].copy()
    holdout_adata = adata[holdout_mask].copy()

    # Store holdout metadata
    holdout_adata.uns["holdout_info"] = {
        "dimension_ranges": dimension_ranges,
        "pca_key_used": actual_pca_key,
        "n_components": n_components,
        "volume_bounds": {
            f"PC{i}": {
                "range_pct": dimension_ranges.get(i, 0.3),
                "bounds": [
                    pca_center[i] - (pca_range[i] * dimension_ranges.get(i, 0.3)) / 2,
                    pca_center[i] + (pca_range[i] * dimension_ranges.get(i, 0.3)) / 2,
                ],
            }
            for i in range(n_components)
        },
    }

    training_adata.uns["training_info"] = {
        "holdout_created": True,
        "holdout_dimensions": dimension_ranges,
        "n_holdout_cells": n_holdout,
    }

    if verbose:
        print(f"    Created training AnnData: {training_adata.n_obs} cells")
        print(f"    Created holdout AnnData: {holdout_adata.n_obs} cells")

    return training_adata, holdout_adata


def transform_to_gene_space(
    pca_coordinates: np.ndarray,
    adata: ad.AnnData,
    use_logcounts: bool = True,
    clip_negative: bool = True,
    sparsity_method: str = "adaptive_per_gene",
    verbose: bool = True,
) -> dict:
    """
    STANDALONE INVERSE PCA: Transform PCA coordinates back to gene expression space.

    Args:
        pca_coordinates: PCA coordinates to transform [n_samples, n_components]
        adata: AnnData object with PCA components in .varm['PCs']
        use_logcounts: Use logcounts scale for consistent comparison (default: True)
        clip_negative: Clip negative values to ensure biological validity (default: True)
        sparsity_method: Method for sparsity preservation ('none', 'quantile', 'adaptive_per_gene')
        verbose: Print progress information

    Returns
    -------
        Dict containing:
        - 'gene_expression': Gene expression matrix [n_samples, n_genes] or None
        - 'transformation_metrics': Quality metrics for the transformation
        - 'transformation_info': Metadata about the transformation process
    """
    if verbose:
        print(f" Transforming {pca_coordinates.shape[0]} samples to gene expression space...")

    n_samples, n_input_components = pca_coordinates.shape

    # Check for PCA components
    if not hasattr(adata, "varm") or "PCs" not in adata.varm:
        if verbose:
            print("   [ERROR] No PCA components found in adata.varm['PCs']")
        return {
            "gene_expression": None,
            "transformation_metrics": None,
            "transformation_info": {
                "success": False,
                "error": "No PCA components available",
                "components_available": False,
            },
        }

    pca_components = adata.varm["PCs"]  # [n_genes, n_components]
    n_genes, n_stored_components = pca_components.shape

    if verbose:
        print(f"   [STATS] PCA components: {n_genes} genes × {n_stored_components} components")
        print(f"   [STATS] Input coordinates: {n_samples} samples × {n_input_components} components")

    # Handle dimension mismatch
    coords_for_transform = pca_coordinates.copy()
    dimension_handling = "exact_match"

    if n_input_components > n_stored_components:
        if verbose:
            print(f"   [WARNING]  Truncating input from {n_input_components} to {n_stored_components} components")
        coords_for_transform = coords_for_transform[:, :n_stored_components]
        dimension_handling = "truncated"
    elif n_input_components < n_stored_components:
        if verbose:
            print(f"   [WARNING]  Padding input from {n_input_components} to {n_stored_components} components (zeros)")
        padding = np.zeros((n_samples, n_stored_components - n_input_components))
        coords_for_transform = np.hstack([coords_for_transform, padding])
        dimension_handling = "zero_padded"

    # Get gene expression mean (subtracted during PCA computation)
    pca_mean_source = "computed"
    if "pca" in adata.uns and "mean" in adata.uns["pca"]:
        # Use stored gene expression mean from scanpy PCA computation
        pca_mean = adata.uns["pca"]["mean"]
        pca_mean_source = "stored_pca"
        if verbose:
            print("   [OK] Using stored gene expression mean from adata.uns['pca']['mean']")
    elif hasattr(adata, "varm") and "PCs_mean" in adata.varm:
        # Alternative storage location for gene expression mean
        pca_mean = adata.varm["PCs_mean"]
        pca_mean_source = "stored_varm"
        if verbose:
            print("   [OK] Using stored gene expression mean from adata.varm['PCs_mean']")
    else:
        # Fallback: compute gene expression mean from appropriate layer
        if use_logcounts and "logcounts" in adata.layers:
            gene_data = adata.layers["logcounts"]
            pca_mean_source = "computed_logcounts"
            if verbose:
                print("   [OK] Computing gene expression mean from adata.layers['logcounts']")
        else:
            gene_data = adata.X
            pca_mean_source = "computed_X"
            if verbose:
                print("   [WARNING]  Computing gene expression mean from adata.X")
                print("   [WARNING]  Consider using stored PCA mean for consistency")

        if hasattr(gene_data, "toarray"):  # Handle sparse matrices
            gene_data = gene_data.toarray()
        pca_mean = gene_data.mean(axis=0)

    # Inverse PCA transformation: PCA_coords @ PCs.T + mean
    gene_expression = coords_for_transform @ pca_components.T + pca_mean

    # Apply biological constraints
    original_range = {"min": float(gene_expression.min()), "max": float(gene_expression.max())}

    if clip_negative:
        # Clip negative values to ensure biological validity
        gene_expression = np.maximum(gene_expression, 0.0)
        if verbose and original_range["min"] < 0:
            print(f"    Clipped {(gene_expression == 0).sum()} negative values to zero")

    # CRITICAL FIX: Apply sparsity preservation using real data distribution
    sparsity_info = {}
    if sparsity_method != "none":
        if verbose:
            print(f"    Applying sparsity preservation: {sparsity_method}")

        # Get original gene data for reference
        if use_logcounts and "logcounts" in adata.layers:
            original_genes = adata.layers["logcounts"]
        else:
            original_genes = adata.X

        if hasattr(original_genes, "toarray"):
            original_genes = original_genes.toarray()

        original_sparsity = (original_genes == 0).mean()
        raw_sparsity = (gene_expression == 0).mean()

        if sparsity_method == "quantile":
            # Simple quantile matching to original sparsity level
            threshold_quantile = np.percentile(gene_expression.flatten(), original_sparsity * 100)
            gene_expression[gene_expression <= threshold_quantile] = 0.0

        elif sparsity_method == "adaptive_per_gene":
            # Per-gene adaptive thresholding using real data distribution
            for gene_idx in range(n_genes):
                original_gene = original_genes[:, gene_idx]
                reconstructed_gene = gene_expression[:, gene_idx]

                # Calculate gene-specific statistics
                original_nonzero = original_gene[original_gene > 0]
                original_gene_sparsity = (original_gene == 0).mean()

                if len(original_nonzero) > 0 and original_gene_sparsity > 0.1:  # Only for sparse genes
                    # Strategy 1: Use percentile of non-zero values as threshold
                    gene_threshold = np.percentile(original_nonzero, 20)  # 20th percentile

                    # Strategy 2: Adjust threshold based on original sparsity level
                    # More sparse genes get higher thresholds
                    sparsity_factor = min(2.0, original_gene_sparsity * 3)  # Scale threshold
                    adjusted_threshold = gene_threshold * sparsity_factor

                    # Apply threshold
                    gene_expression[reconstructed_gene < adjusted_threshold, gene_idx] = 0.0

        preserved_sparsity = (gene_expression == 0).mean()
        sparsity_info = {
            "method": sparsity_method,
            "original_sparsity": float(original_sparsity),
            "raw_sparsity": float(raw_sparsity),
            "preserved_sparsity": float(preserved_sparsity),
            "sparsity_recovery_factor": float(preserved_sparsity / raw_sparsity) if raw_sparsity > 0 else 1.0,
        }

        if verbose:
            print(f"     [STATS] Original sparsity: {original_sparsity * 100:.1f}%")
            print(f"      Raw reconstruction: {raw_sparsity * 100:.1f}%")
            print(f"      Preserved sparsity: {preserved_sparsity * 100:.1f}%")
            print(f"     [STATS] Recovery factor: {sparsity_info['sparsity_recovery_factor']:.1f}x")

    # Scale consistency check
    scale_info = {
        "use_logcounts": use_logcounts,
        "pca_mean_source": pca_mean_source,
        "clip_negative": clip_negative,
        "sparsity_method": sparsity_method,
        "original_range": original_range,
        "final_range": {"min": float(gene_expression.min()), "max": float(gene_expression.max())},
    }

    # Calculate transformation quality metrics
    transformation_metrics = {
        "n_genes_reconstructed": n_genes,
        "expression_range": {
            "min": float(gene_expression.min()),
            "max": float(gene_expression.max()),
            "mean": float(gene_expression.mean()),
            "std": float(gene_expression.std()),
        },
        "component_usage": {
            "input_components": n_input_components,
            "stored_components": n_stored_components,
            "components_used": min(n_input_components, n_stored_components),
            "dimension_handling": dimension_handling,
        },
        "pca_mean_source": pca_mean_source,
        "scale_info": scale_info,
        "sparsity_info": sparsity_info,
    }

    if verbose:
        print(f"   [OK] Gene expression reconstructed: {gene_expression.shape}")
        print(f"   [STATS] Expression range: [{gene_expression.min():.3f}, {gene_expression.max():.3f}]")
        print(f"   [STATS] Mean expression: {gene_expression.mean():.3f} ± {gene_expression.std():.3f}")

    return {
        "gene_expression": gene_expression,
        "transformation_metrics": transformation_metrics,
        "transformation_info": {
            "success": True,
            "n_samples": n_samples,
            "n_genes": n_genes,
            "components_available": True,
            "dimension_handling": dimension_handling,
            "pca_mean_source": pca_mean_source,
        },
    }


def analyze_archetype_constraints(archetype_weights: np.ndarray, tolerance: float = 1e-5, verbose: bool = True) -> dict:
    """
    Analyze constraint satisfaction for archetype weights.

    Args:
        archetype_weights: Archetypal (barycentric) coordinates [n_samples, n_archetypes]
        tolerance: Numerical tolerance for constraint checking
        verbose: Print analysis results

    Returns
    -------
        Dict with constraint analysis:
        - 'non_negativity_violations': Mask and count of negative weights
        - 'sum_to_one_violations': Mask and count of non-unit sums
        - 'violation_rate': Overall violation rate
        - 'weight_statistics': Basic statistics
    """
    n_samples, n_archetypes = archetype_weights.shape

    # Check non-negativity constraint
    non_neg_violations = archetype_weights < -tolerance
    non_neg_violation_rate = (non_neg_violations.any(axis=1)).mean()

    # Check sum-to-one constraint
    row_sums = archetype_weights.sum(axis=1)
    sum_violations = np.abs(row_sums - 1.0) > tolerance
    sum_violation_rate = sum_violations.mean()

    # Overall violation rate
    any_violation = non_neg_violations.any(axis=1) | sum_violations
    overall_violation_rate = any_violation.mean()

    # Weight statistics
    weight_stats = {
        "mean": archetype_weights.mean(),
        "std": archetype_weights.std(),
        "min": archetype_weights.min(),
        "max": archetype_weights.max(),
        "mean_row_sum": row_sums.mean(),
        "std_row_sum": row_sums.std(),
    }

    results = {
        "non_negativity_violations": {
            "mask": non_neg_violations,
            "count": non_neg_violations.sum(),
            "rate": non_neg_violation_rate,
        },
        "sum_to_one_violations": {"mask": sum_violations, "count": sum_violations.sum(), "rate": sum_violation_rate},
        "violation_rate": overall_violation_rate,
        "weight_statistics": weight_stats,
        "analysis_info": {"n_samples": n_samples, "n_archetypes": n_archetypes, "tolerance": tolerance},
    }

    if verbose:
        print("     Constraint Analysis:")
        print(f"     - Non-negativity violations: {non_neg_violation_rate * 100:.1f}% of samples")
        print(f"     - Sum-to-one violations: {sum_violation_rate * 100:.1f}% of samples")
        print(f"     - Overall violation rate: {overall_violation_rate * 100:.1f}%")
        print(f"     - Weight range: [{archetype_weights.min():.3f}, {archetype_weights.max():.3f}]")
        print(f"     - Row sum range: [{row_sums.min():.3f}, {row_sums.max():.3f}]")

    return results


def validate_pca_distributions(
    generated_pca: np.ndarray,
    real_pca_data: np.ndarray,
    archetype_coords: np.ndarray = None,
    comparison_type: str = "full",
    nearest_percentile: float = 0.1,
    verbose: bool = True,
) -> dict:
    """
    MODULAR PCA VALIDATION: Compare PCA coordinate distributions between generated and real data.

    This function performs three key analyses:
    1. Cosine similarity between generated and nearest real cells
    2. KS tests comparing per-PC component distributions
    3. Archetype distance distribution comparisons (if archetype coordinates provided)

    Args:
        generated_pca: Generated PCA coordinates [n_gen, n_components]
        real_pca_data: Real PCA coordinates [n_real, n_components]
            - For 'full' comparison: all available real data (random sampling validation)
            - For 'regional' comparison: holdout slice from volumetric region (targeted validation)
        archetype_coords: Optional archetype coordinates [n_archetypes, n_components] for distance analysis
        comparison_type: 'full' (random real data) or 'regional' (volumetric holdout slice)
        nearest_percentile: Fraction of real cells to use as nearest neighbors (distance-based)
        verbose: Print progress information

    Returns
    -------
        Dict containing:
        - 'cosine_similarities': Cell-wise cosine similarity analysis between generated and nearest real cells
        - 'distribution_stats': KS test results for each PC component distribution
        - 'archetype_analysis': Distance distribution comparisons to each archetype (if provided)
        - 'comparison_info': Metadata about comparison strategy and parameters

    Priority Outputs:
        1. distribution_stats: KS tests show overall PCA distribution similarity
        2. archetype_analysis: Distance distributions should be similar across archetypes
    """
    if verbose:
        print(f"[STATS] Validating PCA distributions ({comparison_type} comparison)...")
        print(f"   Generated: {generated_pca.shape[0]} samples")
        print(f"   Real: {real_pca_data.shape[0]} samples")

    n_generated = generated_pca.shape[0]
    n_real = real_pca_data.shape[0]
    n_nearest = max(1, int(n_real * nearest_percentile))

    # 1. Cell-wise cosine similarities in PCA space
    cosine_similarities = []

    from sklearn.metrics.pairwise import cosine_similarity

    for i in range(min(n_generated, 2500)):  # Limit for computational efficiency
        # Find nearest real cells in PCA space
        distances_to_real = cdist([generated_pca[i]], real_pca_data, metric="euclidean")[0]
        nearest_indices = np.argsort(distances_to_real)[:n_nearest]

        # Compute cosine similarity with nearest real cells
        for idx in nearest_indices:
            similarity = cosine_similarity([generated_pca[i]], [real_pca_data[idx]])[0, 0]
            if not np.isnan(similarity):
                cosine_similarities.append(similarity)

    cosine_similarity_results = {
        "similarities": cosine_similarities,
        "mean_similarity": np.mean(cosine_similarities) if cosine_similarities else 0.0,
        "median_similarity": np.median(cosine_similarities) if cosine_similarities else 0.0,
        "std_similarity": np.std(cosine_similarities) if cosine_similarities else 0.0,
        "n_comparisons": len(cosine_similarities),
    }

    # 2. Distribution statistics per PC component
    distribution_stats = {}
    for pc_idx in range(generated_pca.shape[1]):
        gen_pc = generated_pca[:, pc_idx]
        real_pc = real_pca_data[:, pc_idx]

        # KS test for this PC component
        ks_stat, ks_pval = stats.ks_2samp(gen_pc, real_pc)

        distribution_stats[f"PC{pc_idx}"] = {
            "ks_statistic": ks_stat,
            "ks_p_value": ks_pval,
            "ks_significant": ks_pval < 0.05,
            "gen_mean": float(gen_pc.mean()),
            "real_mean": float(real_pc.mean()),
            "gen_std": float(gen_pc.std()),
            "real_std": float(real_pc.std()),
        }

    # 3. PCA distribution comparison summary

    # 4. Archetype distance analysis
    archetype_analysis = None
    if archetype_coords is not None:
        if verbose:
            print("   Computing archetype distances...")

        # Compute distances from generated cells to archetypes
        gen_to_archetypes = cdist(generated_pca, archetype_coords, metric="euclidean")
        real_to_archetypes = cdist(real_pca_data, archetype_coords, metric="euclidean")

        n_archetypes = archetype_coords.shape[0]
        archetype_stats = {}

        for arch_idx in range(n_archetypes):
            gen_distances = gen_to_archetypes[:, arch_idx]
            real_distances = real_to_archetypes[:, arch_idx]

            # KS test comparing distance distributions
            ks_stat, ks_pval = stats.ks_2samp(gen_distances, real_distances)

            archetype_stats[f"archetype_{arch_idx}"] = {
                "gen_mean_distance": float(gen_distances.mean()),
                "real_mean_distance": float(real_distances.mean()),
                "gen_std_distance": float(gen_distances.std()),
                "real_std_distance": float(real_distances.std()),
                "distance_ks_stat": ks_stat,
                "distance_ks_pval": ks_pval,
                "distance_ks_significant": ks_pval < 0.05,
            }

        # Overall archetype distance analysis
        archetype_analysis = {
            "per_archetype_stats": archetype_stats,
            "n_archetypes": n_archetypes,
            "gen_mean_min_distance": float(np.min(gen_to_archetypes, axis=1).mean()),
            "real_mean_min_distance": float(np.min(real_to_archetypes, axis=1).mean()),
            "significant_distance_tests": sum(stats["distance_ks_significant"] for stats in archetype_stats.values()),
        }

        if verbose:
            print(
                f"     Archetype distance tests significant: {archetype_analysis['significant_distance_tests']}/{n_archetypes}"
            )
    else:
        if verbose:
            print("     No archetype coordinates provided, skipping archetype distance analysis")

    if verbose:
        print("   [OK] PCA validation complete:")
        print(f"     Mean cosine similarity: {cosine_similarity_results['mean_similarity']:.3f}")
        significant_ks_tests = sum(stats["ks_significant"] for stats in distribution_stats.values())
        print(f"     KS tests significant: {significant_ks_tests}/{len(distribution_stats)} PC components")
        if archetype_analysis is not None:
            print(
                f"     Archetype distance tests significant: {archetype_analysis['significant_distance_tests']}/{archetype_analysis['n_archetypes']} archetypes"
            )

    return {
        "cosine_similarities": cosine_similarity_results,
        "distribution_stats": distribution_stats,
        "archetype_analysis": archetype_analysis,
        "comparison_info": {
            "comparison_type": comparison_type,
            "n_generated": n_generated,
            "n_real": n_real,
            "nearest_percentile": nearest_percentile,
            "n_components": generated_pca.shape[1],
            "has_archetype_analysis": archetype_analysis is not None,
        },
    }


def validate_gene_correlations(
    generated_genes: np.ndarray,
    real_gene_data: np.ndarray,
    gene_names: list[str] | None = None,
    max_genes: int | None = 2000,  # FIXED: Allow None to test all genes
    max_samples: int = 100,
    correlation_method: str = "spearman",
    balance_sample_sizes: bool = True,
    use_hvg: bool = False,  # NEW: Focus on highly variable genes
    hvg_subset: np.ndarray | None = None,  # NEW: Boolean mask for HVG subset
    verbose: bool = True,
) -> dict:
    """
    MODULAR GENE VALIDATION: Compare gene expression correlations between generated and real data.

    Args:
        generated_genes: Generated gene expression [n_gen, n_genes]
        real_gene_data: Real gene expression [n_real, n_genes]
        gene_names: Optional gene names for reporting
        max_genes: Maximum genes to test (None = test all genes, int = limit for computational efficiency)
        max_samples: Maximum samples to use per dataset
        correlation_method: Correlation method to use ('spearman' or 'pearson')
            - 'spearman': Rank-based correlation, robust to outliers and non-normal distributions (DEFAULT)
            - 'pearson': Linear correlation, assumes normal distributions
        balance_sample_sizes: Balance sample sizes between generated and real for fair comparison
        use_hvg: If True, focus analysis on highly variable genes (requires hvg_subset or computed from data)
        hvg_subset: Boolean mask [n_genes] indicating which genes are highly variable
        verbose: Print progress information

    Returns
    -------
        Dict containing:
        - 'gene_correlations': Per-gene correlation analysis using specified method
        - 'sample_correlations': Per-sample correlation analysis using specified method
        - 'expression_stats': Expression distribution comparison
        - 'correlation_summary': Overall correlation assessment
    """
    # Validate correlation method
    if correlation_method not in ["spearman", "pearson"]:
        raise ValueError(f"correlation_method must be 'spearman' or 'pearson', got '{correlation_method}'")

    if verbose:
        print(" Validating gene expression correlations...")
        print(f"   Generated genes: {generated_genes.shape}")
        print(f"   Real genes: {real_gene_data.shape}")
        print(f"   Correlation method: {correlation_method}")
        if use_hvg:
            print("    HIGHLY VARIABLE GENES MODE: Focusing on informative genes")
        elif max_genes is None:
            print(f"    Testing ALL {min(generated_genes.shape[1], real_gene_data.shape[1])} genes")
        else:
            print(f"    Testing {max_genes} genes (limited for computational efficiency)")

    # Handle highly variable genes subset
    gene_subset_mask = None
    if use_hvg:
        if hvg_subset is not None:
            gene_subset_mask = hvg_subset
            if verbose:
                n_hvg = gene_subset_mask.sum()
                total_genes = len(gene_subset_mask)
                print(f"    Using provided HVG subset: {n_hvg}/{total_genes} genes ({n_hvg / total_genes * 100:.1f}%)")
        else:
            if verbose:
                print("   [WARNING]  HVG mode requested but no hvg_subset provided - falling back to all genes")
                print("   NOTE:  Consider computing HVGs first: sc.pp.highly_variable_genes(adata)")
            use_hvg = False

    # Handle max_genes parameter and HVG subset
    total_available_genes = min(generated_genes.shape[1], real_gene_data.shape[1])

    if use_hvg and gene_subset_mask is not None:
        # Use HVG subset - subset both generated and real data
        generated_genes_subset = generated_genes[:, gene_subset_mask]
        real_gene_data_subset = real_gene_data[:, gene_subset_mask]
        n_genes = generated_genes_subset.shape[1]

        # Update gene names if provided
        if gene_names is not None:
            gene_names_subset = [gene_names[i] for i in range(len(gene_names)) if gene_subset_mask[i]]
        else:
            gene_names_subset = None

        if verbose:
            print(f"    HVG subset applied: {n_genes} highly variable genes selected")
    else:
        # Use all genes or max_genes limit
        if max_genes is None:
            n_genes = total_available_genes
        else:
            n_genes = min(total_available_genes, max_genes)

        generated_genes_subset = generated_genes[:, :n_genes]
        real_gene_data_subset = real_gene_data[:, :n_genes]
        gene_names_subset = gene_names[:n_genes] if gene_names is not None else None

    # Balance sample sizes for fair comparison if requested
    if balance_sample_sizes:
        n_available_gen = generated_genes.shape[0]
        n_available_real = real_gene_data.shape[0]
        n_balanced = min(n_available_gen, n_available_real, max_samples)
        n_gen_samples = n_balanced
        n_real_samples = n_balanced
        if verbose:
            print(f"     Balanced sample sizes: {n_gen_samples} generated, {n_real_samples} real")
    else:
        n_gen_samples = min(generated_genes.shape[0], max_samples)
        n_real_samples = min(real_gene_data.shape[0], max_samples)

    # 1. Feature-wise correlations (per gene across samples)
    gene_correlations = []
    significant_genes = 0

    for gene_idx in range(n_genes):
        gen_gene_expr = generated_genes_subset[:n_gen_samples, gene_idx]
        real_gene_expr = real_gene_data_subset[:n_real_samples, gene_idx]

        # Only compute correlation if both have variation
        if len(np.unique(gen_gene_expr)) > 1 and len(np.unique(real_gene_expr)) > 1:
            if correlation_method == "spearman":
                corr, p_val = stats.spearmanr(gen_gene_expr, real_gene_expr)
            else:  # pearson
                corr, p_val = stats.pearsonr(gen_gene_expr, real_gene_expr)
            if not np.isnan(corr):
                gene_correlations.append(
                    {
                        "gene_idx": gene_idx,
                        "gene_name": gene_names_subset[gene_idx] if gene_names_subset else f"gene_{gene_idx}",
                        "correlation": corr,
                        "p_value": p_val,
                        "significant": p_val < 0.05,
                    }
                )
                if p_val < 0.05:
                    significant_genes += 1

    # 2. Sample-wise correlations (per sample across genes)
    sample_correlations = []

    for sample_idx in range(min(n_gen_samples, 50)):  # Limit samples for efficiency
        gen_sample = generated_genes_subset[sample_idx, :n_genes]

        # Find most similar real sample
        sample_similarities = []
        for real_idx in range(min(n_real_samples, 100)):
            real_sample = real_gene_data_subset[real_idx, :n_genes]

            # Compute correlation between sample profiles
            if len(np.unique(gen_sample)) > 1 and len(np.unique(real_sample)) > 1:
                if correlation_method == "spearman":
                    corr, _ = stats.spearmanr(gen_sample, real_sample)
                else:  # pearson
                    corr, _ = stats.pearsonr(gen_sample, real_sample)
                if not np.isnan(corr):
                    sample_similarities.append(corr)

        if sample_similarities:
            sample_correlations.append(
                {
                    "sample_idx": sample_idx,
                    "max_correlation": max(sample_similarities),
                    "mean_correlation": np.mean(sample_similarities),
                    "median_correlation": np.median(sample_similarities),
                }
            )

    # 3. Expression distribution statistics
    gen_expr_stats = {
        "mean_expression": float(generated_genes_subset[:n_gen_samples, :n_genes].mean()),
        "std_expression": float(generated_genes_subset[:n_gen_samples, :n_genes].std()),
        "min_expression": float(generated_genes_subset[:n_gen_samples, :n_genes].min()),
        "max_expression": float(generated_genes_subset[:n_gen_samples, :n_genes].max()),
        "zero_fraction": float((generated_genes_subset[:n_gen_samples, :n_genes] == 0).mean()),
    }

    real_expr_stats = {
        "mean_expression": float(real_gene_data_subset[:n_real_samples, :n_genes].mean()),
        "std_expression": float(real_gene_data_subset[:n_real_samples, :n_genes].std()),
        "min_expression": float(real_gene_data_subset[:n_real_samples, :n_genes].min()),
        "max_expression": float(real_gene_data_subset[:n_real_samples, :n_genes].max()),
        "zero_fraction": float((real_gene_data_subset[:n_real_samples, :n_genes] == 0).mean()),
    }

    # 4. Overall correlation summary
    gene_corr_values = [g["correlation"] for g in gene_correlations]
    sample_max_corrs = [s["max_correlation"] for s in sample_correlations]

    correlation_summary = {
        "correlation_method": correlation_method,
        "gene_correlations": {
            "mean": np.mean(gene_corr_values) if gene_corr_values else 0.0,
            "median": np.median(gene_corr_values) if gene_corr_values else 0.0,
            "std": np.std(gene_corr_values) if gene_corr_values else 0.0,
            "significant_fraction": significant_genes / len(gene_correlations) if gene_correlations else 0.0,
        },
        "sample_correlations": {
            "mean_max": np.mean(sample_max_corrs) if sample_max_corrs else 0.0,
            "median_max": np.median(sample_max_corrs) if sample_max_corrs else 0.0,
            "std_max": np.std(sample_max_corrs) if sample_max_corrs else 0.0,
        },
        "n_genes_tested": len(gene_correlations),
        "n_samples_tested": len(sample_correlations),
    }

    if verbose:
        print("   [OK] Gene validation complete:")
        print(
            f"     {correlation_method.title()} correlation - Gene: {correlation_summary['gene_correlations']['mean']:.3f} (n={len(gene_correlations)})"
        )
        print(
            f"     Significant genes: {significant_genes}/{len(gene_correlations)} ({significant_genes / len(gene_correlations) * 100:.1f}%)"
        )
        print(f"     Sample max correlations: {correlation_summary['sample_correlations']['mean_max']:.3f}")

    return {
        "gene_correlations": gene_correlations,
        "sample_correlations": sample_correlations,
        "expression_stats": {"generated": gen_expr_stats, "real": real_expr_stats},
        "correlation_summary": correlation_summary,
        "validation_info": {
            "n_genes_tested": n_genes,
            "n_gen_samples": n_gen_samples,
            "n_real_samples": n_real_samples,
            "max_genes": max_genes,
            "max_samples": max_samples,
            "use_hvg": use_hvg,
            "hvg_fraction": n_genes / total_available_genes if use_hvg and gene_subset_mask is not None else None,
            "total_available_genes": total_available_genes,
        },
    }


def validate_archetypal_distributions(
    generated_weights: np.ndarray, real_weights: np.ndarray, verbose: bool = True
) -> dict:
    """
    SIMPLIFIED ARCHETYPAL VALIDATION: Compare real vs generated archetype weight distributions.

    Args:
        generated_weights: Generated archetype weights [n_gen, n_archetypes]
        real_weights: Real archetype weights [n_real, n_archetypes] (from .obsm after training)
        verbose: Print progress information

    Returns
    -------
        Dict with KS test results comparing weight distributions per archetype
    """
    # Validate input shapes
    if generated_weights.shape[1] != real_weights.shape[1]:
        raise ValueError(
            f"Archetype dimension mismatch: generated {generated_weights.shape[1]} vs real {real_weights.shape[1]}"
        )

    n_archetypes = generated_weights.shape[1]

    if verbose:
        print("🧪 Validating archetypal weight distributions...")
        print(f"   Comparing {generated_weights.shape[0]} generated vs {real_weights.shape[0]} real samples")
        print(f"   Testing {n_archetypes} archetype weight distributions")

    # KS tests for each archetype weight distribution
    ks_results = []
    for arch_idx in range(n_archetypes):
        gen_arch_weights = generated_weights[:, arch_idx]
        real_arch_weights = real_weights[:, arch_idx]

        # KS test comparing weight distributions
        ks_stat, ks_pval = stats.ks_2samp(gen_arch_weights, real_arch_weights)

        arch_result = {
            "archetype_index": arch_idx,
            "ks_statistic": ks_stat,
            "ks_p_value": ks_pval,
            "ks_significant": ks_pval < 0.05,
            "distribution_stats": {
                "gen_mean": float(gen_arch_weights.mean()),
                "real_mean": float(real_arch_weights.mean()),
                "gen_std": float(gen_arch_weights.std()),
                "real_std": float(real_arch_weights.std()),
                "gen_range": [float(gen_arch_weights.min()), float(gen_arch_weights.max())],
                "real_range": [float(real_arch_weights.min()), float(real_arch_weights.max())],
            },
        }

        ks_results.append(arch_result)

        if verbose:
            print(
                f"   Archetype {arch_idx}: KS statistic={ks_stat:.3f}, p-value={ks_pval:.3f} {'(significant)' if ks_pval < 0.05 else ''}"
            )

    # Overall summary
    significant_tests = sum(r["ks_significant"] for r in ks_results)

    summary = {
        "n_archetypes_tested": n_archetypes,
        "significant_tests": significant_tests,
        "fraction_significant": significant_tests / n_archetypes if n_archetypes > 0 else 0.0,
        "overall_significant": significant_tests > 0,
    }

    if verbose:
        print("   [OK] Archetype validation complete:")
        print(f"     Significant weight distribution differences: {significant_tests}/{n_archetypes} archetypes")
        print(f"     Fraction significant: {summary['fraction_significant']:.1%}")

    return {
        "archetype_ks_results": ks_results,
        "summary": summary,
        "test_info": {
            "n_generated": generated_weights.shape[0],
            "n_real": real_weights.shape[0],
            "n_archetypes": n_archetypes,
            "validation_type": "weight_distributions_only",
        },
    }


def create_generated_samples_anndata(
    pca_coordinates: np.ndarray,
    archetype_weights: np.ndarray,
    gene_expression: np.ndarray | None = None,
    sample_source: str = "generated",
    var_names: list[str] | None = None,
    archetype_coordinates: np.ndarray | None = None,
    source_adata_uns: dict | None = None,
    verbose: bool = True,
) -> ad.AnnData:
    """
    Create AnnData object from generated samples for analysis and visualization.

    Args:
        pca_coordinates: Generated PCA coordinates [n_samples, n_components]
        archetype_weights: Archetype weights [n_samples, n_archetypes]
        gene_expression: Gene expression [n_samples, n_genes] (optional)
        sample_source: Label for sample origin (e.g., 'generated', 'holdout_sampled')
        var_names: Gene names (required if gene_expression provided)
        archetype_coordinates: Archetype coordinates for visualization [n_archetypes, n_components]
        source_adata_uns: Original adata.uns dict to copy metadata (especially archetype_coordinates)
        verbose: Print progress information

    Returns
    -------
        AnnData object with generated samples
    """
    n_samples = pca_coordinates.shape[0]

    if verbose:
        print(f" Creating AnnData for {n_samples} generated samples...")

    # Use gene expression as X if available, otherwise use PCA coordinates
    if gene_expression is not None:
        if var_names is None:
            var_names = [f"gene_{i}" for i in range(gene_expression.shape[1])]
        X_data = gene_expression
        if verbose:
            print(f"   Using gene expression as X: {gene_expression.shape}")
    else:
        var_names = [f"PC_{i}" for i in range(pca_coordinates.shape[1])]
        X_data = pca_coordinates
        if verbose:
            print(f"   Using PCA coordinates as X: {pca_coordinates.shape}")

    # Create AnnData
    adata = ad.AnnData(X=X_data)

    # Set observation names
    adata.obs_names = [f"{sample_source}_{i}" for i in range(n_samples)]
    adata.var_names = var_names

    # Add sample metadata
    adata.obs["sample_source"] = sample_source
    adata.obs["is_generated"] = True

    # Store PCA coordinates in obsm
    adata.obsm["X_pca"] = pca_coordinates

    # Store archetype weights in obsm
    adata.obsm["archetype_weights"] = archetype_weights

    # Add archetype assignments using distance-based binning
    # First, compute distances to archetypes if coordinates available
    if archetype_coordinates is not None:
        from .analysis import bin_cells_by_archetype, compute_archetype_distances

        # Temporarily store archetype coordinates for distance computation
        adata.uns["archetype_coordinates"] = archetype_coordinates

        # Compute distances and bin cells
        _ = compute_archetype_distances(adata, verbose=False)
        _ = bin_cells_by_archetype(adata, percentage_per_archetype=0.1, verbose=False)

        if verbose:
            print("   Assigned cells to archetypes using distance-based binning")
    else:
        # Fallback: simple dominant archetype assignment
        dominant_archetypes = np.argmax(archetype_weights, axis=1)
        adata.obs["dominant_archetype"] = [
            f"archetype_{dominant_archetypes[i]}" for i in range(len(dominant_archetypes))
        ]
        adata.obs["dominant_archetype"] = adata.obs["dominant_archetype"].astype("category")

        if verbose:
            print("   Used fallback dominant archetype assignment (no archetype coordinates provided)")

    # Add max weight as a continuous variable for visualization
    adata.obs["max_archetype_weight"] = np.max(archetype_weights, axis=1)

    # Copy additional metadata for visualization
    if archetype_coordinates is not None and verbose:
        print(f"   [STATS] Added archetype coordinates: {archetype_coordinates.shape}")
    elif source_adata_uns is not None and "archetype_coordinates" in source_adata_uns:
        adata.uns["archetype_coordinates"] = source_adata_uns["archetype_coordinates"]
        if verbose:
            print("   [STATS] Copied archetype coordinates from source adata.uns")

    # Copy other relevant metadata from source adata.uns
    if source_adata_uns is not None:
        # Copy PCA metadata if available
        if "pca" in source_adata_uns:
            adata.uns["pca"] = source_adata_uns["pca"].copy()
        # Copy any other archetype-related metadata
        for key in source_adata_uns:
            if "archetype" in key.lower() and key not in adata.uns:
                adata.uns[key] = source_adata_uns[key]
        if verbose:
            print(f"    Copied metadata keys: {list(source_adata_uns.keys())}")

    if verbose:
        print(f"   [OK] Generated AnnData: {adata}")
        if "archetypes" in adata.obs:
            print(f"     Archetype assignments: {adata.obs['archetypes'].value_counts().to_dict()}")
        elif "dominant_archetype" in adata.obs:
            print(f"     Dominant archetypes: {adata.obs['dominant_archetype'].value_counts().to_dict()}")
        print(f"     Mean max weight: {adata.obs['max_archetype_weight'].mean():.3f}")
        print(f"     Available .uns keys: {list(adata.uns.keys())}")

    return adata


# Convenience function for complete workflow


def generate_and_validate_samples(
    model,
    adata: ad.AnnData,
    n_samples: int = 1000,
    strategy: str = "poisson",
    validation_fraction: float = 0.3,
    pca_key: str = "X_pca",
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """
    Complete workflow: generate samples and validate against holdout data.

    Args:
        model: Trained archetypal model
        adata: Full AnnData object
        n_samples: Number of samples to generate
        strategy: Sampling strategy for generation
        validation_fraction: Fraction of data to use for validation
        pca_key: Key for PCA coordinates
        device: PyTorch device
        verbose: Print progress

    Returns
    -------
        Dict with generation results, validation results, and summary
    """
    if verbose:
        print(" Complete generative sampling and validation workflow")
        print("=" * 60)

    # Create random holdout for validation (simpler than volumetric for now)
    n_total = adata.n_obs
    n_validation = int(n_total * validation_fraction)

    validation_indices = np.random.choice(n_total, size=n_validation, replace=False)
    training_mask = np.ones(n_total, dtype=bool)
    training_mask[validation_indices] = False

    training_adata = adata[training_mask]
    validation_adata = adata[validation_indices]

    if verbose:
        print(f"[STATS] Data split: {training_adata.n_obs} training, {validation_adata.n_obs} validation")

    # Generate samples using training data bounds
    sampling_results = sample_pca_coordinates(
        adata=training_adata, n_samples=n_samples, strategy=strategy, pca_key=pca_key, random_seed=42, verbose=verbose
    )

    generation_results = generate_synthetic_data(
        model=model, pca_coordinates=sampling_results["pca_coordinates"], device=device, verbose=verbose
    )

    # Prepare generated data for validation
    generated_data = {
        "pca_coordinates": sampling_results["pca_coordinates"],
        "archetype_weights": generation_results["archetype_weights"],
        "reconstructed_pca": generation_results["reconstructed_pca"],
        "constraint_violations": generation_results["constraint_violations"],
        "generation_info": {"strategy": strategy, "archetypal_r2": generation_results["archetypal_r2"]},
    }

    # Validate against holdout data using modular functions
    validation_data = (
        validation_adata.obsm[pca_key] if pca_key in validation_adata.obsm else validation_adata.obsm["X_pca"]
    )

    # Extract archetype coordinates from model for distance analysis
    archetype_coords = None
    try:
        if hasattr(model, "archetypes"):
            # Get archetype positions in original space
            archetype_positions = model.archetypes.detach().cpu().numpy()

            # Transform to PCA space if possible
            if hasattr(adata, "varm") and "PCs" in adata.varm:
                pca_components = adata.varm["PCs"]
                pca_mean = adata.var["mean"] if "mean" in adata.var else adata.X.mean(axis=0)
                archetype_coords = (archetype_positions - pca_mean) @ pca_components
            elif "pca" in adata.uns and "mean" in adata.uns["pca"]:
                # Use scanpy PCA transformation
                pca_mean = adata.uns["pca"]["mean"]
                pca_components = adata.varm["PCs"] if "PCs" in adata.varm else None
                if pca_components is not None:
                    archetype_coords = (archetype_positions - pca_mean) @ pca_components
    except Exception as e:
        if verbose:
            print(f"   [WARNING]  Could not extract archetype coordinates: {e}")

    # PCA validation
    pca_validation = validate_pca_distributions(
        generated_pca=generated_data["reconstructed_pca"],
        real_pca_data=validation_data,
        archetype_coords=archetype_coords,
        comparison_type="full",  # Using random holdout data
        nearest_percentile=0.1,  # 10% nearest real cells for cosine similarity
        verbose=verbose,
    )

    # Compile validation results
    validation_results = {
        "pca_validation": pca_validation,
        "cosine_similarities": pca_validation["cosine_similarities"],
        "constraint_analysis": generated_data["constraint_violations"],
        "summary_stats": {
            "n_generated": generated_data["pca_coordinates"].shape[0],
            "n_real": validation_data.shape[0],
            "mean_cosine_similarity": pca_validation["cosine_similarities"]["mean_similarity"],
            "validation_timestamp": pd.Timestamp.now(),
        },
    }

    # Compile complete results
    complete_results = {
        "generated_data": generated_data,
        "validation_results": validation_results,
        "data_split_info": {
            "n_training": training_adata.n_obs,
            "n_validation": validation_adata.n_obs,
            "validation_fraction": validation_fraction,
        },
        "workflow_summary": {
            "n_samples_generated": n_samples,
            "strategy": strategy,
            "constraint_violation_rate": generated_data["constraint_violations"]["violation_rate"],
            "mean_cosine_similarity": validation_results["cosine_similarities"]["mean_similarity"],
            "workflow_timestamp": pd.Timestamp.now(),
        },
    }

    if verbose:
        print("\n Workflow Summary:")
        print(
            f"   Generated {n_samples} samples with {generated_data['constraint_violations']['violation_rate'] * 100:.1f}% constraint violations"
        )
        cosine_sim = validation_results["cosine_similarities"]["mean_similarity"]
        print(f"   Mean cosine similarity: {cosine_sim:.3f}")
        print("   [OK] Complete workflow finished")

    return complete_results

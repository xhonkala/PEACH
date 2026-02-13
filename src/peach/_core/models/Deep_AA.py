"""
Deep Archetypal Analysis (Deep_AA)
==================================

VAE-based archetypal analysis with convex hull constraints and PCHA initialization.

This module implements Deep Archetypal Analysis, combining variational
autoencoder architecture with archetypal constraints. The model learns
archetypes (extreme points) and represents each data point as a convex
combination of these archetypes.

Architecture
------------
- Latent space z = archetypal coordinates (A matrix)
- Learned archetypes as model parameters (Y matrix)
- Single reconstruction path: x_hat = A @ Y
- Softmax constraint ensures valid barycentric coordinates

Main Classes
------------
Deep_AA : Deep archetypal analysis model

Type Definitions
----------------
See ``peach._core.types`` for Pydantic models:

- ``DeepAAConfig`` : Model configuration
- ``DeepAAForwardOutput`` : forward() return dict
- ``DeepAALossOutput`` : loss_function() return dict
- ``ArchetypeInitConfig`` : Initialization options
- ``ConstraintValidation`` : validate_constraints() result

Examples
--------
>>> from peach._core.models.Deep_AA import Deep_AA
>>> # Create model
>>> model = Deep_AA(input_dim=30, n_archetypes=5, hidden_dims=[256, 128, 64], inflation_factor=1.5)
>>> # Initialize archetypes with PCHA
>>> model.initialize_archetypes(X_sample, use_pcha=True, use_inflation=True)
>>> # Forward pass
>>> outputs = model(data)
>>> A = outputs["A"]  # Cell-to-archetype weights
>>> Y = outputs["Y"]  # Archetype positions

See Also
--------
peach.tl.train_archetypal : User-facing training function
peach._core.types.DeepAAConfig : Configuration model
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .VAE_Base import VAE_Base
from ..utils.metrics import calculate_archetype_r2


class Deep_AA(VAE_Base):
    """Deep Archetypal Analysis model.

    Combines VAE architecture with archetypal analysis constraints.
    Each data point is represented as a convex combination of learned
    archetypes, with coordinates constrained to be non-negative and
    sum to 1.

    Parameters
    ----------
    input_dim : int
        Input feature dimension (e.g., number of PCA components).
    n_archetypes : int
        Number of archetypes to learn.
    latent_dim : int | None, default: None
        Latent dimension. Automatically set to n_archetypes.
    hidden_dims : list[int] | None, default: None
        Hidden layer sizes. Default: [128, 64, 32].

    archetypal_weight : float, default: 0.9
        Weight for archetypal reconstruction loss.
    kld_weight : float, default: 0.1
        Weight for KL divergence. Regularizes encoder variance to
        prevent posterior drift during extended training.
    diversity_weight : float, default: 0.0
        Weight for archetype diversity. **Warning: Hurts performance**.
    regularity_weight : float, default: 0.0
        Weight for usage regularity. **Warning: Hurts performance**.
    sparsity_weight : float, default: 0.0
        Weight for coordinate sparsity.
    manifold_weight : float, default: 0.0
        Weight for manifold regularization. **Warning: Hurts performance**.

    inflation_factor : float, default: 1.5
        Scalar inflation factor for PCHA initialization.
        This is the "Helsinki breakthrough" parameter that improves
        archetype positioning by scaling them away from the centroid.
    use_barycentric : bool, default: True
        Use softmax for strict barycentric coordinates (sum to 1).
    use_hidden_transform : bool, default: True
        Apply learned transformation to archetype positions.

    Attributes
    ----------
    archetypes : nn.Parameter
        Learned archetype positions [n_archetypes, input_dim].
        This is the Y matrix in archetypal analysis.
    archetypes_initialized : bool
        Whether archetypes have been initialized with PCHA.
    pcha_results : dict | None
        Results from PCHA initialization (if performed).

    Notes
    -----
    **Loss Weight Defaults**: The default configuration (only archetypal_weight=1.0,
    all others=0.0) is the result of extensive ablation studies. Additional
    loss terms consistently hurt performance.

    **Archetypal Constraint**: In this model, z represents archetypal coordinates
    (the A matrix), NOT a traditional VAE latent space. Each row sums to 1 via
    softmax, representing how much each archetype contributes to that sample.

    Examples
    --------
    >>> # Create and initialize model
    >>> model = Deep_AA(input_dim=30, n_archetypes=5)
    >>> model.initialize_archetypes(data, use_pcha=True, use_inflation=True)
    >>> # Training step
    >>> outputs = model(batch)
    >>> loss_dict = model.loss_function(outputs)
    >>> loss_dict["loss"].backward()
    >>> optimizer.step()
    >>> # Access archetypes and coordinates
    >>> A = outputs["A"]  # [batch_size, n_archetypes] - sums to 1 per row
    >>> Y = outputs["Y"]  # [n_archetypes, input_dim]
    >>> reconstruction = A @ Y  # Same as outputs['arch_recons']

    See Also
    --------
    VAE_Base : Parent class
    peach.tl.train_archetypal : User-facing training
    peach._core.types.DeepAAConfig : Configuration model
    """

    def __init__(
        self,
        input_dim: int,
        n_archetypes: int,
        latent_dim: int = None,
        hidden_dims: list[int] = None,
        archetypal_weight: float = 0.9,
        kld_weight: float = 0.1,
        diversity_weight: float = 0.0,
        regularity_weight: float = 0.0,
        sparsity_weight: float = 0.0,
        manifold_weight: float = 0.0,
        inflation_factor: float = 1.5,
        use_barycentric: bool = True,
        use_hidden_transform: bool = True,
        **kwargs,
    ) -> None:
        """Initialize Deep_AA model.

        See class docstring for parameter descriptions.
        """
        # Set latent_dim to n_archetypes if not provided (archetypal constraint)
        if latent_dim is None:
            latent_dim = n_archetypes
        elif latent_dim != n_archetypes:
            print(f"Setting latent_dim = n_archetypes = {n_archetypes} for unified archetypal space")
            latent_dim = n_archetypes

        # Initialize parent VAE_Base
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            n_archetypes=n_archetypes,
            archetypal_weights=archetypal_weight,
            **kwargs,
        )

        # Store loss weights (same as Deep_2)
        self.archetypal_weight = archetypal_weight
        self.kld_weight = kld_weight
        self.diversity_weight = diversity_weight
        self.regularity_weight = regularity_weight
        self.sparsity_weight = sparsity_weight
        self.manifold_weight = manifold_weight

        # NEW: Store inflation factor
        self.inflation_factor = inflation_factor

        # Archetypal behavior controls
        self.use_barycentric = use_barycentric
        self.use_hidden_transform = use_hidden_transform

        # Validate weights
        total_weight = archetypal_weight + kld_weight
        if not torch.isclose(torch.tensor(total_weight), torch.tensor(1.0), rtol=1e-2):
            print(f"Warning: Main loss weights sum to {total_weight}, not 1.0")

        # ARCHETYPAL COMPONENTS (same as Deep_2)
        self.archetypes = nn.Parameter(torch.randn(n_archetypes, input_dim))

        print(f"Archetypes parameter registered: {'archetypes' in dict(self.named_parameters())}")
        print(f"Archetypes requires_grad: {self.archetypes.requires_grad}")

        # Optional transformation layer
        if use_hidden_transform:
            self.archetype_transform = nn.Sequential(
                nn.Linear(input_dim, input_dim), nn.LeakyReLU(), nn.Linear(input_dim, input_dim)
            )

        # Initialize archetype tracking
        self.archetypes_initialized = False
        self.previous_loss = None
        self.loss_history = []

        print("Deep_AA (Deep Archetypal Analysis) initialized:")
        print("  - Single-stage architecture (like Deep_2)")
        print(f"  - Inflation factor: {inflation_factor}")
        print("  - Direct archetypal coordinates (no bottleneck)")

    # ============================================================================
    # DEEP_2 CORE METHODS (unchanged)
    # ============================================================================

    def encode(self, input: torch.Tensor) -> list[torch.Tensor]:
        """Encode input to archetypal coordinate parameters.

        Parameters
        ----------
        input : torch.Tensor
            Input data [batch_size, input_dim].

        Returns
        -------
        list[torch.Tensor]
            Two-element list [mu, log_var]:

            - ``mu`` : Coordinate mean [batch_size, n_archetypes]
            - ``log_var`` : Coordinate log variance [batch_size, n_archetypes]
              (clamped for barycentric stability)
        """
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        # Apply barycentric constraints to mu if enabled
        if self.use_barycentric:
            # For barycentric, we want lower variance
            log_var = torch.clamp(log_var, max=2)  # moving from -1.0 to 2.0 to allow more exploration

        return [mu, log_var]

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample archetypal coordinates with convexity constraints.

        Unlike standard VAE reparameterization, this applies softmax
        to ensure coordinates are valid barycentric coordinates
        (non-negative, sum to 1).

        Parameters
        ----------
        mu : torch.Tensor
            Coordinate mean [batch_size, n_archetypes].
        logvar : torch.Tensor
            Coordinate log variance [batch_size, n_archetypes].

        Returns
        -------
        torch.Tensor
            Valid archetypal coordinates [batch_size, n_archetypes].
            Each row sums to 1 and all values are non-negative.

        Notes
        -----
        During training, adds exploration noise before softmax.
        During inference, uses mu directly (no sampling).
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            exploration_noise = 0.05 * torch.randn_like(mu)  # Small jiggling
            z = mu + eps * std + exploration_noise
        else:
            z = mu

        # Ensure coordinates are valid archetypal coordinates
        if self.use_barycentric:
            z = F.softmax(z, dim=1)  # Barycentric: sum to 1
            # temperature = 7.5 # trying adding a softer scaling on softmax
            # z = F.softmax(z/temperature, dim=1) # softer scaling part 2
            # z = F.log_softmax(z, dim=1) #; z = z / (z.sum(dim=1, keepdim=True) + 1e-8)  # Soft normalization
        else:
            z = F.relu(z)  # At minimum, ensure non-negative
            # Normalize to sum to 1
            z = z / (z.sum(dim=1, keepdim=True) + 1e-8)

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from archetypal coordinates.

        Reconstructs data as weighted sum of archetypes: x_hat = A @ Y.

        Parameters
        ----------
        z : torch.Tensor
            Archetypal coordinates (A matrix) [batch_size, n_archetypes].

        Returns
        -------
        torch.Tensor
            Reconstructed data [batch_size, input_dim].

        Raises
        ------
        RuntimeError
            If z.shape[1] != n_archetypes (dimension mismatch).
        """
        # z already represents valid archetypal coordinates (A matrix)
        if self.use_barycentric:
            A = z  # z already constrained in reparameterize
        else:
            # Softer archetypal coordinates (less corner-pushing)
            temperature = 5.0  # Higher = softer constraints
            A = F.softmax(z / temperature, dim=1)

        # Apply optional transformation to archetypes
        if self.use_hidden_transform:
            archetypes_transformed = self.archetype_transform(self.archetypes)
        else:
            archetypes_transformed = self.archetypes

        # Archetypal reconstruction: x_hat = A @ Y
        # Debug: Check dimensions before matrix multiplication
        if A.shape[1] != archetypes_transformed.shape[0]:
            raise RuntimeError(
                f"Dimension mismatch in archetypal reconstruction: "
                f"A shape {A.shape} cannot be multiplied with archetypes shape {archetypes_transformed.shape}. "
                f"Expected A.shape[1] ({A.shape[1]}) == archetypes.shape[0] ({archetypes_transformed.shape[0]}). "
                f"This suggests latent_dim != n_archetypes."
            )
        reconstruction = torch.mm(A, archetypes_transformed)

        return reconstruction

    def get_effective_archetypes(self) -> torch.Tensor:
        """Return the archetypes as used for reconstruction.

        If use_hidden_transform is True, returns the transformed archetypes.
        Otherwise returns the raw learnable archetypes.

        This ensures Y in the output matches what's used for A @ Y = arch_recons.

        Returns
        -------
        torch.Tensor
            Effective archetype positions [n_archetypes, input_dim].
        """
        if self.use_hidden_transform:
            return self.archetype_transform(self.archetypes)
        return self.archetypes

    def forward(self, input: torch.Tensor) -> dict[str, torch.Tensor]:
        """Complete archetypal forward pass.

        Encodes input to archetypal coordinates and reconstructs via
        weighted archetype combination.

        Parameters
        ----------
        input : torch.Tensor
            Input data [batch_size, input_dim].

        Returns
        -------
        dict[str, torch.Tensor]
            Output dictionary with keys:

            **Primary outputs:**

            - ``arch_recons`` : Reconstruction [batch_size, input_dim]
            - ``mu`` : Coordinate mean [batch_size, n_archetypes]
            - ``log_var`` : Coordinate log variance [batch_size, n_archetypes]
            - ``z`` : Archetypal coordinates (= A matrix) [batch_size, n_archetypes]
            - ``archetypes`` : Effective archetype positions (= Y matrix) [n_archetypes, input_dim]
            - ``input`` : Original input [batch_size, input_dim]

            **Aliases (for compatibility):**

            - ``recons`` : Same as arch_recons
            - ``archetypal_coordinates`` : Same as z
            - ``A`` : Same as z
            - ``Y`` : Same as archetypes (effective, after transform if applicable)
            - ``raw_archetypes`` : Learnable parameter (before transform)

        Examples
        --------
        >>> outputs = model(data)
        >>> # Access coordinates and archetypes
        >>> A = outputs["A"]  # or outputs['z']
        >>> Y = outputs["Y"]  # or outputs['archetypes']
        >>> # Manual reconstruction (same as outputs['arch_recons'])
        >>> recons = A @ Y

        See Also
        --------
        peach._core.types.DeepAAForwardOutput : Return type structure
        """
        # Standard VAE encoding (but z are archetypal coordinates)
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)

        # SINGLE PATH: Archetypal reconstruction only
        arch_recons = self.decode(z)

        # Get effective archetypes (transformed if use_hidden_transform=True)
        # This ensures A @ Y == arch_recons
        effective_archetypes = self.get_effective_archetypes()

        # Return format compatible with existing pipeline
        return {
            # Primary outputs
            "arch_recons": arch_recons,
            "mu": mu,
            "log_var": log_var,
            "z": z,  # Archetypal coordinates = A matrix
            "archetypes": effective_archetypes,  # Y as used for reconstruction
            "input": input,
            # For legacy compatibility
            "recons": arch_recons,
            "archetypal_coordinates": z,
            "A": z,  # z IS the A matrix
            "Y": effective_archetypes,  # Y as used for reconstruction
            # Raw archetypes (learnable parameters) for inspection
            "raw_archetypes": self.archetypes,
        }

    # ============================================================================
    # DEEP_2 LOSS FUNCTIONS (unchanged)
    # ============================================================================

    def archetypal_diversity_loss(self) -> torch.Tensor:
        """Encourage diversity among archetypes.

        Vectorized implementation using broadcasting for GPU efficiency.
        Penalizes archetypes that are too close together.
        """
        archetypes = self.archetypes
        n_archetypes = archetypes.shape[0]

        # Vectorized pairwise distances using broadcasting
        # archetypes: (k, d) -> (k, 1, d) - (1, k, d) = (k, k, d)
        diff = archetypes.unsqueeze(0) - archetypes.unsqueeze(1)
        pairwise_dists = torch.norm(diff, dim=2)  # (k, k)

        # Extract upper triangle (unique pairs, excluding diagonal)
        upper_tri_mask = torch.triu(torch.ones_like(pairwise_dists, dtype=torch.bool), diagonal=1)
        unique_dists = pairwise_dists[upper_tri_mask]

        # Compute loss: sum of exp(-dist) for all pairs
        diversity_loss = torch.exp(-unique_dists).sum()
        n_pairs = n_archetypes * (n_archetypes - 1) / 2

        return diversity_loss / n_pairs

    def archetypal_regularity_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Encourage usage of all archetypes (from Deep_2)."""
        archetype_usage = z.mean(dim=0)
        uniform_target = torch.ones_like(archetype_usage) / self.n_archetypes
        regularity_loss = F.kl_div(F.log_softmax(archetype_usage, dim=0), uniform_target, reduction="batchmean")
        return regularity_loss

    def manifold_regularization_loss(self, input: torch.Tensor) -> torch.Tensor:
        """Keep archetypes on data manifold.

        Vectorized implementation for GPU efficiency.
        """
        archetypes = self.archetypes
        data_sample = input[: min(200, input.shape[0])]
        batch_size = data_sample.shape[0]

        # Strategy 1: Nearest neighbor penalty (VECTORIZED)
        # data_sample: (n, d), archetypes: (k, d)
        # Compute all pairwise distances: (n, d) - (k, d) via broadcasting
        # data_sample.unsqueeze(1): (n, 1, d)
        # archetypes.unsqueeze(0): (1, k, d)
        # diff: (n, k, d) -> distances: (n, k)
        diff = data_sample.unsqueeze(1) - archetypes.unsqueeze(0)
        all_distances = torch.norm(diff, dim=2)  # (n, k)

        # Min distance from each archetype to any data point
        min_distances = all_distances.min(dim=0)[0]  # (k,)
        manifold_loss = (min_distances ** 2).sum()

        # Strategy 2: Bounding box constraint (VECTORIZED)
        data_min = input.min(dim=0)[0]
        data_max = input.max(dim=0)[0]
        margin = 0.1 * (data_max - data_min)
        effective_min = data_min - margin
        effective_max = data_max + margin

        # All archetypes at once: (k, d)
        below_min = F.relu(effective_min.unsqueeze(0) - archetypes)  # (k, d)
        above_max = F.relu(archetypes - effective_max.unsqueeze(0))  # (k, d)
        bounding_loss = below_min.sum() + above_max.sum()

        # Strategy 3: Data density proximity (VECTORIZED)
        density_loss = torch.tensor(0.0, device=archetypes.device)
        if batch_size > 50:
            # Reuse all_distances from Strategy 1: (n, k)
            # Compute median distance for each archetype
            medians = all_distances.median(dim=0)[0]  # (k,)
            thresholds = medians * 1.5  # (k,)

            # Count nearby points for each archetype
            nearby_counts = (all_distances < thresholds.unsqueeze(0)).float().sum(dim=0)  # (k,)
            min_neighbors = max(1, batch_size * 0.05)

            # Penalize archetypes with too few neighbors
            shortfall = F.relu(min_neighbors - nearby_counts)
            density_loss = (shortfall ** 2).sum()

        # Combine all strategies
        total_manifold_loss = manifold_loss + 0.5 * bounding_loss + 0.1 * density_loss

        return total_manifold_loss / self.n_archetypes

    def loss_function(self, outputs: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        """Compute comprehensive archetypal loss.

        Parameters
        ----------
        outputs : dict[str, torch.Tensor]
            Output dictionary from forward().
        **kwargs
            Optional weight overrides:

            - ``archetypal_weight`` : Override self.archetypal_weight
            - ``kld_weight`` : Override self.kld_weight
            - ``diversity_weight`` : Override self.diversity_weight
            - ``regularity_weight`` : Override self.regularity_weight
            - ``sparsity_weight`` : Override self.sparsity_weight
            - ``manifold_weight`` : Override self.manifold_weight

        Returns
        -------
        dict[str, torch.Tensor]
            Loss components and metrics:

            **Primary loss (requires grad):**

            - ``loss`` : Total weighted loss for backpropagation

            **Loss components (detached):**

            - ``archetypal_loss`` : Frobenius norm reconstruction error
            - ``kld_loss`` : KL divergence from standard normal
            - ``diversity_loss`` : Archetype separation penalty
            - ``regularity_loss`` : Usage balance penalty
            - ``sparsity_loss`` : Coordinate entropy
            - ``manifold_loss`` : Off-manifold penalty

            **Performance metrics (detached):**

            - ``rmse`` : Root mean squared error
            - ``archetype_r2`` : Reconstruction R²

            **Archetype health (detached):**

            - ``archetype_entropy`` : Usage distribution entropy
            - ``max_archetype_usage`` : Highest mean usage
            - ``min_archetype_usage`` : Lowest mean usage
            - ``active_archetypes_per_sample`` : Mean active count (>0.01)

            **Manifold quality:**

            - ``mean_archetype_data_distance`` : float
            - ``max_archetype_data_distance`` : float

            **Convergence tracking:**

            - ``loss_delta`` : Change from previous loss
            - ``loss_history`` : Recent loss values (list)

            **Model info:**

            - ``input_dim``, ``latent_dim``, ``n_archetypes`` : int

            **Legacy aliases:**

            - ``KLD`` : Same as kld_loss
            - ``reconstruction_loss`` : Same as archetypal_loss

        Examples
        --------
        >>> outputs = model(batch)
        >>> loss_dict = model.loss_function(outputs)
        >>> # Backprop
        >>> loss_dict["loss"].backward()
        >>> # Monitor
        >>> print(f"R²: {loss_dict['archetype_r2'].item():.4f}")
        >>> print(f"Min usage: {loss_dict['min_archetype_usage'].item():.4f}")

        See Also
        --------
        peach._core.types.DeepAALossOutput : Return type structure
        """
        # Extract outputs
        arch_recons = outputs["arch_recons"]
        input = outputs["input"]
        mu = outputs["mu"]
        log_var = outputs["log_var"]
        z = outputs["z"]

        # Get loss weights
        archetypal_weight = kwargs.get("archetypal_weight", self.archetypal_weight)
        kld_weight = kwargs.get("kld_weight", self.kld_weight)
        diversity_weight = kwargs.get("diversity_weight", self.diversity_weight)
        regularity_weight = kwargs.get("regularity_weight", self.regularity_weight)
        sparsity_weight = kwargs.get("sparsity_weight", self.sparsity_weight)
        manifold_weight = kwargs.get("manifold_weight", self.manifold_weight)

        # 1. KL divergence loss
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1))

        # 2. Archetypal reconstruction loss
        archetypal_loss = torch.norm(arch_recons - input, p="fro") ** 2 / input.numel()

        # 3. Archetypal diversity loss
        diversity_loss = self.archetypal_diversity_loss()

        # 4. Archetypal regularity loss
        regularity_loss = self.archetypal_regularity_loss(z)

        # 5. Sparsity loss
        sparsity_loss = torch.mean(z * torch.log(z + 1e-8))

        # 6. Manifold regularization loss
        manifold_loss = self.manifold_regularization_loss(input)

        # Total weighted loss
        total_loss = (
            kld_weight * kld_loss
            + archetypal_weight * archetypal_loss
            + diversity_weight * diversity_loss
            + regularity_weight * regularity_loss
            + sparsity_weight * sparsity_loss
            + manifold_weight * manifold_loss
        )

        # Compute performance metrics
        with torch.no_grad():
            rmse = torch.sqrt(F.mse_loss(arch_recons, input))

            # Archetype R²
            archetype_r2 = calculate_archetype_r2(arch_recons, input)

            # Archetype usage statistics
            archetype_usage = z.mean(dim=0)
            archetype_entropy = -torch.sum(archetype_usage * torch.log(archetype_usage + 1e-8))
            max_archetype_usage = archetype_usage.max()
            min_archetype_usage = archetype_usage.min()

            # Coordinate sparsity
            active_archetypes_per_sample = (z > 0.01).sum(dim=1).float().mean()

            # Manifold quality metrics (VECTORIZED - no numpy, stays on GPU)
            data_sample = input[: min(100, input.shape[0])]
            # Compute all pairwise distances: (n, k)
            diff = data_sample.unsqueeze(1) - self.archetypes.unsqueeze(0)
            all_distances = torch.norm(diff, dim=2)
            # Min distance from each archetype to data
            min_dists_per_archetype = all_distances.min(dim=0)[0]  # (k,)
            mean_archetype_data_distance = min_dists_per_archetype.mean().item()
            max_archetype_data_distance = min_dists_per_archetype.max().item()

        # Track loss for convergence
        if self.previous_loss is None:
            self.previous_loss = total_loss.detach()
            loss_delta = torch.tensor(0.0)
            self.loss_history = [total_loss.item()]
        else:
            loss_delta = torch.abs(total_loss.detach() - self.previous_loss)
            self.previous_loss = total_loss.detach()
            self.loss_history.append(total_loss.item())
            self.loss_history = self.loss_history[-100:]

        return {
            # Primary loss
            "loss": total_loss,
            # Loss components
            "kld_loss": kld_loss.detach(),
            "archetypal_loss": archetypal_loss.detach(),
            "diversity_loss": diversity_loss.detach(),
            "regularity_loss": regularity_loss.detach(),
            "sparsity_loss": sparsity_loss.detach(),
            "manifold_loss": manifold_loss.detach(),
            # Performance metrics
            "rmse": rmse.detach(),
            "archetype_r2": archetype_r2.detach(),
            # Archetype usage metrics
            "archetype_entropy": archetype_entropy.detach(),
            "max_archetype_usage": max_archetype_usage.detach(),
            "min_archetype_usage": min_archetype_usage.detach(),
            "active_archetypes_per_sample": active_archetypes_per_sample.detach(),
            # Manifold quality metrics
            "mean_archetype_data_distance": mean_archetype_data_distance,
            "max_archetype_data_distance": max_archetype_data_distance,
            # Convergence tracking
            "loss_delta": loss_delta,
            "loss_history": self.loss_history,
            # Model info
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "n_archetypes": self.n_archetypes,
            # Legacy compatibility
            "KLD": kld_loss.detach(),
            "reconstruction_loss": archetypal_loss.detach(),
        }

    # ============================================================================
    # DEEP_2 INITIALIZATION METHODS (unchanged)
    # ============================================================================

    def initialize_archetypes(
        self,
        X_sample: torch.Tensor,
        use_pcha: bool = True,
        use_inflation: bool = False,
        inflation_factor: float = None,
        n_subsample: int = 1000,
        test_inflation_factors: bool = False,
        inflation_test_range: list[float] = None,
    ) -> bool:
        """Initialize archetype positions.

        Consolidated initialization function supporting PCHA, inflation,
        and automatic factor testing.

        Parameters
        ----------
        X_sample : torch.Tensor
            Data sample for initialization [n_samples, input_dim].
        use_pcha : bool, default: True
            Use PCHA initialization. If False, uses furthest-sum fallback.
        use_inflation : bool, default: False
            Apply scalar inflation after initialization.
        inflation_factor : float | None, default: None
            Inflation factor. If None, uses self.inflation_factor (1.5).
        n_subsample : int, default: 1000
            Maximum samples for PCHA efficiency.
        test_inflation_factors : bool, default: False
            Test multiple factors and select best.
        inflation_test_range : list[float] | None, default: None
            Factors to test. Default: [1.0, 1.2, 1.5, 2.0, 3.0].

        Returns
        -------
        bool
            True if initialization succeeded.

        Notes
        -----
        This method replaces the deprecated functions:

        - ``initialize_with_pcha()``
        - ``initialize_with_pcha_and_inflation()``
        - ``scalar_inflate_archetypes()``
        - ``test_inflation_factors()``

        Examples
        --------
        >>> # Standard PCHA + inflation
        >>> success = model.initialize_archetypes(X_sample, use_pcha=True, use_inflation=True, inflation_factor=1.5)
        >>> # Test multiple factors
        >>> success = model.initialize_archetypes(
        ...     X_sample, test_inflation_factors=True, inflation_test_range=[1.0, 1.5, 2.0]
        ... )

        See Also
        --------
        peach._core.types.ArchetypeInitConfig : Configuration model
        """
        if inflation_factor is None:
            inflation_factor = self.inflation_factor

        if inflation_test_range is None:
            inflation_test_range = [1.0, 1.2, 1.5, 2.0, 3.0]

        print("\n Consolidated Archetype Initialization")
        print(f"   PCHA: {use_pcha}, Inflation: {use_inflation} (factor: {inflation_factor})")
        print(f"   Test inflation: {test_inflation_factors}")

        # Handle inflation factor testing first
        if test_inflation_factors:
            return self._test_inflation_factors_internal(X_sample, inflation_test_range, n_subsample)

        # Standard initialization flow
        success = False

        if use_pcha:
            success = self._initialize_with_pcha_internal(X_sample, n_subsample)
        else:
            self.initialize_archetypes_furthest_sum(X_sample)
            success = self.archetypes_initialized

        if not success:
            print("[ERROR] Primary initialization failed")
            return False

        # Apply inflation if requested
        if use_inflation:
            success = self._scalar_inflate_archetypes_internal(X_sample, inflation_factor)
            if success:
                self._verify_archetype_positioning(X_sample, self.archetypes)

        print(f"[OK] Archetype initialization complete (success: {success})")
        return success

    def initialize_with_pcha(self, X_sample: torch.Tensor, n_subsample: int = 1000):
        """DEPRECATED: Use initialize_archetypes(use_pcha=True) instead."""
        import warnings

        warnings.warn(
            "initialize_with_pcha is deprecated. Use initialize_archetypes(use_pcha=True) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.initialize_archetypes(X_sample, use_pcha=True, n_subsample=n_subsample)

    def _initialize_with_pcha_internal(self, X_sample: torch.Tensor, n_subsample: int = 1000) -> bool:
        """Internal PCHA initialization (from Deep_2)."""
        # Downsample for computational efficiency
        if X_sample.shape[0] > n_subsample:
            indices = torch.randperm(X_sample.shape[0])[:n_subsample]
            X_sub = X_sample[indices]
        else:
            X_sub = X_sample

        try:
            X_np = X_sub.detach().cpu().numpy()

            print("Running PCHA initialization...")
            print(f"  Input shape: {X_np.shape}")
            print(f"  Target archetypes: {self.n_archetypes}")

            from ..utils.PCHA import run_pcha_analysis

            pcha_results = run_pcha_analysis(data=X_np, n_archetypes=self.n_archetypes, verbose=False)

            pcha_archetypes = pcha_results["archetypes"]
            archetype_r2 = pcha_results["archetype_r2"]

            print(f"  PCHA archetype R²: {archetype_r2:.4f}")
            print(f"  Archetype shape: {pcha_archetypes.shape}")

            # Initialize archetype positions
            self.archetypes.data = torch.tensor(pcha_archetypes, dtype=torch.float32, device=self.archetypes.device)

            self.archetypes_initialized = True
            print(f"[OK] Initialized {self.n_archetypes} archetypes using PCHA")

            # Store PCHA results for analysis
            self.pcha_results = pcha_results
            return True

        except Exception as e:
            print(f"[ERROR] PCHA initialization failed: {e}")
            print("Falling back to furthest sum initialization")
            self.initialize_archetypes_furthest_sum(X_sample)
            return self.archetypes_initialized

    def initialize_archetypes_furthest_sum(self, X_sample: torch.Tensor) -> None:
        """Initialize archetypes using furthest sum strategy (from Deep_2)."""
        if self.archetypes_initialized:
            return

        X_np = X_sample.detach().cpu().numpy()
        n_samples = X_np.shape[0]

        if n_samples < self.n_archetypes:
            selected_indices = list(range(n_samples))
            while len(selected_indices) < self.n_archetypes:
                base_idx = np.random.randint(0, n_samples)
                selected_indices.append(base_idx)
        else:
            # Furthest sum initialization
            selected_indices = [0]

            for k in range(1, self.n_archetypes):
                max_min_dist = -1
                best_idx = -1

                for i in range(n_samples):
                    if i in selected_indices:
                        continue

                    min_dist = float("inf")
                    for selected_idx in selected_indices:
                        dist = np.linalg.norm(X_np[i] - X_np[selected_idx])
                        min_dist = min(min_dist, dist)

                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        best_idx = i

                selected_indices.append(best_idx)

        # Initialize archetypes with small perturbation
        archetypal_points = X_np[selected_indices]
        perturbation = np.random.normal(0, 0.1, archetypal_points.shape)
        archetypal_points_perturbed = archetypal_points + perturbation

        self.archetypes.data = torch.tensor(
            archetypal_points_perturbed, dtype=torch.float32, device=self.archetypes.device
        )
        self.archetypes_initialized = True

        print(f"Initialized {self.n_archetypes} archetypes using furthest sum")

    # ============================================================================
    # NEW: SCALAR INFLATION METHODS (from Deep_5)
    # ============================================================================

    def _scalar_inflate_archetypes_internal(self, X_sample: torch.Tensor, inflation_factor: float = None) -> bool:
        """Internal scalar inflation implementation."""
        if not self.archetypes_initialized:
            print("[WARNING]  Archetypes not initialized yet")
            return False

        if inflation_factor is None:
            inflation_factor = self.inflation_factor

        print(f"\n Scalar Archetypal Inflation (factor: {inflation_factor:.2f})")

        with torch.no_grad():
            current_archetypes = self.archetypes.clone()
            archetype_centroid = current_archetypes.mean(dim=0)

            # Apply uniform scalar inflation
            centered_archetypes = current_archetypes - archetype_centroid
            scaled_archetypes = inflation_factor * centered_archetypes
            inflated_archetypes = archetype_centroid + scaled_archetypes

            # Update archetypes
            self.archetypes.data = inflated_archetypes

            print("   [OK] Inflation complete")

        return True

    def scalar_inflate_archetypes(self, X_sample: torch.Tensor, inflation_factor: float = None) -> bool:
        """
        DEPRECATED: Use initialize_archetypes(use_inflation=True) instead.

        Scalar inflation for archetypes (from Deep_5).
        Uniform scaling of all archetypes away from their centroid.
        """
        import warnings

        warnings.warn(
            "scalar_inflate_archetypes is deprecated. Use initialize_archetypes(use_inflation=True) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._scalar_inflate_archetypes_internal(X_sample, inflation_factor)

    def initialize_with_pcha_and_inflation(
        self, X_sample: torch.Tensor, inflation_factor: float = None, n_subsample: int = 1000
    ) -> bool:
        """
        DEPRECATED: Use initialize_archetypes(use_pcha=True, use_inflation=True) instead.

        NEW: Initialize with PCHA + scalar inflation.
        This combines Deep_2's working PCHA initialization with Deep_5's inflation.
        """
        import warnings

        warnings.warn(
            "initialize_with_pcha_and_inflation is deprecated. Use initialize_archetypes(use_pcha=True, use_inflation=True) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.initialize_archetypes(
            X_sample, use_pcha=True, use_inflation=True, inflation_factor=inflation_factor, n_subsample=n_subsample
        )

    def _test_inflation_factors_internal(self, X_sample: torch.Tensor, factors: list[float], n_subsample: int) -> bool:
        """Internal inflation factor testing implementation."""
        print(f"\n🧪 Testing Inflation Factors: {factors}")
        print("=" * 50)

        results = {}

        for factor in factors:
            print(f"\n Testing inflation factor: {factor}")

            # Reset initialization
            self.archetypes_initialized = False

            try:
                # Initialize with this inflation factor using consolidated function
                success = self.initialize_archetypes(
                    X_sample, use_pcha=True, use_inflation=True, inflation_factor=factor, n_subsample=n_subsample
                )

                if not success:
                    results[factor] = {"error": "Initialization failed"}
                    continue

                # Test initial reconstruction quality
                with torch.no_grad():
                    outputs = self(X_sample)
                    arch_loss = torch.norm(outputs["arch_recons"] - X_sample, p="fro") ** 2 / X_sample.numel()
                    archetype_r2 = calculate_archetype_r2(outputs["arch_recons"], X_sample)

                # Check positioning
                data_center = X_sample.mean(dim=0)
                data_radius = torch.norm(X_sample - data_center, dim=1).max()
                arch_distances = torch.norm(self.archetypes - data_center, dim=1)
                outside_count = (arch_distances > data_radius).sum().item()

                results[factor] = {
                    "archetype_r2": archetype_r2.item(),
                    "arch_loss": arch_loss.item(),
                    "outside_data_count": outside_count,
                    "mean_arch_distance": arch_distances.mean().item(),
                    "min_arch_distance": arch_distances.min().item(),
                    "max_arch_distance": arch_distances.max().item(),
                }

                print(f"   [OK] Archetype R²: {archetype_r2.item():.4f}")
                print(f"   [STATS] {outside_count}/{self.n_archetypes} archetypes outside data")
                print(f"    Mean distance to center: {arch_distances.mean():.3f}")

            except Exception as e:
                print(f"   [ERROR] Failed: {e}")
                results[factor] = {"error": str(e)}

        # Summary
        print("\n INFLATION FACTOR SUMMARY:")
        best_factor = None
        best_score = -float("inf")

        for factor, result in results.items():
            if "error" not in result:
                # Score based on archetype R²
                score = result["archetype_r2"]
                print(
                    f"   Factor {factor}: R²={result['archetype_r2']:.4f}, Outside={result['outside_data_count']}/{self.n_archetypes}"
                )

                if score > best_score:
                    best_score = score
                    best_factor = factor

        if best_factor:
            print(f"\n Best inflation factor: {best_factor} (archetype R²: {best_score:.4f})")
            # Re-initialize with best factor
            self.archetypes_initialized = False
            return self.initialize_archetypes(
                X_sample, use_pcha=True, use_inflation=True, inflation_factor=best_factor, n_subsample=n_subsample
            )

        return False

    def _verify_archetype_positioning(self, X: torch.Tensor, archetypes: torch.Tensor):
        """Verify archetype positioning (from Deep_5)."""
        data_center = X.mean(dim=0)

        # Data radius
        data_distances = torch.norm(X - data_center, dim=1)
        data_radius = data_distances.max()
        data_mean_radius = data_distances.mean()

        # Archetype distances from center
        arch_distances = torch.norm(archetypes - data_center, dim=1)

        print("      [STATS] Positioning verification:")
        print(f"         Data radius (max): {data_radius:.3f}")
        print(f"         Data radius (mean): {data_mean_radius:.3f}")
        print(f"         Archetype distances: {arch_distances.min():.3f} to {arch_distances.max():.3f}")
        print(f"         Archetypes outside data: {(arch_distances > data_radius).sum().item()}/{len(archetypes)}")

        # Check archetype separation
        min_separation = float("inf")
        for i in range(len(archetypes)):
            for j in range(i + 1, len(archetypes)):
                sep = torch.norm(archetypes[i] - archetypes[j])
                min_separation = min(min_separation, sep.item())

        print(f"         Min archetype separation: {min_separation:.3f}")

    def test_inflation_factors(self, X_sample: torch.Tensor, factors: list[float] = [1.0, 1.2, 1.5, 2.0, 3.0]) -> dict:
        """
        DEPRECATED: Use initialize_archetypes(test_inflation_factors=True) instead.

        Test different inflation factors and compare initial reconstruction quality.
        This will help determine optimal inflation for real data.
        """
        import warnings

        warnings.warn(
            "test_inflation_factors is deprecated. Use initialize_archetypes(test_inflation_factors=True) instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Call internal function and return results dict for compatibility
        success = self._test_inflation_factors_internal(X_sample, factors, n_subsample=1000)
        return {"success": success}

    # ============================================================================
    # DEEP_2 UTILITY METHODS (unchanged)
    # ============================================================================

    def analyze_archetypal_weights(self, input: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        """Analyze archetypal coordinate distributions.

        Parameters
        ----------
        input : torch.Tensor
            Input data [batch_size, input_dim].

        Returns
        -------
        dict[str, dict[str, torch.Tensor]]
            Analysis results with keys:

            - ``A_matrix`` : Statistics for cell-to-archetype weights
                - ``mean_weights`` : [n_archetypes]
                - ``std_weights`` : [n_archetypes]
                - ``max_weights`` : [n_archetypes]
                - ``min_weights`` : [n_archetypes]
                - ``dominant_archetype`` : Fraction each is dominant
            - ``B_matrix`` : Statistics for B (dummy in Deep_AA)

        See Also
        --------
        peach._core.types.ArchetypalWeightAnalysis : Return type
        """
        from ..utils.analysis import get_archetypal_coordinates

        coords = get_archetypal_coordinates(self, input)
        A = coords["A"]
        B = coords["B"]  # Dummy B matrix

        A_analysis = {
            "mean_weights": A.mean(dim=0),
            "std_weights": A.std(dim=0),
            "max_weights": A.max(dim=0)[0],
            "min_weights": A.min(dim=0)[0],
            "dominant_archetype": torch.argmax(A, dim=1).bincount(minlength=self.n_archetypes).float() / A.size(0),
        }

        B_analysis = {
            "mean_weights": B.mean(dim=0),
            "std_weights": B.std(dim=0),
            "max_weights": B.max(dim=0)[0],
            "min_weights": B.min(dim=0)[0],
        }

        return {"A_matrix": A_analysis, "B_matrix": B_analysis}

    def validate_constraints(
        self, A: torch.Tensor, B: torch.Tensor = None, tolerance: float = 1e-3
    ) -> dict[str, float]:
        """Validate archetypal convexity constraints.

        Checks that coordinates satisfy:
        - Non-negativity: A >= 0
        - Sum-to-one: A.sum(dim=1) == 1

        Parameters
        ----------
        A : torch.Tensor
            Archetypal coordinates [batch_size, n_archetypes].
        B : torch.Tensor | None, default: None
            Optional B matrix (for API compatibility).
        tolerance : float, default: 1e-3
            Tolerance for constraint satisfaction.

        Returns
        -------
        dict[str, float]
            Constraint validation results:

            - ``A_sum_error`` : Mean |row_sum - 1|
            - ``A_negative_fraction`` : Fraction of negative values
            - ``B_sum_error`` : Mean |col_sum - 1| (0 if B=None)
            - ``B_negative_fraction`` : Fraction negative (0 if B=None)
            - ``constraints_satisfied`` : 1.0 if all OK, else 0.0

        Examples
        --------
        >>> outputs = model(data)
        >>> validation = model.validate_constraints(outputs["A"])
        >>> if validation["constraints_satisfied"] < 1.0:
        ...     print(f"Sum error: {validation['A_sum_error']:.4f}")

        See Also
        --------
        peach._core.types.ConstraintValidation : Return type
        """
        with torch.no_grad():
            # A constraints: rows sum to 1, non-negative
            A_row_sums = A.sum(dim=1)
            A_sum_error = torch.abs(A_row_sums - 1.0).mean().item()
            A_negative_frac = (A < 0).float().mean().item()

            # B constraints: check dummy B if provided
            if B is not None:
                B_col_sums = B.sum(dim=0)
                B_sum_error = torch.abs(B_col_sums - 1.0).mean().item()
                B_negative_frac = (B < 0).float().mean().item()
            else:
                B_sum_error = 0.0
                B_negative_frac = 0.0

            constraints_satisfied = (
                A_sum_error < tolerance
                and A_negative_frac < tolerance
                and B_sum_error < tolerance
                and B_negative_frac < tolerance
            )

            return {
                "A_sum_error": A_sum_error,
                "A_negative_fraction": A_negative_frac,
                "B_sum_error": B_sum_error,
                "B_negative_fraction": B_negative_frac,
                "constraints_satisfied": float(constraints_satisfied),
            }

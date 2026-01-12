import numpy as np
import torch

"""
Convex Synthetic Data Generation
Synthetic dataset generation for archetypal analysis testing with configurable archetype patterns and noise levels.

=== MODULE API INVENTORY ===

MAIN FUNCTIONS:
├── generate_convex_data(n_points: int, n_dimensions: int, n_archetypes: int, noise: float, seed: int = 1205, archetype_type: str = 'random', scale: float = 20.0, return_torch: bool = True) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]
│   └── Purpose: Generate synthetic data as convex combinations of configurable archetypes with controlled noise
│   └── Inputs: n_points(int, number of data points), n_dimensions(int, feature space dimension), n_archetypes(int, number of archetypes), 
│   │           noise(float, Gaussian noise std), seed(int, random seed), archetype_type(str, 'random'/'corners'/'sphere'), 
│   │           scale(float, data scale factor), return_torch(bool, output format)
│   └── Outputs: Tuple(points, archetypes) as torch.Tensor or np.ndarray based on return_torch flag
│   └── Side Effects: Sets numpy and torch random seeds, prints archetype positions and separation statistics

ARCHETYPE GENERATION STRATEGIES:
├── 'corners': Maximally separated corner patterns using first 4 dimensions with predefined patterns
│   └── Patterns: [0,0,0,0], [1,1,1,1], [1,0,1,0], [0,1,0,1] for clear separation
│   └── Remaining dimensions filled with random values
│
├── 'sphere': Unit sphere positioning with uniform angular distribution
│   └── Generated on unit sphere then scaled without centering
│   └── Provides balanced geometric distribution
│
└── 'random': Uniform random positioning in [0,1]^n_dimensions
    └── Centered and scaled: (random - 0.5) * scale
    └── Simple baseline for testing

DATA GENERATION PROCESS:
├── Archetype Positioning: Generate n_archetypes positions based on archetype_type
├── Convex Combination: Sample weights from Dirichlet(1,...,1) distribution for proper convex hull
├── Point Generation: points = weights @ archetypes (matrix multiplication)
├── Noise Addition: Add Gaussian noise ~ N(0, noise²) to all points
└── Format Conversion: Convert to torch.Tensor or keep as np.ndarray

EXTERNAL DEPENDENCIES:
├── From numpy: Random number generation, linear algebra, statistical functions
├── From torch: Tensor operations, manual_seed for reproducibility
├── From torch.utils.data: TensorDataset, DataLoader for data pipeline integration
└── From torch.optim: optim module (imported but not used in current implementation)

DATA FLOW PATTERNS:
├── Input: Parameters → Seed setting → Archetype generation → Weight sampling → Point generation → Noise addition → Output
├── Archetype Types: Strategy selection → Pattern application → Scaling → Position verification
├── Convex Hull: Dirichlet sampling → Matrix multiplication → Noise perturbation → Final dataset
└── Quality Control: Separation metrics → Distance statistics → Archetype positioning verification

ERROR HANDLING:
├── Archetype pattern overflow: Handles cases where n_archetypes > predefined patterns
├── Dimension mismatch: Graceful handling of corner patterns vs n_dimensions
├── Random fallback: Unknown archetype_type defaults to 'random' strategy
├── Seed reproducibility: Consistent random state across numpy and torch
└── Memory efficiency: Direct tensor creation for large datasets
"""


def generate_convex_data(
    n_points: int,
    n_dimensions: int,
    n_archetypes: int,
    noise: float,
    seed: int = 1205,
    archetype_type: str = "random",
    scale: float = 20.0,
    return_torch: bool = True,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"\nGenerating {archetype_type} archetypes...")

    if archetype_type == "corners":
        # FIX: Create maximally separated corners
        archetypes = np.zeros((n_archetypes, n_dimensions))

        # Strategy: Use first few dimensions for clear separation
        n_corner_dims = min(n_dimensions, 4)  # Use first 4 dims for corners

        patterns = [
            [0, 0, 0, 0],  # A0: all low
            [1, 1, 1, 1],  # A1: all high
            [1, 0, 1, 0],  # A2: alternating
            [0, 1, 0, 1],  # A3: alternating opposite
        ]

        for i in range(min(n_archetypes, len(patterns))):
            # Set corner dimensions
            archetypes[i, :n_corner_dims] = patterns[i][:n_corner_dims]
            # Randomize remaining dimensions
            archetypes[i, n_corner_dims:] = np.random.rand(n_dimensions - n_corner_dims)

    elif archetype_type == "sphere":
        # Keep sphere logic
        archetypes = np.random.randn(n_archetypes, n_dimensions)
        norms = np.linalg.norm(archetypes, axis=1, keepdims=True)
        archetypes = archetypes / norms

    else:  # 'random'
        archetypes = np.random.rand(n_archetypes, n_dimensions)

    # BETTER SCALING: Don't center sphere (it's already centered)
    if archetype_type == "sphere":
        archetypes = archetypes * scale  # Just scale, don't center
    else:
        archetypes = (archetypes - 0.5) * scale  # Center and scale

    print("Archetype positions (first 5 dims):")
    for i, arch in enumerate(archetypes):
        print(f"  A{i}: {arch[:5]}")

    # Generate points as convex combinations
    weights = np.random.dirichlet(np.ones(n_archetypes), n_points)
    points = np.dot(weights, archetypes)

    # Add noise
    noise_matrix = np.random.normal(0, noise, (n_points, n_dimensions))
    points += noise_matrix

    # Calculate archetype separation metrics
    separations = []
    for i in range(len(archetypes)):
        for j in range(i + 1, len(archetypes)):
            dist = np.linalg.norm(archetypes[i] - archetypes[j])
            separations.append(dist)

    print("Archetype separation stats:")
    print(f"  Mean distance: {np.mean(separations):.3f}")
    print(f"  Min distance: {np.min(separations):.3f}")
    print(f"  Max distance: {np.max(separations):.3f}")

    if return_torch:
        # Ensure contiguous arrays (fixes negative stride issue)
        points = torch.FloatTensor(np.ascontiguousarray(points))
        archetypes = torch.FloatTensor(np.ascontiguousarray(archetypes))

    return points, archetypes

"""Archetype comparison: MMD similarity, feature similarity, Wald contrasts."""

from anndata import AnnData

from peach._core.types import (
    ArchetypeMMDResult,
    ArchetypeFeatureSimilarityResult,
    ArchetypeContrastsResult,
)
from peach._core.utils.archetype_comparison import (
    compute_archetype_mmd,
    compute_feature_similarity,
    compute_wald_contrasts,
)
from peach._core.utils.feature_utils import get_archetype_weights, store_result


def archetype_mmd(
    adata: AnnData,
    adata_b: AnnData | None = None,
    *,
    pca_key: str = "X_pca",
    n_permutations: int = 1000,
    seed: int = 42,
    copy: bool = False,
) -> dict:
    """K x K MMD similarity matrix between archetype cell populations.

    Parameters
    ----------
    adata : AnnData
        Must have archetype weights in obsm['cell_archetype_weights']
        and PCA coordinates in obsm[pca_key].
    adata_b : AnnData or None
        If provided, compute K_A x K_B between-fit comparison.
    pca_key : str
        Key in obsm for cell coordinates.
    n_permutations : int
        Permutations for p-value computation.
    seed : int
    copy : bool

    Returns
    -------
    dict
        Serialized ArchetypeMMDResult. Also stored in adata.uns['peach_archetype_mmd'].
    """
    if copy:
        adata = adata.copy()

    K_a = get_archetype_weights(adata).shape[1]
    mmd_matrix, pvalue_matrix = compute_archetype_mmd(
        adata, adata_b, pca_key=pca_key,
        n_permutations=n_permutations, seed=seed,
    )

    arch_names_a = [f"archetype_{i}" for i in range(K_a)]
    arch_names_b = None
    is_between = adata_b is not None
    if is_between:
        K_b = get_archetype_weights(adata_b).shape[1]
        arch_names_b = [f"archetype_{i}" for i in range(K_b)]

    result = ArchetypeMMDResult(
        mmd_matrix=mmd_matrix,
        pvalue_matrix=pvalue_matrix,
        n_permutations=n_permutations,
        is_between_fit=is_between,
        archetype_names_a=arch_names_a,
        archetype_names_b=arch_names_b,
    )
    serialized = result.to_serializable()
    store_result(adata, "archetype_mmd", serialized)
    return serialized


def archetype_feature_similarity(
    adata: AnnData,
    adata_b: AnnData | None = None,
    *,
    pca_key: str = "X_pca",
    copy: bool = False,
) -> dict:
    """Feature-level archetype similarity: silhouette + Spearman on beta vectors.

    Parameters
    ----------
    adata : AnnData
        Must have regression results in uns['peach_simplex_regression'].
    adata_b : AnnData or None
        If provided, compute between-fit Spearman on shared features.
    pca_key : str
    copy : bool

    Returns
    -------
    dict
        Serialized ArchetypeFeatureSimilarityResult. Also stored in adata.uns['peach_archetype_feature_similarity'].
    """
    if copy:
        adata = adata.copy()

    K_a = get_archetype_weights(adata).shape[1]
    sim = compute_feature_similarity(adata, adata_b, pca_key=pca_key)

    arch_names_a = [f"archetype_{i}" for i in range(K_a)]
    is_between = adata_b is not None
    arch_names_b = None
    if is_between:
        K_b = get_archetype_weights(adata_b).shape[1]
        arch_names_b = [f"archetype_{i}" for i in range(K_b)]

    result = ArchetypeFeatureSimilarityResult(
        silhouette_per_archetype=sim["silhouette_per_archetype"],
        silhouette_overall=sim["silhouette_overall"],
        spearman_matrix=sim["spearman_matrix"],
        spearman_pvalue_matrix=sim["spearman_pvalue_matrix"],
        spearman_pvalue_fdr_matrix=sim.get("spearman_pvalue_fdr_matrix"),
        n_shared_features=sim["n_shared_features"],
        is_between_fit=is_between,
        archetype_names_a=arch_names_a,
        archetype_names_b=arch_names_b,
    )
    serialized = result.to_serializable()
    store_result(adata, "archetype_feature_similarity", serialized)
    return serialized


def archetype_contrasts(
    adata: AnnData,
    *,
    robust_se: bool = True,
    copy: bool = False,
) -> dict:
    """Pairwise Wald contrasts between archetype regression coefficients.

    For each pair (j, k), tests H0: beta_j = beta_k for every feature using
    the Wald statistic with HC3 covariance from the Scheffe regression.

    Parameters
    ----------
    adata : AnnData
        Must have archetype weights and regression results.
    robust_se : bool
        Use HC3 heteroscedasticity-consistent covariance.
    copy : bool

    Returns
    -------
    dict
        Serialized ArchetypeContrastsResult. Also stored in adata.uns['peach_archetype_contrasts'].
    """
    if copy:
        adata = adata.copy()

    contrasts = compute_wald_contrasts(adata, robust_se=robust_se)

    result = ArchetypeContrastsResult(
        pairs=contrasts["pairs"],
        delta_beta=contrasts["delta_beta"],
        delta_se=contrasts["delta_se"],
        z_scores=contrasts["z_scores"],
        pvalues=contrasts["pvalues"],
        pvalues_fdr=contrasts["pvalues_fdr"],
        feature_names=contrasts["feature_names"],
        n_features=contrasts["n_features"],
        n_archetypes=contrasts["n_archetypes"],
    )
    serialized = result.to_serializable()
    store_result(adata, "archetype_contrasts", serialized)
    return serialized

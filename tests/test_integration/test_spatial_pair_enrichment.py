import numpy as np
import pytest
from anndata import AnnData
import scipy.sparse as sp


@pytest.fixture
def spatial_adata():
    """AnnData with spatial coords, connectivity, and archetype weights."""
    rng = np.random.default_rng(42)
    n = 200
    K = 3

    weights = rng.dirichlet([1] * K, size=n)
    X = rng.standard_normal((n, 20))
    adata = AnnData(X)
    adata.obsm["cell_archetype_weights"] = weights
    adata.obsm["spatial"] = rng.standard_normal((n, 2)) * 100

    # Create fake spatial connectivity (k-nearest neighbors)
    from scipy.spatial.distance import cdist
    coords = adata.obsm["spatial"]
    dists = cdist(coords, coords)
    conn = np.zeros_like(dists)
    for i in range(n):
        nearest = np.argsort(dists[i])[:11]  # 10 nearest + self
        conn[i, nearest] = 1
        conn[nearest, i] = 1
    np.fill_diagonal(conn, 0)
    adata.obsp["spatial_connectivities"] = sp.csr_matrix(conn)

    return adata


class TestArchetypePairEnrichment:
    def test_basic_run(self, spatial_adata):
        """Runs on spatial adata with archetype weights."""
        from peach.tl.spatial import archetype_pair_enrichment

        result = archetype_pair_enrichment(
            spatial_adata, n_permutations=50
        )
        assert isinstance(result, dict)
        assert len(result) == 3  # K*(K-1)/2 = 3

    def test_pvalues_bounded(self, spatial_adata):
        """P-values in [0, 1]."""
        from peach.tl.spatial import archetype_pair_enrichment

        result = archetype_pair_enrichment(
            spatial_adata, n_permutations=50
        )
        for pair, res in result.items():
            assert 0 <= res["p_value"] <= 1

    def test_weight_threshold_filtering(self, spatial_adata):
        """Higher threshold means fewer participating cells."""
        from peach.tl.spatial import archetype_pair_enrichment

        result_low = archetype_pair_enrichment(
            spatial_adata, weight_threshold=0.1, n_permutations=20
        )
        result_high = archetype_pair_enrichment(
            spatial_adata, weight_threshold=0.7, n_permutations=20
        )
        # Higher threshold should have fewer cells
        for pair in result_low:
            assert result_high[pair]["n_cells_i"] <= result_low[pair]["n_cells_i"]

    def test_all_pairs_enumerated(self, spatial_adata):
        """'all' generates K*(K-1)/2 pairs."""
        from peach.tl.spatial import archetype_pair_enrichment

        result = archetype_pair_enrichment(
            spatial_adata, archetype_pairs="all", n_permutations=20
        )
        K = 3
        assert len(result) == K * (K - 1) // 2

    def test_stored_in_adata(self, spatial_adata):
        """Results stored in adata.uns['peach_pair_enrichment']."""
        from peach.tl.spatial import archetype_pair_enrichment

        archetype_pair_enrichment(spatial_adata, n_permutations=20)
        assert "peach_pair_enrichment" in spatial_adata.uns

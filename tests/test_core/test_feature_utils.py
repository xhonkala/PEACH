import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData


class TestResolveFeatures:
    """Test resolve_features() input resolution."""

    def test_default_uses_adata_x_dense(self):
        """feature_matrix=None -> adata.X, feature_names from var_names."""
        from peach._core.utils.feature_utils import resolve_features

        X = np.random.rand(100, 50)
        adata = AnnData(X, var={"gene": [f"gene_{i}" for i in range(50)]})
        adata.var_names = [f"gene_{i}" for i in range(50)]
        mat, names = resolve_features(adata)
        assert isinstance(mat, np.ndarray)
        assert mat.shape == (100, 50)
        assert len(names) == 50
        assert names[0] == "gene_0"

    def test_default_uses_adata_x_sparse(self):
        """Sparse adata.X stays sparse — never densified."""
        from peach._core.utils.feature_utils import resolve_features

        X = sp.random(100, 50, density=0.3, format="csr")
        adata = AnnData(X)
        mat, names = resolve_features(adata)
        assert sp.issparse(mat)
        assert mat.shape == (100, 50)

    def test_obsm_key_string(self):
        """feature_matrix='pathway_scores' -> adata.obsm['pathway_scores']."""
        from peach._core.utils.feature_utils import resolve_features

        adata = AnnData(np.zeros((100, 10)))
        adata.obsm["pathway_scores"] = np.random.rand(100, 20)
        mat, names = resolve_features(adata, feature_matrix="pathway_scores")
        assert mat.shape == (100, 20)
        assert len(names) == 20

    def test_direct_array_passthrough(self):
        """feature_matrix=np.ndarray -> use directly."""
        from peach._core.utils.feature_utils import resolve_features

        adata = AnnData(np.zeros((100, 10)))
        custom = np.random.rand(100, 30)
        custom_names = [f"feat_{i}" for i in range(30)]
        mat, names = resolve_features(adata, feature_matrix=custom, feature_names=custom_names)
        np.testing.assert_array_equal(mat, custom)
        assert names == custom_names

    def test_missing_obsm_key_raises(self):
        """Missing obsm key raises KeyError."""
        from peach._core.utils.feature_utils import resolve_features

        adata = AnnData(np.zeros((100, 10)))
        with pytest.raises(KeyError):
            resolve_features(adata, feature_matrix="nonexistent_key")

    def test_shape_mismatch_raises(self):
        """Array with wrong n_cells raises ValueError."""
        from peach._core.utils.feature_utils import resolve_features

        adata = AnnData(np.zeros((100, 10)))
        wrong_shape = np.random.rand(50, 10)
        with pytest.raises(ValueError, match="n_cells"):
            resolve_features(adata, feature_matrix=wrong_shape)


class TestGetArchetypeWeights:
    """Test get_archetype_weights() extraction + validation."""

    def test_valid_weights(self):
        """Weights summing to 1 are returned."""
        from peach._core.utils.feature_utils import get_archetype_weights

        adata = AnnData(np.zeros((100, 10)))
        weights = np.random.dirichlet([1] * 4, size=100)
        adata.obsm["cell_archetype_weights"] = weights
        result = get_archetype_weights(adata)
        np.testing.assert_array_almost_equal(result, weights)

    def test_missing_weights_raises(self):
        """Missing weights key raises KeyError."""
        from peach._core.utils.feature_utils import get_archetype_weights

        adata = AnnData(np.zeros((100, 10)))
        with pytest.raises(KeyError):
            get_archetype_weights(adata)

    def test_bad_sum_raises(self):
        """Weights not summing to 1 raises ValueError — never silently renormalizes."""
        from peach._core.utils.feature_utils import get_archetype_weights

        rng = np.random.default_rng(42)
        adata = AnnData(np.zeros((100, 10)))
        weights = rng.random((100, 4))  # won't sum to 1
        adata.obsm["cell_archetype_weights"] = weights
        with pytest.raises(ValueError, match="sum-to-1"):
            get_archetype_weights(adata)

    def test_nan_weights_raises(self):
        """NaN in weights raises ValueError."""
        from peach._core.utils.feature_utils import get_archetype_weights

        rng = np.random.default_rng(42)
        adata = AnnData(np.zeros((100, 10)))
        weights = rng.dirichlet([1, 1, 1], size=100)
        weights[0, 0] = np.nan
        adata.obsm["cell_archetype_weights"] = weights
        with pytest.raises(ValueError, match="NaN"):
            get_archetype_weights(adata)

    def test_negative_weights_raises(self):
        """Negative weights raise ValueError."""
        from peach._core.utils.feature_utils import get_archetype_weights

        adata = AnnData(np.zeros((100, 10)))
        weights = np.array([[0.5, 0.7, -0.2]] * 100)  # sum=1 but negative
        adata.obsm["cell_archetype_weights"] = weights
        with pytest.raises(ValueError, match="negative"):
            get_archetype_weights(adata)


class TestStoreResult:
    """Test store_result() storage helper."""

    def test_stores_in_uns(self):
        """Result stored with peach_ prefix in uns."""
        from peach._core.utils.feature_utils import store_result

        adata = AnnData(np.zeros((10, 5)))
        store_result(adata, "simplex_regression", {"r2": 0.5})
        assert "peach_simplex_regression" in adata.uns
        assert adata.uns["peach_simplex_regression"]["r2"] == 0.5

    def test_stores_in_obsm(self):
        """Result stored in obsm when domain='obsm'."""
        from peach._core.utils.feature_utils import store_result

        adata = AnnData(np.zeros((10, 5)))
        arr = np.zeros(10)
        store_result(adata, "gmm_labels", arr, domain="obsm")
        assert "peach_gmm_labels" in adata.obsm

    def test_overwrite_warns(self, caplog):
        """Overwriting existing result logs warning."""
        import logging
        from peach._core.utils.feature_utils import store_result

        adata = AnnData(np.zeros((10, 5)))
        store_result(adata, "test_key", {"v": 1})
        with caplog.at_level(logging.WARNING):
            store_result(adata, "test_key", {"v": 2})
        assert "overwriting" in caplog.text.lower()

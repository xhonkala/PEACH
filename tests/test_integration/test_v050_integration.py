"""Lightweight integration test for v0.5.0 continuous characterization pipeline."""

import numpy as np
import pytest
from anndata import AnnData


@pytest.fixture
def v050_adata():
    """Synthetic AnnData with everything needed for v0.5.0 pipeline."""
    rng = np.random.default_rng(42)
    K = 3
    n = 300
    n_genes = 30

    weights = rng.dirichlet([1] * K, size=n)
    true_beta = rng.standard_normal((n_genes, K)) * 5
    true_beta[0] = [10.0, 0.0, 0.0]  # exclusive
    true_beta[1] = [3.0, 3.0, 3.0]  # flat
    noise = rng.normal(0, 0.3, size=(n, n_genes))
    X = weights @ true_beta.T + noise

    adata = AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obsm["cell_archetype_weights"] = weights
    adata.obsm["X_pca"] = rng.standard_normal((n, 10))
    adata.varm["PCs"] = rng.standard_normal((n_genes, 10))
    return adata


class TestV050Pipeline:
    def test_regression_pipeline(self, v050_adata):
        """Simplex regression -> classification -> summary."""
        import peach as pc

        # Regression
        reg = pc.tl.feature_simplex_regression(v050_adata, n_bootstrap=0)
        assert reg.vertex_coefficients.shape == (30, 3)
        assert len(reg.r_squared_degree1) == 30
        assert "peach_simplex_regression" in v050_adata.uns

        # Classification
        patterns = pc.tl.classify_feature_patterns(v050_adata)
        assert sum(patterns.pattern_counts.values()) == 30
        assert patterns.classifications[0]["pattern"] == "archetype-exclusive"
        assert "peach_feature_patterns" in v050_adata.uns

        # Summary
        summaries = pc.tl.archetype_summary(v050_adata)
        assert len(summaries) == 3
        assert "top_enriched" in summaries[0]
        assert "top_depleted" in summaries[0]

    def test_driver_regression(self, v050_adata):
        """Driver regression: features predict archetype weights."""
        import peach as pc

        driver = pc.tl.archetype_driver_regression(
            v050_adata, n_bootstrap=0, max_degree=1
        )
        assert driver.main_coefficients_ilr.shape == (2, 30)  # K-1=2 ILR dims
        assert driver.main_coefficients.shape == (3, 30)  # K=3 simplex
        assert len(driver.r_squared) == 2
        assert "peach_driver_regression" in v050_adata.uns

    def test_gmm_pipeline(self, v050_adata):
        """GMM decomposition."""
        import peach as pc

        gmm = pc.tl.feature_simplex_decomposition(
            v050_adata,
            n_initializations=3,
            n_components_range=(2, 4),
        )
        assert gmm.n_components_stable >= 1
        assert len(gmm.component_assignments) == 300
        assert "peach_gmm" in v050_adata.uns
        assert "peach_gmm_labels" in v050_adata.obsm

    def test_regression_viz(self, v050_adata):
        """Regression visualization suite."""
        import peach as pc

        pc.tl.feature_simplex_regression(v050_adata, n_bootstrap=0, max_degree=2)
        pc.tl.classify_feature_patterns(v050_adata)

        fig1 = pc.pl.coefficient_heatmap(v050_adata, show=False)
        fig2 = pc.pl.r2_barplot(v050_adata, show=False)
        fig3 = pc.pl.pattern_summary(v050_adata, show=False)
        assert fig1 is not None
        assert fig2 is not None
        assert fig3 is not None

    def test_ternary_viz(self, v050_adata):
        """Ternary plot."""
        import peach as pc

        fig = pc.pl.ternary_facet(v050_adata, archetypes=(0, 1, 2), show=False)
        assert fig is not None

    def test_gmm_viz(self, v050_adata):
        """GMM visualization suite."""
        import peach as pc

        pc.tl.feature_simplex_decomposition(
            v050_adata,
            n_initializations=3,
            n_components_range=(2, 4),
        )

        fig1 = pc.pl.component_scatter(v050_adata, show=False)
        fig2 = pc.pl.gmm_bic_curve(v050_adata, show=False)
        fig3 = pc.pl.component_stability(v050_adata, show=False)
        assert fig1 is not None
        assert fig2 is not None
        assert fig3 is not None

    def test_types_index_consistency(self):
        """Verify types_index covers all v0.5.0 functions."""
        from peach._core.types_index import FUNCTION_RETURNS

        v050_tl = [
            "tl.feature_simplex_regression",
            "tl.gene_simplex_regression",
            "tl.pathway_simplex_regression",
            "tl.classify_feature_patterns",
            "tl.archetype_driver_regression",
            "tl.feature_simplex_decomposition",
            "tl.archetype_summary",
            "tl.flow_within",
            "tl.flow_between",
            "tl.flow_gene_alignment",
            "tl.flow_jacobian",
            "tl.flow_significance",
            "tl.archetype_pair_enrichment",
        ]
        v050_pl = [
            "pl.ternary_facet",
            "pl.ternary_facet_grid",
            "pl.coefficient_heatmap",
            "pl.interaction_heatmap",
            "pl.r2_barplot",
            "pl.vertex_radar",
            "pl.regression_volcano",
            "pl.pattern_summary",
            "pl.component_scatter",
            "pl.gmm_bic_curve",
            "pl.component_heatmap",
            "pl.component_stability",
            "pl.velocity_quiver",
            "pl.gene_alignment_barplot",
            "pl.jacobian_heatmap",
            "pl.trajectory_ribbon",
            "pl.flow_magnitude",
            "pl.density_comparison",
            "pl.archetype_correspondence",
        ]

        missing = []
        for func in v050_tl + v050_pl:
            if func not in FUNCTION_RETURNS:
                missing.append(func)
        assert missing == [], f"Missing from FUNCTION_RETURNS: {missing}"

    def test_tools_schema_consistency(self):
        """Verify tools_schema covers key v0.5.0 functions."""
        from peach._core.tools_schema import TOOL_SCHEMAS

        required_schemas = [
            "tl.feature_simplex_regression",
            "tl.classify_feature_patterns",
            "tl.archetype_driver_regression",
            "tl.feature_simplex_decomposition",
            "tl.flow_within",
            "tl.archetype_summary",
        ]

        missing = []
        for func in required_schemas:
            if func not in TOOL_SCHEMAS:
                missing.append(func)
        assert missing == [], f"Missing from TOOL_SCHEMAS: {missing}"

    def test_use_get_for_coverage(self):
        """Verify USE_GET_FOR includes v0.5.0 optional fields."""
        from peach._core.types_index import USE_GET_FOR

        expected = {
            "interaction_coefficients",
            "interaction_pvalues",
            "interaction_se",
            "r_squared_degree2",
            "vertex_ci_lower",
            "vertex_ci_upper",
            "interaction_ci_lower",
            "interaction_ci_upper",
            "component_feature_profiles",
            "archetype_correspondence",
        }

        missing = expected - USE_GET_FOR
        assert missing == set(), f"Missing from USE_GET_FOR: {missing}"

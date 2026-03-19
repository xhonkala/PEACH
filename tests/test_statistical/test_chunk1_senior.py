"""Senior adversarial review tests for Chunk 1: flow module fixes.

Tests target:
- Boundary conditions (t=0.0, t=1.0) in flow_gene_alignment
- t=None vs t=0.5 with model present (must differ)
- flow_significance with negative improvement (mmd_after > mmd_before)
- compute_mmd with exactly 2 points
- velocity_mode key correctness
- per_cell alignment with t parameter
- OT-CFM reproducibility across polluted global numpy state
- Centrality values differ between sparse and dense adjacency
- tools_schema.py consistency with actual function signatures
"""

import numpy as np
import pytest
import torch
import warnings


# ---------------------------------------------------------------------------
# Shared fixture: small trained flow with model
# ---------------------------------------------------------------------------


def _make_flow_fixture(n_cells=100, n_genes=50, n_pcs=10, seed=42, return_model=True):
    """Build AnnData + trained flow, optionally with model."""
    import anndata as ad
    from peach._core.utils.flow_matching import FlowModel

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
    adata = ad.AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    # Source centered at 0, target shifted by +3 so the flow is obvious
    pca_source = rng.standard_normal((n_cells // 2, n_pcs)).astype(np.float32)
    pca_target = (rng.standard_normal((n_cells // 2, n_pcs)) + 3.0).astype(np.float32)
    adata.obsm["X_pca"] = np.vstack([pca_source, pca_target])
    adata.varm["PCs"] = rng.standard_normal((n_genes, n_pcs)).astype(np.float32)
    adata.obs["condition"] = (
        ["source"] * (n_cells // 2) + ["target"] * (n_cells - n_cells // 2)
    )

    import peach as pc

    flow_result = pc.tl.flow_within(
        adata,
        source={"condition": "source"},
        target={"condition": "target"},
        n_epochs=80,
        hidden_dims=(32, 32),
        return_model=return_model,
        random_state=seed,
    )
    return adata, flow_result


# ===========================================================================
# 1. Boundary conditions: t=0.0 and t=1.0
# ===========================================================================


class TestBoundaryT:
    """t=0.0 and t=1.0 are the extremes of the flow.  The network was
    trained over the entire [0, 1] interval, so it must return finite,
    non-trivial velocities at both boundaries."""

    def test_t0_finite_and_nonzero(self):
        adata, fr = _make_flow_fixture()
        import peach as pc

        result = pc.tl.flow_gene_alignment(adata, fr, t=0.0)
        scores = result["alignment_scores"]
        assert np.all(np.isfinite(scores)), "NaN/Inf at t=0.0"
        assert not np.allclose(scores, 0, atol=1e-10), "Scores all zero at t=0.0"
        assert result["velocity_mode"] == "instantaneous"

    def test_t1_finite_and_nonzero(self):
        adata, fr = _make_flow_fixture()
        import peach as pc

        result = pc.tl.flow_gene_alignment(adata, fr, t=1.0)
        scores = result["alignment_scores"]
        assert np.all(np.isfinite(scores)), "NaN/Inf at t=1.0"
        assert not np.allclose(scores, 0, atol=1e-10), "Scores all zero at t=1.0"
        assert result["velocity_mode"] == "instantaneous"

    def test_t0_and_t1_differ(self):
        """The velocity field is time-dependent so t=0 and t=1 must
        produce different alignment scores."""
        adata, fr = _make_flow_fixture()
        import peach as pc

        r0 = pc.tl.flow_gene_alignment(adata, fr, t=0.0)
        r1 = pc.tl.flow_gene_alignment(adata, fr, t=1.0)
        assert not np.allclose(
            r0["alignment_scores"], r1["alignment_scores"], atol=1e-6
        ), "t=0.0 and t=1.0 gave identical alignment — network is not time-dependent?"


# ===========================================================================
# 2. t=None vs explicit t — must differ when model is present
# ===========================================================================


class TestTNoneVsExplicit:
    """t=None uses full-trajectory displacement (transported - source).
    t=0.5 uses instantaneous velocity from the model.  These are
    fundamentally different quantities and must produce different results."""

    def test_none_vs_half(self):
        adata, fr = _make_flow_fixture()
        import peach as pc

        r_none = pc.tl.flow_gene_alignment(adata, fr, t=None)
        r_half = pc.tl.flow_gene_alignment(adata, fr, t=0.5)

        assert r_none["velocity_mode"] == "displacement"
        assert r_half["velocity_mode"] == "instantaneous"
        assert not np.allclose(
            r_none["alignment_scores"], r_half["alignment_scores"], atol=1e-6
        ), "t=None and t=0.5 gave identical alignment — displacement and velocity should differ"

    def test_none_is_default(self):
        """Calling with no t argument should behave identically to t=None."""
        adata, fr = _make_flow_fixture()
        import peach as pc

        r_default = pc.tl.flow_gene_alignment(adata, fr)
        r_none = pc.tl.flow_gene_alignment(adata, fr, t=None)

        np.testing.assert_array_equal(
            r_default["alignment_scores"], r_none["alignment_scores"]
        )
        assert r_default["velocity_mode"] == "displacement"


# ===========================================================================
# 3. flow_significance with negative improvement (bad transport)
# ===========================================================================


class TestNegativeImprovement:
    """If mmd_after > mmd_before the transport WORSENED the distribution
    match.  observed_improvement < 0.  The p-value should be high (close
    to 1) because no null permutation would do worse than random."""

    def test_negative_improvement_high_pvalue(self):
        import anndata as ad
        from peach.tl.flow import flow_significance

        rng = np.random.default_rng(99)
        n = 100
        dim = 5
        pca = rng.standard_normal((2 * n, dim)).astype(np.float32)
        adata = ad.AnnData(rng.standard_normal((2 * n, 10)))
        adata.obsm["X_pca"] = pca
        adata.obs["group"] = ["A"] * n + ["B"] * n

        source_mask = np.array([True] * n + [False] * n)
        target_mask = ~source_mask

        # Fabricate a flow_result where transport made things worse
        flow_result = {
            "source_mask": source_mask,
            "target_mask": target_mask,
            "transported": pca[:n] * 100,  # wildly wrong transport
            "pca_key": "X_pca",
            "mmd_before": 0.5,
            "mmd_after": 2.0,  # WORSE than before
        }

        sig = flow_significance(
            adata,
            flow_result,
            n_permutations=10,
            n_epochs_per_perm=20,
            hidden_dims=(16, 16),
            n_steps=5,
        )
        # observed_improvement = 0.5 - 2.0 = -1.5
        assert sig["observed_stat"] < 0, (
            f"Expected negative observed_stat, got {sig['observed_stat']}"
        )
        # p-value should be high because null permutations should mostly do
        # better than a negative improvement
        assert sig["p_value"] > 0.5, (
            f"Expected p_value > 0.5 for worsened transport, got {sig['p_value']}"
        )

    def test_observed_stat_matches_flow_result(self):
        """The observed_stat must exactly match mmd_before - mmd_after
        from the supplied flow_result, not some retrained value."""
        import anndata as ad
        from peach.tl.flow import flow_significance

        rng = np.random.default_rng(42)
        n = 100
        dim = 5
        pca = rng.standard_normal((2 * n, dim)).astype(np.float32)
        adata = ad.AnnData(rng.standard_normal((2 * n, 10)))
        adata.obsm["X_pca"] = pca
        adata.obs["group"] = ["A"] * n + ["B"] * n

        source_mask = np.array([True] * n + [False] * n)
        target_mask = ~source_mask

        mmd_before = 1.234
        mmd_after = 0.567
        flow_result = {
            "source_mask": source_mask,
            "target_mask": target_mask,
            "transported": pca[:n],
            "pca_key": "X_pca",
            "mmd_before": mmd_before,
            "mmd_after": mmd_after,
        }

        sig = flow_significance(
            adata,
            flow_result,
            n_permutations=3,
            n_epochs_per_perm=10,
            hidden_dims=(16, 16),
            n_steps=5,
        )
        expected = mmd_before - mmd_after
        assert abs(sig["observed_stat"] - expected) < 1e-12, (
            f"observed_stat={sig['observed_stat']} != expected={expected}"
        )


# ===========================================================================
# 4. compute_mmd with exactly 2 points (boundary of the < 2 guard)
# ===========================================================================


class TestMMDBoundary:
    def test_exactly_2_points_returns_finite(self):
        """With exactly 2 points the unbiased estimator uses n*(n-1)=2
        in the denominator.  Must return a finite float, not NaN."""
        from peach._core.utils.flow_matching import compute_mmd

        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        Y = np.array([[2.0, 2.0], [3.0, 3.0]])
        result = compute_mmd(X, Y)
        assert np.isfinite(result), f"compute_mmd returned {result} for n=2"

    def test_exactly_1_point_returns_nan(self):
        """Confirm the guard: n=1 must return NaN."""
        from peach._core.utils.flow_matching import compute_mmd

        X = np.array([[0.0, 0.0]])
        Y = np.array([[1.0, 1.0], [2.0, 2.0]])
        assert np.isnan(compute_mmd(X, Y))

    def test_zero_points_returns_nan(self):
        from peach._core.utils.flow_matching import compute_mmd

        X = np.zeros((0, 3))
        Y = np.ones((10, 3))
        assert np.isnan(compute_mmd(X, Y))

    def test_both_sides_exactly_2_symmetric(self):
        """MMD(X,Y) == MMD(Y,X) even for n=m=2."""
        from peach._core.utils.flow_matching import compute_mmd

        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        Y = np.array([[5.0, 5.0], [6.0, 6.0]])
        mmd_xy = compute_mmd(X, Y)
        mmd_yx = compute_mmd(Y, X)
        np.testing.assert_almost_equal(mmd_xy, mmd_yx, decimal=10)


# ===========================================================================
# 5. velocity_mode key correctness
# ===========================================================================


class TestVelocityModeKey:
    def test_displacement_when_no_model(self):
        adata, fr = _make_flow_fixture(return_model=False)
        import peach as pc

        result = pc.tl.flow_gene_alignment(adata, fr)
        assert result["velocity_mode"] == "displacement"
        assert result["t"] is None

    def test_displacement_when_t_none_with_model(self):
        adata, fr = _make_flow_fixture(return_model=True)
        import peach as pc

        result = pc.tl.flow_gene_alignment(adata, fr, t=None)
        assert result["velocity_mode"] == "displacement"
        assert result["t"] is None

    def test_instantaneous_with_model_and_t(self):
        adata, fr = _make_flow_fixture(return_model=True)
        import peach as pc

        result = pc.tl.flow_gene_alignment(adata, fr, t=0.3)
        assert result["velocity_mode"] == "instantaneous"
        assert result["t"] == 0.3

    def test_warning_when_t_specified_without_model(self):
        """When t is given but there's no model, a UserWarning must fire
        and the result should fall back to displacement mode."""
        adata, fr = _make_flow_fixture(return_model=False)
        import peach as pc

        with pytest.warns(UserWarning, match="no model"):
            result = pc.tl.flow_gene_alignment(adata, fr, t=0.5)
        assert result["velocity_mode"] == "displacement"


# ===========================================================================
# 6. per_cell alignment with t parameter
# ===========================================================================


class TestPerCellWithT:
    def test_per_cell_instantaneous_shape(self):
        """per_cell=True with t and model should produce per-cell scores
        using instantaneous velocity."""
        adata, fr = _make_flow_fixture()
        import peach as pc

        result = pc.tl.flow_gene_alignment(adata, fr, t=0.5, per_cell=True)
        n_source = fr["source_mask"].sum()
        n_genes = adata.n_vars
        n_top_feat = min(2500, n_genes)
        assert result["per_cell_alignment"].shape == (n_source, n_top_feat)
        assert result["velocity_mode"] == "instantaneous"

    def test_per_cell_displacement_shape(self):
        """per_cell=True with t=None should use displacement."""
        adata, fr = _make_flow_fixture()
        import peach as pc

        result = pc.tl.flow_gene_alignment(adata, fr, t=None, per_cell=True)
        n_source = fr["source_mask"].sum()
        n_genes = adata.n_vars
        n_top_feat = min(2500, n_genes)
        assert result["per_cell_alignment"].shape == (n_source, n_top_feat)
        assert result["velocity_mode"] == "displacement"

    def test_per_cell_values_in_minus1_plus1(self):
        """Per-cell alignment is a cosine similarity and should be in [-1, 1]."""
        adata, fr = _make_flow_fixture()
        import peach as pc

        result = pc.tl.flow_gene_alignment(adata, fr, t=0.5, per_cell=True)
        pca = result["per_cell_alignment"]
        assert np.all(pca >= -1.0 - 1e-6), f"min per_cell_alignment = {pca.min()}"
        assert np.all(pca <= 1.0 + 1e-6), f"max per_cell_alignment = {pca.max()}"

    def test_per_cell_t_differs_from_displacement(self):
        """per_cell with instantaneous velocity should differ from
        per_cell with displacement."""
        adata, fr = _make_flow_fixture()
        import peach as pc

        r_disp = pc.tl.flow_gene_alignment(adata, fr, t=None, per_cell=True)
        r_inst = pc.tl.flow_gene_alignment(adata, fr, t=0.5, per_cell=True)
        assert not np.allclose(
            r_disp["per_cell_alignment"],
            r_inst["per_cell_alignment"],
            atol=1e-6,
        ), "per_cell alignment is the same for displacement and instantaneous — should differ"


# ===========================================================================
# 7. OT-CFM reproducibility across polluted global numpy state
# ===========================================================================


class TestOTCFMReproducibility:
    @pytest.mark.skipif(
        not pytest.importorskip("ot", reason="POT not installed"),
        reason="POT required",
    )
    def test_reproducible_despite_global_numpy_state(self):
        """Training with use_ot=True and the same random_state must
        produce identical losses regardless of global np.random state.

        After the fix, FlowModel.__init__ accepts random_state to seed
        torch before weight initialization, ensuring full reproducibility.
        """
        from peach._core.utils.flow_matching import FlowModel

        rng = np.random.default_rng(42)
        source = rng.standard_normal((60, 5)).astype(np.float32)
        target = (rng.standard_normal((60, 5)) + 2.0).astype(np.float32)

        # Run 1: pollute global state with seed=12345
        np.random.seed(12345)
        torch.manual_seed(99999)
        m1 = FlowModel(5, hidden_dims=(32, 32), lr=1e-3, random_state=42)
        losses1 = m1.train(
            source, target, n_epochs=15, batch_size=32,
            use_ot=True, random_state=42,
        )

        # Run 2: different global state
        np.random.seed(77777)
        torch.manual_seed(11111)
        m2 = FlowModel(5, hidden_dims=(32, 32), lr=1e-3, random_state=42)
        losses2 = m2.train(
            source, target, n_epochs=15, batch_size=32,
            use_ot=True, random_state=42,
        )

        np.testing.assert_allclose(
            losses1, losses2, atol=1e-6,
            err_msg="OT-CFM losses differ despite same random_state — "
                    "global state is leaking into training",
        )

    @pytest.mark.skipif(
        not pytest.importorskip("ot", reason="POT not installed"),
        reason="POT required",
    )
    def test_different_seeds_differ(self):
        """Different random_state values must produce different losses."""
        from peach._core.utils.flow_matching import FlowModel

        rng = np.random.default_rng(42)
        source = rng.standard_normal((60, 5)).astype(np.float32)
        target = (rng.standard_normal((60, 5)) + 2.0).astype(np.float32)

        m1 = FlowModel(5, hidden_dims=(32, 32), lr=1e-3, random_state=42)
        losses1 = m1.train(
            source, target, n_epochs=15, batch_size=32,
            use_ot=True, random_state=42,
        )
        m2 = FlowModel(5, hidden_dims=(32, 32), lr=1e-3, random_state=999)
        losses2 = m2.train(
            source, target, n_epochs=15, batch_size=32,
            use_ot=True, random_state=999,
        )

        assert not np.allclose(losses1, losses2, atol=1e-6), (
            "Different random_state produced identical losses"
        )


# ===========================================================================
# 8. Centrality from sparse vs dense adjacency
# ===========================================================================


class TestCentralitySparseVsDense:
    def test_centrality_from_sparse_not_dense(self):
        """The fix ensures centrality is computed from the SPARSIFIED
        adjacency.  We verify that the centrality values are consistent
        with the sparsified matrix (which has zeros where edges were
        pruned) and NOT with the full dense matrix."""
        import peach as pc

        adata, fr = _make_flow_fixture(return_model=True)
        model = fr["model"]

        result = pc.tl.flow_feature_graph(
            adata, fr, model,
            n_top_genes=20, n_timepoints=5, n_eval_points=30,
            random_state=42,
        )

        adj_sparse = result["adjacency_matrix"]
        abs_sparse = np.abs(adj_sparse)

        # Recompute centrality from the sparse adjacency
        expected_out = abs_sparse.sum(axis=1)
        expected_in = abs_sparse.sum(axis=0)
        expected_flow = expected_out * expected_in

        np.testing.assert_array_almost_equal(result["out_centrality"], expected_out)
        np.testing.assert_array_almost_equal(result["in_centrality"], expected_in)
        np.testing.assert_array_almost_equal(result["flow_centrality"], expected_flow)

    def test_sparse_adjacency_has_zeros(self):
        """The sparsified adjacency should have actual zeros (pruned edges),
        proving it's not just the full dense matrix."""
        import peach as pc

        adata, fr = _make_flow_fixture(return_model=True)
        model = fr["model"]

        result = pc.tl.flow_feature_graph(
            adata, fr, model,
            n_top_genes=20, n_timepoints=5, n_eval_points=30,
            random_state=42,
        )

        adj = result["adjacency_matrix"]
        n_zeros = np.sum(adj == 0.0)
        n_total = adj.size
        # Top 5% means ~95% should be zero (plus diagonal)
        frac_zero = n_zeros / n_total
        assert frac_zero > 0.8, (
            f"Only {frac_zero:.1%} of adjacency is zero — "
            "sparsification may not be working (expected ~95%)"
        )


# ===========================================================================
# 9. flow_significance requires mmd keys
# ===========================================================================


class TestFlowSignificanceValidation:
    def test_missing_mmd_keys_raises(self):
        """flow_significance must raise ValueError when flow_result
        doesn't have mmd_before/mmd_after."""
        import anndata as ad
        from peach.tl.flow import flow_significance

        rng = np.random.default_rng(42)
        n = 50
        adata = ad.AnnData(rng.standard_normal((2 * n, 10)))
        adata.obsm["X_pca"] = rng.standard_normal((2 * n, 5)).astype(np.float32)
        adata.obs["group"] = ["A"] * n + ["B"] * n

        # flow_result without mmd keys
        flow_result = {
            "source_mask": np.array([True] * n + [False] * n),
            "target_mask": np.array([False] * n + [True] * n),
            "transported": rng.standard_normal((n, 5)),
            "pca_key": "X_pca",
        }

        with pytest.raises(ValueError, match="mmd_before"):
            flow_significance(adata, flow_result, n_permutations=3, n_epochs_per_perm=5)


# ===========================================================================
# 10. tools_schema.py consistency for flow_gene_alignment t parameter
# ===========================================================================


class TestToolsSchemaConsistency:
    def test_flow_gene_alignment_t_default_matches_schema(self):
        """The tools_schema t default must match the actual function default (None)."""
        import inspect
        from peach.tl.flow import flow_gene_alignment
        from peach._core.tools_schema import get_tool_schema

        sig = inspect.signature(flow_gene_alignment)
        actual_default = sig.parameters["t"].default
        assert actual_default is None, (
            f"flow_gene_alignment default for t should be None, got {actual_default}"
        )

        schema = get_tool_schema("tl.flow_gene_alignment")
        schema_t = [p for p in schema.parameters if p.name == "t"][0]
        assert schema_t.default is None, (
            f"tools_schema default for t should be None to match function, "
            f"got {schema_t.default}"
        )


# ===========================================================================
# 11. Non-standard OT-CFM — model weight initialization reproducibility
# ===========================================================================


class TestModelWeightInit:
    """After the fix, FlowModel.__init__ accepts random_state and seeds
    torch before weight initialization.  This guarantees reproducibility
    regardless of external torch state.
    """

    def test_random_state_in_init_reproducible(self):
        """FlowModel(random_state=42) should produce identical weights
        regardless of external torch state."""
        from peach._core.utils.flow_matching import FlowModel

        rng = np.random.default_rng(42)
        source = rng.standard_normal((50, 5)).astype(np.float32)
        target = (rng.standard_normal((50, 5)) + 2.0).astype(np.float32)

        torch.manual_seed(999)  # pollute
        m1 = FlowModel(5, hidden_dims=(32, 32), lr=1e-3, random_state=42)
        losses1 = m1.train(source, target, n_epochs=15, batch_size=32, random_state=42)

        torch.manual_seed(777)  # different pollution
        m2 = FlowModel(5, hidden_dims=(32, 32), lr=1e-3, random_state=42)
        losses2 = m2.train(source, target, n_epochs=15, batch_size=32, random_state=42)

        np.testing.assert_allclose(losses1, losses2, atol=1e-6)

    def test_no_random_state_uses_global_torch(self):
        """Without random_state, FlowModel uses whatever global torch
        state exists, so different global states produce different weights."""
        from peach._core.utils.flow_matching import FlowModel

        rng = np.random.default_rng(42)
        source = rng.standard_normal((50, 5)).astype(np.float32)
        target = (rng.standard_normal((50, 5)) + 2.0).astype(np.float32)

        torch.manual_seed(42)
        m1 = FlowModel(5, hidden_dims=(32, 32), lr=1e-3)
        losses1 = m1.train(source, target, n_epochs=15, batch_size=32, random_state=42)

        torch.manual_seed(999)
        m2 = FlowModel(5, hidden_dims=(32, 32), lr=1e-3)
        losses2 = m2.train(source, target, n_epochs=15, batch_size=32, random_state=42)

        # These SHOULD differ because init weights differ
        assert not np.allclose(losses1, losses2, atol=1e-6), (
            "Different global torch seeds at init produced identical losses — "
            "init is somehow ignoring global state?"
        )


# ===========================================================================
# 12. Jacobian at t=0 and t=1 boundaries
# ===========================================================================


class TestJacobianBoundaries:
    """The Jacobian should be well-defined at t=0 and t=1."""

    def test_jacobian_at_t0(self):
        from peach._core.utils.flow_matching import FlowModel

        rng = np.random.default_rng(42)
        source = rng.standard_normal((30, 3)).astype(np.float32)
        target = (source + 2.0).astype(np.float32)
        model = FlowModel(3, hidden_dims=(16, 16))
        model.train(source, target, n_epochs=30, batch_size=16)

        jac = model.jacobian(source[:5], t=0.0)
        assert jac.shape == (5, 3, 3)
        assert np.all(np.isfinite(jac))

    def test_jacobian_at_t1(self):
        from peach._core.utils.flow_matching import FlowModel

        rng = np.random.default_rng(42)
        source = rng.standard_normal((30, 3)).astype(np.float32)
        target = (source + 2.0).astype(np.float32)
        model = FlowModel(3, hidden_dims=(16, 16))
        model.train(source, target, n_epochs=30, batch_size=16)

        jac = model.jacobian(source[:5], t=1.0)
        assert jac.shape == (5, 3, 3)
        assert np.all(np.isfinite(jac))


# ===========================================================================
# 13. compute_mmd unbiased estimator sanity
# ===========================================================================


class TestMMDUnbiased:
    """The unbiased MMD estimator should give a negative value when the
    two sets are drawn from the same distribution (with finite samples,
    the unbiased estimator can be slightly negative, unlike the biased one)."""

    def test_same_distribution_near_zero(self):
        """MMD of two large iid samples should be close to zero."""
        from peach._core.utils.flow_matching import compute_mmd

        rng = np.random.default_rng(42)
        X = rng.standard_normal((500, 5))
        Y = rng.standard_normal((500, 5))
        mmd = compute_mmd(X, Y)
        assert abs(mmd) < 0.05, f"MMD of same-distribution samples = {mmd}, expected ~0"

    def test_well_separated_large(self):
        """Well-separated distributions should have large positive MMD."""
        from peach._core.utils.flow_matching import compute_mmd

        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5))
        Y = rng.standard_normal((200, 5)) + 10.0
        mmd = compute_mmd(X, Y)
        assert mmd > 0.5, f"MMD of shifted distributions = {mmd}, expected >> 0"


# ===========================================================================
# 14. flow_within with return_model=True actually returns a model
# ===========================================================================


class TestReturnModel:
    def test_model_present(self):
        adata, fr = _make_flow_fixture(return_model=True)
        assert "model" in fr, "return_model=True but 'model' not in result"
        from peach._core.utils.flow_matching import FlowModel
        assert isinstance(fr["model"], FlowModel)

    def test_model_absent(self):
        adata, fr = _make_flow_fixture(return_model=False)
        assert "model" not in fr, "return_model=False but 'model' is in result"


# ===========================================================================
# 15. Edge case: n_top > n_genes in flow_gene_alignment
# ===========================================================================


class TestNTopEdge:
    def test_n_top_larger_than_n_genes(self):
        """When n_top > number of genes, should return all genes without error."""
        adata, fr = _make_flow_fixture(n_genes=10)
        import peach as pc

        result = pc.tl.flow_gene_alignment(adata, fr, n_top=100)
        # Should just return all 10 genes
        assert len(result["top_aligned"]) == 10
        assert len(result["top_opposed"]) == 10

    def test_n_top_equals_n_genes(self):
        """n_top == n_genes should work without error."""
        adata, fr = _make_flow_fixture(n_genes=10)
        import peach as pc

        result = pc.tl.flow_gene_alignment(adata, fr, n_top=10)
        assert len(result["top_aligned"]) == 10
        assert len(result["top_opposed"]) == 10

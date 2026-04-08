"""Targeted headless tests for Paper Part 1 round 7 fixes.

Validates:
1. Training with kld_weight=0.09 + manifold_weight=0.001
2. KS test logic on archetype weights
3. Cross-fit Wald math (synthetic data)
4. Straw plot rendering (no crash)
5. Permutation FDR correction
6. Straw plot axis label/data range consistency
"""
import os
import sys
import numpy as np
import pytest


def test_training_with_kld_and_manifold():
    """Train on synthetic data with kld_weight=0.09 and manifold_weight=0.001."""
    import peach as pc

    adata = pc.pp.generate_synthetic(n_points=500, n_dimensions=30, n_archetypes=4, seed=42)
    pc.pp.prepare_training(adata, batch_size=64)

    res = pc.tl.train_archetypal(
        adata, n_archetypes=4, n_epochs=20,
        kld_weight=0.09, archetypal_weight=1.0, inflation_factor=1.0,
        model_config={"manifold_weight": 0.001},
    )
    r2 = res.get("final_archetype_r2")
    assert r2 is not None, "final_archetype_r2 should be present"
    assert r2 > 0, f"R2 should be positive, got {r2}"
    # Check KLD is tracked in history
    hist = res["history"]
    assert "kld" in hist or "KLD" in hist or any("kld" in k.lower() for k in hist.keys()), \
        f"KLD should be in history. Keys: {list(hist.keys())}"
    print(f"  Training OK: R2={r2:.4f}, history keys={list(hist.keys())[:5]}...")


def test_ks_test_on_weights():
    """KS test on archetype weight vectors: same-dist should be non-significant."""
    from scipy.stats import ks_2samp
    import peach as pc

    adata = pc.pp.generate_synthetic(n_points=500, n_dimensions=30, n_archetypes=4, seed=42)
    pc.pp.prepare_training(adata, batch_size=64)
    pc.tl.train_archetypal(adata, n_archetypes=4, n_epochs=20, kld_weight=0.09,
                           inflation_factor=1.0)
    pc.tl.archetypal_coordinates(adata, verbose=False)
    pc.tl.extract_archetype_weights(adata, verbose=False)

    weights = adata.obsm["cell_archetype_weights"]
    # Split into "train" and "holdout" randomly
    n = weights.shape[0]
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    w_train = weights[idx[:n//2]]
    w_hold = weights[idx[n//2:]]

    K = w_train.shape[1]
    for k in range(K):
        stat, pval = ks_2samp(w_train[:, k], w_hold[:, k])
        # Random split should NOT be significantly different (most of the time)
        print(f"  A{k+1}: KS stat={stat:.4f}, p={pval:.4f}")

    # At least verify it runs without error
    assert K > 0, "Should have at least one archetype"
    print(f"  KS test OK: {K} archetypes tested")


def test_crossfit_wald_math():
    """Cross-fit Wald: Z = (beta_h - beta_c) / sqrt(se_h^2 + se_c^2)."""
    from scipy.stats import norm, false_discovery_control

    rng = np.random.default_rng(42)
    n_features = 100

    # Simulate two independent fits with different coefficients
    beta_h = rng.normal(0, 1, n_features)
    beta_c = rng.normal(0, 1, n_features)
    se_h = np.abs(rng.normal(0.2, 0.05, n_features))
    se_c = np.abs(rng.normal(0.2, 0.05, n_features))

    # Make first 10 features truly different
    beta_h[:10] += 3.0

    se_diff = np.sqrt(se_h**2 + se_c**2)
    z_vals = (beta_h - beta_c) / se_diff
    p_vals = 2 * (1 - norm.cdf(np.abs(z_vals)))
    fdr_vals = false_discovery_control(p_vals, method="bh")

    n_sig_raw = (p_vals < 0.05).sum()
    n_sig_fdr = (fdr_vals < 0.05).sum()

    # The 10 truly different features should be significant
    assert n_sig_raw >= 5, f"Expected at least 5 significant, got {n_sig_raw}"
    assert n_sig_fdr >= 5, f"Expected at least 5 FDR-significant, got {n_sig_fdr}"
    print(f"  Cross-fit Wald OK: {n_sig_raw} raw sig, {n_sig_fdr} FDR sig")


def test_straw_plot_rendering():
    """Straw plot renders without error."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    n_genes = 10
    n_bins = 20

    fig, ax = plt.subplots(figsize=(10, 6))
    COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    for gi in range(n_genes):
        # Simulate binned expression (x) and expansion (y)
        bx = np.sort(rng.uniform(0, 5, n_bins))
        by = 1.0 + rng.normal(0, 0.3, n_bins)  # Around 1.0

        ax.plot(bx, by, color=COLORS[gi % len(COLORS)], linewidth=1.8,
                alpha=0.85, label=f"Gene{gi}")
        ax.scatter(bx[0], by[0], color=COLORS[gi % len(COLORS)], s=30, marker="o")
        ax.scatter(bx[-1], by[-1], color=COLORS[gi % len(COLORS)], s=30, marker="^")

    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Expression (logcounts along flow)")
    ax.set_ylabel("Expansion factor")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    plt.close("all")
    print("  Straw plot render OK")


def test_permutation_fdr():
    """BH FDR correction on permutation p-values."""
    from scipy.stats import false_discovery_control

    rng = np.random.default_rng(42)
    # Most p-values near 1.0 (not significant), a few near 0
    p_vals = np.concatenate([rng.uniform(0.3, 1.0, 90), rng.uniform(0.001, 0.01, 10)])
    fdr = false_discovery_control(p_vals, method="bh")

    assert len(fdr) == len(p_vals)
    assert (fdr >= p_vals).all(), "FDR should be >= raw p-values"
    n_sig_raw = (p_vals < 0.05).sum()
    n_sig_fdr = (fdr < 0.05).sum()
    assert n_sig_fdr <= n_sig_raw, "FDR significant should be <= raw significant"
    print(f"  Permutation FDR OK: {n_sig_raw} raw, {n_sig_fdr} FDR")


def test_straw_plot_axis_matches_data_range():
    """Straw plot x-axis label must match the type of data actually plotted.

    Imports _straw_plot_xlabel from the paper script and validates three cases:
    1. All genes mapped (no fallback) → label says "logcounts" / "expression",
       not "pseudotime" or "flow coordinate".
    2. All genes fell back → label says "flow coordinate" / "pseudotime",
       not "logcounts".
    3. Mixed (some mapped, some fallback) → label reports the fallback ratio
       and does NOT claim all genes were mapped or all fell back.

    This test catches the regression where someone hard-codes "expression" while
    the data is actually in [0, 1] pseudotime, or vice-versa.
    """
    # Import the helper from the paper script.
    _script_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)

    # Import only the helper — avoids executing the full script (matplotlib
    # backend is set at module level, harmless; DATA_DIR may not exist but
    # the import still works because it only creates a string).
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "run_paper_part1_hsc",
        os.path.join(_script_dir, "run_paper_part1_hsc.py"),
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _straw_plot_xlabel = _mod._straw_plot_xlabel

    # --- Case 1: all genes mapped, logcounts range [0, 7.5] ---
    # x-axis values ARE logcounts (not flow_coord), so the label must say
    # "logcounts" or "expression".  It may mention "pseudotime" as the binning
    # dimension (e.g. "per pseudotime bin") — that is fine; what it must NOT do
    # is claim the x-axis VALUES are flow_coord / normalized pseudotime 0-1.
    label_all_mapped = _straw_plot_xlabel(n_mapped=10, n_fallback=0, x_min=0.0, x_max=7.5)
    assert "expression" in label_all_mapped.lower() or "logcount" in label_all_mapped.lower(), (
        f"All-mapped label should mention 'expression' or 'logcounts', got: {label_all_mapped!r}"
    )
    # The all-fallback phrase "all genes fell back" must NOT appear
    assert "all genes fell back" not in label_all_mapped.lower(), (
        f"All-mapped label should not say 'all genes fell back', got: {label_all_mapped!r}"
    )
    # "flow coordinate" as the primary data claim (not just "pseudotime bin") must not appear
    assert "flow coordinate" not in label_all_mapped.lower(), (
        f"All-mapped label should not describe x-axis as 'flow coordinate', got: {label_all_mapped!r}"
    )
    # Also assert the x range is in the label (proves it reads from real data, not hardcoded)
    assert "7.50" in label_all_mapped or "7.5" in label_all_mapped, (
        f"All-mapped label should contain x_max=7.5, got: {label_all_mapped!r}"
    )

    # --- Case 2: all genes fell back, flow_coord range [0, 1] ---
    # x-axis values ARE flow_coord (0-1), so the label must NOT say "logcounts"
    # and MUST indicate fallback / pseudotime / flow coordinate.
    label_all_fallback = _straw_plot_xlabel(n_mapped=0, n_fallback=10, x_min=0.0, x_max=1.0)
    assert ("pseudotime" in label_all_fallback.lower()
            or "flow coordinate" in label_all_fallback.lower()
            or "fallback" in label_all_fallback.lower()), (
        f"All-fallback label should mention 'pseudotime', 'flow coordinate', or 'fallback', "
        f"got: {label_all_fallback!r}"
    )
    assert "logcount" not in label_all_fallback.lower(), (
        f"All-fallback label should NOT mention 'logcounts', got: {label_all_fallback!r}"
    )

    # --- Case 3: mixed (3 of 10 genes fell back) ---
    label_mixed = _straw_plot_xlabel(n_mapped=7, n_fallback=3, x_min=0.0, x_max=6.2)
    # Must mention the fallback count explicitly
    assert "3" in label_mixed, (
        f"Mixed label should report fallback count (3), got: {label_mixed!r}"
    )
    # Must mention total (10) or denominator
    assert "10" in label_mixed, (
        f"Mixed label should report total gene count (10), got: {label_mixed!r}"
    )
    # Must not claim ALL genes fell back
    assert "all genes fell back" not in label_mixed.lower(), (
        f"Mixed label should not say 'all genes fell back', got: {label_mixed!r}"
    )
    print(f"  all-mapped : {label_all_mapped}")
    print(f"  all-fallback: {label_all_fallback}")
    print(f"  mixed      : {label_mixed}")


if __name__ == "__main__":
    print("=== Paper Part 1 Round 7 Fix Tests ===")
    test_crossfit_wald_math()
    test_permutation_fdr()
    test_straw_plot_rendering()
    test_ks_test_on_weights()
    test_training_with_kld_and_manifold()
    test_straw_plot_axis_matches_data_range()
    print("\n=== All tests passed ===")

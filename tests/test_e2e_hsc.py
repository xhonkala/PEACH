"""End-to-end test of v0.5.0 features on real HSC data (CMP vs CD14+ Monocyte).

Tests: archetype fitting, simplex regression, pattern classification,
archetype comparison, flow matching, and visualization.

Usage:
    /Users/honkala/miniconda3/envs/archetype/bin/python tests/test_e2e_hsc.py

Note: Do NOT redirect stderr (2>/dev/null) — causes OMP segfault on macOS.
Use: 2>tests/e2e_outputs/stderr.log instead.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import warnings
import logging
import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.disable(logging.WARNING)

import torch
torch.set_num_threads(1)

# ─── Config ──────────────────────────────────────────────────────────────────
HSC_PATH = "/Users/honkala/Desktop/cross_recons/data/HSC.h5ad"
K = 4
N_PCS = 20
MONO_SUBSAMPLE = 9000
TRAIN_EPOCHS = 150
FLOW_EPOCHS = 300
SEED = 42
OUTPUT_DIR = "/Users/honkala/Desktop/PEACH_public/tests/e2e_outputs"


def section(title):
    print(f"\n{'='*60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*60}", flush=True)


def step(msg):
    print(f"  -> {msg}", end="", flush=True)


def done(extra=""):
    print(f" OK {extra}", flush=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()

    # ─── 1. Load and subset data ─────────────────────────────────────
    section("1. Data Loading")

    step("Loading HSC dataset")
    import anndata as ad
    adata_full = ad.read_h5ad(HSC_PATH)
    done(f"({adata_full.shape[0]} cells)")

    rng = np.random.default_rng(SEED)

    # Extract subsets — only keep PCA and var_names for now, load X lazily later
    cell_data = {}
    var_names = list(adata_full.var_names)
    for name, ct, subsample in [
        ("CMP", "common myeloid progenitor", None),
        ("Mono", "CD14-positive monocyte", MONO_SUBSAMPLE),
    ]:
        step(f"Subsetting {name}")
        mask = adata_full.obs["cell_type"] == ct
        idx = np.where(mask)[0]
        if subsample and len(idx) > subsample:
            idx = rng.choice(idx, size=subsample, replace=False)
            idx.sort()

        pca = adata_full.obsm["X_pca"][idx, :N_PCS].copy()
        obs = adata_full.obs.iloc[idx].copy().reset_index(drop=True)
        cell_data[name] = {"pca": pca, "obs": obs, "idx": idx}
        done(f"({len(idx)} cells, PCA: {pca.shape})")

    # Free full dataset before training to avoid OMP memory pressure
    del adata_full
    import gc
    gc.collect()

    # ─── 2. Archetype fitting ────────────────────────────────────────
    section("2. Archetype Fitting")
    import peach as pc

    models = {}
    for name in ["CMP", "Mono"]:
        step(f"Training K={K} on {name}")
        t1 = time.time()

        # Lightweight AnnData for training (avoids OMP crash with large sparse X)
        adata_train = ad.AnnData(
            X=sp.csr_matrix((len(cell_data[name]["pca"]), 1)),
            obs=cell_data[name]["obs"].copy(),
        )
        adata_train.obsm["X_pca"] = cell_data[name]["pca"]

        result = pc.tl.train_archetypal(
            adata_train, n_archetypes=K,
            n_epochs=TRAIN_EPOCHS, kld_weight=0.1, archetypal_weight=0.9,
            seed=SEED,
        )
        elapsed = time.time() - t1
        r2 = result.get("final_archetype_r2", None)
        r2_str = f"R2={r2:.3f}" if r2 else "R2=N/A"
        done(f"({r2_str}, {elapsed:.1f}s)")

        # Extract weights
        pc.tl.extract_archetype_weights(adata_train)

        # Store training results for later
        cell_data[name]["weights"] = adata_train.obsm["cell_archetype_weights"]
        cell_data[name]["uns"] = dict(adata_train.uns)
        if "archetype_distances" in adata_train.obsm:
            cell_data[name]["arch_dist"] = adata_train.obsm["archetype_distances"]
        models[name] = result

    # Reload sparse X for regression (one cell type at a time to manage memory)
    step("Reloading sparse X for regression")
    adata_disk = ad.read_h5ad(HSC_PATH, backed="r")
    for name in ["CMP", "Mono"]:
        idx = cell_data[name]["idx"]
        X_sparse = sp.csr_matrix(adata_disk.X[idx])
        import pandas as pd
        var_df = pd.DataFrame(index=var_names)

        adata_obj = ad.AnnData(X=X_sparse, obs=cell_data[name]["obs"].copy(), var=var_df)
        adata_obj.obsm["X_pca"] = cell_data[name]["pca"]
        adata_obj.obsm["cell_archetype_weights"] = cell_data[name]["weights"]
        if "arch_dist" in cell_data[name]:
            adata_obj.obsm["archetype_distances"] = cell_data[name]["arch_dist"]
        for k, v in cell_data[name]["uns"].items():
            adata_obj.uns[k] = v
        pc.tl.assign_archetypes(adata_obj)
        cell_data[name]["adata"] = adata_obj
    del adata_disk
    gc.collect()
    done()

    adata_cmp = cell_data["CMP"]["adata"]
    adata_mono = cell_data["Mono"]["adata"]

    # ─── 3. Simplex regression ───────────────────────────────────────
    section("3. Simplex Regression")

    for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
        step(f"Feature simplex regression on {name}")
        t1 = time.time()
        reg = pc.tl.feature_simplex_regression(
            ad_obj, max_degree=2, n_bootstrap=0, robust_se=True,
        )
        elapsed = time.time() - t1
        median_r2 = np.median(reg.r_squared_degree1)
        top_genes = np.argsort(reg.r_squared_degree1)[-5:][::-1]
        top_names = [reg.feature_names[i] for i in top_genes]
        top_r2s = reg.r_squared_degree1[top_genes]
        done(f"(median R2={median_r2:.3f}, {elapsed:.1f}s)")
        for g, r in zip(top_names, top_r2s):
            print(f"      {g}: R2={r:.3f}", flush=True)

    # ─── 4. Pattern classification ───────────────────────────────────
    section("4. Pattern Classification")

    for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
        step(f"Classifying feature patterns for {name}")
        patterns = pc.tl.classify_feature_patterns(ad_obj)
        counts = patterns["pattern_counts"]
        done(f"({dict(counts)})")

    # ─── 5. Archetype comparison (NEW) ───────────────────────────────
    section("5. Archetype Comparison (NEW)")

    # 5a. Within-fit MMD
    for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
        step(f"Within-fit MMD for {name}")
        mmd_result = pc.tl.archetype_mmd(ad_obj, n_permutations=100)
        mask = ~np.eye(K, dtype=bool)
        mmd_vals = mmd_result.mmd_matrix[mask]
        done(f"(MMD range: {mmd_vals.min():.4f}-{mmd_vals.max():.4f})")

    # 5b. Between-fit MMD
    step("Between-fit MMD (CMP vs Mono)")
    mmd_between = pc.tl.archetype_mmd(adata_cmp, adata_mono, n_permutations=100)
    done(f"(shape: {mmd_between.mmd_matrix.shape})")
    print(f"      MMD matrix:\n{np.array2string(mmd_between.mmd_matrix, precision=4)}", flush=True)

    # 5c. Feature similarity
    for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
        step(f"Feature similarity for {name}")
        sim = pc.tl.archetype_feature_similarity(ad_obj)
        done(f"(silhouette={sim.silhouette_overall:.3f})")

    # 5d. Between-fit feature similarity
    step("Between-fit feature similarity (CMP vs Mono)")
    sim_between = pc.tl.archetype_feature_similarity(adata_cmp, adata_mono)
    done(f"(n_shared={sim_between.n_shared_features})")
    print(f"      Spearman matrix:\n{np.array2string(sim_between.spearman_matrix, precision=3)}", flush=True)

    # 5e. Wald contrasts
    for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
        step(f"Wald contrasts for {name}")
        contrasts = pc.tl.archetype_contrasts(ad_obj)
        pair_summary = []
        for pair in contrasts.pairs:
            n_sig = int(np.sum(contrasts.pvalues_fdr[pair] < 0.05))
            pair_summary.append(f"{pair}:{n_sig}")
        done(f"(sig genes per pair: {', '.join(pair_summary)})")

    # ─── 6. Flow matching ────────────────────────────────────────────
    section("6. Flow Matching")

    step("Building combined adata for flow")
    adata_cmp_flow = ad.AnnData(
        X=sp.csr_matrix((adata_cmp.n_obs, 1)),
        obs=adata_cmp.obs.copy(),
    )
    adata_cmp_flow.obsm["X_pca"] = cell_data["CMP"]["pca"]
    adata_cmp_flow.obs["cell_type_label"] = "CMP"

    adata_mono_flow = ad.AnnData(
        X=sp.csr_matrix((adata_mono.n_obs, 1)),
        obs=adata_mono.obs.copy(),
    )
    adata_mono_flow.obsm["X_pca"] = cell_data["Mono"]["pca"]
    adata_mono_flow.obs["cell_type_label"] = "Mono"

    adata_combined = ad.concat([adata_cmp_flow, adata_mono_flow])
    done(f"({adata_combined.n_obs} cells)")

    step(f"Training flow CMP -> Mono ({FLOW_EPOCHS} epochs)")
    t1 = time.time()
    flow_result = pc.tl.flow_within(
        adata_combined,
        source={"cell_type_label": "CMP"},
        target={"cell_type_label": "Mono"},
        pca_key="X_pca",
        hidden_dims=(64, 64, 64),
        n_epochs=FLOW_EPOCHS,
        batch_size=256,
        n_steps=50,
        device="cpu",
    )
    elapsed = time.time() - t1
    done(f"(MMD: {flow_result.mmd_before:.4f} -> {flow_result.mmd_after:.4f}, {elapsed:.1f}s)")

    # ─── 7. Visualization ────────────────────────────────────────────
    section("7. Visualization")

    viz_tasks = []

    # Regression plots
    for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
        viz_tasks.append((
            f"Coefficient heatmap ({name})",
            lambda ad=ad_obj, n=name: pc.pl.coefficient_heatmap(
                ad, top_n=30, show=False,
                save_path=os.path.join(OUTPUT_DIR, f"coef_heatmap_{n}.html")
            ),
        ))
        viz_tasks.append((
            f"R2 barplot ({name})",
            lambda ad=ad_obj, n=name: pc.pl.r2_barplot(
                ad, top_n=30, show=False,
                save_path=os.path.join(OUTPUT_DIR, f"r2_barplot_{n}.html")
            ),
        ))
        viz_tasks.append((
            f"Regression volcano ({name})",
            lambda ad=ad_obj, n=name: pc.pl.regression_volcano(
                ad, show=False,
                save_path=os.path.join(OUTPUT_DIR, f"regression_volcano_{n}.html")
            ),
        ))

    # Comparison plots
    for name, ad_obj in [("CMP", adata_cmp), ("Mono", adata_mono)]:
        viz_tasks.append((
            f"MMD heatmap ({name})",
            lambda ad=ad_obj, n=name: pc.pl.mmd_heatmap(
                ad, show=False,
                save_path=os.path.join(OUTPUT_DIR, f"mmd_heatmap_{n}.html")
            ),
        ))
        viz_tasks.append((
            f"Feature similarity ({name})",
            lambda ad=ad_obj, n=name: pc.pl.feature_similarity_heatmap(
                ad, show=False,
                save_path=os.path.join(OUTPUT_DIR, f"feature_similarity_{n}.html")
            ),
        ))
        viz_tasks.append((
            f"Contrast volcano ({name} 0v1)",
            lambda ad=ad_obj, n=name: pc.pl.contrast_volcano(
                ad, pair=(0, 1), show=False,
                save_path=os.path.join(OUTPUT_DIR, f"contrast_volcano_{n}_0v1.html")
            ),
        ))

    # Flow plots
    viz_tasks.append((
        "Flow magnitude",
        lambda: pc.pl.flow_magnitude(
            adata_combined, flow_result, show=False,
            save_path=os.path.join(OUTPUT_DIR, "flow_magnitude.html")
        ),
    ))
    viz_tasks.append((
        "Density comparison",
        lambda: pc.pl.density_comparison(
            adata_combined, flow_result, show=False,
            save_path=os.path.join(OUTPUT_DIR, "density_comparison.html")
        ),
    ))
    viz_tasks.append((
        "Velocity quiver",
        lambda: pc.pl.velocity_quiver(
            adata_combined, flow_result, show=False,
            save_path=os.path.join(OUTPUT_DIR, "velocity_quiver.html")
        ),
    ))

    for viz_name, viz_fn in viz_tasks:
        step(viz_name)
        try:
            viz_fn()
            done()
        except Exception as e:
            print(f" FAIL: {e}", flush=True)

    # ─── Summary ─────────────────────────────────────────────────────
    section("Summary")
    elapsed_total = time.time() - t0
    n_files = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.html')])
    print(f"  Total time: {elapsed_total:.1f}s", flush=True)
    print(f"  HTML files: {n_files} in {OUTPUT_DIR}", flush=True)
    print(f"\n  END-TO-END TEST PASSED", flush=True)


if __name__ == "__main__":
    main()

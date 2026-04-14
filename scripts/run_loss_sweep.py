#!/usr/bin/env python
"""Loss term sweep on HSC: kld_weight × sparsity_weight with manifold_weight=0.001.

Holds manifold_weight=0.001 (the sweet spot from the prior sweep) and
sweeps kld_weight and sparsity_weight to find the best combination for
on-hull archetypes + high R².

Grid:
  kld_weight:      [0.01, 0.05, 0.09, 0.15, 0.25]
  sparsity_weight: [0.0, 0.001, 0.01, 0.05, 0.1]
  manifold_weight: 0.001 (fixed)
  archetypal_weight: 0.9 (fixed)

= 25 configs × ~2 min each ≈ 50 min on 2000-cell subsample.
"""
import matplotlib
matplotlib.use("Agg")

import base64
import io
import itertools
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "diagnostic")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HSC_TRAIN = os.path.join(PROJECT_DIR, "data", "paper_part1", "adata_hsc_train.h5ad")

# Sweep grid
KLD_WEIGHTS = [0.01, 0.05, 0.09, 0.15, 0.25]
SPARSITY_WEIGHTS = [0.0, 0.001, 0.01, 0.05, 0.1]
MANIFOLD_WEIGHT = 0.001  # fixed from prior sweep

# Fixed params (pDC-validated)
K = 7
HIDDEN_DIMS = [64, 128]
ARCHETYPAL_WEIGHT = 0.9
INFLATION = 1.0
N_EPOCHS = 150
SUBSAMPLE = 2000
SEED = 42


def fig_to_b64(fig, dpi=130):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;">'


def main():
    import anndata as ad
    import peach as pc
    import torch

    print(f"Loading {HSC_TRAIN}...")
    adata_full = ad.read_h5ad(HSC_TRAIN)

    rng = np.random.default_rng(SEED)
    idx = rng.choice(adata_full.shape[0], size=min(SUBSAMPLE, adata_full.shape[0]), replace=False)
    adata = adata_full[idx].copy()
    print(f"  Subsample: {adata.shape}")

    X = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
    data_centroid = X.mean(axis=0)
    data_dists = np.linalg.norm(X - data_centroid, axis=1)
    data_max_dist = data_dists.max()
    print(f"  Data cloud: max_dist={data_max_dist:.1f}")

    configs = list(itertools.product(KLD_WEIGHTS, SPARSITY_WEIGHTS))
    print(f"  Grid: {len(configs)} configs (kld × sparsity)")

    results = []

    for i, (kld_w, sp_w) in enumerate(configs):
        label = f"kld={kld_w}, sp={sp_w}"
        print(f"\n[{i+1}/{len(configs)}] {label}")

        adata_run = adata.copy()
        pc.pp.prepare_training(adata_run, batch_size=64)

        t0 = time.time()
        try:
            res = pc.tl.train_archetypal(
                adata_run,
                n_archetypes=K,
                n_epochs=N_EPOCHS,
                hidden_dims=HIDDEN_DIMS,
                kld_weight=kld_w,
                archetypal_weight=ARCHETYPAL_WEIGHT,
                inflation_factor=INFLATION,
                pcha_init=False,
                model_config={
                    "manifold_weight": MANIFOLD_WEIGHT,
                    "sparsity_weight": sp_w,
                },
                seed=SEED,
            )
            elapsed = time.time() - t0

            r2 = res.get("final_archetype_r2", float("nan"))
            model = res["model"]
            model.eval()
            with torch.no_grad():
                out = model(torch.FloatTensor(X[:64]))
                Y = out["Y"].cpu().numpy()

            arch_dists = np.linalg.norm(Y - data_centroid, axis=1)
            ratio_max = arch_dists.max() / data_max_dist
            n_outside = int((arch_dists > data_max_dist).sum())

            history = res.get("history", {})
            kld_hist = history.get("kld_loss", history.get("KLD", []))
            final_kld = kld_hist[-1] if kld_hist else float("nan")

        except Exception as e:
            print(f"  FAILED: {e}")
            elapsed = time.time() - t0
            r2, ratio_max, n_outside, final_kld = float("nan"), float("nan"), -1, float("nan")
            Y = None

        row = {
            "kld_weight": kld_w,
            "sparsity_weight": sp_w,
            "R2": r2,
            "ratio_max": ratio_max,
            "n_outside": n_outside,
            "final_kld": final_kld,
            "elapsed": elapsed,
            "_Y": Y,
        }
        results.append(row)
        print(f"  R2={r2:.4f}, ratio={ratio_max:.2f}, outside={n_outside}/{K}, "
              f"kld_final={final_kld:.4f}, time={elapsed:.0f}s")

    # ---- Build HTML report ----
    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
    <title>Loss Term Sweep</title>
    <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px 40px;
           background: #fafafa; color: #222; max-width: 1400px; }}
    table {{ border-collapse: collapse; margin: 15px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: right; font-size: 0.9em; }}
    th {{ background: #4a90d9; color: white; }}
    .good {{ background: #d4edda !important; }}
    .ok {{ background: #fff3cd !important; }}
    .bad {{ background: #f8d7da !important; }}
    img {{ max-width: 100%; margin: 10px 0; }}
    </style></head><body>
    <h1>Loss Term Sweep: HSC K={K}, manifold_weight={MANIFOLD_WEIGHT}</h1>
    <p>kld_weight x sparsity_weight grid, {SUBSAMPLE}-cell subsample, {N_EPOCHS} epochs.
    Fixed: archetypal_weight={ARCHETYPAL_WEIGHT}, hidden_dims={HIDDEN_DIMS},
    inflation={INFLATION}, pcha_init=False.</p>
    """

    # Summary table
    html += "<h2>Summary (sorted by R2 descending, highlighting on-hull configs)</h2><table>"
    html += ("<tr><th>kld</th><th>sparsity</th><th>R2</th>"
             "<th>hull ratio</th><th>outside</th><th>final KLD</th><th>time</th></tr>")

    sorted_results = sorted(results, key=lambda r: -r["R2"])
    for r in sorted_results:
        on_hull = r["ratio_max"] <= 1.05
        good_r2 = r["R2"] > 0.8
        if on_hull and good_r2:
            cls = "good"
        elif on_hull or good_r2:
            cls = "ok"
        else:
            cls = "bad"
        html += (f"<tr class='{cls}'>"
                 f"<td>{r['kld_weight']}</td><td>{r['sparsity_weight']}</td>"
                 f"<td>{r['R2']:.4f}</td><td>{r['ratio_max']:.2f}</td>"
                 f"<td>{r['n_outside']}/{K}</td>"
                 f"<td>{r['final_kld']:.4f}</td><td>{r['elapsed']:.0f}s</td></tr>")
    html += "</table>"

    # Heatmap: R2
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    r2_grid = np.full((len(SPARSITY_WEIGHTS), len(KLD_WEIGHTS)), np.nan)
    ratio_grid = np.full_like(r2_grid, np.nan)
    kld_final_grid = np.full_like(r2_grid, np.nan)

    for r in results:
        ki = KLD_WEIGHTS.index(r["kld_weight"])
        si = SPARSITY_WEIGHTS.index(r["sparsity_weight"])
        r2_grid[si, ki] = r["R2"]
        ratio_grid[si, ki] = r["ratio_max"]
        kld_final_grid[si, ki] = r["final_kld"]

    for ax, data, title, cmap in [
        (axes[0], r2_grid, "Archetype R2", "YlGn"),
        (axes[1], ratio_grid, "Hull ratio (arch max / data max)", "RdYlGn_r"),
        (axes[2], kld_final_grid, "Final KLD loss", "YlOrRd"),
    ]:
        im = ax.imshow(data, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(KLD_WEIGHTS)))
        ax.set_xticklabels([str(k) for k in KLD_WEIGHTS])
        ax.set_yticks(range(len(SPARSITY_WEIGHTS)))
        ax.set_yticklabels([str(s) for s in SPARSITY_WEIGHTS])
        ax.set_xlabel("kld_weight")
        ax.set_ylabel("sparsity_weight")
        ax.set_title(title)
        # Annotate cells
        for si in range(len(SPARSITY_WEIGHTS)):
            for ki in range(len(KLD_WEIGHTS)):
                val = data[si, ki]
                if not np.isnan(val):
                    ax.text(ki, si, f"{val:.3f}", ha="center", va="center",
                            fontsize=7, color="black" if val < np.nanmedian(data) * 1.5 else "white")
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Loss Sweep Heatmaps (manifold_weight={MANIFOLD_WEIGHT} fixed)", fontsize=13)
    fig.tight_layout()
    html += f"<h2>Heatmaps</h2>{fig_to_b64(fig)}"

    # Top-5 3D scatters
    html += "<h2>Top 5 Configs (by R2, on-hull first)</h2>"
    on_hull_results = [r for r in sorted_results if r["ratio_max"] <= 1.1]
    show_results = (on_hull_results or sorted_results)[:5]

    for r in show_results:
        Y = r["_Y"]
        if Y is None:
            continue
        fig = plt.figure(figsize=(7, 5.5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=1, alpha=0.3, c="gray")
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], s=100, c="red", marker="D",
                   edgecolors="black", linewidths=1, zorder=10)
        for k_i in range(K):
            ax.text(Y[k_i, 0], Y[k_i, 1], Y[k_i, 2], f"A{k_i+1}", fontsize=8)
        ax.set_title(f"kld={r['kld_weight']}, sparsity={r['sparsity_weight']}, "
                     f"R2={r['R2']:.4f}, ratio={r['ratio_max']:.2f}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        html += fig_to_b64(fig)

    html += "</body></html>"

    report_path = os.path.join(OUTPUT_DIR, "loss_sweep.html")
    with open(report_path, "w") as f:
        f.write(html)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()

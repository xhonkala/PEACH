#!/usr/bin/env python
"""Quick manifold_weight sweep on HSC data.

Tests whether the fixed manifold_regularization_loss (now on effective
archetype positions instead of raw) can keep archetypes on the data hull
while maintaining good R².

Sweeps manifold_weight in {0, 0.001, 0.01, 0.05, 0.1, 0.5} with the
user's pDC-validated params: hidden_dims=[64,128], kld=0.09,
archetypal=0.9, sparsity=0.01, inflation=1.0, pcha_init=False.

Tests K=7 (user's predicted HSC optimum) on a 2000-cell subsample for
speed (~2 min per config × 6 configs = ~12 min total).
"""
import matplotlib
matplotlib.use("Agg")

import base64
import io
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

MANIFOLD_WEIGHTS = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5]
K = 7
HIDDEN_DIMS = [64, 128]
KLD_WEIGHT = 0.09
ARCHETYPAL_WEIGHT = 0.9
SPARSITY_WEIGHT = 0.01
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
    print(f"  Full: {adata_full.shape}")

    # Subsample for speed
    rng = np.random.default_rng(SEED)
    idx = rng.choice(adata_full.shape[0], size=min(SUBSAMPLE, adata_full.shape[0]), replace=False)
    adata = adata_full[idx].copy()
    print(f"  Subsample: {adata.shape}")

    X = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
    data_centroid = X.mean(axis=0)
    data_dists = np.linalg.norm(X - data_centroid, axis=1)
    data_max_dist = data_dists.max()
    data_p99_dist = np.percentile(data_dists, 99)
    print(f"  Data cloud: max_dist={data_max_dist:.1f}, p99={data_p99_dist:.1f}")

    results = []

    for mw in MANIFOLD_WEIGHTS:
        print(f"\n{'='*60}")
        print(f"manifold_weight = {mw}")
        print(f"{'='*60}")

        adata_run = adata.copy()
        pc.pp.prepare_training(adata_run, batch_size=64)

        t0 = time.time()
        res = pc.tl.train_archetypal(
            adata_run,
            n_archetypes=K,
            n_epochs=N_EPOCHS,
            hidden_dims=HIDDEN_DIMS,
            kld_weight=KLD_WEIGHT,
            archetypal_weight=ARCHETYPAL_WEIGHT,
            inflation_factor=INFLATION,
            pcha_init=False,
            model_config={
                "manifold_weight": mw,
                "sparsity_weight": SPARSITY_WEIGHT,
            },
            seed=SEED,
        )
        elapsed = time.time() - t0

        r2 = res.get("final_archetype_r2", float("nan"))

        # Get effective archetype positions
        model = res["model"]
        model.eval()
        with torch.no_grad():
            sample = torch.FloatTensor(X[:64])
            out = model(sample)
            Y = out["Y"].cpu().numpy()

        arch_dists = np.linalg.norm(Y - data_centroid, axis=1)
        ratio_max = arch_dists.max() / data_max_dist
        ratio_mean = arch_dists.mean() / data_dists.mean()

        # How many archetypes are outside the data hull?
        n_outside = (arch_dists > data_max_dist).sum()

        # Manifold loss from history
        history = res.get("history", {})
        manifold_hist = history.get("manifold_loss", [])
        final_manifold = manifold_hist[-1] if manifold_hist else float("nan")

        row = {
            "manifold_weight": mw,
            "R2": r2,
            "arch_max_dist": arch_dists.max(),
            "arch_mean_dist": arch_dists.mean(),
            "ratio_max": ratio_max,
            "ratio_mean": ratio_mean,
            "n_outside_hull": n_outside,
            "final_manifold_loss": final_manifold,
            "elapsed_s": elapsed,
        }
        results.append(row)

        print(f"  R2={r2:.4f}, arch_max/data_max={ratio_max:.2f}, "
              f"n_outside={n_outside}/{K}, time={elapsed:.0f}s")

        # Store model for plotting
        row["_Y"] = Y
        row["_history"] = history

    # Build HTML report
    html = """<!DOCTYPE html><html><head><meta charset="UTF-8">
    <title>Manifold Weight Sweep</title>
    <style>
    body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px 40px;
           background: #fafafa; color: #222; max-width: 1200px; }
    table { border-collapse: collapse; margin: 15px 0; }
    th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: right; }
    th { background: #4a90d9; color: white; }
    tr:nth-child(even) { background: #f2f2f2; }
    .good { background: #d4edda !important; }
    .bad { background: #f8d7da !important; }
    img { max-width: 100%; margin: 10px 0; }
    </style></head><body>
    <h1>Manifold Weight Sweep: HSC K=7</h1>
    <p>Fixed manifold_regularization_loss to use <b>effective</b> (post-transform)
    archetype positions. Params: hidden_dims=[64,128], kld=0.09, arch=0.9,
    sparsity=0.01, inflation=1.0, pcha_init=False, n_epochs=150,
    subsample=2000 cells.</p>
    """

    # Summary table
    html += "<h2>Summary</h2><table>"
    html += ("<tr><th>manifold_weight</th><th>R2</th><th>arch max/data max</th>"
             "<th>n outside hull</th><th>final manifold loss</th><th>time (s)</th></tr>")
    for r in results:
        on_hull = r["ratio_max"] <= 1.05
        cls = "good" if on_hull and r["R2"] > 0.85 else ("bad" if r["ratio_max"] > 1.3 else "")
        html += (f"<tr class='{cls}'><td>{r['manifold_weight']}</td>"
                 f"<td>{r['R2']:.4f}</td><td>{r['ratio_max']:.2f}</td>"
                 f"<td>{r['n_outside_hull']}/{K}</td>"
                 f"<td>{r['final_manifold_loss']:.4f}</td>"
                 f"<td>{r['elapsed_s']:.0f}</td></tr>")
    html += "</table>"

    # R2 vs ratio plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    mws = [r["manifold_weight"] for r in results]
    r2s = [r["R2"] for r in results]
    ratios = [r["ratio_max"] for r in results]

    ax1.plot(mws, r2s, "o-", color="#0072B2", linewidth=2, markersize=8)
    ax1.set_xlabel("manifold_weight")
    ax1.set_ylabel("Archetype R2")
    ax1.set_title("R2 vs manifold_weight")
    ax1.axhline(0.9, color="green", linestyle="--", alpha=0.5, label="R2=0.9")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(mws, ratios, "o-", color="#D55E00", linewidth=2, markersize=8)
    ax2.set_xlabel("manifold_weight")
    ax2.set_ylabel("max archetype dist / max data dist")
    ax2.set_title("Hull ratio vs manifold_weight")
    ax2.axhline(1.0, color="green", linestyle="--", alpha=0.5, label="on hull")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    html += f"<h2>R2 and Hull Ratio vs manifold_weight</h2>{fig_to_b64(fig)}"

    # Per-config 3D scatter (first 3 PCs)
    html += "<h2>Archetype Positions (first 3 PCs)</h2>"
    for r in results:
        Y = r["_Y"]
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=1, alpha=0.3, c="gray")
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], s=100, c="red", marker="D",
                   edgecolors="black", linewidths=1, zorder=10)
        for k_i in range(K):
            ax.text(Y[k_i, 0], Y[k_i, 1], Y[k_i, 2], f"A{k_i+1}", fontsize=8)
        ax.set_title(f"manifold_weight={r['manifold_weight']}, R2={r['R2']:.4f}, "
                     f"ratio={r['ratio_max']:.2f}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        html += f"{fig_to_b64(fig)}"

    html += "</body></html>"

    report_path = os.path.join(OUTPUT_DIR, "manifold_sweep.html")
    with open(report_path, "w") as f:
        f.write(html)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()

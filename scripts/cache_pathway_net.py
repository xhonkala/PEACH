"""One-shot script to pre-download and cache the c5_bp pathway network to parquet.

Run once with network available:
    conda run -n archetype python scripts/cache_pathway_net.py
"""
import os
import sys

import pandas as pd

CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "paper_part1",
                          "pathway_net_c5bp_cache.parquet")

if os.path.exists(CACHE_PATH):
    net = pd.read_parquet(CACHE_PATH)
    print(f"Cache already exists: {net.shape[0]} rows, {net['source'].nunique()} pathways")
    sys.exit(0)

import peach as pc
print("Downloading c5_bp from MSigDB...")
net = pc.pp.load_pathway_networks(sources=["c5_bp"])
print(f"Downloaded: {net.shape[0]} rows, {net['source'].nunique()} pathways")
net.to_parquet(CACHE_PATH)
print(f"Saved to {CACHE_PATH}")

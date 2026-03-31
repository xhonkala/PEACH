#!/usr/bin/env python
"""Programmatic audit: compare types_index.py claims against actual return types and uns keys."""
import inspect
import json
import numpy as np

import peach.tl as tl
import peach.pp as pp
import peach.pl as pl

# Load types_index
from peach._core.types_index import get_return_type, ADATA_KEYS, USE_GET_FOR

print("=" * 80)
print("DOCUMENTATION DRIFT AUDIT: types_index.py vs actual return annotations")
print("=" * 80)

# Load actual signatures
with open("/tmp/peach_actual_sigs.json") as f:
    actual = json.load(f)

# Check return types for all tl functions
print("\n--- RETURN TYPE COMPARISON (tl.*) ---\n")
mismatches = []
for func_key, sig in sorted(actual.items()):
    if not func_key.startswith("tl."):
        continue
    func_name = func_key  # e.g., "tl.train_archetypal"
    actual_return = sig["return"]

    try:
        documented_type, documented_keys = get_return_type(func_name)
    except (KeyError, ValueError):
        print(f"  NOT IN types_index: {func_name} (actual returns: {actual_return[:60]})")
        continue

    # Normalize actual return for comparison
    actual_short = actual_return.replace("<class '", "").replace("'>", "").split(".")[-1]

    # Check if documented type matches actual
    if documented_type and actual_short:
        # dict vs Pydantic type name
        if actual_short == "dict" and documented_type not in ("dict", "None"):
            status = "DRIFT"
        elif actual_short != documented_type and documented_type not in actual_return:
            status = "MISMATCH"
        else:
            status = "OK"

        if status != "OK":
            mismatches.append((func_name, documented_type, actual_short, actual_return))
            print(f"  {status}: {func_name}")
            print(f"    types_index says: {documented_type}")
            print(f"    actual annotation: {actual_return[:80]}")
            if documented_keys:
                print(f"    documented keys: {documented_keys[:5]}...")
            print()

print(f"\n  Total tl.* functions: {sum(1 for k in actual if k.startswith('tl.'))}")
print(f"  Return type mismatches: {len(mismatches)}")

# Check ADATA_KEYS
print("\n--- ADATA_KEYS SPOT CHECK ---\n")
for storage_type in ["obsm", "obs", "uns"]:
    if storage_type in ADATA_KEYS:
        keys = ADATA_KEYS[storage_type]
        print(f"  {storage_type}: {len(keys)} documented keys")
        for key_name, description in list(keys.items())[:5]:
            print(f"    '{key_name}': {description[:60]}")
        if len(keys) > 5:
            print(f"    ... and {len(keys) - 5} more")

# Check USE_GET_FOR
print(f"\n--- USE_GET_FOR ---")
print(f"  {len(USE_GET_FOR)} optional fields documented")
for f in sorted(USE_GET_FOR)[:10]:
    print(f"    {f}")
if len(USE_GET_FOR) > 10:
    print(f"    ... and {len(USE_GET_FOR) - 10} more")

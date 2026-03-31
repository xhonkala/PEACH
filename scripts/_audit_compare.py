#!/usr/bin/env python
"""Compare actual function signatures against tools_schema.py documentation."""
import json
import sys
sys.path.insert(0, "src")

from peach._core.tools_schema import get_tool_schema, TOOL_SCHEMAS

# Load actual signatures
with open("/tmp/peach_actual_sigs.json") as f:
    actual = json.load(f)

print("=" * 80)
print("DOCUMENTATION DRIFT AUDIT: tools_schema.py vs actual signatures")
print("=" * 80)

# What's in the registry?
documented = set()
for key in TOOL_SCHEMAS:
    documented.add(key)

actual_keys = set(actual.keys())

# Map schema keys (like "tl.train_archetypal") to actual keys
print(f"\n--- COVERAGE ---")
print(f"Documented in tools_schema: {len(documented)}")
print(f"Actually exported: {len(actual_keys)}")

# Find what's documented but not exported
doc_only = documented - actual_keys
if doc_only:
    print(f"\nIn schema but NOT exported ({len(doc_only)}):")
    for k in sorted(doc_only):
        print(f"  {k}")

# Find what's exported but not documented
actual_only = actual_keys - documented
if actual_only:
    print(f"\nExported but NOT in schema ({len(actual_only)}):")
    for k in sorted(actual_only):
        print(f"  {k}")

# For functions in both, compare parameters
print(f"\n--- PARAMETER COMPARISON ---\n")
both = documented & actual_keys
n_perfect = 0
n_mismatch = 0

for func_key in sorted(both):
    schema = get_tool_schema(func_key)
    if schema is None:
        continue

    actual_params = actual[func_key]["params"]
    schema_params = {p.name: p for p in schema.parameters}

    # Compare
    schema_names = set(schema_params.keys()) - {"adata", "adata_key"}
    actual_names = set(actual_params.keys()) - {"adata", "kwargs"}

    in_schema_not_actual = schema_names - actual_names
    in_actual_not_schema = actual_names - schema_names

    if not in_schema_not_actual and not in_actual_not_schema:
        n_perfect += 1
        continue

    n_mismatch += 1
    print(f"MISMATCH: {func_key}")
    if in_schema_not_actual:
        print(f"  In schema but NOT in source: {sorted(in_schema_not_actual)}")
    if in_actual_not_schema:
        print(f"  In source but NOT in schema: {sorted(in_actual_not_schema)}")

    # Check defaults for shared params
    shared = schema_names & actual_names
    for pname in sorted(shared):
        sp = schema_params[pname]
        ap = actual_params[pname]
        schema_default = str(sp.default) if sp.default is not None else "None"
        actual_default = ap["default"]
        if actual_default == "__REQUIRED__":
            actual_default = "REQUIRED"
        # Normalize for comparison
        sd_norm = schema_default.strip("'\"")
        ad_norm = actual_default.strip("'\"")
        if sd_norm != ad_norm and not (sd_norm == "None" and ad_norm == "None"):
            print(f"  Default mismatch '{pname}': schema={schema_default}, actual={actual_default}")
    print()

print(f"\n--- SUMMARY ---")
print(f"Functions in both:    {len(both)}")
print(f"Perfect match:        {n_perfect}")
print(f"Parameter mismatch:   {n_mismatch}")
print(f"In schema only:       {len(doc_only)}")
print(f"In source only:       {len(actual_only)}")

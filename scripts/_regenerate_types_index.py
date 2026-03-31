#!/usr/bin/env python
"""Regenerate types_index.py from ground-truth return annotations and adata storage.

Extracts return types from inspect, parses docstrings for key fields and
adata storage locations. Outputs the complete types_index module.
"""
import inspect
import re

import peach.pp as pp
import peach.tl as tl
import peach.pl as pl


def get_return_info(func):
    """Extract return type and docstring info about return keys."""
    sig = inspect.signature(func)
    ret = sig.return_annotation
    doc = inspect.getdoc(func) or ""

    # Clean return annotation
    if ret is inspect.Parameter.empty:
        ret_str = "unspecified"
    else:
        ret_str = str(ret)
        ret_str = ret_str.replace("<class '", "").replace("'>", "")
        # Simplify
        if "dict" in ret_str:
            ret_str = "dict"
        elif "DataFrame" in ret_str:
            ret_str = "DataFrame"
        elif "ndarray" in ret_str:
            ret_str = "np.ndarray"
        elif "AnnData" in ret_str:
            ret_str = "AnnData"
        elif "Figure" in ret_str or "plotly" in ret_str or "matplotlib" in ret_str:
            ret_str = "Figure"
        elif ret_str == "None" or ret_str == "<class 'NoneType'>":
            ret_str = "None"
        else:
            ret_str = ret_str.split(".")[-1]

    # Parse Returns section from docstring
    return_keys = []
    in_returns = False
    for line in doc.split("\n"):
        stripped = line.strip()
        if stripped in ("Returns", "Returns:"):
            in_returns = True
            continue
        if stripped.startswith("---") and in_returns:
            continue
        if stripped in ("Raises", "Raises:", "Notes", "Notes:", "Examples",
                       "Examples:", "See Also", "Parameters", "Parameters:"):
            in_returns = False
            continue
        if in_returns and stripped:
            return_keys.append(stripped)

    # Parse for adata modifications
    modifies = []
    for pattern in [
        r"adata\.uns\['([^']+)'\]",
        r"adata\.obsm\['([^']+)'\]",
        r"adata\.obs\['([^']+)'\]",
        r"adata\.obsp\['([^']+)'\]",
    ]:
        for match in re.finditer(pattern, doc):
            key = match.group(1)
            # Determine storage type
            if ".uns[" in match.group(0):
                modifies.append(f"uns['{key}']")
            elif ".obsm[" in match.group(0):
                modifies.append(f"obsm['{key}']")
            elif ".obs[" in match.group(0):
                modifies.append(f"obs['{key}']")
            elif ".obsp[" in match.group(0):
                modifies.append(f"obsp['{key}']")

    return ret_str, return_keys, modifies


def main():
    modules = {"pp": pp, "tl": tl, "pl": pl}

    # Collect all function info
    all_funcs = {}
    for mod_name, mod in sorted(modules.items()):
        for name in sorted(dir(mod)):
            if name.startswith("_"):
                continue
            obj = getattr(mod, name)
            if not callable(obj) or isinstance(obj, type):
                continue
            key = f"{mod_name}.{name}"
            try:
                ret_str, return_keys, modifies = get_return_info(obj)
                all_funcs[key] = {
                    "return_type": ret_str,
                    "return_keys": return_keys,
                    "modifies": modifies,
                }
            except Exception as e:
                all_funcs[key] = {
                    "return_type": f"ERROR: {e}",
                    "return_keys": [],
                    "modifies": [],
                }

    # Output the module
    print('"""PEACH types index — AUTO-GENERATED from inspect + docstring parsing.')
    print()
    print("DO NOT EDIT MANUALLY. Regenerate with: python scripts/_regenerate_types_index.py")
    print()
    print("Maps every public PEACH function to its return type and key fields.")
    print('"""')
    print()
    print("from __future__ import annotations")
    print()

    # RETURN_TYPES dict
    print("# Function -> (return_type, [key_fields])")
    print("RETURN_TYPES: dict[str, tuple[str, list[str]]] = {")
    for key, info in sorted(all_funcs.items()):
        ret = info["return_type"]
        keys = info["return_keys"]
        # Format keys as list of strings, max 5 per entry
        if len(keys) > 5:
            keys_str = repr(keys[:5])[:-1] + ", '...']"
        else:
            keys_str = repr(keys) if keys else "[]"
        print(f'    "{key}": ("{ret}", {keys_str}),')
    print("}")
    print()

    # get_return_type function
    print("""
def get_return_type(func_name: str) -> tuple[str, list[str]]:
    \"\"\"Get return type and key fields for a PEACH function.

    Parameters
    ----------
    func_name : str
        Function name, e.g. 'tl.train_archetypal'

    Returns
    -------
    tuple of (type_name, key_fields)
    \"\"\"
    if func_name in RETURN_TYPES:
        return RETURN_TYPES[func_name]
    raise KeyError(f"No return type for '{func_name}'. Available: {list(RETURN_TYPES.keys())}")
""")

    # ADATA_KEYS — scan all docstrings for adata storage patterns
    print()
    print("# adata storage keys extracted from docstrings")
    obsm_keys = {}
    obs_keys = {}
    uns_keys = {}
    obsp_keys = {}

    for key, info in all_funcs.items():
        for mod_entry in info["modifies"]:
            if mod_entry.startswith("obsm["):
                k = mod_entry.split("'")[1]
                obsm_keys[k] = f"Set by {key}"
            elif mod_entry.startswith("obs["):
                k = mod_entry.split("'")[1]
                obs_keys[k] = f"Set by {key}"
            elif mod_entry.startswith("uns["):
                k = mod_entry.split("'")[1]
                uns_keys[k] = f"Set by {key}"
            elif mod_entry.startswith("obsp["):
                k = mod_entry.split("'")[1]
                obsp_keys[k] = f"Set by {key}"

    print("ADATA_KEYS = {")
    print('    "obsm": {')
    for k, v in sorted(obsm_keys.items()):
        print(f'        "{k}": "{v}",')
    print("    },")
    print('    "obs": {')
    for k, v in sorted(obs_keys.items()):
        print(f'        "{k}": "{v}",')
    print("    },")
    print('    "uns": {')
    for k, v in sorted(uns_keys.items()):
        print(f'        "{k}": "{v}",')
    print("    },")
    print('    "obsp": {')
    for k, v in sorted(obsp_keys.items()):
        print(f'        "{k}": "{v}",')
    print("    },")
    print("}")
    print()

    # USE_GET_FOR — fields that are optional in return dicts
    print("# Optional fields that may not be present in return dicts.")
    print("# Always use .get() for these.")
    print("USE_GET_FOR = {")
    # These are manually curated based on known optional fields
    optional_fields = [
        "final_archetype_r2",
        "final_model",
        "model",
        "convergence_epoch",
        "degree_comparison",
        "interaction_coefficients",
        "interaction_pairs",
        "interaction_pvalues",
        "interaction_pvalues_fdr",
        "vertex_covariance",
        "bootstrap_ci_lower",
        "bootstrap_ci_upper",
        "residuals",
        "holdout_mmd",
        "holdout_fraction",
        "alignment_pvalues",
        "alignment_pvalues_fdr",
        "expansion_pvalues",
        "expansion_pvalues_fdr",
        "per_cell_alignment",
        "per_cell_expansion",
        "per_cell_expansion_gene_names",
        "per_cell_gene_names",
        "component_feature_profiles",
    ]
    for f in sorted(optional_fields):
        print(f'    "{f}",')
    print("}")
    print()

    # PITFALLS
    print('PITFALLS = {')
    print('    "final_archetype_r2": "Optional in TrainingResults - use .get(). Train-mode R2 includes reparameterization noise; eval-mode is the real metric.",')
    print('    "X_pca": "Check for X_pca, X_PCA, or pca variants in adata.obsm.",')
    print('    "CVSummary_ranking": "Use ranked[i][\'metric_value\'] NOT ranked[i].mean_archetype_r2.",')
    print('    "distance_vs_weight": "Distance-based and weight-based archetype assignment disagree for ~60% of cells - this is expected.",')
    print('    "FRGeom_returns": "torch.Tensor, NOT numpy arrays.",')
    print("}")


if __name__ == "__main__":
    main()

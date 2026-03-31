#!/usr/bin/env python
"""Programmatic audit: extract actual function signatures from PEACH modules."""
import inspect
import json
import peach.pp as pp
import peach.tl as tl
import peach.pl as pl

modules = {"pp": pp, "tl": tl, "pl": pl}
signatures = {}

for mod_name, mod in modules.items():
    for name in sorted(dir(mod)):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name)
        if not callable(obj):
            continue
        try:
            sig = inspect.signature(obj)
        except (ValueError, TypeError):
            continue

        params = {}
        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            d = param.default
            if d is inspect.Parameter.empty:
                default_str = "__REQUIRED__"
            elif d is None:
                default_str = "None"
            else:
                default_str = repr(d)

            annotation = param.annotation
            if annotation is inspect.Parameter.empty:
                type_str = "Any"
            else:
                type_str = str(annotation)

            params[pname] = {
                "default": default_str,
                "type": type_str,
                "kind": str(param.kind),
            }

        func_key = f"{mod_name}.{name}"
        signatures[func_key] = {
            "params": params,
            "return": str(sig.return_annotation) if sig.return_annotation is not inspect.Parameter.empty else "unspecified",
        }

# Print as JSON for machine consumption
print(json.dumps(signatures, indent=2))

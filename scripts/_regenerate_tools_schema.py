#!/usr/bin/env python
"""Regenerate tools_schema.py from ground-truth function signatures.

Extracts parameters via inspect.signature(), descriptions from docstrings,
return types from annotations. Outputs the complete TOOL_SCHEMAS dict.
"""
import inspect
import re
import textwrap
from typing import Any

import peach.pp as pp
import peach.tl as tl
import peach.pl as pl


def get_param_type_str(annotation, default):
    """Map Python type annotation to ParamType string."""
    ann_str = str(annotation) if annotation is not inspect.Parameter.empty else ""

    if "AnnData" in ann_str:
        return "ParamType.ADATA_REF"
    if "bool" in ann_str.lower():
        return "ParamType.BOOLEAN"
    if "int" in ann_str.lower() and "float" not in ann_str.lower():
        return "ParamType.INTEGER"
    if "float" in ann_str.lower() or "number" in ann_str.lower():
        return "ParamType.FLOAT"
    if "list" in ann_str.lower() or "tuple" in ann_str.lower():
        return "ParamType.ARRAY"
    if "dict" in ann_str.lower():
        return "ParamType.OBJECT"
    if "str" in ann_str.lower():
        return "ParamType.STRING"

    # Infer from default
    if default is not inspect.Parameter.empty and default is not None:
        if isinstance(default, bool):
            return "ParamType.BOOLEAN"
        if isinstance(default, int):
            return "ParamType.INTEGER"
        if isinstance(default, float):
            return "ParamType.FLOAT"
        if isinstance(default, str):
            return "ParamType.STRING"
        if isinstance(default, (list, tuple)):
            return "ParamType.ARRAY"
        if isinstance(default, dict):
            return "ParamType.OBJECT"

    return "ParamType.STRING"  # fallback


def parse_docstring_params(func):
    """Extract parameter descriptions from numpy-style docstring."""
    doc = inspect.getdoc(func)
    if not doc:
        return {}, ""

    # Get first line as description
    lines = doc.strip().split("\n")
    description = lines[0].strip()

    # Parse Parameters section
    param_descs = {}
    in_params = False
    current_param = None
    current_desc = []

    for line in lines:
        stripped = line.strip()
        if stripped in ("Parameters", "Parameters:"):
            in_params = True
            continue
        if stripped.startswith("---") and in_params:
            continue
        if stripped in ("Returns", "Returns:", "Raises", "Raises:", "Notes", "Notes:",
                       "Examples", "Examples:", "See Also", "See Also:"):
            if current_param:
                param_descs[current_param] = " ".join(current_desc).strip()
            in_params = False
            continue

        if in_params:
            # Check if this is a parameter definition line
            match = re.match(r'^(\w+)\s*:', stripped)
            if match and not stripped.startswith("    "):
                if current_param:
                    param_descs[current_param] = " ".join(current_desc).strip()
                current_param = match.group(1)
                # Description might be on the same line after the type
                rest = stripped[match.end():].strip()
                current_desc = [rest] if rest else []
            elif current_param and stripped:
                current_desc.append(stripped)

    if current_param:
        param_descs[current_param] = " ".join(current_desc).strip()

    return param_descs, description


def get_return_annotation(func):
    """Get the return type annotation as a clean string."""
    sig = inspect.signature(func)
    ret = sig.return_annotation
    if ret is inspect.Parameter.empty:
        return "unspecified"
    ret_str = str(ret)
    # Clean up
    ret_str = ret_str.replace("<class '", "").replace("'>", "")
    if "dict" in ret_str:
        return "dict"
    if "DataFrame" in ret_str:
        return "DataFrame"
    if "ndarray" in ret_str:
        return "np.ndarray"
    if "AnnData" in ret_str:
        return "AnnData"
    if "Figure" in ret_str:
        return "Figure"
    if ret_str == "None":
        return "None"
    return ret_str.split(".")[-1]


def generate_schema_entry(mod_name, func_name, func):
    """Generate a single ToolSchema entry."""
    sig = inspect.signature(func)
    param_descs, description = parse_docstring_params(func)
    ret_type = get_return_annotation(func)

    params_code = []
    for pname, param in sig.parameters.items():
        if pname in ("self", "kwargs"):
            continue

        ptype = get_param_type_str(param.annotation, param.default)
        default = param.default
        required = default is inspect.Parameter.empty

        # Format default value
        if required:
            default_str = "None"
        elif default is None:
            default_str = "None"
        elif isinstance(default, bool):
            default_str = str(default)
        elif isinstance(default, str):
            default_str = f"'{default}'"
        elif isinstance(default, (list, tuple)):
            default_str = repr(default)
        else:
            default_str = repr(default)

        pdesc = param_descs.get(pname, "")
        # Truncate long descriptions and escape quotes
        if len(pdesc) > 80:
            pdesc = pdesc[:77] + "..."
        pdesc = pdesc.replace("\\", "\\\\").replace('"', '\\"')

        params_code.append(
            f'        Parameter("{pname}", {ptype}, "{pdesc}", '
            f'required={required}, default={default_str})'
        )

    params_str = ",\n".join(params_code)
    key = f"{mod_name}.{func_name}"

    # Escape description for string
    description = description.replace('"', '\\"')

    return f'''    "{key}": ToolSchema(
        name="{key}",
        description="{description}",
        parameters=[
{params_str},
        ],
        returns="{ret_type}",
    ),'''


def main():
    modules = {"pp": pp, "tl": tl, "pl": pl}

    # Header
    print('"""PEACH tools schema — AUTO-GENERATED from inspect.signature().')
    print()
    print("DO NOT EDIT MANUALLY. Regenerate with: python scripts/_regenerate_tools_schema.py")
    print('"""')
    print()
    print("from __future__ import annotations")
    print()
    print("from dataclasses import dataclass, field")
    print("from enum import Enum")
    print("from typing import Any")
    print()

    # ParamType enum
    print("""
class ParamType(Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    ADATA_REF = "adata_reference"


@dataclass
class Parameter:
    name: str
    type: ParamType
    description: str
    required: bool = False
    default: Any = None
    enum: list[str] | None = None
    items_type: ParamType | None = None


@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: list[Parameter] = field(default_factory=list)
    returns: str = ""
    returns_description: str = ""
    modifies_adata: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)

    def to_tool_definition(self) -> dict[str, Any]:
        props = {}
        required = []
        for p in self.parameters:
            prop = {"type": p.type.value, "description": p.description}
            if p.default is not None:
                prop["default"] = p.default
            if p.enum:
                prop["enum"] = p.enum
            props[p.name] = prop
            if p.required:
                required.append(p.name)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        }
""")

    # Generate all schemas
    print()
    print("TOOL_SCHEMAS: dict[str, ToolSchema] = {")

    for mod_name, mod in sorted(modules.items()):
        print(f"\n    # --- {mod_name} module ---")
        for name in sorted(dir(mod)):
            if name.startswith("_"):
                continue
            obj = getattr(mod, name)
            if not callable(obj) or isinstance(obj, type):
                continue
            try:
                entry = generate_schema_entry(mod_name, name, obj)
                print(entry)
            except Exception as e:
                print(f'    # ERROR generating schema for {mod_name}.{name}: {e}')

    print("}")
    print()

    # Utility functions
    print("""
def get_tool_schema(func_name: str) -> ToolSchema:
    if func_name in TOOL_SCHEMAS:
        return TOOL_SCHEMAS[func_name]
    raise KeyError(f"No schema for '{func_name}'. Available: {list(TOOL_SCHEMAS.keys())}")


def generate_tool_definitions(func_names: list[str] | None = None) -> list[dict[str, Any]]:
    if func_names is None:
        func_names = list(TOOL_SCHEMAS.keys())
    return [TOOL_SCHEMAS[name].to_tool_definition() for name in func_names]


def print_tool_summary():
    print("=" * 70)
    print("PEACH TOOLS SUMMARY")
    print("=" * 70)
    for name, schema in sorted(TOOL_SCHEMAS.items()):
        n_params = len(schema.parameters)
        print(f"  {name:45s} {n_params:2d} params -> {schema.returns}")
    print(f"\\nTotal: {len(TOOL_SCHEMAS)} tools")
""")


if __name__ == "__main__":
    main()

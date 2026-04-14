"""Loader for the curated stress-response gene signatures in mmc2.xlsx.

The source workbook stores the signatures in a multi-column format on the
"Stress signatures" tab: row 0 is a free-text title, row 1 is the
column header (signature name), and rows 2+ are gene symbols with NaN
padding at the tail of shorter columns. This module flattens that
layout into:

  - ``STRESS_SIGNATURES`` : dict[str, list[str]] — signature name to
    unique, sorted list of gene symbols
  - ``STRESS_GENES_FLAT`` : list[str] — union of all signature genes,
    sorted and deduplicated
  - ``STRESS_GENE_TO_SIGS`` : dict[str, list[str]] — reverse map from
    each gene symbol to the signatures that contain it

The loader also writes two artefacts alongside the workbook so
downstream code can avoid pulling in pandas / openpyxl just to filter:

  - ``stress_genes_flat.txt`` : one gene per line
  - ``stress_signatures.json`` : ``{signature: [genes], ...}`` plus a
    ``_flat`` key with the deduplicated union

Signatures in the workbook
--------------------------
HSR  – Heat Shock Response
OSR  – Oxidative Stress Response
UPR  – Unfolded Protein Response
HySR – Hypoxic Stress Response
DDR  – DNA Damage Response
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import pandas as pd

_XLSX_PATH = os.path.join(os.path.dirname(__file__), "mmc2.xlsx")
_SHEET_STRESS = "Stress signatures"
_FLAT_TXT = os.path.join(os.path.dirname(__file__), "stress_genes_flat.txt")
_SIG_JSON = os.path.join(os.path.dirname(__file__), "stress_signatures.json")


def _parse_stress_sheet(xlsx_path: str = _XLSX_PATH) -> Dict[str, List[str]]:
    """Parse the 'Stress signatures' sheet into {signature: [genes]}.

    Row 0 is a free-text title; row 1 holds the signature column
    headers. Gene symbols start at row 2. Blank / NaN cells in each
    column's tail are dropped; remaining values are upper-cased,
    deduplicated, and sorted for deterministic downstream filtering.
    """
    df = pd.read_excel(xlsx_path, sheet_name=_SHEET_STRESS, header=None)
    # Header row = index 1 (row 0 is a free-text title)
    header_row = df.iloc[1].astype(str).str.strip().tolist()
    out: Dict[str, List[str]] = {}
    for col_idx, sig_name in enumerate(header_row):
        if not sig_name or sig_name.lower() == "nan":
            continue
        col = df.iloc[2:, col_idx]
        genes = (
            col.dropna()
            .astype(str)
            .str.strip()
            .str.upper()
        )
        genes = genes[genes != ""]
        out[sig_name] = sorted(set(genes.tolist()))
    return out


def _build_reverse_map(sigs: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """gene -> [signatures that contain it]."""
    rev: Dict[str, List[str]] = {}
    for sig, genes in sigs.items():
        for g in genes:
            rev.setdefault(g, []).append(sig)
    for g, sig_list in rev.items():
        rev[g] = sorted(set(sig_list))
    return rev


def load_stress_signatures(xlsx_path: str = _XLSX_PATH):
    """Return (sigs_dict, flat_list, reverse_map).

    Results are cached on the module level via ``STRESS_*`` constants.
    """
    sigs = _parse_stress_sheet(xlsx_path)
    flat = sorted({g for genes in sigs.values() for g in genes})
    rev = _build_reverse_map(sigs)
    return sigs, flat, rev


def regenerate_artefacts(xlsx_path: str = _XLSX_PATH) -> None:
    """Rewrite ``stress_genes_flat.txt`` and ``stress_signatures.json``.

    Call this after the source workbook changes.
    """
    sigs, flat, _ = load_stress_signatures(xlsx_path)
    with open(_FLAT_TXT, "w") as f:
        f.write("\n".join(flat) + "\n")
    payload = {**sigs, "_flat": flat}
    with open(_SIG_JSON, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


# Eager module-level constants so downstream code can `from
# stress_genes.load_stress_genes import STRESS_GENES_FLAT`.
try:
    STRESS_SIGNATURES, STRESS_GENES_FLAT, STRESS_GENE_TO_SIGS = load_stress_signatures()
except Exception:
    STRESS_SIGNATURES = {}
    STRESS_GENES_FLAT = []
    STRESS_GENE_TO_SIGS = {}


if __name__ == "__main__":
    regenerate_artefacts()
    print(f"Signatures: {list(STRESS_SIGNATURES.keys())}")
    for sig, genes in STRESS_SIGNATURES.items():
        print(f"  {sig}: {len(genes)} genes")
    print(f"Flat union: {len(STRESS_GENES_FLAT)} unique genes")
    print(f"Wrote: {_FLAT_TXT}")
    print(f"Wrote: {_SIG_JSON}")

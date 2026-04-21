"""Pull GO:0006950 (response to stress) + all descendants, propagated to gene symbols."""
from goatools.obo_parser import GODag
from goatools.anno.gaf_reader import GafReader
from collections import defaultdict
import json

OBO = "/Users/honkala/Desktop/FRTNBC/data/gene_sets/go-basic.obo"
GAF = "/Users/honkala/Desktop/FRTNBC/data/gene_sets/goa_human.gaf"
ROOT = "GO:0006950"

godag = GODag(OBO, optional_attrs={"relationship"})
root_term = godag[ROOT]
descendants = root_term.get_all_children() | {ROOT}
print(f"GO:0006950 + descendants: {len(descendants)} terms")

gaf = GafReader(GAF)
ns2assoc = gaf.get_ns2assc()
bp = ns2assoc["BP"]

term2genes = defaultdict(set)
gene_set = set()
for gene_uniprot, terms in bp.items():
    hits = terms & descendants
    if hits:
        for t in hits:
            term2genes[t].add(gene_uniprot)
        gene_set.add(gene_uniprot)

print(f"Unique UniProt accessions annotated: {len(gene_set)}")

import csv
uniprot2symbol = {}
with open(GAF) as f:
    for line in f:
        if line.startswith("!"):
            continue
        parts = line.split("\t")
        if len(parts) < 11:
            continue
        if parts[0] != "UniProtKB":
            continue
        upid = parts[1]
        symbol = parts[2]
        uniprot2symbol[upid] = symbol

symbols = sorted({uniprot2symbol[u] for u in gene_set if u in uniprot2symbol})
print(f"Unique gene symbols: {len(symbols)}")

# Also build per-term gene lists (symbol-level), with term names
term_records = []
for t, ups in term2genes.items():
    syms = sorted({uniprot2symbol[u] for u in ups if u in uniprot2symbol})
    if syms:
        term_records.append({
            "go_id": t,
            "name": godag[t].name if t in godag else t,
            "depth": godag[t].depth if t in godag else None,
            "n_genes": len(syms),
            "genes": syms,
        })
term_records.sort(key=lambda x: -x["n_genes"])

OUT_DIR = "/Users/honkala/Desktop/FRTNBC/data/gene_sets"
with open(f"{OUT_DIR}/GO_0006950_response_to_stress.symbols.txt", "w") as f:
    for s in symbols:
        f.write(s + "\n")

with open(f"{OUT_DIR}/GO_0006950_per_term.json", "w") as f:
    json.dump(term_records, f, indent=2)

# Also a compact GMT (Broad-style) for downstream tooling
with open(f"{OUT_DIR}/GO_0006950_response_to_stress.gmt", "w") as f:
    f.write("GO_0006950_response_to_stress\thttp://amigo.geneontology.org/amigo/term/GO:0006950\t" + "\t".join(symbols) + "\n")
    for rec in term_records:
        if rec["n_genes"] >= 10:
            name = rec["name"].replace("\t", " ")
            f.write(f"{rec['go_id']}_{name}\thttp://amigo.geneontology.org/amigo/term/{rec['go_id']}\t" + "\t".join(rec["genes"]) + "\n")

print("Wrote:")
print(f"  {OUT_DIR}/GO_0006950_response_to_stress.symbols.txt  (n={len(symbols)})")
print(f"  {OUT_DIR}/GO_0006950_per_term.json  ({len(term_records)} child terms with gene records)")
print(f"  {OUT_DIR}/GO_0006950_response_to_stress.gmt")

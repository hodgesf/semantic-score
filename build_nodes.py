"""Build data/nodes.parquet from the tier-0 nodes.jsonl in one pass.

Keeps everything a semantics-only score may use: name, description, full
Biolink lineage, most specific category, identifier namespaces, taxon and
symbol attributes. The information_content column is kept for evaluation only.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from paths import DATA, TIER0_DIR

NODES_FILE = TIER0_DIR / "nodes.jsonl"
OUT = DATA / "nodes.parquet"

_GENERIC_CATS = {
    "biolink:NamedThing",
    "biolink:Entity",
    "biolink:BiologicalEntity",
    "biolink:OntologyClass",
    "biolink:ThingWithTaxon",
    "biolink:PhysicalEssence",
    "biolink:PhysicalEssenceOrOccurrent",
    "biolink:Occurrent",
    "biolink:PhysicalEntity",
}


def most_specific_category(cats: list[str]) -> str:
    """Return the first category that is neither generic, a union nor a mixin."""
    for c in cats:
        if c in _GENERIC_CATS or "Or" in c or c.endswith("Mixin"):
            continue
        return c
    return cats[0] if cats else ""


def node_row(rec: dict[str, Any]) -> dict[str, Any]:
    """Flatten one nodes.jsonl record into a parquet row."""
    cats = rec["category"]
    eq = rec.get("equivalent_identifiers", [])
    prefixes = sorted({e.split(":", 1)[0] for e in eq})
    return {
        "id": rec["id"],
        "prefix": rec["id"].split(":", 1)[0],
        "name": rec.get("name", "") or "",
        "description": rec.get("description"),
        "category": most_specific_category(cats),
        "lineage": "|".join(c for c in cats if c not in _GENERIC_CATS),
        "namespaces": "|".join(prefixes),
        "n_namespaces": len(prefixes),
        "has_inchikey": "INCHIKEY" in prefixes,
        "taxon": (rec.get("in_taxon") or rec.get("taxon") or [""])[0]
        if isinstance(rec.get("in_taxon") or rec.get("taxon"), list)
        else (rec.get("in_taxon") or rec.get("taxon") or ""),
        "symbol": rec.get("symbol") or "",
        "information_content": rec.get("information_content"),
    }


def read_rows() -> list[dict[str, Any]]:
    """Read every record from NODES_FILE, printing progress every 250k rows."""
    rows: list[dict[str, Any]] = []
    with open(NODES_FILE, encoding="utf-8") as handle:
        for line in handle:
            rows.append(node_row(json.loads(line)))
            if len(rows) % 250_000 == 0:
                print(f"  {len(rows):,}", flush=True)
    return rows


def main() -> None:
    """Build the node table and write it to OUT."""
    df = pd.DataFrame.from_records(read_rows())
    for col in ("prefix", "category"):
        df[col] = df[col].astype("category")
    df.to_parquet(OUT, index=False)
    print(df.shape, "->", OUT)
    print(df["prefix"].value_counts().head(20))


if __name__ == "__main__":
    main()

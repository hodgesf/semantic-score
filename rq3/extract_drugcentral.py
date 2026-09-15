"""Extract every tier0 edge with infores:drugcentral among its sources.

Reads edges.jsonl from TIER0_DIR and records subject, predicate, object, primary source,
qualifiers, knowledge level and agent type. Output: data/drugcentral_edges.parquet, from which
the RQ3 precision@k ground truth is derived (predicate choice in rq3_spec.md).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import RQ3, TIER0_DIR  # noqa: E402

EDGES = TIER0_DIR / "edges.jsonl"
OUT = RQ3 / "data" / "drugcentral_edges.parquet"


def edge_row(e: dict[str, Any]) -> dict[str, Any] | None:
    """Return the record for an edge whose sources include DrugCentral, else None."""
    srcs = e.get("sources") or []
    if not any(s.get("resource_id") == "infores:drugcentral" for s in srcs):
        return None
    prim = next(
        (s["resource_id"] for s in srcs if s.get("resource_role") == "primary_knowledge_source"),
        None,
    )
    return {
        "subject": e["subject"],
        "predicate": e["predicate"],
        "object": e["object"],
        "primary_source": prim,
        "qualifiers": json.dumps(e.get("qualifiers") or [], sort_keys=True),
        "knowledge_level": e.get("knowledge_level"),
        "agent_type": e.get("agent_type"),
    }


def main() -> None:
    """Stream the grep-prefiltered edges, keep the DrugCentral ones and write OUT."""
    rows = []
    proc = subprocess.Popen(
        ["grep", "infores:drugcentral", str(EDGES)], stdout=subprocess.PIPE, text=True
    )
    for line in proc.stdout:
        row = edge_row(json.loads(line))
        if row is not None:
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    print(len(df))
    print(df.groupby(["predicate", "primary_source"]).size().sort_values(ascending=False).head(20))


if __name__ == "__main__":
    main()

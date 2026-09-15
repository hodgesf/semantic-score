"""Split the DrugCentral treats subjects into a training half and a held-back half.

Reads rq3/data/drugcentral_edges.parquet, takes every distinct subject of a
biolink:treats edge, and assigns it to "train" when the MD5 hash of its identifier
is even and to "dev" otherwise. Writes v2/drugcentral_split.parquet.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import RQ3, V2  # noqa: E402


def half(identifier: str) -> str:
    """Return "train" for an even MD5 hash of the identifier and "dev" otherwise."""
    digest = int(hashlib.md5(identifier.encode()).hexdigest(), 16)
    return "train" if digest % 2 == 0 else "dev"


def main() -> None:
    """Write the split table for every DrugCentral treats subject."""
    edges = pd.read_parquet(RQ3 / "data" / "drugcentral_edges.parquet")
    drugs = sorted(set(edges.query("predicate == 'biolink:treats'").subject))
    out = pd.DataFrame({"id": drugs, "split": [half(i) for i in drugs]})
    out.to_parquet(V2 / "drugcentral_split.parquet", index=False)
    print(out.split.value_counts().to_dict())


if __name__ == "__main__":
    main()

"""Draw the held-out replication query set for RQ3.

200 further diseases are drawn (seed 2655) from the same candidate pool as draw_queries.py (at
least two DrugCentral indications in tier0, MONDO ids, tier0 category Disease, not among the 48
sprint-6 test-suite diseases), additionally excluding the original 200 in data/query_set_wide.csv.
The v2 ensemble, its epoch selection and the cutoff (0) were fixed before this set was drawn.
Output: data/query_set_heldout.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from draw_queries import candidate_pool, draw, report, test_suite_diseases  # noqa: E402

from paths import RQ3  # noqa: E402

SEED = 2655
PREFIX = "H"
OUT = RQ3 / "data" / "query_set_heldout.csv"


def main() -> None:
    """Draw the held-out set and write it to OUT."""
    excl = test_suite_diseases()
    excl |= set(pd.read_csv(RQ3 / "data" / "query_set_wide.csv").disease)
    cand, nodes = candidate_pool(excl)
    q = draw(cand, nodes, SEED, PREFIX)
    q.to_csv(OUT, index=False)
    report(cand, q)


if __name__ == "__main__":
    main()

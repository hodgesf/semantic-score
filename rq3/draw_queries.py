"""Draw the wide treats query set for RQ3.

200 diseases are drawn at random (seed 2654) from the diseases with at least two DrugCentral
indications in tier0, restricted to MONDO ids with tier0 category Disease and excluding the 48
sprint-6 test-suite diseases. Output: data/query_set_wide.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, RQ3, TRANSLATOR_TESTS  # noqa: E402

SEED = 2654
N_QUERIES = 200
PREFIX = "W"
OUT = RQ3 / "data" / "query_set_wide.csv"


def test_suite_diseases() -> set[str]:
    """Return the input diseases of every sprint-6 test case."""
    return {
        v["test_case_input_id"] for v in json.load(open(TRANSLATOR_TESTS))["test_cases"].values()
    }


def candidate_pool(excl: set[str]) -> tuple[pd.Series, pd.DataFrame]:
    """Return DrugCentral indication counts of the eligible diseases and the tier0 node table."""
    dc = pd.read_parquet(RQ3 / "data" / "drugcentral_edges.parquet").query(
        "predicate == 'biolink:treats'"
    )
    cnt = dc.groupby("object").subject.nunique()
    nodes = pd.read_parquet(DATA / "nodes.parquet", columns=["id", "name", "category"]).set_index(
        "id"
    )
    cand = cnt[(cnt >= 2) & cnt.index.str.startswith("MONDO:") & ~cnt.index.isin(excl)]
    cand = cand[nodes.reindex(cand.index).category.eq("biolink:Disease").to_numpy()]
    return cand, nodes


def draw(cand: pd.Series, nodes: pd.DataFrame, seed: int, prefix: str) -> pd.DataFrame:
    """Draw N_QUERIES diseases without replacement and return the query table."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(cand.index.to_numpy(), size=N_QUERIES, replace=False)
    return pd.DataFrame(
        {
            "case": [f"{prefix}{i:03d}" for i in range(N_QUERIES)],
            "disease": pick,
            "name": nodes.reindex(pick).name.to_numpy(),
            "n_drugcentral": cand[pick].to_numpy(),
        }
    )


def report(cand: pd.Series, q: pd.DataFrame) -> None:
    """Print the pool size and a summary of the drawn set."""
    print(
        f"candidates {len(cand)}; drew {len(q)}; indications per disease median "
        f"{q.n_drugcentral.median()} (min {q.n_drugcentral.min()}, max {q.n_drugcentral.max()})"
    )
    print(q.sample(12, random_state=1).to_string(index=False))


def main() -> None:
    """Draw the wide set and write it to OUT."""
    cand, nodes = candidate_pool(test_suite_diseases())
    q = draw(cand, nodes, SEED, PREFIX)
    q.to_csv(OUT, index=False)
    report(cand, q)


if __name__ == "__main__":
    main()

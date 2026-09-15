"""Draw the stratified query-audit sample.

Reads the wide answer scores and DrugCentral edges from rq3/data, the v2 scores and query-answer
features from v2/, and the hand-list removals from rq3/results_wide_v1lt0_iclt50. Writes
data/audit_manifest.csv with judge-visible fields (name, category, description) and hidden
covariates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import LLM_PANEL, RQ3, SEED, V2  # noqa: E402

CAPS = {
    "S1_v2only": 350,
    "S2_handonly": 10**6,
    "S3_both": 60,
    "S4_near": 80,
    "S5_ret_noIC": 60,
    "S6_ret_IC": 60,
}
COLUMNS = [
    "item",
    "id",
    "name",
    "category",
    "description",
    "stratum",
    "stratum_size",
    "sample_size",
    "sampling_prob",
    "v2",
    "ic",
    "hand",
    "v2_removed",
    "dc_drug",
    "n_queries",
    "best_rank",
]


def stratum(r: pd.Series) -> str:
    """Assign one answer concept to its audit stratum."""
    if r.v2_removed and not r.hand:
        return "S1_v2only"
    if r.hand and not r.v2_removed:
        return "S2_handonly"
    if r.v2_removed and r.hand:
        return "S3_both"
    if 0 <= r.v2 < 1:
        return "S4_near"
    return "S5_ret_noIC" if pd.isna(r.ic) else "S6_ret_IC"


def load_population() -> pd.DataFrame:
    """Answer concepts with v2 score, hand-list flag, DrugCentral flag, query counts and stratum."""
    S = pd.read_parquet(RQ3 / "data" / "answer_scores_wide.parquet").set_index("id")
    v2 = pd.read_parquet(V2 / "wide_answers_ft_ens.parquet")["score_ft_ens"]
    S["v2"] = v2.reindex(S.index).to_numpy()
    b = pd.read_csv(RQ3 / "results_wide_v1lt0_iclt50" / "removed_blocklist.csv")
    S["hand"] = S.index.isin(set(b.id))
    dc = pd.read_parquet(RQ3 / "data" / "drugcentral_edges.parquet")
    dc = dc.query("predicate=='biolink:treats'")
    S["dc_drug"] = S.index.isin(set(dc.subject))
    D = pd.read_parquet(V2 / "query_answer_features.parquet")
    S["n_queries"] = D.groupby("id").case.nunique().reindex(S.index).fillna(0).astype(int)
    S["best_rank"] = D.groupby("id")["rank"].min().reindex(S.index)
    S["v2_removed"] = S.v2 < 0
    S["stratum"] = S.apply(stratum, axis=1)
    return S


def draw(S: pd.DataFrame) -> pd.DataFrame:
    """Sample each stratum up to its cap without replacement, then shuffle and number the items."""
    rng = np.random.default_rng(SEED)
    parts = []
    for s, g in S.groupby("stratum"):
        n = min(CAPS[s], len(g))
        idx = rng.choice(g.index.to_numpy(), size=n, replace=False)
        parts.append(
            g.loc[idx].assign(stratum_size=len(g), sample_size=n, sampling_prob=n / len(g))
        )
    M = pd.concat(parts).reset_index().rename(columns={"index": "id"})
    M["description"] = M.description.where(M.description.notna(), None)
    M = M.sample(frac=1, random_state=SEED).reset_index(drop=True)
    M["item"] = np.arange(1, len(M) + 1)
    return M


def has_desc(s: pd.Series) -> float:
    """Fraction of items with a description, rounded to two decimals."""
    return s.notna().mean().round(2)


def main() -> None:
    """Draw the sample, write the manifest and print the per-stratum summary."""
    M = draw(load_population())
    M[COLUMNS].to_csv(LLM_PANEL / "data" / "audit_manifest.csv", index=False)
    summary = M.groupby("stratum").agg(
        sampled=("id", "size"),
        of=("stratum_size", "first"),
        has_desc=("description", has_desc),
        dc=("dc_drug", "mean"),
    )
    print(summary.round(2))
    print("total", len(M))


if __name__ == "__main__":
    main()

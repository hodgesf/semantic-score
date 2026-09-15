"""Wide-run analysis of the four arms at fixed cutoffs.

v1 < V1_CUT and IC < IC_CUT count as generic (answers without IC pass the IC arm); the arms none,
blocklist, ic and v1 are applied to the whole result list. Usage: python arms_wide.py V1_CUT
IC_CUT [alt_scores.parquet column], where the optional pair replaces v1 by another score column
and tags the output directory; RQ3_SET selects the query set ("wide", default, or "heldout").
Outputs in results_<set>_v1lt<V1>_iclt<IC>[_<column>]/: summary.md, per_query.csv and
removed_<arm>.csv.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arax_client import answers  # noqa: E402
from blocklist import is_blocked  # noqa: E402

from paths import RQ3  # noqa: E402

SET = os.environ.get("RQ3_SET", "wide")
RESP = RQ3 / f"responses_{SET}"
ARMS = ["none", "blocklist", "ic", "v1"]
REMOVAL_ARMS = ["blocklist", "ic", "v1"]
TABLE_HEADER = (
    "| arm | answers removed (total) | share of all answers | queries touched | "
    "this-disease indications removed | any DrugCentral drug removed | "
    "indications in top-10 (sum) |"
)


def load_scores(argv: list[str]) -> pd.DataFrame:
    """Answer scores of the set, with v1 optionally replaced by argv[4] of parquet argv[3]."""
    S = pd.read_parquet(RQ3 / "data" / f"answer_scores_{SET}.parquet").set_index("id")
    if len(argv) > 3:
        alt = pd.read_parquet(argv[3])
        if "id" in alt.columns:
            S["v1"] = alt.set_index("id")[argv[4]].reindex(S.index).to_numpy()
        else:
            S["v1"] = alt[argv[4]].reindex(S.index).to_numpy()
    return S


def responses() -> list[tuple[str, dict[str, Any]]]:
    """(case, response) for every unfiltered response of the set, in file order."""
    out = []
    for p in sorted(glob.glob(str(RESP / "*_nofilter.json"))):
        out.append((os.path.basename(p).split("_")[0], json.load(open(p))))
    return out


def per_query(
    Q: pd.DataFrame,
    S: pd.DataFrame,
    dc_by_dz: dict[str, set[str]],
    dc_any: set[str],
    v1_cut: float,
    ic_cut: float,
) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    """Per-query counts for every arm and the removed answers per arm."""
    rows = []
    removed: dict[str, list[dict[str, Any]]] = {a: [] for a in REMOVAL_ARMS}
    for case, d in responses():
        dz = Q.loc[case, "disease"]
        kg = d["message"]["knowledge_graph"]["nodes"]
        full = [x for x, _ in answers(d)[0]]
        if not full:
            rows.append(
                {"case": case, "disease": dz, "disease_name": Q.loc[case, "name"], "n_results": 0}
            )
            continue
        ic = S.ic.reindex(full)
        v1 = S.v1.reindex(full)
        arms = {
            "none": full,
            "blocklist": [x for x in full if not is_blocked(x, kg[x])],
            "ic": [x for x, v in zip(full, ic) if np.isnan(v) or v >= ic_cut],
            "v1": [x for x, v in zip(full, v1) if v >= v1_cut],
        }
        ind = dc_by_dz.get(dz, set())
        ind_present = ind & set(full)
        r = {
            "case": case,
            "disease": dz,
            "disease_name": Q.loc[case, "name"],
            "n_results": len(full),
            "n_no_ic": int(ic.isna().sum()),
            "dc_indications_present": len(ind_present),
        }
        for a, lst in arms.items():
            s = set(lst)
            rem = [x for x in full if x not in s]
            r[f"{a}_kept"] = len(lst)
            r[f"{a}_removed"] = len(rem)
            r[f"{a}_indications_removed"] = len(ind_present - s)
            r[f"{a}_anydrug_removed"] = len([x for x in rem if x in dc_any])
            r[f"{a}_indications_top10"] = len(ind & set(lst[:10]))
            if a != "none":
                for x in rem:
                    removed[a].append(
                        {
                            "case": case,
                            "disease": Q.loc[case, "name"],
                            "rank": full.index(x) + 1,
                            "id": x,
                            "name": kg[x].get("name"),
                            "category": S.category.get(x),
                            "ic": S.ic.get(x),
                            "v1": round(S.v1.get(x), 2) if x in S.index else None,
                            "is_indication_for_this_disease": x in ind,
                            "is_drugcentral_drug": x in dc_any,
                        }
                    )
        rows.append(r)
    return pd.DataFrame(rows), removed


def top_names(removed: dict[str, list[dict[str, Any]]], a: str, n: int = 40) -> pd.DataFrame:
    """Most frequently removed distinct answers of arm ``a``, by number of queries."""
    df = pd.DataFrame(removed[a])
    g = (
        df.groupby(["id", "name"])
        .agg(
            n_queries=("case", "nunique"),
            median_rank=("rank", "median"),
            ic=("ic", "first"),
            v1=("v1", "first"),
            drug=("is_drugcentral_drug", "first"),
        )
        .reset_index()
    )
    return g.sort_values("n_queries", ascending=False).head(n)


def drug_collateral(df: pd.DataFrame) -> str:
    """Most frequently removed DrugCentral-indicated drugs of one arm, as a table."""
    df = df[df.is_drugcentral_drug]
    return (
        df.groupby(["id", "name"])
        .agg(
            n_queries=("case", "nunique"),
            median_rank=("rank", "median"),
            ic=("ic", "first"),
            v1=("v1", "first"),
        )
        .reset_index()
        .sort_values("n_queries", ascending=False)
        .head(40)
        .to_string(index=False)
    )


def survivors(S: pd.DataFrame, v1_cut: float, ic_cut: float) -> pd.DataFrame:
    """The 40 lowest-v1 answers kept by the blocklist, ic and v1 arms alike."""
    surv = []
    for _case, d in responses():
        kg = d["message"]["knowledge_graph"]["nodes"]
        for x in {x for x, _ in answers(d)[0]}:
            if (
                x in S.index
                and S.v1[x] >= v1_cut
                and (np.isnan(S.ic[x]) or S.ic[x] >= ic_cut)
                and not is_blocked(x, kg[x])
            ):
                surv.append((x, kg[x].get("name"), S.ic[x], round(S.v1[x], 2)))
    sv = pd.DataFrame(surv, columns=["id", "name", "ic", "v1"]).drop_duplicates("id")
    return sv.nsmallest(40, "v1")


def summary(
    PQ: pd.DataFrame,
    removed: dict[str, list[dict[str, Any]]],
    S: pd.DataFrame,
    v1_cut: float,
    ic_cut: float,
) -> str:
    """The markdown summary of the run."""
    sets = {a: {(d["case"], d["id"]) for d in lst} for a, lst in removed.items()}
    n_answers = PQ.n_results.sum()
    L = [
        f"# Wide run — fixed cutoffs v1 < {v1_cut:g}, IC < {ic_cut:g} (no-IC passes through)\n",
        f"{len(PQ)} queries, {int(n_answers)} answers total (mean {PQ.n_results.mean():.0f}/query, "
        f"{100 * PQ.n_no_ic.sum() / max(1, n_answers):.0f}% of answers have no IC)\n",
        TABLE_HEADER,
        "|---|---|---|---|---|---|---|",
    ]
    for a in ARMS:
        L.append(
            f"| {a} | {int(PQ[f'{a}_removed'].sum())} | "
            f"{100 * PQ[f'{a}_removed'].sum() / n_answers:.1f}% | "
            f"{int((PQ[f'{a}_removed'] > 0).sum())}/{len(PQ)} | "
            f"{int(PQ[f'{a}_indications_removed'].sum())} / "
            f"{int(PQ.dc_indications_present.sum())} | {int(PQ[f'{a}_anydrug_removed'].sum())} | "
            f"{int(PQ[f'{a}_indications_top10'].sum())} |"
        )
    L.append("\nRemoval overlap (case, answer) pairs:")
    L.append(
        f"  blocklist ∩ v1: {len(sets['blocklist'] & sets['v1'])}   "
        f"blocklist only: {len(sets['blocklist'] - sets['v1'])}   "
        f"v1 only: {len(sets['v1'] - sets['blocklist'])}"
    )
    L.append(
        f"  ic ∩ v1: {len(sets['ic'] & sets['v1'])}   ic only: {len(sets['ic'] - sets['v1'])}   "
        f"v1 only (vs ic): {len(sets['v1'] - sets['ic'])}"
    )
    L.append(f"  blocklist ∩ ic: {len(sets['blocklist'] & sets['ic'])}")
    for a in REMOVAL_ARMS:
        L.append(f"\n## Most frequently removed by {a} (distinct answers, by number of queries)\n")
        L.append(top_names(removed, a).to_string(index=False))
    L.append("\n## DrugCentral-indicated drugs removed by v1 (collateral), most frequent\n")
    L.append(drug_collateral(pd.DataFrame(removed["v1"])))
    L.append("\n## DrugCentral-indicated drugs removed by blocklist (collateral)\n")
    L.append(drug_collateral(pd.DataFrame(removed["blocklist"])))
    L.append(
        "\n## What survives every arm but looks generic: lowest-v1 answers kept by blocklist "
        "AND ic AND v1\n"
    )
    L.append(survivors(S, v1_cut, ic_cut).to_string(index=False))
    L.append(
        "\n## Blocklisted answers that v1 keeps (v1 >= 2.5) — where v1 disagrees with the "
        "hand list\n"
    )
    bk = pd.DataFrame(removed["blocklist"])
    bk = bk[bk.v1.notna() & (bk.v1 >= v1_cut)]
    L.append(
        bk.groupby(["id", "name"])
        .agg(
            n_queries=("case", "nunique"),
            ic=("ic", "first"),
            v1=("v1", "first"),
            drug=("is_drugcentral_drug", "first"),
        )
        .reset_index()
        .sort_values("n_queries", ascending=False)
        .head(40)
        .to_string(index=False)
    )
    return "\n".join(L)


def main() -> None:
    """Parse the cutoffs, evaluate the arms and write the per-query, removed and summary files."""
    v1_cut, ic_cut = float(sys.argv[1]), float(sys.argv[2])
    tag = "_" + sys.argv[4] if len(sys.argv) > 3 else ""
    out = RQ3 / f"results_{SET}_v1lt{v1_cut:g}_iclt{ic_cut:g}{tag}"
    os.makedirs(out, exist_ok=True)
    S = load_scores(sys.argv)
    Q = pd.read_csv(RQ3 / "data" / f"query_set_{SET}.csv").set_index("case")
    dc = pd.read_parquet(RQ3 / "data" / "drugcentral_edges.parquet").query(
        "predicate == 'biolink:treats'"
    )
    dc_by_dz = dc.groupby("object").subject.apply(set).to_dict()
    dc_any = set(dc.subject)
    PQ, removed = per_query(Q, S, dc_by_dz, dc_any, v1_cut, ic_cut)
    PQ.to_csv(out / "per_query.csv", index=False)
    for a, lst in removed.items():
        pd.DataFrame(lst).sort_values(["case", "rank"]).to_csv(
            out / f"removed_{a}.csv", index=False
        )
    txt = summary(PQ, removed, S, v1_cut, ic_cut)
    print(txt)
    open(out / "summary.md", "w").write(txt + "\n")


if __name__ == "__main__":
    main()

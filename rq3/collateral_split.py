"""RQ3 held-out replication and DrugCentral train/dev split of the collateral removals.

For each query set (wide = original 200, heldout = fresh 200) and each arm (hand list, IC < 50,
v2 ensemble < 0, v1 < 0) the removed answers that are DrugCentral-indicated drugs are counted and
split by whether the drug was in the v2 training half of DrugCentral or in the held-out dev half
(used only through the <= 1% epoch-selection constraint on the original 200), with query-level
bootstrap 95% CIs. Usage: python collateral_split.py [wide heldout ...]; output
results_heldout/HELDOUT_RESULTS.md plus per_query_<set>.csv and drug_answers_<set>.csv.
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

from paths import RQ3, SEED, V2  # noqa: E402

IC_CUT = 50.0
CUT = 0.0
CAP = 500
OUT = RQ3 / "results_heldout"
ARMS = ["blocklist", "ic", "v2", "v1"]
HALVES = ["train", "dev"]
LABELS = {
    "wide": "original 200 (development set)",
    "heldout": "held-out 200 (replication set)",
    "wide_rerun": "original 200 re-run 2026-08-28 with Retriever responding",
}
rng = np.random.default_rng(SEED)


def has_retriever(d: dict[str, Any]) -> bool:
    """True if any knowledge-graph edge cites infores:retriever."""
    return any(
        any(src.get("resource_id") == "infores:retriever" for src in e.get("sources", []))
        for e in d["message"]["knowledge_graph"]["edges"].values()
    )


def empty_row(case: str, retriever: bool) -> dict[str, Any]:
    """Per-query row for a query without results."""
    return {
        "case": case,
        "n": 0,
        "n_no_ic": 0,
        "ind_present": 0,
        "top10_none": 0,
        "retriever": retriever,
        "blk_n": 0,
        "blk_cov_v2": 0,
        "blk_cov_v1": 0,
        **{f"{a}_{k}": 0 for a in ARMS for k in ("removed", "ind_removed", "top10")},
    }


def load(
    set_name: str, dc_by_dz: dict[str, set[str]], dc_any: set[str], split: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-query counts, (query, DrugCentral drug answer) pairs and the scores of one set."""
    S = pd.read_parquet(RQ3 / "data" / f"answer_scores_{set_name}.parquet").set_index("id")
    E = pd.read_parquet(V2 / f"{set_name}_answers_ft_ens.parquet")
    if "id" in E.columns:
        S["v2"] = E.set_index("id").score_ft_ens.reindex(S.index).to_numpy()
    else:
        S["v2"] = E.score_ft_ens.reindex(S.index).to_numpy()
    assert S.v2.notna().all(), "every answer needs an ensemble score"
    Q = pd.read_csv(RQ3 / "data" / f"query_set_{set_name}.csv").set_index("case")
    per_q, pairs = [], []
    for p in sorted(glob.glob(str(RQ3 / f"responses_{set_name}" / "*_nofilter.json"))):
        case = os.path.basename(p).split("_")[0]
        dz = Q.loc[case, "disease"]
        d = json.load(open(p))
        kg = d["message"]["knowledge_graph"]["nodes"]
        full = [x for x, _ in answers(d)[0]]
        has_ret = has_retriever(d)
        if not full:
            per_q.append(empty_row(case, has_ret))
            continue
        ind = dc_by_dz.get(dz, set())
        keep = {
            "blocklist": [x for x in full if not is_blocked(x, kg[x])],
            "ic": [x for x, v in zip(full, S.ic.reindex(full)) if np.isnan(v) or v >= IC_CUT],
            "v2": [x for x, v in zip(full, S.v2.reindex(full)) if v >= CUT],
            "v1": [x for x, v in zip(full, S.v1.reindex(full)) if v >= CUT],
        }
        r = {
            "case": case,
            "retriever": has_ret,
            "n": len(full),
            "n_no_ic": int(S.ic.reindex(full).isna().sum()),
            "ind_present": len(ind & set(full)),
            "top10_none": len(ind & set(full[:CAP][:10])),
        }
        for a in ARMS:
            ks = set(keep[a])
            r[f"{a}_removed"] = len(full) - len(ks)
            r[f"{a}_ind_removed"] = len((ind & set(full)) - ks)
            r[f"{a}_top10"] = len(ind & set(keep[a][:10]))
        blk = set(full) - set(keep["blocklist"])
        r["blk_n"] = len(blk)
        r["blk_cov_v2"] = len(blk - set(keep["v2"]))
        r["blk_cov_v1"] = len(blk - set(keep["v1"]))
        for x in full:
            if x in dc_any:
                pairs.append(
                    {
                        "case": case,
                        "id": x,
                        "name": kg[x].get("name"),
                        "half": split[x],
                        "own_ind": x in ind,
                        "rank": full.index(x) + 1,
                        **{f"{a}_removed": x not in set(keep[a]) for a in ARMS},
                        "v2": S.v2[x],
                        "v1": S.v1[x],
                        "ic": S.ic[x],
                    }
                )
        per_q.append(r)
    return pd.DataFrame(per_q), pd.DataFrame(pairs), S


def boot_ci(per_case_values: pd.Series, B: int = 2000) -> tuple[int, int]:
    """Query-level bootstrap 95% interval of the sum of per-query counts."""
    v = np.asarray(per_case_values)
    idx = rng.integers(0, len(v), size=(B, len(v)))
    s = v[idx].sum(1)
    return int(np.percentile(s, 2.5)), int(np.percentile(s, 97.5))


def agg(d: pd.DataFrame, n: int) -> str:
    """Most frequent distinct answers of a removed_<arm>.csv table, as a string."""
    return (
        d.groupby(["id", "name"])
        .agg(
            n_queries=("case", "nunique"),
            v2=("v1", "first"),
            ic=("ic", "first"),
            drugcentral=("is_drugcentral_drug", "first"),
        )
        .reset_index()
        .sort_values("n_queries", ascending=False)
        .head(n)
        .to_string(index=False)
    )


def arm_table(PQ: pd.DataFrame, P: pd.DataFrame) -> list[str]:
    """The per-arm removal table with bootstrap intervals."""
    L = [
        "| arm | answers removed | share | own indications removed [CI] | "
        "DrugCentral drug answers removed [CI] | hand-list coverage | indications in top-10 |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in ARMS:
        if a == "blocklist":
            cov = ""
        elif a in ("v1", "v2"):
            cov = (
                f"{100 * PQ[f'blk_cov_{a}'].sum() / PQ.blk_n.sum():.0f}% "
                f"({int(PQ[f'blk_cov_{a}'].sum())}/{int(PQ.blk_n.sum())})"
            )
        else:
            cov = "n/a"
        drug_rm = P.groupby("case")[f"{a}_removed"].sum().reindex(PQ.case).fillna(0)
        L.append(
            f"| {a} | {int(PQ[f'{a}_removed'].sum())} | "
            f"{100 * PQ[f'{a}_removed'].sum() / PQ.n.sum():.1f}% | "
            f"{int(PQ[f'{a}_ind_removed'].sum())} {boot_ci(PQ[f'{a}_ind_removed'])} | "
            f"{int(drug_rm.sum())} {boot_ci(drug_rm)} | {cov} | "
            f"{int(PQ[f'{a}_top10'].sum())} (unfiltered {int(PQ.top10_none.sum())}) |"
        )
    return L


def half_table(P: pd.DataFrame, label: str) -> list[str]:
    """Collateral split by DrugCentral half."""
    L = [
        f"\n### Collateral by DrugCentral half ({label})\n",
        "(query, answer) pairs where the answer is a DrugCentral-indicated drug, split by the v2 "
        "training half vs the untouched dev half.\n",
        "| half | drug answers | distinct drugs | own-indication answers | hand list removed | "
        "IC removed | v2 removed (own ind.) | v1 removed (own ind.) | "
        "drugs scoring v2 < 0 (distinct) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for h in HALVES:
        Ph = P[P.half == h]
        own = Ph[Ph.own_ind]
        dist = Ph.drop_duplicates("id")
        L.append(
            f"| {h} | {len(Ph)} | {len(dist)} | {len(own)} | {int(Ph.blocklist_removed.sum())} | "
            f"{int(Ph.ic_removed.sum())} | "
            f"{int(Ph.v2_removed.sum())} ({100 * Ph.v2_removed.mean():.2f}%) "
            f"({int(own.v2_removed.sum())}) | "
            f"{int(Ph.v1_removed.sum())} ({100 * Ph.v1_removed.mean():.1f}%) "
            f"({int(own.v1_removed.sum())}) | "
            f"{int((dist.v2 < 0).sum())}/{len(dist)} ({100 * (dist.v2 < 0).mean():.2f}%) |"
        )
    L.append("\nDrugCentral drug answers removed by v2 (all, by half):\n")
    rm = (
        P[P.v2_removed]
        .groupby(["half", "id", "name"])
        .agg(
            n_queries=("case", "nunique"),
            own_ind=("own_ind", "sum"),
            v2=("v2", "first"),
            ic=("ic", "first"),
        )
        .reset_index()
        .sort_values(["half", "n_queries"], ascending=[True, False])
    )
    L.append(rm.to_string(index=False) if len(rm) else "(none)")
    return L


def hand_list_comparison(set_name: str, label: str) -> list[str]:
    """v2 removals beyond the hand list and hand-list removals v2 keeps (wide-run tables)."""
    res = RQ3 / f"results_{set_name}_v1lt0_iclt50_score_ft_ens"
    rv = pd.read_csv(res / "removed_v1.csv")
    rb = pd.read_csv(res / "removed_blocklist.csv")
    kv, kb = set(zip(rv.case, rv.id)), set(zip(rb.case, rb.id))
    only = rv[[k not in kb for k in zip(rv.case, rv.id)]]
    keep = rb[[k not in kv for k in zip(rb.case, rb.id)]]
    return [
        f"\nv2 removals beyond the hand list ({label}): {len(only)} (query, answer) pairs, "
        f"{only.id.nunique()} distinct answers, {int(only.is_drugcentral_drug.sum())} DrugCentral "
        "drug answers. Most frequent:\n",
        agg(only, 30),
        f"\nHand-list removals that v2 keeps ({label}): {len(keep)} pairs, "
        f"{keep.id.nunique()} distinct answers:\n",
        agg(keep, 30),
    ]


def set_section(set_name: str, PQ: pd.DataFrame, P: pd.DataFrame) -> list[str]:
    """The report section of one query set; also writes its per-query and drug-answer tables."""
    label = LABELS[set_name]
    L = [
        f"\n## {label}: {len(PQ)} queries ({int((PQ.n > 0).sum())} with results), "
        f"{int(PQ.n.sum())} answers (median {int(PQ.n.median())}/query; "
        f"{100 * PQ.n_no_ic.sum() / PQ.n.sum():.0f}% without IC), "
        f"{int(PQ.ind_present.sum())} own-indication answers present; "
        f"queries with retriever edges: {int(PQ.retriever.sum())}/{len(PQ)}\n"
    ]
    L += arm_table(PQ, P)
    L += half_table(P, label)
    L += hand_list_comparison(set_name, label)
    PQ.to_csv(OUT / f"per_query_{set_name}.csv", index=False)
    P.to_csv(OUT / f"drug_answers_{set_name}.csv", index=False)
    dist = P.drop_duplicates("id")
    L.append(
        f"\nDistinct DrugCentral drugs among answers, v2 score: train median "
        f"{dist[dist.half == 'train'].v2.median():.2f} (n {int((dist.half == 'train').sum())}), "
        f"dev median {dist[dist.half == 'dev'].v2.median():.2f} "
        f"(n {int((dist.half == 'dev').sum())})"
    )
    return L


def main() -> None:
    """Evaluate the requested sets in order and write HELDOUT_RESULTS.md."""
    sets = sys.argv[1:] or ["wide", "heldout"]
    os.makedirs(OUT, exist_ok=True)
    dc = pd.read_parquet(RQ3 / "data" / "drugcentral_edges.parquet").query(
        "predicate == 'biolink:treats'"
    )
    dc_by_dz = dc.groupby("object").subject.apply(set).to_dict()
    dc_any = set(dc.subject)
    split = pd.read_parquet(V2 / "drugcentral_split.parquet").set_index("id").split
    assert dc_any <= set(split.index), "split table must cover every DrugCentral drug"
    L = [
        "# RQ3 held-out replication and DrugCentral train/dev split of collateral\n",
        "Arms: hand list (production blocklist), IC < 50 (no-IC passes), v2 ensemble < 0, v1 < 0. "
        "Cutoffs and the ensemble were frozen before the held-out set was drawn. "
        "Query-level bootstrap 95% CIs (2,000 resamples of queries).\n",
    ]
    for set_name in sets:
        PQ, P, _S = load(set_name, dc_by_dz, dc_any, split)
        L += set_section(set_name, PQ, P)
    txt = "\n".join(L)
    print(txt)
    open(OUT / "HELDOUT_RESULTS.md", "w").write(txt + "\n")


if __name__ == "__main__":
    main()

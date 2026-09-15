"""Build per (query, answer) features for the query-conditional experiments.

Inputs: the unfiltered responses in rq3/responses_wide/*_nofilter.json, rq3/data/query_set_wide.csv,
rq3/data/answer_scores_wide.parquet, rq3/data/drugcentral_edges.parquet, data/nodes.parquet and
data/generality_axis.parquet. Output: v2/query_answer_features.parquet with rank, v1, IC,
promiscuity, direct-edge evidence (predicates and primary sources, including support-graph
edges), xDTD probability, and per-query disease axis/IC and list-level v1 statistics.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, RQ3, V2  # noqa: E402


def bound_edge_ids(result: dict, kg_edges: dict, aux: dict) -> set[str]:
    """Collect every edge id bound to a result, expanding support graphs transitively."""
    eids: set[str] = set()
    for an in result["analyses"]:
        for eb in an["edge_bindings"].values():
            eids |= {e["id"] for e in eb}
        for sg in an.get("support_graphs") or []:
            eids |= set(aux.get(sg, {}).get("edges", []))
    frontier = list(eids)
    while frontier:
        e = kg_edges.get(frontier.pop(), {})
        for at in e.get("attributes") or []:
            if at.get("attribute_type_id") == "biolink:support_graphs":
                for g in at["value"]:
                    new = set(aux.get(g, {}).get("edges", [])) - eids
                    eids |= new
                    frontier += list(new)
    return eids


def direct_evidence(
    eids: set[str], kg_edges: dict, x: str, dz: str
) -> tuple[set[str], set[str], float, bool]:
    """Predicates, primary sources and max probability of edges directly joining x and dz."""
    preds: set[str] = set()
    srcs: set[str] = set()
    prob = np.nan
    direct = False
    for e_id in eids:
        e = kg_edges.get(e_id, {})
        if {e.get("subject"), e.get("object")} != {x, dz}:
            continue
        direct = True
        preds.add(e.get("predicate"))
        for s in e.get("sources") or []:
            if s.get("resource_role") == "primary_knowledge_source":
                srcs.add(s.get("resource_id"))
        for at in e.get("attributes") or []:
            name = str(at.get("original_attribute_name", ""))
            type_id = str(at.get("attribute_type_id", ""))
            if "probability" in name or "probability" in type_id:
                try:
                    value = float(at["value"])
                    prob = max(prob, value) if not np.isnan(prob) else value
                except Exception:
                    pass
    return preds, srcs, prob, direct


def response_rows(
    path: str, queries: pd.DataFrame, indications: dict[str, set[str]], dc_any: set[str]
) -> list[dict]:
    """Parse one response file into one feature row per result."""
    case = os.path.basename(path).split("_")[0]
    dz = queries.loc[case, "disease"]
    with open(path) as fh:
        m = json.load(fh)["message"]
    kg_edges = m["knowledge_graph"]["edges"]
    aux = m.get("auxiliary_graphs") or {}
    rows = []
    for rank, r in enumerate(m["results"], 1):
        x = r["node_bindings"]["SN"][0]["id"]
        eids = bound_edge_ids(r, kg_edges, aux)
        preds, srcs, prob, direct = direct_evidence(eids, kg_edges, x, dz)
        rows.append(
            {
                "case": case,
                "disease": dz,
                "rank": rank,
                "id": x,
                "n_results": len(m["results"]),
                "direct_edge": direct,
                "direct_preds": "|".join(sorted(p for p in preds if p)),
                "direct_sources": "|".join(sorted(s for s in srcs if s)),
                "xdtd_prob": prob,
                "n_edges_bound": len(eids),
                "score": r["analyses"][0].get("score"),
                "is_indication": x in indications.get(dz, set()),
                "is_dcdrug": x in dc_any,
            }
        )
    return rows


def share_below0(s: pd.Series) -> float:
    """Share of values below 0."""
    return (s < 0).mean()


def add_derived(
    d: pd.DataFrame, scores: pd.DataFrame, nodes_ic: pd.Series, axis: pd.Series
) -> pd.DataFrame:
    """Add v1, IC, name, promiscuity, disease genericness and list-level v1 statistics."""
    d["v1"] = scores.v1.reindex(d.id).to_numpy()
    d["ic"] = scores.ic.reindex(d.id).to_numpy()
    d["name"] = scores.name.reindex(d.id).to_numpy()
    d["promiscuity"] = d.groupby("id").case.transform("nunique")
    d["disease_axis_spec"] = -axis.reindex(d.disease).to_numpy()
    d["disease_ic"] = nodes_ic.reindex(d.disease).to_numpy()
    g = d.groupby("case").v1
    d["list_v1_median"] = g.transform("median")
    d["list_v1_sd"] = g.transform("std")
    d["list_share_below0"] = g.transform(share_below0)
    return d


def print_summary(d: pd.DataFrame) -> None:
    """Print coverage and evidence summaries of the feature table."""
    print(len(d), "rows;", d.id.nunique(), "answers")
    print(
        "direct edge present:",
        d.direct_edge.mean().round(3),
        "| xDTD prob present:",
        d.xdtd_prob.notna().mean().round(3),
    )
    direct = d[d.direct_edge]
    print(
        "direct predicates:",
        direct.direct_preds.str.split("|").explode().value_counts().head(8).to_dict(),
    )
    print(
        "primary sources on direct edges:",
        direct.direct_sources.str.split("|").explode().value_counts().head(12).to_dict(),
    )
    print(
        "\npromiscuity of answers with v1<0 vs >=0 (median n queries):",
        d[d.v1 < 0].promiscuity.median(),
        d[d.v1 >= 0].promiscuity.median(),
    )
    per_query = d.drop_duplicates("case")
    rho = per_query[["disease_axis_spec", "list_share_below0"]].corr(method="spearman").iloc[0, 1]
    print("disease genericness (axis_spec) vs share of list below v1 0 — Spearman:", rho.round(3))
    print("disease IC coverage:", per_query.disease_ic.notna().mean().round(2))


def main() -> None:
    """Parse every response, build the feature table, write it and print summaries."""
    scores = pd.read_parquet(RQ3 / "data" / "answer_scores_wide.parquet").set_index("id")
    queries = pd.read_csv(RQ3 / "data" / "query_set_wide.csv").set_index("case")
    nodes_ic = (
        pd.read_parquet(DATA / "nodes.parquet", columns=["id", "information_content"])
        .set_index("id")
        .information_content
    )
    axis = (
        pd.read_parquet(DATA / "generality_axis.parquet", columns=["id", "gen_axis"])
        .set_index("id")
        .gen_axis
    )
    dc = pd.read_parquet(RQ3 / "data" / "drugcentral_edges.parquet").query(
        "predicate=='biolink:treats'"
    )
    indications = dc.groupby("object").subject.apply(set).to_dict()
    dc_any = set(dc.subject)
    rows = []
    for p in sorted(glob.glob(str(RQ3 / "responses_wide" / "*_nofilter.json"))):
        rows += response_rows(p, queries, indications, dc_any)
    d = add_derived(pd.DataFrame(rows), scores, nodes_ic, axis)
    d.to_parquet(V2 / "query_answer_features.parquet", index=False)
    print_summary(d)


if __name__ == "__main__":
    main()

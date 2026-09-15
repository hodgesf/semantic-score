"""Helpers shared by the RQ3 scripts.

Builds a treats query graph for a disease CURIE, posts it to the local ARAX server in either
the production ("prod") or unfiltered ("nofilter") configuration, saves the TRAPI response and
extracts the ranked answers from it.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any

URL = "http://localhost:5001/api/arax/v1.4/query"
DSL_PROD = [
    "expand()",
    "overlay(action=compute_ngd, virtual_relation_label=N1, "
    "subject_qnode_key=ON, object_qnode_key=SN)",
    "filter_kg(action=remove_general_concept_nodes,perform_action=True)",
    "resultify()",
    "filter_results(action=limit_number_of_results, max_results=500)",
]
# Unfiltered: blocklist off and no result cap, so the post-hoc arms see the full ranked list;
# the production cap (CAP) is applied after each arm's filter in the analysis.
DSL_NOFILTER = [
    "expand()",
    "overlay(action=compute_ngd, virtual_relation_label=N1, "
    "subject_qnode_key=ON, object_qnode_key=SN)",
    "filter_kg(action=remove_general_concept_nodes,perform_action=False)",
    "resultify()",
]
CAP = 500


def query_graph(curie: str) -> dict[str, Any]:
    """Return the TRAPI query graph for an inferred treats query on ``curie``."""
    return {
        "nodes": {
            "ON": {"ids": [curie], "categories": ["biolink:Disease"]},
            "SN": {"categories": ["biolink:ChemicalEntity"]},
        },
        "edges": {
            "e01": {
                "subject": "SN",
                "object": "ON",
                "predicates": ["biolink:treats"],
                "knowledge_type": "inferred",
            }
        },
    }


def body(curie: str, mode: str) -> dict[str, Any]:
    """Return the request body for ``curie`` in mode ``prod`` or ``nofilter``."""
    return {
        "message": {"query_graph": query_graph(curie)},
        "operations": {"actions": DSL_PROD if mode == "prod" else DSL_NOFILTER},
    }


def post(
    curie: str, mode: str, path: str | Path, timeout: int = 3600
) -> tuple[dict[str, Any], float]:
    """Post the query to ARAX, save the response to ``path``, return it with the elapsed seconds."""
    req = urllib.request.Request(
        URL,
        data=json.dumps(body(curie, mode)).encode(),
        headers={"Content-Type": "application/json"},
    )
    t = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=timeout))
    dt = time.time() - t
    with open(path, "w") as fh:
        json.dump(d, fh)
    return d, dt


def answers(d: dict[str, Any]) -> tuple[list[tuple[str, float | None]], dict[str, str | None]]:
    """Return [(answer_id, score)] in result order and the id -> name map of the knowledge graph."""
    m = d["message"]
    names = {k: v.get("name") for k, v in m["knowledge_graph"]["nodes"].items()}
    rows = [
        (r["node_bindings"]["SN"][0]["id"], r["analyses"][0].get("score")) for r in m["results"]
    ]
    return rows, names

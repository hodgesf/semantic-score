"""Score every answer node of the unfiltered responses with IC, the label-free axis and v1.

The recipe is the same as v1/seal_sample_b.py: BioLORD embedding plus generality axis, scored by
the v1 logistic model. Non-tier0 answers get BioLORD from the TRAPI name and the axis from
SapBERT(name) @ w standardised with the global tier0 statistics; their IC is absent. Run with the
GPU environment; RQ3_RESPONSES and RQ3_SCORES select the response directory and output name.
Output: data/answer_scores.parquet.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, NODE_EMBEDDINGS, RQ3, SEED  # noqa: E402

RESP_DIR = os.environ.get("RQ3_RESPONSES", "responses")
OUT_NAME = os.environ.get("RQ3_SCORES", "answer_scores.parquet")
SAPBERT = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
BIOLORD = "FremyCompany/BioLORD-2023"


def collect_answers() -> pd.DataFrame:
    """Distinct answer nodes over all unfiltered responses (a superset of the production ones)."""
    ans = {}
    for p in sorted(glob.glob(str(RQ3 / RESP_DIR / "*_nofilter.json"))):
        m = json.load(open(p))["message"]
        kg = m["knowledge_graph"]["nodes"]
        for r in m["results"]:
            nid = r["node_bindings"]["SN"][0]["id"]
            if nid not in ans:
                ans[nid] = {
                    "id": nid,
                    "trapi_name": kg[nid].get("name"),
                    "trapi_category": (kg[nid].get("categories") or [None])[0],
                }
    A = pd.DataFrame(ans.values())
    print("distinct answer nodes:", len(A))
    return A


def attach_tier0(A: pd.DataFrame) -> None:
    """Add name, description, category and IC from tier0 where the answer is a tier0 node."""
    nodes = pd.read_parquet(
        DATA / "nodes.parquet",
        columns=["id", "name", "description", "category", "information_content"],
    ).set_index("id")
    A["in_tier0"] = A.id.isin(nodes.index)
    t0 = nodes.reindex(A.id)
    A["name"] = np.where(A.in_tier0, t0.name.to_numpy(), A.trapi_name.to_numpy())
    A["description"] = np.where(A.in_tier0, t0.description.to_numpy(), None)
    A["category"] = np.where(A.in_tier0, t0.category.to_numpy(), A.trapi_category.to_numpy())
    A["ic"] = t0.information_content.to_numpy()
    print("in tier0:", int(A.in_tier0.sum()), "| with IC:", int(A.ic.notna().sum()))


def attach_axis(A: pd.DataFrame, axis: pd.Series) -> None:
    """Add the generality axis; compute it from SapBERT(name) for answers outside tier0."""
    A["gen_axis"] = axis.reindex(A.id).to_numpy()
    missing = A[A.gen_axis.isna()]
    if not len(missing):
        return
    w = np.load(DATA / "generality_axis_w.npy").astype(np.float32)
    E = np.load(NODE_EMBEDDINGS, mmap_mode="r")
    raw = np.concatenate(
        [np.asarray(E[s : s + 200_000]) @ w for s in range(0, E.shape[0], 200_000)]
    )
    mu, sd = float(raw.mean()), float(raw.std())
    print(f"axis raw stats mu={mu:.4f} sd={sd:.4f}")
    tok = AutoTokenizer.from_pretrained(SAPBERT)
    sap = AutoModel.from_pretrained(SAPBERT).cuda().eval()
    vecs = []
    with torch.inference_mode():
        for s in range(0, len(missing), 256):
            b = tok(
                [str(n) for n in missing.name.iloc[s : s + 256]],
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to("cuda")
            v = sap(**b).last_hidden_state[:, 0]
            vecs.append(torch.nn.functional.normalize(v, dim=1).float().cpu().numpy())
    g = (np.vstack(vecs) @ w - mu) / sd
    A.loc[missing.index, "gen_axis"] = g
    print("axis computed from SapBERT(name) for", len(missing), "non-tier0 answers")


def biolord_embeddings(A: pd.DataFrame) -> np.ndarray:
    """Mean-pooled, normalised BioLORD embeddings of "name: description" for every answer."""
    tok = AutoTokenizer.from_pretrained(BIOLORD)
    bl = AutoModel.from_pretrained(BIOLORD).cuda().eval().half()
    texts = [
        f"{n}: {d}" if isinstance(d, str) and d else str(n) for n, d in zip(A.name, A.description)
    ]
    vecs = []
    with torch.inference_mode():
        for s in range(0, len(texts), 256):
            b = tok(
                texts[s : s + 256],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to("cuda")
            h = bl(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1)
            vecs.append(
                torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=1)
                .float()
                .cpu()
                .numpy()
            )
    return np.vstack(vecs)


def fit_v1(axis: pd.Series) -> Pipeline:
    """Fit the v1 logistic model on the training set, as in seal_sample_b.py."""
    train = pd.read_parquet(DATA / "training_set_v1.parquet")
    z = np.load(DATA / "biolord_v1.npz", allow_pickle=True)
    row = pd.Series(np.arange(len(z["ids"])), index=z["ids"])
    F = np.hstack([z["emb"][row[train.id].to_numpy()], axis[train.id].to_numpy()[:, None]])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED),
    )
    return model.fit(F, train.label.to_numpy())


def main() -> None:
    """Collect the answers, compute the scores and write the parquet."""
    A = collect_answers()
    attach_tier0(A)
    axis = pd.read_parquet(DATA / "generality_axis.parquet", columns=["id", "gen_axis"]).set_index(
        "id"
    )["gen_axis"]
    attach_axis(A, axis)
    Eb = biolord_embeddings(A)
    model = fit_v1(axis)
    A["v1"] = -model.decision_function(np.hstack([Eb, A.gen_axis.to_numpy()[:, None]]))
    A["axis_spec"] = -A.gen_axis
    A.to_parquet(RQ3 / "data" / OUT_NAME, index=False)
    print(A[["ic", "gen_axis", "v1"]].describe().round(3))
    print("most generic by v1:")
    print(A.nsmallest(15, "v1")[["name", "category", "ic", "v1"]].to_string())


if __name__ == "__main__":
    main()

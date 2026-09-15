"""Score Sample B with the v1 arms (BioLORD+axis, SapBERT+axis) and the label-free axis.

Refits each arm on data/training_set_v1.parquet, embeds the Sample B nodes with BioLORD-2023
and writes results/sample_b_scores_SEALED.csv. The file is written before any panel rating
exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, NODE_EMBEDDINGS, RESULTS, SAMPLE_B  # noqa: E402

KEY_FILE = SAMPLE_B / "data" / "sample_b_key.csv"
OUT_FILE = RESULTS / "sample_b_scores_SEALED.csv"
MODEL = "FremyCompany/BioLORD-2023"
SEED = 2654


def embed_biolord(texts: list[str]) -> np.ndarray:
    """Mean-pooled, unit-normalised BioLORD embeddings of the texts in one GPU batch."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).cuda().eval().half()
    with torch.inference_mode():
        b = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        b = b.to("cuda")
        h = model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1)
        v = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=1)
    return v.float().cpu().numpy()


def fit_score(
    e_train: np.ndarray,
    e_b: np.ndarray,
    axis_train: np.ndarray,
    axis_b: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Fit embedding+axis logistic regression on the training set and score Sample B."""
    f = np.hstack([e_train, axis_train[:, None]])
    fb = np.hstack([e_b, axis_b[:, None]])
    m = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED),
    ).fit(f, y)
    return -m.decision_function(fb)


def main() -> None:
    """Embed Sample B, fit both v1 arms and write the sealed scores."""
    key = pd.read_csv(KEY_FILE)
    train = pd.read_parquet(DATA / "training_set_v1.parquet")
    nodes = pd.read_parquet(DATA / "nodes.parquet", columns=["id", "name", "description"])
    nodes = nodes.set_index("id")
    row_of = pd.Series(np.arange(len(nodes)), index=nodes.index)

    sub = nodes.loc[key.id]
    texts = [
        f"{r.name}: {r.description}" if isinstance(r.description, str) and r.description else r.name
        for r in sub.itertuples()
    ]
    v = embed_biolord(texts)

    bl = np.load(DATA / "biolord_v1.npz", allow_pickle=True)
    bl_row = pd.Series(np.arange(len(bl["ids"])), index=bl["ids"])
    sap = np.load(NODE_EMBEDDINGS, mmap_mode="r")
    axis = pd.read_parquet(DATA / "generality_axis.parquet", columns=["id", "gen_axis"])
    axis = axis.set_index("id")["gen_axis"]
    y = train.label.to_numpy()
    axis_train = axis[train.id].to_numpy()
    axis_b = axis[key.id].to_numpy()

    out = key[["item", "id", "name", "most_specific_category"]].copy()
    out["v1_biolord_axis"] = fit_score(
        bl["emb"][bl_row[train.id].to_numpy()], v, axis_train, axis_b, y
    )
    out["v1_sapbert_axis"] = fit_score(
        np.asarray(sap[row_of[train.id].to_numpy()]),
        np.asarray(sap[row_of[key.id].to_numpy()]),
        axis_train,
        axis_b,
        y,
    )
    out["ic_axis"] = -axis[key.id].to_numpy()
    out.to_csv(OUT_FILE, index=False)
    print("sealed", len(out), "->", OUT_FILE)
    print(out.describe().T.round(2))


if __name__ == "__main__":
    main()

"""Embed the v1 training-set and Sample A nodes with BioLORD-2023 on the GPU.

Each node is encoded as "name: description" (name alone without a description), mean-pooled
and unit-normalised, matching the SapBERT file. Output: data/biolord_v1.npz with ids and
float32 vectors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, PANEL  # noqa: E402

SAMPLE_A_KEY = PANEL / "data" / "sample_a_key.csv"
OUT_FILE = DATA / "biolord_v1.npz"
MODEL = "FremyCompany/BioLORD-2023"
BATCH = 256


def node_texts() -> tuple[pd.Index, list[str]]:
    """Return the training-set plus Sample A ids and one text per id."""
    train = pd.read_parquet(DATA / "training_set_v1.parquet")
    key = pd.read_csv(SAMPLE_A_KEY)
    ids = pd.Index(train["id"]).union(pd.Index(key["id"]))
    nodes = pd.read_parquet(DATA / "nodes.parquet", columns=["id", "name", "description"])
    nodes = nodes.set_index("id").loc[ids]
    texts = [
        f"{r.name}: {r.description}" if isinstance(r.description, str) and r.description else r.name
        for r in nodes.itertuples()
    ]
    return ids, texts


def embed(texts: list[str]) -> np.ndarray:
    """Mean-pool and unit-normalise BioLORD embeddings of the texts, in BATCH-sized steps."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).cuda().eval().half()
    out = np.empty((len(texts), 768), dtype=np.float32)
    with torch.inference_mode():
        for s in range(0, len(texts), BATCH):
            b = tok(
                texts[s : s + BATCH],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to("cuda")
            h = model(**b).last_hidden_state
            mask = b["attention_mask"].unsqueeze(-1)
            v = (h * mask).sum(1) / mask.sum(1)
            v = torch.nn.functional.normalize(v, dim=1)
            out[s : s + BATCH] = v.float().cpu().numpy()
            if (s // BATCH) % 20 == 0:
                print(f"  {s + len(b['input_ids']):,}/{len(texts):,}", flush=True)
    return out


def main() -> None:
    """Embed every node and save ids and vectors."""
    ids, texts = node_texts()
    print(f"{len(texts)} texts")
    out = embed(texts)
    np.savez(OUT_FILE, ids=np.array(ids), emb=out)
    print("saved", out.shape)


if __name__ == "__main__":
    main()

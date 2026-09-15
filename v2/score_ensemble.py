"""Score answer nodes with the frozen v2 ensemble of three fine-tuned BioLORD checkpoints.

Inputs: an answer table with an ``id`` column, data/nodes.parquet, data/generality_axis.parquet
and the checkpoints v2/v2ft_best.pt, v2/v2ft_best_seed2.pt and v2/v2ft_best_seed3.pt (seeds
2654 / 2 / 3). Output: a parquet with one score column per seed and score_ft_ens, the mean of
the three (score = -logit, 0 = boundary). Text and axis features are those of finetune_v2.py.
Usage: python score_ensemble.py <answer_scores.parquet> <out.parquet>.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, V2  # noqa: E402

MODEL_NAME = "FremyCompany/BioLORD-2023"
CKPTS = {"2654": "v2ft_best.pt", "2": "v2ft_best_seed2.pt", "3": "v2ft_best_seed3.pt"}


class M(nn.Module):
    """BioLORD encoder with mean pooling, the generality axis appended, and a linear head."""

    def __init__(self) -> None:
        """Load the pretrained encoder and create the dropout + linear head."""
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL_NAME)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(769, 1))

    def forward(self, ids: torch.Tensor, att: torch.Tensor, ax: torch.Tensor) -> torch.Tensor:
        """Return one logit per input from the normalised pooled embedding and the axis."""
        h = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        m = att.unsqueeze(-1)
        p = (h * m).sum(1) / m.sum(1)
        p = nn.functional.normalize(p, dim=1)
        return self.head(torch.cat([p, ax[:, None]], 1)).squeeze(-1)


def text(nodes: pd.DataFrame, i: str) -> str:
    """Return 'name: description' for node ``i``, or just the name when no description."""
    d = nodes.description.get(i)
    if isinstance(d, str) and d:
        return f"{nodes.name[i]}: {d}"
    return str(nodes.name.get(i, i))


@torch.no_grad()
def predict(
    model: M,
    tok: PreTrainedTokenizerBase,
    nodes: pd.DataFrame,
    axis: pd.Series,
    ids: list[str],
) -> np.ndarray:
    """Score ``ids`` with the model in eval mode; score = -logit."""
    model.eval()
    res = []
    for s in range(0, len(ids), 256):
        b = ids[s : s + 256]
        t = tok(
            [text(nodes, i) for i in b],
            padding=True,
            truncation=True,
            max_length=96,
            return_tensors="pt",
        )
        ax = torch.tensor(axis.reindex(b).fillna(0).to_numpy(), dtype=torch.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(t["input_ids"].cuda(), t["attention_mask"].cuda(), ax.cuda())
        res.append((-logit).float().cpu().numpy())
    return np.concatenate(res)


def main(inp: str, out: str) -> None:
    """Score the answers in ``inp`` with each checkpoint and write the ensemble to ``out``."""
    answers = pd.read_parquet(inp)
    ids = answers.id.tolist()
    nodes = pd.read_parquet(
        DATA / "nodes.parquet", columns=["id", "name", "description"]
    ).set_index("id")
    axis = pd.read_parquet(DATA / "generality_axis.parquet", columns=["id", "gen_axis"]).set_index(
        "id"
    )["gen_axis"]
    missing = [i for i in ids if i not in nodes.index]
    print("answers:", len(ids), "| not in tier0:", len(missing))
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = M().cuda()
    result = pd.DataFrame({"id": ids})
    for seed, ck in CKPTS.items():
        model.load_state_dict(torch.load(V2 / ck))
        col = f"score_ft_seed{seed}"
        result[col] = predict(model, tok, nodes, axis, ids)
        print("seed", seed, "done; below 0:", round(100 * float((result[col] < 0).mean()), 1), "%")
    result["score_ft_ens"] = result[[f"score_ft_seed{s}" for s in CKPTS]].mean(1)
    result.to_parquet(out, index=False)
    print(result.describe().round(3))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

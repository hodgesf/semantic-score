"""Evaluate a fine-tuned checkpoint on Sample A and Sample B with bootstrap CIs vs v1 and IC.

Inputs: v2/v2ft_best{TAG}.pt, data/nodes.parquet, data/generality_axis.parquet and the
Sample A/B panel score tables. Outputs: v2/sample_a_ft{TAG}.csv, v2/sample_b_ft{TAG}.csv and
v2/ft_eval_ab{TAG}.json. Usage: python ft_eval_ab.py [TAG].
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, ROOT, SEED, V2  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else ""
MODEL_NAME = "FremyCompany/BioLORD-2023"
N_BOOT = 10000
RATERS = ["roach", "ramsey", "koslicki"]


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
        p = nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=1)
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
    """Score ``ids`` in a single batch; score = -logit."""
    t = tok(
        [text(nodes, i) for i in ids],
        padding=True,
        truncation=True,
        max_length=96,
        return_tensors="pt",
    )
    ax = torch.tensor(axis.reindex(ids).fillna(0).to_numpy(), dtype=torch.float32)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logit = model(t["input_ids"].cuda(), t["attention_mask"].cuda(), ax.cuda())
    return (-logit).float().cpu().numpy()


def boot(
    rng: np.random.Generator, x: pd.Series, y: pd.Series, n: int = N_BOOT
) -> tuple[float, float]:
    """Bootstrap 2.5/97.5 percentiles of Spearman rho between ``x`` and ``y``."""
    stats = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        stats.append(spearmanr(x.iloc[i], y.iloc[i]).statistic)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return lo, hi


def diff(
    rng: np.random.Generator, a: pd.Series, b: pd.Series, y: pd.Series, n: int = N_BOOT
) -> tuple[float, float, float, float]:
    """Bootstrap the rho difference rho(a, y) - rho(b, y): mean, CI bounds, P(>0)."""
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        vals.append(
            spearmanr(a.iloc[i], y.iloc[i]).statistic - spearmanr(b.iloc[i], y.iloc[i]).statistic
        )
    v = np.array(vals)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return v.mean(), lo, hi, (v > 0).mean()


def evaluate(
    rng: np.random.Generator, lab: str, d: pd.DataFrame, cols: dict[str, str], out: dict
) -> None:
    """Print rho with CIs per score column, the ft-minus-baseline differences and per-rater rho."""
    for k, c in cols.items():
        r = spearmanr(d[c], d.panel).statistic
        lo, hi = boot(rng, d[c], d.panel)
        out[f"{lab} {k}"] = [round(r, 3), round(lo, 3), round(hi, 3)]
        print(f"{lab} {k:3s} rho {r:.3f} [{lo:.3f}, {hi:.3f}]")
    for k, c in cols.items():
        if k != "ft":
            m, lo, hi, p = diff(rng, d.ft, d[c], d.panel)
            out[f"{lab} ft-{k}"] = [round(m, 3), round(lo, 3), round(hi, 3), round(p, 3)]
            print(f"  ft − {k}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}] P(>0)={p:.3f}")
    per_rater = {r: round(spearmanr(d.ft, d[r]).statistic, 3) for r in RATERS if r in d}
    print("  per-rater:", per_rater)


def main() -> None:
    """Score Sample A/B with the checkpoint, write the score tables and the bootstrap summary."""
    nodes = pd.read_parquet(
        DATA / "nodes.parquet", columns=["id", "name", "description"]
    ).set_index("id")
    axis = pd.read_parquet(DATA / "generality_axis.parquet", columns=["id", "gen_axis"]).set_index(
        "id"
    )["gen_axis"]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = M().cuda()
    model.load_state_dict(torch.load(V2 / f"v2ft_best{TAG}.pt"))
    model.eval()

    sample_a = pd.read_csv(ROOT / "v1" / "sample_a_v1_scores.csv")
    sample_b = pd.read_csv(ROOT / "sample_b" / "results" / "sample_b_panel_scores.csv")
    sample_a["ft"] = predict(model, tok, nodes, axis, sample_a.id.tolist())
    sample_b["ft"] = predict(model, tok, nodes, axis, sample_b.id.tolist())
    sample_a[["id", "panel", "information_content", "biolord+axis", "ft"]].to_csv(
        V2 / f"sample_a_ft{TAG}.csv", index=False
    )
    sample_b[["id", "panel", "v1_biolord_axis", "ft"]].to_csv(
        V2 / f"sample_b_ft{TAG}.csv", index=False
    )

    rng = np.random.default_rng(SEED)
    out: dict = {}
    datasets = [
        ("Sample A", sample_a, {"ft": "ft", "v1": "biolord+axis", "ic": "information_content"}),
        ("Sample B", sample_b, {"ft": "ft", "v1": "v1_biolord_axis"}),
    ]
    for lab, d, cols in datasets:
        evaluate(rng, lab, d, cols, out)
    with open(V2 / f"ft_eval_ab{TAG}.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()

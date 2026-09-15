"""Fine-tune BioLORD-2023 end to end with a classification head on the v2c training set.

Inputs: data/nodes.parquet, data/generality_axis.parquet, data/training_set_v1.parquet,
v2/drugcentral_split.parquet, the wide answer scores and the Sample A/B panel scores.
Outputs: v2/v2ft_best{TAG}.pt, v2/wide_answers_ft_scores{TAG}.parquet and
v2/finetune_v2_results{TAG}.json, where TAG is the FT_TAG environment variable and the
seed is FT_SEED. Epoch selection: largest separation at score 0 subject to at most 1% of
held-back drugs below 0.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, ROOT, RQ3, SEED, V2  # noqa: E402

FT_SEED = int(os.environ.get("FT_SEED", SEED))
TAG = os.environ.get("FT_TAG", "")
MODEL_NAME = "FremyCompany/BioLORD-2023"
HOLDOUT_FRAC = 0.1
BATCH_SIZE = 48
N_EPOCHS = 3
EXAMPLE_NAMES = [
    "Prednisone",
    "Methylprednisolone",
    "Cortisone",
    "Hydrocortisone",
    "penicillin",
    "Phylloquinone",
    "Cefazolin",
    "Rituximab",
    "steroid",
    "glucocorticoid",
    "Antibiotics",
    "Pharmaceutical Preparations",
    "Vitamin D Drug Class",
    "Vitamin A",
    "Oxygen",
    "Ethanol",
]


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


def build_training_set(nodes: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build the shuffled v2c training set and return it with the DrugCentral dev ids."""
    t1 = pd.read_parquet(DATA / "training_set_v1.parquet")[["id", "label"]]
    split = pd.read_parquet(V2 / "drugcentral_split.parquet")
    dc_train = split[split.split == "train"].id.tolist()
    dc_dev = split[split.split == "dev"].id.tolist()
    t1 = t1[~t1.id.isin(dc_dev)]
    iupac = t1.id.map(nodes.prefix).isin(["PUBCHEM.COMPOUND", "CHEMBL.COMPOUND"]) & (t1.label == 0)
    add = pd.DataFrame({"id": [i for i in dc_train if i not in set(t1.id)], "label": 0})
    full = pd.concat([t1[~iupac], add]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return full, dc_dev


def batches(
    df: pd.DataFrame,
    bs: int,
    shuffle: bool,
    tok: PreTrainedTokenizerBase,
    nodes: pd.DataFrame,
    axis: pd.Series,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield (input_ids, attention_mask, axis, label) CUDA tensors in batches of ``bs``."""
    idx = np.random.permutation(len(df)) if shuffle else np.arange(len(df))
    for s in range(0, len(df), bs):
        b = df.iloc[idx[s : s + bs]]
        t = tok(
            [text(nodes, i) for i in b.id],
            padding=True,
            truncation=True,
            max_length=96,
            return_tensors="pt",
        )
        ax = torch.tensor(axis.reindex(b.id).fillna(0).to_numpy(), dtype=torch.float32)
        y = torch.tensor(b.label.to_numpy(), dtype=torch.float32)
        yield t["input_ids"].cuda(), t["attention_mask"].cuda(), ax.cuda(), y.cuda()


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
    out = []
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
        out.append((-logit).float().cpu().numpy())
    return np.concatenate(out)


def train_epoch(
    model: M,
    opt: torch.optim.Optimizer,
    lossf: nn.Module,
    tr: pd.DataFrame,
    tok: PreTrainedTokenizerBase,
    nodes: pd.DataFrame,
    axis: pd.Series,
) -> float:
    """Run one training epoch over ``tr`` and return the mean batch loss."""
    model.train()
    tot = 0
    nb = 0
    for ids, att, ax, y in batches(tr, BATCH_SIZE, True, tok, nodes, axis):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(ids, att, ax)
        loss = lossf(logit.float(), y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.item()
        nb += 1
    return tot / nb


def is_better(r: dict, best: tuple[int, dict] | None) -> bool:
    """Epoch selection rule: largest sep subject to dcdev_below0_pct <= 1.0."""
    if best is None:
        return True
    if r["dcdev_below0_pct"] <= 1.0 and r["sep"] > best[1]["sep"]:
        return True
    return best[1]["dcdev_below0_pct"] > 1.0 and r["dcdev_below0_pct"] <= 1.0


def print_examples(wide: pd.DataFrame, sc: str) -> None:
    """Print v1 and fine-tuned scores for a fixed list of example answer names."""
    for nm in EXAMPLE_NAMES:
        h = wide[wide.name == nm]
        if len(h):
            print(f"  {nm:28s} v1 {h.v1.iloc[0]:6.2f} -> ft {h[sc].iloc[0]:6.2f}")


def main() -> None:
    """Fine-tune, select the epoch on the dev proxies and write scores and the log."""
    torch.manual_seed(FT_SEED)
    np.random.seed(FT_SEED)
    nodes = pd.read_parquet(
        DATA / "nodes.parquet", columns=["id", "name", "description", "prefix"]
    ).set_index("id")
    axis = pd.read_parquet(DATA / "generality_axis.parquet", columns=["id", "gen_axis"]).set_index(
        "id"
    )["gen_axis"]
    full, dc_dev = build_training_set(nodes)
    n_hold = int(HOLDOUT_FRAC * len(full))
    hold = full.iloc[:n_hold]
    tr = full.iloc[n_hold:]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = M().cuda()
    opt = torch.optim.AdamW(
        [
            {"params": model.enc.parameters(), "lr": 2e-5},
            {"params": model.head.parameters(), "lr": 1e-3},
        ],
        weight_decay=0.01,
    )
    pos_w = torch.tensor((tr.label == 0).sum() / (tr.label == 1).sum(), device="cuda")
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    wide = pd.read_parquet(RQ3 / "data" / "answer_scores_wide.parquet").set_index("id")
    sample_b = pd.read_csv(ROOT / "sample_b" / "results" / "sample_b_panel_scores.csv")
    sample_a = pd.read_csv(ROOT / "v1" / "sample_a_v1_scores.csv")
    blocked = sorted(
        set(pd.read_csv(RQ3 / "results_wide_v1lt0_iclt50" / "removed_blocklist.csv").id)
    )
    ckpt = V2 / f"v2ft_best{TAG}.pt"

    best: tuple[int, dict] | None = None
    log = []
    for ep in range(N_EPOCHS):
        train_loss = train_epoch(model, opt, lossf, tr, tok, nodes, axis)
        s_h = -predict(model, tok, nodes, axis, hold.id.tolist())
        auc = roc_auc_score(hold.label, s_h)
        s_dev = predict(model, tok, nodes, axis, dc_dev)
        s_blk = predict(model, tok, nodes, axis, blocked)
        s_w = predict(model, tok, nodes, axis, wide.index.tolist())
        r = {
            "epoch": ep + 1,
            "train_loss": round(train_loss, 4),
            "holdout_auc": round(auc, 4),
            "dcdev_below0_pct": round(100 * float((s_dev < 0).mean()), 2),
            "dcdev_median": round(float(np.median(s_dev)), 2),
            "blocked_below0_pct": round(100 * float((s_blk < 0).mean()), 1),
            "blocked_median": round(float(np.median(s_blk)), 2),
            "wide_below0_pct": round(100 * float((s_w < 0).mean()), 1),
        }
        r["sep"] = round(r["blocked_below0_pct"] - r["dcdev_below0_pct"], 2)
        log.append(r)
        print(r, flush=True)
        wide[f"score_ft_ep{ep + 1}"] = s_w
        if is_better(r, best):
            best = (ep + 1, r)
            torch.save(model.state_dict(), ckpt)

    print("\nSELECTED epoch:", best[0])
    model.load_state_dict(torch.load(ckpt))
    rho_a = spearmanr(predict(model, tok, nodes, axis, sample_a.id.tolist()), sample_a.panel)
    rho_b = spearmanr(predict(model, tok, nodes, axis, sample_b.id.tolist()), sample_b.panel)
    print(f"  Sample A rho = {rho_a.statistic:.3f}   Sample B rho = {rho_b.statistic:.3f}")
    print_examples(wide, f"score_ft_ep{best[0]}")
    wide.to_parquet(V2 / f"wide_answers_ft_scores{TAG}.parquet")
    with open(V2 / f"finetune_v2_results{TAG}.json", "w") as fh:
        json.dump({"selected_epoch": best[0], "log": log}, fh, indent=1)


if __name__ == "__main__":
    main()

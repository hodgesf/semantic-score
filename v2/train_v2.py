"""Train the v2 candidate variants: the v1 recipe with DrugCentral train-half drugs as label 0.

Inputs: data/nodes.parquet, data/generality_axis.parquet, data/training_set_v1.parquet,
data/biolord_v1.npz, v2/drugcentral_split.parquet, the wide answer scores and the Sample A/B
panel scores. Missing embeddings are computed and cached in v2/dc_blocked_biolord.npz
(DrugCentral drugs and hand-list removals) and v2/answers_sampleb_biolord.npz (wide answers and
Sample B). Outputs: v2/wide_answers_v2_scores.parquet and v2/train_v2_results.json. Variant
selection: largest separation at score 0 (sep = %blocklisted answers < 0 - %DrugCentral dev < 0)
subject to at most 1% of held-back drugs below 0; Sample A/B rho is printed only for the
selected variant and for v1.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, ROOT, RQ3, SEED, V2  # noqa: E402

MODEL_NAME = "FremyCompany/BioLORD-2023"
COMMON_DRUGS = [
    "Prednisone",
    "Methylprednisolone",
    "Dexamethasone",
    "Cortisone",
    "Cefazolin",
    "Rituximab",
    "Hydrocortisone",
    "penicillin",
    "Phylloquinone",
    "Aspirin",
]
GENERIC_CLASSES = [
    "Pharmaceutical Preparations",
    "Antibiotics",
    "glucocorticoid",
    "Immunomodulators",
    "Vitamin D Drug Class",
    "Adrenal Cortex Hormones",
    "antibacterial drug",
    "Diuretics",
    "steroid",
    "Anti-Inflammatory Agents",
]


def node_text(nodes: pd.DataFrame, i: str) -> str:
    """Return 'name: description' for node ``i``, or just the name when no description."""
    if isinstance(nodes.description.get(i), str) and nodes.description[i]:
        return f"{nodes.name[i]}: {nodes.description[i]}"
    return str(nodes.name.get(i, i))


def embed(
    tok: PreTrainedTokenizerBase, bl: PreTrainedModel, nodes: pd.DataFrame, ids: list[str]
) -> np.ndarray:
    """Mean-pooled, L2-normalised BioLORD embeddings of the node texts for ``ids``."""
    texts = [node_text(nodes, i) for i in ids]
    out = []
    with torch.inference_mode():
        for s in range(0, len(texts), 256):
            t = tok(
                texts[s : s + 256],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to("cuda")
            h = bl(**t).last_hidden_state
            m = t["attention_mask"].unsqueeze(-1)
            pooled = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=1)
            out.append(pooled.float().cpu().numpy())
    return np.vstack(out)


def load_embedding_store(
    tok: PreTrainedTokenizerBase,
    bl: PreTrainedModel,
    nodes: pd.DataFrame,
    wide: pd.DataFrame,
    sample_b: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Collect embeddings of the training set, DrugCentral drugs, wide answers and Sample B."""
    z = np.load(DATA / "biolord_v1.npz", allow_pickle=True)
    emb = {i: e for i, e in zip(z["ids"], z["emb"])}
    dc_path = V2 / "dc_blocked_biolord.npz"
    if not dc_path.exists():
        split = pd.read_parquet(V2 / "drugcentral_split.parquet")
        blocked = pd.read_csv(RQ3 / "results_wide_v1lt0_iclt50" / "removed_blocklist.csv").id
        need = sorted(set(split.id) | set(blocked))
        np.savez(dc_path, ids=np.array(need), full=embed(tok, bl, nodes, need))
    d = np.load(dc_path, allow_pickle=True)
    emb.update({i: e for i, e in zip(d["ids"], d["full"])})
    cache = V2 / "answers_sampleb_biolord.npz"
    if os.path.exists(cache):
        c = np.load(cache, allow_pickle=True)
        emb.update({i: e for i, e in zip(c["ids"], c["emb"])})
    else:
        need = [i for i in list(wide.index) + list(sample_b.id) if i not in emb]
        e_new = embed(tok, bl, nodes, need)
        np.savez(cache, ids=np.array(need), emb=e_new)
        emb.update({i: e for i, e in zip(need, e_new)})
    return emb


def feats(emb: dict[str, np.ndarray], axis: pd.Series, ids: list[str]) -> np.ndarray:
    """Stack embeddings for ``ids`` with the generality axis appended."""
    return np.hstack([np.vstack([emb[i] for i in ids]), axis.reindex(ids).to_numpy()[:, None]])


def build_variants(
    nodes: pd.DataFrame,
) -> tuple[dict[str, tuple[pd.DataFrame, np.ndarray | None]], list[str]]:
    """Build the training-set variants (frame, sample weights) and the DrugCentral dev ids."""
    t1 = pd.read_parquet(DATA / "training_set_v1.parquet")[["id", "label"]]
    split = pd.read_parquet(V2 / "drugcentral_split.parquet")
    dc_train = split[split.split == "train"].id.tolist()
    dc_dev = split[split.split == "dev"].id.tolist()
    t1 = t1[~t1.id.isin(dc_dev)]
    iupac = t1.id.map(nodes.prefix).isin(["PUBCHEM.COMPOUND", "CHEMBL.COMPOUND"]) & (t1.label == 0)
    add = pd.DataFrame({"id": [i for i in dc_train if i not in set(t1.id)], "label": 0})
    variants = {
        "v1_ref": (t1, None),
        "v2a_dctrain": (pd.concat([t1, add]), None),
        "v2b_dctrain_w3": (pd.concat([t1, add]), np.r_[np.ones(len(t1)), 3 * np.ones(len(add))]),
        "v2c_dctrain_noIUPAC": (pd.concat([t1[~iupac], add]), None),
        "v2d_dctrain_w3_noIUPAC": (
            pd.concat([t1[~iupac], add]),
            np.r_[np.ones((~iupac).sum()), 3 * np.ones(len(add))],
        ),
    }
    return variants, dc_dev


def logit_pipeline() -> Pipeline:
    """Return the v1 recipe: scaled, class-balanced logistic regression."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=SEED),
    )


def proxy_row(
    name: str,
    n_train: int,
    cv: float,
    s_dev: np.ndarray,
    s_blk: np.ndarray,
    s_w: np.ndarray,
) -> dict:
    """Summarise the dev-proxy scores of one variant."""
    r = {
        "variant": name,
        "n_train": n_train,
        "cv_auc": round(cv, 4),
        "dcdev_below0_pct": round(100 * (s_dev < 0).mean(), 2),
        "dcdev_median": round(np.median(s_dev), 2),
        "dcdev_p05": round(np.quantile(s_dev, 0.05), 2),
        "blocked_below0_pct": round(100 * (s_blk < 0).mean(), 1),
        "blocked_median": round(np.median(s_blk), 2),
        "wide_answers_below0_pct": round(100 * (s_w < 0).mean(), 1),
    }
    r["sep"] = round(r["blocked_below0_pct"] - r["dcdev_below0_pct"], 2)
    return r


def to_json_value(v: object) -> object:
    """Convert numpy scalars to Python floats for JSON output."""
    return float(v) if hasattr(v, "item") else v


def print_named(wide: pd.DataFrame, names: list[str], sc: str, label: str, width: int) -> None:
    """Print v1 and selected-variant scores for the given answer names."""
    for nm in names:
        hit = wide[wide.name == nm]
        if len(hit):
            print(f"  {nm:{width}s} v1 {hit.v1.iloc[0]:6.2f} -> {label}{hit[sc].iloc[0]:6.2f}")


def main() -> None:
    """Fit every variant, select one on the dev proxies and write scores and the table."""
    nodes = pd.read_parquet(
        DATA / "nodes.parquet", columns=["id", "name", "description", "prefix"]
    ).set_index("id")
    axis = pd.read_parquet(DATA / "generality_axis.parquet", columns=["id", "gen_axis"]).set_index(
        "id"
    )["gen_axis"]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    bl = AutoModel.from_pretrained(MODEL_NAME).cuda().eval().half()
    wide = pd.read_parquet(RQ3 / "data" / "answer_scores_wide.parquet").set_index("id")
    sample_b = pd.read_csv(ROOT / "sample_b" / "results" / "sample_b_panel_scores.csv")
    emb = load_embedding_store(tok, bl, nodes, wide, sample_b)
    variants, dc_dev = build_variants(nodes)
    blocked = sorted(
        set(pd.read_csv(RQ3 / "results_wide_v1lt0_iclt50" / "removed_blocklist.csv").id)
    )
    dev_ids = [i for i in dc_dev if i in emb]
    sample_a = pd.read_csv(ROOT / "v1" / "sample_a_v1_scores.csv")

    rows = []
    models: dict[str, Pipeline] = {}
    for name, (train, w) in variants.items():
        f = feats(emb, axis, train.id.tolist())
        y = train.label.to_numpy()
        m = logit_pipeline()
        fit_kw = {"logisticregression__sample_weight": w} if w is not None else {}
        m.fit(f, y, **fit_kw)
        models[name] = m
        cv = cross_val_score(
            logit_pipeline(),
            f,
            y,
            cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
            scoring="roc_auc",
        ).mean()
        s_dev = -m.decision_function(feats(emb, axis, dev_ids))
        s_blk = -m.decision_function(feats(emb, axis, blocked))
        s_w = -m.decision_function(feats(emb, axis, wide.index.tolist()))
        rows.append(proxy_row(name, len(train), cv, s_dev, s_blk, s_w))
        wide[f"score_{name}"] = s_w

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    elig = table[(table.variant != "v1_ref") & (table.dcdev_below0_pct <= 1.0)]
    candidates = elig if len(elig) else table[table.variant != "v1_ref"]
    sel = candidates.sort_values("sep", ascending=False).variant.iloc[0]
    print("\nSELECTED:", sel)
    for name in ["v1_ref", sel]:
        m = models[name]
        s_a = -m.decision_function(feats(emb, axis, sample_a.id.tolist()))
        s_b = -m.decision_function(feats(emb, axis, sample_b.id.tolist()))
        rho_a = spearmanr(s_a, sample_a.panel).statistic
        rho_b = spearmanr(s_b, sample_b.panel).statistic
        print(f"  {name:22s} Sample A rho = {rho_a:.3f}   Sample B rho = {rho_b:.3f}")
    wide.to_parquet(V2 / "wide_answers_v2_scores.parquet")
    summary = {
        "selected": sel,
        "table": [{k: to_json_value(v) for k, v in r.items()} for r in rows],
    }
    with open(V2 / "train_v2_results.json", "w") as fh:
        json.dump(summary, fh, indent=1)

    sc = f"score_{sel}"
    print("\nselected variant — most common drugs:")
    print_named(wide, COMMON_DRUGS, sc, f"{sel} ", 20)
    print("\nselected variant — generic classes:")
    print_named(wide, GENERIC_CLASSES, sc, "", 28)
    print("\nselected variant — lowest-scoring DrugCentral-dev drugs:")
    dv = pd.DataFrame(
        {
            "id": dev_ids,
            "name": nodes.name.reindex(dev_ids).to_numpy(),
            "s": -models[sel].decision_function(feats(emb, axis, dev_ids)),
        }
    ).nsmallest(15, "s")
    print(dv.to_string(index=False))


if __name__ == "__main__":
    main()

"""Evaluation tables for one or more score columns of data/scores.parquet.

For each column: score percentiles by curated-IC band, band-versus-top-band AUCs,
Spearman and Kendall agreement with curated IC, and the curated-versus-matched AUC
within node-degree quintiles together with the AUC of log degree alone. Node degree
and the predicted-IC baseline are read from PREDICTED_IC_DIR. Writes
results/report_tables_<cols>.txt.

Usage: python report_tables.py <col> [<col> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score

from paths import DATA, PREDICTED_IC_DIR, RESULTS, SEED

BAND_EDGES = [-1, 50, 66, 88, 95, 100.01]
BAND_LABELS = ["0-50", "50-66", "66-88", "88-95", "95-100"]


def read_ids(path: Path) -> set[str]:
    """Return the non-empty lines of a text file as a set."""
    with open(path) as handle:
        return {line.strip() for line in handle}


def load_table(cols: list[str]) -> pd.DataFrame:
    """Join nodes, scores, degree, predicted IC and the curated-generic flag."""
    nodes = pd.read_parquet(
        DATA / "nodes.parquet", columns=["id", "name", "category", "information_content"]
    )
    nodes["category"] = nodes.category.astype(str)
    sc = pd.read_parquet(DATA / "scores.parquet", columns=["id", *cols])
    feats = pd.read_parquet(
        PREDICTED_IC_DIR / "generic_concept_features.parquet", columns=["id", "degree"]
    )
    pred = pd.read_parquet(PREDICTED_IC_DIR / "predicted_ic_full.parquet").rename(
        columns={"predicted_ic": "pred_ic"}
    )
    df = nodes.merge(sc, on="id").merge(feats, on="id", how="left").merge(pred, on="id", how="left")
    pos = set(pd.read_parquet(DATA / "positive_generic.parquet").id)
    df["curated"] = df.id.isin(pos)
    return df


def curated_with_negatives(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Return the curated generics with 20 category-matched unlabelled negatives each."""
    llm_pos = read_ids(DATA / "llm_confirmed_generics.txt")
    neg: list[int] = []
    for cat, k in df[df.curated].category.value_counts().items():
        pool = df.index[(df.category == cat) & ~df.curated & ~df.id.isin(llm_pos)]
        neg.extend(rng.choice(pool, min(len(pool), 20 * k), replace=False))
    return pd.concat([df[df.curated].assign(y=1), df.loc[neg].assign(y=0)])


def band_lines(ic: pd.DataFrame, col: str) -> list[str]:
    """Return the IC-band percentile table, band AUCs and IC agreement for one column."""
    s = ic.dropna(subset=[col])
    s = s.assign(pct=s[col].rank(pct=True))
    t = s.groupby("band", observed=True).pct.agg(
        n="size",
        median="median",
        q25=lambda v: v.quantile(0.25),
        q75=lambda v: v.quantile(0.75),
    )
    out = [
        "score percentile (within IC-covered nodes) by Babel IC band:\n" + t.round(3).to_string()
    ]
    for lo in BAND_LABELS[:-1]:
        sub = s[s.band.isin([lo, "95-100"])]
        y = (sub.band == lo).astype(int)
        out.append(f"  AUC band {lo} vs 95-100: {roc_auc_score(y, -sub[col]):.3f}")
    rho = spearmanr(s[col], s.information_content).statistic
    tau = kendalltau(
        s[col].sample(200_000, random_state=1),
        s.information_content.sample(200_000, random_state=1),
    ).statistic
    out.append(f"  Spearman with Babel IC {rho:.3f}; Kendall tau {tau:.3f}")
    return out


def degree_lines(both: pd.DataFrame, col: str) -> list[str]:
    """Return curated-versus-matched AUCs within degree quintiles for one column."""
    b = both.dropna(subset=[col, "degree"]).copy()
    b["dec"] = pd.qcut(np.log1p(b.degree).rank(method="first"), 5, labels=False)
    rows = []
    for d, g in b.groupby("dec"):
        if g.y.sum() >= 5 and (g.y == 0).sum() >= 5:
            rows.append(
                f"    degree quintile {d}: n_pos={int(g.y.sum())} n_neg={int((g.y == 0).sum())} "
                f"AUC={roc_auc_score(g.y, -g[col]):.3f} "
                f"(degree range {int(g.degree.min())}-{int(g.degree.max())})"
            )
    return [
        "curated-vs-matched AUC within degree quintiles:\n" + "\n".join(rows),
        f"  curated AUC overall {roc_auc_score(b.y, -b[col]):.3f}; "
        f"AUC of log degree itself {roc_auc_score(b.y, b.degree):.3f}",
    ]


def main(cols: list[str]) -> None:
    """Write the tables for the requested columns plus the predicted-IC baseline."""
    df = load_table(cols)
    rng = np.random.default_rng(SEED)
    both = curated_with_negatives(df, rng)
    ic = df.dropna(subset=["information_content"]).copy()
    ic["band"] = pd.cut(ic.information_content, BAND_EDGES, labels=BAND_LABELS)
    out: list[str] = []
    for col in [*cols, "pred_ic"]:
        out.append(f"\n##### {col}")
        out.extend(band_lines(ic, col))
        out.extend(degree_lines(both, col))
    text = "\n".join(out)
    print(text)
    with open(RESULTS / f"report_tables_{'_'.join(cols)}.txt", "w") as handle:
        handle.write(text)


if __name__ == "__main__":
    main(sys.argv[1:])

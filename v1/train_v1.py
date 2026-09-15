"""Fit the v1 reference score and score Sample A once.

The v1 score is a class-balanced logistic regression on the BioLORD-2023 "name: description"
embedding with the generality axis appended, fitted on data/training_set_v1.parquet. The v2
scripts read the Sample A table it writes for the item ids, the panel mean and this reference
column. Inputs: data/training_set_v1.parquet, panel/sample_a_ratings.csv, data/biolord_v1.npz and
data/generality_axis.parquet. Output: v1/sample_a_v1_scores.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, PANEL, V1  # noqa: E402

PANEL_CSV = PANEL / "sample_a_ratings.csv"
OUT_FILE = V1 / "sample_a_v1_scores.csv"
SEED = 2654
RATERS = ["roach", "ramsey", "koslicki"]
ARM = "biolord+axis"


def features(ids: np.ndarray) -> np.ndarray:
    """BioLORD embedding with the generality axis appended, one row per id."""
    bl = np.load(DATA / "biolord_v1.npz", allow_pickle=True)
    row = pd.Series(np.arange(len(bl["ids"])), index=bl["ids"])
    axis = pd.read_parquet(DATA / "generality_axis.parquet", columns=["id", "gen_axis"])
    axis = axis.set_index("id")["gen_axis"]
    return np.hstack([bl["emb"][row[ids].to_numpy()], axis[ids].to_numpy()[:, None]])


def model() -> Pipeline:
    """Standardised, class-balanced logistic regression."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED),
    )


def score() -> pd.DataFrame:
    """Fit on the training set and return the Sample A table with the v1 column."""
    train = pd.read_parquet(DATA / "training_set_v1.parquet")
    panel = pd.read_csv(PANEL_CSV)
    panel["panel"] = panel[RATERS].mean(axis=1)
    m = model().fit(features(train["id"].to_numpy()), train["label"].to_numpy())
    res = panel[["id", "panel", "information_content"]].copy()
    res[ARM] = -m.decision_function(features(panel["id"].to_numpy()))  # higher = specific
    return res


def main() -> None:
    """Score Sample A and write the table."""
    res = score()
    print(f"Sample A rho ({ARM}) = {spearmanr(res[ARM], res.panel).statistic:.3f}")
    res.to_csv(OUT_FILE, index=False)
    print(f"wrote {OUT_FILE}")


if __name__ == "__main__":
    main()

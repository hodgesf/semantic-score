"""RQ1 analysis: agreement between Babel/UberGraph IC and expert genericness ratings.

Reads the final Sample A panel CSV (roach, ramsey, koslicki) and data/nodes.parquet for
coverage numbers (overall, per category with at least 2,000 nodes, and the large categories
with almost no IC), and writes numbers.json.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, PANEL, RQ1  # noqa: E402

CSV = PANEL / "sample_a_ratings.csv"
NODES = DATA / "nodes.parquet"
NUMBERS = RQ1 / "numbers.json"
RATERS = ["roach", "ramsey", "koslicki"]
BANDS = [-1, 50, 66, 88, 95, 100.01]
BAND_LABELS = ["0-50", "50-66", "66-88", "88-95", "95-100"]
SEED = 2654
N_BOOT = 10_000

EXAMPLE_COLS = ["name", "category", "information_content", "panel"]


def load_panel() -> pd.DataFrame:
    """Read the panel CSV and add the panel mean and IC band columns."""
    df = pd.read_csv(CSV)
    df["panel"] = df[RATERS].mean(axis=1)
    df["band"] = pd.cut(df.information_content, BANDS, labels=BAND_LABELS)
    return df


def ordinal_distance(nc: np.ndarray, a: int, b: int) -> float:
    """Squared ordinal distance between rating categories a and b given marginal counts."""
    lo, hi = min(a, b), max(a, b)
    return (nc[lo : hi + 1].sum() - (nc[lo] + nc[hi]) / 2) ** 2


def kripp_ordinal(mat: np.ndarray) -> float:
    """Krippendorff's alpha (ordinal metric) for a units x raters matrix of 1-5 ratings."""
    o = np.zeros((5, 5))
    for row in mat:
        row = row[~np.isnan(row)]
        m = len(row)
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[int(row[i]) - 1, int(row[j]) - 1] += 1 / (m - 1)
    nc = o.sum(axis=1)
    n = o.sum()
    d_o = sum(o[a, b] * ordinal_distance(nc, a, b) for a in range(5) for b in range(5))
    d_e = sum(nc[a] * nc[b] * ordinal_distance(nc, a, b) for a in range(5) for b in range(5))
    d_e = d_e / (n - 1)
    return 1 - d_o / d_e


def ic_vs_panel(df: pd.DataFrame, rng: np.random.Generator) -> tuple[float, dict]:
    """Spearman rho of IC against the panel mean with bootstrap CI, tau and per-rater rho."""
    rho = spearmanr(df.information_content, df.panel).statistic
    boot = [
        spearmanr(df.information_content.iloc[i], df.panel.iloc[i]).statistic
        for i in (rng.integers(0, len(df), len(df)) for _ in range(N_BOOT))
    ]
    stats = {
        "spearman": round(rho, 3),
        "ci": [round(np.percentile(boot, 2.5), 3), round(np.percentile(boot, 97.5), 3)],
        "kendall": round(kendalltau(df.information_content, df.panel).statistic, 3),
        "per_rater": {
            r: round(spearmanr(df.information_content, df[r]).statistic, 3) for r in RATERS
        },
    }
    return rho, stats


def band_stats(df: pd.DataFrame) -> tuple[list[dict], dict[str, float]]:
    """Within-band rho and panel summaries, and AUC of each lower band against 95-100."""
    bands = []
    for b, g in df.groupby("band", observed=True):
        bands.append(
            {
                "band": b,
                "n": len(g),
                "rho": round(spearmanr(g.information_content, g.panel).statistic, 3),
                "panel_mean": round(g.panel.mean(), 2),
                "panel_sd": round(g.panel.std(), 2),
            }
        )
    band_auc = {}
    for lo in BAND_LABELS[:-1]:
        sub = df[df.band.isin([lo, "95-100"])]
        y = (sub.band == lo).astype(int)
        band_auc[lo] = round(roc_auc_score(y, -sub.information_content), 3)
    return bands, band_auc


def category_stats(df: pd.DataFrame) -> list[dict]:
    """Within-category rho for categories with at least six items, largest first."""
    cats = []
    for cat, g in df.groupby("category"):
        if len(g) >= 6:
            cats.append(
                {
                    "category": cat,
                    "n": len(g),
                    "rho": round(spearmanr(g.information_content, g.panel).statistic, 3),
                }
            )
    cats.sort(key=lambda r: -r["n"])
    return cats


def threshold_stats(df: pd.DataFrame) -> dict[str, dict]:
    """ROC of IC against panel consensus (generic = panel mean <= cut) and the Youden cut."""
    threshold = {}
    for cut in (2.0, 2.5):
        y = (df.panel <= cut).astype(int)
        auc = roc_auc_score(y, -df.information_content)
        fpr, tpr, thr = roc_curve(y, -df.information_content)
        j = np.argmax(tpr - fpr)
        threshold[str(cut)] = {
            "n_generic": int(y.sum()),
            "auc": round(auc, 3),
            "ic_thresh": round(-thr[j], 1),
            "sens": round(tpr[j], 2),
            "spec": round(1 - fpr[j], 2),
        }
    return threshold


def coverage_frac(v: pd.Series) -> float:
    """Fraction of non-missing values."""
    return v.notna().mean()


def coverage_stats() -> tuple[dict, list[dict]]:
    """IC coverage over the node table and the large categories with almost no IC."""
    nodes = pd.read_parquet(NODES, columns=["category", "information_content"])
    nodes["category"] = nodes.category.astype(str)
    coverage = {
        "total": len(nodes),
        "with_ic": int(nodes.information_content.notna().sum()),
        "pct": round(100 * nodes.information_content.notna().mean(), 1),
    }
    zero = nodes.groupby("category").information_content.agg(sz="size", cov=coverage_frac)
    zero = zero[(zero["sz"] >= 500) & (zero["cov"] < 0.05)].sort_values("sz", ascending=False)
    zero_ic_cats = [
        {"category": c.split(":")[1], "n": int(r["sz"]), "cov": round(100 * r["cov"], 1)}
        for c, r in zero.iterrows()
    ]
    return coverage, zero_ic_cats


def category_coverage() -> list[dict]:
    """IC coverage per category with at least 2,000 nodes, highest coverage first."""
    nodes = pd.read_parquet(NODES, columns=["category", "information_content"])
    nodes["category"] = nodes["category"].astype(str)
    g = nodes.groupby("category").information_content.agg(n="size", cov=coverage_frac)
    g = g[g["n"] >= 2000].sort_values("cov", ascending=False)
    return [
        {"category": c.split(":")[1], "n": int(r["n"]), "cov": round(100 * r["cov"], 1)}
        for c, r in g.iterrows()
    ]


def example_stats(df: pd.DataFrame) -> dict[str, list[dict]]:
    """Agreeing extremes and the largest IC-versus-panel rank disagreements."""
    df["ic_rank"] = df.information_content.rank()
    df["p_rank"] = df.panel.rank()
    df["d"] = df.ic_rank - df.p_rank
    agree = df[df.d.abs() <= 12]
    return {
        "agree_low": agree.nsmallest(4, "panel")[EXAMPLE_COLS].to_dict("records"),
        "agree_high": agree.nlargest(4, "panel")[EXAMPLE_COLS].to_dict("records"),
        "ic_high_panel_low": df.nlargest(5, "d")[EXAMPLE_COLS].to_dict("records"),
        "ic_low_panel_high": df.nsmallest(5, "d")[EXAMPLE_COLS].to_dict("records"),
    }


def main() -> None:
    """Compute every RQ1 number and write numbers.json."""
    df = load_panel()
    out: dict = {"n": len(df)}

    out["pairwise"] = {
        f"{a}-{b}": round(spearmanr(df[a], df[b]).statistic, 3) for a, b in combinations(RATERS, 2)
    }
    out["alpha"] = round(kripp_ordinal(df[RATERS].to_numpy()), 3)

    rng = np.random.default_rng(SEED)
    _, out["ic_panel"] = ic_vs_panel(df, rng)
    out["bands"], out["band_auc"] = band_stats(df)
    out["categories"] = category_stats(df)
    out["threshold"] = threshold_stats(df)
    out["coverage"], out["zero_ic_cats"] = coverage_stats()
    out["category_coverage"] = category_coverage()
    out.update(example_stats(df))

    with open(NUMBERS, "w") as fh:
        json.dump(out, fh, indent=1, default=str)

    print(json.dumps(out, indent=1, default=str)[:3500])


if __name__ == "__main__":
    main()

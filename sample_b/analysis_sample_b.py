"""Sample B coverage test: sealed v1 arms and the label-free axis against the panel mean.

Reads the Sample B ratings workbook, data/sample_b_key.csv and results/sample_b_scores_SEALED.csv,
computes panel agreement, Spearman rho per arm with bootstrap CIs and paired differences, and
post-hoc AUC, per-category and disagreement listings. Writes sample_b_panel_scores.csv,
sample_b_analysis.txt and sample_b_numbers.json under sample_b/results/.
"""

from __future__ import annotations

import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import RESULTS, SAMPLE_B, SAMPLE_B_XLSX  # noqa: E402

XLSX = SAMPLE_B_XLSX
KEY_FILE = SAMPLE_B / "data" / "sample_b_key.csv"
SEALED_FILE = RESULTS / "sample_b_scores_SEALED.csv"
OUT_DIR = SAMPLE_B / "results"
RATERS = ["roach", "ramsey", "koslicki"]
ARMS = {
    "v1_biolord_axis": "v1 (BioLORD+axis) — primary",
    "v1_sapbert_axis": "v1 SapBERT+axis",
    "ic_axis": "label-free axis",
}
ARM_PAIRS = [
    ("v1_biolord_axis", "v1_sapbert_axis"),
    ("v1_biolord_axis", "ic_axis"),
    ("v1_sapbert_axis", "ic_axis"),
]
SEED = 2654
N_BOOT = 10000


def read_ratings(name_to_id: dict[str, str]) -> dict[str, pd.Series]:
    """Read each rater's WHO-labelled block from the consolidated sheet, indexed by node id."""
    raw = pd.read_excel(XLSX, sheet_name="Form Responses 1", header=None)
    ratings: dict[str, pd.Series] = {}
    i = 0
    while i < len(raw):
        if str(raw.iat[i, 0]).strip() == "WHO":
            header = [re.sub(r"^\d+\.\s*", "", str(c)).strip() for c in raw.iloc[i, 1:]]
            who = str(raw.iat[i + 1, 0]).strip().lower()
            vals = pd.to_numeric(raw.iloc[i + 1, 1:], errors="coerce").to_numpy()
            ids = [name_to_id[h] for h in header]
            assert len(set(ids)) == 50 and not np.isnan(vals).any()
            assert set(vals) <= {1, 2, 3, 4, 5}
            ratings[who] = pd.Series(vals, index=ids)
            i += 2
        else:
            i += 1
    assert set(ratings) == set(RATERS), ratings.keys()
    return ratings


def cross_check(ratings: dict[str, pd.Series], name_to_id: dict[str, str]) -> None:
    """Verify that each raw per-form sheet matches exactly one rater's block."""
    for sheet in ("Form Responses 2", "Form Responses 3"):
        f = pd.read_excel(XLSX, sheet_name=sheet)
        cols = [c for c in f.columns if re.match(r"^\d+\. ", str(c))]
        index = [name_to_id[re.sub(r"^\d+\.\s*", "", c).strip()] for c in cols]
        s = pd.Series(pd.to_numeric(f[cols].iloc[0]).to_numpy(), index=index)
        who = [r for r in RATERS if (ratings[r].reindex(s.index) == s).all()]
        assert len(who) == 1, (sheet, who)
        print(f"{sheet} == {who[0]} block (cross-check OK)")


def build_table(key: pd.DataFrame, ratings: dict[str, pd.Series]) -> pd.DataFrame:
    """Join key, ratings and sealed scores into one table and write sample_b_panel_scores.csv."""
    panel = key[["item", "id", "name", "most_specific_category"]].copy()
    panel["category"] = panel.most_specific_category.str.split(":").str[1]
    for r in RATERS:
        panel[r] = ratings[r].reindex(panel.id).to_numpy()
    panel["panel"] = panel[RATERS].mean(axis=1)
    sealed = pd.read_csv(SEALED_FILE)
    assert (sealed.id == panel.id).all()
    table = panel.merge(sealed[["id"] + list(ARMS)], on="id")
    table.to_csv(OUT_DIR / "sample_b_panel_scores.csv", index=False)
    return table


def boot_rho(
    rng: np.random.Generator, x: pd.Series, y: pd.Series, n: int = N_BOOT
) -> tuple[float, float]:
    """Return the 2.5 and 97.5 percentiles of the bootstrapped Spearman rho of x and y."""
    v = [
        spearmanr(x.iloc[i], y.iloc[i]).statistic
        for i in (rng.integers(0, len(x), len(x)) for _ in range(n))
    ]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def boot_diff(
    rng: np.random.Generator, a: pd.Series, b: pd.Series, y: pd.Series, n: int = N_BOOT
) -> tuple[float, float, float, float]:
    """Bootstrap rho(a, y) - rho(b, y): mean, 2.5 and 97.5 percentiles, P(diff > 0)."""
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        v.append(
            spearmanr(a.iloc[i], y.iloc[i]).statistic - spearmanr(b.iloc[i], y.iloc[i]).statistic
        )
    arr = np.array(v)
    return (
        float(arr.mean()),
        float(np.percentile(arr, 2.5)),
        float(np.percentile(arr, 97.5)),
        float((arr > 0).mean()),
    )


def ordinal_distance(nc: np.ndarray, a: int, b: int) -> float:
    """Squared ordinal distance between rating categories a and b given marginal counts."""
    lo, hi = min(a, b), max(a, b)
    return (nc[lo : hi + 1].sum() - (nc[lo] + nc[hi]) / 2) ** 2


def kripp(mat: np.ndarray) -> float:
    """Krippendorff's alpha (ordinal metric) for a units x raters matrix of 1-5 ratings."""
    o = np.zeros((5, 5))
    for row in mat:
        m = len(row)
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[int(row[i]) - 1, int(row[j]) - 1] += 1 / (m - 1)
    nc = o.sum(1)
    n = o.sum()
    d_o = sum(o[a, b] * ordinal_distance(nc, a, b) for a in range(5) for b in range(5))
    d_e = sum(nc[a] * nc[b] * ordinal_distance(nc, a, b) for a in range(5) for b in range(5))
    d_e = d_e / (n - 1)
    return 1 - d_o / d_e


def arm_label(col: str) -> str:
    """Short label for an arm column in the per-category listing."""
    return "axis" if col == "ic_axis" else col.split("_")[1]


def panel_section(table: pd.DataFrame, out: dict, lines: list[str]) -> None:
    """Add panel agreement, rater means and rating distributions."""
    out["alpha"] = round(kripp(table[RATERS].to_numpy()), 3)
    out["pairwise"] = {
        f"{a}-{b}": round(spearmanr(table[a], table[b]).statistic, 3)
        for a, b in combinations(RATERS, 2)
    }
    out["rater_means"] = {r: round(table[r].mean(), 2) for r in RATERS}
    out["rater_dist"] = {r: table[r].value_counts().sort_index().to_dict() for r in RATERS}
    out["panel_mean"] = round(table.panel.mean(), 2)
    out["panel_sd"] = round(table.panel.std(), 2)
    lines.append(f"Panel: Krippendorff alpha (ordinal) = {out['alpha']}  (Sample A: 0.727)")
    lines.append("  pairwise rho: " + ", ".join(f"{k} {v:.2f}" for k, v in out["pairwise"].items()))
    lines.append(
        "  rater means: "
        + ", ".join(f"{r} {m}" for r, m in out["rater_means"].items())
        + f";  panel mean {out['panel_mean']} (sd {out['panel_sd']})"
    )
    lines.append(
        "  rating distributions: "
        + "; ".join(
            f"{r} " + " ".join(f"{k}:{v}" for k, v in d.items())
            for r, d in out["rater_dist"].items()
        )
    )


def prespecified_section(
    table: pd.DataFrame, out: dict, lines: list[str], rng: np.random.Generator
) -> None:
    """Add per-arm rho with bootstrap CI, tau, per-rater rho and paired arm differences."""
    lines.append(
        "\nPRE-SPECIFIED: Spearman rho with panel mean, 95% bootstrap CI (10k), Kendall tau, "
        "per-rater rho"
    )
    out["arms"] = {}
    for col, lab in ARMS.items():
        rho = spearmanr(table[col], table.panel).statistic
        ci = boot_rho(rng, table[col], table.panel)
        out["arms"][col] = {
            "rho": round(rho, 3),
            "ci": [round(c, 3) for c in ci],
            "tau": round(kendalltau(table[col], table.panel).statistic, 3),
            "per_rater": {r: round(spearmanr(table[col], table[r]).statistic, 3) for r in RATERS},
        }
        a = out["arms"][col]
        lines.append(
            f"  {lab:32s} rho = {a['rho']:+.3f} [{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}]  "
            f"tau = {a['tau']:+.3f}   per-rater: "
            + ", ".join(f"{r} {v:+.2f}" for r, v in a["per_rater"].items())
        )
    out["diffs"] = {}
    for a, b in ARM_PAIRS:
        m, lo, hi, p = boot_diff(rng, table[a], table[b], table.panel)
        out["diffs"][f"{a}-{b}"] = {
            "mean": round(m, 3),
            "ci": [round(lo, 3), round(hi, 3)],
            "p_gt0": round(p, 3),
        }
        lines.append(
            f"  paired diff {a} − {b}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  P(diff>0) = {p:.3f}"
        )


def exploratory_section(table: pd.DataFrame, out: dict, lines: list[str]) -> None:
    """Add post-hoc AUC, per-category rho, and the most generic and most discordant items."""
    lines.append("\nEXPLORATORY (post hoc)")
    # Scores are specificity; negate for generic-positive AUC.
    for thr in (2.0, 2.5):
        yb = (table.panel <= thr).astype(int)
        if yb.nunique() == 2:
            aucs = {col: round(roc_auc_score(yb, -table[col]), 3) for col in ARMS}
            out[f"auc_generic_le{thr}"] = {"n_generic": int(yb.sum()), **aucs}
            lines.append(
                f"  AUC (specific vs generic, generic = panel mean <= {thr}, "
                f"n_generic={int(yb.sum())}): " + ", ".join(f"{c} {v}" for c, v in aucs.items())
            )
    out["cats"] = []
    for c, g in table.groupby("category"):
        row = {"category": c, "n": len(g), "panel_mean": round(g.panel.mean(), 2)}
        if len(g) >= 4 and g.panel.nunique() > 1:
            for col in ARMS:
                row[col] = round(spearmanr(g[col], g.panel).statistic, 3)
        out["cats"].append(row)
    lines.append("  per category (rho only where n>=4): ")
    for row in sorted(out["cats"], key=lambda r: -r["n"]):
        lines.append(
            f"    {row['category']:20s} n={row['n']:2d} panel mean {row['panel_mean']:.2f}  "
            + "  ".join(f"{arm_label(k)} {row[k]:+.2f}" for k in ARMS if k in row)
        )
    table["rank_panel"] = table.panel.rank()
    table["rank_v1"] = table.v1_biolord_axis.rank()
    table["rank_gap"] = table.rank_v1 - table.rank_panel
    lines.append(
        "\n  Items the panel rated most GENERIC (mean <= 2) with primary v1 score rank "
        "(1 = most generic of 50):"
    )
    for r in table[table.panel <= 2].sort_values("panel").itertuples():
        lines.append(
            f"    {r.panel:.2f}  {r.name[:60]:60s} {r.category:18s} v1 rank {int(r.rank_v1):2d}  "
            f"sap rank {int(table.v1_sapbert_axis.rank()[r.Index]):2d}  "
            f"axis rank {int(table.ic_axis.rank()[r.Index]):2d}"
        )
    lines.append(
        "  Largest disagreements, primary v1 vs panel (rank gap; + = v1 says more specific "
        "than panel):"
    )
    top = table.reindex(table.rank_gap.abs().sort_values(ascending=False).index).head(10)
    for r in top.itertuples():
        lines.append(
            f"    gap {r.rank_gap:+5.1f}  panel {r.panel:.2f} "
            f"({r.roach:.0f}/{r.ramsey:.0f}/{r.koslicki:.0f})  {r.name[:55]:55s} {r.category}"
        )
    out["examples_generic"] = table[table.panel <= 2][
        ["name", "category", "panel", "rank_v1"]
    ].to_dict("records")


def main() -> None:
    """Read ratings and sealed scores, run every section and write the outputs."""
    rng = np.random.default_rng(SEED)
    key = pd.read_csv(KEY_FILE)
    name_to_id = dict(zip(key.name.str.strip(), key.id))
    assert len(name_to_id) == 50
    ratings = read_ratings(name_to_id)
    cross_check(ratings, name_to_id)
    table = build_table(key, ratings)

    out: dict = {"n": len(table), "desc_pct": round(100 * key.description.notna().mean(), 1)}
    lines: list[str] = []
    lines.append(
        f"SAMPLE B — pre-specified coverage test (n={len(table)}, no item has IC; "
        f"{out['desc_pct']}% have a description)\n"
    )
    panel_section(table, out, lines)
    prespecified_section(table, out, lines, rng)
    exploratory_section(table, out, lines)

    txt = "\n".join(lines)
    print(txt)
    with open(OUT_DIR / "sample_b_analysis.txt", "w") as fh:
        fh.write(txt + "\n")
    with open(OUT_DIR / "sample_b_numbers.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()

"""Sample A/B agreement and development-set filter arms for the v2 ensemble score.

Averages the three per-seed Sample A and Sample B score tables written by ft_eval_ab.py into
v2/sample_a_ft_ens.csv and v2/sample_b_ft_ens.csv, then computes, on Sample A, Spearman rho with
10,000-resample bootstrap intervals for IC, the generality axis and the ensemble, the paired
bootstrap differences, within-band rho, rater agreement, example nodes and the largest ensemble
disagreements; on Sample B, rho with its interval and the bottom-ten hit count; and the removal
counts of the four filter arms over the 200-disease development answer lists. Inputs:
v2/sample_{a,b}_ft.csv, v2/sample_{a,b}_ft_seed2.csv, v2/sample_{a,b}_ft_seed3.csv,
panel/sample_a_ratings.csv, results/sample_a_scores_SEALED.csv,
sample_b/results/sample_b_panel_scores.csv, v2/query_answer_features.parquet,
v2/wide_answers_ft_ens.parquet and rq3/results_wide_v1lt0_iclt50/removed_blocklist.csv.
Output: v2/numbers_v2.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import PANEL, RESULTS, RQ3, SAMPLE_B, SEED, V2  # noqa: E402

RATERS = ["roach", "ramsey", "koslicki"]
BANDS = [-1, 50, 66, 88, 95, 100.01]
BL = ["0-50", "50-66", "66-88", "88-95", "95-100"]
EX_COLS = ["name", "category", "IC", "panel"]
SEED_TAGS = ["", "_seed2", "_seed3"]


def ensemble(sample: str) -> pd.Series:
    """Average the per-seed scores of one sample, write the ensemble table and return it by id."""
    runs = [pd.read_csv(V2 / f"sample_{sample}_ft{tag}.csv") for tag in SEED_TAGS]
    out = runs[0][["id", "panel"]].copy()
    out["ft_ens"] = np.mean([r.set_index("id").ft.reindex(out.id).to_numpy() for r in runs], axis=0)
    out.to_csv(V2 / f"sample_{sample}_ft_ens.csv", index=False)
    return out.set_index("id").ft_ens


def boot(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    rng: np.random.Generator,
    n: int = 10000,
) -> np.ndarray:
    """Bootstrap 95% interval of the Spearman correlation between x and y."""
    x = np.asarray(x)
    y = np.asarray(y)
    idx = (rng.integers(0, len(x), len(x)) for _ in range(n))
    return np.percentile([spearmanr(x[i], y[i]).statistic for i in idx], [2.5, 97.5])


def diff(
    a: pd.Series | np.ndarray,
    b: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    rng: np.random.Generator,
    n: int = 10000,
) -> tuple[float, float, float, float]:
    """Bootstrap of rho(a, y) - rho(b, y): mean, 2.5th and 97.5th percentiles, P(diff > 0)."""
    a, b, y = map(np.asarray, (a, b, y))
    idx = (rng.integers(0, len(y), len(y)) for _ in range(n))
    v = np.array([spearmanr(a[i], y[i]).statistic - spearmanr(b[i], y[i]).statistic for i in idx])
    return v.mean(), *np.percentile(v, [2.5, 97.5]), (v > 0).mean()


def load_sample_a(rng: np.random.Generator, v2: pd.Series) -> pd.DataFrame:
    """Sample A with panel mean, IC, ensemble score, generality axis and IC band."""
    A = pd.read_csv(PANEL / "sample_a_ratings.csv")
    A = A.rename(columns={"information_content": "IC", "mean_rating": "panel"})
    A["v2"] = v2.reindex(A.id).to_numpy()
    axis = pd.read_csv(RESULTS / "sample_a_scores_SEALED.csv").set_index("id").ic_axis
    A["axis"] = axis.reindex(A.id).to_numpy()
    A["band"] = pd.cut(A.IC, BANDS, labels=BL)
    # The published intervals were drawn after a 100-value plotting jitter from the same
    # generator; drawing it here keeps the random stream, and so the intervals, identical.
    rng.normal(0, 0.045, len(A))
    return A


def records(rows: pd.DataFrame, cols: list[str]) -> list[dict]:
    """Selected columns of ``rows`` rounded to two decimals, as a list of records."""
    return rows[cols].round(2).to_dict("records")


def sample_a_numbers(A: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Correlations, bootstrap contrasts, band statistics, rater agreement and example nodes."""
    out = {}
    for col in ["IC", "axis", "v2"]:
        out[col] = {
            "rho": round(spearmanr(A[col], A.panel).statistic, 3),
            "ci": [round(v, 3) for v in boot(A[col], A.panel, rng)],
            "tau": round(kendalltau(A[col], A.panel).statistic, 3),
            "per_rater": {r: round(spearmanr(A[col], A[r]).statistic, 3) for r in RATERS},
        }
    for a, b in [("v2", "IC"), ("axis", "IC"), ("v2", "axis")]:
        m, lo, hi, p = diff(A[a], A[b], A.panel, rng)
        out[f"{a}-{b}"] = [round(m, 3), round(lo, 3), round(hi, 3), round(p, 3)]
    out["bands"] = [
        {
            "band": b,
            "n": len(g),
            "mean": round(g.panel.mean(), 2),
            "sd": round(g.panel.std(), 2),
            "ic": round(spearmanr(g.IC, g.panel).statistic, 3),
            "axis": round(spearmanr(g.axis, g.panel).statistic, 3),
            "v2": round(spearmanr(g.v2, g.panel).statistic, 3),
        }
        for b, g in A.groupby("band", observed=True)
    ]
    out["loo_human"] = {
        r: round(spearmanr(A[r], A[[x for x in RATERS if x != r]].mean(axis=1)).statistic, 3)
        for r in RATERS
    }
    out["pairwise"] = {
        f"{a}-{b}": round(spearmanr(A[a], A[b]).statistic, 3)
        for a, b in [("roach", "ramsey"), ("roach", "koslicki"), ("ramsey", "koslicki")]
    }
    # Examples: two disagreements in each direction and two agreements (rank gap <= 10).
    A["_d"] = A.IC.rank() - A.panel.rank()
    agree = A[A._d.abs() <= 10]
    out["ex"] = {
        "ic_hi_p_lo": records(A.nlargest(2, "_d"), EX_COLS),
        "ic_lo_p_hi": records(A.nsmallest(2, "_d"), EX_COLS),
        "agree_generic": records(agree.nsmallest(2, "panel"), EX_COLS),
        "agree_specific": records(agree[agree.name.str.len() < 45].nlargest(2, "panel"), EX_COLS),
    }
    A["_dv"] = A.v2.rank() - A.panel.rank()
    worst = pd.concat([A.nsmallest(3, "_dv"), A.nlargest(3, "_dv")])
    out["v2_misses"] = records(worst, ["name", "category", "IC", "v2", "panel"])
    return out


def sample_b_numbers(v2: pd.Series, rng: np.random.Generator) -> dict:
    """Correlation, bootstrap interval, per-rater rho and bottom-10 hit count for Sample B."""
    B = pd.read_csv(SAMPLE_B / "results" / "sample_b_panel_scores.csv")
    B["v2"] = v2.reindex(B.id).to_numpy()
    return {
        "rho": round(spearmanr(B.v2, B.panel).statistic, 3),
        "ci": [round(v, 3) for v in boot(B.v2, B.panel, rng)],
        "per_rater": {r: round(spearmanr(B.v2, B[r]).statistic, 3) for r in RATERS},
        "bottom10_generic": int((B.v2.rank(method="min")[B.panel <= 2] <= 10).sum()),
    }


def arms() -> dict:
    """Removal counts, collateral counts and hand-list coverage for the four filter arms."""
    D = pd.read_parquet(V2 / "query_answer_features.parquet")
    v2 = pd.read_parquet(V2 / "wide_answers_ft_ens.parquet")["score_ft_ens"]
    D["v2"] = v2.reindex(D.id).to_numpy()
    b = pd.read_csv(RQ3 / "results_wide_v1lt0_iclt50" / "removed_blocklist.csv")
    bl = set(zip(b.case, b.id))
    D["blocked"] = [(c, i) in bl for c, i in zip(D.case, D.id)]
    n_blocked = D.blocked.sum()

    def arm(mask: pd.Series) -> dict:
        return dict(
            removed=int(mask.sum()),
            ind=int((mask & D.is_indication).sum()),
            dc=int((mask & D.is_dcdrug).sum()),
            cov=round(100 * (mask & D.blocked).sum() / n_blocked, 1),
        )

    return {
        "Hand-curated list": arm(D.blocked),
        "IC < 50": arm(D.ic < 50),
        "IC < 80": arm(D.ic < 80),
        "Semantic score < 0": arm(D.v2 < 0),
    }


def compute() -> dict:
    """Every number in the output file, in the order the random stream requires."""
    rng = np.random.default_rng(SEED)
    A = load_sample_a(rng, ensemble("a"))
    out = sample_a_numbers(A, rng)
    out["B"] = sample_b_numbers(ensemble("b"), rng)
    out["arms"] = arms()
    return out


def main() -> None:
    """Compute the numbers and write v2/numbers_v2.json."""
    out = compute()
    with open(V2 / "numbers_v2.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()

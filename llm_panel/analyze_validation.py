"""Validation of the LLM panel against the expert panel on Samples A and B.

Reads data/expert_items.csv and raw/<judge>.jsonl for the judges present, and writes
results/validation.txt, results/validation.json and results/expert_items_with_llm.csv.
Verdict rule: tier VALIDATED if Sample A rho >= 0.70, WEAK if >= 0.59, else FAILED; the generic
call condition needs AUC >= 0.85, specificity on >= 4 items >= 0.85 and sensitivity >= 0.60;
judges with per-judge rho < 0.50 are dropped.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import LLM_PANEL, SEED  # noqa: E402

DATA = LLM_PANEL / "data"
RAW = LLM_PANEL / "raw"
RESULTS = LLM_PANEL / "results"
JUDGES = ["gemma", "nemo", "phi"]
RATERS = ["roach", "ramsey", "koslicki"]


def load_ratings(path: Path, subset: str) -> pd.Series:
    """Return the parsed rating per concept id from one judge file, restricted to ``subset``."""
    with open(path) as fh:
        R = pd.DataFrame([json.loads(line) for line in fh])
    R = R[R.set == subset].drop_duplicates("id").set_index("id")
    return R.rating


def load_items() -> tuple[pd.DataFrame, list[str]]:
    """Attach the available judges' ratings to the expert items; return the table and judges."""
    E = pd.read_csv(DATA / "expert_items.csv")
    for j in JUDGES:
        p = RAW / f"{j}.jsonl"
        if p.exists():
            E[j] = load_ratings(p, "expert").reindex(E.id).to_numpy()
    avail = [j for j in JUDGES if j in E]
    E["llm_mean"] = E[avail].mean(axis=1)
    E["llm_median"] = E[avail].median(axis=1)
    return E, avail


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
    v = [spearmanr(x[i], y[i]).statistic for i in idx]
    return np.percentile(v, [2.5, 97.5])


def kripp(mat: np.ndarray) -> float:
    """Krippendorff's ordinal alpha for an items-by-raters matrix of 1..5 ratings with NaN gaps."""
    o = np.zeros((5, 5))
    for row in mat:
        vals = [int(v) for v in row if not np.isnan(v)]
        m = len(vals)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[vals[i] - 1, vals[j] - 1] += 1 / (m - 1)
    nc = o.sum(1)
    n = o.sum()

    def d(a: int, b: int) -> float:
        """Squared ordinal distance between categories a and b."""
        lo = min(a, b)
        hi = max(a, b)
        return (nc[lo : hi + 1].sum() - (nc[lo] + nc[hi]) / 2) ** 2

    Do = sum(o[a, b] * d(a, b) for a in range(5) for b in range(5))
    De = sum(nc[a] * nc[b] * d(a, b) for a in range(5) for b in range(5)) / (n - 1)
    return 1 - Do / De


def sample_stats(D: pd.DataFrame, avail: list[str], rng: np.random.Generator) -> dict:
    """Agreement statistics between the LLM panel and the expert panel for one sample."""
    res = {}
    rho = spearmanr(D.llm_mean, D.panel).statistic
    lo, hi = boot(D.llm_mean, D.panel, rng)
    tau = kendalltau(D.llm_mean, D.panel).statistic
    err = D.llm_mean - D.panel
    res.update(
        rho=round(rho, 3),
        ci=[round(lo, 3), round(hi, 3)],
        tau=round(tau, 3),
        rho_median=round(spearmanr(D.llm_median, D.panel).statistic, 3),
        mae=round(float(err.abs().mean()), 3),
        within1=round(float((err.abs() <= 1).mean()), 3),
        calibration_offset=round(float(err.mean()), 3),
        per_judge={
            j: round(spearmanr(D[j], D.panel, nan_policy="omit").statistic, 3) for j in avail
        },
        inter_judge={
            f"{a}-{b}": round(spearmanr(D[a], D[b], nan_policy="omit").statistic, 3)
            for a, b in combinations(avail, 2)
        },
        inter_judge_alpha=round(kripp(D[avail].to_numpy()), 3),
        loo_human={
            r: round(spearmanr(D[r], D[[x for x in RATERS if x != r]].mean(axis=1)).statistic, 3)
            for r in RATERS
        },
    )
    y = (D.panel <= 2).astype(int)
    if y.nunique() == 2:
        res["auc_generic_le2"] = round(roc_auc_score(y, -D.llm_mean), 3)
        pred = D.llm_mean <= 2
        spec_items = D.panel >= 4
        res["sens_at_le2"] = round(float(pred[y == 1].mean()), 3)
        res["spec_on_ge4"] = round(float((~pred[spec_items]).mean()), 3)
        res["n_generic"] = int(y.sum())
        res["n_specific_ge4"] = int(spec_items.sum())
        pred25 = D.llm_mean <= 2.5
        res["sens_at_le2.5"] = round(float(pred25[y == 1].mean()), 3)
        res["spec_on_ge4_at_le2.5"] = round(float((~pred25[spec_items]).mean()), 3)
    return res


def sample_lines(s: str, D: pd.DataFrame, res: dict) -> list[str]:
    """Report lines for one sample."""
    L = [
        f"Sample {s} (n={len(D)}): rho {res['rho']} {res['ci']}  tau {res['tau']}  "
        f"MAE {res['mae']}  ±1 {res['within1']}  offset {res['calibration_offset']}"
    ]
    L.append(
        f"  per judge {res['per_judge']}  inter-judge {res['inter_judge']}  "
        f"alpha {res['inter_judge_alpha']}"
    )
    L.append(f"  human LOO {res['loo_human']}")
    if "auc_generic_le2" in res:
        L.append(
            f"  generic call: AUC {res['auc_generic_le2']}  sens@<=2 {res['sens_at_le2']}  "
            f"spec(>=4)@<=2 {res['spec_on_ge4']}  | @<=2.5: sens {res['sens_at_le2.5']} "
            f"spec {res['spec_on_ge4_at_le2.5']}"
        )
    if s == "B":
        L.append(
            "  human-generic items in LLM bottom-10: "
            f"{res['human_generic_in_llm_bottom10']}/{res['n_human_generic']}"
        )
    return L


def verdict(out: dict, avail: list[str]) -> str:
    """Apply the tier, generic-call and judge-dropping rules to Sample A; return the verdict."""
    a = out["sample_A"]
    tier = "VALIDATED" if a["rho"] >= 0.70 else ("WEAK" if a["rho"] >= 0.59 else "FAILED")
    gen_ok = (
        a.get("auc_generic_le2", 0) >= 0.85
        and a.get("spec_on_ge4", 0) >= 0.85
        and a.get("sens_at_le2", 0) >= 0.60
    )
    dropped = [j for j in avail if a["per_judge"][j] < 0.50]
    out["tier"] = tier
    out["generic_call_condition_met"] = bool(gen_ok)
    out["judges_dropped_by_rule"] = dropped
    return (
        f"\nPRE-REGISTERED VERDICT: tier {tier}; generic-call condition "
        f"{'MET' if gen_ok else 'NOT met'}; judges dropped (rho<0.5): {dropped}"
    )


def main() -> None:
    """Compute the validation statistics, print the report and write the outputs."""
    rng = np.random.default_rng(SEED)
    E, avail = load_items()
    out = {"judges": avail, "n_unparseable": {j: int(E[j].isna().sum()) for j in avail}}
    L = [f"LLM panel validation — judges: {avail}; unparseable per judge: {out['n_unparseable']}\n"]
    for s in ["A", "B"]:
        D = E[E["sample"] == s].dropna(subset=["llm_mean"])
        res = sample_stats(D, avail, rng)
        if s == "B":
            gen = D[D.panel <= 2]
            bottom10 = D.llm_mean.rank(method="min").reindex(gen.index) <= 10
            res["human_generic_in_llm_bottom10"] = int(bottom10.sum())
            res["n_human_generic"] = len(gen)
        out[f"sample_{s}"] = res
        L.extend(sample_lines(s, D, res))
    L.append(verdict(out, avail))
    txt = "\n".join(L)
    print(txt)
    with open(RESULTS / "validation.txt", "w") as fh:
        fh.write(txt + "\n")
    with open(RESULTS / "validation.json", "w") as fh:
        json.dump(out, fh, indent=1)
    E.to_csv(RESULTS / "expert_items_with_llm.csv", index=False)


if __name__ == "__main__":
    main()

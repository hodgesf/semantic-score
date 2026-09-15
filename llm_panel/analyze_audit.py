"""Query-audit analysis with the validated LLM panel.

Reads results/validation.json for the judges to use, data/audit_manifest.csv and raw/<judge>.jsonl,
and writes results/audit_items_with_llm.csv, results/audit.txt and results/audit.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import LLM_PANEL, SEED  # noqa: E402

DATA = LLM_PANEL / "data"
RAW = LLM_PANEL / "raw"
RESULTS = LLM_PANEL / "results"


def load_ratings(judge: str, subset: str) -> pd.Series:
    """Return the parsed rating per concept id for one judge, restricted to ``subset``."""
    with open(RAW / f"{judge}.jsonl") as fh:
        R = pd.DataFrame([json.loads(line) for line in fh])
    R = R[R.set == subset].drop_duplicates("id").set_index("id")
    return R.rating


def bci(x: pd.Series | np.ndarray, rng: np.random.Generator, n: int = 10000) -> list[float]:
    """Bootstrap 95% interval of the mean (NaNs dropped), rounded to three decimals."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return [np.nan, np.nan]
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [round(v, 3) for v in np.percentile(means, [2.5, 97.5])]


def build_items(judges: list[str]) -> pd.DataFrame:
    """Attach per-judge ratings, panel summaries and generic/specific calls to the manifest."""
    M = pd.read_csv(DATA / "audit_manifest.csv")
    for j in judges:
        M[j] = load_ratings(j, "audit").reindex(M.id).to_numpy()
    M["llm_mean"] = M[judges].mean(axis=1)
    M["llm_median"] = M[judges].median(axis=1)
    M["generic"] = M.llm_mean <= 2
    M["generic25"] = M.llm_mean <= 2.5
    M["specific"] = M.llm_mean >= 4
    M.to_csv(RESULTS / "audit_items_with_llm.csv", index=False)
    return M


def stratum_table(
    M: pd.DataFrame, V: dict, judges: list[str], rng: np.random.Generator
) -> tuple[list[str], dict]:
    """Return the report header and per-stratum table lines, plus the per-stratum summary dict."""
    unparseable = {j: int(M[j].isna().sum()) for j in judges}
    L = [
        f"LLM-panel query audit — tier {V['tier']}, judges {judges}, n={len(M)} concepts, "
        "unparseable per judge: " + str(unparseable) + "\n"
    ]
    L.append(
        "| stratum | n | of | mean | median | P(generic<=2) [CI] | P(<=2.5) | "
        "P(specific>=4) [CI] | rating distribution 1..5 |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|")
    out = {}
    for s, g in M.groupby("stratum"):
        dist = np.histogram(g.llm_mean.round(), bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])[0].tolist()
        r = dict(
            n=len(g),
            of=int(g.stratum_size.iloc[0]),
            mean=round(g.llm_mean.mean(), 2),
            median=round(g.llm_mean.median(), 2),
            p_generic=round(g.generic.mean(), 3),
            ci_generic=bci(g.generic, rng),
            p_generic25=round(g.generic25.mean(), 3),
            p_specific=round(g.specific.mean(), 3),
            ci_specific=bci(g.specific, rng),
            dist=dist,
        )
        out[s] = r
        L.append(
            f"| {s} | {r['n']} | {r['of']} | {r['mean']} | {r['median']} | "
            f"{r['p_generic']} {r['ci_generic']} | {r['p_generic25']} | "
            f"{r['p_specific']} {r['ci_specific']} | {dist} |"
        )
    return L, out


def named(rows: pd.DataFrame, with_v2: bool = False) -> str:
    """Join concept names with their panel mean (and v2 score) as one line."""
    if with_v2:
        return "; ".join(
            f"{r['name']} ({r.llm_mean:.1f}, v2 {r.v2:.1f})" for _, r in rows.iterrows()
        )
    return "; ".join(f"{r['name']} ({r.llm_mean:.1f})" for _, r in rows.iterrows())


def headline_lines(M: pd.DataFrame, rng: np.random.Generator) -> list[str]:
    """Return the headline, weighted-population, covariate and example lines of the report."""
    L = []
    s1 = M[M.stratum == "S1_v2only"]
    s2 = M[M.stratum == "S2_handonly"]
    L.append(
        "\nHEADLINE: among concepts removed by v2 but NOT by the hand list "
        f"(S1, n={len(s1)} of {s1.stratum_size.iloc[0]}): "
        f"judged generic (<=2) {s1.generic.mean():.3f} {bci(s1.generic, rng)}, "
        f"<=2.5 {s1.generic25.mean():.3f}, "
        f"judged specific (>=4) {s1.specific.mean():.3f} {bci(s1.specific, rng)}"
    )
    L.append(
        "          among concepts removed by the hand list but KEPT by v2 "
        f"(S2, n={len(s2)}): judged specific (>=4) {s2.specific.mean():.3f} "
        f"{bci(s2.specific, rng)}, generic {s2.generic.mean():.3f}"
    )
    L.append("  S2 items: " + named(s2))
    # Population estimates weighted by inverse sampling probability (strata partition the set).
    w = 1 / M.sampling_prob
    removed = M.v2_removed
    n_all = int(M.stratum_size.groupby(M.stratum).first().sum())
    L.append(
        f"\nPOPULATION (weighted by inverse sampling probability, all {n_all} answer concepts): "
        f"P(generic) {np.average(M.generic, weights=w):.3f}; among v2-removed concepts: "
        f"P(generic) {np.average(M[removed].generic, weights=w[removed]):.3f}, "
        f"P(specific) {np.average(M[removed].specific, weights=w[removed]):.3f}; "
        f"among v2-retained: P(generic) {np.average(M[~removed].generic, weights=w[~removed]):.3f}"
    )
    no_ic = s1.ic.isna()
    L.append(
        f"\nS1 by IC availability: no-IC P(generic) {s1[no_ic].generic.mean():.3f} "
        f"(n={int(no_ic.sum())}); with IC {s1[~no_ic].generic.mean():.3f} (n={int((~no_ic).sum())})"
    )
    L.append(f"S1 DrugCentral drugs (n={int(s1.dc_drug.sum())}): " + named(s1[s1.dc_drug]))
    s4 = M[M.stratum == "S4_near"]
    L.append(
        "S4 near-threshold (0<=v2<1, both retained): "
        f"P(generic) {s4.generic.mean():.3f}, P(specific) {s4.specific.mean():.3f}"
    )
    L.append(
        "\nS1 concepts the panel judged SPECIFIC (>=4) — v2 removals the panel disagrees with:"
    )
    L.append("  " + named(s1[s1.specific].sort_values("llm_mean", ascending=False), with_v2=True))
    L.append("\nS1 concepts the panel judged GENERIC (<=2) — sample of 40:")
    L.append("  " + named(s1[s1.generic].head(40)))
    L.append("\nS5/S6 retained concepts the panel judged GENERIC — v2 misses:")
    retained = M.stratum.isin(["S5_ret_noIC", "S6_ret_IC", "S4_near"]) & M.generic
    L.append("  " + named(M[retained], with_v2=True))
    return L


def main() -> None:
    """Build the audit table, print the report and write the text and JSON summaries."""
    rng = np.random.default_rng(SEED)
    with open(RESULTS / "validation.json") as fh:
        V = json.load(fh)
    judges = [j for j in V["judges"] if j not in V["judges_dropped_by_rule"]]
    M = build_items(judges)
    L, out = stratum_table(M, V, judges, rng)
    L.extend(headline_lines(M, rng))
    txt = "\n".join(L)
    print(txt)
    with open(RESULTS / "audit.txt", "w") as fh:
        fh.write(txt + "\n")
    with open(RESULTS / "audit.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()

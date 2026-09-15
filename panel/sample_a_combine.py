"""Combine Sample A rating responses into one table keyed by node id and print agreement.

The workbook holds one Google Forms export block per rater on any tab: a header row
("Timestamp", "1. <name>", ..., "100. <name>") followed by a rating row ("<Rater>", rating,
...), ratings 1 (generic) to 5 (specific). Each block is joined back to
data/sample_a_sheet_<rater>.csv and data/sample_a_key.csv and written to
panel/sample_a_ratings.csv with one column per rater.
Usage: python sample_a_combine.py <workbook.xlsx> [--exclude RATER ...]
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import PANEL  # noqa: E402

DATA = PANEL / "data"
ITEM_RE = re.compile(r"^\s*(\d+)\.\s*(.*)$", re.S)
RATERS = ("roach", "ramsey", "koslicki", "hodges")

Block = dict[int, tuple[str, float]]


def header_signature(rater: str) -> tuple[str, ...]:
    """Return the first five item names of a rater's frozen sheet, in item order."""
    sheet = pd.read_csv(DATA / f"sample_a_sheet_{rater}.csv")
    return tuple(sheet.sort_values("item")["name"].str.strip().head(5))


def parse_workbook(path: str) -> Iterator[tuple[str, str, Block]]:
    """Yield (rater, tab, {item_number: (name, rating)}) for every header/rating block.

    The rater comes from column A of the rating row, or from the item order (unique per
    rater) when that cell is not a rater name.
    """
    sigs = {header_signature(r): r for r in RATERS}
    x = pd.ExcelFile(path)
    for tab in x.sheet_names:
        df = x.parse(tab, header=None)
        for i in range(len(df) - 1):
            hdr, val = df.iloc[i], df.iloc[i + 1]
            if not (isinstance(hdr[1], str) and ITEM_RE.match(hdr[1])):
                continue
            rater = str(val[0]).strip().lower()
            if rater not in RATERS:
                sig = tuple(ITEM_RE.match(str(h)).group(2).strip() for h in hdr[1:6])
                rater = sigs.get(sig)
                if rater is None:
                    continue
            if not pd.api.types.is_number(val[1]) and not str(val[1]).strip().isdigit():
                continue
            items: Block = {}
            for h, v in zip(hdr[1:], val[1:]):
                m = ITEM_RE.match(str(h)) if isinstance(h, str) else None
                if m and pd.notna(v):
                    items[int(m.group(1))] = (m.group(2).strip(), float(v))
            yield rater, tab, items


def to_ids(rater: str, items: Block) -> dict[str, float]:
    """Map a block's item numbers to node ids through the rater's frozen sheet."""
    sheet = pd.read_csv(DATA / f"sample_a_sheet_{rater}.csv").set_index("item")
    recs = []
    for n, (name, rating) in items.items():
        if n not in sheet.index:
            sys.exit(f"{rater}: form item {n} not in frozen sheet")
        if sheet.loc[n, "name"].strip() != name:
            sys.exit(
                f"{rater}: item {n} name mismatch\n"
                f"  sheet: {sheet.loc[n, 'name']!r}\n"
                f"  form:  {name!r}"
            )
        recs.append((sheet.loc[n, "id"], rating))
    return dict(recs)


def krippendorff_alpha(values: np.ndarray, metric: str = "ordinal") -> float:
    """Krippendorff's alpha for a units x raters array (NaN missing), ordinal or interval."""
    v = np.asarray(values, dtype=float)
    cats = np.unique(v[~np.isnan(v)])
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    o = np.zeros((k, k))
    for row in v:
        r = row[~np.isnan(row)]
        m = len(r)
        if m < 2:
            continue
        for a, b in itertools.permutations(r, 2):
            o[idx[a], idx[b]] += 1.0 / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    if metric == "interval":
        delta = (cats[:, None] - cats[None, :]) ** 2
    else:
        cum = np.cumsum(n_c)
        delta = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                lo, hi = min(i, j), max(i, j)
                delta[i, j] = (cum[hi] - (cum[lo] - n_c[lo]) - (n_c[i] + n_c[j]) / 2) ** 2
    d_o = (o * delta).sum()
    d_e = (np.outer(n_c, n_c) * delta).sum() / (n - 1)
    return 1 - d_o / d_e


def collect(path: str, exclude: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
    """Read every rater block, attach ratings to the key and write sample_a_ratings.csv."""
    key = pd.read_csv(DATA / "sample_a_key.csv")
    out = key[["id", "name", "most_specific_category", "information_content", "ic_band"]].copy()
    seen: dict[str, dict[str, float]] = {}
    for rater, tab, items in parse_workbook(path):
        if rater in exclude:
            print(f"  {rater:9s} excluded from panel (tab {tab!r})")
            continue
        ids = to_ids(rater, items)
        if rater in seen:
            same = seen[rater] == ids
            print(
                f"  {rater:9s} duplicate block in tab {tab!r}: "
                f"{'identical' if same else 'DIFFERS'} -> ignored"
            )
            if not same:
                sys.exit(f"conflicting copies of {rater}'s ratings; resolve in the workbook")
            continue
        seen[rater] = ids
        print(f"  {rater:9s} {len(ids):3d} ratings from tab {tab!r}")
        out[rater] = out["id"].map(ids)
    raters = [r for r in RATERS if r in seen]
    if len(raters) == 0:
        sys.exit("no rater blocks found")
    out["n_raters"] = out[raters].notna().sum(axis=1)
    out["mean_rating"] = out[raters].mean(axis=1)
    dest = PANEL / "sample_a_ratings.csv"
    out.to_csv(dest, index=False)
    print(
        f"\nwrote {dest}: {len(out)} concepts x {len(raters)} raters; "
        f"{(out.n_raters < len(raters)).sum()} concepts missing a rating"
    )
    return out, raters


def report(out: pd.DataFrame, raters: list[str]) -> None:
    """Print rating distributions, inter-rater agreement and IC versus rating summaries."""
    print("\nrating distribution (1=generic .. 5=specific):")
    dist = {r: out[r].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0) for r in raters}
    print(pd.DataFrame(dist).T.to_string())

    if len(raters) > 1:
        print("\npairwise Spearman:")
        print(out[raters].corr(method="spearman").round(3).to_string())
        print("\npairwise exact agreement / within 1:")
        for a, b in itertools.combinations(raters, 2):
            d = (out[a] - out[b]).abs()
            print(
                f"  {a:9s} {b:9s} exact={np.mean(d == 0):.2f}  "
                f"within1={np.mean(d <= 1):.2f}  mean|diff|={d.mean():.2f}"
            )
        vals = out[raters].to_numpy()
        print(
            f"\nKrippendorff alpha: ordinal={krippendorff_alpha(vals, 'ordinal'):.3f}  "
            f"interval={krippendorff_alpha(vals, 'interval'):.3f}"
        )

    ic = out.dropna(subset=["information_content"])
    rho, p = spearmanr(ic["information_content"], ic["mean_rating"])
    tau, _ = kendalltau(ic["information_content"], ic["mean_rating"])
    print(
        f"\nIC vs mean rating (n={len(ic)}): Spearman rho={rho:.3f} (p={p:.2g}), "
        f"Kendall tau={tau:.3f}"
    )
    for r in raters:
        rr, _ = spearmanr(ic["information_content"], ic[r])
        print(f"  IC vs {r:9s} rho={rr:.3f}")
    print("\nmean rating by IC band:")
    by_band = out.groupby("ic_band", observed=True)["mean_rating"]
    print(by_band.agg(["count", "mean", "std"]).round(2).to_string())
    print("\nmean rating by category:")
    by_cat = out.groupby("most_specific_category")["mean_rating"].agg(["count", "mean"])
    print(by_cat.round(2).sort_values("count", ascending=False).to_string())


def main(path: str, exclude: Sequence[str] = ()) -> None:
    """Combine the workbook's rater blocks and print the agreement report."""
    out, raters = collect(path, exclude)
    report(out, raters)


def parse_args() -> argparse.Namespace:
    """Parse the workbook path and the optional list of raters to exclude."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("workbook")
    ap.add_argument(
        "--exclude", nargs="*", default=[], help="rater names to leave out of the panel"
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.workbook, exclude=[r.lower() for r in args.exclude])

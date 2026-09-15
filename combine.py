"""Assemble candidate scores into data/scores.parquet for report_tables.py and seal_sample_a.py.

All columns are IC-oriented (higher = more specific): ic_lex and ic_sem from
semantic_ic.py, ic_axis from generality_axis.py, ic_rankmean (mean percentile
rank of ic_sem and ic_axis), ic_zsum (z(ic_axis) + z(-log1p(sem_desc))),
ic_tiebreak (ic_sem with ties broken by ic_axis scaled below one IC unit) and
ic_zsum_p (ic_zsum minus the plural-head marker). The plural-head, hubness and
relational-axis columns are added only when their parquet files exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from paths import DATA


def pct(s: pd.Series) -> pd.Series:
    """Percentile rank."""
    return s.rank(pct=True)


def z(s: pd.Series) -> pd.Series:
    """Standardise to zero mean and unit standard deviation."""
    return (s - s.mean()) / s.std()


def main() -> None:
    """Merge the component scores, derive the combinations and write scores.parquet."""
    sem = pd.read_parquet(DATA / "semantic_ic.parquet")
    axis = pd.read_parquet(DATA / "generality_axis.parquet")
    df = sem.merge(axis, on="id")
    extra = [c for c in ("ic_hub",) if (DATA / "hubness.parquet").exists()]
    if extra:
        hub = pd.read_parquet(DATA / "hubness.parquet")
        df = df.merge(hub, on="id", how="left")

    rel_path = DATA / "relational_axis.parquet"
    if rel_path.exists():
        df = df.merge(
            pd.read_parquet(rel_path, columns=["id", "ic_rel_only", "ic_emb_rel"]),
            on="id",
            how="left",
        )
    plural_path = DATA / "plural_head.parquet"
    has_plural = plural_path.exists()
    if has_plural:
        df = df.merge(pd.read_parquet(plural_path), on="id", how="left")
        df["plural"] = df["plural"].fillna(False).astype(float)

    df["ic_rankmean"] = 100 * (pct(df.ic_sem) + pct(df.ic_axis)) / 2
    df["ic_zsum"] = z(df.ic_axis) + z(-np.log1p(df.sem_desc))
    a = df.ic_axis
    df["ic_tiebreak"] = df.ic_sem + 0.99 * (a - a.min()) / (a.max() - a.min())
    if has_plural:
        df["ic_zsum_p"] = df.ic_zsum - 1.0 * df.plural
    if extra:
        df["ic_zsum3"] = df.ic_zsum + z(df.ic_hub)
        if has_plural:
            df["ic_zsum3_p"] = df.ic_zsum3 - 1.0 * df.plural
    if "ic_emb_rel" in df:
        df["ic_zsum_rel"] = z(df.ic_emb_rel) + z(-np.log1p(df.sem_desc))
    cols = [
        "id",
        "ic_lex",
        "ic_sem",
        "ic_axis",
        "ic_rankmean",
        "ic_zsum",
        "ic_tiebreak",
        *(["ic_zsum_p"] if has_plural else []),
        *(["ic_hub", "ic_zsum3"] if extra else []),
        *(["ic_zsum3_p"] if extra and has_plural else []),
        *(["ic_rel_only", "ic_emb_rel", "ic_zsum_rel"] if "ic_emb_rel" in df else []),
    ]
    df[cols].to_parquet(DATA / "scores.parquet", index=False)
    print("wrote", cols)


if __name__ == "__main__":
    main()

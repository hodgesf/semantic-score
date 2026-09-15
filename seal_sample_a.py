"""Score the frozen Sample A items with every candidate score.

Reads panel/data/sample_a_key.csv and data/scores.parquet; writes
results/sample_a_scores_SEALED.csv, which is held back from the panel ratings
until the panel is complete.
"""

from __future__ import annotations

import pandas as pd

from paths import DATA, PANEL, RESULTS

KEY = PANEL / "data" / "sample_a_key.csv"
OUT = RESULTS / "sample_a_scores_SEALED.csv"


def main() -> None:
    """Merge the Sample A key with the score table and write the sealed CSV."""
    key = pd.read_csv(KEY)
    scores = pd.read_parquet(DATA / "scores.parquet")
    merged = key.merge(scores, on="id", how="left")
    missing = merged[scores.columns[1]].isna().sum()
    merged.to_csv(OUT, index=False)
    print(f"sealed {len(merged)} Sample A rows ({missing} unscored) -> {OUT}")
    print(merged.columns.tolist())


if __name__ == "__main__":
    main()

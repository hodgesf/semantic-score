"""Assemble the expert-rated items the LLM judges are validated on.

Sample A rows come from panel/sample_a_ratings.csv with category and description from
panel/data/sample_a_key.csv; Sample B rows come from sample_b/results/sample_b_panel_scores.csv
with descriptions from sample_b/data/sample_b_key.csv. Judges see name, category and description;
the panel mean, per-rater ratings and IC are kept for the validation analysis.
Output: llm_panel/data/expert_items.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import LLM_PANEL, PANEL, SAMPLE_B  # noqa: E402

COLUMNS = ["sample", "id", "name", "category", "description", "panel"]
RATERS = ["roach", "ramsey", "koslicki"]


def sample_a() -> pd.DataFrame:
    """Sample A items in rating-table order."""
    ratings = pd.read_csv(PANEL / "sample_a_ratings.csv")
    key = pd.read_csv(PANEL / "data" / "sample_a_key.csv").set_index("id")
    return ratings.assign(
        sample="A",
        category=key.most_specific_category.reindex(ratings.id).to_numpy(),
        description=key.description.reindex(ratings.id).to_numpy(),
        panel=ratings.mean_rating,
        ic=ratings.information_content,
    )[[*COLUMNS, *RATERS, "ic"]]


def sample_b() -> pd.DataFrame:
    """Sample B items in panel-score-table order; these nodes carry no IC."""
    scores = pd.read_csv(SAMPLE_B / "results" / "sample_b_panel_scores.csv")
    key = pd.read_csv(SAMPLE_B / "data" / "sample_b_key.csv").set_index("id")
    return scores.assign(
        sample="B",
        category=scores.most_specific_category,
        description=key.description.reindex(scores.id).to_numpy(),
        ic=float("nan"),
    )[[*COLUMNS, *RATERS, "ic"]]


def main() -> None:
    """Write the combined item table."""
    out = pd.concat([sample_a(), sample_b()], ignore_index=True)
    dest = LLM_PANEL / "data" / "expert_items.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"wrote {len(out)} items to {dest}")


if __name__ == "__main__":
    main()

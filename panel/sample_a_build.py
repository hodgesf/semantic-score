"""Draw Sample A, the expert-panel validation set of nodes that carry curated IC.

Reads the node table and the curated and LLM label files, excludes every node used in
training, and draws PER_BAND nodes per IC band with a per-band category cap. Writes
data/sample_a_key.csv and one blinded sheet per rater, data/sample_a_sheet_<rater>.csv, each
in a per-rater shuffled order. If the key file exists the sample is frozen and only the
sheets are regenerated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, PANEL  # noqa: E402

PANEL_DATA = PANEL / "data"
NODES_TABLE = PANEL_DATA / "nodes_table.parquet"
POSITIVES = DATA / "positive_generic.parquet"
CONFIRMED = DATA / "llm_confirmed_generics.txt"
HARD_NEG = DATA / "llm_hard_negatives.txt"
KEY_FILE = PANEL_DATA / "sample_a_key.csv"
SHEET_FILE = str(PANEL_DATA / "sample_a_sheet_{rater}.csv")

RATERS = ("roach", "ramsey", "koslicki", "hodges")
IC_BANDS = (0.0, 50.0, 66.0, 88.0, 95.0, 100.0)
PER_BAND = 20
CATEGORY_CAP = 3
SEED = 2654


def read_ids(path: Path) -> set[str]:
    """Read one node id per line; a missing file is an error."""
    with open(path, encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def firewall(nodes: pd.DataFrame) -> pd.DataFrame:
    """Drop every node the training pipeline has touched, printing the overlap per source."""
    excluded = {
        "curated positives": set(pd.read_parquet(POSITIVES, columns=["id"])["id"]),
        "LLM confirmed generics": read_ids(CONFIRMED),
        "LLM hard negatives": read_ids(HARD_NEG),
    }

    print("firewall overlap with the IC-labelled pool:")
    keep = np.ones(len(nodes), dtype=bool)
    for source, ids in excluded.items():
        member = nodes["id"].isin(ids).to_numpy()
        print(f"  {source:<24}{len(ids):>7,} ids   {int(member.sum()):>6,} in pool")
        keep &= ~member

    print(f"pool: {len(nodes):,} labelled, {int(keep.sum()):,} eligible")
    return nodes[keep]


def name_tokens(name: str) -> set[str]:
    """Return the lower-cased tokens of a name, with commas treated as spaces."""
    return set(name.lower().replace(",", " ").split())


def near_duplicate(name: str, taken_tokens: list[set[str]]) -> bool:
    """Return True if the name's tokens contain or are contained by an item already drawn."""
    tokens = name_tokens(name)
    return any(tokens <= other or other <= tokens for other in taken_tokens)


def draw_band(band: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Draw PER_BAND rows from one IC band in a single shuffled pass with a category cap.

    While the band is short the cap is raised one step and the leftovers rescanned; the
    near-duplicate guard never relaxes.
    """
    shuffled = band.iloc[rng.permutation(len(band))]

    taken: list[int] = []
    taken_tokens: list[set[str]] = []
    counts: dict[str, int] = {}
    cap = CATEGORY_CAP
    remaining = shuffled
    while len(taken) < PER_BAND and len(remaining):
        leftovers = []
        for position, category, name in zip(
            remaining.index, remaining["most_specific_category"], remaining["name"]
        ):
            if len(taken) == PER_BAND:
                break
            if near_duplicate(name, taken_tokens):
                continue
            if counts.get(category, 0) >= cap:
                leftovers.append(position)
                continue
            counts[category] = counts.get(category, 0) + 1
            taken.append(position)
            taken_tokens.append(name_tokens(name))
        remaining = shuffled.loc[leftovers]
        cap += 1

    return band.loc[taken]


def write_sheets(sample: pd.DataFrame) -> None:
    """Write one blinded rating sheet per rater, shuffled with a rater-specific seed."""
    visible = sample[["id", "name", "description", "most_specific_category"]].copy()
    visible["category"] = visible.pop("most_specific_category").str.removeprefix("biolink:")
    visible["rating"] = ""
    visible["rationale"] = ""

    for rater in RATERS:
        rng = np.random.default_rng([SEED, *rater.encode()])
        sheet = visible.iloc[rng.permutation(len(visible))].copy()
        sheet.insert(0, "item", range(1, len(sheet) + 1))
        path = SHEET_FILE.format(rater=rater)
        sheet.to_csv(path, index=False)
        print(f"wrote {path}")


def print_summary(sample: pd.DataFrame) -> None:
    """Print per-band counts and category totals of the drawn sample."""
    header = f"{'ic band':<16}{'n':>4}{'categories':>12}{'described':>11}"
    print(f"\n{header}\n{'-' * len(header)}")
    for band, group in sample.groupby("ic_band"):
        print(
            f"{band:<16}{len(group):>4}"
            f"{group['most_specific_category'].nunique():>12}"
            f"{group['description'].notna().mean():>11.0%}"
        )

    print("\ncategory totals:")
    totals = sample["most_specific_category"].value_counts()
    for category, count in totals.items():
        print(f"  {category.removeprefix('biolink:'):<28}{count:>4}")


def main() -> None:
    """Draw the sample unless the key file exists, then write the rating sheets."""
    if KEY_FILE.exists():
        print(
            f"{KEY_FILE} exists; Sample A is frozen. Regenerating "
            "rating sheets from it without redrawing. Delete the key "
            "file only if no rater has seen a sheet."
        )
        write_sheets(pd.read_csv(KEY_FILE))
        return

    nodes = pd.read_parquet(
        NODES_TABLE,
        columns=["id", "name", "description", "most_specific_category", "information_content"],
    )
    nodes = firewall(nodes[nodes["information_content"].notna()])

    rng = np.random.default_rng(SEED)
    bands = pd.cut(nodes["information_content"], bins=IC_BANDS, include_lowest=True)
    sample = pd.concat([draw_band(group, rng) for _, group in nodes.groupby(bands, observed=True)])

    sample = sample.copy()
    sample["ic_band"] = pd.cut(
        sample["information_content"], bins=IC_BANDS, include_lowest=True
    ).astype(str)

    print_summary(sample)

    sample.to_csv(KEY_FILE, index=False)
    print(f"\nwrote {len(sample)} nodes to {KEY_FILE}")
    write_sheets(sample)


if __name__ == "__main__":
    main()

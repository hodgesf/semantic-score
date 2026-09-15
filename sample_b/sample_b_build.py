"""Draw Sample B, 50 tier-0 nodes without curated IC, stratified by Biolink category.

Nodes are drawn uniformly at random within each category of ALLOCATION after excluding every
label set, the v1 training set and Sample A, with a token-containment near-duplicate guard
against Sample A names and earlier Sample B names; no model score is consulted. Writes
data/sample_b_key.csv, data/sample_b_sheet_<rater>.csv and data/sample_b_forms.gs (same
rubric text as Sample A); if the key exists only the sheets and forms are regenerated.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panel.sample_a_forms_build import MAX_DESCRIPTION, RUBRIC, TEMPLATE  # noqa: E402
from paths import DATA, PANEL, SAMPLE_B  # noqa: E402

SEED = 2655
RATERS = ("roach", "ramsey", "koslicki")
OUT = SAMPLE_B / "data"
KEY_FILE = OUT / "sample_b_key.csv"
SHEET_FILE = str(OUT / "sample_b_sheet_{rater}.csv")
SCRIPT_FILE = OUT / "sample_b_forms.gs"
SAMPLE_A_KEY = PANEL / "data" / "sample_a_key.csv"
ALLOCATION = {
    "biolink:Drug": 8,
    "biolink:ChemicalEntity": 7,
    "biolink:SmallMolecule": 5,
    "biolink:Disease": 5,
    "biolink:Pathway": 5,
    "biolink:GeneFamily": 5,
    "biolink:PhenotypicFeature": 4,
    "biolink:Procedure": 4,
    "biolink:Protein": 3,
    "biolink:MolecularMixture": 2,
    "biolink:AnatomicalEntity": 2,
}
assert sum(ALLOCATION.values()) == 50
TOKEN = re.compile(r"[a-z0-9]+")
LABEL_FILES = (
    "llm_confirmed_generics.txt",
    "llm_hard_negatives.txt",
    "confirmed_generics.txt",
    "hard_negatives.txt",
)


def tokens(name: str) -> set[str]:
    """Return the lower-cased alphanumeric tokens of a name."""
    return set(TOKEN.findall(name.lower()))


def near_duplicate(name: str, taken: list[set[str]]) -> bool:
    """Return True if the name's tokens contain or are contained by a taken name's tokens."""
    t = tokens(name)
    return any(t <= u or u <= t for u in taken if u and t)


def read_ids(path: Path) -> set[str]:
    """Read one node id per line."""
    with open(path) as fh:
        return {line.strip() for line in fh if line.strip()}


def firewall_ids() -> set[str]:
    """Return every id in the curated positives, the label files or the v1 training set."""
    ids = set(pd.read_parquet(DATA / "positive_generic.parquet")["id"])
    for fn in LABEL_FILES:
        ids |= read_ids(DATA / fn)
    ids |= set(pd.read_parquet(DATA / "training_set_v1.parquet")["id"])
    return ids


def draw() -> pd.DataFrame:
    """Draw the sample per ALLOCATION from the firewalled no-IC pool and write the key."""
    rng = np.random.default_rng(SEED)
    nodes = pd.read_parquet(
        DATA / "nodes.parquet",
        columns=["id", "name", "description", "category", "information_content"],
    )
    nodes["category"] = nodes["category"].astype(str)
    no_ic = nodes[nodes.information_content.isna() & nodes.name.ne("")]
    fw = firewall_ids()
    key_a = pd.read_csv(SAMPLE_A_KEY)
    print(f"no-IC pool {len(no_ic):,}; firewall ids {len(fw):,}; Sample A ids {len(key_a)}")
    taken = [tokens(n) for n in key_a["name"]]
    eligible = no_ic[~no_ic.id.isin(fw) & ~no_ic.id.isin(set(key_a.id))]
    print(f"eligible after firewall {len(eligible):,}")

    rows = []
    for cat, n in ALLOCATION.items():
        pool = eligible[eligible.category == cat]
        order = rng.permutation(len(pool))
        got = 0
        for i in order:
            r = pool.iloc[i]
            if near_duplicate(r["name"], taken):
                continue
            taken.append(tokens(r["name"]))
            rows.append(r)
            got += 1
            if got == n:
                break
        print(f"  {cat.split(':')[1]:20s} pool {len(pool):>8,}  drew {got}/{n}")
    key = pd.DataFrame(rows)[["id", "name", "description", "category"]]
    key = key.rename(columns={"category": "most_specific_category"})
    key.insert(0, "item", range(1, len(key) + 1))
    key.to_csv(KEY_FILE, index=False)
    print(f"wrote {KEY_FILE}: {len(key)} nodes")
    return key


def write_sheets(key: pd.DataFrame) -> None:
    """Write one rating sheet per rater, shuffled with a rater-specific seed."""
    for rater in RATERS:
        rng = np.random.default_rng([SEED, *rater.encode()])
        sheet = key.iloc[rng.permutation(len(key))].reset_index(drop=True)
        sheet = sheet.drop(columns=["item"])
        sheet.insert(0, "item", range(1, len(sheet) + 1))
        sheet["rating"] = ""
        sheet.to_csv(SHEET_FILE.format(rater=rater), index=False)


def write_forms() -> None:
    """Emit the Apps Script that builds one Google Form per rater from the sheets."""
    items = {}
    for rater in RATERS:
        sheet = pd.read_csv(SHEET_FILE.format(rater=rater))
        rows = []
        for r in sheet.itertuples():
            d = r.description if isinstance(r.description, str) else ""
            if len(d) > MAX_DESCRIPTION:
                d = d[:MAX_DESCRIPTION].rsplit(" ", 1)[0] + " [...]"
            rows.append(
                {
                    "n": int(r.item),
                    "name": r.name,
                    "category": r.most_specific_category,
                    "description": d,
                }
            )
        items[rater] = rows
    rubric = RUBRIC.replace("rating 100 biomedical", "rating 50 biomedical")
    builders = "\n\n".join(f"function build_{r}() {{ buildForm('{r}'); }}" for r in RATERS)
    calls = "\n".join(f"  build_{r}();" for r in RATERS)
    script = TEMPLATE.format(
        rubric=json.dumps(rubric),
        choices=json.dumps(["1", "2", "3", "4", "5"]),
        items=json.dumps(items, indent=1),
        builders=builders,
        calls=calls,
    )
    script = script.replace(
        "'Generic concept rating - ' + rater", "'Generic concept rating (Sample B) - ' + rater"
    )
    with open(SCRIPT_FILE, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(script)
    print(
        f"wrote {SCRIPT_FILE}: {sum(len(v) for v in items.values())} items "
        f"across {len(RATERS)} forms"
    )


def main() -> None:
    """Draw the sample unless the key exists, then write the sheets and forms."""
    OUT.mkdir(parents=True, exist_ok=True)
    key = pd.read_csv(KEY_FILE) if KEY_FILE.exists() else draw()
    write_sheets(key)
    write_forms()


if __name__ == "__main__":
    main()

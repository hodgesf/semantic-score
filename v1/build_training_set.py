"""Build the v1 training set.

Positives are the curated and LLM-confirmed generics minus REMOVED_POSITIVES and Sample A;
negatives are the LLM hard negatives outside DROP_NEG_CATS, topped up per category with
random unlabelled nodes to NEG_PER_POS negatives per positive. Output:
data/training_set_v1.parquet with columns id, label (1 = generic), category and source
(curated/llm/hardneg/random).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, PANEL  # noqa: E402

SAMPLE_A_KEY = PANEL / "data" / "sample_a_key.csv"
OUT_FILE = DATA / "training_set_v1.parquet"
SEED = 2654
NEG_PER_POS = 3
DROP_NEG_CATS = {
    "biolink:AnatomicalEntity",
    "biolink:GrossAnatomicalStructure",
    "biolink:Cell",
    "biolink:CellularComponent",
}
REMOVED_POSITIVES = [
    "NCBIGene:100196603",
    "NCBIGene:100328933",
    "NCBIGene:1714",
    "NCBIGene:176861",
    "NCBIGene:183998",
    "NCBIGene:2056",
    "NCBIGene:32321",
    "NCBIGene:33427",
    "NCBIGene:36456",
    "NCBIGene:373310",
    "NCBIGene:40675",
    "NCBIGene:582058",
    "NCBIGene:7018",
    "NCBIGene:778733",
    "CHEMBL.COMPOUND:CHEMBL4525089",
    "CHEMBL.COMPOUND:CHEMBL4525090",
    "CHEMBL.COMPOUND:CHEMBL4525112",
    "CHEMBL.COMPOUND:CHEMBL4525114",
    "CHEMBL.COMPOUND:CHEMBL4525123",
    "CHEMBL.COMPOUND:CHEMBL4525139",
    "CHEMBL.COMPOUND:CHEMBL4526634",
    "UMLS:C3501972",
    "PUBCHEM.COMPOUND:168475924",
    "PUBCHEM.COMPOUND:70701942",
    "PUBCHEM.COMPOUND:71463400",
    "PUBCHEM.COMPOUND:71463404",
    "PUBCHEM.COMPOUND:89554759",
    "PUBCHEM.COMPOUND:9934678",
]


def read_ids(path: Path) -> set[str]:
    """Read one node id per line."""
    with open(path) as fh:
        return {line.strip() for line in fh if line.strip()}


def summarize(out: pd.DataFrame) -> None:
    """Print counts by label and source and the negative/positive ratio per category."""
    print(out.groupby(["label", "source"]).size())
    top = out.pivot_table(index="category", columns="label", aggfunc="size", fill_value=0)
    top["ratio"] = top[0] / top[1].clip(lower=1)
    print(top.sort_values(1, ascending=False).head(15))


def main() -> None:
    """Assemble positives and negatives and write the training set."""
    rng = np.random.default_rng(SEED)
    nodes = pd.read_parquet(DATA / "nodes.parquet", columns=["id", "category"])
    nodes["category"] = nodes["category"].astype(str)
    nodes = nodes.set_index("id")

    curated = set(pd.read_parquet(DATA / "positive_generic.parquet")["id"])
    llm_pos = read_ids(DATA / "llm_confirmed_generics.txt")
    hard_neg = read_ids(DATA / "llm_hard_negatives.txt")
    sample_a = set(pd.read_csv(SAMPLE_A_KEY)["id"])

    pos = (curated | llm_pos) - set(REMOVED_POSITIVES) - sample_a
    pos = [i for i in pos if i in nodes.index]
    print(
        f"positives: {len(pos)} "
        f"(removed {len(REMOVED_POSITIVES)} per memo, "
        f"{len((curated | llm_pos) & sample_a)} Sample A overlaps)"
    )

    pos_cat = nodes.loc[pos, "category"]
    hn = pd.Index([i for i in hard_neg if i in nodes.index])
    hn_cat = nodes.loc[hn, "category"]
    dropped_anat = int(hn_cat.isin(DROP_NEG_CATS).sum())
    hn = hn[~hn_cat.isin(DROP_NEG_CATS) & ~hn.isin(sample_a)]
    print(
        f"hard negatives: {len(hard_neg)} -> {len(hn)} "
        f"({dropped_anat} anatomy/cell dropped per memo)"
    )

    labeled = set(pos) | curated | llm_pos | hard_neg | sample_a
    rows = [
        {
            "id": i,
            "label": 1,
            "category": pos_cat[i],
            "source": "curated" if i in curated else "llm",
        }
        for i in pos
    ]
    for cat, n_pos in pos_cat.value_counts().items():
        need = n_pos * NEG_PER_POS
        take_hn = hn[nodes.loc[hn, "category"] == cat]
        take_hn = take_hn[:need] if len(take_hn) > need else take_hn
        rows += [{"id": i, "label": 0, "category": cat, "source": "hardneg"} for i in take_hn]
        need -= len(take_hn)
        if need > 0:
            pool = nodes.index[(nodes["category"] == cat) & ~nodes.index.isin(labeled)]
            take = rng.choice(pool, min(need, len(pool)), replace=False)
            rows += [{"id": i, "label": 0, "category": cat, "source": "random"} for i in take]

    out = pd.DataFrame(rows)
    out.to_parquet(OUT_FILE, index=False)
    print(f"{len(out)} rows -> {OUT_FILE}")
    summarize(out)


if __name__ == "__main__":
    main()

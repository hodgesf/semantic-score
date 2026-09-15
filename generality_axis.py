"""Fit a self-supervised generality axis in SapBERT space and score every node.

The supervision signal is the subsumption relation mined from names and
definitions (data/semantic_parents.parquet): for every (child, parent) pair the
parent is by construction the more generic concept, so one direction w is fitted
such that w . (e_parent - e_child) > 0 (logistic ranking on difference vectors,
no intercept) and every node is scored by g = w . e. Reads data/nodes.parquet,
data/semantic_parents.parquet and the SapBERT node embeddings; writes
data/generality_axis.parquet (id, gen_axis z-scored with higher = more generic,
ic_axis = -gen_axis) and data/generality_axis_w.npy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

from paths import DATA, NODE_EMBEDDINGS, SEED

EMB = NODE_EMBEDDINGS
OUT = DATA / "generality_axis.parquet"
MAX_CHILDREN_PER_PARENT = 30
MAX_PAIRS_PER_CATEGORY = 150_000


def load_pairs(nodes: pd.DataFrame) -> pd.DataFrame:
    """Load the subsumption pairs with node indices and child category attached."""
    idx = pd.Series(np.arange(len(nodes)), index=nodes["id"])
    rel = pd.read_parquet(DATA / "semantic_parents.parquet")
    rel = rel.drop_duplicates(["child", "parent"])
    rel["c"] = idx[rel["child"]].to_numpy()
    rel["p"] = idx[rel["parent"]].to_numpy()
    rel["ccat"] = nodes["category"].astype(str).to_numpy()[rel["c"]]
    return rel


def balance(rel: pd.DataFrame) -> pd.DataFrame:
    """Cap children per parent, then pairs per child category, after a seeded shuffle."""
    rel = rel.sample(frac=1.0, random_state=SEED)
    rel = rel.groupby("p", sort=False).head(MAX_CHILDREN_PER_PARENT)
    return rel.groupby("ccat", sort=False).head(MAX_PAIRS_PER_CATEGORY)


def fit(E: np.ndarray, pairs: pd.DataFrame) -> np.ndarray:
    """Logistic ranking on symmetrised difference vectors without intercept.

    P(parent more generic) = sigma(w . (e_p - e_c)).
    """
    D = E[pairs["p"].to_numpy()] - E[pairs["c"].to_numpy()]
    X = np.vstack([D, -D]).astype(np.float32)
    t = np.r_[np.ones(len(D)), np.zeros(len(D))]
    clf = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        fit_intercept=False,
        max_iter=30,
        random_state=SEED,
        tol=None,
    )
    clf.fit(X, t)
    return clf.coef_[0].astype(np.float64)


def score_all(E: np.ndarray, w: np.ndarray, n: int) -> np.ndarray:
    """Project the first n embeddings onto w in chunks and z-score the result."""
    g = np.empty(n, dtype=np.float32)
    for start in range(0, n, 200_000):
        g[start : start + 200_000] = E[start : start + 200_000] @ w.astype(np.float32)
    return (g - g.mean()) / g.std()


def print_summary(view: pd.DataFrame) -> None:
    """Print the most generic and most specific nodes and per-category means."""
    pd.set_option("display.width", 160)
    print("\nmost generic by axis:")
    print(
        view.sort_values("gen_axis", ascending=False)
        .head(40)[["name", "category", "gen_axis"]]
        .to_string()
    )
    print("\nmost specific by axis:")
    print(view.sort_values("gen_axis").head(15)[["name", "category", "gen_axis"]].to_string())
    print("\ncategory means:")
    print(
        view.groupby("category", observed=True)["gen_axis"]
        .agg(["mean", "std", "size"])
        .sort_values("mean", ascending=False)
        .head(30)
        .to_string()
    )


def main() -> None:
    """Fit the axis with a held-out check by parent, refit on all pairs and score nodes."""
    rng = np.random.default_rng(SEED)
    nodes = pd.read_parquet(DATA / "nodes.parquet", columns=["id", "name", "category"])
    rel = load_pairs(nodes)
    print(
        f"{len(rel):,} subsumption pairs, {rel.p.nunique():,} parents, {rel.c.nunique():,} children"
    )

    rel = balance(rel)
    print(f"after balancing: {len(rel):,} pairs")
    print(rel["ccat"].value_counts().head(12).to_string())

    # split by parent for the held-out pairwise-accuracy check
    parents = rel["p"].unique()
    rng.shuffle(parents)
    test_parents = set(parents[: len(parents) // 5])
    test = rel[rel["p"].isin(test_parents)]
    train = rel[~rel["p"].isin(test_parents)]

    print("loading embeddings", flush=True)
    E = np.load(EMB, mmap_mode="r")
    E = np.ascontiguousarray(E, dtype=np.float32)

    w = fit(E, train)
    Dt = E[test["p"].to_numpy()] - E[test["c"].to_numpy()]
    acc = float((Dt @ w > 0).mean())
    print(
        f"held-out pairwise accuracy (parent scored more generic than "
        f"child, unseen parents): {acc:.3f} on {len(test):,} pairs"
    )
    per_cat = pd.Series(Dt @ w > 0, index=test.index).groupby(test["ccat"]).mean().sort_values()
    print(per_cat.to_string())

    w = fit(E, rel)  # final fit on all pairs
    g = score_all(E, w, len(nodes))
    out = pd.DataFrame({"id": nodes["id"], "gen_axis": g, "ic_axis": -g})
    out.to_parquet(OUT, index=False)
    np.save(DATA / "generality_axis_w.npy", w)

    print_summary(nodes.assign(gen_axis=g))


if __name__ == "__main__":
    main()

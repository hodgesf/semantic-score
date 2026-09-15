"""Axis robustness to the choice of subsumption pairs.

Fits the ranking axis on all pairs, lexical-only pairs, genus-only pairs and two
random halves of the parents, and prints the curated AUC, Spearman rho against
Babel IC and the pairwise agreement of the resulting scores. Reads
data/nodes.parquet, data/semantic_parents.parquet, data/positive_generic.parquet,
data/llm_confirmed_generics.txt and the SapBERT node embeddings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

from paths import DATA, NODE_EMBEDDINGS, SEED

EMB = NODE_EMBEDDINGS


def read_ids(path: str) -> set[str]:
    """Read one identifier per line."""
    with open(path) as handle:
        return {line.strip() for line in handle}


def fit(pairs: pd.DataFrame, E: np.ndarray, n: int) -> tuple[np.ndarray, int]:
    """Balance the pairs, fit the logistic ranking axis and score all n nodes."""
    pairs = (
        pairs.sample(frac=1, random_state=SEED)
        .groupby("p", sort=False)
        .head(30)
        .groupby("ccat", sort=False)
        .head(150_000)
    )
    D = E[pairs.p.to_numpy()] - E[pairs.c.to_numpy()]
    X = np.vstack([D, -D])
    t = np.r_[np.ones(len(D)), np.zeros(len(D))]
    clf = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        fit_intercept=False,
        max_iter=30,
        random_state=SEED,
        tol=None,
    ).fit(X, t)
    w = clf.coef_[0].astype(np.float32)
    g = np.empty(n, np.float32)
    for s in range(0, n, 200_000):
        g[s : s + 200_000] = E[s : s + 200_000] @ w
    return g, len(pairs)


def main() -> None:
    """Run the pair ablations and print the comparison lines."""
    nodes = pd.read_parquet(
        DATA / "nodes.parquet", columns=["id", "category", "information_content"]
    )
    nodes["category"] = nodes.category.astype(str)
    n = len(nodes)
    idx = pd.Series(np.arange(n), index=nodes.id)
    rel = pd.read_parquet(DATA / "semantic_parents.parquet").drop_duplicates(
        ["child", "parent", "source"]
    )
    rel["c"] = idx[rel.child].to_numpy()
    rel["p"] = idx[rel.parent].to_numpy()
    rel["ccat"] = nodes.category.to_numpy()[rel.c]
    pos = set(pd.read_parquet(DATA / "positive_generic.parquet").id)
    nodes["curated"] = nodes.id.isin(pos)
    llm_pos = read_ids(str(DATA / "llm_confirmed_generics.txt"))
    rng = np.random.default_rng(SEED)
    neg: list[int] = []
    for cat, k in nodes[nodes.curated].category.value_counts().items():
        pool = nodes.index[(nodes.category == cat) & ~nodes.curated & ~nodes.id.isin(llm_pos)]
        neg.extend(rng.choice(pool, min(len(pool), 20 * k), replace=False))
    ev = np.r_[nodes.index[nodes.curated].to_numpy(), np.asarray(neg)]
    y = np.r_[np.ones(nodes.curated.sum()), np.zeros(len(neg))]
    icm = nodes.information_content.notna().to_numpy()
    icv = nodes.information_content.to_numpy()[icm]
    E = np.ascontiguousarray(np.load(EMB, mmap_mode="r"), dtype=np.float32)

    def report(name: str, sub: pd.DataFrame) -> np.ndarray:
        """Fit on one pair subset, print its evaluation line and return the scores."""
        g, m = fit(sub, E, n)
        print(
            f"{name:13s} pairs={m:>7,}  curated AUC {roc_auc_score(y, g[ev]):.3f}  "
            f"rho(IC) {spearmanr(-g[icm], icv).statistic:.3f}"
        )
        return g

    scores: dict[str, np.ndarray] = {}
    for name, sub in [
        ("all", rel),
        ("lexical_only", rel[rel.source == "lex"]),
        ("genus_only", rel[rel.source == "genus"]),
    ]:
        scores[name] = report(name, sub)
    parents = rel.p.unique()
    rng.shuffle(parents)
    half = set(parents[: len(parents) // 2])
    for name, sub in [("half_A", rel[rel.p.isin(half)]), ("half_B", rel[~rel.p.isin(half)])]:
        scores[name] = report(name, sub)
    samp = rng.choice(n, 300_000, replace=False)
    print("score agreement (Spearman on 300k random nodes):")
    keys = list(scores)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            rho = spearmanr(scores[keys[i]][samp], scores[keys[j]][samp]).statistic
            print(f"  {keys[i]:13s} vs {keys[j]:13s} {rho:.3f}")


if __name__ == "__main__":
    main()

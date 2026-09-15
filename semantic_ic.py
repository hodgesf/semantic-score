"""Semantics-only information content from names and definitions (no edges, hierarchy or labels).

Genericness is estimated as the number of concepts a concept subsumes. A lexical
link d is-a c holds when the normalised name of d ends with the normalised name of
c on a word boundary, the token before the match is not a preposition or negating
modifier ("regulation OF transport" is not a transport; "NON small cell lung
carcinoma" is not a small cell lung carcinoma) and the Biolink categories are
compatible; a genus link holds when the genus noun phrase of d's definition ("Any
METABOLIC PROCESS that ...") resolves to c exactly or by the same suffix rule.
Transitive closure over the union gives semantic descendants and Seco's formula
ic = 100 * (1 - log(1 + descendants) / log(N)) turns counts into an IC on 0-100.
Reads data/nodes.parquet; writes data/semantic_ic.parquet (per-component counts
and ICs) and data/semantic_parents.parquet (direct parent relation).
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from paths import DATA

NODES = DATA / "nodes.parquet"
OUT = DATA / "semantic_ic.parquet"
PARENTS_OUT = DATA / "semantic_parents.parquet"

# a match whose preceding token is one of these makes c the complement, not
# the head ("disorder OF knee" is not a knee)
BLOCK_BEFORE = {
    "of",
    "to",
    "in",
    "by",
    "from",
    "with",
    "for",
    "into",
    "on",
    "via",
    "and",
    "or",
    "versus",
    "per",
    "at",
    "as",
    "than",
    "without",
    "non",
    "not",
    "anti",
    "pseudo",
    "un",
    "de",
    "pro",
    "pre",
    "ex",
    "sub",
    "vs",
    "the",
    "a",
    "an",
    "containing",
    "related",
    "associated",
    "induced",
    "dependent",
    "like",
    "type",
    "mediated",
    "binding",
    "specific",
    "free",
    "rich",
    "deficient",
}
TRAIL_PAREN = re.compile(r"\s*\([^()]*\)\s*$")
SPLIT = re.compile(r"[\s\-_/,;:\\]+")
CLEAN = re.compile(r"[^0-9a-zͰ-ϿÀ-ɏ+']")
GENUS = re.compile(
    r"^(?:a|an|any|the|one of the|a type of|a kind of|a class of|a group of|"
    r"a form of|a member of the|a family of|a set of|a group of|any of the)\s+"
    r"(.+?)\s+(?:that|which|who|whose|characterized|characterised|involving|"
    r"involved|in which|resulting|consisting|composed|defined|where|with|"
    r"produced|derived|found|located|occurring|formed|containing|having|"
    r"encoded|belonging|used|capable|responsible|comprising|caused|affecting|"
    r"present|obtained|isolated|based|whereby|wherein|arising|made|resulting|"
    r"due|associated|related|from|of|for|in|by|at|on|to|expressed|such|"
    r"surrounding|lining|connecting|forming|covering|extending|lying|serving|"
    r"acting|functioning|providing|representing|constituting|enclosing|"
    r"bounded|situated|attached|extending|carrying|bearing|lacking|showing|"
    r"exhibiting|displaying|marked|manifested|accompanied|mediated|"
    r"generated|secreted|released|synthesized|synthesised|catalyzing|"
    r"catalysing|regulating|controlling|modulating|transporting|binding|"
    r"including|comprised|referring|describing|denoting|designating|"
    r"indicating|measuring|assessing|performed|carried|done|undertaken)\b"
)
NESTED_OF = re.compile(
    r"^(?:class|group|family|type|kind|set|member|members|"
    r"subclass|subgroup|category|series|range|variety)"
    r"\s+of\s+(?:the\s+)?"
)
DESC_SENTENCE = re.compile(r"(?<=[.!?])\s")
WORD = re.compile(r"[a-zͰ-ϿÀ-ɏ]{3,}")
# "the conjugate base OF X", "an ester OF X": the complement is the hypernym
REL_NOUN = (
    r"(?:conjugate base|conjugate acid|enantiomer|salt|hydrochloride|"
    r"derivative|isomer|stereoisomer|analog|analogue|form|subtype|"
    r"variant|isoform|ortholog|orthologue|homolog|member|subunit|"
    r"fragment|precursor|metabolite|ester|ether|anion|cation|"
    r"zwitterion|tautomer|hydrate|solvate|polymer|oligomer|dimer|"
    r"mixture|class|group|type|kind|family|superfamily|subfamily|"
    r"species|strain|genus|serotype|complex|product|portion|part|"
    r"region|segment|branch|layer|surface|wall|lumen|cavity)"
)
GENUS_OF = re.compile(
    r"^(?:a|an|the|any)\s+(?:[\w\-\(\)\+,']+\s+){0,2}?(?:[\(\)\+\-,']+)?"
    + REL_NOUN
    + r"\s+of\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:\s+(?:that|which|in which|"
    r"where|with|having|containing|obtained|formed|produced|derived|"
    r"resulting|found|located|arising|comprising|consisting|characterized|"
    r"characterised|used|and|or|whose|at|by|from|for|to|on|via|through)\b|"
    r"[.,;:(]|$)"
)
MESH_INVERTED = re.compile(r"^([^,\d()]+), ([^,\d()]+)$")


def singular(tok: str) -> str:
    """Return a crude singular form of an English token."""
    if len(tok) <= 3 or not tok.isalpha():
        return tok
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith(("ses", "xes", "zes", "ches", "shes")):
        return tok[:-2]
    if tok.endswith("ss") or tok.endswith("us") or tok.endswith("is"):
        return tok
    if tok.endswith("s"):
        return tok[:-1]
    return tok


def real_word(tok: str) -> bool:
    """True when the token contains at least three consecutive letters."""
    return bool(WORD.search(tok))


def norm_tokens(text: str) -> tuple[str, ...]:
    """Normalise a name to lower-case tokens with the head token singularised."""
    t = text.lower().strip()
    m = MESH_INVERTED.match(t)  # "receptors, cytoplasmic"
    if m:
        t = f"{m.group(2).strip()} {m.group(1).strip()}"
    while True:
        t2 = TRAIL_PAREN.sub("", t)
        if t2 == t or not t2:
            break
        t = t2
    toks = [CLEAN.sub("", x).strip("'+") for x in SPLIT.split(t)]
    toks = [x for x in toks if x]
    if toks:
        toks[-1] = singular(toks[-1])
    return tuple(toks)


def genus_tokens(description: str | None) -> tuple[str, ...]:
    """Normalised tokens of the genus phrase in a definition's first sentence, or ()."""
    if not description:
        return ()
    first = DESC_SENTENCE.split(description.strip(), maxsplit=1)[0].lower()
    m = GENUS_OF.match(first)
    if not m:
        m = GENUS.match(first)
    if not m:
        return ()
    g = m.group(1)
    while True:
        g2 = NESTED_OF.sub("", g)
        if g2 == g:
            break
        g = g2
    g = re.sub(
        r"\b(?:very|highly|large|small|major|common|rare|specific|"
        r"particular|certain|distinct|unique|generic|general)\b",
        " ",
        g,
    )
    return norm_tokens(g)


def compatible(cat_a: str, lin_a: set[str], cat_b: str, lin_b: set[str]) -> bool:
    """True when either category lies in the other's Biolink lineage."""
    return cat_a in lin_b or cat_b in lin_a


def main() -> None:
    """Mine lexical and genus parents, close them transitively and write the ICs."""
    nodes = pd.read_parquet(NODES, columns=["id", "name", "description", "category", "lineage"])
    n = len(nodes)
    ids = nodes["id"].to_numpy()
    cats = nodes["category"].astype(str).to_numpy()
    lineages = [set(x.split("|")) if x else set() for x in nodes["lineage"]]

    print("normalising names", flush=True)
    toks = [norm_tokens(x) for x in nodes["name"]]
    by_name: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for i, t in enumerate(toks):
        if t:
            by_name[t].append(i)
    print(f"  {len(by_name):,} distinct normalised names", flush=True)

    def resolve_suffix(t: tuple[str, ...], self_idx: int, start: int) -> list[int]:
        """Node indices whose name is a proper suffix of t (all lengths)."""
        hits = []
        for k in range(start, len(t)):
            suf = t[k:]
            if t[k - 1] in BLOCK_BEFORE or not real_word(t[k - 1]):
                continue
            if not real_word(suf[-1]):
                continue
            for j in by_name.get(suf, ()):
                if j != self_idx and compatible(
                    cats[self_idx], lineages[self_idx], cats[j], lineages[j]
                ):
                    hits.append(j)
        return hits

    print("lexical hyponymy", flush=True)
    lex_parents: list[list[int]] = [[] for _ in range(n)]
    for i, t in enumerate(toks):
        if len(t) > 1:
            lex_parents[i] = resolve_suffix(t, i, 1)
        if i % 250_000 == 0:
            print(f"  {i:,}", flush=True)

    print("definition genus", flush=True)
    gen_parents: list[list[int]] = [[] for _ in range(n)]
    genus_found = np.zeros(n, dtype=bool)
    genus_resolved = np.zeros(n, dtype=bool)
    self_genus = np.zeros(n, dtype=bool)
    descs = nodes["description"].to_numpy()
    for i in range(n):
        g = genus_tokens(descs[i])
        if not g:
            continue
        genus_found[i] = True
        if g == toks[i]:
            self_genus[i] = True
        hits = [
            j
            for j in by_name.get(g, ())
            if j != i and compatible(cats[i], lineages[i], cats[j], lineages[j])
        ]
        if not hits and len(g) > 1:
            hits = resolve_suffix(g, i, 1)
        if hits:
            genus_resolved[i] = True
            gen_parents[i] = hits
        if i % 250_000 == 0:
            print(f"  {i:,}", flush=True)

    def closure_counts(
        parents: list[list[int]], label: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Descendant counts via memoised ancestor sets (cycle-safe)."""
        print(f"closure: {label}", flush=True)
        anc: list[frozenset[int] | None] = [None] * n
        empty: frozenset[int] = frozenset()
        sys.setrecursionlimit(10_000)
        counts = np.zeros(n, dtype=np.int64)
        same_cat = np.zeros(n, dtype=np.int64)
        for i in range(n):
            if anc[i] is not None:
                continue
            # iterative DFS
            stack = [(i, 0)]
            inprog = {i}
            while stack:
                v, state = stack[-1]
                if state == 0:
                    stack[-1] = (v, 1)
                    for p in parents[v]:
                        if anc[p] is None and p not in inprog:
                            inprog.add(p)
                            stack.append((p, 0))
                else:
                    s = set()
                    for p in parents[v]:
                        if anc[p] is None:  # cycle edge: ignore
                            continue
                        s.add(p)
                        s |= anc[p]
                    anc[v] = frozenset(s) if s else empty
                    inprog.discard(v)
                    stack.pop()
        flat = []
        flat_same = []
        for i in range(n):
            a = anc[i]
            if a:
                flat.extend(a)
                ci = cats[i]
                flat_same.extend(j for j in a if cats[j] == ci)
        counts = np.bincount(np.asarray(flat, dtype=np.int64), minlength=n)
        same_cat = np.bincount(np.asarray(flat_same, dtype=np.int64), minlength=n)
        n_parents = np.fromiter((len(x) for x in anc), dtype=np.int64, count=n)
        return counts, same_cat, n_parents

    lex_desc, lex_same, lex_anc = closure_counts(lex_parents, "lexical")
    union = [sorted(set(a) | set(b)) for a, b in zip(lex_parents, gen_parents)]
    sem_desc, sem_same, sem_anc = closure_counts(union, "lexical+genus")
    gen_direct = np.bincount(
        np.asarray([p for ps in gen_parents for p in ps], dtype=np.int64), minlength=n
    )
    lex_direct = np.bincount(
        np.asarray([p for ps in lex_parents for p in ps], dtype=np.int64), minlength=n
    )

    logN = np.log(n)
    out = pd.DataFrame(
        {
            "id": ids,
            "n_tokens": [len(t) for t in toks],
            "lex_direct": lex_direct,
            "lex_desc": lex_desc,
            "lex_same": lex_same,
            "lex_ancestors": lex_anc,
            "genus_found": genus_found,
            "genus_resolved": genus_resolved,
            "self_genus": self_genus,
            "genus_children": gen_direct,
            "sem_desc": sem_desc,
            "sem_same": sem_same,
            "sem_ancestors": sem_anc,
            "ic_lex": 100.0 * (1.0 - np.log1p(lex_desc) / logN),
            "ic_sem": 100.0 * (1.0 - np.log1p(sem_desc) / logN),
        }
    )
    out.to_parquet(OUT, index=False)

    # keep the direct parent relation for error analysis
    rel = [(ids[i], ids[p], "lex") for i, ps in enumerate(lex_parents) for p in ps]
    rel += [(ids[i], ids[p], "genus") for i, ps in enumerate(gen_parents) for p in ps]
    pd.DataFrame(rel, columns=["child", "parent", "source"]).to_parquet(PARENTS_OUT, index=False)

    print(out.describe().T)
    print(
        "genus found",
        genus_found.mean(),
        "resolved",
        genus_resolved.mean(),
        "self",
        self_genus.sum(),
    )
    top = out.merge(nodes[["id", "name", "category"]], on="id")
    print(
        top.sort_values("sem_desc", ascending=False)
        .head(40)[["name", "category", "lex_desc", "genus_children", "sem_desc", "ic_sem"]]
        .to_string()
    )


if __name__ == "__main__":
    main()

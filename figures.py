"""Manuscript figures to Frontiers specifications.

Reads data/nodes.parquet, results/sample_a_scores_SEALED.csv, the v2 scores and query-answer
features in v2/, sample_b/results/sample_b_panel_scores.csv and
rq3/results_wide_v1lt0_iclt50/removed_blocklist.csv. Writes fig1_coverage, fig2_validation,
fig3_bands, fig4_filtering and figS1_sample_b as PDF and 300 dpi RGB LZW TIFF (180 mm
two-column width, text at least 8 pt, data lines at least 2 pt, no red/green indicators).

Usage: python figures.py [<output directory>]   (default: results/figures)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image
from scipy.stats import spearmanr

from paths import DATA, PANEL, RESULTS, RQ3, SAMPLE_B, SEED, V2

matplotlib.use("Agg")

FIGS = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
C = {
    "ic": "#C46A1F",
    "score": "#2F6DB5",
    "hand": "#8E4B8B",
    "axis": "#8E4B8B",
    "gray": "#4B5563",
    "grid": "#E5E7EB",
    "ink": "#1F2937",
    "fill": "#D6E4F5",
}
W2 = 180 / 25.4
W1 = 85 / 25.4
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.titlesize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "axes.edgecolor": C["gray"],
        "xtick.color": C["gray"],
        "ytick.color": C["gray"],
        "axes.grid": True,
        "grid.color": C["grid"],
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
        "legend.frameon": False,
        "text.color": C["ink"],
        "axes.labelcolor": C["ink"],
        "font.family": "DejaVu Sans",
    }
)
BANDS = [-1, 50, 66, 88, 95, 100.01]
BL = ["0 to 50", "50 to 66", "66 to 88", "88 to 95", "95 to 100"]
ANN_IC = [
    ("toxin sequestering activity", "toxin sequestering", -118, -22),
    ("regulation of cell budding", "regulation of cell budding", -44, -42),
    ("presenilin-1", "presenilin-1", -14, 9),
    ("GABA secretion,\nneurotransmission", "gamma-aminobutyric acid secretion", 10, -44),
]
ANN_V2 = [
    ("glutamate GPCR", "glutamate G-protein", -84, 12),
    ("intestinal epithelium", "intestinal epithelium", -14, -30),
    ("beta-Alanine", "beta-Alanine", -60, 10),
    ("onium betaine", "onium betaine", -96, -8),
    ("zinc finger protein ham-2", "zinc finger protein ham-2", -128, 20),
]
# (concept name, label text, x offset, y offset in points)
SAMPLE_B_LABELS = [
    ("Uncertainty", "Uncertainty", 10, 9),
    ("outcomes otolaryngology hearing", "outcomes otolaryngology hearing", 10, 7),
    (
        "ZINC FINGER AND SCAN DOMAIN-CONTAINING",
        "ZINC FINGER AND SCAN\nDOMAIN-CONTAINING",
        -11,
        -14,
    ),
    ("Epistatus", "Epistatus", 9, 6),
]


def save(fig: Figure, name: str, width: float = W2, relayout: bool = True) -> None:
    """Write the figure as PDF and 300 dpi RGB LZW TIFF at exactly ``width`` inches.

    ``bbox_inches="tight"`` crops to the artists, so text that overhangs the axes makes the
    saved figure wider than ``figsize``. Frontiers allows only 85 mm or 180 mm, so shrink the
    canvas until the tight bounding box measures ``width`` exactly; font sizes are in points
    and so are unaffected. The TIFF is then flattened to RGB, since Frontiers does not accept
    an alpha channel.
    """
    for _ in range(6):
        if relayout:
            fig.tight_layout()
        fig.canvas.draw()
        delta = fig.get_tightbbox(fig.canvas.get_renderer()).width - width
        if abs(delta) < 0.002:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w - delta, h)
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight", pad_inches=0)
    tiff = FIGS / f"{name}.tiff"
    fig.savefig(
        tiff,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    with Image.open(tiff) as im:
        if im.mode == "RGB":
            flat = im.copy()
        else:
            flat = Image.new("RGB", im.size, "white")
            flat.paste(im, mask=im.split()[-1] if im.mode == "RGBA" else None)
    target = round(width * 300)
    if flat.width != target:
        flat = flat.resize((target, round(flat.height * target / flat.width)), Image.LANCZOS)
    flat.save(tiff, dpi=(300, 300), compression="tiff_lzw")


def ic_coverage(s: pd.Series) -> float:
    """Percentage of non-missing IC values."""
    return 100 * s.notna().mean()


def fig1_coverage() -> None:
    """Figure 1: IC coverage per category (categories with at least 2000 nodes)."""
    N = pd.read_parquet(DATA / "nodes.parquet", columns=["category", "information_content"])
    cov = N.groupby("category").information_content.apply(ic_coverage)
    n = N.category.value_counts()
    cats = n[n >= 2000].index
    cov = cov[cats].sort_values()
    total = 100 * N.information_content.notna().mean()
    fig, ax = plt.subplots(figsize=(W2, 0.16 * len(cov) + 0.9))
    y = np.arange(len(cov))
    ax.barh(y, cov.values, color=C["score"], height=0.7, edgecolor="none")
    ax.axvline(total, color=C["gray"], lw=2, ls=":")
    for yi, (c, v) in enumerate(cov.items()):
        ax.text(v + 1, yi, f"{v:.0f}%  ({n[c]:,} nodes)", va="center", fontsize=8, color=C["ink"])
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("biolink:", "") for c in cov.index])
    ax.set_xlim(0, 135)
    ax.set_xlabel("Nodes with a curated IC value (%)")
    ax.grid(axis="y", visible=False)
    ax.text(total + 1, len(cov) - 0.3, f"all nodes {total:.0f}%", fontsize=8, color=C["gray"])
    save(fig, "fig1_coverage")


def load_sample_a(rng: np.random.Generator) -> pd.DataFrame:
    """List A with panel mean, IC, v2 score, generality axis and vertical jitter."""
    A = pd.read_csv(PANEL / "sample_a_ratings.csv")
    A = A.rename(columns={"information_content": "IC", "mean_rating": "panel"})
    v2 = pd.read_csv(V2 / "sample_a_ft_ens.csv").set_index("id").ft_ens
    A["v2"] = v2.reindex(A.id).to_numpy()
    A["jit"] = rng.normal(0, 0.045, len(A))
    axis = pd.read_csv(RESULTS / "sample_a_scores_SEALED.csv").set_index("id").ic_axis
    A["axis"] = axis.reindex(A.id).to_numpy()
    return A


def panel(
    ax: Axes,
    A: pd.DataFrame,
    col: str,
    xlabel: str,
    ann: list[tuple[str, str, int, int]],
    color: str,
    letter: str,
) -> None:
    """Scatter of one score against the panel rating with annotated example nodes."""
    ax.scatter(A[col], A.panel + A.jit, s=20, color=color, alpha=0.55, linewidths=0, zorder=3)
    for nm, key, dx, dy in ann:
        r = A[A.name.str.startswith(key)].iloc[0]
        y = r.panel + r.jit
        ax.scatter(
            [r[col]], [y], s=40, facecolors="none", edgecolors="#111", linewidths=2, zorder=4
        )
        ax.annotate(
            nm,
            (r[col], y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            color="#333",
            zorder=5,
            arrowprops={"arrowstyle": "-", "color": "#9CA3AF", "linewidth": 0.8, "shrinkB": 3},
        )
    ax.set_xlabel(xlabel)
    ax.set_yticks(range(1, 6))
    rho = spearmanr(A[col], A.panel).statistic
    ax.set_title(f"({letter}) Spearman rho = {rho:.2f}", loc="left")


def fig2_validation(A: pd.DataFrame) -> None:
    """Figure 2: IC and the SPS against the List A panel rating."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 3.4), sharey=True)
    panel(a1, A, "IC", "Information content (0 to 100)", ANN_IC, C["ic"], "A")
    panel(a2, A, "v2", "SPS (higher = more specific)", ANN_V2, C["score"], "B")
    a1.set_ylabel("Panel mean rating (1 = generic, 5 = specific)")
    fig.tight_layout()
    save(fig, "fig2_validation")


def fig3_bands(A: pd.DataFrame) -> None:
    """Figure 3: panel ratings between IC bands and within-band correlations (List A)."""
    A["band"] = pd.cut(A.IC, BANDS, labels=BL)
    bands = [
        {
            "ic": spearmanr(g.IC, g.panel).statistic,
            "axis": spearmanr(g.axis, g.panel).statistic,
            "score": spearmanr(g.v2, g.panel).statistic,
        }
        for b, g in A.groupby("band", observed=True)
    ]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 3.1), gridspec_kw={"width_ratios": [1.1, 1]})
    a1.boxplot(
        [A[A.band == b].panel for b in BL],
        tick_labels=BL,
        widths=0.5,
        patch_artist=True,
        medianprops={"color": "#111", "linewidth": 2},
        boxprops={"facecolor": C["fill"], "edgecolor": C["score"], "linewidth": 2},
        whiskerprops={"color": C["score"], "linewidth": 2},
        capprops={"color": C["score"], "linewidth": 2},
        flierprops={
            "marker": "o",
            "markersize": 3.5,
            "markerfacecolor": C["gray"],
            "markeredgecolor": "none",
        },
    )
    a1.set_xlabel("IC band")
    a1.set_ylabel("Panel mean rating")
    a1.set_yticks(range(1, 6))
    a1.set_title("(A) Between bands", loc="left")
    a1.tick_params(axis="x", labelsize=8)
    w = 0.26
    x = np.arange(len(BL))
    series = [("ic", "IC", C["ic"]), ("axis", "generality score", C["axis"])]
    series.append(("score", "SPS", C["score"]))
    for k, (key, lab, col) in enumerate(series):
        a2.bar(x + (k - 1) * w, [b[key] for b in bands], width=w, color=col, label=lab)
    a2.axhline(0, color=C["gray"], linewidth=2)
    a2.set_xticks(x)
    a2.set_xticklabels(BL, fontsize=8)
    a2.set_xlabel("IC band")
    a2.set_ylabel("Spearman rho within band")
    a2.set_ylim(-0.5, 0.85)
    a2.legend(loc="upper left")
    a2.set_title("(B) Within bands, n = 20 each", loc="left")
    fig.tight_layout()
    save(fig, "fig3_bands")


def load_answers() -> tuple[pd.DataFrame, int]:
    """Query answers with the SPS and block-list flag; returns the table and block-list count."""
    D = pd.read_parquet(V2 / "query_answer_features.parquet")
    v2 = pd.read_parquet(V2 / "wide_answers_ft_ens.parquet")["score_ft_ens"]
    D["v2"] = v2.reindex(D.id).to_numpy()
    b = pd.read_csv(RQ3 / "results_wide_v1lt0_iclt50" / "removed_blocklist.csv")
    bl = set(zip(b.case, b.id))
    D["blocked"] = [(c, i) in bl for c, i in zip(D.case, D.id)]
    return D, D.blocked.sum()


def arm(D: pd.DataFrame, nB: int, m: pd.Series) -> dict:
    """Removal counts, collateral counts and block-list coverage for one filter mask."""
    return dict(
        removed=int(m.sum()),
        ind=int((m & D.is_indication).sum()),
        dc=int((m & D.is_dcdrug).sum()),
        cov=100 * (m & D.blocked).sum() / nB,
    )


def sweep(D: pd.DataFrame, nB: int, col: str, cuts: list[float]) -> tuple[list[float], list[int]]:
    """Block-list coverage and DrugCentral drugs removed for each ``col < cut`` filter."""
    xs = []
    ys = []
    for c in cuts:
        m = D[col] < c
        xs.append(100 * (m & D.blocked).sum() / nB)
        ys.append(int((m & D.is_dcdrug).sum()))
    return xs, ys


def fig4_filtering(D: pd.DataFrame, nB: int) -> None:
    """Figure 4: removals per filter arm and the coverage-collateral trade-off."""
    arms = {
        "Block-list": arm(D, nB, D.blocked),
        "IC < 50": arm(D, nB, D.ic < 50),
        "IC < 80": arm(D, nB, D.ic < 80),
        "SPS < 0": arm(D, nB, D.v2 < 0),
    }
    fig = plt.figure(figsize=(W2, 3.4))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.2, 1.05, 1.05, 0.25, 1.9], wspace=0.5)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    names = list(arms)
    cols = [C["hand"], C["ic"], C["ic"], C["score"]]
    titles = ["Answers\nremoved", "Own indications\nremoved (of 978)", "DrugCentral\ndrugs removed"]
    for ax, key, title in zip(axs, ["removed", "ind", "dc"], titles):
        vals = [arms[n][key] for n in names]
        y = np.arange(len(names))[::-1]
        ax.barh(y, vals, color=cols, height=0.62, edgecolor="none")
        for yi, v in zip(y, vals):
            ax.text(v + max(vals) * 0.03, yi, f"{v:,}", va="center", fontsize=8, color=C["ink"])
        ax.set_yticks(y)
        ax.set_yticklabels(names if ax is axs[0] else [""] * 4)
        ax.set_title(title, loc="left")
        ax.set_xlim(0, max(vals) * 1.45)
        ax.grid(axis="y", visible=False)
    axs[0].set_title("(A) Removed\n(of 33,161)", loc="left")
    ax = fig.add_subplot(gs[0, 4])
    xs, ys = sweep(D, nB, "v2", list(np.arange(-6, 4.01, 0.25)))
    ax.plot(xs, ys, color=C["score"], lw=2, label="SPS, cutoff swept")
    for c in [-2, 0, 1, 2]:
        m = D.v2 < c
        x0 = 100 * (m & D.blocked).sum() / nB
        y0 = (m & D.is_dcdrug).sum()
        ax.scatter([x0], [y0], s=26, color=C["score"], zorder=3, edgecolor="white", linewidth=1)
        ax.annotate(
            f"cutoff {c:+g}",
            (x0, y0),
            textcoords="offset points",
            xytext=(5, -3),
            fontsize=8,
            color=C["ink"],
        )
    xs, ys = sweep(D, nB, "ic", [30, 40, 50, 60, 70, 75, 80, 85, 90])
    ax.plot(xs, ys, color=C["ic"], lw=2, label="IC, cutoff swept")
    ax.scatter([xs[6]], [ys[6]], s=26, color=C["ic"], zorder=3, edgecolor="white", linewidth=1)
    ax.annotate(
        "IC < 80",
        (xs[6], ys[6]),
        textcoords="offset points",
        xytext=(5, 2),
        fontsize=8,
        color=C["ink"],
    )
    h = arms["Block-list"]
    ax.scatter(
        [h["cov"]],
        [h["dc"]],
        s=55,
        marker="D",
        color=C["hand"],
        zorder=4,
        edgecolor="white",
        linewidth=1,
        label="block-list",
    )
    ax.set_yscale("symlog", linthresh=50)
    ax.set_xlabel("Block-list coverage (%)")
    ax.set_ylabel("DrugCentral drugs removed")
    ax.set_title("(B) Coverage vs. collateral", loc="left")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "fig4_filtering", relayout=False)


def figs1_sample_b(rng: np.random.Generator) -> None:
    """Supplementary Figure S1: the SPS against the List B panel rating."""
    B = pd.read_csv(SAMPLE_B / "results" / "sample_b_panel_scores.csv")
    v2 = pd.read_csv(V2 / "sample_b_ft_ens.csv").set_index("id").ft_ens
    B["v2"] = v2.reindex(B.id).to_numpy()
    fig, ax = plt.subplots(figsize=(W1 * 1.6, 3.4))
    jb = rng.normal(0, 0.05, len(B))
    ax.axvline(0, color=C["gray"], lw=2, ls="--", zorder=1)
    ax.scatter(B.v2, B.panel + jb, s=24, color=C["score"], alpha=0.75, linewidths=0, zorder=3)
    for nm, text, dx, dy in SAMPLE_B_LABELS:
        i = B.index[B.name == nm][0]
        ax.annotate(
            text,
            (B.v2[i], B.panel[i] + jb[i]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8,
            color=C["ink"],
            ha="left" if dx > 0 else "right",
            va="center",
            zorder=4,
            arrowprops={"arrowstyle": "-", "color": C["gray"], "lw": 0.7, "shrinkB": 3},
        )
    ax.set_xlim(-4.0, 5.0)
    ax.set_ylim(0.6, 5.6)
    ax.set_xlabel("SPS (higher = more specific)")
    ax.set_ylabel("Panel mean rating (1 = generic, 5 = specific)")
    ax.set_yticks(range(1, 6))
    ax.text(0.10, 0.72, "filter cutoff", rotation=90, fontsize=8, color=C["gray"], va="bottom")
    rho = spearmanr(B.v2, B.panel).statistic
    ax.set_title(f"List B, 50 nodes without IC, Spearman rho = {rho:.2f}", loc="left")
    fig.tight_layout()
    save(fig, "figS1_sample_b", width=W1 * 1.6)


def main() -> None:
    """Draw all figures in order with one random generator shared across them."""
    rng = np.random.default_rng(SEED)
    fig1_coverage()
    A = load_sample_a(rng)
    fig2_validation(A)
    fig3_bands(A)
    D, nB = load_answers()
    fig4_filtering(D, nB)
    figs1_sample_b(rng)
    print("figures written")


if __name__ == "__main__":
    main()

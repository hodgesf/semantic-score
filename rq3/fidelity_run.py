"""Fidelity check of the unfiltered ARAX configuration on five test cases.

Posts each case in both the production and the unfiltered mode, saves the responses under
fidelity/ and compares the production list with the unfiltered list after removing the answers
production dropped. Output: fidelity/fidelity_summary.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arax_client import CAP, answers, post  # noqa: E402

from paths import RQ3  # noqa: E402

FID = RQ3 / "fidelity"
CASES = {
    "TestCase_12": "MONDO:0001505",
    "TestCase_0": "MONDO:0011705",
    "TestCase_1": "MONDO:0016063",
    "TestCase_6": "MONDO:0005301",
    "TestCase_10": "MONDO:0010808",
}


def run_case(case: str, curie: str) -> dict[str, Any]:
    """Query both modes for one case and return the comparison record."""
    p, tp = post(curie, "prod", FID / f"{case}_prod.json")
    n, tn = post(curie, "nofilter", FID / f"{case}_nofilter.json")
    P, _ = answers(p)
    N, names = answers(n)
    prod = [x for x, _ in P]
    nof = [x for x, _ in N]
    removed = [x for x in nof if x not in set(prod)]
    replay = [x for x in nof if x not in set(removed)][:CAP]
    extra = [x for x in prod if x not in set(nof)]
    rec = {
        "case": case,
        "curie": curie,
        "prod": len(prod),
        "prod_s": round(tp),
        "nofilter": len(nof),
        "nofilter_s": round(tn),
        "removed": len(removed),
        "removed_names": [names.get(x) for x in removed],
        "extra_in_prod": extra,
        "order_identical": replay == prod,
        "top30_identical": replay[:30] == prod[:30],
        "same_top30_set": set(replay[:30]) == set(prod[:30]),
        "same_set": set(replay) == set(prod),
    }
    print(
        f"{case} {curie}: prod {len(prod)} ({tp:.0f}s) nofilter {len(nof)} ({tn:.0f}s) "
        f"removed {len(removed)} order={rec['order_identical']} top30={rec['top30_identical']} "
        f"top30set={rec['same_top30_set']} set={rec['same_set']} extra={len(extra)}"
    )
    return rec


def main() -> None:
    """Run all five cases and write the summary."""
    out = [run_case(case, curie) for case, curie in CASES.items()]
    json.dump(out, open(FID / "fidelity_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()

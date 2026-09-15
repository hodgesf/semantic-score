"""Run every disease of a query set as an unfiltered treats query against the local ARAX server.

RQ3_QUERIES, RQ3_RESPONSES and RQ3_LOG select the query csv, the response directory and the log
file (defaults: the original 200 in data/query_set_wide.csv, responses_wide, logs/run_wide.log;
the held-out 200 use query_set_heldout.csv, responses_heldout and logs/run_heldout.log).
Existing responses are skipped.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arax_client import answers, post  # noqa: E402

from paths import RQ3  # noqa: E402

Q = os.environ.get("RQ3_QUERIES", str(RQ3 / "data" / "query_set_wide.csv"))
R = os.environ.get("RQ3_RESPONSES", str(RQ3 / "responses_wide"))
LOG = os.environ.get("RQ3_LOG", str(RQ3 / "logs" / "run_wide.log"))


def main() -> None:
    """Query each disease in the set, appending progress to LOG."""
    os.makedirs(R, exist_ok=True)
    q = pd.read_csv(Q)
    log = open(LOG, "a")
    for i, r in enumerate(q.itertuples(), 1):
        path = f"{R}/{r.case}_nofilter.json"
        if os.path.exists(path):
            continue
        try:
            d, dt = post(r.disease, "nofilter", path)
            rows, _ = answers(d)
            msg = (
                f"{i:3d}/{len(q)} {r.case} {r.disease} {r.name[:40]}: {d.get('status')} "
                f"{len(rows)} results {dt:.0f}s"
            )
        except Exception as e:
            msg = f"{i:3d}/{len(q)} {r.case} {r.disease}: FAILED {e!r}"
            traceback.print_exc(file=log)
        print(msg)
        log.write(msg + "\n")
        log.flush()
    print("DONE")


if __name__ == "__main__":
    main()

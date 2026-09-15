"""Filesystem locations shared by every script in this repository.

Repository paths are resolved relative to ROOT. Inputs that are not distributed
with the code (the tier-0 graph, SapBERT node embeddings, the ARAX code base,
the Translator test suite and the earlier predicted-IC baseline) are read from
environment variables, falling back to the locations used during the study.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PANEL = ROOT / "panel"
V1 = ROOT / "v1"
V2 = ROOT / "v2"
SAMPLE_B = ROOT / "sample_b"
RQ1 = ROOT / "rq1_report"
RQ3 = ROOT / "rq3"
LLM_PANEL = ROOT / "llm_panel"

SAMPLE_B_XLSX = SAMPLE_B / "data" / "sampleB.xlsx"

SEED = 2654


def _env_path(name: str, default: str) -> Path:
    """Return the path named by environment variable ``name``, or ``default``."""
    return Path(os.environ.get(name, os.path.expanduser(default)))


TIER0_DIR = _env_path(
    "SEMSCORE_TIER0_DIR", "~/Desktop/code/database/tier0-20260621/knowledge_graph"
)
NODE_EMBEDDINGS = _env_path("SEMSCORE_EMBEDDINGS", "~/Desktop/code/large_files/node_embeddings.npy")
RTX_DIR = _env_path("SEMSCORE_RTX_DIR", "~/Desktop/code/RTX")
TRANSLATOR_TESTS = _env_path(
    "SEMSCORE_TESTS", "~/Desktop/code/Tests/test_suites/sprint_6_tests.json"
)
PREDICTED_IC_DIR = _env_path(
    "SEMSCORE_PREDICTED_IC_DIR", "~/Desktop/code/generic_concepts_archive/data"
)

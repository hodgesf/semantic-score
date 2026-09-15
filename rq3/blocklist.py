"""Offline replica of ARAX's production blocklist filter.

Uses ARAX's own RemoveNodes._is_general_concept (Filter_KG/remove_nodes.py) together with its
master general_concepts.json, applied to TRAPI node dicts. ARAX's loader depends on the working
directory, so the import is done from inside the ARAXQuery directory.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import RTX_DIR  # noqa: E402

_ARAXQUERY = RTX_DIR / "code" / "ARAX" / "ARAXQuery"
sys.path.append(str(_ARAXQUERY))
_cwd = os.getcwd()
os.chdir(_ARAXQUERY)
from Filter_KG.remove_nodes import RemoveNodes  # noqa: E402

os.chdir(_cwd)
RemoveNodes.load_block_list_file()
_bl = RemoveNodes._get_block_list_dict()
_rn = RemoveNodes.__new__(RemoveNodes)
_rn.block_list_synonyms = set(map(str.lower, _bl["synonyms"]))
_rn.block_list_curies = set(map(str.lower, _bl["curies"]))
_rn.block_list_patterns = [re.compile(p, re.IGNORECASE) for p in _bl["patterns"]]
N_CURIES = len(_bl["curies"])
N_SYN = len(_bl["synonyms"])
N_PAT = len(_bl["patterns"])


def is_blocked(node_id: str, node: dict[str, Any]) -> bool:
    """Return True if the TRAPI node (name, categories, attributes) matches the blocklist.

    ARAX checks the node's own CURIE through its xref/equivalent_identifiers attributes; the id
    itself is also checked against the curie set.
    """
    if node_id.lower() in _rn.block_list_curies:
        return True
    n = dict(node)
    n.setdefault("attributes", [])
    return bool(_rn._is_general_concept(n))

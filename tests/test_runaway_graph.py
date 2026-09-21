# -*- coding: utf-8 -*-
"""A graph whose ring closures ran away is withheld before anything can copy it.

    python tests/test_runaway_graph.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CHEMEAGLE_NETWORK", "0")
os.environ.setdefault("API_KEY", "not-used-here")   # importing the agent module asks for one
from get_molecular_agent import RUNAWAY_SMILES_CHARS, _withhold_runaway_graph  # noqa: E402

checked = 0

# 1. the reading that lost ajoc.202200438 example 4: a cage repeated until the reply was cut off inside it
runaway = {"category": "[Mol]", "bbox": [0.1, 0.1, 0.2, 0.2],
           "smiles": "CC.CCCCCCCCC1CC2C1" + "C1C2C2C1" * 400,
           "symbols": ["C"] * 3200, "coords": [[0.1, 0.1]] * 3200, "edges": [[0] * 3200] * 3200,
           "atoms": [{"atom_symbol": "C"}] * 3200, "bonds": [], "molfile": "..."}
assert _withhold_runaway_graph(runaway) is True
assert runaway["smiles"] == "*" and runaway["symbols"] == ["*"], runaway["smiles"][:40]
assert runaway["coords"] == [[0.5, 0.5]] and runaway["edges"] == [[0]]
assert runaway["atoms"] == [{"atom_symbol": "*", "x": 0.5, "y": 0.5}] and runaway["bonds"] == []
assert "molfile" not in runaway and runaway["runaway_graph"] is True
assert runaway["bbox"] == [0.1, 0.1, 0.2, 0.2], "the box keeps its place"
checked += 1

# 2. a long but real molecule is left alone: the longest in this benchmark's ground truth is 132 characters
longest_real = ("CC(C)(C)OC(=O)N1CCC(CC1)Oc1ccc(cc1)C(=O)N1CCN(CC1)c1ccc(cc1)C(=O)NC1CCN(CC1)"
                "c1ccccc1C(F)(F)F")
assert len(longest_real) < RUNAWAY_SMILES_CHARS
box = {"smiles": longest_real, "symbols": ["C"], "coords": [[0.1, 0.1]], "edges": [[0]]}
assert _withhold_runaway_graph(box) is False
assert box["smiles"] == longest_real
checked += 1

# 3. nothing to judge: no SMILES at all, or one that is not a string
for value in (None, "", 42):
    box = {"smiles": value}
    assert _withhold_runaway_graph(box) is False
    checked += 1

print("runaway graph: %d cases pass" % checked)

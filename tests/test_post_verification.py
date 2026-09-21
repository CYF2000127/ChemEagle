# -*- coding: utf-8 -*-
"""post_verification runs the repairs in order and leaves the result in place.

    python tests/test_post_verification.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CHEMEAGLE_NETWORK", "0")
from chemietoolkit import helper as h  # noqa: E402
from rdkit import Chem  # noqa: E402


def canon(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else None


checked = 0

# 1. the steps that need no extra argument: an unreadable SMILES is repaired, a name resolves from the alias map,
#    and a charge the drawing does not show is dropped
data = {"reactions": [{"reactants": [{"smiles": "Cc1ccc([S-])cc1"}],
                       "products": [{"smiles": "c1ccc(SeSec2ccccc2)cc1"}],
                       "conditions": [{"role": "reagent", "text": "DBU (15 mol%)"}]}]}
data = h.post_verification(data)          # the first step rebuilds the tree, so take what it returns
rxn = data["reactions"][0]
assert rxn["reactants"][0]["smiles"] == canon("Cc1ccc(S)cc1"), rxn["reactants"][0]
assert canon(rxn["products"][0]["smiles"]) is not None, rxn["products"][0]
assert rxn["conditions"][0].get("smiles") == "C1CCC2=NCCCN2CC1", rxn["conditions"][0]
checked += 1

# 2. a condition naming a compound the figure draws takes that drawing over a name lookup
data = {"reactions": [{"reactants": [], "products": [],
                       "conditions": [{"role": "catalyst", "text": "B (10 mol%)", "label": "B"}]}]}
data = h.post_verification(data, label_structures={"B": "CC(C)c1cccc(C(C)C)c1"})
assert data["reactions"][0]["conditions"][0]["smiles"] == "CC(C)c1cccc(C(C)C)c1", data["reactions"][0]["conditions"][0]
checked += 1

# 3. the structure drawn once beside the arrow reaches every reaction of the figure
rows = [{"reactants": [], "products": [], "conditions": [{"role": "reagent", "text": "oxidant (4)"}]},
        {"reactants": [], "products": [], "conditions": [{"role": "reagent", "text": "oxidant (4)"}]}]
out = h.post_verification({"reactions": rows}, drawn_conditions=["O=C1C=CC(=O)C=C1"])
assert all(r["conditions"][0].get("smiles") == "O=C1C=CC(=O)C=C1" for r in out["reactions"]), out
checked += 1

# 4. the order is the one the pipeline needs: the charge repair runs last, over a structure an earlier step added
data = {"reactions": [{"reactants": [], "products": [],
                       "conditions": [{"role": "catalyst", "text": "A (10 mol%)", "label": "A"}]}]}
data = h.post_verification(data, label_structures={"A": "C[C-](O)c1ccccc1"})
assert data["reactions"][0]["conditions"][0]["smiles"] == canon("CC(O)c1ccccc1"), data["reactions"][0]["conditions"][0]
checked += 1

print("post_verification: %d cases pass" % checked)

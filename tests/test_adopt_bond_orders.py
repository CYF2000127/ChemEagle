# -*- coding: utf-8 -*-
"""The bond order guard in adopt: a disagreement about how many lines a bond has keeps the reaction agent's
graph, while every other kind of disagreement still hands the box to the molecular agent.

    python tests/test_adopt_bond_orders.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chemietoolkit.mol_edit_plan.reconcile import _bond_order_disagreement, adopt  # noqa: E402

checked = 0

# 1. what the guard is for
for own, donor in (("C#Cc1ccccc1", "C=Cc1ccccc1"),                 # phenylacetylene read as styrene
                   ("CC#N", "CC=N"),
                   ("O=C1CCCC1", "OC1=CCCC1")):                    # a ketone read as an enol: same atoms, other bonds
    assert _bond_order_disagreement(own, donor) is True, (own, donor)
    checked += 1

# 2. what it must not catch
for own, donor in (("C#Cc1ccccc1", "C#Cc1ccccc1"),                 # the same molecule
                   (r"*/C=C\C([H])=O", "*/C=C/C([H])=O"),          # E/Z only: geometry is not a bond order
                   ("CS(=O)(=O)/C=C(O)c1ccccc1", "CS(C)(=O)=CC(=O)c1ccccc1"),   # atoms differ: the donor wins
                   ("*C=C(C#N)C#N", "N#CC(C#N)=Cc1ccccc1"),        # a template against an expanded variant
                   ("not a smiles", "C=Cc1ccccc1"), ("C=Cc1ccccc1", None)):
    assert _bond_order_disagreement(own, donor) is False, (own, donor)
    checked += 1


def box(smiles, bbox, symbols=("C", "C")):
    return {"category": "[Mol]", "bbox": list(bbox), "smiles": smiles, "symbols": list(symbols),
            "coords": [[0.1, 0.1], [0.2, 0.2]], "edges": [[0, 1], [1, 0]]}


# 3. end to end: the alkyne box keeps its own graph, a box whose atoms differ takes the donor's
rxn = [{"reactants": [box("C#Cc1ccccc1", (0.1, 0.1, 0.2, 0.2)),
                      box("CS(=O)(=O)C=C(O)c1ccccc1", (0.4, 0.1, 0.5, 0.2))],
        "products": [], "conditions": []}]
donors = [box("C=Cc1ccccc1", (0.1, 0.1, 0.2, 0.2)),
          box("CS(C)(=O)=CC(=O)c1ccccc1", (0.4, 0.1, 0.5, 0.2), symbols=("C", "S", "C"))]
records = adopt(rxn, donors)
assert rxn[0]["reactants"][0]["smiles"] == "C#Cc1ccccc1", rxn[0]["reactants"][0]["smiles"]
assert rxn[0]["reactants"][1]["smiles"] == "CS(C)(=O)=CC(=O)c1ccccc1", rxn[0]["reactants"][1]["smiles"]
assert [r.get("reason") for r in records] == ["donor differs only in bond orders", None], records
checked += 1

print("adopt bond order guard: %d cases pass" % checked)

# -*- coding: utf-8 -*-
"""The charge and radical repair: what it neutralises, and what it must leave exactly as drawn.

    python tests/test_charge_repair.py
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

# 1. what the recogniser charged and the figure draws neutral
for before, after in (("Cc1ccc([S-])cc1", "Cc1ccc(S)cc1"),                       # a thiol read as a thiolate
                      ("C[C-](O)c1ccccc1", "CC(O)c1ccccc1"),                     # an alcohol carbon as a carbanion
                      ("[CH3]", "C"), ("[C]", "C"), ("[NH2]", "N"),              # lone radicals from drawing dirt
                      ("*[SH2]C1CC1", "*SC1CC1"),                                # a hypervalent sulfur
                      ("C#[N+]C(C)(C)C", "[C-]#[N+]C(C)(C)C")):                  # a nitrilium is really an isocyanide
    got = h._charge_repaired_smiles(before)
    assert got == canon(after), (before, got, canon(after))
    checked += 1

# a ring CH read as a radical is repaired, and so is an amide nitrogen read as a cation
assert h._charge_repaired_smiles("CC(=O)NN1[CH]C(C)CC1=O") == canon("CC(=O)NN1CC(C)CC1=O")
assert h._charge_repaired_smiles("[H]C(=O)C(=O)[N+]1CCOCC1") == canon("O=CC(=O)N1CCOCC1")
assert h._charge_repaired_smiles("Brc1cc[c]cc1") == canon("Brc1ccccc1")
checked += 3

# a sulfoxonium ylide read as a cation: neutralising it fills the sulfur's valence exactly, so no hydrogen is added.
# The answer spells the sulfur in brackets, since it now carries no implicit hydrogen, and is the same molecule.
assert canon(h._charge_repaired_smiles("C[S+](C)(=O)=CC(=O)c1ccccc1")) == canon("CS(C)(=O)=CC(=O)c1ccccc1")
checked += 1

# 2. what must survive untouched
for kept in ("[C-]#[N+]C(C)(C)C",                                                # already the charge separated form
             "CCCC[N+](CCCC)(CCCC)CCCC.[F-]",                                    # a drawn salt, charges balanced
             "[O-]S(=O)(=O)[O-].[Cu+2]",                                         # sulfate with its counter cation
             "C[N+](C)(C)CC(=O)[O-]",                                            # a betaine, neutral overall
             "CC(C)c1cccc(C(C)C)c1-[n+]1csc2c1CCCC2",                            # an azolium whose counter ion was missed
             "[Na]", "[Li]", "[Se]", "C[Si](C)C",                                # a metal or a metalloid is never hydrided
             "CC1(C)CCCC(C)(C)N1[O]",                                            # TEMPO: an aminoxyl radical is drawn as such
             "CC(C)c1cccc(C(C)C)c1N1[C]N(c2ccccc2)C=C1",                         # an N-heterocyclic carbene keeps its carbene
             "Cl.[1*][C@H]([NH3+])C(=O)O[2*]",                                   # a hydrochloride whose HCl was read neutral
             "COC(=O)[C@H](C)[NH3+].Cl",
             "*SC1CC1", "CCO"):                                                  # nothing to do
    assert h._charge_repaired_smiles(kept) is None, (kept, h._charge_repaired_smiles(kept))
    checked += 1

# 3. the tree walk reaches reactants, products and conditions alike
data = {"reactions": [{"reactants": [{"smiles": "Cc1ccc([S-])cc1"}],
                       "products": [{"smiles": "[CH3]"}],
                       "conditions": [{"role": "reagent", "smiles": "C#[N+]C(C)(C)C"},
                                      {"role": "solvent", "smiles": "[O-]S(=O)(=O)[O-].[Cu+2]"}]}]}
h.repair_charges_and_radicals_in_data(data)
rxn = data["reactions"][0]
assert rxn["reactants"][0]["smiles"] == canon("Cc1ccc(S)cc1"), rxn["reactants"][0]
assert rxn["products"][0]["smiles"] == "C", rxn["products"][0]
assert rxn["conditions"][0]["smiles"] == canon("[C-]#[N+]C(C)(C)C"), rxn["conditions"][0]
assert rxn["conditions"][1]["smiles"] == "[O-]S(=O)(=O)[O-].[Cu+2]", rxn["conditions"][1]
checked += 1

# 4. an unreadable SMILES is left as it is rather than dropped
assert h._charge_repaired_smiles("not a smiles at all") is None
assert h._charge_repaired_smiles("") is None
checked += 1

print("charge repair: %d cases pass" % checked)

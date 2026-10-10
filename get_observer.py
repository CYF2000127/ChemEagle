import base64
import json
import os
import time
from typing import Any, List, Optional

from openai import AzureOpenAI, OpenAI
import llm_client as llm
from openai import InternalServerError, RateLimitError, APIError


API_KEY = os.getenv("API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = API_KEY or os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = AZURE_ENDPOINT or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = API_VERSION or "2024-06-01"
        
        if not api_key or not azure_endpoint:
            return None
        
        _client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
        )
    return _client

PLAN_PROMPT_TEMPLATE = """System Message: 
You are a plan observer. Given the graphic and the current list of agent calls (plan), decide whether the plan is sufficient.

User Message: 
Given the graphic and the current list of agent calls (plan), please recheck the component of the graphic and decide whether the corresponding agent is called and the plan is sufficient.
Valid agents:
1. Reaction template parsing agent
Parses the reaction scheme to identify reactants, products, and label mappings, and outputs a structured reaction template.

2. Molecular recognition agent
Detects other molecules in the graphic except in the reaction template, recognizes their structures, and returns normalized representations (e.g., SMILES, labels, positions).

3. Structure-based R-group substitution agent
Uses structure panels / variant images to extract R-group values and generate enumerated products from a core scaffold based on structural information.

4. Text-based R-group substitution agent
Reads R-group tables and enumerates products or substituents on top of a given core scaffold using text information.

5. Condition interpretation agent
Extracts and normalizes reaction conditions (catalysts, reagents, solvent, temperature, time, atmosphere, etc.) from the graphic.

6. Text extraction agent
Performs chmical NER and text-based reaction extraction on the text description.

If the plan is acceptable, return the original plan as-is.
If adjustments are required, provide the improved list of agents and briefly explain the changes.

Key distinction between Structure-based and Text-based R-group agents:
The deciding factor is what each TABLE CELL contains:
- COMPLETE product/substrate molecules (full structures) → Structure-based R-group substitution agent
- R-group FRAGMENTS or substituent values (text like "Me"/"Ph", or partial structures with wavy bonds showing only the R-group portion) → Text-based R-group substitution agent
Even if every cell is a drawn structure, if those structures are R-group FRAGMENTS (not complete products), use the Text-based agent.

Always respond in valid JSON with the structure:
{{
  "list_of_agents": [...],
  "redo": true/false,
  "reason": "If changed is true, give an explanation; otherwise leave blank."
}}

Current plan (JSON):
{plan_json}
"""

# The action observer prompt is assembled per agent: a shared header, the
# deliverable and checklist of the agent under review, shared rules, the
# output format, then the agent output itself (see build_action_prompt).
ACTION_PROMPT_HEADER = """You are the action observer of ChemEagle, a multi-agent system that turns chemistry figures (reaction schemes, substrate scope panels, R-group tables, condition tables) into machine-readable reactions. One agent has just finished. You receive the original figure and that agent's output. Audit the output against the figure: find every concrete error you can verify by looking at the figure, and decide whether the output is so broken that the agent has to run again.

How your answer is used:
- The issues you list are forwarded to the final synthesis step, which sees the same figure and the outputs of all agents and writes the final reactions. That step can only fix what you point to precisely, so for every issue give where it is (reaction_id, side, molecule label or list index, table entry), what the output says, and what the figure shows.
- redo=true makes the program run the same agent once more with the same input. The rerun does not see your issues and part of its input is cached, so a rerun cannot fix an individual error. Set redo only for the whole-output failures listed under "When to set redo".
"""

ACTION_AGENT_SPECS = {
    "get_multi_molecular_full": """Agent under review: Molecular recognition agent (get_multi_molecular_full)

What it should deliver: a list with one entry per chemical structure drawn in the figure, each {"smiles", "texts", "bbox_id"}. "texts" holds the text printed next to the structure: normally its label, sometimes a yield or a definition such as "Ar = Ph". This agent reads structures only; it does not assign reaction roles or conditions.

Checklist:
- Coverage: every drawn structure appears exactly once, including catalysts, ligands, additives and structures drawn above or below an arrow or inside a table. Plain text, arrows and table borders get no entry. Two drawings must not be merged into one entry, and one drawing must not be split into several.
- Structure: each SMILES matches its drawing (see the SMILES rules below). In a template drawn with R, R1, Ar, X and so on, exactly the variable positions are "*" and everything else is concrete.
- Labels: "texts" carries the label printed next to that drawing (1a, 3b, B1, "oxidant (4)"). Report a label attached to the wrong structure, a clearly printed label that is missing, and two structures sharing one label.
- Inline definitions: when the figure defines a variant next to a drawing ("B: Ar = Ph", "R = Me"), the entry for that variant carries the defined group, not "*".
""",
    "get_full_reaction_template": """Agent under review: Reaction template parsing agent (get_full_reaction_template)

What it should deliver: {"molecule_coref": [...]}, one entry per structure drawn in and around the reaction scheme, each {"smiles", "texts", "bbox_id"}, where "texts" holds the label or other text printed next to the structure. In the current version the reaction roles (reactant, product, condition) are not part of this output; the final synthesis step assigns them from the figure, so do not report their absence. If the output does contain reactions (reactants, conditions, products), check them as well: arrow direction, every reactant and product of each step present, structures drawn over or under an arrow placed in conditions, one reaction per arrow.

Checklist:
- Coverage: every structure of the scheme appears exactly once: starting materials, products, intermediates, and catalysts or reagents drawn as structures. Plain text, arrows and table borders get no entry. Two drawings must not be merged, and one drawing must not be split.
- Structure: each SMILES matches its drawing (see the SMILES rules below). In a template drawn with R, R1, Ar, X and so on, exactly the variable positions are "*" and everything else is concrete.
- Labels: "texts" carries the label printed next to that drawing. Report a label attached to the wrong structure, a clearly printed label that is missing, and two structures sharing one label.
- Inline definitions: a variant defined next to a drawing ("A1: R = TBS", "C: Ar = Mes") carries the defined group, not "*".
""",
    "get_reaction_con": """Agent under review: Condition interpretation agent (get_reaction_con)

What it should deliver: {"reaction_template": {"conditions": [...]}, "reactions": {"<entry number or product label>": {"conditions": [...]}}}. Every condition is {"role", "text"} with optional "smiles" and "label". The role is one of: reagent (catalysts, ligands, bases and additives included), solvent, temperature, time, yield, add_info (everything else: ee, er, dr, conversion, ratios, atmosphere, light source, footnote remarks).

Checklist:
- Format: both keys present and filled, "reactions" keyed by the entry numbers or product labels printed in the figure. Reasoning notes, an "error" key or echoed tool calls in place of this structure are a format failure.
- Coverage: every condition text of the figure is captured: the text above and below each arrow, the footnotes ("a Reaction conditions: ...", "b at 60 °C"), and every row of a condition or optimization table. Count the table rows and the entries: each table row needs its own entry under the same number.
- Row alignment, the most damaging error: each entry carries exactly the values of its own table row (catalyst, reagent, solvent, temperature, time, yield, ee, dr). Compare the first row, the last row and several rows in between cell by cell. Values shifted to the neighbouring row, or one row's values copied into others, are high severity.
- Shared versus specific: conditions printed on the arrow or in the general footnote belong to reaction_template; values that vary per row belong to the entries; a footnote mark printed on some rows applies to those rows only.
- Text fidelity: numbers, units and names copied exactly (mol %, equiv, °C, h; Cs2CO3 not Cs2C03; catalyst label A not 4). "20 mol % C" names the same catalyst the figure labels C.
- Roles: a solvent tagged reagent, a yield tagged add_info, "rt" tagged reagent, and similar misclassifications.
- SMILES, when given, matches the named reagent (K2CO3, DCE, a drawn catalyst referenced by its label). A condition whose "smiles" is "*", or whose text is really a CCDC number, a compound label or a yield, is a misread.
""",
    "process_reaction_image_with_product_variant_R_group": """Agent under review: Structure-based R-group substitution agent (process_reaction_image_with_product_variant_R_group)

What it should deliver: {"reactions": [...]}. reaction_id "0_1" (note "reaction template") is the drawn template: reactants and products with "*" or "[n*]" at the variable positions, plus the shared conditions. Every other reaction is one drawn product variant of the scope panel: "products" holds that drawn product with its printed label, "reactants" holds the template reactants with this variant's groups filled in, "conditions" holds the shared conditions and the text printed under that product (yield, ee, dr, footnote mark). The agent's internal molecule list has been removed from what you see.

Checklist:
- Coverage: count the drawn product variants (follow the label sequence, for example 3a to 3o) and compare with the number of non-template reactions. Report every missing label, every duplicate, and every reaction whose product is not a drawn variant (a reagent, or the template itself counted as a variant).
- Products: each product SMILES matches the drawing with that label: substituent identity, substituent position, heteroatoms, rings.
- Reactants: in each variant reaction the reactants are the template reactants carrying exactly the groups this product carries at the variable positions. Report reactants that still contain "*", reactants copied from another row, a group placed on the wrong reactant or the wrong atom, and reactants that cannot give the product under the template.
- Template: the reactant and product templates match the drawn scheme and use the same placeholder number for the same group; no concrete group where the figure draws R, and no "*" where the figure draws a concrete group.
- Leftover placeholders: outside reaction 0_1 a "*" is an error unless the drawing of that product itself shows a variable group.
- Labels and per-product text: each product carries its own printed label and its own yield, ee or dr; text taken from a neighbouring product is an error. A condition with "smiles": "*", or with a label that is really a CCDC number or a yield, is a misread.
""",
    "process_reaction_image_with_table_R_group": """Agent under review: Text-based R-group substitution agent (process_reaction_image_with_table_R_group)

What it should deliver: {"reactions": [...]}. reaction_id "0_1" is the drawn template: reactants and products with "*" or "[n*]" at R, R1, Ar, X and so on. Every other reaction is one row of the R-group table: the template with that row's groups written into reactants and products, the conditions, and "additional_info" with the values printed in that row (entry, R values, time, yield, ee, product label).

Checklist:
- Coverage: count the table rows (every panel and continuation column) and compare with the number of non-template reactions. Report missing rows, duplicated rows, and rows taken from a different table. A row the table marks as failed ("N.R.", "trace", "0%") still counts as a row.
- Values: additional_info of each reaction matches its table row cell by cell (entry number, every R column, yield, ee or dr, time, product label).
- Substitution: in every row each variable position is filled with the group from the right column, in reactants and products alike; an R1 value goes to the placeholder that stands for R1, not to R2. Check that the SMILES really contains that group: 4-BrC6H4 is para-bromophenyl bonded through a ring carbon, OMe is methoxy, n-Bu is not t-Bu, 2-naphthyl is not 1-naphthyl, 2-thienyl is not 3-thienyl, "H" means hydrogen.
- Leftover placeholders: outside reaction 0_1 any "*" means a group was not substituted.
- Core: the part of the template the table does not vary is identical in every row. A row whose core differs from the template, or a template that differs from the drawing, is high severity.
- Row-specific conditions: time, temperature, catalyst or reagent columns and footnote marks of a row end up in that row's conditions or additional_info.
""",
    "text_extraction_agent": """Agent under review: Text extraction agent (text_extraction_agent)

What it should deliver: {"annotated_text": "..."}: the prose or caption text of the figure reproduced verbatim, with tags inserted after the chemical mentions: a role tag ([reactant], [product], [reagent], [catalyst], [solvent] ...), optional type tags ([ABBREVIATION], [SYSTEMATIC], [IDENTIFIERS], [MULTIPLE] ...), and [SMILES:...] where a structure can be assigned, aligned with the molecules found in the graphic.

Checklist:
- Text fidelity: the annotated text reproduces the figure's prose; no sentence dropped, invented or paraphrased, and no OCR garble left in chemical names or numbers.
- Entities: every chemical mention in the text is tagged, and nothing that is not a chemical (authors, units, procedures) is tagged as one.
- Roles: the role of each mention agrees with the text and with the graphic (a compound the scheme shows as a product is not tagged reactant).
- SMILES: a [SMILES:...] tag matches the named compound or the drawn molecule with that label; a label in the text ("3a", "G32") is mapped to the molecule with the same label in the graphic.
""",
}

ACTION_GENERIC_SPEC = """Agent under review: {agent_name}

Checklist:
- Coverage: everything in the figure that this agent is responsible for appears in the output, once.
- Structures: every SMILES matches the drawing it comes from (see the SMILES rules below).
- Values: labels, conditions, yields and other texts match the figure and are attached to the right molecule or reaction.
"""

ACTION_PROMPT_CONVENTIONS = """Conventions that are NOT errors (do not report them):
- SMILES notation: atom order, canonical or non-canonical form, aromatic or Kekulé rings, explicit [H], ring-closure digits.
- Placeholders: "*" and "[n*]" (for example [1*], [5*]) stand for a variable group (R, R1, Ar, X, Y, a wavy-bond attachment). Their numbers need not match the R numbers printed in the figure; only consistency matters (the same group carries the same number in a template's reactants and products).
- Salts: counter-ions and salt partners written as "."-separated fragments (BF4-, PF6-, I-, Cl-, OTf-, HCl) are correct when the figure draws or states them.
- Stereochemistry absent from the output when the figure draws no wedges, hashes or defined double-bond geometry.
- Information the figure does not contain, and tasks that belong to another agent (for example conditions missing from the molecular recognition output).
- Chemistry you find implausible but the figure actually draws. The figure is the reference, not your expectation of the chemistry.
- The pipeline's own markers: an empty "bbox_id", fields such as "compound_id" or "note", and hint texts the pipeline writes itself ("There is no label or failed to detect, please recheck the image again", "(the label maybe wrong, please check the image again)"). Such a hint is not an error by itself; a printed label that is missing or wrong is.
"""

ACTION_SMILES_FULL = """Reading SMILES: count the atoms instead of judging by string length, and parse branches fully before you claim a difference. In "CCC=C(X)Y" the third carbon is the alkene carbon, so the group on it is ethyl; "CCC(=O)R" is an ethyl ketone; "c1ccc2ccccc2c1" bonded through its first atom is 2-naphthyl, "c1cccc2ccccc12" is 1-naphthyl. Most wrong reports come from misreading a correct SMILES, so before you report a structure error, write the SMILES of the drawing yourself (figure_smiles below) and compare the two.

Errors to look for in every SMILES you check:
- Skeleton: wrong ring size or ring fusion, a ring missing or added, wrong chain length.
- Atoms: wrong heteroatom or heteroatom position (O, N, S; Cl, Br, I), abbreviation expanded wrongly (OMe or Me, CO2Et or COEt, n-Bu or t-Bu, Ts or Ms, 2-thienyl or 3-thienyl, Mes, Bpin, TBS).
- Position: substituent on the wrong ring atom (ortho, meta, para) or on the wrong ring.
- Bond order and functional group: ketone, ester or alcohol confused; alkene or alkane; alkyne; nitrile; nitro.
- Charge: a drawn charge missing or on the wrong atom; a drawn counter-ion missing.
- Stray fragments: "."-separated pieces that are not drawn, typically OCR residue of labels, yields or specks: ".I", ".[B]", ".O", "F.", "C=C.", "[2H]", a lone "*". Quote the fragment.
- Unresolved placeholder: "*" where the figure shows a defined group, or an abbreviation left unexpanded.
- Stereochemistry that contradicts a drawn wedge, hash or double-bond geometry.
"""

# Used instead of ACTION_SMILES_FULL while MAIN_BODY_LOCK is on.
ACTION_SMILES_LOCKED = """Scope of structure checks: leave the main body of every molecule alone. The main body is the largest "."-separated part of a SMILES: its atoms, bonds, ring positions and stereochemistry. Do not report errors inside it (ring size, substituent position, chain length, bond order, functional group, stereochemistry); the program discards such reports, whatever the checklist above asks. The structure problems you do report are only these:
- Stray fragments: "."-separated pieces that are not drawn, typically OCR residue of labels, yields or specks: ".I", ".[B]", ".O", "F.", "C=C.", "[2H]", a lone "*". Quote the fragment.
- Counter-ions and charges: a drawn or stated counter-ion missing or wrong (BF4-, PF6-, ClO4-, OTf-, I-), a drawn charge missing, a salt whose charges do not balance.
- Unresolved placeholders: "*" or "[n*]" left where the figure defines the group, by drawing it or by printing it ("R = Me", "Ar = 4-ClC6H4", "n = 6").
- Invalid SMILES: a SMILES that is not valid chemistry (impossible valences, garbled ring closures).
For each of them give figure_smiles: the whole structure, counter-ions included, with the main body copied unchanged from the output; only for an invalid SMILES or an unresolved placeholder write the main body from the figure.
"""

ACTION_PROMPT_AUDIT = """How to audit:
1. Before reading the output, take inventory of the figure: list what this agent is responsible for (see the checklist) and count it.
2. Compare the inventory with the output: missing items, extra items, duplicates, items attached to the wrong label or row.
3. Scan every item for the cheap textual faults: leftover "*", stray fragments, empty or "None" SMILES. Then compare structures and values with the figure; in a long list compare at least the first item, the last item and every item whose SMILES or label looks unusual.
4. Report only what you can point to in the figure. Where you cannot read the relevant part of the figure, say nothing rather than guess.

When to set redo=true (only these cases):
- The output is empty, is an error message, is truncated, or does not follow the agent's output format (for example reasoning notes and no result).
- Coverage collapse: roughly a third or more of the expected items (structures, product variants, table rows, condition entries) are missing, or the output describes a different part of the figure. Items are whole structures, reactions, rows and entries; missing yields, ee values, footnotes or labels never count toward this.
- A systematic failure that spoils most items: every variant row still contains "*", every product equals the template, the table values are shifted by one row throughout.
In every other case set redo=false, even when you list many issues.

Respond with one JSON object and nothing else:
{
  "redo": false,
  "reason": "One or two sentences on the most important problems; empty when the output is acceptable.",
  "issues": [
    {
      "location": "where the error is, e.g. reaction 4_1, product (label 7d)",
      "category": "one of the categories below",
      "severity": "high or low",
      "output_says": "the value in the output, quoted; for a structure error the complete SMILES as written in the output",
      "figure_shows": "what the figure shows instead",
      "figure_smiles": "for wrong_structure, stray_fragment and unresolved_placeholder: the complete SMILES of the structure as it should be, counter-ions included (see the SMILES rules above); otherwise omit"
    }
  ]
}
category is one of: missing_item, extra_item, duplicate, wrong_structure, stray_fragment, unresolved_placeholder, wrong_label, row_misalignment, wrong_role, wrong_value, format.
Report each error once, under the category that fits best; do not split one wrong SMILES into several issues.
severity "high": the error changes a structure, reactant, product or condition, loses or invents a reaction, row or structure, or confuses which structure a label belongs to (two structures with one label, a label on the wrong structure, a missing label that table rows or conditions refer to). A wrong SMILES is high in every agent, the text extraction agent included. "low": missing yield, ee, dr or footnote text, add_info placement, label spelling and other details that leave the structures and reactions unchanged.
An acceptable output gives {"redo": false, "reason": "", "issues": []}. Do not invent issues to fill the list.
"""

# Fields that only carry drawing geometry or atom bookkeeping; they cost tokens
# and give the observer nothing to compare with the figure.
_OBSERVER_DROP_KEYS = {"symbols", "atoms", "bonds", "molfile", "coords", "bbox", "category",
                       "category_id", "score", "corefs", "edges"}
# Top-level keys that are an agent's internal state, not its deliverable. The
# product-variant agent's original_molecule_list keeps raw recogniser SMILES
# (with label specks such as ".I") that its reconstruction already cleans.
_OBSERVER_DROP_TOP = {"original_molecule_list"}


def _compact_for_observer(obj: Any, top: bool = True) -> Any:
    if isinstance(obj, dict):
        return {k: _compact_for_observer(v, False) for k, v in obj.items()
                if k not in _OBSERVER_DROP_KEYS and not (top and k in _OBSERVER_DROP_TOP)}
    if isinstance(obj, list):
        return [_compact_for_observer(v, False) for v in obj]
    return obj


def _split_tool_result(tool_result: Any):
    """(agent name, agent output) from what the dispatcher passes:
    [{"name": ..., "result": ...}], a bare dict of that shape, or a bare result."""
    item = tool_result[0] if isinstance(tool_result, list) and len(tool_result) == 1 else tool_result
    if isinstance(item, dict) and "name" in item and "result" in item:
        return str(item["name"]), item["result"]
    return "", tool_result


def build_action_prompt(tool_result: Any) -> str:
    agent_name, result = _split_tool_result(tool_result)
    base_name = agent_name[:-len("_azure")] if agent_name.endswith("_azure") else agent_name
    spec = ACTION_AGENT_SPECS.get(base_name) or ACTION_GENERIC_SPEC.replace("{agent_name}", agent_name or "unknown")
    result_json = json.dumps(_compact_for_observer(result), ensure_ascii=False, indent=1)
    smiles_rules = ACTION_SMILES_LOCKED if MAIN_BODY_LOCK else ACTION_SMILES_FULL
    return (ACTION_PROMPT_HEADER + "\n" + spec + "\n" + ACTION_PROMPT_CONVENTIONS + "\n" + smiles_rules
            + "\n" + ACTION_PROMPT_AUDIT + "\nAgent output (JSON):\n" + result_json + "\n")


def _canonical(smiles: Any) -> Optional[str]:
    if not isinstance(smiles, str):
        return None
    s = smiles.strip().strip('"\'` ')
    if not s or " " in s:
        return None
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:
        return None


def _contradicts_itself(issue: dict) -> bool:
    """A structure issue whose figure_smiles is the very molecule the output
    already has: the observer misread the output SMILES, not the figure."""
    if issue.get("category") not in ("wrong_structure", "stray_fragment", "unresolved_placeholder"):
        return False
    said, drawn = _canonical(issue.get("output_says")), _canonical(issue.get("figure_smiles"))
    return said is not None and said == drawn


# With the lock on (the default) the observer may not change the main body of a
# molecule: the largest "."-separated part of its SMILES, with its atoms, bonds
# and stereo. A structure claim survives only when it concerns what lies outside
# that body (counter-ions, stray fragments), a formal charge, an unresolved "*",
# or a SMILES that is not valid at all. Its readings of drawn skeletons were the
# main source of wrong reports (1- vs 2-naphthyl, chain lengths).
MAIN_BODY_LOCK = os.getenv("OBSERVER_MAIN_BODY_LOCK", "1") != "0"
_STRUCTURE_CATEGORIES = ("wrong_structure", "stray_fragment", "unresolved_placeholder")


def _mol(smiles: Any, sanitize: bool = True):
    if not isinstance(smiles, str):
        return None
    s = smiles.strip().strip('"\'` ')
    if not s or " " in s:
        return None
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        return Chem.MolFromSmiles(s, sanitize=sanitize)
    except Exception:
        return None


def _skeleton(mol) -> str:
    """Atoms and bonds only: charges, hydrogens, isotopes and stereo removed."""
    from rdkit import Chem
    m = Chem.RWMol(mol)
    for a in m.GetAtoms():
        a.SetFormalCharge(0)
        a.SetNumExplicitHs(0)
        a.SetNoImplicit(True)
        a.SetIsotope(0)
    m.UpdatePropertyCache(strict=False)
    return Chem.MolToSmiles(m, isomericSmiles=False)


def _split_main_body(mol):
    """(main-body skeleton, other fragments, main-body charge, main body with stereo)."""
    from rdkit import Chem
    frags = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False))
    main = max(range(len(frags)), key=lambda k: frags[k].GetNumAtoms())
    others = sorted(Chem.MolToSmiles(f, isomericSmiles=False) for k, f in enumerate(frags) if k != main)
    return (_skeleton(frags[main]), others, Chem.GetFormalCharge(frags[main]),
            Chem.MolToSmiles(frags[main]))


def _main_body_verdict(issue: dict) -> Optional[str]:
    """None when the issue may stand, else why it is discarded under the lock."""
    category, figure, said = issue.get("category"), issue.get("figure_smiles"), issue.get("output_says")
    if category not in _STRUCTURE_CATEGORIES and (not figure or _mol(said, sanitize=False) is None):
        return None                      # not a claim about an existing SMILES (missing item, label, row ...)
    out = _mol(said)
    if out is None:
        if _mol(said, sanitize=False) is not None:
            issue["output_invalid"] = True
            return None                  # not valid chemistry: the body is broken anyway
        if category == "unresolved_placeholder" and "*" in str(said or ""):
            return None
        return "the claim cannot be checked against an output SMILES"
    if category == "unresolved_placeholder":
        return None if any(a.GetAtomicNum() == 0 for a in out.GetAtoms()) else "the output has no '*'"
    drawn = _mol(figure)
    if drawn is None:
        from rdkit import Chem
        if category == "stray_fragment" and len(Chem.GetMolFrags(out)) > 1:
            return None
        return "no valid figure_smiles to check the claim"
    o, d = _split_main_body(out), _split_main_body(drawn)
    if o[0] != d[0]:
        return "changes the main body"
    if o[1] != d[1] or o[2] != d[2]:
        return None                      # differs only outside the body or in charge
    return "changes only stereo or hydrogens of the main body" if o[3] != d[3] else "figure_smiles equals the output"


def _parse_action_reply(parsed: dict) -> dict:
    issues = parsed.get("issues")
    issues = [i for i in issues if isinstance(i, dict)] if isinstance(issues, list) else []
    kept, dropped = [], []
    for i in issues:
        if MAIN_BODY_LOCK:
            why = _main_body_verdict(i)
        else:
            why = "figure_smiles equals the output" if _contradicts_itself(i) else None
        if why:
            dropped.append(dict(i, dropped_because=why))
        else:
            kept.append(i)
    issues = kept
    return {
        "dropped_issues": dropped,
        "redo": bool(parsed.get("redo", False)),
        "reason": parsed.get("reason", "") or "",
        "issues": issues,
        "list_of_agents": parsed.get("list_of_agents", []),
    }


def observer_note(agent_name: str, check: dict, max_issues: int = 12) -> str:
    """The line handed to the final synthesis step: the observer's summary plus
    its high-severity issues, or "" when there is nothing worth forwarding."""
    high = [i for i in check.get("issues") or [] if str(i.get("severity", "high")).lower() != "low"]
    reason = (check.get("reason") or "").strip()
    if not high and not (check.get("redo") and reason):
        return ""
    parts = []
    for i in high[:max_issues]:
        text = f"[{i.get('location', '?')}] {i.get('category', '')}"
        if i.get("output_says"):
            text += f": output has {i['output_says']}"
        if i.get("figure_shows"):
            text += f"; figure shows {i['figure_shows']}"
        # under the lock a reading is passed on only to replace a SMILES that is not valid
        if i.get("figure_smiles") and (i.get("output_invalid") or not MAIN_BODY_LOCK):
            text += f" (observer's reading, verify before use: {i['figure_smiles']})"
        parts.append(text)
    if len(high) > max_issues:
        parts.append(f"{len(high) - max_issues} more")
    note = f"{agent_name}: {reason}" if reason else agent_name
    return note + (" Issues: " + " | ".join(parts) if parts else "")


def _encode_image(image_path: str) -> str | None:
    if not image_path or not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def plan_observer_agent_azure(image_path: str, tool_calls: List[Any]) -> dict:
    """Returns {"list_of_agents": list, "redo": bool, "reason": str}."""
    default = {"list_of_agents": tool_calls, "redo": False, "reason": ""}
    base64_image = _encode_image(image_path)
    plan_json = json.dumps(tool_calls or [], ensure_ascii=False, indent=2)
    prompt = PLAN_PROMPT_TEMPLATE.format(plan_json=plan_json)

    user_content = [{"type": "text", "text": prompt}]
    if base64_image:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            }
        )

    try:
        client = _get_client()
        if client is None:
            return default
        
        response = client.chat.completions.create(
            model="gpt-5-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
            ],
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        return {
            "list_of_agents": parsed.get("list_of_agents", parsed.get("plan", tool_calls)),
            "redo": bool(parsed.get("redo", False)),
            "reason": parsed.get("reason", ""),
        }
    except Exception:
        return default


def action_observer_agent_azure(image_path: str, tool_result: Any) -> dict:
    """Returns {"redo": bool, "reason": str, "issues": list, "list_of_agents": list}."""
    default = {"redo": False, "reason": "", "issues": [], "list_of_agents": []}
    base64_image = _encode_image(image_path)
    prompt = build_action_prompt(tool_result)

    user_content = [{"type": "text", "text": prompt}]
    if base64_image:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            }
        )

    try:
        client = _get_client()
        if client is None:
            return default
        
        response = client.chat.completions.create(
            model="gpt-5-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
            ],
        )
        content = response.choices[0].message.content
        return _parse_action_reply(json.loads(content))
    except Exception:
        return default


def retry_api_call(func, max_retries=3, base_delay=2, backoff_factor=2, *args, **kwargs):
    """
    Generic API call retry function with exponential backoff support.
    
    Args:
        func: function to call
        max_retries: maximum number of retries
        base_delay: base delay time (seconds)
        backoff_factor: backoff factor (retry delay = base_delay * backoff_factor^attempt)
        *args, **kwargs: parameters passed to func
    
    Returns:
        return value of func
    
    Raises:
        exception from the final attempt
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except (InternalServerError, RateLimitError, APIError) as e:
            last_exception = e
            error_code = getattr(e, 'status_code', None) or getattr(e, 'code', None)
            error_message = str(e)
            
            # Check whether this is a 503 error or another retryable error
            if error_code == 503 or 'overloaded' in error_message.lower() or '503' in error_message:
                if attempt < max_retries - 1:
                    delay = base_delay * (backoff_factor ** attempt)
                    print(f"⚠️ API call failed (503/overloaded), attempt {attempt + 1}/{max_retries}. Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"❌ API call failed, reached maximum retries ({max_retries})")
                    raise
            else:
                # Other error types, raise directly
                raise
        except Exception as e:
            # Other unknown errors, raise directly
            raise
    
    # If all retries failed
    if last_exception:
        raise last_exception
    raise RuntimeError("API call failed, unknown error")


############################### ChemEagle (any OpenAI-compatible endpoint)
def plan_observer_agent(
    image_path: str,
    tool_calls: List[Any],
    *,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    """
    the LLM endpoint version of plan_observer_agent, using the OpenAI-compatible endpoint configured in llm.get_client().

    Returns:
        dict: {"list_of_agents": list, "redo": bool, "reason": str}
    """
    default = {"list_of_agents": tool_calls, "redo": False, "reason": ""}
    model_name = llm.resolve_model(model_name)
    base_url = llm.resolve_base_url(base_url)
    api_key = llm.resolve_key(api_key)
    _mk = llm.model_kwargs(model_name)

    client = llm.get_client(api_key=api_key, base_url=base_url)

    base64_image = _encode_image(image_path)
    plan_json = json.dumps(tool_calls or [], ensure_ascii=False, indent=2)
    prompt = PLAN_PROMPT_TEMPLATE.format(plan_json=plan_json)

    user_content = [{"type": "text", "text": prompt}]
    if base64_image:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            }
        )

    try:
        response = retry_api_call(
            client.chat.completions.create,
            max_retries=5,
            base_delay=3,
            backoff_factor=2,
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
            ],
            **_mk,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            print(f"⚠️ Warning: plan_observer_agent could not parse JSON, returning the original plan")
            return default
        
        return {
            "list_of_agents": parsed.get("list_of_agents", parsed.get("plan", tool_calls)),
            "redo": bool(parsed.get("redo", False)),
            "reason": parsed.get("reason", ""),
        }
    except Exception as e:
        print(f"⚠️ Warning: plan_observer_agent error: {e}, returning the original plan")
        return default


def action_observer_agent(
    image_path: str,
    tool_result: Any,
    *,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    """
    the LLM endpoint version of action_observer_agent, using the OpenAI-compatible endpoint configured in llm.get_client().

    Returns:
        dict: {"redo": bool, "reason": str, "issues": list, "list_of_agents": list}
    """
    default = {"redo": False, "reason": "", "issues": [], "list_of_agents": []}
    model_name = llm.resolve_model(model_name)
    base_url = llm.resolve_base_url(base_url)
    api_key = llm.resolve_key(api_key)
    _mk = llm.model_kwargs(model_name)

    client = llm.get_client(api_key=api_key, base_url=base_url)

    base64_image = _encode_image(image_path)
    prompt = build_action_prompt(tool_result)

    user_content = [{"type": "text", "text": prompt}]
    if base64_image:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            }
        )

    try:
        response = retry_api_call(
            client.chat.completions.create,
            max_retries=5,
            base_delay=3,
            backoff_factor=2,
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
            ],
            **_mk,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            print(f"⚠️ Warning: action_observer_agent could not parse JSON, returning no redo")
            return default

        return _parse_action_reply(parsed)
    except Exception as e:
        print(f"⚠️ Warning: action_observer_agent error: {e}, returning no redo")
        return default

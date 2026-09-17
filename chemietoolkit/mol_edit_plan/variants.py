# -*- coding: utf-8 -*-
"""Deterministic reshaping of the structure-variant agent's output (get_R_group_sub_agent.normalize_product_variant_output).

The id-mode agent labels every drawn molecule with [label, text, role]; the tool result gives the reaction template
(RxnIM) and the drawn products expanded by the R-group back-out. Two figure types need a fixed rule on top of that:

* condition-role override: a charged molecule (a catalyst salt) the agent calls "conditions" that RxnIM placed
  among the reactants is moved to the conditions of the template and of every row; neutral ones stay, they are
  usually true reactants;
* catalyst screening: several condition-role molecules that each carry their own outcome ("4, 49% yield, 50% ee")
  or that form a labelled series (N1 ... N7) are one reaction each in the benchmark, not one reaction with
  every catalyst in its conditions.
"""
import re

CONDITION_ROLES = {'condition', 'conditions'}
ROLE_WORDS = CONDITION_ROLES | {'product', 'reactant', 'reactant template', 'product template'}
_YIELD = re.compile(r'(?<![\w.])(<\s*\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?\s*%)(?!\s*ee\b)(?:\s*yield)?', re.I)
_EE = re.compile(r'(-?\d+(?:\.\d+)?\s*%\s*ee|(?<!\w)ee\s*[=:]?\s*-?\d+(?:\.\d+)?\s*%)', re.I)
_NO_YIELD = re.compile(r'\b(n\.?\s?d\.?|n\.?\s?r\.?|trace|no reaction|not detected)\b', re.I)
_SERIES = re.compile(r'^([A-Za-z]{0,3})-?(\d+)[a-z]?$')


def _role_of(info):
    return next((str(x).lower().strip() for x in info if isinstance(x, str) and str(x).lower().strip() in ROLE_WORDS), None)


def canonical(smiles):
    """RDKit canonical form for comparisons (the back-out re-emits reactants through RDKit, so the same
    molecule can carry two spellings); the raw string when RDKit cannot parse it."""
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
        m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        return Chem.MolToSmiles(m) if m is not None else smiles
    except Exception:
        return smiles


def is_charged(smiles):
    """A salt or ion (a bracket atom with a formal charge): the shape of a drawn catalyst (azolium BF4-, ammonium
    salt). Neutral reagents the agent also calls "conditions" (NFSI, a diol) are often the figure's reactants."""
    return bool(re.search(r"\[[^\]]*[+-][^\]]*\]", smiles or ""))


def condition_role_smiles(original_molecule_list):
    """Charged molecules the agent labelled as a condition (catalyst salts), keyed by canonical form -> the
    agent's spelling. Only these may be moved out of the reactants: on the 2026-09-14 logs the neutral
    "conditions" that RxnIM placed among the reactants (NFSI in 298-300, the diol in 3c01437) are reactants in
    the benchmark, the charged one (163's azolium) is a condition."""
    return {canonical(s): s for s, info in (original_molecule_list or {}).items()
            if isinstance(info, list) and _role_of(info) in CONDITION_ROLES and is_charged(s)}


def outcome_text(info):
    """The yield / ee text printed under a condition-role molecule (the info items that are neither the label,
    a role word nor an id), or None when it carries no outcome."""
    parts = [str(x) for x in info[1:] if isinstance(x, str) and not x.startswith(('bbox_id=', 'id='))
             and x.lower().strip() not in ROLE_WORDS]
    text = ', '.join(p for p in parts if p.strip())
    return text if text and (_YIELD.search(text) or _EE.search(text) or _NO_YIELD.search(text)) else None


def label_series(labels):
    """True when the printed labels form one numbered series (N1, N2, N3 / 15, 16, 17 / 4a, 4b, 4c)."""
    parsed = [_SERIES.match(str(l).strip()) for l in labels]
    if len(parsed) < 3 or not all(parsed):
        return False
    prefixes = {m.group(1).lower() for m in parsed}
    return len(prefixes) == 1 and len({m.group(0).lower() for m in parsed}) == len(parsed)


def move_condition_molecules(reactions, original_molecule_list):
    """Condition-role override on the normalised reactions (template first, rows after), in place. A reactant
    entry whose SMILES the agent labelled as a condition becomes a reagent of that reaction; a reaction never
    loses its last reactant. Returns the moved SMILES."""
    cond = condition_role_smiles(original_molecule_list)
    moved = set()
    for rx in reactions or []:
        reactants = rx.get('reactants') or []
        hits = [r for r in reactants if isinstance(r, dict) and canonical(r.get('smiles')) in cond]
        if not hits or len(hits) == len(reactants):
            continue
        rx['reactants'] = [r for r in reactants if r not in hits]
        conditions = rx.setdefault('conditions', [])
        for r in hits:
            spelling = cond[canonical(r['smiles'])]
            if not any(isinstance(c, dict) and canonical(c.get('smiles')) == canonical(spelling) for c in conditions):
                info = original_molecule_list.get(spelling) or []
                entry = {'role': 'reagent', 'smiles': spelling}
                if info and isinstance(info[0], str) and info[0].strip():
                    entry['label'] = info[0].strip()
                conditions.append(entry)
            moved.add(spelling)
    if moved:
        print(f"[roles] {len(moved)} condition-role molecule(s) moved from reactants to conditions: {sorted(moved)}")
    return moved


def expand_catalyst_screening(reactions, original_molecule_list):
    """Catalyst-screening figures: one reaction template and several condition-role molecules that each carry
    their own outcome, or that form a labelled series. The tool result puts them all into the template's
    conditions; the benchmark counts one reaction per catalyst. When no product-variant row exists, the
    template keeps the other conditions and one row per screened catalyst is appended, with the template's
    reactants and products, that catalyst as the reagent and its yield / ee as text conditions."""
    if not reactions or reactions[0].get('note') != 'reaction template' or len(reactions) != 1:
        return reactions
    template = reactions[0]
    candidates = []
    for cond in template.get('conditions') or []:
        smiles = cond.get('smiles') if isinstance(cond, dict) else None
        info = original_molecule_list.get(smiles) if smiles else None
        if isinstance(info, list) and info and _role_of(info) in CONDITION_ROLES:
            candidates.append((cond, str(info[0]).strip(), outcome_text(info)))
    with_outcome = [c for c in candidates if c[2]]
    if len(with_outcome) >= 2:
        screened = with_outcome
    elif len(candidates) >= 3 and label_series([c[1] for c in candidates]):
        screened = candidates
    else:
        return reactions
    keep = [c for c in template['conditions'] if not any(c is s[0] for s in screened)]
    template['conditions'] = keep
    out = [template]
    for k, (cond, label, text) in enumerate(screened, start=1):
        conditions = [{'role': 'reagent', 'smiles': cond['smiles'], 'label': label}]
        if text:
            y, e = _YIELD.search(text), _EE.search(text)
            if y:
                conditions.append({'role': 'yield', 'text': y.group(1).strip()})
            elif _NO_YIELD.search(text):
                conditions.append({'role': 'yield', 'text': _NO_YIELD.search(text).group(1)})
            if e:
                conditions.append({'role': 'ee', 'text': e.group(1).strip()})
        conditions.extend(keep)
        out.append({'reaction_id': f'{k}_1', 'note': f'catalyst screening: {label}',
                    'reactants': [dict(r) for r in template.get('reactants', [])],
                    'conditions': conditions,
                    'products': [dict(p) for p in template.get('products', [])]})
    print(f"[screening] {len(screened)} condition molecules -> one reaction each: {[s[1] for s in screened]}")
    return out

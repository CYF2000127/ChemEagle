"""Validated, immutable label edits, single-valued definitions and local variant
expansion for the molecular recognition agent. No graph repair.

Derived from new-chat/work/rgroup_plan/postprocess.py (2026-09-09). Additions:

* ``definitions``: single-valued equations written as standalone text
  (``Ar = 4-OMeC6H4``). They are applied in place to every whole ``[Ar]`` token in
  their scope (all molecules when ``scope`` is empty), without cloning molecules,
  without derived labels and without touching corefs. This is the case the
  original prototype could only express by faking a compound id, which cloned
  the molecule and dropped its real label link.
* decision action ``substitute`` for molecules whose variables were resolved by a
  definition; decisions are required only for molecules that still carry a
  variable after definitions were applied.
* ``[n*]`` (the vision tool's own notation for Rn) is addressable through an OCR
  correction to ``[Rn]``; it is not treated as a variable until then.

Composite tokens that embed a variable (``[SO2Ar]``, ``[OR]``) get the value spliced in
by the program: for a definition when the variable is defined, and for an expansion group
when the model lists the composite atom as that variable (``[OR]`` with rows R = TBS and
R = H gives ``[OTBS]`` and ``[OH]``). A splice is kept only when Graph2SMILES expands it
without a wildcard.
"""
import argparse
import copy
import hashlib
import json
import pathlib
import re

import jsonschema


class PlanError(ValueError):
    """A plan entry failed validation. ``entry`` = (plan section, index) of the entry being checked
    when the check ran, or None for a cross-entry check."""

    def __init__(self, message, entry=None):
        super().__init__(message)
        self.entry = entry


_CURRENT_ENTRY = None      # (section, index) of the plan entry under validation, for PlanError.entry


def _entering(section, index):
    global _CURRENT_ENTRY
    _CURRENT_ENTRY = (section, index)


def _leaving():
    global _CURRENT_ENTRY
    _CURRENT_ENTRY = None


def obj(**fields):
    return {'type': 'object', 'properties': fields, 'required': list(fields), 'additionalProperties': False}


def arr(item):
    return {'type': 'array', 'items': item}


S = {'type': 'string', 'minLength': 1}
NULLABLE_S = {'type': ['string', 'null']}
PERCENT = {'type': ['number', 'null'], 'minimum': 0, 'maximum': 100}
SCHEMA = obj(
    schema_version={'type': 'string', 'enum': ['1.3']},
    ocr_corrections=arr(obj(molecule_id=S, atom_id=S, expected_symbol=S, corrected_symbol=S, evidence=S)),
    atom_corrections=arr(obj(molecule_id=S, atom_id=S, expected_symbol=S, corrected_symbol=S,
                             kind={'type': 'string', 'enum': ['charge', 'label_from_atom', 'lookalike', 'anion_element']}, evidence=S)),
    text_corrections=arr(obj(text_id=S, corrected_text=arr(S), evidence=S)),
    definitions=arr(obj(name=S, literal_value=S, replacement_symbol=S, source_text_id=NULLABLE_S,
                        evidence=S, scope=arr(S))),
    groups=arr(obj(molecule_id=S, source_text_id=NULLABLE_S, evidence=S,
                   variables=arr(obj(name=S, atom_id=S, expected_symbol=S)),
                   variants=arr(obj(compound_id=S, source_text=S,
                                    bindings=arr(obj(name=S, literal_value=S, replacement_symbol=S)),
                                    yield_percent=PERCENT, ee_percent=PERCENT)))),
    charged_molecules=arr(S),
    counter_ions=arr(obj(molecule_id=S, ion=S, evidence=S)),
    decisions=arr(obj(molecule_id=S, action={'type': 'string', 'enum': ['expand', 'substitute', 'keep_generic', 'review']}, reason=S)),
    structure_warnings=arr(obj(molecule_id=S, reason=S)))

# Placeholder vocabulary of the vision tool (molnextr.constants.RGROUP_SYMBOLS): R, R1.., Ra-Rf,
# X, Y, Z, Q, A, E, EWG, Nu, Ar, Ar1.., optionally primed.
VARIABLE = re.compile(r"\[(?:R\d*|R[abcdf]|X\d*|Y\d*|Z\d*|Q|A|E|EWG|Nu|Ar\d*)'*\]$")
LABEL = re.compile(r"\[[A-Za-z0-9()+\-',.]+\]$")
PROTECTED = {'[H]', '[C@H]', '[C@@H]', '[C@]', '[C@@]', '[nH]', '[N+]', '[O-]'}
# Atomic tokens are structural data, not OCR abbreviations. Ar/R-style
# placeholders take precedence over an element spelling in this label format.
ELEMENTS = set('H He Li Be B C N O F Ne Na Mg Al Si P S Cl K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og'.split())
ATOMIC = re.compile(r'\[(?:\d+)?([A-Z][a-z]?|[bcnops])(?:H\d*)?(?:[+\-]\d*|\+\+|--)?(?::\d+)?\]$')
# Elements that legitimately occur as drawn atoms in organic / organometallic schemes. Any other
# element token ([Re], [Pr], [Pa], [Fl] ...) is a "lookalike": almost always a misread label, and the
# vision tool itself uses rare elements (Lv, Lu, Nd, Yb, At, Fm, Er) as placeholder atoms.
COMMON_ELEMENTS = set('H B C N O F Si P S Cl Br I Se Te Sn Ge Li Na K Mg Ca Zn Cu Pd Pt Ni Co Fe Ru Rh Ir Au Ag Al Ti Zr Hf Mn Cr Mo W Hg Pb Bi Sb As Ga In Tl Cs Rb Ba Sr Sc La Ce Sm Eu Gd Yb Lu Os V Nb Ta Cd'.split())
RARE_ELEMENTS = ELEMENTS - COMMON_ELEMENTS
# Common elements a drawn placeholder is still misread as: the metals. A terminal one may be corrected back to
# a placeholder the scheme draws elsewhere (R1 read as [Ti] or [Li]). Hydrogen, the halogens and the elements
# that carry substituents in their own right (B, Si, Sn, Ge, P, S, Se) are never touched: they are drawn atoms.
MISREAD_METALS = COMMON_ELEMENTS - set('H B C N O F Si P S Cl Br I Se Te Sn Ge'.split())
CHARGED = re.compile(r"\[[A-Za-z][A-Za-z0-9]*[+\-]\d?\]$")          # [N+] [Cl-] [BF4-] [S+]
BARE_ATOM = re.compile(r"(?:[A-Z][a-z]?|\*|\[\d+\*\])$")             # C, N, Cl, *, [2*]
CHARGE_BEARERS = {'N', 'P', 'S', 'O'}


def element_of(token):
    """Element letters of an atom-like token, charges/brackets/digits/stereo removed (aromatic lowercase upper-cased)."""
    core = re.sub(r"[\[\]+\-@\d]", "", token or "")
    return core[:1].upper() + core[1:] if core else core


def degrees(box):
    """Bond count per atom from the tool's edge matrix, or None when the box carries no graph."""
    edges = box.get('edges')
    if not edges:
        return None
    return [sum(1 for j, v in enumerate(row) if v and j != i) for i, row in enumerate(edges)]


NEUTRAL_VALENCE = {'N': 3, 'P': 3, 'O': 2, 'S': 2}


def valences(box):
    """Bond-order sum per atom from the tool's edge matrix (aromatic 1.5, wedge/dash 1), or None."""
    edges = box.get('edges')
    if not edges:
        return None
    order = {1: 1.0, 2: 2.0, 3: 3.0, 4: 1.5, 5: 1.0, 6: 1.0}
    return [sum(order.get(v, 1.0) for j, v in enumerate(row) if v and j != i) for i, row in enumerate(edges)]


def charge_candidates(symbols, degs, vals=None):
    """Atoms whose bonding could require a positive charge. First choice: uncharged N/P/S/O whose
    bond-order sum exceeds the neutral valence (an N with a double bond and two single bonds must be
    N+; its bridgehead neighbour with three single bonds must not). Only when no atom is over-valent
    does the older degree rule apply: for each element the atoms with the highest degree (>= 3),
    equivalent ring N sharing the tie."""
    out = []
    if degs is None:
        return out
    if vals is not None:
        over = [j for j, s in enumerate(symbols)
                if element_of(s) in NEUTRAL_VALENCE and not CHARGED.fullmatch(s) and vals[j] > NEUTRAL_VALENCE[element_of(s)] + 1e-6]
        if over:
            return sorted(over)
    for el in sorted(CHARGE_BEARERS):
        same = [j for j, s in enumerate(symbols) if element_of(s) == el and not CHARGED.fullmatch(s)]
        if same:
            top = max(degs[j] for j in same)
            if top >= 3:
                out.extend(j for j in same if degs[j] == top)
    return sorted(out)


ANION_TOKENS = {'F', 'Cl', 'Br', 'I', 'BF4', 'PF6', 'ClO4', 'OTf', 'TfO', 'NTf2', 'SbF6', 'OAc', 'AcO', 'OTs', 'TsO',
                'OMs', 'MsO', 'NO3', 'HSO4', 'BArF', 'B(C6F5)4', 'I3', 'Br3', 'CN', 'SCN', 'N3', 'OH', 'CO3', 'PF6-'}


def is_anion_token(symbol):
    return clean(symbol) in ANION_TOKENS and not CHARGED.fullmatch(symbol)


def ring_sets(box):
    """Ring membership from the tool's edge matrix (RDKit ring perception on a dummy graph); [] if unavailable."""
    edges = box.get('edges')
    if not edges:
        return []
    try:
        from rdkit import Chem
    except Exception:
        return []
    n = len(edges)
    m = Chem.RWMol()
    for _ in range(n):
        m.AddAtom(Chem.Atom(6))
    for i in range(n):
        for j in range(i + 1, n):
            if edges[i][j]:
                m.AddBond(i, j, Chem.BondType.SINGLE)
    Chem.FastFindRings(m)
    return [set(r) for r in m.GetRingInfo().AtomRings()]


def ring_equivalent(cands, rings):
    """True when every pair of candidate atoms shares a ring: the formal charge can sit on any of
    them without changing the species (azolium-type resonance)."""
    return all(any(a in r and b in r for r in rings) for a in cands for b in cands if a < b)


def composite_token_ok(token):
    """Can Graph2SMILES turn this composite superatom (e.g. [SO2(4-MeOC6H4)]) into a valid,
    wildcard-free fragment? Probed on a two-atom graph. None when the converter is unavailable."""
    try:
        from molnextr.chemistry import _convert_graph_to_smiles
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
    except Exception:
        return None
    try:
        smi, _, _ = _convert_graph_to_smiles([[0.3, 0.5], [0.7, 0.5]], ['C', token], [[0, 1], [1, 0]])
    except Exception:
        return False
    return bool(smi) and '*' not in smi and Chem.MolFromSmiles(smi) is not None


def composite_candidates(symbol, name, value, taken=()):
    """Rewrites of a composite token that embeds the defined variable: [SO2Ar] with Ar = 4-MeOC6H4
    gives [SO2(4-MeOC6H4)] and [SO24-MeOC6H4]; [NR2] with R = Me gives [N(Me)2] and [NMe2]. The
    caller keeps the first one Graph2SMILES accepts. A whole-token variable is never rewritten, the
    variable must not be followed by a lowercase letter or a prime (Ar in [Ar]/[Arx], R in [R']), and
    a trailing number is allowed only when name+number is not itself a defined variable ([NR2] with
    both R and R2 defined is left alone)."""
    inner = symbol[1:-1]
    if inner == name or VARIABLE.fullmatch(symbol):
        return []
    for m in re.finditer(r'(?<![a-z])' + re.escape(name) + r"(\d*)(?![a-z'])", inner):
        if m.group(1) and name + m.group(1) in taken:
            continue
        pre, suf = inner[:m.start()], inner[m.start() + len(name):]
        return [f'[{pre}({value}){suf}]', f'[{pre}{value}{suf}]']
    return []


def composite_variable(symbol, name):
    """True when symbol is a composite label with the variable name embedded in it ([OR] or [CO2R]
    for R, [SO2Ar] for Ar), so an expansion group can splice each variant's value into it."""
    return (isinstance(symbol, str) and LABEL.fullmatch(symbol) is not None and not VARIABLE.fullmatch(symbol)
            and VARIABLE.fullmatch(f'[{name}]') is not None and bool(composite_candidates(symbol, name, 'H')))


# Variable names looked for inside composite labels when listing them for the model. Only the R and Ar
# families: single-letter names (A, E, X, Q) also occur in ordinary abbreviations ([DMAP], [SEM], [XPhos]).
COMPOSITE_NAME = re.compile(r"(?<![a-z])(Ar\d*|R\d*)(?![a-z'])")


def composite_variable_names(symbol):
    """Variable names (R family, Ar family) embedded in a composite label: [OR] -> ['R'], [CO2R1] -> ['R1']."""
    if not isinstance(symbol, str) or not LABEL.fullmatch(symbol) or VARIABLE.fullmatch(symbol) or re.search(r'[+\-]', symbol):
        return []
    names = dict.fromkeys(m.group(1) for m in COMPOSITE_NAME.finditer(symbol[1:-1]))
    return [n for n in names if composite_variable(symbol, n)]


def smiles_parses(smiles):
    """True/False when RDKit is available and the tool SMILES is a string; None otherwise."""
    if not isinstance(smiles, str) or not smiles or smiles == '<invalid>':
        return False if smiles == '<invalid>' else None
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
    except Exception:
        return None
    return Chem.MolFromSmiles(smiles) is not None
# No alias table: the model transcribes, the downstream Graph2SMILES tables interpret.


def unwrap(data):
    data = data.get('result', data) if isinstance(data, dict) else data
    if isinstance(data, list):
        if len(data) != 1:
            raise PlanError('Exactly one image result is required')
        data = data[0]
    if not isinstance(data, dict) or not isinstance(data.get('bboxes'), list):
        raise PlanError('Expected a bboxes result object')
    return data


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def editable(symbol):
    if not isinstance(symbol, str) or not symbol.startswith('[') or not symbol.endswith(']') or symbol in PROTECTED or '@' in symbol:
        return False
    atomic = ATOMIC.fullmatch(symbol)
    return bool(VARIABLE.fullmatch(symbol)) or not (atomic and (atomic[1] in ELEMENTS or atomic[1] in set('bcnops')))


def catalog(data):
    """Immutable ids for the LLM: molecules, their label atoms and text boxes.

    Only the fields the model needs are exposed (category, bbox, symbols, text);
    graph fields such as coords/edges/molfile never reach the prompt."""
    source = unwrap(data)
    mols, texts, atoms = {}, {}, {}
    for i, box in enumerate(source['bboxes']):
        if box.get('category') == '[Mol]':
            mid = f'mol_{i:03d}'
            labels = []
            for j, symbol in enumerate(box.get('symbols', [])):
                aid = f'{mid}:a{j:03d}'
                if editable(symbol):
                    labels.append({'atom_id': aid, 'symbol': symbol})
                    atoms[aid] = (i, j)
            symbols = list(box.get('symbols', []))
            degs = degrees(box)
            coords = box.get('coords') or []
            hints = []
            for j, symbol in enumerate(symbols):
                d = degs[j] if degs is not None and j < len(degs) else None
                if symbol != 'C' or symbol == '*' or (d is not None and d <= 1):
                    h = {'atom_id': f'{mid}:a{j:03d}', 'symbol': symbol}
                    if d is not None:
                        h['degree'] = d
                    if j < len(coords) and isinstance(coords[j], (list, tuple)) and len(coords[j]) >= 2:
                        h['x'], h['y'] = round(float(coords[j][0]), 3), round(float(coords[j][1]), 3)
                    hints.append(h)
            cands = charge_candidates(symbols, degs, valences(box))
            isolated = [j for j, d in enumerate(degs or []) if d == 0]
            mols[mid] = {'molecule_id': mid, 'source_bbox_index': i, 'bbox': box.get('bbox'),
                         'smiles': box.get('smiles'), 'smiles_parses': smiles_parses(box.get('smiles')),
                         'label_atoms': labels, 'symbols': symbols, 'atom_hints': hints,
                         'charge_candidates': [f'{mid}:a{j:03d}' for j in cands],
                         'isolated_atoms': [f'{mid}:a{j:03d}' for j in isolated]}
        elif box.get('category') == '[Idt]':
            tid = f'text_{i:03d}'
            texts[tid] = {'text_id': tid, 'source_bbox_index': i, 'bbox': box.get('bbox'), 'text': box.get('text', [])}
    cat = {'source_sha256': digest({'bboxes': [{k: b.get(k) for k in ('category', 'bbox', 'symbols', 'text')} for b in source['bboxes']],
                                    'corefs': source.get('corefs', [])}),
           'molecules': list(mols.values()), 'texts': list(texts.values()),
           'corefs': [list(c) for c in source.get('corefs', [])],
           'variable_molecule_ids': [mid for mid, m in mols.items() if any(VARIABLE.fullmatch(a['symbol']) for a in m['label_atoms'])],
           'composite_variable_molecule_ids': [mid for mid, m in mols.items() if any(composite_variable_names(a['symbol']) for a in m['label_atoms'])]}
    return cat, mols, texts, atoms


def clean(value):
    """Printed spelling with Unicode digits, spaces and TeX braces removed; no alias mapping."""
    value = value.translate(str.maketrans('₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹', '01234567890123456789'))
    return value.strip().strip('[]').replace(' ', '').replace('^', '').replace('{', '').replace('}', '')


def norm(value):
    """Kept for callers; identical to clean() now that no alias mapping is applied."""
    return clean(value)


def row_defines(row, name, literal):
    """Does the printed row assign `literal` to `name`? Accepts "R1 = F" and chained
    equalities "R2 = R3 = H" (every member gets H)."""
    row = row.replace('¹', '1').replace('²', '2').replace('^', '').replace('{', '').replace('}', '')
    if re.search(r'(?<![A-Za-z0-9])' + re.escape(name) + r'\s*=\s*' + re.escape(literal) + r'(?=\s|[,;.:)\]]|$)', row):
        return True
    # value-first rows of a shared-placeholder list: "R = tBu: 3i, 84%; OMe: 3j, 84%" gives rows like "OMe: 3j, 84%"
    if re.match(r'\s*' + re.escape(literal) + r'(?=\s|[,;.:)\]]|$)', row):
        return True
    for m in re.finditer(r"((?:[A-Za-z][A-Za-z0-9']*\s*=\s*)+)([^\s,;:()\]]+)", row):
        members = re.findall(r"[A-Za-z][A-Za-z0-9']*", m.group(1))
        if name in members and m.group(2).strip() == literal.strip():
            return True
    return False


def row_defines_positional(row, compound_id, names, name, literal):
    """Does the printed row assign `literal` to `name` by position rather than by equation? A figure that heads
    its list with the variables in order ("7 (X, Y)") writes each member as values in that same order
    ("7a (Me, H)"): the k-th value belongs to the k-th variable. `names` is the group's variable order."""
    if name not in names:
        return False
    m = re.search(r'(?<![A-Za-z0-9])' + re.escape(unprime(str(compound_id))) + r'\s*\(([^()]*)\)', unprime(row))
    if m is None:
        return False
    values = [v.strip() for v in m.group(1).split(',')]
    if len(values) != len(names):
        return False
    return clean(values[names.index(name)]) == clean(literal)


def literal_matches(literal, value, audit, **context):
    """Transcription only: the replacement token must be the printed right-hand side
    itself (spaces, TeX braces and Unicode digits normalised). No abbreviation,
    expansion, alias or name interpretation by the model: Graph2SMILES resolves
    printed names and abbreviations through its own tables."""
    return clean(literal) == clean(value)


# Free counter-ions a figure prints beside a charged molecule. Printed spelling -> the label token the
# Graph2SMILES table expands (molnextr.constants). Only anions of salts that are drawn as separate labels.
COUNTER_IONS = {
    'BF4-': '[BF4-]', 'BF4': '[BF4-]', 'HBF4': '[BF4-]', 'HBF': '[BF4-]', 'A-HBF4': '[BF4-]', '20BF4': '[BF4-]', '38F4': '[BF4-]',   # the OCR spellings molnextr.constants also maps
    'PF6-': '[PF6-]', 'PF6': '[PF6-]',
    'SbF6-': '[SbF6-]', 'SbF6': '[SbF6-]',
    'OTf-': '[OTf-]', 'OTf': '[OTf-]', 'TfO-': '[OTf-]', 'TfO': '[OTf-]',
    'NTf2-': '[NTf2-]', 'NTf2': '[NTf2-]', 'Tf2N-': '[NTf2-]',
    'ClO4-': '[ClO4-]', 'ClO4': '[ClO4-]',
    'Cl-': '[Cl-]', 'Br-': '[Br-]', 'I-': '[I-]', 'F-': '[F-]',
    'MeOSO3-': '[MeOSO3-]', 'MeSO4-': '[MeOSO3-]', 'OSO3Me': '[MeOSO3-]', 'MeOSO3': '[MeOSO3-]', 'SO4Me-': '[MeOSO3-]', 'SO4Me': '[MeOSO3-]',
    'OSO3Me-': '[MeOSO3-]', 'MeSO4': '[MeOSO3-]', 'CH3OSO3-': '[MeOSO3-]', 'CH3SO4-': '[MeOSO3-]',
    'OTs-': '[OTs-]', 'TsO-': '[OTs-]', 'OTs': '[OTs-]', 'NO3-': '[NO3-]', 'NO3': '[NO3-]', 'OAc-': '[OAc-]', 'AcO-': '[OAc-]',
}


def counter_ion_token(printed):
    """The label token for a printed counter-ion spelling, None when it is not a known free anion."""
    key = re.sub(r'[\s−⁻]', '-', str(printed or '')).replace('--', '-').strip('[]').replace('(-)', '-')
    key = key.replace('–', '-')
    key = key.lstrip('.-')            # ".BF4-", "-BF4" (a leading minus sign read as a dash)
    return COUNTER_IONS.get(key) or COUNTER_IONS.get(key.rstrip('-')) if key else None


def add_counter_ion_node(box, token):
    """Append one isolated atom carrying the counter-ion label to a full graph box (coords, symbols,
    edges, atoms, bonds): the molecule becomes a salt when Graph2SMILES expands it. In place."""
    n = len(box.get('symbols') or [])
    box['symbols'] = list(box.get('symbols') or []) + [token]
    coords = [list(c) for c in (box.get('coords') or [])]
    coords.append([0.98, 0.02])
    box['coords'] = coords
    edges = [list(r) + [0] for r in (box.get('edges') or [])]
    edges.append([0] * (n + 1))
    box['edges'] = edges
    if isinstance(box.get('atoms'), list):
        box['atoms'] = list(box['atoms']) + [{'atom_symbol': token, 'x': 0.98, 'y': 0.02}]
    return box


BARE_ELEMENT = re.compile(r'^(?:B|C|N|O|F|P|S|Cl|Br|I|Si|Se)$')
RADICAL_TOKEN = re.compile(r'^\[(?:C|N|O|S|P|CH|CH2|NH|OH)\]$')      # an element in brackets with no charge: a label scrap
# Solvents a figure prints inside or beside a structure's box (canonical RDKit SMILES).
SOLVENT_FRAGMENTS = {'CCOC(C)=O', 'C1CCOC1', 'ClCCl', 'Cc1ccccc1', 'CS(C)=O', 'CC#N', 'ClC(Cl)Cl', 'CCOCC', 'COCCOC',
                     'C1COCCO1', 'CO', 'CCO', 'CC(C)=O', 'c1ccccc1', 'ClCCCl', 'CN(C)C=O', 'CC(C)O', 'CCCCCC', 'CCCCC',
                     'O=C1CCCN1C', 'Cc1ccc(C)cc1', 'CC(=O)O', 'CCCCO', 'C1CCCCC1'}


def _fragment_is_solvent(box, members):
    """True when the sub-graph on `members` is one of the printed solvents."""
    try:
        from molnextr.chemistry import _convert_graph_to_smiles
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
        coords = [box['coords'][i] for i in members]
        symbols = [box['symbols'][i] for i in members]
        edges = [[box['edges'][i][j] for j in members] for i in members]
        smi = _convert_graph_to_smiles(coords, symbols, edges)[0]
        m = Chem.MolFromSmiles(smi) if smi else None
        return m is not None and Chem.MolToSmiles(m) in SOLVENT_FRAGMENTS
    except Exception:
        return False


DIRT_TOKEN = re.compile(r"\[?[.,:;'`·•∙]+\]?$")      # a speck in the drawing read as an atom


def drop_dirt_atoms(box, min_core=5):
    """Drop disconnected single atoms that are dirt in the drawing rather than chemistry (in place):
    a speck read as "[.]" or "*", or a lone uncharged carbon, beside a real molecule (a bonded core of at
    least ``min_core`` atoms). Graph2SMILES renders them as an extra "*." or "C." fragment, which makes
    every reaction built from that molecule miss. Counter-ions (charged), a free ".Cl" of an HCl salt and
    every bonded atom are kept. Returns the dropped tokens."""
    symbols, edges = box.get('symbols') or [], box.get('edges') or []
    if len(symbols) < 2 or len(edges) != len(symbols):
        return []
    deg = [sum(1 for j, v in enumerate(row) if v and j != i) for i, row in enumerate(edges)]
    if sum(1 for d in deg if d) < min_core:
        return []
    drop = [i for i, d in enumerate(deg)
            if d == 0 and isinstance(symbols[i], str)
            and (symbols[i] == '*' or symbols[i] == 'C' or DIRT_TOKEN.fullmatch(symbols[i]))]
    if not drop or len(drop) >= len(symbols) - 1:
        return []
    keep = [i for i in range(len(symbols)) if i not in drop]
    remap = {old: new for new, old in enumerate(keep)}
    dropped = [symbols[i] for i in drop]
    box['symbols'] = [symbols[i] for i in keep]
    if isinstance(box.get('coords'), list) and len(box['coords']) == len(symbols):
        box['coords'] = [box['coords'][i] for i in keep]
    box['edges'] = [[edges[i][j] for j in keep] for i in keep]
    if isinstance(box.get('atoms'), list) and len(box['atoms']) == len(symbols):
        box['atoms'] = [box['atoms'][i] for i in keep]
    if isinstance(box.get('bonds'), list):
        bonds = []
        for b in box['bonds']:
            ends = b.get('endpoint_atoms') if isinstance(b, dict) else None
            if ends and all(e in remap for e in ends):
                nb = dict(b)
                nb['endpoint_atoms'] = tuple(remap[e] for e in ends)
                bonds.append(nb)
        box['bonds'] = bonds
    return dropped


def tidy_isolated_atoms(box):
    """Drop stray isolated atoms of a molecule box (in place): bare element tokens (a "Br" read from a
    label next to the drawing) when the box has a bonded core, and free anions beyond the number the
    core's positive charges can balance (a BF4- read twice). Returns the dropped tokens."""
    symbols = box.get('symbols') or []
    edges = box.get('edges') or []
    if len(symbols) < 2 or len(edges) != len(symbols):
        return []
    # Connected components. The core is the largest one; a smaller component made only of bare
    # element tokens and free-ion tokens is label noise read as a structure ("B.HBF4" printed beside a
    # salt becomes Br-[HBF]): its internal bonds are cut so its atoms fall under the isolated-atom rules.
    n = len(symbols)
    comp, seen = [], set()
    for i in range(n):
        if i in seen:
            continue
        stack, members = [i], []
        seen.add(i)
        while stack:
            a = stack.pop()
            members.append(a)
            for j in range(n):
                if edges[a][j] and j not in seen:
                    seen.add(j)
                    stack.append(j)
        comp.append(sorted(members))
    core_comp = max(comp, key=len)
    if len(core_comp) < 2:
        return []

    def junk_token(t):
        return isinstance(t, str) and (BARE_ELEMENT.fullmatch(t) is not None
                                       or (t.startswith('[') and (counter_ion_token(t.strip('[]')) is not None or t.endswith('-]'))))

    cut, junk = [], []
    for members in comp:
        if members is core_comp:
            continue
        toks = [symbols[i] if isinstance(symbols[i], str) else '' for i in members]
        if len(members) == 2 and all(junk_token(t) for t in toks) and any(counter_ion_token(t.strip('[]')) for t in toks):
            # "Br-[HBF]" from a printed "B.HBF4": cut the bond, the atoms fall under the isolated-atom rules
            cut.extend(members)
        elif len(members) <= 2 and any(t == '*' or RADICAL_TOKEN.fullmatch(t) for t in toks):
            # "*=[N]", "[CH]": a scrap of a label read as a bonded fragment, never a drawn reactant
            junk.extend(members)
        elif 2 <= len(members) <= 12 and _fragment_is_solvent(box, members):
            # EtOAc / THF / DCM printed inside the box of a reactant: not part of the structure
            junk.extend(members)
    if cut:
        edges = [list(row) for row in edges]
        for i in cut:
            for j in range(n):
                edges[i][j] = edges[j][i] = 0
        box['edges'] = edges
        if isinstance(box.get('bonds'), list):
            box['bonds'] = [b for b in box['bonds'] if not (isinstance(b, dict) and b.get('endpoint_atoms')
                                                              and any(e in cut for e in b['endpoint_atoms']))]
    deg = [sum(1 for v in row if v) for row in edges]
    core = [i for i, d in enumerate(deg) if d > 0]
    if not core:
        return []
    plus = sum(1 for i in core if isinstance(symbols[i], str) and symbols[i].startswith('[') and '+' in symbols[i])
    cut_set = set(cut)
    drop, anions = [], []
    for i, d in enumerate(deg):
        if d > 0:
            continue
        t = symbols[i] if isinstance(symbols[i], str) else ''
        if BARE_ELEMENT.fullmatch(t):
            if i in cut_set:                       # only the remains of a cut label fragment; a free ".Cl" (an HCl salt) stays
                drop.append(i)
        elif t.startswith('[') and counter_ion_token(t.strip('[]')):
            anions.append((0 if i not in cut_set else 1, i))
        elif t.startswith('[') and t.endswith('-]'):
            anions.append((2 if i not in cut_set else 3, i))
    # Free anions beyond what the core's positive charges balance: keep the most credible ones
    # (a known counter-ion token first, one that was drawn free before one cut out of a label).
    if plus and len(anions) > plus:
        anions.sort()
        drop.extend(i for _, i in anions[plus:])
    drop = sorted(set(drop) | set(junk))
    if not drop:
        return [f'cut bonds of {symbols[i]}' for i in cut]
    if len(drop) >= len(symbols) - 1:
        return []
    keep = [i for i in range(len(symbols)) if i not in drop]
    remap = {old: new for new, old in enumerate(keep)}
    dropped = [symbols[i] for i in drop]
    box['symbols'] = [symbols[i] for i in keep]
    if isinstance(box.get('coords'), list) and len(box['coords']) == len(symbols):
        box['coords'] = [box['coords'][i] for i in keep]
    box['edges'] = [[edges[i][j] for j in keep] for i in keep]
    if isinstance(box.get('atoms'), list) and len(box['atoms']) == len(symbols):
        box['atoms'] = [box['atoms'][i] for i in keep]
    if isinstance(box.get('bonds'), list):
        bonds = []
        for b in box['bonds']:
            ends = b.get('endpoint_atoms') if isinstance(b, dict) else None
            if ends and all(e in remap for e in ends):
                nb = dict(b)
                nb['endpoint_atoms'] = tuple(remap[e] for e in ends)
                bonds.append(nb)
        box['bonds'] = bonds
    return dropped


# ---- ring double-bond repair --------------------------------------------------------------------------
RING_VALENCE = {'C': 4, 'N': 3, 'P': 3, 'O': 2, 'S': 2, 'B': 3, 'Si': 4}
_BOND_NAME = {1: 'single', 2: 'double', 3: 'triple', 4: 'aromatic', 5: 'solid wedge', 6: 'dashed wedge'}
_PLAIN_ATOM = re.compile(r"\[?([A-Z][a-z]?)(H(\d?))?\]?$")


def neutral_atom(symbol):
    """(element, explicit H count) for a neutral plain-element token (C, N, [NH], [CH2]); None for labels,
    ions, wildcards and elements without a fixed neutral valence."""
    if not isinstance(symbol, str) or CHARGED.fullmatch(symbol) or symbol.startswith('[') != symbol.endswith(']'):
        return None
    m = _PLAIN_ATOM.fullmatch(symbol)
    if not m or m.group(1) not in RING_VALENCE or (m.group(2) and not symbol.startswith('[')):
        return None
    h = int(m.group(3)) if m.group(3) else (1 if m.group(2) else 0)
    return m.group(1), h


def _ring_bonds_per_ring(edges):
    """The bond sets of the SSSR rings of the tool's graph (RDKit ring perception on a dummy graph)."""
    from rdkit import Chem
    n = len(edges)
    m = Chem.RWMol()
    for _ in range(n):
        m.AddAtom(Chem.Atom(6))
    for i in range(n):
        for j in range(i + 1, n):
            if edges[i][j]:
                m.AddBond(i, j, Chem.BondType.SINGLE)
    rings = []
    for ring in Chem.GetSymmSSSR(m):            # smallest rings: FastFindRings may return a fused envelope instead
        members = set(int(a) for a in ring)
        rings.append(sorted((i, j) for i in members for j in members if i < j and edges[i][j]))
    return rings


def repair_ring_bonds(box, max_ring=8, max_changes=4):
    """Move misplaced double bonds inside one ring so that no neutral ring atom exceeds its valence (a
    pyrazole the tool wrote as *C1=NN(*)=CC1 becomes *c1ccn(*)n1). One SSSR ring at a time, never across a
    fused system; only rings drawn with plain single/double bonds; the number of double bonds in the ring is
    kept; the new assignment must leave every ring atom sp2-like (a double bond in or out of the ring, or a
    heteroatom lone pair), i.e. the ring reads as an aromatic candidate; the assignment closest to the
    tool's wins. Nothing changes when no such assignment exists: an azolium or pyridinium drawn without
    its charge sign stays as read, so the plan's charge step still applies. Edits edges/bonds in place;
    returns the list of (i, j, old_order, new_order)."""
    import itertools
    symbols, edges = box.get('symbols'), box.get('edges')
    if not isinstance(symbols, list) or not isinstance(edges, list) or len(edges) != len(symbols):
        return []
    n = len(symbols)
    order = {1: 1.0, 2: 2.0, 3: 3.0, 4: 1.5, 5: 1.0, 6: 1.0}
    atom = [neutral_atom(s) for s in symbols]
    hetero = [bool(atom[i]) and atom[i][0] in ('N', 'O', 'S', 'P') for i in range(n)]

    def valence(i):
        return sum(order.get(edges[i][j], 1.0) for j in range(n) if j != i and edges[i][j]) + (atom[i][1] if atom[i] else 0)

    def over_valent():
        return [i for i in range(n) if atom[i] and valence(i) > RING_VALENCE[atom[i][0]] + 1e-6]

    bad = over_valent()
    if not bad:
        return []
    try:
        rings = _ring_bonds_per_ring(edges)
    except Exception:
        return []
    changed = []
    for bonds in rings:
        members = sorted({a for b in bonds for a in b})
        if not any(i in members for i in bad) or len(bonds) > max_ring:
            continue
        if any(edges[a][b] not in (1, 2) for a, b in bonds):
            continue
        k = sum(1 for a, b in bonds if edges[a][b] == 2)
        if k == 0:
            continue
        incident = {i: [c for c, (a, b) in enumerate(bonds) if i in (a, b)] for i in members}
        base = {i: valence(i) - sum(order[edges[bonds[c][0]][bonds[c][1]]] for c in incident[i]) for i in members}
        # multiple bonds outside this ring (exocyclic C=O, a fused ring's double bond) already make the atom sp2
        outside_double = {i: any(edges[i][j] in (2, 3) for j in range(n) if j != i and (min(i, j), max(i, j)) not in bonds)
                          for i in members}
        limit = {i: (RING_VALENCE[atom[i][0]] if atom[i] else None) for i in members}
        best = None
        for chosen in itertools.combinations(range(len(bonds)), k):
            used = set()
            ok = True
            for c in chosen:
                a, b = bonds[c]
                if a in used or b in used:
                    ok = False
                    break
                used.update((a, b))
            if not ok:
                continue
            for i in members:
                if limit[i] is not None and base[i] + sum(2.0 if c in chosen else 1.0 for c in incident[i]) > limit[i] + 1e-6:
                    ok = False
                    break
                if not (i in used or outside_double[i] or hetero[i]):
                    ok = False          # a saturated ring atom would remain: not an aromatic candidate
                    break
            if not ok:
                continue
            diff = sum(1 for c, (a, b) in enumerate(bonds) if (2 if c in chosen else 1) != edges[a][b])
            if diff <= max_changes and (best is None or diff < best[0]):
                best = (diff, chosen)
        if best is None or best[0] == 0:
            continue
        for c, (a, b) in enumerate(bonds):
            new = 2 if c in best[1] else 1
            if new != edges[a][b]:
                changed.append((a, b, edges[a][b], new))
                edges[a][b] = edges[b][a] = new
        bad = over_valent()
        if not bad:
            break
    if changed and isinstance(box.get('bonds'), list):
        by_ends = {(min(a, b), max(a, b)): new for a, b, _o, new in changed}
        for bd in box['bonds']:
            ends = bd.get('endpoint_atoms') if isinstance(bd, dict) else None
            if ends and len(ends) == 2 and (min(ends), max(ends)) in by_ends:
                bd['bond_type'] = _BOND_NAME[by_ends[(min(ends), max(ends))]]
    return changed

PRIMES = "'′’´"          # ASCII apostrophe, prime, right single quote, acute accent


def unprime(text):
    """The text with every prime-like mark written as the ASCII apostrophe, so an identifier printed as 2a′
    still matches the OCR's 2a'."""
    if not isinstance(text, str):
        return text
    for ch in PRIMES[1:]:
        text = text.replace(ch, "'")
    return text


def require(condition, message):
    if not condition:
        raise PlanError(message, entry=_CURRENT_ENTRY)


def process(data, plan):
    """All validation is local/transactional. Source is never mutated; no partial result on error."""
    try:
        jsonschema.Draft202012Validator(SCHEMA).validate(plan)
    except jsonschema.ValidationError as exc:
        raise PlanError(f'Schema error at {list(exc.path)}: {exc.message}') from exc
    source = unwrap(data)
    cat, mols, texts, atoms = catalog(source)
    edited = copy.deepcopy(source['bboxes'])
    audit, touched, applied = [], set(), {}

    def placeholders_elsewhere(i):
        """Placeholder tokens the figure draws on its other molecules ([R1], [Ar2] ...)."""
        out = set()
        for k, box in enumerate(edited):
            if k == i or box.get('category') != '[Mol]':
                continue
            out.update(s for s in box.get('symbols', []) or [] if isinstance(s, str) and VARIABLE.fullmatch(s))
        return out

    def atom(mid, aid):
        require(mid in mols, f'Unknown molecule {mid}')
        require(aid in atoms and aid.startswith(mid + ':'), f'Unknown/non-label/wrong-owner atom {aid}')
        return atoms[aid]

    # 1. OCR corrections at label atoms. A correction filed here but aimed at a non-label atom
    # (a rare-element lookalike such as [Pr], or a bare atom / *) is handled by the atom-level
    # rules of step 1b instead of rejecting the plan; the kind is inferred and audited.
    rerouted = []
    for _k, patch in enumerate(plan['ocr_corrections']):
        _entering('ocr_corrections', _k)
        mid, aid = patch['molecule_id'], patch['atom_id']
        if mid in mols and aid not in atoms:
            m = re.fullmatch(re.escape(mid) + r':a(\d{3,})', aid)
            i = mols[mid]['source_bbox_index']
            if m is not None and int(m.group(1)) < len(edited[i]['symbols']):
                src = edited[i]['symbols'][int(m.group(1))]
                kind = None
                if re.fullmatch(r'\[([A-Z][a-z]?)\]', src) and re.fullmatch(r'\[([A-Z][a-z]?)\]', src).group(1) in RARE_ELEMENTS:
                    kind = 'lookalike'
                elif BARE_ATOM.fullmatch(src):
                    kind = 'label_from_atom'
                elif CHARGED.fullmatch(patch['corrected_symbol']) and element_of(src) == element_of(patch['corrected_symbol']):
                    kind = 'charge'
                elif CHARGED.fullmatch(src) and '-' in src and CHARGED.fullmatch(patch['corrected_symbol']) and '-' in patch['corrected_symbol']:
                    kind = 'anion_element'      # an isolated [O-] printed as Cl-: the element of a free anion was misread
                if kind is not None:
                    rerouted.append({**patch, 'kind': kind, '_ocr_index': _k})
                    audit.append({'operation': 'reroute_ocr_to_atom_correction', **patch, 'kind': kind})
                    continue
        i, j = atom(patch['molecule_id'], patch['atom_id'])
        value = patch['corrected_symbol']
        if (i, j) in touched:
            # the same atom filed twice, usually once per section: a repeat of what is already applied is
            # dropped, a second and different value is a real conflict
            require(applied.get((i, j)) == value, f'Conflicting corrections at {i}:{j}')
            audit.append({'operation': 'duplicate_correction_dropped', **patch})
            continue
        require(edited[i]['symbols'][j] == patch['expected_symbol'], 'OCR expected_symbol mismatch')
        require(LABEL.fullmatch(value) and editable(value), 'Only bracketed label OCR corrections are allowed')
        require(value != patch['expected_symbol'], 'No-op OCR correction')
        edited[i]['symbols'][j] = value
        touched.add((i, j))
        applied[(i, j)] = value
        audit.append({'operation': 'ocr', **patch, 'source_bbox_index': i, 'symbol_index': j})

    # 1b. Atom-level OCR: three mechanically checkable kinds, never an element change
    model_charged = set()
    _leaving()
    for _k, patch in enumerate(list(plan['atom_corrections']) + rerouted):
        _entering('ocr_corrections', patch['_ocr_index']) if '_ocr_index' in patch else _entering('atom_corrections', _k)
        patch = {k: v for k, v in patch.items() if k != '_ocr_index'}
        mid, aid, kind = patch['molecule_id'], patch['atom_id'], patch['kind']
        require(mid in mols, f'Unknown molecule {mid}')
        m = re.fullmatch(re.escape(mid) + r':a(\d{3,})', aid)
        i = mols[mid]['source_bbox_index']
        require(m is not None and int(m.group(1)) < len(edited[i]['symbols']), f'Unknown atom {aid}')
        j = int(m.group(1))
        src, dst = patch['expected_symbol'], patch['corrected_symbol']
        if (i, j) in touched:
            require(applied.get((i, j)) == dst, f'Conflicting corrections at {i}:{j}')
            audit.append({'operation': 'duplicate_correction_dropped', **patch})
            continue
        require(edited[i]['symbols'][j] == src, 'atom correction expected_symbol mismatch')
        require(dst != src, 'No-op atom correction')
        degs = degrees(source['bboxes'][i])
        if editable(src) and kind != 'charge':
            # a label atom filed under atom_corrections: its symbol is not an element, so this is the plain
            # OCR correction of step 1 whatever kind the model picked ([P2] -> [R2], [Rl] -> [R1], [3*] -> [R3])
            require(LABEL.fullmatch(dst) and editable(dst), 'Only bracketed label OCR corrections are allowed')
            edited[i]['symbols'][j] = dst
            touched.add((i, j))
            applied[(i, j)] = dst
            atoms[aid] = (i, j)
            audit.append({'operation': 'ocr', **{k: v for k, v in patch.items() if k != 'kind'},
                          'source_bbox_index': i, 'symbol_index': j, 'filed_as': kind})
            continue
        if kind == 'charge':
            model_charged.add(mid)
            require(CHARGED.fullmatch(dst), 'charge correction must produce a charged element token such as [N+], [Cl-], [BF4-]')
            require(element_of(src) == element_of(dst), 'charge correction must keep the element')
            require(degs is not None, 'charge correction needs the graph edges (pass the full vision result)')
            if '+' in dst:
                require(element_of(dst) in CHARGE_BEARERS, 'positive charge only on N, P, S or O')
                require(j in charge_candidates(edited[i]['symbols'], degs), 'positive charge only on an atom whose bonding requires it (highest-degree atom of that element, degree >= 3)')
            else:
                require(degs[j] <= 1, 'negative charge only on a counter-ion or a terminal atom')
        elif kind == 'anion_element':
            require(CHARGED.fullmatch(src) and '-' in src and CHARGED.fullmatch(dst) and '-' in dst, 'anion_element needs charged single-element tokens on both sides')
            require(degs is not None and degs[j] == 0, 'anion_element only on an isolated atom (a free counter-ion)')
        elif kind == 'label_from_atom':
            require(BARE_ATOM.fullmatch(src), 'label_from_atom source must be a bare atom, * or [n*]')
            require(LABEL.fullmatch(dst) and editable(dst) and not CHARGED.fullmatch(dst), 'label_from_atom target must be a bracketed label, not an element or charge token')
            require(degs is not None and degs[j] <= 1, 'label_from_atom only on a terminal atom (a collapsed text label has one bond)')
        else:  # lookalike
            m2 = re.fullmatch(r'\[([A-Z][a-z]?)\]', src)
            require(m2 is not None, 'lookalike source must be a bracketed element token such as [Re], [Pr], [Ti]')
            require(LABEL.fullmatch(dst) and editable(dst), 'lookalike target must be a bracketed label')
            if m2.group(1) not in RARE_ELEMENTS:
                # A common element is structural data: Li, Ti, B and Si do occur as drawn atoms. It is still a
                # misread placeholder when it is a metal on a terminal atom and is corrected to a placeholder the
                # figure draws on another molecule but not on this one (R1 read as [Ti] on one side of a scheme
                # whose other side carries R1). Everything else keeps the element.
                require(m2.group(1) in MISREAD_METALS, 'only a metal token is a lookalike among the common elements; H, halogens, B, Si, Sn, P, S stay')
                require(VARIABLE.fullmatch(dst), 'a common element is only corrected to a placeholder such as [R1] or [Ar]')
                require(degs is not None and degs[j] <= 1, 'a common element is only corrected on a terminal atom')
                require(dst in placeholders_elsewhere(i), f'{dst} is drawn on no other molecule of this figure')
        edited[i]['symbols'][j] = dst
        touched.add((i, j))
        applied[(i, j)] = dst
        if editable(dst):
            atoms[aid] = (i, j)      # a recovered label / placeholder can now be defined or expanded
        audit.append({'operation': 'atom_ocr', **patch, 'source_bbox_index': i, 'symbol_index': j})

    # 1c. Program-side charge placement. The model transcribes WHICH molecules carry a drawn
    # + or - sign (charged_molecules, or an explicit charge correction); the program decides the
    # atom, which is chemistry, not vision:
    #   * the tool's SMILES is invalid, no positive charge is present, and the atoms whose bonding
    #     needs the charge are ring-equivalent -> put [El+] on the first of them;
    #   * a molecule carrying a positive charge has an isolated halide / common anion -> [X-].
    # An invalid tool SMILES alone is not enough: bond-order misreads look the same, so nothing is
    # placed in molecules the model did not report as charged.
    _leaving()
    charged = set(plan['charged_molecules'])
    require(charged <= set(mols), f'Unknown molecule in charged_molecules: {sorted(charged - set(mols))}')
    program_fixes = []
    for mid, mol in mols.items():
        if mid not in charged and mid not in model_charged:
            continue
        i = mol['source_bbox_index']
        box = source['bboxes'][i]
        syms = edited[i]['symbols']
        degs = degrees(box)
        if degs is None:
            continue
        touched_here = {jj for (ii, jj) in touched if ii == i}
        has_plus = any(CHARGED.fullmatch(t) and '+' in t for t in syms)
        if not has_plus and smiles_parses(box.get('smiles')) is False:
            cands = charge_candidates(syms, degs, valences(box))
            if cands and not (set(cands) & touched_here):
                if len(cands) == 1 or ring_equivalent(cands, ring_sets(box)):
                    j = cands[0]
                    syms[j] = f'[{element_of(syms[j])}+]'
                    touched.add((i, j))
                    has_plus = True
                    fix = {'operation': 'auto_charge', 'molecule_id': mid, 'source_bbox_index': i, 'symbol_index': j,
                           'candidates': cands, 'reason': 'tool SMILES invalid; over-valent charge bearer' if valences(box) and any(valences(box)[k] > NEUTRAL_VALENCE.get(element_of(syms[k]), 9) + 1e-6 for k in cands) else 'tool SMILES invalid; highest-degree charge bearer; candidates ring-equivalent'}
                else:
                    fix = {'operation': 'charge_needed_ambiguous', 'molecule_id': mid, 'source_bbox_index': i, 'candidates': cands,
                           'reason': 'tool SMILES invalid but the candidate atoms are not ring-equivalent; needs the drawn position'}
                program_fixes.append(fix)
                audit.append(fix)
        if has_plus:
            for j, t in enumerate(syms):
                if degs[j] == 0 and (i, j) not in touched and is_anion_token(t):
                    syms[j] = '[' + clean(t) + '-]'
                    touched.add((i, j))
                    fix = {'operation': 'auto_counter_ion', 'molecule_id': mid, 'source_bbox_index': i, 'symbol_index': j,
                           'reason': 'isolated anion token in a molecule that carries a positive charge'}
                    program_fixes.append(fix)
                    audit.append(fix)

    # 2. Text corrections
    touched_text = set()
    for _k, patch in enumerate(plan['text_corrections']):
        _entering('text_corrections', _k)
        tid = patch['text_id']
        require(tid in texts and tid not in touched_text, f'Unknown/duplicate text ID {tid}')
        require(patch['corrected_text'], 'Text correction cannot erase text')
        i = texts[tid]['source_bbox_index']
        audit.append({'operation': 'text_ocr', 'original_text': copy.deepcopy(edited[i].get('text', [])), **patch})
        edited[i]['text'] = copy.deepcopy(patch['corrected_text'])
        touched_text.add(tid)

    # 3. Single-valued definitions: in-place whole-token substitution, no cloning
    equations = []
    defined = {}
    substituted_mols = set()
    definition_scopes = {}
    defined_names = {d['name'] for d in plan['definitions']}
    _leaving()
    for _k, definition in enumerate(plan['definitions']):
        _entering('definitions', _k)
        name = definition['name']
        token = f'[{name}]'
        require(VARIABLE.fullmatch(token), f'Unsupported definition variable {name}')
        scope_set = set(definition['scope']) or set(mols)
        for previous in definition_scopes.get(name, []):
            require(not (previous & scope_set), f'Conflicting definitions for {name}: two values for one placeholder of one template are an expansion, not definitions')
        definition_scopes.setdefault(name, []).append(scope_set)
        tid = definition['source_text_id']
        require(tid is None or tid in texts, 'Unknown source_text_id in definition')
        raw_value = definition['replacement_symbol']
        value = '[' + clean(raw_value) + ']'
        if value != raw_value:
            audit.append({'operation': 'normalize_definition', 'name': name, 'original': raw_value, 'normalized': value})
        require(LABEL.fullmatch(value) and '@' not in value and not VARIABLE.fullmatch(value), 'Invalid or unresolved definition symbol')
        require(literal_matches(definition['literal_value'], value, audit, name=name), 'replacement_symbol must be the printed value verbatim (definition)')
        scope = definition['scope']
        require(len(set(scope)) == len(scope) and all(m in mols for m in scope), 'Unknown/duplicate molecule in definition scope')
        targets = scope or list(mols)
        hits = 0
        for mid in targets:
            i = mols[mid]['source_bbox_index']
            for j, symbol in enumerate(edited[i]['symbols']):
                if symbol == token:
                    require((i, j) not in defined, f'Conflicting definitions for {name} at {mid}; two values for one placeholder of one template are an expansion, not definitions')
                    edited[i]['symbols'][j] = value
                    defined[(i, j)] = value
                    substituted_mols.add(mid)
                    hits += 1
                    audit.append({'operation': 'define', 'molecule_id': mid, 'source_bbox_index': i, 'symbol_index': j, 'name': name, 'value': value})
        require(hits > 0 or not scope, f'Definition of {name} matches no atom in its explicit scope')
        # Composite tokens that embed the variable ([SO2Ar], [OAr], [CH2Ar]) get the value
        # spliced in, but only when Graph2SMILES can expand the result without a wildcard.
        inner_value = clean(raw_value)
        for mid in targets:
            i = mols[mid]['source_bbox_index']
            for j, symbol in enumerate(edited[i]['symbols']):
                if (i, j) in defined or (i, j) in touched or not isinstance(symbol, str) or not symbol.startswith('['):
                    continue
                for cand in composite_candidates(symbol, name, inner_value, taken=defined_names):
                    if composite_token_ok(cand):
                        audit.append({'operation': 'define_composite', 'molecule_id': mid, 'source_bbox_index': i, 'symbol_index': j,
                                      'name': name, 'original': symbol, 'value': cand})
                        edited[i]['symbols'][j] = cand
                        defined[(i, j)] = cand
                        substituted_mols.add(mid)
                        break
        equations.append(f'{name} = {definition["literal_value"]}')

    # 4. Expansion groups (joint rows, cloned per variant)
    groups, compound_ids = {}, set()
    _leaving()
    for _k, group in enumerate(plan['groups']):
        _entering('groups', _k)
        mid = group['molecule_id']
        require(mid in mols and mid not in groups, f'Unknown/duplicate expansion group {mid}')
        tid = group['source_text_id']
        require(tid is None or tid in texts, 'Unknown source_text_id')
        require(group['variables'] and group['variants'], 'Empty variable or variant list')
        # One variable name may sit at several atoms of the molecule (the same
        # substituent drawn twice); every listed atom gets the row's value.
        # An atom is either the variable itself ([R]) or a composite label that embeds it ([OR]);
        # composite atoms get each row's value spliced in below.
        positions, used_atoms, composite_at = {}, set(), {}
        # the order the model lists the variables in is the order the figure's header prints them
        # ("7 (X, Y)"), which is what a row written as "7a (Me, H)" assigns by position
        declared = list(dict.fromkeys(v['name'] for v in group['variables']))
        for variable in group['variables']:
            i, j = atom(mid, variable['atom_id'])
            name = variable['name']
            symbol = variable['expected_symbol']
            require(j not in used_atoms, 'Duplicate atom in group')
            require(edited[i]['symbols'][j] == symbol, 'Variable precondition failed after OCR/definitions')
            if not (symbol == f'[{name}]' and VARIABLE.fullmatch(symbol)):
                require(composite_variable(symbol, name), 'Variable name/symbol mismatch or unsupported variable')
                composite_at[j] = symbol
            positions.setdefault(name, []).append(j)
            used_atoms.add(j)
        i = mols[mid]['source_bbox_index']
        require({j for j, s in enumerate(edited[i]['symbols']) if VARIABLE.fullmatch(s)} == used_atoms - set(composite_at),
                'Expansion must cover every local variable atom; unresolved nesting requires review')
        cooked = []
        for variant in group['variants']:
            cid = variant['compound_id']
            # a printed identifier may carry a prime (2a', 3aa'), in either the ASCII or the typographic form;
            # rejecting those cost every variant of the figure
            require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.\-" + PRIMES + r"]*", cid), 'Invalid compound ID')
            require(cid not in compound_ids, f'Duplicate compound ID {cid}')
            require(re.search(r'(?<![A-Za-z0-9])' + re.escape(unprime(cid)) + r'(?![A-Za-z0-9])', unprime(variant['source_text'])),
                    'Compound ID absent from source_text')
            bindings = {}
            for binding in variant['bindings']:
                name, value = binding['name'], binding['replacement_symbol']
                require(name in positions and name not in bindings, 'Unknown/duplicate binding name')
                raw_value = value
                value = '[' + clean(value) + ']'
                if value != raw_value:
                    audit.append({'operation': 'normalize_binding', 'compound_id': cid, 'name': name, 'original': raw_value, 'normalized': value})
                require(LABEL.fullmatch(value) and '@' not in value and not VARIABLE.fullmatch(value), 'Invalid or unresolved replacement symbol')
                require(literal_matches(binding['literal_value'], value, audit, compound_id=cid, name=name), 'replacement_symbol must be the printed value verbatim')
                # Mechanical corroboration, not a substitute for visual verification.
                literal = binding['literal_value']
                require(row_defines(variant['source_text'], name, literal)
                        or row_defines_positional(variant['source_text'], cid, declared, name, literal),
                        'Binding lacks explicit equation in source_text')
                bindings[name] = value
                equations.append(f'{cid}: {name} = {literal}')
            require(set(bindings) == set(positions), 'Every variant must bind all local variables once')
            spliced = {}
            for name, value in bindings.items():
                for j in positions[name]:
                    if j not in composite_at:
                        continue
                    token = next((cand for cand in composite_candidates(composite_at[j], name, value[1:-1], taken=set(positions) | defined_names)
                                  if composite_token_ok(cand)), None)
                    require(token is not None, f'Composite label {composite_at[j]} cannot take {name} = {value} for {cid}')
                    spliced[j] = token
                    audit.append({'operation': 'expand_composite', 'compound_id': cid, 'molecule_id': mid, 'symbol_index': j,
                                  'name': name, 'original': composite_at[j], 'value': token})
            compound_ids.add(cid)
            cooked.append((variant, bindings, spliced))
        groups[mid] = (group, positions, cooked)

    # 5. Decisions: required for every molecule that still carries a variable
    decisions = {}
    _leaving()
    for _k, decision in enumerate(plan['decisions']):
        _entering('decisions', _k)
        mid = decision['molecule_id']
        require(mid in mols and mid not in decisions, 'Unknown/duplicate decision')
        require((decision['action'] == 'expand') == (mid in groups), 'Decision/group disagreement')
        require(decision['action'] != 'substitute' or mid in substituted_mols, 'substitute decision on a molecule no definition touched')
        decisions[mid] = decision
    required_decisions = {mid for mid, m in mols.items()
                          if any(VARIABLE.fullmatch(s) for s in edited[m['source_bbox_index']]['symbols'])}
    require(required_decisions <= set(decisions), f'Missing decisions for variable-bearing molecules: {sorted(required_decisions - set(decisions))}')
    require(set(groups) <= set(decisions), 'Expansion group lacks decision')
    # 7. Counter-ions the figure prints beside a charged molecule (BF4-, PF6-, OTf- ...): recorded per
    # source box; the merge step appends the ion as an isolated atom of every output molecule that comes
    # from that box (the template and, when it was expanded, each variant).
    ions_by_source = {}
    _leaving()
    for _k, entry in enumerate(plan.get('counter_ions') or []):
        _entering('counter_ions', _k)
        mid = entry['molecule_id']
        require(mid in mols, f'Unknown molecule {mid} in counter_ions')
        token = counter_ion_token(entry['ion'])
        require(token is not None, f'Unknown counter-ion {entry["ion"]!r}; known: {sorted(set(COUNTER_IONS))}')
        i = mols[mid]['source_bbox_index']
        require(i not in ions_by_source, f'Duplicate counter-ion for {mid}')
        require(not any(counter_ion_token(str(t).strip('[]')) for t in edited[i]['symbols'] if isinstance(t, str) and t.startswith('[')),
                f'{mid} already carries a counter-ion atom')
        ions_by_source[i] = token
        audit.append({'operation': 'counter_ion', 'molecule_id': mid, 'source_bbox_index': i, 'ion': entry['ion'], 'token': token})
    _leaving()
    for _k, warning in enumerate(plan['structure_warnings']):
        _entering('structure_warnings', _k)
        require(warning['molecule_id'] in mols, 'Unknown structure warning molecule')
    _leaving()

    # 6. Assemble output: unchanged/edited boxes keep their order; expanded molecules become variant + label pairs
    out, links, index_map, provenance = [], [], {}, []
    expanded_sources = {mols[mid]['source_bbox_index'] for mid in groups}
    for i, box in enumerate(edited):
        mid = f'mol_{i:03d}'
        if mid not in groups:
            index_map[i] = len(out)
            out.append(copy.deepcopy(box))
            if i in ions_by_source:
                out[-1]['counter_ion'] = ions_by_source[i]
            provenance.append({'output_index': len(out) - 1, 'source_bbox_index': i})
            continue
        group, positions, variants = groups[mid]
        for variant, bindings, spliced in variants:
            molecule = copy.deepcopy(box)
            for name, value in bindings.items():
                for j in positions[name]:
                    molecule['symbols'][j] = spliced.get(j, value)
            mi = len(out)
            if i in ions_by_source:
                molecule['counter_ion'] = ions_by_source[i]
            out.append(molecule)
            tid = group['source_text_id']
            label = {'category': '[Idt]', 'bbox': copy.deepcopy(texts[tid]['bbox']) if tid else None,
                     'text': [variant['source_text']], 'compound_id': variant['compound_id'],
                     'derived': True, 'bbox_provenance': 'source_text_region' if tid else 'unknown',
                     'source_text_id': tid}
            ti = len(out)
            out.append(label)
            links.append([mi, ti])
            provenance.append({'output_index': mi, 'source_bbox_index': i, 'compound_id': variant['compound_id'],
                               'variant_id': f'{mid}:{variant["compound_id"]}', 'label_index': ti,
                               'bindings': copy.deepcopy(variant['bindings']), 'yield_percent': variant['yield_percent'],
                               'ee_percent': variant['ee_percent']})
            audit.append({'operation': 'instantiate', 'source_bbox_index': i, 'output_index': mi,
                          'compound_id': variant['compound_id'], 'positions': positions, 'bindings': bindings})
    for link in source.get('corefs', []):
        require(isinstance(link, list) and len(link) == 2 and all(type(x) is int and 0 <= x < len(edited) for x in link), 'Invalid source coref')
        require({edited[x].get('category') for x in link} == {'[Mol]', '[Idt]'}, 'Source coref must connect molecule and text')
        if any(x in expanded_sources for x in link):
            audit.append({'operation': 'replace_parent_coref', 'original': link})
        else:
            links.append([index_map[x] for x in link])
    reviews = [d for d in decisions.values() if d['action'] == 'review']
    ambiguous = [f for f in program_fixes if f['operation'] == 'charge_needed_ambiguous']
    result = {'extracted_explicit_rgroup_equations(without any reasoning and infer)': equations or ['No explicit rgroup equation that written as standalone text'],
              'bboxes': out, 'corefs': links,
              'postprocess': {'version': '1.2', 'source_sha256': cat['source_sha256'], 'plan_sha256': digest(plan),
                              'status': 'needs_review' if reviews or plan['structure_warnings'] or ambiguous else 'label_plan_applied',
                              'program_fixes': program_fixes,
                              'graph2smiles_required': True, 'graph_validated': False,
                              'decisions': list(decisions.values()), 'structure_warnings': plan['structure_warnings'],
                              'provenance': provenance, 'audit': audit}}
    # Invariants independent of any chemical oracle.
    expected_count = sum(b.get('category') == '[Mol]' for b in edited) + sum(len(x[2]) - 1 for x in groups.values())
    require(sum(b.get('category') == '[Mol]' for b in out) == expected_count, 'Internal expansion count error')
    for row in provenance:
        candidate, original = out[row['output_index']], edited[row['source_bbox_index']]
        if candidate.get('category') == '[Mol]':
            require(len(candidate['symbols']) == len(original['symbols']), 'Internal atom count mutation')
            require(candidate.get('bbox') == original.get('bbox') and candidate.get('smiles') == original.get('smiles'), 'Internal immutable field mutation')
    return result


def process_lenient(data, plan, max_drops=60):
    """process() that survives bad entries: an entry that fails validation is dropped (audited), a
    decision that disagrees with the surviving groups/definitions is replaced, and molecules left
    without a decision get keep_generic. Everything the model got right is still applied. Raises
    PlanError only when the plan cannot be made valid this way (or the schema itself is broken)."""
    plan = copy.deepcopy(plan)
    dropped, added = [], []
    for _ in range(max_drops):
        try:
            result = process(data, plan)
            break
        except PlanError as exc:
            msg = str(exc)
            if exc.entry is not None:
                section, index = exc.entry
                gone = plan[section].pop(index)
                dropped.append({'section': section, 'entry': gone, 'reason': msg})
                continue
            if msg.startswith('Missing decisions'):
                group_ids = {g['molecule_id'] for g in plan['groups']}
                have = {d['molecule_id'] for d in plan['decisions']}
                for mid in re.findall(r"mol_\d+", msg):
                    if mid not in have:
                        d = {'molecule_id': mid, 'action': 'expand' if mid in group_ids else 'keep_generic', 'reason': 'program: decision added after an invalid entry was dropped'}
                        plan['decisions'].append(d)
                        added.append(d)
                continue
            if msg.startswith('Expansion group lacks decision'):
                have = {d['molecule_id'] for d in plan['decisions']}
                for g in plan['groups']:
                    if g['molecule_id'] not in have:
                        d = {'molecule_id': g['molecule_id'], 'action': 'expand', 'reason': 'program: decision added for a surviving group'}
                        plan['decisions'].append(d)
                        added.append(d)
                continue
            raise
    else:
        raise PlanError(f'plan still invalid after dropping {len(dropped)} entries')
    post = result['postprocess']
    for d in dropped:
        post['audit'].append({'operation': 'dropped_entry', **d})
    for d in added:
        post['audit'].append({'operation': 'added_decision', **d})
    post['dropped_entries'] = dropped
    post['added_decisions'] = added
    if dropped or added:
        post['status'] = post['status'] + '_lenient'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tool_json', type=pathlib.Path)
    parser.add_argument('plan_json', type=pathlib.Path)
    parser.add_argument('output_json', type=pathlib.Path)
    args = parser.parse_args()
    require(not args.output_json.exists(), 'Refusing to overwrite output')
    result = process(json.loads(args.tool_json.read_text(encoding='utf-8')), json.loads(args.plan_json.read_text(encoding='utf-8')))
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

import copy
import traceback
import numpy as np
import multiprocessing
import itertools
import re

import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import rdFMCS, rdDepictor


rdkit.RDLogger.DisableLog('rdApp.*')

from SmilesPE.pretokenizer import atomwise_tokenizer

from .constants import RGROUP_SYMBOLS, ABBREVIATIONS, VALENCES, FORMULA_REGEX
import re
import json
import urllib.parse
from typing import List, Optional, Dict, Tuple
import requests
import os
from openai import AzureOpenAI

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("Please set API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")

# ========== External service base URLs ==========
OPSIN_BASE   = "https://opsin.ch.cam.ac.uk/opsin/"
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CIR_BASE     = "https://cactus.nci.nih.gov/chemical/structure"

# ========== Shorthand lexicon ==========
# Only does a small "shorthand -> parsable English name" mapping; does not enumerate SMILES
GROUP_LEXICON: Dict[str, str] = {
    # halo / alkyl
    "F":"fluoro","Cl":"chloro","Br":"bromo","I":"iodo",
    "Me":"methyl","Et":"ethyl","nPr":"propyl","iPr":"propan-2-yl",
    "nBu":"butyl","sBu":"butan-2-yl","iBu":"2-methylpropyl","tBu":"tert-butyl",
    # alkoxy / ester / protecting groups
    "OMe":"methoxy","MeO":"methoxy","OEt":"ethoxy","OiPr":"propan-2-yloxy",
    "OtBu":"tert-butoxy","OBn":"benzyloxy","OPh":"phenoxy","OCF3":"trifluoromethoxy",
    "OAc":"acetoxy","OPiv":"pivaloyloxy",
    "OTs":"4-tolylsulfonyloxy","OMs":"methanesulfonyloxy","OTf":"trifluoromethanesulfonyloxy",
    # nitrogen-containing / carbonyl / strong electron-withdrawing
    "NO2":"nitro","NH2":"amino","NMe2":"dimethylamino","NEt2":"diethylamino",
    "CHO":"formyl","Ac":"acetyl","COCH3":"acetyl","Bz":"benzoyl",
    "CF3":"trifluoromethyl","CN":"cyano",
    "CONH2":"carbamoyl","CONMe2":"dimethylcarbamoyl",
    "CO2Me":"methoxycarbonyl","CO2Et":"ethoxycarbonyl",
    "CO2iPr":"propan-2-yloxycarbonyl","CO2tBu":"tert-butoxycarbonyl",
    "Boc":"tert-butoxycarbonyl","Cbz":"benzyloxycarbonyl",
    # thio / sulfonyl
    "SMe":"methylsulfanyl","SEt":"ethylsulfanyl","SPh":"phenylsulfanyl",
    "SO2Me":"methanesulfonyl","SO2Ph":"phenylsulfonyl","SO2CF3":"trifluoromethanesulfonyl",
    # others
    "CF3O":"trifluoromethoxy","N3":"azido",
}

# "Main functional group suffix" type (needs to be attached to a parent)
SUFFIX_GROUPS: Dict[str, str] = {
    "B(OH)2": "boronic acid",  # boronic acid
}

# Family recognition (ring code -> family name & typically allowed positions, for reference only, not strictly enforced)
_RING_FAMILIES: List[Tuple[re.Pattern, str, set]] = [
    (re.compile(r"^C6H\d+$"),  "phenyl",   {2,3,4,5,6}),         # phenyl
    (re.compile(r"^C10H\d+$"), "naphthyl", set(range(1,9))),     # naphthyl
    (re.compile(r"^C5H\d+N$"), "pyridyl",  {2,3,4}),             # pyridyl
    (re.compile(r"^C4H\d+S$"), "thienyl",  {2,3}),               # thienyl
    (re.compile(r"^C4H\d+O$"), "furyl",    {2,3}),               # furyl
    (re.compile(r"^C8H\d+N$"), "indolyl",  {1,2,3,4,5,6,7}),     # indolyl
]

# Family -> parent name (used for suffix functional groups)
FAMILY_TO_PARENT: Dict[str, str] = {
    "phenyl":"benzene","naphthyl":"naphthalene","pyridyl":"pyridine",
    "thienyl":"thiophene","furyl":"furan","indolyl":"indole",
}

# Alias rings (e.g. Ph / Py)
RING_ALIASES = {"Ph":"phenyl","Py":"pyridyl","Th":"thienyl","Fur":"furyl","Np":"naphthyl","Ind":"indolyl"}

# ========== Basic utilities ==========
def _norm(s: str) -> str:
    return s.replace(" ", "").replace("−","-").replace("–","-").replace("—","-")

def _infer_ring_family(ring_code: str) -> Optional[Tuple[str,set]]:
    if ring_code in RING_ALIASES:
        fam = RING_ALIASES[ring_code]
        allowed = next((al for pat,f,al in _RING_FAMILIES if f==fam), set())
        return fam, allowed
    for pat,fam,allowed in _RING_FAMILIES:
        if pat.match(ring_code):
            return fam, allowed
    return None

def _positions_to_prefix(positions: List[int], group_name: str) -> str:
    if len(positions)==1:
        return f"{positions[0]}-{group_name}"
    pos_str = ",".join(str(p) for p in positions)
    mult = {2:"bis",3:"tris",4:"tetrakis"}.get(len(positions),f"{len(positions)}-kis")
    return f"{pos_str}-{mult}({group_name})"

def _strip_group_multiplier(grp_raw: str) -> Tuple[str, Optional[int]]:
    m = re.fullmatch(r"\(([^()]+)\)(\d+)", grp_raw)
    if not m: return grp_raw, None
    return m.group(1), int(m.group(2))

# ========== External parsers ==========
def _opsin_name_to_smiles(name: str) -> Optional[str]:
    url = f"{OPSIN_BASE}{urllib.parse.quote(name)}.json"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code!=200: return None
        js = r.json()
        if js.get("status")!="SUCCESS": return None
        return js.get("smiles")
    except Exception:
        return None

def _pubchem_name_to_smiles(name: str) -> Optional[str]:
    # name -> CID
    url = f"{PUBCHEM_BASE}/compound/name/{urllib.parse.quote(name)}/cids/JSON"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code!=200: return None
        cids = r.json().get("IdentifierList",{}).get("CID",[])
        if not cids: return None
        cid_str = ",".join(map(str, cids[:5]))  # take the first few candidates
        url2 = f"{PUBCHEM_BASE}/compound/cid/{cid_str}/property/CanonicalSMILES/JSON"
        r2 = requests.get(url2, timeout=20)
        if r2.status_code!=200: return None
        props = r2.json().get("PropertyTable",{}).get("Properties",[])
        return props[0].get("CanonicalSMILES") if props else None
    except Exception:
        return None

def _cir_name_to_smiles(name: str) -> Optional[str]:
    url = f"{CIR_BASE}/{urllib.parse.quote(name)}/smiles"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code==200 and r.text.strip():
            return r.text.strip()
    except Exception:
        pass
    return None

# ========== Shorthand -> candidate canonical names (key part) ==========
def _shorthand_candidates(token: str) -> List[str]:
    """
    Input examples:
      4-NO2C6H4, 3,5-CF3C6H3, 2-OMeC6H4, 4-BrC6H4,
      1-NO2C10H7, 2-OMeC5H4N, 2-BrC4H3S, 2-OMeC4H3O,
      4-B(OH)2C6H4, 3,5-B(OH)2C6H3
    Returns a set of "parsable" name candidates, ordered from most to least likely to succeed.
    """
    t = _norm(token)
    pat = (
        r"(?P<pos>\d(?:,\d)*)-"                       # positions  e.g. 3,5
        r"(?P<grp>\([A-Za-z0-9]+\)\d+|[A-Za-z0-9()]+)"# group  e.g. CF3 / (MeO)3 / B(OH)2
        r"(?P<ring>(?:C[0-9]+H[0-9]+[NOS]?)|(?:Ph|Py|Th|Fur|Np|Ind))"  # ring formula/alias
    )
    m = re.fullmatch(pat, t)
    if not m:
        return []

    positions = list(map(int, m.group("pos").split(",")))
    grp_raw   = m.group("grp")
    ring_code = m.group("ring")

    fam = _infer_ring_family(ring_code)
    if not fam:
        return []
    ring_family, _ = fam

    # handle (X)3 multiplier
    grp_core, _mult = _strip_group_multiplier(grp_raw)

    # suffix type (e.g. B(OH)2)
    if grp_core in SUFFIX_GROUPS:
        suffix = SUFFIX_GROUPS[grp_core]
        parent = FAMILY_TO_PARENT.get(ring_family, ring_family)

        if len(positions) <= 1:
            # single position: parent-<locant>-boronic acid; default to 1 if no position given
            loc = positions[0] if positions else 1
            return [f"{parent}-{loc}-{suffix}"]  # e.g. benzene-4-boronic acid
        else:
            # multiple positions: parent-1,3-di/tri/tetra boronic acid
            pos_str = ",".join(str(p) for p in positions)
            mult = {2:"di",3:"tri",4:"tetra"}.get(len(positions), f"{len(positions)}")
            return [f"{parent}-{pos_str}-{mult}{suffix}"]  # e.g. benzene-1,3-diboronic acid

    # prefix type (e.g. CF3/NO2/OMe...)
    group_name = GROUP_LEXICON.get(grp_core, grp_core)  # pass unknown words directly to the parser to try
    prefix = _positions_to_prefix(positions, group_name)

    # Route A: treat it as "substituent + phenyl/naphthyl ..." (e.g. 3,5-bis(trifluoromethyl)phenyl)
    candA = f"{prefix}{ring_family}"

    # Route B: treat it as "parent molecule + multi-substitution" (e.g. 1,3-bis(trifluoromethyl)benzene)
    # Heuristic: if family is phenyl -> parent is benzene; naphthyl -> naphthalene; etc.
    parent = FAMILY_TO_PARENT.get(ring_family, None)
    cands = [candA]
    if parent:
        # Choose the parent numbering with "canonical start 1": commonly fix the first position to 1,
        # and normalize the rest by nearest ortho/meta/para (simple heuristic: contains 2->1,2; contains 4->1,4; else 1,3)
        if len(positions) == 1:
            posB = f"{positions[0]}"
            candB = f"{parent}-{posB}-{group_name}"
        else:
            # rough normalization: o(contains 2 or 6)->1,2; p(contains 4)->1,4; else m->1,3,... (for multi-substitution, commonly 1,3,5 as in examples)
            if any(p in (2,6) for p in positions): canon = [1,2] + ([4] if len(positions)>=3 else [])
            elif 4 in positions:                   canon = [1,4] + ([2] if len(positions)>=3 else [])
            else:                                   canon = [1,3] + ([5] if len(positions)>=3 else [])
            posB = ",".join(map(str, canon[:len(positions)]))
            mult = {2:"bis",3:"tris",4:"tetrakis"}.get(len(positions), f"{len(positions)}-kis")
            candB = f"{posB}-{mult}({group_name}){parent}"
        cands.append(candB)

    # Return order: try "parent molecule" first, then "substituent"
    return cands

# ========== Main interface ==========
def name2smiles(name: str, allow_shorthand: bool = True) -> Optional[str]:
    """
    Input any name:
      - direct IUPAC or common English name: OPSIN first -> then PubChem -> then CIR
      - shorthand (e.g. 3,5-CF3C6H3 / 4-B(OH)2C6H4): first generate several "parsable candidate names", try each in turn
    Returns the first successfully parsed SMILES; otherwise None
    """
    s = name.strip()
    # 1) Direct parsing first (some shorthands may also be recognized by the databases)
    for fn in (_opsin_name_to_smiles, _pubchem_name_to_smiles, _cir_name_to_smiles):
        smi = fn(s)
        if smi: return smi

    # 2) Shorthand path
    if allow_shorthand:
        cands = _shorthand_candidates(s)
        print(f"cands: {cands}")
        for cand in cands:
            for fn in (_opsin_name_to_smiles, _pubchem_name_to_smiles, _cir_name_to_smiles):
                smi = fn(cand)
                if smi: return smi
    return None

# ========== LLM helper functions ==========
def _load_prompt_template(prompt_file: str = "prompt/prompt_symbol_to_smiles.txt") -> str:
    """
    Load the prompt template file.
    
    Args:
        prompt_file: prompt file path (relative to the project root directory)
    
    Returns:
        prompt template string; returns the default template if the file does not exist
    """
    import os
    import pathlib
    
    # Try multiple possible paths
    possible_paths = [
        prompt_file,  # absolute path or current working directory
        os.path.join(os.path.dirname(os.path.dirname(__file__)), prompt_file),  # up from the molnextr directory
        os.path.join(os.getcwd(), prompt_file),  # current working directory
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception:
                continue
    
    # If the file does not exist, return the default template
    return """You are a cheminformatics expert. Please convert the following chemical symbol or name to SMILES format.

Input symbol: {symbol}

Requirements:
1. The output must be a valid SMILES string
2. **The connection point atom must be enclosed in square brackets []**, for example:
   - [C] represents a connection point carbon atom
   - [Si] represents a connection point silicon atom
   - [c] represents a connection point aromatic carbon atom
   - [O] represents a connection point oxygen atom
   - [N] represents a connection point nitrogen atom
   - [S] represents a connection point sulfur atom
3. Output must be in JSON format: {{"smiles": "your_smiles_string"}}
4. Only output the JSON object, do not add any explanations or additional text

Examples:
- Input: "CO2CH2Bn" or "CO2CH2Ph" → Output: {{"smiles": "[C](=O)O[CH2]c1ccccc1"}}
- Input: "TIPS" → Output: {{"smiles": "[Si](C(C)C)(C(C)C)C(C)C"}}
- Input: "C6F5" → Output: {{"smiles": "[c]1c(F)c(F)c(F)c(F)c1(F)"}}
- Input: "4-BrC6H4" → Output: {{"smiles": "[c]1ccc(Br)cc1"}}
- Input: "OMe" → Output: {{"smiles": "[O]C"}}
- Input: "NO2" → Output: {{"smiles": "[N+](=O)[O-]"}}

Please output the JSON object:"""


def _llm_symbol_to_smiles(symbol: str, 
                          api_key: Optional[str] = None,
                          api_endpoint: Optional[str] = None,
                          api_version: Optional[str] = None,
                          model: str = "gpt-5-mini",
                          prompt_file: Optional[str] = None) -> Optional[str]:
    """
    Convert chemical symbol to SMILES string using large language model.
    
    Requirements:
    - Output must be valid SMILES format
    - Connection point atoms must be enclosed in square brackets [] (e.g., [C], [Si], [c])
    - Output must be in JSON format: {"smiles": "xxx"}
    
    Args:
        symbol: Chemical symbol to convert
        api_key: Azure OpenAI API key (if None, uses API_KEY from module or environment variable)
        api_endpoint: Azure OpenAI endpoint (if None, uses AZURE_ENDPOINT from module or environment variable)
        api_version: API version (if None, uses API_VERSION from module)
        model: Model name to use
        prompt_file: Path to prompt file (optional, defaults to prompt/prompt_symbol_to_smiles.txt)
    
    Returns:
        SMILES string on success, None on failure
    """
    # Use module-level config as defaults, then parameters, then environment variables
    # Priority: function parameters > module variables > environment variables
    # Within the same module, module-level variables can be accessed directly
    api_key = api_key or globals().get("API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    api_endpoint = api_endpoint or globals().get("AZURE_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = api_version or globals().get("API_VERSION", "2024-10-21")
    
    # If there is no API_KEY or endpoint, silently return None (LLM not enabled)
    if not api_key or not api_endpoint:
        return None
    
    try:
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=api_endpoint
        )
        
        # Load the prompt template
        prompt_template = _load_prompt_template(
            prompt_file or "prompt/prompt_symbol_to_smiles.txt"
        )
        prompt = prompt_template.format(symbol=symbol)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional cheminformatics assistant specialized in converting chemical symbols to SMILES format."},
                {"role": "user", "content": prompt}
            ],
            #temperature=0,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content.strip()
        #print(f"content: {content}")
        
        # Parse the JSON output
        try:
            result = json.loads(content)
            smiles = result.get("smiles", "").strip()
            
            if not smiles:
                return None
            
            # Validate that the output is a valid SMILES (using RDKit)
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    return smiles
            except Exception:
                pass
        except json.JSONDecodeError:
            # If JSON parsing fails, try direct extraction (fault tolerance)
            # Clean up possible markdown code block markers
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
                content = content.strip()
            
            # Try to extract JSON (may be embedded in text)
            json_match = re.search(r'\{[^}]*"smiles"[^}]*\}', content)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    smiles = result.get("smiles", "").strip()
                    if smiles:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None:
                            return smiles
                except Exception:
                    pass
        
        return None
        
    except Exception as e:
        # Silent failure, return None
        return None




def get_smiles_stereo_list(smiles):
    pat = re.compile(r'\[C@[\w\d]*\]|\[C@@[\w\d]*\]')
    lst = []
    for m in pat.finditer(smiles):
        if '@@' in m.group():
            lst.append('@@')
        else:
            lst.append('@')
    return lst

def flip_stereo_in_smiles(smiles, flip_indices):
    pat = re.compile(r'\[C@[\w\d]*\]|\[C@@[\w\d]*\]')
    matches = list(pat.finditer(smiles))
    assert len(matches) >= max(flip_indices, default=-1) + 1, "index out of range"
    smiles_new = smiles
    offset = 0
    for idx in flip_indices:
        m = matches[idx]
        start, end = m.start() + offset, m.end() + offset
        orig = smiles_new[start:end]
        if '@@' in orig:
            flipped = orig.replace('@@', '@')
        else:
            flipped = orig.replace('@', '@@')
        smiles_new = smiles_new[:start] + flipped + smiles_new[end:]
        offset += len(flipped) - len(orig)
    return smiles_new

def chirality_sign(coords):
    p0, p1, p2 = coords[:3]
    v1 = np.array([p1.x - p0.x, p1.y - p0.y])
    v2 = np.array([p2.x - p0.x, p2.y - p0.y])
    cross = v1[0]*v2[1] - v1[1]*v2[0]
    return np.sign(cross)

def align_chirality(smiles1, smiles2):
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        # 1. Find MCS
        res = rdFMCS.FindMCS(
            [mol1, mol2],
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder
        )
        mcs_mol = Chem.MolFromSmarts(res.smartsString)
        match1 = mol1.GetSubstructMatch(mcs_mol)
        match2 = mol2.GetSubstructMatch(mcs_mol)
        if not match1 or not match2:
            return smiles2

        # 2. 2D coords for spatial arrangement comparison
        rdDepictor.Compute2DCoords(mol1)
        rdDepictor.Compute2DCoords(mol2)
        coords1 = [mol1.GetConformer().GetAtomPosition(i) for i in match1]
        coords2 = [mol2.GetConformer().GetAtomPosition(i) for i in match2]
        sign1 = chirality_sign(coords1)
        sign2 = chirality_sign(coords2)
        is_mirror = (sign1 != sign2)

        # 3. Find chiral centers
        chiral1 = list(Chem.FindMolChiralCenters(
            mol1, includeUnassigned=False, includeCIP=True
        ))
        chiral2 = list(Chem.FindMolChiralCenters(
            mol2, includeUnassigned=False, includeCIP=True
        ))

        # Case: no defined centers, use SMILES stereo flags
        if len(chiral1) == 0:
            stereo1 = get_smiles_stereo_list(smiles1)
            stereo2 = get_smiles_stereo_list(smiles2)
            flip_indices = []
            for i, (s1, s2) in enumerate(zip(stereo1, stereo2)):
                target = s1 if not is_mirror else ('@@' if s1 == '@' else '@')
                if s2 != target:
                    flip_indices.append(i)
            return flip_stereo_in_smiles(smiles2, flip_indices)

        # Mismatch in number of centers
        if len(chiral2) != len(chiral1):
            return smiles2

        # 4. Align absolute R/S on matched centers
        mol2_edit = Chem.RWMol(mol2)
        chiral1_dict = dict(chiral1)
        chiral2_dict = dict(chiral2)
        for i, idx1 in enumerate(match1):
            idx2 = match2[i]
            if idx1 in chiral1_dict:
                ref_chirality = chiral1_dict[idx1]
                if is_mirror:
                    ref_chirality = {'R':'S','S':'R'}.get(ref_chirality, ref_chirality)
                rs2 = chiral2_dict.get(idx2)
                if rs2 is not None and ref_chirality != rs2:
                    atom = mol2_edit.GetAtomWithIdx(idx2)
                    tag = atom.GetChiralTag()
                    if tag == Chem.CHI_TETRAHEDRAL_CW:
                        atom.SetChiralTag(Chem.CHI_TETRAHEDRAL_CCW)
                    elif tag == Chem.CHI_TETRAHEDRAL_CCW:
                        atom.SetChiralTag(Chem.CHI_TETRAHEDRAL_CW)

        # 5. Return updated SMILES
        mol2_new = mol2_edit.GetMol()
        Chem.SanitizeMol(mol2_new)
        return Chem.MolToSmiles(mol2_new, isomericSmiles=True)

    except Exception:
        return smiles2

    







def is_valid_mol(s, format_='atomtok'):
    if format_ == 'atomtok':
        mol = Chem.MolFromSmiles(s)
    elif format_ == 'inchi':
        if not s.startswith('InChI=1S'):
            s = f"InChI=1S/{s}"
        mol = Chem.MolFromInchi(s)
    else:
        raise NotImplemented
    return mol is not None


def _convert_smiles_to_inchi(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        inchi = Chem.MolToInchi(mol)
    except:
        inchi = None
    return inchi


def convert_smiles_to_inchi(smiles_list, num_workers=16):
    with multiprocessing.Pool(num_workers) as p:
        inchi_list = p.map(_convert_smiles_to_inchi, smiles_list, chunksize=128)
    n_success = sum([x is not None for x in inchi_list])
    r_success = n_success / len(inchi_list)
    inchi_list = [x if x else 'InChI=1S/H2O/h1H2' for x in inchi_list]
    return inchi_list, r_success


def merge_inchi(inchi1, inchi2):
    replaced = 0
    inchi1 = copy.deepcopy(inchi1)
    for i in range(len(inchi1)):
        if inchi1[i] == 'InChI=1S/H2O/h1H2':
            inchi1[i] = inchi2[i]
            replaced += 1
    return inchi1, replaced


def _get_num_atoms(smiles):
    try:
        return Chem.MolFromSmiles(smiles).GetNumAtoms()
    except:
        return 0


def get_num_atoms(smiles, num_workers=16):
    if type(smiles) is str:
        return _get_num_atoms(smiles)
    with multiprocessing.Pool(num_workers) as p:
        num_atoms = p.map(_get_num_atoms, smiles)
    return num_atoms


def normalize_nodes(nodes, flip_y=True):
    x, y = nodes[:, 0], nodes[:, 1]
    minx, maxx = min(x), max(x)
    miny, maxy = min(y), max(y)
    x = (x - minx) / max(maxx - minx, 1e-6)
    if flip_y:
        y = (maxy - y) / max(maxy - miny, 1e-6)
    else:
        y = (y - miny) / max(maxy - miny, 1e-6)
    return np.stack([x, y], axis=1)


def _verify_chirality(mol, coords, symbols, edges, debug=False):
    try:
        n = mol.GetNumAtoms()
        
        # Make a temp mol to find chiral centers
        mol_tmp = mol.GetMol()
        
        Chem.SanitizeMol(mol_tmp)
        #print(f"n: {n}", f"mol_tmp: {mol_tmp}")

        # chiral_centers = Chem.FindMolChiralCenters(
        #     mol_tmp, includeUnassigned=True, includeCIP=False, useLegacyImplementation=False)
        # chiral_center_ids = [idx for idx, _ in chiral_centers]
        # print(f"chiral_center_ids: {chiral_center_ids}")  # List[Tuple[int, any]] -> List[int]

        # if not chiral_center_ids:
        # # symbols is a list of atom symbols (e.g. ['[F3C]', '[C@]', ...])
        chiral_tags = ['[C@]', '[C@@]', '[C@H]', '[C@@H]']
        chiral_center_ids = [i for i, sym in enumerate(symbols) if any(tag in sym for tag in chiral_tags)]
        #print(f"chiral_center_ids (from symbols): {chiral_center_ids}")
        
        # correction to clear pre-condition violation (for some corner cases)
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.SINGLE:
                bond.SetBondDir(Chem.BondDir.NONE)

        # Create conformer from 2D coordinate
        conf = Chem.Conformer(n)
        conf.Set3D(True)
        for i, (x, y) in enumerate(coords):
            conf.SetAtomPosition(i, (x, 1 - y, 0))
        mol.AddConformer(conf)
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistryFrom3D(mol)
        # NOTE: seems that only AssignStereochemistryFrom3D can handle double bond E/Z
        # So we do this first, remove the conformer and add back the 2D conformer for chiral correction

        mol.RemoveAllConformers()
        conf = Chem.Conformer(n)
        conf.Set3D(False)
        for i, (x, y) in enumerate(coords):
            conf.SetAtomPosition(i, (x, 1 - y, 0))
        mol.AddConformer(conf)

        # Magic, inferring chirality from coordinates and BondDir. DO NOT CHANGE.
        Chem.SanitizeMol(mol)
        Chem.AssignChiralTypesFromBondDirs(mol)
        Chem.AssignStereochemistry(mol, force=True)

        # Second loop to reset any wedge/dash bond to be starting from the chiral center)
        for i in chiral_center_ids:
            for j in range(n):
                if edges[i][j] == 5:
                    # assert edges[j][i] == 6
                    mol.RemoveBond(i, j)
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                    mol.GetBondBetweenAtoms(i, j).SetBondDir(Chem.BondDir.BEGINWEDGE)
                elif edges[i][j] == 6:
                    # assert edges[j][i] == 5
                    mol.RemoveBond(i, j)
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                    mol.GetBondBetweenAtoms(i, j).SetBondDir(Chem.BondDir.BEGINDASH)
            Chem.AssignChiralTypesFromBondDirs(mol)
            Chem.AssignStereochemistry(mol, force=True)

        # reset chiral tags for non-carbon atom
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "C":
                atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
        mol = mol.GetMol()

    except Exception as e:
        if debug:
            raise e
        pass
    return mol


def _parse_tokens(tokens: list):
    """
    Parse tokens of condensed formula into list of pairs `(elt, num)`
    where `num` is the multiplicity of the atom (or nested condensed formula) `elt`
    Used by `_parse_formula`, which does the same thing but takes a formula in string form as input
    """
    elements = []
    i = 0
    j = 0
    while i < len(tokens):
        if tokens[i] == '(':
            while j < len(tokens) and tokens[j] != ')':
                j += 1
            elt = _parse_tokens(tokens[i + 1:j])
        else:
            elt = tokens[i]
        j += 1
        if j < len(tokens) and tokens[j].isnumeric():
            num = int(tokens[j])
            j += 1
        else:
            num = 1
        elements.append((elt, num))
        i = j
    return elements


def _parse_formula(formula: str):
    """
    Parse condensed formula into list of pairs `(elt, num)`
    where `num` is the subscript to the atom (or nested condensed formula) `elt`
    Example: "C2H4O" -> [('C', 2), ('H', 4), ('O', 1)]
    """
    tokens = FORMULA_REGEX.findall(formula)
    # if ''.join(tokens) != formula:
    #     tokens = FORMULA_REGEX_BACKUP.findall(formula)
    return _parse_tokens(tokens)


def _expand_carbon(elements: list):
    """
    Given list of pairs `(elt, num)`, output single list of all atoms in order,
    expanding carbon sequences (CaXb where a > 1 and X is halogen) if necessary
    Example: [('C', 2), ('H', 4), ('O', 1)] -> ['C', 'H', 'H', 'C', 'H', 'H', 'O'])
    """
    expanded = []
    i = 0
    while i < len(elements):
        elt, num = elements[i]
        # expand carbon sequence
        if elt == 'C' and num > 1 and i + 1 < len(elements):
            next_elt, next_num = elements[i + 1]
            quotient, remainder = next_num // num, next_num % num
            for _ in range(num):
                expanded.append('C')
                for _ in range(quotient):
                    expanded.append(next_elt)
            for _ in range(remainder):
                expanded.append(next_elt)
            i += 2
        # recurse if `elt` itself is a list (nested formula)
        elif isinstance(elt, list):
            new_elt = _expand_carbon(elt)
            for _ in range(num):
                expanded.append(new_elt)
            i += 1
        # simplest case: simply append `elt` `num` times
        else:
            for _ in range(num):
                expanded.append(elt)
            i += 1
    return expanded


def _expand_abbreviation(abbrev):
    """
    Expand abbreviation into its SMILES; also converts [Rn] to [n*]
    Used in `_condensed_formula_list_to_smiles` when encountering abbrev. in condensed formula
    """
    if abbrev in ABBREVIATIONS:
        return ABBREVIATIONS[abbrev].smiles
    if abbrev in RGROUP_SYMBOLS or (abbrev[0] == 'R' and abbrev[1:].isdigit()):
        if abbrev[1:].isdigit():
            return f'[{abbrev[1:]}*]'
        return '*'
    return f'[{abbrev}]'


def _get_bond_symb(bond_num):
    """
    Get SMILES symbol for a bond given bond order
    Used in `_condensed_formula_list_to_smiles` while writing the SMILES string
    """
    if bond_num == 0:
        return '.'
    if bond_num == 1:
        return ''
    if bond_num == 2:
        return '='
    if bond_num == 3:
        return '#'
    return ''


def _condensed_formula_list_to_smiles(formula_list, start_bond, end_bond=None, direction=None):
    """
    Converts condensed formula (in the form of a list of symbols) to smiles
    Input:
    `formula_list`: e.g. ['C', 'H', 'H', 'N', ['C', 'H', 'H', 'H'], ['C', 'H', 'H', 'H']] for CH2N(CH3)2
    `start_bond`: # bonds attached to beginning of formula
    `end_bond`: # bonds attached to end of formula (deduce automatically if None)
    `direction` (1, -1, or None): direction in which to process the list (1: left to right; -1: right to left; None: deduce automatically)
    Returns:
    `smiles`: smiles corresponding to input condensed formula
    `bonds_left`: bonds remaining at the end of the formula (for connecting back to main molecule); should equal `end_bond` if specified
    `num_trials`: number of trials
    `success` (bool): whether conversion was successful
    """
    # `direction` not specified: try left to right; if fails, try right to left
    if direction is None:
        num_trials = 1
        for dir_choice in [1, -1]:
            smiles, bonds_left, trials, success = _condensed_formula_list_to_smiles(formula_list, start_bond, end_bond, dir_choice)
            num_trials += trials
            if success:
                return smiles, bonds_left, num_trials, success
        return None, None, num_trials, False
    assert direction == 1 or direction == -1

    def dfs(smiles, bonds_left, cur_idx, add_idx):
        """
        `smiles`: SMILES string so far
        `cur_idx`: index (in list `formula`) of current atom (i.e. atom to which subsequent atoms are being attached)
        `cur_flat_idx`: index of current atom in list of atom tokens of SMILES so far
        `bonds_left`: bonds remaining on current atom for subsequent atoms to be attached to
        `add_idx`: index (in list `formula`) of atom to be attached to current atom
        `add_flat_idx`: index of atom to be added in list of atom tokens of SMILES so far
        Note: "atom" could refer to nested condensed formula (e.g. CH3 in CH2N(CH3)2)
        """
        num_trials = 1
        # end of formula: return result
        if (direction == 1 and add_idx == len(formula_list)) or (direction == -1 and add_idx == -1):
            if end_bond is not None and end_bond != bonds_left:
                return smiles, bonds_left, num_trials, False
            return smiles, bonds_left, num_trials, True

        # no more bonds but there are atoms remaining: conversion failed
        if bonds_left <= 0:
            return smiles, bonds_left, num_trials, False
        to_add = formula_list[add_idx]  # atom to be added to current atom

        if isinstance(to_add, list):  # "atom" added is a list (i.e. nested condensed formula): assume valence of 1
            if bonds_left > 1:
                # "atom" added does not use up remaining bonds of current atom
                # get smiles of "atom" (which is itself a condensed formula)
                add_str, val, trials, success = _condensed_formula_list_to_smiles(to_add, 1, None, direction)
                if val > 0:
                    add_str = _get_bond_symb(val + 1) + add_str
                num_trials += trials
                if not success:
                    return smiles, bonds_left, num_trials, False
                # put smiles of "atom" in parentheses and append to smiles; go to next atom to add to current atom
                result = dfs(smiles + f'({add_str})', bonds_left - 1, cur_idx, add_idx + direction)
            else:
                # "atom" added uses up remaining bonds of current atom
                # get smiles of "atom" and bonds left on it
                add_str, bonds_left, trials, success = _condensed_formula_list_to_smiles(to_add, 1, None, direction)
                num_trials += trials
                if not success:
                    return smiles, bonds_left, num_trials, False
                # append smiles of "atom" (without parentheses) to smiles; it becomes new current atom
                result = dfs(smiles + add_str, bonds_left, add_idx, add_idx + direction)
            smiles, bonds_left, trials, success = result
            num_trials += trials
            return smiles, bonds_left, num_trials, success

        # atom added is a single symbol (as opposed to nested condensed formula)
        for val in VALENCES.get(to_add, [1]):  # try all possible valences of atom added
            add_str = _expand_abbreviation(to_add)  # expand to smiles if symbol is abbreviation
            if bonds_left > val:  # atom added does not use up remaining bonds of current atom; go to next atom to add to current atom
                if cur_idx >= 0:
                    add_str = _get_bond_symb(val) + add_str
                result = dfs(smiles + f'({add_str})', bonds_left - val, cur_idx, add_idx + direction)
            else:  # atom added uses up remaining bonds of current atom; it becomes new current atom
                if cur_idx >= 0:
                    add_str = _get_bond_symb(bonds_left) + add_str
                result = dfs(smiles + add_str, val - bonds_left, add_idx, add_idx + direction)
            trials, success = result[2:]
            num_trials += trials
            if success:
                return result[0], result[1], num_trials, success
            if num_trials > 10000:
                break
        return smiles, bonds_left, num_trials, False

    cur_idx = -1 if direction == 1 else len(formula_list)
    add_idx = 0 if direction == 1 else len(formula_list) - 1
    return dfs('', start_bond, cur_idx, add_idx)


def get_smiles_from_symbol(symbol, mol,atom, bonds, 
                           use_llm: bool = True,
                           llm_api_key: Optional[str] = None,
                           llm_api_endpoint: Optional[str] = None,
                           llm_model: str = "gpt-5-mini"):

    if symbol in ABBREVIATIONS:
        return ABBREVIATIONS[symbol].smiles

    try_mol = Chem.MolFromSmiles(symbol)
    if try_mol is not None:    
        return symbol
        
    opsin_smiles = name2smiles(symbol)
    if opsin_smiles:
        return opsin_smiles 

    total_bonds = int(sum([bond.GetBondTypeAsDouble() for bond in bonds]))
    formula_list = _expand_carbon(_parse_formula(symbol))
    smiles, bonds_left, num_trails, success = _condensed_formula_list_to_smiles(formula_list, total_bonds, None)
    if success:
        # Validate whether the SMILES is a viable molecule
        test_mol = Chem.MolFromSmiles(smiles)
        if test_mol is not None:
            return smiles
    
    # Final step: use a large language model (only enabled when API_KEY is configured)
    if use_llm:
        # Check whether there is usable API configuration (parameters, module variables, or environment variables)
        has_api_config = (
            llm_api_key or 
            llm_api_endpoint or 
            globals().get("API_KEY") or 
            globals().get("AZURE_ENDPOINT") or 
            os.getenv("AZURE_OPENAI_API_KEY") or 
            os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        if has_api_config:
            llm_smiles = _llm_symbol_to_smiles(
                symbol, 
                api_key=llm_api_key,
                api_endpoint=llm_api_endpoint,
                model=llm_model
            )
            if llm_smiles:
                return llm_smiles

    return None


def _replace_functional_group(smiles):
    smiles = smiles.replace('<unk>', 'C')
    for i, r in enumerate(RGROUP_SYMBOLS):
        symbol = f'[{r}]'
        if symbol in smiles:
            if r[0] == 'R' and r[1:].isdigit():
                smiles = smiles.replace(symbol, f'[{int(r[1:])}*]')
            else:
                smiles = smiles.replace(symbol, '*')
    # For unknown tokens (i.e. rdkit cannot parse), replace them with [{isotope}*], where isotope is an identifier.
    tokens = atomwise_tokenizer(smiles)
    new_tokens = []
    mappings = {}  # isotope : symbol
    isotope = 50
    for token in tokens:
        if token[0] == '[':
            if token[1:-1] in ABBREVIATIONS or Chem.AtomFromSmiles(token) is None:
                while f'[{isotope}*]' in smiles or f'[{isotope}*]' in new_tokens:
                    isotope += 1
                placeholder = f'[{isotope}*]'
                mappings[isotope] = token[1:-1]
                new_tokens.append(placeholder)
                continue
        new_tokens.append(token)
    smiles = ''.join(new_tokens)
    return smiles, mappings


def convert_smiles_to_mol(smiles):
    if smiles is None or smiles == '':
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        return None
    return mol


BOND_TYPES = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}


def _num_swaps_to_interconvert(orders):
    n = len(orders)
    seen = [False] * n
    nswaps = 0
    for i in range(n):
        if not seen[i]:
            j = i
            while orders[j] != i:
                j = orders[j]
                if j >= n:
                    raise ValueError("_num_swaps_to_interconvert: index outside range")
                seen[j] = True
                nswaps += 1
    return nswaps

    
def _expand_functional_group(mol, mappings, debug=True):
    def _need_expand(mol, mappings):
        return any([len(Chem.GetAtomAlias(atom)) > 0 for atom in mol.GetAtoms()]) or len(mappings) > 0

    if _need_expand(mol, mappings):
        mol_w = Chem.RWMol(mol)
        num_atoms = mol_w.GetNumAtoms()

        # Reset radical electrons on all atoms
        for atom in mol_w.GetAtoms():
            atom.SetNumRadicalElectrons(0)

        atoms_to_remove = []

        for i in range(num_atoms):
            atom = mol_w.GetAtomWithIdx(i)
            if atom.GetSymbol() != '*':
                continue

            symbol = Chem.GetAtomAlias(atom)
            isotope = atom.GetIsotope()
            if isotope > 0 and isotope in mappings:
                symbol = mappings[isotope]

            if not (isinstance(symbol, str) and len(symbol) > 0):
                continue

            # R-group markers (R1/R2 etc.) are not expanded
            if symbol in RGROUP_SYMBOLS:
                continue

            bonds = atom.GetBonds()
            sub_smiles = get_smiles_from_symbol(symbol, mol_w, atom, bonds)

            # Get the functional group molecule from SMILES
            mol_r = convert_smiles_to_mol(sub_smiles)
            if mol_r is None:
                # If it can't be expanded, treat it as a regular C (or keep it as *, depending on your logic)
                atom.SetIsotope(0)
                continue

            # ====== Record original bond info & potentially affected chiral centers ======
            bond_infos = []
            chiral_centers_affected = set()
            bonds_list = list(bonds)

            for bond in bonds_list:
                adj_idx = bond.GetOtherAtomIdx(i)
                bond_infos.append(
                    (
                        adj_idx,
                        int(round(bond.GetBondTypeAsDouble())),
                        bond.GetBondDir()
                    )
                )
                adj_atom = mol_w.GetAtomWithIdx(adj_idx)
                if adj_atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
                    chiral_centers_affected.add(adj_idx)

            # ====== molzip_like style: mark neighbor order of chiral centers before connecting ======
            chiral_mark_dict = {}  # {chiral_idx: mark_name}
            for chiral_idx in chiral_centers_affected:
                chiral_atom = mol_w.GetAtomWithIdx(chiral_idx)
                tag = chiral_atom.GetChiralTag()
                if tag not in (
                    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                ):
                    continue

                mark_name = f"__expand_chiral_mark_{chiral_idx}"
                chiral_mark_dict[chiral_idx] = mark_name

                neighbors_before = list(chiral_atom.GetNeighbors())
                order = 0
                for nbr in neighbors_before:
                    nbr.SetIntProp(mark_name, order)
                    order += 1

                if debug:
                    print(f"  Marking neighbor order of chiral center {chiral_idx} (before expansion)")
                    for idx_nb, nbr in enumerate(neighbors_before):
                        if nbr.HasProp(mark_name):
                            print(
                                f"    Neighbor {nbr.GetIdx()}: order {nbr.GetIntProp(mark_name)} "
                                f"(the {idx_nb}-th in GetNeighbors order)"
                            )

            if debug:
                print(f"Expanding functional group {symbol} (atom {i})")
                print(f"  bond_infos: {bond_infos}")
                print(f"  chiral_centers_affected: {chiral_centers_affected}")

            # ====== Break all bonds between * and the main body, and "remember" bond order via radicals ======
            adjacent_indices = [bond.GetOtherAtomIdx(i) for bond in bonds_list]
            for adjacent_idx in adjacent_indices:
                mol_w.RemoveBond(i, adjacent_idx)

            adjacent_atoms = [mol_w.GetAtomWithIdx(adj_idx) for adj_idx in adjacent_indices]
            for adjacent_atom, bond in zip(adjacent_atoms, bonds_list):
                adjacent_atom.SetNumRadicalElectrons(int(bond.GetBondTypeAsDouble()))

            bonding_atoms_w = adjacent_indices  # connection points on the main body side

            if debug:
                print(f"  Main molecule connection points (bonding_atoms_w): {bonding_atoms_w}")

            # ====== Analyze connection point order in sub_smiles (functional group side) ======
            sub_smiles_atoms = []
            if sub_smiles:
                try:
                    temp_mol = Chem.MolFromSmiles(sub_smiles)
                    if temp_mol:
                        for atm in temp_mol.GetAtoms():
                            if atm.GetNumRadicalElectrons() > 0:
                                sub_smiles_atoms.append(atm.GetIdx())
                        # If the starting atom of the SMILES has no radical, but you treat the first atom as a connection point too, add it
                        if sub_smiles.startswith('*') or sub_smiles.startswith('['):
                            first_atom = temp_mol.GetAtomWithIdx(0)
                            if first_atom.GetNumRadicalElectrons() == 0 and 0 not in sub_smiles_atoms:
                                sub_smiles_atoms.insert(0, 0)
                except Exception as e:
                    if debug:
                        print(f"  Failed to parse sub_smiles: {e}")

            bonding_atoms_r = []

            # Method 1: determine connection points by radical order in sub_smiles
            if sub_smiles and len(sub_smiles_atoms) > 0:
                base_idx = mol_w.GetNumAtoms()
                for star_idx in sub_smiles_atoms:
                    # star_idx is the atom index in mol_r
                    bonding_atoms_r.append(base_idx + star_idx)

            # Method 2: fallback: default the first atom is the main connection point
            if len(bonding_atoms_r) == 0:
                base_idx = mol_w.GetNumAtoms()
                bonding_atoms_r = [base_idx]
                for atm in mol_r.GetAtoms():
                    if atm.GetNumRadicalElectrons() and atm.GetIdx() > 0:
                        bonding_atoms_r.append(base_idx + atm.GetIdx())

            if debug:
                print(f"  Functional group connection points (bonding_atoms_r estimated): {bonding_atoms_r}")
                print(f"  sub_smiles: {sub_smiles}")
                print(f"  sub_smiles_atoms: {sub_smiles_atoms}")

            # ====== Combine the main body with the functional group ======
            combo = Chem.CombineMols(mol_w, mol_r)
            mol_w = Chem.RWMol(combo)

            # ====== Decide the final paired target_atoms (functional group side connection points) ======
            target_atoms = []
            if len(bonding_atoms_r) == len(bonding_atoms_w):
                target_atoms = bonding_atoms_r
                if debug:
                    print(f"  Connection points count matches, matching in order: {bonding_atoms_w} -> {target_atoms}")
            elif len(bonding_atoms_r) >= len(bonding_atoms_w):
                target_atoms = bonding_atoms_r[:len(bonding_atoms_w)]
                if debug:
                    print(f"  More functional group connection points, taking first {len(bonding_atoms_w)}: {target_atoms}")
            else:
                if bonding_atoms_r:
                    target_atoms = bonding_atoms_r + [bonding_atoms_r[-1]] * (
                        len(bonding_atoms_w) - len(bonding_atoms_r)
                    )
                else:
                    # Extreme fallback: if nothing can be found, use the last atom
                    target_atoms = [mol_w.GetNumAtoms() - 1] * len(bonding_atoms_w)
                if debug:
                    print(f"  Fewer functional group connection points, repeating the last one: {target_atoms}")

            # ====== Add bonds + inherit direction + propagate chirality marks ======
            for info, target_idx in zip(bond_infos, target_atoms):
                adj_idx, order_val, bond_dir = info
                order_val = max(1, min(3, order_val))
                mol_w.GetAtomWithIdx(adj_idx).SetNumRadicalElectrons(order_val)

                # Avoid adding a duplicate bond
                existing_bond = mol_w.GetBondBetweenAtoms(adj_idx, target_idx)
                if existing_bond is None:
                    mol_w.AddBond(
                        adj_idx,
                        target_idx,
                        order=BOND_TYPES.get(order_val, Chem.rdchem.BondType.SINGLE),
                    )

                new_bond = mol_w.GetBondBetweenAtoms(adj_idx, target_idx)
                if new_bond is not None and bond_dir != Chem.BondDir.NONE:
                    new_bond.SetBondDir(bond_dir)

                # ====== Core fix: inherit chiral neighbor order marks from dummy(*) ======
                # i is the index of the '*' atom currently being expanded
                dummy_atom = mol_w.GetAtomWithIdx(i)
                for chiral_idx, mark_name in chiral_mark_dict.items():
                    if dummy_atom.HasProp(mark_name):
                        adj_order = dummy_atom.GetIntProp(mark_name)
                        target_atom = mol_w.GetAtomWithIdx(target_idx)
                        target_atom.SetIntProp(mark_name, adj_order)
                        if debug:
                            print(
                                f"  Transferring mark: dummy atom {i} (order {adj_order}, chiral center {chiral_idx})"
                                f" -> new atom {target_idx}"
                            )

            # ====== Restore chirality after connection (same idea as molzip_like) ======
            for chiral_idx in chiral_centers_affected:
                if chiral_idx not in chiral_mark_dict:
                    continue

                mark_name = chiral_mark_dict[chiral_idx]
                chiral_atom = mol_w.GetAtomWithIdx(chiral_idx)
                tag = chiral_atom.GetChiralTag()

                if tag not in (
                    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                ):
                    continue

                neighbors_after = list(chiral_atom.GetNeighbors())
                orders_after = []
                all_have_mark = True
                for nbr in neighbors_after:
                    if not nbr.HasProp(mark_name):
                        all_have_mark = False
                        break
                    orders_after.append(nbr.GetIntProp(mark_name))

                if all_have_mark and len(orders_after) > 0:
                    if debug:
                        print(f"  Chiral center {chiral_idx}: neighbor order after connection {orders_after}")
                    try:
                        if set(orders_after) == set(range(len(orders_after))):
                            nswaps = _num_swaps_to_interconvert(orders_after)
                            if debug:
                                print(f"  Number of swaps: {nswaps}")
                            if nswaps % 2 == 1:
                                if tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
                                    chiral_atom.SetChiralTag(
                                        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW
                                    )
                                    if debug:
                                        print(f"  Flipping chiral center {chiral_idx}: CW -> CCW")
                                elif tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
                                    chiral_atom.SetChiralTag(
                                        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW
                                    )
                                    if debug:
                                        print(f"  Flipping chiral center {chiral_idx}: CCW -> CW")
                        else:
                            if debug:
                                print(
                                    f"  Warning: Neighbor marks of chiral center {chiral_idx} are not a valid permutation: {orders_after}"
                                )
                    except Exception as e:
                        if debug:
                            print(f"  Error calculating number of swaps (chiral center {chiral_idx}): {e}")
                else:
                    if debug:
                        missing = [
                            nbr.GetIdx()
                            for nbr in neighbors_after
                            if not nbr.HasProp(mark_name)
                        ]
                        print(
                            f"  Chiral center {chiral_idx}: Some neighbors lack marks, cannot determine chirality change (neighbors missing marks: {missing})"
                        )

            # Clear temporary radicals
            for atm_idx in bonding_atoms_w:
                mol_w.GetAtomWithIdx(atm_idx).SetNumRadicalElectrons(0)
            for atm_idx in bonding_atoms_r:
                if 0 <= atm_idx < mol_w.GetNumAtoms():
                    mol_w.GetAtomWithIdx(atm_idx).SetNumRadicalElectrons(0)

            # Local sanitize (do not force recomputation of stereochemistry)
            try:
                Chem.SanitizeMol(mol_w)
            except Exception as e:
                if debug:
                    print(f"Warning: Failed to sanitize after expanding {symbol}: {e}")

            # Record the '*' atom to be removed
            atoms_to_remove.append(i)

        # ====== Remove all * atoms (starting from the largest index) ======
        atoms_to_remove = sorted(set(atoms_to_remove), reverse=True)
        for idx in atoms_to_remove:
            if idx < mol_w.GetNumAtoms():
                mol_w.RemoveAtom(idx)

        # Clean up temporary chirality mark properties
        for atom in mol_w.GetAtoms():
            for prop in list(atom.GetPropNames()):
                if prop.startswith("__expand_chiral_mark"):
                    atom.ClearProp(prop)

        # Final sanitize
        try:
            Chem.SanitizeMol(mol_w)
        except Exception as e:
            if debug:
                print("Warning: Failed to final sanitize after expanding functional groups:", e)

        smiles = Chem.MolToSmiles(mol_w, isomericSmiles=True)
        mol = mol_w.GetMol()
    else:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        mol = mol

    return smiles, mol


def _convert_graph_to_smiles(coords, symbols, edges, image=None, debug=False):
    print(f"symbols: {symbols}")
    mol = Chem.RWMol()
    n = len(symbols)
    ids = []
    for i in range(n):
        symbol = symbols[i]
        if symbol[0] == '[':
            symbol = symbol[1:-1]
        if symbol in RGROUP_SYMBOLS:
            atom = Chem.Atom("*")
            if symbol[0] == 'R' and symbol[1:].isdigit():
                atom.SetIsotope(int(symbol[1:]))
            Chem.SetAtomAlias(atom, symbol)
        elif symbol in ABBREVIATIONS:
            atom = Chem.Atom("*")
            Chem.SetAtomAlias(atom, symbol)
        else:
            try:  # try to get SMILES of atom
                atom = Chem.AtomFromSmiles(symbols[i])
                atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
            except:  # otherwise, abbreviation or condensed formula
                atom = Chem.Atom("*")
                Chem.SetAtomAlias(atom, symbol)

        if atom.GetSymbol() == '*':
            atom.SetProp('molFileAlias', symbol)

        idx = mol.AddAtom(atom)
        assert idx == i
        ids.append(idx)

    for i in range(n):
        for j in range(i + 1, n):
            if edges[i][j] == 1:
                mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
            elif edges[i][j] == 2:
                mol.AddBond(ids[i], ids[j], Chem.BondType.DOUBLE)
            elif edges[i][j] == 3:
                mol.AddBond(ids[i], ids[j], Chem.BondType.TRIPLE)
            elif edges[i][j] == 4:
                mol.AddBond(ids[i], ids[j], Chem.BondType.AROMATIC)
            elif edges[i][j] == 5:
                mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
                mol.GetBondBetweenAtoms(ids[i], ids[j]).SetBondDir(Chem.BondDir.BEGINWEDGE)
            elif edges[i][j] == 6:
                mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
                mol.GetBondBetweenAtoms(ids[i], ids[j]).SetBondDir(Chem.BondDir.BEGINDASH)

    try:
        pred_smiles = rdkit.Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception as e:
        pred_smiles = '<invalid>'
    #print(f"initial_SMILES: {smiles}")
    try:
        # TODO: move to an util function
        if image is not None:
            height, width, _ = image.shape
            ratio = width / height
            coords = [[x * ratio * 10, y * 10] for x, y in coords]
        
        mol = _verify_chirality(mol, coords, symbols, edges, debug)
        smiles1 = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        #print(f"after_chirality_SMILES: {smiles1}")
        # molblock is obtained before expanding func groups, otherwise the expanded group won't have coordinates.
        # TODO: make sure molblock has the abbreviation information
        pred_molblock = Chem.MolToMolBlock(mol)
        pred_smiles, mol = _expand_functional_group(mol, {}, debug)
        #print(f"after_expansion_SMILES: {pred_smiles}")
        #pred_smiles = align_chirality(smiles1, pred_smiles)
        #print(f"final_SMILES: {pred_smiles}\n")
        mol = Chem.MolFromSmiles(pred_smiles)
        success = True
    except Exception as e:
        if debug:
            print(traceback.format_exc())
        pred_molblock = ''
        success = False

    if debug:
        return pred_smiles, pred_molblock, mol, success
    return pred_smiles, pred_molblock, success


def convert_graph_to_smiles(coords, symbols, edges, images=None, num_workers=16):
    if images is None:
        args_zip = zip(coords, symbols, edges)
    else:
        args_zip = zip(coords, symbols, edges, images)

    if num_workers <= 1:
        results = itertools.starmap(_convert_graph_to_smiles, args_zip)
        results = list(results)
    else:
        with multiprocessing.Pool(num_workers) as p:
            results = p.starmap(_convert_graph_to_smiles, args_zip, chunksize=128)

    smiles_list, molblock_list, success = zip(*results)
    r_success = np.mean(success)
    return smiles_list, molblock_list, r_success


def _postprocess_smiles(smiles, coords=None, symbols=None, edges=None, molblock=False, debug=False):
    if type(smiles) is not str or smiles == '':
        return '', False
    mol = None
    pred_molblock = ''
    try:
        pred_smiles = smiles
        pred_smiles, mappings = _replace_functional_group(pred_smiles)
        if coords is not None and symbols is not None and edges is not None:
            pred_smiles = pred_smiles.replace('@', '').replace('/', '').replace('\\', '')
            mol = Chem.RWMol(Chem.MolFromSmiles(pred_smiles, sanitize=False))
            mol = _verify_chirality(mol, coords, symbols, edges, debug)
        else:
            mol = Chem.MolFromSmiles(pred_smiles, sanitize=False)
        # pred_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        if molblock:
            pred_molblock = Chem.MolToMolBlock(mol)
        pred_smiles, mol = _expand_functional_group(mol, mappings)
        success = True
    except Exception as e:
        if debug:
            print(traceback.format_exc())
        pred_smiles = smiles
        pred_molblock = ''
        success = False
    if debug:
        return pred_smiles, pred_molblock, mol, success
    return pred_smiles, pred_molblock, success


def postprocess_smiles(smiles, coords=None, symbols=None, edges=None, molblock=False, num_workers=16):
    with multiprocessing.Pool(num_workers) as p:
        if coords is not None and symbols is not None and edges is not None:
            results = p.starmap(_postprocess_smiles, zip(smiles, coords, symbols, edges), chunksize=128)
        else:
            results = p.map(_postprocess_smiles, smiles, chunksize=128)
    smiles_list, molblock_list, success = zip(*results)
    r_success = np.mean(success)
    return smiles_list, molblock_list, r_success


def _keep_main_molecule(smiles, debug=False):
    try:
        mol = Chem.MolFromSmiles(smiles)
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            num_atoms = [m.GetNumAtoms() for m in frags]
            main_mol = frags[np.argmax(num_atoms)]
            smiles = Chem.MolToSmiles(main_mol)
    except Exception as e:
        if debug:
            print(traceback.format_exc())
    return smiles


def keep_main_molecule(smiles, num_workers=16):
    with multiprocessing.Pool(num_workers) as p:
        results = p.map(_keep_main_molecule, smiles, chunksize=128)
    return results

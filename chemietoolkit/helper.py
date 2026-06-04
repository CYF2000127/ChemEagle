import re
from typing import List, Optional
import sys
import json
import numpy as np
from PIL import Image
import os
import base64
from typing import Optional, Dict, Any

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    Chem = None
    RDKIT_AVAILABLE = False

def _validate_and_fix_smiles(smiles: str) -> str:
    if not RDKIT_AVAILABLE or not smiles:
        return smiles
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return smiles  # SMILES is valid, no fix needed
    except Exception:
        pass
    
    import re
    
    n_pattern = r'(?<!\[)N(?!\])'
    matches = list(re.finditer(n_pattern, smiles))
    
    if not matches:
        return smiles

    for match in matches:
        pos = match.start()
        test_smiles = smiles[:pos] + '[N+]' + smiles[pos+1:]
        
        try:
            mol = Chem.MolFromSmiles(test_smiles)
            if mol is not None:
                print(f"[SMILES Fix] Fixed invalid SMILES by adding charge to N:\n  Original: {smiles}\n  Fixed:    {test_smiles}")
                return test_smiles
        except Exception:
            continue

    return smiles


def fallback_validate_and_fix_smiles_in_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if key == 'smiles' and isinstance(value, str):
                # Fix SMILES
                result[key] = _validate_and_fix_smiles(value)
            elif isinstance(value, (dict, list)):
                # Process recursively
                result[key] = fallback_validate_and_fix_smiles_in_dict(value)
            else:
                result[key] = value
        return result
    elif isinstance(data, list):
        return [fallback_validate_and_fix_smiles_in_dict(item) for item in data]
    else:
        return data


# ============================================================================
# PubChem fallback for condition (solvent/reagent/etc.) SMILES
# ----------------------------------------------------------------------------
# For text-only condition entries like "DMSO", "THF", "1,4-dioxane", or
# stoichiometric prefixes like "10 mol% Cs2CO3", look up a canonical SMILES on
# PubChem and override the agent's `smiles` field if a hit is returned. If the
# lookup fails, the original agent output is kept.
# ============================================================================
import urllib.parse as _urlparse
import urllib.request as _urlreq

_PUBCHEM_SMILES_CACHE: Dict[str, Optional[str]] = {}

# ---------------------------------------------------------------------------
# Persistent cache (survives process restarts).
# Override path with env var CHEMEAGLE_CACHE_DIR.
# ---------------------------------------------------------------------------
_CACHE_DIR = os.path.expanduser(os.environ.get('CHEMEAGLE_CACHE_DIR', '~/.cache/chemeagle'))
_CACHE_FILE = os.path.join(_CACHE_DIR, 'name2smiles.json')
_CACHE_VERSION = 1


def _load_persistent_cache() -> None:
    try:
        with open(_CACHE_FILE, 'r', encoding='utf-8') as f:
            blob = json.load(f)
        if isinstance(blob, dict) and blob.get('version') == _CACHE_VERSION:
            entries = blob.get('entries', {})
            if isinstance(entries, dict):
                _PUBCHEM_SMILES_CACHE.update(entries)
                print(f"[name->SMILES cache] loaded {len(entries)} entries from {_CACHE_FILE}")
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[name->SMILES cache] load failed: {e}")


def _save_persistent_cache() -> None:
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        tmp = _CACHE_FILE + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump({'version': _CACHE_VERSION, 'entries': _PUBCHEM_SMILES_CACHE}, f)
        os.replace(tmp, _CACHE_FILE)
    except Exception as e:
        print(f"[name->SMILES cache] save failed: {e}")


# ---------------------------------------------------------------------------
# Curated alias map: queried BEFORE PubChem.
# Covers high-frequency abbreviations that PubChem fails on, or whose strict
# synonym check returns wrong answers (e.g. NaH -> niacin). Keys are matched
# case-insensitively. Extend liberally; SMILES should be canonical and parse
# under RDKit.
# ---------------------------------------------------------------------------
_ALIAS_SMILES: Dict[str, str] = {
    # halogenated solvents
    'dcm': 'ClCCl', 'dichloromethane': 'ClCCl', 'ch2cl2': 'ClCCl', 'methylene chloride': 'ClCCl',
    'dce': 'ClCCCl', '1,2-dce': 'ClCCCl', '1,2-dichloroethane': 'ClCCCl',
    'chcl3': 'C(Cl)(Cl)Cl', 'chloroform': 'C(Cl)(Cl)Cl',
    'ccl4': 'C(Cl)(Cl)(Cl)Cl', 'carbon tetrachloride': 'C(Cl)(Cl)(Cl)Cl',
    # ethers
    'thf': 'C1CCOC1', 'tetrahydrofuran': 'C1CCOC1',
    '2-mecthf': 'CC1CCCO1', '2-methyltetrahydrofuran': 'CC1CCCO1', 'metthf': 'CC1CCCO1',
    'et2o': 'CCOCC', 'diethyl ether': 'CCOCC', 'ether': 'CCOCC',
    'mtbe': 'COC(C)(C)C', 'tbme': 'COC(C)(C)C',
    'dme': 'COCCOC', '1,2-dimethoxyethane': 'COCCOC', 'glyme': 'COCCOC',
    'diglyme': 'COCCOCCOC', 'triglyme': 'COCCOCCOCCOC',
    'dioxane': 'C1COCCO1', '1,4-dioxane': 'C1COCCO1', '1,3-dioxane': 'C1CCOCO1',
    # amides / sulfoxides / nitriles
    'dmf': 'CN(C)C=O', 'n,n-dimethylformamide': 'CN(C)C=O',
    'dma': 'CN(C)C(C)=O', 'dmac': 'CN(C)C(C)=O', 'n,n-dimethylacetamide': 'CN(C)C(C)=O',
    'nmp': 'CN1CCCC1=O', 'n-methylpyrrolidone': 'CN1CCCC1=O',
    'dmso': 'CS(=O)C', 'dimethyl sulfoxide': 'CS(=O)C',
    'mecn': 'CC#N', 'acn': 'CC#N', 'acetonitrile': 'CC#N',
    'hmpa': 'CN(C)P(=O)(N(C)C)N(C)C',
    # alcohols / water
    'meoh': 'CO', 'methanol': 'CO',
    'etoh': 'CCO', 'ethanol': 'CCO',
    'iproh': 'CC(C)O', 'i-proh': 'CC(C)O', 'ipa': 'CC(C)O', 'isopropanol': 'CC(C)O', '2-propanol': 'CC(C)O',
    't-buoh': 'CC(C)(C)O', 'tert-butanol': 'CC(C)(C)O', 'tbuoh': 'CC(C)(C)O',
    'h2o': 'O', 'water': 'O',
    'd2o': '[2H]O[2H]',
    # esters
    'etoac': 'CCOC(=O)C', 'ethyl acetate': 'CCOC(=O)C', 'ea': 'CCOC(=O)C',
    'meoac': 'COC(=O)C', 'methyl acetate': 'COC(=O)C',
    # fluorinated
    'tfa': 'OC(=O)C(F)(F)F', 'trifluoroacetic acid': 'OC(=O)C(F)(F)F',
    'tfaa': 'O=C(OC(=O)C(F)(F)F)C(F)(F)F',
    'tfe': 'OCC(F)(F)F', '2,2,2-trifluoroethanol': 'OCC(F)(F)F',
    'hfip': 'OC(C(F)(F)F)C(F)(F)F', 'hexafluoroisopropanol': 'OC(C(F)(F)F)C(F)(F)F',
    # hydrocarbons
    'hexane': 'CCCCCC', 'hexanes': 'CCCCCC', 'n-hexane': 'CCCCCC',
    'pentane': 'CCCCC', 'heptane': 'CCCCCCC',
    'cyclohexane': 'C1CCCCC1',
    'benzene': 'c1ccccc1',
    'toluene': 'Cc1ccccc1',
    'mesitylene': 'Cc1cc(C)cc(C)c1', '1,3,5-trimethylbenzene': 'Cc1cc(C)cc(C)c1',
    'pyridine': 'c1ccncc1',
    # amines / bases (organic)
    'tea': 'CCN(CC)CC', 'et3n': 'CCN(CC)CC', 'triethylamine': 'CCN(CC)CC',
    'dipea': 'CCN(C(C)C)C(C)C', 'i-pr2net': 'CCN(C(C)C)C(C)C', "hunig's base": 'CCN(C(C)C)C(C)C',
    'dbu': 'C1CCC2=NCCCN2CC1',
    'dbn': 'C1CCC2=NCCN2C1',
    'dabco': 'C1CN2CCN1CC2',
    'dmap': 'CN(C)c1ccncc1', '4-dmap': 'CN(C)c1ccncc1',
    'tmeda': 'CN(C)CCN(C)C',
    'tba': 'CCCCN(CCCC)CCCC',
    # strong bases / hydrides
    'nah': '[H-].[Na+]', 'sodium hydride': '[H-].[Na+]',
    'kh': '[H-].[K+]',
    'lda': 'CC(C)[N-]C(C)C.[Li+]',
    'lihmds': 'C[Si](C)(C)[N-][Si](C)(C)C.[Li+]',
    'khmds': 'C[Si](C)(C)[N-][Si](C)(C)C.[K+]',
    'nahmds': 'C[Si](C)(C)[N-][Si](C)(C)C.[Na+]',
    'tbaf': '[F-].CCCC[N+](CCCC)(CCCC)CCCC',
    'tbab': '[Br-].CCCC[N+](CCCC)(CCCC)CCCC',
    'tbai': '[I-].CCCC[N+](CCCC)(CCCC)CCCC',
    'tbac': '[Cl-].CCCC[N+](CCCC)(CCCC)CCCC',
    'nbu4nf': '[F-].CCCC[N+](CCCC)(CCCC)CCCC',
    # inorganic bases / salts
    'naoh': '[Na+].[OH-]', 'sodium hydroxide': '[Na+].[OH-]',
    'koh': '[K+].[OH-]',
    'k2co3': '[K+].[K+].[O-]C([O-])=O', 'potassium carbonate': '[K+].[K+].[O-]C([O-])=O',
    'cs2co3': '[Cs+].[Cs+].[O-]C([O-])=O', 'cesium carbonate': '[Cs+].[Cs+].[O-]C([O-])=O',
    'na2co3': '[Na+].[Na+].[O-]C([O-])=O',
    'nahco3': '[Na+].OC([O-])=O',
    'khco3': '[K+].OC([O-])=O',
    'na2so4': '[Na+].[Na+].[O-]S(=O)(=O)[O-]',
    'mgso4': '[Mg+2].[O-]S(=O)(=O)[O-]',
    'cacl2': '[Cl-].[Cl-].[Ca+2]',
    # acids
    'hcl': 'Cl', 'h2so4': 'OS(=O)(=O)O', 'hno3': 'O[N+](=O)[O-]',
    'acoh': 'CC(=O)O', 'acetic acid': 'CC(=O)O', 'hoac': 'CC(=O)O',
    # palladium / phosphine ligands
    'pd(oac)2': 'CC(=O)O[Pd]OC(C)=O', 'palladium(ii) acetate': 'CC(=O)O[Pd]OC(C)=O',
    'pd(tfa)2': 'O=C(O[Pd]OC(=O)C(F)(F)F)C(F)(F)F',
    'pdcl2': '[Cl-].[Cl-].[Pd+2]',
    'cui': '[Cu]I',
    'cuoac': 'CC(=O)O[Cu]',
    'dppe': 'C(P(c1ccccc1)c1ccccc1)CP(c1ccccc1)c1ccccc1',
    'dppp': 'C(CP(c1ccccc1)c1ccccc1)CP(c1ccccc1)c1ccccc1',
    'dppb': 'C(CCP(c1ccccc1)c1ccccc1)CP(c1ccccc1)c1ccccc1',
    # misc / strict-reject overrides
    'toluene (anhydrous)': 'Cc1ccccc1',
}
# Sanity: validate each alias parses (printed once on import).
if RDKIT_AVAILABLE:
    _bad_aliases = []
    for _k, _v in list(_ALIAS_SMILES.items()):
        try:
            if Chem.MolFromSmiles(_v) is None:
                _bad_aliases.append(_k)
        except Exception:
            _bad_aliases.append(_k)
    if _bad_aliases:
        print(f"[alias] WARNING: invalid SMILES dropped: {_bad_aliases}")
        for _k in _bad_aliases:
            _ALIAS_SMILES.pop(_k, None)

_load_persistent_cache()

# Strip leading stoichiometry like "10 mol%", "2.0 equiv", "5 mg", "0.5 M" so
# "10 mol% Cs2CO3" -> "Cs2CO3"
_STOICH_PREFIX_RE = re.compile(
    r'^\s*\d+(?:\.\d+)?\s*(?:mol\s*%|wt\s*%|vol\s*%|%|equiv\.?|eq\.?|mol|mmol|mg|kg|g|mL|L|M|N|x|×)\s*[.,:;]?\s*',
    re.IGNORECASE,
)


def _strip_stoichiometry(text: str) -> str:
    """Remove leading numeric/unit prefixes so a chemical name is exposed.

    Examples
    --------
    "10 mol% Cs2CO3"       -> "Cs2CO3"
    "2.0 equiv K2CO3"      -> "K2CO3"
    "0.5 M HCl in dioxane" -> "HCl in dioxane"  (caller may further split)
    """
    if not text:
        return text
    out = text.strip()
    # Iteratively strip up to a few leading quantity tokens (e.g. "2.0 equiv 1.5 mol% X")
    for _ in range(3):
        new = _STOICH_PREFIX_RE.sub('', out)
        if new == out:
            break
        out = new
    # Drop trailing parentheticals like " (cat.)" or " (anhydrous)"
    out = re.sub(r'\s*\([^)]*\)\s*$', '', out).strip()
    return out


def _candidate_chem_names(item: Dict[str, Any]) -> List[str]:
    """Collect candidate chemical names from a condition-like item."""
    cands: List[str] = []
    seen = set()

    def add(s):
        if not isinstance(s, str):
            return
        s = s.strip().strip(',;:')
        if not s or s in seen:
            return
        seen.add(s)
        cands.append(s)

    for key in ('text', 'name'):
        v = item.get(key)
        if isinstance(v, str):
            add(v)
            add(_strip_stoichiometry(v))
        elif isinstance(v, list):
            for t in v:
                if isinstance(t, str):
                    add(t)
                    add(_strip_stoichiometry(t))
    return cands


# Mixed solvent / co-solvent helpers
# Splits things like:
#   "ethylene glycol/toluene = 2:1"   -> ["ethylene glycol", "toluene"]
#   "THF/H2O (4:1)"                    -> ["THF", "H2O"]
#   "DCM/MeOH 9:1"                     -> ["DCM", "MeOH"]
#   "dioxane and water"                -> ["dioxane", "water"]
_MIX_SPLIT_RE = re.compile(r'\s*(?:/|\s+and\s+|\s+or\s+)\s*', re.IGNORECASE)
# Strips trailing ratio specifications: "= 2:1", "(4:1)", " 9:1", " 9 : 1"
_RATIO_TAIL_RE = re.compile(
    r'\s*(?:=\s*)?\(?\s*\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?(?:\s*:\s*\d+(?:\.\d+)?)*\s*\)?\s*$'
)


def _split_mixed_solvent(text: str) -> List[str]:
    """Return component names if ``text`` looks like a mixed solvent, else []."""
    if not isinstance(text, str):
        return []
    s = _strip_stoichiometry(text)
    s = _RATIO_TAIL_RE.sub('', s).strip()
    if not s:
        return []
    parts = [p.strip().strip(',;:') for p in _MIX_SPLIT_RE.split(s) if p and p.strip()]
    # Require at least 2 non-trivial parts; reject if any part still looks
    # like a ratio fragment or is too long to be a single chemical name.
    if len(parts) < 2:
        return []
    if any(re.match(r'^\s*\d', p) for p in parts):
        return []
    if any(len(p) > 60 for p in parts):
        return []
    return parts


# Generic mixture components that intentionally have no SMILES
# (buffers, generic aqueous phase, pH descriptors, etc). When seen inside a
# mixed-solvent string we silently skip them so the rest of the mixture can
# still resolve. We do NOT skip them when they appear as a stand-alone name.
_GENERIC_MIX_COMPONENTS = {
    'buffer', 'buffers', 'buffer solution', 'aqueous buffer',
    'phosphate buffer', 'tris buffer', 'hepes buffer', 'mes buffer',
    'pbs', 'tris', 'hepes', 'mes',
    'aq', 'aq.', 'aqueous', 'aqueous solution',
    'brine', 'sat. brine', 'saturated brine',
    'solvent', 'co-solvent', 'cosolvent',
}
_PH_TOKEN_RE = re.compile(r'^\s*ph\s*\d', re.IGNORECASE)


def _is_generic_mix_component(p: str) -> bool:
    if not isinstance(p, str):
        return False
    k = p.strip().lower()
    if not k:
        return True
    if k in _GENERIC_MIX_COMPONENTS:
        return True
    if _PH_TOKEN_RE.match(k):           # "pH 7", "pH 7.4 buffer"
        return True
    if 'buffer' in k and len(k) <= 30:  # e.g. "KPi buffer", "citrate buffer"
        return True
    return False


def _resolve_mixed_solvent_smiles(text: str) -> Optional[str]:
    """Try to resolve a mixed-solvent string into a dot-joined SMILES.

    Lenient mode: components that are generic / unresolvable on purpose
    (e.g. ``buffer``, ``aq``, ``pH 7 buffer``) are silently skipped. As long as
    at least one component resolves we return the dot-joined SMILES of the
    resolved components. Returns ``None`` if the input is not a mixture or no
    component resolves.
    """
    parts = _split_mixed_solvent(text)
    if not parts:
        return None
    smiles: List[str] = []
    for p in parts:
        if _is_generic_mix_component(p):
            print(f"[MIX] skipping generic component '{p}'")
            continue
        smi = _resolve_name_to_smiles(p)
        if smi:
            smiles.append(smi)
        else:
            print(f"[MIX] unresolved component '{p}' in mixture '{text}'")
    if not smiles:
        return None
    return '.'.join(smiles)


def _normalize_chem_name(s: str) -> str:
    """Normalize a chemical name (preserves case)."""
    if not isinstance(s, str):
        return ''
    out = s.strip()
    # Collapse whitespace and remove dots, but keep case, digits, dashes,
    # commas, parentheses — all chemically meaningful (e.g. "1,4-dioxane").
    out = re.sub(r'\s+', '', out)
    out = out.replace('.', '')
    return out


def _has_mixed_case(s: str) -> bool:
    """True if s has BOTH a lower-case ASCII letter AND an upper-case one.

    Used to decide whether case must be respected during synonym matching:
    - Mixed-case formulas like ``NaH``, ``Cs2CO3``, ``Pd(OAc)2`` rely on case to
      distinguish elements; we require an exact case-sensitive match to avoid
      matching unrelated all-caps abbreviations (e.g. niacin's synonym
      ``NAH``).
    - All-lowercase ("toluene") or all-uppercase ("DMSO", "SODIUM HYDRIDE")
      inputs do not carry case meaning, so case-insensitive matching is safe.
    """
    if not isinstance(s, str):
        return False
    has_lower = any('a' <= ch <= 'z' for ch in s)
    has_upper = any('A' <= ch <= 'Z' for ch in s)
    return has_lower and has_upper


def _query_pubchem_smiles(name: str, timeout: float = 5.0) -> Optional[str]:
    """Query PubChem PUG-REST for a SMILES, with STRICT synonym verification.

    PubChem's name endpoint is fuzzy (e.g. "NaH" -> niacin). To avoid wrong
    overrides, this function:
      1. Resolves the query name to a CID via PUG-REST.
      2. Fetches the CID's synonyms.
      3. Accepts ONLY if the normalized query name exactly matches one of the
         synonyms (case/whitespace/dot insensitive).
      4. Then fetches SMILES for that CID.

    Returns None on miss, fuzzy match rejection, or any error.

    PubChem renamed the `CanonicalSMILES` property in 2025; we accept any of
    `SMILES` / `IsomericSMILES` / `CanonicalSMILES` / `ConnectivitySMILES`.

    Results (including misses) are cached in-process.
    """
    if not name:
        return None
    # Case-sensitive cache key: "NaH" and "nah" go through different code
    # paths in the strict synonym check, so they MUST cache separately.
    key = name.strip()
    if key in _PUBCHEM_SMILES_CACHE:
        cached = _PUBCHEM_SMILES_CACHE[key]
        print(f"[PubChem cache] hit: '{key}' -> {cached!r}")
        return cached
    encoded = _urlparse.quote(name.strip(), safe='')
    try:
        # 1) name -> CID(s)
        url_cids = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{encoded}/cids/JSON"
        )
        with _urlreq.urlopen(url_cids, timeout=timeout) as resp:
            cid_payload = json.loads(resp.read().decode('utf-8'))
        cids = cid_payload.get('IdentifierList', {}).get('CID', []) or []
        cid = cids[0] if cids else None
        if not cid:
            _PUBCHEM_SMILES_CACHE[key] = None
            return None

        # 2) CID -> synonyms; STRICT verify the query name is among them
        url_syn = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
            f"{cid}/synonyms/JSON"
        )
        with _urlreq.urlopen(url_syn, timeout=timeout) as resp:
            syn_payload = json.loads(resp.read().decode('utf-8'))
        info = syn_payload.get('InformationList', {}).get('Information', [])
        synonyms: List[str] = info[0].get('Synonym', []) if info else []
        wanted = _normalize_chem_name(name)
        norm_syns = {_normalize_chem_name(s) for s in synonyms}
        # Decide whether to compare case-sensitively.
        # - Mixed-case formula tokens (NaH, Cs2CO3, Pd(OAc)2) carry case info
        #   ➜ require an exact case-sensitive match.
        # - All-lowercase / all-uppercase inputs (toluene, DMSO, MeOH if all
        #   letters share one case) ➜ allow case-insensitive match.
        if _has_mixed_case(name):
            accepted = wanted in norm_syns
        else:
            wanted_ci = wanted.lower()
            accepted = any(s.lower() == wanted_ci for s in norm_syns)
        if not accepted:
            print(
                f"[PubChem strict] reject '{name}' -> CID {cid} "
                f"(no acceptable synonym match; e.g. {synonyms[:3]})"
            )
            _PUBCHEM_SMILES_CACHE[key] = None
            return None

        # 3) CID -> SMILES (try the new field names first, fall back to old)
        url_smi = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
            f"{cid}/property/SMILES,ConnectivitySMILES,CanonicalSMILES,IsomericSMILES/JSON"
        )
        with _urlreq.urlopen(url_smi, timeout=timeout) as resp:
            prop_payload = json.loads(resp.read().decode('utf-8'))
        props = prop_payload.get('PropertyTable', {}).get('Properties', [])
        smi = None
        if props:
            row = props[0]
            for k in ('SMILES', 'IsomericSMILES', 'CanonicalSMILES', 'ConnectivitySMILES'):
                v = row.get(k)
                if isinstance(v, str) and v.strip():
                    smi = v.strip()
                    break
        if smi and RDKIT_AVAILABLE:
            try:
                if Chem.MolFromSmiles(smi) is None:
                    smi = None
            except Exception:
                pass
        _PUBCHEM_SMILES_CACHE[key] = smi
        return smi
    except Exception as e:
        print(f"[PubChem] lookup failed for '{name}': {e}")
        _PUBCHEM_SMILES_CACHE[key] = None
        return None


def _query_opsin_smiles(name: str, timeout: float = 5.0) -> Optional[str]:
    """OPSIN (IUPAC name -> SMILES) fallback. Free, deterministic, name-only.

    Best at systematic IUPAC names like ``3-methylpyridine``, ``1,4-dioxane``.
    Returns ``None`` for non-IUPAC strings (404 from OPSIN) or any error.
    """
    if not isinstance(name, str):
        return None
    s = name.strip()
    if not s or len(s) > 200:
        return None
    enc = _urlparse.quote(s, safe='')
    url = f"https://opsin.ch.cam.ac.uk/opsin/{enc}.smi"
    try:
        with _urlreq.urlopen(url, timeout=timeout) as resp:
            text = resp.read().decode('utf-8', errors='ignore').strip()
        if not text or any(ch.isspace() for ch in text):
            return None
        if RDKIT_AVAILABLE:
            try:
                if Chem.MolFromSmiles(text) is None:
                    return None
            except Exception:
                return None
        return text
    except Exception:
        # 404 from OPSIN just means the name wasn't parseable; not noteworthy.
        return None


def _resolve_name_to_smiles(name: str) -> Optional[str]:
    """Unified name -> SMILES resolver.

    Order: curated alias map -> persistent cache -> PubChem (strict synonym) ->
    OPSIN -> ``molnextr.chemistry.resolve_symbol_to_smiles`` (OCSR shorthand /
    ArCHO / condensed formula). Caches hits; persists across process restarts.
    """
    if not isinstance(name, str) or not name.strip():
        return None
    key = name.strip()
    alias = _ALIAS_SMILES.get(key.lower())
    if alias:
        print(f"[alias] '{key}' -> {alias}")
        return alias
    if key in _PUBCHEM_SMILES_CACHE:
        cached = _PUBCHEM_SMILES_CACHE[key]
        if cached is not None:
            print(f"[name->SMILES cache] hit: '{key}' -> {cached!r}")
            return cached
        # Previously hit a negative cache (None): retry OPSIN/OCSR, since rules may have been updated
        print(f"[name->SMILES cache] retry negative for '{key}'")
        smi = None
    else:
        smi = _query_pubchem_smiles(key)  # writes _PUBCHEM_SMILES_CACHE[key]
    if smi is None:
        opsin_smi = _query_opsin_smiles(key)
        if opsin_smi:
            print(f"[OPSIN] '{key}' -> {opsin_smi}")
            smi = opsin_smi
            _PUBCHEM_SMILES_CACHE[key] = smi  # override the PubChem miss
    if smi is None:
        try:
            from molnextr.chemistry import resolve_symbol_to_smiles  # lazy
            ocsr_smi = resolve_symbol_to_smiles(key, use_llm=False)
        except Exception:
            ocsr_smi = None
        if ocsr_smi:
            print(f"[OCSR] '{key}' -> {ocsr_smi}")
            smi = ocsr_smi
            _PUBCHEM_SMILES_CACHE[key] = smi
    _save_persistent_cache()
    return smi


def _resolve_smiles_for_condition_item(item: Dict[str, Any]) -> None:
    """Look up a SMILES on PubChem for a condition item and write it back.

    Runs whenever ``role`` is ``solvent`` or ``reagent`` (case-insensitive),
    regardless of whether the item already carries a ``smiles`` key:

      - no ``smiles`` yet  -> add one if PubChem returns a hit
      - existing ``smiles`` -> overwrite with the PubChem hit (otherwise leave
        the original value untouched)
    """
    if not isinstance(item, dict):
        return
    role = item.get('role', '')
    if not isinstance(role, str) or role.strip().lower() not in {'solvent', 'reagent'}:
        return
    for name in _candidate_chem_names(item):
        smi = _resolve_name_to_smiles(name)
        if smi:
            old = item.get('smiles', '')
            if old != smi:
                tag = 'added' if 'smiles' not in item or not old else 'fallback'
                print(f"[PubChem {tag}] '{name}' -> {smi}" + (f" (was: '{old}')" if old else ''))
            item['smiles'] = smi
            return

    # Fallback: try to interpret the text as a mixed solvent and join components.
    raw_text = item.get('text') if isinstance(item.get('text'), str) else None
    if raw_text:
        mixed = _resolve_mixed_solvent_smiles(raw_text)
        if mixed:
            old = item.get('smiles', '')
            if old != mixed:
                tag = 'added' if 'smiles' not in item or not old else 'fallback'
                print(f"[PubChem {tag}/mixed] '{raw_text}' -> {mixed}" + (f" (was: '{old}')" if old else ''))
            item['smiles'] = mixed


def fallback_resolve_condition_smiles_in_data(data: Any) -> Any:
    """Walk the result tree; for every entry inside a `conditions` list, attempt
    a PubChem-based SMILES override.

    Mutates dicts in place and also returns `data` for convenience.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key == 'conditions' and isinstance(value, list):
                for it in value:
                    if isinstance(it, dict):
                        _resolve_smiles_for_condition_item(it)
                        fallback_resolve_condition_smiles_in_data(it)
            else:
                fallback_resolve_condition_smiles_in_data(value)
    elif isinstance(data, list):
        for it in data:
            fallback_resolve_condition_smiles_in_data(it)
    return data


# ---------------------------------------------------------------------------
# Reactant / product text -> SMILES (with R-group placeholder substitution)
# ---------------------------------------------------------------------------

# additional_info keys that are NOT placeholders (they're real metadata).
_NON_PLACEHOLDER_KEYS = {
    'yield', 'ee', 'er', 'dr', 'de', 'note', 'notes', 'text', 'role',
    'smiles', 'time', 'temperature', 'solvent', 'reagent', 'entry', 'no',
    'product', 'pressure', 'atmosphere', 'conversion', 'selectivity',
    'rxn', 'reaction', 'reaction_id', 'id', 'label',
}
# OCR-normalisation: 'CHo' -> 'CHO' (very common OCR error).
_OCR_CHO_RE = re.compile(r'CH[oO0]\b')


def _extract_placeholder_subs(additional_info: Any) -> Dict[str, str]:
    """Pull placeholder->value pairs (e.g. {"Ar": "2-MeOC6H4"}) from
    ``additional_info``. Anything that looks like real metadata (yield/ee/note/
    ...) is ignored.
    """
    subs: Dict[str, str] = {}
    if not isinstance(additional_info, list):
        return subs
    for info in additional_info:
        if not isinstance(info, dict):
            continue
        for k, v in info.items():
            if not isinstance(k, str) or not isinstance(v, str):
                continue
            k_clean = k.strip()
            v_clean = v.strip()
            if not k_clean or not v_clean:
                continue
            if k_clean.lower() in _NON_PLACEHOLDER_KEYS:
                continue
            # Placeholders are short tokens like R, R1, R', Ar, X, Y, Z, n, m
            if len(k_clean) > 4:
                continue
            subs.setdefault(k_clean, v_clean)
    return subs


def _apply_placeholder_subs(text: str, subs: Dict[str, str]) -> str:
    """Replace chemical placeholders in ``text``. A placeholder is matched when
    it is NOT preceded by another letter AND NOT followed by a lowercase
    letter — so ``Ar`` matches inside ``ArCHO`` (next char ``C`` is uppercase)
    but not inside ``Aryl`` or ``Argon``. Longest key first so ``Ar1`` is tried
    before ``Ar``.
    """
    if not subs or not isinstance(text, str) or not text:
        return text
    out = text
    for k in sorted(subs.keys(), key=len, reverse=True):
        pattern = r'(?<![A-Za-z])' + re.escape(k) + r'(?![a-z])'
        out = re.sub(pattern, subs[k], out)
    return out


def _normalize_chem_ocr(text: str) -> str:
    """Fix the most common OCR confusions inside chemical-name tokens."""
    if not isinstance(text, str):
        return text
    return _OCR_CHO_RE.sub('CHO', text)


def _resolve_smiles_for_named_item(item: Dict[str, Any],
                                   placeholder_subs: Dict[str, str]) -> None:
    """If a reactant/product/condition entry has only a textual name (no
    SMILES), try to derive a SMILES via the unified resolver, optionally
    substituting R-group placeholders first.

    Mutates ``item`` in place. No-op if a SMILES is already present.
    """
    if not isinstance(item, dict):
        return
    if item.get('smiles'):
        return
    raw = item.get('text')
    candidates: List[str] = []
    if isinstance(raw, str):
        candidates.append(raw)
    elif isinstance(raw, list):
        candidates.extend([t for t in raw if isinstance(t, str) and t.strip()])
    label = item.get('label') if isinstance(item.get('label'), str) else None
    if label:
        candidates.append(label)
    for txt in candidates:
        subbed = _apply_placeholder_subs(txt, placeholder_subs)
        subbed = _normalize_chem_ocr(subbed)
        smi = _resolve_name_to_smiles(subbed)
        if smi:
            item['smiles'] = smi
            if subbed != txt:
                print(f"[txt->SMILES] '{txt}' (subs={placeholder_subs}) -> "
                      f"'{subbed}' -> {smi}")
            else:
                print(f"[txt->SMILES] '{txt}' -> {smi}")
            return


def fallback_resolve_reactant_product_smiles_in_data(data: Any) -> Any:
    """Walk the result tree; for any dict containing ``reactants``/``products``
    lists, fill in missing SMILES using the reactant/product text (after
    optional R-group placeholder substitution drawn from this reaction's
    ``additional_info``).
    """
    if isinstance(data, dict):
        if 'reactants' in data or 'products' in data:
            subs = _extract_placeholder_subs(data.get('additional_info'))
            for key in ('reactants', 'products'):
                lst = data.get(key)
                if isinstance(lst, list):
                    for it in lst:
                        _resolve_smiles_for_named_item(it, subs)
        for v in data.values():
            fallback_resolve_reactant_product_smiles_in_data(v)
    elif isinstance(data, list):
        for it in data:
            fallback_resolve_reactant_product_smiles_in_data(it)
    return data


def _normalize_base_url_for_ipv4(base_url: str) -> str:
    """
    Use IPv4 for localhost to avoid 'Address family not supported by protocol' (errno 97)
    in environments where IPv6 is disabled (e.g. some SLURM/container setups).
    """
    if not base_url:
        return base_url
    # Prefer 127.0.0.1 over localhost or [::1] so httpx uses IPv4
    base_url = base_url.strip()
    if "localhost" in base_url:
        base_url = base_url.replace("localhost", "127.0.0.1")
    if "[::1]" in base_url:
        base_url = base_url.replace("[::1]", "127.0.0.1")
    return base_url


def _normalize_tool_args(raw_args: Optional[dict], image_path: str) -> dict:
    if not isinstance(raw_args, dict):
        return {"image_path": image_path}
    normalized = dict(raw_args)
    placeholder_values = {"[img]", "<img>", "[image]", "<image>", "<<<IMAGE>>>", "IMAGE_PATH", "image.png","image_path"}
    arg_path = normalized.get("image_path")
    if arg_path in placeholder_values or arg_path is None or not os.path.isfile(arg_path):
        normalized["image_path"] = image_path
    return normalized







AGENT_NAME_TO_TOOL = {
    "structure-based r-group substitution agent": "process_reaction_image_with_product_variant_R_group",
    "text-based r-group substitution agent": "process_reaction_image_with_table_R_group",
    "reaction template parsing agent": "get_full_reaction_template",
    "molecular recognition agent": "get_multi_molecular_full",
    "text extraction agent": "text_extraction_agent",
}


def _clean_agent_name(raw_name: str) -> str:
    """Strip leading numbering (e.g. '1.', '2)', '- ') from an agent name."""
    cleaned = re.sub(r'^[\d]+[.):\-\s]+', '', raw_name.strip())
    cleaned = re.sub(r'^[-•*]\s*', '', cleaned)
    return cleaned.strip()


def _parse_planner_output(raw_output: str) -> List[str]:
    """Parse planner text output into a clean list of agent names."""
    cleaned = re.sub(r'[{}]', '', raw_output).strip()
    agents = [_clean_agent_name(a) for a in cleaned.split(',') if a.strip()]
    return [a for a in agents if a]


def _select_main_area(agent_names_lower: List[str]) -> str:
    """Select the main area tool name from a list of agent names (substring match).
    Priority: structure R-group > text R-group > reaction template > molecular recognition."""
    priority = [
        ("structure-based r-group substitution agent", "process_reaction_image_with_product_variant_R_group"),
        ("text-based r-group substitution agent", "process_reaction_image_with_table_R_group"),
        ("reaction template parsing agent", "get_full_reaction_template"),
        ("molecular recognition agent", "get_multi_molecular_full"),
    ]
    for keyword, tool_name in priority:
        if any(keyword in agent for agent in agent_names_lower):
            return tool_name
    return "get_full_reaction_template"


def _has_text_extraction(agent_names_lower: List[str]) -> bool:
    """Check if text extraction agent is in the agent list (substring match)."""
    return any("text extraction agent" in a or "text_extraction_agent" in a
               for a in agent_names_lower)

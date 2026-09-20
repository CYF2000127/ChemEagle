import re
from typing import List, Optional
import sys
import json
import numpy as np
from PIL import Image
import os
import base64
from typing import Optional, Dict, Any



def print(*args, **kwargs):  # noqa: A001 - intentional shadow of builtin
    pass

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    Chem = None
    RDKIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Structural repair of unparsable SMILES (2026-09-11)
# ---------------------------------------------------------------------------
# The vision models write drawn catalysts and reagents with a few recurring
# mistakes that make the whole SMILES unreadable, so the compound is lost even
# though every atom and ring is right:
#   * a counter-ion bonded into the cation: c1cccc(F[B-](F)(F)F)c1 (BF4- drawn
#     next to the ring becomes a substituent);
#   * surplus explicit hydrogens on an atom whose bonds are all written:
#     Ar[NH2+]2=CN3...;
#   * double bonds misplaced around a ring heteroatom: [N+]1=CN2C(...)N=1, the
#     triazolium of NHC precatalysts with two double bonds on one nitrogen;
#   * a two-letter element outside brackets: c1ccc(SeSec2ccccc2)cc1.
# The rules only run on a SMILES RDKit cannot read, and a result is accepted
# only when it sanitizes, so valid SMILES are never touched. Over-valent carbon
# is left alone: there the intended hydrogen count cannot be recovered.
_REPAIR_HALOGENS = frozenset({9, 17, 35, 53})
_REPAIR_ANION_CENTRES = frozenset({5, 13, 15, 33, 51})      # B, Al, P, As, Sb: BF4-, AlCl4-, PF6-, AsF6-, SbF6-
_REPAIR_RING_ATOMS = frozenset({6, 7, 8, 15, 16, 34})
_REPAIR_HETEROATOMS = frozenset({7, 8, 15, 16, 34})
_ORGANIC_FIRST_LETTERS = frozenset("BCNOPSFI")
_AROMATIC_SECOND_LETTERS = frozenset("bcnops")
_TWO_LETTER_ELEMENTS = frozenset((
    "He Li Be Ne Na Mg Al Si Ar Ca Sc Ti Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Kr Rb Sr Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn "
    "Sb Te Xe Cs Ba La Ce Pr Nd Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn").split())
_TWO_LETTER_TOKEN = re.compile(r"[A-Z][a-z]")


def _bracket_two_letter_elements(smiles: str) -> str:
    """Se -> [Se] outside brackets, skipping pairs that also read as two organic atoms (Sc, Co, Sn ...)."""
    def _sub(m):
        t = m.group(0)
        if t not in _TWO_LETTER_ELEMENTS or (t[0] in _ORGANIC_FIRST_LETTERS and t[1] in _AROMATIC_SECOND_LETTERS):
            return t
        return "[" + t + "]"
    parts = re.split(r"(\[[^\]]*\])", smiles)
    return "".join(p if p.startswith("[") else _TWO_LETTER_TOKEN.sub(_sub, p) for p in parts)


def _aromatize_ring_around(rw, idx: int) -> bool:
    """Let RDKit re-place the double bonds of the smallest 5/6 ring holding atom idx."""
    rings = sorted((tuple(r) for r in Chem.GetSymmSSSR(rw) if idx in r and len(r) in (5, 6)), key=len)
    for ring in rings:
        members = set(ring)
        usable = True
        for i in ring:
            atom = rw.GetAtomWithIdx(i)
            if atom.GetAtomicNum() not in _REPAIR_RING_ATOMS or atom.GetDegree() > 3:
                usable = False
                break
            if any(b.GetOtherAtomIdx(i) not in members and b.GetBondType() != Chem.BondType.SINGLE for b in atom.GetBonds()):
                usable = False
                break
        if not usable:
            continue
        for k, i in enumerate(ring):
            rw.GetAtomWithIdx(i).SetIsAromatic(True)
            bond = rw.GetBondBetweenAtoms(i, ring[(k + 1) % len(ring)])
            bond.SetBondType(Chem.BondType.AROMATIC)
            bond.SetIsAromatic(True)
        return True
    return False


_SYNTAX_REPAIRS = (
    (re.compile(r"\[@@\]"), "[C@@]"),          # a stereo marker that lost its atom: "CCOC(=O)[@@]1(...)"
    (re.compile(r"\[@\]"), "[C@]"),
    (re.compile(r"\(R\d*\)"), "(*)"),          # an R label written inline instead of as a site: "C[C@H](R1)C(=O)O"
    (re.compile(r"\(H\)"), ""),                # hydrogen written as a substituent: "CC[Si](H)(H)CC"
    (re.compile(r"(?<=\))H(?![a-z])"), ""),     # a trailing "H" after a branch: "O=C(O)H"
    (re.compile(r"=O=O"), "=O"),                # a carbonyl written twice: "CCC(=O=O)c1ccc(Cl)cc1"
    (re.compile(r"\[O-\]="), "O="),             # a charged oxygen carrying a double bond
)


def _repair_smiles_syntax(smiles: str):
    """A parsable rewrite of a SMILES the model or the converter wrote with a token RDKit cannot read, or None.

    These are spelling accidents rather than chemistry: a stereo marker whose atom went missing, an R label
    left inline, a hydrogen written as a substituent. Each rewrite is applied only when the result parses, so
    a string that is merely unusual is returned untouched."""
    if not RDKIT_AVAILABLE or not isinstance(smiles, str) or not smiles:
        return None
    for pattern, replacement in _SYNTAX_REPAIRS:
        if not pattern.search(smiles):
            continue
        candidate = pattern.sub(replacement, smiles)
        if candidate != smiles and Chem.MolFromSmiles(candidate) is not None:
            return candidate
    # several accidents in one string: apply every rewrite that matches and check once
    candidate = smiles
    for pattern, replacement in _SYNTAX_REPAIRS:
        candidate = pattern.sub(replacement, candidate)
    if candidate != smiles and Chem.MolFromSmiles(candidate) is not None:
        return candidate
    return None


def _repair_unparsable_smiles(smiles: str) -> str:
    """Return a sanitizable repair of an unreadable SMILES, or the input unchanged."""
    if not RDKIT_AVAILABLE or not isinstance(smiles, str) or not smiles:
        return smiles
    syntactic = _repair_smiles_syntax(smiles)
    if syntactic is not None:
        return syntactic
    try:
        bracketed = _bracket_two_letter_elements(smiles)
        if bracketed != smiles and Chem.MolFromSmiles(bracketed) is not None:
            return bracketed
        mol = Chem.MolFromSmiles(bracketed, sanitize=False)
        if mol is None:
            return smiles
        rw = Chem.RWMol(mol)
        for _ in range(8):
            rw.UpdatePropertyCache(strict=False)
            problems = Chem.DetectChemistryProblems(rw)
            if not problems:
                break
            changed = False
            for problem in problems:
                if problem.GetType() != "AtomValenceException":
                    continue
                i = problem.GetAtomIdx()
                atom = rw.GetAtomWithIdx(i)
                if atom.GetAtomicNum() in _REPAIR_HALOGENS and atom.GetDegree() >= 2:
                    centres = [n.GetIdx() for n in atom.GetNeighbors()
                               if n.GetAtomicNum() in _REPAIR_ANION_CENTRES and n.GetFormalCharge() < 0]
                    if centres:
                        for j in [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != centres[0]]:
                            rw.RemoveBond(i, j)
                        changed = True
                        break
                if atom.GetNumExplicitHs() > 0:
                    atom.SetNumExplicitHs(atom.GetNumExplicitHs() - 1)
                    changed = True
                    break
                if atom.GetAtomicNum() in _REPAIR_HETEROATOMS and _aromatize_ring_around(rw, i):
                    changed = True
                    break
            if not changed:
                return smiles
        fixed = rw.GetMol()
        Chem.SanitizeMol(fixed)
        out = Chem.MolToSmiles(fixed)
        return out if Chem.MolFromSmiles(out) is not None else smiles
    except Exception:
        return smiles


def _validate_and_fix_smiles(smiles: str) -> str:
    if not RDKIT_AVAILABLE or not smiles:
        return smiles
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return smiles  # SMILES is valid, no fix needed
    except Exception:
        pass

    # 1) a tetravalent N written without its charge
    n_pattern = r'(?<!\[)N(?!\])'
    for match in re.finditer(n_pattern, smiles):
        pos = match.start()
        test_smiles = smiles[:pos] + '[N+]' + smiles[pos+1:]
        try:
            mol = Chem.MolFromSmiles(test_smiles)
            if mol is not None:
                print(f"[SMILES Fix] Fixed invalid SMILES by adding charge to N:\n  Original: {smiles}\n  Fixed:    {test_smiles}")
                return test_smiles
        except Exception:
            continue

    # 2) counter-ion bonded in, surplus H, misplaced ring double bonds, unbracketed element
    repaired = _repair_unparsable_smiles(smiles)
    if repaired != smiles:
        print(f"[SMILES Fix] Repaired unparsable SMILES:\n  Original: {smiles}\n  Fixed:    {repaired}")
    return repaired


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
import time as _time
import urllib.error as _urlerr
import urllib.parse as _urlparse
import urllib.request as _urlreq


class _ServiceUnavailable(Exception):
    """A lookup service could not be reached, as opposed to answering that it
    does not know the name. The distinction matters: a name the service does
    not know is settled and worth caching, while an unreachable service leaves
    the question open and must not poison the cache."""


# PubChem asks for no more than 5 requests/second and answers 503 when a client
# runs hot, so a retry that waits is also the polite thing to do.
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


# ---------------------------------------------------------------------------
# Network switch. Name lookups are the only step of the pipeline that leaves
# the machine, and a name no service knows costs seconds of waiting, so a run
# that does not need them should be able to say so. Turned off, the alias map,
# the persistent cache and the local OPSIN jar answer on their own.
#
# The setting is read from the environment at import (CHEMEAGLE_NETWORK=0, or
# CHEMEAGLE_OFFLINE=1) and can be changed at runtime with set_network_enabled,
# which is what ChemEagle()'s use_network argument calls.
# ---------------------------------------------------------------------------
_TRUE_WORDS = frozenset({'1', 'true', 'yes', 'on'})
_FALSE_WORDS = frozenset({'0', 'false', 'no', 'off'})


def _env_flag(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    word = raw.strip().lower()
    if word in _TRUE_WORDS:
        return True
    if word in _FALSE_WORDS:
        return False
    return None


def _network_default() -> bool:
    flag = _env_flag('CHEMEAGLE_NETWORK')
    if flag is None:
        offline = _env_flag('CHEMEAGLE_OFFLINE')
        flag = None if offline is None else not offline
    return True if flag is None else flag


NETWORK_ENABLED = _network_default()


def network_enabled() -> bool:
    """Whether name lookups may leave the machine."""
    return NETWORK_ENABLED


def set_network_enabled(enabled: bool) -> bool:
    """Turn name lookups over the network on or off; returns the setting that
    was in force before, so a caller can put it back.

    The environment variable is set as well, because the OCSR resolver lives in
    another package and reads the setting from there rather than importing this
    module back.
    """
    global NETWORK_ENABLED
    previous = NETWORK_ENABLED
    NETWORK_ENABLED = bool(enabled)
    os.environ['CHEMEAGLE_NETWORK'] = '1' if NETWORK_ENABLED else '0'
    os.environ.pop('CHEMEAGLE_OFFLINE', None)
    if previous != NETWORK_ENABLED:
        print('[network] name lookups %s' % ('on' if NETWORK_ENABLED else
                                             'off: alias map, cache and local OPSIN only'))
    return previous


def _get_json_with_retry(url: str, timeout: float = 5.0,
                         attempts: int = 3, base_delay: float = 1.0) -> Any:
    """Fetch JSON, retrying transport failures and transient server codes with
    exponential backoff. Raises _ServiceUnavailable once the attempts run out;
    any other error propagates unchanged, since retrying will not help it."""
    last: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            with _urlreq.urlopen(url, timeout=timeout) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except _urlerr.HTTPError as exc:
            if exc.code not in _RETRY_STATUS:
                raise
            last = exc
        except (_urlerr.URLError, TimeoutError, OSError) as exc:
            last = exc
        if attempt < attempts - 1:
            delay = base_delay * (2 ** attempt)
            print(f"[PubChem] {last}; retrying in {delay:.0f}s "
                  f"({attempt + 1}/{attempts - 1})")
            _time.sleep(delay)
    raise _ServiceUnavailable(str(last))

_PUBCHEM_SMILES_CACHE: Dict[str, Optional[str]] = {}

# Names recorded as a miss that this process has already re-asked about. A recorded miss means every
# service answered and none knew the name, so it is worth one retry per run (a database gains entries)
# but not one per reaction the name appears in: 24 conditions reading '10 mol% PLP' cost 24 chains of
# web requests, each of them seconds long, for the same settled answer.
_RETRIED_MISSES: set = set()

# ---------------------------------------------------------------------------
# Persistent cache (survives process restarts).
# Override path with env var CHEMEAGLE_CACHE_DIR.
# Default: same directory as this helper.py (chemietoolkit/).
# ---------------------------------------------------------------------------
_CACHE_DIR = os.path.expanduser(
    os.environ.get('CHEMEAGLE_CACHE_DIR', os.path.dirname(os.path.abspath(__file__)))
)
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
    """Collect candidate chemical names from a condition-like item.

    Search order priority:
      1. ``label``   — labels (e.g. "B17", "DMAP", "Cs2CO3") are short, exact
         tokens and often match a curated alias or a previously cached entry
         more reliably than the free-form ``text`` (which may carry
         stoichiometry / "or" alternatives / mixture syntax).
      2. ``text`` / ``name``  — full free-form description (and a
         stoichiometry-stripped variant).
    """
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

    # 1) label first (also covers the case where only `label` is present)
    label = item.get('label')
    if isinstance(label, str):
        add(label)
        add(_strip_stoichiometry(label))

    # 2) free-form text / name
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
    if not NETWORK_ENABLED:
        # Nothing is cached here: the name was never asked about, so it stays
        # an open question rather than becoming a recorded miss.
        return None
    encoded = _urlparse.quote(name.strip(), safe='')
    try:
        # 1) name -> CID(s)
        url_cids = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{encoded}/cids/JSON"
        )
        cid_payload = _get_json_with_retry(url_cids, timeout=timeout)
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
        syn_payload = _get_json_with_retry(url_syn, timeout=timeout)
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
        prop_payload = _get_json_with_retry(url_smi, timeout=timeout)
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
    except _ServiceUnavailable as e:
        # Deliberately not cached: the service was unreachable, so the name is
        # still an open question. Caching a miss here would persist to
        # name2smiles.json and make one outage permanent.
        print(f"[PubChem] unreachable for '{name}': {e}")
        raise
    except Exception as e:
        print(f"[PubChem] lookup failed for '{name}': {e}")
        _PUBCHEM_SMILES_CACHE[key] = None
        return None


def _accept_opsin_smiles(text: Optional[str]) -> Optional[str]:
    """OPSIN answers with a bare SMILES, or with nothing when it cannot parse
    the name. Whitespace in the answer means it is not a SMILES."""
    if not text:
        return None
    s = text.strip()
    if not s or any(ch.isspace() for ch in s):
        return None
    if RDKIT_AVAILABLE:
        try:
            if Chem.MolFromSmiles(s) is None:
                return None
        except Exception:
            return None
    return s


def _local_opsin_smiles(name: str) -> Optional[str]:
    """OPSIN run locally through py2opsin, which bundles the OPSIN jar and so
    needs a Java runtime on PATH. Same parser as the web service, so it gives
    the same answer without leaving the machine."""
    try:
        from py2opsin import py2opsin
    except ImportError:
        return None
    try:
        return _accept_opsin_smiles(py2opsin(name))
    except Exception:
        return None


def _query_opsin_smiles(name: str, timeout: float = 5.0) -> Optional[str]:
    """OPSIN (IUPAC name -> SMILES) fallback. Free, deterministic, name-only.

    Best at systematic IUPAC names like ``3-methylpyridine``, ``1,4-dioxane``.
    Queries the OPSIN web service first and falls back to the bundled local
    OPSIN when the service cannot be reached, so a machine without network
    access still resolves systematic names. Returns ``None`` for non-IUPAC
    strings (404 from OPSIN) or any error.
    """
    if not isinstance(name, str):
        return None
    s = name.strip()
    if not s or len(s) > 200:
        return None
    if not NETWORK_ENABLED:
        return _local_opsin_smiles(s)
    enc = _urlparse.quote(s, safe='')
    url = f"https://opsin.ch.cam.ac.uk/opsin/{enc}.smi"
    try:
        with _urlreq.urlopen(url, timeout=timeout) as resp:
            text = resp.read().decode('utf-8', errors='ignore').strip()
    except _urlerr.HTTPError as exc:
        # 404 is OPSIN saying the name is not parseable. The local library is
        # the same parser and would answer the same, so only a server-side
        # failure is worth retrying offline.
        return None if exc.code == 404 else _local_opsin_smiles(s)
    except Exception:
        # No route to the service: offline, DNS failure, timeout, proxy.
        return _local_opsin_smiles(s)
    return _accept_opsin_smiles(text)


_LABEL_TOKEN_RE = re.compile(
    r"^(?:"
    r"[A-Za-z]{1,3}\d{1,3}[a-z'’]?"   
    r"|\d{1,3}[A-Za-z]{1,4}'?"       
    r"|\d{1,3}"                        
    r")$"
)


# One or two bare letters beside an arrow are a compound label ("20 mol % B", "cat. A"), never a name: PubChem
# answers [B] for B and [C] for C, which would put an element where the figure draws a catalyst.
_SHORT_LABEL_RE = re.compile(r"^[A-Za-z]{1,2}'?$")


def _is_label_like(name: str) -> bool:
    if not isinstance(name, str):
        return False
    k = name.strip()
    if not k:
        return False
    return bool(_LABEL_TOKEN_RE.match(k)) or bool(_SHORT_LABEL_RE.match(k))


# A condition line that reads as a sentence rather than a name. No lookup service will know it, and each attempt
# costs three of them; chemical names stay short ("2,4,6-collidine", "tert-Butyl isocyanide", "N-Boc-L-proline").
_PROSE_RE = re.compile(r'\b(of|and|or|then|were|was|instead|added|using|with|under|for)\b', re.IGNORECASE)


def _looks_like_prose(name: str) -> bool:
    if not isinstance(name, str):
        return False
    k = _strip_stoichiometry(name).strip()
    return k.count(' ') >= 4 or (bool(_PROSE_RE.search(k)) and ' ' in k)


def _resolve_name_to_smiles(name: str,
                            status: Optional[Dict[str, Any]] = None,
                            offline: bool = False) -> Optional[str]:
    """Resolve a chemical name to a SMILES through alias map, cache, PubChem,
    OPSIN and finally local OCSR.

    Pass a dict as ``status`` to learn why a lookup came back empty: it gets a
    ``reason`` of ``service-unavailable`` when a web service could not be
    reached, or ``not-found`` when the services answered and none knew the
    name. The caller can then mark the field for later completion rather than
    silently leaving a gap.

    ``offline`` answers from the alias map and the persistent cache only. A
    name lookup over the network costs seconds and misses nine times out of
    ten, so it is spent on the entries that carry no structure at all rather
    than on re-checking the ones that already have one.
    """
    if not isinstance(name, str) or not name.strip():
        return None
    key = name.strip()
    alias = _ALIAS_SMILES.get(key.lower())
    if alias:
        print(f"[alias] '{key}' -> {alias}")
        return alias
    if _is_label_like(key):
        # Label-like token: only honour an explicit hit in the persistent
        # cache (i.e. something the user added by hand). Never touch the
        # network and never write a new entry.
        cached = _PUBCHEM_SMILES_CACHE.get(key)
        if cached:
            print(f"[label cache] hit: '{key}' -> {cached!r}")
            return cached
        print(f"[label] skip resolution for label-like token '{key}'")
        return None
    if key in _PUBCHEM_SMILES_CACHE:
        cached = _PUBCHEM_SMILES_CACHE[key]
        if cached is not None:
            print(f"[name->SMILES cache] hit: '{key}' -> {cached!r}")
            return cached
        if offline or not NETWORK_ENABLED or key in _RETRIED_MISSES:
            return None
        _RETRIED_MISSES.add(key)
        print(f"[name->SMILES cache] retry negative for '{key}'")
        smi = None
    elif offline:
        return None
    else:
        try:
            smi = _query_pubchem_smiles(key)  # writes _PUBCHEM_SMILES_CACHE[key]
        except _ServiceUnavailable:
            # Keep going: OPSIN and the local OCSR step may still answer, and
            # an offline OPSIN needs no network at all.
            if status is not None:
                status['service_unavailable'] = True
            smi = None
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
    if status is not None and smi is None:
        status['reason'] = ('service-unavailable'
                            if status.get('service_unavailable') else 'not-found')
    return smi


def _mark_smiles_unresolved(item: Dict[str, Any], status: Dict[str, Any]) -> None:
    """Record on the item that its SMILES could not be filled in, and why.

    Only entries that end up without a SMILES are marked, so an item the agent
    already supplied a structure for stays clean. ``service-unavailable`` says
    the answer is still open and worth retrying later; ``not-found`` says the
    services answered and none recognised the name.
    """
    if item.get('smiles'):
        return
    reason = status.get('reason')
    if reason:
        item['smiles_unresolved'] = reason


def _resolve_smiles_for_condition_item(item: Dict[str, Any]) -> None:
    """Look up a SMILES on PubChem for a condition item and write it back.

    Runs for every chemical role (solvent, reagent, catalyst, or none given):

      - no ``smiles`` yet  -> add one if a lookup returns a hit
      - existing ``smiles`` -> overwritten by the hit for a solvent or reagent,
        where the name is the authority; a catalyst keeps what it has, because
        its structure comes from the drawing beside the label, not from a name.
    """
    if not isinstance(item, dict):
        return
    role = item.get('role', '')
    if not isinstance(role, str) or role.strip().lower() not in _CHEMICAL_CONDITION_ROLES:
        return
    has_structure = _readable_smiles_key(item.get('smiles')) is not None
    if has_structure and role.strip().lower() not in {'solvent', 'reagent'}:
        return
    status: Dict[str, Any] = {}
    for name in _candidate_chem_names(item):
        if _is_label_like(_strip_stoichiometry(name)):
            continue      # "20 mol % B" names the catalyst the figure draws as B, not the element
        if _looks_like_prose(name):
            continue      # "1 equiv. of aldehyde and 1.5 equiv. of 3-oxobutanoate were used" is a sentence
        # an entry that already carries a structure is only re-checked against the alias map and the cache:
        # the network is for the entries that have none
        smi = _resolve_name_to_smiles(name, status=status, offline=has_structure)
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
            return

    _mark_smiles_unresolved(item, status)


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
    status: Dict[str, Any] = {}
    for txt in candidates:
        subbed = _apply_placeholder_subs(txt, placeholder_subs)
        subbed = _normalize_chem_ocr(subbed)
        smi = _resolve_name_to_smiles(subbed, status=status)
        if smi:
            item['smiles'] = smi
            if subbed != txt:
                print(f"[txt->SMILES] '{txt}' (subs={placeholder_subs}) -> "
                      f"'{subbed}' -> {smi}")
            else:
                print(f"[txt->SMILES] '{txt}' -> {smi}")
            return

    _mark_smiles_unresolved(item, status)


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


# ---------------------------------------------------------------------------
# Condition structures shared across the reactions of one figure (2026-09-11)
# ---------------------------------------------------------------------------
# A figure draws its catalyst once, but the agents often attach the structure
# to only some of its reactions: the template row carries "10 mol% B27" with
# a SMILES while the substrate rows carry the bare name, a "G1 or G7" entry
# stays empty although G1 and G7 are resolved in the rows below, or substrate
# rows lose the catalyst entry altogether. Two conservative passes, each
# confined to the reaction list of one figure:
#   1. a chemical condition entry without a readable SMILES takes the structure
#      of the same name (label, else text without amounts) from another
#      reaction; "G1 or G7" becomes one entry per compound. A name that maps
#      to two different structures in the figure is left alone.
#   2. a catalyst drawn only on the reaction that carries the figure's full
#      catalyst set (typically the template row) is copied to the other
#      reactions lacking it, when their other conditions are identical (same
#      reagent/solvent structures, same temperature/time text). A catalyst that
#      some other reaction already carries is never copied: a template listing
#      "A1 or B6" over rows that each use one of them means alternatives, not
#      a missing co-catalyst. A reaction that still names a catalyst without a
#      structure is left alone.
# Entries added or completed this way carry "smiles_source".
import copy as _copy

_CHEMICAL_CONDITION_ROLES = frozenset({'', 'reagent', 'reagents', 'solvent', 'solvents', 'catalyst', 'catalysts'})
_OUTCOME_CONDITION_ROLES = frozenset({'yield', 'yields', 'ee', 'er', 'dr', 'de', 'selectivity', 'conversion', 'note', 'notes'})
_CONDITION_AMOUNT_RE = re.compile(
    r'\(?\s*\d+(?:\.\d+)?\s*(?:mol\s*%|equiv\.?|equiv|eq\.?|mmol|mol|mg|g|mL|ml|uL|M|%)(?![A-Za-z])\s*\)?')
_CONDITION_NAME_SPLIT_RE = re.compile(r'\s*(?:\bor\b|\band\b|,|;|/|&)\s*', re.I)
_CATALYST_TEXT_RE = re.compile(r'mol\s*%|\bcat(?:alyst|alytic)?\b', re.I)


def _condition_role(item):
    return str(item.get('role') or '').strip().lower()


def _readable_smiles_key(smiles):
    if not RDKIT_AVAILABLE or not isinstance(smiles, str):
        return None
    s = smiles.strip()
    if not s or s.lower() in ('none', 'null', 'n/a'):
        return None
    try:
        mol = Chem.MolFromSmiles(s)
    except Exception:
        return None
    return Chem.MolToSmiles(mol) if mol is not None else None


def _condition_names(item):
    for field in ('label', 'text'):
        raw = item.get(field)
        if not isinstance(raw, str) or not raw.strip() or raw.strip().lower() == 'none':
            continue
        names = [n.strip(' .:') for n in _CONDITION_NAME_SPLIT_RE.split(_CONDITION_AMOUNT_RE.sub(' ', raw))]
        names = [n for n in names if n]
        if names:
            return names
    return []


def _is_catalyst_entry(item):
    role = _condition_role(item)
    if role in ('catalyst', 'catalysts'):
        return True
    if role not in _CHEMICAL_CONDITION_ROLES:
        return False
    return bool(_CATALYST_TEXT_RE.search(' '.join(str(item.get(k) or '') for k in ('text', 'label'))))


def _fill_condition_structures_by_name(rows):
    by_name = {}
    for row in rows:
        for item in row['conditions']:
            if not isinstance(item, dict) or _condition_role(item) not in _CHEMICAL_CONDITION_ROLES:
                continue
            key = _readable_smiles_key(item.get('smiles'))
            names = _condition_names(item)
            if key and len(names) == 1:
                by_name.setdefault(names[0].lower(), {}).setdefault(key, item['smiles'])
    filled = 0
    for row in rows:
        out = []
        for item in row['conditions']:
            if (isinstance(item, dict) and _condition_role(item) in _CHEMICAL_CONDITION_ROLES
                    and not _readable_smiles_key(item.get('smiles'))):
                names = _condition_names(item)
                found = [by_name.get(n.lower(), {}) for n in names]
                if names and all(len(f) == 1 for f in found):
                    for name, f in zip(names, found):
                        new_item = dict(item)
                        new_item.pop('smiles_unresolved', None)
                        new_item['smiles'] = next(iter(f.values()))
                        if len(names) > 1:
                            new_item['label'] = name
                        new_item['smiles_source'] = 'same compound in another reaction of this figure'
                        out.append(new_item)
                    filled += 1
                    continue
            out.append(item)
        row['conditions'] = out
    return filled


def _condition_block_signature(row):
    structures, texts = set(), set()
    for item in row['conditions']:
        if not isinstance(item, dict) or _is_catalyst_entry(item):
            continue
        role = _condition_role(item)
        if role in _CHEMICAL_CONDITION_ROLES:
            key = _readable_smiles_key(item.get('smiles'))
            if key:
                structures.add(key)
        elif role not in _OUTCOME_CONDITION_ROLES:
            texts.add((role, re.sub(r'\s+', '', str(item.get('text') or '')).lower()))
    return frozenset(structures), frozenset(texts)


def _propagate_catalyst_entries(rows):
    info = []
    for row in rows:
        own, every, unresolved = {}, set(), False
        for c in row['conditions']:
            if not isinstance(c, dict) or not _is_catalyst_entry(c):
                continue
            key = _readable_smiles_key(c.get('smiles'))
            if not key:
                unresolved = True
                continue
            every.add(key)
            if not c.get('smiles_source'):
                own.setdefault(key, c)
        info.append((own, every, unresolved, _condition_block_signature(row)))
    full = set().union(*(set(own) for own, _, _, _ in info))
    is_source = [bool(full) and set(own) == full for own, _, _, _ in info]
    if not any(is_source):
        return 0
    elsewhere = set().union(*(set(own) for (own, _, _, _), src in zip(info, is_source) if not src))
    exclusive = full - elsewhere
    if not exclusive:
        return 0
    sources = [(own, block) for (own, _, _, block), src in zip(info, is_source) if src]
    added = 0
    for row, (own, every, unresolved, block), src in zip(rows, info, is_source):
        missing = exclusive - every
        if src or not missing or unresolved or not block[0]:
            continue
        match = next((cand for cand, cand_block in sources if cand_block == block), None)
        if match is None:
            continue
        copies = []
        for key in sorted(missing):
            new_item = _copy.deepcopy(match[key])
            new_item['smiles_source'] = 'catalyst of another reaction of this figure'
            copies.append(new_item)
        row['conditions'] = copies + row['conditions']
        added += 1
    return added


# The leading token of a printed identifier. Its own pattern: it is deliberately loose, because it only has to
# cut a line at its first token, while _LABEL_TOKEN_RE above decides whether a name is a compound label and must
# stay strict (it used to be shadowed by this one, which stopped every name lookup from ever leaving the alias map).
_IDT_LEADING_TOKEN_RE = re.compile(r"^([A-Za-z]{0,4}\s?[-–]?\s?\d{1,3}[a-z]{0,3}'?|[A-Za-z]{1,4}\d{0,2}'?)")


def _label_tokens(texts):
    """The labels an [Idt] box carries: the whole line plus its leading token
    ("5 (NHC 5'. HBF4)" gives "5 (NHC 5'. HBF4)" and "5")."""
    items = texts if isinstance(texts, (list, tuple)) else [texts]
    out = []
    for raw in items:
        text = str(raw or '').strip()
        if not text:
            continue
        out.append(text)
        m = _IDT_LEADING_TOKEN_RE.match(text)
        if m and m.group(1).strip():
            out.append(m.group(1).strip())
    return out


def drawn_labels_from_mol_result(mol_result):
    """(label -> drawn SMILES) for one figure, taken from the molecule agent's [Mol]/[Idt] corefs.

    Generic templates (a SMILES with a wildcard) are skipped: they stand for a whole substituent
    table, not for the compound the label names.
    """
    found = {}
    for item in mol_result or []:
        if not isinstance(item, dict):
            continue
        boxes = item.get('bboxes') or []
        for pair in item.get('corefs') or []:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            mi, ii = pair[0], pair[1]
            if not isinstance(mi, int) or not isinstance(ii, int) or mi >= len(boxes) or ii >= len(boxes):
                continue
            smiles, texts = boxes[mi].get('smiles'), boxes[ii].get('text')
            if not smiles or not texts or '*' in str(smiles):
                continue
            for label in _label_tokens(texts):
                found.setdefault(label.lower(), smiles)
    return found


_MIN_DRAWN_LABEL_ATOMS = 5
_DEGENERATE_LOOKUP_ATOMS = 2
_MAX_DRAWN_LABEL_LEN = 12


def _drawn_label_candidates(item, label_map):
    """Labels of the drawn-structure map that this condition entry names."""
    names = []
    raw_label = item.get('label')
    if isinstance(raw_label, str) and raw_label.strip():
        names.append(raw_label.strip())
    names.extend(_condition_names(item))
    for name in names:
        hit = label_map.get(name.lower())
        if hit:
            return hit
    text = ' '.join(str(item.get(k) or '') for k in ('label', 'text'))
    found = [smi for lab, smi in label_map.items()
             if re.search(r"(?<![A-Za-z0-9])" + re.escape(lab) + r"(?![A-Za-z0-9'])", text, re.I)]
    unique = {_readable_smiles_key(smi) or smi for smi in found}
    return found[0] if len(unique) == 1 else None


def attach_drawn_labelled_structures(data, label_map):
    """Conditions that name a compound drawn in the figure take the drawn structure (in place).

    The molecule agent links every drawn structure to its printed label (get_molecular_agent.
    register_label_structures). A condition entry naming one of those labels gets that structure even
    when it already carries one from a name lookup: the drawing is direct evidence of what the figure
    means by "B (10 mol%)", while the lookup is a guess (it returned elemental boron).
    """
    usable = {lab: smi for lab, smi in (label_map or {}).items()
              if len(lab) <= _MAX_DRAWN_LABEL_LEN and _readable_smiles_key(smi)
              and Chem.MolFromSmiles(smi).GetNumHeavyAtoms() >= _MIN_DRAWN_LABEL_ATOMS}
    if not usable:
        return data
    _attach_drawn_labels(data, usable)
    return data


def _attach_drawn_labels(node, label_map):
    if isinstance(node, dict):
        conditions = node.get('conditions')
        if isinstance(conditions, list):
            present = {_readable_smiles_key(it.get('smiles')) for it in conditions
                       if isinstance(it, dict) and it.get('smiles')}
            present.discard(None)
            for item in conditions:
                if not isinstance(item, dict) or _condition_role(item) not in _CHEMICAL_CONDITION_ROLES:
                    continue
                drawn = _drawn_label_candidates(item, label_map)
                if not drawn:
                    continue
                key = _readable_smiles_key(drawn)
                if key is None or key in present:
                    continue
                current = _readable_smiles_key(item.get('smiles'))
                if current == key:
                    continue
                if current is not None and Chem.MolFromSmiles(current).GetNumHeavyAtoms() > _DEGENERATE_LOOKUP_ATOMS:
                    continue          # the entry already carries a real structure; only degenerate
                                      # name lookups ("B" -> elemental boron) are replaced
                item['smiles'] = drawn
                item['smiles_source'] = 'structure drawn next to this label in the figure'
                item.pop('smiles_unresolved', None)
                present.add(key)
        for value in node.values():
            _attach_drawn_labels(value, label_map)
    elif isinstance(node, list):
        for value in node:
            _attach_drawn_labels(value, label_map)


R_GROUP_AGENTS = ("process_reaction_image_with_product_variant_R_group", "process_reaction_image_with_table_R_group")


def rgroup_fallback_agent(ordered_agents, molecule_smiles, drawn_threshold=5):
    """The R-group agent to run although the planner did not ask for one, or None.

    A figure whose drawing carries an R site needs its substituents resolved from somewhere: the drawn product
    variants (the product-variant agent) or a substituent table (the table agent). When the planner routes such
    a figure to the plain template agents instead, every row of the figure comes back as a copy of the template
    and the whole figure is lost. The signal is the molecular agent's own output: a molecule with a wildcard
    means a template is drawn, and the number of fully resolved molecules says whether the variants are drawn
    (a scope scheme) or listed as text (a table).
    """
    if any(name in ordered_agents for name in R_GROUP_AGENTS):
        return None, ""
    smiles = [s for s in (molecule_smiles or []) if isinstance(s, str) and s]
    if not any("*" in s for s in smiles):
        return None, ""
    resolved = sum(1 for s in smiles if "*" not in s)
    if resolved >= drawn_threshold:
        return R_GROUP_AGENTS[0], ("the figure draws a template and %d resolved molecules, so the variants are drawn" % resolved)
    return R_GROUP_AGENTS[1], ("the figure draws a template and only %d resolved molecules, so the variants are listed as text" % resolved)


def _canonical_or_none(smiles):
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
    except Exception:
        return None
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles else None
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _propagate_drawn_condition(rows, drawn):
    """The one structure a figure draws beside its arrow belongs to all of its reactions.

    A scheme that draws a single molecule in the condition region (an oxidant, an NHC precatalyst) means it for
    every row, but the agents write it into the row they read it on and leave the others without it. It is added
    to the rows that carry it neither as a condition nor as a reactant or product. A figure that draws several
    condition molecules is a screening set, where each row picks its own, so nothing is shared there.
    """
    canon = [c for c in (_canonical_or_none(s) for s in drawn or []) if c]
    if len(set(canon)) != 1:
        return 0
    target = canon[0]
    source = next((c for row in rows for c in row['conditions']
                   if isinstance(c, dict) and _canonical_or_none(c.get('smiles')) == target), None)
    added = 0
    for row in rows:
        present = {_canonical_or_none(c.get('smiles')) for c in row['conditions'] if isinstance(c, dict)}
        for key in ('reactants', 'products'):
            present |= {_canonical_or_none(m.get('smiles')) for m in (row.get(key) or []) if isinstance(m, dict)}
        if target in present:
            continue
        item = _copy.deepcopy(source) if source else {'role': 'reagent', 'text': 'drawn beside the arrow', 'smiles': target}
        item['smiles_source'] = 'drawn beside the arrow in this figure'
        row['conditions'] = [item] + row['conditions']
        added += 1
    if added:
        print(f"[condition propagation] the drawn condition structure was added to {added} reaction(s)")
    return added


def propagate_condition_structures_in_data(data, drawn=None):
    """Share drawn catalyst / labelled reagent structures across the reactions of one figure (in place)."""
    if isinstance(data, dict):
        for value in data.values():
            propagate_condition_structures_in_data(value, drawn)
    elif isinstance(data, list):
        rows = [it for it in data if isinstance(it, dict) and isinstance(it.get('conditions'), list)]
        if len(rows) >= 2:
            try:
                _fill_condition_structures_by_name(rows)
                _propagate_catalyst_entries(rows)
                if drawn:
                    _propagate_drawn_condition(rows, drawn)
            except Exception as exc:
                print(f"[condition propagation] skipped: {exc}")
        for it in data:
            propagate_condition_structures_in_data(it, drawn)
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
    "condition interpretation agent": "get_reaction_con",
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


def _resolve_ordered_agents(agent_list: List[str]):
    ordered = []
    for agent in agent_list:
        name_lower = str(agent).lower().strip()
        tool = None
        for key, mapped in AGENT_NAME_TO_TOOL.items():
            if key in name_lower:
                tool = mapped
                break
        if tool is None:
            # tolerate tool-style names such as 'get_full_reaction_template_azure'
            normalized = name_lower.replace(' ', '_')
            for mapped in AGENT_NAME_TO_TOOL.values():
                if mapped in normalized:
                    tool = mapped
                    break
        if tool and tool not in ordered:
            ordered.append(tool)

    has_text_extraction = "text_extraction_agent" in ordered
    ordered = [t for t in ordered if t != "text_extraction_agent"]

    rgroup_agents = {
        "process_reaction_image_with_product_variant_R_group",
        "process_reaction_image_with_table_R_group",
    }
    if rgroup_agents & set(ordered):
        ordered = [t for t in ordered
                   if t not in ("get_multi_molecular_full","get_full_reaction_template", "get_reaction_con")]

    if not ordered:
        ordered = ["get_full_reaction_template"]
    return ordered, has_text_extraction



def _is_wildcard_symbol(sym):
    if not isinstance(sym, str):
        return False
    s = sym.strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    if not s:
        return False
    if s == '*':
        return True
    if s[0] in ('R', 'r') and (len(s) == 1 or s[1:].isdigit()):
        return True
    if s.startswith('Ar') and (len(s) == 2 or s[2:].isdigit()):
        return True
    return False


# MolNexTR drops the central carbon of a drawn ketene R1R2C=C=O and returns R1R2C=O, so the
# patch below re-inserts it. Until 2026-09-11 it fired on every carbonyl whose other neighbours
# are all R groups, which turned the generic ketone / aldehyde templates of 104-1, 104-2, 147, 262
# and two openchemie figures into ketenes. The dropped carbon leaves a trace in the geometry: the
# predicted C and O then sit two bond lengths apart (1.45 x the molecule's median bond length in
# 277, 281, 282, 291) while a real carbonyl measures 1.0-1.1 x (104-1, 104-2, 262). A site is now
# accepted only when the C-O distance is at least KETENE_STRETCH times the median bond length;
# without coordinates nothing is patched, and the old SMILES-only rewrite '*C(*)=O' -> '*C(*)=C=O'
# is gone for the same reason.
KETENE_STRETCH = 1.25


def _median_bond_length(coords, edges):
    lengths = []
    n = min(len(coords), len(edges))
    for i in range(n):
        for j in range(i + 1, n):
            if j < len(edges[i]) and edges[i][j]:
                dx = float(coords[i][0]) - float(coords[j][0])
                dy = float(coords[i][1]) - float(coords[j][1])
                lengths.append((dx * dx + dy * dy) ** 0.5)
    if not lengths:
        return 0.0
    lengths.sort()
    m = len(lengths)
    return lengths[m // 2] if m % 2 else 0.5 * (lengths[m // 2 - 1] + lengths[m // 2])


def _find_sites_in_graph(symbols, edges, coords=None, stretch=None):
    """(carbon, oxygen) pairs of a carbonyl whose other neighbours are all R groups and whose
    drawn C-O distance is stretched to two bond lengths: a ketene that lost its central carbon."""
    if coords is None:
        return []
    stretch = KETENE_STRETCH if stretch is None else stretch
    median = _median_bond_length(coords, edges)
    if median <= 0:
        return []
    sites = []
    n = len(symbols)
    for i, sym in enumerate(symbols):
        s = sym[1:-1] if isinstance(sym, str) and sym.startswith('[') and sym.endswith(']') else sym
        if s != 'C':
            continue
        o_idx = None
        non_o_neighbours = []
        for j in range(n):
            if j == i:
                continue
            order = edges[i][j] if i < len(edges) and j < len(edges[i]) else 0
            if order == 0:
                continue
            jsym = symbols[j]
            jbare = jsym[1:-1] if isinstance(jsym, str) and jsym.startswith('[') and jsym.endswith(']') else jsym
            if order == 2 and jbare == 'O':
                o_other = 0
                for k in range(n):
                    if k == j or k == i:
                        continue
                    if (j < len(edges) and k < len(edges[j]) and edges[j][k]) or \
                       (k < len(edges) and j < len(edges[k]) and edges[k][j]):
                        o_other += 1
                if o_other == 0:
                    o_idx = j
                    continue
            non_o_neighbours.append(j)
        if o_idx is None or not non_o_neighbours:
            continue
        if not all(_is_wildcard_symbol(symbols[k]) for k in non_o_neighbours):
            continue
        if i >= len(coords) or o_idx >= len(coords):
            continue
        dx = float(coords[i][0]) - float(coords[o_idx][0])
        dy = float(coords[i][1]) - float(coords[o_idx][1])
        if (dx * dx + dy * dy) ** 0.5 >= stretch * median:
            sites.append((i, o_idx))
    return sites


def _insert_central_C_in_graph(symbols, coords, edges, c_idx, o_idx):
    n = len(symbols)
    symbols.append('C')
    cx, cy = coords[c_idx][0], coords[c_idx][1]
    ox, oy = coords[o_idx][0], coords[o_idx][1]
    coords.append([(cx + ox) / 2.0, (cy + oy) / 2.0])
    for row in edges:
        row.append(0)
    edges.append([0] * (n + 1))
    edges[c_idx][o_idx] = 0
    edges[o_idx][c_idx] = 0
    edges[c_idx][n] = 2
    edges[n][c_idx] = 2
    edges[n][o_idx] = 2
    edges[o_idx][n] = 2
    return n


def _rebuild_atoms_bonds(item):
    symbols = item.get('symbols')
    coords = item.get('coords')
    edges = item.get('edges')
    if symbols is None or coords is None or edges is None:
        return
    if 'atoms' in item:
        new_atoms = []
        for sym, (x, y) in zip(symbols, coords):
            new_atoms.append({
                'atom_symbol': sym,
                'x': round(float(x), 3),
                'y': round(float(y), 3),
            })
        item['atoms'] = new_atoms
    if 'bonds' in item:
        _ORDER2NAME = {1: 'single', 2: 'double', 3: 'triple', 4: 'aromatic',
                       5: 'solid wedge', 6: 'dashed wedge'}
        new_bonds = []
        n = len(symbols)
        for i in range(n):
            for j in range(i + 1, n):
                order = edges[i][j] if i < len(edges) and j < len(edges[i]) else 0
                if order:
                    new_bonds.append({
                        'bond_type': _ORDER2NAME.get(order, 'single'),
                        'endpoint_atoms': (i, j),
                    })
        item['bonds'] = new_bonds


def _patch_item_inplace(item, conversion_function, tag=''):
    symbols = item.get('symbols')
    coords = item.get('coords')
    edges = item.get('edges')
    if not (isinstance(symbols, list) and isinstance(coords, list) and isinstance(edges, list)):
        return False
    sites = _find_sites_in_graph(symbols, edges, coords)
    if not sites:
        return False
    old_smiles = item.get('smiles')
    for c_idx, o_idx in sorted(sites, key=lambda p: -p[0]):
        _insert_central_C_in_graph(symbols, coords, edges, c_idx, o_idx)
    try:
        new_smiles, new_molfile, _extra = conversion_function(coords, symbols, edges)
        item['smiles'] = new_smiles
        item['molfile'] = new_molfile
    except Exception as e:
        print(f"[ketene-patch{tag}] WARNING: graph->smiles failed: {e}")
        return False
    _rebuild_atoms_bonds(item)
    print(f"[ketene-patch{tag}] {old_smiles!r} -> {item.get('smiles')!r}  "
          f"(graph: +1 atom, sites={sites})")
    return True



def _patch_to_reaction(updated_data):
    if not updated_data:
        return updated_data
    from molnextr.chemistry import _convert_graph_to_smiles
    for rxn in updated_data:
        for key in ('reactants', 'conditions', 'products'):
            for item in rxn.get(key, []) or []:
                if 'symbols' in item and 'coords' in item and 'edges' in item:
                    _patch_item_inplace(item, _convert_graph_to_smiles, tag=f'[rxn:{key}]')
                # items without a graph are left alone (2026-09-11): a bare '*C(*)=O' cannot be told from a ketone template
    return updated_data    



def _patch_to_mol(updated_data):
    if not updated_data:
        return updated_data
    from molnextr.chemistry import _convert_graph_to_smiles
    for item in updated_data:
        for bbox in item.get('bboxes', []) or []:
            if 'symbols' in bbox and 'coords' in bbox and 'edges' in bbox:
                _patch_item_inplace(bbox, _convert_graph_to_smiles, tag='[mol]')
            # bboxes without a graph are left alone (2026-09-11), see _patch_to_reaction
    return updated_data

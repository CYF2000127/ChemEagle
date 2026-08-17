"""ChemEAGLE tool suite exposed as an MCP server.

"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import re
import sys
import time
import threading
from typing import Any, Dict, List, Optional

# The tools live in the repo root; import from there regardless of cwd.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("chemeagle-tools")

# --------------------------------------------------------------------------- #
# Config / sandbox                                                             #
# --------------------------------------------------------------------------- #
_CFG: Dict[str, Any] = {"image_root": None, "trace": None, "device": "cpu"}
_TRACE_LOCK = threading.Lock()
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def _trace(tool: str, args: Dict[str, Any], t0: float,
           ok: bool, err: str = "", n_out: Optional[int] = None) -> None:
    path = _CFG.get("trace")
    if not path:
        return
    rec = {
        "ts": time.time(),
        "tool": tool,
        "args": {k: (v if isinstance(v, (int, float, bool)) else str(v)[:200])
                 for k, v in args.items()},
        "latency_s": round(time.time() - t0, 3),
        "ok": ok,
        "error": err[:500],
        "n_out": n_out,
    }
    with _TRACE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _quiet(fn):
    """Redirect a tool's stdout to stderr.

    CRITICAL: MCP stdio uses stdout as the JSON-RPC channel, but several
    underlying tools print to stdout (e.g. MolNexTR emits "symbols: [...]" on
    every call, and model loaders print progress).  Un-redirected, that output
    is interleaved into the protocol stream and corrupts it.
    """
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        with contextlib.redirect_stdout(sys.stderr):
            return fn(*a, **kw)
    return wrapper


def _split_sentences(text: str) -> List[str]:
    """Sentence splitter matching get_text_agent.split_text_into_sentences.

    Reimplemented here rather than imported: importing get_text_agent runs
    configure_tesseract() at module scope, which prints to stdout (corrupting
    the MCP stream) and raises UnicodeEncodeError under a GBK console.
    """
    parts = re.split(r'([.!?]+)', text)
    out: List[str] = []
    for i in range(0, len(parts) - 1, 2):
        s = (parts[i] + parts[i + 1]).strip() if i + 1 < len(parts) else parts[i].strip()
        if s:
            out.append(s)
    if not out:
        out = [l.strip() for l in text.splitlines() if l.strip()]
    if not out:
        out = [text] if len(text) <= 500 else [
            " ".join(text.split()[i:i + 60]) for i in range(0, len(text.split()), 60)]
    return out


def _patch_chemrxnextractor() -> None:
    """Compat shim for ChemRxnExtractor under transformers>=4.

    BertCRFForRoleLabeling.forward reads the CLS representation as `outputs[1]`
    (BertModel's pooler_output).  Modern transformers builds
    BertForTokenClassification with add_pooling_layer=False, so pooler_output is
    None, to_tuple() has a single element, and `outputs[1]` raises
    IndexError('tuple index out of range') for EVERY sentence in which the
    product-tagging stage finds a product -- i.e. the role stage fails on every
    real reaction while quietly succeeding on reaction-free text.

    The role checkpoint does contain trained `bert.pooler.*` weights (they are
    reported as "not used" at load time), so restoring the pooling layer both
    fixes the crash and lets those trained weights load, which is faithful to
    how the model was trained.
    """
    from transformers import BertModel
    from chemrxnextractor.models import model as crx

    if getattr(crx.BertCRFForRoleLabeling, "_pooler_patched", False):
        return
    orig = crx.BertCRFForRoleLabeling.__init__

    def patched(self, config, *a, **kw):
        orig(self, config, *a, **kw)
        if getattr(self, "use_cls", False) and getattr(self.bert, "pooler", None) is None:
            self.bert = BertModel(config, add_pooling_layer=True)

    crx.BertCRFForRoleLabeling.__init__ = patched
    crx.BertCRFForRoleLabeling._pooler_patched = True


def _resolve(image_path: str) -> str:
    """Resolve + sandbox an image path. Raises on escape attempts.

    Two guards, because when one warm server is shared across a whole run the
    image root is the benchmark directory -- which also contains the ground
    truth (GT1.json, GT3.csv, ...):
      1. path must stay under --image-root
      2. path must be an image file, so no tool can be pointed at a GT file
    """
    root = _CFG.get("image_root")
    p = os.path.abspath(image_path)
    if not os.path.exists(p) and root:
        cand = os.path.abspath(os.path.join(root, os.path.basename(image_path)))
        if os.path.exists(cand):
            p = cand
    if root:
        r = os.path.abspath(root)
        try:
            same = os.path.commonpath([r, p]) == r
        except ValueError:      # different drives
            same = False
        if not same:
            raise ValueError(
                f"access denied: {image_path!r} is outside the permitted image root")
    if not p.lower().endswith(IMAGE_EXTS):
        raise ValueError(
            f"access denied: {os.path.basename(p)!r} is not an image file")
    if not os.path.exists(p):
        raise FileNotFoundError(f"no such image: {image_path}")
    return p


# --------------------------------------------------------------------------- #
# Lazy model loading (weights are heavy: pix2seq ckpt alone is ~432 MB)        #
# --------------------------------------------------------------------------- #
_M: Dict[str, Any] = {}
_LOAD_LOCK = threading.Lock()


def _toolkit():
    with _LOAD_LOCK:
        if "toolkit" not in _M:
            import torch
            from chemietoolkit import ChemIEToolkit
            _M["toolkit"] = ChemIEToolkit(device=torch.device(_CFG["device"]))
        return _M["toolkit"]


def _rxn():
    with _LOAD_LOCK:
        if "rxn" not in _M:
            import torch
            from rxnim import RxnIM
            ckpt = os.path.join(_REPO, "rxn.ckpt")
            _M["rxn"] = RxnIM(ckpt, device=torch.device(_CFG["device"]))
        return _M["rxn"]


def _pil(path: str):
    from PIL import Image
    return Image.open(path).convert("RGB")


def _jsonable(x: Any) -> Any:
    """Make numpy / torch / PIL-laden tool output JSON-serializable."""
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items() if k not in ("image", "figure")}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "item") and hasattr(x, "dtype"):   # numpy scalar
        try:
            return x.item()
        except Exception:
            return str(x)
    if hasattr(x, "tolist"):                          # numpy array / tensor
        try:
            return x.tolist()
        except Exception:
            return str(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


# --------------------------------------------------------------------------- #
# Vision tools                                                                 #
# --------------------------------------------------------------------------- #
@mcp.tool()
@_quiet
def detect_molecules(image_path: str) -> str:
    """Detect every molecule sub-image in a chemical graphic (tool: MolDetector).

    Returns a JSON list of bounding boxes, one per detected molecule:
      [{"category": str, "bbox": [x1, y1, x2, y2], "score": float}, ...]
    Coordinates are pixels in the input image's own frame.

    Detects only that a molecule IS present and where; it does not read the
    structure. Returns [] if no molecule is found.
    """
    t0 = time.time()
    try:
        p = _resolve(image_path)
        out = _toolkit().extract_molecule_bboxes_from_figures([_pil(p)])[0]
        out = _jsonable(out)
        _trace("detect_molecules", {"image_path": image_path}, t0, True, n_out=len(out))
        return json.dumps(out, ensure_ascii=False)
    except Exception as e:
        _trace("detect_molecules", {"image_path": image_path}, t0, False, repr(e))
        return json.dumps({"error": repr(e)})


@mcp.tool()
@_quiet
def image_to_graph(image_path: str) -> str:
    """Convert a single molecule image into an explicit molecular graph
    (tool: Image2Graph, built on MolNexTR).

    Returns JSON:
      {"smiles":  str,           # SMILES decoded from the graph
       "symbols": [str, ...],    # per-atom label, e.g. "C", "N", "R1", "Ar", "iPr"
       "coords":  [[x, y], ...], # per-atom 2D coordinates
       "edges":   [[int, ...]],  # N x N adjacency; 0 none, 1 single, 2 double,
                                 # 3 triple, 4 aromatic, 5 wedge, 6 dash
       "atoms":   [{"atom_symbol": str, "x": float, "y": float}, ...],
       "bonds":   [{"bond_type": str, "endpoint_atoms": [i, j]}, ...]}

    `symbols` may contain non-element labels for R-group / abbreviation nodes
    (e.g. "R1", "Ar", "iPr") and may contain OCR errors on those labels
    (e.g. "iPr" read as "iP1").

    You can EDIT `symbols` -- to fix an OCR error, or to replace an R-group
    placeholder with the substituent the graphic defines for it -- and then call
    `graph_to_smiles(symbols, coords, edges)` to regenerate the SMILES from the
    corrected graph. Editing the graph and re-deriving is more reliable than
    editing a SMILES string by hand.

    Input should be ONE molecule; passing a whole multi-molecule graphic gives
    unreliable output.
    """
    t0 = time.time()
    try:
        p = _resolve(image_path)
        out = _jsonable(_toolkit().molnextr.predict_image_file(
            p, return_atoms_bonds=True, return_confidence=False))
        _trace("image_to_graph", {"image_path": image_path}, t0, True)
        return json.dumps(out, ensure_ascii=False)
    except Exception as e:
        _trace("image_to_graph", {"image_path": image_path}, t0, False, repr(e))
        return json.dumps({"error": repr(e)})


@mcp.tool()
@_quiet
def graph_to_smiles(symbols: list, coords: list, edges: list) -> str:
    """Regenerate SMILES from a (possibly edited) molecular graph
    (tool: Graph2SMILES, the same molnextr._convert_graph_to_smiles routine used
    to decode Image2Graph output).

    This is the counterpart of `image_to_graph`. The intended loop is:
    image_to_graph -> inspect/edit `symbols` -> graph_to_smiles.

    Args:
        symbols: per-atom labels. Element symbols are used as-is; recognised
                 R-group tokens (R, R1, R2, Ar, X, Y, ...) become wildcard atoms;
                 an abbreviation token (e.g. "iPr", "OMe", "Ph", or an explicit
                 substituent such as "2-ClC6H4") is expanded to its structure
                 where the abbreviation is known.
        coords:  per-atom [x, y] from image_to_graph (used for stereo perception,
                 so pass it through unchanged).
        edges:   N x N adjacency matrix from image_to_graph; 0 none, 1 single,
                 2 double, 3 triple, 4 aromatic, 5 wedge, 6 dash.

    Returns JSON: {"smiles": str, "molfile": str}, or {"error": str} if the
    edited graph cannot be sanitized into a valid molecule.
    """
    t0 = time.time()
    args = {"n_symbols": len(symbols) if hasattr(symbols, "__len__") else -1}
    try:
        from molnextr.chemistry import _convert_graph_to_smiles
        # note the argument order of the underlying routine: (coords, symbols, edges)
        smiles, molfile, _ok = _convert_graph_to_smiles(coords, symbols, edges)
        if not smiles:
            raise ValueError("conversion produced an empty SMILES")
        _trace("graph_to_smiles", args, t0, True)
        return json.dumps({"smiles": smiles, "molfile": molfile}, ensure_ascii=False)
    except Exception as e:
        _trace("graph_to_smiles", args, t0, False, repr(e))
        return json.dumps({"error": repr(e)})


@mcp.tool()
@_quiet
def parse_reaction_image(image_path: str) -> str:
    """Parse a reaction scheme into structured reaction roles
    (tool: RxnImgParser, built on RxnIM).

    Returns a JSON list of reactions; each reaction lists its components with a
    role and bounding box:
      [{"reactants": [{"category": str, "bbox": [...], "smiles": str}, ...],
        "conditions": [...], "products": [...]}, ...]

    Resolves roles across multi-step schemes (a species can be a product of one
    step and a reactant of the next). Operates on the drawn scheme itself.
    """
    t0 = time.time()
    try:
        p = _resolve(image_path)
        out = _jsonable(_rxn().predict_image_file(p, molnextr=True, ocr=True))
        _trace("parse_reaction_image", {"image_path": image_path}, t0, True,
               n_out=len(out) if isinstance(out, list) else None)
        return json.dumps(out, ensure_ascii=False)
    except Exception as e:
        _trace("parse_reaction_image", {"image_path": image_path}, t0, False, repr(e))
        return json.dumps({"error": repr(e)})


@mcp.tool()
@_quiet
def extract_molecules_with_labels(image_path: str) -> str:
    """Detect molecules AND recognize each one's structure and printed label in
    one pass (tools: MolDetector + Image2Graph + coreference resolution).

    Returns JSON:
      {"bboxes": [{"category": str, "bbox": [...], "smiles": str, ...}, ...],
       "corefs": [[mol_idx, label_idx], ...]}

    `corefs` links a molecule box to the text box holding its label (e.g. "3a").
    Use when a graphic shows several labelled molecules (e.g. a scope panel).
    """
    t0 = time.time()
    try:
        p = _resolve(image_path)
        out = _jsonable(_toolkit().extract_molecule_corefs_from_figures(
            [_pil(p)], molnextr=True, ocr=True)[0])
        _trace("extract_molecules_with_labels", {"image_path": image_path}, t0, True)
        return json.dumps(out, ensure_ascii=False)
    except Exception as e:
        _trace("extract_molecules_with_labels", {"image_path": image_path}, t0, False, repr(e))
        return json.dumps({"error": repr(e)})


@mcp.tool()
@_quiet
def ocr_image(image_path: str) -> str:
    """Read text embedded in a chemical graphic (tool: TesseractOCR).

    Returns JSON: {"text": str, "lines": [str, ...]}

    Chemical graphics are symbol-heavy; expect errors on sub/superscripts and
    on abbreviations (e.g. "iPr" may come back as "iP1"). Returns raw text only,
    with no chemical interpretation.
    """
    t0 = time.time()
    try:
        p = _resolve(image_path)
        import pytesseract
        from PIL import Image
        # The repo bundles Tesseract but does not put it on PATH; point at it so
        # this tool behaves identically for every arm on a clean machine.
        _bundled = os.path.join(_REPO, "Tesseract-OCR", "tesseract.exe")
        if os.path.exists(_bundled):
            pytesseract.pytesseract.tesseract_cmd = _bundled
            os.environ.setdefault("TESSDATA_PREFIX",
                                  os.path.join(_REPO, "Tesseract-OCR", "tessdata"))
        txt = pytesseract.image_to_string(Image.open(p))
        out = {"text": txt, "lines": [l for l in txt.splitlines() if l.strip()]}
        _trace("ocr_image", {"image_path": image_path}, t0, True, n_out=len(out["lines"]))
        return json.dumps(out, ensure_ascii=False)
    except Exception as e:
        _trace("ocr_image", {"image_path": image_path}, t0, False, repr(e))
        return json.dumps({"error": repr(e)})


# --------------------------------------------------------------------------- #
# Chemistry / assembly tools                                                   #
# --------------------------------------------------------------------------- #
@mcp.tool()
@_quiet
def reconstruct_smiles(template_smiles: str, variant_smiles: str,
                       target_template_smiles: str) -> str:
    """Transfer R-group substituents between a template and a variant
    (tool: SMILESReconstructor, RDKit-based).

    Given a product template (with R-group wildcards, e.g. "*"), a fully drawn
    product variant, and a target template (typically the reactant template),
    this performs substructure matching to isolate the variant's R-group
    fragments and grafts them onto the target template.

    Args:
        template_smiles: template the variant is an instance of (has wildcards)
        variant_smiles: the fully specified molecule
        target_template_smiles: template to graft the extracted fragments onto

    Returns JSON: {"smiles": str} on success, or {"error": str} if the
    substructure match fails or the result cannot be sanitized.
    """
    t0 = time.time()
    args = {"template_smiles": template_smiles, "variant_smiles": variant_smiles,
            "target_template_smiles": target_template_smiles}
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdRGroupDecomposition  # noqa: F401

        core = Chem.MolFromSmiles(template_smiles)
        var = Chem.MolFromSmiles(variant_smiles)
        tgt = Chem.MolFromSmiles(target_template_smiles)
        if core is None or var is None or tgt is None:
            raise ValueError("one or more inputs are not parseable SMILES")

        res, _ = rdRGroupDecomposition.RGroupDecompose([core], [var], asSmiles=True)
        if not res:
            raise ValueError("R-group decomposition found no match between "
                             "template and variant")
        groups = {k: v for k, v in res[0].items() if k.startswith("R")}

        out_smi = target_template_smiles
        repl = [v for k, v in sorted(groups.items()) if v and v not in ("[H]",)]
        mol = tgt
        for frag in repl:
            fm = Chem.MolFromSmiles(frag)
            if fm is None:
                continue
            star = Chem.MolFromSmiles("*")
            try:
                prods = AllChem.ReplaceSubstructs(mol, star, fm, replaceAll=False)
                if prods:
                    mol = prods[0]
            except Exception:
                continue
        Chem.SanitizeMol(mol)
        out_smi = Chem.MolToSmiles(mol)

        _trace("reconstruct_smiles", args, t0, True)
        return json.dumps({"smiles": out_smi, "r_groups": groups}, ensure_ascii=False)
    except Exception as e:
        _trace("reconstruct_smiles", args, t0, False, repr(e))
        return json.dumps({"error": repr(e)})


@mcp.tool()
@_quiet
def resolve_name_to_smiles(name: str) -> str:
    """Resolve a chemical name / abbreviation / IUPAC string to SMILES using the
    three web services (OPSIN, then PubChem, then NCI/CADD CIR).

    Args:
        name: e.g. "toluene", "Cs2CO3", "PhMe", "2-chlorobenzaldehyde"

    Returns JSON: {"smiles": str, "source": "opsin"|"pubchem"|"cir"} on success,
    or {"error": "unresolved"} if no service recognizes the string.

    These are live network services: they can be slow, rate-limited, or down.
    """
    t0 = time.time()
    try:
        import requests
        from urllib.parse import quote

        # 1) OPSIN: algorithmic IUPAC parsing
        try:
            r = requests.get(
                f"https://opsin.ch.cam.ac.uk/opsin/{quote(name)}.smi", timeout=15)
            if r.ok and r.text.strip():
                s = r.text.strip()
                _trace("resolve_name_to_smiles", {"name": name}, t0, True)
                return json.dumps({"smiles": s, "source": "opsin"})
        except Exception:
            pass

        # 2) PubChem exact compound match
        try:
            import pubchempy as pcp
            hits = pcp.get_compounds(name, "name")
            if hits:
                s = hits[0].isomeric_smiles or hits[0].canonical_smiles
                if s:
                    _trace("resolve_name_to_smiles", {"name": name}, t0, True)
                    return json.dumps({"smiles": s, "source": "pubchem"})
        except Exception:
            pass

        # 3) NCI/CADD Chemical Identifier Resolver
        try:
            r = requests.get(
                f"https://cactus.nci.nih.gov/chemical/structure/{quote(name)}/smiles",
                timeout=15)
            if r.ok and r.text.strip() and "<" not in r.text:
                s = r.text.strip().split()[0]
                _trace("resolve_name_to_smiles", {"name": name}, t0, True)
                return json.dumps({"smiles": s, "source": "cir"})
        except Exception:
            pass

        _trace("resolve_name_to_smiles", {"name": name}, t0, False, "unresolved")
        return json.dumps({"error": "unresolved"})
    except Exception as e:
        _trace("resolve_name_to_smiles", {"name": name}, t0, False, repr(e))
        return json.dumps({"error": repr(e)})


@mcp.tool()
@_quiet
def validate_smiles(smiles: str) -> str:
    """Check a SMILES string with RDKit and return its canonical form.

    Returns JSON: {"valid": bool, "canonical_smiles": str, "error": str}
    Wildcard atoms ("*") are permitted, so templates validate too.
    """
    t0 = time.time()
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        m = Chem.MolFromSmiles(smiles)
        out = ({"valid": False, "canonical_smiles": "", "error": "unparseable"}
               if m is None else
               {"valid": True, "canonical_smiles": Chem.MolToSmiles(m), "error": ""})
        _trace("validate_smiles", {"smiles": smiles}, t0, m is not None)
        return json.dumps(out, ensure_ascii=False)
    except Exception as e:
        _trace("validate_smiles", {"smiles": smiles}, t0, False, repr(e))
        return json.dumps({"valid": False, "canonical_smiles": "", "error": repr(e)})


# --------------------------------------------------------------------------- #
# Text tools                                                                   #
# --------------------------------------------------------------------------- #
@mcp.tool()
@_quiet
def interpret_conditions(image_path: str) -> str:
    """Extract and classify the reaction-condition text in a reaction graphic
    (tool: RxnConInterpreter, built on RxnIM).

    Returns JSON, one entry per detected reaction:
      {"conditions": [[{"text": str, "category": str, "bbox": [...]}, ...], ...]}

    `category` is the condition role assigned to the text span (e.g. reagent,
    solvent, temperature, time, yield). Reads the condition regions of the drawn
    scheme; it does not resolve names to SMILES (see resolve_name_to_smiles).
    """
    t0 = time.time()
    try:
        p = _resolve(image_path)
        raw = _rxn().predict_image_file(p, molnextr=True, ocr=True)
        per_reaction: List[List[Dict[str, Any]]] = []
        for reaction in raw:
            per_reaction.append([
                {"text": it.get("text", ""),
                 "category": it.get("category", ""),
                 "bbox": _jsonable(it.get("bbox", []))}
                for it in reaction.get("conditions", [])
            ])
        _trace("interpret_conditions", {"image_path": image_path}, t0, True,
               n_out=len(per_reaction))
        return json.dumps({"conditions": per_reaction}, ensure_ascii=False)
    except Exception as e:
        _trace("interpret_conditions", {"image_path": image_path}, t0, False, repr(e))
        return json.dumps({"error": repr(e)})


@mcp.tool()
@_quiet
def extract_reactions_from_text(text: str = "", image_path: str = "") -> str:
    """Extract structured reactions from prose (tool: ChemRxnExtractor).

    Two-stage model: tags product mentions, then assigns reaction roles
    (reactant / catalyst / solvent / ...) to each product.

    Args:
        text: the prose to analyze.
        image_path: alternatively, an image -- its text is OCR'd first.
                    Supply exactly one of `text` or `image_path`.

    Returns JSON: {"reactions": [...], "paragraph": str} where each reaction
    holds the tagged product and its role-labelled arguments. Returns an empty
    list when the text describes no reaction. Roles are labelled spans of text,
    not SMILES.
    """
    t0 = time.time()
    args = {"text": text[:120], "image_path": image_path}
    try:
        if bool(text.strip()) == bool(image_path.strip()):
            raise ValueError("supply exactly one of `text` or `image_path`")

        if image_path.strip():
            p = _resolve(image_path)
            import pytesseract
            from PIL import Image
            _bundled = os.path.join(_REPO, "Tesseract-OCR", "tesseract.exe")
            if os.path.exists(_bundled):
                pytesseract.pytesseract.tesseract_cmd = _bundled
                os.environ.setdefault(
                    "TESSDATA_PREFIX", os.path.join(_REPO, "Tesseract-OCR", "tessdata"))
            text = pytesseract.image_to_string(Image.open(p))

        paragraph = " ".join(l.strip() for l in text.splitlines() if l.strip())
        sents = _split_sentences(paragraph)

        with _LOAD_LOCK:
            if "cre" not in _M:
                _patch_chemrxnextractor()
                from chemrxnextractor import RxnExtractor
                _M["cre"] = RxnExtractor(os.path.join(_REPO, "cre_models_v0.1"),
                                         use_cuda=(_CFG["device"] == "cuda"))
        try:
            rxns = _M["cre"].get_reactions(sents)
        except AssertionError:
            # mirror ChemEAGLE's own fallback: retry sentence-by-sentence
            rxns = []
            for s in sents:
                try:
                    rxns.extend(_M["cre"].get_reactions([s]))
                except Exception:
                    continue
        rxns = [r for r in _jsonable(rxns) if r.get("reactions")]
        _trace("extract_reactions_from_text", args, t0, True, n_out=len(rxns))
        return json.dumps({"reactions": rxns, "paragraph": paragraph}, ensure_ascii=False)
    except Exception as e:
        _trace("extract_reactions_from_text", args, t0, False, repr(e))
        return json.dumps({"error": repr(e)})


@mcp.tool()
@_quiet
def chemical_ner(text: str) -> str:
    """Recognize chemical entity mentions in text (tool: MolNER, a BioBERT-Large
    model fine-tuned on CHEMDNER with BIO tagging).

    Args:
        text: a sentence or paragraph
    Returns JSON: [{"text": str, "category": str, "start": int, "end": int}, ...]
    Identifies WHERE chemical names occur; it does not resolve them to structures.
    """
    t0 = time.time()
    try:
        out = _jsonable(_toolkit().chemner.predict_strings([text])[0])
        _trace("chemical_ner", {"text": text}, t0, True,
               n_out=len(out) if isinstance(out, list) else None)
        return json.dumps(out, ensure_ascii=False)
    except Exception as e:
        _trace("chemical_ner", {"text": text}, t0, False, repr(e))
        return json.dumps({"error": repr(e)})


# --------------------------------------------------------------------------- #
def _selftest(image: str) -> int:
    """Exercise every tool once against a real image and report status."""
    print(f"self-test image: {image}\n" + "-" * 64)
    checks = [
        ("detect_molecules", lambda: detect_molecules(image)),
        ("image_to_graph", lambda: image_to_graph(image)),
        ("parse_reaction_image", lambda: parse_reaction_image(image)),
        ("extract_molecules_with_labels", lambda: extract_molecules_with_labels(image)),
        ("ocr_image", lambda: ocr_image(image)),
        ("interpret_conditions", lambda: interpret_conditions(image)),
        ("extract_reactions_from_text", lambda: extract_reactions_from_text(
            text="Treatment of benzaldehyde with Cs2CO3 in DMSO at 60 C afforded "
                 "the ketone 3aa in 85% yield.")),
        ("chemical_ner", lambda: chemical_ner("Treatment of benzaldehyde with Cs2CO3 in DMSO.")),
        ("resolve_name_to_smiles", lambda: resolve_name_to_smiles("toluene")),
        ("validate_smiles", lambda: validate_smiles("c1ccccc1C=O")),
        ("reconstruct_smiles", lambda: reconstruct_smiles(
            "*C(=O)NN=CC(F)(F)F", "O=C(NN=CC(F)(F)F)c1ccccc1", "*C(=O)N1NC(C(F)(F)F)N=C1N")),
    ]
    n_ok = 0
    for name, fn in checks:
        t0 = time.time()
        try:
            r = fn()
            d = json.loads(r)
            # note: a truthy "error" means failure; tools that always carry an
            # (empty) error field are fine.
            bad = isinstance(d, dict) and bool(d.get("error"))
            status = "FAIL" if bad else "ok"
            n_ok += 0 if bad else 1
            detail = (d.get("error", "") if bad else json.dumps(d, ensure_ascii=False))[:110]
        except Exception as e:
            status, detail = "EXC", repr(e)[:110]
        print(f"  {name:32s} {status:5s} {time.time()-t0:6.2f}s  {detail}")
    print("-" * 64)
    print(f"{n_ok}/{len(checks)} tools OK")
    return 0 if n_ok == len(checks) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root", default=None,
                    help="sandbox: only images under this dir may be read")
    ap.add_argument("--trace", default=None, help="JSONL tool-call trace file")
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--selftest", default=None, metavar="IMAGE",
                    help="run every tool once against IMAGE and exit")
    ap.add_argument("--transport", default="stdio",
                    choices=["stdio", "sse", "streamable-http"],
                    help="stdio spawns one server per agent session (models "
                         "reload every time); streamable-http serves one warm "
                         "server for a whole benchmark run")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8931)
    ap.add_argument("--warm", action="store_true",
                    help="load all models at startup instead of on first use")
    args = ap.parse_args()

    _CFG["image_root"] = args.image_root
    _CFG["trace"] = args.trace
    _CFG["device"] = args.device

    if args.selftest:
        return _selftest(args.selftest)

    if args.warm:
        with contextlib.redirect_stdout(sys.stderr):
            print("pre-warming models...", file=sys.stderr)
            t0 = time.time()
            _toolkit()
            _rxn()
            try:
                _patch_chemrxnextractor()
                from chemrxnextractor import RxnExtractor
                _M["cre"] = RxnExtractor(os.path.join(_REPO, "cre_models_v0.1"),
                                         use_cuda=(_CFG["device"] == "cuda"))
            except Exception as e:
                print(f"warn: could not pre-warm ChemRxnExtractor: {e!r}", file=sys.stderr)
            print(f"warm in {time.time() - t0:.0f}s", file=sys.stderr)

    if args.transport == "stdio":
        mcp.run()
    else:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        print(f"serving {args.transport} on {args.host}:{args.port}", file=sys.stderr)
        mcp.run(transport=args.transport)
    return 0


if __name__ == "__main__":
    sys.exit(main())

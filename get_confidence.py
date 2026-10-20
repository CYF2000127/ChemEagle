"""Confidence of the final output: a calibrated score for every extracted reaction.

Runs after the final synthesis in ``main.ChemEagle`` (on by default, ``use_confidence``). Two kinds of evidence feed a
gradient-boosted classifier with isotonic calibration, trained on the GT of the 324 benchmark figures:

* deterministic features of the output and of the run: placeholders left, fragments, charges (also of condition
  structures), stereocentres left unassigned where the figure reports a selectivity or the template assigns them,
  whether each molecule was copied from an agent result or written by the synthesis, which agents ran, events in
  the log;
* optionally (``confidence_llm``, off by default), the answers of a checker LLM that sees the figure and the
  extracted reactions once and answers eight fixed yes / no / unsure questions per reaction, including whether the
  figure draws wedged or hashed bonds on the product (compared by the program with the product SMILES) and whether
  a counter-ion the figure gives is missing. It never gives a number and never edits anything; its answers are
  features, the number comes from the calibrated model. The checker is the extraction model unless another is
  named; it is only called when ``confidence_models/`` holds a model calibrated on that checker's answers.

``score`` is the calibrated probability that the reaction is fully correct under the benchmark's hard rule
(tetrahedral stereocentres compared, every fragment matched). ``level``: high (score >= 0.85), medium (0.5 to 0.85),
low (< 0.5). Nothing is removed from the output; callers filter on the level.

The models in ``confidence_models/`` are written by the benchmark's ``confidence/export_models_v2.py``; the
feature code here must stay identical to the code that built their training data (the stereo and ion features are
computed for training by the functions of this module).
"""
import base64
import contextlib
import json
import math
import os
import re
from typing import Optional

from rdkit import Chem
from rdkit.rdBase import BlockLogs

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_HERE, "confidence_models")
PROMPT_PATH = os.path.join(_HERE, "prompt", "prompt_confidence_checker.txt")
LEVELS = ((0.85, "high"), (0.5, "medium"), (0.0, "low"))
QUESTIONS = ("drawn", "complete", "structures", "conditions", "resolved", "duplicate", "stereo_drawn", "counterion_missing")


def env_switch(value: Optional[bool], name: str, default: bool) -> bool:
    """An explicit argument wins; otherwise the environment variable (1/0, true/false, on/off); else default."""
    if value is not None:
        return bool(value)
    raw = (os.getenv(name) or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


# ---------------------------------------------------------------- run log
class _Tee:
    """Writes through to the current stream and keeps a copy: three features come from the run log."""

    def __init__(self, target, chunks):
        self._target, self._chunks = target, chunks

    def write(self, s):
        self._chunks.append(s)
        return self._target.write(s)

    def flush(self):
        self._target.flush()

    def __getattr__(self, name):
        return getattr(self._target, name)


@contextlib.contextmanager
def capture_log():
    """Copy of everything printed to stdout / stderr inside the block, as a list of strings."""
    import sys
    chunks = []
    with contextlib.redirect_stdout(_Tee(sys.stdout, chunks)), contextlib.redirect_stderr(_Tee(sys.stderr, chunks)):
        yield chunks


# ---------------------------------------------------------------- molecule helpers
_INVALID_TOKENS = {"", "none", "null", "n/a", "na", "?", "*"}
_METAL_Z = frozenset(list(range(3, 5)) + list(range(11, 14)) + list(range(19, 32))
                     + list(range(37, 51)) + list(range(55, 85)) + list(range(87, 104))) \
    - frozenset({5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 33, 34, 35, 36, 52, 53, 54, 85, 86})


def _bench_canon(smi):
    """The benchmark's comparison form (evaluation.canon_smiles, stereo stripped, largest fragment)."""
    if smi is None:
        return None
    s = (smi if isinstance(smi, str) else str(smi)).strip()
    if not s or s.lower() in _INVALID_TOKENS:
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        for a in mol.GetAtoms():
            if a.GetAtomicNum() == 0:
                a.SetIsotope(0)
                a.SetAtomMapNum(0)
        Chem.RemoveStereochemistry(mol)
        if (not any(a.GetAtomicNum() == 0 for a in mol.GetAtoms())
                and any(a.GetFormalCharge() or a.GetAtomicNum() in _METAL_Z for a in mol.GetAtoms())):
            try:
                m2 = Chem.MolFromInchi(Chem.MolToInchi(mol))
                if m2 is not None:
                    mol = m2
            except Exception:
                pass
        out = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        reparsed = Chem.MolFromSmiles(out)
        if reparsed is not None:
            out = Chem.MolToSmiles(reparsed, canonical=True, isomericSmiles=False)
        return out
    except Exception:
        return None


def _drawn_condition_smiles(rx):
    """evaluation.drawn_condition_smiles: condition structures the figure draws (label, catalyst or drawn=true)."""
    out = []
    for c in rx.get("conditions") or []:
        if not isinstance(c, dict):
            continue
        role = str(c.get("role", "")).strip().lower()
        label = str(c.get("label") or "").strip()
        drawn = c.get("drawn")
        if not isinstance(drawn, bool):
            drawn = bool(label and label.lower() != "none") or role == "catalyst"
        if not drawn:
            continue
        s = _bench_canon(c.get("smiles"))
        if s:
            out.append(s)
    return out


def _canon(s):
    """Canonical SMILES with stereo kept (None when RDKit cannot read it)."""
    if not isinstance(s, str) or not s.strip() or s.strip().lower() == "none":
        return None
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m is not None else None


def _mol_info(s):
    info = {"present": isinstance(s, str) and bool(s.strip()) and s.strip().lower() != "none"}
    m = Chem.MolFromSmiles(s) if info["present"] else None
    info["valid"] = m is not None
    info["wild"] = info["present"] and "*" in s
    if m is None:
        return info
    frags = Chem.GetMolFrags(m, asMols=True)
    main = max(frags, key=lambda f: f.GetNumHeavyAtoms())
    info.update(n_frag=len(frags), heavy=main.GetNumHeavyAtoms(), rings=main.GetRingInfo().NumRings(),
                net_charge=Chem.GetFormalCharge(m), main_charge=Chem.GetFormalCharge(main),
                stereo=any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in m.GetAtoms())
                or any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in m.GetBonds()),
                hetero=sum(1 for a in main.GetAtoms() if a.GetAtomicNum() not in (1, 6)),
                metal=any(a.GetAtomicNum() in (3, 11, 19, 37, 55, 12, 20) or 21 <= a.GetAtomicNum() <= 30
                          or 39 <= a.GetAtomicNum() <= 48 or 72 <= a.GetAtomicNum() <= 80 for a in m.GetAtoms()))
    return info


def _agent_molecules(art):
    """Canonical SMILES of every molecule the agents produced, and the agents' reaction signatures."""
    mols, sigs = set(), set()
    for log in art.get("execution_logs") or []:
        r = log.get("result")
        items = r.get("molecule_coref") if isinstance(r, dict) else r if isinstance(r, list) else None
        for m in items or []:
            if isinstance(m, dict):
                c = _canon(m.get("smiles"))
                if c:
                    mols.add(c)
        if isinstance(r, dict) and isinstance(r.get("reactions"), list):
            for rx in r["reactions"]:
                if not isinstance(rx, dict):
                    continue
                rs = tuple(sorted(filter(None, (_canon(m.get("smiles")) for m in rx.get("reactants") or [] if isinstance(m, dict)))))
                ps = tuple(sorted(filter(None, (_canon(m.get("smiles")) for m in rx.get("products") or [] if isinstance(m, dict)))))
                mols.update(rs)
                mols.update(ps)
                for c in rx.get("conditions") or []:
                    if isinstance(c, dict):
                        cs = _canon(c.get("smiles"))
                        if cs:
                            mols.add(cs)
                sigs.add((rs, ps))
    return mols, sigs


# ---------------------------------------------------------------- deterministic features
_ROUTES = {"process_reaction_image_with_product_variant_R_group": "route_variant",
           "process_reaction_image_with_table_R_group": "route_table",
           "get_full_reaction_template": "route_template",
           "get_multi_molecular_full": "route_molecular",
           "get_reaction_con": "route_condition"}


def _figure_context(rxns, art, log_text):
    n = len(rxns)
    prods = [tuple(sorted(filter(None, (_canon(m.get("smiles")) for m in rx.get("products") or [] if isinstance(m, dict)))))
             for rx in rxns]
    reacts = [tuple(sorted(filter(None, (_canon(m.get("smiles")) for m in rx.get("reactants") or [] if isinstance(m, dict)))))
              for rx in rxns]
    wild = [any("*" in (m.get("smiles") or "") for side in ("reactants", "products") for m in rx.get(side) or [] if isinstance(m, dict))
            for rx in rxns]
    planned = art.get("ordered_agents") or []
    notes = art.get("observer_notes") or []
    ctx = {
        "fig_n_reactions": n,
        "fig_frac_wild": sum(wild) / n if n else 0.0,
        "fig_all_same_reactants": float(n > 2 and len(set(reacts)) == 1),
        "fig_n_distinct_products": len(set(prods)),
        "fig_failed_agents": len(art.get("failed_agents") or []),
        "fig_rgroup_fallback": float(any("was run although the plan omitted it" in x for x in notes)),
        "fig_text_agent": float(art.get("text_extraction_result") is not None),
        "fig_plan_rejected": float("plan rejected" in log_text),
        "fig_plan_lenient": float("_lenient" in log_text),
        "fig_stream_retry": float("retrying" in log_text),
    }
    for agent, name in _ROUTES.items():
        ctx[name] = float(agent in planned)
    return ctx, prods


def _reaction_features(rx, idx, prods_all, agent_mols, agent_sigs):
    f = {}
    rs = [m for m in rx.get("reactants") or [] if isinstance(m, dict)]
    ps = [m for m in rx.get("products") or [] if isinstance(m, dict)]
    cs = [c for c in rx.get("conditions") or [] if isinstance(c, dict)]
    ri = [_mol_info(m.get("smiles")) for m in rs]
    pi = [_mol_info(m.get("smiles")) for m in ps]
    allm = ri + pi
    f["n_reactants"], f["n_products"] = len(rs), len(ps)
    f["n_cond"] = len(cs)
    f["n_cond_smiles"] = sum(1 for c in cs if _canon(c.get("smiles")))
    f["n_cond_drawn"] = len(_drawn_condition_smiles(rx))
    f["n_cond_invalid"] = sum(1 for c in cs if c.get("smiles") and str(c.get("smiles")).lower() != "none" and _canon(c.get("smiles")) is None)
    f["n_cond_wild"] = sum(1 for c in cs if "*" in str(c.get("smiles") or ""))
    f["n_missing_smiles"] = sum(1 for x in allm if not x["present"])
    f["n_invalid"] = sum(1 for x in allm if x["present"] and not x["valid"])
    f["n_wild_reactant"] = sum(1 for x in ri if x["wild"])
    f["n_wild_product"] = sum(1 for x in pi if x["wild"])
    f["n_multifrag"] = sum(1 for x in allm if x.get("n_frag", 1) > 1)
    f["n_unbalanced"] = sum(1 for x in allm if x.get("net_charge", 0) != 0)
    f["n_charged_main"] = sum(1 for x in allm if x.get("main_charge", 0) != 0)
    f["n_metal"] = sum(1 for x in allm if x.get("metal"))
    f["prod_heavy"] = max([x.get("heavy", 0) for x in pi] or [0])
    f["prod_rings"] = max([x.get("rings", 0) for x in pi] or [0])
    f["prod_hetero"] = max([x.get("hetero", 0) for x in pi] or [0])
    f["react_heavy_sum"] = sum(x.get("heavy", 0) for x in ri)
    f["heavy_balance"] = f["prod_heavy"] - f["react_heavy_sum"]
    f["prod_stereo"] = float(any(x.get("stereo") for x in pi))
    f["prod_label"] = float(any(str(m.get("label") or "").strip() not in ("", "None") for m in ps))
    rc = [_canon(m.get("smiles")) for m in rs]
    pc = [_canon(m.get("smiles")) for m in ps]
    f["prod_equals_reactant"] = float(bool(set(filter(None, pc)) & set(filter(None, rc))))
    key = tuple(sorted(filter(None, pc)))
    f["dup_product_in_figure"] = float(sum(1 for p in prods_all if p == key and key) > 1)
    known = [c for c in rc + pc if c]
    f["frac_copied"] = (sum(1 for c in known if c in agent_mols) / len(known)) if known else 0.0
    f["n_written"] = sum(1 for c in known if c not in agent_mols)
    f["product_copied"] = float(bool(pc) and all(c in agent_mols for c in pc if c))
    f["reaction_copied"] = float((tuple(sorted(filter(None, rc))), key) in agent_sigs)
    f["is_first"] = float(idx == 0)
    texts = " ".join(str(c.get("text") or "") for c in cs) + " " + json.dumps(rx.get("additional_info") or "", ensure_ascii=False)
    f["has_yield_text"] = float(bool(re.search(r"\d\s*%", texts)))
    f.update(_stereo_ion_features(rx))
    return f


# stereochemistry left unassigned and counter-ion problems (scorer v2, 2026-10-20): errors that leave a valid molecule
_SELECTIVITY = re.compile(r"(\bee\b|e\.e\.|\bdr\b|d\.r\.|\ber\b|e\.r\.|\bcis\b|\btrans\b|\bsyn\b|\banti\b|enantio|diastereo)",
                          re.I)


def _stereo_counts(s):
    """(assigned, unassigned) potential tetrahedral stereocentres of one SMILES"""
    m = Chem.MolFromSmiles(s) if isinstance(s, str) and s.strip() and s.strip().lower() != "none" else None
    if m is None:
        return 0, 0
    c = Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False)
    return sum(1 for _, lab in c if lab != "?"), sum(1 for _, lab in c if lab == "?")


def _ion_flags(s):
    """(net charge != 0, stray fragments: <= 3 carbons and nothing else, or a lone '*'; neutral halogen next to a cation)"""
    m = Chem.MolFromSmiles(s) if isinstance(s, str) and s.strip() and s.strip().lower() != "none" else None
    if m is None:
        return 0, 0, 0
    frags = Chem.GetMolFrags(m, asMols=True)
    big = max(frags, key=lambda f: f.GetNumHeavyAtoms())
    stray = 0
    for fr in frags:
        if fr is big:
            continue
        syms = [a.GetSymbol() for a in fr.GetAtoms()]
        if syms == ["*"] or (all(x == "C" for x in syms) and len(syms) <= 3 and Chem.GetFormalCharge(fr) == 0):
            stray += 1
    cation = any(Chem.GetFormalCharge(fr) > 0 for fr in frags)
    halide = any(fr.GetNumAtoms() == 1 and fr.GetAtomWithIdx(0).GetSymbol() in ("F", "Cl", "Br", "I")
                 and Chem.GetFormalCharge(fr) == 0 for fr in frags)
    return int(Chem.GetFormalCharge(m) != 0), stray, int(cation and halide)


def _rx_text(rx):
    cs = [c for c in rx.get("conditions") or [] if isinstance(c, dict)]
    return " ".join(str(c.get("text") or "") for c in cs) + " " + json.dumps(rx.get("additional_info") or "", ensure_ascii=False)


def _stereo_ion_features(rx):
    sm = lambda side: [m.get("smiles") for m in rx.get(side) or [] if isinstance(m, dict)]  # noqa: E731
    cond = [c.get("smiles") for c in rx.get("conditions") or [] if isinstance(c, dict)]
    pa = pu = 0
    for s in sm("products"):
        a, u = _stereo_counts(s)
        pa, pu = pa + a, pu + u
    cf = [_ion_flags(s) for s in cond]
    af = [_ion_flags(s) for s in sm("reactants") + sm("products")]
    return {"prod_n_assigned": pa, "prod_n_unassigned": pu,
            "rx_selectivity_text": float(bool(_SELECTIVITY.search(_rx_text(rx)))),
            "n_cond_charged": sum(x[0] for x in cf), "n_charged_any": sum(x[0] for x in cf + af),
            "n_stray_fragment": sum(x[1] for x in cf + af), "n_neutral_halide_on_cation": sum(x[2] for x in cf + af)}


def _figure_stereo(rxns, rows):
    """Figure-level stereo features, added to the per-reaction rows (which already hold _stereo_ion_features)."""
    fig_sel = any(_SELECTIVITY.search(_rx_text(rx)) for rx in rxns)
    tmpl_assigned = 0
    for rx in rxns:
        for m in rx.get("products") or []:
            s = m.get("smiles") if isinstance(m, dict) else None
            if isinstance(s, str) and "*" in s:
                tmpl_assigned = max(tmpl_assigned, _stereo_counts(s)[0])
    frac = sum(1 for r in rows if r["prod_n_unassigned"] > 0) / len(rows) if rows else 0.0
    for rx, r in zip(rxns, rows):
        concrete = not any(isinstance(m, dict) and "*" in (m.get("smiles") or "") for m in rx.get("products") or [])
        r["fig_selectivity_text"] = float(fig_sel)
        r["unassigned_with_selectivity"] = float(r["prod_n_unassigned"] > 0 and fig_sel)
        r["fewer_than_template"] = float(concrete and tmpl_assigned > r["prod_n_assigned"])
        r["fig_frac_unassigned"] = frac
        r["alone_unassigned"] = float(r["prod_n_unassigned"] > 0 and frac < 0.5)
    return rows


def extraction_family(model_name: Optional[str]) -> Optional[str]:
    """The extraction model as the training data knows it (one-hot model_gemini / model_mini / model_luna)."""
    name = (model_name or "").lower().split("/")[-1]
    if name.startswith("gemini"):
        return "gemini"
    if name.startswith("gpt-5-mini"):
        return "mini"
    if "luna" in name:
        return "luna"
    return None


def deterministic_features(rxns, artifacts, log_text, model_name):
    """One feature dict per reaction (same names and code as the training data)."""
    with BlockLogs():
        ctx, prods_all = _figure_context(rxns, artifacts, log_text)
        agent_mols, agent_sigs = _agent_molecules(artifacts)
        family = extraction_family(model_name)
        rows = []
        for i, rx in enumerate(rxns):
            row = dict(ctx)
            row.update(_reaction_features(rx, i, prods_all, agent_mols, agent_sigs))
            for fam in ("gemini", "mini", "luna"):
                row["model_" + fam] = float(family == fam)
            rows.append(row)
        _figure_stereo(rxns, rows)
    return rows


# ---------------------------------------------------------------- checker
def _compact(rx, k):
    def mol(m):
        out = {"smiles": m.get("smiles")}
        if str(m.get("label") or "").strip() not in ("", "None"):
            out["label"] = m["label"]
        return out
    conds = []
    for c in rx.get("conditions") or []:
        if isinstance(c, dict):
            d = {key: c[key] for key in ("role", "text", "smiles", "label") if c.get(key) not in (None, "", "None")}
            if d:
                conds.append(d)
        elif isinstance(c, str):
            conds.append({"text": c})
    return {"id": "R%d" % (k + 1),
            "reactants": [mol(m) for m in rx.get("reactants") or [] if isinstance(m, dict)],
            "products": [mol(m) for m in rx.get("products") or [] if isinstance(m, dict)],
            "conditions": conds}


def checker_prompt(rxns):
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        template = f.read()
    return template.replace("{reactions}", json.dumps([_compact(r, k) for k, r in enumerate(rxns)], ensure_ascii=False))


def run_checker(image_path, rxns, checker_model, base_url=None, api_key=None):
    """One call: the figure plus the extracted reactions; returns (parsed reply, raw text, usage)."""
    import llm_client as llm
    client = llm.get_client(api_key=api_key, base_url=base_url)
    with open(image_path, "rb") as f:
        img = base64.b64encode(f.read()).decode()
    r = client.chat.completions.create(
        model=checker_model, **llm.model_kwargs(checker_model), response_format={"type": "json_object"},
        messages=[{"role": "system", "content": "You are a careful reviewer of extracted chemical reactions."},
                  {"role": "user", "content": [{"type": "text", "text": checker_prompt(rxns)},
                                               {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}}]}])
    raw = r.choices[0].message.content
    u = getattr(r, "usage", None)
    usage = {"in": getattr(u, "prompt_tokens", None), "out": getattr(u, "completion_tokens", None)}
    return llm.loads_lenient(raw), raw, usage


def _answer(v):
    return {"yes": 1.0, "no": 0.0, "unsure": 0.5}.get(str(v).strip().lower(), math.nan)


def checker_features(parsed, n):
    """One dict of v_* features per reaction, from the checker's parsed reply (NaN where unanswered)."""
    by_id = {}
    for x in parsed.get("reactions") or []:
        if not isinstance(x, dict):
            continue
        try:
            by_id[int(str(x.get("id")).strip().lstrip("Rr")) - 1] = x
        except ValueError:
            continue
    exp = parsed.get("expected_reactions")
    exp = float(exp) if isinstance(exp, (int, float)) else math.nan
    rows = []
    for i in range(n):
        x = by_id.get(i, {})
        f = {"v_answered": float(bool(x)),
             "v_entry": float(bool(str(x.get("entry") or "").strip())),
             "v_problem": float(bool(str(x.get("problem") or "").strip())) if x else math.nan,
             "v_expected_minus_pred": exp - n if exp == exp else math.nan,
             "v_n_missing": float(len(parsed.get("missing") or []))}
        for q in QUESTIONS:
            f["v_" + q] = _answer(x.get(q)) if x else math.nan
        rows.append(f)
    return rows, by_id


def checker_derived(det_row, ver_row):
    """The program's comparison of the checker's stereo_drawn answer with the product SMILES (the checker is not
    asked whether a configuration is right): drawn and a centre left unassigned, drawn and no centre assigned."""
    sd = ver_row.get("v_stereo_drawn", math.nan)
    if sd != sd:
        return {"v_drawn_but_unassigned": math.nan, "v_drawn_but_none": math.nan}
    drawn = sd == 1.0
    return {"v_drawn_but_unassigned": float(drawn and det_row["prod_n_unassigned"] > 0),
            "v_drawn_but_none": float(drawn and det_row["prod_n_assigned"] == 0)}


# ---------------------------------------------------------------- scoring
_BUNDLES = {}


def _bundle(name):
    if name not in _BUNDLES:
        import joblib
        _BUNDLES[name] = joblib.load(os.path.join(MODEL_DIR, name + ".joblib"))
    return _BUNDLES[name]


def available_checkers():
    if not os.path.isdir(MODEL_DIR):
        return []
    return sorted(f[len("checker_"):-len(".joblib")] for f in os.listdir(MODEL_DIR)
                  if f.startswith("checker_") and f.endswith(".joblib"))


def level_of(p):
    return next(name for cut, name in LEVELS if p >= cut)


def _wild_products(rx):
    return [m.get("smiles") for m in rx.get("products") or [] if isinstance(m, dict) and "*" in (m.get("smiles") or "")]


def template_reactions(rxns):
    """Indices of generic template reactions: a product still carries placeholders (R, Ar, X as *) and a
    concrete product of another reaction in the same figure is an instance of it (placeholders match any
    atom). Templates are part of the benchmark GT and score like other reactions; the flag only tells them
    apart from concrete reactions, whose leftover placeholders are flagged unresolved_placeholder."""
    params = Chem.AdjustQueryParameters()
    params.makeDummiesQueries = True
    params.adjustDegree = False
    params.adjustRingCount = False
    concrete = []
    with BlockLogs():
        for j, rx in enumerate(rxns):
            for m in rx.get("products") or []:
                s = m.get("smiles") if isinstance(m, dict) else None
                if isinstance(s, str) and s and "*" not in s:
                    pm = Chem.MolFromSmiles(s)
                    if pm is not None:
                        concrete.append((j, pm))
        out = set()
        for i, rx in enumerate(rxns):
            queries = [Chem.AdjustQueryProperties(m, params) for m in
                       (Chem.MolFromSmiles(s) for s in _wild_products(rx)) if m is not None]
            if queries and any(j != i and pm.HasSubstructMatch(q) for j, pm in concrete for q in queries):
                out.add(i)
    return out


def _flags(det, ver, template=False):
    out = []
    if template:
        out.append("template_reaction")
    elif det["n_wild_reactant"] + det["n_wild_product"]:
        out.append("unresolved_placeholder")
    if det["n_invalid"] or det["n_missing_smiles"]:
        out.append("invalid_or_missing_smiles")
    if det["n_multifrag"]:
        out.append("multi_fragment_molecule")
    if det["n_unbalanced"] or det["n_cond_charged"]:
        out.append("unbalanced_charge")
    if det["n_stray_fragment"]:
        out.append("stray_fragment")
    if det["unassigned_with_selectivity"] or det["fewer_than_template"] or (ver and ver.get("v_stereo_drawn") == 1
                                                                             and det["prod_n_unassigned"]):
        out.append("unassigned_stereocenter")
    if det["dup_product_in_figure"]:
        out.append("duplicate_product")
    if det["n_written"]:
        out.append("molecule_written_by_synthesis")
    if ver:
        for q, flag in (("v_structures", "checker_structures_no"), ("v_complete", "checker_incomplete"),
                        ("v_conditions", "checker_conditions_no"), ("v_resolved", "checker_unresolved")):
            if ver.get(q) == 0:
                out.append(flag)
        if ver.get("v_counterion_missing") == 1:
            out.append("checker_counterion_missing")
    return out


def attach_confidence(output, image_path, artifacts, log_text, *, model_name, use_llm=False,
                      checker_model=None, base_url=None, api_key=None):
    """Adds ``confidence`` to every reaction of ``output`` and a ``confidence_summary``; returns ``output``.

    ``checker_model`` defaults to ``model_name``. The checker's raw reply and the features go to
    ``artifacts['confidence']`` for later recalibration. A checker that fails, or has no calibrated model,
    leaves the deterministic scores and a warning."""
    import numpy as np
    if not isinstance(output, dict):
        return output
    rxns = [r for r in output.get("reactions") or [] if isinstance(r, dict)]
    artifacts = artifacts if artifacts is not None else {}
    warnings = []
    checker_model = checker_model or model_name
    family = extraction_family(model_name)
    if family is None:
        warnings.append(f"extraction model {model_name} is not in the calibration data (gemini-3.7-flash, "
                        f"gpt-5-mini, gpt-5.6-luna); scores are less reliable")
    summary = {"scorer": "deterministic", "checker_model": None, "output_reactions": len(rxns)}
    record = {"model_name": model_name, "checker_model": None}
    if not rxns:
        summary.update(levels={"high": 0, "medium": 0, "low": 0}, low_confidence_reactions=[],
                       template_reactions=[], warnings=warnings)
        output["confidence_summary"] = summary
        artifacts["confidence"] = record
        return output

    det = deterministic_features(rxns, artifacts, log_text, model_name)
    ver, by_id, parsed, cal = None, {}, None, None
    if use_llm:
        have = available_checkers()
        cal = checker_model if checker_model in have else str(checker_model or "").split("/")[-1]   # provider/model ids
        if cal not in have:
            warnings.append(f"no calibrated model for checker {checker_model} (have {', '.join(have)}); "
                            f"deterministic scores only, checker not called")
        else:
            try:
                parsed, raw, usage = run_checker(image_path, rxns, checker_model, base_url=base_url, api_key=api_key)
                if not isinstance(parsed, dict):
                    raise ValueError("checker reply is not a JSON object")
                ver, by_id = checker_features(parsed, len(rxns))
                record.update(checker_model=checker_model, checker_calibration=cal, checker_raw=raw, checker_usage=usage)
            except Exception as exc:
                warnings.append(f"checker {checker_model} failed ({type(exc).__name__}: {str(exc)[:200]}); "
                                f"deterministic scores only")
                print(f"WARNING [confidence]: checker {checker_model} failed, deterministic scores only: {exc!r}")
                parsed, ver, by_id = None, None, {}

    name = ("checker_" + cal) if ver is not None else "deterministic"
    bundle = _bundle(name)
    rows = [dict(d, **(dict(ver[i], **checker_derived(d, ver[i])) if ver is not None else {})) for i, d in enumerate(det)]
    X = np.array([[float(r[k]) for k in bundle["features"]] for r in rows])   # KeyError = feature code out of step
    p = bundle["calibrator"].predict(bundle["model"].predict_proba(X)[:, 1])

    counts = {"high": 0, "medium": 0, "low": 0}
    low_ids = []
    templates = template_reactions(rxns)
    for i, rx in enumerate(rxns):
        lv = level_of(float(p[i]))
        counts[lv] += 1
        conf = {"score": round(float(p[i]), 3), "level": lv, "flags": _flags(det[i], ver[i] if ver else None, i in templates)}
        note = str((by_id.get(i) or {}).get("problem") or "").strip()
        if note:
            conf["checker_note"] = note
        rx["confidence"] = conf
        if lv == "low":
            low_ids.append(rx.get("reaction_id", str(i)))
    summary.update(scorer="deterministic+checker" if ver is not None else "deterministic",
                   checker_model=checker_model if ver is not None else None,
                   levels=counts, low_confidence_reactions=low_ids,
                   template_reactions=[rxns[i].get("reaction_id", str(i)) for i in sorted(templates)])
    if parsed is not None:
        summary["expected_reactions_by_checker"] = parsed.get("expected_reactions")
        summary["missing_by_checker"] = parsed.get("missing") or []
    summary["warnings"] = warnings
    output["confidence_summary"] = summary
    record.update(model=name, features=rows, scores=[float(x) for x in p], warnings=warnings)
    artifacts["confidence"] = record
    return output

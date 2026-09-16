from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS
from scipy.optimize import linear_sum_assignment


RDLogger.DisableLog("rdApp.*")


# Metals whose salts InChI disconnects, so the covalent and ionic spellings of
# the same substance unify (B, C, N, O, F, Ne, Si, P, S, Cl, Ar, As, Se, Br, Kr,
# Te, I, Xe, At, Rn are excluded: those are part of the molecule proper).
_METAL_Z = frozenset(list(range(3, 5)) + list(range(11, 14)) + list(range(19, 32))
                     + list(range(37, 51)) + list(range(55, 85)) + list(range(87, 104))) \
    - frozenset({5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 33, 34, 35, 36, 52, 53, 54, 85, 86})

# GED compares molecules through a maximum common substructure search, which is
# slow: at most this many seconds per molecule pair.
MCS_TIMEOUT = 5


# =========================================================================== #
#  I/O                                                                         #
# =========================================================================== #
def load_samples(path: str) -> Dict[str, dict]:
    """Load a predictions / ground-truth JSON into {file_name: sample}.

    Accepts the three shapes in use: a dict keyed by file name
    (benchmark_gt_canonical.json, predictions_plan_*.json), a list of entries
    carrying ``file_name`` (GT1..GT4.json), and a list without any id (keyed
    positionally, which is only safe when both sides share the order).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return {str(k): v for k, v in data.items()}
    if isinstance(data, list):
        out: Dict[str, dict] = {}
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            sid = (item.get("file_name") or item.get("id") or item.get("image_id")
                   or item.get("image_path") or f"sample_{i}")
            out[str(sid)] = item
        return out
    raise ValueError(f"Unsupported JSON top-level type in {path}: {type(data)}")


def load_ground_truth(paths: Sequence[str]) -> Dict[str, dict]:
    """Merge one or more GT files (GT1..GT4 or the canonical file) into one dict."""
    gts: Dict[str, dict] = {}
    for p in paths:
        part = load_samples(p)
        dup = set(part) & set(gts)
        if dup:
            raise ValueError(f"{p}: {len(dup)} file_name(s) already present in an earlier GT file, "
                             f"e.g. {sorted(dup)[:3]}")
        gts.update(part)
    return gts


# =========================================================================== #
#  SMILES canonicalisation                                                     #
# =========================================================================== #
_INVALID_TOKENS = {"", "none", "null", "n/a", "na", "?", "*"}


@functools.lru_cache(maxsize=1_000_000)
def canon_smiles(smi: Any) -> Optional[str]:
    if smi is None:
        return None
    if not isinstance(smi, str):
        smi = str(smi)
    s = smi.strip()
    if not s or s.lower() in _INVALID_TOKENS:
        return None
    mol = Chem.MolFromSmiles(s)
    stereo = False   # with or w/o stereo.
    ions = True     # with or w/o ions.
    if mol is None:
        return None
    try:
        if not ions:
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            if len(frags) > 1:
                mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())

        for a in mol.GetAtoms():
            if a.GetAtomicNum() == 0:
                a.SetIsotope(0)
                a.SetAtomMapNum(0)

        if stereo:
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            for b in mol.GetBonds():
                b.SetStereo(Chem.BondStereo.STEREONONE)
                b.SetBondDir(Chem.BondDir.NONE)
        else:
            Chem.RemoveStereochemistry(mol)

        if (not any(a.GetAtomicNum() == 0 for a in mol.GetAtoms())
                and any(a.GetFormalCharge() or a.GetAtomicNum() in _METAL_Z for a in mol.GetAtoms())):
            try:
                m2 = Chem.MolFromInchi(Chem.MolToInchi(mol))
                if m2 is not None:
                    mol = m2
            except Exception:
                pass

        out = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=stereo)
        reparsed = Chem.MolFromSmiles(out)
        if reparsed is not None:
            out = Chem.MolToSmiles(reparsed, canonical=True, isomericSmiles=stereo)
        return out
    except Exception:
        return None


def _iter_smiles_from_list(items: Any) -> Iterable[str]:
    """Yield raw SMILES strings from a list of dicts (or of plain strings)."""
    if not isinstance(items, list):
        return
    for it in items:
        if isinstance(it, dict):
            s = it.get("smiles")
            if isinstance(s, str):
                yield s
        elif isinstance(it, str):
            yield it


def extract_reactant_smiles(rxn: dict) -> List[str]:
    return [c for c in (canon_smiles(s) for s in _iter_smiles_from_list(rxn.get("reactants"))) if c]


def extract_product_smiles(rxn: dict) -> List[str]:
    return [c for c in (canon_smiles(s) for s in _iter_smiles_from_list(rxn.get("products"))) if c]


_CONDITION_SMILES_ROLES = {"reagent", "reagents", "solvent", "solvents", "catalyst", "catalysts"}


def extract_condition_smiles(rxn: dict) -> List[str]:
    """Canonical SMILES of the condition entries that carry a chemical role."""
    out: List[str] = []
    conds = rxn.get("conditions")
    if not isinstance(conds, list):
        return out
    for c in conds:
        if not isinstance(c, dict):
            continue
        role = str(c.get("role", "")).strip().lower()
        if role and role not in _CONDITION_SMILES_ROLES:
            continue
        s = canon_smiles(c.get("smiles"))
        if s:
            out.append(s)
    return out


def drawn_condition_smiles(rxn: dict) -> List[str]:
    """The condition molecules the figure DRAWS as structures (the hard-match set).

    A condition counts as drawn when it carries a compound label (5b, N1,
    ent-A7) or is filed as a catalyst, unless it is explicitly annotated
    ``"drawn": false`` (named in the text, structure not in this figure).
    """
    out: List[str] = []
    for c in rxn.get("conditions") or []:
        if not isinstance(c, dict):
            continue
        role = str(c.get("role", "")).strip().lower()
        label = str(c.get("label") or "").strip()
        drawn = (label and label.lower() != "none") or role == "catalyst"
        if c.get("drawn") is False:
            drawn = False
        if not drawn:
            continue
        s = canon_smiles(c.get("smiles"))
        if s:
            out.append(s)
    return out


def extract_all_smiles(rxn: dict) -> List[str]:
    return extract_reactant_smiles(rxn) + extract_product_smiles(rxn) + extract_condition_smiles(rxn)


# =========================================================================== #
#  Reaction matching                                                           #
# =========================================================================== #
def _multiset(xs: Iterable[str]) -> Tuple[Tuple[str, int], ...]:
    counts: Dict[str, int] = defaultdict(int)
    for x in xs:
        counts[x] += 1
    return tuple(sorted(counts.items()))


def soft_signature(rxn: dict) -> Tuple[Any, Any]:
    return (_multiset(extract_reactant_smiles(rxn)), _multiset(extract_product_smiles(rxn)))


def hard_signature(rxn: dict) -> Tuple[Any, Any, Any]:
    return (_multiset(extract_reactant_smiles(rxn)), _multiset(extract_product_smiles(rxn)),
            _multiset(extract_condition_smiles(rxn)))


def _reconcile_roles(pred_r: List[str], pred_c: List[str], gt_r: List[str], gt_c: List[str]):
    """Drop the reactant / condition confusion from one candidate pair.

    The surplus predicted reactants the GT files as conditions are removed, the
    GT reactants the prediction files as conditions count as found, and a
    catalyst read as a reactant still satisfies the hard rule.
    """
    pr, gr, pc, gc = Counter(pred_r), Counter(gt_r), Counter(pred_c), Counter(gt_c)
    forgiven_pred = (pr - gr) & gc
    forgiven_gt = (gr - pr) & pc
    return (list((pr - forgiven_pred).elements()), list((gr - forgiven_gt).elements()),
            list((pc + forgiven_pred).elements()))


def count_matches(pred_rxns: List[dict], gt_rxns: List[dict], use_conditions: bool) -> int:
    """Reactions matched one-to-one (optimal assignment over the candidate pairs)."""
    n, m = len(pred_rxns), len(gt_rxns)
    if n == 0 or m == 0:
        return 0
    pred_r = [extract_reactant_smiles(r) for r in pred_rxns]
    pred_p = [extract_product_smiles(r) for r in pred_rxns]
    pred_c = [extract_condition_smiles(r) for r in pred_rxns]
    gt_r = [extract_reactant_smiles(r) for r in gt_rxns]
    gt_p = [extract_product_smiles(r) for r in gt_rxns]
    gt_c = [extract_condition_smiles(r) for r in gt_rxns]
    gt_drawn = [drawn_condition_smiles(r) for r in gt_rxns]

    size = max(n, m)
    cost = np.ones((size, size), dtype=float)
    for i in range(n):
        for j in range(m):
            pr, gr, pc = _reconcile_roles(pred_r[i], pred_c[i], gt_r[j], gt_c[j])
            ok = Counter(pr) == Counter(gr) and Counter(pred_p[i]) == Counter(gt_p[j])
            if ok and use_conditions and gt_drawn[j]:
                # every drawn GT condition has to appear among the predicted ones;
                # extra predicted conditions do not hurt
                pcc = Counter(pc)
                ok = all(pcc.get(smi, 0) >= cnt for smi, cnt in Counter(gt_drawn[j]).items())
            if ok:
                cost[i, j] = 0.0
    row, col = linear_sum_assignment(cost)
    return int(sum(1 for i, j in zip(row, col) if i < n and j < m and cost[i, j] == 0.0))


def aggregate_prf(tp: int, n_pred: int, n_gt: int) -> Dict[str, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt if n_gt else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "n_pred": n_pred, "n_gt": n_gt}


# =========================================================================== #
#  GED                                                                         #
# =========================================================================== #
def _mol_size(mol: Chem.Mol) -> int:
    return mol.GetNumAtoms() + mol.GetNumBonds()


@functools.lru_cache(maxsize=500_000)
def _size_of(smi: str) -> int:
    mol = Chem.MolFromSmiles(smi)
    return _mol_size(mol) if mol is not None else 0


@functools.lru_cache(maxsize=2_000_000)
def _pair_ged_cached(key: Tuple[str, str]) -> int:
    a, b = key
    mol_a, mol_b = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    if mol_a is None and mol_b is None:
        return 0
    if mol_a is None:
        return _mol_size(mol_b)
    if mol_b is None:
        return _mol_size(mol_a)
    try:
        mcs = rdFMCS.FindMCS([mol_a, mol_b], timeout=MCS_TIMEOUT, matchValences=False,
                             ringMatchesRingOnly=False, completeRingsOnly=False)
    except Exception:
        return _mol_size(mol_a) + _mol_size(mol_b)
    mcs_size = (mcs.numAtoms or 0) + (mcs.numBonds or 0)
    return max(0, _mol_size(mol_a) + _mol_size(mol_b) - 2 * mcs_size)


def _pair_ged(smi_a: str, smi_b: str) -> int:
    """MCS-based graph edit distance between two canonical SMILES.

    The same molecule pair (a shared catalyst, a shared substrate) recurs in
    nearly every reaction of a figure, and MCS is symmetric, so the result is
    cached under an order-normalised key.
    """
    if smi_a == smi_b:
        return 0
    return _pair_ged_cached(tuple(sorted((smi_a, smi_b))))


def molecule_set_ged(pred_smiles: Sequence[str], gt_smiles: Sequence[str]) -> float:
    """Optimal-assignment GED between two unordered molecule sets.

    An unmatched molecule contributes its full topological size (|V| + |E|).
    """
    pred_smiles, gt_smiles = list(pred_smiles), list(gt_smiles)
    n, m = len(pred_smiles), len(gt_smiles)
    if n == 0 and m == 0:
        return 0.0
    if n == 0:
        return float(sum(_size_of(s) for s in gt_smiles))
    if m == 0:
        return float(sum(_size_of(s) for s in pred_smiles))
    size = max(n, m)
    cost = np.zeros((size, size), dtype=float)
    pred_sizes = [_size_of(s) for s in pred_smiles]
    gt_sizes = [_size_of(s) for s in gt_smiles]
    for i in range(size):
        for j in range(size):
            if i < n and j < m:
                cost[i, j] = _pair_ged(pred_smiles[i], gt_smiles[j])
            elif i < n:
                cost[i, j] = pred_sizes[i]
            elif j < m:
                cost[i, j] = gt_sizes[j]
    row, col = linear_sum_assignment(cost)
    return float(cost[row, col].sum())


def _reaction_pair_cost(pred_rxn: dict, gt_rxn: dict) -> float:
    return molecule_set_ged(extract_all_smiles(pred_rxn), extract_all_smiles(gt_rxn))


def _reaction_self_cost(rxn: dict) -> float:
    return float(sum(_size_of(s) for s in extract_all_smiles(rxn)))


def align_reactions(pred_rxns: List[dict],
                    gt_rxns: List[dict]) -> List[Tuple[Optional[int], Optional[int], float]]:
    """Hungarian alignment of predicted to GT reactions, by GED.

    Returns (pred_idx, gt_idx, cost) for every matched and unmatched reaction;
    the unmatched side is None.
    """
    n, m = len(pred_rxns), len(gt_rxns)
    if n == 0 and m == 0:
        return []
    size = max(n, m, 1)
    cost = np.zeros((size, size), dtype=float)
    pred_self = [_reaction_self_cost(r) for r in pred_rxns]
    gt_self = [_reaction_self_cost(r) for r in gt_rxns]
    for i in range(size):
        for j in range(size):
            if i < n and j < m:
                cost[i, j] = _reaction_pair_cost(pred_rxns[i], gt_rxns[j])
            elif i < n:
                cost[i, j] = pred_self[i]
            elif j < m:
                cost[i, j] = gt_self[j]
    row, col = linear_sum_assignment(cost)
    out: List[Tuple[Optional[int], Optional[int], float]] = []
    for i, j in zip(row, col):
        pi = i if i < n else None
        gj = j if j < m else None
        if pi is None and gj is None:
            continue
        out.append((pi, gj, float(cost[i, j])))
    return out


# =========================================================================== #
#  Scoring                                                                     #
# =========================================================================== #
def get_reactions(sample: Any) -> List[dict]:
    if isinstance(sample, dict):
        rxns = sample.get("reactions")
        if isinstance(rxns, list):
            return [r for r in rxns if isinstance(r, dict)]
    if isinstance(sample, list):
        return [r for r in sample if isinstance(r, dict)]
    return []


def score(preds: Dict[str, Any], gts: Dict[str, dict], subset: str = "",
          only_predicted: bool = False, skip_ged: bool = False) -> Dict[str, Any]:
    """Corpus scores over every GT image (or one subset of them)."""
    names = sorted(gts)
    if subset:
        names = [n for n in names if (gts[n] or {}).get("subset") == subset]
    if only_predicted:
        # Pilot mode only. Never use for a full-run comparison: it lets an arm
        # silently drop the images it finds hard and still look good.
        names = [n for n in names if n in preds]

    soft_tp = hard_tp = n_pred = n_gt = n_missing = 0
    total_ged = 0.0
    ged_slots = 0

    for fn in names:
        gt_rxns = get_reactions(gts[fn])
        p = preds.get(fn)
        if p is None:
            n_missing += 1
            pred_rxns: List[dict] = []
        else:
            pred_rxns = get_reactions(p)

        soft_tp += count_matches(pred_rxns, gt_rxns, use_conditions=False)
        hard_tp += count_matches(pred_rxns, gt_rxns, use_conditions=True)
        n_pred += len(pred_rxns)
        n_gt += len(gt_rxns)

        if not skip_ged:
            total_ged += sum(c for _, _, c in align_reactions(pred_rxns, gt_rxns))
        ged_slots += max(len(pred_rxns), len(gt_rxns), 1)

    return {
        "n_graphics": len(names),
        "n_missing_predictions": n_missing,
        "n_gt_reactions": n_gt,
        "n_pred_reactions": n_pred,
        "soft": aggregate_prf(soft_tp, n_pred, n_gt),
        "hard": aggregate_prf(hard_tp, n_pred, n_gt),
        "avg_ged_per_reaction": (total_ged / ged_slots) if ged_slots else 0.0,
        "ged_skipped": skip_ged,
    }


def per_image_scores(preds: Dict[str, Any], gts: Dict[str, dict],
                     only_predicted: bool = False, skip_ged: bool = True) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for fn in sorted(gts):
        if only_predicted and fn not in preds:
            continue
        r = score({fn: preds[fn]} if fn in preds else {}, {fn: gts[fn]}, skip_ged=skip_ged)
        out[fn] = {
            "subset": (gts[fn] or {}).get("subset"),
            "n_gt": r["n_gt_reactions"], "n_pred": r["n_pred_reactions"],
            "soft_tp": r["soft"]["tp"], "hard_tp": r["hard"]["tp"],
            "soft_f1": r["soft"]["f1"], "hard_f1": r["hard"]["f1"],
            "ged": r["avg_ged_per_reaction"],
        }
    return out


def evaluate(pred_path: str, gt_path, skip_ged: bool = False) -> Dict[str, Any]:
    """Score one predictions file against one or more GT files."""
    preds = load_samples(pred_path)
    gts = load_ground_truth([gt_path] if isinstance(gt_path, str) else list(gt_path))
    report = score(preds, gts, skip_ged=skip_ged)
    report["per_image"] = per_image_scores(preds, gts)
    report["predictions_without_gt"] = sorted(set(preds) - set(gts))
    return report


# =========================================================================== #
#  CLI                                                                         #
# =========================================================================== #
_HDR = (f"{'arm':16s} {'gfx':>4s} {'#pred':>6s} "
        f"{'sP':>7s} {'sR':>7s} {'sF1':>7s} {'hP':>7s} {'hR':>7s} {'hF1':>7s} "
        f"{'GED':>8s} {'miss':>5s}")


def _fmt(tag: str, r: Dict[str, Any]) -> str:
    s, h = r["soft"], r["hard"]
    return (f"{tag:16s} {r['n_graphics']:4d} {r['n_pred_reactions']:6d} "
            f"{s['precision']*100:7.2f} {s['recall']*100:7.2f} {s['f1']*100:7.2f} "
            f"{h['precision']*100:7.2f} {h['recall']*100:7.2f} {h['f1']*100:7.2f} "
            f"{r['avg_ged_per_reaction']:8.2f} {r['n_missing_predictions']:5d}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ChemEagle benchmark scorer (2026-09-13 rules).")
    ap.add_argument("--pred", action="append", required=True,
                    help="predictions JSON keyed by file_name; repeat to compare several arms")
    ap.add_argument("--tag", action="append", default=[], help="name for each --pred")
    ap.add_argument("--gt", action="append", required=True,
                    help="ground truth JSON; repeat to merge GT1..GT4")
    ap.add_argument("--skip-ged", action="store_true", help="skip the slow GED alignment (GED reads 0)")
    ap.add_argument("--by-subset", action="store_true", help="also break the scores down by GT subset")
    ap.add_argument("--only-predicted", action="store_true",
                    help="pilot mode: score only the images the arm attempted")
    ap.add_argument("--per-image", default="", help="write per-image scores to this JSON file")
    ap.add_argument("--out", default="", help="write the full report to this JSON file")
    args = ap.parse_args(argv)

    for p in list(args.pred) + list(args.gt):
        if not os.path.exists(p):
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            return 2
    gts = load_ground_truth(args.gt)
    tags = args.tag or [os.path.splitext(os.path.basename(p))[0] for p in args.pred]

    print(f"GT: {len(gts)} graphics, {sum(len(get_reactions(g)) for g in gts.values())} reactions")
    print(_HDR)
    print("-" * len(_HDR))

    all_results: Dict[str, Any] = {}
    for path, tag in zip(args.pred, tags):
        preds = load_samples(path)
        r = score(preds, gts, only_predicted=args.only_predicted, skip_ged=args.skip_ged)
        all_results[tag] = {"overall": r}
        print(_fmt(tag, r))
        if args.by_subset:
            for sub in sorted({(g or {}).get("subset") for g in gts.values()} - {None}):
                rs = score(preds, gts, subset=sub, only_predicted=args.only_predicted,
                           skip_ged=args.skip_ged)
                all_results[tag][sub] = rs
                print(_fmt(f"  {sub}", rs))
        if args.per_image:
            per = per_image_scores(preds, gts, only_predicted=args.only_predicted,
                                   skip_ged=args.skip_ged)
            all_results[tag]["per_image"] = per
            with open(args.per_image, "w", encoding="utf-8") as f:
                json.dump({tag: per}, f, ensure_ascii=False, indent=1)
            print(f"wrote {args.per_image}")
        print()

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

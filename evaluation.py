from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdFMCS
from scipy.optimize import linear_sum_assignment

RDLogger.DisableLog("rdApp.*")
_STEREO = True
_STRIP_SALTS = True
_THRESHOLD = 0.5

def load_samples(path: str) -> Dict[str, dict]:
    """Load a predictions / ground-truth JSON file into a {id: sample} dict."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return {str(k): v for k, v in data.items()}
    if isinstance(data, list):
        out: Dict[str, dict] = {}
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            sid = (
                item.get("id")
                or item.get("image_id")
                or item.get("image_path")
                or f"sample_{i}"
            )
            out[str(sid)] = item
        return out
    raise ValueError(f"Unsupported JSON top-level type in {path}: {type(data)}")


# --------------------------------------------------------------------------- #
# SMILES handling                                                              #
# --------------------------------------------------------------------------- #
_INVALID_TOKENS = {"", "none", "null", "n/a", "na", "?", "*"}


def canon_smiles(smi: Any) -> Optional[str]:
    """Canonicalize a SMILES string.  Returns None for invalid / blank input.

    When ``_STEREO`` is True, stereochemistry (chiral tags, double-bond
    E/Z, atom parity) is stripped before canonicalization so that e.g.
    ``C[C@H](O)Cl`` and ``CC(O)Cl`` count as the same molecule.

    When ``_STRIP_SALTS`` is True, multi-component SMILES (joined by ``.``) are
    reduced to their largest fragment by heavy-atom count, so a prediction that
    omits free counter-ions / cocrystal solvents still matches the GT main
    molecule.
    """
    if smi is None:
        return None
    if not isinstance(smi, str):
        smi = str(smi)
    s = smi.strip()
    if not s or s.lower() in _INVALID_TOKENS:
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    try:
        if _STRIP_SALTS:
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            if len(frags) > 1:
                mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        if _STEREO:
            Chem.RemoveStereochemistry(mol)
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _iter_smiles_from_list(items: Any) -> Iterable[str]:
    """Yield raw smiles strings from a list-of-dicts (or list-of-strings)."""
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


_CONDITION_SMILES_ROLES = {"reagent", "reagents", "solvent", "solvents",
                           "catalyst", "catalysts"}


def extract_condition_smiles(rxn: dict) -> List[str]:
    """Pick the canonicalized SMILES from condition entries with a chemical role."""
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


def extract_all_smiles(rxn: dict) -> List[str]:
    return (
        extract_reactant_smiles(rxn)
        + extract_product_smiles(rxn)
        + extract_condition_smiles(rxn)
    )


# --------------------------------------------------------------------------- #
# Soft / hard match                                                            #
# --------------------------------------------------------------------------- #
def _multiset(xs: Iterable[str]) -> Tuple[Tuple[str, int], ...]:
    """A hashable multiset signature (sorted list of (smi, count))."""
    counts: Dict[str, int] = defaultdict(int)
    for x in xs:
        counts[x] += 1
    return tuple(sorted(counts.items()))


def soft_signature(rxn: dict) -> Tuple[Any, Any]:
    return (_multiset(extract_reactant_smiles(rxn)),
            _multiset(extract_product_smiles(rxn)))


def hard_signature(rxn: dict) -> Tuple[Any, Any, Any]:
    return (_multiset(extract_reactant_smiles(rxn)),
            _multiset(extract_product_smiles(rxn)),
            _multiset(extract_condition_smiles(rxn)))


def _count_matches(pred_signatures: List[Any], gt_signatures: List[Any]) -> int:
    """One-to-one match count between two multisets of signatures."""
    pred_counts: Dict[Any, int] = defaultdict(int)
    for sig in pred_signatures:
        pred_counts[sig] += 1
    matched = 0
    for sig in gt_signatures:
        if pred_counts.get(sig, 0) > 0:
            pred_counts[sig] -= 1
            matched += 1
    return matched


def _multiset_jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Jaccard overlap on multisets (sum(min) / sum(max))."""
    ca: Dict[str, int] = defaultdict(int)
    cb: Dict[str, int] = defaultdict(int)
    for x in a:
        ca[x] += 1
    for x in b:
        cb[x] += 1
    keys = set(ca) | set(cb)
    if not keys:
        return 1.0
    inter = sum(min(ca[k], cb[k]) for k in keys)
    union = sum(max(ca[k], cb[k]) for k in keys)
    return inter / union if union else 1.0


def _count_threshold_matches(
    pred_rxns: List[dict],
    gt_rxns: List[dict],
    threshold: float,
    use_conditions: bool,
) -> int:
    """Hungarian-based reaction matching using Jaccard overlap >= threshold."""
    n, m = len(pred_rxns), len(gt_rxns)
    if n == 0 or m == 0:
        return 0
    pred_r = [extract_reactant_smiles(r) for r in pred_rxns]
    pred_p = [extract_product_smiles(r) for r in pred_rxns]
    pred_c = [extract_condition_smiles(r) for r in pred_rxns]
    gt_r = [extract_reactant_smiles(r) for r in gt_rxns]
    gt_p = [extract_product_smiles(r) for r in gt_rxns]
    gt_c = [extract_condition_smiles(r) for r in gt_rxns]

    size = max(n, m)
    cost = np.ones((size, size), dtype=float)
    for i in range(n):
        for j in range(m):
            jr = _multiset_jaccard(pred_r[i], gt_r[j])
            jp = _multiset_jaccard(pred_p[i], gt_p[j])
            ok = jr >= threshold and jp >= threshold
            if ok and use_conditions:
                jc = _multiset_jaccard(pred_c[i], gt_c[j])
                ok = jc >= threshold
            if ok:
                cost[i, j] = 0.0
    row, col = linear_sum_assignment(cost)
    return int(sum(1 for i, j in zip(row, col)
                   if i < n and j < m and cost[i, j] == 0.0))


def aggregate_prf(tp: int, n_pred: int, n_gt: int) -> Dict[str, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt if n_gt else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f,
            "tp": tp, "n_pred": n_pred, "n_gt": n_gt}


# --------------------------------------------------------------------------- #
# GED                                                                          #
# --------------------------------------------------------------------------- #
def _mol_size(mol: Chem.Mol) -> int:
    return mol.GetNumAtoms() + mol.GetNumBonds()


def _pair_ged(smi_a: str, smi_b: str,
              mcs_timeout: int = 5) -> int:
    """MCS-based GED between two canonical SMILES strings."""
    if smi_a == smi_b:
        return 0
    mol_a = Chem.MolFromSmiles(smi_a)
    mol_b = Chem.MolFromSmiles(smi_b)
    if mol_a is None and mol_b is None:
        return 0
    if mol_a is None:
        return _mol_size(mol_b)
    if mol_b is None:
        return _mol_size(mol_a)
    try:
        mcs = rdFMCS.FindMCS(
            [mol_a, mol_b],
            timeout=mcs_timeout,
            matchValences=False,
            ringMatchesRingOnly=False,
            completeRingsOnly=False,
        )
    except Exception:
        return _mol_size(mol_a) + _mol_size(mol_b)
    mcs_size = (mcs.numAtoms or 0) + (mcs.numBonds or 0)
    return max(0, _mol_size(mol_a) + _mol_size(mol_b) - 2 * mcs_size)


def molecule_set_ged(pred_smiles: Sequence[str],
                     gt_smiles: Sequence[str]) -> float:
    """
    Optimal-assignment GED between two unordered sets of canonical SMILES.

    Unmatched molecules contribute their full topological size (|V| + |E|).
    """
    pred_smiles = list(pred_smiles)
    gt_smiles = list(gt_smiles)
    n, m = len(pred_smiles), len(gt_smiles)
    if n == 0 and m == 0:
        return 0.0
    if n == 0:
        return float(sum(_size_of(s) for s in gt_smiles))
    if m == 0:
        return float(sum(_size_of(s) for s in pred_smiles))

    # Build cost matrix with size = max(n, m); pad with size-of-unmatched cost.
    size = max(n, m)
    cost = np.zeros((size, size), dtype=float)
    pred_sizes = [_size_of(s) for s in pred_smiles]
    gt_sizes = [_size_of(s) for s in gt_smiles]
    for i in range(size):
        for j in range(size):
            if i < n and j < m:
                cost[i, j] = _pair_ged(pred_smiles[i], gt_smiles[j])
            elif i < n:                  # predicted i unmatched
                cost[i, j] = pred_sizes[i]
            elif j < m:                  # gt j unmatched
                cost[i, j] = gt_sizes[j]
            else:
                cost[i, j] = 0.0
    row, col = linear_sum_assignment(cost)
    return float(cost[row, col].sum())


def _size_of(smi: str) -> int:
    mol = Chem.MolFromSmiles(smi)
    return _mol_size(mol) if mol is not None else 0


# --------------------------------------------------------------------------- #
# Reaction-level alignment                                                     #
# --------------------------------------------------------------------------- #
def _reaction_pair_cost(pred_rxn: dict, gt_rxn: dict) -> float:
    """Combined GED cost between two reactions (used for reaction alignment)."""
    return molecule_set_ged(
        extract_all_smiles(pred_rxn),
        extract_all_smiles(gt_rxn),
    )


def _reaction_self_cost(rxn: dict) -> float:
    """Cost of leaving a reaction unmatched: total |V|+|E| of its molecules."""
    return float(sum(_size_of(s) for s in extract_all_smiles(rxn)))


def align_reactions(pred_rxns: List[dict],
                    gt_rxns: List[dict]) -> List[Tuple[Optional[int], Optional[int], float]]:
    """
    Hungarian alignment between predicted and GT reactions.

    Returns a list of (pred_idx, gt_idx, ged_cost) entries covering every
    matched and unmatched reaction.  pred_idx / gt_idx are None for the
    unmatched side.
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
            else:
                cost[i, j] = 0.0
    row, col = linear_sum_assignment(cost)
    out: List[Tuple[Optional[int], Optional[int], float]] = []
    for i, j in zip(row, col):
        pi = i if i < n else None
        gj = j if j < m else None
        if pi is None and gj is None:
            continue
        out.append((pi, gj, float(cost[i, j])))
    return out


# --------------------------------------------------------------------------- #
# Per-sample / corpus evaluation                                               #
# --------------------------------------------------------------------------- #
def get_reactions(sample: Any) -> List[dict]:
    if isinstance(sample, dict):
        rxns = sample.get("reactions")
        if isinstance(rxns, list):
            return [r for r in rxns if isinstance(r, dict)]
    if isinstance(sample, list):
        return [r for r in sample if isinstance(r, dict)]
    return []


def evaluate(pred_path: str, gt_path: str) -> Dict[str, Any]:
    preds = load_samples(pred_path)
    gts = load_samples(gt_path)

    common_ids = sorted(set(preds.keys()) & set(gts.keys()))
    only_pred = sorted(set(preds.keys()) - set(gts.keys()))
    only_gt = sorted(set(gts.keys()) - set(preds.keys()))

    soft_tp = soft_pred = soft_gt = 0
    hard_tp = hard_pred = hard_gt = 0

    total_ged = 0.0
    total_reactions_for_ged = 0
    per_sample: Dict[str, Dict[str, Any]] = {}

    for sid in common_ids:
        pred_rxns = get_reactions(preds[sid])
        gt_rxns = get_reactions(gts[sid])

        pred_soft = [soft_signature(r) for r in pred_rxns]
        gt_soft = [soft_signature(r) for r in gt_rxns]
        pred_hard = [hard_signature(r) for r in pred_rxns]
        gt_hard = [hard_signature(r) for r in gt_rxns]

        s_tp = _count_threshold_matches(pred_rxns, gt_rxns,
                                        _THRESHOLD,
                                        use_conditions=False)
        h_tp = _count_threshold_matches(pred_rxns, gt_rxns,
                                        _THRESHOLD,
                                        use_conditions=True)

        soft_tp += s_tp
        hard_tp += h_tp
        soft_pred += len(pred_soft)
        hard_pred += len(pred_hard)
        soft_gt += len(gt_soft)
        hard_gt += len(gt_hard)

        alignment = align_reactions(pred_rxns, gt_rxns)
        sample_ged_sum = sum(cost for _, _, cost in alignment)
        n_slots = max(len(pred_rxns), len(gt_rxns), 1)
        total_ged += sample_ged_sum
        total_reactions_for_ged += n_slots

        per_sample[sid] = {
            "n_pred": len(pred_rxns),
            "n_gt": len(gt_rxns),
            "soft_tp": s_tp,
            "hard_tp": h_tp,
            "ged_sum": sample_ged_sum,
            "ged_per_reaction": sample_ged_sum / n_slots,
        }

    soft = aggregate_prf(soft_tp, soft_pred, soft_gt)
    hard = aggregate_prf(hard_tp, hard_pred, hard_gt)
    avg_ged_per_reaction = (
        total_ged / total_reactions_for_ged if total_reactions_for_ged else 0.0
    )

    return {
        "n_samples_evaluated": len(common_ids),
        "n_samples_pred_only": len(only_pred),
        "n_samples_gt_only": len(only_gt),
        "soft_match": soft,
        "hard_match": hard,
        "ged": {
            "avg_per_reaction": avg_ged_per_reaction,
            "total_cost": total_ged,
            "n_reaction_slots": total_reactions_for_ged,
        },
        "per_sample": per_sample,
        "missing_in_gt": only_pred,
        "missing_in_pred": only_gt,
    }


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _print_block(title: str, m: Dict[str, float]) -> None:
    print(title)
    print(f"  precision = {m['precision']:.4f}")
    print(f"  recall    = {m['recall']:.4f}")
    print(f"  F1        = {m['f1']:.4f}")
    print(f"  TP/Pred/GT = {m['tp']}/{m['n_pred']}/{m['n_gt']}")


def _print_summary(report: Dict[str, Any]) -> None:
    settings = report.get("settings", {})
    ged = report["ged"]
    print("=" * 60)
    print(f"Samples evaluated : {report['n_samples_evaluated']}")
    print(f"Pred-only ids     : {report['n_samples_pred_only']}")
    print(f"GT-only ids       : {report['n_samples_gt_only']}")
    print("-" * 60)
    _print_block("Soft match  (reactants + products)", report["soft_match"])
    _print_block("Hard match  (+ condition SMILES)", report["hard_match"])
    print("-" * 60)
    print("GED")
    print(f"  avg per reaction = {ged['avg_per_reaction']:.4f}")
    print(f"  total cost       = {ged['total_cost']:.2f}  "
          f"over {ged['n_reaction_slots']} reaction slots")
    print("=" * 60)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ChemEAGLE evaluation script.")
    parser.add_argument("--pred", required=True, help="Path to predictions JSON.")
    parser.add_argument("--gt", required=True, help="Path to ground truth JSON.")
    parser.add_argument("--out", default=None,
                        help="Optional path to write the full JSON report.")
    args = parser.parse_args(argv)

    if not os.path.exists(args.pred):
        print(f"ERROR: predictions file not found: {args.pred}", file=sys.stderr)
        return 2
    if not os.path.exists(args.gt):
        print(f"ERROR: ground truth file not found: {args.gt}", file=sys.stderr)
        return 2

    report = evaluate(args.pred, args.gt)
    _print_summary(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nFull report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

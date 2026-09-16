"""Cross-check the two vision passes ChemEAGLE runs on the same image.

The reaction agent (RxnIM crop -> MolNexTR) and the molecular recognition agent
(MolDetector crop -> MolNexTR) each produce a graph for the same drawn molecule
and never compare notes. When one graph yields an RDKit-invalid SMILES and the
other a valid one for the same box (IoU >= threshold), the invalid graph is
replaced by the valid one. Pure functions on box dicts; no model calls.
"""
import copy

GRAPH_KEYS = ('smiles', 'symbols', 'coords', 'edges', 'molfile', 'atoms', 'bonds')


def smiles_valid(smiles):
    """True/False by RDKit; None when RDKit is unavailable or the value is not a string."""
    if not isinstance(smiles, str) or not smiles:
        return False
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
    except Exception:
        return None
    return Chem.MolFromSmiles(smiles) is not None


def iou(a, b):
    if not (a and b and len(a) == 4 and len(b) == 4):
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def is_derived(box):
    """A box the edit plan cloned from a template (an expanded variant): it shares the parent's
    bbox but is a different molecule, so it is never a donor and never repaired from the parent."""
    return bool(box.get('compound_id') or box.get('derived') or box.get('bbox_provenance') == 'parent_molecule')


def _copy_graph(src, dst):
    for k in GRAPH_KEYS:
        if k in src:
            dst[k] = copy.deepcopy(src[k])
        elif k in dst:
            del dst[k]


def _best(candidates, bbox, threshold):
    best, best_iou = None, 0.0
    for c in candidates:
        v = iou(list(c['bbox']), list(bbox))
        if v > best_iou:
            best, best_iou = c, v
    return (best, best_iou) if best is not None and best_iou >= threshold else (None, 0.0)


def adopt(reaction_results, vision_boxes, threshold=0.5):
    """Make the molecular agent's boxes the reaction's molecules.

    Every molecule entry of the reaction (reactants / products / drawn conditions, entries with a
    graph and a bbox) takes over the graph AND the bbox of the molecular agent's box with the
    largest overlap (IoU >= threshold): the MolDetector box is the one canonical box of a drawn
    molecule, and its graph already carries the edit plan's OCR corrections and definitions. The
    RxnIM box is kept as ``rxnim_bbox``. Entries without an overlapping box keep their own graph.
    Returns audit records (one per entry, matched or not)."""
    audit = []
    donors = [b for b in vision_boxes or [] if b.get('category') == '[Mol]' and b.get('bbox') and 'symbols' in b]
    for rx in reaction_results or []:
        for section in ('reactants', 'products', 'conditions'):
            for entry in rx.get(section, []) or []:
                if not (isinstance(entry, dict) and entry.get('bbox') and 'symbols' in entry):
                    continue
                best, overlap = _best(donors, entry['bbox'], threshold)
                rec = {'operation': 'reaction_molecule_from_detector_box', 'section': section,
                       'rxnim_bbox': list(entry['bbox']), 'old_smiles': entry.get('smiles')}
                if best is None:
                    rec.update(matched=False)
                    audit.append(rec)
                    continue
                entry['rxnim_bbox'] = list(entry['bbox'])
                entry['bbox'] = list(best['bbox'])
                _copy_graph(best, entry)
                # The plan's symbol corrections were replayed on `symbols` only; the R-group back-out reads
                # atoms[i]['atom_symbol'], so keep the two in step.
                atoms = entry.get('atoms')
                if isinstance(atoms, list) and len(atoms) == len(entry.get('symbols') or []):
                    for atom, sym in zip(atoms, entry['symbols']):
                        if isinstance(atom, dict):
                            atom['atom_symbol'] = sym
                rec.update(matched=True, iou=round(overlap, 3), bbox=list(best['bbox']), new_smiles=entry.get('smiles'))
                audit.append(rec)
    return audit


def reconcile(reaction_results, vision_boxes, mol_boxes=None, threshold=0.7, prefer_final=False):
    """Repair invalid molecule graphs by borrowing the other pass's graph.

    reaction_results: the reaction agent's list of reactions (reactants / products /
        conditions entries with bbox + graph fields); repaired in place.
    vision_boxes: the molecular agent's vision boxes, ideally with the plan's symbol corrections
        replayed (generic templates as read from the image): first-tier donors for the reaction side.
    mol_boxes: the molecular agent's final boxes (after LLM edits) when available. Valid
        non-derived final boxes with a bbox of their own (a box sharing its bbox with another final
        box is an expanded variant) are second-tier donors (used only when no first-tier box overlaps);
        an invalid non-derived final box is repaired from the reaction side.
    prefer_final: put the final boxes in the first tier (free-form mode, where the LLM's rewrite
        of the boxes is the only OCR correction and there is no plan audit to replay).
    Returns a list of audit records.
    """
    audit = []
    all_final = [b for b in mol_boxes or [] if b.get('category') == '[Mol]' and b.get('bbox')]

    def cloned(b):      # expanded variants share the parent's bbox; the free-form agent marks them in no other way
        return any(o is not b and not is_derived(o) and iou(list(o['bbox']), list(b['bbox'])) > 0.95 for o in all_final)

    final_boxes = [b for b in all_final if not is_derived(b) and not cloned(b)]
    tiers = [[b for b in vision_boxes or [] if b.get('category') == '[Mol]' and b.get('bbox') and smiles_valid(b.get('smiles'))],
             [b for b in final_boxes if smiles_valid(b.get('smiles'))]]
    if prefer_final:
        tiers.reverse()
    reaction_entries = [(sec, e) for rx in reaction_results or [] for sec in ('reactants', 'products', 'conditions')
                        for e in (rx.get(sec, []) or []) if isinstance(e, dict) and 'smiles' in e and e.get('bbox')]
    for section, entry in reaction_entries:
        if smiles_valid(entry.get('smiles')) is not False:
            continue
        best, overlap = None, 0.0
        for tier in tiers:
            best, overlap = _best(tier, entry['bbox'], threshold)
            if best is not None:
                break
        if best is None:
            continue
        old = entry.get('smiles')
        _copy_graph(best, entry)
        audit.append({'operation': 'reaction_graph_from_molecular_agent', 'section': section, 'bbox': list(entry['bbox']),
                      'iou': round(overlap, 3), 'old_smiles': old, 'new_smiles': entry.get('smiles')})
    if mol_boxes:
        valid_entries = [e for _, e in reaction_entries if smiles_valid(e.get('smiles'))]
        for box in final_boxes:
            if smiles_valid(box.get('smiles')) is not False:
                continue
            best, overlap = _best(valid_entries, box['bbox'], threshold)
            if best is None:
                continue
            old = box.get('smiles')
            _copy_graph(best, box)
            audit.append({'operation': 'molecular_graph_from_reaction_agent', 'bbox': list(box['bbox']),
                          'iou': round(overlap, 3), 'old_smiles': old, 'new_smiles': box.get('smiles')})
    return audit

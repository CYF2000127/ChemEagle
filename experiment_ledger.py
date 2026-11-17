"""Evidence-preserving assembly for the hybrid pipeline. No model/runtime imports."""
import copy
import hashlib
import json
from pathlib import Path
from uuid import uuid4

VERSION = 'entry-ledger-v1'
HIDDEN = {'coords', 'edges', 'atoms', 'bonds', 'molfile'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def compact(value):
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items() if k not in HIDDEN}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


class Checkpoints:
    def __init__(self, directory, source, settings):
        self.manifest = {'version': VERSION, 'source_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                         'settings': settings}
        self.root = Path(directory) / digest(self.manifest) / uuid4().hex
        self.root.mkdir(parents=True, exist_ok=True)
        self.save('manifest', self.manifest)

    def save(self, name, value):
        path = self.root / (name + '.json')
        data = json.dumps(value, ensure_ascii=False, indent=2, default=str)
        temp = path.with_suffix('.tmp')
        temp.write_text(data)
        temp.replace(path)
        return {'path': str(path), 'sha256': hashlib.sha256(data.encode()).hexdigest()}


class Ledger:
    def __init__(self, paper_id, region):
        self.paper_id, self.region = paper_id, region
        self.entries = {}
        self.history = []

    def inventory(self, observations, method):
        for obs in observations:
            if not isinstance(obs, dict) or not str(obs.get('entry_label') or '').strip():
                continue
            key = [self.paper_id, obs.get('source_reference') or self.region,
                   str(obs['entry_label']).strip(), str(obs.get('outcome_label') or 'reported')]
            identifier = 'exp-' + digest(key)[:24]
            item = self.entries.setdefault(identifier, {
                'experiment_id': identifier, 'paper_id': self.paper_id, 'source_reference': key[1],
                'entry_label': key[2], 'outcome_label': key[3], 'observations': [],
                'disposition': 'awaiting_review', 'record': None, 'attempts': [],
            })
            observation = {'method': method, **copy.deepcopy(obs)}
            if observation not in item['observations']:
                item['observations'].append(observation)
        return self

    def assemble(self, candidates, only_missing=False):
        unmatched = []
        for record in candidates:
            if not isinstance(record, dict):
                continue
            label = str(record.get('entry_label') or '').strip()
            labels = {str(c.get('label') or '').strip() for c in record.get('products', []) if isinstance(c, dict)}
            if label:
                labels = {label}
            matches = [e for e in self.entries.values()
                       if (record.get('experiment_id') == e['experiment_id'] or e['entry_label'] in labels)
                       and (not record.get('source_reference') or record['source_reference'] == e['source_reference'])
                       and (not record.get('outcome_label') or str(record['outcome_label']) == e['outcome_label'])]
            if len(matches) != 1:
                unmatched.append(copy.deepcopy(record))
                continue
            entry = matches[0]
            if entry['record'] is not None and only_missing:
                continue
            if entry['record'] is not None:
                entry.setdefault('alternatives', []).append(copy.deepcopy(record))
                entry['disposition'] = 'awaiting_review'
                continue
            entry['record'] = copy.deepcopy(record)
            entry['disposition'] = 'extracted_unresolved'
        return unmatched

    def attach_recognized(self, structures):
        # Recognition alone is not proof that a compound is a measured product or an experiment.
        for label, smiles in structures.items():
            matching = [e for e in self.entries.values() if e['entry_label'] == label]
            if not matching:
                self.inventory([{'entry_label': label, 'evidence': [], 'recognized_smiles': smiles}], 'recognizer')
                matching = [e for e in self.entries.values() if e['entry_label'] == label]
            for entry in matching:
                entry.setdefault('recognized_candidates', []).append({'label': label, 'smiles': smiles})

    def recover(self, callback, total_budget=4, checkpoint=None):
        used = 0
        for entry in self.entries.values():
            if entry['record'] is not None or entry['disposition'] == 'excluded_with_evidence':
                continue
            for method in ('region_reconstruction', 'definition_and_recognizer_review'):
                if used >= total_budget or len(entry['attempts']) >= 2:
                    break
                if any(a['method'] == method for a in entry['attempts']):
                    continue
                used += 1
                before = copy.deepcopy(entry)
                try:
                    proposal = callback(copy.deepcopy(entry), method)
                except Exception as exc:
                    proposal = {'error_type': type(exc).__name__}
                accepted = False
                if isinstance(proposal, dict) and proposal.get('experiment_id') == entry['experiment_id']:
                    evidence = proposal.get('evidence')
                    # A nonempty source quotation or located source region is required.
                    grounded = isinstance(evidence, list) and any(
                        isinstance(e, dict) and isinstance(e.get('bbox'), list)
                        and len(e['bbox']) == 4
                        and all(type(n) in (int, float) for n in e['bbox'])
                        and 0 <= e['bbox'][0] < e['bbox'][2] <= 1
                        and 0 <= e['bbox'][1] < e['bbox'][3] <= 1 for e in evidence)
                    if grounded and proposal.get('reason'):
                        if proposal.get('exclude') is True:
                            entry['disposition'] = 'excluded_with_evidence'
                            accepted = True
                        elif isinstance(proposal.get('record'), dict):
                            row = proposal['record']
                            if row.get('entry_label') == entry['entry_label']:
                                entry['record'] = copy.deepcopy(row)
                                entry['disposition'] = 'extracted_unresolved'
                                accepted = True
                attempt = {'method': method, 'proposal': proposal, 'accepted': accepted}
                entry['attempts'].append(attempt)
                self.history.append({'experiment_id': entry['experiment_id'], 'before': before, **attempt})
                if checkpoint is not None:
                    checkpoint(self.export())
                if accepted:
                    break

    def export(self, unmatched=()):
        rows = []
        for entry in self.entries.values():
            if entry['disposition'] == 'excluded_with_evidence':
                continue
            row = copy.deepcopy(entry['record']) if entry['record'] is not None else {
                'reactants': [], 'products': [], 'conditions': [],
                'notes': ['Inventoried entry remains unresolved; no reaction was fabricated.'],
            }
            row.update(experiment_id=entry['experiment_id'], entry_label=entry['entry_label'],
                       source_reference=entry['source_reference'], disposition=entry['disposition'])
            if entry.get('recognized_candidates'):
                row['recognized_candidates'] = copy.deepcopy(entry['recognized_candidates'])
            rows.append(row)
        return {'reactions': rows, 'entry_ledger': list(self.entries.values()),
                'unassigned_candidates': list(unmatched), 'recovery_history': self.history,
                'completeness': {'inventoried': len(self.entries), 'accounted_for': len(self.entries),
                                 'complete_checked': 0,
                                 'inventory_disagreements': [e['experiment_id'] for e in self.entries.values()
                                     if len({o['method'] for o in e['observations']}) < 2]}}


def observations_from_records(value):
    """Only explicit reaction records; do not promote arbitrary OCR tokens to experiments."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    found = []
    if isinstance(value, dict):
        if 'products' in value and isinstance(value['products'], list):
            label = value.get('entry_label')
            labels = [label] if label else [c.get('label') for c in value['products'] if isinstance(c, dict)]
            for label in labels:
                if label:
                    found.append({'entry_label': label, 'source_reference': value.get('source_reference'),
                                  'outcome_label': value.get('outcome_label'),
                                  'evidence': value.get('field_evidence', {})})
        else:
            for child in value.values():
                found.extend(observations_from_records(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(observations_from_records(child))
    return found


def reaction_records(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    if isinstance(value, dict):
        if isinstance(value.get('products'), list):
            return [copy.deepcopy(value)]
        return [r for child in value.values() for r in reaction_records(child)]
    if isinstance(value, list):
        return [r for child in value for r in reaction_records(child)]
    return []

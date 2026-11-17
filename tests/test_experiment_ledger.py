import json
from experiment_ledger import Ledger, Checkpoints, compact


def test_missing_epoxidation_and_iodination_entries_survive_without_inventing_reactions():
    for labels in [('3ai', '3aj'), ('3i', '3j')]:
        ledger = Ledger('paper', 'scheme')
        ledger.inventory([{'entry_label': label, 'evidence': [{'bbox': [0, 0, 1, 1]}]}
                          for label in labels], 'image')
        ledger.attach_recognized({labels[1]: 'CCO'})
        ledger.assemble([{'entry_label': labels[0], 'products': [{'label': labels[0], 'smiles': 'CO'}]}])
        result = ledger.export()
        assert len(result['reactions']) == 2
        pending = result['reactions'][1]
        assert pending['entry_label'] == labels[1] and pending['products'] == []
        assert pending['recognized_candidates'][0]['smiles'] == 'CCO'
        assert pending['disposition'] == 'awaiting_review'
        assert result['completeness']['complete_checked'] == 0


def test_recovery_is_bounded_changes_method_and_retains_original():
    ledger = Ledger('paper', 'Table 1').inventory([{'entry_label': '1'}], 'parser')
    calls = []
    def repair(entry, method):
        calls.append(method)
        return {'experiment_id': entry['experiment_id'], 'record': {'entry_label': '1',
                'catalysts': [{'label': 'CPA', 'smiles': None}]}, 'reason': 'visible row',
                'evidence': [] if len(calls) == 1 else [{'bbox': [0, 0, 1, 1]}]}
    ledger.recover(repair)
    ledger.recover(repair)
    assert calls == ['region_reconstruction', 'definition_and_recognizer_review']
    assert ledger.export()['reactions'][0]['disposition'] == 'extracted_unresolved'
    assert ledger.history[0]['accepted'] is False
    assert ledger.history[1]['before']['record'] is None


def test_ambiguous_outcomes_checkpoint_integrity_and_prompt_compaction(tmp_path):
    ledger = Ledger('paper', 'table').inventory([
        {'entry_label': '1', 'outcome_label': '1h'}, {'entry_label': '1', 'outcome_label': '2h'}], 'parser')
    unmatched = ledger.assemble([{'entry_label': '1', 'products': []}])
    assert len(unmatched) == 1
    assert all(r['disposition'] == 'awaiting_review' for r in ledger.export()['reactions'])
    source = tmp_path / 'source.png'
    source.write_bytes(b'source')
    store = Checkpoints(tmp_path / 'runs', source, {'model': 'pinned'})
    raw = {'smiles': 'CCO', 'coords': [[1, 2]], 'edges': [[0, 1]]}
    store.save('recognition', raw)
    assert compact(raw) == {'smiles': 'CCO'}
    assert json.loads((store.root / 'recognition.json').read_text()) == raw

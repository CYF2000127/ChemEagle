import json
from experiment_ledger import Checkpoints, Ledger, compact, observations_from_records


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


def test_explicit_id_wins_across_tables_without_inventing_an_inventory_row():
    ledger = Ledger('paper', 'page-3').inventory([
        {'entry_label': '1', 'source_reference': 'Table 1'},
        {'entry_label': '1', 'source_reference': 'Table 2'},
    ], 'image')
    first, second = ledger.entries.values()
    row = {'experiment_id': first['experiment_id'], 'entry_label': '1', 'products': []}
    ledger.inventory(observations_from_records(row), 'synthesis')
    assert len(ledger.entries) == 2
    assert ledger.assemble([row]) == []
    assert first['record'] == row
    assert second['record'] is None


def test_unknown_id_or_conflicting_metadata_stays_unassigned():
    ledger = Ledger('paper', 'page-3').inventory([
        {'entry_label': '1', 'source_reference': 'Table 1'},
    ], 'image')
    entry, = ledger.entries.values()
    row = {'experiment_id': entry['experiment_id'], 'entry_label': '1', 'products': []}
    unresolved = [
        dict(row, experiment_id='exp-unknown'),
        dict(row, source_reference='Table 2'),
        dict(row, entry_label='2'),
    ]
    ledger.inventory(observations_from_records(unresolved), 'synthesis')
    assert len(ledger.entries) == 1
    assert ledger.assemble(unresolved) == unresolved
    assert entry['record'] is None


def test_model_only_exclusions_remain_visible_for_review():
    ledger = Ledger('paper', 'Table 1').inventory([{'entry_label': '7'}], 'image')

    def suggest_exclusion(entry, method):
        return {'experiment_id': entry['experiment_id'], 'exclude': True,
                'reason': 'duplicate', 'evidence': [{'bbox': [0, 0, 1, 1]}]}

    ledger.recover(suggest_exclusion, total_budget=1)
    result = ledger.export()
    row, = result['reactions']
    assert row['entry_label'] == '7'
    assert row['disposition'] == 'awaiting_review'
    attempt, = result['recovery_history']
    assert attempt['proposal']['exclude'] is True
    assert attempt['accepted'] is False

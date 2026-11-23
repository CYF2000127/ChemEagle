import json
import sys
from types import SimpleNamespace

import pytest
from experiment_ledger import Ledger, coalesce_metric_observations
from extraction_quality import audit_graphs, audit_record, parse_pdf_json, parse_specialist_json, usage_report
from pdf_evidence import extract_pdf, prepare_pdf


def test_unreported_conversion_and_unresolved_stereo_are_not_exported_as_exact():
    raw={'er':'86:14','ee':'72%','products':[{'smiles':'CCO','structure_status':'exact','stereochemistry_status':'unresolved'}]}
    row=audit_record(raw,'3i, 74% yield, 86:14 er')
    assert row['ee'] is None and row['derived_or_unverified']['ee']=='72%'
    assert row['products'][0]['smiles'] is None
    assert row['products'][0]['partial_smiles']=='CCO'
    assert raw['ee']=='72%'
    valid=audit_record({'ee':'72%','er':'86:14','field_evidence':{'ee':[{'quote':'72% ee'}]}},'72% ee; 86:14 er')
    assert valid['ee']=='72%'


def test_molecule_definitions_do_not_become_experiments_and_unknown_cost_is_not_zero():
    ledger=Ledger('paper','table').inventory([{'entry_label':'1'}],'parser')
    ledger.attach_recognized({'P2':'OP(=O)(O)O','1':'CCO'})
    output=ledger.export()
    assert len(output['reactions'])==1
    assert output['unassigned_structures'][0]['label']=='P2'
    usage=[{'usage':{'prompt_tokens':1000,'completion_tokens':100,'prompt_tokens_details':{'cached_tokens':200}},'seconds':2}]
    assert usage_report(usage)['estimated_usd'] is None
    assert usage_report(usage,{'input_per_million':2,'cached_input_per_million':1,'output_per_million':10})['estimated_usd']==pytest.approx(.0028)


def test_pdf_page_validation(tmp_path):
    source=tmp_path/'not-a-pdf.pdf';source.write_bytes(b'not a pdf')
    with pytest.raises(ValueError,match='PDF'):
        prepare_pdf(source,tmp_path/'out',[1])


def test_literal_pdf_controls_are_preserved_without_repairing_chemical_json():
    data, warnings = parse_pdf_json('{"quote":"at \x0260 C","smiles":"COc1ccco1"}')
    assert data['quote'] == 'at \x0260 C'
    assert data['smiles'] == 'COc1ccco1' and warnings
    with pytest.raises(json.JSONDecodeError):
        parse_pdf_json('{"reactions":[{"smiles":"CO"} }')


def test_specialist_prose_is_retained_and_ambiguous_answers_are_rejected():
    raw = 'The spans lack text. I will align them with the caption.{"reactions":[{"er":"86:14","smiles":"CCO"}]}'
    parsed = parse_specialist_json(raw)
    assert parsed['reactions'] == [{'er': '86:14', 'smiles': 'CCO'}]
    assert parsed['_response_parse']['raw'] == raw
    for invalid in ('First {"yield":74} then {"yield":72}',
                    'Result {"reactions":[{"smiles":"CCO"} }',
                    'Result {"yield":74} unfinished explanation',
                    'First [74] then {"yield":72}'):
        with pytest.raises(json.JSONDecodeError):
            parse_specialist_json(invalid)


def test_quality_audit_does_not_promote_pending_ledger_placeholders():
    row = audit_record({'entry_label': '3i', 'reactants': [], 'products': [],
                        'disposition': 'awaiting_review'}, '3i 74% yield')
    assert row['disposition'] == 'awaiting_review'
    assert any(issue['field'] == 'experiment' for issue in row['quality_issues'])
    assert audit_record({'disposition': 'complete_checked'}, '')['disposition'] == 'extracted_unresolved'


def test_inventory_merges_metrics_but_not_distinct_runs_or_source_regions():
    import copy
    row = {'entry_label': '8', 'source_reference': 'Table 1',
           'evidence': [{'bbox': [0.5, 0.3, 0.9, 0.31], 'quote': '8 73 95'}]}
    observations = [dict(copy.deepcopy(row), outcome_label='yield'), dict(copy.deepcopy(row), outcome_label='ee')]
    merged = coalesce_metric_observations(observations)
    assert len(merged) == 1 and merged[0]['outcome_label'] == 'reported'
    assert merged[0]['merged_outcome_labels'] == ['yield', 'ee']
    assert observations[0]['outcome_label'] == 'yield'
    # Two yields can represent repeated experiments; never silently collapse them.
    assert len(coalesce_metric_observations(observations + [observations[0]])) == 3
    other_region = copy.deepcopy(observations)
    other_region[1]['evidence'][0]['bbox'] = [0.5, 0.4, 0.9, 0.41]
    assert len(coalesce_metric_observations(other_region)) == 2
    explicit_runs = [dict(row, outcome_label='room temperature'), dict(row, outcome_label='-20 C')]
    assert coalesce_metric_observations(explicit_runs) == explicit_runs


def test_synthesis_without_outcome_metadata_reuses_one_known_row():
    ledger = Ledger('paper', 'page').inventory([
        {'entry_label': '13', 'source_reference': 'Table 1', 'outcome_label': 'no reaction'}], 'image')
    ledger.inventory([{'entry_label': '13', 'source_reference': 'Table 1'}], 'synthesis')
    assert len(ledger.entries) == 1
    ledger.assemble([{'entry_label': '13', 'source_reference': 'Table 1', 'products': [], 'yield': 'N.R.'}])
    assert next(iter(ledger.entries.values()))['record']['yield'] == 'N.R.'
    ledger.inventory([{'entry_label': '13', 'source_reference': 'Table 1', 'outcome_label': '50 C'}], 'image')
    assert len(ledger.entries) == 2


def test_selected_page_entry_point_enables_preservation(monkeypatch, tmp_path):
    packet = {'image': 'page.png', 'page_number': 1, 'paper_sha256': 'paper'}
    monkeypatch.setattr('pdf_evidence.prepare_pdf', lambda *args: [packet])
    received = {}

    def capture(image, **options):
        received.update(options)
        return {'reactions': []}

    monkeypatch.setitem(sys.modules, 'main', SimpleNamespace(ChemEagle=capture))
    result = extract_pdf('article.pdf', tmp_path, [1])
    assert received['preserve_entries'] is True
    assert received['source_context'] == packet
    assert result['pages'][0]['status'] == 'completed'


def test_audit_checks_exported_outcome_values_against_the_cited_text():
    raw = {
        'reactants': [{'smiles': 'CCO'}],
        'additional_info': [
            {'role': 'yield', 'text': '88%'},
            {'role': 'er', 'text': '86:14'},
            {'role': 'ee', 'text': '72%'},
        ],
        'conditions': [{'role': 'temperature', 'text': '25 °C'}],
        'field_evidence': {
            'yield': [{'quote': '12% yield'}],
            'er': [{'quote': '86 : 14 er'}],
            'ee': [{'quote': '66% ee'}],
            'temperature': [{'quote': '25 °C'}],
        },
    }
    row = audit_record(raw, '12% yield at 25 °C, 86 : 14 er, 66% ee')
    assert {issue['field'] for issue in row['quality_issues']} == {'yield', 'ee'}
    assert row['additional_info'][2]['text'] is None
    assert row['derived_or_unverified']['ee'] == '72%'
    assert raw['additional_info'][2]['text'] == '72%'


def test_graph_audit_quarantines_invalid_catalyst_smiles_in_conditions():
    raw = {'conditions': [{'role': 'catalyst', 'label': 'C1', 'smiles': 'C('},
                          {'role': 'solvent', 'text': 'toluene'}]}
    row = audit_graphs(raw)
    catalyst, solvent = row['conditions']
    assert catalyst['smiles'] is None
    assert catalyst['rejected_graph_candidates']['smiles'] == 'C('
    assert catalyst['chemical_identity_verified'] is False
    assert solvent == raw['conditions'][1]
    assert raw['conditions'][0]['smiles'] == 'C('


def test_audit_preserves_explicit_not_determined_selectivity():
    row = audit_record({
        'additional_info': [{'role': 'er', 'text': '86:14'}, {'role': 'ee', 'text': 'ND'}],
    }, '')
    assert row['additional_info'][1]['text'] == 'ND'
    assert 'derived_or_unverified' not in row

import json
import pytest
from experiment_ledger import Ledger, coalesce_metric_observations
from extraction_quality import audit_record, usage_report, parse_pdf_json, parse_specialist_json
from pdf_evidence import prepare_pdf


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

import json
import pytest
from experiment_ledger import Ledger
from extraction_quality import audit_record, usage_report, parse_pdf_json
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

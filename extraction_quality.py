"""Mechanical evidence gates and usage accounting; never a chemistry correctness claim."""
import copy
import json
import re
import unicodedata


EVIDENCE_FIELDS = (
    'yield', 'ee', 'er', 'dr', 'de', 'conversion', 'selectivity',
    'temperature', 'time', 'solvent', 'catalyst_loading', 'catalyst',
)


def _normalize_text(text):
    normalized = unicodedata.normalize('NFKC', str(text)).casefold().replace('−', '-')
    normalized = re.sub(r'\s*:\s*', ':', ' '.join(normalized.split()))
    return re.sub(r'\s+%', '%', normalized)


def _reported_fields(row):
    for field in EVIDENCE_FIELDS:
        if row.get(field) is not None:
            yield field, row[field], row
    for section in ('additional_info', 'conditions'):
        for item in row.get(section) or []:
            if not isinstance(item, dict) or item.get('role') not in EVIDENCE_FIELDS:
                continue
            value = item.get('value')
            if value is None:
                value = item.get('text')
            if value is not None:
                yield item['role'], value, item


def _compound_graphs(row):
    for role in ('reactants', 'products', 'catalyst_structures', 'conditions'):
        for compound in row.get(role) or []:
            if not isinstance(compound, dict):
                continue
            if role == 'conditions' and not (compound.get('smiles') or compound.get('partial_smiles')):
                continue
            yield role, compound


def audit_record(record, source_text):
    row = copy.deepcopy(record)
    issues = []
    if not row.get('reactants') and not row.get('products'):
        issues.append({'field': 'experiment', 'reason': 'No assembled reactant or product record; review required'})
    evidence = row.get('field_evidence')
    evidence = evidence if isinstance(evidence, dict) else {}
    source = _normalize_text(source_text)
    fields = list(_reported_fields(row))
    has_er = any(field == 'er' for field, _, _ in fields)
    for field, value, item in fields:
        citations = evidence.get(field)
        value_text = _normalize_text(value)
        pattern = r'(?<![\w.+-])' + re.escape(value_text) + r'(?!\w|\.\d)'
        verified_quotes = []
        for citation in citations if isinstance(citations, list) else []:
            if not isinstance(citation, dict) or not citation.get('quote'):
                continue
            quote = _normalize_text(citation['quote'])
            if quote and value_text and quote in source and re.search(pattern, quote):
                verified_quotes.append(quote)
        if not verified_quotes:
            issues.append({'field': field, 'reason': 'No verifiable source-text citation; image review required'})
        numeric_ee = field == 'ee' and re.match(r'^[<>≤≥~]?\s*[+-]?\s*\d', value_text)
        if numeric_ee and has_er and not any(
            re.search(r'\be\.?\s*e\.?\b|enantiomeric excess', quote) for quote in verified_quotes
        ):
            row.setdefault('derived_or_unverified', {})['ee'] = value
            if item is row:
                item['ee'] = None
            else:
                for key in ('value', 'text'):
                    if key in item:
                        item[key] = None
            issues.append({'field': 'ee', 'reason': 'Unreported er-to-ee conversion withheld pending source review'})
    for role, compound in _compound_graphs(row):
        smiles = compound.get('smiles')
        if not smiles or '*' in str(smiles):
            issues.append({'field': role, 'label': compound.get('label'), 'reason': 'Exact graph unresolved'})
        if compound.get('stereochemistry_status') == 'unresolved' and compound.get('structure_status') == 'exact':
            compound['structure_status'] = 'partial'
            compound['partial_smiles'] = smiles
            compound['smiles'] = None
            issues.append({'field': role, 'reason': 'Unresolved stereochemistry is not an exact graph'})
    row['quality_issues'] = issues
    # The ledger's pending rows must remain pending after mechanical auditing.
    # A model-provided "complete_checked" claim still cannot pass this gate.
    if row.get('disposition') != 'awaiting_review':
        row['disposition'] = 'extracted_unresolved'
    return row


def usage_report(responses, rates=None):
    totals = {'input_tokens': 0, 'cached_input_tokens': 0, 'output_tokens': 0, 'seconds': 0,
              'calls': len(responses), 'calls_missing_usage': 0}
    for response in responses:
        usage = response.get('usage')
        totals['seconds'] += response.get('seconds', 0)
        if not usage:
            totals['calls_missing_usage'] += 1
            continue
        totals['input_tokens'] += usage.get('prompt_tokens', 0)
        totals['output_tokens'] += usage.get('completion_tokens', 0)
        totals['cached_input_tokens'] += (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
    totals['estimated_usd'] = None
    if rates and not totals['calls_missing_usage']:
        totals['estimated_usd'] = ((totals['input_tokens']-totals['cached_input_tokens']) * rates['input_per_million']
                                  + totals['cached_input_tokens'] * rates['cached_input_per_million']
                                  + totals['output_tokens'] * rates['output_per_million']) / 1e6
    totals['billed_usd'] = None
    totals['cost_status'] = 'Estimate only' if rates else 'Unpriced: proxy tariff/billing unavailable'
    return totals


def parse_pdf_json(raw):
    """Accept literal PDF control characters inside JSON strings, never repair structure or values."""
    try:
        return json.loads(raw), []
    except json.JSONDecodeError as exc:
        if not exc.msg.startswith('Invalid control character'):
            raise
        return json.loads(raw, strict=False), ['Literal PDF control characters preserved from model strings']


def parse_specialist_json(raw):
    """Read one complete JSON object after prose; retain the unmodified response.

    Never merge objects, fill missing brackets, or change chemical values. The
    full remaining suffix must parse, so multiple answers stay unresolved.
    """
    try:
        value, warnings = parse_pdf_json(raw)
    except json.JSONDecodeError:
        start = raw.find('{')
        if start <= 0 or any(c in raw[:start] for c in '[]}'):
            raise
        value, warnings = parse_pdf_json(raw[start:])
        warnings = ['Prose before a single complete JSON object retained in raw response'] + warnings
    if warnings and isinstance(value, dict):
        value['_response_parse'] = {'warnings': warnings, 'raw': raw}
    return value


def audit_graphs(record):
    """Reject invalid graphs while preserving originals; RDKit syntax is not identity/stereo proof."""
    row = copy.deepcopy(record)
    try:
        from rdkit import Chem
    except ImportError:
        row['graph_validation'] = 'unavailable: RDKit is not installed'
        return row
    errors = []
    for role, compound in _compound_graphs(row):
        for field in ('smiles', 'partial_smiles'):
            value = compound.get(field)
            if not value or (isinstance(value, str) and value.strip().casefold() in {'', 'none', 'null', 'n/a'}):
                continue
            mol = Chem.MolFromSmiles(value) if isinstance(value, str) else None
            if mol is None:
                compound.setdefault('rejected_graph_candidates', {})[field] = value
                compound[field] = None
                compound['structure_status'] = 'unresolved'
                errors.append({'role': role, 'label': compound.get('label'), 'reason': 'Invalid SMILES'})
            elif field == 'smiles' and any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
                compound['partial_smiles'] = value
                compound['smiles'] = None
                compound['structure_status'] = 'partial'
        compound['chemical_identity_verified'] = False
    row['graph_validation'] = {'method': 'RDKit syntax and wildcard checks only', 'errors': errors}
    return row

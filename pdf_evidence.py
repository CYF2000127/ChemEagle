"""Content-addressed PDF evidence packets for hybrid and API extraction backends."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

VERSION = 'pdf-evidence-v1'


def _run(args):
    return subprocess.run(args, check=True, capture_output=True, timeout=120).stdout


def prepare_pdf(pdf, directory, pages, reference_pages=(), supporting_information=()):
    pdf = Path(pdf).resolve()
    if not pdf.is_file() or pdf.read_bytes()[:5] != b'%PDF-':
        raise ValueError('Expected a readable PDF file')
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    info = _run(['pdfinfo', str(pdf)]).decode(errors='replace')
    count = int(re.search(r'^Pages:\s+(\d+)', info, re.M)[1])
    pages, references = list(dict.fromkeys(pages)), list(dict.fromkeys(reference_pages))
    if len(references) > 2:
        raise ValueError('Select at most two reference pages per packet')
    if not pages or any(type(p) is not int or not 1 <= p <= count for p in pages + references):
        raise ValueError('Page numbers must be inside the PDF, starting at 1')
    settings = {'version': VERSION, 'pdf_sha256': sha, 'pages': pages, 'references': references,
                'dpi': 180, 'parser': subprocess.run(['pdftotext', '-v'], capture_output=True, timeout=10).stderr.decode(errors='replace')}
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    root = Path(directory) / key
    root.mkdir(parents=True, exist_ok=True)
    evidence = {}
    for page in sorted(set(pages + references)):
        image, text = root / f'page-{page}.png', root / f'page-{page}.txt'
        if not image.exists():
            _run(['pdftoppm', '-f', str(page), '-l', str(page), '-r', '180', '-singlefile', '-png',
                  str(pdf), str(image.with_suffix(''))])
        if not text.exists():
            _run(['pdftotext', '-layout', '-f', str(page), '-l', str(page), str(pdf), str(text)])
        evidence[page] = {'page_number': page, 'image': str(image.resolve()), 'text': text.read_text(),
                          'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest()}
    si = []
    for item in supporting_information:
        # SI is explicitly supplied and hashed, never silently fetched or mixed with another paper.
        path = Path(item).resolve()
        text = _run(['pdftotext', '-layout', str(path), '-']).decode(errors='replace')
        si.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'text': text})
    packets = [{**evidence[page], 'paper_sha256': sha, 'pdf': str(pdf), 'region_id': f'page-{page}',
                'reference_pages': [evidence[p] for p in references if p != page], 'supporting_information': si,
                'scan_needs_ocr': len(evidence[page]['text'].strip()) < 40} for page in pages]
    (root / 'manifest.json').write_text(json.dumps({**settings, 'packets': packets}, indent=2))
    return packets


def extract_pdf(pdf, output_dir, pages, reference_pages=(), supporting_information=(), backend=None, **options):
    """Run selected PDF pages through the real hybrid backend, preserving a failure per page."""
    if backend is None:
        from main import ChemEagle
        backend = ChemEagle
        options.setdefault('preserve_entries', True)
    packets = prepare_pdf(pdf, output_dir, pages, reference_pages, supporting_information)
    results = []
    for packet in packets:
        try:
            value = backend(packet['image'], source_context=packet,
                            checkpoint_dir=str(Path(output_dir) / 'extraction'), **options)
            results.append({'page': packet['page_number'], 'status': 'completed', 'result': value})
        except Exception as exc:
            results.append({'page': packet['page_number'], 'status': 'failed', 'error_type': type(exc).__name__})
        Path(output_dir, packets[0]['paper_sha256'] + '-document-progress.json').write_text(json.dumps(results, indent=2))
    return {'paper_sha256': packets[0]['paper_sha256'], 'pages': results,
            'complete_checked': False, 'selected_pages_only': True}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdf')
    parser.add_argument('--output', required=True)
    parser.add_argument('--pages', nargs='+', type=int, required=True)
    parser.add_argument('--references', nargs='*', type=int, default=[])
    parser.add_argument('--si', nargs='*', default=[])
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--model')
    parser.add_argument('--recovery-budget', type=int, default=4)
    args = parser.parse_args()
    if args.prepare_only:
        packets = prepare_pdf(args.pdf, args.output, args.pages, args.references, args.si)
        print(json.dumps({'prepared_pages': [p['page_number'] for p in packets],
                          'scan_pages': [p['page_number'] for p in packets if p['scan_needs_ocr']]}))
    else:
        result = extract_pdf(args.pdf, args.output, args.pages, args.references, args.si,
                             model_name=args.model, recovery_budget=args.recovery_budget)
        Path(args.output, result['paper_sha256'] + '-result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({'completed_pages': sum(p['status']=='completed' for p in result['pages']),
                          'failed_pages': sum(p['status']=='failed' for p in result['pages'])}))

"""Freeze source families and rebuild exports only from hash-verified completed checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(paths, output):
    baseline = subprocess.check_output(['git', 'rev-parse', '479e812^{commit}'], text=True).strip()
    current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    # Caller supplies article + SI together; no random page-level split that leaks SI into test.
    families = json.loads(Path(paths).read_text())
    seen = set()
    for family in families:
        if family['split'] not in ('development', 'test'):
            raise ValueError('Specify development/test at family level')
        for source in family['sources']:
            source['sha256'] = sha(source['path'])
            if source['sha256'] in seen:
                raise ValueError('Duplicate source across families')
            seen.add(source['sha256'])
    Path(output).write_text(json.dumps({'upstream': baseline, 'fork': current, 'families': families,
        'settings': 'Must be populated per measured run; no benchmark has been executed.'}, indent=2))


def reconcile(root, output):
    records, audit = [], []
    for receipt in sorted(Path(root).rglob('completed.json')):
        try:
            manifest = json.loads(receipt.read_text())
            final = receipt.parent / 'final.json'
            if manifest['status'] != 'complete' or sha(final) != manifest['final']['sha256']:
                raise ValueError('Checkpoint hash/status mismatch')
            result = json.loads(final.read_text())
            if result['provenance']['source_sha256'] != manifest['source_sha256']:
                raise ValueError('Source mismatch')
            records.append({'checkpoint': str(final), 'sha256': sha(final), 'result': result})
            audit.append({'checkpoint': str(receipt), 'status': 'included'})
        except (KeyError, ValueError, OSError) as exc:
            audit.append({'checkpoint': str(receipt), 'status': 'unverified', 'reason': type(exc).__name__})
    # Keep distinct configuration runs distinct. Never silently select one as ground truth.
    Path(output).write_text(json.dumps({'runs': records, 'audit': audit}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'reconcile'])
    parser.add_argument('source')
    parser.add_argument('output')
    args = parser.parse_args()
    {'freeze': freeze, 'reconcile': reconcile}[args.action](args.source, args.output)

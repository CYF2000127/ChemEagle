"""Exercise the upstream public wrapper without loading GPU model dependencies."""
import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest


@pytest.mark.parametrize('confidence_enabled', [False, True])
def test_public_wrapper_forwards_preservation_options(confidence_enabled):
    tree = ast.parse((Path(__file__).parents[1] / 'main.py').read_text())
    public = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'ChemEagle')
    internal = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_run_chemeagle')
    # Retain the real internal signature; expose received arguments instead of
    # invoking external chemistry models. Signature mismatches still raise.
    internal.body = ast.parse('return {"received": locals()}').body
    module = ast.fix_missing_locations(ast.Module(body=[public, internal], type_ignores=[]))

    @contextmanager
    def capture_log():
        yield []

    namespace = {
        'Optional': Optional,
        'confidence': SimpleNamespace(
            env_switch=lambda value, name, default: default if value is None else value,
            capture_log=capture_log,
            attach_confidence=lambda output, *args, **kwargs: output),
        'llm': SimpleNamespace(resolve_model=lambda m: m, resolve_base_url=lambda u: u,
                               resolve_key=lambda k: k),
    }
    exec(compile(module, 'main.py', 'exec'), namespace)
    context = {'paper_sha256': 'paper', 'region_id': 'page-3'}
    inventory = [{'entry_label': '8'}]
    output = namespace['ChemEagle'](
        'page.png', model_name='test', use_confidence=confidence_enabled,
        confidence_checker='test', preserve_entries=False, entry_inventory=inventory,
        source_context=context, checkpoint_dir='custom/checkpoints', recovery_budget=0)
    received = output['received']
    assert received['preserve_entries'] is False
    assert received['entry_inventory'] is inventory
    assert received['source_context'] is context
    assert received['checkpoint_dir'] == 'custom/checkpoints'
    assert received['recovery_budget'] == 0
